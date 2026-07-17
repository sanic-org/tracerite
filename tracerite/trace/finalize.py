from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

from tracerite.logging import logger

from .collect import collect_exception_chain
from .core import libdir_match
from .digest import (
    digest_exception_chain,
    is_exception_group,
)
from .order import (
    apply_base_exception_suppression,
    build_chronological_frames,
    filter_hidden_frames,
)


def exception_info(exc: dict[str, Any]) -> dict[str, Any]:
    """Return a JSON-compatible exception-info dict."""
    return {
        "type": exc.get("type"),
        "message": exc.get("message"),
        "summary": exc.get("summary"),
        "from": exc.get("from"),
    }


def function_display(function: str | None, function_suffix: str) -> str | None:
    """Return the display string for a function name with an optional suffix."""
    if function:
        return f"{function}{function_suffix}"
    return function_suffix or None


def normalize_variable(var_info: Any) -> tuple[str, str, Any, str]:
    """Normalize a VarInfo dict or old tuple into (name, typename, value, fmt)."""
    if isinstance(var_info, dict):
        return (
            var_info["name"],
            var_info["typename"],
            var_info["value"],
            var_info["format_hint"],
        )
    name, typename, value = var_info
    return name, typename, value, "inline"


def call_run_ranges(
    frames: list[dict[str, Any]], min_run_length: int = 10
) -> list[tuple[int, int]]:
    """Return (start, end) ranges of consecutive 'call' frames to collapse."""
    ranges = []
    run_start = None
    for i, frinfo in enumerate(frames):
        if frinfo["relevance"] == "call":
            if run_start is None:
                run_start = i
        else:
            if run_start is not None:
                run_length = i - run_start
                if run_length >= min_run_length and run_length > 2:
                    ranges.append((run_start, i - 1))
                run_start = None
    if run_start is not None:
        run_length = len(frames) - run_start
        if run_length >= min_run_length and run_length > 2:
            ranges.append((run_start, len(frames) - 1))
    return ranges


def collect_leaf_exception_types(subexceptions: list[list[dict]]) -> list[str]:
    """Recursively collect leaf exception type names from subexception chains."""
    return [
        leaf
        for sub in subexceptions
        for leaf in (
            collect_leaf_exception_types(sub[-1]["subexceptions"])
            if sub and sub[-1].get("subexceptions")
            else [sub[-1].get("type", "Exception")]
            if sub
            else []
        )
    ]


def attach_leaf_types(exc_chain: list[dict], chrono_frames: list[dict]) -> None:
    """Attach ExceptionGroup leaf types to the final exception banner in frames."""
    if not exc_chain:
        return
    subexceptions = exc_chain[-1].get("subexceptions")
    if not subexceptions:
        return
    leaf_types = collect_leaf_exception_types(subexceptions)
    if not leaf_types:
        return
    for frame in reversed(chrono_frames):
        if frame.get("exception"):
            frame["exception"]["leaf_types"] = leaf_types
            break


def build_chain_header(frames: list[dict]) -> str:
    """Build a header message from a chronological frame list."""
    if not frames:
        return ""

    main_chain = [f["exception"] for f in frames if f.get("exception")]
    if not main_chain:
        return ""

    last_exc = main_chain[-1]
    leaf_types = last_exc.get("leaf_types", [])
    if leaf_types:
        exc_type = " | ".join(leaf_types)
        if len(main_chain) == 1:
            return f"⚠️  {exc_type}"
    else:
        exc_type = last_exc.get("type", "Exception")

    if len(main_chain) == 1:
        return f"⚠️  Uncaught {exc_type}"

    parts = [f"⚠️  {exc_type}"]
    for i in range(len(main_chain) - 2, -1, -1):
        exc = main_chain[i]
        next_exc = main_chain[i + 1]
        from_type = next_exc.get("from", "none")
        joiner = "from" if from_type == "cause" else "while handling"
        parts.append(f"{joiner} {exc.get('type', 'Exception')}")

    return " ".join(parts)


# =============================================================================
# Pipeline entry points
# =============================================================================


def extract_chain(exc=None, **kwargs) -> list:
    """Extract chronological traceback frames for the current exception."""
    exc = exc or sys.exc_info()[1]
    source_cache: dict = {}
    chain = collect_exception_chain(exc, **kwargs)
    chain = digest_exception_chain(chain, cache=source_cache)
    set_chain_relevances(chain)
    chronological = build_chronological_frames(chain, cache=source_cache)
    chronological = finalize_chronological(chronological, chain)
    return chronological


def extract_chain_exceptions(exc=None, **kwargs) -> list:
    """Extract raw exception info dicts, oldest first (internal)."""
    chain = digest_exception_chain(collect_exception_chain(exc, **kwargs))
    set_chain_relevances(chain)
    return chain


def set_chain_relevances(chain: list[dict]) -> None:
    """Set error/stop/warning relevances on each exception's raw frames."""
    for exc in chain:
        e = exc.pop("_exc")
        set_relevances(exc.get("frames", []), e)
        for sub_chain in exc.get("subexceptions") or []:
            set_chain_relevances(sub_chain)


def finalize_chronological(chronological: list[dict], chain: list[dict]) -> list[dict]:
    """Apply all final-stage passes to the chronological frame list."""
    chronological = filter_hidden_frames(chronological)
    chronological = apply_base_exception_suppression(chronological, chain)
    attach_leaf_types(chain, chronological)
    fill_chronological_variables(chronological)
    return chronological


def set_relevances(frames: list, e: BaseException) -> None:
    """Mark the error, stop, warning, and call frames."""
    if not frames:
        return

    # Last frame is where the exception occurred
    # ExceptionGroups get "stop" like BaseExceptions - the real errors are in subexceptions
    is_regular_exception = isinstance(e, Exception) and not is_exception_group(e)
    frames[-1]["relevance"] = "error" if is_regular_exception else "stop"

    # Check if the last frame (error frame) is in user code
    last_filename = (
        frames[-1].get("original_filename") or frames[-1].get("filename") or ""
    )
    if libdir_match(Path(last_filename).as_posix()) is None:
        return
    # Error is in library code - find the last user code frame to mark as warning
    for frame in reversed(frames[:-1]):  # Exclude the last frame  # pragma: no cover
        filename = frame.get("original_filename") or frame.get("filename") or ""
        if libdir_match(Path(filename).as_posix()) is None:
            # This is user code - mark as warning (bug origin)
            frame["relevance"] = "warning"
            break


def fill_frame_variables(frame: dict, exc_message: str | None = None) -> None:
    """Extract variables for a single frame and drop its live frame object."""
    from tracerite.inspector import extract_variables

    frame_obj = frame.pop("frame_obj", None)
    variable_source = frame.pop("variable_source", None)

    if frame.get("hidden") or frame_obj is None or variable_source is None:
        frame.setdefault("variables", [])
        return

    try:
        frame["variables"] = extract_variables(
            frame_obj.f_locals,
            variable_source,
            exc_message=exc_message,
        )
    except Exception:
        logger.exception("Error extracting variables")
        frame["variables"] = []


def fill_variables(frames: list[dict], exc_message: str | None = None) -> None:
    """Populate the variables field for a Python-order frame list."""
    last_idx = len(frames) - 1
    for idx, frame in enumerate(frames):
        fill_frame_variables(
            frame, exc_message=exc_message if idx == last_idx else None
        )


def fill_chronological_variables(chrono_frames: list[dict]) -> None:
    """Fill variables on the final occurrence of each frame in chrono order."""
    seen: set[int] = set()

    def _process_branch(branch: list[dict]) -> None:
        for frame in reversed(branch):
            idframe = frame.get("idframe")
            if idframe is not None and idframe not in seen and "frame_obj" in frame:
                exc_message = frame.get("exception", {}).get("message")
                fill_frame_variables(frame, exc_message=exc_message)
                seen.add(idframe)
            else:
                frame.setdefault("variables", [])
                frame.pop("frame_obj", None)
                frame.pop("variable_source", None)
            for sub_branch in frame.get("parallel", []):
                _process_branch(sub_branch)

    _process_branch(chrono_frames)

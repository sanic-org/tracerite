from __future__ import annotations

import os
import re
import sys
import unicodedata
import warnings
from pathlib import Path
from typing import TYPE_CHECKING, Any, TextIO

from .trace.core import EMPHASIS_BEG, EMPHASIS_FIN, chainmsg, symbols, symdesc

if TYPE_CHECKING:
    from typing import TypedDict

    from .trace.typing import (
        Chain,
        ExceptionInfo,
        Fragment,
        FragmentLine,
        FrameInfo,
        Range,
    )

    class _ChronoFrameInfo(TypedDict):
        """Precomputed display info for one chronological frame."""

        location_part: str
        function_part: str
        fragments: list[FragmentLine]
        frame_range: Range | None
        relevance: str
        exc_info: ExceptionInfo | None
        marked_lines: list[FragmentLine]
        frinfo: FrameInfo


from .trace.finalize import (
    call_run_ranges,
    extract_chain,
    function_display,
    normalize_variable,
)

__all__ = ["load", "unload", "tty_traceback"]

# Variable inspector line: (colored_line, plain_width, value_start_col)
InspectorLine = tuple[str, int, int]
InspectorLines = list[InspectorLine]


def load(capture_logging: bool = True) -> None:
    """Load TraceRite as the default exception handler.

    .. deprecated::
        Use :func:`tracerite.load` instead.
    """
    warnings.warn(
        "tracerite.tty.load is deprecated; use tracerite.load instead",
        DeprecationWarning,
        stacklevel=2,
    )
    from .hooks import load

    return load(capture_logging=capture_logging)


def unload() -> None:
    """Restore the original exception handlers.

    .. deprecated::
        Use :func:`tracerite.unload` instead.
    """
    warnings.warn(
        "tracerite.tty.unload is deprecated; use tracerite.unload instead",
        DeprecationWarning,
        stacklevel=2,
    )
    from .hooks import unload

    return unload()


# ANSI escape codes for terminal colors (can be monkeypatched for styling)
ESC = "\x1b["
RESET = f"{ESC}0m"
DIM = f"{ESC}2m"
LINE_PREFIX_TOP = f"{DIM}╭{RESET} "  # Dim rounded top-left corner for first line
LINE_PREFIX = f"{DIM}│{RESET} "  # Dim vertical line prefix for middle lines
LINE_PREFIX_BOT = f"{DIM}╰{RESET} "  # Dim rounded bottom-left corner for last line
EOL = f"\n{LINE_PREFIX}"  # End of line: newline, add prefix
MARK_BG = f"{ESC}48;5;220m"  # Yellow background (xterm256 #ffd700, fixed shade)
MARK_TEXT = f"{ESC}22;38;5;16m"  # Black text (xterm256 #000000), no bold
EM = f"{ESC}1;38;5;196m"  # Bold, red (xterm256 #ff0000)
LOCFN = f"{ESC}32m"
EM_CALL = f"{ESC}93m"  # Bright yellow
EXC = f"{ESC}90m"  # Dark grey for exception text
ELLIPSIS = f"{ESC}90m"  # Dark grey for ellipsis/skipped calls
LOC_LINENO = f"{ESC}90m"  # Dark grey for :lineno
TYPE_COLOR = f"{ESC}32m"  # Green for type in inspector (matches HTML --tracerite-type)
NO_SOURCE = f"{ESC}2m"  # Dim for 'source code not available' message
SYMBOLDESC = f"{ESC}1m"  # Bright white for symbol desc (e.g. Call from your code)
FUNC = f"{ESC}38;5;153m"  # Light blue (xterm256 LightSkyBlue1) for function names
VAR = f"{ESC}38;5;153m"  # Light blue (xterm256 LightSkyBlue1) for variable names (same as FUNC)
BOLD = f"{ESC}1m"

# Box drawing characters
BOX_H = "─"
BOX_V = "│"
BOX_VL = "┤"  # Vertical with left branch
BOX_TL = "╭"  # Rounded top-left
BOX_BL = "╰"  # Rounded bottom-left
BOX_TR = "╮"  # Rounded top-right
BOX_BR = "╯"  # Rounded bottom-right
ARROW_LEFT = "◀"
SINGLE_T = "❴"  # T-junction for single line
INDENT = ""  # No indent for function/location lines
CODE_INDENT = "  "  # Indent for code in frame

# Regex pattern to strip ANSI escape sequences
ANSI_ESCAPE_RE = re.compile(r"\x1b\[[0-9;:]*[A-Za-z]")

# FastAPI rich log format prefix  " ▕" (plain space + possibly colored bar).
RICHPREFIX_RE = re.compile(rf"^ (?:{ANSI_ESCAPE_RE.pattern})*▕")

# Token regex: either a full ANSI escape sequence or any single character.
_TOKEN_RE = re.compile(r"\x1b\[[0-9;:]*[A-Za-z]|.", re.DOTALL)


def _display_width(s: str) -> int:
    """Calculate the display width of a string in terminal columns."""
    plain = ANSI_ESCAPE_RE.sub("", s)
    return sum(
        2
        if unicodedata.east_asian_width(c) in "WF"
        else 0
        if unicodedata.category(c) in ("Mn", "Me", "Cf")
        else 1
        for c in plain
    )


_LINE_PREFIX_WIDTH = _display_width(LINE_PREFIX)


def tty_traceback(
    exc: BaseException | None = None,
    chain: Chain | None = None,
    *,
    file: TextIO | None = None,
    msg: str | None = None,
    tag: str = "",
    term_width: int | None = None,
    **extract_args: Any,
) -> None:
    """Format and print a traceback for terminal output (TTY).

    Outputs directly to the terminal (or specified file) to adapt to
    terminal features like window size. The chain is printed with the
    oldest exception first (order they occurred).

    Args:
        exc: The exception to format. If None, uses the current exception.
        chain: Pre-extracted data from `extract_chain` (dict with `header` and `frames`).
            If provided, exc is ignored.
        file: Output file. Defaults to sys.stderr.
        msg: Header message. If None, builds from exception chain.
        tag: Optional tag to display after the message (e.g., "#TR1").
        term_width: Terminal width. Auto-detected if None.
        **extract_args: Additional arguments passed to extract_chain().
    """
    if chain is None:
        chain = extract_chain(exc=exc, **extract_args)
    frames = chain["frames"]

    # Build header message if not provided
    if msg is None:
        msg = chain["header"] or None

    if file is None:
        file = sys.stderr

    is_tty = hasattr(file, "isatty") and file.isatty()
    no_color = "NO_COLOR" in os.environ or not (is_tty or "FORCE_COLOR" in os.environ)
    no_inspector = no_color or not is_tty  # journalctl doesn't like cursor positioning

    # Start with rounded top corner
    output = LINE_PREFIX_TOP

    # Print the original log message if provided, with optional tag at the end
    if msg:
        # Strip trailing newlines and left-trim two spaces if present (to align with prefix)
        msg = msg.rstrip("\n")
        if msg.startswith("  "):
            msg = msg[2:]
        elif RICHPREFIX_RE.match(msg):
            msg = msg[1:]
            output = output.removesuffix(" ")

        # Append tag (dim color) if provided
        if tag:
            msg = f"{msg} {DIM}{tag}{RESET}"

        output += msg + EOL

    if term_width is None:
        try:
            term_width = os.get_terminal_size(file.fileno()).columns
        except (OSError, ValueError):
            term_width = 80

        # Very narrow terminals are assumed temporary; honour 40+ deliberately.
        if term_width < 40:
            term_width = 80

    chrono_output, last_banner_start = _print_chronological(
        frames, term_width, no_inspector
    )
    if last_banner_start is not None:
        last_banner_start += len(output)
    output += chrono_output

    if not frames:
        # No traceback: add a dim note where the exception banner would appear.
        output += NO_SOURCE + "(no traceback)" + EOL

    # Strip trailing empty border.
    output = output.removesuffix(f"\n{LINE_PREFIX}")

    # Curve the bottom border; for banners, continuation lines hang loose.
    if last_banner_start is not None:
        prefix_pos = last_banner_start - len(LINE_PREFIX)
        output = output[:prefix_pos] + LINE_PREFIX_BOT + output[last_banner_start:]
        first_line_end = output.find("\n", last_banner_start)
        if first_line_end != -1:
            output = output[:first_line_end] + output[first_line_end:].replace(
                "\n" + LINE_PREFIX, "\n  "
            )
    else:
        last_prefix_pos = output.rfind(LINE_PREFIX)
        if last_prefix_pos != -1:
            output = (
                output[:last_prefix_pos]
                + LINE_PREFIX_BOT
                + output[last_prefix_pos + len(LINE_PREFIX) :]
            )

    output += "\n" + RESET

    if no_color:
        # Strip all ANSI escape sequences for non-TTY output
        output = ANSI_ESCAPE_RE.sub("", output)

    file.write(output)


def _find_all_inspector_frames(
    frame_info_list: list[_ChronoFrameInfo],
) -> list[int]:
    """Find all non-call frames that have variables to show."""
    return [
        i
        for i, info in enumerate(frame_info_list)
        if info["relevance"] != "call" and info["frinfo"].get("variables")
    ]


def _find_frame_line_range(
    output_lines: list[tuple[str, int, int, bool]], inspector_frame_idx: int
) -> tuple[int, int]:
    """Find the line range for the inspector frame in output_lines."""
    indices = [
        li
        for li, (_, _, fidx, _) in enumerate(output_lines)
        if fidx == inspector_frame_idx
    ]
    # By contract, the frame must exist in output_lines
    assert indices
    return indices[0], indices[-1]


def _find_last_marked_line(
    output_lines: list[tuple[str, int, int, bool]],
    frame_line_start: int,
    frame_line_end: int,
) -> int:
    """Find the last marked line within the frame range."""
    return max(
        (
            li
            for li in range(frame_line_start, frame_line_end + 1)
            if output_lines[li][3]
        ),
        default=frame_line_end,
    )


def _find_collapsible_call_runs(
    frame_info_list: list[_ChronoFrameInfo], min_run_length: int = 10
) -> list[tuple[int, int]]:
    """Find consecutive runs of 'call' frames that should be collapsed."""
    runs = call_run_ranges(frame_info_list, min_run_length)
    # Chronological frames always end with a non-call (error) frame, so any
    # call run should have been closed above.
    assert not runs or runs[-1][1] < len(frame_info_list) - 1
    return runs


def _print_chronological(
    frames: list[FrameInfo],
    term_width: int,
    no_inspector: bool = False,
) -> tuple[str, int | None]:
    """Print frames in chronological order; returns ``(output, last_banner_start)``."""
    output = ""
    last_banner_start = None
    chrono_frames = frames
    if not chrono_frames:
        return output, last_banner_start

    # Build frame info list for all chronological frames
    frame_info_list = [_get_chrono_frame_info(frinfo) for frinfo in chrono_frames]

    # Find collapsible call runs
    collapse_ranges = _find_collapsible_call_runs(frame_info_list, min_run_length=10)

    # Build set of frame indices to skip
    skip_indices = {i for start, end in collapse_ranges for i in range(start + 1, end)}
    ellipsis_after = {start: end - start - 1 for start, end in collapse_ranges}

    # Calculate max location and function widths for alignment
    location_widths = [
        _display_width(info["location_part"])
        for i, info in enumerate(frame_info_list)
        if i not in skip_indices
    ]
    function_widths = [
        _display_width(info["function_part"])
        for i, info in enumerate(frame_info_list)
        if i not in skip_indices
    ]
    location_width = max(location_widths, default=0)
    function_width = max(function_widths, default=0)

    # Build output lines
    output_lines = []
    exception_banners = []  # List of (insert_after_line_idx, banner_output)

    for i, info in enumerate(frame_info_list):
        if i in skip_indices:
            continue

        lines = _build_chrono_frame_lines(
            info, location_width, function_width, term_width
        )
        for line, plain_len, is_marked in lines:
            output_lines.append((line, plain_len, i, is_marked))

        # Add ellipsis after first frame of collapsed run
        if i in ellipsis_after:
            skipped = ellipsis_after[i]
            ellipsis_line = f"{INDENT}{ELLIPSIS}⋮ {skipped} more calls{RESET}"
            ellipsis_plain_len = len(INDENT) + 2 + len(f"{skipped} more calls")
            output_lines.append((ellipsis_line, ellipsis_plain_len, i, False))

        # Check if this frame has parallel branches (subexceptions) to print
        parallel_branches = info["frinfo"].get("parallel")
        if parallel_branches:
            # Build subexception summaries
            sub_output = _build_subexception_summaries(parallel_branches, term_width)
            exception_banners.append((len(output_lines), sub_output))

        # Check if this frame has exception info to print after it
        exc_info = info["frinfo"].get("exception")
        if exc_info:
            # Record that we need to insert exception banner after this frame's lines
            banner = _build_exception_banner(exc_info, term_width)
            exception_banners.append((len(output_lines), banner))

    # Get variable inspector lines for all frames with variables
    all_inspector_lines = []
    all_inspector_min_widths = []
    inspector_frame_indices = []
    if not no_inspector:
        inspector_frame_indices = _find_all_inspector_frames(frame_info_list)
        for frame_idx in inspector_frame_indices:
            frinfo = frame_info_list[frame_idx]["frinfo"]
            variables = frinfo.get("variables", [])
            insp_lines, min_width = _build_variable_inspector(variables, term_width)
            all_inspector_lines.append(insp_lines)
            all_inspector_min_widths.append(min_width)

    # Build final output, inserting exception banners at the right positions
    if all_inspector_lines:
        # Complex case: merge inspectors and banners
        chrono_output, last_banner_start = _merge_chrono_output(
            output_lines,
            all_inspector_lines,
            all_inspector_min_widths,
            term_width,
            inspector_frame_indices,
            exception_banners,
            frame_info_list,
        )
        output += chrono_output
    else:
        # Simpler case: just insert banners
        banner_idx = 0
        for li, (line, _, _, _) in enumerate(output_lines):
            output += f"{line}{EOL}"
            # Check if we need to insert a banner after this line
            while banner_idx < len(exception_banners):
                insert_pos, banner = exception_banners[banner_idx]
                if li + 1 == insert_pos:
                    last_banner_start = len(output)
                    output += banner
                    banner_idx += 1
                else:
                    break
        # Any remaining banners (when banner position > last output line)
        for _, banner in exception_banners[banner_idx:]:  # pragma: no cover
            last_banner_start = len(output)
            output += banner

    return output, last_banner_start


def _wrap_text(text: str, width: int) -> list[str]:
    """Wrap *text* so each line fits within *width* terminal columns."""
    # In contrast to textwrap.wrap(), this function counts emoji etc. display width, ignoring ANSI codes.
    if not text or width <= 0:
        return [text]

    lines: list[str] = []
    line = ""
    line_width = 0

    for m in re.finditer(r"\s*\S+", text):
        word = m.group(0).lstrip()
        word_width = _display_width(word)

        if line_width and line_width + 1 + word_width > width:
            lines.append(line)
            line, line_width = "", 0

        if word_width > width:
            # Hard break an unbreakable word at the display-width boundary.
            chunk_width = 0
            chunk_chars: list[str] = []
            for char in word:
                char_width = _display_width(char)
                if chunk_width + char_width > width:
                    lines.append("".join(chunk_chars))
                    chunk_chars = []
                    chunk_width = 0
                chunk_chars.append(char)
                chunk_width += char_width
            line = "".join(chunk_chars)
            line_width = chunk_width
        else:
            if line_width:
                line += " "
                line_width += 1
            line += word
            line_width += word_width

    if line:
        lines.append(line)

    return lines or [text]


def _truncate_ansi(colored: str, max_width: int) -> str:
    """Truncate a colored string, appending a dim ellipsis."""
    ellipsis = f"{RESET}{DIM}…{RESET}"
    ellipsis_width = _display_width(ellipsis)
    if max_width <= ellipsis_width:
        return ellipsis
    chunk = _wrap_code_line(colored, max_width - ellipsis_width)[0]
    return chunk + (RESET if not chunk.endswith(RESET) else "") + ellipsis


def _wrap_code_line(colored: str, max_width: int) -> list[str]:
    """Wrap a colored code line, restoring active styles on each continuation."""
    if not colored or max_width <= 0:
        return [colored]

    chunks: list[str] = []
    current: list[str] = []
    color: list[str] = []
    width = 0

    for m in _TOKEN_RE.finditer(colored):
        token = m.group(0)
        if token.startswith("\x1b"):
            current.append(token)
            if token.endswith("m"):
                # Track active SGR sequences; replaying them in order
                # reproduces the exact terminal state (RESET clears).
                color = [] if token[2:-1] == "0" else color + [token]
            continue

        w = _display_width(token)
        if width + w > max_width:
            chunks.append("".join(current))
            current, width = [], 0
            if color:
                # Restore active styles at the start of the continuation
                current.append("".join(color))
        current.append(token)
        width += w

    chunks.append("".join(current))
    return chunks


def _build_exception_banner(exc_info: ExceptionInfo, term_width: int) -> str:
    """Build exception-banner content lines (the formatter adds the border)."""
    exc_type = exc_info.get("type", "Exception")
    summary = exc_info.get("summary", "")
    message = exc_info.get("message", "")
    from_type = exc_info.get("from", "none")

    chain_suffix = chainmsg.get(from_type, "")
    type_prefix = f"{exc_type}{chain_suffix}: "
    type_prefix_colored = f"{EXC}{type_prefix}{RESET}"
    type_prefix_width = _display_width(type_prefix)

    # First banner line pays the border; continuations also pay the half-block.
    first_line_width = max(1, term_width - _LINE_PREFIX_WIDTH)
    cont_width = max(1, term_width - _LINE_PREFIX_WIDTH - _display_width("▐ "))

    summary_lines: list[str] = []
    if summary:
        # Keep the type prefix on the first line and wrap the summary after it;
        # continuation lines are indented by the half-block marker.
        first_summary_width = max(1, first_line_width - type_prefix_width)
        wrapped_summary = _wrap_text(summary, first_summary_width)
        summary_lines.append(f"{type_prefix_colored}{BOLD}{wrapped_summary[0]}{RESET}")
        if len(wrapped_summary) > 1:
            remaining = summary[len(wrapped_summary[0]) :]
            summary_lines.extend(
                f"{BOLD}{line}{RESET}" for line in _wrap_text(remaining, cont_width)
            )
    else:
        summary_lines.append(type_prefix_colored)

    body_lines_raw: list[str] = []
    if summary != message:
        body = message
        if summary and body.startswith(summary):
            body = body[len(summary) :]
            if body.startswith("\n"):
                body = body[1:]
        body_lines_raw = body.split("\n")

    body_lines_wrapped: list[str] = []
    for para in body_lines_raw:
        body_lines_wrapped.extend(_wrap_text(para, cont_width) if para else [""])

    lines = summary_lines + body_lines_wrapped

    if len(lines) > 100:
        skipped = len(lines) - 40
        lines = lines[:20] + [f"{ELLIPSIS}⋮ {skipped} more lines{RESET}"] + lines[-20:]

    if len(lines) > 1:
        marker = f"{DIM}▐{RESET} "
        lines[1:] = [marker + line for line in lines[1:]]

    return "".join(line + EOL for line in lines)


def _build_subexception_summaries(
    parallel_branches: list[list[FrameInfo]], term_width: int
) -> str:
    """Build one-line summaries for each subexception branch."""
    output = ""
    border_width = _LINE_PREFIX_WIDTH  # "│ "
    marker = f"{DIM}▐{RESET} "
    marker_width = _display_width(marker)

    for branch in parallel_branches:
        # Get the summary for this branch, reserving space for the border and
        # the half-block prefix that visually groups it under the parent.
        summary = _get_branch_summary(branch, term_width - border_width - marker_width)
        output += f"{marker}{summary}{EOL}"

    return output


def _get_branch_summary(branch: list[FrameInfo], max_width: int) -> str:
    """Get a one-line summary for a subexception branch."""
    if not branch:
        return f"{EXC}(empty){RESET}"

    # Find the last frame with an exception (the final error in this branch)
    last_exc_info = None
    last_frame = None
    last_frame_with_parallel = None
    for frame in branch:
        if frame.get("exception"):
            last_exc_info = frame["exception"]
            last_frame = frame
        if frame.get("parallel"):
            last_frame_with_parallel = frame

    # If there are nested parallel branches, show them recursively
    if last_frame_with_parallel and last_frame_with_parallel.get("parallel"):
        nested = last_frame_with_parallel["parallel"]
        nested_summaries = []
        for nested_branch in nested:
            nested_summaries.append(_get_branch_summary(nested_branch, max_width - 4))
        return (
            f"{EXC}[{RESET}"
            + f"{EXC}, {RESET}".join(nested_summaries)
            + f"{EXC}]{RESET}"
        )

    if not last_exc_info:
        return f"{EXC}(no exception){RESET}"

    # Build location:lineno function prefix
    loc_prefix = ""
    if last_frame:  # pragma: no cover
        location = last_frame["location"]
        lineno = last_frame["cursor_line"]
        function = last_frame["function"]
        notebook_cell = last_frame["notebook_cell"]

        # Notebook cells (In [N]) don't need line numbers displayed
        if location and lineno and not notebook_cell:
            loc_prefix = f"{LOCFN}{location}{LOC_LINENO}:{lineno}{RESET} "
            if function:
                loc_prefix += f"{FUNC}{function}{RESET}: "
        elif location and notebook_cell:
            loc_prefix = f"{LOCFN}{location}{RESET} "
            if function:
                loc_prefix += f"{FUNC}{function}{RESET}: "
        elif function:
            loc_prefix = f"{FUNC}{function}{RESET}: "

    exc_type = last_exc_info.get("type", "Exception")
    summary = last_exc_info.get("summary", "")

    # Calculate plain text length (without ANSI codes)
    loc_plain = ANSI_ESCAPE_RE.sub("", loc_prefix)
    exc_part = f"{exc_type}: {summary}"
    total_plain_len = len(loc_plain) + len(exc_part)

    # Truncate summary if too long
    if total_plain_len > max_width:
        available = max_width - len(loc_plain) - len(exc_type) - 3  # ": " + "…"
        summary = summary[:available] + "…" if available > 0 else "…"

    return f"{loc_prefix}{EXC}{exc_type}:{RESET} {BOLD}{summary}{RESET}"


def _get_frame_label(frinfo: FrameInfo) -> tuple[str, str]:
    """Get the label for a frame (path:lineno function)."""
    cursor_line = frinfo["cursor_line"]
    notebook_cell = frinfo["notebook_cell"]

    # Use relative path if file is within CWD, otherwise use prettified location
    filename = frinfo["filename"]  # Full path (may be None for notebook cells)
    location = frinfo["location"]  # Display path (always set)
    if filename:
        try:
            fn = Path(filename)
            cwd = Path.cwd()
            if fn.is_absolute() and cwd in fn.parents:
                location = fn.relative_to(cwd).as_posix()
        except (ValueError, OSError):
            pass

    # Build label with colors: green filename, dark grey :lineno, light blue function
    # Location comes first, then function (if present)
    # Colon goes after function if present, otherwise after location
    function_label = function_display(frinfo["function"], frinfo["function_suffix"])

    # Build the location text with colors
    location_text = (
        location if notebook_cell else f"{location}{LOC_LINENO}:{cursor_line}{RESET}"
    )
    location_suffix = "" if function_label else ":"
    location_part = f"{LOCFN}{location_text}{location_suffix}{RESET}"

    function_part = f"{FUNC}{function_label}{RESET}:" if function_label else ""

    return location_part, function_part


def _get_chrono_frame_info(frinfo: FrameInfo) -> _ChronoFrameInfo:
    """Gather info needed to print a chronological frame."""
    location_part, function_part = _get_frame_label(frinfo)
    fragments = frinfo["fragments"]
    frame_range = frinfo["range"]
    relevance = frinfo["relevance"]
    exc_info = frinfo.get("exception")

    # Get marked lines
    marked_lines = [
        li for li in fragments if any(f.get("mark") for f in li.get("fragments", []))
    ]

    return {
        "location_part": location_part,
        "function_part": function_part,
        "fragments": fragments,
        "frame_range": frame_range,
        "relevance": relevance,
        "exc_info": exc_info,
        "marked_lines": marked_lines,
        "frinfo": frinfo,
    }


def _build_chrono_frame_lines(
    info: _ChronoFrameInfo, location_width: int, function_width: int, term_width: int
) -> list[tuple[str, int, bool]]:
    """Build output lines for a chronological frame."""
    location_part = info["location_part"]
    function_part = info["function_part"]
    fragments = info["fragments"]
    frame_range = info["frame_range"]
    relevance = info["relevance"]
    info["exc_info"]
    frinfo = info["frinfo"]

    # Calculate padding for alignment
    loc_pad = " " * (location_width - _display_width(location_part))
    func_pad = " " * (function_width - _display_width(function_part))
    label = f"{location_part}{loc_pad} {function_part}{func_pad}"

    start = frinfo.get("linenostart", 1)
    symbol = symbols.get(relevance, "")
    symbol_colored = f"{EM_CALL}{symbol}{RESET}" if symbol else ""

    desc = frinfo.get("symbol_desc") or symdesc[relevance]

    # Width available for the content after the left border "│ "
    content_width = max(1, term_width - _LINE_PREFIX_WIDTH)
    single_marked = len(info["marked_lines"]) == 1

    raw_lines: list[tuple[str, bool, bool]] = []
    symbol_suffix = f"{symbol_colored} {SYMBOLDESC}{desc}{RESET}" if symbol else ""

    if not fragments:
        # Show "(no source code)" with the symbol emoji like a code line would have
        line = f"{INDENT}{label} {NO_SOURCE}(no source code){symbol_colored}{RESET}"
        raw_lines.append((line, False, bool(symbol)))
    elif relevance == "call":
        # One-liner for call frames
        if info["marked_lines"]:
            # Build full code with em parts
            code_parts = []
            # Also track em parts for potential collapsing
            em_ranges = []
            em_start = None
            plain_len = 0  # Track position for em_ranges

            for line_info in info["marked_lines"]:
                for fragment in line_info.get("fragments", []):
                    mark = fragment.get("mark")
                    em = fragment.get("em")
                    if mark:
                        colored = _format_fragment_call(fragment)
                        plain = fragment["code"].rstrip("\n\r")
                        # Track em ranges
                        if em in EMPHASIS_BEG:
                            em_start = plain_len
                        code_parts.append(colored)
                        plain_len += len(plain)
                        if em in EMPHASIS_FIN and em_start is not None:
                            em_ranges.append((em_start, plain_len))
                            em_start = None
                # Add space between marked regions from different lines
                if (
                    code_parts and line_info != info["marked_lines"][-1]
                ):  # pragma: no cover
                    code_parts.append(" ")
                    plain_len += 1
            code_colored = "".join(code_parts)
            code_plain = ANSI_ESCAPE_RE.sub("", code_colored)

            # Collapse em parts longer than 20 chars
            if em_ranges:  # pragma: no cover
                em_start_pos = min(s for s, e in em_ranges)
                em_end_pos = max(e for s, e in em_ranges)
                em_text = code_plain[em_start_pos:em_end_pos]

                if len(em_text) > 20:
                    collapsed_em = em_text[0] + "…" + em_text[-1]
                    code_plain = (
                        code_plain[:em_start_pos]
                        + collapsed_em
                        + code_plain[em_end_pos:]
                    )
                    # Rebuild colored version
                    code_colored = (
                        code_plain[:em_start_pos]
                        + EM_CALL
                        + collapsed_em
                        + RESET
                        + code_plain[em_start_pos + len(collapsed_em) :]
                    )

            line = f"{INDENT}{label} {code_colored}{symbol_colored}"
            raw_lines.append((line, False, bool(symbol)))
        else:  # pragma: no cover
            line = f"{INDENT}{label} {symbol_colored}"
            raw_lines.append((line, False, bool(symbol)))
    else:
        # Full format for error/warning/stop/except frames
        label_line = f"{INDENT}{label}"
        raw_lines.append((label_line, False, False))

        marked_line_nums = set()
        for ml in info["marked_lines"]:
            marked_line_nums.add(ml["line"])

        for line_info in fragments:
            line_num = line_info["line"]
            abs_line = start + line_num - 1
            line_fragments = line_info.get("fragments", [])
            is_marked = line_num in marked_line_nums

            code_colored = "".join(_format_fragment(f) for f in line_fragments)
            code_part = f"{CODE_INDENT}{code_colored}"

            if frame_range and abs_line == frame_range["lfinal"] and symbol:
                if (
                    _display_width(code_part) + 1 + _display_width(symbol_suffix)
                    <= content_width
                ):
                    raw_lines.append((f"{code_part} {symbol_suffix}", is_marked, True))
                else:
                    raw_lines.append((code_part, is_marked, True))
                    raw_lines.append((symbol_suffix, False, True))
            else:
                raw_lines.append((code_part, is_marked, False))

    lines: list[tuple[str, int, bool]] = []
    for line, is_marked, has_symbol in raw_lines:
        width = _display_width(line)
        if width <= content_width:
            if is_marked and not line.endswith(RESET):
                line = line + RESET
            if (
                line == symbol_suffix
                and lines
                and lines[-1][1] + 1 + width <= content_width
            ):
                prev, prev_width, prev_marked = lines[-1]
                lines[-1] = (f"{prev} {line}", prev_width + 1 + width, prev_marked)
                continue
            lines.append((line, width, is_marked))
            continue

        should_wrap = has_symbol or (is_marked and single_marked)
        if should_wrap:
            for chunk in _wrap_code_line(line, content_width):
                if is_marked and not chunk.endswith(RESET):
                    chunk = chunk + RESET
                lines.append((chunk, _display_width(chunk), is_marked))
        else:
            lines.append(
                (_truncate_ansi(line, content_width), content_width, is_marked)
            )

    return lines


def _find_call_line_ranges(
    output_lines: list[tuple[str, int, int, bool]],
    frame_info_list: list[_ChronoFrameInfo],
) -> list[tuple[int, int]]:
    """Find line ranges for call frames that can be used for inspector expansion."""
    call_ranges = []
    current_start = None

    for li, (_, _, fidx, _) in enumerate(output_lines):  # pragma: no cover
        if fidx < len(frame_info_list) and frame_info_list[fidx]["relevance"] == "call":
            if current_start is None:
                current_start = li
        else:
            if current_start is not None:
                call_ranges.append((current_start, li - 1))
                current_start = None

    # Handle trailing call frames
    if current_start is not None:  # pragma: no cover
        call_ranges.append((current_start, len(output_lines) - 1))

    return call_ranges


def _compute_inspector_positions(
    output_lines: list[tuple[str, int, int, bool]],
    inspector_frames: list[int],
    inspector_data: list[tuple[InspectorLines, int]],  # [(lines, error_line), ...]
    frame_info_list: list[_ChronoFrameInfo],
) -> tuple[list[int], int]:
    """Compute vertical positions for all inspectors, avoiding overlap."""
    if not inspector_frames:  # pragma: no cover
        return [], 0

    # Get call line ranges that can be used for expansion
    call_ranges = _find_call_line_ranges(output_lines, frame_info_list)

    positions = []
    extra_lines_after = {}  # frame_idx -> extra lines needed after it
    min_allowed_start = 0  # Tracks where next inspector can start (prevents overlap)

    for idx, frame_idx in enumerate(inspector_frames):
        inspector_lines, error_line = inspector_data[idx]
        inspector_height = len(inspector_lines)

        # Find frame boundaries
        frame_start, frame_end = _find_frame_line_range(output_lines, frame_idx)

        # Account for any extra lines added after previous frames
        extra_before = sum(v for k, v in extra_lines_after.items() if k < frame_idx)
        frame_start += extra_before
        frame_end += extra_before
        error_line += extra_before

        # Find adjacent call lines that we can expand into
        expandable_above = frame_start
        expandable_below = frame_end

        # Look for call frames before this frame
        for call_start, call_end in call_ranges:  # pragma: no cover
            adj_call_start = call_start + extra_before
            adj_call_end = call_end + extra_before
            if adj_call_end == frame_start - 1:
                expandable_above = adj_call_start
            if adj_call_start == frame_end + 1:
                expandable_below = adj_call_end

        # Respect minimum start to prevent overlap
        expandable_above = max(expandable_above, min_allowed_start)

        # Calculate ideal position (arrow in middle, pointing at error line)
        ideal_arrow_pos = inspector_height // 2
        ideal_start = error_line - ideal_arrow_pos

        # Strategy 1: Try to fit within own frame
        if frame_end - frame_start + 1 >= inspector_height:
            # Enough space in frame, position centered on error line
            inspector_start = max(
                frame_start, min(ideal_start, frame_end - inspector_height + 1)
            )
        # Strategy 2: Expand to adjacent call lines
        elif (
            expandable_below - expandable_above + 1 >= inspector_height
        ):  # pragma: no cover
            # Enough space with expansion
            inspector_start = max(
                expandable_above,
                min(ideal_start, expandable_below - inspector_height + 1),
            )
        # Strategy 3: Add extra empty lines after frame
        else:  # pragma: no cover
            # Not enough space even with expansion, need extra lines
            available_space = expandable_below - expandable_above + 1
            needed_extra = inspector_height - available_space
            extra_lines_after[frame_idx] = needed_extra

            # Position at the top of available space
            inspector_start = expandable_above

        # Ensure we respect the minimum allowed start
        inspector_start = max(inspector_start, min_allowed_start)

        # Ensure arrow points at error line
        arrow_line_idx = error_line - inspector_start
        if arrow_line_idx < 0:  # pragma: no cover
            inspector_start = error_line
            arrow_line_idx = 0
        elif arrow_line_idx >= inspector_height:  # pragma: no cover
            inspector_start = error_line - inspector_height + 1
            arrow_line_idx = inspector_height - 1

        positions.append(inspector_start)

        # Update minimum start for next inspector (prevent overlap)
        inspector_end = inspector_start + inspector_height
        # Add any extra lines we're adding after this frame
        if frame_idx in extra_lines_after:
            inspector_end += extra_lines_after[frame_idx]
        min_allowed_start = inspector_end

    total_extra = sum(extra_lines_after.values())
    return positions, total_extra


def _truncate_inspector_line(
    insp_line: str, insp_width: int, value_start: int, available_for_content: int
) -> str:
    """Truncate an inspector line to fit the space available at render time."""
    if available_for_content <= 0:
        return "…"
    if insp_width <= available_for_content:
        return insp_line

    # Split the coloured line into prefix and value at the value_start plain-text
    # boundary, keeping the prefix's ANSI colouring intact.
    plain_idx = 0
    colored_idx = 0
    while plain_idx < value_start and colored_idx < len(insp_line):
        if insp_line[colored_idx] == "\x1b":
            while colored_idx < len(insp_line) and insp_line[colored_idx] != "m":
                colored_idx += 1
            colored_idx += 1  # skip the 'm'
        else:
            plain_idx += 1
            colored_idx += 1

    prefix_colored = insp_line[:colored_idx]
    value_colored = insp_line[colored_idx:]
    prefix_width = _display_width(prefix_colored)

    available_for_value = available_for_content - prefix_width
    if available_for_value <= 1:
        return "…"

    value_plain = ANSI_ESCAPE_RE.sub("", value_colored)
    inline_marker = " … "
    if inline_marker in value_plain:
        left, _, right = value_plain.partition(inline_marker)
        marker_width = _display_width(inline_marker)
        left_width = _display_width(left)

        # Shorten the right side, preserving its end.
        best_right = ""
        for i in range(1, len(right) + 1):
            suffix = right[-i:]
            if (
                left_width + marker_width + _display_width(suffix)
                <= available_for_value
            ):
                best_right = suffix
            else:
                break

        if best_right:
            truncated = f"{left}{inline_marker}{best_right}"
        else:
            # Right side removed entirely; shorten the left side and keep the
            # ellipsis at the end.
            end_marker = "…"
            end_width = _display_width(end_marker)
            best_left = ""
            for i in range(1, len(left) + 1):
                prefix = left[:i]
                if _display_width(prefix) + end_width <= available_for_value:
                    best_left = prefix
                else:
                    break
            truncated = f"{best_left}{end_marker}"
    else:
        # Truncate the value in display columns using the column-aware wrapper,
        # reserving one column for the trailing ellipsis.
        truncated = _wrap_code_line(value_colored, available_for_value - 1)[0]
        truncated = f"{truncated}…"

    # Restore the trailing reset if the line was coloured so later output is
    # not affected by leaked ANSI styles.
    if "\x1b" in prefix_colored or "\x1b" in value_colored:
        truncated = f"{truncated}{RESET}"

    return f"{prefix_colored}{truncated}"


def _merge_chrono_output(
    output_lines: list[tuple[str, int, int, bool]],
    all_inspector_lines: list[InspectorLines],
    all_inspector_min_widths: list[int],
    term_width: int,
    inspector_frame_indices: list[int],
    exception_banners: list[tuple[int, str]],
    frame_info_list: list[_ChronoFrameInfo],
) -> tuple[str, int | None]:
    """Merge chronological output with multiple inspectors and exception banners."""
    if not inspector_frame_indices:  # pragma: no cover
        # No inspectors, just output lines and banners
        output = ""
        last_banner_start: int | None = None
        banner_idx = 0
        for li, (line, _, _, _) in enumerate(output_lines):
            output += f"{line}{EOL}"
            while banner_idx < len(exception_banners):
                insert_pos, banner = exception_banners[banner_idx]
                if li + 1 == insert_pos:
                    last_banner_start = len(output)
                    output += banner
                    banner_idx += 1
                else:
                    break
        for _, banner in exception_banners[banner_idx:]:
            last_banner_start = len(output)
            output += banner
        return output, last_banner_start

    # Build inspector data: (lines, error_line) for each inspector
    inspector_data = []
    for i, frame_idx in enumerate(inspector_frame_indices):
        frame_start, frame_end = _find_frame_line_range(output_lines, frame_idx)
        error_line = _find_last_marked_line(output_lines, frame_start, frame_end)
        inspector_data.append((all_inspector_lines[i], error_line))

    # Compute positions
    positions, total_extra = _compute_inspector_positions(
        output_lines, inspector_frame_indices, inspector_data, frame_info_list
    )

    # Build a map of which inspector is active at each line
    inspector_at: dict[
        int, tuple[int, int, int, int]
    ] = {}  # line -> (insp_idx, insp_line_idx, arrow_line, col)

    # Calculate column for each inspector and populate inspector_at
    for insp_idx, (frame_idx, insp_lines) in enumerate(
        zip(inspector_frame_indices, all_inspector_lines, strict=True)
    ):
        inspector_start = positions[insp_idx]
        inspector_height = len(insp_lines)

        # Get error line for this inspector (adjusted for any extra lines)
        frame_start, frame_end = _find_frame_line_range(output_lines, frame_idx)
        error_line = _find_last_marked_line(output_lines, frame_start, frame_end)

        # Account for extra lines added by earlier inspectors
        extra_before = 0
        for prev_idx in range(insp_idx):  # pragma: no cover
            prev_frame_idx = inspector_frame_indices[prev_idx]
            prev_frame_start, prev_frame_end = _find_frame_line_range(
                output_lines, prev_frame_idx
            )
            if prev_frame_end < frame_start:
                # Check if this inspector needed extra lines
                prev_height = len(all_inspector_lines[prev_idx])
                prev_available = prev_frame_end - max(0, positions[prev_idx]) + 1
                if prev_height > prev_available:
                    extra_before += prev_height - prev_available

        # Calculate arrow position
        arrow_line = error_line - inspector_start
        if arrow_line < 0:  # pragma: no cover
            arrow_line = 0
        elif arrow_line >= inspector_height:  # pragma: no cover
            arrow_line = inspector_height - 1

        # Calculate column position for this inspector. Use the frame width as
        # the starting point, but cap it so the inspector always has a usable
        # amount of space (at least a third of the terminal, or 20 columns).
        max_line_len = 0
        for li in range(  # pragma: no cover
            inspector_start, min(inspector_start + inspector_height, len(output_lines))
        ):
            if li < len(output_lines):
                max_line_len = max(max_line_len, output_lines[li][1])

        min_content_width = max(20, term_width // 3)
        max_inspector_col = max(
            _LINE_PREFIX_WIDTH + 2,
            term_width - min_content_width - 4,
        )
        inspector_col = min(max_line_len + 2, max_inspector_col)

        # Calculate available space - inspector must not overlap with code
        # Available = term_width - inspector_col - 4 (for box/arrow chars)
        available_width = term_width - inspector_col - 4

        # Get minimum required width for this inspector (name: type = + some value)
        min_required = all_inspector_min_widths[insp_idx]

        # Skip this inspector if not enough space to show meaningful content
        if available_width < min_required:
            continue

        # Mark lines where this inspector is active
        for insp_line_idx in range(inspector_height):
            line_idx = inspector_start + insp_line_idx
            inspector_at[line_idx] = (
                insp_idx,
                insp_line_idx,
                arrow_line,
                inspector_col,
            )

    # Build output
    output = ""
    last_banner_start = None
    banner_idx = 0

    for li in range(len(output_lines)):
        line, plain_len, frame_idx, is_marked = output_lines[li]

        # Check if inspector is active at this line
        if li in inspector_at:
            insp_idx, insp_line_idx, arrow_line, inspector_col = inspector_at[li]
            insp_lines = all_inspector_lines[insp_idx]
            insp_line, insp_width, value_start = insp_lines[insp_line_idx]
            inspector_height = len(insp_lines)

            cursor_pos = f"{ESC}{inspector_col + 1}G"
            is_first = insp_line_idx == 0
            is_last = insp_line_idx == inspector_height - 1
            is_arrow = insp_line_idx == arrow_line

            # Truncate inspector content to the width actually available now
            # that the inspector column is known.
            available_for_content = (
                term_width - inspector_col - 4
            )  # 4 = box chars + space
            insp_line = _truncate_inspector_line(
                insp_line, insp_width, value_start, available_for_content
            )

            if is_arrow:
                if is_first and is_last:
                    box_char = SINGLE_T
                elif is_first:
                    box_char = BOX_TR
                elif is_last:
                    box_char = BOX_BR
                else:
                    box_char = BOX_VL
                output += f"{line}{cursor_pos}{DIM}{ARROW_LEFT}{BOX_H}{box_char}{RESET} {insp_line}{EOL}"
            else:
                if is_first:
                    box_char = BOX_TL
                elif is_last:
                    box_char = BOX_BL  # pragma: no cover
                else:
                    box_char = BOX_V
                output += f"{line}{cursor_pos}  {DIM}{box_char}{RESET} {insp_line}{EOL}"

        else:
            output += f"{line}{EOL}"

        # Emit any inspector lines that extend past this frame before inserting
        # exception banners. The previous line's EOL already provides the box
        # border, so we only need to position the inspector content.
        for insp_idx, fidx in enumerate(inspector_frame_indices):
            frame_start, frame_end = _find_frame_line_range(output_lines, fidx)
            if li == frame_end:
                inspector_start = positions[insp_idx]
                inspector_height = len(all_inspector_lines[insp_idx])
                inspector_end = inspector_start + inspector_height

                for extra_li in range(li + 1, inspector_end):
                    if extra_li not in inspector_at:
                        continue
                    _, insp_line_idx, _, inspector_col = inspector_at[extra_li]
                    insp_lines = all_inspector_lines[insp_idx]
                    insp_line, insp_width, value_start = insp_lines[insp_line_idx]

                    available_for_content = (
                        term_width - inspector_col - 4
                    )  # 4 = box chars + space
                    insp_line = _truncate_inspector_line(
                        insp_line, insp_width, value_start, available_for_content
                    )

                    cursor_pos = f"{ESC}{inspector_col + 1}G"
                    is_last = insp_line_idx == len(insp_lines) - 1
                    box_char = BOX_BL if is_last else BOX_V
                    output += f"{cursor_pos}  {DIM}{box_char}{RESET} {insp_line}{EOL}"

        # Insert exception banner if needed
        while banner_idx < len(exception_banners):
            insert_pos, banner = exception_banners[banner_idx]
            if li + 1 == insert_pos:
                last_banner_start = len(output)
                output += banner
                banner_idx += 1
            else:
                break

    # Any remaining banners
    for _, banner in exception_banners[banner_idx:]:
        last_banner_start = len(output)
        output += banner

    return output, last_banner_start


def _format_fragment(fragment: Fragment) -> str:
    """Format a fragment returning colored string."""
    code = fragment["code"].rstrip("\n\r")
    mark = fragment.get("mark")
    em = fragment.get("em")

    colored_parts = []

    # Open mark if starting
    if mark in EMPHASIS_BEG:
        colored_parts.append(MARK_BG + MARK_TEXT)

    # Open em if starting (red text within the mark)
    if em in EMPHASIS_BEG:
        colored_parts.append(EM)

    # Add the code
    colored_parts.append(code)

    # Close em if ending; em is only used within mark, so when the mark
    # continues, restate its attributes to back out of the em styling.
    if em in EMPHASIS_FIN and mark in {"beg", "mid"}:
        colored_parts.append(MARK_BG + MARK_TEXT)

    # Close mark if ending
    if mark in EMPHASIS_FIN:
        colored_parts.append(RESET)

    return "".join(colored_parts)


def _format_fragment_call(fragment: Fragment) -> str:
    """Format a fragment for call frames: default color, only em in yellow."""
    code = fragment["code"].rstrip("\n\r")
    em = fragment.get("em")

    colored_parts = []

    # Open em if starting (yellow text)
    if em in EMPHASIS_BEG:
        colored_parts.append(EM_CALL)

    # Add the code
    colored_parts.append(code)

    # Close em if ending
    if em in EMPHASIS_FIN:
        colored_parts.append(RESET)

    return "".join(colored_parts)


def _build_variable_inspector(
    variables: list[Any], term_width: int
) -> tuple[InspectorLines, int]:
    """Build variable inspector lines."""
    if not variables:
        return [], 0

    # First pass: collect variable info and filter out non-displayable values
    var_data = []  # [(name, typename, val_str, fmt_hint), ...]
    for var_info in variables:
        name, typename, value, fmt_hint = normalize_variable(var_info)

        # Format the value as a string
        if isinstance(value, str):
            val_str = value
        elif isinstance(value, dict) and value.get("type") == "keyvalue":
            # Format key-value pairs inline
            pairs = [f"{k}: {v}" for k, v in value.get("rows", [])]
            val_str = "{" + ", ".join(pairs) + "}"
        elif isinstance(value, dict) and value.get("type") == "array":
            # Format array inline (simplified)
            rows = value.get("rows", [])
            if rows:
                val_str = (
                    "[" + ", ".join(str(c) for c in rows[0] if c is not None) + ", ...]"
                )
            else:
                val_str = "[]"
        elif isinstance(value, list):
            # Simple matrix/list format
            if value and isinstance(value[0], list):
                val_str = (
                    "["
                    + ", ".join(str(c) for c in value[0] if c is not None)
                    + ", ...]"
                )
            else:
                val_str = str(value)
        else:
            val_str = str(value)

        # Skip variables with no displayable value
        if val_str == "⋯":
            continue

        var_data.append((name, typename, val_str, fmt_hint))

    if not var_data:
        return [], 0

    # Calculate max width of "name: type" or "name" part for alignment
    max_name_part_len = 0
    for name, typename, _, _ in var_data:
        name_part_len = len(name) + len(": ") + len(typename) if typename else len(name)
        max_name_part_len = max(max_name_part_len, name_part_len)

    # Calculate minimum required width: "name: type = " + at least 5 chars of value
    prefix_width = max_name_part_len + len(" = ")
    min_required_width = prefix_width + 5

    # Build variable lines with right-aligned names.
    # Each result entry: (colored_line, display_width, value_start_col).
    # Lines are NOT truncated here; the single truncation pass in
    # _merge_chrono_output knows the actual inspector column.
    result = []
    for name, typename, val_str, fmt_hint in var_data:
        name_part = f"{name}: {typename}" if typename else name
        padding = " " * (max_name_part_len - len(name_part))
        indent = " " * prefix_width

        # Handle multi-line block format
        if fmt_hint == "block" and "\n" in val_str:  # pragma: no cover
            for i, val_line in enumerate(val_str.split("\n")):
                val_line_colored = val_line
                if i == 0:
                    # First line with name and type
                    if typename:
                        line = f"{VAR}{padding}{name}: {TYPE_COLOR}{typename} = {VAR}{val_line_colored}{RESET}"
                    else:
                        line = f"{VAR}{padding}{name} = {val_line_colored}{RESET}"
                    result.append((line, _display_width(line), prefix_width))
                else:
                    # Continuation lines (indented) - all value
                    line = f"{VAR}{indent}{val_line_colored}{RESET}"
                    result.append((line, _display_width(line), prefix_width))
        else:
            # Single line format
            val_str_colored = val_str
            if typename:
                line = f"{VAR}{padding}{name}: {TYPE_COLOR}{typename} = {VAR}{val_str_colored}{RESET}"
            else:
                line = f"{VAR}{padding}{name} = {val_str_colored}{RESET}"
            result.append((line, _display_width(line), prefix_width))

    return result, min_required_width

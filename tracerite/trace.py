from __future__ import annotations

import inspect
import linecache
import re
import sys
import tokenize
from collections import deque, namedtuple
from contextlib import suppress
from pathlib import Path
from secrets import token_urlsafe
from typing import Any
from urllib.parse import quote

from . import trace_cpy
from .chain_analysis import build_chronological_frames
from .inspector import extract_variables
from .logging import logger
from .syntaxerror import clean_syntax_error_message, extract_enhanced_positions

# Position range: lines are 1-based inclusive, columns are 0-based exclusive
Range = namedtuple("Range", ["lfirst", "lfinal", "cbeg", "cend"])


def compute_cursor_position(
    mark_range: Range | None,
    em_ranges: Range | list[Range] | None,
    linenostart: int,
    common_indent: str = "",
) -> tuple[int, int]:
    """Return the preferred cursor (line, column) for highlighting."""
    target = None
    if em_ranges:
        if isinstance(em_ranges, list) and em_ranges:
            target = em_ranges[-1]
        elif isinstance(em_ranges, Range):
            target = em_ranges
    if target is None:
        target = mark_range

    if target is None:
        return (linenostart, 0)

    return (
        linenostart + target.lfinal - 1,
        target.cend + len(common_indent),
    )


# Will be set to an instance if loaded as an IPython extension by %load_ext
ipython: Any = None

# Locations considered to be bug-free (library code, not user code), capture pretty suffix
libdir = re.compile(
    r".*(?:site-packages|dist-packages)/(.+)"
    r"|.*/lib/python\d+\.\d+/(.+)"
    r"|.*/bin/([^/]+)(?<!\.py)"  # CLI scripts
    r"|.*/\.cache/(.+)"
)

# Messages for exception chaining (oldest-first order)
# Suffix added to exception type when chained from a previous exception
chainmsg = {
    "cause": " from previous",
    "context": " in except",
    "none": "",
}

# Symbol descriptions for display in HTML and TTY outputs
symdesc = {
    "call": "Call",
    "warning": "Call from your code",
    "except": "Call from except",
    "error": "",
    "stop": "",
}

# Symbols for each frame relevance type
symbols = {"call": "➤", "warning": "⚠️", "error": "💣", "stop": "🛑", "except": "⚠️"}


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
    """Normalize a VarInfo namedtuple or old tuple into (name, typename, value, fmt)."""
    if hasattr(var_info, "name"):
        return (
            var_info.name,
            var_info.typename,
            var_info.value,
            var_info.format_hint,
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


def _collect_leaf_exception_types(subexceptions: list[list[dict]]) -> list[str]:
    """Recursively collect leaf exception type names from subexception chains."""
    return [
        leaf
        for sub in subexceptions
        for leaf in (
            _collect_leaf_exception_types(sub[-1]["subexceptions"])
            if sub and sub[-1].get("subexceptions")
            else [sub[-1].get("type", "Exception")]
            if sub
            else []
        )
    ]


def _attach_leaf_types(exc_chain: list[dict], chrono_frames: list[dict]) -> None:
    """Attach ExceptionGroup leaf types to the final exception banner in frames."""
    if not exc_chain:
        return
    subexceptions = exc_chain[-1].get("subexceptions")
    if not subexceptions:
        return
    leaf_types = _collect_leaf_exception_types(subexceptions)
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
    chain = _collect_exception_chain(exc, **kwargs)
    chain = _digest_exception_chain(chain)
    chronological = build_chronological_frames(chain)
    chronological = _finalize_chronological(chronological, chain)
    return chronological


def extract_chain_exceptions(exc=None, **kwargs) -> list:
    """Extract raw exception info dicts, oldest first (internal)."""
    return _digest_exception_chain(_collect_exception_chain(exc, **kwargs))


# =============================================================================
# Stage 1: collect the raw exception chain
# =============================================================================


def _collect_exception_chain(exc=None, **kwargs) -> list[dict]:
    """Return raw exception objects in chronological order, oldest first.

    Each element is a small metadata dict with the live exception object and
    the kwargs that should be passed when it is digested.  Skip-related kwargs
    are attached only to the outermost (newest) exception.
    """
    chain = []
    exc = exc or sys.exc_info()[1]
    while exc:
        chain.append(exc)
        exc = exc.__cause__ or None if exc.__suppress_context__ else exc.__context__
    chain = list(reversed(chain))
    return [{"exc": e, "kwargs": kwargs if e is chain[-1] else {}} for e in chain]


# =============================================================================
# Stage 2: digest each exception into raw frames
# =============================================================================


def _digest_exception_chain(raw_chain: list[dict]) -> list[dict]:
    """Convert the raw chain into fully digested exception info dicts."""
    return [
        extract_exception(item["exc"], _defer_variables=True, **item["kwargs"])
        for item in raw_chain
    ]


# =============================================================================
# Stage 4: finalize chronological frames
# =============================================================================


def _finalize_chronological(chronological: list[dict], chain: list[dict]) -> list[dict]:
    """Fill variables and attach metadata that belongs to the final view."""
    _fill_chronological_variables(chronological)
    _attach_leaf_types(chain, chronological)
    return chronological


def _create_summary(message):
    """Extract the first line of the exception message as summary."""
    return message.split("\n", 1)[0]


def _chain_reason(e: BaseException) -> str:
    """Return the chaining relationship for an exception."""
    if e.__cause__:
        return "cause"
    if e.__context__ and not e.__suppress_context__:
        return "context"
    return "none"


def _set_relevances(frames: list, e: BaseException) -> None:
    """Mark the error, stop, warning, and call frames."""
    if not frames:
        return

    # Last frame is where the exception occurred
    # ExceptionGroups get "stop" like BaseExceptions - the real errors are in subexceptions
    is_regular_exception = isinstance(e, Exception) and not _is_exception_group(e)
    frames[-1]["relevance"] = "error" if is_regular_exception else "stop"

    # Check if the last frame (error frame) is in user code
    last_filename = (
        frames[-1].get("original_filename") or frames[-1].get("filename") or ""
    )
    if _libdir_match(Path(last_filename).as_posix()) is None:
        return
    # Error is in library code - find the last user code frame to mark as warning
    for frame in reversed(frames[:-1]):  # Exclude the last frame  # pragma: no cover
        filename = frame.get("original_filename") or frame.get("filename") or ""
        if _libdir_match(Path(filename).as_posix()) is None:
            # This is user code - mark as warning (bug origin)
            frame["relevance"] = "warning"
            break


def extract_exception(
    e, *, skip_outmost=0, skip_until=None, _defer_variables=False
) -> dict:
    raw_tb = e.__traceback__
    try:
        tb = inspect.getinnerframes(raw_tb)
    except IndexError:  # Bug in inspect internals, find_source()
        logger.exception("Bug in inspect?")
        tb = []
        raw_tb = None

    # For SyntaxError, check if the error is in user code (notebook cell or matching skip_until)
    syntax_frame = None
    if isinstance(e, SyntaxError):
        syntax_frame = _extract_syntax_error_frame(e)
        if syntax_frame:
            # Check if this is a notebook cell (using IPython's filename map) or matches skip_until
            is_user_code = _is_notebook_cell(e.filename) or (
                skip_until and skip_until in (e.filename or "")
            )
            if is_user_code:
                skip_outmost = len(tb)  # Skip all frames

    if skip_until and skip_outmost == 0:
        for i, frame in enumerate(tb):
            if skip_until in frame.filename:
                skip_outmost = i
                break
    tb = tb[skip_outmost:]

    # Also skip the same number of frames from raw_tb
    if raw_tb and skip_outmost > 0:
        for _ in range(skip_outmost):
            if raw_tb:
                raw_tb = raw_tb.tb_next

    # Header and exception message
    message = getattr(e, "message", "") or str(e)
    # For SyntaxError, trim redundant location info from message
    if isinstance(e, SyntaxError):
        message = clean_syntax_error_message(message)
    summary = _create_summary(message)
    # Check if context is suppressed (raise X from None) - affects source trimming
    f = _chain_reason(e)
    try:
        frames = extract_frames(
            tb,
            raw_tb,
            except_block=(f != "none"),
            exc=e,
            exc_message=message,
            _defer_variables=_defer_variables,
        )
        # For SyntaxError, add the synthetic frame showing the problematic code
        if syntax_frame:
            # Demote the previous frame (compile, exec, etc.) to call only
            if frames and frames[-1]["relevance"] == "error":
                frames[-1]["relevance"] = "call"
            frames.append(syntax_frame)
    except Exception:
        logger.exception("Error extracting traceback")
        frames = None

    # Determine if this is a "stop" type exception (BaseException or ExceptionGroup)
    # These suppress inner library frames, showing only up to the last user code frame.
    # ExceptionGroups suppress because the interesting parts are in subexceptions.
    is_stop_type = not isinstance(e, Exception) or _is_exception_group(e)

    result = {
        "type": type(e).__name__,
        "message": message,
        "summary": summary,
        "from": f,
        "repr": repr(e),
        "frames": frames or [],
        "suppress_inner": is_stop_type,
    }

    # Extract subexceptions for ExceptionGroups (Python 3.11+)
    # These form parallel timelines within the group's traceback
    subexceptions = _extract_subexceptions(
        e,
        skip_outmost=skip_outmost,
        skip_until=skip_until,
        _defer_variables=_defer_variables,
    )
    if subexceptions:
        result["subexceptions"] = subexceptions

    return result


def _extract_subexceptions(
    e, *, skip_outmost=0, skip_until=None, _defer_variables=False
) -> list[list[dict]] | None:
    """Return parallel exception chains for an ExceptionGroup, or None."""
    # Check if this is an ExceptionGroup (Python 3.11+)
    # BaseExceptionGroup is the base class for both ExceptionGroup and BaseExceptionGroup
    if not hasattr(e, "exceptions") or not isinstance(
        getattr(e, "exceptions", None), (tuple, list)
    ):
        return None

    subexceptions = e.exceptions
    if not subexceptions:
        return None

    # Extract each subexception as its own chain
    # Each subexception may itself be an ExceptionGroup with nested subexceptions
    parallel_chains = []
    for sub_exc in subexceptions:
        # Recursively extract the chain for this subexception
        # This handles nested ExceptionGroups and exception chaining within each sub
        sub_chain = _extract_subexception_chain(
            sub_exc,
            skip_outmost=skip_outmost,
            skip_until=skip_until,
            _defer_variables=_defer_variables,
        )
        if sub_chain:  # pragma: no cover
            parallel_chains.append(sub_chain)

    return parallel_chains if parallel_chains else None


def _extract_subexception_chain(
    exc, *, skip_outmost=0, skip_until=None, _defer_variables=False
) -> list[dict]:
    """Extract the exception chain for one ExceptionGroup subexception."""
    chain = []
    current = exc
    while current:
        chain.append(current)
        current = (
            current.__cause__ or None
            if current.__suppress_context__
            else current.__context__
        )
    # Reverse to get oldest first
    chain = list(reversed(chain))

    # Extract info for each exception in the chain
    # Pass skip args only to the last one (the actual subexception)
    kwargs = {"skip_outmost": skip_outmost, "skip_until": skip_until}
    result = [
        extract_exception(
            e, _defer_variables=_defer_variables, **(kwargs if e is chain[-1] else {})
        )
        for e in chain
    ]
    return result


def _is_notebook_cell(filename):
    """Check if the filename corresponds to a Jupyter notebook cell."""
    try:
        return filename in ipython.compile._filename_map  # type: ignore[attr-defined]
    except (AttributeError, KeyError, TypeError):
        return False


def _is_exception_group(e: BaseException) -> bool:
    """Check if exception is an ExceptionGroup (Python 3.11+)."""
    # Check for BaseExceptionGroup which is the base class for both
    # ExceptionGroup and BaseExceptionGroup
    return hasattr(e, "exceptions") and isinstance(
        getattr(e, "exceptions", None), (tuple, list)
    )


def _find_except_start_for_line(frame, lineno: int) -> int | None:
    """Return the 'except' line number if lineno is inside a handler."""
    from .chain_analysis import (
        find_try_block_for_except_line,
        parse_source_for_try_except,
    )

    try:
        filename = frame.f_code.co_filename
        blocks = parse_source_for_try_except(filename)
        # Find the innermost except block containing this line
        block = find_try_block_for_except_line(blocks, lineno)
        if block:
            return block.except_start
    except Exception:  # pragma: no cover
        pass
    return None


def _get_source_lines_from_code(code, lineno: int, end_lineno: int | None = None):
    """Fallback source-line retrieval for code objects (e.g. REPL, exec)."""
    # Python 3.13+ has linecache._getline_from_code for interactive code
    if not hasattr(linecache, "_getline_from_code"):
        return None, None  # pragma: no cover

    # First, check if we can get the error line at all
    error_line = linecache._getline_from_code(code, lineno)
    if not error_line:
        return None, None

    first_lineno = code.co_firstlineno
    is_module = code.co_name in (
        "<module>",
        "<listcomp>",
        "<dictcomp>",
        "<setcomp>",
        "<genexpr>",
    )

    # For module level, just get context around the error line
    if is_module:
        start = max(1, lineno - 10)
        final = (end_lineno or lineno) + 3
        lines = []
        actual_start = None
        for ln in range(start, final + 1):
            line = linecache._getline_from_code(code, ln)
            if line:
                if actual_start is None:
                    actual_start = ln
                lines.append(line)
            elif lines and ln > (end_lineno or lineno):  # pragma: no cover
                break  # Stop at empty lines after error (e.g., end of source)
        # Defensive: error_line check above guarantees we have lines
        if not lines or actual_start is None:  # pragma: no cover
            return None, None
        return lines, actual_start

    # For functions/methods, collect all lines starting from definition
    # then use inspect.getblock to find the function boundaries
    all_lines = []
    ln = first_lineno
    while True:
        line = linecache._getline_from_code(code, ln)
        if not line:
            break
        all_lines.append(line)
        ln += 1

    # Defensive: error_line check above guarantees we have lines
    if not all_lines:  # pragma: no cover
        return None, None

    # Use inspect.getblock to find the function's extent (same as inspect.getsourcelines)
    try:
        block_lines = inspect.getblock(all_lines)
    except (IndentationError, SyntaxError, tokenize.TokenError):  # pragma: no cover
        # Fallback: just use lines up to a reasonable extent
        block_lines = all_lines[: (end_lineno or lineno) - first_lineno + 3]

    return block_lines, first_lineno


def extract_source_lines(
    frame, lineno, end_lineno=None, *, notebook_cell=False, except_block=False
):
    try:
        lines, start = _get_source_from_frame(frame)
        except_start = (
            _find_except_start_for_line(frame, lineno) if except_block else None
        )

        lines, start = _slice_source_context(
            lines, start, lineno, end_lineno, notebook_cell, except_start
        )

        error_idx, end_idx = _error_indices(lineno, end_lineno, start)
        if not _valid_error_position(lines, error_idx):
            return "", lineno, ""

        error_indent = _line_indent(lines[error_idx])
        lines, error_idx, end_idx, start = _trim_leading_lines(
            lines, error_idx, end_idx, start, error_indent
        )
        lines = _trim_trailing_lines(lines, end_idx, error_indent)

        lines, common_indent = _dedent_lines(lines)
        return "".join(lines), start, common_indent
    except OSError:
        return _fallback_source_lines(frame, lineno, end_lineno)


def _get_source_from_frame(frame):
    """Get source lines and starting line number for a frame."""
    lines, start = inspect.getsourcelines(frame)
    if start == 0:
        start = 1
    return lines, start


def _slice_source_context(
    lines, start, lineno, end_lineno, notebook_cell, except_start
):
    """Slice source lines to the desired context window around the error."""
    if notebook_cell:
        if except_start is not None and except_start >= start:
            lines_before = lineno - except_start  # pragma: no cover
        else:
            lines_before = 0
        lines_after = (end_lineno - lineno) if end_lineno else 0
    else:
        lines_before = 10
        lines_after = (end_lineno - lineno + 2) if end_lineno else 2

    slice_start = max(0, lineno - start - lines_before)
    slice_end = max(0, lineno - start + lines_after + 1)

    slice_start = _find_clean_start_line(lines, slice_start)
    lines = lines[slice_start:slice_end]
    start += slice_start

    return _trim_to_except_line(lines, start, except_start)


def _trim_to_except_line(lines, start, except_start):
    """Trim lines so they start from the except line when applicable."""
    if except_start is not None and except_start > start:
        skip = except_start - start
        if skip < len(lines):  # pragma: no branch
            lines = lines[skip:]
            start = except_start
    return lines, start


def _error_indices(lineno, end_lineno, start):
    """Return (error_idx, end_idx) within the sliced source window."""
    error_idx = lineno - start
    end_idx = (end_lineno - start) if end_lineno else error_idx
    return error_idx, end_idx


def _valid_error_position(lines, error_idx):
    """Check that the error line index falls within the available lines."""
    return bool(lines) and 0 <= error_idx < len(lines)


def _line_indent(line):
    """Return the indentation width of a line."""
    return len(line) - len(line.lstrip(" \t"))


def _trim_leading_lines(lines, error_idx, end_idx, start, error_indent):
    """Drop leading lines that are more indented than the error line."""
    while lines and error_idx > 0:
        first_line = lines[0]
        if first_line.strip() and _line_indent(first_line) <= error_indent:
            break
        start += 1
        lines.pop(0)
        error_idx -= 1
        end_idx -= 1
    return lines, error_idx, end_idx, start


def _trim_trailing_lines(lines, end_idx, error_indent):
    """Drop trailing lines that are less indented than the error line."""
    trim_after = end_idx + 1
    bracket_depth = _count_bracket_depth("".join(lines[: end_idx + 1]))
    while trim_after < len(lines):
        line = lines[trim_after]
        if bracket_depth > 0:  # pragma: no cover
            bracket_depth += _count_bracket_depth(line)
            trim_after += 1
            continue
        if line.strip() and _line_indent(line) < error_indent:
            break
        trim_after += 1
    return lines[:trim_after]


def _fallback_source_lines(frame, lineno, end_lineno):  # pragma: no cover
    """Fallback source retrieval for interactive code objects."""
    code = frame.f_code if hasattr(frame, "f_code") else frame
    fallback_lines, fallback_start = _get_source_lines_from_code(
        code, lineno, end_lineno
    )
    if fallback_lines:
        lines, common_indent = _dedent_lines(fallback_lines)
        return "".join(lines), fallback_start, common_indent
    return "", lineno, ""


class _CodeScanner:
    """Stateful scanner that skips Python strings, comments, and escapes.

    Tracks whether the scanner is currently inside a string literal and the
    net bracket nesting depth. The state can be queried across multiple lines,
    allowing callers to find safe split points in source code.
    """

    __slots__ = ("in_string", "bracket_depth", "escape_next")

    def __init__(self):
        self.in_string = None
        self.bracket_depth = 0
        self.escape_next = False

    def step(self, text: str, i: int) -> int:
        """Process one logical unit starting at index i; return next index."""
        if self.escape_next:
            self.escape_next = False
            return i + 1

        char = text[i]
        if char == "\\":
            self.escape_next = True
            return i + 1

        if self.in_string is None and char == "#":
            return len(text)

        if self.in_string is None:
            if char in ('"', "'") and text[i : i + 3] in ('"""', "'''"):
                self.in_string = text[i : i + 3]
                return i + 3
            if char in ('"', "'"):
                self.in_string = char
                return i + 1
            if char in "([{":
                self.bracket_depth += 1
            elif char in ")]}":
                self.bracket_depth -= 1
            return i + 1

        # Inside a string literal
        if self.in_string in ('"""', "'''") and text[i : i + 3] == self.in_string:
            self.in_string = None
            return i + 3
        if len(self.in_string) == 1 and char == self.in_string:
            self.in_string = None
        return i + 1

    def process(self, text: str) -> None:
        """Advance scanner state across a whole string."""
        i = 0
        while i < len(text):
            i = self.step(text, i)

    @property
    def in_code_context(self) -> bool:
        """Return True if scanner is not inside a string or unclosed brackets."""
        return self.in_string is None and self.bracket_depth <= 0


def _count_bracket_depth(text: str) -> int:
    """Count net bracket depth change, ignoring brackets in strings/comments."""
    scanner = _CodeScanner()
    scanner.process(text)
    return scanner.bracket_depth


def _find_clean_start_line(lines: list[str], target_idx: int) -> int:
    """Return the first line at/after target_idx outside an unclosed string/bracket."""
    if target_idx <= 0 or target_idx >= len(lines):
        return target_idx

    scanner = _CodeScanner()
    for line in lines[:target_idx]:
        scanner.process(line)

    if scanner.in_code_context:
        return target_idx

    # Scan forward until the unclosed context ends.
    for idx in range(target_idx, len(lines)):  # pragma: no cover
        scanner.process(lines[idx])
        if scanner.in_code_context:
            return idx + 1

    return target_idx  # pragma: no cover


def _get_full_source(frame, lineno=None):
    """Return (source, start_line) for a frame, using inspect and fallbacks."""
    try:
        lines, start = inspect.getsourcelines(frame)
        if start == 0:
            start = 1
        return "".join(lines), start
    except OSError:
        # Fallback: try to get source from code object (Python 3.13+ interactive code)
        # This is tested via subprocess tests in test_tty.py::TestInteractiveSourceRetrieval
        code = frame.f_code if hasattr(frame, "f_code") else frame  # pragma: no cover
        if lineno is None:  # pragma: no cover
            lineno = getattr(frame, "f_lineno", code.co_firstlineno)
        fallback_lines, fallback_start = _get_source_lines_from_code(
            code, lineno
        )  # pragma: no cover
        if fallback_lines:  # pragma: no cover
            return "".join(fallback_lines), fallback_start
        return None, None


def _libdir_match(path):
    """Check if path is in a library directory and return the short suffix if so."""
    m = libdir.fullmatch(path)
    if m:
        return next((g for g in m.groups() if g), "")
    return None


def format_location(filename, lineno, col=1):
    """Return (filename, location_string, urls) for a frame."""
    urls = {}
    location = None
    try:
        ipython_in = ipython.compile._filename_map[filename]  # type: ignore[attr-defined]
        location = f"In [{ipython_in}]"
        filename = None
    except (AttributeError, KeyError):
        pass
    if filename and Path(filename).is_file():
        fn = Path(filename).resolve()
        # vscode:// URLs use format vscode://file/path:line:col
        urls["VS Code"] = f"vscode://file{quote(fn.as_posix())}:{lineno}:{col}"
        cwd = Path.cwd()
        if cwd in fn.parents:
            fn = fn.relative_to(cwd)
            if ipython is not None:
                urls["Jupyter"] = f"/edit/{quote(fn.as_posix())}"
        filename = fn.as_posix()
    if not location and filename:
        # Use library short path if available, otherwise truncate long paths
        location = _libdir_match(filename)
        if location is None:
            split = (
                filename.rfind("/", 10, len(filename) - 20) + 1
                if len(filename) > 40
                else 0
            )
            location = filename[split:]
    # Ensure location is never None (fallback for edge cases)
    if not location:
        location = "<unknown>"
    return filename, location, urls


def _get_qualified_function_name(frame, function):
    """Get qualified function name with class prefix if available."""
    if function == "<module>":
        return None
    try:
        cls = next(
            v.__class__ if n == "self" else v
            for n, v in frame.f_locals.items()
            if n in ("self", "cls") and v is not None
        )
        function = f"{cls.__name__}.{function}"
    except StopIteration:
        pass
    return ".".join(function.split(".")[-2:])


def _extract_text_from_range(lines: str, mark_range) -> str | None:
    """Return the source text covered by a Range."""
    if mark_range is None:
        return None

    lines_list = lines.splitlines(keepends=True)

    # Convert to 0-based line indices
    start_line_idx = mark_range.lfirst - 1
    end_line_idx = mark_range.lfinal - 1

    # Bounds check
    if start_line_idx < 0 or end_line_idx >= len(lines_list):
        return None

    extracted_parts = []
    for line_idx in range(start_line_idx, end_line_idx + 1):
        line = lines_list[line_idx].rstrip("\r\n")

        if line_idx == start_line_idx == end_line_idx:
            # Single line case
            extracted_parts.append(line[mark_range.cbeg : mark_range.cend])
        elif line_idx == start_line_idx:
            # First line of multi-line
            extracted_parts.append(line[mark_range.cbeg :])
        elif line_idx == end_line_idx:
            # Last line of multi-line
            extracted_parts.append(line[: mark_range.cend])
        else:
            # Middle lines of multi-line
            extracted_parts.append(line)

    return " ".join(extracted_parts)


def _find_comprehension_range(lines: str, lineno: int, start: int):
    """Return the (start, end) line indices of a comprehension, or None."""
    import ast

    # Try to parse the source and find comprehensions containing the error line
    try:
        tree = ast.parse(lines)
    except SyntaxError:
        return None

    error_line_in_source = lineno - start + 1

    # Find comprehension nodes that contain the error line
    comprehension_types = (ast.ListComp, ast.SetComp, ast.DictComp, ast.GeneratorExp)
    for node in ast.walk(tree):
        if isinstance(
            node, comprehension_types
        ) and node.lineno <= error_line_in_source <= (node.end_lineno or node.lineno):
            comp_start = node.lineno - 1  # 0-based
            comp_end = node.end_lineno or node.lineno  # 1-based, inclusive
            return (comp_start, comp_end)

    return None


def _trim_source_to_comprehension(lines: str, lineno: int, start: int):
    """Trim source context to the enclosing comprehension if any."""
    result = _find_comprehension_range(lines, lineno, start)
    if result:
        lines_list = lines.splitlines(keepends=True)
        comp_start_idx, comp_end_idx = result
        trimmed = "".join(lines_list[comp_start_idx:comp_end_idx])
        new_start = start + comp_start_idx
        return trimmed, new_start
    return lines, start


def _get_variable_source_for_comprehension(
    lines: str, lineno: int, start: int, mark_range
) -> str:
    """Return source code for variable extraction, expanding comprehensions."""
    # Check if we're inside a comprehension
    comp_range = _find_comprehension_range(lines, lineno, start)

    if comp_range is not None:
        # Inside a comprehension: use full comprehension text
        lines_list = lines.splitlines(keepends=True)
        comp_start_idx, comp_end_idx = comp_range
        return "".join(lines_list[comp_start_idx:comp_end_idx])

    # Not in a comprehension: use marked text or fall back to full lines
    marked_text = _extract_text_from_range(lines, mark_range)
    return marked_text or lines


def _fallback_mark_range_for_line(lines, error_line_in_context):
    """Build a single-line mark Range when caret columns are missing."""
    lines_list = lines.splitlines(keepends=True)
    if not (1 <= error_line_in_context <= len(lines_list)):
        return None
    line = lines_list[error_line_in_context - 1]
    content, _ = _split_line_content(line)
    stripped = content.lstrip()
    if not stripped:
        return None
    start_col = len(content) - len(stripped)

    comment_start = _find_comment_start(content)
    if comment_start is not None:
        code_end = content[:comment_start].rstrip()
        end_col = len(code_end)
    else:
        end_col = len(content.rstrip())

    if end_col <= start_col:
        end_col = start_col + 1
    return Range(error_line_in_context, error_line_in_context, start_col, end_col)


def _extract_emphasis_columns(
    lines, error_line_in_context, end_line, start_col, end_col, start
):
    """Return the emphasis Range from caret anchors, or None."""
    if not (end_line and start_col is not None and end_col is not None):
        return None

    all_lines = lines.splitlines(keepends=True)
    segment_start = error_line_in_context - 1  # Convert to 0-based for indexing
    segment_end = end_line if end_line else error_line_in_context

    if not (0 <= segment_start < len(all_lines) and segment_end <= len(all_lines)):
        return None

    # Extract the segment using CPython's approach
    relevant_lines = all_lines[segment_start:segment_end]
    if not relevant_lines:
        # This can happen when re-raising an existing exception where CPython's
        # position info refers to the original raise site but end_line < error_line
        return None

    segment = "".join(relevant_lines)

    # Trim segment using start_col and end_col
    segment = segment[start_col : len(segment) - (len(relevant_lines[-1]) - end_col)]
    # Attempt to parse for anchors
    anchors = None
    with suppress(Exception):
        anchors = trace_cpy._extract_caret_anchors_from_line_segment(segment)
    if not anchors:
        return None

    l0, l1, c0, c1 = (
        anchors.left_end_lineno,
        anchors.right_start_lineno,
        anchors.left_end_offset,
        anchors.right_start_offset,
    )
    # We get 0-based line numbers and offsets within the segment,
    # so we need to adjust them to match the original code.
    if l0 == 0:
        c0 += start_col
    if l1 == 0:
        c1 += start_col

    # Convert to 1-based inclusive line numbers for consistency
    lfirst = l0 + segment_start + 1
    lfinal = l1 + segment_start + 1

    return Range(lfirst, lfinal, c0, c1)


def _build_position_map(raw_tb):
    """Map each frame object to its list of code positions."""
    position_map = {}
    if not raw_tb:
        return position_map
    try:
        for frame_obj, positions in trace_cpy._walk_tb_with_full_positions(raw_tb):
            position_map.setdefault(frame_obj, deque()).append(positions)
    except Exception:
        logger.exception("Error extracting position information")
    return position_map


def _extract_syntax_error_frame(e):
    """Create a synthetic frame dict for a SyntaxError showing the problematic code."""
    if not isinstance(e, SyntaxError):
        return None

    filename, lineno, end_lineno, start_col, end_col = _syntax_error_positions(e)
    if not filename or not lineno:
        return None

    notebook_cell = _is_notebook_cell(filename)
    lines, lines_list, start, source_from_text = _resolve_syntax_error_source(
        e, filename, notebook_cell
    )
    if not lines:
        return None

    start_col, end_col = _clamp_syntax_columns(lines_list, lineno, start_col, end_col)

    enhanced_mark, enhanced_em = extract_enhanced_positions(e, lines_list)
    if enhanced_mark:
        lineno = enhanced_mark.lfirst
        end_lineno = enhanced_mark.lfinal
        start_col = enhanced_mark.cbeg
        end_col = enhanced_mark.cend

    lines, lines_list, start, error_line_in_context, end_line = (
        _slice_syntax_error_window(
            lines_list, lineno, end_lineno, start, source_from_text
        )
    )

    mark_range, em_ranges = _build_syntax_mark_ranges(
        enhanced_mark,
        enhanced_em,
        start,
        error_line_in_context,
        end_line,
        start_col,
        end_col,
        lines,
    )

    fragments = _parse_lines_to_fragments(lines, mark_range, em_ranges)

    cursor_line, cursor_col = compute_cursor_position(mark_range, em_ranges, start, "")

    fmt_filename, location, urls = format_location(filename, cursor_line, cursor_col)

    codeline = lines_list[error_line_in_context - 1].strip() if lines_list else None

    return {
        "id": _make_trace_id(),
        "relevance": "error",
        "idframe": id(e),
        "filename": fmt_filename,
        "location": location,
        "notebook_cell": notebook_cell,
        "codeline": codeline,
        "range": (
            Range(lineno, end_lineno or lineno, start_col, end_col)
            if start_col is not None
            else None
        ),
        "cursor_line": cursor_line,
        "cursor_col": cursor_col,
        "linenostart": start,
        "lines": lines,
        "fragments": fragments,
        "function": None,
        "function_suffix": "",
        "urls": urls,
        "variables": [],
    }


def _syntax_error_positions(e):
    """Extract position information from a SyntaxError."""
    filename = e.filename
    lineno = e.lineno
    end_lineno = getattr(e, "end_lineno", None) or lineno
    start_col = (e.offset - 1) if e.offset else 0
    end_col = getattr(e, "end_offset", None)

    if end_col:
        end_col = end_col - 1
        if end_col <= start_col and end_lineno == lineno:
            end_col = start_col + 1
    else:
        end_col = start_col + 1

    return filename, lineno, end_lineno, start_col, end_col


def _resolve_syntax_error_source(e, filename, notebook_cell):
    """Resolve source text for a SyntaxError from linecache or e.text."""
    import linecache

    lines = None
    start = 1
    source_from_text = False

    try:
        if notebook_cell and ipython:
            try:
                cell_source = ipython.compile._filename_map.get(filename)
                if cell_source is not None:
                    all_lines = linecache.getlines(filename)
                    if all_lines:
                        lines = "".join(all_lines)
            except Exception:
                pass

        if not lines:
            all_lines = linecache.getlines(filename)
            if all_lines:
                lines = "".join(all_lines)

        if not lines and e.text:
            lines = e.text if e.text.endswith("\n") else e.text + "\n"
            start = e.lineno
            source_from_text = True
    except Exception:
        if e.text:
            lines = e.text if e.text.endswith("\n") else e.text + "\n"
            start = e.lineno
            source_from_text = True

    if not lines:
        return None, None, start, source_from_text

    return lines, lines.splitlines(keepends=True), start, source_from_text


def _clamp_syntax_columns(lines_list, lineno, start_col, end_col):
    """Clamp SyntaxError columns to the actual line length."""
    if lines_list and 1 <= lineno <= len(lines_list):
        max_col = len(lines_list[lineno - 1].rstrip("\n\r"))
        start_col = min(start_col, max(0, max_col - 1))
        end_col = max(min(end_col, max_col), start_col + 1)
        end_col = min(end_col, max_col)
    return start_col, end_col


def _slice_syntax_error_window(lines_list, lineno, end_lineno, start, source_from_text):
    """Slice SyntaxError source to a small window around the error."""
    error_line_in_context = lineno - start + 1
    end_line = end_lineno - start + 1 if end_lineno else error_line_in_context
    lines = "".join(lines_list)

    if source_from_text:
        return lines, lines_list, start, error_line_in_context, end_line

    display_first = min(lineno, len(lines_list))
    display_last = min(end_lineno or lineno, len(lines_list))

    slice_start = max(0, display_first - 3)
    slice_end = min(len(lines_list), display_last + 2)

    while slice_start < display_first - 1 and not lines_list[slice_start].strip():
        slice_start += 1
    while slice_end > display_last and not lines_list[slice_end - 1].strip():
        slice_end -= 1

    lines_list = lines_list[slice_start:slice_end]
    lines = "".join(lines_list)
    start = slice_start + 1
    error_line_in_context = lineno - start + 1
    end_line = end_lineno - start + 1 if end_lineno else None

    return lines, lines_list, start, error_line_in_context, end_line


def _build_syntax_mark_ranges(
    enhanced_mark,
    enhanced_em,
    start,
    error_line_in_context,
    end_line,
    start_col,
    end_col,
    lines,
):
    """Build mark/emphasis ranges for a SyntaxError frame."""
    if enhanced_mark:
        mark_range = Range(
            enhanced_mark.lfirst - start + 1,
            enhanced_mark.lfinal - start + 1,
            enhanced_mark.cbeg,
            enhanced_mark.cend,
        )
        em_ranges = (
            [
                Range(
                    em.lfirst - start + 1,
                    em.lfinal - start + 1,
                    em.cbeg,
                    em.cend,
                )
                for em in enhanced_em
            ]
            if enhanced_em
            else None
        )
    else:
        mark_lfinal = end_line or error_line_in_context
        mark_range = Range(error_line_in_context, mark_lfinal, start_col, end_col)
        em_ranges = _extract_emphasis_columns(
            lines,
            error_line_in_context,
            end_line,
            start_col,
            end_col,
            start,
        )

    return mark_range, em_ranges


def extract_frames(
    tb,
    raw_tb=None,
    *,
    except_block=False,
    exc=None,
    exc_message=None,
    _defer_variables=False,
) -> list:
    if not tb:
        return []

    position_map = _build_position_map(raw_tb)

    frames = []
    for frame, filename, lineno, function, codeline, _ in tb:
        hide = frame.f_globals.get("__tracebackhide__") or frame.f_locals.get(
            "__tracebackhide__"
        )
        if hide:
            if hide == "until":
                # Hide this frame and all previous frames
                frames = []
                continue
            hidden = True
        else:
            hidden = False

        frame_positions = position_map.get(frame)
        pos = frame_positions.popleft() if frame_positions else [None] * 4

        is_last_frame = frame is tb[-1][0]
        frame_info = _extract_single_frame(
            frame,
            filename,
            lineno,
            function,
            codeline,
            pos,
            hidden,
            is_last_frame,
            except_block=except_block,
        )
        if frame_info is not None:
            frames.append(frame_info)

    if exc is not None:
        _set_relevances(frames, exc)
    if _defer_variables:
        for frame in frames:
            frame.setdefault("variables", [])
    else:
        _fill_variables(frames, exc_message)
    return frames


def _extract_single_frame(
    frame,
    filename,
    lineno,
    function,
    codeline,
    pos,
    hidden,
    is_last_frame,
    *,
    except_block=False,
):
    """Extract a single frame's worth of traceback information."""
    pos_end_lineno, start_col, end_col = pos[1], pos[2], pos[3]
    notebook_cell = _is_notebook_cell(filename)

    lines, start, original_common_indent = extract_source_lines(
        frame,
        lineno,
        pos_end_lineno,
        notebook_cell=notebook_cell,
        except_block=except_block,
    )

    if not lines and not is_last_frame:
        if hidden:
            # Still include hidden frames with minimal info for chain analysis
            full_source, full_source_start = _get_full_source(frame)
            return {
                "id": _make_trace_id(),
                "relevance": "call",
                "hidden": True,
                "idframe": id(frame),
                "lineno": lineno,
                "variables": [],
                "full_source": full_source,
                "full_source_start": full_source_start,
            }
        return None

    full_source, full_source_start = _get_full_source(frame)

    lines, start = _trim_source_to_comprehension(lines, lineno, start)
    lines_list = lines.splitlines(keepends=True)
    lines, extra_indent = _dedent_lines(lines_list)
    lines = "".join(lines)
    total_indent = len(original_common_indent) + len(extra_indent)

    original_filename = filename
    function = _get_qualified_function_name(frame, function)

    error_line_in_context = lineno - start + 1
    end_line = pos_end_lineno - start + 1 if pos_end_lineno else None

    frame_range, mark_range = _build_frame_ranges(
        lineno,
        pos_end_lineno,
        error_line_in_context,
        end_line,
        start_col,
        end_col,
        total_indent,
        lines,
    )

    em_range = _extract_emphasis_columns(
        lines,
        error_line_in_context,
        end_line,
        mark_range.cbeg if mark_range else None,
        mark_range.cend if mark_range else None,
        start,
    )
    fragments = _parse_lines_to_fragments(lines, mark_range, em_range)

    cursor_line, cursor_col = compute_cursor_position(
        mark_range, em_range, start, original_common_indent + extra_indent
    )

    filename, location, urls = format_location(
        original_filename, cursor_line, cursor_col
    )

    variable_source = _get_variable_source_for_comprehension(
        lines, lineno, start, mark_range
    )

    result = {
        "id": _make_trace_id(),
        "relevance": "call",
        "hidden": hidden,
        "idframe": id(frame),
        "frame_obj": frame,
        "filename": filename,
        "original_filename": original_filename,
        "location": location,
        "notebook_cell": notebook_cell,
        "codeline": codeline[0].strip() if codeline else None,
        "range": frame_range,
        "lineno": lineno,
        "cursor_line": cursor_line,
        "cursor_col": cursor_col,
        "linenostart": start,
        "lines": lines,
        "fragments": fragments,
        "function": function,
        "function_suffix": "",
        "urls": urls,
        "variable_source": variable_source,
        "full_source": full_source,
        "full_source_start": full_source_start,
    }

    return result


def _fill_frame_variables(frame: dict, exc_message: str | None = None) -> None:
    """Extract variables for a single frame and drop its live frame object."""
    frame_obj = frame.pop("frame_obj", None)
    variable_source = frame.pop("variable_source", None)

    if frame.get("hidden") or frame_obj is None or variable_source is None:
        frame.setdefault("variables", [])
        return

    frame["variables"] = extract_variables(
        frame_obj.f_locals,
        variable_source,
        exc_message=exc_message,
    )


def _fill_variables(frames: list[dict], exc_message: str | None = None) -> None:
    """Populate the variables field for a Python-order frame list."""
    last_idx = len(frames) - 1
    for idx, frame in enumerate(frames):
        _fill_frame_variables(
            frame, exc_message=exc_message if idx == last_idx else None
        )


def _fill_chronological_variables(chrono_frames: list[dict]) -> None:
    """Fill variables on the final occurrence of each frame in chrono order."""
    seen: set[int] = set()

    def _process_branch(branch: list[dict]) -> None:
        for frame in reversed(branch):
            idframe = frame.get("idframe")
            if idframe is not None and idframe not in seen and "frame_obj" in frame:
                exc_message = frame.get("exception", {}).get("message")
                _fill_frame_variables(frame, exc_message=exc_message)
                seen.add(idframe)
            else:
                frame.setdefault("variables", [])
                frame.pop("frame_obj", None)
                frame.pop("variable_source", None)
            for sub_branch in frame.get("parallel", []):
                _process_branch(sub_branch)

    _process_branch(chrono_frames)


def _build_frame_ranges(
    lineno,
    pos_end_lineno,
    error_line_in_context,
    end_line,
    start_col,
    end_col,
    total_indent,
    lines,
):
    """Build the original and displayed source ranges for a frame."""
    if start_col is None or end_col is None:
        if error_line_in_context:
            fallback = _fallback_mark_range_for_line(lines, error_line_in_context)
            if fallback:
                frame_range = Range(
                    lineno,
                    pos_end_lineno or lineno,
                    fallback.cbeg + total_indent,
                    fallback.cend + total_indent,
                )
                return frame_range, fallback
        return None, None

    adjusted_start_col = max(0, start_col - total_indent)
    adjusted_end_col = max(0, end_col - total_indent)
    frame_range = Range(lineno, pos_end_lineno or lineno, start_col, end_col)
    mark_range = Range(
        error_line_in_context,
        end_line or error_line_in_context,
        adjusted_start_col,
        adjusted_end_col,
    )
    return frame_range, mark_range


def _make_trace_id() -> str:
    """Generate a short unique identifier for a traceback frame."""
    return f"tb-{token_urlsafe(12)}"


def _dedent_lines(lines: list[str]) -> tuple[list[str], str]:
    """Return (dedented_lines, common_indent)."""
    common_indent = _calculate_common_indent(lines)
    return [ln.removeprefix(common_indent) for ln in lines], common_indent


def _collect_positions_from_ranges(ranges, lines: list[str]) -> set[int]:
    """Collect character positions from a single Range or list of Ranges."""
    positions = set()
    if not ranges:
        return positions
    for rng in ranges if isinstance(ranges, list) else [ranges]:
        positions |= _convert_range_to_positions(rng, lines)
    return positions


def _calculate_common_indent(lines):
    """Calculate common indentation across all non-empty lines."""
    non_empty_lines = [line.rstrip("\r\n") for line in lines if line.strip()]
    if not non_empty_lines:
        return ""
    indent_len = min(len(ln) - len(ln.lstrip(" \t")) for ln in non_empty_lines)
    return non_empty_lines[0][:indent_len]


def _convert_range_to_positions(range_obj, lines):
    """Convert Range (1-based inclusive lines, 0-based exclusive columns) to absolute character positions."""
    positions = set()

    if not range_obj:
        return positions

    # Convert to 0-based line indices for processing
    start_line_idx = range_obj.lfirst - 1
    end_line_idx = range_obj.lfinal - 1

    # Calculate absolute positions
    char_pos = 0
    for line_idx, line in enumerate(lines):
        if start_line_idx <= line_idx <= end_line_idx:
            line_content = line.rstrip("\r\n")

            if line_idx == start_line_idx == end_line_idx:
                # Single line case
                for col in range(
                    max(0, range_obj.cbeg), min(len(line_content), range_obj.cend)
                ):
                    positions.add(char_pos + col)
            elif line_idx == start_line_idx:
                # First line of multi-line
                for col in range(max(0, range_obj.cbeg), len(line_content)):
                    positions.add(char_pos + col)
            elif line_idx == end_line_idx:
                # Last line of multi-line
                for col in range(0, min(len(line_content), range_obj.cend)):
                    positions.add(char_pos + col)
            else:
                # Middle lines of multi-line
                for col in range(len(line_content)):
                    positions.add(char_pos + col)

        char_pos += len(line)

    return positions


def _create_unified_fragments(lines_text, common_indent, mark_positions, em_positions):
    """Create fragments with unified mark/em highlighting."""
    lines = lines_text.splitlines(keepends=True)
    result = []

    for line_idx, line in enumerate(lines):
        line_num = line_idx + 1
        fragments = _parse_line_to_fragments_unified(
            line,
            common_indent,
            mark_positions,
            em_positions,
            sum(len(lines[i]) for i in range(line_idx)),  # char offset for this line
        )
        result.append({"line": line_num, "fragments": fragments})

    return result


def _parse_line_to_fragments_unified(
    line, common_indent, mark_positions, em_positions, line_char_offset
):
    """Parse a single line into fragments using unified highlighting."""
    line_content, line_ending = _split_line_content(line)
    if not line_content and not line_ending:
        return []

    # Process indentation
    fragments, remaining, pos = _process_indentation(line_content, common_indent)

    # Find comment split
    comment_start = _find_comment_start(remaining)

    if comment_start is not None:
        # Handle line with comment
        code_part = remaining[:comment_start]
        comment_part = remaining[comment_start:]

        # Process code part (with trimming)
        code_trimmed = code_part.rstrip()
        code_whitespace = code_part[len(code_trimmed) :]

        if code_trimmed:
            fragments.extend(
                _create_highlighted_fragments_unified(
                    code_trimmed, line_char_offset + pos, mark_positions, em_positions
                )
            )

        # Process comment part
        comment_trimmed = comment_part.rstrip()
        comment_trailing = comment_part[len(comment_trimmed) :]

        comment_with_leading_space = code_whitespace + comment_trimmed
        fragments.append({"code": comment_with_leading_space, "comment": "solo"})

        # Add trailing content
        trailing_content = comment_trailing + line_ending
        if trailing_content:
            fragments.append({"code": trailing_content, "trailing": "solo"})
    else:
        # Handle line without comment
        code_trimmed = remaining.rstrip()
        trailing_whitespace = remaining[len(code_trimmed) :]

        if code_trimmed:
            fragments.extend(
                _create_highlighted_fragments_unified(
                    code_trimmed, line_char_offset + pos, mark_positions, em_positions
                )
            )

        trailing_content = trailing_whitespace + line_ending
        if trailing_content:
            fragments.append({"code": trailing_content, "trailing": "solo"})

    return fragments


def _create_highlighted_fragments_unified(
    text, start_pos, mark_positions, em_positions
):
    """Create fragments with mark/em highlighting using unified position sets."""
    if not text:
        return []

    # Convert absolute positions to text-relative positions
    text_mark_positions = set()
    text_em_positions = set()

    for i in range(len(text)):
        abs_pos = start_pos + i
        if abs_pos in mark_positions:
            text_mark_positions.add(i)
        if abs_pos in em_positions:
            text_em_positions.add(i)

    # Create fragments using existing logic
    return _create_fragments_with_highlighting(
        text, text_mark_positions, text_em_positions
    )


def _parse_lines_to_fragments(lines_text, mark_range=None, em_ranges=None):
    """Split code lines into highlighted fragments."""
    lines = lines_text.splitlines(keepends=True)
    if not lines:
        return []

    common_indent = _calculate_common_indent(lines)

    # Convert both mark and em to position sets using unified logic
    mark_positions = _convert_range_to_positions(mark_range, lines)
    em_positions = _collect_positions_from_ranges(em_ranges, lines)

    # Create fragments using unified highlighting
    return _create_unified_fragments(
        lines_text, common_indent, mark_positions, em_positions
    )


def _split_line_content(line):
    """Split line into content and line ending."""
    if line.endswith("\r\n"):
        return line[:-2], "\r\n"
    elif line.endswith("\n"):
        return line[:-1], "\n"
    elif line.endswith("\r"):
        return line[:-1], "\r"
    else:
        return line, ""


def _process_indentation(line_content, common_indent):
    """Process dedent and additional indentation, return fragments and remaining content."""
    fragments = []
    pos = 0

    # Handle dedent (common indentation)
    if common_indent and len(line_content) > len(common_indent):
        dedent_text = line_content[: len(common_indent)]
        fragments.append({"code": dedent_text, "dedent": "solo"})
        pos = len(common_indent)

    # Handle additional indentation
    remaining = line_content[pos:]
    indent_match = re.match(r"^(\s+)", remaining)
    if indent_match:
        indent_text = indent_match.group(1)
        fragments.append({"code": indent_text, "indent": "solo"})
        pos += len(indent_text)
        remaining = remaining[len(indent_text) :]

    return fragments, remaining, pos


def _find_comment_start(text: str) -> int | None:
    """Find the start of a comment, ignoring # inside strings."""
    scanner = _CodeScanner()
    i = 0
    while i < len(text):
        if scanner.in_string is None and not scanner.escape_next and text[i] == "#":
            return i
        i = scanner.step(text, i)
    return None


def _positions_to_consecutive_ranges(positions):
    """Convert a set/list of positions to consecutive (start, end) ranges."""
    if not positions:
        return []

    sorted_positions = sorted(set(positions))
    ranges = []
    start = sorted_positions[0]
    end = start + 1

    for pos in sorted_positions[1:]:
        if pos == end:
            # Consecutive position, extend current range
            end = pos + 1
        else:
            # Gap found, close current range and start new one
            ranges.append((start, end))
            start = pos
            end = pos + 1

    # Close the last range
    ranges.append((start, end))
    return ranges


def _get_highlight_boundaries(text, mark_positions, em_positions):
    """Get all boundaries for highlighting (start/end of mark and em regions)."""
    boundaries = {0, len(text)}

    # Add mark boundaries
    for start, end in _positions_to_consecutive_ranges(mark_positions):
        boundaries.add(start)
        boundaries.add(end)

    # Add em boundaries
    for start, end in _positions_to_consecutive_ranges(em_positions):
        boundaries.add(start)
        boundaries.add(end)

    return sorted(boundaries)


def _create_fragments_with_highlighting(text, mark_positions, em_positions):
    """Create fragments with mark/em highlighting using beg/mid/fin/solo logic."""
    if not text:
        return []

    # Get all boundaries and create fragments
    boundaries = _get_highlight_boundaries(text, mark_positions, em_positions)
    mark_ranges = _positions_to_consecutive_ranges(mark_positions)
    em_ranges = _positions_to_consecutive_ranges(em_positions)

    fragments = []
    for i in range(len(boundaries) - 1):
        start = boundaries[i]
        end = boundaries[i + 1]

        if start >= len(text):
            break

        fragment_text = text[start:end]
        fragment = {"code": fragment_text}

        # Determine mark status
        mark_status = _get_highlight_status(start, end, mark_ranges)
        if mark_status:
            fragment["mark"] = mark_status

        # Determine em status
        em_status = _get_highlight_status(start, end, em_ranges)
        if em_status:
            fragment["em"] = em_status

        fragments.append(fragment)

    return fragments


def _get_highlight_status(frag_start, frag_end, ranges):
    """Determine beg/mid/fin/solo status for a fragment within ranges."""
    # Find overlapping ranges
    overlapping = []
    for range_start, range_end in ranges:
        if frag_start < range_end and frag_end > range_start:
            overlapping.append((range_start, range_end))

    if not overlapping:
        return None

    # Use the first overlapping range (they should align with fragment boundaries)
    range_start, range_end = overlapping[0]

    is_start = frag_start <= range_start
    is_end = frag_end >= range_end

    if is_start and is_end:
        return "solo"
    elif is_start:
        return "beg"
    elif is_end:
        return "fin"
    else:
        return "mid"

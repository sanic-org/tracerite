from __future__ import annotations

import logging
import os
import sys
import textwrap
import threading
from pathlib import Path
from typing import Any, TextIO

from .trace import chainmsg, extract_chain

# ANSI escape codes for terminal colors (can be monkeypatched for styling)
ESC = "\x1b["
RESET = f"{ESC}0m"
BG = f"{ESC}48;5;232m"  # Very dark grey background
RST = RESET + BG  # Reset but preserve dark grey background
CLR = f"{ESC}K"  # Clear to end of line (extends background color)
EOL = f"{CLR}\n{BG}"  # End of line: clear, newline, restore background
MARK_BG = f"{ESC}103m"  # Bright yellow background
MARK_TEXT = f"{ESC}30m"
EM = f"{ESC}31m"
LOCFN = f"{ESC}32m"
EM_CALL = f"{ESC}93m"  # Bright yellow
DARK_GREY = f"{ESC}90m"
EXC = f"{ESC}90m"  # Dark grey for exception text
ELLIPSIS = f"{ESC}90m"  # Dark grey for ellipsis/skipped calls
LOC_LINENO = f"{ESC}90m"  # Dark grey for :lineno
INS_TYPE = f"{ESC}90m"  # Dark grey for message text in inspector
SYMBOLDESC = f"{ESC}90m"  # Dark grey for symbol desc / exception type
FUNC = f"{ESC}94m"
VAR = f"{ESC}36m"
BOLD = f"{ESC}1m"
DIM = f"{ESC}2m"

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

INDENT = "  "  # Indent for call frame lines
CODE_INDENT = "    "  # Indent for code in frame

symbols = {"call": "➤", "warning": "⚠️", "error": "💣", "stop": "🛑"}
symdesc = {
    "call": "Call",
    "warning": "Call from your code",
    "error": "{type}",
    "stop": "{type}",
}

# Store the original hooks for unload
_original_excepthook = None
_original_threading_excepthook = None
_original_stream_handler_emit = None


def load(capture_logging: bool = True) -> None:
    """Load TraceRite as the default exception handler.

    Replaces sys.excepthook to use TraceRite's pretty TTY formatting
    for all unhandled exceptions, including those in threads and
    logging.exception() calls.
    Call unload() to restore the original exception handlers.

    Args:
        capture_logging: Whether to monkeypatch logging.StreamHandler.emit
            to format exceptions in logging.exception() calls. Defaults to True.

    Usage:
        import tracerite
        tracerite.load()  # Captures logging by default
        tracerite.load(capture_logging=False)  # Only captures sys.excepthook
    """
    global \
        _original_excepthook, \
        _original_threading_excepthook, \
        _original_stream_handler_emit

    if _original_excepthook is None:
        _original_excepthook = sys.excepthook

    if _original_threading_excepthook is None:
        _original_threading_excepthook = threading.excepthook

    if capture_logging and _original_stream_handler_emit is None:
        _original_stream_handler_emit = logging.StreamHandler.emit

    def _tracerite_excepthook(exc_type, exc_value, exc_tb):
        try:
            tty_traceback(exc=exc_value)
        except Exception:
            # Fall back to original excepthook on any error
            if _original_excepthook:
                _original_excepthook(exc_type, exc_value, exc_tb)
            else:
                sys.__excepthook__(exc_type, exc_value, exc_tb)

    def _tracerite_threading_excepthook(args):  # pragma: no cover (pytest intercepts)
        try:
            tty_traceback(exc=args.exc_value)
        except Exception:
            # Fall back to original threading excepthook on any error
            if _original_threading_excepthook:
                _original_threading_excepthook(args)
            else:
                sys.__excepthook__(args.exc_type, args.exc_value, args.exc_traceback)

    def _tracerite_stream_handler_emit(self, record: logging.LogRecord) -> None:
        """Emit a log record with TraceRite formatting for exceptions."""
        try:
            # Check if we have exception info to format specially
            if not record.exc_info or record.exc_info[1] is None:
                # No exception, use original emit
                return _original_stream_handler_emit(self, record)
            # Temporarily clear exc_info so format() doesn't include traceback
            exc_info = record.exc_info
            record.exc_info = None
            record.exc_text = None
            try:
                msg = self.format(record)
            finally:
                record.exc_info = exc_info

            # Temporarily restore original handler to avoid recursion
            original_emit = logging.StreamHandler.emit
            logging.StreamHandler.emit = _original_stream_handler_emit
            try:
                # Now format and write the exception using TraceRite
                tty_traceback(
                    exc=exc_info[1], file=self.stream, msg=msg + self.terminator
                )
            finally:
                logging.StreamHandler.emit = original_emit
        except RecursionError:
            raise
        except Exception:
            self.handleError(record)

    sys.excepthook = _tracerite_excepthook
    threading.excepthook = _tracerite_threading_excepthook
    if capture_logging:
        logging.StreamHandler.emit = _tracerite_stream_handler_emit  # type: ignore[attr-defined]


def unload() -> None:
    """Restore the original exception handlers.

    Removes TraceRite from sys.excepthook, threading.excepthook, and
    logging.StreamHandler.emit, restoring the previous handlers.
    """
    global \
        _original_excepthook, \
        _original_threading_excepthook, \
        _original_stream_handler_emit

    if _original_excepthook is not None:
        sys.excepthook = _original_excepthook
        _original_excepthook = None

    if _original_threading_excepthook is not None:
        threading.excepthook = _original_threading_excepthook
        _original_threading_excepthook = None

    if _original_stream_handler_emit is not None:
        logging.StreamHandler.emit = _original_stream_handler_emit
        _original_stream_handler_emit = None


def tty_traceback(
    exc: BaseException | None = None,
    chain: list[dict[str, Any]] | None = None,
    *,
    file: TextIO | None = None,
    msg: str | None = None,
    term_width: int | None = None,
    **extract_args: Any,
) -> None:
    """Format and print a traceback for terminal output (TTY).

    Outputs directly to the terminal (or specified file) to adapt to
    terminal features like window size. The chain is printed with the
    oldest exception first (order they occurred).
    """
    import re

    chain = chain or extract_chain(exc=exc, **extract_args)
    # Chain is already oldest-first from extract_chain

    if file is None:
        file = sys.stderr

    is_tty = file.isatty() if hasattr(file, "isatty") else False
    no_color = not is_tty
    no_inspector = not is_tty

    # Set dark grey background for entire traceback
    output = BG

    # Print the original log message if provided
    if msg:
        # Preserve custom background color for full lines, beyond resets
        output += CLR + msg.replace(RESET, RST).replace("\n", f"{CLR}\n") + EOL

    if term_width is None:
        try:
            term_width = os.get_terminal_size(file.fileno()).columns
        except (OSError, ValueError):
            term_width = 80

    # Pre-scan all frames to find duplicates and determine which should show inspector
    # Key: (filename, function) -> list of (exception_idx, frame_idx, relevance)
    frame_occurrences = {}
    for ei, e in enumerate(chain):
        for fi, frinfo in enumerate(e["frames"]):
            key = (frinfo.get("filename"), frinfo.get("function"))
            if key not in frame_occurrences:
                frame_occurrences[key] = []
            frame_occurrences[key].append((ei, fi, frinfo.get("relevance", "call")))

    # Determine which frame occurrences should show inspector:
    # Only the LAST non-call occurrence of each unique frame
    inspector_allowed = set()  # Set of (exception_idx, frame_idx)
    for occurrences in frame_occurrences.values():
        # Find the last non-call occurrence
        last_non_call = None
        for ei, fi, relevance in occurrences:
            if relevance != "call":
                last_non_call = (ei, fi)
        if last_non_call:
            inspector_allowed.add(last_non_call)

    for i, e in enumerate(chain):
        # Get chaining suffix for exception header
        chain_suffix = ""
        if i > 0:
            output += EOL  # Empty line between exceptions
            chain_suffix = chainmsg.get(e.get("from", "none"), "")

        output += _print_exception(
            e, term_width, i, inspector_allowed, chain_suffix, no_inspector
        )

    # Reset to original terminal colors
    output += EOL + RESET + CLR

    if no_color:
        # Strip all ANSI escape sequences for non-TTY output
        output = re.sub(r"\x1b\[[0-9;]*[A-Za-z]", "", output)

    file.write(output)


def _find_inspector_frame_idx(
    frame_info_list: list[dict[str, Any]],
    exception_idx: int,
    inspector_allowed: set[tuple[int, int]] | None,
) -> int | None:
    """Find the first non-call frame that is allowed to show inspector.

    Returns the frame index, or None if no suitable frame is found.
    """
    for i, info in enumerate(frame_info_list):
        if info["relevance"] != "call" and (
            inspector_allowed is None or (exception_idx, i) in inspector_allowed
        ):
            return i
    return None


def _find_frame_line_range(
    output_lines: list[tuple[str, int, int, bool]], inspector_frame_idx: int
) -> tuple[int, int]:
    """Find the line range for the inspector frame in output_lines.

    Returns (frame_line_start, frame_line_end) - both are valid indices.
    The caller guarantees inspector_frame_idx exists in output_lines.
    """
    frame_line_start = -1
    frame_line_end = -1

    for li, (_, _, fidx, _) in enumerate(output_lines):
        if fidx == inspector_frame_idx:
            if frame_line_start == -1:
                frame_line_start = li
            frame_line_end = li

    # By contract, the frame must exist in output_lines
    assert frame_line_start >= 0 and frame_line_end >= 0
    return frame_line_start, frame_line_end


def _find_last_marked_line(
    output_lines: list[tuple[str, int, int, bool]],
    frame_line_start: int,
    frame_line_end: int,
) -> int:
    """Find the last marked line within the frame range.

    Returns the line index of the last marked line, or frame_line_end if none are marked.
    """
    last_marked = frame_line_end  # Fallback to last line of frame

    for li in range(frame_line_start, frame_line_end + 1):
        _, _, _, is_marked = output_lines[li]
        if is_marked:
            last_marked = li

    return last_marked


def _find_collapsible_call_runs(
    frame_info_list: list[dict[str, Any]], min_run_length: int = 10
) -> list[tuple[int, int]]:
    """Find consecutive runs of 'call' frames that should be collapsed.

    Returns list of (start_idx, end_idx) tuples for runs of consecutive
    call frames with length >= min_run_length. end_idx is inclusive.
    """
    runs = []
    run_start = None

    for i, info in enumerate(frame_info_list):
        if info["relevance"] == "call":
            if run_start is None:
                run_start = i
        else:
            # End of a call run
            if run_start is not None:
                run_length = i - run_start
                if run_length >= min_run_length:
                    runs.append((run_start, i - 1))
                run_start = None

    return runs


def _print_exception(
    e: dict[str, Any],
    term_width: int,
    exception_idx: int = 0,
    inspector_allowed: set[tuple[int, int]] | None = None,
    chain_suffix: str = "",
    no_inspector: bool = False,
) -> str:
    """Print a single exception with its frames."""
    output = _build_exception_header(e, term_width, chain_suffix)
    output += _build_frames_output(
        e, term_width, exception_idx, inspector_allowed, no_inspector
    )
    return output


def _build_exception_header(
    e: dict[str, Any], term_width: int, chain_suffix: str
) -> str:
    """Build the exception header output."""
    output = ""
    # Exception header (not indented)
    summary, message = e["summary"], e["message"]
    exc_type = e["type"]
    type_prefix = f"{exc_type}{chain_suffix}: "
    type_prefix_len = len(type_prefix)
    cont_prefix = f"{EXC}{BOX_V}{RST} "
    cont_prefix_len = 2  # "│ "

    # Check if the full title fits on one line
    full_title_len = type_prefix_len + len(summary)
    if full_title_len <= term_width:
        # Fits on one line
        output += f"{EXC}{type_prefix}{RST}{BOLD}{summary}{RST}{EOL}"
    elif len(summary) <= term_width - cont_prefix_len:
        # Summary fits on its own line after wrapping
        output += f"{EXC}{type_prefix}{RST}{EOL}"
        output += f"{cont_prefix}{BOLD}{summary}{RST}{EOL}"
    else:
        # Word wrap with uniform width by padding first line to account for type_prefix
        # The padding simulates the type_prefix width, then we strip it from output
        padding = "\x00" * (type_prefix_len - cont_prefix_len)
        wrapped = textwrap.wrap(
            padding + summary,
            width=term_width - cont_prefix_len,
            break_long_words=False,
            break_on_hyphens=False,
        )
        # Remove padding from first line
        wrapped[0] = wrapped[0].lstrip("\x00")

        # Print lines
        for i, line in enumerate(wrapped):
            if i == 0:
                output += f"{EXC}{type_prefix}{RST}{BOLD}{line}{RST}{EOL}"
            else:
                output += f"{cont_prefix}{BOLD}{line}{RST}{EOL}"

    if summary != message:
        if message.startswith(summary):
            message = message[len(summary) :].strip("\n")
        # Format additional message lines with BOX_V prefix
        wrap_width = term_width - cont_prefix_len
        for line in message.split("\n"):
            if line:
                wrapped = textwrap.wrap(
                    line,
                    width=wrap_width,
                    break_long_words=False,
                    break_on_hyphens=False,
                ) or [line]
                for wrapped_line in wrapped:
                    output += f"{cont_prefix}{wrapped_line}{EOL}"
            else:
                # Preserve empty lines
                output += f"{cont_prefix.rstrip()}{EOL}"

    return output


def _build_frames_output(
    e: dict[str, Any],
    term_width: int,
    exception_idx: int,
    inspector_allowed: set[tuple[int, int]] | None,
    no_inspector: bool,
) -> str:
    """Build the frames output for an exception."""
    output = ""
    # Frames: caller first, then callee (deepest last)
    frames = e["frames"]

    # Pre-calculate frame info for alignment
    frame_info_list = []
    for frinfo in frames:
        info = _get_frame_info(e, frinfo)
        frame_info_list.append(info)

    # Find consecutive runs of call frames to collapse (>=10 consecutive calls)
    # Returns list of (start_idx, end_idx) for runs to collapse
    collapse_ranges = _find_collapsible_call_runs(frame_info_list, min_run_length=10)

    # Build set of frame indices to skip (middle frames in collapsed runs)
    skip_indices = set()
    ellipsis_after = {}  # frame_idx -> count of skipped frames
    for start_idx, end_idx in collapse_ranges:
        # Keep first and last, skip middle
        skipped_count = end_idx - start_idx - 1
        for i in range(start_idx + 1, end_idx):
            skip_indices.add(i)
        ellipsis_after[start_idx] = skipped_count

    # Find first non-call frame that is allowed to show inspector
    inspector_frame_idx = _find_inspector_frame_idx(
        frame_info_list, exception_idx, inspector_allowed
    )

    # Calculate max label width for call frame alignment
    # Only consider call frames that won't be skipped
    call_label_widths = [
        len(info["label_plain"])
        for i, info in enumerate(frame_info_list)
        if info["relevance"] == "call" and i not in skip_indices
    ]
    label_width = max(call_label_widths, default=0)

    # Build output lines with their plain text lengths
    output_lines = []  # List of (colored_line, plain_length, frame_idx, is_marked)
    for i, info in enumerate(frame_info_list):
        if i in skip_indices:
            continue
        lines = _build_frame_lines(info, label_width, term_width)
        for line, plain_len, is_marked in lines:
            output_lines.append((line, plain_len, i, is_marked))
        # Add ellipsis line after first frame of a collapsed run
        if i in ellipsis_after:
            skipped = ellipsis_after[i]
            ellipsis_line = f"{INDENT}{ELLIPSIS}⋮ {skipped} more calls{RST}"
            ellipsis_plain_len = len(INDENT) + 2 + len(f"{skipped} more calls")
            output_lines.append((ellipsis_line, ellipsis_plain_len, i, False))

    # Get variable inspector lines if we have a non-call frame
    inspector_lines = []
    if not no_inspector and inspector_frame_idx is not None:
        frinfo = frame_info_list[inspector_frame_idx]["frinfo"]
        variables = frinfo.get("variables", [])
        if variables:
            inspector_lines = _build_variable_inspector(variables, term_width)

    # Merge output with inspector
    if inspector_lines:
        output += _merge_inspector_output(
            output_lines, inspector_lines, term_width, inspector_frame_idx
        )
    else:
        # No inspector or no frame lines found, just print the code
        for line, _, _, _ in output_lines:
            output += f"{line}{EOL}"
    return output


def _merge_inspector_output(
    output_lines: list[tuple[str, int, int, bool]],
    inspector_lines: list[tuple[str, int]],
    term_width: int,
    inspector_frame_idx: int | None,
) -> str:
    """Merge output lines with inspector lines using cursor positioning."""
    output = ""
    assert inspector_frame_idx is not None
    frame_line_start, frame_line_end = _find_frame_line_range(
        output_lines, inspector_frame_idx
    )
    error_line = _find_last_marked_line(output_lines, frame_line_start, frame_line_end)
    inspector_height = len(inspector_lines)

    # Smart centering logic:
    # 1. Center around error line (last marked line where 💣 appears)
    # 2. Shift if centering goes out of bounds
    # 3. Prefer extending into call frames (above) over empty lines (below)
    # 4. Only add empty lines at the end as last resort

    # Calculate ideal centered position (error line in middle of inspector)
    ideal_arrow_pos = inspector_height // 2
    ideal_start = error_line - ideal_arrow_pos

    # Determine bounds: can extend upward into call frames (line 0),
    # but only extend into "call" relevance frames, not beyond output
    # The minimum start is 0 (can use all call frames above)
    min_start = 0

    # Apply centering with constraints
    inspector_start = ideal_start

    # Shift down if we're trying to go above available lines
    if inspector_start < min_start:
        inspector_start = min_start

    # Shift up if we're extending too far below available lines
    # First, see how many lines of output we have below inspector_start
    lines_available_below = len(output_lines) - inspector_start
    if lines_available_below < inspector_height:
        # Try to shift up, but not beyond min_start
        needed_shift = inspector_height - lines_available_below
        inspector_start = max(min_start, inspector_start - needed_shift)

    # Arrow line is where the error line falls within inspector
    arrow_line_idx = error_line - inspector_start
    assert 0 <= arrow_line_idx < inspector_height

    # Calculate inspector column: find the max line length in the range we'll use
    max_line_len = 0
    for li in range(
        inspector_start,
        min(inspector_start + inspector_height, len(output_lines)),
    ):
        max_line_len = max(max_line_len, output_lines[li][1])

    # Inspector width: arrow/spaces(2) + bar(1) + space(1) + content
    max_insp_width = max(w for _, w in inspector_lines) if inspector_lines else 0
    total_insp_width = 4 + max_insp_width  # "◀─❴ " or "  │ " + content

    # Place inspector right after the longest line, with some padding
    inspector_col = max_line_len + 2

    # But don't go beyond terminal width
    if inspector_col + total_insp_width > term_width:
        inspector_col = term_width - total_insp_width

    # Build output with inspector merged using cursor positioning
    inspector_count = len(inspector_lines)
    for li, (line, *_) in enumerate(output_lines):
        insp_idx = li - inspector_start
        if 0 <= insp_idx < inspector_count:
            insp_line, insp_width = inspector_lines[insp_idx]
            # Use cursor positioning to place inspector
            cursor_pos = (
                f"{ESC}{inspector_col + 1}G"  # +1 because columns are 1-indexed
            )
            # Determine which box character to use
            is_first = insp_idx == 0
            is_last = insp_idx == inspector_count - 1
            is_arrow = insp_idx == arrow_line_idx

            if is_arrow:
                # Arrow line: use appropriate corner or T-junction
                if is_first and is_last:
                    box_char = SINGLE_T  # {-junction for single line
                elif is_first:  # pragma: no cover
                    box_char = BOX_TR  # curved corner for first+arrow
                elif is_last:
                    box_char = BOX_BR  # curved corner for last+arrow
                else:
                    box_char = BOX_VL  # ┤-junction for middle arrow
                output += f"{line}{cursor_pos}{DIM}{ARROW_LEFT}{BOX_H}{box_char}{RST} {insp_line}{EOL}"
            else:
                # Non-arrow line: use corner or vertical
                if is_first:
                    box_char = BOX_TL  # ╭ curved corner for first
                elif is_last:
                    box_char = BOX_BL  # ╰ curved corner for last
                else:
                    box_char = BOX_V  # │ vertical for middle
                output += f"{line}{cursor_pos}  {DIM}{box_char}{RST} {insp_line}{EOL}"
        else:
            output += f"{line}{EOL}"

    # If inspector is taller than available lines, print remaining
    remaining_start = len(output_lines) - inspector_start
    if remaining_start < inspector_count:
        for idx in range(remaining_start, inspector_count):
            insp_line, insp_width = inspector_lines[idx]
            cursor_pos = f"{ESC}{inspector_col + 1}G"
            is_last = idx == inspector_count - 1
            box_char = BOX_BL if is_last else BOX_V
            output += f"{cursor_pos}  {DIM}{box_char}{RST} {insp_line}{EOL}"
    return output


def _get_frame_label(frinfo: dict[str, Any]) -> tuple[str, str]:
    """Get the label for a frame (path:lineno function)."""
    frame_range = frinfo.get("range")
    lineno = frame_range.lfirst if frame_range else "?"

    # Use relative path if file is within CWD, otherwise use prettified location
    filename = frinfo.get("filename")
    location = frinfo["location"]
    if filename:
        try:
            fn = Path(filename)
            cwd = Path.cwd()
            if fn.is_absolute() and cwd in fn.parents:
                location = fn.relative_to(cwd).as_posix()
        except (ValueError, OSError):  # pragma: no cover
            pass

    # Build label with colors: light blue function, green filename, dark grey :lineno
    label = ""
    label_plain = ""
    if frinfo["function"]:
        label_plain += f"{frinfo['function']} "
        label += f"{FUNC}{frinfo['function']} {RST}"
    label_plain += f"{location}:{lineno}"
    label += f"{LOCFN}{location}{LOC_LINENO}:{lineno}{RST}"
    if frinfo["relevance"] != "call":
        label += ":"
    return label, label_plain


def _get_frame_info(e: dict[str, Any], frinfo: dict[str, Any]) -> dict[str, Any]:
    """Gather all info needed to print a frame."""
    label, label_plain = _get_frame_label(frinfo)
    fragments = frinfo.get("fragments", [])
    frame_range = frinfo.get("range")
    relevance = frinfo.get("relevance", "call")
    is_deepest = frinfo is e["frames"][-1]

    # Get marked lines (lines with any mark attribute)
    marked_lines = [
        li for li in fragments if any(f.get("mark") for f in li["fragments"])
    ]

    return {
        "label": label,
        "label_plain": label_plain,
        "fragments": fragments,
        "frame_range": frame_range,
        "relevance": relevance,
        "is_deepest": is_deepest,
        "marked_lines": marked_lines,
        "frinfo": frinfo,
        "e": e,
    }


def _build_frame_lines(
    info: dict[str, Any], label_width: int, term_width: int
) -> list[tuple[str, int, bool]]:
    """Build output lines for a frame. Returns list of (colored_line, plain_length, is_marked)."""
    label = info["label"]
    label_plain = info["label_plain"]
    fragments = info["fragments"]
    frame_range = info["frame_range"]
    relevance = info["relevance"]
    is_deepest = info["is_deepest"]
    frinfo = info["frinfo"]
    e = info["e"]

    lines = []

    if not fragments:
        if is_deepest:
            msg = f"Source code not available but {e['type']} was raised from here"
        else:
            msg = "Source code not available"
        line = f"{INDENT}{label}  {INS_TYPE}{msg}{RST}"
        plain_len = len(INDENT) + len(label_plain) + 2 + len(msg)
        lines.append((line, plain_len, False))
        return lines

    start = frinfo["linenostart"]
    symbol = symbols.get(relevance, "")
    symbol_colored = f"{EM_CALL}{symbol}{RST}" if symbol else ""
    desc = symdesc[relevance].format(**e, **frinfo)

    if relevance == "call":
        # One-liner for call frames: label + marked region only + symbol
        padding = " " * (label_width - len(label_plain))

        if info["marked_lines"]:
            line_info = info["marked_lines"][0]
            # Extract only marked fragments, with em in red, rest in default color
            code_parts = []
            code_plain = ""
            for fragment in line_info["fragments"]:
                mark = fragment.get("mark")
                if mark:  # Only include marked fragments
                    colored, plain = _format_fragment_call(fragment)
                    code_parts.append(colored)
                    code_plain += plain
            code_colored = "".join(code_parts)

            # Check if it fits
            full_plain_len = (
                len(INDENT)
                + len(label_plain)
                + len(padding)
                + 1
                + len(code_plain)
                + 2
                + len(symbol)
            )
            if full_plain_len <= term_width:
                line = f"{INDENT}{label}{padding} {code_colored} {symbol_colored}"
                lines.append((line, full_plain_len, False))
            else:
                # Doesn't fit, just print symbol
                line = f"{INDENT}{label}{padding} {symbol_colored}"
                lines.append(
                    (
                        line,
                        len(INDENT) + len(label_plain) + len(padding) + 1 + len(symbol),
                        False,
                    )
                )
        else:
            # No marked lines, just label and symbol
            line = f"{INDENT}{label}{padding} {symbol_colored}"
            lines.append(
                (
                    line,
                    len(INDENT) + len(label_plain) + len(padding) + 1 + len(symbol),
                    False,
                )
            )
    else:
        # Full format for non-call frames
        lines.append((f"{INDENT}{label}", len(INDENT) + len(label_plain), False))

        # Track which lines are marked
        marked_line_nums = set()
        for ml in info["marked_lines"]:
            marked_line_nums.add(ml["line"])

        for line_info in fragments:
            line_num = line_info["line"]
            abs_line = start + line_num - 1
            line_fragments = line_info["fragments"]
            is_marked = line_num in marked_line_nums

            code_parts = []
            code_plain = ""
            for fragment in line_fragments:
                colored, plain = _format_fragment(fragment)
                code_parts.append(colored)
                code_plain += plain
            code_colored = "".join(code_parts)

            # Add symbol and desc on final line
            if frame_range and abs_line == frame_range.lfinal and symbol:
                line = f"{CODE_INDENT}{code_colored} {symbol_colored}  {SYMBOLDESC}{desc}{RST}"
                plain_len = (
                    len(CODE_INDENT) + len(code_plain) + 1 + len(symbol) + 2 + len(desc)
                )
            else:
                line = f"{CODE_INDENT}{code_colored}"
                plain_len = len(CODE_INDENT) + len(code_plain)

            lines.append((line, plain_len, is_marked))

    return lines


def _format_fragment(fragment: dict[str, Any]) -> tuple[str, str]:
    """Format a fragment returning (colored_string, plain_string)."""
    code = fragment["code"].rstrip("\n\r")
    mark = fragment.get("mark")
    em = fragment.get("em")

    colored_parts = []

    # Open mark if starting
    if mark in ("solo", "beg"):
        colored_parts.append(MARK_BG + MARK_TEXT)

    # Open em if starting (red text within the mark)
    if em in ("solo", "beg"):
        colored_parts.append(EM)

    # Add the code
    colored_parts.append(code)

    # Close em if ending
    if em in ("fin", "solo") and mark not in ("fin", "solo"):
        colored_parts.append(MARK_TEXT)

    # Close mark if ending
    if mark in ("fin", "solo"):
        colored_parts.append(RST)

    return "".join(colored_parts), code


def _format_fragment_call(fragment: dict[str, Any]) -> tuple[str, str]:
    """Format a fragment for call frames: default color, only em in yellow."""
    code = fragment["code"].rstrip("\n\r")
    em = fragment.get("em")

    colored_parts = []

    # Open em if starting (yellow text)
    if em in ("solo", "beg"):
        colored_parts.append(EM_CALL)

    # Add the code
    colored_parts.append(code)

    # Close em if ending
    if em in ("fin", "solo"):
        colored_parts.append(RST)

    return "".join(colored_parts), code


def _print_fragment(file: TextIO, fragment: dict[str, Any]) -> None:
    """Print a single fragment with appropriate ANSI styling.

    Follows the same nesting order as html.py:
    1. Open mark (yellow background)
    2. Open em (red text) inside mark
    3. Print code
    4. Close em
    5. Close mark
    """
    code = fragment["code"].rstrip("\n\r")
    mark = fragment.get("mark")
    em = fragment.get("em")

    # Open mark if starting
    if mark in ("solo", "beg"):
        print(MARK_BG + MARK_TEXT, end="", file=file)

    # Open em if starting (red text within the mark)
    if em in ("solo", "beg"):
        print(EM, end="", file=file)

    # Print the code
    print(code, end="", file=file)

    # Close em if ending
    if em in ("fin", "solo") and mark not in ("fin", "solo"):
        print(MARK_TEXT, end="", file=file)

    # Close mark if ending
    if mark in ("fin", "solo"):
        print(RST, end="", file=file)


def _build_variable_inspector(
    variables: list[Any], term_width: int
) -> list[tuple[str, int]]:
    """Build variable inspector lines. Returns list of (colored_line, width).

    Uses simple left-side vertical bar only, no top/bottom borders.
    """
    if not variables:
        return []

    # Build variable lines: "name: type = value" or "name = value"
    var_lines = []
    for var_info in variables:
        # Handle both old tuple format and new VarInfo namedtuple
        if hasattr(var_info, "name"):
            name, typename, value, _fmt = (
                var_info.name,
                var_info.typename,
                var_info.value,
                var_info.format_hint,
            )
        else:
            name, typename, value = var_info

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

        # Build the line
        if typename:
            line = f"{VAR}{name}{RST}{DIM}: {typename}{RST} = {val_str}"
            line_plain = f"{name}: {typename} = {val_str}"
        else:
            line = f"{VAR}{name}{RST} = {val_str}"
            line_plain = f"{name} = {val_str}"

        var_lines.append((line, line_plain))

    # Calculate width: just "| content" (bar + space + content)
    max_content_width = max(len(lp) for _, lp in var_lines)
    max_width = min(
        max_content_width, term_width // 2 - 4
    )  # Leave space for arrow "<-"

    result = []

    # Variable lines (bar added during printing to handle arrow line differently)
    for line, line_plain in var_lines:
        # Truncate if too long
        if len(line_plain) > max_width:
            truncated_plain = line_plain[: max_width - 1] + "…"
            line = f"{VAR}{line_plain[: max_width - 1]}…{RST}"
            line_plain = truncated_plain

        # Just content, bar added during printing
        result.append((line, len(line_plain)))

    return result

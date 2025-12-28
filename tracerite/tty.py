from __future__ import annotations

import logging
import os
import sys
import textwrap
import threading
from pathlib import Path
from typing import Any, TextIO

from .chain_analysis import build_chronological_frames
from .trace import build_chain_header, chainmsg, extract_chain, symbols, symdesc

# ANSI escape codes for terminal colors (can be monkeypatched for styling)
ESC = "\x1b["
RESET = f"{ESC}0m"
DIM = f"{ESC}2m"
LINE_PREFIX_TOP = f"{DIM}╭{RESET} "  # Dim rounded top-left corner for first line
LINE_PREFIX = f"{DIM}│{RESET} "  # Dim vertical line prefix for middle lines
LINE_PREFIX_BOT = f"{DIM}╰{RESET} "  # Dim rounded bottom-left corner for last line
EOL = f"\n{LINE_PREFIX}"  # End of line: newline, add prefix
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
                tty_traceback(exc=exc_info[1], file=self.stream, msg=msg)
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

    Args:
    """
    import re

    chain = chain or extract_chain(exc=exc, **extract_args)
    # Chain is already oldest-first from extract_chain

    # Build header message if not provided
    if msg is None and chain:
        msg = build_chain_header(chain)

    if file is None:
        file = sys.stderr

    is_tty = file.isatty() if hasattr(file, "isatty") else False
    no_color = not is_tty
    no_inspector = not is_tty

    # Start with rounded top corner
    output = LINE_PREFIX_TOP

    # Print the original log message if provided
    if msg:
        # Strip trailing newlines and left-trim two spaces if present (to align with prefix)
        msg = msg.rstrip("\n")
        if msg.startswith("  "):
            msg = msg[2:]
        output += msg.replace("\n", EOL) + EOL

    if term_width is None:
        try:
            term_width = os.get_terminal_size(file.fileno()).columns
        except (OSError, ValueError):
            term_width = 80

    output += _print_chronological(chain, term_width, no_inspector)

    # Strip trailing EOL (which ends with LINE_PREFIX for an empty line we don't want)
    eol_suffix = f"\n{LINE_PREFIX}"
    if output.endswith(eol_suffix):
        output = output[: -len(eol_suffix)]

    # Replace the last line prefix with bottom corner and reset to original terminal colors
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
        output = re.sub(r"\x1b\[[0-9;]*[A-Za-z]", "", output)

    file.write(output)


def _find_inspector_frame_idx(
    frame_info_list: list[dict[str, Any]],
) -> int | None:
    """Find the first non-call frame that has variables to show.

    Returns the frame index, or None if no suitable frame is found.
    Variables are deduplicated in trace.py, so we just check if any exist.
    """
    for i, info in enumerate(frame_info_list):
        if info["relevance"] != "call" and info["frinfo"].get("variables"):
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


def _print_chronological(
    chain: list[dict[str, Any]],
    term_width: int,
    no_inspector: bool = False,
) -> str:
    """Print frames in chronological order with exception info after error frames."""
    output = ""
    chrono_frames = build_chronological_frames(chain)
    if not chrono_frames:
        # No frames, but still show exception banners for any exceptions in chain
        for exc in chain:
            exc_info = {
                "type": exc.get("type"),
                "message": exc.get("message"),
                "summary": exc.get("summary"),
                "from": exc.get("from"),
            }
            output += _build_exception_banner(exc_info, term_width)
        return output

    # Build frame info list for all chronological frames
    frame_info_list = []
    for frinfo in chrono_frames:
        info = _get_chrono_frame_info(frinfo)
        frame_info_list.append(info)

    # Find collapsible call runs
    collapse_ranges = _find_collapsible_call_runs(frame_info_list, min_run_length=10)

    # Build set of frame indices to skip
    skip_indices = set()
    ellipsis_after = {}
    for start_idx, end_idx in collapse_ranges:
        skipped_count = end_idx - start_idx - 1
        for i in range(start_idx + 1, end_idx):
            skip_indices.add(i)
        ellipsis_after[start_idx] = skipped_count

    # Find first non-call frame with variables for inspector
    inspector_frame_idx = _find_inspector_frame_idx(frame_info_list)

    # Calculate max label width for call frame alignment
    call_label_widths = [
        len(info["label_plain"])
        for i, info in enumerate(frame_info_list)
        if info["relevance"] == "call" and i not in skip_indices
    ]
    label_width = max(call_label_widths, default=0)

    # Build output lines
    output_lines = []
    exception_banners = []  # List of (insert_after_line_idx, banner_output)

    for i, info in enumerate(frame_info_list):
        if i in skip_indices:
            continue

        lines = _build_chrono_frame_lines(info, label_width, term_width)
        len(output_lines)
        for line, plain_len, is_marked in lines:
            output_lines.append((line, plain_len, i, is_marked))

        # Add ellipsis after first frame of collapsed run
        if i in ellipsis_after:
            skipped = ellipsis_after[i]
            ellipsis_line = f"{INDENT}{ELLIPSIS}⋮ {skipped} more calls{RESET}"
            ellipsis_plain_len = len(INDENT) + 2 + len(f"{skipped} more calls")
            output_lines.append((ellipsis_line, ellipsis_plain_len, i, False))

        # Check if this frame has exception info to print after it
        exc_info = info["frinfo"].get("exception")
        info["relevance"]
        if exc_info:
            # Record that we need to insert exception banner after this frame's lines
            banner = _build_exception_banner(exc_info, term_width)
            exception_banners.append((len(output_lines), banner))

    # Get variable inspector lines (inspector_frame_idx is only set for frames with variables)
    inspector_lines = []
    if not no_inspector and inspector_frame_idx is not None:
        frinfo = frame_info_list[inspector_frame_idx]["frinfo"]
        variables = frinfo.get("variables", [])
        inspector_lines = _build_variable_inspector(variables, term_width)

    # Build final output, inserting exception banners at the right positions
    if inspector_lines:
        # Complex case: merge inspector and banners
        output += _merge_chrono_output(
            output_lines,
            inspector_lines,
            term_width,
            inspector_frame_idx,
            exception_banners,
        )
    else:
        # Simpler case: just insert banners
        banner_idx = 0
        for li, (line, _, _, _) in enumerate(output_lines):
            output += f"{line}{EOL}"
            # Check if we need to insert a banner after this line
            while banner_idx < len(exception_banners):
                insert_pos, banner = exception_banners[banner_idx]
                if li + 1 == insert_pos:
                    output += banner
                    banner_idx += 1
                else:
                    break
        # Any remaining banners (when banner position > last output line)
        for _, banner in exception_banners[banner_idx:]:  # pragma: no cover
            output += banner

    return output


def _build_exception_banner(exc_info: dict[str, Any], term_width: int) -> str:
    """Build exception banner output to show after error frame."""
    output = ""
    exc_type = exc_info.get("type", "Exception")
    summary = exc_info.get("summary", "")
    message = exc_info.get("message", "")
    from_type = exc_info.get("from", "none")

    chain_suffix = chainmsg.get(from_type, "")
    type_prefix = f"{exc_type}{chain_suffix}: "
    type_prefix_len = len(type_prefix)
    cont_prefix = f"{EXC}{BOX_V}{RESET} "
    cont_prefix_len = 2

    # Check if the full title fits on one line
    full_title_len = type_prefix_len + len(summary)
    if full_title_len <= term_width:
        output += f"{EXC}{type_prefix}{RESET}{BOLD}{summary}{RESET}{EOL}"
    elif len(summary) <= term_width - cont_prefix_len:
        output += f"{EXC}{type_prefix}{RESET}{EOL}"
        output += f"{cont_prefix}{BOLD}{summary}{RESET}{EOL}"
    else:
        padding = "\x00" * (type_prefix_len - cont_prefix_len)
        wrapped = textwrap.wrap(
            padding + summary,
            width=term_width - cont_prefix_len,
            break_long_words=False,
            break_on_hyphens=False,
        )
        wrapped[0] = wrapped[0].lstrip("\x00")
        for i, line in enumerate(wrapped):
            if i == 0:
                output += f"{EXC}{type_prefix}{RESET}{BOLD}{line}{RESET}{EOL}"
            else:
                output += f"{cont_prefix}{BOLD}{line}{RESET}{EOL}"

    if summary != message:
        if message.startswith(summary):
            message = message[len(summary) :].strip("\n")
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
                output += f"{cont_prefix.rstrip()}{EOL}"

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
        except (ValueError, OSError):
            pass

    # Build label with colors: light blue function, green filename, dark grey :lineno
    label = ""
    label_plain = ""
    if frinfo["function"]:
        function_display = f"{frinfo['function']}{frinfo.get('function_suffix', '')}"
        label_plain += f"{function_display} "
        label += f"{FUNC}{function_display} {RESET}"
    label_plain += f"{location}:{lineno}"
    label += f"{LOCFN}{location}{LOC_LINENO}:{lineno}{RESET}"
    if frinfo["relevance"] != "call":
        label += ":"
    return label, label_plain


def _get_chrono_frame_info(frinfo: dict[str, Any]) -> dict[str, Any]:
    """Gather info needed to print a chronological frame."""
    label, label_plain = _get_frame_label(frinfo)
    fragments = frinfo.get("fragments", [])
    frame_range = frinfo.get("range")
    relevance = frinfo.get("relevance", "call")
    exc_info = frinfo.get("exception")

    # Get marked lines
    marked_lines = [
        li for li in fragments if any(f.get("mark") for f in li.get("fragments", []))
    ]

    return {
        "label": label,
        "label_plain": label_plain,
        "fragments": fragments,
        "frame_range": frame_range,
        "relevance": relevance,
        "exc_info": exc_info,
        "marked_lines": marked_lines,
        "frinfo": frinfo,
    }


def _build_chrono_frame_lines(
    info: dict[str, Any], label_width: int, term_width: int
) -> list[tuple[str, int, bool]]:
    """Build output lines for a chronological frame."""
    label = info["label"]
    label_plain = info["label_plain"]
    fragments = info["fragments"]
    frame_range = info["frame_range"]
    relevance = info["relevance"]
    exc_info = info["exc_info"]
    frinfo = info["frinfo"]

    lines = []

    if not fragments:
        if exc_info:
            msg = f"Source code not available but {exc_info.get('type', 'Exception')} was raised from here"
        else:
            msg = "Source code not available"
        line = f"{INDENT}{label}  {INS_TYPE}{msg}{RESET}"
        plain_len = len(INDENT) + len(label_plain) + 2 + len(msg)
        lines.append((line, plain_len, False))
        return lines

    start = frinfo.get("linenostart", 1)
    symbol = symbols.get(relevance, "")
    symbol_colored = f"{EM_CALL}{symbol}{RESET}" if symbol else ""

    desc = symdesc[relevance]

    if relevance == "call":
        # One-liner for call frames
        padding = " " * (label_width - len(label_plain))

        if info["marked_lines"]:
            line_info = info["marked_lines"][0]
            code_parts = []
            code_plain = ""
            for fragment in line_info.get("fragments", []):
                mark = fragment.get("mark")
                if mark:
                    colored, plain = _format_fragment_call(fragment)
                    code_parts.append(colored)
                    code_plain += plain
            code_colored = "".join(code_parts)

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
                line = f"{INDENT}{label}{padding} {symbol_colored}"
                lines.append(
                    (
                        line,
                        len(INDENT) + len(label_plain) + len(padding) + 1 + len(symbol),
                        False,
                    )
                )
        else:
            line = f"{INDENT}{label}{padding} {symbol_colored}"
            lines.append(
                (
                    line,
                    len(INDENT) + len(label_plain) + len(padding) + 1 + len(symbol),
                    False,
                )
            )
    else:
        # Full format for error/warning/stop/except frames
        lines.append((f"{INDENT}{label}", len(INDENT) + len(label_plain), False))

        marked_line_nums = set()
        for ml in info["marked_lines"]:
            marked_line_nums.add(ml["line"])

        for line_info in fragments:
            line_num = line_info["line"]
            abs_line = start + line_num - 1
            line_fragments = line_info.get("fragments", [])
            is_marked = line_num in marked_line_nums

            code_parts = []
            code_plain = ""
            for fragment in line_fragments:
                colored, plain = _format_fragment(fragment)
                code_parts.append(colored)
                code_plain += plain
            code_colored = "".join(code_parts)

            if frame_range and abs_line == frame_range.lfinal and symbol:
                line = f"{CODE_INDENT}{code_colored} {symbol_colored} {SYMBOLDESC}{desc}{RESET}"
                plain_len = (
                    len(CODE_INDENT) + len(code_plain) + 1 + len(symbol) + 2 + len(desc)
                )
            else:
                line = f"{CODE_INDENT}{code_colored}"
                plain_len = len(CODE_INDENT) + len(code_plain)

            lines.append((line, plain_len, is_marked))

    return lines


def _merge_chrono_output(
    output_lines: list[tuple[str, int, int, bool]],
    inspector_lines: list[tuple[str, int]],
    term_width: int,
    inspector_frame_idx: int | None,
    exception_banners: list[tuple[int, str]],
) -> str:
    """Merge chronological output with inspector and exception banners."""
    assert inspector_frame_idx is not None
    output = ""
    frame_line_start, frame_line_end = _find_frame_line_range(
        output_lines, inspector_frame_idx
    )
    error_line = _find_last_marked_line(output_lines, frame_line_start, frame_line_end)
    inspector_height = len(inspector_lines)

    ideal_arrow_pos = inspector_height // 2
    ideal_start = error_line - ideal_arrow_pos
    min_start = 0
    inspector_start = ideal_start

    if inspector_start < min_start:
        inspector_start = min_start

    lines_available_below = len(output_lines) - inspector_start
    if lines_available_below < inspector_height:
        needed_shift = inspector_height - lines_available_below
        inspector_start = max(min_start, inspector_start - needed_shift)

    arrow_line_idx = error_line - inspector_start
    assert 0 <= arrow_line_idx < inspector_height

    max_line_len = 0
    for li in range(
        inspector_start,
        min(inspector_start + inspector_height, len(output_lines)),
    ):
        max_line_len = max(max_line_len, output_lines[li][1])

    max_insp_width = max(w for _, w in inspector_lines) if inspector_lines else 0
    total_insp_width = 4 + max_insp_width
    inspector_col = max_line_len + 2

    if inspector_col + total_insp_width > term_width:
        inspector_col = term_width - total_insp_width

    inspector_count = len(inspector_lines)
    banner_idx = 0

    for li, (line, *_) in enumerate(output_lines):
        insp_idx = li - inspector_start
        if 0 <= insp_idx < inspector_count:
            insp_line, insp_width = inspector_lines[insp_idx]
            cursor_pos = f"{ESC}{inspector_col + 1}G"
            is_first = insp_idx == 0
            is_last = insp_idx == inspector_count - 1
            is_arrow = insp_idx == arrow_line_idx

            if is_arrow:
                if is_first and is_last:
                    box_char = SINGLE_T
                elif is_first:  # pragma: no cover
                    box_char = BOX_TR
                elif is_last:
                    box_char = BOX_BR
                else:
                    box_char = BOX_VL
                output += f"{line}{cursor_pos}{DIM}{ARROW_LEFT}{BOX_H}{box_char}{RESET} {insp_line}{EOL}"
            else:
                if is_first:
                    box_char = BOX_TL
                elif is_last:  # pragma: no cover
                    box_char = BOX_BL
                else:
                    box_char = BOX_V
                output += f"{line}{cursor_pos}  {DIM}{box_char}{RESET} {insp_line}{EOL}"
        else:
            output += f"{line}{EOL}"

        # Insert exception banner if needed
        while banner_idx < len(exception_banners):
            insert_pos, banner = exception_banners[banner_idx]
            if li + 1 == insert_pos:
                output += banner
                banner_idx += 1
            else:
                break

    # Remaining inspector lines
    remaining_start = len(output_lines) - inspector_start
    if remaining_start < inspector_count:
        for idx in range(remaining_start, inspector_count):
            insp_line, insp_width = inspector_lines[idx]
            cursor_pos = f"{ESC}{inspector_col + 1}G"
            is_last = idx == inspector_count - 1
            box_char = BOX_BL if is_last else BOX_V
            output += f"{cursor_pos}  {DIM}{box_char}{RESET} {insp_line}{EOL}"

    # Any remaining banners
    for _, banner in exception_banners[banner_idx:]:
        output += banner

    return output


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
        colored_parts.append(RESET)

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
        colored_parts.append(RESET)

    return "".join(colored_parts), code


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
            line = f"{VAR}{name}{RESET}{DIM}: {typename}{RESET} = {val_str}"
            line_plain = f"{name}: {typename} = {val_str}"
        else:
            line = f"{VAR}{name}{RESET} = {val_str}"
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
            line = f"{VAR}{line_plain[: max_width - 1]}…{RESET}"
            line_plain = truncated_plain

        # Just content, bar added during printing
        result.append((line, len(line_plain)))

    return result

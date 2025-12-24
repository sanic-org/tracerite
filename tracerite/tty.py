from __future__ import annotations

import os
import sys
import threading
from pathlib import Path

from .trace import chainmsg, extract_chain

# ANSI escape codes for terminal colors
RESET = "\033[0m"
YELLOW_BG = "\033[43m"
BLACK_TEXT = "\033[30m"
RED_TEXT = "\033[31m"
GREEN = "\033[32m"
YELLOW = "\033[93m"  # Bright yellow
DARK_GREY = "\033[90m"
LIGHT_BLUE = "\033[94m"
CYAN = "\033[36m"
BOLD = "\033[1m"
DIM = "\033[2m"

# Box drawing characters
BOX_TL = "┌"
BOX_TR = "┐"
BOX_BL = "└"
BOX_BR = "┘"
BOX_H = "─"
BOX_V = "│"
BOX_VL = "┤"  # Vertical with left branch
BOX_ROUND_TL = "╭"  # Rounded top-left (curves down-right)
BOX_ROUND_BL = "╰"  # Rounded bottom-left (curves up-right)
BOX_ROUND_TR = "╮"  # Rounded top-right (curves down-left, for arrow on first line)
BOX_ROUND_BR = "╯"  # Rounded bottom-right (curves up-left, for arrow on last line)
ARROW_LEFT = "◀"

INDENT = "  "
CODE_INDENT = "    "  # Double indent for code lines

symbols = {"call": "➤", "warning": "⚠️", "error": "💣", "stop": "🛑"}

# Store the original hooks for uninstall
_original_excepthook = None
_original_threading_excepthook = None


def install():
    """Install TraceRite as the default exception handler.

    Replaces sys.excepthook to use TraceRite's pretty TTY formatting
    for all unhandled exceptions, including those in threads.
    Call uninstall() to restore the original exception handlers.

    Usage:
        import tracerite
        tracerite.install()
    """
    global _original_excepthook, _original_threading_excepthook

    if _original_excepthook is None:
        _original_excepthook = sys.excepthook

    if _original_threading_excepthook is None:
        _original_threading_excepthook = threading.excepthook

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

    sys.excepthook = _tracerite_excepthook
    threading.excepthook = _tracerite_threading_excepthook


def uninstall():
    """Restore the original exception handlers.

    Removes TraceRite from sys.excepthook and threading.excepthook,
    restoring the previous handlers.
    """
    global _original_excepthook, _original_threading_excepthook

    if _original_excepthook is not None:
        sys.excepthook = _original_excepthook
        _original_excepthook = None

    if _original_threading_excepthook is not None:
        threading.excepthook = _original_threading_excepthook
        _original_threading_excepthook = None


def tty_traceback(exc=None, chain=None, *, file=None, **extract_args):
    """Format and print a traceback for terminal output (TTY).

    Outputs directly to the terminal (or specified file) to adapt to
    terminal features like window size. The chain is printed with the
    oldest exception first (order they occurred).
    """
    chain = chain or extract_chain(exc=exc, **extract_args)
    # Chain is already oldest-first from extract_chain

    if file is None:
        file = sys.stderr

    # Get terminal width for potential future use
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
            print(file=file)  # Empty line between exceptions
            chain_suffix = chainmsg.get(e.get("from", "none"), "")

        _print_exception(file, e, term_width, i, inspector_allowed, chain_suffix)


def _find_inspector_frame_idx(
    frame_info_list: list, exception_idx: int, inspector_allowed: set | None
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
    output_lines: list, inspector_frame_idx: int
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
    output_lines: list, frame_line_start: int, frame_line_end: int
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


def _print_exception(
    file, e, term_width, exception_idx=0, inspector_allowed=None, chain_suffix=""
):
    """Print a single exception with its frames."""
    # Exception header (not indented)
    summary, message = e["summary"], e["message"]
    print(
        f"{DARK_GREY}{e['type']}{chain_suffix}: {RESET}{BOLD}{summary}{RESET}",
        file=file,
    )
    if summary != message:
        if message.startswith(summary):
            message = message[len(summary) :]
        print(message, file=file)

    # Frames: caller first, then callee (deepest last)
    frames = e["frames"]

    # Pre-calculate frame info for alignment
    frame_info_list = []
    for frinfo in frames:
        info = _get_frame_info(e, frinfo)
        frame_info_list.append(info)

    # Find first non-call frame that is allowed to show inspector
    inspector_frame_idx = _find_inspector_frame_idx(
        frame_info_list, exception_idx, inspector_allowed
    )

    # Calculate max label width for call frame alignment
    # Only consider call frames since non-call frames display differently
    call_label_widths = [
        len(info["label_plain"])
        for info in frame_info_list
        if info["relevance"] == "call"
    ]
    label_width = max(call_label_widths, default=0)

    # Build output lines with their plain text lengths
    output_lines = []  # List of (colored_line, plain_length, frame_idx, is_marked)
    for i, info in enumerate(frame_info_list):
        lines = _build_frame_lines(info, label_width, term_width)
        for line, plain_len, is_marked in lines:
            output_lines.append((line, plain_len, i, is_marked))

    # Get variable inspector lines if we have a non-call frame
    inspector_lines = []
    if inspector_frame_idx is not None:
        frinfo = frame_info_list[inspector_frame_idx]["frinfo"]
        variables = frinfo.get("variables", [])
        if variables:
            inspector_lines = _build_variable_inspector(variables, term_width)

    # Merge output with inspector
    if inspector_lines:
        frame_line_start, frame_line_end = _find_frame_line_range(
            output_lines, inspector_frame_idx
        )
        error_line = _find_last_marked_line(
            output_lines, frame_line_start, frame_line_end
        )
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
        total_insp_width = 4 + max_insp_width  # "◀─┤ " or "  │ " + content

        # Place inspector right after the longest line, with some padding
        inspector_col = max_line_len + 2

        # But don't go beyond terminal width
        if inspector_col + total_insp_width > term_width:
            inspector_col = term_width - total_insp_width

        # Print output with inspector merged using cursor positioning
        inspector_count = len(inspector_lines)
        for li, (line, *_) in enumerate(output_lines):
            insp_idx = li - inspector_start
            if 0 <= insp_idx < inspector_count:
                insp_line, insp_width = inspector_lines[insp_idx]
                # Use cursor positioning to place inspector
                cursor_pos = (
                    f"\033[{inspector_col + 1}G"  # +1 because columns are 1-indexed
                )
                # Determine which box character to use
                is_first = insp_idx == 0
                is_last = insp_idx == inspector_count - 1
                is_arrow = insp_idx == arrow_line_idx

                if is_arrow:
                    # Arrow line: use appropriate corner or T-junction
                    if is_first:
                        box_char = BOX_ROUND_TR  # ╮ curved corner for first+arrow
                    elif is_last:
                        box_char = BOX_ROUND_BR  # ╯ curved corner for last+arrow
                    else:
                        box_char = BOX_VL  # ┤ T-junction for middle arrow
                    print(
                        f"{line}{cursor_pos}{DIM}{ARROW_LEFT}{BOX_H}{box_char}{RESET} {insp_line}",
                        file=file,
                    )
                else:
                    # Non-arrow line: use corner or vertical
                    if is_first:
                        box_char = BOX_ROUND_TL  # ╭ curved corner for first
                    elif is_last:
                        box_char = BOX_ROUND_BL  # ╰ curved corner for last
                    else:
                        box_char = BOX_V  # │ vertical for middle
                    print(
                        f"{line}{cursor_pos}  {DIM}{box_char}{RESET} {insp_line}",
                        file=file,
                    )
            else:
                print(line, file=file)

        # If inspector is taller than available lines, print remaining
        remaining_start = len(output_lines) - inspector_start
        if remaining_start < inspector_count:
            for idx in range(remaining_start, inspector_count):
                insp_line, insp_width = inspector_lines[idx]
                cursor_pos = f"\033[{inspector_col + 1}G"
                is_last = idx == inspector_count - 1
                box_char = BOX_ROUND_BL if is_last else BOX_V
                print(
                    f"{cursor_pos}  {DIM}{box_char}{RESET} {insp_line}",
                    file=file,
                )
        return

    # No inspector or no frame lines found, just print the code
    for line, _, _, _ in output_lines:
        print(line, file=file)


def _get_frame_label(frinfo):
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
        label += f"{LIGHT_BLUE}{frinfo['function']} {RESET}"
    label_plain += f"{location}:{lineno}"
    label += f"{GREEN}{location}{DARK_GREY}:{lineno}{RESET}"

    return label, label_plain


def _get_frame_info(e, frinfo):
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


def _build_frame_lines(info, label_width, term_width):
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
        lines.append((f"{INDENT}{label}", len(INDENT) + len(label_plain), False))
        lines.append(
            (f"{CODE_INDENT}Source code not available", len(CODE_INDENT) + 24, False)
        )
        if is_deepest:
            msg = f"but {e['type']} was raised from here"
            lines.append((f"{CODE_INDENT}{msg}", len(CODE_INDENT) + len(msg), False))
        return lines

    start = frinfo["linenostart"]
    symbol = symbols.get(relevance, "")
    symbol_colored = f"{YELLOW}{symbol}{RESET}" if symbol else ""

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

            # Add symbol on final line
            if frame_range and abs_line == frame_range.lfinal and symbol:
                line = f"{CODE_INDENT}{code_colored} {symbol_colored}"
                plain_len = len(CODE_INDENT) + len(code_plain) + 1 + len(symbol)
            else:
                line = f"{CODE_INDENT}{code_colored}"
                plain_len = len(CODE_INDENT) + len(code_plain)

            lines.append((line, plain_len, is_marked))

    return lines


def _format_fragment(fragment):
    """Format a fragment returning (colored_string, plain_string)."""
    code = fragment["code"].rstrip("\n\r")
    mark = fragment.get("mark")
    em = fragment.get("em")

    colored_parts = []

    # Open mark if starting
    if mark in ("solo", "beg"):
        colored_parts.append(YELLOW_BG + BLACK_TEXT)

    # Open em if starting (red text within the mark)
    if em in ("solo", "beg"):
        colored_parts.append(RED_TEXT)

    # Add the code
    colored_parts.append(code)

    # Close em if ending
    if em in ("fin", "solo") and mark not in ("fin", "solo"):
        colored_parts.append(BLACK_TEXT)

    # Close mark if ending
    if mark in ("fin", "solo"):
        colored_parts.append(RESET)

    return "".join(colored_parts), code


def _format_fragment_call(fragment):
    """Format a fragment for call frames: default color, only em in red."""
    code = fragment["code"].rstrip("\n\r")
    em = fragment.get("em")

    colored_parts = []

    # Open em if starting (red text)
    if em in ("solo", "beg"):
        colored_parts.append(RED_TEXT)

    # Add the code
    colored_parts.append(code)

    # Close em if ending
    if em in ("fin", "solo"):
        colored_parts.append(RESET)

    return "".join(colored_parts), code


def _print_fragment(file, fragment):
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
        print(YELLOW_BG + BLACK_TEXT, end="", file=file)

    # Open em if starting (red text within the mark)
    if em in ("solo", "beg"):
        print(RED_TEXT, end="", file=file)

    # Print the code
    print(code, end="", file=file)

    # Close em if ending
    if em in ("fin", "solo") and mark not in ("fin", "solo"):
        print(BLACK_TEXT, end="", file=file)

    # Close mark if ending
    if mark in ("fin", "solo"):
        print(RESET, end="", file=file)


def _build_variable_inspector(variables, term_width):
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
            line = f"{CYAN}{name}{RESET}{DIM}: {typename}{RESET} = {val_str}"
            line_plain = f"{name}: {typename} = {val_str}"
        else:
            line = f"{CYAN}{name}{RESET} = {val_str}"
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
            line = f"{CYAN}{line_plain[: max_width - 1]}…{RESET}"
            line_plain = truncated_plain

        # Just content, bar added during printing
        result.append((line, len(line_plain)))

    return result

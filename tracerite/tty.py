from __future__ import annotations

import os
import re
import sys
import textwrap
import unicodedata
import warnings
from pathlib import Path
from typing import Any, TextIO

from .chain_analysis import build_chronological_frames
from .trace import build_chain_header, chainmsg, extract_chain, symbols, symdesc

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
MARK_BG = f"{ESC}103m"  # Bright yellow background
MARK_TEXT = f"{ESC}30m"
EM = f"{ESC}31m"
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
BLOCK_HALF = "▐"  # Right half block, continuation marker for multi-line messages

INDENT = ""  # No indent for function/location lines
CODE_INDENT = "  "  # Indent for code in frame

# Regex pattern to strip ANSI escape sequences
ANSI_ESCAPE_RE = re.compile(r"\x1b\[[0-9;]*[A-Za-z]")


def _display_width(s: str) -> int:
    """Calculate the display width of a string in terminal columns."""
    plain = ANSI_ESCAPE_RE.sub("", s)
    return sum(2 if unicodedata.east_asian_width(c) in "WF" else 1 for c in plain)


def tty_traceback(
    exc: BaseException | None = None,
    chain: list[dict[str, Any]] | None = None,
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
        chain: Pre-extracted exception chain. If provided, exc is ignored.
        file: Output file. Defaults to sys.stderr.
        msg: Header message. If None, builds from exception chain.
        tag: Optional tag to display after the message (e.g., "#TR1").
        term_width: Terminal width. Auto-detected if None.
        **extract_args: Additional arguments passed to extract_chain().
    """
    chain = chain or extract_chain(exc=exc, **extract_args)

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

    # Print the original log message if provided, with optional tag at the end
    if msg:
        # Strip trailing newlines and left-trim two spaces if present (to align with prefix)
        msg = msg.rstrip("\n")
        if msg.startswith("  "):
            msg = msg[2:]

        # Append tag (dim color) if provided
        if tag:
            msg = f"{msg} {DIM}{tag}{RESET}"

        output += msg + EOL

    if term_width is None:
        try:
            term_width = os.get_terminal_size(file.fileno()).columns
        except (OSError, ValueError):
            term_width = 80

        # If the terminal is reported as very narrow, assume it is temporary and
        # will be expanded back to a normal width.  Widths from 40 up are still
        # honoured so users can deliberately narrow the window.
        if term_width < 40:
            term_width = 80

    chrono_output, last_banner_start = _print_chronological(
        chain, term_width, no_inspector
    )
    if last_banner_start is not None:
        last_banner_start += len(output)
    output += chrono_output

    # Strip trailing EOL (which ends with LINE_PREFIX for an empty line we don't want)
    eol_suffix = f"\n{LINE_PREFIX}"
    if output.endswith(eol_suffix):
        output = output[: -len(eol_suffix)]

    # Side bracket termination.  If the last emitted item is an exception
    # banner, the curved ending goes at the banner's first line and any
    # continuation lines hang without the left border.
    if last_banner_start is not None:
        prefix_pos = last_banner_start - len(LINE_PREFIX)
        if prefix_pos >= 0 and output[prefix_pos:last_banner_start] == LINE_PREFIX:
            output = output[:prefix_pos] + LINE_PREFIX_BOT + output[last_banner_start:]
        first_line_end = output.find("\n", last_banner_start)
        if first_line_end != -1:
            output = output[:first_line_end] + output[first_line_end:].replace(
                "\n" + LINE_PREFIX, "\n  "
            )
    else:
        # No banner: curve at the last line as before.
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
    frame_info_list: list[dict[str, Any]],
) -> list[int]:
    """Find all non-call frames that have variables to show.

    Returns list of frame indices with variables.
    """
    result = []
    for i, info in enumerate(frame_info_list):
        if info["relevance"] != "call" and info["frinfo"].get("variables"):
            result.append(i)
    return result


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
) -> tuple[str, int | None]:
    """Print frames in chronological order with exception info after error frames.

    Returns:
        A tuple of ``(output, last_banner_start)`` where *last_banner_start*
        is the character index of the first line of the last exception banner,
        or ``None`` if no banner was emitted.
    """
    output = ""
    last_banner_start = None
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
            last_banner_start = len(output)
            output += _build_exception_banner(exc_info, term_width)
        return output, last_banner_start

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
        len(output_lines)
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
        info["relevance"]
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


def _wrap_text(text: str, width: int, max_lines: int = 12) -> tuple[list[str], bool]:
    """Wrap *text* so that each line fits within *width* terminal columns.

    Hard linefeeds in *text* are expected to be split by the caller; this
    helper wraps a single paragraph.  If wrapping would produce an
    unreasonably long run of lines, the middle of the original text is
    cut out and shown as a single truncated line instead.

    Returns:
        A tuple of ``(lines, was_truncated)``.  The plain-text ellipsis
        (``…``) is left for the caller to style; callers that want a dim
        ellipsis can post-process the returned lines.
    """
    if not text:
        return [""], False

    wrapped = textwrap.wrap(
        text,
        width=width,
        break_long_words=True,
        break_on_hyphens=False,
        replace_whitespace=False,
    ) or [text]

    if len(wrapped) <= max_lines:
        return wrapped, False

    # Pathological input: keep the beginning and end, cut out the middle.
    # The rendered line is followed by a newline, so it can use the full
    # available width.
    half = max(0, (width - 1) // 2)
    last_len = max(0, width - 1 - half)
    truncated_line = text[:half] + "…" + text[-last_len:]
    return [truncated_line], True


def _dim_ellipsis_plain(line: str) -> str:
    """Dim the middle ellipsis in a plain (non-bold) line."""
    if "…" not in line:
        return line
    before, after = line.split("…", 1)
    return f"{before}{DIM}…{RESET}{after}"


def _dim_ellipsis_bold(line: str) -> str:
    """Keep text bold but render the middle ellipsis in dim color."""
    if "…" not in line:
        return f"{BOLD}{line}{RESET}"
    before, after = line.split("…", 1)
    return f"{BOLD}{before}{RESET}{DIM}…{RESET}{BOLD}{after}{RESET}"


def _char_display_width(char: str) -> int:
    """Return the terminal display width of a single character."""
    return 2 if unicodedata.east_asian_width(char) in "WF" else 1


def _truncate_ansi(colored: str, max_width: int) -> str:
    """Truncate a colored string so it fits within *max_width* columns.

    A dim ellipsis is appended at the end.  ANSI escape sequences are
    preserved and never counted towards the width.
    """
    if max_width < 1:
        return ""

    ellipsis = f"{RESET}{DIM}…{RESET}"
    ellipsis_width = _display_width(ellipsis)
    if max_width <= ellipsis_width:
        return ellipsis

    result_parts: list[str] = []
    width = 0
    i = 0
    n = len(colored)
    available = max_width - ellipsis_width

    while i < n:
        c = colored[i]
        if c == "\x1b":
            m = ANSI_ESCAPE_RE.match(colored, i)
            if m:
                seq = m.group(0)
                result_parts.append(seq)
                i += len(seq)
                continue
        w = _char_display_width(c)
        if width + w > available:
            break
        result_parts.append(c)
        width += w
        i += 1

    result_parts.append(ellipsis)
    return "".join(result_parts)


def _wrap_ansi(colored: str, max_width: int) -> list[tuple[str, str]]:
    """Wrap a colored string into chunks that fit within *max_width*.

    Returns a list of ``(chunk, active_codes)`` tuples.  *chunk* is the
    text/ANSI content for that chunk; *active_codes* is the ANSI SGR
    sequence that should be emitted before the chunk so that styles
    (such as the mark background) continue seamlessly on continuation
    lines.
    """
    if not colored:
        return [("", "")]

    chunks: list[tuple[str, str]] = []
    current: list[str] = []
    active_params: list[str] = []
    pending_active = ""
    width = 0
    i = 0
    n = len(colored)

    while i < n:
        c = colored[i]
        if c == "\x1b":
            m = ANSI_ESCAPE_RE.match(colored, i)
            if m:
                seq = m.group(0)
                current.append(seq)
                if seq.endswith("m"):
                    params = seq[2:-1]
                    if params == "0":
                        active_params = []
                    else:
                        active_params.append(params)
                i += len(seq)
                continue

        w = _char_display_width(c)
        if width + w > max_width:
            chunks.append(("".join(current), pending_active))
            pending_active = f"{ESC}{';'.join(active_params)}m" if active_params else ""
            current = []
            width = 0

        current.append(c)
        width += w
        i += 1

    if current:
        chunks.append(("".join(current), pending_active))

    return chunks


def _wrap_code_line(colored: str, max_width: int) -> list[str]:
    """Wrap a colored code line, restoring active styles on each continuation."""
    chunks = _wrap_ansi(colored, max_width)
    result: list[str] = []
    for i, (chunk, active) in enumerate(chunks):
        if i > 0 and active:
            chunk = active + chunk
        result.append(chunk)
    return result


# Maximum number of wrapped lines to render for a single exception banner.
# Messages that produce more wrapped lines are collapsed in the middle, like
# the "N more calls" ellipsis used for hidden call frames.
MAX_BANNER_LINES = 100
BANNER_HEAD_LINES = 20
BANNER_TAIL_LINES = 20


def _truncate_banner_lines(lines: list[str]) -> list[str]:
    """Collapse the middle of *lines* when there are too many to display."""
    if len(lines) <= MAX_BANNER_LINES:
        return lines
    skipped = len(lines) - BANNER_HEAD_LINES - BANNER_TAIL_LINES
    skipped_line = f"{ELLIPSIS}⋮ {skipped} more lines{RESET}"
    return lines[:BANNER_HEAD_LINES] + [skipped_line] + lines[-BANNER_TAIL_LINES:]


def _build_exception_banner(exc_info: dict[str, Any], term_width: int) -> str:
    """Build exception banner output to show after error frame.

    The banner is a sequence of logical content lines.  No box-drawing
    prefix is prepended here; the surrounding TTY formatter adds the
    ``│ `` prefix between lines via :data:`EOL`.  This keeps the left
    border single and consistent even for multi-line exception messages.
    """
    exc_type = exc_info.get("type", "Exception")
    summary = exc_info.get("summary", "")
    message = exc_info.get("message", "")
    from_type = exc_info.get("from", "none")

    chain_suffix = chainmsg.get(from_type, "")
    type_prefix = f"{exc_type}{chain_suffix}: "
    type_prefix_colored = f"{EXC}{type_prefix}{RESET}"
    type_prefix_width = _display_width(type_prefix)

    # The first banner line only has the "│ " border prefix.  Every
    # continuation line also carries the dim half-block marker "▐ ", so it
    # has 2 fewer usable columns.
    first_line_width = max(1, term_width - _display_width(LINE_PREFIX))
    cont_width = max(
        1, term_width - _display_width(LINE_PREFIX) - _display_width(f"{BLOCK_HALF} ")
    )

    lines: list[str] = []

    # --- Title row(s) ------------------------------------------------------
    # The first row is special: it carries the exception type and the
    # summary, and is emphasised in bold.
    title_width = type_prefix_width + _display_width(summary)
    title_first_available = first_line_width - type_prefix_width

    if title_width <= first_line_width:
        # The whole title fits on a single line.
        lines.append(f"{type_prefix_colored}{BOLD}{summary}{RESET}")
    elif _display_width(summary) <= cont_width:
        # Type prefix does not fit together with the summary, but the
        # summary alone fits: put them on consecutive lines.
        lines.append(type_prefix_colored)
        wrapped_summary, _ = _wrap_text(summary, cont_width)
        for line in wrapped_summary:
            lines.append(_dim_ellipsis_bold(line))
    else:
        # Both the prefix and the summary are too wide for one line.
        # Wrap the title normally: the first line carries the type prefix,
        # and continuation lines are indented by the half-block marker width
        # so the same wrap width applies to every logical line.
        cont_indent = " " * _display_width(f"{BLOCK_HALF} ")
        if title_first_available < 1:
            lines.append(type_prefix_colored)
            wrapped_summary, _ = _wrap_text(summary, cont_width)
            for line in wrapped_summary:
                lines.append(_dim_ellipsis_bold(line))
        else:
            wrapped_title = textwrap.wrap(
                summary,
                width=first_line_width,
                initial_indent=type_prefix,
                subsequent_indent=cont_indent,
                break_long_words=True,
                break_on_hyphens=False,
                replace_whitespace=False,
            ) or [type_prefix + summary]
            for i, line in enumerate(wrapped_title):
                if i == 0:
                    content = line[len(type_prefix) :]
                    lines.append(f"{type_prefix_colored}{_dim_ellipsis_bold(content)}")
                else:
                    content = line[len(cont_indent) :]
                    lines.append(_dim_ellipsis_bold(content))

    # --- Message body rows -------------------------------------------------
    if summary != message:
        body = message
        if summary and body.startswith(summary):
            body = body[len(summary) :]
            # Drop exactly the newline that separates the summary from the body.
            # Subsequent newlines are intentional blank lines and must be kept.
            if body.startswith("\n"):
                body = body[1:]

        for para in body.split("\n"):
            if para:
                wrapped_body, _ = _wrap_text(para, cont_width)
                for line in wrapped_body:
                    lines.append(_dim_ellipsis_plain(line))
            else:
                # Preserve intentional blank lines from the message.
                lines.append("")

    lines = _truncate_banner_lines(lines)

    # For multi-line messages, mark every continuation line with a dim half
    # block so the block visually hangs off the first line.
    if len(lines) > 1:
        cont_marker = f"{DIM}{BLOCK_HALF}{RESET} "
        lines = [lines[0]] + [f"{cont_marker}{line}" for line in lines[1:]]

    return "".join(line + EOL for line in lines)


def _build_subexception_summaries(
    parallel_branches: list[list[dict[str, Any]]], term_width: int
) -> str:
    """Build one-line summaries for each subexception branch.

    For TTY output, we don't have space for full tracebacks, so we show
    a compact summary: location, function, exception type and message.

    Like exception banners, the box-drawing prefix is supplied by the
    surrounding formatter through :data:`EOL`; this avoids a double
    left border when multiple summaries are emitted in a row.
    """
    output = ""
    border_width = _display_width(LINE_PREFIX)  # "│ "

    for branch in parallel_branches:
        # Get the summary for this branch, reserving space for the border.
        summary = _get_branch_summary(branch, term_width - border_width)
        output += f"{summary}{EOL}"

    return output


def _get_branch_summary(branch: list[dict[str, Any]], max_width: int) -> str:
    """Get a one-line summary for a subexception branch.

    Returns something like "file.py:10 func: ValueError: message"
    Truncates at the end if needed to fit max_width.
    """
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


def _get_frame_label(frinfo: dict[str, Any]) -> tuple[str, str]:
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
    function_name = frinfo["function"]
    function_suffix = frinfo["function_suffix"]
    if function_name:
        function_display = f"{function_name}{function_suffix}"
    elif function_suffix:
        function_display = function_suffix
    else:
        function_display = None

    # Build the location text with colors
    if notebook_cell:
        location_text = location
        location_suffix = "" if function_display else ":"
        location_part = f"{LOCFN}{location_text}{location_suffix}{RESET}"
    else:
        location_text = f"{location}{LOC_LINENO}:{cursor_line}{RESET}"
        location_suffix = "" if function_display else ":"
        location_part = f"{LOCFN}{location_text}{location_suffix}{RESET}"

    function_part = f"{FUNC}{function_display}{RESET}:" if function_display else ""

    return location_part, function_part


def _get_chrono_frame_info(frinfo: dict[str, Any]) -> dict[str, Any]:
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
    info: dict[str, Any], location_width: int, function_width: int, term_width: int
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
    location_width + 1 + function_width

    start = frinfo.get("linenostart", 1)
    symbol = symbols.get(relevance, "")
    symbol_colored = f"{EM_CALL}{symbol}{RESET}" if symbol else ""

    desc = symdesc[relevance]

    # Width available for the content after the left border "│ "
    content_width = max(1, term_width - _display_width(LINE_PREFIX))
    single_marked = len(info["marked_lines"]) == 1

    raw_lines: list[tuple[str, int, bool, bool]] = []
    suffix_texts: set[str] = set()

    if not fragments:
        # Show "(no source code)" with the symbol emoji like a code line would have
        line = f"{INDENT}{label} {NO_SOURCE}(no source code){symbol_colored}{RESET}"
        raw_lines.append((line, _display_width(line), False, bool(symbol)))
    elif relevance == "call":
        # One-liner for call frames
        if info["marked_lines"]:
            # Build full code with em parts
            code_parts = []
            # Also track em parts for potential collapsing
            em_ranges = []  # [(start_idx, end_idx), ...] in plain text
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
                        if em in ("solo", "beg"):
                            em_start = plain_len
                        code_parts.append(colored)
                        plain_len += len(plain)
                        if em in ("solo", "fin") and em_start is not None:
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
            raw_lines.append((line, _display_width(line), False, bool(symbol)))
        else:  # pragma: no cover
            line = f"{INDENT}{label} {symbol_colored}"
            raw_lines.append((line, _display_width(line), False, bool(symbol)))
    else:
        # Full format for error/warning/stop/except frames
        label_line = f"{INDENT}{label}"
        raw_lines.append((label_line, _display_width(label_line), False, False))

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

            if frame_range and abs_line == frame_range.lfinal and symbol:
                # Keep the symbol together with the code when it fits; otherwise
                # put the explanatory symbol line on its own.
                suffix = f"{symbol_colored}  {SYMBOLDESC}{desc}{RESET}"
                if (
                    _display_width(code_part) + 1 + _display_width(suffix)
                    <= content_width
                ):
                    line = f"{code_part} {suffix}"
                    raw_lines.append((line, _display_width(line), is_marked, True))
                else:
                    # Both parts belong to the caret line and should be
                    # wrapped rather than truncated with an ellipsis.
                    raw_lines.append(
                        (code_part, _display_width(code_part), is_marked, True)
                    )
                    raw_lines.append((suffix, _display_width(suffix), False, True))
                    suffix_texts.add(suffix)
            else:
                raw_lines.append(
                    (code_part, _display_width(code_part), is_marked, False)
                )

    # Fit the assembled lines to the terminal width.
    lines: list[tuple[str, int, bool]] = []
    for line, width, is_marked, has_symbol in raw_lines:
        if width <= content_width:
            # Always close a marked line before the newline so the background
            # does not bleed into the left border of the next row.
            if is_marked and not line.endswith(RESET):
                line = line + RESET

            # If this is a suffix line and the previous line has room, put it
            # on the same line as the wrapped code instead of a new row.
            if (
                line in suffix_texts
                and lines
                and lines[-1][1] + 1 + width <= content_width
            ):
                prev, prev_width, prev_marked = lines[-1]
                lines[-1] = (f"{prev} {line}", prev_width + 1 + width, prev_marked)
                continue

            lines.append((line, _display_width(line), is_marked))
            continue

        # Lines that carry the caret, and the only marked line in a frame,
        # are wrapped so the important part stays visible.  Everything else
        # is shortened with a trailing dim ellipsis.
        should_wrap = has_symbol or (is_marked and single_marked)
        if should_wrap:
            for chunk in _wrap_code_line(line, content_width):
                # Each wrapped chunk of a marked line is closed before the
                # newline; the next chunk re-opens the active style.
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
    frame_info_list: list[dict[str, Any]],
) -> list[tuple[int, int]]:
    """Find line ranges for call frames that can be used for inspector expansion.

    Returns list of (start_line, end_line) tuples for call frames.
    """
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
    frame_info_list: list[dict[str, Any]],
) -> tuple[list[int], int]:
    """Compute vertical positions for all inspectors, avoiding overlap.

    Positioning rules:
    1. Ideally stay within own frame
    2. If needed, expand to surrounding call lines
    3. If still not enough, add empty lines after the frame

    Returns (list of inspector_start positions, total_extra_lines needed).
    """
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


def _merge_chrono_output(
    output_lines: list[tuple[str, int, int, bool]],
    all_inspector_lines: list[InspectorLines],
    all_inspector_min_widths: list[int],
    term_width: int,
    inspector_frame_indices: list[int],
    exception_banners: list[tuple[int, str]],
    frame_info_list: list[dict[str, Any]],
) -> tuple[str, int | None]:
    """Merge chronological output with multiple inspectors and exception banners.

    Args:
        output_lines: List of (line, plain_len, frame_idx, is_marked) tuples
        all_inspector_lines: List of inspector line lists, one per inspector frame
            Each line is (colored_line, plain_width, value_start_col)
        all_inspector_min_widths: Minimum required widths for each inspector
        term_width: Terminal width
        inspector_frame_indices: List of frame indices that have inspectors
        exception_banners: List of (insert_pos, banner) tuples
        frame_info_list: Frame info for all frames
    """
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

        # Calculate column position for this inspector
        max_line_len = 0
        for li in range(  # pragma: no cover
            inspector_start, min(inspector_start + inspector_height, len(output_lines))
        ):
            if li < len(output_lines):
                max_line_len = max(max_line_len, output_lines[li][1])

        inspector_col = max_line_len + 2

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

            # Truncate inspector content if it exceeds available width
            # Only truncate the value portion, preserving name/type coloring
            available_for_content = (
                term_width - inspector_col - 5
            )  # 5 = box chars + space
            if insp_width > available_for_content > 0:
                # Get plain text to find where to truncate
                insp_plain = ANSI_ESCAPE_RE.sub("", insp_line)
                # Available space for value = total available - prefix
                available_for_value = available_for_content - value_start
                if available_for_value > 1:
                    # Truncate value only, keep prefix with its coloring
                    # Find the ANSI position corresponding to value_start in plain text
                    # by scanning the colored string
                    plain_idx = 0
                    colored_idx = 0
                    while plain_idx < value_start and colored_idx < len(insp_line):
                        if insp_line[colored_idx] == "\x1b":
                            # Skip ANSI sequence
                            while (
                                colored_idx < len(insp_line)
                                and insp_line[colored_idx] != "m"
                            ):
                                colored_idx += 1
                            colored_idx += 1  # skip the 'm'
                        else:
                            plain_idx += 1
                            colored_idx += 1
                    # colored_idx is now at the start of the value in the colored string
                    prefix_colored = insp_line[:colored_idx]
                    value_plain = insp_plain[value_start:]
                    truncated_value = value_plain[: available_for_value - 1] + "…"
                    insp_line = f"{prefix_colored}{VAR}{truncated_value}{RESET}"
                elif available_for_content > 0:  # pragma: no cover
                    # Not enough space, just show ellipsis
                    insp_line = f"{VAR}…{RESET}"

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

        # Insert exception banner if needed
        while banner_idx < len(exception_banners):
            insert_pos, banner = exception_banners[banner_idx]
            if li + 1 == insert_pos:
                last_banner_start = len(output)
                output += banner
                banner_idx += 1
            else:
                break

        # Check if we need to emit extra lines for inspector overflow
        # Find if any inspector ends after this line and needs extra lines
        for insp_idx, frame_idx in enumerate(inspector_frame_indices):
            frame_start, frame_end = _find_frame_line_range(output_lines, frame_idx)
            if li == frame_end:
                # Check if this inspector needs extra lines
                inspector_start = positions[insp_idx]
                inspector_height = len(all_inspector_lines[insp_idx])
                inspector_end = inspector_start + inspector_height

                # How many lines extend beyond the current output?
                if inspector_end > li + 1:  # pragma: no cover
                    for extra_li in range(li + 1, inspector_end):
                        if extra_li in inspector_at:
                            _, insp_line_idx, arrow_line, inspector_col = inspector_at[
                                extra_li
                            ]
                            insp_lines = all_inspector_lines[insp_idx]
                            insp_line, _, _ = insp_lines[insp_line_idx]

                            cursor_pos = f"{ESC}{inspector_col + 1}G"
                            is_last = insp_line_idx == len(insp_lines) - 1
                            is_arrow = insp_line_idx == arrow_line

                            if is_arrow:
                                box_char = BOX_BR if is_last else BOX_VL
                                output += f"{cursor_pos}{DIM}{ARROW_LEFT}{BOX_H}{box_char}{RESET} {insp_line}{EOL}"
                            else:
                                box_char = BOX_BL if is_last else BOX_V
                                output += f"{cursor_pos}  {DIM}{box_char}{RESET} {insp_line}{EOL}"

    # Any remaining banners
    for _, banner in exception_banners[banner_idx:]:
        last_banner_start = len(output)
        output += banner

    return output, last_banner_start


def _format_fragment(fragment: dict[str, Any]) -> str:
    """Format a fragment returning colored string."""
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

    return "".join(colored_parts)


def _format_fragment_call(fragment: dict[str, Any]) -> str:
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

    return "".join(colored_parts)


def _build_variable_inspector(
    variables: list[Any], term_width: int
) -> tuple[InspectorLines, int]:
    """Build variable inspector lines.

    Returns:
        tuple: (list of (colored_line, plain_width, value_start_col), min_required_width)
            - list of lines for the inspector with metadata for smart truncation
            - minimum width needed to display "name: type = " + some value chars

    Uses simple left-side vertical bar only, no top/bottom borders.
    Multi-line string values are rendered with continuation lines properly indented.
    Variable names are right-aligned so that = signs line up.
    """
    if not variables:
        return [], 0

    # First pass: collect variable info and filter out non-displayable values
    var_data = []  # [(name, typename, val_str, fmt_hint), ...]
    for var_info in variables:
        # Handle both old tuple format and new VarInfo namedtuple
        if hasattr(var_info, "name"):
            name, typename, value, fmt_hint = (
                var_info.name,
                var_info.typename,
                var_info.value,
                var_info.format_hint,
            )
        else:
            name, typename, value = var_info
            fmt_hint = "inline"

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

    # Pre-truncate values to a reasonable width (final truncation happens at render)
    value_width = max(5, term_width // 2 - 4 - prefix_width)

    # Build variable lines with right-aligned names
    # Each result entry: (colored_line, plain_width, value_start_col)
    # value_start_col is where the value starts, for smart truncation at render time
    result = []
    for name, typename, val_str, fmt_hint in var_data:
        name_part = f"{name}: {typename}" if typename else name
        padding = " " * (max_name_part_len - len(name_part))
        indent = " " * prefix_width

        # Handle multi-line block format
        if fmt_hint == "block" and "\n" in val_str:  # pragma: no cover
            for i, val_line in enumerate(val_str.split("\n")):
                # Truncate value line if needed
                if len(val_line) > value_width:
                    val_line = val_line[: value_width - 1] + "…"
                if i == 0:
                    # First line with name and type
                    if typename:
                        line = f"{VAR}{padding}{name}: {TYPE_COLOR}{typename} = {VAR}{val_line}{RESET}"
                    else:
                        line = f"{VAR}{padding}{name} = {val_line}{RESET}"
                    plain = f"{padding}{name_part} = {val_line}"
                    result.append((line, len(plain), prefix_width))
                else:
                    # Continuation lines (indented) - all value
                    line = f"{VAR}{indent}{val_line}{RESET}"
                    plain = f"{indent}{val_line}"
                    result.append((line, len(plain), prefix_width))
        else:
            # Single line format
            if len(val_str) > value_width:
                val_str = val_str[: value_width - 1] + "…"
            if typename:
                line = f"{VAR}{padding}{name}: {TYPE_COLOR}{typename} = {VAR}{val_str}{RESET}"
            else:
                line = f"{VAR}{padding}{name} = {val_str}{RESET}"
            plain = f"{padding}{name_part} = {val_str}"
            result.append((line, len(plain), prefix_width))

    return result, min_required_width

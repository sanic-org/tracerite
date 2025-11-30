import inspect
import re
import sys
from collections import namedtuple
from contextlib import suppress
from pathlib import Path
from secrets import token_urlsafe
from textwrap import dedent
from urllib.parse import quote

from . import trace_cpy
from .inspector import extract_variables
from .logging import logger

# Position range: lines are 1-based inclusive, columns are 0-based exclusive
Range = namedtuple("Range", ["lfirst", "lfinal", "cbeg", "cend"])

# Will be set to an instance if loaded as an IPython extension by %load_ext
ipython = None

# Locations considered to be bug-free
libdir = re.compile(r"/usr/.*|.*(site-packages|dist-packages).*")


def extract_chain(exc=None, **kwargs) -> list:
    """Extract information on current exception."""
    chain = []
    exc = exc or sys.exc_info()[1]
    while exc:
        chain.append(exc)
        exc = exc.__cause__ or None if exc.__suppress_context__ else exc.__context__
    # Newest exception first
    return [extract_exception(e, **(kwargs if e is chain[0] else {})) for e in chain]


def _create_summary(message):
    """Create a truncated summary of the exception message."""
    summary = message.split("\n", 1)[0]
    if len(summary) <= 100:
        return summary

    if len(message) > 1000:
        # Sometimes the useful bit is at the end of a very long message
        return f"{message[:40]} ··· {message[-40:]}"
    else:
        return f"{summary[:60]} ···"


def extract_exception(e, *, skip_outmost=0, skip_until=None) -> dict:
    raw_tb = e.__traceback__
    try:
        tb = inspect.getinnerframes(raw_tb)
    except IndexError:  # Bug in inspect internals, find_source()
        logger.exception("Bug in inspect?")
        tb = []
        raw_tb = None
    if skip_until:
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
    summary = _create_summary(message)
    try:
        # KeyboardErrors and such need not be reported all the way
        suppress = not isinstance(e, Exception)
        frames = extract_frames(tb, raw_tb, suppress_inner=suppress)
    except Exception:
        logger.exception("Error extracting traceback")
        frames = None
    return {
        "type": type(e).__name__,
        "message": message,
        "summary": summary,
        "has": ("cause" if e.__cause__ else "context" if e.__context__ else "none"),
        "repr": repr(e),
        "frames": frames or [],
    }


def extract_source_lines(frame, lineno, end_lineno=None):
    try:
        lines, start = inspect.getsourcelines(frame)
        if start == 0:
            start = 1
        # If we have end_lineno, make sure we extract enough lines to include it
        # Otherwise, default to showing lineno + 3 lines after
        lines_after = (end_lineno - lineno + 3) if end_lineno else 3
        lines = lines[
            max(0, lineno - start - 15) : max(0, lineno - start + lines_after)
        ]
        start += max(0, lineno - start - 15)

        # Calculate common indentation before dedenting
        common_indent = _calculate_common_indent(lines)

        # Return both dedented content and the common indentation amount
        dedented_content = dedent("".join(lines))
        return dedented_content, start, common_indent
    except OSError:
        return "", lineno, 0  # Source not available (non-Python module)


def format_location(filename, lineno):
    urls = {}
    location = None
    try:
        ipython_in = ipython.compile._filename_map[filename]
        location = f"In [{ipython_in}]"
        filename = None
    except (AttributeError, KeyError):
        pass
    if filename and Path(filename).is_file():
        fn = Path(filename).resolve()
        urls["VS Code"] = f"vscode://file/{quote(fn.as_posix())}:{lineno}"
        cwd = Path.cwd()
        if cwd in fn.parents:
            fn = fn.relative_to(cwd)
            if ipython is not None:
                urls["Jupyter"] = f"/edit/{quote(fn.as_posix())}"
        filename = fn.as_posix()
    if not location and filename:
        split = (
            filename.rfind("/", 10, len(filename) - 20) + 1 if len(filename) > 40 else 0
        )
        location = filename[split:]
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


def _get_frame_relevance(is_last_frame, is_bug_frame, suppress_inner):
    """Determine frame relevance: error, warning, call, or stop."""
    if is_last_frame:
        return "stop" if suppress_inner else "error"
    elif is_bug_frame:
        return "warning"
    return "call"


def _extract_emphasis_columns(
    lines, error_line_in_context, end_line, start_col, end_col, start
):
    """Extract emphasis columns using caret anchors from the code segment.

    Returns Range with 1-based inclusive line numbers and 0-based exclusive columns,
    or None if no anchors found.
    """
    if not (end_line and start_col is not None and end_col is not None):
        return None

    all_lines = lines.splitlines(keepends=True)
    segment_start = error_line_in_context - 1  # Convert to 0-based for indexing
    segment_end = end_line if end_line else error_line_in_context

    if not (0 <= segment_start < len(all_lines) and segment_end <= len(all_lines)):
        return None

    # Extract the segment using CPython's approach
    relevant_lines = all_lines[segment_start:segment_end]
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
    """Build mapping from frame objects to position tuples."""
    position_map = {}
    if not raw_tb:
        return position_map
    try:
        for frame_obj, positions in trace_cpy._walk_tb_with_full_positions(raw_tb):
            position_map[frame_obj] = positions
    except Exception:
        logger.exception("Error extracting position information")
    return position_map


def _find_bug_frame(tb):
    """Find the most relevant user code frame in the traceback."""
    return next(
        (
            f
            for f in reversed(tb)
            if f.code_context and not libdir.fullmatch(f.filename)
        ),
        tb[-1],
    ).frame


def extract_frames(tb, raw_tb=None, suppress_inner=False) -> list:
    if not tb:
        return []

    position_map = _build_position_map(raw_tb)
    bug_in_frame = _find_bug_frame(tb)

    frames = []
    for frame, filename, lineno, function, codeline, _ in tb:
        if frame.f_globals.get("__tracebackhide__") or frame.f_locals.get(
            "__tracebackhide__"
        ):
            continue

        is_last_frame = frame is tb[-1][0]
        is_bug_frame = frame is bug_in_frame
        relevance = _get_frame_relevance(is_last_frame, is_bug_frame, suppress_inner)

        # Extract position information first so we can use it for source extraction
        pos = position_map.get(frame, [None] * 4)
        pos_end_lineno, start_col, end_col = pos[1], pos[2], pos[3]

        lines, start, original_common_indent = extract_source_lines(
            frame, lineno, pos_end_lineno
        )
        if not lines and relevance == "call":
            continue

        filename, location, urls = format_location(filename, lineno)
        function = _get_qualified_function_name(frame, function)

        error_line_in_context = lineno - start + 1
        end_line = pos_end_lineno - start + 1 if pos_end_lineno else None

        # Calculate common indentation that will be removed for display
        # (we already have this from extract_source_lines, no need to recalculate)
        common_indent = original_common_indent

        # Adjust column positions to account for dedenting
        # Python's column numbers are based on the original indented code,
        # but we display dedented code, so we need to subtract the common indentation
        adjusted_start_col = (
            start_col - common_indent if start_col is not None else None
        )
        adjusted_end_col = end_col - common_indent if end_col is not None else None

        # Create mark range (1-based inclusive lines, 0-based exclusive columns)
        mark_range = None
        if adjusted_start_col is not None and adjusted_end_col is not None:
            # Ensure columns are not negative after dedenting adjustment
            adjusted_start_col = max(0, adjusted_start_col)
            adjusted_end_col = max(0, adjusted_end_col)
            mark_lfinal = end_line or error_line_in_context
            mark_range = Range(
                error_line_in_context, mark_lfinal, adjusted_start_col, adjusted_end_col
            )

        # Build emphasis range and fragments
        em_range = _extract_emphasis_columns(
            lines,
            error_line_in_context,
            end_line,
            adjusted_start_col,
            adjusted_end_col,
            start,
        )
        fragments = _parse_lines_to_fragments(lines, mark_range, em_range)

        frames.append(
            {
                "id": f"tb-{token_urlsafe(12)}",
                "relevance": relevance,
                "filename": filename,
                "location": location,
                "codeline": codeline[0].strip() if codeline else None,
                "range": Range(lineno, pos_end_lineno or lineno, start_col, end_col)
                if start_col is not None
                else None,
                "linenostart": start,
                "lines": lines,
                "fragments": fragments,
                "function": function,
                "urls": urls,
                "variables": extract_variables(frame.f_locals, lines),
            }
        )

        if suppress_inner and is_bug_frame:
            break

    return frames


def _calculate_common_indent(lines):
    """Calculate common indentation across all non-empty lines."""
    non_empty_lines = [line.rstrip("\r\n") for line in lines if line.strip()]
    if non_empty_lines:
        return min(len(line) - len(line.lstrip()) for line in non_empty_lines)
    return 0


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


def _create_unified_fragments(
    lines_text, common_indent_len, mark_positions, em_positions
):
    """Create fragments with unified mark/em highlighting."""
    lines = lines_text.splitlines(keepends=True)
    result = []

    for line_idx, line in enumerate(lines):
        line_num = line_idx + 1
        fragments = _parse_line_to_fragments_unified(
            line,
            common_indent_len,
            mark_positions,
            em_positions,
            sum(len(lines[i]) for i in range(line_idx)),  # char offset for this line
        )
        result.append({"line": line_num, "fragments": fragments})

    return result


def _parse_line_to_fragments_unified(
    line, common_indent_len, mark_positions, em_positions, line_char_offset
):
    """Parse a single line into fragments using unified highlighting."""
    line_content, line_ending = _split_line_content(line)
    if not line_content and not line_ending:
        return []

    # Process indentation
    fragments, remaining, pos = _process_indentation(line_content, common_indent_len)

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


def _parse_lines_to_fragments(lines_text, mark_range=None, em_range=None):
    """
    Parse lines of code into fragments with mark/em highlighting information.

    Args:
        lines_text: The multi-line string containing code
        mark_range: Range object for mark highlighting (or None)
        em_range: Range object for em highlighting (or None)

    Returns:
        List of line dictionaries with fragment information
    """
    lines = lines_text.splitlines(keepends=True)
    if not lines:
        return []

    common_indent_len = _calculate_common_indent(lines)

    # Convert both mark and em to position sets using unified logic
    mark_positions = _convert_range_to_positions(mark_range, lines)
    em_positions = _convert_range_to_positions(em_range, lines)

    # Create fragments using unified highlighting
    return _create_unified_fragments(
        lines_text, common_indent_len, mark_positions, em_positions
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


def _process_indentation(line_content, common_indent_len):
    """Process dedent and additional indentation, return fragments and remaining content."""
    fragments = []
    pos = 0

    # Handle dedent (common indentation)
    if common_indent_len > 0 and len(line_content) > common_indent_len:
        dedent_text = line_content[:common_indent_len]
        fragments.append({"code": dedent_text, "dedent": "solo"})
        pos = common_indent_len

    # Handle additional indentation
    remaining = line_content[pos:]
    indent_match = re.match(r"^(\s+)", remaining)
    if indent_match:
        indent_text = indent_match.group(1)
        fragments.append({"code": indent_text, "indent": "solo"})
        pos += len(indent_text)
        remaining = remaining[len(indent_text) :]

    return fragments, remaining, pos


def _find_comment_start(text):
    """Find the start of a comment, ignoring # inside strings."""
    in_string = False
    string_char = None
    escape_next = False

    for i, char in enumerate(text):
        if escape_next:
            escape_next = False
            continue
        if char == "\\":
            escape_next = True
            continue
        if not in_string and char == "#":
            return i
        if not in_string and char in ('"', "'"):
            in_string = True
            string_char = char
        elif in_string and char == string_char:
            in_string = False
            string_char = None

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

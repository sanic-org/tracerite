import inspect
import re
import sys
from pathlib import Path
from secrets import token_urlsafe
from textwrap import dedent
from urllib.parse import quote

from . import trace_cpy
from .inspector import extract_variables
from .logging import logger

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
    summary = message.split("\n", 1)[0]
    if len(summary) > 100:
        if len(message) > 1000:
            # Sometimes the useful bit is at the end of a very long message
            summary = f"{message[:40]} ··· {message[-40:]}"
        else:
            summary = f"{summary[:60]} ···"
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


def extract_source_lines(frame, lineno):
    try:
        lines, start = inspect.getsourcelines(frame)
        if start == 0:
            start = 1
        lines = lines[max(0, lineno - start - 15) : max(0, lineno - start + 3)]
        start += max(0, lineno - start - 15)
        return dedent("".join(lines)), start
    except OSError:
        return "", 1


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
    """Extract emphasis columns using caret anchors from the code segment."""
    em_columns = {}

    if not (end_line and start_col is not None and end_col is not None):
        return em_columns

    try:
        lines_list = lines.splitlines(keepends=True)
        segment_start = error_line_in_context - 1  # Convert to 0-based index
        segment_end = end_line if end_line else error_line_in_context

        if not (
            0 <= segment_start < len(lines_list) and segment_end <= len(lines_list)
        ):
            return em_columns

        segment_lines = lines_list[segment_start:segment_end]
        segment = "".join(segment_lines)

        # For single-line errors, try to extract just the expression part
        if len(segment_lines) == 1:
            anchors = _extract_single_line_anchors(
                segment_lines[0], start_col, end_col, segment_start, em_columns
            )
            if not anchors:
                # Fallback to full segment approach
                _extract_full_segment_anchors(segment, segment_start, em_columns)
        else:
            # Multi-line segment
            _extract_full_segment_anchors(segment, segment_start, em_columns)
            if not em_columns:
                # Fallback for complex multiline cases
                _fallback_multiline_operator_detection(
                    segment_lines, segment_start, em_columns
                )

    except Exception:
        logger.exception("Error extracting caret anchors")

    return em_columns


def _extract_single_line_anchors(
    line_content, start_col, end_col, segment_start, em_columns
):
    """Extract anchors for single-line errors."""
    line_content = line_content.rstrip("\r\n")
    if not (start_col < len(line_content) and end_col <= len(line_content)):
        return False

    expression_segment = line_content[start_col:end_col]
    anchors = trace_cpy._extract_caret_anchors_from_line_segment(expression_segment)

    if not anchors:
        return False

    left_line = segment_start + 1  # Convert back to 1-based line number in context
    left_col = start_col + anchors.left_end_offset

    # Add emphasis columns for the anchor positions
    if left_line not in em_columns:
        em_columns[left_line] = []
    em_columns[left_line].append(left_col)

    # For operators that span multiple characters (like subscripts)
    if anchors.right_start_offset > anchors.left_end_offset + 1:
        right_col = start_col + anchors.right_start_offset - 1
        for col in range(left_col + 1, right_col + 1):
            em_columns[left_line].append(col)

    return True


def _extract_full_segment_anchors(segment, segment_start, em_columns):
    """Extract anchors from full segment (fallback or multi-line)."""
    anchors = trace_cpy._extract_caret_anchors_from_line_segment(segment)
    if not anchors:
        return

    # Convert anchors to em_columns format
    left_line = anchors.left_end_lineno + segment_start + 1  # Convert back to 1-based
    right_line = anchors.right_start_lineno + segment_start + 1

    # Add emphasis columns for the anchor positions
    if left_line not in em_columns:
        em_columns[left_line] = []
    em_columns[left_line].append(anchors.left_end_offset)

    if right_line not in em_columns:
        em_columns[right_line] = []
    if right_line != left_line or anchors.right_start_offset != anchors.left_end_offset:
        em_columns[right_line].append(anchors.right_start_offset)


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

        lines, start = extract_source_lines(frame, lineno)
        if not lines and relevance == "call":
            continue

        filename, location, urls = format_location(filename, lineno)
        function = _get_qualified_function_name(frame, function)

        # Extract position information
        pos = position_map.get(frame, [None] * 4)
        pos_end_lineno, start_col, end_col = pos[1], pos[2], pos[3]

        error_line_in_context = lineno - start + 1
        end_line = pos_end_lineno - start + 1 if pos_end_lineno else None

        # Build emphasis columns and fragments
        em_columns = _extract_emphasis_columns(
            lines, error_line_in_context, end_line, start_col, end_col, start
        )
        fragments = _parse_lines_to_fragments(
            lines,
            error_line_in_context,
            start_col,
            end_col,
            end_line,
            em_columns=em_columns,
        )

        frames.append(
            {
                "id": f"tb-{token_urlsafe(12)}",
                "relevance": relevance,
                "filename": filename,
                "location": location,
                "codeline": codeline[0].strip() if codeline else None,
                "lineno": lineno,
                "end_lineno": pos_end_lineno or lineno,
                "colno": start_col,
                "end_colno": end_col,
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


def _parse_lines_to_fragments(
    lines_text,
    error_line,
    start_col=None,
    end_col=None,
    end_line=None,
    em_columns=None,
    mark_error_line=False,
):
    """
    Parse lines of code into fragments with mark/em highlighting information.

    Args:
        lines_text: The multi-line string containing code
        error_line: The line number where the error occurred (1-based)
        start_col: Start column for highlighting (0-based, optional)
        end_col: End column for highlighting (0-based, optional)
        end_line: End line for multi-line errors (1-based, optional)
        em_columns: Dict mapping line numbers to lists of column positions for emphasis
        mark_error_line: If True and no column info, mark the entire error line

    Returns:
        List of line dictionaries with fragment information
    """
    if not lines_text:
        return []


def _calculate_mark_range(
    line_num, line_content, error_line, start_col, end_col, end_line, mark_error_line
):
    """Calculate mark range for a given line."""
    if start_col is not None and end_col is not None:
        if end_line is not None and end_line != error_line:
            # Multi-line error
            if line_num == error_line:
                return (start_col, len(line_content.rstrip()))
            elif line_num == end_line:
                return (len(line_content) - len(line_content.lstrip()), end_col)
            elif error_line < line_num < end_line:
                return (
                    len(line_content) - len(line_content.lstrip()),
                    len(line_content.rstrip()),
                )
        elif line_num == error_line:
            return (start_col, end_col)
    elif mark_error_line and line_num == error_line:
        # Mark the entire error line when no column info is available
        leading_spaces = len(line_content) - len(line_content.lstrip())
        trimmed_end = len(line_content.rstrip())
        if trimmed_end > leading_spaces:
            return (leading_spaces, trimmed_end)
    return None


def _calculate_common_indent(lines):
    """Calculate common indentation across all non-empty lines."""
    non_empty_lines = [line.rstrip("\r\n") for line in lines if line.strip()]
    if non_empty_lines:
        return min(len(line) - len(line.lstrip()) for line in non_empty_lines)
    return 0


def _parse_lines_to_fragments(
    lines_text,
    error_line,
    start_col=None,
    end_col=None,
    end_line=None,
    em_columns=None,
    mark_error_line=False,
):
    """
    Parse lines of code into fragments with mark/em highlighting information.

    Args:
        lines_text: The multi-line string containing code
        error_line: The line number where the error occurred (1-based)
        start_col: Start column for highlighting (0-based, optional)
        end_col: End column for highlighting (0-based, optional)
        end_line: End line for multi-line errors (1-based, optional)
        em_columns: Dict mapping line numbers to lists of column positions for emphasis
        mark_error_line: If True and no column info, mark the entire error line

    Returns:
        List of line dictionaries with fragment information
    """
    if not lines_text:
        return []

    lines = lines_text.splitlines(keepends=True)
    if not lines:
        return []

    common_indent_len = _calculate_common_indent(lines)
    result = []

    for line_idx, line in enumerate(lines):
        line_num = line_idx + 1
        line_content = line.rstrip("\r\n")

        # Calculate mark range for this line
        mark_range = _calculate_mark_range(
            line_num,
            line_content,
            error_line,
            start_col,
            end_col,
            end_line,
            mark_error_line,
        )

        # Get emphasis columns for this line
        em_cols = em_columns.get(line_num, []) if em_columns else []

        # Parse line into fragments
        fragments = _parse_line_to_fragments(
            line, common_indent_len, mark_range, em_cols
        )

        result.append({"line": line_num, "fragments": fragments})

    return result


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


def _create_content_fragments(content, start_pos, mark_range, em_columns, content_type):
    """Create fragments for content with optional highlighting."""
    if not content:
        return []

    if content_type == "comment":
        return [{"code": content, "comment": "solo"}]
    elif content_type == "trailing":
        return [{"code": content, "trailing": "solo"}]
    else:  # code content
        return _create_highlighted_fragments(content, start_pos, mark_range, em_columns)


def _parse_line_to_fragments(line, common_indent_len, mark_range, em_columns):
    """Parse a single line into fragments with proper marking."""
    if not line:
        return []

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
                _create_content_fragments(
                    code_trimmed, pos, mark_range, em_columns, "code"
                )
            )

        # Process comment part
        comment_trimmed = comment_part.rstrip()
        comment_trailing = comment_part[len(comment_trimmed) :]

        comment_with_leading_space = code_whitespace + comment_trimmed
        if comment_with_leading_space:
            fragments.extend(
                _create_content_fragments(
                    comment_with_leading_space, pos, mark_range, em_columns, "comment"
                )
            )

        # Add trailing content
        trailing_content = comment_trailing + line_ending
        if trailing_content:
            fragments.extend(
                _create_content_fragments(
                    trailing_content, pos, mark_range, em_columns, "trailing"
                )
            )
    else:
        # Handle line without comment
        code_trimmed = remaining.rstrip()
        trailing_whitespace = remaining[len(code_trimmed) :]

        if code_trimmed:
            fragments.extend(
                _create_content_fragments(
                    code_trimmed, pos, mark_range, em_columns, "code"
                )
            )

        trailing_content = trailing_whitespace + line_ending
        if trailing_content:
            fragments.extend(
                _create_content_fragments(
                    trailing_content, pos, mark_range, em_columns, "trailing"
                )
            )

    return fragments


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


def _create_highlighted_fragments(text, start_pos, mark_range, em_columns):
    """Create fragments with mark/em highlighting using unified logic."""
    if not text:
        return []

    # Convert columns to text-relative positions
    mark_positions = set()
    em_positions = set()

    if mark_range:
        mark_start, mark_end = mark_range
        for i in range(len(text)):
            abs_pos = start_pos + i
            if mark_start <= abs_pos < mark_end:
                mark_positions.add(i)

    # Convert em_columns to positions
    if em_columns:
        valid_cols = [
            col for col in em_columns if start_pos <= col < start_pos + len(text)
        ]
        for col in valid_cols:
            rel_pos = col - start_pos
            if 0 <= rel_pos < len(text):
                em_positions.add(rel_pos)

    # Create fragments using unified logic
    return _create_fragments_with_highlighting(text, mark_positions, em_positions)


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
        if not fragment_text:
            continue

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


def _fallback_multiline_operator_detection(segment_lines, segment_start, em_columns):
    """
    Fallback approach to find operators in multiline segments where
    caret anchor extraction fails. Looks for standalone operators in each line.
    """
    import re

    # Common binary operators that might appear on their own line
    operator_pattern = re.compile(r"^\s*([+\-*/=<>!&|%^]+)\s*(?:#.*)?$")

    for line_idx, line_content in enumerate(segment_lines):
        match = operator_pattern.match(line_content.rstrip("\r\n"))
        if match:
            operator = match.group(1)
            operator_start = match.start(1)

            # Convert to absolute line number (1-based)
            abs_line = segment_start + line_idx + 1

            # Add emphasis for each character of the operator
            if abs_line not in em_columns:
                em_columns[abs_line] = []

            for i in range(len(operator)):
                em_columns[abs_line].append(operator_start + i)

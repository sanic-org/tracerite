import inspect
import re
import sys
from pathlib import Path
from secrets import token_urlsafe
from textwrap import dedent
from urllib.parse import quote

from .inspector import extract_variables
from .logging import logger
from . import trace_cpy

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
        if getattr(exc, "__suppress_context__", False):
            break
        exc = exc.__cause__ or exc.__context__
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
        "repr": repr(e),
        "frames": frames or [],
    }


def extract_frames(tb, raw_tb=None, suppress_inner=False) -> list:
    if not tb:
        return []
    frames = []
    
    # Create mapping from frames to position information
    position_map = {}
    if raw_tb:
        try:
            for frame_obj, positions in trace_cpy._walk_tb_with_full_positions(raw_tb):
                position_map[frame_obj] = positions
        except Exception:
            logger.exception("Error extracting position information")
    
    # Choose a frame to open by default
    # - The innermost non-library frame (libdir regex), or if not found,
    # - The innermost frame
    bug_in_frame = next(
        (
            f
            for f in reversed(tb)
            if f.code_context and not libdir.fullmatch(f.filename)
        ),
        tb[-1],
    ).frame
    
    for frame, filename, lineno, function, codeline, _ in tb:
        if frame.f_globals.get("__tracebackhide__", False):
            continue
        if frame.f_locals.get("__tracebackhide__", False):
            continue
        if frame is tb[-1][0]:
            # Exception was raised here, show an error bomb normally, but
            # stop for KeyboardInterrupt, CancelledError (BaseException)
            relevance = "stop" if suppress_inner else "error"
        elif frame is bug_in_frame:
            relevance = "warning"
        else:
            relevance = "call"
        # Extract source code lines
        lines = []
        try:
            lines, start = inspect.getsourcelines(frame)
            if start == 0:
                start = 1  # Zero is always returned for modules; fix that.
            # Limit lines shown
            lines = lines[max(0, lineno - start - 15) : max(0, lineno - start + 3)]
            start += max(0, lineno - start - 15)
            # Deindent
            lines = dedent("".join(lines))
        except OSError:
            lines = ""
            # Skip non-Python modules unless particularly relevant
            if relevance == "call":
                continue
        urls = {}
        # Format location
        location = None
        try:
            ipython_in = ipython.compile._filename_map[filename]
            location = f"In [{ipython_in}]"
            filename = None
        except (AttributeError, KeyError):
            pass
        # Format filename
        if filename and Path(filename).is_file():
            fn = Path(filename).resolve()
            urls["VS Code"] = f"vscode://file/{quote(fn.as_posix())}:{lineno}"
            cwd = Path.cwd()
            # Use relative path if in current working directory
            if cwd in fn.parents:
                fn = fn.relative_to(cwd)
                if ipython is not None:
                    urls["Jupyter"] = f"/edit/{quote(fn.as_posix())}"
            filename = fn.as_posix()  # Use forward slashes
        # Shorten filename to use as displayable location
        if not location and filename is not None:
            split = 0
            if len(filename) > 40:
                split = filename.rfind("/", 10, len(filename) - 20) + 1
            location = filename[split:]

        if function == "<module>":
            function = None
        else:
            # Add class name to methods (if self or cls is the first local variable)
            try:
                cls = next(
                    v.__class__ if n == "self" else v
                    for n, v in frame.f_locals.items()
                    if n in ("self", "cls") and v is not None
                )
                function = f"{cls.__name__}.{function}"
            except StopIteration:
                pass
            # Remove long module paths (keep last two items)
            function = ".".join(function.split(".")[-2:])

        # Parse lines into fragments for enhanced highlighting
        fragments = []
        if lines:
            # Determine error position for fragment parsing
            error_line_in_context = (
                lineno - start + 1
            )  # Convert to 1-based index within the lines context

            # Get position information from trace_cpy
            start_col = None
            end_col = None
            end_line = None
            
            if frame in position_map:
                positions = position_map[frame]
                # positions is tuple: (lineno, end_lineno, colno, end_colno)
                if len(positions) >= 4:
                    pos_lineno, pos_end_lineno, pos_colno, pos_end_colno = positions
                    
                    # Use position data if available
                    if pos_colno is not None:
                        start_col = pos_colno
                    if pos_end_colno is not None:
                        end_col = pos_end_colno
                    if pos_end_lineno is not None and pos_end_lineno != lineno:
                        # Convert to context-relative line number
                        end_line = pos_end_lineno - start + 1

            # Parse fragments with position information
            if start_col is None and end_col is None:
                # No column info available, use the mark_error_line approach
                fragments = _parse_lines_to_fragments(
                    lines,
                    error_line_in_context,
                    start_col=None,
                    end_col=None,
                    end_line=None,
                    em_columns=None,
                    mark_error_line=True,
                )
            else:
                # Use the enhanced column information
                # Default end_line to error_line if not set
                end_line_in_context = (
                    end_line if end_line is not None else error_line_in_context
                )

                fragments = _parse_lines_to_fragments(
                    lines,
                    error_line_in_context,
                    start_col,
                    end_col,
                    end_line_in_context,
                    em_columns=None,
                )

        frames.append(
            {
                "id": f"tb-{token_urlsafe(12)}",
                "relevance": relevance,
                "filename": filename,
                "location": location,
                "codeline": codeline[0].strip() if codeline else None,
                "lineno": lineno,
                "linenostart": start,
                "lines": lines,
                "fragments": fragments,  # New fragment-based structure
                "function": function,
                "urls": urls,
                "variables": extract_variables(frame.f_locals, lines),
            }
        )
        if suppress_inner and frame is bug_in_frame:
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

    lines = lines_text.splitlines(keepends=True)
    if not lines:
        return []

    # Calculate common indentation (dedent)
    common_indent_len = 0
    non_empty_lines = [line.rstrip("\r\n") for line in lines if line.strip()]
    if non_empty_lines:
        common_indent_len = min(
            len(line) - len(line.lstrip()) for line in non_empty_lines
        )

    result = []

    for line_idx, line in enumerate(lines):
        line_num = line_idx + 1

        # Determine mark range for this line
        mark_range = None
        if start_col is not None and end_col is not None:
            # Work with the line content without line endings for range calculations
            line_content = line.rstrip("\r\n")
            if end_line is not None and end_line != error_line:
                # Multi-line error
                if line_num == error_line:
                    mark_range = (start_col, len(line_content.rstrip()))
                elif line_num == end_line:
                    mark_range = (
                        len(line_content) - len(line_content.lstrip()),
                        end_col,
                    )
                elif error_line < line_num < end_line:
                    mark_range = (
                        len(line_content) - len(line_content.lstrip()),
                        len(line_content.rstrip()),
                    )
            elif line_num == error_line:
                mark_range = (start_col, end_col)
        elif mark_error_line and line_num == error_line:
            # Mark the entire error line when no column info is available
            line_content = line.rstrip("\r\n")
            # Mark from first non-whitespace to last non-whitespace character
            leading_spaces = len(line_content) - len(line_content.lstrip())
            trimmed_end = len(line_content.rstrip())
            if trimmed_end > leading_spaces:
                mark_range = (leading_spaces, trimmed_end)

        # Get emphasis columns for this line
        em_cols = em_columns.get(line_num, []) if em_columns else []

        # Parse line into fragments - always create fragments for all lines
        # regardless of whether they have marking/highlighting
        fragments = _parse_line_to_fragments(
            line, common_indent_len, mark_range, em_cols
        )

        line_info = {"line": line_num, "fragments": fragments}
        result.append(line_info)

    return result


def _parse_line_to_fragments(line, common_indent_len, mark_range, em_columns):
    """Parse a single line into fragments with proper marking."""
    if not line:
        return []

    # Separate line content from line ending
    line_ending = ""
    if line.endswith("\r\n"):
        line_content = line[:-2]
        line_ending = "\r\n"
    elif line.endswith("\n"):
        line_content = line[:-1]
        line_ending = "\n"
    elif line.endswith("\r"):
        line_content = line[:-1]
        line_ending = "\r"
    else:
        line_content = line

    if not line_content and not line_ending:
        return []

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

    # Find comment
    comment_start = _find_comment_start(remaining)
    if comment_start is not None:
        # Split at comment boundary, keeping original spacing
        code_part = remaining[:comment_start]
        comment_part = remaining[comment_start:]

        # Remove trailing whitespace from code part for processing
        code_part_trimmed = code_part.rstrip()
        code_whitespace = code_part[len(code_part_trimmed) :]

        # Parse code part with highlighting
        if code_part_trimmed:
            code_fragments = _create_highlighted_fragments(
                code_part_trimmed, pos, mark_range, em_columns
            )
            fragments.extend(code_fragments)

        # Add whitespace between code and comment
        if code_whitespace:
            fragments.append({"code": code_whitespace, "trailing": "solo"})

        # Add comment with line ending
        comment_with_ending = comment_part + line_ending
        fragments.append({"code": comment_with_ending, "comment": "solo"})

    else:
        # No comment, parse entire remaining as code
        code_part_trimmed = remaining.rstrip()
        trailing_whitespace = remaining[len(code_part_trimmed) :]

        # Parse code part with highlighting
        if code_part_trimmed:
            code_fragments = _create_highlighted_fragments(
                code_part_trimmed, pos, mark_range, em_columns
            )
            fragments.extend(code_fragments)

        # Add trailing whitespace and line ending
        trailing_content = trailing_whitespace + line_ending
        if trailing_content:
            fragments.append({"code": trailing_content, "trailing": "solo"})

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

    # Convert em_columns to positions and group consecutive ones
    em_ranges = _columns_to_ranges(em_columns, start_pos, len(text))
    for start, end in em_ranges:
        for i in range(start - start_pos, end - start_pos):
            if 0 <= i < len(text):
                em_positions.add(i)

    # Create fragments using unified logic
    return _create_fragments_with_highlighting(text, mark_positions, em_positions)


def _columns_to_ranges(columns, start_pos, text_len):
    """Convert list of column positions to consecutive ranges."""
    if not columns:
        return []

    # Filter and sort columns that are within our text range
    valid_cols = sorted(
        [col for col in columns if start_pos <= col < start_pos + text_len]
    )
    if not valid_cols:
        return []

    ranges = []
    start = valid_cols[0]
    end = start + 1

    for col in valid_cols[1:]:
        if col == end:
            # Consecutive column, extend range
            end = col + 1
        else:
            # Gap found, close current range and start new one
            ranges.append((start, end))
            start = col
            end = col + 1

    # Close the last range
    ranges.append((start, end))
    return ranges


def _create_fragments_with_highlighting(text, mark_positions, em_positions):
    """Create fragments with mark/em highlighting using beg/mid/fin/solo logic."""
    if not text:
        return []

    # Find all boundaries (start/end of mark and em regions)
    boundaries = {0, len(text)}

    # Add mark boundaries
    mark_ranges = _positions_to_ranges(mark_positions)
    for start, end in mark_ranges:
        boundaries.add(start)
        boundaries.add(end)

    # Add em boundaries
    em_ranges = _positions_to_ranges(em_positions)
    for start, end in em_ranges:
        boundaries.add(start)
        boundaries.add(end)

    boundaries = sorted(boundaries)
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


def _positions_to_ranges(positions):
    """Convert a set of positions to a list of (start, end) ranges."""
    if not positions:
        return []

    sorted_positions = sorted(positions)
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

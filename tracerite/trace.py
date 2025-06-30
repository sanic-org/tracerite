import ast
import inspect
import re
import sys
import traceback as _traceback_module
from pathlib import Path
from textwrap import dedent
from urllib.parse import quote

from .inspector import extract_variables
from .logging import logger

# Will be set to an instance if loaded as an IPython extension by %load_ext
ipython = None

# Locations considered to be bug-free
libdir = re.compile(r"/usr/.*|.*(site-packages|dist-packages).*")


def _find_caret_position(line, start_col, end_col):
    """
    Find the exact highlighting information within a line segment using AST parsing.
    Returns information about what parts of the code should be highlighted.

    Returns a dict with highlighting information, or None if no specific highlighting is appropriate.
    The dict can contain:
    - 'type': 'caret' for single character, 'range' for a range, 'ranges' for multiple ranges
    - 'offset': single character offset (for 'caret' type)
    - 'start', 'end': range bounds (for 'range' type)
    - 'highlights': list of (start, end) tuples (for 'ranges' type)
    """
    try:
        # Extract the segment that contains the error
        segment = line.strip()
        if not segment:
            return None

        # Try to parse the segment as an expression
        try:
            # Wrap in parentheses to make it parseable as an expression
            tree = ast.parse(f"({segment})", mode="eval")
        except SyntaxError:
            try:
                # Try as a statement
                tree = ast.parse(segment, mode="exec")
            except SyntaxError:
                return None

        # Find the best node to analyze
        if hasattr(tree, "body") and tree.body:
            if hasattr(tree, "body") and len(tree.body) > 0:
                if hasattr(tree.body[0], "value"):
                    # Expression statement
                    node = tree.body[0].value
                else:
                    node = tree.body[0]
            else:
                return None
        elif hasattr(tree, "body"):
            node = tree.body
        else:
            return None

        # Calculate relative position within the segment
        indent_len = len(line) - len(line.lstrip())
        segment_start_col = start_col - indent_len
        segment_end_col = end_col - indent_len

        # Find the most appropriate highlighting based on AST node type
        highlight_info = _find_ast_highlights(
            node, segment_start_col, segment_end_col, segment
        )

        return highlight_info

    except Exception:
        # Don't highlight if we can't parse properly
        return None


def _find_ast_highlights(node, start_col, end_col, segment):
    """
    Find the best highlighting information for an AST node.
    Returns a dict with highlighting info, or None if no specific highlighting is appropriate.
    """
    lines = segment.splitlines()
    if not lines:
        return None

    def normalize_col(lineno, col_offset):
        """Convert byte offset to character offset"""
        if lineno <= 0 or lineno > len(lines):
            return col_offset
        line = lines[lineno - 1]
        return len(line.encode("utf-8")[:col_offset].decode("utf-8", errors="replace"))

    def make_range_highlight(start, end):
        """Create a range highlight relative to error start"""
        return {
            "type": "range",
            "start": max(0, start - start_col),
            "end": max(0, end - start_col),
        }

    def make_caret_highlight(pos):
        """Create a single character caret highlight relative to error start"""
        return {"type": "caret", "offset": max(0, pos - start_col)}

    # Handle different AST node types appropriately

    # Function calls - highlight the function name
    if isinstance(node, ast.Call):
        if hasattr(node.func, "col_offset") and hasattr(node.func, "end_col_offset"):
            func_start = normalize_col(1, node.func.col_offset)
            func_end = normalize_col(1, node.func.end_col_offset)
            return make_range_highlight(func_start, func_end)
        elif hasattr(node.func, "end_col_offset"):
            # Fallback: highlight up to the opening parenthesis
            paren_pos = normalize_col(1, node.func.end_col_offset)
            while paren_pos < len(segment) and segment[paren_pos] != "(":
                paren_pos += 1
            if paren_pos < len(segment) and segment[paren_pos] == "(":
                return make_caret_highlight(paren_pos)

    # Binary operations - highlight just the operator
    elif isinstance(node, ast.BinOp):
        if hasattr(node.left, "end_col_offset") and hasattr(node.right, "col_offset"):
            left_end = normalize_col(1, node.left.end_col_offset)
            right_start = normalize_col(1, node.right.col_offset)

            # Find the operator between left and right
            op_start = left_end
            while (
                op_start < right_start
                and op_start < len(segment)
                and segment[op_start].isspace()
            ):
                op_start += 1

            op_end = op_start
            while (
                op_end < right_start
                and op_end < len(segment)
                and not segment[op_end].isspace()
            ):
                op_end += 1

            if op_start < len(segment) and op_end > op_start:
                return make_range_highlight(op_start, op_end)

    # Comparison operations - highlight the first operator
    elif isinstance(node, ast.Compare):
        if hasattr(node.left, "end_col_offset") and node.comparators:
            left_end = normalize_col(1, node.left.end_col_offset)
            if hasattr(node.comparators[0], "col_offset"):
                right_start = normalize_col(1, node.comparators[0].col_offset)

                # Find the comparison operator
                op_start = left_end
                while (
                    op_start < right_start
                    and op_start < len(segment)
                    and segment[op_start].isspace()
                ):
                    op_start += 1

                op_end = op_start
                while (
                    op_end < right_start
                    and op_end < len(segment)
                    and not segment[op_end].isspace()
                ):
                    op_end += 1

                if op_start < len(segment) and op_end > op_start:
                    return make_range_highlight(op_start, op_end)

    # Attribute access - highlight the attribute name
    elif isinstance(node, ast.Attribute):
        if hasattr(node, "end_col_offset") and hasattr(node.value, "end_col_offset"):
            # Highlight the attribute name part
            dot_pos = normalize_col(1, node.value.end_col_offset)
            attr_end = normalize_col(1, node.end_col_offset)

            # Find the dot and attribute name
            while dot_pos < len(segment) and segment[dot_pos] != ".":
                dot_pos += 1

            if dot_pos < len(segment) and segment[dot_pos] == ".":
                attr_start = dot_pos + 1
                return make_range_highlight(attr_start, attr_end)

    # Subscript operations - highlight the subscript part
    elif isinstance(node, ast.Subscript):
        if hasattr(node.value, "end_col_offset") and hasattr(node, "end_col_offset"):
            bracket_start = normalize_col(1, node.value.end_col_offset)
            subscript_end = normalize_col(1, node.end_col_offset)

            # Find the opening bracket
            while bracket_start < len(segment) and segment[bracket_start] != "[":
                bracket_start += 1

            if bracket_start < len(segment) and segment[bracket_start] == "[":
                return make_range_highlight(bracket_start, subscript_end)

    # Name references - highlight the whole name
    elif isinstance(node, ast.Name):
        if hasattr(node, "col_offset") and hasattr(node, "end_col_offset"):
            name_start = normalize_col(1, node.col_offset)
            name_end = normalize_col(1, node.end_col_offset)
            return make_range_highlight(name_start, name_end)

    # Constants - highlight the literal value
    elif isinstance(node, ast.Constant):
        if hasattr(node, "col_offset") and hasattr(node, "end_col_offset"):
            const_start = normalize_col(1, node.col_offset)
            const_end = normalize_col(1, node.end_col_offset)
            return make_range_highlight(const_start, const_end)

    # List/tuple/dict literals - highlight the opening bracket/brace
    elif isinstance(node, (ast.List, ast.Tuple, ast.Dict, ast.Set)):
        if hasattr(node, "col_offset"):
            start_pos = normalize_col(1, node.col_offset)
            if start_pos < len(segment):
                bracket = segment[start_pos]
                if bracket in "([{":
                    return make_caret_highlight(start_pos)

    # Boolean operations - highlight the operator (and/or)
    elif isinstance(node, ast.BoolOp):
        if (
            hasattr(node, "col_offset")
            and len(node.values) >= 2
            and hasattr(node.values[0], "end_col_offset")
            and hasattr(node.values[1], "col_offset")
        ):
            left_end = normalize_col(1, node.values[0].end_col_offset)
            right_start = normalize_col(1, node.values[1].col_offset)

            # Find 'and' or 'or' between the values
            op_start = left_end
            while (
                op_start < right_start
                and op_start < len(segment)
                and segment[op_start].isspace()
            ):
                op_start += 1

            if op_start < len(segment):
                if segment[op_start : op_start + 3] == "and":
                    return make_range_highlight(op_start, op_start + 3)
                elif segment[op_start : op_start + 2] == "or":
                    return make_range_highlight(op_start, op_start + 2)

    # Unary operations - highlight the operator
    elif (
        isinstance(node, ast.UnaryOp)
        and hasattr(node, "col_offset")
        and hasattr(node.operand, "col_offset")
    ):
        op_start = normalize_col(1, node.col_offset)
        operand_start = normalize_col(1, node.operand.col_offset)

        if op_start < operand_start:
            return make_range_highlight(op_start, operand_start)

    # For complex expressions or unsupported types, don't highlight specifically
    return None


def extract_chain(exc=None, **kwargs) -> list:
    """Extract information on current exception."""
    chain = []
    exc = exc or sys.exc_info()[1]
    while exc:
        chain.append(exc)
        # Follow the explicit cause chain first, then context if no cause
        exc = exc.__cause__ or exc.__context__
        if exc and hasattr(exc, "__suppress_context__") and exc.__suppress_context__:
            # If the context is suppressed, we stop here
            break
    # Newest exception first
    return [extract_exception(e, **(kwargs if e is chain[0] else {})) for e in chain]


def extract_exception(e, *, skip_outmost=0, skip_until=None) -> dict:
    tb = e.__traceback__
    try:
        tb = inspect.getinnerframes(tb)
    except IndexError:  # Bug in inspect internals, find_source()
        logger.exception("Bug in inspect?")
        tb = []
    if skip_until:
        for i, frame in enumerate(tb):
            if skip_until in frame.filename:
                skip_outmost = i
                break
    tb = tb[skip_outmost:]
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
        frames = extract_frames(tb, suppress_inner=suppress, exc=e)
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


def extract_frames(tb, suppress_inner=False, exc=None) -> list:
    if not exc:
        return []

    frames = []

    try:
        # Use TracebackException to get enhanced traceback information
        te = _traceback_module.TracebackException.from_exception(
            exc, capture_locals=True
        )
        stack = te.stack

        if not stack:
            return []

        # Find the frame that should be highlighted as the bug location
        # - The innermost non-library frame, or if not found, the innermost frame
        bug_frame_idx = None
        for i in reversed(range(len(stack))):
            summary = stack[i]
            if summary.line and not libdir.fullmatch(summary.filename):
                bug_frame_idx = i
                break
        if bug_frame_idx is None:
            bug_frame_idx = len(stack) - 1

        for i, summary in enumerate(stack):
            # Check for traceback hiding
            if (
                hasattr(summary, "locals")
                and summary.locals
                and summary.locals.get("__tracebackhide__", False)
            ):
                continue

            filename = summary.filename
            lineno = summary.lineno
            function = summary.name
            codeline = summary.line

            # Determine relevance
            if i == len(stack) - 1:
                # Exception was raised here
                relevance = "stop" if suppress_inner else "error"
            elif i == bug_frame_idx:
                relevance = "warning"
            else:
                relevance = "call"

            # Extract source code lines
            lines = ""
            start = lineno
            try:
                # Get source lines around the error
                with open(filename, encoding="utf-8") as f:
                    all_lines = f.readlines()

                # Calculate range to show
                start_idx = max(0, lineno - 16)  # Show 15 lines before
                end_idx = min(len(all_lines), lineno + 3)  # Show 2 lines after

                selected_lines = all_lines[start_idx:end_idx]
                lines = dedent("".join(selected_lines))
                start = start_idx + 1

            except (OSError, UnicodeDecodeError):
                # If we can't read the file, at least show the single line from the traceback
                if codeline:
                    lines = codeline
                    start = lineno
                else:
                    lines = ""
                # Skip non-Python modules unless particularly relevant
                if relevance == "call" and not lines:
                    continue

            urls = {}
            location = None

            # Guard ipython.compile usage
            if ipython is not None and hasattr(ipython, "compile"):
                try:
                    ipython_in = ipython.compile._filename_map.get(filename)
                    if ipython_in is not None:
                        location = f"In [{ipython_in}]"
                        filename = None
                except Exception:
                    pass

            # Format filename and create URLs
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
            if location is None and filename is not None:
                split = 0
                if len(filename) > 40:
                    split = filename.rfind("/", 10, len(filename) - 20) + 1
                location = filename[split:]

            # Format function name
            if function == "<module>":
                function = None
            elif function and hasattr(summary, "locals") and summary.locals:
                # Add class name to methods (if self or cls is the first local variable)
                try:
                    for n, v in summary.locals.items():
                        if n in ("self", "cls") and v is not None:
                            cls = v.__class__ if n == "self" else v
                            function = f"{cls.__name__}.{function}"
                            break
                except Exception:
                    pass
                # Remove long module paths (keep last two items)
                function = ".".join(function.split(".")[-2:])

            # Extract variables from locals
            variables = []
            if hasattr(summary, "locals") and summary.locals:
                variables = extract_variables(summary.locals, lines)

            # Parse lines into fragments for enhanced highlighting
            fragments = []
            if lines:
                # Determine error position for fragment parsing
                error_line_in_context = (
                    lineno - start + 1
                )  # Convert to 1-based index within the lines context
                start_col = getattr(summary, "colno", None)
                end_col = getattr(summary, "end_colno", None)
                end_line = getattr(summary, "end_lineno", None)

                # Convert end_line to context-relative if it exists
                if end_line is not None:
                    end_line_in_context = end_line - start + 1
                else:
                    end_line_in_context = None

                fragments = _parse_lines_to_fragments(
                    lines,
                    error_line_in_context,
                    start_col,
                    end_col,
                    end_line_in_context,
                    em_columns=None,  # TODO: Add AST analysis for emphasis
                )

            # Create frame info
            frameinfo = {
                "id": f"tb-{i}",  # Use index as stable ID
                "relevance": relevance,
                "filename": filename,
                "location": location,
                "codeline": codeline.strip() if codeline else None,
                "lineno": lineno,
                "linenostart": start,
                "lines": lines,
                "fragments": fragments,  # New fragment-based structure
                "function": function,
                "urls": urls,
                "variables": variables,
            }

            # Add precise column positions if available (Python 3.10+)
            if hasattr(summary, "colno") and summary.colno is not None:
                frameinfo["colno"] = summary.colno
                frameinfo["end_colno"] = getattr(summary, "end_colno", None)
                frameinfo["end_lineno"] = getattr(summary, "end_lineno", None)

                # Handle multi-line errors (Python 3.10+)
                if (
                    frameinfo["end_lineno"] is not None
                    and frameinfo["end_lineno"] != lineno
                ):
                    # Multi-line error span
                    frameinfo["is_multiline_error"] = True
                    # Use the multiline highlighting function
                    highlight_info = _find_multiline_highlights(
                        lines,
                        lineno,
                        frameinfo["end_lineno"],
                        summary.colno,
                        frameinfo["end_colno"],
                    )
                    if highlight_info is not None:
                        frameinfo["highlight_info"] = highlight_info
                else:
                    # Single-line error - use existing AST parsing
                    end_colno = frameinfo["end_colno"]
                    if end_colno is None:
                        # Fallback for when end_colno is not available
                        end_colno = summary.colno + len((summary.line or "").rstrip())

                    if codeline:
                        highlight_info = _find_caret_position(
                            codeline, summary.colno, end_colno
                        )
                        if highlight_info is not None:
                            frameinfo["highlight_info"] = highlight_info
            else:
                # For older Python versions (3.8-3.9) without column info,
                # implement whitespace trimming highlighting
                if codeline:
                    stripped_line = codeline.strip()
                    if stripped_line and codeline != stripped_line:
                        # Calculate the amount of leading/trailing whitespace removed
                        leading_spaces = len(codeline) - len(codeline.lstrip())
                        trailing_spaces = (
                            len(codeline.rstrip()) - len(stripped_line)
                            if codeline.rstrip() != stripped_line
                            else 0
                        )

                        highlight_info = {
                            "type": "whitespace_trimmed",
                            "leading_spaces": leading_spaces,
                            "trailing_spaces": trailing_spaces,
                            "trimmed_length": len(stripped_line),
                        }
                        frameinfo["highlight_info"] = highlight_info

            frames.append(frameinfo)

            if suppress_inner and i == bug_frame_idx:
                break

    except Exception:
        logger.exception("Error extracting frames from TracebackException")
        return []

    return frames


def _find_multiline_highlights(lines_text, start_line, end_line, start_col, end_col):
    """
    Find highlighting information for multi-line errors.
    Returns a dict with highlighting info for each line in the error span.
    """
    if not lines_text:
        return None

    lines = lines_text.splitlines()
    if not lines or start_line < 1:
        return None

    # Convert to 0-based indexing for array access
    start_idx = start_line - 1
    end_idx = min(end_line - 1, len(lines) - 1) if end_line else start_idx

    highlights = []

    for line_idx in range(start_idx, end_idx + 1):
        if line_idx >= len(lines):
            break

        line = lines[line_idx]
        line_num = line_idx + 1

        if line_num == start_line and line_num == end_line:
            # Single line case (shouldn't happen here, but handle it)
            highlight = {
                "line": line_num,
                "type": "range",
                "start": start_col or 0,
                "end": end_col or len(line),
            }
        elif line_num == start_line:
            # First line: highlight from start_col to end of line (trimmed)
            trimmed_end = len(line.rstrip())
            highlight = {
                "line": line_num,
                "type": "range",
                "start": start_col or 0,
                "end": trimmed_end,
            }
        elif line_num == end_line:
            # Last line: highlight from start to end_col (with leading whitespace trimmed)
            leading_spaces = len(line) - len(line.lstrip())
            highlight = {
                "line": line_num,
                "type": "range",
                "start": leading_spaces,
                "end": end_col or len(line.rstrip()),
            }
        else:
            # Middle lines: highlight the whole line (with whitespace trimmed)
            leading_spaces = len(line) - len(line.lstrip())
            trimmed_end = len(line.rstrip())
            highlight = {
                "line": line_num,
                "type": "range",
                "start": leading_spaces,
                "end": trimmed_end,
            }

        highlights.append(highlight)

    return {"type": "multiline", "highlights": highlights}


def _parse_lines_to_fragments(
    lines_text, error_line, start_col=None, end_col=None, end_line=None, em_columns=None
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

    Returns:
        List of line dictionaries with fragment information
    """
    if not lines_text:
        return []

    lines = lines_text.splitlines()
    if not lines:
        return []

    # Calculate common indentation (dedent)
    common_indent_len = 0
    non_empty_lines = [line for line in lines if line.strip()]
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
            if end_line is not None and end_line != error_line:
                # Multi-line error
                if line_num == error_line:
                    mark_range = (start_col, len(line.rstrip()))
                elif line_num == end_line:
                    mark_range = (len(line) - len(line.lstrip()), end_col)
                elif error_line < line_num < end_line:
                    mark_range = (len(line) - len(line.lstrip()), len(line.rstrip()))
            elif line_num == error_line:
                mark_range = (start_col, end_col)

        # Get emphasis columns for this line
        em_cols = em_columns.get(line_num, []) if em_columns else []

        # Parse line into fragments
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

    fragments = []
    pos = 0

    # Handle dedent (common indentation)
    if common_indent_len > 0 and len(line) > common_indent_len:
        dedent_text = line[:common_indent_len]
        fragments.append({"code": dedent_text, "dedent": "solo"})
        pos = common_indent_len

    # Handle additional indentation
    remaining = line[pos:]
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

        # Add comment
        fragments.append({"code": comment_part, "comment": "solo"})

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

        # Add trailing whitespace
        if trailing_whitespace:
            fragments.append({"code": trailing_whitespace, "trailing": "solo"})

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


# Will be set to an instance if loaded as an IPython extension by %load_ext

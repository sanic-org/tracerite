from __future__ import annotations

import inspect
import linecache
import tokenize
from collections import deque
from contextlib import suppress
from pathlib import Path
from urllib.parse import quote

from tracerite.logging import logger

from . import trace_cpy
from .chain_analysis import (
    find_try_block_for_except_line,
    parse_source_for_try_except,
)
from .collect import collect_exception_objects
from .core import (
    COMP_CODE_NAMES,
    Range,
    chain_reason,
    compute_cursor_position,
    create_summary,
    libdir_match,
)
from .fragments import (
    build_frame_ranges,
    count_bracket_depth,
    dedent_lines,
    find_clean_start_line,
    make_trace_id,
    parse_lines_to_fragments,
)
from .syntaxerror import clean_syntax_error_message, extract_enhanced_positions


def digest_exception_chain(
    raw_chain: list[dict],
    *,
    cache: dict | None = None,
) -> list[dict]:
    """Convert the raw chain into digested exception info dicts."""
    return [
        digest_exception(item["exc"], cache=cache, **item["kwargs"])
        for item in raw_chain
    ]


def digest_exception(
    e,
    *,
    skip_outmost=0,
    skip_until=None,
    cache: dict | None = None,
) -> dict:
    """Digest one exception into raw frames and metadata.

    The returned dict still contains the live exception object under the
    private ``_exc`` key so that finalization passes can use it later.
    """
    raw_tb = e.__traceback__
    try:
        tb = inspect.getinnerframes(raw_tb)
    except IndexError:  # Bug in inspect internals, find_source()
        logger.exception("Bug in inspect?")
        tb = []
        raw_tb = None

    # For SyntaxError, check if the error is in user code (notebook cell or matching skip_until)
    syntax_frame = None
    is_syntax_error = isinstance(e, SyntaxError)
    if is_syntax_error:
        syntax_frame = extract_syntax_error_frame(e)
        if syntax_frame:
            is_user_code = is_notebook_cell(e.filename) or (
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
    if is_syntax_error:
        message = clean_syntax_error_message(message)
    summary = create_summary(message)
    f = chain_reason(e)

    try:
        frames = digest_frames(tb, raw_tb, except_block=(f != "none"), cache=cache)
        # For SyntaxError, add the synthetic frame showing the problematic code
        if syntax_frame:
            frames.append(syntax_frame)
    except Exception:  # pragma: no cover
        logger.exception("Error extracting traceback")
        frames = None

    # Determine if this is a "stop" type exception (BaseException or ExceptionGroup)
    is_stop_type = not isinstance(e, Exception) or is_exception_group(e)

    result = {
        "type": type(e).__name__,
        "message": message,
        "summary": summary,
        "from": f,
        "repr": repr(e),
        "frames": frames or [],
        "suppress_inner": is_stop_type,
        "_exc": e,
    }

    # Extract subexceptions for ExceptionGroups (Python 3.11+)
    subexceptions = extract_subexceptions(
        e,
        skip_outmost=skip_outmost,
        skip_until=skip_until,
        cache=cache,
    )
    if subexceptions:
        result["subexceptions"] = subexceptions

    return result


def extract_subexceptions(
    e,
    *,
    skip_outmost=0,
    skip_until=None,
    cache: dict | None = None,
) -> list[list[dict]] | None:
    """Return parallel raw exception chains for an ExceptionGroup, or None."""
    if not is_exception_group(e):
        return None

    subexceptions = e.exceptions
    if not subexceptions:
        return None

    parallel_chains = []
    for sub_exc in subexceptions:
        sub_chain = extract_subexception_chain(
            sub_exc,
            skip_outmost=skip_outmost,
            skip_until=skip_until,
            cache=cache,
        )
        if sub_chain:  # pragma: no cover
            parallel_chains.append(sub_chain)

    return parallel_chains if parallel_chains else None


def extract_subexception_chain(
    exc,
    *,
    skip_outmost=0,
    skip_until=None,
    cache: dict | None = None,
) -> list[dict]:
    """Extract the raw exception chain for one ExceptionGroup subexception."""
    chain = collect_exception_objects(exc)
    kwargs = {"skip_outmost": skip_outmost, "skip_until": skip_until}
    return [
        digest_exception(e, cache=cache, **(kwargs if e is chain[-1] else {}))
        for e in chain
    ]


def is_notebook_cell(filename):
    """Check if the filename corresponds to a Jupyter notebook cell."""
    from . import ipython

    try:
        return filename in ipython.compile._filename_map  # type: ignore[attr-defined]
    except (AttributeError, KeyError, TypeError):
        return False


def is_exception_group(e: BaseException) -> bool:
    """Check if exception is an ExceptionGroup (Python 3.11+)."""
    # Check for BaseExceptionGroup which is the base class for both
    # ExceptionGroup and BaseExceptionGroup
    return hasattr(e, "exceptions") and isinstance(
        getattr(e, "exceptions", None), (tuple, list)
    )


def find_except_start_for_line(
    frame,
    lineno: int,
    *,
    cache: dict | None = None,
) -> int | None:
    """Return the 'except' line number if lineno is inside a handler."""

    try:
        filename = frame.f_code.co_filename
        blocks = parse_source_for_try_except(filename, _cache=cache)
        # Find the innermost except block containing this line
        block = find_try_block_for_except_line(blocks, lineno)
        if block:
            return block.except_start
    except Exception:  # pragma: no cover
        pass
    return None


def get_source_lines_from_code(code, lineno: int, end_lineno: int | None = None):
    """Fallback source-line retrieval for code objects (e.g. REPL, exec)."""
    # Python 3.13+ has linecache._getline_from_code for interactive code
    if not hasattr(linecache, "_getline_from_code"):
        return None, None  # pragma: no cover

    # First, check if we can get the error line at all
    error_line = linecache._getline_from_code(code, lineno)
    if not error_line:
        return None, None

    first_lineno = code.co_firstlineno
    is_module = code.co_name in COMP_CODE_NAMES

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
    frame,
    lineno,
    end_lineno=None,
    *,
    notebook_cell=False,
    except_block=False,
    cache: dict | None = None,
):
    try:
        lines, start = _get_source_from_frame(frame, cache=cache)
        except_start = (
            find_except_start_for_line(frame, lineno, cache=cache)
            if except_block
            else None
        )

        lines, start = slice_source_context(
            lines, start, lineno, end_lineno, notebook_cell, except_start
        )

        error_idx, end_idx = _error_indices(lineno, end_lineno, start)
        if not valid_error_position(lines, error_idx):
            return "", lineno, "", None

        error_indent = _line_indent(lines[error_idx])
        lines, error_idx, end_idx, start = _trim_leading_lines(
            lines, error_idx, end_idx, start, error_indent
        )
        lines = _trim_trailing_lines(lines, end_idx, error_indent)

        lines, common_indent = dedent_lines(lines)
        return "".join(lines), start, common_indent, except_start
    except OSError:
        return *fallback_source_lines(frame, lineno, end_lineno), None


def _get_source_from_frame(frame, *, cache: dict | None = None):
    """Get source lines and starting line number for a frame."""
    code = frame.f_code
    key = ("source_block", code)
    if cache is not None and key in cache:
        stored_lines, start = cache[key]
        return list(stored_lines), start

    lines, start = inspect.getsourcelines(frame)
    if start == 0:
        start = 1

    if cache is not None:
        cache[key] = (tuple(lines), start)
    return lines, start


def slice_source_context(lines, start, lineno, end_lineno, notebook_cell, except_start):
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

    slice_start = find_clean_start_line(lines, slice_start)
    lines = lines[slice_start:slice_end]
    start += slice_start

    return trim_to_except_line(lines, start, except_start)


def trim_to_except_line(lines, start, except_start):
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


def valid_error_position(lines, error_idx):
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
    bracket_depth = count_bracket_depth("".join(lines[: end_idx + 1]))
    while trim_after < len(lines):
        line = lines[trim_after]
        if bracket_depth > 0:  # pragma: no cover
            bracket_depth += count_bracket_depth(line)
            trim_after += 1
            continue
        if line.strip() and _line_indent(line) < error_indent:
            break
        trim_after += 1
    return lines[:trim_after]


def fallback_source_lines(frame, lineno, end_lineno):  # pragma: no cover
    """Fallback source retrieval for interactive code objects."""
    code = frame.f_code if hasattr(frame, "f_code") else frame
    fallback_lines, fallback_start = get_source_lines_from_code(
        code, lineno, end_lineno
    )
    if fallback_lines:
        lines, common_indent = dedent_lines(fallback_lines)
        return "".join(lines), fallback_start, common_indent
    return "", lineno, ""


def get_full_source(frame, lineno=None, *, cache: dict | None = None):
    """Return (source, start_line) for a frame, using inspect and fallbacks."""
    code = frame.f_code if hasattr(frame, "f_code") else frame  # pragma: no cover
    key = ("full_source", code)
    if cache is not None and key in cache:
        return cache[key]

    try:
        lines, start = inspect.getsourcelines(frame)
        if start == 0:
            start = 1
        result = ("".join(lines), start)
    except OSError:
        # Fallback: try to get source from code object (Python 3.13+ interactive code)
        # This is tested via subprocess tests in test_tty.py::TestInteractiveSourceRetrieval
        if lineno is None:  # pragma: no cover
            lineno = getattr(frame, "f_lineno", code.co_firstlineno)
        fallback_lines, fallback_start = get_source_lines_from_code(
            code, lineno
        )  # pragma: no cover
        if fallback_lines:  # pragma: no cover
            result = ("".join(fallback_lines), fallback_start)
        else:
            result = (None, None)

    if cache is not None:
        cache[key] = result
    return result


def format_location(filename, lineno, col=1):
    """Return (filename, location_string, urls) for a frame."""
    from . import ipython

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
        location = libdir_match(filename)
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


def get_qualified_function_name(frame, function):
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


def extract_text_from_range(lines: str, mark_range) -> str | None:
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


def find_comprehension_range(lines: str, lineno: int, start: int):
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


def trim_source_to_comprehension(lines: str, lineno: int, start: int):
    """Trim source context to the enclosing comprehension if any."""
    result = find_comprehension_range(lines, lineno, start)
    if result:
        lines_list = lines.splitlines(keepends=True)
        comp_start_idx, comp_end_idx = result
        trimmed = "".join(lines_list[comp_start_idx:comp_end_idx])
        new_start = start + comp_start_idx
        return trimmed, new_start
    return lines, start


def get_variable_source_for_comprehension(
    lines: str, lineno: int, start: int, mark_range
) -> str:
    """Return source code for variable extraction, expanding comprehensions."""
    # Check if we're inside a comprehension
    comp_range = find_comprehension_range(lines, lineno, start)

    if comp_range is not None:
        # Inside a comprehension: use full comprehension text
        lines_list = lines.splitlines(keepends=True)
        comp_start_idx, comp_end_idx = comp_range
        return "".join(lines_list[comp_start_idx:comp_end_idx])

    # Not in a comprehension: use marked text or fall back to full lines
    marked_text = extract_text_from_range(lines, mark_range)
    return marked_text or lines


def extract_emphasis_columns(
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


def build_position_map(raw_tb):
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


def extract_syntax_error_frame(e):
    """Create a synthetic frame dict for a SyntaxError showing the problematic code."""
    if not isinstance(e, SyntaxError):
        return None

    filename, lineno, end_lineno, start_col, end_col = syntax_error_positions(e)
    if not filename or not lineno:
        return None

    notebook_cell = is_notebook_cell(filename)
    lines, lines_list, start, source_from_text = resolve_syntax_error_source(
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
        slice_syntax_error_window(
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

    fragments = parse_lines_to_fragments(lines, mark_range, em_ranges)

    cursor_line, cursor_col = compute_cursor_position(mark_range, em_ranges, start, "")

    fmt_filename, location, urls = format_location(filename, cursor_line, cursor_col)

    codeline = lines_list[error_line_in_context - 1].strip() if lines_list else None

    return {
        "id": make_trace_id(),
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


def syntax_error_positions(e):
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


def resolve_syntax_error_source(e, filename, notebook_cell):
    """Resolve source text for a SyntaxError from linecache or e.text."""
    import linecache

    from . import ipython

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


def slice_syntax_error_window(lines_list, lineno, end_lineno, start, source_from_text):
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
        em_ranges = extract_emphasis_columns(
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
) -> list:
    """Extract finalized frames from a traceback frame list (Python order)."""
    frames = digest_frames(tb, raw_tb, except_block=except_block)
    # Local import to avoid a top-level circular dependency with finalize.py.
    from .finalize import fill_variables, finalize_python_order_frames

    if exc is not None:
        finalize_python_order_frames(frames, exc, exc_message)
    else:
        fill_variables(frames, exc_message)
    return frames


def digest_frames(
    tb, raw_tb=None, *, except_block=False, cache: dict | None = None
) -> list[dict]:
    """Convert a traceback into raw frame dicts without relevances/variables."""
    if not tb:
        return []

    position_map = build_position_map(raw_tb)

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
        frame_info = extract_single_frame(
            frame,
            filename,
            lineno,
            function,
            codeline,
            pos,
            hidden,
            is_last_frame,
            except_block=except_block,
            cache=cache,
        )
        if frame_info is not None:
            frames.append(frame_info)

    return frames


def extract_single_frame(
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
    cache: dict | None = None,
):
    """Extract a single frame's worth of traceback information."""
    pos_end_lineno, start_col, end_col = pos[1], pos[2], pos[3]
    notebook_cell = is_notebook_cell(filename)

    lines, start, original_common_indent, except_start = extract_source_lines(
        frame,
        lineno,
        pos_end_lineno,
        notebook_cell=notebook_cell,
        except_block=except_block,
        cache=cache,
    )

    if not lines and not is_last_frame:
        if hidden:
            # Still include hidden frames with minimal info for chain analysis
            full_source, full_source_start = get_full_source(frame, cache=cache)
            return {
                "id": make_trace_id(),
                "relevance": "call",
                "hidden": True,
                "idframe": id(frame),
                "frame_obj": frame,
                "lineno": lineno,
                "full_source": full_source,
                "full_source_start": full_source_start,
            }
        return None

    full_source, full_source_start = get_full_source(frame, cache=cache)

    lines, start = trim_source_to_comprehension(lines, lineno, start)
    lines_list = lines.splitlines(keepends=True)
    lines, extra_indent = dedent_lines(lines_list)
    lines = "".join(lines)
    total_indent = len(original_common_indent) + len(extra_indent)

    original_filename = filename
    function = get_qualified_function_name(frame, function)

    error_line_in_context = lineno - start + 1
    end_line = pos_end_lineno - start + 1 if pos_end_lineno else None

    frame_range, mark_range = build_frame_ranges(
        lineno,
        pos_end_lineno,
        error_line_in_context,
        end_line,
        start_col,
        end_col,
        total_indent,
        lines,
    )

    em_range = extract_emphasis_columns(
        lines,
        error_line_in_context,
        end_line,
        mark_range.cbeg if mark_range else None,
        mark_range.cend if mark_range else None,
        start,
    )
    fragments = parse_lines_to_fragments(lines, mark_range, em_range)

    cursor_line, cursor_col = compute_cursor_position(
        mark_range, em_range, start, original_common_indent + extra_indent
    )

    filename, location, urls = format_location(
        original_filename, cursor_line, cursor_col
    )

    variable_source = get_variable_source_for_comprehension(
        lines, lineno, start, mark_range
    )

    result = {
        "id": make_trace_id(),
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
        "_except_start": except_start,
    }

    return result

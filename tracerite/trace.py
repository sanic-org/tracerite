from __future__ import annotations

import inspect
import re
import sys
from collections import namedtuple
from contextlib import suppress
from pathlib import Path
from secrets import token_urlsafe
from urllib.parse import quote

from . import trace_cpy
from .inspector import extract_variables
from .logging import logger
from .syntaxerror import clean_syntax_error_message, extract_enhanced_positions

# Position range: lines are 1-based inclusive, columns are 0-based exclusive
Range = namedtuple("Range", ["lfirst", "lfinal", "cbeg", "cend"])

# Will be set to an instance if loaded as an IPython extension by %load_ext
ipython = None

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


def build_chain_header(chain: list[dict]) -> str:
    """Build a header message describing the exception chain."""
    if not chain:
        return ""

    # Chain is oldest-first: chain[0] is first exception, chain[-1] is last (uncaught)
    if len(chain) == 1:
        exc_type = chain[0].get("type", "Exception")
        return f"⚠️ Uncaught {exc_type}"

    # Build from last to first
    parts = [f"⚠️ Uncaught {chain[-1].get('type', 'Exception')}"]

    # Add each previous exception with appropriate joiner
    for i in range(len(chain) - 2, -1, -1):
        exc = chain[i]
        next_exc = chain[i + 1]
        from_type = next_exc.get("from", "none")
        joiner = "from" if from_type == "cause" else "while handling"
        parts.append(f"{joiner} {exc.get('type', 'Exception')}")

    return " ".join(parts)


def extract_chain(exc=None, **kwargs) -> list:
    """Extract information on current exception.

    Returns a list of exception info dicts, ordered from oldest to newest
    (i.e., the original exception first, then any exceptions that occurred
    while handling it or were raised from it).
    """
    chain = []
    exc = exc or sys.exc_info()[1]
    while exc:
        chain.append(exc)
        exc = exc.__cause__ or None if exc.__suppress_context__ else exc.__context__
    # Reverse to get oldest first (chain is built newest-first)
    chain = list(reversed(chain))
    result = [extract_exception(e, **(kwargs if e is chain[-1] else {})) for e in chain]
    # Deduplicate variable inspectors: only keep variables for the last occurrence
    # of each (filename, function) pair across the entire chain
    _deduplicate_variables(result)
    return result


def _deduplicate_variables(chain: list) -> None:
    """Remove duplicate variables from inspectors, showing each only once.

    Variables are only shown if they appear in the frame's highlighted code
    (the lines indicated by the error range, expanded to include full
    comprehensions). If a variable appears in multiple frames' highlighted
    code (same filename/function), it's only shown in the last frame where
    it appears.
    """

    def _get_highlighted_lines(frame: dict) -> str:
        """Extract the highlighted lines from a frame based on its range.

        Expands to include full comprehension if error is inside one.
        """
        lines = frame.get("lines", "")
        range_obj = frame.get("range")
        if not range_obj or not lines:
            return lines  # Fall back to all lines if no range

        start = frame.get("linenostart", 1)
        lfirst, lfinal = range_obj.lfirst, range_obj.lfinal

        # Check if error is inside a comprehension - if so, return full comprehension
        comp_range = _find_comprehension_range(lines, lfirst, start)
        if comp_range is not None:
            # Error is inside a comprehension - return full lines (already trimmed to comprehension)
            return lines

        # No comprehension, return just the highlighted lines
        lines_list = lines.splitlines()

        # Convert to 0-based indices relative to displayed lines
        first_idx = lfirst - start
        final_idx = lfinal - start + 1

        if first_idx < 0 or first_idx >= len(lines_list):
            return lines  # Fall back if range is invalid

        return "\n".join(lines_list[first_idx:final_idx])

    def _variable_in_code(name: str, lines: str) -> bool:
        """Check if a variable name appears in the code as a word."""
        return bool(re.search(rf"\b{re.escape(name)}\b", lines))

    # First pass: collect frames by (filename, function) key
    # Maps key -> list of (exception_idx, frame_idx)
    frame_groups: dict[tuple, list[tuple[int, int]]] = {}
    for ei, exc in enumerate(chain):
        for fi, frame in enumerate(exc.get("frames", [])):
            if frame.get("relevance") == "call":
                continue
            key = (frame.get("filename"), frame.get("function"))
            if key not in frame_groups:
                frame_groups[key] = []
            frame_groups[key].append((ei, fi))

    # Second pass: for each group, determine which variables to show in each frame
    for _key, occurrences in frame_groups.items():
        # For each variable, find the LAST frame where it appears in highlighted code
        # variable_name -> (exception_idx, frame_idx) of last appearance in highlighted code
        last_appearance: dict[str, tuple[int, int]] = {}

        for ei, fi in occurrences:
            frame = chain[ei]["frames"][fi]
            highlighted = _get_highlighted_lines(frame)
            for v in frame.get("variables", []):
                if v.name and _variable_in_code(v.name, highlighted):
                    # Update to this frame (later frames overwrite earlier)
                    last_appearance[v.name] = (ei, fi)

        # Now filter each frame's variables: keep only if this is the last appearance
        for ei, fi in occurrences:
            frame = chain[ei]["frames"][fi]
            frame["variables"] = [
                v
                for v in frame.get("variables", [])
                if v.name and last_appearance.get(v.name) == (ei, fi)
            ]


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
    try:
        # KeyboardErrors and such need not be reported all the way
        suppress = not isinstance(e, Exception)
        frames = extract_frames(tb, raw_tb, suppress_inner=suppress)
        # For SyntaxError, add the synthetic frame showing the problematic code
        if syntax_frame:
            # Demote the previous frame (compile, exec, etc.) to call only
            if frames and frames[-1]["relevance"] == "error":
                frames[-1]["relevance"] = "call"
            frames.append(syntax_frame)
    except Exception:
        logger.exception("Error extracting traceback")
        frames = None
    return {
        "type": type(e).__name__,
        "message": message,
        "summary": summary,
        # "from" describes how this exception relates to its cause:
        # - "cause": explicitly raised from another (raise X from Y)
        # - "context": occurred while handling another
        # - "none": root exception (no prior cause)
        "from": ("cause" if e.__cause__ else "context" if e.__context__ else "none"),
        "repr": repr(e),
        "frames": frames or [],
    }


def _is_notebook_cell(filename):
    """Check if the filename corresponds to a Jupyter notebook cell."""
    try:
        return filename in ipython.compile._filename_map  # type: ignore[attr-defined]
    except (AttributeError, KeyError, TypeError):
        return False


def _find_except_start_for_line(frame, lineno: int) -> int | None:
    """If lineno is inside an except handler, return the except line number.

    Uses AST analysis to find if the given line is within an except block.
    Returns the line number of the 'except' keyword, or None if not in an except block.
    """
    from .chain_analysis import parse_source_for_try_except

    try:
        filename = frame.f_code.co_filename
        blocks = parse_source_for_try_except(filename)
        for block in blocks:
            if block.contains_in_except(lineno):
                return block.except_start
    except Exception:  # pragma: no cover
        pass
    return None


def extract_source_lines(frame, lineno, end_lineno=None, *, notebook_cell=False):
    try:
        lines, start = inspect.getsourcelines(frame)
        if start == 0:
            start = 1

        # Check if lineno is inside an except handler BEFORE trimming
        # This ensures we include the except line even for notebook cells
        except_start = _find_except_start_for_line(frame, lineno)

        # For notebook cells, show only the error lines (no context)
        # For regular files, show 10 lines before and 2 lines after
        # Exception: if we're in an except block, ensure except line is included
        if notebook_cell:
            if except_start is not None and except_start >= start:
                # In except block: include from except line to lineno
                lines_before = lineno - except_start
            else:
                lines_before = 0
            lines_after = (end_lineno - lineno) if end_lineno else 0
        else:
            lines_before = 10
            lines_after = (end_lineno - lineno + 2) if end_lineno else 2
        lines = lines[
            max(0, lineno - start - lines_before) : max(
                0, lineno - start + lines_after + 1
            )
        ]
        start += max(0, lineno - start - lines_before)

        # If lineno is inside an except handler, trim to start from the except line
        # (For non-notebook cells, this may still trim if lines_before > distance to except)
        if except_start is not None and except_start > start:
            skip = except_start - start
            if skip < len(lines):  # pragma: no branch
                lines = lines[skip:]
                start = except_start

        # Calculate error line position
        error_idx = lineno - start
        end_idx = (end_lineno - start) if end_lineno else error_idx

        # Safety check: ensure error_idx is valid
        if not lines or error_idx < 0 or error_idx >= len(lines):
            return "", lineno, ""

        # Get the indentation of the first marked line (error line) before any dedenting
        error_indent = 0
        error_line = lines[error_idx]
        error_indent = len(error_line) - len(error_line.lstrip(" \t"))

        # Trim leading lines that have more indentation than error line
        while lines and error_idx > 0:
            first_line = lines[0]
            if first_line.strip():
                first_indent = len(first_line) - len(first_line.lstrip(" \t"))
                if first_indent <= error_indent:
                    break  # This line has same or less indent, keep it
            start += 1
            lines.pop(0)
            error_idx -= 1
            end_idx -= 1

        # Trim trailing lines with less indentation than the error line
        # (hides external structures like else/except that aren't relevant)
        trim_after = end_idx + 1
        while trim_after < len(lines):
            line = lines[trim_after]
            # Keep empty lines, but check non-empty lines for indentation
            if line.strip():
                line_indent = len(line) - len(line.lstrip(" \t"))
                if line_indent < error_indent:
                    break  # Found a line with less indent, trim from here
            trim_after += 1
        lines = lines[:trim_after]

        # Calculate common indentation and dedent AFTER pruning
        common_indent = _calculate_common_indent(lines)
        lines = [ln.removeprefix(common_indent) for ln in lines]

        return "".join(lines), start, common_indent
    except OSError:
        return "", lineno, ""  # Source not available (non-Python module)


def _get_full_source(frame):
    """Get the full source code for a frame using inspect.

    Returns (source, start_line) tuple. This works with any source Python
    knows about, including notebook cells and exec'd strings.
    """
    try:
        lines, start = inspect.getsourcelines(frame)
        if start == 0:
            start = 1
        return "".join(lines), start
    except OSError:
        return None, None


def _libdir_match(path):
    """Check if path is in a library directory and return the short suffix if so."""
    m = libdir.fullmatch(path)
    if m:
        return next((g for g in m.groups() if g), "")
    return None


def format_location(filename, lineno):
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
        urls["VS Code"] = f"vscode://file/{quote(fn.as_posix())}:{lineno}"
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


def _expand_source_for_comprehension(lines: str, lineno: int, start: int) -> str:
    """Expand source to include full comprehension/generator expression if error is inside one.

    This helps show relevant variables like the iterator source (e.g., `data` in `for item in data`).

    Args:
        lines: The source code snippet
        lineno: The 1-based line number where the error occurred
        start: The 1-based starting line number of the snippet

    Returns:
        Source code that includes the full comprehension, or original lines if not in one.
    """
    result = _find_comprehension_range(lines, lineno, start)
    if result:
        lines_list = lines.splitlines(keepends=True)
        comp_start, comp_end = result
        return "".join(lines_list[comp_start:comp_end])
    return lines


def _find_comprehension_range(lines: str, lineno: int, start: int):
    """Find the line range of a comprehension containing the error line.

    Args:
        lines: The source code snippet
        lineno: The 1-based line number where the error occurred
        start: The 1-based starting line number of the snippet

    Returns:
        Tuple of (start_idx, end_idx) as 0-based indices into lines_list,
        or None if error is not inside a comprehension.
    """
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
    """Trim source context to just the comprehension if error is inside one.

    Args:
        lines: The source code snippet
        lineno: The 1-based line number where the error occurred
        start: The 1-based starting line number of the snippet

    Returns:
        Tuple of (trimmed_lines, new_start) where new_start is adjusted line number,
        or (lines, start) if not inside a comprehension.
    """
    result = _find_comprehension_range(lines, lineno, start)
    if result:
        lines_list = lines.splitlines(keepends=True)
        comp_start_idx, comp_end_idx = result
        trimmed = "".join(lines_list[comp_start_idx:comp_end_idx])
        new_start = start + comp_start_idx
        return trimmed, new_start
    return lines, start


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
            if f.code_context and _libdir_match(Path(f.filename).as_posix()) is None
        ),
        tb[-1],
    ).frame


def _extract_syntax_error_frame(e):
    """Create a synthetic frame dict for a SyntaxError showing the problematic code."""
    if not isinstance(e, SyntaxError):
        return None

    filename = e.filename
    lineno = e.lineno
    if not filename or not lineno:
        return None

    # SyntaxError attributes: filename, lineno, offset, text, end_lineno, end_offset
    end_lineno = getattr(e, "end_lineno", None) or lineno
    # offset is 1-based in SyntaxError, convert to 0-based for our Range
    start_col = (e.offset - 1) if e.offset else 0
    end_col = getattr(e, "end_offset", None)

    if end_col:
        end_col = end_col - 1  # Convert to 0-based
        # Ensure we have at least one character highlighted
        if end_col <= start_col and end_lineno == lineno:
            end_col = start_col + 1
    else:
        end_col = start_col + 1  # Default to single character

    assert start_col is not None and end_col is not None

    # Get source lines
    notebook_cell = _is_notebook_cell(filename)
    lines = None
    all_lines = None
    start = 1  # For SyntaxErrors, we want full source to show bracket matches etc.

    # Try to get source from the file or notebook
    try:
        import linecache

        # For notebook cells, try to get from IPython's cache
        if notebook_cell and ipython:
            try:
                cell_source = ipython.compile._filename_map.get(filename)
                if cell_source is not None:
                    # Get the cell content from the history
                    all_lines = linecache.getlines(filename)
                    if all_lines:
                        # For SyntaxErrors, get full source to enable bracket matching
                        lines = "".join(all_lines)
            except Exception:
                pass

        # Fallback: try linecache directly
        if not lines:
            all_lines = linecache.getlines(filename)
            if all_lines:
                # For SyntaxErrors, get full source to enable bracket matching
                lines = "".join(all_lines)

        # Last resort: use the text attribute from SyntaxError itself
        if not lines and e.text:
            lines = e.text if e.text.endswith("\n") else e.text + "\n"
            start = lineno
    except Exception:
        if e.text:
            lines = e.text if e.text.endswith("\n") else e.text + "\n"
            start = lineno

    if not lines:
        return None

    # Calculate error position within the displayed lines
    error_line_in_context = lineno - start + 1
    end_line = end_lineno - start + 1 if end_lineno else None

    # Calculate common indentation
    lines_list = lines.splitlines(keepends=True)
    common_indent = _calculate_common_indent(lines_list)

    # Try enhanced SyntaxError position extraction for better highlighting
    enhanced_mark, enhanced_em = extract_enhanced_positions(e, lines_list)

    if enhanced_mark:
        # Override lineno/end_lineno with the enhanced range (e.g., from opening bracket)
        lineno = enhanced_mark.lfirst
        end_lineno = enhanced_mark.lfinal

        # Trim source to start from the mark's first line
        lines_list = lines_list[lineno - 1 :]
        lines = "".join(lines_list)
        start = lineno
        common_indent = _calculate_common_indent(lines_list)

        error_line_in_context = 1  # Now lineno is the first line
        end_line = end_lineno - start + 1

        # Adjust enhanced ranges from absolute line numbers to context-relative
        mark_range = Range(
            1,
            enhanced_mark.lfinal - start + 1,
            max(0, enhanced_mark.cbeg - len(common_indent)),
            max(0, enhanced_mark.cend - len(common_indent)),
        )
        # Convert list of em ranges to context-relative
        em_ranges = (
            [
                Range(
                    em.lfirst - start + 1,
                    em.lfinal - start + 1,
                    max(0, em.cbeg - len(common_indent)),
                    max(0, em.cend - len(common_indent)),
                )
                for em in enhanced_em
            ]
            if enhanced_em
            else None
        )
    else:
        # Fallback to Python's positions
        # Adjust columns for dedenting
        adjusted_start_col = max(0, start_col - len(common_indent))
        adjusted_end_col = max(0, end_col - len(common_indent))

        # Create mark range
        mark_range = None
        mark_lfinal = end_line or error_line_in_context
        mark_range = Range(
            error_line_in_context, mark_lfinal, adjusted_start_col, adjusted_end_col
        )

        # Build emphasis range
        em_ranges = _extract_emphasis_columns(
            lines,
            error_line_in_context,
            end_line,
            adjusted_start_col,
            adjusted_end_col,
            start,
        )

    fragments = _parse_lines_to_fragments(lines, mark_range, em_ranges)

    # Format location info (after enhanced positions may have updated lineno)
    fmt_filename, location, urls = format_location(filename, lineno)

    # Get the code line for display
    codeline = lines_list[error_line_in_context - 1].strip() if lines_list else None

    return {
        "id": f"tb-{token_urlsafe(12)}",
        "relevance": "error",
        "filename": fmt_filename,
        "location": location,
        "codeline": codeline,
        "range": Range(lineno, end_lineno or lineno, start_col, end_col)
        if start_col is not None
        else None,
        "linenostart": start,
        "lines": lines,
        "fragments": fragments,
        "function": None,
        "urls": urls,
        "variables": [],
    }


def extract_frames(tb, raw_tb=None, suppress_inner=False) -> list:
    if not tb:
        return []

    position_map = _build_position_map(raw_tb)
    bug_in_frame = _find_bug_frame(tb)

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

        is_last_frame = frame is tb[-1][0]
        is_bug_frame = frame is bug_in_frame
        relevance = _get_frame_relevance(is_last_frame, is_bug_frame, suppress_inner)

        # Extract position information first so we can use it for source extraction
        pos = position_map.get(frame, [None] * 4)
        pos_end_lineno, start_col, end_col = pos[1], pos[2], pos[3]

        # Check if this is a notebook cell (to reduce context)
        notebook_cell = _is_notebook_cell(filename)

        lines, start, original_common_indent = extract_source_lines(
            frame, lineno, pos_end_lineno, notebook_cell=notebook_cell
        )
        if not lines and relevance == "call":
            continue

        # Get full source for chain analysis (AST parsing for try-except matching)
        # This uses inspect which works with any source Python knows about
        full_source, full_source_start = _get_full_source(frame)

        # For comprehensions/generators, trim context to just the expression
        lines, start = _trim_source_to_comprehension(lines, lineno, start)
        # Recalculate common indent after trimming and dedent again if needed
        lines_list = lines.splitlines(keepends=True)
        extra_indent = _calculate_common_indent(lines_list)
        lines = "".join(ln.removeprefix(extra_indent) for ln in lines_list)
        # Total indent removed is original + any extra from trimming
        total_indent = len(original_common_indent) + len(extra_indent)

        # Preserve original filename for chain analysis (needed for AST parsing)
        original_filename = filename
        filename, location, urls = format_location(filename, lineno)
        function = _get_qualified_function_name(frame, function)

        error_line_in_context = lineno - start + 1
        end_line = pos_end_lineno - start + 1 if pos_end_lineno else None

        # Adjust column positions to account for dedenting
        # Python's column numbers are based on the original indented code,
        # but we display dedented code, so we need to subtract total indentation removed
        adjusted_start_col = start_col - total_indent if start_col is not None else None
        adjusted_end_col = end_col - total_indent if end_col is not None else None

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

        # Expand source to include full comprehension for variable inspection
        variable_source = _expand_source_for_comprehension(lines, lineno, start)

        frames.append(
            {
                "id": f"tb-{token_urlsafe(12)}",
                "relevance": relevance,
                "filename": filename,
                "original_filename": original_filename,  # For chain analysis AST parsing
                "location": location,
                "codeline": codeline[0].strip() if codeline else None,
                "range": Range(lineno, pos_end_lineno or lineno, start_col, end_col)
                if start_col is not None
                else None,
                "lineno": lineno,  # Actual error line from traceback (always available)
                "linenostart": start,
                "lines": lines,
                "fragments": fragments,
                "function": function,
                "urls": urls,
                "variables": extract_variables(frame.f_locals, variable_source),
                # Full source for chain analysis (try-except matching via AST)
                "full_source": full_source,
                "full_source_start": full_source_start,
            }
        )

        if suppress_inner and is_bug_frame:
            break

    return frames


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
    """
    Parse lines of code into fragments with mark/em highlighting information.

    Args:
        lines_text: The multi-line string containing code
        mark_range: Range object for mark highlighting (or None)
        em_ranges: Range object or list of Range objects for em highlighting (or None)

    Returns:
        List of line dictionaries with fragment information
    """
    lines = lines_text.splitlines(keepends=True)
    if not lines:
        return []

    common_indent = _calculate_common_indent(lines)

    # Convert both mark and em to position sets using unified logic
    mark_positions = _convert_range_to_positions(mark_range, lines)

    # Handle em_ranges as either a single Range or a list of Ranges
    em_positions = set()
    if em_ranges:
        if isinstance(em_ranges, list):
            for em_range in em_ranges:
                em_positions |= _convert_range_to_positions(em_range, lines)
        else:
            em_positions = _convert_range_to_positions(em_ranges, lines)

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

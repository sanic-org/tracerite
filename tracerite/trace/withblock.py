"""Analysis and source ranges for with/async-with statement errors.

Python 3.12+ marks errors raised by ``__enter__``/``__exit__`` (e.g. TaskGroup
ExceptionGroups) on the with statement's context expression, even when the
block body already ran.  This module detects such frames, tells apart enter
and exit failures, and builds ranges that mark the whole statement with the
failing context expression emphasized.
"""

from __future__ import annotations

import ast
import dis
import re
from typing import TYPE_CHECKING

from tracerite.logging import logger

from .fragments import CodeScanner, find_comment_start, split_line_content

if TYPE_CHECKING:
    from .typing import Range, WithBlock

# Maximum number of with-block body lines shown when exiting an entered block.
MAX_WITH_BLOCK_CONTEXT_LINES = 20

WITH_ENTER_FUNCTIONS = frozenset(("__enter__", "__aenter__"))
WITH_EXIT_FUNCTIONS = frozenset(("__exit__", "__aexit__"))
WITH_ENTER_EXIT_FUNCTIONS = WITH_ENTER_FUNCTIONS | WITH_EXIT_FUNCTIONS

# Every with/async-with statement contains this keyword; used as a cheap
# pre-filter so source is only parsed when a with block can be present.
WITH_KEYWORD = re.compile(r"\bwith\b").search


def detect_with_block_error(
    frame,
    lineno,
    lasti,
    next_func,
    source,
    source_start,
    *,
    is_last_frame=False,
    cache: dict | None = None,
):
    """Detect a frame stopped on a with statement due to enter/exit failure.

    Returns ``(stage, block)`` where stage is ``"enter"`` or ``"exit"`` and
    block is the matching WithBlock, or ``(None, None)`` when the context
    expression itself failed (the block never ran) or the frame is not on a
    with statement header.

    Detection needs source parsing, so it is gated on the only situations
    where it can match: the frame calls an enter/exit function next, or it
    is the last frame (a C-level exit leaves no frame below).  Every other
    frame is a regular call or a context expression failure.  The stage is
    told apart by the next frame in the traceback (the enter or exit
    function being called), falling back to the instruction offset for
    C-level exit functions that leave no Python frame (e.g. lock release).
    """
    if next_func not in WITH_ENTER_EXIT_FUNCTIONS and not is_last_frame:
        return None, None
    if not source or not WITH_KEYWORD(source):
        return None, None
    block = find_with_block(source, source_start, lineno, cache)
    if block is None:
        return None, None
    if next_func in WITH_ENTER_FUNCTIONS:
        return "enter", block
    if next_func in WITH_EXIT_FUNCTIONS:
        return "exit", block
    # No Python enter/exit frame below: a C-level __exit__ may still have
    # raised after the block ran; the frame then stops past the body start.
    body_offset = next(
        (
            ins.offset
            for ins in dis.get_instructions(frame.f_code)
            if (pos := getattr(ins, "positions", None))
            and pos.lineno is not None
            and pos.lineno >= block["body_start"]
        ),
        None,
    )
    if body_offset is not None and lasti is not None and lasti >= body_offset:
        return "exit", block
    return None, None


def block_context_end(stage, block, lineno):
    """End line of the source context window for a with statement frame.

    Exit failures show the whole block body that ran; enter failures cover
    the statement header (never the body that never ran) plus at least the
    default two following lines.  Both capped at MAX_WITH_BLOCK_CONTEXT_LINES.
    """
    if stage == "exit":
        return min(block["block_end"], lineno + MAX_WITH_BLOCK_CONTEXT_LINES)
    if stage == "enter":
        return max(
            lineno + 2,
            min(block["body_start"] - 1, lineno + MAX_WITH_BLOCK_CONTEXT_LINES),
        )
    return None


def find_with_block(
    source: str, start_line: int, lineno: int, cache: dict | None = None
) -> WithBlock | None:
    """Find the innermost with/async-with header covering lineno, or None."""
    key = ("with_blocks", source, start_line)
    cached = cache.get(key) if cache is not None else None
    blocks: list[WithBlock] = (
        cached if cached is not None else _parse_with_blocks(source, start_line)
    )
    if cached is None and cache is not None:
        cache[key] = blocks
    matching = [b for b in blocks if b["header_start"] <= lineno < b["body_start"]]
    if not matching:
        return None
    return min(matching, key=lambda b: b["block_end"] - b["header_start"])


def _parse_with_blocks(source: str, start_line: int) -> list[WithBlock]:
    """Extract with/async-with statements from source with adjusted positions."""
    # inspect.getsourcelines keeps the original indentation of nested
    # functions, which does not parse standalone: dedent by the first line's
    # indent (item column offsets are shifted back below).
    indent = len(source) - len(source.lstrip())
    if indent:
        prefix = source[:indent]
        source = "".join(
            line[indent:] if line.startswith(prefix) else line
            for line in source.splitlines(keepends=True)
        )
    try:
        tree = ast.parse(source)
    except (SyntaxError, ValueError):
        logger.debug("Failed to parse source for with-block analysis")
        return []
    offset = start_line - 1
    blocks: list[WithBlock] = []
    for node in ast.walk(tree):
        if isinstance(node, (ast.With, ast.AsyncWith)) and node.body:
            blocks.append(
                {
                    "header_start": node.lineno + offset,
                    "body_start": node.body[0].lineno + offset,
                    "block_end": (node.end_lineno or node.body[-1].lineno) + offset,
                    "items": [
                        {
                            "lfirst": e.lineno + offset,
                            "lfinal": (e.end_lineno or e.lineno) + offset,
                            "cbeg": e.col_offset + indent,
                            "cend": (e.end_col_offset or e.col_offset + 1) + indent,
                        }
                        for item in node.items
                        for e in (item.context_expr,)
                    ],
                }
            )
    return blocks


def find_with_item_expression(block, lineno, code, lasti):
    """Find the context expression range of the with item that failed, or None.

    A single item is identified trivially.  With multiple comma-separated
    items (which may share a line), the failing instruction identifies the
    item: its position is the failing item's own span on newer CPython,
    while on older versions the Nth enter call belongs to the Nth item and
    the Nth exit call to the Nth item in reverse (exits run innermost
    first).  Returns None rather than guessing when the item cannot be
    identified.
    """
    items = block["items"]
    if len(items) == 1:
        return items[0]
    if lasti is not None:
        enter_offsets, exit_calls, wes_offsets = [], [], []
        for ins in dis.get_instructions(code):
            pos = getattr(ins, "positions", None)
            if pos is None or pos.lineno is None:
                continue
            if ins.offset == lasti:
                for item in items:
                    if (
                        item["lfirst"] <= pos.lineno <= item["lfinal"]
                        and not (
                            pos.lineno == item["lfirst"]
                            and pos.col_offset < item["cbeg"]
                        )
                        and not (
                            pos.lineno == item["lfinal"]
                            and pos.col_offset >= item["cend"]
                        )
                    ):
                        return item
            if not block["header_start"] <= pos.lineno < block["body_start"]:
                continue
            if ins.opname == "BEFORE_WITH" or (
                ins.opname == "LOAD_SPECIAL" and ins.argval == "__enter__"
            ):
                enter_offsets.append(ins.offset)
            elif ins.opname == "WITH_EXCEPT_START":
                wes_offsets.append(ins.offset)
            elif ins.opname == "CALL" and ins.argval in (2, 3):
                exit_calls.append(ins.offset)
        if lasti in enter_offsets:
            # The Nth enter call of the header belongs to the Nth item
            return items[min(enter_offsets.index(lasti), len(items) - 1)]
        # Only calls after the last enter can be exits (item expressions may
        # themselves be calls); the first exit call belongs to the last item.
        last_enter = enter_offsets[-1] if enter_offsets else -1
        for offsets in (exit_calls, wes_offsets):
            offsets = [o for o in offsets if o > last_enter]
            if lasti in offsets:
                index = min(offsets.index(lasti), len(items) - 1)
                return items[len(items) - 1 - index]
    matching = [item for item in items if item["lfirst"] <= lineno <= item["lfinal"]]
    return matching[0] if len(matching) == 1 else None


def build_with_statement_ranges(block, lineno, code, lasti, start, total_indent, lines):
    """Build (frame_range, mark_range, em_range) for a with statement frame.

    The mark covers the entire statement header; the em covers the context
    expression whose enter/exit call failed, when identified.
    """
    lines_list = lines.splitlines(keepends=True)
    first_idx = max(0, block["header_start"] - start)
    last_idx = _statement_end_line(lines_list, first_idx)
    if last_idx is None:
        # Header end not in the window; clamp to the line before the body
        last_idx = min(len(lines_list) - 1, block["body_start"] - 1 - start)

    first_line, _ = split_line_content(lines_list[first_idx])
    cbeg = len(first_line) - len(first_line.lstrip())
    last_line, _ = split_line_content(lines_list[last_idx])
    comment_start = find_comment_start(last_line)
    if comment_start is not None:
        last_line = last_line[:comment_start]
    cend = len(last_line.rstrip())

    frame_range = {
        "lfirst": start + first_idx,
        "lfinal": start + last_idx,
        "cbeg": cbeg + total_indent,
        "cend": cend + total_indent,
    }
    mark_range = {
        "lfirst": first_idx + 1,
        "lfinal": last_idx + 1,
        "cbeg": cbeg,
        "cend": cend,
    }
    em_range = None
    item = find_with_item_expression(block, lineno, code, lasti)
    if item:
        em_range: Range = {
            "lfirst": max(1, item["lfirst"] - start + 1),
            "lfinal": max(1, item["lfinal"] - start + 1),
            "cbeg": max(0, item["cbeg"] - total_indent),
            "cend": max(0, item["cend"] - total_indent),
        }
    return frame_range, mark_range, em_range


def _statement_end_line(lines_list, first_idx):
    """Index of the line where a compound statement header ends (the colon)."""
    scanner = CodeScanner()
    for idx in range(first_idx, len(lines_list)):
        line, _ = split_line_content(lines_list[idx])
        scanner.process(line)
        comment_start = find_comment_start(line)
        code = line[:comment_start] if comment_start is not None else line
        if scanner.in_code_context and code.rstrip().endswith(":"):
            return idx
    return None

"""AST-based analysis for exception-chain try-except and with-block matching.

This module provides the pure try-except and with-statement utilities used by
the pipeline in ``tracerite.trace``.  The chronological-order construction
itself lives in ``order.py`` so the whole pipeline is visible in one place.
"""

from __future__ import annotations

import ast
import linecache
from typing import TYPE_CHECKING

from tracerite.logging import logger

from .core import (
    block_contains_in_except,
    block_contains_in_try,
    block_offset_by,
    with_block_offset_by,
)

if TYPE_CHECKING:
    from .typing import TryExceptBlock, WithBlock

__all__ = [
    "parse_source_for_try_except",
    "parse_source_string_for_try_except",
    "find_try_block_for_except_line",
    "find_matching_try_for_inner_exception",
    "parse_source_string_for_with_blocks",
    "find_with_block_for_header_line",
]


class TryExceptVisitor(ast.NodeVisitor):
    """AST visitor that collects all try-except blocks with their line ranges."""

    def __init__(self):
        self.try_except_blocks: list[TryExceptBlock] = []

    def visit_Try(self, node: ast.Try):
        """Visit a Try node and record its structure."""
        try_body_end = self._get_last_line(node.body)

        except_start = None
        except_end = None
        if node.handlers:
            except_start = node.handlers[0].lineno
            except_end = self._get_last_line(list(node.handlers))

        finally_start = None
        finally_end = None
        if node.finalbody:
            finally_start = node.finalbody[0].lineno
            finally_end = self._get_last_line(node.finalbody)

        if except_start is not None:
            block: TryExceptBlock = {
                "try_start": node.lineno,
                "try_end": try_body_end,
                "except_start": except_start,
                "except_end": except_end,
                "finally_start": finally_start,
                "finally_end": finally_end,
            }
            self.try_except_blocks.append(block)

        self.generic_visit(node)

    @staticmethod
    def _get_last_line(nodes) -> int:
        """Get the last line number from a list of AST nodes."""
        return max(
            (getattr(node, "end_lineno", node.lineno) for node in nodes),
            default=0,
        )


def parse_source_for_try_except(
    filename: str,
    function_name: str | None = None,
    *,
    _cache: dict | None = None,
) -> list[TryExceptBlock]:
    """Parse source file and extract try-except blocks.

    Args:
        filename: Path to the source file
        function_name: Optional function name to limit scope
        _cache: Optional append-only cache mapping source keys to parsed blocks.
            Intended for single-call use only; not persisted across calls.

    Returns:
        List of TryExceptBlock objects found in the source
    """
    key = ("file", filename, function_name)
    if _cache is not None and key in _cache:
        return _cache[key]

    try:
        lines = linecache.getlines(filename)
        if not lines:
            result: list[TryExceptBlock] = []
        else:
            source = "".join(lines)
            tree = ast.parse(source, filename=filename)

            visitor = TryExceptVisitor()
            visitor.visit(tree)

            result = visitor.try_except_blocks
    except (SyntaxError, OSError, ValueError) as e:
        logger.debug(f"Failed to parse {filename} for try-except analysis: {e}")
        result = []

    if _cache is not None:
        _cache[key] = result
    return result


def parse_source_string_for_try_except(
    source: str,
    start_line: int = 1,
    *,
    _cache: dict | None = None,
) -> list[TryExceptBlock]:
    """Parse source string and extract try-except blocks.

    Args:
        source: The source code as a string
        start_line: The line number where this source starts (for offset adjustment)
        _cache: Optional append-only cache mapping source keys to parsed blocks.
            Intended for single-call use only; not persisted across calls.

    Returns:
        List of TryExceptBlock objects found in the source
    """
    key = ("string", source, start_line)
    if _cache is not None and key in _cache:
        return _cache[key]

    try:
        if not source:
            result: list[TryExceptBlock] = []
        else:
            tree = ast.parse(source)

            visitor = TryExceptVisitor()
            visitor.visit(tree)

            blocks = visitor.try_except_blocks
            if start_line != 1:
                offset = start_line - 1
                blocks = [block_offset_by(block, offset) for block in blocks]

            result = blocks
    except (SyntaxError, ValueError) as e:
        logger.debug(f"Failed to parse source string for try-except analysis: {e}")
        result = []

    if _cache is not None:
        _cache[key] = result
    return result


def find_try_block_for_except_line(
    blocks: list[TryExceptBlock], except_lineno: int
) -> TryExceptBlock | None:
    """Find the try-except block that contains the given line in its except handler."""
    matching_blocks = [b for b in blocks if block_contains_in_except(b, except_lineno)]
    return (
        max(matching_blocks, key=lambda b: b["try_start"]) if matching_blocks else None
    )


def find_matching_try_for_inner_exception(
    blocks: list[TryExceptBlock], inner_first_lineno: int, outer_except_lineno: int
) -> TryExceptBlock | None:
    """Find the try block that links an inner and outer exception."""
    for block in blocks:
        if block_contains_in_except(
            block, outer_except_lineno
        ) and block_contains_in_try(block, inner_first_lineno):
            return block
    return None


class WithBlockVisitor(ast.NodeVisitor):
    """AST visitor that collects all with/async-with statements with their line ranges."""

    def __init__(self):
        self.with_blocks: list[WithBlock] = []

    def visit_With(self, node: ast.With):
        self._record(node)

    def visit_AsyncWith(self, node: ast.AsyncWith):
        self._record(node)

    def _record(self, node):
        """Record a with statement's header and body line ranges."""
        if not node.body:  # pragma: no cover
            return
        block: WithBlock = {
            "header_start": node.lineno,
            "body_start": node.body[0].lineno,
            "block_end": node.end_lineno or node.body[-1].lineno,
            "items": [
                {
                    "lfirst": item.context_expr.lineno,
                    "lfinal": item.context_expr.end_lineno or item.context_expr.lineno,
                    "cbeg": item.context_expr.col_offset,
                    "cend": item.context_expr.end_col_offset
                    or item.context_expr.col_offset + 1,
                }
                for item in node.items
            ],
        }
        self.with_blocks.append(block)
        self.generic_visit(node)


def dedent_source(source: str) -> tuple[str, int]:
    """Remove the first line's indent from all lines that share it.

    inspect.getsourcelines keeps the original indentation for nested
    functions, which does not parse as a standalone module.  Column offsets
    of parsed nodes must be shifted back by the returned amount.
    """
    first_line = next((line for line in source.splitlines() if line.strip()), "")
    indent = len(first_line) - len(first_line.lstrip(" \t"))
    if not indent:
        return source, 0
    prefix = first_line[:indent]
    dedented = "".join(
        line[indent:] if line.startswith(prefix) else line
        for line in source.splitlines(keepends=True)
    )
    return dedented, indent


def parse_source_string_for_with_blocks(
    source: str,
    start_line: int = 1,
    *,
    _cache: dict | None = None,
) -> list[WithBlock]:
    """Parse source string and extract with/async-with blocks.

    Args:
        source: The source code as a string
        start_line: The line number where this source starts (for offset adjustment)
        _cache: Optional append-only cache mapping source keys to parsed blocks.
            Intended for single-call use only; not persisted across calls.

    Returns:
        List of WithBlock objects found in the source
    """
    key = ("with_blocks_string", source, start_line)
    if _cache is not None and key in _cache:
        return _cache[key]

    try:
        if not source:
            result: list[WithBlock] = []
        else:
            dedented, col_offset = dedent_source(source)
            tree = ast.parse(dedented)

            visitor = WithBlockVisitor()
            visitor.visit(tree)

            blocks = visitor.with_blocks
            if start_line != 1 or col_offset:
                offset = start_line - 1
                blocks = [
                    with_block_offset_by(block, offset, col_offset) for block in blocks
                ]

            result = blocks
    except (SyntaxError, ValueError) as e:
        logger.debug(f"Failed to parse source string for with-block analysis: {e}")
        result = []

    if _cache is not None:
        _cache[key] = result
    return result


def find_with_block_for_header_line(
    blocks: list[WithBlock], lineno: int
) -> WithBlock | None:
    """Find the innermost with block whose statement header covers the given line.

    Only matches lines within the header itself (up to but excluding the first
    body line), so a frame stopped inside the block body never matches.
    """
    matching = [b for b in blocks if b["header_start"] <= lineno < b["body_start"]]
    if not matching:
        return None
    return min(
        matching, key=lambda b: (b["block_end"] - b["header_start"], -b["header_start"])
    )

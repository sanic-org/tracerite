"""AST-based analysis for exception-chain try-except block matching.

This module provides the pure try-except utilities used by the pipeline in
``tracerite.trace``.  The chronological-order construction itself lives in
``trace.py`` so the whole pipeline is visible in one place.
"""

from __future__ import annotations

import ast
import linecache
from dataclasses import dataclass

from .logging import logger

__all__ = [
    "TryExceptBlock",
    "ChainLink",
    "parse_source_for_try_except",
    "parse_source_string_for_try_except",
    "find_try_block_for_except_line",
    "find_matching_try_for_inner_exception",
    "analyze_exception_chain_links",
    "enrich_chain_with_links",
    "build_chronological_frames",
]


@dataclass
class TryExceptBlock:
    """Represents a try-except block with its line ranges."""

    try_start: int  # First line of try keyword
    try_end: int  # Last line of try body (before except/else/finally)
    except_start: int | None  # First line of except handlers
    except_end: int | None  # Last line of except handlers
    finally_start: int | None = None
    finally_end: int | None = None

    def contains_in_try(self, lineno: int) -> bool:
        """Check if a line number is within the try body."""
        return self.try_start <= lineno <= self.try_end

    def contains_in_except(self, lineno: int) -> bool:
        """Check if a line number is within an except handler."""
        if self.except_start is None or self.except_end is None:
            return False
        return self.except_start <= lineno <= self.except_end

    def offset_by(self, offset: int) -> TryExceptBlock:
        """Return a new block with all line numbers shifted by offset."""

        def add(x: int | None) -> int | None:
            return x + offset if x is not None else None

        return TryExceptBlock(
            try_start=self.try_start + offset,
            try_end=self.try_end + offset,
            except_start=add(self.except_start),
            except_end=add(self.except_end),
            finally_start=add(self.finally_start),
            finally_end=add(self.finally_end),
        )


@dataclass
class ChainLink:
    """Represents a link between two exceptions in a chain.

    Attributes:
        outer_frame_idx: Index of the frame in the outer exception that's in the except block
        try_block: The TryExceptBlock that links the inner and outer exceptions
        matched: Whether we successfully matched the try-except relationship
    """

    outer_frame_idx: int
    try_block: TryExceptBlock | None
    matched: bool


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
            block = TryExceptBlock(
                try_start=node.lineno,
                try_end=try_body_end,
                except_start=except_start,
                except_end=except_end,
                finally_start=finally_start,
                finally_end=finally_end,
            )
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
    filename: str, function_name: str | None = None
) -> list[TryExceptBlock]:
    """Parse source file and extract try-except blocks.

    Args:
        filename: Path to the source file
        function_name: Optional function name to limit scope

    Returns:
        List of TryExceptBlock objects found in the source
    """
    return _parse_source_for_try_except(filename, function_name)


def _parse_source_for_try_except(
    filename: str, function_name: str | None = None
) -> list[TryExceptBlock]:
    try:
        lines = linecache.getlines(filename)
        if not lines:
            return []

        source = "".join(lines)
        tree = ast.parse(source, filename=filename)

        visitor = TryExceptVisitor()
        visitor.visit(tree)

        return visitor.try_except_blocks
    except (SyntaxError, OSError, ValueError) as e:
        logger.debug(f"Failed to parse {filename} for try-except analysis: {e}")
        return []


def parse_source_string_for_try_except(
    source: str, start_line: int = 1
) -> list[TryExceptBlock]:
    """Parse source string and extract try-except blocks.

    Args:
        source: The source code as a string
        start_line: The line number where this source starts (for offset adjustment)

    Returns:
        List of TryExceptBlock objects found in the source
    """
    try:
        if not source:
            return []

        tree = ast.parse(source)

        visitor = TryExceptVisitor()
        visitor.visit(tree)

        if start_line != 1:
            offset = start_line - 1
            return [block.offset_by(offset) for block in visitor.try_except_blocks]

        return visitor.try_except_blocks
    except (SyntaxError, ValueError) as e:
        logger.debug(f"Failed to parse source string for try-except analysis: {e}")
        return []


def find_try_block_for_except_line(
    blocks: list[TryExceptBlock], except_lineno: int
) -> TryExceptBlock | None:
    """Find the try-except block that contains the given line in its except handler."""
    matching_blocks = [b for b in blocks if b.contains_in_except(except_lineno)]
    return max(matching_blocks, key=lambda b: b.try_start) if matching_blocks else None


def find_matching_try_for_inner_exception(
    blocks: list[TryExceptBlock], inner_first_lineno: int, outer_except_lineno: int
) -> TryExceptBlock | None:
    """Find the try block that links an inner and outer exception."""
    for block in blocks:
        if block.contains_in_except(outer_except_lineno) and block.contains_in_try(
            inner_first_lineno
        ):
            return block
    return None


# The chronological-order builders live in trace.py.  The names below are kept
# here so existing imports from ``tracerite.chain_analysis`` continue to work.


def analyze_exception_chain_links(chain: list[dict]) -> list:
    """Analyze an exception chain to find try-except relationships."""
    from .trace import _analyze_exception_chain_links

    return _analyze_exception_chain_links(chain)


def enrich_chain_with_links(chain: list[dict]) -> list[dict]:
    """Enrich exception chain with try-except link information."""
    from .trace import _enrich_chain_with_links

    return _enrich_chain_with_links(chain)


def build_chronological_frames(chain: list[dict]) -> list[dict]:
    """Build a chronological list of frames showing the actual sequence of events."""
    from .trace import _build_chronological_frames

    return _build_chronological_frames(chain)


# Private helpers that moved to trace.py together with the pipeline code.
# They are re-exported here for backward compatibility with existing tests.


def _get_frame_lineno(frame: dict):
    from .trace import _get_frame_lineno

    return _get_frame_lineno(frame)


def _frame_in_except_handler(frame: dict) -> bool:
    from .trace import _frame_in_except_handler

    return _frame_in_except_handler(frame)


def _find_chain_link(inner_exc: dict, outer_exc: dict):
    from .trace import _find_chain_link

    return _find_chain_link(inner_exc, outer_exc)


def _filter_hidden_frames(chronological: list[dict]) -> list[dict]:
    from .trace import _filter_hidden_frames

    return _filter_hidden_frames(chronological)


def _apply_base_exception_suppression(
    chronological: list[dict], chain: list[dict]
) -> list[dict]:
    from .trace import _apply_base_exception_suppression

    return _apply_base_exception_suppression(chronological, chain)

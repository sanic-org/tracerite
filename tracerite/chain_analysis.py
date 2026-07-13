"""AST-based analysis for exception chain try-except block matching.

This module provides utilities to build a chronological chain of events
during a multi-exception chain by analyzing the AST to identify which
try block contains the code that raised the inner exception, relative
to the except block where the outer exception was raised.

Key insight about Python exception frames:
- Inner exception frames start from the try block (not app entry point)
  So the first frame is always in the try body where the exception occurred.
- Outer exception frames start from app entry point and traverse through
  the call stack. We search these to find a frame in an except handler
  that corresponds to the inner's try block.

ExceptionGroups (Python 3.11+) introduce parallel timelines:
- An ExceptionGroup contains multiple subexceptions that occurred concurrently
- Each subexception has its own traceback chain
- These parallel timelines are represented as branches in the chronological view
- The ExceptionGroup's own traceback provides the surrounding context
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


def analyze_exception_chain_links(chain: list[dict]) -> list[ChainLink | None]:
    """Analyze an exception chain to find try-except relationships."""
    if len(chain) <= 1:
        return [None] * len(chain)

    links: list[ChainLink | None] = [None]
    for i in range(1, len(chain)):
        links.append(_find_chain_link(chain[i - 1], chain[i]))
    return links


def _get_frame_lineno(frame: dict) -> int | None:
    """Extract the most precise line number from a frame dict."""
    frame_range = frame.get("range")
    if frame_range:
        return frame_range[0]
    if frame.get("lineno"):
        return frame.get("lineno")
    return frame.get("linenostart")


def _get_try_except_blocks(frame: dict) -> list[TryExceptBlock]:
    """Get try-except blocks for a frame using its full source or filename."""
    full_source = frame.get("full_source")
    if full_source:
        blocks = parse_source_string_for_try_except(
            full_source, frame.get("full_source_start", 1)
        )
        if blocks:
            return blocks
    filename = frame.get("original_filename") or frame.get("filename")
    return parse_source_for_try_except(filename) if filename else []


def _frame_in_except_handler(frame: dict) -> bool:
    """Check whether a frame's line falls inside an ``except`` handler."""
    lineno = _get_frame_lineno(frame)
    if lineno is None:
        return False
    try:
        blocks = _get_try_except_blocks(frame)
    except Exception:
        return False
    return any(block.contains_in_except(lineno) for block in blocks)


def _find_chain_link(inner_exc: dict, outer_exc: dict) -> ChainLink | None:
    """Find the try-except link between two consecutive exceptions."""
    inner_frames = inner_exc.get("frames", [])
    outer_frames = outer_exc.get("frames", [])
    if not inner_frames or not outer_frames:
        return None

    inner_first_frame = inner_frames[0]
    inner_first_lineno = _get_frame_lineno(inner_first_frame)
    if inner_first_lineno is None:
        return None

    try_except_blocks = _get_try_except_blocks(inner_first_frame)
    if not try_except_blocks:
        return None

    inner_filename = inner_first_frame.get(
        "original_filename"
    ) or inner_first_frame.get("filename")

    for frame_idx, frame in enumerate(outer_frames):
        frame_lineno = _get_frame_lineno(frame)
        if frame_lineno is None:
            continue

        outer_filename = frame.get("original_filename") or frame.get("filename")
        if inner_filename and outer_filename != inner_filename:
            continue

        matching_block = find_matching_try_for_inner_exception(
            try_except_blocks, inner_first_lineno, frame_lineno
        )
        if matching_block:
            return ChainLink(
                outer_frame_idx=frame_idx,
                try_block=matching_block,
                matched=True,
            )
    return None


def enrich_chain_with_links(chain: list[dict]) -> list[dict]:
    """Enrich exception chain with try-except link information."""
    links = analyze_exception_chain_links(chain)

    for exc, link in zip(chain, links, strict=True):
        if link and link.matched and (try_block := link.try_block):
            exc["chain_link"] = {
                "outer_frame_idx": link.outer_frame_idx,
                "try_start": try_block.try_start,
                "try_end": try_block.try_end,
                "except_start": try_block.except_start,
                "except_end": try_block.except_end,
            }
        else:
            exc["chain_link"] = None

    return chain


def build_chronological_frames(chain: list[dict]) -> list[dict]:
    """Build a chronological list of frames showing the actual sequence of events."""
    if not chain:
        return []

    links = analyze_exception_chain_links(chain)
    chronological: list[dict] = []

    outermost = chain[-1]
    _build_backbone_frames(
        chronological,
        outermost,
        len(chain) - 1,
        outermost.get("frames", []),
        links,
        chain,
    )

    chronological = _filter_hidden_frames(chronological)
    chronological = _apply_base_exception_suppression(chronological, chain)

    return chronological


def _filter_hidden_frames(chronological: list[dict]) -> list[dict]:
    """Filter out hidden frames, handling parallel branches recursively."""
    result = []
    for frame in chronological:
        if frame.get("hidden"):
            continue
        if "parallel" in frame:
            filtered_branches = []
            for branch in frame["parallel"]:
                filtered_branch = _filter_hidden_frames(branch)
                if filtered_branch:
                    filtered_branches.append(filtered_branch)
            if filtered_branches:
                frame = {**frame, "parallel": filtered_branches}
                result.append(frame)
        else:
            result.append(frame)
    return result


def _apply_base_exception_suppression(
    chronological: list[dict], chain: list[dict]
) -> list[dict]:
    """Suppress library frames after the last user code frame."""
    if not chronological or not chain:
        return chronological
    if not any(exc.get("suppress_inner") for exc in chain):
        return chronological

    last_bug = _find_last_bug_frame(chronological)
    if last_bug is None:
        return chronological

    result, keepers, suppressed = _split_suppressed_frames(chronological, last_bug)
    return _merge_suppressed_data(result, keepers, suppressed)


def _find_last_bug_frame(chronological: list[dict]) -> int | None:
    return next(
        (
            idx
            for idx, frame in enumerate(chronological)
            if frame.get("relevance") == "warning"
        ),
        None,
    )


def _split_suppressed_frames(
    chronological: list[dict], last_bug: int
) -> tuple[list[dict], list[dict], dict]:
    """Split chronological frames into kept, kept-error, and suppressed metadata."""
    result = chronological[: last_bug + 1]
    keepers = []
    suppressed = {"exception": None, "parallel": None}
    for frame in chronological[last_bug + 1 :]:
        if frame.get("relevance") in {"except", "error"}:
            keepers.append(frame)
            continue
        if frame.get("exception") and suppressed["exception"] is None:
            suppressed["exception"] = frame["exception"]
        if frame.get("parallel") and suppressed["parallel"] is None:
            suppressed["parallel"] = frame["parallel"]
    return result, keepers, suppressed


def _merge_suppressed_data(
    result: list[dict], keepers: list[dict], suppressed: dict
) -> list[dict]:
    """Transfer suppressed exception/parallel info onto the bug frame."""
    keeper_exc_types = {f.get("exception", {}).get("type") for f in keepers}
    keeper_has_parallel = any(f.get("parallel") for f in keepers)

    if (
        suppressed["exception"]
        and not result[-1].get("exception")
        and suppressed["exception"].get("type") not in keeper_exc_types
    ):
        result[-1] = {**result[-1], "exception": suppressed["exception"]}
    if (
        suppressed["parallel"]
        and not result[-1].get("parallel")
        and not keeper_has_parallel
    ):
        result[-1] = {**result[-1], "parallel": suppressed["parallel"]}

    result[-1] = {**result[-1], "relevance": "stop"}
    result.extend(keepers)
    return result


def _build_backbone_frames(
    chronological: list[dict],
    exc: dict,
    exc_idx: int,
    frames: list[dict],
    links: list,
    chain: list[dict],
) -> None:
    """Build chronological frames using this exception's frames as backbone."""
    inner_exc_idx = exc_idx - 1
    inner_link = links[exc_idx] if exc_idx > 0 else None

    if inner_link and inner_link.matched and inner_exc_idx >= 0:
        _build_linked_backbone(
            chronological,
            exc,
            exc_idx,
            frames,
            inner_exc_idx,
            inner_link,
            links,
            chain,
        )
    else:
        _build_unlinked_backbone(chronological, exc, exc_idx, frames, links, chain)


def _build_linked_backbone(
    chronological: list[dict],
    exc: dict,
    exc_idx: int,
    frames: list[dict],
    inner_exc_idx: int,
    inner_link: ChainLink,
    links: list,
    chain: list[dict],
) -> None:
    """Build backbone when the inner exception links to an except handler."""
    except_frame_idx = inner_link.outer_frame_idx
    inner_exc = chain[inner_exc_idx]
    inner_frames = inner_exc.get("frames", [])

    for frame_idx in range(except_frame_idx):
        _append_copied_frame(chronological, frames[frame_idx])

    _build_backbone_frames(
        chronological, inner_exc, inner_exc_idx, inner_frames, links, chain
    )

    last_idx = len(frames) - 1
    for frame_idx in range(except_frame_idx, len(frames)):
        chrono_frame = _append_copied_frame(chronological, frames[frame_idx])
        if frame_idx == except_frame_idx:
            _promote_to_except(chrono_frame)
        if frame_idx == last_idx:
            chrono_frame["exception"] = _make_exception_banner(exc, exc_idx)
            _add_subexceptions_to_frame(chrono_frame, exc)


def _build_unlinked_backbone(
    chronological: list[dict],
    exc: dict,
    exc_idx: int,
    frames: list[dict],
    links: list,
    chain: list[dict],
) -> None:
    """Build backbone without a matched try-except link."""
    if exc_idx > 0:
        inner_exc = chain[exc_idx - 1]
        inner_frames = inner_exc.get("frames", [])
        if inner_frames:
            _build_backbone_frames(
                chronological, inner_exc, exc_idx - 1, inner_frames, links, chain
            )

    re_raise = _find_re_raise_frames(exc, frames)
    order = [i for i in range(len(frames)) if i not in re_raise] + re_raise
    last_idx = len(frames) - 1
    banner = _make_exception_banner(exc, exc_idx)

    for idx, frame_idx in enumerate(order):
        chrono_frame = _append_copied_frame(chronological, frames[frame_idx])
        is_last = frame_idx == last_idx
        is_final = idx == len(order) - 1

        if is_last:
            _add_subexceptions_to_frame(chrono_frame, exc)
            if not re_raise:
                chrono_frame["exception"] = banner
        if is_final and re_raise:
            chrono_frame["exception"] = banner
        if frame_idx in re_raise:
            _promote_to_except(chrono_frame)


def _find_re_raise_frames(exc: dict, frames: list[dict]) -> list[int]:
    """Find frames (except the last) that are inside an except handler."""
    if not exc.get("subexceptions"):
        return []
    return [i for i, frame in enumerate(frames[:-1]) if _frame_in_except_handler(frame)]


def _make_exception_banner(exc: dict, exc_idx: int) -> dict:
    """Create the exception info dict attached to a frame."""
    return {
        "type": exc.get("type"),
        "message": exc.get("message"),
        "summary": exc.get("summary"),
        "from": exc.get("from"),
        "exc_idx": exc_idx,
    }


def _promote_to_except(frame: dict) -> None:
    """Promote a frame's relevance to indicate it represents an except handler."""
    if frame.get("relevance") in ("call", "warning"):
        frame["relevance"] = "except"
    frame["function_suffix"] = "⚡except"


def _append_copied_frame(chronological: list[dict], frame: dict, **overrides) -> dict:
    """Append a shallow copy of a frame with optional overrides."""
    copied = {**frame, **overrides}
    chronological.append(copied)
    return copied


def _add_subexceptions_to_frame(frame: dict, exc: dict) -> None:
    """Add subexceptions from an ExceptionGroup as parallel branches."""
    subexceptions = exc.get("subexceptions")
    if not subexceptions:
        return

    parallel_branches = [
        sub_chrono
        for sub_chain in subexceptions
        if (sub_chrono := build_chronological_frames(sub_chain))
    ]

    if parallel_branches:
        frame["parallel"] = parallel_branches

"""Shared data structures and small helpers for the traceback pipeline."""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any


def Range(
    lfirst: int,
    lfinal: int | None = None,
    cbeg: int | None = None,
    cend: int | None = None,
) -> dict[str, int | None]:
    """Create a position range dict. Lines are 1-based inclusive, columns 0-based exclusive."""
    return {"lfirst": lfirst, "lfinal": lfinal, "cbeg": cbeg, "cend": cend}


@dataclass(slots=True)
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


@dataclass(slots=True)
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


def compute_cursor_position(
    mark_range: dict[str, int] | None,
    em_ranges: dict[str, int] | list[dict[str, int]] | None,
    linenostart: int,
    common_indent: str = "",
) -> tuple[int, int]:
    """Return the preferred cursor (line, column) for highlighting."""
    target = None
    if em_ranges:
        if isinstance(em_ranges, list) and em_ranges:
            target = em_ranges[-1]
        elif isinstance(em_ranges, dict):
            target = em_ranges
    if target is None:
        target = mark_range

    if target is None:
        return (linenostart, 0)

    return (
        linenostart + target["lfinal"] - 1,
        target["cend"] + len(common_indent),
    )


# Will be set to an instance if loaded as an IPython extension by %load_ext
ipython: Any = None

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

# Fixed membership sets used throughout the pipeline
TRIPLE_QUOTES = frozenset(('"""', "'''"))
QUOTES = frozenset(('"', "'"))
STRING_PREFIX_PAIRS = frozenset(("fr", "rf", "br", "rb"))
STRING_PREFIXES = frozenset(("f", "r", "b", "u"))
EMPHASIS_BEG = frozenset(("solo", "beg"))
EMPHASIS_FIN = frozenset(("solo", "fin"))
EMPHASIS_MARKS = frozenset(("solo", "beg", "fin"))
COMP_CODE_NAMES = frozenset(
    ("<module>", "<listcomp>", "<dictcomp>", "<setcomp>", "<genexpr>")
)
KEEP_AFTER_SUPPRESSION = frozenset(("except", "error"))
HIGHLIGHT_RELEVANCES = frozenset(("error", "stop"))
PROMOTABLE_RELEVANCES = frozenset(("call", "warning"))


def libdir_match(path):
    """Check if path is in a library directory and return the short suffix if so."""
    m = libdir.fullmatch(path)
    if m:
        return next((g for g in m.groups() if g), "")
    return None


def create_summary(message: str) -> str:
    """Extract the first line of the exception message as summary."""
    return message.split("\n", 1)[0]


def chain_reason(e: BaseException) -> str:
    """Return the chaining relationship for an exception."""
    if e.__cause__:
        return "cause"
    if e.__context__ and not e.__suppress_context__:
        return "context"
    return "none"

"""Shared data structures and small helpers for the traceback pipeline."""

from __future__ import annotations

import re
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from .typing import Range, TryExceptBlock


def block_contains_in_try(block: TryExceptBlock, lineno: int) -> bool:
    """Check if a line number is within the try body of *block*."""
    try_start = block["try_start"]
    try_end = block["try_end"]
    return (
        try_start is not None and try_end is not None and try_start <= lineno <= try_end
    )


def block_contains_in_except(block: TryExceptBlock, lineno: int) -> bool:
    """Check if a line number is within an except handler of *block*."""
    except_start = block.get("except_start")
    except_end = block.get("except_end")
    return (
        except_start is not None
        and except_end is not None
        and except_start <= lineno <= except_end
    )


def block_offset_by(block: TryExceptBlock, offset: int) -> TryExceptBlock:
    """Return a new try-except block with all line numbers shifted by *offset*."""

    def add(x: int | None) -> int | None:
        return x + offset if x is not None else None

    return {
        "try_start": add(block["try_start"]),
        "try_end": add(block["try_end"]),
        "except_start": add(block.get("except_start")),
        "except_end": add(block.get("except_end")),
        "finally_start": add(block.get("finally_start")),
        "finally_end": add(block.get("finally_end")),
    }


def compute_cursor_position(
    mark_range: Range | None,
    em_ranges: Range | list[Range] | None,
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

# Symbols for each frame relevance type.  Narrow (1ch wide) emoji are padded
# with a trailing space to the 2ch width of the others, so that format
# strings can use a single space after any symbol.
symbols = {"call": "➤", "warning": "⚠️ ", "error": "💣", "stop": "🛑", "except": "⚠️ "}

# Fixed membership sets used throughout the pipeline
TRIPLE_QUOTES = frozenset(('"""', "'''"))
QUOTES = frozenset(('"', "'"))
STRING_PREFIX_PAIRS = frozenset(("fr", "rf", "br", "rb"))
STRING_PREFIXES = frozenset(("f", "r", "b", "u"))
EMPHASIS_BEG = frozenset(("solo", "beg"))
EMPHASIS_FIN = frozenset(("solo", "fin"))
COMP_CODE_NAMES = frozenset(
    ("<module>", "<listcomp>", "<dictcomp>", "<setcomp>", "<genexpr>")
)
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

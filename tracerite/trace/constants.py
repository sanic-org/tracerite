from __future__ import annotations

import re
from collections import namedtuple
from typing import Any

# Position range: lines are 1-based inclusive, columns are 0-based exclusive
Range = namedtuple("Range", ["lfirst", "lfinal", "cbeg", "cend"])


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
        elif isinstance(em_ranges, Range):
            target = em_ranges
    if target is None:
        target = mark_range

    if target is None:
        return (linenostart, 0)

    return (
        linenostart + target.lfinal - 1,
        target.cend + len(common_indent),
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

"""Type definitions for the traceback pipeline.

These types are only needed for static type checking.  They are kept in a
separate module so that runtime code can import them conditionally under
``TYPE_CHECKING``.  Everything is a plain dict at runtime; since the dicts
are built up incrementally across the pipeline, most keys are optional
(``total=False``).
"""

from __future__ import annotations

from typing import Any, TypeAlias, TypedDict


class Range(TypedDict):
    """A line/column span: 1-based lines, 0-based columns."""

    lfirst: int
    lfinal: int
    cbeg: int
    cend: int


class TryExceptBlock(TypedDict):
    """Line ranges of a try/except/finally block in a source file."""

    try_start: int | None
    try_end: int | None
    except_start: int | None
    except_end: int | None
    finally_start: int | None
    finally_end: int | None


class ChainLink(TypedDict, total=False):
    """Try-except link between two chained exceptions."""

    matched: bool
    outer_frame_idx: int
    try_block: TryExceptBlock
    try_start: int | None
    try_end: int | None
    except_start: int | None
    except_end: int | None


class _FragmentBase(TypedDict):
    code: str


class Fragment(_FragmentBase, total=False):
    """A piece of source code with optional highlight markers."""

    mark: str
    em: str
    comment: str
    trailing: str
    dedent: str
    indent: str


class FragmentLine(TypedDict):
    """Fragments of one source line, with its 1-based line number."""

    line: int
    fragments: list[Fragment]


class VarInfo(TypedDict):
    """A variable name/value row for the inspector."""

    name: str
    typename: str
    value: Any
    format_hint: str


class FrameInfo(TypedDict, total=False):
    """One call frame in a traceback.

    The internal-only temp keys are removed before public emission.
    """

    id: str
    relevance: str
    hidden: bool
    idframe: int
    filename: str | None
    original_filename: str | None
    location: str
    notebook_cell: bool
    codeline: str | None
    range: Range | None
    lineno: int
    cursor_line: int
    cursor_col: int
    linenostart: int
    lines: str
    fragments: list[FragmentLine]
    function: str | None
    function_suffix: str
    urls: dict[str, str]
    full_source: str | None
    full_source_start: int | None
    exception: ExceptionInfo
    parallel: list[list[FrameInfo]]
    variables: list[VarInfo]
    _frame_obj: Any  # temp: digest -> finalize
    _variable_source: str | None  # temp: digest -> finalize
    _except_start: int | None  # temp: digest -> order -> finalize


ExcChain: TypeAlias = list["ExceptionInfo"]

# Functional syntax: "from" is a Python keyword and cannot be a class attribute.
ExceptionInfo = TypedDict(
    "ExceptionInfo",
    {
        "type": str,
        "message": str,
        "summary": str,
        "from": str,  # "cause" | "context" | "none"
        "repr": str,
        "frames": list[FrameInfo],
        "suppress_inner": bool,
        "subexceptions": list[ExcChain],
        "leaf_types": list[str],
        "chain_link": ChainLink | None,
        "exc_idx": int,
        "_exc": BaseException,  # temp: digest -> removed in finalize
    },
    total=False,
)


class Chain(TypedDict):
    """Public result of extract_chain()."""

    header: str
    frames: list[FrameInfo]


class RawChainItem(TypedDict):
    exc: BaseException
    kwargs: dict[str, Any]


RawChain: TypeAlias = list[RawChainItem]

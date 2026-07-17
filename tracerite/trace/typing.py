"""Type aliases for the traceback pipeline.

These aliases are only needed for static type checking.  They are kept in a
separate module so that runtime code can import them conditionally under
``TYPE_CHECKING``.
"""

from __future__ import annotations

from typing import Any, TypeAlias

# Native dict shapes that used to be namedtuple/dataclass/factory helpers.
Range: TypeAlias = dict[str, int]
TryExceptBlock: TypeAlias = dict[str, int | None]
ChainLink: TypeAlias = dict[str, Any]

# Public API shape aliases for consumers of the pipeline.
FrameInfo: TypeAlias = dict[str, Any]
ExceptionInfo: TypeAlias = dict[str, Any]
ExcChain: TypeAlias = list[ExceptionInfo]
Chain: TypeAlias = dict[str, Any]
Fragment: TypeAlias = dict[str, Any]
RawChain: TypeAlias = list[dict[str, Any]]

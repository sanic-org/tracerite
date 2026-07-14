from __future__ import annotations

import sys
from types import ModuleType
from typing import Any

from . import core
from .finalize import extract_chain

__all__ = ["extract_chain"]

ipython: Any = core.ipython


class _TraceModule(ModuleType):
    """Proxy module that forwards ``ipython`` to ``tracerite.trace.core``."""

    @property
    def ipython(self):
        return core.ipython

    @ipython.setter
    def ipython(self, value):
        core.ipython = value


sys.modules[__name__].__class__ = _TraceModule

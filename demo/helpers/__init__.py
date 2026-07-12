"""Shared error-scenario helpers for the TraceRite demos."""

from __future__ import annotations

import inspect
from typing import Any

from demo.helpers import scenarios


def discover_scenarios() -> list[tuple[str, Any]]:
    """Return all public functions from the scenarios module in definition order."""
    return [
        (name, func)
        for name, func in vars(scenarios).items()
        if (
            not name.startswith("_")
            and inspect.isfunction(func)
            and func.__module__ == scenarios.__name__
        )
    ]

"""Asynchronous and concurrent error scenarios."""

from __future__ import annotations

import asyncio

from . import calc, data


async def _fail_soon(message: str) -> None:
    """Async helper that fails after a short delay."""
    await asyncio.sleep(0.01)
    raise ValueError(message)


async def _crash_after_load() -> None:
    """Async helper that parses invalid config and fails."""
    await asyncio.sleep(0.01)
    data.load_config('{"host": "example.com" "port": 80}')


async def _crash_after_calc() -> None:
    """Async helper that divides by zero."""
    await asyncio.sleep(0.01)
    calc.compute_ratio(10, 0)


async def run_concurrent_tasks() -> None:
    """Run several tasks concurrently; failures surface as an ExceptionGroup."""
    async with asyncio.TaskGroup() as tg:
        tg.create_task(_fail_soon("service A"))
        tg.create_task(_crash_after_load())
        tg.create_task(_crash_after_calc())

"""Shared helper functions for TraceRite demo scenarios."""

from __future__ import annotations

import asyncio
import json
import re


def perform_calculation(foo, bar, divisor, multiplier):
    """Perform a calculation with multiple parameters."""
    return (foo.x + foo.y + bar.value) * multiplier / divisor


# ---------------------------------------------------------------------------
# Recursion
# ---------------------------------------------------------------------------


def recurse(n: int) -> int:
    """Infinite recursion with an incrementing argument."""
    return recurse(n + 1)


# ---------------------------------------------------------------------------
# Cause chain
# ---------------------------------------------------------------------------


def risky_computation(y) -> float:
    """Inner computation that divides by zero."""
    x = 10
    return x / y


def bad_handler(arg):
    if arg <= 0:
        raise Exception


def innerstep() -> None:
    """Middle of the cause chain: accidentally raises TypeError in except:."""
    where = "inner step"
    try:
        risky_computation(0)
    except ZeroDivisionError as e:
        bad_handler(where)


def middlestep() -> None:
    """Re-raise hop: catches the exception, adds a note and re-raises."""
    try:
        innerstep()
    except Exception as e:
        e.add_note("Hello from middlestep error handler")
        raise e


def outerstep() -> None:
    """Top level: wraps the exception from middlestep in ValueError."""
    action = "compute"
    try:
        middlestep()
    except Exception as e:
        detail = "inner step failed"
        raise ValueError(f"Could not {action}: {detail}") from e


# ---------------------------------------------------------------------------
# Regex callback
# ---------------------------------------------------------------------------


def regex_sub_callback(match: re.Match[str]) -> str:
    """This should format the regex replacement"""
    return match.group(0) + match.group(1)


# ---------------------------------------------------------------------------
# Concurrent failures
# ---------------------------------------------------------------------------


async def failsoon() -> None:
    """Async helper that fails after a short delay."""
    await asyncio.sleep(0.01)
    raise ValueError("Failed Soon")


async def jsoncrash() -> None:
    """Async helper that parses invalid config and fails."""
    await asyncio.sleep(0.01)
    json.loads('{"host": "example.com" "port": 80}')


async def zerocrash() -> None:
    """Async helper that divides by zero."""
    await asyncio.sleep(0.01)
    return 10 / 0


async def run_concurrent_tasks() -> None:
    """Run several tasks concurrently; consume one sub-exception with ``except*``."""
    try:
        async with asyncio.TaskGroup() as tg:
            tg.create_task(failsoon())
            tg.create_task(jsoncrash())
            tg.create_task(zerocrash())
    except* json.JSONDecodeError:
        # Consume the JSON decode error; the remaining sub-exceptions propagate.
        pass


# ---------------------------------------------------------------------------
# With statement enter/exit failures
# ---------------------------------------------------------------------------

class NoOp:
    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False

class EnterRaises(NoOp):
    def __enter__(self):
        raise RuntimeError("Could not acquire the resource")


class ExitRaises(NoOp):
    def __exit__(self, *exc):
        raise RuntimeError("Could not release the resource")

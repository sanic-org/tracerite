"""Ensure extract_chain output is JSON serializable for complex scenarios."""

from __future__ import annotations

import json
import sys

import pytest

from tracerite import extract_chain


def _risky_computation(y: float) -> float:
    x = 10
    return x / y


def _bad_handler(arg: object) -> None:
    if arg <= 0:  # type: ignore[operator]
        raise Exception("bad handler")


def _innerstep() -> None:
    where = "inner step"
    try:
        _risky_computation(0)
    except ZeroDivisionError:
        _bad_handler(where)


def _outerstep() -> None:
    action = "compute"
    try:
        _innerstep()
    except Exception as exc:
        detail = "inner step failed"
        raise ValueError(f"Could not {action}: {detail}") from exc


def _run_chain() -> None:
    _outerstep()


CONCURRENT_SCENARIO = '''\
import asyncio
import json

async def _failsoon():
    await asyncio.sleep(0.01)
    raise ValueError("Failed Soon")

async def _jsoncrash():
    await asyncio.sleep(0.01)
    json.loads('{"host": "example.com" "port": 80}')

async def _zerocrash():
    await asyncio.sleep(0.01)
    return 10 / 0

async def _run_concurrent_tasks():
    try:
        async with asyncio.TaskGroup() as tg:
            tg.create_task(_failsoon())
            tg.create_task(_jsoncrash())
            tg.create_task(_zerocrash())
    except* json.JSONDecodeError:
        pass

async def _run():
    try:
        await _run_concurrent_tasks()
    except Exception as exc:
        raise RuntimeError("Application crashed inside asyncio.run()") from exc

def _run_concurrent():
    asyncio.run(_run())
'''


def _extract_from(scenario_func) -> list[dict]:
    """Run a scenario, catch its exception, and return extract_chain output."""
    try:
        scenario_func()
    except Exception as exc:
        return extract_chain(exc)
    raise RuntimeError(f"Scenario {scenario_func.__name__} did not raise an exception")


def _assert_json_compatible(value: object) -> None:
    """Recursively assert that *value* contains only JSON-compatible types."""
    if isinstance(value, dict):
        for key, item in value.items():
            _assert_json_compatible(key)
            _assert_json_compatible(item)
    elif isinstance(value, list):
        for item in value:
            _assert_json_compatible(item)
    elif isinstance(value, (str, int, float, bool, type(None))):
        return
    else:
        raise AssertionError(
            f"Found non-JSON-compatible value of type {type(value).__name__}: {value!r}"
        )


def test_extract_chain_is_json_serializable_chain() -> None:
    """A cause chain must be composed of JSON-compatible types."""
    chain = _extract_from(_run_chain)

    _assert_json_compatible(chain)

    assert isinstance(chain, dict)
    assert "header" in chain
    assert "frames" in chain
    assert len(chain["frames"]) > 0

    serialized = json.dumps(chain)
    deserialized = json.loads(serialized)

    assert isinstance(deserialized, dict)
    assert len(deserialized["frames"]) == len(chain["frames"])


@pytest.mark.skipif(
    sys.version_info < (3, 11), reason="ExceptionGroup/except* requires Python 3.11+"
)
def test_extract_chain_is_json_serializable_concurrent() -> None:
    """A concurrent ExceptionGroup scenario must be composed of JSON-compatible types."""
    ns: dict = {}
    exec(CONCURRENT_SCENARIO, ns)  # noqa: S102
    run_concurrent = ns["_run_concurrent"]

    chain = _extract_from(run_concurrent)

    _assert_json_compatible(chain)

    assert isinstance(chain, dict)
    assert "header" in chain
    assert "frames" in chain
    assert len(chain["frames"]) > 0

    serialized = json.dumps(chain)
    deserialized = json.loads(serialized)

    assert isinstance(deserialized, dict)
    assert len(deserialized["frames"]) == len(chain["frames"])

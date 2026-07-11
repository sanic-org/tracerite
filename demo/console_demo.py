#!/usr/bin/env -S uv run
# /// script
# requires-python = ">=3.11"
# dependencies = [ "tracerite" ]
# tool.uv.sources.tracerite = { path = "../", editable = true }
# ///
"""Console demo for TraceRite multi-line exception message formatting.

Run with:
    ./demo/console_demo.py
or:
    uv run demo/console_demo.py
"""

from __future__ import annotations

import argparse
import inspect
import sys

from tracerite import load

from demo.helpers import scenarios


def _discover() -> list[tuple[str, object]]:
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


def run_scenario(title: str, func: object) -> None:
    # Print to stderr so the header appears right before the tracerite
    # traceback, which is also written to stderr.
    doc = (func.__doc__ or "").strip()
    print(f"{title}: {doc}", file=sys.stderr)
    __tracebackhide__ = True
    try:
        func()  # type: ignore[operator]
    except Exception:
        # Render with the current exception hook (TraceRite by default, or
        # Python's builtin hook when --builtin is used), then continue.
        sys.excepthook(*sys.exc_info())
    print(file=sys.stderr)  # blank line between scenarios

def main() -> None:
    parser = argparse.ArgumentParser(description="TraceRite console demo")
    parser.add_argument(
        "scenarios",
        nargs="*",
        help="scenario names to run (default: run all)",
    )
    parser.add_argument(
        "--builtin",
        action="store_true",
        help="disable TraceRite and use Python's default exception handling",
    )
    args = parser.parse_args()

    if not args.builtin:
        load()

    available = dict(_discover())
    if args.scenarios:
        for name in args.scenarios:
            if name not in available:
                parser.error(f"unknown scenario: {name!r}")
        selected = [(name, available[name]) for name in args.scenarios]
    else:
        selected = list(available.items())

    for title, func in selected:
        run_scenario(title, func)


if __name__ == "__main__":
    main()

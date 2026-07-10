#!/usr/bin/env -S uv run
# /// script
# requires-python = ">=3.11"
# dependencies = [ "tracerite" ]
# tool.uv.sources.tracerite = { path = "../", editable = true }
# ///
from __future__ import annotations

import argparse
import io
import re
import timeit
from collections.abc import Callable

from demo.console_demo import SCENARIOS as DEMO_SCENARIOS
from tracerite import extract_chain, html_traceback, tty_traceback


def _slug(title: str) -> str:
    """Turn a scenario title into a compact command-line key."""
    return re.sub(r"[^a-z0-9]+", "-", title.lower()).strip("-")


def _make_exc(func: Callable[[], object]) -> Callable[[], BaseException]:
    """Wrap a scenario so it returns the raised exception instead of propagating."""

    def wrapper() -> BaseException:
        try:
            func()
        except BaseException as exc:
            return exc
        raise RuntimeError("Expected an exception")

    return wrapper


# Representative scenarios for benchmarking: realistic cross-module traces,
# stdlib callbacks, exception chains/groups, plus one large message.
BENCHMARK_KEYS = [
    "mixed-content",
    "deep-api-pipeline",
    "function-call",
    "order-processing",
    "chained-pipeline",
    "config-load",
    "callback-error",
    "record-batch",
    "variable-inspector",
    "concurrent-failures",
]

SCENARIOS: dict[str, tuple[str, Callable[[], BaseException]]] = {
    _slug(title): (title, _make_exc(func))
    for title, func in DEMO_SCENARIOS
    if _slug(title) in BENCHMARK_KEYS
}


def _time_ms(fn: Callable[[], object], number: int) -> float:
    """Time `fn` over `number` calls and return the average time in milliseconds."""
    fn()  # Warm up.
    total = timeit.timeit(fn, number=number)
    return total * 1000 / number


def _row(
    name: str,
    extract: float,
    html: float,
    tty: float,
    name_width: int,
) -> str:
    """Format one benchmark row."""
    return (
        f"{name:<{name_width}}"
        f"{extract:>13.2f}"
        f"{html:>13.2f}"
        f"{extract + html:>13.2f}"
        f"{tty:>13.2f}"
        f"{extract + tty:>13.2f}"
    )


def _benchmark_scenario(
    name: str,
    make_exc: Callable[[], BaseException],
    number: int,
    term_width: int,
) -> tuple[float, float, float]:
    """Measure one scenario and return (extract_ms, html_ms, tty_ms)."""
    exc = make_exc()
    chain = extract_chain(exc)

    # Warm up all render paths.
    str(html_traceback(chain=chain))
    tty_traceback(chain=chain, file=io.StringIO(), term_width=term_width)

    extract_ms = _time_ms(lambda: extract_chain(exc), number)
    html_ms = _time_ms(lambda: html_traceback(chain=chain), number)
    tty_ms = _time_ms(
        lambda: tty_traceback(chain=chain, file=io.StringIO(), term_width=term_width),
        number,
    )

    return extract_ms, html_ms, tty_ms


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Benchmark tracerite traceback rendering."
    )
    parser.add_argument(
        "--scenario",
        choices=list(SCENARIOS),
        default=None,
        help="Which demo traceback to render (default: all)",
    )
    parser.add_argument(
        "-n",
        "--number",
        type=int,
        default=1000,
        help="Number of renders to time",
    )
    parser.add_argument(
        "--term-width",
        type=int,
        default=120,
        help="Terminal width for TTY rendering",
    )
    args = parser.parse_args()

    scenarios = (
        {args.scenario: SCENARIOS[args.scenario]} if args.scenario else SCENARIOS
    )

    name_width = max(len("Scenario"), *(len(name) for name, _ in scenarios.values()))
    header = (
        f"{'Scenario':<{name_width}}"
        f"{'extract_chain':>13}"
        f"{'html format':>13}"
        f"{'html total':>13}"
        f"{'tty render':>13}"
        f"{'tty total':>13}"
    )
    print(f"Renders per scenario: {args.number}")
    print()
    print(header)
    print("-" * len(header))

    totals = [0.0, 0.0, 0.0]
    for _key, (name, make_exc) in scenarios.items():
        extract_ms, html_ms, tty_ms = _benchmark_scenario(
            name, make_exc, args.number, args.term_width
        )
        print(_row(name, extract_ms, html_ms, tty_ms, name_width))
        totals[0] += extract_ms
        totals[1] += html_ms
        totals[2] += tty_ms

    print("-" * len(header))
    count = len(scenarios)
    print(_row("Mean", totals[0] / count, totals[1] / count, totals[2] / count, name_width))


if __name__ == "__main__":
    main()

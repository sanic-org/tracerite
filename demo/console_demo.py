"""Console demo for TraceRite multi-line exception message formatting.

Run with:
    uv run python demo/console_demo.py
or, after activating the project's virtual environment:
    python demo/console_demo.py

Use ``--interactive`` to choose scenarios one at a time.
"""

from __future__ import annotations

import asyncio
import json
import sys

from tracerite import load

from demo.helpers import async_tasks, calc, data, messages, objects

load()


def demo_mixed_content() -> None:
    """Very long message mixing prose, code blocks, and a long list."""
    messages.mixed_content()


def demo_chained_multiline() -> None:
    """Two chained exceptions, each with a multi-line message."""
    messages.chained_multiline()


def demo_deep_api_pipeline() -> None:
    """Deep call chain jumping between modules and ending in a division by zero."""
    calc.process_user_data(500)


def demo_function_call() -> None:
    """Error inside a multi-line function call with keyword arguments."""
    calc.complex_function_call()


def demo_order_processing() -> None:
    """Cross-module order processing ending in a wrapped domain error."""
    calc.process_order(42)


def demo_chained_pipeline() -> None:
    """Explicit three-level exception cause chain."""
    calc.run_chained_pipeline()


def demo_config_load() -> None:
    """Malformed JSON parsed in another module, re-raised as a domain error."""
    try:
        data.load_config('{"host": "example.com" "port": 80}')
    except json.JSONDecodeError as e:
        raise RuntimeError("Configuration is malformed") from e


def demo_callback_error() -> None:
    """Failure inside a stdlib regex callback, wrapped at the call site."""
    try:
        data.apply_regex_discounts("50 0 25")
    except ZeroDivisionError as e:
        raise ValueError("Invalid percentage input") from e


def demo_record_batch() -> None:
    """Batch processing via ``map()`` combining JSON parsing and calculation."""
    records = [
        '{"value": 100, "divisor": 10}',
        '{"value": 100, "divisor": 0}',
    ]
    data.process_records(records)


def demo_variable_inspector() -> None:
    """Error with objects that have good and poor string representations."""
    objects.inspect_variables()


def demo_string_concat() -> None:
    """Multi-line string concatenation with a type mismatch."""
    objects.build_greeting("World", 123)


def demo_concurrent_failures() -> None:
    """Multiple concurrent failures propagated through asyncio.run and wrapped."""
    try:
        asyncio.run(async_tasks.run_concurrent_tasks())
    except ExceptionGroup as eg:
        raise RuntimeError("Concurrent tasks failed") from eg


SCENARIOS = [
    ("Mixed content", demo_mixed_content),
    ("Chained multi-line", demo_chained_multiline),
    ("Deep API pipeline", demo_deep_api_pipeline),
    ("Function call", demo_function_call),
    ("Order processing", demo_order_processing),
    ("Chained pipeline", demo_chained_pipeline),
    ("Config load", demo_config_load),
    ("Callback error", demo_callback_error),
    ("Record batch", demo_record_batch),
    ("Variable inspector", demo_variable_inspector),
    ("String concat", demo_string_concat),
    ("Concurrent failures", demo_concurrent_failures),
]


def run_scenario(title: str, func: object) -> None:
    # Print to stderr so the header appears right before the tracerite
    # traceback, which is also written to stderr.
    print("\n" + "=" * 60, file=sys.stderr)
    print(f"Scenario: {title}", file=sys.stderr)
    print("=" * 60, file=sys.stderr)
    try:
        func()  # type: ignore[operator]
    except Exception:
        # Render with the tracerite hook that load() installed, then continue.
        sys.excepthook(*sys.exc_info())


def run_all() -> None:
    for title, func in SCENARIOS:
        run_scenario(title, func)


def run_interactive() -> None:
    print("\nTraceRite console demo")
    print("=" * 60)
    for i, (title, _) in enumerate(SCENARIOS, start=1):
        print(f"{i}. {title}")
    print("0. Run all")
    print("q. Quit")

    while True:
        choice = input("\nChoice: ").strip().lower()
        if choice in ("q", "quit", "exit"):
            break
        if choice == "0":
            run_all()
            continue
        try:
            index = int(choice) - 1
            title, func = SCENARIOS[index]
        except (ValueError, IndexError):
            print("Unknown choice; try again.")
            continue
        run_scenario(title, func)


def main() -> None:
    if len(sys.argv) > 1 and sys.argv[1] in ("-i", "--interactive"):
        run_interactive()
    else:
        run_all()


if __name__ == "__main__":
    main()

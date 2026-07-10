"""Object representation and type-mismatch error scenarios."""

from __future__ import annotations

import json
from dataclasses import dataclass


@dataclass
class Foo:
    """Object with a good __repr__ from dataclass."""

    x: int = 10
    y: int = 20


class Bar:
    """Object with a poor __str__ and mixed member representations."""

    def __init__(self) -> None:
        self.value = 42
        self.text = "test"
        # Scanner object has a poor __str__ representation
        self.bad = json.JSONDecoder().scan_once

    def __str__(self) -> str:
        return "<Bar object at 0x12345678>"


def inspect_variables() -> float:
    """Trigger an error while interesting variables are in scope."""
    foo = Foo(x=100, y=200)
    bar = Bar()
    result = foo.x + bar.value + len(bar.text)
    return result / (foo.x - 100)  # Error: division by zero


def build_greeting(name: str, suffix) -> str:
    """Build a greeting by concatenating strings and a wrong-typed value."""
    return (
        "Hello "
        + name
        + " "
        + suffix  # Error: can't concatenate str and int
        + "!"
    )

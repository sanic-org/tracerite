import json
from dataclasses import dataclass


@dataclass
class Foo:
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

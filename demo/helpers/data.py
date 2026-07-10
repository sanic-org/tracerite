"""Data parsing and stdlib-callback error scenarios."""

from __future__ import annotations

import json
import re

from . import calc


def load_config(raw: str) -> dict:
    """Load and validate a JSON configuration string."""
    parsed = _parse_config(raw)
    return _normalize_config(parsed)


def _parse_config(raw: str) -> dict:
    """Parse raw JSON configuration."""
    return json.loads(raw)


def _normalize_config(config: dict) -> dict:
    """Normalize values and ensure required fields exist."""
    if "timeout" not in config:
        raise KeyError("missing required 'timeout' field")
    timeout = int(config["timeout"])
    config["timeout"] = timeout
    return config


def format_percentage(match: re.Match[str]) -> str:
    """Convert a regex match to a percentage string."""
    n = int(match.group(0))
    return f"{100 // n}%"


def apply_regex_discounts(text: str) -> str:
    """Apply a regex substitution that can fail inside the callback."""
    return re.sub(r"\d+", format_percentage, text)


def parse_record(raw: str) -> dict:
    """Parse a JSON record and enrich it with a computed score."""
    record = json.loads(raw)
    record["score"] = calc.compute_ratio(record["value"], record["divisor"])
    return record


def process_records(records: list[str]) -> list[dict]:
    """Process a batch of records via a stdlib callback."""
    return list(map(parse_record, records))

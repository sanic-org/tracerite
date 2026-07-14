"""Tests for tracerite.trace.order."""

from tracerite.trace.order import find_last_bug_frame


def test_find_last_bug_frame_returns_last_warning():
    """When multiple warning frames exist, the last one must be used."""
    chronological = [
        {"relevance": "call"},
        {"relevance": "warning"},
        {"relevance": "call"},
        {"relevance": "warning"},
        {"relevance": "error"},
    ]
    assert find_last_bug_frame(chronological) == 3


def test_find_last_bug_frame_no_warning():
    """No warning frames means no bug origin index."""
    chronological = [
        {"relevance": "call"},
        {"relevance": "error"},
    ]
    assert find_last_bug_frame(chronological) is None

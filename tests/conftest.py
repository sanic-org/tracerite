"""Shared pytest fixtures for the tracerite test suite."""

import pytest


@pytest.fixture(autouse=True)
def _clear_color_env(monkeypatch):
    """Keep externally set NO_COLOR/FORCE_COLOR from leaking into tests.

    The tty module honors these environment variables, so an externally set
    value would change test outcomes depending on the caller's environment.
    Tests for this behavior set the variables explicitly via monkeypatch.
    """
    monkeypatch.delenv("NO_COLOR", raising=False)
    monkeypatch.delenv("FORCE_COLOR", raising=False)

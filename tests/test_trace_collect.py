"""Tests for tracerite.trace.collect."""

from tracerite.trace.collect import collect_exception_objects


def test_prefers_explicit_cause_even_when_context_not_suppressed():
    """Explicit ``__cause__`` must be followed even if ``__suppress_context__`` is False."""
    cause = ValueError("explicit cause")
    context = TypeError("implicit context")
    outer = RuntimeError("outer")
    outer.__cause__ = cause
    outer.__context__ = context
    outer.__suppress_context__ = False

    chain = collect_exception_objects(outer)

    assert chain == [cause, outer]

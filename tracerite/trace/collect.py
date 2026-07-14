from __future__ import annotations

__all__ = ["_collect_exception_objects", "_collect_exception_chain"]


def _collect_exception_objects(exc=None) -> list:
    """Return the live exception objects in chronological order, oldest first."""
    import sys

    chain = []
    exc = exc or sys.exc_info()[1]
    while exc:
        chain.append(exc)
        exc = exc.__cause__ or None if exc.__suppress_context__ else exc.__context__
    return list(reversed(chain))


def _collect_exception_chain(exc=None, **kwargs) -> list[dict]:
    """Return raw exception objects in chronological order, oldest first.

    Each element is a small metadata dict with the live exception object and
    the kwargs that should be passed when it is digested.  Skip-related kwargs
    are attached only to the outermost (newest) exception.
    """
    objects = _collect_exception_objects(exc)
    return [{"exc": e, "kwargs": kwargs if e is objects[-1] else {}} for e in objects]

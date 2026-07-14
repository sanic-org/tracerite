"""Test-only helpers that wrap internal trace stages for convenience."""

from __future__ import annotations

from tracerite.trace.digest import digest_exception, digest_frames
from tracerite.trace.finalize import fill_variables, set_relevances


def extract_exception(exc, **kwargs):
    """Digest *exc* and apply relevance/variable finalization like the old API."""
    info = digest_exception(exc, **kwargs)
    frames = info.get("frames", [])
    set_relevances(frames, exc)
    fill_variables(frames, info.get("message"))
    return info


def extract_frames(tb, raw_tb=None, *, exc=None, except_block=False, cache=None):
    """Digest frames and, when *exc* is given, apply relevance/variables."""
    frames = digest_frames(tb, raw_tb, except_block=except_block, cache=cache)
    if exc is not None:
        set_relevances(frames, exc)
        fill_variables(frames, str(exc))
    else:
        for frame in frames:
            frame.setdefault("variables", [])
    return frames

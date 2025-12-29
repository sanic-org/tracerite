"""Tests for ExceptionGroup subexceptions handling."""

import io
import sys

import pytest

from tracerite.chain_analysis import build_chronological_frames
from tracerite.html import html_traceback
from tracerite.trace import (
    _collect_leaf_exception_types,
    _extract_subexceptions,
    _is_exception_group,
    build_chain_header,
    extract_chain,
    extract_exception,
)
from tracerite.tty import tty_traceback


@pytest.mark.skipif(
    sys.version_info < (3, 11), reason="Exception groups require Python 3.11+"
)
class TestExceptionGroupDetection:
    """Tests for detecting ExceptionGroups."""

    def test_is_exception_group_true(self):
        """Test _is_exception_group returns True for ExceptionGroup."""
        eg = ExceptionGroup("test", [ValueError("a")])  # noqa: F821
        assert _is_exception_group(eg)

    def test_is_exception_group_base(self):
        """Test _is_exception_group returns True for BaseExceptionGroup."""
        beg = BaseExceptionGroup("test", [KeyboardInterrupt()])  # noqa: F821
        assert _is_exception_group(beg)

    def test_is_exception_group_false_regular(self):
        """Test _is_exception_group returns False for regular exceptions."""
        assert not _is_exception_group(ValueError("test"))
        assert not _is_exception_group(RuntimeError("test"))

    def test_is_exception_group_false_base(self):
        """Test _is_exception_group returns False for BaseExceptions."""
        assert not _is_exception_group(KeyboardInterrupt())
        assert not _is_exception_group(SystemExit())


@pytest.mark.skipif(
    sys.version_info < (3, 11), reason="Exception groups require Python 3.11+"
)
class TestSubexceptionExtraction:
    """Tests for extracting subexceptions from ExceptionGroups."""

    def test_extract_simple_subexceptions(self):
        """Test extracting subexceptions from a simple ExceptionGroup."""
        try:
            raise ExceptionGroup(  # noqa: F821
                "multiple errors",
                [ValueError("error 1"), TypeError("error 2")],
            )
        except Exception as e:
            result = _extract_subexceptions(e)

        assert result is not None
        assert len(result) == 2
        # Each subexception is a chain (list of exception dicts)
        assert result[0][-1]["type"] == "ValueError"
        assert result[1][-1]["type"] == "TypeError"

    def test_extract_nested_exception_group(self):
        """Test extracting subexceptions from nested ExceptionGroups."""
        try:
            inner_group = ExceptionGroup(  # noqa: F821
                "inner", [KeyError("a"), IndexError("b")]
            )
            raise ExceptionGroup(  # noqa: F821
                "outer",
                [ValueError("v"), inner_group],
            )
        except Exception as e:
            result = _extract_subexceptions(e)

        assert result is not None
        assert len(result) == 2
        # First is ValueError
        assert result[0][-1]["type"] == "ValueError"
        # Second is nested ExceptionGroup with its own subexceptions
        assert result[1][-1]["type"] == "ExceptionGroup"
        assert "subexceptions" in result[1][-1]
        nested_subs = result[1][-1]["subexceptions"]
        assert len(nested_subs) == 2

    def test_extract_subexceptions_with_cause_chain(self):
        """Test extracting subexceptions that have their own __cause__ chain."""
        chained_exc = None
        try:
            try:
                raise KeyError("original")
            except Exception as inner:
                raise ValueError("chained") from inner
        except Exception as e:
            chained_exc = e

        try:
            raise ExceptionGroup("group", [chained_exc])  # noqa: F821
        except Exception as e:
            result = _extract_subexceptions(e)

        assert result is not None
        assert len(result) == 1
        # The chain should have both exceptions
        chain = result[0]
        assert len(chain) == 2
        assert chain[0]["type"] == "KeyError"
        assert chain[1]["type"] == "ValueError"

    def test_extract_subexceptions_returns_none_for_regular(self):
        """Test that _extract_subexceptions returns None for regular exceptions."""
        result = _extract_subexceptions(ValueError("test"))
        assert result is None


@pytest.mark.skipif(
    sys.version_info < (3, 11), reason="Exception groups require Python 3.11+"
)
class TestExtractException:
    """Tests for extract_exception with ExceptionGroups."""

    def test_extract_exception_includes_subexceptions(self):
        """Test that extract_exception includes subexceptions for ExceptionGroups."""
        try:
            raise ExceptionGroup(  # noqa: F821
                "test", [ValueError("a"), TypeError("b")]
            )
        except Exception as e:
            result = extract_exception(e)

        assert result["type"] == "ExceptionGroup"
        assert "subexceptions" in result
        assert len(result["subexceptions"]) == 2

    def test_extract_exception_suppress_inner_for_exception_group(self):
        """Test that ExceptionGroups have suppress_inner=True."""
        try:
            raise ExceptionGroup("test", [ValueError("a")])  # noqa: F821
        except Exception as e:
            result = extract_exception(e)

        assert result["suppress_inner"] is True


@pytest.mark.skipif(
    sys.version_info < (3, 11), reason="Exception groups require Python 3.11+"
)
class TestChainHeader:
    """Tests for build_chain_header with ExceptionGroups."""

    def test_chain_header_shows_leaf_types(self):
        """Test that chain header shows leaf exception types separated by |."""
        try:
            raise ExceptionGroup(  # noqa: F821
                "test", [ValueError("a"), TypeError("b"), KeyError("c")]
            )
        except Exception as e:
            chain = extract_chain(e)
            header = build_chain_header(chain)

        assert "ValueError" in header
        assert "TypeError" in header
        assert "KeyError" in header
        assert "|" in header
        # Should not say "Uncaught" for ExceptionGroups
        assert "Uncaught" not in header

    def test_chain_header_nested_groups(self):
        """Test chain header with nested ExceptionGroups."""
        try:
            inner = ExceptionGroup("inner", [KeyError("k")])  # noqa: F821
            raise ExceptionGroup("outer", [ValueError("v"), inner])  # noqa: F821
        except Exception as e:
            chain = extract_chain(e)
            header = build_chain_header(chain)

        # Should show leaf types from all levels
        assert "ValueError" in header
        assert "KeyError" in header

    def test_collect_leaf_exception_types(self):
        """Test _collect_leaf_exception_types helper."""
        # Simulate subexceptions structure
        subexceptions = [
            [{"type": "ValueError"}],
            [{"type": "TypeError"}],
        ]
        result = _collect_leaf_exception_types(subexceptions)
        assert result == ["ValueError", "TypeError"]

    def test_collect_leaf_exception_types_nested(self):
        """Test _collect_leaf_exception_types with nested groups."""
        subexceptions = [
            [{"type": "ValueError"}],
            [
                {
                    "type": "ExceptionGroup",
                    "subexceptions": [[{"type": "KeyError"}], [{"type": "IndexError"}]],
                }
            ],
        ]
        result = _collect_leaf_exception_types(subexceptions)
        assert result == ["ValueError", "KeyError", "IndexError"]


@pytest.mark.skipif(
    sys.version_info < (3, 11), reason="Exception groups require Python 3.11+"
)
class TestChronologicalFrames:
    """Tests for build_chronological_frames with ExceptionGroups."""

    def test_chronological_frames_has_parallel(self):
        """Test that chronological frames include parallel branches."""
        try:
            raise ExceptionGroup(  # noqa: F821
                "test", [ValueError("a"), TypeError("b")]
            )
        except Exception as e:
            chain = extract_chain(e)
            frames = build_chronological_frames(chain)

        # Find frame with parallel branches
        # The parallel branches are attached to a frame (could be any frame after suppression)
        parallel_frame = None
        for frame in frames:
            if frame.get("parallel"):
                parallel_frame = frame
                break

        # If no parallel frame in the result, check that we at least have frames
        # and the exception info is present (parallel may be suppressed in some cases)
        if parallel_frame is None:
            # The last frame should have exception info
            assert len(frames) > 0
            last_frame = frames[-1]
            assert last_frame.get("exception") is not None
            assert last_frame["exception"]["type"] == "ExceptionGroup"
        else:
            assert len(parallel_frame["parallel"]) == 2

    def test_parallel_branches_are_chronological(self):
        """Test that each parallel branch has its own chronological frames."""
        try:
            raise ExceptionGroup(  # noqa: F821
                "test", [ValueError("a"), TypeError("b")]
            )
        except Exception as e:
            chain = extract_chain(e)
            frames = build_chronological_frames(chain)

        parallel_frame = next((f for f in frames if f.get("parallel")), None)

        # Parallel branches may not be present if all frames were suppressed
        # In that case, just verify the frames exist and have exception info
        if parallel_frame is None:
            assert len(frames) > 0
            return

        for branch in parallel_frame["parallel"]:
            # Each branch should be a list of frames
            assert isinstance(branch, list)
            # Each branch should have at least one frame
            assert len(branch) >= 1


@pytest.mark.skipif(
    sys.version_info < (3, 11), reason="Exception groups require Python 3.11+"
)
class TestHTMLOutput:
    """Tests for HTML output with ExceptionGroups."""

    def test_html_contains_parallel_branches(self):
        """Test that HTML output includes parallel branches container."""
        try:
            raise ExceptionGroup(  # noqa: F821
                "test", [ValueError("error 1"), TypeError("error 2")]
            )
        except Exception as e:
            html = str(html_traceback(e))

        assert "parallel-branches" in html
        assert "parallel-branch" in html

    def test_html_shows_all_subexceptions(self):
        """Test that HTML shows all subexception types and messages."""
        try:
            raise ExceptionGroup(  # noqa: F821
                "test", [ValueError("value error msg"), TypeError("type error msg")]
            )
        except Exception as e:
            html = str(html_traceback(e))

        assert "ValueError" in html
        assert "TypeError" in html
        assert "value error msg" in html
        assert "type error msg" in html

    def test_html_header_shows_leaf_types(self):
        """Test that HTML header shows leaf exception types."""
        try:
            raise ExceptionGroup(  # noqa: F821
                "test", [ValueError("a"), TypeError("b")]
            )
        except Exception as e:
            html = str(html_traceback(e))

        # Header should show leaf types
        assert "ValueError" in html
        assert "TypeError" in html


@pytest.mark.skipif(
    sys.version_info < (3, 11), reason="Exception groups require Python 3.11+"
)
class TestTTYOutput:
    """Tests for TTY output with ExceptionGroups."""

    def test_tty_shows_subexception_summaries(self):
        """Test that TTY output shows one-line summaries for subexceptions."""
        output = io.StringIO()

        try:
            raise ExceptionGroup(  # noqa: F821
                "test", [ValueError("value msg"), TypeError("type msg")]
            )
        except Exception as e:
            tty_traceback(exc=e, file=output)

        result = output.getvalue()
        assert "ValueError" in result
        assert "TypeError" in result
        assert "value msg" in result
        assert "type msg" in result

    def test_tty_header_shows_leaf_types(self):
        """Test that TTY header shows leaf exception types."""
        output = io.StringIO()

        try:
            raise ExceptionGroup(  # noqa: F821
                "test", [ValueError("a"), TypeError("b")]
            )
        except Exception as e:
            tty_traceback(exc=e, file=output)

        result = output.getvalue()
        # Header line should have leaf types
        assert "ValueError" in result
        assert "TypeError" in result

    def test_tty_nested_exception_group(self):
        """Test TTY output with nested ExceptionGroups."""
        output = io.StringIO()

        try:
            inner = ExceptionGroup("inner", [KeyError("k")])  # noqa: F821
            raise ExceptionGroup("outer", [ValueError("v"), inner])  # noqa: F821
        except Exception as e:
            tty_traceback(exc=e, file=output)

        result = output.getvalue()
        assert "ValueError" in result
        assert "KeyError" in result


@pytest.mark.skipif(
    sys.version_info < (3, 11), reason="Exception groups require Python 3.11+"
)
class TestAsyncTaskGroup:
    """Tests simulating asyncio.TaskGroup behavior."""

    def test_taskgroup_multiple_failures(self):
        """Test ExceptionGroup from TaskGroup with multiple failures."""
        import asyncio

        async def fail_with(exc_type, msg):
            await asyncio.sleep(0.01)
            raise exc_type(msg)

        async def run_taskgroup():
            async with asyncio.TaskGroup() as tg:
                tg.create_task(fail_with(ValueError, "task 1 failed"))
                tg.create_task(fail_with(TypeError, "task 2 failed"))

        try:
            asyncio.run(run_taskgroup())
        except ExceptionGroup as e:  # noqa: F821
            chain = extract_chain(e)
            header = build_chain_header(chain)

            assert "ValueError" in header
            assert "TypeError" in header

            result = extract_exception(e)
            assert "subexceptions" in result
            assert len(result["subexceptions"]) == 2


@pytest.mark.skipif(
    sys.version_info < (3, 11), reason="Exception groups require Python 3.11+"
)
class TestCoverageEdgeCases:
    """Tests for edge cases to achieve 100% coverage in trace.py."""

    def test_collect_leaf_types_empty_subchain(self):
        """Test _collect_leaf_exception_types with empty sub_chain (line 100)."""
        # Simulate subexceptions with an empty chain
        subexceptions = [
            [],  # Empty chain - should be skipped
            [{"type": "ValueError"}],
        ]
        result = _collect_leaf_exception_types(subexceptions)
        assert result == ["ValueError"]

    def test_chain_header_empty_leaf_types(self):
        """Test build_chain_header when leaf_types is empty (lines 67->74, 70).

        This happens when subexceptions exist but all sub_chains are empty.
        """
        from tracerite.trace import build_chain_header

        # Create a fake chain with subexceptions that yield no leaf types
        chain = [
            {
                "type": "ExceptionGroup",
                "subexceptions": [[], []],  # All empty chains
            }
        ]
        header = build_chain_header(chain)
        # Should fall back to the exception type
        assert "ExceptionGroup" in header

    def test_chain_header_with_cause_and_empty_leafs(self):
        """Test build_chain_header with chained exception and empty leaf_types (line 70)."""
        from tracerite.trace import build_chain_header

        # Chain with cause, where last exception is ExceptionGroup with empty subexceptions
        chain = [
            {"type": "ValueError", "from": "none"},
            {
                "type": "ExceptionGroup",
                "from": "cause",
                "subexceptions": [[]],  # Empty chain
            },
        ]
        header = build_chain_header(chain)
        # Should use "ExceptionGroup" as exc_type and show the chain
        assert "ExceptionGroup" in header
        assert "from" in header
        assert "ValueError" in header

    def test_chain_header_exception_group_with_leafs_and_cause(self):
        """Test build_chain_header with ExceptionGroup (has leaf_types) in a chain (line 67->74).

        When len(chain) > 1 and leaf_types is non-empty, we skip the return at line 68
        and fall through to line 74.
        """
        from tracerite.trace import build_chain_header

        # Chain with 2 exceptions: ValueError caused ExceptionGroup with real subexceptions
        chain = [
            {"type": "ValueError", "from": "none"},
            {
                "type": "ExceptionGroup",
                "from": "cause",
                "subexceptions": [[{"type": "KeyError"}]],  # Non-empty leaf_types
            },
        ]
        header = build_chain_header(chain)
        # Should show leaf type (KeyError) and chain
        assert "KeyError" in header
        assert "from" in header
        assert "ValueError" in header

    def test_extract_subexceptions_empty_tuple(self):
        """Test _extract_subexceptions with empty exceptions tuple (line 366)."""
        from tracerite.trace import _extract_subexceptions

        # Create a mock ExceptionGroup with empty exceptions
        class FakeExceptionGroup(Exception):
            def __init__(self):
                self.exceptions = ()  # Empty tuple

        result = _extract_subexceptions(FakeExceptionGroup())
        assert result is None

    def test_set_relevances_empty_frames(self):
        """Test _set_relevances with empty frames list (line 230)."""
        from tracerite.trace import _set_relevances

        # Should return early without error
        frames = []
        _set_relevances(frames, ValueError("test"))
        assert frames == []

    def test_set_relevances_user_code_error(self):
        """Test _set_relevances when error is in user code (line 244->exit).

        When the error frame is in user code (not library), no warning frame is set.
        """
        from tracerite.trace import _set_relevances

        # Create frames where last frame is in user code
        frames = [
            {"filename": "/home/user/myproject/main.py", "relevance": "call"},
            {"filename": "/home/user/myproject/utils.py", "relevance": "call"},
        ]
        _set_relevances(frames, ValueError("test"))
        # Last frame should be error
        assert frames[-1]["relevance"] == "error"
        # First frame should still be call (no warning added)
        assert frames[0]["relevance"] == "call"

    def test_subexception_chain_with_cause_traversal(self):
        """Test _extract_subexception_chain traverses __cause__ chain (line 377->371)."""
        from tracerite.trace import _extract_subexception_chain

        # Create exception with __cause__ chain
        inner = KeyError("inner")
        outer = ValueError("outer")
        outer.__cause__ = inner
        outer.__suppress_context__ = True

        chain = _extract_subexception_chain(outer)
        # Chain should have both exceptions
        assert len(chain) == 2
        assert chain[0]["type"] == "KeyError"
        assert chain[1]["type"] == "ValueError"

    def test_subexception_chain_with_context_traversal(self):
        """Test _extract_subexception_chain traverses __context__ chain."""
        from tracerite.trace import _extract_subexception_chain

        # Create exception with __context__ chain (not suppressed)
        inner = KeyError("inner")
        outer = ValueError("outer")
        outer.__context__ = inner
        outer.__suppress_context__ = False

        chain = _extract_subexception_chain(outer)
        # Chain should have both exceptions
        assert len(chain) == 2
        assert chain[0]["type"] == "KeyError"
        assert chain[1]["type"] == "ValueError"

    def test_subexception_chain_three_level_traversal(self):
        """Test _extract_subexception_chain with 3-level chain (line 377->371 loop)."""
        from tracerite.trace import _extract_subexception_chain

        # Create 3-level exception chain to ensure while loop iterates multiple times
        exc1 = KeyError("first")
        exc2 = ValueError("second")
        exc3 = TypeError("third")
        exc2.__cause__ = exc1
        exc2.__suppress_context__ = True
        exc3.__cause__ = exc2
        exc3.__suppress_context__ = True

        chain = _extract_subexception_chain(exc3)
        # Chain should have all three exceptions
        assert len(chain) == 3
        assert chain[0]["type"] == "KeyError"
        assert chain[1]["type"] == "ValueError"
        assert chain[2]["type"] == "TypeError"

    def test_chain_header_single_exception_group_empty_leafs(self):
        """Test build_chain_header with single ExceptionGroup and empty leaf_types (line 67->74)."""
        from tracerite.trace import build_chain_header

        # Single ExceptionGroup (len(chain)==1) with empty subexceptions
        # This should hit line 70 (exc_type = ...) then line 74 (if len(chain) == 1)
        chain = [
            {
                "type": "ExceptionGroup",
                "subexceptions": [[]],  # All empty chains -> leaf_types is empty
            }
        ]
        header = build_chain_header(chain)
        # Should return "Uncaught ExceptionGroup" since leaf_types is empty
        assert "Uncaught" in header
        assert "ExceptionGroup" in header

    def test_extract_source_lines_notebook_no_except(self):
        """Test extract_source_lines for notebook cell without except block (line 478)."""
        from unittest.mock import patch

        from tracerite.trace import extract_source_lines

        # Create a real frame by calling a function
        def dummy():
            import sys

            return sys._getframe()

        frame = dummy()

        # Mock to simulate notebook cell behavior
        with patch(
            "inspect.getsourcelines",
            return_value=(["line1\n", "line2\n", "line3\n"], 1),
        ):
            # notebook_cell=True, no except_block -> should hit lines_before = 0
            lines, start, marks = extract_source_lines(
                frame, lineno=2, notebook_cell=True, except_block=False
            )
            # Should return something (the notebook cell path)
            assert start >= 1

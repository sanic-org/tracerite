"""Tests for with/async-with block error handling.

Python 3.12+ marks errors raised by ``__enter__``/``__exit__`` (e.g. TaskGroup
ExceptionGroups) on the with statement's context expression, even though the
block body already ran.  Three cases are handled:

1. Context expression failure (block never entered): Python's own highlight
   on the failing expression is kept, relevance stays call/error.
2. Enter failure: the entire with statement is marked and the initialising
   expression is emphasized, relevance stop, but the context is not extended
   (the block never ran).
3. Exit failure: the entire with statement is marked without emphasis,
   relevance stop, and the context is extended to cover the block body
   (e.g. all tasks created in a TaskGroup block).

The scenario functions live in errorcases.py (lint-exempt) because their
marker assignments are the source material under test.
"""

import sys
import textwrap
from types import SimpleNamespace

import pytest

from tests.errorcases import (
    WithPassthrough,
    with_body_raises,
    with_c_level_exit_raises,
    with_enter_raises,
    with_exit_raises,
    with_exit_raises_on_error,
    with_expression_raises,
    with_long_block,
    with_multiline_header,
)
from tracerite.trace.chain_analysis import (
    find_with_block_for_header_line,
    parse_source_for_with_blocks,
)
from tracerite.trace.digest import detect_with_block_error

from .helpers import extract_exception

pytestmark = pytest.mark.skipif(
    sys.version_info < (3, 11),
    reason="with-block error detection requires Python 3.11+",
)


def _frame_for(info, name):
    """Return the digested frame for the given function name."""
    return next(f for f in info["frames"] if f["function"] == name)


def _marked_text(frame):
    """Concatenate the marked fragment text of a frame."""
    return "".join(
        frag["code"]
        for line in frame["fragments"]
        for frag in line["fragments"]
        if frag.get("mark")
    )


def _has_emphasis(frame):
    """Check whether any fragment of the frame carries caret emphasis."""
    return any(
        "em" in frag for line in frame["fragments"] for frag in line["fragments"]
    )


def _emphasized_text(frame):
    """Concatenate the emphasized fragment text of a frame."""
    return "".join(
        frag["code"]
        for line in frame["fragments"]
        for frag in line["fragments"]
        if frag.get("em")
    )


class TestWithExitFailure:
    """Exit failure: full statement marked, block context shown, stop relevance."""

    def test_exit_failure_extends_context(self):
        """The whole block body is shown, past the default two lines."""
        try:
            with_exit_raises()
        except RuntimeError as e:
            info = extract_exception(e)

        frame = _frame_for(info, "with_exit_raises")
        assert "exit_marker_three = 3" in frame["lines"]

    def test_exit_failure_marks_entire_statement(self):
        """The mark covers the with statement header, not just the expression."""
        try:
            with_exit_raises()
        except RuntimeError as e:
            info = extract_exception(e)

        frame = _frame_for(info, "with_exit_raises")
        assert _marked_text(frame) == "with ExitRaises() as cm:"
        assert not _has_emphasis(frame)

    def test_exit_failure_relevance_and_trace_continue(self):
        """With frame is stop, and the trace continues to the __exit__ error."""
        try:
            with_exit_raises()
        except RuntimeError as e:
            info = extract_exception(e)

        frame = _frame_for(info, "with_exit_raises")
        assert frame["_with_stage"] == "exit"
        assert frame["relevance"] == "stop"
        exit_frame = _frame_for(info, "ExitRaises.__exit__")
        assert exit_frame["relevance"] == "error"

    def test_exit_failure_while_handling_extends_context(self):
        """__exit__ raising while handling a body error also extends context."""
        try:
            with_exit_raises_on_error()
        except RuntimeError as e:
            info = extract_exception(e)

        frame = _frame_for(info, "with_exit_raises_on_error")
        assert frame["_with_stage"] == "exit"
        assert "chained_marker_three = 3" in frame["lines"]

    def test_c_level_exit_failure_extends_context(self):
        """C-level __exit__ failure (no Python frame below) is still detected."""
        try:
            with_c_level_exit_raises()
        except RuntimeError as e:
            info = extract_exception(e)

        frame = _frame_for(info, "with_c_level_exit_raises")
        assert frame["_with_stage"] == "exit"
        assert frame["relevance"] == "stop"
        assert "c_marker_three = 3" in frame["lines"]

    def test_multiline_header_marks_entire_statement(self):
        """Multi-line parenthesized with headers are marked in full."""
        try:
            with_multiline_header()
        except RuntimeError as e:
            info = extract_exception(e)

        frame = _frame_for(info, "with_multiline_header")
        marked = _marked_text(frame)
        assert marked.startswith("with (")
        assert marked.endswith("):")
        assert "ExitRaises() as b," in marked
        assert "multi_marker_three = 3" in frame["lines"]

    def test_context_capped_at_twenty_lines(self):
        """Block context is shown at most 20 lines into the block."""
        try:
            with_long_block()
        except RuntimeError as e:
            info = extract_exception(e)

        frame = _frame_for(info, "with_long_block")
        assert "long_line_20 = 20" in frame["lines"]
        assert "long_line_21 = 21" not in frame["lines"]


class TestWithEnterFailure:
    """Enter failure: full statement marked, default context, stop relevance."""

    def test_enter_failure_keeps_default_context(self):
        """The block never ran, so its lines are not added."""
        try:
            with_enter_raises()
        except RuntimeError as e:
            info = extract_exception(e)

        frame = _frame_for(info, "with_enter_raises")
        assert frame["_with_stage"] == "enter"
        assert "enter_marker_two = 2" in frame["lines"]
        assert "enter_marker_three = 3" not in frame["lines"]

    def test_enter_failure_marks_entire_statement(self):
        """The whole statement is marked, with the initialising expression emphasized."""
        try:
            with_enter_raises()
        except RuntimeError as e:
            info = extract_exception(e)

        frame = _frame_for(info, "with_enter_raises")
        assert _marked_text(frame) == "with EnterRaises() as cm:"
        assert _emphasized_text(frame) == "EnterRaises()"

    def test_enter_failure_relevance_and_trace_continue(self):
        """With frame is stop, and the trace continues to the __enter__ error."""
        try:
            with_enter_raises()
        except RuntimeError as e:
            info = extract_exception(e)

        frame = _frame_for(info, "with_enter_raises")
        assert frame["relevance"] == "stop"
        enter_frame = _frame_for(info, "EnterRaises.__enter__")
        assert enter_frame["relevance"] == "error"


class TestWithExpressionFailure:
    """Expression failure: block never entered, Python's highlight is kept."""

    def test_expression_failure_keeps_default_context(self):
        """No block context is added beyond the default two lines."""
        try:
            with_expression_raises()
        except FileNotFoundError as e:
            info = extract_exception(e)

        frame = _frame_for(info, "with_expression_raises")
        assert frame["_with_stage"] is None
        assert "expr_marker_two = 2" in frame["lines"]
        assert "expr_marker_three = 3" not in frame["lines"]

    def test_expression_failure_keeps_expression_mark(self):
        """Python's own expression mark and error relevance are kept."""
        try:
            with_expression_raises()
        except FileNotFoundError as e:
            info = extract_exception(e)

        frame = _frame_for(info, "with_expression_raises")
        assert _marked_text(frame) == 'open("/nonexistent/path/file.txt")'
        assert frame["relevance"] == "error"


class TestWithBodyFailure:
    """Ordinary body errors mark the body line, not the with header."""

    def test_body_failure_marks_body_line_not_header(self):
        try:
            with_body_raises()
        except ZeroDivisionError as e:
            info = extract_exception(e)

        frame = _frame_for(info, "with_body_raises")
        assert frame["_with_stage"] is None
        assert frame["codeline"] == "body_marker = 1 / 0"
        assert frame["relevance"] == "error"


class TestTaskGroupContext:
    """TaskGroup failures show all tasks created inside the block."""

    def test_taskgroup_shows_all_created_tasks(self):
        import asyncio

        async def fail_soon():
            await asyncio.sleep(0.001)
            raise ValueError("task failed")

        async def run_failing_taskgroup():
            async with asyncio.TaskGroup() as tg:
                tg.create_task(fail_soon())
                tg.create_task(fail_soon())
                tg.create_task(fail_soon())

        try:
            asyncio.run(run_failing_taskgroup())
        except BaseException as e:
            info = extract_exception(e)

        frame = _frame_for(info, "run_failing_taskgroup")
        assert frame["_with_stage"] == "exit"
        # All three create_task lines are shown, not just the default two
        assert frame["lines"].count("tg.create_task(fail_soon())") == 3

    def test_taskgroup_marks_entire_statement(self):
        import asyncio

        async def fail_soon():
            await asyncio.sleep(0.001)
            raise ValueError("task failed")

        async def run_failing_taskgroup():
            async with asyncio.TaskGroup() as tg:
                tg.create_task(fail_soon())

        try:
            asyncio.run(run_failing_taskgroup())
        except BaseException as e:
            info = extract_exception(e)

        frame = _frame_for(info, "run_failing_taskgroup")
        assert _marked_text(frame) == "async with asyncio.TaskGroup() as tg:"
        assert not _has_emphasis(frame)


class TestWithBlockParsing:
    """Unit tests for the AST with-block analysis helpers."""

    def test_parse_and_find_with_blocks(self, tmp_path):
        src = textwrap.dedent(
            """\
            with a():
                with b():
                    x = 1

            async def f():
                async with c() as y:
                    await y
            """
        )
        path = tmp_path / "with_blocks.py"
        path.write_text(src)

        blocks = parse_source_for_with_blocks(str(path))
        assert len(blocks) == 3

        outer = find_with_block_for_header_line(blocks, 1)
        assert outer["header_start"] == 1
        inner = find_with_block_for_header_line(blocks, 2)
        assert inner["header_start"] == 2
        # A line inside the block body never matches
        assert find_with_block_for_header_line(blocks, 3) is None
        # async with headers are found as well
        assert find_with_block_for_header_line(blocks, 6)["body_start"] == 7

    def test_oneliner_with_never_matches(self, tmp_path):
        path = tmp_path / "oneliner.py"
        path.write_text("with a(): pass\n")
        blocks = parse_source_for_with_blocks(str(path))
        assert find_with_block_for_header_line(blocks, 1) is None

    def test_parse_unreadable_file_returns_empty(self):
        assert parse_source_for_with_blocks("/nonexistent/module.py") == []

    def test_detect_with_block_error_no_with(self):
        """Frames not on a with statement header yield no stage."""
        frame = sys._getframe()
        stage, block = detect_with_block_error(frame, 1, None, "__enter__")
        assert stage is None
        assert block is None

    def test_detect_with_block_error_without_lasti(self):
        """C-level fallback without an instruction offset yields no stage."""

        def with_block_holder():
            with WithPassthrough():
                pass

        code = with_block_holder.__code__
        fake_frame = SimpleNamespace(f_code=code)
        stage, block = detect_with_block_error(
            fake_frame, code.co_firstlineno + 1, None, None
        )
        assert stage is None
        assert block is None

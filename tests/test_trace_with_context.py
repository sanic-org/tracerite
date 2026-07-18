"""Tests for with/async-with block error handling.

Python 3.12+ marks errors raised by ``__enter__``/``__exit__`` (e.g. TaskGroup
ExceptionGroups) on the with statement's context expression, even though the
block body already ran.  Three cases are handled:

1. Context expression failure (block never entered): Python's own highlight
   on the failing expression is kept, relevance stays call/error.
2. Enter failure: the entire with statement is marked with the initialising
   expression emphasized, relevance stop ("Entering with block"), but the
   context is not extended (the block never ran).
3. Exit failure: the entire with statement is marked with the context
   expression emphasized, relevance stop ("Exiting with block"), and the
   context is extended to cover the block body (e.g. all tasks created in a
   TaskGroup block).

The scenario functions live in errorcases.py (lint-exempt) because their
marker assignments are the source material under test.
"""

import sys
import textwrap

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
    with_multi_item_enter_fails,
    with_multi_item_exit_fails,
    with_multi_item_first_enter_fails,
    with_multiline_enter_fails,
    with_multiline_header,
)
from tracerite.trace.chain_analysis import (
    find_with_block_for_header_line,
    parse_source_string_for_with_blocks,
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
        """The statement is marked, with the context expression emphasized."""
        try:
            with_exit_raises()
        except RuntimeError as e:
            info = extract_exception(e)

        frame = _frame_for(info, "with_exit_raises")
        assert _marked_text(frame) == "with ExitRaises() as cm:"
        assert _emphasized_text(frame) == "ExitRaises()"
        assert frame["symbol_desc"] == "Exiting with block"

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
        if sys.version_info >= (3, 12):
            # 3.12+ positions identify the failing item; 3.11 marks the
            # whole statement, so no item is emphasized there.
            assert _emphasized_text(frame) == "ExitRaises()"
        else:
            assert not _has_emphasis(frame)
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
        assert frame["symbol_desc"] == "Entering with block"

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
        assert "symbol_desc" not in frame


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


class TestWithMultipleItems:
    """Comma-separated with items: emphasize the one that failed to enter."""

    def test_multi_item_enter_fails_emphasizes_failing_item(self):
        """With several items on one line, em is on the item that failed."""
        try:
            with_multi_item_enter_fails()
        except RuntimeError as e:
            info = extract_exception(e)

        frame = _frame_for(info, "with_multi_item_enter_fails")
        assert frame["_with_stage"] == "enter"
        assert _marked_text(frame) == "with WithPassthrough() as a, EnterRaises() as b:"
        assert _emphasized_text(frame) == "EnterRaises()"
        assert "multi_enter_marker_three = 3" not in frame["lines"]

    def test_multi_item_first_enter_fails(self):
        """The first item is emphasized when it is the one that failed."""
        try:
            with_multi_item_first_enter_fails()
        except RuntimeError as e:
            info = extract_exception(e)

        frame = _frame_for(info, "with_multi_item_first_enter_fails")
        assert frame["_with_stage"] == "enter"
        assert _marked_text(frame) == "with EnterRaises() as a, WithPassthrough() as b:"
        assert _emphasized_text(frame) == "EnterRaises()"

    def test_multiline_enter_fails(self):
        """Multi-line header: full statement marked, failing item emphasized."""
        try:
            with_multiline_enter_fails()
        except RuntimeError as e:
            info = extract_exception(e)

        frame = _frame_for(info, "with_multiline_enter_fails")
        assert frame["_with_stage"] == "enter"
        marked = _marked_text(frame)
        assert marked.startswith("with (")
        assert marked.endswith("):")
        assert "EnterRaises() as b," in marked
        assert _emphasized_text(frame) == "EnterRaises()"
        assert "ml_enter_marker_three = 3" not in frame["lines"]

    def test_multi_item_exit_fails(self):
        """Exit failure on a later item still extends to the block end."""
        try:
            with_multi_item_exit_fails()
        except RuntimeError as e:
            info = extract_exception(e)

        frame = _frame_for(info, "with_multi_item_exit_fails")
        assert frame["_with_stage"] == "exit"
        assert _marked_text(frame) == "with WithPassthrough() as a, ExitRaises() as b:"
        if sys.version_info >= (3, 12):
            assert _emphasized_text(frame) == "ExitRaises()"
        else:
            assert not _has_emphasis(frame)
        assert "multi_exit_marker_three = 3" in frame["lines"]


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
        assert _emphasized_text(frame) == "asyncio.TaskGroup()"


class TestWithBlockParsing:
    """Unit tests for the AST with-block analysis helpers."""

    def test_parse_and_find_with_blocks(self):
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
        blocks = parse_source_string_for_with_blocks(src)
        assert len(blocks) == 3

        outer = find_with_block_for_header_line(blocks, 1)
        assert outer["header_start"] == 1
        inner = find_with_block_for_header_line(blocks, 2)
        assert inner["header_start"] == 2
        # A line inside the block body never matches
        assert find_with_block_for_header_line(blocks, 3) is None
        # async with headers are found as well
        assert find_with_block_for_header_line(blocks, 6)["body_start"] == 7

    def test_oneliner_with_never_matches(self):
        blocks = parse_source_string_for_with_blocks("with a(): pass\n")
        assert find_with_block_for_header_line(blocks, 1) is None

    def test_parse_empty_source_returns_empty(self):
        assert parse_source_string_for_with_blocks("") == []

    def test_detect_with_block_error_no_with(self):
        """Frames not on a with statement header yield no stage."""
        frame = sys._getframe()
        stage, block = detect_with_block_error(frame, 1, None, "__enter__")
        assert stage is None
        assert block is None

    def test_detect_with_block_error_without_lasti(self):
        """C-level fallback without an instruction offset yields no stage."""
        frame = None

        def with_block_holder():
            nonlocal frame
            frame = sys._getframe()
            with WithPassthrough():
                pass

        with_block_holder()
        code = frame.f_code
        stage, block = detect_with_block_error(
            frame, code.co_firstlineno + 3, None, None
        )
        assert stage is None
        assert block is None

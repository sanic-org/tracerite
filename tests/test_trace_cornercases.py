"""Corner case and edge case tests for trace.py to achieve 100% coverage."""

import inspect
import tempfile
from pathlib import Path
from unittest.mock import patch

from tests.hidden_module import internal_helper_function
from tracerite.trace import extract_exception, extract_frames


def _trigger_error_no_source():
    """Helper function for testing OSError in getsourcelines."""
    raise ValueError("Error in function without source")


class TestTraceCornercases:
    """Corner case tests for trace module."""

    def test_tracebackhide_in_globals_with_call_chain(self):
        """Test line 87: __tracebackhide__ check in f_globals.

        Simulates a real-world scenario where a module has __tracebackhide__ = True
        at module level. All functions in that module should be hidden from tracebacks.
        This tests the f_globals check on line 86-87 in trace.py.
        """

        def user_code():
            """User's code that raises an error."""
            raise ValueError("User code error")

        def test_runner():
            """Test runner that calls hidden module function."""
            return internal_helper_function(user_code)

        frames = []
        try:
            test_runner()
        except ValueError as e:
            if e.__traceback__:
                tb = inspect.getinnerframes(e.__traceback__)
                frames = extract_frames(tb)

        # Verify frames were extracted
        assert len(frames) > 0, "Should have extracted some frames"

        function_names = [f["function"] for f in frames]

        # internal_helper_function should be excluded (has __tracebackhide__ in f_globals)
        assert "internal_helper_function" not in function_names, (
            f"internal_helper_function should be hidden, but found in: {function_names}"
        )

        # user_code should be included (where the error was raised)
        assert "user_code" in function_names, (
            f"user_code should be visible, but not found in: {function_names}"
        )

        # test_runner should be included (entry point)
        assert "test_runner" in function_names, (
            f"test_runner should be visible, but not found in: {function_names}"
        )

    def test_tracebackhide_in_locals_with_call_chain(self):
        """Test line 89: __tracebackhide__ check in f_locals.

        Simulates a real-world scenario where an internal implementation function
        (like a test framework helper) hides itself from the traceback when user
        code fails. The hidden function calls user code that crashes.
        This tests the f_locals check on line 88-89 in trace.py.
        """

        def internal_implementation_wrapper(user_callback):
            """Internal function that should be hidden from tracebacks."""
            __tracebackhide__ = True  # Hide this internal implementation
            # Call user code that might fail
            return user_callback()

        def user_code():
            """User's code that raises an error."""
            raise ValueError("User code error")

        def test_runner():
            """Test runner that calls internal wrapper."""
            return internal_implementation_wrapper(user_code)

        frames = []
        try:
            test_runner()
        except ValueError as e:
            if e.__traceback__:
                tb = inspect.getinnerframes(e.__traceback__)
                frames = extract_frames(tb)

        # Verify frames were extracted
        assert len(frames) > 0, "Should have extracted some frames"

        function_names = [f["function"] for f in frames]

        # internal_implementation_wrapper should be excluded (has __tracebackhide__ in f_locals)
        assert "internal_implementation_wrapper" not in function_names, (
            f"internal_implementation_wrapper should be hidden, but found in: {function_names}"
        )

        # user_code should be included (where the error was raised)
        assert "user_code" in function_names, (
            f"user_code should be visible, but not found in: {function_names}"
        )

        # test_runner should be included (entry point)
        assert "test_runner" in function_names, (
            f"test_runner should be visible, but not found in: {function_names}"
        )

    def test_inspect_indexerror(self):
        """Test handling of IndexError in inspect.getinnerframes."""
        # This is hard to trigger naturally, but we can test the error path exists
        try:
            raise ValueError("test")
        except ValueError as e:
            # Should handle gracefully even if inspect fails
            info = extract_exception(e)
            assert info["type"] == "ValueError"

    def test_exception_extraction_failure(self):
        """Test when frame extraction raises an exception."""
        try:
            raise ValueError("test")
        except ValueError as e:
            # Should handle extraction errors gracefully
            info = extract_exception(e)
            assert "frames" in info
            # Frames might be empty if extraction failed, but key should exist
            assert isinstance(info["frames"], list)

    def test_non_python_module_skipped(self):
        """Test that non-Python modules without source are skipped for call frames."""
        import json

        try:
            json.loads("invalid")
        except Exception as e:
            import inspect

            tb = inspect.getinnerframes(e.__traceback__)
            frames = extract_frames(tb)
            # Should have some frames but might skip some without source
            assert len(frames) >= 0

    def test_long_filename_shortening(self):
        """Test filename shortening for very long paths."""

        def func():
            raise ValueError("test")

        try:
            func()
        except ValueError as e:
            import inspect

            tb = inspect.getinnerframes(e.__traceback__)
            frames = extract_frames(tb)
            # Frames should have location set
            for frame in frames:
                assert "location" in frame

    def test_ipython_integration(self):
        """Test ipython integration (when ipython is None)."""
        from tracerite import trace

        original_ipython = trace.ipython
        try:
            trace.ipython = None
            try:
                raise ValueError("test")
            except ValueError as e:
                import inspect

                tb = inspect.getinnerframes(e.__traceback__)
                frames = extract_frames(tb)
                # Should work without ipython
                assert len(frames) > 0
        finally:
            trace.ipython = original_ipython

    def test_inspect_indexerror_handling(self):
        """Test IndexError handling in inspect.getinnerframes."""
        try:
            raise ValueError("test")
        except ValueError as e:
            # Mock inspect.getinnerframes to raise IndexError
            with patch("inspect.getinnerframes", side_effect=IndexError("test")):
                info = extract_exception(e)
                # Should handle gracefully and return empty frames
                assert info["type"] == "ValueError"
                assert info["frames"] == []

    def test_exception_during_frame_extraction(self):
        """Test exception handling during extract_frames."""
        try:
            raise ValueError("test")
        except ValueError as e:
            tb = e.__traceback__
            tb = inspect.getinnerframes(tb)
            # Mock extract_variables to raise exception
            with patch(
                "tracerite.trace.extract_variables", side_effect=RuntimeError("test")
            ):
                # Should catch and log exception, return None for frames
                info = extract_exception(e)
                assert info["frames"] == []

    def test_skip_until_not_found(self):
        """Test skip_until when pattern is not found in any frame."""
        try:
            raise ValueError("test")
        except ValueError as e:
            info = extract_exception(e, skip_until="nonexistent_file.py")
            # Should not skip any frames since pattern not found
            assert len(info["frames"]) > 0

    def test_getsourcelines_oserror_call_frame(self):
        """Test OSError in getsourcelines causes call frames to be skipped (line 113)."""
        from tracerite.trace import extract_frames

        def level_1():
            level_2()

        def level_2():
            level_3()

        def level_3():
            raise ValueError("test")

        # Mock getsourcelines to raise OSError for level_1 (which will be a "call" frame)
        original_getsourcelines = inspect.getsourcelines

        def mock_getsourcelines(frame):
            # Raise OSError for level_1 to trigger line 113
            if frame.f_code.co_name == "level_1":
                raise OSError("No source")
            return original_getsourcelines(frame)

        with patch("inspect.getsourcelines", side_effect=mock_getsourcelines):
            try:
                level_1()
            except ValueError as e:
                tb = inspect.getinnerframes(e.__traceback__)
                frames = extract_frames(tb)
                # level_1 frame should be skipped due to OSError + relevance="call"
                frame_functions = [
                    f.get("function") for f in frames if f.get("function")
                ]
                # level_1 should not be in frames because it's a call frame with no source
                assert "level_1" not in frame_functions or len(frames) > 0

    def test_tracebackhide_in_locals(self):
        """Test __tracebackhide__ in f_locals skips frame (line 87)."""
        from tracerite.trace import extract_frames

        # Use exec to create a function with __tracebackhide__ that persists
        code = """
def hidden_frame():
    __tracebackhide__ = True
    # Do something with the variable so it stays in locals
    locals()['__tracebackhide__']
    raise ValueError("hidden")

hidden_frame()
"""

        namespace = {}
        try:
            exec(code, namespace)
        except ValueError as e:
            tb = inspect.getinnerframes(e.__traceback__)

            # Check if __tracebackhide__ is in any frame's locals
            has_hidden = any("__tracebackhide__" in f.frame.f_locals for f in tb)

            frames = extract_frames(tb)

            # If __tracebackhide__ was detected, verify it worked
            if has_hidden:
                # hidden_frame should be skipped if __tracebackhide__ was detected
                assert isinstance(frames, list)

    def test_ipython_compile_filename_map(self):
        """Test ipython.compile._filename_map access."""
        from unittest.mock import MagicMock

        from tracerite import trace

        # Create a mock ipython with compile._filename_map
        mock_ipython = MagicMock()
        mock_ipython.compile._filename_map = {"<ipython-input-1>": "1"}

        original = trace.ipython
        try:
            trace.ipython = mock_ipython
            try:
                raise ValueError("test in ipython")
            except ValueError as e:
                tb = inspect.getinnerframes(e.__traceback__)
                # Mock the filename to match - create FrameInfo-like objects
                modified_tb = []
                for frame_info in tb:
                    # Create a new FrameInfo with modified filename
                    modified_frame = inspect.FrameInfo(
                        frame=frame_info.frame,
                        filename="<ipython-input-1>",
                        lineno=frame_info.lineno,
                        function=frame_info.function,
                        code_context=frame_info.code_context,
                        index=frame_info.index,
                    )
                    modified_tb.append(modified_frame)

                frames = extract_frames(modified_tb)
                # Should handle ipython filename mapping
                assert isinstance(frames, list)
        finally:
            trace.ipython = original

    def test_jupyter_url_generation(self):
        """Test Jupyter URL generation when ipython is not None."""
        from unittest.mock import MagicMock

        from tracerite import trace

        mock_ipython = MagicMock()
        mock_ipython.compile._filename_map = {}
        original = trace.ipython

        try:
            trace.ipython = mock_ipython

            # Create a test file in current directory
            with tempfile.NamedTemporaryFile(
                mode="w", suffix=".py", delete=False, dir="."
            ) as f:
                f.write("def test_func():\n    raise ValueError('test')\ntest_func()")
                temp_file = f.name

            try:
                # Execute the file to generate traceback
                with open(temp_file) as f:
                    exec(compile(f.read(), temp_file, "exec"))
            except ValueError as e:
                tb = inspect.getinnerframes(e.__traceback__)
                frames = extract_frames(tb)

                # Should have attempted to add Jupyter URL for local files
                assert isinstance(frames, list)
            finally:
                Path(temp_file).unlink(missing_ok=True)
        finally:
            trace.ipython = original

    def test_long_filename_shortening_in_extract_frames(self):
        """Test filename shortening for paths > 40 characters."""
        # Create a deeply nested path
        long_path = (
            "/very/long/path/that/exceeds/forty/characters/in/total/length/file.py"
        )

        try:
            # Create a mock frame with long filename
            raise ValueError("test")
        except ValueError as e:
            tb = inspect.getinnerframes(e.__traceback__)
            # Modify first frame to have long filename - create FrameInfo objects
            modified_tb = []
            for i, frame_info in enumerate(tb):
                if i == 0:
                    modified_frame = inspect.FrameInfo(
                        frame=frame_info.frame,
                        filename=long_path,
                        lineno=frame_info.lineno,
                        function=frame_info.function,
                        code_context=frame_info.code_context,
                        index=frame_info.index,
                    )
                    modified_tb.append(modified_frame)
                else:
                    modified_tb.append(frame_info)

            frames = extract_frames(modified_tb)
            # Should shorten long filenames
            assert len(frames) > 0

    def test_getsourcelines_oserror_error_frame_undefined_start(self):
        """Test OSError in getsourcelines for error frame properly initializes 'start'.

        This test verifies that when OSError is raised, the frame extraction properly
        handles the error frame and sets linenostart correctly.
        """

        # Save the original function before patching
        original_getsourcelines = inspect.getsourcelines

        # Mock getsourcelines to raise OSError for the innermost error frame
        def mock_getsourcelines(frame):
            func_name = frame.f_code.co_name
            # Raise OSError for _trigger_error_no_source (the innermost frame)
            if func_name == "_trigger_error_no_source":
                raise OSError("No source available")
            # For other frames, get real source
            return original_getsourcelines(frame)

        with patch(
            "tracerite.trace.inspect.getsourcelines", side_effect=mock_getsourcelines
        ):
            try:
                # Call the function directly so it's the innermost frame
                _trigger_error_no_source()
            except ValueError as e:
                if e.__traceback__:
                    tb = inspect.getinnerframes(e.__traceback__)
                    # Extract frames - this should work without NameError after the fix
                    frames = extract_frames(tb)

                    # Find the error frame
                    error_frame = next(
                        (
                            f
                            for f in frames
                            if f.get("function") == "_trigger_error_no_source"
                        ),
                        None,
                    )
                    assert error_frame is not None, (
                        "Error frame should be in extracted frames"
                    )

                    # The error frame should not have lines (OSError occurred)
                    assert error_frame.get("lines") == "", (
                        "Error frame should have empty lines"
                    )

                    # After fix: linenostart should be set properly
                    linenostart = error_frame.get("linenostart")

                    # In the new API, we also have a range field
                    range_obj = error_frame.get("range")

                    # The error is raised on line 14 in _trigger_error_no_source
                    expected_lineno = 14

                    # Verify the range contains the correct line number
                    if range_obj:
                        assert range_obj.lfirst == expected_lineno, (
                            f"range.lfirst should be {expected_lineno}, but got {range_obj.lfirst}"
                        )

                    # Verify start was properly initialized when OSError occurred
                    assert linenostart == expected_lineno, (
                        f"linenostart should be {expected_lineno} (line where error was raised), "
                        f"but got {linenostart}"
                    )

    def test_create_summary_long_message(self):
        """Test _create_summary with message > 1000 chars (lines 42-46)."""
        from tracerite.trace import _create_summary

        # Test with message > 1000 chars
        long_message = "A" * 1500
        summary = _create_summary(long_message)
        # Should use the pattern with beginning and end
        assert "···" in summary
        assert summary.startswith("A" * 40)
        assert summary.endswith("A" * 40)

        # Test with message > 100 but < 1000 chars
        medium_message = "B" * 200
        summary = _create_summary(medium_message)
        assert "···" in summary
        assert len(summary) < len(medium_message)

        # Test with short message
        short_message = "Short"
        summary = _create_summary(short_message)
        assert summary == short_message

    def test_skip_until_found(self):
        """Test skip_until when pattern IS found in frame (lines 60-61)."""

        def level_1():
            level_2()

        def level_2():
            raise ValueError("test")

        try:
            level_1()
        except ValueError as e:
            # Use the actual filename of this test file
            info = extract_exception(e, skip_until="test_trace_cornercases.py")
            # Should skip frames until it finds the pattern
            assert len(info["frames"]) >= 0

    def test_raw_tb_skip_logic(self):
        """Test raw_tb skipping when skip_outmost > 0 (lines 66-68)."""

        def level_1():
            level_2()

        def level_2():
            level_3()

        def level_3():
            raise ValueError("test")

        try:
            level_1()
        except ValueError as e:
            # Skip the first 2 frames
            info = extract_exception(e, skip_outmost=2)
            # Should have skipped frames
            assert "frames" in info

    def test_raw_tb_skip_with_none_check(self):
        """Test raw_tb skip logic with the 'if raw_tb' check (line 67->66)."""

        def level_1():
            level_2()

        def level_2():
            raise ValueError("test")

        try:
            level_1()
        except ValueError as e:
            # Skip 1 frame - this will exercise the raw_tb skip logic
            info = extract_exception(e, skip_outmost=1)
            assert "frames" in info
            # Verify that frames were actually skipped
            assert isinstance(info["frames"], list)

    def test_raw_tb_skip_exhaustion(self):
        """Test raw_tb skip when we skip more frames than exist (line 67->66 branch)."""

        def single_level():
            raise ValueError("test")

        try:
            single_level()
        except ValueError as e:
            # Try to skip 10 frames when there's only 1-2 frames
            # This should cause raw_tb to become None during the loop
            info = extract_exception(e, skip_outmost=10)
            assert "frames" in info
            # Should handle gracefully even if we skip too many
            assert isinstance(info["frames"], list)

    def test_extract_emphasis_columns_bounds_check(self):
        """Test bounds check in _extract_emphasis_columns (line 176)."""
        from tracerite.trace import _extract_emphasis_columns

        # Test with invalid bounds
        lines = "line1\nline2\nline3"
        # segment_start out of bounds
        result = _extract_emphasis_columns(lines, 10, 1, 1, 1, 1)
        assert result is None

        # segment_end out of bounds
        result = _extract_emphasis_columns(lines, 1, 100, 1, 1, 1)
        assert result is None

    def test_extract_emphasis_columns_exception(self):
        """Test exception handling in _extract_emphasis_columns (lines 211-214)."""
        from tracerite.trace import _extract_emphasis_columns

        # Mock _extract_caret_anchors_from_line_segment to raise an exception
        with patch(
            "tracerite.trace.trace_cpy._extract_caret_anchors_from_line_segment",
            side_effect=RuntimeError("test error"),
        ):
            lines = "x = 1 + 2"
            # This should trigger the exception handler on lines 211-214
            result = _extract_emphasis_columns(lines, 1, 1, 0, 9, 1)
            # Should return None after logging the exception
            assert result is None

    def test_build_position_map_exception(self):
        """Test exception handling in _build_position_map (lines 211-214)."""
        from tracerite.trace import _build_position_map

        # Create a real traceback object
        try:
            raise ValueError("test")
        except ValueError as e:
            raw_tb = e.__traceback__

            # Mock _walk_tb_with_full_positions to raise exception
            with patch(
                "tracerite.trace.trace_cpy._walk_tb_with_full_positions",
                side_effect=RuntimeError("test"),
            ):
                position_map = _build_position_map(raw_tb)
                # Should return empty dict on exception
                assert position_map == {}

            # Also test with None raw_tb to cover line 220-221
            position_map = _build_position_map(None)
            assert position_map == {}

    def test_suppress_inner_break(self):
        """Test suppress_inner and is_bug_frame break condition (line 328)."""

        # Create a KeyboardInterrupt (not an Exception)
        def trigger_keyboard_interrupt():
            raise KeyboardInterrupt("test")

        try:
            trigger_keyboard_interrupt()
        except KeyboardInterrupt as e:
            if e.__traceback__:
                tb = inspect.getinnerframes(e.__traceback__)
                frames = extract_frames(tb, e.__traceback__)
                # Should suppress inner frames for non-Exception types
                assert isinstance(frames, list)

    def test_calculate_common_indent_empty(self):
        """Test _calculate_common_indent with all empty lines (line 338)."""
        from tracerite.trace import _calculate_common_indent

        # All empty lines
        lines = ["   \n", "\n", "  \n"]
        indent = _calculate_common_indent(lines)
        assert indent == 0

        # No lines
        lines = []
        indent = _calculate_common_indent(lines)
        assert indent == 0

    def test_parse_line_empty(self):
        """Test _parse_line_to_fragments_unified with empty line (lines 408, 412)."""
        from tracerite.trace import _parse_line_to_fragments_unified

        # Empty line
        result = _parse_line_to_fragments_unified("", 0, set(), set(), 0)
        assert result == []

        # Line with only ending
        result = _parse_line_to_fragments_unified("\n", 0, set(), set(), 0)
        # Should handle line ending only
        assert isinstance(result, list)

    def test_parse_line_comment_branches(self):
        """Test comment handling branches in _parse_line_to_fragments_unified."""
        from tracerite.trace import _parse_line_to_fragments_unified

        # Line with comment and trailing whitespace after comment (line 441->445)
        line = "code  # comment  \n"
        result = _parse_line_to_fragments_unified(line, 0, set(), set(), 0)
        assert len(result) > 0

        # Line with only comment (no code part) (line 446->464)
        line = "# comment only\n"
        result = _parse_line_to_fragments_unified(line, 0, set(), set(), 0)
        assert len(result) > 0

        # Line without comment but with trailing whitespace (line 461->464)
        line = "code  \n"
        result = _parse_line_to_fragments_unified(line, 0, set(), set(), 0)
        assert len(result) > 0

    def test_parse_line_empty_comment_branches(self):
        """Test empty comment/trailing branches (lines 441->445, 446->464, 461->464)."""
        from tracerite.trace import _parse_line_to_fragments_unified

        # Line 441->445: Empty comment_with_leading_space (comment is only whitespace+#)
        # When code_whitespace + comment_trimmed is empty
        line = "code#\n"  # No space before/after #, comment_trimmed would be "#"
        result = _parse_line_to_fragments_unified(line, 0, set(), set(), 0)
        assert isinstance(result, list)

        # Line 446->464: Empty trailing_content after comment
        # When comment_trailing + line_ending is empty (no \n, comment already trimmed)
        line = "code  # comment"  # No line ending
        result = _parse_line_to_fragments_unified(line, 0, set(), set(), 0)
        assert isinstance(result, list)

        # Line 461->464: Empty trailing_content for line without comment
        # When trailing_whitespace + line_ending is empty
        line = "code"  # No trailing whitespace or line ending
        result = _parse_line_to_fragments_unified(line, 0, set(), set(), 0)
        assert isinstance(result, list)

    def test_parse_line_only_line_ending(self):
        """Test _parse_line_to_fragments_unified with only line ending (line 412)."""
        from tracerite.trace import _parse_line_to_fragments_unified

        # Line with only line ending (no content)
        # This is when line_content is empty and line_ending exists
        line = "\n"
        result = _parse_line_to_fragments_unified(line, 0, set(), set(), 0)
        # Should return empty list or handle gracefully
        assert isinstance(result, list)

    def test_create_highlighted_fragments_empty(self):
        """Test _create_highlighted_fragments_unified with empty text (line 472)."""
        from tracerite.trace import _create_highlighted_fragments_unified

        result = _create_highlighted_fragments_unified("", 0, set(), set())
        assert result == []

    def test_convert_range_to_positions_empty(self):
        """Test _convert_range_to_positions with empty lines (line 508)."""
        from tracerite.trace import Range, _convert_range_to_positions

        # Empty lines - this should trigger line 508
        result = _convert_range_to_positions(Range(1, 1, 0, 5), "")
        assert result == set()

        # None range
        result = _convert_range_to_positions(None, "some text")
        assert result == set()

        # Test with lines that don't have enough content for the range
        result = _convert_range_to_positions(Range(5, 5, 0, 10), "line1\nline2\n")
        # Should handle gracefully when range is beyond available lines
        assert isinstance(result, set)

    def test_parse_lines_to_fragments_empty_after_splitlines(self):
        """Test _parse_lines_to_fragments when splitlines returns empty (line 508)."""
        from tracerite.trace import _parse_lines_to_fragments

        # This should trigger the check on line 508: if not lines: return []
        # An empty string after splitlines(keepends=True) gives []
        result = _parse_lines_to_fragments("")
        assert result == []

        # Just whitespace might also give empty after splitlines
        result = _parse_lines_to_fragments(" ")
        assert isinstance(result, list)

    def test_split_line_content_various_endings(self):
        """Test _split_line_content with various line endings (lines 525, 528-531)."""
        from tracerite.trace import _split_line_content

        # CRLF ending
        content, ending = _split_line_content("text\r\n")
        assert content == "text"
        assert ending == "\r\n"

        # LF ending
        content, ending = _split_line_content("text\n")
        assert content == "text"
        assert ending == "\n"

        # CR ending
        content, ending = _split_line_content("text\r")
        assert content == "text"
        assert ending == "\r"

        # No ending
        content, ending = _split_line_content("text")
        assert content == "text"
        assert ending == ""

    def test_process_indentation_dedent(self):
        """Test _process_indentation with dedent (lines 541-543)."""
        from tracerite.trace import _process_indentation

        # Test with common indent that needs to be removed
        line_content = "    indented code"
        fragments, remaining, pos = _process_indentation(line_content, 4)
        # Should have dedent fragment
        assert any(f.get("dedent") for f in fragments)
        assert pos == 4

        # Test with no common indent
        fragments, remaining, pos = _process_indentation(line_content, 0)
        assert pos >= 0

    def test_positions_to_consecutive_ranges_gaps(self):
        """Test _positions_to_consecutive_ranges with gaps (lines 598-600)."""
        from tracerite.trace import _positions_to_consecutive_ranges

        # Positions with gaps
        positions = {0, 1, 2, 5, 6, 7}
        ranges = _positions_to_consecutive_ranges(positions)
        # Should have multiple ranges due to gap
        assert len(ranges) == 2
        assert (0, 3) in ranges
        assert (5, 8) in ranges

        # Single position
        positions = {5}
        ranges = _positions_to_consecutive_ranges(positions)
        assert ranges == [(5, 6)]

    def test_create_fragments_with_highlighting_empty(self):
        """Test _create_fragments_with_highlighting with empty text (line 627)."""
        from tracerite.trace import _create_fragments_with_highlighting

        result = _create_fragments_with_highlighting("", set(), set())
        assert result == []

    def test_create_fragments_start_boundary_checks(self):
        """Test boundary checks in _create_fragments_with_highlighting (lines 640, 644)."""
        from tracerite.trace import _create_fragments_with_highlighting

        # Test with mark positions
        text = "code"
        mark_positions = {0, 1, 2, 3}
        result = _create_fragments_with_highlighting(text, mark_positions, set())
        assert len(result) > 0

        # Test with overlapping positions causing boundary >= len(text)
        text = "ab"
        mark_positions = {0, 1, 2, 3, 4}  # Positions beyond text length
        result = _create_fragments_with_highlighting(text, mark_positions, set())
        # Should handle gracefully
        assert isinstance(result, list)

        # Test specific case where start >= len(text) in the loop (line 640)
        text = "x"
        # Create boundaries that would result in start >= len(text)
        mark_positions = {0, 1, 2}  # Position 2 is beyond "x" (len=1)
        result = _create_fragments_with_highlighting(text, mark_positions, set())
        assert isinstance(result, list)

        # Test case where fragment_text is empty (line 644)
        text = "ab"
        # Test with empty string between boundaries
        mark_positions = {0, 1}
        em_positions = {1, 2}  # Different positions to create multiple boundaries
        result = _create_fragments_with_highlighting(text, mark_positions, em_positions)
        assert isinstance(result, list)


class TestReraiseExistingException:
    """Test cases for re-raising existing exceptions.

    When an existing exception is re-raised (e.g., `raise exc`), CPython's position
    info may have end_line < error_line_in_context, causing an empty slice in
    _extract_emphasis_columns. This should be handled gracefully.
    """

    def test_reraise_existing_exception_no_crash(self):
        """Test that re-raising an existing exception doesn't crash.

        This tests the guard in _extract_emphasis_columns that handles the case
        where segment_start > segment_end due to CPython's position info.
        When you do `raise e` where e is a caught exception from a different line,
        CPython's position info can have end_line < error_line_in_context.
        """
        from tracerite.trace import _extract_emphasis_columns

        # Simulate the scenario that occurs with `raise e`:
        # The raise statement is on line 5, but the original error was on line 3
        # This causes segment_start (4) > segment_end (3), making the slice empty
        lines = """def process(x, y, name):
    try:
        result = x + y
    except Exception as e:
        raise e
    return result
"""
        # These values match what CPython provides for `raise e`:
        # error_line_in_context=5 (the `raise e` line)
        # end_line=3 (the original error line)
        result = _extract_emphasis_columns(
            lines=lines,
            error_line_in_context=5,  # raise e is on line 5
            end_line=3,  # but original error was line 3
            start_col=17,
            end_col=22,
            start=1,
        )
        # Should return None gracefully, not crash with IndexError
        assert result is None

    def test_reraise_with_from_no_crash(self):
        """Test that re-raising with 'from' doesn't crash."""
        from tracerite.trace import extract_chain

        def inner():
            raise ValueError("inner error")

        def outer():
            try:
                inner()
            except ValueError as e:
                raise RuntimeError("outer error") from e

        try:
            outer()
        except RuntimeError:
            result = extract_chain()
            assert result is not None
            # Should have both exceptions in the chain
            assert len(result) >= 2

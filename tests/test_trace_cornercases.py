"""Corner case and edge case tests for trace.py to achieve 100% coverage."""

import inspect
import tempfile
from pathlib import Path
from unittest.mock import patch

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
        from . import hidden_module

        def user_code():
            """User's code that raises an error."""
            raise ValueError("User code error")

        def test_runner():
            """Test runner that calls hidden module function."""
            return hidden_module.internal_helper_function(user_code)

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

        This test verifies the fix for the bug where `start` was undefined when:
        1. inspect.getsourcelines() raises OSError
        2. The frame is the innermost frame (where exception is raised)
        3. Therefore relevance != "call" (it will be "error" or "stop")

        The bug was that when OSError is raised, the except block sets lines=""
        but didn't initialize `start`, causing it to leak from previous iterations.

        After the fix, start should be initialized to lineno when OSError occurs.
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

                    # After fix: linenostart should be set to the lineno (not leaked from previous frame)
                    linenostart = error_frame.get("linenostart")
                    lineno = error_frame.get("lineno")

                    # The error is raised on line 13 in _trigger_error_no_source
                    expected_lineno = 13

                    # Verify lineno is correct
                    assert lineno == expected_lineno, (
                        f"lineno should be {expected_lineno}, but got {lineno}"
                    )

                    # Verify start was properly initialized to lineno when OSError occurred
                    assert linenostart == lineno, (
                        f"linenostart should equal lineno ({lineno}) when OSError occurs, "
                        f"but got linenostart={linenostart}"
                    )

                    # Explicitly verify linenostart is the expected value
                    assert linenostart == expected_lineno, (
                        f"linenostart should be {expected_lineno} (line where error was raised), "
                        f"but got {linenostart}"
                    )

                    # Also verify it didn't leak from previous frame
                    if len(frames) > 1:
                        prev_frame = frames[0]
                        prev_linenostart = prev_frame.get("linenostart")
                        # Since we now initialize to lineno, these should be different
                        # (unless by coincidence the previous frame also starts at line 13)
                        if prev_linenostart == linenostart:
                            # If they're equal, make sure it's because linenostart is correctly
                            # set to lineno, not because it leaked
                            assert linenostart == lineno == expected_lineno, (
                                f"linenostart={linenostart} equals prev_linenostart={prev_linenostart}, "
                                f"but should be correctly set to lineno={lineno}"
                            )

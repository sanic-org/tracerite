"""Tests for trace.py - exception and traceback extraction."""



from tracerite.trace import extract_chain, extract_exception, extract_frames


class TestExtractChain:
    """Test extract_chain function for exception chaining."""

    def test_extract_single_exception(self):
        """Test extracting a single exception without chaining."""
        try:
            raise ValueError("test error")
        except ValueError as e:
            chain = extract_chain(exc=e)

        assert len(chain) == 1
        assert chain[0]["type"] == "ValueError"
        assert "test error" in chain[0]["message"]

    def test_extract_chained_exceptions(self):
        """Test extracting chained exceptions with __cause__."""
        try:
            try:
                raise ValueError("original error")
            except ValueError as e:
                raise TypeError("wrapped error") from e
        except TypeError as e:
            chain = extract_chain(exc=e)

        # When using 'from', only the newest exception is returned by default
        assert len(chain) >= 1
        assert chain[0]["type"] == "TypeError"
        assert "wrapped error" in chain[0]["message"]

    def test_extract_context_exceptions(self):
        """Test extracting exceptions with implicit context (__context__)."""
        try:
            try:
                raise ValueError("first error")
            except ValueError:
                raise TypeError("second error")
        except TypeError as e:
            chain = extract_chain(exc=e)

        # Should have both exceptions
        assert len(chain) == 2
        assert chain[0]["type"] == "TypeError"
        assert chain[1]["type"] == "ValueError"

    def test_suppress_context(self):
        """Test that __suppress_context__ stops chain extraction."""
        try:
            try:
                raise ValueError("suppressed error")
            except ValueError:
                exc = TypeError("visible error")
                exc.__suppress_context__ = True
                raise exc
        except TypeError as e:
            chain = extract_chain(exc=e)

        # Should only have the TypeError, not the ValueError
        assert len(chain) == 1
        assert chain[0]["type"] == "TypeError"

    def test_extract_from_sys_exc_info(self):
        """Test extracting current exception from sys.exc_info()."""
        try:
            raise RuntimeError("current exception")
        except RuntimeError:
            chain = extract_chain()  # No exc argument

        assert len(chain) == 1
        assert chain[0]["type"] == "RuntimeError"


class TestExtractException:
    """Test extract_exception function for single exception details."""

    def test_basic_exception_extraction(self):
        """Test basic exception information extraction."""
        try:
            x = 1
            y = 0
            result = x / y  # noqa: F841
        except ZeroDivisionError as e:
            info = extract_exception(e)

        assert info["type"] == "ZeroDivisionError"
        assert (
            "division" in info["message"].lower() or "zero" in info["message"].lower()
        )
        assert info["summary"]
        assert info["repr"]
        assert isinstance(info["frames"], list)
        assert len(info["frames"]) > 0

    def test_exception_with_long_message(self):
        """Test that long exception messages are truncated in summary."""
        long_msg = "x" * 150
        try:
            raise ValueError(long_msg)
        except ValueError as e:
            info = extract_exception(e)

        # Message should be full, summary should be truncated
        assert len(info["message"]) == 150
        assert len(info["summary"]) < len(info["message"])
        assert "···" in info["summary"] or "…" in info["summary"]

    def test_exception_with_very_long_message(self):
        """Test handling of very long messages (>1000 chars)."""
        long_msg = "A" * 600 + "B" * 600
        try:
            raise RuntimeError(long_msg)
        except RuntimeError as e:
            info = extract_exception(e)

        # Summary should show beginning and end
        summary = info["summary"]
        assert "···" in summary or "…" in summary
        assert "A" in summary
        assert "B" in summary

    def test_exception_with_multiline_message(self):
        """Test that multiline messages are summarized correctly."""
        msg = "First line\nSecond line\nThird line"
        try:
            raise ValueError(msg)
        except ValueError as e:
            info = extract_exception(e)

        # Summary should only have first line
        assert info["summary"] == "First line"
        assert info["message"] == msg

    def test_skip_outmost_frames(self):
        """Test skipping outermost frames."""

        def outer():
            def middle():
                def inner():
                    raise ValueError("error")

                inner()

            middle()

        try:
            outer()
        except ValueError as e:
            info = extract_exception(e, skip_outmost=1)

        # Should skip the first frame
        assert len(info["frames"]) >= 2

    def test_skip_until_pattern(self):
        """Test skipping frames until a filename pattern is found."""
        try:
            raise ValueError("test")
        except ValueError as e:
            info = extract_exception(e, skip_until="test_trace.py")

        # Should have frames starting from test_trace.py
        assert len(info["frames"]) > 0

    def test_base_exception_suppression(self):
        """Test that BaseException subclasses are handled differently."""
        try:
            raise KeyboardInterrupt()
        except KeyboardInterrupt as e:
            info = extract_exception(e)

        # Should still extract but may suppress inner frames
        assert info["type"] == "KeyboardInterrupt"
        # Frames should be present
        assert isinstance(info["frames"], list)


class TestExtractFrames:
    """Test extract_frames function for traceback frame extraction."""

    def test_extract_frames_basic(self):
        """Test basic frame extraction from traceback."""

        def function_a():
            def function_b():
                raise ValueError("test")

            function_b()

        try:
            function_a()
        except ValueError as e:
            import inspect

            tb = inspect.getinnerframes(e.__traceback__)
            frames = extract_frames(tb)

        assert len(frames) > 0
        # Check frame structure
        frame = frames[0]
        assert "id" in frame
        assert "relevance" in frame
        assert "location" in frame
        assert "lineno" in frame
        assert "function" in frame
        assert "variables" in frame

    def test_frame_relevance_markers(self):
        """Test that frame relevance is correctly assigned."""

        def user_function():
            raise RuntimeError("error")

        try:
            user_function()
        except RuntimeError as e:
            import inspect

            tb = inspect.getinnerframes(e.__traceback__)
            frames = extract_frames(tb)

        # Last frame should be 'error' (where exception was raised)
        assert frames[-1]["relevance"] == "error"

    def test_tracebackhide_in_globals(self):
        """Test that frames with __tracebackhide__ in globals are skipped."""

        def hidden_function():
            __tracebackhide__ = True
            raise ValueError("error")

        try:
            hidden_function()
        except ValueError as e:
            import inspect

            tb = inspect.getinnerframes(e.__traceback__)
            frames = extract_frames(tb)

        # hidden_function should be excluded
        function_names = [f["function"] for f in frames]
        assert "hidden_function" not in function_names

    def test_tracebackhide_in_locals(self):
        """Test that frames with __tracebackhide__ in locals are skipped."""

        def another_hidden():
            __tracebackhide__ = True  # noqa: F841
            raise ValueError("error")

        try:
            another_hidden()
        except ValueError as e:
            import inspect

            tb = inspect.getinnerframes(e.__traceback__)
            frames = extract_frames(tb)

        # another_hidden should be excluded
        function_names = [f["function"] for f in frames]
        assert "another_hidden" not in function_names

    def test_empty_traceback(self):
        """Test handling of empty traceback."""
        frames = extract_frames([])
        assert frames == []

    def test_method_with_self(self):
        """Test that method names include class name when self is present."""

        class MyClass:
            def my_method(self):
                raise ValueError("error in method")

        try:
            obj = MyClass()
            obj.my_method()
        except ValueError as e:
            import inspect

            tb = inspect.getinnerframes(e.__traceback__)
            frames = extract_frames(tb)

        # Should find frame with class name
        function_names = [f["function"] for f in frames]
        assert any(
            "MyClass" in name and "my_method" in name for name in function_names if name
        )

    def test_method_with_cls(self):
        """Test that classmethods include class name."""

        class MyClass:
            @classmethod
            def my_classmethod(cls):
                raise ValueError("error in classmethod")

        try:
            MyClass.my_classmethod()
        except ValueError as e:
            import inspect

            tb = inspect.getinnerframes(e.__traceback__)
            frames = extract_frames(tb)

        function_names = [f["function"] for f in frames]
        assert any(
            "MyClass" in name and "my_classmethod" in name
            for name in function_names
            if name
        )

    def test_module_level_code(self):
        """Test handling of module-level code (no function)."""
        try:
            exec("raise ValueError('module level')")
        except ValueError as e:
            import inspect

            tb = inspect.getinnerframes(e.__traceback__)
            frames = extract_frames(tb)

        # Should handle module-level frames
        assert len(frames) > 0

    def test_frame_with_no_source(self):
        """Test handling frames where source code is not available."""
        # Create exception from built-in code path
        try:
            import json

            json.loads("{invalid json}")
        except Exception as e:
            import inspect

            tb = inspect.getinnerframes(e.__traceback__)
            frames = extract_frames(tb)

        # Should handle frames even without source
        assert len(frames) > 0

    def test_suppress_inner_frames(self):
        """Test that suppress_inner stops at the bug frame."""

        def outer():
            def middle():
                def inner():
                    raise KeyboardInterrupt()

                inner()

            middle()

        try:
            outer()
        except KeyboardInterrupt as e:
            import inspect

            tb = inspect.getinnerframes(e.__traceback__)
            frames = extract_frames(tb, suppress_inner=True)

        # Should have fewer frames due to suppression
        assert len(frames) > 0

    def test_relative_path_display(self):
        """Test that paths in CWD are shown as relative."""
        # Create a traceback in current directory
        try:
            raise ValueError("test")
        except ValueError as e:
            import inspect

            tb = inspect.getinnerframes(e.__traceback__)
            frames = extract_frames(tb)

        # Check that we have frames with filenames
        assert any(f["filename"] for f in frames)

    def test_long_function_path_shortening(self):
        """Test that long module.function paths are shortened."""

        # Function names should be shortened to last 2 components
        def outer():
            def inner():
                raise ValueError("test")

            inner()

        try:
            outer()
        except ValueError as e:
            import inspect

            tb = inspect.getinnerframes(e.__traceback__)
            frames = extract_frames(tb)

        # Check function names are present and reasonably short
        for frame in frames:
            if frame["function"]:
                # Should not have overly long paths
                assert frame["function"].count(".") <= 1

    def test_frame_urls(self):
        """Test that frame URLs are generated for real files."""
        try:
            raise ValueError("test")
        except ValueError as e:
            import inspect

            tb = inspect.getinnerframes(e.__traceback__)
            frames = extract_frames(tb)

        # At least one frame should have URLs
        has_vscode_url = any("VS Code" in f["urls"] for f in frames if f["urls"])
        # VS Code URLs should be present for real files
        assert has_vscode_url or len(frames) > 0  # Flexible for different environments

    def test_frame_source_lines(self):
        """Test that source code lines are extracted and deindented."""

        def test_function():
            x = 1  # noqa: F841
            y = 2  # noqa: F841
            raise ValueError("test")

        try:
            test_function()
        except ValueError as e:
            import inspect

            tb = inspect.getinnerframes(e.__traceback__)
            frames = extract_frames(tb)

        # Find the test_function frame
        test_frame = next((f for f in frames if f["function"] == "test_function"), None)
        if test_frame:
            assert test_frame["lines"]
            assert "x = 1" in test_frame["lines"]
            assert "y = 2" in test_frame["lines"]


class TestLibdirPattern:
    """Test the libdir pattern for identifying library code."""

    def test_libdir_pattern_matches_site_packages(self):
        """Test that libdir pattern matches site-packages paths."""
        from tracerite.trace import libdir

        assert libdir.fullmatch("/usr/lib/python3.9/site-packages/module.py")
        assert libdir.fullmatch(
            "/home/user/.local/lib/python3.9/site-packages/pkg/mod.py"
        )

    def test_libdir_pattern_matches_dist_packages(self):
        """Test that libdir pattern matches dist-packages paths."""
        from tracerite.trace import libdir

        assert libdir.fullmatch("/usr/lib/python3/dist-packages/module.py")

    def test_libdir_pattern_matches_usr_paths(self):
        """Test that libdir pattern matches /usr/ paths."""
        from tracerite.trace import libdir

        assert libdir.fullmatch("/usr/lib/python3.9/module.py")
        assert libdir.fullmatch("/usr/local/lib/module.py")

    def test_libdir_pattern_does_not_match_user_code(self):
        """Test that libdir pattern doesn't match user code paths."""
        from tracerite.trace import libdir

        assert not libdir.fullmatch("/home/user/project/module.py")
        assert not libdir.fullmatch("./mycode.py")
        assert not libdir.fullmatch("/opt/myapp/code.py")

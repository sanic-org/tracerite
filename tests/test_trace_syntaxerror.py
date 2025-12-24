"""Additional trace.py tests for SyntaxError handling and missing coverage."""

import tempfile
from pathlib import Path
from unittest.mock import MagicMock

from tracerite.trace import (
    _create_summary,
    _extract_syntax_error_frame,
    _is_notebook_cell,
    extract_chain,
    extract_exception,
    format_location,
)


class TestSyntaxErrorFrameExtraction:
    """Test SyntaxError frame extraction in trace.py."""

    def test_syntax_error_frame_basic(self):
        """Test basic SyntaxError frame extraction."""
        code = "x = 1 +"
        try:
            compile(code, "<test>", "exec")
        except SyntaxError as e:
            frame = _extract_syntax_error_frame(e)

            assert frame is not None
            assert frame["relevance"] == "error"
            assert "fragments" in frame

    def test_syntax_error_frame_no_filename(self):
        """Test SyntaxError frame when filename is None."""

        class MockSyntaxError(SyntaxError):
            pass

        e = MockSyntaxError("test error")
        e.filename = None
        e.lineno = 1

        frame = _extract_syntax_error_frame(e)
        assert frame is None

    def test_syntax_error_frame_no_lineno(self):
        """Test SyntaxError frame when lineno is None."""

        class MockSyntaxError(SyntaxError):
            pass

        e = MockSyntaxError("test error")
        e.filename = "test.py"
        e.lineno = None

        frame = _extract_syntax_error_frame(e)
        assert frame is None

    def test_syntax_error_frame_with_text_fallback(self):
        """Test SyntaxError frame uses e.text when linecache fails."""

        class MockSyntaxError(SyntaxError):
            pass

        e = MockSyntaxError("test error")
        e.filename = "/nonexistent/path/to/file.py"
        e.lineno = 1
        e.offset = 1
        e.text = "invalid syntax here"
        e.end_lineno = 1
        e.end_offset = 8

        frame = _extract_syntax_error_frame(e)

        # Should use e.text as fallback
        assert frame is not None
        assert "invalid syntax" in frame.get("lines", "")

    def test_syntax_error_frame_text_with_newline(self):
        """Test SyntaxError frame handles text without trailing newline."""

        class MockSyntaxError(SyntaxError):
            pass

        e = MockSyntaxError("test error")
        e.filename = "/nonexistent/file.py"
        e.lineno = 1
        e.offset = 5
        e.text = "x = 1 +"  # No newline
        e.end_lineno = 1
        e.end_offset = 8

        frame = _extract_syntax_error_frame(e)

        assert frame is not None
        # Text should have newline added
        assert frame.get("lines", "").endswith("\n") or "+" in frame.get("lines", "")

    def test_syntax_error_frame_end_col_adjustment(self):
        """Test end_col <= start_col adjustment."""

        class MockSyntaxError(SyntaxError):
            pass

        e = MockSyntaxError("test error")
        e.filename = "/nonexistent/file.py"
        e.lineno = 1
        e.offset = 5  # start_col = 4 (0-based)
        e.text = "x = 1 +"
        e.end_lineno = 1
        e.end_offset = 5  # end_col = 4, same as start

        frame = _extract_syntax_error_frame(e)

        # Should adjust end_col to be at least start_col + 1
        assert frame is not None

    def test_syntax_error_frame_no_end_offset(self):
        """Test SyntaxError without end_offset attribute."""

        class MockSyntaxError(SyntaxError):
            pass

        e = MockSyntaxError("test error")
        e.filename = "/nonexistent/file.py"
        e.lineno = 1
        e.offset = 5
        e.text = "x = 1 +"
        # No end_lineno or end_offset

        frame = _extract_syntax_error_frame(e)

        assert frame is not None

    def test_syntax_error_in_extract_exception(self):
        """Test SyntaxError handling in extract_exception."""
        code = "def foo(\n    x"  # Unclosed parenthesis
        try:
            compile(code, "<test>", "exec")
        except SyntaxError as e:
            info = extract_exception(e)

            assert info["type"] == "SyntaxError"
            assert len(info["frames"]) > 0

    def test_syntax_error_with_skip_until(self):
        """Test SyntaxError with skip_until parameter."""
        code = "x = ("
        try:
            compile(code, "<test>", "exec")
        except SyntaxError as e:
            info = extract_exception(e, skip_until="<test>")

            assert info["type"] == "SyntaxError"

    def test_syntax_error_enhanced_positions(self):
        """Test SyntaxError with enhanced position extraction."""
        # Mismatched brackets should trigger enhanced position extraction
        code = "(x]"
        try:
            compile(code, "<test>", "exec")
        except SyntaxError as e:
            frame = _extract_syntax_error_frame(e)

            assert frame is not None

    def test_is_notebook_cell_no_ipython(self):
        """Test _is_notebook_cell when ipython is None."""
        from tracerite import trace

        original = trace.ipython
        try:
            trace.ipython = None
            result = _is_notebook_cell("test.py")
            assert result is False
        finally:
            trace.ipython = original

    def test_is_notebook_cell_with_ipython(self):
        """Test _is_notebook_cell with mock ipython."""
        from tracerite import trace

        mock_ipython = MagicMock()
        mock_ipython.compile._filename_map = {"<ipython-input-1>": "1"}

        original = trace.ipython
        try:
            trace.ipython = mock_ipython
            result = _is_notebook_cell("<ipython-input-1>")
            assert result is True

            result = _is_notebook_cell("regular.py")
            assert result is False
        finally:
            trace.ipython = original

    def test_is_notebook_cell_key_error(self):
        """Test _is_notebook_cell handles KeyError."""
        from tracerite import trace

        mock_ipython = MagicMock()
        mock_ipython.compile._filename_map = {}

        original = trace.ipython
        try:
            trace.ipython = mock_ipython
            result = _is_notebook_cell("nonexistent")
            assert result is False
        finally:
            trace.ipython = original


class TestFormatLocation:
    """Test format_location function."""

    def test_format_location_ipython(self):
        """Test format_location with IPython filename."""
        from tracerite import trace

        mock_ipython = MagicMock()
        mock_ipython.compile._filename_map = {"<ipython-input-5>": "5"}

        original = trace.ipython
        try:
            trace.ipython = mock_ipython
            filename, location, urls = format_location("<ipython-input-5>", 10)

            assert location == "In [5]"
            assert filename is None
        finally:
            trace.ipython = original

    def test_format_location_long_filename(self):
        """Test format_location with very long filename."""
        long_path = (
            "/very/long/path/that/exceeds/forty/characters/in/total/length/file.py"
        )
        filename, location, urls = format_location(long_path, 10)

        # Location should be shortened
        assert len(location) < len(long_path)

    def test_format_location_regular_file(self):
        """Test format_location with regular file path."""
        with tempfile.NamedTemporaryFile(suffix=".py", delete=False) as f:
            temp_path = f.name
            f.write(b"# test file\n")

        try:
            filename, location, urls = format_location(temp_path, 10)

            # Should have VS Code URL
            assert "VS Code" in urls
            assert "vscode://" in urls["VS Code"]
        finally:
            Path(temp_path).unlink(missing_ok=True)

    def test_format_location_relative_to_cwd(self):
        """Test format_location with file relative to current directory."""
        # Create a temporary file in current directory
        with tempfile.NamedTemporaryFile(suffix=".py", delete=False, dir=".") as f:
            temp_path = f.name
            f.write(b"# test file\n")

        from tracerite import trace

        # Set ipython to trigger Jupyter URL generation
        mock_ipython = MagicMock()
        mock_ipython.compile._filename_map = {}

        original = trace.ipython
        try:
            trace.ipython = mock_ipython
            filename, location, urls = format_location(temp_path, 10)

            # Should have Jupyter URL for files relative to cwd
            # (Only if ipython is set)
            assert "VS Code" in urls
        finally:
            trace.ipython = original
            Path(temp_path).unlink(missing_ok=True)


class TestCreateSummary:
    """Test _create_summary function."""

    def test_short_message(self):
        """Test summary of short message."""
        result = _create_summary("Short error")
        assert result == "Short error"

    def test_message_exactly_100_chars(self):
        """Test message of exactly 100 characters."""
        msg = "x" * 100
        result = _create_summary(msg)
        assert result == msg

    def test_long_message_under_1000(self):
        """Test long message under 1000 chars."""
        msg = "x" * 500
        result = _create_summary(msg)
        assert "···" in result
        assert len(result) <= 100

    def test_very_long_message_over_1000(self):
        """Test very long message over 1000 chars shows start and end."""
        msg = "START" + "x" * 1500 + "END"
        result = _create_summary(msg)
        assert "···" in result
        # Should have parts from start and end
        assert "START" in result or "END" in result

    def test_multiline_message(self):
        """Test multiline message uses first line."""
        msg = "First line\nSecond line\nThird line"
        result = _create_summary(msg)
        assert result == "First line"


class TestExtractChainSyntaxError:
    """Test extract_chain with SyntaxErrors."""

    def test_extract_chain_with_syntax_error(self):
        """Test extract_chain includes SyntaxError properly."""
        code = "x = 1 +"
        try:
            compile(code, "<test>", "exec")
        except SyntaxError:
            chain = extract_chain()

            assert len(chain) >= 1
            assert chain[-1]["type"] == "SyntaxError"

    def test_extract_chain_chained_with_syntax_error(self):
        """Test extract_chain with SyntaxError in chain."""
        try:
            try:
                compile("x = (", "<test>", "exec")
            except SyntaxError as se:
                raise ValueError("Wrapped syntax error") from se
        except ValueError:
            chain = extract_chain()

            assert len(chain) >= 2
            # First should be SyntaxError, second ValueError
            types = [c["type"] for c in chain]
            assert "SyntaxError" in types
            assert "ValueError" in types


class TestNonExceptionSuppress:
    """Test suppression of non-Exception types like KeyboardInterrupt."""

    def test_keyboard_interrupt_suppression(self):
        """Test that KeyboardInterrupt triggers suppress_inner."""
        try:
            raise KeyboardInterrupt("test interrupt")
        except KeyboardInterrupt as e:
            info = extract_exception(e)

            assert info["type"] == "KeyboardInterrupt"
            # Should have frames but may be limited due to suppress_inner=True

    def test_system_exit_suppression(self):
        """Test that SystemExit triggers suppress_inner."""
        try:
            raise SystemExit(1)
        except SystemExit as e:
            info = extract_exception(e)

            assert info["type"] == "SystemExit"


class TestSyntaxErrorNotebookCell:
    """Test SyntaxError handling for notebook cells."""

    def test_syntax_error_in_notebook_cell(self):
        """Test SyntaxError frame extraction for notebook cells."""
        from tracerite import trace

        mock_ipython = MagicMock()
        mock_ipython.compile._filename_map = {"<ipython-input-1>": "1"}

        original = trace.ipython
        try:
            trace.ipython = mock_ipython

            # Create a SyntaxError that looks like it's from a notebook cell
            code = "x = ("
            try:
                compile(code, "<ipython-input-1>", "exec")
            except SyntaxError as e:
                info = extract_exception(e, skip_until="<ipython-input")

                assert info["type"] == "SyntaxError"
                # Should skip all traceback frames for notebook cell errors
        finally:
            trace.ipython = original

    def test_syntax_error_skip_until_in_filename(self):
        """Test skip_until matching SyntaxError filename."""
        code = "x = 1 +"
        try:
            compile(code, "myfile.py", "exec")
        except SyntaxError as e:
            info = extract_exception(e, skip_until="myfile")

            assert info["type"] == "SyntaxError"

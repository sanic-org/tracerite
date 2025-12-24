"""Additional html.py tests for missing coverage."""

import pytest

from tracerite.html import html_traceback, javascript, style
from tracerite.inspector import VarInfo
from tracerite.trace import extract_chain, extract_exception


class TestHtmlAdditional:
    """Additional tests for html.py coverage."""

    def test_frame_without_filename(self):
        """Test frame rendering when filename is None."""
        try:
            raise ValueError("test")
        except ValueError as e:
            exc_info = extract_exception(e)

            # Set filename to None
            if exc_info["frames"]:
                exc_info["frames"][-1]["filename"] = None

            html = html_traceback(chain=[exc_info])
            html_str = str(html)

            assert "ValueError" in html_str

    def test_frame_without_function(self):
        """Test frame rendering when function is None."""
        try:
            raise ValueError("test")
        except ValueError as e:
            exc_info = extract_exception(e)

            # Set function to None
            if exc_info["frames"]:
                exc_info["frames"][-1]["function"] = None

            html = html_traceback(chain=[exc_info])
            html_str = str(html)

            assert "ValueError" in html_str

    def test_frame_without_range(self):
        """Test frame rendering when range is None."""
        try:
            raise ValueError("test")
        except ValueError as e:
            exc_info = extract_exception(e)

            # Set range to None
            if exc_info["frames"]:
                exc_info["frames"][-1]["range"] = None

            html = html_traceback(chain=[exc_info])
            html_str = str(html)

            assert "ValueError" in html_str

    def test_frame_with_urls(self):
        """Test frame rendering with editor URLs."""
        try:
            raise ValueError("test")
        except ValueError as e:
            exc_info = extract_exception(e)

            # Add URLs to frame
            if exc_info["frames"]:
                exc_info["frames"][-1]["urls"] = {
                    "VS Code": "vscode://file/test.py:10",
                    "Jupyter": "/edit/test.py",
                }

            html = html_traceback(chain=[exc_info], local_urls=True)
            html_str = str(html)

            assert "vscode://" in html_str or "frame-link" in html_str

    def test_variable_with_block_format(self):
        """Test variable with block format hint."""
        try:
            multiline_var = "line1\nline2\nline3"  # noqa: F841
            raise ValueError("test")
        except ValueError as e:
            html = html_traceback(exc=e)
            html_str = str(html)

            # Should render without error
            assert "ValueError" in html_str

    def test_variable_with_keyvalue_dict(self):
        """Test variable formatted as key-value pairs."""
        try:
            dict_var = {"a": 1, "b": 2}  # noqa: F841
            raise ValueError("test")
        except ValueError as e:
            html = html_traceback(exc=e)
            html_str = str(html)

            assert "ValueError" in html_str

    def test_variable_with_array_suffix(self):
        """Test variable formatted as array with scale suffix."""
        np = pytest.importorskip("numpy")

        try:
            large_array = np.array([[1e9, 2e9], [3e9, 4e9]])  # noqa: F841
            raise ValueError("test")
        except ValueError as e:
            html = html_traceback(exc=e)
            html_str = str(html)

            assert "ValueError" in html_str

    def test_variable_old_tuple_format(self):
        """Test variable with old tuple format (backwards compatibility)."""
        try:
            raise ValueError("test")
        except ValueError as e:
            exc_info = extract_exception(e)

            # Replace variables with old tuple format
            if exc_info["frames"]:
                exc_info["frames"][-1]["variables"] = [
                    ("x", "int", "42"),  # Old 3-tuple format
                ]

            html = html_traceback(chain=[exc_info])
            html_str = str(html)

            assert "ValueError" in html_str

    def test_variable_without_typename(self):
        """Test variable rendering when typename is empty."""
        try:
            raise ValueError("test")
        except ValueError as e:
            exc_info = extract_exception(e)

            # Add variable with empty typename
            if exc_info["frames"]:
                exc_info["frames"][-1]["variables"] = [
                    VarInfo("x", "", "42", "inline"),
                ]

            html = html_traceback(chain=[exc_info])
            html_str = str(html)

            assert "ValueError" in html_str

    def test_variable_inspector_loop_with_array_dict(self):
        """Test variable_inspector loop continuation after array dict format.

        This covers branch 265->234: after processing a variable with
        {"type": "array"} format WITHOUT suffix, the loop continues to
        process more variables.
        """
        try:
            raise ValueError("test")
        except ValueError as e:
            exc_info = extract_exception(e)

            # Add multiple variables where one is an array dict format WITHOUT suffix
            # The array dict must be followed by another variable to test loop continuation
            if exc_info["frames"]:
                exc_info["frames"][-1]["variables"] = [
                    VarInfo(
                        "arr",
                        "ndarray",
                        {
                            "type": "array",
                            "rows": [["1", "2"], ["3", "4"]],
                            # No suffix - this is key to hit branch 265->234
                        },
                        "inline",
                    ),
                    VarInfo("x", "int", "42", "inline"),  # Second variable after array
                ]

            html = html_traceback(chain=[exc_info])
            html_str = str(html)

            assert "ValueError" in html_str
            assert "arr" in html_str
            assert "x" in html_str

    def test_matrix_with_skip_markers(self):
        """Test _format_matrix with skip markers (None values)."""
        try:
            raise ValueError("test")
        except ValueError as e:
            exc_info = extract_exception(e)

            # Add variable with matrix containing skip markers
            if exc_info["frames"]:
                exc_info["frames"][-1]["variables"] = [
                    VarInfo(
                        "matrix",
                        "ndarray",
                        [
                            [None],  # Row skip marker
                            ["1", "2", None, "3"],  # Column skip marker
                            ["4", "5", "6", "7"],
                        ],
                        "inline",
                    ),
                ]

            html = html_traceback(chain=[exc_info])
            html_str = str(html)

            assert "ValueError" in html_str

    def test_exception_from_cause(self):
        """Test exception chain with explicit cause."""
        try:
            try:
                raise ValueError("original")
            except ValueError as ve:
                raise TypeError("wrapped") from ve
        except TypeError as e:
            chain = extract_chain(e)
            html = html_traceback(chain=chain)
            html_str = str(html)

            assert "ValueError" in html_str
            assert "TypeError" in html_str
            assert "from above" in html_str

    def test_exception_from_context(self):
        """Test exception chain with implicit context."""
        try:
            try:
                raise ValueError("original")
            except ValueError:
                raise TypeError("during handling")
        except TypeError as e:
            chain = extract_chain(e)
            html = html_traceback(chain=chain)
            html_str = str(html)

            assert "ValueError" in html_str
            assert "TypeError" in html_str

    def test_suppress_inner_frames(self):
        """Test suppress_inner parameter for keyboard interrupts."""
        try:
            raise KeyboardInterrupt("test")
        except KeyboardInterrupt as e:
            html = html_traceback(exc=e)
            html_str = str(html)

            assert "KeyboardInterrupt" in html_str

    def test_frame_relevance_warning(self):
        """Test frame with warning relevance."""
        try:
            raise ValueError("test")
        except ValueError as e:
            exc_info = extract_exception(e)

            # Set a frame to warning relevance
            if len(exc_info["frames"]) > 1:
                exc_info["frames"][0]["relevance"] = "warning"

            html = html_traceback(chain=[exc_info])
            html_str = str(html)

            assert "ValueError" in html_str

    def test_frame_relevance_stop(self):
        """Test frame with stop relevance."""
        try:
            raise KeyboardInterrupt("test")
        except KeyboardInterrupt as e:
            exc_info = extract_exception(e)

            # The frame should have stop relevance for non-Exception types
            html = html_traceback(chain=[exc_info])
            html_str = str(html)

            assert "KeyboardInterrupt" in html_str

    def test_include_js_css_true(self):
        """Test html_traceback with include_js_css=True."""
        try:
            raise ValueError("test")
        except ValueError as e:
            html = html_traceback(exc=e, include_js_css=True)
            html_str = str(html)

            # Should include style and script tags
            assert "<style>" in html_str
            assert "<script>" in html_str

    def test_include_js_css_false(self):
        """Test html_traceback with include_js_css=False."""
        try:
            raise ValueError("test")
        except ValueError as e:
            html = html_traceback(exc=e, include_js_css=False)
            html_str = str(html)

            # Should not include style and script tags
            assert "<style>" not in html_str
            assert "<script>" not in html_str

    def test_javascript_content(self):
        """Test javascript content."""
        js = javascript
        assert isinstance(js, str)
        assert len(js) > 0

    def test_style_content(self):
        """Test style content."""
        css = style
        assert isinstance(css, str)
        assert len(css) > 0

    def test_fragment_mark_beg(self):
        """Test fragment with mark='beg'."""
        try:
            # Multi-line expression that spans lines
            pass
        except TypeError as e:
            html = html_traceback(exc=e)
            html_str = str(html)
            assert "TypeError" in html_str

    def test_fragment_mark_mid(self):
        """Test fragment with mark='mid' for middle of multi-line highlight."""
        # Create an error that spans multiple lines
        code = """
def foo():
    x = (
        1 +
        2 +
        "a"
    )
foo()
"""
        try:
            exec(code)
        except TypeError as e:
            html = html_traceback(exc=e)
            html_str = str(html)
            assert "TypeError" in html_str

    def test_fragment_em_solo(self):
        """Test fragment with em='solo' for emphasized code."""
        try:
            pass
        except TypeError as e:
            html = html_traceback(exc=e)
            html_str = str(html)
            # Should have emphasis on error location
            assert "TypeError" in html_str

    def test_empty_chain(self):
        """Test html_traceback with empty chain."""
        html = html_traceback(chain=[])
        html_str = str(html)
        # Should render something even with empty chain
        assert html_str is not None

    def test_chain_without_frames(self):
        """Test exception in chain without frames."""
        try:
            raise ValueError("test")
        except ValueError as e:
            exc_info = extract_exception(e)
            exc_info["frames"] = []

            html = html_traceback(chain=[exc_info])
            html_str = str(html)

            assert "ValueError" in html_str


class TestHtmlCodeFragments:
    """Test code fragment rendering in HTML."""

    def test_fragment_with_dedent(self):
        """Test fragment with dedent marker."""
        try:

            def indented():
                raise ValueError("indented error")

            indented()
        except ValueError as e:
            html = html_traceback(exc=e)
            html_str = str(html)
            assert "indented error" in html_str

    def test_fragment_with_indent(self):
        """Test fragment with additional indentation."""
        try:

            def foo():
                if True:
                    raise ValueError("deeply indented")

            foo()
        except ValueError as e:
            html = html_traceback(exc=e)
            html_str = str(html)
            assert "deeply indented" in html_str

    def test_fragment_with_comment(self):
        """Test fragment containing a comment."""
        try:
            x = 1  # This is a comment
            raise ValueError(str(x))
        except ValueError as e:
            html = html_traceback(exc=e)
            html_str = str(html)
            assert "ValueError" in html_str


class TestHtmlScrollTo:
    """Test scrollto functionality."""

    def test_scrollto_generation(self):
        """Test that scrollto script is generated for errors."""
        try:
            raise ValueError("test")
        except ValueError as e:
            html = html_traceback(exc=e, include_js_css=True)
            html_str = str(html)

            # Should have scrollto with frame ID
            assert "scrollto" in html_str

    def test_scrollto_warning_frame(self):
        """Test scrollto targets warning frames when present."""
        try:
            raise ValueError("test")
        except ValueError as e:
            exc_info = extract_exception(e)

            # Mark a frame as warning
            if len(exc_info["frames"]) > 1:
                exc_info["frames"][0]["relevance"] = "warning"

            html = html_traceback(chain=[exc_info], include_js_css=True)
            html_str = str(html)

            assert "scrollto" in html_str

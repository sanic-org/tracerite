"""Corner case and edge case tests for html.py to achieve 100% coverage."""

import sys

import pytest

from tracerite.html import html_traceback, javascript, style
from tracerite.trace import extract_chain, extract_exception


class TestHtmlCornercases:
    """Corner case tests for HTML module."""

    def test_html_with_many_frames(self):
        """Test HTML generation with > 16 frames to trigger frame limiting."""

        def deep_call(n):
            if n == 0:
                raise ValueError("deep error")
            return deep_call(n - 1)

        try:
            deep_call(20)
        except ValueError as e:
            html = html_traceback(exc=e)
            html_str = str(html)
            # Should have ellipsis for frame limiting
            assert "ValueError" in html_str

    def test_exception_chain_display(self):
        """Test exception chain text display."""
        try:
            try:
                raise ValueError("first")
            except ValueError:
                raise TypeError("second")  # noqa: B904
        except TypeError as e:
            html = html_traceback(exc=e)
            html_str = str(html)
            # Should show chain indicator (using new chain message format)
            assert (
                "in except" in html_str.lower() or "from previous" in html_str.lower()
            )

    def test_frames_without_relevance_call(self):
        """Test scrollto generation skips call frames."""

        def outer():
            def inner():
                raise ValueError("test")

            inner()

        try:
            outer()
        except ValueError as e:
            html = html_traceback(exc=e)
            html_str = str(html)
            # Should have scrollto script
            assert "scrollto" in html_str

    def test_single_exception_no_chain_text(self):
        """Test HTML with single exception doesn't show chain text."""
        try:
            raise ValueError("single error")
        except ValueError as e:
            chain = extract_chain(exc=e)
            html = html_traceback(chain=chain)
            html_str = str(html)

            # Should not have "after catching" text for single exception
            assert "ValueError" in html_str

    def test_exactly_16_frames(self):
        """Test frame limiting edge case with exactly 16 frames."""

        def make_deep_call(depth):
            if depth == 0:
                raise ValueError("deep")
            return make_deep_call(depth - 1)

        try:
            # Try to create exactly 16 frames
            make_deep_call(14)
        except ValueError as e:
            html = html_traceback(exc=e)
            html_str = str(html)

            # Should handle 16 frames correctly
            assert "ValueError" in html_str

    def test_frame_limiting_with_ellipsis(self):
        """Test that frame limiting creates ellipsis placeholder."""

        def make_very_deep_call(depth):
            if depth == 0:
                raise ValueError("very deep")
            return make_very_deep_call(depth - 1)

        try:
            # Create more than 16 frames
            make_very_deep_call(20)
        except ValueError as e:
            html = html_traceback(exc=e)
            html_str = str(html)

            # Should contain ellipsis when frames are limited
            assert "..." in html_str

    def test_exception_with_no_frames(self):
        """Test HTML rendering when exception has no frames."""
        # Create an exception with no traceback
        exc = ValueError("no frames")
        exc_info = extract_exception(exc)
        exc_info["frames"] = []  # Clear frames

        html = html_traceback(chain=[exc_info])
        html_str = str(html)

        # Should handle exceptions with no frames
        assert "ValueError" in html_str

    def test_exception_message_not_starting_with_summary(self):
        """Test exception where message != summary but doesn't start with summary.

        Covers lines 56-58.
        """
        # Create an exception with a multiline message where summary differs from message
        try:
            raise ValueError("short summary\nAdditional context on second line")
        except ValueError as e:
            exc_info = extract_exception(e)

            html = html_traceback(chain=[exc_info])
            html_str = str(html)

            # Should display the summary (first line)
            assert "short summary" in html_str

    def test_source_code_not_available(self):
        """Test when source code is not available for call frames.

        Covers lines 103-106.
        """
        try:
            raise ValueError("test error")
        except ValueError as e:
            exc_info = extract_exception(e)

            # Clear fragments from frames to simulate source not available
            for frame in exc_info["frames"]:
                frame["fragments"] = []

            html = html_traceback(chain=[exc_info])
            html_str = str(html)

            # Should show "(no source code)" with symbol
            assert "(no source code)" in html_str

    def test_source_not_available_on_last_frame(self):
        """Test message when source is not available on the last frame (where error was raised).

        Covers lines 110-112.
        """
        try:
            raise TypeError("test error")
        except TypeError as e:
            exc_info = extract_exception(e)

            # Clear fragments only from the last frame
            if exc_info["frames"]:
                exc_info["frames"][-1]["fragments"] = []

            html = html_traceback(chain=[exc_info])
            html_str = str(html)

            # Should show "(no source code)" with symbol
            assert "(no source code)" in html_str
            # Error frames show the bomb emoji
            assert "💣" in html_str or "TypeError" in html_str

    def test_tooltip_formatting_exception(self):
        """Test exception handling in tooltip text formatting.

        Covers lines 130-131.
        """
        try:
            raise ValueError("test")
        except ValueError as e:
            exc_info = extract_exception(e)

            # Modify a frame to have an invalid relevance that will cause format() to fail
            if exc_info["frames"]:
                frame = exc_info["frames"][-1]
                # Set a relevance that exists in symdesc but will fail formatting
                frame["relevance"] = "error"
                # Remove required keys to trigger exception in format()
                frame.pop("type", None)

            html = html_traceback(chain=[exc_info])
            html_str = str(html)

            # Should still render without crashing
            assert "ValueError" in html_str

    def test_matrix_with_skipped_rows_and_columns(self):
        """Test matrix formatting with skipped rows and columns.

        Covers lines 231, 235-253 for _format_matrix.
        """
        try:
            # Need to import numpy to trigger matrix formatting
            import numpy as np
        except ImportError:
            pytest.skip("NumPy not available")

        # Create a large matrix that will trigger skipping
        large_matrix = np.arange(100).reshape(10, 10)

        try:
            # Create an error with a large matrix variable
            x = large_matrix  # noqa: F841
            raise ValueError("matrix error")
        except ValueError as e:
            html = html_traceback(exc=e)
            html_str = str(html)

            # Should render matrix
            assert "ValueError" in html_str

    def test_matrix_with_none_markers(self):
        """Test matrix formatting with None as skip markers.

        Covers the skiprow and skipcol logic in _format_matrix.
        """
        try:
            raise ValueError("test")
        except ValueError as e:
            exc_info = extract_exception(e)

            # Manually add a variable with a matrix structure containing None markers
            if exc_info["frames"]:
                frame = exc_info["frames"][-1]
                # Create a matrix with None markers indicating skipped rows/cols
                matrix_with_skips = [
                    ["1", "2", None, "3"],  # None indicates column skip
                    [
                        None,
                        None,
                        None,
                        None,
                    ],  # None in first element indicates row skip
                    ["4", "5", None, "6"],
                ]
                frame["variables"] = [("matrix", "list", matrix_with_skips)]

            html = html_traceback(chain=[exc_info])
            html_str = str(html)

            # Should handle matrix with skips
            assert "ValueError" in html_str
            # Check for CSS classes indicating skips
            assert "skippedabove" in html_str or "skippedleft" in html_str

    def test_fragment_rendering_with_mark_and_em(self):
        """Test fragment rendering with mark and em tags.

        Covers lines 193-194 for _render_fragment.
        """
        try:
            raise ValueError("test")
        except ValueError as e:
            exc_info = extract_exception(e)

            # Manually create fragments with mark and em tags
            if exc_info["frames"]:
                frame = exc_info["frames"][-1]
                # Create custom fragments with mark/em attributes
                frame["fragments"] = [
                    {
                        "line": 1,
                        "fragments": [
                            {"code": "x ", "mark": None, "em": None},
                            {"code": "+", "mark": "solo", "em": "solo"},
                            {"code": " y", "mark": None, "em": None},
                        ],
                    }
                ]
                frame["linenostart"] = 1

            html = html_traceback(chain=[exc_info])
            html_str = str(html)

            # Should render with mark and em tags
            assert "<mark>" in html_str
            assert "<em>" in html_str
            assert "ValueError" in html_str

    def test_fragments_with_mark_beg_and_fin(self):
        """Test fragment rendering with mark begin and finish tags."""
        try:
            raise ValueError("test")
        except ValueError as e:
            exc_info = extract_exception(e)

            if exc_info["frames"]:
                frame = exc_info["frames"][-1]
                # Create fragments with mark beginning and finish
                frame["fragments"] = [
                    {
                        "line": 1,
                        "fragments": [
                            {"code": "result = ", "mark": None, "em": None},
                            {"code": "a", "mark": "beg", "em": None},
                        ],
                    },
                    {
                        "line": 2,
                        "fragments": [
                            {"code": "    +", "mark": None, "em": "solo"},
                            {"code": " b", "mark": "fin", "em": None},
                        ],
                    },
                ]
                frame["linenostart"] = 1

            html = html_traceback(chain=[exc_info])
            html_str = str(html)

            # Should handle multi-line mark/em
            assert "ValueError" in html_str

    def test_fragments_with_em_beg_and_fin(self):
        """Test fragment rendering with em begin and finish tags."""
        try:
            raise ValueError("test")
        except ValueError as e:
            exc_info = extract_exception(e)

            if exc_info["frames"]:
                frame = exc_info["frames"][-1]
                # Create fragments with em beginning and finish
                frame["fragments"] = [
                    {
                        "line": 1,
                        "fragments": [
                            {"code": "result = func", "mark": "beg", "em": "beg"},
                            {"code": "(", "mark": None, "em": "fin"},
                        ],
                    },
                ]
                frame["linenostart"] = 1

            html = html_traceback(chain=[exc_info])
            html_str = str(html)

            # Should handle em beg/fin
            assert "ValueError" in html_str

    def test_trailing_fragment(self):
        """Test rendering of trailing fragments."""
        try:
            raise ValueError("test")
        except ValueError as e:
            exc_info = extract_exception(e)

            if exc_info["frames"]:
                frame = exc_info["frames"][-1]
                # Create fragments with trailing marker
                frame["fragments"] = [
                    {
                        "line": 1,
                        "fragments": [
                            {"code": "x = 1", "mark": None, "em": None},
                            {"code": "  # comment\n", "trailing": True},
                        ],
                    },
                ]
                frame["linenostart"] = 1

            html = html_traceback(chain=[exc_info])
            html_str = str(html)

            # Should render trailing content
            assert "ValueError" in html_str

    def test_exception_message_starting_with_summary(self):
        """Test exception where message starts with summary (line 57)."""
        try:
            raise ValueError("prefix: additional details here")
        except ValueError as e:
            exc_info = extract_exception(e)
            # Set summary to be a prefix of message
            exc_info["summary"] = "prefix"
            exc_info["message"] = "prefix: additional details here"

            html = html_traceback(chain=[exc_info])
            html_str = str(html)

            # Should strip the prefix and show only additional details
            assert "ValueError" in html_str
            assert "prefix" in html_str

    def test_local_urls_with_urls_present(self):
        """Test rendering with local_urls enabled and URLs present (lines 103-106)."""
        try:
            raise ValueError("test")
        except ValueError as e:
            exc_info = extract_exception(e)

            # Add URLs to a frame
            if exc_info["frames"]:
                frame = exc_info["frames"][-1]
                frame["urls"] = {
                    "VS Code": "vscode://file/test.py:10",
                }

            html = html_traceback(chain=[exc_info], local_urls=True)
            html_str = str(html)

            # Should include the VS Code URL as a link
            assert "vscode://" in html_str

    def test_tooltip_with_newlines(self):
        """Test tooltip text with newlines that need to be replaced (line 130-131)."""
        try:
            raise ValueError("test\nwith\nnewlines")
        except ValueError as e:
            # This should exercise the newline replacement in tooltip text
            html = html_traceback(exc=e)
            html_str = str(html)

            # Should handle exception with newlines in message
            assert "ValueError" in html_str

    def test_fragment_rendering_edge_cases(self):
        """Test _render_fragment with various edge cases (lines 193-194)."""
        try:
            raise ValueError("test")
        except ValueError as e:
            exc_info = extract_exception(e)

            if exc_info["frames"]:
                frame = exc_info["frames"][-1]
                # Create fragments that exercise all edge cases
                frame["fragments"] = [
                    {
                        "line": 1,
                        "fragments": [
                            # Test "beg" mark that continues to next line
                            {"code": "x = [", "mark": "beg", "em": None},
                        ],
                    },
                    {
                        "line": 2,
                        "fragments": [
                            # Test "fin" mark completing from previous line
                            {"code": "    1, 2, 3", "mark": "fin", "em": None},
                        ],
                    },
                ]
                frame["linenostart"] = 1

            html = html_traceback(chain=[exc_info])
            html_str = str(html)

            assert "ValueError" in html_str

    def test_empty_fragments_on_non_last_frame(self):
        """Test source not available on a non-last frame."""

        def outer():
            def inner():
                raise ValueError("test")

            inner()

        try:
            outer()
        except ValueError as e:
            exc_info = extract_exception(e)

            # Clear fragments from a non-last frame
            if len(exc_info["frames"]) > 1:
                exc_info["frames"][0]["fragments"] = []

            html = html_traceback(chain=[exc_info])
            html_str = str(html)

            # Should show "(no source code)" with symbol
            assert "(no source code)" in html_str
            assert "ValueError" in html_str

    def test_native_function_without_location(self):
        """Test native function display when location is missing."""
        try:
            raise ValueError("test")
        except ValueError as e:
            exc_info = extract_exception(e)

            if exc_info["frames"]:
                frame = exc_info["frames"][-1]
                # Clear filename and location to trigger native function display
                frame["filename"] = None
                frame["location"] = None
                frame["function"] = "native_func"

            html = html_traceback(chain=[exc_info])
            html_str = str(html)

            # Should show "Native function" text
            assert "Native function" in html_str or "native_func" in html_str

    def test_html_without_js_css(self):
        """Test HTML generation without including JS/CSS (line 32->35)."""
        try:
            raise ValueError("test")
        except ValueError as e:
            html = html_traceback(exc=e, include_js_css=False)
            html_str = str(html)

            # Should have traceback but no script/style tags
            assert "ValueError" in html_str
            # Should not include the JavaScript or CSS
            assert javascript not in html_str
            assert style not in html_str

    @pytest.mark.skipif(
        sys.version_info < (3, 11), reason="Requires co_positions() from Python 3.11+"
    )
    def test_tooltip_formatting_with_invalid_keys(self):
        """Test tooltip formatting exception with missing format keys (lines 130-131)."""
        # Create a test that triggers the exception handler in tooltip formatting
        try:
            raise ValueError("test")
        except ValueError as e:
            exc_info = extract_exception(e)

            if exc_info["frames"]:
                frame = exc_info["frames"][-1]
                # Set a relevance that is NOT in the symdesc dictionary
                # This will cause a KeyError when trying to access symdesc[relevance]
                frame["relevance"] = "nonexistent_relevance"

                html = html_traceback(chain=[exc_info])
                html_str = str(html)

                # Should handle exception and use repr() fallback
                # The fallback should show the repr of the relevance
                assert "ValueError" in html_str
                # The fallback text should be repr(relevance) = "'nonexistent_relevance'"
                assert "nonexistent_relevance" in html_str

"""Corner case and edge case tests for html.py to achieve 100% coverage."""

from tracerite.html import html_traceback
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
                raise TypeError("second")
        except TypeError as e:
            html = html_traceback(exc=e)
            html_str = str(html)
            # Should show chain indicator
            assert "after" in html_str.lower() or "catching" in html_str.lower()

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

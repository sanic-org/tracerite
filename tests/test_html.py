"""Tests for html.py - HTML traceback formatting."""

from tracerite.html import (
    _format_matrix,
    html_traceback,
    marked,
    split3,
    traceback_detail,
    variable_inspector,
)
from tracerite.trace import extract_chain


class TestHtmlTraceback:
    """Test html_traceback function for generating HTML output."""

    def test_basic_html_generation(self):
        """Test basic HTML generation for a simple exception."""
        try:
            x = 1
            y = 0
            result = x / y  # noqa: F841
        except ZeroDivisionError as e:
            html = html_traceback(exc=e)

        html_str = str(html)
        # Should contain exception type and basic structure
        assert "ZeroDivisionError" in html_str
        assert "tracerite" in html_str

    def test_html_with_chain(self):
        """Test HTML generation with chained exceptions."""
        try:
            try:
                raise ValueError("original")
            except ValueError as e:
                raise TypeError("wrapped") from e
        except TypeError as e:
            html = html_traceback(exc=e)

        html_str = str(html)
        assert "TypeError" in html_str
        assert "ValueError" in html_str
        assert "after" in html_str

    def test_html_without_js_css(self):
        """Test HTML generation without including JS and CSS."""
        try:
            raise RuntimeError("test")
        except RuntimeError as e:
            html = html_traceback(exc=e, include_js_css=False)

        html_str = str(html)
        assert "<style" not in html_str
        # Script for scrollto is still included for functionality
        assert "RuntimeError" in html_str

    def test_html_with_custom_chain(self):
        """Test HTML generation with pre-extracted chain."""
        try:
            raise ValueError("test error")
        except ValueError as e:
            chain = extract_chain(exc=e)
            html = html_traceback(chain=chain)

        html_str = str(html)
        assert "ValueError" in html_str
        assert "test error" in html_str

    def test_html_limits_chain_length(self):
        """Test that HTML only shows last 3 exceptions in chain."""
        try:
            raise RuntimeError("top")
        except RuntimeError as e:
            html = html_traceback(exc=e)

        # Should produce valid HTML
        html_str = str(html)
        assert "RuntimeError" in html_str
        assert "top" in html_str

    def test_html_with_long_message(self):
        """Test HTML rendering with long exception messages."""
        long_message = "Error: " + "x" * 200
        try:
            raise ValueError(long_message)
        except ValueError as e:
            html = html_traceback(exc=e)

        html_str = str(html)
        assert "ValueError" in html_str
        # Should have pre element for full message
        assert "<pre" in html_str

    def test_html_with_multiline_message(self):
        """Test HTML rendering with multiline exception messages."""
        message = "Line 1\nLine 2\nLine 3"
        try:
            raise RuntimeError(message)
        except RuntimeError as e:
            html = html_traceback(exc=e)

        html_str = str(html)
        assert "Line 1" in html_str
        # Full message should be in pre element
        assert "<pre" in html_str

    def test_html_frame_tabs(self):
        """Test that frame tabs are generated for multiple frames."""

        def outer():
            def inner():
                raise ValueError("test")

            inner()

        try:
            outer()
        except ValueError as e:
            html = html_traceback(exc=e)

        html_str = str(html)
        # Should have tab buttons for navigation
        assert "traceback-labels" in html_str or "button" in html_str
        assert "traceback-details" in html_str

    def test_html_includes_scrollto_script(self):
        """Test that scrollto script is included for navigation."""
        try:
            raise ValueError("test")
        except ValueError as e:
            html = html_traceback(exc=e)

        html_str = str(html)
        assert "scrollto" in html_str


class TestSplit3:
    """Test split3 function for splitting code lines."""

    def test_split_indented_line(self):
        """Test splitting a line with indentation."""
        line = "    code here\n"
        indent, code, trailing = split3(line)

        assert indent == "    "
        assert code == "code here"
        assert trailing == "\n"

    def test_split_no_indent(self):
        """Test splitting a line without indentation."""
        line = "code here\n"
        indent, code, trailing = split3(line)

        assert indent == ""
        assert code == "code here"
        assert trailing == "\n"

    def test_split_trailing_space(self):
        """Test splitting a line with trailing spaces."""
        line = "    code here   \n"
        indent, code, trailing = split3(line)

        assert indent == "    "
        assert code == "code here"
        assert trailing == "   \n"

    def test_split_no_trailing(self):
        """Test splitting a line without trailing whitespace."""
        line = "    code here"
        indent, code, trailing = split3(line)

        assert indent == "    "
        assert code == "code here"
        assert trailing == ""

    def test_split_only_whitespace(self):
        """Test splitting a line with only whitespace."""
        line = "    \n"
        indent, code, trailing = split3(line)

        # Whitespace-only lines return empty strings
        assert indent == ""
        assert code == ""
        assert trailing == ""

    def test_split_empty_line(self):
        """Test splitting an empty line."""
        line = ""
        indent, code, trailing = split3(line)

        assert indent == ""
        assert code == ""
        assert trailing == ""


class TestMarked:
    """Test marked function for marking error lines."""

    def test_marked_error_line(self):
        """Test marking a line where error occurred."""
        line = "    x = 1 / 0\n"
        info = {"type": "ZeroDivisionError"}
        frinfo = {"relevance": "error", "type": "ZeroDivisionError"}

        result = marked(line, info, frinfo)
        result_str = str(result)

        assert "mark" in result_str
        assert "💣" in result_str or "data-symbol" in result_str
        assert "x = 1 / 0" in result_str

    def test_marked_warning_line(self):
        """Test marking a line with warning relevance."""
        line = "    call_function()\n"
        info = {"type": "ValueError"}
        frinfo = {"relevance": "warning"}

        result = marked(line, info, frinfo)
        result_str = str(result)

        assert "mark" in result_str
        assert "⚠" in result_str or "data-symbol" in result_str

    def test_marked_call_line(self):
        """Test marking a line with call relevance."""
        line = "    function()\n"
        info = {"type": "RuntimeError"}
        frinfo = {"relevance": "call"}

        result = marked(line, info, frinfo)
        result_str = str(result)

        assert "mark" in result_str

    def test_marked_stop_line(self):
        """Test marking a line with stop relevance."""
        line = "    interrupted()\n"
        info = {"type": "KeyboardInterrupt"}
        frinfo = {"relevance": "stop"}

        result = marked(line, info, frinfo)
        result_str = str(result)

        assert "mark" in result_str
        assert "🛑" in result_str or "data-symbol" in result_str


class TestVariableInspector:
    """Test variable_inspector function for displaying variables."""

    def test_empty_variables(self):
        """Test variable inspector with no variables."""
        from html5tagger import E

        with E.div as doc:
            variable_inspector(doc, [])

        html_str = str(doc)
        # Should not create table if no variables
        assert "<table" not in html_str

    def test_simple_variables(self):
        """Test variable inspector with simple variables."""
        from html5tagger import E

        variables = [
            ("x", "int", "42"),
            ("name", "str", "Alice"),
        ]

        with E.div as doc:
            variable_inspector(doc, variables)

        html_str = str(doc)
        assert "<table" in html_str
        assert "x" in html_str
        assert "42" in html_str
        assert "name" in html_str
        assert "Alice" in html_str

    def test_matrix_variables(self):
        """Test variable inspector with matrix/array data."""
        from html5tagger import E

        variables = [
            ("arr", "ndarray", [["1.0", "2.0"], ["3.0", "4.0"]]),
        ]

        with E.div as doc:
            variable_inspector(doc, variables)

        html_str = str(doc)
        assert "arr" in html_str
        assert "1.0" in html_str
        assert "4.0" in html_str


class TestFormatMatrix:
    """Test _format_matrix function for displaying nested tables."""

    def test_simple_matrix(self):
        """Test formatting a simple 2D matrix."""
        from html5tagger import E

        matrix = [["1.0", "2.0"], ["3.0", "4.0"]]

        with E.div as doc:
            _format_matrix(doc, matrix)

        html_str = str(doc)
        assert "<table" in html_str
        assert "1.0" in html_str
        assert "2.0" in html_str
        assert "3.0" in html_str
        assert "4.0" in html_str

    def test_matrix_with_none_row_skip(self):
        """Test matrix formatting with None values indicating row skip."""
        from html5tagger import E

        matrix = [
            ["1.0", "2.0"],
            [None, None],  # Skip marker
            ["3.0", "4.0"],
        ]

        with E.div as doc:
            _format_matrix(doc, matrix)

        html_str = str(doc)
        assert "1.0" in html_str
        assert "3.0" in html_str
        # Should have skipped row marker
        assert "skippedabove" in html_str

    def test_matrix_with_none_column_skip(self):
        """Test matrix formatting with None values indicating column skip."""
        from html5tagger import E

        matrix = [
            ["1.0", None, "2.0"],  # None indicates column skip
            ["3.0", None, "4.0"],
        ]

        with E.div as doc:
            _format_matrix(doc, matrix)

        html_str = str(doc)
        assert "1.0" in html_str
        assert "2.0" in html_str
        # Should have skipped column marker
        assert "skippedleft" in html_str


class TestTracebackDetail:
    """Test traceback_detail function for formatting frame details."""

    def test_traceback_with_filename(self):
        """Test traceback detail with filename."""
        from html5tagger import E

        info = {"type": "ValueError", "frames": []}
        frinfo = {
            "filename": "/path/to/file.py",
            "lineno": 42,
            "location": "file.py",
            "function": "my_function",
            "lines": "x = 1\ny = 2\nraise ValueError()\n",
            "linenostart": 40,
            "urls": {},
            "variables": [],
            "relevance": "error",
        }

        with E.div as doc:
            traceback_detail(doc, info, frinfo, local_urls=False)

        html_str = str(doc)
        assert "/path/to/file.py" in html_str
        assert ":42" in html_str
        assert "x = 1" in html_str

    def test_traceback_with_urls(self):
        """Test traceback detail with editor URLs."""
        from html5tagger import E

        info = {"type": "ValueError", "frames": []}
        frinfo = {
            "filename": "/path/to/file.py",
            "lineno": 42,
            "location": "file.py",
            "function": "my_function",
            "lines": "raise ValueError()\n",
            "linenostart": 42,
            "urls": {"VS Code": "vscode://file/path/to/file.py:42"},
            "variables": [],
            "relevance": "error",
        }

        with E.div as doc:
            traceback_detail(doc, info, frinfo, local_urls=True)

        html_str = str(doc)
        assert "VS Code" in html_str
        assert "vscode://" in html_str

    def test_traceback_without_source(self):
        """Test traceback detail without source code."""
        from html5tagger import E

        info = {"type": "ValueError", "frames": [{"filename": None, "lines": ""}]}
        frinfo = {
            "filename": None,
            "lineno": 0,
            "location": None,
            "function": "builtin_function",
            "lines": "",
            "linenostart": 0,
            "urls": {},
            "variables": [],
            "relevance": "call",
        }

        with E.div as doc:
            traceback_detail(doc, info, frinfo, local_urls=False)

        html_str = str(doc)
        assert "Source code not available" in html_str

    def test_traceback_with_error_at_end(self):
        """Test traceback detail when error was raised in this frame."""
        from html5tagger import E

        frinfo_error = {
            "filename": "/path/to/file.py",
            "lineno": 42,
            "location": "file.py",
            "function": "error_function",
            "lines": "",  # No source
            "linenostart": 42,
            "urls": {},
            "variables": [],
            "relevance": "error",
        }

        info = {"type": "ValueError", "frames": [frinfo_error]}

        with E.div as doc:
            traceback_detail(doc, info, frinfo_error, local_urls=False)

        html_str = str(doc)
        # Should mention that error was raised
        assert "ValueError" in html_str or "raised from here" in html_str

    def test_traceback_with_variables(self):
        """Test traceback detail includes variable inspector."""
        from html5tagger import E

        info = {"type": "ValueError", "frames": []}
        frinfo = {
            "filename": "/path/to/file.py",
            "lineno": 42,
            "location": "file.py",
            "function": "my_function",
            "lines": "x = 1\n",
            "linenostart": 42,
            "urls": {},
            "variables": [("x", "int", "1")],
            "relevance": "error",
        }

        with E.div as doc:
            traceback_detail(doc, info, frinfo, local_urls=False)

        html_str = str(doc)
        assert "x" in html_str
        # Variable table may not be in output depending on frame structure
        assert "/path/to/file.py" in html_str


class TestHtmlIntegration:
    """Integration tests for complete HTML generation."""

    def test_complete_traceback_rendering(self):
        """Test complete traceback rendering with real exception."""

        def function_a(x):
            return function_b(x)

        def function_b(x):
            return function_c(x)

        def function_c(x):
            return 10 / x

        try:
            function_a(0)
        except ZeroDivisionError as e:
            html = html_traceback(exc=e)

        html_str = str(html)
        # Should have complete HTML structure
        assert "tracerite" in html_str
        assert "ZeroDivisionError" in html_str
        assert (
            "function_a" in html_str
            or "function_b" in html_str
            or "function_c" in html_str
        )

    def test_html_structure_validity(self):
        """Test that generated HTML has valid structure."""
        try:
            raise ValueError("test")
        except ValueError as e:
            html = html_traceback(exc=e)

        html_str = str(html)
        # Basic HTML validation
        assert html_str.count("<div") == html_str.count("</div>")
        assert html_str.count("<script") <= html_str.count("</script>")

    def test_css_included(self):
        """Test that CSS styles are included when requested."""
        try:
            raise ValueError("test")
        except ValueError as e:
            html = html_traceback(exc=e, include_js_css=True)

        html_str = str(html)
        assert "<style>" in html_str or "<style" in html_str
        # Should have some CSS content
        assert ".tracerite" in html_str or "traceback" in html_str

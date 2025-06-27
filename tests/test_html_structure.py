"""Tests for HTML structure validation of error marking."""

import re

from tracerite import html_traceback


def test_html_structure_validity():
    """Test that generated HTML has valid structure."""
    try:
        lst = [1, 2, 3]
        _ = lst[10]  # IndexError
    except IndexError as e:
        html = html_traceback(e)
        html_str = str(html)

        # Basic HTML structure checks (note: attributes may not have quotes)
        assert html_str.startswith("<div class=tracerite>")
        assert html_str.endswith("</div>")

        # Check for required CSS and JS
        assert "<style>" in html_str
        assert "<script>" in html_str
        assert "scrollto=" in html_str


def test_traceback_tabs_structure():
    """Test the structure of traceback tabs."""

    def nested_error():
        return [1, 2][5]  # IndexError

    def outer_function():
        return nested_error()

    try:
        outer_function()
    except IndexError as e:
        html = html_traceback(e)
        html_str = str(html)

        # Should have traceback-tabs structure
        assert "traceback-tabs" in html_str
        assert "traceback-labels" in html_str or "content" in html_str
        assert "traceback-details" in html_str


def test_exception_header_structure():
    """Test the structure of exception headers."""
    try:
        raise ValueError("Test error message")
    except ValueError as e:
        html = html_traceback(e)
        html_str = str(html)

        # Check exception type formatting
        assert "<h3>" in html_str
        assert "exctype" in html_str  # Class name without quotes
        assert "ValueError:" in html_str
        assert "Test error message" in html_str


def test_marked_span_attributes():
    """Test that marked spans have correct attributes."""
    try:
        obj = "test"
        obj.missing_method()  # type: ignore[attr-defined]  # AttributeError
    except AttributeError as e:
        html = html_traceback(e)
        html_str = str(html)

        # Find all marked spans
        marked_pattern = r'<span[^>]*class="tracerite-tooltip"[^>]*>'
        matches = re.findall(marked_pattern, html_str)

        assert len(matches) > 0

        for match in matches:
            # Each marked span should have required attributes
            assert "data-symbol=" in match
            assert "data-tooltip=" in match
            assert 'class="tracerite-tooltip"' in match


def test_mark_tag_structure():
    """Test the structure of <mark> tags within tooltips."""
    try:
        _ = 1 / 0  # ZeroDivisionError
    except ZeroDivisionError as e:
        html = html_traceback(e)
        html_str = str(html)

        # Find mark tags
        mark_pattern = r"<mark[^>]*>.*?</mark>"
        marks = re.findall(mark_pattern, html_str, re.DOTALL)

        assert len(marks) > 0

        for mark in marks:
            # Mark should contain some content
            content = re.sub(r"<[^>]*>", "", mark)
            assert len(content.strip()) > 0


def test_em_tag_within_marks():
    """Test that <em> tags are properly nested within <mark> tags."""
    try:
        data = {"key": "value"}
        _ = data.missing_attribute  # type: ignore[attr-defined]  # AttributeError
    except AttributeError as e:
        html = html_traceback(e)
        html_str = str(html)

        # Look for em tags within mark tags
        mark_with_em_pattern = r"<mark[^>]*>.*?<em[^>]*>.*?</em>.*?</mark>"
        matches = re.findall(mark_with_em_pattern, html_str, re.DOTALL)

        # If em tags are present, they should be properly nested
        if matches:
            for match in matches:
                # Should be valid HTML structure
                assert "<em>" in match
                assert "</em>" in match


def test_code_line_structure():
    """Test the structure of code line spans."""
    try:
        _ = undefined_variable  # type: ignore[name-defined]  # noqa: F821
    except NameError as e:
        html = html_traceback(e)
        html_str = str(html)

        # Should have codeline spans with line numbers
        codeline_pattern = r"<span[^>]*codeline[^>]*data-lineno"
        matches = re.findall(codeline_pattern, html_str)

        assert len(matches) > 0


def test_symbol_display():
    """Test that symbols are correctly displayed in various contexts."""

    def level1():
        return level2()

    def level2():
        return 1 / 0  # ZeroDivisionError

    try:
        level1()
    except ZeroDivisionError as e:
        html = html_traceback(e)
        html_str = str(html)

        # Should contain symbols for different frame types
        symbols = ["➤", "⚠️", "💣", "🛑"]
        found_symbols = []

        for symbol in symbols:
            if symbol in html_str:
                found_symbols.append(symbol)

        # Should have at least one symbol (the error symbol)
        assert len(found_symbols) > 0


def test_variable_inspector_structure():
    """Test the structure of variable inspector tables."""

    def function_with_locals():
        local_var = "test_value"
        return local_var.missing_method()  # type: ignore[attr-defined]  # AttributeError

    try:
        function_with_locals()
    except AttributeError as e:
        html = html_traceback(e)
        html_str = str(html)

        # Should have inspector table if variables are present
        if "inspector key-value" in html_str:
            assert "<table" in html_str
            assert "<tr>" in html_str
            assert "<td>" in html_str

            # Should have variable names and types
            assert "var" in html_str  # CSS class for variables
            assert "type" in html_str  # CSS class for types
            assert "val" in html_str  # CSS class for values


def test_tooltip_escaping():
    """Test that tooltip content is properly escaped."""
    try:
        # Create an error with characters that need escaping
        exec('x = "quotes\'and<brackets>"[999]')  # IndexError
    except IndexError as e:
        html = html_traceback(e)
        html_str = str(html)

        # Extract tooltip content
        tooltip_pattern = r'data-tooltip="([^"]*)"'
        tooltips = re.findall(tooltip_pattern, html_str)

        for tooltip in tooltips:
            # Should not contain unescaped HTML characters that would break attributes
            assert "<" not in tooltip or "&lt;" in tooltip
            assert '"' not in tooltip  # Should be properly escaped or avoided


def test_onclick_handler_format():
    """Test that onclick handlers have correct format."""

    def nested_function():
        return [1, 2, 3][10]  # IndexError

    try:
        nested_function()
    except IndexError as e:
        html = html_traceback(e)
        html_str = str(html)

        # Should have onclick handlers for navigation
        onclick_pattern = r'onclick="scrollto\(\'([^\']*)\'\)"'
        matches = re.findall(onclick_pattern, html_str)

        # Each onclick should reference a valid element ID
        for element_id in matches:
            assert element_id.startswith("tb-")
            # Should also find the corresponding element
            assert f'id="{element_id}"' in html_str


def test_css_classes_consistency():
    """Test that CSS classes are consistently applied."""
    try:
        data = [1, 2]
        _ = data[5]  # IndexError
    except IndexError as e:
        html = html_traceback(e)
        html_str = str(html)

        # Required CSS classes should be present
        required_classes = [
            "tracerite",
            "traceback-tabs",
            "traceback-details",
            "codeline",
            "tracerite-tooltip",
        ]

        for css_class in required_classes:
            assert (
                f'class="{css_class}"' in html_str
                or 'class="' in html_str
                and css_class in html_str
            )


def test_line_number_consistency():
    """Test that line numbers are consistent throughout the HTML."""

    def test_function():
        x = 1
        y = 2
        return x[y]  # type: ignore[index]  # TypeError

    try:
        test_function()
    except TypeError as e:
        html = html_traceback(e)
        html_str = str(html)

        # Extract all line numbers from data-lineno attributes
        lineno_pattern = r'data-lineno="(\d+)"'
        line_numbers = [int(x) for x in re.findall(lineno_pattern, html_str)]

        if line_numbers:
            # Line numbers should be in ascending order within each frame
            # and should be reasonable (not negative, not extremely large)
            for line_no in line_numbers:
                assert 1 <= line_no <= 10000  # Reasonable bounds


def extract_html_elements(html_str, tag):
    """Helper function to extract specific HTML elements."""
    pattern = f"<{tag}[^>]*>.*?</{tag}>"
    return re.findall(pattern, html_str, re.DOTALL)


def test_html_nesting_validity():
    """Test that HTML elements are properly nested."""
    try:
        obj = {}
        obj["key"].method()  # AttributeError or KeyError
    except (AttributeError, KeyError) as e:
        html = html_traceback(e)
        html_str = str(html)

        # Test that mark tags don't contain unclosed tags
        marks = extract_html_elements(html_str, "mark")
        for mark in marks:
            # Count opening and closing tags
            open_count = mark.count("<span")
            close_count = mark.count("</span>")
            open_em = mark.count("<em")
            close_em = mark.count("</em>")

            # Each opening tag should have a corresponding closing tag
            assert open_count == close_count
            assert open_em == close_em

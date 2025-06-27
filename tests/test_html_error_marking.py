"""Tests for HTML error marking and highlighting in tracerite."""

import re

from tracerite import extract_chain, html_traceback


def test_simple_name_error():
    """Test HTML output for NameError with variable highlighting."""
    try:
        _ = undefined_variable  # type: ignore[name-defined]  # This will cause a NameError
    except NameError as e:
        html = html_traceback(e)
        html_str = str(html)

        # Check that the error line is marked
        assert 'class="tracerite-tooltip"' in html_str
        assert "undefined_variable" in html_str
        assert "<mark>" in html_str

        # Extract the marked content and check for the variable name
        mark_pattern = r"<mark[^>]*>(.*?)</mark>"
        marks = re.findall(mark_pattern, html_str, re.DOTALL)
        assert len(marks) > 0

        # Check that the variable name appears in at least one mark (may include HTML tags)
        found_variable = False
        for mark in marks:
            # Remove HTML tags to get text content
            text_content = re.sub(r"<[^>]*>", "", mark)
            if "undefined_variable" in text_content:
                found_variable = True
                break
        assert found_variable, f"Variable name not found in marks: {marks}"


def test_attribute_error():
    """Test HTML output for AttributeError with attribute highlighting."""
    try:
        obj = "string"
        obj.nonexistent_method()  # type: ignore[attr-defined]  # This will cause an AttributeError
    except AttributeError as e:
        html = html_traceback(e)
        html_str = str(html)

        # Check that the error line is marked with attribute highlighting
        assert 'class="tracerite-tooltip"' in html_str
        assert "nonexistent_method" in html_str
        assert "<mark>" in html_str


def test_type_error_function_call():
    """Test HTML output for TypeError in function call with function name highlighting."""
    try:
        int("not_a_number", "invalid_base")  # type: ignore[call-overload]  # This will cause a TypeError
    except TypeError as e:
        html = html_traceback(e)
        html_str = str(html)

        # Check that the error line is marked
        assert 'class="tracerite-tooltip"' in html_str
        assert "<mark>" in html_str
        assert "int(" in html_str


def test_index_error():
    """Test HTML output for IndexError with subscript highlighting."""
    try:
        lst = [1, 2, 3]
        lst[10]  # This will cause an IndexError
    except IndexError as e:
        html = html_traceback(e)
        html_str = str(html)

        # Check that the error line is marked
        assert 'class="tracerite-tooltip"' in html_str
        assert "<mark>" in html_str
        assert "lst[10]" in html_str


def test_key_error():
    """Test HTML output for KeyError with subscript highlighting."""
    try:
        d = {"a": 1, "b": 2}
        d["nonexistent"]  # This will cause a KeyError
    except KeyError as e:
        html = html_traceback(e)
        html_str = str(html)

        # Check that the error line is marked
        assert 'class="tracerite-tooltip"' in html_str
        assert "<mark>" in html_str


def test_zero_division_error():
    """Test HTML output for ZeroDivisionError with operator highlighting."""
    try:
        _ = 1 / 0  # This will cause a ZeroDivisionError
    except ZeroDivisionError as e:
        html = html_traceback(e)
        html_str = str(html)

        # Check that the error line is marked
        assert 'class="tracerite-tooltip"' in html_str
        assert "<mark>" in html_str
        assert "1 / 0" in html_str


def test_comparison_error():
    """Test HTML output for TypeError in comparison with operator highlighting."""
    try:
        _ = "string" < 42  # type: ignore[operator]  # This will cause a TypeError
    except TypeError as e:
        html = html_traceback(e)
        html_str = str(html)

        # Check that the error line is marked
        assert 'class="tracerite-tooltip"' in html_str
        assert "<mark>" in html_str


def test_nested_function_error():
    """Test HTML output for error in nested function calls."""

    def inner_function():
        _ = undefined_var  # type: ignore[name-defined]  # This will cause a NameError  # noqa: F821

    def outer_function():
        inner_function()

    try:
        outer_function()
    except NameError as e:
        html = html_traceback(e)
        html_str = str(html)

        # Check that multiple frames are present
        assert "outer_function" in html_str
        assert "inner_function" in html_str

        # Check that the error line in inner_function is marked
        assert 'class="tracerite-tooltip"' in html_str
        assert "<mark>" in html_str
        assert "undefined_var" in html_str


def test_chained_exceptions():
    """Test HTML output for chained exceptions."""
    try:
        try:
            lst = [1, 2, 3]
            lst[10]  # IndexError
        except IndexError as inner:
            raise ValueError("Wrapped error") from inner
    except ValueError as e:
        html = html_traceback(e)
        html_str = str(html)

        # Check that both exceptions are present
        assert "IndexError" in html_str
        assert "ValueError" in html_str
        assert "Wrapped error" in html_str

        # Check that error lines are marked (at least one mark should be present)
        mark_count = html_str.count("<mark>")
        assert mark_count >= 1  # At least one mark should be present

        # Check that the specific error content is marked
        mark_pattern = r"<mark[^>]*>(.*?)</mark>"
        marks = re.findall(mark_pattern, html_str, re.DOTALL)
        marked_content = "".join(marks)
        assert "lst[10]" in marked_content  # The actual error should be marked


def test_syntax_error_like_construct():
    """Test HTML output for errors in complex expressions."""
    try:
        # Create a complex expression that will fail
        data = {"items": [1, 2, 3]}
        _ = data["items"][5].nonexistent  # type: ignore[attr-defined]  # Multiple potential error points
    except (IndexError, AttributeError) as e:
        html = html_traceback(e)
        html_str = str(html)

        # Check that the error line is marked
        assert 'class="tracerite-tooltip"' in html_str
        assert "<mark>" in html_str


def test_multiple_frames_with_different_relevance():
    """Test that different frame relevances get marked appropriately."""

    def level3():
        return 1 / 0  # ZeroDivisionError

    def level2():
        return level3()

    def level1():
        return level2()

    try:
        level1()
    except ZeroDivisionError as e:
        chain = extract_chain(e)
        assert len(chain) > 0

        frames = chain[0]["frames"]
        assert len(frames) >= 3

        # Check that different relevance levels are assigned
        relevances = [frame["relevance"] for frame in frames]
        assert "call" in relevances  # Should have call frames
        assert "error" in relevances or "stop" in relevances  # Should have error frame

        html = html_traceback(e)
        html_str = str(html)

        # Check that the error frame is marked
        assert 'class="tracerite-tooltip"' in html_str
        assert "<mark>" in html_str
        assert "1 / 0" in html_str


def test_tooltip_and_symbol_attributes():
    """Test that tooltip and symbol attributes are correctly set."""
    try:
        _ = undefined_variable  # type: ignore[name-defined]  # noqa: F821
    except NameError as e:
        html = html_traceback(e)
        html_str = str(html)

        # Check for tooltip attributes
        assert "data-tooltip=" in html_str
        assert "data-symbol=" in html_str

        # Verify tooltip contains error information
        tooltip_pattern = r'data-tooltip="([^"]*)"'
        tooltips = re.findall(tooltip_pattern, html_str)
        assert len(tooltips) > 0

        # Should contain exception type or other relevant info
        found_relevant_tooltip = False
        for tooltip in tooltips:
            if any(
                keyword in tooltip.lower()
                for keyword in ["exception", "error", "raised", "nameerror"]
            ):
                found_relevant_tooltip = True
                break
        assert found_relevant_tooltip


def test_precise_column_highlighting():
    """Test that column information is used for precise highlighting."""
    try:
        # Create an expression where we want precise column highlighting
        data = [1, 2, 3]
        _ = data[10]  # IndexError at specific position
    except IndexError as e:
        chain = extract_chain(e)
        frames = chain[0]["frames"]
        error_frame = frames[-1]  # Last frame should be the error

        # Check if column information is available (Python 3.11+)
        if "colno" in error_frame:
            assert "end_colno" in error_frame
            html = html_traceback(e)
            html_str = str(html)

            # Should have precise highlighting
            assert "<mark>" in html_str
            assert 'class="tracerite-tooltip"' in html_str


def test_highlight_info_structure():
    """Test that highlight_info contains the expected structure."""
    try:
        obj = "test"
        obj.missing_method()  # type: ignore[attr-defined]  # AttributeError
    except AttributeError as e:
        chain = extract_chain(e)
        frames = chain[0]["frames"]
        error_frame = frames[-1]

        # Check if highlight_info is present and properly structured
        if "highlight_info" in error_frame:
            highlight_info = error_frame["highlight_info"]
            assert "type" in highlight_info
            assert highlight_info["type"] in ["caret", "range", "ranges"]

            if highlight_info["type"] == "caret":
                assert "offset" in highlight_info
            elif highlight_info["type"] == "range":
                assert "start" in highlight_info
                assert "end" in highlight_info


def test_unary_operator_error():
    """Test HTML output for error in unary operation."""
    try:
        _ = -"string"  # type: ignore[operator]  # TypeError with unary operator
    except TypeError as e:
        html = html_traceback(e)
        html_str = str(html)

        # Check that the error line is marked
        assert 'class="tracerite-tooltip"' in html_str
        assert "<mark>" in html_str


def test_binary_operation_error():
    """Test HTML output for error in binary operation."""
    try:
        _ = "string" + 42  # type: ignore[operator]  # TypeError with binary operator
    except TypeError as e:
        html = html_traceback(e)
        html_str = str(html)

        # Check that the error line is marked
        assert 'class="tracerite-tooltip"' in html_str
        assert "<mark>" in html_str


def test_em_tag_for_specific_highlighting():
    """Test that <em> tags are used for specific parts within marked regions."""
    try:
        obj = "test"
        obj.missing_method()  # type: ignore[attr-defined]  # AttributeError
    except AttributeError as e:
        html = html_traceback(e)
        html_str = str(html)

        # Look for <em> tags within <mark> regions
        mark_with_em_pattern = r"<mark[^>]*>.*?<em[^>]*>.*?</em>.*?</mark>"
        if re.search(mark_with_em_pattern, html_str, re.DOTALL):
            # If <em> is used, it should be highlighting a specific part
            em_pattern = r"<em[^>]*>(.*?)</em>"
            em_contents = re.findall(em_pattern, html_str)
            assert len(em_contents) > 0


def test_fallback_highlighting():
    """Test fallback highlighting when precise column info is not available."""
    try:
        # Create an error where column info might not be precise
        exec("_ = undefined_var")  # NameError through exec  # noqa: F821
    except NameError as e:
        html = html_traceback(e)
        html_str = str(html)

        # Should still have some form of highlighting
        assert 'class="tracerite-tooltip"' in html_str
        # May not have <mark> if fallback to full line highlighting
        assert "data-tooltip=" in html_str


def test_multi_line_error_context():
    """Test HTML output when error spans multiple lines or has context."""

    def complex_function():
        data = {"key1": "value1", "key2": "value2"}
        # Error on a different line than definition
        return data["nonexistent_key"]

    try:
        complex_function()
    except KeyError as e:
        html = html_traceback(e)
        html_str = str(html)

        # Should show context lines
        assert "data = {" in html_str or '"key1"' in html_str
        # Error line should be marked
        assert 'class="tracerite-tooltip"' in html_str
        assert "<mark>" in html_str


def extract_marked_content(html_str):
    """Helper function to extract content within <mark> tags."""
    mark_pattern = r"<mark[^>]*>(.*?)</mark>"
    marks = re.findall(mark_pattern, html_str, re.DOTALL)
    # Remove HTML tags from marked content to get text
    text_marks = []
    for mark in marks:
        text_content = re.sub(r"<[^>]*>", "", mark)
        text_marks.append(text_content)
    return text_marks


def test_marked_content_accuracy():
    """Test that the content within <mark> tags is accurate and relevant."""
    test_cases = [
        (lambda: exec("_ = undefined_var"), "undefined_var"),  # noqa: F821
        (lambda: [1, 2][5], "[5]"),
        (lambda: {"a": 1}["b"], '["b"]'),
        (lambda: "str".nonexistent(), "nonexistent"),  # type: ignore[attr-defined]
        (lambda: 1 / 0, "/"),
    ]

    for error_func, expected_in_mark in test_cases:
        try:
            error_func()
        except Exception as e:
            html = html_traceback(e)
            html_str = str(html)
            marked_contents = extract_marked_content(html_str)

            # Check that expected content appears in at least one mark
            found = any(expected_in_mark in marked for marked in marked_contents)
            assert found, (
                f"Expected '{expected_in_mark}' not found in marked content: {marked_contents}"
            )

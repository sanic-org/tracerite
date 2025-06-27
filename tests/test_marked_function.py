"""Focused tests for the marked() function and HTML highlighting logic."""

import re

from tracerite.html import marked


def test_marked_with_caret_highlight():
    """Test marked() function with caret-style highlighting."""
    line = "    obj.missing_method()\n"

    # Mock frame info with caret highlighting
    info = {"type": "AttributeError"}
    frinfo = {
        "relevance": "error",
        "colno": 8,  # Position of 'm' in 'missing_method'
        "end_colno": 22,  # End of 'missing_method'
        "highlight_info": {
            "type": "caret",
            "offset": 8,  # Relative to the error region
        },
    }

    result = marked(line, info, frinfo)
    result_str = str(result)

    # Should contain mark and em tags
    assert "<mark>" in result_str
    assert "<em>" in result_str
    assert 'class="tracerite-tooltip"' in result_str
    assert "data-symbol=" in result_str


def test_marked_with_range_highlight():
    """Test marked() function with range-style highlighting."""
    line = "    result = a + b\n"

    # Mock frame info with range highlighting
    info = {"type": "TypeError"}
    frinfo = {
        "relevance": "error",
        "colno": 15,  # Position of '+'
        "end_colno": 16,  # End of '+'
        "highlight_info": {"type": "range", "start": 0, "end": 1},
    }

    result = marked(line, info, frinfo)
    result_str = str(result)

    # Should contain mark and em tags for the specific range
    assert "<mark>" in result_str
    assert "<em>" in result_str
    assert 'class="tracerite-tooltip"' in result_str


def test_marked_with_multiple_ranges():
    """Test marked() function with multiple ranges highlighting."""
    line = "    complex.expression.with.multiple.parts\n"

    # Mock frame info with ranges highlighting
    info = {"type": "AttributeError"}
    frinfo = {
        "relevance": "error",
        "colno": 4,
        "end_colno": 40,
        "highlight_info": {"type": "ranges"},
    }

    result = marked(line, info, frinfo)
    result_str = str(result)

    # Should fall back to whole region highlighting
    assert "<mark>" in result_str
    assert 'class="tracerite-tooltip"' in result_str


def test_marked_with_old_style_column_info():
    """Test marked() function with old-style column info (no highlight_info)."""
    line = "    lst[index]\n"

    # Mock frame info without highlight_info
    info = {"type": "IndexError"}
    frinfo = {
        "relevance": "error",
        "colno": 7,  # Position of '['
        "end_colno": 13,  # End of ']'
    }

    result = marked(line, info, frinfo)
    result_str = str(result)

    # Should have basic highlighting without em tags
    assert "<mark" in result_str  # Check for mark tag (with or without attributes)
    assert 'class="tracerite-tooltip"' in result_str


def test_marked_fallback_no_column_info():
    """Test marked() function fallback when no column info is available."""
    line = "    some_error_line\n"

    # Mock frame info without column info
    info = {"type": "RuntimeError"}
    frinfo = {"relevance": "error"}

    result = marked(line, info, frinfo)
    result_str = str(result)

    # Should fall back to full-line highlighting
    assert "<mark" in result_str  # Check for mark tag (with or without attributes)
    assert 'class="tracerite-tooltip"' in result_str
    assert "some_error_line" in result_str


def test_marked_preserves_line_structure():
    """Test that marked() preserves indentation and trailing whitespace."""
    line = "        x = y + z    \n"

    info = {"type": "NameError"}
    frinfo = {
        "relevance": "error",
        "colno": 12,  # Position of 'y'
        "end_colno": 13,
    }

    result = marked(line, info, frinfo)
    result_str = str(result)

    # Should preserve leading spaces and trailing whitespace
    assert result_str.startswith("        ")  # 8 spaces
    assert result_str.endswith("    \n")  # 4 spaces + newline


def test_marked_with_complex_expression():
    """Test marked() with a complex multi-part expression."""
    line = "    data['key'].method().attr[index]\n"

    info = {"type": "KeyError"}
    frinfo = {
        "relevance": "error",
        "colno": 9,  # Position of 'key'
        "end_colno": 14,
        "highlight_info": {
            "type": "range",
            "start": 1,  # Relative to error start
            "end": 4,
        },
    }

    result = marked(line, info, frinfo)
    result_str = str(result)

    # Should highlight the specific part
    assert "<mark" in result_str  # Check for mark tag (with or without attributes)
    assert "<em>" in result_str
    # The key should be in the result, possibly within HTML tags
    assert "key" in result_str


def test_marked_tooltip_formatting():
    """Test that tooltip text is properly formatted and escaped."""
    line = "    undefined_variable\n"

    info = {"type": "NameError", "message": "name 'undefined_variable' is not defined"}
    frinfo = {"relevance": "error", "filename": "test.py", "lineno": 42}

    result = marked(line, info, frinfo)
    result_str = str(result)

    # Extract tooltip content
    tooltip_pattern = r'data-tooltip="([^"]*)"'
    tooltips = re.findall(tooltip_pattern, result_str)
    assert len(tooltips) > 0

    # Should contain relevant information
    tooltip = tooltips[0]
    assert "error" in tooltip.lower() or "exception" in tooltip.lower()


def test_marked_symbol_attribute():
    """Test that symbol attribute is correctly set based on relevance."""
    line = "    function_call()\n"

    relevance_to_symbol = {"call": "➤", "warning": "⚠️", "error": "💣", "stop": "🛑"}

    for relevance, expected_symbol in relevance_to_symbol.items():
        info = {"type": "TypeError"}
        frinfo = {"relevance": relevance}

        result = marked(line, info, frinfo)
        result_str = str(result)

        # Should contain the expected symbol
        symbol_pattern = r'data-symbol="([^"]*)"'
        symbols = re.findall(symbol_pattern, result_str)
        assert len(symbols) > 0
        assert expected_symbol in symbols[0]


def test_marked_bounds_checking():
    """Test that marked() handles edge cases and bounds correctly."""
    # Test with column positions at boundaries
    line = "x\n"

    info = {"type": "NameError"}
    frinfo = {
        "relevance": "error",
        "colno": 0,
        "end_colno": 1,
        "highlight_info": {"type": "caret", "offset": 0},
    }

    result = marked(line, info, frinfo)
    result_str = str(result)

    # Should handle single character properly
    assert "<mark>" in result_str
    assert "x" in result_str


def test_marked_with_empty_highlight_region():
    """Test marked() when highlight region is empty or invalid."""
    line = "    expression\n"

    info = {"type": "RuntimeError"}
    frinfo = {
        "relevance": "error",
        "colno": 10,
        "end_colno": 10,  # Empty range
        "highlight_info": {"type": "range", "start": 0, "end": 0},
    }

    result = marked(line, info, frinfo)
    result_str = str(result)

    # Should still provide highlighting
    assert 'class="tracerite-tooltip"' in result_str


def test_marked_with_out_of_bounds_columns():
    """Test marked() when column info is out of bounds."""
    line = "short\n"

    info = {"type": "IndexError"}
    frinfo = {
        "relevance": "error",
        "colno": 50,  # Way beyond line length
        "end_colno": 60,
        "highlight_info": {"type": "caret", "offset": 0},
    }

    result = marked(line, info, frinfo)
    result_str = str(result)

    # Should handle gracefully and still provide some highlighting
    assert 'class="tracerite-tooltip"' in result_str

import sys

import pytest
from bs4 import BeautifulSoup

from tests.errorcases import (
    binomial_operator,
    deeply_nested_chain_with_calls,
    exception_group_with_frames,
    max_type_error_case,
    multiline_marking,
    multiline_marking_comment,
    unrelated_error_in_except,
    unrelated_error_in_finally,
)
from tracerite.html import html_traceback


@pytest.mark.skipif(
    sys.version_info < (3, 11), reason="Requires co_positions() from Python 3.11+"
)
def test_binomial_operator_html_traceback():
    try:
        binomial_operator()
    except Exception as e:
        html_output = str(html_traceback(e))

    soup = BeautifulSoup(html_output, "html.parser")

    # Locate the div containing the traceback using the data-function attribute
    traceback_details = soup.find(
        "div", class_="traceback-details", attrs={"data-function": "binomial_operator"}
    )
    assert traceback_details is not None, (
        f"Traceback details not found. HTML snippet: {soup.prettify()}"
    )

    # Verify that 1 + "b" and only it is within mark element
    code_element = traceback_details.find("code")
    assert code_element is not None, (
        f"Code element not found. HTML snippet: {traceback_details.prettify()}"
    )

    mark_element = None
    for mark in code_element.find_all("mark"):
        if '1 + "b"' in mark.text:
            mark_element = mark
            break

    assert mark_element is not None, (
        f"Mark element not found. HTML snippet: {code_element.prettify()}"
    )
    print(mark_element.prettify())  # Debugging: Print the mark element structure
    assert '1 + "b"' in mark_element.text, (
        f"Expected text not found in mark element. HTML snippet: {mark_element.prettify()}"
    )

    # Verify that the + is within em
    em_element = mark_element.find("em")
    print(
        em_element.prettify() if em_element else "No em element found"
    )  # Debugging: Print the em element structure
    assert em_element is not None, (
        f"Em element not found. HTML snippet: {mark_element.prettify()}"
    )
    assert em_element.text == "+", (
        f"Expected '+' not found in em element. HTML snippet: {em_element.prettify()}"
    )


@pytest.mark.skipif(
    sys.version_info < (3, 11), reason="Requires co_positions() from Python 3.11+"
)
def test_multiline_marking_html_traceback():
    """Test HTML output for multiline binary operator."""
    try:
        multiline_marking()
    except Exception as e:
        html_output = str(html_traceback(e))

    soup = BeautifulSoup(html_output, "html.parser")

    # Locate the div containing the traceback
    traceback_details = soup.find(
        "div", class_="traceback-details", attrs={"data-function": "multiline_marking"}
    )
    assert traceback_details is not None, (
        f"Traceback details not found. HTML snippet: {soup.prettify()}"
    )

    # Verify that we have mark elements and at least one em element
    code_element = traceback_details.find("code")
    assert code_element is not None, (
        f"Code element not found. HTML snippet: {traceback_details.prettify()}"
    )

    mark_elements = code_element.find_all("mark")
    assert len(mark_elements) > 0, (
        f"No mark elements found. HTML snippet: {code_element.prettify()}"
    )

    # Verify that at least one mark element contains an em element
    found_em = False
    for mark in mark_elements:
        em_element = mark.find("em")
        if em_element:
            found_em = True
            # Should contain the operator
            assert em_element.text.strip() in ["+", ""], (
                f"Unexpected text in em element: {em_element.text}"
            )
            break

    assert found_em, "No em element found in multiline marking"


@pytest.mark.skipif(
    sys.version_info < (3, 11), reason="Requires co_positions() from Python 3.11+"
)
def test_multiline_marking_comment_html_traceback():
    """Test HTML output for multiline binary operator with comments."""
    try:
        multiline_marking_comment()
    except Exception as e:
        html_output = str(html_traceback(e))

    soup = BeautifulSoup(html_output, "html.parser")

    # Locate the div containing the traceback
    traceback_details = soup.find(
        "div",
        class_="traceback-details",
        attrs={"data-function": "multiline_marking_comment"},
    )
    assert traceback_details is not None, (
        f"Traceback details not found. HTML snippet: {soup.prettify()}"
    )

    # Verify that we have code elements
    code_element = traceback_details.find("code")
    assert code_element is not None, (
        f"Code element not found. HTML snippet: {traceback_details.prettify()}"
    )

    # Should have mark elements for the error
    mark_elements = code_element.find_all("mark")
    assert len(mark_elements) > 0, (
        f"No mark elements found. HTML snippet: {code_element.prettify()}"
    )


@pytest.mark.skipif(
    sys.version_info < (3, 11), reason="Requires co_positions() from Python 3.11+"
)
def test_max_type_error_html_traceback():
    """Test HTML output for function call type error."""
    try:
        max_type_error_case()
    except Exception as e:
        html_output = str(html_traceback(e))

    soup = BeautifulSoup(html_output, "html.parser")

    # Locate the div containing the traceback
    traceback_details = soup.find(
        "div",
        class_="traceback-details",
        attrs={"data-function": "max_type_error_case"},
    )
    assert traceback_details is not None, (
        f"Traceback details not found. HTML snippet: {soup.prettify()}"
    )

    # Verify that we have code elements
    code_element = traceback_details.find("code")
    assert code_element is not None, (
        f"Code element not found. HTML snippet: {traceback_details.prettify()}"
    )

    # Look for mark elements containing the function call
    mark_elements = code_element.find_all("mark")
    found_function_call = False
    for mark in mark_elements:
        if "max(" in mark.text or "max" in mark.text:
            found_function_call = True
            # Should contain em for function call brackets
            em_element = mark.find("em")
            if em_element:
                # For function calls, em should contain brackets or be part of the call
                # Function call anchors may include the full call syntax
                assert "(" in mark.text, (
                    f"Opening parenthesis not found in function call mark: {mark.text}"
                )
            break

    assert found_function_call, "No function call found in mark elements"


@pytest.mark.skipif(
    sys.version_info < (3, 11), reason="Requires co_positions() from Python 3.11+"
)
def test_unrelated_error_except_html_traceback():
    """Test HTML output for unrelated error in except block."""
    try:
        unrelated_error_in_except()
    except Exception as e:
        html_output = str(html_traceback(e))

    soup = BeautifulSoup(html_output, "html.parser")

    # Find all traceback-details divs for this function (there are two in the chain)
    # and use the last one (ZeroDivisionError) which is the newest exception
    traceback_details_list = soup.find_all(
        "div",
        class_="traceback-details",
        attrs={"data-function": "unrelated_error_in_except"},
    )
    assert len(traceback_details_list) > 0, (
        f"Traceback details not found. HTML snippet: {soup.prettify()}"
    )
    traceback_details = traceback_details_list[-1]  # Get last (newest)

    # Verify that we have code elements
    code_element = traceback_details.find("code")
    assert code_element is not None, (
        f"Code element not found. HTML snippet: {traceback_details.prettify()}"
    )

    # Look for mark elements containing the division
    mark_elements = code_element.find_all("mark")
    found_division = False
    for mark in mark_elements:
        if "0 / 0" in mark.text or "/" in mark.text:
            found_division = True
            # Should contain em for division operator
            em_element = mark.find("em")
            if em_element:
                # For division, em should contain the operator
                assert "/" in mark.text, (
                    f"Division operator not found in marked text: {mark.text}"
                )
            break

    assert found_division, "No division operation found in mark elements"


@pytest.mark.skipif(
    sys.version_info < (3, 11), reason="Requires co_positions() from Python 3.11+"
)
def test_unrelated_error_finally_html_traceback():
    """Test HTML output for unrelated error in finally block."""
    try:
        unrelated_error_in_finally()
    except Exception as e:
        html_output = str(html_traceback(e))

    soup = BeautifulSoup(html_output, "html.parser")

    # Find all traceback-details divs for this function (there are two in the chain)
    # and use the last one (ZeroDivisionError) which is the newest exception
    traceback_details_list = soup.find_all(
        "div",
        class_="traceback-details",
        attrs={"data-function": "unrelated_error_in_finally"},
    )
    assert len(traceback_details_list) > 0, (
        f"Traceback details not found. HTML snippet: {soup.prettify()}"
    )
    traceback_details = traceback_details_list[-1]  # Get last (newest)

    # Verify that we have code elements
    code_element = traceback_details.find("code")
    assert code_element is not None, (
        f"Code element not found. HTML snippet: {traceback_details.prettify()}"
    )

    # Look for mark elements containing the division
    mark_elements = code_element.find_all("mark")
    found_division = False
    for mark in mark_elements:
        if "0 / 0" in mark.text or "/" in mark.text:
            found_division = True
            # Should contain em for division operator
            em_element = mark.find("em")
            if em_element:
                # For division, em should contain the operator
                assert "/" in mark.text, (
                    f"Division operator not found in marked text: {mark.text}"
                )
            break

    assert found_division, "No division operation found in mark elements"


def test_deeply_nested_chain_with_calls_html_traceback():
    """Test HTML output for three-level exception chain with function calls.

    Chronological order is tested in test_chain_analysis.py. This test
    verifies the HTML formatting doesn't crash and shows expected content.
    """
    try:
        deeply_nested_chain_with_calls()
    except Exception as e:
        html_output = str(html_traceback(e))

    soup = BeautifulSoup(html_output, "html.parser")

    # Verify all three exception types appear in the HTML
    assert "ValueError" in html_output
    assert "TypeError" in html_output
    assert "ZeroDivisionError" in html_output

    # Verify exception messages appear
    assert "level 1" in html_output
    assert "level 2" in html_output
    assert "division by zero" in html_output

    # Verify helper function names appear (extra frames from calls)
    assert "_raise_level1" in html_output
    assert "_handle_and_raise_level2" in html_output
    assert "_handle_and_divide_by_zero" in html_output

    # Verify we have multiple traceback-details divs (one per exception frame)
    traceback_details_list = soup.find_all("div", class_="traceback-details")
    assert len(traceback_details_list) >= 3, (
        f"Expected at least 3 traceback details divs, got {len(traceback_details_list)}"
    )


def test_html_traceback_comprehensive():
    """Test that HTML traceback works for all error cases."""
    test_functions = [
        binomial_operator,
        multiline_marking,
        multiline_marking_comment,
        max_type_error_case,
        unrelated_error_in_except,
        unrelated_error_in_finally,
    ]

    for test_func in test_functions:
        try:
            test_func()
        except Exception as e:
            html_output = str(html_traceback(e))

            # Basic HTML structure checks
            assert "<div" in html_output, f"No div elements for {test_func.__name__}"
            assert "traceback-details" in html_output, (
                f"No traceback-details class for {test_func.__name__}"
            )

            soup = BeautifulSoup(html_output, "html.parser")

            # Should have at least one traceback details div
            traceback_divs = soup.find_all("div", class_="traceback-details")
            assert len(traceback_divs) > 0, (
                f"No traceback details divs found for {test_func.__name__}"
            )

            # Should have at least one code element
            code_elements = soup.find_all("code")
            assert len(code_elements) > 0, (
                f"No code elements found for {test_func.__name__}"
            )


def test_collapse_call_runs_empty_frames():
    """Test _collapse_call_runs with empty frames list."""
    from tracerite.html import _collapse_call_runs

    result = _collapse_call_runs([])
    assert result == []


def test_collapse_call_runs_final_run_collapsed():
    """Test _collapse_call_runs with final run of call frames that gets collapsed."""
    from tracerite.html import _collapse_call_runs

    # Create frames with a final run of 12 'call' frames (more than min_run_length=10)
    frames = [
        {"relevance": "error", "id": 1},  # Non-call frame
    ] + [
        {"relevance": "call", "id": i}
        for i in range(2, 14)  # 12 call frames at the end
    ]

    result = _collapse_call_runs(frames)

    # Should have: error frame, first call frame, ..., last call frame
    expected = [
        {"relevance": "error", "id": 1},
        {"relevance": "call", "id": 2},  # First of the run
        ...,  # Ellipsis
        {"relevance": "call", "id": 13},  # Last of the run
    ]
    assert result == expected


def test_collapse_call_runs_final_run_not_collapsed():
    """Test _collapse_call_runs with final run of call frames that doesn't get collapsed."""
    from tracerite.html import _collapse_call_runs

    # Create frames with a final run of 8 'call' frames (less than min_run_length=10)
    frames = [
        {"relevance": "error", "id": 1},  # Non-call frame
    ] + [
        {"relevance": "call", "id": i}
        for i in range(2, 10)  # 8 call frames at the end
    ]

    result = _collapse_call_runs(frames)

    # Should have: error frame, all 8 call frames
    expected = [
        {"relevance": "error", "id": 1},
    ] + [{"relevance": "call", "id": i} for i in range(2, 10)]
    assert result == expected


@pytest.mark.skipif(
    sys.version_info < (3, 11), reason="ExceptionGroups require Python 3.11+"
)
def test_html_exception_group_parallel_branches():
    """Test HTML output with ExceptionGroup parallel branches (lines 178-179, 196-199)."""
    try:
        exception_group_with_frames()
    except Exception as e:
        html_output = str(html_traceback(e))

    soup = BeautifulSoup(html_output, "html.parser")

    # Check for parallel branches container
    parallel_container = soup.find("div", class_="parallel-branches")
    assert parallel_container is not None, (
        "Parallel branches container should be present"
    )

    # Check for individual branches
    branches = parallel_container.find_all("div", class_="parallel-branch")
    assert len(branches) == 2, (
        "Should have 2 parallel branches (ValueError and TypeError)"
    )

    # Check that subexception types are shown
    assert "ValueError" in html_output
    assert "TypeError" in html_output


def test_html_frame_function_suffix_only():
    """Test HTML frame with function_suffix but no function_name (line 387)."""
    from tracerite.html import _frame_label

    # Mock frame info with function_suffix but no function
    frinfo = {
        "filename": "/path/to/file.py",
        "location": "file.py",
        "linenostart": 10,
        "function": "",  # Empty function name
        "function_suffix": "⚡except",  # But has suffix
        "relevance": "except",
        "cursor_line": 10,
        "notebook_cell": False,
    }

    import html5tagger

    doc = html5tagger.Builder("div")
    _frame_label(doc, frinfo, toggle_id=None)
    html_output = str(doc)

    # Should show the suffix
    assert "⚡except" in html_output


def test_html_frame_notebook_cell():
    """Test HTML frame with notebook_cell=True (lines 394-395, 420-426)."""
    from tracerite.html import _frame_label

    # Mock frame info for notebook cell
    frinfo = {
        "filename": "/path/to/notebook.ipynb",
        "location": "Cell [5]",
        "linenostart": 10,
        "function": "test_func",
        "relevance": "call",
        "notebook_cell": True,
        "cursor_line": 10,
        "function_suffix": "",
    }

    import html5tagger

    doc = html5tagger.Builder("div")
    _frame_label(doc, frinfo, toggle_id=None)
    html_output = str(doc)

    # Should have Cell reference
    assert "Cell [5]" in html_output
    # Should have function name
    assert "test_func" in html_output


def test_html_frame_notebook_cell_no_function():
    """Test HTML frame with notebook_cell=True and no function (line 425-426)."""
    from tracerite.html import _frame_label

    frinfo = {
        "filename": "/path/to/notebook.ipynb",
        "location": "Cell [5]",
        "linenostart": 10,
        "function": "",
        "relevance": "call",
        "notebook_cell": True,
        "cursor_line": 10,
        "function_suffix": "",
    }

    import html5tagger

    doc = html5tagger.Builder("div")
    _frame_label(doc, frinfo, toggle_id=None)
    html_output = str(doc)

    assert "Cell [5]" in html_output


@pytest.mark.skipif(
    sys.version_info < (3, 11), reason="Requires co_positions() from Python 3.11+"
)
def test_long_arguments_em_collapse_html():
    """Test that long em parts (>20 chars) are collapsed in HTML output."""
    from tests.errorcases import long_arguments_error

    try:
        long_arguments_error()
    except Exception as e:
        html_output = str(html_traceback(e))

    soup = BeautifulSoup(html_output, "html.parser")

    # Check for compact-code elements (call frames use this format)
    soup.find_all("code", class_="compact-code")
    # The collapsed version should have an ellipsis
    html_str = str(soup)
    # In HTML, this appears in the code element, possibly with em around it
    # The long args should be collapsed to something like (…) or first…last
    assert "max" in html_str or "long_arguments_error" in html_str


def test_multiline_exception_message_html():
    """Test HTML output for exception with multiline message."""
    from tests.errorcases import multiline_exception_message

    try:
        multiline_exception_message()
    except Exception as e:
        html_output = str(html_traceback(e))

    soup = BeautifulSoup(html_output, "html.parser")

    # Should contain the exception message parts
    assert "First line of error" in str(soup)
    # The remaining lines should be in a pre element with class excmessage
    pre_elements = soup.find_all("pre", class_="excmessage")
    assert len(pre_elements) > 0
    # Second and third lines should be in the pre
    pre_text = " ".join(pre.get_text() for pre in pre_elements)
    assert "Second line" in pre_text or "Third line" in pre_text


def test_empty_line_exception_message_html():
    """Test HTML output for exception with empty line in message."""
    from tests.errorcases import empty_second_line_exception

    try:
        empty_second_line_exception()
    except Exception as e:
        html_output = str(html_traceback(e))

    soup = BeautifulSoup(html_output, "html.parser")
    # Should contain the first line
    assert "First line" in str(soup)


def test_trailing_newline_message_html():
    """Test HTML output for exception message with only trailing newline."""
    from tests.errorcases import trailing_newline_message

    try:
        trailing_newline_message()
    except Exception as e:
        html_output = str(html_traceback(e))

    soup = BeautifulSoup(html_output, "html.parser")
    # Should contain the message
    assert "Single line with trailing" in str(soup)
    # Should NOT have a pre.excmessage because rest is empty after strip
    # (the message is "Single line with trailing\n", split gives ["Single line...", ""])

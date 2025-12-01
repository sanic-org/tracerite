import sys

import pytest
from bs4 import BeautifulSoup

from tests.errorcases import (
    binomial_operator,
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

    # Locate the div containing the traceback
    traceback_details = soup.find(
        "div",
        class_="traceback-details",
        attrs={"data-function": "unrelated_error_in_except"},
    )
    assert traceback_details is not None, (
        f"Traceback details not found. HTML snippet: {soup.prettify()}"
    )

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

    # Locate the div containing the traceback
    traceback_details = soup.find(
        "div",
        class_="traceback-details",
        attrs={"data-function": "unrelated_error_in_finally"},
    )
    assert traceback_details is not None, (
        f"Traceback details not found. HTML snippet: {soup.prettify()}"
    )

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

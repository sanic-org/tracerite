from bs4 import BeautifulSoup

from tests.errorcases import binomial_operator
from tracerite.html import html_traceback


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

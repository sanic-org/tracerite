"""Tests for html_page() full-page template."""

from bs4 import BeautifulSoup, Tag
from html5tagger import HTML

from tracerite import html_page


def _sample_exception():
    try:
        raise ValueError("sample error")
    except Exception as exc:
        return exc


def _text(tag: Tag | None) -> str:
    assert tag is not None
    return tag.get_text()


def test_html_page_doctype_and_structure():
    html = html_page(_sample_exception())
    assert html.startswith("<!DOCTYPE html>")
    soup = BeautifulSoup(html, "html.parser")
    assert soup.html is not None
    assert soup.find("meta", charset="utf-8") is not None
    assert soup.title is not None
    assert soup.h1 is not None
    assert soup.main is not None


def test_html_page_title_from_exception():
    html = html_page(_sample_exception())
    soup = BeautifulSoup(html, "html.parser")
    assert _text(soup.title) == "ValueError"
    assert _text(soup.h1) == "ValueError"


def test_html_page_custom_title_and_heading():
    html = html_page(
        _sample_exception(),
        title="My App Error",
        heading="500 Server Error",
    )
    soup = BeautifulSoup(html, "html.parser")
    assert _text(soup.title) == "My App Error"
    assert _text(soup.h1) == "500 Server Error"


def test_html_page_default_ingress():
    html = html_page(_sample_exception())
    soup = BeautifulSoup(html, "html.parser")
    ingress = soup.find("p")
    assert ingress is not None
    assert "error occurred" in ingress.get_text()


def test_html_page_custom_ingress():
    html = html_page(
        _sample_exception(),
        heading="Oops",
        ingress="Something went wrong.",
    )
    soup = BeautifulSoup(html, "html.parser")
    assert _text(soup.h1) == "Oops"
    # Ingress appears inside <main>, before the traceback.
    main = soup.find("main")
    assert main is not None
    main_html = str(main)
    assert "Something went wrong." in main_html


def test_html_page_header_footer_override():
    html = html_page(
        _sample_exception(),
        header=HTML("<header><nav>App</nav><h1>Custom</h1></header>"),
        footer=HTML("<footer>Support</footer>"),
    )
    soup = BeautifulSoup(html, "html.parser")
    assert soup.header is not None
    nav = soup.header.find("nav")
    h1 = soup.header.find("h1")
    assert nav is not None and nav.get_text() == "App"
    assert h1 is not None and h1.get_text() == "Custom"
    assert _text(soup.footer) == "Support"


def test_html_page_default_has_no_header_footer_tags():
    """By default Page inserts Header/Footer slots without wrapper tags."""
    html = html_page(_sample_exception())
    assert "<header>" not in html
    assert "<footer>" not in html


def test_html_page_includes_tracerite_assets():
    html = html_page(_sample_exception())
    assert "--tracerite-ui-font" in html
    assert "tracerite.autodark" in html


def test_html_page_body_font():
    html = html_page(_sample_exception())
    assert ":root { color-scheme: light dark; font-family: system-ui, sans-serif }" in html


def test_html_page_traceback_embedded_without_fragment_assets():
    """The .tracerite fragment should not bring its own style/script tags."""
    html = html_page(_sample_exception())
    soup = BeautifulSoup(html, "html.parser")
    styles = soup.find_all("style")
    scripts = soup.find_all("script")
    # Page has exactly two styles (tracerite + page) and no script
    assert len(styles) == 2
    assert len(scripts) == 0
    tracerite_div = soup.find("div", class_="tracerite")
    assert tracerite_div is not None
    # The tracerite div itself contains no style/script children
    assert tracerite_div.find("style") is None
    assert tracerite_div.find("script") is None


def test_html_page_preserves_exception_h2():
    html = html_page(_sample_exception())
    soup = BeautifulSoup(html, "html.parser")
    h2 = soup.find("h2")
    assert h2 is not None
    assert "ValueError" in h2.get_text()


def test_html_page_no_exception():
    html = html_page(title="Empty", heading="Nothing to see")
    soup = BeautifulSoup(html, "html.parser")
    assert _text(soup.title) == "Empty"
    assert _text(soup.h1) == "Nothing to see"

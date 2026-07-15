"""Shared HTML index-page helpers for the framework demos."""

from __future__ import annotations

from html5tagger import HTML, E

import tracerite

from demo.helpers import discover_scenarios
from tracerite.html import Header, Page, html_traceback

SCENARIOS = discover_scenarios()


def first_sentence(doc: str | None) -> str:
    if not doc:
        return ""
    first = doc.strip().split(".")[0]
    return f"{first}."


async def generate_previews() -> list[tuple[str, str]]:
    """Generate HTML previews for all scenarios."""
    __tracebackhide__ = True
    tracerite.load()
    previews: list[tuple[str, str]] = []
    for _name, _func in SCENARIOS:
        _async_impl = getattr(_func, "_async_impl", None)
        try:
            if _async_impl is not None:
                await _async_impl()
            else:
                _func()
        except BaseException as _exc:
            _html = str(html_traceback(exc=_exc, include_js_css=False))
            previews.append((_name, _html))
        else:
            previews.append((_name, ""))
    return previews


async def build_index_html(*, framework: str | None = None) -> str:
    """Generate previews and build the demo index page HTML."""
    previews = await generate_previews()
    return _build_index_html(previews, framework=framework)


def _build_index_html(
    previews: list[tuple[str, str]], *, framework: str | None = None
) -> str:
    """Build the demo index page HTML from generated previews."""
    preview_map = dict(previews)
    with E.div() as content:
        for name, func in SCENARIOS:
            with content.h2:
                content.a[".open-link"]("▶ ", name.upper(), href=name)
            content.p[".scenario-doc"](func.__doc__)
            _preview = preview_map.get(name, "")
            if _preview:
                content.div(HTML(_preview))

    return str(
        Page(
            Title="TraceRite",
            Header=E.style(INDEX_STYLE),
            Heading=Header(
                Heading="Python tracebacks for Humans (and Machines)",
                Ingress=(
                    "You are viewing the TraceRite standalone HTML demo. "
                    "The examples below show rendered reports for various error scenarios."
                    if framework is None
                    else f"You are viewing the TraceRite {framework} demo. "
                    "The examples below show rendered reports for various error scenarios. "
                    f"Click the ▶ buttons to run a scenario live and let {framework} handle it."
                ),
            ),
            Content=content,
            Footer="",
        )
    )


INDEX_STYLE = """\
h2 { margin: 1em 0 0 0; }
.open-link {
  display: inline-block;
  padding: 0.1em 0.5em;
  margin: 0;
  font-size: 0.8em;
  background: #06c;
  color: #fff;
  border-radius: 0.1em;
  text-decoration: none;
}
.open-link:hover { background: #0055aa; }
code { background: #f5f5f5; padding: 2px 5px; border-radius: 3px; }
"""

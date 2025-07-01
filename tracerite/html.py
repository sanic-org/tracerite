from importlib.resources import files

from html5tagger import HTML, E

from .trace import extract_chain

style = files(__package__).joinpath("style.css").read_text(encoding="UTF-8")

detail_show = "{display: inherit}"

symbols = {"call": "➤", "warning": "⚠️", "error": "💣", "stop": "🛑"}
tooltips = {
    "call": "Function call",
    "warning": "Bug may be here\n(call from user code)",
    "error": "Exception {type} raised",
    "stop": "Execution interrupted\n(BaseException)",
}
javascript = """const scrollto=id=>document.getElementById(id).scrollIntoView({behavior:'smooth',block:'nearest',inline:'start'})"""

chainmsg = {
    "cause": "The above exception was raised from",
    "context": "The above exception was raised while handling",
    "none": "",  # Shouldn't happen between any two exceptions (only at the initial i.e. last in chain)
}


def html_traceback(
    exc=None, chain=None, *, include_js_css=True, local_urls=False, **extract_args
):
    chain = chain or extract_chain(exc=exc, **extract_args)[-3:]
    with E.div(class_="tracerite") as doc:
        if include_js_css:
            doc._script(javascript)
            doc._style(style)
        msg = None
        for e in chain:
            if msg:
                doc.p(msg, class_="after")
            msg = chainmsg[e["has"]]
            _exception(doc, e, local_urls=local_urls)

        with doc.script:
            for e in reversed(chain):
                for info in e["frames"]:
                    if info["relevance"] != "call":
                        doc(f"scrollto('{info['id']}')\n")
                        break
    return doc


def _exception(doc, info, *, local_urls=False):
    """Format single exception message and traceback"""
    summary, message = info["summary"], info["message"]
    doc.h3(E.span(f"{info['type']}:", class_="exctype")(f" {summary}"))
    if summary != message:
        if message.startswith(summary):
            message = message[len(summary) :]
        doc.pre(message, class_="excmessage")
    # Traceback available?
    frames = info["frames"]
    if not frames:
        return
    # Format call chain
    limitedframes = [*frames[:10], ..., *frames[-4:]] if len(frames) > 16 else frames
    with doc.div(class_="traceback-tabs"):
        if len(limitedframes) > 1:
            with doc.div(class_="traceback-labels"):
                for frinfo in limitedframes:
                    if frinfo is ...:
                        doc("...")
                        continue
                    _tab_header(doc, frinfo)
        with doc.div(class_="content"):
            for frinfo in limitedframes:
                if frinfo is ...:
                    with doc.div(class_="traceback-details"):
                        doc.p("...")
                    continue
                with doc.div(
                    class_="traceback-details",
                    data_function=frinfo["function"],
                    id=frinfo["id"],
                ):
                    traceback_detail(doc, info, frinfo, local_urls=local_urls)
                    variable_inspector(doc, frinfo["variables"])


def _tab_header(doc, frinfo):
    with doc.button(onclick=f"scrollto('{frinfo['id']}')"):
        doc.strong(frinfo["location"]).br.small(frinfo["function"] or "－")
        if frinfo["relevance"] != "call":
            doc.span(symbols.get(frinfo["relevance"]), class_="symbol")


def traceback_detail(doc, info, frinfo, *, local_urls):
    function = frinfo["function"]
    if frinfo["filename"]:
        doc.div.b(frinfo["filename"])(f":{frinfo['lineno']}")
        urls = frinfo["urls"]
        if local_urls and urls:
            for name, href in urls.items():
                doc(" ").a(name, href=href)
    else:
        doc.div.b(frinfo["location"] or E("Native function ").strong(function), ":")
    # Code printout
    fragments = frinfo.get("fragments", [])
    if not fragments:
        doc.p("Source code not available")
        if frinfo is info["frames"][-1]:
            doc(" but ").strong(info["type"])(" was raised from here")
    else:
        with doc.pre, doc.code:
            start = frinfo["linenostart"]
            for line_info in fragments:
                line_num = line_info["line"]
                abs_line = start + line_num - 1
                fragments = line_info["fragments"]

                # Render content fragments inside the codeline span
                with doc.span(class_="codeline", data_lineno=abs_line):
                    for fragment in fragments:
                        if "trailing" in fragment:
                            break

                        # Prepare tooltip attributes for mark fragments on final line
                        tooltip_attrs = {}
                        if line_num == frinfo["end_lineno"]:
                            relevance = frinfo["relevance"]
                            symbol = symbols.get(relevance, frinfo["relevance"])
                            try:
                                text = tooltips[relevance].format(**info, **frinfo)
                            except Exception:
                                text = repr(relevance)
                            tooltip_attrs = {
                                "class": "tracerite-tooltip",
                                "data-symbol": symbol,
                                "data-tooltip": text,
                            }

                        _render_fragment(doc, fragment, tooltip_attrs)
                    else:
                        fragment = None
                # Render trailing fragment outside the span
                if fragment:
                    _render_fragment(doc, fragment, {})


def _render_line_fragments(doc, fragments, info, frinfo, is_error_line):
    """Render a line using the fragment-based structure."""
    for fragment in fragments:
        _render_fragment(doc, fragment)


def _render_fragment(doc, fragment, tooltip_attrs=None):
    """Render a single fragment with appropriate styling."""
    code = fragment["code"]
    tooltip_attrs = tooltip_attrs or {}

    mark = fragment.get("mark")
    em = fragment.get("em")

    # Render opening tags for "mark" and "em" if applicable
    if mark in ["solo", "beg"]:
        # Apply tooltip attributes if this is a mark fragment that should have them
        doc(HTML(str(E.mark(**tooltip_attrs)).removesuffix("</mark>")))
    if em in ["solo", "beg"]:
        doc(HTML("<em>"))

    # Render the code
    doc(code)

    # Render closing tags for "mark" and "em" if applicable
    if em in ["fin", "solo"]:
        doc(HTML("</em>"))
    if mark in ["fin", "solo"]:
        doc(HTML("</mark>"))


def variable_inspector(doc, variables):
    if not variables:
        return
    with doc.table(class_="inspector key-value"):
        for n, t, v in variables:
            doc.tr.td.span(n, class_="var")(": ").span(t, class_="type")(" = ").td(
                class_="val"
            )
            if isinstance(v, str):
                doc(v)
            else:
                _format_matrix(doc, v)


def _format_matrix(doc, v):
    skipcol = skiprow = False
    with doc.table:
        for row in v:
            if row[0] is None:
                skiprow = True
                continue
            doc.tr()
            if skiprow:
                skiprow = False
                doc(class_="skippedabove")
            for e in row:
                if e is None:
                    skipcol = True
                    continue
                if skipcol:
                    skipcol = False
                    doc.td(e, class_="skippedleft")
                else:
                    doc.td(e)

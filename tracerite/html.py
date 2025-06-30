from importlib.resources import files

from html5tagger import E

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


def html_traceback(
    exc=None, chain=None, *, include_js_css=True, local_urls=False, **extract_args
):
    chain = chain or extract_chain(exc=exc, **extract_args)[-3:]
    with E.div(class_="tracerite") as doc:
        if include_js_css:
            doc._script(javascript)
            doc._style(style)
        for e in chain:
            if e is not chain[0]:
                doc.p("The above exception occurred after catching", class_="after")
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
                with doc.div(class_="traceback-details", id=frinfo["id"]):
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
            lineno = frinfo["lineno"]
            for line_info in fragments:
                line_num = line_info["line"]
                abs_line = start + line_num - 1
                with doc.span(class_="codeline", data_lineno=abs_line):
                    _render_line_fragments(
                        doc, line_info["fragments"], info, frinfo, abs_line == lineno
                    )


def variable_inspector(doc, variables):
    if not variables:
        return
    with doc.table(class_="inspector key-value"):
        for n, t, v in variables:
            doc.tr.td.span(n, class_="var")(": ").span(t, class_="type")(
                "\xa0=\xa0"
            ).td(class_="val")
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


def _render_line_fragments(doc, fragments, info, frinfo, is_error_line):
    """Render a line using the new fragment-based structure."""
    if not fragments:
        return

    # If this is the error line, apply error styling
    if is_error_line:
        symbol = symbols.get(frinfo["relevance"])
        try:
            tooltip_text = tooltips[frinfo["relevance"]].format(**info, **frinfo)
        except Exception:
            tooltip_text = repr(frinfo["relevance"])

        with doc.span(
            data_symbol=symbol,
            data_tooltip=tooltip_text,
            class_="tracerite-tooltip",
        ):
            for fragment in fragments:
                _render_fragment(doc, fragment)
    else:
        # Regular line, just render fragments without error styling
        for fragment in fragments:
            _render_fragment(doc, fragment)


def _render_fragment(doc, fragment):
    """Render a single fragment with appropriate styling."""
    code = fragment["code"]

    # Determine if this fragment has highlighting
    has_mark = "mark" in fragment
    has_em = "em" in fragment

    if has_mark and has_em:
        # Both mark and em - handle nesting based on beg/mid/fin/solo
        mark_type = fragment["mark"]
        em_type = fragment["em"]

        if mark_type == "solo":
            # Complete mark tag
            if em_type == "solo":
                doc.mark(doc.em(code))
            elif em_type == "beg":
                doc.mark(doc("<em>"), code)
            elif em_type == "mid":
                doc.mark(code)
            elif em_type == "fin":
                doc.mark(code, doc("</em>"))
        elif mark_type == "beg":
            # Opening mark tag
            if em_type == "solo":
                doc("<mark>").em(code)
            elif em_type == "beg":
                doc("<mark><em>", code)
            elif em_type == "mid":
                doc("<mark>", code)
            elif em_type == "fin":
                doc("<mark>", code, "</em>")
        elif mark_type == "mid":
            # No mark tags
            if em_type == "solo":
                doc.em(code)
            elif em_type == "beg":
                doc("<em>", code)
            elif em_type == "mid":
                doc(code)
            elif em_type == "fin":
                doc(code, "</em>")
        elif mark_type == "fin":
            # Closing mark tag
            if em_type == "solo":
                doc.em(code)("</mark>")
            elif em_type == "beg":
                doc("<em>", code, "</mark>")
            elif em_type == "mid":
                doc(code, "</mark>")
            elif em_type == "fin":
                doc(code, "</em></mark>")

    elif has_mark:
        # Just mark highlighting
        mark_type = fragment["mark"]
        if mark_type == "solo":
            doc.mark(code)
        elif mark_type == "beg":
            doc("<mark>", code)
        elif mark_type == "mid":
            doc(code)
        elif mark_type == "fin":
            doc(code, "</mark>")

    elif has_em:
        # Just em highlighting
        em_type = fragment["em"]
        if em_type == "solo":
            doc.em(code)
        elif em_type == "beg":
            doc("<em>", code)
        elif em_type == "mid":
            doc(code)
        elif em_type == "fin":
            doc(code, "</em>")
    else:
        # Regular fragment, check for special types
        fragment_class = None
        if "dedent" in fragment:
            fragment_class = "dedent"
        elif "indent" in fragment:
            fragment_class = "indent"
        elif "comment" in fragment:
            fragment_class = "comment"
        elif "trailing" in fragment:
            fragment_class = "trailing"

        if fragment_class:
            doc.span(code, class_=fragment_class)
        else:
            doc(code)

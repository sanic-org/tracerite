from importlib.resources import files

from html5tagger import HTML, E

from .trace import extract_chain

style = files(__package__).joinpath("style.css").read_text(encoding="UTF-8")

detail_show = "{display: inherit}"

symbols = {"call": "➤", "warning": "⚠️", "error": "💣", "stop": "🛑"}
tooltips = {
    "call": "Call",
    "warning": "Call from your code",
    "error": "{type}",
    "stop": "{type}",
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
        doc.div.b(frinfo["filename"])(
            f":{frinfo['range'].lfirst if frinfo['range'] else '?'}"
        )
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

                # Prepare tooltip attributes for tooltip span on final line
                tooltip_attrs = {}
                if frinfo["range"] and abs_line == frinfo["range"].lfinal:
                    relevance = frinfo["relevance"]
                    symbol = symbols.get(relevance, frinfo["relevance"])
                    try:
                        text = tooltips[relevance].format(**info, **frinfo)
                        # Replace newlines with spaces for HTML attribute
                        text = text.replace("\n", " ")
                    except Exception:
                        text = repr(relevance)
                    tooltip_attrs = {
                        "class": "tracerite-tooltip",
                        "data-symbol": symbol,
                        "data-tooltip": text,
                    }

                # Render content fragments inside the codeline span
                with doc.span(class_="codeline", data_lineno=abs_line):
                    # Find the first non-trailing fragment to start the tooltip span
                    non_trailing_fragments = []
                    trailing_fragment = None
                    for fragment in fragments:
                        if "trailing" in fragment:
                            trailing_fragment = fragment
                            break
                        non_trailing_fragments.append(fragment)

                    # Render leading whitespace/indentation first (outside tooltip span)
                    if non_trailing_fragments:
                        first_fragment = non_trailing_fragments[0]
                        code = first_fragment["code"]
                        leading_whitespace = code[: len(code) - len(code.lstrip())]
                        if leading_whitespace:
                            doc(leading_whitespace)
                            # Create modified first fragment without leading whitespace
                            first_fragment_modified = {
                                **first_fragment,
                                "code": code.lstrip(),
                            }
                            non_trailing_fragments[0] = first_fragment_modified

                    # Render the tooltip span around the actual code content
                    if tooltip_attrs and non_trailing_fragments:
                        with doc.span(
                            class_="tracerite-tooltip",
                            data_tooltip=tooltip_attrs["data-tooltip"],
                        ):
                            for fragment in non_trailing_fragments:
                                _render_fragment(doc, fragment)
                        # Add separate symbol and tooltip text elements
                        doc.span(
                            class_="tracerite-symbol",
                            data_symbol=tooltip_attrs["data-symbol"],
                        )
                        doc.span(
                            class_="tracerite-tooltip-text",
                            data_tooltip=tooltip_attrs["data-tooltip"],
                        )
                    else:
                        for fragment in non_trailing_fragments:
                            _render_fragment(doc, fragment)

                    # Set fragment for trailing handling
                    fragment = trailing_fragment
                # Render trailing fragment outside the span
                if fragment:
                    _render_fragment(doc, fragment)


def _render_fragment(doc, fragment):
    """Render a single fragment with appropriate styling."""
    code = fragment["code"]

    mark = fragment.get("mark")
    em = fragment.get("em")

    # Render opening tags for "mark" and "em" if applicable
    if mark in ["solo", "beg"]:
        doc(HTML("<mark>"))
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
        for var_info in variables:
            # Handle both old tuple format and new VarInfo namedtuple
            if hasattr(var_info, "name"):
                n, t, v, fmt = (
                    var_info.name,
                    var_info.typename,
                    var_info.value,
                    var_info.format_hint,
                )
            else:
                # Backwards compatibility with old tuple format
                n, t, v = var_info
                fmt = "inline"

            # Put the equals into the same span as the type so they stay bound.
            eq = "\u00a0=\u00a0"
            doc.tr.td.span(n, class_="var")(": ").span(t + eq, class_="type").td(
                class_=f"val val-{fmt}"
            )
            if isinstance(v, str):
                if fmt == "block":
                    # For block format, use <pre> tag for proper formatting
                    doc.pre(v)
                else:
                    doc(v)
            elif isinstance(v, dict) and v.get("type") == "dict":
                _format_dict(doc, v["rows"])
            else:
                _format_matrix(doc, v)


def _format_dict(doc, rows):
    """Format a dict as a definition list."""
    with doc.dl(class_="dict-dl"):
        for key, val in rows:
            doc.dt(key)
            doc.dd(val)


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

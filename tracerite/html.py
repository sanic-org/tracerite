from importlib.resources import files

from html5tagger import HTML, E

from .trace import chainmsg, extract_chain

style = files(__package__).joinpath("style.css").read_text(encoding="UTF-8")
javascript = files(__package__).joinpath("script.js").read_text(encoding="UTF-8")

detail_show = "{display: inherit}"

symbols = {"call": "➤", "warning": "⚠️", "error": "💣", "stop": "🛑"}
tooltips = {
    "call": "Call",
    "warning": "Call from your code",
    "error": "{type}",
    "stop": "{type}",
}


def html_traceback(
    exc=None,
    chain=None,
    *,
    include_js_css=True,
    local_urls=False,
    replace_previous=False,
    **extract_args,
):
    chain = chain or extract_chain(exc=exc, **extract_args)[-3:]
    # Chain is oldest-first from extract_chain
    with E.div(
        class_="tracerite", data_replace_previous="1" if replace_previous else None
    ) as doc:
        if include_js_css:
            doc._style(style)
        for i, e in enumerate(chain):
            # Get chaining suffix for exception header
            chain_suffix = ""
            if i > 0:
                chain_suffix = chainmsg.get(e.get("from", "none"), "")
            _exception(doc, e, local_urls=local_urls, chain_suffix=chain_suffix)

        if include_js_css:
            # Build scrollto calls
            scrollto_calls = []
            for e in reversed(chain):
                for info in e["frames"]:
                    if info["relevance"] != "call":
                        scrollto_calls.append(f"tracerite_scrollto('{info['id']}')")
                        break
            doc._script(javascript + "\n" + "\n".join(scrollto_calls))
    return doc


def _exception(doc, info, *, local_urls=False, chain_suffix=""):
    """Format single exception message and traceback"""
    summary, message = info["summary"], info["message"]
    doc.h3(E.span(f"{info['type']}{chain_suffix}:", class_="exctype")(f" {summary}"))
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
    # Collect symbols for floating indicators
    frame_symbols = []
    for frinfo in limitedframes:
        if frinfo is not ... and frinfo["relevance"] != "call":
            frame_symbols.append((frinfo["id"], symbols.get(frinfo["relevance"], "")))
    with doc.div(class_="traceback-wrapper"):
        # Floating overlay symbols (one per frame, positioned dynamically)
        for fid, sym in frame_symbols:
            with doc.span(class_="floating-symbol", data_frame=fid):
                doc.span("◂", class_="arrow arrow-left")
                doc.span(sym, class_="sym")
                doc.span("▸", class_="arrow arrow-right")
        with doc.div(class_="traceback-frames"):
            # Render frames
            for frinfo in limitedframes:
                if frinfo is ...:
                    with doc.div(class_="traceback-details traceback-ellipsis"):
                        doc.p("...")
                    continue
                attrs = {
                    "class_": "traceback-details",
                    "data_function": frinfo["function"],
                    "id": frinfo["id"],
                }
                with doc.div(**attrs):
                    _frame_label(doc, frinfo, local_urls=local_urls)
                    with doc.div(class_="frame-content"):
                        traceback_detail(doc, info, frinfo)
                        variable_inspector(doc, frinfo["variables"])


def _frame_label(doc, frinfo, *, local_urls=False):
    """Render sticky label for a code frame with optional editor links."""
    # Build title text: full path with line number
    if frinfo["filename"]:
        lineno = frinfo["range"].lfirst if frinfo["range"] else "?"
        title = f"{frinfo['filename']}:{lineno}"
    else:
        title = frinfo["location"] or frinfo["function"] or ""

    with doc.div(class_="frame-label", title=title):
        doc.strong(frinfo["location"])
        if frinfo["function"]:
            doc(" ")
            doc.span(frinfo["function"], class_="frame-function")
        # Editor links if available
        urls = frinfo.get("urls", {})
        if local_urls and urls:
            for name, href in urls.items():
                doc.a(name, href=href, class_="frame-link")


def traceback_detail(doc, info, frinfo):
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

            doc.tr.td.span(n, class_="var")
            if t:
                doc(": ").span(f"{t}\u00a0=\u00a0", class_="type")
            else:
                doc("\u00a0").span("=\u00a0", class_="type")  # No type printed
            doc.td(class_=f"val val-{fmt}")
            if isinstance(v, str):
                if fmt == "block":
                    # For block format, use <pre> tag for proper formatting
                    doc.pre(v)
                else:
                    doc(v)
            elif isinstance(v, dict) and v.get("type") == "keyvalue":
                _format_keyvalue(doc, v["rows"])
            elif isinstance(v, dict) and v.get("type") == "array":
                with doc.div(class_="array-with-scale"):
                    _format_matrix(doc, v["rows"])
                    if v.get("suffix"):
                        doc.span(v["suffix"], class_="scale-suffix")
            else:
                _format_matrix(doc, v)


def _format_keyvalue(doc, rows):
    """Format key-value pairs (dicts, dataclasses) as a definition list."""
    with doc.dl(class_="keyvalue-dl"):
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

from importlib.resources import files

from html5tagger import E

from .trace import extract_chain

style = files(__package__).joinpath("style.css").read_text(encoding="UTF-8")

detail_show = "{display: inherit}"

symbols = dict(call="➤", warning="⚠️", error="💣", stop="🛑")
tooltips = dict(
    call="Function call",
    warning="Bug may be here\n(call from user code)",
    error="Exception {type} raised",
    stop="Execution interrupted\n(BaseException)",
)
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
    lines = frinfo["lines"].splitlines(keepends=True)
    if not lines:
        doc.p("Source code not available")
        if frinfo is info["frames"][-1]:
            doc(" but ").strong(info["type"])(" was raised from here")
    else:
        with doc.pre, doc.code:
            start = frinfo["linenostart"]
            lineno = frinfo["lineno"]
            for i, line in enumerate(lines, start=start):
                with doc.span(class_="codeline", data_lineno=i):
                    doc(marked(line, info, frinfo) if i == lineno else line)


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
            doc.tr
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


def marked(line, info, frinfo):
    indent, code, trailing = split3(line)
    symbol = symbols.get(frinfo["relevance"])
    try:
        text = tooltips[frinfo["relevance"]].format(**info, **frinfo)
    except Exception:
        text = repr(frinfo["relevance"])

    # Use precise column offsets if available
    colno = frinfo.get("colno")
    end_colno = frinfo.get("end_colno")
    highlight_info = frinfo.get("highlight_info")

    if colno is not None and highlight_info is not None:
        # Calculate relative indices within code
        indent_len = len(indent)
        start_idx = max(0, colno - indent_len)
        if end_colno is not None:
            end_idx = max(start_idx, end_colno - indent_len)
        else:
            end_idx = len(code)

        # Clamp to valid bounds
        start_idx = min(start_idx, len(code))
        end_idx = min(end_idx, len(code))

        before = code[:start_idx]
        highlight_region = code[start_idx:end_idx]
        after = code[end_idx:]

        # Apply intelligent highlighting based on AST analysis
        with E.span(
            data_symbol=symbol,
            data_tooltip=text,
            class_="tracerite-tooltip",
        ) as doc:
            doc(indent, before)

            # Handle different highlight types
            if highlight_info["type"] == "caret":
                # Single character caret
                offset = highlight_info["offset"]
                if offset < len(highlight_region):
                    prior = highlight_region[:offset]
                    caret_char = highlight_region[offset : offset + 1]
                    rest = highlight_region[offset + 1 :]

                    with doc.mark:
                        doc(prior)
                        if caret_char:
                            doc.em(caret_char)
                        doc(rest)
                else:
                    doc.mark(highlight_region)

            elif highlight_info["type"] == "range":
                # Range highlighting within the error region
                range_start = highlight_info["start"]
                range_end = highlight_info["end"]

                # Clamp to highlight region bounds
                range_start = max(0, min(range_start, len(highlight_region)))
                range_end = max(range_start, min(range_end, len(highlight_region)))

                pre_range = highlight_region[:range_start]
                range_part = highlight_region[range_start:range_end]
                post_range = highlight_region[range_end:]

                with doc.mark:
                    doc(pre_range)
                    if range_part:
                        doc.em(range_part)
                    doc(post_range)

            elif highlight_info["type"] == "ranges":
                # Multiple ranges (future expansion)
                # For now, just highlight the whole region
                doc.mark(highlight_region)
            else:
                # Unknown type, fallback to whole region
                doc.mark(highlight_region)

            doc(after)

        doc(trailing)  # endline
        return doc

    elif colno is not None:
        # Old-style column highlighting without AST info
        indent_len = len(indent)
        start_idx = max(0, colno - indent_len)
        if end_colno is not None:
            end_idx = max(start_idx, end_colno - indent_len)
        else:
            end_idx = len(code)

        # Clamp to valid bounds
        start_idx = min(start_idx, len(code))
        end_idx = min(end_idx, len(code))

        before = code[:start_idx]
        highlight = code[start_idx:end_idx]
        after = code[end_idx:]

        # Simple fallback highlighting
        result = E(indent)(before).mark(
            E.span(highlight),
            data_symbol=symbol,
            data_tooltip=text,
            class_="tracerite-tooltip",
        )(after + trailing)

        return result

    # Fallback to full-line highlight
    return E(indent).mark(
        E.span(code), data_symbol=symbol, data_tooltip=text, class_="tracerite-tooltip"
    )(trailing)


def split3(s):
    """Split s into indent, code and trailing whitespace"""
    a, b, c = s.rstrip(), s.strip(), s.lstrip()
    codelen = len(b)
    return a[:-codelen], b, c[codelen:]

import pkg_resources
from html5tagger import E

from .trace import extract_chain

style = pkg_resources.resource_string(__name__, "style.css").decode()

detail_show = "{display: inherit}"

symbols = dict(call="➤", warning="⚠️", error="💣", stop="🛑")
tooltips = dict(
    call="Function call",
    warning="Bug may be here\n(call from user code)",
    error="Exception {type} raised",
    stop="Execution interrupted\n(BaseException)",
)
javascript = """const scrollto=id=>document.getElementById(id).scrollIntoView({behavior:'smooth',block:'nearest',inline:'start'})"""


def html_traceback(exc=None, chain=None, *, include_js_css=True, local_urls=False, **extract_args):
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
            message = message[len(summary):]
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
                with doc.div(class_="traceback-details", id=frinfo['id']):
                    traceback_detail(doc, info, frinfo, local_urls=local_urls)
                    variable_inspector(doc, frinfo["variables"])

def _tab_header(doc, frinfo):
    with doc.button(onclick=f"scrollto('{frinfo['id']}')"):
        doc.strong(frinfo["location"]).br.small(frinfo["function"] or "－")
        if frinfo["relevance"] != "call":
            doc.span(symbols.get(frinfo["relevance"]), class_="symbol")

def traceback_detail(doc, info, frinfo, *, local_urls):
    function = frinfo["function"]
    if frinfo['filename']:
        doc.div.b(frinfo['filename'])(f":{frinfo['lineno']}")
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
                    doc(
                        marked(line, info, frinfo)
                        if i == lineno else
                        line
                    )

def variable_inspector(doc, variables):
    if not variables:
        return
    with doc.table(class_="inspector key-value"):
        for n, t, v in variables:
            doc.tr.td.span(n, class_="var")(": ").span(t, class_="type")("\xA0=\xA0").td(class_="val")
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
    relevance = frinfo["relevance"]
    symbol = symbols.get(relevance)
    try:
        text = tooltips[relevance].format(**info, **frinfo)
    except Exception:
        text = repr(relevance)
    return E(indent).mark(E.span(code), data_symbol=symbol, data_tooltip=text, class_="tracerite-tooltip")(trailing)


def split3(s):
    """Split s into indent, code and trailing whitespace"""
    a, b, c = s.rstrip(), s.strip(), s.lstrip()
    codelen = len(b)
    return a[:-codelen], b, c[codelen:]

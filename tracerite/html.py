from __future__ import annotations

from importlib.resources import files
from typing import Any, cast

from html5tagger import HTML, E  # type: ignore[import]

from .chain_analysis import build_chronological_frames
from .trace import build_chain_header, chainmsg, extract_chain, symbols, symdesc

style = files(cast(str, __package__)).joinpath("style.css").read_text(encoding="UTF-8")
javascript = (
    files(cast(str, __package__)).joinpath("script.js").read_text(encoding="UTF-8")
)

detail_show = "{display: inherit}"


def _collapse_call_runs(
    frames: list[dict[str, Any]], min_run_length: int = 10
) -> list[Any]:
    """Collapse consecutive runs of 'call' frames, keeping first and last of each run.

    Only collapses runs of frames with relevance='call'. Non-call frames
    (error, warning, stop) are never collapsed.
    """
    if not frames:
        return frames

    result = []
    run_start = None

    for i, frinfo in enumerate(frames):
        if frinfo.get("relevance", "call") == "call":
            if run_start is None:
                run_start = i
        else:
            # End of a call run - process it
            if run_start is not None:
                run_length = i - run_start
                if run_length >= min_run_length:
                    # Keep first and last of the run, add ellipsis
                    result.append(frames[run_start])
                    result.append(...)
                    result.append(frames[i - 1])
                else:
                    # Run too short, keep all
                    result.extend(frames[run_start:i])
                run_start = None
            # Add the non-call frame
            result.append(frinfo)

    # Handle final run at end
    if run_start is not None:
        run_length = len(frames) - run_start
        if run_length >= min_run_length:
            result.append(frames[run_start])
            result.append(...)
            result.append(frames[-1])
        else:
            result.extend(frames[run_start:])

    return result


def html_traceback(
    exc: BaseException | None = None,
    chain: list[dict[str, Any]] | None = None,
    *,
    msg: str | None = ...,  # type: ignore[assignment]
    include_js_css: bool = True,
    local_urls: bool = False,
    replace_previous: bool = False,
    autodark: bool = True,
    chronological: bool = True,
    **extract_args: Any,
) -> Any:
    chain = chain or extract_chain(exc=exc, **extract_args)[-3:]
    # Chain is oldest-first from extract_chain
    classes = "tracerite autodark" if autodark else "tracerite"
    with E.div(
        class_=classes, data_replace_previous="1" if replace_previous else None
    ) as doc:
        if include_js_css:
            doc._style(style)

        # Add chain header (use default if msg is ..., skip if msg is None/empty)
        if msg is ...:
            msg = build_chain_header(chain) if chain else None
        if msg:
            doc.h2(msg)

        if chronological:
            _chronological_output(doc, chain, local_urls=local_urls)
        else:
            for i, e in enumerate(chain):
                # Get chaining suffix for exception header
                chain_suffix = ""
                if i > 0:
                    chain_suffix = chainmsg.get(e.get("from", "none"), "")
                _exception(doc, e, local_urls=local_urls, chain_suffix=chain_suffix)

        if include_js_css:
            # Build scrollto calls for chronological mode
            if chronological:
                chrono_frames = build_chronological_frames(chain)
                scrollto_calls = []
                for frinfo in reversed(chrono_frames):
                    if frinfo.get("relevance") not in ("call", "except"):
                        scrollto_calls.append(f"tracerite_scrollto('{frinfo['id']}')")
                        break
            else:
                scrollto_calls = []
                for e in reversed(chain):
                    for info in e["frames"]:
                        if info["relevance"] != "call":
                            scrollto_calls.append(f"tracerite_scrollto('{info['id']}')")
                            break
            doc._script(javascript + "\n" + "\n".join(scrollto_calls))
    return doc


def _chronological_output(
    doc: Any, chain: list[dict[str, Any]], *, local_urls: bool = False
) -> None:
    """Output frames in chronological order with exception info after error frames."""
    chrono_frames = build_chronological_frames(chain)
    if not chrono_frames:
        # No frames, but still show exception banners for any exceptions in chain
        for exc in chain:
            exc_info = {
                "type": exc.get("type"),
                "message": exc.get("message"),
                "summary": exc.get("summary"),
                "from": exc.get("from"),
            }
            _exception_banner(doc, exc_info)
        return

    # Collapse consecutive call runs
    limited_frames = _collapse_call_runs(chrono_frames, min_run_length=10)

    for frinfo in limited_frames:
        if frinfo is ...:
            doc.p("...", class_="traceback-ellipsis")
            continue

        relevance = frinfo.get("relevance", "call")
        exc_info = frinfo.get("exception")

        attrs = {
            "class_": f"traceback-details traceback-{relevance}",
            "data_function": frinfo.get("function"),
            "id": frinfo["id"],
        }
        with doc.div(**attrs):
            _frame_label(doc, frinfo, local_urls=local_urls)
            if relevance == "call":
                with doc.span(class_="compact-call-line"):
                    _compact_call_line_chrono(doc, frinfo)
            with doc.div(class_="frame-content"):
                _traceback_detail_chrono(doc, frinfo)
                variable_inspector(doc, frinfo.get("variables", []))

        # Print exception info AFTER the error frame
        if exc_info:
            _exception_banner(doc, exc_info)


def _exception_banner(doc: Any, exc_info: dict[str, Any]) -> None:
    """Output exception type and message as a banner after the error frame."""
    exc_type = exc_info.get("type", "Exception")
    summary = exc_info.get("summary", "")
    message = exc_info.get("message", "")
    from_type = exc_info.get("from", "none")

    chain_suffix = chainmsg.get(from_type, "")

    doc.h3(E.span(f"{exc_type}{chain_suffix}:", class_="exctype")(f" {summary}"))
    if summary != message:
        if message.startswith(summary):
            message = message[len(summary) :]
        doc.pre(message, class_="excmessage")


def _compact_call_line_chrono(doc: Any, frinfo: dict[str, Any]) -> None:
    """Render compact one-liner for call frames in chronological mode."""
    fragments = frinfo.get("fragments", [])
    symbol = symbols.get(frinfo.get("relevance", "call"), symbols["call"])

    if fragments:
        with doc.code(class_="compact-code"):
            inmark = False
            for line_info in fragments:
                for fragment in line_info.get("fragments", []):
                    mark = fragment.get("mark")
                    if mark in ["solo", "beg"]:
                        inmark = True
                    if inmark:
                        em = fragment.get("em")
                        if em in ["solo", "beg"]:
                            doc(HTML("<em>"))
                        doc(fragment["code"])
                        if em in ["solo", "fin"]:
                            doc(HTML("</em>"))
                    if mark in ["fin", "solo"]:
                        inmark = False
    doc.span(symbol, class_="compact-symbol")


def _traceback_detail_chrono(doc: Any, frinfo: dict[str, Any]) -> None:
    """Render frame detail in chronological mode."""
    fragments = frinfo.get("fragments", [])
    exc_info = frinfo.get("exception")

    if not fragments:
        doc.p("Source code not available")
        if exc_info:
            doc(" but ").strong(exc_info.get("type", "Exception"))(
                " was raised from here"
            )
        return

    with doc.pre, doc.code:
        start = frinfo.get("linenostart", 1)
        for line_info in fragments:
            line_num = line_info["line"]
            abs_line = start + line_num - 1
            line_fragments = line_info["fragments"]

            # Prepare tooltip attributes for tooltip span on final line
            tooltip_attrs = {}
            frame_range = frinfo.get("range")
            if frame_range and abs_line == frame_range.lfinal:
                relevance = frinfo.get("relevance", "call")
                symbol = symbols.get(relevance, relevance)
                exc_info = frinfo.get("exception")
                text = "" if exc_info else symdesc.get(relevance, relevance)
                text = text.replace("\n", " ")
                tooltip_attrs = {
                    "class": "tracerite-tooltip",
                    "data-symbol": symbol,
                    "data-tooltip": text,
                }

            with doc.span(class_="codeline", data_lineno=abs_line):
                non_trailing_fragments = []
                trailing_fragment = None
                for fragment in line_fragments:
                    if "trailing" in fragment:
                        trailing_fragment = fragment
                        break
                    non_trailing_fragments.append(fragment)

                if non_trailing_fragments:
                    first_fragment = non_trailing_fragments[0]
                    code = first_fragment["code"]
                    leading_whitespace = code[: len(code) - len(code.lstrip())]
                    if leading_whitespace:
                        doc(leading_whitespace)
                        first_fragment_modified = {
                            **first_fragment,
                            "code": code.lstrip(),
                        }
                        non_trailing_fragments[0] = first_fragment_modified

                if tooltip_attrs and non_trailing_fragments:
                    with doc.span(
                        class_="tracerite-tooltip",
                        data_tooltip=tooltip_attrs["data-tooltip"],
                    ):
                        for fragment in non_trailing_fragments:
                            _render_fragment(doc, fragment)
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

                fragment = trailing_fragment
            if fragment:
                _render_fragment(doc, fragment)


def _exception(
    doc: Any, info: dict[str, Any], *, local_urls: bool = False, chain_suffix: str = ""
) -> None:
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
    # Format call chain, suppress middle of consecutive call runs if too long
    limitedframes = _collapse_call_runs(frames, min_run_length=10)
    # Render frames
    for frinfo in limitedframes:
        if frinfo is ...:
            doc.p("...", class_="traceback-ellipsis")
            continue
        relevance = frinfo.get("relevance", "call")
        attrs = {
            "class_": f"traceback-details traceback-{relevance}",
            "data_function": frinfo["function"],
            "id": frinfo["id"],
        }
        with doc.div(**attrs):
            # All frame types: output frame-function, frame-location directly for grid
            _frame_label(doc, frinfo, local_urls=local_urls)
            if relevance == "call":
                with doc.span(class_="compact-call-line"):
                    _compact_call_line_html(doc, info, frinfo)
            with doc.div(class_="frame-content"):
                traceback_detail(doc, info, frinfo)
                variable_inspector(doc, frinfo["variables"])


def _frame_label(doc: Any, frinfo: dict[str, Any], local_urls: bool = False) -> None:
    if frinfo["function"]:
        doc.span(frinfo["function"], class_="frame-function")
        # Display suffix (e.g. ⚡except) separately for visual purposes only
        if frinfo.get("function_suffix"):
            doc.span(frinfo["function_suffix"], class_="frame-function-suffix")
        doc(" ")
    lineno = None
    if frinfo["relevance"] == "call" and frinfo["linenostart"]:
        lineno = E.span(f":{frinfo['linenostart']}", class_="frame-lineno")
    # Add colon for non-call frames (where code follows)
    colon = E.span(":", class_="frame-colon") if frinfo["relevance"] != "call" else None
    doc.span(frinfo["location"], lineno, colon, class_="frame-location")
    urls = frinfo.get("urls", {})
    if local_urls and urls:
        for name, href in urls.items():
            doc.a(name, href=href, class_="frame-link")


def _compact_call_line_html(
    doc: Any, info: dict[str, Any], frinfo: dict[str, Any]
) -> None:
    """Render compact one-liner for call frames: function location marked_code symbol."""
    fragments = frinfo.get("fragments", [])
    symbol = symbols["call"]
    # Extract only marked code from the FIRST marked line only (like TTY)
    if fragments:
        with doc.code(class_="compact-code"):
            inmark = False
            for line_info in fragments:
                for fragment in line_info.get("fragments", []):
                    mark = fragment.get("mark")
                    if mark in ["solo", "beg"]:
                        inmark = True
                    if inmark:
                        em = fragment.get("em")
                        if em in ["solo", "beg"]:
                            doc(HTML("<em>"))
                        doc(fragment["code"])
                        if em in ["solo", "fin"]:
                            doc(HTML("</em>"))
                    if mark in ["fin", "solo"]:
                        inmark = False
    # Symbol
    doc.span(symbol, class_="compact-symbol")


def traceback_detail(doc: Any, info: dict[str, Any], frinfo: dict[str, Any]) -> None:
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
                    text = symdesc.get(relevance, "")
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


def _render_fragment(doc: Any, fragment: dict[str, Any]) -> None:
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


def variable_inspector(doc: Any, variables: list[Any]) -> None:
    if not variables:
        return
    with doc.dl(class_="inspector"):
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

            doc.dt.span(n, class_="var")
            if t:
                doc(": ").span(f"{t}\u00a0=\u00a0", class_="type")
            else:
                doc("\u00a0").span("=\u00a0", class_="type")  # No type printed
            doc.dd(class_=f"val val-{fmt}")
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


def _format_keyvalue(doc: Any, rows: list[tuple[str, Any]]) -> None:
    """Format key-value pairs (dicts, dataclasses) as a definition list."""
    with doc.dl(class_="keyvalue-dl"):
        for key, val in rows:
            doc.dt(key)
            doc.dd(val)


def _format_matrix(doc: Any, v: Any) -> None:
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

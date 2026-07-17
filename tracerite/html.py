from __future__ import annotations

from importlib.resources import files
from types import EllipsisType
from typing import Any, cast

from html5tagger import HTML, Document, E, Template

from .trace.core import (
    EMPHASIS_BEG,
    EMPHASIS_FIN,
    HIGHLIGHT_RELEVANCES,
    chainmsg,
    symbols,
    symdesc,
)
from .trace.finalize import (
    call_run_ranges,
    exception_info,
    extract_chain,
    extract_chain_exceptions,
    function_display,
    normalize_variable,
)

style = files(cast(str, __package__)).joinpath("style.css").read_text(encoding="UTF-8")
javascript = (
    files(cast(str, __package__)).joinpath("script.js").read_text(encoding="UTF-8")
)

detail_show = "{display: inherit}"

PAGE_STYLE = """\
:root { color-scheme: light dark; font-family: system-ui, sans-serif }
main { padding: 1em }
p { margin: 0 0 0.5em 0 }
"""

# fmt: off
Page = Template(
    Document(E.Title, lang="en")
    .style(style, id="tracerite-style")
    .style(PAGE_STYLE)
    .Header
    .main(E.Heading.Content)
    .Footer
)
# fmt: on

Header = Template(E.h1.Heading.p.Ingress)


def html_page(
    exc: BaseException | None = None,
    *,
    title: str | None = None,
    heading: str | None = None,
    ingress: str | None = None,
    header: Any | None = None,
    footer: Any | None = None,
    msg: str | None | EllipsisType = ...,
    chain: dict[str, Any] | None = None,
    autodark: bool = True,
    local_urls: bool = False,
    **extract_args: Any,
) -> HTML:
    """Render a full HTML5 document containing a TraceRite traceback."""
    chain = extract_chain(exc=exc, **extract_args) if chain is None else chain
    frames = chain["frames"]
    page_title = (
        title
        if title is not None
        else (frames[-1].get("exception", {}).get("type") if frames else "TraceRite")
    )
    page_heading = heading if heading is not None else page_title
    page_ingress = (
        ingress
        if ingress is not None
        else "An unexpected error occurred while processing."
    )
    traceback_html = html_traceback(
        exc=exc,
        chain=chain,
        msg=msg,
        include_js_css=False,
        autodark=autodark,
        local_urls=local_urls,
    )
    return Page(
        Title=page_title,
        Header="" if header is None else header,
        Heading=Header(Heading=page_heading, Ingress=page_ingress),
        Content=traceback_html,
        Footer="" if footer is None else footer,
    )


def _collapse_call_runs(
    frames: list[dict[str, Any]], min_run_length: int = 10
) -> list[Any]:
    """Collapse consecutive runs of 'call' frames, keeping first and last of each run."""
    if not frames:
        return frames

    result = []
    prev_end = 0
    for start, end in call_run_ranges(frames, min_run_length):
        result.extend(frames[prev_end:start])
        run_length = end - start + 1
        skipped = run_length - 2
        result.append(frames[start])
        result.append((..., skipped))
        result.append(frames[end])
        prev_end = end + 1
    result.extend(frames[prev_end:])
    return result


def html_traceback(
    exc: BaseException | None = None,
    chain: dict[str, Any] | None = None,
    *,
    msg: str | None | EllipsisType = ...,
    include_js_css: bool = True,
    local_urls: bool = False,
    clear: bool = False,
    autodark: bool = True,
    **extract_args: Any,
) -> Any:
    """Render an exception as an interactive HTML fragment.

    Returns an html5tagger HTML fragment wrapped in a ``<div class="tracerite">``.
    By default the fragment includes the TraceRite stylesheet and JavaScript;
    set ``include_js_css=False`` when embedding it in a page that already
    provides them.
    """
    chain = chain or extract_chain(exc=exc, **extract_args)
    frames = chain["frames"]
    with E.div[".tracerite"](
        classes={"autodark": autodark},
        data_cleanup_mode="replace" if clear else None,
    ) as doc:
        if include_js_css:
            doc._style(style)

        # Add chain header (use default if msg is ..., skip if msg is None/empty)
        if msg is ...:
            msg = chain["header"] or None
        if msg:
            doc.h2(msg)

        if frames:
            _chronological_output(doc, frames, local_urls=local_urls)
        elif exc is not None:
            for exc_dict in extract_chain_exceptions(exc):
                _exception_banner(doc, exception_info(exc_dict))

        if include_js_css:
            doc._script(javascript)
    return doc


def _chronological_output(
    doc: Any,
    frames: list[dict[str, Any]],
    *,
    local_urls: bool = False,
) -> None:
    """Output frames in chronological order with exception info after error frames."""
    _render_frame_list(doc, frames, local_urls=local_urls)


def _render_frame_list(
    doc: Any,
    frames: list[dict[str, Any]],
    *,
    local_urls: bool = False,
) -> None:
    """Render a list of chronological frames, handling parallel branches."""
    # Collapse consecutive call runs
    limited_frames = _collapse_call_runs(frames, min_run_length=10)

    for frinfo in limited_frames:
        if isinstance(frinfo, tuple):
            assert frinfo[0] is ...
            skipped = frinfo[1]
            doc.p(f"⋮ {skipped} more calls", class_="traceback-ellipsis")
            continue

        relevance = frinfo["relevance"]
        exc_info = frinfo.get("exception")
        parallel_branches = frinfo.get("parallel")

        attrs = {
            "class_": f"traceback-details traceback-{relevance}",
            "data_function": frinfo["function"],
            "id": frinfo["id"],
        }
        with doc.div(**attrs):
            # Hidden checkbox for CSS-only toggle (all frames are collapsible)
            toggle_id = f"toggle-{frinfo['id']}"
            # Non-call frames are open by default (checked)
            if relevance == "call":
                doc.input_(
                    type="checkbox", id=toggle_id, class_="frame-toggle-checkbox"
                )
            else:
                doc.input_(
                    type="checkbox",
                    id=toggle_id,
                    class_="frame-toggle-checkbox",
                    checked="checked",
                )
            _frame_label(
                doc,
                frinfo,
                local_urls=local_urls,
                toggle_id=toggle_id,
            )
            with doc.span(class_="compact-call-line"):
                _compact_code_line(doc, frinfo)
            # Animated wrapper for expandable content
            with doc.div(class_="expand-wrapper"), doc.div(class_="expand-content"):
                _traceback_detail_chrono(doc, frinfo)
                variable_inspector(doc, frinfo["variables"])

        # Render parallel branches (subexceptions) before the exception banner
        if parallel_branches:
            _render_parallel_branches(doc, parallel_branches, local_urls=local_urls)

        # Print exception info AFTER the error frame (and parallel branches)
        if exc_info:
            _exception_banner(doc, exc_info)


def _render_parallel_branches(
    doc: Any,
    branches: list[list[dict[str, Any]]],
    *,
    local_urls: bool = False,
) -> None:
    """Render parallel exception branches from an ExceptionGroup."""
    with doc.div(class_="parallel-branches"):
        for branch in branches:
            with doc.div(class_="parallel-branch"):
                _render_frame_list(doc, branch, local_urls=local_urls)


def _exception_banner(doc: Any, exc_info: dict[str, Any]) -> None:
    """Output exception type and message as a banner after the error frame."""
    exc_type = exc_info.get("type", "Exception")
    summary = exc_info.get("summary", "")
    message = exc_info.get("message", "")
    from_type = exc_info.get("from", "none")

    chain_suffix = chainmsg.get(from_type, "")

    doc.h3(
        E.span(f"{exc_type}{chain_suffix}:", class_="exctype"),
        E.span(f" {summary}", class_="excsummary"),
    )
    # Show remaining lines in pre only if message has multiple lines.
    # Summary is always the first line, so we strip that from pre.
    parts = message.split("\n", 1)
    if len(parts) > 1:
        rest = parts[1].rstrip()  # Only strip trailing whitespace, preserve leading
        if rest:
            lines = rest.split("\n")
            visual_counts = [1 + len(line) // 80 for line in lines]
            total_visual = sum(visual_counts)
            marker_index: int | None = None
            if total_visual > 100 and len(lines) > 40:
                shown_visual = sum(visual_counts[:20]) + sum(visual_counts[-20:])
                skipped = total_visual - shown_visual
                lines = lines[:20] + [f"⋮ {skipped} more lines"] + lines[-20:]
                marker_index = 20
            with doc.pre(class_="excmessage"):
                for i, line in enumerate(lines):
                    if i > 0:
                        doc("\n")
                    if i == marker_index:
                        doc.span(line, class_="excmessage-ellipsis")
                    else:
                        doc(line)


def _compact_code_line(doc: Any, frinfo: dict[str, Any]) -> None:
    """Render compact one-liner showing all marked code regions."""
    fragments = frinfo["fragments"]
    relevance = frinfo["relevance"]
    symbol = symbols.get(relevance, symbols["call"])
    # Use highlight styling (yellow bg, red caret) for error/stop frames
    use_highlight = relevance in HIGHLIGHT_RELEVANCES

    if fragments:
        # First pass: collect text and track em ranges
        code_parts = []  # [(text, is_em), ...]

        for line_info in fragments:
            for fragment in line_info.get("fragments", []):
                mark = fragment.get("mark")
                if mark:
                    em = fragment.get("em")
                    text = fragment["code"]
                    is_em = em is not None
                    code_parts.append((text, is_em))

        # Find the em span (from first em start to last em end)
        em_indices = [i for i, (_, is_em) in enumerate(code_parts) if is_em]

        # Collapse em parts longer than 20 chars
        if em_indices:
            first_em_idx = min(em_indices)
            last_em_idx = max(em_indices)
            em_text = "".join(
                text
                for i, (text, _) in enumerate(code_parts)
                if first_em_idx <= i <= last_em_idx
            )
            if len(em_text) > 20:
                # Collapse: keep first and last char (typically parentheses)
                collapsed = em_text[0] + "…" + em_text[-1]
                # Rebuild code_parts with collapsed em
                new_parts = []
                for i, (text, _is_em) in enumerate(code_parts):
                    if i < first_em_idx:
                        new_parts.append((text, False))
                    elif i == first_em_idx:
                        new_parts.append((collapsed, True))
                    elif i > last_em_idx:  # pragma: no cover
                        new_parts.append((text, False))
                    # Skip parts within em range (already collapsed)
                code_parts = new_parts

        with doc.code(class_="compact-code"):
            if use_highlight:
                doc(HTML("<mark>"))

            for text, is_em in code_parts:
                if is_em:
                    doc(HTML("<em>"))
                doc(text)
                if is_em:
                    doc(HTML("</em>"))

            if use_highlight:
                doc(HTML("</mark>"))
    else:
        with doc.code(class_="compact-code"):
            doc.span("(no source code)", class_="no-source")
    # Add space before symbol for error/stop frames
    if use_highlight:
        doc(" ")
    doc.span(symbol, class_="compact-symbol")


def _traceback_detail_chrono(doc: Any, frinfo: dict[str, Any]) -> None:
    """Render frame detail in chronological mode."""
    fragments = frinfo["fragments"]
    relevance = frinfo["relevance"]
    symbol = symbols.get(relevance, "")
    desc = symdesc.get(relevance, "")
    symbol_text = f"{symbol} {desc}" if desc else symbol

    if not fragments:
        # Show "(no source code)" with the symbol emoji like a code line would have
        doc.p[".no-source"]("(no source code) ")
        doc.span(class_="tracerite-symbol", data_text=symbol_text)
        return

    with doc.pre, doc.code:
        start = frinfo["linenostart"]
        for line_info in fragments:
            line_num = line_info["line"]
            abs_line = start + line_num - 1
            line_fragments = line_info["fragments"]

            # Show the symbol next to the final line of the frame range
            show_symbol = False
            frame_range = frinfo["range"]
            if frame_range and abs_line == frame_range["lfinal"]:
                show_symbol = True

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

                for fragment in non_trailing_fragments:
                    _render_fragment(doc, fragment)

                if show_symbol and non_trailing_fragments:
                    doc.span(class_="tracerite-symbol", data_text=symbol_text)

                fragment = trailing_fragment
            if fragment:
                _render_fragment(doc, fragment)


def _frame_label(
    doc: Any,
    frinfo: dict[str, Any],
    local_urls: bool = False,
    toggle_id: str | None = None,
) -> None:
    function_label = function_display(
        frinfo["function"], frinfo.get("function_suffix", "")
    )

    # Location comes first, with line number for non-notebook frames
    # Notebook cells (In [N]) don't need line numbers displayed
    notebook_cell = frinfo.get("notebook_cell", False)
    if notebook_cell:
        lineno = None
    else:
        # Use cursor_line for display (preferred error position)
        cursor_line = frinfo.get("cursor_line") or frinfo.get("linenostart", 1)
        lineno = E.span(
            f":{cursor_line}",
            class_="frame-lineno",
        )

    # Colon after function name, or after location if no function
    colon = E.span(":", class_="frame-colon")

    # Get VS Code URL if local_urls enabled
    urls = frinfo.get("urls", {})
    vscode_url = urls.get("VS Code") if local_urls else None

    # Build location element - wrap in <a> if we have a VS Code URL
    def render_location(with_colon: bool = False):
        location_parts = (
            (frinfo["location"], lineno) if lineno else (frinfo["location"],)
        )
        if with_colon:
            location_parts = (*location_parts, colon)
        if vscode_url:
            doc.a(*location_parts, href=vscode_url, class_="frame-location")
        else:
            doc.span(*location_parts, class_="frame-location")

    if toggle_id:
        # Wrap both location and function in a label for click-to-toggle
        with doc.label(for_=toggle_id, class_="frame-label-wrapper"):
            if function_label:
                render_location()
                doc.span(function_label, colon, class_="frame-function")
            else:
                # No function: colon goes with location, empty span for grid column 2
                render_location(with_colon=True)
                doc.span(class_="frame-function")
    else:
        if function_label:
            render_location()
            doc.span(function_label, colon, class_="frame-function")
        else:
            # No function: colon goes with location, empty span for grid column 2
            render_location(with_colon=True)
            doc.span(class_="frame-function")


def _render_fragment(doc: Any, fragment: dict[str, Any]) -> None:
    """Render a single fragment with appropriate styling."""
    code = fragment["code"]

    mark = fragment.get("mark")
    em = fragment.get("em")

    # Render opening tags for "mark" and "em" if applicable
    if mark in EMPHASIS_BEG:
        doc(HTML("<mark>"))
    if em in EMPHASIS_BEG:
        doc(HTML("<em>"))

    # Render the code
    doc(code)

    # Render closing tags for "mark" and "em" if applicable
    if em in EMPHASIS_FIN:
        doc(HTML("</em>"))
    if mark in EMPHASIS_FIN:
        doc(HTML("</mark>"))


def variable_inspector(doc: Any, variables: list[Any]) -> None:
    if not variables:
        return
    with doc.dl(class_="inspector"):
        for var_info in variables:
            n, t, v, fmt = normalize_variable(var_info)

            doc.dt.span(n, class_="var")
            if t:
                doc(": ").span(t, class_="type")
            doc("\u00a0=\u00a0")
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

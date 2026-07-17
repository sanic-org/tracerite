# TraceRite API

TraceRite handles an exception in three stages, and the public API mirrors them:

1. **Chain extraction** — `extract_chain` turns an exception (including its
   causes, context, and `ExceptionGroup` subexceptions) into a machine-readable
   dict of frames.
2. **Formatting** — `html_traceback`, `html_page`, and `tty_traceback` render
   an exception, or a pre-extracted chain, for browsers and terminals.
3. **Hooks** — `load` and the related loaders install TraceRite as the default
   exception handler, so formatting happens automatically for uncaught
   exceptions, threads, and `logging.exception()` calls.

Everything documented here is importable directly from the `tracerite` package.
The IPython/Jupyter extension (`%load_ext tracerite`) is covered in the README.

## Formatting

All three formatters share the same first two parameters and pass any remaining
keyword arguments through to `extract_chain` (so `skip_outmost` and
`skip_until` work everywhere):

- **exc** — the exception to render. Defaults to the current exception from
  `sys.exc_info()`.
- **chain** — pre-extracted data from `extract_chain`. When given, **exc**
  is ignored.

### html_page

```python
html_page(exc=None, *, title=None, heading=None, ingress=None, header=None,
          footer=None, msg=..., chain=None, autodark=True, local_urls=False,
          **extract_args)
```

Renders a complete HTML5 document containing the traceback — a
ready-to-serve error page. Pure CSS, no scripts and no externally linked resources.

```python
from tracerite import html_page

try:
    risky_operation()
except Exception as exc:
    html = html_page(exc)
```

Parameters (in addition to **exc**, **chain**, **msg**, **autodark**, and
**local_urls**, which behave as in `html_traceback`):

- **title** — the document `<title>` and default heading. Defaults to the
  exception type. May be an empty string.
- **heading** — the `<h1>` text. Defaults to **title**. May be an empty string.
- **ingress** — an introductory paragraph below the heading. Defaults to a
  generic message. May be an empty string.
- **header**, **footer** — content injected before and after `<main>` (e.g.
  site navigation or extra `<style>` tags). Empty by default.

ℹ️ The function returns a `HTML` object that is a `str` with `__html__` and `_repr_html_` conversions understood by templating engines and frameworks as HTML code not needing escaping, but that also works where plain string is expected.

#### Web framework integration

```python
from flask import Flask
from tracerite import html_page

app = Flask(__name__)

@app.errorhandler(Exception)
def handle_error(exc):
    return html_page(exc, title="Internal Server Error"), 500
```

### html_traceback

```python
html_traceback(exc=None, chain=None, *, msg=..., include_js_css=True,
               local_urls=False, clear=False, autodark=True, **extract_args)
```

Renders an exception as an interactive HTML fragment wrapped in `<div class="tracerite">`. This interface allows you to fully composite your own error document, or to return the plain div fragment (useful with htmx or `fetch()`).

ℹ️ The function returns a `HTML` object that is a `str` with `__html__` and `_repr_html_` conversions understood by templating engines and frameworks as HTML code not needing escaping, but that also works where plain string is expected.


```python
from tracerite import html_traceback

try:
    risky_operation()
except Exception:
    html = html_traceback()  # Captures current exception
```

Parameters (in addition to **exc** and **chain**):

- **msg** — the heading above the traceback. The default `...` uses the chain
  header; pass `None` or an empty string for no heading, or any string to
  override it.
- **include_js_css** — include the style and script in the fragment (default). See below.
- **local_urls** — link source locations to local files (VS Code / file URLs with full path)
- **clear** — replace previous TraceRite reports on the page when a new one appears.
- **autodark** — follow the system dark-mode preference (default). The
  fragment carries an `autodark` class that the stylesheet reacts to.


**💡 Recommended:** When embedding fragments in a larger page, provide the TraceRite stylesheet once in your page without including them in every fragment you render. By default, each fragment is self-contained and includes both a `<style>` tag and a `<script>` tag. The `<script>` tag is only needed for the `clear=True` replacement behavior and other minor functions useful when we don't control the main document (e.g. Jupyter Notebooks); for most pages the stylesheet alone is sufficient. Use page-level assets whenever you have control over that.


```python
from tracerite import html_traceback, html_style

html = html_traceback(include_js_css=False)

# In your page template:
# <style>{{ html_style }}</style>
# ...
# {{ html }}
```



### tty_traceback

```python
tty_traceback(exc=None, chain=None, *, file=None, msg=None, tag="",
              term_width=None, **extract_args) -> None
```

Prints a colorful terminal traceback with ANSI colors and Unicode box drawing.
It writes directly to the terminal instead of returning a string, so it can
adapt to terminal features like the window size and color support (isatty).

```python
from tracerite import tty_traceback

try:
    risky_operation()
except Exception:
    tty_traceback()  # Captures current exception, prints to stderr
```

ℹ️ To capture plain-text output into a string:

```python
from io import StringIO
from tracerite import tty_traceback

try:
    risky_operation()
except Exception:
    buf = StringIO()
    tty_traceback(file=buf)
    plain_text = buf.getvalue()
```

Parameters (in addition to **exc** and **chain**):

- **file** — output destination, `sys.stderr` by default.
- **msg** — the header line. By default it is built from the exception chain; pass a string to override it.
- **tag** — a short tag displayed after the header (e.g. `"#TR1"`), handy as a reference to the TraceRite report
- **term_width** — the width to wrap output to. Auto-detected from the terminal when omitted.

## Hiding frames with `__tracebackhide__`

Frames can be hidden from TraceRite output by setting `__tracebackhide__` in the frame's globals or locals. This is primarily used by frameworks to suppress their own internal plumbing so that tracebacks point straight at the user's code.

The variable accepts two useful values:

- **Any truthy value** (commonly `True`) — hides the individual frame. This
  convention is recognized by pytest and others.
- **`"until"`** — hides this frame and all frames before it. This is a
  TraceRite extension: everything from the start of execution to the frame that set `"until"` itself is dropped, leaving only later frames visible.

Hiding is ignored when an exception actually occurs inside a hidden frame. In
that case the frame is shown anyway, because the exact location of the error is
always relevant.

```python
# In a helper module that should not appear in tracebacks
__tracebackhide__ = True

def framework_helper(user_func):
    with something:
        user_func()
```

```python
# Hide everything leading here, only showing from user_handler
def request_router():
    __tracebackhide__ = "until"
    user_handler()
```

ℹ️ The `load_suppressions` function is useful for setting this in library modules that are otherwise out of your control.

## Hooks

Hooks make TraceRite the default exception handler, so unhandled exceptions and `logging.exception()` are formatted automatically. Loaders can be called multiple times without further effects. Everything installed by so far gets restored on first unload - further unloads do nothing, and unloading at all is optional.

### load / unload

```python
load(*, hooks=True, suppressions=True, capture_logging=True) -> None
unload() -> None
```

```python
import tracerite

tracerite.load()  # Any unhandled exception now gets TraceRite formatting

tracerite.unload()  # Optionally restore the original handlers
```

`load()` combines the three loaders below; each can be turned off:

- **hooks** — replace `sys.excepthook` and `threading.excepthook`
- **suppressions** — set `__tracebackhide__` on selected standard library modules
- **capture_logging** — capture `logging.StreamHandler.emit` for `logging.exception()`

`unload()` restores all original handlers and attributes, regardless of which
loaders were used.

### Individual loaders

For finer control, use the loaders separately:

- **load_hooks() / unload_hooks()** — `sys.excepthook` and `threading.excepthook`.
- **load_logging_capture() / unload_logging_capture()** — capture `logging.StreamHandler.emit` for exceptions.
- **load_suppressions(extra=None) / unload_suppressions()** — set `__tracebackhide__` on library modules. The `extra` argument takes additional `{module_name: value}` mappings to apply alongside the built-in ones e.g. for the framework you are using.

### patch_fastapi

```python
patch_fastapi(*, tty=True) -> None
```

Monkey-patches Starlette's debug error handler so that FastAPI, in debug mode, answers HTML clients with a TraceRite page (via `html_page`) instead of the default debug HTML. It also hides FastAPI's own routing frames and, by default, loads the TTY hooks for console output. Call once when your app starts; it is a no-op when FastAPI/Starlette is not installed.

```python
from tracerite import patch_fastapi

patch_fastapi()          # HTML debug pages + TTY formatting
patch_fastapi(tty=False) # HTML debug pages only
```

## Chain extraction

### extract_chain

```python
extract_chain(exc=None, *, skip_outmost=0, skip_until=None) -> dict
```

Extracts traceback information without rendering anything. This is the JSON-compatible intermediate format that all formatters consume, and the right entry point if you want to build your own output. Note that the frames are returned in the chronological order of execution, where exception-raising frames may appear at any point (and at the end).

Parameters:

- **exc** — the exception to analyse. When omitted, the current exception inside an `except` block is used.
- **skip_outmost** — number of outermost frames to drop. Defaults to 0.
- **skip_until** — drop all frames before the first one whose filename contains this substring (the matching frame itself is kept). The notebook extension uses `"<ipython-input-"` to hide IPython internals.

The skip options are low level options applied early on the non-chronological trace of the most recent exception. In general you should prefer the chronological `__tracebackhide__` mechanism instead (see below).

#### Return value

```json
{
  "header": "⚠️  Uncaught TypeError",
  "frames": [
    {
      "id": "tb-XOO6MWWqUI6p4Z0z",
      "relevance": "call",
      "hidden": false,
      "idframe": 140582493173840,
      "filename": "dev/project/example_module.py",
      "original_filename": "/home/user/dev/project/example_module.py",
      "location": "project/example_module.py",
      "notebook_cell": false,
      "codeline": "risky()",
      "range": { "lfirst": 13, "lfinal": 13, "cbeg": 4, "cend": 11 },
      "lineno": 13,
      "cursor_line": 13,
      "cursor_col": 11,
      "linenostart": 4,
      "lines": "def risky():\n    result = (...)\n    ...",
      "fragments": [...],
      "function": null,
      "function_suffix": "",
      "urls": { "VS Code": "vscode://file/.../project/example_module.py:13:11" },
      "full_source": "from tracerite import extract_chain\n...",
      "full_source_start": 1,
      "variables": []
    },
    {
      "id": "tb-ioDJ-ymtUQqzJz08",
      "relevance": "error",
      "lineno": 6,
      "cursor_line": 7,
      "linenostart": 4,
      "function": "risky",
      "exception": {
        "type": "TypeError",
        "message": "can only concatenate str (not \"int\") to str",
        "summary": "can only concatenate str (not \"int\") to str",
        "from": "none",
        "exc_idx": 0
      },
      ...
    }
  ]
}
```

A dict with two keys:

- **header** — a combined summary of the whole chain, e.g. `"⚠️  Uncaught ValueError"`.
- **frames** — a chronological list of frame dicts: oldest call first, final error last.

Each frame dict contains some or all of the following:

- **id** — unique trace ID, used as the anchor for collapsible HTML frames.
- **relevance** — one of `"call"`, `"error"`, `"stop"`, `"warning"`, or `"except"`.
- **filename**, **original_filename**, **location** — source location.
- **lineno**, **linenostart**, **lines**, **fragments** — source code context.
- **range** — a dict `{"lfirst": ..., "lfinal": ..., "cbeg": ..., "cend": ...}`
  marking the emphasis region within the lines window, or `None`.
- **function**, **function_suffix** — function name, with `"⚡except"` as the suffix for promoted frames.
- **urls** — source links.
- **variables** — variable dicts as returned by `extract_variables` (see *Variable inspection*), populated only on the final occurrence of each unique frame.
- **exception** — an exception banner dict on the final frame of each exception:
- **parallel** — for `ExceptionGroup` / `BaseExceptionGroup` (Python 3.11+), a list of parallel branches, each a list of frame dicts.

Notice how the fields relate to each other:

- **filename** is the path resolved relative to the current working directory; **original_filename** keeps the absolute path from the traceback. **location** is the displayed location, which may be truncated or shortened for library code, or `In [N]` for ipython.
- **lineno** is the line where the traceback says execution is; **cursor_line** is the line TraceRite prefers to highlight. They may differ on multi-line expressions. **linenostart** is the first line of the source window shown for the frame.


## Variable inspection

The inspector formats the variable values shown with each frame. It is also
useful on its own — for debugging, logging, or building developer tools.

### prettyvalue

```python
prettyvalue(val) -> (value, format_hint)
```

Formats a single value with smart truncation and type-aware rendering, and
returns a pair: the formatted value and a hint for how to lay it out.

- `"inline"` — a compact single-line representation, meant to sit on the same
  line as the variable name. This is the default for almost everything.
- `"block"` — a multi-line string, meant to be shown as a preformatted block
  below the name (TraceRite's HTML uses a `<pre>` element). Only multi-line
  strings produce this.

The value is usually a string, but containers come back as structured data for
renderers to lay out: dicts and dataclasses as
`{"type": "keyvalue", "rows": [...]}`, and arrays as
`{"type": "array", "rows": ..., "suffix": ...}`.

```python
from tracerite import prettyvalue

value, format_hint = prettyvalue([1, 2, 3])
# value = "1, 2, 3"
# format_hint = "inline"
```

What different types get:

- **NumPy/PyTorch arrays** — shape, dtype, and device in the type name, values
  formatted with SI scaling.
- **Dicts and dataclasses** — structured key-value display.
- **Lists and tuples** — truncated with item counts.
- **Strings** — multi-line strings preserved as blocks, long strings truncated.
- **Scalars** — smart numeric formatting: integers without decimals, floats
  with minimal precision, `NaN` and `∞` where applicable.

### extract_variables

```python
extract_variables(variables, sourcecode, exc_message=None) -> list[dict]
```

Extracts and formats the variables relevant to a line of source code: only
names that actually appear in `sourcecode` are included. This is what populates
the per-frame `variables` in an extracted chain.

```python
from tracerite import extract_variables

x, y = 10, 20
variables = extract_variables(locals(), "x + y")
for var in variables:
    print(f"{var['name']}: {var['value']}")
# x: 10
# y: 20
```

A few refinements keep the output focused:

- Objects with an uninformative default `repr` are expanded into their members
  (`obj.attr`) when those members appear in the code.
- A string variable identical to the exception message is suppressed as
  redundant.
- The type name of arrays carries shape, dtype, and non-CPU device info.

Each returned dict has the keys **name** (possibly with a `.member` suffix),
**typename**, **value**, and **format_hint** — the same value/hint contract as
`prettyvalue`.

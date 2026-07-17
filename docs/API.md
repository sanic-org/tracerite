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

## Chain extraction

### extract_chain

```python
extract_chain(exc=None, *, skip_outmost=0, skip_until=None) -> dict
```

Extracts traceback information without rendering anything. This is the
JSON-compatible intermediate format that all formatters consume, and the right
entry point if you want to build your own output.

```python
from tracerite import extract_chain

try:
    risky_operation()
except Exception:
    data = extract_chain()
    print(data["header"])
    for frame in data["frames"]:
        if frame.get("exception"):
            print(frame["exception"]["type"], frame["exception"]["message"])
```

Parameters:

- **exc** — the exception to extract. When omitted, the current exception from
  `sys.exc_info()` is used, so a bare `extract_chain()` works inside an
  `except` block.
- **skip_outmost** — number of outermost frames (where execution began) to
  drop, e.g. to hide framework entry points. Defaults to 0.
- **skip_until** — drop all frames before the first one whose filename contains
  this substring (the matching frame itself is kept). The notebook extension
  uses `"<ipython-input-"` to hide IPython internals.

The skip options apply to the newest exception in the chain, which is normally
the one you caught.

#### Return value

A dict with two keys:

- **header** — a combined summary of the whole chain, e.g.
  `"⚠️  Uncaught ValueError"`.
- **frames** — a chronological list of frame dicts: oldest call first, final
  error last.

Each frame dict contains some or all of the following:

- **id** — unique trace ID, used as the anchor for collapsible HTML frames.
- **relevance** — one of `"call"`, `"error"`, `"stop"`, `"warning"`, or
  `"except"`.
- **filename**, **original_filename**, **location** — source location.
- **lineno**, **linenostart**, **lines**, **fragments** — source code context.
- **range** — a dict `{"lfirst": ..., "lfinal": ..., "cbeg": ..., "cend": ...}`
  marking the emphasis region within the lines window, or `None`.
- **function**, **function_suffix** — function name, with `"⚡except"` as the
  suffix for promoted frames.
- **urls** — source links.
- **variables** — variable dicts as returned by `extract_variables` (see
  *Variable inspection*), populated only on the final occurrence of each unique
  frame.
- **exception** — an exception banner dict on the final frame of each
  exception:

  ```python
  {
      "type": "ValueError",
      "message": "something went wrong",
      "summary": "something went wrong",
      "from": "cause",              # or "context" / "none"
      "exc_idx": 0,                 # index in the raw exception chain
      "leaf_types": ["KeyError"],   # only for ExceptionGroups
  }
  ```

- **parallel** — for `ExceptionGroup` / `BaseExceptionGroup` (Python 3.11+), a
  list of parallel branches, each a list of frame dicts.

## Formatting

All three formatters share the same first two parameters and pass any remaining
keyword arguments through to `extract_chain` (so `skip_outmost` and
`skip_until` work everywhere):

- **exc** — the exception to render. Defaults to the current exception from
  `sys.exc_info()`.
- **chain** — pre-extracted data from `extract_chain`. When given, **exc**
  is ignored.

### html_traceback

```python
html_traceback(exc=None, chain=None, *, msg=..., include_js_css=True,
               local_urls=False, clear=False, autodark=True, **extract_args)
```

Renders an exception as an interactive HTML fragment: an `html5tagger` tree
wrapped in `<div class="tracerite">`. Convert it with `str()` or drop it into
an f-string/template.

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
- **include_js_css** — include the TraceRite `<style>` and `<script>` tags in
  the fragment (default). Set to `False` when embedding in a page that already
  provides them.
- **local_urls** — link source locations to local files (VS Code / `file://`
  URLs) instead of anything web-facing. For local development.
- **clear** — replace previous TraceRite reports on the page instead of
  appending to them. Used by the notebook extension; only takes effect when the
  JavaScript is present.
- **autodark** — follow the system dark-mode preference (default). The
  fragment carries an `autodark` class that the stylesheet reacts to.

#### Web framework integration

```python
from sanic import Sanic, response
from tracerite import html_traceback

app = Sanic("MyApp")

@app.exception(Exception)
async def handle_error(request, exception):
    return response.html(str(html_traceback(exception)))
```

The same pattern works with FastAPI, Flask, or any framework that lets you
return HTML responses.

### html_page

```python
html_page(exc=None, *, title=None, heading=None, ingress=None, header=None,
          footer=None, msg=..., chain=None, autodark=True, local_urls=False,
          **extract_args)
```

Renders a complete HTML5 document containing a TraceRite traceback — a
ready-to-serve page rather than a fragment.

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

#### Web framework integration

```python
from fastapi import FastAPI
from fastapi.responses import HTMLResponse
from tracerite import html_page

app = FastAPI()

@app.exception_handler(Exception)
async def handle_error(request, exc):
    return HTMLResponse(html_page(exc, title="My App Error"), status_code=500)
```

For FastAPI there is also `patch_fastapi` (see *Hooks*), which installs this
behavior application-wide.

### tty_traceback

```python
tty_traceback(exc=None, chain=None, *, file=None, msg=None, tag="",
              term_width=None, **extract_args) -> None
```

Prints a colorful terminal traceback with ANSI colors and Unicode box drawing.
It writes directly to the terminal instead of returning a string, so it can
adapt to terminal features like the window size.

```python
from tracerite import tty_traceback

try:
    risky_operation()
except Exception:
    tty_traceback()  # Captures current exception, prints to stderr
```

Parameters (in addition to **exc** and **chain**):

- **file** — output destination, `sys.stderr` by default. Must support
  `write()` and `fileno()`.
- **msg** — the header line. By default it is built from the exception chain;
  pass a string to override it.
- **tag** — a short tag displayed after the header (e.g. `"#TR1"`), handy as a
  reference in bug reports.
- **term_width** — the width to wrap output to. Auto-detected from the terminal
  when omitted.

## Hooks

Hooks make TraceRite the default exception handler, so unhandled exceptions —
including those raised in threads — are formatted automatically. Everything
installed by a loader is restored by the matching unloader.

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

- **hooks** — replace `sys.excepthook` and `threading.excepthook` so uncaught
  exceptions (including in threads) print as TraceRite terminal tracebacks.
- **suppressions** — set `__tracebackhide__` on library modules such as
  `asyncio` and `importlib`, keeping their frames out of the output.
- **capture_logging** — patch `logging.StreamHandler.emit` so that
  `logging.exception()` formats the exception with TraceRite.

`unload()` restores all original handlers and attributes, regardless of which
loaders were used.

### Individual loaders

For finer control, use the loaders separately:

- **load_hooks() / unload_hooks()** — replace / restore `sys.excepthook` and
  `threading.excepthook`.
- **load_logging_capture() / unload_logging_capture()** — patch / restore
  `logging.StreamHandler.emit`.
- **load_suppressions(extra=None) / unload_suppressions()** — set / restore
  `__tracebackhide__` on library modules. The `extra` argument takes additional
  `{module_name: value}` mappings to apply alongside the built-in ones.

### patch_fastapi

```python
patch_fastapi(*, tty=True) -> None
```

Monkey-patches Starlette's debug error handler so that FastAPI, in debug mode,
answers HTML clients with a TraceRite page (via `html_page`) instead of the
default debug HTML. It also hides FastAPI's own routing frames and, by default,
loads the TTY hooks for console output. Call once when your app starts; it is a
no-op when FastAPI/Starlette is not installed.

```python
from tracerite import patch_fastapi

patch_fastapi()          # HTML debug pages + TTY formatting
patch_fastapi(tty=False) # HTML debug pages only
```

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

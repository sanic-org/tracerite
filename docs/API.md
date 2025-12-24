# TraceRite API

This document covers the public API for using TraceRite outside of Jupyter notebooks.

## `html_traceback`

Renders the current exception (or a provided one) as interactive HTML.

```python
from tracerite import html_traceback

try:
    risky_operation()
except Exception:
    html = html_traceback()  # Captures current exception
```

### Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `exc` | `Exception` | `None` | Exception to render. If `None`, uses the current exception from `sys.exc_info()`. |
| `chain` | `list` | `None` | Pre-extracted exception chain (from `extract_chain`). Overrides `exc`. |
| `include_js_css` | `bool` | `True` | Include `<style>` and `<script>` tags in output. Set to `False` if embedding in a page that already has TraceRite styles. |
| `local_urls` | `bool` | `False` | Use `file://` URLs for source links (for local development). |
| `skip_outmost` | `int` | `0` | Number of outermost frames to skip. |
| `skip_until` | `str` | `None` | Skip frames until one matches this substring (e.g., `"<ipython-input-"`). |

### Web Framework Integration

```python
from sanic import Sanic, response
from tracerite import html_traceback

app = Sanic("MyApp")

@app.exception(Exception)
async def handle_error(request, exception):
    return response.html(str(html_traceback(exception)))
```

The same pattern works with FastAPI, Flask, or any framework that lets you return HTML responses.

## `extract_chain`

Extracts exception information without rendering. Useful for logging or custom formatting.

```python
from tracerite import extract_chain

try:
    risky_operation()
except Exception:
    chain = extract_chain()
    for exc_info in chain:
        print(exc_info["type"], exc_info["message"])
```

## TTY / Terminal Output

For command-line applications, TraceRite provides colorful terminal tracebacks with ANSI colors and Unicode box drawing.

### `load` / `unload`

Load TraceRite as the default exception handler for all unhandled exceptions.

```python
import tracerite

# Load at startup
tracerite.load()

# Your code here - any unhandled exception will show TraceRite formatting

# Optionally restore original handlers
tracerite.unload()
```

This replaces both `sys.excepthook` and `threading.excepthook`, so exceptions in threads are also handled.

### `tty_traceback`

Renders an exception as colorful terminal output.

```python
from tracerite import tty_traceback

try:
    risky_operation()
except Exception:
    tty_traceback()  # Captures current exception, prints to stderr
```

### Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `exc` | `Exception` | `None` | Exception to render. If `None`, uses the current exception from `sys.exc_info()`. |
| `chain` | `list` | `None` | Pre-extracted exception chain (from `extract_chain`). Overrides `exc`. |
| `file` | file object | `sys.stderr` | Output destination. Must support `.write()` and `.fileno()`. |

## Variable Inspector

The inspector module formats Python values for display—useful beyond exception handling for debugging, logging, or building developer tools.

### `prettyvalue`

Formats a single value with smart truncation and type-aware rendering.

```python
from tracerite import prettyvalue

value, format_hint = prettyvalue([1, 2, 3])
# value = "1, 2, 3"
# format_hint = "inline"
```

Handles:
- **NumPy/PyTorch arrays** — Shows shape, dtype, device, and formatted values with SI scaling
- **Dicts and dataclasses** — Structured key-value display
- **Lists and tuples** — Truncated with item counts
- **Strings** — Multi-line preserved as blocks, long strings truncated
- **Scalars** — Smart numeric formatting (integers without decimals, floats with minimal precision)

### `extract_variables`

Extracts and formats variables relevant to a line of source code.

```python
from tracerite import extract_variables

x, y = 10, 20
variables = extract_variables(locals(), "x + y")
for var in variables:
    print(f"{var.name}: {var.value}")
# x: 10
# y: 20
```

Returns a list of `VarInfo` namedtuples with fields:
- `name` — Variable name (may include `.member` for object attributes)
- `typename` — Type with array shape/dtype info when applicable
- `value` — Formatted string or structured data for rendering
- `format_hint` — `"inline"` or `"block"`

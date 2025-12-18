# TraceRite

**Beautiful, readable error messages for Python notebooks and web frameworks.**

```ipython
%pip install tracerite
%load_ext tracerite
```

That's it! Your Jupyter notebook now has cleaner error messages.

## What It Looks Like

When an error occurs in a NumPy operation, TraceRite shows you exactly what went wrong:

![NumPy error with TraceRite](https://raw.githubusercontent.com/sanic-org/tracerite/master/docs/numpy.webp)

The error message highlights the problematic line, and the built-in variable inspector lets you see array shapes and values at a glance—no more guessing why your shapes don't match.

### Handling Complex Call Chains

Real-world code often involves deep call stacks through libraries. TraceRite intelligently collapses library internals while keeping your code front and center:

![Complex call chain](https://raw.githubusercontent.com/sanic-org/tracerite/main/docs/complex.webp)

### Nested Exceptions

When exceptions chain together, TraceRite keeps them organized and puts the most relevant exception on top to avoid any scrolling.

![Nested exceptions](https://raw.githubusercontent.com/sanic-org/tracerite/main/docs/numpydeep.webp)

Each frame includes a variable inspector so you can trace exactly how values flowed through your code.

### Interrupts Made Clear

Even a simple keyboard interrupt becomes informative—TraceRite shows exactly where you stopped execution:

![Call chain interrupt](https://raw.githubusercontent.com/sanic-org/tracerite/main/docs/callchain.webp)

## Features

- **Smart frame filtering** — Shows library internals but focuses on where the flow touched your code
- **Variable inspection** — See the values of your variables in a pretty printed HTML format
- **JSON output** - Intermediady dict format is JSON-compatible, useful for machine processing and used by our HTML module.
- **HTML output** — Works in Jupyter, Colab, and web frameworks such as FastAPI and Sanic as the debug mode error handler.

## Usage

### `html_traceback(exc)`

Renders an exception as interactive HTML. Pass an exception object, or call with no arguments inside an `except` block to use the current exception.

### `extract_chain(exc)`

Extracts exception information as a list of dictionaries—useful for logging, custom formatting, or machine processing.

### `tracerite.inspector`

The inspector module formats Python values for display. Useful beyond exceptions for debugging tools or custom logging:

- `prettyvalue(value)` — Formats any value with smart truncation, array shape display, and SI-scaled numerics
- `extract_variables(locals, source)` — Extracts variables mentioned in a line of code

See the [API documentation](https://github.com/sanic-org/tracerite/blob/main/docs/API.md) for details, or [Development guide](https://github.com/sanic-org/tracerite/blob/main/docs/DEVELOPMENT.md) for contributors.

## License

Public Domain or equivalent.

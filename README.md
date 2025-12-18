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

### Long Call Chains

Often your simple function call dwelves deep into the library code. In this situation TraceRite highlights the position where it happened in your code, but allows inspecting the library internals also, to get to the root of it. In this case the variable inspector on the first frame already shows what is the problem: we are accidentally passing in `None` for file extension.

![Call chain into library code](https://raw.githubusercontent.com/sanic-org/tracerite/main/docs/callchain.webp)

## Features

- **Only relevant things shown** - Shows library internals but focuses on where the flow touched your code, and the variables used at the spot.
- **Variable inspection** - See the values of your variables in a pretty printed HTML format (or JSON-compatible machine-readable dict)
- **JSON output** - Intermediady dict format is JSON-compatible, useful for machine processing and used by our HTML module.
- **HTML output** - Works in Jupyter, Colab, and web frameworks such as FastAPI and Sanic as the debug mode error handler.
- **Custom CSS** - Implement dark mode or custom look like that in the Sanic Framework with CSS variable overrides.

## Usage

### `html_traceback(exc)`

Renders an exception as interactive HTML that can be included on a page. Pass an exception object, or call with no arguments inside an `except` block to use the current exception.

### `extract_chain(exc)`

Extracts exception information as a list of dictionaries—useful for logging, custom formatting, or machine processing.

### `prettyvalue(value)`

Formats any value with smart truncation, array shape display, and SI-scaled numerics. Useful beyond exceptions for debugging tools or custom logging.

### `extract_variables(locals, source)`

Extracts and formats variables mentioned in a line of source code.

See the [API documentation](https://github.com/sanic-org/tracerite/blob/main/docs/API.md) for details, or [Development guide](https://github.com/sanic-org/tracerite/blob/main/docs/Development.md) for contributors.

## License

Public Domain or equivalent.

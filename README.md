# Tracebacks for Humans (and Machines)

**Fixing bugs is easier with well formatted error messages. Console, HTML and JSON.**

[![PyPI version](https://badge.fury.io/py/tracerite.svg)](https://pypi.org/project/tracerite/)
![Tests](https://raw.githubusercontent.com/sanic-org/tracerite/main/docs/img/tests-badge.svg)
![Coverage](https://raw.githubusercontent.com/sanic-org/tracerite/main/docs/img/coverage-badge.svg)

![TraceRite in Notebook/HTML and terminal](https://raw.githubusercontent.com/sanic-org/tracerite/main/docs/screenshots/features-composite.webp)
*TraceRite formats Python errors in Jupyter notebooks, in plain HTML, and in the terminal: rich variable inspection (left), plus syntax-error highlighting and compact terminal tracebacks (right).*

## Installation

### Python scripts or REPL

```sh
pip install tracerite
```

```python
import tracerite; tracerite.load()
```

Any error message after that call will be prettified. Handles any syntax errors and uncaught exceptions, even captures `logging.exception` (optionally).

### IPython or Jupyter Notebook

```ipython
%pip install tracerite
%load_ext tracerite
```

This enables tracebacks in text or HTML format depending on where you are running. Add to `~/.ipython/profile_default/startup/tracerite.ipy` to make it load automatically for all your ipython and notebook sessions. Alternatively, put the two lines at the top of your notebook.

If you are using UV, consider running with your other dependencies:
```sh
uvx --with tracerite --with numpy jupyter lab
```

### FastAPI

Add the extension loader at the top of your app module:

```python
from tracerite import patch_fastapi; patch_fastapi()
```

This monkeypatches Starlette error handling and FastAPI routing to work with HTML tracebacks. Note: this runs regardless of whether you are in production mode or debug mode, so you might want to call that function only conditionally in the latter.

### Sanic

Comes with TraceRite built in. HTML reports are available in debug mode, and console messages are always formatted by Tracerite.

## Clarity in complex situations

![Exception chain comparison](https://raw.githubusercontent.com/sanic-org/tracerite/main/docs/screenshots/chain-comparison.webp)
*TraceRite renders complex exception chains in chronological order (left). The Python 3.15 built in traceback needs illustrative arrows to follow execution (right).*

## Features

- **Chronological order** - Single timeline with the program entry point is at top, and the finally uncaught exception bottom.
- **Minimalistic output** - Smart pruning to show only relevant pieces of information, excluding library internals where not relevant and avoiding any repetition.
- **ExceptionGroups** - Full tracebacks of the subexceptions from exceptions that occurred in parallel execution.
- **Variable inspection** - See the values of your variables in a pretty printed HTML format, Terminal or JSON-compatible machine-readable dict.
- **JSON output** - Intermediady dict format is JSON-compatible, useful for machine processing and used by our HTML and TTY modules.
- **HTML output** - Works in Jupyter, Colab, and web frameworks such as FastAPI and Sanic as the debug mode error handler.
- **TTY output** - Colorful, formatted tracebacks for terminal applications and Python REPL.
- **Custom Styling** - Theme with your colors by defining CSS variables or tty.COLORS.
- **Automatic dark mode** - Saves your eyes.

![ExceptionGroup in HTML and terminal](https://raw.githubusercontent.com/sanic-org/tracerite/main/docs/screenshots/group-comparison.webp)
*Python 3.11+ concurrent execution errors with `except*` handling are shown clearly in both HTML and terminal output.*

## Documentation

For the public API — including HTML rendering, terminal output, and the machine-readable chain format returned by `extract_chain` — see the [API documentation](https://github.com/sanic-org/tracerite/blob/main/docs/API.md). For contributors, see the [Development guide](https://github.com/sanic-org/tracerite/blob/main/docs/Development.md).

## License

Public Domain or equivalent.

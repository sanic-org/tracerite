# Development Guide

Instructions for contributing to TraceRite.

## Setup

1. Install [uv](https://docs.astral.sh/uv/getting-started/installation/)

2. Clone and set up:
   ```bash
   git clone https://github.com/sanic-org/tracerite.git
   cd tracerite
   uv sync
   ```

3. Activate the virtual environment (optional, to skip `uv run` prefix):
   ```bash
   source .venv/bin/activate
   ```

## Nox Tasks

The project uses [Nox](https://nox.thea.codes/) for task automation:

```bash
uv run nox          # Run default tasks (format, lint, test)
uv run nox -l       # List all available tasks
uv run nox -s tests # Run only tests
uv run nox -s lint  # Run only linting
```

## Project Structure

| Module | Purpose |
|--------|---------|
| `trace.py` | Traceback extraction and frame processing |
| `html.py` | HTML rendering |
| `tty.py` | Terminal (TTY) rendering with ANSI colors |
| `inspector.py` | Variable formatting |
| `notebook.py` | Jupyter/IPython integration |
| `logging.py` | Python logging integration |

## Running the Demo

The `sanic_demo.py` file provides a test server with various error scenarios:

```bash
uvx --with-editable . sanic sanic_demo:app --dev
```

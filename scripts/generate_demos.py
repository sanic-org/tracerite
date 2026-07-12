#!/usr/bin/env -S uv run
# /// script
# requires-python = ">=3.11"
# dependencies = [ "tracerite", "nbclient", "ipykernel", "nbformat" ]
# tool.uv.sources.tracerite = { path = "../", editable = true }
# ///
"""Generate FastAPI, Sanic and notebook demos from ``demo.helpers.scenarios``.

Run from the repository root:

    uv run scripts/generate_demos.py

This script introspects ``demo/helpers/scenarios.py`` and writes explicit
handler functions for every scenario instead of the previous auto-discovered
``_handler`` loop.  HTML index pages are built with ``html5tagger``.
"""

from __future__ import annotations

import argparse
import ast
import inspect
import textwrap
from pathlib import Path
from typing import Any

import nbclient
import nbformat

from demo.helpers import discover_scenarios


class _AsyncioRunToAwait(ast.NodeTransformer):
    """Rewrite ``asyncio.run(coroutine())`` to ``await coroutine()``."""

    def visit_Expr(self, node: ast.Expr) -> ast.AST:
        value = node.value
        if isinstance(value, ast.Call) and self._is_asyncio_run(value):
            return ast.Expr(value=ast.Await(value=value.args[0]))
        return self.generic_visit(node)

    @staticmethod
    def _is_asyncio_run(call: ast.Call) -> bool:
        func = call.func
        return (
            isinstance(func, ast.Attribute)
            and isinstance(func.value, ast.Name)
            and func.value.id == "asyncio"
            and func.attr == "run"
            and len(call.args) == 1
        )


def _handler_body_source(func: Any) -> str:
    """Return the scenario body for a live handler.

    If the scenario exposes an ``_async_impl`` helper, the handler awaits it
    directly (e.g. ``await acme.run_concurrent_tasks()``).  Otherwise the sync
    body is copied and ``asyncio.run(...)`` is rewritten to ``await ...``.
    """
    async_impl = getattr(func, "_async_impl", None)
    if async_impl is not None:
        return f"await acme.{async_impl.__name__}()"

    source = textwrap.dedent(inspect.getsource(func))
    tree = ast.parse(source)
    func_def = tree.body[0]
    assert isinstance(func_def, ast.FunctionDef)
    body = func_def.body

    # Strip the docstring; it will be attached to the generated handler.
    if (
        body
        and isinstance(body[0], ast.Expr)
        and isinstance(body[0].value, ast.Constant)
        and isinstance(body[0].value.value, str)
    ):
        body = body[1:]

    body = [_AsyncioRunToAwait().visit(stmt) for stmt in body]
    for stmt in body:
        ast.fix_missing_locations(stmt)

    return "\n".join(ast.unparse(stmt) for stmt in body)


def _fastapi_handlers(items: list[tuple[str, Any]]) -> str:
    parts: list[str] = []
    for name, func in items:
        body = _handler_body_source(func)
        doc = (func.__doc__ or "").strip()
        parts.append(
            f'''@app.get("/{name}")
async def {name}():
    """{doc}"""
{textwrap.indent(body, "    ")}
'''
        )
    return "\n".join(parts)


def _sanic_handlers(items: list[tuple[str, Any]]) -> str:
    parts: list[str] = []
    for name, func in items:
        body = _handler_body_source(func)
        doc = (func.__doc__ or "").strip()
        parts.append(
            f'''@app.get("/{name}")
async def {name}(request):
    """{doc}"""
{textwrap.indent(body, "    ")}
'''
        )
    return "\n".join(parts)


def _startup_handler_source(for_sanic: bool = False) -> str:
    if for_sanic:
        return """@app.before_server_start
async def _build_index(app):
    app.ctx.index_html = await build_index_html()
"""
    return """@app.on_event("startup")
async def _build_index():
    app.state.index_html = await build_index_html()
"""


def _fastapi_app(items: list[tuple[str, Any]]) -> str:
    return rf'''#!/usr/bin/env -S uv run
# /// script
# requires-python = ">=3.11"
# dependencies = [ "fastapi[standard]", "tracerite" ]
# tool.uv.sources.tracerite = {{ path = "../", editable = true }}
# ///

import importlib
import json
import re
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.responses import HTMLResponse

from demo.helpers import acme
from demo.helpers.html import build_index_html
from demo.helpers.types import Bar, Foo
from tracerite.fastapi import patch_fastapi

patch_fastapi()

@asynccontextmanager
async def _lifespan(app):
    app.state.index_html = await build_index_html()
    yield

app = FastAPI(title="TraceRiteDemo", debug=True, lifespan=_lifespan)


@app.get("/")
async def index():
    """Main index page with scenario previews."""
    return HTMLResponse(app.state.index_html)

{_fastapi_handlers(items)}

if __name__ == "__main__":
    import uvicorn

    uvicorn.run("demo.fastapi_demo:app", host="localhost", port=0, reload=True)
'''


def _sanic_app(items: list[tuple[str, Any]]) -> str:
    return rf'''#!/usr/bin/env -S uv run
# /// script
# requires-python = ">=3.11"
# dependencies = [ "sanic" ]
# tool.uv.sources.tracerite = {{ path = "../", editable = true }}
# ///

import importlib
import json
import re

from sanic import Sanic, response

from demo.helpers import acme
from demo.helpers.html import build_index_html
from demo.helpers.types import Bar, Foo

app = Sanic("TraceRiteDemo")

{_startup_handler_source(for_sanic=True)}
@app.get("/")
async def index(request):
    """Main index page with scenario previews."""
    return response.html(request.app.ctx.index_html)

{_sanic_handlers(items)}

if __name__ == "__main__":
    app.run(host="localhost", port=8765, debug=True, auto_reload=True)
'''


def _notebook_setup_cells() -> list[Any]:
    return [
        nbformat.v4.new_code_cell(source="%load_ext tracerite\n%tracerite keep"),
        nbformat.v4.new_code_cell(
            source=(
                "import importlib\n"
                "import json\n"
                "import re\n"
                "import asyncio\n"
                "\n"
                "from demo.helpers import acme\n"
                "from demo.helpers.types import Bar, Foo\n"
            )
        ),
    ]


def _build_notebook(items: list[tuple[str, Any]]) -> nbformat.NotebookNode:
    cells: list[Any] = _notebook_setup_cells()
    for name, func in items:
        doc = (func.__doc__ or "").strip()
        cells.append(nbformat.v4.new_markdown_cell(f"## {name.capitalize()}\n\n{doc}"))
        cells.append(nbformat.v4.new_code_cell(source=_handler_body_source(func)))
    return nbformat.v4.new_notebook(cells=cells)


def _execute_notebook(notebook: nbformat.NotebookNode) -> nbformat.NotebookNode:
    client = nbclient.NotebookClient(notebook, timeout=60, allow_errors=True)
    client.execute()
    return notebook


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate FastAPI/Sanic demo apps and a scenarios notebook."
    )
    parser.add_argument(
        "--fastapi-path",
        default="demo/fastapi_demo.py",
        help="Output path for the FastAPI demo app (default: demo/fastapi_demo.py)",
    )
    parser.add_argument(
        "--sanic-path",
        default="demo/sanic_demo.py",
        help="Output path for the Sanic demo app (default: demo/sanic_demo.py)",
    )
    parser.add_argument(
        "--notebook-path",
        default="demo/Scenarios.ipynb",
        help="Output path for the generated notebook (default: demo/Scenarios.ipynb)",
    )
    args = parser.parse_args()

    items = discover_scenarios()
    if not items:
        raise SystemExit("No scenarios found in demo.helpers.scenarios")

    fastapi_path = Path(args.fastapi_path)
    sanic_path = Path(args.sanic_path)
    notebook_path = Path(args.notebook_path)

    fastapi_path.write_text(_fastapi_app(items), encoding="utf-8")
    fastapi_path.chmod(0o755)
    sanic_path.write_text(_sanic_app(items), encoding="utf-8")
    sanic_path.chmod(0o755)
    notebook = _execute_notebook(_build_notebook(items))
    notebook_path.write_text(nbformat.writes(notebook), encoding="utf-8")

    print(f"Wrote {fastapi_path}")
    print(f"Wrote {sanic_path}")
    print(f"Wrote {notebook_path}")


if __name__ == "__main__":
    main()

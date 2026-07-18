#!/usr/bin/env -S uv run
# /// script
# requires-python = ">=3.11"
# dependencies = [ "tracerite", "nbformat", "numpy" ]
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
import asyncio
import inspect
import textwrap
from pathlib import Path
from typing import Any

import nbformat

from demo.helpers import discover_scenarios
from demo.helpers.html import build_index_html


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

    Source formatting is preserved for statements that do not need rewriting;
    only transformed statements fall back to ``ast.unparse``.
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

    transformed = [_AsyncioRunToAwait().visit(stmt) for stmt in body]
    for stmt in transformed:
        ast.fix_missing_locations(stmt)

    def _dedent_by_col_offset(segment: str, col_offset: int) -> str:
        lines = segment.splitlines()
        dedented: list[str] = []
        for line in lines:
            stripped = (
                line[col_offset:]
                if line.startswith(" " * col_offset)
                else line.lstrip()
            )
            dedented.append(stripped)
        return "\n".join(dedented)

    parts: list[str] = []
    for original, tx in zip(body, transformed, strict=False):
        if ast.dump(original) == ast.dump(tx):
            segment = ast.get_source_segment(source, original)
            assert segment is not None
            parts.append(_dedent_by_col_offset(segment, original.col_offset))
        else:
            parts.append(ast.unparse(tx))

    return "\n".join(parts)


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
    app.ctx.index_html = await build_index_html(framework="Sanic")
"""
    return """@app.on_event("startup")
async def _build_index():
    app.state.index_html = await build_index_html()
"""


def _fastapi_app(items: list[tuple[str, Any]]) -> str:
    return rf'''#!/usr/bin/env -S uv run
# /// script
# requires-python = ">=3.11"
# dependencies = [ "fastapi[standard]", "tracerite", "numpy" ]
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
    app.state.index_html = await build_index_html(framework="FastAPI")
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
# dependencies = [ "sanic", "numpy" ]
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
        nbformat.v4.new_markdown_cell(
            "# Python tracebacks for Humans (and Machines)\n\n"
            "You are viewing the TraceRite Jupyter Notebook demo. The examples below show rendered reports for various error scenarios when you run the cells."
        ),
        nbformat.v4.new_code_cell(source="%load_ext tracerite\n%tracerite keep"),
        nbformat.v4.new_code_cell(
            source=(
                "import importlib\n"
                "import json\n"
                "import re\n"
                "\n"
                "from demo.helpers import acme\n"
                "from demo.helpers.types import Bar, Foo"
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


def _assign_deterministic_cell_ids(
    notebook: nbformat.NotebookNode,
) -> nbformat.NotebookNode:
    """Assign stable cell IDs so regenerated notebooks diff cleanly."""
    for i, cell in enumerate(notebook.cells):
        cell.id = f"cell-{i}"
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
        default="demo/Notebook.ipynb",
        help="Output path for the generated notebook (default: demo/Notebook.ipynb)",
    )
    parser.add_argument(
        "--demo-html-path",
        default="demo/Demo.html",
        help="Output path for the standalone HTML demo (default: demo/Demo.html)",
    )
    args = parser.parse_args()

    items = discover_scenarios()
    if not items:
        raise SystemExit("No scenarios found in demo.helpers.scenarios")

    fastapi_path = Path(args.fastapi_path)
    sanic_path = Path(args.sanic_path)
    notebook_path = Path(args.notebook_path)
    demo_html_path = Path(args.demo_html_path)

    fastapi_path.write_text(_fastapi_app(items), encoding="utf-8")
    fastapi_path.chmod(0o755)
    sanic_path.write_text(_sanic_app(items), encoding="utf-8")
    sanic_path.chmod(0o755)
    notebook = _build_notebook(items)
    notebook = _assign_deterministic_cell_ids(notebook)
    notebook_path.write_text(nbformat.writes(notebook), encoding="utf-8")
    demo_html_path.write_text(asyncio.run(build_index_html()), encoding="utf-8")

    print(f"Wrote {fastapi_path}")
    print(f"Wrote {sanic_path}")
    print(f"Wrote {notebook_path}")
    print(f"Wrote {demo_html_path}")


if __name__ == "__main__":
    main()

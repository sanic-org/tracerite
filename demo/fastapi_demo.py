#!/usr/bin/env -S uv run
# /// script
# requires-python = ">=3.11"
# dependencies = [ "fastapi[standard]", "tracerite", "numpy" ]
# tool.uv.sources.tracerite = { path = "../", editable = true }
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

@app.get("/syntax")
async def syntax():
    """SyntaxError while importing a malformed module."""
    importlib.import_module("demo.helpers.broken_syntax")

@app.get("/inspector")
async def inspector():
    """Show variables at crash site."""
    acme.perform_calculation(
        Foo(),
        Bar(),
        divisor=0,
        multiplier=5,
    )

@app.get("/numpy")
async def numpy():
    """Inspector tensor matrix pretty printing (Numpy, Torch and others)."""
    import numpy as np
    a = np.diag((1, 1.5, np.pi))
    rng = np.random.default_rng(42)
    b = rng.normal(size=(4, 3)) * 1e6
    a[1, 0] = 0.01
    a[2, 0] = 1e-6
    a[0, 1] = float("NaN")
    a[1, 2] = float("inf")
    a[2, 1] = float("-inf")
    _ = a @ b

@app.get("/recursion")
async def recursion():
    """Deep recursion shortened."""
    acme.recurse(0)

@app.get("/chain")
async def chain():
    """Chronological cause chain with three exceptions."""
    acme.outerstep()

@app.get("/callfrom")
async def callfrom():
    """Error occurs in library code, but we show call site in user code."""
    try:
        foo = json.loads(
            '{"host": "example.com" "port": 80}'
        )
    except json.JSONDecodeError as e:
        raise RuntimeError("Configuration is malformed") from e

@app.get("/callback")
async def callback():
    """Call chain via library code."""
    re.sub(r"\d+", acme.regex_sub_callback, "50 0 25")

@app.get("/comprehension")
async def comprehension():
    """Crash inside a list comprehension."""
    _ = [100 // x for x in (10, 5, 0, 2)]

@app.get("/longmsg")
async def longmsg():
    """Long exception messages word wrapped and shortened."""
    try:
        msg = (
            "The configuration validation failed because the supplied manifest references "
            "several deprecated fields and contains sections that cannot be parsed automatically, "
            "so you will need to review them manually before the deployment can continue safely."
        )
        raise ValueError(msg)
    except ValueError as e:
        details = (
            "Configuration validation failed for the requested pipeline.\n"
            "\n"
            "The supplied manifest references several deprecated fields and "
            "contains a few sections that cannot be parsed automatically, so "
            "you will need to review them manually before the deployment can "
            "continue safely.\n"
            "\n"
            "Offending values:\n"
            "- `metadata.labels['app.kubernetes.io/very-long-component-name']` "
            "exceeds the maximum allowed length of 63 characters\n"
            "- `spec.template.spec.containers[0].resources.limits.cpu` is set to "
            "`1000000000000000000000000000000000000000000000000000000m` which is not a valid quantity\n"
            "- `spec.template.spec.containers[0].image` uses tag `latest`\n"
            "- `spec.replicas` is `0` which disables the service entirely\n"
            "\n"
            "Suggested fix:\n"
            "```python\n"
            "config = load_manifest('deployment.yaml')\n"
            "config['metadata']['labels']['app.kubernetes.io/component'] = 'api'\n"
            "config['spec']['replicas'] = max(1, config['spec']['replicas'])\n"
            "validate_and_apply(config)\n"
            "```\n"
            "\n"
            "For additional context, the full set of validation errors encountered "
            "while scanning the manifest is listed below. Each error includes the "
            "field path, the offending value, and a short explanation of why the "
            "value was rejected by the schema validator.\n"
            "\n"
            + "\n".join(
                f"[{i:03d}] validation error in field `spec.paths.{i}.method`: method "
                f"name is too long and contains invalid characters"
                for i in range(80)
            )
        )
        raise RuntimeError(details) from e

@app.get("/concurrent")
async def concurrent():
    """Async tasks failing in parallel, except* handling."""
    await acme.run_concurrent_tasks()


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("demo.fastapi_demo:app", host="localhost", port=0, reload=True)

#!/usr/bin/env -S uv run
# /// script
# requires-python = ">=3.11"
# dependencies = [ "sanic", "numpy" ]
# tool.uv.sources.tracerite = { path = "../", editable = true }
# ///

import importlib
import json
import re

from sanic import Sanic, response

from demo.helpers import acme
from demo.helpers.html import build_index_html
from demo.helpers.types import Bar, Foo

app = Sanic("TraceRiteDemo")

@app.before_server_start
async def _build_index(app):
    app.ctx.index_html = await build_index_html()

@app.get("/")
async def index(request):
    """Main index page with scenario previews."""
    return response.html(request.app.ctx.index_html)

@app.get("/syntax")
async def syntax(request):
    """SyntaxError while importing a malformed module."""
    importlib.import_module('demo.helpers.broken_syntax')

@app.get("/inspector")
async def inspector(request):
    """Show variables at crash site."""
    acme.perform_calculation(Foo(), Bar(), divisor=0, multiplier=5)

@app.get("/numpy")
async def numpy(request):
    """Inspector tensor matrix pretty printing (Numpy, Torch and others)."""
    import numpy as np
    a = np.diag((1, 1.5, np.pi))
    rng = np.random.default_rng(42)
    b = rng.normal(size=(4, 3)) * 1000000.0
    a[1, 0] = 0.01
    a[2, 0] = 1e-06
    a[0, 1] = float('NaN')
    a[1, 2] = float('inf')
    a[2, 1] = float('-inf')
    _ = a @ b

@app.get("/recursion")
async def recursion(request):
    """Deep recursion shortened."""
    acme.recurse(0)

@app.get("/chainmsg")
async def chainmsg(request):
    """Two chained exceptions, each with a multi-line message."""
    try:
        raise ValueError('Original problem\nwith extra detail')
    except ValueError as e:
        raise RuntimeError('While handling the original error\na second failure occurred.\nTerminating!')

@app.get("/causechain")
async def causechain(request):
    """Chronological cause chain with three exceptions."""
    acme.outerstep()

@app.get("/callfrom")
async def callfrom(request):
    """Error occurs in library code, but we show call site in user code."""
    try:
        foo = json.loads('{"host": "example.com" "port": 80}')
    except json.JSONDecodeError as e:
        raise RuntimeError('Configuration is malformed') from e

@app.get("/callback")
async def callback(request):
    """Call chain via library code."""
    re.sub('\\d+', acme.regex_sub_callback, '50 0 25')

@app.get("/comprehension")
async def comprehension(request):
    """Crash inside a list comprehension."""
    _ = [100 // x for x in (10, 5, 0, 2)]

@app.get("/longmsg")
async def longmsg(request):
    """Very long exception message mixing prose, code blocks, and many lines."""
    msg = "Configuration validation failed for the requested pipeline.\n\nThe supplied manifest references several deprecated fields and contains a few sections that cannot be parsed automatically, so you will need to review them manually before the deployment can continue safely.\n\nOffending values:\n- `metadata.labels['app.kubernetes.io/very-long-component-name']` exceeds the maximum allowed length of 63 characters\n- `spec.template.spec.containers[0].resources.limits.cpu` is set to `1000000000000000000000000000000000000000000000000000000m` which is not a valid quantity\n- `spec.template.spec.containers[0].image` uses tag `latest`\n- `spec.replicas` is `0` which disables the service entirely\n\nSuggested fix:\n```python\nconfig = load_manifest('deployment.yaml')\nconfig['metadata']['labels']['app.kubernetes.io/component'] = 'api'\nconfig['spec']['replicas'] = max(1, config['spec']['replicas'])\nvalidate_and_apply(config)\n```\n\nFor additional context, the full set of validation errors encountered while scanning the manifest is listed below. Each error includes the field path, the offending value, and a short explanation of why the value was rejected by the schema validator.\n\n" + '\n'.join((f'[{i:03d}] validation error in field `spec.paths.{i}.method`: method name is too long and contains invalid characters' for i in range(80)))
    raise ValueError(msg)

@app.get("/concurrent")
async def concurrent(request):
    """Async tasks failing in parallel, except* handling."""
    await acme.run_concurrent_tasks()


if __name__ == "__main__":
    app.run(host="localhost", port=8765, debug=True, auto_reload=True)

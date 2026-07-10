"""FastAPI Demo Application for Testing Tracebacks

This application demonstrates various error scenarios to test traceback
rendering with TraceRite. Run with: fastapi dev demo/fastapi_demo.py

Or within source repository:
  uvx --with-editable . fastapi dev demo/fastapi_demo.py
"""

from __future__ import annotations

from fastapi import FastAPI
from fastapi.responses import HTMLResponse, JSONResponse

from demo.helpers import async_tasks, calc, data, objects
from tracerite.fastapi import patch_fastapi

# Load TraceRite extension before creating the app
patch_fastapi()

app = FastAPI(title="TraceRiteDemo", debug=True)


@app.get("/", response_class=HTMLResponse)
async def index() -> str:
    """Main index page with links to all error scenarios."""
    return """
    <!DOCTYPE html>
    <html>
    <head>
        <title>FastAPI TraceRite Demo</title>
        <style>
            body { font-family: Arial, sans-serif; max-width: 800px; margin: 50px auto; padding: 20px; }
            h1 { color: #333; }
            .error-link { display: block; padding: 10px; margin: 10px 0;
                          background: #f0f0f0; text-decoration: none; color: #0066cc;
                          border-radius: 5px; }
            .error-link:hover { background: #e0e0e0; }
            code { background: #f5f5f5; padding: 2px 5px; border-radius: 3px; }
        </style>
    </head>
    <body>
        <h1>TraceRite FastAPI Demo</h1>
        <p>Click any link below to trigger different types of errors:</p>

        <a href="/deep-api-pipeline" class="error-link">
            <strong>Deep API Pipeline</strong> - Cross-module call chain ending in division by zero
        </a>

        <a href="/function-call" class="error-link">
            <strong>Multi-line Function Call</strong> - Error in a function called with wrapped arguments
        </a>

        <a href="/order-processing" class="error-link">
            <strong>Order Processing</strong> - Domain error wrapping a low-level failure
        </a>

        <a href="/chained-pipeline" class="error-link">
            <strong>Chained Pipeline</strong> - Three-level exception cause chain
        </a>

        <a href="/config-load" class="error-link">
            <strong>Config Load</strong> - Malformed JSON parsed in another module
        </a>

        <a href="/callback-error" class="error-link">
            <strong>Callback Error</strong> - Failure inside a stdlib regex callback
        </a>

        <a href="/record-batch" class="error-link">
            <strong>Record Batch</strong> - Batch processing combining JSON parsing and calculation
        </a>

        <a href="/variable-inspector" class="error-link">
            <strong>Variable Inspector</strong> - Objects with good and poor representations
        </a>

        <a href="/string-concat" class="error-link">
            <strong>String Concatenation</strong> - TypeError in multi-line string concat
        </a>

        <a href="/concurrent-failures" class="error-link">
            <strong>Concurrent Failures</strong> - Multiple failures in an asyncio TaskGroup
        </a>
    </body>
    </html>
    """


@app.get("/deep-api-pipeline")
async def deep_api_pipeline() -> JSONResponse:
    """Trigger error through a deep cross-module call chain."""
    result = calc.process_user_data(500)
    return JSONResponse({"result": result})


@app.get("/function-call")
async def function_call() -> JSONResponse:
    """Trigger error in a multi-line function call."""
    result = calc.complex_function_call()
    return JSONResponse({"result": result})


@app.get("/order-processing")
async def order_processing() -> JSONResponse:
    """Trigger a domain error wrapping a low-level failure."""
    result = calc.process_order(42)
    return JSONResponse({"result": result})


@app.get("/chained-pipeline")
async def chained_pipeline() -> JSONResponse:
    """Trigger a three-level exception cause chain."""
    calc.run_chained_pipeline()
    return JSONResponse({"result": "ok"})


@app.get("/config-load")
async def config_load() -> JSONResponse:
    """Trigger a JSON parsing error across modules."""
    result = data.load_config('{"host": "example.com" "port": 80}')
    return JSONResponse({"result": result})


@app.get("/callback-error")
async def callback_error() -> JSONResponse:
    """Trigger an error inside a stdlib regex callback."""
    result = data.apply_regex_discounts("50 0 25")
    return JSONResponse({"result": result})


@app.get("/record-batch")
async def record_batch() -> JSONResponse:
    """Trigger an error while batch-processing records."""
    records = [
        '{"value": 100, "divisor": 10}',
        '{"value": 100, "divisor": 0}',
    ]
    result = data.process_records(records)
    return JSONResponse({"result": result})


@app.get("/variable-inspector")
async def variable_inspector() -> JSONResponse:
    """Trigger an error with interesting variables in scope."""
    result = objects.inspect_variables()
    return JSONResponse({"result": result})


@app.get("/string-concat")
async def string_concat() -> JSONResponse:
    """Trigger a TypeError in multi-line string concatenation."""
    result = objects.build_greeting("World", 123)
    return JSONResponse({"result": result})


@app.get("/concurrent-failures")
async def concurrent_failures() -> JSONResponse:
    """Trigger an ExceptionGroup of concurrent failures."""
    await async_tasks.run_concurrent_tasks()
    return JSONResponse({"result": "ok"})


if __name__ == "__main__":
    print("This app should be run using the FastAPI CLI:")
    print("  fastapi dev demo/fastapi_demo.py")
    print("\nOr with uvicorn directly:")
    print("  uvicorn demo.fastapi_demo:app --reload")

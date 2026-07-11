"""FastAPI Demo Application for Testing Tracebacks

This application demonstrates various error scenarios to test traceback
rendering with TraceRite. Run with: fastapi dev demo/fastapi_demo.py

Or within source repository:
  uvx --with-editable . fastapi dev demo/fastapi_demo.py
"""

from __future__ import annotations

import inspect

from fastapi import FastAPI
from fastapi.responses import HTMLResponse, JSONResponse

from demo.helpers import scenarios
from tracerite.fastapi import patch_fastapi

# Load TraceRite extension before creating the app
patch_fastapi()

app = FastAPI(title="TraceRiteDemo", debug=True)


def _discover() -> list[tuple[str, object]]:
    """Return all public functions from the scenarios module in definition order."""
    return [
        (name, func)
        for name, func in vars(scenarios).items()
        if (
            not name.startswith("_")
            and inspect.isfunction(func)
            and func.__module__ == scenarios.__name__
        )
    ]


def _index_html(scenario_items: list[tuple[str, object]]) -> str:
    """Build the index page with a link for every discovered scenario."""
    links = "\n\n".join(
        f'''        <a href="/{name}" class="error-link">
            <strong>{name.capitalize()}</strong> - {(func.__doc__ or "").split(".")[0]}.
        </a>'''
        for name, func in scenario_items
    )

    return f"""
    <!DOCTYPE html>
    <html>
    <head>
        <title>FastAPI TraceRite Demo</title>
        <style>
            body {{ font-family: Arial, sans-serif; max-width: 800px; margin: 50px auto; padding: 20px; }}
            h1 {{ color: #333; }}
            .error-link {{ display: block; padding: 10px; margin: 10px 0;
                          background: #f0f0f0; text-decoration: none; color: #0066cc;
                          border-radius: 5px; }}
            .error-link:hover {{ background: #e0e0e0; }}
            code {{ background: #f5f5f5; padding: 2px 5px; border-radius: 3px; }}
        </style>
    </head>
    <body>
        <h1>TraceRite FastAPI Demo</h1>
        <p>Click any link below to trigger different types of errors:</p>

{links}
    </body>
    </html>
    """


@app.get("/", response_class=HTMLResponse)
async def index() -> str:
    """Main index page with links to all error scenarios."""
    return _index_html(_discover())


# Register a route for each discovered scenario.
for _name, _func in _discover():

    async def _handler(func: object = _func) -> JSONResponse:
        async_impl = getattr(func, "_async_impl", None)
        if async_impl is not None:
            await async_impl()
        elif inspect.iscoroutinefunction(func):
            await func()
        else:
            func()  # type: ignore[operator]
        return JSONResponse({"result": "ok"})

    _handler.__name__ = _func.__name__
    _handler.__doc__ = _func.__doc__
    app.get(f"/{_name}", response_class=JSONResponse)(_handler)


if __name__ == "__main__":
    print("This app should be run using the FastAPI CLI:")
    print("  fastapi dev demo/fastapi_demo.py")
    print("\nOr with uvicorn directly:")
    print("  uvicorn demo.fastapi_demo:app --reload")

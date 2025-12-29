"""
Sanic Demo Application for Testing Tracebacks

This application demonstrates various error scenarios to test traceback
rendering with TraceRite. Run with: sanic sanic_demo:app

Or within source repository:
  uvx --with-editable . sanic sanic_demo:app --dev
"""

import json
from dataclasses import dataclass

from sanic import Request, Sanic, response

app = Sanic("TraceriteDemo")


# Deep call chain example - each function in a different frame
def process_user_data(user_id):
    """Top level: fetch and process user data."""
    raw_data = fetch_user_from_api(user_id)
    return transform_user_data(raw_data)


def fetch_user_from_api(user_id):
    """Second level: simulate API call."""
    api_response = validate_user_id(user_id)
    return parse_api_response(api_response)


def validate_user_id(user_id):
    """Third level: validate the user ID."""
    if user_id < 1000:
        data = build_user_query(user_id)
        return data
    return {"id": user_id, "name": "Admin"}


def build_user_query(user_id):
    """Fourth level: build query parameters."""
    params = prepare_query_params(user_id)
    return {"query": params, "id": user_id}


def prepare_query_params(user_id):
    """Fifth level: prepare params - this will fail."""
    # Multi-line expression with binary operators
    result = (
        user_id * 2
        + 100 / (user_id - 500)  # Error: division by zero when user_id == 500
    )
    return {"value": result}


def parse_api_response(response):
    """Parse API response data."""
    return response


def transform_user_data(data):
    """Transform the data format."""
    return data


# Multi-line JSON parsing error (no function calls)
@app.get("/multiline-json")
async def multiline_json(request: Request):
    """Trigger JSONDecodeError in multi-line JSON - no extra calls."""
    json_data = """\
    {
        "name": "John",
        "age": 30,
        "address": {
            "street": "123 Main St",
            "city": "Boston"
            "state": "MA"
        }
    }"""  # Missing comma after "Boston"
    data = json.loads(json_data)
    return response.json(data)


# Multi-line list comprehension error (no function calls)
@app.get("/list-comprehension")
async def list_comprehension(request: Request):
    """Trigger error in multi-line list comprehension - no extra calls."""
    data = [1, 2, 3, 4, 5]
    result = [
        item * 2
        for item in data
        if item > 0 and item < 10 / (item - 3)  # Error when item == 3
    ]
    return response.json({"result": result})


# Multi-line string concatenation with type error (no function calls)
@app.get("/string-concat")
async def string_concat(request: Request):
    """Trigger TypeError in multi-line string concatenation - no extra calls."""
    suffix = 123  # Wrong type
    text = (
        "Hello "
        + "World "
        + suffix  # Error: can't concatenate str and int
        + "!"
    )
    return response.json({"result": text})


# Variable inspector test with Foo (good repr) and Bar (bad members)
@dataclass
class Foo:
    """Object with good __repr__ from dataclass."""

    x: int = 10
    y: int = 20


class Bar:
    """Object with poor __str__ and mixed member representations."""

    def __init__(self):
        self.value = 42
        self.text = "test"
        # Scanner object has poor __str__ representation
        self.bad = json.JSONDecoder().scan_once

    def __str__(self):
        return "<Bar object at 0x12345678>"


@app.get("/variable-inspector")
async def variable_inspector(request: Request):
    """Test variable inspector with objects having different representations.

    Foo has a good dataclass repr, so it displays normally.
    Bar has a poor __str__, so inspector shows its members:
    - good_value and good_name should be visible (good representations)
    - bad_scanner should be suppressed (poor <_json.Scanner object> representation)
    """
    foo = Foo(x=100, y=200)
    bar = Bar()
    # Use the variables in an expression
    result = foo.x + bar.value + len(bar.text)
    # Trigger error to see variable inspector
    return result / (foo.x - 100)  # Error: division by zero


# Multi-line function call with error (kept for call chain testing)
def complex_function_call():
    """Make a function call with multiple arguments across lines."""
    value = perform_calculation(
        10,
        20,
        30,
        divisor=get_divisor(),  # This will return 0
        multiplier=5,
    )
    return value


def perform_calculation(a, b, c, divisor, multiplier):
    """Perform calculation with multiple params."""
    return (a + b + c) * multiplier / divisor  # Error: division by zero


def get_divisor():
    """Get divisor value."""
    return 0


# Chained exception example
def outer_operation():
    """Outer operation that calls middle."""
    try:
        return middle_operation()
    except ValueError as e:
        outer_handle(e)


def outer_handle(e: Exception):
    # Trying to access e.message which doesn't exist
    raise RuntimeError(f"Outer operation failed {e.message}")


def middle_operation():
    """Middle operation that fails directly - no function calls."""
    try:
        # This will raise ZeroDivisionError directly in middle_operation
        x = 10
        y = 0
        result = x / y
        return result
    except ZeroDivisionError as e:
        # Transform exception type
        raise ValueError("Invalid calculation in middle") from e


# Route handlers
@app.get("/")
async def index(request: Request):
    """Main index page with links to all error scenarios."""
    html = """
    <!DOCTYPE html>
    <html>
    <head>
        <title>Sanic TraceRite Demo</title>
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
        <h1>TraceRite Sanic Demo</h1>
        <p>Click any link below to trigger different types of errors:</p>

        <a href="/deep-call-chain" class="error-link">
            <strong>Deep Call Chain</strong> - Multi-line expression with binary operators across 5+ function calls
        </a>

        <a href="/multiline-json" class="error-link">
            <strong>Multi-line JSON Error</strong> - JSONDecodeError in multi-line JSON string
        </a>

        <a href="/list-comprehension" class="error-link">
            <strong>List Comprehension</strong> - Error in multi-line list comprehension
        </a>

        <a href="/function-call" class="error-link">
            <strong>Multi-line Function Call</strong> - Error in function called with multiple line-wrapped arguments
        </a>

        <a href="/chained-exceptions" class="error-link">
            <strong>Chained Exceptions</strong> - Multiple exceptions with __cause__ chain
        </a>

        <a href="/exception-group" class="error-link">
            <strong>Exception Group</strong> - Multiple concurrent failures in asyncio TaskGroup
        </a>

        <a href="/string-concat" class="error-link">
            <strong>String Concatenation</strong> - TypeError in multi-line string concat
        </a>

        <a href="/variable-inspector" class="error-link">
            <strong>Variable Inspector</strong> - Test inspector with Foo (good repr) and Bar (mixed member representations)
        </a>
    </body>
    </html>
    """
    return response.html(html)


@app.get("/deep-call-chain")
async def deep_call_chain(request: Request):
    """Trigger error through deep call chain with multi-line expression."""
    result = process_user_data(500)  # Will cause division by zero
    return response.json({"result": result})


@app.get("/function-call")
async def function_call(request: Request):
    """Trigger error in multi-line function call."""
    result = complex_function_call()
    return response.json({"result": result})


@app.get("/chained-exceptions")
async def chained_exceptions(request: Request):
    """Trigger chained exceptions across multiple functions."""
    result = outer_operation()
    return response.json({"result": result})


# ExceptionGroup demo with asyncio TaskGroup
import asyncio


async def fail_soon(text):
    await asyncio.sleep(0.1)
    raise ValueError(text)


async def failure_too():
    await asyncio.sleep(0.1)
    xxx  # noqa: F821


async def exception_group_main():
    async with asyncio.TaskGroup() as tg:
        tg.create_task(fail_soon("Task A"))
        tg.create_task(fail_soon("Task B"))
        tg.create_task(failure_too())
        tg.create_task(asyncio.sleep(1))


@app.get("/exception-group")
async def exception_group(request: Request):
    """Trigger ExceptionGroup with multiple concurrent failures."""
    await exception_group_main()
    return response.json({"result": "ok"})


if __name__ == "__main__":
    print("This app should be run using the Sanic CLI:")
    print("  sanic sanic_demo:app")
    print("\nOr with auto-reload during development:")
    print("  sanic sanic_demo:app --dev")

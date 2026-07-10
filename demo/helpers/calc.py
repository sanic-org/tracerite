"""Calculation and business-logic error scenarios."""

from __future__ import annotations


def compute_ratio(numerator: float, denominator: float) -> float:
    """Divide two numbers; a zero denominator triggers an error."""
    return numerator / denominator


# Deep API-style call chain with a multi-line arithmetic expression.
def process_user_data(user_id: int) -> dict:
    """Top level: fetch and process user data."""
    raw_data = fetch_user_from_api(user_id)
    return transform_user_data(raw_data)


def fetch_user_from_api(user_id: int) -> dict:
    """Second level: simulate an API call."""
    api_response = validate_user_id(user_id)
    return parse_api_response(api_response)


def validate_user_id(user_id: int) -> dict:
    """Third level: validate the user ID."""
    if user_id < 1000:
        return build_user_query(user_id)
    return {"id": user_id, "name": "Admin"}


def build_user_query(user_id: int) -> dict:
    """Fourth level: build query parameters."""
    params = prepare_query_params(user_id)
    return {"query": params, "id": user_id}


def prepare_query_params(user_id: int) -> dict:
    """Fifth level: prepare params - this will fail for user_id == 500."""
    result = (
        user_id * 2
        + 100 / (user_id - 500)  # Error: division by zero when user_id == 500
    )
    return {"value": result}


def parse_api_response(response: dict) -> dict:
    """Parse API response data."""
    return response


def transform_user_data(data: dict) -> dict:
    """Transform the data format."""
    return data


# Multi-line function call with error in kwargs.
def complex_function_call() -> float:
    """Make a function call with several arguments across lines."""
    value = perform_calculation(
        10,
        20,
        30,
        divisor=get_divisor(),  # This will return 0
        multiplier=5,
    )
    return value


def perform_calculation(
    a: float, b: float, c: float, divisor: float, multiplier: float
) -> float:
    """Perform a calculation with multiple parameters."""
    return (a + b + c) * multiplier / divisor


def get_divisor() -> int:
    """Get the divisor value."""
    return 0


# Order-processing chain that wraps a low-level error in a domain error.
def process_order(order_id: int) -> dict:
    """Process an order through validation and discount calculation."""
    validated = validate_order(order_id)
    return apply_discount(validated)


def validate_order(order_id: int) -> int:
    """Ensure the order id is acceptable."""
    if order_id <= 0:
        raise ValueError("order id must be positive")
    return order_id


def apply_discount(order_id: int) -> dict:
    """Compute the discount for the given order."""
    discount = compute_ratio(100, order_id - 42)  # ZeroDivisionError for order_id == 42
    return {"order_id": order_id, "discount": discount}


# Explicit three-level exception chain.
def run_chained_pipeline() -> None:
    """Run a pipeline that fails and re-raises twice with cause."""
    try:
        _inner_pipeline_step()
    except ValueError as e:
        raise RuntimeError("Pipeline aborted") from e


def _inner_pipeline_step() -> None:
    try:
        _risky_computation()
    except ZeroDivisionError as e:
        raise ValueError("Invalid computation result") from e


def _risky_computation() -> float:
    x = 10
    y = 0
    return x / y

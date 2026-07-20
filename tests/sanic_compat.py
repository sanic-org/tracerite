"""Smoke test for Sanic's internal tracerite-backed error rendering.

Sanic depends on tracerite and uses it for HTML error pages. This test
creates a tiny Sanic app with an endpoint that raises a chained exception,
requests the debug HTML error page, and verifies that Sanic renders it
without crashing.
"""

from sanic import Sanic


def test_sanic_chain_error_renders():
    """Sanic should return a 500 HTML error page using tracerite."""
    app = Sanic("TraceriteSanicSmoke")

    @app.get("/chain")
    async def chain(request):
        try:
            raise ZeroDivisionError("division by zero")
        except ZeroDivisionError as e:
            raise ValueError("outer") from e

    _, response = app.test_client.get(
        "/chain",
        debug=True,
        headers={"Accept": "text/html"},
    )
    assert response.status == 500
    assert "text/html" in response.content_type
    assert response.text
    assert "Traceback" in response.text or "ZeroDivisionError" in response.text

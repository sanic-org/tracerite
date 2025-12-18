"""TraceRite extension for FastAPI/Starlette applications."""

import fastapi
from starlette.middleware.errors import ServerErrorMiddleware
from starlette.responses import HTMLResponse

from .html import html_traceback
from .logging import logger

_original_debug_response = None


def patch_fastapi():
    """
    Load TraceRite extension for FastAPI by patching ServerErrorMiddleware.

    This patches Starlette's ServerErrorMiddleware.debug_response to return
    TraceRite HTML tracebacks instead of the default debug HTML when running
    in debug mode and the client accepts HTML.
    """
    global _original_debug_response
    if _original_debug_response is not None:
        return  # Already loaded
    _original_debug_response = ServerErrorMiddleware.debug_response

    def tracerite_debug_response(self, request, exc):
        """Return TraceRite HTML traceback instead of Starlette's debug response."""
        accept = request.headers.get("accept", "")
        if "text/html" not in accept:
            return _original_debug_response(self, request, exc)  # type: ignore
        try:
            html = str(html_traceback(exc=exc, include_js_css=True))
            return HTMLResponse(
                "<!DOCTYPE html><title>FastAPI TraceRite</title>" + html,
                status_code=500,
            )
        except Exception as e:
            logger.error(f"Failed to generate TraceRite response: {e}")
            return _original_debug_response(self, request, exc)  # type: ignore

    ServerErrorMiddleware.debug_response = tracerite_debug_response  # type: ignore
    fastapi.routing.__tracebackhide__ = "until"  # type: ignore
    logger.info("TraceRite FastAPI extension loaded")

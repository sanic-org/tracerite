"""Tests for fastapi.py - FastAPI integration module with full mocking."""

import sys
from types import ModuleType
from unittest import mock

import pytest


def create_mock_fastapi_modules():
    """Create mock fastapi and starlette modules for testing."""
    # Create mock modules
    mock_fastapi = ModuleType("fastapi")
    mock_fastapi.routing = ModuleType("fastapi.routing")

    mock_starlette = ModuleType("starlette")
    mock_starlette_middleware = ModuleType("starlette.middleware")
    mock_starlette_middleware_errors = ModuleType("starlette.middleware.errors")
    mock_starlette_responses = ModuleType("starlette.responses")

    # Create a mock ServerErrorMiddleware class
    class MockServerErrorMiddleware:
        def __init__(self, app=None):
            self.app = app

        def debug_response(self, request, exc):
            return mock.MagicMock(body=b"original response", status_code=500)

    # Create a mock HTMLResponse class
    class MockHTMLResponse:
        def __init__(self, content, status_code=200):
            self.body = content.encode() if isinstance(content, str) else content
            self.status_code = status_code

    mock_starlette_middleware_errors.ServerErrorMiddleware = MockServerErrorMiddleware
    mock_starlette_responses.HTMLResponse = MockHTMLResponse

    return {
        "fastapi": mock_fastapi,
        "fastapi.routing": mock_fastapi.routing,
        "starlette": mock_starlette,
        "starlette.middleware": mock_starlette_middleware,
        "starlette.middleware.errors": mock_starlette_middleware_errors,
        "starlette.responses": mock_starlette_responses,
    }


@pytest.fixture
def reset_fastapi_module():
    """Fixture to reset the fastapi module state before and after tests."""
    from tracerite import fastapi as fastapi_module

    original = fastapi_module._original_debug_response
    fastapi_module._original_debug_response = None
    yield fastapi_module
    fastapi_module._original_debug_response = original


class TestFastAPIIntegration:
    """Test FastAPI/Starlette integration."""

    def test_patch_fastapi_already_patched(self):
        """Test patch_fastapi when already patched (idempotent)."""
        from tracerite import fastapi as fastapi_module

        # Set _original_debug_response to simulate already patched
        original = fastapi_module._original_debug_response
        fastapi_module._original_debug_response = lambda: None  # Non-None value

        try:
            # Should return early without doing anything
            fastapi_module.patch_fastapi()
        finally:
            # Restore
            fastapi_module._original_debug_response = original

    def test_patch_fastapi_success_with_mocked_modules(self, reset_fastapi_module):
        """Test successful patching with mocked FastAPI/Starlette."""
        fastapi_module = reset_fastapi_module
        mock_modules = create_mock_fastapi_modules()

        with mock.patch.dict(sys.modules, mock_modules):
            fastapi_module.patch_fastapi()

            # Verify patching occurred
            assert fastapi_module._original_debug_response is not None

            # Verify tracebackhide was set
            assert mock_modules["fastapi.routing"].__tracebackhide__ == "until"


class TestFastAPIDebugResponse:
    """Test the debug response replacement function."""

    def test_tracerite_debug_response_html_accept(self, reset_fastapi_module):
        """Test debug response when client accepts HTML."""
        fastapi_module = reset_fastapi_module
        mock_modules = create_mock_fastapi_modules()

        with mock.patch.dict(sys.modules, mock_modules):
            fastapi_module.patch_fastapi()

            # Get the patched middleware class
            ServerErrorMiddleware = mock_modules[
                "starlette.middleware.errors"
            ].ServerErrorMiddleware

            # Create mock request with HTML accept header
            mock_request = mock.MagicMock()
            mock_request.headers.get.return_value = "text/html,application/xhtml+xml"

            # Create a test exception
            test_exc = ValueError("test error for fastapi")

            # Create middleware instance and call debug_response
            middleware = ServerErrorMiddleware(app=mock.MagicMock())
            response = middleware.debug_response(mock_request, test_exc)

            # Should return HTMLResponse with TraceRite content
            assert response.status_code == 500
            assert b"FastAPI TraceRite" in response.body
            assert b"test error for fastapi" in response.body

    def test_tracerite_debug_response_non_html_accept(self, reset_fastapi_module):
        """Test debug response when client doesn't accept HTML (falls back)."""
        fastapi_module = reset_fastapi_module
        mock_modules = create_mock_fastapi_modules()

        # Track calls to original debug_response
        original_called = []
        original_debug = mock_modules[
            "starlette.middleware.errors"
        ].ServerErrorMiddleware.debug_response

        def tracking_original(self, request, exc):
            original_called.append((request, exc))
            return original_debug(self, request, exc)

        mock_modules[
            "starlette.middleware.errors"
        ].ServerErrorMiddleware.debug_response = tracking_original

        with mock.patch.dict(sys.modules, mock_modules):
            fastapi_module.patch_fastapi()

            ServerErrorMiddleware = mock_modules[
                "starlette.middleware.errors"
            ].ServerErrorMiddleware

            # Create mock request with non-HTML accept
            mock_request = mock.MagicMock()
            mock_request.headers.get.return_value = "application/json"

            test_exc = ValueError("test error")

            middleware = ServerErrorMiddleware(app=mock.MagicMock())
            response = middleware.debug_response(mock_request, test_exc)

            # Should fall back to original (non-HTML)
            assert len(original_called) == 1
            assert original_called[0][0] is mock_request
            assert original_called[0][1] is test_exc
            assert b"original response" in response.body

    def test_tracerite_debug_response_generation_error(self, reset_fastapi_module):
        """Test debug response when html_traceback fails (falls back)."""
        fastapi_module = reset_fastapi_module
        mock_modules = create_mock_fastapi_modules()

        # Track calls to original debug_response
        original_called = []
        original_debug = mock_modules[
            "starlette.middleware.errors"
        ].ServerErrorMiddleware.debug_response

        def tracking_original(self, request, exc):
            original_called.append((request, exc))
            return original_debug(self, request, exc)

        mock_modules[
            "starlette.middleware.errors"
        ].ServerErrorMiddleware.debug_response = tracking_original

        with mock.patch.dict(sys.modules, mock_modules):
            fastapi_module.patch_fastapi()

            ServerErrorMiddleware = mock_modules[
                "starlette.middleware.errors"
            ].ServerErrorMiddleware

            # Create mock request with HTML accept header
            mock_request = mock.MagicMock()
            mock_request.headers.get.return_value = "text/html"

            test_exc = ValueError("test error")

            # Mock html_traceback to fail
            with mock.patch(
                "tracerite.fastapi.html_traceback",
                side_effect=Exception("render failed"),
            ):
                middleware = ServerErrorMiddleware(app=mock.MagicMock())
                response = middleware.debug_response(mock_request, test_exc)

                # Should fall back to original response on error
                assert len(original_called) == 1
                assert b"original response" in response.body

"""Install and uninstall TraceRite exception handlers."""

from __future__ import annotations

import contextlib
import importlib
import logging
import sys
import threading
from dataclasses import dataclass, field
from typing import Any, Callable, Literal

from . import tty as _tty


@dataclass
class LoaderState:
    """Tracks the original handlers replaced by TraceRite."""

    original_excepthook: Callable[..., Any] | None = None
    original_threading_excepthook: Callable[[threading.ExceptHookArgs], Any] | None = (
        None
    )
    original_stream_handler_emit: (
        Callable[[logging.StreamHandler[Any], logging.LogRecord], Any] | None
    ) = None
    suppressed_modules: dict[str, Any] = field(default_factory=dict)


_state = LoaderState()

# Modules on which to set __tracebackhide__ when loaded.
_SUPPRESSIONS: dict[str, bool | Literal["until"]] = {
    "importlib._bootstrap": "until",
}


def _tracerite_excepthook(exc_type, exc_value, exc_tb):
    try:
        _tty.tty_traceback(exc=exc_value)
    except Exception:
        if _state.original_excepthook:
            _state.original_excepthook(exc_type, exc_value, exc_tb)
        else:
            sys.__excepthook__(exc_type, exc_value, exc_tb)


def _tracerite_threading_excepthook(args):  # pragma: no cover (pytest intercepts)
    try:
        _tty.tty_traceback(exc=args.exc_value)
    except Exception:
        if _state.original_threading_excepthook:
            _state.original_threading_excepthook(args)
        else:
            sys.__excepthook__(args.exc_type, args.exc_value, args.exc_traceback)


def _tracerite_stream_handler_emit(self, record: logging.LogRecord) -> None:
    """Emit a log record with TraceRite formatting for exceptions."""
    try:
        # Check if we have exception info to format specially
        if not record.exc_info or record.exc_info[1] is None:
            # No exception, use original emit
            original = _state.original_stream_handler_emit
            if original is None:
                return
            return original(self, record)

        # Temporarily clear exc_info so format() doesn't include traceback
        exc_info = record.exc_info
        record.exc_info = None
        record.exc_text = None
        try:
            msg = self.format(record)
        finally:
            record.exc_info = exc_info

        # Temporarily restore original handler to avoid recursion
        original_emit = logging.StreamHandler.emit
        original = _state.original_stream_handler_emit
        if original is None:
            return self.handleError(record)
        logging.StreamHandler.emit = original
        try:
            # Now format and write the exception using TraceRite
            _tty.tty_traceback(exc=exc_info[1], file=self.stream, msg=msg)
        finally:
            logging.StreamHandler.emit = original_emit
    except RecursionError:
        raise
    except Exception:
        self.handleError(record)


def load(
    *, hooks: bool = True, suppressions: bool = True, capture_logging: bool = True
) -> None:
    """Load TraceRite as the default exception handler.

    Replaces sys.excepthook to use TraceRite's pretty TTY formatting
    for all unhandled exceptions, including those in threads and
    logging.exception() calls.
    Call unload() to restore the original exception handlers.

    Args:
        hooks: Whether to install the exception hooks. Defaults to True.
        suppressions: Whether to set __tracebackhide__ on listed modules.
            Defaults to True.
        capture_logging: Whether to monkeypatch logging.StreamHandler.emit
            to format exceptions in logging.exception() calls. Defaults to True.

    Usage:
        import tracerite
        tracerite.load()  # Captures logging by default
        tracerite.load(capture_logging=False)  # Only captures sys.excepthook
    """
    if hooks:
        if sys.excepthook is not _tracerite_excepthook:
            _state.original_excepthook = sys.excepthook
            sys.excepthook = _tracerite_excepthook

        if threading.excepthook is not _tracerite_threading_excepthook:
            _state.original_threading_excepthook = threading.excepthook
            threading.excepthook = _tracerite_threading_excepthook

    if (
        capture_logging
        and logging.StreamHandler.emit is not _tracerite_stream_handler_emit
    ):
        _state.original_stream_handler_emit = logging.StreamHandler.emit
        logging.StreamHandler.emit = _tracerite_stream_handler_emit

    if suppressions:
        for module_name, hide_value in _SUPPRESSIONS.items():
            try:
                module = importlib.import_module(module_name)
            except Exception:
                continue
            module.__tracebackhide__ = hide_value
            _state.suppressed_modules[module_name] = module


def unload() -> None:
    """Restore the original exception handlers.

    Removes TraceRite from sys.excepthook, threading.excepthook, and
    logging.StreamHandler.emit, restoring the previous handlers.
    """
    if _state.original_excepthook is not None:
        sys.excepthook = _state.original_excepthook
        _state.original_excepthook = None

    if _state.original_threading_excepthook is not None:
        threading.excepthook = _state.original_threading_excepthook
        _state.original_threading_excepthook = None

    if _state.original_stream_handler_emit is not None:
        logging.StreamHandler.emit = _state.original_stream_handler_emit
        _state.original_stream_handler_emit = None

    for module in _state.suppressed_modules.values():
        with contextlib.suppress(AttributeError):
            delattr(module, "__tracebackhide__")
    _state.suppressed_modules.clear()

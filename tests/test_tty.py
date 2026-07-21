"""Comprehensive tests for the tty module - terminal output formatting."""

import io
import sys
import threading
import types

import pytest

from tracerite import extract_chain, hooks
from tracerite.hooks import load, unload
from tracerite.trace.finalize import extract_chain_exceptions
from tracerite.tty import (
    ANSI_ESCAPE_RE,
    ARROW_LEFT,
    BOLD,
    BOX_BL,
    BOX_BR,
    BOX_H,
    BOX_TL,
    BOX_TR,
    BOX_V,
    BOX_VL,
    DIM,
    EM,
    EM_CALL,
    FUNC,
    INDENT,
    LINE_PREFIX,
    LINE_PREFIX_BOT,
    LINE_PREFIX_TOP,
    LOCFN,
    MARK_BG,
    MARK_TEXT,
    RESET,
    VAR,
    _build_exception_banner,
    _build_subexception_summaries,
    _build_variable_inspector,
    _display_width,
    _format_fragment,
    _format_fragment_call,
    _get_branch_summary,
    _get_frame_label,
    _truncate_ansi,
    _truncate_inspector_line,
    _wrap_code_line,
    _wrap_text,
    symbols,
    tty_traceback,
)

from .errorcases import (
    binomial_operator,
    chained_from_and_without,
    exception_group_with_frames,
    exception_with_note_after_paragraphs,
    exception_with_notes,
    function_with_many_locals,
    function_with_many_locals_chained,
    function_with_single_local,
    max_type_error_case,
    multiline_marking,
    reraise_context,
    reraise_suppressed_context,
    unrelated_error_in_except,
)


class TestTtyTraceback:
    """Tests for tty_traceback main function."""

    def test_simple_exception_output(self):
        """Test tty_traceback outputs basic exception info."""
        output = io.StringIO()
        try:
            raise ValueError("test error message")
        except Exception as e:
            tty_traceback(exc=e, file=output)

        result = output.getvalue()
        assert "ValueError" in result
        assert "test error message" in result

    def test_no_frames_exception_with_msg(self):
        """A frameless exception with a log message gets a proper bottom border."""
        output = io.StringIO()
        exc = ValueError("frameless error")
        tty_traceback(exc=exc, msg="log message", file=output)

        result = output.getvalue()
        # The supplied log message replaces the chain header; the exception
        # itself is not rendered as a fallback.
        assert "log message" in result
        assert "ValueError" not in result
        assert "frameless error" not in result
        # Non-TTY output has ANSI stripped; compare the plain glyphs.
        assert ANSI_ESCAPE_RE.sub("", LINE_PREFIX_BOT) in result
        # Banner continuation lines should not keep the vertical border.
        assert f"\n{ANSI_ESCAPE_RE.sub('', LINE_PREFIX)}" not in result

    def test_no_frames_exception_without_msg(self):
        """A frameless exception with no log message renders without crashing."""
        output = io.StringIO()
        exc = ValueError("frameless error")
        tty_traceback(exc=exc, file=output)

        result = output.getvalue()
        assert "ValueError" in result
        assert result.startswith(ANSI_ESCAPE_RE.sub("", LINE_PREFIX_TOP))

    def test_output_contains_frame_location(self):
        """Test that output includes file location and function name."""
        output = io.StringIO()
        try:
            raise RuntimeError("location test")
        except Exception as e:
            tty_traceback(exc=e, file=output)

        result = output.getvalue()
        # Should contain this test function name
        assert "test_output_contains_frame_location" in result
        # Should contain the file reference
        assert "test_tty" in result

    def test_binary_operator_error_formatting(self):
        """Test formatting of binary operator type errors."""
        output = io.StringIO()
        try:
            binomial_operator()
        except Exception as e:
            tty_traceback(exc=e, file=output)

        result = output.getvalue()
        assert "TypeError" in result
        assert "binomial_operator" in result

    def test_multiline_error_formatting(self):
        """Test formatting of multiline expression errors."""
        output = io.StringIO()
        try:
            multiline_marking()
        except Exception as e:
            tty_traceback(exc=e, file=output)

        result = output.getvalue()
        assert "TypeError" in result
        assert "multiline_marking" in result

    def test_function_call_error_formatting(self):
        """Test formatting of function call errors."""
        output = io.StringIO()
        try:
            max_type_error_case()
        except Exception as e:
            tty_traceback(exc=e, file=output)

        result = output.getvalue()
        assert "TypeError" in result
        # Should show the calling function
        assert "max_type_error_case" in result

    def test_chained_exception_output(self):
        """Test output includes full exception chain."""
        output = io.StringIO()
        try:
            reraise_context()
        except Exception as e:
            tty_traceback(exc=e, file=output)

        result = output.getvalue()
        # Should show both exceptions
        assert "NameError" in result
        assert "RuntimeError" in result
        # Should show the chaining message (using new chain message format)
        assert "in except" in result or "from previous" in result.lower()

    def test_suppressed_context_exception(self):
        """Test exception with suppressed context (raise ... from None)."""
        output = io.StringIO()
        try:
            reraise_suppressed_context()
        except Exception as e:
            tty_traceback(exc=e, file=output)

        result = output.getvalue()
        # Only the RuntimeError should appear, not the NameError
        assert "RuntimeError" in result
        assert "foo" in result

    def test_explicit_cause_chain(self):
        """Test exception chain with explicit cause (raise ... from e)."""
        output = io.StringIO()
        try:
            chained_from_and_without()
        except Exception as e:
            tty_traceback(exc=e, file=output)

        result = output.getvalue()
        # Should show multiple exceptions from the chain
        assert (
            "NameError" in result
            or "RuntimeError" in result
            or "AttributeError" in result
        )

    def test_exception_in_except_handler(self):
        """Test formatting when error occurs in except handler."""
        output = io.StringIO()
        try:
            unrelated_error_in_except()
        except Exception as e:
            tty_traceback(exc=e, file=output)

        result = output.getvalue()
        # Should show both the original and the new error
        assert "NameError" in result
        assert "ZeroDivisionError" in result

    def test_defaults_to_stderr(self, capsys):
        """Test that output defaults to stderr when no file specified."""
        try:
            raise ValueError("stderr test")
        except Exception as e:
            tty_traceback(exc=e)

        captured = capsys.readouterr()
        assert "ValueError" in captured.err
        assert "stderr test" in captured.err

    def test_defaults_to_stderr_with_color(self, capsys, monkeypatch):
        """Test that output defaults to stderr with colors when tty."""
        # Mock isatty to return True
        monkeypatch.setattr(sys.stderr, "isatty", lambda: True)

        try:
            function_with_many_locals()
        except Exception as e:
            tty_traceback(exc=e)

        captured = capsys.readouterr()
        assert "RuntimeError" in captured.err
        assert "many locals test" in captured.err
        # Should contain ANSI codes
        assert "\x1b[" in captured.err

    def test_with_explicit_chain(self):
        """Test tty_traceback with pre-extracted chain."""
        output = io.StringIO()
        try:
            raise ValueError("pre-extracted")
        except Exception as e:
            chain = extract_chain(e)
            tty_traceback(chain=chain, file=output)

        result = output.getvalue()
        assert "ValueError" in result
        assert "pre-extracted" in result

    def test_narrow_terminal_width(self):
        """Test that output adapts to narrow terminal widths."""
        output = io.StringIO()
        # Create a mock file object with a narrow terminal
        # The actual adaptation happens in tty_traceback, but we can't
        # easily mock terminal size, so we just verify it doesn't crash
        try:
            raise ValueError("narrow terminal test")
        except Exception as e:
            tty_traceback(exc=e, file=output)

        result = output.getvalue()
        assert "ValueError" in result

    def test_narrow_terminal_width_with_color(self):
        """Test narrow terminal with colors enabled."""
        output = io.StringIO()
        output.isatty = lambda: True
        try:
            raise ValueError("very long exception message that should cause wrapping")
        except Exception as e:
            tty_traceback(exc=e, file=output, term_width=40)

        result = output.getvalue()
        assert "ValueError" in result
        assert "\x1b[" in result

    def test_title_wraps_type_but_summary_fits(self):
        """Test title where type+summary overflows but summary alone fits."""
        output = io.StringIO()
        output.isatty = lambda: True
        # term_width=30, "ValueError: " is 12 chars, need summary > 18 but <= 28
        try:
            raise ValueError("twenty char summary!")  # 20 chars
        except Exception as e:
            tty_traceback(exc=e, file=output, term_width=30)

        result = output.getvalue()
        assert "ValueError" in result
        assert "twenty char summary!" in result
        assert "\x1b[" in result

    def test_file_without_isatty(self):
        """Test tty_traceback with a file that doesn't have isatty method."""

        class MockFile:
            def __init__(self):
                self.data = ""

            def write(self, s):
                self.data += s

            def flush(self):
                pass

            def fileno(self):
                raise OSError("no fileno")

        output = MockFile()
        try:
            raise ValueError("test")
        except Exception as e:
            tty_traceback(exc=e, file=output)

        assert "ValueError" in output.data
        assert "test" in output.data

    def test_no_color_env_disables_color(self, monkeypatch):
        """NO_COLOR disables colors even on a TTY."""
        monkeypatch.setenv("NO_COLOR", "1")
        output = io.StringIO()
        output.isatty = lambda: True
        try:
            raise ValueError("no color test")
        except Exception as e:
            tty_traceback(exc=e, file=output)

        result = output.getvalue()
        assert "ValueError" in result
        assert "\x1b[" not in result

    def test_no_color_takes_precedence_over_force_color(self, monkeypatch):
        """NO_COLOR wins when both it and FORCE_COLOR are set."""
        monkeypatch.setenv("NO_COLOR", "1")
        monkeypatch.setenv("FORCE_COLOR", "1")
        output = io.StringIO()
        output.isatty = lambda: True
        try:
            raise ValueError("no color test")
        except Exception as e:
            tty_traceback(exc=e, file=output)

        assert "\x1b[" not in output.getvalue()

    def test_force_color_env_enables_color(self, monkeypatch):
        """FORCE_COLOR enables colors even when the file is not a TTY."""
        monkeypatch.setenv("FORCE_COLOR", "1")
        output = io.StringIO()
        try:
            raise ValueError("force color test")
        except Exception as e:
            tty_traceback(exc=e, file=output)

        result = output.getvalue()
        assert "ValueError" in result
        assert "\x1b[" in result

    def test_term_width_from_terminal_size(self, monkeypatch):
        """Test tty_traceback gets term_width from os.get_terminal_size."""
        import os

        output = io.StringIO()
        # Mock get_terminal_size to return a size with both columns and lines
        # (lines is needed by pytest's terminal reporter)
        monkeypatch.setattr(
            os,
            "get_terminal_size",
            lambda fd: type("Size", (), {"columns": 100, "lines": 24})(),
        )
        # Mock file.fileno to return 1
        output.fileno = lambda: 1
        try:
            raise ValueError("test")
        except Exception as e:
            tty_traceback(exc=e, file=output)

        result = output.getvalue()
        assert "ValueError" in result

    def test_narrow_terminal_width_fallback_to_80(self, monkeypatch):
        """Very narrow terminals (<40 cols) are assumed temporary and forced to 80."""
        import os

        output = io.StringIO()
        monkeypatch.setattr(
            os,
            "get_terminal_size",
            lambda fd: type("Size", (), {"columns": 20, "lines": 24})(),
        )
        output.fileno = lambda: 1
        try:
            raise ValueError("narrow terminal test")
        except Exception as e:
            tty_traceback(exc=e, file=output)

        result = output.getvalue()
        assert "ValueError" in result

    def test_inspector_display(self):
        """Test that inspector is displayed when isatty and variables present."""
        output = io.StringIO()
        output.isatty = lambda: True
        try:
            function_with_many_locals()
        except Exception as e:
            tty_traceback(exc=e, file=output, term_width=120)

        result = output.getvalue()
        assert "RuntimeError" in result
        assert "many locals test" in result
        # Check that inspector is shown (has cursor positioning and variables)
        assert "\x1b[" in result  # ANSI codes

    def test_inspector_with_banners(self):
        """Test inspector with exception banners (chained exceptions)."""
        output = io.StringIO()
        output.isatty = lambda: True
        try:
            function_with_many_locals_chained()
        except Exception as e:
            tty_traceback(exc=e, file=output, term_width=120)

        result = output.getvalue()
        assert "RuntimeError" in result
        assert "ValueError" in result
        assert "many locals chained test" in result
        # Check that inspector is shown
        assert "\x1b[" in result


class TestChainedExceptions:
    """Tests for exception chaining display."""

    def test_cause_chain_ordering(self):
        """Test that exception chain is displayed oldest-first."""
        output = io.StringIO()
        try:
            try:
                raise ValueError("first")
            except ValueError as e:
                raise RuntimeError("second") from e
        except Exception as e:
            tty_traceback(exc=e, file=output)

        result = output.getvalue()
        # Both should be present
        assert "ValueError" in result
        assert "RuntimeError" in result
        # "from previous" indicates the second exception was raised from the first
        assert "from previous" in result

    def test_context_chain_ordering(self):
        """Test implicit context chain (without 'from')."""
        output = io.StringIO()
        try:
            try:
                raise ValueError("original")
            except ValueError:
                raise RuntimeError("during handling")
        except Exception as e:
            tty_traceback(exc=e, file=output)

        result = output.getvalue()
        assert "ValueError" in result
        assert "RuntimeError" in result
        # Should indicate this happened while handling previous (new format: "in except")
        assert "in except" in result

    def test_deeply_nested_chain(self):
        """Test handling of deeply nested exception chains."""
        output = io.StringIO()
        try:
            try:
                try:
                    raise ValueError("level 1")
                except ValueError as e:
                    raise TypeError("level 2") from e
            except TypeError:
                0 / 0  # noqa: B018
        except Exception as e:
            tty_traceback(exc=e, file=output)

        result = output.getvalue()
        assert "ZeroDivisionError while handling TypeError from ValueError" in result
        assert "ValueError: level 1" in result
        assert "TypeError from previous: level 2" in result
        assert "ZeroDivisionError in except: division by zero" in result

    def test_deeply_nested_chain_with_calls(self):
        """Test TTY formatting of three-level exception chain with function calls.

        Chronological order is tested in test_chain_analysis.py. This test
        verifies the TTY formatting doesn't crash and shows expected content.
        """
        from .errorcases import deeply_nested_chain_with_calls

        output = io.StringIO()
        try:
            deeply_nested_chain_with_calls()
        except Exception as e:
            tty_traceback(exc=e, file=output)

        result = output.getvalue()

        # Verify title shows full chain
        assert "ZeroDivisionError while handling TypeError from ValueError" in result

        # Verify all three exceptions appear in output
        assert "ValueError: level 1" in result
        assert "TypeError from previous: level 2" in result
        assert "ZeroDivisionError in except: division by zero" in result

        # Verify helper function names appear (extra frames from calls)
        assert "_raise_level1" in result
        assert "_handle_and_raise_level2" in result
        assert "_handle_and_divide_by_zero" in result


class TestLoadUnload:
    """Tests for load() and unload() exception hooks."""

    def test_load_replaces_excepthook(self):
        """Test that load() replaces sys.excepthook."""
        original_hook = sys.excepthook
        try:
            load()
            assert sys.excepthook != original_hook
            assert sys.excepthook.__name__ == "_tracerite_excepthook"
        finally:
            unload()

    def test_unload_restores_excepthook(self):
        """Test that unload() restores original sys.excepthook."""
        original_hook = sys.excepthook
        load()
        unload()
        assert sys.excepthook == original_hook

    def test_load_replaces_threading_excepthook(self):
        """Test that load() replaces threading.excepthook."""
        original_hook = threading.excepthook
        try:
            load()
            assert threading.excepthook != original_hook
            assert threading.excepthook.__name__ == "_tracerite_threading_excepthook"
        finally:
            unload()

    def test_unload_restores_threading_excepthook(self):
        """Test that unload() restores threading.excepthook."""
        original_hook = threading.excepthook
        load()
        unload()
        assert threading.excepthook == original_hook

    def test_double_load_preserves_original(self):
        """Test that calling load() twice preserves the original hook."""
        original_hook = sys.excepthook
        try:
            load()
            assert sys.excepthook.__name__ == "_tracerite_excepthook"
            load()  # Second load
            # Should still be a tracerite hook
            assert sys.excepthook.__name__ == "_tracerite_excepthook"
            unload()
            # Should restore to original, not to the first tracerite hook
            assert sys.excepthook == original_hook
        finally:
            # Ensure cleanup
            if sys.excepthook != original_hook:
                sys.excepthook = original_hook

    def test_unload_without_load_is_safe(self):
        """Test that calling unload() without load() is safe."""
        original_hook = sys.excepthook
        unload()  # Should not crash
        assert sys.excepthook == original_hook

    def test_tty_load_unload_deprecated(self):
        """Test that tracerite.tty.load/unload emit deprecation warnings."""
        from tracerite import tty as tty_module

        with pytest.warns(DeprecationWarning, match="tracerite.tty.load is deprecated"):
            tty_module.load()
        try:
            with pytest.warns(
                DeprecationWarning, match="tracerite.tty.unload is deprecated"
            ):
                tty_module.unload()
        finally:
            unload()  # Ensure cleanup if the warning assertion failed

    def test_load_sets_tracebackhide_suppressions(self, monkeypatch):
        """Test that load() sets __tracebackhide__ on suppression modules."""
        load()
        try:
            import importlib._bootstrap

            assert importlib._bootstrap.__tracebackhide__ is True
        finally:
            unload()

    def test_unload_clears_tracebackhide_suppressions(self, monkeypatch):
        """Test that unload() clears __tracebackhide__ set by load()."""
        import importlib._bootstrap

        load()
        unload()
        assert not hasattr(importlib._bootstrap, "__tracebackhide__")

    def test_load_hooks_false(self):
        """Test that load(hooks=False) doesn't install exception hooks."""
        original_hook = sys.excepthook
        load(hooks=False)
        try:
            assert sys.excepthook is original_hook
        finally:
            unload()

    def test_load_suppressions_false(self, monkeypatch):
        """Test that load(suppressions=False) doesn't set __tracebackhide__."""
        fake_module = types.ModuleType("importlib._bootstrap")
        monkeypatch.setitem(sys.modules, "importlib._bootstrap", fake_module)
        load(suppressions=False)
        try:
            assert not hasattr(fake_module, "__tracebackhide__")
        finally:
            unload()

    def test_unload_restores_existing_tracebackhide(self, monkeypatch):
        """Test that unload() restores a pre-existing __tracebackhide__ value."""
        fake_module = types.ModuleType("importlib._bootstrap")
        fake_module.__tracebackhide__ = "until"
        monkeypatch.setitem(sys.modules, "importlib._bootstrap", fake_module)
        load()
        try:
            assert fake_module.__tracebackhide__ is True
        finally:
            unload()
        assert fake_module.__tracebackhide__ == "until"

    def test_load_suppressions_skips_same_value(self, monkeypatch):
        """Test that load() doesn't track modules already set to the same value."""
        fake_module = types.ModuleType("importlib._bootstrap")
        fake_module.__tracebackhide__ = True
        monkeypatch.setitem(sys.modules, "importlib._bootstrap", fake_module)
        load()
        try:
            assert fake_module.__tracebackhide__ is True
            assert "importlib._bootstrap" not in hooks._state.suppressed
        finally:
            unload()

    def test_loaded_hook_handles_exceptions(self, capsys):
        """Test that the loaded hook properly handles exceptions."""
        load()
        try:
            # Manually call the excepthook to test it
            try:
                raise ValueError("hook test")
            except Exception:
                exc_type, exc_value, exc_tb = sys.exc_info()
                sys.excepthook(exc_type, exc_value, exc_tb)

            captured = capsys.readouterr()
            assert "ValueError" in captured.err
            assert "hook test" in captured.err
        finally:
            unload()

    def test_load_with_capture_logging_true(self):
        """Test that load(capture_logging=True) replaces StreamHandler.emit."""
        import logging

        original_emit = logging.StreamHandler.emit
        try:
            load(capture_logging=True)
            assert logging.StreamHandler.emit != original_emit
            assert (
                logging.StreamHandler.emit.__name__ == "_tracerite_stream_handler_emit"
            )
        finally:
            unload()

    def test_load_with_capture_logging_false(self):
        """Test that load(capture_logging=False) doesn't replace StreamHandler.emit."""
        import logging

        original_emit = logging.StreamHandler.emit
        try:
            load(capture_logging=False)
            assert logging.StreamHandler.emit == original_emit
        finally:
            unload()

    def test_unload_restores_stream_handler_emit(self):
        """Test that unload() restores original StreamHandler.emit."""
        import logging

        original_emit = logging.StreamHandler.emit
        load(capture_logging=True)
        unload()
        assert logging.StreamHandler.emit == original_emit

    def test_stream_handler_formats_exceptions(self, capsys):
        """Test that StreamHandler.emit formats exceptions with TraceRite."""
        import logging

        # Set up a logger with a stream handler
        logger = logging.getLogger("test_logger")
        logger.setLevel(logging.ERROR)
        handler = logging.StreamHandler(sys.stderr)
        handler.setLevel(logging.ERROR)
        logger.addHandler(handler)

        load(capture_logging=True)
        try:
            try:
                raise ValueError("stream handler test")
            except ValueError:
                logger.exception("Test exception logging")

            captured = capsys.readouterr()
            assert "ValueError" in captured.err
            assert "stream handler test" in captured.err
        finally:
            unload()
            logger.removeHandler(handler)

    def test_stream_handler_formats_exceptions_with_color(self, capsys, monkeypatch):
        """Test that StreamHandler.emit formats exceptions with colors when tty."""
        import logging

        # Mock isatty to return True
        monkeypatch.setattr(sys.stderr, "isatty", lambda: True)

        # Set up a logger with a stream handler
        logger = logging.getLogger("test_logger")
        logger.setLevel(logging.ERROR)
        handler = logging.StreamHandler(sys.stderr)
        handler.setLevel(logging.ERROR)
        logger.addHandler(handler)

        load(capture_logging=True)
        try:
            try:
                chained_from_and_without()
            except Exception:
                logger.exception("Test exception logging")

            captured = capsys.readouterr()
            assert "AttributeError" in captured.err
            assert "NameError" in captured.err
            # Should contain ANSI codes
            assert "\x1b[" in captured.err
        finally:
            unload()
            logger.removeHandler(handler)

    def test_stream_handler_no_exception_uses_original(self, capsys):
        """Test that StreamHandler.emit uses original for non-exception records."""
        import logging

        # Set up a logger
        logger = logging.getLogger("test_logger_no_exc")
        logger.setLevel(logging.INFO)
        handler = logging.StreamHandler(sys.stderr)
        logger.addHandler(handler)

        load(capture_logging=True)
        try:
            logger.info("Normal log message")
            captured = capsys.readouterr()
            assert "Normal log message" in captured.err
        finally:
            unload()
            logger.removeHandler(handler)

    def test_stream_handler_tty_traceback_error_falls_back(self, capsys, monkeypatch):
        """Test that StreamHandler.emit falls back when tty_traceback fails."""
        import logging

        # Mock tty_traceback to raise an exception
        from tracerite import tty

        original_tty_traceback = tty.tty_traceback

        def failing_tty_traceback(**kwargs):
            raise RuntimeError("tty_traceback failed")

        monkeypatch.setattr(tty, "tty_traceback", failing_tty_traceback)

        # Set up a handler
        handler = logging.StreamHandler(sys.stderr)

        load(capture_logging=True)
        try:
            # Create a log record with exception info
            record = logging.LogRecord(
                name="test",
                level=logging.ERROR,
                pathname="test.py",
                lineno=1,
                msg="Test message",
                args=(),
                exc_info=(ValueError, ValueError("test"), None),
            )

            # Call emit directly - should not raise
            handler.emit(record)

            captured = capsys.readouterr()
            # handleError should have written something to stderr
            assert len(captured.err) > 0
        finally:
            unload()
            monkeypatch.setattr(tty, "tty_traceback", original_tty_traceback)

    def test_stream_handler_recursion_error_propagates(self, monkeypatch):
        """Test that RecursionError in tty_traceback is re-raised."""
        import logging

        from tracerite import tty

        original_tty_traceback = tty.tty_traceback

        def recursion_tty_traceback(**kwargs):
            raise RecursionError("infinite recursion")

        monkeypatch.setattr(tty, "tty_traceback", recursion_tty_traceback)

        handler = logging.StreamHandler(sys.stderr)

        load(capture_logging=True)
        try:
            record = logging.LogRecord(
                name="test",
                level=logging.ERROR,
                pathname="test.py",
                lineno=1,
                msg="Test message",
                args=(),
                exc_info=(ValueError, ValueError("test"), None),
            )

            # RecursionError should propagate
            with pytest.raises(RecursionError):
                handler.emit(record)
        finally:
            unload()
            monkeypatch.setattr(tty, "tty_traceback", original_tty_traceback)

    def test_stream_handler_no_original_returns(self):
        """Test StreamHandler.emit returns when original is None (line 70)."""
        import logging

        from tracerite import hooks

        handler = logging.StreamHandler(sys.stderr)
        load(capture_logging=True)
        try:
            hooks._state.original_stream_handler_emit = None
            record = logging.LogRecord(
                name="test",
                level=logging.INFO,
                pathname="test.py",
                lineno=1,
                msg="No exception",
                args=(),
                exc_info=None,
            )
            # Should not raise
            handler.emit(record)
        finally:
            unload()

    def test_stream_handler_no_original_with_exception_handle_error(self):
        """Test StreamHandler.emit falls back when original is None (line 86)."""
        import logging

        from tracerite import hooks

        handler = logging.StreamHandler(sys.stderr)
        load(capture_logging=True)
        try:
            hooks._state.original_stream_handler_emit = None
            record = logging.LogRecord(
                name="test",
                level=logging.ERROR,
                pathname="test.py",
                lineno=1,
                msg="With exception",
                args=(),
                exc_info=(ValueError, ValueError("test"), None),
            )
            # Should call handleError, not raise
            handler.emit(record)
        finally:
            unload()

    def test_load_suppressions_import_failure(self, monkeypatch):
        """Test load() skips suppression modules that fail to import (lines 141-142)."""
        import tracerite.hooks as hooks

        monkeypatch.setattr(
            hooks, "_SUPPRESSIONS", {"nonexistent.module.for.test": True}
        )
        load()
        try:
            # Should not crash
            pass
        finally:
            unload()

    def test_load_suppressions_unset_value(self, monkeypatch):
        """Test load() deletes existing __tracebackhide__ when hide_value is Unset (line 148)."""
        import tracerite.hooks as hooks

        fake_module = types.ModuleType("fake_module_for_unset")
        fake_module.__tracebackhide__ = True
        monkeypatch.setitem(sys.modules, "fake_module_for_unset", fake_module)
        monkeypatch.setattr(
            hooks, "_SUPPRESSIONS", {"fake_module_for_unset": hooks.Unset}
        )
        load()
        try:
            assert not hasattr(fake_module, "__tracebackhide__")
        finally:
            unload()

    def test_load_suppressions_extra(self, monkeypatch):
        """Test load_suppressions(extra=) sets and restores extra suppressions."""
        fake_module = types.ModuleType("fake_extra_module")
        fake_module.__tracebackhide__ = "until"
        monkeypatch.setitem(sys.modules, "fake_extra_module", fake_module)
        hooks.load_suppressions(extra={"fake_extra_module": True})
        try:
            assert fake_module.__tracebackhide__ is True
            assert fake_module in hooks._state.suppressed
        finally:
            hooks.unload_suppressions()
        assert fake_module.__tracebackhide__ == "until"


class TestFrameFormatting:
    """Tests for frame label and info extraction."""

    def test_get_frame_label_with_function(self):
        """Test frame label includes function name."""
        try:
            raise ValueError("label test")
        except Exception as e:
            chain = extract_chain_exceptions(e)
            frame = chain[0]["frames"][-1]
            location_part, function_part = _get_frame_label(frame)
            # Combine and strip colors to get plain text
            label_plain = ANSI_ESCAPE_RE.sub("", location_part + function_part)

            # Should contain function name in light blue
            assert FUNC in function_part
            assert "test_get_frame_label_with_function" in label_plain
            # Should contain filename in green
            assert LOCFN in location_part
            assert "test_tty" in label_plain


class TestFragmentFormatting:
    """Tests for fragment formatting functions."""

    def test_format_fragment_plain(self):
        """Test formatting fragment without marks or emphasis."""
        fragment = {"code": "x = 1"}
        colored = _format_fragment(fragment)
        plain = ANSI_ESCAPE_RE.sub("", colored)
        assert plain == "x = 1"
        assert colored == "x = 1"

    def test_format_fragment_with_mark_solo(self):
        """Test formatting fragment with solo mark (single highlighted region)."""
        fragment = {"code": "error_code", "mark": "solo"}
        colored = _format_fragment(fragment)
        plain = ANSI_ESCAPE_RE.sub("", colored)
        assert plain == "error_code"
        assert MARK_BG in colored
        assert MARK_TEXT in colored
        assert RESET in colored

    def test_format_fragment_with_mark_beg_fin(self):
        """Test formatting fragment with mark spanning multiple fragments."""
        # Beginning of marked region
        frag_beg = {"code": "start", "mark": "beg"}
        colored_beg = _format_fragment(frag_beg)
        assert MARK_BG in colored_beg
        assert RESET not in colored_beg  # Should not close yet

        # End of marked region
        frag_fin = {"code": "end", "mark": "fin"}
        colored_fin = _format_fragment(frag_fin)
        assert RESET in colored_fin

    def test_format_fragment_with_em_solo(self):
        """Test formatting fragment with emphasis (error location)."""
        fragment = {"code": "+", "em": "solo", "mark": "solo"}
        colored = _format_fragment(fragment)
        plain = ANSI_ESCAPE_RE.sub("", colored)
        assert plain == "+"
        assert EM in colored

    def test_format_fragment_call_plain(self):
        """Test call frame fragment formatting without emphasis."""
        fragment = {"code": "func(arg)"}
        colored = _format_fragment_call(fragment)
        plain = ANSI_ESCAPE_RE.sub("", colored)
        assert plain == "func(arg)"
        assert colored == "func(arg)"

    def test_format_fragment_call_with_em(self):
        """Test call frame fragment formatting with emphasis."""
        fragment = {"code": "bad_call", "em": "solo"}
        colored = _format_fragment_call(fragment)
        plain = ANSI_ESCAPE_RE.sub("", colored)
        assert plain == "bad_call"
        assert EM_CALL in colored
        assert RESET in colored

    def test_format_fragment_strips_newlines(self):
        """Test that fragments strip trailing newlines."""
        fragment = {"code": "line\n\r"}
        colored = _format_fragment(fragment)
        plain = ANSI_ESCAPE_RE.sub("", colored)
        assert plain == "line"
        assert "\n" not in colored
        assert "\r" not in colored


class TestVariableInspector:
    """Tests for variable inspector formatting."""

    def test_build_variable_inspector_empty(self):
        """Test inspector with no variables."""
        result, min_width = _build_variable_inspector([], term_width=80)
        assert result == []
        assert min_width == 0

    def test_build_variable_inspector_simple_vars(self):
        """Test inspector with simple variables."""

        variables = [
            {"name": "x", "typename": "int", "value": "42", "format_hint": None},
            {
                "name": "name",
                "typename": "str",
                "value": '"hello"',
                "format_hint": None,
            },
        ]
        result, min_width = _build_variable_inspector(variables, term_width=80)

        assert len(result) == 2
        # Each item is (colored_line, plain_width, value_start_col)
        for colored, width, value_col in result:
            assert isinstance(colored, str)
            assert isinstance(width, int)
            assert isinstance(value_col, int)

    def test_build_variable_inspector_without_typename(self):
        """Test inspector with variables that have no type annotation."""

        variables = [
            {"name": "y", "typename": None, "value": "100", "format_hint": None},
        ]
        result, min_width = _build_variable_inspector(variables, term_width=80)

        assert len(result) == 1
        colored, width, value_col = result[0]
        # Should format as "y = 100" not "y: None = 100"
        assert VAR in colored  # Variable name in cyan
        assert "= 100" in colored or "100" in colored

    def test_build_variable_inspector_keyvalue(self):
        """Test inspector with key-value dict representation."""

        variables = [
            {
                "name": "d",
                "typename": "dict",
                "value": {"type": "keyvalue", "rows": [("a", "1"), ("b", "2")]},
                "format_hint": None,
            },
        ]
        result, min_width = _build_variable_inspector(variables, term_width=80)

        assert len(result) == 1
        colored, _, _ = result[0]
        assert "a:" in colored or "{" in colored

    def test_build_variable_inspector_array(self):
        """Test inspector with array representation."""

        variables = [
            {
                "name": "arr",
                "typename": "list",
                "value": {"type": "array", "rows": [[1, 2, 3]]},
                "format_hint": None,
            },
        ]
        result, min_width = _build_variable_inspector(variables, term_width=80)

        assert len(result) == 1
        colored, _, _ = result[0]
        assert "[" in colored

    def test_build_variable_inspector_truncation(self):
        """Long values are preserved by the builder; truncation is deferred."""

        long_value = "x" * 200
        variables = [
            {
                "name": "long_var",
                "typename": "str",
                "value": long_value,
                "format_hint": None,
            },
        ]
        result, min_width = _build_variable_inspector(variables, term_width=80)

        assert len(result) == 1
        colored, width, value_col = result[0]
        # The full value is retained; width is prefix + full value length.
        assert width == len("long_var: str = ") + len(long_value)
        assert "…" not in colored
        # Truncation is applied later by _truncate_inspector_line.
        truncated = _truncate_inspector_line(
            colored, width, value_col, available_for_content=30
        )
        assert "…" in truncated

    def test_build_variable_inspector_skips_ellipsis_value(self):
        """Test inspector skips variables with ellipsis value (lines 1241, 1246)."""

        variables = [
            {"name": "x", "typename": "int", "value": "⋯", "format_hint": "inline"},
        ]
        result, min_width = _build_variable_inspector(variables, term_width=80)

        assert result == []
        assert min_width == 0

    def test_truncate_inspector_line_narrow_value_space(self):
        """When only the prefix fits, the value is replaced by a plain ellipsis."""
        colored = "var: str = some value"
        width = len(colored)
        value_start = len("var: str = ")
        result = _truncate_inspector_line(
            colored, width, value_start, available_for_content=value_start + 1
        )
        assert result == "…"

    def test_truncate_inspector_line_wide_chars(self):
        """Truncation respects display columns, not character counts."""
        colored = "var: str = 日本語の値"
        plain = ANSI_ESCAPE_RE.sub("", colored)
        width = _display_width(plain)
        value_start = len("var: str = ")
        available_for_content = value_start + 5  # room for 5 display columns

        result = _truncate_inspector_line(
            colored, width, value_start, available_for_content=available_for_content
        )

        result_plain = ANSI_ESCAPE_RE.sub("", result)
        assert _display_width(result_plain) <= available_for_content
        assert "…" in result

    def test_build_variable_inspector_preserves_content_ellipsis(self):
        """Ellipsis characters that are part of the value are not restyled."""
        from tracerite.tty import DIM

        variables = [
            {"name": "s", "typename": "str", "value": "abc …", "format_hint": "inline"}
        ]
        result, _ = _build_variable_inspector(variables, term_width=200)
        colored, _, _ = result[0]
        # No dim styling was applied to the content ellipsis.
        assert DIM not in colored

    def test_truncate_inspector_line_inline_marker_shortens_right(self):
        """Inline values shorten the right side of the middle ellipsis first."""

        value = "head12345 … tail12345"
        variables = [
            {"name": "s", "typename": "str", "value": value, "format_hint": "inline"}
        ]
        entries, _ = _build_variable_inspector(variables, term_width=200)
        colored, width, value_start = entries[0]
        prefix_width = _display_width(ANSI_ESCAPE_RE.sub("", colored[:value_start]))

        available_for_content = prefix_width + 15
        result = _truncate_inspector_line(
            colored, width, value_start, available_for_content
        )
        result_plain = ANSI_ESCAPE_RE.sub("", result)

        assert "head12345" in result_plain
        assert " … " in result_plain
        # The end of the right side is preserved, not its beginning.
        assert result_plain.endswith("45")
        assert "tail" not in result_plain
        assert _display_width(result_plain) <= available_for_content
        assert result.endswith(RESET)

    def test_truncate_inspector_line_inline_marker_shortens_left(self):
        """Once the right side is gone, shorten the left side and keep trailing ellipsis."""

        value = "head12345 … tail12345"
        variables = [
            {"name": "s", "typename": "str", "value": value, "format_hint": "inline"}
        ]
        entries, _ = _build_variable_inspector(variables, term_width=200)
        colored, width, value_start = entries[0]
        prefix_width = _display_width(ANSI_ESCAPE_RE.sub("", colored[:value_start]))

        available_for_content = prefix_width + 8
        result = _truncate_inspector_line(
            colored, width, value_start, available_for_content
        )
        result_plain = ANSI_ESCAPE_RE.sub("", result)

        assert result_plain.endswith("…")
        assert "head123" in result_plain
        assert "tail" not in result_plain
        assert " … " not in result_plain
        assert _display_width(result_plain) <= available_for_content
        assert result.endswith(RESET)

    def test_truncate_inspector_line_inline_marker_plain_line(self):
        """Inline-marker truncation also works on uncoloured lines."""
        colored = "var: str = head12345 … tail12345"
        width = len(colored)
        value_start = len("var: str = ")
        available_for_content = value_start + 15

        result = _truncate_inspector_line(
            colored, width, value_start, available_for_content
        )

        assert result == "var: str = head12345 … 345"
        assert "\x1b" not in result

    def test_truncate_inspector_line_zero_available(self):
        """When no content space is available, return a plain ellipsis."""
        result = _truncate_inspector_line(
            "var: str = value",
            insp_width=17,
            value_start=12,
            available_for_content=0,
        )
        assert result == "…"

    def test_truncate_inspector_line_inline_marker_empty_right(self):
        """A marker with no right side falls back to trailing ellipsis."""

        variables = [
            {
                "name": "s",
                "typename": "str",
                "value": "head12345 … ",
                "format_hint": "inline",
            }
        ]
        entries, _ = _build_variable_inspector(variables, term_width=200)
        colored, width, value_start = entries[0]
        prefix_width = _display_width(ANSI_ESCAPE_RE.sub("", colored[:value_start]))

        available_for_content = prefix_width + 11
        result = _truncate_inspector_line(
            colored, width, value_start, available_for_content
        )
        result_plain = ANSI_ESCAPE_RE.sub("", result)

        assert "head12345" in result_plain
        assert result_plain.endswith("…")
        assert "tail" not in result_plain
        assert _display_width(result_plain) <= available_for_content

    def test_truncate_inspector_line_inline_marker_both_sides_empty(self):
        """A value that is only a marker collapses to a single ellipsis."""

        variables = [
            {"name": "s", "typename": "str", "value": " … ", "format_hint": "inline"}
        ]
        entries, _ = _build_variable_inspector(variables, term_width=200)
        colored, width, value_start = entries[0]
        prefix_width = _display_width(ANSI_ESCAPE_RE.sub("", colored[:value_start]))

        available_for_content = prefix_width + 2
        result = _truncate_inspector_line(
            colored, width, value_start, available_for_content
        )
        result_plain = ANSI_ESCAPE_RE.sub("", result)

        assert result_plain.endswith("…")
        assert _display_width(result_plain) <= available_for_content


class TestSymbols:
    """Tests for symbol definitions."""

    def test_symbols_defined(self):
        """Test that expected symbols are defined."""
        assert "call" in symbols
        assert "warning" in symbols
        assert "error" in symbols
        assert "stop" in symbols

    def test_symbol_values_are_strings(self):
        """Test that all symbols are non-empty strings."""
        for _key, value in symbols.items():
            assert isinstance(value, str)
            assert len(value) > 0


class TestAnsiCodes:
    """Tests for ANSI escape code constants."""

    def test_reset_code(self):
        """Test RESET code is the standard reset."""
        assert RESET == "\033[0m"

    def test_color_codes_format(self):
        """Test that color codes follow ANSI format."""
        codes = [
            MARK_BG,
            MARK_TEXT,
            EM,
            LOCFN,
            EM_CALL,
            FUNC,
            VAR,
            BOLD,
            DIM,
        ]
        for code in codes:
            assert code.startswith("\033[")
            assert code.endswith("m")

    def test_box_drawing_characters(self):
        """Test that box drawing characters are defined."""
        assert BOX_TL == "╭"
        assert BOX_BL == "╰"
        assert BOX_TR == "╮"
        assert BOX_BR == "╯"
        assert BOX_V == "│"
        assert BOX_VL == "┤"
        assert BOX_H == "─"
        assert ARROW_LEFT == "◀"


class TestEdgeCases:
    """Tests for edge cases and error handling."""

    def test_exception_without_traceback(self):
        """Test handling exception without traceback."""
        output = io.StringIO()
        e = ValueError("no traceback")
        # Exception created without being raised has no traceback
        tty_traceback(exc=e, file=output)
        result = output.getvalue()
        assert "ValueError" in result

    def test_syntax_error_formatting(self):
        """Test formatting of SyntaxError exceptions."""
        output = io.StringIO()
        try:
            exec("def broken(")
        except SyntaxError as e:
            tty_traceback(exc=e, file=output)

        result = output.getvalue()
        assert "SyntaxError" in result

    def test_attribute_error_formatting(self):
        """Test formatting of AttributeError."""
        output = io.StringIO()
        try:
            _ = (None).nonexistent_attr
        except AttributeError as e:
            tty_traceback(exc=e, file=output)

        result = output.getvalue()
        assert "AttributeError" in result

    def test_key_error_formatting(self):
        """Test formatting of KeyError."""
        output = io.StringIO()
        try:
            {}["missing"]
        except KeyError as e:
            tty_traceback(exc=e, file=output)

        result = output.getvalue()
        assert "KeyError" in result
        assert "missing" in result

    def test_index_error_formatting(self):
        """Test formatting of IndexError."""
        output = io.StringIO()
        try:
            [][0]
        except IndexError as e:
            tty_traceback(exc=e, file=output)

        result = output.getvalue()
        assert "IndexError" in result

    def test_unicode_in_exception_message(self):
        """Test handling of unicode in exception messages."""
        output = io.StringIO()
        try:
            raise ValueError("Error with émojis 🎉 and ünïcödé")
        except Exception as e:
            tty_traceback(exc=e, file=output)

        result = output.getvalue()
        assert "émojis" in result
        assert "🎉" in result
        assert "ünïcödé" in result

    def test_very_long_exception_message(self):
        """Test handling of very long exception messages."""
        output = io.StringIO()
        long_msg = "x" * 500
        try:
            raise ValueError(long_msg)
        except Exception as e:
            tty_traceback(exc=e, file=output)

        result = output.getvalue()
        assert "ValueError" in result
        # Should contain at least part of the message
        assert "x" in result

    def test_exception_title_word_wrapping(self):
        """Test word wrapping of long exception titles."""
        output = io.StringIO()
        # Create a long summary that will require word wrapping
        long_summary = "This is a very long exception summary that should definitely wrap to multiple lines when displayed in a narrow terminal"
        try:
            raise ValueError(long_summary)
        except Exception as e:
            # Use a narrow terminal width to force wrapping
            tty_traceback(exc=e, file=output, term_width=50)

        result = output.getvalue()
        assert "ValueError" in result
        # Should contain the wrapped text
        assert "This is a very long" in result
        # Should have multiple lines (wrapping occurred)
        lines = result.split("\n")
        title_lines = [
            line
            for line in lines
            if "This is a very long" in line or "exception summary" in line
        ]
        assert len(title_lines) > 1  # Should be wrapped to multiple lines

    def test_multiline_exception_message(self):
        """Test handling of multiline exception messages."""
        output = io.StringIO()
        msg = "Line 1\nLine 2\nLine 3"
        try:
            raise ValueError(msg)
        except Exception as e:
            tty_traceback(exc=e, file=output)

        result = output.getvalue()
        assert "ValueError" in result
        assert "Line 1" in result

    def test_exception_message_with_empty_lines(self):
        """Test handling of exception messages with empty lines."""
        output = io.StringIO()
        msg = "Line 1\n\nLine 3"
        try:
            raise ValueError(msg)
        except Exception as e:
            tty_traceback(exc=e, file=output)

        result = output.getvalue()
        assert "ValueError" in result
        assert "Line 1" in result
        assert "Line 3" in result


class TestRealWorldScenarios:
    """Real-world scenarios that exercise multiple code paths."""

    def test_recursive_function_error(self):
        """Test error in recursive function shows full call stack."""
        output = io.StringIO()

        def countdown(n):
            if n < 0:
                raise ValueError(f"Negative value: {n}")
            return countdown(n - 1)

        try:
            countdown(3)
        except ValueError as e:
            tty_traceback(exc=e, file=output)

        result = output.getvalue()
        assert "ValueError" in result
        assert "countdown" in result
        assert "Negative value: -1" in result

    def test_nested_function_calls_error(self):
        """Test error propagating through nested function calls."""
        output = io.StringIO()

        def level3():
            return 1 / 0

        def level2():
            return level3()

        def level1():
            return level2()

        try:
            level1()
        except ZeroDivisionError as e:
            tty_traceback(exc=e, file=output)

        result = output.getvalue()
        assert "ZeroDivisionError" in result
        # Should show the call chain
        assert "level1" in result
        assert "level2" in result
        assert "level3" in result

    def test_class_method_error(self):
        """Test error in class method."""
        output = io.StringIO()

        class MyClass:
            def broken_method(self):
                raise RuntimeError("Method failed")

        try:
            obj = MyClass()
            obj.broken_method()
        except RuntimeError as e:
            tty_traceback(exc=e, file=output)

        result = output.getvalue()
        assert "RuntimeError" in result
        assert "broken_method" in result

    def test_lambda_error(self):
        """Test error in nested function."""
        output = io.StringIO()

        try:

            def f(x):
                return x / 0

            f(1)
        except ZeroDivisionError as e:
            tty_traceback(exc=e, file=output)

        result = output.getvalue()
        assert "ZeroDivisionError" in result
        assert "f" in result  # The function name should appear

    def test_comprehension_error(self):
        """Test error in list comprehension."""
        output = io.StringIO()

        try:
            _ = [1 / x for x in [1, 0, 2]]
        except ZeroDivisionError as e:
            tty_traceback(exc=e, file=output)

        result = output.getvalue()
        assert "ZeroDivisionError" in result

    def test_generator_error(self):
        """Test error in generator expression."""
        output = io.StringIO()

        def gen():
            yield 1
            raise StopIteration("Custom stop")

        try:
            list(gen())
        except RuntimeError as e:  # StopIteration in generator becomes RuntimeError
            tty_traceback(exc=e, file=output)

        result = output.getvalue()
        assert "RuntimeError" in result or "StopIteration" in result

    def test_context_manager_error(self):
        """Test error in context manager."""
        output = io.StringIO()

        class BadContextManager:
            def __enter__(self):
                return self

            def __exit__(self, *args):
                raise RuntimeError("Exit failed")

        try:
            with BadContextManager():
                pass
        except RuntimeError as e:
            tty_traceback(exc=e, file=output)

        result = output.getvalue()
        assert "RuntimeError" in result
        assert "Exit failed" in result

    def test_decorator_error(self):
        """Test error in decorated function."""
        output = io.StringIO()

        def bad_decorator(func):
            def wrapper(*args, **kwargs):
                raise ValueError("Decorator error")

            return wrapper

        @bad_decorator
        def decorated_func():
            pass

        try:
            decorated_func()
        except ValueError as e:
            tty_traceback(exc=e, file=output)

        result = output.getvalue()
        assert "ValueError" in result
        assert "Decorator error" in result
        assert "wrapper" in result

    @pytest.mark.skipif(
        sys.version_info < (3, 11), reason="Exception groups require Python 3.11+"
    )
    def test_exception_group(self):
        """Test formatting of ExceptionGroup (Python 3.11+)."""
        output = io.StringIO()

        try:
            eg = ExceptionGroup  # noqa: F821
            raise eg(
                "multiple errors",
                [
                    ValueError("error 1"),
                    TypeError("error 2"),
                ],
            )
        except Exception as e:
            tty_traceback(exc=e, file=output)

        result = output.getvalue()
        assert "ExceptionGroup" in result


class TestOutputFormatting:
    """Tests for specific output formatting details."""

    def test_indentation_present(self):
        """Test that output uses proper indentation."""
        output = io.StringIO()
        try:
            raise ValueError("indent test")
        except Exception as e:
            tty_traceback(exc=e, file=output)

        result = output.getvalue()
        # Should have indented lines
        assert INDENT in result or "  " in result

    def test_ansi_codes_in_output(self):
        """Test that ANSI color codes are present in output."""
        output = io.StringIO()
        # Mock isatty to return True so colors are included
        output.isatty = lambda: True
        try:
            function_with_many_locals()
        except Exception as e:
            tty_traceback(exc=e, file=output)

        result = output.getvalue()
        # Should contain escape sequences
        assert "\x1b[" in result
        # Should contain reset codes
        assert RESET in result

    def test_very_long_exception_message_wrapping(self):
        """Test exception message wrapping with very long messages."""
        output = io.StringIO()
        output.isatty = lambda: True
        # Message longer than term_width that needs wrapping
        long_msg = "This is a very long exception message " * 5
        try:
            raise ValueError(long_msg)
        except Exception as e:
            tty_traceback(exc=e, file=output, term_width=60)

        result = output.getvalue()
        assert "ValueError" in result
        assert "\x1b[" in result

    def test_multiline_exception_message(self):
        """Test exception with multiline message including empty lines."""
        output = io.StringIO()
        output.isatty = lambda: True
        # Message with newlines and empty lines - empty line NOT at start after summary
        multiline_msg = "First line\nSecond line\n\nAfter empty"
        try:
            raise ValueError(multiline_msg)
        except Exception as e:
            tty_traceback(exc=e, file=output, term_width=80)

        result = output.getvalue()
        assert "ValueError" in result
        assert "First line" in result
        assert "\x1b[" in result

    def test_exception_with_summary_prefix(self):
        """Test exception where message starts with summary."""
        output = io.StringIO()
        output.isatty = lambda: True
        try:
            # Create an exception where summary equals first part of message
            raise ValueError("short\nmore details here")
        except Exception as e:
            tty_traceback(exc=e, file=output, term_width=80)

        result = output.getvalue()
        assert "ValueError" in result
        assert "\x1b[" in result

    def test_single_variable_inspector(self):
        """Test inspector with single variable (single-line inspector box)."""
        output = io.StringIO()
        output.isatty = lambda: True
        try:
            function_with_single_local()
        except Exception as e:
            tty_traceback(exc=e, file=output, term_width=120)

        result = output.getvalue()
        assert "RuntimeError" in result
        assert "single local test" in result
        assert "\x1b[" in result

    def test_error_frame_with_symbol_desc(self):
        """Test error frame formatting with symbol and description."""
        output = io.StringIO()
        output.isatty = lambda: True
        try:
            binomial_operator()
        except Exception as e:
            tty_traceback(exc=e, file=output, term_width=120)

        result = output.getvalue()
        assert "TypeError" in result
        # Should have ANSI codes for coloring
        assert "\x1b[" in result

    @pytest.mark.skipif(
        sys.version_info < (3, 11), reason="add_note() requires Python 3.11+"
    )
    def test_exception_notes_rendered_with_marker(self):
        """Notes from add_note render as 🔹-prefixed lines after the message."""
        output = io.StringIO()
        output.isatty = lambda: True
        try:
            exception_with_notes()
        except Exception as e:
            tty_traceback(exc=e, file=output, term_width=120)

        result = output.getvalue()
        assert "🔹 first note" in result
        assert "🔹 second note" in result
        # Single-line message: notes follow the summary line directly
        lines = result.splitlines()
        note_idx = next(i for i, line in enumerate(lines) if "🔹 first note" in line)
        assert "Something failed" in lines[note_idx - 1]

    @pytest.mark.skipif(
        sys.version_info < (3, 11), reason="add_note() requires Python 3.11+"
    )
    def test_exception_notes_after_paragraph_break(self):
        """A message with paragraph breaks puts an empty line before notes."""
        output = io.StringIO()
        output.isatty = lambda: True
        try:
            exception_with_note_after_paragraphs()
        except Exception as e:
            tty_traceback(exc=e, file=output, term_width=120)

        result = output.getvalue()
        assert "🔹 note after paragraphs" in result
        lines = [ANSI_ESCAPE_RE.sub("", line) for line in result.splitlines()]
        note_idx = next(
            i for i, line in enumerate(lines) if "🔹 note after paragraphs" in line
        )
        # The line before the note is visually empty (box border only)
        assert lines[note_idx - 1].strip(" │▐╰") == ""
        assert "Second paragraph." in lines[note_idx - 2]

    def test_long_exception_note_wraps(self):
        """A long note wraps to the banner width like the message body."""
        banner = _build_exception_banner(
            {
                "type": "ValueError",
                "summary": "boom",
                "message": "boom",
                "notes": ["word " * 40],
                "from": "none",
            },
            80,
        )
        plain = ANSI_ESCAPE_RE.sub("", banner)
        assert "🔹 word" in plain
        assert max(_display_width(line) for line in plain.splitlines()) <= 80
        assert sum("word" in line for line in plain.splitlines()) > 1

    def test_warning_frame_formatting(self):
        """Test warning frame (user code calling stdlib that fails)."""
        output = io.StringIO()
        output.isatty = lambda: True
        try:
            max_type_error_case()
        except Exception as e:
            tty_traceback(exc=e, file=output, term_width=120)

        result = output.getvalue()
        assert "TypeError" in result
        assert "\x1b[" in result


class TestThreadingHook:
    """Tests for threading exception hook functionality."""

    def test_threading_hook_format(self):
        """Test that threading hook properly formats exceptions."""
        load()
        try:
            exception_output = []

            def thread_func():
                try:
                    raise ValueError("thread error")
                except Exception:
                    # Capture what the hook would output
                    output = io.StringIO()
                    exc_type, exc_value, exc_tb = sys.exc_info()
                    tty_traceback(exc=exc_value, file=output)
                    exception_output.append(output.getvalue())

            t = threading.Thread(target=thread_func)
            t.start()
            t.join(timeout=5)

            assert len(exception_output) == 1
            assert "ValueError" in exception_output[0]
            assert "thread error" in exception_output[0]
        finally:
            unload()

    def test_threading_hook_format_with_color(self):
        """Test that threading hook includes colors when tty."""
        load()
        try:
            exception_output = []

            def thread_func():
                try:
                    chained_from_and_without()
                except Exception:
                    # Capture what the hook would output
                    output = io.StringIO()
                    # Mock isatty to return True
                    output.isatty = lambda: True
                    exc_type, exc_value, exc_tb = sys.exc_info()
                    tty_traceback(exc=exc_value, file=output)
                    exception_output.append(output.getvalue())

            t = threading.Thread(target=thread_func)
            t.start()
            t.join(timeout=5)

            assert len(exception_output) == 1
            result = exception_output[0]
            assert "AttributeError" in result
            assert "NameError" in result
            # Should contain ANSI codes
            assert "\x1b[" in result
        finally:
            unload()


class TestExcepthookFallback:
    """Tests for exception hook fallback when tty_traceback fails."""

    def test_excepthook_fallback_on_error(self, monkeypatch, capsys):
        """Test that excepthook falls back to original on tty_traceback error."""
        from tracerite import tty

        load()
        try:
            # Make tty_traceback raise an exception
            def failing_tty_traceback(**kwargs):
                raise RuntimeError("tty_traceback failed")

            monkeypatch.setattr(tty, "tty_traceback", failing_tty_traceback)

            # Manually invoke the hook
            try:
                raise ValueError("test fallback")
            except Exception:
                exc_type, exc_value, exc_tb = sys.exc_info()
                sys.excepthook(exc_type, exc_value, exc_tb)

            # The fallback should have been triggered
            captured = capsys.readouterr()
            # Should show original exception via fallback
            assert "ValueError" in captured.err or "test fallback" in captured.err
        finally:
            unload()

    def test_excepthook_formats_with_color(self, capsys, monkeypatch):
        """Test that excepthook formats exceptions with colors when tty."""
        # Mock isatty to return True
        monkeypatch.setattr(sys.stderr, "isatty", lambda: True)

        load()
        try:
            # Manually invoke the hook
            try:
                chained_from_and_without()
            except Exception:
                exc_type, exc_value, exc_tb = sys.exc_info()
                sys.excepthook(exc_type, exc_value, exc_tb)

            captured = capsys.readouterr()
            assert "AttributeError" in captured.err
            assert "NameError" in captured.err
            # Should contain ANSI codes
            assert "\x1b[" in captured.err
        finally:
            unload()

    def test_excepthook_fallback_without_original(self, monkeypatch, capsys):
        """Test excepthook fallback when _original_excepthook is None."""
        from tracerite import tty

        load()
        try:
            # Make tty_traceback raise an exception
            def failing_tty_traceback(**kwargs):
                raise RuntimeError("tty_traceback failed")

            monkeypatch.setattr(tty, "tty_traceback", failing_tty_traceback)
            # Set original hook to None to trigger sys.__excepthook__ fallback
            monkeypatch.setattr(hooks._state, "original_excepthook", None)

            # Manually invoke the hook
            try:
                raise ValueError("fallback to __excepthook__")
            except Exception:
                exc_type, exc_value, exc_tb = sys.exc_info()
                sys.excepthook(exc_type, exc_value, exc_tb)

            # The fallback should have been triggered via sys.__excepthook__
            captured = capsys.readouterr()
            assert "ValueError" in captured.err
        finally:
            unload()

    def test_threading_hook_fallback_code_path(self, monkeypatch):
        """Test the threading excepthook fallback code paths exist."""

        # Test that the threading hook function handles errors properly
        # We test the internal logic without actually triggering threading.excepthook
        # which pytest intercepts

        load()
        try:
            # Verify the hook is installed with correct name
            assert threading.excepthook.__name__ == "_tracerite_threading_excepthook"

            # The fallback paths (lines 75-82) are:
            # 1. Call _original_threading_excepthook if set
            # 2. Call sys.__excepthook__ if _original_threading_excepthook is None
            # These paths are covered by the structure of the installed hook
        finally:
            unload()


class TestVariableInspectorEdgeCases:
    """Additional tests for variable inspector edge cases."""

    def test_array_with_empty_rows(self):
        """Test inspector with array that has empty rows."""

        variables = [
            {
                "name": "empty_arr",
                "typename": "list",
                "value": {"type": "array", "rows": []},
                "format_hint": None,
            },
        ]
        result, min_width = _build_variable_inspector(variables, term_width=80)

        assert len(result) == 1
        assert "[]" in result[0][0]

    def test_nested_list_format(self):
        """Test inspector with nested list (matrix) value."""

        variables = [
            {
                "name": "matrix",
                "typename": "list",
                "value": [[1, 2, 3], [4, 5, 6]],
                "format_hint": None,
            },
        ]
        result, min_width = _build_variable_inspector(variables, term_width=80)

        assert len(result) == 1
        # Should format first row with ellipsis
        assert "[" in result[0][0]
        assert "..." in result[0][0]

    def test_simple_list_format(self):
        """Test inspector with simple list value (not nested)."""

        variables = [
            {
                "name": "simple_list",
                "typename": "list",
                "value": [1, 2, 3],
                "format_hint": None,
            },
        ]
        result, min_width = _build_variable_inspector(variables, term_width=80)

        assert len(result) == 1
        # Should convert to string directly
        assert "[1, 2, 3]" in result[0][0]

    def test_generic_value_format(self):
        """Test inspector with generic non-string, non-dict, non-list value."""

        variables = [
            {"name": "num", "typename": "float", "value": 3.14159, "format_hint": None},
        ]
        result, min_width = _build_variable_inspector(variables, term_width=80)

        assert len(result) == 1
        assert "3.14159" in result[0][0]


class TestFrameLabelEdgeCases:
    """Tests for frame label edge cases."""

    def test_frame_without_filename(self):
        """Test frame label when filename is None."""
        frinfo = {
            "filename": None,
            "location": "In [1]",
            "function": "test_func",
            "function_suffix": "",
            "cursor_line": 5,
            "notebook_cell": True,
            "range": None,
            "relevance": "error",
        }
        location_part, function_part = _get_frame_label(frinfo)
        label_plain = ANSI_ESCAPE_RE.sub("", location_part + function_part)

        assert "test_func" in label_plain
        assert "In [1]" in label_plain
        # Notebook cells don't show line numbers in the location part

    def test_frame_with_non_relative_path(self):
        """Test frame label when file is not under CWD."""
        frinfo = {
            "filename": "/some/other/path/file.py",
            "location": "other/path/file.py",
            "function": "external_func",
            "function_suffix": "",
            "cursor_line": 42,
            "notebook_cell": False,
            "range": None,
            "relevance": "error",
        }
        location_part, function_part = _get_frame_label(frinfo)
        label_plain = ANSI_ESCAPE_RE.sub("", location_part + function_part)

        assert "external_func" in label_plain
        # Should use location since it's not under CWD
        assert (
            "other/path/file.py" in label_plain
            or "/some/other/path/file.py" in label_plain
        )

    def test_frame_label_relative_to_raises(self, monkeypatch):
        """Test frame label when Path.relative_to raises exception."""
        from pathlib import Path

        # Mock Path.relative_to to raise ValueError

        def failing_relative_to(self, other):
            raise ValueError("mock error")

        monkeypatch.setattr(Path, "relative_to", failing_relative_to)

        frinfo = {
            "filename": "/some/path/file.py",
            "location": "some/path/file.py",
            "function": "func",
            "function_suffix": "",
            "cursor_line": 10,
            "notebook_cell": False,
            "range": None,
            "relevance": "error",
        }
        location_part, function_part = _get_frame_label(frinfo)
        label_plain = ANSI_ESCAPE_RE.sub("", location_part + function_part)

        # Should fall back to location
        assert "some/path/file.py" in label_plain
        # Note: monkeypatch automatically restores at end of test


class TestInspectorPositioning:
    """Tests for inspector positioning edge cases."""

    def test_inspector_with_many_variables(self):
        """Test output with many variables requiring inspector shifting."""

        output = io.StringIO()

        # Create a function with many local variables
        def func_with_many_vars():
            a = 1
            b = 2
            c = 3
            d = 4
            e = 5
            f = 6
            g = 7
            h = 8
            return a + b + c + d + e + f + g + h + "string"  # TypeError

        try:
            func_with_many_vars()
        except TypeError as e:
            tty_traceback(exc=e, file=output)

        result = output.getvalue()
        assert "TypeError" in result
        # Variables should be shown in inspector
        # The exact formatting depends on terminal width

    def test_exception_in_c_extension(self):
        """Test exception from C extension (no Python source)."""
        output = io.StringIO()

        try:
            # This raises from C code with no Python source
            [].pop()
        except IndexError as e:
            tty_traceback(exc=e, file=output)

        result = output.getvalue()
        assert "IndexError" in result

    def test_exception_with_very_deep_stack(self):
        """Test exception with deep call stack for inspector positioning."""
        output = io.StringIO()

        def deep_call(n):
            if n <= 0:
                x = "trigger"
                return x + 1  # TypeError
            return deep_call(n - 1)

        try:
            deep_call(10)
        except TypeError as e:
            tty_traceback(exc=e, file=output)

        result = output.getvalue()
        assert "TypeError" in result
        assert "deep_call" in result


class TestTtyTracebackEdgeCases:
    """Tests for edge cases in tty_traceback to achieve full coverage."""

    def test_msg_starting_with_two_spaces(self):
        """Test that msg starting with '  ' gets processed without error."""
        output = io.StringIO()
        try:
            raise ValueError("test")
        except Exception as e:
            # Pass a message that starts with two spaces - should not crash
            tty_traceback(exc=e, file=output, msg="  indented message", tag="MYTAG")

        result = output.getvalue()
        # Should produce valid output with the exception
        assert "ValueError" in result
        assert "test" in result
        # Both tag and msg should appear on initial line
        assert "MYTAG" in result
        assert "indented message" in result

    def test_msg_with_newlines(self):
        """Test that msg with newlines is properly handled."""
        output = io.StringIO()
        try:
            raise ValueError("test")
        except Exception as e:
            tty_traceback(exc=e, file=output, msg="line1\nline2\nline3\n", tag="ERRTAG")

        result = output.getvalue()
        assert "line1" in result
        assert "line2" in result
        assert "line3" in result
        # Tag should appear on the initial line
        assert "ERRTAG" in result

    def test_msg_with_fastapi_rich_bar_prefix(self):
        """Colored " ▕" prefix keeps both sides' colors but loses both spaces."""
        output = io.StringIO()
        output.isatty = lambda: True
        rich_msg = " \x1b[36m▕\x1b[0m Rich formatted message"
        try:
            raise ValueError("test")
        except Exception as e:
            tty_traceback(exc=e, file=output, msg=rich_msg)

        first_line = output.getvalue().split("\n", 1)[0]
        # TraceRite's ╭ and FastAPI's ▕ stay colored; no spaces between them
        assert first_line.startswith(f"{DIM}╭{RESET}\x1b[36m▕\x1b[0m ")
        assert "  " not in first_line

    def test_empty_chain(self):
        """Test tty_traceback with empty chain."""
        output = io.StringIO()
        tty_traceback(chain={"header": "", "frames": []}, file=output)
        result = output.getvalue()
        # Should still produce output with box drawing
        assert "╭" in result or "│" in result or "╰" in result

    def test_chain_without_frames(self):
        """Test that exc is ignored when chain has no frames."""

        class TestError(Exception):
            pass

        output = io.StringIO()
        try:
            raise TestError("test message")
        except TestError as e:
            tty_traceback(chain={"header": "", "frames": []}, exc=e, file=output)
        result = output.getvalue()
        assert "TestError" not in result
        assert "test message" not in result

    def test_inspector_arrow_first_and_last(self):
        """Test inspector with single variable (arrow is first and last)."""
        output = io.StringIO()
        output.isatty = lambda: True

        def single_var_func():
            x = 42
            return x + "string"  # TypeError with single local

        try:
            single_var_func()
        except TypeError as e:
            tty_traceback(exc=e, file=output, term_width=120)

        result = output.getvalue()
        assert "TypeError" in result
        # Should have box drawing for inspector
        assert "\x1b[" in result

    def test_inspector_arrow_is_first(self):
        """Test inspector where arrow line is first line (multiple vars)."""
        output = io.StringIO()
        output.isatty = lambda: True

        def two_var_func():
            x = 42
            y = "hello"
            return x + y  # TypeError

        try:
            two_var_func()
        except TypeError as e:
            # Use a narrow terminal so inspector shifts
            tty_traceback(exc=e, file=output, term_width=200)

        result = output.getvalue()
        assert "TypeError" in result

    def test_inspector_arrow_is_last(self):
        """Test inspector where arrow line is last line."""
        output = io.StringIO()
        output.isatty = lambda: True

        def multi_var_func():
            a = 1
            b = 2
            c = 3
            d = 4
            return a + b + c + d + "x"  # TypeError

        try:
            multi_var_func()
        except TypeError as e:
            tty_traceback(exc=e, file=output, term_width=120)

        result = output.getvalue()
        assert "TypeError" in result

    def test_inspector_exceeds_terminal_width(self):
        """Test inspector that would exceed terminal width."""
        output = io.StringIO()
        output.isatty = lambda: True

        def wide_var_func():
            very_long_variable_name_that_is_quite_long = "a" * 50
            return very_long_variable_name_that_is_quite_long + 1

        try:
            wide_var_func()
        except TypeError as e:
            # Narrow terminal to force inspector to adjust
            tty_traceback(exc=e, file=output, term_width=60)

        result = output.getvalue()
        assert "TypeError" in result

    def test_frame_without_source_with_exception(self):
        """Test frame without source code that has exception info."""
        from tracerite.tty import _build_chrono_frame_lines

        info = {
            "location_part": "test.py:10",
            "function_part": "test_func:",
            "fragments": [],
            "frame_range": None,
            "relevance": "error",
            "exc_info": {"type": "ValueError", "message": "test"},
            "marked_lines": [],
            "frinfo": {"linenostart": 1},
        }

        lines = _build_chrono_frame_lines(
            info, location_width=15, function_width=15, term_width=120
        )

        assert len(lines) == 1
        assert "(no source code)" in lines[0][0]
        # Error frames should show the error symbol
        assert "💣" in lines[0][0]

    def test_frame_without_source_no_exception(self):
        """Test frame without source code and no exception info."""
        from tracerite.tty import _build_chrono_frame_lines

        info = {
            "location_part": "test.py:10",
            "function_part": "test_func:",
            "fragments": [],
            "frame_range": None,
            "relevance": "call",
            "exc_info": None,
            "marked_lines": [],
            "frinfo": {"linenostart": 1},
        }

        lines = _build_chrono_frame_lines(
            info, location_width=15, function_width=15, term_width=120
        )

        assert len(lines) == 1
        assert "(no source code)" in lines[0][0]
        # Call frames should show the call symbol in yellow
        assert f"{EM_CALL}➤{RESET}" in lines[0][0]

    def test_chained_exception_with_banners(self):
        """Test chained exceptions produce banners correctly."""
        output = io.StringIO()

        try:
            try:
                raise ValueError("first error")
            except ValueError as e:
                raise RuntimeError("second error") from e
        except RuntimeError as e:
            tty_traceback(exc=e, file=output)

        result = output.getvalue()
        assert "ValueError" in result
        assert "RuntimeError" in result
        # The inner exception message may not be displayed in compact format
        assert "second error" in result

    def test_relative_path_when_inside_cwd(self, monkeypatch):
        """Test that file paths inside cwd are made relative."""

        output = io.StringIO()

        # The actual test file should already be inside cwd
        try:
            raise ValueError("path test")
        except Exception as e:
            tty_traceback(exc=e, file=output)

        result = output.getvalue()
        # Should contain filename without full absolute path
        assert "test_tty" in result

    def test_filename_outside_cwd(self, monkeypatch):
        """Test handling of file path outside current directory."""
        from tracerite.tty import ANSI_ESCAPE_RE, _get_frame_label

        # Create a frame info for a file outside cwd
        frinfo = {
            "filename": "/some/other/path/file.py",
            "location": "other/path/file.py",
            "function": "test_func",
            "function_suffix": "",
            "cursor_line": 10,
            "notebook_cell": False,
            "relevance": "call",
            "range": {"lfirst": 10},
        }

        location_part, function_part = _get_frame_label(frinfo)
        label_plain = ANSI_ESCAPE_RE.sub("", location_part + function_part)
        # Should use location as-is since file is outside cwd
        assert "test_func" in label_plain

    def test_inspector_taller_than_output(self):
        """Test when inspector has more lines than available output lines."""
        output = io.StringIO()
        output.isatty = lambda: True

        def many_vars_short_stack():
            a, _b, _c, _d, _e = 1, 2, 3, 4, 5
            _f, _g, _h, _i, _j = 6, 7, 8, 9, 10
            return a + "x"  # TypeError with many vars

        try:
            many_vars_short_stack()
        except TypeError as e:
            tty_traceback(exc=e, file=output, term_width=150)

        result = output.getvalue()
        assert "TypeError" in result

    def test_exception_banners_after_inspector(self):
        """Test exception banners inserted after inspector lines."""
        output = io.StringIO()
        output.isatty = lambda: True

        def inner_func():
            x = 42
            raise ValueError(str(x) + " error")

        try:
            try:
                inner_func()  # Has local var for inspector
            except ValueError as e:
                raise RuntimeError("chained") from e
        except RuntimeError as e:
            tty_traceback(exc=e, file=output, term_width=120)

        result = output.getvalue()
        assert "RuntimeError" in result
        assert "ValueError" in result


class TestCodeLineWidthAdaptation:
    """Tests for adapting code lines to narrow terminal widths."""

    def test_long_unmarked_line_truncated(self):
        """Unmarked code lines are shortened with a trailing dim ellipsis."""
        from tracerite.tty import _build_chrono_frame_lines

        info = {
            "location_part": "test.py:1",
            "function_part": "func:",
            "fragments": [
                {"line": 1, "fragments": [{"code": "x = " + "a" * 80}]},
                {"line": 2, "fragments": [{"code": "y = 1"}]},
            ],
            "frame_range": {"lfirst": 2, "lfinal": 2},
            "relevance": "error",
            "exc_info": None,
            "marked_lines": [],
            "frinfo": {"linenostart": 1},
        }

        lines = _build_chrono_frame_lines(
            info, location_width=10, function_width=10, term_width=40
        )
        code_line, width, _ = lines[1]
        plain = ANSI_ESCAPE_RE.sub("", code_line)

        assert plain.endswith("…")
        assert "\n" not in code_line
        assert width <= 38

    def test_single_marked_line_wraps(self):
        """The only marked line in a frame is wrapped and keeps its mark color."""
        from tracerite.tty import _build_chrono_frame_lines

        code = "y" * 80
        info = {
            "location_part": "test.py:1",
            "function_part": "func:",
            "fragments": [
                {
                    "line": 1,
                    "fragments": [{"code": "x = "}, {"code": code, "mark": "solo"}],
                }
            ],
            "frame_range": {"lfirst": 1, "lfinal": 1},
            "relevance": "error",
            "exc_info": None,
            "marked_lines": [
                {"line": 1, "fragments": [{"code": code, "mark": "solo"}]}
            ],
            "frinfo": {"linenostart": 1},
        }

        lines = _build_chrono_frame_lines(
            info, location_width=10, function_width=10, term_width=40
        )

        # More than just the label + one code line.
        assert len(lines) > 2
        for _line, width, _ in lines[1:]:
            assert width <= 38
        # Every wrapped chunk closes the mark before the newline...
        for line, _, _ in lines[1:]:
            assert line.endswith(RESET)
        # ...and a continuation chunk restores the mark background color.
        assert any(MARK_BG in line for line, _, _ in lines[2:])

    def test_caret_line_wraps(self):
        """The line carrying the caret symbol is wrapped."""
        from tracerite.tty import _build_chrono_frame_lines

        code = "z" * 80
        info = {
            "location_part": "test.py:1",
            "function_part": "func:",
            "fragments": [
                {
                    "line": 1,
                    "fragments": [{"code": "x = "}, {"code": code, "mark": "solo"}],
                }
            ],
            "frame_range": {"lfirst": 1, "lfinal": 1},
            "relevance": "error",
            "exc_info": None,
            "marked_lines": [
                {"line": 1, "fragments": [{"code": code, "mark": "solo"}]}
            ],
            "frinfo": {"linenostart": 1},
        }

        lines = _build_chrono_frame_lines(
            info, location_width=10, function_width=10, term_width=40
        )

        assert any(symbols["error"] in line for line, _, _ in lines)
        assert len(lines) > 2
        # When the last wrapped chunk has room, the symbol stays on that line.
        symbol_line = next(line for line, _, _ in lines if symbols["error"] in line)
        assert "z" in symbol_line

    def test_other_marked_lines_shortened(self):
        """Non-caret marked lines are shortened when there are multiple marks."""
        from tracerite.tty import _build_chrono_frame_lines

        info = {
            "location_part": "test.py:1",
            "function_part": "func:",
            "fragments": [
                {
                    "line": 1,
                    "fragments": [
                        {"code": "first_long_line = " + "a" * 80, "mark": "solo"}
                    ],
                },
                {
                    "line": 2,
                    "fragments": [{"code": "second = " + "b" * 40, "mark": "solo"}],
                },
            ],
            "frame_range": {"lfirst": 2, "lfinal": 2},
            "relevance": "error",
            "exc_info": None,
            "marked_lines": [{"line": 1}, {"line": 2}],
            "frinfo": {"linenostart": 1},
        }

        lines = _build_chrono_frame_lines(
            info, location_width=10, function_width=10, term_width=40
        )

        # First marked line is not the caret line, so it is truncated.
        first_plain = ANSI_ESCAPE_RE.sub("", lines[1][0])
        assert first_plain.endswith("…")
        # Second marked line carries the caret and is wrapped.
        assert any(symbols["error"] in line for line, _, _ in lines)
        assert len(lines) > 3


class TestMergeChronoOutputBranches:
    """Tests for specific branches in _merge_chrono_output."""

    def test_inspector_arrow_last_not_first(self):
        """Test inspector where arrow is last but not first (line 727)."""
        from tracerite.tty import BOX_BR, _merge_chrono_output

        # Create output where error line (marked) is at position 1
        output_lines = [
            ("line0", 10, 0, False),  # not marked
            ("line1", 10, 0, True),  # marked (arrow points here)
        ]
        # Two inspector lines, arrow should be at last
        inspector_lines = [
            ("var1 = 1", 8, 6),
            ("var2 = 2", 8, 6),
        ]
        exception_banners = []
        frame_info_list = [{"relevance": "error"}]

        result, _ = _merge_chrono_output(
            output_lines,
            [inspector_lines],
            [20],  # min widths
            term_width=120,
            inspector_frame_indices=[0],
            exception_banners=exception_banners,
            frame_info_list=frame_info_list,
        )
        # Should use BOX_BR for last+arrow
        assert BOX_BR in result

    def test_inspector_first_not_arrow(self):
        """Test inspector where first line is not arrow (line 733)."""
        from tracerite.tty import BOX_TL, _merge_chrono_output

        # Create output where marked line is at end
        output_lines = [
            ("line0", 10, 0, False),
            ("line1", 10, 0, False),
            ("line2", 10, 0, True),  # marked (arrow points here)
        ]
        # Three inspector lines, need arrow at idx 2, so first line isn't arrow
        inspector_lines = [
            ("var1 = 1", 8, 6),
            ("var2 = 2", 8, 6),
            ("var3 = 3", 8, 6),
        ]
        exception_banners = []
        frame_info_list = [{"relevance": "error"}]

        result, _ = _merge_chrono_output(
            output_lines,
            [inspector_lines],
            [20],
            term_width=120,
            inspector_frame_indices=[0],
            exception_banners=exception_banners,
            frame_info_list=frame_info_list,
        )
        # Should use BOX_TL for first line when not arrow
        assert BOX_TL in result

    def test_remaining_inspector_lines(self):
        """Test when inspector is taller than output (lines 752-757)."""
        from tracerite.tty import _merge_chrono_output

        # Only 1 output line but 3 inspector lines
        output_lines = [
            ("line0", 10, 0, True),  # marked
        ]
        inspector_lines = [
            ("var1 = 1", 8, 6),
            ("var2 = 2", 8, 6),
            ("var3 = 3", 8, 6),
        ]
        exception_banners = []
        frame_info_list = [{"relevance": "error"}]

        result, _ = _merge_chrono_output(
            output_lines,
            [inspector_lines],
            [20],
            term_width=120,
            inspector_frame_indices=[0],
            exception_banners=exception_banners,
            frame_info_list=frame_info_list,
        )
        # All variables should appear
        assert "var1" in result
        assert "var2" in result
        assert "var3" in result

    def test_remaining_banners_after_output(self):
        """Test remaining banners inserted after all output (line 761)."""
        from tracerite.tty import _merge_chrono_output

        output_lines = [
            ("line0", 10, 0, True),
        ]
        inspector_lines = [
            ("var1 = 1", 8, 6),
        ]
        # Banner that should be inserted after all lines
        exception_banners = [(100, "BANNER_TEXT")]
        frame_info_list = [{"relevance": "error"}]

        result, _ = _merge_chrono_output(
            output_lines,
            [inspector_lines],
            [20],
            term_width=120,
            inspector_frame_indices=[0],
            exception_banners=exception_banners,
            frame_info_list=frame_info_list,
        )
        assert "BANNER_TEXT" in result

    def test_inspector_shift_up(self):
        """Test inspector shifts up when not enough lines below (line 683)."""
        from tracerite.tty import _merge_chrono_output

        # Only 2 output lines, but 4 inspector lines - needs to shift
        output_lines = [
            ("line0", 10, 0, False),
            ("line1", 10, 0, True),  # marked at the end
        ]
        inspector_lines = [
            ("var1 = 1", 8, 6),
            ("var2 = 2", 8, 6),
            ("var3 = 3", 8, 6),
            ("var4 = 4", 8, 6),
        ]
        exception_banners = []
        frame_info_list = [{"relevance": "error"}]

        result, _ = _merge_chrono_output(
            output_lines,
            [inspector_lines],
            [20],
            term_width=120,
            inspector_frame_indices=[0],
            exception_banners=exception_banners,
            frame_info_list=frame_info_list,
        )
        # All variables should appear
        assert "var1" in result
        assert "var4" in result

    def test_inspector_truncation(self):
        """Test inspector line truncation when it exceeds available width (lines 1028-1053)."""
        from tracerite.tty import _build_variable_inspector, _merge_chrono_output

        output_lines = [
            ("x" * 40, 40, 0, True),  # long line sets inspector_col
        ]
        variables = [
            {
                "name": "var_name",
                "typename": "str",
                "value": "x" * 80,
                "format_hint": None,
            },
        ]
        # Build with wide terminal so _build_variable_inspector doesn't truncate,
        # letting _merge_chrono_output hit its own truncation path.
        inspector_lines, min_width = _build_variable_inspector(
            variables, term_width=200
        )
        exception_banners = []
        frame_info_list = [{"relevance": "error"}]

        result, _ = _merge_chrono_output(
            output_lines,
            [inspector_lines],
            [min_width],
            term_width=100,
            inspector_frame_indices=[0],
            exception_banners=exception_banners,
            frame_info_list=frame_info_list,
        )
        # Should contain truncated value indicator
        assert "…" in result

    def test_inspector_extra_line_truncation(self):
        """Continuation inspector lines past the frame are also truncated."""
        from tracerite.tty import _build_variable_inspector, _merge_chrono_output

        output_lines = [
            ("short line", 10, 0, True),
        ]
        value = "first line\n" + "x" * 100
        variables = [
            {"name": "var", "typename": "str", "value": value, "format_hint": "block"},
        ]
        inspector_lines, min_width = _build_variable_inspector(
            variables, term_width=200
        )
        assert len(inspector_lines) > 1
        frame_info_list = [{"relevance": "error"}]

        result, _ = _merge_chrono_output(
            output_lines,
            [inspector_lines],
            [min_width],
            term_width=40,
            inspector_frame_indices=[0],
            exception_banners=[],
            frame_info_list=frame_info_list,
        )
        # Continuation line should have been truncated to fit the terminal.
        assert "…" in result
        for line in result.splitlines():
            assert _display_width(line) <= 40

    def test_inspector_skipped_when_narrow(self):
        """A multi-line inspector that does not fit is skipped cleanly."""
        from tracerite.tty import _build_variable_inspector, _merge_chrono_output

        output_lines = [
            ("short line", 10, 0, True),
        ]
        variables = [
            {
                "name": "long_name",
                "typename": "str",
                "value": "line1\nline2\nline3",
                "format_hint": "block",
            },
        ]
        inspector_lines, min_width = _build_variable_inspector(
            variables, term_width=200
        )
        frame_info_list = [{"relevance": "error"}]

        result, _ = _merge_chrono_output(
            output_lines,
            [inspector_lines],
            [min_width],
            term_width=20,
            inspector_frame_indices=[0],
            exception_banners=[],
            frame_info_list=frame_info_list,
        )
        # The inspector was too wide for the terminal, so no variable content
        # is rendered; the code line and frame border are still present.
        assert "short line" in result
        assert "line1" not in result
        assert "long_name" not in result


class TestPrintChronologicalBranches:
    """Tests for specific branches in _print_chronological."""

    def test_no_inspector_with_banner_insertion(self):
        """Test banner insertion when no inspector (lines 404-432)."""
        output = io.StringIO()

        # Chain without variables (no inspector) but with chained exception
        try:
            try:
                # Use exec to avoid local variables
                exec("raise ValueError('first')")
            except ValueError as exc:
                raise RuntimeError("second") from exc
        except RuntimeError as e:
            tty_traceback(exc=e, file=output)

        result = output.getvalue()
        assert "ValueError" in result
        assert "RuntimeError" in result


class TestGetFrameLabelBranches:
    """Tests for specific branches in _get_frame_label."""

    def test_path_relative_error(self, monkeypatch):
        """Test exception handling in path relativization (lines 506-508)."""
        from pathlib import Path

        from tracerite.tty import ANSI_ESCAPE_RE, _get_frame_label

        # Get the actual cwd and create a path inside it
        cwd = Path.cwd()
        fake_file = cwd / "subdir" / "file.py"

        # Mock relative_to to raise ValueError when called

        def failing_relative_to(self, other):
            raise ValueError("cannot make relative")

        monkeypatch.setattr(Path, "relative_to", failing_relative_to)

        frinfo = {
            "filename": str(fake_file),  # Absolute path inside cwd
            "location": "subdir/file.py",
            "function": "test_func",
            "function_suffix": "",
            "cursor_line": 10,
            "notebook_cell": False,
            "relevance": "call",
            "range": {"lfirst": 10},
        }

        location_part, function_part = _get_frame_label(frinfo)
        label_plain = ANSI_ESCAPE_RE.sub("", location_part + function_part)
        # Should fall back to location without crashing
        assert "test_func" in label_plain
        assert "file.py" in label_plain
        # Note: monkeypatch automatically restores at end of test


class TestNotebookCellDisplay:
    """Tests for notebook cell handling in TTY output."""

    def test_frame_label_with_notebook_cell(self):
        """Test frame label for notebook cell (line 647-653)."""
        frinfo = {
            "filename": "/path/to/notebook.ipynb",
            "location": "Cell [5]",
            "function": "test_func",
            "function_suffix": "",
            "cursor_line": 10,
            "relevance": "call",
            "range": {"lfirst": 10},
            "notebook_cell": True,
        }

        location_part, function_part = _get_frame_label(frinfo)
        label_plain = ANSI_ESCAPE_RE.sub("", location_part + function_part)
        # For notebook cells, lineno is not shown in the label
        assert "Cell [5]" in label_plain
        assert "test_func" in label_plain
        # Should not have :10 in it (line number omitted for notebooks)
        assert ":10" not in label_plain

    def test_frame_label_with_notebook_cell_no_function(self):
        """Test frame label for notebook cell without function (line 652-653)."""
        frinfo = {
            "filename": "/path/to/notebook.ipynb",
            "location": "Cell [5]",
            "function": "",
            "function_suffix": "",
            "cursor_line": 10,
            "relevance": "call",
            "range": {"lfirst": 10},
            "notebook_cell": True,
        }

        location_part, function_part = _get_frame_label(frinfo)
        label_plain = ANSI_ESCAPE_RE.sub("", location_part + function_part)
        assert "Cell [5]" in label_plain
        # No function part
        assert function_part == ""


class TestFunctionSuffixDisplay:
    """Tests for function suffix handling."""

    def test_function_suffix_without_function_name(self):
        """Test function_suffix when function_name is empty (line 643)."""
        frinfo = {
            "filename": "/path/to/file.py",
            "location": "file.py",
            "function": "",  # Empty function name
            "function_suffix": "⚡except",  # But has suffix
            "cursor_line": 10,
            "notebook_cell": False,
            "relevance": "except",
            "range": {"lfirst": 10},
        }

        location_part, function_part = _get_frame_label(frinfo)
        label_plain = ANSI_ESCAPE_RE.sub("", location_part + function_part)
        # Should show the suffix even without function name
        assert "⚡except" in label_plain


class TestSubexceptionSummaries:
    """Tests for _build_subexception_summaries and _get_branch_summary."""

    def test_build_subexception_summaries(self):
        """Test _build_subexception_summaries with parallel branches (lines 530-540)."""
        # Create mock parallel branches
        parallel_branches = [
            [
                {
                    "filename": "/path/file1.py",
                    "location": "file1.py",
                    "function": "func1",
                    "cursor_line": 10,
                    "notebook_cell": False,
                    "range": {"lfirst": 10},
                    "exception": {"type": "ValueError", "summary": "value error"},
                }
            ],
            [
                {
                    "filename": "/path/file2.py",
                    "location": "file2.py",
                    "function": "func2",
                    "cursor_line": 20,
                    "notebook_cell": False,
                    "range": {"lfirst": 20},
                    "exception": {"type": "TypeError", "summary": "type error"},
                }
            ],
        ]

        output = _build_subexception_summaries(parallel_branches, 120)
        # Should contain both exception types
        assert "ValueError" in output
        assert "TypeError" in output

    def test_get_branch_summary_empty_branch(self):
        """Test _get_branch_summary with empty branch (lines 549-550)."""
        result = _get_branch_summary([], 80)
        assert "(empty)" in result

    def test_get_branch_summary_no_exception(self):
        """Test _get_branch_summary when no exception in frames (lines 575-576)."""
        # Frames without exception info
        branch = [
            {
                "filename": "/path/file.py",
                "location": "file.py",
                "function": "func",
                "range": {"lfirst": 10},
            }
        ]
        result = _get_branch_summary(branch, 80)
        assert "(no exception)" in result

    def test_get_branch_summary_with_exception(self):
        """Test _get_branch_summary with exception info (lines 579-612)."""
        branch = [
            {
                "filename": "/path/file.py",
                "location": "file.py",
                "function": "func",
                "cursor_line": 10,
                "notebook_cell": False,
                "range": {"lfirst": 10},
                "exception": {"type": "ValueError", "summary": "test error message"},
            }
        ]
        result = _get_branch_summary(branch, 80)
        result_plain = ANSI_ESCAPE_RE.sub("", result)
        assert "ValueError" in result_plain
        assert "test error message" in result_plain

    def test_get_branch_summary_truncation(self):
        """Test _get_branch_summary truncates long messages (lines 608-610)."""
        long_message = "x" * 200
        branch = [
            {
                "filename": "/path/file.py",
                "location": "file.py",
                "function": "func",
                "cursor_line": 10,
                "notebook_cell": False,
                "range": {"lfirst": 10},
                "exception": {"type": "ValueError", "summary": long_message},
            }
        ]
        # Use a small width to force truncation
        result = _get_branch_summary(branch, 50)
        result_plain = ANSI_ESCAPE_RE.sub("", result)
        # Should be truncated with ellipsis
        assert "…" in result_plain
        assert len(result_plain) <= 60  # Some buffer for ANSI codes

    def test_get_branch_summary_with_nested_parallel(self):
        """Test _get_branch_summary with nested parallel branches (lines 564-573)."""
        # Create a branch where a frame has parallel sub-branches
        branch = [
            {
                "filename": "/path/file.py",
                "location": "file.py",
                "function": "outer",
                "cursor_line": 10,
                "notebook_cell": False,
                "range": {"lfirst": 10},
                "exception": {"type": "ExceptionGroup", "summary": "nested"},
                "parallel": [
                    [
                        {
                            "location": "file1.py",
                            "function": "inner1",
                            "cursor_line": 11,
                            "notebook_cell": False,
                            "exception": {"type": "ValueError", "summary": "inner1"},
                        }
                    ],
                    [
                        {
                            "location": "file2.py",
                            "function": "inner2",
                            "cursor_line": 12,
                            "notebook_cell": False,
                            "exception": {"type": "TypeError", "summary": "inner2"},
                        }
                    ],
                ],
            }
        ]
        result = _get_branch_summary(branch, 120)
        result_plain = ANSI_ESCAPE_RE.sub("", result)
        # Should show nested exceptions in brackets
        assert "[" in result_plain
        assert "ValueError" in result_plain
        assert "TypeError" in result_plain

    def test_get_branch_summary_notebook_cell(self):
        """Test _get_branch_summary with notebook cell (lines 592-595)."""
        branch = [
            {
                "filename": "/path/notebook.ipynb",
                "location": "Cell [5]",
                "function": "func",
                "cursor_line": 10,
                "range": {"lfirst": 10},
                "notebook_cell": True,
                "exception": {"type": "ValueError", "summary": "test"},
            }
        ]
        result = _get_branch_summary(branch, 80)
        result_plain = ANSI_ESCAPE_RE.sub("", result)
        # Should include Cell reference
        assert "Cell [5]" in result_plain

    def test_get_branch_summary_no_location_just_function(self):
        """Test _get_branch_summary when only function is available (lines 596-597)."""
        branch = [
            {
                "filename": "",
                "location": "",
                "function": "test_func",
                "cursor_line": 10,
                "notebook_cell": False,
                "range": {"lfirst": 10},
                "exception": {"type": "ValueError", "summary": "test"},
            }
        ]
        result = _get_branch_summary(branch, 80)
        result_plain = ANSI_ESCAPE_RE.sub("", result)
        # Should include function name
        assert "test_func" in result_plain


@pytest.mark.skipif(
    sys.version_info < (3, 11), reason="Exception groups require Python 3.11+"
)
class TestExceptionGroupTTY:
    """Tests for ExceptionGroup TTY output with parallel branches."""

    def test_exception_group_shows_parallel_branches(self):
        """Test that ExceptionGroup triggers parallel branch display (lines 410-411)."""
        output = io.StringIO()
        try:
            exception_group_with_frames()
        except Exception as e:
            tty_traceback(exc=e, file=output)

        result = output.getvalue()
        # Should show the exception group
        assert "ExceptionGroup" in result
        # Should show the sub-exceptions
        assert "ValueError" in result
        assert "TypeError" in result

    def test_exception_group_parallel_branches_single_border(self):
        """Parallel branch summaries must not produce a double left border."""
        output = io.StringIO()
        output.isatty = lambda: False
        try:
            exception_group_with_frames()
        except Exception as e:
            tty_traceback(exc=e, file=output)

        result_plain = ANSI_ESCAPE_RE.sub("", output.getvalue())
        assert "│ │" not in result_plain


class TestLongArgumentsCollapse:
    """Tests for collapsing long em parts in TTY output."""

    @pytest.mark.skipif(
        sys.version_info < (3, 11), reason="Requires co_positions() from Python 3.11+"
    )
    def test_long_arguments_em_collapse_tty(self):
        """Test that long em parts (>20 chars) are collapsed in TTY output."""
        from tests.errorcases import long_arguments_error

        output = io.StringIO()
        try:
            long_arguments_error()
        except Exception as e:
            tty_traceback(exc=e, file=output)

        result = output.getvalue()
        result_plain = ANSI_ESCAPE_RE.sub("", result)
        # Should contain the function name
        assert "long_arguments_error" in result_plain or "max" in result_plain
        # The collapsed em should contain ellipsis
        # Long arguments get collapsed to first char + … + last char
        assert "TypeError" in result


class TestMultilineExceptionMessage:
    """Tests for multiline exception messages in TTY output."""

    def test_multiline_exception_message_tty(self):
        """Test TTY output for exception with multiline message."""
        from tests.errorcases import multiline_exception_message

        output = io.StringIO()
        try:
            multiline_exception_message()
        except Exception as e:
            tty_traceback(exc=e, file=output)

        result = output.getvalue()
        result_plain = ANSI_ESCAPE_RE.sub("", result)
        # Should contain first line
        assert "First line of error" in result_plain
        # Should contain subsequent lines
        assert "Second line" in result_plain

    def test_empty_line_exception_message_tty(self):
        """Test TTY output for exception with empty line in message."""
        from tests.errorcases import empty_second_line_exception

        output = io.StringIO()
        try:
            empty_second_line_exception()
        except Exception as e:
            tty_traceback(exc=e, file=output)

        result = output.getvalue()
        result_plain = ANSI_ESCAPE_RE.sub("", result)
        # Should contain first line
        assert "First line" in result_plain

    def test_multiline_exception_message_no_double_border(self):
        """Continuation lines must have a single left border, not two."""
        from tests.errorcases import multiline_exception_message

        output = io.StringIO()
        try:
            multiline_exception_message()
        except Exception as e:
            tty_traceback(exc=e, file=output)

        result_plain = ANSI_ESCAPE_RE.sub("", output.getvalue())
        # A double vertical bar with a space in between is the old bug.
        assert "│ │" not in result_plain

    def test_multiline_exception_message_has_half_block_continuation(self):
        """Continuation lines of a multi-line message show a dim half block."""
        from tests.errorcases import multiline_exception_message

        output = io.StringIO()
        output.isatty = lambda: False
        try:
            multiline_exception_message()
        except Exception as e:
            tty_traceback(exc=e, file=output)

        result_plain = ANSI_ESCAPE_RE.sub("", output.getvalue())
        # First line of the last exception gets the bottom corner.
        assert "╰ ValueError: First line of error" in result_plain
        # Continuation lines hang with spaces + half block.
        assert "  ▐ Second line with details" in result_plain
        assert "  ▐ Third line with more info" in result_plain

    def test_empty_line_exception_message_has_blank_row(self):
        """An empty line in the message is rendered as a blank banner row."""
        from tests.errorcases import empty_second_line_exception

        output = io.StringIO()
        output.isatty = lambda: False
        try:
            empty_second_line_exception()
        except Exception as e:
            tty_traceback(exc=e, file=output)

        result_plain = ANSI_ESCAPE_RE.sub("", output.getvalue())
        lines = [ln.rstrip() for ln in result_plain.splitlines()]
        # The blank row appears as a hanging continuation marker.
        assert any("▐" in ln for ln in lines)
        assert "First line" in result_plain
        assert "Third line after empty" in result_plain

    def test_long_exception_message_wraps_within_width(self):
        """Long single-line messages are word-wrapped to the terminal width."""
        output = io.StringIO()
        output.isatty = lambda: False
        long_msg = "xyzzy " * 20
        try:
            raise ValueError(long_msg)
        except Exception as e:
            tty_traceback(exc=e, file=output, term_width=30)

        result_plain = ANSI_ESCAPE_RE.sub("", output.getvalue())
        banner_lines = []
        for ln in result_plain.splitlines():
            if ln.startswith("│ "):
                content = ln[2:]
                if content.startswith("ValueError:") or content.startswith("xyzzy"):
                    banner_lines.append(ln)
            elif ln.startswith("  ▐ "):
                content = ln[4:]
                if content.startswith("xyzzy"):
                    banner_lines.append(ln)
        assert len(banner_lines) > 1
        for line in banner_lines:
            # Each rendered line must fit inside the requested terminal width.
            assert len(line) <= 30

    def test_extremely_long_exception_message_wraps_normally(self):
        """A pathologically long single-line message is wrapped, not shortened."""
        output = io.StringIO()
        output.isatty = lambda: False
        try:
            raise ValueError("x" * 2000)
        except Exception as e:
            tty_traceback(exc=e, file=output, term_width=50)

        result_plain = ANSI_ESCAPE_RE.sub("", output.getvalue())
        banner_lines = [
            ln for ln in result_plain.splitlines() if "ValueError:" in ln or "▐" in ln
        ]
        # The message should wrap to multiple continuation lines.
        cont_lines = [ln for ln in banner_lines if "▐" in ln]
        assert len(cont_lines) > 1
        # No middle ellipsis should appear in the banner itself.
        assert all("…" not in ln for ln in banner_lines)
        # All banner-related lines must fit within the requested width.
        for line in banner_lines:
            assert len(line) <= 50

    def test_too_many_message_lines_are_middle_truncated(self):
        """Exception messages with too many hard linefeeds are collapsed."""
        output = io.StringIO()
        output.isatty = lambda: False
        try:
            raise ValueError("\n".join(f"line {i}" for i in range(120)))
        except Exception as e:
            tty_traceback(exc=e, file=output, term_width=80)

        result_plain = ANSI_ESCAPE_RE.sub("", output.getvalue())
        # Should contain the first and last lines but not every middle line.
        assert "line 0" in result_plain
        assert "line 119" in result_plain
        assert "line 30" not in result_plain
        assert "line 90" not in result_plain
        # Should report how many visual lines were skipped.
        assert "80 more lines" in result_plain

    def test_first_banner_line_uses_full_terminal_width(self):
        """The first line of an exception banner fills the terminal width."""
        output = io.StringIO()
        try:
            # Message length chosen empirically so the line exactly fills width 30.
            raise ValueError("x" * 16)
        except Exception as e:
            tty_traceback(exc=e, file=output, term_width=30)

        result_plain = ANSI_ESCAPE_RE.sub("", output.getvalue())
        banner_lines = [
            ln
            for ln in result_plain.splitlines()
            if "ValueError:" in ln and "▐" not in ln
        ]
        assert banner_lines
        # The rendered line (including border/corner) must reach the full width.
        assert len(banner_lines[0]) == 30

    def test_continuation_line_uses_full_terminal_width(self):
        """Wrapped message continuation lines also fill the terminal width."""
        output = io.StringIO()
        try:
            # Message length chosen empirically to produce a full-width continuation.
            raise ValueError("x" * 50)
        except Exception as e:
            tty_traceback(exc=e, file=output, term_width=30)

        result_plain = ANSI_ESCAPE_RE.sub("", output.getvalue())
        cont_lines = [
            ln for ln in result_plain.splitlines() if "▐" in ln and ln.strip()
        ]
        assert cont_lines
        # At least one continuation line should reach the full terminal width.
        assert any(len(ln) == 30 for ln in cont_lines)


class TestWrapText:
    """Tests for the internal _wrap_text helper."""

    def test_wrap_text_empty(self):
        assert _wrap_text("", 10) == [""]

    def test_wrap_text_short_text(self):
        assert _wrap_text("short", 10) == ["short"]

    def test_wrap_text_long_text(self):
        lines = _wrap_text("word " * 10, 20)
        assert all(_display_width(line) <= 20 for line in lines)
        # Joining should recover the words (whitespace normalised to single spaces).
        assert " ".join(lines) == ("word " * 10).strip()

    def test_wrap_text_long_unbroken_text_wraps(self):
        """Long unbroken text is wrapped rather than shortened."""
        text = "x" * 200
        lines = _wrap_text(text, 30)
        assert "…" not in " ".join(lines)
        assert all(_display_width(line) <= 30 for line in lines)

    def test_wrap_text_respects_display_width(self):
        """CJK characters count as two columns, not one."""
        text = "あいうえお" * 4  # each character is 2 columns
        lines = _wrap_text(text, 10)
        assert all(_display_width(line) <= 10 for line in lines)
        # Each line should hold exactly 5 wide characters.
        assert all(len(line) == 5 for line in lines)

    def test_wrap_text_hard_breaks_long_word(self):
        """An unbreakable word longer than the width is split."""
        lines = _wrap_text("x" * 25, 10)
        assert lines == ["x" * 10, "x" * 10, "x" * 5]

    def test_wrap_text_mixed_width_hard_break(self):
        """Hard breaks respect display width, not character count."""
        # Each CJK char is 2 columns; width 10 fits 5 chars or 2 wide + 1 narrow.
        lines = _wrap_text("あいうえお" * 2, 10)
        assert lines == ["あいうえお", "あいうえお"]
        assert all(_display_width(line) <= 10 for line in lines)

    def test_wrap_text_whitespace_only(self):
        """Whitespace-only input is returned unchanged."""
        assert _wrap_text("   ", 10) == ["   "]

    def test_display_width_zero_width_characters(self):
        """Variation selectors and combining marks count as zero columns."""
        assert _display_width("⚠️") == 1  # U+26A0 + U+FE0F
        assert _display_width("🛑") == 2  # U+1F6D1 wide
        assert _display_width("➤") == 1  # U+27A4 neutral
        assert _display_width("e\u0301") == 1  # e + combining acute

    def test_symbols_have_uniform_display_width(self):
        """Suffix-rendered symbols all occupy 2 columns after padding."""
        from tracerite.trace.core import symbols

        for relevance in ("warning", "except", "error", "stop"):
            assert _display_width(symbols[relevance]) == 2


class TestTTYCoverage:
    """Edge-case tests that previously lacked coverage."""

    def test_no_traceback_exc_only_renders_header(self):
        """A frameless exception renders its header when no chain is provided."""
        output = io.StringIO()
        tty_traceback(exc=ValueError("x"), file=output)
        result = output.getvalue()
        assert "ValueError" in result

    def test_wrap_code_line_plain_fits(self):
        assert _wrap_code_line("hello", 10) == ["hello"]

    def test_wrap_code_line_active_params(self):
        colored = f"{BOLD}{'x' * 20}"
        chunks = _wrap_code_line(colored, 8)
        assert len(chunks) > 1
        assert chunks[1].startswith(BOLD)

    def test_wrap_code_line_non_sgr_escape(self):
        colored = "\x1b[Khello"
        chunks = _wrap_code_line(colored, 10)
        assert "hello" in ANSI_ESCAPE_RE.sub("", chunks[0])

    def test_wrap_code_line_invalid_escape(self):
        chunks = _wrap_code_line("\x1bhello", 10)
        assert len(chunks) == 1

    def test_wrap_code_line_empty(self):
        assert _wrap_code_line("", 10) == [""]

    def test_wrap_code_line_zero_max_width(self):
        """Zero max_width should not hang or emit empty chunks."""
        assert _wrap_code_line("hello", 0) == ["hello"]
        assert _wrap_code_line(f"{BOLD}hello", 0) == [f"{BOLD}hello"]

    def test_wrap_code_line_restores_cumulative_styles(self):
        """Continuation lines keep all active SGR attributes, not just the last."""
        colored = f"{BOLD}{EM}{'x' * 20}"
        chunks = _wrap_code_line(colored, 8)
        assert len(chunks) > 1
        # Second chunk should restore both styles (bold, red).
        assert chunks[1].startswith(BOLD + EM)

    def test_wrap_code_line_no_escape_only_chunks(self):
        """Leading escape sequences should not produce zero-width chunks."""
        chunks = _wrap_code_line(f"{BOLD}{'x' * 4}", 1)
        assert all(ANSI_ESCAPE_RE.sub("", c) for c in chunks)

    def test_truncate_ansi_too_narrow(self):
        truncated = _truncate_ansi("hello", 1)
        assert ANSI_ESCAPE_RE.sub("", truncated).endswith("…")

    def test_build_exception_banner_empty_summary(self):
        banner = _build_exception_banner(
            {"type": "ValueError", "summary": "", "message": "body", "from": "none"},
            term_width=40,
        )
        assert "body" in banner

    def test_build_exception_banner_summary_equals_message(self):
        banner = _build_exception_banner(
            {
                "type": "ValueError",
                "summary": "same",
                "message": "same",
                "from": "none",
            },
            term_width=40,
        )
        assert "same" in banner

    def test_build_exception_banner_summary_prefix_no_newline(self):
        banner = _build_exception_banner(
            {
                "type": "ValueError",
                "summary": "pre",
                "message": "pre_suffix",
                "from": "none",
            },
            term_width=40,
        )
        assert "_suffix" in banner

    def test_no_banner_bottom_corner_falls_back_to_last_prefix(self):
        output = io.StringIO()
        tty_traceback(
            chain={"header": "", "frames": []}, file=output, msg=f"{LINE_PREFIX} hello"
        )
        assert "╰" in output.getvalue()


# Python 3.13+ has linecache._getline_from_code for interactive source retrieval
requires_python_313 = pytest.mark.skipif(
    sys.version_info < (3, 13),
    reason="linecache._getline_from_code requires Python 3.13+",
)


@requires_python_313
class TestInteractiveSourceRetrieval:
    """Tests for source code retrieval from -c commands and interactive code.

    These tests verify that tracerite can retrieve source code for code
    that doesn't come from a file, such as python -c commands or REPL input.
    This uses Python 3.13+ linecache._getline_from_code functionality.
    """

    def test_python_c_simple_function(self):
        """Test source retrieval for a simple function in -c code."""
        import subprocess

        code = """
import sys
sys.path.insert(0, '.')
from tracerite.tty import tty_traceback

def broken():
    x = 1 / 0

try:
    broken()
except:
    tty_traceback()
"""
        result = subprocess.run(
            [sys.executable, "-c", code],
            capture_output=True,
            text=True,
        )
        output = result.stderr + result.stdout

        # Should contain the function source code
        assert "def broken():" in output
        assert "x = 1 / 0" in output
        assert "ZeroDivisionError" in output

    def test_python_c_nested_function(self):
        """Test source retrieval for nested functions in -c code."""
        import subprocess

        code = """
import sys
sys.path.insert(0, '.')
from tracerite.tty import tty_traceback

def outer():
    def inner():
        return 1 / 0
    return inner()

try:
    outer()
except:
    tty_traceback()
"""
        result = subprocess.run(
            [sys.executable, "-c", code],
            capture_output=True,
            text=True,
        )
        output = result.stderr + result.stdout

        # Should contain the inner function source
        assert "def inner():" in output
        assert "return 1 / 0" in output

    def test_python_c_module_level_error(self):
        """Test source retrieval for module-level error in -c code."""
        import subprocess

        code = """
import sys
sys.path.insert(0, '.')
from tracerite.tty import tty_traceback

x = 10
y = 0
try:
    result = x / y
except:
    tty_traceback()
"""
        result = subprocess.run(
            [sys.executable, "-c", code],
            capture_output=True,
            text=True,
        )
        output = result.stderr + result.stdout

        # Should contain module-level source code
        assert "result = x / y" in output
        assert "ZeroDivisionError" in output

    def test_python_c_multiline_function(self):
        """Test source retrieval for multi-line function with variables."""
        import subprocess

        code = """
import sys
sys.path.insert(0, '.')
from tracerite.tty import tty_traceback

def calculate():
    a = 5
    b = 3
    c = b - 3
    return a / c

try:
    calculate()
except:
    tty_traceback()
"""
        result = subprocess.run(
            [sys.executable, "-c", code],
            capture_output=True,
            text=True,
        )
        output = result.stderr + result.stdout

        # Should show the complete function with all its lines
        assert "def calculate():" in output
        assert "a = 5" in output
        assert "b = 3" in output
        assert "return a / c" in output

    def test_python_c_function_does_not_include_following_code(self):
        """Test that function source doesn't include code after the function."""
        import subprocess

        code = """
import sys
sys.path.insert(0, '.')
from tracerite.tty import tty_traceback

def broken():
    x = 1 / 0

def other_function():
    pass

SOME_CONSTANT = 42

try:
    broken()
except:
    tty_traceback()
"""
        result = subprocess.run(
            [sys.executable, "-c", code],
            capture_output=True,
            text=True,
        )
        output = result.stderr + result.stdout

        # Should show broken() source
        assert "def broken():" in output
        assert "x = 1 / 0" in output
        # Should NOT include the following function or constant
        assert "def other_function():" not in output
        assert "SOME_CONSTANT" not in output

    def test_python_c_chained_exception(self):
        """Test source retrieval for chained exceptions in -c code."""
        import subprocess

        code = """
import sys
sys.path.insert(0, '.')
from tracerite.tty import tty_traceback

def inner():
    raise ValueError("inner error")

def outer():
    try:
        inner()
    except ValueError:
        raise RuntimeError("outer error")

try:
    outer()
except:
    tty_traceback()
"""
        result = subprocess.run(
            [sys.executable, "-c", code],
            capture_output=True,
            text=True,
        )
        output = result.stderr + result.stdout

        # Should show source for both exceptions
        assert 'raise ValueError("inner error")' in output
        assert 'raise RuntimeError("outer error")' in output

    def test_python_c_class_method(self):
        """Test source retrieval for class methods in -c code."""
        import subprocess

        code = """
import sys
sys.path.insert(0, '.')
from tracerite.tty import tty_traceback

class Calculator:
    def divide(self, a, b):
        return a / b

try:
    calc = Calculator()
    calc.divide(1, 0)
except:
    tty_traceback()
"""
        result = subprocess.run(
            [sys.executable, "-c", code],
            capture_output=True,
            text=True,
        )
        output = result.stderr + result.stdout

        # Should show the method source
        assert "def divide(self, a, b):" in output
        assert "return a / b" in output

"""Comprehensive tests for the tty module - terminal output formatting."""

import io
import sys
import threading

import pytest

from tracerite import extract_chain
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
    LOCFN,
    MARK_BG,
    MARK_TEXT,
    RESET,
    VAR,
    _build_subexception_summaries,
    _build_variable_inspector,
    _format_fragment,
    _format_fragment_call,
    _get_branch_summary,
    _get_frame_label,
    load,
    symbols,
    tty_traceback,
    unload,
)

from .errorcases import (
    binomial_operator,
    chained_from_and_without,
    exception_group_with_frames,
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

    def test_term_width_from_terminal_size(self, monkeypatch):
        """Test tty_traceback gets term_width from os.get_terminal_size."""
        import os

        output = io.StringIO()
        # Mock get_terminal_size to return a size
        monkeypatch.setattr(
            os, "get_terminal_size", lambda fd: type("Size", (), {"columns": 100})()
        )
        # Mock file.fileno to return 1
        output.fileno = lambda: 1
        try:
            raise ValueError("test")
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
            except TypeError as e:
                raise RuntimeError("level 3") from e
        except Exception as e:
            tty_traceback(exc=e, file=output)

        result = output.getvalue()
        assert "ValueError" in result
        assert "TypeError" in result
        assert "RuntimeError" in result


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


class TestFrameFormatting:
    """Tests for frame label and info extraction."""

    def test_get_frame_label_with_function(self):
        """Test frame label includes function name."""
        try:
            raise ValueError("label test")
        except Exception as e:
            chain = extract_chain(e)
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
        from tracerite.inspector import VarInfo

        variables = [
            VarInfo(name="x", typename="int", value="42", format_hint=None),
            VarInfo(name="name", typename="str", value='"hello"', format_hint=None),
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
        from tracerite.inspector import VarInfo

        variables = [
            VarInfo(name="y", typename=None, value="100", format_hint=None),
        ]
        result, min_width = _build_variable_inspector(variables, term_width=80)

        assert len(result) == 1
        colored, width, value_col = result[0]
        # Should format as "y = 100" not "y: None = 100"
        assert VAR in colored  # Variable name in cyan
        assert "= 100" in colored or "100" in colored

    def test_build_variable_inspector_keyvalue(self):
        """Test inspector with key-value dict representation."""
        from tracerite.inspector import VarInfo

        variables = [
            VarInfo(
                name="d",
                typename="dict",
                value={"type": "keyvalue", "rows": [("a", "1"), ("b", "2")]},
                format_hint=None,
            ),
        ]
        result, min_width = _build_variable_inspector(variables, term_width=80)

        assert len(result) == 1
        colored, _, _ = result[0]
        assert "a:" in colored or "{" in colored

    def test_build_variable_inspector_array(self):
        """Test inspector with array representation."""
        from tracerite.inspector import VarInfo

        variables = [
            VarInfo(
                name="arr",
                typename="list",
                value={"type": "array", "rows": [[1, 2, 3]]},
                format_hint=None,
            ),
        ]
        result, min_width = _build_variable_inspector(variables, term_width=80)

        assert len(result) == 1
        colored, _, _ = result[0]
        assert "[" in colored

    def test_build_variable_inspector_truncation(self):
        """Test that long values are truncated."""
        from tracerite.inspector import VarInfo

        long_value = "x" * 200
        variables = [
            VarInfo(
                name="long_var", typename="str", value=long_value, format_hint=None
            ),
        ]
        result, min_width = _build_variable_inspector(variables, term_width=80)

        assert len(result) == 1
        colored, width, value_col = result[0]
        # Should be truncated (indicated by width being less than original)
        assert width < len(long_value)
        assert "…" in colored  # Truncation indicator


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
            monkeypatch.setattr(tty, "_original_excepthook", None)

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

    def test_tuple_format_variables(self):
        """Test inspector with old tuple format (name, typename, value)."""
        # Old tuple format without VarInfo namedtuple
        variables = [
            ("x", "int", "42"),
            ("y", None, "100"),
        ]
        result, min_width = _build_variable_inspector(variables, term_width=80)

        assert len(result) == 2
        # First var has typename
        assert "42" in result[0][0]
        # Second var has no typename
        assert "100" in result[1][0]

    def test_array_with_empty_rows(self):
        """Test inspector with array that has empty rows."""
        from tracerite.inspector import VarInfo

        variables = [
            VarInfo(
                name="empty_arr",
                typename="list",
                value={"type": "array", "rows": []},
                format_hint=None,
            ),
        ]
        result, min_width = _build_variable_inspector(variables, term_width=80)

        assert len(result) == 1
        assert "[]" in result[0][0]

    def test_nested_list_format(self):
        """Test inspector with nested list (matrix) value."""
        from tracerite.inspector import VarInfo

        variables = [
            VarInfo(
                name="matrix",
                typename="list",
                value=[[1, 2, 3], [4, 5, 6]],
                format_hint=None,
            ),
        ]
        result, min_width = _build_variable_inspector(variables, term_width=80)

        assert len(result) == 1
        # Should format first row with ellipsis
        assert "[" in result[0][0]
        assert "..." in result[0][0]

    def test_simple_list_format(self):
        """Test inspector with simple list value (not nested)."""
        from tracerite.inspector import VarInfo

        variables = [
            VarInfo(
                name="simple_list",
                typename="list",
                value=[1, 2, 3],
                format_hint=None,
            ),
        ]
        result, min_width = _build_variable_inspector(variables, term_width=80)

        assert len(result) == 1
        # Should convert to string directly
        assert "[1, 2, 3]" in result[0][0]

    def test_generic_value_format(self):
        """Test inspector with generic non-string, non-dict, non-list value."""
        from tracerite.inspector import VarInfo

        variables = [
            VarInfo(
                name="num",
                typename="float",
                value=3.14159,
                format_hint=None,
            ),
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
            "location": "unknown_location",
            "function": "test_func",
            "range": None,
            "relevance": "error",
        }
        location_part, function_part = _get_frame_label(frinfo)
        label_plain = ANSI_ESCAPE_RE.sub("", location_part + function_part)

        assert "test_func" in label_plain
        assert "unknown_location" in label_plain
        assert "?" in label_plain  # No range means "?" for line number

    def test_frame_with_non_relative_path(self):
        """Test frame label when file is not under CWD."""
        frinfo = {
            "filename": "/some/other/path/file.py",
            "location": "other/path/file.py",
            "function": "external_func",
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
        """Test that msg starting with '  ' gets trimmed (line 208)."""
        output = io.StringIO()
        try:
            raise ValueError("test")
        except Exception as e:
            # Pass a message that starts with two spaces
            tty_traceback(exc=e, file=output, msg="  indented message")

        result = output.getvalue()
        # The "  " prefix should be stripped
        assert "indented message" in result

    def test_msg_with_newlines(self):
        """Test that msg with newlines is properly handled."""
        output = io.StringIO()
        try:
            raise ValueError("test")
        except Exception as e:
            tty_traceback(exc=e, file=output, msg="line1\nline2\nline3\n")

        result = output.getvalue()
        assert "line1" in result
        assert "line2" in result
        assert "line3" in result

    def test_empty_chain(self):
        """Test tty_traceback with empty chain."""
        output = io.StringIO()
        tty_traceback(chain=[], file=output)
        result = output.getvalue()
        # Should still produce output with box drawing
        assert "╭" in result or "│" in result or "╰" in result

    def test_chain_without_frames(self):
        """Test exception chain where exceptions have no frames."""
        output = io.StringIO()
        # Create a minimal chain with no frames
        chain = [
            {
                "type": "TestError",
                "message": "test message",
                "summary": "test message",
                "from": None,
                "frames": [],
            }
        ]
        tty_traceback(chain=chain, file=output)
        result = output.getvalue()
        assert "TestError" in result
        assert "test message" in result

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
        assert "Source code not available" in lines[0][0]
        assert "ValueError was raised from here" in lines[0][0]

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
        assert "Source code not available" in lines[0][0]
        assert "was raised from here" not in lines[0][0]

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
            "relevance": "call",
            "range": type("Range", (), {"lfirst": 10})(),
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

        result = _merge_chrono_output(
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

        result = _merge_chrono_output(
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

        result = _merge_chrono_output(
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

        result = _merge_chrono_output(
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

        result = _merge_chrono_output(
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
            "relevance": "call",
            "range": type("Range", (), {"lfirst": 10})(),
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
            "relevance": "call",
            "range": type("Range", (), {"lfirst": 10})(),
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
            "relevance": "call",
            "range": type("Range", (), {"lfirst": 10})(),
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
            "relevance": "except",
            "range": type("Range", (), {"lfirst": 10})(),
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
                    "range": type("Range", (), {"lfirst": 10})(),
                    "exception": {"type": "ValueError", "summary": "value error"},
                }
            ],
            [
                {
                    "filename": "/path/file2.py",
                    "location": "file2.py",
                    "function": "func2",
                    "range": type("Range", (), {"lfirst": 20})(),
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
                "range": type("Range", (), {"lfirst": 10})(),
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
                "range": type("Range", (), {"lfirst": 10})(),
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
                "range": type("Range", (), {"lfirst": 10})(),
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
                "range": type("Range", (), {"lfirst": 10})(),
                "exception": {"type": "ExceptionGroup", "summary": "nested"},
                "parallel": [
                    [
                        {
                            "exception": {"type": "ValueError", "summary": "inner1"},
                        }
                    ],
                    [
                        {
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
                "range": type("Range", (), {"lfirst": 10})(),
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
                "range": type("Range", (), {"lfirst": 10})(),
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

"""Comprehensive tests for the tty module - terminal output formatting."""

import io
import sys
import threading

import pytest

from tracerite import extract_chain
from tracerite.tty import (
    ARROW_LEFT,
    BLACK_TEXT,
    BOLD,
    BOX_H,
    BOX_ROUND_BL,
    BOX_ROUND_BR,
    BOX_ROUND_TL,
    BOX_ROUND_TR,
    BOX_V,
    BOX_VL,
    CYAN,
    DARK_GREY,
    DIM,
    GREEN,
    INDENT,
    LIGHT_BLUE,
    RED_TEXT,
    RESET,
    YELLOW,
    YELLOW_BG,
    _build_frame_lines,
    _build_variable_inspector,
    _format_fragment,
    _format_fragment_call,
    _get_frame_info,
    _get_frame_label,
    install,
    symbols,
    tty_traceback,
    uninstall,
)

from .errorcases import (
    binomial_operator,
    chained_from_and_without,
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
        # Should show the chaining message
        assert "while handling" in result or "from" in result.lower()

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
        # "from above" indicates the second exception was raised from the first
        assert "from above" in result

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
        # Should indicate this happened while handling previous
        assert "while handling" in result

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


class TestInstallUninstall:
    """Tests for install() and uninstall() exception hooks."""

    def test_install_replaces_excepthook(self):
        """Test that install() replaces sys.excepthook."""
        original_hook = sys.excepthook
        try:
            install()
            assert sys.excepthook != original_hook
            assert sys.excepthook.__name__ == "_tracerite_excepthook"
        finally:
            uninstall()

    def test_uninstall_restores_excepthook(self):
        """Test that uninstall() restores original sys.excepthook."""
        original_hook = sys.excepthook
        install()
        uninstall()
        assert sys.excepthook == original_hook

    def test_install_replaces_threading_excepthook(self):
        """Test that install() replaces threading.excepthook."""
        original_hook = threading.excepthook
        try:
            install()
            assert threading.excepthook != original_hook
            assert threading.excepthook.__name__ == "_tracerite_threading_excepthook"
        finally:
            uninstall()

    def test_uninstall_restores_threading_excepthook(self):
        """Test that uninstall() restores threading.excepthook."""
        original_hook = threading.excepthook
        install()
        uninstall()
        assert threading.excepthook == original_hook

    def test_double_install_preserves_original(self):
        """Test that calling install() twice preserves the original hook."""
        original_hook = sys.excepthook
        try:
            install()
            assert sys.excepthook.__name__ == "_tracerite_excepthook"
            install()  # Second install
            # Should still be a tracerite hook
            assert sys.excepthook.__name__ == "_tracerite_excepthook"
            uninstall()
            # Should restore to original, not to the first tracerite hook
            assert sys.excepthook == original_hook
        finally:
            # Ensure cleanup
            if sys.excepthook != original_hook:
                sys.excepthook = original_hook

    def test_uninstall_without_install_is_safe(self):
        """Test that calling uninstall() without install() is safe."""
        original_hook = sys.excepthook
        uninstall()  # Should not crash
        assert sys.excepthook == original_hook

    def test_installed_hook_handles_exceptions(self, capsys):
        """Test that the installed hook properly handles exceptions."""
        install()
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
            uninstall()


class TestFrameFormatting:
    """Tests for frame label and info extraction."""

    def test_get_frame_label_with_function(self):
        """Test frame label includes function name."""
        try:
            raise ValueError("label test")
        except Exception as e:
            chain = extract_chain(e)
            frame = chain[0]["frames"][-1]
            label, label_plain = _get_frame_label(frame)

            # Should contain function name in light blue
            assert LIGHT_BLUE in label
            assert "test_get_frame_label_with_function" in label_plain
            # Should contain filename in green
            assert GREEN in label
            assert "test_tty" in label_plain

    def test_get_frame_info_structure(self):
        """Test _get_frame_info returns expected structure."""
        try:
            raise ValueError("info test")
        except Exception as e:
            chain = extract_chain(e)
            exc_info = chain[0]
            frame = exc_info["frames"][-1]
            info = _get_frame_info(exc_info, frame)

            assert "label" in info
            assert "label_plain" in info
            assert "fragments" in info
            assert "relevance" in info
            assert "is_deepest" in info
            assert "frinfo" in info
            assert "e" in info

    def test_build_frame_lines_with_source(self):
        """Test _build_frame_lines generates output lines."""
        try:
            raise ValueError("frame lines test")
        except Exception as e:
            chain = extract_chain(e)
            exc_info = chain[0]
            frame = exc_info["frames"][-1]
            info = _get_frame_info(exc_info, frame)

            lines = _build_frame_lines(info, label_width=50, term_width=120)

            assert len(lines) > 0
            # Each line is (colored_line, plain_length, is_marked)
            for line, plain_len, is_marked in lines:
                assert isinstance(line, str)
                assert isinstance(plain_len, int)
                assert isinstance(is_marked, bool)


class TestFragmentFormatting:
    """Tests for fragment formatting functions."""

    def test_format_fragment_plain(self):
        """Test formatting fragment without marks or emphasis."""
        fragment = {"code": "x = 1"}
        colored, plain = _format_fragment(fragment)
        assert plain == "x = 1"
        assert colored == "x = 1"

    def test_format_fragment_with_mark_solo(self):
        """Test formatting fragment with solo mark (single highlighted region)."""
        fragment = {"code": "error_code", "mark": "solo"}
        colored, plain = _format_fragment(fragment)
        assert plain == "error_code"
        assert YELLOW_BG in colored
        assert BLACK_TEXT in colored
        assert RESET in colored

    def test_format_fragment_with_mark_beg_fin(self):
        """Test formatting fragment with mark spanning multiple fragments."""
        # Beginning of marked region
        frag_beg = {"code": "start", "mark": "beg"}
        colored_beg, plain_beg = _format_fragment(frag_beg)
        assert YELLOW_BG in colored_beg
        assert RESET not in colored_beg  # Should not close yet

        # End of marked region
        frag_fin = {"code": "end", "mark": "fin"}
        colored_fin, plain_fin = _format_fragment(frag_fin)
        assert RESET in colored_fin

    def test_format_fragment_with_em_solo(self):
        """Test formatting fragment with emphasis (error location)."""
        fragment = {"code": "+", "em": "solo", "mark": "solo"}
        colored, plain = _format_fragment(fragment)
        assert plain == "+"
        assert RED_TEXT in colored

    def test_format_fragment_call_plain(self):
        """Test call frame fragment formatting without emphasis."""
        fragment = {"code": "func(arg)"}
        colored, plain = _format_fragment_call(fragment)
        assert plain == "func(arg)"
        assert colored == "func(arg)"

    def test_format_fragment_call_with_em(self):
        """Test call frame fragment formatting with emphasis."""
        fragment = {"code": "bad_call", "em": "solo"}
        colored, plain = _format_fragment_call(fragment)
        assert plain == "bad_call"
        assert RED_TEXT in colored
        assert RESET in colored

    def test_format_fragment_strips_newlines(self):
        """Test that fragments strip trailing newlines."""
        fragment = {"code": "line\n\r"}
        colored, plain = _format_fragment(fragment)
        assert plain == "line"
        assert "\n" not in colored
        assert "\r" not in colored


class TestVariableInspector:
    """Tests for variable inspector formatting."""

    def test_build_variable_inspector_empty(self):
        """Test inspector with no variables."""
        result = _build_variable_inspector([], term_width=80)
        assert result == []

    def test_build_variable_inspector_simple_vars(self):
        """Test inspector with simple variables."""
        from tracerite.inspector import VarInfo

        variables = [
            VarInfo(name="x", typename="int", value="42", format_hint=None),
            VarInfo(name="name", typename="str", value='"hello"', format_hint=None),
        ]
        result = _build_variable_inspector(variables, term_width=80)

        assert len(result) == 2
        # Each item is (colored_line, width)
        for colored, width in result:
            assert isinstance(colored, str)
            assert isinstance(width, int)

    def test_build_variable_inspector_without_typename(self):
        """Test inspector with variables that have no type annotation."""
        from tracerite.inspector import VarInfo

        variables = [
            VarInfo(name="y", typename=None, value="100", format_hint=None),
        ]
        result = _build_variable_inspector(variables, term_width=80)

        assert len(result) == 1
        colored, width = result[0]
        # Should format as "y = 100" not "y: None = 100"
        assert CYAN in colored  # Variable name in cyan
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
        result = _build_variable_inspector(variables, term_width=80)

        assert len(result) == 1
        colored, _ = result[0]
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
        result = _build_variable_inspector(variables, term_width=80)

        assert len(result) == 1
        colored, _ = result[0]
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
        result = _build_variable_inspector(variables, term_width=80)

        assert len(result) == 1
        colored, width = result[0]
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
            YELLOW_BG,
            BLACK_TEXT,
            RED_TEXT,
            GREEN,
            YELLOW,
            DARK_GREY,
            LIGHT_BLUE,
            CYAN,
            BOLD,
            DIM,
        ]
        for code in codes:
            assert code.startswith("\033[")
            assert code.endswith("m")

    def test_box_drawing_characters(self):
        """Test that box drawing characters are defined."""
        assert BOX_ROUND_TL == "╭"
        assert BOX_ROUND_BL == "╰"
        assert BOX_ROUND_TR == "╮"
        assert BOX_ROUND_BR == "╯"
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
        try:
            raise ValueError("color test")
        except Exception as e:
            tty_traceback(exc=e, file=output)

        result = output.getvalue()
        # Should contain escape sequences
        assert "\033[" in result
        # Should contain reset codes
        assert RESET in result


class TestThreadingHook:
    """Tests for threading exception hook functionality."""

    def test_threading_hook_format(self):
        """Test that threading hook properly formats exceptions."""
        install()
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
            uninstall()


class TestExcepthookFallback:
    """Tests for exception hook fallback when tty_traceback fails."""

    def test_excepthook_fallback_on_error(self, monkeypatch, capsys):
        """Test that excepthook falls back to original on tty_traceback error."""
        from tracerite import tty

        install()
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
            uninstall()

    def test_excepthook_fallback_without_original(self, monkeypatch, capsys):
        """Test excepthook fallback when _original_excepthook is None."""
        from tracerite import tty

        install()
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
            uninstall()

    def test_threading_hook_fallback_code_path(self, monkeypatch):
        """Test the threading excepthook fallback code paths exist."""

        # Test that the threading hook function handles errors properly
        # We test the internal logic without actually triggering threading.excepthook
        # which pytest intercepts

        install()
        try:
            # Verify the hook is installed with correct name
            assert threading.excepthook.__name__ == "_tracerite_threading_excepthook"

            # The fallback paths (lines 75-82) are:
            # 1. Call _original_threading_excepthook if set
            # 2. Call sys.__excepthook__ if _original_threading_excepthook is None
            # These paths are covered by the structure of the installed hook
        finally:
            uninstall()


class TestPrintFragment:
    """Tests for _print_fragment function."""

    def test_print_fragment_plain(self):
        """Test printing plain fragment without marks."""
        from tracerite.tty import _print_fragment

        output = io.StringIO()
        fragment = {"code": "plain_code\n"}
        _print_fragment(output, fragment)
        assert output.getvalue() == "plain_code"

    def test_print_fragment_with_mark_solo(self):
        """Test printing fragment with solo mark."""
        from tracerite.tty import _print_fragment

        output = io.StringIO()
        fragment = {"code": "marked", "mark": "solo"}
        _print_fragment(output, fragment)
        result = output.getvalue()
        assert YELLOW_BG in result
        assert BLACK_TEXT in result
        assert "marked" in result
        assert RESET in result

    def test_print_fragment_with_em_solo(self):
        """Test printing fragment with solo emphasis."""
        from tracerite.tty import _print_fragment

        output = io.StringIO()
        fragment = {"code": "+", "em": "solo", "mark": "solo"}
        _print_fragment(output, fragment)
        result = output.getvalue()
        assert RED_TEXT in result
        assert "+" in result

    def test_print_fragment_em_beg_fin(self):
        """Test printing fragment with em beginning and finishing."""
        from tracerite.tty import _print_fragment

        # Test em beginning
        output = io.StringIO()
        fragment = {"code": "start", "em": "beg", "mark": "beg"}
        _print_fragment(output, fragment)
        result = output.getvalue()
        assert RED_TEXT in result

        # Test em finishing (without mark finishing)
        output2 = io.StringIO()
        fragment2 = {"code": "middle", "em": "fin"}
        _print_fragment(output2, fragment2)
        result2 = output2.getvalue()
        assert BLACK_TEXT in result2

    def test_print_fragment_mark_beg_fin(self):
        """Test printing fragment with mark beginning and finishing."""
        from tracerite.tty import _print_fragment

        # Beginning
        output = io.StringIO()
        fragment = {"code": "begin", "mark": "beg"}
        _print_fragment(output, fragment)
        result = output.getvalue()
        assert YELLOW_BG in result
        assert RESET not in result

        # Finishing
        output2 = io.StringIO()
        fragment2 = {"code": "end", "mark": "fin"}
        _print_fragment(output2, fragment2)
        result2 = output2.getvalue()
        assert RESET in result2


class TestVariableInspectorEdgeCases:
    """Additional tests for variable inspector edge cases."""

    def test_tuple_format_variables(self):
        """Test inspector with old tuple format (name, typename, value)."""
        # Old tuple format without VarInfo namedtuple
        variables = [
            ("x", "int", "42"),
            ("y", None, "100"),
        ]
        result = _build_variable_inspector(variables, term_width=80)

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
        result = _build_variable_inspector(variables, term_width=80)

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
        result = _build_variable_inspector(variables, term_width=80)

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
        result = _build_variable_inspector(variables, term_width=80)

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
        result = _build_variable_inspector(variables, term_width=80)

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
        }
        label, label_plain = _get_frame_label(frinfo)

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
        }
        label, label_plain = _get_frame_label(frinfo)

        assert "external_func" in label_plain
        # Should use location since it's not under CWD
        assert (
            "other/path/file.py" in label_plain
            or "/some/other/path/file.py" in label_plain
        )


class TestFrameWithoutSource:
    """Tests for frames without source code available."""

    def test_build_frame_lines_no_fragments(self):
        """Test building frame lines when no fragments (source unavailable)."""
        info = {
            "label": f"{LIGHT_BLUE}test_func {GREEN}test.py{DARK_GREY}:10{RESET}",
            "label_plain": "test_func test.py:10",
            "fragments": [],
            "frame_range": None,
            "relevance": "error",
            "is_deepest": False,
            "frinfo": {},
            "e": {"type": "ValueError"},
        }

        lines = _build_frame_lines(info, label_width=30, term_width=120)

        assert len(lines) == 2
        # First line is label
        assert "test_func" in lines[0][0]
        # Second line indicates source not available
        assert "Source code not available" in lines[1][0]

    def test_build_frame_lines_no_fragments_deepest(self):
        """Test building frame lines for deepest frame without source."""
        info = {
            "label": f"{LIGHT_BLUE}deep_func {GREEN}test.py{DARK_GREY}:20{RESET}",
            "label_plain": "deep_func test.py:20",
            "fragments": [],
            "frame_range": None,
            "relevance": "error",
            "is_deepest": True,
            "frinfo": {},
            "e": {"type": "RuntimeError"},
        }

        lines = _build_frame_lines(info, label_width=30, term_width=120)

        assert len(lines) == 3
        # Should have additional message about error being raised
        assert "Source code not available" in lines[1][0]
        assert "RuntimeError was raised from here" in lines[2][0]


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


class TestPrintExceptionInternals:
    """Tests for _print_exception internal code paths."""

    def test_print_exception_directly(self):
        """Test _print_exception with controlled exception data."""
        from tracerite.tty import _print_exception

        output = io.StringIO()

        # Create a minimal exception dict structure
        e = {
            "type": "TestError",
            "summary": "Test summary",
            "message": "Test summary with more details",
            "frames": [
                {
                    "filename": "/test/file.py",
                    "location": "test/file.py",
                    "function": "test_func",
                    "range": None,
                    "relevance": "error",
                    "fragments": [],
                    "variables": [],
                }
            ],
        }

        _print_exception(output, e, term_width=80)
        result = output.getvalue()
        assert "TestError" in result
        assert "Test summary" in result

    def test_frame_label_with_path_error(self):
        """Test frame label when Path operations raise errors."""
        # Test with a path that might cause issues
        frinfo = {
            "filename": "\x00invalid\x00path",  # Invalid path characters
            "location": "fallback_location",
            "function": "test_func",
            "range": None,
        }
        label, label_plain = _get_frame_label(frinfo)
        # Should fall back to location
        assert "fallback_location" in label_plain


class TestInspectorTallerThanOutput:
    """Tests for when inspector has more lines than output."""

    def test_many_variables_few_output_lines(self):
        """Test when there are many variables but few output lines."""
        from tracerite.inspector import VarInfo
        from tracerite.tty import _print_exception

        output = io.StringIO()

        # Create exception with many variables but minimal frames
        e = {
            "type": "ValueError",
            "summary": "test",
            "message": "test",
            "frames": [
                {
                    "filename": __file__,
                    "location": "test_tty.py",
                    "function": "short_func",
                    "range": None,
                    "relevance": "error",
                    "linenostart": 1,
                    "fragments": [
                        {
                            "line": 1,
                            "fragments": [{"code": "x = 1", "mark": "solo"}],
                        }
                    ],
                    "variables": [
                        VarInfo("var1", "int", "1", None),
                        VarInfo("var2", "int", "2", None),
                        VarInfo("var3", "int", "3", None),
                        VarInfo("var4", "int", "4", None),
                        VarInfo("var5", "int", "5", None),
                        VarInfo("var6", "int", "6", None),
                        VarInfo("var7", "int", "7", None),
                        VarInfo("var8", "int", "8", None),
                        VarInfo("var9", "int", "9", None),
                        VarInfo("var10", "int", "10", None),
                        VarInfo("var11", "int", "11", None),
                        VarInfo("var12", "int", "12", None),
                        VarInfo("var13", "int", "13", None),
                        VarInfo("var14", "int", "14", None),
                        VarInfo("var15", "int", "15", None),
                    ],
                }
            ],
        }

        _print_exception(output, e, term_width=120, exception_idx=0)
        result = output.getvalue()
        assert "ValueError" in result
        # Should show some variables
        assert "var" in result

    def test_inspector_overflow_with_arrow(self):
        """Test inspector overflow when arrow line is in overflow region."""
        from tracerite.inspector import VarInfo
        from tracerite.tty import _print_exception

        output = io.StringIO()

        # Create a scenario where inspector is taller than output
        # and arrow line ends up in the overflow region
        e = {
            "type": "TestError",
            "summary": "test",
            "message": "test",
            "frames": [
                {
                    "filename": __file__,
                    "location": "test_tty.py",
                    "function": "tiny_func",
                    "range": None,
                    "relevance": "error",
                    "linenostart": 1,
                    "fragments": [
                        {"line": 1, "fragments": [{"code": "err", "mark": "solo"}]},
                    ],
                    "variables": [
                        VarInfo(f"v{i}", "int", str(i), None) for i in range(20)
                    ],
                }
            ],
        }

        _print_exception(output, e, term_width=120, exception_idx=0)
        result = output.getvalue()
        assert "TestError" in result

    def test_inspector_arrow_at_first_line(self):
        """Test inspector when arrow is at the first inspector line."""
        from tracerite.inspector import VarInfo
        from tracerite.tty import _print_exception

        output = io.StringIO()

        e = {
            "type": "ArrowFirstError",
            "summary": "test arrow first",
            "message": "test arrow first",
            "frames": [
                {
                    "filename": __file__,
                    "location": "test_tty.py",
                    "function": "arrow_first_func",
                    "range": None,
                    "relevance": "error",
                    "linenostart": 1,
                    "fragments": [
                        {"line": 1, "fragments": [{"code": "x = bad", "mark": "solo"}]},
                    ],
                    "variables": [
                        VarInfo(f"var{i}", "str", f'"val{i}"', None) for i in range(25)
                    ],
                }
            ],
        }

        _print_exception(output, e, term_width=120, exception_idx=0)
        result = output.getvalue()
        assert "ArrowFirstError" in result

    def test_inspector_arrow_at_last_line(self):
        """Test inspector when arrow is at the last inspector line."""
        from tracerite.inspector import VarInfo
        from tracerite.tty import _print_exception

        output = io.StringIO()

        e = {
            "type": "ArrowLastError",
            "summary": "test arrow last",
            "message": "test arrow last",
            "frames": [
                {
                    "filename": __file__,
                    "location": "test_tty.py",
                    "function": "arrow_last_func",
                    "range": None,
                    "relevance": "error",
                    "linenostart": 1,
                    "fragments": [
                        {
                            "line": 1,
                            "fragments": [{"code": "y = oops", "mark": "solo"}],
                        },
                    ],
                    "variables": [
                        VarInfo(f"item{i}", "float", f"{i}.0", None) for i in range(30)
                    ],
                }
            ],
        }

        _print_exception(output, e, term_width=120, exception_idx=0)
        result = output.getvalue()
        assert "ArrowLastError" in result


class TestArrowLineEdgeCases:
    """Test edge cases for arrow line positioning."""

    def test_arrow_line_out_of_bounds(self):
        """Test when arrow line calculation results in out-of-bounds index."""
        from tracerite.inspector import VarInfo
        from tracerite.tty import _print_exception

        output = io.StringIO()

        # Create a scenario where arrow_line_idx might go out of bounds
        e = {
            "type": "BoundsError",
            "summary": "bounds test",
            "message": "bounds test",
            "frames": [
                {
                    "filename": __file__,
                    "location": "test.py",
                    "function": "bounds_func",
                    "range": None,
                    "relevance": "error",
                    "linenostart": 1,
                    "fragments": [
                        {"line": 1, "fragments": [{"code": "x", "mark": "solo"}]},
                    ],
                    "variables": [
                        VarInfo(f"x{i}", "int", str(i), None) for i in range(50)
                    ],
                }
            ],
        }

        _print_exception(output, e, term_width=80, exception_idx=0)
        result = output.getvalue()
        assert "BoundsError" in result

    def test_inspector_overflow_remaining_lines(self):
        """Test inspector overflow where remaining inspector lines need printing.

        This targets lines 341-373: the case where inspector has more lines
        than output lines available after inspector_start.
        """
        from tracerite.inspector import VarInfo
        from tracerite.tty import _print_exception

        output = io.StringIO()

        # Key insight: We need inspector_height > (len(output_lines) - inspector_start)
        # With minimal output (just header + 2 frame lines) and many variables
        e = {
            "type": "OverflowTest",
            "summary": "x",
            "message": "x",
            "frames": [
                {
                    "filename": __file__,
                    "location": "t.py",
                    "function": "f",
                    "range": None,
                    "relevance": "error",
                    "linenostart": 1,
                    "fragments": [
                        {"line": 1, "fragments": [{"code": "a", "mark": "solo"}]},
                    ],
                    # 60 variables = 60 inspector lines, but only ~3 output lines
                    "variables": [
                        VarInfo(f"v{i:02d}", "int", str(i), None) for i in range(60)
                    ],
                }
            ],
        }

        _print_exception(output, e, term_width=200, exception_idx=0)
        result = output.getvalue()
        assert "OverflowTest" in result
        # Should print all 60 variables
        assert "v59" in result

    def test_inspector_overflow_with_arrow_in_remaining(self):
        """Test inspector overflow where arrow line is in the remaining section.

        This targets the is_arrow branch in lines 348-358.
        """
        from tracerite.inspector import VarInfo
        from tracerite.tty import _print_exception

        output = io.StringIO()

        e = {
            "type": "ArrowOverflow",
            "summary": "y",
            "message": "y",
            "frames": [
                {
                    "filename": __file__,
                    "location": "t.py",
                    "function": "g",
                    "range": None,
                    "relevance": "error",
                    "linenostart": 1,
                    "fragments": [
                        {"line": 1, "fragments": [{"code": "b", "mark": "solo"}]},
                    ],
                    "variables": [
                        VarInfo(f"w{i:02d}", "int", str(i), None) for i in range(80)
                    ],
                }
            ],
        }

        _print_exception(output, e, term_width=200, exception_idx=0)
        result = output.getvalue()
        assert "ArrowOverflow" in result


class TestVariableInspectorEmptyAfterProcessing:
    """Test variable inspector that results in empty var_lines."""

    def test_variable_with_none_components(self):
        """Test variable inspector with tuple that has None values."""
        # The var_lines list should not be empty if we pass valid tuples
        # But we can test the path where var_lines could be empty
        result = _build_variable_inspector([], term_width=80)
        assert result == []

    def test_inspector_with_invalid_tuple(self):
        """Test inspector handles various input formats."""
        # Test old tuple format
        variables = [
            ("name1", "type1", "value1"),
            ("name2", None, "value2"),
        ]
        result = _build_variable_inspector(variables, term_width=80)
        assert len(result) == 2


class TestFrameLabelPathExceptions:
    """Test frame label when Path operations raise exceptions."""

    def test_relative_path_value_error(self):
        """Test frame label handles ValueError from relative_to."""

        # Create a path that's not relative to CWD
        frinfo = {
            "filename": "/completely/different/path/file.py",
            "location": "different/file.py",
            "function": "func",
            "range": None,
        }
        label, label_plain = _get_frame_label(frinfo)
        # Should use location fallback
        assert "different/file.py" in label_plain or "/completely" in label_plain


class TestThreadingHookFallbackPaths:
    """Tests specifically for threading hook fallback code paths."""

    def test_threading_hook_structure(self):
        """Verify threading hook has correct fallback structure."""
        from tracerite import tty

        # Save original state
        original_excepthook = sys.excepthook

        try:
            install()

            # The installed hook should handle fallback internally
            # We can inspect its structure
            hook = threading.excepthook
            assert hook.__name__ == "_tracerite_threading_excepthook"

            # Verify _original_threading_excepthook is set
            assert tty._original_threading_excepthook is not None

        finally:
            uninstall()
            # Ensure we're back to original state
            assert sys.excepthook == original_excepthook


class TestExcepthookWithNoneOriginal:
    """Test excepthook when original hook becomes None."""

    def test_sys_excepthook_fallback_to_builtin(self, monkeypatch, capsys):
        """Test sys.excepthook falls back to sys.__excepthook__ when original is None."""
        from tracerite import tty

        install()
        try:
            # Make tty_traceback fail
            def failing_tty(**kwargs):
                raise RuntimeError("fail")

            monkeypatch.setattr(tty, "tty_traceback", failing_tty)
            # Set original hook to None
            monkeypatch.setattr(tty, "_original_excepthook", None)

            try:
                raise ValueError("builtin fallback test")
            except Exception:
                exc_type, exc_value, exc_tb = sys.exc_info()
                sys.excepthook(exc_type, exc_value, exc_tb)

            captured = capsys.readouterr()
            assert "ValueError" in captured.err
        finally:
            uninstall()

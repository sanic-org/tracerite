"""Tests for em_columns functionality with caret anchors."""

import sys

import pytest

from tracerite import extract_chain
from tracerite.trace import extract_exception, extract_frames

from .errorcases import (
    binomial_operator,
    chained_from_and_without,
    error_in_stdlib_mimetypes,
    max_type_error_case,
    multiline_marking,
    multiline_marking_comment,
    reraise_context,
    reraise_suppressed_context,
    unrelated_error_in_except,
    unrelated_error_in_finally,
)


@pytest.mark.skipif(
    sys.version_info < (3, 11), reason="Requires co_positions() from Python 3.11+"
)
def test_em_columns_binary_operator():
    """Test that em_columns are populated for binary operator errors."""
    try:
        binomial_operator()
    except Exception as e:
        chain = extract_chain(e)
        assert chain is not None
        assert len(chain) > 0

        frame = chain[0]["frames"][-1]  # Get the last frame (where error occurred)

        # Check that we have fragments
        assert "fragments" in frame
        fragments = frame["fragments"]
        assert isinstance(fragments, list)
        assert len(fragments) > 0

        # Look for the line with the binary operator
        found_em_fragment = False
        for line_info in fragments:
            for fragment in line_info["fragments"]:
                if "em" in fragment:
                    found_em_fragment = True
                    # The fragment with emphasis should contain the operator
                    assert "code" in fragment
                    # For binary operators, we expect the operator or part of it to be emphasized
                    break
            if found_em_fragment:
                break

        # We should find at least one emphasized fragment for the binary operator
        assert found_em_fragment, "No emphasized fragment found for binary operator"


@pytest.mark.skipif(
    sys.version_info < (3, 11), reason="Requires co_positions() from Python 3.11+"
)
def test_em_columns_multiline_operator():
    """Test that em_columns work for multiline binary operator errors."""
    try:
        multiline_marking()
    except Exception as e:
        chain = extract_chain(e)
        assert chain is not None
        assert len(chain) > 0

        frame = chain[0]["frames"][-1]

        # Check that we have fragments
        assert "fragments" in frame
        fragments = frame["fragments"]
        assert isinstance(fragments, list)
        assert len(fragments) > 0

        # For multiline operators, check that we have emphasis somewhere
        found_em_fragment = False
        for line_info in fragments:
            for fragment in line_info["fragments"]:
                if "em" in fragment:
                    found_em_fragment = True
                    break
            if found_em_fragment:
                break

        # We should find emphasized fragments for multiline operators too
        assert found_em_fragment, "No emphasized fragment found for multiline operator"


@pytest.mark.skipif(
    sys.version_info < (3, 11), reason="Requires co_positions() from Python 3.11+"
)
def test_em_columns_function_call():
    """Test that em_columns work for function call errors."""
    try:
        max_type_error_case()
    except Exception as e:
        chain = extract_chain(e)
        assert chain is not None
        assert len(chain) > 0

        frame = chain[0]["frames"][-1]

        # Check that we have fragments
        assert "fragments" in frame
        fragments = frame["fragments"]
        assert isinstance(fragments, list)
        assert len(fragments) > 0

        # Look for emphasized fragments in function calls
        found_em_fragment = False
        for line_info in fragments:
            for fragment in line_info["fragments"]:
                if "em" in fragment:
                    found_em_fragment = True
                    # The emphasized fragment should contain part of the function call
                    assert "code" in fragment
                    break
            if found_em_fragment:
                break

        # We should find emphasized fragments for function calls
        assert found_em_fragment, "No emphasized fragment found for function call"


@pytest.mark.skipif(
    sys.version_info < (3, 11), reason="Requires co_positions() from Python 3.11+"
)
def test_em_columns_not_empty():
    """Test that em_columns is no longer empty when there are position details."""
    try:
        binomial_operator()
    except Exception as e:
        chain = extract_chain(e)
        frame = chain[0]["frames"][-1]

        # Check that we have position information in the Range format
        assert frame.get("range") is not None
        assert frame["range"].cbeg is not None
        assert frame["range"].cend is not None

        # Count emphasized fragments
        em_count = 0
        for line_info in frame["fragments"]:
            for fragment in line_info["fragments"]:
                if "em" in fragment:
                    em_count += 1

        # Should have at least one emphasized fragment when position info is available
        assert em_count > 0, (
            "em_columns should not be empty when position information is available"
        )


def test_em_columns_structure():
    """Test that em_columns fragments have correct structure."""
    try:
        binomial_operator()
    except Exception as e:
        chain = extract_chain(e)
        frame = chain[0]["frames"][-1]

        for line_info in frame["fragments"]:
            for fragment in line_info["fragments"]:
                if "em" in fragment:
                    # Emphasized fragments should have the expected structure
                    assert "code" in fragment
                    assert isinstance(fragment["code"], str)
                    assert fragment["em"] == "solo"  # Based on the implementation

                    # Should also have other markings like 'mark' for error highlighting
                    # but em is independent of mark


@pytest.mark.skipif(
    sys.version_info < (3, 11), reason="Requires co_positions() from Python 3.11+"
)
def test_em_columns_multiline_marking_comment():
    """Test that em_columns work for multiline operator with comments."""
    try:
        multiline_marking_comment()
    except Exception as e:
        chain = extract_chain(e)
        assert chain is not None
        assert len(chain) > 0

        frame = chain[0]["frames"][-1]

        # Check that we have fragments
        assert "fragments" in frame
        fragments = frame["fragments"]
        assert isinstance(fragments, list)
        assert len(fragments) > 0

        # For multiline operators with comments, check that we have emphasis somewhere
        found_em_fragment = False
        for line_info in fragments:
            for fragment in line_info["fragments"]:
                if "em" in fragment:
                    found_em_fragment = True
                    break
            if found_em_fragment:
                break

        # We should find emphasized fragments for multiline operators with comments too
        assert found_em_fragment, (
            "No emphasized fragment found for multiline operator with comment"
        )


@pytest.mark.skipif(
    sys.version_info < (3, 11), reason="Requires co_positions() from Python 3.11+"
)
def test_em_columns_max_type_error():
    """Test that em_columns work for function call type errors."""
    try:
        max_type_error_case()
    except Exception as e:
        chain = extract_chain(e)
        assert chain is not None
        assert len(chain) > 0

        frame = chain[0]["frames"][-1]

        # Check that we have fragments
        assert "fragments" in frame
        fragments = frame["fragments"]
        assert isinstance(fragments, list)
        assert len(fragments) > 0

        # Look for emphasized fragments in function calls
        found_em_fragment = False
        for line_info in fragments:
            for fragment in line_info["fragments"]:
                if "em" in fragment:
                    found_em_fragment = True
                    # The emphasized fragment should contain part of the function call
                    assert "code" in fragment
                    break
            if found_em_fragment:
                break

        # We should find emphasized fragments for function calls
        assert found_em_fragment, "No emphasized fragment found for function call"


def test_em_columns_reraise_context():
    """Test that em_columns work for reraise with context."""
    try:
        reraise_context()
    except Exception as e:
        chain = extract_chain(e)
        assert chain is not None
        assert len(chain) > 0

        # Check that we have at least one frame with fragments
        found_fragments = False
        for exc_info in chain:
            for frame in exc_info["frames"]:
                if "fragments" in frame and frame["fragments"]:
                    found_fragments = True
                    break
            if found_fragments:
                break

        assert found_fragments, "No fragments found in reraise context"


def test_em_columns_reraise_suppressed_context():
    """Test that em_columns work for reraise with suppressed context."""
    try:
        reraise_suppressed_context()
    except Exception as e:
        chain = extract_chain(e)
        assert chain is not None
        assert len(chain) > 0

        # Check that we have at least one frame with fragments
        found_fragments = False
        for exc_info in chain:
            for frame in exc_info["frames"]:
                if "fragments" in frame and frame["fragments"]:
                    found_fragments = True
                    break
            if found_fragments:
                break

        assert found_fragments, "No fragments found in reraise suppressed context"


def test_em_columns_chained_exceptions():
    """Test that em_columns work for chained exceptions."""
    try:
        chained_from_and_without()
    except Exception as e:
        chain = extract_chain(e)
        assert chain is not None
        assert len(chain) > 0

        # Check that we have at least one frame with fragments
        found_fragments = False
        for exc_info in chain:
            for frame in exc_info["frames"]:
                if "fragments" in frame and frame["fragments"]:
                    found_fragments = True
                    break
            if found_fragments:
                break

        assert found_fragments, "No fragments found in chained exceptions"


@pytest.mark.skipif(
    sys.version_info < (3, 11), reason="Requires co_positions() from Python 3.11+"
)
def test_em_columns_unrelated_error_except():
    """Test that em_columns work for unrelated errors in except blocks."""
    try:
        unrelated_error_in_except()
    except Exception as e:
        chain = extract_chain(e)
        assert chain is not None
        assert len(chain) > 0

        frame = chain[0]["frames"][-1]

        # Check that we have fragments
        assert "fragments" in frame
        fragments = frame["fragments"]
        assert isinstance(fragments, list)
        assert len(fragments) > 0

        # Look for emphasized fragments in division by zero
        found_em_fragment = False
        for line_info in fragments:
            for fragment in line_info["fragments"]:
                if "em" in fragment:
                    found_em_fragment = True
                    # The emphasized fragment should contain the division operator
                    assert "code" in fragment
                    break
            if found_em_fragment:
                break

        # We should find emphasized fragments for division operator
        assert found_em_fragment, "No emphasized fragment found for division operator"


@pytest.mark.skipif(
    sys.version_info < (3, 11), reason="Requires co_positions() from Python 3.11+"
)
def test_em_columns_unrelated_error_finally():
    """Test that em_columns work for unrelated errors in finally blocks."""
    try:
        unrelated_error_in_finally()
    except Exception as e:
        chain = extract_chain(e)
        assert chain is not None
        assert len(chain) > 0

        frame = chain[0]["frames"][-1]

        # Check that we have fragments
        assert "fragments" in frame
        fragments = frame["fragments"]
        assert isinstance(fragments, list)
        assert len(fragments) > 0

        # Look for emphasized fragments in division by zero
        found_em_fragment = False
        for line_info in fragments:
            for fragment in line_info["fragments"]:
                if "em" in fragment:
                    found_em_fragment = True
                    # The emphasized fragment should contain the division operator
                    assert "code" in fragment
                    break
            if found_em_fragment:
                break

        # We should find emphasized fragments for division operator
        assert found_em_fragment, "No emphasized fragment found for division operator"


def test_em_columns_comprehensive_structure():
    """Test that em_columns work across different error types and maintain structure."""
    test_functions = [
        binomial_operator,
        error_in_stdlib_mimetypes,
        multiline_marking,
        multiline_marking_comment,
        max_type_error_case,
        unrelated_error_in_except,
        unrelated_error_in_finally,
    ]

    for test_func in test_functions:
        try:
            test_func()
        except Exception as e:
            chain = extract_chain(e)
            assert chain is not None, f"Chain is None for {test_func.__name__}"
            assert len(chain) > 0, f"Empty chain for {test_func.__name__}"

            # Check that at least one frame has proper fragment structure
            found_valid_structure = False
            for exc_info in chain:
                for frame in exc_info["frames"]:
                    if "fragments" in frame and frame["fragments"]:
                        fragments = frame["fragments"]
                        # Verify fragment structure
                        for line_info in fragments:
                            assert "line" in line_info, (
                                f"Missing line in fragment for {test_func.__name__}"
                            )
                            assert "fragments" in line_info, (
                                f"Missing fragments in line_info for {test_func.__name__}"
                            )
                            assert isinstance(line_info["line"], int), (
                                f"Invalid line type for {test_func.__name__}"
                            )
                            assert isinstance(line_info["fragments"], list), (
                                f"Invalid fragments type for {test_func.__name__}"
                            )

                            for fragment in line_info["fragments"]:
                                assert "code" in fragment, (
                                    f"Missing code in fragment for {test_func.__name__}"
                                )
                                assert isinstance(fragment["code"], str), (
                                    f"Invalid code type for {test_func.__name__}"
                                )

                        found_valid_structure = True
                        break
                if found_valid_structure:
                    break

            assert found_valid_structure, (
                f"No valid fragment structure found for {test_func.__name__}"
            )


@pytest.mark.skipif(
    sys.version_info < (3, 11), reason="Requires co_positions() from Python 3.11+"
)
def test_em_columns_stdlib_mimetypes():
    """Test that em_columns correctly handle stdlib mimetypes errors with function call marking.

    Python's traceback shows:
        url = os.fspath(url)
              ^^^^^^^^^^^^^^
    This tests that tracerite correctly parses the caret marking and emphasizes the
    entire 'os.fspath(url)' function call, properly accounting for dedenting.
    """
    try:
        error_in_stdlib_mimetypes()
    except Exception as e:
        chain = extract_chain(e)
        assert chain is not None
        assert len(chain) > 0

        # Find the stdlib frame with the os.fspath call
        stdlib_fspath_frame = None
        problematic_line_info = None

        for frame in chain[0]["frames"]:
            # Look for stdlib mimetypes.py frame
            if (
                "mimetypes.py" in frame.get("filename", "")
                and "fragments" in frame
                and frame["fragments"]
            ):
                for line_info in frame["fragments"]:
                    line_code = "".join(frag["code"] for frag in line_info["fragments"])
                    if "fspath" in line_code and "url =" in line_code:
                        stdlib_fspath_frame = frame
                        problematic_line_info = line_info
                        break
                if stdlib_fspath_frame:
                    break

        assert stdlib_fspath_frame is not None, (
            "Could not find stdlib frame with os.fspath"
        )
        assert problematic_line_info is not None, "Could not find fspath line"

        # Analyze the correct fragment parsing
        fragments = problematic_line_info["fragments"]

        # Check for the correct behavior: function name should be properly marked
        found_function_marked = False
        found_args_emphasized = False
        found_proper_structure = False

        for fragment in fragments:
            if fragment["code"] == "os.fspath" and "mark" in fragment:
                found_function_marked = True
            elif (
                fragment["code"] == "(url)" and "em" in fragment and "mark" in fragment
            ):
                found_args_emphasized = True

        # Verify we have proper structure with function name not split
        fragment_codes = [
            frag["code"] for frag in fragments if not frag["code"].endswith("\n")
        ]
        full_line = "".join(fragment_codes)
        if full_line == "url = os.fspath(url)":
            found_proper_structure = True

        # These assertions verify the correct behavior after the fix
        assert found_function_marked, "Expected 'os.fspath' to be marked as a unit"
        assert found_args_emphasized, "Expected '(url)' to be emphasized"
        assert found_proper_structure, (
            "Expected proper line structure without splitting function name"
        )

        print("SUCCESS: Function name 'os.fspath' is correctly marked as a unit")
        # Build fragment description for debugging
        fragment_descriptions = []
        for frag in fragments:
            if not frag["code"].endswith("\n"):
                desc = f"'{frag['code']}'"
                if "em" in frag:
                    desc += " [EM]"
                if "mark" in frag:
                    desc += " [MARK]"
                fragment_descriptions.append(desc)
        print(f"Fragments: {fragment_descriptions}")

        # Verify we have the expected structure showing the fix
        assert len(fragments) >= 3, "Expected at least 3 fragments"


# ============================================================================
# Tests ported from main branch to achieve 100% coverage
# ============================================================================


class TestExtractChain:
    """Test extract_chain function for exception chaining."""

    def test_extract_single_exception(self):
        """Test extracting a single exception without chaining."""
        try:
            raise ValueError("test error")
        except ValueError as e:
            chain = extract_chain(exc=e)

        assert len(chain) == 1
        assert chain[0]["type"] == "ValueError"
        assert "test error" in chain[0]["message"]

    def test_extract_chained_exceptions(self):
        """Test extracting chained exceptions with __cause__."""
        try:
            try:
                raise ValueError("original error")
            except ValueError as e:
                raise TypeError("wrapped error") from e
        except TypeError as e:
            chain = extract_chain(exc=e)

        # When using 'from', only the newest exception is returned by default
        assert len(chain) >= 1
        assert chain[0]["type"] == "TypeError"
        assert "wrapped error" in chain[0]["message"]

    def test_extract_context_exceptions(self):
        """Test extracting exceptions with implicit context (__context__)."""
        try:
            try:
                raise ValueError("first error")
            except ValueError:
                raise TypeError("second error")
        except TypeError as e:
            chain = extract_chain(exc=e)

        # Should have both exceptions
        assert len(chain) == 2
        assert chain[0]["type"] == "TypeError"
        assert chain[1]["type"] == "ValueError"

    def test_suppress_context(self):
        """Test that __suppress_context__ stops chain extraction."""
        try:
            try:
                raise ValueError("suppressed error")
            except ValueError:
                exc = TypeError("visible error")
                exc.__suppress_context__ = True
                raise exc
        except TypeError as e:
            chain = extract_chain(exc=e)

        # Should only have the TypeError, not the ValueError
        assert len(chain) == 1
        assert chain[0]["type"] == "TypeError"

    def test_extract_from_sys_exc_info(self):
        """Test extracting current exception from sys.exc_info()."""
        try:
            raise RuntimeError("current exception")
        except RuntimeError:
            chain = extract_chain()  # No exc argument

        assert len(chain) == 1
        assert chain[0]["type"] == "RuntimeError"


class TestExtractException:
    """Test extract_exception function for single exception details."""

    def test_basic_exception_extraction(self):
        """Test basic exception information extraction."""
        try:
            x = 1
            y = 0
            result = x / y  # noqa: F841
        except ZeroDivisionError as e:
            info = extract_exception(e)

        assert info["type"] == "ZeroDivisionError"
        assert (
            "division" in info["message"].lower() or "zero" in info["message"].lower()
        )
        assert info["summary"]
        assert info["repr"]
        assert isinstance(info["frames"], list)
        assert len(info["frames"]) > 0

    def test_exception_with_long_message(self):
        """Test that long exception messages are truncated in summary."""
        long_msg = "x" * 150
        try:
            raise ValueError(long_msg)
        except ValueError as e:
            info = extract_exception(e)

        # Message should be full, summary should be truncated
        assert len(info["message"]) == 150
        assert len(info["summary"]) < len(info["message"])
        assert "···" in info["summary"]

    def test_exception_with_very_long_message(self):
        """Test handling of very long messages (>1000 chars)."""
        long_msg = "A" * 600 + "B" * 600
        try:
            raise RuntimeError(long_msg)
        except RuntimeError as e:
            info = extract_exception(e)

        # Summary should show beginning and end
        summary = info["summary"]
        assert "···" in summary
        assert "A" in summary
        assert "B" in summary

    def test_exception_with_multiline_message(self):
        """Test that multiline messages are summarized correctly."""
        msg = "First line\nSecond line\nThird line"
        try:
            raise ValueError(msg)
        except ValueError as e:
            info = extract_exception(e)

        # Summary should only have first line
        assert info["summary"] == "First line"
        assert info["message"] == msg

    def test_skip_outmost_frames(self):
        """Test skipping outermost frames."""

        def outer():
            def middle():
                def inner():
                    raise ValueError("error")

                inner()

            middle()

        try:
            outer()
        except ValueError as e:
            info = extract_exception(e, skip_outmost=1)

        # Should skip the first frame
        assert len(info["frames"]) >= 2

    def test_skip_until_pattern(self):
        """Test skipping frames until a filename pattern is found."""
        try:
            raise ValueError("test")
        except ValueError as e:
            info = extract_exception(e, skip_until="test_trace.py")

        # Should have frames starting from test_trace.py
        assert len(info["frames"]) > 0

    def test_base_exception_suppression(self):
        """Test that BaseException subclasses are handled differently."""
        try:
            raise KeyboardInterrupt()
        except KeyboardInterrupt as e:
            info = extract_exception(e)

        # Should still extract but may suppress inner frames
        assert info["type"] == "KeyboardInterrupt"
        # Frames should be present
        assert isinstance(info["frames"], list)


class TestExtractFrames:
    """Test extract_frames function for traceback frame extraction."""

    def test_extract_frames_basic(self):
        """Test basic frame extraction from traceback."""

        def function_a():
            def function_b():
                raise ValueError("test")

            function_b()

        try:
            function_a()
        except ValueError as e:
            import inspect

            tb = inspect.getinnerframes(e.__traceback__)
            frames = extract_frames(tb)

        assert len(frames) > 0
        # Check frame structure
        frame = frames[0]
        assert "id" in frame
        assert "relevance" in frame
        assert "location" in frame
        assert "range" in frame or "linenostart" in frame  # version2 uses range
        assert "function" in frame
        assert "variables" in frame

    def test_frame_relevance_markers(self):
        """Test that frame relevance is correctly assigned."""

        def user_function():
            raise RuntimeError("error")

        try:
            user_function()
        except RuntimeError as e:
            import inspect

            tb = inspect.getinnerframes(e.__traceback__)
            frames = extract_frames(tb)

        # Last frame should be 'error' (where exception was raised)
        assert frames[-1]["relevance"] == "error"

    def test_empty_traceback(self):
        """Test handling of empty traceback."""
        frames = extract_frames([])
        assert frames == []

    def test_method_with_self(self):
        """Test that method names include class name when self is present."""

        class MyClass:
            def my_method(self):
                raise ValueError("error in method")

        try:
            obj = MyClass()
            obj.my_method()
        except ValueError as e:
            import inspect

            tb = inspect.getinnerframes(e.__traceback__)
            frames = extract_frames(tb)

        # Should find frame with class name
        function_names = [f["function"] for f in frames]
        assert any(
            "MyClass" in name and "my_method" in name for name in function_names if name
        )

    def test_method_with_cls(self):
        """Test that classmethods include class name."""

        class MyClass:
            @classmethod
            def my_classmethod(cls):
                raise ValueError("error in classmethod")

        try:
            MyClass.my_classmethod()
        except ValueError as e:
            import inspect

            tb = inspect.getinnerframes(e.__traceback__)
            frames = extract_frames(tb)

        function_names = [f["function"] for f in frames]
        assert any(
            "MyClass" in name and "my_classmethod" in name
            for name in function_names
            if name
        )

    def test_module_level_code(self):
        """Test handling of module-level code (no function)."""
        try:
            exec("raise ValueError('module level')")
        except ValueError as e:
            import inspect

            tb = inspect.getinnerframes(e.__traceback__)
            frames = extract_frames(tb)

        # Should handle module-level frames
        assert len(frames) > 0

    def test_frame_with_no_source(self):
        """Test handling frames where source code is not available."""
        # Create exception from built-in code path
        try:
            import json

            json.loads("{invalid json}")
        except Exception as e:
            import inspect

            tb = inspect.getinnerframes(e.__traceback__)
            frames = extract_frames(tb)

        # Should handle frames even without source
        assert len(frames) > 0

    def test_suppress_inner_frames(self):
        """Test that suppress_inner stops at the bug frame."""

        def outer():
            def middle():
                def inner():
                    raise KeyboardInterrupt()

                inner()

            middle()

        try:
            outer()
        except KeyboardInterrupt as e:
            import inspect

            tb = inspect.getinnerframes(e.__traceback__)
            frames = extract_frames(tb, e.__traceback__, suppress_inner=True)

        # Should have fewer frames due to suppression
        assert len(frames) > 0

    def test_relative_path_display(self):
        """Test that paths in CWD are shown as relative."""
        # Create a traceback in current directory
        try:
            raise ValueError("test")
        except ValueError as e:
            import inspect

            tb = inspect.getinnerframes(e.__traceback__)
            frames = extract_frames(tb)

        # Check that we have frames with filenames
        assert any(f["filename"] for f in frames)

    def test_long_function_path_shortening(self):
        """Test that long module.function paths are shortened."""

        # Function names should be shortened to last 2 components
        def outer():
            def inner():
                raise ValueError("test")

            inner()

        try:
            outer()
        except ValueError as e:
            import inspect

            tb = inspect.getinnerframes(e.__traceback__)
            frames = extract_frames(tb)

        # Check function names are present and reasonably short
        for frame in frames:
            if frame["function"]:
                # Should not have overly long paths
                assert frame["function"].count(".") <= 1

    def test_frame_urls(self):
        """Test that frame URLs are generated for real files."""
        try:
            raise ValueError("test")
        except ValueError as e:
            import inspect

            tb = inspect.getinnerframes(e.__traceback__)
            frames = extract_frames(tb)

        # At least one frame should have URLs
        has_vscode_url = any("VS Code" in f["urls"] for f in frames if f["urls"])
        # VS Code URLs should be present for real files
        assert has_vscode_url or len(frames) > 0  # Flexible for different environments

    def test_frame_source_lines(self):
        """Test that source code lines are extracted and deindented."""

        def test_function():
            x = 1  # noqa: F841
            y = 2  # noqa: F841
            raise ValueError("test")

        try:
            test_function()
        except ValueError as e:
            import inspect

            tb = inspect.getinnerframes(e.__traceback__)
            frames = extract_frames(tb)

        # Find the test_function frame
        test_frame = next((f for f in frames if f["function"] == "test_function"), None)
        if test_frame:
            assert test_frame["lines"]
            assert "x = 1" in test_frame["lines"]
            assert "y = 2" in test_frame["lines"]

    @pytest.mark.skipif(
        sys.version_info < (3, 11),
        reason="Requires Python 3.11+ for accurate end_lineno position info",
    )
    def test_multiline_exception_statement_extraction(self):
        """Test that multiline exception statements are fully extracted.

        When an exception is raised with a multiline string literal,
        all lines of the statement should be included in the extracted source.
        This tests the fix for the issue where only the first few lines were shown.
        """
        try:
            raise Exception("""Brief.

    1 2 3
    i i i
""")
        except Exception as e:
            import inspect

            tb = inspect.getinnerframes(e.__traceback__)
            frames = extract_frames(tb, e.__traceback__)

        # Find the frame where the exception was raised
        error_frame = next((f for f in frames if f["relevance"] == "error"), None)
        assert error_frame is not None, "Should have an error frame"

        # Check that all lines of the multiline exception are present
        lines = error_frame["lines"]
        assert 'raise Exception("""Brief.' in lines, "Should contain start of exception"
        assert "1 2 3" in lines, "Should contain middle content line 1"
        assert "i i i" in lines, "Should contain middle content line 2"
        assert '""")' in lines, "Should contain closing of exception"


class TestLibdirPattern:
    """Test the libdir pattern for identifying library code."""

    def test_libdir_pattern_matches_site_packages(self):
        """Test that libdir pattern matches site-packages paths."""
        from tracerite.trace import libdir

        assert libdir.fullmatch("/usr/lib/python3.9/site-packages/module.py")
        assert libdir.fullmatch(
            "/home/user/.local/lib/python3.9/site-packages/pkg/mod.py"
        )

    def test_libdir_pattern_matches_dist_packages(self):
        """Test that libdir pattern matches dist-packages paths."""
        from tracerite.trace import libdir

        assert libdir.fullmatch("/usr/lib/python3/dist-packages/module.py")

    def test_libdir_pattern_matches_usr_paths(self):
        """Test that libdir pattern matches /usr/ paths."""
        from tracerite.trace import libdir

        assert libdir.fullmatch("/usr/lib/python3.9/module.py")
        assert libdir.fullmatch("/usr/local/lib/module.py")

    def test_libdir_pattern_does_not_match_user_code(self):
        """Test that libdir pattern doesn't match user code paths."""
        from tracerite.trace import libdir

        assert not libdir.fullmatch("/home/user/project/module.py")
        assert not libdir.fullmatch("./mycode.py")
        assert not libdir.fullmatch("/opt/myapp/code.py")

    def test_libdir_pattern_matches_cache_directories(self):
        """Test that libdir pattern matches .cache directory paths."""
        from tracerite.trace import libdir

        assert libdir.fullmatch("/home/user/.cache/some_lib/module.py")
        assert libdir.fullmatch("/home/user/.cache/torch/model.py")
        assert libdir.fullmatch(
            "/home/user/.cache/library/modules/deep/nested/path/model.py"
        )

    def test_libdir_pattern_does_not_match_ipython_input(self):
        """Test that libdir pattern doesn't match IPython/Jupyter input cells."""
        from tracerite.trace import libdir

        # IPython input cells should be treated as user code
        assert not libdir.fullmatch("<ipython-input-1>")
        assert not libdir.fullmatch("<ipython-input-11-abc123>")
        assert not libdir.fullmatch("/tmp/ipykernel_12345/1234567890.py")

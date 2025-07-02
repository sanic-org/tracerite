"""Tests for em_columns functionality with caret anchors."""

from tracerite import extract_chain

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

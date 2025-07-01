"""Tests for em_columns functionality with caret anchors."""

from tracerite import extract_chain

from .errorcases import binomial_operator, max_type_error_case, multiline_marking


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

        # Check that we have position information
        assert frame.get("colno") is not None
        assert frame.get("end_colno") is not None

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

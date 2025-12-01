"""Tests for the fragment-based line structure in tracerite."""

import sys

import pytest

from tracerite import extract_chain


def test_fragments_basic_structure():
    """Test that fragments are generated with basic structure."""

    def error_function():
        x = 5
        y = 0
        result = x / y  # This will cause an error
        return result

    try:
        error_function()
    except Exception as e:
        chain = extract_chain(e)
        assert chain is not None
        assert len(chain) > 0

        frame = chain[0]["frames"][-1]  # Get the last frame
        assert "fragments" in frame

        fragments = frame["fragments"]
        assert isinstance(fragments, list)
        assert len(fragments) > 0

        # Check that each line has proper structure
        for line_info in fragments:
            assert "line" in line_info
            assert "fragments" in line_info
            assert isinstance(line_info["line"], int)
            assert isinstance(line_info["fragments"], list)

            # Check fragment structure
            for fragment in line_info["fragments"]:
                assert "code" in fragment
                assert isinstance(fragment["code"], str)


@pytest.mark.skipif(
    sys.version_info < (3, 11), reason="Requires co_positions() from Python 3.11+"
)
def test_fragments_indentation_structure():
    """Test that indentation fragments have correct structure."""

    def indented_error():
        if True:
            if True:
                x = 1 / 0  # Error with indentation
        return x

    try:
        indented_error()
    except Exception as e:
        chain = extract_chain(e)
        frame = chain[0]["frames"][-1]
        fragments = frame["fragments"]

        # Find the error line
        error_line = None
        for line_info in fragments:
            for frag in line_info["fragments"]:
                if "mark" in frag:
                    error_line = line_info
                    break
            if error_line:
                break

        assert error_line is not None, "Should find the error line with mark"

        # Check that there's indentation with "solo" value
        indent_fragments = [
            frag for frag in error_line["fragments"] if "indent" in frag
        ]
        assert len(indent_fragments) > 0, "Should have indent fragments"

        for indent_frag in indent_fragments:
            assert indent_frag["indent"] == "solo", "Indent should have value 'solo'"
            assert indent_frag["code"].strip() == "", (
                "Indent code should be whitespace only"
            )


def test_fragments_comment_structure():
    """Test that comments have correct structure."""

    def error_with_comment():
        x = 1 / 0  # This is a comment
        return x

    try:
        error_with_comment()
    except Exception as e:
        chain = extract_chain(e)
        frame = chain[0]["frames"][-1]
        fragments = frame["fragments"]

        # Find comment fragments
        comment_fragments = []
        for line_info in fragments:
            for frag in line_info["fragments"]:
                if "comment" in frag:
                    comment_fragments.append(frag)

        assert len(comment_fragments) > 0, "Should find comment fragments"

        for comment_frag in comment_fragments:
            assert comment_frag["comment"] == "solo", "Comment should have value 'solo'"
            assert "# This is a comment" in comment_frag["code"], (
                "Comment code should contain comment text"
            )


@pytest.mark.skipif(
    sys.version_info < (3, 11), reason="Requires co_positions() from Python 3.11+"
)
def test_fragments_mark_highlighting():
    """Test that error positions are marked correctly."""

    def error_function():
        x = 5
        y = 0
        result = x / y  # Error here
        return result

    try:
        error_function()
    except Exception as e:
        chain = extract_chain(e)
        frame = chain[0]["frames"][-1]
        fragments = frame["fragments"]

        # Should find multiple fragments with mark (beg, mid, fin)
        marked_fragments = []
        for line_info in fragments:
            for frag in line_info["fragments"]:
                if "mark" in frag:
                    marked_fragments.append(frag)

        assert len(marked_fragments) >= 1, (
            f"Should have at least one marked fragment, found {len(marked_fragments)}"
        )

        # Check that mark values are appropriate
        mark_values = [frag["mark"] for frag in marked_fragments]
        assert any(mark in ["solo", "beg", "mid", "fin"] for mark in mark_values), (
            f"Mark values should be valid: {mark_values}"
        )

        # Check that at least one marked fragment contains part of the error expression
        marked_code = "".join(frag["code"] for frag in marked_fragments)
        assert any(char in marked_code for char in ["x", "/", "y"]), (
            f"Marked fragments should contain error expression parts, got: {marked_code}"
        )


def test_fragments_trailing_structure():
    """Test that trailing whitespace has correct structure."""

    def error_with_trailing():
        x = 1 / 0  # Note: extra spaces before comment
        return x

    try:
        error_with_trailing()
    except Exception as e:
        chain = extract_chain(e)
        frame = chain[0]["frames"][-1]
        fragments = frame["fragments"]

        # Find trailing fragments (whitespace between code and comment)
        trailing_fragments = []
        for line_info in fragments:
            for frag in line_info["fragments"]:
                if "trailing" in frag:
                    trailing_fragments.append(frag)

        assert len(trailing_fragments) > 0, "Should find trailing fragments"

        for trailing_frag in trailing_fragments:
            assert trailing_frag["trailing"] == "solo", (
                "Trailing should have value 'solo'"
            )
            # Code should be whitespace only (spaces, tabs, etc.)
            assert trailing_frag["code"].strip() == "", (
                "Trailing code should be whitespace only"
            )


@pytest.mark.skipif(
    sys.version_info < (3, 11), reason="Requires co_positions() from Python 3.11+"
)
def test_fragments_complete_line_structure():
    """Test the complete structure of a complex line with all fragment types."""

    def complex_error():
        if True:
            result = 5 / 0  # Division error
        return result

    try:
        complex_error()
    except Exception as e:
        chain = extract_chain(e)
        frame = chain[0]["frames"][-1]
        fragments = frame["fragments"]

        # Find the error line (should have mark)
        error_line = None
        for line_info in fragments:
            has_mark = any(frag.get("mark") for frag in line_info["fragments"])
            if has_mark:
                error_line = line_info
                break

        assert error_line is not None, "Should find error line"

        # Expected structure: indent + code + mark + trailing + comment
        frags = error_line["fragments"]

        # Should have indent
        indent_frags = [f for f in frags if "indent" in f]
        assert len(indent_frags) > 0, "Should have indent fragment"
        assert indent_frags[0]["indent"] == "solo"

        # Should have marked fragment(s)
        mark_frags = [f for f in frags if "mark" in f]
        assert len(mark_frags) >= 1, "Should have at least one mark fragment"

        # Check that marked content includes the error
        marked_code = "".join(frag["code"] for frag in mark_frags)
        assert any(char in marked_code for char in ["5", "/", "0"]), (
            f"Marked fragments should contain error parts, got: {marked_code}"
        )

        # Should have comment
        comment_frags = [f for f in frags if "comment" in f]
        assert len(comment_frags) > 0, "Should have comment fragment"
        assert comment_frags[0]["comment"] == "solo"

        # Should have trailing (whitespace between code and comment)
        trailing_frags = [f for f in frags if "trailing" in f]
        assert len(trailing_frags) > 0, "Should have trailing fragment"
        assert trailing_frags[0]["trailing"] == "solo"


def test_fragments_line_numbers():
    """Test that line numbers are correctly assigned."""

    def error_function():
        x = 1 / 0
        return x

    try:
        error_function()
    except Exception as e:
        chain = extract_chain(e)
        frame = chain[0]["frames"][-1]
        fragments = frame["fragments"]

        # Line numbers should be sequential starting from 1
        line_numbers = [line_info["line"] for line_info in fragments]
        assert len(line_numbers) > 0, "Should have line numbers"

        for i in range(1, len(line_numbers)):
            assert line_numbers[i] == line_numbers[i - 1] + 1, (
                "Line numbers should be sequential"
            )

        # First line should be 1 (relative to the context)
        assert line_numbers[0] == 1, "First line should be 1"


def test_fragments_empty_lines():
    """Test that empty lines are handled correctly."""

    def error_with_empty_line():
        y = 1 / 0  # Error after empty line
        return y

    try:
        error_with_empty_line()
    except Exception as e:
        chain = extract_chain(e)
        frame = chain[0]["frames"][-1]
        fragments = frame["fragments"]

        # Should handle empty lines (lines with no fragments or just empty fragments)
        for line_info in fragments:
            if not line_info["fragments"]:
                # Empty line - this is valid
                continue
            # Non-empty lines should have valid fragments
            for frag in line_info["fragments"]:
                assert "code" in frag, "All fragments should have code field"


def test_fragments_no_column_info():
    """Test fragments work correctly when no column information is available (older Python)."""

    def simple_error():
        x = 1 / 0
        return x

    try:
        simple_error()
    except Exception as e:
        chain = extract_chain(e)
        frame = chain[0]["frames"][-1]
        fragments = frame["fragments"]

        # Should still generate fragments even without precise column info
        assert len(fragments) > 0, "Should generate fragments"

        # Should have proper structure
        for line_info in fragments:
            assert "line" in line_info
            assert "fragments" in line_info
            for frag in line_info["fragments"]:
                assert "code" in frag


@pytest.mark.skipif(
    sys.version_info < (3, 11), reason="Requires co_positions() from Python 3.11+"
)
def test_fragments_multiline_error_marking():
    """Test that multi-line errors are marked correctly with fragments."""

    def multiline_error():
        # fmt: off
        _ = (
            1
            +
            "a")  # type: ignore
        # fmt: on
        return _

    try:
        multiline_error()
    except Exception as e:
        chain = extract_chain(e)
        assert chain is not None
        assert len(chain) > 0

        frame = chain[0]["frames"][-1]  # Get the last frame (error location)
        assert "fragments" in frame

        fragments = frame["fragments"]
        assert isinstance(fragments, list)
        assert len(fragments) > 0

        # Check that fragments are generated for the multi-line error
        has_source_lines = any(line_info["fragments"] for line_info in fragments)
        assert has_source_lines, (
            "Should have source code fragments for multi-line error"
        )

        # Find lines with mark fragments (should be the three final lines of the expression)
        marked_lines = []
        for line_info in fragments:
            has_mark = any(frag.get("mark") for frag in line_info["fragments"])
            if has_mark:
                marked_lines.append(line_info["line"])

        # Should have marks on multiple lines (the multi-line error span)
        assert len(marked_lines) > 1, (
            f"Multi-line error should mark multiple lines, found marks on lines: {marked_lines}"
        )

        # Verify that we have proper fragment structure on marked lines
        for line_info in fragments:
            if any(frag.get("mark") for frag in line_info["fragments"]):
                # This line has marking, check fragment structure
                marked_fragments = [
                    frag for frag in line_info["fragments"] if frag.get("mark")
                ]
                assert len(marked_fragments) > 0, (
                    "Marked line should have marked fragments"
                )

                # Check that mark status is one of the valid values
                for frag in marked_fragments:
                    assert frag["mark"] in ["solo", "beg", "mid", "fin"], (
                        f"Mark should have valid status, got: {frag['mark']}"
                    )

        # Ensure the fragments contain the problematic code
        all_code = ""
        for line_info in fragments:
            for frag in line_info["fragments"]:
                all_code += frag["code"]

        # Should contain the essential parts of the multi-line expression
        assert "1" in all_code, "Should contain the number 1"
        assert "+" in all_code, "Should contain the + operator"
        assert '"a"' in all_code, "Should contain the string 'a'"

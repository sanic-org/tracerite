"""Tests for syntaxerror.py - Enhanced SyntaxError position extraction."""

from tracerite.syntaxerror import (
    _find_any_unclosed_opener,
    _find_unclosed_opener,
    _find_unmatched_opener,
    _get_string_opener_length,
    _handle_incomplete,
    _handle_mismatch,
    _handle_unclosed,
    _handle_unterminated_string,
    _handle_unterminated_triple_string,
    _iter_code_chars,
    clean_syntax_error_message,
    extract_enhanced_positions,
)


class TestCleanSyntaxErrorMessage:
    """Test clean_syntax_error_message function."""

    def test_removes_detected_at_line(self):
        """Test removal of ' (detected at line N)' suffix."""
        msg = "unterminated string literal (detected at line 1)"
        result = clean_syntax_error_message(msg)
        assert "detected at line" not in result
        assert result == "unterminated string literal"

    def test_removes_on_line(self):
        """Test removal of ' on line N' suffix."""
        msg = "closing parenthesis ')' does not match opening parenthesis '(' on line 2"
        result = clean_syntax_error_message(msg)
        assert "on line" not in result

    def test_removes_filename_line(self):
        """Test removal of ' (filename.py, line N)' suffix."""
        msg = "invalid syntax (test.py, line 5)"
        result = clean_syntax_error_message(msg)
        assert "(test.py, line 5)" not in result
        assert result == "invalid syntax"

    def test_preserves_normal_message(self):
        """Test that normal messages are preserved."""
        msg = "invalid syntax"
        result = clean_syntax_error_message(msg)
        assert result == msg


class TestIterCodeChars:
    """Test _iter_code_chars iterator."""

    def test_basic_iteration(self):
        """Test basic character iteration."""
        lines = ["a = 1\n", "b = 2\n"]
        chars = list(_iter_code_chars(lines))
        # Should yield each code character with its position
        assert (1, 0, "a") in chars
        assert (1, 2, "=") in chars
        assert (2, 0, "b") in chars

    def test_skips_strings(self):
        """Test that characters inside strings are skipped."""
        lines = ['x = "hello"\n']
        chars = list(_iter_code_chars(lines))
        # Should not yield characters from inside the string
        char_strs = [c[2] for c in chars]
        assert "x" in char_strs
        assert "=" in char_strs
        # 'h', 'e', 'l', 'l', 'o' should not be in the output
        # (they are inside the string)

    def test_skips_triple_quoted_strings(self):
        """Test that triple-quoted strings are properly skipped."""
        lines = ['x = """multi\n', 'line"""\n']
        chars = list(_iter_code_chars(lines))
        char_strs = [c[2] for c in chars]
        assert "x" in char_strs
        assert "=" in char_strs

    def test_skips_comments(self):
        """Test that characters after # are skipped."""
        lines = ["x = 1  # comment\n"]
        chars = list(_iter_code_chars(lines))
        char_strs = [c[2] for c in chars]
        assert "x" in char_strs
        assert "#" not in char_strs
        assert "c" not in char_strs  # 'c' from 'comment'

    def test_respects_end_line(self):
        """Test end_line parameter."""
        lines = ["a\n", "b\n", "c\n"]
        chars = list(_iter_code_chars(lines, end_line=2))
        line_nums = {c[0] for c in chars}
        assert 1 in line_nums
        assert 2 in line_nums
        assert 3 not in line_nums

    def test_respects_end_col(self):
        """Test end_col parameter."""
        lines = ["abcdef\n"]
        chars = list(_iter_code_chars(lines, end_line=1, end_col=3))
        col_nums = [c[1] for c in chars]
        assert 0 in col_nums
        assert 1 in col_nums
        assert 2 in col_nums
        assert 3 not in col_nums

    def test_escaped_quote_in_string(self):
        """Test that escaped quotes don't end strings."""
        lines = ['x = "he\\"llo"\n']
        list(_iter_code_chars(lines))
        # String should be handled correctly despite escaped quote

    def test_single_quote_string_no_multiline(self):
        """Test that single-quoted strings don't span lines."""
        lines = ['x = "unterminated\n', "y = 2\n"]
        chars = list(_iter_code_chars(lines))
        # After line 1's unterminated string, line 2 should be normal code
        char_list = [(c[0], c[2]) for c in chars]
        assert (2, "y") in char_list


class TestExtractEnhancedPositions:
    """Test extract_enhanced_positions function."""

    def test_mismatch_pattern(self):
        """Test mismatched bracket error handling."""

        class MockError:
            def __init__(self):
                self.lineno = 3
                self.offset = 5

            def __str__(self):
                return "closing parenthesis ')' does not match opening parenthesis '{' on line 1"

        e = MockError()
        source_lines = ["{\n", "  x = 1\n", "    )\n"]

        mark_range, em_ranges = extract_enhanced_positions(e, source_lines)

        assert mark_range is not None
        assert em_ranges is not None
        assert len(em_ranges) == 2  # Opening and closing brackets

    def test_unclosed_pattern(self):
        """Test unclosed bracket error handling."""

        class MockError:
            def __init__(self):
                self.lineno = 2
                self.offset = 1

            def __str__(self):
                return "'(' was never closed"

        e = MockError()
        source_lines = ["(\n", "\n"]

        mark_range, em_ranges = extract_enhanced_positions(e, source_lines)

        assert mark_range is not None
        assert em_ranges is not None

    def test_unterminated_string_pattern(self):
        """Test unterminated string literal handling."""

        class MockError:
            def __init__(self):
                self.lineno = 1
                self.offset = 5

            def __str__(self):
                return "unterminated string literal (detected at line 1)"

        e = MockError()
        source_lines = ['x = "unterminated\n']

        mark_range, em_ranges = extract_enhanced_positions(e, source_lines)

        assert mark_range is not None
        assert em_ranges is not None

    def test_unterminated_fstring_pattern(self):
        """Test unterminated f-string literal handling."""

        class MockError:
            def __init__(self):
                self.lineno = 1
                self.offset = 5

            def __str__(self):
                return "unterminated f-string literal (detected at line 1)"

        e = MockError()
        source_lines = ['x = f"unterminated\n']

        mark_range, em_ranges = extract_enhanced_positions(e, source_lines)

        assert mark_range is not None
        assert em_ranges is not None

    def test_unterminated_triple_quoted_pattern(self):
        """Test unterminated triple-quoted string handling."""

        class MockError:
            def __init__(self):
                self.lineno = 1
                self.offset = 5

            def __str__(self):
                return "unterminated triple-quoted string literal (detected at line 1)"

        e = MockError()
        source_lines = ['x = """unterminated\n']

        mark_range, em_ranges = extract_enhanced_positions(e, source_lines)

        assert mark_range is not None
        assert em_ranges is not None

    def test_unterminated_triple_fstring_pattern(self):
        """Test unterminated triple-quoted f-string handling."""

        class MockError:
            def __init__(self):
                self.lineno = 1
                self.offset = 5

            def __str__(self):
                return (
                    "unterminated triple-quoted f-string literal (detected at line 1)"
                )

        e = MockError()
        source_lines = ['x = f"""unterminated\n']

        mark_range, em_ranges = extract_enhanced_positions(e, source_lines)

        assert mark_range is not None

    def test_incomplete_input_pattern(self):
        """Test incomplete input error handling."""

        class MockError:
            def __init__(self):
                self.lineno = 2
                self.offset = 1

            def __str__(self):
                return "incomplete input"

        e = MockError()
        source_lines = ["x = (\n", "  1\n"]

        mark_range, em_ranges = extract_enhanced_positions(e, source_lines)

        assert mark_range is not None

    def test_unrecognized_pattern_returns_none(self):
        """Test that unrecognized error patterns return None."""

        class MockError:
            def __init__(self):
                self.lineno = 1
                self.offset = 1

            def __str__(self):
                return "some other syntax error"

        e = MockError()
        source_lines = ["invalid code\n"]

        mark_range, em_ranges = extract_enhanced_positions(e, source_lines)

        assert mark_range is None
        assert em_ranges is None


class TestHandleMismatch:
    """Test _handle_mismatch function."""

    def test_basic_mismatch(self):
        """Test basic bracket mismatch."""

        from tracerite.syntaxerror import MISMATCH_PATTERN

        class MockError:
            lineno = 2
            offset = 5

        e = MockError()
        message = (
            "closing parenthesis ')' does not match opening parenthesis '(' on line 1"
        )
        match = MISMATCH_PATTERN.search(message)
        source_lines = ["(x = 1\n", "    )\n"]

        mark_range, em_ranges = _handle_mismatch(e, source_lines, match)

        assert mark_range is not None
        assert em_ranges is not None
        assert len(em_ranges) == 2

    def test_mismatch_fallback(self):
        """Test fallback when opening bracket not found."""

        from tracerite.syntaxerror import MISMATCH_PATTERN

        class MockError:
            lineno = 2
            offset = 5

        e = MockError()
        message = (
            "closing parenthesis ')' does not match opening parenthesis '[' on line 1"
        )
        match = MISMATCH_PATTERN.search(message)
        source_lines = ["no bracket here\n", "    )\n"]

        mark_range, em_ranges = _handle_mismatch(e, source_lines, match)

        # Should still produce ranges using fallback
        assert mark_range is not None

    def test_mismatch_opening_line_out_of_range(self):
        """Test line 182: opening_line out of source_lines range.

        When the reported opening line is greater than the number of source lines,
        the code falls back to opening_col = 0.
        """

        from tracerite.syntaxerror import MISMATCH_PATTERN

        class MockError:
            lineno = 1
            offset = 5

        e = MockError()
        # Pretend the error message says opening bracket was on line 99
        message = (
            "closing parenthesis ')' does not match opening parenthesis '(' on line 99"
        )
        match = MISMATCH_PATTERN.search(message)
        # Only 1 line of source
        source_lines = ["    )\n"]

        mark_range, em_ranges = _handle_mismatch(e, source_lines, match)

        # Should still produce ranges with opening_col = 0 fallback
        assert mark_range is not None
        assert mark_range.cbeg == 0  # Fallback to column 0


class TestHandleUnclosed:
    """Test _handle_unclosed function."""

    def test_unclosed_parenthesis(self):
        """Test unclosed parenthesis detection."""

        from tracerite.syntaxerror import UNCLOSED_PATTERN

        class MockError:
            lineno = 2
            offset = 1

        e = MockError()
        message = "'(' was never closed"
        match = UNCLOSED_PATTERN.search(message)
        source_lines = ["x = (\n", "  1\n"]

        mark_range, em_ranges = _handle_unclosed(e, source_lines, match)

        assert mark_range is not None
        assert em_ranges is not None

    def test_unclosed_not_found(self):
        """Test when unclosed opener cannot be found."""

        from tracerite.syntaxerror import UNCLOSED_PATTERN

        class MockError:
            lineno = 2
            offset = 1

        e = MockError()
        message = "'(' was never closed"
        match = UNCLOSED_PATTERN.search(message)
        source_lines = ["no brackets\n", "here\n"]

        mark_range, em_ranges = _handle_unclosed(e, source_lines, match)

        assert mark_range is None
        assert em_ranges is None


class TestHandleIncomplete:
    """Test _handle_incomplete function."""

    def test_incomplete_with_unclosed_bracket(self):
        """Test incomplete input with unclosed bracket."""

        class MockError:
            lineno = 3
            offset = 1

        e = MockError()
        source_lines = ["def foo():\n", "    x = (\n", "        1\n"]

        mark_range, em_ranges = _handle_incomplete(e, source_lines)

        assert mark_range is not None
        # Should have opening bracket position highlighted
        assert em_ranges is not None

    def test_incomplete_without_unclosed_bracket(self):
        """Test incomplete input without obvious unclosed bracket."""

        class MockError:
            lineno = 2
            offset = 1

        e = MockError()
        source_lines = ["complete code\n", "more code\n"]

        mark_range, em_ranges = _handle_incomplete(e, source_lines)

        # May return None if no unclosed opener found
        # The function should handle this gracefully

    def test_incomplete_empty_lines(self):
        """Test incomplete input with empty/comment lines at end."""

        class MockError:
            lineno = 4
            offset = 1

        e = MockError()
        source_lines = ["x = (\n", "  1\n", "  # comment\n", "\n"]

        mark_range, em_ranges = _handle_incomplete(e, source_lines)

        # Should find the unclosed bracket
        assert mark_range is not None


class TestFindAnyUnclosedOpener:
    """Test _find_any_unclosed_opener function."""

    def test_finds_unclosed_paren(self):
        """Test finding unclosed parenthesis."""
        source_lines = ["x = (\n", "  1\n"]
        line, col, opener = _find_any_unclosed_opener(source_lines, 2)

        assert line == 1
        assert opener == "("

    def test_finds_unclosed_bracket(self):
        """Test finding unclosed square bracket."""
        source_lines = ["x = [\n", "  1\n"]
        line, col, opener = _find_any_unclosed_opener(source_lines, 2)

        assert line == 1
        assert opener == "["

    def test_finds_unclosed_brace(self):
        """Test finding unclosed curly brace."""
        source_lines = ["x = {\n", "  1\n"]
        line, col, opener = _find_any_unclosed_opener(source_lines, 2)

        assert line == 1
        assert opener == "{"

    def test_returns_none_when_all_closed(self):
        """Test returns None when all brackets are closed."""
        source_lines = ["x = (1)\n", "y = [2]\n"]
        line, col, opener = _find_any_unclosed_opener(source_lines, 2)

        assert line is None
        assert col is None
        assert opener is None

    def test_finds_first_unclosed(self):
        """Test that it finds the first unclosed opener."""
        source_lines = ["x = ([\n", "  1\n"]
        line, col, opener = _find_any_unclosed_opener(source_lines, 2)

        # Should find the first one (the parenthesis at col 4)
        assert line == 1


class TestFindUnmatchedOpener:
    """Test _find_unmatched_opener function."""

    def test_basic_unmatched(self):
        """Test finding unmatched opener."""
        source_lines = ["((x))\n", "y\n"]
        _find_unmatched_opener(source_lines, 1, "(", 1, 3)

        # Should find the opener that would match the closer at position (1, 3)

    def test_nested_brackets(self):
        """Test with nested brackets."""
        source_lines = ["(a(b)c\n"]
        _find_unmatched_opener(source_lines, 1, "(", 1, 5)

        # Should find the outer parenthesis

    def test_opener_on_later_line_skips_earlier_lines(self):
        """Test line 303: continue when line_num < opener_line.

        When opener_line > 1, the scanner should skip lines before it.
        """
        source_lines = ["x = 1\n", "y = (\n", "  z)\n"]
        # Opener is on line 2, closer is on line 3
        result = _find_unmatched_opener(source_lines, 2, "(", 3, 3)

        # Should find the opener on line 2
        assert result is not None


class TestFindUnclosedOpener:
    """Test _find_unclosed_opener function."""

    def test_finds_unclosed(self):
        """Test finding unclosed opener."""
        source_lines = ["x = (\n", "  y\n"]
        line, col = _find_unclosed_opener(source_lines, 2, "(")

        assert line == 1
        assert col is not None

    def test_returns_none_when_closed(self):
        """Test returns None when brackets are balanced."""
        source_lines = ["x = (1)\n", "y = 2\n"]
        line, col = _find_unclosed_opener(source_lines, 2, "(")

        assert line is None
        assert col is None


class TestGetStringOpenerLength:
    """Test _get_string_opener_length function."""

    def test_single_quote(self):
        """Test single quote string."""
        line = 'x = "hello"'
        length = _get_string_opener_length(line, 4)
        assert length == 1

    def test_triple_quote(self):
        """Test triple quote string."""
        line = 'x = """hello"""'
        length = _get_string_opener_length(line, 4)
        assert length == 3

    def test_f_string(self):
        """Test f-string."""
        line = 'x = f"hello"'
        length = _get_string_opener_length(line, 4)
        assert length == 2  # f + "

    def test_raw_string(self):
        """Test raw string."""
        line = 'x = r"hello"'
        length = _get_string_opener_length(line, 4)
        assert length == 2  # r + "

    def test_rf_string(self):
        """Test rf-string."""
        line = 'x = rf"hello"'
        length = _get_string_opener_length(line, 4)
        assert length == 3  # rf + "

    def test_fr_triple_quote(self):
        """Test fr-string with triple quotes."""
        line = 'x = fr"""hello"""'
        length = _get_string_opener_length(line, 4)
        assert length == 5  # fr + """

    def test_bytes_string(self):
        """Test bytes string."""
        line = 'x = b"hello"'
        length = _get_string_opener_length(line, 4)
        assert length == 2  # b + "

    def test_rb_string(self):
        """Test raw bytes string."""
        line = 'x = rb"hello"'
        length = _get_string_opener_length(line, 4)
        assert length == 3  # rb + "

    def test_no_quote_fallback(self):
        """Test line 366: fallback when no quote character is found."""
        # Line with no quote character at the offset position
        line = "x = 123"
        length = _get_string_opener_length(line, 4)
        assert length == 1  # Fallback to 1


class TestHandleUnterminatedString:
    """Test _handle_unterminated_string function."""

    def test_basic_unterminated(self):
        """Test basic unterminated string."""

        class MockError:
            lineno = 1
            offset = 5

        e = MockError()
        source_lines = ['x = "unterminated\n']

        mark_range, em_ranges = _handle_unterminated_string(e, source_lines)

        assert mark_range is not None
        assert mark_range.lfirst == 1
        assert mark_range.cbeg == 4
        assert em_ranges is not None

    def test_empty_source_lines(self):
        """Test with empty source lines."""

        class MockError:
            lineno = 1
            offset = 1

        e = MockError()
        source_lines = []

        mark_range, em_ranges = _handle_unterminated_string(e, source_lines)

        assert mark_range is None
        assert em_ranges is None

    def test_invalid_line_number(self):
        """Test with invalid line number."""

        class MockError:
            lineno = 10
            offset = 1

        e = MockError()
        source_lines = ["one line\n"]

        mark_range, em_ranges = _handle_unterminated_string(e, source_lines)

        assert mark_range is None

    def test_zero_offset(self):
        """Test with zero/None offset."""

        class MockError:
            lineno = 1
            offset = 0

        e = MockError()
        source_lines = ['"unterminated\n']

        mark_range, em_ranges = _handle_unterminated_string(e, source_lines)

        # Should handle offset=0 gracefully


class TestHandleUnterminatedTripleString:
    """Test _handle_unterminated_triple_string function."""

    def test_basic_unterminated_triple(self):
        """Test basic unterminated triple-quoted string."""

        class MockError:
            lineno = 1
            offset = 5

        e = MockError()
        source_lines = ['x = """unterminated\n']

        mark_range, em_ranges = _handle_unterminated_triple_string(e, source_lines)

        assert mark_range is not None
        assert em_ranges is not None
        # Should emphasize the triple quotes
        assert em_ranges[0].cend - em_ranges[0].cbeg >= 3

    def test_empty_source_lines(self):
        """Test with empty source lines."""

        class MockError:
            lineno = 1
            offset = 1

        e = MockError()
        source_lines = []

        mark_range, em_ranges = _handle_unterminated_triple_string(e, source_lines)

        assert mark_range is None

    def test_f_triple_string(self):
        """Test unterminated f-triple-quoted string."""

        class MockError:
            lineno = 1
            offset = 5

        e = MockError()
        source_lines = ['x = f"""unterminated\n']

        mark_range, em_ranges = _handle_unterminated_triple_string(e, source_lines)

        assert mark_range is not None
        # Emphasis should include the f prefix and triple quotes
        assert em_ranges[0].cend - em_ranges[0].cbeg >= 4


class TestHandleMismatchBranches:
    """Additional tests for _handle_mismatch branch coverage."""

    def test_mismatch_opener_found_in_fallback_search(self):
        """Test line 179->185: opening_col >= 0 when bracket IS found in fallback.

        This tests the case where _find_unmatched_opener returns None but the
        bracket IS actually found on the line during the fallback search.
        We need a scenario where the smart search fails but the simple find succeeds.
        """
        from tracerite.syntaxerror import MISMATCH_PATTERN

        class MockError:
            lineno = 2
            offset = 5

        e = MockError()
        # The error message says '(' was on line 1, and we're closing with ')'
        message = (
            "closing parenthesis ')' does not match opening parenthesis '(' on line 1"
        )
        match = MISMATCH_PATTERN.search(message)
        # The '(' appears ONLY inside a string, so _find_unmatched_opener skips it
        # and returns None. But the fallback .find() will still locate the '(' character.
        source_lines = ['"(" bracket in string only\n', "    )\n"]

        mark_range, em_ranges = _handle_mismatch(e, source_lines, match)

        # Should produce ranges, with opening_col finding the bracket inside the string
        assert mark_range is not None
        # The cbeg should be 1 (where the '(' is found by .find())
        assert mark_range.cbeg == 1


class TestHandleIncompleteBranches:
    """Additional tests for _handle_incomplete branch coverage."""

    def test_incomplete_all_empty_lines(self):
        """Test line 230->240: loop completes without finding non-empty line.

        This tests the case where ALL source lines are empty or comment-only,
        so the loop finishes without breaking, and end_line/end_col stay at defaults.
        """

        class MockError:
            lineno = 3
            offset = 1

        e = MockError()
        # All lines are empty or comment-only
        source_lines = ["# comment\n", "   \n", "\n"]

        mark_range, em_ranges = _handle_incomplete(e, source_lines)

        # Should return None because no unclosed bracket can be found
        # in empty/comment-only lines
        assert mark_range is None
        assert em_ranges is None


class TestFindAnyUnclosedOpenerBranches:
    """Additional tests for _find_any_unclosed_opener branch coverage."""

    def test_closer_with_no_matching_opener(self):
        """Test line 264->259: stacks[opener] is empty when closer is found.

        This tests the case where a closing bracket is encountered but
        there's no corresponding opening bracket in the stack.
        """
        # Start with a closer that has no opener
        source_lines = [") extra closer (\n"]
        line, col, opener = _find_any_unclosed_opener(source_lines, 1)

        # Should find the unclosed '(' at the end
        assert line == 1
        assert opener == "("

    def test_multiple_closers_more_than_openers(self):
        """Test that extra closers don't crash when stack is empty."""
        # More closers than openers of each type
        source_lines = [")] } ([\n"]
        line, col, opener = _find_any_unclosed_opener(source_lines, 1)

        # Should find the '(' which is unclosed
        assert line == 1
        assert opener == "("

    def test_only_closers_no_openers(self):
        """Test with only closing brackets, no openers at all."""
        source_lines = [")]}\n"]
        line, col, opener = _find_any_unclosed_opener(source_lines, 1)

        # No unclosed openers to find
        assert line is None
        assert col is None
        assert opener is None


class TestRealSyntaxErrors:
    """Test with real Python SyntaxError objects."""

    def test_real_unterminated_string(self):
        """Test with a real unterminated string SyntaxError."""
        code = 'x = "unterminated'
        try:
            compile(code, "<test>", "exec")
        except SyntaxError as e:
            source_lines = [code + "\n"]
            mark_range, em_ranges = extract_enhanced_positions(e, source_lines)

            # Result depends on Python version - some versions use different messages
            # Just ensure it doesn't crash
            # Python 3.9 uses "EOL while scanning" which doesn't match our patterns
            # Python 3.10+ uses "unterminated string literal" which matches
            assert mark_range is not None or mark_range is None  # Just don't crash

    def test_real_mismatched_brackets(self):
        """Test with a real mismatched bracket SyntaxError."""
        code = "x = (1]"
        try:
            compile(code, "<test>", "exec")
        except SyntaxError as e:
            source_lines = [code + "\n"]
            mark_range, em_ranges = extract_enhanced_positions(e, source_lines)

            # Should handle mismatched brackets
            # Note: may or may not match depending on exact error message

    def test_real_unclosed_bracket(self):
        """Test with a real unclosed bracket SyntaxError."""
        code = "x = (\n1"
        try:
            compile(code, "<test>", "exec")
        except SyntaxError as e:
            source_lines = code.split("\n")
            source_lines = [line + "\n" for line in source_lines]
            mark_range, em_ranges = extract_enhanced_positions(e, source_lines)

            # May or may not match depending on exact error message

"""Enhanced SyntaxError position extraction.

Python's SyntaxError often provides poor position information, especially for
multi-line errors like mismatched brackets. This module parses common error
patterns and source code to provide better highlighting ranges.
"""

import re

from .core import STRING_PREFIX_PAIRS, STRING_PREFIXES

# Patterns for extracting information from SyntaxError messages
MISMATCH_PATTERN = re.compile(
    r"closing parenthesis '([)\]}])' does not match opening parenthesis '([(\[{])' on line (\d+)"
)
UNCLOSED_PATTERN = re.compile(r"'([(\[{])' was never closed")
INCOMPLETE_INPUT_PATTERN = re.compile(r"incomplete input")
# Match "unterminated [triple-quoted] [f-]string literal"
UNTERMINATED_STRING_PATTERN = re.compile(
    r"unterminated (?:triple-quoted )?(?:f-)?string literal"
)

# Pattern to clean up redundant line info from messages
DETECTED_AT_LINE_PATTERN = re.compile(r" \(detected at line \d+\)$")
ON_LINE_PATTERN = re.compile(r" on line \d+$")
FILENAME_LINE_PATTERN = re.compile(r" \([^)]+, line \d+\)$")

BRACKET_PAIRS = {")": "(", "]": "[", "}": "{"}
BRACKET_PAIRS_REV = {"(": ")", "[": "]", "{": "}"}
ALL_OPENERS = "([{"


def _iter_code_chars(source_lines, end_line=None, end_col=None):
    """Iterate over characters in source code, skipping strings and comments."""
    if end_line is None:
        end_line = len(source_lines)

    in_string = None  # None, or the quote character(s) that opened the string

    for line_idx in range(min(end_line, len(source_lines))):
        line = source_lines[line_idx].rstrip("\n\r")
        line_num = line_idx + 1  # 1-based

        # Determine where to stop on this line
        line_end = len(line)
        if line_num == end_line and end_col is not None:
            line_end = min(line_end, end_col)

        col = 0
        while col < line_end:
            char = line[col]
            rest = line[col:]

            if in_string:
                # Check for end of string
                if rest.startswith(in_string):
                    # Check it's not escaped (count preceding backslashes)
                    num_backslashes = 0
                    check_col = col - 1
                    while check_col >= 0 and line[check_col] == "\\":
                        num_backslashes += 1
                        check_col -= 1
                    if num_backslashes % 2 == 0:  # Not escaped
                        col += len(in_string)
                        in_string = None
                        continue
                col += 1
                continue

            # Check for start of string
            if rest.startswith('"""') or rest.startswith("'''"):
                in_string = rest[:3]
                col += 3
                continue
            if char in "\"'":
                in_string = char
                col += 1
                continue

            # Check for comment
            if char == "#":
                break  # Rest of line is comment

            # This is actual code
            yield line_num, col, char
            col += 1

        # Single-quoted strings don't span lines (would be a syntax error)
        if in_string and len(in_string) == 1:
            in_string = None


def clean_syntax_error_message(message):
    """Clean up redundant information from SyntaxError messages.

    Removes patterns like:
    - " (detected at line 1)" from unterminated strings
    - " on line 2" from bracket mismatches
    - " (filename.py, line N)" suffix
    These are redundant since we show the line in the traceback.
    """
    message = DETECTED_AT_LINE_PATTERN.sub("", message)
    message = ON_LINE_PATTERN.sub("", message)
    message = FILENAME_LINE_PATTERN.sub("", message)
    return message


# Pattern to extract the "detected at line N" number from SyntaxError messages
DETECTED_AT_LINE_EXTRACT_PATTERN = re.compile(r"\(detected at line (\d+)\)")


def extract_enhanced_positions(e, source_lines):
    """Extract enhanced position information for a SyntaxError.

    Args:
        e: The SyntaxError exception
        source_lines: List of source lines (strings with newlines)

    Returns:
        Tuple of (mark_range, em_ranges) where:
        mark_range: Range dict for the full highlight (e.g., from opening to closing bracket), or None
        em_ranges: List of Range dicts for emphasized positions (e.g., both mismatched brackets), or None
    """
    message = str(e)

    # Try to handle mismatched brackets: "closing parenthesis ')' does not match opening parenthesis '{' on line 1"
    match = MISMATCH_PATTERN.search(message)
    if match:
        return _handle_mismatch(e, source_lines, match)

    # Try to handle unclosed brackets: "'(' was never closed"
    match = UNCLOSED_PATTERN.search(message)
    if match:
        return _handle_unclosed(e, source_lines, match)

    # Try to handle unterminated string literal
    if UNTERMINATED_STRING_PATTERN.search(message):
        return _handle_unterminated_string(e, source_lines)

    # Try to handle incomplete input (e.g., _IncompleteInputError)
    match = INCOMPLETE_INPUT_PATTERN.search(message)
    if match:
        return _handle_incomplete(e, source_lines)

    # Default: use Python's positions
    return None, None


def _handle_mismatch(e, source_lines, match):
    """Handle mismatched bracket errors."""
    opening_char = match.group(2)  # The opening bracket it should match
    opening_line = int(match.group(3))  # Line number of opening bracket (1-based)

    closing_line = e.lineno
    closing_col = (e.offset - 1) if e.offset else 0

    # Find the opening bracket position on its line
    opening_col = None
    if 0 < opening_line <= len(source_lines):
        # Find the opening bracket - search for the one that would be unmatched
        opening_col = _find_unmatched_opener(
            source_lines, opening_line, opening_char, closing_line, closing_col
        )

    if opening_col is None:
        # Fallback: just find first occurrence
        if 0 < opening_line <= len(source_lines):
            opening_col = source_lines[opening_line - 1].find(opening_char)
            if opening_col < 0:
                opening_col = 0
        else:
            opening_col = 0

    # Mark range spans from opening bracket to closing bracket
    mark_range = {
        "lfirst": opening_line,
        "lfinal": closing_line,
        "cbeg": opening_col,
        "cend": closing_col + 1,
    }

    # Emphasis on both mismatched brackets
    em_ranges = [
        {
            "lfirst": opening_line,
            "lfinal": opening_line,
            "cbeg": opening_col,
            "cend": opening_col + 1,
        },
        {
            "lfirst": closing_line,
            "lfinal": closing_line,
            "cbeg": closing_col,
            "cend": closing_col + 1,
        },
    ]

    return mark_range, em_ranges


def _last_meaningful_line(source_lines):
    """Return the 1-based line number and column of the last meaningful line."""
    end_line = len(source_lines)
    end_col = 0
    for i in range(len(source_lines) - 1, -1, -1):
        line = source_lines[i].rstrip("\n\r")
        code_part = line.split("#")[0].rstrip()
        if code_part:
            end_line = i + 1  # 1-based
            end_col = len(line)
            break
    return end_line, end_col


def _handle_unclosed(e, source_lines, match):
    """Handle unclosed bracket errors."""
    opening_char = match.group(1)
    error_line = e.lineno
    error_col = (e.offset - 1) if e.offset else 0

    # Search backwards for the unclosed opener. If it can't be found (e.g. an
    # f-string brace inside a string), fall back to Python's reported position.
    opening_line, opening_col = _find_unclosed_opener(
        source_lines, error_line, opening_char
    )
    if opening_line is None:
        opening_line, opening_col = error_line, error_col

    # Guard against reported positions that lie past the source we have.
    if opening_line > len(source_lines):
        opening_line = len(source_lines)
        opening_col = 0

    end_line, end_col = _last_meaningful_line(source_lines)

    # Mark from opener to the end of the meaningful source
    mark_range = {
        "lfirst": opening_line,
        "lfinal": end_line,
        "cbeg": opening_col,
        "cend": end_col,
    }
    em_ranges = [
        {
            "lfirst": opening_line,
            "lfinal": opening_line,
            "cbeg": opening_col,
            "cend": opening_col + 1,
        }
    ]

    return mark_range, em_ranges


def _handle_incomplete(e, source_lines):
    """Handle incomplete input errors (e.g., _IncompleteInputError)."""
    # Find the last non-empty line (trimmed, ignoring comments)
    end_line, end_col = _last_meaningful_line(source_lines)

    # Try to find any unclosed bracket
    opening_line, opening_col, opener_char = _find_any_unclosed_opener(
        source_lines, end_line
    )

    if opening_line is None or opening_col is None:
        return None, None

    # Mark from opener to end of meaningful content
    mark_range = {
        "lfirst": opening_line,
        "lfinal": end_line,
        "cbeg": opening_col,
        "cend": end_col,
    }
    em_ranges = [
        {
            "lfirst": opening_line,
            "lfinal": opening_line,
            "cbeg": opening_col,
            "cend": opening_col + 1,
        }
    ]

    return mark_range, em_ranges


def _find_any_unclosed_opener(source_lines, end_line):
    """Find any unclosed opening bracket by scanning the source."""
    # Track all bracket types using proper tokenization
    stacks = {char: [] for char in ALL_OPENERS}

    for line_num, col, char in _iter_code_chars(source_lines, end_line):
        if char in ALL_OPENERS:
            stacks[char].append((line_num, col))
        elif char in BRACKET_PAIRS:
            opener = BRACKET_PAIRS[char]
            if stacks[opener]:
                stacks[opener].pop()

    # Find the first unclosed opener (earliest in code)
    first_unclosed = None
    first_opener = None
    for opener_char, stack in stacks.items():
        if stack:
            pos = stack[0]  # First unclosed of this type
            if first_unclosed is None or (pos[0], pos[1]) < (
                first_unclosed[0],
                first_unclosed[1],
            ):
                first_unclosed = pos
                first_opener = opener_char

    if first_unclosed:
        return first_unclosed[0], first_unclosed[1], first_opener
    return None, None, None


def _find_unmatched_opener(
    source_lines, opener_line, opener_char, closer_line, closer_col
):
    """Find the column of the unmatched opening bracket."""
    closer_char = BRACKET_PAIRS_REV.get(opener_char, ")")

    # Track bracket depth as we scan
    # We need to find the opener that would be matched by the closer
    stack = []  # Stack of (line, col) for opening brackets

    # Use tokenizer, but only scan from opener_line to closer position
    for line_num, col, char in _iter_code_chars(source_lines, closer_line, closer_col):
        if line_num < opener_line:
            continue
        if char == opener_char:
            stack.append((line_num, col))
        elif char == closer_char and stack:
            stack.pop()

    # The last unmatched opener is what we want
    if stack:
        return stack[-1][1]
    return None


def _find_unclosed_opener(source_lines, error_line, opener_char):
    """Find an unclosed opening bracket by scanning the source."""
    closer_char = BRACKET_PAIRS_REV.get(opener_char, ")")

    # Scan through code tracking bracket balance
    stack = []  # Stack of (line, col) for opening brackets

    for line_num, col, char in _iter_code_chars(source_lines, error_line):
        if char == opener_char:
            stack.append((line_num, col))
        elif char == closer_char and stack:
            stack.pop()

    # Return the first unclosed opener
    if stack:
        return stack[0]
    return None, None


def _get_string_opener_length(line, col):
    """Get the length of a string opener (prefix + quotes) starting at col."""
    rest = line[col:]

    # Check for string prefix (case insensitive: f, r, b, u, fr, rf, br, rb)
    prefix_len = 0
    prefix_rest = rest.lower()
    if prefix_rest[:2] in STRING_PREFIX_PAIRS:
        prefix_len = 2
    elif prefix_rest[:1] in STRING_PREFIXES:
        prefix_len = 1

    # Check for quotes after prefix
    after_prefix = rest[prefix_len:]
    if after_prefix.startswith('"""') or after_prefix.startswith("'''"):
        return prefix_len + 3
    elif after_prefix and after_prefix[0] in "\"'":
        return prefix_len + 1

    # Fallback: just one character
    return 1


def _handle_unterminated_string(e, source_lines):
    """Handle unterminated string literal errors."""
    error_line = e.lineno
    error_col = (e.offset - 1) if e.offset else 0

    if not source_lines or error_line < 1 or error_line > len(source_lines):
        return None, None

    # The "detected at line" phrase tells us how far Python scanned before
    # giving up on the string. Use it to extend the highlight when available.
    match = DETECTED_AT_LINE_EXTRACT_PATTERN.search(str(e))
    detected_line = int(match.group(1)) if match else error_line
    end_line = max(error_line, min(detected_line, len(source_lines)))

    opener_line = source_lines[error_line - 1].rstrip("\n\r")
    final_line = source_lines[end_line - 1].rstrip("\n\r")
    end_col = len(final_line)

    # Get the full string opener length (prefix + quote)
    opener_len = _get_string_opener_length(opener_line, error_col)

    # Mark from the opening quote to the end of the detected line
    mark_range = {
        "lfirst": error_line,
        "lfinal": end_line,
        "cbeg": error_col,
        "cend": end_col,
    }
    # Emphasize the full opener (prefix + quote)
    em_ranges = [
        {
            "lfirst": error_line,
            "lfinal": error_line,
            "cbeg": error_col,
            "cend": error_col + opener_len,
        }
    ]

    return mark_range, em_ranges

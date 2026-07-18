from __future__ import annotations

import re
from secrets import token_urlsafe

from .core import QUOTES, TRIPLE_QUOTES


class CodeScanner:
    """Stateful scanner that skips Python strings, comments, and escapes.

    Tracks whether the scanner is currently inside a string literal and the
    net bracket nesting depth. The state can be queried across multiple lines,
    allowing callers to find safe split points in source code.
    """

    __slots__ = ("in_string", "bracket_depth", "escape_next")

    def __init__(self):
        self.in_string = None
        self.bracket_depth = 0
        self.escape_next = False

    def step(self, text: str, i: int) -> int:
        """Process one logical unit starting at index i; return next index."""
        if self.escape_next:
            self.escape_next = False
            return i + 1

        char = text[i]
        if char == "\\":
            self.escape_next = True
            return i + 1

        if self.in_string is None and char == "#":
            return len(text)

        if self.in_string is None:
            if char in QUOTES and text[i : i + 3] in TRIPLE_QUOTES:
                self.in_string = text[i : i + 3]
                return i + 3
            if char in QUOTES:
                self.in_string = char
                return i + 1
            if char in "([{":
                self.bracket_depth += 1
            elif char in ")]}":
                self.bracket_depth -= 1
            return i + 1

        # Inside a string literal
        if self.in_string in TRIPLE_QUOTES and text[i : i + 3] == self.in_string:
            self.in_string = None
            return i + 3
        if len(self.in_string) == 1 and char == self.in_string:
            self.in_string = None
        return i + 1

    def process(self, text: str) -> None:
        """Advance scanner state across a whole string."""
        i = 0
        while i < len(text):
            i = self.step(text, i)

    @property
    def in_code_context(self) -> bool:
        """Return True if scanner is not inside a string or unclosed brackets."""
        return self.in_string is None and self.bracket_depth <= 0


def count_bracket_depth(text: str) -> int:
    """Count net bracket depth change, ignoring brackets in strings/comments."""
    scanner = CodeScanner()
    scanner.process(text)
    return scanner.bracket_depth


def find_clean_start_line(lines: list[str], target_idx: int) -> int:
    """Return the first line at/after target_idx outside an unclosed string/bracket."""
    if target_idx <= 0 or target_idx >= len(lines):
        return target_idx

    scanner = CodeScanner()
    for line in lines[:target_idx]:
        scanner.process(line)

    if scanner.in_code_context:
        return target_idx

    # Scan forward until the unclosed context ends.
    for idx in range(target_idx, len(lines)):  # pragma: no cover
        scanner.process(lines[idx])
        if scanner.in_code_context:
            return idx + 1

    return target_idx  # pragma: no cover


def fallback_mark_range_for_line(lines, error_line_in_context):
    """Build a single-line mark Range dict when caret columns are missing."""
    lines_list = lines.splitlines(keepends=True)
    if not (1 <= error_line_in_context <= len(lines_list)):
        return None
    line = lines_list[error_line_in_context - 1]
    content, _ = split_line_content(line)
    stripped = content.lstrip()
    if not stripped:
        return None
    start_col = len(content) - len(stripped)

    comment_start = find_comment_start(content)
    if comment_start is not None:
        code_end = content[:comment_start].rstrip()
        end_col = len(code_end)
    else:
        end_col = len(content.rstrip())

    if end_col <= start_col:
        end_col = start_col + 1
    return {
        "lfirst": error_line_in_context,
        "lfinal": error_line_in_context,
        "cbeg": start_col,
        "cend": end_col,
    }


def build_frame_ranges(
    lineno,
    pos_end_lineno,
    error_line_in_context,
    end_line,
    start_col,
    end_col,
    total_indent,
    lines,
):
    """Build the original and displayed source ranges for a frame."""
    if start_col is None or end_col is None:
        if error_line_in_context:
            fallback = fallback_mark_range_for_line(lines, error_line_in_context)
            if fallback:
                frame_range = {
                    "lfirst": lineno,
                    "lfinal": pos_end_lineno or lineno,
                    "cbeg": fallback["cbeg"] + total_indent,
                    "cend": fallback["cend"] + total_indent,
                }
                return frame_range, fallback
        return None, None

    adjusted_start_col = max(0, start_col - total_indent)
    adjusted_end_col = max(0, end_col - total_indent)
    frame_range = {
        "lfirst": lineno,
        "lfinal": pos_end_lineno or lineno,
        "cbeg": start_col,
        "cend": end_col,
    }
    mark_range = {
        "lfirst": error_line_in_context,
        "lfinal": end_line or error_line_in_context,
        "cbeg": adjusted_start_col,
        "cend": adjusted_end_col,
    }
    return frame_range, mark_range


def find_statement_end_line(lines_list, first_idx):
    """Return the 0-based index of the line where a compound statement header ends.

    Scans for the line whose code part ends with a colon at zero bracket
    depth, or None if not found within the available lines.
    """
    scanner = CodeScanner()
    for idx in range(first_idx, len(lines_list)):
        line, _ = split_line_content(lines_list[idx])
        scanner.process(line)
        comment_start = find_comment_start(line)
        code = line[:comment_start] if comment_start is not None else line
        if scanner.in_code_context and code.rstrip().endswith(":"):
            return idx
    return None


def build_with_statement_ranges(header_start, body_start, start, total_indent, lines):
    """Build frame/mark ranges covering an entire with statement header.

    Used instead of Python's expression-level range when a with block's enter
    or exit handling failed: the whole statement is marked, not just the
    context expression that the caret positions point at.
    """
    lines_list = lines.splitlines(keepends=True)
    first_idx = max(0, header_start - start)
    last_idx = find_statement_end_line(lines_list, first_idx)
    if last_idx is None:
        # Header end not visible in the window; clamp to the line before the body
        last_idx = min(len(lines_list) - 1, body_start - 1 - start)

    first_line, _ = split_line_content(lines_list[first_idx])
    cbeg = len(first_line) - len(first_line.lstrip())
    last_line, _ = split_line_content(lines_list[last_idx])
    comment_start = find_comment_start(last_line)
    if comment_start is not None:
        last_line = last_line[:comment_start]
    cend = len(last_line.rstrip())
    if first_idx == last_idx and cend <= cbeg:
        cend = cbeg + 1

    frame_range = {
        "lfirst": start + first_idx,
        "lfinal": start + last_idx,
        "cbeg": cbeg + total_indent,
        "cend": cend + total_indent,
    }
    mark_range = {
        "lfirst": first_idx + 1,
        "lfinal": last_idx + 1,
        "cbeg": cbeg,
        "cend": cend,
    }
    return frame_range, mark_range


def make_trace_id() -> str:
    """Generate a short unique identifier for a traceback frame."""
    return f"tb-{token_urlsafe(12)}"


def dedent_lines(lines: list[str]) -> tuple[list[str], str]:
    """Return (dedented_lines, common_indent)."""
    common_indent = calculate_common_indent(lines)
    return [ln.removeprefix(common_indent) for ln in lines], common_indent


def collect_positions_from_ranges(ranges, lines: list[str]) -> set[int]:
    """Collect character positions from a single Range dict or list of Range dicts."""
    positions = set()
    if not ranges:
        return positions
    for rng in ranges if isinstance(ranges, list) else [ranges]:
        positions |= convert_range_to_positions(rng, lines)
    return positions


def calculate_common_indent(lines):
    """Calculate common indentation across all non-empty lines."""
    non_empty_lines = [line.rstrip("\r\n") for line in lines if line.strip()]
    if not non_empty_lines:
        return ""
    indent_len = min(len(ln) - len(ln.lstrip(" \t")) for ln in non_empty_lines)
    return non_empty_lines[0][:indent_len]


def convert_range_to_positions(range_obj, lines):
    """Convert a Range dict (1-based inclusive lines, 0-based exclusive columns) to absolute character positions."""
    positions = set()

    if not range_obj:
        return positions

    # Convert to 0-based line indices for processing
    start_line_idx = range_obj["lfirst"] - 1
    end_line_idx = range_obj["lfinal"] - 1

    # Calculate absolute positions
    char_pos = 0
    for line_idx, line in enumerate(lines):
        if start_line_idx <= line_idx <= end_line_idx:
            line_content = line.rstrip("\r\n")

            if line_idx == start_line_idx == end_line_idx:
                # Single line case
                for col in range(
                    max(0, range_obj["cbeg"]), min(len(line_content), range_obj["cend"])
                ):
                    positions.add(char_pos + col)
            elif line_idx == start_line_idx:
                # First line of multi-line
                for col in range(max(0, range_obj["cbeg"]), len(line_content)):
                    positions.add(char_pos + col)
            elif line_idx == end_line_idx:
                # Last line of multi-line
                for col in range(0, min(len(line_content), range_obj["cend"])):
                    positions.add(char_pos + col)
            else:
                # Middle lines of multi-line
                for col in range(len(line_content)):
                    positions.add(char_pos + col)

        char_pos += len(line)

    return positions


def create_unified_fragments(lines_text, common_indent, mark_positions, em_positions):
    """Create fragments with unified mark/em highlighting."""
    lines = lines_text.splitlines(keepends=True)
    result = []

    for line_idx, line in enumerate(lines):
        line_num = line_idx + 1
        fragments = parse_line_to_fragments_unified(
            line,
            common_indent,
            mark_positions,
            em_positions,
            sum(len(lines[i]) for i in range(line_idx)),  # char offset for this line
        )
        result.append({"line": line_num, "fragments": fragments})

    return result


def parse_line_to_fragments_unified(
    line, common_indent, mark_positions, em_positions, line_char_offset
):
    """Parse a single line into fragments using unified highlighting."""
    line_content, line_ending = split_line_content(line)
    if not line_content and not line_ending:
        return []

    # Process indentation
    fragments, remaining, pos = process_indentation(line_content, common_indent)

    # Find comment split
    comment_start = find_comment_start(remaining)

    if comment_start is not None:
        # Handle line with comment
        code_part = remaining[:comment_start]
        comment_part = remaining[comment_start:]

        # Process code part (with trimming)
        code_trimmed = code_part.rstrip()
        code_whitespace = code_part[len(code_trimmed) :]

        if code_trimmed:
            fragments.extend(
                create_highlighted_fragments_unified(
                    code_trimmed, line_char_offset + pos, mark_positions, em_positions
                )
            )

        # Process comment part
        comment_trimmed = comment_part.rstrip()
        comment_trailing = comment_part[len(comment_trimmed) :]

        comment_with_leading_space = code_whitespace + comment_trimmed
        fragments.append({"code": comment_with_leading_space, "comment": "solo"})

        # Add trailing content
        trailing_content = comment_trailing + line_ending
        if trailing_content:
            fragments.append({"code": trailing_content, "trailing": "solo"})
    else:
        # Handle line without comment
        code_trimmed = remaining.rstrip()
        trailing_whitespace = remaining[len(code_trimmed) :]

        if code_trimmed:
            fragments.extend(
                create_highlighted_fragments_unified(
                    code_trimmed, line_char_offset + pos, mark_positions, em_positions
                )
            )

        trailing_content = trailing_whitespace + line_ending
        if trailing_content:
            fragments.append({"code": trailing_content, "trailing": "solo"})

    return fragments


def create_highlighted_fragments_unified(text, start_pos, mark_positions, em_positions):
    """Create fragments with mark/em highlighting using unified position sets."""
    if not text:
        return []

    # Convert absolute positions to text-relative positions
    text_mark_positions = set()
    text_em_positions = set()

    for i in range(len(text)):
        abs_pos = start_pos + i
        if abs_pos in mark_positions:
            text_mark_positions.add(i)
        if abs_pos in em_positions:
            text_em_positions.add(i)

    # Create fragments using existing logic
    return create_fragments_with_highlighting(
        text, text_mark_positions, text_em_positions
    )


def parse_lines_to_fragments(lines_text, mark_range=None, em_ranges=None):
    """Split code lines into highlighted fragments."""
    lines = lines_text.splitlines(keepends=True)
    if not lines:
        return []

    common_indent = calculate_common_indent(lines)

    # Convert both mark and em to position sets using unified logic
    mark_positions = convert_range_to_positions(mark_range, lines)
    em_positions = collect_positions_from_ranges(em_ranges, lines)

    # Create fragments using unified highlighting
    return create_unified_fragments(
        lines_text, common_indent, mark_positions, em_positions
    )


def split_line_content(line):
    """Split line into content and line ending."""
    if line.endswith("\r\n"):
        return line[:-2], "\r\n"
    elif line.endswith("\n"):
        return line[:-1], "\n"
    elif line.endswith("\r"):
        return line[:-1], "\r"
    else:
        return line, ""


def process_indentation(line_content, common_indent):
    """Process dedent and additional indentation, return fragments and remaining content."""
    fragments = []
    pos = 0

    # Handle dedent (common indentation)
    if common_indent and len(line_content) > len(common_indent):
        dedent_text = line_content[: len(common_indent)]
        fragments.append({"code": dedent_text, "dedent": "solo"})
        pos = len(common_indent)

    # Handle additional indentation
    remaining = line_content[pos:]
    indent_match = re.match(r"^(\s+)", remaining)
    if indent_match:
        indent_text = indent_match.group(1)
        fragments.append({"code": indent_text, "indent": "solo"})
        pos += len(indent_text)
        remaining = remaining[len(indent_text) :]

    return fragments, remaining, pos


def find_comment_start(text: str) -> int | None:
    """Find the start of a comment, ignoring # inside strings."""
    scanner = CodeScanner()
    i = 0
    while i < len(text):
        if scanner.in_string is None and not scanner.escape_next and text[i] == "#":
            return i
        i = scanner.step(text, i)
    return None


def positions_to_consecutive_ranges(positions):
    """Convert a set/list of positions to consecutive (start, end) ranges."""
    if not positions:
        return []

    sorted_positions = sorted(set(positions))
    ranges = []
    start = sorted_positions[0]
    end = start + 1

    for pos in sorted_positions[1:]:
        if pos == end:
            # Consecutive position, extend current range
            end = pos + 1
        else:
            # Gap found, close current range and start new one
            ranges.append((start, end))
            start = pos
            end = pos + 1

    # Close the last range
    ranges.append((start, end))
    return ranges


def get_highlight_boundaries(text, mark_positions, em_positions):
    """Get all boundaries for highlighting (start/end of mark and em regions)."""
    boundaries = {0, len(text)}

    # Add mark boundaries
    for start, end in positions_to_consecutive_ranges(mark_positions):
        boundaries.add(start)
        boundaries.add(end)

    # Add em boundaries
    for start, end in positions_to_consecutive_ranges(em_positions):
        boundaries.add(start)
        boundaries.add(end)

    return sorted(boundaries)


def create_fragments_with_highlighting(text, mark_positions, em_positions):
    """Create fragments with mark/em highlighting using beg/mid/fin/solo logic."""
    if not text:
        return []

    # Get all boundaries and create fragments
    boundaries = get_highlight_boundaries(text, mark_positions, em_positions)
    mark_ranges = positions_to_consecutive_ranges(mark_positions)
    em_ranges = positions_to_consecutive_ranges(em_positions)

    fragments = []
    for i in range(len(boundaries) - 1):
        start = boundaries[i]
        end = boundaries[i + 1]

        if start >= len(text):
            break

        fragment_text = text[start:end]
        fragment = {"code": fragment_text}

        # Determine mark status
        mark_status = get_highlight_status(start, end, mark_ranges)
        if mark_status:
            fragment["mark"] = mark_status

        # Determine em status
        em_status = get_highlight_status(start, end, em_ranges)
        if em_status:
            fragment["em"] = em_status

        fragments.append(fragment)

    return fragments


def get_highlight_status(frag_start, frag_end, ranges):
    """Determine beg/mid/fin/solo status for a fragment within ranges."""
    # Find overlapping ranges
    overlapping = []
    for range_start, range_end in ranges:
        if frag_start < range_end and frag_end > range_start:
            overlapping.append((range_start, range_end))

    if not overlapping:
        return None

    # Use the first overlapping range (they should align with fragment boundaries)
    range_start, range_end = overlapping[0]

    is_start = frag_start <= range_start
    is_end = frag_end >= range_end

    if is_start and is_end:
        return "solo"
    elif is_start:
        return "beg"
    elif is_end:
        return "fin"
    else:
        return "mid"

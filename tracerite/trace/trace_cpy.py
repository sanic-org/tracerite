# fmt: off
# ruff: noqa

# Copied from https://github.com/python/cpython/blob/main/Lib/traceback.py
# We need to use internal functions that are not part of the public API,
# and that are not available in earlier Python versions.
#
# This snapshot is based on upstream commit cffcee4d1b and has been updated with
# subsequent changes to the internal functions we keep.
#
# Unused functionality has been omitted and we add Py 3.10 compatibility
"""Extract, format and print information about Python stack traces."""

import collections
import itertools


def _walk_tb_with_full_positions(tb):
    # Internal version of walk_tb that yields full code positions including
    # end line and column information.
    while tb is not None:
        positions = _get_code_position(tb.tb_frame.f_code, tb.tb_lasti)
        # Yield tb_lineno when co_positions does not have a line number to
        # maintain behavior with walk_tb.
        if positions[0] is None:
            yield tb.tb_frame, (tb.tb_lineno, ) + positions[1:]
        else:
            yield tb.tb_frame, positions
        tb = tb.tb_next


def _get_code_position(code, instruction_index):
    if instruction_index < 0:
        return (None, None, None, None)
    # TRACERITE MODIFICATION: co_positions() was added in Python 3.11
    # Fallback for Python 3.10 compatibility
    if not hasattr(code, "co_positions"):
        return (None, None, None, None)
    positions_gen = code.co_positions()
    return next(itertools.islice(positions_gen, instruction_index // 2, None))


def _byte_offset_to_character_offset(str, offset):
    as_utf8 = str.encode('utf-8')
    return len(as_utf8[:offset].decode("utf-8", errors="replace"))


_Anchors = collections.namedtuple(
    "_Anchors",
    [
        "left_end_lineno",
        "left_end_offset",
        "right_start_lineno",
        "right_start_offset",
        "primary_char",
        "secondary_char",
    ],
    defaults=["~", "^"]
)


def _extract_caret_anchors_from_line_segment(segment):
    """
    Given source code `segment` corresponding to a FrameSummary, determine:
        - for binary ops, the location of the binary op
        - for indexing and function calls, the location of the brackets.
    `segment` is expected to be a valid Python expression.
    """
    import ast

    try:
        tree = ast.parse(f"(\n{segment}\n)")
    except SyntaxError:
        return None

    if len(tree.body) != 1:
        return None

    lines = segment.splitlines()

    def normalize(lineno, offset):
        """Get character index given byte offset"""
        return _byte_offset_to_character_offset(lines[lineno], offset)

    def next_valid_char(lineno, col):
        """Gets the next valid character index in `lines`, if
        the current location is not valid. Handles empty lines.
        """
        while lineno < len(lines) and col >= len(lines[lineno]):
            col = 0
            lineno += 1
        assert lineno < len(lines) and col < len(lines[lineno])
        return lineno, col

    def increment(lineno, col):
        """Get the next valid character index in `lines`."""
        col += 1
        lineno, col = next_valid_char(lineno, col)
        return lineno, col

    def nextline(lineno, col):
        """Get the next valid character at least on the next line"""
        col = 0
        lineno += 1
        lineno, col = next_valid_char(lineno, col)
        return lineno, col

    def increment_until(lineno, col, stop):
        """Get the next valid non-"\\#" character that satisfies the `stop` predicate"""
        while True:
            ch = lines[lineno][col]
            if ch in "\\#":
                lineno, col = nextline(lineno, col)
            elif not stop(ch):
                lineno, col = increment(lineno, col)
            else:
                break
        return lineno, col

    def setup_positions(expr, force_valid=True):
        """Get the lineno/col position of the end of `expr`. If `force_valid` is True,
        forces the position to be a valid character (e.g. if the position is beyond the
        end of the line, move to the next line)
        """
        # -2 since end_lineno is 1-indexed and because we added an extra
        # bracket + newline to `segment` when calling ast.parse
        lineno = expr.end_lineno - 2
        col = normalize(lineno, expr.end_col_offset)
        return next_valid_char(lineno, col) if force_valid else (lineno, col)

    statement = tree.body[0]
    match statement:
        case ast.Expr(expr):
            match expr:
                case ast.BinOp():
                    # ast gives these locations for BinOp subexpressions
                    # ( left_expr ) + ( right_expr )
                    #   left^^^^^       right^^^^^
                    lineno, col = setup_positions(expr.left)

                    # First operator character is the first non-space/')' character
                    lineno, col = increment_until(lineno, col, lambda x: not x.isspace() and x != ')')

                    # binary op is 1 or 2 characters long, on the same line,
                    # before the right subexpression
                    right_col = col + 1
                    if (
                        right_col < len(lines[lineno])
                        and (
                            # operator char should not be in the right subexpression
                            expr.right.lineno - 2 > lineno or
                            right_col < normalize(expr.right.lineno - 2, expr.right.col_offset)
                        )
                        and not (ch := lines[lineno][right_col]).isspace()
                        and ch not in "\\#"
                    ):
                        right_col += 1

                    # right_col can be invalid since it is exclusive
                    return _Anchors(lineno, col, lineno, right_col)
                case ast.Subscript():
                    # ast gives these locations for value and slice subexpressions
                    # ( value_expr ) [ slice_expr ]
                    #   value^^^^^     slice^^^^^
                    # subscript^^^^^^^^^^^^^^^^^^^^

                    # find left bracket
                    left_lineno, left_col = setup_positions(expr.value)
                    left_lineno, left_col = increment_until(left_lineno, left_col, lambda x: x == '[')
                    # find right bracket (final character of expression)
                    right_lineno, right_col = setup_positions(expr, force_valid=False)
                    return _Anchors(left_lineno, left_col, right_lineno, right_col)
                case ast.Call():
                    # ast gives these locations for function call expressions
                    # ( func_expr ) (args, kwargs)
                    #   func^^^^^
                    # call^^^^^^^^^^^^^^^^^^^^^^^^

                    # find left bracket
                    left_lineno, left_col = setup_positions(expr.func)
                    left_lineno, left_col = increment_until(left_lineno, left_col, lambda x: x == '(')
                    # find right bracket (final character of expression)
                    right_lineno, right_col = setup_positions(expr, force_valid=False)
                    return _Anchors(left_lineno, left_col, right_lineno, right_col)

    return None

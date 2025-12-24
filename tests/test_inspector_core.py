"""Core tests for inspector.py - variable extraction and pretty printing."""

import types

from tracerite.inspector import extract_variables, prettyvalue, safe_vars


class TestExtractVariables:
    """Test extract_variables function with various data types and edge cases."""

    def test_simple_variables(self):
        """Test extraction of simple variables used in source code."""
        variables = {"x": 42, "name": "Alice", "pi": 3.14}
        sourcecode = "result = x + pi"
        rows = extract_variables(variables, sourcecode)

        # Should extract x and pi (used in source), but not name
        names = {row[0] for row in rows}
        assert "x" in names
        assert "pi" in names
        assert "name" not in names

    def test_blacklisted_names(self):
        """Test that blacklisted names are excluded."""
        variables = {
            "_": "underscore",
            "In": "jupyter input",
            "Out": "jupyter output",
            "x": 5,
        }
        sourcecode = "x + _"
        rows = extract_variables(variables, sourcecode)

        names = {row[0] for row in rows}
        assert "_" not in names
        assert "In" not in names
        assert "Out" not in names
        assert "x" in names

    def test_blacklisted_types(self):
        """Test that blacklisted types (modules, functions) are excluded."""

        def sample_func():
            pass

        variables = {
            "func": sample_func,
            "module": types,
            "method": "".join,
            "x": 10,
        }
        sourcecode = "x + func()"
        rows = extract_variables(variables, sourcecode)

        names = {row[0] for row in rows}
        assert "x" in names
        assert "func" not in names
        assert "module" not in names
        assert "method" not in names

    def test_list_and_tuple(self):
        """Test extraction of list and tuple variables."""
        variables = {
            "short_list": [1, 2, 3],
            "long_list": list(range(20)),
            "tuple_data": (4, 5, 6),
            "empty_list": [],
        }
        sourcecode = "short_list + long_list + list(tuple_data) + empty_list"
        rows = extract_variables(variables, sourcecode)

        row_dict = {row[0]: row[2] for row in rows}

        # Short list should have elements listed
        assert "1, 2, 3" in row_dict["short_list"]

        # Long list should show count
        assert "20 items" in row_dict["long_list"]

        # Tuple should have elements
        assert "4, 5, 6" in row_dict["tuple_data"]

        # Empty list should show count
        assert "0 items" in row_dict["empty_list"]

    def test_type_as_value(self):
        """Test that type objects are excluded from extraction (they are blacklisted)."""
        variables = {"cls": int, "x": 5}
        sourcecode = "isinstance(x, cls)"
        rows = extract_variables(variables, sourcecode)

        # Types are now blacklisted, so cls should not be in output
        names = {row[0] for row in rows}
        assert "cls" not in names
        assert "x" in names

    def test_object_with_members(self):
        """Test extraction of object members when object has no str representation."""

        class SimpleClass:
            def __init__(self):
                self.value = 42
                self.name = "test"

        obj = SimpleClass()
        variables = {"obj": obj, "x": 1}
        sourcecode = "obj.value + obj.name + x"
        rows = extract_variables(variables, sourcecode)

        names = {row[0] for row in rows}
        # Should extract obj.value and obj.name since they're in sourcecode
        assert "obj.value" in names
        assert "obj.name" in names
        assert "x" in names

    def test_string_value_extraction(self):
        """Test that strings with different representations are handled."""
        variables = {
            "empty_str": "",
            "normal_str": "hello",
            "quoted_str": '"world"',
        }
        sourcecode = "empty_str + normal_str + quoted_str"
        rows = extract_variables(variables, sourcecode)

        row_dict = {row[0]: row[2] for row in rows}
        # Empty string behavior may vary
        assert "empty_str" in row_dict
        assert "hello" in row_dict["normal_str"]

    def test_long_value_truncation(self):
        """Test that long values are truncated with ellipsis."""
        variables = {"long_str": "a" * 200}
        sourcecode = "len(long_str)"
        rows = extract_variables(variables, sourcecode)

        row_dict = {row[0]: row[2] for row in rows}
        # Long string should be truncated
        assert "…" in row_dict["long_str"] or "..." in row_dict["long_str"]
        assert len(row_dict["long_str"]) <= 210  # Truncated with ellipsis marker

    def test_failed_str_or_repr(self):
        """Test variables that fail str() or repr() are skipped."""

        class BrokenClass:
            def __str__(self):
                raise ValueError("str failed")

            def __repr__(self):
                raise ValueError("repr failed")

        variables = {"broken": BrokenClass(), "x": 5}
        sourcecode = "str(broken) + str(x)"
        rows = extract_variables(variables, sourcecode)

        names = {row[0] for row in rows}
        # broken should be skipped, x should remain
        assert "broken" not in names
        assert "x" in names


class TestPrettyValue:
    """Test prettyvalue function with various data types."""

    def test_short_list(self):
        """Test pretty printing of short lists."""
        result, fmt = prettyvalue([1, 2, 3, 4])
        assert "1" in result and "4" in result

    def test_long_list(self):
        """Test that long lists show item count."""
        result, fmt = prettyvalue(list(range(100)))
        assert "100 items" in result

    def test_type_value(self):
        """Test pretty printing of type objects."""
        result, fmt = prettyvalue(str)
        assert "builtins.str" in result

    def test_float_value(self):
        """Test pretty printing of floats with scientific notation."""
        result, fmt = prettyvalue(3.14159)
        assert "3.1" in result

    def test_string_value(self):
        """Test that strings are returned as-is."""
        result, fmt = prettyvalue("hello world")
        assert result == "hello world"

    def test_long_repr_truncation(self):
        """Test that long repr values are truncated."""
        result, fmt = prettyvalue("x" * 200)
        assert "…" in result or "..." in result
        assert len(result) <= 210  # Truncated with ellipsis marker


class TestSafeVars:
    """Test safe_vars function for extracting object attributes."""

    def test_regular_object(self):
        """Test safe_vars on a regular object."""

        class Sample:
            def __init__(self):
                self.x = 1
                self.y = 2

        obj = Sample()
        result = safe_vars(obj)
        assert "x" in result
        assert "y" in result
        assert result["x"] == 1
        assert result["y"] == 2

    def test_object_with_slots(self):
        """Test safe_vars on objects with __slots__."""

        class SlottedClass:
            __slots__ = ("a", "b", "c")

            def __init__(self):
                self.a = 10
                self.b = 20
                # c is not set

        obj = SlottedClass()
        result = safe_vars(obj)
        assert "a" in result
        assert "b" in result
        assert result["a"] == 10
        assert result["b"] == 20
        # c should not raise an error even though not set

    def test_builtin_object(self):
        """Test safe_vars on builtin objects."""
        result = safe_vars(42)
        # Should return dict of attributes without errors
        assert isinstance(result, dict)


class TestArrayLikeHandling:
    """Test handling of numpy arrays (see test_inspector_numpy.py and test_inspector_torch.py for more)."""

    def test_prettyvalue_with_numpy_1d_array(self):
        """Test pretty printing of 1D numpy arrays."""
        import numpy as np

        arr = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        result, fmt = prettyvalue(arr)
        # Should format as comma-separated values
        assert "1" in result
        assert "5" in result

    def test_prettyvalue_with_numpy_long_1d_array(self):
        """Test pretty printing of long 1D numpy arrays."""
        import numpy as np

        arr = np.arange(200, dtype=np.float64)
        result, fmt = prettyvalue(arr)
        # Should show first and last few elements with ellipsis
        assert "…" in result

    def test_prettyvalue_with_numpy_2d_array(self):
        """Test pretty printing of small 2D numpy arrays."""
        import numpy as np

        arr = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
        result, fmt = prettyvalue(arr)
        # Should return nested list representation
        assert isinstance(result, list)
        assert len(result) == 2

    def test_object_member_with_poor_repr(self):
        """Test that object members with poor repr are skipped (lines 57-58)."""

        class InnerObjectWithBrokenStr:
            """An object where str() raises an exception."""

            def __str__(self):
                raise RuntimeError("str() failed")

        class OuterObject:
            """Container with a member that has broken str()."""

            def __init__(self):
                self.broken_member = InnerObjectWithBrokenStr()

        variables = {"obj": OuterObject()}
        sourcecode = "obj.broken_member"
        rows = extract_variables(variables, sourcecode)

        # The member with broken str() should be skipped
        member_names = [row[0] for row in rows]
        assert "obj.broken_member" not in member_names


class TestDictFormatting:
    """Test dict formatting in prettyvalue."""

    def test_empty_dict(self):
        """Test formatting of empty dict."""
        result, fmt = prettyvalue({})
        assert result == "{}"
        assert fmt == "inline"

    def test_small_dict_structured(self):
        """Test formatting of small dict returns structured data."""
        result, fmt = prettyvalue({"a": 1, "b": 2, "c": 3})
        # Now returns structured data for rendering
        assert isinstance(result, dict)
        assert result["type"] == "keyvalue"
        assert len(result["rows"]) == 3
        assert fmt == "inline"

    def test_dict_key_value_pairs(self):
        """Test that dict key-value pairs are correctly extracted."""
        result, fmt = prettyvalue({"name": "Alice", "age": 30})
        assert isinstance(result, dict)
        assert result["type"] == "keyvalue"
        rows = result["rows"]
        # Check that keys and values are present
        keys = [row[0] for row in rows]
        assert "name" in keys
        assert "age" in keys

    def test_large_dict(self):
        """Test formatting of large dict with > 10 items shows summary."""
        large_dict = {f"key_{i}": i for i in range(15)}
        result, fmt = prettyvalue(large_dict)
        # Large dicts show item count
        assert "15 items" in result
        assert fmt == "inline"


class TestMultilineFormatting:
    """Test formatting of values with newlines."""

    def test_inline_with_newlines(self):
        """Test inline format with newlines collapsed (line 171)."""

        # Create a non-string object with newlines in repr
        class MultilineRepr:
            def __repr__(self):
                return "Line1\nLine2\nLine3"

        result, fmt = prettyvalue(MultilineRepr())
        # Should collapse newlines for inline format
        assert "\n" not in result
        assert "Line1" in result and "Line3" in result
        assert fmt == "inline"

    def test_block_format_many_lines(self):
        """Test block format with > 20 lines (branch 176->177)."""
        # Create a string with more than 20 lines
        many_lines = "\n".join([f"Line {i}" for i in range(30)])
        result, fmt = prettyvalue(many_lines)
        # Should truncate to first 10 and last 10 lines with ellipsis
        assert "⋯" in result
        assert "Line 0" in result  # First line
        assert "Line 29" in result  # Last line
        assert (
            result.count("\n") == 20
        )  # 10 lines + ellipsis + 10 lines = 21 lines, 20 newlines
        assert fmt == "block"

    def test_short_multiline_string(self):
        """Test block format with <= 20 lines."""
        few_lines = "\n".join([f"Line {i}" for i in range(5)])
        result, fmt = prettyvalue(few_lines)
        # Should not truncate
        assert "⋯" not in result
        assert "Line 0" in result
        assert "Line 4" in result
        assert fmt == "block"

    def test_block_format_between_1_and_20_lines(self):
        """Test block format with 2-20 lines (branch 176->185)."""
        # This specifically tests the path where format_hint is "block"
        # but neither the > 20 lines condition nor the single-line >= 200 char condition is true
        medium_lines = "\n".join([f"Line number {i}" for i in range(15)])
        result, fmt = prettyvalue(medium_lines)
        # Should not truncate
        assert "⋯" not in result
        assert " … " not in result
        assert "Line number 0" in result
        assert "Line number 14" in result
        assert result.count("\n") == 14  # 15 lines = 14 newlines
        assert fmt == "block"

    def test_block_format_single_line_under_200_chars(self):
        """Test that long single-line strings are truncated inline."""
        # Single-line string > 120 chars gets truncated
        single_long = "x" * 150
        result, fmt = prettyvalue(single_long)
        # Long single-line strings are truncated with ellipsis
        assert "…" in result or " … " in result
        assert len(result) < 150
        assert fmt == "inline"

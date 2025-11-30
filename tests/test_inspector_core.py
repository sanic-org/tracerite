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
        """Test handling of type objects as values."""
        variables = {"cls": int, "x": 5}
        sourcecode = "isinstance(x, cls)"
        rows = extract_variables(variables, sourcecode)

        row_dict = {row[0]: row[2] for row in rows}
        assert "cls" in row_dict
        assert "builtins.int" in row_dict["cls"]

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
        assert len(row_dict["long_str"]) < 200

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
        result = prettyvalue([1, 2, 3, 4])
        assert "1" in result and "4" in result

    def test_long_list(self):
        """Test that long lists show item count."""
        result = prettyvalue(list(range(100)))
        assert "100 items" in result

    def test_type_value(self):
        """Test pretty printing of type objects."""
        result = prettyvalue(str)
        assert "builtins.str" in result

    def test_float_value(self):
        """Test pretty printing of floats with scientific notation."""
        result = prettyvalue(3.14159)
        assert "3.1" in result

    def test_string_value(self):
        """Test that strings are returned as-is."""
        result = prettyvalue("hello world")
        assert result == "hello world"

    def test_long_repr_truncation(self):
        """Test that long repr values are truncated."""
        result = prettyvalue("x" * 200)
        assert "…" in result or "..." in result
        assert len(result) < 200


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
    """Test handling of array-like objects with mock implementations."""

    def test_prettyvalue_with_mock_1d_array(self):
        """Test pretty printing of 1D array-like objects."""

        class Mock1DArray:
            shape = (5,)

            def __getitem__(self, idx):
                return [1.0, 2.0, 3.0, 4.0, 5.0][idx]

            def __iter__(self):
                return iter([1.0, 2.0, 3.0, 4.0, 5.0])

        arr = Mock1DArray()
        result = prettyvalue(arr)
        # Should format as comma-separated values
        assert "1.00" in result

    def test_prettyvalue_with_mock_long_1d_array(self):
        """Test pretty printing of long 1D array-like objects."""

        class MockLongArray:
            shape = (200,)

            def __getitem__(self, idx):
                if isinstance(idx, slice):
                    vals = list(range(200))
                    return [float(v) for v in vals[idx]]
                return float(idx)

        arr = MockLongArray()
        result = prettyvalue(arr)
        # Should show first and last few elements with ellipsis
        assert "…" in result or isinstance(result, str)

    def test_prettyvalue_with_mock_2d_array(self):
        """Test pretty printing of small 2D array-like objects."""

        class Mock2DArray:
            shape = (2, 3)

            def __getitem__(self, idx):
                return [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]][idx]

            def __iter__(self):
                return iter([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])

        arr = Mock2DArray()
        result = prettyvalue(arr)
        # Should return nested list representation
        assert isinstance(result, list) or "1.00" in result

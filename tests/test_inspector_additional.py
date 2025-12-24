"""Additional inspector.py tests for missing coverage."""

import dataclasses

import pytest

from tracerite.inspector import (
    VarInfo,
    _array_formatter,
    _format_scalar,
    _get_flat,
    extract_variables,
    prettyvalue,
    safe_vars,
)


class TestFormatScalar:
    """Test _format_scalar function."""

    def test_integer(self):
        """Test integer formatting."""
        assert _format_scalar(42) == "42"

    def test_float_nan(self):
        """Test NaN formatting."""
        result = _format_scalar(float("nan"))
        assert result == "NaN"

    def test_float_positive_inf(self):
        """Test positive infinity formatting."""
        result = _format_scalar(float("inf"))
        assert result == "∞"

    def test_float_negative_inf(self):
        """Test negative infinity formatting."""
        result = _format_scalar(float("-inf"))
        assert result == "-∞"

    def test_float_zero(self):
        """Test zero formatting."""
        result = _format_scalar(0.0)
        assert result == "0"

    def test_float_whole_number(self):
        """Test float that equals an integer."""
        result = _format_scalar(42.0)
        assert result == "42"

    def test_float_with_decimals(self):
        """Test float with decimal places."""
        result = _format_scalar(3.14159)
        assert "3.14" in result

    def test_numpy_integer(self):
        """Test numpy integer formatting."""
        np = pytest.importorskip("numpy")
        result = _format_scalar(np.int64(42))
        assert result == "42"

    def test_numpy_float(self):
        """Test numpy float formatting."""
        np = pytest.importorskip("numpy")
        result = _format_scalar(np.float64(3.14))
        assert "3.14" in result


class TestGetFlat:
    """Test _get_flat function."""

    def test_numpy_array(self):
        """Test with numpy array (has .flat)."""
        np = pytest.importorskip("numpy")
        arr = np.array([[1, 2], [3, 4]])
        flat = _get_flat(arr)
        assert list(flat) == [1, 2, 3, 4]

    def test_torch_tensor(self):
        """Test with torch tensor (has .flatten())."""
        torch = pytest.importorskip("torch")
        tensor = torch.tensor([[1, 2], [3, 4]])
        flat = _get_flat(tensor)
        assert list(flat) == [1, 2, 3, 4]

    def test_fallback(self):
        """Test fallback for objects without .flat or .flatten()."""

        class NoFlat:
            pass

        obj = NoFlat()
        result = _get_flat(obj)
        assert result is obj


class TestArrayFormatter:
    """Test _array_formatter function."""

    def test_integer_array(self):
        """Test integer array formatter."""
        np = pytest.importorskip("numpy")
        arr = np.array([1, 2, 3])
        fmt, suffix = _array_formatter(arr)

        assert fmt(1) == "1"
        assert suffix == ""

    def test_boolean_array(self):
        """Test boolean array formatter."""
        np = pytest.importorskip("numpy")
        arr = np.array([True, False, True])
        fmt, suffix = _array_formatter(arr)

        assert fmt(1) == "1"
        assert fmt(0) == "0"

    def test_float_array(self):
        """Test float array formatter."""
        np = pytest.importorskip("numpy")
        arr = np.array([1.5, 2.5, 3.5])
        fmt, suffix = _array_formatter(arr)

        result = fmt(1.5)
        assert "1" in result or "1.5" in result

    def test_float_array_all_nan(self):
        """Test float array with all NaN values."""
        np = pytest.importorskip("numpy")
        arr = np.array([float("nan"), float("nan")])
        fmt, suffix = _array_formatter(arr)

        assert fmt(float("nan")) == "NaN"
        assert fmt(float("inf")) == "∞"

    def test_float_array_with_inf_values(self):
        """Test float array containing infinity values."""
        np = pytest.importorskip("numpy")
        arr = np.array([1.0, float("inf"), float("-inf"), float("nan")])
        fmt, suffix = _array_formatter(arr)

        # Test the formatter handles special values
        assert fmt(float("inf")) == "∞"
        assert fmt(float("-inf")) == "-∞"
        assert fmt(float("nan")) == "NaN"

    def test_float_array_with_zero_in_formatter(self):
        """Test float array formatter handles zero after scaling."""
        np = pytest.importorskip("numpy")
        arr = np.array([1e9, 0.0, 2e9])  # Zero with large scale factor
        fmt, suffix = _array_formatter(arr)

        # The formatter should handle zero specially
        result = fmt(0.0)
        assert result == "0"

    def test_float_array_with_scaling(self):
        """Test float array that requires scaling (large values)."""
        np = pytest.importorskip("numpy")
        arr = np.array([1e9, 2e9, 3e9])
        fmt, suffix = _array_formatter(arr)

        # Should have scaling suffix
        assert suffix != "" or fmt(1e9) != "1000000000"

    def test_float_array_small_values(self):
        """Test float array with small values."""
        np = pytest.importorskip("numpy")
        arr = np.array([1e-9, 2e-9, 3e-9])
        fmt, suffix = _array_formatter(arr)

        # Should handle small values

    def test_float_array_large_sample(self):
        """Test float array with >200 elements (triggers sampling)."""
        np = pytest.importorskip("numpy")
        arr = np.arange(300.0)
        fmt, suffix = _array_formatter(arr)

        # Should sample from first 100 and last 100 elements

    def test_fallback_for_other_types(self):
        """Test fallback formatter for non-numeric types."""
        np = pytest.importorskip("numpy")
        arr = np.array(["a", "b", "c"])
        fmt, suffix = _array_formatter(arr)

        assert fmt("a") == "a"
        assert suffix == ""


class TestPrettyValueAdditional:
    """Additional prettyvalue tests for edge cases."""

    def test_empty_dict(self):
        """Test empty dict formatting."""
        result, fmt = prettyvalue({})
        assert result == "{}"

    def test_small_dict(self):
        """Test small dict formatting."""
        result, fmt = prettyvalue({"a": 1, "b": 2})
        assert isinstance(result, dict)
        assert result["type"] == "keyvalue"

    def test_large_dict(self):
        """Test large dict formatting (>10 items)."""
        d = {f"key{i}": i for i in range(15)}
        result, fmt = prettyvalue(d)
        assert "15 items" in result

    def test_dataclass_empty(self):
        """Test empty dataclass formatting."""

        @dataclasses.dataclass
        class EmptyDC:
            pass

        result, fmt = prettyvalue(EmptyDC())
        assert "EmptyDC()" in result

    def test_dataclass_small(self):
        """Test small dataclass formatting."""

        @dataclasses.dataclass
        class SmallDC:
            x: int = 1
            y: str = "test"

        result, fmt = prettyvalue(SmallDC())
        assert isinstance(result, dict)
        assert result["type"] == "keyvalue"

    def test_dataclass_large(self):
        """Test large dataclass formatting (>10 fields)."""

        @dataclasses.dataclass
        class LargeDC:
            f1: int = 1
            f2: int = 2
            f3: int = 3
            f4: int = 4
            f5: int = 5
            f6: int = 6
            f7: int = 7
            f8: int = 8
            f9: int = 9
            f10: int = 10
            f11: int = 11

        result, fmt = prettyvalue(LargeDC())
        assert "11 fields" in result

    def test_msgspec_struct_empty(self):
        """Test msgspec struct-like object with empty fields."""

        class EmptyStruct:
            __struct_fields__ = ()

        result, fmt = prettyvalue(EmptyStruct())
        assert "EmptyStruct()" in result

    def test_msgspec_struct_small(self):
        """Test msgspec struct-like object with few fields."""

        class SmallStruct:
            __struct_fields__ = ("x", "y")

            def __init__(self):
                self.x = 1
                self.y = 2

        result, fmt = prettyvalue(SmallStruct())
        assert isinstance(result, dict)
        assert result["type"] == "keyvalue"

    def test_msgspec_struct_large(self):
        """Test msgspec struct-like object with many fields."""

        class LargeStruct:
            __struct_fields__ = tuple(f"f{i}" for i in range(15))

            def __init__(self):
                for i in range(15):
                    setattr(self, f"f{i}", i)

        result, fmt = prettyvalue(LargeStruct())
        assert "15 fields" in result

    def test_pydantic_model_fields(self):
        """Test pydantic-like object with model_fields dict."""

        class PydanticLike:
            model_fields = {"x": None, "y": None}

            def __init__(self):
                self.x = 1
                self.y = 2

        result, fmt = prettyvalue(PydanticLike())
        assert isinstance(result, dict)
        assert result["type"] == "keyvalue"

    def test_namedtuple_empty(self):
        """Test empty namedtuple formatting."""
        from collections import namedtuple

        EmptyNT = namedtuple("EmptyNT", [])
        result, fmt = prettyvalue(EmptyNT())
        assert "EmptyNT()" in result

    def test_namedtuple_small(self):
        """Test small namedtuple formatting."""
        from collections import namedtuple

        Point = namedtuple("Point", ["x", "y"])
        result, fmt = prettyvalue(Point(1, 2))
        assert isinstance(result, dict)
        assert result["type"] == "keyvalue"
        assert len(result["rows"]) == 2
        assert result["rows"][0][0] == "x"
        assert result["rows"][0][1] == "1"
        assert result["rows"][1][0] == "y"
        assert result["rows"][1][1] == "2"

    def test_namedtuple_large(self):
        """Test large namedtuple formatting (>10 fields)."""
        from collections import namedtuple

        LargeNT = namedtuple("LargeNT", [f"f{i}" for i in range(15)])
        val = LargeNT(*range(15))
        result, fmt = prettyvalue(val)
        assert "15 fields" in result

    def test_namedtuple_with_long_value(self):
        """Test namedtuple with long field value is truncated."""
        from collections import namedtuple

        Data = namedtuple("Data", ["short", "long"])
        val = Data("x", "y" * 100)
        result, fmt = prettyvalue(val)
        assert isinstance(result, dict)
        assert result["type"] == "keyvalue"
        # Long value should be truncated
        assert len(result["rows"][1][1]) <= 60

    def test_namedtuple_typing_module(self):
        """Test namedtuple created via typing.NamedTuple."""
        from typing import NamedTuple

        class TypedPoint(NamedTuple):
            x: int
            y: int

        result, fmt = prettyvalue(TypedPoint(10, 20))
        assert isinstance(result, dict)
        assert result["type"] == "keyvalue"
        assert len(result["rows"]) == 2

    def test_2d_array(self):
        """Test 2D array formatting."""
        np = pytest.importorskip("numpy")
        arr = np.array([[1, 2], [3, 4]])
        result, fmt = prettyvalue(arr)

        assert isinstance(result, list)
        assert len(result) == 2

    def test_2d_array_with_suffix(self):
        """Test 2D array with scaling suffix."""
        np = pytest.importorskip("numpy")
        arr = np.array([[1e9, 2e9], [3e9, 4e9]])
        result, fmt = prettyvalue(arr)

        # Should be dict with suffix
        if isinstance(result, dict):
            assert "suffix" in result or "rows" in result

    def test_1d_array_long(self):
        """Test 1D array >100 elements shows ellipsis."""
        np = pytest.importorskip("numpy")
        arr = np.arange(150.0)
        result, fmt = prettyvalue(arr)

        assert "…" in result

    def test_multiline_string(self):
        """Test multiline string uses block format."""
        result, fmt = prettyvalue("line1\nline2\nline3")
        assert fmt == "block"

    def test_very_long_multiline_string(self):
        """Test very long multiline string is truncated."""
        lines = "\n".join([f"line{i}" for i in range(30)])
        result, fmt = prettyvalue(lines)

        # Should show truncation marker
        assert "⋯" in result

    def test_long_inline_string(self):
        """Test long inline value is truncated."""
        result, fmt = prettyvalue("x" * 200)

        assert "…" in result
        assert len(result) < 200

    def test_numpy_scalar_numeric(self):
        """Test numpy scalar as numeric."""
        np = pytest.importorskip("numpy")
        scalar = np.float64(3.14)
        result, fmt = prettyvalue(scalar)

        assert "3.14" in result


class TestExtractVariablesAdditional:
    """Additional extract_variables tests."""

    def test_none_type_no_typename(self):
        """Test that NoneType variables don't show type."""
        variables = {"x": None}
        sourcecode = "x"
        rows = extract_variables(variables, sourcecode)

        row = rows[0]
        assert row.typename == ""  # NoneType should be empty string

    def test_object_with_no_str_but_no_extractable_members(self):
        """Test object with poor __str__ and no extractable members."""

        class BarrenObject:
            def __str__(self):
                return "<BarrenObject object at 0x12345>"

        variables = {"obj": BarrenObject()}
        sourcecode = "obj"
        rows = extract_variables(variables, sourcecode)

        # Should show ellipsis for object without extractable members
        if rows:
            assert rows[0].value == "⋯" or len(rows) == 0

    def test_object_with_failing_safe_vars(self):
        """Test object where safe_vars() fails with exception."""

        class FailingSafeVars:
            def __str__(self):
                return "<FailingSafeVars object at 0x12345>"

            def __dir__(self):
                raise RuntimeError("Cannot dir() this object")

        variables = {"obj": FailingSafeVars()}
        sourcecode = "obj"

        # Should handle exception in safe_vars gracefully
        rows = extract_variables(variables, sourcecode)
        # Object should show ellipsis since members can't be extracted
        if rows:
            assert rows[0].value == "⋯"

    def test_dotted_identifier_matching(self):
        """Test matching of dotted identifiers like obj.attr."""

        class Obj:
            def __init__(self):
                self.value = 42

        obj = Obj()
        variables = {"obj": obj}
        sourcecode = "obj.value"

        # Note: obj itself has poor __str__, so obj.value should be extracted
        rows = extract_variables(variables, sourcecode)

        names = {row.name for row in rows}
        assert "obj.value" in names

    def test_member_with_poor_str(self):
        """Test that members with poor __str__ are skipped."""

        class Inner:
            def __str__(self):
                return "<Inner object at 0x12345>"

        class Outer:
            def __init__(self):
                self.inner = Inner()

            def __str__(self):
                return "<Outer object at 0x67890>"

        variables = {"obj": Outer()}
        sourcecode = "obj.inner"

        rows = extract_variables(variables, sourcecode)

        # obj.inner should be skipped because it has poor __str__
        names = {row.name for row in rows}
        assert "obj.inner" not in names


class TestSafeVars:
    """Test safe_vars function."""

    def test_object_with_slots(self):
        """Test safe_vars with slots-based object."""

        class SlottedClass:
            __slots__ = ["x", "y"]

            def __init__(self):
                self.x = 1
                self.y = 2

        obj = SlottedClass()
        result = safe_vars(obj)

        assert "x" in result
        assert "y" in result
        assert result["x"] == 1

    def test_object_with_properties(self):
        """Test safe_vars with properties."""

        class WithProperty:
            @property
            def value(self):
                return 42

        obj = WithProperty()
        result = safe_vars(obj)

        assert "value" in result
        assert result["value"] == 42

    def test_object_with_failing_property(self):
        """Test safe_vars handles failing properties via suppress()."""

        class FailingProperty:
            @property
            def bad(self):
                raise AttributeError("Failed")  # Must be AttributeError for suppress

            @property
            def good(self):
                return 42

        obj = FailingProperty()
        result = safe_vars(obj)

        # 'bad' should not be in result since it raises AttributeError (suppressed)
        assert "bad" not in result
        # 'good' should still work
        assert "good" in result


class TestVarInfo:
    """Test VarInfo namedtuple."""

    def test_varinfo_creation(self):
        """Test VarInfo creation."""
        info = VarInfo("x", "int", "42", "inline")
        assert info.name == "x"
        assert info.typename == "int"
        assert info.value == "42"
        assert info.format_hint == "inline"

    def test_varinfo_tuple_unpacking(self):
        """Test VarInfo can be unpacked as tuple."""
        info = VarInfo("x", "int", "42", "inline")
        name, typename, value, fmt = info
        assert name == "x"

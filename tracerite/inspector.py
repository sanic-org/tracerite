from __future__ import annotations

import ast
import contextlib
import dataclasses
import math
import re
import types
from collections import namedtuple
from collections.abc import Callable
from typing import Any

from .logging import logger

# Minimum length for a string value to be considered a match against the
# exception message.  Short strings are too likely to collide with unrelated
# variables by accident.
_EXCEPTION_MESSAGE_MIN_MATCH_LEN = 12

# Variable info with formatting metadata
VarInfo = namedtuple("VarInfo", ["name", "typename", "value", "format_hint"])

blacklist_names = {"_", "In", "Out"}
blacklist_types = (
    type,
    types.ModuleType,
    types.FunctionType,
    types.MethodType,
    types.BuiltinFunctionType,
)
no_str_conv = re.compile(r"<.* object at 0x[0-9a-fA-F]{5,}>")

# Superscript digits for formatting powers of 10
_SUPERSCRIPT_DIGITS = str.maketrans("-0123456789", "⁻⁰¹²³⁴⁵⁶⁷⁸⁹")


def _format_scalar(v: Any) -> str:
    """Format a single numeric value intelligently."""
    # Integer types (including numpy integers) - display without decimals
    if isinstance(v, int) or (hasattr(v, "dtype") and "int" in str(v.dtype)):
        return str(int(v))
    # Float types
    if isinstance(v, float) or (hasattr(v, "dtype") and "float" in str(v.dtype)):
        v = float(v)
        if v != v:  # NaN
            return "NaN"
        if abs(v) == float("inf"):
            return "∞" if v > 0 else "-∞"
        if v == 0:
            return "0"
        if v == int(v) and abs(v) < 1e15:
            return str(int(v))
        # Compact fixed-point: strip trailing zeros
        return f"{v:.6f}".rstrip("0").rstrip(".")
    return str(v)


def _get_flat(arr: Any) -> list[Any]:
    """Get a flat/1D view of an array, supporting numpy and torch."""
    if hasattr(arr, "flat"):
        return arr.flat
    if hasattr(arr, "flatten"):
        return arr.flatten()
    return arr


def _array_formatter(arr: Any) -> tuple[Callable[[Any], str], str]:
    """
    Create an optimal formatter for displaying array values consistently.

    For integers: display as integers without decimals.
    For floats: determine scale from max(abs(values)), apply SI-style
    scaling (×10⁶, ×10⁻³, etc.), and display with consistent fixed precision.
    Returns (formatter_func, scale_suffix) where scale_suffix may be empty.
    """
    try:
        dtype_str = str(arr.dtype)
    except AttributeError:
        dtype_str = ""

    # Integer arrays - display as integers, no scaling
    if "int" in dtype_str or "bool" in dtype_str:
        return lambda v: str(int(v)), ""

    # For float arrays, analyze the values to determine optimal formatting
    if "float" in dtype_str or "complex" in dtype_str:
        flat = _get_flat(arr)
        n = len(flat)
        if n <= 200:
            sample = [float(v) for v in flat]
        else:
            sample = [float(flat[i]) for i in range(100)]
            sample += [float(flat[n - 100 + i]) for i in range(100)]

        finite = [abs(v) for v in sample if v == v and abs(v) != float("inf")]
        if not finite:
            return lambda v: "NaN" if v != v else ("∞" if v > 0 else "-∞"), ""

        max_abs = max(finite)
        log_max = math.log10(max_abs) if max_abs > 0 else 0
        scale_power = int(log_max // 3) * 3
        if scale_power in (-3, 0, 3):
            scale_power = 0
        scale_suffix = (
            f"×10{str(scale_power).translate(_SUPERSCRIPT_DIGITS)}"
            if scale_power
            else ""
        )
        scale_factor = 10.0 ** (-scale_power) if scale_power else 1.0
        log_scaled = log_max - scale_power if scale_power else log_max
        decimals = max(0, 2 - math.floor(log_scaled)) if max_abs > 0 else 0

        def fmt(v: Any, sf: float = scale_factor, d: int = decimals) -> str:
            if v != v:
                return "NaN"
            if v == float("inf"):
                return "∞"
            if v == float("-inf"):
                return "-∞"
            scaled = v * sf
            if scaled == 0:
                return "0"
            return f"{scaled:.{d}f}"

        return fmt, scale_suffix

    return (lambda v: f"{v}"), ""


class _IdentifierVisitor(ast.NodeVisitor):
    """AST visitor that collects variable identifiers from source code."""

    def __init__(self):
        self.identifiers: set[str] = set()

    def visit_Name(self, node: ast.Name) -> None:
        self.identifiers.add(node.id)
        self.generic_visit(node)

    def visit_Attribute(self, node: ast.Attribute) -> None:
        parts = [node.attr]
        current = node.value
        while isinstance(current, ast.Attribute):
            parts.append(current.attr)
            current = current.value
        if isinstance(current, ast.Name):
            parts.append(current.id)
            self.identifiers.add(".".join(reversed(parts)))
            for i in range(len(parts)):
                self.identifiers.add(".".join(reversed(parts[i:])))
        self.generic_visit(node)


def _extract_identifiers_ast(sourcecode: str) -> set[str] | None:
    """
    Extract variable identifiers from source code using AST.

    Returns a set of variable names (including attribute access like "obj.attr"),
    or None if AST parsing fails.
    """
    # Try parsing as an expression first (most common case for error lines)
    for wrapper in ("({})", "{}"):
        try:
            code = wrapper.format(sourcecode)
            tree = ast.parse(code, mode="eval")
            break
        except SyntaxError:
            continue
    else:
        try:
            tree = ast.parse(sourcecode, mode="exec")
        except SyntaxError:
            return None

    visitor = _IdentifierVisitor()
    visitor.visit(tree)
    return visitor.identifiers


def _extract_identifiers_regex(sourcecode: str) -> set[str]:
    """
    Extract variable identifiers from source code using regex (fallback).

    This is less accurate than AST as it can match names in strings/comments,
    but works when AST parsing fails.
    """
    return {
        m.group(0) for p in (r"\w+", r"\w+\.\w+") for m in re.finditer(p, sourcecode)
    }


def extract_variables(
    variables: dict[str, Any], sourcecode: str, exc_message: str | None = None
) -> list[VarInfo]:
    """Extract variable values that appear in sourcecode for display."""
    identifiers = _extract_identifiers_ast(sourcecode)
    if identifiers is None:
        identifiers = _extract_identifiers_regex(sourcecode)

    rows = []
    for name, value in variables.items():
        if name in blacklist_names or isinstance(value, blacklist_types):
            continue
        try:
            rows.extend(
                _extract_variable_rows(
                    name, value, identifiers, sourcecode, exc_message
                )
            )
        except Exception:
            logger.exception("Variable inspector failed (please report a bug)")
    return rows


def _extract_variable_rows(
    name: str,
    value: Any,
    identifiers: set[str],
    sourcecode: str,
    exc_message: str | None,
) -> list[VarInfo]:
    """Return VarInfo rows for a single variable, expanding members if needed."""
    typename = type(value).__name__
    if name not in identifiers:
        return []

    try:
        strvalue = str(value)
        reprvalue = repr(value)
    except Exception:
        return []

    if not strvalue and reprvalue:
        strvalue = reprvalue

    if _is_exception_message_variable(strvalue, typename, exc_message):
        return []

    if no_str_conv.fullmatch(strvalue):
        member_rows = _extract_member_rows(name, value, identifiers, sourcecode)
        if member_rows:
            return member_rows
        value = "⋯"

    typename = _annotate_array_type(typename, value)
    val_str, val_fmt = prettyvalue(value)
    if typename == "NoneType":
        typename = ""
    return [VarInfo(name, typename, val_str, val_fmt)]


def _is_exception_message_variable(
    strvalue: str, typename: str, exc_message: str | None
) -> bool:
    """Check whether a string variable is the exception message itself."""
    return (
        exc_message is not None
        and typename == "str"
        and len(exc_message) >= _EXCEPTION_MESSAGE_MIN_MATCH_LEN
        and strvalue == exc_message
    )


def _extract_member_rows(
    name: str, value: Any, identifiers: set[str], sourcecode: str
) -> list[VarInfo]:
    """Extract members of an object with a poor __str__ representation."""
    rows = []
    try:
        members = safe_vars(value).items()
    except Exception:
        members = []

    typename = type(value).__name__
    for n, v in members:
        mname = f"{name}.{n}"
        if sourcecode and mname not in identifiers:
            continue
        if isinstance(v, blacklist_types):
            continue
        try:
            member_str = str(v)
            if no_str_conv.fullmatch(member_str):
                continue
        except Exception:
            continue
        tname = f"{type(v).__name__} in {typename}"
        val_str, val_fmt = prettyvalue(v)
        rows.append(VarInfo(mname, tname, val_str, val_fmt))
    return rows


def _annotate_array_type(typename: str, value: Any) -> str:
    """Append dtype/shape/device info for array/tensor-like values."""
    try:
        dtype = str(object.__getattribute__(value, "dtype")).rsplit(".", 1)[-1]
        if typename == dtype:
            raise AttributeError
        shape = object.__getattribute__(value, "shape")
        dims = "×".join(str(d + 0) for d in shape) + " " if shape else ""
        try:
            dev = object.__getattribute__(value, "device")
            dev = f"@{dev}" if dev and dev.type != "cpu" else ""
        except AttributeError:
            dev = ""
        return f"{typename} of {dims}{dtype}{dev}"
    except AttributeError:
        return typename


def safe_vars(obj: Any) -> dict[str, Any]:
    """Like vars(), but also supports objects with slots."""
    ret = {}
    for attr in dir(obj):
        with contextlib.suppress(AttributeError):
            ret[attr] = object.__getattribute__(obj, attr)
    return ret


def prettyvalue(val: Any) -> tuple[Any, str]:
    """
    Format a value for display in the inspector.

    Returns:
        tuple: (formatted_value, format_hint) where format_hint is one of:
               'block' - left-aligned block format (for multi-line strings)
               'inline' - inline right-aligned format (default)
    """
    result = _format_keyvalue_container(val)
    if result is not None:
        return result

    if isinstance(val, (list, tuple)):
        if not 0 < len(val) <= 10:
            return (f"({len(val)} items)", "inline")
        return (", ".join(repr(v)[:80] for v in val), "inline")

    if isinstance(val, type):
        return (f"{val.__module__}.{val.__name__}", "inline")

    result = _format_array_like(val)
    if result is not None:
        return result

    result = _format_scalar_value(val)
    if result is not None:
        return result

    return _format_default_value(val)


def _format_keyvalue_container(val: Any) -> tuple[Any, str] | None:
    """Format namedtuple, dict, dataclass, or struct as a key-value table."""
    if (
        isinstance(val, tuple)
        and hasattr(val, "_fields")
        and isinstance(val._fields, tuple)
    ):
        return _build_keyvalue_table(val, val._fields, "fields", getattr)
    if isinstance(val, dict):
        if not val:
            return ("{}", "inline")
        return _build_keyvalue_table(val, list(val.keys()), "items", lambda d, k: d[k])
    if dataclasses.is_dataclass(val) and not isinstance(val, type):
        return _build_keyvalue_table(
            val,
            [f.name for f in dataclasses.fields(val)],
            "fields",
            object.__getattribute__,
        )
    struct_fields = _struct_fields(val)
    if struct_fields is not None:
        return _build_keyvalue_table(
            val, struct_fields, "fields", object.__getattribute__
        )
    return None


def _build_keyvalue_table(
    val: Any,
    fields: tuple[Any, ...] | list[Any],
    item_label: str,
    getter: Callable[[Any, Any], Any],
) -> tuple[Any, str]:
    """Build a key-value table or summary for a container with named fields."""
    if not fields:
        return (f"{type(val).__name__}()", "inline")
    if len(fields) > 10:
        return (f"({len(fields)} {item_label})", "inline")

    rows = []
    for name in fields:
        key_str = name if len(name) <= 40 else name[:37] + "…"
        field_val = getter(val, name)
        val_str = f"{field_val!s}"
        if len(val_str) > 60:
            val_str = val_str[:57] + "…"
        rows.append([key_str, val_str])
    return ({"type": "keyvalue", "rows": rows}, "inline")


def _struct_fields(val: Any) -> tuple[str, ...] | None:
    """Return field names for msgspec Struct or Pydantic BaseModel, if applicable."""
    if isinstance(fields := getattr(type(val), "__struct_fields__", None), tuple):
        return fields
    if isinstance(fields := getattr(type(val), "model_fields", None), dict):
        return tuple(fields.keys())
    return None


def _format_array_like(val: Any) -> tuple[Any, str] | None:
    """Format numpy/torch arrays and similar array-like objects."""
    try:
        shape = object.__getattribute__(val, "shape")
        if not (isinstance(shape, tuple) and val.shape):
            return None

        numelem = math.prod(shape)
        if numelem <= 1:
            flat = _get_flat(val)
            return (_format_scalar(flat[0]), "inline")

        if len(shape) == 1:
            return _format_1d_array(val, shape[0])

        if len(shape) == 2 and shape[0] <= 10 and shape[1] <= 10:
            return _format_2d_array(val)
    except (AttributeError, ValueError):
        return None
    except Exception:
        logger.exception(
            "Pretty-printing in variable inspector failed (please report a bug)"
        )
    return None


def _format_1d_array(val: Any, length: int) -> tuple[str, str]:
    """Format a one-dimensional array."""
    fmt, suffix = _array_formatter(val)
    if length <= 100:
        result = ", ".join(fmt(v) for v in val)
    else:
        formatted = [fmt(v) for v in (*val[:3], *val[-3:])]
        result = ", ".join([*formatted[:3], "…", *formatted[-3:]])
    if suffix:
        result = f"{result} {suffix}"
    return result, "inline"


def _format_2d_array(val: Any) -> tuple[Any, str]:
    """Format a small two-dimensional array."""
    fmt, suffix = _array_formatter(val)
    table = [[fmt(v) for v in row] for row in val]
    if suffix:
        return ({"type": "array", "rows": table, "suffix": suffix}, "inline")
    return table, "inline"


def _format_scalar_value(val: Any) -> tuple[Any, str] | None:
    """Format numpy scalars and plain numeric values."""
    try:
        dtype_str = str(getattr(val, "dtype", ""))
        shape = getattr(val, "shape", ())
        is_scalar = not shape or (isinstance(shape, tuple) and len(shape) == 0)
        is_numeric = isinstance(val, (int, float)) or (dtype_str and is_scalar)
        if is_numeric and not isinstance(val, bool):
            return (_format_scalar(val), "inline")
    except (AttributeError, TypeError, ValueError):
        pass
    return None


def _format_default_value(val: Any) -> tuple[Any, str]:
    """Format arbitrary values using str/repr and apply truncation rules."""
    if isinstance(val, BaseException):
        ret = str(val)
    elif isinstance(val, str):
        ret = str(val)
        if "\n" in ret.rstrip():
            return _format_block_string(ret)
        return _format_inline_string(ret)
    else:
        ret = repr(val)

    return _format_inline_string(ret)


def _format_inline_string(ret: str) -> tuple[str, str]:
    """Collapse newlines and truncate long single-line values."""
    if "\n" in ret:
        ret = " ".join(line.strip() for line in ret.split("\n") if line.strip())
    if len(ret) > 120:
        ret = ret[:30] + " … " + ret[-30:]
    return ret, "inline"


def _format_block_string(ret: str) -> tuple[str, str]:
    """Format multi-line strings, truncating very long ones."""
    lines = ret.split("\n")
    if len(lines) <= 15:
        return ret, "block"

    first = lines[:5]
    while first and first[-1].strip() == "":
        first.pop()

    last: list[str] = []
    for line in reversed(lines):
        if line.strip() != "":
            last.append(line)
            if len(last) == 2:
                break
    last.reverse()

    return "\n".join(first + ["⋮"] + last), "block"

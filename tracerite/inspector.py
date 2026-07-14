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

from tracerite.logging import logger

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
    dtype = str(v.dtype) if hasattr(v, "dtype") else ""
    if isinstance(v, int) or "int" in dtype:
        return str(int(v))
    if isinstance(v, float) or "float" in dtype:
        v = float(v)
        if math.isnan(v):
            return "NaN"
        if math.isinf(v):
            return "∞" if v > 0 else "-∞"
        if v == 0:
            return "0"
        if v == int(v) and abs(v) < 1e15:
            return str(int(v))
        # Compact fixed-point: strip trailing zeros
        return f"{v:.6f}".rstrip("0").rstrip(".")
    return str(v)


def _get_flat(arr: Any) -> Any:
    """Get a flat/1D view of an array, supporting numpy and torch."""
    if hasattr(arr, "flat"):
        return arr.flat
    if hasattr(arr, "flatten"):
        return arr.flatten()
    return arr


def _array_formatter(arr: Any) -> tuple[Callable[[Any], str], str]:
    """Create an optimal formatter for displaying array values consistently."""
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
        sample = [float(v) for v in (flat if n <= 200 else (*flat[:100], *flat[-100:]))]

        finite = [abs(v) for v in sample if math.isfinite(v)]
        if not finite:
            return (
                lambda v: "NaN" if math.isnan(float(v)) else ("∞" if v > 0 else "-∞"),
                "",
            )

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
            v = float(v)
            if math.isnan(v):
                return "NaN"
            if math.isinf(v):
                return "∞" if v > 0 else "-∞"
            scaled = v * sf
            if scaled == 0:
                return "0"
            return f"{scaled:.{d}f}"

        return fmt, scale_suffix

    return lambda v: f"{v}", ""


class _IdentifierVisitor(ast.NodeVisitor):
    """AST visitor that collects variable identifiers from source code."""

    def __init__(self):
        self.identifiers: set[str] = set()

    def visit_Name(self, node: ast.Name) -> None:
        self.identifiers.add(node.id)
        self.generic_visit(node)

    def visit_Attribute(self, node: ast.Attribute) -> None:
        parts: list[str] = []
        current: ast.AST = node
        while isinstance(current, ast.Attribute):
            parts.append(current.attr)
            current = current.value
        if isinstance(current, ast.Name):
            parts.append(current.id)
            self.identifiers.update(
                ".".join(reversed(parts[i:])) for i in range(len(parts))
            )
        self.generic_visit(node)


def _extract_identifiers_ast(sourcecode: str) -> set[str] | None:
    """Extract variable identifiers from source code using AST."""
    tree: ast.AST | None = None
    for wrapper in ("({})", "{}"):
        with contextlib.suppress(SyntaxError):
            tree = ast.parse(wrapper.format(sourcecode), mode="eval")
            break
    if tree is None:
        with contextlib.suppress(SyntaxError):
            tree = ast.parse(sourcecode, mode="exec")
    if tree is None:
        return None

    visitor = _IdentifierVisitor()
    visitor.visit(tree)
    return visitor.identifiers


def _extract_identifiers_regex(sourcecode: str) -> set[str]:
    """Extract variable identifiers from source code using regex (fallback)."""
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

    rows: list[VarInfo] = []
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
    if name not in identifiers:
        return []

    try:
        strvalue = str(value)
        reprvalue = repr(value)
    except Exception:
        return []

    if not strvalue and reprvalue:
        strvalue = reprvalue

    typename = type(value).__name__
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
        typename == "str"
        and exc_message is not None
        and len(exc_message) >= _EXCEPTION_MESSAGE_MIN_MATCH_LEN
        and strvalue == exc_message
    )


def _safe_str(obj: Any) -> str | None:
    """Return str(obj), or None if it raises."""
    with contextlib.suppress(Exception):
        return str(obj)
    return None


def _extract_member_rows(
    name: str, value: Any, identifiers: set[str], sourcecode: str
) -> list[VarInfo]:
    """Extract members of an object with a poor __str__ representation."""
    try:
        members = safe_vars(value).items()
    except Exception:
        members = []

    typename = type(value).__name__
    return [
        VarInfo(mname, f"{type(v).__name__} in {typename}", *prettyvalue(v))
        for n, v in members
        if (not sourcecode or (mname := f"{name}.{n}") in identifiers)
        and not isinstance(v, blacklist_types)
        and (member_str := _safe_str(v)) is not None
        and not no_str_conv.fullmatch(member_str)
    ]


def _annotate_array_type(typename: str, value: Any) -> str:
    """Append dtype/shape/device info for array/tensor-like values."""
    try:
        dtype = str(object.__getattribute__(value, "dtype")).rsplit(".", 1)[-1]
        if typename == dtype:
            raise AttributeError
        shape = object.__getattribute__(value, "shape")
        dims = "×".join(str(d) for d in shape) + " " if shape else ""
        dev = ""
        with contextlib.suppress(AttributeError):
            device = object.__getattribute__(value, "device")
            if device and device.type != "cpu":
                dev = f"@{device}"
        return f"{typename} of {dims}{dtype}{dev}"
    except AttributeError:
        return typename


_MISSING = object()


def _get_attr_safe(obj: Any, attr: str) -> Any:
    """Return object.__getattribute__(obj, attr), or _MISSING if it raises."""
    with contextlib.suppress(AttributeError):
        return object.__getattribute__(obj, attr)
    return _MISSING


def safe_vars(obj: Any) -> dict[str, Any]:
    """Like vars(), but also supports objects with slots."""
    return {
        attr: val
        for attr in dir(obj)
        if (val := _get_attr_safe(obj, attr)) is not _MISSING
    }


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
        if 0 < len(val) <= 10:
            return (", ".join(repr(v)[:80] for v in val), "inline")
        return (f"({len(val)} items)", "inline")

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


def _truncated(s: str, limit: int = 60) -> str:
    """Truncate a string with an ellipsis if it exceeds the limit."""
    return s if len(s) <= limit else s[: limit - 3] + "…"


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

    rows = [
        [_truncated(name, 40), _truncated(f"{getter(val, name)!s}", 60)]
        for name in fields
    ]
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
        if not (isinstance(shape, tuple) and shape):
            return None

        if math.prod(shape) <= 1:
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
        items = [fmt(v) for v in (*val[:3], *val[-3:])]
        result = ", ".join([*items[:3], "…", *items[-3:]])
    return (f"{result} {suffix}".rstrip(), "inline")


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
        if (
            isinstance(val, (int, float)) or (dtype_str and is_scalar)
        ) and not isinstance(val, bool):
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

    last = [line for line in reversed(lines) if line.strip() != ""][:2]
    last.reverse()

    return "\n".join(first + ["⋮"] + last), "block"

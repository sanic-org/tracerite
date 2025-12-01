import contextlib
import re
import types
from collections import namedtuple
from functools import reduce

from .logging import logger

# Variable info with formatting metadata
VarInfo = namedtuple("VarInfo", ["name", "typename", "value", "format_hint"])

blacklist_names = {"_", "In", "Out"}
blacklist_types = (
    types.ModuleType,
    types.FunctionType,
    types.MethodType,
    types.BuiltinFunctionType,
)
no_str_conv = re.compile(r"<.* object at 0x[0-9a-fA-F]{5,}>")


def extract_variables(variables, sourcecode):
    identifiers = {
        m.group(0) for p in (r"\w+", r"\w+\.\w+") for m in re.finditer(p, sourcecode)
    }
    rows = []
    for name, value in variables.items():
        if name in blacklist_names or isinstance(value, blacklist_types):
            continue
        try:
            typename = type(value).__name__
            if name not in identifiers:
                continue
            try:
                strvalue = str(value)
                reprvalue = repr(value)
            except Exception:
                continue  # Skip variables failing str() or repr()
            # Using repr is better for empty strings and some other cases
            if not strvalue and reprvalue:
                strvalue = reprvalue
            # Try to print members of objects that don't have proper __str__
            elif no_str_conv.fullmatch(strvalue):
                found = False
                for n, v in safe_vars(value).items():
                    mname = f"{name}.{n}"
                    if sourcecode and mname not in identifiers:
                        continue
                    tname = type(v).__name__
                    if isinstance(v, blacklist_types):
                        continue
                    # Check if the member also has a poor representation
                    try:
                        member_str = str(v)
                        if no_str_conv.fullmatch(member_str):
                            continue  # Skip members with poor representations
                    except Exception:
                        continue  # Skip members that fail str()
                    tname += f" in {typename}"
                    val_str, val_fmt = prettyvalue(v)
                    rows += (VarInfo(mname, tname, val_str, val_fmt),)
                    found = True
                if found:
                    continue
                value = "⋯"
            # Full types for Numpy-like arrays, PyTorch tensors, etc.
            try:
                dtype = str(object.__getattribute__(value, "dtype")).rsplit(".", 1)[-1]
                if typename == dtype:
                    raise AttributeError  # Numpy scalars need no further info
                shape = object.__getattribute__(value, "shape")
                dims = "×".join(str(d + 0) for d in shape) + " " if shape else ""
                try:
                    dev = object.__getattribute__(value, "device")
                    dev = f"@{dev}" if dev and dev.type != "cpu" else ""
                except AttributeError:
                    dev = ""
                typename += f" of {dims}{dtype}{dev}"
            except AttributeError:
                pass
            val_str, val_fmt = prettyvalue(value)
            rows += (VarInfo(name, typename, val_str, val_fmt),)
        except Exception:
            logger.exception("Variable inspector failed (please report a bug)")
    return rows


def safe_vars(obj):
    """Like vars(), but also supports objects with slots."""
    ret = {}
    for attr in dir(obj):
        with contextlib.suppress(AttributeError):
            ret[attr] = object.__getattribute__(obj, attr)
    return ret


def prettyvalue(val):
    """
    Format a value for display in the inspector.

    Returns:
        tuple: (formatted_value, format_hint) where format_hint is one of:
               'block' - left-aligned block format (for multi-line strings)
               'inline' - inline right-aligned format (default)
    """
    if isinstance(val, (list, tuple)):
        if not 0 < len(val) <= 10:
            return (f"({len(val)} items)", "inline")
        return (", ".join(repr(v)[:80] for v in val), "inline")
    if isinstance(val, dict):
        # Handle dict formatting specially
        if not val:
            return ("{}", "inline")
        if len(val) <= 5:
            # Try to fit on single line for small dicts
            items = [f"{k!r}: {v!r}" for k, v in list(val.items())[:5]]
            single_line = "{" + ", ".join(items) + "}"
            if len(single_line) <= 120:
                return (single_line, "inline")
        # For larger dicts or those that don't fit, show summary
        return (f"({len(val)} items)", "inline")
    if isinstance(val, type):
        return (f"{val.__module__}.{val.__name__}", "inline")
    try:
        # This only works for Numpy-like arrays, and should cause exceptions otherwise
        shape = object.__getattribute__(val, "shape")
        if isinstance(shape, tuple) and val.shape:
            numelem = reduce(lambda x, y: x * y, shape)
            if numelem <= 1:
                return (f"{val[0]:.2g}", "inline")
            # 1D arrays
            if len(shape) == 1:
                if shape[0] <= 100:
                    return (", ".join(f"{v:.2f}" for v in val), "inline")
                else:
                    fmt = [f"{v:.2f}" for v in (*val[:3], *val[-3:])]
                    return (", ".join([*fmt[:3], "…", *fmt[-3:]]), "inline")
            # 2D arrays
            if len(shape) == 2 and shape[0] <= 10 and shape[1] <= 10:
                return ([[f"{v:.2f}" for v in row] for row in val], "inline")
    except (AttributeError, ValueError):
        pass
    except Exception:
        logger.exception(
            "Pretty-printing in variable inspector failed (please report a bug)"
        )

    try:
        floaty = isinstance(val, float) or "float" in str(val.dtype)
        if floaty:
            ret = f"{val:.2g}"
        else:
            ret = None
    except (AttributeError, TypeError):
        floaty = False
        ret = None

    # Determine format hint based on content
    format_hint = "inline"

    if floaty and ret:
        pass
    elif isinstance(val, str):
        ret = str(val)
        # Multi-line strings should be displayed as blocks
        if "\n" in ret or len(ret) > 80:
            format_hint = "block"
    else:
        ret = repr(val)

    # For inline format, collapse newlines to avoid display issues
    if format_hint == "inline":
        if "\n" in ret:
            ret = " ".join(line.strip() for line in ret.split("\n") if line.strip())
        # Only truncate inline values
        if len(ret) > 120:
            ret = ret[:30] + " … " + ret[-30:]
    # For block format, don't truncate but limit line count if needed
    elif format_hint == "block":
        lines = ret.split("\n")
        if len(lines) > 20:
            # Show first 10 and last 10 lines
            ret = "\n".join(lines[:10] + ["⋯"] + lines[-10:])

    return (ret, format_hint)

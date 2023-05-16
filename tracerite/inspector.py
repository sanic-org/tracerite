import re
import types
from functools import reduce

from .logging import logger

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
        m.group(0)
        for p in (r'\w+', r'\w+\.\w+')
        for m in re.finditer(p, sourcecode)
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
                    mname = f'{name}.{n}'
                    if sourcecode and mname not in identifiers:
                        continue
                    tname = type(v).__name__
                    if tname in blacklist_types:
                        continue
                    tname += f' in {typename}'
                    rows += (mname, tname, prettyvalue(v)),
                    found = True
                if found:
                    continue
                value = '⋯'
            # Full types for Numpy-like arrays, PyTorch tensors, etc.
            try:
                dtype = str(object.__getattribute__(value, 'dtype')).rsplit(".", 1)[-1]
                if typename == dtype:
                    raise AttributeError   # Numpy scalars need no further info
                shape = object.__getattribute__(value, 'shape')
                dims = "×".join(str(d + 0) for d in shape) + " " if shape else ""
                try:
                    dev = object.__getattribute__(value, 'device')
                    dev = f"@{dev}" if dev and dev.type != "cpu" else ""
                except AttributeError:
                    dev = ""
                typename += f' of {dims}{dtype}{dev}'
            except AttributeError:
                pass
            rows += (name, typename, prettyvalue(value)),
        except Exception:
            logger.exception("Variable inspector failed (please report a bug)")
    return rows


def safe_vars(obj):
    """Like vars(), but also supports objects with slots."""
    ret = {}
    for attr in dir(obj):
        try:
            ret[attr] = object.__getattribute__(obj, attr)
        except AttributeError:
            pass  # Slots that haven't been set
    return ret


def prettyvalue(val):
    if isinstance(val, (list, tuple)):
        if not 0 < len(val) <= 10:
            return f'({len(val)} items)'
        return ", ".join(repr(v)[:80] for v in val)
    if isinstance(val, type):
        return f"{val.__module__}.{val.__name__}"
    try:
        # This only works for Numpy-like arrays, and should cause exceptions otherwise
        shape = object.__getattribute__(val, 'shape')
        if isinstance(shape, tuple) and val.shape:
            numelem = reduce(lambda x, y: x * y, shape)
            if numelem <= 1:
                return f"{val[0]:.2g}"
            # 1D arrays
            if len(shape) == 1:
                if shape[0] <= 100:
                    return ", ".join(f"{v:.2f}" for v in val)
                else:
                    fmt = [f"{v:.2f}" for v in (*val[:3], *val[-3:])]
                    return ", ".join([*fmt[:3], "…", *fmt[-3:]])
            # 2D arrays
            if len(shape) == 2 and shape[0] <= 10 and shape[1] <= 10:
                return [[f"{v:.2f}" for v in row] for row in val]
    except (AttributeError, ValueError):
        pass
    except Exception:
        logger.exception("Pretty-printing in variable inspector failed (please report a bug)")

    try:
        floaty = isinstance(val, float) or "float" in str(val.dtype)
    except AttributeError:
        floaty = False

    if floaty: ret = f"{val:.2g}"
    elif isinstance(val, str): ret = str(val)
    else: ret = repr(val)

    if len(ret) > 120:
        return ret[:30] + " … " + ret[-30:]
    return ret

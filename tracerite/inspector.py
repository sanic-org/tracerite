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
no_str_conv = re.compile(r"<.* object at 0x[0-9a-f]{5,}>")


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
            except Exception:
                continue  # Skip variables failing str()
            # Try to print members of objects that don't have proper __str__
            if no_str_conv.fullmatch(strvalue):
                found = False
                for n, v in vars(value).items():
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
                value = ''
            # Append dtype on Numpy-style arrays (but not on np.float64 etc)
            # Access attributes directly to avoid triggering __getattr__
            try:
                dtype = object.__getattribute__(value, 'dtype')
                shape = object.__getattribute__(value, 'shape')
                if dtype and shape:
                    try:
                        dims = "×".join(str(d + 0) for d in shape) + " "
                    except ValueError:
                        dims = ""
                    typename += f' of {dims}{dtype}'
            except AttributeError:
                pass
            rows += (name, typename, prettyvalue(value)),
        except Exception:
            logger.exception("Variable inspector failed (please report a bug)")
    return rows


def prettyvalue(val):
    if isinstance(val, (list, tuple)):
        if not 0 < len(val) <= 10:
            return f'({len(val)} items)'
        return ", ".join(repr(v)[:80] for v in val)
    try:
        # This only works for Numpy-like arrays, and should cause exceptions otherwise
        shape = object.__getattribute__(val, 'shape')
        if isinstance(shape, tuple) and val.shape:
            if len(shape) <= 2 and reduce(lambda a, b: a * b, shape) <= 100:
                return val
    except AttributeError:
        pass
    except Exception:
        logger.exception("Pretty-printing in variable inspector failed (please report a bug)")
    ret = f"{val}"
    if len(ret) > 120:
        return ret[:30] + " … " + ret[-30:]
    return ret

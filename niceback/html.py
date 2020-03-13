import inspect
import os
import re
import sys

from urllib.parse import quote

from html5tagger import E

style = """\
    details summary {margin-left: -1em}
    details:not([open]) .hideable {display: none}
    details[open] .revhideable {display: none}
    code {background: #eee}
    .exctype {color: gray}
"""

def html_traceback(skip_outmost=1):
    ipython_input = re.compile(r"<ipython-input-(\d+)-\w+>")
    # Locations considered to be bug-free
    libdir = re.compile(r'/usr/.*|.*(site-packages|dist-packages).*')
    exc, val, tb = sys.exc_info()
    try:
        tb = inspect.getinnerframes(tb)[skip_outmost:]
    except IndexError:  # Bug in inspect internals, find_source()
        tb = None
    doc = E.style(style)
    # Header and exception message
    message = val.message if hasattr(val, 'message') else str(val)
    summary = message.split("\n", 1)[0]
    if len(summary) > 100:
        if len(message) > 1000:
            # Sometimes the useful bit is at the end of a very long message
            summary = f"{message[:40]} ··· {message[-40:]}"
        else:
            summary = f"{summary[:60]} ···"
    doc.h3(E.span(f"{exc.__name__}:", class_="exctype")(f" {summary}"))
    if summary != message:
        # Full longer/multi-line error message
        if len(message) > 1000:
            doc.details(E.summary(E.strong("Full message").span(" (click to open)", class_="revhideable")).pre(message))
        else:
            doc.pre(message)
    # Traceback
    if not tb:
        return doc.p("No traceback available!")
    # Choose a frame to open by default
    bug_in_frame = next(
        (f for f in reversed(tb) if f.code_context and not libdir.fullmatch(f.filename)),
        tb[-1]
    ).frame
    for frameinfo in tb:
        frame, filename, lineno, function, codeline, _ = frameinfo
        codeline = codeline[0].strip() if codeline else None
        # Add class name to methods (if self or cls is the first local variable)
        try:
            cls = next(
                v.__class__ if n == 'self' else v
                for n, v in frame.f_locals.items()
                if n in ('self', 'cls')
            )
            function = f'{cls.__name__}.{function}'
        except:
            pass
        function = '.'.join(function.split('.')[-2:])
        split = 0
        if len(filename) > 40:
            split = filename.rfind("/", 10, len(filename) - 20) + 1
        m = ipython_input.fullmatch(filename)
        with doc.details(open=frame is bug_in_frame):
            with doc.summary:
                if m:
                    doc(f"In [{m.group(1)}]")
                else:
                    doc.span(filename[:split], class_="hideable")
                doc(filename[split:], E.span(f":{lineno}", class_="hideable"))
                if function != "<module>":
                    doc(", ").strong(function)
                if codeline:
                    doc(": ").code(codeline, class_="revhideable")
                lines = []
                try:
                    lines, start = inspect.getsourcelines(frame)
                    if start == 0:
                        start = 1  # Zero is always returned for modules; fix that.
                    # Limit lines shown
                    lines = lines[max(0, lineno - start - 15):max(0, lineno - start + 3)]
                    start += max(0, lineno - start - 15)
                    # Deindent
                    while lines and lines[0][:1] in ' \t' and all(line[:1] == lines[0][0] for line in lines):
                        lines = [line[1:] for line in lines]
                except:
                    pass
            with doc.pre:
                for i, line in enumerate(lines, start=start):
                    doc(marked(line) if i == lineno else line)
            urls = {}
            if os.path.isfile(filename):
                urls["VS Code"] = f"vscode://file/{quote(filename)}:{lineno}"
            if filename.startswith(os.getcwd()):
                urls["Jupyter"] = f"/edit{quote(filename[len(os.getcwd()):])}"
            if urls:
                doc.p("Open in")
                for name, href in urls.items():
                    doc(" ").a(name, href=href)
            doc(variable_inspector(frame.f_locals, '\n'.join(lines)))
    return doc


def marked(line):
    indent, code, trailing = split3(line)
    return E(indent).mark(code)(trailing)


def split3(s):
    """Split s into indent, code and trailing whitespace"""
    a, b, c = s.rstrip(), s.strip(), s.lstrip()
    l = len(b)
    return a[:-l], b, c[l:]


def variable_inspector(variables, sourcecode):
    blacklist = "module", "method", "function"
    identifiers = {
        m.group(0)
        for p in (r'\w+', r'\w+\.\w+')
        for m in re.finditer(p, sourcecode)
    }
    rows = []
    for name, value in variables.items():
        if name in ("_", "In", "Out"):
            continue  # Hide IPython objects
        try:
            typename = type(value).__name__
            if typename in blacklist:
                continue
            if re.compile(r"<.* object at 0x[0-9a-f]{5,}>").fullmatch(str(value)):
                found = False
                for n, v in vars(value).items():
                    mname = f'{name}.{n}'
                    if sourcecode and mname not in identifiers:
                        continue
                    tname = type(v).__name__
                    if tname in blacklist:
                        continue
                    tname += f' in {typename}'
                    rows += (mname, tname, v),
                    found = True
                if found:
                    continue
                value = ''
            if name not in identifiers:
                continue
            # Append dtype on Numpy-style arrays (but not on np.float64 etc)
            if hasattr(value, 'dtype') and hasattr(value, "__iter__"):
                typename += f' of {value.dtype}'
            rows += (name, typename, value),
        except:
            pass
    if rows:
        with E.table as doc:
            doc.thead.tr.th("Variable").th("Type").th("Value").tbody
            for name, typename, value in rows:
                doc.tr.td(name).td(typename).td(prettyvalue(value))
        return doc

def prettyvalue(val):
    if isinstance(val, list):
        if len(val) > 10:
            return f'({len(val)} items)'
        return E(", ".join(repr(v)[:80] for v in val))
    try:
        # This only works for Numpy-like arrays, and should cause exceptions otherwise
        if val.size > 100:
            return f'({"×".join(str(d) for d in val.shape)})'
        elif len(val.shape) == 2:
            with E.table as doc:
                for row in val:
                    doc.tr
                    for num in row:
                        doc.td(f'{num:.2g}' if isinstance(num, float) else num)
            return doc
    except:
        pass
    ret = str(val)
    if len(ret) > 80:
        return ret[:30] + " … " + ret[-30:]
    return E(ret)

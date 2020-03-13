import inspect
import os
import re
import sys
from html import escape

import IPython

__version__ = "0.0.2"

def html_traceback(skip_outmost=1):
    ipython_input = re.compile(r"<ipython-input-(\d+)-\w+>")
    libdir = re.compile(r'/usr/.*|.*(site-packages|dist-packages).*')  # Locations considered to be bug-free
    exc, val, tb = sys.exc_info()
    try:
        tb = inspect.getinnerframes(tb)[skip_outmost:]
    except IndexError:  # Bug in inspect internals, find_source()
        tb = None
    html = """
        <style>
            details summary {margin-left: -1em}
            details:not([open]) .hideable {display: none}
            details[open] .revhideable {display: none}
            code {background: #eee}
        </style>
    """
    # Header and exception message
    message = val.message if hasattr(val, 'message') else str(val)
    summary = message.split("\n", 1)[0]
    if len(summary) > 100:
        if len(message) > 1000:
            # Sometimes the useful bit is at the end of a very long message
            summary = f"{message[:40]} ··· {message[-40:]}"
        else:
            summary = f"{summary[:60]} ···"
    html += f"<h3><span style=color:grey>{exc.__name__}:</span> {summary}</h3>"
    if summary != message:
        # Full longer/multi-line error message
        msghtml = f"<pre>{escape(message)}</pre>"
        if len(message) > 1000:
            msghtml = f"<details><summary><b>Full message</b><span class=revhideable> (click to open)</span></summary>{msghtml}</details>"
        html += msghtml
    # Traceback
    if not tb:
        html += "<p>No traceback available!"
        return
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
            cls = next(v.__class__ if n == 'self' else v for n, v in frame.f_locals.items() if n in ('self', 'cls'))
            function = f'{cls.__name__}.{function}'
        except:
            pass
        function = '.'.join(function.split('.')[-2:])
        split = 0
        if len(filename) > 40:
            split = filename.rfind("/", 10, len(filename) - 20) + 1
        m = ipython_input.fullmatch(filename)
        html += f"<details{' open' if frame is bug_in_frame else ''}><summary>"
        html += f"In [{m.group(1)}]" if m else f'<span class=hideable>{escape(filename[:split])}</span>{escape(filename[split:])}<span class=hideable>:{lineno}</span>'
        if function != "<module>": html += f", <b>{escape(function)}</b>"
        if codeline: html += f": <code class=revhideable>{escape(codeline)}</code>"
        lines = []
        try:
            lines, start = inspect.getsourcelines(frame)
            if start == 0: start = 1  # Zero is always returned for modules; fix that.
            # Limit lines shown
            lines = lines[max(0, lineno - start - 15) : max(0, lineno - start + 3)]
            start += max(0, lineno - start - 15)
            # Deindent
            while lines and lines[0][:1] in ' \t' and all(line[:1] == lines[0][0] for line in lines):
                lines = [line[1:] for line in lines]
        except:
            pass
        html += "</summary><pre>\n"
        html += "".join([
            marked(line) if i == lineno else escape(line)
            for i, line in enumerate(lines, start=start)
        ])
        html += "</pre>"
        showin = ""
        if os.path.isfile(filename): showin += f' <a href="vscode://file/{escape(filename)}:{lineno}">VS Code</a>'
        if filename.startswith(os.getcwd()): showin += f' <a href="/edit{escape(filename[len(os.getcwd()):])}">Jupyter</a>'
        if showin: html += f'<p>Show in{showin}'
        html += variable_inspector(frame.f_locals, '\n'.join(lines))
        html += "</details>"
    html += "</ul>"
    return IPython.display.HTML(html)

def marked(line):
    indent, code, trailing = split3(line)
    return f"{indent}<mark>{escape(code)}</mark>{trailing}"

def split3(s):
    """Split s into indent, code and trailing whitespace"""
    a, b, c = s.rstrip(), s.strip(), s.lstrip()
    l = len(b)
    return a[:-l], b, c[l:]

def variable_inspector(variables, sourcecode):
    html = ""
    blacklist = "module", "method", "function"
    identifiers = {m.group(0) for p in (r'\w+', r'\w+\.\w+') for m in re.finditer(p, sourcecode)}
    for name, value in variables.items():
        if name in ("_", "In", "Out"): continue  # Hide IPython objects
        try:
            typename = type(value).__name__
            if typename in blacklist: continue
            if re.compile(r"<.* object at 0x[0-9a-f]{5,}>").fullmatch(str(value)):
                found = False
                for n, v in vars(value).items():
                    mname = f'{name}.{n}'
                    if sourcecode and mname not in identifiers: continue
                    tname = type(v).__name__
                    if tname in blacklist: continue
                    tname += f' in {typename}'
                    html += f'<tr><td>{escape(mname)}<td>{escape(tname)}<td>{prettyvalue(v)}'
                    found = True
                if found: continue
                value = ''
            if name not in identifiers: continue
            # Append dtype on Numpy-style arrays (but not on np.float64 etc)
            if hasattr(value, 'dtype') and hasattr(value, "__iter__"):
                typename += f' of {value.dtype}'
            html += f"<tr><td>{escape(name)}<td>{escape(typename)}<td>{prettyvalue(value)}"
        except:
            pass
    return f'<table><thead><tr><th>Variable<th>Type<th>Value<tbody>{html}</table>' if html else ''

def prettyvalue(val):
    if isinstance(val, list):
        if len(val) > 10: return f'({len(val)} items)'
        return escape(", ".join(repr(v)[:80] for v in val))
    try:
        # This only works for Numpy-like arrays, and should cause exceptions otherwise
        if val.size > 100:
            return f'({"×".join(str(d) for d in val.shape)})'
        elif len(val.shape) == 2:
            ret = ""
            for row in val:
                ret += "<tr>"
                for num in row:
                    ret += "<td>" + escape(f'{num:.2g}' if isinstance(num, float) else escape(str(num)))
            return f'<table>{ret}</table>'
    except:
        pass
    ret = str(val)
    if len(ret) > 80:
        return ret[:30] + " … " + ret[-30:]
    return escape(ret)

def showtraceback(*args, **kwargs):
    try:
        IPython.display.display(html_traceback())
    except:
        original_showtraceback(*args, **kwargs)

def can_display_html():
    try:
        return get_ipython().__class__.__name__ != 'TerminalInteractiveShell'
    except:
        return False

if can_display_html():
    if not "original_showtraceback" in locals():
        original_showtraceback = IPython.core.interactiveshell.InteractiveShell.showtraceback
    IPython.core.interactiveshell.InteractiveShell.showtraceback = showtraceback

def load_jupyter_server_extension(nb_app):
    global notebook_dir
    web_app = nb_app.web_app
    host_pattern = '.*$'
    # here
    notebook_dir = nb_app.notebook_dir
    print(">>>>>>>>>>>>>>>>>", notebook_dir)

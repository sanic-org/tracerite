import inspect
import os
import re
import sys

from pathlib import Path
from secrets import token_urlsafe
from textwrap import dedent
from urllib.parse import quote

from .inspector import extract_variables
from .logging import logger

# Will be set to an instance if loaded as an IPython extension by %load_ext
ipython = None

# Locations considered to be bug-free
libdir = re.compile(r'/usr/.*|.*(site-packages|dist-packages).*')


def extract_chain(exc=None, **kwargs) -> list:
    """Extract information on current exception."""
    chain = []
    exc = exc or sys.exc_info()[1]
    while exc:
        chain.append(exc)
        if getattr(exc, "__suppress_context__", False):
            break
        exc = getattr(exc, "__cause__") or getattr(exc, "__context__")
    # Newest exception first
    return [
        extract_exception(e, **(kwargs if e is chain[0] else {})) for e in chain
    ]


def extract_exception(e, *, skip_outmost=0, skip_until=None) -> dict:
    tb = e.__traceback__
    try:
        tb = inspect.getinnerframes(tb)
    except IndexError:  # Bug in inspect internals, find_source()
        logger.exception("Bug in inspect?")
        tb = []
    if skip_until:
        for i, frame in enumerate(tb):
            if skip_until in frame.filename:
                skip_outmost = i
                break
    tb = tb[skip_outmost:]
    # Header and exception message
    message = getattr(e, "message", "") or str(e)
    summary = message.split("\n", 1)[0]
    if len(summary) > 100:
        if len(message) > 1000:
            # Sometimes the useful bit is at the end of a very long message
            summary = f"{message[:40]} ··· {message[-40:]}"
        else:
            summary = f"{summary[:60]} ···"
    try:
        # KeyboardErrors and such need not be reported all the way
        suppress = not isinstance(e, Exception)
        frames = extract_frames(tb, suppress_inner=suppress)
    except Exception:
        logger.exception("Error extracting traceback")
        frames = None
    return dict(
        type=type(e).__name__,
        message=message,
        summary=summary,
        repr=repr(e),
        frames=frames or []
    )


def extract_frames(tb, suppress_inner=False) -> list:
    if not tb:
        return []
    frames = []
    # Choose a frame to open by default
    # - The innermost non-library frame (libdir regex), or if not found,
    # - The innermost frame
    bug_in_frame = next(
        (f for f in reversed(tb) if f.code_context and not libdir.fullmatch(f.filename)),
        tb[-1]
    ).frame
    for frame, filename, lineno, function, codeline, _ in tb:
        if frame.f_globals.get("__tracebackhide__", False):
            continue
        if frame.f_locals.get("__tracebackhide__", False):
            continue
        if frame is tb[-1][0]:
            # Exception was raised here, show an error bomb normally, but
            # stop for KeyboardInterrupt, CancelledError (BaseException)
            relevance = "stop" if suppress_inner else "error"
        elif frame is bug_in_frame:
            relevance = "warning"
        else:
            relevance = "call"
        # Extract source code lines
        lines = []
        try:
            lines, start = inspect.getsourcelines(frame)
            if start == 0:
                start = 1  # Zero is always returned for modules; fix that.
            # Limit lines shown
            lines = lines[max(0, lineno - start - 15):max(0, lineno - start + 3)]
            start += max(0, lineno - start - 15)
            # Deindent
            lines = dedent("".join(lines))
        except OSError:
            lines = ""
            # Skip non-Python modules unless particularly relevant
            if relevance == "call":
                continue
        urls = {}
        # Format location
        location = None
        try:
            ipython_in = ipython.compile._filename_map[filename]
            location = f"In [{ipython_in}]"
            filename = None
        except (AttributeError, KeyError):
            pass
        # Format filename
        if filename and Path(filename).is_file():
            fn = Path(filename).resolve()
            urls["VS Code"] = f"vscode://file/{quote(fn.as_posix())}:{lineno}"
            cwd = Path.cwd()
            # Use relative path if in current working directory
            if cwd in fn.parents:
                fn = fn.relative_to(cwd)
                if ipython is not None:
                   urls["Jupyter"] = f"/edit/{quote(fn.as_posix())}"
            filename = fn.as_posix()  # Use forward slashes
        # Shorten filename to use as displayable location
        if not location:
            split = 0
            if len(filename) > 40:
                split = filename.rfind("/", 10, len(filename) - 20) + 1
            location = filename[split:]

        if function == "<module>":
            function = None
        else:
            # Add class name to methods (if self or cls is the first local variable)
            try:
                cls = next(
                    v.__class__ if n == 'self' else v
                    for n, v in frame.f_locals.items()
                    if n in ('self', 'cls') and v is not None
                )
                function = f'{cls.__name__}.{function}'
            except StopIteration:
                pass
            # Remove long module paths (keep last two items)
            function = '.'.join(function.split('.')[-2:])

        frames.append(dict(
            id=f"tb-{token_urlsafe(12)}",
            relevance=relevance,
            filename=filename,
            location=location,
            codeline=codeline[0].strip() if codeline else None,
            lineno=lineno,
            linenostart=start,
            lines=lines,
            function=function,
            urls=urls,
            variables=extract_variables(frame.f_locals, lines),
        ))
        if suppress_inner and frame is bug_in_frame:
            break
    return frames

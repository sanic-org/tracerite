import contextlib
import sys

from . import trace
from .html import html_traceback
from .logging import logger
from .tty import tty_traceback


def _can_display_html():
    # Spyder runs IPython ZMQInteractiveShell but lacks HTML support. Using
    # argv seems like the most portable way to autodetect HTML capability.
    #
    # "ipykernel_launcher.py" in Jupyter Notebook/Lab
    # "ipykernel/__main__.py" in Azure Notebooks
    # "colab_kernel_launcher.py" in Google Colab
    return any(name in sys.argv[0] for name in ["ipykernel", "colab_kernel_launcher"])


def load_ipython_extension(ipython):
    trace.ipython = ipython

    if _can_display_html():
        # Use HTML output for Jupyter notebooks
        def showtraceback(*args, **kwargs):
            try:
                from IPython.display import display

                # TraceRite HTML output
                display(
                    html_traceback(skip_until="<ipython-input-", replace_previous=True)
                )
            except Exception:
                # Fall back to built-in showtraceback
                ipython.__class__.showtraceback(ipython, *args, **kwargs)

        def showsyntaxerror(*args, **kwargs):
            try:
                from IPython.display import display

                # TraceRite HTML output for syntax errors
                display(
                    html_traceback(skip_until="<ipython-input-", replace_previous=True)
                )
            except Exception:
                # Fall back to built-in showsyntaxerror
                ipython.__class__.showsyntaxerror(ipython, *args, **kwargs)
    else:
        # Use TTY output for terminal-based IPython (Spyder, plain ipython, etc.)
        def showtraceback(*args, **kwargs):
            try:
                tty_traceback(skip_until="<ipython-input-")
            except Exception:
                # Fall back to built-in showtraceback
                ipython.__class__.showtraceback(ipython, *args, **kwargs)

        def showsyntaxerror(*args, **kwargs):
            try:
                tty_traceback(skip_until="<ipython-input-")
            except Exception:
                # Fall back to built-in showsyntaxerror
                ipython.__class__.showsyntaxerror(ipython, *args, **kwargs)

    # Install the handlers
    try:
        ipython.showtraceback = showtraceback
        ipython.showsyntaxerror = showsyntaxerror
    except Exception:
        logger.error("Unable to load Tracerite (please report a bug!)")
        raise


def unload_ipython_extension(ipython):
    with contextlib.suppress(AttributeError):
        del ipython.showtraceback
    with contextlib.suppress(AttributeError):
        del ipython.showsyntaxerror
    trace.ipython = None

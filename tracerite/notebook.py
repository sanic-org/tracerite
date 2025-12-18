import contextlib
import sys

from . import trace
from .html import html_traceback
from .logging import logger


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

    def showtraceback(*args, **kwargs):
        try:
            from IPython.display import display

            # TraceRite HTML output
            display(html_traceback(skip_until="<ipython-input-", replace_previous=True))
        except Exception:
            # Fall back to built-in showtraceback
            ipython.__class__.showtraceback(ipython, *args, **kwargs)

    def showsyntaxerror(*args, **kwargs):
        try:
            from IPython.display import display

            # TraceRite HTML output for syntax errors
            display(html_traceback(skip_until="<ipython-input-", replace_previous=True))
        except Exception:
            # Fall back to built-in showsyntaxerror
            ipython.__class__.showsyntaxerror(ipython, *args, **kwargs)

    # Install the handlers
    try:
        if _can_display_html():
            ipython.showtraceback = showtraceback
            ipython.showsyntaxerror = showsyntaxerror
        else:
            logger.warning("TraceRite not loaded: No HTML notebook detected")
    except Exception:
        logger.error("Unable to load Tracerite (please report a bug!)")
        raise


def unload_ipython_extension(ipython):
    with contextlib.suppress(AttributeError):
        del ipython.showtraceback
    with contextlib.suppress(AttributeError):
        del ipython.showsyntaxerror
    trace.ipython = None

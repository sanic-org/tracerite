import sys
from . import trace
from .html import html_traceback
from .logging import logger


def _can_display_html():
    # Spyder runs IPython ZMQInteractiveShell but lacks HTML support. Using
    # argv seems like the most portable way to autodetect HTML capability.
    #
    # "ipykernel_launcher.py" in Jupyter Notebook/Lab, Google Colab
    # "ipykernel/__main__.py" in Azure Notebooks
    return "ipykernel" in sys.argv[0]


def load_ipython_extension(ipython):
    trace.ipython = ipython
    def showtraceback(*args, **kwargs):
        try:
            from IPython.display import display
            # TraceRite HTML output
            display(html_traceback(skip_until="<ipython-input-"))
        except Exception:
            # Fall back to built-in showtraceback
            ipython.__class__.showtraceback(ipython, *args, **kwargs)

    # Install the handler
    try:
        if _can_display_html():
            ipython.showtraceback = showtraceback
        else:
            logger.warning("TraceRite not loaded: No HTML notebook detected")
    except Exception:
        logger.error("Unable to load Tracerite (please report a bug!)")
        raise


def unload_ipython_extension(ipython):
    try:
        del ipython.showtraceback
    except AttributeError:
        pass
    trace.ipython = None

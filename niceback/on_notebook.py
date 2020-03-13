"""Importing this module on a Jupyter Notebook / Jupyterlab installs the handler."""

import logging

import IPython

from niceback import html_traceback


def showtraceback(*args, **kwargs):
    try:
        IPython.display.display(html_traceback())
    except Exception:
        original_showtraceback(*args, **kwargs)

def can_display_html():
    try:
        return get_ipython().__class__.__name__ != 'TerminalInteractiveShell'
    except NameError:
        return False

try:
    if can_display_html():
        if "original_showtraceback" not in globals():
            original_showtraceback = IPython.core.interactiveshell.InteractiveShell.showtraceback
        IPython.core.interactiveshell.InteractiveShell.showtraceback = showtraceback
    else:
        logging.warning("Niceback not loaded: No HTML notebook detected")
except Exception:
    logging.error("Unable to load Niceback (please report a bug!)")
    raise

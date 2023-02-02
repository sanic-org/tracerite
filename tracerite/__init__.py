import pkg_resources

from .html import html_traceback
from .notebook import load_ipython_extension, unload_ipython_extension
from .trace import extract_chain

__all__ = ["html_traceback", "extract_chain"]
__version__ = pkg_resources.require(__name__)[0].version

del pkg_resources

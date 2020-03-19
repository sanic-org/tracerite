import pkg_resources

from niceback.html import html_traceback
from niceback.trace import extract_chain

__all__ = ["html_traceback", "extract_chain"]
__version__ = pkg_resources.require(__name__)[0].version

del pkg_resources

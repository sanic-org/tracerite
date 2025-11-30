"""
Helper module for testing __tracebackhide__ in globals.

This module has __tracebackhide__ = True at module level, which means
all functions in this module should be hidden from tracebacks.
"""

__tracebackhide__ = True


def internal_helper_function(callback):
    """
    Internal implementation function that should be hidden from tracebacks.

    This function has __tracebackhide__ in its f_globals (module level),
    so when it calls user code that crashes, this frame should be omitted.
    """
    return callback()


def nested_internal_helper(callback):
    """Another internal helper that should be hidden."""
    return internal_helper_function(callback)

# type: ignore
# ruff: noqa


def binomial_operator():
    a = 1 + "b"  # Comment


def multiline_marking():
    # fmt: off
    _ = (
        1
        +
        "a"
    ) # fmt: on


def multiline_marking_comment():
    # fmt: off
    _ = (1
        +   # Comment
        "a")
    # fmt: on


def max_type_error_case():
    max("a", 1)


def reraise_suppressed_context():
    try:
        wrongname
    except NameError:
        raise RuntimeError("foo") from None


def reraise_context():
    try:
        wrongname
    except NameError:
        raise RuntimeError("foo")


def chained_from_and_without():
    try:
        wrongname
    except NameError as e:
        try:
            raise RuntimeError("foo") from e
        except Exception as e2:
            raise AttributeError("bar")


def unrelated_error_in_except():
    try:
        wrongname
    except NameError:
        0 / 0


def unrelated_error_in_finally():
    try:
        wrongname
    finally:
        0 / 0


def _helper_raise():
    """Helper that raises an error."""
    raise RuntimeError("from helper")


def error_via_call_in_except():
    """Exception raised from a function call within an except block."""
    try:
        wrongname
    except NameError:
        _helper_raise()  # Call from except block


def error_in_stdlib_mimetypes():
    import mimetypes

    mimetypes.guess_type(123)  # It expects a string, not an int


def function_with_many_locals():
    a = 1
    b = 2
    c = 3
    d = 4
    e = 5
    f = 6
    g = 7
    h = 8
    i = 9
    j = 10
    k = "long string value that takes up space"
    l = [1, 2, 3, 4, 5]
    m = {"key": "value", "another": "item"}
    n = None
    o = True
    p = False
    q = 3.14
    r = complex(1, 2)
    s = set([1, 2, 3])
    t = frozenset([4, 5, 6])
    u = tuple((7, 8, 9))
    v = range(10)
    w = slice(1, 10, 2)
    x = Exception("test")
    y = ValueError("another test")
    z = TypeError("yet another")
    raise RuntimeError("many locals test")


def function_with_many_locals_chained():
    a = 1
    b = 2
    c = 3
    d = 4
    e = 5
    f = 6
    g = 7
    h = 8
    i = 9
    j = 10
    k = "long string value that takes up space"
    l = [1, 2, 3, 4, 5]
    m = {"key": "value", "another": "item"}
    n = None
    o = True
    p = False
    q = 3.14
    r = complex(1, 2)
    s = set([1, 2, 3])
    t = frozenset([4, 5, 6])
    u = tuple((7, 8, 9))
    v = range(10)
    w = slice(1, 10, 2)
    x = Exception("test")
    y = ValueError("another test")
    z = TypeError("yet another")
    try:
        raise ValueError("inner")
    except ValueError:
        raise RuntimeError("many locals chained test") from None


def function_with_single_local():
    """Function with exactly one local variable for single-line inspector."""
    x = 42
    raise RuntimeError("single local test")


def function_with_few_locals():
    """Function with a few local variables for multi-line inspector."""
    a = 1
    b = 2
    c = 3
    raise RuntimeError("few locals test")


# fmt: off
def oneliner_with_args(a, b): 1 / 0
# fmt: on


def multiline_before_error(a, b):
    """Function where error is several lines down, causing inspector to start at error."""
    x = 1
    y = 2
    z = 3
    w = 4
    v = 5
    # Error on last line - inspector should start here, arrow on first inspector line
    return a / b


def comprehension_error():
    """Error inside a list comprehension."""
    return [x / 0 for x in [1, 2, 3]]


def exception_group_with_frames():
    """Create an ExceptionGroup where subexceptions have proper tracebacks.

    This is needed because inline exceptions like ExceptionGroup("test", [ValueError("a")])
    don't have tracebacks attached to them.
    """
    import sys

    if sys.version_info < (3, 11):
        raise RuntimeError("ExceptionGroup requires Python 3.11+")

    def raise_value_error():
        raise ValueError("value error with frames")

    def raise_type_error():
        raise TypeError("type error with frames")

    # Capture exceptions with full tracebacks
    exceptions = []
    try:
        raise_value_error()
    except Exception as e:
        exceptions.append(e)
    try:
        raise_type_error()
    except Exception as e:
        exceptions.append(e)

    raise ExceptionGroup("multiple errors", exceptions)  # noqa: F821


def long_arguments_error():
    """Error in a function call with very long arguments (>20 chars).

    This triggers the em collapse code path in both HTML and TTY rendering.
    """
    max("this_is_a_very_long_string_argument", 123)


def nested_exception_group():
    """Create a nested ExceptionGroup for testing.

    This creates an ExceptionGroup containing another ExceptionGroup.
    """
    import sys

    if sys.version_info < (3, 11):
        raise RuntimeError("ExceptionGroup requires Python 3.11+")

    def raise_inner():
        raise ValueError("inner error")

    def raise_outer():
        raise TypeError("outer error")

    inner_exceptions = []
    try:
        raise_inner()
    except Exception as e:
        inner_exceptions.append(e)

    outer_exceptions = []
    try:
        raise_outer()
    except Exception as e:
        outer_exceptions.append(e)
    outer_exceptions.append(ExceptionGroup("inner group", inner_exceptions))  # noqa: F821

    raise ExceptionGroup("outer group", outer_exceptions)  # noqa: F821


def multiline_exception_message():
    """Raise an exception with a multiline message."""
    raise ValueError(
        "First line of error\nSecond line with details\nThird line with more info"
    )


def empty_second_line_exception():
    """Raise an exception with an empty line in the middle."""
    raise ValueError("First line\n\nThird line after empty")


def base_exception_error():
    """Raise a BaseException (KeyboardInterrupt) for testing stop frame relevance."""
    raise KeyboardInterrupt("User cancelled")


def trailing_newline_message():
    """Raise an exception with only trailing newline (empty rest after split)."""
    raise ValueError("Single line with trailing\n")


def long_args_with_suffix():
    """Error in function call with long args followed by more code.

    This tests the collapse em code path where there's code after the em.
    """
    # Call max with long arg, then access something on result
    result = max("this_is_a_very_long_string_argument", 123)
    return result + 1  # This line won't execute but shows pattern

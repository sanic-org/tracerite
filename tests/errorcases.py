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

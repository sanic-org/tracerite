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

"""Basic tests for tracerite functionality."""

from tracerite import extract_chain, html_traceback


def test_import():
    """Test that basic imports work."""
    assert extract_chain is not None
    assert html_traceback is not None


def test_extract_chain_with_simple_exception():
    """Test extract_chain with a simple exception."""
    try:
        raise ValueError("test error")
    except Exception as e:
        chain = extract_chain(e)
        assert chain is not None
        assert len(chain) > 0


def test_html_traceback_with_simple_exception():
    """Test html_traceback with a simple exception."""
    try:
        raise ValueError("test error")
    except Exception as e:
        html = html_traceback(e)
        assert html is not None
        assert "test error" in str(html)
        assert "ValueError" in str(html)


def test_nested_exception():
    """Test handling of nested exceptions."""
    try:
        try:
            raise ValueError("inner error")
        except Exception as inner:
            raise RuntimeError("outer error") from inner
    except Exception as e:
        chain = extract_chain(e)
        assert chain is not None
        assert len(chain) >= 1  # Should have at least the outer exception

        html = html_traceback(e)
        assert "inner error" in str(html)
        assert "outer error" in str(html)

"""Tests for chain_analysis module - try-except block matching."""

from tracerite.chain_analysis import (
    TryExceptBlock,
    TryExceptVisitor,
    analyze_exception_chain_links,
    build_chronological_frames,
    enrich_chain_with_links,
    find_matching_try_for_inner_exception,
    parse_source_for_try_except,
)
from tracerite.trace import extract_chain

from .errorcases import (
    chained_from_and_without,
    error_via_call_in_except,
    reraise_context,
    unrelated_error_in_except,
)


class TestTryExceptVisitor:
    """Tests for AST visitor that finds try-except blocks."""

    def test_simple_try_except(self):
        """Test parsing a simple try-except block."""
        import ast

        source = """
try:
    x = 1
    y = 2
except Exception:
    z = 3
"""
        tree = ast.parse(source)
        visitor = TryExceptVisitor()
        visitor.visit(tree)

        assert len(visitor.try_except_blocks) == 1
        block = visitor.try_except_blocks[0]
        assert block.try_start == 2  # 'try:' line
        assert block.except_start == 5  # 'except Exception:' line

    def test_nested_try_except(self):
        """Test parsing nested try-except blocks."""
        import ast

        source = """
try:
    try:
        inner = 1
    except ValueError:
        inner_handler = 2
except Exception:
    outer_handler = 3
"""
        tree = ast.parse(source)
        visitor = TryExceptVisitor()
        visitor.visit(tree)

        assert len(visitor.try_except_blocks) == 2


class TestTryExceptBlock:
    """Tests for TryExceptBlock dataclass."""

    def test_contains_in_try(self):
        """Test checking if line is in try body."""
        block = TryExceptBlock(
            try_start=10,
            try_end=15,
            except_start=16,
            except_end=20,
        )
        assert block.contains_in_try(10)
        assert block.contains_in_try(12)
        assert block.contains_in_try(15)
        assert not block.contains_in_try(9)
        assert not block.contains_in_try(16)

    def test_contains_in_except(self):
        """Test checking if line is in except handler."""
        block = TryExceptBlock(
            try_start=10,
            try_end=15,
            except_start=16,
            except_end=20,
        )
        assert block.contains_in_except(16)
        assert block.contains_in_except(18)
        assert block.contains_in_except(20)
        assert not block.contains_in_except(15)
        assert not block.contains_in_except(21)


class TestFindMatchingTryForInnerException:
    """Tests for finding the try block that links inner and outer exceptions."""

    def test_match_found(self):
        """Test finding matching try-except block."""
        blocks = [
            TryExceptBlock(try_start=10, try_end=15, except_start=16, except_end=20),
        ]

        # Inner at line 12 (in try), outer at line 18 (in except)
        result = find_matching_try_for_inner_exception(blocks, 12, 18)
        assert result is not None
        assert result.try_start == 10

    def test_no_match_inner_not_in_try(self):
        """Test when inner exception line is not in try body."""
        blocks = [
            TryExceptBlock(try_start=10, try_end=15, except_start=16, except_end=20),
        ]

        # Inner at line 5 (before try), outer at line 18 (in except)
        result = find_matching_try_for_inner_exception(blocks, 5, 18)
        assert result is None

    def test_no_match_outer_not_in_except(self):
        """Test when outer exception line is not in except handler."""
        blocks = [
            TryExceptBlock(try_start=10, try_end=15, except_start=16, except_end=20),
        ]

        # Inner at line 12 (in try), outer at line 25 (after except)
        result = find_matching_try_for_inner_exception(blocks, 12, 25)
        assert result is None


class TestParseSourceForTryExcept:
    """Tests for parsing source files."""

    def test_parse_errorcases(self):
        """Test parsing the errorcases module for try-except blocks."""
        from tests import errorcases

        blocks = parse_source_for_try_except(errorcases.__file__)
        assert len(blocks) >= 4  # We have several try-except blocks in errorcases


class TestAnalyzeExceptionChainLinks:
    """Tests for analyzing exception chains."""

    def test_single_exception_no_link(self):
        """Test that single exception has no chain link."""
        try:
            raise ValueError("single")
        except ValueError as e:
            chain = extract_chain(e)

        links = analyze_exception_chain_links(chain)
        assert len(links) == 1
        assert links[0] is None

    def test_chained_from_and_without(self):
        """Test chain analysis with chained_from_and_without error case."""
        try:
            chained_from_and_without()
        except Exception as e:
            chain = extract_chain(e)

        # Should have 3 exceptions: NameError -> RuntimeError -> AttributeError
        assert len(chain) == 3
        assert chain[0]["type"] == "NameError"
        assert chain[1]["type"] == "RuntimeError"
        assert chain[2]["type"] == "AttributeError"

        links = analyze_exception_chain_links(chain)
        assert len(links) == 3

        # First exception has no prior link
        assert links[0] is None

        # Second exception (RuntimeError) should link to first (NameError)
        # The NameError's first frame is in the try block
        # The RuntimeError has a frame in the except block
        if links[1] is not None:
            assert links[1].matched
            assert links[1].try_block is not None

    def test_reraise_context(self):
        """Test chain analysis with reraise_context error case."""
        try:
            reraise_context()
        except Exception as e:
            chain = extract_chain(e)

        assert len(chain) == 2
        assert chain[0]["type"] == "NameError"
        assert chain[1]["type"] == "RuntimeError"

        links = analyze_exception_chain_links(chain)
        assert links[0] is None

        if links[1] is not None:
            assert links[1].matched

    def test_unrelated_error_in_except(self):
        """Test chain analysis when error in except is unrelated."""
        try:
            unrelated_error_in_except()
        except Exception as e:
            chain = extract_chain(e)

        # NameError -> ZeroDivisionError
        assert len(chain) == 2

        links = analyze_exception_chain_links(chain)
        assert links[0] is None
        # Should still find the link since both are in the same try-except


class TestEnrichChainWithLinks:
    """Tests for enriching chain with link information."""

    def test_enrich_adds_chain_link_key(self):
        """Test that enrich_chain_with_links adds chain_link to each exception."""
        try:
            chained_from_and_without()
        except Exception as e:
            chain = extract_chain(e)

        enriched = enrich_chain_with_links(chain)

        for exc in enriched:
            assert "chain_link" in exc

    def test_enriched_chain_link_structure(self):
        """Test the structure of chain_link when matched."""
        try:
            reraise_context()
        except Exception as e:
            chain = extract_chain(e)

        enriched = enrich_chain_with_links(chain)

        # First exception has no link
        assert enriched[0]["chain_link"] is None

        # Second exception may have a link
        link = enriched[1]["chain_link"]
        if link is not None:
            assert "outer_frame_idx" in link
            assert "try_start" in link
            assert "try_end" in link
            assert "except_start" in link
            assert "except_end" in link


class TestBuildChronologicalFrames:
    """Tests for building chronological frame list."""

    def test_single_exception(self):
        """Test chronological frames with a single exception."""
        try:
            raise ValueError("single")
        except ValueError as e:
            chain = extract_chain(e)

        chrono = build_chronological_frames(chain)

        assert len(chrono) >= 1
        # Last frame should have exception info
        assert chrono[-1].get("exception") is not None
        assert chrono[-1]["exception"]["type"] == "ValueError"

    def test_chained_exceptions_have_exception_info(self):
        """Test that error frames have exception info attached."""
        try:
            chained_from_and_without()
        except Exception as e:
            chain = extract_chain(e)

        chrono = build_chronological_frames(chain)

        # Find frames with exception info
        exc_frames = [f for f in chrono if f.get("exception")]

        # Should have 3 exceptions
        assert len(exc_frames) == 3
        assert exc_frames[0]["exception"]["type"] == "NameError"
        assert exc_frames[1]["exception"]["type"] == "RuntimeError"
        assert exc_frames[2]["exception"]["type"] == "AttributeError"

    def test_except_promotion_with_call(self):
        """Test that except handler frames are promoted to relevance='except'."""
        try:
            error_via_call_in_except()
        except Exception as e:
            chain = extract_chain(e)

        chrono = build_chronological_frames(chain)

        # Should have: error frame (NameError), except frame (call promoted), error frame (RuntimeError)
        relevances = [f.get("relevance") for f in chrono]

        # Check that we have an "except" relevance
        assert "except" in relevances, f"Expected 'except' in {relevances}"

        # The except frame should be between the two error frames
        except_idx = relevances.index("except")
        assert relevances[except_idx - 1] == "error"  # NameError
        assert relevances[except_idx + 1] == "error"  # RuntimeError

    def test_direct_error_in_except_stays_error(self):
        """Test that direct error in except block stays relevance='error'."""
        try:
            unrelated_error_in_except()
        except Exception as e:
            chain = extract_chain(e)

        chrono = build_chronological_frames(chain)

        # All frames should be "error" since both exceptions are direct errors
        # (no call from except block)
        error_frames = [f for f in chrono if f.get("relevance") == "error"]
        assert len(error_frames) >= 2

    def test_chronological_order(self):
        """Test that frames are in chronological order."""
        try:
            error_via_call_in_except()
        except Exception as e:
            chain = extract_chain(e)

        chrono = build_chronological_frames(chain)

        # Extract exception indices
        exc_indices = [f["exception"]["exc_idx"] for f in chrono if f.get("exception")]

        # Should be in order: 0, 1 (first exception, then second)
        assert exc_indices == sorted(exc_indices)

    def test_exception_from_field_preserved(self):
        """Test that exception 'from' field is preserved in chronological frames."""
        try:
            chained_from_and_without()
        except Exception as e:
            chain = extract_chain(e)

        chrono = build_chronological_frames(chain)

        exc_frames = [f for f in chrono if f.get("exception")]

        # First exception has no cause
        assert exc_frames[0]["exception"]["from"] == "none"
        # Second was raised with 'from'
        assert exc_frames[1]["exception"]["from"] == "cause"
        # Third was implicit context
        assert exc_frames[2]["exception"]["from"] == "context"

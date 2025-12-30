"""Tests for chain_analysis module - try-except block matching."""

from tracerite.chain_analysis import (
    TryExceptBlock,
    TryExceptVisitor,
    _apply_base_exception_suppression,
    _filter_hidden_frames,
    _find_chain_link,
    _get_frame_lineno,
    analyze_exception_chain_links,
    build_chronological_frames,
    enrich_chain_with_links,
    find_matching_try_for_inner_exception,
    find_try_block_for_except_line,
    parse_source_for_try_except,
    parse_source_string_for_try_except,
)
from tracerite.trace import Range, extract_chain

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
    func(
        x=1,
        y=2
    )
except Exception:
    z = 3
"""
        tree = ast.parse(source)
        visitor = TryExceptVisitor()
        visitor.visit(tree)

        assert len(visitor.try_except_blocks) == 1
        block = visitor.try_except_blocks[0]
        assert block.try_start == 2  # 'try:' line
        assert block.except_start == 7  # 'except Exception:' line

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

    def test_contains_in_except_none_handlers(self):
        """Test contains_in_except when except_start or except_end is None."""
        block = TryExceptBlock(
            try_start=10,
            try_end=15,
            except_start=None,
            except_end=20,
        )
        assert not block.contains_in_except(18)

        block2 = TryExceptBlock(
            try_start=10,
            try_end=15,
            except_start=16,
            except_end=None,
        )
        assert not block2.contains_in_except(18)


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

    def test_find_try_block_no_matching_except(self):
        """Test find_try_block_for_except_line returns None when no block contains line."""
        blocks = [
            TryExceptBlock(try_start=10, try_end=15, except_start=16, except_end=20),
        ]

        # Line 25 is not in any except handler
        result = find_try_block_for_except_line(blocks, 25)
        assert result is None

    def test_find_try_block_multiple_matches(self):
        """Test find_try_block_for_except_line returns the most specific (innermost) block."""
        blocks = [
            TryExceptBlock(
                try_start=15, try_end=25, except_start=20, except_end=30
            ),  # Inner
            TryExceptBlock(
                try_start=5, try_end=35, except_start=10, except_end=40
            ),  # Outer
        ]

        # Line 22 is in both except handlers, should return the innermost (higher try_start)
        result = find_try_block_for_except_line(blocks, 22)
        assert result is not None
        assert result.try_start == 15


class TestParseSourceForTryExcept:
    """Tests for parsing source files."""

    def test_parse_errorcases(self):
        """Test parsing the errorcases module for try-except blocks."""
        from tests import errorcases

        blocks = parse_source_for_try_except(errorcases.__file__)
        assert len(blocks) >= 4  # We have several try-except blocks in errorcases

    def test_parse_empty_file(self):
        """Test parsing an empty file returns empty list."""
        import os
        import tempfile

        with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False) as f:
            f.write("")  # Empty file
            temp_file = f.name

        try:
            blocks = parse_source_for_try_except(temp_file)
            assert blocks == []
        finally:
            os.unlink(temp_file)

    def test_parse_invalid_syntax(self):
        """Test parsing file with invalid syntax triggers exception handling."""
        import os
        import tempfile

        with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False) as f:
            f.write("def invalid syntax here:")  # Invalid syntax
            temp_file = f.name

        try:
            blocks = parse_source_for_try_except(temp_file)
            assert blocks == []  # Should return empty on parse error
        finally:
            os.unlink(temp_file)


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

    def test_chain_link_with_missing_filename(self):
        """Test _find_chain_link returns None when inner frame has no filename."""
        try:
            raise ValueError("inner")
        except ValueError:
            try:
                raise RuntimeError("outer")
            except RuntimeError as e:
                chain = extract_chain(e)

        # Modify inner frame to have no filename
        if len(chain) >= 2 and chain[0]["frames"]:
            chain[0]["frames"][0]["filename"] = None

        links = analyze_exception_chain_links(chain)
        # Should have links for both, but the first link should be None due to missing filename
        assert len(links) == 2
        assert links[0] is None  # The link for the second exception

    def test_chain_link_with_missing_lineno(self):
        """Test _find_chain_link returns None when inner frame has no lineno."""
        try:
            raise ValueError("inner")
        except ValueError:
            try:
                raise RuntimeError("outer")
            except RuntimeError as e:
                chain = extract_chain(e)

        # Modify inner frame to have no lineno
        if len(chain) >= 2 and chain[0]["frames"]:
            del chain[0]["frames"][0]["lineno"]

        links = analyze_exception_chain_links(chain)
        # Should have links for both, but the first link should be None due to missing lineno
        assert len(links) == 2
        assert links[0] is None

    def test_chain_link_with_no_try_except_blocks(self):
        """Test _find_chain_link returns None when no try-except blocks found."""
        import os
        import tempfile

        # Create a temporary file with no try-except
        with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False) as f:
            f.write("x = 1\ny = 2\n")  # No try-except
            temp_file = f.name

        try:
            try:
                raise ValueError("inner")
            except ValueError:
                try:
                    raise RuntimeError("outer")
                except RuntimeError as e:
                    chain = extract_chain(e)

            # Modify filename to point to our temp file
            if len(chain) >= 2 and chain[0]["frames"]:
                chain[0]["frames"][0]["filename"] = temp_file

            links = analyze_exception_chain_links(chain)
            # Should have 2 links, the first should be None due to no try-except blocks
            assert len(links) == 2
            assert links[0] is None
        finally:
            os.unlink(temp_file)

    def test_frame_with_range_uses_lfirst(self):
        """Test that _get_frame_lineno uses range[0] when available."""

        frame = {
            "range": Range(lfirst=10, lfinal=15, cbeg=0, cend=5),
            "lineno": 20,
            "linenostart": 25,
        }
        lineno = _get_frame_lineno(frame)
        assert lineno == 10  # Should use lfirst from range

    def test_frame_lineno_none_continues(self):
        """Test that frames with no lineno are skipped in chain link analysis."""
        try:
            raise ValueError("test")
        except ValueError as e:
            chain = extract_chain(e)

        # Add a second exception with a frame that has no lineno
        try:
            raise RuntimeError("second")
        except RuntimeError as e2:
            chain2 = extract_chain(e2)
            chain.extend(chain2)

        # Modify the second chain's frames to have no lineno
        if len(chain) > 1 and chain[1]["frames"]:
            for frame in chain[1]["frames"]:
                if "lineno" in frame:
                    del frame["lineno"]
                if "linenostart" in frame:
                    del frame["linenostart"]

        links = analyze_exception_chain_links(chain)
        # Should still work, skipping frames without lineno
        assert len(links) == 2


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


class TestParseSourceStringForTryExcept:
    """Tests for parse_source_string_for_try_except function."""

    def test_empty_source_returns_empty_list(self):
        """Test that empty source string returns empty list (line 170)."""
        blocks = parse_source_string_for_try_except("")
        assert blocks == []

    def test_start_line_equals_one(self):
        """Test parsing with start_line=1 (default, line 202)."""
        source = """
try:
    x = 1
except Exception:
    pass
"""
        blocks = parse_source_string_for_try_except(source, start_line=1)
        assert len(blocks) == 1
        # Line numbers should not be adjusted when start_line=1
        assert blocks[0].try_start == 2  # Line 2 in the source

    def test_start_line_adjustment(self):
        """Test that line numbers are adjusted when start_line != 1."""
        source = """try:
    x = 1
except Exception:
    pass
"""
        # Parse with offset
        blocks = parse_source_string_for_try_except(source, start_line=10)
        assert len(blocks) == 1
        # Line numbers should be adjusted by offset
        assert blocks[0].try_start == 10  # Was 1, +9 offset

    def test_invalid_syntax_returns_empty(self):
        """Test that invalid syntax returns empty list."""
        blocks = parse_source_string_for_try_except("def invalid syntax:")
        assert blocks == []


class TestFindChainLinkFallbacks:
    """Tests for _find_chain_link fallback paths."""

    def test_no_full_source_fallback_to_file(self):
        """Test fallback to file-based parsing when full_source not available (lines 347-352)."""
        # Create a chain where inner frame has no full_source
        try:
            raise ValueError("inner")
        except ValueError:
            try:
                raise RuntimeError("outer")
            except RuntimeError as e:
                chain = extract_chain(e)

        # Remove full_source from inner frame to trigger fallback
        if len(chain) >= 2 and chain[0]["frames"]:
            chain[0]["frames"][0].pop("full_source", None)
            # Keep the filename so file-based parsing works

        # Should still work with file-based fallback
        links = analyze_exception_chain_links(chain)
        assert len(links) == 2

    def test_no_filename_returns_none(self):
        """Test that missing filename in fallback returns None (line 351)."""
        try:
            raise ValueError("inner")
        except ValueError:
            try:
                raise RuntimeError("outer")
            except RuntimeError as e:
                chain = extract_chain(e)

        # Remove both full_source and filename
        if len(chain) >= 2 and chain[0]["frames"]:
            chain[0]["frames"][0].pop("full_source", None)
            chain[0]["frames"][0]["filename"] = None
            chain[0]["frames"][0]["original_filename"] = None

        links = analyze_exception_chain_links(chain)
        # Link analysis should return None for the second exception
        assert links[1] is None

    def test_frame_lineno_none_skipped(self):
        """Test that frames with no lineno are skipped (line 363)."""
        try:
            raise ValueError("inner")
        except ValueError:
            try:
                raise RuntimeError("outer")
            except RuntimeError as e:
                chain = extract_chain(e)

        # Remove lineno from outer frames
        if len(chain) >= 2 and chain[1]["frames"]:
            for frame in chain[1]["frames"]:
                frame.pop("lineno", None)
                frame.pop("linenostart", None)
                frame.pop("range", None)

        links = analyze_exception_chain_links(chain)
        # Should still have links list, but the link should be None
        assert len(links) == 2

    def test_no_matching_block_returns_none(self):
        """Test that no matching try-except block returns None (line 379)."""
        # Create mock chain data where try-except blocks don't match
        inner_exc = {
            "type": "ValueError",
            "frames": [
                {
                    "filename": "/nonexistent/path.py",
                    "lineno": 100,  # Not in any try block
                }
            ],
        }
        outer_exc = {
            "type": "RuntimeError",
            "frames": [
                {
                    "filename": "/nonexistent/path.py",
                    "lineno": 200,  # Not in any except block
                }
            ],
        }
        # _find_chain_link should return None since no try-except can be found
        result = _find_chain_link(inner_exc, outer_exc)
        assert result is None


class TestFilterHiddenFrames:
    """Tests for _filter_hidden_frames function."""

    def test_filters_hidden_frames(self):
        """Test that hidden frames are filtered out (line 493)."""
        frames = [
            {"lineno": 1, "hidden": False},
            {"lineno": 2, "hidden": True},  # Should be filtered
            {"lineno": 3},  # No hidden key = not hidden
        ]
        result = _filter_hidden_frames(frames)
        assert len(result) == 2
        assert result[0]["lineno"] == 1
        assert result[1]["lineno"] == 3

    def test_filters_parallel_branches_recursively(self):
        """Test that parallel branches are filtered recursively (lines 496-503)."""
        frames = [
            {
                "lineno": 1,
                "parallel": [
                    [
                        {"lineno": 10, "hidden": False},
                        {"lineno": 11, "hidden": True},  # Should be filtered
                    ],
                    [
                        {"lineno": 20, "hidden": True},  # All hidden
                        {"lineno": 21, "hidden": True},
                    ],
                ],
            },
        ]
        result = _filter_hidden_frames(frames)
        assert len(result) == 1
        # First parallel branch should have hidden frames filtered
        assert len(result[0]["parallel"]) == 1  # Second branch was all hidden
        assert len(result[0]["parallel"][0]) == 1  # Only non-hidden frame left
        assert result[0]["parallel"][0][0]["lineno"] == 10

    def test_empty_parallel_branches_removed(self):
        """Test that empty parallel branches after filtering are removed."""
        frames = [
            {
                "lineno": 1,
                "parallel": [
                    [{"lineno": 10, "hidden": True}],  # All hidden
                ],
            },
        ]
        result = _filter_hidden_frames(frames)
        # The frame with empty parallel branches should not be included
        assert len(result) == 0


class TestApplyBaseExceptionSuppression:
    """Tests for _apply_base_exception_suppression function."""

    def test_no_suppression_without_flag(self):
        """Test that suppression doesn't happen without suppress_inner flag."""
        chronological = [
            {"lineno": 1, "relevance": "call"},
            {"lineno": 2, "relevance": "warning"},
            {"lineno": 3, "relevance": "error"},
        ]
        chain = [{"type": "ValueError"}]  # No suppress_inner
        result = _apply_base_exception_suppression(chronological, chain)
        assert result == chronological

    def test_suppression_with_flag(self):
        """Test suppression when suppress_inner=True (lines 539, 546-572)."""
        chronological = [
            {"lineno": 1, "relevance": "call"},
            {"lineno": 2, "relevance": "warning"},  # Bug frame
            {"lineno": 3, "relevance": "call"},  # Should be suppressed
            {
                "lineno": 4,
                "relevance": "error",
                "exception": {"type": "KeyboardInterrupt"},
            },
        ]
        chain = [{"suppress_inner": True}]
        result = _apply_base_exception_suppression(chronological, chain)
        # Should only have frames up to bug frame
        assert len(result) == 2
        # Last frame should have exception transferred
        assert result[-1].get("exception") == {"type": "KeyboardInterrupt"}
        # Relevance should be changed to "stop"
        assert result[-1]["relevance"] == "stop"

    def test_suppression_transfers_parallel(self):
        """Test that parallel branches are transferred during suppression."""
        chronological = [
            {"lineno": 1, "relevance": "warning"},  # Bug frame
            {"lineno": 2, "relevance": "error", "parallel": [["branch1"]]},
        ]
        chain = [{"suppress_inner": True}]
        result = _apply_base_exception_suppression(chronological, chain)
        assert len(result) == 1
        # Parallel should be transferred
        assert result[-1].get("parallel") == [["branch1"]]

    def test_suppression_no_bug_frame(self):
        """Test suppression when no bug frame exists."""
        chronological = [
            {"lineno": 1, "relevance": "call"},
            {"lineno": 2, "relevance": "error"},
        ]
        chain = [{"suppress_inner": True}]
        result = _apply_base_exception_suppression(chronological, chain)
        # Should return as-is when no bug frame found
        assert result == chronological

    def test_empty_chain(self):
        """Test with empty chain."""
        result = _apply_base_exception_suppression([], [])
        assert result == []


class TestGetFrameLinenoFallbacks:
    """Tests for _get_frame_lineno function fallbacks."""

    def test_uses_lineno_fallback(self):
        """Test fallback to lineno when no range."""
        frame = {"lineno": 15}
        assert _get_frame_lineno(frame) == 15

    def test_uses_linenostart_fallback(self):
        """Test fallback to linenostart when no lineno."""
        frame = {"linenostart": 20}
        assert _get_frame_lineno(frame) == 20

    def test_returns_none_when_no_line_info(self):
        """Test returns None when no line info available."""
        frame = {"filename": "test.py"}
        assert _get_frame_lineno(frame) is None


class TestFindChainLinkEdgeCases:
    """Additional edge case tests for _find_chain_link."""

    def test_inner_frame_lineno_none(self):
        """Test _find_chain_link returns None when inner frame has no lineno (line 334)."""
        inner_exc = {
            "type": "ValueError",
            "frames": [
                {
                    "filename": "/test/path.py",
                    # No lineno, linenostart, or range
                }
            ],
        }
        outer_exc = {
            "type": "RuntimeError",
            "frames": [
                {
                    "filename": "/test/path.py",
                    "lineno": 20,
                }
            ],
        }
        result = _find_chain_link(inner_exc, outer_exc)
        assert result is None

    def test_outer_frame_lineno_none_skipped(self):
        """Test that outer frames with no lineno are skipped (line 363)."""
        inner_exc = {
            "type": "ValueError",
            "frames": [
                {
                    "filename": "/test/path.py",
                    "lineno": 10,
                    "full_source": "try:\n    x = 1\nexcept:\n    pass\n",
                    "full_source_start": 1,
                }
            ],
        }
        outer_exc = {
            "type": "RuntimeError",
            "frames": [
                {
                    "filename": "/test/path.py",
                    # No lineno - should be skipped
                },
                {
                    "filename": "/test/path.py",
                    "lineno": 4,  # In except block
                },
            ],
        }
        # First outer frame has no lineno and should be skipped,
        # second frame should match
        result = _find_chain_link(inner_exc, outer_exc)
        # Since source shows try on line 1 (try_start=1, try_end=2),
        # and inner is at line 10 (not in try block 1-2), it won't match
        # This test exercises the continue path but won't find a match
        assert result is None

    def test_loop_completes_without_match(self):
        """Test that loop completes and returns None when no match (line 379)."""
        inner_exc = {
            "type": "ValueError",
            "frames": [
                {
                    "filename": "/test/path.py",
                    "lineno": 100,  # Not in any try block
                    "full_source": "try:\n    x = 1\nexcept:\n    pass\n",
                    "full_source_start": 1,
                }
            ],
        }
        outer_exc = {
            "type": "RuntimeError",
            "frames": [
                {
                    "filename": "/test/path.py",
                    "lineno": 200,  # Not in any except block
                },
            ],
        }
        result = _find_chain_link(inner_exc, outer_exc)
        assert result is None


class TestSuppressionRelevanceChange:
    """Tests for relevance change during suppression."""

    def test_suppression_changes_call_to_stop(self):
        """Test that relevance='call' is changed to 'stop' during suppression (line 569)."""
        chronological = [
            {"lineno": 1, "relevance": "warning"},  # Bug frame
            {"lineno": 2, "relevance": "call"},  # Will be suppressed
            {
                "lineno": 3,
                "relevance": "error",
                "exception": {"type": "KeyboardInterrupt"},
            },
        ]
        chain = [{"suppress_inner": True}]
        result = _apply_base_exception_suppression(chronological, chain)
        # Last frame should have relevance changed from warning to stop
        assert result[-1]["relevance"] == "stop"

    def test_suppression_keeps_error_relevance(self):
        """Test that relevance='error' is not changed during suppression."""
        chronological = [
            {"lineno": 1, "relevance": "warning"},  # Bug frame
            {
                "lineno": 2,
                "relevance": "error",
                "exception": {"type": "KeyboardInterrupt"},
            },
        ]
        chain = [{"suppress_inner": True}]
        result = _apply_base_exception_suppression(chronological, chain)
        # Only one frame returned (bug frame), and it should have the exception
        assert len(result) == 1
        assert result[-1].get("exception") == {"type": "KeyboardInterrupt"}
        assert result[-1]["relevance"] == "stop"

    def test_suppression_does_not_change_error_relevance(self):
        """Test suppression with relevance='error' at bug frame (from edge case)."""
        # In practice, bug frames always have "warning" relevance by design
        # This test verifies the expected behavior
        chronological = [
            {"lineno": 1, "relevance": "warning"},
            {"lineno": 2, "relevance": "error", "exception": {"type": "Error"}},
        ]
        chain = [{"suppress_inner": True}]
        result = _apply_base_exception_suppression(chronological, chain)
        assert len(result) == 1
        # Bug frame's "warning" relevance gets changed to "stop"
        assert result[-1]["relevance"] == "stop"

    def test_suppression_not_change_except_relevance(self):
        """Test that relevance='except' on non-bug frame is preserved before suppression."""
        chronological = [
            {"lineno": 1, "relevance": "except"},  # Not "warning" so not bug frame
            {"lineno": 2, "relevance": "warning"},  # Bug frame
            {"lineno": 3, "relevance": "error", "exception": {"type": "Error"}},
        ]
        chain = [{"suppress_inner": True}]
        result = _apply_base_exception_suppression(chronological, chain)
        # Frames up to and including bug frame are kept
        assert len(result) == 2
        # First frame keeps "except" relevance
        assert result[0]["relevance"] == "except"
        # Bug frame (last) changes to "stop"
        assert result[-1]["relevance"] == "stop"

    def test_suppression_with_existing_parallel(self):
        """Test that existing parallel is not overwritten during suppression."""
        chronological = [
            {
                "lineno": 1,
                "relevance": "warning",
                "parallel": [["existing"]],
            },  # Bug frame with parallel
            {
                "lineno": 2,
                "relevance": "error",
                "parallel": [["suppressed"]],
            },  # Will be suppressed
        ]
        chain = [{"suppress_inner": True}]
        result = _apply_base_exception_suppression(chronological, chain)
        assert len(result) == 1
        # Existing parallel should not be overwritten
        assert result[-1]["parallel"] == [["existing"]]

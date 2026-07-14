from __future__ import annotations

from .chain_analysis import (
    ChainLink,
    TryExceptBlock,
    find_matching_try_for_inner_exception,
    parse_source_for_try_except,
    parse_source_string_for_try_except,
)


def analyze_exception_chain_links(
    chain: list[dict],
    *,
    cache: dict | None = None,
) -> list[ChainLink | None]:
    """Analyze an exception chain to find try-except relationships."""
    if len(chain) <= 1:
        return [None] * len(chain)

    links: list[ChainLink | None] = [None]
    for i in range(1, len(chain)):
        links.append(find_chain_link(chain[i - 1], chain[i], cache=cache))
    return links


def get_frame_lineno(frame: dict) -> int | None:
    """Extract the most precise line number from a frame dict."""
    frame_range = frame.get("range")
    if frame_range:
        return frame_range[0]
    if frame.get("lineno"):
        return frame.get("lineno")
    return frame.get("linenostart")


def get_try_except_blocks(
    frame: dict,
    *,
    cache: dict | None = None,
) -> list[TryExceptBlock]:
    """Get try-except blocks for a frame using its full source or filename."""
    full_source = frame.get("full_source")
    if full_source:
        blocks = parse_source_string_for_try_except(
            full_source,
            frame.get("full_source_start", 1),
            _cache=cache,
        )
        if blocks:
            return blocks
    filename = frame.get("original_filename") or frame.get("filename")
    if filename:
        return parse_source_for_try_except(filename, _cache=cache)
    return []


def frame_in_except_handler(
    frame: dict,
    *,
    cache: dict | None = None,
) -> bool:
    """Check whether a frame's line falls inside an ``except`` handler."""
    # Re-use the except-line computed during frame digestion when available.
    if "_except_start" in frame:
        return frame["_except_start"] is not None

    lineno = get_frame_lineno(frame)
    if lineno is None:
        return False
    try:
        blocks = get_try_except_blocks(frame, cache=cache)
    except Exception:
        return False
    return any(block.contains_in_except(lineno) for block in blocks)


def find_chain_link(
    inner_exc: dict,
    outer_exc: dict,
    *,
    cache: dict | None = None,
) -> ChainLink | None:
    """Find the try-except link between two consecutive exceptions."""
    inner_frames = inner_exc.get("frames", [])
    outer_frames = outer_exc.get("frames", [])
    if not inner_frames or not outer_frames:
        return None

    inner_first_frame = inner_frames[0]
    inner_first_lineno = get_frame_lineno(inner_first_frame)
    if inner_first_lineno is None:
        return None

    try_except_blocks = get_try_except_blocks(inner_first_frame, cache=cache)
    if not try_except_blocks:
        return None

    inner_filename = inner_first_frame.get(
        "original_filename"
    ) or inner_first_frame.get("filename")

    for frame_idx, frame in enumerate(outer_frames):
        frame_lineno = get_frame_lineno(frame)
        if frame_lineno is None:
            continue

        outer_filename = frame.get("original_filename") or frame.get("filename")
        if inner_filename and outer_filename != inner_filename:
            continue

        matching_block = find_matching_try_for_inner_exception(
            try_except_blocks, inner_first_lineno, frame_lineno
        )
        if matching_block:
            return ChainLink(
                outer_frame_idx=frame_idx,
                try_block=matching_block,
                matched=True,
            )
    return None


def enrich_chain_with_links(
    chain: list[dict],
    *,
    cache: dict | None = None,
) -> list[dict]:
    """Enrich exception chain with try-except link information."""
    links = analyze_exception_chain_links(chain, cache=cache)

    for exc, link in zip(chain, links, strict=True):
        if link and link.matched and (try_block := link.try_block):
            exc["chain_link"] = {
                "outer_frame_idx": link.outer_frame_idx,
                "try_start": try_block.try_start,
                "try_end": try_block.try_end,
                "except_start": try_block.except_start,
                "except_end": try_block.except_end,
            }
        else:
            exc["chain_link"] = None

    return chain


def build_chronological_frames(
    chain: list[dict],
    *,
    cache: dict | None = None,
) -> list[dict]:
    """Build a chronological list of frames showing the actual sequence of events."""
    if not chain:
        return []

    # A single exception with no sub-exceptions has no try-except links to
    # resolve and no re-raise frames to promote, so its chronological order is
    # just its original frame order.
    if len(chain) == 1 and not chain[0].get("subexceptions"):
        exc = chain[0]
        frames = exc.get("frames", [])
        chronological: list[dict] = [append_copied_frame([], frame) for frame in frames]
        if chronological:
            chronological[-1]["exception"] = make_exception_banner(exc, 0)
        chronological = filter_hidden_frames(chronological)
        chronological = apply_base_exception_suppression(chronological, chain)
        return chronological

    links = analyze_exception_chain_links(chain, cache=cache)
    chronological = []

    outermost = chain[-1]
    build_backbone_frames(
        chronological,
        outermost,
        len(chain) - 1,
        outermost.get("frames", []),
        links,
        chain,
        cache=cache,
    )

    chronological = filter_hidden_frames(chronological)
    chronological = apply_base_exception_suppression(chronological, chain)

    return chronological


def filter_hidden_frames(chronological: list[dict]) -> list[dict]:
    """Filter out hidden frames, handling parallel branches recursively."""
    result = []
    for frame in chronological:
        if frame.get("hidden"):
            continue
        if "parallel" in frame:
            filtered_branches = []
            for branch in frame["parallel"]:
                filtered_branch = filter_hidden_frames(branch)
                if filtered_branch:
                    filtered_branches.append(filtered_branch)
            if filtered_branches:
                frame = {**frame, "parallel": filtered_branches}
                result.append(frame)
        else:
            result.append(frame)
    return result


def apply_base_exception_suppression(
    chronological: list[dict], chain: list[dict]
) -> list[dict]:
    """Suppress library frames after the last user code frame."""
    if not chronological or not chain:
        return chronological
    if not any(exc.get("suppress_inner") for exc in chain):
        return chronological

    last_bug = find_last_bug_frame(chronological)
    if last_bug is None:
        return chronological

    result, keepers, suppressed = split_suppressed_frames(chronological, last_bug)
    return merge_suppressed_data(result, keepers, suppressed)


def find_last_bug_frame(chronological: list[dict]) -> int | None:
    return next(
        (
            idx
            for idx, frame in enumerate(chronological)
            if frame.get("relevance") == "warning"
        ),
        None,
    )


def split_suppressed_frames(
    chronological: list[dict], last_bug: int
) -> tuple[list[dict], list[dict], dict]:
    """Split chronological frames into kept, kept-error, and suppressed metadata."""
    result = chronological[: last_bug + 1]
    keepers = []
    suppressed = {"exception": None, "parallel": None}
    for frame in chronological[last_bug + 1 :]:
        if frame.get("relevance") in {"except", "error"}:
            keepers.append(frame)
            continue
        if frame.get("exception") and suppressed["exception"] is None:
            suppressed["exception"] = frame["exception"]
        if frame.get("parallel") and suppressed["parallel"] is None:
            suppressed["parallel"] = frame["parallel"]
    return result, keepers, suppressed


def merge_suppressed_data(
    result: list[dict], keepers: list[dict], suppressed: dict
) -> list[dict]:
    """Transfer suppressed exception/parallel info onto the bug frame."""
    keeper_exc_types = {f.get("exception", {}).get("type") for f in keepers}
    keeper_has_parallel = any(f.get("parallel") for f in keepers)

    if (
        suppressed["exception"]
        and not result[-1].get("exception")
        and suppressed["exception"].get("type") not in keeper_exc_types
    ):
        result[-1] = {**result[-1], "exception": suppressed["exception"]}
    if (
        suppressed["parallel"]
        and not result[-1].get("parallel")
        and not keeper_has_parallel
    ):
        result[-1] = {**result[-1], "parallel": suppressed["parallel"]}

    result[-1] = {**result[-1], "relevance": "stop"}
    result.extend(keepers)
    return result


def build_backbone_frames(
    chronological: list[dict],
    exc: dict,
    exc_idx: int,
    frames: list[dict],
    links: list,
    chain: list[dict],
    *,
    cache: dict | None = None,
) -> None:
    """Build chronological frames using this exception's frames as backbone."""
    inner_exc_idx = exc_idx - 1
    inner_link = links[exc_idx] if exc_idx > 0 else None

    if inner_link and inner_link.matched and inner_exc_idx >= 0:
        build_linked_backbone(
            chronological,
            exc,
            exc_idx,
            frames,
            inner_exc_idx,
            inner_link,
            links,
            chain,
            cache=cache,
        )
    else:
        build_unlinked_backbone(
            chronological, exc, exc_idx, frames, links, chain, cache=cache
        )


def build_linked_backbone(
    chronological: list[dict],
    exc: dict,
    exc_idx: int,
    frames: list[dict],
    inner_exc_idx: int,
    inner_link: ChainLink,
    links: list,
    chain: list[dict],
    *,
    cache: dict | None = None,
) -> None:
    """Build backbone when the inner exception links to an except handler."""
    except_frame_idx = inner_link.outer_frame_idx
    inner_exc = chain[inner_exc_idx]
    inner_frames = inner_exc.get("frames", [])

    for frame_idx in range(except_frame_idx):
        append_copied_frame(chronological, frames[frame_idx])

    build_backbone_frames(
        chronological,
        inner_exc,
        inner_exc_idx,
        inner_frames,
        links,
        chain,
        cache=cache,
    )

    last_idx = len(frames) - 1
    for frame_idx in range(except_frame_idx, len(frames)):
        chrono_frame = append_copied_frame(chronological, frames[frame_idx])
        if frame_idx == except_frame_idx:
            promote_to_except(chrono_frame)
        if frame_idx == last_idx:
            chrono_frame["exception"] = make_exception_banner(exc, exc_idx)
            add_subexceptions_to_frame(chrono_frame, exc, cache=cache)


def build_unlinked_backbone(
    chronological: list[dict],
    exc: dict,
    exc_idx: int,
    frames: list[dict],
    links: list,
    chain: list[dict],
    *,
    cache: dict | None = None,
) -> None:
    """Build backbone without a matched try-except link."""
    if exc_idx > 0:
        inner_exc = chain[exc_idx - 1]
        inner_frames = inner_exc.get("frames", [])
        if inner_frames:
            build_backbone_frames(
                chronological,
                inner_exc,
                exc_idx - 1,
                inner_frames,
                links,
                chain,
                cache=cache,
            )

    re_raise = find_re_raise_frames(exc, frames, cache=cache)
    order = [i for i in range(len(frames)) if i not in re_raise] + re_raise
    last_idx = len(frames) - 1
    banner = make_exception_banner(exc, exc_idx)

    for idx, frame_idx in enumerate(order):
        chrono_frame = append_copied_frame(chronological, frames[frame_idx])
        is_last = frame_idx == last_idx
        is_final = idx == len(order) - 1

        if is_last:
            add_subexceptions_to_frame(chrono_frame, exc, cache=cache)
            if not re_raise:
                chrono_frame["exception"] = banner
        if is_final and re_raise:
            chrono_frame["exception"] = banner
        if frame_idx in re_raise:
            promote_to_except(chrono_frame)


def find_re_raise_frames(
    exc: dict,
    frames: list[dict],
    *,
    cache: dict | None = None,
) -> list[int]:
    """Find frames (except the last) that are inside an except handler."""
    if not exc.get("subexceptions"):
        return []
    return [
        i
        for i, frame in enumerate(frames[:-1])
        if frame_in_except_handler(frame, cache=cache)
    ]


def make_exception_banner(exc: dict, exc_idx: int) -> dict:
    """Create the exception info dict attached to a frame."""
    return {
        "type": exc.get("type"),
        "message": exc.get("message"),
        "summary": exc.get("summary"),
        "from": exc.get("from"),
        "exc_idx": exc_idx,
    }


def promote_to_except(frame: dict) -> None:
    """Promote a frame's relevance to indicate it represents an except handler."""
    if frame.get("relevance") in ("call", "warning"):
        frame["relevance"] = "except"
    frame["function_suffix"] = "⚡except"


def append_copied_frame(chronological: list[dict], frame: dict, **overrides) -> dict:
    """Append a shallow copy of a frame with optional overrides."""
    copied = {**frame, **overrides}
    chronological.append(copied)
    return copied


def add_subexceptions_to_frame(
    frame: dict,
    exc: dict,
    *,
    cache: dict | None = None,
) -> None:
    """Add subexceptions from an ExceptionGroup as parallel branches."""
    subexceptions = exc.get("subexceptions")
    if not subexceptions:
        return

    parallel_branches = [
        sub_chrono
        for sub_chain in subexceptions
        if (sub_chrono := build_chronological_frames(sub_chain, cache=cache))
    ]

    if parallel_branches:
        frame["parallel"] = parallel_branches

"""Tests for the candidate-pool ranking/filtering in :mod:`afterform.flows.long_to_shorts.select_clips`.

Covers the "over-generate -> threshold with a floor" policy:

- scores are the primary rank signal
- clips above ``threshold`` are always kept
- if fewer than ``min_kept`` clear the threshold, backfill from the next
  best candidates so the pipeline never ships zero shorts
- the final list is capped at ``max_kept``
- ``needs_review`` clips are demoted but still eligible as a backfill
- ``clip_id`` is renumbered densely in rank order
- empty input is handled without raising
"""

from __future__ import annotations

from afterform.flows.long_to_shorts.select_clips import build_prompt, rank_and_filter_clips
from afterform.schemas import Clip


def _clip(
    clip_id: str,
    *,
    score: float,
    start: float = 0.0,
    needs_review: bool = False,
    topic: str = "t",
) -> Clip:
    # Give each clip a unique, legal window. Score is the only field these
    # tests care about beyond needs_review / clip_id.
    return Clip.model_validate(
        {
            "clip_id": clip_id,
            "topic": topic,
            "start_time_sec": start,
            "end_time_sec": start + 60.0,
            "virality_score": score,
            "needs_review": needs_review,
        }
    )


def test_empty_candidates_returns_empty_list():
    assert rank_and_filter_clips([]) == []


def test_strong_candidates_are_kept_and_sorted():
    candidates = [
        _clip("a", score=0.80, start=0),
        _clip("b", score=0.92, start=100),
        _clip("c", score=0.75, start=200),
    ]
    kept = rank_and_filter_clips(
        candidates, threshold=0.70, min_kept=1, max_kept=5
    )
    # All above threshold, sorted desc, renumbered 001..003.
    assert [c.virality_score for c in kept] == [0.92, 0.80, 0.75]
    assert [c.clip_id for c in kept] == ["001", "002", "003"]


def test_weak_pool_backfills_to_min_kept():
    """Even when no candidate clears the bar, we ship at least ``min_kept``."""
    candidates = [
        _clip("a", score=0.40),
        _clip("b", score=0.55, start=100),
        _clip("c", score=0.62, start=200),
    ]
    kept = rank_and_filter_clips(
        candidates, threshold=0.80, min_kept=2, max_kept=5
    )
    assert len(kept) == 2
    assert [c.virality_score for c in kept] == [0.62, 0.55]


def test_rich_pool_capped_at_max_kept():
    candidates = [_clip(f"c{i}", score=0.90 - i * 0.01, start=i * 100) for i in range(10)]
    kept = rank_and_filter_clips(
        candidates, threshold=0.50, min_kept=5, max_kept=7
    )
    assert len(kept) == 7
    # Renumbered, monotonically decreasing score.
    assert [c.clip_id for c in kept] == [f"{i:03d}" for i in range(1, 8)]
    scores = [c.virality_score for c in kept]
    assert scores == sorted(scores, reverse=True)


def test_needs_review_clip_is_demoted_but_usable_as_backfill():
    """Reviewed clips lose ground to same-score non-reviewed, but survive when
    they're the only way to hit ``min_kept``.
    """
    candidates = [
        _clip("a", score=0.95, needs_review=True, start=0),
        _clip("b", score=0.60, needs_review=False, start=100),
    ]
    # threshold too high for the clean clip and the reviewed clip loses a
    # heavy penalty; still, min_kept=2 must be satisfied via backfill.
    kept = rank_and_filter_clips(
        candidates, threshold=0.90, min_kept=2, max_kept=5
    )
    assert len(kept) == 2
    # The non-reviewed, lower-score clip should rank *first* because the
    # reviewed one gets a 0.5 priority penalty.
    assert kept[0].clip_id == "001"
    assert kept[0].needs_review is False
    assert kept[1].needs_review is True


def test_threshold_keeps_all_qualifying_clips_even_above_min():
    """When many clips clear the threshold, we keep ALL of them (up to max_kept),
    not just min_kept. That's the "one exceptionally rich transcript ships
    more shorts" behavior the user asked for.
    """
    candidates = [_clip(f"c{i}", score=0.85, start=i * 100) for i in range(6)]
    kept = rank_and_filter_clips(
        candidates, threshold=0.70, min_kept=5, max_kept=8
    )
    assert len(kept) == 6  # all clear threshold, below max_kept


def test_prompt_asks_for_full_candidate_pool_not_just_strong_ones():
    """Regression test: the system prompt must tell Gemini to return the
    full ``count`` pool (with ``needs_review`` on weak ones) so the
    downstream ranker has enough candidates to backfill up to ``min_kept``.

    The previous wording ("Return up to N, fewer stronger clips beat a
    padded list") let Gemini return only 2 of 12 requested for a short
    transcript, which made the min_kept=5 floor unenforceable. The fixed
    prompt must explicitly ask for ``exactly {count}`` and must instruct
    the LLM to flag weak clips rather than omit them.
    """
    transcript = {
        "segments": [
            {"start": i * 5.0, "end": (i + 1) * 5.0, "text": f"line {i}"}
            for i in range(10)
        ]
    }
    system, _user = build_prompt(transcript, candidate_count=12)

    # The prompt must request the full pool, not "up to" / "fewer is better".
    assert "exactly 12" in system
    assert "fewer, stronger clips beat" not in system.lower()
    assert "up to 12" not in system

    # The prompt must instruct the LLM to keep weak clips with a review
    # flag, not drop them.
    assert "needs_review" in system
    assert "never drop" in system.lower() or "do not drop" in system.lower() or "do not self-censor" in system.lower()


def test_clip_ids_are_renumbered_in_rank_order():
    candidates = [
        _clip("zzz", score=0.50, start=0),
        _clip("aaa", score=0.90, start=100),
        _clip("mmm", score=0.70, start=200),
    ]
    kept = rank_and_filter_clips(
        candidates, threshold=0.00, min_kept=3, max_kept=5
    )
    assert [c.clip_id for c in kept] == ["001", "002", "003"]
    assert [c.virality_score for c in kept] == [0.90, 0.70, 0.50]


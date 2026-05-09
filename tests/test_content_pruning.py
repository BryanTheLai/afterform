"""Tests for Stage 2.5 content pruning.

No network. The Gemini client is always mocked. These tests cover:

- clamping (max-pct cap, min-duration floor, hook protection)
- decision -> clip mapping (``apply_prune_decisions``)
- prompt construction (clip-relative segments, hook window line)
- ``request_prune_decisions`` ships the right args to the Gemini SDK and
  parses the JSON response
- ``run_content_pruning_stage`` happy path, cache hit, cache invalidation on
  level change, and graceful failure when the LLM blows up
- ``off`` level short-circuits (no call, all trims zero)
"""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import pytest

from afterform.config import MIN_CLIP_DURATION_SEC, PipelineConfig
from afterform.flows.long_to_shorts.prune_content import (
    PRUNE_ARTIFACT_FILENAME,
    PRUNE_META_FILENAME,
    PRUNE_RAW_FILENAME,
    _build_user_message,
    _clamp_decision,
    _clips_fingerprint,
    _complete_keep_range_tail,
    _complete_trim_end_to_transcript_boundary,
    _looks_like_default_hook,
    _parse_decisions,
    _PruneDecision,
    _segments_within_clip,
    _snap_trims_to_segment_boundaries,
    apply_prune_decisions,
    request_prune_decisions,
    run_content_pruning_stage,
)
from afterform.schemas import Clip


# ---------------------------------------------------------------------------
# Fixtures / helpers
# ---------------------------------------------------------------------------


def _clip(
    clip_id: str = "001",
    *,
    start: float = 100.0,
    end: float = 190.0,
    hook_start: float | None = None,
    hook_end: float | None = None,
    topic: str = "topic",
) -> Clip:
    return Clip.model_validate(
        {
            "clip_id": clip_id,
            "topic": topic,
            "start_time_sec": start,
            "end_time_sec": end,
            "hook_start_sec": hook_start,
            "hook_end_sec": hook_end,
        }
    )


def _transcript_for(start: float, end: float, *, step: float = 5.0) -> dict:
    segs = []
    t = start - 3.0
    while t < end + 3.0:
        segs.append({"start": t, "end": t + step, "text": f"text {t:.0f}"})
        t += step
    return {"segments": segs}


@pytest.fixture
def cfg(tmp_path: Path) -> PipelineConfig:
    return PipelineConfig(
        youtube_url="https://youtu.be/abc",
        work_dir=tmp_path,
        gemini_model="gemini-test",
        prune_level="balanced",
        filled_pause_pruning=False,
    )


@pytest.fixture(autouse=True)
def _stub_audio_keep_ranges(monkeypatch):
    def _apply(
        clips,
        *,
        source_audio_path,
        transcript=None,
        filled_pause_enabled=False,
        require_filled_pause=False,
    ):
        updated = []
        diagnostics = {}
        for clip in clips:
            keep_end = max(clip.trim_start_sec, clip.duration_sec - clip.trim_end_sec)
            updated.append(
                clip.model_copy(
                    update={"keep_ranges_sec": [(clip.trim_start_sec, keep_end)]}
                )
            )
            diagnostics[clip.clip_id] = {
                "audio_backend": {
                    "speech": "stub",
                    "filled_pause": "stub" if filled_pause_enabled else "disabled",
                },
                "warnings": [],
                "require_filled_pause": require_filled_pause,
            }
        return updated, diagnostics

    monkeypatch.setattr("afterform.flows.long_to_shorts.prune_content.apply_audio_keep_ranges", _apply)


# ---------------------------------------------------------------------------
# Clamping
# ---------------------------------------------------------------------------


def test_clamp_respects_max_pct_balanced():
    clip = _clip(end=200.0)  # duration = 100s
    ts, te, stats = _clamp_decision(clip, 40.0, 10.0, level="balanced")
    assert ts + te == pytest.approx(20.0)  # 20% cap on 100s
    assert stats.max_pct_protected is True


def test_clamp_keeps_minimum_duration():
    clip = _clip(end=160.0)  # duration = 60s; balanced pct cap = 12s
    ts, te, stats = _clamp_decision(clip, 30.0, 30.0, level="aggressive")
    final = clip.duration_sec - ts - te
    assert final >= MIN_CLIP_DURATION_SEC - 1e-6
    assert stats.min_duration_protected is True


def test_clamp_preserves_hook_start_and_end():
    clip = _clip(end=200.0, hook_start=2.0, hook_end=8.0)  # duration 100s
    ts, te, _ = _clamp_decision(clip, 50.0, 50.0, level="aggressive")
    assert ts <= max(0.0, clip.hook_start_sec - 0.24)
    assert te <= max(0.0, clip.duration_sec - clip.hook_end_sec - 0.24)
    final = clip.duration_sec - ts - te
    assert final >= MIN_CLIP_DURATION_SEC - 1e-6


def test_clamp_level_off_nulls_trim():
    clip = _clip(end=200.0)
    ts, te, _ = _clamp_decision(clip, 10.0, 5.0, level="off")
    assert ts == 0.0 and te == 0.0


def test_default_hook_fingerprint_is_recognised():
    """The clip-selection prompt's 0.0-3.0s placeholder must be detected.

    This is the exact fingerprint we observed Gemini echoing verbatim for
    every clip in the Cathy Wood run, which silently disabled Stage 2.5
    start-trims until P1 was fixed.
    """
    assert _looks_like_default_hook(0.0, 3.0) is True
    assert _looks_like_default_hook(None, None) is False
    assert _looks_like_default_hook(0.0, 2.9) is False
    assert _looks_like_default_hook(0.05, 3.0) is False
    assert _looks_like_default_hook(1.2, 4.8) is False  # real hook, untouched


def test_clamp_ignores_default_hook_window():
    """Regression test for P1: a fake [0.0, 3.0] hook must not gate trim_start.

    Before the fix, every clip that arrived at Stage 2.5 with the default
    fallback hook had its ``trim_start_sec`` clamped to 0.0, because the
    protection rule said "do not trim past hook_start_sec" and
    hook_start_sec was 0.0. This test reproduces the Cathy Wood clip 001
    scenario where the LLM requested a 9.62s start-trim.
    """
    clip = _clip(end=226.7, hook_start=0.0, hook_end=3.0)  # duration ~58.4s
    ts, te, stats = _clamp_decision(clip, 9.62, 0.0, level="balanced")

    assert ts > 0.0, "trim_start must not be zeroed by a fake [0, 3] hook"
    assert stats.hook_protected is False
    # balanced cap on ~58.4s is ~11.7s; 9.62s fits under the cap so it
    # should land roughly at the requested value (no reshape).
    assert ts == pytest.approx(9.62, abs=0.05)


def test_clamp_still_protects_real_hook():
    """A non-default hook window must still cap trim_start_sec.

    This is the positive case: the hook detector set a real
    ``[1.5, 5.0]`` window, so Stage 2.5 should not trim past 1.5s
    (minus the 0.25s safety margin).
    """
    clip = _clip(end=200.0, hook_start=1.5, hook_end=5.0)
    ts, te, stats = _clamp_decision(clip, 10.0, 0.0, level="aggressive")
    assert ts == pytest.approx(1.25, abs=1e-6)  # 1.5 - 0.25
    assert stats.hook_protected is True


def test_clamp_negatives_become_zero():
    clip = _clip(end=200.0)
    ts, te, stats = _clamp_decision(clip, -5.0, -2.0, level="balanced")
    assert ts == 0.0 and te == 0.0
    assert stats.clamped_start is True
    assert stats.clamped_end is True


# ---------------------------------------------------------------------------
# apply_prune_decisions
# ---------------------------------------------------------------------------


def test_apply_decisions_maps_by_clip_id():
    a = _clip("001", end=200.0)
    b = _clip("002", start=300.0, end=400.0)
    decisions = [
        _PruneDecision(clip_id="001", trim_start_sec=3.0, trim_end_sec=2.0),
        _PruneDecision(clip_id="002", trim_start_sec=1.0, trim_end_sec=1.0),
    ]
    out = apply_prune_decisions([a, b], decisions, level="balanced")
    assert out[0].trim_start_sec == pytest.approx(3.0)
    assert out[0].trim_end_sec == pytest.approx(2.0)
    assert out[1].trim_start_sec == pytest.approx(1.0)
    assert out[1].trim_end_sec == pytest.approx(1.0)


def test_apply_decisions_missing_id_is_no_op():
    a = _clip("001", end=200.0)
    out = apply_prune_decisions([a], decisions=[], level="balanced")
    assert out[0].trim_start_sec == 0.0
    assert out[0].trim_end_sec == 0.0


def test_apply_decisions_off_level_zeroes_everything():
    a = _clip("001", end=200.0)
    decisions = [_PruneDecision(clip_id="001", trim_start_sec=3.0, trim_end_sec=3.0)]
    out = apply_prune_decisions([a], decisions, level="off")
    assert out[0].trim_start_sec == 0.0
    assert out[0].trim_end_sec == 0.0


# ---------------------------------------------------------------------------
# Segment-boundary snapping (fixes mid-sentence cuts)
# ---------------------------------------------------------------------------


def _transcript_with_segments(base_start: float, rel_segments: list[tuple[float, float]]) -> dict:
    """Build a transcript dict with segments expressed relative to ``base_start``."""
    return {
        "segments": [
            {"start": base_start + s, "end": base_start + e, "text": f"seg {i}"}
            for i, (s, e) in enumerate(rel_segments)
        ]
    }


def test_snap_trim_end_forward_when_cut_falls_mid_sentence():
    """Regression test for the "this could be..." bug.

    A 58.4s clip with trim_end=6.38 cuts effectively at 52.02s. The nearest
    segment end is at 53.5s (1.5s later, well within tolerance). Snap must
    extend the clip forward to finish the sentence, reducing trim_end to
    58.4 - 53.5 = 4.9s.
    """
    clip = _clip(start=168.3, end=226.7)
    clip = clip.model_copy(update={"trim_start_sec": 0.04, "trim_end_sec": 6.38})
    # Segments (clip-relative): include one that ends just after the
    # requested cut point, and one just before. Forward snap should win.
    transcript = _transcript_with_segments(
        base_start=168.3,
        rel_segments=[
            (0.0, 4.5),
            (4.5, 10.0),
            (10.0, 30.0),
            (30.0, 51.4),   # ends 0.62s BEFORE target out (52.02)
            (51.4, 53.5),   # ends 1.48s AFTER target out -- cleanest sentence end
            (53.5, 58.4),
        ],
    )

    ts, te = _snap_trims_to_segment_boundaries(clip, transcript, level="balanced")

    assert te == pytest.approx(58.4 - 53.5, abs=0.05), \
        "trim_end must snap forward to the next segment end so the sentence finishes"
    # Confirm the snap moved the boundary, not just rounding error.
    assert abs(te - 6.38) > 0.5


def test_snap_trim_end_backward_when_no_forward_segment_in_tolerance():
    """If every forward segment end is outside tolerance, fall back to the
    nearest backward boundary (cut slightly more than requested).

    Clip is intentionally sized so the extra trim still respects
    ``MIN_CLIP_DURATION_SEC`` and the level's ``max_pct`` cap; the snap
    should never trade off correctness for boundary cleanliness.
    """
    # Use an 80s clip so trimming 11s still leaves 69s >= MIN_CLIP_DURATION_SEC.
    clip = _clip(start=0.0, end=80.0)
    clip = clip.model_copy(update={"trim_start_sec": 0.0, "trim_end_sec": 10.0})
    # Target effective out-point = 80 - 10 = 70s. Only segment end 69s
    # (1s backward) is within the 3s tolerance; 76s is outside it.
    transcript = _transcript_with_segments(
        base_start=0.0,
        rel_segments=[
            (0.0, 30.0),
            (30.0, 69.0),
            (76.0, 80.0),  # 6s forward -- outside 3s tolerance
        ],
    )

    ts, te = _snap_trims_to_segment_boundaries(
        clip, transcript, level="aggressive"
    )

    # Snap went to end=69.0 so trim_end = 80 - 69 = 11.0.
    assert te == pytest.approx(11.0, abs=0.05)


def test_snap_trim_start_forward_to_segment_start():
    """``trim_start`` should snap to the next segment start within tolerance
    so we drop lead-in filler cleanly.
    """
    clip = _clip(start=0.0, end=60.0)
    clip = clip.model_copy(update={"trim_start_sec": 0.8, "trim_end_sec": 0.0})
    transcript = _transcript_with_segments(
        base_start=0.0,
        rel_segments=[(0.0, 1.5), (1.5, 5.0), (5.0, 60.0)],
    )

    ts, _ = _snap_trims_to_segment_boundaries(clip, transcript, level="balanced")

    # A segment starts at 1.5s (0.7s forward from request) -- snap there.
    assert ts == pytest.approx(1.5, abs=0.05)


def test_snap_is_noop_when_no_segments_cover_clip():
    clip = _clip(start=0.0, end=60.0)
    clip = clip.model_copy(update={"trim_start_sec": 1.0, "trim_end_sec": 2.0})
    empty = {"segments": []}

    ts, te = _snap_trims_to_segment_boundaries(clip, empty, level="balanced")

    assert ts == 1.0
    assert te == 2.0


def test_snap_reverts_when_result_would_violate_min_duration():
    """If snapping would push the clip below MIN_CLIP_DURATION_SEC, bail out
    rather than silently shipping a too-short clip.
    """
    clip = _clip(start=0.0, end=MIN_CLIP_DURATION_SEC + 4.0)
    clip = clip.model_copy(update={"trim_start_sec": 1.0, "trim_end_sec": 2.0})
    # Segment end that would add a huge extra trim (pushing below min).
    rel_segments = [(0.0, 1.0), (1.0, 3.5), (3.5, MIN_CLIP_DURATION_SEC + 4.0)]
    transcript = _transcript_with_segments(0.0, rel_segments)

    ts, te = _snap_trims_to_segment_boundaries(
        clip, transcript, level="balanced"
    )
    # Final duration must stay legal, not lose 3+ more seconds via snapping.
    final = clip.duration_sec - ts - te
    assert final >= MIN_CLIP_DURATION_SEC - 1e-6


def test_snap_respects_real_hook_window():
    """A real (non-placeholder) hook must not be eaten by snapping."""
    clip = _clip(start=0.0, end=60.0, hook_start=2.0, hook_end=5.0)
    clip = clip.model_copy(update={"trim_start_sec": 1.0, "trim_end_sec": 0.0})
    # Snapping trim_start forward to 3.0 would cross the hook_start. Must revert.
    transcript = _transcript_with_segments(
        base_start=0.0,
        rel_segments=[(0.0, 1.0), (3.0, 10.0), (10.0, 60.0)],
    )

    ts, _ = _snap_trims_to_segment_boundaries(clip, transcript, level="balanced")

    # The only forward candidate (3.0) crosses the hook -> revert to original.
    assert ts == pytest.approx(1.0)


def test_apply_decisions_with_transcript_performs_snap():
    """End-to-end: ``apply_prune_decisions`` with a transcript snaps to
    segment boundaries. Without a transcript, existing behavior is
    preserved (backward compat for legacy callers).
    """
    clip = _clip(start=0.0, end=60.0)
    decisions = [_PruneDecision(clip_id="001", trim_start_sec=0.0, trim_end_sec=8.0)]
    transcript = _transcript_with_segments(
        base_start=0.0,
        rel_segments=[(0.0, 30.0), (30.0, 54.0), (54.0, 60.0)],
    )

    without_snap = apply_prune_decisions(
        [clip], decisions, level="balanced"
    )
    with_snap = apply_prune_decisions(
        [clip], decisions, level="balanced", transcript=transcript
    )

    # Without a transcript, trim_end stays at the LLM-requested 8.0.
    assert without_snap[0].trim_end_sec == pytest.approx(8.0, abs=0.05)
    # With the transcript, the cut first snaps to 54.0, then extends through
    # the immediate continuation so the rendered clip does not stop early.
    assert with_snap[0].trim_end_sec == pytest.approx(0.0, abs=0.05)


def test_trim_end_completion_keeps_one_word_sentence_tail():
    clip = Clip(
        clip_id="002",
        topic="worst case",
        start_time_sec=432.9,
        end_time_sec=515.989990234375,
        trim_end_sec=0.54,
    )
    transcript = {
        "segments": [
            {
                "start": 511.53,
                "end": 515.4500122070312,
                "text": "people factor in how valuable this learning is when they're doing their worst case scenario",
            },
            {"start": 515.4500122070312, "end": 515.989990234375, "text": "analysis."},
            {
                "start": 516.61,
                "end": 519.49,
                "text": "As a startup founder, you're responsible for making everything happen,",
            },
        ]
    }

    trim_end = _complete_trim_end_to_transcript_boundary(clip, transcript)

    assert trim_end == pytest.approx(0.0, abs=0.01)


def test_audio_keep_range_tail_completion_keeps_sentence_tail():
    clip = Clip(
        clip_id="002",
        topic="worst case",
        start_time_sec=432.9,
        end_time_sec=515.989990234375,
        keep_ranges_sec=[(75.65, 82.55)],
    )
    transcript = {
        "segments": [
            {
                "start": 511.53,
                "end": 515.4500122070312,
                "text": "people factor in how valuable this learning is when they're doing their worst case scenario",
            },
            {"start": 515.4500122070312, "end": 515.989990234375, "text": "analysis."},
        ]
    }

    completed = _complete_keep_range_tail(clip, transcript)

    assert completed.keep_ranges_sec[-1][1] == pytest.approx(
        clip.duration_sec, abs=0.01
    )


# ---------------------------------------------------------------------------
# Prompt / user-message construction
# ---------------------------------------------------------------------------


def test_segments_are_clip_relative():
    clip = _clip(start=100.0, end=190.0)
    transcript = {
        "segments": [
            {"start": 80.0, "end": 95.0, "text": "before"},
            {"start": 100.0, "end": 110.0, "text": "hello"},
            {"start": 150.0, "end": 160.0, "text": "middle"},
            {"start": 190.0, "end": 200.0, "text": "after"},
        ]
    }
    segs = _segments_within_clip(transcript, clip)
    texts = [s["text"] for s in segs]
    assert texts == ["hello", "middle"]
    assert segs[0]["start"] == pytest.approx(0.0)
    assert segs[0]["end"] == pytest.approx(10.0)
    assert segs[1]["start"] == pytest.approx(50.0)
    assert segs[1]["end"] == pytest.approx(60.0)


def test_build_user_message_includes_hook_when_set():
    clip = _clip(start=100.0, end=190.0, hook_start=0.0, hook_end=3.0)
    transcript = _transcript_for(100.0, 190.0)
    msg = _build_user_message([clip], transcript)
    assert "clip_id: 001" in msg
    assert "hook_window_sec" in msg
    assert "[0.00, 3.00]" in msg
    assert "duration_sec: 90.00" in msg


def test_build_user_message_separates_clips():
    clips = [
        _clip("001", start=100.0, end=160.0),
        _clip("002", start=300.0, end=380.0),
    ]
    transcript = _transcript_for(100.0, 380.0)
    msg = _build_user_message(clips, transcript)
    assert "clip_id: 001" in msg
    assert "clip_id: 002" in msg
    assert "===" in msg  # block separator


# ---------------------------------------------------------------------------
# JSON parsing
# ---------------------------------------------------------------------------


def test_parse_decisions_object_form():
    raw = json.dumps(
        {
            "decisions": [
                {"clip_id": "001", "trim_start_sec": 1.2, "trim_end_sec": 0.8, "reason": "ok"}
            ]
        }
    )
    decisions = _parse_decisions(raw)
    assert len(decisions) == 1
    assert decisions[0].clip_id == "001"
    assert decisions[0].trim_start_sec == pytest.approx(1.2)


def test_parse_decisions_array_form():
    raw = json.dumps(
        [{"clip_id": "001", "trim_start_sec": 1.0, "trim_end_sec": 0.0, "reason": "x"}]
    )
    decisions = _parse_decisions(raw)
    assert len(decisions) == 1


def test_parse_decisions_skips_malformed_items_in_array():
    raw = json.dumps(
        [
            {"clip_id": "001", "trim_start_sec": 1.0, "trim_end_sec": 0.0},
            {"not a decision": True},
        ]
    )
    decisions = _parse_decisions(raw)
    assert len(decisions) == 1
    assert decisions[0].clip_id == "001"


# ---------------------------------------------------------------------------
# Gemini call plumbing
# ---------------------------------------------------------------------------


@patch("afterform.flows.long_to_shorts.prune_content.call_structured_llm")
def test_request_prune_decisions_calls_provider_layer(mock_call):
    mock_call.return_value = SimpleNamespace(
        raw_text=json.dumps(
            {
                "decisions": [
                    {"clip_id": "001", "trim_start_sec": 2.0, "trim_end_sec": 1.0, "reason": "r"}
                ]
            }
        ),
        parsed=None,
    )

    clip = _clip("001", end=200.0)
    transcript = _transcript_for(100.0, 200.0)

    decisions, raw = request_prune_decisions(
        [clip], transcript, level="balanced", gemini_model="gemini-x"
    )

    mock_call.assert_called_once()
    request = mock_call.call_args.args[0]
    assert request.model == "gemini-x"
    assert request.stage_name == "content pruning"
    assert "clip_id: 001" in request.user_text
    assert mock_call.call_args.kwargs["provider"] == "gemini"

    assert len(decisions) == 1
    assert decisions[0].trim_start_sec == pytest.approx(2.0)
    assert json.loads(raw)["decisions"][0]["clip_id"] == "001"


def test_request_prune_decisions_off_level_is_no_op(monkeypatch):
    with patch("afterform.flows.long_to_shorts.prune_content.call_structured_llm") as mock_call:
        decisions, raw = request_prune_decisions(
            [_clip()], transcript={"segments": []}, level="off"
        )
        assert decisions == []
        assert json.loads(raw)["decisions"] == []
        mock_call.assert_not_called()


# ---------------------------------------------------------------------------
# Stage entrypoint + cache
# ---------------------------------------------------------------------------


def _mock_gemini_ok(mock_call, *, ts: float = 3.0, te: float = 2.0):
    mock_call.return_value = SimpleNamespace(
        raw_text=json.dumps(
            {
                "decisions": [
                    {"clip_id": "001", "trim_start_sec": ts, "trim_end_sec": te, "reason": "ok"}
                ]
            }
        ),
        parsed=None,
    )
    return mock_call


@patch("afterform.flows.long_to_shorts.prune_content.call_structured_llm")
def test_run_stage_writes_artifacts_and_applies_trims(mock_call, cfg):
    _mock_gemini_ok(mock_call, ts=3.0, te=2.0)

    clip = _clip("001", end=200.0)
    # Empty segments -> snap is a no-op, so this stays a pure plumbing test
    # (LLM request -> applied trims -> artifact). Segment-boundary snapping
    # behaviour is covered by the dedicated tests above.
    transcript = {"segments": []}

    out = run_content_pruning_stage(
        cfg.work_dir,
        [clip],
        transcript,
        transcript_fp="fp-1",
        config=cfg,
    )

    assert len(out) == 1
    assert out[0].trim_start_sec == pytest.approx(3.0)
    assert out[0].trim_end_sec == pytest.approx(2.0)

    assert (cfg.work_dir / PRUNE_META_FILENAME).is_file()
    assert (cfg.work_dir / PRUNE_ARTIFACT_FILENAME).is_file()
    assert (cfg.work_dir / PRUNE_RAW_FILENAME).is_file()
    meta = json.loads((cfg.work_dir / PRUNE_META_FILENAME).read_text())
    assert meta["prune_level"] == "balanced"
    assert meta["transcript_sha256"] == "fp-1"
    assert meta["filled_pause_pruning"] is False
    assert meta["require_filled_pause_pruning"] is False


@patch("afterform.flows.long_to_shorts.prune_content.call_structured_llm")
def test_run_stage_is_cached_on_second_call(mock_call, cfg):
    mock_inst = _mock_gemini_ok(mock_call, ts=3.0, te=2.0)

    clip = _clip("001", end=200.0)
    # Empty segments so snap is inert; this test isolates cache behaviour.
    transcript = {"segments": []}

    run_content_pruning_stage(cfg.work_dir, [clip], transcript, transcript_fp="fp", config=cfg)
    assert mock_inst.call_count == 1

    out2 = run_content_pruning_stage(
        cfg.work_dir, [clip], transcript, transcript_fp="fp", config=cfg
    )
    assert mock_inst.call_count == 1  # still 1 -> cache hit
    assert out2[0].trim_start_sec == pytest.approx(3.0)


@patch("afterform.flows.long_to_shorts.prune_content.call_structured_llm")
def test_run_stage_cache_invalidates_on_level_change(mock_call, cfg):
    mock_inst = _mock_gemini_ok(mock_call, ts=3.0, te=2.0)

    clip = _clip("001", end=200.0)
    transcript = _transcript_for(100.0, 200.0)

    run_content_pruning_stage(cfg.work_dir, [clip], transcript, transcript_fp="fp", config=cfg)
    assert mock_inst.call_count == 1

    cfg2 = PipelineConfig(
        youtube_url=cfg.youtube_url,
        work_dir=cfg.work_dir,
        gemini_model=cfg.gemini_model,
        prune_level="aggressive",
    )
    run_content_pruning_stage(
        cfg2.work_dir, [clip], transcript, transcript_fp="fp", config=cfg2
    )
    assert mock_inst.call_count == 2


@patch("afterform.flows.long_to_shorts.prune_content.call_structured_llm")
def test_run_stage_cache_invalidates_on_filled_pause_flag_change(mock_call, cfg):
    mock_inst = _mock_gemini_ok(mock_call, ts=3.0, te=2.0)

    clip = _clip("001", end=200.0)
    transcript = _transcript_for(100.0, 200.0)

    run_content_pruning_stage(cfg.work_dir, [clip], transcript, transcript_fp="fp", config=cfg)
    assert mock_inst.call_count == 1

    cfg2 = PipelineConfig(
        youtube_url=cfg.youtube_url,
        work_dir=cfg.work_dir,
        gemini_model=cfg.gemini_model,
        prune_level=cfg.prune_level,
        filled_pause_pruning=True,
    )
    run_content_pruning_stage(
        cfg2.work_dir, [clip], transcript, transcript_fp="fp", config=cfg2
    )
    assert mock_inst.call_count == 2

    artifact = json.loads((cfg.work_dir / PRUNE_ARTIFACT_FILENAME).read_text())
    assert artifact["clips"][0]["diagnostics"]["audio_backend"]["filled_pause"] == "stub"


@patch("afterform.flows.long_to_shorts.prune_content.call_structured_llm")
def test_run_stage_off_level_short_circuits(mock_call, tmp_path):
    cfg = PipelineConfig(
        youtube_url="https://youtu.be/abc",
        work_dir=tmp_path,
        gemini_model="gemini-test",
        prune_level="off",
    )
    clip = _clip("001", end=200.0)
    out = run_content_pruning_stage(
        cfg.work_dir, [clip], _transcript_for(100.0, 200.0), transcript_fp="fp", config=cfg
    )
    assert out[0].trim_start_sec == 0.0
    assert out[0].trim_end_sec == 0.0
    mock_call.assert_not_called()
    assert not (cfg.work_dir / PRUNE_META_FILENAME).exists()


@patch("afterform.flows.long_to_shorts.prune_content.call_structured_llm")
def test_run_stage_swallows_llm_errors(mock_call, cfg, caplog):
    mock_call.side_effect = RuntimeError("boom")

    clip = _clip("001", end=200.0)
    transcript = _transcript_for(100.0, 200.0)

    with caplog.at_level("WARNING", logger="afterform.flows.long_to_shorts.prune_content"):
        out = run_content_pruning_stage(
            cfg.work_dir, [clip], transcript, transcript_fp="fp", config=cfg
        )
    assert out[0].trim_start_sec == 0.0
    assert out[0].trim_end_sec == 0.0
    assert any("Content pruning call failed" in r.message for r in caplog.records)


@patch("afterform.flows.long_to_shorts.prune_content.call_structured_llm")
def test_run_stage_force_bypasses_cache(mock_call, cfg):
    mock_inst = _mock_gemini_ok(mock_call, ts=3.0, te=2.0)
    clip = _clip("001", end=200.0)
    transcript = _transcript_for(100.0, 200.0)

    run_content_pruning_stage(cfg.work_dir, [clip], transcript, transcript_fp="fp", config=cfg)
    assert mock_inst.call_count == 1

    cfg_force = PipelineConfig(
        youtube_url=cfg.youtube_url,
        work_dir=cfg.work_dir,
        gemini_model=cfg.gemini_model,
        prune_level=cfg.prune_level,
        force_content_pruning=True,
    )
    run_content_pruning_stage(
        cfg_force.work_dir, [clip], transcript, transcript_fp="fp", config=cfg_force
    )
    assert mock_inst.call_count == 2


# ---------------------------------------------------------------------------
# Fingerprinting
# ---------------------------------------------------------------------------


def test_clips_fingerprint_is_trim_independent():
    a = _clip("001", end=200.0)
    b = a.model_copy(update={"trim_start_sec": 5.0, "trim_end_sec": 3.0})
    assert _clips_fingerprint([a]) == _clips_fingerprint([b])


def test_clips_fingerprint_changes_on_window_change():
    a = _clip("001", end=200.0)
    b = _clip("001", end=201.0)
    assert _clips_fingerprint([a]) != _clips_fingerprint([b])


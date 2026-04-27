"""Tests for Stage 2.25 hook detection.

No network. The Gemini client is always mocked. These tests cover:

- validation (window bounds, duration limits, placeholder rejection)
- ``apply_hook_decisions`` overwrites only with valid windows and logs
- prompt construction includes clip-relative segments + selector hook text
- ``request_hook_decisions`` sends the right args and parses the response
- ``run_hook_detection_stage`` happy path, cache hit, force re-run,
  disabled short-circuit, graceful failure on LLM error
"""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import pytest

from afterform.config import PipelineConfig
from afterform.flows.long_to_shorts.detect_hooks import (
    HOOK_ARTIFACT_FILENAME,
    HOOK_META_FILENAME,
    HOOK_RAW_FILENAME,
    _build_user_message,
    _clips_fingerprint,
    _HookDecision,
    _parse_decisions,
    _validate_hook_window,
    apply_hook_decisions,
    request_hook_decisions,
    run_hook_detection_stage,
)
from afterform.schemas import Clip


def _clip(
    clip_id: str = "001",
    *,
    start: float = 100.0,
    end: float = 190.0,
    hook_start: float | None = 0.0,
    hook_end: float | None = 3.0,
    viral_hook: str = "",
) -> Clip:
    return Clip.model_validate(
        {
            "clip_id": clip_id,
            "topic": "topic",
            "start_time_sec": start,
            "end_time_sec": end,
            "hook_start_sec": hook_start,
            "hook_end_sec": hook_end,
            "viral_hook": viral_hook,
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
        detect_hooks=True,
    )


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------


def test_validate_accepts_real_hook_window():
    clip = _clip(end=200.0)  # duration 100s
    assert _validate_hook_window(clip, 4.0, 7.5) == (4.0, 7.5)


def test_validate_rejects_default_placeholder():
    """The whole point of this stage is to eliminate [0.0, 3.0]."""
    clip = _clip(end=200.0)
    assert _validate_hook_window(clip, 0.0, 3.0) is None


def test_validate_rejects_too_short_hook():
    clip = _clip(end=200.0)
    assert _validate_hook_window(clip, 4.0, 4.5) is None  # 0.5s < 1.0s floor


def test_validate_rejects_too_long_hook():
    clip = _clip(end=200.0)
    assert _validate_hook_window(clip, 0.0, 12.0) is None  # > 10.0s ceiling


def test_validate_rejects_inverted_window():
    clip = _clip(end=200.0)
    assert _validate_hook_window(clip, 7.0, 4.0) is None
    assert _validate_hook_window(clip, 4.0, 4.0) is None


def test_validate_clamps_trailing_rounding():
    """End up to 0.5s past the clip end gets clamped (LLM rounding grace)."""
    clip = _clip(end=200.0)  # duration = 100s
    out = _validate_hook_window(clip, 92.0, 100.3)
    assert out is not None
    assert out[1] == pytest.approx(100.0)


def test_validate_rejects_far_beyond_duration():
    clip = _clip(end=200.0)
    assert _validate_hook_window(clip, 92.0, 101.0) is None


# ---------------------------------------------------------------------------
# apply_hook_decisions
# ---------------------------------------------------------------------------


def test_apply_overwrites_placeholder_with_valid_window():
    clip = _clip("001", end=200.0, hook_start=0.0, hook_end=3.0)
    decisions = [
        _HookDecision(clip_id="001", hook_start_sec=12.5, hook_end_sec=16.0, reason="r")
    ]
    out = apply_hook_decisions([clip], decisions)
    assert out[0].hook_start_sec == pytest.approx(12.5)
    assert out[0].hook_end_sec == pytest.approx(16.0)


def test_apply_keeps_clip_unchanged_when_decision_invalid():
    clip = _clip("001", end=200.0, hook_start=0.0, hook_end=3.0)
    decisions = [
        _HookDecision(clip_id="001", hook_start_sec=0.0, hook_end_sec=3.0)
    ]
    out = apply_hook_decisions([clip], decisions)
    assert out[0].hook_start_sec == 0.0
    assert out[0].hook_end_sec == 3.0


def test_apply_leaves_unmatched_clips_alone():
    clips = [
        _clip("001", end=200.0, hook_start=None, hook_end=None),
        _clip("002", start=300.0, end=380.0, hook_start=None, hook_end=None),
    ]
    decisions = [
        _HookDecision(clip_id="001", hook_start_sec=1.0, hook_end_sec=4.0)
    ]
    out = apply_hook_decisions(clips, decisions)
    assert out[0].hook_start_sec == pytest.approx(1.0)
    assert out[1].hook_start_sec is None


# ---------------------------------------------------------------------------
# Prompt / user-message
# ---------------------------------------------------------------------------


def test_build_user_message_includes_clip_relative_segments_and_hint():
    clip = _clip(
        "001",
        start=100.0,
        end=190.0,
        hook_start=0.0,
        hook_end=3.0,
        viral_hook="A claim about markets",
    )
    transcript = {
        "segments": [
            {"start": 90.0, "end": 100.0, "text": "before"},
            {"start": 100.0, "end": 110.0, "text": "hello"},
            {"start": 150.0, "end": 160.0, "text": "middle"},
        ]
    }
    msg = _build_user_message([clip], transcript)
    assert "clip_id: 001" in msg
    assert "duration_sec: 90.00" in msg
    assert "viral_hook_text: A claim about markets" in msg
    assert "selector_hook_window_sec: [0.00, 3.00]" in msg
    assert "placeholder" in msg  # sanity hint to LLM
    assert "[0.00s - 10.00s] hello" in msg
    assert "before" not in msg  # pre-clip segment excluded


# ---------------------------------------------------------------------------
# JSON parsing
# ---------------------------------------------------------------------------


def test_parse_decisions_object_form():
    raw = json.dumps(
        {
            "hooks": [
                {
                    "clip_id": "001",
                    "hook_start_sec": 4.2,
                    "hook_end_sec": 7.8,
                    "hook_text": "a claim",
                    "reason": "starts on a claim",
                }
            ]
        }
    )
    decisions = _parse_decisions(raw)
    assert len(decisions) == 1
    assert decisions[0].hook_start_sec == pytest.approx(4.2)


def test_parse_decisions_skips_malformed_array_items():
    raw = json.dumps(
        [
            {"clip_id": "001", "hook_start_sec": 1.0, "hook_end_sec": 3.5},
            {"bad": "item"},
        ]
    )
    decisions = _parse_decisions(raw)
    assert len(decisions) == 1
    assert decisions[0].clip_id == "001"


# ---------------------------------------------------------------------------
# Gemini call plumbing
# ---------------------------------------------------------------------------


@patch("afterform.flows.long_to_shorts.detect_hooks.call_structured_llm")
def test_request_hook_decisions_calls_provider_layer(mock_call):
    mock_call.return_value = SimpleNamespace(
        raw_text=json.dumps(
            {"hooks": [{"clip_id": "001", "hook_start_sec": 4.0, "hook_end_sec": 7.0}]}
        ),
        parsed=None,
    )
    clip = _clip("001", end=200.0)
    transcript = _transcript_for(100.0, 200.0)

    decisions, raw = request_hook_decisions(
        [clip], transcript, gemini_model="gemini-x"
    )

    mock_call.assert_called_once()
    request = mock_call.call_args.args[0]
    assert request.model == "gemini-x"
    assert request.stage_name == "hook detection"
    assert "clip_id: 001" in request.user_text
    assert mock_call.call_args.kwargs["provider"] == "gemini"
    assert len(decisions) == 1
    assert json.loads(raw)["hooks"][0]["clip_id"] == "001"


# ---------------------------------------------------------------------------
# Stage entrypoint + cache
# ---------------------------------------------------------------------------


def _mock_gemini_ok(mock_call, *, hs: float = 4.0, he: float = 7.0):
    mock_call.return_value = SimpleNamespace(
        raw_text=json.dumps(
            {
                "hooks": [
                    {
                        "clip_id": "001",
                        "hook_start_sec": hs,
                        "hook_end_sec": he,
                        "hook_text": "ok",
                        "reason": "ok",
                    }
                ]
            }
        ),
        parsed=None,
    )
    return mock_call


@patch("afterform.flows.long_to_shorts.detect_hooks.call_structured_llm")
def test_run_stage_writes_artifacts_and_updates_hook(mock_call, cfg):
    _mock_gemini_ok(mock_call, hs=4.0, he=7.0)

    clip = _clip("001", end=200.0, hook_start=0.0, hook_end=3.0)
    transcript = _transcript_for(100.0, 200.0)

    out = run_hook_detection_stage(
        cfg.work_dir, [clip], transcript, transcript_fp="fp-1", config=cfg
    )
    assert out[0].hook_start_sec == pytest.approx(4.0)
    assert out[0].hook_end_sec == pytest.approx(7.0)

    assert (cfg.work_dir / HOOK_META_FILENAME).is_file()
    assert (cfg.work_dir / HOOK_ARTIFACT_FILENAME).is_file()
    assert (cfg.work_dir / HOOK_RAW_FILENAME).is_file()
    meta = json.loads((cfg.work_dir / HOOK_META_FILENAME).read_text())
    assert meta["transcript_sha256"] == "fp-1"


@patch("afterform.flows.long_to_shorts.detect_hooks.call_structured_llm")
def test_run_stage_cache_hit_skips_llm(mock_call, cfg):
    mock_inst = _mock_gemini_ok(mock_call, hs=4.0, he=7.0)

    clip = _clip("001", end=200.0, hook_start=0.0, hook_end=3.0)
    transcript = _transcript_for(100.0, 200.0)

    run_hook_detection_stage(
        cfg.work_dir, [clip], transcript, transcript_fp="fp", config=cfg
    )
    assert mock_inst.call_count == 1

    out2 = run_hook_detection_stage(
        cfg.work_dir, [clip], transcript, transcript_fp="fp", config=cfg
    )
    assert mock_inst.call_count == 1  # still 1 -> cache hit
    assert out2[0].hook_start_sec == pytest.approx(4.0)


@patch("afterform.flows.long_to_shorts.detect_hooks.call_structured_llm")
def test_run_stage_force_bypasses_cache(mock_call, cfg):
    mock_inst = _mock_gemini_ok(mock_call)

    clip = _clip("001", end=200.0, hook_start=0.0, hook_end=3.0)
    transcript = _transcript_for(100.0, 200.0)

    run_hook_detection_stage(
        cfg.work_dir, [clip], transcript, transcript_fp="fp", config=cfg
    )
    assert mock_inst.call_count == 1

    cfg_force = PipelineConfig(
        youtube_url=cfg.youtube_url,
        work_dir=cfg.work_dir,
        gemini_model=cfg.gemini_model,
        force_hook_detection=True,
    )
    run_hook_detection_stage(
        cfg_force.work_dir, [clip], transcript, transcript_fp="fp", config=cfg_force
    )
    assert mock_inst.call_count == 2


@patch("afterform.flows.long_to_shorts.detect_hooks.call_structured_llm")
def test_run_stage_disabled_short_circuits(mock_call, tmp_path):
    cfg = PipelineConfig(
        youtube_url="https://youtu.be/abc",
        work_dir=tmp_path,
        gemini_model="gemini-test",
        detect_hooks=False,
    )
    clip = _clip("001", end=200.0)
    out = run_hook_detection_stage(
        cfg.work_dir, [clip], _transcript_for(100.0, 200.0), transcript_fp="fp", config=cfg
    )
    assert out[0].hook_start_sec == 0.0
    assert out[0].hook_end_sec == 3.0
    mock_call.assert_not_called()
    assert not (cfg.work_dir / HOOK_META_FILENAME).exists()


@patch("afterform.flows.long_to_shorts.detect_hooks.call_structured_llm")
def test_run_stage_swallows_llm_errors(mock_call, cfg, caplog):
    """A failing LLM call must not kill the pipeline; clips pass through unchanged."""
    mock_call.side_effect = RuntimeError("boom")

    clip = _clip("001", end=200.0, hook_start=0.0, hook_end=3.0)
    transcript = _transcript_for(100.0, 200.0)

    with caplog.at_level("WARNING", logger="afterform.flows.long_to_shorts.detect_hooks"):
        out = run_hook_detection_stage(
            cfg.work_dir, [clip], transcript, transcript_fp="fp", config=cfg
        )
    assert out[0].hook_start_sec == 0.0
    assert out[0].hook_end_sec == 3.0
    assert any("Hook detection call failed" in r.message for r in caplog.records)


# ---------------------------------------------------------------------------
# Fingerprinting
# ---------------------------------------------------------------------------


def test_clips_fingerprint_stable_across_hook_changes():
    """We must not invalidate the hook cache when hooks change; only windows."""
    a = _clip("001", end=200.0, hook_start=0.0, hook_end=3.0)
    b = a.model_copy(update={"hook_start_sec": 4.0, "hook_end_sec": 7.0})
    assert _clips_fingerprint([a]) == _clips_fingerprint([b])


def test_clips_fingerprint_changes_on_window_change():
    a = _clip("001", end=200.0)
    b = _clip("001", end=201.0)
    assert _clips_fingerprint([a]) != _clips_fingerprint([b])


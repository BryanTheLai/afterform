"""Audio-first pruning helpers."""

from __future__ import annotations

import numpy as np
import pytest

from afterform.flows.long_to_shorts.audio_pruning import AudioBuffer, compute_audio_keep_ranges
from afterform.schemas import Clip


def _clip(**overrides) -> Clip:
    base = {
        "clip_id": "001",
        "topic": "t",
        "start_time_sec": 0.0,
        "end_time_sec": 10.0,
    }
    base.update(overrides)
    return Clip.model_validate(base)


def test_compute_audio_keep_ranges_offsets_outer_trim_and_subtracts_pauses(monkeypatch):
    monkeypatch.setattr(
        "afterform.flows.long_to_shorts.audio_pruning.detect_speech_ranges",
        lambda samples, sample_rate: ([(0.0, 1.0), (1.5, 3.0)], "speech_stub", []),
    )
    monkeypatch.setattr(
        "afterform.flows.long_to_shorts.audio_pruning.detect_filled_pause_ranges",
        lambda samples, sample_rate: ([(1.8, 2.0)], "pause_stub", []),
    )

    clip = _clip(trim_start_sec=2.0, trim_end_sec=1.0)
    audio = AudioBuffer(sample_rate=16_000, samples=np.zeros(16_000 * 12, dtype=np.float32))

    result = compute_audio_keep_ranges(audio, clip, filled_pause_enabled=True)

    assert result.keep_ranges_sec == [(2.0, 3.0), (3.5, 3.8), (4.0, 5.0)]
    assert result.diagnostics["audio_backend"]["speech"] == "speech_stub"
    assert result.diagnostics["audio_backend"]["filled_pause"] == "pause_stub"


def test_compute_audio_keep_ranges_falls_back_to_outer_window_when_detectors_empty(monkeypatch):
    monkeypatch.setattr(
        "afterform.flows.long_to_shorts.audio_pruning.detect_speech_ranges",
        lambda samples, sample_rate: ([], "speech_stub", []),
    )
    monkeypatch.setattr(
        "afterform.flows.long_to_shorts.audio_pruning.detect_filled_pause_ranges",
        lambda samples, sample_rate: ([], "pause_stub", []),
    )

    clip = _clip(trim_start_sec=1.0, trim_end_sec=2.0)
    audio = AudioBuffer(sample_rate=16_000, samples=np.zeros(16_000 * 12, dtype=np.float32))

    result = compute_audio_keep_ranges(audio, clip)

    assert result.keep_ranges_sec == [(1.0, 8.0)]


def test_compute_audio_keep_ranges_skips_filled_pause_detector_by_default(monkeypatch):
    monkeypatch.setattr(
        "afterform.flows.long_to_shorts.audio_pruning.detect_speech_ranges",
        lambda samples, sample_rate: ([(0.0, 3.0)], "speech_stub", []),
    )

    def fail_if_called(samples, sample_rate):
        raise AssertionError("filled-pause detector should be opt-in")

    monkeypatch.setattr(
        "afterform.flows.long_to_shorts.audio_pruning.detect_filled_pause_ranges",
        fail_if_called,
    )

    clip = _clip()
    audio = AudioBuffer(sample_rate=16_000, samples=np.zeros(16_000 * 12, dtype=np.float32))

    result = compute_audio_keep_ranges(audio, clip)

    assert result.filled_pause_ranges_sec == []
    assert result.diagnostics["audio_backend"]["filled_pause"] == "disabled"


def test_compute_audio_keep_ranges_requires_filled_pause_detector_when_configured(monkeypatch):
    monkeypatch.setattr(
        "afterform.flows.long_to_shorts.audio_pruning.detect_speech_ranges",
        lambda samples, sample_rate: ([(0.0, 3.0)], "speech_stub", []),
    )
    monkeypatch.setattr(
        "afterform.flows.long_to_shorts.audio_pruning.detect_filled_pause_ranges",
        lambda samples, sample_rate: ([], "none", ["Filled-pause model unavailable."]),
    )

    clip = _clip()
    audio = AudioBuffer(sample_rate=16_000, samples=np.zeros(16_000 * 12, dtype=np.float32))

    with pytest.raises(RuntimeError, match="Filled-pause pruning required"):
        compute_audio_keep_ranges(
            audio,
            clip,
            filled_pause_enabled=True,
            require_filled_pause=True,
        )


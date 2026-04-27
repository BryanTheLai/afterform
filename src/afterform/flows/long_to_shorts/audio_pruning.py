"""Audio-first keep-range detection for silence and filled-pause pruning."""

from __future__ import annotations

import logging
import wave
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from afterform.schemas import Clip

logger = logging.getLogger(__name__)

_FILLED_PAUSE_MODEL_ID = "classla/wav2vecbert2-filledPause"
_AUDIO_SAMPLE_RATE = 16_000
_FRAME_SEC = 0.02
_ENERGY_FRAME_SEC = 0.02
_ENERGY_HOP_SEC = 0.01
_ENERGY_PAD_SEC = 0.05
_ENERGY_MIN_SPEECH_SEC = 0.25
_ENERGY_MERGE_GAP_SEC = 0.22
_FILLED_PAUSE_THRESHOLD = 0.55
_FILLED_PAUSE_CHUNK_SEC = 20.0
_FILLED_PAUSE_MIN_SEC = 0.08
_FILLED_PAUSE_PAD_SEC = 0.04
_KEEP_MIN_SEC = 0.20
_KEEP_MERGE_GAP_SEC = 0.12
_DEFAULT_HOOK_FINGERPRINT: tuple[float, float] = (0.0, 3.0)
_DEFAULT_HOOK_EPS: float = 1e-3
_FILLED_PAUSE_RUNTIME: tuple[Any, Any, Any] | None = None


@dataclass
class AudioBuffer:
    sample_rate: int
    samples: np.ndarray


@dataclass
class AudioKeepResult:
    keep_ranges_sec: list[tuple[float, float]]
    speech_ranges_sec: list[tuple[float, float]]
    filled_pause_ranges_sec: list[tuple[float, float]]
    outer_window_sec: tuple[float, float]
    diagnostics: dict[str, Any]


def load_audio_buffer(path: Path) -> AudioBuffer:
    with wave.open(str(path), "rb") as wf:
        channels = wf.getnchannels()
        sample_width = wf.getsampwidth()
        sample_rate = wf.getframerate()
        frame_count = wf.getnframes()
        raw = wf.readframes(frame_count)

    if sample_width != 2:
        raise RuntimeError(f"Unsupported WAV sample width {sample_width}; expected 16-bit PCM")

    samples = np.frombuffer(raw, dtype="<i2").astype(np.float32) / 32768.0
    if channels > 1:
        samples = samples.reshape(-1, channels).mean(axis=1)
    if sample_rate != _AUDIO_SAMPLE_RATE:
        raise RuntimeError(
            f"Unsupported WAV sample rate {sample_rate}; expected {_AUDIO_SAMPLE_RATE} Hz mono PCM."
        )
    return AudioBuffer(sample_rate=sample_rate, samples=samples)


def _looks_like_default_hook(hook_start: float | None, hook_end: float | None) -> bool:
    if hook_start is None or hook_end is None:
        return False
    return (
        abs(hook_start - _DEFAULT_HOOK_FINGERPRINT[0]) < _DEFAULT_HOOK_EPS
        and abs(hook_end - _DEFAULT_HOOK_FINGERPRINT[1]) < _DEFAULT_HOOK_EPS
    )


def compute_audio_keep_ranges(
    audio: AudioBuffer,
    clip: Clip,
) -> AudioKeepResult:
    outer_start = max(0.0, min(clip.duration_sec, clip.trim_start_sec))
    outer_end = max(outer_start, min(clip.duration_sec, clip.duration_sec - clip.trim_end_sec))
    if outer_end <= outer_start:
        outer_start, outer_end = 0.0, clip.duration_sec

    start_idx = int(round((clip.start_time_sec + outer_start) * audio.sample_rate))
    end_idx = int(round((clip.start_time_sec + outer_end) * audio.sample_rate))
    end_idx = max(start_idx + 1, min(len(audio.samples), end_idx))
    clip_audio = audio.samples[start_idx:end_idx]
    if clip_audio.size == 0:
        return AudioKeepResult(
            keep_ranges_sec=[(outer_start, outer_end)],
            speech_ranges_sec=[(0.0, outer_end - outer_start)],
            filled_pause_ranges_sec=[],
            outer_window_sec=(outer_start, outer_end),
            diagnostics={
                "audio_backend": "empty_fallback",
                "warnings": ["No audio samples overlapped the clip window; kept the outer window."],
            },
        )

    speech_ranges, vad_backend, vad_warnings = detect_speech_ranges(
        clip_audio, audio.sample_rate
    )
    filled_pause_ranges, fp_backend, fp_warnings = detect_filled_pause_ranges(
        clip_audio, audio.sample_rate
    )
    protected_ranges = _protected_hook_ranges(clip, outer_start=outer_start, outer_end=outer_end)
    filled_pause_ranges = _subtract_protected_ranges(filled_pause_ranges, protected_ranges)

    keep_ranges = _subtract_ranges(speech_ranges, filled_pause_ranges)
    keep_ranges = _merge_ranges(keep_ranges, max_gap_sec=_KEEP_MERGE_GAP_SEC)
    keep_ranges = [rng for rng in keep_ranges if (rng[1] - rng[0]) >= _KEEP_MIN_SEC]

    if not keep_ranges:
        keep_ranges = [(0.0, outer_end - outer_start)]

    keep_ranges = [
        (
            round(outer_start + start, 3),
            round(outer_start + end, 3),
        )
        for start, end in keep_ranges
    ]

    return AudioKeepResult(
        keep_ranges_sec=keep_ranges,
        speech_ranges_sec=[(round(start, 3), round(end, 3)) for start, end in speech_ranges],
        filled_pause_ranges_sec=[
            (round(start, 3), round(end, 3)) for start, end in filled_pause_ranges
        ],
        outer_window_sec=(round(outer_start, 3), round(outer_end, 3)),
        diagnostics={
            "audio_backend": {
                "speech": vad_backend,
                "filled_pause": fp_backend,
            },
            "warnings": vad_warnings + fp_warnings,
            "protected_hook_ranges_sec": [
                (round(start, 3), round(end, 3)) for start, end in protected_ranges
            ],
        },
    )


def detect_speech_ranges(
    samples: np.ndarray,
    sample_rate: int,
) -> tuple[list[tuple[float, float]], str, list[str]]:
    """Return audio-relative speech ranges using a model VAD when available."""
    try:
        return _detect_speech_ranges_silero(samples, sample_rate)
    except Exception as exc:
        logger.info("Silero VAD unavailable, falling back to energy VAD: %s", exc)
        ranges = _detect_speech_ranges_energy(samples, sample_rate)
        return ranges, "energy_vad", [f"Silero VAD unavailable; used energy VAD fallback ({exc})."]


def _detect_speech_ranges_silero(
    samples: np.ndarray,
    sample_rate: int,
) -> tuple[list[tuple[float, float]], str, list[str]]:
    from silero_vad import get_speech_timestamps, load_silero_vad  # type: ignore

    import torch

    model = load_silero_vad()
    tensor = torch.from_numpy(samples.astype(np.float32))
    timestamps = get_speech_timestamps(
        tensor,
        model,
        sampling_rate=sample_rate,
        return_seconds=True,
    )
    ranges = [
        (float(item["start"]), float(item["end"]))
        for item in timestamps
        if float(item["end"]) > float(item["start"])
    ]
    ranges = _merge_ranges(ranges, max_gap_sec=_ENERGY_MERGE_GAP_SEC)
    ranges = _pad_ranges(ranges, pad_sec=_ENERGY_PAD_SEC, max_end=len(samples) / sample_rate)
    ranges = [rng for rng in ranges if (rng[1] - rng[0]) >= _ENERGY_MIN_SPEECH_SEC]
    if not ranges:
        ranges = [(0.0, len(samples) / sample_rate)]
    return ranges, "silero_vad", []


def _detect_speech_ranges_energy(
    samples: np.ndarray,
    sample_rate: int,
) -> list[tuple[float, float]]:
    frame = max(1, int(round(sample_rate * _ENERGY_FRAME_SEC)))
    hop = max(1, int(round(sample_rate * _ENERGY_HOP_SEC)))
    duration = len(samples) / float(sample_rate)
    if len(samples) <= frame:
        return [(0.0, duration)]

    rms_values: list[float] = []
    centers: list[float] = []
    for start in range(0, max(1, len(samples) - frame + 1), hop):
        chunk = samples[start : start + frame]
        rms = float(np.sqrt(np.mean(np.square(chunk)) + 1e-9))
        rms_values.append(rms)
        centers.append((start + frame / 2.0) / sample_rate)

    rms_db = 20.0 * np.log10(np.asarray(rms_values) + 1e-9)
    noise_floor = float(np.percentile(rms_db, 20))
    threshold = min(-24.0, max(-42.0, noise_floor + 9.0))
    mask = rms_db >= threshold
    if not bool(mask.any()):
        return [(0.0, duration)]

    ranges: list[tuple[float, float]] = []
    current_start: float | None = None
    for idx, active in enumerate(mask.tolist()):
        center = centers[idx]
        if active and current_start is None:
            current_start = max(0.0, center - _ENERGY_FRAME_SEC / 2.0)
        if not active and current_start is not None:
            end = min(duration, center + _ENERGY_FRAME_SEC / 2.0)
            ranges.append((current_start, end))
            current_start = None
    if current_start is not None:
        ranges.append((current_start, duration))

    ranges = _merge_ranges(ranges, max_gap_sec=_ENERGY_MERGE_GAP_SEC)
    ranges = _pad_ranges(ranges, pad_sec=_ENERGY_PAD_SEC, max_end=duration)
    ranges = [rng for rng in ranges if (rng[1] - rng[0]) >= _ENERGY_MIN_SPEECH_SEC]
    if not ranges:
        ranges = [(0.0, duration)]
    return ranges


def detect_filled_pause_ranges(
    samples: np.ndarray,
    sample_rate: int,
) -> tuple[list[tuple[float, float]], str, list[str]]:
    try:
        ranges = _detect_filled_pause_ranges_transformer(samples, sample_rate)
        if not ranges:
            return [], _FILLED_PAUSE_MODEL_ID, []
        ranges = _merge_ranges(ranges, max_gap_sec=_FILLED_PAUSE_PAD_SEC)
        ranges = _pad_ranges(ranges, pad_sec=_FILLED_PAUSE_PAD_SEC, max_end=len(samples) / sample_rate)
        return ranges, _FILLED_PAUSE_MODEL_ID, []
    except Exception as exc:
        logger.warning("Filled-pause detection unavailable: %s", exc)
        return [], "none", [f"Filled-pause model unavailable; skipped filled-pause removal ({exc})."]


def _detect_filled_pause_ranges_transformer(
    samples: np.ndarray,
    sample_rate: int,
) -> list[tuple[float, float]]:
    if sample_rate != _AUDIO_SAMPLE_RATE:
        raise RuntimeError(
            f"Unsupported sample rate {sample_rate}; expected {_AUDIO_SAMPLE_RATE} Hz."
        )

    import torch

    feature_extractor, model, device = _load_filled_pause_runtime()
    ranges: list[tuple[float, float]] = []
    chunk_samples = int(round(sample_rate * _FILLED_PAUSE_CHUNK_SEC))

    for chunk_start in range(0, len(samples), chunk_samples):
        chunk = samples[chunk_start : chunk_start + chunk_samples]
        if len(chunk) < int(sample_rate * 0.25):
            continue
        with torch.no_grad():
            inputs = feature_extractor(
                [chunk],
                return_tensors="pt",
                sampling_rate=sample_rate,
            ).to(device)
            logits = model(**inputs).logits[0]
        probs = torch.softmax(logits, dim=-1)[:, 1].detach().cpu().numpy()
        chunk_duration = len(chunk) / float(sample_rate)
        ranges.extend(
            _frame_scores_to_ranges(
                probs,
                offset_sec=chunk_start / float(sample_rate),
                chunk_duration_sec=chunk_duration,
                threshold=_FILLED_PAUSE_THRESHOLD,
            )
        )
    return ranges


def _load_filled_pause_runtime() -> tuple[Any, Any, Any]:
    global _FILLED_PAUSE_RUNTIME
    if _FILLED_PAUSE_RUNTIME is not None:
        return _FILLED_PAUSE_RUNTIME

    import torch
    from transformers import AutoFeatureExtractor, Wav2Vec2BertForAudioFrameClassification

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    feature_extractor = AutoFeatureExtractor.from_pretrained(_FILLED_PAUSE_MODEL_ID)
    model = Wav2Vec2BertForAudioFrameClassification.from_pretrained(_FILLED_PAUSE_MODEL_ID)
    model.to(device)
    model.eval()
    _FILLED_PAUSE_RUNTIME = (feature_extractor, model, device)
    return _FILLED_PAUSE_RUNTIME


def _frame_scores_to_ranges(
    scores: np.ndarray,
    *,
    offset_sec: float,
    chunk_duration_sec: float,
    threshold: float,
) -> list[tuple[float, float]]:
    if scores.size == 0:
        return []
    frame_sec = chunk_duration_sec / float(len(scores))
    active = scores >= threshold
    ranges: list[tuple[float, float]] = []
    start_idx: int | None = None
    for idx, value in enumerate(active.tolist()):
        if value and start_idx is None:
            start_idx = idx
        if not value and start_idx is not None:
            start = offset_sec + start_idx * frame_sec
            end = offset_sec + idx * frame_sec
            ranges.append((start, end))
            start_idx = None
    if start_idx is not None:
        ranges.append((offset_sec + start_idx * frame_sec, offset_sec + len(scores) * frame_sec))

    trimmed: list[tuple[float, float]] = []
    for start, end in ranges:
        if end - start < _FILLED_PAUSE_MIN_SEC:
            continue
        if abs(start - offset_sec) < frame_sec + 1e-6:
            continue
        if abs(end - (offset_sec + chunk_duration_sec)) < frame_sec + 1e-6:
            continue
        trimmed.append((start, end))
    return trimmed


def _protected_hook_ranges(
    clip: Clip,
    *,
    outer_start: float,
    outer_end: float,
) -> list[tuple[float, float]]:
    if (
        clip.hook_start_sec is None
        or clip.hook_end_sec is None
        or _looks_like_default_hook(clip.hook_start_sec, clip.hook_end_sec)
    ):
        return []
    start = max(outer_start, float(clip.hook_start_sec))
    end = min(outer_end, float(clip.hook_end_sec))
    if end <= start:
        return []
    return [(start - outer_start, end - outer_start)]


def _subtract_protected_ranges(
    ranges: list[tuple[float, float]],
    protected: list[tuple[float, float]],
) -> list[tuple[float, float]]:
    out: list[tuple[float, float]] = []
    for start, end in ranges:
        pieces = [(start, end)]
        for p_start, p_end in protected:
            next_pieces: list[tuple[float, float]] = []
            for cur_start, cur_end in pieces:
                if p_end <= cur_start or p_start >= cur_end:
                    next_pieces.append((cur_start, cur_end))
                    continue
                if cur_start < p_start:
                    next_pieces.append((cur_start, p_start))
                if p_end < cur_end:
                    next_pieces.append((p_end, cur_end))
            pieces = next_pieces
        out.extend(piece for piece in pieces if piece[1] > piece[0])
    return out


def _subtract_ranges(
    source_ranges: list[tuple[float, float]],
    remove_ranges: list[tuple[float, float]],
) -> list[tuple[float, float]]:
    out: list[tuple[float, float]] = []
    for start, end in source_ranges:
        pieces = [(start, end)]
        for rem_start, rem_end in remove_ranges:
            next_pieces: list[tuple[float, float]] = []
            for cur_start, cur_end in pieces:
                if rem_end <= cur_start or rem_start >= cur_end:
                    next_pieces.append((cur_start, cur_end))
                    continue
                if cur_start < rem_start:
                    next_pieces.append((cur_start, rem_start))
                if rem_end < cur_end:
                    next_pieces.append((rem_end, cur_end))
            pieces = next_pieces
        out.extend(piece for piece in pieces if piece[1] > piece[0])
    return out


def _merge_ranges(
    ranges: list[tuple[float, float]],
    *,
    max_gap_sec: float,
) -> list[tuple[float, float]]:
    if not ranges:
        return []
    ordered = sorted(ranges)
    merged = [ordered[0]]
    for start, end in ordered[1:]:
        prev_start, prev_end = merged[-1]
        if start <= prev_end + max_gap_sec:
            merged[-1] = (prev_start, max(prev_end, end))
            continue
        merged.append((start, end))
    return merged


def _pad_ranges(
    ranges: list[tuple[float, float]],
    *,
    pad_sec: float,
    max_end: float,
) -> list[tuple[float, float]]:
    padded = [
        (max(0.0, start - pad_sec), min(max_end, end + pad_sec)) for start, end in ranges
    ]
    return _merge_ranges(padded, max_gap_sec=pad_sec)


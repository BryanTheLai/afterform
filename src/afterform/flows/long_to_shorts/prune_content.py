"""Stage 2.5 - Content pruning inside each selected clip.

This is the HIVE "irrelevant content pruning" sub-task, applied at the
*inner-clip* scale rather than the scene scale. After the clip selector has
chosen 5 x 50-90s windows, we ask the configured LLM to tighten each window by dropping
weak lead-in (throat-clears, false starts, slow setup) and weak tail content
(trailing ramble, fade-out talk).

Design choices:

- **Keep the useful semantic trim stage.** The stage LLM still decides the outer
  in/out window because that is the existing product behavior and it is
  already cached.
- **Add audio-first inner keep ranges.** Silence and filled-pause removal are
  computed from `source_audio.wav`, not ASR text, and written to
  ``Clip.keep_ranges_sec`` for honest downstream concat/subtitle timing.
- **Strict clamping** after the LLM returns, so the final duration always
  respects ``MIN_CLIP_DURATION_SEC`` and any declared hook window is
  preserved.
- **Never fatal.** Any failure (API error, malformed JSON, missing audio
  model, missing clip_id) degrades to no-op for that sub-part of the stage.
"""

from __future__ import annotations

import hashlib
import json
import logging
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Literal, TypeVar

from pydantic import BaseModel, Field, ValidationError

from afterform.schemas import Clip

from afterform.flows.long_to_shorts.audio_pruning import compute_audio_keep_ranges, load_audio_buffer
from afterform.flows.long_to_shorts.select_clips import complete_clip_boundaries
from afterform.config import (
    MAX_CLIP_DURATION_SEC,
    MIN_CLIP_DURATION_SEC,
    PipelineConfig,
)
from afterform.providers.llm import (
    StructuredLlmRequest,
    call_structured_llm,
    resolved_llm_identity,
    resolved_llm_provider,
    resolved_text_model,
)
from afterform.flows.long_to_shorts.load_prompts import content_pruning_system_prompt
from afterform.flows.long_to_shorts.render_window import clip_output_duration

logger = logging.getLogger(__name__)

T = TypeVar("T")

PRUNE_META_VERSION = 3
PRUNE_META_FILENAME = "prune.meta.json"
PRUNE_RAW_FILENAME = "prune_raw.json"
PRUNE_ARTIFACT_FILENAME = "prune.json"
_AUDIO_POLICY_VERSION = 2

LLM_MAX_ATTEMPTS = 3
LLM_RETRY_DELAY_SEC = 2.0

PruneLevel = Literal["off", "conservative", "balanced", "aggressive"]

VALID_LEVELS: tuple[PruneLevel, ...] = ("off", "conservative", "balanced", "aggressive")

# The clip-selection prompt uses `[0.0, 3.0]` as an example / fallback hook
# window. Gemini frequently copies this placeholder verbatim instead of
# localising the real hook, which silently disables Stage 2.5 start-trims for
# every clip (the hook clamp below refuses to trim past `hook_start_sec`, so
# any `trim_start_sec > 0` returned by the prune LLM gets zeroed).
#
# Treat this exact fingerprint as "no real hook" for clamp purposes. The real
# fix is the Stage 2.25 hook detector (``afterform.flows.long_to_shorts.hook_detector``) which
# overwrites the clip's hook fields with a localised window before pruning
# runs. This constant is the belt-and-suspenders guard for the case where
# hook detection is disabled, fails, or cache-hits stale data.
_DEFAULT_HOOK_FINGERPRINT: tuple[float, float] = (0.0, 3.0)
_DEFAULT_HOOK_EPS: float = 1e-3


def _looks_like_default_hook(hook_start: float | None, hook_end: float | None) -> bool:
    """True when the hook window matches the prompt's 0-3s placeholder.

    This is intentionally a narrow, exact-match check so a real hook that
    happens to open at t=0 with a 3.0s window is still respected.
    """
    if hook_start is None or hook_end is None:
        return False
    return (
        abs(hook_start - _DEFAULT_HOOK_FINGERPRINT[0]) < _DEFAULT_HOOK_EPS
        and abs(hook_end - _DEFAULT_HOOK_FINGERPRINT[1]) < _DEFAULT_HOOK_EPS
    )

# Per-level cap on the fraction of the original clip the LLM is allowed to
# trim. Even if the LLM tries to be more eager, we clamp. Final duration is
# additionally clamped to ``MIN_CLIP_DURATION_SEC``.
_MAX_TOTAL_TRIM_PCT: dict[PruneLevel, float] = {
    "off": 0.0,
    "conservative": 0.10,
    "balanced": 0.20,
    "aggressive": 0.35,
}


class _PruneDecision(BaseModel):
    """Per-clip decision returned by the stage LLM (clip-relative seconds)."""

    clip_id: str
    trim_start_sec: float = Field(default=0.0, ge=0.0)
    trim_end_sec: float = Field(default=0.0, ge=0.0)
    reason: str = ""


class _PruneResponse(BaseModel):
    decisions: list[_PruneDecision] = Field(default_factory=list)


@dataclass
class _ClampStats:
    """Diagnostics for why a returned trim got reshaped."""

    clamped_start: bool = False
    clamped_end: bool = False
    hook_protected: bool = False
    min_duration_protected: bool = False
    max_pct_protected: bool = False


def _retry_llm(name: str, fn: Callable[[], T], attempts: int = LLM_MAX_ATTEMPTS) -> T:
    last: Exception | None = None
    for i in range(attempts):
        try:
            return fn()
        except Exception as e:
            last = e
            if i < attempts - 1:
                logger.warning("%s attempt %d/%d failed: %s", name, i + 1, attempts, e)
                time.sleep(LLM_RETRY_DELAY_SEC * (i + 1))
    assert last is not None
    raise last


# ---------------------------------------------------------------------------
# Clamping
# ---------------------------------------------------------------------------


def _clamp_decision(
    clip: Clip,
    trim_start: float,
    trim_end: float,
    *,
    level: PruneLevel,
) -> tuple[float, float, _ClampStats]:
    """Clamp a raw (trim_start, trim_end) pair so the resulting clip is legal.

    Guarantees:
    - ``trim_start`` and ``trim_end`` are non-negative.
    - Final duration (``clip.duration_sec - trim_start - trim_end``) is at
      least ``MIN_CLIP_DURATION_SEC`` (or the original duration, whichever is
      smaller - we never *extend* a clip that was already too short).
    - Combined trim does not exceed the level's allowed fraction of the
      original duration.
    - If ``hook_start_sec`` / ``hook_end_sec`` are set on the clip, the hook
      window stays fully inside the result.
    """
    stats = _ClampStats()
    duration = clip.duration_sec

    ts = max(0.0, float(trim_start))
    te = max(0.0, float(trim_end))
    if ts != trim_start:
        stats.clamped_start = True
    if te != trim_end:
        stats.clamped_end = True

    max_pct = _MAX_TOTAL_TRIM_PCT.get(level, 0.0)
    max_total_trim = duration * max_pct
    if ts + te > max_total_trim:
        scale = max_total_trim / max(ts + te, 1e-9)
        ts = ts * scale
        te = te * scale
        stats.max_pct_protected = True

    # Only protect the hook when the clip carries a *real* localised hook
    # window. The clip-selection LLM frequently echoes the prompt's
    # 0.0-3.0s placeholder, which would otherwise lock ``trim_start`` to 0
    # for every clip and silently disable the entire pruning stage. See
    # ``_looks_like_default_hook`` for the fingerprint rationale.
    hook_is_real = (
        clip.hook_start_sec is not None
        and clip.hook_end_sec is not None
        and not _looks_like_default_hook(clip.hook_start_sec, clip.hook_end_sec)
    )
    if hook_is_real:
        hook_lo = clip.hook_start_sec  # type: ignore[assignment]
        hook_hi = clip.hook_end_sec  # type: ignore[assignment]
        if ts > max(0.0, hook_lo - 0.25):
            ts = max(0.0, hook_lo - 0.25)
            stats.hook_protected = True
        if te > max(0.0, duration - hook_hi - 0.25):
            te = max(0.0, duration - hook_hi - 0.25)
            stats.hook_protected = True

    min_final = min(float(MIN_CLIP_DURATION_SEC), duration)
    max_total_by_min = max(0.0, duration - min_final)
    if ts + te > max_total_by_min:
        overflow = ts + te - max_total_by_min
        te_cut = min(te, overflow)
        te -= te_cut
        overflow -= te_cut
        if overflow > 0:
            ts = max(0.0, ts - overflow)
        stats.min_duration_protected = True

    ts = max(0.0, min(ts, duration))
    te = max(0.0, min(te, duration - ts))
    return ts, te, stats


# Tolerance used when snapping trim boundaries to WhisperX segment edges. A
# 3s window comfortably covers "finish the current sentence" cases without
# materially deviating from what the LLM asked for. Tuned on the reported
# mid-sentence cut in clip 001 of the ``PdVv_vLkUgk`` run (6.38s trim vs a
# sentence that ended ~1.5s later).
_SEGMENT_SNAP_TOLERANCE_SEC: float = 3.0


def _snap_trims_to_segment_boundaries(
    clip: Clip,
    transcript: dict,
    *,
    level: PruneLevel,
    tolerance_sec: float = _SEGMENT_SNAP_TOLERANCE_SEC,
) -> tuple[float, float]:
    """Snap an already-clamped ``(trim_start, trim_end)`` to phrase boundaries.

    WhisperX segments correspond to natural phrase / sentence groupings.
    Landing cuts on segment edges eliminates the "this could be..." class of
    mid-sentence truncation, even when the LLM rounds to an arbitrary
    syllable.

    Direction preference:

    - ``trim_start``: prefer the nearest segment START at-or-after the
      current in-point (trim a hair more to drop lead-in filler). Fallback
      is the nearest segment start behind, within tolerance.
    - ``trim_end``: prefer the nearest segment END at-or-after the current
      out-point (let the sentence finish, keeping MORE content). Fallback
      is the nearest segment end before, within tolerance.

    Safety: the snapped pair is reverted if it would violate
    ``MIN_CLIP_DURATION_SEC``, exceed the level's ``max_pct`` trim cap, or
    eat into a real (non-placeholder) hook window. Snapping can only
    *improve* a decision, never break it.
    """
    ts0 = float(clip.trim_start_sec)
    te0 = float(clip.trim_end_sec)
    if ts0 < 0.05 and te0 < 0.05:
        return ts0, te0

    segs = _segments_within_clip(transcript, clip)
    if not segs:
        return ts0, te0

    duration = clip.duration_sec
    seg_starts = [float(s["start"]) for s in segs]
    seg_ends = [float(s["end"]) for s in segs]

    new_ts = ts0
    if ts0 >= 0.05:
        forward = [s for s in seg_starts if s >= ts0 and (s - ts0) <= tolerance_sec]
        backward = [s for s in seg_starts if s < ts0 and (ts0 - s) <= tolerance_sec]
        if forward:
            new_ts = min(forward)
        elif backward:
            new_ts = max(backward)

    new_te = te0
    if te0 >= 0.05:
        out0 = duration - te0
        forward = [e for e in seg_ends if e >= out0 and (e - out0) <= tolerance_sec]
        backward = [e for e in seg_ends if e < out0 and (out0 - e) <= tolerance_sec]
        if forward:
            new_out = min(forward)
        elif backward:
            new_out = max(backward)
        else:
            new_out = out0
        new_te = max(0.0, duration - new_out)

    new_ts = max(0.0, min(new_ts, duration))
    new_te = max(0.0, min(new_te, duration - new_ts))

    min_final = min(float(MIN_CLIP_DURATION_SEC), duration)
    if duration - new_ts - new_te < min_final - 1e-6:
        return ts0, te0

    max_pct = _MAX_TOTAL_TRIM_PCT.get(level, 0.0)
    if max_pct > 0.0 and (new_ts + new_te) > duration * max_pct + 1e-6:
        return ts0, te0

    if (
        clip.hook_start_sec is not None
        and clip.hook_end_sec is not None
        and not _looks_like_default_hook(clip.hook_start_sec, clip.hook_end_sec)
    ):
        hook_lo = float(clip.hook_start_sec)
        hook_hi = float(clip.hook_end_sec)
        if new_ts > max(0.0, hook_lo - 0.25) + 1e-6:
            return ts0, te0
        if duration - new_te < hook_hi + 0.25 - 1e-6:
            return ts0, te0

    return new_ts, new_te


def apply_prune_decisions(
    clips: list[Clip],
    decisions: list[_PruneDecision],
    *,
    level: PruneLevel,
    transcript: dict | None = None,
) -> list[Clip]:
    """Return new clips with trim_start / trim_end set from LLM decisions.

    Clips whose ``clip_id`` is missing from ``decisions`` are returned with
    trims of 0 / 0 (no-op). Decisions are always clamped; no exception is
    raised if the model returned invalid numbers.

    When ``transcript`` is provided, each clamped trim pair is additionally
    snapped to the nearest WhisperX segment boundary (see
    :func:`_snap_trims_to_segment_boundaries`) so cuts never land
    mid-sentence. The clamp is authoritative -- snapping only ever produces
    an equally-safe boundary, never a looser one.
    """
    by_id = {d.clip_id: d for d in decisions}
    out: list[Clip] = []
    for clip in clips:
        d = by_id.get(clip.clip_id)
        if d is None or level == "off":
            out.append(
                clip.model_copy(
                    update={
                        "trim_start_sec": 0.0,
                        "trim_end_sec": 0.0,
                        "keep_ranges_sec": [],
                    }
                )
            )
            continue
        ts, te, stats = _clamp_decision(
            clip, d.trim_start_sec, d.trim_end_sec, level=level
        )
        # Surface every non-trivial clamp so silent degradations (e.g. a
        # fake hook nuking every trim) are visible in INFO logs, not just
        # buried in ``prune_raw.json``.
        requested = d.trim_start_sec + d.trim_end_sec
        applied = ts + te
        reshaped = (
            stats.hook_protected
            or stats.min_duration_protected
            or stats.max_pct_protected
            or (requested > 0.0 and abs(applied - requested) > 0.05)
        )
        if reshaped:
            logger.info(
                "Clip %s: prune decision clamped (hook=%s min=%s cap=%s) "
                "requested %.2f/%.2f -> applied %.2f/%.2f",
                clip.clip_id,
                stats.hook_protected,
                stats.min_duration_protected,
                stats.max_pct_protected,
                d.trim_start_sec,
                d.trim_end_sec,
                ts,
                te,
            )
        candidate = clip.model_copy(
            update={"trim_start_sec": ts, "trim_end_sec": te, "keep_ranges_sec": []}
        )
        if transcript is not None:
            snapped_ts, snapped_te = _snap_trims_to_segment_boundaries(
                candidate, transcript, level=level
            )
            snapped_te = _complete_trim_end_to_transcript_boundary(
                candidate.model_copy(
                    update={"trim_start_sec": snapped_ts, "trim_end_sec": snapped_te}
                ),
                transcript,
            )
            if abs(snapped_ts - ts) > 1e-3 or abs(snapped_te - te) > 1e-3:
                logger.info(
                    "Clip %s: prune boundaries snapped to segment edges "
                    "%.2f/%.2f -> %.2f/%.2f",
                    clip.clip_id,
                    ts,
                    te,
                    snapped_ts,
                    snapped_te,
                )
                candidate = candidate.model_copy(
                    update={"trim_start_sec": snapped_ts, "trim_end_sec": snapped_te}
                )
        out.append(candidate)
    return out


def apply_audio_keep_ranges(
    clips: list[Clip],
    *,
    source_audio_path: Path,
    transcript: dict | None = None,
    filled_pause_enabled: bool = False,
    require_filled_pause: bool = False,
) -> tuple[list[Clip], dict[str, dict[str, Any]]]:
    """Compute audio-first keep ranges for each clip from the shared WAV."""
    if not source_audio_path.is_file():
        logger.warning(
            "Missing source_audio.wav (%s); skipping audio keep-range pruning.",
            source_audio_path,
        )
        return clips, {
            clip.clip_id: {
                "audio_backend": {"speech": "none", "filled_pause": "none"},
                "warnings": ["source_audio.wav missing; skipped audio keep-range pruning."],
            }
            for clip in clips
        }

    try:
        audio = load_audio_buffer(source_audio_path)
    except Exception as exc:
        logger.warning(
            "Failed to load source_audio.wav (%s); skipping audio keep-range pruning.",
            exc,
        )
        return clips, {
            clip.clip_id: {
                "audio_backend": {"speech": "none", "filled_pause": "none"},
                "warnings": [f"source_audio.wav unreadable; skipped audio keep-range pruning ({exc})."],
            }
            for clip in clips
        }
    diagnostics: dict[str, dict[str, Any]] = {}
    updated: list[Clip] = []
    for clip in clips:
        try:
            result = compute_audio_keep_ranges(
                audio,
                clip,
                filled_pause_enabled=filled_pause_enabled,
                require_filled_pause=require_filled_pause,
            )
            updated_clip = clip.model_copy(update={"keep_ranges_sec": result.keep_ranges_sec})
            if transcript is not None:
                updated_clip = _complete_keep_range_tail(updated_clip, transcript)
            updated.append(updated_clip)
            diagnostics[clip.clip_id] = {
                "outer_window_sec": list(result.outer_window_sec),
                "speech_ranges_sec": [list(rng) for rng in result.speech_ranges_sec],
                "filled_pause_ranges_sec": [list(rng) for rng in result.filled_pause_ranges_sec],
                **result.diagnostics,
            }
        except Exception as exc:
            logger.warning(
                "Clip %s: audio keep-range detection failed (%s); keeping trimmed window only.",
                clip.clip_id,
                exc,
            )
            updated.append(clip.model_copy(update={"keep_ranges_sec": []}))
            diagnostics[clip.clip_id] = {
                "audio_backend": {"speech": "none", "filled_pause": "none"},
                "warnings": [f"audio keep-range detection failed: {exc}"],
            }
    return updated, diagnostics


def _complete_trim_end_to_transcript_boundary(clip: Clip, transcript: dict) -> float:
    """Reduce ``trim_end_sec`` when it would cut before a sentence completes."""
    if clip.trim_end_sec <= 0.05:
        return clip.trim_end_sec
    outer_end = clip.end_time_sec - clip.trim_end_sec
    probe = clip.model_copy(
        update={
            "end_time_sec": outer_end,
            "trim_start_sec": 0.0,
            "trim_end_sec": 0.0,
            "keep_ranges_sec": [],
        }
    )
    [completed] = complete_clip_boundaries([probe], transcript)
    completed_end = min(clip.end_time_sec, completed.end_time_sec)
    return max(0.0, clip.end_time_sec - completed_end)


def _complete_keep_range_tail(clip: Clip, transcript: dict) -> Clip:
    """Extend the final audio keep range when it ends before the words do."""
    if not clip.keep_ranges_sec:
        return clip
    outer_end = max(0.0, min(clip.duration_sec, clip.duration_sec - clip.trim_end_sec))
    last_start, last_end = clip.keep_ranges_sec[-1]
    probe = clip.model_copy(
        update={
            "end_time_sec": clip.start_time_sec + float(last_end),
            "trim_start_sec": 0.0,
            "trim_end_sec": 0.0,
            "keep_ranges_sec": [],
        }
    )
    [completed] = complete_clip_boundaries([probe], transcript)
    completed_rel_end = min(outer_end, completed.end_time_sec - clip.start_time_sec)
    if completed_rel_end <= float(last_end) + 1e-3:
        return clip
    ranges = list(clip.keep_ranges_sec)
    ranges[-1] = (float(last_start), completed_rel_end)
    logger.info(
        "Clip %s: audio keep-range tail extended %.2fs -> %.2fs to finish transcript boundary.",
        clip.clip_id,
        last_end,
        completed_rel_end,
    )
    return clip.model_copy(update={"keep_ranges_sec": ranges})


# ---------------------------------------------------------------------------
# Prompt construction
# ---------------------------------------------------------------------------


def _segments_within_clip(transcript: dict, clip: Clip) -> list[dict]:
    """Return transcript segments that overlap the clip window, with times
    expressed as seconds relative to the clip start.
    """
    s0 = clip.start_time_sec
    s1 = clip.end_time_sec
    lines: list[dict] = []
    for seg in transcript.get("segments", []):
        start = float(seg.get("start", 0.0))
        end = float(seg.get("end", start))
        if end <= s0 or start >= s1:
            continue
        rel_start = max(0.0, start - s0)
        rel_end = min(clip.duration_sec, end - s0)
        if rel_end <= rel_start:
            continue
        lines.append(
            {
                "start": rel_start,
                "end": rel_end,
                "text": (seg.get("text") or "").strip(),
            }
        )
    return lines


def _build_user_message(clips: list[Clip], transcript: dict) -> str:
    """Render a compact textual view of every clip for the LLM user turn."""
    blocks: list[str] = []
    for clip in clips:
        seg_lines = _segments_within_clip(transcript, clip)
        header = (
            f"clip_id: {clip.clip_id}\n"
            f"duration_sec: {clip.duration_sec:.2f}\n"
            f"topic: {clip.topic}"
        )
        if clip.hook_start_sec is not None and clip.hook_end_sec is not None:
            header += (
                f"\nhook_window_sec: [{clip.hook_start_sec:.2f}, {clip.hook_end_sec:.2f}]"
            )
        body = "\n".join(
            f"[{seg['start']:.2f}s - {seg['end']:.2f}s] {seg['text']}" for seg in seg_lines
        )
        if not body:
            body = "(no segments overlap this clip window)"
        blocks.append(f"{header}\n---\n{body}")
    return "\n\n===\n\n".join(blocks)


# ---------------------------------------------------------------------------
# Cache
# ---------------------------------------------------------------------------


def _clips_fingerprint(clips: list[Clip]) -> str:
    """Fingerprint the clip *windows* (not trims, so the cache ignores previous
    prune results when deciding whether to re-ask the LLM).
    """
    payload = json.dumps(
        [
            {
                "id": c.clip_id,
                "s": round(c.start_time_sec, 3),
                "e": round(c.end_time_sec, 3),
                "hs": c.hook_start_sec,
                "he": c.hook_end_sec,
            }
            for c in clips
        ],
        sort_keys=True,
        ensure_ascii=False,
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _audio_fingerprint(audio_path: Path) -> str:
    sha = hashlib.sha256()
    with open(audio_path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            sha.update(chunk)
    return sha.hexdigest()


def _prune_meta(
    *,
    transcript_fp: str,
    clips_fp: str,
    audio_fp: str,
    config: PipelineConfig,
    level: PruneLevel,
) -> dict[str, Any]:
    return {
        "version": PRUNE_META_VERSION,
        "transcript_sha256": transcript_fp,
        "clips_sha256": clips_fp,
        "audio_sha256": audio_fp,
        "llm": resolved_llm_identity(config),
        "prune_level": level,
        "filled_pause_pruning": bool(config.filled_pause_pruning),
        "require_filled_pause_pruning": bool(config.require_filled_pause_pruning),
        "audio_policy_version": _AUDIO_POLICY_VERSION,
    }


def _load_cached_clips(work_dir: Path, clips: list[Clip]) -> list[Clip] | None:
    artifact = work_dir / PRUNE_ARTIFACT_FILENAME
    if not artifact.is_file():
        return None
    try:
        with open(artifact, "r", encoding="utf-8") as f:
            data = json.load(f)
        cached = {item["clip_id"]: item for item in data.get("clips", [])}
    except Exception as e:
        logger.warning("Prune cache artifact unreadable (%s); re-running.", e)
        return None
    out: list[Clip] = []
    for clip in clips:
        cached_c = cached.get(clip.clip_id)
        if cached_c is None:
            return None
        out.append(
            clip.model_copy(
                update={
                    "trim_start_sec": float(cached_c.get("trim_start_sec", 0.0)),
                    "trim_end_sec": float(cached_c.get("trim_end_sec", 0.0)),
                    "keep_ranges_sec": list(cached_c.get("keep_ranges_sec") or []),
                }
            )
        )
    return out


def _write_cache(
    work_dir: Path,
    *,
    pruned: list[Clip],
    meta: dict[str, Any],
    raw_response: str,
    diagnostics: dict[str, dict[str, Any]],
) -> None:
    work_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "version": PRUNE_META_VERSION,
        "clips": [
            {
                "clip_id": c.clip_id,
                "trim_start_sec": c.trim_start_sec,
                "trim_end_sec": c.trim_end_sec,
                "keep_ranges_sec": [list(rng) for rng in c.keep_ranges_sec],
                "diagnostics": diagnostics.get(c.clip_id, {}),
            }
            for c in pruned
        ]
    }
    (work_dir / PRUNE_ARTIFACT_FILENAME).write_text(
        json.dumps(payload, indent=2), encoding="utf-8"
    )
    (work_dir / PRUNE_RAW_FILENAME).write_text(raw_response, encoding="utf-8")
    with open(work_dir / PRUNE_META_FILENAME, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)
        f.write("\n")
    logger.info(
        "Wrote %s, %s and %s",
        PRUNE_META_FILENAME,
        PRUNE_ARTIFACT_FILENAME,
        PRUNE_RAW_FILENAME,
    )


def _prune_cache_valid(
    work_dir: Path,
    *,
    transcript_fp: str,
    clips_fp: str,
    audio_fp: str,
    config: PipelineConfig,
    level: PruneLevel,
) -> bool:
    meta_path = work_dir / PRUNE_META_FILENAME
    if not meta_path.is_file():
        return False
    try:
        with open(meta_path, encoding="utf-8") as f:
            meta = json.load(f)
    except Exception:
        return False
    if meta.get("version") != PRUNE_META_VERSION:
        return False
    if meta.get("transcript_sha256") != transcript_fp:
        return False
    if meta.get("clips_sha256") != clips_fp:
        return False
    if meta.get("audio_sha256") != audio_fp:
        return False
    if meta.get("llm") != resolved_llm_identity(config):
        return False
    if meta.get("prune_level") != level:
        return False
    if bool(meta.get("filled_pause_pruning", False)) != bool(config.filled_pause_pruning):
        return False
    if bool(meta.get("require_filled_pause_pruning", False)) != bool(
        config.require_filled_pause_pruning
    ):
        return False
    if meta.get("audio_policy_version") != _AUDIO_POLICY_VERSION:
        return False
    return True


# ---------------------------------------------------------------------------
# LLM call
# ---------------------------------------------------------------------------


def _parse_decisions(raw_json: str) -> list[_PruneDecision]:
    """Parse a raw JSON response into decisions; bare arrays accepted too."""
    data = json.loads(raw_json)
    if isinstance(data, dict) and "decisions" in data:
        try:
            return _PruneResponse.model_validate(data).decisions
        except ValidationError as e:
            logger.warning("Prune response failed validation: %s", e)
            return []
    if isinstance(data, list):
        decisions: list[_PruneDecision] = []
        for item in data:
            try:
                decisions.append(_PruneDecision.model_validate(item))
            except ValidationError:
                continue
        return decisions
    return []


def request_prune_decisions(
    clips: list[Clip],
    transcript: dict,
    *,
    level: PruneLevel,
    config: PipelineConfig | None = None,
    gemini_model: str | None = None,
) -> tuple[list[_PruneDecision], str]:
    """Call the configured LLM for (potentially) one decision per clip.

    Returns ``(decisions, raw_response)``. ``raw_response`` is the literal
    string the model returned (cached to ``prune_raw.json`` for audit). On
    transport or parse failure this raises; callers should catch and treat as
    no-op.
    """
    if level == "off" or not clips:
        return [], "{\"decisions\": []}"

    system = content_pruning_system_prompt(
        min_dur=MIN_CLIP_DURATION_SEC,
        max_dur=MAX_CLIP_DURATION_SEC,
        level=level,
    )
    user_text = _build_user_message(clips, transcript)

    provider = resolved_llm_provider(config)
    model_name = resolved_text_model(config, model_override=gemini_model)

    def _call() -> str:
        logger.info(
            "%s content pruning (model=%s, level=%s, clips=%d)...",
            provider,
            model_name,
            level,
            len(clips),
        )
        response = call_structured_llm(
            StructuredLlmRequest(
                stage_name="content pruning",
                model=model_name,
                system_instruction=system,
                user_text=user_text,
                temperature=0.2,
                response_schema=_PruneResponse,
            ),
            provider=provider,
        )
        if not response.raw_text and response.parsed is None:
            raise RuntimeError("LLM returned empty response text for content pruning")
        return response.raw_text or response.parsed.model_dump_json()

    raw = _retry_llm("Content pruning", _call)
    decisions = _parse_decisions(raw)
    return decisions, raw


# ---------------------------------------------------------------------------
# Public stage entrypoint (used by pipeline.run_pipeline)
# ---------------------------------------------------------------------------


def run_content_pruning_stage(
    work_dir: Path,
    clips: list[Clip],
    transcript: dict,
    *,
    transcript_fp: str,
    config: PipelineConfig,
) -> list[Clip]:
    """Apply Stage 2.5 pruning to ``clips`` and return the new list.

    - When ``config.prune_level == "off"``, this is a cheap no-op: returns a
      copy of the clips with trim_start/end zeroed.
    - Otherwise, tries the cache first, then calls the configured LLM. A failing call
      degrades to no-op (the pipeline is never killed by Stage 2.5).
    """
    level = _validated_level(config.prune_level)
    if level == "off":
        logger.info("Content pruning disabled (prune_level=off); skipping Stage 2.5.")
        return [
            clip.model_copy(
                update={"trim_start_sec": 0.0, "trim_end_sec": 0.0, "keep_ranges_sec": []}
            )
            for clip in clips
        ]

    clips_fp = _clips_fingerprint(clips)
    source_audio_path = work_dir / "source_audio.wav"
    audio_fp = _audio_fingerprint(source_audio_path) if source_audio_path.is_file() else ""

    if not config.force_content_pruning and _prune_cache_valid(
        work_dir,
        transcript_fp=transcript_fp,
        clips_fp=clips_fp,
        audio_fp=audio_fp,
        config=config,
        level=level,
    ):
        cached = _load_cached_clips(work_dir, clips)
        if cached is not None:
            logger.info(
                "Content pruning cache hit (level=%s, %d clips); skipping LLM.",
                level,
                len(clips),
            )
            return cached

    raw = '{"decisions": []}'
    try:
        decisions, raw = request_prune_decisions(
            clips,
            transcript,
            level=level,
            config=config,
            gemini_model=config.gemini_model,
        )
    except Exception as e:
        logger.warning(
            "Content pruning call failed (%s); continuing with zero outer trims.", e
        )
        decisions = []

    pruned = apply_prune_decisions(
        clips, decisions, level=level, transcript=transcript
    )
    pruned, diagnostics = apply_audio_keep_ranges(
        pruned,
        source_audio_path=source_audio_path,
        transcript=transcript,
        filled_pause_enabled=config.filled_pause_pruning,
        require_filled_pause=config.require_filled_pause_pruning,
    )
    _log_prune_summary(pruned, clips)

    meta = _prune_meta(
        transcript_fp=transcript_fp,
        clips_fp=clips_fp,
        audio_fp=audio_fp,
        config=config,
        level=level,
    )
    try:
        _write_cache(
            work_dir,
            pruned=pruned,
            meta=meta,
            raw_response=raw,
            diagnostics=diagnostics,
        )
    except Exception as e:
        logger.warning("Failed to write prune cache (%s); continuing.", e)
    return pruned


def _validated_level(level: str | None) -> PruneLevel:
    lvl = (level or "balanced").strip().lower()
    if lvl not in VALID_LEVELS:
        logger.warning("Unknown prune_level=%r; falling back to 'balanced'.", level)
        return "balanced"
    return lvl  # type: ignore[return-value]


def _log_prune_summary(pruned: list[Clip], original: list[Clip]) -> None:
    total_before = sum(c.duration_sec for c in original)
    total_after = sum(clip_output_duration(c) for c in pruned)
    removed = total_before - total_after
    pct = (removed / total_before * 100.0) if total_before > 0 else 0.0
    logger.info(
        "Content pruning done: removed %.1fs across %d clips (%.1f%% of total).",
        removed,
        len(pruned),
        pct,
    )
    for c in pruned:
        if c.trim_start_sec > 0 or c.trim_end_sec > 0 or c.keep_ranges_sec:
            final = clip_output_duration(c)
            logger.info(
                "  [%s] trim=%.2fs/%.2fs keep_ranges=%d  %.1fs -> %.1fs",
                c.clip_id,
                c.trim_start_sec,
                c.trim_end_sec,
                len(c.keep_ranges_sec),
                c.duration_sec,
                final,
            )


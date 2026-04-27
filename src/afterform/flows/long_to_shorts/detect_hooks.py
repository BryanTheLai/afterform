"""Stage 2.25 - Hook detection.

The clip-selection LLM returns a ``hook_start_sec`` / ``hook_end_sec`` pair
per clip, but in practice it almost always echoes the ``[0.0, 3.0]``
placeholder from the prompt instead of localising the real hook sentence.
That placeholder is toxic to Stage 2.5 pruning -- the clamp refuses to
trim past ``hook_start_sec``, so every ``trim_start_sec > 0`` the pruner
returns gets zeroed out silently.

This module is a dedicated Stage 2.25 that runs between clip selection and
content pruning. For each clip it:

1. Prepares a clip-relative segment listing (same format as pruning uses).
2. Asks the configured LLM, in one batched JSON call, to localise the hook sentence of
   every clip with `hook_start_sec`, `hook_end_sec`, `hook_text`, `reason`.
3. Validates the returned window against the clip's duration + the "real
   hook" heuristics, then overwrites ``clip.hook_start_sec`` /
   ``clip.hook_end_sec`` on a copy of the clip.

The stage is:

- **Cached** (``hooks.json`` / ``hooks.meta.json`` in ``work_dir``) on
  ``transcript_sha256 + clips_sha256 + llm identity``.
- **Never fatal.** Any failure (API error, malformed JSON, clip not
  returned, window that still looks like the 0.0-3.0 placeholder) falls
  back to the original clip with its original hook -- pruning will then
  skip hook protection via the fingerprint guard in
  :func:`afterform.flows.long_to_shorts.prune_content._looks_like_default_hook`.

The stage writes three artifacts to ``work_dir`` for audit:

- ``hooks.meta.json``: cache key (version, fingerprints, model).
- ``hooks.json``: structured per-clip hook windows actually applied.
- ``hooks_raw.json``: verbatim LLM response text (for prompt tuning).
"""

from __future__ import annotations

import hashlib
import json
import logging
import time
from pathlib import Path
from typing import Any, Callable, TypeVar

from pydantic import BaseModel, Field, ValidationError

from afterform.schemas import Clip

from afterform.config import PipelineConfig
from afterform.flows.long_to_shorts.prune_content import _looks_like_default_hook, _segments_within_clip
from afterform.providers.llm import (
    StructuredLlmRequest,
    call_structured_llm,
    resolved_llm_identity,
    resolved_llm_provider,
    resolved_text_model,
)
from afterform.flows.long_to_shorts.load_prompts import hook_detection_system_prompt

logger = logging.getLogger(__name__)

T = TypeVar("T")

HOOK_META_VERSION = 2
HOOK_META_FILENAME = "hooks.meta.json"
HOOK_ARTIFACT_FILENAME = "hooks.json"
HOOK_RAW_FILENAME = "hooks_raw.json"

LLM_MAX_ATTEMPTS = 3
LLM_RETRY_DELAY_SEC = 2.0

# Hook window validation thresholds. The prompt asks for 1.5-7.0s windows;
# we enforce 1.0-10.0s to be lenient on rounding while still rejecting
# obvious "LLM returned the whole paragraph" mistakes.
_MIN_HOOK_DURATION_SEC = 1.0
_MAX_HOOK_DURATION_SEC = 10.0


class _HookDecision(BaseModel):
    """Per-clip hook window returned by the stage LLM (clip-relative seconds)."""

    clip_id: str
    hook_start_sec: float = Field(ge=0.0)
    hook_end_sec: float = Field(ge=0.0)
    hook_text: str = ""
    reason: str = ""


class _HookResponse(BaseModel):
    hooks: list[_HookDecision] = Field(default_factory=list)


def _retry_llm(name: str, fn: Callable[[], T], attempts: int = LLM_MAX_ATTEMPTS) -> T:
    last: Exception | None = None
    for i in range(attempts):
        try:
            return fn()
        except Exception as e:  # noqa: BLE001 - rethrown below
            last = e
            if i < attempts - 1:
                logger.warning("%s attempt %d/%d failed: %s", name, i + 1, attempts, e)
                time.sleep(LLM_RETRY_DELAY_SEC * (i + 1))
    assert last is not None
    raise last


# ---------------------------------------------------------------------------
# Prompt construction
# ---------------------------------------------------------------------------


def _build_user_message(clips: list[Clip], transcript: dict) -> str:
    """Render clip-relative segments + selector-guessed hook text for each clip."""
    blocks: list[str] = []
    for clip in clips:
        segs = _segments_within_clip(transcript, clip)
        header_lines = [
            f"clip_id: {clip.clip_id}",
            f"duration_sec: {clip.duration_sec:.2f}",
            f"topic: {clip.topic}",
        ]
        if clip.viral_hook:
            header_lines.append(f"viral_hook_text: {clip.viral_hook}")
        if clip.hook_start_sec is not None and clip.hook_end_sec is not None:
            header_lines.append(
                f"selector_hook_window_sec: [{clip.hook_start_sec:.2f}, "
                f"{clip.hook_end_sec:.2f}] (may be a placeholder; verify)"
            )
        header = "\n".join(header_lines)
        body = "\n".join(
            f"[{seg['start']:.2f}s - {seg['end']:.2f}s] {seg['text']}" for seg in segs
        )
        if not body:
            body = "(no segments overlap this clip window)"
        blocks.append(f"{header}\n---\n{body}")
    return "\n\n===\n\n".join(blocks)


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------


def _validate_hook_window(
    clip: Clip, hook_start: float, hook_end: float
) -> tuple[float, float] | None:
    """Return a valid (hook_start, hook_end) or None if rejected.

    Rules:
    - ``0 <= hook_start < hook_end <= duration_sec``
    - hook duration between ``_MIN_HOOK_DURATION_SEC`` and ``_MAX_HOOK_DURATION_SEC``
    - NOT the ``(0.0, 3.0)`` placeholder fingerprint (we'd rather keep the
      selector's value untouched than re-apply the same fake hook).
    """
    if hook_start < 0.0 or hook_end <= hook_start:
        return None
    if hook_end > clip.duration_sec + 1e-3:
        # Clamp trailing rounding to duration; reject anything beyond.
        if hook_end - clip.duration_sec > 0.5:
            return None
        hook_end = clip.duration_sec
    dur = hook_end - hook_start
    if dur < _MIN_HOOK_DURATION_SEC or dur > _MAX_HOOK_DURATION_SEC:
        return None
    if _looks_like_default_hook(hook_start, hook_end):
        return None
    return float(hook_start), float(hook_end)


# ---------------------------------------------------------------------------
# Apply decisions -> new clips
# ---------------------------------------------------------------------------


def apply_hook_decisions(
    clips: list[Clip],
    decisions: list[_HookDecision],
) -> list[Clip]:
    """Return new clips whose hook fields reflect validated decisions.

    Clips without a matching valid decision are returned unchanged (their
    original hook metadata, placeholder or not, is preserved).
    """
    by_id = {d.clip_id: d for d in decisions}
    out: list[Clip] = []
    changed = 0
    rejected = 0
    for clip in clips:
        d = by_id.get(clip.clip_id)
        if d is None:
            out.append(clip)
            continue
        validated = _validate_hook_window(clip, d.hook_start_sec, d.hook_end_sec)
        if validated is None:
            logger.info(
                "Clip %s: rejected hook window [%.2f, %.2f] (failed validation); "
                "keeping selector hook.",
                clip.clip_id,
                d.hook_start_sec,
                d.hook_end_sec,
            )
            rejected += 1
            out.append(clip)
            continue
        hs, he = validated
        if (
            clip.hook_start_sec is not None
            and clip.hook_end_sec is not None
            and abs(clip.hook_start_sec - hs) < 1e-3
            and abs(clip.hook_end_sec - he) < 1e-3
        ):
            out.append(clip)
            continue
        changed += 1
        logger.info(
            "Clip %s: hook set to [%.2f, %.2f] (was [%s, %s]) -- %s",
            clip.clip_id,
            hs,
            he,
            f"{clip.hook_start_sec:.2f}" if clip.hook_start_sec is not None else "None",
            f"{clip.hook_end_sec:.2f}" if clip.hook_end_sec is not None else "None",
            d.reason[:120] if d.reason else "(no reason)",
        )
        out.append(
            clip.model_copy(update={"hook_start_sec": hs, "hook_end_sec": he})
        )
    logger.info(
        "Hook detection: updated %d / %d clips (%d rejected, %d kept as-is).",
        changed,
        len(clips),
        rejected,
        len(clips) - changed - rejected,
    )
    return out


# ---------------------------------------------------------------------------
# Cache
# ---------------------------------------------------------------------------


def _clips_fingerprint(clips: list[Clip]) -> str:
    payload = json.dumps(
        [
            {"id": c.clip_id, "s": round(c.start_time_sec, 3), "e": round(c.end_time_sec, 3)}
            for c in clips
        ],
        sort_keys=True,
        ensure_ascii=False,
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _hook_meta(
    *,
    transcript_fp: str,
    clips_fp: str,
    config: PipelineConfig,
) -> dict[str, Any]:
    return {
        "version": HOOK_META_VERSION,
        "transcript_sha256": transcript_fp,
        "clips_sha256": clips_fp,
        "llm": resolved_llm_identity(config),
    }


def _hook_cache_valid(
    work_dir: Path,
    *,
    transcript_fp: str,
    clips_fp: str,
    config: PipelineConfig,
) -> bool:
    meta_path = work_dir / HOOK_META_FILENAME
    if not meta_path.is_file():
        return False
    try:
        with open(meta_path, encoding="utf-8") as f:
            meta = json.load(f)
    except Exception:
        return False
    if meta.get("version") != HOOK_META_VERSION:
        return False
    if meta.get("transcript_sha256") != transcript_fp:
        return False
    if meta.get("clips_sha256") != clips_fp:
        return False
    if meta.get("llm") != resolved_llm_identity(config):
        return False
    return True


def _load_cached_hooks(
    work_dir: Path, clips: list[Clip]
) -> list[Clip] | None:
    artifact = work_dir / HOOK_ARTIFACT_FILENAME
    if not artifact.is_file():
        return None
    try:
        with open(artifact, "r", encoding="utf-8") as f:
            data = json.load(f)
        cached = {item["clip_id"]: item for item in data.get("hooks", [])}
    except Exception as e:  # noqa: BLE001 - surfaced as warning below
        logger.warning("Hook cache artifact unreadable (%s); re-running.", e)
        return None
    out: list[Clip] = []
    for clip in clips:
        c = cached.get(clip.clip_id)
        if c is None:
            out.append(clip)
            continue
        hs = c.get("hook_start_sec")
        he = c.get("hook_end_sec")
        if hs is None or he is None:
            out.append(clip)
            continue
        out.append(
            clip.model_copy(
                update={"hook_start_sec": float(hs), "hook_end_sec": float(he)}
            )
        )
    return out


def _write_cache(
    work_dir: Path,
    *,
    clips_with_hooks: list[Clip],
    decisions: list[_HookDecision],
    meta: dict[str, Any],
    raw_response: str,
) -> None:
    work_dir.mkdir(parents=True, exist_ok=True)
    reasons = {d.clip_id: d for d in decisions}
    payload = {
        "hooks": [
            {
                "clip_id": c.clip_id,
                "hook_start_sec": c.hook_start_sec,
                "hook_end_sec": c.hook_end_sec,
                "hook_text": (reasons.get(c.clip_id).hook_text if reasons.get(c.clip_id) else ""),
                "reason": (reasons.get(c.clip_id).reason if reasons.get(c.clip_id) else ""),
            }
            for c in clips_with_hooks
        ]
    }
    (work_dir / HOOK_ARTIFACT_FILENAME).write_text(
        json.dumps(payload, indent=2), encoding="utf-8"
    )
    (work_dir / HOOK_RAW_FILENAME).write_text(raw_response, encoding="utf-8")
    with open(work_dir / HOOK_META_FILENAME, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)
        f.write("\n")
    logger.info(
        "Wrote %s, %s and %s",
        HOOK_META_FILENAME,
        HOOK_ARTIFACT_FILENAME,
        HOOK_RAW_FILENAME,
    )


# ---------------------------------------------------------------------------
# LLM call
# ---------------------------------------------------------------------------


def _parse_decisions(raw_json: str) -> list[_HookDecision]:
    data = json.loads(raw_json)
    if isinstance(data, dict) and "hooks" in data:
        try:
            return _HookResponse.model_validate(data).hooks
        except ValidationError as e:
            logger.warning("Hook response failed validation: %s", e)
            return []
    if isinstance(data, list):
        out: list[_HookDecision] = []
        for item in data:
            try:
                out.append(_HookDecision.model_validate(item))
            except ValidationError:
                continue
        return out
    return []


def request_hook_decisions(
    clips: list[Clip],
    transcript: dict,
    *,
    config: PipelineConfig | None = None,
    gemini_model: str | None = None,
) -> tuple[list[_HookDecision], str]:
    """Ask the configured LLM to localise the hook sentence for each clip.

    Returns ``(decisions, raw_response)``. ``raw_response`` is the literal
    JSON text from the model (cached to ``hooks_raw.json`` for audit). On
    transport/parse failure this raises; callers should catch and treat as
    no-op.
    """
    if not clips:
        return [], '{"hooks": []}'

    system = hook_detection_system_prompt()
    user_text = _build_user_message(clips, transcript)

    provider = resolved_llm_provider(config)
    model_name = resolved_text_model(config, model_override=gemini_model)

    def _call() -> str:
        logger.info(
            "%s hook detection (model=%s, clips=%d)...",
            provider,
            model_name,
            len(clips),
        )
        response = call_structured_llm(
            StructuredLlmRequest(
                stage_name="hook detection",
                model=model_name,
                system_instruction=system,
                user_text=user_text,
                temperature=0.2,
                response_schema=_HookResponse,
            ),
            provider=provider,
        )
        if not response.raw_text and response.parsed is None:
            raise RuntimeError("LLM returned empty response text for hook detection")
        return response.raw_text or response.parsed.model_dump_json()

    raw = _retry_llm("Hook detection", _call)
    decisions = _parse_decisions(raw)
    return decisions, raw


# ---------------------------------------------------------------------------
# Public stage entrypoint
# ---------------------------------------------------------------------------


def run_hook_detection_stage(
    work_dir: Path,
    clips: list[Clip],
    transcript: dict,
    *,
    transcript_fp: str,
    config: PipelineConfig,
) -> list[Clip]:
    """Run Stage 2.25 hook detection and return clips with localised hooks.

    - Disabled (``config.detect_hooks is False``): return clips unchanged.
    - Cache hit: read ``hooks.json`` and apply cached windows.
    - LLM failure: log a warning and return clips unchanged. The downstream
      content pruner's fingerprint guard will treat any remaining placeholder
      hooks as "no hook" so pruning still runs.
    """
    if not config.detect_hooks:
        logger.info("Hook detection disabled (detect_hooks=False); skipping Stage 2.25.")
        return clips
    if not clips:
        return clips

    clips_fp = _clips_fingerprint(clips)

    if not config.force_hook_detection and _hook_cache_valid(
        work_dir,
        transcript_fp=transcript_fp,
        clips_fp=clips_fp,
        config=config,
    ):
        cached = _load_cached_hooks(work_dir, clips)
        if cached is not None:
            logger.info(
                "Hook detection cache hit (%d clips); skipping LLM.", len(clips)
            )
            return cached

    try:
        decisions, raw = request_hook_decisions(
            clips,
            transcript,
            config=config,
            gemini_model=config.gemini_model,
        )
    except Exception as e:  # noqa: BLE001 - pipeline must not die here
        logger.warning(
            "Hook detection call failed (%s); continuing with selector hooks. "
            "Content pruning will treat any [0.0, 3.0] placeholder as 'no hook'.",
            e,
        )
        return clips

    updated = apply_hook_decisions(clips, decisions)

    meta = _hook_meta(
        transcript_fp=transcript_fp, clips_fp=clips_fp, config=config
    )
    try:
        _write_cache(
            work_dir,
            clips_with_hooks=updated,
            decisions=decisions,
            meta=meta,
            raw_response=raw,
        )
    except Exception as e:  # noqa: BLE001 - cache failure is not fatal
        logger.warning("Failed to write hook cache (%s); continuing.", e)

    return updated


"""Persist clip-selection output and cache the kept set honestly.

The clip-selection stage has two distinct layers of reuse:

1. Transcript + provider/model unchanged -> the raw candidate pool can be reused.
2. Ranking policy unchanged -> the already-kept `clips.json` can be reused.

If only the ranking policy changes, we should re-rank the cached raw LLM
pool instead of paying for another model call.
"""

from __future__ import annotations

import hashlib
import json
import logging
from pathlib import Path
from typing import Any

from afterform.config import PipelineConfig
from afterform.providers.llm import resolved_llm_identity, resolved_llm_provider, resolved_text_model

logger = logging.getLogger(__name__)

# v4: provider-aware llm identity. v3: explicit ranking-policy fingerprint. v2: Gemini-only meta
# (no llm_provider). v1 legacy supported in cache_valid.
CURRENT_META_VERSION = 4
META_FILENAME = "clips.meta.json"
RAW_FILENAME = "clip_selection_raw.json"


def transcript_fingerprint(transcript: dict) -> str:
    payload = json.dumps(transcript, sort_keys=True, ensure_ascii=False)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def clip_selection_policy(config: PipelineConfig) -> dict[str, Any]:
    """Return the cache-significant post-LLM ranking policy."""
    try:
        from afterform.flows.long_to_shorts.select_clips import (
            CLIP_SELECTION_POLICY_VERSION,
            CLIP_SELECTION_RULE_WEIGHTS,
        )
    except Exception:
        return {
            "version": 0,
            "candidate_count": int(config.clip_selection_candidate_count),
            "quality_threshold": float(config.clip_selection_quality_threshold),
            "min_kept": int(config.clip_selection_min_kept),
            "max_kept": int(config.clip_selection_max_kept),
        }

    return {
        "version": int(CLIP_SELECTION_POLICY_VERSION),
        "candidate_count": int(config.clip_selection_candidate_count),
        "quality_threshold": float(config.clip_selection_quality_threshold),
        "min_kept": int(config.clip_selection_min_kept),
        "max_kept": int(config.clip_selection_max_kept),
        "rule_weights": dict(CLIP_SELECTION_RULE_WEIGHTS),
    }


def policy_fingerprint(config: PipelineConfig) -> str:
    payload = json.dumps(
        clip_selection_policy(config),
        sort_keys=True,
        ensure_ascii=False,
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def load_meta(work_dir: Path) -> dict[str, Any] | None:
    path = work_dir / META_FILENAME
    if not path.is_file():
        return None
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def load_raw_response(work_dir: Path) -> str | None:
    path = work_dir / RAW_FILENAME
    if not path.is_file():
        return None
    return path.read_text(encoding="utf-8")


def model_inputs_match(meta: dict[str, Any], fingerprint: str, config: PipelineConfig) -> bool:
    if meta.get("transcript_sha256") != fingerprint:
        return False
    ver = meta.get("version", 1)
    if ver >= CURRENT_META_VERSION:
        return meta.get("llm") == resolved_llm_identity(config)
    # Legacy v1: had llm_provider + model fields
    if meta.get("llm_provider") == "openai":
        return False
    if resolved_llm_provider(config) != "gemini":
        return False
    return meta.get("gemini_model") == resolved_text_model(config)


def ranking_policy_matches(meta: dict[str, Any], config: PipelineConfig) -> bool:
    ver = meta.get("version", 1)
    if ver < CURRENT_META_VERSION:
        return False
    return meta.get("ranking_policy_sha256") == policy_fingerprint(config)


def should_rerank(meta: dict[str, Any], fingerprint: str, config: PipelineConfig) -> bool:
    return model_inputs_match(meta, fingerprint, config) and not ranking_policy_matches(
        meta, config
    )


def cache_valid(meta: dict[str, Any], fingerprint: str, config: PipelineConfig) -> bool:
    return model_inputs_match(meta, fingerprint, config) and ranking_policy_matches(meta, config)


def write_artifacts(
    work_dir: Path,
    *,
    transcript: dict,
    config: PipelineConfig,
    raw_response: str,
) -> None:
    work_dir.mkdir(parents=True, exist_ok=True)
    fp = transcript_fingerprint(transcript)
    meta: dict[str, Any] = {
        "version": CURRENT_META_VERSION,
        "transcript_sha256": fp,
        "llm": resolved_llm_identity(config),
        "ranking_policy_sha256": policy_fingerprint(config),
        "ranking_policy": clip_selection_policy(config),
    }
    (work_dir / RAW_FILENAME).write_text(raw_response, encoding="utf-8")
    with open(work_dir / META_FILENAME, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)
        f.write("\n")
    logger.info("Wrote %s and %s", META_FILENAME, RAW_FILENAME)


"""Step 2 - Clip selection via a swappable structured LLM provider."""

from __future__ import annotations

import json
import logging
import re
import time
from pathlib import Path
from typing import Any, Callable, TypeVar

from pydantic import BaseModel, Field

from afterform.schemas import Clip, ClipPlan, RuleScore

from afterform.config import (
    MAX_CLIP_DURATION_SEC,
    MIN_CLIP_DURATION_SEC,
    PipelineConfig,
    TARGET_CLIP_COUNT,
)
from afterform.providers.llm import (
    ReasoningEffort,
    StructuredLlmRequest,
    TextVerbosity,
    call_structured_llm,
    resolved_llm_provider,
    resolved_text_model,
)
from afterform.flows.long_to_shorts.load_prompts import clip_selection_prompts

logger = logging.getLogger(__name__)

T = TypeVar("T")

LLM_MAX_ATTEMPTS = 3
LLM_RETRY_DELAY_SEC = 2.0

# Over-generation defaults (also exposed via PipelineConfig so callers can
# override per-run without touching code). Rationale:
#
# - Ask Gemini for a *pool* of ~12 candidates at temperature 0.7 so the model
#   considers a wider slice of the transcript instead of locking onto the
#   first 5 obvious ones. More candidates -> more chance the actual gold
#   nugget is in the list.
# - Then rank by ``virality_score`` and keep everything >= threshold, but
#   always keep at least ``min_kept`` and at most ``max_kept`` clips. This
#   lets a single strong clip survive a weak transcript ("keep the best 5
#   even if no one clears the bar") AND lets an exceptionally rich
#   transcript ship 7-8 strong shorts instead of artificially capping at 5.
DEFAULT_CANDIDATE_COUNT = 12
DEFAULT_QUALITY_THRESHOLD = 0.70
DEFAULT_MIN_KEPT = TARGET_CLIP_COUNT
DEFAULT_MAX_KEPT = 8
# Higher than the old 0.3 so the pool is meaningfully different from
# "the same five most-obvious clips every run". Still well below 1.0 so we
# do not get word-salad IDs or timestamps.
DEFAULT_CANDIDATE_TEMPERATURE = 0.7
DEFAULT_SELECTION_MODE = "curated"
CLIP_SELECTION_MIN_OUTPUT_TOKENS = 4096
CLIP_SELECTION_OUTPUT_TOKENS_PER_CANDIDATE = 4000
CLIP_SELECTION_MAX_OUTPUT_TOKENS = 32000

# Operator-visible ranking contract. If these weights change, the kept set
# can change even when Gemini returns the same raw candidate pool, so the
# clip-selection cache must stop trusting the old `clips.json`.
CLIP_SELECTION_POLICY_VERSION = 4
CLIP_SELECTION_DEDUPE_POLICY_VERSION = 1
CLIP_SELECTION_RULE_WEIGHTS: dict[str, float] = {
    "hook_strength": 0.30,
    "counterintuitive_claim": 0.30,
    "chart_reference": 0.20,
    "self_contained": 0.15,
    "named_entity": 0.05,
}

_BOUNDARY_EPS_SEC = 0.05
_IMMEDIATE_CONTINUATION_GAP_SEC = 0.35
_DANGLING_CONTINUATION_GAP_SEC = 1.25
_MAX_BOUNDARY_EXTENSION_SEC = 30.0
_DANGLING_END_RE = re.compile(
    r"(?:[,;:]|--|-|\b(?:and|because|but|for|if|of|or|so|that|the|to|when|where|which|while|with))$",
    re.IGNORECASE,
)
_CONTINUATION_START_RE = re.compile(
    r"^(?:and|because|but|so|that|then|therefore|this|these|those|it|they|which)\b",
    re.IGNORECASE,
)
_CONTINUATION_STOPWORDS = {
    "about",
    "after",
    "also",
    "are",
    "because",
    "being",
    "but",
    "can",
    "for",
    "from",
    "have",
    "into",
    "just",
    "more",
    "not",
    "one",
    "that",
    "the",
    "their",
    "them",
    "then",
    "there",
    "they",
    "this",
    "were",
    "what",
    "when",
    "with",
    "would",
    "you",
    "your",
}


class _ClipSelectionCandidate(BaseModel):
    """Structured Gemini candidate before we coerce it to the shared Clip schema."""

    clip_id: str
    topic: str
    start_time_sec: float = Field(ge=0.0)
    end_time_sec: float = Field(gt=0.0)
    viral_hook: str = ""
    virality_score: float = Field(default=0.0, ge=0.0, le=1.0)
    transcript: str = ""
    suggested_overlay_title: str = ""
    hook_start_sec: float | None = None
    hook_end_sec: float | None = None
    trim_start_sec: float = Field(default=0.0, ge=0.0)
    trim_end_sec: float = Field(default=0.0, ge=0.0)
    shorts_title: str = ""
    description: str = ""
    hashtags: list[str] = Field(default_factory=list)
    layout_hint: str | None = None
    needs_review: bool = False
    review_reason: str = ""
    selection_reason: str = ""
    rule_scores: list[RuleScore] = Field(default_factory=list)


class ClipSelectionResponse(BaseModel):
    clips: list[_ClipSelectionCandidate] = Field(default_factory=list)


class DedupeDecision(BaseModel):
    kept_clip_id: str
    dropped_clip_id: str
    reason: str
    score_delta: float


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


def build_prompt(
    transcript: dict, *, candidate_count: int = DEFAULT_CANDIDATE_COUNT
) -> tuple[str, str]:
    """Return ``(system_prompt, user_message)`` for the clip-selector LLM call.

    ``candidate_count`` is the size of the candidate POOL we ask Gemini for.
    A downstream ranker (``rank_and_filter_clips``) then keeps the top
    clips that clear the quality threshold. Defaults preserve the previous
    visible output (5 clips) when the pool is narrow.
    """
    lines = []
    for seg in transcript.get("segments", []):
        start = seg.get("start", 0)
        end = seg.get("end", 0)
        text = seg.get("text", "").strip()
        lines.append(f"[{start:.1f}s - {end:.1f}s] {text}")

    transcript_text = "\n".join(lines)

    system, user = clip_selection_prompts(
        transcript_text=transcript_text,
        min_dur=MIN_CLIP_DURATION_SEC,
        max_dur=MAX_CLIP_DURATION_SEC,
        count=candidate_count,
    )
    return system, user


def _composite_rule_score(rule_scores: list[RuleScore]) -> float | None:
    if not rule_scores:
        return None
    by_id = {r.rule_id: max(0.0, min(1.0, float(r.score))) for r in rule_scores}
    total_weight = sum(CLIP_SELECTION_RULE_WEIGHTS.values())
    if total_weight <= 0.0:
        return None
    composite = sum(
        CLIP_SELECTION_RULE_WEIGHTS[rule_id] * by_id.get(rule_id, 0.0)
        for rule_id in CLIP_SELECTION_RULE_WEIGHTS
    )
    return round(composite / total_weight, 4)


def _default_selection_reason(rule_scores: list[RuleScore], score: float) -> str:
    if not rule_scores:
        return f"Fallback virality_score {score:.2f} (no structured rule scores provided)."
    parts = []
    by_id = {r.rule_id: r for r in rule_scores}
    for rule_id, weight in CLIP_SELECTION_RULE_WEIGHTS.items():
        item = by_id.get(rule_id)
        if item is None:
            parts.append(f"{weight:.2f}*{rule_id}=0.00")
            continue
        parts.append(f"{weight:.2f}*{rule_id}={item.score:.2f}")
    return f"Composite {score:.2f} from " + ", ".join(parts)


def clip_selection_max_output_tokens(candidate_count: int) -> int:
    count = max(1, int(candidate_count))
    return min(
        CLIP_SELECTION_MAX_OUTPUT_TOKENS,
        max(
            CLIP_SELECTION_MIN_OUTPUT_TOKENS,
            count * CLIP_SELECTION_OUTPUT_TOKENS_PER_CANDIDATE,
        ),
    )


def _openai_reasoning_options(
    provider: str,
    model_name: str,
) -> tuple[ReasoningEffort | None, TextVerbosity | None]:
    if provider not in {"openai", "azure"}:
        return None, None
    if not model_name.strip().lower().startswith("gpt-5"):
        return None, None
    return "none", "low"


def _candidate_to_clip(candidate: _ClipSelectionCandidate) -> Clip:
    payload = candidate.model_dump()
    payload.pop("clip_id", None)
    score = _composite_rule_score(candidate.rule_scores)
    final_score = candidate.virality_score if score is None else score
    selection_reason = candidate.selection_reason or _default_selection_reason(
        candidate.rule_scores, final_score
    )
    payload["clip_id"] = candidate.clip_id
    payload["virality_score"] = final_score
    payload["selection_reason"] = selection_reason
    return Clip.model_validate(payload)


def _normalised_segments(transcript: dict) -> list[dict[str, Any]]:
    segments: list[dict[str, Any]] = []
    for seg in transcript.get("segments", []):
        try:
            start = float(seg.get("start", 0.0))
            end = float(seg.get("end", start))
        except (TypeError, ValueError):
            continue
        text = str(seg.get("text") or "").strip()
        if end <= start or not text:
            continue
        segments.append({"start": start, "end": end, "text": text, "words": seg.get("words") or []})
    return sorted(segments, key=lambda item: float(item["start"]))


def _segment_idx_at_end(
    segments: list[dict[str, Any]], end_time_sec: float
) -> int | None:
    candidate: int | None = None
    for idx, seg in enumerate(segments):
        start = float(seg["start"])
        end = float(seg["end"])
        if end_time_sec + _BOUNDARY_EPS_SEC < start:
            break
        if start - _BOUNDARY_EPS_SEC <= end_time_sec <= end + _BOUNDARY_EPS_SEC:
            return idx
        if end <= end_time_sec + _BOUNDARY_EPS_SEC:
            candidate = idx
    return candidate


def _is_dangling_boundary(text: str) -> bool:
    cleaned = text.strip().rstrip("\"')")
    if not cleaned:
        return False
    if _DANGLING_END_RE.search(cleaned):
        return True
    return cleaned[-1] not in ".?!"


def _meaningful_tokens(text: str) -> set[str]:
    tokens = re.findall(r"[a-z0-9]+", text.lower())
    return {token for token in tokens if len(token) > 2 and token not in _CONTINUATION_STOPWORDS}


def _is_continuation_segment(current_text: str, next_text: str) -> bool:
    cleaned_next = next_text.strip()
    if _CONTINUATION_START_RE.match(cleaned_next):
        return True
    return len(_meaningful_tokens(current_text) & _meaningful_tokens(cleaned_next)) >= 2


def _word_bounds(word: dict[str, Any]) -> tuple[float, float] | None:
    try:
        start = float(word.get("start", word.get("start_time", 0.0)))
        end = float(word.get("end", word.get("end_time", start)))
    except (TypeError, ValueError):
        return None
    if end < start:
        return None
    return start, end


def _word_text(word: dict[str, Any]) -> str:
    return str(word.get("word") or word.get("text") or "").strip()


def _first_sentence_end_in_segment(seg: dict[str, Any], after_time_sec: float) -> float | None:
    timed_words = []
    for word in seg.get("words") or []:
        if not isinstance(word, dict):
            continue
        bounds = _word_bounds(word)
        if bounds is None:
            continue
        timed_words.append((word, bounds))

    text_tokens = re.findall(r"\S+", str(seg.get("text") or ""))
    if text_tokens and len(text_tokens) == len(timed_words):
        first_idx = 0
        for idx, (_word, (_start, end)) in enumerate(timed_words):
            if end > after_time_sec + _BOUNDARY_EPS_SEC:
                first_idx = idx
                break
        for idx in range(first_idx, len(text_tokens)):
            token = text_tokens[idx].rstrip("\"')")
            if token.endswith((".", "?", "!")):
                return timed_words[idx][1][1]

    for word, (_start, end) in timed_words:
        if end <= after_time_sec + _BOUNDARY_EPS_SEC:
            continue
        token = _word_text(word).rstrip("\"')")
        if token.endswith((".", "?", "!")):
            return end
    return None


def _segment_text_for_window(seg: dict[str, Any], start_time_sec: float, end_time_sec: float) -> str:
    words = []
    for word in seg.get("words") or []:
        if not isinstance(word, dict):
            continue
        bounds = _word_bounds(word)
        if bounds is None:
            continue
        start, end = bounds
        if end <= start_time_sec or start >= end_time_sec:
            continue
        text = _word_text(word)
        if text:
            words.append(text)
    if words:
        return " ".join(words)
    return str(seg["text"]).strip()


def _clip_transcript_for_window(
    segments: list[dict[str, Any]], start_time_sec: float, end_time_sec: float
) -> str:
    texts = [
        _segment_text_for_window(seg, start_time_sec, end_time_sec)
        for seg in segments
        if float(seg["end"]) > start_time_sec and float(seg["start"]) < end_time_sec
    ]
    return " ".join(text for text in texts if text)


def _complete_clip_boundary(
    clip: Clip, segments: list[dict[str, Any]]
) -> Clip:
    idx = _segment_idx_at_end(segments, clip.end_time_sec)
    if idx is None:
        return clip

    original_end = float(clip.end_time_sec)
    end = original_end
    current = segments[idx]
    if end < float(current["end"]) - _BOUNDARY_EPS_SEC:
        end = float(current["end"])

    while idx + 1 < len(segments):
        current_text = str(segments[idx]["text"])
        next_seg = segments[idx + 1]
        next_text = str(next_seg["text"])
        gap = float(next_seg["start"]) - end
        current_is_dangling = _is_dangling_boundary(current_text)
        max_gap = (
            _DANGLING_CONTINUATION_GAP_SEC
            if current_is_dangling
            else _IMMEDIATE_CONTINUATION_GAP_SEC
        )
        if gap > max_gap:
            break
        if not current_is_dangling and not _is_continuation_segment(current_text, next_text):
            break
        sentence_end = _first_sentence_end_in_segment(next_seg, end)
        next_end = float(next_seg["end"]) if sentence_end is None else sentence_end
        over_extension_budget = next_end - original_end > _MAX_BOUNDARY_EXTENSION_SEC
        if over_extension_budget and not current_is_dangling:
            break
        end = max(end, next_end)
        if sentence_end is not None:
            break
        idx += 1
        if not _is_dangling_boundary(str(next_seg["text"])):
            next_gap = (
                float(segments[idx + 1]["start"]) - end
                if idx + 1 < len(segments)
                else float("inf")
            )
            if next_gap > _IMMEDIATE_CONTINUATION_GAP_SEC:
                break

    if end <= original_end + _BOUNDARY_EPS_SEC:
        return clip

    transcript = _clip_transcript_for_window(segments, clip.start_time_sec, end)
    logger.info(
        "Clip %s: extended end %.2fs -> %.2fs to finish transcript boundary.",
        clip.clip_id,
        original_end,
        end,
    )
    return clip.model_copy(update={"end_time_sec": end, "transcript": transcript})


def complete_clip_boundaries(clips: list[Clip], transcript: dict) -> list[Clip]:
    """Extend selected clip ends to finish the active sentence or point.

    The LLM still picks the interesting window. This deterministic pass repairs
    unsafe endpoints where the speaker is visibly continuing, and it may exceed
    ``MAX_CLIP_DURATION_SEC`` when that is required to avoid cutting a sentence.
    """
    segments = _normalised_segments(transcript)
    if not segments:
        return clips
    return [_complete_clip_boundary(clip, segments) for clip in clips]


def _normalise_similarity_text(text: str) -> str:
    return re.sub(r"\s+", " ", re.sub(r"[^a-z0-9\s]", " ", text.lower())).strip()


def _token_set(text: str) -> set[str]:
    return {token for token in _normalise_similarity_text(text).split() if token}


def _jaccard(a: str, b: str) -> float:
    a_tokens = _token_set(a)
    b_tokens = _token_set(b)
    if not a_tokens or not b_tokens:
        return 0.0
    return len(a_tokens & b_tokens) / len(a_tokens | b_tokens)


def _time_iou(a: Clip, b: Clip) -> float:
    overlap = max(0.0, min(a.end_time_sec, b.end_time_sec) - max(a.start_time_sec, b.start_time_sec))
    union = max(a.end_time_sec, b.end_time_sec) - min(a.start_time_sec, b.start_time_sec)
    if union <= 0.0:
        return 0.0
    return overlap / union


def _contains_window(a: Clip, b: Clip) -> bool:
    return (
        a.start_time_sec <= b.start_time_sec + _BOUNDARY_EPS_SEC
        and a.end_time_sec >= b.end_time_sec - _BOUNDARY_EPS_SEC
    )


def _duplicate_reason(candidate: Clip, kept: Clip) -> str | None:
    if _time_iou(candidate, kept) >= 0.60:
        return "time_overlap"
    transcript_similarity = _jaccard(candidate.transcript or candidate.topic, kept.transcript or kept.topic)
    if (_contains_window(candidate, kept) or _contains_window(kept, candidate)) and transcript_similarity >= 0.80:
        return "contained_window"
    if _jaccard(candidate.viral_hook, kept.viral_hook) >= 0.88:
        return "hook_similarity"
    if transcript_similarity >= 0.82:
        return "text_similarity"
    if candidate.topic and kept.topic and _jaccard(candidate.topic, kept.topic) >= 0.90:
        return "topic_similarity"
    return None


def dedupe_clips(clips: list[Clip]) -> tuple[list[Clip], list[DedupeDecision]]:
    """Drop redundant candidates while preserving the strongest representative."""
    kept: list[Clip] = []
    decisions: list[DedupeDecision] = []
    ordered = sorted(clips, key=lambda c: (c.virality_score, -c.duration_sec), reverse=True)
    for candidate in ordered:
        duplicate_of: Clip | None = None
        reason: str | None = None
        for existing in kept:
            reason = _duplicate_reason(candidate, existing)
            if reason is not None:
                duplicate_of = existing
                break
        if duplicate_of is None or reason is None:
            kept.append(candidate)
            continue
        decisions.append(
            DedupeDecision(
                kept_clip_id=duplicate_of.clip_id,
                dropped_clip_id=candidate.clip_id,
                reason=reason,
                score_delta=round(duplicate_of.virality_score - candidate.virality_score, 4),
            )
        )
    return kept, decisions


def save_dedupe_report(
    output_path: Path,
    *,
    decisions: list[DedupeDecision],
    candidate_count: int,
    kept_count: int,
    mode: str,
) -> Path:
    payload = {
        "mode": mode,
        "dedupe_policy_version": CLIP_SELECTION_DEDUPE_POLICY_VERSION,
        "candidate_count": candidate_count,
        "kept_count": kept_count,
        "dropped_duplicate_count": len(decisions),
        "decisions": [decision.model_dump() for decision in decisions],
    }
    output_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    logger.info("Saved clip-selection dedupe report to %s", output_path)
    return output_path


def filter_clip_candidates(
    clips: list[Clip],
    *,
    threshold: float = DEFAULT_QUALITY_THRESHOLD,
    min_kept: int = DEFAULT_MIN_KEPT,
    max_kept: int | None = DEFAULT_MAX_KEPT,
    mode: str = DEFAULT_SELECTION_MODE,
) -> tuple[list[Clip], list[DedupeDecision]]:
    if not clips:
        return [], []

    normalized_mode = mode.strip().lower()
    if normalized_mode not in {"curated", "exhaustive"}:
        raise ValueError("clip selection mode must be 'curated' or 'exhaustive'")

    def _priority(c: Clip) -> tuple[float, float]:
        review_penalty = 0.5 if c.needs_review else 0.0
        return (c.virality_score - review_penalty, c.virality_score)

    ordered = sorted(clips, key=_priority, reverse=True)

    if normalized_mode == "exhaustive":
        kept = [c for c in ordered if c.virality_score >= threshold and not c.needs_review]
        kept, decisions = dedupe_clips(kept)
    else:
        strong = [c for c in ordered if c.virality_score >= threshold and not c.needs_review]
        kept = list(strong)
        if len(kept) < min_kept:
            backfill = [c for c in ordered if c not in kept]
            for c in backfill:
                if len(kept) >= min_kept:
                    break
                kept.append(c)
        decisions = []

    if max_kept is not None and len(kept) > max_kept:
        kept = kept[:max_kept]

    renumbered: list[Clip] = []
    id_map: dict[str, str] = {}
    for i, c in enumerate(kept, start=1):
        new_id = f"{i:03d}"
        id_map[c.clip_id] = new_id
        renumbered.append(c if c.clip_id == new_id else c.model_copy(update={"clip_id": new_id}))

    remapped_decisions = [
        decision.model_copy(
            update={
                "kept_clip_id": id_map.get(decision.kept_clip_id, decision.kept_clip_id),
            }
        )
        for decision in decisions
    ]

    dropped = len(ordered) - len(kept)
    max_label = "none" if max_kept is None else str(max_kept)
    logger.info(
        "Clip ranking: kept %d / %d candidates (mode=%s, threshold=%.2f, min=%d, max=%s, dropped=%d).",
        len(renumbered),
        len(ordered),
        normalized_mode,
        threshold,
        min_kept,
        max_label,
        dropped,
    )
    for c in renumbered:
        logger.info(
            "  [%s] score=%.2f %s %s",
            c.clip_id,
            c.virality_score,
            "(review)" if c.needs_review else "",
            c.topic,
        )
    return renumbered, remapped_decisions


def rank_and_filter_clips(
    clips: list[Clip],
    *,
    threshold: float = DEFAULT_QUALITY_THRESHOLD,
    min_kept: int = DEFAULT_MIN_KEPT,
    max_kept: int | None = DEFAULT_MAX_KEPT,
    mode: str = DEFAULT_SELECTION_MODE,
) -> list[Clip]:
    """Rank ``clips`` by ``virality_score`` and apply the threshold+floor+cap.

    Rules (in order, with clear precedence):

    1. Sort descending by ``virality_score``.
    2. Keep clips with ``virality_score >= threshold`` (or ``needs_review``
       cleared). Reviewed-out clips (``needs_review=True``) are always sent
       to the back of the priority queue.
    3. If fewer than ``min_kept`` clips passed the threshold, fill up from
       the remaining clips in rank order until we reach ``min_kept`` (or
       run out of candidates).
    4. Cap the final list at ``max_kept`` entries.
    5. Renumber ``clip_id`` to ``001``, ``002``, ... so downstream artifacts
       (keyframes, subtitles, output filenames) stay dense and ordered.

    This is the "threshold with a floor" policy the user asked for: quality
    first, but never ship zero shorts when the transcript is weak.
    """
    kept, _decisions = filter_clip_candidates(
        clips,
        threshold=threshold,
        min_kept=min_kept,
        max_kept=max_kept,
        mode=mode,
    )
    return kept


def select_clips(
    transcript: dict,
    *,
    config: PipelineConfig | None = None,
    gemini_model: str | None = None,
    candidate_count: int = DEFAULT_CANDIDATE_COUNT,
    quality_threshold: float = DEFAULT_QUALITY_THRESHOLD,
    min_kept: int = DEFAULT_MIN_KEPT,
    max_kept: int | None = DEFAULT_MAX_KEPT,
    temperature: float = DEFAULT_CANDIDATE_TEMPERATURE,
    selection_mode: str = DEFAULT_SELECTION_MODE,
    dedupe_report_path: Path | None = None,
) -> tuple[list[Clip], str]:
    """
    Call the configured LLM to select clips. Returns ``(clips, raw_json)`` for caching / debugging.

    The returned clip list has already been ranked + filtered by
    :func:`rank_and_filter_clips`. ``raw_json`` is the untouched LLM
    response so the cache artifact reflects the entire candidate pool for
    audit / re-ranking without another LLM call.
    """
    provider = resolved_llm_provider(config)
    model_name = resolved_text_model(config, model_override=gemini_model)
    system_prompt, user_text = build_prompt(
        transcript, candidate_count=candidate_count
    )
    output_token_budget = clip_selection_max_output_tokens(candidate_count)
    reasoning_effort, verbosity = _openai_reasoning_options(provider, model_name)

    def _call() -> tuple[str, ClipSelectionResponse | None]:
        logger.info(
            "%s clip selection (model=%s, candidate_pool=%d, temp=%.2f, max_output_tokens=%d)...",
            provider,
            model_name,
            candidate_count,
            temperature,
            output_token_budget,
        )
        response = call_structured_llm(
            StructuredLlmRequest(
                stage_name="clip selection",
                model=model_name,
                system_instruction=system_prompt,
                user_text=user_text,
                temperature=temperature,
                response_schema=ClipSelectionResponse,
                max_output_tokens=output_token_budget,
                reasoning_effort=reasoning_effort,
                verbosity=verbosity,
            ),
            provider=provider,
        )
        raw = response.raw_text
        parsed = response.parsed if isinstance(response.parsed, ClipSelectionResponse) else None
        if not raw and parsed is None:
            raise RuntimeError("LLM returned neither text nor parsed response for clip selection")
        if not raw and parsed is not None:
            raw = parsed.model_dump_json()
        assert raw is not None
        return raw, parsed

    raw, parsed = _retry_llm("Clip selection", _call)
    candidates = complete_clip_boundaries(_parse_clips(raw, parsed=parsed), transcript)
    # The ranker can only backfill from the pool the model returned. If it
    # under-delivered (e.g. returned 2 of a requested 12), the min_kept floor
    # is unenforceable -- warn loudly so we do not silently ship fewer shorts
    # than the caller expected.
    if len(candidates) < min_kept:
        logger.warning(
            "Clip selection: LLM returned only %d candidates (requested %d, floor %d). "
            "Output will be capped at %d shorts -- check prompt or transcript length.",
            len(candidates),
            candidate_count,
            min_kept,
            len(candidates),
        )
    elif len(candidates) < candidate_count:
        logger.info(
            "Clip selection: LLM returned %d of %d requested candidates "
            "(pool still >= floor of %d).",
            len(candidates),
            candidate_count,
            min_kept,
        )
    clips, decisions = filter_clip_candidates(
        candidates,
        threshold=quality_threshold,
        min_kept=min_kept,
        max_kept=max_kept,
        mode=selection_mode,
    )
    if dedupe_report_path is not None:
        save_dedupe_report(
            dedupe_report_path,
            decisions=decisions,
            candidate_count=len(candidates),
            kept_count=len(clips),
            mode=selection_mode,
        )
    return clips, raw


def _parse_clips(
    raw_json: str,
    *,
    parsed: ClipSelectionResponse | None = None,
) -> list[Clip]:
    """Parse and validate the LLM's JSON response into Clip objects."""
    if parsed is not None:
        candidates = parsed.clips
    else:
        data = json.loads(raw_json)
        clips_data = data.get("clips", data) if isinstance(data, dict) else data
        candidates = [_ClipSelectionCandidate.model_validate(item) for item in clips_data]

    clips: list[Clip] = []
    for candidate in candidates:
        clip = _candidate_to_clip(candidate)

        actual_dur = clip.end_time_sec - clip.start_time_sec
        stated_dur = getattr(candidate, "duration_sec", None)
        if stated_dur is not None and abs(actual_dur - float(stated_dur)) > 1.0:
            logger.warning(
                "Clip %s: stated duration %.1fs doesn't match (%.1f-%.1f = %.1f).",
                clip.clip_id,
                float(stated_dur),
                clip.start_time_sec,
                clip.end_time_sec,
                actual_dur,
            )
        clips.append(clip)

    logger.info("Parsed %d clips from LLM response", len(clips))
    return clips


def load_candidate_pool_from_raw_response(
    raw_json: str, *, transcript: dict | None = None
) -> list[Clip]:
    """Parse the cached raw LLM response into a candidate pool.

    This is used when transcript/model inputs still match but the ranking
    policy changed; re-ranking the cached pool is much cheaper than another
    LLM call.
    """
    clips = _parse_clips(raw_json)
    if transcript is not None:
        return complete_clip_boundaries(clips, transcript)
    return clips


def save_clips(clips: list[Clip], output_path: Path) -> Path:
    """Persist clips to a JSON file using the shared Pydantic schema."""
    plan = ClipPlan(source_path="", clips=list(clips))
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(plan.model_dump_json(indent=2))
    logger.info("Saved %d clips to %s", len(clips), output_path)
    return output_path


def load_clips(clips_path: Path) -> list[Clip]:
    """Load clips from a previously saved JSON file."""
    with open(clips_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    if isinstance(data, dict) and "clips" in data:
        return [Clip.model_validate(c) for c in data["clips"]]
    return [Clip.model_validate(c) for c in data]


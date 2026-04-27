"""Clip selection: pick the strongest 30-60s segments from a long source.

Two backends, same contract:

* ``select_clips_heuristic`` — greedy word-density scoring. Uses the
  transcript alone; zero model calls. Good baseline when transcript exists.
* ``select_clips_with_llm`` — pluggable LLM hook. Caller provides a
  ``(prompt_text) -> str`` function that must return strict JSON matching
  the ``ClipPlan`` schema. We re-validate before returning.

Both return a ``ClipPlan``.
"""

from __future__ import annotations

import json
from typing import Callable

from ..schemas import Clip, ClipPlan, TranscriptWord


LLMTextFn = Callable[[str], str]


CLIP_SELECTOR_PROMPT_TEMPLATE = """You are a viral-clip selector for a podcast editor.
Return ONLY JSON matching this shape:

{{
  "source_path": "{source_path}",
  "clips": [
    {{
      "clip_id": "001",
      "topic": "<short topic>",
      "start_time_sec": <float>,
      "end_time_sec": <float>,
      "viral_hook": "<one line>",
      "virality_score": <0..1>,
      "transcript": "<full clip transcript>",
      "suggested_overlay_title": "<<=6 words>"
    }}
  ]
}}

Pick {target_count} clips, each {min_sec}-{max_sec} seconds long, NO overlaps, sorted by virality_score desc.

Transcript (word, start, end):
{transcript}
"""


def _words_in_window(
    words: list[TranscriptWord], start: float, end: float
) -> list[TranscriptWord]:
    return [w for w in words if w.start_time >= start and w.end_time <= end]


def select_clips_heuristic(
    source_path: str,
    words: list[TranscriptWord],
    duration_sec: float,
    *,
    target_count: int = 5,
    min_sec: float = 30.0,
    max_sec: float = 60.0,
    step_sec: float = 5.0,
) -> ClipPlan:
    """Greedy: slide a window, score by words/sec, take top non-overlapping picks."""

    if duration_sec <= min_sec or not words:
        # No sensible windowing possible; return one clip of the whole thing.
        end = min(duration_sec, max_sec) if duration_sec > 0 else max_sec
        return ClipPlan(
            source_path=source_path,
            clips=[
                Clip(
                    clip_id="001",
                    topic="Full source",
                    start_time_sec=0.0,
                    end_time_sec=max(end, 1.0),
                    viral_hook="",
                    virality_score=0.5,
                    transcript=" ".join(w.word for w in words),
                    suggested_overlay_title="Highlight",
                )
            ],
        )

    candidates: list[tuple[float, float, float, str]] = []
    window = (min_sec + max_sec) / 2.0
    t = 0.0
    while t + window <= duration_sec:
        ws = _words_in_window(words, t, t + window)
        if ws:
            density = len(ws) / window
            text = " ".join(w.word for w in ws)
            candidates.append((density, t, t + window, text))
        t += step_sec

    candidates.sort(key=lambda c: c[0], reverse=True)
    picked: list[tuple[float, float, float, str]] = []
    for c in candidates:
        if len(picked) >= target_count:
            break
        if all(c[2] <= p[1] or c[1] >= p[2] for p in picked):
            picked.append(c)
    picked.sort(key=lambda c: c[1])

    clips: list[Clip] = []
    for i, (density, s, e, text) in enumerate(picked, start=1):
        norm = min(1.0, density / 3.0)  # ~3 words/sec is dense talking
        clips.append(
            Clip(
                clip_id=f"{i:03d}",
                topic=text.split(".")[0][:60] or f"Clip {i}",
                start_time_sec=round(s, 2),
                end_time_sec=round(e, 2),
                viral_hook=text[:120],
                virality_score=round(norm, 3),
                transcript=text,
                suggested_overlay_title=(text.split(".")[0][:40] or f"Clip {i}"),
            )
        )
    return ClipPlan(source_path=source_path, clips=clips)


def select_clips_with_llm(
    source_path: str,
    words: list[TranscriptWord],
    *,
    target_count: int,
    min_sec: float,
    max_sec: float,
    text_fn: LLMTextFn,
) -> ClipPlan:
    transcript_lines = "\n".join(
        f"{w.word}\t{w.start_time:.2f}\t{w.end_time:.2f}" for w in words
    )
    prompt = CLIP_SELECTOR_PROMPT_TEMPLATE.format(
        source_path=source_path,
        target_count=target_count,
        min_sec=min_sec,
        max_sec=max_sec,
        transcript=transcript_lines,
    )
    raw = text_fn(prompt)
    try:
        data = json.loads(raw)
    except json.JSONDecodeError as e:
        raise ValueError(f"LLM did not return JSON: {e}; raw={raw[:200]!r}") from e
    return ClipPlan.model_validate(data)


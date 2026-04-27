"""Vision-LLM + OCR primitive — the alt path to per-scene framing decisions.

Design (Bryan's "big screen change -> v3 images -> LLM+OCR -> bbox" idea):

1. Scene detection already produces one keyframe per scene (deterministic,
   local, cheap). That is ``primitives/ingest.py::extract_keyframes``.
2. For each keyframe, call a pluggable vision LLM with an OCR hint. The
   model returns normalized bboxes for the on-screen roles it cares about
   (``person``, ``chart``) plus any OCR text it reads.
3. Fold those bboxes into ``LayoutInstruction`` values so the existing
   layout planner (``primitives/layouts.py``) does the actual ffmpeg math.

Why this shape:

* **Pluggable**. Caller supplies ``LLMRegionFn``. We never hard-code a
  provider. The same primitive works for Gemini, GPT-4o, internal models,
  tests, or mocks.
* **Schema-validated**. Raw model output is parsed into ``SceneRegions``
  (Pydantic). Malformed output degrades to ``None`` regions rather than
  crashing or corrupting downstream state.
* **Separable**. ``detect_regions_with_llm`` is one function. Mapping
  regions to ``LayoutInstruction`` is another. Mapping a ``LayoutKind``
  guess from regions is a third. Each is independently testable.
"""

from __future__ import annotations

import json
from typing import Callable

from ..schemas import (
    BoundingBox,
    LayoutInstruction,
    LayoutKind,
    Scene,
    SceneClassification,
    SceneRegions,
)


LLMRegionFn = Callable[[str, str], str]
"""Signature: (keyframe_path, prompt) -> raw model string (expected JSON).

The caller is responsible for any image encoding (base64, multipart, etc.).
The primitive only passes the path + prompt and re-validates the reply.
"""


REGION_PROMPT = """You are a vision+OCR system for a short-video editor.
Look at the provided keyframe and return a STRICT JSON object of this shape:

{
  "person_bbox": {"x1": <0..1>, "y1": <0..1>, "x2": <0..1>, "y2": <0..1>, "confidence": <0..1>} | null,
  "chart_bbox":  {"x1": <0..1>, "y1": <0..1>, "x2": <0..1>, "y2": <0..1>, "confidence": <0..1>} | null,
  "ocr_text":    "<text visible on screen, empty string if none>",
  "reason":      "<= 20 words of rationale"
}

Rules:
- All bbox coordinates are normalized to the frame (0=left/top, 1=right/bottom).
- x2 > x1, y2 > y1.
- Return null for any region that is not present (e.g. a pure talking-head
  scene has no chart).
- "person_bbox" is the *speaker's* body/head region if visible.
- "chart_bbox" is any chart, graph, slide, screenshare, or diagram.
- OCR text should be the readable text on screen (titles, labels, chart
  axis values). Omit subtitle captions.
- NO markdown, NO prose outside JSON. JSON only.
"""


# ---------------------------------------------------------------------------
# Core: detect regions per scene via pluggable LLM
# ---------------------------------------------------------------------------


def detect_regions_with_llm(
    scenes: list[Scene], vision_fn: LLMRegionFn
) -> list[SceneRegions]:
    """Call ``vision_fn`` for each scene's keyframe and return parsed regions.

    Parse failures degrade to an empty ``SceneRegions`` with ``raw_reason``
    describing the error — never raise — so a single bad scene can't take
    down the whole pipeline.
    """

    out: list[SceneRegions] = []
    for s in scenes:
        if not s.keyframe_path:
            out.append(
                SceneRegions(scene_id=s.scene_id, raw_reason="no keyframe available")
            )
            continue
        raw = vision_fn(s.keyframe_path, REGION_PROMPT)
        out.append(_parse_region_reply(s.scene_id, raw))
    return out


def _parse_region_reply(scene_id: str, raw: str) -> SceneRegions:
    try:
        data = json.loads(raw)
    except json.JSONDecodeError as e:
        return SceneRegions(scene_id=scene_id, raw_reason=f"JSON parse error: {e!r}")

    def _opt_bbox(value: object) -> BoundingBox | None:
        if not value:
            return None
        try:
            return BoundingBox.model_validate(value)
        except Exception:
            return None

    return SceneRegions(
        scene_id=scene_id,
        person_bbox=_opt_bbox(data.get("person_bbox")),
        chart_bbox=_opt_bbox(data.get("chart_bbox")),
        ocr_text=str(data.get("ocr_text", ""))[:4000],
        raw_reason=str(data.get("reason", ""))[:400],
    )


# ---------------------------------------------------------------------------
# Derivation: regions -> LayoutKind / LayoutInstruction
# ---------------------------------------------------------------------------


# Width threshold: if the chart bbox covers this much of the frame width, it
# is wide enough to treat the scene as a split_chart_person. Tuned for the
# source videos described in the spec (chart ~2/3 of width).
_CHART_WIDTH_SPLIT_THRESHOLD = 0.45


def classify_from_regions(regions: SceneRegions) -> SceneClassification:
    """Pick a ``LayoutKind`` for a scene using only its ``SceneRegions``.

    Priority:
      1. If ``chart_bbox`` is present and wide, it's ``SPLIT_CHART_PERSON``.
      2. Else if ``person_bbox`` is present and tight, ``ZOOM_CALL_CENTER``.
      3. Else default to ``SIT_CENTER`` with low confidence.

    "Tight" ≈ the person covers more than half the frame width (zoom-call
    webcam framing). "Wide" for a chart ≈ 45% of frame width or more.
    """

    if regions.chart_bbox and regions.chart_bbox.width >= _CHART_WIDTH_SPLIT_THRESHOLD:
        return SceneClassification(
            scene_id=regions.scene_id,
            layout=LayoutKind.SPLIT_CHART_PERSON,
            confidence=float(min(1.0, 0.5 + regions.chart_bbox.width / 2.0)),
            reason=f"chart bbox covers {regions.chart_bbox.width:.2f} of width",
        )
    if regions.person_bbox and regions.person_bbox.width >= 0.5:
        return SceneClassification(
            scene_id=regions.scene_id,
            layout=LayoutKind.ZOOM_CALL_CENTER,
            confidence=float(min(1.0, 0.5 + regions.person_bbox.width / 2.0)),
            reason=f"person bbox wide ({regions.person_bbox.width:.2f}) — tight framing",
        )
    if regions.person_bbox:
        return SceneClassification(
            scene_id=regions.scene_id,
            layout=LayoutKind.SIT_CENTER,
            confidence=0.7,
            reason="person present, no wide chart, wider framing",
        )
    return SceneClassification(
        scene_id=regions.scene_id,
        layout=LayoutKind.SIT_CENTER,
        confidence=0.3,
        reason=regions.raw_reason or "no regions detected — defaulting to sit_center",
    )


def layout_instruction_from_regions(
    regions: SceneRegions,
    classification: SceneClassification,
    *,
    clip_id: str | None = None,
    zoom: float = 1.0,
) -> LayoutInstruction:
    """Build a ``LayoutInstruction`` whose knobs are populated from bboxes.

    ``person_x_norm`` uses the person bbox center when available; falls back
    to 0.5 (center). ``chart_x_norm`` uses the chart bbox left edge; falls
    back to 0.0.
    """

    person_x = regions.person_bbox.center_x if regions.person_bbox else 0.5
    chart_x = regions.chart_bbox.x1 if regions.chart_bbox else 0.0
    return LayoutInstruction(
        clip_id=clip_id or classification.scene_id,
        layout=classification.layout,
        zoom=zoom,
        person_x_norm=person_x,
        chart_x_norm=chart_x,
    )


def classify_scenes_with_vision_llm(
    scenes: list[Scene], vision_fn: LLMRegionFn
) -> list[tuple[SceneRegions, SceneClassification]]:
    """One-shot helper: keyframes -> regions -> classifications.

    Returns ``(regions, classification)`` pairs per scene so the caller can
    keep both artefacts on disk (regions = deep detail, classification =
    what a renderer consumes).
    """

    regions = detect_regions_with_llm(scenes, vision_fn)
    return [(r, classify_from_regions(r)) for r in regions]


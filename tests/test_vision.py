"""Tests for the scene-change + vision-LLM + OCR bbox primitive.

Covers:
* happy path: well-formed JSON -> populated ``SceneRegions``.
* bad JSON: degrade to empty regions + raw_reason, never raise.
* bad bbox: one malformed bbox does not take down the whole scene record.
* classification dispatch: chart width -> SPLIT; wide person -> ZOOM; else SIT.
* layout instruction derivation: ``person_x_norm`` / ``chart_x_norm`` come
  from the bboxes when present, defaults when not.
"""

import json

import pytest

from afterform.primitives.vision import (
    _CHART_WIDTH_SPLIT_THRESHOLD,
    classify_from_regions,
    classify_scenes_with_vision_llm,
    detect_regions_with_llm,
    layout_instruction_from_regions,
)
from afterform.schemas import (
    BoundingBox,
    LayoutKind,
    Scene,
    SceneClassification,
    SceneRegions,
)


# ---------------------------------------------------------------------------
# Schema
# ---------------------------------------------------------------------------


def test_bounding_box_requires_x2_gt_x1():
    BoundingBox(x1=0.1, y1=0.1, x2=0.2, y2=0.2)
    with pytest.raises(ValueError):
        BoundingBox(x1=0.2, y1=0.1, x2=0.1, y2=0.2)
    with pytest.raises(ValueError):
        BoundingBox(x1=0.1, y1=0.2, x2=0.2, y2=0.1)


def test_bounding_box_center_and_width():
    b = BoundingBox(x1=0.2, y1=0.4, x2=0.6, y2=0.9)
    assert b.center_x == pytest.approx(0.4)
    assert b.center_y == pytest.approx(0.65)
    assert b.width == pytest.approx(0.4)


# ---------------------------------------------------------------------------
# detect_regions_with_llm
# ---------------------------------------------------------------------------


def _scene(i: int, kf: str | None = "/tmp/x.jpg") -> Scene:
    return Scene(scene_id=f"s{i}", start_time=float(i), end_time=float(i) + 1.0, keyframe_path=kf)


def test_detect_regions_happy_path():
    scenes = [_scene(0)]

    def vision_fn(_img: str, _prompt: str) -> str:
        return json.dumps(
            {
                "person_bbox": {"x1": 0.7, "y1": 0.1, "x2": 0.98, "y2": 0.9, "confidence": 0.9},
                "chart_bbox": {"x1": 0.02, "y1": 0.05, "x2": 0.65, "y2": 0.95, "confidence": 0.8},
                "ocr_text": "Inflation YoY",
                "reason": "explainer layout",
            }
        )

    out = detect_regions_with_llm(scenes, vision_fn)
    assert len(out) == 1
    r = out[0]
    assert r.scene_id == "s0"
    assert r.person_bbox and r.person_bbox.center_x > 0.8
    assert r.chart_bbox and r.chart_bbox.width > 0.6
    assert "Inflation" in r.ocr_text


def test_detect_regions_bad_json_is_safe():
    scenes = [_scene(0)]

    def vision_fn(*_a) -> str:
        return "not json"

    out = detect_regions_with_llm(scenes, vision_fn)
    assert out[0].person_bbox is None
    assert out[0].chart_bbox is None
    assert "parse error" in out[0].raw_reason.lower()


def test_detect_regions_missing_keyframe_is_safe():
    scenes = [_scene(0, kf=None)]

    def vision_fn(*_a) -> str:  # pragma: no cover - should not be called
        raise AssertionError("vision_fn must not be called without a keyframe")

    out = detect_regions_with_llm(scenes, vision_fn)
    assert out[0].person_bbox is None
    assert "no keyframe" in out[0].raw_reason.lower()


def test_detect_regions_bad_bbox_degrades_gracefully():
    scenes = [_scene(0)]

    def vision_fn(*_a) -> str:
        return json.dumps(
            {
                "person_bbox": {"x1": 0.5, "y1": 0.1, "x2": 0.3, "y2": 0.9},
                "chart_bbox": {"x1": 0.02, "y1": 0.05, "x2": 0.65, "y2": 0.95},
                "ocr_text": "",
                "reason": "person bbox inverted",
            }
        )

    out = detect_regions_with_llm(scenes, vision_fn)
    assert out[0].person_bbox is None
    assert out[0].chart_bbox is not None


# ---------------------------------------------------------------------------
# classify_from_regions
# ---------------------------------------------------------------------------


def test_classify_wide_chart_is_split():
    r = SceneRegions(
        scene_id="s0",
        chart_bbox=BoundingBox(x1=0.0, y1=0.0, x2=0.66, y2=1.0),
        person_bbox=BoundingBox(x1=0.72, y1=0.1, x2=0.99, y2=0.95),
    )
    c = classify_from_regions(r)
    assert c.layout == LayoutKind.SPLIT_CHART_PERSON
    assert c.confidence > 0.5


def test_classify_narrow_chart_not_split():
    r = SceneRegions(
        scene_id="s0",
        chart_bbox=BoundingBox(x1=0.4, y1=0.2, x2=0.5, y2=0.4),
        person_bbox=BoundingBox(x1=0.3, y1=0.1, x2=0.85, y2=0.95),
    )
    c = classify_from_regions(r)
    # chart width (0.1) is below the split threshold -> not split
    assert c.layout != LayoutKind.SPLIT_CHART_PERSON


def test_classify_wide_person_is_zoom_call():
    r = SceneRegions(
        scene_id="s0",
        person_bbox=BoundingBox(x1=0.1, y1=0.05, x2=0.9, y2=0.98),
    )
    c = classify_from_regions(r)
    assert c.layout == LayoutKind.ZOOM_CALL_CENTER


def test_classify_small_person_is_sit_center():
    r = SceneRegions(
        scene_id="s0",
        person_bbox=BoundingBox(x1=0.4, y1=0.2, x2=0.6, y2=0.8),
    )
    c = classify_from_regions(r)
    assert c.layout == LayoutKind.SIT_CENTER


def test_classify_nothing_detected_defaults_sit_center_low_conf():
    r = SceneRegions(scene_id="s0", raw_reason="model returned null")
    c = classify_from_regions(r)
    assert c.layout == LayoutKind.SIT_CENTER
    assert c.confidence <= 0.5


def test_chart_threshold_is_exported():
    # guard against the tuning constant silently being removed
    assert 0.0 < _CHART_WIDTH_SPLIT_THRESHOLD < 1.0


# ---------------------------------------------------------------------------
# layout_instruction_from_regions
# ---------------------------------------------------------------------------


def test_layout_instruction_from_regions_split():
    r = SceneRegions(
        scene_id="s0",
        chart_bbox=BoundingBox(x1=0.0, y1=0.0, x2=0.66, y2=1.0),
        person_bbox=BoundingBox(x1=0.72, y1=0.1, x2=0.99, y2=0.95),
    )
    c = classify_from_regions(r)
    instr = layout_instruction_from_regions(r, c)
    assert instr.layout == LayoutKind.SPLIT_CHART_PERSON
    # person_x_norm = center of (0.72, 0.99) = 0.855
    assert instr.person_x_norm == pytest.approx(0.855, rel=1e-3)
    # chart_x_norm = left edge = 0.0
    assert instr.chart_x_norm == pytest.approx(0.0)


def test_layout_instruction_defaults_when_no_regions():
    r = SceneRegions(scene_id="s0")
    c = SceneClassification(
        scene_id="s0", layout=LayoutKind.SIT_CENTER, confidence=0.3, reason="default"
    )
    instr = layout_instruction_from_regions(r, c)
    assert instr.person_x_norm == 0.5
    assert instr.chart_x_norm == 0.0


def test_classify_scenes_with_vision_llm_returns_pairs():
    scenes = [_scene(0)]

    def vision_fn(*_a) -> str:
        return json.dumps(
            {
                "person_bbox": {"x1": 0.1, "y1": 0.1, "x2": 0.95, "y2": 0.95},
                "chart_bbox": None,
                "ocr_text": "",
                "reason": "solo subject",
            }
        )

    pairs = classify_scenes_with_vision_llm(scenes, vision_fn)
    assert len(pairs) == 1
    regions, classification = pairs[0]
    assert regions.person_bbox is not None
    assert classification.layout == LayoutKind.ZOOM_CALL_CENTER


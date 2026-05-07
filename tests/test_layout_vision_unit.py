"""layout_vision parsing (no API calls)."""

import math
from types import SimpleNamespace

import pytest

from afterform.flows.long_to_shorts.plan_layouts import (
    SampledFrame,
    _GeminiMultiFrameResponse,
    _VISION_HTTP_TIMEOUT_MS,
    _VISION_RETRY_ATTEMPTS,
    _call_gemini_vision,
    _face_center_x,
    infer_layout_instructions,
    _instruction_from_gemini_json,
)
from afterform.primitives.layouts import plan_layout
from afterform.schemas import BoundingBox, LayoutKind


def test_instruction_from_gemini_json_split_with_bboxes():
    data = {
        "layout": "split_chart_person",
        "person_bbox": {"x1": 0.62, "y1": 0.05, "x2": 0.99, "y2": 0.95},
        "chart_bbox": {"x1": 0.02, "y1": 0.05, "x2": 0.58, "y2": 0.92},
        "reason": "webinar",
    }
    instr = _instruction_from_gemini_json("005", data)
    assert instr.layout == LayoutKind.SPLIT_CHART_PERSON
    assert instr.split_chart_region is not None
    assert instr.split_person_region is not None


def test_instruction_from_gemini_json_sit_center():
    data = {
        "layout": "sit_center",
        "person_bbox": {"x1": 0.3, "y1": 0.1, "x2": 0.7, "y2": 0.9},
        "chart_bbox": None,
        "reason": "talking head",
    }
    instr = _instruction_from_gemini_json("001", data)
    assert instr.layout == LayoutKind.SIT_CENTER
    assert instr.split_chart_region is None


def test_sit_center_with_large_visual_and_person_promotes_to_split():
    warnings: list[str] = []
    data = {
        "layout": "sit_center",
        "person_bbox": {"x1": 0.64, "y1": 0.06, "x2": 0.98, "y2": 0.96},
        "face_bbox": {"x1": 0.72, "y1": 0.08, "x2": 0.86, "y2": 0.35},
        "chart_bbox": {"x1": 0.03, "y1": 0.18, "x2": 0.58, "y2": 0.86},
        "reason": "Model called it sit_center, but a large image/article is visible beside the speaker.",
    }

    instr = _instruction_from_gemini_json("004", data, warnings=warnings)

    assert instr.layout == LayoutKind.SPLIT_CHART_PERSON
    assert instr.split_chart_region is not None
    assert instr.split_person_region is not None
    assert "large visual and person bboxes; promoted to split_chart_person" in warnings


def test_face_bbox_pulls_person_x_norm_toward_the_face():
    """Regression for the off-center subject bug.

    Reproduces clip 001 from the Dr. Mike failing run: subject sitting in
    profile, head around xâ‰ˆ0.23, tank top + arm extend the body bbox out
    to x2=0.75. The wide person_bbox center alone gave person_x_norm=0.415,
    which cropped the final 9:16 short on the torso and pushed the face off
    the left edge. With the face_bbox hint, person_x_norm must track the
    face instead.
    """
    data = {
        "layout": "sit_center",
        "person_bbox": {"x1": 0.08, "y1": 0.10, "x2": 0.75, "y2": 0.95},
        "face_bbox":   {"x1": 0.18, "y1": 0.12, "x2": 0.30, "y2": 0.32},
        "chart_bbox": None,
        "reason": "profile speaker off-center left",
    }
    instr = _instruction_from_gemini_json("001", data)
    assert instr.layout == LayoutKind.SIT_CENTER
    # Face center is 0.24. person-bbox center is 0.415. Must follow the face.
    assert math.isclose(instr.person_x_norm, 0.24, abs_tol=1e-6), (
        f"person_x_norm should track face center (0.24), got {instr.person_x_norm}"
    )


def test_face_bbox_missing_falls_back_to_person_bbox_center():
    data = {
        "layout": "sit_center",
        "person_bbox": {"x1": 0.30, "y1": 0.10, "x2": 0.70, "y2": 0.90},
        "face_bbox": None,
        "chart_bbox": None,
        "reason": "centered talking head",
    }
    instr = _instruction_from_gemini_json("002", data)
    assert math.isclose(instr.person_x_norm, 0.50, abs_tol=1e-6)


def test_face_bbox_rejected_when_as_wide_as_person_bbox():
    """If Gemini echoes the person bbox into face_bbox we get no new info.

    In that case fall back to the person-bbox center, not a spurious face
    center â€” we don't want the "fix" to regress the centered case.
    """
    data = {
        "layout": "sit_center",
        "person_bbox": {"x1": 0.10, "y1": 0.10, "x2": 0.90, "y2": 0.95},
        "face_bbox":   {"x1": 0.10, "y1": 0.10, "x2": 0.90, "y2": 0.95},
        "chart_bbox": None,
        "reason": "echoed bbox",
    }
    instr = _instruction_from_gemini_json("003", data)
    # Fall back to person-bbox center (0.5) â€” face_bbox too wide to trust.
    assert math.isclose(instr.person_x_norm, 0.50, abs_tol=1e-6)


def test_face_bbox_outside_person_bbox_is_ignored():
    """If face_bbox center sits outside person_bbox the model got confused."""
    data = {
        "layout": "sit_center",
        "person_bbox": {"x1": 0.60, "y1": 0.10, "x2": 0.95, "y2": 0.95},
        "face_bbox":   {"x1": 0.05, "y1": 0.10, "x2": 0.15, "y2": 0.25},
        "chart_bbox": None,
        "reason": "mismatched face and person",
    }
    instr = _instruction_from_gemini_json("004", data)
    # Person bbox center = 0.775; we must not jump to face center (0.10).
    assert math.isclose(instr.person_x_norm, 0.775, abs_tol=1e-6)


def test_face_center_helper_unit():
    # Clean case: tight face inside the body.
    face = BoundingBox(x1=0.20, y1=0.10, x2=0.30, y2=0.25)
    body = BoundingBox(x1=0.10, y1=0.10, x2=0.70, y2=0.95)
    assert _face_center_x(face, body) == 0.25

    # No face.
    assert _face_center_x(None, body) is None

    # Face suspiciously wide (> 40% of frame): ignore.
    wide = BoundingBox(x1=0.10, y1=0.10, x2=0.60, y2=0.95)
    assert _face_center_x(wide, body) is None


def test_instruction_from_gemini_json_accepts_0_to_1000_bboxes():
    data = {
        "layout": "split_chart_person",
        "person_bbox": {"x1": 620, "y1": 50, "x2": 980, "y2": 950},
        "chart_bbox": {"x1": 20, "y1": 50, "x2": 580, "y2": 920},
        "reason": "webinar",
    }
    warnings: list[str] = []
    instr = _instruction_from_gemini_json("005", data, warnings=warnings)
    assert instr.layout == LayoutKind.SPLIT_CHART_PERSON
    assert instr.split_chart_region is not None
    assert instr.split_chart_region.x2 == 0.58
    assert any("normalized gemini_1000" in warning for warning in warnings)


def test_split_chart_person_face_bbox_emits_render_friendly_regions():
    data = {
        "layout": "split_chart_person",
        "person_bbox": {"x1": 0.59, "y1": 0.072, "x2": 1.00, "y2": 1.00},
        "face_bbox": {"x1": 0.72, "y1": 0.082, "x2": 0.86, "y2": 0.48},
        "chart_bbox": {"x1": 0.021, "y1": 0.028, "x2": 0.584, "y2": 0.722},
        "reason": "chart left, speaker right",
    }
    instr = _instruction_from_gemini_json("001", data)

    assert instr.layout == LayoutKind.SPLIT_CHART_PERSON
    assert instr.split_person_region is not None
    assert instr.split_person_region.y1 <= 0.001
    assert instr.split_person_region.y2 < 0.85
    assert instr.top_band_ratio < 0.45


def test_split_chart_person_render_friendly_regions_reduce_crop_pressure():
    data = {
        "layout": "split_chart_person",
        "person_bbox": {"x1": 0.59, "y1": 0.072, "x2": 1.00, "y2": 1.00},
        "face_bbox": {"x1": 0.72, "y1": 0.082, "x2": 0.86, "y2": 0.48},
        "chart_bbox": {"x1": 0.021, "y1": 0.028, "x2": 0.584, "y2": 0.722},
        "reason": "chart left, speaker right",
    }
    instr = _instruction_from_gemini_json("001", data)
    fg = plan_layout(instr, out_w=1080, out_h=1920, src_w=1920, src_h=1080).filtergraph

    assert "scale=1080:730" in fg
    assert "scale=1080:1190" in fg
    assert "crop=794:1080:1126:0" in fg


def test_edge_clipped_split_speaker_with_clear_side_pane_stays_split():
    data = {
        "layout": "split_chart_person",
        "person_bbox": {"x1": 0.839, "y1": 0.0, "x2": 1.0, "y2": 0.768},
        "face_bbox": {"x1": 0.839, "y1": 0.128, "x2": 1.0, "y2": 0.512},
        "chart_bbox": {"x1": 0.036, "y1": 0.28, "x2": 0.465, "y2": 0.518},
        "reason": "left visual plus right speaker pane, but LLM boxed only the visible edge face",
    }

    warnings: list[str] = []
    instr = _instruction_from_gemini_json("004", data, warnings=warnings)

    assert instr.layout == LayoutKind.SPLIT_CHART_PERSON
    assert instr.split_chart_region is not None
    assert instr.split_person_region is not None
    assert instr.split_person_region.x1 == pytest.approx(0.465)
    assert instr.split_person_region.x2 == 1.0
    assert "edge-clipped speaker bbox expanded to side pane for split render" in warnings


def test_edge_clipped_left_speaker_with_clear_side_pane_stays_split():
    data = {
        "layout": "split_chart_person",
        "person_bbox": {"x1": 0.0, "y1": 0.04, "x2": 0.13, "y2": 0.82},
        "face_bbox": {"x1": 0.0, "y1": 0.14, "x2": 0.13, "y2": 0.46},
        "chart_bbox": {"x1": 0.57, "y1": 0.12, "x2": 0.96, "y2": 0.72},
        "reason": "right visual plus left speaker pane, but LLM boxed only the visible edge face",
    }

    warnings: list[str] = []
    instr = _instruction_from_gemini_json("004", data, warnings=warnings)

    assert instr.layout == LayoutKind.SPLIT_CHART_PERSON
    assert instr.split_person_region is not None
    assert instr.split_person_region.x1 == 0.0
    assert instr.split_person_region.x2 == pytest.approx(0.57)
    assert "edge-clipped speaker bbox expanded to side pane for split render" in warnings


def test_edge_clipped_right_speaker_with_wide_visual_still_uses_available_pane():
    data = {
        "layout": "split_chart_person",
        "person_bbox": {"x1": 0.94, "y1": 0.02, "x2": 1.0, "y2": 0.72},
        "face_bbox": {"x1": 0.94, "y1": 0.12, "x2": 1.0, "y2": 0.42},
        "chart_bbox": {"x1": 0.04, "y1": 0.08, "x2": 0.72, "y2": 0.88},
        "reason": "wide left visual leaves a narrow but renderer-usable right speaker pane",
    }

    warnings: list[str] = []
    instr = _instruction_from_gemini_json("004", data, warnings=warnings)

    assert instr.layout == LayoutKind.SPLIT_CHART_PERSON
    assert instr.split_person_region is not None
    assert instr.split_person_region.x1 == pytest.approx(0.72)
    assert instr.split_person_region.x2 == 1.0
    assert "edge-clipped speaker bbox expanded to side pane for split render" in warnings


def test_edge_clipped_split_speaker_without_clear_side_pane_promotes_to_wide_visual():
    data = {
        "layout": "split_chart_person",
        "person_bbox": {"x1": 0.91, "y1": 0.0, "x2": 1.0, "y2": 0.768},
        "face_bbox": {"x1": 0.91, "y1": 0.128, "x2": 1.0, "y2": 0.512},
        "chart_bbox": {"x1": 0.036, "y1": 0.12, "x2": 0.82, "y2": 0.88},
        "reason": "visual consumes the right side, leaving only a true speaker sliver",
    }

    warnings: list[str] = []
    instr = _instruction_from_gemini_json("004", data, warnings=warnings)

    assert instr.layout == LayoutKind.WIDE_VISUAL
    assert instr.split_chart_region is not None
    assert "edge-clipped speaker with large visual; promoted to wide_visual" in warnings


def test_sit_center_tight_closeup_uses_contained_framing():
    data = {
        "layout": "sit_center",
        "person_bbox": {"x1": 0.216, "y1": 0.103, "x2": 0.660, "y2": 1.0},
        "face_bbox": {"x1": 0.390, "y1": 0.120, "x2": 0.521, "y2": 0.556},
        "reason": "Tight event close-up where a 9:16 cover crop would over-zoom.",
    }

    warnings: list[str] = []
    instr = _instruction_from_gemini_json("004", data, warnings=warnings)

    assert instr.layout == LayoutKind.SIT_CENTER
    assert instr.zoom < 1.0
    assert "sit_center tight subject; using contained framing" in warnings


def test_sit_center_full_height_speaker_uses_contained_framing():
    data = {
        "layout": "sit_center",
        "person_bbox": {"x1": 0.334, "y1": 0.039, "x2": 0.692, "y2": 1.0},
        "face_bbox": {"x1": 0.430, "y1": 0.080, "x2": 0.574, "y2": 0.429},
        "reason": "Centered seated speaker already fills the source height.",
    }

    warnings: list[str] = []
    instr = _instruction_from_gemini_json("004", data, warnings=warnings)

    assert instr.layout == LayoutKind.SIT_CENTER
    assert instr.zoom < 1.0
    assert "sit_center tight subject; using contained framing" in warnings


def test_sit_center_without_subject_bbox_uses_contained_framing():
    data = {
        "layout": "sit_center",
        "person_bbox": None,
        "face_bbox": None,
        "chart_bbox": None,
        "reason": "Transition frame with no readable subject to preserve.",
    }

    warnings: list[str] = []
    instr = _instruction_from_gemini_json("004", data, warnings=warnings)

    assert instr.layout == LayoutKind.SIT_CENTER
    assert instr.zoom < 1.0
    assert "sit_center missing subject bbox; using contained framing" in warnings


def test_zoom_call_center_uses_subject_width_not_hard_floor():
    data = {
        "layout": "zoom_call_center",
        "person_bbox": {"x1": 0.10, "y1": 0.10, "x2": 0.75, "y2": 0.95},
        "face_bbox": {"x1": 0.22, "y1": 0.15, "x2": 0.36, "y2": 0.35},
        "reason": "wide already-close subject",
    }
    instr = _instruction_from_gemini_json("009", data)
    assert instr.layout == LayoutKind.ZOOM_CALL_CENTER
    assert instr.zoom == 1.0


def test_call_gemini_vision_uses_bounded_http_retry_budget(monkeypatch, tmp_path):
    img = tmp_path / "frame.jpg"
    img.write_bytes(b"\xff\xd8\xff\xd9")
    frame = SampledFrame(
        frame_id="f0",
        timestamp_sec=1.23,
        path=str(img),
        width=1920,
        height=1080,
    )
    seen: dict[str, object] = {}
    parsed_payload = _GeminiMultiFrameResponse.model_validate(
        {
            "frames": [
                {
                    "frame_index": 0,
                    "timestamp_sec": 1.23,
                    "layout": "sit_center",
                    "reason": "ok",
                }
            ],
            "merged": {
                "layout": "sit_center",
                "reason": "ok",
            },
        }
    )

    def fake_call(request, *, provider):
        seen["request"] = request
        seen["provider"] = provider
        return SimpleNamespace(raw_text=parsed_payload.model_dump_json(), parsed=parsed_payload)

    monkeypatch.setattr("afterform.flows.long_to_shorts.plan_layouts.call_structured_llm", fake_call)

    raw, parsed = _call_gemini_vision([frame], "gemini-3-flash-preview", provider="gemini")

    assert parsed.merged.layout == "sit_center"
    assert raw
    request = seen["request"]
    assert seen["provider"] == "gemini"
    assert request.timeout_ms == _VISION_HTTP_TIMEOUT_MS
    assert request.max_retries == _VISION_RETRY_ATTEMPTS
    assert request.response_schema is _GeminiMultiFrameResponse
    assert len(request.images) == 1


def test_infer_layout_instructions_records_request_budget_on_failure(monkeypatch, tmp_path):
    source = tmp_path / "source.mp4"
    source.write_bytes(b"fake")

    frame = SampledFrame(
        frame_id="f0",
        timestamp_sec=1.23,
        path=str(tmp_path / "frame.jpg"),
        width=1920,
        height=1080,
    )

    clip = SimpleNamespace(
        clip_id="001",
        start_time_sec=0.0,
        end_time_sec=10.0,
        keep_ranges_sec=[(0.0, 10.0)],
    )

    monkeypatch.setattr(
        "afterform.flows.long_to_shorts.plan_layouts._sample_clip_frames",
        lambda *args, **kwargs: ([frame], []),
    )
    monkeypatch.setattr(
        "afterform.flows.long_to_shorts.plan_layouts._call_gemini_vision",
        lambda *args, **kwargs: (_ for _ in ()).throw(RuntimeError("boom")),
    )

    instructions, payload = infer_layout_instructions(
        source,
        [clip],
        gemini_vision_model="gemini-3.1-flash-lite-preview",
        provider="gemini",
        keyframes_root=tmp_path / "keyframes",
    )

    assert instructions["001"].layout == LayoutKind.SIT_CENTER
    clip_payload = payload["001"]
    assert clip_payload["raw"]["request"] == {
        "provider": "gemini",
        "model": "gemini-3.1-flash-lite-preview",
        "frame_count": 1,
        "timeout_ms": _VISION_HTTP_TIMEOUT_MS,
        "max_retries": _VISION_RETRY_ATTEMPTS,
    }
    assert "Layout vision model failed" in clip_payload["warnings"][0]


def test_infer_layout_instructions_uses_layout_hint_when_sampling_unavailable(monkeypatch, tmp_path):
    source = tmp_path / "source.mp4"
    source.write_bytes(b"fake")

    clip = SimpleNamespace(
        clip_id="001",
        start_time_sec=0.0,
        end_time_sec=10.0,
        keep_ranges_sec=[(0.0, 10.0)],
        layout_hint=LayoutKind.SPLIT_CHART_PERSON,
        layout=None,
    )

    monkeypatch.setattr(
        "afterform.flows.long_to_shorts.plan_layouts._sample_clip_frames",
        lambda *args, **kwargs: ([], ["OpenCV unavailable; cannot sample layout frames (No module named 'cv2')."]),
    )

    instructions, payload = infer_layout_instructions(
        source,
        [clip],
        gemini_vision_model="gpt-5.4",
        provider="azure",
        keyframes_root=tmp_path / "keyframes",
    )

    assert instructions["001"].layout == LayoutKind.SPLIT_CHART_PERSON
    clip_payload = payload["001"]
    assert clip_payload["instruction"]["layout"] == "split_chart_person"
    assert clip_payload["raw"]["layout"] == "split_chart_person"
    assert clip_payload["warnings"][-1] == "No sampled frames; defaulted to split_chart_person."


def test_infer_layout_instructions_uses_layout_hint_when_vision_call_fails(monkeypatch, tmp_path):
    source = tmp_path / "source.mp4"
    source.write_bytes(b"fake")

    frame = SampledFrame(
        frame_id="f0",
        timestamp_sec=1.23,
        path=str(tmp_path / "frame.jpg"),
        width=1920,
        height=1080,
    )

    clip = SimpleNamespace(
        clip_id="001",
        start_time_sec=0.0,
        end_time_sec=10.0,
        keep_ranges_sec=[(0.0, 10.0)],
        layout_hint=LayoutKind.SPLIT_CHART_PERSON,
        layout=None,
    )

    monkeypatch.setattr(
        "afterform.flows.long_to_shorts.plan_layouts._sample_clip_frames",
        lambda *args, **kwargs: ([frame], []),
    )
    monkeypatch.setattr(
        "afterform.flows.long_to_shorts.plan_layouts._call_gemini_vision",
        lambda *args, **kwargs: (_ for _ in ()).throw(RuntimeError("boom")),
    )

    instructions, payload = infer_layout_instructions(
        source,
        [clip],
        gemini_vision_model="gpt-5.4",
        provider="azure",
        keyframes_root=tmp_path / "keyframes",
    )

    assert instructions["001"].layout == LayoutKind.SPLIT_CHART_PERSON
    clip_payload = payload["001"]
    assert clip_payload["instruction"]["layout"] == "split_chart_person"
    assert clip_payload["raw"]["request"]["provider"] == "azure"


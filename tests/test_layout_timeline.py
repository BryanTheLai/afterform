from types import SimpleNamespace

import pytest

from afterform.flows.long_to_shorts.plan_layouts import (
    SampledFrame,
    _build_layout_timeline,
    _instruction_from_gemini_json,
    _layout_seed_timestamps,
    _merge_layout_timestamps,
    _uniform_source_timestamps,
    _visual_transition_drop_ranges,
)
from afterform.schemas import (
    BoundingBox,
    Clip,
    ClipLayoutPlan,
    LayoutInstruction,
    LayoutKind,
    LayoutTimelineSegment,
)


def _clip(**overrides) -> Clip:
    data = {
        "clip_id": "004",
        "topic": "mixed visuals",
        "start_time_sec": 100.0,
        "end_time_sec": 160.0,
    }
    data.update(overrides)
    return Clip.model_validate(data)


def _frame(ts: float) -> SampledFrame:
    return SampledFrame(
        frame_id=f"f{ts}",
        timestamp_sec=ts,
        path=f"{ts}.jpg",
        width=1920,
        height=1080,
    )


def _instr(kind: LayoutKind) -> LayoutInstruction:
    return LayoutInstruction(clip_id="004", layout=kind)


def test_uniform_source_timestamps_include_start_and_end_boundaries():
    timestamps = _uniform_source_timestamps([(10.0, 30.0)], 4)

    assert timestamps[0] == pytest.approx(10.0)
    assert timestamps[-1] == pytest.approx(30.0)


def test_layout_seed_timestamps_include_early_output_seconds():
    keep_ranges = [
        (555.6, 557.25),
        (557.45, 564.45),
        (564.75, 569.35),
        (569.95, 574.95),
    ]

    timestamps = _layout_seed_timestamps(
        keep_ranges,
        total_duration=18.25,
        uniform_count=5,
    )

    assert 555.6 in timestamps
    assert 557.8 in timestamps
    assert 560.8 in timestamps


def test_merge_layout_timestamps_preserves_seed_coverage_when_capped():
    seed = [555.6, 557.8, 560.8, 577.2, 602.65, 626.88, 649.7]
    peaks = [558.0, 559.0, 560.0, 561.0, 562.0, 563.0, 564.0, 565.0]

    timestamps = _merge_layout_timestamps(seed, peaks, max_count=8)

    assert 555.6 in timestamps
    assert 557.8 in timestamps
    assert 649.7 in timestamps
    assert len(timestamps) == 8


def test_visual_timeline_boundary_waits_for_visual_evidence_sample():
    clip = _clip()
    frames = [_frame(100.0), _frame(110.0)]
    instructions = [
        _instr(LayoutKind.SIT_CENTER),
        LayoutInstruction(
            clip_id="004",
            layout=LayoutKind.SPLIT_CHART_PERSON,
            split_chart_region=BoundingBox(x1=0.0, y1=0.2, x2=0.45, y2=0.8),
            split_person_region=BoundingBox(x1=0.55, y1=0.0, x2=1.0, y2=0.8),
        ),
    ]

    timeline = _build_layout_timeline(
        clip,
        frames,
        instructions,
        fallback_instruction=_instr(LayoutKind.SIT_CENTER),
    )

    assert timeline[0].end_sec == pytest.approx(9.25)
    assert timeline[1].start_sec == pytest.approx(9.25)


def test_visual_timeline_boundary_waits_for_confirmed_visual_exit():
    clip = _clip()
    frames = [_frame(100.0), _frame(110.0)]
    instructions = [
        LayoutInstruction(
            clip_id="004",
            layout=LayoutKind.SPLIT_CHART_PERSON,
            split_chart_region=BoundingBox(x1=0.0, y1=0.2, x2=0.45, y2=0.8),
            split_person_region=BoundingBox(x1=0.55, y1=0.0, x2=1.0, y2=0.8),
        ),
        _instr(LayoutKind.SIT_CENTER),
    ]

    timeline = _build_layout_timeline(
        clip,
        frames,
        instructions,
        fallback_instruction=_instr(LayoutKind.SIT_CENTER),
    )

    assert timeline[0].end_sec == pytest.approx(10.0)
    assert timeline[1].start_sec == pytest.approx(10.0)


def test_visual_transition_drop_ranges_remove_exit_tail():
    clip = _clip()
    frames = [_frame(100.0), _frame(110.0)]
    instructions = [
        LayoutInstruction(
            clip_id="004",
            layout=LayoutKind.SPLIT_CHART_PERSON,
            split_chart_region=BoundingBox(x1=0.0, y1=0.2, x2=0.45, y2=0.8),
            split_person_region=BoundingBox(x1=0.55, y1=0.0, x2=1.0, y2=0.8),
        ),
        _instr(LayoutKind.SIT_CENTER),
    ]

    assert _visual_transition_drop_ranges(clip, frames, instructions) == [(9.25, 10.0)]


def test_visual_transition_drop_ranges_remove_entry_tail():
    clip = _clip()
    frames = [_frame(100.0), _frame(110.0)]
    instructions = [
        _instr(LayoutKind.SIT_CENTER),
        LayoutInstruction(
            clip_id="004",
            layout=LayoutKind.SPLIT_CHART_PERSON,
            split_chart_region=BoundingBox(x1=0.0, y1=0.2, x2=0.45, y2=0.8),
            split_person_region=BoundingBox(x1=0.55, y1=0.0, x2=1.0, y2=0.8),
        ),
    ]

    assert _visual_transition_drop_ranges(clip, frames, instructions) == [(9.25, 10.0)]


def test_build_layout_timeline_preserves_mixed_frame_layouts():
    clip = _clip()
    frames = [_frame(100.0), _frame(110.0), _frame(130.0), _frame(150.0)]
    instructions = [
        _instr(LayoutKind.SIT_CENTER),
        _instr(LayoutKind.SIT_CENTER),
        _instr(LayoutKind.SPLIT_TWO_PERSONS),
        _instr(LayoutKind.SPLIT_CHART_PERSON),
    ]

    timeline = _build_layout_timeline(
        clip,
        frames,
        instructions,
        fallback_instruction=_instr(LayoutKind.SIT_CENTER),
    )

    assert [segment.instruction.layout for segment in timeline] == [
        LayoutKind.SIT_CENTER,
        LayoutKind.SPLIT_TWO_PERSONS,
        LayoutKind.SPLIT_CHART_PERSON,
    ]
    assert timeline[0].start_sec == pytest.approx(0.0)
    assert timeline[-1].end_sec == pytest.approx(60.0)
    assert timeline[1].start_sec == pytest.approx(20.0)
    assert timeline[1].end_sec == pytest.approx(40.0)


def test_build_layout_timeline_collapses_stable_single_layout():
    clip = _clip()
    frames = [_frame(100.0), _frame(120.0), _frame(140.0)]
    instructions = [_instr(LayoutKind.SIT_CENTER) for _ in frames]

    timeline = _build_layout_timeline(
        clip,
        frames,
        instructions,
        fallback_instruction=_instr(LayoutKind.SIT_CENTER),
    )

    assert len(timeline) == 1
    assert timeline[0].instruction.layout == LayoutKind.SIT_CENTER
    assert timeline[0].start_sec == pytest.approx(0.0)
    assert timeline[0].end_sec == pytest.approx(60.0)


def test_chart_only_frame_promotes_to_wide_visual():
    warnings: list[str] = []
    instr = _instruction_from_gemini_json(
        "005",
        {
            "layout": "sit_center",
            "chart_bbox": {"x1": 72, "y1": 120, "x2": 928, "y2": 645},
            "person_bbox": None,
            "face_bbox": None,
            "reason": "Full-screen article/screenshot dominates frame.",
        },
        warnings=warnings,
    )

    assert instr.layout == LayoutKind.WIDE_VISUAL
    assert instr.split_chart_region is not None
    assert instr.split_chart_region.x1 == pytest.approx(0.072)
    assert "chart_bbox without person; promoted to wide_visual" in warnings


def test_timeline_splits_same_layout_when_bbox_geometry_changes():
    clip = _clip()
    frames = [_frame(100.0), _frame(120.0)]
    instructions = [
        LayoutInstruction(
            clip_id="004",
            layout=LayoutKind.SPLIT_CHART_PERSON,
            split_chart_region=BoundingBox(x1=0.53, y1=0.2, x2=0.97, y2=0.82),
            split_person_region=BoundingBox(x1=0.0, y1=0.0, x2=0.50, y2=0.83),
        ),
        LayoutInstruction(
            clip_id="004",
            layout=LayoutKind.SPLIT_CHART_PERSON,
            split_chart_region=BoundingBox(x1=0.04, y1=0.39, x2=0.47, y2=0.67),
            split_person_region=BoundingBox(x1=0.84, y1=0.05, x2=1.0, y2=0.88),
        ),
    ]

    timeline = _build_layout_timeline(
        clip,
        frames,
        instructions,
        fallback_instruction=_instr(LayoutKind.SIT_CENTER),
    )

    assert [segment.instruction.layout for segment in timeline] == [
        LayoutKind.SPLIT_CHART_PERSON,
        LayoutKind.SPLIT_CHART_PERSON,
    ]
    assert timeline[0].end_sec == pytest.approx(10.0)
    assert timeline[1].start_sec == pytest.approx(10.0)


def test_timeline_splits_sit_center_when_crop_center_changes():
    clip = _clip()
    frames = [_frame(100.0), _frame(120.0)]
    instructions = [
        LayoutInstruction(
            clip_id="004",
            layout=LayoutKind.SIT_CENTER,
            person_x_norm=0.50,
            zoom=1.0,
        ),
        LayoutInstruction(
            clip_id="004",
            layout=LayoutKind.SIT_CENTER,
            person_x_norm=0.455,
            zoom=0.85,
        ),
    ]

    timeline = _build_layout_timeline(
        clip,
        frames,
        instructions,
        fallback_instruction=_instr(LayoutKind.SIT_CENTER),
    )

    assert [segment.instruction.person_x_norm for segment in timeline] == [0.50, 0.455]
    assert [segment.instruction.zoom for segment in timeline] == [1.0, 0.85]


def test_timeline_merges_contained_sit_center_when_only_crop_center_changes():
    clip = _clip()
    frames = [_frame(100.0), _frame(120.0)]
    instructions = [
        LayoutInstruction(
            clip_id="004",
            layout=LayoutKind.SIT_CENTER,
            person_x_norm=0.774,
            zoom=0.85,
        ),
        LayoutInstruction(
            clip_id="004",
            layout=LayoutKind.SIT_CENTER,
            person_x_norm=0.504,
            zoom=0.85,
        ),
    ]

    timeline = _build_layout_timeline(
        clip,
        frames,
        instructions,
        fallback_instruction=_instr(LayoutKind.SIT_CENTER),
    )

    assert len(timeline) == 1
    assert timeline[0].instruction.zoom == 0.85


def test_timeline_ignores_samples_removed_by_keep_ranges():
    clip = _clip(end_time_sec=130.0, keep_ranges_sec=[(0.0, 4.6), (5.2, 30.0)])
    frames = [_frame(100.0), _frame(104.8), _frame(113.0)]
    instructions = [
        LayoutInstruction(
            clip_id="004",
            layout=LayoutKind.SPLIT_CHART_PERSON,
            split_chart_region=BoundingBox(x1=0.03, y1=0.2, x2=0.47, y2=0.8),
            split_person_region=BoundingBox(x1=0.52, y1=0.0, x2=1.0, y2=0.8),
        ),
        LayoutInstruction(
            clip_id="004",
            layout=LayoutKind.SIT_CENTER,
            person_x_norm=0.5,
            zoom=1.0,
        ),
        LayoutInstruction(
            clip_id="004",
            layout=LayoutKind.SIT_CENTER,
            person_x_norm=0.504,
            zoom=0.85,
        ),
    ]

    timeline = _build_layout_timeline(
        clip,
        frames,
        instructions,
        fallback_instruction=_instr(LayoutKind.SIT_CENTER),
    )

    assert not any(
        segment.instruction.layout == LayoutKind.SIT_CENTER and segment.instruction.zoom == 1.0
        for segment in timeline
    )


def test_clip_layout_plan_cache_payload_round_trips_timeline_and_drops():
    plan = ClipLayoutPlan(
        clip_id="004",
        instruction=_instr(LayoutKind.SIT_CENTER),
        layout_timeline=[
            LayoutTimelineSegment(
                start_sec=0.0,
                end_sec=3.0,
                instruction=_instr(LayoutKind.WIDE_VISUAL),
                reason="visual intro",
            )
        ],
        visual_drop_ranges_sec=[(4.2, 4.8)],
    )

    payload = plan.to_layout_cache_payload()
    restored = ClipLayoutPlan.from_layout_cache_payload("004", payload)

    assert restored == plan
    assert payload["layout_timeline"][0]["instruction"]["layout"] == "wide_visual"
    assert payload["visual_drop_ranges_sec"] == [[4.2, 4.8]]


def test_infer_layout_instructions_payload_contains_layout_timeline(monkeypatch, tmp_path):
    from afterform.flows.long_to_shorts.plan_layouts import infer_layout_instructions

    source = tmp_path / "source.mp4"
    source.write_bytes(b"fake")
    frames = [_frame(100.0), _frame(120.0), _frame(140.0)]

    parsed = SimpleNamespace(
        frames=[
            SimpleNamespace(model_dump=lambda: {"layout": "sit_center", "reason": "speaker"}),
            SimpleNamespace(
                model_dump=lambda: {
                    "layout": "split_two_persons",
                    "person_bbox": {"x1": 100, "y1": 0, "x2": 450, "y2": 1000},
                    "second_person_bbox": {
                        "x1": 550,
                        "y1": 0,
                        "x2": 950,
                        "y2": 1000,
                    },
                    "reason": "two people",
                }
            ),
            SimpleNamespace(
                model_dump=lambda: {
                    "layout": "split_chart_person",
                    "person_bbox": {"x1": 550, "y1": 0, "x2": 950, "y2": 1000},
                    "chart_bbox": {"x1": 0, "y1": 0, "x2": 500, "y2": 800},
                    "reason": "chart plus speaker",
                }
            ),
        ],
        merged=SimpleNamespace(
            model_dump=lambda: {"layout": "sit_center", "reason": "dominant speaker"}
        ),
    )

    monkeypatch.setattr(
        "afterform.flows.long_to_shorts.plan_layouts._sample_clip_frames",
        lambda *args, **kwargs: (frames, []),
    )
    monkeypatch.setattr(
        "afterform.flows.long_to_shorts.plan_layouts._call_gemini_vision",
        lambda *args, **kwargs: ("{}", parsed),
    )

    plans, payload = infer_layout_instructions(
        source,
        [_clip()],
        gemini_vision_model="gpt-5.4",
        provider="azure",
        keyframes_root=tmp_path / "keyframes",
    )

    assert plans["004"].instruction.layout == LayoutKind.SIT_CENTER
    assert [s.instruction.layout for s in plans["004"].layout_timeline] == [
        LayoutKind.SIT_CENTER,
        LayoutKind.SPLIT_TWO_PERSONS,
        LayoutKind.SPLIT_CHART_PERSON,
    ]
    assert [s["instruction"]["layout"] for s in payload["004"]["layout_timeline"]] == [
        "sit_center",
        "split_two_persons",
        "split_chart_person",
    ]

import pytest

from afterform.schemas import (
    Clip,
    ClipPlan,
    ClipSubtitleWords,
    FocusStackOrder,
    LayoutInstruction,
    LayoutKind,
    RenderRequest,
    Scene,
    TranscriptWord,
)


def test_scene_requires_end_after_start():
    Scene(scene_id="s1", start_time=0.0, end_time=1.0)
    with pytest.raises(ValueError):
        Scene(scene_id="s1", start_time=5.0, end_time=5.0)
    with pytest.raises(ValueError):
        Scene(scene_id="s1", start_time=5.0, end_time=1.0)


def test_layout_instruction_defaults_and_bounds():
    li = LayoutInstruction(clip_id="c", layout=LayoutKind.SIT_CENTER)
    assert li.zoom == 1.0
    assert 0 <= li.person_x_norm <= 1
    assert li.focus_stack_order == FocusStackOrder.CHART_THEN_PERSON
    with pytest.raises(ValueError):
        LayoutInstruction(clip_id="c", layout=LayoutKind.SIT_CENTER, zoom=0.0)
    with pytest.raises(ValueError):
        LayoutInstruction(clip_id="c", layout=LayoutKind.SIT_CENTER, person_x_norm=2.0)


def test_clip_duration():
    c = Clip(
        clip_id="1",
        topic="t",
        start_time_sec=10.0,
        end_time_sec=42.5,
    )
    assert c.duration_sec == pytest.approx(32.5)


def test_clip_hook_relative_to_clip_in_point():
    c = Clip(
        clip_id="1",
        topic="t",
        start_time_sec=100.0,
        end_time_sec=130.0,
        hook_start_sec=0.0,
        hook_end_sec=3.0,
    )
    assert c.hook_end_sec == 3.0


def test_clip_hook_must_be_within_duration():
    with pytest.raises(ValueError, match="hook window"):
        Clip(
            clip_id="1",
            topic="t",
            start_time_sec=0.0,
            end_time_sec=10.0,
            hook_start_sec=0.0,
            hook_end_sec=15.0,
        )


def test_clip_hook_both_or_neither():
    with pytest.raises(ValueError, match="hook_start_sec and hook_end_sec"):
        Clip(
            clip_id="1",
            topic="t",
            start_time_sec=0.0,
            end_time_sec=10.0,
            hook_start_sec=1.0,
            hook_end_sec=None,
        )


def test_clip_trim_cannot_exceed_duration():
    with pytest.raises(ValueError, match="trim"):
        Clip(
            clip_id="1",
            topic="t",
            start_time_sec=0.0,
            end_time_sec=10.0,
            trim_start_sec=6.0,
            trim_end_sec=6.0,
        )


def test_clip_plan_roundtrip():
    plan = ClipPlan(
        source_path="/tmp/x.mp4",
        clips=[
            Clip(clip_id="1", topic="t", start_time_sec=0.0, end_time_sec=30.0)
        ],
    )
    d = plan.model_dump()
    assert ClipPlan.model_validate(d) == plan


def test_clip_subtitle_words_relative_times():
    w = ClipSubtitleWords(
        words=[TranscriptWord(word="hi", start_time=0.0, end_time=0.2)]
    )
    assert w.words[0].start_time == 0.0


def test_render_request_modes():
    c = Clip(clip_id="1", topic="t", start_time_sec=0.0, end_time_sec=30.0)
    li = LayoutInstruction(clip_id="1", layout=LayoutKind.ZOOM_CALL_CENTER)
    req = RenderRequest(
        source_path="/tmp/x.mp4",
        clip=c,
        layout=li,
        output_path="/tmp/out.mp4",
    )
    assert req.mode == "normal"
    req2 = RenderRequest(**{**req.model_dump(), "mode": "dry_run"})
    assert req2.mode == "dry_run"


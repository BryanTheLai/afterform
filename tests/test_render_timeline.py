from pathlib import Path

import pytest

from afterform.flows.long_to_shorts.render import reframe_clip_timeline_ffmpeg
from afterform.flows.long_to_shorts.render_window import subclip_for_output_range
from afterform.schemas import (
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
        "keep_ranges_sec": [(0.0, 10.0), (20.0, 40.0), (45.0, 60.0)],
    }
    data.update(overrides)
    return Clip.model_validate(data)


def _instr(kind: LayoutKind) -> LayoutInstruction:
    return LayoutInstruction(clip_id="004", layout=kind)


def test_subclip_for_output_range_maps_across_keep_range_gaps():
    clip = _clip()

    subclip = subclip_for_output_range(clip, 8.0, 22.0)

    assert subclip.start_time_sec == pytest.approx(108.0)
    assert subclip.end_time_sec == pytest.approx(132.0)
    assert subclip.keep_ranges_sec == [(0.0, 2.0), (12.0, 24.0)]


def test_timeline_render_uses_segment_layouts_then_final_subtitle_pass(monkeypatch, tmp_path):
    clip = _clip(keep_ranges_sec=[])
    plan = ClipLayoutPlan(
        clip_id="004",
        instruction=_instr(LayoutKind.SIT_CENTER),
        layout_timeline=[
            LayoutTimelineSegment(
                start_sec=0.0,
                end_sec=15.0,
                instruction=_instr(LayoutKind.SIT_CENTER),
                reason="speaker",
            ),
            LayoutTimelineSegment(
                start_sec=15.0,
                end_sec=60.0,
                instruction=_instr(LayoutKind.SPLIT_CHART_PERSON),
                reason="visual",
            ),
        ],
    )
    source = tmp_path / "source.mp4"
    source.write_bytes(b"fake")
    subtitle = tmp_path / "clip.ass"
    subtitle.write_text("[Script Info]\n", encoding="utf-8")
    output = tmp_path / "short_004.mp4"

    rendered_layouts: list[LayoutKind] = []
    subprocess_calls: list[list[str]] = []

    def fake_reframe(**kwargs):
        rendered_layouts.append(kwargs["layout_instruction"].layout)
        Path(kwargs["output_path"]).write_bytes(b"segment")

    def fake_run(cmd, check=False, capture_output=False):
        subprocess_calls.append(list(cmd))
        Path(cmd[-1]).write_bytes(b"out")
        return object()

    monkeypatch.setattr("afterform.flows.long_to_shorts.render.reframe_clip_ffmpeg", fake_reframe)
    monkeypatch.setattr("afterform.flows.long_to_shorts.render.subprocess.run", fake_run)
    monkeypatch.setattr("afterform.flows.long_to_shorts.render.shutil.which", lambda exe: exe)

    reframe_clip_timeline_ffmpeg(
        input_path=source,
        output_path=output,
        clip=clip,
        layout_plan=plan,
        subtitle_path=subtitle,
        title_text="Failed Founders",
    )

    assert rendered_layouts == [LayoutKind.SIT_CENTER, LayoutKind.SPLIT_CHART_PERSON]
    assert any("-f" in cmd and "concat" in cmd for cmd in subprocess_calls)
    final_cmd = subprocess_calls[-1]
    assert "subtitles='" in final_cmd[final_cmd.index("-filter_complex") + 1]
    assert str(output) == final_cmd[-1]

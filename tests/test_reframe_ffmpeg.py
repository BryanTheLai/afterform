"""Tests for the ffmpeg-backend reframer in ``afterform.flows.long_to_shorts``.

These verify the instruction-building logic with ``dry_run=True`` so no
ffmpeg process is spawned.
"""

import pytest

from afterform.flows.long_to_shorts.render import layout_for_clip, reframe_clip_ffmpeg
from afterform.schemas import Clip, LayoutKind


def _clip() -> Clip:
    return Clip(
        clip_id="c1",
        topic="inflation",
        start_time_sec=0.0,
        end_time_sec=30.0,
        viral_hook="hook",
        transcript="words",
        suggested_overlay_title="Title",
    )


def test_layout_for_clip_defaults_centered_without_regions():
    instr = layout_for_clip(_clip())
    assert instr.layout == LayoutKind.SIT_CENTER
    assert instr.person_x_norm == 0.5
    assert instr.chart_x_norm == 0.0


def test_layout_for_clip_respects_clip_layout_hint():
    c = _clip()
    c.layout = LayoutKind.ZOOM_CALL_CENTER
    instr = layout_for_clip(c)
    assert instr.layout == LayoutKind.ZOOM_CALL_CENTER


def test_reframe_clip_ffmpeg_dry_run_builds_cmd(tmp_path):
    out = tmp_path / "out.mp4"
    subtitle = tmp_path / "clip.srt"
    subtitle.write_text("1\n00:00:00,000 --> 00:00:01,000\nhello\n", encoding="utf-8")
    req = reframe_clip_ffmpeg(
        input_path=tmp_path / "src.mp4",
        output_path=out,
        clip=_clip(),
        subtitle_path=subtitle,
        dry_run=True,
    )
    assert req.mode == "dry_run"
    assert str(req.output_path) == str(out)
    assert req.clip.clip_id == "c1"
    assert req.subtitle_path == str(subtitle)


def test_reframe_clip_ffmpeg_raises_on_missing_source(tmp_path):
    # With mode='normal' and a nonexistent source, ffmpeg must fail and we
    # must surface that as a RuntimeError rather than silently returning.
    out = tmp_path / "out.mp4"
    with pytest.raises(RuntimeError):
        reframe_clip_ffmpeg(
            input_path=tmp_path / "does_not_exist.mp4",
            output_path=out,
            clip=_clip(),
            dry_run=False,
        )

import shutil
import subprocess

import pytest

from afterform.primitives.compile import build_ffmpeg_cmd, plan_title_drawtext
from afterform.schemas import Clip, LayoutInstruction, LayoutKind, RenderRequest


def _req(**overrides):
    c = Clip(clip_id="1", topic="t", start_time_sec=10.0, end_time_sec=40.0)
    li = LayoutInstruction(clip_id="1", layout=LayoutKind.SIT_CENTER)
    data = dict(
        source_path="/tmp/src.mp4",
        clip=c,
        layout=li,
        output_path="/tmp/out.mp4",
        mode="dry_run",
    )
    data.update(overrides)
    return RenderRequest(**data)


def test_ffmpeg_cmd_has_ss_duration_filtergraph_output():
    cmd = build_ffmpeg_cmd(_req())
    assert "-ss" in cmd
    assert "-t" in cmd
    assert "-filter_complex" in cmd
    # duration = 30.0
    t_idx = cmd.index("-t")
    assert float(cmd[t_idx + 1]) == 30.0
    ss_idx = cmd.index("-ss")
    assert float(cmd[ss_idx + 1]) == 10.0
    assert cmd[-1] == "/tmp/out.mp4"


def test_title_text_injects_drawtext():
    cmd = build_ffmpeg_cmd(_req(title_text="Hello: world's"))
    fg = cmd[cmd.index("-filter_complex") + 1]
    assert "drawtext" in fg
    # colon should be escaped
    assert "Hello\\:" in fg
    assert "worlds" in fg
    assert "world's" not in fg
    assert "expansion=none" in fg


def test_map_vout_and_primary_audio():
    cmd = build_ffmpeg_cmd(_req())
    assert "[vout]" in cmd
    assert "[aout]" in cmd
    fg = cmd[cmd.index("-filter_complex") + 1]
    assert "[0:a:0]volume=8dB,alimiter=limit=0.95,aresample=44100[aout]" in fg


def test_inner_keep_ranges_switch_to_concat_filtergraph():
    clip = Clip(
        clip_id="1",
        topic="t",
        start_time_sec=10.0,
        end_time_sec=40.0,
        keep_ranges_sec=[(0.0, 4.0), (6.0, 10.0)],
    )
    cmd = build_ffmpeg_cmd(_req(clip=clip))
    fg = cmd[cmd.index("-filter_complex") + 1]
    assert "trim=start=10.000:end=14.000" in fg
    assert "atrim=start=16.000:end=20.000" in fg
    assert "concat=n=2:v=1:a=1[vclip][aclip]" in fg
    assert "[aclip]volume=8dB,alimiter=limit=0.95,aresample=44100[aout]" in fg
    assert "[aout]" in cmd


def test_keep_ranges_trim_absolute_source_without_input_seek():
    """Interior keep-range gaps must be removed on the source timeline.

    Regression for rendered shorts leaking transition/junk frames: input-level
    ``-ss/-t`` plus relative concat trims can decode the enclosing continuous
    source window instead of the explicit kept spans.
    """
    clip = Clip(
        clip_id="1",
        topic="t",
        start_time_sec=100.0,
        end_time_sec=120.0,
        keep_ranges_sec=[(0.0, 2.0), (5.0, 7.0)],
    )
    cmd = build_ffmpeg_cmd(_req(clip=clip))
    input_idx = cmd.index("-i")
    assert "-ss" not in cmd[:input_idx]
    assert "-t" not in cmd[:input_idx]

    fg = cmd[cmd.index("-filter_complex") + 1]
    assert "trim=start=100.000:end=102.000" in fg
    assert "atrim=start=105.000:end=107.000" in fg
    assert "trim=start=0.000:end=2.000" not in fg


def test_keep_ranges_render_only_selected_source_spans(tmp_path):
    ffmpeg = shutil.which("ffmpeg")
    ffprobe = shutil.which("ffprobe")
    if not ffmpeg or not ffprobe:
        pytest.skip("ffmpeg/ffprobe not available")

    src = tmp_path / "source.mp4"
    out = tmp_path / "out.mp4"
    colors = ["red", "green", "blue", "yellow", "magenta", "cyan", "white", "black"]
    gen_cmd = [ffmpeg, "-y"]
    for color in colors:
        gen_cmd.extend(["-f", "lavfi", "-i", f"color=c={color}:s=160x90:d=1:r=5"])
    concat_inputs = "".join(f"[{idx}:v]" for idx in range(len(colors)))
    gen_cmd.extend(
        [
            "-filter_complex",
            f"{concat_inputs}concat=n={len(colors)}:v=1:a=0,format=yuv420p[v]",
            "-map",
            "[v]",
            str(src),
        ]
    )
    subprocess.run(gen_cmd, check=True, capture_output=True)

    clip = Clip(
        clip_id="1",
        topic="t",
        start_time_sec=0.0,
        end_time_sec=8.0,
        keep_ranges_sec=[(1.0, 2.0), (4.0, 5.0)],
    )
    req = _req(source_path=str(src), output_path=str(out), clip=clip)
    cmd = build_ffmpeg_cmd(req, src_w=160, src_h=90, include_audio=False)
    subprocess.run(cmd, check=True, capture_output=True)

    duration = subprocess.run(
        [
            ffprobe,
            "-v",
            "error",
            "-show_entries",
            "format=duration",
            "-of",
            "default=noprint_wrappers=1:nokey=1",
            str(out),
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    assert float(duration.stdout.strip()) == pytest.approx(2.0, abs=0.25)

    def center_pixel(ts: float) -> tuple[int, int, int]:
        sample = subprocess.run(
            [
                ffmpeg,
                "-v",
                "error",
                "-ss",
                f"{ts:.3f}",
                "-i",
                str(out),
                "-frames:v",
                "1",
                "-vf",
                "crop=1:1:540:960,format=rgb24",
                "-f",
                "rawvideo",
                "-",
            ],
            check=True,
            capture_output=True,
        )
        return tuple(sample.stdout[:3])

    r, g, b = center_pixel(0.5)
    assert g > r + 40 and g > b + 40
    r, g, b = center_pixel(1.5)
    assert r > g + 40 and b > g + 40


def test_subtitle_style_uses_requested_font_and_margin():
    cmd = build_ffmpeg_cmd(
        _req(subtitle_path="/tmp/clip.srt", subtitle_font_size=18, subtitle_margin_v=64)
    )
    fg = cmd[cmd.index("-filter_complex") + 1]
    assert "subtitles='" in fg
    assert "FontSize=18" in fg
    assert "MarginV=64" in fg
    # Smart word wrap so long captions break into multiple readable lines.
    assert "WrapStyle=0" in fg


def test_subtitle_original_size_pins_libass_to_output_resolution():
    """Without original_size=W x H, libass uses PlayResY=288 and blows up fonts/margins.

    This is the root cause of the "subtitles floating in the middle of the
    frame / blocked" bug the user reported.
    """
    cmd = build_ffmpeg_cmd(_req(subtitle_path="/tmp/clip.srt"))
    fg = cmd[cmd.index("-filter_complex") + 1]
    assert "original_size=1080x1920" in fg


def test_subtitles_applied_after_crop_and_title():
    """Order: crop/compose -> drawtext title -> subtitles.

    The pipeline must crop **first**, then draw text on the finished frame.
    """
    cmd = build_ffmpeg_cmd(
        _req(title_text="Hook", subtitle_path="/tmp/clip.srt")
    )
    fg = cmd[cmd.index("-filter_complex") + 1]
    crop_pos = fg.index("[0:v]crop=")
    drawtext_pos = fg.index("drawtext")
    subs_pos = fg.index("subtitles=")
    assert crop_pos < drawtext_pos < subs_pos


def test_build_is_layout_specific():
    c = Clip(clip_id="1", topic="t", start_time_sec=0, end_time_sec=10)
    split_req = _req(
        clip=c,
        layout=LayoutInstruction(clip_id="1", layout=LayoutKind.SPLIT_CHART_PERSON),
    )
    cmd = build_ffmpeg_cmd(split_req)
    fg = cmd[cmd.index("-filter_complex") + 1]
    assert "vstack" in fg


def test_title_is_suppressed_on_split_layouts():
    """Split layouts already contain a slide/chart with its own title.

    Overlaying an additional drawtext title just obscures content -- that's
    what was happening in the Cathy Wood "chart overlaps subject" report.
    """
    for kind in (
        LayoutKind.SPLIT_CHART_PERSON,
        LayoutKind.SPLIT_TWO_PERSONS,
        LayoutKind.SPLIT_TWO_CHARTS,
        LayoutKind.WIDE_VISUAL,
    ):
        cmd = build_ffmpeg_cmd(
            _req(
                layout=LayoutInstruction(clip_id="1", layout=kind),
                title_text="This should not render",
            )
        )
        fg = cmd[cmd.index("-filter_complex") + 1]
        assert "drawtext" not in fg, f"title leaked into split layout {kind}"


def test_title_is_drawn_on_single_subject_layouts():
    """Titles are still rendered on ZOOM_CALL_CENTER and SIT_CENTER."""
    for kind in (LayoutKind.ZOOM_CALL_CENTER, LayoutKind.SIT_CENTER):
        cmd = build_ffmpeg_cmd(
            _req(
                layout=LayoutInstruction(clip_id="1", layout=kind),
                title_text="Hook title",
            )
        )
        fg = cmd[cmd.index("-filter_complex") + 1]
        assert "drawtext=text='Hook title'" in fg


# ---------------------------------------------------------------------------
# Title wrapping / auto-shrink (P2: fixes the "Prediction Markets vs
# Derivatives" clipped-title bug reported against the Cathy Wood run).
# ---------------------------------------------------------------------------


def test_plan_title_short_stays_single_line_at_72px():
    """Backward compat: short titles keep the pre-P2 single-drawtext form.

    Byte-identical output for short titles is important because it keeps
    previously-calibrated visual output unchanged and avoids needless cache
    churn on existing renders.
    """
    frag = plan_title_drawtext("Hook title", out_w=1080)
    assert frag is not None
    assert frag.count("drawtext=") == 1
    assert "fontsize=72" in frag
    assert "y=80" in frag
    assert "drawtext=text='Hook title'" in frag


def test_plan_title_long_wraps_to_two_lines_below_72px():
    """Long titles wrap at the best word boundary and shrink to fit.

    "Prediction Markets vs Derivatives" is 33 chars â€” it overflows a 1080px
    canvas at 72px. It must wrap into "Prediction Markets" / "vs Derivatives"
    (balanced halves) at a smaller font.
    """
    frag = plan_title_drawtext("Prediction Markets vs Derivatives", out_w=1080)
    assert frag is not None
    assert frag.count("drawtext=") == 2, "long titles must split into two drawtext calls"
    assert "drawtext=text='Prediction Markets'" in frag
    assert "drawtext=text='vs Derivatives'" in frag
    assert "fontsize=72" not in frag, "two-line layout must use a smaller font"
    # Both lines share the same shrunken fontsize.
    import re

    sizes = re.findall(r"fontsize=(\d+)", frag)
    assert len(sizes) == 2 and sizes[0] == sizes[1]
    assert 44 <= int(sizes[0]) <= 64


def test_plan_title_empty_returns_none():
    assert plan_title_drawtext("", out_w=1080) is None
    assert plan_title_drawtext("   ", out_w=1080) is None


def test_plan_title_single_huge_word_shrinks_instead_of_wrapping():
    """A single word cannot be word-wrapped; it must shrink to fit."""
    frag = plan_title_drawtext("Supercalifragilisticexpialidocious", out_w=1080)
    assert frag is not None
    assert frag.count("drawtext=") == 1  # no wrap possible
    assert "fontsize=72" not in frag


def test_title_uses_arial_font_not_default_serif():
    """Titles must render in Arial (matching the ASS subtitle font), not the
    platform default which is Times New Roman on Windows.

    Regression test for the "ugly serif title on the finance short" bug.
    Both the single-line and the two-line drawtext variants must carry a
    ``font=Arial`` directive so fontconfig resolves to the same family as
    the subtitle ``Fontname=Arial``.
    """
    short = plan_title_drawtext("Hook title", out_w=1080)
    assert short is not None
    assert "font=Arial" in short

    long_frag = plan_title_drawtext("Prediction Markets vs Derivatives", out_w=1080)
    assert long_frag is not None
    # Two drawtext calls => font directive appears twice, once per line.
    assert long_frag.count("font=Arial") == 2


def test_title_font_matches_subtitle_font_family():
    """Title overlay and subtitle captions must read as one typographic
    family. Both routes through ``build_ffmpeg_cmd`` should carry the same
    Arial reference.
    """
    cmd = build_ffmpeg_cmd(
        _req(
            title_text="Hook title",
            subtitle_path="/tmp/clip.ass",
        )
    )
    fg = cmd[cmd.index("-filter_complex") + 1]
    assert "font=Arial" in fg
    assert "Fontname=Arial" in fg


def test_long_title_pipes_through_build_ffmpeg_cmd():
    """End-to-end: a long title routed through the full command builder
    produces a valid filtergraph with two drawtext filters and no syntax
    errors ffmpeg would choke on.
    """
    cmd = build_ffmpeg_cmd(_req(title_text="Prediction Markets vs Derivatives"))
    fg = cmd[cmd.index("-filter_complex") + 1]
    assert fg.count("drawtext=") == 2
    assert "[v_prepad]drawtext=text='Prediction Markets'" in fg
    assert "[vout]" in fg
    assert ";;" not in fg  # no empty chain links
    assert ",," not in fg  # no stray commas


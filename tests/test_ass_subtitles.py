"""ASS caption output: PlayResY must match the output video so libass' font
and margin scaling is 1:1. This was the root cause of the "captions are
huge and floating in the middle of the frame" bug.
"""

from __future__ import annotations

from afterform.flows.long_to_shorts.transcript_align import format_ass


def test_play_res_matches_output_so_libass_scale_is_one_to_one():
    ass = format_ass(
        [(0.0, 1.0, "hello")],
        play_res_x=1080,
        play_res_y=1920,
        font_size=48,
        margin_v=160,
    )
    assert "PlayResX: 1080" in ass
    assert "PlayResY: 1920" in ass


def test_style_row_encodes_font_size_and_margins():
    ass = format_ass(
        [(0.0, 1.0, "hello")],
        play_res_x=1080,
        play_res_y=1920,
        font_size=48,
        margin_v=160,
        margin_h=60,
        font_name="Arial",
    )
    assert "Style: Default,Arial,48," in ass
    # ...,MarginL,MarginR,MarginV,Encoding
    assert ",60,60,160,0" in ass


def test_dialogue_times_are_formatted_as_h_mm_ss_cs():
    ass = format_ass(
        [(0.0, 1.234, "hi"), (12.50, 15.999, "bye")],
        play_res_x=1080,
        play_res_y=1920,
        font_size=48,
        margin_v=160,
    )
    assert "Dialogue: 0,0:00:00.00,0:00:01.23,Default,," in ass
    # 15.999s centiseconds clamp to 99 (no carry into the seconds place).
    assert "Dialogue: 0,0:00:12.50,0:00:15.99,Default,," in ass


def test_escapes_curly_braces_and_newlines():
    ass = format_ass(
        [(0.0, 1.0, "a {weird} line\nbreak")],
        play_res_x=1080,
        play_res_y=1920,
        font_size=48,
        margin_v=160,
    )
    assert r"\{weird\}" in ass
    assert r"\N" in ass


def test_wrap_style_and_opaque_box_background():
    """WrapStyle=0 = smart line break; BorderStyle=4 = opaque box behind text."""
    ass = format_ass(
        [],
        play_res_x=1080,
        play_res_y=1920,
        font_size=48,
        margin_v=160,
    )
    assert "WrapStyle: 0" in ass
    # BorderStyle=4 is the 16th field in the style row.
    assert ",4,0,0,2," in ass


def test_empty_cue_list_still_produces_valid_header():
    ass = format_ass(
        [],
        play_res_x=1080,
        play_res_y=1920,
        font_size=48,
        margin_v=160,
    )
    for required in (
        "[Script Info]",
        "PlayResY: 1920",
        "[V4+ Styles]",
        "Style: Default,",
        "[Events]",
        "Format: Layer, Start, End, Style",
    ):
        assert required in ass, f"missing '{required}' in minimal ASS output"


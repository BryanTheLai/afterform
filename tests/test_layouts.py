import re

from afterform.primitives.layouts import (
    _center_crop_to_9x16,
    _crop_box,
    plan_layout,
)
from afterform.schemas import (
    BoundingBox,
    FocusStackOrder,
    LayoutInstruction,
    LayoutKind,
)


def test_crop_box_aspect_exact():
    cw, ch, x, y = _crop_box(1920, 1080, 9 / 16, 1.0, 0.5, 0.5)
    # 9:16 inside 1920x1080 -> height-limited: ch=1080, cw ~= 608
    assert ch == 1080
    assert abs(cw / ch - 9 / 16) < 0.01
    assert 0 <= x <= 1920 - cw
    assert y == 0


def test_crop_box_clamps_inside_frame():
    cw, ch, x, y = _crop_box(1920, 1080, 9 / 16, 2.0, 0.99, 0.5)
    assert x + cw <= 1920
    assert y + ch <= 1080


def test_crop_box_zoom_tightens():
    cw_small, ch_small, _, _ = _center_crop_to_9x16(1920, 1080, 2.0, 0.5)
    cw_large, ch_large, _, _ = _center_crop_to_9x16(1920, 1080, 1.0, 0.5)
    assert cw_small < cw_large
    assert ch_small < ch_large


def test_even_dimensions():
    cw, ch, x, y = _crop_box(1921, 1081, 9 / 16, 1.3, 0.4, 0.5)
    assert cw % 2 == 0 and ch % 2 == 0
    assert x % 2 == 0 and y % 2 == 0


def _contains(s: str, *subs: str) -> bool:
    return all(sub in s for sub in subs)


def test_zoom_call_layout_filtergraph_shape():
    instr = LayoutInstruction(
        clip_id="c", layout=LayoutKind.ZOOM_CALL_CENTER, zoom=1.5, person_x_norm=0.5
    )
    plan = plan_layout(instr, out_w=1080, out_h=1920)
    fg = plan.filtergraph
    assert _contains(fg, "[0:v]crop=", "scale=1080:1920", "[vout]")


def test_sit_center_layout_filtergraph_shape():
    instr = LayoutInstruction(clip_id="c", layout=LayoutKind.SIT_CENTER)
    plan = plan_layout(instr, out_w=1080, out_h=1920)
    assert "[vout]" in plan.filtergraph
    assert plan.out_label == "vout"


def test_split_layout_contains_vstack():
    instr = LayoutInstruction(
        clip_id="c",
        layout=LayoutKind.SPLIT_CHART_PERSON,
        person_x_norm=0.83,
        chart_x_norm=0.0,
    )
    plan = plan_layout(instr, out_w=1080, out_h=1920)
    fg = plan.filtergraph
    assert _contains(fg, "split=2", "vstack=inputs=2", "[vout]")
    assert "[top]" in fg and "[bot]" in fg


def test_split_layout_person_crop_is_right_third():
    """Chart uses left 2/3; person uses right 1/3 (non-overlapping)."""
    instr = LayoutInstruction(clip_id="c", layout=LayoutKind.SPLIT_CHART_PERSON)
    fg = plan_layout(instr, out_w=1080, out_h=1920, src_w=1920, src_h=1080).filtergraph
    # Right third: x=1280, w=640 for 1920-wide source.
    assert "crop=640:1080:1280:0" in fg


def test_split_layout_can_swap_stack_order():
    """PERSON_THEN_CHART puts the right-strip (person) crop into the top band."""
    chart_first = plan_layout(
        LayoutInstruction(
            clip_id="c",
            layout=LayoutKind.SPLIT_CHART_PERSON,
            focus_stack_order=FocusStackOrder.CHART_THEN_PERSON,
        ),
        out_w=1080,
        out_h=1920,
    ).filtergraph
    person_first = plan_layout(
        LayoutInstruction(
            clip_id="c",
            layout=LayoutKind.SPLIT_CHART_PERSON,
            focus_stack_order=FocusStackOrder.PERSON_THEN_CHART,
        ),
        out_w=1080,
        out_h=1920,
    ).filtergraph

    def top_crop(fg: str) -> str:
        m = re.search(r"\[src1\]crop=(\d+:\d+:\d+:\d+)", fg)
        assert m is not None, fg
        return m.group(1)

    # chart strip = left 1280px of source (2/3 split seam).
    assert top_crop(chart_first) == "1280:1080:0:0"
    # person strip = right 640px -> x=1280.
    assert top_crop(person_first) == "640:1080:1280:0"
    assert "vstack=inputs=2" in chart_first
    assert "vstack=inputs=2" in person_first


def test_split_layout_person_clamped():
    instr = LayoutInstruction(
        clip_id="c", layout=LayoutKind.SPLIT_CHART_PERSON, person_x_norm=1.0
    )
    plan = plan_layout(instr, out_w=1080, out_h=1920)
    assert "crop=" in plan.filtergraph  # no OOB math crash


def test_plan_layout_dispatch_covers_all_kinds():
    for k in LayoutKind:
        instr = LayoutInstruction(clip_id="c", layout=k)
        plan = plan_layout(instr)
        assert plan.out_label == "vout"
        assert plan.filtergraph.endswith("[vout]")


def test_default_split_is_even_50_50_bands():
    """The user-requested symmetric look: top and bottom bands are equal."""
    instr = LayoutInstruction(clip_id="c", layout=LayoutKind.SPLIT_CHART_PERSON)
    fg = plan_layout(instr, out_w=1080, out_h=1920).filtergraph
    # Each strip should scale to the same height (half of 1920).
    heights = re.findall(r"scale=1080:(\d+):force_original_aspect_ratio", fg)
    assert heights == ["960", "960", "960"], heights


def test_top_band_ratio_honored_for_uneven_splits():
    instr = LayoutInstruction(
        clip_id="c", layout=LayoutKind.SPLIT_CHART_PERSON, top_band_ratio=0.6
    )
    fg = plan_layout(instr, out_w=1080, out_h=1920).filtergraph
    heights = re.findall(r"scale=1080:(\d+):force_original_aspect_ratio", fg)
    assert heights == ["1152", "1152", "768"], heights


def test_split_seam_is_midpoint_between_bboxes():
    """When both bboxes are provided, strips partition the source -- no overlap, no gap."""
    instr = LayoutInstruction(
        clip_id="c",
        layout=LayoutKind.SPLIT_CHART_PERSON,
        split_chart_region=BoundingBox(x1=0.0, y1=0.0, x2=0.50, y2=1.0),
        split_person_region=BoundingBox(x1=0.55, y1=0.0, x2=1.0, y2=1.0),
    )
    fg = plan_layout(instr, out_w=1080, out_h=1920, src_w=1920, src_h=1080).filtergraph
    # chart.x2 = 960px, person.x1 = 1056px -> midpoint = 1008 -> even -> 1008.
    # Chart strip: x=0, cw=1008. Person strip: x=1008, cw=912.
    top_crop = re.search(r"\[src1\]crop=(\d+:\d+:\d+:\d+)", fg).group(1)
    bot_crop = re.search(r"\[src2\]crop=(\d+:\d+:\d+:\d+)", fg).group(1)
    assert top_crop == "1008:1080:0:0"
    assert bot_crop == "912:1080:1008:0"


def test_split_chart_person_handles_person_left_visual_right():
    instr = LayoutInstruction(
        clip_id="c",
        layout=LayoutKind.SPLIT_CHART_PERSON,
        split_person_region=BoundingBox(x1=0.0, y1=0.0, x2=0.50, y2=0.83),
        split_chart_region=BoundingBox(x1=0.53, y1=0.2, x2=0.97, y2=0.82),
    )

    fg = plan_layout(instr, out_w=1080, out_h=1920, src_w=1920, src_h=1080).filtergraph

    # Person-left / visual-right must put the visual crop on the right side
    # and the person crop on the left side. The old code assumed chart-left.
    assert "[src1]crop=932:" in fg
    assert ":988:" in fg
    assert "[src2]crop=988:" in fg
    assert ":0:" in fg


def test_split_chart_person_focuses_right_edge_person_bbox():
    instr = LayoutInstruction(
        clip_id="c",
        layout=LayoutKind.SPLIT_CHART_PERSON,
        split_chart_region=BoundingBox(x1=0.03, y1=0.39, x2=0.47, y2=0.67),
        split_person_region=BoundingBox(x1=0.84, y1=0.05, x2=1.0, y2=0.72),
    )

    fg = plan_layout(instr, out_w=1080, out_h=1920, src_w=1920, src_h=1080).filtergraph

    person_crop = re.search(r"\[src2\]crop=(\d+):(\d+):(\d+):(\d+)", fg)
    assert person_crop is not None
    person_x = int(person_crop.group(3))

    assert person_x >= 1000


def test_split_uses_full_side_pane_height_when_bboxes_define_both_sides():
    """Side-by-side frames must not over-trust tight LLM face/photo bboxes."""
    instr = LayoutInstruction(
        clip_id="c",
        layout=LayoutKind.SPLIT_CHART_PERSON,
        split_chart_region=BoundingBox(x1=0.0, y1=0.1, x2=0.5, y2=0.7),
        split_person_region=BoundingBox(x1=0.55, y1=0.0, x2=1.0, y2=1.0),
    )
    fg = plan_layout(instr, out_w=1080, out_h=1920, src_w=1920, src_h=1080).filtergraph
    assert "crop=1008:1080:0:0" in fg


def test_split_photo_insert_does_not_render_eye_only_crop():
    instr = LayoutInstruction(
        clip_id="c",
        layout=LayoutKind.SPLIT_CHART_PERSON,
        split_chart_region=BoundingBox(x1=0.053, y1=0.284, x2=0.458, y2=0.482),
        split_person_region=BoundingBox(x1=0.5, y1=0.0, x2=1.0, y2=0.784),
    )

    fg = plan_layout(instr, out_w=1080, out_h=1920, src_w=1920, src_h=1080).filtergraph

    assert "crop=918:1080:0:0" in fg
    assert ":214:" not in fg
    assert ":306:" not in fg


def test_split_minimum_strip_width_enforced():
    """If chart/person bboxes are pathological (seam at edge), don't starve a strip."""
    instr = LayoutInstruction(
        clip_id="c",
        layout=LayoutKind.SPLIT_CHART_PERSON,
        split_chart_region=BoundingBox(x1=0.0, y1=0.0, x2=0.05, y2=1.0),
        split_person_region=BoundingBox(x1=0.05, y1=0.0, x2=1.0, y2=1.0),
    )
    fg = plan_layout(instr, out_w=1080, out_h=1920, src_w=1920, src_h=1080).filtergraph
    widths = [int(m) for m in re.findall(r"crop=(\d+):\d+:\d+:\d+", fg)]
    # Min strip = 20% of 1920 = 384 px. Neither strip should be narrower.
    assert all(w >= 384 for w in widths), widths


def test_split_two_persons_stacks_two_crops():
    instr = LayoutInstruction(
        clip_id="c",
        layout=LayoutKind.SPLIT_TWO_PERSONS,
        split_person_region=BoundingBox(x1=0.0, y1=0.05, x2=0.5, y2=0.95),
        split_second_person_region=BoundingBox(x1=0.5, y1=0.05, x2=1.0, y2=0.95),
    )
    fg = plan_layout(instr, out_w=1080, out_h=1920, src_w=1920, src_h=1080).filtergraph
    assert "split=2" in fg and "vstack=inputs=2" in fg
    # Seam at x=960. bbox y: 0.05..0.95 -> y=54, ch=972 (even).
    assert "[src1]crop=960:972:0:54" in fg
    assert "[src2]crop=960:972:960:54" in fg


def test_split_two_charts_stacks_two_crops():
    instr = LayoutInstruction(
        clip_id="c",
        layout=LayoutKind.SPLIT_TWO_CHARTS,
        split_chart_region=BoundingBox(x1=0.0, y1=0.0, x2=0.5, y2=1.0),
        split_second_chart_region=BoundingBox(x1=0.5, y1=0.0, x2=1.0, y2=1.0),
    )
    fg = plan_layout(instr, out_w=1080, out_h=1920, src_w=1920, src_h=1080).filtergraph
    assert "split=2" in fg and "vstack=inputs=2" in fg
    assert "[src1]crop=960:1080:0:0" in fg
    assert "[src2]crop=960:1080:960:0" in fg


def test_split_two_persons_without_bboxes_defaults_to_centered():
    """No bboxes -> centered 50/50 seam, full source height fallback."""
    instr = LayoutInstruction(
        clip_id="c", layout=LayoutKind.SPLIT_TWO_PERSONS
    )
    fg = plan_layout(instr, out_w=1080, out_h=1920, src_w=1920, src_h=1080).filtergraph
    assert "[src1]crop=960:1080:0:0" in fg
    assert "[src2]crop=960:1080:960:0" in fg


def test_split_visual_band_contains_and_person_band_covers():
    """Visual inserts must stay readable; speaker crop can cover its band."""
    instr = LayoutInstruction(clip_id="c", layout=LayoutKind.SPLIT_CHART_PERSON)
    fg = plan_layout(instr, out_w=1080, out_h=1920, src_w=1920, src_h=1080).filtergraph
    assert fg.count("force_original_aspect_ratio=decrease") == 1
    assert "gblur=sigma=18" in fg
    assert "overlay=(W-w)/2:(H-h)/2" in fg
    assert fg.count("force_original_aspect_ratio=increase") == 2
    assert fg.count("setsar=1") == 2


def test_zoom_tighter_means_smaller_crop_window():
    from afterform.primitives.layouts import plan_zoom_call_center

    wide = plan_zoom_call_center(
        LayoutInstruction(clip_id="c", layout=LayoutKind.ZOOM_CALL_CENTER, zoom=1.0),
        out_w=1080,
        out_h=1920,
    )
    tight = plan_zoom_call_center(
        LayoutInstruction(clip_id="c", layout=LayoutKind.ZOOM_CALL_CENTER, zoom=2.0),
        out_w=1080,
        out_h=1920,
    )
    # Parse crop=CW:CH:X:Y out of each filtergraph.
    import re

    def crop(fg: str) -> tuple[int, int]:
        m = re.search(r"crop=(\d+):(\d+):", fg)
        assert m is not None
        return int(m.group(1)), int(m.group(2))

    wcw, wch = crop(wide.filtergraph)
    tcw, tch = crop(tight.filtergraph)
    assert tcw < wcw and tch < wch


def test_wide_visual_without_region_contains_source_without_center_crop():
    instr = LayoutInstruction(clip_id="c", layout=LayoutKind.WIDE_VISUAL)
    fg = plan_layout(instr, out_w=1080, out_h=1920, src_w=1920, src_h=1080).filtergraph

    assert "[0:v]scale=1080:1920:force_original_aspect_ratio=decrease" in fg
    assert "pad=1080:1920:(ow-iw)/2:(oh-ih)/2:black" in fg


def test_sit_center_zoom_out_contains_source_over_blur():
    instr = LayoutInstruction(
        clip_id="c",
        layout=LayoutKind.SIT_CENTER,
        zoom=0.85,
        person_x_norm=0.45,
    )
    fg = plan_layout(instr, out_w=1080, out_h=1920, src_w=1920, src_h=1080).filtergraph

    assert "[0:v]crop=1920:1080:0:0" in fg
    assert "force_original_aspect_ratio=decrease" in fg
    assert "force_original_aspect_ratio=increase" in fg
    assert "gblur=sigma=18" in fg
    assert "overlay=(W-w)/2:(H-h)/2" in fg


def test_wide_visual_with_region_uses_readable_blur_contain():
    instr = LayoutInstruction(
        clip_id="c",
        layout=LayoutKind.WIDE_VISUAL,
        split_chart_region=BoundingBox(x1=0.07, y1=0.36, x2=0.93, y2=0.58),
    )
    fg = plan_layout(instr, out_w=1080, out_h=1920, src_w=1920, src_h=1080).filtergraph

    assert "[0:v]crop=" in fg
    assert "gblur=sigma=18" in fg
    assert "overlay=(W-w)/2:(H-h)/2" in fg
    assert "force_original_aspect_ratio=decrease" in fg
    assert "force_original_aspect_ratio=increase" in fg
    assert "pad=1080:1920" not in fg


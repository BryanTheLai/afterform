"""Split layout uses optional normalized bbox regions (Gemini vision)."""

from afterform.primitives.layouts import plan_layout
from afterform.schemas import BoundingBox, FocusStackOrder, LayoutInstruction, LayoutKind


def test_split_with_bbox_regions_not_fixed_thirds():
    instr = LayoutInstruction(
        clip_id="c",
        layout=LayoutKind.SPLIT_CHART_PERSON,
        focus_stack_order=FocusStackOrder.CHART_THEN_PERSON,
        split_chart_region=BoundingBox(x1=0.0, y1=0.0, x2=0.64, y2=1.0),
        split_person_region=BoundingBox(x1=0.64, y1=0.0, x2=1.0, y2=1.0),
    )
    fg = plan_layout(instr, out_w=1080, out_h=1920, src_w=1920, src_h=1080).filtergraph
    assert "crop=1228:1080:0:0" in fg or "crop=1224:1080:0:0" in fg
    assert "vstack=inputs=2" in fg


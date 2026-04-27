from afterform.flows.long_to_shorts.flow import run_pipeline
from afterform.primitives.layouts import plan_layout
from afterform.schemas import LayoutInstruction, LayoutKind


def test_afterform_namespace_exposes_flow_and_primitives():
    assert callable(run_pipeline)
    assert callable(plan_layout)
    assert LayoutKind.SIT_CENTER.value == "sit_center"
    assert LayoutInstruction(clip_id="001", layout=LayoutKind.SIT_CENTER).layout == LayoutKind.SIT_CENTER


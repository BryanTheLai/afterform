"""Prompt templates ship with the package and render with clip bounds."""

from afterform.flows.long_to_shorts.load_prompts import clip_selection_prompts


def test_clip_selection_prompts_include_bounds():
    system, user = clip_selection_prompts(
        transcript_text="[0.0s - 1.0s] hello",
        min_dur=35.0,
        max_dur=90.0,
        count=5,
    )
    assert "35.0" in system or "35" in system
    assert "90" in system
    assert "hello" in user


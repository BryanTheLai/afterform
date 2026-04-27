import json

from afterform.primitives.classify import (
    classify_scenes_heuristic,
    classify_scenes_with_llm,
)
from afterform.schemas import LayoutKind, Scene


def test_heuristic_no_keyframe_defaults_sit_center():
    scenes = [Scene(scene_id="s0", start_time=0.0, end_time=1.0, keyframe_path=None)]
    result = classify_scenes_heuristic(scenes)
    assert len(result) == 1
    assert result[0].scene_id == "s0"
    assert result[0].layout == LayoutKind.SIT_CENTER


def test_llm_classifier_uses_callback_and_validates():
    scenes = [Scene(scene_id="s0", start_time=0.0, end_time=1.0, keyframe_path="/tmp/x.jpg")]

    def fake_vision(image_path: str, prompt: str) -> str:
        return json.dumps(
            {"layout": "split_chart_person", "confidence": 0.88, "reason": "chart left"}
        )

    result = classify_scenes_with_llm(scenes, fake_vision)
    assert result[0].layout == LayoutKind.SPLIT_CHART_PERSON
    assert result[0].confidence == 0.88


def test_llm_classifier_parse_error_is_safe():
    scenes = [Scene(scene_id="s0", start_time=0.0, end_time=1.0, keyframe_path="/tmp/x.jpg")]

    def bad_vision(image_path: str, prompt: str) -> str:
        return "not json"

    result = classify_scenes_with_llm(scenes, bad_vision)
    assert result[0].layout == LayoutKind.SIT_CENTER
    assert "parse error" in result[0].reason.lower()


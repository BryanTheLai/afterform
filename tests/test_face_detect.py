"""Tests for the MediaPipe-backed face detection primitive.

Uses a stub ``face_fn`` so MediaPipe itself is not required to run the
tests â€” the primitive contract is what we care about: *given* a face
bbox, does the primitive produce the right ``SceneRegions``.
"""

from afterform.primitives.face_detect import detect_face_regions
from afterform.schemas import BoundingBox, Scene


def _scene(i: int, kf: str | None = "/tmp/k.jpg") -> Scene:
    return Scene(scene_id=f"s{i}", start_time=float(i), end_time=float(i) + 1.0, keyframe_path=kf)


def test_no_keyframe_returns_raw_reason():
    out = detect_face_regions([_scene(0, kf=None)], face_fn=lambda _p: None)
    assert out[0].person_bbox is None
    assert "no keyframe" in out[0].raw_reason.lower()


def test_no_face_detected_returns_raw_reason():
    out = detect_face_regions([_scene(0)], face_fn=lambda _p: None)
    assert out[0].person_bbox is None
    assert "no face" in out[0].raw_reason.lower()


def test_face_centered_produces_person_only():
    centered = BoundingBox(x1=0.4, y1=0.2, x2=0.6, y2=0.7, label="face", confidence=0.9)
    out = detect_face_regions([_scene(0)], face_fn=lambda _p: centered)
    r = out[0]
    assert r.person_bbox is not None
    assert r.person_bbox.center_x == centered.center_x
    assert r.chart_bbox is None


def test_face_pushed_right_synthesises_chart_bbox():
    # face center x ~ 0.86 -> above default threshold 0.65 -> chart bbox inferred
    face = BoundingBox(x1=0.75, y1=0.1, x2=0.97, y2=0.9, label="face", confidence=0.95)
    out = detect_face_regions([_scene(0)], face_fn=lambda _p: face)
    r = out[0]
    assert r.person_bbox is not None
    assert r.chart_bbox is not None
    assert r.chart_bbox.x1 == 0.0
    assert r.chart_bbox.x2 <= 0.75  # can't overlap the face
    assert r.chart_bbox.x2 <= 0.65  # bounded by threshold too
    assert "synthetic chart" in r.raw_reason


def test_face_detector_exception_is_isolated_per_scene():
    scenes = [_scene(0), _scene(1)]
    calls: list[str] = []

    def flaky_fn(path: str) -> BoundingBox | None:
        calls.append(path)
        if len(calls) == 1:
            raise RuntimeError("boom")
        return BoundingBox(x1=0.3, y1=0.2, x2=0.7, y2=0.8)

    out = detect_face_regions(scenes, face_fn=flaky_fn)
    assert out[0].person_bbox is None
    assert "error" in out[0].raw_reason.lower()
    assert out[1].person_bbox is not None


def test_custom_threshold_prevents_false_chart_split():
    face = BoundingBox(x1=0.75, y1=0.1, x2=0.97, y2=0.9)
    out = detect_face_regions(
        [_scene(0)],
        face_fn=lambda _p: face,
        chart_split_threshold=0.95,
    )
    assert out[0].chart_bbox is None


"""Local face-detection primitive â€” the MediaPipe path as another ``SceneRegions`` producer.

Three detection backends share the *same output schema* (``SceneRegions``):

* ``primitives/classify.py``          â€” pixel variance heuristic, no model.
* ``primitives/face_detect.py``       â€” MediaPipe face rectangle (this file).
* ``primitives/vision.py``            â€” multimodal LLM + OCR bboxes.

Because all three emit ``SceneRegions``, the layout planner in
``primitives/vision.py`` (``classify_from_regions`` + ``layout_instruction_from_regions``)
works on all of them unchanged. That is the whole point of the primitive
boundary â€” the *detector* is swappable, the *renderer* is fixed.

MediaPipe is imported lazily so it remains an optional extra.
"""

from __future__ import annotations

import logging
from typing import Callable

from ..schemas import BoundingBox, Scene, SceneRegions

logger = logging.getLogger(__name__)


# A bbox loader for any future cloud face API. Takes a keyframe path,
# returns a normalized face bbox or ``None``. Same shape as the MediaPipe
# wrapper below, which lets tests pass a stub and skip MediaPipe.
FaceBBoxFn = Callable[[str], BoundingBox | None]


def detect_face_regions(
    scenes: list[Scene],
    face_fn: FaceBBoxFn | None = None,
    chart_split_threshold: float = 0.65,
) -> list[SceneRegions]:
    """Populate ``SceneRegions.person_bbox`` (+ ``chart_bbox``) from a face detector.

    The face bbox is treated as the *person bbox*. If the face sits in the
    right ``(1 - chart_split_threshold)`` of the frame, a *chart bbox* is
    synthesised over the left region â€” mirroring the original
    ``reframe.py`` split heuristic.

    Args:
        scenes: scenes with ``keyframe_path`` populated.
        face_fn: pluggable face detector. Defaults to MediaPipe (lazy
            import) if not supplied. Pass a stub in tests.
        chart_split_threshold: face x-center above this normalized value
            triggers a synthetic chart bbox on the left.
    """

    if face_fn is None:
        face_fn = _mediapipe_face_bbox

    out: list[SceneRegions] = []
    for s in scenes:
        if not s.keyframe_path:
            out.append(SceneRegions(scene_id=s.scene_id, raw_reason="no keyframe available"))
            continue
        try:
            face = face_fn(s.keyframe_path)
        except Exception as e:  # one bad scene should not kill the batch
            logger.warning("face detector failed on %s: %r", s.keyframe_path, e)
            out.append(SceneRegions(scene_id=s.scene_id, raw_reason=f"face detector error: {e!r}"))
            continue

        if face is None:
            out.append(SceneRegions(scene_id=s.scene_id, raw_reason="no face detected"))
            continue

        chart = None
        if face.center_x >= chart_split_threshold:
            # Face pushed right â†’ assume a chart occupies the left region.
            chart = BoundingBox(
                x1=0.0,
                y1=0.0,
                x2=min(chart_split_threshold, face.x1),
                y2=1.0,
                label="chart_inferred",
                confidence=max(0.0, face.center_x - chart_split_threshold + 0.5),
            )

        out.append(
            SceneRegions(
                scene_id=s.scene_id,
                person_bbox=face,
                chart_bbox=chart,
                raw_reason="face detected" + (" + synthetic chart bbox" if chart else ""),
            )
        )

    return out


def _mediapipe_face_bbox(keyframe_path: str) -> BoundingBox | None:
    """Return the largest-confidence face as a ``BoundingBox``, or ``None``.

    Imports MediaPipe + OpenCV lazily so they remain optional dependencies
    (install ``afterform[face]``).
    """

    try:
        import cv2  # type: ignore
        import mediapipe as mp  # type: ignore
    except ImportError as e:
        raise RuntimeError(
            "MediaPipe face detection requires `pip install afterform[face]`"
        ) from e

    img = cv2.imread(keyframe_path)
    if img is None:
        return None
    rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    with mp.solutions.face_detection.FaceDetection(
        model_selection=1, min_detection_confidence=0.5
    ) as detector:
        results = detector.process(rgb)
        if not results.detections:
            return None
        best = max(results.detections, key=lambda d: d.score[0])
        box = best.location_data.relative_bounding_box
        x1 = max(0.0, min(1.0, float(box.xmin)))
        y1 = max(0.0, min(1.0, float(box.ymin)))
        x2 = max(x1 + 1e-6, min(1.0, x1 + float(box.width)))
        y2 = max(y1 + 1e-6, min(1.0, y1 + float(box.height)))
        return BoundingBox(
            x1=x1,
            y1=y1,
            x2=x2,
            y2=y2,
            label="face",
            confidence=float(best.score[0]),
        )


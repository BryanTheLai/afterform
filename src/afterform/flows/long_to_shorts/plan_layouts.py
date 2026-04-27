"""Per-clip multi-frame layout vision via a swappable multimodal LLM."""

from __future__ import annotations

import hashlib
import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from pydantic import BaseModel, Field

from afterform.schemas import BoundingBox, Clip, LayoutInstruction, LayoutKind
from afterform.primitives.vision import layout_instruction_from_regions
from afterform.schemas import SceneClassification, SceneRegions

from afterform.config import PipelineConfig
from afterform.providers.llm import (
    LlmImageInput,
    StructuredLlmRequest,
    call_structured_llm,
    resolved_llm_identity,
    resolved_llm_provider,
    resolved_vision_model as resolved_provider_vision_model,
)
from afterform.flows.long_to_shorts.render_window import clip_for_render, clip_output_duration, source_keep_ranges

logger = logging.getLogger(__name__)

LAYOUT_VISION_META = "layout_vision.meta.json"
LAYOUT_VISION_JSON = "layout_vision.json"
LAYOUT_VISION_META_VERSION = 3
_LAYOUT_POLICY_VERSION = 3
_MAX_SAMPLED_FRAMES = 6
_COARSE_FRAME_STEP_SEC = 1.0
_VISION_HTTP_TIMEOUT_MS = 120_000
_VISION_RETRYABLE_STATUS_CODES = [429, 500, 502, 503, 504]
# Stage 3 requests are large multimodal calls. Blind immediate retries mainly turn
# one deadline into a multi-minute wait; operators can rerun this stage directly.
_VISION_RETRY_ATTEMPTS = 0
_VISION_RETRY_INITIAL_DELAY_SEC = 1.0
_VISION_RETRY_MAX_DELAY_SEC = 4.0
_VISION_RETRY_EXP_BASE = 2.0
_VISION_RETRY_JITTER = 0.0
_MIN_SPLIT_STRIP_FRAC = 0.2
_SPLIT_TOP_RATIO_MIN = 0.32
_SPLIT_TOP_RATIO_MAX = 0.48
_SPLIT_FACE_REGION_MIN_HEIGHT = 0.62
_SPLIT_FACE_REGION_HEIGHT_MULT = 2.0
_SPLIT_FACE_TOP_PAD_MULT = 0.30

GEMINI_LAYOUT_VISION_PROMPT = """You are framing a vertical short (9:16) from MULTIPLE keyframes of the same clip.

Return ONLY one JSON object with this exact shape:
{
  "frames": [
    {
      "frame_index": 0,
      "timestamp_sec": 12.34,
      "layout": "zoom_call_center" | "sit_center" | "split_chart_person" | "split_two_persons" | "split_two_charts",
      "person_bbox":        {"x1": 0, "y1": 0, "x2": 1000, "y2": 1000} | null,
      "face_bbox":          {"x1": 0, "y1": 0, "x2": 1000, "y2": 1000} | null,
      "chart_bbox":         {"x1": 0, "y1": 0, "x2": 1000, "y2": 1000} | null,
      "second_person_bbox": {"x1": 0, "y1": 0, "x2": 1000, "y2": 1000} | null,
      "second_face_bbox":   {"x1": 0, "y1": 0, "x2": 1000, "y2": 1000} | null,
      "second_chart_bbox":  {"x1": 0, "y1": 0, "x2": 1000, "y2": 1000} | null,
      "reason": "short rationale"
    }
  ],
  "merged": {
    "layout": "zoom_call_center" | "sit_center" | "split_chart_person" | "split_two_persons" | "split_two_charts",
    "person_bbox":        {"x1": 0, "y1": 0, "x2": 1000, "y2": 1000} | null,
    "face_bbox":          {"x1": 0, "y1": 0, "x2": 1000, "y2": 1000} | null,
    "chart_bbox":         {"x1": 0, "y1": 0, "x2": 1000, "y2": 1000} | null,
    "second_person_bbox": {"x1": 0, "y1": 0, "x2": 1000, "y2": 1000} | null,
    "second_face_bbox":   {"x1": 0, "y1": 0, "x2": 1000, "y2": 1000} | null,
    "second_chart_bbox":  {"x1": 0, "y1": 0, "x2": 1000, "y2": 1000} | null,
    "reason": "one merged rationale"
  }
}

Rules:
- The images are ordered by time. Return one `frames[]` entry for each image in that same order.
- Bboxes use Gemini's preferred 0..1000 coordinate scale. Require x2 > x1 and y2 > y1 for every non-null box.
- person_bbox / second_person_bbox: tight box around each visible speaker's head and upper body.
- face_bbox / second_face_bbox: tight box around the visible face only. No torso, shoulders, hands, mug, or chair.
- chart_bbox / second_chart_bbox: slide, chart, graph, or large on-screen graphic.
- If two speakers are visible, `person_bbox` is the LEFT speaker and `second_person_bbox` is the RIGHT speaker.
- If two charts are visible, `chart_bbox` is the LEFT chart and `second_chart_bbox` is the RIGHT chart.
- merged = one clip-level decision that best fits the dominant layout across the sampled frames. If the clip changes, pick the safest layout that keeps the primary subject(s) readable for most of the clip.
- When in doubt prefer `sit_center`.
- No markdown. JSON only.
"""


class _GeminiBBox1000(BaseModel):
    x1: float = Field(ge=0.0, le=1000.0)
    y1: float = Field(ge=0.0, le=1000.0)
    x2: float = Field(ge=0.0, le=1000.0)
    y2: float = Field(ge=0.0, le=1000.0)


class _GeminiLayoutDecision(BaseModel):
    layout: str = "sit_center"
    person_bbox: _GeminiBBox1000 | None = None
    face_bbox: _GeminiBBox1000 | None = None
    chart_bbox: _GeminiBBox1000 | None = None
    second_person_bbox: _GeminiBBox1000 | None = None
    second_face_bbox: _GeminiBBox1000 | None = None
    second_chart_bbox: _GeminiBBox1000 | None = None
    reason: str = ""


class _GeminiFrameDecision(_GeminiLayoutDecision):
    frame_index: int = Field(ge=0)
    timestamp_sec: float = Field(ge=0.0)


class _GeminiMultiFrameResponse(BaseModel):
    frames: list[_GeminiFrameDecision] = Field(default_factory=list)
    merged: _GeminiLayoutDecision = Field(default_factory=_GeminiLayoutDecision)


@dataclass
class SampledFrame:
    frame_id: str
    timestamp_sec: float
    path: str
    width: int
    height: int


def _clip_windows_fingerprint(clips: list[Clip]) -> str:
    payload = json.dumps(
        [
            {
                "id": clip.clip_id,
                "start": round(rclip.start_time_sec, 3),
                "end": round(rclip.end_time_sec, 3),
                "keep_ranges": [
                    [round(start, 3), round(end, 3)] for start, end in rclip.keep_ranges_sec
                ],
            }
            for clip in clips
            for rclip in [clip_for_render(clip)]
        ],
        sort_keys=True,
        ensure_ascii=False,
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def layout_cache_valid(
    work_dir: Path,
    *,
    transcript_fp: str,
    clip_windows_fp: str,
    llm_identity: dict[str, str],
) -> bool:
    meta_path = work_dir / LAYOUT_VISION_META
    data_path = work_dir / LAYOUT_VISION_JSON
    if not meta_path.is_file() or not data_path.is_file():
        return False
    try:
        meta: dict[str, Any] = json.loads(meta_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return False
    return (
        meta.get("version") == LAYOUT_VISION_META_VERSION
        and meta.get("transcript_sha256") == transcript_fp
        and meta.get("clip_windows_sha256") == clip_windows_fp
        and meta.get("llm") == llm_identity
        and meta.get("layout_policy_version") == _LAYOUT_POLICY_VERSION
    )


def load_layout_cache(work_dir: Path) -> dict[str, dict[str, Any]] | None:
    p = work_dir / LAYOUT_VISION_JSON
    if not p.is_file():
        return None
    try:
        data = json.loads(p.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return None
    clips = data.get("clips")
    return clips if isinstance(clips, dict) else None


def write_layout_cache(
    work_dir: Path,
    *,
    transcript_fp: str,
    clip_windows_fp: str,
    llm_identity: dict[str, str],
    clips_payload: dict[str, dict[str, Any]],
) -> None:
    work_dir.mkdir(parents=True, exist_ok=True)
    meta = {
        "version": LAYOUT_VISION_META_VERSION,
        "transcript_sha256": transcript_fp,
        "clip_windows_sha256": clip_windows_fp,
        "llm": llm_identity,
        "layout_policy_version": _LAYOUT_POLICY_VERSION,
    }
    (work_dir / LAYOUT_VISION_META).write_text(
        json.dumps(meta, indent=2) + "\n", encoding="utf-8"
    )
    (work_dir / LAYOUT_VISION_JSON).write_text(
        json.dumps({"clips": clips_payload}, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    logger.info("Wrote %s and %s", LAYOUT_VISION_META, LAYOUT_VISION_JSON)


def _parse_bbox(
    raw: object,
    *,
    warnings: list[str],
    field_name: str,
    frame_width: int | None = None,
    frame_height: int | None = None,
) -> BoundingBox | None:
    if raw is None:
        return None
    if not isinstance(raw, dict):
        warnings.append(f"{field_name}: ignored non-object bbox {raw!r}")
        return None

    try:
        coords = {key: float(raw[key]) for key in ("x1", "y1", "x2", "y2")}
    except Exception as exc:
        warnings.append(f"{field_name}: malformed bbox fields ({exc})")
        return None

    scale = "normalized"
    if max(coords.values()) <= 1.0:
        normalized = coords
    elif max(coords.values()) <= 1000.0:
        scale = "gemini_1000"
        normalized = {key: value / 1000.0 for key, value in coords.items()}
    elif frame_width and frame_height:
        scale = "pixel_fallback"
        normalized = {
            "x1": coords["x1"] / float(frame_width),
            "y1": coords["y1"] / float(frame_height),
            "x2": coords["x2"] / float(frame_width),
            "y2": coords["y2"] / float(frame_height),
        }
    else:
        warnings.append(
            f"{field_name}: dropped bbox {coords!r} (coords exceed 1000 and frame size unknown)"
        )
        return None

    try:
        box = BoundingBox.model_validate(normalized)
    except Exception as exc:
        warnings.append(f"{field_name}: dropped malformed {scale} bbox {coords!r} ({exc})")
        return None

    if scale != "normalized":
        warnings.append(f"{field_name}: normalized {scale} bbox -> [0,1]")
    return box


def _subject_width_zoom(person: BoundingBox | None, face: BoundingBox | None) -> float:
    width = person.width if person is not None else 0.0
    if width <= 0.0 and face is not None:
        width = min(1.0, face.width * 2.2)
    if width <= 0.0:
        return 1.1
    zoom = 1.0 + max(0.0, 0.55 - width) * 1.1
    return round(max(1.0, min(1.3, zoom)), 3)


def _render_safe_split_person_region(
    person: BoundingBox,
    face: BoundingBox | None,
) -> BoundingBox:
    """Convert a detection box into a render-friendly split crop region.

    Raw ``person_bbox`` is asked to cover head + upper body. Feeding that
    directly into the split renderer makes the bottom band too tall and cover
    crop trims the head. When we have a sane ``face_bbox``, bias the region
    toward head + shoulders instead of the full torso.
    """

    if _face_center_x(face, person) is None or face is None:
        return person

    face_h = max(0.0, face.y2 - face.y1)
    if face_h <= 0.0:
        return person

    target_h = min(
        person.y2 - person.y1,
        max(_SPLIT_FACE_REGION_MIN_HEIGHT, face_h * _SPLIT_FACE_REGION_HEIGHT_MULT),
    )
    top = max(0.0, min(person.y1, face.y1 - face_h * _SPLIT_FACE_TOP_PAD_MULT))
    bottom = min(person.y2, top + target_h)
    if bottom - top < target_h:
        top = max(0.0, bottom - target_h)
    if bottom - top <= face_h:
        return person

    return person.model_copy(update={"y1": top, "y2": bottom})


def _split_chart_person_top_band_ratio(
    chart: BoundingBox,
    person: BoundingBox,
) -> float:
    """Allocate stacked-band height from the two source-region aspect needs."""

    seam = (chart.x2 + person.x1) / 2.0
    seam = max(_MIN_SPLIT_STRIP_FRAC, min(1.0 - _MIN_SPLIT_STRIP_FRAC, seam))
    chart_w = max(1e-6, seam)
    person_w = max(1e-6, 1.0 - seam)
    chart_need = max(1e-6, (chart.y2 - chart.y1) / chart_w)
    person_need = max(1e-6, (person.y2 - person.y1) / person_w)
    ratio = chart_need / (chart_need + person_need)
    return round(max(_SPLIT_TOP_RATIO_MIN, min(_SPLIT_TOP_RATIO_MAX, ratio)), 3)


def _instruction_from_gemini_json(
    scene_id: str,
    data: dict[str, Any],
    *,
    frame_width: int | None = None,
    frame_height: int | None = None,
    warnings: list[str] | None = None,
) -> LayoutInstruction:
    """Translate Gemini JSON into a validated :class:`LayoutInstruction`."""
    warnings = warnings if warnings is not None else []

    layout_str = str(data.get("layout", "sit_center")).strip()
    try:
        kind = LayoutKind(layout_str)
    except ValueError:
        warnings.append(f"layout: unknown value {layout_str!r}; defaulted to sit_center")
        kind = LayoutKind.SIT_CENTER

    pb = _parse_bbox(
        data.get("person_bbox"),
        warnings=warnings,
        field_name="person_bbox",
        frame_width=frame_width,
        frame_height=frame_height,
    )
    fb = _parse_bbox(
        data.get("face_bbox"),
        warnings=warnings,
        field_name="face_bbox",
        frame_width=frame_width,
        frame_height=frame_height,
    )
    cb = _parse_bbox(
        data.get("chart_bbox"),
        warnings=warnings,
        field_name="chart_bbox",
        frame_width=frame_width,
        frame_height=frame_height,
    )
    p2 = _parse_bbox(
        data.get("second_person_bbox"),
        warnings=warnings,
        field_name="second_person_bbox",
        frame_width=frame_width,
        frame_height=frame_height,
    )
    f2 = _parse_bbox(
        data.get("second_face_bbox"),
        warnings=warnings,
        field_name="second_face_bbox",
        frame_width=frame_width,
        frame_height=frame_height,
    )
    c2 = _parse_bbox(
        data.get("second_chart_bbox"),
        warnings=warnings,
        field_name="second_chart_bbox",
        frame_width=frame_width,
        frame_height=frame_height,
    )
    reason = str(data.get("reason", ""))[:400]

    if kind == LayoutKind.SPLIT_CHART_PERSON and (pb is None or cb is None):
        warnings.append("split_chart_person missing required boxes; downgraded to sit_center")
        kind = LayoutKind.SIT_CENTER
    if kind == LayoutKind.SPLIT_TWO_PERSONS and (pb is None or p2 is None):
        warnings.append("split_two_persons missing required boxes; downgraded to sit_center")
        kind = LayoutKind.SIT_CENTER
    if kind == LayoutKind.SPLIT_TWO_CHARTS and (cb is None or c2 is None):
        warnings.append("split_two_charts missing required boxes; downgraded to sit_center")
        kind = LayoutKind.SIT_CENTER

    regions = SceneRegions(scene_id=scene_id, person_bbox=pb, chart_bbox=cb, raw_reason=reason)
    classification = SceneClassification(
        scene_id=scene_id,
        layout=kind,
        confidence=1.0,
        reason=reason,
    )
    instr = layout_instruction_from_regions(regions, classification, clip_id=scene_id)

    updates: dict[str, Any] = {}
    face_center = _face_center_x(fb, pb)
    if face_center is not None:
        updates["person_x_norm"] = face_center

    if kind == LayoutKind.ZOOM_CALL_CENTER:
        updates["zoom"] = _subject_width_zoom(pb, fb)

    if kind == LayoutKind.SPLIT_CHART_PERSON and pb is not None and cb is not None:
        render_person = _render_safe_split_person_region(pb, fb)
        updates["split_chart_region"] = cb
        updates["split_person_region"] = render_person
        updates["top_band_ratio"] = _split_chart_person_top_band_ratio(cb, render_person)
    elif kind == LayoutKind.SPLIT_TWO_PERSONS and pb is not None and p2 is not None:
        left, right = sorted((pb, p2), key=lambda b: b.center_x)
        updates["split_person_region"] = left
        updates["split_second_person_region"] = right
        face_centers = [(left, fb), (right, f2)]
        if face_centers[0][1] is not None:
            updates["person_x_norm"] = _face_center_x(face_centers[0][1], left) or left.center_x
    elif kind == LayoutKind.SPLIT_TWO_CHARTS and cb is not None and c2 is not None:
        left, right = sorted((cb, c2), key=lambda b: b.center_x)
        updates["split_chart_region"] = left
        updates["split_second_chart_region"] = right

    if updates:
        instr = instr.model_copy(update=updates)
    return instr


def _face_center_x(
    face: BoundingBox | None,
    person: BoundingBox | None,
) -> float | None:
    """Pick a horizontal center to aim the 9:16 crop at."""
    if face is None:
        return None
    face_w = max(0.0, face.x2 - face.x1)
    if face_w <= 0.0:
        return None
    if face_w > 0.40:
        return None
    if person is not None and not (person.x1 - 0.02 <= face.center_x <= person.x2 + 0.02):
        return None
    return float(face.center_x)


def _source_time_from_output_time(
    keep_ranges: list[tuple[float, float]],
    output_time_sec: float,
) -> float:
    remaining = output_time_sec
    for start, end in keep_ranges:
        span = end - start
        if remaining <= span:
            return start + remaining
        remaining -= span
    return keep_ranges[-1][1]


def _uniform_source_timestamps(keep_ranges: list[tuple[float, float]], count: int) -> list[float]:
    total = sum(end - start for start, end in keep_ranges)
    if total <= 0.0 or count <= 0:
        return []
    if count == 1:
        return [_source_time_from_output_time(keep_ranges, total / 2.0)]
    return [
        _source_time_from_output_time(keep_ranges, total * ratio)
        for ratio in np.linspace(0.15, 0.85, count)
    ]


def _sample_frame_signature(cap: Any, timestamp_sec: float) -> np.ndarray | None:
    import cv2

    cap.set(cv2.CAP_PROP_POS_MSEC, timestamp_sec * 1000.0)
    ok, frame = cap.read()
    if not ok or frame is None:
        return None
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    return cv2.resize(gray, (160, 90), interpolation=cv2.INTER_AREA)


def _frame_diff_peak_timestamps(
    source_video: Path,
    keep_ranges: list[tuple[float, float]],
    *,
    peak_count: int,
) -> tuple[list[float], list[str]]:
    if peak_count <= 0:
        return [], []
    warnings: list[str] = []
    try:
        import cv2
    except Exception as exc:
        return [], [f"OpenCV unavailable; skipped frame-diff peaks ({exc})."]

    timestamps: list[float] = []
    for start, end in keep_ranges:
        cursor = start
        while cursor < end:
            timestamps.append(cursor)
            cursor += _COARSE_FRAME_STEP_SEC
        timestamps.append(end)
    timestamps = sorted({round(ts, 3) for ts in timestamps})
    if len(timestamps) < 2:
        return [], warnings

    cap = cv2.VideoCapture(str(source_video))
    if not cap.isOpened():
        return [], [f"Failed to open source video for frame sampling: {source_video}"]

    diffs: list[tuple[float, float]] = []
    prev_sig: np.ndarray | None = None
    prev_ts: float | None = None
    try:
        for ts in timestamps:
            sig = _sample_frame_signature(cap, ts)
            if sig is None:
                warnings.append(f"Skipped unreadable coarse frame at {ts:.2f}s")
                continue
            if prev_sig is not None and prev_ts is not None:
                diff = float(np.mean(np.abs(sig.astype(np.float32) - prev_sig.astype(np.float32))))
                diffs.append((diff, ts))
            prev_sig = sig
            prev_ts = ts
    finally:
        cap.release()

    picked: list[float] = []
    min_distance = 1.5
    for _, ts in sorted(diffs, key=lambda item: item[0], reverse=True):
        if any(abs(ts - other) < min_distance for other in picked):
            continue
        picked.append(ts)
        if len(picked) >= peak_count:
            break
    return sorted(picked), warnings


def _sample_clip_frames(
    source_video: Path,
    clip: Clip,
    *,
    keyframes_root: Path,
) -> tuple[list[SampledFrame], list[str]]:
    warnings: list[str] = []
    try:
        import cv2
    except Exception as exc:
        return [], [f"OpenCV unavailable; cannot sample layout frames ({exc})."]

    rclip = clip_for_render(clip)
    keep_ranges = source_keep_ranges(rclip)
    total_duration = clip_output_duration(rclip)
    if not keep_ranges or total_duration <= 0.0:
        return [], [f"Clip {clip.clip_id} has no non-empty keep ranges for layout sampling."]

    uniform_count = 4 if total_duration >= 20.0 else 3
    peak_count = 2 if total_duration < 20.0 else 3
    timestamps = _uniform_source_timestamps(keep_ranges, uniform_count)
    peak_ts, peak_warnings = _frame_diff_peak_timestamps(
        source_video,
        keep_ranges,
        peak_count=peak_count,
    )
    warnings.extend(peak_warnings)
    timestamps.extend(peak_ts)
    timestamps = sorted({round(ts, 3) for ts in timestamps})[:_MAX_SAMPLED_FRAMES]

    clip_dir = keyframes_root / clip.clip_id
    clip_dir.mkdir(parents=True, exist_ok=True)
    cap = cv2.VideoCapture(str(source_video))
    if not cap.isOpened():
        return [], [f"Failed to open source video for clip frame sampling: {source_video}"]

    sampled: list[SampledFrame] = []
    try:
        for idx, ts in enumerate(timestamps):
            cap.set(cv2.CAP_PROP_POS_MSEC, ts * 1000.0)
            ok, frame = cap.read()
            if not ok or frame is None:
                warnings.append(f"Skipped unreadable sampled frame at {ts:.2f}s")
                continue
            out_path = clip_dir / f"{clip.clip_id}_{idx:02d}_{int(round(ts * 1000)):06d}.jpg"
            cv2.imwrite(str(out_path), frame)
            height, width = frame.shape[:2]
            sampled.append(
                SampledFrame(
                    frame_id=f"{clip.clip_id}_{idx:02d}",
                    timestamp_sec=ts,
                    path=str(out_path),
                    width=width,
                    height=height,
                )
            )
    finally:
        cap.release()
    return sampled, warnings


def _call_gemini_vision(
    sampled_frames: list[SampledFrame],
    model_name: str,
    *,
    provider: str,
) -> tuple[str, _GeminiMultiFrameResponse]:
    images: list[LlmImageInput] = []
    for idx, frame in enumerate(sampled_frames):
        path = Path(frame.path)
        mime = "image/jpeg" if path.suffix.lower() in (".jpg", ".jpeg") else "image/png"
        images.append(
            LlmImageInput(
                path=path,
                mime_type=mime,
                label=f"FRAME {idx}: timestamp_sec={frame.timestamp_sec:.2f}",
            )
        )

    response = call_structured_llm(
        StructuredLlmRequest(
            stage_name="layout vision",
            model=model_name,
            system_instruction=GEMINI_LAYOUT_VISION_PROMPT,
            temperature=0.2,
            response_schema=_GeminiMultiFrameResponse,
            images=tuple(images),
            timeout_ms=_VISION_HTTP_TIMEOUT_MS,
            max_retries=_VISION_RETRY_ATTEMPTS,
        ),
        provider=provider,
    )
    parsed = response.parsed if isinstance(response.parsed, _GeminiMultiFrameResponse) else None
    raw = response.raw_text
    if parsed is None:
        if not raw:
            raise RuntimeError("Layout vision model returned neither text nor parsed response")
        parsed = _GeminiMultiFrameResponse.model_validate_json(raw)
    if not raw:
        raw = parsed.model_dump_json()
    return raw, parsed


def _fallback_layout_instruction(clip: Clip) -> LayoutInstruction:
    """Best-effort render fallback when Stage 3 cannot produce vision boxes.

    The selector already emits ``layout_hint`` from transcript semantics. If
    frame sampling or the multimodal call fails, preserving that hint is much
    safer than silently collapsing chart-heavy clips to ``sit_center``.
    """

    layout_hint = getattr(clip, "layout_hint", None)
    layout_current = getattr(clip, "layout", None)
    layout = layout_hint or layout_current or LayoutKind.SIT_CENTER
    return LayoutInstruction(clip_id=clip.clip_id, layout=layout)


def _fallback_merge(
    clip: Clip,
    frame_instructions: list[LayoutInstruction],
) -> LayoutInstruction:
    if not frame_instructions:
        return _fallback_layout_instruction(clip)
    counts: dict[LayoutKind, int] = {}
    for instr in frame_instructions:
        counts[instr.layout] = counts.get(instr.layout, 0) + 1
    dominant = max(counts.items(), key=lambda item: (item[1], item[0].value))[0]
    candidates = [instr for instr in frame_instructions if instr.layout == dominant]
    chosen = candidates[len(candidates) // 2]
    return chosen.model_copy(update={"clip_id": clip.clip_id})


def infer_layout_instructions(
    source_video: Path,
    clips: list[Clip],
    *,
    gemini_vision_model: str,
    provider: str,
    keyframes_root: Path,
) -> tuple[dict[str, LayoutInstruction], dict[str, dict[str, Any]]]:
    """Return ``(clip_id -> LayoutInstruction, clip_id -> cache payload)``."""
    out: dict[str, LayoutInstruction] = {}
    payload_by_clip: dict[str, dict[str, Any]] = {}
    model_name = gemini_vision_model.strip()

    for clip in clips:
        warnings: list[str] = []
        sampled_frames, sample_warnings = _sample_clip_frames(
            source_video,
            clip,
            keyframes_root=keyframes_root,
        )
        warnings.extend(sample_warnings)
        if not sampled_frames:
            fallback = _fallback_layout_instruction(clip)
            out[clip.clip_id] = fallback
            payload_by_clip[clip.clip_id] = {
                "instruction": json.loads(fallback.model_dump_json()),
                "sampled_frames": [],
                "frame_results": [],
                "raw": {"error": "no sampled frames", "layout": fallback.layout.value},
                "warnings": warnings + [
                    f"No sampled frames; defaulted to {fallback.layout.value}."
                ],
            }
            continue

        frame_results_payload: list[dict[str, Any]] = []
        frame_instructions: list[LayoutInstruction] = []
        merged_instruction: LayoutInstruction | None = None
        raw_payload: dict[str, Any]
        try:
            raw_text, parsed = _call_gemini_vision(
                sampled_frames,
                model_name,
                provider=provider,
            )
            raw_payload = json.loads(raw_text)

            for idx, frame in enumerate(sampled_frames):
                frame_data = (
                    parsed.frames[idx].model_dump()
                    if idx < len(parsed.frames)
                    else {
                        "frame_index": idx,
                        "timestamp_sec": frame.timestamp_sec,
                        "layout": "sit_center",
                        "reason": "missing frame result",
                    }
                )
                frame_warnings: list[str] = []
                frame_instruction = _instruction_from_gemini_json(
                    clip.clip_id,
                    frame_data,
                    frame_width=frame.width,
                    frame_height=frame.height,
                    warnings=frame_warnings,
                )
                frame_results_payload.append(
                    {
                        "frame_id": frame.frame_id,
                        "timestamp_sec": frame.timestamp_sec,
                        "instruction": json.loads(frame_instruction.model_dump_json()),
                        "raw": frame_data,
                        "warnings": frame_warnings,
                    }
                )
                frame_instructions.append(frame_instruction)

            merged_warnings: list[str] = []
            merged_instruction = _instruction_from_gemini_json(
                clip.clip_id,
                parsed.merged.model_dump(),
                frame_width=sampled_frames[0].width,
                frame_height=sampled_frames[0].height,
                warnings=merged_warnings,
            )
            warnings.extend(merged_warnings)
        except Exception as exc:
            failure_context = {
                "provider": provider,
                "model": model_name,
                "frame_count": len(sampled_frames),
                "timeout_ms": _VISION_HTTP_TIMEOUT_MS,
                "max_retries": _VISION_RETRY_ATTEMPTS,
            }
            warnings.append(
                "Layout vision model failed "
                f"({json.dumps(failure_context, sort_keys=True)}): {exc}"
            )
            raw_payload = {
                "error": str(exc),
                "layout": "sit_center",
                "request": failure_context,
            }

        if merged_instruction is None:
            merged_instruction = _fallback_merge(clip, frame_instructions)

        out[clip.clip_id] = merged_instruction
        payload_by_clip[clip.clip_id] = {
            "instruction": json.loads(merged_instruction.model_dump_json()),
            "sampled_frames": [
                {
                    "frame_id": frame.frame_id,
                    "timestamp_sec": frame.timestamp_sec,
                    "path": frame.path,
                    "width": frame.width,
                    "height": frame.height,
                }
                for frame in sampled_frames
            ],
            "frame_results": frame_results_payload,
            "raw": raw_payload,
            "warnings": warnings,
        }

    return out, payload_by_clip


def resolved_vision_model(config: PipelineConfig) -> str:
    return resolved_provider_vision_model(config)


def run_layout_vision_stage(
    work_dir: Path,
    *,
    source_video: Path,
    clips: list[Clip],
    transcript_fp: str,
    config: PipelineConfig,
) -> dict[str, LayoutInstruction]:
    """Load cache or call the configured layout vision model on multiple frames per clip."""
    clip_windows_fp = _clip_windows_fingerprint(clips)
    provider = resolved_llm_provider(config)
    vm = resolved_vision_model(config)
    llm_identity = resolved_llm_identity(config, vision=True)

    if (
        not config.force_layout_vision
        and layout_cache_valid(
            work_dir,
            transcript_fp=transcript_fp,
            clip_windows_fp=clip_windows_fp,
            llm_identity=llm_identity,
        )
    ):
        cached = load_layout_cache(work_dir)
        if cached:
            logger.info("Layout vision cache hit; skipping model calls.")
            return {
                clip_id: LayoutInstruction.model_validate(payload["instruction"])
                for clip_id, payload in cached.items()
                if isinstance(payload, dict) and "instruction" in payload
            }

    keyframes_root = work_dir / "keyframes"
    instructions, payload = infer_layout_instructions(
        source_video,
        clips,
        gemini_vision_model=vm,
        provider=provider,
        keyframes_root=keyframes_root,
    )
    write_layout_cache(
        work_dir,
        transcript_fp=transcript_fp,
        clip_windows_fp=clip_windows_fp,
        llm_identity=llm_identity,
        clips_payload=payload,
    )
    return instructions


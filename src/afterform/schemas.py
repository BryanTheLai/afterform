"""Strict JSON contracts shared by Afterform primitives and flows."""

from __future__ import annotations

from enum import Enum
from typing import Any, Literal

from pydantic import BaseModel, Field, field_validator, model_validator


# ---------------------------------------------------------------------------
# Extraction artifacts
# ---------------------------------------------------------------------------


class Scene(BaseModel):
    """A single shot/scene detected in the source video."""

    scene_id: str
    start_time: float = Field(ge=0)
    end_time: float = Field(gt=0)
    keyframe_path: str | None = None

    @field_validator("end_time")
    @classmethod
    def _end_after_start(cls, v: float, info) -> float:
        start = info.data.get("start_time", 0.0)
        if v <= start:
            raise ValueError("end_time must be strictly greater than start_time")
        return v

    @property
    def duration(self) -> float:
        return self.end_time - self.start_time


class TranscriptWord(BaseModel):
    """One ASR token with times in **seconds on the source video** timeline."""

    word: str
    start_time: float = Field(ge=0)
    end_time: float = Field(ge=0)


class ClipSubtitleWords(BaseModel):
    """Words for one clip with times in **seconds relative to clip start** (t=0 at cut in-point)."""

    words: list[TranscriptWord] = Field(default_factory=list)


class FocusStackOrder(str, Enum):
    """Vertical order for split layouts: which item occupies the top vs bottom band.

    Bands are split by :attr:`LayoutInstruction.top_band_ratio` (default 0.5 = even).
    For ``SPLIT_CHART_PERSON`` this picks chart-on-top vs person-on-top.
    For ``SPLIT_TWO_PERSONS`` / ``SPLIT_TWO_CHARTS`` it has no visible meaning
    (both bands hold the same kind of item); the enum value is retained only
    so a single stacking recipe drives all three split layouts.
    """

    CHART_THEN_PERSON = "chart_then_person"
    PERSON_THEN_CHART = "person_then_chart"


class IngestResult(BaseModel):
    """Everything Stage 1 (deterministic local extraction) produces."""

    source_path: str
    duration_sec: float
    scenes: list[Scene]
    transcript_words: list[TranscriptWord]
    keyframes_dir: str | None = None


# ---------------------------------------------------------------------------
# Layout system â€” the 5 "thrusters" (max 2 on-screen items per short)
# ---------------------------------------------------------------------------


class LayoutKind(str, Enum):
    """The 9:16 layouts. A short contains **at most two** on-screen items.

    An "item" is one of ``person`` (a human speaker) or ``chart`` (slide, graph,
    data visual, screenshare). Six combinations are allowed:

    - ``ZOOM_CALL_CENTER``:   **1 person**, tight webcam/zoom-call framing, centered.
    - ``SIT_CENTER``:         **1 person**, interview/seated framing, centered.
    - ``WIDE_VISUAL``:        **1 wide visual**, full source contained in 9:16.
    - ``SPLIT_CHART_PERSON``: **1 chart + 1 person** â€” chart + speaker share the
                              source frame. Output stacks them vertically
                              (by default ``focus_stack_order`` = chart-on-top).
    - ``SPLIT_TWO_PERSONS``:  **2 persons** â€” two speakers (e.g. interview two-up).
                              Output stacks them vertically.
    - ``SPLIT_TWO_CHARTS``:   **2 charts** â€” two charts/slides side-by-side in source.
                              Output stacks them vertically.

    The "max 2 items" constraint is the keep-it-simple rule: every rendered short
    is either one item centered, or two items stacked evenly top/bottom.
    """

    ZOOM_CALL_CENTER = "zoom_call_center"
    SIT_CENTER = "sit_center"
    WIDE_VISUAL = "wide_visual"
    SPLIT_CHART_PERSON = "split_chart_person"
    SPLIT_TWO_PERSONS = "split_two_persons"
    SPLIT_TWO_CHARTS = "split_two_charts"


# Layouts that stack two items vertically in the 9:16 output.
SPLIT_LAYOUTS: frozenset[LayoutKind] = frozenset(
    {
        LayoutKind.SPLIT_CHART_PERSON,
        LayoutKind.SPLIT_TWO_PERSONS,
        LayoutKind.SPLIT_TWO_CHARTS,
    }
)


class LayoutInstruction(BaseModel):
    """Per-clip decision telling the compiler which layout to apply and how to crop.

    Every short is described by exactly one of these, keyed by ``clip_id``. Split
    layouts additionally carry up to two normalized bounding boxes (chart/person
    or two-of-a-kind) so the compiler crops source strips that **partition** the
    source width without overlap or gap.
    """

    clip_id: str
    layout: LayoutKind
    # Optional per-layout knobs. Defaults are sane for a 1920x1080 source.
    zoom: float = Field(default=1.0, gt=0, le=4.0)
    person_x_norm: float = Field(
        default=0.5,
        ge=0.0,
        le=1.0,
        description="Normalized x-center of the human subject in source frame (0=left, 1=right).",
    )
    chart_x_norm: float = Field(
        default=0.0,
        ge=0.0,
        le=1.0,
        description=(
            "split_chart_person only: left-edge trim of the chart strip, as a fraction of the "
            "left 2/3 pane (0 = use full chart area)."
        ),
    )
    focus_stack_order: FocusStackOrder = Field(
        default=FocusStackOrder.CHART_THEN_PERSON,
        description="For split_chart_person only: chart-on-top vs person-on-top in the 9:16 stack.",
    )
    split_chart_region: BoundingBox | None = Field(
        default=None,
        description=(
            "Optional normalized rect for the chart/slide crop (Gemini vision). "
            "When set with split_person_region, the split layout uses these boxes instead of fixed 2/3|1/3."
        ),
    )
    split_person_region: BoundingBox | None = Field(
        default=None,
        description="Optional normalized rect for the speaker crop (Gemini vision).",
    )
    split_second_chart_region: BoundingBox | None = Field(
        default=None,
        description=(
            "For ``SPLIT_TWO_CHARTS`` only: second chart bbox. The first chart occupies "
            "the top output band, this one occupies the bottom band."
        ),
    )
    split_second_person_region: BoundingBox | None = Field(
        default=None,
        description=(
            "For ``SPLIT_TWO_PERSONS`` only: second speaker bbox. The first person "
            "occupies the top output band, this one occupies the bottom band."
        ),
    )
    top_band_ratio: float = Field(
        default=0.5,
        ge=0.2,
        le=0.8,
        description=(
            "Fraction of 9:16 output height used by the top band for split layouts. "
            "0.5 = EVEN 50/50 split (default â€” the user-requested symmetric look). "
            "0.6 historically matched the 'chart dominant / person small' look."
        ),
    )


class LayoutTimelineSegment(BaseModel):
    """Output-relative span rendered with one layout instruction.

    ``start_sec`` / ``end_sec`` are measured after trim + keep-range pruning,
    i.e. in the final short's timeline before subtitles are burned.
    """

    start_sec: float = Field(ge=0.0)
    end_sec: float = Field(gt=0.0)
    instruction: LayoutInstruction
    reason: str = ""

    @model_validator(mode="after")
    def _timing_consistency(self) -> "LayoutTimelineSegment":
        if self.end_sec <= self.start_sec:
            raise ValueError("layout timeline segment end_sec must be greater than start_sec")
        return self


class ClipLayoutPlan(BaseModel):
    """Render layout plan for a clip.

    ``instruction`` is the backward-compatible single-layout fallback.
    ``layout_timeline`` is optional; when it contains multiple segments the
    renderer can switch layouts inside one short.
    """

    clip_id: str
    instruction: LayoutInstruction
    layout_timeline: list[LayoutTimelineSegment] = Field(default_factory=list)
    visual_drop_ranges_sec: list[tuple[float, float]] = Field(
        default_factory=list,
        description=(
            "Clip-relative spans removed because frame-level vision detected "
            "short visual transition cards while audio continued."
        ),
    )

    @property
    def layout(self) -> LayoutKind:
        """Backward-compatible access to the fallback instruction layout."""
        return self.instruction.layout

    @classmethod
    def from_layout_cache_payload(cls, clip_id: str, payload: dict[str, Any]) -> "ClipLayoutPlan":
        """Build a layout plan from one ``layout_vision.json`` clip payload."""
        return cls.model_validate(
            {
                "clip_id": clip_id,
                "instruction": payload["instruction"],
                "layout_timeline": [
                    segment
                    for segment in payload.get("layout_timeline", []) or []
                    if isinstance(segment, dict)
                ],
                "visual_drop_ranges_sec": payload.get("visual_drop_ranges_sec", []) or [],
            }
        )

    def to_layout_cache_payload(self) -> dict[str, Any]:
        """Serialize the plan fields persisted in ``layout_vision.json``."""
        return {
            "instruction": self.instruction.model_dump(mode="json"),
            "layout_timeline": [
                segment.model_dump(mode="json") for segment in self.layout_timeline
            ],
            "visual_drop_ranges_sec": [[start, end] for start, end in self.visual_drop_ranges_sec],
        }


class SceneClassification(BaseModel):
    """Result of the classifier: which layout should a given scene use."""

    scene_id: str
    layout: LayoutKind
    confidence: float = Field(ge=0.0, le=1.0)
    reason: str = ""


# ---------------------------------------------------------------------------
# Vision bounding boxes â€” the LLM+OCR path (alt to pixel heuristics)
# ---------------------------------------------------------------------------


class BoundingBox(BaseModel):
    """Normalized [0..1] bounding box in the source frame coordinate space.

    Normalized coords keep these outputs portable across source resolutions
    and stop the model hallucinating pixel values. ``x2 > x1`` and
    ``y2 > y1`` are enforced.
    """

    x1: float = Field(ge=0.0, le=1.0)
    y1: float = Field(ge=0.0, le=1.0)
    x2: float = Field(ge=0.0, le=1.0)
    y2: float = Field(ge=0.0, le=1.0)
    label: str = ""
    confidence: float = Field(default=1.0, ge=0.0, le=1.0)

    @field_validator("x2")
    @classmethod
    def _x2_after_x1(cls, v: float, info) -> float:
        x1 = info.data.get("x1", 0.0)
        if v <= x1:
            raise ValueError("x2 must be > x1")
        return v

    @field_validator("y2")
    @classmethod
    def _y2_after_y1(cls, v: float, info) -> float:
        y1 = info.data.get("y1", 0.0)
        if v <= y1:
            raise ValueError("y2 must be > y1")
        return v

    @property
    def center_x(self) -> float:
        return (self.x1 + self.x2) / 2.0

    @property
    def center_y(self) -> float:
        return (self.y1 + self.y2) / 2.0

    @property
    def width(self) -> float:
        return self.x2 - self.x1


class SceneRegions(BaseModel):
    """Vision-LLM output for a single scene keyframe.

    Flow: detect a scene change locally (cheap) -> extract one keyframe per
    scene -> send that keyframe to a vision LLM with an OCR hint -> get
    normalized bounding boxes for the on-screen roles (``person``,
    ``chart``). Those boxes drive ``person_x_norm`` / ``chart_x_norm`` on a
    ``LayoutInstruction`` without any pixel code running in Python.
    """

    scene_id: str
    person_bbox: BoundingBox | None = None
    chart_bbox: BoundingBox | None = None
    ocr_text: str = ""
    raw_reason: str = ""


# ---------------------------------------------------------------------------
# Clip planning
# ---------------------------------------------------------------------------


class RuleScore(BaseModel):
    """One named rule outcome produced by clip selection."""

    rule_id: str
    score: float = Field(
        ge=0.0,
        le=1.0,
        description="Normalized rule score (0 = absent, 1 = very strong).",
    )
    reason: str = ""


class Clip(BaseModel):
    clip_id: str
    topic: str
    start_time_sec: float = Field(ge=0)
    end_time_sec: float = Field(gt=0)
    viral_hook: str = ""
    virality_score: float = Field(default=0.0, ge=0.0, le=1.0)
    transcript: str = ""
    suggested_overlay_title: str = ""
    layout: LayoutKind | None = None

    # Optional LLM metadata (source timeline is start_time_sec / end_time_sec).
    hook_start_sec: float | None = Field(
        default=None,
        description="Seconds from clip in-point where the viral hook begins (0 = clip start).",
    )
    hook_end_sec: float | None = Field(
        default=None,
        description="Seconds from clip in-point where the hook ends (exclusive upper bound).",
    )
    trim_start_sec: float = Field(
        default=0.0,
        ge=0,
        description="Seconds to remove from the start of this segment when exporting.",
    )
    trim_end_sec: float = Field(
        default=0.0,
        ge=0,
        description="Seconds to remove from the end of this segment when exporting.",
    )
    shorts_title: str = ""
    description: str = ""
    hashtags: list[str] = Field(default_factory=list)
    layout_hint: LayoutKind | None = None
    selection_reason: str = ""
    rule_scores: list[RuleScore] = Field(default_factory=list)
    needs_review: bool = False
    review_reason: str = ""
    keep_ranges_sec: list[tuple[float, float]] = Field(
        default_factory=list,
        description=(
            "Clip-relative [start, end] spans to KEEP after inner audio pruning. "
            "Empty means keep the whole clip after trim_start_sec/trim_end_sec."
        ),
    )

    @model_validator(mode="after")
    def _timing_consistency(self) -> "Clip":
        eps = 1e-6
        if self.end_time_sec <= self.start_time_sec:
            raise ValueError("end_time_sec must be greater than start_time_sec")
        dur = self.end_time_sec - self.start_time_sec
        hs, he = self.hook_start_sec, self.hook_end_sec
        if (hs is None) ^ (he is None):
            raise ValueError("hook_start_sec and hook_end_sec must both be set or both omitted")
        if hs is not None and he is not None:
            if hs < -eps or he > dur + eps or not (hs + eps < he):
                raise ValueError(
                    "hook window must satisfy 0 <= hook_start_sec < hook_end_sec <= clip duration"
                )
        if self.trim_start_sec + self.trim_end_sec > dur + eps:
            raise ValueError("trim_start_sec + trim_end_sec must not exceed clip duration")
        prev_end = -1.0
        for idx, (start, end) in enumerate(self.keep_ranges_sec):
            if start < -eps or end > dur + eps or not (start + eps < end):
                raise ValueError(
                    "each keep_ranges_sec span must satisfy 0 <= start < end <= clip duration"
                )
            if idx > 0 and start < prev_end - eps:
                raise ValueError("keep_ranges_sec spans must be sorted and non-overlapping")
            prev_end = end
        return self

    @property
    def duration_sec(self) -> float:
        return self.end_time_sec - self.start_time_sec


class ClipPlan(BaseModel):
    """Output of the clip-selection stage â€” a list of clips + their layouts."""

    source_path: str
    clips: list[Clip]


# ---------------------------------------------------------------------------
# Render
# ---------------------------------------------------------------------------


class RenderRequest(BaseModel):
    source_path: str
    clip: Clip
    layout: LayoutInstruction
    output_path: str
    width: int = 1080
    height: int = 1920
    subtitle_path: str | None = None
    subtitle_font_size: int = Field(
        default=48,
        ge=10,
        le=120,
        description=(
            "Caption font size in **output pixels** (libass is pinned to "
            "``original_size=width x height`` by the compiler, so this is a "
            "true pixel value, not the old PlayResY=288 unit)."
        ),
    )
    subtitle_margin_v: int = Field(
        default=160,
        ge=0,
        le=800,
        description="Vertical caption margin in output pixels (bottom-anchored).",
    )
    title_text: str = ""
    mode: Literal["normal", "dry_run"] = "normal"


class RenderResult(BaseModel):
    clip_id: str
    output_path: str
    ffmpeg_cmd: list[str]
    success: bool
    error: str = ""


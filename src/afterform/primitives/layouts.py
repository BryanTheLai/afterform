"""The 9:16 layout thrusters â€” deterministic crop + compose math.

First principles: this video format has a hard constraint of **at most two
on-screen items** per short (see :class:`afterform.schemas.LayoutKind`). That
gives exactly five recipes:

* 1 person alone, tight  â†’ ``ZOOM_CALL_CENTER``
* 1 person alone, wider  â†’ ``SIT_CENTER``
* 1 chart + 1 person     â†’ ``SPLIT_CHART_PERSON``
* 2 persons              â†’ ``SPLIT_TWO_PERSONS``
* 2 charts               â†’ ``SPLIT_TWO_CHARTS``

Each planner returns a pure ``ffmpeg -filter_complex`` fragment ending in
``[vout]``. The compiler (``compile.py``) glues the fragment to the cut +
audio + subtitle chain. Because every planner is a pure function that
returns a string, the whole layout system is unit-testable without ever
invoking ffmpeg.

Split layouts share one contract:

* Output: 9:16 frame split into a **top band** and **bottom band**.
  Band heights are driven by :attr:`LayoutInstruction.top_band_ratio`.
  Default is ``0.5`` (even 50/50), matching the user-requested symmetric look.
* Source strips for the two items are **complementary** â€” they partition
  the source width at a single seam so the two items never overlap and
  together cover the full frame width.
* Each strip is scaled to fill its output band using the "cover"
  convention (``force_original_aspect_ratio=increase`` + center crop), so
  the band is fully painted (no letterbox bars, no stretch).
"""

from __future__ import annotations

from dataclasses import dataclass

from ..schemas import BoundingBox, FocusStackOrder, LayoutInstruction, LayoutKind


# Source geometry assumption. Most podcast sources are 1920x1080; we still
# normalize everything by the actual source size so changing this is safe.
DEFAULT_SRC_W = 1920
DEFAULT_SRC_H = 1080


@dataclass(frozen=True)
class FilterPlan:
    """Result of planning a layout.

    ``filtergraph`` is the body of ``-filter_complex`` and ends with
    ``[vout]`` as the final labelled stream.
    """

    filtergraph: str
    out_label: str = "vout"


# ---------------------------------------------------------------------------
# Tiny pixel helpers
# ---------------------------------------------------------------------------


def _clamp01(v: float) -> float:
    return max(0.0, min(1.0, v))


def _even(v: int) -> int:
    """Floor ``v`` to an even integer (ffmpeg ``crop``/``scale`` need even dims)."""
    return v - (v % 2)


def _bbox_to_crop_pixels(
    box: BoundingBox, src_w: int, src_h: int
) -> tuple[int, int, int, int]:
    """Normalized bbox â†’ ``(cw, ch, x, y)`` with even dimensions for ffmpeg."""
    x1 = int(round(_clamp01(box.x1) * float(src_w)))
    y1 = int(round(_clamp01(box.y1) * float(src_h)))
    x2 = int(round(_clamp01(box.x2) * float(src_w)))
    y2 = int(round(_clamp01(box.y2) * float(src_h)))
    x1 = max(0, min(src_w - 2, x1))
    y1 = max(0, min(src_h - 2, y1))
    x2 = max(x1 + 2, min(src_w, x2))
    y2 = max(y1 + 2, min(src_h, y2))
    cw = _even(x2 - x1)
    ch = _even(y2 - y1)
    return max(2, cw), max(2, ch), _even(x1), _even(y1)


def _crop_box(
    src_w: int,
    src_h: int,
    target_aspect: float,
    zoom: float,
    center_x_norm: float,
    center_y_norm: float = 0.5,
) -> tuple[int, int, int, int]:
    """Return ``(cw, ch, x, y)`` crop values for a centered aspect-ratio crop.

    ``zoom > 1`` means tighter crop (smaller window around the center). The
    function always keeps the crop window fully inside the source frame.
    """

    zoom = max(1.0, zoom)
    if src_w / src_h >= target_aspect:
        base_ch = src_h
        base_cw = int(round(base_ch * target_aspect))
    else:
        base_cw = src_w
        base_ch = int(round(base_cw / target_aspect))

    cw = _even(max(2, int(round(base_cw / zoom))))
    ch = _even(max(2, int(round(base_ch / zoom))))

    cx = int(round(_clamp01(center_x_norm) * src_w))
    cy = int(round(_clamp01(center_y_norm) * src_h))
    x = _even(max(0, min(src_w - cw, cx - cw // 2)))
    y = _even(max(0, min(src_h - ch, cy - ch // 2)))
    return cw, ch, x, y


def _center_crop_to_9x16(
    src_w: int, src_h: int, zoom: float, person_x_norm: float
) -> tuple[int, int, int, int]:
    return _crop_box(src_w, src_h, 9 / 16, zoom, person_x_norm, 0.5)


# ---------------------------------------------------------------------------
# Split helpers â€” shared by all three split layouts
# ---------------------------------------------------------------------------


# Minimum source-strip width for a split, as a fraction of source width.
# Prevents a chart/person bbox that hugs one edge from starving the other.
_MIN_SPLIT_STRIP_FRAC = 0.2


@dataclass(frozen=True)
class _SplitStrip:
    """A source-frame crop rectangle destined for one output band."""

    cw: int
    ch: int
    x: int
    y: int

    def filter_crop(self, input_label: str, out_w: int, band_h: int, out_label: str) -> str:
        """Return ``[input]crop=...,scale=...,crop=...,setsar=1[out_label]``.

        Uses the "cover" convention: scale so the band is fully painted, then
        center-crop any overflow. Bands always get filled â€” no letterbox bars.
        """
        return (
            f"[{input_label}]crop={self.cw}:{self.ch}:{self.x}:{self.y},"
            f"scale={out_w}:{band_h}:force_original_aspect_ratio=increase,"
            f"crop={out_w}:{band_h},setsar=1[{out_label}]"
        )

    def filter_contain(self, input_label: str, out_w: int, band_h: int, out_label: str) -> str:
        """Return readable contained foreground over a blurred cover background."""
        bg_src = f"{out_label}_bg_src"
        fg_src = f"{out_label}_fg_src"
        bg = f"{out_label}_bg"
        fg = f"{out_label}_fg"
        return (
            f"[{input_label}]crop={self.cw}:{self.ch}:{self.x}:{self.y},"
            f"split=2[{bg_src}][{fg_src}];"
            f"[{bg_src}]scale={out_w}:{band_h}:force_original_aspect_ratio=increase,"
            f"crop={out_w}:{band_h},gblur=sigma=18[{bg}];"
            f"[{fg_src}]scale={out_w}:{band_h}:force_original_aspect_ratio=decrease[{fg}];"
            f"[{bg}][{fg}]overlay=(W-w)/2:(H-h)/2,setsar=1[{out_label}]"
        )


def _bbox_strip(
    box: BoundingBox | None,
    *,
    src_w: int,
    src_h: int,
    x_start: int,
    x_end: int,
) -> _SplitStrip:
    """Build a source crop for one band.

    Horizontal range is fixed by ``[x_start, x_end)`` (from the seam math so
    strips partition the source width). Vertical range comes from ``box``
    when available â€” that's what makes the chart **fill** the output band
    instead of being squashed inside full-height source context.
    """
    x = _even(max(0, min(src_w - 2, x_start)))
    cw = _even(max(2, min(src_w - x, x_end - x)))

    if box is not None:
        y1 = int(round(_clamp01(box.y1) * float(src_h)))
        y2 = int(round(_clamp01(box.y2) * float(src_h)))
        y = _even(max(0, min(src_h - 2, y1)))
        ch = _even(max(2, min(src_h - y, y2 - y)))
    else:
        y = 0
        ch = _even(src_h)

    return _SplitStrip(cw=cw, ch=ch, x=x, y=y)


def _compute_seam(
    *,
    left_box: BoundingBox | None,
    right_box: BoundingBox | None,
    src_w: int,
    src_h: int,
    default_fraction: float = 0.5,
) -> int:
    """Return an even x-coordinate that partitions the source into two strips.

    When both bboxes are known, the seam is the midpoint of the gap/overlap
    between ``left_box.x2`` and ``right_box.x1``. Falls back to
    ``default_fraction * src_w`` (0.5 = even) otherwise. The seam is clamped
    so neither strip is thinner than :data:`_MIN_SPLIT_STRIP_FRAC` of source.
    """
    if left_box is not None and right_box is not None:
        _, _, left_x, _ = _bbox_to_crop_pixels(left_box, src_w, src_h)
        left_cw, _, _, _ = _bbox_to_crop_pixels(left_box, src_w, src_h)
        _, _, right_x, _ = _bbox_to_crop_pixels(right_box, src_w, src_h)

        left_right = left_x + left_cw
        seam = int(round((left_right + right_x) / 2.0))
    else:
        seam = int(round(default_fraction * float(src_w)))

    seam = _even(seam)
    min_strip = _even(max(2, int(round(src_w * _MIN_SPLIT_STRIP_FRAC))))
    if min_strip * 2 >= src_w:
        min_strip = _even(max(2, src_w // 4))
    return max(min_strip, min(src_w - min_strip, seam))


def _band_heights(out_h: int, top_ratio: float) -> tuple[int, int]:
    """Return ``(top_h, bot_h)`` even band heights that sum to ``out_h``."""
    top_h = _even(int(round(out_h * top_ratio)))
    top_h = max(2, min(out_h - 2, top_h))
    bot_h = out_h - top_h
    return top_h, bot_h


def _stack_filtergraph(
    *,
    top_strip: _SplitStrip,
    bot_strip: _SplitStrip,
    out_w: int,
    top_h: int,
    bot_h: int,
    input_label: str,
) -> str:
    """Compose the split filter graph: ``[0:v]split=2 â†’ two crops â†’ vstack â†’ [vout]``."""
    top_fg = top_strip.filter_crop("src1", out_w, top_h, "top")
    bot_fg = bot_strip.filter_crop("src2", out_w, bot_h, "bot")
    return (
        f"[{input_label}]split=2[src1][src2];"
        f"{top_fg};"
        f"{bot_fg};"
        f"[top][bot]vstack=inputs=2[vout]"
    )


# ---------------------------------------------------------------------------
# Layout: single-subject (centered) â€” 1 person
# ---------------------------------------------------------------------------


def plan_zoom_call_center(
    instruction: LayoutInstruction,
    *,
    out_w: int,
    out_h: int,
    src_w: int = DEFAULT_SRC_W,
    src_h: int = DEFAULT_SRC_H,
    input_label: str = "0:v",
) -> FilterPlan:
    """1 person, tight zoom-call framing driven by the caller's chosen zoom."""
    zoom = max(instruction.zoom, 1.0)
    cw, ch, x, y = _center_crop_to_9x16(src_w, src_h, zoom, instruction.person_x_norm)
    fg = (
        f"[{input_label}]crop={cw}:{ch}:{x}:{y},"
        f"scale={out_w}:{out_h}:flags=lanczos,setsar=1[vout]"
    )
    return FilterPlan(filtergraph=fg)


def plan_sit_center(
    instruction: LayoutInstruction,
    *,
    out_w: int,
    out_h: int,
    src_w: int = DEFAULT_SRC_W,
    src_h: int = DEFAULT_SRC_H,
    input_label: str = "0:v",
) -> FilterPlan:
    """1 person, interview/seated framing. Vertical center biased to ``0.48``
    so faces sit slightly above the 9:16 middle instead of centered on a
    subject's chest.
    """
    if instruction.zoom < 1.0:
        strip = _SplitStrip(cw=_even(src_w), ch=_even(src_h), x=0, y=0)
        return FilterPlan(filtergraph=strip.filter_contain(input_label, out_w, out_h, "vout"))

    zoom = max(instruction.zoom, 1.0)
    cw, ch, x, y = _crop_box(
        src_w, src_h, 9 / 16, zoom, instruction.person_x_norm, 0.48
    )
    fg = (
        f"[{input_label}]crop={cw}:{ch}:{x}:{y},"
        f"scale={out_w}:{out_h}:flags=lanczos,setsar=1[vout]"
    )
    return FilterPlan(filtergraph=fg)


def plan_wide_visual(
    instruction: LayoutInstruction,
    *,
    out_w: int,
    out_h: int,
    src_w: int = DEFAULT_SRC_W,
    src_h: int = DEFAULT_SRC_H,
    input_label: str = "0:v",
) -> FilterPlan:
    """1 wide visual/article/screenshot.

    If vision gave us a very wide/thin region, use a cover crop so article
    text/screenshots become readable in 9:16. Without a region, preserve the
    full source frame with letterbox padding.
    """
    prefix = f"[{input_label}]"
    if instruction.split_chart_region is not None:
        region = instruction.split_chart_region
        cw, ch, x, y = _bbox_to_crop_pixels(region, src_w, src_h)
        pad_x = _even(max(0, int(round(cw * 0.08))))
        pad_y = _even(max(0, int(round(ch * 0.12))))
        x = _even(max(0, x - pad_x))
        y = _even(max(0, y - pad_y))
        cw = _even(max(2, min(src_w - x, cw + pad_x * 2)))
        ch = _even(max(2, min(src_h - y, ch + pad_y * 2)))
        strip = _SplitStrip(cw=cw, ch=ch, x=x, y=y)
        return FilterPlan(filtergraph=strip.filter_contain(input_label, out_w, out_h, "vout"))
    fg = (
        f"{prefix}scale={out_w}:{out_h}:force_original_aspect_ratio=decrease,"
        f"pad={out_w}:{out_h}:(ow-iw)/2:(oh-ih)/2:black,setsar=1[vout]"
    )
    return FilterPlan(filtergraph=fg)


# ---------------------------------------------------------------------------
# Split layouts â€” 2 items stacked vertically
# ---------------------------------------------------------------------------


def plan_split_chart_person(
    instruction: LayoutInstruction,
    *,
    out_w: int,
    out_h: int,
    src_w: int = DEFAULT_SRC_W,
    src_h: int = DEFAULT_SRC_H,
    input_label: str = "0:v",
) -> FilterPlan:
    """1 chart + 1 person.

    **Horizontal partition.** Chart occupies the left source strip, person the
    right strip. When both bboxes are set (Gemini vision), the seam sits at
    the midpoint between ``chart.x2`` and ``person.x1`` so the strips are
    complementary (no overlap, no gap). Otherwise the seam defaults to a
    2/3 | 1/3 split (chart left, person right), matching the Ark-style
    explainer-slide geometry this codebase was originally written against.

    **Vertical crop.** Each strip's vertical extent comes from the
    corresponding bbox when provided â€” crucial so the chart **fills** its
    output band instead of being lost inside full-height source context
    (plant, background, lower-third graphics, etc.). Falls back to full
    source height when bboxes are unavailable.

    **Output bands.** Controlled by :attr:`LayoutInstruction.top_band_ratio`
    (default 0.5 = even 50/50 â€” the user-requested symmetric look). Focus
    stack order picks chart-on-top (default) vs person-on-top.
    """

    top_h, bot_h = _band_heights(out_h, instruction.top_band_ratio)

    chart_box = instruction.split_chart_region
    person_box = instruction.split_person_region

    if chart_box is not None and person_box is not None:
        chart_is_left = chart_box.center_x <= person_box.center_x
        left_box = chart_box if chart_is_left else person_box
        right_box = person_box if chart_is_left else chart_box
        seam = _compute_seam(
            left_box=left_box, right_box=right_box, src_w=src_w, src_h=src_h
        )
        chart_start = 0 if chart_is_left else seam
        chart_end = seam if chart_is_left else src_w
        person_start = seam if chart_is_left else 0
        person_end = src_w if chart_is_left else seam
    else:
        # Historical default: chart = left 2/3, person = right 1/3 (the
        # Ark-style explainer-slide geometry this codebase was originally
        # written against). ``chart_x_norm`` trims the chart strip from its
        # left edge when we have no vision bbox to do it precisely.
        seam = _even(max(2, min(src_w - 2, int(round((2.0 / 3.0) * float(src_w))))))
        trim = int(round(_clamp01(instruction.chart_x_norm) * float(seam)))
        chart_start = _even(max(0, min(seam - 2, trim)))
        chart_end = seam
        person_start = seam
        person_end = src_w

    # The LLM often reports tight face boxes inside a larger photo insert
    # (for example a founder headshot) as ``chart_bbox``. If we use that
    # bbox vertically, cover-scaling turns the top band into an eye/nose crop.
    # Once both side panes are known, render the full pane height; the seam
    # still preserves left/right composition while avoiding bbox over-trust.
    use_full_side_panes = chart_box is not None and person_box is not None
    chart_strip = _bbox_strip(
        None if use_full_side_panes else chart_box,
        src_w=src_w,
        src_h=src_h,
        x_start=chart_start,
        x_end=chart_end,
    )
    person_strip = _bbox_strip(
        None if use_full_side_panes else person_box,
        src_w=src_w,
        src_h=src_h,
        x_start=person_start,
        x_end=person_end,
    )
    return _emit_split(
        chart_strip=chart_strip,
        person_strip=person_strip,
        order=instruction.focus_stack_order,
        out_w=out_w,
        top_h=top_h,
        bot_h=bot_h,
        input_label=input_label,
    )


def _emit_split(
    *,
    chart_strip: _SplitStrip,
    person_strip: _SplitStrip,
    order: FocusStackOrder,
    out_w: int,
    top_h: int,
    bot_h: int,
    input_label: str,
) -> FilterPlan:
    if order == FocusStackOrder.CHART_THEN_PERSON:
        top_fg = chart_strip.filter_contain("src1", out_w, top_h, "top")
        bot_fg = person_strip.filter_crop("src2", out_w, bot_h, "bot")
        fg = (
            f"[{input_label}]split=2[src1][src2];"
            f"{top_fg};"
            f"{bot_fg};"
            f"[top][bot]vstack=inputs=2[vout]"
        )
    else:
        top_fg = person_strip.filter_crop("src1", out_w, top_h, "top")
        bot_fg = chart_strip.filter_contain("src2", out_w, bot_h, "bot")
        fg = (
            f"[{input_label}]split=2[src1][src2];"
            f"{top_fg};"
            f"{bot_fg};"
            f"[top][bot]vstack=inputs=2[vout]"
        )
    return FilterPlan(filtergraph=fg)


def plan_split_two_persons(
    instruction: LayoutInstruction,
    *,
    out_w: int,
    out_h: int,
    src_w: int = DEFAULT_SRC_W,
    src_h: int = DEFAULT_SRC_H,
    input_label: str = "0:v",
) -> FilterPlan:
    """2 persons (interview two-up) stacked vertically.

    First person = ``split_person_region``, second person = ``split_second_person_region``.
    Seam sits at the midpoint between the two bboxes when both are known;
    otherwise defaults to a centered 50/50 split.
    """
    top_h, bot_h = _band_heights(out_h, instruction.top_band_ratio)

    left_box = instruction.split_person_region
    right_box = instruction.split_second_person_region

    seam = _compute_seam(
        left_box=left_box, right_box=right_box, src_w=src_w, src_h=src_h
    )

    left_strip = _bbox_strip(
        left_box, src_w=src_w, src_h=src_h, x_start=0, x_end=seam
    )
    right_strip = _bbox_strip(
        right_box, src_w=src_w, src_h=src_h, x_start=seam, x_end=src_w
    )
    fg = _stack_filtergraph(
        top_strip=left_strip,
        bot_strip=right_strip,
        out_w=out_w,
        top_h=top_h,
        bot_h=bot_h,
        input_label=input_label,
    )
    return FilterPlan(filtergraph=fg)


def plan_split_two_charts(
    instruction: LayoutInstruction,
    *,
    out_w: int,
    out_h: int,
    src_w: int = DEFAULT_SRC_W,
    src_h: int = DEFAULT_SRC_H,
    input_label: str = "0:v",
) -> FilterPlan:
    """2 charts stacked vertically.

    First chart = ``split_chart_region``, second chart = ``split_second_chart_region``.
    Uses the same seam/bbox-y-crop recipe as the other splits, so each chart
    fills its output band instead of being surrounded by source context.
    """
    top_h, bot_h = _band_heights(out_h, instruction.top_band_ratio)

    left_box = instruction.split_chart_region
    right_box = instruction.split_second_chart_region

    seam = _compute_seam(
        left_box=left_box, right_box=right_box, src_w=src_w, src_h=src_h
    )

    left_strip = _bbox_strip(
        left_box, src_w=src_w, src_h=src_h, x_start=0, x_end=seam
    )
    right_strip = _bbox_strip(
        right_box, src_w=src_w, src_h=src_h, x_start=seam, x_end=src_w
    )
    fg = _stack_filtergraph(
        top_strip=left_strip,
        bot_strip=right_strip,
        out_w=out_w,
        top_h=top_h,
        bot_h=bot_h,
        input_label=input_label,
    )
    return FilterPlan(filtergraph=fg)


_DISPATCH = {
    LayoutKind.ZOOM_CALL_CENTER: plan_zoom_call_center,
    LayoutKind.SIT_CENTER: plan_sit_center,
    LayoutKind.WIDE_VISUAL: plan_wide_visual,
    LayoutKind.SPLIT_CHART_PERSON: plan_split_chart_person,
    LayoutKind.SPLIT_TWO_PERSONS: plan_split_two_persons,
    LayoutKind.SPLIT_TWO_CHARTS: plan_split_two_charts,
}


def plan_layout(
    instruction: LayoutInstruction,
    *,
    out_w: int = 1080,
    out_h: int = 1920,
    src_w: int = DEFAULT_SRC_W,
    src_h: int = DEFAULT_SRC_H,
    input_label: str = "0:v",
) -> FilterPlan:
    """Dispatch to one of the five thrusters.

    Exhaustive over :class:`LayoutKind` â€” adding a new layout requires adding
    a planner above **and** an entry in :data:`_DISPATCH`.
    """

    fn = _DISPATCH.get(instruction.layout)
    if fn is None:
        raise ValueError(f"Unknown layout: {instruction.layout!r}")
    return fn(
        instruction,
        out_w=out_w,
        out_h=out_h,
        src_w=src_w,
        src_h=src_h,
        input_label=input_label,
    )


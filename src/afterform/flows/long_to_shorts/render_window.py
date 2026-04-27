"""Map clip timing metadata to honest source ranges for render/layout/subtitles."""

from __future__ import annotations

from afterform.schemas import Clip


def effective_keep_ranges(clip: Clip) -> list[tuple[float, float]]:
    """Return clip-relative spans that survive trim + inner keep-ranges.

    Contract:
    - ``trim_start_sec`` / ``trim_end_sec`` define the outer legal window.
    - ``keep_ranges_sec`` further removes interior filler/silence inside that
      outer window.
    - Empty ``keep_ranges_sec`` means "keep the whole trimmed window".
    """
    duration = clip.duration_sec
    outer_start = max(0.0, min(duration, clip.trim_start_sec))
    outer_end = max(outer_start, min(duration, duration - clip.trim_end_sec))
    if outer_end <= outer_start:
        outer_start, outer_end = 0.0, duration

    if not clip.keep_ranges_sec:
        return [(outer_start, outer_end)]

    kept: list[tuple[float, float]] = []
    for start, end in clip.keep_ranges_sec:
        lo = max(outer_start, float(start))
        hi = min(outer_end, float(end))
        if hi <= lo:
            continue
        if kept and lo <= kept[-1][1] + 1e-6:
            prev_lo, prev_hi = kept[-1]
            kept[-1] = (prev_lo, max(prev_hi, hi))
            continue
        kept.append((lo, hi))

    if kept:
        return kept
    return [(outer_start, outer_end)]


def source_keep_ranges(clip: Clip) -> list[tuple[float, float]]:
    """Return absolute source-timeline spans that survive pruning."""
    return [
        (clip.start_time_sec + start, clip.start_time_sec + end)
        for start, end in effective_keep_ranges(clip)
    ]


def clip_output_duration(clip: Clip) -> float:
    """Total output duration after concatenating all kept spans."""
    return sum(end - start for start, end in effective_keep_ranges(clip))


def effective_export_bounds(clip: Clip) -> tuple[float, float]:
    """Return the outer source bounds containing all kept spans.

    This is not the final output duration when ``keep_ranges_sec`` has holes.
    It is the bounding source window that encloses every kept span.
    """
    ranges = source_keep_ranges(clip)
    if not ranges:
        return clip.start_time_sec, clip.end_time_sec
    return ranges[0][0], ranges[-1][1]


def clip_for_render(clip: Clip) -> Clip:
    """Copy with start/end set to outer kept bounds and keeps normalized to that window."""
    keep_ranges = effective_keep_ranges(clip)
    if not keep_ranges:
        t0, t1 = round(clip.start_time_sec, 6), round(clip.end_time_sec, 6)
        normalized_keep_ranges: list[tuple[float, float]] = []
    else:
        rel_start = keep_ranges[0][0]
        rel_end = keep_ranges[-1][1]
        total = round(rel_end - rel_start, 6)
        t0 = round(clip.start_time_sec + rel_start, 6)
        t1 = round(t0 + total, 6)
        normalized_keep_ranges = []
        for start, end in keep_ranges:
            lo = max(0.0, min(total, start - rel_start))
            hi = max(lo, min(total, end - rel_start))
            if abs(lo) < 1e-9:
                lo = 0.0
            if abs(total - hi) < 1e-9:
                hi = total
            lo = round(lo, 6)
            hi = round(hi, 6)
            if hi <= lo:
                continue
            normalized_keep_ranges.append((lo, hi))
        if (
            len(normalized_keep_ranges) == 1
            and abs(normalized_keep_ranges[0][0]) < 1e-6
            and abs(normalized_keep_ranges[0][1] - total) < 1e-6
        ):
            normalized_keep_ranges = []

    payload = clip.model_dump()
    payload.update(
        {
            "start_time_sec": t0,
            "end_time_sec": t1,
            "trim_start_sec": 0.0,
            "trim_end_sec": 0.0,
            "hook_start_sec": None,
            "hook_end_sec": None,
            "keep_ranges_sec": normalized_keep_ranges,
        }
    )
    return Clip.model_validate(payload)


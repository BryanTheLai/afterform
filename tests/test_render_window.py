"""Tests for trim/hook/keep-ranges â†’ ffmpeg source windows."""

import pytest

from afterform.flows.long_to_shorts.render_window import (
    apply_visual_drop_ranges,
    clip_for_render,
    effective_export_bounds,
    effective_keep_ranges,
    source_keep_ranges,
)
from afterform.schemas import Clip, LayoutInstruction, LayoutKind, RenderRequest


def _clip(**kwargs) -> Clip:
    base = dict(
        clip_id="1",
        topic="t",
        start_time_sec=100.0,
        end_time_sec=130.0,
    )
    base.update(kwargs)
    return Clip.model_validate(base)


def test_trim_only():
    c = _clip(trim_start_sec=5.0, trim_end_sec=3.0)
    lo, hi = effective_export_bounds(c)
    assert lo == 105.0
    assert hi == 127.0


def test_hook_does_not_shorten_export_window():
    c = _clip(
        trim_start_sec=0.0,
        trim_end_sec=0.0,
        hook_start_sec=2.0,
        hook_end_sec=8.0,
    )
    lo, hi = effective_export_bounds(c)
    assert lo == 100.0
    assert hi == 130.0


def test_clip_for_render_clears_timing_fields():
    c = _clip(
        trim_start_sec=0.0,
        trim_end_sec=0.0,
        hook_start_sec=2.0,
        hook_end_sec=10.0,
    )
    r = clip_for_render(c)
    assert r.start_time_sec == 100.0
    assert r.end_time_sec == 130.0
    assert r.trim_start_sec == 0.0
    assert r.hook_start_sec is None


def test_keep_ranges_are_intersected_with_outer_trim_window():
    c = _clip(
        trim_start_sec=5.0,
        trim_end_sec=3.0,
        keep_ranges_sec=[(0.0, 8.0), (12.0, 20.0), (25.0, 29.0)],
    )
    assert effective_keep_ranges(c) == [(5.0, 8.0), (12.0, 20.0), (25.0, 27.0)]
    assert source_keep_ranges(c) == [(105.0, 108.0), (112.0, 120.0), (125.0, 127.0)]


def test_apply_visual_drop_ranges_splits_existing_keep_span():
    c = _clip(keep_ranges_sec=[(0.0, 10.0), (12.0, 20.0)])

    updated = apply_visual_drop_ranges(c, [(4.0, 5.0), (14.0, 15.5)])

    assert updated.keep_ranges_sec == [
        (0.0, 4.0),
        (5.0, 10.0),
        (12.0, 14.0),
        (15.5, 20.0),
    ]


def test_clip_for_render_normalizes_inner_keep_gaps():
    c = _clip(
        trim_start_sec=4.0,
        trim_end_sec=2.0,
        keep_ranges_sec=[(4.0, 10.0), (15.0, 20.0)],
    )
    r = clip_for_render(c)
    assert r.start_time_sec == 104.0
    assert r.end_time_sec == 120.0
    assert r.keep_ranges_sec == [(0.0, 6.0), (11.0, 16.0)]


def test_clip_for_render_clamps_float_boundary_to_duration():
    c = Clip(
        clip_id="003",
        topic="t",
        start_time_sec=595.2,
        end_time_sec=658.7,
        trim_start_sec=6.35,
        trim_end_sec=0.0,
        keep_ranges_sec=[(6.35, 7.34), (61.47, 62.62), (62.62, 63.5)],
    )
    r = clip_for_render(c)
    assert r.duration_sec == pytest.approx(57.15)
    assert max(end for _, end in r.keep_ranges_sec) <= r.duration_sec
    assert r.keep_ranges_sec[-1][1] == pytest.approx(r.duration_sec)


def test_clip_for_render_output_validates_in_render_request_for_float_edge_case():
    c = Clip(
        clip_id="002",
        topic="t",
        start_time_sec=1433.6,
        end_time_sec=1524.3,
        trim_start_sec=1.11,
        trim_end_sec=0.01,
        keep_ranges_sec=[
            (1.11, 2.231),
            (3.332, 5.69),
            (6.555, 7.19),
            (7.31, 9.29),
            (10.159, 12.24),
            (12.7, 15.51),
            (16.03, 18.5),
            (18.99, 22.09),
            (22.25, 23.69),
            (24.533, 25.51),
            (25.66, 27.56),
            (27.92, 32.65),
            (32.99, 35.585),
            (36.025, 36.96),
            (37.967, 42.11),
            (42.41, 45.32),
            (45.45, 47.4),
            (47.96, 49.63),
            (49.76, 51.78),
            (52.19, 56.74),
            (57.19, 57.95),
            (58.34, 59.08),
            (59.749, 61.62),
            (63.1, 64.293),
            (64.794, 72.62),
            (73.18, 78.04),
            (78.42, 80.01),
            (80.99, 85.22),
            (85.58, 90.69),
        ],
    )
    r = clip_for_render(c)
    req = RenderRequest(
        source_path="in.mp4",
        clip=r,
        layout=LayoutInstruction(clip_id="002", layout=LayoutKind.SPLIT_CHART_PERSON),
        output_path="out.mp4",
        mode="dry_run",
    )
    assert req.clip.keep_ranges_sec[-1][1] <= req.clip.duration_sec + 1e-6


"""Thin adapter from the product pipeline to the reusable render primitive."""

from __future__ import annotations

import logging
from pathlib import Path

from afterform.primitives import compile as compile_mod
from afterform.schemas import (
    Clip,
    LayoutInstruction,
    LayoutKind,
    RenderRequest,
)

logger = logging.getLogger(__name__)


def layout_for_clip(
    clip: Clip,
    default_layout: LayoutKind = LayoutKind.SIT_CENTER,
    zoom: float = 1.0,
) -> LayoutInstruction:
    """Build the layout instruction for a clip using the shared schema."""
    layout = clip.layout or default_layout
    return LayoutInstruction(clip_id=clip.clip_id, layout=layout, zoom=zoom)


def reframe_clip_ffmpeg(
    input_path: Path | str,
    output_path: Path | str,
    clip: Clip,
    *,
    zoom: float = 1.0,
    layout_instruction: LayoutInstruction | None = None,
    subtitle_path: Path | str | None = None,
    subtitle_font_size: int = 48,
    subtitle_margin_v: int = 160,
    title_text: str = "",
    dry_run: bool = False,
) -> RenderRequest:
    """Render a single clip to 9:16 via one ffmpeg call.

    If ``layout_instruction`` is set (e.g. from Gemini vision), it is used in full
    including ``person_x_norm``, ``chart_x_norm``, and optional split bbox fields.
    Otherwise defaults are derived from ``clip.layout`` via ``layout_for_clip``.
    """

    instr = layout_instruction if layout_instruction is not None else layout_for_clip(clip, zoom=zoom)
    req = RenderRequest(
        source_path=str(input_path),
        clip=clip,
        layout=instr,
        output_path=str(output_path),
        subtitle_path=str(subtitle_path) if subtitle_path else None,
        subtitle_font_size=subtitle_font_size,
        subtitle_margin_v=subtitle_margin_v,
        title_text=title_text,
        mode="dry_run" if dry_run else "normal",
    )
    result = compile_mod.render_clip(req)
    if not result.success and not dry_run:
        raise RuntimeError(f"ffmpeg failed for clip {clip.clip_id}: {result.error}")
    logger.info(
        "reframe_clip_ffmpeg: clip=%s layout=%s output=%s success=%s",
        clip.clip_id,
        instr.layout.value,
        output_path,
        result.success,
    )
    return req


"""Thin adapter from the product pipeline to the reusable render primitive."""

from __future__ import annotations

import logging
import shutil
import subprocess
from pathlib import Path

from afterform.primitives import compile as compile_mod
from afterform.flows.long_to_shorts.render_window import subclip_for_output_range
from afterform.schemas import (
    Clip,
    ClipLayoutPlan,
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


def _ffmpeg_exe(dry_run: bool) -> str:
    if dry_run:
        return "ffmpeg"
    exe = shutil.which("ffmpeg")
    if not exe:
        raise RuntimeError("ffmpeg not found on PATH")
    return exe


def _escape_filter_path(path: Path | str) -> str:
    return str(path).replace("\\", "/").replace(":", "\\:")


def _concat_segments(
    segment_paths: list[Path],
    concat_list_path: Path,
    concat_output_path: Path,
    *,
    dry_run: bool,
) -> list[str]:
    concat_list_path.write_text(
        "".join(f"file '{path.as_posix()}'\n" for path in segment_paths),
        encoding="utf-8",
    )
    cmd = [
        _ffmpeg_exe(dry_run),
        "-y",
        "-f",
        "concat",
        "-safe",
        "0",
        "-i",
        str(concat_list_path),
        "-c",
        "copy",
        str(concat_output_path),
    ]
    if not dry_run:
        subprocess.run(cmd, check=True, capture_output=True)
    return cmd


def _apply_subtitles_to_concat(
    concat_output_path: Path,
    final_output_path: Path,
    *,
    subtitle_path: Path | str | None,
    subtitle_font_size: int,
    subtitle_margin_v: int,
    dry_run: bool,
) -> list[str]:
    if subtitle_path is None:
        if not dry_run:
            shutil.copyfile(concat_output_path, final_output_path)
        return []

    subtitle_esc = _escape_filter_path(subtitle_path)
    fg = (
        f"[0:v]subtitles='{subtitle_esc}':"
        "original_size=1080x1920:"
        f"force_style='Fontname=Arial,FontSize={subtitle_font_size},Alignment=2,"
        f"MarginV={subtitle_margin_v},MarginL=60,MarginR=60,"
        "WrapStyle=0,BorderStyle=4,BackColour=&H70000000&,"
        "PrimaryColour=&H00FFFFFF&,Outline=0,Shadow=0,Bold=1'[vout]"
    )
    cmd = [
        _ffmpeg_exe(dry_run),
        "-y",
        "-i",
        str(concat_output_path),
        "-filter_complex",
        fg,
        "-map",
        "[vout]",
        "-map",
        "0:a:0",
        "-c:v",
        "libx264",
        "-preset",
        "veryfast",
        "-crf",
        "20",
        "-c:a",
        "copy",
        "-movflags",
        "+faststart",
        str(final_output_path),
    ]
    if not dry_run:
        subprocess.run(cmd, check=True, capture_output=True)
    return cmd


def reframe_clip_timeline_ffmpeg(
    input_path: Path | str,
    output_path: Path | str,
    clip: Clip,
    *,
    layout_plan: ClipLayoutPlan,
    subtitle_path: Path | str | None = None,
    subtitle_font_size: int = 48,
    subtitle_margin_v: int = 160,
    title_text: str = "",
    dry_run: bool = False,
) -> list[RenderRequest]:
    """Render one clip by switching layouts across output-relative segments."""

    timeline = layout_plan.layout_timeline
    if len(timeline) <= 1:
        instr = timeline[0].instruction if timeline else layout_plan.instruction
        return [
            reframe_clip_ffmpeg(
                input_path=input_path,
                output_path=output_path,
                clip=clip,
                layout_instruction=instr,
                subtitle_path=subtitle_path,
                subtitle_font_size=subtitle_font_size,
                subtitle_margin_v=subtitle_margin_v,
                title_text=title_text,
                dry_run=dry_run,
            )
        ]

    out_path = Path(output_path)
    work_dir = out_path.parent / f".{out_path.stem}_timeline"
    work_dir.mkdir(parents=True, exist_ok=True)
    segment_paths: list[Path] = []
    requests: list[RenderRequest] = []
    for idx, segment in enumerate(timeline):
        segment_clip = subclip_for_output_range(clip, segment.start_sec, segment.end_sec)
        segment_path = work_dir / f"{out_path.stem}_segment_{idx:03d}.mp4"
        segment_paths.append(segment_path)
        requests.append(
            reframe_clip_ffmpeg(
                input_path=input_path,
                output_path=segment_path,
                clip=segment_clip,
                layout_instruction=segment.instruction,
                subtitle_path=None,
                subtitle_font_size=subtitle_font_size,
                subtitle_margin_v=subtitle_margin_v,
                title_text=title_text,
                dry_run=dry_run,
            )
        )

    concat_output = work_dir / f"{out_path.stem}_concat.mp4"
    concat_list = work_dir / f"{out_path.stem}_concat.txt"
    _concat_segments(segment_paths, concat_list, concat_output, dry_run=dry_run)
    _apply_subtitles_to_concat(
        concat_output,
        out_path,
        subtitle_path=subtitle_path,
        subtitle_font_size=subtitle_font_size,
        subtitle_margin_v=subtitle_margin_v,
        dry_run=dry_run,
    )
    return requests


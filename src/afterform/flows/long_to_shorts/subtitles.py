"""Subtitle helpers for the product pipeline."""

import logging
from pathlib import Path

from afterform.schemas import Clip

from afterform.flows.long_to_shorts.transcript_align import (
    clip_subtitle_words,
    clip_words_to_srt_lines,
    format_ass,
    format_srt,
)

logger = logging.getLogger(__name__)


def generate_srt(
    clip: Clip,
    transcript: dict,
    output_dir: Path,
    *,
    max_words_per_cue: int = 8,
    max_cue_sec: float = 4.0,
) -> Path:
    """
    Build an SRT file from word-level ASR aligned to this clip's timeline.

    ``transcript`` is the persisted ``transcript.json`` (segments with optional
    per-word timestamps). Times are shifted so 0 = clip in-point.
    """
    srt_path = output_dir / f"clip_{clip.clip_id}.srt"
    aligned = clip_subtitle_words(transcript, clip)
    lines = clip_words_to_srt_lines(
        aligned.words,
        max_words_per_cue=max_words_per_cue,
        max_cue_sec=max_cue_sec,
    )
    srt_path.write_text(format_srt(lines), encoding="utf-8")
    logger.info("Generated SRT: %s (%d cues)", srt_path, len(lines))
    return srt_path


def generate_ass(
    clip: Clip,
    transcript: dict,
    output_dir: Path,
    *,
    max_words_per_cue: int = 4,
    max_cue_sec: float = 2.2,
    play_res_x: int = 1080,
    play_res_y: int = 1920,
    font_size: int = 48,
    margin_v: int = 160,
    margin_h: int = 60,
    font_name: str = "Arial",
) -> Path:
    """Generate an ASS caption file tuned for direct libass rendering.

    Unlike SRT â†’ libass (default PlayResY=288), an ASS file with
    ``PlayResY = output_height`` means libass' scale factor is 1.0, so the
    ``font_size`` / ``margin_v`` arguments below are honest output pixels.

    This is the root-cause fix for the "captions rendering in the middle of
    the frame, four times too large" bug the user reported.
    """
    ass_path = output_dir / f"clip_{clip.clip_id}.ass"
    aligned = clip_subtitle_words(transcript, clip)
    lines = clip_words_to_srt_lines(
        aligned.words,
        max_words_per_cue=max_words_per_cue,
        max_cue_sec=max_cue_sec,
    )
    ass_path.write_text(
        format_ass(
            lines,
            play_res_x=play_res_x,
            play_res_y=play_res_y,
            font_size=font_size,
            margin_v=margin_v,
            margin_h=margin_h,
            font_name=font_name,
        ),
        encoding="utf-8",
    )
    logger.info("Generated ASS: %s (%d cues)", ass_path, len(lines))
    return ass_path


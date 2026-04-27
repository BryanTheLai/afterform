"""Stage control, artifact reconstruction, and inspection helpers."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

from afterform.primitives.compile import build_ffmpeg_cmd
from afterform.schemas import Clip, LayoutInstruction, LayoutKind, RenderRequest

from afterform.flows.long_to_shorts.clip_selection_cache import transcript_fingerprint
from afterform.flows.long_to_shorts.select_clips import load_clips
from afterform.flows.long_to_shorts.render_window import clip_for_render, source_keep_ranges
from afterform.flows.long_to_shorts.transcript_align import clip_subtitle_words, clip_words_to_srt_lines

from afterform.config import PipelineConfig

StageName = Literal[
    "ingest",
    "clip-selection",
    "hook-detection",
    "content-pruning",
    "layout-vision",
    "render",
]

STAGE_ORDER: tuple[StageName, ...] = (
    "ingest",
    "clip-selection",
    "hook-detection",
    "content-pruning",
    "layout-vision",
    "render",
)


class StageArtifactError(RuntimeError):
    """Required stage artifact is missing or invalid."""


@dataclass
class PipelineState:
    work_dir: Path
    source_video: Path | None = None
    source_audio: Path | None = None
    transcript: dict[str, Any] | None = None
    transcript_fp: str | None = None
    clips: list[Clip] | None = None
    layout_instructions: dict[str, LayoutInstruction] | None = None


def normalize_stage(stage: str | None) -> StageName | None:
    if stage is None:
        return None
    value = stage.strip().lower()
    aliases: dict[str, StageName] = {
        "ingest": "ingest",
        "clip-selection": "clip-selection",
        "clip_selection": "clip-selection",
        "clips": "clip-selection",
        "hook-detection": "hook-detection",
        "hook_detection": "hook-detection",
        "hooks": "hook-detection",
        "content-pruning": "content-pruning",
        "content_pruning": "content-pruning",
        "pruning": "content-pruning",
        "layout-vision": "layout-vision",
        "layout_vision": "layout-vision",
        "layout": "layout-vision",
        "render": "render",
    }
    normalized = aliases.get(value)
    if normalized is None:
        valid = ", ".join(STAGE_ORDER)
        raise ValueError(f"Unknown stage {stage!r}. Expected one of: {valid}")
    return normalized


def stage_index(stage: StageName) -> int:
    return STAGE_ORDER.index(stage)


def stage_range(
    *,
    start_at: StageName | None,
    stop_after: StageName | None,
) -> tuple[StageName, StageName]:
    start = start_at or STAGE_ORDER[0]
    stop = stop_after or STAGE_ORDER[-1]
    if stage_index(start) > stage_index(stop):
        raise ValueError(f"start-at {start!r} must be before or equal to stop-after {stop!r}")
    return start, stop


def inspection_path(work_dir: Path, stage: StageName, clip_id: str | None = None) -> Path:
    out_dir = work_dir / "inspections"
    out_dir.mkdir(parents=True, exist_ok=True)
    suffix = f"_{clip_id}" if clip_id else ""
    return out_dir / f"{stage}{suffix}.json"


def write_inspection(
    work_dir: Path,
    *,
    stage: StageName,
    payload: dict[str, Any],
    clip_id: str | None,
) -> Path:
    path = inspection_path(work_dir, stage, clip_id)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return path


def artifact_paths(work_dir: Path) -> dict[str, Path]:
    return {
        "source_video": work_dir / "source.mp4",
        "source_audio": work_dir / "source_audio.wav",
        "transcript": work_dir / "transcript.json",
        "clips": work_dir / "clips.json",
        "clip_selection_raw": work_dir / "clip_selection_raw.json",
        "hooks": work_dir / "hooks.json",
        "hooks_raw": work_dir / "hooks_raw.json",
        "prune": work_dir / "prune.json",
        "prune_raw": work_dir / "prune_raw.json",
        "layout_vision": work_dir / "layout_vision.json",
        "subtitles_dir": work_dir / "subtitles",
    }


def _read_json(path: Path, *, label: str) -> dict[str, Any]:
    if not path.is_file():
        raise StageArtifactError(f"Missing {label}: {path}")
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise StageArtifactError(f"Unreadable {label} ({path}): {exc}") from exc


def _load_transcript(work_dir: Path) -> tuple[dict[str, Any], str]:
    transcript = _read_json(artifact_paths(work_dir)["transcript"], label="transcript.json")
    return transcript, transcript_fingerprint(transcript)


def _load_selected_clips(work_dir: Path) -> list[Clip]:
    path = artifact_paths(work_dir)["clips"]
    if not path.is_file():
        raise StageArtifactError(f"Missing clips.json: {path}")
    return load_clips(path)


def _apply_hook_artifact(work_dir: Path, clips: list[Clip]) -> list[Clip]:
    data = _read_json(artifact_paths(work_dir)["hooks"], label="hooks.json")
    hooks = {str(item.get("clip_id")): item for item in data.get("hooks", [])}
    out: list[Clip] = []
    for clip in clips:
        raw = hooks.get(clip.clip_id)
        if raw is None:
            out.append(clip)
            continue
        hs = raw.get("hook_start_sec")
        he = raw.get("hook_end_sec")
        if hs is None or he is None:
            out.append(clip)
            continue
        out.append(
            clip.model_copy(
                update={"hook_start_sec": float(hs), "hook_end_sec": float(he)}
            )
        )
    return out


def _apply_prune_artifact(work_dir: Path, clips: list[Clip]) -> list[Clip]:
    data = _read_json(artifact_paths(work_dir)["prune"], label="prune.json")
    decisions = {str(item.get("clip_id")): item for item in data.get("clips", [])}
    out: list[Clip] = []
    for clip in clips:
        raw = decisions.get(clip.clip_id)
        if raw is None:
            out.append(clip)
            continue
        update: dict[str, Any] = {
            "trim_start_sec": float(raw.get("trim_start_sec", 0.0)),
            "trim_end_sec": float(raw.get("trim_end_sec", 0.0)),
        }
        if "keep_ranges_sec" in raw:
            normalized_ranges: list[tuple[float, float]] = []
            for idx, item in enumerate(raw.get("keep_ranges_sec") or []):
                if not isinstance(item, (list, tuple)) or len(item) != 2:
                    raise StageArtifactError(
                        f"prune.json clip {clip.clip_id} has malformed keep_ranges_sec[{idx}]: {item!r}"
                    )
                normalized_ranges.append((float(item[0]), float(item[1])))
            update["keep_ranges_sec"] = normalized_ranges
        out.append(clip.model_copy(update=update))
    return out


def _load_layout_instructions(work_dir: Path) -> dict[str, LayoutInstruction]:
    data = _read_json(artifact_paths(work_dir)["layout_vision"], label="layout_vision.json")
    clips = data.get("clips")
    if not isinstance(clips, dict):
        raise StageArtifactError("layout_vision.json is missing the top-level 'clips' object")
    out: dict[str, LayoutInstruction] = {}
    for clip_id, payload in clips.items():
        if not isinstance(payload, dict) or "instruction" not in payload:
            continue
        out[str(clip_id)] = LayoutInstruction.model_validate(payload["instruction"])
    return out


def _clip_or_all(clips: list[Clip], clip_id: str | None) -> list[Clip]:
    if clip_id is None:
        return clips
    chosen = [clip for clip in clips if clip.clip_id == clip_id]
    if not chosen:
        raise StageArtifactError(f"clip_id {clip_id!r} not found in clips.json")
    return chosen


def _clip_excerpt(transcript: dict[str, Any], clip: Clip) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for seg in transcript.get("segments", []) or []:
        start = float(seg.get("start", 0.0))
        end = float(seg.get("end", start))
        if end <= clip.start_time_sec or start >= clip.end_time_sec:
            continue
        out.append(
            {
                "start": start,
                "end": end,
                "text": (seg.get("text") or "").strip(),
            }
        )
    return out


def load_state_before_stage(
    work_dir: Path,
    *,
    stage: StageName,
    config: PipelineConfig,
) -> PipelineState:
    state = PipelineState(work_dir=work_dir)
    paths = artifact_paths(work_dir)
    state.source_video = paths["source_video"] if paths["source_video"].is_file() else None
    state.source_audio = paths["source_audio"] if paths["source_audio"].is_file() else None
    state.transcript, state.transcript_fp = _load_transcript(work_dir)

    if stage_index(stage) >= stage_index("hook-detection"):
        state.clips = _load_selected_clips(work_dir)

    if stage_index(stage) >= stage_index("content-pruning") and state.clips is not None:
        if config.detect_hooks:
            state.clips = _apply_hook_artifact(work_dir, state.clips)

    if stage_index(stage) >= stage_index("layout-vision") and state.clips is not None:
        if (config.prune_level or "balanced").strip().lower() != "off":
            state.clips = _apply_prune_artifact(work_dir, state.clips)

    if stage_index(stage) >= stage_index("render"):
        state.layout_instructions = _load_layout_instructions(work_dir)

    return state


def build_stage_inspection(
    work_dir: Path,
    *,
    stage: StageName,
    clip_id: str | None,
    config: PipelineConfig,
) -> dict[str, Any]:
    paths = artifact_paths(work_dir)
    transcript, transcript_fp = _load_transcript(work_dir)

    if stage == "ingest":
        return {
            "stage": stage,
            "artifacts": {
                "source_video": str(paths["source_video"]),
                "source_audio": str(paths["source_audio"]),
                "transcript": str(paths["transcript"]),
            },
            "summary": {
                "transcript_sha256": transcript_fp,
                "segment_count": len(transcript.get("segments", []) or []),
            },
        }

    clips = _load_selected_clips(work_dir)

    if stage == "clip-selection":
        selected = _clip_or_all(clips, clip_id)
        return {
            "stage": stage,
            "artifacts": {
                "transcript": str(paths["transcript"]),
                "clips": str(paths["clips"]),
                "clip_selection_raw": str(paths["clip_selection_raw"]),
            },
            "summary": {
                "transcript_sha256": transcript_fp,
                "clip_count": len(clips),
            },
            "clips": [json.loads(clip.model_dump_json()) for clip in selected],
            "transcript_excerpt": (
                _clip_excerpt(transcript, selected[0]) if clip_id and selected else None
            ),
        }

    clips_before_hooks = [clip.model_copy() for clip in clips]
    hooks_json = _read_json(paths["hooks"], label="hooks.json")
    clips_after_hooks = _apply_hook_artifact(work_dir, clips)

    if stage == "hook-detection":
        selected_before = _clip_or_all(clips_before_hooks, clip_id)
        selected_after = _clip_or_all(clips_after_hooks, clip_id)
        hooks = hooks_json.get("hooks", [])
        if clip_id is not None:
            hooks = [item for item in hooks if str(item.get("clip_id")) == clip_id]
        return {
            "stage": stage,
            "artifacts": {
                "clips": str(paths["clips"]),
                "hooks": str(paths["hooks"]),
                "hooks_raw": str(paths["hooks_raw"]),
            },
            "input_clips": [json.loads(clip.model_dump_json()) for clip in selected_before],
            "output_clips": [json.loads(clip.model_dump_json()) for clip in selected_after],
            "hooks": hooks,
        }

    prune_json = _read_json(paths["prune"], label="prune.json")
    clips_after_prune = _apply_prune_artifact(work_dir, clips_after_hooks)

    if stage == "content-pruning":
        selected_before = _clip_or_all(clips_after_hooks, clip_id)
        selected_after = _clip_or_all(clips_after_prune, clip_id)
        prune_clips = prune_json.get("clips", [])
        if clip_id is not None:
            prune_clips = [item for item in prune_clips if str(item.get("clip_id")) == clip_id]
        return {
            "stage": stage,
            "artifacts": {
                "source_audio": str(paths["source_audio"]),
                "hooks": str(paths["hooks"]),
                "prune": str(paths["prune"]),
                "prune_raw": str(paths["prune_raw"]),
            },
            "input_clips": [json.loads(clip.model_dump_json()) for clip in selected_before],
            "output_clips": [json.loads(clip.model_dump_json()) for clip in selected_after],
            "prune": prune_clips,
        }

    layout_json = _read_json(paths["layout_vision"], label="layout_vision.json")
    layout_payload = layout_json.get("clips", {})
    if clip_id is not None:
        layout_payload = {clip_id: layout_payload.get(clip_id)}
    if stage == "layout-vision":
        return {
            "stage": stage,
            "artifacts": {
                "layout_vision": str(paths["layout_vision"]),
            },
            "clips": [json.loads(clip.model_dump_json()) for clip in _clip_or_all(clips_after_prune, clip_id)],
            "layout_vision": layout_payload,
        }

    render_clips = _clip_or_all(clips_after_prune, clip_id)
    layout_instructions = _load_layout_instructions(work_dir)
    subtitle_payload: list[dict[str, Any]] = []
    for clip in render_clips:
        instr = layout_instructions.get(clip.clip_id)
        if instr is None:
            hint = clip.layout_hint or LayoutKind.SIT_CENTER
            instr = LayoutInstruction(clip_id=clip.clip_id, layout=hint)
        clip.layout = instr.layout
        aligned = clip_subtitle_words(transcript, clip)
        lines = clip_words_to_srt_lines(
            aligned.words,
            max_words_per_cue=config.subtitle_max_words_per_cue,
            max_cue_sec=config.subtitle_max_cue_sec,
        )
        preview_req = RenderRequest(
            source_path=str(paths["source_video"]),
            clip=clip_for_render(clip),
            layout=instr,
            output_path=str((work_dir / "output_preview" / f"short_{clip.clip_id}.mp4")),
            subtitle_path=str(paths["subtitles_dir"] / f"clip_{clip.clip_id}.ass"),
            subtitle_font_size=config.subtitle_font_size,
            subtitle_margin_v=config.subtitle_margin_v,
            title_text=clip.suggested_overlay_title,
            mode="dry_run",
        )
        subtitle_payload.append(
            {
                "clip": json.loads(clip.model_dump_json()),
                "layout_instruction": json.loads(instr.model_dump_json()),
                "source_keep_ranges_sec": source_keep_ranges(clip),
                "subtitle_words": [json.loads(word.model_dump_json()) for word in aligned.words],
                "subtitle_lines": [
                    {"start": start, "end": end, "text": text} for start, end, text in lines
                ],
                "ffmpeg_cmd": build_ffmpeg_cmd(preview_req),
            }
        )
    return {
        "stage": stage,
        "artifacts": {
            "subtitles_dir": str(paths["subtitles_dir"]),
            "layout_vision": str(paths["layout_vision"]),
        },
        "render": subtitle_payload,
    }


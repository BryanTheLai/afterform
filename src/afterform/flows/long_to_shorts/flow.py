"""End-to-end product pipeline with explicit stage controls."""

from __future__ import annotations

import json
import logging
from pathlib import Path

from afterform.schemas import LayoutInstruction, LayoutKind

from afterform.flows.long_to_shorts.clip_selection_cache import (
    cache_valid,
    load_meta,
    load_raw_response,
    should_rerank,
    transcript_fingerprint,
    write_artifacts,
)
from afterform.flows.long_to_shorts.select_clips import (
    filter_clip_candidates,
    load_candidate_pool_from_raw_response,
    load_clips,
    save_dedupe_report,
    save_clips,
    select_clips,
)
from afterform.config import PipelineConfig
from afterform.flows.long_to_shorts.prune_content import run_content_pruning_stage
from afterform.flows.long_to_shorts.subtitles import generate_ass
from afterform.flows.long_to_shorts.detect_hooks import run_hook_detection_stage
from afterform.flows.long_to_shorts.ingest import download_video, extract_audio, transcribe_whisperx
from afterform.flows.long_to_shorts.plan_layouts import run_layout_vision_stage
from afterform.flows.long_to_shorts.stage_inspection import (
    PipelineState,
    build_stage_inspection,
    load_state_before_stage,
    normalize_stage,
    stage_range,
    write_inspection,
)
from afterform.flows.long_to_shorts.render_window import apply_visual_drop_ranges, clip_for_render
from afterform.flows.long_to_shorts.render import reframe_clip_ffmpeg, reframe_clip_timeline_ffmpeg
from afterform.flows.long_to_shorts.run_ledger import RunLedger
from afterform.flows.long_to_shorts.video_cache import (
    extract_youtube_video_id,
    ingest_complete,
    read_youtube_info_json,
    resolve_work_directory,
    upsert_manifest_from_info,
)

logger = logging.getLogger(__name__)


def _ensure_work_dir(config: PipelineConfig) -> None:
    """Resolve ``config.work_dir`` when unset (per-video cache) or ensure it exists."""
    if config.work_dir is not None:
        return
    if not config.youtube_url:
        raise RuntimeError("--work-dir is required when no URL is provided.")
    config.work_dir = resolve_work_directory(
        youtube_url=config.youtube_url,
        explicit_work_dir=None,
        use_video_cache=config.use_video_cache,
        cache_root=config.cache_root,
    )


def _write_stage_inspection_if_requested(
    config: PipelineConfig,
    *,
    stage: str,
) -> None:
    inspect_stage = normalize_stage(config.inspect_stage)
    if inspect_stage != stage:
        return
    assert config.work_dir is not None
    payload = build_stage_inspection(
        config.work_dir,
        stage=inspect_stage,
        clip_id=config.clip_id,
        config=config,
    )
    path = write_inspection(
        config.work_dir,
        stage=inspect_stage,
        payload=payload,
        clip_id=config.clip_id,
    )
    logger.info("Wrote %s inspection: %s", inspect_stage, path)


def _run_ingest_stage(config: PipelineConfig, state: PipelineState) -> PipelineState:
    if not config.youtube_url:
        raise RuntimeError("Stage 'ingest' requires a source URL.")

    logger.info("--- STAGE 1: INGESTION ---")

    source_video = config.work_dir / "source.mp4"
    transcript_path = config.work_dir / "transcript.json"

    if ingest_complete(config.work_dir):
        logger.info("Cached ingest found for this URL (reusing source + transcript).")
    elif source_video.exists():
        logger.info("Source video already downloaded, skipping download.")
    else:
        source_video = download_video(config.youtube_url, config.work_dir)

    if transcript_path.exists():
        logger.info("Transcript already exists, loading.")
        with open(transcript_path, "r", encoding="utf-8") as f:
            transcript = json.load(f)
    else:
        audio_path = extract_audio(source_video, config.work_dir)
        transcript = transcribe_whisperx(audio_path, config.work_dir)

    vid = extract_youtube_video_id(config.youtube_url)
    info = read_youtube_info_json(config.work_dir)
    if not info and vid:
        info = {"id": vid, "webpage_url": config.youtube_url}
    if info:
        upsert_manifest_from_info(
            work_dir=config.work_dir,
            youtube_url=config.youtube_url,
            info=info,
            cache_root=config.cache_root,
        )

    state.source_video = source_video
    state.source_audio = config.work_dir / "source_audio.wav"
    state.transcript = transcript
    state.transcript_fp = transcript_fingerprint(transcript)
    return state


def _run_clip_selection_stage(config: PipelineConfig, state: PipelineState) -> PipelineState:
    logger.info("--- STAGE 2: CLIP SELECTION ---")
    assert state.transcript is not None
    assert state.transcript_fp is not None

    clips_path = config.work_dir / "clips.json"
    dedupe_report_path = config.work_dir / "clip_selection_dedupe.json"
    fp = state.transcript_fp
    meta = load_meta(config.work_dir)

    cache_hit = (
        clips_path.is_file()
        and not config.force_clip_selection
        and meta is not None
        and cache_valid(meta, fp, config)
    )

    rerank_hit = (
        clips_path.is_file()
        and not config.force_clip_selection
        and meta is not None
        and should_rerank(meta, fp, config)
    )

    if cache_hit:
        clips = load_clips(clips_path)
        logger.info(
            "Clip selection cache hit (transcript + provider/model + ranking policy unchanged); "
            "skipping stage LLM."
        )
    elif rerank_hit:
        raw = load_raw_response(config.work_dir)
        if raw is None:
            raise RuntimeError(
                "clips.meta.json says re-rank is possible, but clip_selection_raw.json is missing."
            )
        candidates = load_candidate_pool_from_raw_response(raw, transcript=state.transcript)
        clips, decisions = filter_clip_candidates(
            candidates,
            threshold=config.clip_selection_quality_threshold,
            min_kept=config.clip_selection_min_kept,
            max_kept=config.clip_selection_max_kept,
            mode=config.clip_selection_mode,
        )
        save_dedupe_report(
            dedupe_report_path,
            decisions=decisions,
            candidate_count=len(candidates),
            kept_count=len(clips),
            mode=config.clip_selection_mode,
        )
        save_clips(clips, clips_path)
        write_artifacts(
            config.work_dir,
            transcript=state.transcript,
            config=config,
            raw_response=raw,
        )
        logger.info("Clip selection cache re-rank hit (reused raw LLM pool, no new LLM call).")
    else:
        clips, raw = select_clips(
            state.transcript,
            config=config,
            gemini_model=config.gemini_model,
            candidate_count=config.clip_selection_candidate_count,
            quality_threshold=config.clip_selection_quality_threshold,
            min_kept=config.clip_selection_min_kept,
            max_kept=config.clip_selection_max_kept,
            selection_mode=config.clip_selection_mode,
            dedupe_report_path=dedupe_report_path,
        )
        save_clips(clips, clips_path)
        write_artifacts(
            config.work_dir,
            transcript=state.transcript,
            config=config,
            raw_response=raw,
        )

    state.clips = clips
    logger.info("Selected %d clips:", len(clips))
    for clip in clips:
        logger.info(
            "  [%s] %.1fs-%.1fs (%.1fs) score=%.2f - %s",
            clip.clip_id,
            clip.start_time_sec,
            clip.end_time_sec,
            clip.duration_sec,
            clip.virality_score,
            clip.topic,
        )
    return state


def _run_hook_stage(config: PipelineConfig, state: PipelineState) -> PipelineState:
    logger.info("--- STAGE 2.25: HOOK DETECTION (enabled=%s) ---", config.detect_hooks)
    assert state.clips is not None
    assert state.transcript is not None
    assert state.transcript_fp is not None
    state.clips = run_hook_detection_stage(
        config.work_dir,
        state.clips,
        state.transcript,
        transcript_fp=state.transcript_fp,
        config=config,
    )
    return state


def _run_pruning_stage(config: PipelineConfig, state: PipelineState) -> PipelineState:
    logger.info("--- STAGE 2.5: CONTENT PRUNING (level=%s) ---", config.prune_level)
    assert state.clips is not None
    assert state.transcript is not None
    assert state.transcript_fp is not None
    state.clips = run_content_pruning_stage(
        config.work_dir,
        state.clips,
        state.transcript,
        transcript_fp=state.transcript_fp,
        config=config,
    )
    return state


def _run_layout_stage(config: PipelineConfig, state: PipelineState) -> PipelineState:
    logger.info("--- STAGE 3: CLIP LAYOUTS ---")
    assert state.clips is not None
    assert state.source_video is not None
    assert state.transcript_fp is not None
    state.layout_instructions = run_layout_vision_stage(
        config.work_dir,
        source_video=state.source_video,
        clips=state.clips,
        transcript_fp=state.transcript_fp,
        config=config,
    )
    return state


def _run_render_stage(config: PipelineConfig, state: PipelineState) -> list[Path]:
    logger.info("--- STAGE 4: RENDER ---")
    assert state.clips is not None
    assert state.source_video is not None

    final_outputs: list[Path] = []
    subtitles_dir = config.work_dir / "subtitles"
    subtitles_dir.mkdir(parents=True, exist_ok=True)
    layout_instructions = state.layout_instructions or {}

    for clip in state.clips:
        layout_plan = layout_instructions.get(clip.clip_id)
        if layout_plan is None:
            hint = clip.layout_hint or LayoutKind.SIT_CENTER
            instr = LayoutInstruction(clip_id=clip.clip_id, layout=hint)
        else:
            instr = layout_plan.instruction
        clip.layout = instr.layout
        if layout_plan is not None and layout_plan.visual_drop_ranges_sec:
            clip = apply_visual_drop_ranges(clip, layout_plan.visual_drop_ranges_sec)
        rclip = clip_for_render(clip)
        subtitle_path = generate_ass(
            rclip,
            state.transcript,
            subtitles_dir,
            max_words_per_cue=config.subtitle_max_words_per_cue,
            max_cue_sec=config.subtitle_max_cue_sec,
            play_res_x=1080,
            play_res_y=1920,
            font_size=config.subtitle_font_size,
            margin_v=config.subtitle_margin_v,
        )
        final_path = config.output_dir / f"short_{clip.clip_id}.mp4"
        if final_path.exists() and not config.overwrite_outputs:
            logger.info("Clip %s already rendered, skipping.", clip.clip_id)
            final_outputs.append(final_path)
            continue
        if final_path.exists() and config.overwrite_outputs:
            logger.info("Clip %s exists; overwriting due to clean-run settings.", clip.clip_id)

        if layout_plan is not None and len(layout_plan.layout_timeline) > 1:
            reframe_clip_timeline_ffmpeg(
                input_path=state.source_video,
                output_path=final_path,
                clip=rclip,
                layout_plan=layout_plan,
                subtitle_path=subtitle_path,
                subtitle_font_size=config.subtitle_font_size,
                subtitle_margin_v=config.subtitle_margin_v,
                title_text=clip.suggested_overlay_title,
            )
        else:
            reframe_clip_ffmpeg(
                input_path=state.source_video,
                output_path=final_path,
                clip=rclip,
                layout_instruction=instr,
                subtitle_path=subtitle_path,
                subtitle_font_size=config.subtitle_font_size,
                subtitle_margin_v=config.subtitle_margin_v,
                title_text=clip.suggested_overlay_title,
            )
        final_outputs.append(final_path)

    return final_outputs


def run_pipeline(config: PipelineConfig) -> list[Path]:
    """Execute the pipeline or a controlled stage slice."""
    _ensure_work_dir(config)
    assert config.work_dir is not None

    start_stage, stop_stage = stage_range(
        start_at=normalize_stage(config.start_at),
        stop_after=normalize_stage(config.stop_after),
    )

    logger.info("=" * 60)
    logger.info("AFTERFORM PIPELINE START")
    logger.info("URL: %s", config.youtube_url or "(artifact-only run)")
    logger.info("Output: %s", config.output_dir)
    logger.info("Work dir: %s", config.work_dir)
    logger.info("Stage window: %s -> %s", start_stage, stop_stage)
    logger.info("=" * 60)

    ledger = RunLedger(config, start_stage=start_stage, stop_stage=stop_stage)

    state = (
        PipelineState(work_dir=config.work_dir)
        if start_stage == "ingest"
        else load_state_before_stage(config.work_dir, stage=start_stage, config=config)
    )

    final_outputs: list[Path] = []
    try:
        if start_stage == "ingest":
            ledger.start_stage("ingest")
            try:
                state = _run_ingest_stage(config, state)
            except Exception as exc:
                ledger.fail_stage("ingest", exc)
                raise
            ledger.finish_stage("ingest")
            _write_stage_inspection_if_requested(config, stage="ingest")
            if stop_stage == "ingest":
                ledger.finish_run(final_outputs)
                return final_outputs

        if start_stage in {"ingest", "clip-selection"}:
            ledger.start_stage("clip-selection")
            try:
                state = _run_clip_selection_stage(config, state)
            except Exception as exc:
                ledger.fail_stage("clip-selection", exc)
                raise
            ledger.finish_stage("clip-selection")
            _write_stage_inspection_if_requested(config, stage="clip-selection")
            if stop_stage == "clip-selection":
                ledger.finish_run(final_outputs)
                return final_outputs

        if start_stage in {"ingest", "clip-selection", "hook-detection"}:
            ledger.start_stage("hook-detection")
            try:
                state = _run_hook_stage(config, state)
            except Exception as exc:
                ledger.fail_stage("hook-detection", exc)
                raise
            ledger.finish_stage("hook-detection")
            _write_stage_inspection_if_requested(config, stage="hook-detection")
            if stop_stage == "hook-detection":
                ledger.finish_run(final_outputs)
                return final_outputs

        if start_stage in {"ingest", "clip-selection", "hook-detection", "content-pruning"}:
            ledger.start_stage("content-pruning")
            try:
                state = _run_pruning_stage(config, state)
            except Exception as exc:
                ledger.fail_stage("content-pruning", exc)
                raise
            ledger.finish_stage("content-pruning")
            _write_stage_inspection_if_requested(config, stage="content-pruning")
            if stop_stage == "content-pruning":
                ledger.finish_run(final_outputs)
                return final_outputs

        if start_stage in {
            "ingest",
            "clip-selection",
            "hook-detection",
            "content-pruning",
            "layout-vision",
        }:
            ledger.start_stage("layout-vision")
            try:
                state = _run_layout_stage(config, state)
            except Exception as exc:
                ledger.fail_stage("layout-vision", exc)
                raise
            ledger.finish_stage("layout-vision")
            _write_stage_inspection_if_requested(config, stage="layout-vision")
            if stop_stage == "layout-vision":
                ledger.finish_run(final_outputs)
                return final_outputs

        if start_stage in {
            "ingest",
            "clip-selection",
            "hook-detection",
            "content-pruning",
            "layout-vision",
            "render",
        }:
            ledger.start_stage("render")
            try:
                final_outputs = _run_render_stage(config, state)
            except Exception as exc:
                ledger.fail_stage("render", exc)
                raise
            ledger.finish_stage("render", extra_artifacts=[str(p) for p in final_outputs])
            _write_stage_inspection_if_requested(config, stage="render")
    except Exception:
        raise

    logger.info("=" * 60)
    logger.info("PIPELINE COMPLETE - %d shorts generated:", len(final_outputs))
    for p in final_outputs:
        logger.info("  -> %s", p)
    logger.info("=" * 60)
    ledger.finish_run(final_outputs)
    return final_outputs


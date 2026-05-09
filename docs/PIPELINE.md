# Pipeline

This document describes the current production flow:

`afterform.flows.long_to_shorts.flow.run_pipeline`

## High-level flow

```text
YouTube URL
  -> Stage 1    ingest
  -> Stage 2    clip selection
  -> Stage 2.25 hook detection
  -> Stage 2.5  content pruning
  -> Stage 3    layout planning
  -> Stage 4    render
```

Default work directory:

- `<AFTERFORM_CACHE_ROOT>/videos/<video_id>/`
- or `./.afterform_work` when `--no-video-cache` is set
- or `--work-dir` when provided

## Stage 1: Ingest

Goal:

- `source.mp4`
- `transcript.json`
- optional `source.info.json`

Main modules:

- `afterform.flows.long_to_shorts.video_cache.resolve_work_directory`
- `afterform.flows.long_to_shorts.ingest.download_video`
- `afterform.flows.long_to_shorts.ingest.extract_audio`
- `afterform.flows.long_to_shorts.ingest.transcribe_whisperx`

Artifacts:

- `source.mp4`
- `source_audio.wav`
- `transcript.json`
- `source.info.json`
- `video_cache_manifest.json` under the cache root

## Stage 2: Clip selection

Goal:

- `clips.json`
- `clip_selection_raw.json`
- `clips.meta.json`

Main modules:

- `afterform.flows.long_to_shorts.select_clips`
- `afterform.flows.long_to_shorts.clip_selection_cache`
- `afterform.providers.llm`

Behavior:

- provider-swappable structured LLM call
- over-generate a candidate pool
- rank locally
- keep a thresholded set with a floor and cap
- optional exhaustive mode keeps every distinct quality-clearing candidate
  without an arbitrary final cap
- exhaustive mode writes `clip_selection_dedupe.json` so duplicate drops are
  inspectable

## Stage 2.25: Hook detection

Goal:

- localize a real hook window per clip

Artifacts:

- `hooks.json`
- `hooks_raw.json`
- `hooks.meta.json`

Main module:

- `afterform.flows.long_to_shorts.detect_hooks`

Behavior:

- batched structured LLM call
- validates returned hook windows
- non-fatal on failure

## Stage 2.5: Content pruning

Goal:

- tighten each clip by trimming weak lead-in and tail content

Artifacts:

- `prune.json`
- `prune_raw.json`
- `prune.meta.json`

Main module:

- `afterform.flows.long_to_shorts.prune_content`

Behavior:

- structured LLM decides outer trims
- audio-first keep-ranges refine the interior
- clamps enforce min duration and hook protection
- non-fatal on failure

## Stage 3: Layout planning

Goal:

- one `LayoutInstruction` per clip

Artifacts:

- `layout_vision.json`
- `layout_vision.meta.json`
- sampled keyframes under `keyframes/<clip_id>/`

Main module:

- `afterform.flows.long_to_shorts.plan_layouts`

Behavior:

- sample multiple frames across the kept render window
- call a multimodal LLM with structured output
- normalize bbox output into the shared schema
- fall back to `layout_hint` if sampling or the model fails

## Stage 4: Render

Goal:

- `output/short_<clip_id>.mp4`

Main modules:

- `afterform.flows.long_to_shorts.render_window`
- `afterform.flows.long_to_shorts.transcript_align`
- `afterform.flows.long_to_shorts.render`
- `afterform.primitives.compile`
- `afterform.primitives.layouts`

Behavior:

- convert clip timing metadata into honest source ranges
- generate ASS subtitles
- build the render request
- render with ffmpeg

## Shared contracts

The shared runtime schema lives in:

- `afterform.schemas`

The reusable deterministic building blocks live in:

- `afterform.primitives`

## Cache invalidation summary

| Change | Clip selection | Hook detection | Content pruning | Layout planning |
|--------|----------------|----------------|-----------------|-----------------|
| Transcript changes | miss | miss | miss | miss |
| Clip window changes | n/a | miss | miss | miss |
| Text model changes | miss | miss | miss | maybe |
| Vision model changes | no effect | no effect | no effect | miss |
| `--force-*` flag | bypasses that stage cache | bypasses that stage cache | bypasses that stage cache | bypasses that stage cache |

## CLI mapping

| Flag | Config field |
|------|--------------|
| `--llm-provider` | `PipelineConfig.llm_provider` |
| `--llm-model` | `PipelineConfig.llm_model` |
| `--llm-vision-model` | `PipelineConfig.llm_vision_model` |
| `--clip-mode` | `clip_selection_mode` |
| `--clip-candidate-count` | `clip_selection_candidate_count` |
| `--clip-quality-threshold` | `clip_selection_quality_threshold` |
| `--max-clips` | `clip_selection_max_kept` |
| `--review-only-clips` | `stop_after="clip-selection"` |
| `--force-clip-selection` | `force_clip_selection` |
| `--force-hook-detection` | `force_hook_detection` |
| `--force-content-pruning` | `force_content_pruning` |
| `--force-layout-vision` | `force_layout_vision` |
| `--no-hook-detection` | `detect_hooks=False` |
| `--prune-level` | `prune_level` |
| `--work-dir` | `work_dir` |
| `--cache-root` | `cache_root` |
| `--start-at`, `--stop-after` | stage window |
| `--inspect-stage`, `--clip-id` | inspection output |

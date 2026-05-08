# Environment Variables

This is the source of truth for how `afterform` reads configuration.

## Loading

`afterform.config` bootstraps `.env` from the current working directory with `python-dotenv`.

Practical rule:

- run from the repo with `uv run afterform ...`
- keep provider secrets in `.env`

## Stage LLMs

These drive stages 2, 2.25, 2.5, and 3.

| Variable | Default | Meaning |
|----------|---------|---------|
| `AFTERFORM_LLM_PROVIDER` | `gemini` | One of `gemini`, `openai`, `azure`. |
| `AFTERFORM_LLM_MODEL` | *(unset)* | Text-stage model or deployment id. |
| `AFTERFORM_LLM_VISION_MODEL` | *(unset)* | Optional separate model or deployment id for layout planning. Falls back to `AFTERFORM_LLM_MODEL`. |

## Gemini

| Variable | Meaning |
|----------|---------|
| `GOOGLE_API_KEY` | Preferred Gemini API key. |
| `GEMINI_API_KEY` | Fallback Gemini API key if `GOOGLE_API_KEY` is unset. |
| `GEMINI_MODEL` | Fallback Gemini text model when `AFTERFORM_LLM_MODEL` is unset and provider is Gemini. |
| `GEMINI_VISION_MODEL` | Fallback Gemini vision model when `AFTERFORM_LLM_VISION_MODEL` is unset and provider is Gemini. |

## OpenAI / Azure OpenAI

| Variable | Meaning |
|----------|---------|
| `OPENAI_API_KEY` | Required for `AFTERFORM_LLM_PROVIDER=openai`. |
| `OPENAI_BASE_URL` | Optional OpenAI-compatible gateway URL. |
| `AZURE_OPENAI_API_KEY` | Required for `AFTERFORM_LLM_PROVIDER=azure`. |
| `AZURE_OPENAI_BASE_URL` / `AZURE_BASE_URL` | Preferred Azure OpenAI-compatible base URL. |
| `AZURE_OPENAI_ENDPOINT` / `AZURE_ENDPOINT` | Endpoint-style Azure resource URL if no base URL is set. |
| `AZURE_OPENAI_DEPLOYMENT` / `AZURE_DEPLOYMENT` | Optional deployment name for endpoint-style Azure resources. |
| `AZURE_OPENAI_API_VERSION` / `OPENAI_API_VERSION` | Required only for endpoint-style Azure resources. |

## Prompt overrides

Built-in prompts live under `src/afterform/flows/long_to_shorts/prompts/`.

| Variable | Meaning |
|----------|---------|
| `AFTERFORM_PROMPTS_DIR` | Override directory for `.jinja2` prompt files. |

## Transcription

| Variable | Meaning |
|----------|---------|
| `AFTERFORM_TRANSCRIBE_PROVIDER` | `auto`, `openai`, or `whisperx`. |

## Cache

| Variable | Meaning |
|----------|---------|
| `AFTERFORM_CACHE_ROOT` | Cache root. Default: `~/.cache/afterform` on Unix, `%LOCALAPPDATA%/afterform` on Windows. |

Cache layout:

- per-video work dir: `<cache_root>/videos/<11-char-video-id>/`
- global manifest: `<cache_root>/video_cache_manifest.json`
- `--no-video-cache`: use `./.afterform_work` unless `--work-dir` is set

## CLI cross-reference

| Flag | Env |
|------|-----|
| `--llm-provider` | `AFTERFORM_LLM_PROVIDER` |
| `--llm-model` | `AFTERFORM_LLM_MODEL` |
| `--llm-vision-model` | `AFTERFORM_LLM_VISION_MODEL` |
| `--clip-mode` | explicit CLI only |
| `--clip-candidate-count` | explicit CLI only |
| `--clip-quality-threshold` | explicit CLI only |
| `--max-clips` | explicit CLI only |
| `--review-only-clips` | explicit CLI only |
| `--cache-root` | `AFTERFORM_CACHE_ROOT` |
| `--work-dir` | explicit intermediate-artifact directory |
| `--run-dir` | expands to `<run-dir>/work` and `<run-dir>/output` |
| `--no-video-cache` | disables per-video cache dirs |
| `--force-clip-selection` | bypasses clip-selection cache |
| `--force-hook-detection` | bypasses hook-detection cache |
| `--force-content-pruning` | bypasses pruning cache |
| `--filled-pause-pruning` | explicit enable; default behavior is already on |
| `--no-filled-pause-pruning` | disable filled-pause pruning for one run |
| `--require-filled-pause-pruning` | fail if the filled-pause runtime cannot run |
| `--force-layout-vision` | bypasses layout-vision cache |
| `--clean-run` | forces a fresh work dir and no cache reuse |

## Filled-pause pruning runtime

Filled-pause pruning is on by default for CLI runs.

Requirements:

- install the optional runtime with `uv sync --extra filled-pause`
- first use downloads `classla/wav2vecbert2-filledPause`

Behavior:

- if the runtime is installed and the model is available, Stage 2.5 removes short `um/uh/hmm` spans from `source_audio.wav`
- if the runtime is missing, the stage degrades to a no-op unless `--require-filled-pause-pruning` is set
- use `--no-filled-pause-pruning` when you want exact source audio preserved

## Recommended run layout

If you care about preserving partial progress and keeping outputs separated, do not
reuse the default `./output` folder across unrelated runs.

Preferred pattern:

- one run folder per attempt
- `--run-dir` for the parent folder
- `gpt-5.4` as the default Azure text and vision model unless there is a measured
  reason to pay for `gpt-5.5`

Example:

```bash
uv run afterform run long-to-shorts "<youtube_url>" \
  --run-dir ".afterform_runs/yt_20260508_093000" \
  --llm-provider azure \
  --llm-model gpt-5.4 \
  --llm-vision-model gpt-5.4
```

This expands to:

- work dir: `.afterform_runs/yt_20260508_093000/work`
- output dir: `.afterform_runs/yt_20260508_093000/output`

Resume pattern:

```bash
uv run afterform run long-to-shorts \
  --work-dir ".afterform_runs/yt_20260508_093000/work" \
  --output ".afterform_runs/yt_20260508_093000/output" \
  --start-at layout-vision \
  --llm-provider azure \
  --llm-model gpt-5.4 \
  --llm-vision-model gpt-5.4
```

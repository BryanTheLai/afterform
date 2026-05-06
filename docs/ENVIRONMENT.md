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
| `--no-video-cache` | disables per-video cache dirs |
| `--force-clip-selection` | bypasses clip-selection cache |
| `--force-hook-detection` | bypasses hook-detection cache |
| `--force-content-pruning` | bypasses pruning cache |
| `--force-layout-vision` | bypasses layout-vision cache |
| `--clean-run` | forces a fresh work dir and no cache reuse |

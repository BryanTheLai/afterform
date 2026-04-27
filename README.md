# afterform

Turn long-form video into short-form video.

Today, `afterform` ships one flow:

- `long-to-shorts`

The package model is simple:

- `afterform`
- `afterform.primitives`
- `afterform.flows.long_to_shorts`

## Demo

Original  
https://m.youtube.com/watch?v=PdVv_vLkUgk

After  
https://m.youtube.com/shorts/qxZVuwb6YPw

## What It Does

Given a YouTube video, `afterform` will:

1. ingest the source
2. transcribe it
3. select short-worthy clips
4. detect hooks
5. prune weak internal sections
6. plan layouts
7. render vertical shorts

## Install

Requirements:

- Python 3.10+
- `uv`
- `ffmpeg` on `PATH`

Setup:

```bash
uv venv
uv sync
```

Optional extras:

```bash
uv sync --extra dev
uv sync --extra whisper
uv sync --extra face
```

## Run

```bash
uv run afterform run long-to-shorts "https://www.youtube.com/watch?v=VIDEO_ID"
```

Useful variants:

```bash
uv run afterform run long-to-shorts "https://www.youtube.com/watch?v=VIDEO_ID" --clean-run --verbose
uv run afterform run long-to-shorts "https://www.youtube.com/watch?v=VIDEO_ID" --llm-provider azure --llm-model gpt-5.4 --llm-vision-model gpt-5.4
uv run afterform --help
```

## Current Scope

- one package
- one primitives namespace
- one flows namespace
- one production flow

More flows can be added later without changing the package model.

## Repo Layout

| Path | Purpose |
|------|---------|
| `src/afterform/` | installable package |
| `src/afterform/primitives/` | reusable building blocks |
| `src/afterform/flows/long_to_shorts/` | current production flow |
| `tests/` | test suite |
| `docs/` | design, pipeline, environment, and architecture docs |

## Docs

- [`docs/README.md`](docs/README.md)
- [`docs/PIPELINE.md`](docs/PIPELINE.md)
- [`docs/ENVIRONMENT.md`](docs/ENVIRONMENT.md)
- [`docs/mcp_architecture.md`](docs/mcp_architecture.md)

## Verify

```bash
uv run pytest
python -m build
```

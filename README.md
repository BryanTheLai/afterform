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
uv sync --extra filled-pause
```

Filled-pause pruning runtime:

- default runs now try to remove short `um/uh/hmm` spans during Stage 2.5
- install the runtime with `uv sync --extra filled-pause`
- first use downloads the Hugging Face model `classla/wav2vecbert2-filledPause`
- if the runtime is unavailable, Afterform degrades safely and keeps the original audio
- use `--require-filled-pause-pruning` to fail instead of silently skipping

## Run

```bash
uv run afterform run long-to-shorts "https://www.youtube.com/watch?v=VIDEO_ID"
```

Recommended pattern for real work:

- use one isolated run folder per attempt
- use `--run-dir` so intermediates and finals do not collide
- default Azure recommendation: `gpt-5.4`, not `gpt-5.5`

```bash
uv run afterform run long-to-shorts "https://www.youtube.com/watch?v=4SlNgM4PjvQ" \
  --run-dir ".afterform/runs/yt_20260508_093000" \
  --llm-provider azure \
  --llm-model gpt-5.4 \
  --llm-vision-model gpt-5.4 \
  --verbose
```

```powershell
uv run afterform run long-to-shorts "https://www.youtube.com/watch?v=4SlNgM4PjvQ" --run-dir ".afterform/runs/waterloo-sam-altman" --llm-provider azure --llm-model gpt-5.4 --llm-vision-model gpt-5.4 --verbose

```

What `--run-dir` does:

- writes intermediates to `<run-dir>/work`
- writes final shorts to `<run-dir>/output`
- writes run tracking to `<run-dir>/run.json` and `<run-dir>/config.json`

Why this matters:

- default URL runs now create `.afterform/runs/<run_id>/`
- `--clean-run` enables output overwrite behavior
- reusing one folder makes it easy to lose track of partial progress

Resume an interrupted run from cached artifacts:

```bash
uv run afterform run long-to-shorts \
  --work-dir ".afterform/runs/yt_20260508_093000/work" \
  --output ".afterform/runs/yt_20260508_093000/output" \
  --start-at layout-vision \
  --llm-provider azure \
  --llm-model gpt-5.4 \
  --llm-vision-model gpt-5.4 \
  --verbose
```

Useful variants:

```bash
uv run afterform run long-to-shorts "https://www.youtube.com/watch?v=VIDEO_ID" --clean-run --verbose
uv run afterform run long-to-shorts "https://www.youtube.com/watch?v=VIDEO_ID" --run-dir ".afterform/runs/yt_20260508_093000" --llm-provider azure --llm-model gpt-5.4 --llm-vision-model gpt-5.4
uv run afterform run long-to-shorts "https://www.youtube.com/watch?v=VIDEO_ID" --no-filled-pause-pruning
uv run afterform run long-to-shorts "https://www.youtube.com/watch?v=VIDEO_ID" --require-filled-pause-pruning
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
| `docs/` | public setup and usage docs |

## Docs

- [`docs/README.md`](docs/README.md)
- [`docs/PIPELINE.md`](docs/PIPELINE.md)
- [`docs/ENVIRONMENT.md`](docs/ENVIRONMENT.md)
- [`docs/long-to-shorts.md`](docs/long-to-shorts.md)
- [`docs/SHARING.md`](docs/SHARING.md)
- [`TERMINOLOGY.md`](TERMINOLOGY.md)

## Verify

```bash
uv run pytest
python -m build
```

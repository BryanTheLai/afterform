# Sharing this project with someone else

Large binaries do not belong in git. `.gitignore` excludes **`output/`**, **`.afterform/`**, **`*.mp4`**, **`*.wav`**, and runtime work dirs. That is intentional.

## What lives in the repo

- Markdown in `docs/`
- code in `src/afterform/`
- tests in `tests/`

## Easiest ways to show work

1. **Public GitHub repo** - share the source, docs, and tests.
2. **Rendered docs on GitHub** - point people at `README.md`, `docs/PIPELINE.md`, and `docs/long-to-shorts.md`.
3. **Raw file links** - useful for logs or machine-readable artifacts you decide to commit later.
4. **YouTube or cloud storage** - share rendered shorts without bloating the repo.

## What you cannot share via git alone

- Final **`short_*.mp4`** files unless you remove ignore rules or use **GitHub Releases** or external storage.
- Local run records under **`.afterform/runs/`** unless you copy the specific `run.json` or `config.json` into a tracked note.

Use YouTube or Releases for MP4s; keep git for source and documentation.

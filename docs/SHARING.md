# Sharing this project with someone else

Large binaries do not belong in git. `.gitignore` excludes **`output/`**, **`*.mp4`**, **`*.wav`**, and runtime work dirs. That is intentional.

## What lives in the repo

- Markdown in `docs/`
- code in `src/contentflow/`
- tests in `tests/`

## Easiest ways to show work

1. **Public GitHub repo** - share the source, docs, and tests.
2. **Rendered docs on GitHub** - point people at `README.md`, `docs/PIPELINE.md`, and `docs/TARGET_VIDEO_ANALYSIS.md`.
3. **Raw file links** - useful for logs or machine-readable artifacts you decide to commit later.
4. **YouTube or cloud storage** - share rendered shorts without bloating the repo.

## What you cannot share via git alone

- Final **`short_*.mp4`** files unless you remove ignore rules or use **GitHub Releases** or external storage.

Use YouTube or Releases for MP4s; keep git for source and documentation.

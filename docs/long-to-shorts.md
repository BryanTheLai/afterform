---
title: Long-to-Shorts Flow
type: mvp-spec
status: draft
---

# Long-to-shorts

**Goal:** Turn one long-form source video into several **9:16** MP4 shorts with burned subtitles and layout-aware framing.

**CLI:** `uv run afterform run long-to-shorts "<youtube_url>"` (see repo root [`README.md`](../README.md) for install and flags).

**How it works (one sentence):** Download + transcript -> structured clip selection -> hook detection -> content pruning -> layout planning -> ffmpeg render.

**Canonical detail:** [`docs/PIPELINE.md`](PIPELINE.md) - stages, caches, artifacts, and config mapping.

**Terminology:** [`TERMINOLOGY.md`](../TERMINOLOGY.md) - time window vs crop/layout.

**Current gaps:** [`docs/PROJECT_ISSUES.md`](PROJECT_ISSUES.md).

**Prompt/runtime caveats:** [`docs/KNOWN_LIMITATIONS_AND_PROMPT_CONTRACT_GAP.md`](KNOWN_LIMITATIONS_AND_PROMPT_CONTRACT_GAP.md).

**Why the name is `long-to-shorts`:** this flow is about transforming a long source into multiple short outputs. That is precise today and does not box future flows into podcasts.

**Current package model:** one installable package, reusable primitives under `afterform.primitives`, workflows under `afterform.flows.*`.

**Open product question:** Stage 3 already handles one merged layout per clip; the next meaningful upgrade is a layout timeline for clips whose visual structure changes mid-run.

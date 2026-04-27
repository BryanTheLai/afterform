# Project Issues Overview

This is the consolidated issue inventory for this repo as of **2026-04-27**.

It separates:

- **Confirmed runtime gaps**: current code does not do the desired thing.
- **Open backlog**: desired capabilities that are documented but not built.
- **Doc boundaries**: which documents describe live behavior versus product intent.

Read this after [`PIPELINE.md`](PIPELINE.md) if you want the blunt picture.

## Executive summary

The biggest current product gaps are:

1. **Clip selection is still transcript-only.** Visual context does not influence which windows survive Stage 2.
2. **Stage 3 still emits one merged layout per clip.** Multi-frame sampling exists, but there is no layout timeline when a 60-second clip genuinely changes structure mid-run.
3. **Upload metadata is still dead weight.** `shorts_title`, `description`, and `hashtags` are stored but not consumed by the product pipeline.
4. **Hook windows still do not change the render in-point.** They protect pruning, not final clip start.
5. **There is still no automated canonical-video integration test.** The Cathie Wood regression is still verified manually.

Important shipped changes that are **no longer open issues**:

- Stage 3 samples multiple frames per clip.
- The model-facing bbox contract is `0..1000`, normalized back to internal `[0,1]`.
- Structured LLM calls use `response_schema` at the provider boundary.
- Stage 3 fallback preserves `clip.layout_hint` instead of collapsing chart-heavy clips to `sit_center`.
- Layout cache invalidation keys off clip windows, not arbitrary `clips.json` byte changes.

## A. Confirmed runtime gaps

| ID | Issue | Evidence | Why it matters |
|----|-------|----------|----------------|
| R1 | **Clip selection has no visual context.** | [`src/afterform/flows/long_to_shorts/flow.py`](../src/afterform/flows/long_to_shorts/flow.py) still runs Stage 2 before Stage 3; there is no `narrative_context.json` pre-selection artifact. | The selector cannot use chart appearances, slide OCR, or scene structure when choosing highlight windows. |
| R2 | **Stage 3 collapses to one clip-level layout.** | [`src/afterform/flows/long_to_shorts/plan_layouts.py`](../src/afterform/flows/long_to_shorts/plan_layouts.py) samples multiple frames and merges them into one `LayoutInstruction`; render consumes one instruction per clip. | If a clip changes from talking head to chart reveal halfway through, the pipeline still has to choose one dominant layout. |
| R3 | **Hook windows do not change export start.** | [`src/afterform/flows/long_to_shorts/render_window.py`](../src/afterform/flows/long_to_shorts/render_window.py) narrows export bounds only with trims; hook fields are not render inputs. | If product intent is "start exactly on the hook sentence", current code does not do that. |
| R4 | **`shorts_title`, `description`, and `hashtags` are persisted but unused.** | The fields exist on [`src/afterform/schemas.py`](../src/afterform/schemas.py), but the render path does not consume them. | Model tokens are spent on metadata that never affects render or export. |
| R5 | **No automated canonical-video regression test.** | The Cathie Wood Azure rerender was verified from real outputs and logs, but no committed end-to-end test exercises that path. | The exact demo source that caught the Stage 3 bug is not yet part of automated regression coverage. |
| R6 | **Stage 3 quality still depends on local frame sampling working.** | [`src/afterform/flows/long_to_shorts/plan_layouts.py`](../src/afterform/flows/long_to_shorts/plan_layouts.py) requires `cv2` to sample frames; when sampling fails it falls back safely to `layout_hint`. | The product no longer breaks catastrophically, but quality still drops if sampling dependencies are missing or the source cannot be read. |

## B. Open backlog that is documented but not built

| ID | Backlog item | Where documented | What is missing |
|----|--------------|------------------|-----------------|
| B1 | **`narrative_context.json` before clip selection** | [`docs/KNOWN_LIMITATIONS_AND_PROMPT_CONTRACT_GAP.md`](KNOWN_LIMITATIONS_AND_PROMPT_CONTRACT_GAP.md) | No Stage 1.5 visual-summary artifact yet; Stage 2 still depends only on transcript plus hashes. |
| B2 | **Clip selector consumes visual context** | [`docs/KNOWN_LIMITATIONS_AND_PROMPT_CONTRACT_GAP.md`](KNOWN_LIMITATIONS_AND_PROMPT_CONTRACT_GAP.md) | No prompt or cache input today changes when the source visuals change but the transcript does not. |
| B3 | **Layout timeline or per-segment layout output** | [`docs/KNOWN_LIMITATIONS_AND_PROMPT_CONTRACT_GAP.md`](KNOWN_LIMITATIONS_AND_PROMPT_CONTRACT_GAP.md) | Multi-frame evidence is merged to one clip-level answer; there is no time-varying layout plan yet. |
| B4 | **Cross-run memory module** | repo discussion only | No memory/state layer beyond one work dir and its caches. |
| B5 | **Automated canonical-video integration fixture** | [`docs/TARGET_VIDEO_ANALYSIS.md`](TARGET_VIDEO_ANALYSIS.md) | No committed regression harness for the real target source. |

## C. Doc boundaries

| ID | Boundary | What to treat as truth |
|----|----------|------------------------|
| D1 | **`PIPELINE.md` is runtime truth.** | Use [`PIPELINE.md`](PIPELINE.md) when you need exact stages, artifacts, and CLI mapping. |
| D2 | **`PROJECT_ISSUES.md` is state assessment, not a spec.** | Use this doc for gaps and backlog, not for exact CLI or cache behavior. |
| D3 | **`KNOWN_LIMITATIONS_AND_PROMPT_CONTRACT_GAP.md` is issue-focused, not a feature overview.** | Use it for prompt/runtime mismatches and fix maps, not onboarding. |

## D. Clear current-state picture

1. **What works today**
   - Ingest, transcript, structured clip selection, hook detection, content pruning, multi-frame layout vision, subtitles, and ffmpeg render.
   - Split layouts preserve chart geometry again after the 2026-04-22 Stage 3 fallback fix.
   - Stage LLMs are provider-swappable across `gemini`, `openai`, and `azure`.
   - There are unit and component tests around ranking, caching, bbox parsing, layout planning, CLI behavior, and rendering primitives.

2. **What is still weak today**
   - Clip selection still cannot see visuals before choosing clips.
   - Layout decisions are still one per clip rather than time-varying.
   - Some LLM-generated metadata fields still never reach a product surface.
   - The canonical demo source is still verified manually rather than by automation.

3. **What to fix first if output quality is the priority**
   - Add pre-selection multimodal context.
   - Replace one merged layout per clip with a layout timeline when a clip has real visual changes.
   - Either consume or remove dead upload metadata fields.
   - Add a real canonical-video regression harness.

## Recommended reading order from here

1. [`PIPELINE.md`](PIPELINE.md) for current runtime behavior.
2. [`KNOWN_LIMITATIONS_AND_PROMPT_CONTRACT_GAP.md`](KNOWN_LIMITATIONS_AND_PROMPT_CONTRACT_GAP.md) for prompt-vs-code mismatches.
3. [`TARGET_VIDEO_ANALYSIS.md`](TARGET_VIDEO_ANALYSIS.md) for the canonical regression source.
4. [`README.md`](../README.md) for install and package layout.

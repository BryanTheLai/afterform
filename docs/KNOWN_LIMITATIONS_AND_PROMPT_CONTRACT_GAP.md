# Known limitations, prompt-code contract gaps, and fix map

This document is the single trail for recurring "what does the prompt imply vs what does Python actually do?" questions.

**Last verified against repo:** 2026-04-27

## 1. Ranking contract: rule-scored ranking is now live

### Verdict

The old `score_breakdown` and `reasoning` mismatch is gone. Stage 2 now asks for:

- `selection_reason`
- `rule_scores`
- `virality_score` consistent with the weighted composite of those rules

The runtime keeps those fields, derives `virality_score` from `rule_scores` when present, and then ranks on that score with a `needs_review` penalty.

### Evidence

- Prompt contract: [`src/afterform/flows/long_to_shorts/prompts/clip_selection_system.jinja2`](../src/afterform/flows/long_to_shorts/prompts/clip_selection_system.jinja2)
- Candidate schema + score folding: [`src/afterform/flows/long_to_shorts/select_clips.py`](../src/afterform/flows/long_to_shorts/select_clips.py)
- Shared persisted fields: [`src/afterform/schemas.py`](../src/afterform/schemas.py)

### Remaining limitation

Clip selection is still **transcript-only**. The rule scores are better structured than before, but they still do not see charts or OCR before Stage 3.

## 2. Hook window vs export start

### Verdict

`hook_start_sec` and `hook_end_sec` are **clip-relative** and used for **pruning clamps** and **hook-detection prompts**, not to shift the ffmpeg `-ss` in-point. The exported slice is `[start_time_sec, end_time_sec]` narrowed only by `trim_start_sec`, `trim_end_sec`, and optional `keep_ranges_sec`.

### Evidence

- Export bounds: [`src/afterform/flows/long_to_shorts/render_window.py`](../src/afterform/flows/long_to_shorts/render_window.py)
- Hook protection inside pruning: [`src/afterform/flows/long_to_shorts/prune_content.py`](../src/afterform/flows/long_to_shorts/prune_content.py)

### Fix map

If product intent becomes "open exactly on the hook", the change belongs in:

- [`src/afterform/flows/long_to_shorts/render_window.py`](../src/afterform/flows/long_to_shorts/render_window.py)
- [`src/afterform/primitives/compile.py`](../src/afterform/primitives/compile.py)
- [`docs/PIPELINE.md`](PIPELINE.md)
- [`TERMINOLOGY.md`](../TERMINOLOGY.md)

## 3. Clip JSON fields persisted but unused in the product path

### Verdict

These fields exist on `Clip` and appear in `clips.json`, but no product runtime stage consumes them for render or export:

- `shorts_title`
- `description`
- `hashtags`

### Evidence

The fields are defined in [`src/afterform/schemas.py`](../src/afterform/schemas.py), but the render pipeline in [`src/afterform/flows/long_to_shorts/render.py`](../src/afterform/flows/long_to_shorts/render.py) does not read them.

### Fix map

| Goal | File |
|------|------|
| Emit upload sidecar JSON | new module, e.g. `src/afterform/flows/long_to_shorts/upload_metadata.py` |
| Burn `shorts_title` instead of overlay title | [`src/afterform/flows/long_to_shorts/render.py`](../src/afterform/flows/long_to_shorts/render.py) |
| Drop the fields from the prompt to save tokens | [`src/afterform/flows/long_to_shorts/prompts/clip_selection_system.jinja2`](../src/afterform/flows/long_to_shorts/prompts/clip_selection_system.jinja2) |

## 4. Stage 3 is now multi-frame, but still clip-level

### Verdict

The old "one midpoint keyframe per clip" description is obsolete. Stage 3 now samples multiple frames per clip, validates a structured multi-frame response, and merges those frame opinions into one clip-level `LayoutInstruction`.

What is still missing is a **layout timeline**. A clip that starts as talking head and ends as chart reveal still gets one dominant layout choice for the full render.

### Evidence

- Multi-frame sampling and merge: [`src/afterform/flows/long_to_shorts/plan_layouts.py`](../src/afterform/flows/long_to_shorts/plan_layouts.py)
- One instruction consumed per clip: [`src/afterform/flows/long_to_shorts/flow.py`](../src/afterform/flows/long_to_shorts/flow.py)

### Fix map

| Approach | Files |
|----------|-------|
| Keep multi-frame sampling but emit a per-segment layout plan | [`src/afterform/flows/long_to_shorts/plan_layouts.py`](../src/afterform/flows/long_to_shorts/plan_layouts.py), [`src/afterform/schemas.py`](../src/afterform/schemas.py) |
| Add a render-time layout timeline instead of one clip instruction | [`src/afterform/flows/long_to_shorts/flow.py`](../src/afterform/flows/long_to_shorts/flow.py), [`src/afterform/primitives/compile.py`](../src/afterform/primitives/compile.py) |
| Add pre-selection multimodal context so layout-heavy segments influence Stage 2 | this document, future `src/afterform/flows/long_to_shorts/narrative_context.py` |

## 5. Stage 3 bbox contract and fallback: current truth

### Verdict

The current Stage 3 contract is:

- **Model-facing bbox format:** integer-like `0..1000`
- **Internal runtime bbox format:** normalized `[0,1]`
- **Defensive parser support:** legacy normalized boxes and accidental pixel boxes still normalize when frame size is known
- **Failure fallback:** preserve `clip.layout_hint` or current layout instead of blindly forcing `sit_center`

This is the behavior that fixed the Cathie Wood split-layout regression on `2026-04-22`.

### Evidence

- Prompt + response schema: [`src/afterform/flows/long_to_shorts/plan_layouts.py`](../src/afterform/flows/long_to_shorts/plan_layouts.py)
- Shared runtime schema: [`src/afterform/schemas.py`](../src/afterform/schemas.py)
- Regression tests: [`tests/test_layout_vision_unit.py`](../tests/test_layout_vision_unit.py)

### Remaining limitation

Stage 3 quality still depends on frame sampling actually succeeding. Missing `cv2` or unreadable source video no longer breaks the render completely, but it still degrades the result to transcript-era `layout_hint` quality.

## 6. `startup/PROMPT.md` is process guidance, not runtime code

`startup/PROMPT.md` lives outside this repo and is not imported by `afterform` at runtime. Use it for human or agent process. Use this file plus [`PIPELINE.md`](PIPELINE.md) for product truth.

## 7. Quick cross-index

| Topic | Canonical doc |
|-------|----------------|
| Stages, caches, artifacts | [`docs/PIPELINE.md`](PIPELINE.md) |
| Current runtime gaps | [`docs/PROJECT_ISSUES.md`](PROJECT_ISSUES.md) |
| Canonical regression source | [`docs/TARGET_VIDEO_ANALYSIS.md`](TARGET_VIDEO_ANALYSIS.md) |
| Temporal vs spatial concepts | [`TERMINOLOGY.md`](../TERMINOLOGY.md) |

## 8. Verification commands

After changing clip-selection or layout contracts:

```bash
uv run pytest tests/test_clip_ranking.py tests/test_clip_selector.py tests/test_clip_selection_cache.py tests/test_layout_vision_unit.py tests/test_cli.py
```

For a live CLI sanity check:

```bash
uv run afterform --help
```

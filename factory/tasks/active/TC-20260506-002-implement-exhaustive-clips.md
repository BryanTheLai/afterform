---
id: TC-20260506-002
version: 1
created_at: 2026-05-06
updated_at: 2026-05-06
status: active
repo: afterform
area: long-to-shorts clip selection
owner: Bryan
created_by: Codex
priority: high
risk: medium
approval: yellow
supersedes: TC-20260506-001
expected_artifact: exhaustive clip-selection runtime
---

# Task Card: Implement Exhaustive Clip Selection Mode

## Goal

Add a runnable `exhaustive` clip-selection mode that keeps every distinct quality-clearing candidate instead of applying an arbitrary final cap.

## Scope

- `src/afterform/config.py`
- `src/afterform/cli.py`
- `src/afterform/flows/long_to_shorts/select_clips.py`
- `src/afterform/flows/long_to_shorts/flow.py`
- `src/afterform/flows/long_to_shorts/clip_selection_cache.py`
- `tests/test_clip_ranking.py`
- `tests/test_clip_selection_cache.py`
- `tests/test_cli.py`
- `docs/PIPELINE.md`
- `docs/ENVIRONMENT.md`

## Non-Goals

- Do not implement chunked multi-call LLM selection yet.
- Do not add new dependencies.
- Do not change render, hook detection, pruning, or layout behavior.
- Do not commit, push, or clean unrelated dirty files.

## Context Links

- `docs/EXHAUSTIVE_DISTINCT_CLIPS_SPEC.md`
- `factory/tasks/active/TC-20260506-001-exhaustive-distinct-clips.md`

## Assumptions To Check

- Assumption: v1 can discover "as many as possible" within the returned candidate pool.
- How to check: implement uncapped exhaustive filtering now; leave chunked generation as the next task.

## Proof

Required proof:

- command: `uv run pytest tests/test_clip_ranking.py tests/test_clip_selection_cache.py tests/test_cli.py tests/test_clip_selector.py`
- expected exit code: `0`
- expected artifact: `clip_selection_dedupe.json` support in code and tests

## Definition Of Done

- [ ] curated mode remains default
- [ ] exhaustive mode supports `max_kept=None`
- [ ] weak backfill is disabled in exhaustive mode
- [ ] deterministic dedupe exists
- [ ] dedupe report can be written
- [ ] cache policy includes mode and dedupe version
- [ ] CLI exposes the mode and review-only path
- [ ] focused tests pass

## Approval Rules

- green: read files, edit scoped source/tests/docs, run focused local tests
- yellow: alter clip-selection behavior behind an explicit mode
- red: delete files, commit, push, PR, live LLM/API spend

## Careful Review

| Action | Color | Why | Bryan Approval Needed? |
|---|---|---|---|
| Add explicit clip mode | yellow | new runtime behavior, default preserved | no |
| Commit/push/PR | red | external state change | yes |
| Clean workspace | red | unrelated dirty files exist | yes |

## Stop Conditions

- stop if tests fail after three root-cause attempts
- stop before commit/push/PR

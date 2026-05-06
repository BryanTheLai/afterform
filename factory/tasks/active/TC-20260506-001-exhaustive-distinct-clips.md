---
id: TC-20260506-001
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
supersedes:
expected_artifact: docs/EXHAUSTIVE_DISTINCT_CLIPS_SPEC.md
---

# Task Card: Exhaustive Distinct Clips Spec

## Goal

Define the production architecture for an Afterform mode that creates every distinct, non-redundant, quality-clearing short from one long source video instead of stopping at an arbitrary clip cap.

## Scope

- `docs/EXHAUSTIVE_DISTINCT_CLIPS_SPEC.md`
- `factory/tasks/active/TC-20260506-001-exhaustive-distinct-clips.md`
- `factory/runs/RR-20260506-001-exhaustive-distinct-clips.md`

## Non-Goals

- Do not change runtime code in this task.
- Do not delete existing `.afterform_runs`, output, cache, or user-modified files.
- Do not commit, push, or open a PR from this sandbox run.
- Do not add new dependencies.

## Context Links

- `docs/PIPELINE.md`
- `docs/PROJECT_ISSUES.md`
- `docs/KNOWN_LIMITATIONS_AND_PROMPT_CONTRACT_GAP.md`
- `src/afterform/config.py`
- `src/afterform/flows/long_to_shorts/select_clips.py`
- `src/afterform/flows/long_to_shorts/flow.py`
- `src/afterform/flows/long_to_shorts/prompts/clip_selection_system.jinja2`
- `tests/test_clip_ranking.py`

Search before working:

```powershell
rg -n "clip_selection|max_kept|candidate_count|duplicate|redundant|threshold" docs src tests
```

## Assumptions To Check

- Assumption: `{project_directory}` means `afterform`.
- How to check: user explicitly named Afterform in the previous turn; repo exists at `C:\Users\wbrya\OneDrive\Documents\GitHub\afterform`.

- Assumption: `{target_goal_link}` means the prior conversation goal: "create as many clips from the video as possible, no limits, no duplicates or redundant clips."
- How to check: no concrete link was provided; mark as unresolved blocker.

- Assumption: `{agents_guidelines_file}` means root `Agents.md`.
- How to check: root has `Agents.md`; it already contains the exact requested philosophy text.

- Assumption: `{code_factory_tool}` means the local `code-factory` repo.
- How to check: `code-factory/AGENTS.md` exists and instructs agents to use `docs/principles.md`, `docs/protocol.md`, and templates.

## Proof

Required proof:

- command: `uv run pytest tests/test_clip_ranking.py tests/test_clip_selection_cache.py tests/test_cli.py`
- expected exit code: `0`
- expected artifact: `docs/EXHAUSTIVE_DISTINCT_CLIPS_SPEC.md`

Doc-only fallback proof if test environment is unavailable:

- command: `Test-Path docs/EXHAUSTIVE_DISTINCT_CLIPS_SPEC.md`
- expected exit code: `0`
- expected artifact: `factory/runs/RR-20260506-001-exhaustive-distinct-clips.md`

## Definition Of Done

- [ ] root guideline text verified
- [ ] current clip-selection cap and cache behavior documented
- [ ] exhaustive mode requirements defined
- [ ] data models, enums, services, connectors, converters, and outputs specified
- [ ] ML strategy and local hardware constraints documented
- [ ] observability, resilience, reliability, and presentation workflow specified
- [ ] implementation plan and tests specified
- [ ] Run Record written
- [ ] one next task suggested

## Approval Rules

- green: read files, create docs, create factory artifacts, run local read-only checks
- yellow: add implementation code under `src/afterform/flows/long_to_shorts` and tests
- red: delete files, clean caches, commit, push, open PR, spend API credits, call paid LLMs, change secrets

## Careful Review

| Action | Color | Why | Bryan Approval Needed? |
|---|---|---|---|
| Create spec docs | green | Reversible doc artifact | no |
| Modify runtime clip selection | yellow | Behavior change across pipeline | no, if scoped and tested |
| Delete cache/output files | red | Existing worktree has user changes and deleted run artifacts | yes |
| Commit/push/PR | red | External state change and repo has unrelated dirty files | yes |
| Spend API credits on live LLM runs | red | Cost and secrets boundary | yes |

## Stop Conditions

- stop if the task requires the actual `{target_goal_link}` to decide scope
- stop if implementation would overwrite unrelated user changes
- stop if a live API call, push, PR, or destructive cleanup is required

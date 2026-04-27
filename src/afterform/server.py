"""FastMCP server exposing Afterform primitives.

Every primitive is exposed as a single MCP ``tool``. Each tool takes and
returns strict Pydantic-validated JSON, so an MCP client (Cursor, Claude
Desktop, etc.) can compose a full long-to-short pipeline without guessing
any interface.

Tools:

    afterform.ingest                      â€” Stage 1 extraction (scenes + keyframes [+ transcript])
    afterform.classify_scenes             â€” Assign one of 5 layouts to each scene (pixel heuristic)
    afterform.classify_scenes_with_vision â€” Assign layouts using bboxes from a vision LLM + OCR
    afterform.detect_scene_regions        â€” Raw LLM bbox output per scene keyframe (OCR-assisted)
    afterform.select_clips                â€” Pick top clips from a transcript (heuristic)
    afterform.plan_layout                 â€” Return the ffmpeg filtergraph for a given layout
    afterform.build_render_cmd            â€” Build the full ffmpeg command (dry-run safe)
    afterform.render_clip                 â€” Build + actually run ffmpeg to produce a 9:16 clip
    afterform.list_layouts                â€” List the 5 available layouts (discovery)

Resources:

    afterform://layouts       â€” JSON listing of the 5 layouts + description
"""

from __future__ import annotations

import json
from typing import Any

from mcp.server.fastmcp import FastMCP

from .primitives import classify as classify_mod
from .primitives import compile as compile_mod
from .primitives import ingest as ingest_mod
from .primitives import layouts as layouts_mod
from .primitives import select_clips as select_mod
from .primitives import vision as vision_mod
from .schemas import (
    IngestResult,
    LayoutInstruction,
    LayoutKind,
    RenderRequest,
    RenderResult,
    Scene,
    SceneRegions,
    TranscriptWord,
)


mcp = FastMCP(
    "afterform-mcp",
    instructions=(
        "Afterform MCP: reusable primitives for turning source media into transformed outputs. "
        "Compose tools in this order: ingest -> classify_scenes -> select_clips -> "
        "plan_layout/build_render_cmd -> render_clip. All IO is strict JSON."
    ),
)


# ---------------------------------------------------------------------------
# Discovery
# ---------------------------------------------------------------------------


@mcp.tool()
def list_layouts() -> dict[str, Any]:
    """Return the 5 fixed 9:16 layouts this server supports.

    Every short shows **at most two** on-screen items (person/chart), which
    gives exactly five recipes. Use this to discover the set of
    :class:`LayoutKind` values before classifying scenes or requesting
    renders.
    """

    return {
        "layouts": [
            {
                "kind": LayoutKind.ZOOM_CALL_CENTER.value,
                "items": ["person"],
                "description": "1 person, tight zoom-call / webcam framing, centered.",
            },
            {
                "kind": LayoutKind.SIT_CENTER.value,
                "items": ["person"],
                "description": "1 person, interview / seated framing, centered.",
            },
            {
                "kind": LayoutKind.SPLIT_CHART_PERSON.value,
                "items": ["chart", "person"],
                "description": (
                    "1 chart + 1 person. Source is partitioned left/right by the chart and "
                    "person bboxes (falling back to a 2/3 | 1/3 split); each strip is scaled "
                    "to fill its output band. Bands default to an even 50/50 vertical split; "
                    "configurable via ``top_band_ratio`` and swappable via ``focus_stack_order``."
                ),
            },
            {
                "kind": LayoutKind.SPLIT_TWO_PERSONS.value,
                "items": ["person", "person"],
                "description": (
                    "2 people (interview two-up / panel). Left speaker in the top band, right "
                    "speaker in the bottom band; seam sits between the two person bboxes."
                ),
            },
            {
                "kind": LayoutKind.SPLIT_TWO_CHARTS.value,
                "items": ["chart", "chart"],
                "description": (
                    "2 charts / slides side-by-side in source. Left chart on top, right chart "
                    "on bottom; each is scaled to fill its band."
                ),
            },
        ]
    }


@mcp.resource("afterform://layouts")
def layouts_resource() -> str:
    return json.dumps(list_layouts(), indent=2)


# ---------------------------------------------------------------------------
# Landing gear: ingest
# ---------------------------------------------------------------------------


@mcp.tool()
def ingest(
    source_path: str,
    work_dir: str,
    with_transcript: bool = False,
    whisper_model: str = "base",
) -> dict[str, Any]:
    """Run deterministic local extraction (scenes + keyframes, optional transcript).

    Args:
        source_path: absolute path to a local video file.
        work_dir: directory where keyframes/ and temp artifacts will be written.
        with_transcript: if True, run faster-whisper word-level transcription.
        whisper_model: whisper model name (e.g. "tiny", "base", "small").
    """

    result: IngestResult = ingest_mod.ingest(
        source_path,
        work_dir,
        with_transcript=with_transcript,
        whisper_model=whisper_model,
    )
    return result.model_dump()


# ---------------------------------------------------------------------------
# Pilot: classify scenes
# ---------------------------------------------------------------------------


@mcp.tool()
def classify_scenes(scenes: list[dict[str, Any]]) -> dict[str, Any]:
    """Classify each scene into exactly one of the 5 supported layouts.

    Uses an offline pixel heuristic on each scene's keyframe. Agents that
    want a smarter classifier can post-process or overwrite the result,
    or call ``classify_scenes_with_vision`` with bboxes from a vision LLM.
    """

    parsed = [Scene.model_validate(s) for s in scenes]
    results = classify_mod.classify_scenes_heuristic(parsed)
    return {"classifications": [r.model_dump() for r in results]}


# ---------------------------------------------------------------------------
# Pilot (alt path): vision-LLM + OCR bbox classifier
# ---------------------------------------------------------------------------


@mcp.tool()
def detect_scene_regions(scenes: list[dict[str, Any]]) -> dict[str, Any]:
    """Return the prompt + per-scene stubs used for LLM+OCR bbox detection.

    This tool is the *adapter* half of the vision primitive. The MCP server
    itself never calls an LLM â€” the agent does. So this endpoint returns:

    1. the exact ``REGION_PROMPT`` to send along with each keyframe, and
    2. a list of ``{scene_id, keyframe_path, prompt}`` jobs.

    The agent runs its own vision model for each job, then feeds the
    resulting JSON back via ``classify_scenes_with_vision``.
    """

    parsed = [Scene.model_validate(s) for s in scenes]
    return {
        "prompt": vision_mod.REGION_PROMPT,
        "jobs": [
            {
                "scene_id": s.scene_id,
                "keyframe_path": s.keyframe_path,
                "prompt": vision_mod.REGION_PROMPT,
            }
            for s in parsed
        ],
    }


@mcp.tool()
def classify_scenes_with_vision(regions: list[dict[str, Any]]) -> dict[str, Any]:
    """Classify scenes from already-gathered ``SceneRegions`` bbox records.

    Input is a list of ``SceneRegions`` JSON dicts (output of the agent's
    vision-LLM pass). Output is a ``{classifications, layout_instructions}``
    pair â€” the layout kind per scene plus a ready-to-render
    ``LayoutInstruction`` with ``person_x_norm`` / ``chart_x_norm`` already
    populated from the bboxes.
    """

    parsed_regions = [SceneRegions.model_validate(r) for r in regions]
    classifications = [vision_mod.classify_from_regions(r) for r in parsed_regions]
    instructions = [
        vision_mod.layout_instruction_from_regions(r, c)
        for r, c in zip(parsed_regions, classifications)
    ]
    return {
        "classifications": [c.model_dump() for c in classifications],
        "layout_instructions": [i.model_dump() for i in instructions],
    }


# ---------------------------------------------------------------------------
# Pilot: select clips
# ---------------------------------------------------------------------------


@mcp.tool()
def select_clips(
    source_path: str,
    transcript_words: list[dict[str, Any]],
    duration_sec: float,
    target_count: int = 5,
    min_sec: float = 30.0,
    max_sec: float = 60.0,
) -> dict[str, Any]:
    """Heuristically select top clips from a word-level transcript.

    Scoring is word-density per window. Returns a ``ClipPlan`` with up to
    ``target_count`` non-overlapping clips.
    """

    words = [TranscriptWord.model_validate(w) for w in transcript_words]
    plan = select_mod.select_clips_heuristic(
        source_path,
        words,
        duration_sec,
        target_count=target_count,
        min_sec=min_sec,
        max_sec=max_sec,
    )
    return plan.model_dump()


# ---------------------------------------------------------------------------
# Thrusters: plan + render
# ---------------------------------------------------------------------------


@mcp.tool()
def plan_layout(
    layout: str,
    out_w: int = 1080,
    out_h: int = 1920,
    src_w: int = 1920,
    src_h: int = 1080,
    zoom: float = 1.0,
    person_x_norm: float = 0.5,
    chart_x_norm: float = 0.0,
    clip_id: str = "preview",
) -> dict[str, Any]:
    """Return the ffmpeg filter_complex fragment for one layout.

    This is the pure, deterministic function underpinning the 5 thrusters.
    No rendering is performed. Useful for agents that want to preview the
    filtergraph or compose it with their own ffmpeg invocation.
    """

    instr = LayoutInstruction(
        clip_id=clip_id,
        layout=LayoutKind(layout),
        zoom=zoom,
        person_x_norm=person_x_norm,
        chart_x_norm=chart_x_norm,
    )
    fp = layouts_mod.plan_layout(instr, out_w=out_w, out_h=out_h, src_w=src_w, src_h=src_h)
    return {"filtergraph": fp.filtergraph, "out_label": fp.out_label}


@mcp.tool()
def build_render_cmd(request: dict[str, Any]) -> dict[str, Any]:
    """Build (but do NOT run) the ffmpeg command for a render request.

    ``request`` must conform to the ``RenderRequest`` schema. This is a
    dry-run helper so an agent can review the command before executing it.
    """

    req = RenderRequest.model_validate({**request, "mode": "dry_run"})
    result = compile_mod.render_clip(req)
    return result.model_dump()


@mcp.tool()
def render_clip(request: dict[str, Any]) -> dict[str, Any]:
    """Render a single 9:16 clip with the specified layout.

    ``request`` must conform to ``RenderRequest``. If ``request.mode`` is
    ``"dry_run"`` the ffmpeg command is returned without execution.
    """

    req = RenderRequest.model_validate(request)
    result: RenderResult = compile_mod.render_clip(req)
    return result.model_dump()


# ---------------------------------------------------------------------------
# Entrypoint
# ---------------------------------------------------------------------------


def main() -> None:
    """stdio entrypoint for ``afterform-mcp`` and legacy alias scripts."""

    mcp.run()


if __name__ == "__main__":
    main()


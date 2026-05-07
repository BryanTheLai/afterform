"""Stage inspection helpers should read runtime artifacts, not fake state."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from afterform.config import PipelineConfig
from afterform.flows.long_to_shorts.stage_inspection import (
    StageArtifactError,
    build_stage_inspection,
    load_state_before_stage,
    write_inspection,
)
from afterform.schemas import Clip, LayoutInstruction, LayoutKind


def _write_json(path: Path, data: dict) -> None:
    path.write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")


def _seed_workdir(tmp_path: Path) -> None:
    _write_json(
        tmp_path / "transcript.json",
        {
            "segments": [
                {
                    "start": 0.0,
                    "end": 1.0,
                    "text": "hello",
                    "words": [{"word": "hello", "start": 0.0, "end": 0.5}],
                },
                {
                    "start": 3.0,
                    "end": 4.0,
                    "text": "world",
                    "words": [{"word": "world", "start": 3.0, "end": 3.5}],
                },
            ]
        },
    )
    clip = Clip(
        clip_id="001",
        topic="t",
        start_time_sec=0.0,
        end_time_sec=5.0,
        transcript="hello world",
    )
    _write_json(tmp_path / "clips.json", {"source_path": "", "clips": [json.loads(clip.model_dump_json())]})
    _write_json(
        tmp_path / "hooks.json",
        {
            "hooks": [
                {
                    "clip_id": "001",
                    "hook_start_sec": 0.2,
                    "hook_end_sec": 0.8,
                    "hook_text": "hello",
                    "reason": "hook",
                }
            ]
        },
    )
    _write_json(
        tmp_path / "prune.json",
        {
            "clips": [
                {
                    "clip_id": "001",
                    "trim_start_sec": 0.0,
                    "trim_end_sec": 0.0,
                    "keep_ranges_sec": [[0.0, 1.0], [3.0, 4.0]],
                    "diagnostics": {"audio_backend": {"speech": "energy_vad", "filled_pause": "none"}},
                }
            ]
        },
    )
    instr = LayoutInstruction(clip_id="001", layout=LayoutKind.SIT_CENTER)
    _write_json(
        tmp_path / "layout_vision.json",
        {
            "clips": {
                "001": {
                    "instruction": json.loads(instr.model_dump_json()),
                    "sampled_frames": [],
                    "frame_results": [],
                    "raw": {"layout": "sit_center"},
                    "warnings": [],
                }
            }
        },
    )
    (tmp_path / "source.mp4").write_bytes(b"fake")
    (tmp_path / "subtitles").mkdir()


def test_load_state_before_stage_requires_hook_artifact_when_enabled(tmp_path: Path):
    _seed_workdir(tmp_path)
    (tmp_path / "hooks.json").unlink()
    cfg = PipelineConfig(
        youtube_url="https://youtu.be/abc",
        work_dir=tmp_path,
        detect_hooks=True,
        prune_level="balanced",
    )
    with pytest.raises(StageArtifactError):
        load_state_before_stage(tmp_path, stage="content-pruning", config=cfg)


def test_content_pruning_inspection_reads_runtime_keep_ranges(tmp_path: Path):
    _seed_workdir(tmp_path)
    cfg = PipelineConfig(
        youtube_url="https://youtu.be/abc",
        work_dir=tmp_path,
        detect_hooks=True,
        prune_level="balanced",
    )
    payload = build_stage_inspection(
        tmp_path,
        stage="content-pruning",
        clip_id="001",
        config=cfg,
    )
    assert payload["prune"][0]["keep_ranges_sec"] == [[0.0, 1.0], [3.0, 4.0]]
    assert payload["prune"][0]["diagnostics"]["audio_backend"]["speech"] == "energy_vad"


def test_render_state_loads_visual_drop_ranges_from_layout_cache(tmp_path: Path):
    _seed_workdir(tmp_path)
    data = json.loads((tmp_path / "layout_vision.json").read_text(encoding="utf-8"))
    data["clips"]["001"]["visual_drop_ranges_sec"] = [[0.4, 0.6]]
    _write_json(tmp_path / "layout_vision.json", data)
    cfg = PipelineConfig(
        youtube_url="https://youtu.be/abc",
        work_dir=tmp_path,
        detect_hooks=True,
        prune_level="balanced",
    )

    state = load_state_before_stage(tmp_path, stage="render", config=cfg)

    assert state.layout_instructions is not None
    assert state.layout_instructions["001"].visual_drop_ranges_sec == [(0.4, 0.6)]


def test_render_inspection_exposes_concat_spans_and_command(tmp_path: Path):
    _seed_workdir(tmp_path)
    cfg = PipelineConfig(
        youtube_url="https://youtu.be/abc",
        work_dir=tmp_path,
        detect_hooks=True,
        prune_level="balanced",
    )
    payload = build_stage_inspection(
        tmp_path,
        stage="render",
        clip_id="001",
        config=cfg,
    )
    render = payload["render"][0]
    assert render["source_keep_ranges_sec"] == [(0.0, 1.0), (3.0, 4.0)]
    assert any("concat=n=2:v=1:a=1" in part for part in render["ffmpeg_cmd"])

    path = write_inspection(tmp_path, stage="render", payload=payload, clip_id="001")
    assert path.is_file()


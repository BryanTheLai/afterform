from __future__ import annotations

import json
from pathlib import Path

from afterform.config import PipelineConfig
from afterform.flows.long_to_shorts.run_ledger import RunLedger, resolve_run_root


def test_resolve_run_root_uses_run_dir(tmp_path: Path):
    config = PipelineConfig(
        youtube_url="https://www.youtube.com/watch?v=abcdefghijk",
        run_dir=tmp_path / "run_001",
        work_dir=tmp_path / "run_001" / "work",
        output_dir=tmp_path / "run_001" / "output",
    )
    root = resolve_run_root(config)
    assert root == config.run_dir
    assert root.is_dir()


def test_run_ledger_writes_stage_and_outputs(tmp_path: Path):
    run_dir = tmp_path / "run_002"
    work_dir = run_dir / "work"
    output_dir = run_dir / "output"
    work_dir.mkdir(parents=True)
    output_dir.mkdir(parents=True)
    (work_dir / "transcript.json").write_text("{}", encoding="utf-8")
    output_path = output_dir / "short_001.mp4"
    output_path.write_text("fake", encoding="utf-8")

    config = PipelineConfig(
        youtube_url="https://www.youtube.com/watch?v=abcdefghijk",
        run_dir=run_dir,
        work_dir=work_dir,
        output_dir=output_dir,
    )
    ledger = RunLedger(config, start_stage="ingest", stop_stage="render")
    ledger.start_stage("ingest")
    ledger.finish_stage("ingest")
    ledger.finish_run([output_path])

    payload = json.loads((run_dir / "run.json").read_text(encoding="utf-8"))
    config_payload = json.loads((run_dir / "config.json").read_text(encoding="utf-8"))
    assert payload["status"] == "completed"
    assert payload["run_id"] == "run_002"
    assert config_payload["run_dir"] == str(run_dir)
    assert payload["stages"]["ingest"]["status"] == "completed"
    assert str(work_dir / "transcript.json") in payload["stages"]["ingest"]["artifacts"]
    assert payload["outputs"] == [str(output_path)]


def test_run_ledger_marks_run_failed_without_stage(tmp_path: Path):
    run_dir = tmp_path / "run_003"
    config = PipelineConfig(
        youtube_url="https://www.youtube.com/watch?v=abcdefghijk",
        run_dir=run_dir,
        work_dir=run_dir / "work",
        output_dir=run_dir / "output",
    )
    ledger = RunLedger(config, start_stage="clip-selection", stop_stage="render")

    ledger.fail_run(RuntimeError("missing clips"))

    payload = json.loads((run_dir / "run.json").read_text(encoding="utf-8"))
    assert payload["status"] == "failed"
    assert payload["finished_at"] is not None
    assert payload["error"] == {"stage": None, "type": "RuntimeError", "message": "missing clips"}


def test_run_ledger_dedupes_render_artifacts(tmp_path: Path):
    run_dir = tmp_path / "run_004"
    work_dir = run_dir / "work"
    output_dir = run_dir / "output"
    output_dir.mkdir(parents=True)
    output_path = output_dir / "short_001.mp4"
    output_path.write_text("fake", encoding="utf-8")
    config = PipelineConfig(
        youtube_url="https://www.youtube.com/watch?v=abcdefghijk",
        run_dir=run_dir,
        work_dir=work_dir,
        output_dir=output_dir,
    )
    ledger = RunLedger(config, start_stage="render", stop_stage="render")

    ledger.start_stage("render")
    ledger.finish_stage("render", extra_artifacts=[str(output_path), str(output_path)])

    payload = json.loads((run_dir / "run.json").read_text(encoding="utf-8"))
    assert payload["stages"]["render"]["artifacts"] == [str(output_path)]

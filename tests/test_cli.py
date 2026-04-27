from __future__ import annotations

import sys
from pathlib import Path

from afterform import cli


def test_inspect_only_does_not_require_url(monkeypatch, tmp_path: Path, capsys):
    calls: dict[str, object] = {}

    def fake_build_stage_inspection(work_dir, stage, clip_id, config):
        calls["work_dir"] = work_dir
        calls["stage"] = stage
        calls["clip_id"] = clip_id
        calls["youtube_url"] = config.youtube_url
        return {"stage": stage, "clip_id": clip_id}

    def fake_write_inspection(work_dir, stage, payload, clip_id):
        calls["payload"] = payload
        path = work_dir / "inspect_clip-selection_003.json"
        path.write_text("{}", encoding="utf-8")
        return path

    def fail_run_pipeline(config):
        raise AssertionError("run_pipeline should not execute for inspect-only mode")

    monkeypatch.setattr(cli, "build_stage_inspection", fake_build_stage_inspection)
    monkeypatch.setattr(cli, "write_inspection", fake_write_inspection)
    monkeypatch.setattr(cli, "run_pipeline", fail_run_pipeline)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "afterform",
            "--work-dir",
            str(tmp_path),
            "--inspect-stage",
            "clip-selection",
            "--clip-id",
            "003",
        ],
    )

    cli.main()

    captured = capsys.readouterr()
    assert "Inspection written:" in captured.out
    assert calls["work_dir"] == tmp_path
    assert calls["stage"] == "clip-selection"
    assert calls["clip_id"] == "003"
    assert calls["youtube_url"] is None



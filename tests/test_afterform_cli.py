from __future__ import annotations

import sys
from pathlib import Path

import pytest

from afterform import cli


def test_afterform_run_dir_expands_to_work_and_output(monkeypatch, tmp_path: Path):
    calls: dict[str, object] = {}

    def fake_run_pipeline(config):
        calls["youtube_url"] = config.youtube_url
        calls["run_dir"] = config.run_dir
        calls["work_dir"] = config.work_dir
        calls["output_dir"] = config.output_dir
        return []

    monkeypatch.setattr(cli, "run_pipeline", fake_run_pipeline)

    run_dir = tmp_path / "run_001"
    cli.main(
        [
            "run",
            "long-to-shorts",
            "https://youtube.com/watch?v=abc123",
            "--run-dir",
            str(run_dir),
        ]
    )

    assert calls["youtube_url"] == "https://youtube.com/watch?v=abc123"
    assert calls["run_dir"] == run_dir
    assert calls["work_dir"] == run_dir / "work"
    assert calls["output_dir"] == run_dir / "output"


def test_clean_run_defaults_to_local_afterform_runs(monkeypatch):
    calls: dict[str, object] = {}

    def fake_run_pipeline(config):
        calls["run_dir"] = config.run_dir
        calls["work_dir"] = config.work_dir
        calls["output_dir"] = config.output_dir
        calls["use_video_cache"] = config.use_video_cache
        return []

    monkeypatch.setattr(cli, "run_pipeline", fake_run_pipeline)

    cli.main(
        [
            "run",
            "long-to-shorts",
            "https://youtube.com/watch?v=abc123",
            "--clean-run",
        ]
    )

    run_dir = calls["run_dir"]
    assert isinstance(run_dir, Path)
    assert run_dir.parts[:2] == (".afterform", "runs")
    assert calls["work_dir"] == run_dir / "work"
    assert calls["output_dir"] == run_dir / "output"
    assert calls["use_video_cache"] is False


def test_default_run_gets_isolated_run_dir(monkeypatch):
    calls: dict[str, object] = {}

    def fake_run_pipeline(config):
        calls["run_dir"] = config.run_dir
        calls["work_dir"] = config.work_dir
        calls["output_dir"] = config.output_dir
        calls["use_video_cache"] = config.use_video_cache
        return []

    monkeypatch.setattr(cli, "run_pipeline", fake_run_pipeline)

    cli.main(["run", "long-to-shorts", "https://youtube.com/watch?v=abc123"])

    run_dir = calls["run_dir"]
    assert isinstance(run_dir, Path)
    assert run_dir.parts[:2] == (".afterform", "runs")
    assert calls["work_dir"] is None
    assert calls["output_dir"] == run_dir / "output"
    assert calls["use_video_cache"] is True


def test_default_no_video_cache_run_uses_isolated_work_dir(monkeypatch):
    calls: dict[str, object] = {}

    def fake_run_pipeline(config):
        calls["run_dir"] = config.run_dir
        calls["work_dir"] = config.work_dir
        calls["output_dir"] = config.output_dir
        calls["use_video_cache"] = config.use_video_cache
        return []

    monkeypatch.setattr(cli, "run_pipeline", fake_run_pipeline)

    cli.main(
        [
            "run",
            "long-to-shorts",
            "https://youtube.com/watch?v=abc123",
            "--no-video-cache",
        ]
    )

    run_dir = calls["run_dir"]
    assert isinstance(run_dir, Path)
    assert run_dir.parts[:2] == (".afterform", "runs")
    assert calls["work_dir"] == run_dir / "work"
    assert calls["output_dir"] == run_dir / "output"
    assert calls["use_video_cache"] is False


def test_no_video_cache_artifact_only_run_still_requires_work_dir(monkeypatch, capsys):
    def fail_run_pipeline(config):
        raise AssertionError("run_pipeline should not execute when work-dir validation fails")

    monkeypatch.setattr(cli, "run_pipeline", fail_run_pipeline)

    with pytest.raises(SystemExit) as exc:
        cli.main(["run", "long-to-shorts", "--start-at", "render", "--no-video-cache"])

    captured = capsys.readouterr()
    assert exc.value.code == 2
    assert "--work-dir is required when the source URL is omitted." in captured.err


def test_afterform_inspect_only_does_not_require_url(monkeypatch, tmp_path: Path, capsys):
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
            "run",
            "long-to-shorts",
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

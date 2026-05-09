from __future__ import annotations

import logging
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


def test_filled_pause_flags_are_passed_to_pipeline(monkeypatch, tmp_path: Path):
    calls: dict[str, object] = {}

    def fake_run_pipeline(config):
        calls["filled_pause_pruning"] = config.filled_pause_pruning
        calls["require_filled_pause_pruning"] = config.require_filled_pause_pruning
        calls["output_dir"] = config.output_dir
        return []

    monkeypatch.setattr(cli, "run_pipeline", fake_run_pipeline)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "afterform",
            "run",
            "long-to-shorts",
            "https://youtu.be/abc",
            "--output",
            str(tmp_path / "out"),
            "--filled-pause-pruning",
            "--require-filled-pause-pruning",
        ],
    )

    cli.main()

    assert calls["filled_pause_pruning"] is True
    assert calls["require_filled_pause_pruning"] is True


def test_filled_pause_pruning_defaults_on(monkeypatch, tmp_path: Path):
    calls: dict[str, object] = {}

    def fake_run_pipeline(config):
        calls["filled_pause_pruning"] = config.filled_pause_pruning
        calls["require_filled_pause_pruning"] = config.require_filled_pause_pruning
        return []

    monkeypatch.setattr(cli, "run_pipeline", fake_run_pipeline)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "afterform",
            "run",
            "long-to-shorts",
            "https://youtu.be/abc",
            "--output",
            str(tmp_path / "out"),
        ],
    )

    cli.main()

    assert calls["filled_pause_pruning"] is True
    assert calls["require_filled_pause_pruning"] is False


def test_no_filled_pause_pruning_flag_disables_default(monkeypatch, tmp_path: Path):
    calls: dict[str, object] = {}

    def fake_run_pipeline(config):
        calls["filled_pause_pruning"] = config.filled_pause_pruning
        return []

    monkeypatch.setattr(cli, "run_pipeline", fake_run_pipeline)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "afterform",
            "run",
            "long-to-shorts",
            "https://youtu.be/abc",
            "--output",
            str(tmp_path / "out"),
            "--no-filled-pause-pruning",
        ],
    )

    cli.main()

    assert calls["filled_pause_pruning"] is False


def test_exhaustive_clip_flags_are_passed_to_pipeline(monkeypatch, tmp_path: Path):
    calls: dict[str, object] = {}

    def fake_run_pipeline(config):
        calls["clip_selection_mode"] = config.clip_selection_mode
        calls["clip_selection_candidate_count"] = config.clip_selection_candidate_count
        calls["clip_selection_quality_threshold"] = config.clip_selection_quality_threshold
        calls["clip_selection_max_kept"] = config.clip_selection_max_kept
        calls["stop_after"] = config.stop_after
        return []

    monkeypatch.setattr(cli, "run_pipeline", fake_run_pipeline)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "afterform",
            "run",
            "long-to-shorts",
            "https://youtu.be/abc",
            "--output",
            str(tmp_path / "out"),
            "--clip-mode",
            "exhaustive",
            "--clip-candidate-count",
            "24",
            "--clip-quality-threshold",
            "0.75",
            "--max-clips",
            "none",
            "--review-only-clips",
        ],
    )

    cli.main()

    assert calls["clip_selection_mode"] == "exhaustive"
    assert calls["clip_selection_candidate_count"] == 24
    assert calls["clip_selection_quality_threshold"] == 0.75
    assert calls["clip_selection_max_kept"] is None
    assert calls["stop_after"] == "clip-selection"


def test_verbose_formatter_pretty_prints_and_trims_large_data_urls():
    image_url = "data:image/png;base64," + ("A" * 60) + ("B" * 60)
    formatter = cli._PrettyJsonLogFormatter("%(message)s")
    record = logging.LogRecord(
        name="openai._base_client",
        level=logging.DEBUG,
        pathname=__file__,
        lineno=1,
        msg="Request options: %s",
        args=(
            {
                "json_data": {
                    "input": [
                        {
                            "role": "user",
                            "content": [
                                {
                                    "type": "input_image",
                                    "image_url": image_url,
                                    "detail": "high",
                                }
                            ],
                        }
                    ]
                }
            },
        ),
        exc_info=None,
    )

    rendered = formatter.format(record)
    trimmed = cli._trim_log_string(image_url)

    assert "Request options: {\n" in rendered
    assert f'"image_url": "{trimmed}"' in rendered
    assert image_url not in rendered


def test_verbose_formatter_stringifies_nested_non_serializable_objects():
    class NonSerializable:
        def __repr__(self) -> str:
            return "<NonSerializable>"

    formatter = cli._PrettyJsonLogFormatter("%(message)s")
    record = logging.LogRecord(
        name="openai._base_client",
        level=logging.DEBUG,
        pathname=__file__,
        lineno=1,
        msg="Request options: %s",
        args=(
            {
                "files": [("file", NonSerializable())],
                "json_data": {"model": "whisper-1"},
            },
        ),
        exc_info=None,
    )

    rendered = formatter.format(record)

    assert "Request options: {\n" in rendered
    assert '"files": [\n' in rendered
    assert '"<NonSerializable>"' in rendered



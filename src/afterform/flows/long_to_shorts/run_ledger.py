"""Run-level JSON record for one pipeline execution."""

from __future__ import annotations

import json
from dataclasses import asdict, is_dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from afterform.config import PipelineConfig


STAGE_ARTIFACTS: dict[str, list[str]] = {
    "ingest": ["source.mp4", "source_audio.wav", "transcript.json", "source.info.json"],
    "clip-selection": ["clips.json", "clip_selection_raw.json", "clips.meta.json", "clip_selection_dedupe.json"],
    "hook-detection": ["hooks.json", "hooks_raw.json", "hooks.meta.json"],
    "content-pruning": ["prune.json", "prune_raw.json", "prune.meta.json"],
    "layout-vision": ["layout_vision.json", "layout_vision.meta.json"],
    "render": [],
}


def _utc_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def _jsonify(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if is_dataclass(value):
        return {k: _jsonify(v) for k, v in asdict(value).items()}
    if isinstance(value, dict):
        return {str(k): _jsonify(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonify(v) for v in value]
    return value


def resolve_run_root(config: PipelineConfig) -> Path:
    if config.run_dir is not None:
        config.run_dir.mkdir(parents=True, exist_ok=True)
        return config.run_dir
    assert config.work_dir is not None
    config.work_dir.mkdir(parents=True, exist_ok=True)
    return config.work_dir


class RunLedger:
    def __init__(self, config: PipelineConfig, *, start_stage: str, stop_stage: str):
        self.config = config
        self.root = resolve_run_root(config)
        self.path = self.root / "run.json"
        self.config_path = self.root / "config.json"
        self.data: dict[str, Any] = {
            "run_id": self.root.name,
            "status": "running",
            "started_at": _utc_now(),
            "finished_at": None,
            "youtube_url": config.youtube_url,
            "paths": {
                "run_dir": str(config.run_dir) if config.run_dir is not None else None,
                "work_dir": str(config.work_dir) if config.work_dir is not None else None,
                "output_dir": str(config.output_dir),
            },
            "stage_window": {"start": start_stage, "stop": stop_stage},
            "config": _jsonify(config),
            "stages": {},
            "outputs": [],
            "error": None,
        }
        self.config_path.write_text(json.dumps(_jsonify(config), indent=2), encoding="utf-8")
        self.write()

    def write(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text(json.dumps(self.data, indent=2), encoding="utf-8")

    def start_stage(self, stage: str) -> None:
        self.data["stages"][stage] = {
            "status": "running",
            "started_at": _utc_now(),
            "finished_at": None,
            "error": None,
            "artifacts": [],
        }
        self.write()

    def finish_stage(self, stage: str, *, extra_artifacts: list[str] | None = None) -> None:
        stage_data = self.data["stages"].setdefault(stage, {})
        stage_data["status"] = "completed"
        stage_data["finished_at"] = _utc_now()
        stage_data["artifacts"] = self._collect_stage_artifacts(stage, extra_artifacts=extra_artifacts)
        self.write()

    def fail_stage(self, stage: str, exc: BaseException) -> None:
        stage_data = self.data["stages"].setdefault(stage, {})
        stage_data["status"] = "failed"
        stage_data["finished_at"] = _utc_now()
        stage_data["error"] = {"type": exc.__class__.__name__, "message": str(exc)}
        stage_data["artifacts"] = self._collect_stage_artifacts(stage)
        self.data["status"] = "failed"
        self.data["finished_at"] = _utc_now()
        self.data["error"] = {"stage": stage, "type": exc.__class__.__name__, "message": str(exc)}
        self.write()

    def fail_run(self, exc: BaseException, *, stage: str | None = None) -> None:
        self.data["status"] = "failed"
        self.data["finished_at"] = _utc_now()
        self.data["error"] = {"stage": stage, "type": exc.__class__.__name__, "message": str(exc)}
        self.write()

    def finish_run(self, outputs: list[Path]) -> None:
        self.data["status"] = "completed"
        self.data["finished_at"] = _utc_now()
        self.data["outputs"] = [str(p) for p in outputs]
        self.write()

    def _collect_stage_artifacts(self, stage: str, *, extra_artifacts: list[str] | None = None) -> list[str]:
        assert self.config.work_dir is not None
        artifacts: list[str] = []
        seen: set[str] = set()

        def append_artifact(path: Path) -> None:
            value = str(path)
            if value in seen:
                return
            seen.add(value)
            artifacts.append(value)

        for rel in STAGE_ARTIFACTS.get(stage, []):
            path = self.config.work_dir / rel
            if path.exists():
                append_artifact(path)
        if stage == "render":
            for path in sorted(self.config.output_dir.glob("short_*.mp4")):
                append_artifact(path)
        if extra_artifacts:
            for rel in extra_artifacts:
                path = Path(rel)
                append_artifact(path)
        return artifacts

import subprocess
import sys
from pathlib import Path

import pytest

from afterform.flows.long_to_shorts.ingest import download_video


def test_download_video_uses_project_ytdlp_and_audio_bearing_format(monkeypatch, tmp_path: Path):
    calls: list[list[str]] = []

    def fake_run(cmd, **kwargs):
        calls.append(cmd)
        if cmd[:3] == [sys.executable, "-m", "yt_dlp"]:
            (tmp_path / "source.mp4").write_bytes(b"fake mp4")
            return subprocess.CompletedProcess(cmd, 0, stdout="", stderr="")
        if cmd[0] == "ffprobe":
            return subprocess.CompletedProcess(cmd, 0, stdout="audio\n", stderr="")
        raise AssertionError(f"unexpected command: {cmd}")

    monkeypatch.setattr(subprocess, "run", fake_run)

    assert download_video("https://www.youtube.com/watch?v=FlCWg-KkUN4", tmp_path) == (
        tmp_path / "source.mp4"
    )

    ytdlp_cmd = calls[0]
    assert ytdlp_cmd[:3] == [sys.executable, "-m", "yt_dlp"]
    selected_format = ytdlp_cmd[ytdlp_cmd.index("--format") + 1]
    assert "bestaudio[ext=m4a]" in selected_format
    assert "acodec!=none" in selected_format
    assert "--merge-output-format" in ytdlp_cmd


def test_download_video_fails_if_downloaded_source_has_no_audio(monkeypatch, tmp_path: Path):
    def fake_run(cmd, **kwargs):
        if cmd[:3] == [sys.executable, "-m", "yt_dlp"]:
            (tmp_path / "source.mp4").write_bytes(b"fake mp4")
            return subprocess.CompletedProcess(cmd, 0, stdout="", stderr="")
        if cmd[0] == "ffprobe":
            return subprocess.CompletedProcess(cmd, 0, stdout="", stderr="")
        raise AssertionError(f"unexpected command: {cmd}")

    monkeypatch.setattr(subprocess, "run", fake_run)

    with pytest.raises(RuntimeError, match="no audio stream"):
        download_video("https://www.youtube.com/watch?v=FlCWg-KkUN4", tmp_path)

"""Landing gear: deterministic, local extraction.

Everything here can run without a GPU, without an API key, and without the
internet (once inputs are present). This follows the HIVE guide's rule
"extraction stays local; LLMs only reason".

Functions:
    probe_duration      — ffprobe wrapper
    detect_scenes       — PySceneDetect (ContentDetector)
    extract_keyframes   — ffmpeg snapshot at each scene midpoint
    transcribe_audio    — faster-whisper (optional dependency)
    ingest              — one-shot convenience runner that returns IngestResult
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
from pathlib import Path

from ..schemas import IngestResult, Scene, TranscriptWord


class IngestError(RuntimeError):
    pass


def _require(binary: str) -> str:
    path = shutil.which(binary)
    if not path:
        raise IngestError(
            f"Required binary not on PATH: {binary!r}. Install it or add the path."
        )
    return path


def probe_duration(source_path: str) -> float:
    ffprobe = _require("ffprobe")
    out = subprocess.run(
        [
            ffprobe,
            "-v",
            "error",
            "-show_entries",
            "format=duration",
            "-of",
            "json",
            source_path,
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    data = json.loads(out.stdout)
    return float(data["format"]["duration"])


def detect_scenes(
    source_path: str, threshold: float = 27.0, min_scene_sec: float = 1.0
) -> list[Scene]:
    """Use PySceneDetect's ContentDetector to split the video into scenes."""

    try:
        from scenedetect import detect, ContentDetector  # type: ignore
    except ModuleNotFoundError as e:
        # scenedetect depends on OpenCV; surface the real missing module.
        missing = getattr(e, "name", "") or str(e)
        hint = "pip install 'scenedetect[opencv]'" if "cv2" in missing else "pip install scenedetect"
        raise IngestError(
            f"Scene detection unavailable (missing module: {missing}). Install with: {hint}"
        ) from e

    result = detect(
        source_path,
        ContentDetector(threshold=threshold, min_scene_len=int(min_scene_sec * 24)),
    )
    scenes: list[Scene] = []
    for i, (start, end) in enumerate(result):
        scenes.append(
            Scene(
                scene_id=f"s{i:04d}",
                start_time=float(start.get_seconds()),
                end_time=float(end.get_seconds()),
            )
        )
    # Guard: if PySceneDetect returns empty (e.g. a single long shot),
    # fall back to one scene spanning the whole video.
    if not scenes:
        duration = probe_duration(source_path)
        scenes.append(Scene(scene_id="s0000", start_time=0.0, end_time=duration))
    return scenes


def extract_keyframes(
    source_path: str, scenes: list[Scene], out_dir: str
) -> list[Scene]:
    """Extract one JPG per scene at its midpoint. Mutates nothing; returns copies."""

    ffmpeg = _require("ffmpeg")
    Path(out_dir).mkdir(parents=True, exist_ok=True)
    updated: list[Scene] = []
    for s in scenes:
        mid = s.start_time + (s.end_time - s.start_time) / 2.0
        out_path = os.path.join(out_dir, f"{s.scene_id}.jpg")
        subprocess.run(
            [
                ffmpeg,
                "-y",
                "-loglevel",
                "error",
                "-ss",
                f"{mid:.3f}",
                "-i",
                source_path,
                "-frames:v",
                "1",
                "-q:v",
                "3",
                out_path,
            ],
            check=True,
        )
        updated.append(s.model_copy(update={"keyframe_path": out_path}))
    return updated


def transcribe_audio(
    source_path: str, model_name: str = "base", language: str | None = None
) -> list[TranscriptWord]:
    """Word-level transcript via faster-whisper. Optional dependency."""

    try:
        from faster_whisper import WhisperModel  # type: ignore
    except ImportError as e:
        raise IngestError(
            "faster-whisper is not installed. pip install faster-whisper"
        ) from e

    model = WhisperModel(model_name, device="auto", compute_type="auto")
    segments, _info = model.transcribe(source_path, word_timestamps=True, language=language)
    words: list[TranscriptWord] = []
    for seg in segments:
        for w in getattr(seg, "words", []) or []:
            if w.word is None:
                continue
            words.append(
                TranscriptWord(
                    word=str(w.word).strip(),
                    start_time=float(w.start or 0.0),
                    end_time=float(w.end or 0.0),
                )
            )
    return words


def ingest(
    source_path: str,
    work_dir: str,
    *,
    with_transcript: bool = False,
    whisper_model: str = "base",
) -> IngestResult:
    """Run all extraction stages and return a single ``IngestResult``."""

    if not os.path.exists(source_path):
        raise IngestError(f"source_path does not exist: {source_path}")

    Path(work_dir).mkdir(parents=True, exist_ok=True)
    keyframes_dir = os.path.join(work_dir, "keyframes")

    duration = probe_duration(source_path)
    scenes = detect_scenes(source_path)
    scenes = extract_keyframes(source_path, scenes, keyframes_dir)

    words: list[TranscriptWord] = []
    if with_transcript:
        words = transcribe_audio(source_path, model_name=whisper_model)

    return IngestResult(
        source_path=os.path.abspath(source_path),
        duration_sec=duration,
        scenes=scenes,
        transcript_words=words,
        keyframes_dir=keyframes_dir,
    )


"""Video ingest cache: YouTube id â†’ work directory + manifest on disk."""

from __future__ import annotations

import json
import logging
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field

from afterform.env import default_afterform_cache_root

logger = logging.getLogger(__name__)

# Typical watch / short / embed URLs (11-char id).
_YOUTUBE_ID_RE = re.compile(
    r"(?:youtube\.com/watch\?v=|youtu\.be/|youtube\.com/embed/|youtube\.com/v/)([a-zA-Z0-9_-]{11})"
)

MANIFEST_VERSION = 1
MANIFEST_NAME = "video_cache_manifest.json"


class VideoCacheEntry(BaseModel):
    """One row in the global cache manifest (machine-checkable, Pydantic-only)."""

    video_id: str
    url: str = ""
    title: str = ""
    channel: str = ""
    work_dir: str
    source_mp4: str
    transcript_json: str
    downloaded_at: str = ""  # ISO 8601 UTC when ingest completed


class VideoCacheManifest(BaseModel):
    version: int = MANIFEST_VERSION
    entries: dict[str, VideoCacheEntry] = Field(default_factory=dict)


def extract_youtube_video_id(url: str) -> str | None:
    """Return the 11-character video id, or None if not a recognized YouTube URL."""
    m = _YOUTUBE_ID_RE.search(url)
    return m.group(1) if m else None


def manifest_path(cache_root: Path | None = None) -> Path:
    root = cache_root if cache_root is not None else default_afterform_cache_root()
    root.mkdir(parents=True, exist_ok=True)
    return root / MANIFEST_NAME


def load_manifest(cache_root: Path | None = None) -> VideoCacheManifest:
    path = manifest_path(cache_root)
    if not path.exists():
        return VideoCacheManifest()
    with open(path, encoding="utf-8") as f:
        data: Any = json.load(f)
    return VideoCacheManifest.model_validate(data)


def save_manifest(manifest: VideoCacheManifest, cache_root: Path | None = None) -> Path:
    path = manifest_path(cache_root)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        f.write(manifest.model_dump_json(indent=2))
    return path


def resolve_work_directory(
    *,
    youtube_url: str,
    explicit_work_dir: Path | None,
    use_video_cache: bool,
    cache_root: Path | None,
) -> Path:
    """Pick the directory for ``source.mp4``, ``transcript.json``, ``clips.json``, etc.

    - If ``explicit_work_dir`` is set (CLI ``--work-dir``), use it.
    - Else if video cache is disabled or the URL has no YouTube id, use ``.afterform_work``.
    - Else use ``<cache_root>/videos/<video_id>/`` (creates parents as needed).
    """
    if explicit_work_dir is not None:
        p = Path(explicit_work_dir).resolve()
        p.mkdir(parents=True, exist_ok=True)
        return p

    if not use_video_cache:
        p = Path(".afterform_work").resolve()
        p.mkdir(parents=True, exist_ok=True)
        return p

    vid = extract_youtube_video_id(youtube_url)
    if not vid:
        p = Path(".afterform_work").resolve()
        p.mkdir(parents=True, exist_ok=True)
        return p

    root = cache_root if cache_root is not None else default_afterform_cache_root()
    p = (root / "videos" / vid).resolve()
    p.mkdir(parents=True, exist_ok=True)
    return p


def ingest_complete(work_dir: Path) -> bool:
    """Return True if both video and transcript exist (repeat-run reuse)."""
    return (work_dir / "source.mp4").is_file() and (work_dir / "transcript.json").is_file()


def read_youtube_info_json(work_dir: Path) -> dict[str, Any]:
    """Read ``source.info.json`` written by yt-dlp ``--write-info-json``."""
    p = work_dir / "source.info.json"
    if not p.is_file():
        return {}
    with open(p, encoding="utf-8") as f:
        return json.load(f)


def upsert_manifest_from_info(
    *,
    work_dir: Path,
    youtube_url: str,
    info: dict[str, Any],
    cache_root: Path | None = None,
) -> None:
    """Merge or add a manifest entry after successful ingest."""
    vid = (info.get("id") or extract_youtube_video_id(youtube_url) or "").strip()
    if not vid:
        logger.debug("No video id for manifest; skipping.")
        return

    now = datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")
    wd = work_dir.resolve()
    entry = VideoCacheEntry(
        video_id=vid,
        url=str(info.get("webpage_url") or youtube_url),
        title=str(info.get("title") or ""),
        channel=str(info.get("channel") or info.get("uploader") or ""),
        work_dir=str(wd),
        source_mp4=str((wd / "source.mp4").resolve()),
        transcript_json=str((wd / "transcript.json").resolve()),
        downloaded_at=now,
    )

    manifest = load_manifest(cache_root)
    manifest.entries[vid] = entry
    path = save_manifest(manifest, cache_root)
    logger.info("Updated video cache manifest: %s", path)


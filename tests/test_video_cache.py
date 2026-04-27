"""Video id extraction, work-dir resolution, and manifest helpers."""

import json

from afterform.flows.long_to_shorts.video_cache import (
    VideoCacheEntry,
    VideoCacheManifest,
    extract_youtube_video_id,
    ingest_complete,
    load_manifest,
    manifest_path,
    resolve_work_directory,
    save_manifest,
)


def test_extract_youtube_video_id_watch():
    assert extract_youtube_video_id("https://www.youtube.com/watch?v=PdVv_vLkUgk") == "PdVv_vLkUgk"


def test_extract_youtube_video_id_short():
    assert extract_youtube_video_id("https://youtu.be/PdVv_vLkUgk") == "PdVv_vLkUgk"


def test_extract_youtube_video_id_none():
    assert extract_youtube_video_id("https://example.com") is None


def test_resolve_explicit_work_dir(tmp_path):
    explicit = tmp_path / "custom"
    got = resolve_work_directory(
        youtube_url="https://youtu.be/PdVv_vLkUgk",
        explicit_work_dir=explicit,
        use_video_cache=True,
        cache_root=None,
    )
    assert got == explicit
    assert got.is_dir()


def test_resolve_no_cache_uses_afterform_work(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    got = resolve_work_directory(
        youtube_url="https://youtu.be/PdVv_vLkUgk",
        explicit_work_dir=None,
        use_video_cache=False,
        cache_root=None,
    )
    assert got == tmp_path / ".afterform_work"


def test_resolve_video_cache_path(tmp_path):
    got = resolve_work_directory(
        youtube_url="https://youtu.be/abcdefghijk",
        explicit_work_dir=None,
        use_video_cache=True,
        cache_root=tmp_path / "cache",
    )
    assert got == tmp_path / "cache" / "videos" / "abcdefghijk"


def test_ingest_complete_requires_both(tmp_path):
    assert ingest_complete(tmp_path) is False
    (tmp_path / "source.mp4").write_bytes(b"x")
    assert ingest_complete(tmp_path) is False
    (tmp_path / "transcript.json").write_text("{}")
    assert ingest_complete(tmp_path) is True


def test_manifest_roundtrip(tmp_path):
    m = VideoCacheManifest(
        entries={
            "abc": VideoCacheEntry(
                video_id="abc",
                url="u",
                title="t",
                channel="c",
                work_dir="/w",
                source_mp4="/w/s.mp4",
                transcript_json="/w/t.json",
                downloaded_at="2026-01-01T00:00:00Z",
            )
        }
    )
    path = save_manifest(m, cache_root=tmp_path)
    assert path == manifest_path(tmp_path)
    loaded = load_manifest(tmp_path)
    assert loaded.entries["abc"].video_id == "abc"


def test_manifest_json_stable(tmp_path):
    """Manifest on disk is valid JSON with expected top-level keys."""
    save_manifest(VideoCacheManifest(), cache_root=tmp_path)
    raw = json.loads((tmp_path / "video_cache_manifest.json").read_text(encoding="utf-8"))
    assert raw["version"] == 1
    assert raw["entries"] == {}


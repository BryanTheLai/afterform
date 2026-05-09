"""Clip selection cache fingerprint (provider-aware meta)."""

from afterform.flows.long_to_shorts.clip_selection_cache import (
    CURRENT_META_VERSION,
    cache_valid,
    clip_selection_policy,
    load_meta,
    transcript_fingerprint,
    write_artifacts,
)
from afterform.config import PipelineConfig


def test_transcript_fingerprint_stable():
    t1 = {"segments": [{"start": 1.0, "end": 2.0, "text": "a"}]}
    t2 = {"segments": [{"start": 1.0, "end": 2.0, "text": "a"}]}
    assert transcript_fingerprint(t1) == transcript_fingerprint(t2)


def test_transcript_fingerprint_key_order():
    t1 = {"a": 1, "b": 2}
    t2 = {"b": 2, "a": 1}
    assert transcript_fingerprint(t1) == transcript_fingerprint(t2)


def test_cache_roundtrip_v4(tmp_path):
    tr = {"segments": []}
    cfg = PipelineConfig(youtube_url="https://youtu.be/x", gemini_model="m")
    write_artifacts(tmp_path, transcript=tr, config=cfg, raw_response='{"clips":[]}')
    meta = load_meta(tmp_path)
    assert meta is not None
    assert meta.get("version") == CURRENT_META_VERSION
    assert meta["llm"] == {"provider": "gemini", "model": "m"}
    assert cache_valid(meta, transcript_fingerprint(tr), cfg)


def test_cache_invalidates_on_model_change(tmp_path):
    tr = {"segments": []}
    cfg = PipelineConfig(youtube_url="https://youtu.be/x", gemini_model="m")
    write_artifacts(tmp_path, transcript=tr, config=cfg, raw_response="{}")
    meta = load_meta(tmp_path)
    cfg2 = PipelineConfig(youtube_url="https://youtu.be/x", gemini_model="other")
    assert not cache_valid(meta, transcript_fingerprint(tr), cfg2)


def test_cache_invalidates_on_provider_change(tmp_path):
    tr = {"segments": []}
    cfg = PipelineConfig(youtube_url="https://youtu.be/x", gemini_model="m")
    write_artifacts(tmp_path, transcript=tr, config=cfg, raw_response="{}")
    meta = load_meta(tmp_path)
    cfg2 = PipelineConfig(
        youtube_url="https://youtu.be/x",
        llm_provider="openai",
        llm_model="gpt-5.4",
    )
    assert not cache_valid(meta, transcript_fingerprint(tr), cfg2)


def test_legacy_v1_openai_meta_invalidates(tmp_path):
    """Old meta from --provider openai runs must not hit cache."""
    tr = {"segments": []}
    cfg = PipelineConfig(youtube_url="https://youtu.be/x", gemini_model="gemini-2.0-flash")
    meta = {
        "version": 1,
        "transcript_sha256": transcript_fingerprint(tr),
        "llm_provider": "openai",
        "openai_model": "gpt-4o",
    }
    assert not cache_valid(meta, transcript_fingerprint(tr), cfg)


def test_cache_policy_records_exhaustive_mode_and_uncapped_max():
    cfg = PipelineConfig(
        youtube_url="https://youtu.be/x",
        clip_selection_mode="exhaustive",
        clip_selection_max_kept=None,
    )
    policy = clip_selection_policy(cfg)
    assert policy["mode"] == "exhaustive"
    assert policy["max_kept"] is None
    assert policy["dedupe_policy_version"] >= 1


def test_cache_invalidates_on_clip_selection_mode_change(tmp_path):
    tr = {"segments": []}
    cfg = PipelineConfig(youtube_url="https://youtu.be/x", clip_selection_mode="curated")
    write_artifacts(tmp_path, transcript=tr, config=cfg, raw_response="{}")
    meta = load_meta(tmp_path)
    cfg2 = PipelineConfig(
        youtube_url="https://youtu.be/x",
        clip_selection_mode="exhaustive",
        clip_selection_max_kept=None,
    )
    assert not cache_valid(meta, transcript_fingerprint(tr), cfg2)


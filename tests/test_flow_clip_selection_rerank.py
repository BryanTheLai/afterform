import json

from afterform.config import PipelineConfig
from afterform.flows.long_to_shorts.clip_selection_cache import write_artifacts
from afterform.flows.long_to_shorts.flow import run_pipeline
from afterform.flows.long_to_shorts.select_clips import save_clips
from afterform.schemas import Clip


def test_clip_selection_rerank_reuses_raw_pool(tmp_path):
    transcript = {
        "segments": [
            {"start": 0.0, "end": 55.0, "text": "A complete thought because"},
            {"start": 55.0, "end": 61.0, "text": "the sentence completes."},
        ]
    }
    raw_response = json.dumps(
        {
            "clips": [
                {
                    "clip_id": "old",
                    "topic": "rerank",
                    "start_time_sec": 0.0,
                    "end_time_sec": 55.0,
                    "virality_score": 0.9,
                }
            ]
        }
    )
    config = PipelineConfig(
        work_dir=tmp_path,
        output_dir=tmp_path / "out",
        llm_provider="gemini",
        gemini_model="test-model",
        start_at="clip-selection",
        stop_after="clip-selection",
        clip_selection_min_kept=1,
    )
    (tmp_path / "transcript.json").write_text(json.dumps(transcript), encoding="utf-8")
    save_clips(
        [
            Clip(
                clip_id="old",
                topic="stale",
                start_time_sec=0.0,
                end_time_sec=55.0,
                virality_score=0.1,
            )
        ],
        tmp_path / "clips.json",
    )
    write_artifacts(
        tmp_path,
        transcript=transcript,
        config=config,
        raw_response=raw_response,
    )
    meta_path = tmp_path / "clips.meta.json"
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    meta["ranking_policy_sha256"] = "old-policy"
    meta_path.write_text(json.dumps(meta), encoding="utf-8")

    outputs = run_pipeline(config)

    saved = json.loads((tmp_path / "clips.json").read_text(encoding="utf-8"))["clips"]
    assert outputs == []
    assert saved[0]["clip_id"] == "001"
    assert saved[0]["end_time_sec"] == 61.0

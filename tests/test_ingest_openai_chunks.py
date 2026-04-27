from unittest.mock import patch

from afterform.flows.long_to_shorts.ingest import (
    _merge_transcripts,
    _offset_transcript_timestamps,
    _plan_openai_chunk_ranges,
    transcribe_whisperx,
)


def test_plan_openai_chunk_ranges_single_chunk_when_under_limit():
    ranges = _plan_openai_chunk_ranges(duration_sec=600.0, file_size_bytes=10 * 1024 * 1024)
    assert ranges == [(0.0, 600.0)]


def test_plan_openai_chunk_ranges_splits_large_file():
    ranges = _plan_openai_chunk_ranges(duration_sec=3600.0, file_size_bytes=80 * 1024 * 1024)
    assert len(ranges) >= 2
    assert ranges[0][0] == 0.0
    total_duration = sum(duration for _, duration in ranges)
    assert abs(total_duration - 3600.0) < 0.01


def test_offset_transcript_timestamps_shifts_segments_and_words():
    transcript = {
        "language": "en",
        "segments": [
            {
                "start": 1.0,
                "end": 3.0,
                "text": "hello world",
                "words": [
                    {"word": "hello", "start": 1.0, "end": 1.5},
                    {"word": "world", "start": 1.5, "end": 2.0},
                ],
            }
        ],
    }

    shifted = _offset_transcript_timestamps(transcript, 120.0)
    segment = shifted["segments"][0]
    assert segment["start"] == 121.0
    assert segment["end"] == 123.0
    assert segment["words"][0]["start"] == 121.0
    assert segment["words"][1]["end"] == 122.0


def test_merge_transcripts_concatenates_segments():
    merged = _merge_transcripts(
        [
            {"language": "en", "segments": [{"start": 0.0, "end": 1.0, "text": "a", "words": []}]},
            {"language": "en", "segments": [{"start": 1.0, "end": 2.0, "text": "b", "words": []}]},
        ]
    )
    assert merged["language"] == "en"
    assert len(merged["segments"]) == 2


def test_transcribe_provider_openai_calls_openai_api(monkeypatch, tmp_path):
    """When AFTERFORM_TRANSCRIBE_PROVIDER=openai, do not require whisperx."""
    monkeypatch.setenv("AFTERFORM_TRANSCRIBE_PROVIDER", "openai")
    audio = tmp_path / "a.wav"
    audio.write_bytes(b"x")
    out = {"segments": [], "language": "en"}
    with patch("afterform.flows.long_to_shorts.ingest._transcribe_openai_api", return_value=out) as m:
        r = transcribe_whisperx(audio, tmp_path)
    m.assert_called_once_with(audio)
    assert r == out
    assert (tmp_path / "transcript.json").read_text(encoding="utf-8").strip()


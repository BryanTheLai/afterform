from afterform.primitives.select_clips import select_clips_heuristic
from afterform.flows.long_to_shorts.select_clips import complete_clip_boundaries
from afterform.schemas import Clip
from afterform.schemas import TranscriptWord


def _words(start: float, end: float, n: int) -> list[TranscriptWord]:
    step = (end - start) / max(1, n)
    return [
        TranscriptWord(word=f"w{i}", start_time=start + i * step, end_time=start + (i + 1) * step)
        for i in range(n)
    ]


def test_no_transcript_returns_single_clip():
    plan = select_clips_heuristic("/tmp/x.mp4", [], duration_sec=600.0)
    assert len(plan.clips) == 1


def test_prefers_dense_windows():
    # dense between 30-90, sparse elsewhere
    dense = _words(30.0, 90.0, 240)  # 4 words/sec
    sparse_before = _words(0.0, 30.0, 6)
    sparse_after = _words(90.0, 600.0, 30)
    words = sparse_before + dense + sparse_after
    plan = select_clips_heuristic(
        "/tmp/x.mp4", words, duration_sec=600.0, target_count=1, min_sec=30, max_sec=60
    )
    assert len(plan.clips) == 1
    c = plan.clips[0]
    assert 30 <= c.start_time_sec <= 90
    assert c.end_time_sec <= 120


def test_no_overlap_when_multiple_picked():
    dense_a = _words(30.0, 90.0, 240)
    dense_b = _words(200.0, 260.0, 240)
    words = dense_a + dense_b
    plan = select_clips_heuristic(
        "/tmp/x.mp4",
        words,
        duration_sec=400.0,
        target_count=3,
        min_sec=30,
        max_sec=60,
    )
    # Should pick both dense regions without overlap.
    assert len(plan.clips) >= 2
    starts_ends = sorted((c.start_time_sec, c.end_time_sec) for c in plan.clips)
    for (s1, e1), (s2, e2) in zip(starts_ends, starts_ends[1:]):
        assert e1 <= s2


def test_complete_clip_boundaries_extends_dangling_clause():
    clip = Clip(
        clip_id="001",
        topic="founder skill",
        start_time_sec=118.4,
        end_time_sec=186.1,
        transcript="The component of entrepreneurship,",
        virality_score=0.9,
    )
    transcript = {
        "segments": [
            {"start": 182.94, "end": 186.12, "text": "The component of entrepreneurship,"},
            {
                "start": 186.28,
                "end": 188.82,
                "text": "I can never quite say that word with a straight face,",
            },
            {"start": 189.78, "end": 192.94, "text": "that really matters is domain expertise."},
        ]
    }

    [completed] = complete_clip_boundaries([clip], transcript)

    assert completed.end_time_sec == 192.94
    assert "domain expertise" in completed.transcript


def test_complete_clip_boundaries_extends_immediate_continuation():
    clip = Clip(
        clip_id="001",
        topic="side projects",
        start_time_sec=0.0,
        end_time_sec=56.2,
        transcript="They were all just side projects.",
        virality_score=0.9,
    )
    transcript = {
        "segments": [
            {"start": 51.7, "end": 56.2, "text": "They were all just side projects."},
            {
                "start": 56.2,
                "end": 61.68,
                "text": "The very best ideas almost have to start as side projects because they're always such outliers",
            },
            {
                "start": 61.68,
                "end": 65.82,
                "text": "that your conscious mind would reject them as ideas for companies.",
            },
            {
                "start": 67.2,
                "end": 71.0,
                "text": "So how do you turn your mind into a type that has startup ideas unconsciously?",
            },
        ]
    }

    [completed] = complete_clip_boundaries([clip], transcript)

    assert completed.end_time_sec == 65.82
    assert "conscious mind" in completed.transcript


def test_complete_clip_boundaries_allows_max_duration_overrun_to_finish_sentence():
    clip = Clip(
        clip_id="001",
        topic="long point",
        start_time_sec=0.0,
        end_time_sec=90.0,
        transcript="This is the setup,",
        virality_score=0.9,
    )
    transcript = {
        "segments": [
            {"start": 87.0, "end": 90.0, "text": "This is the setup,"},
            {"start": 90.1, "end": 96.0, "text": "and this is the sentence finishing."},
        ]
    }

    [completed] = complete_clip_boundaries([clip], transcript)

    assert completed.duration_sec == 96.0


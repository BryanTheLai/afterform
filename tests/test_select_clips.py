from afterform.primitives.select_clips import select_clips_heuristic
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


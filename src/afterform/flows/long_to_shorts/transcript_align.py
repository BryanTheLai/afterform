"""Map source-timeline ASR words to per-clip subtitle timings (t=0 at clip in-point)."""

from __future__ import annotations

from afterform.schemas import Clip, ClipSubtitleWords, TranscriptWord

from afterform.flows.long_to_shorts.render_window import clip_output_duration, source_keep_ranges

# Whisper / WhisperX / OpenAI-normalized segment shapes
_MAX_WORDS_PER_CUE = 8
_MAX_CUE_SEC = 4.0


def _iter_words_from_segments(transcript: dict) -> list[TranscriptWord]:
    out: list[TranscriptWord] = []
    for seg in transcript.get("segments", []) or []:
        words = seg.get("words") or []
        if words:
            for raw in words:
                w = str(raw.get("word", "")).strip()
                if not w:
                    continue
                out.append(
                    TranscriptWord(
                        word=w,
                        start_time=float(raw["start"]),
                        end_time=float(raw["end"]),
                    )
                )
            continue
        # Segment-level only (no word list): treat whole segment as one token
        text = str(seg.get("text", "")).strip()
        if text:
            out.append(
                TranscriptWord(
                    word=text,
                    start_time=float(seg.get("start", 0.0)),
                    end_time=float(seg.get("end", 0.0)),
                )
            )
    return out


def clip_subtitle_words(transcript: dict, clip: Clip) -> ClipSubtitleWords:
    """Words overlapping ``clip`` with times shifted to concatenated clip-local output."""
    keep_ranges = source_keep_ranges(clip)
    words = _iter_words_from_segments(transcript)
    local: list[TranscriptWord] = []
    output_cursor = 0.0
    for range_start, range_end in keep_ranges:
        for w in words:
            if w.end_time <= range_start or w.start_time >= range_end:
                continue
            t0 = output_cursor + (max(w.start_time, range_start) - range_start)
            t1 = output_cursor + (min(w.end_time, range_end) - range_start)
            if t1 <= t0:
                continue
            local.append(TranscriptWord(word=w.word, start_time=t0, end_time=t1))
        output_cursor += range_end - range_start

    if local:
        return ClipSubtitleWords(words=local)

    return ClipSubtitleWords(words=_fallback_even_words(clip))


def _fallback_even_words(clip: Clip) -> list[TranscriptWord]:
    """Even split over clip duration when no word timestamps exist."""
    text = (clip.transcript or "").strip()
    if not text:
        return []
    parts = text.split()
    if not parts:
        return []
    d = clip_output_duration(clip)
    step = d / len(parts)
    out: list[TranscriptWord] = []
    for i, p in enumerate(parts):
        out.append(
            TranscriptWord(
                word=p,
                start_time=i * step,
                end_time=(i + 1) * step if i < len(parts) - 1 else d,
            )
        )
    return out


def clip_words_to_srt_lines(
    words: list[TranscriptWord],
    *,
    max_words_per_cue: int = _MAX_WORDS_PER_CUE,
    max_cue_sec: float = _MAX_CUE_SEC,
) -> list[tuple[float, float, str]]:
    """Group words into SRT cues: max N words and max duration per cue."""
    if not words:
        return []
    max_words_per_cue = max(1, int(max_words_per_cue))
    max_cue_sec = max(0.2, float(max_cue_sec))
    lines: list[tuple[float, float, str]] = []
    i = 0
    n = len(words)
    while i < n:
        chunk: list[TranscriptWord] = [words[i]]
        t0 = words[i].start_time
        end_t = words[i].end_time
        j = i + 1
        while j < n:
            w = words[j]
            if len(chunk) >= max_words_per_cue:
                break
            if w.start_time - t0 > max_cue_sec:
                break
            chunk.append(w)
            end_t = w.end_time
            j += 1
        text = " ".join(w.word for w in chunk)
        lines.append((t0, end_t, text))
        i = j
    return lines


def format_srt(lines: list[tuple[float, float, str]]) -> str:
    blocks: list[str] = []
    for idx, (start, end, text) in enumerate(lines, start=1):
        blocks.append(
            f"{idx}\n{_fmt_time(start)} --> {_fmt_time(end)}\n{text}\n"
        )
    return "\n".join(blocks)


def _fmt_time(seconds: float) -> str:
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = int(seconds % 60)
    millis = int(round((seconds % 1) * 1000))
    if millis >= 1000:
        millis = 999
    return f"{hours:02d}:{minutes:02d}:{secs:02d},{millis:03d}"


# ---------------------------------------------------------------------------
# ASS / SubStation Alpha output (the format libass natively renders)
# ---------------------------------------------------------------------------


def _fmt_ass_time(seconds: float) -> str:
    """ASS time format: ``H:MM:SS.cs`` (centiseconds)."""
    seconds = max(0.0, seconds)
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = seconds % 60
    whole = int(secs)
    cs = int(round((secs - whole) * 100))
    if cs >= 100:
        cs = 99
    return f"{hours:d}:{minutes:02d}:{whole:02d}.{cs:02d}"


def _escape_ass_text(text: str) -> str:
    """Escape characters that are significant to the ASS dialogue parser."""
    return (
        text.replace("\\", r"\\")
        .replace("{", r"\{")
        .replace("}", r"\}")
        .replace("\n", r"\N")
    )


def format_ass(
    lines: list[tuple[float, float, str]],
    *,
    play_res_x: int,
    play_res_y: int,
    font_size: int,
    margin_v: int,
    margin_h: int = 60,
    font_name: str = "Arial",
) -> str:
    """Render captions as an ASS script whose PlayRes matches the output video.

    Why this exists: libass' font/margin scaling multiplies every pixel-ish
    value by ``video_height / PlayResY``. The default ``PlayResY=288`` blew
    ``FontSize=48`` up to ~320 output pixels and pushed ``MarginV`` to the
    middle of the frame. Pinning ``PlayResY`` to the actual output height
    makes that scale factor exactly 1.0, so ``font_size`` and ``margin_v``
    below are honest output pixel values.
    """

    header = (
        "[Script Info]\n"
        "ScriptType: v4.00+\n"
        f"PlayResX: {play_res_x}\n"
        f"PlayResY: {play_res_y}\n"
        "WrapStyle: 0\n"
        "ScaledBorderAndShadow: yes\n"
        "YCbCr Matrix: None\n"
        "\n"
        "[V4+ Styles]\n"
        "Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, "
        "OutlineColour, BackColour, Bold, Italic, Underline, StrikeOut, "
        "ScaleX, ScaleY, Spacing, Angle, BorderStyle, Outline, Shadow, "
        "Alignment, MarginL, MarginR, MarginV, Encoding\n"
        # Bold=-1, Italic=0, ScaleX/Y=100, BorderStyle=4 (opaque box),
        # Outline=0, Shadow=0, Alignment=2 (bottom-center).
        f"Style: Default,{font_name},{font_size},&H00FFFFFF,&H000000FF,"
        f"&H00000000,&H70000000,-1,0,0,0,100,100,0,0,4,0,0,2,"
        f"{margin_h},{margin_h},{margin_v},0\n"
        "\n"
        "[Events]\n"
        "Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text\n"
    )

    events = []
    for start, end, text in lines:
        events.append(
            f"Dialogue: 0,{_fmt_ass_time(start)},{_fmt_ass_time(end)},Default,,"
            f"0,0,0,,{_escape_ass_text(text)}"
        )
    return header + "\n".join(events) + ("\n" if events else "")


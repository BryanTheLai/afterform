"""Compiler: assemble a final 9:16 clip from source + clip + layout instruction.

Builds the ffmpeg invocation, optionally runs it. Keeping ``dry_run`` as a
first-class mode means the MCP server can return the exact command without
executing — ideal for an agent that wants to review before spending CPU.

Rendering order is fixed and intentional:

1. **Cut + crop/compose.** ``plan_layout`` produces the base filtergraph
   that takes the source, applies the layout-specific crops, and emits a
   labelled ``[vout]`` at the exact output resolution (e.g. 1080x1920).
2. **Overlay title** (``drawtext``) — skipped for split layouts because
   the source itself already has a slide/chart title and an extra overlay
   just obscures content.
3. **Subtitles.** ``subtitles`` filter runs **last** so text is drawn over
   the finished composition, not the source. ``original_size`` is pinned
   to the output resolution so libass coordinate math (MarginV, FontSize)
   is in *output pixels*, not libass's default PlayResY=288 — which was
   the bug behind the "subtitles blocked / floating in the middle" look.
4. **Mux** with normalized source audio so quiet source uploads do not render
   as effectively muted shorts.
"""

from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

from ..schemas import SPLIT_LAYOUTS, LayoutKind, RenderRequest, RenderResult
from .layouts import plan_layout


def _ensure_ffmpeg() -> str:
    exe = shutil.which("ffmpeg")
    if not exe:
        raise RuntimeError("ffmpeg not found on PATH")
    return exe


def _escape_drawtext(text: str) -> str:
    # drawtext quoting is brittle across ffmpeg builds. Keep it simple:
    # collapse whitespace, drop apostrophes, and escape the characters
    # that are still significant to the filter parser.
    safe = " ".join(text.split()).replace("'", "")
    return safe.replace("\\", "\\\\").replace(":", "\\:")


# ---------------------------------------------------------------------------
# Title overlay planning
# ---------------------------------------------------------------------------
#
# ffmpeg ``drawtext`` does not wrap text by itself; whatever you hand it is
# emitted as a single line. With a fixed 72px font and no width budget, the
# "Prediction Markets vs Derivatives" title on a 1080px canvas would spill
# past both edges and show up clipped (the user reported exactly this bug).
#
# The helpers below plan a title layout BEFORE it hits drawtext:
#
# 1. Short titles (fit at 72px single line): emit the existing single
#    ``drawtext`` call unchanged so golden tests and previously-calibrated
#    visuals stay byte-for-byte identical.
# 2. Long titles: split at the best word boundary into two balanced lines and
#    emit two stacked ``drawtext`` filters at a slightly smaller font
#    (60px / 52px / 44px, auto-shrinking until both lines fit).
# 3. Single-word titles that still overflow: shrink the single line until it
#    fits, then hard-truncate with an ellipsis as a last resort.
#
# The character-width estimate is deliberately conservative (0.55 * fontsize)
# so mixed-case prose with wide letters like W/M still clears the margin.
# Calibrated visually against Arial Bold on 1080p output.

_TITLE_PRIMARY_SIZE = 72   # Current "hero" title size; preserved for short titles.
_TITLE_MIN_SIZE = 44       # Readability floor at 1080x1920 output.
_TITLE_MARGIN_PX = 60      # Horizontal safe-area on each side.
_TITLE_Y_TOP = 80          # Pixel offset of the top title baseline (matches pre-P2 look).
_TITLE_CHAR_WIDTH_RATIO = 0.55
_TITLE_LINE_SPACING_RATIO = 1.3

# Keep the overlay font explicit. Without a ``font=`` directive, drawtext
# falls back to fontconfig's "Sans", which resolves to a serif (Times New
# Roman) on default Windows installs — the "ugly serif title" bug reported
# against v1. Arial matches the ASS subtitle ``Fontname`` below so the
# title and captions read as a single typographic family. Keep this in
# sync with the ``Fontname=Arial`` in the subtitle filter if it ever
# changes.
_TITLE_FONT_NAME = "Arial"
_AUDIO_GAIN_FILTER = "volume=8dB,alimiter=limit=0.95,aresample=44100"


def _title_char_px(size_px: int) -> float:
    return size_px * _TITLE_CHAR_WIDTH_RATIO


def _title_fits(text: str, size_px: int, usable_w: int) -> bool:
    return int(len(text) * _title_char_px(size_px)) <= usable_w


def _wrap_title_two_lines(text: str) -> tuple[str, str]:
    """Split ``text`` at the word boundary that most balances the two halves.

    Returns ``(line1, line2)``. If ``text`` has fewer than two words, returns
    ``(text, "")`` and the caller should fall back to single-line shrinking.
    """
    words = text.split()
    if len(words) < 2:
        return text, ""
    best_idx = 1
    best_delta = 10**9
    for i in range(1, len(words)):
        left = " ".join(words[:i])
        right = " ".join(words[i:])
        delta = abs(len(left) - len(right))
        if delta < best_delta:
            best_delta = delta
            best_idx = i
    return " ".join(words[:best_idx]), " ".join(words[best_idx:])


def _drawtext_single(text: str, size: int, y: int) -> str:
    esc = _escape_drawtext(text)
    return (
        f"drawtext=text='{esc}':"
        "expansion=none:"
        f"font={_TITLE_FONT_NAME}:"
        f"fontcolor=white:fontsize={size}:borderw=4:bordercolor=black:"
        f"x=(w-text_w)/2:y={y}"
    )


def _drawtext_two(line1: str, line2: str, size: int, y_top: int) -> str:
    """Two drawtext filters chained by comma — one ffmpeg filter chain, two lines."""
    esc1 = _escape_drawtext(line1)
    esc2 = _escape_drawtext(line2)
    y_bottom = y_top + int(round(size * _TITLE_LINE_SPACING_RATIO))
    return (
        f"drawtext=text='{esc1}':"
        "expansion=none:"
        f"font={_TITLE_FONT_NAME}:"
        f"fontcolor=white:fontsize={size}:borderw=4:bordercolor=black:"
        f"x=(w-text_w)/2:y={y_top},"
        f"drawtext=text='{esc2}':"
        "expansion=none:"
        f"font={_TITLE_FONT_NAME}:"
        f"fontcolor=white:fontsize={size}:borderw=4:bordercolor=black:"
        f"x=(w-text_w)/2:y={y_bottom}"
    )


def plan_title_drawtext(title_text: str, out_w: int = 1080) -> str | None:
    """Return the ``drawtext`` filter fragment for ``title_text`` or None to skip.

    The returned string is intended to be spliced into the main filtergraph
    between the ``[v_prepad]`` and ``[vout]`` labels by
    :func:`build_ffmpeg_cmd`. It does NOT include those labels itself.

    Backward compatibility: when the title fits on one line at the original
    72px size, the output is identical to the pre-P2 single-``drawtext``
    form (same x/y/fontsize/borderw), so golden ffmpeg tests stay green.
    """
    text = " ".join((title_text or "").split())
    if not text:
        return None
    usable_w = max(1, out_w - 2 * _TITLE_MARGIN_PX)

    if _title_fits(text, _TITLE_PRIMARY_SIZE, usable_w):
        return _drawtext_single(text, _TITLE_PRIMARY_SIZE, _TITLE_Y_TOP)

    line1, line2 = _wrap_title_two_lines(text)
    if line2:
        for size in (60, 52, _TITLE_MIN_SIZE):
            if _title_fits(line1, size, usable_w) and _title_fits(line2, size, usable_w):
                return _drawtext_two(line1, line2, size, _TITLE_Y_TOP)

    for size in (64, 56, 52, _TITLE_MIN_SIZE):
        if _title_fits(text, size, usable_w):
            return _drawtext_single(text, size, _TITLE_Y_TOP)

    max_chars = max(4, int(usable_w / _title_char_px(_TITLE_MIN_SIZE)))
    truncated = text[: max_chars - 1].rstrip() + "..."
    return _drawtext_single(truncated, _TITLE_MIN_SIZE, _TITLE_Y_TOP)


def _escape_filter_path(path: str) -> str:
    return path.replace("\\", "/").replace(":", "\\:").replace("'", "\\'")


def _has_audio_stream(media_path: str) -> bool:
    probe = shutil.which("ffprobe")
    if not probe:
        return False
    out = subprocess.run(
        [
            probe,
            "-v",
            "error",
            "-select_streams",
            "a:0",
            "-show_entries",
            "stream=codec_type",
            "-of",
            "csv=p=0",
            media_path,
        ],
        check=False,
        capture_output=True,
        text=True,
    )
    return out.returncode == 0 and "audio" in (out.stdout or "").lower()


def _build_concat_prefix(
    keep_ranges: list[tuple[float, float]],
    *,
    base_start_sec: float = 0.0,
    include_audio: bool,
) -> tuple[str, str, str | None]:
    video_parts: list[str] = []
    audio_parts: list[str] = []
    concat_inputs: list[str] = []

    for idx, (start, end) in enumerate(keep_ranges):
        abs_start = base_start_sec + start
        abs_end = base_start_sec + end
        video_label = f"v{idx}"
        video_parts.append(
            f"[0:v]trim=start={abs_start:.3f}:end={abs_end:.3f},setpts=PTS-STARTPTS[{video_label}]"
        )
        concat_inputs.append(f"[{video_label}]")
        if include_audio:
            audio_label = f"a{idx}"
            audio_parts.append(
                f"[0:a:0]atrim=start={abs_start:.3f}:end={abs_end:.3f},asetpts=PTS-STARTPTS[{audio_label}]"
            )
            concat_inputs.append(f"[{audio_label}]")

    concat_label = "[vclip][aclip]" if include_audio else "[vclip]"
    concat = (
        "".join(concat_inputs)
        + f"concat=n={len(keep_ranges)}:v=1:a={1 if include_audio else 0}{concat_label}"
    )
    parts = video_parts + audio_parts + [concat]
    return ";".join(parts), "vclip", "aclip" if include_audio else None


def build_ffmpeg_cmd(
    req: RenderRequest,
    *,
    src_w: int = 1920,
    src_h: int = 1080,
    include_audio: bool = True,
) -> list[str]:
    exe = _ensure_ffmpeg() if req.mode != "dry_run" else "ffmpeg"
    use_concat = bool(req.clip.keep_ranges_sec)
    input_label = "0:v"
    audio_label: str | None = None
    prefix = ""
    if use_concat:
        prefix, input_label, audio_label = _build_concat_prefix(
            list(req.clip.keep_ranges_sec),
            base_start_sec=req.clip.start_time_sec,
            include_audio=include_audio,
        )

    plan = plan_layout(
        req.layout,
        out_w=req.width,
        out_h=req.height,
        src_w=src_w,
        src_h=src_h,
        input_label=input_label,
    )
    fg = plan.filtergraph if not prefix else f"{prefix};{plan.filtergraph}"
    normalized_audio_label: str | None = None
    if include_audio:
        audio_input = f"[{audio_label}]" if audio_label is not None else "[0:a:0]"
        normalized_audio_label = "aout"
        fg = f"{fg};{audio_input}{_AUDIO_GAIN_FILTER}[{normalized_audio_label}]"

    # Skip the drawtext title overlay on split layouts: the top band already
    # shows a slide/chart with its own baked-in title, so adding an overlay
    # on top of that is pure noise (and was stacking over the chart title
    # in the SPLIT_CHART_PERSON Cathy Wood shorts).
    title_allowed = req.layout.layout not in SPLIT_LAYOUTS and req.layout.layout != LayoutKind.WIDE_VISUAL
    if req.title_text and title_allowed:
        # ``plan_title_drawtext`` returns a full filter fragment (possibly
        # two chained ``drawtext`` calls) that fits within the output width.
        # For short titles it is byte-identical to the pre-P2 single-line
        # form, keeping existing golden tests green while fixing the
        # "Prediction Markets vs Derivatives" edge-clip report.
        title_fragment = plan_title_drawtext(req.title_text, out_w=req.width)
        if title_fragment:
            fg = fg.replace(
                "[vout]",
                f"[v_prepad];[v_prepad]{title_fragment}[vout]",
            )

    if req.subtitle_path:
        subtitle_esc = _escape_filter_path(req.subtitle_path)
        # ``original_size`` pins libass's PlayResY to the actual output so
        # ``FontSize`` and ``MarginV`` are interpreted in output pixels. Without
        # this, libass defaults to PlayResY=288 and then upscales to the real
        # canvas (1920) -- blowing font sizes and pushing subtitles to the
        # middle of the frame. ``WrapStyle=0`` enables smart word wrap so long
        # lines break into readable stacks instead of running off-screen.
        fg = fg.replace(
            "[vout]",
            "[v_sub_in];"
            f"[v_sub_in]subtitles='{subtitle_esc}':"
            f"original_size={req.width}x{req.height}:"
            f"force_style='Fontname=Arial,"
            f"FontSize={req.subtitle_font_size},Alignment=2,"
            f"MarginV={req.subtitle_margin_v},MarginL=60,MarginR=60,"
            "WrapStyle=0,BorderStyle=4,"
            "BackColour=&H70000000&,PrimaryColour=&H00FFFFFF&,"
            "Outline=0,Shadow=0,Bold=1'[vout]",
        )

    start = req.clip.start_time_sec
    dur = max(0.1, req.clip.duration_sec)

    Path(Path(req.output_path).parent).mkdir(parents=True, exist_ok=True)

    cmd: list[str] = [exe, "-y"]
    if not use_concat:
        cmd.extend(["-ss", f"{start:.3f}", "-t", f"{dur:.3f}"])
    cmd.extend(
        [
            "-i",
            req.source_path,
            "-filter_complex",
            fg,
            "-map",
            "[vout]",
            "-c:v",
            "libx264",
            "-preset",
            "veryfast",
            "-crf",
            "20",
        ]
    )

    if include_audio:
        cmd.extend(
            [
                "-map",
                f"[{normalized_audio_label}]",
                "-c:a",
                "aac",
                "-b:a",
                "160k",
            ]
        )

    cmd.extend(["-movflags", "+faststart", req.output_path])
    return cmd


def probe_source_size(source_path: str) -> tuple[int, int]:
    exe = shutil.which("ffprobe")
    if not exe:
        return 1920, 1080
    out = subprocess.run(
        [
            exe,
            "-v",
            "error",
            "-select_streams",
            "v:0",
            "-show_entries",
            "stream=width,height",
            "-of",
            "csv=p=0",
            source_path,
        ],
        check=False,
        capture_output=True,
        text=True,
    )
    try:
        w, h = out.stdout.strip().split(",")
        return int(w), int(h)
    except Exception:
        return 1920, 1080


def render_clip(req: RenderRequest) -> RenderResult:
    try:
        src_w, src_h = probe_source_size(req.source_path) if req.mode != "dry_run" else (1920, 1080)
    except Exception:
        src_w, src_h = 1920, 1080

    include_audio = True
    if req.mode != "dry_run":
        include_audio = _has_audio_stream(req.source_path)
        if not include_audio:
            return RenderResult(
                clip_id=req.clip.clip_id,
                output_path=req.output_path,
                ffmpeg_cmd=[],
                success=False,
                error="Source media has no detectable audio stream (a:0).",
            )

    cmd = build_ffmpeg_cmd(req, src_w=src_w, src_h=src_h, include_audio=include_audio)

    if req.mode == "dry_run":
        return RenderResult(
            clip_id=req.clip.clip_id,
            output_path=req.output_path,
            ffmpeg_cmd=cmd,
            success=True,
        )
    try:
        subprocess.run(cmd, check=True, capture_output=True)
        if include_audio and not _has_audio_stream(req.output_path):
            return RenderResult(
                clip_id=req.clip.clip_id,
                output_path=req.output_path,
                ffmpeg_cmd=cmd,
                success=False,
                error="Rendered output is missing audio stream (a:0).",
            )
        return RenderResult(
            clip_id=req.clip.clip_id,
            output_path=req.output_path,
            ffmpeg_cmd=cmd,
            success=True,
        )
    except subprocess.CalledProcessError as e:
        return RenderResult(
            clip_id=req.clip.clip_id,
            output_path=req.output_path,
            ffmpeg_cmd=cmd,
            success=False,
            error=e.stderr.decode("utf-8", errors="replace")[-4000:] if e.stderr else str(e),
        )


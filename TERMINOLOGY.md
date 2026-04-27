# TERMINOLOGY — what every word in this repo actually means

Bryan asked for the full glossary of every term used across the pipeline
so there's no ambiguity about "subject", "cropped", "layout", etc. This
file is the single source of truth. If a term appears in code or docs
and isn't here, it's a bug in this file — please add it.

Everything below is oriented around one sentence of product law:

> **A short shows at most TWO on-screen items. An item is either a
> `person` or a `chart`. That's it.**

---

## Core vocabulary

### Source
The original long video (YouTube podcast, keynote, etc.). We do **not**
modify the source; it's read-only input to the pipeline. Usually 16:9
(1920×1080) but any resolution works — the pipeline probes it.

### Short / Clip
The 9:16 vertical video we produce (1080×1920, a.k.a. TikTok / Reels /
Shorts format). The word **clip** is used for the *timing window* cut
out of the source (start time → end time + transcript slice); the word
**short** is used for the finished rendered MP4.

### Item
Something that occupies space on screen. There are exactly **two** kinds:

- **`person`** — a human speaker. Face + shoulders/upper body.
- **`chart`** — anything graphical that carries information: a slide,
  a line/bar/scatter graph, an infographic, a screenshare, a diagram.

The hard rule: a single short never shows more than **two** items.

### Subject
Informal synonym for "the person item" — "the subject is the
speaker". We use `person` in code and schemas; "subject" in prose only.

### Layout
A recipe for placing up to two items into the 9:16 output. There are
exactly five layouts, named by what they show:

| `LayoutKind`           | Item(s)         | Look                                |
| ---------------------- | --------------- | ----------------------------------- |
| `ZOOM_CALL_CENTER`     | 1 person        | Tight webcam / video-call headshot. |
| `SIT_CENTER`           | 1 person        | Interview / seated framing.         |
| `SPLIT_CHART_PERSON`   | 1 chart + 1 person | Chart top, speaker bottom.       |
| `SPLIT_TWO_PERSONS`    | 2 persons       | Two speakers stacked top/bottom.    |
| `SPLIT_TWO_CHARTS`     | 2 charts        | Two charts stacked top/bottom.      |

"Split" means the 9:16 frame is divided into a **top band** and a
**bottom band**. The bands always stack vertically — we never stack
horizontally in 9:16. Splits are the three bottom entries above.

### Bounding box (bbox)
A rectangle identifying *where* an item sits in the source frame. We
always use **normalized** coordinates — `x1/y1/x2/y2` are all in `[0, 1]`
— so the same bbox is valid whether the source is 1920×1080 or 3840×2160.
`(x1, y1)` is the top-left corner, `(x2, y2)` is the bottom-right corner.

The relevant bboxes on a `LayoutInstruction`:

| Field                          | What it points at                                  |
| ------------------------------ | -------------------------------------------------- |
| `split_person_region`          | The (first) person in a split layout.              |
| `split_second_person_region`   | The second person in `SPLIT_TWO_PERSONS`.          |
| `split_chart_region`           | The (first) chart in a split layout.               |
| `split_second_chart_region`    | The second chart in `SPLIT_TWO_CHARTS`.            |

### Crop
The operation of cutting a rectangular window out of a frame. Every
layout starts with one or more crops of the source. In ffmpeg speak,
`crop=CW:CH:X:Y`:
- `CW`, `CH` — crop width/height in source pixels.
- `X`, `Y` — top-left corner of the crop in source pixels.

"Cropped" = "cut down to a smaller rectangle". This is not the same as
*scaling* (resizing) — crop just throws pixels away.

### Scale
Resizing (zooming/shrinking) a frame. In ffmpeg: `scale=W:H`. After the
crop we scale the surviving pixels up or down to fill the target band.

### Strip
A **source-side** crop produced for a split layout — i.e. "the chart
strip" is the rectangle we cut out of the source that contains the
chart. Each strip is destined for one **band** in the output.

### Band
An **output-side** horizontal slab of the 9:16 frame in a split layout.
Every split layout has exactly two bands: a **top band** and a **bottom
band**. Heights sum to 1920; the split between them is controlled by
`top_band_ratio`.

### Seam
The vertical line in the source frame that separates the two strips in
a split layout. The seam is chosen so the two strips are
**complementary** (no overlap, no gap). When both bboxes are known, the
seam sits at the midpoint between `left.x2` and `right.x1`; otherwise
it falls back to a 50/50 centerline (two-of-a-kind splits) or 2/3|1/3
(legacy chart/person fallback).

### Cover vs contain (scale modes)
When a source strip doesn't exactly match its target band's aspect
ratio, we have to decide: **cover** (fill the band, crop the overflow —
no letterbox bars) or **contain** (fit entirely, pad the leftover — has
bars). The split layouts always use **cover**:
`force_original_aspect_ratio=increase` + a second `crop` to remove the
overflow. That guarantees bands are painted edge to edge.

### `top_band_ratio`
The fraction of 9:16 output height used by the top band. `0.5` means
the top and bottom bands are equal (the symmetric "even" look Bryan
asked for). `0.6` means 60% top / 40% bottom (the pre-fix default).
Range is `[0.2, 0.8]`.

### `focus_stack_order`
For `SPLIT_CHART_PERSON`, which item goes in the top band vs the bottom
band:

- `chart_then_person` (default) — chart on top, person on bottom.
- `person_then_chart` — person on top, chart on bottom.

Has no effect on the other split layouts (both bands hold the same
kind of item).

### `zoom`
Multiplier ≥ 1 that tightens a centered-single-subject crop. `zoom=2`
means the crop window is half the size in each axis, i.e. 4× smaller by
area — a tighter close-up. Only meaningful on `ZOOM_CALL_CENTER` and
`SIT_CENTER`.

### `person_x_norm` / `chart_x_norm`
Legacy single-value knobs (kept for non-vision fallbacks):

- `person_x_norm` ∈ `[0, 1]` — normalized x-center of the speaker in
  the source frame. Drives centered crops when no bbox is available.
- `chart_x_norm` ∈ `[0, 1]` — normalized left-edge trim for the chart
  strip in `SPLIT_CHART_PERSON` when no `split_chart_region` bbox is
  available.

If you have a bbox, use a bbox — it's strictly more information.

### Subtitle / Caption
The on-screen text of what the speaker is saying, timed word-by-word.
We render subtitles with ffmpeg's `subtitles` filter (which uses libass
internally) from an `.ass` (Advanced SubStation Alpha) file we
generate from the transcript. An SRT path is still accepted end-to-end
for backward compatibility, but the product pipeline emits ASS because
it's the only format where we control `PlayResY` directly (see below).

### libass `PlayResY` / `original_size`
libass is the subtitle renderer inside ffmpeg. By default it assumes a
`PlayResY` of 288 and then *scales* every styling value (font size,
margins, outline) up to the real output resolution — which is why a
"font size 20" request came out 6× too large and a "MarginV 120" came
out in the middle of the frame. We fix this two ways, belt-and-braces:

1. We write the ASS file with `PlayResY = output_height` (1920). That
   makes libass's internal unit equal to one output pixel, so
   `FontSize=48` really is 48 output pixels tall and `MarginV=160`
   really is 160 output pixels above the bottom edge.
2. We pass `original_size=1080x1920` to the `subtitles` filter. That
   tells libass to match its coordinate system to the output canvas
   instead of falling back to 288 when the ASS header is missing.

With both in place, every knob in `RenderRequest.subtitle_*` is in
honest output pixels.

### Title overlay
An optional `drawtext`-rendered headline baked on top of the rendered
frame (used for `suggested_overlay_title` from the LLM). We **suppress**
the title overlay on every split layout because the source already
contains a chart/slide with its own title, and stacking another
title on top just obscures content.

### Transcript
The word-level ASR output: `[{word, start_time, end_time}, ...]`. Used
both to drive clip selection (via an LLM that reads the text) and to
generate subtitle cues.

### Keyframe
One still image extracted per scene or per clip. The layout-vision
stage sends keyframes to Gemini and asks "which layout should this be
and where are the items?" — getting back both a `LayoutKind` and the
bboxes described above.

### Work dir
A per-video cache directory holding everything intermediate: the
downloaded `source.mp4`, `transcript.json`, `clips.json`, keyframes,
`layout_vision.json`, subtitles, etc. Lets us re-run cheaply.

---

## Pipeline stage vocabulary

| Stage            | Input                          | Output                              |
| ---------------- | ------------------------------ | ----------------------------------- |
| **Ingest**       | YouTube URL / mp4              | `source.mp4`, `transcript.json`, keyframes |
| **Clip Selection** | Transcript                   | `clips.json` (`ClipPlan`)           |
| **Layout Vision** | Clip keyframes                | `layout_vision.json` (`LayoutInstruction` per clip) |
| **Render**       | Source + clip + layout + ass   | `short_<id>.mp4`                    |

Each stage reads/writes strict JSON artifacts. That's why we can
re-render without re-downloading, re-select without re-transcribing,
and so on.

---

## Units

| Quantity                     | Unit                                               |
| ---------------------------- | -------------------------------------------------- |
| All bbox coordinates         | Normalized `[0, 1]` in source frame.                |
| `zoom`, `top_band_ratio`     | Dimensionless ratios.                               |
| `FontSize`, `MarginV`        | Output pixels (thanks to `PlayResY=1920`).          |
| Clip times, transcript times | Seconds on the **source** timeline.                 |
| Subtitle times inside an ASS | Seconds on the **clip** timeline (t=0 at cut-in).   |
| ffmpeg `crop=CW:CH:X:Y`      | Source pixels.                                      |

If a field's units aren't in this table, it's almost certainly
normalized or seconds — ask the schema, not me.

# Target Video Analysis

Canonical source:

- URL: `https://www.youtube.com/watch?v=PdVv_vLkUgk`
- Title: *ITK with Cathie Wood: Is AI Winning The War On Inflation?*
- Length: `46:23`
- Channel: `ARK Invest`
- Published: `2026-04-09`

## Why this is the canonical demo

This repo exists to turn long-form, chart-heavy video into vertical shorts. This source hits every important path:

1. Stable talking-head sections.
2. Repeated chart-led sections that must preserve slide geometry.
3. Clear topic shifts that make clip selection and hook detection easy to inspect.

| Chapter | Time  | Topic                                     | Expected layout |
|---------|-------|-------------------------------------------|-----------------|
| 0 | 00:00 | ARK x Kalshi Partnership | `sit_center` |
| 1 | 02:30 | A New Era For Active Investing | `sit_center` |
| 2 | 04:30 | A Multi-Trillion-Dollar Opportunity | `sit_center` |
| 3 | 06:30 | War Disrupts Deficit Progress | `split_chart_person` |
| 4 | 14:15 | Dollar Strength Catches Markets Off Guard | `split_chart_person` |
| 5 | 23:30 | Productivity Boom Is Closer Than Expected | `split_chart_person` |
| 6 | 30:00 | Inflation Pressures Continue To Crack | `split_chart_person` |
| 7 | 42:30 | Credit Markets Show No Signs Of Stress | `sit_center` |
| 8 | 43:30 | Innovation Could Power The Next Bull Market | `sit_center` |

## Why Stage 3 matters here

Transcript-only logic cannot see when a chart becomes the real visual subject. For this source, the important behavior is:

- Stage 2 selects promising windows from transcript alone.
- Stage 3 samples multiple frames per clip and asks the configured multimodal model to choose a render-safe layout.
- The runtime converts model-facing `0..1000` boxes into the internal normalized schema before render.

If Stage 3 regresses, chart-led clips collapse back toward talking-head framing and the demo quality drops immediately.

## Suggested verification commands

Full run:

```bash
uv run contentflow run long-to-shorts "https://www.youtube.com/watch?v=PdVv_vLkUgk"
```

Layout-only rerun on an existing work dir:

```bash
uv run contentflow run long-to-shorts --work-dir .contentflow_work --start-at layout-vision --force-layout-vision
```

## Expected output profile

| Clip type | Source chapters | Expected count | Expected layout |
|-----------|-----------------|---------------:|-----------------|
| Hook / opener | 0, 2 | 1 | `sit_center` |
| Chart reveals | 3, 4, 5, 6 | 2-3 | `split_chart_person` |
| Payoff / close | 8 | 1 | `sit_center` |

Target output: `4-5` shorts, roughly `50-90s`, with burned subtitles and chart-preserving split layouts where the slide is the point.

## Regression value

This remains the canonical regression source for the Stage 3 split-layout failure fixed on `2026-04-22`. The bad path was failed frame sampling followed by an unsafe `sit_center` fallback. The corrected path preserves `split_chart_person` on chart-led clips.

"""CLI entry point for Afterform."""

from __future__ import annotations

import argparse
import json
import logging
import sys
from collections.abc import Mapping, Sequence
from datetime import datetime
from pathlib import Path

from afterform.config import PipelineConfig
from afterform.flows.long_to_shorts.flow import run_pipeline
from afterform.flows.long_to_shorts.stage_inspection import (
    STAGE_ORDER,
    build_stage_inspection,
    normalize_stage,
    write_inspection,
)

_LOG_TRIM_EDGE = 20
_LOG_TRIM_THRESHOLD = 512
_LOG_SPECIAL_KEYS = {"image_url", "url", "data", "b64_json", "base64"}


def _trim_log_string(value: str, *, edge: int = _LOG_TRIM_EDGE) -> str:
    if len(value) <= edge * 2:
        return value
    return f"{value[:edge]}...{value[-edge:]} [len={len(value)}]"


def _sanitize_log_value(value: object, *, key: str | None = None) -> object:
    if isinstance(value, Mapping):
        return {str(k): _sanitize_log_value(v, key=str(k)) for k, v in value.items()}
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        return [_sanitize_log_value(item) for item in value]
    if isinstance(value, (bytes, bytearray)):
        return f"<bytes len={len(value)}>"
    if hasattr(value, "model_dump") and callable(getattr(value, "model_dump")):
        try:
            return _sanitize_log_value(value.model_dump(), key=key)
        except Exception:
            return repr(value)
    if value is None or isinstance(value, (bool, int, float)):
        return value
    if not isinstance(value, str):
        return repr(value)

    key_name = (key or "").lower()
    should_trim = (
        value.startswith("data:")
        or (key_name in _LOG_SPECIAL_KEYS and len(value) > (_LOG_TRIM_EDGE * 2))
        or len(value) > _LOG_TRIM_THRESHOLD
    )
    if should_trim:
        return _trim_log_string(value)
    return value


def _coerce_log_argument(value: object) -> object:
    sanitized = _sanitize_log_value(value)
    if isinstance(sanitized, (dict, list)):
        return json.dumps(sanitized, indent=2, ensure_ascii=False)
    return sanitized


class _PrettyJsonLogFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        rendered = logging.makeLogRecord(record.__dict__.copy())

        if rendered.args:
            if isinstance(rendered.args, Mapping):
                if "%(" in str(rendered.msg):
                    rendered.args = {
                        key: _coerce_log_argument(value) for key, value in rendered.args.items()
                    }
                else:
                    rendered.args = (
                        json.dumps(
                            _sanitize_log_value(rendered.args),
                            indent=2,
                            ensure_ascii=False,
                        ),
                    )
            elif isinstance(rendered.args, tuple):
                rendered.args = tuple(_coerce_log_argument(value) for value in rendered.args)
            else:
                rendered.args = _coerce_log_argument(rendered.args)
        elif isinstance(rendered.msg, (dict, list)):
            rendered.msg = json.dumps(
                _sanitize_log_value(rendered.msg),
                indent=2,
                ensure_ascii=False,
            )
        elif isinstance(rendered.msg, str):
            stripped = rendered.msg.strip()
            if stripped.startswith(("{", "[")):
                try:
                    parsed = json.loads(stripped)
                except json.JSONDecodeError:
                    rendered.msg = _sanitize_log_value(rendered.msg)
                else:
                    rendered.msg = json.dumps(
                        _sanitize_log_value(parsed),
                        indent=2,
                        ensure_ascii=False,
                    )
            else:
                rendered.msg = _sanitize_log_value(rendered.msg)

        return super().format(rendered)


def setup_logging(verbose: bool = False) -> None:
    """Configure logging for the public ``afterform`` CLI."""
    level = logging.DEBUG if verbose else logging.INFO
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(
        _PrettyJsonLogFormatter(
            fmt="%(asctime)s | %(levelname)-7s | %(message)s",
            datefmt="%H:%M:%S",
        )
    )
    logging.basicConfig(
        level=level,
        handlers=[handler],
        force=True,
    )
    logging.getLogger("urllib3").setLevel(logging.WARNING)
    logging.getLogger("httpx").setLevel(logging.WARNING)


def config_default_filled_pause_pruning(flag_value: bool | None) -> bool:
    if flag_value is None:
        field = PipelineConfig.__dataclass_fields__["filled_pause_pruning"]
        return bool(field.default)
    return flag_value


def build_parser() -> argparse.ArgumentParser:
    """Build the ``afterform`` CLI parser."""
    parser = argparse.ArgumentParser(
        prog="afterform",
        description="Afterform - media transformation platform with explicit flows",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  afterform run long-to-shorts "https://youtube.com/watch?v=abc123"
  afterform run long-to-shorts "https://youtube.com/watch?v=abc123" --work-dir .afterform_work
  afterform run long-to-shorts "https://youtube.com/watch?v=abc123" --run-dir .afterform/runs/run_001
  afterform run long-to-shorts "https://youtube.com/watch?v=abc123" --llm-model gemini-2.0-flash
        """,
    )
    parser.add_argument(
        "command",
        nargs="?",
        choices=["run"],
        help="Primary action. Use `run` to execute a flow.",
    )
    parser.add_argument(
        "flow",
        nargs="?",
        choices=["long-to-shorts"],
        help="Flow to execute.",
    )
    parser.add_argument(
        "source",
        nargs="?",
        default=None,
        help="Source input for the selected flow. For `long-to-shorts`, this is a YouTube URL.",
    )
    parser.add_argument(
        "--output",
        "-o",
        type=Path,
        default=Path("output"),
        help="Output directory for final shorts. Defaults to <run-dir>/output for normal URL runs.",
    )
    parser.add_argument(
        "--work-dir",
        type=Path,
        default=None,
        help="Working directory for intermediate files. Default: per-video folder under the cache root.",
    )
    parser.add_argument(
        "--run-dir",
        type=Path,
        default=None,
        help=(
            "Parent directory for one isolated run. Expands to "
            "<run-dir>/work for intermediates and <run-dir>/output for final shorts."
        ),
    )
    parser.add_argument(
        "--no-video-cache",
        action="store_true",
        help="Do not use per-video cache dirs; use the isolated run work dir unless --work-dir is set.",
    )
    parser.add_argument(
        "--cache-root",
        type=Path,
        default=None,
        help="Override cache root for manifests and per-video ingest (env: AFTERFORM_CACHE_ROOT).",
    )
    parser.add_argument(
        "--llm-provider",
        choices=["gemini", "openai", "azure"],
        default=None,
        help=(
            "LLM provider for stages 2/2.25/2.5/3 "
            "(default: AFTERFORM_LLM_PROVIDER env or gemini)."
        ),
    )
    parser.add_argument(
        "--llm-model",
        "--gemini-model",
        dest="llm_model",
        default=None,
        help="Model/deployment id for stages 2/2.25/2.5. Legacy alias: --gemini-model.",
    )
    parser.add_argument(
        "--force-clip-selection",
        action="store_true",
        help="Re-run clip selection even when clips.meta.json matches the transcript.",
    )
    parser.add_argument(
        "--clip-mode",
        choices=["curated", "exhaustive"],
        default="curated",
        help=(
            "Clip-selection policy. curated keeps the threshold/floor/cap behavior; "
            "exhaustive keeps every distinct clip above the quality bar."
        ),
    )
    parser.add_argument(
        "--clip-candidate-count",
        type=int,
        default=12,
        help="Candidate pool size requested from the clip-selection model (default: 12).",
    )
    parser.add_argument(
        "--clip-quality-threshold",
        type=float,
        default=0.70,
        help="Minimum clip quality score for threshold selection (default: 0.70).",
    )
    parser.add_argument(
        "--max-clips",
        default=None,
        help=(
            "Maximum clips to keep. Use 'none' for no arbitrary cap. "
            "Defaults: 8 in curated mode, none in exhaustive mode."
        ),
    )
    parser.add_argument(
        "--review-only-clips",
        action="store_true",
        help="Stop after clip selection so the discovered clip set can be reviewed before render.",
    )
    parser.add_argument(
        "--llm-vision-model",
        "--gemini-vision-model",
        dest="llm_vision_model",
        default=None,
        help="Optional separate model/deployment id for stage 3 layout planning.",
    )
    parser.add_argument(
        "--force-layout-vision",
        action="store_true",
        help="Re-run layout planning even when layout_vision.meta.json matches.",
    )
    parser.add_argument(
        "--prune-level",
        choices=["off", "conservative", "balanced", "aggressive"],
        default="balanced",
        help=(
            "Stage 2.5 inner-clip content pruning aggressiveness. "
            "'off' skips pruning entirely; 'conservative' trims <=10%%, "
            "'balanced' <=20%%, 'aggressive' <=35%% of each clip."
        ),
    )
    parser.add_argument(
        "--force-content-pruning",
        action="store_true",
        help="Re-run content pruning even when prune.meta.json matches.",
    )
    parser.add_argument(
        "--filled-pause-pruning",
        dest="filled_pause_pruning",
        action="store_true",
        default=None,
        help="Enable audio-model removal of filled pauses such as um/uh/hmm (default: on).",
    )
    parser.add_argument(
        "--no-filled-pause-pruning",
        dest="filled_pause_pruning",
        action="store_false",
        help="Disable filled-pause pruning for this run.",
    )
    parser.add_argument(
        "--require-filled-pause-pruning",
        action="store_true",
        help="Fail if --filled-pause-pruning cannot run because optional models are unavailable.",
    )
    parser.add_argument(
        "--no-hook-detection",
        action="store_true",
        help="Skip Stage 2.25 hook detection and carry through the existing hook window.",
    )
    parser.add_argument(
        "--force-hook-detection",
        action="store_true",
        help="Re-run hook detection even when hooks.meta.json matches.",
    )
    parser.add_argument(
        "--start-at",
        choices=STAGE_ORDER,
        default=None,
        help="Start the flow at this stage using cached artifacts from --work-dir.",
    )
    parser.add_argument(
        "--stop-after",
        choices=STAGE_ORDER,
        default=None,
        help="Stop the flow after this stage instead of rendering through the end.",
    )
    parser.add_argument(
        "--inspect-stage",
        choices=STAGE_ORDER,
        default=None,
        help="Write a stable JSON inspection file for one stage.",
    )
    parser.add_argument(
        "--clip-id",
        default=None,
        help="Optional clip id filter for --inspect-stage (for example 003).",
    )
    parser.add_argument(
        "--clean-run",
        action="store_true",
        help=(
            "Run with a fresh work dir and no cache reuse. Implies --no-video-cache, "
            "--force-clip-selection, --force-layout-vision, and overwrite existing outputs."
        ),
    )
    parser.add_argument(
        "--subtitle-font-size",
        type=int,
        default=48,
        help="Caption font size in output pixels (default: 48)",
    )
    parser.add_argument(
        "--subtitle-margin-v",
        type=int,
        default=160,
        help="Caption bottom margin in output pixels (default: 160)",
    )
    parser.add_argument(
        "--subtitle-max-words",
        type=int,
        default=4,
        help="Max words per subtitle cue (default: 4)",
    )
    parser.add_argument(
        "--subtitle-max-cue-sec",
        type=float,
        default=2.2,
        help="Max subtitle cue duration in seconds (default: 2.2)",
    )
    parser.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        help="Enable debug logging",
    )
    return parser


def main(argv: list[str] | None = None) -> None:
    """CLI entry point."""
    parser = build_parser()
    args = parser.parse_args(argv)
    setup_logging(args.verbose)

    if args.command is not None and args.flow != "long-to-shorts":
        parser.error("The `run` command currently supports only the `long-to-shorts` flow.")

    source = args.source
    inspect_only = (
        normalize_stage(args.inspect_stage) is not None
        and normalize_stage(args.start_at) is None
        and normalize_stage(args.stop_after) is None
    )

    if (
        args.command is None
        and not inspect_only
        and args.work_dir is None
    ):
        parser.error("Use `afterform run long-to-shorts <url>`.")

    use_video_cache = not args.no_video_cache
    force_clip_selection = args.force_clip_selection
    force_layout_vision = args.force_layout_vision
    force_content_pruning = args.force_content_pruning
    force_hook_detection = args.force_hook_detection
    detect_hooks = not args.no_hook_detection
    overwrite_outputs = False
    work_dir = args.work_dir
    output_dir = args.output
    stop_after = args.stop_after
    max_clips = 8
    run_dir = args.run_dir
    auto_run_dir = False

    if run_dir is not None:
        if args.work_dir is not None:
            parser.error("--run-dir cannot be combined with --work-dir.")
        if args.output != Path("output"):
            parser.error("--run-dir cannot be combined with an explicit --output.")
        work_dir = run_dir / "work"
        output_dir = run_dir / "output"

    if args.clip_candidate_count <= 0:
        parser.error("--clip-candidate-count must be greater than 0.")
    if not 0.0 <= args.clip_quality_threshold <= 1.0:
        parser.error("--clip-quality-threshold must be between 0 and 1.")
    if args.max_clips is None and args.clip_mode == "exhaustive":
        max_clips = None
    elif args.max_clips is not None:
        if str(args.max_clips).strip().lower() == "none":
            max_clips = None
        else:
            try:
                max_clips = int(args.max_clips)
            except ValueError:
                parser.error("--max-clips must be a positive integer or 'none'.")
            if max_clips <= 0:
                parser.error("--max-clips must be a positive integer or 'none'.")
    if args.review_only_clips:
        if stop_after is not None and normalize_stage(stop_after) != "clip-selection":
            parser.error("--review-only-clips cannot be combined with --stop-after other than clip-selection.")
        stop_after = "clip-selection"

    if work_dir is None and run_dir is None and args.output == Path("output") and not inspect_only:
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        run_dir = Path(".afterform") / "runs" / f"run_{stamp}"
        output_dir = run_dir / "output"
        auto_run_dir = True

    if args.clean_run:
        use_video_cache = False
        force_clip_selection = True
        force_layout_vision = True
        force_content_pruning = True
        force_hook_detection = True
        overwrite_outputs = True

    if work_dir is None and run_dir is not None and auto_run_dir and not use_video_cache and source is not None:
        work_dir = run_dir / "work"

    config = PipelineConfig(
        youtube_url=source,
        output_dir=output_dir,
        run_dir=run_dir
        if run_dir is not None
        else (work_dir.parent if work_dir and work_dir.name == "work" else None),
        work_dir=work_dir,
        use_video_cache=use_video_cache,
        cache_root=args.cache_root,
        llm_provider=args.llm_provider,
        llm_model=args.llm_model,
        llm_vision_model=args.llm_vision_model,
        force_clip_selection=force_clip_selection,
        force_layout_vision=force_layout_vision,
        clean_run=args.clean_run,
        overwrite_outputs=overwrite_outputs,
        prune_level=args.prune_level,
        force_content_pruning=force_content_pruning,
        filled_pause_pruning=(
            True
            if args.require_filled_pause_pruning
            else config_default_filled_pause_pruning(args.filled_pause_pruning)
        ),
        require_filled_pause_pruning=args.require_filled_pause_pruning,
        clip_selection_mode=args.clip_mode,
        clip_selection_candidate_count=args.clip_candidate_count,
        clip_selection_quality_threshold=args.clip_quality_threshold,
        clip_selection_max_kept=max_clips,
        detect_hooks=detect_hooks,
        force_hook_detection=force_hook_detection,
        subtitle_font_size=args.subtitle_font_size,
        subtitle_margin_v=args.subtitle_margin_v,
        subtitle_max_words_per_cue=args.subtitle_max_words,
        subtitle_max_cue_sec=args.subtitle_max_cue_sec,
        start_at=args.start_at,
        stop_after=stop_after,
        inspect_stage=args.inspect_stage,
        clip_id=args.clip_id,
    )

    try:
        if config.youtube_url is None and config.work_dir is None:
            parser.error("--work-dir is required when the source URL is omitted.")
        if (
            not inspect_only
            and config.youtube_url is None
            and normalize_stage(args.start_at) in {None, "ingest"}
        ):
            parser.error("A source URL is required when the run includes Stage 1 ingest.")

        if inspect_only:
            assert config.work_dir is not None
            payload = build_stage_inspection(
                config.work_dir,
                stage=normalize_stage(args.inspect_stage),
                clip_id=config.clip_id,
                config=config,
            )
            path = write_inspection(
                config.work_dir,
                stage=normalize_stage(args.inspect_stage),
                payload=payload,
                clip_id=config.clip_id,
            )
            print(f"Inspection written: {path}")
            return

        outputs = run_pipeline(config)
        if normalize_stage(stop_after) and normalize_stage(stop_after) != "render":
            print(f"\nDone. Stopped after: {stop_after}")
        else:
            print(f"\nDone. {len(outputs)} shorts generated in: {config.output_dir}")
            for output_path in outputs:
                print(f"   -> {output_path}")
    except KeyboardInterrupt:
        print("\nFlow interrupted.")
        sys.exit(1)
    except Exception as exc:
        logging.getLogger(__name__).error("Flow failed: %s", exc, exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()


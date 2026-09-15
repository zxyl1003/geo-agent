"""Run a single-image geolocation workflow."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from geoagent.core.config import load_app_config
from geoagent.core.exceptions import ImageInputError
from geoagent.core.image_validation import validate_image_input
from geoagent.core.logging import setup_logging
from geoagent.tools.registry import build_tools
from geoagent.workflows.react_workflow import ReactWorkflow


def configure_logging(app_config, debug: bool, colorize: bool) -> None:
    logging_config = app_config.system.get("logging", {})
    file_root = Path(logging_config.get("file_root", "logs"))
    if not file_root.is_absolute():
        file_root = ROOT / file_root
    setup_logging(
        app_config.log_level,
        logging_config.get("format"),
        debug=debug,
        colorize=colorize,
        file_enabled=logging_config.get("file_enabled", True),
        file_root=file_root,
        file_name=logging_config.get("file_name", "log.txt"),
    )


def build_workflow(debug: bool = False, colorize: bool = True, memory_mode: str = "off", memory_dir: str = ""):
    app_config = load_app_config(config_dir=ROOT / "configs", env_file=ROOT / ".env")
    if memory_mode != "config":
        memory_config = app_config.system.setdefault("memory", {})
        memory_config["enabled"] = memory_mode != "off"
        if memory_mode != "off":
            if not memory_dir:
                raise ValueError("--memory-dir is required when memory is enabled.")
            memory_config["mode"] = memory_mode
            memory_config["storage_dir"] = memory_dir
    if debug:
        app_config.log_level = "DEBUG"
    configure_logging(app_config, debug=debug, colorize=colorize)
    tools = build_tools(app_config)
    return ReactWorkflow(
        app_config=app_config,
        tools=tools,
        workflow_config=app_config.workflows["react"],
    )


def print_state_summary(state) -> None:
    print("\n=== visual cues ===")
    for cue in state.visual_cues:
        confidence = f"{cue.confidence:.2f}" if cue.confidence is not None else "unknown"
        print(f"- [{cue.cue_type}] {cue.text} ({confidence})")

    print("\n=== hypotheses ===")
    for hyp in state.hypotheses:
        region = f", {hyp.region}" if hyp.region else ""
        print(f"- {hyp.name}, {hyp.country or 'unknown'}{region} | granularity={hyp.granularity} | score={hyp.score:.3f} | {hyp.rationale}")

    print("\n=== selected tools ===")
    selected = [call.tool_name for call in state.tool_calls]
    print(", ".join(selected) if selected else "none")

    print("\n=== recalled memories ===")
    recalled_ids = [
        str(item.get("memory_id"))
        for item in (state.metadata.get("recalled_memories") or [])
        if isinstance(item, dict) and item.get("memory_id")
    ]
    print(", ".join(recalled_ids) if recalled_ids else "none")

    print("\n=== tool results ===")
    for result in state.tool_results:
        compact_data = result.data
        print(
            json.dumps(
                {"tool": result.tool_name, "success": result.success, "error": result.error, "data": compact_data},
                ensure_ascii=False,
                default=str,
            )
        )

    print("\n=== final answer ===")
    print(json.dumps(state.final_answer.model_dump() if state.final_answer else None, ensure_ascii=False, indent=2, default=str))

    print("\n=== confidence ===")
    print(f"{state.uncertainty.confidence:.3f}")

    print("\n=== granularity ===")
    print(state.final_answer.granularity if state.final_answer else "unknown")

    print("\n=== uncertainty radius ===")
    print(state.uncertainty.uncertainty_radius_m)

    print("\n=== tool trace ===")
    for call in state.tool_calls:
        print(f"- step={call.step} tool={call.tool_name} status={call.status}")

    print("\n=== resource usage ===")
    print(
        json.dumps(
            {
                "token_usage": state.token_usage,
                "api_call_count": state.api_call_count,
                "tool_call_count": state.tool_call_count,
                "internal_tool_call_count": state.metadata.get("internal_tool_call_count") or {},
            },
            ensure_ascii=False,
            indent=2,
            default=str,
        )
    )


def build_task_metadata(args) -> dict | None:
    """Assemble ground-truth metadata for memory evaluation and learning."""

    metadata: dict = {}

    ground_truth: dict = {}
    if args.gt_lat is not None or args.gt_lon is not None:
        if args.gt_lat is None or args.gt_lon is None:
            raise ValueError("Provide both --gt-lat and --gt-lon, or neither.")
        ground_truth["lat"] = args.gt_lat
        ground_truth["lon"] = args.gt_lon
    if args.gt_city:
        ground_truth["city"] = args.gt_city
    if args.gt_country:
        ground_truth["country"] = args.gt_country
    if ground_truth:
        metadata["ground_truth"] = ground_truth

    return metadata or None


def main() -> None:
    parser = argparse.ArgumentParser(description="Run geo-agent localization for one image.")
    parser.add_argument(
        "--image-path", default=None
    )
    parser.add_argument("--query", default="这张照片的地理位置是哪里？", help="User query about the image.")
    parser.add_argument("--debug", action="store_true", help="Stream model output and print colored step/tool traces.")
    parser.add_argument("--no-color", action="store_true", help="Disable ANSI colors for copyable logs.")
    parser.add_argument(
        "--memory-mode",
        choices=["off", "config", "retrieve_only", "learn_only", "full"],
        default="off",
        help="External experience memory mode. Default keeps baseline unchanged.",
    )
    parser.add_argument(
        "--memory-dir",
        default="",
        help="Directory for SQLite/Chroma memory files; required when memory is enabled.",
    )
    # Ground truth (optional). Provide lat+lon for coordinate-level scoring, or
    # city/country for coarse scoring. When present, success is computed
    # automatically (feedback_type=ground_truth).
    parser.add_argument("--gt-lat", type=float, default=None, help="Ground-truth latitude (WGS84).")
    parser.add_argument("--gt-lon", type=float, default=None, help="Ground-truth longitude (WGS84).")
    parser.add_argument("--gt-city", default=None, help="Ground-truth city/region name for coarse scoring.")
    parser.add_argument("--gt-country", default=None, help="Ground-truth country name for coarse scoring.")
    args = parser.parse_args()

    try:
        validate_image_input(args.image_path)
    except ImageInputError as exc:
        parser.error(str(exc))

    try:
        task_metadata = build_task_metadata(args)
    except ValueError as exc:
        parser.error(str(exc))

    workflow = build_workflow(
        debug=args.debug,
        colorize=not args.no_color,
        memory_mode=args.memory_mode,
        memory_dir=args.memory_dir,
    )
    state = workflow.run(
        input_image_path=args.image_path,
        user_query=args.query,
        task_metadata=task_metadata,
    )
    print_state_summary(state)
    if task_metadata:
        print("\n=== memory update ===")
        print(json.dumps(state.metadata.get("memory_update") or state.metadata.get("memory_error") or "no memory update recorded (check --memory-mode)", ensure_ascii=False, indent=2, default=str))


if __name__ == "__main__":
    main()

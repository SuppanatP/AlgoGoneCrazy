"""Run ingestion, aggregation, backfill, and consolidation sequentially."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from market_data.orchestration import (  # noqa: E402
    load_workflow_config,
    run_market_data_workflow,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Run all market-data stages sequentially from one JSON "
            "configuration. A failed stage prevents every later stage."
        )
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("config/market_data_pipeline.json"),
        help=(
            "Workflow JSON file "
            "(default: config/market_data_pipeline.json)."
        ),
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    try:
        config = load_workflow_config(args.config)
        run_market_data_workflow(config)
    except Exception as error:
        print(f"Pipeline stopped: {error}", file=sys.stderr)
        raise SystemExit(1) from error


if __name__ == "__main__":
    main()

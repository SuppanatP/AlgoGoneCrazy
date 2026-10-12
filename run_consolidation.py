"""Create the canonical long-format 30-minute market-data CSV."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from market_data.consolidation import (  # noqa: E402
    DEFAULT_REQUIRED_INSTRUMENTS,
    run_consolidation,
)


DEFAULT_INPUT_DIRECTORIES = [
    Path("data/staging/alpaca_market_data/SPY"),
    Path("data/staging/binance_spot/BTCUSDT"),
    Path("data/staging/yahoo_finance/ES3_SI"),
]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Combine selected provider 30-minute outputs into one canonical "
            "long-format CSV without mixing providers for an instrument."
        )
    )
    parser.add_argument(
        "--input-dir",
        type=Path,
        action="append",
        help=(
            "Provider/symbol directory containing the final backfilled CSV. "
            "Repeat for each selected asset. When omitted, the temporary "
            "Alpaca SPY, Binance BTCUSDT, and Yahoo ES3.SI paths "
            "are used."
        ),
    )
    parser.add_argument(
        "--output-file",
        type=Path,
        default=Path("data/canonical/market_bars_30m.csv"),
        help=(
            "Canonical long-format CSV path "
            "(default: data/canonical/market_bars_30m.csv)."
        ),
    )
    parser.add_argument(
        "--summary-file",
        type=Path,
        default=Path("data/canonical/consolidation_summary.json"),
        help=(
            "Consolidation summary path "
            "(default: data/canonical/consolidation_summary.json)."
        ),
    )
    parser.add_argument(
        "--allow-partial",
        action="store_true",
        help="Allow consolidation when one or more of the three assets is absent.",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    input_directories = args.input_dir or DEFAULT_INPUT_DIRECTORIES
    required = None if args.allow_partial else DEFAULT_REQUIRED_INSTRUMENTS
    run_consolidation(
        input_directories,
        args.output_file,
        args.summary_file,
        required_instruments=required,
    )


if __name__ == "__main__":
    main()

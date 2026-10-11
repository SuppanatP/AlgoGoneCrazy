"""Standardize a manually downloaded official FRED DEXSIUS CSV."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from src.portfolio_sizing import load_fred_sgd_usd_csv  # noqa: E402


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config-file",
        type=Path,
        default=Path("config/portfolio_sizing.json"),
    )
    parser.add_argument(
        "--input-file",
        type=Path,
        help="Downloaded FRED CSV override; defaults to the configured path.",
    )
    parser.add_argument("--start", help="Optional inclusive UTC start date.")
    parser.add_argument("--end", help="Optional exclusive UTC end date.")
    parser.add_argument("--output-file", type=Path)
    parser.add_argument("--summary-file", type=Path)
    return parser


def _read_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(f"Configuration not found: {path}")
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as error:
        raise ValueError(f"Invalid JSON in {path}: {error}") from error
    if not isinstance(value, dict):
        raise ValueError(f"Configuration in {path} must be a JSON object")
    return value


def main() -> None:
    args = build_parser().parse_args()
    config = _read_json(args.config_file)
    settings = config.get("fx_ingestion")
    if not isinstance(settings, dict):
        raise ValueError("portfolio_sizing configuration must contain fx_ingestion")
    if str(settings.get("provider", "")).strip().lower() != "fred_download":
        raise ValueError("FX ingestion provider must be fred_download")

    input_file = args.input_file or Path(
        str(settings.get("input_file", "data/fx/raw/DEXSIUS.csv"))
    )
    output_file = args.output_file or Path(
        str(settings.get("output_file", "data/fx/fx_rates_to_usd.csv"))
    )
    summary_file = args.summary_file or Path(
        str(settings.get("summary_file", "data/fx/fx_ingestion_summary.json"))
    )
    series_id = str(settings.get("series_id", "DEXSIUS")).strip()
    availability_lag_days = int(settings.get("availability_lag_days", 1))
    standardized = load_fred_sgd_usd_csv(
        input_file,
        availability_lag_days=availability_lag_days,
        series_id=series_id,
        start=args.start,
        end=args.end,
    )

    output_file.parent.mkdir(parents=True, exist_ok=True)
    summary_file.parent.mkdir(parents=True, exist_ok=True)
    standardized.to_csv(output_file, index=False)
    summary = {
        "generated_at_utc": pd.Timestamp.now(tz="UTC").isoformat(),
        "provider": "fred_download",
        "series_id": series_id,
        "input_file": str(input_file),
        "requested_start_utc": args.start,
        "requested_end_utc_exclusive": args.end,
        "availability_lag_days": availability_lag_days,
        "source_units": "SGD per USD",
        "stored_units": "USD per SGD",
        "conversion": "rate_to_base = 1 / source_rate_sgd_per_usd",
        "raw_rows": standardized.attrs.get("raw_rows"),
        "missing_or_invalid_rate_rows": standardized.attrs.get(
            "missing_or_invalid_rate_rows"
        ),
        "saved_rows": len(standardized),
        "write_mode": "replace_from_downloaded_snapshot",
        "output_file": str(output_file),
    }
    summary_file.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    first = standardized["source_observation_date"].iloc[0]
    last = standardized["source_observation_date"].iloc[-1]
    print(
        f"Standardized {len(standardized):,} {series_id} observations "
        f"from {first} through {last} -> {output_file}"
    )
    print(f"Summary -> {summary_file}")


if __name__ == "__main__":
    main()

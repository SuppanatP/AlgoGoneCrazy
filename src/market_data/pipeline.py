"""Command-line orchestration for extraction, standardization, and validation."""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pandas as pd

from market_data.config import DEFAULT_LOOKBACK_DAYS, INSTRUMENTS, Instrument
from market_data.coverage import audit_market_coverage
from market_data.extractors import fetch_binance_bars, fetch_yahoo_bars
from market_data.schema import (
    BASE_BAR_COLUMNS,
    COVERAGE_GAP_COLUMNS,
    REJECTED_BAR_COLUMNS,
    SESSION_COVERAGE_COLUMNS,
    STANDARD_BAR_COLUMNS,
)
from market_data.sessions import build_market_schedule
from market_data.standardize import (
    restrict_bar_range,
    standardize_binance_bars,
    standardize_yahoo_bars,
)
from market_data.validate import validate_bars


def _parse_utc(value: str) -> datetime:
    timestamp = pd.Timestamp(value)
    if timestamp.tzinfo is None:
        timestamp = timestamp.tz_localize("UTC")
    else:
        timestamp = timestamp.tz_convert("UTC")
    return timestamp.to_pydatetime()


def _standardize_one(
    instrument: Instrument,
    start_utc: datetime,
    end_utc: datetime,
    retrieved_at_utc: datetime,
) -> pd.DataFrame:
    expected = build_market_schedule(
        instrument, start_utc, end_utc, retrieved_at_utc
    ).expected_bars
    if expected.empty:
        print(
            f"{instrument.source_symbol}: no completed bars expected in the "
            "requested market sessions"
        )
        return pd.DataFrame(columns=BASE_BAR_COLUMNS)

    if instrument.provider == "yahoo_finance":
        raw = fetch_yahoo_bars(instrument, start_utc, end_utc)
        standardized = standardize_yahoo_bars(raw, instrument, retrieved_at_utc)
    elif instrument.provider == "binance_spot":
        raw = fetch_binance_bars(instrument, start_utc, end_utc)
        standardized = standardize_binance_bars(raw, instrument, retrieved_at_utc)
    else:
        raise ValueError(f"Unsupported provider: {instrument.provider}")

    selected = restrict_bar_range(standardized, start_utc, end_utc)
    print(
        f"{instrument.source_symbol}: extracted {len(raw):,} provider rows; "
        f"standardized {len(selected):,} rows in the requested range"
    )
    return selected


def _write_csv_atomically(frame: pd.DataFrame, path: Path, columns: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = path.with_suffix(path.suffix + ".tmp")
    frame.reindex(columns=columns).to_csv(temporary_path, index=False)
    temporary_path.replace(path)


def _summary(
    standardized: pd.DataFrame,
    accepted: pd.DataFrame,
    rejected: pd.DataFrame,
    missing_bars: pd.DataFrame,
    sessions: pd.DataFrame,
    start_utc: datetime,
    end_utc: datetime,
    retrieved_at_utc: datetime,
) -> dict[str, object]:
    instruments: dict[str, object] = {}
    for instrument in INSTRUMENTS:
        source_rows = standardized.loc[
            standardized["instrument_id"] == instrument.instrument_id
        ]
        accepted_rows = accepted.loc[
            accepted["instrument_id"] == instrument.instrument_id
        ]
        rejected_rows = rejected.loc[
            rejected["instrument_id"] == instrument.instrument_id
        ]
        missing_rows = missing_bars.loc[
            missing_bars["instrument_id"] == instrument.instrument_id
        ]
        session_rows = sessions.loc[
            sessions["instrument_id"] == instrument.instrument_id
        ]
        expected_count = int(session_rows["expected_bar_count"].sum())
        instruments[instrument.source_symbol] = {
            "calendar_name": instrument.calendar_name,
            "standardized_rows": int(len(source_rows)),
            "accepted_rows": int(len(accepted_rows)),
            "warning_rows": int(
                accepted_rows["quality_status"].eq("warning").sum()
            ),
            "rejected_rows": int(len(rejected_rows)),
            "off_session_rows": int(
                rejected_rows["rejection_reason"]
                .fillna("")
                .str.contains("outside_expected_market_session")
                .sum()
            ),
            "expected_completed_bars": expected_count,
            "missing_expected_bars": int(len(missing_rows)),
            "coverage_ratio": (
                round(len(accepted_rows) / expected_count, 6)
                if expected_count
                else None
            ),
            "open_session_days": int(
                (
                    session_rows["session_status"].isin(
                        ["open", "continuous_24_7"]
                    )
                    & session_rows["expected_bar_count"].gt(0)
                ).sum()
            ),
            "holiday_or_scheduled_closure_days": int(
                session_rows["session_status"]
                .eq("holiday_or_scheduled_closure")
                .sum()
            ),
            "first_accepted_bar_utc": (
                accepted_rows["bar_open_time_utc"].min().isoformat()
                if not accepted_rows.empty
                else None
            ),
            "last_accepted_bar_utc": (
                accepted_rows["bar_open_time_utc"].max().isoformat()
                if not accepted_rows.empty
                else None
            ),
        }

    return {
        "bar_interval": "5m",
        "requested_start_utc": start_utc.isoformat(),
        "requested_end_utc_exclusive": end_utc.isoformat(),
        "coverage_evaluated_at_utc": retrieved_at_utc.isoformat(),
        "instruments": instruments,
    }


def run_ingestion(
    start_utc: datetime,
    end_utc: datetime,
    output_directory: Path,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Run one fail-fast ingestion and write stable output files."""
    if end_utc <= start_utc:
        raise ValueError("end_utc must be later than start_utc")

    retrieved_at_utc = datetime.now(timezone.utc)
    frames = [
        _standardize_one(
            instrument,
            start_utc,
            end_utc,
            retrieved_at_utc,
        )
        for instrument in INSTRUMENTS
    ]
    standardized = pd.concat(frames, ignore_index=True)
    structurally_accepted, structurally_rejected = validate_bars(standardized)
    coverage = audit_market_coverage(
        structurally_accepted,
        start_utc,
        end_utc,
        retrieved_at_utc,
    )
    accepted = coverage.accepted
    rejected = pd.concat(
        [structurally_rejected, coverage.rejected], ignore_index=True
    ).reindex(columns=REJECTED_BAR_COLUMNS)
    rejected = rejected.sort_values(
        ["instrument_id", "bar_open_time_utc"]
    ).reset_index(drop=True)

    accepted_path = output_directory / "market_bars_5m.csv"
    rejected_path = output_directory / "rejected_market_bars_5m.csv"
    missing_path = output_directory / "missing_market_bars_5m.csv"
    sessions_path = output_directory / "market_session_coverage.csv"
    summary_path = output_directory / "ingestion_summary.json"

    _write_csv_atomically(accepted, accepted_path, STANDARD_BAR_COLUMNS)
    _write_csv_atomically(rejected, rejected_path, REJECTED_BAR_COLUMNS)
    _write_csv_atomically(
        coverage.missing_bars, missing_path, COVERAGE_GAP_COLUMNS
    )
    _write_csv_atomically(
        coverage.sessions, sessions_path, SESSION_COVERAGE_COLUMNS
    )
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(
        json.dumps(
            _summary(
                standardized,
                accepted,
                rejected,
                coverage.missing_bars,
                coverage.sessions,
                start_utc,
                end_utc,
                retrieved_at_utc,
            ),
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )

    print(f"Accepted {len(accepted):,} bars -> {accepted_path}")
    print(f"Rejected {len(rejected):,} bars -> {rejected_path}")
    print(
        f"Missing {len(coverage.missing_bars):,} expected bars -> {missing_path}"
    )
    print(f"Session coverage -> {sessions_path}")
    print(f"Summary -> {summary_path}")
    return accepted, rejected


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Extract, standardize, and validate five-minute bars for "
            "SPY, ES3.SI, and BTCUSDT."
        )
    )
    parser.add_argument(
        "--start",
        help="Inclusive UTC start date/time. Defaults to --lookback-days ago.",
    )
    parser.add_argument(
        "--end",
        help="Exclusive UTC end date/time. Defaults to the current time.",
    )
    parser.add_argument(
        "--lookback-days",
        type=int,
        default=DEFAULT_LOOKBACK_DAYS,
        help=f"Lookback when --start is omitted (default: {DEFAULT_LOOKBACK_DAYS}).",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data"),
        help="Output directory (default: data).",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    end_utc = _parse_utc(args.end) if args.end else datetime.now(timezone.utc)
    if args.start:
        start_utc = _parse_utc(args.start)
    else:
        if args.lookback_days < 1 or args.lookback_days > 59:
            raise ValueError("--lookback-days must be between 1 and 59")
        start_utc = end_utc - timedelta(days=args.lookback_days)

    run_ingestion(start_utc, end_utc, args.output_dir)


if __name__ == "__main__":
    main()

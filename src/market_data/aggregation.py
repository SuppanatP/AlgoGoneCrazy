"""Create complete 30-minute OHLCV bars from accepted five-minute bars."""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from market_data.config import (
    AGGREGATION_DURATION,
    AGGREGATION_INTERVAL,
    AGGREGATION_VERSION,
    BAR_INTERVAL,
    EXPECTED_SUBBAR_COUNT,
    INSTRUMENTS,
    PANDAS_AGGREGATION_FREQUENCY,
    Instrument,
)
from market_data.schema import (
    AGGREGATED_BAR_COLUMNS,
    INCOMPLETE_AGGREGATE_COLUMNS,
    STANDARD_BAR_COLUMNS,
)
from market_data.sessions import build_market_schedule


NUMERIC_SOURCE_COLUMNS = [
    "open",
    "high",
    "low",
    "close",
    "adjusted_close",
    "volume",
    "quote_volume",
    "trade_count",
]


@dataclass(frozen=True)
class AggregationResult:
    """Complete aggregates and windows skipped because source bars were absent."""

    complete: pd.DataFrame
    incomplete: pd.DataFrame


def _as_utc_timestamp(value: datetime | str) -> pd.Timestamp:
    timestamp = pd.Timestamp(value)
    if timestamp.tzinfo is None:
        return timestamp.tz_localize("UTC")
    return timestamp.tz_convert("UTC")


def _empty_complete() -> pd.DataFrame:
    return pd.DataFrame(columns=AGGREGATED_BAR_COLUMNS)


def _empty_incomplete() -> pd.DataFrame:
    return pd.DataFrame(columns=INCOMPLETE_AGGREGATE_COLUMNS)


def _prepare_source_bars(bars: pd.DataFrame) -> pd.DataFrame:
    missing_columns = set(STANDARD_BAR_COLUMNS) - set(bars.columns)
    if missing_columns:
        raise ValueError(
            f"Missing accepted source-bar columns: {sorted(missing_columns)}"
        )

    work = bars[STANDARD_BAR_COLUMNS].copy()
    work = work.loc[work["bar_interval"].eq(BAR_INTERVAL)].copy()
    work["bar_open_time_utc"] = pd.to_datetime(
        work["bar_open_time_utc"], errors="coerce", utc=True
    )
    work["bar_close_time_utc"] = pd.to_datetime(
        work["bar_close_time_utc"], errors="coerce", utc=True
    )
    work["retrieved_at_utc"] = pd.to_datetime(
        work["retrieved_at_utc"], errors="coerce", utc=True
    )
    work["is_final"] = work["is_final"].map(
        lambda value: value
        if isinstance(value, bool)
        else str(value).strip().lower() == "true"
    )
    for column in NUMERIC_SOURCE_COLUMNS:
        work[column] = pd.to_numeric(work[column], errors="coerce")

    if work["bar_open_time_utc"].isna().any():
        raise ValueError("Accepted source data contains an invalid opening timestamp")

    duplicate_key = ["instrument_id", "source", "bar_open_time_utc"]
    if work.duplicated(subset=duplicate_key).any():
        raise ValueError("Accepted source data contains duplicate five-minute bars")

    known_ids = {instrument.instrument_id for instrument in INSTRUMENTS}
    unknown_ids = set(work["instrument_id"].dropna()) - known_ids
    if unknown_ids:
        raise ValueError(f"Unknown instrument identifiers: {sorted(unknown_ids)}")

    return work.sort_values(
        ["instrument_id", "bar_open_time_utc"]
    ).reset_index(drop=True)


def _bucket_opens(
    timestamps: pd.Series,
    timezone_name: str,
) -> pd.Series:
    local = timestamps.dt.tz_convert(timezone_name)
    return local.dt.floor(PANDAS_AGGREGATION_FREQUENCY).dt.tz_convert("UTC")


def _full_expected_windows(
    instrument: Instrument,
    start_utc: datetime,
    end_utc: datetime,
    coverage_as_of_utc: datetime,
) -> pd.DataFrame:
    expected = build_market_schedule(
        instrument,
        start_utc,
        end_utc,
        coverage_as_of_utc,
    ).expected_bars.copy()
    if expected.empty:
        return pd.DataFrame(
            columns=[
                "bar_open_time_utc",
                "bar_close_time_utc",
                "session_date_local",
                "expected_times",
            ]
        )

    expected["bar_open_time_utc"] = pd.to_datetime(
        expected["bar_open_time_utc"], utc=True
    )
    expected["aggregate_open_time_utc"] = _bucket_opens(
        expected["bar_open_time_utc"], instrument.session_timezone
    )

    windows: list[dict[str, object]] = []
    five_minutes = pd.Timedelta(minutes=5)
    for aggregate_open, group in expected.groupby(
        "aggregate_open_time_utc", sort=True
    ):
        expected_times = pd.DatetimeIndex(
            group["bar_open_time_utc"].sort_values()
        )
        is_full_window = (
            len(expected_times) == EXPECTED_SUBBAR_COUNT
            and expected_times[0] == aggregate_open
            and expected_times[-1]
            == aggregate_open + five_minutes * (EXPECTED_SUBBAR_COUNT - 1)
        )
        if not is_full_window:
            continue

        windows.append(
            {
                "bar_open_time_utc": aggregate_open,
                "bar_close_time_utc": aggregate_open
                + pd.Timedelta(AGGREGATION_DURATION),
                "session_date_local": group["session_date_local"].iloc[0],
                "expected_times": expected_times,
            }
        )

    return pd.DataFrame(windows)


def _combined_quality_flags(source_rows: pd.DataFrame) -> str:
    flags: set[str] = set()
    for value in source_rows["quality_flags"].fillna("").astype(str):
        flags.update(flag for flag in value.split(";") if flag)
    return ";".join(sorted(flags))


def _nullable_sum(values: pd.Series) -> object:
    result = values.sum(min_count=1)
    return pd.NA if pd.isna(result) else result


def _complete_row(
    instrument: Instrument,
    source_rows: pd.DataFrame,
    window: pd.Series,
    generated_at_utc: pd.Timestamp,
) -> dict[str, object]:
    ordered = source_rows.sort_values("bar_open_time_utc")
    first = ordered.iloc[0]
    last = ordered.iloc[-1]
    quality_flags = _combined_quality_flags(ordered)

    return {
        "instrument_id": instrument.instrument_id,
        "source": instrument.provider,
        "source_symbol": instrument.source_symbol,
        "asset_class": instrument.asset_class,
        "exchange": instrument.exchange,
        "bar_interval": AGGREGATION_INTERVAL,
        "source_bar_interval": BAR_INTERVAL,
        "aggregation_source": "derived_from_5m",
        "aggregation_version": AGGREGATION_VERSION,
        "bar_origin": "derived_complete",
        "backfill_method": "",
        "backfill_version": "",
        "bar_open_time_utc": window["bar_open_time_utc"],
        "bar_close_time_utc": window["bar_close_time_utc"],
        "session_date_local": window["session_date_local"],
        "open": first["open"],
        "high": ordered["high"].max(),
        "low": ordered["low"].min(),
        "close": last["close"],
        "adjusted_close": last["adjusted_close"],
        "volume": _nullable_sum(ordered["volume"]),
        "volume_unit": instrument.volume_unit,
        "quote_volume": _nullable_sum(ordered["quote_volume"]),
        "quote_currency": instrument.quote_currency,
        "trade_count": _nullable_sum(ordered["trade_count"]),
        "is_final": bool(ordered["is_final"].fillna(False).all()),
        "expected_subbar_count": EXPECTED_SUBBAR_COUNT,
        "actual_subbar_count": EXPECTED_SUBBAR_COUNT,
        "synthetic_subbar_count": 0,
        "missing_subbar_count": 0,
        "missing_bar_positions": "",
        "coverage_ratio": 1.0,
        "open_source": "observed",
        "close_source": "observed",
        "range_is_partial": False,
        "volume_is_partial": False,
        "quality_status": "warning" if quality_flags else "pass",
        "quality_flags": quality_flags,
        "latest_source_retrieved_at_utc": ordered["retrieved_at_utc"].max(),
        "generated_at_utc": generated_at_utc,
        "backfilled_at_utc": pd.NaT,
    }


def _incomplete_row(
    instrument: Instrument,
    window: pd.Series,
    actual_count: int,
    missing_times: pd.DatetimeIndex,
    generated_at_utc: pd.Timestamp,
) -> dict[str, object]:
    return {
        "instrument_id": instrument.instrument_id,
        "source": instrument.provider,
        "source_symbol": instrument.source_symbol,
        "asset_class": instrument.asset_class,
        "exchange": instrument.exchange,
        "bar_interval": AGGREGATION_INTERVAL,
        "source_bar_interval": BAR_INTERVAL,
        "bar_open_time_utc": window["bar_open_time_utc"],
        "bar_close_time_utc": window["bar_close_time_utc"],
        "session_date_local": window["session_date_local"],
        "expected_subbar_count": EXPECTED_SUBBAR_COUNT,
        "actual_subbar_count": actual_count,
        "coverage_ratio": actual_count / EXPECTED_SUBBAR_COUNT,
        "missing_bar_times_utc": ";".join(
            timestamp.isoformat() for timestamp in missing_times
        ),
        "rejection_reason": "missing_constituent_bars",
        "generated_at_utc": generated_at_utc,
    }


def aggregate_30m_bars(
    accepted_5m_bars: pd.DataFrame,
    start_utc: datetime,
    end_utc: datetime,
    coverage_as_of_utc: datetime,
    generated_at_utc: datetime | None = None,
) -> AggregationResult:
    """Aggregate complete windows and record incomplete windows without refetching."""
    source = _prepare_source_bars(accepted_5m_bars)
    generated = _as_utc_timestamp(
        generated_at_utc or datetime.now(timezone.utc)
    )
    complete_rows: list[dict[str, object]] = []
    incomplete_rows: list[dict[str, object]] = []

    for instrument in INSTRUMENTS:
        windows = _full_expected_windows(
            instrument,
            start_utc,
            end_utc,
            coverage_as_of_utc,
        )
        instrument_rows = source.loc[
            source["instrument_id"].eq(instrument.instrument_id)
        ].copy()
        if not instrument_rows.empty:
            instrument_rows["aggregate_open_time_utc"] = _bucket_opens(
                instrument_rows["bar_open_time_utc"],
                instrument.session_timezone,
            )
        else:
            instrument_rows["aggregate_open_time_utc"] = pd.Series(
                dtype="datetime64[ns, UTC]"
            )
        grouped = {
            key: group
            for key, group in instrument_rows.groupby(
                "aggregate_open_time_utc", sort=False
            )
        }

        for _, window in windows.iterrows():
            expected_times = pd.DatetimeIndex(window["expected_times"])
            candidates = grouped.get(window["bar_open_time_utc"])
            if candidates is None:
                actual = instrument_rows.iloc[0:0]
            else:
                actual = candidates.loc[
                    candidates["bar_open_time_utc"].isin(expected_times)
                ].copy()

            actual_times = pd.DatetimeIndex(
                actual["bar_open_time_utc"].drop_duplicates().sort_values()
            )
            missing_times = expected_times.difference(actual_times)
            actual_count = len(actual_times)
            if actual_count != EXPECTED_SUBBAR_COUNT or not missing_times.empty:
                incomplete_rows.append(
                    _incomplete_row(
                        instrument,
                        window,
                        actual_count,
                        missing_times,
                        generated,
                    )
                )
                continue

            complete_rows.append(
                _complete_row(instrument, actual, window, generated)
            )

    complete = pd.DataFrame(complete_rows).reindex(columns=AGGREGATED_BAR_COLUMNS)
    incomplete = pd.DataFrame(incomplete_rows).reindex(
        columns=INCOMPLETE_AGGREGATE_COLUMNS
    )
    if complete.empty:
        complete = _empty_complete()
    else:
        complete = complete.sort_values(
            ["instrument_id", "bar_open_time_utc"]
        ).reset_index(drop=True)
    if incomplete.empty:
        incomplete = _empty_incomplete()
    else:
        incomplete = incomplete.sort_values(
            ["instrument_id", "bar_open_time_utc"]
        ).reset_index(drop=True)
    return AggregationResult(complete, incomplete)


def _write_csv_atomically(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = path.with_suffix(path.suffix + ".tmp")
    frame.to_csv(temporary_path, index=False)
    temporary_path.replace(path)


def _load_ingestion_range(summary_path: Path) -> tuple[datetime, datetime, datetime]:
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    start = _as_utc_timestamp(summary["requested_start_utc"])
    end = _as_utc_timestamp(summary["requested_end_utc_exclusive"])
    coverage_value = summary.get("coverage_evaluated_at_utc")
    coverage_as_of = _as_utc_timestamp(
        coverage_value or datetime.now(timezone.utc)
    )
    return start.to_pydatetime(), end.to_pydatetime(), coverage_as_of.to_pydatetime()


def _aggregation_summary(
    result: AggregationResult,
    start_utc: datetime,
    end_utc: datetime,
    coverage_as_of_utc: datetime,
    generated_at_utc: datetime,
) -> dict[str, object]:
    instruments: dict[str, object] = {}
    for instrument in INSTRUMENTS:
        complete_count = int(
            result.complete["instrument_id"].eq(instrument.instrument_id).sum()
        )
        incomplete_count = int(
            result.incomplete["instrument_id"].eq(instrument.instrument_id).sum()
        )
        eligible_count = complete_count + incomplete_count
        instruments[instrument.source_symbol] = {
            "complete_30m_bars": complete_count,
            "incomplete_30m_windows": incomplete_count,
            "completion_ratio": (
                round(complete_count / eligible_count, 6)
                if eligible_count
                else None
            ),
        }
    return {
        "source_bar_interval": BAR_INTERVAL,
        "aggregation_interval": AGGREGATION_INTERVAL,
        "expected_subbars_per_window": EXPECTED_SUBBAR_COUNT,
        "refetch_attempted": False,
        "requested_start_utc": _as_utc_timestamp(start_utc).isoformat(),
        "requested_end_utc_exclusive": _as_utc_timestamp(end_utc).isoformat(),
        "source_coverage_as_of_utc": _as_utc_timestamp(
            coverage_as_of_utc
        ).isoformat(),
        "generated_at_utc": _as_utc_timestamp(generated_at_utc).isoformat(),
        "instruments": instruments,
    }


def run_aggregation(input_directory: Path, output_directory: Path) -> AggregationResult:
    """Read ingestion CSV outputs and write deterministic 30-minute outputs."""
    source_path = input_directory / "market_bars_5m.csv"
    ingestion_summary_path = input_directory / "ingestion_summary.json"
    source_bars = pd.read_csv(source_path)
    start, end, coverage_as_of = _load_ingestion_range(ingestion_summary_path)
    generated_at = datetime.now(timezone.utc)
    result = aggregate_30m_bars(
        source_bars,
        start,
        end,
        coverage_as_of,
        generated_at_utc=generated_at,
    )

    complete_path = output_directory / "market_bars_30m.csv"
    incomplete_path = output_directory / "incomplete_market_bars_30m.csv"
    summary_path = output_directory / "aggregation_summary.json"
    _write_csv_atomically(result.complete, complete_path)
    _write_csv_atomically(result.incomplete, incomplete_path)
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(
        json.dumps(
            _aggregation_summary(
                result,
                start,
                end,
                coverage_as_of,
                generated_at,
            ),
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )

    print(f"Complete {len(result.complete):,} bars -> {complete_path}")
    print(
        f"Incomplete {len(result.incomplete):,} windows -> {incomplete_path}"
    )
    print(f"Summary -> {summary_path}")
    return result


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Aggregate accepted five-minute bars into complete 30-minute bars "
            "without refetching missing source data."
        )
    )
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=Path("data"),
        help="Directory containing ingestion outputs (default: data).",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data"),
        help="Directory for aggregation outputs (default: data).",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    run_aggregation(args.input_dir, args.output_dir)


if __name__ == "__main__":
    main()

"""Apply conservative, provenance-rich backfill methods to incomplete 30m bars."""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from market_data.config import (
    AGGREGATION_INTERVAL,
    AGGREGATION_VERSION,
    BACKFILL_VERSION,
    BAR_INTERVAL,
    EXPECTED_SUBBAR_COUNT,
    AVAILABLE_INSTRUMENTS,
    MIN_BACKFILL_OBSERVED_SUBBARS,
    Instrument,
)
from market_data.schema import (
    AGGREGATED_BAR_COLUMNS,
    INCOMPLETE_AGGREGATE_COLUMNS,
    STANDARD_BAR_COLUMNS,
    UNRESOLVED_BACKFILL_COLUMNS,
)


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
class BackfillResult:
    """Backfilled bars, combined usable bars, and unresolved windows."""

    backfilled: pd.DataFrame
    combined: pd.DataFrame
    unresolved: pd.DataFrame


def _as_utc_timestamp(value: object) -> pd.Timestamp:
    timestamp = pd.Timestamp(value)
    if timestamp.tzinfo is None:
        return timestamp.tz_localize("UTC")
    return timestamp.tz_convert("UTC")


def _empty_backfilled() -> pd.DataFrame:
    return pd.DataFrame(columns=AGGREGATED_BAR_COLUMNS)


def _empty_unresolved() -> pd.DataFrame:
    return pd.DataFrame(columns=UNRESOLVED_BACKFILL_COLUMNS)


def _prepare_source_bars(frame: pd.DataFrame) -> pd.DataFrame:
    missing_columns = set(STANDARD_BAR_COLUMNS) - set(frame.columns)
    if missing_columns:
        raise ValueError(
            f"Missing accepted source-bar columns: {sorted(missing_columns)}"
        )

    work = frame[STANDARD_BAR_COLUMNS].copy()
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
    work["session_date_local"] = pd.to_datetime(
        work["session_date_local"], errors="coerce"
    ).dt.date
    for column in NUMERIC_SOURCE_COLUMNS:
        work[column] = pd.to_numeric(work[column], errors="coerce")

    if work["bar_open_time_utc"].isna().any():
        raise ValueError("Accepted source data contains invalid timestamps")
    duplicate_key = ["instrument_id", "source", "bar_open_time_utc"]
    if work.duplicated(subset=duplicate_key).any():
        raise ValueError("Accepted source data contains duplicate five-minute bars")
    return work.sort_values(
        ["instrument_id", "bar_open_time_utc"]
    ).reset_index(drop=True)


def _prepare_complete_bars(frame: pd.DataFrame) -> pd.DataFrame:
    work = frame.copy()
    defaults: dict[str, object] = {
        "bar_origin": "derived_complete",
        "backfill_method": "",
        "backfill_version": "",
        "synthetic_subbar_count": 0,
        "missing_subbar_count": 0,
        "missing_bar_positions": "",
        "open_source": "observed",
        "close_source": "observed",
        "range_is_partial": False,
        "volume_is_partial": False,
        "backfilled_at_utc": pd.NaT,
    }
    for column in AGGREGATED_BAR_COLUMNS:
        if column not in work.columns:
            work[column] = defaults.get(column, pd.NA)
    work["bar_open_time_utc"] = pd.to_datetime(
        work["bar_open_time_utc"], errors="coerce", utc=True
    )
    work["bar_close_time_utc"] = pd.to_datetime(
        work["bar_close_time_utc"], errors="coerce", utc=True
    )
    return work.reindex(columns=AGGREGATED_BAR_COLUMNS)


def _prepare_incomplete(frame: pd.DataFrame) -> pd.DataFrame:
    missing_columns = set(INCOMPLETE_AGGREGATE_COLUMNS) - set(frame.columns)
    if missing_columns:
        raise ValueError(
            f"Missing incomplete-window columns: {sorted(missing_columns)}"
        )
    work = frame[INCOMPLETE_AGGREGATE_COLUMNS].copy()
    work["bar_open_time_utc"] = pd.to_datetime(
        work["bar_open_time_utc"], errors="coerce", utc=True
    )
    work["bar_close_time_utc"] = pd.to_datetime(
        work["bar_close_time_utc"], errors="coerce", utc=True
    )
    work["session_date_local"] = pd.to_datetime(
        work["session_date_local"], errors="coerce"
    ).dt.date
    return work


def _nullable_sum(values: pd.Series) -> object:
    result = pd.to_numeric(values, errors="coerce").sum(min_count=1)
    return pd.NA if pd.isna(result) else result


def _combined_flags(source_rows: pd.DataFrame, added: set[str]) -> str:
    flags = set(added)
    for value in source_rows["quality_flags"].fillna("").astype(str):
        flags.update(flag for flag in value.split(";") if flag)
    return ";".join(sorted(flags))


def _adjacent_row(
    instrument_rows: pd.DataFrame,
    timestamp: pd.Timestamp,
    session_date_local: object,
    instrument: Instrument,
) -> pd.Series | None:
    matches = instrument_rows.loc[
        instrument_rows["bar_open_time_utc"].eq(timestamp)
    ]
    if matches.empty:
        return None
    row = matches.iloc[-1]
    if (
        instrument.calendar_name != "24/7"
        and row["session_date_local"] != session_date_local
    ):
        return None
    return row


def _unresolved_row(
    window: pd.Series,
    attempted_method: str,
    reason: str,
    evaluated_at_utc: pd.Timestamp,
) -> dict[str, object]:
    result = window.reindex(INCOMPLETE_AGGREGATE_COLUMNS).to_dict()
    result.update(
        {
            "attempted_method": attempted_method,
            "unresolved_reason": reason,
            "evaluated_at_utc": evaluated_at_utc,
        }
    )
    return result


def _backfilled_row(
    instrument: Instrument,
    window: pd.Series,
    observed: pd.DataFrame,
    missing_positions: list[int],
    previous_row: pd.Series | None,
    following_row: pd.Series | None,
    evaluated_at_utc: pd.Timestamp,
) -> dict[str, object]:
    expected_times = pd.date_range(
        window["bar_open_time_utc"],
        periods=EXPECTED_SUBBAR_COUNT,
        freq="5min",
    )
    observed_by_time = {
        row["bar_open_time_utc"]: row for _, row in observed.iterrows()
    }
    first_observed = observed_by_time.get(expected_times[0])
    last_observed = observed_by_time.get(expected_times[-1])

    if first_observed is not None:
        open_price = first_observed["open"]
        open_source = "observed"
    else:
        if previous_row is None:
            raise ValueError("Previous adjacent row is required for synthetic open")
        open_price = previous_row["close"]
        open_source = "previous_bar_close"

    if last_observed is not None:
        close_price = last_observed["close"]
        adjusted_close = last_observed["adjusted_close"]
        close_source = "observed"
    else:
        if following_row is None:
            raise ValueError("Following adjacent row is required for synthetic close")
        close_price = following_row["open"]
        adjustment_factor = (
            following_row["adjusted_close"] / following_row["close"]
        )
        adjusted_close = close_price * adjustment_factor
        close_source = "next_bar_open"

    high = max([open_price, close_price, *observed["high"].tolist()])
    low = min([open_price, close_price, *observed["low"].tolist()])
    boundary_missing = 0 in missing_positions or 5 in missing_positions
    if boundary_missing:
        bar_origin = "synthetic_boundary_backfill"
        backfill_method = "adjacent_boundary"
    else:
        bar_origin = "partial_observed"
        backfill_method = "interior_partial"

    added_flags = {
        "missing_constituent_bars",
        "partial_range",
        "partial_volume",
    }
    if 0 in missing_positions:
        added_flags.add("synthetic_open_from_previous_close")
    if 5 in missing_positions:
        added_flags.add("synthetic_close_from_next_open")
    if not boundary_missing:
        added_flags.add("interior_missing_bars")

    retrieval_times = list(observed["retrieved_at_utc"].dropna())
    if previous_row is not None:
        retrieval_times.append(previous_row["retrieved_at_utc"])
    if following_row is not None:
        retrieval_times.append(following_row["retrieved_at_utc"])
    latest_retrieval = max(retrieval_times) if retrieval_times else pd.NaT

    return {
        "instrument_id": instrument.instrument_id,
        "source": instrument.provider,
        "source_symbol": instrument.source_symbol,
        "asset_class": instrument.asset_class,
        "exchange": instrument.exchange,
        "bar_interval": AGGREGATION_INTERVAL,
        "source_bar_interval": BAR_INTERVAL,
        "aggregation_source": "backfilled_from_available_5m",
        "aggregation_version": AGGREGATION_VERSION,
        "bar_origin": bar_origin,
        "backfill_method": backfill_method,
        "backfill_version": BACKFILL_VERSION,
        "bar_open_time_utc": window["bar_open_time_utc"],
        "bar_close_time_utc": window["bar_close_time_utc"],
        "session_date_local": window["session_date_local"],
        "open": open_price,
        "high": high,
        "low": low,
        "close": close_price,
        "adjusted_close": adjusted_close,
        "volume": _nullable_sum(observed["volume"]),
        "volume_unit": instrument.volume_unit,
        "quote_volume": _nullable_sum(observed["quote_volume"]),
        "quote_currency": instrument.quote_currency,
        "trade_count": _nullable_sum(observed["trade_count"]),
        "is_final": True,
        "expected_subbar_count": EXPECTED_SUBBAR_COUNT,
        "actual_subbar_count": len(observed),
        "synthetic_subbar_count": sum(
            position in {0, 5} for position in missing_positions
        ),
        "missing_subbar_count": len(missing_positions),
        "missing_bar_positions": ";".join(
            f"t{position}" for position in missing_positions
        ),
        "coverage_ratio": len(observed) / EXPECTED_SUBBAR_COUNT,
        "open_source": open_source,
        "close_source": close_source,
        "range_is_partial": True,
        "volume_is_partial": True,
        "quality_status": "warning",
        "quality_flags": _combined_flags(observed, added_flags),
        "latest_source_retrieved_at_utc": latest_retrieval,
        "generated_at_utc": evaluated_at_utc,
        "backfilled_at_utc": evaluated_at_utc,
    }


def backfill_30m_bars(
    accepted_5m_bars: pd.DataFrame,
    complete_30m_bars: pd.DataFrame,
    incomplete_30m_windows: pd.DataFrame,
    evaluated_at_utc: datetime | None = None,
) -> BackfillResult:
    """Apply supported methods and retain unsupported windows for later decisions."""
    source = _prepare_source_bars(accepted_5m_bars)
    complete = _prepare_complete_bars(complete_30m_bars)
    incomplete = _prepare_incomplete(incomplete_30m_windows)
    evaluated = _as_utc_timestamp(
        evaluated_at_utc or datetime.now(timezone.utc)
    )
    instruments = {
        (instrument.instrument_id, instrument.provider): instrument
        for instrument in AVAILABLE_INSTRUMENTS
    }
    backfilled_rows: list[dict[str, object]] = []
    unresolved_rows: list[dict[str, object]] = []

    for _, window in incomplete.iterrows():
        instrument = instruments.get((window["instrument_id"], window["source"]))
        if instrument is None:
            unresolved_rows.append(
                _unresolved_row(
                    window, "none", "unknown_instrument", evaluated
                )
            )
            continue

        instrument_rows = source.loc[
            source["instrument_id"].eq(instrument.instrument_id)
            & source["source"].eq(instrument.provider)
        ]
        expected_times = pd.date_range(
            window["bar_open_time_utc"],
            periods=EXPECTED_SUBBAR_COUNT,
            freq="5min",
        )
        observed = instrument_rows.loc[
            instrument_rows["bar_open_time_utc"].isin(expected_times)
        ].drop_duplicates(subset=["bar_open_time_utc"], keep="last")
        observed = observed.sort_values("bar_open_time_utc")
        observed_times = set(observed["bar_open_time_utc"])
        missing_positions = [
            position
            for position, timestamp in enumerate(expected_times)
            if timestamp not in observed_times
        ]

        if not missing_positions:
            unresolved_rows.append(
                _unresolved_row(
                    window,
                    "none",
                    "source_window_is_now_complete_rerun_aggregation",
                    evaluated,
                )
            )
            continue

        if len(observed) < MIN_BACKFILL_OBSERVED_SUBBARS:
            unresolved_rows.append(
                _unresolved_row(
                    window,
                    "none",
                    "fewer_than_four_observed_subbars",
                    evaluated,
                )
            )
            continue

        previous_row = None
        following_row = None
        attempted_method = (
            "adjacent_boundary"
            if 0 in missing_positions or 5 in missing_positions
            else "interior_partial"
        )
        if 0 in missing_positions:
            previous_row = _adjacent_row(
                instrument_rows,
                window["bar_open_time_utc"] - pd.Timedelta(minutes=5),
                window["session_date_local"],
                instrument,
            )
            if previous_row is None:
                unresolved_rows.append(
                    _unresolved_row(
                        window,
                        attempted_method,
                        "previous_adjacent_bar_unavailable",
                        evaluated,
                    )
                )
                continue

        if 5 in missing_positions:
            following_row = _adjacent_row(
                instrument_rows,
                window["bar_close_time_utc"],
                window["session_date_local"],
                instrument,
            )
            if following_row is None:
                unresolved_rows.append(
                    _unresolved_row(
                        window,
                        attempted_method,
                        "following_adjacent_bar_unavailable",
                        evaluated,
                    )
                )
                continue

        backfilled_rows.append(
            _backfilled_row(
                instrument,
                window,
                observed,
                missing_positions,
                previous_row,
                following_row,
                evaluated,
            )
        )

    backfilled = pd.DataFrame(backfilled_rows).reindex(
        columns=AGGREGATED_BAR_COLUMNS
    )
    unresolved = pd.DataFrame(unresolved_rows).reindex(
        columns=UNRESOLVED_BACKFILL_COLUMNS
    )
    if backfilled.empty:
        backfilled = _empty_backfilled()
    else:
        backfilled = backfilled.sort_values(
            ["instrument_id", "bar_open_time_utc"]
        ).reset_index(drop=True)
    if unresolved.empty:
        unresolved = _empty_unresolved()
    else:
        unresolved = unresolved.sort_values(
            ["instrument_id", "bar_open_time_utc"]
        ).reset_index(drop=True)

    usable_frames = [
        frame for frame in (complete, backfilled) if not frame.empty
    ]
    combined = (
        pd.concat(usable_frames, ignore_index=True)
        if usable_frames
        else _empty_backfilled()
    )
    priority = {
        "derived_complete": 3,
        "partial_observed": 2,
        "synthetic_boundary_backfill": 1,
    }
    combined["_origin_priority"] = combined["bar_origin"].map(priority).fillna(0)
    combined = combined.sort_values(
        ["instrument_id", "bar_open_time_utc", "_origin_priority"],
        ascending=[True, True, False],
    )
    combined = combined.drop_duplicates(
        subset=["instrument_id", "source", "bar_interval", "bar_open_time_utc"],
        keep="first",
    ).drop(columns=["_origin_priority"])
    combined = combined.reindex(columns=AGGREGATED_BAR_COLUMNS).reset_index(drop=True)
    return BackfillResult(backfilled, combined, unresolved)


def _write_csv_atomically(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = path.with_suffix(path.suffix + ".tmp")
    frame.to_csv(temporary_path, index=False)
    temporary_path.replace(path)


def _summary(result: BackfillResult) -> dict[str, object]:
    origins = result.backfilled["bar_origin"].value_counts().to_dict()
    reasons = result.unresolved["unresolved_reason"].value_counts().to_dict()
    return {
        "backfill_version": BACKFILL_VERSION,
        "minimum_observed_subbars": MIN_BACKFILL_OBSERVED_SUBBARS,
        "standard_deviation_calculated": False,
        "backfilled_windows": int(len(result.backfilled)),
        "combined_usable_bars": int(len(result.combined)),
        "unresolved_windows": int(len(result.unresolved)),
        "backfilled_by_origin": {key: int(value) for key, value in origins.items()},
        "unresolved_by_reason": {key: int(value) for key, value in reasons.items()},
    }


def run_backfill(input_directory: Path, output_directory: Path) -> BackfillResult:
    """Read CSV pipeline outputs and write non-destructive backfill outputs."""
    source = pd.read_csv(input_directory / "market_bars_5m.csv")
    complete = pd.read_csv(input_directory / "market_bars_30m.csv")
    incomplete = pd.read_csv(
        input_directory / "incomplete_market_bars_30m.csv"
    )
    result = backfill_30m_bars(source, complete, incomplete)

    backfilled_path = output_directory / "backfilled_market_bars_30m.csv"
    combined_path = output_directory / "market_bars_30m_with_backfill.csv"
    unresolved_path = output_directory / "unresolved_market_bars_30m.csv"
    summary_path = output_directory / "backfill_summary.json"
    _write_csv_atomically(result.backfilled, backfilled_path)
    _write_csv_atomically(result.combined, combined_path)
    _write_csv_atomically(result.unresolved, unresolved_path)
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(
        json.dumps(_summary(result), indent=2) + "\n",
        encoding="utf-8",
    )

    print(f"Backfilled {len(result.backfilled):,} windows -> {backfilled_path}")
    print(f"Combined {len(result.combined):,} usable bars -> {combined_path}")
    print(f"Unresolved {len(result.unresolved):,} windows -> {unresolved_path}")
    print(f"Summary -> {summary_path}")
    return result


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Backfill supported 30-minute gaps from accepted CSV data without "
            "modifying the original aggregate files."
        )
    )
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=Path("data/staging/combined"),
        help=(
            "Directory containing ingestion and aggregation outputs "
            "(default: data/staging/combined)."
        ),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data/staging/combined"),
        help=(
            "Directory for backfill outputs "
            "(default: data/staging/combined)."
        ),
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    run_backfill(args.input_dir, args.output_dir)


if __name__ == "__main__":
    main()

"""Consolidate selected provider outputs into one canonical long-form dataset."""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from market_data.config import AGGREGATION_INTERVAL
from market_data.schema import AGGREGATED_BAR_COLUMNS


FINAL_AGGREGATE_FILE_NAME = "market_bars_30m_with_backfill.csv"
DEFAULT_REQUIRED_INSTRUMENTS = {
    "spy_usd",
    "btc_usdt",
    "sti_es3_sgd",
}
CANONICAL_KEY_COLUMNS = [
    "instrument_id",
    "source",
    "bar_interval",
    "bar_open_time_utc",
]
TIMESTAMP_COLUMNS = [
    "bar_open_time_utc",
    "bar_close_time_utc",
    "latest_source_retrieved_at_utc",
    "generated_at_utc",
    "backfilled_at_utc",
]


@dataclass(frozen=True)
class ConsolidationResult:
    """Canonical rows plus the JSON-serializable consolidation summary."""

    canonical: pd.DataFrame
    summary: dict[str, object]


def _prepare_input(path: Path) -> pd.DataFrame:
    if not path.is_file():
        raise FileNotFoundError(
            f"Final aggregate file not found: {path}. Run aggregation and "
            "backfill for this provider/symbol first."
        )

    frame = pd.read_csv(path)
    missing_columns = set(AGGREGATED_BAR_COLUMNS) - set(frame.columns)
    if missing_columns:
        raise ValueError(
            f"{path} is missing canonical aggregate columns: "
            f"{sorted(missing_columns)}"
        )

    work = frame.loc[:, AGGREGATED_BAR_COLUMNS].copy()
    for column in TIMESTAMP_COLUMNS:
        work[column] = pd.to_datetime(work[column], errors="coerce", utc=True)

    required_non_null = [
        "instrument_id",
        "source",
        "source_symbol",
        "bar_interval",
        "bar_open_time_utc",
        "bar_close_time_utc",
        "open",
        "high",
        "low",
        "close",
        "bar_origin",
    ]
    null_columns = [
        column for column in required_non_null if work[column].isna().any()
    ]
    if null_columns:
        raise ValueError(
            f"{path} contains null values in required columns: {null_columns}"
        )
    if not work["bar_interval"].eq(AGGREGATION_INTERVAL).all():
        intervals = sorted(work["bar_interval"].dropna().astype(str).unique())
        raise ValueError(
            f"{path} contains non-{AGGREGATION_INTERVAL} rows: {intervals}"
        )
    if work.duplicated(subset=CANONICAL_KEY_COLUMNS).any():
        raise ValueError(f"{path} contains duplicate canonical bar keys")
    return work


def _instrument_summary(frame: pd.DataFrame) -> dict[str, object]:
    result: dict[str, object] = {}
    for instrument_id, rows in frame.groupby("instrument_id", sort=True):
        sources = sorted(rows["source"].dropna().astype(str).unique())
        symbols = sorted(rows["source_symbol"].dropna().astype(str).unique())
        origins = rows["bar_origin"].value_counts().to_dict()
        result[str(instrument_id)] = {
            "source": sources[0],
            "source_symbol": symbols[0] if len(symbols) == 1 else symbols,
            "rows": int(len(rows)),
            "first_bar_open_time_utc": rows["bar_open_time_utc"].min().isoformat(),
            "last_bar_open_time_utc": rows["bar_open_time_utc"].max().isoformat(),
            "rows_by_origin": {
                str(origin): int(count) for origin, count in origins.items()
            },
        }
    return result


def consolidate_30m_bars(
    input_directories: list[Path],
    required_instruments: set[str] | None = None,
    generated_at_utc: datetime | None = None,
) -> ConsolidationResult:
    """Build one long dataset while enforcing one selected source per instrument."""
    if not input_directories:
        raise ValueError("At least one --input-dir is required")

    input_files = [
        directory / FINAL_AGGREGATE_FILE_NAME
        for directory in input_directories
    ]
    frames = [_prepare_input(path) for path in input_files]
    canonical = pd.concat(frames, ignore_index=True).reindex(
        columns=AGGREGATED_BAR_COLUMNS
    )

    if canonical.duplicated(subset=CANONICAL_KEY_COLUMNS).any():
        raise ValueError(
            "The selected input directories contain overlapping canonical bar keys"
        )

    source_counts = canonical.groupby("instrument_id")["source"].nunique()
    mixed_instruments = sorted(source_counts.loc[source_counts.gt(1)].index)
    if mixed_instruments:
        raise ValueError(
            "More than one provider was selected for these instruments: "
            f"{mixed_instruments}. Choose one primary provider per instrument."
        )

    available_instruments = set(canonical["instrument_id"].astype(str).unique())
    required = required_instruments or set()
    missing_instruments = sorted(required - available_instruments)
    if missing_instruments:
        raise ValueError(
            f"Required instruments are missing from consolidation: {missing_instruments}"
        )

    canonical = canonical.sort_values(
        ["bar_open_time_utc", "instrument_id", "source"]
    ).reset_index(drop=True)
    generated = pd.Timestamp(generated_at_utc or datetime.now(timezone.utc))
    if generated.tzinfo is None:
        generated = generated.tz_localize("UTC")
    else:
        generated = generated.tz_convert("UTC")

    summary: dict[str, object] = {
        "aggregation_interval": AGGREGATION_INTERVAL,
        "format": "long",
        "generated_at_utc": generated.isoformat(),
        "input_files": [str(path) for path in input_files],
        "canonical_key_columns": CANONICAL_KEY_COLUMNS,
        "row_count": int(len(canonical)),
        "instrument_count": int(canonical["instrument_id"].nunique()),
        "instruments": _instrument_summary(canonical),
    }
    return ConsolidationResult(canonical, summary)


def _write_csv_atomically(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = path.with_suffix(path.suffix + ".tmp")
    frame.to_csv(temporary_path, index=False)
    temporary_path.replace(path)


def run_consolidation(
    input_directories: list[Path],
    output_file: Path,
    summary_file: Path,
    required_instruments: set[str] | None = DEFAULT_REQUIRED_INSTRUMENTS,
) -> ConsolidationResult:
    """Consolidate final aggregate CSVs and write stable canonical outputs."""
    result = consolidate_30m_bars(
        input_directories,
        required_instruments=required_instruments,
    )
    _write_csv_atomically(result.canonical, output_file)
    summary_file.parent.mkdir(parents=True, exist_ok=True)
    summary_file.write_text(
        json.dumps(result.summary, indent=2) + "\n",
        encoding="utf-8",
    )
    print(f"Canonical {len(result.canonical):,} rows -> {output_file}")
    print(f"Summary -> {summary_file}")
    return result

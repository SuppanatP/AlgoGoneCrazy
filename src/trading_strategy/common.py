"""Shared validation and output handling for strategy modules."""

from __future__ import annotations

import pandas as pd


REQUIRED_STRATEGY_COLUMNS = {
    "instrument_id",
    "bar_open_time_utc",
    "adjusted_close",
}


def _as_boolean(values: pd.Series, column: str) -> pd.Series:
    if pd.api.types.is_bool_dtype(values.dtype):
        if values.isna().any():
            raise ValueError(f"{column} contains missing values")
        return values.astype(bool)

    normalized = values.astype("string").str.strip().str.lower()
    mapping = {
        "true": True,
        "1": True,
        "yes": True,
        "false": False,
        "0": False,
        "no": False,
    }
    invalid = normalized.isna() | ~normalized.isin(mapping)
    if invalid.any():
        raise ValueError(f"{column} contains values that are not boolean")
    return normalized.map(mapping).astype(bool)


def prepare_strategy_bars(bars: pd.DataFrame) -> pd.DataFrame:
    """Validate canonical bars and mark rows on which a signal may change.

    Price features are calculated over every supplied row so approved backfills
    can preserve continuity. A strategy may only enter or exit on a completed,
    final bar. Rows without aggregation provenance are treated as eligible,
    which also permits direct use with simpler research frames.
    """
    missing_columns = REQUIRED_STRATEGY_COLUMNS - set(bars.columns)
    if missing_columns:
        raise ValueError(
            f"Strategy input is missing columns: {sorted(missing_columns)}"
        )
    if bars.empty:
        raise ValueError("Strategy input contains no bars")

    work = bars.copy()
    work["bar_open_time_utc"] = pd.to_datetime(
        work["bar_open_time_utc"], errors="coerce", utc=True
    )
    work["adjusted_close"] = pd.to_numeric(
        work["adjusted_close"], errors="coerce"
    )

    if work["instrument_id"].isna().any():
        raise ValueError("Strategy input contains a missing instrument_id")
    if work["bar_open_time_utc"].isna().any():
        raise ValueError("Strategy input contains an invalid bar timestamp")
    if work["adjusted_close"].isna().any():
        raise ValueError("Strategy input contains a missing adjusted_close")
    if work["adjusted_close"].le(0).any():
        raise ValueError("Strategy input contains a non-positive adjusted_close")

    key_columns = ["instrument_id", "bar_open_time_utc"]
    if work.duplicated(subset=key_columns).any():
        raise ValueError(
            "Strategy input contains duplicate instrument/timestamp rows"
        )

    eligible = pd.Series(True, index=work.index, dtype=bool)
    if "is_final" in work.columns:
        eligible &= _as_boolean(work["is_final"], "is_final")
    if "bar_origin" in work.columns:
        eligible &= work["bar_origin"].eq("derived_complete")
    if "quality_status" in work.columns:
        eligible &= work["quality_status"].isin(["pass", "warning"])

    work["signal_eligible"] = eligible
    return work.sort_values(key_columns).reset_index(drop=True)


def sort_signal_output(signals: pd.DataFrame) -> pd.DataFrame:
    """Return long-format signals in chronological, instrument-stable order."""
    return signals.sort_values(
        ["bar_open_time_utc", "instrument_id"]
    ).reset_index(drop=True)

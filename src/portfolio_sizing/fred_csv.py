"""Standardize a manually downloaded official FRED DEXSIUS CSV."""

from __future__ import annotations

from pathlib import Path

import pandas as pd


FRED_SGD_PER_USD_SERIES = "DEXSIUS"


def _optional_utc_date(value: object | None, label: str) -> pd.Timestamp | None:
    if value is None:
        return None
    try:
        timestamp = pd.Timestamp(value)
    except (TypeError, ValueError) as error:
        raise ValueError(f"{label} must be a valid date or timestamp") from error
    if timestamp.tzinfo is None:
        timestamp = timestamp.tz_localize("UTC")
    else:
        timestamp = timestamp.tz_convert("UTC")
    return timestamp.normalize()


def load_fred_sgd_usd_csv(
    path: Path,
    *,
    availability_lag_days: int = 1,
    series_id: str = FRED_SGD_PER_USD_SERIES,
    start: object | None = None,
    end: object | None = None,
) -> pd.DataFrame:
    """Load FRED's CSV and return USD per SGD in the project's FX schema.

    ``start`` is inclusive and ``end`` is exclusive when supplied. FRED
    DEXSIUS is quoted as SGD per one USD, so every valid value is inverted.
    The conservative availability lag prevents the same-day fixing from being
    treated as known earlier that day.
    """
    if not path.is_file():
        raise FileNotFoundError(
            f"Downloaded FRED CSV not found: {path}. Download the maximum "
            "DEXSIUS CSV and save it at this location."
        )
    if availability_lag_days < 0:
        raise ValueError("availability_lag_days must not be negative")

    raw = pd.read_csv(path, low_memory=False)
    date_column = next(
        (
            name
            for name in ("observation_date", "DATE", "date")
            if name in raw.columns
        ),
        None,
    )
    if date_column is None or series_id not in raw.columns:
        raise ValueError(
            f"FRED CSV must contain a date column and {series_id}; found "
            f"{list(raw.columns)}"
        )

    work = pd.DataFrame(
        {
            "source_observation_date": pd.to_datetime(
                raw[date_column], errors="coerce", utc=True
            ).dt.normalize(),
            "source_rate_sgd_per_usd": pd.to_numeric(
                raw[series_id], errors="coerce"
            ),
        }
    )
    invalid_dates = work["source_observation_date"].isna()
    if invalid_dates.any():
        raise ValueError(
            f"FRED CSV contains {int(invalid_dates.sum())} invalid dates"
        )
    if work["source_observation_date"].duplicated().any():
        raise ValueError("FRED CSV contains duplicate observation dates")

    valid_rate = work["source_rate_sgd_per_usd"].gt(0)
    clean = work.loc[valid_rate].copy()
    start_date = _optional_utc_date(start, "start")
    end_date = _optional_utc_date(end, "end")
    if start_date is not None and end_date is not None and end_date <= start_date:
        raise ValueError("end must be later than start")
    if start_date is not None:
        clean = clean.loc[clean["source_observation_date"].ge(start_date)]
    if end_date is not None:
        clean = clean.loc[clean["source_observation_date"].lt(end_date)]
    if clean.empty:
        raise ValueError("FRED CSV has no valid observations in the selected range")
    clean = clean.sort_values("source_observation_date")

    output = pd.DataFrame(
        {
            "rate_time_utc": clean["source_observation_date"]
            + pd.Timedelta(days=availability_lag_days),
            "quote_currency": "SGD",
            "base_currency": "USD",
            "rate_to_base": 1.0 / clean["source_rate_sgd_per_usd"],
            "source": "fred_download",
            "source_series": series_id,
            "source_observation_date": clean[
                "source_observation_date"
            ].dt.date.astype(str),
            "source_rate_sgd_per_usd": clean["source_rate_sgd_per_usd"],
            "is_forward_filled": False,
        }
    )
    output.attrs["raw_rows"] = len(raw)
    output.attrs["missing_or_invalid_rate_rows"] = int((~valid_rate).sum())
    return output.reset_index(drop=True)

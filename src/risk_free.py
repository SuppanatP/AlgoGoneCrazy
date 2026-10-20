"""Official U.S. Treasury risk-free-rate input for performance reporting."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd


def load_risk_free_config(path: Path) -> dict[str, Any]:
    """Read and minimally validate the tracked risk-free-rate manifest."""
    if not path.is_file():
        raise FileNotFoundError(f"Risk-free configuration not found: {path}")
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as error:
        raise ValueError(f"Invalid JSON in {path}: {error}") from error
    if not isinstance(value, dict):
        raise ValueError(f"{path} must contain a JSON object")
    required = {
        "input_file",
        "date_column",
        "yield_column",
        "availability_lag_days",
        "annualization_days",
        "source_name",
        "source_url",
    }
    missing = sorted(required - set(value))
    if missing:
        raise ValueError(
            f"Risk-free configuration is missing: {', '.join(missing)}"
        )
    return value


def load_treasury_annual_rates(
    path: Path,
    *,
    date_column: str = "Date",
    yield_column: str = "3 Mo",
    availability_lag_days: int = 1,
) -> pd.Series:
    """Load annual decimal yields indexed by their conservative availability date.

    Treasury publishes percentages on business days.  Applying a one-calendar-
    day availability lag means a return period never uses a fixing that was not
    known at that period's start.  Alignment to weekends and holidays is handled
    later by backward-looking forward fill.
    """
    if not path.is_file():
        raise FileNotFoundError(
            f"Official Treasury CSV not found: {path}. Download it using the "
            "documented command before running portfolio metrics."
        )
    if availability_lag_days < 0:
        raise ValueError("availability_lag_days must not be negative")

    raw = pd.read_csv(path, low_memory=False)
    missing = [name for name in (date_column, yield_column) if name not in raw]
    if missing:
        raise ValueError(
            f"Treasury CSV is missing columns {missing}; found {list(raw.columns)}"
        )
    dates = pd.to_datetime(raw[date_column], errors="coerce", utc=True).dt.normalize()
    yields_percent = pd.to_numeric(raw[yield_column], errors="coerce")
    valid = dates.notna() & yields_percent.notna() & yields_percent.ge(0)
    rates = pd.Series(
        yields_percent.loc[valid].to_numpy(dtype="float64") / 100.0,
        index=dates.loc[valid] + pd.Timedelta(days=availability_lag_days),
        name="annual_risk_free_rate",
    ).sort_index()
    if rates.empty:
        raise ValueError("Treasury CSV contains no valid risk-free observations")
    if rates.index.has_duplicates:
        raise ValueError("Treasury CSV contains duplicate observation dates")
    rates.attrs.update(
        {
            "input_file": str(path),
            "date_column": date_column,
            "yield_column": yield_column,
            "availability_lag_days": availability_lag_days,
            "source_units": "percent per annum",
            "stored_units": "decimal annual rate",
        }
    )
    return rates


def load_configured_risk_free_rates(
    config_path: Path,
) -> tuple[pd.Series, dict[str, Any]]:
    """Load the tracked manifest and its referenced Treasury CSV."""
    settings = load_risk_free_config(config_path)
    rates = load_treasury_annual_rates(
        Path(str(settings["input_file"])),
        date_column=str(settings["date_column"]),
        yield_column=str(settings["yield_column"]),
        availability_lag_days=int(settings["availability_lag_days"]),
    )
    return rates, settings

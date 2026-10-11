"""Point-in-time currency conversion into one portfolio base currency."""

from __future__ import annotations

from pathlib import Path
from typing import Mapping

import pandas as pd


FX_RATE_COLUMNS = {
    "rate_time_utc",
    "quote_currency",
    "base_currency",
    "rate_to_base",
}


def load_fx_rates(path: Path, base_currency: str) -> pd.DataFrame:
    """Load long-format rates expressed as base currency per quote unit.

    For a USD base, an SGD row must answer: how many USD is one SGD worth?
    Example: rate_to_base=0.78 means SGD 1 = USD 0.78.
    """
    if not path.is_file():
        raise FileNotFoundError(f"FX-rate file not found: {path}")
    rates = pd.read_csv(path, low_memory=False)
    missing = FX_RATE_COLUMNS - set(rates.columns)
    if missing:
        raise ValueError(f"FX-rate file is missing columns: {sorted(missing)}")

    rates["rate_time_utc"] = pd.to_datetime(
        rates["rate_time_utc"], errors="coerce", utc=True
    )
    rates["quote_currency"] = (
        rates["quote_currency"].astype("string").str.strip().str.upper()
    )
    rates["base_currency"] = (
        rates["base_currency"].astype("string").str.strip().str.upper()
    )
    rates["rate_to_base"] = pd.to_numeric(
        rates["rate_to_base"], errors="coerce"
    )
    invalid = (
        rates["rate_time_utc"].isna()
        | rates["quote_currency"].isna()
        | rates["rate_to_base"].isna()
        | rates["rate_to_base"].le(0)
    )
    if invalid.any():
        raise ValueError(f"FX-rate file contains {int(invalid.sum())} invalid rows")

    expected_base = base_currency.strip().upper()
    unexpected = set(rates["base_currency"].dropna()) - {expected_base}
    if unexpected:
        raise ValueError(
            f"FX rates contain bases {sorted(unexpected)}; expected {expected_base}"
        )
    duplicate_key = ["rate_time_utc", "quote_currency"]
    if rates.duplicated(subset=duplicate_key).any():
        raise ValueError("FX-rate file contains duplicate currency/timestamp rows")
    return rates.sort_values(duplicate_key).reset_index(drop=True)


def convert_daily_prices_to_base(
    daily_native_prices: pd.DataFrame,
    quote_currencies: Mapping[str, str],
    base_currency: str,
    *,
    static_rates_to_base: Mapping[str, float] | None = None,
    fx_rates: pd.DataFrame | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Convert daily instrument prices and return an auditable conversion table."""
    base = base_currency.strip().upper()
    static = {
        currency.strip().upper(): float(rate)
        for currency, rate in (static_rates_to_base or {}).items()
    }
    static.setdefault(base, 1.0)
    for currency, rate in static.items():
        if rate <= 0:
            raise ValueError(f"Static {currency}/{base} rate must be positive")

    missing_assets = set(daily_native_prices.columns) - set(quote_currencies)
    if missing_assets:
        raise ValueError(
            f"Missing quote currencies for instruments: {sorted(missing_assets)}"
        )

    dynamic: dict[str, pd.Series] = {}
    if fx_rates is not None and not fx_rates.empty:
        unexpected = set(fx_rates["base_currency"].astype(str).str.upper()) - {base}
        if unexpected:
            raise ValueError(f"FX rates must all use {base} as base currency")
        work = fx_rates.copy()
        work["rate_date"] = pd.to_datetime(
            work["rate_time_utc"], utc=True
        ).dt.normalize()
        for currency, rows in work.groupby("quote_currency", sort=False):
            series = (
                rows.sort_values("rate_time_utc")
                .groupby("rate_date")["rate_to_base"]
                .last()
                .astype("float64")
            )
            dynamic[str(currency).upper()] = series

    converted = pd.DataFrame(index=daily_native_prices.index)
    audit_rows: list[dict[str, object]] = []
    for instrument_id in daily_native_prices.columns:
        currency = quote_currencies[instrument_id].strip().upper()
        native = daily_native_prices[instrument_id].dropna().astype("float64")
        if currency in static:
            rate = pd.Series(static[currency], index=native.index, dtype="float64")
            rate_source = "static"
        else:
            if currency not in dynamic:
                raise ValueError(
                    f"No {currency}-to-{base} rates were supplied for {instrument_id}"
                )
            rate = dynamic[currency].reindex(native.index, method="ffill")
            if rate.isna().any():
                first_missing = rate.index[rate.isna()][0]
                raise ValueError(
                    f"No point-in-time {currency}-to-{base} rate is available "
                    f"for {instrument_id} on or before {first_missing.date()}"
                )
            rate_source = "fx_file"

        base_price = native * rate
        converted.loc[base_price.index, instrument_id] = base_price
        audit_rows.extend(
            {
                "session_date": date,
                "instrument_id": instrument_id,
                "quote_currency": currency,
                "base_currency": base,
                "rate_to_base": float(rate.loc[date]),
                "rate_source": rate_source,
                "native_adjusted_close": float(native.loc[date]),
                "base_adjusted_close": float(base_price.loc[date]),
            }
            for date in native.index
        )

    converted = converted.reindex(daily_native_prices.index).sort_index()
    audit = pd.DataFrame(audit_rows).sort_values(
        ["session_date", "instrument_id"]
    ).reset_index(drop=True)
    return converted, audit

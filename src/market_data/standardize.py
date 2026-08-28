"""Map Yahoo and Binance bars into one provider-neutral schema."""

from __future__ import annotations

from datetime import datetime, timezone

import numpy as np
import pandas as pd

from market_data.config import BAR_DURATION, BAR_INTERVAL, Instrument
from market_data.schema import BASE_BAR_COLUMNS


def _as_utc(value: datetime) -> datetime:
    if value.tzinfo is None:
        return value.replace(tzinfo=timezone.utc)
    return value.astimezone(timezone.utc)


def _yahoo_open_times(index: pd.Index, timezone_name: str) -> pd.DatetimeIndex:
    timestamps = pd.DatetimeIndex(index)
    if timestamps.tz is None:
        timestamps = timestamps.tz_localize(
            timezone_name,
            ambiguous="infer",
            nonexistent="shift_forward",
        )
    return timestamps.tz_convert("UTC")


def _numeric(frame: pd.DataFrame, column: str) -> pd.Series:
    if column not in frame.columns:
        return pd.Series(np.nan, index=frame.index, dtype="float64")
    return pd.to_numeric(frame[column], errors="coerce")


def standardize_yahoo_bars(
    raw: pd.DataFrame,
    instrument: Instrument,
    retrieved_at_utc: datetime,
) -> pd.DataFrame:
    """Convert a single-ticker yfinance frame into canonical source bars."""
    if instrument.provider != "yahoo_finance":
        raise ValueError(f"{instrument.source_symbol} is not a Yahoo instrument")

    retrieved = _as_utc(retrieved_at_utc)
    open_times = _yahoo_open_times(raw.index, instrument.session_timezone)
    close_times = open_times + pd.Timedelta(BAR_DURATION)
    adjusted_close = (
        _numeric(raw, "Adj Close")
        if "Adj Close" in raw.columns
        else _numeric(raw, "Close")
    )

    standardized = pd.DataFrame(
        {
            "instrument_id": instrument.instrument_id,
            "source": instrument.provider,
            "source_symbol": instrument.source_symbol,
            "asset_class": instrument.asset_class,
            "exchange": instrument.exchange,
            "bar_interval": BAR_INTERVAL,
            "bar_open_time_utc": open_times,
            "bar_close_time_utc": close_times,
            "session_date_local": open_times.tz_convert(
                instrument.session_timezone
            ).date,
            "open": _numeric(raw, "Open").to_numpy(),
            "high": _numeric(raw, "High").to_numpy(),
            "low": _numeric(raw, "Low").to_numpy(),
            "close": _numeric(raw, "Close").to_numpy(),
            "adjusted_close": adjusted_close.to_numpy(),
            "volume": _numeric(raw, "Volume").to_numpy(),
            "volume_unit": instrument.volume_unit,
            "quote_volume": np.nan,
            "quote_currency": instrument.quote_currency,
            "trade_count": pd.array([pd.NA] * len(raw), dtype="Int64"),
            "is_final": close_times <= pd.Timestamp(retrieved),
            "retrieved_at_utc": pd.Timestamp(retrieved),
        }
    )
    return standardized[BASE_BAR_COLUMNS]


def standardize_binance_bars(
    raw: pd.DataFrame,
    instrument: Instrument,
    retrieved_at_utc: datetime,
) -> pd.DataFrame:
    """Convert Binance kline rows into canonical source bars."""
    if instrument.provider != "binance_spot":
        raise ValueError(f"{instrument.source_symbol} is not a Binance instrument")

    retrieved = _as_utc(retrieved_at_utc)
    open_times = pd.to_datetime(raw["open_time_ms"], unit="ms", utc=True)
    close_times = open_times + pd.Timedelta(BAR_DURATION)
    provider_close_times = pd.to_datetime(raw["close_time_ms"], unit="ms", utc=True)
    close = pd.to_numeric(raw["close"], errors="coerce")

    standardized = pd.DataFrame(
        {
            "instrument_id": instrument.instrument_id,
            "source": instrument.provider,
            "source_symbol": instrument.source_symbol,
            "asset_class": instrument.asset_class,
            "exchange": instrument.exchange,
            "bar_interval": BAR_INTERVAL,
            "bar_open_time_utc": open_times,
            "bar_close_time_utc": close_times,
            "session_date_local": open_times.dt.date,
            "open": pd.to_numeric(raw["open"], errors="coerce"),
            "high": pd.to_numeric(raw["high"], errors="coerce"),
            "low": pd.to_numeric(raw["low"], errors="coerce"),
            "close": close,
            "adjusted_close": close,
            "volume": pd.to_numeric(raw["volume"], errors="coerce"),
            "volume_unit": instrument.volume_unit,
            "quote_volume": pd.to_numeric(raw["quote_volume"], errors="coerce"),
            "quote_currency": instrument.quote_currency,
            "trade_count": pd.to_numeric(
                raw["trade_count"], errors="coerce"
            ).astype("Int64"),
            "is_final": provider_close_times < pd.Timestamp(retrieved),
            "retrieved_at_utc": pd.Timestamp(retrieved),
        }
    )
    return standardized[BASE_BAR_COLUMNS]


def restrict_bar_range(
    bars: pd.DataFrame,
    start_utc: datetime,
    end_utc: datetime,
) -> pd.DataFrame:
    """Apply an exact half-open UTC interval: start <= bar open < end."""
    start = pd.Timestamp(_as_utc(start_utc))
    end = pd.Timestamp(_as_utc(end_utc))
    selected = bars.loc[
        (bars["bar_open_time_utc"] >= start)
        & (bars["bar_open_time_utc"] < end)
    ]
    return selected.reset_index(drop=True)

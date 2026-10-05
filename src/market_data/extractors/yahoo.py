"""Yahoo Finance extraction with no strategy-specific transformations."""

from __future__ import annotations

from datetime import datetime, timedelta

import pandas as pd
import yfinance as yf

from market_data.config import BAR_INTERVAL, Instrument


def _flatten_single_ticker_columns(
    downloaded: pd.DataFrame, ticker: str
) -> pd.DataFrame:
    if not isinstance(downloaded.columns, pd.MultiIndex):
        return downloaded

    for level in range(downloaded.columns.nlevels):
        if ticker in downloaded.columns.get_level_values(level):
            return downloaded.xs(ticker, axis=1, level=level, drop_level=True)

    raise ValueError(f"Could not locate {ticker} in the yfinance response columns")


def fetch_yahoo_bars(
    instrument: Instrument,
    start_utc: datetime,
    end_utc: datetime,
) -> pd.DataFrame:
    """Fetch provider-native five-minute bars for one Yahoo symbol.

    The end date sent to yfinance is exclusive, so one day is added and the
    exact requested range is enforced after standardization.
    """
    if instrument.provider != "yahoo_finance":
        raise ValueError(f"{instrument.source_symbol} is not a Yahoo instrument")

    downloaded = yf.download(
        instrument.source_symbol,
        start=start_utc.date().isoformat(),
        end=(end_utc.date() + timedelta(days=1)).isoformat(),
        interval=BAR_INTERVAL,
        auto_adjust=False,
        actions=False,
        repair=False,
        keepna=True,
        prepost=False,
        progress=False,
        threads=False,
        ignore_tz=False,
        group_by="column",
        timeout=20,
    )

    if downloaded.empty:
        raise RuntimeError(
            f"Yahoo Finance returned no {BAR_INTERVAL} bars for "
            f"{instrument.source_symbol}"
        )

    return _flatten_single_ticker_columns(downloaded, instrument.source_symbol)

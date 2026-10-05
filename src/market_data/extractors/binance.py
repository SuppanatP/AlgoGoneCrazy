"""Public Binance Spot kline extraction with pagination and retries."""

from __future__ import annotations

from datetime import datetime

import pandas as pd
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

from market_data.config import (
    BAR_DURATION,
    BAR_INTERVAL,
    BINANCE_BASE_URL,
    BINANCE_PAGE_LIMIT,
    Instrument,
)
from market_data.schema import BINANCE_KLINE_COLUMNS


def _milliseconds(value: datetime) -> int:
    return int(value.timestamp() * 1000)


def _retrying_session() -> requests.Session:
    retry = Retry(
        total=4,
        connect=4,
        read=4,
        status=4,
        backoff_factor=0.5,
        status_forcelist=(429, 500, 502, 503, 504),
        allowed_methods=frozenset({"GET"}),
    )
    adapter = HTTPAdapter(max_retries=retry)
    session = requests.Session()
    session.mount("https://", adapter)
    return session


def fetch_binance_bars(
    instrument: Instrument,
    start_utc: datetime,
    end_utc: datetime,
    session: requests.Session | None = None,
) -> pd.DataFrame:
    """Fetch provider-native five-minute Binance klines in a UTC range."""
    if instrument.provider != "binance_spot":
        raise ValueError(f"{instrument.source_symbol} is not a Binance instrument")

    endpoint = f"{BINANCE_BASE_URL}/api/v3/klines"
    cursor_ms = _milliseconds(start_utc)
    end_ms = _milliseconds(end_utc) - 1
    interval_ms = int(BAR_DURATION.total_seconds() * 1000)
    all_rows: list[list[object]] = []

    owns_session = session is None
    active_session = session or _retrying_session()

    try:
        while cursor_ms <= end_ms:
            response = active_session.get(
                endpoint,
                params={
                    "symbol": instrument.source_symbol,
                    "interval": BAR_INTERVAL,
                    "startTime": cursor_ms,
                    "endTime": end_ms,
                    "limit": BINANCE_PAGE_LIMIT,
                    "timeZone": "0",
                },
                timeout=30,
            )
            response.raise_for_status()
            batch = response.json()

            if not batch:
                break

            all_rows.extend(batch)
            next_cursor_ms = int(batch[-1][0]) + interval_ms
            if next_cursor_ms <= cursor_ms:
                raise RuntimeError("Binance pagination did not advance")

            cursor_ms = next_cursor_ms
            if len(batch) < BINANCE_PAGE_LIMIT:
                break
    finally:
        if owns_session:
            active_session.close()

    if not all_rows:
        raise RuntimeError(
            f"Binance returned no {BAR_INTERVAL} bars for {instrument.source_symbol}"
        )

    return pd.DataFrame(all_rows, columns=BINANCE_KLINE_COLUMNS)

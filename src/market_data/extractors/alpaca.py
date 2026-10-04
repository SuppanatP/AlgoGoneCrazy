"""Alpaca historical stock-bar extraction using the Market Data REST API."""

from __future__ import annotations

import os
from datetime import datetime

import pandas as pd
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

from market_data.config import (
    ALPACA_BASE_URL,
    ALPACA_PAGE_LIMIT,
    Instrument,
)


def _required_environment(name: str) -> str:
    value = os.getenv(name, "").strip()
    if not value:
        raise RuntimeError(
            f"Missing required environment variable {name}. "
            "Add it to .env or export it before running ingestion."
        )
    return value


def _rfc3339(value: datetime) -> str:
    timestamp = pd.Timestamp(value)
    if timestamp.tzinfo is None:
        timestamp = timestamp.tz_localize("UTC")
    else:
        timestamp = timestamp.tz_convert("UTC")
    return timestamp.isoformat().replace("+00:00", "Z")


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
    session = requests.Session()
    session.mount("https://", HTTPAdapter(max_retries=retry))
    return session


def _fetch_adjustment(
    active_session: requests.Session,
    endpoint: str,
    headers: dict[str, str],
    instrument: Instrument,
    start_utc: datetime,
    end_utc: datetime,
    feed: str,
    adjustment: str,
) -> pd.DataFrame:
    """Fetch every page for one Alpaca corporate-action adjustment mode."""
    base_parameters: dict[str, object] = {
        "symbols": instrument.source_symbol,
        "timeframe": "5Min",
        "start": _rfc3339(start_utc),
        "end": _rfc3339(end_utc),
        "limit": ALPACA_PAGE_LIMIT,
        "adjustment": adjustment,
        "feed": feed,
        "sort": "asc",
    }
    all_rows: list[dict[str, object]] = []
    page_token: str | None = None
    seen_tokens: set[str] = set()

    while True:
        parameters = dict(base_parameters)
        if page_token:
            parameters["page_token"] = page_token
        response = active_session.get(
            endpoint,
            params=parameters,
            headers=headers,
            timeout=30,
        )
        try:
            response.raise_for_status()
        except requests.HTTPError as error:
            detail = response.text[:500]
            raise RuntimeError(
                f"Alpaca {adjustment} request failed with HTTP "
                f"{response.status_code}: {detail}"
            ) from error

        payload = response.json()
        bars = payload.get("bars", {}).get(instrument.source_symbol, [])
        all_rows.extend(bars)
        next_token = payload.get("next_page_token")
        if not next_token:
            break
        if next_token in seen_tokens:
            raise RuntimeError(
                f"Alpaca {adjustment} pagination token did not advance"
            )
        seen_tokens.add(next_token)
        page_token = next_token

    if not all_rows:
        raise RuntimeError(
            f"Alpaca returned no {adjustment} 5m bars for "
            f"{instrument.source_symbol}"
        )
    return pd.DataFrame(all_rows)


def _combine_raw_and_adjusted(
    raw_bars: pd.DataFrame,
    adjusted_bars: pd.DataFrame,
    source_symbol: str,
) -> pd.DataFrame:
    """Attach adjusted closes without replacing executable raw OHLCV values."""
    required = {"t", "c"}
    for label, frame in (("raw", raw_bars), ("adjusted", adjusted_bars)):
        missing_columns = required - set(frame.columns)
        if missing_columns:
            raise RuntimeError(
                f"Alpaca {label} response for {source_symbol} is missing "
                f"columns: {sorted(missing_columns)}"
            )
        if frame["t"].duplicated().any():
            raise RuntimeError(
                f"Alpaca {label} response for {source_symbol} contains "
                "duplicate timestamps"
            )

    raw_timestamps = pd.Index(raw_bars["t"])
    adjusted_timestamps = pd.Index(adjusted_bars["t"])
    missing_adjusted = raw_timestamps.difference(adjusted_timestamps)
    extra_adjusted = adjusted_timestamps.difference(raw_timestamps)
    if len(missing_adjusted) or len(extra_adjusted):
        raise RuntimeError(
            f"Alpaca raw and adjusted timestamps do not match for "
            f"{source_symbol}: {len(missing_adjusted)} raw bars lack an "
            f"adjusted match and {len(extra_adjusted)} adjusted bars lack a "
            "raw match"
        )

    adjusted_closes = adjusted_bars[["t", "c"]].rename(
        columns={"c": "adjusted_c"}
    )
    return raw_bars.merge(
        adjusted_closes,
        on="t",
        how="left",
        sort=False,
        validate="one_to_one",
    )


def fetch_alpaca_bars(
    instrument: Instrument,
    start_utc: datetime,
    end_utc: datetime,
    session: requests.Session | None = None,
) -> pd.DataFrame:
    """Fetch raw bars and attach an all-actions-adjusted closing price."""
    if instrument.provider != "alpaca_market_data":
        raise ValueError(f"{instrument.source_symbol} is not an Alpaca instrument")

    api_key = _required_environment("ALPACA_API_KEY")
    secret_key = _required_environment("ALPACA_SECRET_KEY")
    feed = os.getenv("ALPACA_DATA_FEED", "sip").strip().lower()
    if feed not in {"sip", "iex", "delayed_sip"}:
        raise ValueError(
            "ALPACA_DATA_FEED must be sip, iex, or delayed_sip"
        )

    endpoint = f"{ALPACA_BASE_URL}/v2/stocks/bars"
    headers = {
        "accept": "application/json",
        "APCA-API-KEY-ID": api_key,
        "APCA-API-SECRET-KEY": secret_key,
    }
    owns_session = session is None
    active_session = session or _retrying_session()

    try:
        raw_bars = _fetch_adjustment(
            active_session,
            endpoint,
            headers,
            instrument,
            start_utc,
            end_utc,
            feed,
            "raw",
        )
        adjusted_bars = _fetch_adjustment(
            active_session,
            endpoint,
            headers,
            instrument,
            start_utc,
            end_utc,
            feed,
            "all",
        )
    finally:
        if owns_session:
            active_session.close()

    return _combine_raw_and_adjusted(
        raw_bars,
        adjusted_bars,
        instrument.source_symbol,
    )

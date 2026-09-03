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


def fetch_alpaca_bars(
    instrument: Instrument,
    start_utc: datetime,
    end_utc: datetime,
    session: requests.Session | None = None,
) -> pd.DataFrame:
    """Fetch raw five-minute stock bars for one Alpaca symbol."""
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
    base_parameters: dict[str, object] = {
        "symbols": instrument.source_symbol,
        "timeframe": "5Min",
        "start": _rfc3339(start_utc),
        "end": _rfc3339(end_utc),
        "limit": ALPACA_PAGE_LIMIT,
        "adjustment": "raw",
        "feed": feed,
        "sort": "asc",
    }
    all_rows: list[dict[str, object]] = []
    page_token: str | None = None
    seen_tokens: set[str] = set()
    owns_session = session is None
    active_session = session or _retrying_session()

    try:
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
                    f"Alpaca request failed with HTTP {response.status_code}: "
                    f"{detail}"
                ) from error

            payload = response.json()
            bars = payload.get("bars", {}).get(instrument.source_symbol, [])
            all_rows.extend(bars)
            next_token = payload.get("next_page_token")
            if not next_token:
                break
            if next_token in seen_tokens:
                raise RuntimeError("Alpaca pagination token did not advance")
            seen_tokens.add(next_token)
            page_token = next_token
    finally:
        if owns_session:
            active_session.close()

    if not all_rows:
        raise RuntimeError(
            f"Alpaca returned no 5m bars for {instrument.source_symbol}"
        )
    return pd.DataFrame(all_rows)

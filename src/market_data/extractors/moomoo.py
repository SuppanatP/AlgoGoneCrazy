"""Moomoo historical candlestick extraction through a local OpenD session."""

from __future__ import annotations

import os
import time
from datetime import datetime

import pandas as pd

from market_data.config import (
    MOOMOO_PAGE_DELAY_SECONDS,
    MOOMOO_PAGE_LIMIT,
    Instrument,
)


def _local_request_dates(
    start_utc: datetime,
    end_utc: datetime,
    timezone_name: str,
) -> tuple[str, str]:
    start = pd.Timestamp(start_utc)
    end = pd.Timestamp(end_utc)
    start = start.tz_localize("UTC") if start.tzinfo is None else start.tz_convert("UTC")
    end = end.tz_localize("UTC") if end.tzinfo is None else end.tz_convert("UTC")
    inclusive_end = end - pd.Timedelta(nanoseconds=1)
    return (
        start.tz_convert(timezone_name).date().isoformat(),
        inclusive_end.tz_convert(timezone_name).date().isoformat(),
    )


def _opend_connection() -> tuple[str, int]:
    host = os.getenv("MOOMOO_OPEND_HOST", "127.0.0.1").strip()
    port_value = os.getenv("MOOMOO_OPEND_PORT", "11111").strip()
    try:
        port = int(port_value)
    except ValueError as error:
        raise ValueError("MOOMOO_OPEND_PORT must be an integer") from error
    return host, port


def fetch_moomoo_bars(
    instrument: Instrument,
    start_utc: datetime,
    end_utc: datetime,
) -> pd.DataFrame:
    """Fetch raw five-minute SGX bars from a running Moomoo OpenD."""
    if instrument.provider != "moomoo_openapi":
        raise ValueError(f"{instrument.source_symbol} is not a Moomoo instrument")

    try:
        from moomoo import (
            AuType,
            KLType,
            OpenQuoteContext,
            RET_OK,
        )
    except ImportError as error:
        raise RuntimeError(
            "The moomoo-api package is required. Install requirements.txt and "
            "ensure Moomoo OpenD is installed, running, and logged in."
        ) from error

    host, port = _opend_connection()
    start_date, end_date = _local_request_dates(
        start_utc,
        end_utc,
        instrument.session_timezone,
    )
    context = OpenQuoteContext(host=host, port=port)
    frames: list[pd.DataFrame] = []
    page_key = None
    seen_page_keys: set[bytes] = set()

    try:
        while True:
            result, data, next_page_key = context.request_history_kline(
                code=instrument.source_symbol,
                start=start_date,
                end=end_date,
                ktype=KLType.K_5M,
                autype=AuType.NONE,
                max_count=MOOMOO_PAGE_LIMIT,
                page_req_key=page_key,
            )
            if result != RET_OK:
                raise RuntimeError(
                    f"Moomoo failed to return {instrument.source_symbol}: {data}"
                )
            if not data.empty:
                frames.append(data.copy())
            if next_page_key is None:
                break
            if next_page_key in seen_page_keys:
                raise RuntimeError("Moomoo pagination key did not advance")
            seen_page_keys.add(next_page_key)
            page_key = next_page_key
            time.sleep(MOOMOO_PAGE_DELAY_SECONDS)
    finally:
        context.close()

    if not frames:
        raise RuntimeError(
            f"Moomoo returned no 5m bars for {instrument.source_symbol}"
        )
    return pd.concat(frames, ignore_index=True)

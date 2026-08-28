"""Instrument metadata and ingestion settings."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import timedelta
from typing import Literal


Provider = Literal["yahoo_finance", "binance_spot"]


@dataclass(frozen=True)
class Instrument:
    """Metadata required to interpret one provider symbol."""

    instrument_id: str
    provider: Provider
    source_symbol: str
    asset_class: str
    exchange: str
    quote_currency: str
    volume_unit: str
    session_timezone: str


BAR_INTERVAL = "5m"
BAR_DURATION = timedelta(minutes=5)
DEFAULT_LOOKBACK_DAYS = 59

BINANCE_BASE_URL = "https://data-api.binance.vision"
BINANCE_PAGE_LIMIT = 1000

INSTRUMENTS = (
    Instrument(
        instrument_id="spy_usd",
        provider="yahoo_finance",
        source_symbol="SPY",
        asset_class="equity_etf",
        exchange="NYSE_ARCA",
        quote_currency="USD",
        volume_unit="shares",
        session_timezone="America/New_York",
    ),
    Instrument(
        instrument_id="sti_es3_sgd",
        provider="yahoo_finance",
        source_symbol="ES3.SI",
        asset_class="equity_etf",
        exchange="SGX",
        quote_currency="SGD",
        volume_unit="units",
        session_timezone="Asia/Singapore",
    ),
    Instrument(
        instrument_id="btc_usdt",
        provider="binance_spot",
        source_symbol="BTCUSDT",
        asset_class="crypto_spot",
        exchange="BINANCE",
        quote_currency="USDT",
        volume_unit="BTC",
        session_timezone="UTC",
    ),
)

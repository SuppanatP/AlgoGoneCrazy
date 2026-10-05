"""Instrument metadata and ingestion settings."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import time, timedelta
from typing import Literal


Provider = Literal[
    "alpaca_market_data",
    "moomoo_openapi",
    "yahoo_finance",
    "binance_spot",
]


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
    calendar_name: str = "24/7"
    regular_break_start_local: time | None = None
    regular_break_end_local: time | None = None


BAR_INTERVAL = "5m"
PANDAS_BAR_FREQUENCY = "5min"
BAR_DURATION = timedelta(minutes=5)
AGGREGATION_INTERVAL = "30m"
PANDAS_AGGREGATION_FREQUENCY = "30min"
AGGREGATION_DURATION = timedelta(minutes=30)
EXPECTED_SUBBAR_COUNT = 6
AGGREGATION_VERSION = "v1"
BACKFILL_VERSION = "v1"
MIN_BACKFILL_OBSERVED_SUBBARS = 4
DEFAULT_LOOKBACK_DAYS = 3 * 365
DEFAULT_DATA_DELAY_MINUTES = 20

ALPACA_BASE_URL = "https://data.alpaca.markets"
ALPACA_PAGE_LIMIT = 10_000
BINANCE_BASE_URL = "https://data-api.binance.vision"
BINANCE_PAGE_LIMIT = 1000
MOOMOO_PAGE_LIMIT = 1000
MOOMOO_PAGE_DELAY_SECONDS = 0.55

INSTRUMENTS = (
    Instrument(
        instrument_id="spy_usd",
        provider="alpaca_market_data",
        source_symbol="SPY",
        asset_class="equity_etf",
        exchange="NYSE_ARCA",
        quote_currency="USD",
        volume_unit="shares",
        session_timezone="America/New_York",
        calendar_name="XNYS",
    ),
    Instrument(
        instrument_id="tlt_usd",
        provider="alpaca_market_data",
        source_symbol="TLT",
        asset_class="fixed_income_etf",
        exchange="NASDAQ",
        quote_currency="USD",
        volume_unit="shares",
        session_timezone="America/New_York",
        calendar_name="XNYS",
    ),
    Instrument(
        instrument_id="sti_es3_sgd",
        provider="moomoo_openapi",
        source_symbol="SG.ES3",
        asset_class="equity_etf",
        exchange="SGX",
        quote_currency="SGD",
        volume_unit="units",
        session_timezone="Asia/Singapore",
        calendar_name="XSES",
        regular_break_start_local=time(12, 0),
        regular_break_end_local=time(13, 0),
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
        calendar_name="24/7",
    ),
)

# Optional provider variants. They are available through the ingestion selectors
# but are not included in the default all-instrument run.
ALTERNATIVE_INSTRUMENTS = (
    Instrument(
        instrument_id="spy_usd",
        provider="yahoo_finance",
        source_symbol="SPY",
        asset_class="equity_etf",
        exchange="NYSE_ARCA",
        quote_currency="USD",
        volume_unit="shares",
        session_timezone="America/New_York",
        calendar_name="XNYS",
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
        calendar_name="XSES",
        regular_break_start_local=time(12, 0),
        regular_break_end_local=time(13, 0),
    ),
)

AVAILABLE_INSTRUMENTS = INSTRUMENTS + ALTERNATIVE_INSTRUMENTS

PROVIDER_ALIASES = {
    "alpaca": "alpaca_market_data",
    "alpaca_market_data": "alpaca_market_data",
    "moomoo": "moomoo_openapi",
    "moomoo_openapi": "moomoo_openapi",
    "yahoo": "yahoo_finance",
    "yfinance": "yahoo_finance",
    "yahoo_finance": "yahoo_finance",
    "binance": "binance_spot",
    "binance_spot": "binance_spot",
}


def canonical_provider(value: str) -> Provider:
    """Resolve a short provider name to the canonical stored source value."""
    normalized = value.strip().lower()
    try:
        return PROVIDER_ALIASES[normalized]  # type: ignore[return-value]
    except KeyError as error:
        choices = ", ".join(sorted(PROVIDER_ALIASES))
        raise ValueError(
            f"Unknown provider {value!r}. Available names: {choices}"
        ) from error


def select_instruments(
    provider: str | None = None,
    symbol: str | None = None,
) -> tuple[Instrument, ...]:
    """Select configured provider-symbol pairs for one ingestion run."""
    if provider is None and symbol is None:
        return INSTRUMENTS

    canonical = canonical_provider(provider) if provider else None
    normalized_symbol = symbol.strip().upper() if symbol else None
    selected = tuple(
        instrument
        for instrument in AVAILABLE_INSTRUMENTS
        if (canonical is None or instrument.provider == canonical)
        and (
            normalized_symbol is None
            or instrument.source_symbol.upper() == normalized_symbol
        )
    )
    if not selected:
        configured = ", ".join(
            f"{item.provider}:{item.source_symbol}"
            for item in AVAILABLE_INSTRUMENTS
        )
        raise ValueError(
            "No configured instrument matches the requested provider and symbol. "
            f"Configured pairs: {configured}"
        )
    if provider is None and len(selected) > 1:
        matches = ", ".join(
            f"{item.provider}:{item.source_symbol}" for item in selected
        )
        raise ValueError(
            f"Symbol {symbol!r} is available from multiple providers: {matches}. "
            "Add --provider to select one."
        )
    return selected

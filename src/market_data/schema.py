"""Canonical column definitions for standardized source bars."""

BASE_BAR_COLUMNS = [
    "instrument_id",
    "source",
    "source_symbol",
    "asset_class",
    "exchange",
    "bar_interval",
    "bar_open_time_utc",
    "bar_close_time_utc",
    "session_date_local",
    "open",
    "high",
    "low",
    "close",
    "adjusted_close",
    "volume",
    "volume_unit",
    "quote_volume",
    "quote_currency",
    "trade_count",
    "is_final",
    "retrieved_at_utc",
]

STANDARD_BAR_COLUMNS = BASE_BAR_COLUMNS + [
    "quality_status",
    "quality_flags",
]

REJECTED_BAR_COLUMNS = STANDARD_BAR_COLUMNS + ["rejection_reason"]

BAR_KEY_COLUMNS = [
    "instrument_id",
    "source",
    "bar_interval",
    "bar_open_time_utc",
]

BINANCE_KLINE_COLUMNS = [
    "open_time_ms",
    "open",
    "high",
    "low",
    "close",
    "volume",
    "close_time_ms",
    "quote_volume",
    "trade_count",
    "taker_buy_base_volume",
    "taker_buy_quote_volume",
    "unused",
]

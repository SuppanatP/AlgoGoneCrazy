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

COVERAGE_GAP_COLUMNS = [
    "instrument_id",
    "source_symbol",
    "exchange",
    "calendar_name",
    "bar_interval",
    "expected_bar_open_time_utc",
    "expected_bar_close_time_utc",
    "session_date_local",
    "gap_reason",
    "detected_at_utc",
]

SESSION_COVERAGE_COLUMNS = [
    "instrument_id",
    "source_symbol",
    "exchange",
    "calendar_name",
    "session_date_local",
    "session_status",
    "session_open_time_utc",
    "break_start_time_utc",
    "break_end_time_utc",
    "session_close_time_utc",
    "expected_bar_count",
    "accepted_bar_count",
    "missing_bar_count",
    "coverage_ratio",
]

EXPECTED_BAR_COLUMNS = [
    "instrument_id",
    "source_symbol",
    "exchange",
    "calendar_name",
    "bar_interval",
    "bar_open_time_utc",
    "bar_close_time_utc",
    "session_date_local",
]

AGGREGATED_BAR_COLUMNS = [
    "instrument_id",
    "source",
    "source_symbol",
    "asset_class",
    "exchange",
    "bar_interval",
    "source_bar_interval",
    "aggregation_source",
    "aggregation_version",
    "bar_origin",
    "backfill_method",
    "backfill_version",
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
    "expected_subbar_count",
    "actual_subbar_count",
    "synthetic_subbar_count",
    "missing_subbar_count",
    "missing_bar_positions",
    "coverage_ratio",
    "open_source",
    "close_source",
    "range_is_partial",
    "volume_is_partial",
    "quality_status",
    "quality_flags",
    "latest_source_retrieved_at_utc",
    "generated_at_utc",
    "backfilled_at_utc",
]

INCOMPLETE_AGGREGATE_COLUMNS = [
    "instrument_id",
    "source",
    "source_symbol",
    "asset_class",
    "exchange",
    "bar_interval",
    "source_bar_interval",
    "bar_open_time_utc",
    "bar_close_time_utc",
    "session_date_local",
    "expected_subbar_count",
    "actual_subbar_count",
    "coverage_ratio",
    "missing_bar_times_utc",
    "rejection_reason",
    "generated_at_utc",
]

UNRESOLVED_BACKFILL_COLUMNS = INCOMPLETE_AGGREGATE_COLUMNS + [
    "attempted_method",
    "unresolved_reason",
    "evaluated_at_utc",
]

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

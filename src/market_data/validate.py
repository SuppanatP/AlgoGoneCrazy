"""Provider-neutral validation for standardized source bars."""

from __future__ import annotations

import numpy as np
import pandas as pd

from market_data.schema import (
    BAR_KEY_COLUMNS,
    BASE_BAR_COLUMNS,
    REJECTED_BAR_COLUMNS,
    STANDARD_BAR_COLUMNS,
)


PRICE_COLUMNS = ["open", "high", "low", "close", "adjusted_close"]


def _append_flag(target: list[list[str]], mask: pd.Series, message: str) -> None:
    for position in np.flatnonzero(mask.fillna(False).to_numpy()):
        target[position].append(message)


def validate_bars(bars: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Split standardized bars into accepted and rejected frames.

    Impossible or incomplete rows are rejected. Plausible but unusual rows are
    retained with warning flags. No prices are forward-filled or imputed.
    """
    missing_columns = set(BASE_BAR_COLUMNS) - set(bars.columns)
    if missing_columns:
        raise ValueError(f"Missing standardized columns: {sorted(missing_columns)}")

    work = bars[BASE_BAR_COLUMNS].copy().reset_index(drop=True)
    for column in PRICE_COLUMNS + ["volume", "quote_volume", "trade_count"]:
        work[column] = pd.to_numeric(work[column], errors="coerce")

    work["bar_open_time_utc"] = pd.to_datetime(
        work["bar_open_time_utc"], errors="coerce", utc=True
    )
    work["bar_close_time_utc"] = pd.to_datetime(
        work["bar_close_time_utc"], errors="coerce", utc=True
    )
    work["retrieved_at_utc"] = pd.to_datetime(
        work["retrieved_at_utc"], errors="coerce", utc=True
    )

    warnings: list[list[str]] = [[] for _ in range(len(work))]
    rejections: list[list[str]] = [[] for _ in range(len(work))]

    missing_times = work[
        ["bar_open_time_utc", "bar_close_time_utc", "retrieved_at_utc"]
    ].isna().any(axis=1)
    _append_flag(rejections, missing_times, "missing_timestamp")

    invalid_time_order = work["bar_close_time_utc"] <= work["bar_open_time_utc"]
    _append_flag(rejections, invalid_time_order, "invalid_bar_time_order")
    _append_flag(rejections, ~work["is_final"].fillna(False), "incomplete_bar")

    missing_prices = work[PRICE_COLUMNS].isna().any(axis=1)
    _append_flag(rejections, missing_prices, "missing_price")

    non_positive_prices = (work[PRICE_COLUMNS] <= 0).any(axis=1)
    _append_flag(rejections, non_positive_prices, "non_positive_price")

    high_too_low = work["high"] < work[["open", "close", "low"]].max(axis=1)
    low_too_high = work["low"] > work[["open", "close", "high"]].min(axis=1)
    _append_flag(rejections, high_too_low | low_too_high, "inconsistent_ohlc")

    _append_flag(rejections, work["volume"] < 0, "negative_volume")
    _append_flag(rejections, work["quote_volume"] < 0, "negative_quote_volume")
    _append_flag(rejections, work["trade_count"] < 0, "negative_trade_count")

    duplicate_rows = work.duplicated(subset=BAR_KEY_COLUMNS, keep="last")
    _append_flag(rejections, duplicate_rows, "duplicate_bar")

    _append_flag(warnings, work["volume"] == 0, "zero_volume")
    relative_range = (work["high"] - work["low"]) / work["close"]
    _append_flag(warnings, relative_range > 0.20, "extreme_intrabar_range")

    work["quality_status"] = ["warning" if flags else "pass" for flags in warnings]
    work["quality_flags"] = [";".join(flags) for flags in warnings]
    work["rejection_reason"] = [";".join(flags) for flags in rejections]

    rejected_mask = work["rejection_reason"].ne("")
    accepted = work.loc[~rejected_mask, STANDARD_BAR_COLUMNS].copy()
    rejected = work.loc[rejected_mask, REJECTED_BAR_COLUMNS].copy()
    rejected["quality_status"] = "reject"

    sort_columns = ["instrument_id", "bar_open_time_utc"]
    accepted = accepted.sort_values(sort_columns).reset_index(drop=True)
    rejected = rejected.sort_values(sort_columns).reset_index(drop=True)
    return accepted, rejected

"""Combine strategy signals with risk weights and convert them to quantities."""

from __future__ import annotations

from collections.abc import Mapping

import numpy as np
import pandas as pd


def apply_signals_to_weights(
    weight_schedule: pd.DataFrame,
    signals: pd.DataFrame,
    *,
    allow_short: bool = False,
) -> pd.DataFrame:
    """Apply weekly/daily risk budgets to timestamped strategy targets.

    Signals are target exposures rather than trade commands. Inactive strategy
    allocations remain cash; active assets are not renormalized upward.
    """
    if weight_schedule.empty:
        raise ValueError("weight_schedule is empty")
    if signals.empty:
        raise ValueError("signals are empty")
    if signals.isna().any().any():
        raise ValueError("signals contain missing values")
    missing_assets = set(signals.columns) - set(weight_schedule.columns)
    if missing_assets:
        raise ValueError(
            f"Weight schedule is missing signal assets: {sorted(missing_assets)}"
        )

    numeric = signals.apply(pd.to_numeric, errors="coerce")
    if numeric.isna().any().any():
        raise ValueError("signals must be numeric")
    if allow_short:
        invalid = numeric.lt(-1) | numeric.gt(1)
    else:
        invalid = numeric.lt(0) | numeric.gt(1)
    if invalid.any().any():
        allowed = "[-1, 1]" if allow_short else "[0, 1]"
        raise ValueError(f"signals must lie within {allowed}")

    signal_index = pd.DatetimeIndex(pd.to_datetime(numeric.index, utc=True))
    numeric.index = signal_index
    schedule = weight_schedule.copy()
    schedule.index = pd.DatetimeIndex(pd.to_datetime(schedule.index, utc=True))
    schedule = schedule.sort_index()
    if schedule.index.has_duplicates:
        raise ValueError("weight_schedule contains duplicate timestamps")

    aligned = schedule.reindex(signal_index, method="ffill")
    if aligned.isna().any().any():
        raise ValueError(
            "At least one signal timestamp precedes the first available weight"
        )
    target = aligned[numeric.columns] * numeric
    target.index.name = signals.index.name
    return target


def target_quantities(
    target_weights: Mapping[str, float],
    portfolio_equity_base: float,
    prices_in_base_currency: Mapping[str, float],
    quantity_steps: Mapping[str, float],
    *,
    allow_short: bool = False,
) -> dict[str, float]:
    """Convert target weights to units using prices in one common currency.

    Callers must convert every price into the portfolio base currency first.
    This function deliberately cannot infer or mix USD, SGD, and USDT prices.
    """
    if portfolio_equity_base <= 0 or not np.isfinite(portfolio_equity_base):
        raise ValueError("portfolio_equity_base must be finite and positive")
    assets = set(target_weights)
    missing_prices = assets - set(prices_in_base_currency)
    missing_steps = assets - set(quantity_steps)
    if missing_prices:
        raise ValueError(f"Missing base-currency prices for: {sorted(missing_prices)}")
    if missing_steps:
        raise ValueError(f"Missing quantity steps for: {sorted(missing_steps)}")

    quantities: dict[str, float] = {}
    gross_target_weight = sum(abs(float(value)) for value in target_weights.values())
    if gross_target_weight > 1.0 + 1e-12:
        raise ValueError(
            "Gross target weight exceeds 100%; leverage is not supported"
        )
    for asset, raw_weight in target_weights.items():
        weight = float(raw_weight)
        if not np.isfinite(weight):
            raise ValueError(f"{asset} target weight must be finite")
        if allow_short:
            if abs(weight) > 1:
                raise ValueError(f"{asset} target weight must be within [-1, 1]")
        elif not 0 <= weight <= 1:
            raise ValueError(f"{asset} target weight must be within [0, 1]")

        price = float(prices_in_base_currency[asset])
        step = float(quantity_steps[asset])
        if price <= 0 or not np.isfinite(price):
            raise ValueError(f"{asset} base-currency price must be positive")
        if step <= 0 or not np.isfinite(step):
            raise ValueError(f"{asset} quantity step must be positive")

        unrounded = weight * portfolio_equity_base / price
        quantity = np.sign(unrounded) * np.floor(abs(unrounded) / step) * step
        quantities[asset] = float(quantity)
    return quantities

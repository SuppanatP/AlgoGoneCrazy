"""Long/cash momentum strategy based on adjusted closing prices."""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from trading_strategy.common import prepare_strategy_bars, sort_signal_output


@dataclass(frozen=True)
class MomentumParameters:
    """Parameters for the EMA and trailing-return momentum rule."""

    fast_window: int = 13
    slow_window: int = 52
    momentum_lookback: int = 26
    entry_threshold: float = 0.0
    exit_threshold: float = 0.0

    def validate(self) -> None:
        if self.fast_window < 1:
            raise ValueError("fast_window must be at least 1")
        if self.slow_window <= self.fast_window:
            raise ValueError("slow_window must be greater than fast_window")
        if self.momentum_lookback < 1:
            raise ValueError("momentum_lookback must be at least 1")
        if self.exit_threshold > self.entry_threshold:
            raise ValueError(
                "exit_threshold must not be greater than entry_threshold"
            )


def _signals_for_instrument(
    bars: pd.DataFrame,
    parameters: MomentumParameters,
) -> pd.DataFrame:
    work = bars.copy()
    price = work["adjusted_close"]
    work["momentum_fast_ema"] = price.ewm(
        span=parameters.fast_window,
        adjust=False,
        min_periods=parameters.fast_window,
    ).mean()
    work["momentum_slow_ema"] = price.ewm(
        span=parameters.slow_window,
        adjust=False,
        min_periods=parameters.slow_window,
    ).mean()
    work["momentum_return"] = price.pct_change(
        periods=parameters.momentum_lookback,
        fill_method=None,
    )
    work["momentum_trend_spread"] = (
        work["momentum_fast_ema"] / work["momentum_slow_ema"] - 1.0
    )

    feature_columns = [
        "momentum_fast_ema",
        "momentum_slow_ema",
        "momentum_return",
        "momentum_trend_spread",
    ]
    features_ready = work[feature_columns].notna().all(axis=1)

    position = 0
    signals: list[str] = []
    target_positions: list[int] = []
    reasons: list[str] = []

    for position_index in range(len(work)):
        row = work.iloc[position_index]
        if not features_ready.iloc[position_index]:
            signal = "HOLD"
            reason = "warmup"
        elif not bool(row["signal_eligible"]):
            signal = "HOLD"
            reason = "bar_not_signal_eligible"
        else:
            entry = (
                row["momentum_fast_ema"] > row["momentum_slow_ema"]
                and row["momentum_return"] > parameters.entry_threshold
            )
            exit_position = (
                row["momentum_fast_ema"] <= row["momentum_slow_ema"]
                or row["momentum_return"] <= parameters.exit_threshold
            )
            if position == 0 and entry:
                position = 1
                signal = "BUY"
                reason = "momentum_entry"
            elif position == 1 and exit_position:
                position = 0
                signal = "SELL"
                reason = "momentum_exit"
            elif position == 1:
                signal = "HOLD"
                reason = "long_condition_active"
            else:
                signal = "HOLD"
                reason = "entry_condition_not_met"

        signals.append(signal)
        target_positions.append(position)
        reasons.append(reason)

    work["strategy_name"] = "momentum"
    work["signal"] = signals
    work["target_position"] = pd.Series(target_positions, dtype="int8")
    work["signal_strength"] = work["momentum_return"]
    work["signal_reason"] = reasons
    return work


def generate_momentum_signals(
    bars: pd.DataFrame,
    parameters: MomentumParameters | None = None,
) -> pd.DataFrame:
    """Generate long/cash momentum signals for each instrument independently.

    A signal is known only after its bar closes. A later execution module must
    delay any resulting order and fill it using a raw tradable price.
    """
    selected_parameters = parameters or MomentumParameters()
    selected_parameters.validate()
    prepared = prepare_strategy_bars(bars)
    frames = [
        _signals_for_instrument(group.reset_index(drop=True), selected_parameters)
        for _, group in prepared.groupby("instrument_id", sort=False)
    ]
    return sort_signal_output(pd.concat(frames, ignore_index=True))

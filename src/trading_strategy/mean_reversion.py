"""Long/cash mean-reversion strategy based on an adjusted-price z-score."""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from trading_strategy.common import prepare_strategy_bars, sort_signal_output


@dataclass(frozen=True)
class MeanReversionParameters:
    """Parameters for the rolling z-score mean-reversion rule."""

    window: int = 26
    entry_z_score: float = -1.5
    exit_z_score: float = 0.0
    maximum_holding_bars: int | None = 52
    minimum_holding_bars: int = 0
    cooldown_bars: int = 0

    def validate(self) -> None:
        if self.window < 2:
            raise ValueError("window must be at least 2")
        if self.entry_z_score >= self.exit_z_score:
            raise ValueError("entry_z_score must be below exit_z_score")
        if (
            self.maximum_holding_bars is not None
            and self.maximum_holding_bars < 1
        ):
            raise ValueError("maximum_holding_bars must be at least 1")
        if self.minimum_holding_bars < 0:
            raise ValueError("minimum_holding_bars must not be negative")
        if self.cooldown_bars < 0:
            raise ValueError("cooldown_bars must not be negative")
        if (
            self.maximum_holding_bars is not None
            and self.minimum_holding_bars > self.maximum_holding_bars
        ):
            raise ValueError(
                "minimum_holding_bars must not exceed maximum_holding_bars"
            )


def _signals_for_instrument(
    bars: pd.DataFrame,
    parameters: MeanReversionParameters,
) -> pd.DataFrame:
    work = bars.copy()
    price = work["adjusted_close"]
    rolling = price.rolling(window=parameters.window, min_periods=parameters.window)
    work["mean_reversion_average"] = rolling.mean()
    work["mean_reversion_std"] = rolling.std(ddof=0)
    nonzero_std = work["mean_reversion_std"].where(
        work["mean_reversion_std"].ne(0)
    )
    work["mean_reversion_z_score"] = (
        price - work["mean_reversion_average"]
    ) / nonzero_std
    features_ready = work["mean_reversion_z_score"].notna()

    position = 0
    holding_bars = 0
    cooldown_remaining = 0
    signals: list[str] = []
    target_positions: list[int] = []
    holding_periods: list[int] = []
    cooldown_periods: list[int] = []
    reasons: list[str] = []

    for position_index in range(len(work)):
        row = work.iloc[position_index]
        if position == 1:
            holding_bars += 1

        if not features_ready.iloc[position_index]:
            signal = "HOLD"
            reason = "warmup"
        elif not bool(row["signal_eligible"]):
            signal = "HOLD"
            reason = "bar_not_signal_eligible"
        elif position == 0 and cooldown_remaining > 0:
            cooldown_remaining -= 1
            signal = "HOLD"
            reason = "cooldown"
        elif (
            position == 0
            and row["mean_reversion_z_score"] <= parameters.entry_z_score
        ):
            position = 1
            holding_bars = 1
            signal = "BUY"
            reason = "mean_reversion_entry"
        elif (
            position == 1
            and holding_bars >= parameters.minimum_holding_bars
            and row["mean_reversion_z_score"] >= parameters.exit_z_score
        ):
            position = 0
            holding_bars = 0
            cooldown_remaining = parameters.cooldown_bars
            signal = "SELL"
            reason = "mean_reversion_exit"
        elif (
            position == 1
            and parameters.maximum_holding_bars is not None
            and holding_bars >= parameters.maximum_holding_bars
        ):
            position = 0
            holding_bars = 0
            cooldown_remaining = parameters.cooldown_bars
            signal = "SELL"
            reason = "maximum_holding_exit"
        elif position == 1:
            signal = "HOLD"
            reason = "waiting_for_reversion"
        else:
            signal = "HOLD"
            reason = "entry_condition_not_met"

        signals.append(signal)
        target_positions.append(position)
        holding_periods.append(holding_bars)
        cooldown_periods.append(cooldown_remaining)
        reasons.append(reason)

    work["strategy_name"] = "mean_reversion"
    work["signal"] = signals
    work["target_position"] = pd.Series(target_positions, dtype="int8")
    work["signal_strength"] = -work["mean_reversion_z_score"]
    work["holding_period_bars"] = pd.Series(holding_periods, dtype="int64")
    work["cooldown_remaining_bars"] = pd.Series(
        cooldown_periods, dtype="int64"
    )
    work["signal_reason"] = reasons
    return work


def generate_mean_reversion_signals(
    bars: pd.DataFrame,
    parameters: MeanReversionParameters | None = None,
) -> pd.DataFrame:
    """Generate long/cash mean-reversion signals per instrument.

    A signal is known only after its bar closes. A later execution module must
    delay any resulting order and fill it using a raw tradable price.
    """
    selected_parameters = parameters or MeanReversionParameters()
    selected_parameters.validate()
    prepared = prepare_strategy_bars(bars)
    frames = [
        _signals_for_instrument(group.reset_index(drop=True), selected_parameters)
        for _, group in prepared.groupby("instrument_id", sort=False)
    ]
    return sort_signal_output(pd.concat(frames, ignore_index=True))

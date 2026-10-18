"""Reusable BTC model inference and stateful target generation.

The target returned for timestamp ``t`` is decided from features through ``t-1``
and applies at the open of bar ``t``. Consumers must not delay it again.
"""
from __future__ import annotations

from pathlib import Path

import lightgbm as lgb
import pandas as pd

from .features import build_features
from .strategy import VOTING_INDICATORS, VotingStrategy


INDICATOR_COLS = tuple(
    dict.fromkeys(
        ("rsi_14", "close_over_sma_20", "close_over_sma_50", "bb_pctb_20", "bb_width_20")
        + VOTING_INDICATORS
    )
)


def load_model_pair(model_up: str | Path, model_down: str | Path):
    """Load and validate the selected LightGBM up/down model pair."""
    up = lgb.Booster(model_file=str(model_up))
    down = lgb.Booster(model_file=str(model_down))
    if up.feature_name() != down.feature_name():
        raise ValueError("up and down models use different feature columns")
    return up, down


def prediction_frame(
    history: pd.DataFrame,
    bars: pd.DataFrame,
    asset: str,
    models,
    all_history: dict[str, pd.DataFrame] | None = None,
) -> pd.DataFrame:
    """Return opens/closes, model probabilities, and lagged indicators for ``bars``."""
    up, down = models
    features = build_features(history, asset, all_history or {asset: history})
    model_columns = up.feature_name()
    missing = [column for column in model_columns if column not in features.columns]
    if missing:
        raise KeyError(f"{asset}: build_features() does not produce model columns {missing}")

    model_input = features.loc[bars.index, model_columns]
    nan_rows = int(model_input.isna().any(axis=1).sum())
    if nan_rows:
        print(f"  warning: {asset}: {nan_rows} bars with NaN features (increase LOOKBACK_DAYS?)")

    return (
        bars[["open", "close"]]
        .assign(
            p_up=pd.Series(up.predict(model_input), index=model_input.index),
            p_down=pd.Series(down.predict(model_input), index=model_input.index),
        )
        .join(features.loc[bars.index, list(INDICATOR_COLS)])
    )


def _reason(strategy, signal: int, target: float, before: dict) -> str:
    """Describe the state transition without changing the strategy decision."""
    direction = "long" if target > 0 else "short"
    if before["position"] == 0:
        if signal:
            return f"signal_entry_{direction}"
        if target:
            return f"daily_fallback_{direction}"
        return "flat_no_entry"

    if signal and before["entry_size"] > before["size"]:
        return f"stronger_signal_{direction}"
    if target == 0:
        if before["hit_stop_or_target"]:
            return "stop_or_take_profit_exit"
        if signal == -before["position"] and strategy.exit_on_opposite:
            return "opposite_signal_exit"
        return "flat_after_prior_exit"
    if before["hit_partial"]:
        return "partial_exit"
    if before["bars_left"] == 1:
        return "holding_final_horizon_bar"
    return "holding"


def generate_target_row(strategy, row: pd.Series, timestamp) -> dict:
    """Advance one strategy bar and return auditable target diagnostics.

    This function has no portfolio, cash, quantity, fill, or fee logic. Its
    ``execution_target_position`` applies at ``timestamp``'s bar open.
    """
    indicators = {column: float(row[column]) for column in INDICATOR_COLS}
    p_up, p_down, price = float(row["p_up"]), float(row["p_down"]), float(row["open"])
    votes = strategy.votes(p_up, p_down, indicators) if isinstance(strategy, VotingStrategy) else {}
    vote_score = (
        sum(strategy.weights[name] * value for name, value in votes.items())
        if isinstance(strategy, VotingStrategy)
        else float("nan")
    )
    _, entry_size = strategy.entry(p_up, p_down, indicators)
    before = {
        "position": strategy.position,
        "size": strategy.size,
        "bars_left": strategy.bars_left,
        "entry_size": entry_size,
        "hit_stop_or_target": strategy._hit_stop_or_target(price),
        "hit_partial": strategy._hit_partial(price),
    }
    signal, target = strategy.step(p_up, p_down, indicators, price, timestamp)
    return {
        "p_up": p_up,
        "p_down": p_down,
        **{f"vote_{name}": value for name, value in votes.items()},
        "vote_score": vote_score,
        "entry_signal": signal,
        "execution_target_position": target,
        "signal_reason": _reason(strategy, signal, target, before),
    }

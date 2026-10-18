"""Rolling inverse-volatility or equal weights with concentration caps."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Mapping

import numpy as np
import pandas as pd


SUPPORTED_WEIGHTING_METHODS = {
    "capped_inverse_volatility",
    "capped_equal_weight",
    "capped_confidence_adjusted_inverse_volatility",
    "capped_history_confidence_inverse_volatility",
    "capped_strategy_performance_adjusted_inverse_volatility",
    "capped_strategy_return_inverse_volatility",
}


def configured_maximum_weight(
    sizing_config: Mapping[str, object],
    instrument_id: str,
) -> float:
    """Return an asset-specific concentration limit, with shared fallback."""
    assets = sizing_config.get("assets")
    asset_config = (
        assets.get(instrument_id)
        if isinstance(assets, Mapping)
        else None
    )
    if isinstance(asset_config, Mapping) and "maximum_weight" in asset_config:
        maximum = float(asset_config["maximum_weight"])
    elif sizing_config.get("maximum_asset_weight") is not None:
        maximum = float(sizing_config["maximum_asset_weight"])
    else:
        raise ValueError(
            f"Sizing configuration must set a maximum weight for {instrument_id}"
        )
    if not 0 < maximum <= 1:
        raise ValueError("maximum_asset_weight must be in (0, 1]")
    return maximum


class InsufficientHistoryError(ValueError):
    """Raised when an asset cannot meet the configured volatility history."""

    def __init__(self, observation_count: int, minimum_observations: int) -> None:
        self.observation_count = observation_count
        super().__init__(
            f"only {observation_count} return observations; "
            f"need {minimum_observations}"
        )


@dataclass(frozen=True)
class AssetRiskSettings:
    """Risk and execution metadata for one strategy sleeve."""

    annualization_periods: int
    maximum_weight: float
    quantity_step: float

    def validate(self, asset: str) -> None:
        if self.annualization_periods < 1:
            raise ValueError(f"{asset} annualization_periods must be positive")
        if not 0 < self.maximum_weight <= 1:
            raise ValueError(f"{asset} maximum_weight must be in (0, 1]")
        if self.quantity_step <= 0:
            raise ValueError(f"{asset} quantity_step must be positive")


@dataclass(frozen=True)
class InverseVolatilitySettings:
    """Portfolio-wide weighting policy and common risk-estimation settings."""

    method: str = "capped_inverse_volatility"
    lookback_observations: int = 60
    minimum_observations: int = 40
    target_gross_exposure: float = 0.95
    volatility_floor: float = 0.05
    rebalance_frequency: str = "weekly"
    confidence_full_observations: int = 252
    confidence_power: float = 0.5
    performance_lookback_observations: int = 60
    minimum_performance_observations: int = 20
    performance_sharpe_scale: float = 2.0
    performance_multiplier_floor: float = 0.25
    performance_multiplier_ceiling: float = 1.5

    def validate(self) -> None:
        if self.method not in SUPPORTED_WEIGHTING_METHODS:
            raise ValueError(
                "method must be one of the supported weighting methods"
            )
        if self.lookback_observations < 2:
            raise ValueError("lookback_observations must be at least 2")
        if not 2 <= self.minimum_observations <= self.lookback_observations:
            raise ValueError(
                "minimum_observations must be between 2 and lookback_observations"
            )
        if not 0 < self.target_gross_exposure <= 1:
            raise ValueError("target_gross_exposure must be in (0, 1]")
        if self.volatility_floor <= 0:
            raise ValueError("volatility_floor must be positive")
        if self.rebalance_frequency not in {"daily", "weekly", "initial_only"}:
            raise ValueError(
                "rebalance_frequency must be daily, weekly, or initial_only"
            )
        if self.confidence_full_observations < 1:
            raise ValueError("confidence_full_observations must be positive")
        if self.confidence_power <= 0:
            raise ValueError("confidence_power must be positive")
        if self.performance_lookback_observations < 2:
            raise ValueError(
                "performance_lookback_observations must be at least 2"
            )
        if not 2 <= self.minimum_performance_observations <= self.performance_lookback_observations:
            raise ValueError(
                "minimum_performance_observations must be between 2 and "
                "performance_lookback_observations"
            )
        if self.performance_sharpe_scale <= 0:
            raise ValueError("performance_sharpe_scale must be positive")
        if not 0 < self.performance_multiplier_floor <= self.performance_multiplier_ceiling:
            raise ValueError(
                "performance multiplier bounds must be positive and ordered"
            )


def _utc_date(value: object, label: str) -> pd.Timestamp:
    try:
        timestamp = pd.Timestamp(value)
    except (TypeError, ValueError) as error:
        raise ValueError(f"{label} must be a valid date or timestamp") from error
    if timestamp.tzinfo is None:
        timestamp = timestamp.tz_localize("UTC")
    else:
        timestamp = timestamp.tz_convert("UTC")
    return timestamp.normalize()


def daily_adjusted_closes(
    canonical_bars: pd.DataFrame,
    instruments: Iterable[str] | None = None,
) -> pd.DataFrame:
    """Return one adjusted closing price per local session date and instrument."""
    required = {"instrument_id", "bar_open_time_utc", "adjusted_close"}
    missing = required - set(canonical_bars.columns)
    if missing:
        raise ValueError(f"Canonical bars are missing columns: {sorted(missing)}")

    work = canonical_bars.copy()
    if instruments is not None:
        selected = set(instruments)
        work = work.loc[work["instrument_id"].isin(selected)]
        missing_assets = selected - set(work["instrument_id"].astype(str).unique())
        if missing_assets:
            raise ValueError(
                f"Canonical bars do not contain instruments: {sorted(missing_assets)}"
            )
    if work.empty:
        raise ValueError("No canonical bars are available for portfolio sizing")

    work["bar_open_time_utc"] = pd.to_datetime(
        work["bar_open_time_utc"], errors="coerce", utc=True
    )
    work["adjusted_close"] = pd.to_numeric(
        work["adjusted_close"], errors="coerce"
    )
    invalid = (
        work["bar_open_time_utc"].isna()
        | work["adjusted_close"].isna()
        | work["adjusted_close"].le(0)
    )
    if invalid.any():
        raise ValueError(
            f"Portfolio-sizing input contains {int(invalid.sum())} invalid rows"
        )

    if "session_date_local" in work.columns:
        session_date = pd.to_datetime(
            work["session_date_local"], errors="coerce", utc=True
        )
        if session_date.isna().any():
            raise ValueError("session_date_local contains invalid dates")
        work["session_date"] = session_date.dt.normalize()
    else:
        work["session_date"] = work["bar_open_time_utc"].dt.normalize()

    last_bars = (
        work.sort_values(["instrument_id", "bar_open_time_utc"])
        .groupby(["session_date", "instrument_id"], as_index=False)
        .tail(1)
    )
    prices = last_bars.pivot(
        index="session_date",
        columns="instrument_id",
        values="adjusted_close",
    ).sort_index()
    prices.columns.name = None
    return prices.astype("float64")


def rebalance_dates(
    start: object,
    end: object,
    frequency: str,
) -> pd.DatetimeIndex:
    """Return an initial sizing date followed by daily or Monday updates."""
    start_date = _utc_date(start, "start")
    end_date = _utc_date(end, "end")
    if end_date <= start_date:
        raise ValueError("end must be later than start")
    if frequency == "initial_only":
        scheduled = pd.DatetimeIndex([start_date])
    elif frequency == "daily":
        scheduled = pd.date_range(
            start=start_date,
            end=end_date - pd.Timedelta(days=1),
            freq="D",
            tz="UTC",
        )
    elif frequency == "weekly":
        mondays = pd.date_range(
            start=start_date,
            end=end_date - pd.Timedelta(days=1),
            freq="W-MON",
            tz="UTC",
        )
        scheduled = pd.DatetimeIndex([start_date]).union(mondays)
    else:
        raise ValueError("frequency must be daily or weekly")
    return scheduled.sort_values().unique()


def allocate_capped_scores(
    scores: pd.Series,
    target_gross_exposure: float,
    maximum_weights: Mapping[str, float],
) -> pd.Series:
    if not 0 < target_gross_exposure <= 1:
        raise ValueError("target_gross_exposure must be in (0, 1]")
    values = pd.to_numeric(scores, errors="coerce")
    if values.empty or values.isna().any() or (~np.isfinite(values)).any():
        raise ValueError("Allocation scores must be finite")
    if values.le(0).any():
        raise ValueError("Allocation scores must be positive")

    assets = list(values.index.astype(str))
    missing_caps = set(assets) - set(maximum_weights)
    if missing_caps:
        raise ValueError(f"Missing maximum weights for: {sorted(missing_caps)}")
    caps = {asset: float(maximum_weights[asset]) for asset in assets}
    if any(cap <= 0 or cap > 1 for cap in caps.values()):
        raise ValueError("Maximum weights must be in (0, 1]")

    score_values = {asset: float(values.loc[asset]) for asset in assets}
    weights = {asset: 0.0 for asset in assets}
    remaining = set(assets)
    remaining_budget = float(target_gross_exposure)

    while remaining and remaining_budget > 1e-12:
        score_total = sum(score_values[asset] for asset in remaining)
        if score_total <= 0:
            break
        proposed = {
            asset: remaining_budget * score_values[asset] / score_total
            for asset in remaining
        }
        capped = [
            asset
            for asset in remaining
            if proposed[asset] > caps[asset] + 1e-12
        ]
        if not capped:
            for asset, weight in proposed.items():
                weights[asset] = weight
            remaining_budget = 0.0
            break
        for asset in capped:
            weights[asset] = caps[asset]
            remaining_budget -= caps[asset]
            remaining.remove(asset)
        remaining_budget = max(remaining_budget, 0.0)

    return pd.Series(weights, dtype="float64", name="target_weight")


def allocate_capped_inverse_volatility(
    volatilities: pd.Series,
    target_gross_exposure: float,
    maximum_weights: Mapping[str, float],
) -> pd.Series:
    """Allocate inverse-volatility weights without breaching asset caps.

    If the configured caps cannot absorb the target exposure, the remainder is
    deliberately left as cash rather than violating a cap.
    """
    values = pd.to_numeric(volatilities, errors="coerce")
    if values.empty or values.isna().any() or (~np.isfinite(values)).any():
        raise ValueError("Volatilities must be finite")
    if values.le(0).any():
        raise ValueError("Volatilities must be positive")
    return allocate_capped_scores(
        1.0 / values,
        target_gross_exposure,
        maximum_weights,
    )


def allocate_capped_equal_weight(
    assets: Iterable[str],
    target_gross_exposure: float,
    maximum_weights: Mapping[str, float],
) -> pd.Series:
    """Allocate equal weights without breaching the common concentration cap."""
    asset_list = list(dict.fromkeys(str(asset) for asset in assets))
    if not asset_list:
        raise ValueError("At least one asset is required for equal weighting")
    return allocate_capped_scores(
        pd.Series(1.0, index=asset_list, dtype="float64"),
        target_gross_exposure,
        maximum_weights,
    )


def _volatility_estimate(
    prices: pd.Series,
    as_of_date: pd.Timestamp,
    settings: InverseVolatilitySettings,
    asset_settings: AssetRiskSettings,
) -> tuple[int, float, float]:
    history = prices.loc[prices.index < as_of_date].dropna().tail(
        settings.lookback_observations + 1
    )
    returns = history.pct_change(fill_method=None).dropna()
    observation_count = int(len(returns))
    if observation_count < settings.minimum_observations:
        raise InsufficientHistoryError(
            observation_count,
            settings.minimum_observations,
        )
    annualized = float(
        returns.std(ddof=1) * np.sqrt(asset_settings.annualization_periods)
    )
    if not np.isfinite(annualized):
        raise ValueError("volatility is not finite")
    effective = max(annualized, settings.volatility_floor)
    return observation_count, annualized, effective


def _strategy_return_volatility_estimate(
    returns: pd.Series,
    as_of_date: pd.Timestamp,
    settings: InverseVolatilitySettings,
    asset_settings: AssetRiskSettings,
) -> tuple[int, float, float]:
    """Estimate point-in-time volatility from standalone strategy returns."""
    history = pd.to_numeric(
        returns.loc[returns.index < as_of_date], errors="coerce"
    ).dropna().tail(settings.lookback_observations)
    observation_count = int(len(history))
    if observation_count < settings.minimum_observations:
        raise InsufficientHistoryError(
            observation_count,
            settings.minimum_observations,
        )
    annualized = float(
        history.std(ddof=1) * np.sqrt(asset_settings.annualization_periods)
    )
    if not np.isfinite(annualized):
        raise ValueError("strategy return volatility is not finite")
    effective = max(annualized, settings.volatility_floor)
    return observation_count, annualized, effective


def _confidence_performance_adjustment(
    prices: pd.Series,
    as_of_date: pd.Timestamp,
    settings: InverseVolatilitySettings,
    asset_settings: AssetRiskSettings,
) -> tuple[int, int, float, float, float]:
    """Return point-in-time history confidence and a clipped Sharpe multiplier."""
    history = prices.loc[prices.index < as_of_date].dropna()
    all_returns = history.pct_change(fill_method=None).dropna()
    history_observations = int(len(all_returns))
    performance_returns = all_returns.tail(
        settings.performance_lookback_observations
    )
    performance_observations = int(len(performance_returns))
    if performance_observations < settings.minimum_performance_observations:
        raise InsufficientHistoryError(
            performance_observations,
            settings.minimum_performance_observations,
        )

    return_std = float(performance_returns.std(ddof=1))
    if not np.isfinite(return_std):
        raise ValueError("performance return volatility is not finite")
    if return_std == 0:
        annualized_sharpe = 0.0
    else:
        annualized_sharpe = float(
            performance_returns.mean()
            / return_std
            * np.sqrt(asset_settings.annualization_periods)
        )
    confidence = min(
        1.0,
        (
            history_observations / settings.confidence_full_observations
        )
        ** settings.confidence_power,
    )
    performance_multiplier = float(
        np.clip(
            1.0 + annualized_sharpe / settings.performance_sharpe_scale,
            settings.performance_multiplier_floor,
            settings.performance_multiplier_ceiling,
        )
    )
    return (
        history_observations,
        performance_observations,
        annualized_sharpe,
        confidence,
        performance_multiplier,
    )


def _history_confidence(
    prices: pd.Series,
    as_of_date: pd.Timestamp,
    settings: InverseVolatilitySettings,
) -> tuple[int, float]:
    history = prices.loc[prices.index < as_of_date].dropna()
    history_observations = int(
        len(history.pct_change(fill_method=None).dropna())
    )
    confidence = min(
        1.0,
        (
            history_observations / settings.confidence_full_observations
        )
        ** settings.confidence_power,
    )
    return history_observations, float(confidence)


def _strategy_performance_adjustment(
    returns: pd.Series,
    as_of_date: pd.Timestamp,
    settings: InverseVolatilitySettings,
    asset_settings: AssetRiskSettings,
) -> tuple[int, float, float]:
    """Return a clipped multiplier from prior after-cost strategy returns."""
    history = pd.to_numeric(
        returns.loc[returns.index < as_of_date], errors="coerce"
    ).dropna()
    performance_returns = history.tail(
        settings.performance_lookback_observations
    )
    performance_observations = int(len(performance_returns))
    if performance_observations < settings.minimum_performance_observations:
        raise InsufficientHistoryError(
            performance_observations,
            settings.minimum_performance_observations,
        )

    return_std = float(performance_returns.std(ddof=1))
    if not np.isfinite(return_std):
        raise ValueError("strategy return volatility is not finite")
    if return_std == 0:
        annualized_sharpe = 0.0
    else:
        annualized_sharpe = float(
            performance_returns.mean()
            / return_std
            * np.sqrt(asset_settings.annualization_periods)
        )
    performance_multiplier = float(
        np.clip(
            1.0 + annualized_sharpe / settings.performance_sharpe_scale,
            settings.performance_multiplier_floor,
            settings.performance_multiplier_ceiling,
        )
    )
    return performance_observations, annualized_sharpe, performance_multiplier


def build_weight_schedule(
    daily_prices: pd.DataFrame,
    dates: Iterable[object],
    settings: InverseVolatilitySettings,
    asset_settings: Mapping[str, AssetRiskSettings],
    strategy_returns: pd.DataFrame | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Calculate weights and long-form diagnostics for each rebalance date.

    Only prices strictly before a rebalance date are used. A date receives no
    weights unless every configured asset has sufficient history.
    """
    settings.validate()
    assets = list(asset_settings)
    if not assets:
        raise ValueError("At least one asset must be configured")
    missing_assets = set(assets) - set(daily_prices.columns)
    if missing_assets:
        raise ValueError(f"Daily prices are missing assets: {sorted(missing_assets)}")
    for asset, item in asset_settings.items():
        item.validate(asset)
    strategy_return_methods = {
        "capped_strategy_performance_adjusted_inverse_volatility",
        "capped_strategy_return_inverse_volatility",
    }
    if settings.method in strategy_return_methods:
        if strategy_returns is None:
            raise ValueError(
                "strategy_returns are required for strategy-return sizing"
            )
        missing_strategy_assets = set(assets) - set(strategy_returns.columns)
        if missing_strategy_assets:
            raise ValueError(
                "Strategy returns are missing assets: "
                f"{sorted(missing_strategy_assets)}"
            )
        strategy_returns = strategy_returns.copy()
        strategy_returns.index = pd.DatetimeIndex(
            pd.to_datetime(strategy_returns.index, utc=True)
        )
        strategy_returns = strategy_returns.sort_index()

    weight_rows: list[dict[str, float | pd.Timestamp]] = []
    diagnostic_rows: list[dict[str, object]] = []
    caps = {asset: asset_settings[asset].maximum_weight for asset in assets}

    for value in dates:
        as_of_date = _utc_date(value, "rebalance date")
        effective_volatility: dict[str, float] = {}
        confidence_adjustments: dict[str, float] = {}
        performance_multipliers: dict[str, float] = {}
        date_diagnostics: list[dict[str, object]] = []
        date_valid = True

        for asset in assets:
            try:
                if settings.method == "capped_strategy_return_inverse_volatility":
                    observations, annualized, effective = (
                        _strategy_return_volatility_estimate(
                            strategy_returns[asset],
                            as_of_date,
                            settings,
                            asset_settings[asset],
                        )
                    )
                    volatility_source = "standalone_strategy_returns_after_costs"
                else:
                    observations, annualized, effective = _volatility_estimate(
                        daily_prices[asset],
                        as_of_date,
                        settings,
                        asset_settings[asset],
                    )
                    volatility_source = "base_currency_adjusted_close_returns"
                effective_volatility[asset] = effective
                history_observations = observations
                performance_observations = observations
                annualized_sharpe = np.nan
                confidence = 1.0
                performance_multiplier = 1.0
                if (
                    settings.method
                    == "capped_confidence_adjusted_inverse_volatility"
                ):
                    (
                        history_observations,
                        performance_observations,
                        annualized_sharpe,
                        confidence,
                        performance_multiplier,
                    ) = _confidence_performance_adjustment(
                        daily_prices[asset],
                        as_of_date,
                        settings,
                        asset_settings[asset],
                    )
                elif (
                    settings.method
                    == "capped_strategy_performance_adjusted_inverse_volatility"
                ):
                    history_observations, confidence = _history_confidence(
                        daily_prices[asset],
                        as_of_date,
                        settings,
                    )
                    (
                        performance_observations,
                        annualized_sharpe,
                        performance_multiplier,
                    ) = _strategy_performance_adjustment(
                        strategy_returns[asset],
                        as_of_date,
                        settings,
                        asset_settings[asset],
                    )
                elif (
                    settings.method
                    == "capped_history_confidence_inverse_volatility"
                ):
                    history_observations, confidence = _history_confidence(
                        daily_prices[asset],
                        as_of_date,
                        settings,
                    )
                confidence_adjustments[asset] = confidence
                performance_multipliers[asset] = performance_multiplier
                date_diagnostics.append(
                    {
                        "rebalance_date_utc": as_of_date,
                        "instrument_id": asset,
                        "status": "ready",
                        "reason": "",
                        "return_observations": observations,
                        "annualized_volatility": annualized,
                        "effective_volatility": effective,
                        "volatility_source": volatility_source,
                        "history_observations": history_observations,
                        "performance_observations": performance_observations,
                        "annualized_performance_sharpe": annualized_sharpe,
                        "history_confidence": confidence,
                        "performance_multiplier": performance_multiplier,
                    }
                )
            except ValueError as error:
                date_valid = False
                date_diagnostics.append(
                    {
                        "rebalance_date_utc": as_of_date,
                        "instrument_id": asset,
                        "status": "insufficient_history",
                        "reason": str(error),
                        "return_observations": getattr(
                            error, "observation_count", 0
                        ),
                        "annualized_volatility": np.nan,
                        "effective_volatility": np.nan,
                        "volatility_source": (
                            "standalone_strategy_returns_after_costs"
                            if settings.method
                            == "capped_strategy_return_inverse_volatility"
                            else "base_currency_adjusted_close_returns"
                        ),
                        "history_observations": 0,
                        "performance_observations": 0,
                        "annualized_performance_sharpe": np.nan,
                        "history_confidence": np.nan,
                        "performance_multiplier": np.nan,
                    }
                )

        if date_valid:
            if settings.method in {
                "capped_inverse_volatility",
                "capped_strategy_return_inverse_volatility",
            }:
                allocation_scores = {
                    asset: 1.0 / effective_volatility[asset]
                    for asset in assets
                }
                weights = allocate_capped_inverse_volatility(
                    pd.Series(effective_volatility),
                    settings.target_gross_exposure,
                    caps,
                )
            elif settings.method == "capped_equal_weight":
                allocation_scores = {asset: 1.0 for asset in assets}
                weights = allocate_capped_equal_weight(
                    assets,
                    settings.target_gross_exposure,
                    caps,
                )
            elif settings.method in {
                "capped_confidence_adjusted_inverse_volatility",
                "capped_history_confidence_inverse_volatility",
                "capped_strategy_performance_adjusted_inverse_volatility",
            }:
                allocation_scores = {
                    asset: (
                        confidence_adjustments[asset]
                        * performance_multipliers[asset]
                        / effective_volatility[asset]
                    )
                    for asset in assets
                }
                weights = allocate_capped_scores(
                    pd.Series(allocation_scores, dtype="float64"),
                    settings.target_gross_exposure,
                    caps,
                )
            else:
                raise ValueError(f"Unsupported weighting method: {settings.method}")
            weight_row: dict[str, float | pd.Timestamp] = {
                "rebalance_date_utc": as_of_date,
                **{asset: float(weights.loc[asset]) for asset in assets},
            }
            weight_rows.append(weight_row)
            for row in date_diagnostics:
                asset = str(row["instrument_id"])
                row["weighting_method"] = settings.method
                row["inverse_volatility_score"] = 1.0 / float(
                    row["effective_volatility"]
                )
                row["allocation_score"] = float(allocation_scores[asset])
                row["target_weight"] = float(weights.loc[asset])
        else:
            for row in date_diagnostics:
                row["weighting_method"] = settings.method
                row["inverse_volatility_score"] = np.nan
                row["allocation_score"] = np.nan
                row["target_weight"] = np.nan
        diagnostic_rows.extend(date_diagnostics)

    if weight_rows:
        weights_output = pd.DataFrame(weight_rows).set_index(
            "rebalance_date_utc"
        )
        weights_output.index = pd.DatetimeIndex(weights_output.index)
    else:
        weights_output = pd.DataFrame(columns=assets, dtype="float64")
        weights_output.index = pd.DatetimeIndex([], tz="UTC")
        weights_output.index.name = "rebalance_date_utc"
    diagnostics_output = pd.DataFrame(diagnostic_rows)
    return weights_output, diagnostics_output

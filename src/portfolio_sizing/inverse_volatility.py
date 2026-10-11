"""Rolling inverse-volatility portfolio weights with concentration caps."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Mapping

import numpy as np
import pandas as pd


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
    """Portfolio-wide inverse-volatility policy."""

    lookback_observations: int = 60
    minimum_observations: int = 40
    target_gross_exposure: float = 0.95
    volatility_floor: float = 0.05
    rebalance_frequency: str = "weekly"

    def validate(self) -> None:
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
        if self.rebalance_frequency not in {"daily", "weekly"}:
            raise ValueError("rebalance_frequency must be daily or weekly")


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
    if frequency == "daily":
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


def allocate_capped_inverse_volatility(
    volatilities: pd.Series,
    target_gross_exposure: float,
    maximum_weights: Mapping[str, float],
) -> pd.Series:
    """Allocate inverse-volatility weights without breaching asset caps.

    If the configured caps cannot absorb the target exposure, the remainder is
    deliberately left as cash rather than violating a cap.
    """
    if not 0 < target_gross_exposure <= 1:
        raise ValueError("target_gross_exposure must be in (0, 1]")
    values = pd.to_numeric(volatilities, errors="coerce")
    if values.empty or values.isna().any() or (~np.isfinite(values)).any():
        raise ValueError("Volatilities must be finite")
    if values.le(0).any():
        raise ValueError("Volatilities must be positive")

    assets = list(values.index.astype(str))
    missing_caps = set(assets) - set(maximum_weights)
    if missing_caps:
        raise ValueError(f"Missing maximum weights for: {sorted(missing_caps)}")
    caps = {asset: float(maximum_weights[asset]) for asset in assets}
    if any(cap <= 0 or cap > 1 for cap in caps.values()):
        raise ValueError("Maximum weights must be in (0, 1]")

    scores = {asset: 1.0 / float(values.loc[asset]) for asset in assets}
    weights = {asset: 0.0 for asset in assets}
    remaining = set(assets)
    remaining_budget = float(target_gross_exposure)

    while remaining and remaining_budget > 1e-12:
        score_total = sum(scores[asset] for asset in remaining)
        if score_total <= 0:
            break
        proposed = {
            asset: remaining_budget * scores[asset] / score_total
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


def build_weight_schedule(
    daily_prices: pd.DataFrame,
    dates: Iterable[object],
    settings: InverseVolatilitySettings,
    asset_settings: Mapping[str, AssetRiskSettings],
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

    weight_rows: list[dict[str, float | pd.Timestamp]] = []
    diagnostic_rows: list[dict[str, object]] = []
    caps = {asset: asset_settings[asset].maximum_weight for asset in assets}

    for value in dates:
        as_of_date = _utc_date(value, "rebalance date")
        effective_volatility: dict[str, float] = {}
        date_diagnostics: list[dict[str, object]] = []
        date_valid = True

        for asset in assets:
            try:
                observations, annualized, effective = _volatility_estimate(
                    daily_prices[asset],
                    as_of_date,
                    settings,
                    asset_settings[asset],
                )
                effective_volatility[asset] = effective
                date_diagnostics.append(
                    {
                        "rebalance_date_utc": as_of_date,
                        "instrument_id": asset,
                        "status": "ready",
                        "reason": "",
                        "return_observations": observations,
                        "annualized_volatility": annualized,
                        "effective_volatility": effective,
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
                    }
                )

        if date_valid:
            weights = allocate_capped_inverse_volatility(
                pd.Series(effective_volatility),
                settings.target_gross_exposure,
                caps,
            )
            weight_row: dict[str, float | pd.Timestamp] = {
                "rebalance_date_utc": as_of_date,
                **{asset: float(weights.loc[asset]) for asset in assets},
            }
            weight_rows.append(weight_row)
            for row in date_diagnostics:
                asset = str(row["instrument_id"])
                row["inverse_volatility_score"] = 1.0 / float(
                    row["effective_volatility"]
                )
                row["target_weight"] = float(weights.loc[asset])
        else:
            for row in date_diagnostics:
                row["inverse_volatility_score"] = np.nan
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

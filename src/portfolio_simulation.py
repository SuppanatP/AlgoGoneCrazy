"""Shared-USD portfolio replay for rule-based and external strategy signals."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import pandas as pd

from market_data.costs import CostAssumptions
from trading_strategy import (
    MeanReversionParameters,
    MomentumParameters,
    generate_mean_reversion_signals,
    generate_momentum_signals,
)
from trading_strategy.common import prepare_strategy_bars

from .backtest import BacktestResult
from .metrics import compute_metrics
from .portfolio_sizing import allocate_capped_scores, configured_maximum_weight
from .strategy_backtest import delayed_execution_targets
from .tracker import Portfolio


PROJECT_COST_ASSET_IDS = {
    "spy_usd": "SP500_ETF",
    "sti_es3_sgd": "STI_ETF",
    "btc_usdt": "BTC",
}


@dataclass
class PortfolioSimulationArtifacts:
    """Outputs from one chronological portfolio replay."""

    result: BacktestResult
    metrics: pd.Series
    decisions: pd.DataFrame
    signals: pd.DataFrame
    asset_summary: pd.DataFrame
    unavailable_signal_assets: list[str]


class BaseCurrencyProjectCostModel:
    """Apply native broker fees and return every cost in portfolio currency."""

    def __init__(
        self,
        assumptions: CostAssumptions,
        quote_currencies: Mapping[str, str],
        *,
        base_currency: str = "USD",
        slippage_bps: float = 0.0,
    ) -> None:
        if base_currency.strip().upper() != "USD":
            raise ValueError("The current project cost model reports in USD")
        if slippage_bps < 0:
            raise ValueError("slippage_bps must not be negative")
        self.assumptions = assumptions
        self.quote_currencies = {
            asset: currency.strip().upper()
            for asset, currency in quote_currencies.items()
        }
        self.slippage_bps = float(slippage_bps)
        self._rates_to_base: dict[str, float] = {}

    def set_rates(self, rates_to_base: Mapping[str, float]) -> None:
        rates = {asset: float(rate) for asset, rate in rates_to_base.items()}
        if any(not np.isfinite(rate) or rate <= 0 for rate in rates.values()):
            raise ValueError("Execution FX rates must be finite and positive")
        self._rates_to_base = rates

    def apply(
        self,
        asset: str,
        qty: float,
        ref_price: float,
    ) -> tuple[float, float, float]:
        if asset not in PROJECT_COST_ASSET_IDS:
            raise ValueError(f"No project cost mapping is configured for {asset}")
        if asset not in self._rates_to_base:
            raise ValueError(f"No execution FX rate is available for {asset}")

        direction = 1.0 if qty > 0 else -1.0
        execution_price = ref_price * (
            1.0 + direction * self.slippage_bps / 10_000.0
        )
        slippage_cost = abs(qty) * abs(execution_price - ref_price)
        rate_to_base = self._rates_to_base[asset]
        native_notional = abs(qty) * execution_price / rate_to_base
        cost_asset_id = PROJECT_COST_ASSET_IDS[asset]
        native_fee = self.assumptions.assets[cost_asset_id].commission(
            native_notional,
            shares=abs(qty),
        )
        fee_base = native_fee * rate_to_base
        if cost_asset_id in self.assumptions.fx.applies_to:
            fee_base += self.assumptions.fx.fee(
                cost_asset_id,
                abs(qty) * execution_price,
            )
        return execution_price, float(fee_base), float(slippage_cost)


def _portfolio_target_weights(
    base_risk_weights: Mapping[str, float],
    latest_signal: Mapping[str, float],
    maximum_weights: Mapping[str, float],
    target_gross_exposure: float,
    inactive_allocation: str,
    *,
    minimum_active_strategies_for_scaling: int,
    maximum_active_scaling_multiplier: float,
    scaling_eligibility_minimum_absolute_target: Mapping[str, float],
) -> tuple[dict[str, float], dict[str, float], int, int]:
    """Map sleeve-relative strategy targets into signed portfolio weights.

    The bounded policy can deploy part of the otherwise idle cash only when
    multiple independent strategies are active. It never changes a target's
    sign and never scales its absolute size by more than the configured
    multiplier or beyond the asset cap.
    """
    assets = list(base_risk_weights)
    raw_targets = {
        asset: float(base_risk_weights[asset]) * float(latest_signal[asset])
        for asset in assets
    }
    active_assets = [
        asset for asset in assets if abs(raw_targets[asset]) > 1e-12
    ]
    scaling_eligible_assets = [
        asset
        for asset in active_assets
        if abs(float(latest_signal[asset]))
        >= float(scaling_eligibility_minimum_absolute_target.get(asset, 0.0))
    ]

    if inactive_allocation == "hold_as_cash":
        return (
            raw_targets.copy(),
            raw_targets,
            len(active_assets),
            len(scaling_eligible_assets),
        )

    if inactive_allocation == "redistribute_to_active_assets_subject_to_common_cap":
        positive_scores = pd.Series(
            {
                asset: raw_targets[asset]
                for asset in assets
                if raw_targets[asset] > 0
            },
            dtype="float64",
        )
        if positive_scores.empty:
            return (
                {asset: 0.0 for asset in assets},
                raw_targets,
                len(active_assets),
                len(scaling_eligible_assets),
            )
        allocated = allocate_capped_scores(
            positive_scores,
            target_gross_exposure,
            {asset: maximum_weights[asset] for asset in positive_scores.index},
        )
        return (
            {asset: float(allocated.get(asset, 0.0)) for asset in assets},
            raw_targets,
            len(active_assets),
            len(scaling_eligible_assets),
        )

    if (
        len(scaling_eligible_assets)
        < minimum_active_strategies_for_scaling
    ):
        return (
            raw_targets.copy(),
            raw_targets,
            len(active_assets),
            len(scaling_eligible_assets),
        )

    active_scores = pd.Series(
        {asset: abs(raw_targets[asset]) for asset in active_assets},
        dtype="float64",
    )
    bounded_caps = {
        asset: min(
            float(maximum_weights[asset]),
            abs(raw_targets[asset]) * maximum_active_scaling_multiplier,
        )
        for asset in active_assets
    }
    deployable_gross = min(
        float(target_gross_exposure),
        float(sum(bounded_caps.values())),
    )
    current_gross = float(active_scores.sum())
    if deployable_gross <= current_gross + 1e-12:
        return (
            raw_targets.copy(),
            raw_targets,
            len(active_assets),
            len(scaling_eligible_assets),
        )

    allocated_magnitudes = allocate_capped_scores(
        active_scores,
        deployable_gross,
        bounded_caps,
    )
    targets = {asset: 0.0 for asset in assets}
    for asset in active_assets:
        targets[asset] = float(
            np.copysign(allocated_magnitudes[asset], raw_targets[asset])
        )
    return (
        targets,
        raw_targets,
        len(active_assets),
        len(scaling_eligible_assets),
    )


def _rule_based_signal(
    bars: pd.DataFrame,
    strategy: str,
    parameters: Mapping[str, Any],
    decision_interval_minutes: int = 30,
) -> pd.DataFrame:
    if decision_interval_minutes < 30 or decision_interval_minutes % 30:
        raise ValueError(
            "decision_interval_minutes must be a positive multiple of 30"
        )
    prepared = prepare_strategy_bars(bars)
    if decision_interval_minutes == 30:
        decision_bars = prepared
    else:
        factor = decision_interval_minutes // 30
        decision_bars = prepared.copy()
        timestamp_gap = decision_bars.groupby(
            "instrument_id", sort=False
        )["bar_open_time_utc"].diff()
        new_segment = timestamp_gap.ne(pd.Timedelta(minutes=30))
        if "session_date_local" in decision_bars.columns:
            new_segment |= decision_bars.groupby(
                "instrument_id", sort=False
            )["session_date_local"].transform(lambda values: values.ne(values.shift()))
        decision_bars["_segment"] = new_segment.groupby(
            decision_bars["instrument_id"], sort=False
        ).cumsum()
        decision_bars["_block"] = decision_bars.groupby(
            ["instrument_id", "_segment"], sort=False
        ).cumcount() // factor
        decision_bars = (
            decision_bars.groupby(
                ["instrument_id", "_segment", "_block"], sort=False
            )
            .tail(1)
            .drop(columns=["_segment", "_block"])
            .reset_index(drop=True)
        )

    if strategy == "momentum":
        signals = generate_momentum_signals(
            decision_bars,
            MomentumParameters(**dict(parameters)),
        )
    elif strategy == "mean_reversion":
        signals = generate_mean_reversion_signals(
            decision_bars,
            MeanReversionParameters(**dict(parameters)),
        )
    else:
        raise ValueError(f"Unsupported rule-based strategy: {strategy!r}")
    if decision_interval_minutes == 30:
        signals["execution_target_position"] = delayed_execution_targets(signals)
        signals["decision_interval_minutes"] = decision_interval_minutes
        return signals

    decision_indexed = signals.set_index("bar_open_time_utc")
    output = prepared.copy()
    base_index = pd.DatetimeIndex(output["bar_open_time_utc"])
    output["target_position"] = (
        decision_indexed["target_position"]
        .reindex(base_index, method="ffill")
        .fillna(0)
        .astype("int8")
        .to_numpy()
    )
    output["strategy_name"] = strategy
    output["signal"] = "HOLD"
    output["signal_reason"] = "between_decision_times"
    decision_locations = output["bar_open_time_utc"].isin(decision_indexed.index)
    for column in ("signal", "signal_reason"):
        output.loc[decision_locations, column] = (
            decision_indexed[column]
            .reindex(output.loc[decision_locations, "bar_open_time_utc"])
            .to_numpy()
        )
    output["execution_target_position"] = delayed_execution_targets(output)
    output["decision_interval_minutes"] = decision_interval_minutes
    return output


def _external_signal(
    bars: pd.DataFrame,
    path: Path,
    instrument_id: str,
    decision_interval_minutes: int,
    *,
    target_column: str = "target_position",
    execution_timing: str = "decision_after_bar_close_execute_next_open",
    allow_short: bool = False,
) -> pd.DataFrame:
    if decision_interval_minutes < 30 or decision_interval_minutes % 30:
        raise ValueError(
            "decision_interval_minutes must be a positive multiple of 30"
        )
    external = pd.read_csv(path, low_memory=False)
    required = {"bar_open_time_utc", target_column}
    missing = required - set(external.columns)
    if missing:
        raise ValueError(
            f"External signal file {path} is missing columns: {sorted(missing)}"
        )
    if "instrument_id" in external.columns:
        external = external.loc[external["instrument_id"].eq(instrument_id)].copy()
    if external.empty:
        raise ValueError(
            f"External signal file {path} contains no rows for {instrument_id}"
        )
    external["bar_open_time_utc"] = pd.to_datetime(
        external["bar_open_time_utc"], errors="coerce", utc=True
    )
    external[target_column] = pd.to_numeric(
        external[target_column], errors="coerce"
    )
    lower_bound = -1.0 if allow_short else 0.0
    invalid = (
        external["bar_open_time_utc"].isna()
        | external[target_column].isna()
        | external[target_column].lt(lower_bound)
        | external[target_column].gt(1.0)
    )
    if invalid.any():
        raise ValueError(
            f"External signal file {path} contains {int(invalid.sum())} invalid rows"
        )
    if external.duplicated(subset=["bar_open_time_utc"]).any():
        raise ValueError(f"External signal file {path} contains duplicate timestamps")

    prepared = prepare_strategy_bars(bars)
    prepared_times = pd.DatetimeIndex(prepared["bar_open_time_utc"])
    latest_prepared_time = prepared_times.max()
    external = external.loc[
        external["bar_open_time_utc"].le(latest_prepared_time)
    ].copy()
    unknown_times = set(
        external.loc[
            external["bar_open_time_utc"].ge(prepared_times.min()),
            "bar_open_time_utc",
        ]
    ) - set(prepared_times)
    if unknown_times:
        first = min(unknown_times)
        raise ValueError(
            f"External signal timestamp {first.isoformat()} is not a canonical "
            f"{instrument_id} bar open"
        )
    decision = (
        external.set_index("bar_open_time_utc")[target_column]
        .sort_index()
        .reindex(prepared_times, method="ffill")
        .fillna(0.0)
        .astype("float64")
    )
    prepared["target_position"] = decision.to_numpy()
    if execution_timing == "at_bar_open":
        prepared["execution_target_position"] = prepared["target_position"]
    elif execution_timing == "decision_after_bar_close_execute_next_open":
        prepared["execution_target_position"] = delayed_execution_targets(prepared)
    else:
        raise ValueError(
            "External execution_timing must be at_bar_open or "
            "decision_after_bar_close_execute_next_open"
        )
    prepared["strategy_name"] = "external"
    external_indexed = external.set_index("bar_open_time_utc").sort_index()
    prepared_index = prepared_times
    if "entry_signal" in external_indexed.columns:
        prepared["signal"] = (
            pd.to_numeric(
                external_indexed["entry_signal"], errors="coerce"
            )
            .reindex(prepared_index)
            .fillna(0.0)
            .to_numpy()
        )
    else:
        prepared["signal"] = "HOLD"
    if "signal_reason" in external_indexed.columns:
        prepared["signal_reason"] = (
            external_indexed["signal_reason"]
            .reindex(prepared_index)
            .fillna("external_target_carried_forward")
            .to_numpy()
        )
    else:
        prepared["signal_reason"] = "external_target"
    prepared["decision_interval_minutes"] = decision_interval_minutes
    prepared["execution_timing"] = execution_timing
    return prepared


def build_execution_signals(
    canonical: pd.DataFrame,
    strategy_config: Mapping[str, Any],
    assets: list[str],
    end: pd.Timestamp,
    *,
    project_root: Path,
    allow_missing_external: bool,
) -> tuple[dict[str, pd.Series], pd.DataFrame, list[str]]:
    """Build next-bar execution positions for every configured portfolio asset."""
    configured = strategy_config.get("assets")
    if not isinstance(configured, Mapping):
        raise ValueError("Strategy configuration must contain an assets object")

    series: dict[str, pd.Series] = {}
    output_frames: list[pd.DataFrame] = []
    unavailable: list[str] = []
    for instrument_id in assets:
        asset_bars = canonical.loc[
            canonical["instrument_id"].eq(instrument_id)
            & canonical["bar_open_time_utc"].lt(end)
        ].copy()
        if asset_bars.empty:
            raise ValueError(f"Canonical data does not contain {instrument_id}")
        asset_config = configured.get(instrument_id)
        if not isinstance(asset_config, Mapping):
            raise ValueError(f"No signal configuration exists for {instrument_id}")
        source = str(asset_config.get("signal_source", "")).strip()

        if source == "rule_based":
            parameters = asset_config.get("parameters")
            if not isinstance(parameters, Mapping):
                raise ValueError(f"{instrument_id} strategy parameters are missing")
            signal_frame = _rule_based_signal(
                asset_bars,
                str(asset_config.get("strategy", "")),
                parameters,
                int(asset_config.get("decision_interval_minutes", 30)),
            )
            signal_frame["execution_timing"] = (
                "decision_after_bar_close_execute_next_open"
            )
            status = "ready"
        elif source == "external_csv":
            decision_interval_minutes = int(
                asset_config.get("decision_interval_minutes", 30)
            )
            raw_path = Path(str(asset_config.get("path", "")))
            path = raw_path if raw_path.is_absolute() else project_root / raw_path
            if path.is_file():
                signal_frame = _external_signal(
                    asset_bars,
                    path,
                    instrument_id,
                    decision_interval_minutes,
                    target_column=str(
                        asset_config.get("target_column", "target_position")
                    ),
                    execution_timing=str(
                        asset_config.get(
                            "execution_timing",
                            "decision_after_bar_close_execute_next_open",
                        )
                    ),
                    allow_short=bool(asset_config.get("allow_short", False)),
                )
                status = "ready"
            elif allow_missing_external:
                if (
                    decision_interval_minutes < 30
                    or decision_interval_minutes % 30
                ):
                    raise ValueError(
                        "decision_interval_minutes must be a positive multiple of 30"
                    )
                signal_frame = prepare_strategy_bars(asset_bars)
                signal_frame["execution_target_position"] = 0
                signal_frame["strategy_name"] = "external"
                signal_frame["signal"] = "HOLD"
                signal_frame["signal_reason"] = "external_signal_unavailable"
                signal_frame["decision_interval_minutes"] = (
                    decision_interval_minutes
                )
                signal_frame["execution_timing"] = str(
                    asset_config.get(
                        "execution_timing",
                        "decision_after_bar_close_execute_next_open",
                    )
                )
                unavailable.append(instrument_id)
                status = "awaiting_external_signal"
            else:
                raise FileNotFoundError(
                    f"External signals for {instrument_id} were not found at {path}"
                )
        else:
            raise ValueError(
                f"Unsupported signal_source {source!r} for {instrument_id}"
            )

        signal_series = signal_frame.set_index("bar_open_time_utc")[
            "execution_target_position"
        ].astype("float64")
        signal_series.name = instrument_id
        series[instrument_id] = signal_series
        output = signal_frame[
            [
                "bar_open_time_utc",
                "execution_target_position",
                "strategy_name",
                "signal",
                "signal_reason",
                "decision_interval_minutes",
                "execution_timing",
            ]
        ].copy()
        output.insert(0, "instrument_id", instrument_id)
        output["signal_source"] = source
        output["source_status"] = status
        output_frames.append(output)

    return series, pd.concat(output_frames, ignore_index=True), unavailable


def _rates_for_bars(
    index: pd.DatetimeIndex,
    instrument_id: str,
    quote_currency: str,
    static_rates: Mapping[str, float],
    fx_rates: pd.DataFrame,
) -> pd.Series:
    currency = quote_currency.strip().upper()
    if currency in static_rates:
        return pd.Series(
            float(static_rates[currency]),
            index=index,
            dtype="float64",
            name="rate_to_base",
        )
    rows = fx_rates.loc[fx_rates["quote_currency"].eq(currency)].copy()
    if rows.empty:
        raise ValueError(
            f"No point-in-time FX rates are available for {instrument_id} ({currency})"
        )
    source = rows.set_index("rate_time_utc")["rate_to_base"].sort_index()
    aligned = source.reindex(index, method="ffill")
    if aligned.isna().any():
        first = aligned.index[aligned.isna()][0]
        raise ValueError(
            f"No {currency} FX rate is known by {first.isoformat()} for {instrument_id}"
        )
    return aligned.astype("float64").rename("rate_to_base")


def _fit_target_to_cash(
    portfolio: Portfolio,
    asset: str,
    target_qty: float,
    price: float,
    step: float,
) -> float:
    current = portfolio.position(asset)
    candidate = target_qty
    for _ in range(10):
        order_qty = candidate - current
        if order_qty <= 0:
            return candidate
        execution_price, fee, _ = portfolio.costs.apply(asset, order_qty, price)
        required = order_qty * execution_price + fee
        if required <= portfolio.cash + 1e-9:
            return candidate
        affordable = max((portfolio.cash - fee) / execution_price, 0.0)
        candidate = current + affordable
        candidate = np.floor(candidate / step) * step
    return max(candidate, current)


def _asset_summary(
    assets: list[str],
    portfolio: Portfolio,
) -> pd.DataFrame:
    fills = portfolio.fills
    trades = portfolio.trades
    rows: list[dict[str, Any]] = []
    for asset in assets:
        asset_fills = fills.loc[fills["asset"].eq(asset)] if not fills.empty else fills
        asset_trades = (
            trades.loc[trades["asset"].eq(asset)] if not trades.empty else trades
        )
        ending_qty = float(portfolio.position(asset))
        ending_price = float(portfolio.last_price.get(asset, np.nan))
        ending_value = ending_qty * ending_price if np.isfinite(ending_price) else 0.0
        execution_cash_flow = (
            float((-(asset_fills["qty"] * asset_fills["price"]) - asset_fills["fee"]).sum())
            if not asset_fills.empty
            else 0.0
        )
        rows.append(
            {
                "instrument_id": asset,
                "orders": int(len(asset_fills)),
                "completed_round_trips": int(len(asset_trades)),
                "ending_quantity": ending_qty,
                "ending_price_usd": ending_price,
                "ending_market_value_usd": ending_value,
                "transaction_cost_usd": (
                    float(asset_fills["total_cost"].sum())
                    if not asset_fills.empty
                    else 0.0
                ),
                "net_pnl_contribution_usd": execution_cash_flow + ending_value,
            }
        )
    return pd.DataFrame(rows)


def daily_strategy_returns_after_costs(
    canonical: pd.DataFrame,
    execution_signals: Mapping[str, pd.Series],
    fx_rates: pd.DataFrame,
    sizing_config: Mapping[str, Any],
    cost_assumptions: CostAssumptions,
    end: pd.Timestamp,
    *,
    slippage_bps: float = 0.0,
) -> pd.DataFrame:
    """Build daily unit-strategy returns using only executable targets.

    Positions change at each bar's raw open. Adjusted prices supply the return
    ratios, while commissions are evaluated at a configurable reference
    notional so minimum and per-share fees use the project's broker model.
    """
    configured_assets = sizing_config["assets"]
    assets = list(configured_assets)
    reference_notional = float(
        sizing_config.get("strategy_performance_reference_notional_base", 100_000.0)
    )
    if reference_notional <= 0 or not np.isfinite(reference_notional):
        raise ValueError(
            "strategy_performance_reference_notional_base must be positive"
        )
    static_rates = {
        str(currency).upper(): float(rate)
        for currency, rate in sizing_config.get("static_rates_to_base", {}).items()
    }
    static_rates.setdefault("USD", 1.0)
    quote_currencies = {
        asset: str(configured_assets[asset]["quote_currency"]).upper()
        for asset in assets
    }
    cost_model = BaseCurrencyProjectCostModel(
        cost_assumptions,
        quote_currencies,
        base_currency=str(sizing_config.get("base_currency", "USD")),
        slippage_bps=slippage_bps,
    )
    outputs: list[pd.Series] = []

    for asset in assets:
        required = [
            "bar_open_time_utc",
            "open",
            "close",
            "adjusted_close",
        ]
        if "session_date_local" in canonical.columns:
            required.append("session_date_local")
        bars = canonical.loc[
            canonical["instrument_id"].eq(asset)
            & canonical["bar_open_time_utc"].lt(end),
            required,
        ].copy()
        if bars.empty:
            raise ValueError(f"No strategy-return history exists for {asset}")
        bars = bars.sort_values("bar_open_time_utc").set_index("bar_open_time_utc")
        rates = _rates_for_bars(
            pd.DatetimeIndex(bars.index),
            asset,
            quote_currencies[asset],
            static_rates,
            fx_rates,
        )
        raw_open = pd.to_numeric(bars["open"], errors="coerce")
        raw_close = pd.to_numeric(bars["close"], errors="coerce")
        adjusted_close = pd.to_numeric(bars["adjusted_close"], errors="coerce")
        invalid = (
            raw_open.isna()
            | raw_close.isna()
            | adjusted_close.isna()
            | raw_open.le(0)
            | raw_close.le(0)
            | adjusted_close.le(0)
        )
        if invalid.any():
            raise ValueError(
                f"Strategy-return history contains {int(invalid.sum())} invalid "
                f"{asset} bars"
            )
        adjustment_factor = adjusted_close / raw_close
        adjusted_open = raw_open * adjustment_factor
        targets = execution_signals[asset].reindex(bars.index)
        if targets.isna().any():
            raise ValueError(f"Strategy returns do not align with every {asset} bar")
        targets = targets.astype("float64")
        prior_targets = targets.shift(1).fillna(0.0)
        prior_close = adjusted_close.shift(1)
        overnight = (adjusted_open / prior_close - 1.0).fillna(0.0)
        intrabar = adjusted_close / adjusted_open - 1.0
        gross_factor = (1.0 + prior_targets * overnight) * (
            1.0 + targets * intrabar
        )
        bar_returns = gross_factor - 1.0

        turnover = (targets - prior_targets).abs()
        for timestamp in turnover.index[turnover.gt(0)]:
            rate = float(rates.loc[timestamp])
            open_base = float(raw_open.loc[timestamp] * rate)
            quantity = float(
                turnover.loc[timestamp] * reference_notional / open_base
            )
            cost_model.set_rates({asset: rate})
            _, fee, slippage = cost_model.apply(asset, quantity, open_base)
            bar_returns.loc[timestamp] -= (fee + slippage) / reference_notional

        if "session_date_local" in bars.columns:
            session_date = pd.to_datetime(
                bars["session_date_local"], errors="coerce", utc=True
            )
        else:
            session_date = pd.Series(bars.index.normalize(), index=bars.index)
        if session_date.isna().any():
            raise ValueError(f"Invalid session dates in {asset} strategy history")
        daily = (1.0 + bar_returns).groupby(session_date).prod() - 1.0
        daily.index = pd.DatetimeIndex(daily.index).normalize()
        daily.name = asset
        outputs.append(daily.astype("float64"))

    result = pd.concat(outputs, axis=1).sort_index()
    result.index.name = "session_date"
    return result


def simulate_portfolio(
    canonical: pd.DataFrame,
    execution_signals: Mapping[str, pd.Series],
    signal_output: pd.DataFrame,
    unavailable_signal_assets: list[str],
    risk_weights: pd.DataFrame,
    fx_rates: pd.DataFrame,
    sizing_config: Mapping[str, Any],
    cost_assumptions: CostAssumptions,
    start: pd.Timestamp,
    end: pd.Timestamp,
    *,
    slippage_bps: float = 0.0,
    risk_free_annual_rates: pd.Series | None = None,
    risk_free_source: str | None = None,
    risk_metric_periods_per_year: int = 365,
) -> PortfolioSimulationArtifacts:
    """Replay asset targets and configured sleeves through one USD cash ledger."""
    configured_assets = sizing_config["assets"]
    assets = list(configured_assets)
    if risk_weights.empty:
        raise ValueError("No portfolio risk weights are available")
    effective_start = max(start, risk_weights.index.min())
    if effective_start >= end:
        raise ValueError("No ready risk weights overlap the requested period")

    static_rates = {
        str(currency).upper(): float(rate)
        for currency, rate in sizing_config.get("static_rates_to_base", {}).items()
    }
    static_rates.setdefault("USD", 1.0)
    quote_currencies = {
        asset: str(configured_assets[asset]["quote_currency"]).upper()
        for asset in assets
    }
    base_bars: dict[str, pd.DataFrame] = {}
    for asset in assets:
        bars = canonical.loc[
            canonical["instrument_id"].eq(asset)
            & canonical["bar_open_time_utc"].ge(effective_start)
            & canonical["bar_open_time_utc"].lt(end),
            ["bar_open_time_utc", "open", "close"],
        ].copy()
        if bars.empty:
            raise ValueError(f"No {asset} bars overlap the effective simulation period")
        bars = bars.set_index("bar_open_time_utc").sort_index()
        rates = _rates_for_bars(
            pd.DatetimeIndex(bars.index),
            asset,
            quote_currencies[asset],
            static_rates,
            fx_rates,
        )
        bars["rate_to_usd"] = rates
        bars["open_usd"] = bars["open"].astype(float) * rates
        bars["close_usd"] = bars["close"].astype(float) * rates
        signal = execution_signals[asset].reindex(bars.index)
        if signal.isna().any():
            raise ValueError(f"Execution signals do not align with every {asset} bar")
        bars["execution_target_position"] = signal.astype(float)
        base_bars[asset] = bars

    timeline = pd.DatetimeIndex([])
    for bars in base_bars.values():
        timeline = timeline.union(bars.index)
    timeline = timeline.sort_values()
    if timeline.empty:
        raise ValueError("The portfolio simulation timeline is empty")

    cost_model = BaseCurrencyProjectCostModel(
        cost_assumptions,
        quote_currencies,
        base_currency=str(sizing_config.get("base_currency", "USD")),
        slippage_bps=slippage_bps,
    )
    initial_cash = float(sizing_config.get("initial_cash_base", 100_000.0))
    portfolio = Portfolio(initial_cash, cost_model=cost_model)
    steps = {
        asset: float(configured_assets[asset]["quantity_step"])
        for asset in assets
    }
    maximum_weights = {
        asset: configured_maximum_weight(sizing_config, asset)
        for asset in assets
    }
    target_gross_exposure = float(
        sizing_config.get("target_gross_exposure", 0.95)
    )
    inactive_allocation = str(
        sizing_config.get(
            "inactive_allocation",
            "redistribute_to_active_assets_subject_to_common_cap",
        )
    )
    supported_inactive_allocation = {
        "bounded_active_capital_scaling",
        "redistribute_to_active_assets_subject_to_common_cap",
        "hold_as_cash",
    }
    if inactive_allocation not in supported_inactive_allocation:
        raise ValueError(
            "inactive_allocation must be one of "
            f"{sorted(supported_inactive_allocation)}"
        )
    minimum_active_strategies_for_scaling = int(
        sizing_config.get("minimum_active_strategies_for_scaling", 2)
    )
    if minimum_active_strategies_for_scaling < 1:
        raise ValueError("minimum_active_strategies_for_scaling must be positive")
    maximum_active_scaling_multiplier = float(
        sizing_config.get("maximum_active_scaling_multiplier", 2.0)
    )
    if maximum_active_scaling_multiplier < 1:
        raise ValueError("maximum_active_scaling_multiplier must be at least 1")
    configured_scaling_thresholds = sizing_config.get(
        "scaling_eligibility_minimum_absolute_target", {}
    )
    if not isinstance(configured_scaling_thresholds, Mapping):
        raise ValueError(
            "scaling_eligibility_minimum_absolute_target must be an object"
        )
    unknown_scaling_threshold_assets = (
        set(configured_scaling_thresholds) - set(assets)
    )
    if unknown_scaling_threshold_assets:
        raise ValueError(
            "Unknown scaling eligibility assets: "
            f"{sorted(unknown_scaling_threshold_assets)}"
        )
    scaling_eligibility_minimum_absolute_target = {
        asset: float(configured_scaling_thresholds.get(asset, 0.0))
        for asset in assets
    }
    if any(
        not 0.0 <= threshold <= 1.0
        for threshold in scaling_eligibility_minimum_absolute_target.values()
    ):
        raise ValueError(
            "Scaling eligibility target thresholds must be between 0 and 1"
        )
    target_weight_change_threshold = float(
        sizing_config.get("target_weight_change_threshold", 0.0)
    )
    if not 0 <= target_weight_change_threshold <= 1:
        raise ValueError("target_weight_change_threshold must be in [0, 1]")
    resize_existing_positions = bool(
        sizing_config.get("resize_existing_positions_on_rebalance", True)
    )
    last_signal: dict[str, float | None] = {asset: None for asset in assets}
    last_target_weight: dict[str, float | None] = {
        asset: None for asset in assets
    }
    last_rebalance: dict[str, pd.Timestamp | None] = {asset: None for asset in assets}
    first_timestamp = pd.Timestamp(timeline[0])
    latest_signal: dict[str, float] = {}
    for asset in assets:
        history = execution_signals[asset].loc[
            execution_signals[asset].index <= first_timestamp
        ]
        latest_signal[asset] = float(history.iloc[-1]) if not history.empty else 0.0
    decisions: list[dict[str, Any]] = []

    for timestamp in timeline:
        present = [asset for asset in assets if timestamp in base_bars[asset].index]
        for asset in present:
            latest_signal[asset] = float(
                base_bars[asset].at[timestamp, "execution_target_position"]
            )
        opens = {
            asset: float(base_bars[asset].at[timestamp, "open_usd"])
            for asset in present
        }
        closes = {
            asset: float(base_bars[asset].at[timestamp, "close_usd"])
            for asset in present
        }
        rates = {
            asset: float(base_bars[asset].at[timestamp, "rate_to_usd"])
            for asset in present
        }
        cost_model.set_rates(rates)
        eligible_weights = risk_weights.loc[risk_weights.index <= timestamp]
        if eligible_weights.empty:
            continue
        rebalance_time = pd.Timestamp(eligible_weights.index[-1])
        base_risk_weights = {
            asset: float(risk_weights.at[rebalance_time, asset])
            for asset in assets
        }
        (
            portfolio_target_weights,
            raw_portfolio_target_weights,
            active_strategy_count,
            scaling_eligible_strategy_count,
        ) = _portfolio_target_weights(
            base_risk_weights,
            latest_signal,
            maximum_weights,
            target_gross_exposure,
            inactive_allocation,
            minimum_active_strategies_for_scaling=(
                minimum_active_strategies_for_scaling
            ),
            maximum_active_scaling_multiplier=(
                maximum_active_scaling_multiplier
            ),
            scaling_eligibility_minimum_absolute_target=(
                scaling_eligibility_minimum_absolute_target
            ),
        )
        equity_at_open = float(portfolio.equity(opens))
        planned: list[dict[str, Any]] = []

        for asset in present:
            signal = latest_signal[asset]
            risk_weight = base_risk_weights[asset]
            raw_target_weight = raw_portfolio_target_weights[asset]
            target_weight = portfolio_target_weights[asset]
            scaling_eligibility_threshold = (
                scaling_eligibility_minimum_absolute_target[asset]
            )
            scaling_eligible = (
                abs(signal) >= scaling_eligibility_threshold
                and abs(raw_target_weight) > 1e-12
            )
            target_scaling_multiplier = (
                abs(target_weight / raw_target_weight)
                if abs(raw_target_weight) > 1e-12
                else 0.0
            )
            rebalance_changed = last_rebalance[asset] != rebalance_time
            signal_changed = last_signal[asset] != signal
            target_changed = (
                last_target_weight[asset] is None
                or abs(float(last_target_weight[asset]) - target_weight)
                >= target_weight_change_threshold
            )
            before = float(portfolio.position(asset))
            desired = before
            should_resize = (
                signal_changed
                or (
                    target_changed
                    and (resize_existing_positions or before == 0.0)
                )
            )
            if should_resize:
                desired = target_weight * equity_at_open / opens[asset]
                desired = (
                    np.sign(desired)
                    * np.floor(abs(desired) / steps[asset])
                    * steps[asset]
                )
            planned.append(
                {
                    "timestamp": timestamp,
                    "instrument_id": asset,
                    "signal": signal,
                    "risk_weight": risk_weight,
                    "raw_target_weight": raw_target_weight,
                    "target_weight": target_weight,
                    "target_scaling_multiplier": target_scaling_multiplier,
                    "active_strategy_count": active_strategy_count,
                    "scaling_eligible": scaling_eligible,
                    "scaling_eligibility_threshold": (
                        scaling_eligibility_threshold
                    ),
                    "scaling_eligible_strategy_count": (
                        scaling_eligible_strategy_count
                    ),
                    "capital_scaling_policy": inactive_allocation,
                    "rebalance_date_utc": rebalance_time,
                    "rebalance_changed": rebalance_changed,
                    "signal_changed": signal_changed,
                    "target_changed": target_changed,
                    "position_resize_enabled": resize_existing_positions,
                    "position_resize_triggered": should_resize,
                    "quantity_before": before,
                    "desired_quantity": float(desired),
                    "open_usd": opens[asset],
                    "close_usd": closes[asset],
                    "rate_to_usd": rates[asset],
                }
            )

        ordered = sorted(
            planned,
            key=lambda row: row["desired_quantity"] - row["quantity_before"],
        )
        fills_by_asset: dict[str, Any] = {}
        for row in ordered:
            asset = str(row["instrument_id"])
            target_qty = float(row["desired_quantity"])
            if target_qty > portfolio.position(asset):
                target_qty = _fit_target_to_cash(
                    portfolio,
                    asset,
                    target_qty,
                    float(row["open_usd"]),
                    steps[asset],
                )
            fill = portfolio.target_position(
                timestamp,
                asset,
                target_qty,
                float(row["open_usd"]),
            )
            if fill is not None:
                fills_by_asset[asset] = fill
            last_signal[asset] = float(row["signal"])
            last_target_weight[asset] = float(row["target_weight"])
            last_rebalance[asset] = pd.Timestamp(row["rebalance_date_utc"])

        portfolio.mark(timestamp, closes)
        equity_at_close = float(portfolio.equity())
        for row in planned:
            asset = str(row["instrument_id"])
            after = float(portfolio.position(asset))
            before = float(row["quantity_before"])
            action = "buy" if after > before else "sell" if after < before else "hold"
            fill = fills_by_asset.get(asset)
            decisions.append(
                {
                    **row,
                    "action": action,
                    "quantity_after": after,
                    "order_quantity": after - before,
                    "transaction_cost_usd": float(fill.total_cost) if fill else 0.0,
                    "portfolio_equity_close_usd": equity_at_close,
                }
            )

    curve = portfolio.equity_curve
    if curve.empty:
        raise ValueError("Portfolio simulation produced no equity observations")
    position_columns = [f"pos_{asset}" for asset in assets]
    positions = curve.reindex(columns=position_columns, fill_value=0.0).copy()
    positions.columns = assets
    result = BacktestResult(
        equity=curve["equity"],
        trades=portfolio.trades,
        fills=portfolio.fills,
        positions=positions,
        curve=curve,
        portfolio=portfolio,
    )
    initial_time = curve.index[0] - pd.Timedelta(nanoseconds=1)
    metric_equity = pd.concat(
        [pd.Series([initial_cash], index=[initial_time]), result.equity]
    )
    metric_positions = pd.concat(
        [pd.DataFrame(0.0, index=[initial_time], columns=assets), positions]
    )
    metrics = compute_metrics(
        metric_equity,
        result.trades,
        metric_positions,
        result.fills,
        periods_per_year=risk_metric_periods_per_year,
        risk_free_rate=(
            risk_free_annual_rates
            if risk_free_annual_rates is not None
            else 0.0
        ),
        metric_frequency="daily",
        risk_free_source=risk_free_source,
        drop_incomplete_final_day=True,
        annualize_by_elapsed_time=True,
    )
    metrics["Initial Equity"] = initial_cash
    metrics["Final Equity"] = float(result.equity.iloc[-1])
    metrics["Net Profit"] = float(result.equity.iloc[-1] - initial_cash)
    metrics["Minimum Cash Weight"] = float((curve["cash"] / curve["equity"]).min())
    metrics["Average Cash Weight"] = float((curve["cash"] / curve["equity"]).mean())
    metrics["Average Gross Exposure"] = float(curve["gross_exposure"].mean())
    metrics["Maximum Gross Exposure"] = float(curve["gross_exposure"].max())

    signal_output = signal_output.loc[
        signal_output["bar_open_time_utc"].ge(effective_start)
        & signal_output["bar_open_time_utc"].lt(end)
    ].reset_index(drop=True)
    return PortfolioSimulationArtifacts(
        result=result,
        metrics=metrics,
        decisions=pd.DataFrame(decisions),
        signals=signal_output,
        asset_summary=_asset_summary(assets, portfolio),
        unavailable_signal_assets=unavailable_signal_assets,
    )

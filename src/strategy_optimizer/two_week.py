"""Cost-aware short-horizon screening for SPY and ES3.SI strategies."""

from __future__ import annotations

import copy
import itertools
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping

import numpy as np
import pandas as pd

from src.market_data.costs import CostAssumptions
from src.portfolio_simulation import (
    PortfolioSimulationArtifacts,
    build_execution_signals,
    daily_strategy_returns_after_costs,
    simulate_portfolio,
)
from src.portfolio_sizing import (
    AssetRiskSettings,
    InverseVolatilitySettings,
    build_weight_schedule,
    configured_maximum_weight,
    convert_daily_prices_to_base,
    daily_adjusted_closes,
    rebalance_dates,
)


@dataclass(frozen=True)
class HorizonCandidate:
    """One strategy and decision-frequency combination."""

    candidate_id: str
    instrument_id: str
    strategy: str
    decision_interval_minutes: int
    parameters: dict[str, Any]
    source: str = "grid"

    def strategy_config(self) -> dict[str, Any]:
        return {
            "assets": {
                self.instrument_id: {
                    "signal_source": "rule_based",
                    "strategy": self.strategy,
                    "decision_interval_minutes": self.decision_interval_minutes,
                    "parameters": self.parameters,
                }
            }
        }


@dataclass(frozen=True)
class WindowDefinition:
    """One independent evaluation window defined by local trading sessions."""

    window_id: str
    start: pd.Timestamp
    end: pd.Timestamp
    session_count: int


def _candidate_key(candidate: HorizonCandidate) -> str:
    return json.dumps(
        {
            "instrument_id": candidate.instrument_id,
            "strategy": candidate.strategy,
            "decision_interval_minutes": candidate.decision_interval_minutes,
            "parameters": candidate.parameters,
        },
        sort_keys=True,
    )


def _median_base_bars_per_session(
    canonical: pd.DataFrame,
    instrument_id: str,
) -> int:
    bars = canonical.loc[canonical["instrument_id"].eq(instrument_id)].copy()
    if bars.empty:
        raise ValueError(f"No canonical bars exist for {instrument_id}")
    if "session_date_local" in bars.columns:
        session = bars["session_date_local"].astype(str)
    else:
        session = bars["bar_open_time_utc"].dt.date.astype(str)
    count = int(round(float(bars.assign(_session=session).groupby("_session").size().median())))
    if count < 1:
        raise ValueError(f"Cannot infer bars per session for {instrument_id}")
    return count


def _decision_bars_per_session(base_bars: int, interval_minutes: int) -> int:
    if interval_minutes < 30 or interval_minutes % 30:
        raise ValueError("Decision intervals must be positive multiples of 30 minutes")
    return max(int(math.ceil(base_bars / (interval_minutes / 30))), 1)


def _scaled_sessions(value: float, decision_bars_per_session: int) -> int:
    return max(int(round(float(value) * decision_bars_per_session)), 1)


def generate_horizon_candidates(
    canonical: pd.DataFrame,
    optimization_config: Mapping[str, Any],
    selected_strategy_config: Mapping[str, Any],
) -> dict[str, list[HorizonCandidate]]:
    """Build compact time-normalized grids and preserve current baselines."""
    configured = optimization_config.get("assets")
    selected = selected_strategy_config.get("assets")
    if not isinstance(configured, Mapping) or not isinstance(selected, Mapping):
        raise ValueError("Both optimisation and selected-strategy configs need assets")

    output: dict[str, list[HorizonCandidate]] = {}
    for instrument_id, asset_config in configured.items():
        selected_asset = selected.get(instrument_id)
        if not isinstance(selected_asset, Mapping):
            raise ValueError(f"No selected baseline strategy for {instrument_id}")
        strategy = str(asset_config.get("strategy", ""))
        baseline = HorizonCandidate(
            candidate_id=f"{instrument_id}__baseline",
            instrument_id=instrument_id,
            strategy=strategy,
            decision_interval_minutes=int(
                selected_asset.get("decision_interval_minutes", 30)
            ),
            parameters=dict(selected_asset.get("parameters", {})),
            source="baseline",
        )
        candidates = [baseline]
        seen = {_candidate_key(baseline)}
        base_bars = _median_base_bars_per_session(canonical, instrument_id)
        sequence = 0

        for interval in asset_config.get("decision_intervals_minutes", []):
            interval = int(interval)
            bars_per_session = _decision_bars_per_session(base_bars, interval)
            if strategy == "momentum":
                combinations = itertools.product(
                    asset_config["fast_horizon_sessions"],
                    asset_config["slow_horizon_sessions"],
                    asset_config["return_lookback_sessions"],
                    asset_config["entry_thresholds"],
                )
                for fast_days, slow_days, return_days, entry in combinations:
                    parameters = {
                        "fast_window": _scaled_sessions(fast_days, bars_per_session),
                        "slow_window": _scaled_sessions(slow_days, bars_per_session),
                        "momentum_lookback": _scaled_sessions(
                            return_days, bars_per_session
                        ),
                        "entry_threshold": float(entry),
                        "exit_threshold": float(asset_config["exit_threshold"]),
                    }
                    if parameters["slow_window"] <= parameters["fast_window"]:
                        continue
                    sequence += 1
                    candidate = HorizonCandidate(
                        candidate_id=f"{instrument_id}__momentum_{sequence:04d}",
                        instrument_id=instrument_id,
                        strategy=strategy,
                        decision_interval_minutes=interval,
                        parameters=parameters,
                    )
                    key = _candidate_key(candidate)
                    if key not in seen:
                        seen.add(key)
                        candidates.append(candidate)
            elif strategy == "mean_reversion":
                combinations = itertools.product(
                    asset_config["lookback_horizon_sessions"],
                    asset_config["entry_z_scores"],
                    asset_config["exit_z_scores"],
                )
                for lookback_days, entry, exit_value in combinations:
                    parameters = {
                        "window": max(
                            _scaled_sessions(lookback_days, bars_per_session), 2
                        ),
                        "entry_z_score": float(entry),
                        "exit_z_score": float(exit_value),
                        "minimum_holding_bars": _scaled_sessions(
                            asset_config["minimum_holding_sessions"],
                            bars_per_session,
                        ),
                        "cooldown_bars": _scaled_sessions(
                            asset_config["cooldown_sessions"], bars_per_session
                        ),
                        "maximum_holding_bars": _scaled_sessions(
                            asset_config["maximum_holding_sessions"],
                            bars_per_session,
                        ),
                    }
                    sequence += 1
                    candidate = HorizonCandidate(
                        candidate_id=f"{instrument_id}__mean_reversion_{sequence:04d}",
                        instrument_id=instrument_id,
                        strategy=strategy,
                        decision_interval_minutes=interval,
                        parameters=parameters,
                    )
                    key = _candidate_key(candidate)
                    if key not in seen:
                        seen.add(key)
                        candidates.append(candidate)
            else:
                raise ValueError(f"Unsupported strategy {strategy!r}")
        output[instrument_id] = candidates
    return output


def build_session_windows(
    canonical: pd.DataFrame,
    instrument_id: str,
    start: pd.Timestamp,
    end: pd.Timestamp,
    sessions_per_window: int,
    step_sessions: int | None = None,
) -> list[WindowDefinition]:
    """Return complete rolling windows of local trading sessions."""
    if sessions_per_window < 1:
        raise ValueError("sessions_per_window must be positive")
    step = sessions_per_window if step_sessions is None else int(step_sessions)
    if step < 1:
        raise ValueError("step_sessions must be positive")
    bars = canonical.loc[
        canonical["instrument_id"].eq(instrument_id)
        & canonical["bar_open_time_utc"].ge(start)
        & canonical["bar_open_time_utc"].lt(end)
    ].copy()
    if bars.empty:
        raise ValueError(f"No {instrument_id} bars overlap the training period")
    if "session_date_local" in bars.columns:
        bars["_session"] = bars["session_date_local"].astype(str)
    else:
        bars["_session"] = bars["bar_open_time_utc"].dt.date.astype(str)
    session_starts = (
        bars.groupby("_session")["bar_open_time_utc"].min().sort_values()
    )
    sessions = list(session_starts.index)
    windows: list[WindowDefinition] = []
    for offset in range(0, len(sessions), step):
        chosen = sessions[offset : offset + sessions_per_window]
        if len(chosen) < sessions_per_window:
            break
        next_offset = offset + sessions_per_window
        window_end = (
            pd.Timestamp(session_starts.iloc[next_offset])
            if next_offset < len(session_starts)
            else end
        )
        windows.append(
            WindowDefinition(
                window_id=f"{instrument_id}__two_week_{len(windows) + 1:03d}",
                start=pd.Timestamp(session_starts.loc[chosen[0]]),
                end=window_end,
                session_count=len(chosen),
            )
        )
    if not windows:
        raise ValueError(
            f"{instrument_id} has fewer than {sessions_per_window} complete sessions"
        )
    return windows


def _single_asset_sizing_config(
    sizing_config: Mapping[str, Any], instrument_id: str
) -> dict[str, Any]:
    output = copy.deepcopy(dict(sizing_config))
    output["assets"] = {
        instrument_id: copy.deepcopy(sizing_config["assets"][instrument_id])
    }
    return output


def _constant_sleeve_weight(
    sizing_config: Mapping[str, Any], instrument_id: str, start: pd.Timestamp
) -> pd.DataFrame:
    maximum = configured_maximum_weight(sizing_config, instrument_id)
    index = pd.DatetimeIndex([start], name="rebalance_date_utc")
    return pd.DataFrame({instrument_id: [maximum]}, index=index)


def build_candidate_signals(
    canonical: pd.DataFrame,
    candidate: HorizonCandidate,
    end: pd.Timestamp,
    project_root: Path,
) -> tuple[dict[str, pd.Series], pd.DataFrame]:
    series, output, unavailable = build_execution_signals(
        canonical,
        candidate.strategy_config(),
        [candidate.instrument_id],
        end,
        project_root=project_root,
        allow_missing_external=False,
    )
    if unavailable:
        raise ValueError("Rule-based candidate unexpectedly lacks signals")
    return series, output


def _simulate_single_asset(
    canonical: pd.DataFrame,
    candidate: HorizonCandidate,
    execution_signals: Mapping[str, pd.Series],
    signal_output: pd.DataFrame,
    fx_rates: pd.DataFrame,
    sizing_config: Mapping[str, Any],
    costs: CostAssumptions,
    start: pd.Timestamp,
    end: pd.Timestamp,
    slippage_bps: float,
) -> PortfolioSimulationArtifacts:
    asset_config = _single_asset_sizing_config(
        sizing_config, candidate.instrument_id
    )
    return simulate_portfolio(
        canonical,
        execution_signals,
        signal_output,
        [],
        _constant_sleeve_weight(sizing_config, candidate.instrument_id, start),
        fx_rates,
        asset_config,
        costs,
        start,
        end,
        slippage_bps=slippage_bps,
    )


def _window_metrics(
    artifacts: PortfolioSimulationArtifacts,
    window: WindowDefinition,
    initial_equity: float,
) -> dict[str, Any]:
    curve = artifacts.result.curve.sort_index()
    in_window = curve.loc[
        (curve.index >= window.start) & (curve.index < window.end)
    ]
    if in_window.empty:
        raise ValueError(f"No equity observations in {window.window_id}")
    prior = curve.loc[curve.index < window.start, "equity"]
    starting_equity = float(prior.iloc[-1]) if not prior.empty else initial_equity
    ending_equity = float(in_window["equity"].iloc[-1])
    path = pd.concat(
        [pd.Series([starting_equity]), in_window["equity"].reset_index(drop=True)]
    )
    drawdown = path / path.cummax() - 1.0

    fills = artifacts.result.fills.copy()
    if fills.empty:
        selected_fills = fills
    else:
        fills["timestamp"] = pd.to_datetime(fills["timestamp"], utc=True)
        selected_fills = fills.loc[
            fills["timestamp"].ge(window.start)
            & fills["timestamp"].lt(window.end)
        ]
    trades = artifacts.result.trades.copy()
    if trades.empty:
        completed = trades
        opened_and_completed = trades
    else:
        trades["entry_time"] = pd.to_datetime(trades["entry_time"], utc=True)
        trades["exit_time"] = pd.to_datetime(trades["exit_time"], utc=True)
        completed = trades.loc[
            trades["exit_time"].ge(window.start)
            & trades["exit_time"].lt(window.end)
        ]
        opened_and_completed = completed.loc[completed["entry_time"].ge(window.start)]

    cost = (
        float(selected_fills["total_cost"].sum())
        if not selected_fills.empty
        else 0.0
    )
    net_profit = ending_equity - starting_equity
    return {
        "window_id": window.window_id,
        "window_start_utc": window.start,
        "window_end_utc_exclusive": window.end,
        "session_count": window.session_count,
        "starting_equity_usd": starting_equity,
        "ending_equity_usd": ending_equity,
        "net_profit_usd": net_profit,
        "net_return": net_profit / starting_equity,
        "gross_profit_before_cost_usd": net_profit + cost,
        "transaction_cost_usd": cost,
        "transaction_cost_rate": cost / starting_equity,
        "max_drawdown": float(drawdown.min()),
        "orders": int(len(selected_fills)),
        "completed_round_trips": int(len(completed)),
        "round_trips_opened_and_closed_in_window": int(len(opened_and_completed)),
        "has_completed_round_trip": bool(len(completed) >= 1),
        "has_executed_trade": bool(len(selected_fills) >= 1),
        "profitable_after_cost": bool(net_profit > 0),
    }


def _summary_from_windows(
    candidate: HorizonCandidate,
    artifacts: PortfolioSimulationArtifacts,
    window_rows: list[dict[str, Any]],
) -> dict[str, Any]:
    windows = pd.DataFrame(window_rows)
    asset_row = artifacts.asset_summary.loc[
        artifacts.asset_summary["instrument_id"].eq(candidate.instrument_id)
    ].iloc[0]
    return {
        "instrument_id": candidate.instrument_id,
        "strategy": candidate.strategy,
        "candidate_id": candidate.candidate_id,
        "candidate_source": candidate.source,
        "decision_interval_minutes": candidate.decision_interval_minutes,
        "parameters": json.dumps(candidate.parameters, sort_keys=True),
        "status": "completed",
        "error": "",
        "full_period_net_return": float(artifacts.metrics["Total Return"]),
        "full_period_net_profit_usd": float(artifacts.metrics["Net Profit"]),
        "full_period_max_drawdown": float(artifacts.metrics["Max Drawdown"]),
        "full_period_orders": int(asset_row["orders"]),
        "full_period_completed_round_trips": int(
            asset_row["completed_round_trips"]
        ),
        "full_period_transaction_cost_usd": float(
            asset_row["transaction_cost_usd"]
        ),
        "two_week_window_count": int(len(windows)),
        "round_trip_window_count": int(windows["has_completed_round_trip"].sum()),
        "round_trip_window_rate": float(
            windows["has_completed_round_trip"].mean()
        ),
        "trade_window_count": int(windows["has_executed_trade"].sum()),
        "trade_window_rate": float(windows["has_executed_trade"].mean()),
        "profitable_window_rate": float(windows["profitable_after_cost"].mean()),
        "median_two_week_net_return": float(windows["net_return"].median()),
        "worst_two_week_net_return": float(windows["net_return"].min()),
        "median_two_week_cost_usd": float(
            windows["transaction_cost_usd"].median()
        ),
        "median_two_week_cost_rate": float(
            windows["transaction_cost_rate"].median()
        ),
    }


def screen_candidates(
    canonical: pd.DataFrame,
    candidates: Iterable[HorizonCandidate],
    windows: list[WindowDefinition],
    fx_rates: pd.DataFrame,
    sizing_config: Mapping[str, Any],
    costs: CostAssumptions,
    training_start: pd.Timestamp,
    training_end: pd.Timestamp,
    project_root: Path,
    *,
    slippage_bps: float = 0.0,
    progress_every: int = 10,
    minimum_trade_window_rate: float = 0.0,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Run each candidate once and derive inexpensive two-week diagnostics."""
    candidate_list = list(candidates)
    summaries: list[dict[str, Any]] = []
    window_output: list[dict[str, Any]] = []
    initial_equity = float(sizing_config.get("initial_cash_base", 100_000.0))

    for index, candidate in enumerate(candidate_list, start=1):
        if index == 1 or index % progress_every == 0 or index == len(candidate_list):
            print(
                f"  {candidate.instrument_id}: candidate {index:,}/"
                f"{len(candidate_list):,} ({candidate.candidate_id})"
            )
        try:
            signals, signal_output = build_candidate_signals(
                canonical, candidate, training_end, project_root
            )
            artifacts = _simulate_single_asset(
                canonical,
                candidate,
                signals,
                signal_output,
                fx_rates,
                sizing_config,
                costs,
                training_start,
                training_end,
                slippage_bps,
            )
            rows = []
            for window in windows:
                metrics = _window_metrics(artifacts, window, initial_equity)
                metrics.update(
                    {
                        "instrument_id": candidate.instrument_id,
                        "candidate_id": candidate.candidate_id,
                        "evaluation_mode": "continuous_screen",
                    }
                )
                rows.append(metrics)
            summaries.append(_summary_from_windows(candidate, artifacts, rows))
            window_output.extend(rows)
        except (ValueError, KeyError, TypeError, IndexError) as error:
            summaries.append(
                {
                    "instrument_id": candidate.instrument_id,
                    "strategy": candidate.strategy,
                    "candidate_id": candidate.candidate_id,
                    "candidate_source": candidate.source,
                    "decision_interval_minutes": candidate.decision_interval_minutes,
                    "parameters": json.dumps(candidate.parameters, sort_keys=True),
                    "status": "failed",
                    "error": str(error),
                }
            )

    summary = pd.DataFrame(summaries)
    completed = summary["status"].eq("completed")
    summary["selection_eligible"] = completed & summary[
        "trade_window_rate"
    ].ge(minimum_trade_window_rate)
    summary["screen_rank"] = pd.Series(pd.NA, index=summary.index, dtype="Int64")
    ranked_index = summary.loc[completed].sort_values(
        [
            "selection_eligible",
            "median_two_week_net_return",
            "profitable_window_rate",
            "worst_two_week_net_return",
            "median_two_week_cost_rate",
        ],
        ascending=[False, False, False, False, True],
        kind="stable",
    ).index
    summary.loc[ranked_index, "screen_rank"] = range(1, len(ranked_index) + 1)
    summary = summary.sort_values(
        ["status", "screen_rank"], ascending=[False, True], na_position="last"
    ).reset_index(drop=True)
    return summary, pd.DataFrame(window_output)


def shortlist_candidates(
    summary: pd.DataFrame,
    candidates: Iterable[HorizonCandidate],
    size: int,
) -> list[HorizonCandidate]:
    """Return the baseline plus the highest-ranked candidates, without selection."""
    by_id = {candidate.candidate_id: candidate for candidate in candidates}
    completed = summary.loc[summary["status"].eq("completed")].sort_values(
        "screen_rank"
    )
    ids = list(completed.head(size)["candidate_id"].astype(str))
    baseline_ids = list(
        completed.loc[completed["candidate_source"].eq("baseline"), "candidate_id"]
        .astype(str)
    )
    ordered = list(dict.fromkeys(baseline_ids + ids))
    return [by_id[candidate_id] for candidate_id in ordered]


def evaluate_independent_windows(
    canonical: pd.DataFrame,
    candidates: Iterable[HorizonCandidate],
    windows: list[WindowDefinition],
    fx_rates: pd.DataFrame,
    sizing_config: Mapping[str, Any],
    costs: CostAssumptions,
    signal_history_end: pd.Timestamp,
    project_root: Path,
    *,
    slippage_bps: float = 0.0,
    minimum_trade_window_rate: float = 0.0,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Reset cash and positions for every shortlisted two-week window."""
    rows: list[dict[str, Any]] = []
    candidate_list = list(candidates)
    for index, candidate in enumerate(candidate_list, start=1):
        print(
            f"  exact windows {candidate.instrument_id}: candidate {index}/"
            f"{len(candidate_list)} ({candidate.candidate_id})"
        )
        signals, signal_output = build_candidate_signals(
            canonical, candidate, signal_history_end, project_root
        )
        for window in windows:
            artifacts = _simulate_single_asset(
                canonical,
                candidate,
                signals,
                signal_output,
                fx_rates,
                sizing_config,
                costs,
                window.start,
                window.end,
                slippage_bps,
            )
            metrics = _window_metrics(
                artifacts,
                window,
                float(sizing_config.get("initial_cash_base", 100_000.0)),
            )
            metrics.update(
                {
                    "instrument_id": candidate.instrument_id,
                    "candidate_id": candidate.candidate_id,
                    "decision_interval_minutes": candidate.decision_interval_minutes,
                    "parameters": json.dumps(candidate.parameters, sort_keys=True),
                    "evaluation_mode": "independent_reset",
                }
            )
            rows.append(metrics)

    detail = pd.DataFrame(rows)
    grouped = detail.groupby(
        [
            "instrument_id",
            "candidate_id",
            "decision_interval_minutes",
            "parameters",
        ],
        as_index=False,
    )
    summary = grouped.agg(
        independent_window_count=("window_id", "count"),
        independent_round_trip_window_count=("has_completed_round_trip", "sum"),
        independent_round_trip_window_rate=("has_completed_round_trip", "mean"),
        independent_trade_window_count=("has_executed_trade", "sum"),
        independent_trade_window_rate=("has_executed_trade", "mean"),
        independent_profitable_window_rate=("profitable_after_cost", "mean"),
        independent_median_net_return=("net_return", "median"),
        independent_worst_net_return=("net_return", "min"),
        independent_median_cost_usd=("transaction_cost_usd", "median"),
        independent_median_cost_rate=("transaction_cost_rate", "median"),
        independent_total_cost_usd=("transaction_cost_usd", "sum"),
    )
    summary["selection_eligible"] = summary[
        "independent_trade_window_rate"
    ].ge(minimum_trade_window_rate)
    summary = summary.sort_values(
        [
            "instrument_id",
            "selection_eligible",
            "independent_median_net_return",
            "independent_profitable_window_rate",
            "independent_worst_net_return",
            "independent_median_cost_rate",
        ],
        ascending=[True, False, False, False, False, True],
        kind="stable",
    ).reset_index(drop=True)
    summary["review_rank"] = summary.groupby("instrument_id").cumcount() + 1
    return detail, summary


def _risk_weights_for_joint_simulation(
    canonical: pd.DataFrame,
    fx_rates: pd.DataFrame,
    sizing_config: Mapping[str, Any],
    start: pd.Timestamp,
    end: pd.Timestamp,
    sizing_method: str,
    strategy_returns: pd.DataFrame | None = None,
) -> pd.DataFrame:
    configured_assets = sizing_config["assets"]
    assets = list(configured_assets)
    daily_native = daily_adjusted_closes(canonical, assets)
    quote_currencies = {
        asset: str(configured_assets[asset]["quote_currency"]).upper()
        for asset in assets
    }
    daily_base, _ = convert_daily_prices_to_base(
        daily_native,
        quote_currencies,
        str(sizing_config.get("base_currency", "USD")),
        static_rates_to_base=sizing_config.get("static_rates_to_base", {}),
        fx_rates=fx_rates,
    )
    policy = InverseVolatilitySettings(
        method=sizing_method,
        lookback_observations=int(sizing_config.get("lookback_observations", 60)),
        minimum_observations=int(sizing_config.get("minimum_observations", 40)),
        target_gross_exposure=float(sizing_config.get("target_gross_exposure", 0.95)),
        volatility_floor=float(sizing_config.get("volatility_floor", 0.05)),
        rebalance_frequency=str(sizing_config.get("rebalance_frequency", "weekly")),
        confidence_full_observations=int(
            sizing_config.get("confidence_full_observations", 252)
        ),
        confidence_power=float(sizing_config.get("confidence_power", 0.5)),
        performance_lookback_observations=int(
            sizing_config.get("performance_lookback_observations", 60)
        ),
        minimum_performance_observations=int(
            sizing_config.get("minimum_performance_observations", 20)
        ),
        performance_sharpe_scale=float(
            sizing_config.get("performance_sharpe_scale", 2.0)
        ),
        performance_multiplier_floor=float(
            sizing_config.get("performance_multiplier_floor", 0.25)
        ),
        performance_multiplier_ceiling=float(
            sizing_config.get("performance_multiplier_ceiling", 1.5)
        ),
    )
    settings = {
        asset: AssetRiskSettings(
            annualization_periods=int(configured_assets[asset]["annualization_periods"]),
            maximum_weight=configured_maximum_weight(sizing_config, asset),
            quantity_step=float(configured_assets[asset]["quantity_step"]),
        )
        for asset in assets
    }
    weights, _ = build_weight_schedule(
        daily_base[assets],
        rebalance_dates(start, end, policy.rebalance_frequency),
        policy,
        settings,
        strategy_returns=strategy_returns,
    )
    if weights.empty:
        raise ValueError("No joint portfolio weights are ready for the diagnostic")
    return weights


def run_joint_diagnostics(
    canonical: pd.DataFrame,
    spy_candidates: Iterable[HorizonCandidate],
    sti_candidates: Iterable[HorizonCandidate],
    selected_strategy_config: Mapping[str, Any],
    fx_rates: pd.DataFrame,
    sizing_config: Mapping[str, Any],
    costs: CostAssumptions,
    start: pd.Timestamp,
    end: pd.Timestamp,
    project_root: Path,
    *,
    sizing_method: str,
    slippage_bps: float = 0.0,
) -> pd.DataFrame:
    """Replay finalist pairs using the actual shared-USD portfolio engine."""
    strategy_performance_sizing = (
        sizing_method
        == "capped_strategy_performance_adjusted_inverse_volatility"
    )
    weights = None
    if not strategy_performance_sizing:
        weights = _risk_weights_for_joint_simulation(
            canonical, fx_rates, sizing_config, start, end, sizing_method
        )
    assets = list(sizing_config["assets"])
    rows: list[dict[str, Any]] = []
    pairs = list(itertools.product(spy_candidates, sti_candidates))
    for index, (spy, sti) in enumerate(pairs, start=1):
        print(
            f"  joint diagnostic {index}/{len(pairs)}: "
            f"{spy.candidate_id} + {sti.candidate_id}"
        )
        strategy_config = copy.deepcopy(dict(selected_strategy_config))
        for candidate in (spy, sti):
            strategy_config["assets"][candidate.instrument_id] = {
                "signal_source": "rule_based",
                "strategy": candidate.strategy,
                "decision_interval_minutes": candidate.decision_interval_minutes,
                "parameters": candidate.parameters,
                "selection_source": "two_week_candidate_diagnostic",
            }
        signals, signal_output, unavailable = build_execution_signals(
            canonical,
            strategy_config,
            assets,
            end,
            project_root=project_root,
            allow_missing_external=True,
        )
        if strategy_performance_sizing:
            strategy_returns = daily_strategy_returns_after_costs(
                canonical,
                signals,
                fx_rates,
                sizing_config,
                costs,
                end,
                slippage_bps=slippage_bps,
            )
            weights = _risk_weights_for_joint_simulation(
                canonical,
                fx_rates,
                sizing_config,
                start,
                end,
                sizing_method,
                strategy_returns=strategy_returns,
            )
        if weights is None:
            raise ValueError("No joint portfolio weights are available")
        artifacts = simulate_portfolio(
            canonical,
            signals,
            signal_output,
            unavailable,
            weights,
            fx_rates,
            sizing_config,
            costs,
            start,
            end,
            slippage_bps=slippage_bps,
        )
        asset_summary = artifacts.asset_summary.set_index("instrument_id")
        row = {
            "sizing_method": sizing_method,
            "spy_candidate_id": spy.candidate_id,
            "sti_candidate_id": sti.candidate_id,
            "spy_decision_interval_minutes": spy.decision_interval_minutes,
            "sti_decision_interval_minutes": sti.decision_interval_minutes,
            "spy_parameters": json.dumps(spy.parameters, sort_keys=True),
            "sti_parameters": json.dumps(sti.parameters, sort_keys=True),
            "requested_start_utc": start,
            "requested_end_utc_exclusive": end,
            "effective_start_utc": artifacts.result.equity.index.min(),
            "net_return": float(artifacts.metrics["Total Return"]),
            "net_profit_usd": float(artifacts.metrics["Net Profit"]),
            "gross_profit_before_cost_usd": float(
                artifacts.metrics["Net Profit"]
                + artifacts.metrics["Total Transaction Cost"]
            ),
            "max_drawdown": float(artifacts.metrics["Max Drawdown"]),
            "orders": int(artifacts.metrics["Number of Orders"]),
            "completed_round_trips": int(artifacts.metrics["Number of Trades"]),
            "transaction_cost_usd": float(
                artifacts.metrics["Total Transaction Cost"]
            ),
            "minimum_cash_weight": float(artifacts.metrics["Minimum Cash Weight"]),
            "average_cash_weight": float(artifacts.metrics["Average Cash Weight"]),
            "btc_signals_unavailable": "btc_usdt" in unavailable,
        }
        for instrument_id in ("spy_usd", "sti_es3_sgd"):
            prefix = "spy" if instrument_id == "spy_usd" else "sti"
            row[f"{prefix}_orders"] = int(asset_summary.at[instrument_id, "orders"])
            row[f"{prefix}_completed_round_trips"] = int(
                asset_summary.at[instrument_id, "completed_round_trips"]
            )
            row[f"{prefix}_transaction_cost_usd"] = float(
                asset_summary.at[instrument_id, "transaction_cost_usd"]
            )
            row[f"{prefix}_net_pnl_contribution_usd"] = float(
                asset_summary.at[instrument_id, "net_pnl_contribution_usd"]
            )
        rows.append(row)
    return pd.DataFrame(rows).sort_values(
        ["net_return", "max_drawdown", "transaction_cost_usd"],
        ascending=[False, False, True],
        kind="stable",
    ).reset_index(drop=True)


def candidates_from_review_summary(
    summary: pd.DataFrame,
    candidates: Iterable[HorizonCandidate],
    size: int,
) -> list[HorizonCandidate]:
    """Take top review rows and always retain the current baseline."""
    by_id = {candidate.candidate_id: candidate for candidate in candidates}
    ordered_summary = summary.sort_values("review_rank")
    ids = list(ordered_summary.head(size)["candidate_id"].astype(str))
    baseline = [
        candidate.candidate_id
        for candidate in candidates
        if candidate.source == "baseline"
    ]
    return [by_id[item] for item in dict.fromkeys(baseline + ids)]

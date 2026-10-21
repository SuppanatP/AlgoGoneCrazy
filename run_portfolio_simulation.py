"""Replay the configured multi-asset portfolio through one USD cash ledger."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from src.market_data.costs import load_cost_assumptions  # noqa: E402
from src.portfolio_simulation import (  # noqa: E402
    build_execution_signals,
    daily_strategy_returns_after_costs,
    simulate_portfolio,
)
from src.portfolio_sizing import (  # noqa: E402
    AssetRiskSettings,
    InverseVolatilitySettings,
    SUPPORTED_WEIGHTING_METHODS,
    build_weight_schedule,
    configured_maximum_weight,
    convert_daily_prices_to_base,
    daily_adjusted_closes,
    load_fx_rates,
    rebalance_dates,
)
from src.strategy_backtest import load_canonical_bars  # noqa: E402
from src.risk_free import load_configured_risk_free_rates  # noqa: E402


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input-file",
        type=Path,
        default=Path("data/canonical/market_bars_30m_simulation.csv"),
    )
    parser.add_argument(
        "--fx-file",
        type=Path,
        default=Path("data/fx/fx_rates_to_usd.csv"),
    )
    parser.add_argument(
        "--sizing-config",
        type=Path,
        default=Path("config/three_asset_portfolio.json"),
    )
    parser.add_argument(
        "--strategy-config",
        type=Path,
        default=Path("config/selected_strategies.json"),
    )
    parser.add_argument(
        "--cost-config",
        type=Path,
        default=Path("config/cost_assumptions.json"),
    )
    parser.add_argument(
        "--risk-free-config",
        type=Path,
        default=Path("config/risk_free_rate.json"),
        help="Official risk-free-rate source and daily reporting convention.",
    )
    parser.add_argument("--start", required=True, help="Inclusive UTC start.")
    parser.add_argument("--end", required=True, help="Exclusive UTC end.")
    parser.add_argument(
        "--allow-missing-external",
        action="store_true",
        help=(
            "Treat an absent external signal as zero and retain its sleeve as "
            "cash. Omit this flag for the final complete run."
        ),
    )
    parser.add_argument("--slippage-bps", type=float, default=0.0)
    parser.add_argument(
        "--sizing-method",
        choices=sorted(SUPPORTED_WEIGHTING_METHODS),
        help="Override the weighting method configured in the sizing file.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("results/portfolio_simulation"),
    )
    return parser


def _read_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(f"JSON file not found: {path}")
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as error:
        raise ValueError(f"Invalid JSON in {path}: {error}") from error
    if not isinstance(value, dict):
        raise ValueError(f"{path} must contain a JSON object")
    return value


def _utc(value: object, label: str) -> pd.Timestamp:
    try:
        timestamp = pd.Timestamp(value)
    except (TypeError, ValueError) as error:
        raise ValueError(f"{label} must be a valid timestamp") from error
    if timestamp.tzinfo is None:
        return timestamp.tz_localize("UTC")
    return timestamp.tz_convert("UTC")


def main() -> None:
    args = build_parser().parse_args()
    start = _utc(args.start, "start")
    end = _utc(args.end, "end")
    if end <= start:
        raise ValueError("end must be later than start")
    if args.slippage_bps < 0:
        raise ValueError("slippage-bps must not be negative")

    sizing_config = _read_json(args.sizing_config)
    strategy_config = _read_json(args.strategy_config)
    configured_assets = sizing_config.get("assets")
    if not isinstance(configured_assets, dict) or not configured_assets:
        raise ValueError("Sizing configuration must contain assets")
    assets = list(configured_assets)

    canonical = load_canonical_bars(args.input_file)
    base_currency = str(sizing_config.get("base_currency", "")).strip().upper()
    quote_currencies = {
        asset: str(configured_assets[asset]["quote_currency"]).strip().upper()
        for asset in assets
    }
    daily_native = daily_adjusted_closes(canonical, assets)
    fx_rates = load_fx_rates(args.fx_file, base_currency)
    daily_base, conversion_audit = convert_daily_prices_to_base(
        daily_native,
        quote_currencies,
        base_currency,
        static_rates_to_base=sizing_config.get("static_rates_to_base", {}),
        fx_rates=fx_rates,
    )

    cost_assumptions = load_cost_assumptions(args.cost_config)
    risk_free_rates, risk_free_settings = load_configured_risk_free_rates(
        args.risk_free_config
    )
    execution_signals, signal_output, unavailable = build_execution_signals(
        canonical,
        strategy_config,
        assets,
        end,
        project_root=PROJECT_ROOT,
        allow_missing_external=args.allow_missing_external,
    )
    selected_sizing_method = str(
        args.sizing_method
        or sizing_config.get("method", "capped_inverse_volatility")
    )
    policy = InverseVolatilitySettings(
        method=selected_sizing_method,
        lookback_observations=int(sizing_config.get("lookback_observations", 60)),
        minimum_observations=int(sizing_config.get("minimum_observations", 40)),
        target_gross_exposure=float(
            sizing_config.get("target_gross_exposure", 0.95)
        ),
        volatility_floor=float(sizing_config.get("volatility_floor", 0.05)),
        rebalance_frequency=str(
            sizing_config.get("rebalance_frequency", "weekly")
        ),
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
    strategy_returns = None
    if selected_sizing_method in {
        "capped_strategy_performance_adjusted_inverse_volatility",
        "capped_strategy_return_inverse_volatility",
    }:
        strategy_returns = daily_strategy_returns_after_costs(
            canonical,
            execution_signals,
            fx_rates,
            sizing_config,
            cost_assumptions,
            end,
            slippage_bps=args.slippage_bps,
        )
    asset_settings = {
        asset: AssetRiskSettings(
            annualization_periods=int(configured_assets[asset]["annualization_periods"]),
            maximum_weight=configured_maximum_weight(sizing_config, asset),
            quantity_step=float(configured_assets[asset]["quantity_step"]),
        )
        for asset in assets
    }
    risk_weights, volatility_diagnostics = build_weight_schedule(
        daily_base[assets],
        rebalance_dates(start, end, policy.rebalance_frequency),
        policy,
        asset_settings,
        strategy_returns=strategy_returns,
    )
    if risk_weights.empty:
        raise ValueError("No sizing date has sufficient history for every asset")

    artifacts = simulate_portfolio(
        canonical,
        execution_signals,
        signal_output,
        unavailable,
        risk_weights,
        fx_rates,
        sizing_config,
        cost_assumptions,
        start,
        end,
        slippage_bps=args.slippage_bps,
        risk_free_annual_rates=risk_free_rates,
        risk_free_source=str(risk_free_settings["source_name"]),
        risk_metric_periods_per_year=int(
            risk_free_settings["annualization_days"]
        ),
    )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    weight_output = risk_weights.copy()
    weight_output["allocated_weight"] = weight_output.sum(axis=1)
    weight_output["strategic_cash_weight"] = 1.0 - weight_output["allocated_weight"]
    weight_output.reset_index().to_csv(
        args.output_dir / "risk_weights.csv", index=False
    )
    volatility_diagnostics.to_csv(
        args.output_dir / "volatility_diagnostics.csv", index=False
    )
    if strategy_returns is not None:
        strategy_returns.reset_index().to_csv(
            args.output_dir / "strategy_return_history.csv", index=False
        )
    conversion_audit.to_csv(args.output_dir / "currency_conversion.csv", index=False)
    artifacts.signals.to_csv(args.output_dir / "execution_signals.csv", index=False)
    artifacts.decisions.to_csv(args.output_dir / "decisions.csv", index=False)
    artifacts.result.fills.to_csv(args.output_dir / "fills.csv", index=False)
    artifacts.result.trades.to_csv(args.output_dir / "trades.csv", index=False)
    artifacts.result.curve.reset_index().to_csv(
        args.output_dir / "equity_curve.csv", index=False
    )
    pd.DataFrame([artifacts.metrics.to_dict()]).to_csv(
        args.output_dir / "performance_summary.csv", index=False
    )
    project_metric_columns = [
        "Net Profit",
        "Total Return",
        "Annual Return",
        "Sharpe Ratio",
        "Max Drawdown",
        "Win Rate",
        "Annual Volatility",
        "Drawdown Count",
        "Number of Trades",
        "Average Holding Period (hours)",
        "Profit per Trade",
    ]
    pd.DataFrame(
        [{column: artifacts.metrics[column] for column in project_metric_columns}]
    ).to_csv(args.output_dir / "project_metrics.csv", index=False)
    artifacts.asset_summary.to_csv(
        args.output_dir / "asset_summary.csv", index=False
    )

    manifest = {
        "generated_at_utc": pd.Timestamp.now(tz="UTC").isoformat(),
        "input_file": str(args.input_file),
        "fx_file": str(args.fx_file),
        "sizing_config": str(args.sizing_config),
        "strategy_config": str(args.strategy_config),
        "cost_config": str(args.cost_config),
        "risk_free_config": str(args.risk_free_config),
        "requested_start_utc": start.isoformat(),
        "requested_end_utc_exclusive": end.isoformat(),
        "effective_start_utc": risk_weights.index.min().isoformat(),
        "base_currency": base_currency,
        "initial_cash_base": float(sizing_config["initial_cash_base"]),
        "external_signal_assets_unavailable": unavailable,
        "missing_external_signals_allowed": bool(args.allow_missing_external),
        "signal_timing": strategy_config.get("signal_timing"),
        "risk_rebalance_frequency": policy.rebalance_frequency,
        "weighting_method": policy.method,
        "maximum_asset_weights": {
            asset: configured_maximum_weight(sizing_config, asset)
            for asset in assets
        },
        "target_weight_change_threshold": float(
            sizing_config.get("target_weight_change_threshold", 0.0)
        ),
        "resize_existing_positions_on_rebalance": bool(
            sizing_config.get("resize_existing_positions_on_rebalance", True)
        ),
        "strategy_inactive_allocation": sizing_config.get("inactive_allocation"),
        "minimum_active_strategies_for_scaling": (
            int(sizing_config.get("minimum_active_strategies_for_scaling", 2))
            if sizing_config.get("inactive_allocation")
            == "bounded_active_capital_scaling"
            else None
        ),
        "maximum_active_scaling_multiplier": (
            float(sizing_config.get("maximum_active_scaling_multiplier", 2.0))
            if sizing_config.get("inactive_allocation")
            == "bounded_active_capital_scaling"
            else None
        ),
        "strategy_performance_measure": sizing_config.get("performance_measure"),
        "strategy_performance_reference_notional_base": float(
            sizing_config.get(
                "strategy_performance_reference_notional_base", 100_000.0
            )
        ),
        "force_liquidation_at_end": False,
        "valuation_prices": "raw_close_converted_point_in_time_to_usd",
        "execution_prices": "next_eligible_raw_open_converted_point_in_time_to_usd",
        "feature_prices": "adjusted_close",
        "dividend_cash_flows_included": False,
        "slippage_bps": args.slippage_bps,
        "risk_metrics": {
            "return_frequency": risk_free_settings[
                "portfolio_return_frequency"
            ],
            "daily_boundary_timezone": risk_free_settings[
                "daily_boundary_timezone"
            ],
            "partial_final_day_policy": risk_free_settings[
                "partial_final_day_policy"
            ],
            "annualization_days": int(
                risk_free_settings["annualization_days"]
            ),
            "annual_return_method": risk_free_settings[
                "annual_return_method"
            ],
            "risk_free_benchmark": risk_free_settings["benchmark"],
            "risk_free_source": risk_free_settings["source_name"],
            "risk_free_source_url": risk_free_settings["source_url"],
            "risk_free_input_file": risk_free_settings["input_file"],
            "risk_free_yield_column": risk_free_settings["yield_column"],
            "risk_free_availability_lag_days": int(
                risk_free_settings["availability_lag_days"]
            ),
            "cash_interest_policy": risk_free_settings[
                "cash_interest_policy"
            ],
            "drawdown_frequency": risk_free_settings["drawdown_frequency"],
        },
    }
    (args.output_dir / "simulation_manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )

    print("\nPortfolio simulation")
    print(f"  effective start: {artifacts.result.equity.index.min().isoformat()}")
    print(f"  final timestamp: {artifacts.result.equity.index.max().isoformat()}")
    print(f"  initial equity: {artifacts.metrics['Initial Equity']:,.2f} USD")
    print(f"  final equity: {artifacts.metrics['Final Equity']:,.2f} USD")
    print(f"  net return: {artifacts.metrics['Total Return']:.2%}")
    print(f"  orders: {int(artifacts.metrics['Number of Orders'])}")
    print(f"  completed round trips: {int(artifacts.metrics['Number of Trades'])}")
    if unavailable:
        print("  awaiting external signals: " + ", ".join(unavailable))
    print("\nPer-asset activity")
    print(
        artifacts.asset_summary[
            [
                "instrument_id",
                "orders",
                "completed_round_trips",
                "net_pnl_contribution_usd",
            ]
        ].to_string(index=False, float_format=lambda value: f"{value:,.2f}")
    )
    print(f"\nOutputs -> {args.output_dir}")


if __name__ == "__main__":
    main()

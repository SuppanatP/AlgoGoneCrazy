"""Reproduce the pre-cutoff SPY/STI parameter-selection workflow."""

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
from src.portfolio_sizing import load_fx_rates  # noqa: E402
from src.portfolio_sizing import SUPPORTED_WEIGHTING_METHODS  # noqa: E402
from src.strategy_backtest import load_canonical_bars  # noqa: E402
from src.strategy_optimizer.two_week import (  # noqa: E402
    build_session_windows,
    candidates_from_review_summary,
    evaluate_independent_windows,
    generate_horizon_candidates,
    run_joint_diagnostics,
    screen_candidates,
    shortlist_candidates,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input-file",
        type=Path,
        default=Path("data/canonical/market_bars_30m_training_validation.csv"),
    )
    parser.add_argument(
        "--fx-file", type=Path, default=Path("data/fx/fx_rates_to_usd.csv")
    )
    parser.add_argument(
        "--config-file",
        type=Path,
        default=Path("config/sti_mean_reversion_expanded_optimization.json"),
    )
    parser.add_argument(
        "--selected-strategies",
        type=Path,
        default=Path("config/strategy_search_baseline.json"),
        help=(
            "Pre-search strategy baseline used as a comparator; it must not be "
            "the frozen output selection."
        ),
    )
    parser.add_argument(
        "--sizing-config",
        type=Path,
        default=Path("config/portfolio_sizing.json"),
    )
    parser.add_argument(
        "--cost-config",
        type=Path,
        default=Path("config/cost_assumptions.json"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("results/final_strategy_selection"),
    )
    parser.add_argument("--slippage-bps", type=float, default=0.0)
    parser.add_argument(
        "--max-candidates",
        type=int,
        help="Limit candidates per asset for a workflow check only.",
    )
    parser.add_argument(
        "--skip-joint-diagnostic",
        action="store_true",
        help=(
            "Skip the rolling sizing-method comparison and configured pre-cutoff "
            "SPY/STI portfolio validation."
        ),
    )
    parser.add_argument("--progress-every", type=int, default=10)
    return parser


def _read_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(f"JSON file not found: {path}")
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"{path} must contain a JSON object")
    return value


def _utc(value: Any, label: str) -> pd.Timestamp:
    try:
        timestamp = pd.Timestamp(value)
    except (TypeError, ValueError) as error:
        raise ValueError(f"{label} must be a valid timestamp") from error
    if timestamp.tzinfo is None:
        return timestamp.tz_localize("UTC")
    return timestamp.tz_convert("UTC")


def main() -> None:
    args = build_parser().parse_args()
    if args.slippage_bps < 0:
        raise ValueError("--slippage-bps must not be negative")
    if args.max_candidates is not None and args.max_candidates < 1:
        raise ValueError("--max-candidates must be at least 1")
    if args.progress_every < 1:
        raise ValueError("--progress-every must be at least 1")

    config = _read_json(args.config_file)
    selected = _read_json(args.selected_strategies)
    sizing = _read_json(args.sizing_config)
    comparison_methods = sizing.get(
        "comparison_methods",
        [sizing.get("method", "capped_inverse_volatility")],
    )
    if not isinstance(comparison_methods, list) or not comparison_methods:
        raise ValueError("comparison_methods must be a non-empty list")
    comparison_methods = [str(method) for method in comparison_methods]
    unsupported_methods = set(comparison_methods) - SUPPORTED_WEIGHTING_METHODS
    if unsupported_methods:
        raise ValueError(
            f"Unsupported sizing methods: {sorted(unsupported_methods)}"
        )
    canonical = load_canonical_bars(args.input_file)
    fx_rates = load_fx_rates(args.fx_file, str(sizing.get("base_currency", "USD")))
    costs = load_cost_assumptions(args.cost_config)
    sessions_per_window = int(config.get("sessions_per_evaluation_window", 10))
    window_step_sessions = int(
        config.get("window_step_sessions", sessions_per_window)
    )
    if sessions_per_window < 1 or window_step_sessions < 1:
        raise ValueError("Evaluation window and step sessions must be positive")
    minimum_trade_window_rate = float(
        config.get("minimum_trade_window_rate", 0.0)
    )
    if not 0 <= minimum_trade_window_rate <= 1:
        raise ValueError("minimum_trade_window_rate must be between zero and one")
    shortlist_size = int(config.get("shortlist_size", 5))
    joint_size = int(config.get("joint_shortlist_size", 3))

    official_test_start = _utc(
        config.get("official_test_start_utc"), "official_test_start_utc"
    )
    candidates_by_asset = generate_horizon_candidates(canonical, config, selected)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    all_summaries: list[pd.DataFrame] = []
    all_screen_windows: list[pd.DataFrame] = []
    all_exact_windows: list[pd.DataFrame] = []
    all_exact_summaries: list[pd.DataFrame] = []
    reviewed_candidates: dict[str, list[Any]] = {}
    windows_by_asset: dict[str, list[Any]] = {}

    for instrument_id, candidates in candidates_by_asset.items():
        asset_config = config["assets"][instrument_id]
        training = asset_config["training_period"]
        start = _utc(training["start"], f"{instrument_id} training start")
        end = _utc(training["end"], f"{instrument_id} training end")
        if end > official_test_start:
            raise ValueError(f"{instrument_id} training enters official test data")
        windows = build_session_windows(
            canonical,
            instrument_id,
            start,
            end,
            sessions_per_window,
            window_step_sessions,
        )
        windows_by_asset[instrument_id] = windows
        if args.max_candidates is not None:
            baseline = [candidate for candidate in candidates if candidate.source == "baseline"]
            grid = [candidate for candidate in candidates if candidate.source != "baseline"]
            candidates = (baseline + grid)[: args.max_candidates]
            candidates_by_asset[instrument_id] = candidates

        print(
            f"\nScreening {instrument_id}: {len(candidates)} candidates across "
            f"{len(windows)} rolling {sessions_per_window}-session windows "
            f"stepped every {window_step_sessions} sessions"
        )
        summary, screen_windows = screen_candidates(
            canonical,
            candidates,
            windows,
            fx_rates,
            sizing,
            costs,
            start,
            end,
            PROJECT_ROOT,
            slippage_bps=args.slippage_bps,
            progress_every=args.progress_every,
            minimum_trade_window_rate=minimum_trade_window_rate,
        )
        shortlist = shortlist_candidates(summary, candidates, shortlist_size)
        exact_windows, exact_summary = evaluate_independent_windows(
            canonical,
            shortlist,
            windows,
            fx_rates,
            sizing,
            costs,
            end,
            PROJECT_ROOT,
            slippage_bps=args.slippage_bps,
            minimum_trade_window_rate=minimum_trade_window_rate,
        )
        reviewed_candidates[instrument_id] = candidates_from_review_summary(
            exact_summary, shortlist, joint_size
        )
        all_summaries.append(summary)
        all_screen_windows.append(screen_windows)
        all_exact_windows.append(exact_windows)
        all_exact_summaries.append(exact_summary)

    candidate_summary = pd.concat(all_summaries, ignore_index=True)
    screen_window_results = pd.concat(all_screen_windows, ignore_index=True)
    exact_window_results = pd.concat(all_exact_windows, ignore_index=True)
    exact_summary = pd.concat(all_exact_summaries, ignore_index=True)
    candidate_summary.to_csv(args.output_dir / "candidate_screening.csv", index=False)
    screen_window_results.to_csv(
        args.output_dir / "two_week_continuous_windows.csv", index=False
    )
    exact_window_results.to_csv(
        args.output_dir / "two_week_independent_windows.csv", index=False
    )
    exact_summary.to_csv(
        args.output_dir / "shortlisted_candidate_review.csv", index=False
    )

    top_candidates: dict[str, Any] = {}
    for instrument_id, candidates in candidates_by_asset.items():
        ranked = exact_summary.loc[
            exact_summary["instrument_id"].eq(instrument_id)
        ].sort_values("review_rank")
        if ranked.empty:
            raise ValueError(f"No reviewed candidate is available for {instrument_id}")
        candidate_id = str(ranked.iloc[0]["candidate_id"])
        candidate_lookup = {
            candidate.candidate_id: candidate for candidate in candidates
        }
        top_candidates[instrument_id] = candidate_lookup[candidate_id]

    joint = pd.DataFrame()
    sizing_window_results = pd.DataFrame()
    sizing_comparison_summary = pd.DataFrame()
    if not args.skip_joint_diagnostic:
        comparison = config["sizing_comparison_period"]
        comparison_start = _utc(comparison["start"], "sizing comparison start")
        comparison_end = _utc(comparison["end"], "sizing comparison end")
        if comparison_end > official_test_start:
            raise ValueError("Sizing comparison enters official test data")
        reference_asset = str(comparison.get("session_reference_asset", "spy_usd"))
        comparison_windows = build_session_windows(
            canonical,
            reference_asset,
            comparison_start,
            comparison_end,
            sessions_per_window,
            window_step_sessions,
        )
        sizing_rows: list[pd.DataFrame] = []
        for sizing_method in comparison_methods:
            print(f"\nRolling sizing comparison: {sizing_method}")
            for window_index, window in enumerate(comparison_windows, start=1):
                print(
                    f"  sizing window {window_index}/{len(comparison_windows)}: "
                    f"{window.window_id}"
                )
                result = run_joint_diagnostics(
                    canonical,
                    [top_candidates["spy_usd"]],
                    [top_candidates["sti_es3_sgd"]],
                    selected,
                    fx_rates,
                    sizing,
                    costs,
                    window.start,
                    window.end,
                    PROJECT_ROOT,
                    sizing_method=sizing_method,
                    slippage_bps=args.slippage_bps,
                )
                result.insert(1, "window_id", window.window_id)
                result.insert(2, "session_count", window.session_count)
                sizing_rows.append(result)
        sizing_window_results = pd.concat(sizing_rows, ignore_index=True)
        sizing_window_results.to_csv(
            args.output_dir / "sizing_method_comparison_windows.csv",
            index=False,
        )
        sizing_comparison_summary = (
            sizing_window_results.groupby("sizing_method", as_index=False)
            .agg(
                spy_candidate_id=("spy_candidate_id", "first"),
                sti_candidate_id=("sti_candidate_id", "first"),
                window_count=("window_id", "count"),
                profitable_window_rate=(
                    "net_return",
                    lambda values: float((values > 0).mean()),
                ),
                median_net_return=("net_return", "median"),
                worst_net_return=("net_return", "min"),
                median_max_drawdown=("max_drawdown", "median"),
                median_transaction_cost_usd=("transaction_cost_usd", "median"),
                total_transaction_cost_usd=("transaction_cost_usd", "sum"),
            )
            .sort_values(
                ["median_net_return", "profitable_window_rate", "worst_net_return"],
                ascending=[False, False, False],
                kind="stable",
            )
            .reset_index(drop=True)
        )
        sizing_comparison_summary["comparison_rank"] = (
            sizing_comparison_summary.index + 1
        )
        sizing_comparison_summary.to_csv(
            args.output_dir / "sizing_method_comparison_summary.csv",
            index=False,
        )

        diagnostic = config["joint_diagnostic_period"]
        diagnostic_start = _utc(diagnostic["start"], "joint diagnostic start")
        diagnostic_end = _utc(diagnostic["end"], "joint diagnostic end")
        if diagnostic_end > official_test_start:
            raise ValueError("Joint diagnostic enters official test data")
        joint_results: list[pd.DataFrame] = []
        for sizing_method in comparison_methods:
            print(f"\nJoint diagnostic sizing method: {sizing_method}")
            joint_results.append(
                run_joint_diagnostics(
                    canonical,
                    reviewed_candidates["spy_usd"],
                    reviewed_candidates["sti_es3_sgd"],
                    selected,
                    fx_rates,
                    sizing,
                    costs,
                    diagnostic_start,
                    diagnostic_end,
                    PROJECT_ROOT,
                    sizing_method=sizing_method,
                    slippage_bps=args.slippage_bps,
                )
            )
        joint = pd.concat(joint_results, ignore_index=True).sort_values(
            ["net_return", "max_drawdown", "transaction_cost_usd"],
            ascending=[False, False, True],
            kind="stable",
        ).reset_index(drop=True)
        joint.to_csv(args.output_dir / "joint_diagnostic_summary.csv", index=False)

    manifest = {
        "generated_at_utc": pd.Timestamp.now(tz="UTC").isoformat(),
        "input_file": str(args.input_file),
        "fx_file": str(args.fx_file),
        "config_file": str(args.config_file),
        "selected_strategies_file": str(args.selected_strategies),
        "sizing_config": str(args.sizing_config),
        "cost_config": str(args.cost_config),
        "official_test_start_utc": official_test_start.isoformat(),
        "official_test_data_used": False,
        "selected_strategies_overwritten": False,
        "sessions_per_evaluation_window": sessions_per_window,
        "window_step_sessions": window_step_sessions,
        "minimum_trade_window_rate": minimum_trade_window_rate,
        "candidate_counts": {
            asset: len(candidates)
            for asset, candidates in candidates_by_asset.items()
        },
        "shortlist_size_requested": shortlist_size,
        "joint_shortlist_size_requested": joint_size,
        "slippage_bps": args.slippage_bps,
        "screening_sizing": "constant_common_maximum_asset_weight",
        "joint_sizing_methods": comparison_methods,
        "sizing_comparison_strategy_pair": {
            asset: candidate.candidate_id
            for asset, candidate in top_candidates.items()
        },
        "sizing_comparison_period": config.get("sizing_comparison_period"),
        "sizing_comparison_window_count": int(
            sizing_window_results["window_id"].nunique()
            if not sizing_window_results.empty
            else 0
        ),
        "execution": "next_underlying_30m_bar_raw_open",
        "valuation": "raw_close_converted_point_in_time_to_usd",
        "features": "adjusted_close_at_configured_decision_frequency",
        "joint_validation_period_role": "pre_cutoff_pair_selection",
        "configuration_snapshot": config,
    }
    (args.output_dir / "optimization_manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )

    print("\nShortlisted independent-window review")
    display_columns = [
        "instrument_id",
        "review_rank",
        "candidate_id",
        "selection_eligible",
        "decision_interval_minutes",
        "independent_trade_window_rate",
        "independent_profitable_window_rate",
        "independent_median_net_return",
        "independent_worst_net_return",
        "independent_total_cost_usd",
    ]
    print(
        exact_summary[display_columns].to_string(
            index=False, float_format=lambda value: f"{value:,.4f}"
        )
    )
    if not joint.empty:
        print("\nConfigured pre-cutoff joint validation")
        joint_columns = [
            "sizing_method",
            "spy_candidate_id",
            "sti_candidate_id",
            "net_return",
            "max_drawdown",
            "completed_round_trips",
            "transaction_cost_usd",
        ]
        print(
            joint[joint_columns].to_string(
                index=False, float_format=lambda value: f"{value:,.4f}"
            )
        )
    if not sizing_comparison_summary.empty:
        print("\nRolling 14-session sizing-method comparison")
        print(
            sizing_comparison_summary.to_string(
                index=False, float_format=lambda value: f"{value:,.4f}"
            )
        )
    print(f"\nReview outputs -> {args.output_dir}")
    print("No selected strategy configuration was changed.")


if __name__ == "__main__":
    main()

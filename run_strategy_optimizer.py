"""Optimise SPY momentum and STI mean-reversion on training folds only."""

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

from src.strategy_backtest import (  # noqa: E402
    load_canonical_bars,
    resolve_instrument_id,
)
from src.strategy_optimizer import (  # noqa: E402
    FoldDefinition,
    evaluate_candidate,
    evaluate_search_space,
    generate_candidates,
    rank_candidates,
    select_best_candidate,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input-file",
        type=Path,
        default=Path("data/canonical/market_bars_30m_training_validation.csv"),
        help="Canonical 30-minute CSV used by the strategy backtester.",
    )
    parser.add_argument(
        "--config-file",
        type=Path,
        default=Path("config/strategy_optimization.json"),
        help="Tracked search-space, fold, and selection configuration.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("results/strategy_optimizer"),
        help="Ignored directory for fold results, ranking, and selection files.",
    )
    parser.add_argument(
        "--instrument",
        nargs="+",
        default=["SPY", "ES3.SI"],
        help="SPY and/or ES3.SI; canonical instrument IDs are also accepted.",
    )
    parser.add_argument("--initial-cash", type=float, default=100_000.0)
    parser.add_argument("--slippage-bps", type=float, default=0.0)
    parser.add_argument(
        "--flat-fee-bps",
        type=float,
        help="Override project commissions with one flat fee in basis points.",
    )
    parser.add_argument(
        "--validate-selected",
        action="store_true",
        help=(
            "After training-only selection, evaluate each winner once on its "
            "configured validation period. Validation never affects ranking."
        ),
    )
    parser.add_argument(
        "--max-candidates",
        type=int,
        help="Evaluate only the first N candidates per asset for a smoke run.",
    )
    parser.add_argument("--top", type=int, default=10)
    parser.add_argument("--progress-every", type=int, default=10)
    return parser


def _read_config(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(f"Optimisation configuration not found: {path}")
    try:
        config = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as error:
        raise ValueError(f"Invalid JSON in {path}: {error}") from error
    if not isinstance(config.get("assets"), dict) or not config["assets"]:
        raise ValueError("Optimisation configuration must contain an assets object")
    return config


def _timestamp(value: Any, label: str) -> pd.Timestamp:
    try:
        timestamp = pd.Timestamp(value)
    except (TypeError, ValueError) as error:
        raise ValueError(f"{label} must be a valid timestamp") from error
    if timestamp.tzinfo is None:
        timestamp = timestamp.tz_localize("UTC")
    else:
        timestamp = timestamp.tz_convert("UTC")
    return timestamp


def _folds_from_config(
    instrument_id: str,
    asset_config: dict[str, Any],
    official_test_start: pd.Timestamp,
) -> list[FoldDefinition]:
    training = asset_config.get(
        "training_period", asset_config.get("development_period", {})
    )
    training_start = _timestamp(
        training.get("start"), f"{instrument_id} training start"
    )
    training_end = _timestamp(
        training.get("end"), f"{instrument_id} training end"
    )
    if training_end > official_test_start:
        raise ValueError(f"{instrument_id} training period enters official test data")

    raw_folds = asset_config.get(
        "training_folds", asset_config.get("development_folds")
    )
    if not isinstance(raw_folds, list) or not raw_folds:
        raise ValueError(f"{instrument_id} must define training_folds")

    folds: list[FoldDefinition] = []
    previous_end: pd.Timestamp | None = None
    seen_ids: set[str] = set()
    for index, raw in enumerate(raw_folds, start=1):
        fold_id = str(raw.get("fold_id", f"fold_{index}"))
        if fold_id in seen_ids:
            raise ValueError(f"Duplicate fold_id {fold_id!r} for {instrument_id}")
        seen_ids.add(fold_id)
        fold = FoldDefinition(
            fold_id=fold_id,
            start=_timestamp(raw.get("start"), f"{fold_id} start"),
            end=_timestamp(raw.get("end"), f"{fold_id} end"),
            phase="training",
        )
        fold.validate()
        if fold.start < training_start or fold.end > training_end:
            raise ValueError(
                f"Fold {fold_id!r} lies outside the {instrument_id} "
                "training period"
            )
        if previous_end is not None and fold.start < previous_end:
            raise ValueError(f"Development folds overlap at {fold_id!r}")
        previous_end = fold.end
        folds.append(fold)
    return folds


def _validation_fold(
    instrument_id: str,
    asset_config: dict[str, Any],
    official_test_start: pd.Timestamp,
) -> FoldDefinition:
    validation = asset_config.get("validation_period", {})
    fold = FoldDefinition(
        fold_id=f"{instrument_id}_validation",
        start=_timestamp(validation.get("start"), f"{instrument_id} validation start"),
        end=_timestamp(validation.get("end"), f"{instrument_id} validation end"),
        phase="validation",
    )
    fold.validate()
    training_period = asset_config.get(
        "training_period", asset_config.get("development_period", {})
    )
    training_end = _timestamp(
        training_period.get("end"),
        f"{instrument_id} training end",
    )
    if fold.start < training_end:
        raise ValueError(f"{instrument_id} validation overlaps training")
    if fold.end > official_test_start:
        raise ValueError(f"{instrument_id} validation enters official test data")
    return fold


def _selected_record(candidate: Any, ranking: pd.DataFrame) -> dict[str, Any]:
    if candidate is None:
        return {
            "status": "no_reliable_candidate",
            "candidate_id": None,
            "strategy": None,
            "parameters": None,
            "training_robustness_score": None,
        }
    row = ranking.loc[ranking["candidate_id"].eq(candidate.candidate_id)].iloc[0]
    return {
        "status": "selected",
        "candidate_id": candidate.candidate_id,
        "strategy": candidate.strategy_name,
        "parameters": candidate.parameters,
        "training_robustness_score": float(row["robustness_score"]),
    }


def _display_ranking(ranking: pd.DataFrame, top: int) -> None:
    columns = [
        "instrument_id",
        "rank",
        "candidate_id",
        "eligible",
        "robustness_score",
        "median_fold_return",
        "median_fold_sharpe",
        "worst_fold_drawdown",
        "total_trades",
        "total_transaction_cost",
        "rejection_reasons",
    ]
    print(
        ranking[columns]
        .head(top)
        .to_string(index=False, float_format=lambda value: f"{value:,.4f}")
    )


def main() -> None:
    args = build_parser().parse_args()
    if args.initial_cash <= 0:
        raise ValueError("--initial-cash must be positive")
    if args.slippage_bps < 0:
        raise ValueError("--slippage-bps must not be negative")
    if args.flat_fee_bps is not None and args.flat_fee_bps < 0:
        raise ValueError("--flat-fee-bps must not be negative")
    if args.max_candidates is not None and args.max_candidates < 1:
        raise ValueError("--max-candidates must be at least 1")
    if args.max_candidates is not None and args.validate_selected:
        raise ValueError(
            "--validate-selected cannot be combined with --max-candidates; "
            "complete the full training search before validation"
        )
    if args.top < 1 or args.progress_every < 1:
        raise ValueError("--top and --progress-every must be at least 1")

    config = _read_config(args.config_file)
    canonical = load_canonical_bars(args.input_file)
    official_test_start = _timestamp(
        config.get("official_test_start_utc"), "official_test_start_utc"
    )
    score_weights = config.get("score_weights", {})
    instruments = list(
        dict.fromkeys(resolve_instrument_id(value) for value in args.instrument)
    )

    all_fold_results: list[pd.DataFrame] = []
    all_rankings: list[pd.DataFrame] = []
    validation_rows: list[dict[str, Any]] = []
    selected: dict[str, dict[str, Any]] = {}
    candidate_counts: dict[str, dict[str, int]] = {}

    if args.max_candidates is not None:
        print(
            "SMOKE RUN ONLY: the candidate grid is truncated, so displayed "
            "winners are not final selections."
        )

    for instrument_id in instruments:
        if instrument_id not in config["assets"]:
            raise ValueError(f"No optimisation configuration for {instrument_id}")
        asset_config = config["assets"][instrument_id]
        strategy_name = str(asset_config.get("strategy", ""))
        candidates = generate_candidates(
            instrument_id,
            strategy_name,
            asset_config.get("search_space", {}),
        )
        generated_count = len(candidates)
        if args.max_candidates is not None:
            candidates = candidates[: args.max_candidates]
        candidate_counts[instrument_id] = {
            "generated": generated_count,
            "evaluated": len(candidates),
        }
        folds = _folds_from_config(
            instrument_id, asset_config, official_test_start
        )

        print(
            f"\nOptimising {instrument_id} / {strategy_name}: "
            f"{len(candidates):,} candidates x {len(folds)} folds"
        )
        fold_results = evaluate_search_space(
            canonical,
            candidates,
            folds,
            initial_cash=args.initial_cash,
            slippage_bps=args.slippage_bps,
            flat_fee_bps=args.flat_fee_bps,
            progress_every=args.progress_every,
        )
        ranking = rank_candidates(
            fold_results,
            candidates,
            expected_fold_count=len(folds),
            constraints=asset_config.get("selection_constraints", {}),
            score_weights=score_weights,
        )
        winner = select_best_candidate(ranking, candidates)
        selected[instrument_id] = _selected_record(winner, ranking)
        if args.max_candidates is not None and winner is not None:
            selected[instrument_id]["status"] = "smoke_run_only"
        all_fold_results.append(fold_results)
        all_rankings.append(ranking)

        print(f"\nTop {instrument_id} training candidates")
        _display_ranking(ranking, args.top)
        if winner is None:
            print(f"No reliable candidate selected for {instrument_id}.")
        else:
            print(
                f"Selected {winner.candidate_id}: "
                f"{json.dumps(winner.parameters, sort_keys=True)}"
            )

        if args.validate_selected and winner is not None:
            validation_fold = _validation_fold(
                instrument_id, asset_config, official_test_start
            )
            validation_rows.extend(
                evaluate_candidate(
                    canonical,
                    winner,
                    [validation_fold],
                    initial_cash=args.initial_cash,
                    slippage_bps=args.slippage_bps,
                    flat_fee_bps=args.flat_fee_bps,
                )
            )

    fold_results_output = pd.concat(all_fold_results, ignore_index=True)
    ranking_output = pd.concat(all_rankings, ignore_index=True)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    fold_results_output.to_csv(args.output_dir / "fold_results.csv", index=False)
    ranking_output.to_csv(args.output_dir / "candidate_ranking.csv", index=False)

    generated_at = pd.Timestamp.now(tz="UTC").isoformat()
    selection_output = {
        "generated_at_utc": generated_at,
        "selection_basis": "training_folds_only",
        "search_complete": args.max_candidates is None,
        "candidate_limit_applied": args.max_candidates,
        "validation_used_for_ranking": False,
        "official_test_data_used": False,
        "assets": selected,
    }
    (args.output_dir / "selected_parameters.json").write_text(
        json.dumps(selection_output, indent=2) + "\n",
        encoding="utf-8",
    )

    validation_path = args.output_dir / "validation_summary.csv"
    if validation_rows:
        validation = pd.DataFrame(validation_rows)
        validation.to_csv(validation_path, index=False)
        display_columns = [
            "instrument_id",
            "candidate_id",
            "status",
            "Total Return",
            "Sharpe Ratio",
            "Max Drawdown",
            "Number of Trades",
            "Total Transaction Cost",
        ]
        print("\nUntouched validation results (not used for ranking)")
        print(
            validation.reindex(columns=display_columns).to_string(
                index=False, float_format=lambda value: f"{value:,.4f}"
            )
        )
    elif validation_path.exists():
        validation_path.unlink()

    manifest = {
        "generated_at_utc": generated_at,
        "input_file": str(args.input_file),
        "config_file": str(args.config_file),
        "instruments": instruments,
        "candidate_counts": candidate_counts,
        "candidate_limit_applied": args.max_candidates,
        "initial_cash_per_run": args.initial_cash,
        "slippage_bps": args.slippage_bps,
        "cost_model": (
            "flat_bps" if args.flat_fee_bps is not None else "project_assumptions"
        ),
        "flat_fee_bps": args.flat_fee_bps,
        "validation_evaluated": bool(args.validate_selected),
        "validation_used_for_ranking": False,
        "official_test_start_utc": official_test_start.isoformat(),
        "official_test_data_used": False,
        "configuration_snapshot": config,
    }
    (args.output_dir / "optimization_manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n",
        encoding="utf-8",
    )
    print(f"\nOutputs -> {args.output_dir}")


if __name__ == "__main__":
    main()

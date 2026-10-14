"""Reliability filters, robust scoring, and candidate selection."""

from __future__ import annotations

import json
from typing import Any, Iterable

import numpy as np
import pandas as pd

from .search_space import OptimizationCandidate


REQUIRED_METRICS = (
    "Total Return",
    "Sharpe Ratio",
    "Max Drawdown",
    "Number of Trades",
    "Total Transaction Cost",
    "Annual Turnover (x equity)",
)


def _finite(value: Any) -> bool:
    try:
        return bool(np.isfinite(float(value)))
    except (TypeError, ValueError):
        return False


def _candidate_summary(
    candidate: OptimizationCandidate,
    rows: pd.DataFrame,
    expected_fold_count: int,
    constraints: dict[str, Any],
    score_weights: dict[str, Any],
) -> dict[str, Any]:
    completed = rows.loc[rows["status"].eq("completed")].copy()
    successful_fold_count = int(completed["fold_id"].nunique())
    failed_fold_count = expected_fold_count - successful_fold_count
    reasons: list[str] = []

    if failed_fold_count:
        reasons.append(f"{failed_fold_count}_failed_or_missing_folds")

    missing_metrics = [
        column for column in REQUIRED_METRICS if column not in completed.columns
    ]
    if missing_metrics:
        reasons.append("missing_required_metrics")

    if completed.empty or missing_metrics:
        median_return = np.nan
        minimum_return = np.nan
        median_sharpe = np.nan
        sharpe_std = np.nan
        worst_drawdown = np.nan
        total_trades = 0
        minimum_fold_trades = 0
        transaction_cost = np.nan
        median_turnover = np.nan
    else:
        median_return = float(completed["Total Return"].median())
        minimum_return = float(completed["Total Return"].min())
        median_sharpe = float(completed["Sharpe Ratio"].median())
        sharpe_std = float(completed["Sharpe Ratio"].std(ddof=0))
        worst_drawdown = float(completed["Max Drawdown"].min())
        total_trades = int(completed["Number of Trades"].sum())
        minimum_fold_trades = int(completed["Number of Trades"].min())
        transaction_cost = float(completed["Total Transaction Cost"].sum())
        median_turnover = float(completed["Annual Turnover (x equity)"].median())

    if not _finite(median_sharpe) or not _finite(sharpe_std):
        reasons.append("non_finite_sharpe")
    if not _finite(worst_drawdown):
        reasons.append("non_finite_drawdown")

    minimum_total_trades = int(constraints.get("minimum_total_trades", 0))
    if total_trades < minimum_total_trades:
        reasons.append("insufficient_total_trades")

    required_fold_trades = int(constraints.get("minimum_trades_per_fold", 0))
    if minimum_fold_trades < required_fold_trades:
        reasons.append("insufficient_trades_in_a_fold")

    drawdown_limit = float(constraints.get("maximum_absolute_drawdown", 1.0))
    if _finite(worst_drawdown) and abs(worst_drawdown) > drawdown_limit:
        reasons.append("drawdown_limit_exceeded")

    if bool(constraints.get("require_positive_median_return", False)):
        if not _finite(median_return) or median_return <= 0:
            reasons.append("non_positive_median_return")

    eligible = not reasons
    if eligible:
        score = (
            median_sharpe
            - float(score_weights.get("sharpe_stability_penalty", 0.5))
            * sharpe_std
            - float(score_weights.get("drawdown_penalty", 2.0))
            * abs(worst_drawdown)
        )
    else:
        score = np.nan

    return {
        "instrument_id": candidate.instrument_id,
        "strategy": candidate.strategy_name,
        "candidate_id": candidate.candidate_id,
        "parameters": json.dumps(candidate.parameters, sort_keys=True),
        "expected_fold_count": expected_fold_count,
        "successful_fold_count": successful_fold_count,
        "failed_fold_count": failed_fold_count,
        "total_trades": total_trades,
        "minimum_fold_trades": minimum_fold_trades,
        "median_fold_return": median_return,
        "worst_fold_return": minimum_return,
        "median_fold_sharpe": median_sharpe,
        "fold_sharpe_std": sharpe_std,
        "worst_fold_drawdown": worst_drawdown,
        "total_transaction_cost": transaction_cost,
        "median_annual_turnover": median_turnover,
        "eligible": eligible,
        "rejection_reasons": ";".join(reasons),
        "robustness_score": score,
    }


def rank_candidates(
    fold_results: pd.DataFrame,
    candidates: Iterable[OptimizationCandidate],
    *,
    expected_fold_count: int,
    constraints: dict[str, Any],
    score_weights: dict[str, Any],
) -> pd.DataFrame:
    """Summarize and rank candidates without using validation observations."""
    candidate_list = list(candidates)
    summaries: list[dict[str, Any]] = []
    for candidate in candidate_list:
        rows = fold_results.loc[
            fold_results["candidate_id"].eq(candidate.candidate_id)
        ]
        summaries.append(
            _candidate_summary(
                candidate,
                rows,
                expected_fold_count,
                constraints,
                score_weights,
            )
        )

    ranking = pd.DataFrame(summaries)
    ranking = ranking.sort_values(
        by=[
            "eligible",
            "robustness_score",
            "worst_fold_drawdown",
            "total_transaction_cost",
            "median_annual_turnover",
        ],
        ascending=[False, False, False, True, True],
        na_position="last",
        kind="stable",
    ).reset_index(drop=True)
    ranking["rank"] = pd.Series(pd.NA, index=ranking.index, dtype="Int64")
    eligible_index = ranking.index[ranking["eligible"]]
    ranking.loc[eligible_index, "rank"] = range(1, len(eligible_index) + 1)
    return ranking


def select_best_candidate(
    ranking: pd.DataFrame,
    candidates: Iterable[OptimizationCandidate],
) -> OptimizationCandidate | None:
    """Return the highest-ranked eligible candidate, or None."""
    eligible = ranking.loc[ranking["eligible"]]
    if eligible.empty:
        return None
    selected_id = str(eligible.iloc[0]["candidate_id"])
    return next(
        candidate
        for candidate in candidates
        if candidate.candidate_id == selected_id
    )

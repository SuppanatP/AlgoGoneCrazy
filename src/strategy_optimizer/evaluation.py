"""Evaluate parameter candidates on chronological folds."""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any, Iterable

import pandas as pd

from src.strategy_backtest import run_strategy_backtest
from trading_strategy import MeanReversionParameters, MomentumParameters

from .search_space import OptimizationCandidate


@dataclass(frozen=True)
class FoldDefinition:
    """One inclusive-start, exclusive-end chronological evaluation period."""

    fold_id: str
    start: pd.Timestamp
    end: pd.Timestamp
    phase: str = "development"

    def validate(self) -> None:
        if self.start.tzinfo is None or self.end.tzinfo is None:
            raise ValueError(f"Fold {self.fold_id!r} timestamps must be timezone-aware")
        if self.end <= self.start:
            raise ValueError(f"Fold {self.fold_id!r} end must be later than start")


def _strategy_parameters(
    candidate: OptimizationCandidate,
) -> tuple[MomentumParameters, MeanReversionParameters]:
    momentum = MomentumParameters()
    mean_reversion = MeanReversionParameters()
    if candidate.strategy_name == "momentum":
        momentum = MomentumParameters(**candidate.parameters)
    elif candidate.strategy_name == "mean_reversion":
        mean_reversion = MeanReversionParameters(**candidate.parameters)
    else:
        raise ValueError(f"Unsupported strategy: {candidate.strategy_name}")
    return momentum, mean_reversion


def _base_row(
    candidate: OptimizationCandidate,
    fold: FoldDefinition,
) -> dict[str, Any]:
    return {
        "instrument_id": candidate.instrument_id,
        "strategy": candidate.strategy_name,
        "candidate_id": candidate.candidate_id,
        "parameters": json.dumps(candidate.parameters, sort_keys=True),
        "phase": fold.phase,
        "fold_id": fold.fold_id,
        "configured_start_utc": fold.start.isoformat(),
        "configured_end_utc": fold.end.isoformat(),
    }


def evaluate_candidate(
    canonical_bars: pd.DataFrame,
    candidate: OptimizationCandidate,
    folds: Iterable[FoldDefinition],
    *,
    initial_cash: float,
    slippage_bps: float,
    flat_fee_bps: float | None,
) -> list[dict[str, Any]]:
    """Backtest one candidate independently on every supplied fold."""
    momentum, mean_reversion = _strategy_parameters(candidate)
    rows: list[dict[str, Any]] = []

    for fold in folds:
        fold.validate()
        row = _base_row(candidate, fold)
        try:
            run = run_strategy_backtest(
                canonical_bars,
                candidate.instrument_id,
                candidate.strategy_name,
                momentum,
                mean_reversion,
                start=fold.start,
                end=fold.end,
                initial_cash=initial_cash,
                slippage_bps=slippage_bps,
                flat_fee_bps=flat_fee_bps,
            )
            row.update(
                {
                    "status": "completed",
                    "error": "",
                    "actual_start_utc": run.signals[
                        "bar_open_time_utc"
                    ].min(),
                    "actual_end_utc": run.signals[
                        "bar_open_time_utc"
                    ].max(),
                    "bar_count": len(run.signals),
                    **run.strategy_metrics.to_dict(),
                }
            )
        except (ValueError, KeyError, TypeError) as error:
            row.update(
                {
                    "status": "failed",
                    "error": str(error),
                    "actual_start_utc": pd.NaT,
                    "actual_end_utc": pd.NaT,
                    "bar_count": 0,
                }
            )
        rows.append(row)
    return rows


def evaluate_search_space(
    canonical_bars: pd.DataFrame,
    candidates: Iterable[OptimizationCandidate],
    folds: Iterable[FoldDefinition],
    *,
    initial_cash: float,
    slippage_bps: float,
    flat_fee_bps: float | None,
    progress_every: int = 10,
) -> pd.DataFrame:
    """Evaluate all candidates and return one row per candidate and fold."""
    candidate_list = list(candidates)
    fold_list = list(folds)
    rows: list[dict[str, Any]] = []

    for index, candidate in enumerate(candidate_list, start=1):
        if index == 1 or index % progress_every == 0 or index == len(candidate_list):
            print(
                f"  {candidate.instrument_id}: candidate {index:,}/"
                f"{len(candidate_list):,} ({candidate.candidate_id})"
            )
        rows.extend(
            evaluate_candidate(
                canonical_bars,
                candidate,
                fold_list,
                initial_cash=initial_cash,
                slippage_bps=slippage_bps,
                flat_fee_bps=flat_fee_bps,
            )
        )
    return pd.DataFrame(rows)

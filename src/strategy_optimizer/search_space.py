"""Create deterministic, validated strategy parameter grids."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from itertools import product
from typing import Any, Iterable

from trading_strategy import MeanReversionParameters, MomentumParameters


@dataclass(frozen=True)
class OptimizationCandidate:
    """One strategy parameter combination for one instrument."""

    candidate_id: str
    instrument_id: str
    strategy_name: str
    parameters: dict[str, int | float | None]


MOMENTUM_FIELDS = (
    "fast_window",
    "slow_window",
    "momentum_lookback",
    "entry_threshold",
    "exit_threshold",
)

MEAN_REVERSION_FIELDS = (
    "window",
    "entry_z_score",
    "exit_z_score",
    "maximum_holding_bars",
)


def _values(search_space: dict[str, Any], fields: Iterable[str]) -> list[list[Any]]:
    values: list[list[Any]] = []
    for field in fields:
        if field not in search_space:
            raise ValueError(f"Search space is missing {field!r}")
        field_values = search_space[field]
        if not isinstance(field_values, list) or not field_values:
            raise ValueError(f"Search-space field {field!r} must be a non-empty list")
        values.append(field_values)
    return values


def _momentum_candidates(
    instrument_id: str,
    search_space: dict[str, Any],
) -> list[OptimizationCandidate]:
    candidates: list[OptimizationCandidate] = []
    combinations = product(*_values(search_space, MOMENTUM_FIELDS))
    for combination in combinations:
        raw = dict(zip(MOMENTUM_FIELDS, combination, strict=True))
        parameters = MomentumParameters(
            fast_window=int(raw["fast_window"]),
            slow_window=int(raw["slow_window"]),
            momentum_lookback=int(raw["momentum_lookback"]),
            entry_threshold=float(raw["entry_threshold"]),
            exit_threshold=float(raw["exit_threshold"]),
        )
        try:
            parameters.validate()
        except ValueError:
            continue
        candidates.append(
            OptimizationCandidate(
                candidate_id=f"momentum_{len(candidates) + 1:04d}",
                instrument_id=instrument_id,
                strategy_name="momentum",
                parameters=asdict(parameters),
            )
        )
    return candidates


def _mean_reversion_candidates(
    instrument_id: str,
    search_space: dict[str, Any],
) -> list[OptimizationCandidate]:
    candidates: list[OptimizationCandidate] = []
    combinations = product(*_values(search_space, MEAN_REVERSION_FIELDS))
    for combination in combinations:
        raw = dict(zip(MEAN_REVERSION_FIELDS, combination, strict=True))
        maximum_holding = raw["maximum_holding_bars"]
        parameters = MeanReversionParameters(
            window=int(raw["window"]),
            entry_z_score=float(raw["entry_z_score"]),
            exit_z_score=float(raw["exit_z_score"]),
            maximum_holding_bars=(
                None if maximum_holding is None else int(maximum_holding)
            ),
        )
        try:
            parameters.validate()
        except ValueError:
            continue
        candidates.append(
            OptimizationCandidate(
                candidate_id=f"mean_reversion_{len(candidates) + 1:04d}",
                instrument_id=instrument_id,
                strategy_name="mean_reversion",
                parameters=asdict(parameters),
            )
        )
    return candidates


def generate_candidates(
    instrument_id: str,
    strategy_name: str,
    search_space: dict[str, Any],
) -> list[OptimizationCandidate]:
    """Return every valid parameter combination in deterministic order."""
    if strategy_name == "momentum":
        candidates = _momentum_candidates(instrument_id, search_space)
    elif strategy_name == "mean_reversion":
        candidates = _mean_reversion_candidates(instrument_id, search_space)
    else:
        raise ValueError(f"Unsupported strategy for optimisation: {strategy_name}")

    if not candidates:
        raise ValueError(
            f"Search space generated no valid {strategy_name} candidates for "
            f"{instrument_id}"
        )
    return candidates

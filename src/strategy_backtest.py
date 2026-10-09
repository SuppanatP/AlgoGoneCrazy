"""Historical backtesting adapter for the rule-based strategy package.

The strategy modules calculate their target after a 30-minute bar closes. This
adapter carries that target forward and permits a position change only at the
next complete, tradable bar's raw opening price.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import pandas as pd

from market_data.costs import CostAssumptions, load_cost_assumptions
from trading_strategy import (
    MeanReversionParameters,
    MomentumParameters,
    generate_mean_reversion_signals,
    generate_momentum_signals,
)

from .backtest import BacktestResult, buy_and_hold, run_backtest
from .metrics import compute_metrics
from .tracker import CostModel


INSTRUMENT_ALIASES = {
    "spy": "spy_usd",
    "spy_usd": "spy_usd",
    "es3": "sti_es3_sgd",
    "es3.si": "sti_es3_sgd",
    "sg.es3": "sti_es3_sgd",
    "sti_es3_sgd": "sti_es3_sgd",
}

PROJECT_COST_ASSET_IDS = {
    "spy_usd": "SP500_ETF",
    "sti_es3_sgd": "STI_ETF",
}

EQUITY_TRADING_DAYS_PER_YEAR = 252


class ProjectCostModel:
    """Adapt the project's commission assumptions to the trade tracker API."""

    def __init__(
        self,
        assumptions: CostAssumptions,
        slippage_bps: float = 0.0,
    ) -> None:
        if slippage_bps < 0:
            raise ValueError("slippage_bps must not be negative")
        self.assumptions = assumptions
        self.slippage_bps = float(slippage_bps)

    def apply(
        self,
        asset: str,
        qty: float,
        ref_price: float,
    ) -> tuple[float, float, float]:
        cost_asset_id = PROJECT_COST_ASSET_IDS.get(asset)
        if cost_asset_id is None:
            raise ValueError(f"No project cost mapping is configured for {asset}")

        direction = 1.0 if qty > 0 else -1.0
        execution_price = ref_price * (
            1.0 + direction * self.slippage_bps / 10_000.0
        )
        slippage_cost = abs(qty) * abs(execution_price - ref_price)
        notional = abs(qty) * execution_price
        fee = self.assumptions.assets[cost_asset_id].commission(
            notional,
            shares=abs(qty),
        )
        return execution_price, float(fee), float(slippage_cost)


@dataclass
class StrategyBacktestRun:
    """All artifacts from one instrument and strategy combination."""

    instrument_id: str
    source_symbol: str
    quote_currency: str
    strategy_name: str
    parameters: dict[str, Any]
    periods_per_year: int
    signals: pd.DataFrame
    result: BacktestResult
    benchmark: BacktestResult
    strategy_metrics: pd.Series
    benchmark_metrics: pd.Series
    initial_cash: float


def resolve_instrument_id(value: str) -> str:
    """Resolve a user-facing symbol or canonical identifier."""
    normalized = value.strip().lower()
    try:
        return INSTRUMENT_ALIASES[normalized]
    except KeyError as error:
        supported = "SPY, ES3.SI, spy_usd, sti_es3_sgd"
        raise ValueError(
            f"Unsupported instrument {value!r}; choose one of: {supported}"
        ) from error


def load_canonical_bars(path: Path) -> pd.DataFrame:
    """Load canonical bars and validate fields needed by strategy and execution."""
    if not path.is_file():
        raise FileNotFoundError(
            f"Canonical market-data file not found: {path}. "
            "Run the market-data pipeline or pass --input-file."
        )

    bars = pd.read_csv(path, low_memory=False)
    required = {
        "instrument_id",
        "source_symbol",
        "quote_currency",
        "bar_open_time_utc",
        "open",
        "close",
        "adjusted_close",
    }
    missing = required - set(bars.columns)
    if missing:
        raise ValueError(
            f"Canonical market data is missing columns: {sorted(missing)}"
        )

    bars["bar_open_time_utc"] = pd.to_datetime(
        bars["bar_open_time_utc"], errors="coerce", utc=True
    )
    for column in ("open", "close", "adjusted_close"):
        bars[column] = pd.to_numeric(bars[column], errors="coerce")

    invalid = (
        bars["bar_open_time_utc"].isna()
        | bars[["open", "close", "adjusted_close"]].isna().any(axis=1)
        | bars[["open", "close", "adjusted_close"]].le(0).any(axis=1)
    )
    if invalid.any():
        raise ValueError(
            f"Canonical market data contains {int(invalid.sum())} rows with "
            "invalid timestamps or execution prices"
        )

    duplicate_keys = ["instrument_id", "bar_open_time_utc"]
    if bars.duplicated(subset=duplicate_keys).any():
        raise ValueError(
            "Canonical market data contains duplicate instrument/timestamp rows"
        )
    return bars.sort_values(duplicate_keys).reset_index(drop=True)


def parse_utc_timestamp(value: str | None, label: str) -> pd.Timestamp | None:
    if value is None:
        return None
    try:
        timestamp = pd.Timestamp(value)
    except (TypeError, ValueError) as error:
        raise ValueError(f"{label} must be a valid timestamp") from error
    if timestamp.tzinfo is None:
        return timestamp.tz_localize("UTC")
    return timestamp.tz_convert("UTC")


def delayed_execution_targets(signals: pd.DataFrame) -> pd.Series:
    """Delay desired positions and hold pending changes until a real bar opens."""
    desired_at_open = signals["target_position"].shift(1)
    current_position = 0
    execution_targets: list[int] = []

    for eligible, desired in zip(
        signals["signal_eligible"],
        desired_at_open,
        strict=True,
    ):
        if bool(eligible) and pd.notna(desired):
            current_position = int(desired)
        execution_targets.append(current_position)

    return pd.Series(
        execution_targets,
        index=signals.index,
        dtype="int8",
        name="execution_target_position",
    )


def infer_periods_per_year(bars: pd.DataFrame) -> int:
    """Infer equity 30-minute periods from the median local session length."""
    if "session_date_local" in bars.columns:
        session = bars["session_date_local"].astype("string")
    else:
        session = bars["bar_open_time_utc"].dt.date.astype("string")
    bars_per_session = bars.assign(_session=session).groupby("_session").size()
    if bars_per_session.empty:
        raise ValueError("Cannot annualize an empty backtest")
    typical_session = int(round(float(bars_per_session.median())))
    return typical_session * EQUITY_TRADING_DAYS_PER_YEAR


def _strategy_signals(
    bars: pd.DataFrame,
    strategy_name: str,
    momentum_parameters: MomentumParameters,
    mean_reversion_parameters: MeanReversionParameters,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    if strategy_name == "momentum":
        return (
            generate_momentum_signals(bars, momentum_parameters),
            asdict(momentum_parameters),
        )
    if strategy_name == "mean_reversion":
        return (
            generate_mean_reversion_signals(bars, mean_reversion_parameters),
            asdict(mean_reversion_parameters),
        )
    raise ValueError(f"Unsupported strategy: {strategy_name}")


def _equity_with_initial_cash(
    result: BacktestResult,
    initial_cash: float,
) -> pd.Series:
    if result.equity.empty:
        raise ValueError("Backtest produced no equity observations")
    initial_time = result.equity.index[0] - pd.Timedelta(nanoseconds=1)
    initial = pd.Series([initial_cash], index=[initial_time], dtype="float64")
    return pd.concat([initial, result.equity.astype("float64")])


def _metrics(
    result: BacktestResult,
    initial_cash: float,
    periods_per_year: int,
) -> pd.Series:
    equity = _equity_with_initial_cash(result, initial_cash)
    metrics = compute_metrics(
        equity,
        result.trades,
        result.positions,
        result.fills,
        periods_per_year=periods_per_year,
    )
    metrics["Final Equity"] = float(result.equity.iloc[-1])
    metrics["Net Profit"] = float(result.equity.iloc[-1] - initial_cash)
    return metrics


def run_strategy_backtest(
    canonical_bars: pd.DataFrame,
    instrument_id: str,
    strategy_name: str,
    momentum_parameters: MomentumParameters,
    mean_reversion_parameters: MeanReversionParameters,
    start: pd.Timestamp | None = None,
    end: pd.Timestamp | None = None,
    initial_cash: float = 100_000.0,
    slippage_bps: float = 0.0,
    cost_assumptions: CostAssumptions | None = None,
    flat_fee_bps: float | None = None,
) -> StrategyBacktestRun:
    """Run one rule-based strategy on one instrument."""
    if initial_cash <= 0:
        raise ValueError("initial_cash must be positive")
    if slippage_bps < 0:
        raise ValueError("slippage_bps must not be negative")
    if start is not None and end is not None and end <= start:
        raise ValueError("end must be later than start")

    instrument_bars = canonical_bars.loc[
        canonical_bars["instrument_id"].eq(instrument_id)
    ].copy()
    if instrument_bars.empty:
        available = sorted(canonical_bars["instrument_id"].astype(str).unique())
        raise ValueError(
            f"No rows found for {instrument_id}; available instruments: {available}"
        )

    if end is not None:
        instrument_bars = instrument_bars.loc[
            instrument_bars["bar_open_time_utc"].lt(end)
        ]
    if instrument_bars.empty:
        raise ValueError(f"No {instrument_id} bars exist before the requested end")

    signals, parameter_values = _strategy_signals(
        instrument_bars,
        strategy_name,
        momentum_parameters,
        mean_reversion_parameters,
    )
    signals["execution_target_position"] = delayed_execution_targets(signals)

    selected = pd.Series(True, index=signals.index)
    if start is not None:
        selected &= signals["bar_open_time_utc"].ge(start)
    if end is not None:
        selected &= signals["bar_open_time_utc"].lt(end)
    run_rows = signals.loc[selected].copy()
    tradable_rows = run_rows.loc[run_rows["signal_eligible"]]
    if tradable_rows.empty:
        raise ValueError(
            f"The requested range contains no executable {instrument_id} bars"
        )
    last_tradable_time = tradable_rows["bar_open_time_utc"].max()
    run_rows = run_rows.loc[
        run_rows["bar_open_time_utc"].le(last_tradable_time)
    ]
    if len(run_rows) < 2:
        raise ValueError(
            f"The requested range contains fewer than two {instrument_id} bars"
        )

    indexed = run_rows.set_index("bar_open_time_utc").sort_index()
    execution_bars = indexed[["open", "close"]].astype("float64")
    positions = indexed["execution_target_position"].astype("float64")

    if flat_fee_bps is None:
        assumptions = cost_assumptions or load_cost_assumptions()
        cost_model: Any = ProjectCostModel(assumptions, slippage_bps)
    else:
        if flat_fee_bps < 0:
            raise ValueError("flat_fee_bps must not be negative")
        cost_model = CostModel(flat_fee_bps, slippage_bps)

    quantity_steps = {instrument_id: 1.0}
    result = run_backtest(
        positions,
        execution_bars,
        asset=instrument_id,
        initial_cash=initial_cash,
        cost_model=cost_model,
        quantity_steps=quantity_steps,
        enforce_cash_limit=True,
    )
    benchmark = buy_and_hold(
        execution_bars,
        asset=instrument_id,
        initial_cash=initial_cash,
        cost_model=cost_model,
        quantity_steps=quantity_steps,
        enforce_cash_limit=True,
    )
    periods_per_year = infer_periods_per_year(run_rows)
    source_symbol = str(run_rows["source_symbol"].iloc[0])
    quote_currency = str(run_rows["quote_currency"].iloc[0])

    return StrategyBacktestRun(
        instrument_id=instrument_id,
        source_symbol=source_symbol,
        quote_currency=quote_currency,
        strategy_name=strategy_name,
        parameters=parameter_values,
        periods_per_year=periods_per_year,
        signals=run_rows,
        result=result,
        benchmark=benchmark,
        strategy_metrics=_metrics(result, initial_cash, periods_per_year),
        benchmark_metrics=_metrics(benchmark, initial_cash, periods_per_year),
        initial_cash=initial_cash,
    )


def _summary_row(
    run: StrategyBacktestRun,
    run_type: str,
    metrics: pd.Series,
) -> dict[str, Any]:
    return {
        "instrument_id": run.instrument_id,
        "source_symbol": run.source_symbol,
        "quote_currency": run.quote_currency,
        "strategy": run.strategy_name if run_type == "strategy" else "buy_and_hold",
        "run_type": run_type,
        "start_time_utc": run.signals["bar_open_time_utc"].min(),
        "end_time_utc": run.signals["bar_open_time_utc"].max(),
        "bar_count": int(len(run.signals)),
        "periods_per_year": run.periods_per_year,
        "initial_cash": run.initial_cash,
        **metrics.to_dict(),
    }


def _write_result_files(
    output_directory: Path,
    prefix: str,
    result: BacktestResult,
) -> None:
    result.curve.reset_index().to_csv(
        output_directory / f"{prefix}__equity.csv",
        index=False,
    )
    result.positions.reset_index().to_csv(
        output_directory / f"{prefix}__positions.csv",
        index=False,
    )
    result.fills.to_csv(output_directory / f"{prefix}__fills.csv", index=False)
    result.trades.to_csv(output_directory / f"{prefix}__trades.csv", index=False)


def write_backtest_outputs(
    runs: list[StrategyBacktestRun],
    output_directory: Path,
    metadata: dict[str, Any],
    include_benchmark: bool = True,
) -> pd.DataFrame:
    """Write detailed artifacts and one comparison table for all runs."""
    if not runs:
        raise ValueError("At least one backtest run is required")
    output_directory.mkdir(parents=True, exist_ok=True)

    summary_rows: list[dict[str, Any]] = []
    benchmark_written: set[str] = set()
    for run in runs:
        prefix = f"{run.instrument_id}__{run.strategy_name}"
        run.signals.to_csv(
            output_directory / f"{prefix}__signals.csv",
            index=False,
        )
        _write_result_files(output_directory, prefix, run.result)
        summary_rows.append(_summary_row(run, "strategy", run.strategy_metrics))

        if include_benchmark and run.instrument_id not in benchmark_written:
            benchmark_prefix = f"{run.instrument_id}__buy_and_hold"
            _write_result_files(
                output_directory,
                benchmark_prefix,
                run.benchmark,
            )
            summary_rows.append(
                _summary_row(run, "benchmark", run.benchmark_metrics)
            )
            benchmark_written.add(run.instrument_id)

    summary = pd.DataFrame(summary_rows)
    summary.to_csv(output_directory / "summary.csv", index=False)

    payload = {
        **metadata,
        "runs": [
            {
                "instrument_id": run.instrument_id,
                "source_symbol": run.source_symbol,
                "quote_currency": run.quote_currency,
                "strategy": run.strategy_name,
                "parameters": run.parameters,
                "periods_per_year": run.periods_per_year,
            }
            for run in runs
        ],
    }
    (output_directory / "parameters.json").write_text(
        json.dumps(payload, indent=2, default=str) + "\n",
        encoding="utf-8",
    )
    return summary

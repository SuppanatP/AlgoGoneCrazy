"""Compare a small, predefined parameter grid before the official test period."""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict, dataclass
from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from src.strategy_backtest import (  # noqa: E402
    StrategyBacktestRun,
    load_canonical_bars,
    resolve_instrument_id,
    run_strategy_backtest,
)
from trading_strategy import (  # noqa: E402
    MeanReversionParameters,
    MomentumParameters,
)


OFFICIAL_TEST_START_UTC = pd.Timestamp("2026-09-30T23:59:59.999999Z")

PERIODS = {
    "spy_usd": {
        "development": ("2023-09-18T00:00:00Z", "2025-10-01T00:00:00Z"),
        "validation": ("2025-10-01T00:00:00Z", "2026-09-30T23:59:59.999999Z"),
    },
    "sti_es3_sgd": {
        "development": ("2026-07-21T00:00:00Z", "2026-09-01T00:00:00Z"),
        "validation": ("2026-09-01T00:00:00Z", "2026-09-30T23:59:59.999999Z"),
    },
}


@dataclass(frozen=True)
class Candidate:
    name: str
    strategy: str
    momentum: MomentumParameters
    mean_reversion: MeanReversionParameters

    @property
    def parameters(self) -> dict[str, object]:
        selected = self.momentum if self.strategy == "momentum" else self.mean_reversion
        return asdict(selected)


BASELINE_MOMENTUM = MomentumParameters()
BASELINE_MEAN_REVERSION = MeanReversionParameters()

CANDIDATES = (
    Candidate(
        "momentum_short",
        "momentum",
        MomentumParameters(fast_window=8, slow_window=26, momentum_lookback=13),
        BASELINE_MEAN_REVERSION,
    ),
    Candidate(
        "momentum_baseline",
        "momentum",
        BASELINE_MOMENTUM,
        BASELINE_MEAN_REVERSION,
    ),
    Candidate(
        "momentum_slow",
        "momentum",
        MomentumParameters(fast_window=26, slow_window=104, momentum_lookback=52),
        BASELINE_MEAN_REVERSION,
    ),
    Candidate(
        "mean_reversion_sensitive",
        "mean_reversion",
        BASELINE_MOMENTUM,
        MeanReversionParameters(
            window=20,
            entry_z_score=-1.25,
            exit_z_score=0.0,
            maximum_holding_bars=40,
        ),
    ),
    Candidate(
        "mean_reversion_baseline",
        "mean_reversion",
        BASELINE_MOMENTUM,
        BASELINE_MEAN_REVERSION,
    ),
    Candidate(
        "mean_reversion_conservative",
        "mean_reversion",
        BASELINE_MOMENTUM,
        MeanReversionParameters(
            window=52,
            entry_z_score=-2.0,
            exit_z_score=-0.25,
            maximum_holding_bars=104,
        ),
    ),
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input-file",
        type=Path,
        default=Path(
            "data/canonical/market_bars_30m_training_validation.csv"
        ),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("results/parameter_experiments"),
    )
    parser.add_argument(
        "--instrument",
        nargs="+",
        default=["SPY", "ES3.SI"],
    )
    parser.add_argument("--initial-cash", type=float, default=100_000.0)
    parser.add_argument("--slippage-bps", type=float, default=0.0)
    parser.add_argument("--flat-fee-bps", type=float)
    return parser


def result_row(
    run: StrategyBacktestRun,
    period_name: str,
    candidate: Candidate | None,
) -> dict[str, object]:
    benchmark = candidate is None
    metrics = run.benchmark_metrics if benchmark else run.strategy_metrics
    parameters = {} if benchmark else candidate.parameters
    return {
        "instrument_id": run.instrument_id,
        "source_symbol": run.source_symbol,
        "quote_currency": run.quote_currency,
        "period": period_name,
        "parameter_set": "buy_and_hold" if benchmark else candidate.name,
        "strategy": "buy_and_hold" if benchmark else candidate.strategy,
        "start_time_utc": run.signals["bar_open_time_utc"].min(),
        "end_time_utc": run.signals["bar_open_time_utc"].max(),
        "bar_count": len(run.signals),
        "parameters": json.dumps(parameters, sort_keys=True),
        **metrics.to_dict(),
    }


def main() -> None:
    args = build_parser().parse_args()
    canonical = load_canonical_bars(args.input_file)
    instruments = list(
        dict.fromkeys(resolve_instrument_id(value) for value in args.instrument)
    )

    rows: list[dict[str, object]] = []
    for instrument_id in instruments:
        for period_name, (start_value, end_value) in PERIODS[instrument_id].items():
            start = pd.Timestamp(start_value)
            end = pd.Timestamp(end_value)
            benchmark_added = False

            for candidate in CANDIDATES:
                print(
                    f"Running {instrument_id} / {period_name} / {candidate.name}..."
                )
                run = run_strategy_backtest(
                    canonical,
                    instrument_id,
                    candidate.strategy,
                    candidate.momentum,
                    candidate.mean_reversion,
                    start=start,
                    end=end,
                    initial_cash=args.initial_cash,
                    slippage_bps=args.slippage_bps,
                    flat_fee_bps=args.flat_fee_bps,
                )
                rows.append(result_row(run, period_name, candidate))
                if not benchmark_added:
                    rows.append(result_row(run, period_name, None))
                    benchmark_added = True

    comparison = pd.DataFrame(rows)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    comparison.to_csv(args.output_dir / "parameter_comparison.csv", index=False)

    manifest = {
        "input_file": str(args.input_file),
        "official_test_start_utc": OFFICIAL_TEST_START_UTC.isoformat(),
        "official_test_data_used": False,
        "initial_cash_per_run": args.initial_cash,
        "slippage_bps": args.slippage_bps,
        "cost_model": (
            "flat_bps" if args.flat_fee_bps is not None else "project_assumptions"
        ),
        "flat_fee_bps": args.flat_fee_bps,
        "periods": PERIODS,
        "candidates": [
            {
                "name": candidate.name,
                "strategy": candidate.strategy,
                "parameters": candidate.parameters,
            }
            for candidate in CANDIDATES
        ],
    }
    (args.output_dir / "experiment_manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n",
        encoding="utf-8",
    )

    display_columns = [
        "instrument_id",
        "period",
        "parameter_set",
        "Total Return",
        "Annual Volatility",
        "Sharpe Ratio",
        "Max Drawdown",
        "Number of Trades",
        "Total Transaction Cost",
    ]
    print("\nParameter comparison")
    print(
        comparison[display_columns].to_string(
            index=False,
            float_format=lambda value: f"{value:,.4f}",
        )
    )
    print(f"\nOutputs -> {args.output_dir}")


if __name__ == "__main__":
    main()

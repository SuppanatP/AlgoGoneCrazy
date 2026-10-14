"""Backtest momentum and mean-reversion on canonical 30-minute bars."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from src.strategy_backtest import (  # noqa: E402
    load_canonical_bars,
    parse_utc_timestamp,
    resolve_instrument_id,
    run_strategy_backtest,
    write_backtest_outputs,
)
from trading_strategy import (  # noqa: E402
    MeanReversionParameters,
    MomentumParameters,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Backtest the rule-based momentum and mean-reversion strategies "
            "using next-real-bar execution."
        )
    )
    parser.add_argument(
        "--input-file",
        type=Path,
        default=Path("data/canonical/market_bars_30m.csv"),
        help="Canonical 30-minute CSV produced by consolidation.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("results/strategy_backtests"),
        help="Directory for summary, signal, trade, fill, position, and equity CSVs.",
    )
    parser.add_argument(
        "--instrument",
        nargs="+",
        default=["SPY", "ES3.SI"],
        help="SPY and/or ES3.SI; canonical instrument IDs are also accepted.",
    )
    parser.add_argument(
        "--strategy",
        nargs="+",
        choices=["momentum", "mean_reversion"],
        default=["momentum", "mean_reversion"],
    )
    parser.add_argument(
        "--start",
        help="Inclusive UTC start timestamp; earlier rows remain available for warmup.",
    )
    parser.add_argument(
        "--end",
        help="Exclusive UTC end timestamp.",
    )
    parser.add_argument("--initial-cash", type=float, default=100_000.0)
    parser.add_argument(
        "--slippage-bps",
        type=float,
        default=0.0,
        help="Execution slippage per order in basis points.",
    )
    parser.add_argument(
        "--flat-fee-bps",
        type=float,
        help=(
            "Override project broker commissions with one flat fee in basis "
            "points. Omit to use config/cost_assumptions.json."
        ),
    )
    parser.add_argument(
        "--no-benchmark",
        action="store_true",
        help="Do not include buy-and-hold in the output comparison.",
    )

    momentum = parser.add_argument_group("momentum parameters")
    momentum.add_argument("--momentum-fast-window", type=int, default=13)
    momentum.add_argument("--momentum-slow-window", type=int, default=52)
    momentum.add_argument("--momentum-lookback", type=int, default=26)
    momentum.add_argument("--momentum-entry-threshold", type=float, default=0.0)
    momentum.add_argument("--momentum-exit-threshold", type=float, default=0.0)

    mean_reversion = parser.add_argument_group("mean-reversion parameters")
    mean_reversion.add_argument("--mean-reversion-window", type=int, default=26)
    mean_reversion.add_argument("--mean-reversion-entry-z", type=float, default=-1.5)
    mean_reversion.add_argument("--mean-reversion-exit-z", type=float, default=0.0)
    mean_reversion.add_argument("--mean-reversion-max-holding", type=int, default=52)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    start = parse_utc_timestamp(args.start, "start")
    end = parse_utc_timestamp(args.end, "end")
    if start is not None and end is not None and end <= start:
        raise SystemExit("--end must be later than --start")

    instruments = list(dict.fromkeys(resolve_instrument_id(x) for x in args.instrument))
    strategies = list(dict.fromkeys(args.strategy))
    canonical = load_canonical_bars(args.input_file)

    momentum_parameters = MomentumParameters(
        fast_window=args.momentum_fast_window,
        slow_window=args.momentum_slow_window,
        momentum_lookback=args.momentum_lookback,
        entry_threshold=args.momentum_entry_threshold,
        exit_threshold=args.momentum_exit_threshold,
    )
    mean_reversion_parameters = MeanReversionParameters(
        window=args.mean_reversion_window,
        entry_z_score=args.mean_reversion_entry_z,
        exit_z_score=args.mean_reversion_exit_z,
        maximum_holding_bars=args.mean_reversion_max_holding,
    )

    runs = []
    for instrument_id in instruments:
        for strategy_name in strategies:
            print(f"Running {instrument_id} / {strategy_name}...")
            runs.append(
                run_strategy_backtest(
                    canonical,
                    instrument_id,
                    strategy_name,
                    momentum_parameters,
                    mean_reversion_parameters,
                    start=start,
                    end=end,
                    initial_cash=args.initial_cash,
                    slippage_bps=args.slippage_bps,
                    flat_fee_bps=args.flat_fee_bps,
                )
            )

    metadata = {
        "input_file": str(args.input_file),
        "start_utc": start.isoformat() if start is not None else None,
        "end_utc_exclusive": end.isoformat() if end is not None else None,
        "initial_cash_per_run": args.initial_cash,
        "cost_model": (
            "flat_bps" if args.flat_fee_bps is not None else "project_assumptions"
        ),
        "flat_fee_bps": args.flat_fee_bps,
        "slippage_bps": args.slippage_bps,
        "execution_rule": (
            "signal after close; execute at next complete real bar open; "
            "force liquidation at the final real bar close"
        ),
        "valuation_rule": "raw close",
        "signal_price_rule": "adjusted close",
        "currency_rule": "each instrument is backtested independently in native currency",
        "dividend_rule": "no separate dividend cash flow in the current raw-price ledger",
    }
    summary = write_backtest_outputs(
        runs,
        args.output_dir,
        metadata,
        include_benchmark=not args.no_benchmark,
    )
    display_columns = [
        "instrument_id",
        "quote_currency",
        "strategy",
        "Total Return",
        "Annual Volatility",
        "Sharpe Ratio",
        "Max Drawdown",
        "Number of Trades",
        "Final Equity",
    ]
    print("\nBacktest summary")
    print(summary[display_columns].to_string(index=False, float_format=lambda x: f"{x:,.4f}"))
    print(f"\nOutputs -> {args.output_dir}")


if __name__ == "__main__":
    main()

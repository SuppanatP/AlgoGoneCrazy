"""Final metrics from the records written by run_daily.py.

Usage:
  python report_metrics.py                    # strategy only
  python report_metrics.py --benchmark        # + equal-weight buy & hold, same costs
  python report_metrics.py --records path/to/records --no-plot

Outputs (in results/):
  metrics_table.csv              all report metrics incl. transaction costs
  per_asset.csv                  trades, P&L before/after costs, fees, slippage per asset
  figures/equity_drawdown.png    equity (net vs before costs), drawdown, cumulative costs
Open positions at the end are valued at the last close (equity) but are not counted
as trades until they are closed.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

import config
from src.backtest import buy_and_hold
from src.data_source import fetch_bars
from src.metrics import (compute_metrics, cumulative_costs, drawdown_series,
                         per_asset_breakdown)
from src.records import load_state, read_records
from src.tracker import CostModel


def load_strategy(records_dir):
    recs = read_records(records_dir)
    if recs["equity"].empty:
        raise SystemExit(f"No records in {records_dir}. Run run_daily.py first.")
    state = load_state(records_dir)

    equity = recs["equity"]["equity"]
    # prepend starting capital so the first bar's return is included
    start = equity.index[0] - pd.Timedelta(config.BAR_FREQ)
    equity = pd.concat([pd.Series([state["portfolio"]["initial_cash"]], index=[start]), equity])
    positions = recs["decisions"].pivot_table(index=recs["decisions"].index, columns="asset",
                                              values="target", aggfunc="last").ffill().fillna(0)
    return equity, recs["trades"], recs["fills"], positions


def benchmark(start, end):
    bars = {a: fetch_bars(start, end, spec["csv"], config.TIME_COL, spec.get("symbol"),
                          getattr(config, "SYMBOL_COL", "symbol"))
            for a, spec in config.ASSETS.items()}
    costs = CostModel(config.FEE_BPS, config.SLIPPAGE_BPS, getattr(config, "MIN_FEE", 0.0),
                      dict(getattr(config, "COST_OVERRIDES", {})))
    weights = {a: spec["weight"] for a, spec in config.ASSETS.items()}
    return buy_and_hold(bars, weights, initial_cash=config.INITIAL_CASH, cost_model=costs)


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--records", default=str(config.RECORDS_DIR))
    ap.add_argument("--output-dir", default=str(config.RESULTS_DIR))
    ap.add_argument("--benchmark", action="store_true")
    ap.add_argument("--no-plot", action="store_true")
    args = ap.parse_args(argv)

    equity, trades, fills, positions = load_strategy(args.records)
    table = {"Strategy": compute_metrics(equity, trades, positions, fills)}
    curves = {"Strategy": (equity, fills)}

    if args.benchmark:
        bh = benchmark(positions.index[0], positions.index[-1] + pd.Timedelta(config.BAR_FREQ))
        table["Buy & Hold"] = compute_metrics(bh.equity, bh.trades, bh.positions, bh.fills)
        curves["Buy & Hold"] = (bh.equity, bh.fills)

    table = pd.DataFrame(table)
    per_asset = per_asset_breakdown(trades, fills)
    fmt = lambda v: f"{v:,.4f}"
    print(f"Period: {equity.index[0]} -> {equity.index[-1]}\n")
    print(table.to_string(float_format=fmt))
    print("\nPer asset (strategy):")
    print(per_asset.to_string(float_format=fmt))

    results = Path(args.output_dir)
    results.mkdir(parents=True, exist_ok=True)
    table.to_csv(results / "metrics_table.csv")
    per_asset.to_csv(results / "per_asset.csv")
    print(f"\nsaved {results / 'metrics_table.csv'} and {results / 'per_asset.csv'}")

    if not args.no_plot:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(12, 9), sharex=True,
                                            gridspec_kw={"height_ratios": [2, 1, 1]})
        for name, (eq, fl) in curves.items():
            costs = cumulative_costs(fl, eq.index)
            line, = ax1.plot(eq.index, eq / eq.iloc[0], label=f"{name} (net)")
            if name == "Strategy":
                ax1.plot(eq.index, (eq + costs) / eq.iloc[0], ls="--", color=line.get_color(),
                         label=f"{name} (before costs)")
            ax2.plot(eq.index, drawdown_series(eq), label=name)
            ax3.plot(eq.index, costs / eq.iloc[0], label=name)
        ax1.set(title="Equity (normalised to 1)", ylabel="growth of 1")
        ax2.set(title="Drawdown", ylabel="drawdown")
        ax3.set(title="Cumulative transaction cost (% of initial capital)", ylabel="cost")
        ax1.legend(); ax3.legend()
        plt.tight_layout()
        (results / "figures").mkdir(exist_ok=True)
        fig.savefig(results / "figures" / "equity_drawdown.png", dpi=150)
        print(f"saved {results / 'figures' / 'equity_drawdown.png'}")


if __name__ == "__main__":
    main()

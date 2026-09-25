"""Bar-by-bar multi-asset backtest: positions -> Portfolio -> equity curve and trade log."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
import pandas as pd

from .tracker import CostModel, Portfolio


def execute_bar(pf: Portfolio, ts, targets: dict, prev_targets: dict, opens: dict,
                closes: dict, weights: dict, leverage: float = 1.0,
                close_at_end: bool = False) -> list:
    """One 30-min bar for all assets.

    targets / opens / closes contain only the assets that have a bar at `ts`.
    Target t for asset a means: hold t x weights[a] x leverage x equity in a
    (e.g. weight 1/3 and t = -1 -> short one third of the portfolio).
    Orders execute at the open, only for assets whose target changed; every order
    in the bar is sized from the same equity snapshot. Then the portfolio is marked
    at the close. Shared by run_portfolio_backtest and the daily runner.
    """
    fills = []
    changed = {a: t for a, t in targets.items() if t != prev_targets.get(a, 0.0)}
    if changed:
        eq = pf.equity(opens)
        for a, t in changed.items():
            qty = t * weights[a] * leverage * eq / opens[a]
            fill = pf.target_position(ts, a, qty, opens[a])
            if fill is not None:
                fills.append(fill)
    if close_at_end:
        pf.close_all(ts, {**pf.last_price, **closes})
    pf.mark(ts, closes)
    return fills


@dataclass
class BacktestResult:
    equity: pd.Series          # portfolio value at each bar close
    trades: pd.DataFrame       # one row per round-trip trade
    fills: pd.DataFrame        # every executed order, with fee and slippage
    positions: pd.DataFrame    # target position per bar and asset
    curve: pd.DataFrame        # equity, cash, exposure, cumulative costs, quantities
    portfolio: Portfolio


def run_portfolio_backtest(positions: pd.DataFrame, bars: dict, weights: Optional[dict] = None,
                           initial_cash: float = 100_000.0, fee_bps: float = 10.0,
                           slippage_bps: float = 0.0, cost_model: Optional[CostModel] = None,
                           leverage: float = 1.0) -> BacktestResult:
    """positions: DataFrame (time x asset) of -1/0/+1; bars: {asset: DataFrame with open, close}.

    weights default to equal (1 / number of assets). Open positions are closed at the end.
    """
    assets = list(bars)
    weights = weights or {a: 1.0 / len(assets) for a in assets}
    index = bars[assets[0]].index
    for a in assets[1:]:
        index = index.union(bars[a].index)
    opens = pd.DataFrame({a: bars[a]["open"] for a in assets}).reindex(index).to_numpy()
    closes = pd.DataFrame({a: bars[a]["close"] for a in assets}).reindex(index).to_numpy()
    pos = positions.reindex(index=index, columns=assets).to_numpy(dtype=float)

    pf = Portfolio(initial_cash, fee_bps, slippage_bps, cost_model)
    prev = {a: 0.0 for a in assets}
    last = len(index) - 1
    for i, ts in enumerate(index):
        have = [j for j in range(len(assets)) if not np.isnan(opens[i, j])]
        targets = {assets[j]: (prev[assets[j]] if np.isnan(pos[i, j]) else pos[i, j]) for j in have}
        execute_bar(pf, ts, targets, prev,
                    {assets[j]: opens[i, j] for j in have},
                    {assets[j]: closes[i, j] for j in have},
                    weights, leverage, close_at_end=(i == last))
        prev.update(targets)

    curve = pf.equity_curve
    return BacktestResult(curve["equity"], pf.trades, pf.fills,
                          pd.DataFrame(pos, index=index, columns=assets).fillna(0.0), curve, pf)


def run_backtest(positions: pd.Series, bars: pd.DataFrame, asset: str = "BTC",
                 **kwargs) -> BacktestResult:
    """Single-asset convenience wrapper (the asset gets the whole portfolio)."""
    return run_portfolio_backtest(positions.to_frame(asset), {asset: bars}, {asset: 1.0}, **kwargs)


def buy_and_hold(bars, weights: Optional[dict] = None, asset: str = "BTC",
                 **kwargs) -> BacktestResult:
    """Benchmark: long every asset at its weight for the whole period, never rebalanced.
    `bars` is {asset: DataFrame} or a single DataFrame (then named `asset`)."""
    if isinstance(bars, pd.DataFrame):
        bars, weights = {asset: bars}, {asset: 1.0}
    pos = pd.DataFrame({a: pd.Series(1.0, index=b.index) for a, b in bars.items()})
    return run_portfolio_backtest(pos, bars, weights, **kwargs)

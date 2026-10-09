"""Performance and transaction-cost metrics for the report table.

Crypto trades 24/7, so annualisation uses 365 days: 48 x 365 = 17,520 thirty-minute bars.
"""
from __future__ import annotations

from typing import Optional

import numpy as np
import pandas as pd

PERIODS_PER_YEAR_30M = 48 * 365


def drawdown_series(equity: pd.Series) -> pd.Series:
    """Percentage below the running peak at each bar (0 at a new high, negative otherwise)."""
    return equity / equity.cummax() - 1


def max_drawdown(equity: pd.Series) -> float:
    return float(drawdown_series(equity).min())


def drawdown_episodes(equity: pd.Series, min_depth: float = 0.0) -> pd.DataFrame:
    """One row per drawdown: from falling below a peak until a new peak is reached.
    Episodes shallower than `min_depth` (0.01 = 1%) are dropped; `end` is NaT if unrecovered."""
    dd = drawdown_series(equity)
    rows, in_dd = [], False
    for ts, d in dd.items():
        if d < 0:
            if not in_dd:
                in_dd, start, trough, trough_ts = True, ts, d, ts
            elif d < trough:
                trough, trough_ts = d, ts
        elif in_dd:
            rows.append((start, trough_ts, ts, trough, True))
            in_dd = False
    if in_dd:
        rows.append((start, trough_ts, pd.NaT, trough, False))
    eps = pd.DataFrame(rows, columns=["start", "trough", "end", "depth", "recovered"])
    if len(eps):
        end_times = pd.to_datetime(eps["end"], utc=True)
        start_times = pd.to_datetime(eps["start"], utc=True)
        eps["duration"] = end_times.fillna(equity.index[-1]) - start_times
    return eps[eps["depth"] <= -min_depth].reset_index(drop=True)


def _annual_return(equity: pd.Series, periods_per_year: int) -> float:
    n_years = (len(equity) - 1) / periods_per_year
    total = equity.iloc[-1] / equity.iloc[0] - 1
    return (1 + total) ** (1 / n_years) - 1 if n_years > 0 and total > -1 else np.nan


def _fills_by_time(fills: pd.DataFrame) -> pd.DataFrame:
    return fills.set_index("timestamp") if "timestamp" in fills.columns else fills


def cumulative_costs(fills: pd.DataFrame, index: pd.Index) -> pd.Series:
    """Running total of fees + slippage, aligned to the equity index."""
    if fills is None or fills.empty:
        return pd.Series(0.0, index=index)
    per_ts = _fills_by_time(fills)["total_cost"].groupby(level=0).sum().cumsum()
    return per_ts.reindex(index.union(per_ts.index)).ffill().reindex(index).fillna(0.0)


def cost_metrics(equity: pd.Series, fills: pd.DataFrame, trades: Optional[pd.DataFrame] = None,
                 periods_per_year: int = PERIODS_PER_YEAR_30M) -> dict:
    """Transaction-cost section of the report."""
    n_years = (len(equity) - 1) / periods_per_year
    fees = fills["fee"].sum() if len(fills) else 0.0
    slip = fills["slippage_cost"].sum() if len(fills) else 0.0
    total = fees + slip
    notional = fills["notional"].sum() if len(fills) else 0.0
    gross_equity = equity + cumulative_costs(fills, equity.index)   # what equity would be with zero costs
    ann_net = _annual_return(equity, periods_per_year)
    ann_gross = _annual_return(gross_equity, periods_per_year)
    n_trades = len(trades) if trades is not None else 0
    return {
        "Annual Return (before costs)": ann_gross,
        "Annual Cost Drag": ann_gross - ann_net,
        "Total Fees": fees,
        "Total Slippage": slip,
        "Total Transaction Cost": total,
        "Transaction Cost (% of initial capital)": total / equity.iloc[0],
        "Number of Orders": len(fills),
        "Traded Notional": notional,
        "Annual Turnover (x equity)": notional / equity.mean() / n_years if n_years > 0 else np.nan,
        "Average Cost per Trade": total / n_trades if n_trades else np.nan,
        "Cost (bps of traded notional)": total / notional * 1e4 if notional else np.nan,
    }


def compute_metrics(equity: pd.Series, trades: Optional[pd.DataFrame] = None,
                    positions=None, fills: Optional[pd.DataFrame] = None,
                    periods_per_year: int = PERIODS_PER_YEAR_30M,
                    risk_free_rate: float = 0.0, dd_min_depth: float = 0.01) -> pd.Series:
    """All report metrics for one strategy.

    equity    : portfolio value per bar
    trades    : round-trip trade log
    positions : position per bar (Series, or DataFrame with one column per asset)
    fills     : order log with fee / slippage_cost / notional -> transaction-cost section
    """
    r = equity.pct_change().dropna()
    std = r.std(ddof=1)
    rf_bar = (1 + risk_free_rate) ** (1 / periods_per_year) - 1
    excess = r - rf_bar
    downside = np.sqrt((np.minimum(excess, 0) ** 2).mean())
    ann_return = _annual_return(equity, periods_per_year)
    mdd = max_drawdown(equity)
    episodes = drawdown_episodes(equity, dd_min_depth)

    out = {
        "Total Return": equity.iloc[-1] / equity.iloc[0] - 1,
        "Annual Return": ann_return,
        "Annual Volatility": std * np.sqrt(periods_per_year),
        "Sharpe Ratio": excess.mean() / std * np.sqrt(periods_per_year) if std > 0 else np.nan,
        "Sortino Ratio": excess.mean() / downside * np.sqrt(periods_per_year) if downside > 0 else np.nan,
        "Max Drawdown": mdd,
        "Calmar Ratio": ann_return / abs(mdd) if mdd < 0 else np.nan,
        f"Drawdown Count (>= {dd_min_depth:.0%})": len(episodes),
        "Longest Drawdown (days)":
            episodes["duration"].max().total_seconds() / 86400 if len(episodes) else 0.0,
    }

    if trades is not None and len(trades):
        pnl = trades["net_pnl"]
        gross_win, gross_loss = pnl[pnl > 0].sum(), -pnl[pnl < 0].sum()
        out.update({
            "Number of Trades": len(trades),
            "Win Rate": (pnl > 0).mean(),
            "Average Holding Period (hours)":
                trades["holding_period"].dt.total_seconds().mean() / 3600,
            "Profit per Trade": pnl.mean(),
            "Average Return per Trade": trades["return_pct"].mean(),
            "Profit Factor": gross_win / gross_loss if gross_loss > 0 else np.nan,
        })
    else:
        out.update({"Number of Trades": 0, "Win Rate": np.nan,
                    "Average Holding Period (hours)": np.nan, "Profit per Trade": np.nan,
                    "Average Return per Trade": np.nan, "Profit Factor": np.nan})

    if positions is not None:
        held = positions != 0
        out["Exposure (time in market)"] = (held.any(axis=1) if held.ndim == 2 else held).mean()

    if fills is not None:
        out.update(cost_metrics(equity, fills, trades, periods_per_year))
    return pd.Series(out)


def per_asset_breakdown(trades: pd.DataFrame, fills: pd.DataFrame) -> pd.DataFrame:
    """Contribution of each asset: trades, win rate, P&L before/after costs, costs."""
    rows = {}
    assets = sorted(set(fills["asset"]) | set(trades["asset"])) if len(fills) else []
    for a in assets:
        t, f = trades[trades["asset"] == a], fills[fills["asset"] == a]
        rows[a] = {
            "Number of Trades": len(t),
            "Win Rate": (t["net_pnl"] > 0).mean() if len(t) else np.nan,
            "P&L before Costs": t["pnl_before_costs"].sum(),
            "Fees": f["fee"].sum(),
            "Slippage": f["slippage_cost"].sum(),
            "Total Cost": f["total_cost"].sum(),
            "Net P&L (closed trades)": t["net_pnl"].sum(),
            "Traded Notional": f["notional"].sum(),
            "Average Holding Period (hours)":
                t["holding_period"].dt.total_seconds().mean() / 3600 if len(t) else np.nan,
        }
    return pd.DataFrame(rows)


def metrics_table(results: dict) -> pd.DataFrame:
    """Side-by-side table, e.g. {"LightGBM": result, "Buy & Hold": bh} -> DataFrame."""
    return pd.DataFrame({name: compute_metrics(res.equity, res.trades, res.positions, res.fills)
                         for name, res in results.items()})

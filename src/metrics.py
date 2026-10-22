"""Performance and transaction-cost metrics for the report table.

Portfolio execution remains intraday.  Report-level risk metrics can instead
use daily portfolio returns so that intraday serial correlation is not silently
treated as independence during annualisation.
"""
from __future__ import annotations

from typing import Optional

import numpy as np
import pandas as pd

PERIODS_PER_YEAR_30M = 48 * 365
CALENDAR_DAYS_PER_YEAR = 365


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
        eps["duration"] = eps["end"].fillna(equity.index[-1]) - eps["start"]
    return eps[eps["depth"] <= -min_depth].reset_index(drop=True)


def _elapsed_years(equity: pd.Series, periods_per_year: int) -> float:
    """Elapsed years, preferring timestamps over an observation count.

    Equity rows are marked with the opening timestamp of the bar whose close
    was used for valuation.  The typical bar length is therefore added to the
    final timestamp when a regular DatetimeIndex is available.
    """
    if isinstance(equity.index, pd.DatetimeIndex) and len(equity) > 1:
        index = equity.index.sort_values()
        positive_differences = index.to_series().diff().dropna()
        positive_differences = positive_differences.loc[
            positive_differences.gt(pd.Timedelta(0))
        ]
        typical_interval = (
            positive_differences.median()
            if not positive_differences.empty
            else pd.Timedelta(0)
        )
        elapsed = index[-1] + typical_interval - index[0]
        elapsed_days = elapsed.total_seconds() / 86_400.0
        if elapsed_days > 0:
            return elapsed_days / CALENDAR_DAYS_PER_YEAR
    return (len(equity) - 1) / periods_per_year


def _annual_return(
    equity: pd.Series,
    periods_per_year: int,
    *,
    use_elapsed_time: bool = False,
) -> float:
    n_years = (
        _elapsed_years(equity, periods_per_year)
        if use_elapsed_time
        else (len(equity) - 1) / periods_per_year
    )
    total = equity.iloc[-1] / equity.iloc[0] - 1
    return (1 + total) ** (1 / n_years) - 1 if n_years > 0 and total > -1 else np.nan


def _typical_interval(index: pd.DatetimeIndex) -> pd.Timedelta:
    differences = index.to_series().diff().dropna()
    differences = differences.loc[differences.gt(pd.Timedelta(0))]
    return differences.median() if not differences.empty else pd.Timedelta(0)


def daily_portfolio_returns(
    equity: pd.Series,
    *,
    drop_incomplete_final_day: bool = True,
) -> pd.Series:
    """Return complete UTC-day portfolio returns from an intraday equity curve.

    The first value is the pre-period capital supplied by the caller.  Every
    later value contains intraday realised/unrealised P&L and transaction costs.
    The final partial UTC day is excluded because treating a few hours as a full
    day would bias annualised volatility and Sharpe.
    """
    if not isinstance(equity.index, pd.DatetimeIndex):
        raise TypeError("Daily risk metrics require a DatetimeIndex")
    if len(equity) < 2:
        return pd.Series(dtype="float64", name="daily_portfolio_return")

    work = equity.astype("float64").sort_index()
    if work.index.tz is None:
        work.index = work.index.tz_localize("UTC")
    else:
        work.index = work.index.tz_convert("UTC")
    if work.index.has_duplicates:
        raise ValueError("Equity timestamps must be unique")

    initial_equity = float(work.iloc[0])
    observations = work.iloc[1:]
    daily_levels = observations.resample("1D").last().dropna()
    if daily_levels.empty:
        return pd.Series(dtype="float64", name="daily_portfolio_return")

    if drop_incomplete_final_day:
        interval = _typical_interval(observations.index)
        final_timestamp = observations.index[-1]
        next_midnight = final_timestamp.normalize() + pd.Timedelta(days=1)
        if final_timestamp + interval < next_midnight:
            daily_levels = daily_levels.iloc[:-1]
    if daily_levels.empty:
        return pd.Series(dtype="float64", name="daily_portfolio_return")

    initial_label = daily_levels.index[0] - pd.Timedelta(days=1)
    levels = pd.concat(
        [
            pd.Series([initial_equity], index=[initial_label], dtype="float64"),
            daily_levels,
        ]
    )
    returns = levels.pct_change().dropna()
    returns.name = "daily_portfolio_return"
    return returns


def _aligned_period_risk_free_returns(
    return_index: pd.Index,
    annual_rate: float | pd.Series,
    periods_per_year: int,
) -> tuple[pd.Series, pd.Series]:
    """Align annual rates without look-ahead and convert them to period returns."""
    if isinstance(annual_rate, pd.Series):
        rates = annual_rate.astype("float64").sort_index()
        if not isinstance(rates.index, pd.DatetimeIndex):
            raise TypeError("Risk-free rate series must use a DatetimeIndex")
        if rates.index.tz is None:
            rates.index = rates.index.tz_localize("UTC")
        else:
            rates.index = rates.index.tz_convert("UTC")
        target = pd.DatetimeIndex(return_index)
        if target.tz is None:
            target = target.tz_localize("UTC")
        else:
            target = target.tz_convert("UTC")
        aligned = rates.reindex(rates.index.union(target)).sort_index().ffill().reindex(target)
        if aligned.isna().any():
            first_missing = aligned.index[aligned.isna()][0]
            raise ValueError(
                "No risk-free observation is available on or before "
                f"{first_missing.isoformat()}"
            )
        aligned.index = return_index
    else:
        value = float(annual_rate)
        if not np.isfinite(value) or value <= -1:
            raise ValueError("Annual risk-free rate must be finite and greater than -1")
        aligned = pd.Series(value, index=return_index, dtype="float64")
    period = (1.0 + aligned) ** (1.0 / periods_per_year) - 1.0
    return aligned, period


def _fills_by_time(fills: pd.DataFrame) -> pd.DataFrame:
    return fills.set_index("timestamp") if "timestamp" in fills.columns else fills


def cumulative_costs(fills: pd.DataFrame, index: pd.Index) -> pd.Series:
    """Running total of fees + slippage, aligned to the equity index."""
    if fills is None or fills.empty:
        return pd.Series(0.0, index=index)
    per_ts = _fills_by_time(fills)["total_cost"].groupby(level=0).sum().cumsum()
    return per_ts.reindex(index.union(per_ts.index)).ffill().reindex(index).fillna(0.0)


def compute_metrics(equity: pd.Series, trades: Optional[pd.DataFrame] = None,
                    positions=None, fills: Optional[pd.DataFrame] = None,
                    periods_per_year: int = PERIODS_PER_YEAR_30M,
                    risk_free_rate: float | pd.Series = 0.0,
                    metric_frequency: str = "bar",
                    risk_free_source: str | None = None,
                    drop_incomplete_final_day: bool = True,
                    annualize_by_elapsed_time: bool = False) -> pd.Series:
    """Project-required performance metrics for one strategy.

    equity    : portfolio value per bar
    trades    : round-trip trade log
    positions : retained for compatibility with existing callers
    fills     : retained for compatibility; costs remain embedded in net equity
    """
    if metric_frequency == "daily":
        r = daily_portfolio_returns(
            equity,
            drop_incomplete_final_day=drop_incomplete_final_day,
        )
    elif metric_frequency == "bar":
        r = equity.pct_change().dropna()
    else:
        raise ValueError("metric_frequency must be 'bar' or 'daily'")
    if len(r) < 2:
        raise ValueError("At least two complete return observations are required")
    std = r.std(ddof=1)
    annual_rf, period_rf = _aligned_period_risk_free_returns(
        r.index,
        risk_free_rate,
        periods_per_year,
    )
    excess = r - period_rf
    excess_std = excess.std(ddof=1)
    ann_return = _annual_return(
        equity,
        periods_per_year,
        use_elapsed_time=annualize_by_elapsed_time,
    )
    mdd = max_drawdown(equity)
    all_episodes = drawdown_episodes(equity, min_depth=0.0)

    out = {
        "Total Return": equity.iloc[-1] / equity.iloc[0] - 1,
        "Annual Return": ann_return,
        "Annual Volatility": std * np.sqrt(periods_per_year),
        "Sharpe Ratio": (
            excess.mean() / excess_std * np.sqrt(periods_per_year)
            if excess_std > 0
            else np.nan
        ),
        "Max Drawdown": mdd,
        "Drawdown Count": len(all_episodes),
        "Risk Metric Return Frequency": metric_frequency,
        "Risk Metric Observations": len(r),
        "Risk Metric Periods per Year": periods_per_year,
        "Average Annual Risk-Free Rate": annual_rf.mean(),
        "Risk-Free Rate Source": risk_free_source or "caller-supplied assumption",
    }

    if trades is not None and len(trades):
        pnl = trades["net_pnl"]
        out.update({
            "Number of Trades": len(trades),
            "Win Rate": (pnl > 0).mean(),
            "Average Holding Period (hours)":
                trades["holding_period"].dt.total_seconds().mean() / 3600,
            "Profit per Trade": pnl.mean(),
        })
    else:
        out.update({"Number of Trades": 0, "Win Rate": np.nan,
                    "Average Holding Period (hours)": np.nan,
                    "Profit per Trade": np.nan})
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

import numpy as np
import pandas as pd
import pytest

from src.metrics import max_drawdown, drawdown_episodes, compute_metrics
from src.backtest import run_backtest, buy_and_hold
from src.strategy import hold_signals

IDX = pd.date_range("2026-01-01", periods=5, freq="30min")


def test_max_drawdown():
    eq = pd.Series([100, 120, 90, 130, 117], index=IDX, dtype=float)
    assert max_drawdown(eq) == pytest.approx(-0.25)


def test_drawdown_episodes():
    eq = pd.Series([100, 120, 90, 130, 117], index=IDX, dtype=float)
    eps = drawdown_episodes(eq, min_depth=0.0)
    assert len(eps) == 2
    assert list(eps["recovered"]) == [True, False]
    assert len(drawdown_episodes(eq, min_depth=0.20)) == 1


def test_trade_metrics():
    eq = pd.Series([100, 101, 102, 101, 103], index=IDX, dtype=float)
    trades = pd.DataFrame({
        "net_pnl": [10.0, -5.0, 3.0, -2.0],
        "return_pct": [0.1, -0.05, 0.03, -0.02],
        "fees": [0.1] * 4,
        "holding_period": pd.to_timedelta([1, 2, 3, 6], unit="h"),
    })
    m = compute_metrics(eq, trades)
    assert m["Number of Trades"] == 4
    assert m["Win Rate"] == pytest.approx(0.5)
    assert m["Average Holding Period (hours)"] == pytest.approx(3.0)
    assert m["Profit per Trade"] == pytest.approx(1.5)
    assert m["Profit Factor"] == pytest.approx(13 / 7)


def test_backtest_buy_and_hold_rising_market():
    idx = pd.date_range("2026-01-01", periods=100, freq="30min")
    px = pd.Series(np.linspace(100, 150, 100), index=idx)
    bars = pd.DataFrame({"open": px, "close": px})
    res = buy_and_hold(bars, fee_bps=0)
    assert len(res.trades) == 1
    assert res.equity.iloc[-1] == pytest.approx(150_000)


def test_hold_signals():
    sig = pd.Series([0, 1, 0, 0, 0, -1, 0, 0])
    pos = hold_signals(sig, horizon=3)
    assert list(pos) == [0, 1, 1, 1, 0, -1, -1, -1]


def test_backtest_short_in_falling_market():
    idx = pd.date_range("2026-01-01", periods=10, freq="30min")
    px = pd.Series(np.linspace(100, 90, 10), index=idx)
    bars = pd.DataFrame({"open": px, "close": px})
    res = run_backtest(pd.Series(-1.0, index=idx), bars, fee_bps=0)
    assert res.equity.iloc[-1] > 100_000
    assert res.trades.iloc[0]["side"] == "short"


def test_multi_asset_backtest_and_costs():
    from src.backtest import run_portfolio_backtest
    from src.metrics import cost_metrics, per_asset_breakdown
    idx = pd.date_range("2026-01-01", periods=20, freq="30min")
    up, down = pd.Series(np.linspace(100, 120, 20), idx), pd.Series(np.linspace(100, 80, 20), idx)
    bars = {"A": pd.DataFrame({"open": up, "close": up}), "B": pd.DataFrame({"open": down, "close": down})}
    pos = pd.DataFrame({"A": 1.0, "B": -1.0}, index=idx)
    res = run_portfolio_backtest(pos, bars, {"A": 0.5, "B": 0.5}, fee_bps=10, slippage_bps=5)
    assert set(res.trades["asset"]) == {"A", "B"}
    assert res.equity.iloc[-1] > 100_000                      # long the riser, short the faller
    # equity + all costs = what equity would have been at zero cost
    free = run_portfolio_backtest(pos, bars, {"A": 0.5, "B": 0.5}, fee_bps=0, slippage_bps=0)
    c = cost_metrics(res.equity, res.fills, res.trades)
    assert c["Total Transaction Cost"] == pytest.approx(res.fills["total_cost"].sum())
    assert c["Number of Orders"] == 4
    assert free.equity.iloc[-1] - res.equity.iloc[-1] == pytest.approx(c["Total Transaction Cost"], rel=0.05)
    pa = per_asset_breakdown(res.trades, res.fills)
    assert pa.loc["Total Cost"].sum() == pytest.approx(c["Total Transaction Cost"])


def test_missing_bar_for_one_asset():
    from src.backtest import run_portfolio_backtest
    idx = pd.date_range("2026-01-01", periods=6, freq="30min")
    px = pd.Series(100.0, idx)
    bars = {"A": pd.DataFrame({"open": px, "close": px}),
            "B": pd.DataFrame({"open": px, "close": px}).drop(idx[2])}
    pos = pd.DataFrame({"A": 1.0, "B": 1.0}, index=idx)
    res = run_portfolio_backtest(pos, bars, fee_bps=0)
    assert len(res.equity) == 6 and res.equity.iloc[-1] == pytest.approx(100_000)

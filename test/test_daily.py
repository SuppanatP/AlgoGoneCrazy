"""End-to-end: the multi-asset daily runner must match the one-shot portfolio backtest."""
from types import SimpleNamespace

import lightgbm as lgb
import numpy as np
import pandas as pd
import pytest

from run_daily import DailyRunner
from src.backtest import run_portfolio_backtest
from src.features import build_features
from src.records import read_records
from src.strategy import HoldStrategy, hold_signals, signals_from_probs
from src.tracker import CostModel

ASSETS = ["BTC", "ETH", "SOL"]


def test_hold_strategy_matches_batch():
    rng = np.random.default_rng(1)
    pu, pdn = pd.Series(rng.uniform(0, .3, 500)), pd.Series(rng.uniform(0, .3, 500))
    batch = hold_signals(signals_from_probs(pu, pdn, .2, .2, .03), horizon=12)
    st = HoldStrategy(.2, .2, .03, 12)
    assert list(batch) == [st.step(a, b)[1] for a, b in zip(pu, pdn)]


@pytest.fixture
def setup(tmp_path):
    rng = np.random.default_rng(0)
    idx = pd.date_range("2026-09-01", periods=48 * 6, freq="30min", tz="UTC")
    spec, frames = {}, {}
    for k, a in enumerate(ASSETS):
        close = (60_000 / (k + 1)) * np.exp(np.cumsum(rng.normal(0, 0.004, len(idx))))
        open_ = np.r_[close[0], close[:-1]]
        df = pd.DataFrame({"bar_open_time_utc": idx, "open": open_, "high": np.maximum(open_, close),
                           "low": np.minimum(open_, close), "close": close,
                           "volume": rng.uniform(10, 100, len(idx))})
        if a == "SOL":                       # a missing bar in the middle of a day
            df = df.drop(index=48 * 4 + 10)
        df.to_csv(tmp_path / f"{a}.csv", index=False)
        frames[a] = df.set_index("bar_open_time_utc")
        feats = build_features(frames[a]).dropna()
        for side in ("up", "down"):
            m = lgb.LGBMClassifier(n_estimators=20, verbose=-1, random_state=k)
            m.fit(feats, rng.integers(0, 2, len(feats)))
            m.booster_.save_model(str(tmp_path / f"{a}_{side}.txt"))
        spec[a] = {"csv": tmp_path / f"{a}.csv", "model_up": tmp_path / f"{a}_up.txt",
                   "model_down": tmp_path / f"{a}_down.txt", "weight": 1 / 3}
    spec["SOL"]["th_up"] = 0.55              # per-asset override

    cfg = SimpleNamespace(ASSETS=spec, TIME_COL="bar_open_time_utc", SYMBOL_COL="symbol",
                          BAR_FREQ="30min", LOOKBACK_DAYS=2, RECORDS_DIR=tmp_path / "records",
                          INITIAL_CASH=100_000.0, LEVERAGE=1.0, FEE_BPS=10.0, SLIPPAGE_BPS=3.0,
                          MIN_FEE=0.0, COST_OVERRIDES={"SOL": {"slippage_bps": 8.0}},
                          TH_UP=0.5, TH_DOWN=0.5, MIN_EDGE=0.0, HORIZON=12, ALLOW_SHORT=True)
    return cfg, frames


def test_daily_runs_match_backtest(setup):
    cfg, frames = setup
    for d in ["2026-09-04", "2026-09-05", "2026-09-06"]:   # new runner = state reloaded from disk
        DailyRunner(cfg).run_day(d)
    recs = read_records(cfg.RECORDS_DIR)

    runner = DailyRunner(cfg)
    period = {a: frames[a].loc["2026-09-04":"2026-09-06"] for a in ASSETS}
    pos = {}
    for a in ASSETS:
        up, down = runner.models[a]
        X = build_features(frames[a]).loc[period[a].index, up.feature_name()]
        th_up = cfg.ASSETS[a].get("th_up", cfg.TH_UP)
        pos[a] = hold_signals(signals_from_probs(pd.Series(up.predict(X), X.index),
                                                 pd.Series(down.predict(X), X.index),
                                                 th_up, cfg.TH_DOWN, cfg.MIN_EDGE), cfg.HORIZON)
    costs = CostModel(cfg.FEE_BPS, cfg.SLIPPAGE_BPS, 0.0, cfg.COST_OVERRIDES)
    bt = run_portfolio_backtest(pd.DataFrame(pos), period, {a: 1 / 3 for a in ASSETS},
                                cost_model=costs)

    assert len(recs["equity"]) == len(bt.equity)
    # identical except the last bar, where the backtest force-closes everything
    np.testing.assert_allclose(recs["equity"]["equity"].iloc[:-1].values,
                               bt.equity.iloc[:-1].values, rtol=1e-9)
    last = bt.equity.index[-1]
    bt_cost = bt.fills.loc[bt.fills["timestamp"] < last, "total_cost"].sum()
    live_cost = recs["fills"].loc[recs["fills"].index < last, "total_cost"].sum()
    assert live_cost == pytest.approx(bt_cost)
    assert (recs["fills"].query("asset == 'SOL'")["slippage_cost"]
            / recs["fills"].query("asset == 'SOL'")["notional"]).max() > 7e-4  # SOL override used


def test_rerunning_a_day_adds_nothing(setup):
    cfg, _ = setup
    assert DailyRunner(cfg).run_day("2026-09-04") == 48
    assert DailyRunner(cfg).run_day("2026-09-04") == 0
    assert len(read_records(cfg.RECORDS_DIR)["decisions"]) == 48 * 3

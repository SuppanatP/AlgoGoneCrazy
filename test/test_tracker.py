import pandas as pd
import pytest

from src.tracker import Portfolio

T0, T1, T2 = pd.Timestamp("2026-01-01 00:00"), pd.Timestamp("2026-01-01 06:00"), pd.Timestamp("2026-01-02 00:00")


def test_long_round_trip_no_fees():
    pf = Portfolio(1_000, fee_bps=0)
    pf.buy(T0, "BTC", 1, 100)
    pf.sell(T1, "BTC", 1, 110)
    t = pf.trades.iloc[0]
    assert pf.cash == pytest.approx(1_010)
    assert t["side"] == "long" and t["net_pnl"] == pytest.approx(10)
    assert t["holding_period"] == pd.Timedelta(hours=6)
    assert not pf.positions


def test_fees_reduce_pnl():
    pf = Portfolio(10_000, fee_bps=10)  # 0.10%
    pf.buy(T0, "BTC", 10, 100)          # fee 1.0
    pf.sell(T1, "BTC", 10, 110)         # fee 1.1
    t = pf.trades.iloc[0]
    assert t["pnl_before_costs"] == pytest.approx(100)
    assert t["fees"] == pytest.approx(2.1)
    assert t["net_pnl"] == pytest.approx(97.9)
    assert pf.cash == pytest.approx(10_097.9)


def test_short_round_trip():
    pf = Portfolio(1_000, fee_bps=0)
    pf.sell(T0, "BTC", 2, 100)
    pf.buy(T1, "BTC", 2, 90)
    t = pf.trades.iloc[0]
    assert t["side"] == "short" and t["net_pnl"] == pytest.approx(20)


def test_flip_long_to_short_creates_two_trades():
    pf = Portfolio(1_000, fee_bps=0)
    pf.buy(T0, "BTC", 1, 100)
    pf.sell(T1, "BTC", 3, 110)   # close long (+10), open short 2 @ 110
    assert pf.position("BTC") == pytest.approx(-2)
    pf.buy(T2, "BTC", 2, 100)    # close short (+20)
    tr = pf.trades
    assert list(tr["side"]) == ["long", "short"]
    assert list(tr["net_pnl"]) == pytest.approx([10, 20])


def test_partial_exits_are_one_trade():
    pf = Portfolio(1_000, fee_bps=0)
    pf.buy(T0, "BTC", 2, 100)
    pf.sell(T1, "BTC", 1, 110)
    assert len(pf.trades) == 0
    pf.sell(T2, "BTC", 1, 120)
    t = pf.trades.iloc[0]
    assert t["net_pnl"] == pytest.approx(30)
    assert t["exit_price"] == pytest.approx(115)


def test_mark_records_equity():
    pf = Portfolio(1_000, fee_bps=0)
    pf.buy(T0, "BTC", 5, 100)
    pf.mark(T0, {"BTC": 104})
    assert pf.equity_curve["equity"].iloc[-1] == pytest.approx(1_020)


def test_target_weight():
    pf = Portfolio(1_000, fee_bps=0)
    pf.target_weight(T0, "BTC", 0.5, 100)
    assert pf.position("BTC") == pytest.approx(5)
    pf.target_weight(T1, "BTC", -1.0, 100)
    assert pf.position("BTC") == pytest.approx(-10)


def test_slippage_and_fee_accounting():
    from src.tracker import CostModel
    pf = Portfolio(10_000, cost_model=CostModel(fee_bps=10, slippage_bps=50))
    f1 = pf.buy(T0, "BTC", 10, 100)      # fills at 100.5
    f2 = pf.sell(T1, "BTC", 10, 110)     # fills at 109.45
    assert f1.price == pytest.approx(100.5) and f2.price == pytest.approx(109.45)
    assert f1.slippage_cost == pytest.approx(5.0) and f2.slippage_cost == pytest.approx(5.5)
    t = pf.trades.iloc[0]
    assert t["pnl_before_costs"] == pytest.approx(100)          # at reference prices
    assert t["slippage"] == pytest.approx(10.5)
    assert t["fees"] == pytest.approx(1.005 + 1.0945)
    assert t["net_pnl"] == pytest.approx(100 - t["slippage"] - t["fees"])
    assert pf.cash - 10_000 == pytest.approx(t["net_pnl"])     # cash agrees with trade log
    assert pf.cum_fees + pf.cum_slippage == pytest.approx(t["total_cost"])


def test_per_asset_costs_and_min_fee():
    from src.tracker import CostModel
    cm = CostModel(fee_bps=10, slippage_bps=0, min_fee=2.0, per_asset={"SOL": {"slippage_bps": 100}})
    pf = Portfolio(10_000, cost_model=cm)
    f = pf.buy(T0, "BTC", 1, 100)                  # fee 0.1 -> min fee 2
    assert f.fee == pytest.approx(2.0) and f.slippage_cost == 0
    f = pf.buy(T0, "SOL", 1, 100)
    assert f.price == pytest.approx(101) and f.slippage_cost == pytest.approx(1.0)


def test_state_round_trip_multi_asset():
    pf = Portfolio(1_000, fee_bps=10)
    pf.buy(T0, "BTC", 1, 100)
    pf.sell(T0, "ETH", 2, 50)
    pf2 = Portfolio.from_state(pf.to_state())
    assert pf2.position("BTC") == 1 and pf2.position("ETH") == -2
    assert pf2.cash == pytest.approx(pf.cash) and pf2.cum_fees == pytest.approx(pf.cum_fees)
    pf2.sell(T1, "BTC", 1, 110)
    assert pf2.trades.iloc[0]["net_pnl"] == pytest.approx(10 - 0.1 - 0.11)

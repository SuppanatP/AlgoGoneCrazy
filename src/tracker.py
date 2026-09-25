"""Trade tracker: multi-asset cash + positions ledger with transaction costs.

Quantities are signed: +qty buys, -qty sells. Shorts are allowed (negative position).
A "trade" (round trip) opens when a position goes from flat to non-zero and closes
when it returns to flat. Adding to a position extends the same trade; a sell larger
than the long position closes the long and opens a new short (a "flip").

Transaction costs have two parts, both recorded on every fill:
  fee       exchange commission = fee_bps x traded notional (at least min_fee)
  slippage  price penalty = slippage_bps x reference price (buys fill higher, sells
            lower); covers half the bid-ask spread plus market impact
"""
from __future__ import annotations

from dataclasses import dataclass, field, asdict
from typing import Optional

import pandas as pd

EPS = 1e-12


@dataclass
class CostModel:
    fee_bps: float = 10.0        # 10 bps = 0.10% of notional per order
    slippage_bps: float = 0.0
    min_fee: float = 0.0         # minimum commission per order (in cash)
    per_asset: dict = field(default_factory=dict)   # {"SOL": {"slippage_bps": 5}}

    def params(self, asset: str) -> tuple[float, float, float]:
        o = self.per_asset.get(asset, {})
        return (o.get("fee_bps", self.fee_bps), o.get("slippage_bps", self.slippage_bps),
                o.get("min_fee", self.min_fee))

    def apply(self, asset: str, qty: float, ref_price: float) -> tuple[float, float, float]:
        """Returns (execution price, fee, slippage cost) for a signed order."""
        fee_bps, slip_bps, min_fee = self.params(asset)
        direction = 1.0 if qty > 0 else -1.0
        exec_price = ref_price * (1 + direction * slip_bps / 1e4)
        slippage_cost = abs(qty) * abs(exec_price - ref_price)
        fee = max(abs(qty) * exec_price * fee_bps / 1e4, min_fee)
        return exec_price, fee, slippage_cost


@dataclass
class Fill:
    timestamp: pd.Timestamp
    asset: str
    qty: float            # + buy, - sell
    ref_price: float      # price before costs (e.g. the bar open)
    price: float          # execution price after slippage
    notional: float       # |qty| x execution price
    fee: float
    slippage_cost: float
    total_cost: float


@dataclass
class _OpenTrade:
    asset: str
    side: int              # +1 long, -1 short
    qty: float             # current signed quantity
    avg_price: float       # average execution price of the current quantity
    entry_time: pd.Timestamp
    opened_qty: float = 0.0
    entry_notional: float = 0.0
    exit_qty: float = 0.0
    exit_notional: float = 0.0
    realized_pnl: float = 0.0   # at execution prices (already net of slippage), before fees
    fees: float = 0.0
    slippage: float = 0.0


class Portfolio:
    """Multi-asset portfolio. Pass a CostModel, or just fee_bps / slippage_bps."""

    def __init__(self, initial_cash: float = 100_000.0, fee_bps: float = 10.0,
                 slippage_bps: float = 0.0, cost_model: Optional[CostModel] = None):
        self.initial_cash = float(initial_cash)
        self.cash = float(initial_cash)
        self.costs = cost_model or CostModel(fee_bps, slippage_bps)
        self.positions: dict[str, _OpenTrade] = {}
        self.last_price: dict[str, float] = {}
        self.cum_fees = 0.0
        self.cum_slippage = 0.0
        self._fills: list[Fill] = []
        self._trades: list[dict] = []
        self._equity: list[dict] = []

    # ------------------------------------------------------------------ state
    def position(self, asset: str) -> float:
        pos = self.positions.get(asset)
        return pos.qty if pos else 0.0

    def equity(self, prices: Optional[dict] = None) -> float:
        """Cash plus market value of positions, using `prices` or the last known prices."""
        px = {**self.last_price, **(prices or {})}
        value = self.cash
        for asset, pos in self.positions.items():
            if asset not in px:
                raise KeyError(f"No price available for {asset}")
            value += pos.qty * px[asset]
        return value

    # ---------------------------------------------------------------- trading
    def order(self, timestamp, asset: str, qty: float, price: float) -> Optional[Fill]:
        """Execute a market order of signed `qty` at reference `price`."""
        if abs(qty) < EPS:
            return None
        exec_price, fee, slip = self.costs.apply(asset, qty, price)
        self.cash -= qty * exec_price + fee
        self.cum_fees += fee
        self.cum_slippage += slip
        self.last_price[asset] = float(price)
        fill = Fill(pd.Timestamp(timestamp), asset, float(qty), float(price), float(exec_price),
                    abs(qty) * exec_price, float(fee), float(slip), float(fee + slip))
        self._fills.append(fill)
        self._apply(fill)
        return fill

    def buy(self, timestamp, asset: str, qty: float, price: float) -> Optional[Fill]:
        return self.order(timestamp, asset, abs(qty), price)

    def sell(self, timestamp, asset: str, qty: float, price: float) -> Optional[Fill]:
        return self.order(timestamp, asset, -abs(qty), price)

    def target_position(self, timestamp, asset: str, target_qty: float, price: float):
        """Trade whatever is needed to hold exactly `target_qty` units."""
        return self.order(timestamp, asset, target_qty - self.position(asset), price)

    def target_weight(self, timestamp, asset: str, weight: float, price: float,
                      equity: Optional[float] = None):
        """Hold `weight` x equity in `asset` (negative = short)."""
        eq = self.equity({asset: price}) if equity is None else equity
        return self.target_position(timestamp, asset, weight * eq / price, price)

    def close_all(self, timestamp, prices: dict):
        for asset in list(self.positions):
            self.target_position(timestamp, asset, 0.0, prices[asset])

    def mark(self, timestamp, prices: dict):
        """Record portfolio value at `timestamp` (call once per bar, at the close).
        Assets without a price in `prices` are valued at their last known price."""
        self.last_price.update({k: float(v) for k, v in prices.items()})
        eq = self.equity()
        gross = sum(abs(p.qty * self.last_price[a]) for a, p in self.positions.items())
        row = {"timestamp": pd.Timestamp(timestamp), "equity": eq, "cash": self.cash,
               "gross_exposure": gross / eq if eq else float("nan"),
               "cum_fees": self.cum_fees, "cum_slippage": self.cum_slippage}
        for asset in sorted(set(self.last_price) | set(self.positions)):
            row[f"pos_{asset}"] = self.position(asset)
        self._equity.append(row)

    # --------------------------------------------------------------- internals
    def _apply(self, fill: Fill):
        qty, fee_left, slip_left = fill.qty, fill.fee, fill.slippage_cost
        pos = self.positions.get(fill.asset)

        # 1) reduce / close an existing position in the opposite direction
        if pos is not None and pos.qty * qty < 0:
            close = min(abs(qty), abs(pos.qty))
            share = close / abs(fill.qty)
            signed_close = close if pos.qty > 0 else -close
            pos.realized_pnl += signed_close * (fill.price - pos.avg_price)
            pos.fees += fill.fee * share
            pos.slippage += fill.slippage_cost * share
            pos.exit_qty += close
            pos.exit_notional += close * fill.price
            pos.qty -= signed_close
            qty += signed_close
            fee_left -= fill.fee * share
            slip_left -= fill.slippage_cost * share
            if abs(pos.qty) < EPS:
                self._record_trade(pos, fill.timestamp)
                del self.positions[fill.asset]
                pos = None

        if abs(qty) < EPS:
            return

        # 2) open a new trade, or add to the existing one in the same direction
        if pos is None:
            self.positions[fill.asset] = _OpenTrade(
                asset=fill.asset, side=1 if qty > 0 else -1, qty=qty,
                avg_price=fill.price, entry_time=fill.timestamp,
                opened_qty=abs(qty), entry_notional=abs(qty) * fill.price,
                fees=fee_left, slippage=slip_left)
        else:
            new_qty = pos.qty + qty
            pos.avg_price = (pos.avg_price * pos.qty + fill.price * qty) / new_qty
            pos.qty = new_qty
            pos.opened_qty += abs(qty)
            pos.entry_notional += abs(qty) * fill.price
            pos.fees += fee_left
            pos.slippage += slip_left

    def _record_trade(self, pos: _OpenTrade, exit_time: pd.Timestamp):
        net = pos.realized_pnl - pos.fees
        self._trades.append({
            "asset": pos.asset,
            "side": "long" if pos.side > 0 else "short",
            "entry_time": pos.entry_time,
            "exit_time": exit_time,
            "qty": pos.opened_qty,
            "entry_price": pos.entry_notional / pos.opened_qty,
            "exit_price": pos.exit_notional / pos.exit_qty,
            "pnl_before_costs": pos.realized_pnl + pos.slippage,
            "fees": pos.fees,
            "slippage": pos.slippage,
            "total_cost": pos.fees + pos.slippage,
            "net_pnl": net,
            "return_pct": net / pos.entry_notional,
            "holding_period": exit_time - pos.entry_time,
        })

    # ------------------------------------------------------------- persistence
    def to_state(self) -> dict:
        """JSON-serialisable snapshot (history is kept in the CSV records, not here)."""
        positions = {}
        for asset, p in self.positions.items():
            d = asdict(p)
            d["entry_time"] = pd.Timestamp(p.entry_time).isoformat()
            positions[asset] = d
        return {"initial_cash": self.initial_cash, "cash": self.cash,
                "costs": asdict(self.costs), "cum_fees": self.cum_fees,
                "cum_slippage": self.cum_slippage, "last_price": dict(self.last_price),
                "positions": positions}

    @classmethod
    def from_state(cls, state: dict, cost_model: Optional[CostModel] = None) -> "Portfolio":
        """Restore a portfolio. Pass `cost_model` to use current settings instead of saved ones."""
        pf = cls(state["initial_cash"], cost_model=cost_model or CostModel(**state["costs"]))
        pf.cash = float(state["cash"])
        pf.cum_fees = float(state["cum_fees"])
        pf.cum_slippage = float(state["cum_slippage"])
        pf.last_price = {k: float(v) for k, v in state["last_price"].items()}
        for asset, d in state["positions"].items():
            d = dict(d)
            d["entry_time"] = pd.Timestamp(d["entry_time"])
            pf.positions[asset] = _OpenTrade(**d)
        return pf

    # ----------------------------------------------------------------- outputs
    @property
    def fills(self) -> pd.DataFrame:
        cols = list(Fill.__dataclass_fields__)
        return pd.DataFrame([asdict(f) for f in self._fills], columns=cols)

    @property
    def trades(self) -> pd.DataFrame:
        cols = ["asset", "side", "entry_time", "exit_time", "qty", "entry_price", "exit_price",
                "pnl_before_costs", "fees", "slippage", "total_cost", "net_pnl", "return_pct",
                "holding_period"]
        return pd.DataFrame(self._trades, columns=cols)

    @property
    def equity_curve(self) -> pd.DataFrame:
        if not self._equity:
            return pd.DataFrame(columns=["equity", "cash", "gross_exposure", "cum_fees",
                                         "cum_slippage"])
        return pd.DataFrame(self._equity).set_index("timestamp").fillna(0.0)

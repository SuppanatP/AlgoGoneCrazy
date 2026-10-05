"""Transaction cost and FX-conversion assumptions.

Interest / cost of capital / funding is assumed zero and not modeled
(see cost_assumptions.json -> decision.interest_and_funding).

Single source of truth for the numbers used by the trade-log / backtest layer
(see trades_log_design.md in the project doc). Values live in
config/cost_assumptions.json so they can be regenerated into a presentation
table (export_cost_table.py) without drifting from what the code actually
uses. Update the JSON, not this file, when a rate changes.

Two commission shapes are supported, matching what real brokers charge:
- pct-of-trade-value with a minimum (STI ETF, BTC, and the old IBKR Lite
  assumption for SP500_ETF/TLT).
- per-share with a minimum and a cap as % of trade value (IBKR Pro Tiered /
  Pro Fixed -- the current SP500_ETF/TLT assumption under the fund/investor
  pitch framing; see cost_assumptions.json -> "decision").
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

DEFAULT_CONFIG_PATH = Path(__file__).resolve().parents[2] / "config" / "cost_assumptions.json"


@dataclass(frozen=True)
class MinFee:
    amount: float
    currency: str


@dataclass(frozen=True)
class AssetCost:
    asset_id: str
    broker: str
    commission_pct: Optional[float] = None
    commission_min: Optional[MinFee] = None
    commission_per_share: Optional[float] = None
    cap_pct_of_trade_value: Optional[float] = None
    notes: str = ""

    def commission(self, trade_value_native: float, shares: Optional[float] = None) -> float:
        """Commission in the asset's native currency for one trade.

        `shares` is required when this asset uses a per-share pricing plan
        (commission_per_share is set) -- e.g. IBKR Pro Tiered/Fixed. For a
        pct-of-trade-value plan (STI ETF, BTC, IBKR Lite), `shares` is ignored.
        """
        trade_value_native = abs(trade_value_native)

        if self.commission_per_share is not None:
            if shares is None:
                raise ValueError(
                    f"{self.asset_id} uses a per-share commission plan; pass `shares`"
                )
            fee = abs(shares) * self.commission_per_share
            if self.commission_min is not None:
                fee = max(fee, self.commission_min.amount)
            if self.cap_pct_of_trade_value is not None:
                fee = min(fee, trade_value_native * self.cap_pct_of_trade_value)
            return fee

        fee = trade_value_native * (self.commission_pct or 0.0)
        if self.commission_min is not None:
            fee = max(fee, self.commission_min.amount)
        return fee


@dataclass(frozen=True)
class FxConversion:
    pair: str
    broker: str
    applies_to: tuple[str, ...]
    fee_pct: float
    fee_min: Optional[MinFee]

    def fee(self, asset_id: str, notional_usd: float) -> float:
        """FX conversion fee in USD; 0 if this asset doesn't convert currency."""
        if asset_id not in self.applies_to:
            return 0.0
        fee = abs(notional_usd) * self.fee_pct
        if self.fee_min is not None:
            fee = max(fee, self.fee_min.amount)
        return fee


@dataclass(frozen=True)
class CostAssumptions:
    as_of: str
    assets: dict[str, AssetCost]
    fx: FxConversion
    sources: dict[str, str]


def _min_fee(raw: Optional[dict]) -> Optional[MinFee]:
    if raw is None:
        return None
    return MinFee(amount=raw["amount"], currency=raw["currency"])


def load_cost_assumptions(path: Path = DEFAULT_CONFIG_PATH) -> CostAssumptions:
    raw = json.loads(Path(path).read_text())

    assets = {
        asset_id: AssetCost(
            asset_id=asset_id,
            broker=entry["broker"],
            commission_pct=entry.get("commission_pct"),
            commission_min=_min_fee(entry.get("commission_min")),
            commission_per_share=entry.get("commission_per_share"),
            cap_pct_of_trade_value=entry.get("cap_pct_of_trade_value"),
            notes=entry.get("notes", ""),
        )
        for asset_id, entry in raw["assets"].items()
    }

    fx_raw = raw["fx_conversion"]
    fx = FxConversion(
        pair=fx_raw["pair"],
        broker=fx_raw["broker"],
        applies_to=tuple(fx_raw["applies_to"]),
        fee_pct=fx_raw["fee_pct"],
        fee_min=_min_fee(fx_raw.get("fee_min")),
    )

    return CostAssumptions(
        as_of=raw["as_of"],
        assets=assets,
        fx=fx,
        sources=raw["sources"],
    )

# Assets that trade in whole shares (round down to the nearest integer).
# Everything else (BTC) trades in fractional units.
WHOLE_SHARE_ASSETS = frozenset({"STI_ETF", "SP500_ETF", "TLT"})


def position_size(asset_id: str, allocated_capital: float, price: float) -> float:
    """Fixed-capital-allocation position sizing (the project's locked-in convention).

    Fully invested when a signal is long/flat: shares = floor(allocated_capital / price)
    for whole-share assets (SP500_ETF, TLT, STI_ETF); BTC uses fractional units.
    See config/cost_assumptions.json -> "decision" for the rationale.
    """
    if price <= 0:
        raise ValueError(f"price must be positive, got {price!r}")
    raw_units = allocated_capital / price
    if asset_id in WHOLE_SHARE_ASSETS:
        return float(math.floor(raw_units))
    return raw_units

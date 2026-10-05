"""Regenerate a presentation-ready Markdown table from config/cost_assumptions.json.

Run this after editing the JSON so the slide/report table and the code
never drift apart:

    python export_cost_table.py > docs/cost_table.md

Paste the printed Markdown straight into the write-up or slides.
"""

from __future__ import annotations

from src.market_data.costs import load_cost_assumptions


def _fmt_min(min_fee) -> str:
    if min_fee is None:
        return "-"
    return f"{min_fee.currency} {min_fee.amount:.2f}"


def _fmt_commission(a) -> str:
    """Render either pricing shape: pct-of-trade-value, or per-share + cap."""
    if a.commission_per_share is not None:
        cap_note = (
            f", capped at {a.cap_pct_of_trade_value * 100:.2f}% of trade value"
            if a.cap_pct_of_trade_value
            else ""
        )
        return f"USD {a.commission_per_share:.4f}/share{cap_note}"
    if a.commission_pct:
        return f"{a.commission_pct * 100:.4f}".rstrip("0").rstrip(".") + "%"
    return "$0"


def main() -> None:
    costs = load_cost_assumptions()

    print(f"_Assumptions as of {costs.as_of}_\n")
    print("| Asset | Broker | Commission | Minimum | Notes |")
    print("|---|---|---|---|---|")
    for asset_id, a in costs.assets.items():
        print(f"| {asset_id} | {a.broker} | {_fmt_commission(a)} | {_fmt_min(a.commission_min)} | {a.notes} |")

    print()
    print("| FX pair | Broker | Fee | Minimum | Applies to |")
    print("|---|---|---|---|---|")
    fx = costs.fx
    print(
        f"| {fx.pair} | {fx.broker} | {fx.fee_pct * 100:.4f}% | {_fmt_min(fx.fee_min)} "
        f"| {', '.join(fx.applies_to)} |"
    )

    print()
    print("_Interest / cost of capital / funding: assumed 0, not modeled._")

if __name__ == "__main__":
    main()

"""Daily runner: replay one or more days bar by bar for all assets and keep the records.

For each 30-min bar of the day, in time order:
  for every asset: features (data through t-1) -> p_up, p_down -> strategy target
  -> buy / sell at the bar's open for assets whose target changed (fee + slippage charged)
  -> mark the whole portfolio at the close -> write rows to records/.
State (cash, open positions, holding clocks, cumulative costs) is saved after every
day, so the next run continues where this one stopped. Bars already processed are
skipped, so running the same day twice is safe.

Usage:
  python run_daily.py                              # yesterday (UTC)
  python run_daily.py --date 2026-09-24
  python run_daily.py --start 2026-03-22 --end 2026-09-01   # replay a whole period
  python run_daily.py --records-dir records/btc_pre_october --start 2026-06-01 --end 2026-09-30
  python run_daily.py --reset --start 2026-10-05   # start trading on Oct 5, run to the latest
                                                   # data (history before it only warms up)
  python run_daily.py --reset --start ...          # delete records and start fresh
"""
from __future__ import annotations

import argparse
import shutil
from pathlib import Path

import numpy as np
import pandas as pd

import config
from src.backtest import execute_bar
from src.data_source import fetch_bars, history_problem
from src.btc_targets import (
    INDICATOR_COLS,
    generate_target_row,
    load_model_pair,
    prediction_frame,
)
from src.records import append_csv, load_state, save_state
from src.strategy import RegimeStrategy, VotingStrategy
from src.tracker import CostModel, Portfolio

STRATEGIES = {"regime": RegimeStrategy, "voting": VotingStrategy}


def cost_model_from(cfg) -> CostModel:
    return CostModel(cfg.FEE_BPS, cfg.SLIPPAGE_BPS, getattr(cfg, "MIN_FEE", 0.0),
                     dict(getattr(cfg, "COST_OVERRIDES", {})))


def strategy_class(cfg):
    name = getattr(cfg, "STRATEGY", "regime")
    if name not in STRATEGIES:
        raise ValueError(f"STRATEGY must be one of {sorted(STRATEGIES)}, got {name!r}")
    return STRATEGIES[name]


def strategy_params(cfg, asset: str) -> dict:
    defaults = {"use_vol_model": getattr(cfg, "USE_VOL_MODEL", True),
                "vol_high": cfg.VOL_HIGH, "horizon": cfg.HORIZON, "allow_short": cfg.ALLOW_SHORT,
                "exit_on_opposite": getattr(cfg, "EXIT_ON_OPPOSITE", True),
                "stop_loss_atr": getattr(cfg, "STOP_LOSS_ATR", 0.0),
                "take_profit_atr": getattr(cfg, "TAKE_PROFIT_ATR", 0.0),
                "partial_tp_atr": getattr(cfg, "PARTIAL_TP_ATR", 0.0),
                "partial_sl_atr": getattr(cfg, "PARTIAL_SL_ATR", 0.0),
                "partial_fraction": getattr(cfg, "PARTIAL_FRACTION", 0.5),
                "daily_fallback": getattr(cfg, "DAILY_FALLBACK", False),
                "fallback_hour": getattr(cfg, "FALLBACK_HOUR", 20),
                "fallback_size": getattr(cfg, "FALLBACK_SIZE", 0.1),
                "fallback_horizon": getattr(cfg, "FALLBACK_HORIZON", 12)}
    if strategy_class(cfg) is VotingStrategy:
        defaults |= {"weights": dict(cfg.RULE_WEIGHTS), "min_score": cfg.MIN_SCORE,
                     "partial_min_score": getattr(cfg, "PARTIAL_MIN_SCORE", 0.0),
                     "partial_vol_high": getattr(cfg, "PARTIAL_VOL_HIGH", None),
                     "partial_entry_size": getattr(cfg, "PARTIAL_ENTRY_SIZE", 0.5),
                     "params": dict(getattr(cfg, "RULE_PARAMS", {}))}
    else:
        defaults |= {"vol_low": cfg.VOL_LOW, "mom_rsi": cfg.MOM_RSI,
                     "mr_rsi": cfg.MR_RSI, "mr_bb": cfg.MR_BB}
    return {k: cfg.ASSETS[asset].get(k, v) for k, v in defaults.items()}


class DailyRunner:
    def __init__(self, cfg=config):
        self.cfg = cfg
        self.assets = list(cfg.ASSETS)
        self.weights = {a: cfg.ASSETS[a]["weight"] for a in self.assets}

        self.models = {}
        for a in self.assets:
            self.models[a] = load_model_pair(
                cfg.ASSETS[a]["model_up"], cfg.ASSETS[a]["model_down"]
            )

        costs = cost_model_from(cfg)
        Strategy = strategy_class(cfg)
        state = load_state(cfg.RECORDS_DIR)
        if state is None:
            self.pf = Portfolio(cfg.INITIAL_CASH, cost_model=costs)
            self.strategies = {a: Strategy(**strategy_params(cfg, a)) for a in self.assets}
            self.prev_targets = {a: 0.0 for a in self.assets}
            self.last_bar_time = {a: None for a in self.assets}
        else:
            if set(state["strategies"]) != set(self.assets):
                raise ValueError(f"records were made for {sorted(state['strategies'])}, "
                                 f"config has {sorted(self.assets)}; use --reset")
            self.pf = Portfolio.from_state(state["portfolio"], cost_model=costs)
            if state["portfolio"]["costs"] != Portfolio(cost_model=costs).to_state()["costs"]:
                print("  warning: transaction-cost settings changed since last run")
            self.strategies, self.prev_targets, self.last_bar_time = {}, {}, {}
            for a in self.assets:
                saved = state["strategies"][a]
                params = strategy_params(cfg, a)
                changed = [k for k in params if saved.get(k) != params[k]]
                if changed:
                    print(f"  warning: {a} strategy settings changed since last run: {changed}")
                self.strategies[a] = Strategy(**params, **{k: saved[k] for k in Strategy.STATE
                                                           if k in saved})
                self.prev_targets[a] = float(state["prev_targets"][a])
                t = state["last_bar_time"][a]
                self.last_bar_time[a] = pd.Timestamp(t) if t else None

    # ------------------------------------------------------------------------
    def _load_day(self, day_start, day_end):
        cfg, hist, today = self.cfg, {}, {}
        for a in self.assets:
            spec = cfg.ASSETS[a]
            hist[a] = fetch_bars(day_start - pd.Timedelta(days=cfg.LOOKBACK_DAYS), day_end,
                                 spec["csv"], cfg.TIME_COL, spec.get("symbol"),
                                 getattr(cfg, "SYMBOL_COL", "symbol"))
            bars = hist[a][hist[a].index >= day_start]
            if self.last_bar_time[a] is not None:
                bars = bars[bars.index > self.last_bar_time[a]]
            today[a] = bars
        return hist, today

    def _predict(self, asset, hist, bars_today):
        frame = prediction_frame(hist[asset], bars_today, asset, self.models[asset], hist)
        return frame["p_up"], frame["p_down"], frame[list(INDICATOR_COLS)]

    def run_day(self, day) -> int:
        """Process every not-yet-processed bar of `day` (UTC). Returns bars processed."""
        cfg = self.cfg
        day_start = pd.Timestamp(day)
        day_start = (day_start.tz_localize("UTC") if day_start.tzinfo is None
                     else day_start.tz_convert("UTC")).normalize()
        day_end = day_start + pd.Timedelta(days=1)

        hist, today = self._load_day(day_start, day_end)
        active = [a for a in self.assets if not today[a].empty]
        if not active:
            print(f"{day_start.date()}: no new bars")
            return 0
        for a in active:
            self._check_gap(a, today[a].index[0])
            problem = history_problem(hist[a], today[a].index[0], cfg.WARMUP_DAYS, cfg.BAR_FREQ)
            if problem:
                raise SystemExit(f"{a}: not enough history to trade from {today[a].index[0]} "
                                 f"(need {cfg.WARMUP_DAYS} days without gaps): {problem}. "
                                 f"Add the missing bars to the CSVs in config.ASSETS first.")

        timeline = today[active[0]].index
        for a in active[1:]:
            timeline = timeline.union(today[a].index)
        data = {}
        for a in active:
            p_up, p_down, ind = self._predict(a, hist, today[a])
            df = today[a][["open", "close"]].assign(p_up=p_up, p_down=p_down).join(ind)
            data[a] = df.reindex(timeline)

        n_fills, n_trades, n_eq = len(self.pf._fills), len(self.pf._trades), len(self.pf._equity)
        decisions = []
        for ts in timeline:
            targets, opens, closes, diagnostics = {}, {}, {}, {}
            for a in active:
                row = data[a].loc[ts]
                if np.isnan(row["open"]):
                    continue                      # no bar for this asset at ts
                diagnostics[a] = generate_target_row(self.strategies[a], row, ts)
                targets[a] = diagnostics[a]["execution_target_position"]
                opens[a], closes[a] = float(row["open"]), float(row["close"])
            before = {a: self.pf.position(a) for a in targets}
            fills = execute_bar(self.pf, ts, targets, self.prev_targets, opens, closes,
                                self.weights, cfg.LEVERAGE)
            self.prev_targets.update(targets)
            cost = {f.asset: f.total_cost for f in fills}
            equity = self.pf.equity()
            for a in targets:
                after = self.pf.position(a)
                action = ("buy" if after > before[a] + 1e-12 else
                          "sell" if after < before[a] - 1e-12 else "hold")
                decisions.append({"timestamp": ts, "asset": a, "open": opens[a],
                                  "close": closes[a], "p_up": data[a].at[ts, "p_up"],
                                  "p_down": data[a].at[ts, "p_down"],
                                  **{k: data[a].at[ts, k] for k in INDICATOR_COLS},
                                  **{k: v for k, v in diagnostics[a].items()
                                     if k.startswith("vote_")},
                                  "signal": diagnostics[a]["entry_signal"],
                                  "target": targets[a], "action": action,
                                  "signal_reason": diagnostics[a]["signal_reason"],
                                  "position_qty": after, "transaction_cost": cost.get(a, 0.0),
                                  "portfolio_equity": equity})
        for a in active:
            self.last_bar_time[a] = today[a].index[-1]

        # append only what happened today
        append_csv(cfg.RECORDS_DIR, "decisions", pd.DataFrame(decisions))
        append_csv(cfg.RECORDS_DIR, "fills", self.pf.fills.iloc[n_fills:])
        append_csv(cfg.RECORDS_DIR, "trades", self.pf.trades.iloc[n_trades:])
        eq_cols = (["timestamp", "equity", "cash", "gross_exposure", "cum_fees", "cum_slippage"]
                   + [f"pos_{a}" for a in self.assets])
        eq = self.pf.equity_curve.iloc[n_eq:].reset_index().reindex(columns=eq_cols).fillna(0.0)
        append_csv(cfg.RECORDS_DIR, "equity", eq)
        self._save()

        day_cost = sum(f.total_cost for f in self.pf._fills[n_fills:])
        pos = ", ".join(f"{a} {self.pf.position(a):+.4f}" for a in self.assets)
        print(f"{day_start.date()}: {len(timeline)} bars, {len(self.pf._fills) - n_fills} orders, "
              f"cost {day_cost:,.2f}, equity {self.pf.equity():,.2f} | {pos}")
        return len(timeline)

    # ------------------------------------------------------------------------
    def _check_gap(self, asset, first_new_bar):
        last = self.last_bar_time[asset]
        if last is not None and first_new_bar > last + pd.Timedelta(self.cfg.BAR_FREQ):
            print(f"  warning: {asset} gap — last processed {last}, next bar {first_new_bar}. "
                  f"Run the missing days first with --start/--end.")

    def _save(self):
        save_state(self.cfg.RECORDS_DIR, {
            "assets": self.assets,
            "portfolio": self.pf.to_state(),
            "strategies": {a: s.to_state() for a, s in self.strategies.items()},
            "prev_targets": self.prev_targets,
            "last_bar_time": {a: (t.isoformat() if t is not None else None)
                              for a, t in self.last_bar_time.items()},
        })


def last_data_day(cfg) -> str:
    """Last UTC day with bars for any asset."""
    far = pd.Timestamp("2100-01-01", tz="UTC")
    last = max(fetch_bars(pd.Timestamp("1970-01-01", tz="UTC"), far, s["csv"], cfg.TIME_COL,
                          s.get("symbol"), getattr(cfg, "SYMBOL_COL", "symbol")).index.max()
               for s in cfg.ASSETS.values())
    return str(last.date())


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--date", help="single day, YYYY-MM-DD (UTC)")
    ap.add_argument("--start", help="first trading day, YYYY-MM-DD; history before it is "
                                    "only used to warm up the features")
    ap.add_argument("--end", help="last day (inclusive), YYYY-MM-DD; default: last day with data")
    ap.add_argument("--reset", action="store_true", help="delete existing records first")
    ap.add_argument(
        "--records-dir",
        help="output directory for state and CSV records; default: records",
    )
    args = ap.parse_args(argv)

    if args.records_dir:
        config.RECORDS_DIR = Path(args.records_dir)

    if args.reset and config.RECORDS_DIR.exists():
        shutil.rmtree(config.RECORDS_DIR)
        print(f"deleted {config.RECORDS_DIR}")

    if args.start:
        end = args.end or last_data_day(config)
        days = pd.date_range(args.start, end, freq="D")
        print(f"trading {days[0].date()} -> {days[-1].date()}" if len(days) else
              f"no data on or after {args.start}")
    elif args.date:
        days = [pd.Timestamp(args.date)]
    else:
        days = [pd.Timestamp.now(tz="UTC").normalize().tz_localize(None) - pd.Timedelta(days=1)]

    runner = DailyRunner(config)
    for day in days:
        runner.run_day(day)


if __name__ == "__main__":
    main()

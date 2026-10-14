"""Grid search over voting-rule combinations and rule parameters.

For every subset of rules (up to --max-rules), every parameter choice in PARAM_GRID,
every MIN_SCORE and every VOL_HIGH, simulate the VotingStrategy on the VALIDATION period
and rank by validation Sharpe. The TEST period is only reported for the top rows (never
used to choose), plus the rank correlation between validation and test over all configs,
which shows how much of the validation ranking is luck.

Simulation: one asset, full position size (+1 / -1 / 0), enter at the bar's open,
cost = (FEE_BPS + SLIPPAGE_BPS) per unit of turnover; same entry / exit logic as
VotingStrategy.step (opposite signal -> flat, at most HORIZON bars per position).

Stage 2 takes the top --exit-top entry configs (by validation Sharpe) and tries every
exit in EXIT_GRID: max holding bars, exit on opposite signal, stop-loss and take-profit
in multiples of ATR(14) at entry (checked at each bar's open, like VotingStrategy.step).

Usage:
  python run_rule_search.py                          # BTC, default splits
  python run_rule_search.py --no-model               # indicators only (no model rule / filter)
  python run_rule_search.py --max-rules 3 --top 30
  python run_rule_search.py --signs                  # also try each rule flipped (weight -1)
  python run_rule_search.py --exit-top 0             # skip the exit search

Live stage (default): all configs (or the top --live-top of stage 1) are simulated again exactly as
run_daily trades them - daily fallback, partial take-profit, time limit, stops, all read from
config.py - and the final ranking / choice uses that simulation.

Outputs: <out> (all entry configs, stage 1), <out>_live.csv (live stage), <out>_top.csv,
<out>_exits.csv (EXIT_GRID stage, only with --live-top 0)
"""
from __future__ import annotations

import argparse
import contextlib
import io
import itertools
import json
import time
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd

import config
from src.backtest import run_portfolio_backtest
from src.data_source import fetch_bars
from src.metrics import compute_metrics
from src.tracker import CostModel
from src.features import build_features
from src.strategy import RULES, VOTING_INDICATORS, VotingStrategy

# default for --model / --no-model. False: indicators only - no "model" rule and no
# volatility filter (p_up / p_down unused)
USE_MODEL = True

# thresholds tried for each rule (each dict is passed to the rule function as kwargs)
PARAM_GRID = {
    "rsi": [{"lo": v} for v in (25.0, 30.0, 35.0, 40.0)],
    "bb": [{"lo": v} for v in (0.0, 0.1, 0.2)],
    "stoch": [{"lo": v} for v in (10.0, 20.0, 30.0)],
    "trend": [{}],
    "macd": [{}],
    "ret": [{"lag": v} for v in (6, 12, 24)],
    "volume": [{"z": v} for v in (1.0, 2.0)],
    "model": [{"edge": v} for v in (0.02, 0.05)],
}
VOL_GRID = (0.30, 0.35, 0.40, None)          # None = USE_VOL_MODEL False

# stage 2: exits tried on the top --exit-top entry configs
EXIT_GRID = {
    "horizon": (6, 12, 24),
    "exit_on_opposite": (True, False),
    "stop_loss_atr": (0.0, 2.0, 4.0),            # 0 = off
    "take_profit_atr": (0.0, 2.0, 4.0, 6.0),     # 0 = off
}
BARS_PER_YEAR = 48 * 365

# default periods, for the models in config (trained to 2026-05-31 by run_train_models.py).
# Both must start AFTER the models' training data ends (checked in main); for the old
# notebook models (trained to 2025-10-09 06:00) use
# --val "2025-10-09 06:30" 2026-03-21 --test 2026-03-22 2026-09-01
VAL = ("2026-06-01 00:00", "2026-08-31 23:30")
TEST = ("2026-09-01 00:00", "2026-09-30 17:30")


def model_train_end(path) -> pd.Timestamp | None:
    """Last training bar of a model file, from the *_meta.json next to it (None if unknown)."""
    path = Path(path)
    for meta_file in path.parent.glob("*meta.json"):
        meta = json.loads(meta_file.read_text())
        if "models" in meta:                                   # run_train_models.py
            if path.name in {m["file"] for m in meta["models"].values()}:
                return pd.Timestamp(meta["train"][1])
        elif path.name in {"btc_up.txt", "btc_down.txt"} and "train_end" in meta:   # notebook
            return pd.Timestamp(meta["train_end"])
    return None


def load_period(cfg, asset, start, end, models):
    spec = cfg.ASSETS[asset]
    t0, t1 = pd.Timestamp(start, tz="UTC"), pd.Timestamp(end, tz="UTC")
    bars = fetch_bars(t0 - pd.Timedelta(days=cfg.LOOKBACK_DAYS), t1 + pd.Timedelta(hours=1),
                      spec["csv"], cfg.TIME_COL, spec.get("symbol"),
                      getattr(cfg, "SYMBOL_COL", "symbol"))
    with contextlib.redirect_stdout(io.StringIO()):          # build_features prints
        feats = build_features(bars, asset)
    idx = feats.loc[t0:t1].index
    up, down = models
    X = feats.loc[idx, up.feature_name()]
    return {"p_up": up.predict(X), "p_down": down.predict(X),
            "ind": feats.loc[idx, list(VOTING_INDICATORS)].to_dict("records"),
            "open": bars.loc[idx, "open"].to_numpy(), "close": bars.loc[idx, "close"].to_numpy(),
            "index": idx}


def precompute_votes(D):
    """votes[(rule, params_key)] -> int8 array, using the same rule functions as live."""
    out = {}
    for rule, grid in PARAM_GRID.items():
        for p in grid:
            out[(rule, key(p))] = np.array([RULES[rule](i, a, b, **p) for i, a, b in
                                            zip(D["ind"], D["p_up"], D["p_down"])], np.int8)
    return out


def key(p: dict) -> str:
    return ",".join(f"{k}={v}" for k, v in p.items())


def positions(sig, horizon):
    """Same as VotingStrategy.step with default exits (opposite signal -> flat, max
    `horizon` bars, no stops); jumps between signals, so it is fast for stage 1."""
    pos = np.zeros(len(sig), np.int8)
    p, entry, expiry = 0, 0, 0
    for t in np.flatnonzero(sig):
        v = sig[t]
        if p != 0 and t >= expiry:
            pos[entry:expiry], p = p, 0
        if p == 0:
            p, entry, expiry = v, t, t + horizon
        elif v == -p:
            pos[entry:t], p = p, 0
    if p != 0:
        pos[entry:min(expiry, len(sig))] = p
    return pos


def positions_exits(sig, opens, atr, horizon, exit_on_opposite, stop_loss_atr, take_profit_atr):
    """Bar-by-bar copy of VotingStrategy.step with all exit options."""
    pos = np.zeros(len(sig), np.int8)
    p = left = 0
    entry_price = entry_atr = 0.0
    for t in range(len(sig)):
        v = sig[t]
        if p == 0:
            if v != 0:
                p, left = v, horizon
                entry_price, entry_atr = opens[t], (atr[t] if atr[t] == atr[t] else 0.0)
        else:
            move = p * (opens[t] / entry_price - 1) / entry_atr if entry_atr else 0.0
            if entry_atr and ((stop_loss_atr > 0 and move <= -stop_loss_atr) or
                              (take_profit_atr > 0 and move >= take_profit_atr)):
                p = left = 0
            elif v == -p and exit_on_opposite:
                p = left = 0
        pos[t] = p
        if p != 0:
            left -= 1
            if left == 0:
                p = 0
    return pos


def entry_score(D, subset, sgn, ps):
    """Weighted vote sum per bar (no threshold, no volatility filter)."""
    return sum(s * D["votes"][(r, key(p))].astype(np.int16) for r, s, p in zip(subset, sgn, ps))


def entry_signals(D, subset, sgn, ps, ms, vh, allow_short, score=None):
    score = entry_score(D, subset, sgn, ps) if score is None else score
    sig = np.where(score >= ms, 1, np.where(score <= -ms, -1, 0)).astype(np.int8)
    if not allow_short:
        sig[sig < 0] = 0
    if vh is not None:
        sig[D["vol"] < vh] = 0
    return sig


def live_settings(cfg) -> dict:
    """Exit / fallback settings run_daily uses (config.py), for the live simulation."""
    return {"horizon": cfg.HORIZON, "exit_on_opposite": getattr(cfg, "EXIT_ON_OPPOSITE", True),
            "stop_loss_atr": getattr(cfg, "STOP_LOSS_ATR", 0.0),
            "take_profit_atr": getattr(cfg, "TAKE_PROFIT_ATR", 0.0),
            "partial_tp_atr": getattr(cfg, "PARTIAL_TP_ATR", 0.0),
            "partial_sl_atr": getattr(cfg, "PARTIAL_SL_ATR", 0.0),
            "partial_fraction": getattr(cfg, "PARTIAL_FRACTION", 0.5),
            "daily_fallback": getattr(cfg, "DAILY_FALLBACK", False),
            "fallback_hour": getattr(cfg, "FALLBACK_HOUR", 20),
            "fallback_size": getattr(cfg, "FALLBACK_SIZE", 0.1),
            "fallback_horizon": getattr(cfg, "FALLBACK_HORIZON", 12)}


def positions_live(sig, score, D, allow_short, horizon, exit_on_opposite, stop_loss_atr,
                   take_profit_atr, partial_tp_atr, partial_sl_atr, partial_fraction,
                   daily_fallback, fallback_hour, fallback_size, fallback_horizon):
    """Bar-by-bar copy of VotingStrategy.step as run_daily runs it (full entries, daily
    fallback, partial exits, stops, time limit). Returns (position per bar, full-entry flags)."""
    opens, atr, hour, day = D["open"], D["atr"], D["hour"], D["day"]
    lean = np.where(score != 0, score, D["p_up"] - D["p_down"])
    n = len(sig)
    pos, full = np.zeros(n), np.zeros(n, bool)
    p, size, left, ep, ea, last_day, done = 0, 1.0, 0, 0.0, 0.0, -1, False
    for t in range(n):
        s = sig[t]
        enter = None
        if p == 0:
            if s != 0:
                enter = (s, 1.0, horizon)
            elif daily_fallback and hour[t] >= fallback_hour and day[t] != last_day:
                d = 1 if lean[t] >= 0 else -1
                enter = (1 if (d == -1 and not allow_short) else d, fallback_size, fallback_horizon)
        elif s != 0 and size < 1.0:
            enter = (s, 1.0, horizon)
        else:
            move = p * (opens[t] / ep - 1) / ea if (ep and ea) else None
            if move is not None and ((stop_loss_atr > 0 and move <= -stop_loss_atr) or
                                     (take_profit_atr > 0 and move >= take_profit_atr)):
                p, left = 0, 0
            elif s == -p and exit_on_opposite:
                p, left = 0, 0
            elif move is not None and not done and (
                    (partial_tp_atr > 0 and move >= partial_tp_atr) or
                    (partial_sl_atr > 0 and move <= -partial_sl_atr)):
                size *= 1 - partial_fraction
                done = True
        if enter is not None:
            p, size, left = enter
            ep, ea = opens[t], (atr[t] if atr[t] == atr[t] else 0.0)
            last_day, done = day[t], False
            full[t] = size == 1.0
        pos[t] = p * size
        if p != 0 and left > 0:
            left -= 1
            if left == 0:
                p = 0
    return pos, full


def report(pos, D, cfg, asset) -> tuple[dict, dict]:
    """Run a position path through the project's backtest (same portfolio / cost model as
    run_daily; open positions closed at the end) -> (report metrics, net return per month)."""
    spec = cfg.ASSETS[asset]
    bars = {asset: pd.DataFrame({"open": D["open"], "close": D["close"]}, index=D["index"])}
    costs = CostModel(cfg.FEE_BPS, cfg.SLIPPAGE_BPS, getattr(cfg, "MIN_FEE", 0.0),
                      dict(getattr(cfg, "COST_OVERRIDES", {})))
    res = run_portfolio_backtest(pd.DataFrame({asset: pos}, index=D["index"]), bars,
                                 {asset: spec["weight"]}, cfg.INITIAL_CASH,
                                 cost_model=costs, leverage=cfg.LEVERAGE)
    metrics = compute_metrics(res.equity, res.trades, res.positions, res.fills).to_dict()
    eq = res.equity
    month_end = eq.groupby(eq.index.strftime("%Y-%m")).last()
    month_start = month_end.shift(1).fillna(cfg.INITIAL_CASH)
    return metrics, (month_end / month_start - 1).to_dict()


def describe(subset, sgn, ps, ms, vh):
    return {"rules": "+".join(f"{'-' if s < 0 else ''}{r}" for r, s in zip(subset, sgn)),
            "params": "; ".join(f"{r}:{key(p)}" for r, p in zip(subset, ps) if p),
            "min_score": ms, "vol_high": vh if vh is not None else "off"}


def evaluate(pos, D, cost, opened=None):
    """Metrics of a position path (fractions allowed). `opened` marks the entries counted as
    trades (default: every move from flat, or a direction flip)."""
    ret = D["ret"]
    pos = pos.astype(float)
    prev = np.r_[0.0, pos[:-1]]
    turnover = np.abs(pos - prev)
    net = pos * ret - turnover * cost
    eq = np.cumsum(net)
    sd = net.std()
    if opened is None:
        opened = (pos != 0) & ((prev == 0) | (np.sign(prev) != np.sign(pos)))
    # trades opened per calendar month (only months with >= 15 days of bars count)
    per_month = np.bincount(D["month"][opened], minlength=D["n_months"])[D["full_months"]]
    month_net = monthly_net(net, D)
    return {"trades": int(opened.sum()), "gross": float((pos * ret).sum()),
            "net": float(eq[-1]), "sharpe": float(net.mean() / sd * np.sqrt(BARS_PER_YEAR)) if sd else 0.0,
            "max_dd": float((eq - np.maximum.accumulate(eq)).min()),
            "exposure": float(np.abs(pos).mean()),
            "trades_per_month": float(per_month.mean()), "min_month_trades": int(per_month.min()),
            "min_month_net": float(month_net.min()), "pos_months": float(np.mean(month_net > 0))}


def monthly_net(net, D):
    """Net return of each full calendar month (a month with no trades counts as 0)."""
    return np.bincount(D["month"], weights=net, minlength=D["n_months"])[D["full_months"]]


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--asset", default=next(iter(config.ASSETS)))
    ap.add_argument("--val", nargs=2, metavar=("START", "END"), default=VAL,
                    help="validation period used to rank; must start after the models' training "
                         "data, e.g. --val 2026-06-01 '2026-08-31 23:30'")
    ap.add_argument("--test", nargs=2, metavar=("START", "END"), default=TEST,
                    help="period only reported, never used to choose")
    ap.add_argument("--models", nargs=2, metavar=("UP", "DOWN"),
                    help="model files (default: the asset's model_up / model_down in config)")
    ap.add_argument("--model", action=argparse.BooleanOptionalAction, default=USE_MODEL,
                    help="search the model rule and the volatility filter too")
    ap.add_argument("--max-rules", type=int, default=4,
                    help="largest number of rules combined")
    ap.add_argument("--horizon", type=int, default=config.HORIZON, help="stage 1 max holding bars")
    ap.add_argument("--min-trades", type=int, default=10, help="on validation, to skip lucky tiny samples")
    ap.add_argument("--min-monthly-trades", type=int, default=0,
                    help="keep configs with at least this many trades in EVERY validation month")
    ap.add_argument("--min-avg-monthly-trades", type=float, default=0.0,
                    help="keep configs averaging at least this many trades per validation month")
    ap.add_argument("--all-months-positive", action="store_true",
                    help="keep configs whose net return is > 0 in EVERY validation month")
    ap.add_argument("--rank-by", choices=("sharpe", "worst_month"), default="sharpe",
                    help="validation metric used to rank: Sharpe or worst monthly net return")
    ap.add_argument("--signs", action="store_true", help="also try each rule with weight -1")
    ap.add_argument("--top", type=int, default=20)
    ap.add_argument("--live-top", type=int, default=None,
                    help="how many of the stage-1 best configs are re-simulated exactly as run_daily "
                         "trades them (fallback, partial exits, ... from config.py) and ranked again; "
                         "default: all configs passing the filters, 0 = skip")
    ap.add_argument("--exit-top", type=int, default=0,
                    help="entry configs used in the EXIT_GRID search (0 = skip; only with --live-top 0)")
    ap.add_argument("--out", default=str(config.RESULTS_DIR / "rule_search.csv"),
                    help="all configs; also writes <out>_top.csv, <out>_live.csv, <out>_exits.csv")
    args = ap.parse_args(argv)

    grid, vol_grid = dict(PARAM_GRID), VOL_GRID
    if not args.model:
        grid.pop("model", None)
        vol_grid = (None,)

    spec = config.ASSETS[args.asset]
    up_path, down_path = args.models or (spec["model_up"], spec["model_down"])
    for p in (up_path, down_path):
        end = model_train_end(p)
        if end is None:
            print(f"warning: training period of {p} unknown; make sure --val starts after it")
        elif pd.Timestamp(args.val[0], tz="UTC") <= end:
            raise SystemExit(f"--val starts {args.val[0]}, but {Path(p).name} was trained on data "
                             f"up to {end}: tune on data the model has not seen")
    models = (lgb.Booster(model_file=str(up_path)), lgb.Booster(model_file=str(down_path)))
    cost = (config.FEE_BPS + config.SLIPPAGE_BPS) / 1e4
    periods = {"val": load_period(config, args.asset, *args.val, models),
               "test": load_period(config, args.asset, *args.test, models)}
    print(f"val {args.val[0]} -> {args.val[1]}, test {args.test[0]} -> {args.test[1]}, "
          f"models {up_path} / {down_path}")
    for D in periods.values():
        D["votes"] = precompute_votes(D)
        D["vol"] = D["p_up"] + D["p_down"]
        D["ret"] = np.r_[D["open"][1:] / D["open"][:-1] - 1, 0.0]   # open t -> open t+1
        D["atr"] = np.array([i["atr_14"] for i in D["ind"]], float)
        months = D["index"].to_period("M")
        D["month"] = pd.factorize(months)[0]
        D["n_months"] = D["month"].max() + 1
        D["full_months"] = np.bincount(D["month"]) >= 48 * 15
        D["month_names"] = [str(m) for m in pd.unique(months)]
        D["month_names"] = [m for m, f in zip(D["month_names"], D["full_months"]) if f]
        D["hour"] = D["index"].hour.to_numpy()
        D["day"] = pd.factorize(D["index"].normalize())[0]
    live = live_settings(config)
    if getattr(config, "PARTIAL_MIN_SCORE", 0) > 0:
        print("warning: partial entries (PARTIAL_MIN_SCORE > 0) are not simulated by this search")

    # ---- stage 1: entry rules, default exits ------------------------------------------
    signs = (1, -1) if args.signs else (1,)
    configs = []
    for k in range(1, args.max_rules + 1):
        for subset in itertools.combinations(list(grid), k):
            for sgn in itertools.product(signs, repeat=k):
                for ps in itertools.product(*(grid[r] for r in subset)):
                    for ms in range(1, k + 1):
                        for vh in vol_grid:
                            configs.append((subset, sgn, ps, ms, vh))
    print(f"stage 1: {len(configs):,} entry configurations, {args.asset}, model "
          f"{'on' if args.model else 'off'}, horizon {args.horizon}, "
          f"cost {cost * 1e4:.1f} bps per side")

    t_start, rows = time.time(), []
    for c in configs:
        row = describe(*c)
        for name, D in periods.items():
            sig = entry_signals(D, *c, config.ALLOW_SHORT)
            m = evaluate(positions(sig, args.horizon), D, cost)
            row |= {f"{name}_{k}": v for k, v in m.items()}
        rows.append(row)
    R = pd.DataFrame(rows)
    print(f"done in {time.time() - t_start:.0f}s")
    config.RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    R.to_csv(args.out, index=False)
    print(f"all results -> {args.out}")

    def keep(df):
        m = ((df["val_trades"] >= args.min_trades)
             & (df["val_min_month_trades"] >= args.min_monthly_trades)
             & (df["val_trades_per_month"] >= args.min_avg_monthly_trades))
        if args.all_months_positive:
            m &= df["val_min_month_net"] > 0
        return df[m]
    rank_col = {"sharpe": "val_sharpe", "worst_month": "val_min_month_net"}[args.rank_by]

    ok = keep(R)
    if ok.empty:
        raise SystemExit("no configuration passes the filters; relax them")
    rho = ok["val_net"].rank().corr(ok["test_net"].rank())
    print(f"\n{len(ok):,} configs pass (>= {args.min_trades} validation trades, >= "
          f"{args.min_monthly_trades} in every month, >= {args.min_avg_monthly_trades:g} "
          f"per month on average{', every month positive' if args.all_months_positive else ''}); "
          f"rank correlation val vs test net return: {rho:+.2f}")
    print(f"share profitable after costs: val {np.mean(ok['val_net'] > 0):.0%}, "
          f"test {np.mean(ok['test_net'] > 0):.0%}; every test month positive: "
          f"{np.mean(ok['test_min_month_net'] > 0):.0%}")

    ranked = ok.sort_values(rank_col, ascending=False)
    top = ranked.head(args.top)
    cols = ["rules", "params", "min_score", "vol_high", "val_trades_per_month",
            "val_net", "val_sharpe", "val_min_month_net", "val_pos_months",
            "test_trades_per_month", "test_net", "test_sharpe", "test_min_month_net",
            "test_pos_months"]
    top_path = args.out.replace(".csv", "_top.csv")
    top.to_csv(top_path, index=False)
    print(f"top {args.top} by {rank_col} -> {top_path}")
    with pd.option_context("display.width", 300, "display.max_colwidth", 60):
        print(f"\nTop {args.top} by {rank_col} (test shown for reference, not used to choose):")
        print(top[cols].round(3).to_string(index=False))
    best = configs[top.index[0]]
    best_exit = dict(horizon=args.horizon, exit_on_opposite=True,
                     stop_loss_atr=0.0, take_profit_atr=0.0)
    expected = top.iloc[0]["val_net"]
    best_live = False

    # ---- live stage: re-simulate the best configs exactly as run_daily trades -----------
    if args.live_top is None or args.live_top > 0:
        ids = list(ranked.index if args.live_top is None else ranked.index[:args.live_top])
        print(f"\nlive stage: top {len(ids):,} configs re-simulated with config.py exits and "
              f"fallback: " + ", ".join(f"{k}={v}" for k, v in live.items()))
        t_start, rows = time.time(), []
        for i in ids:
            row = describe(*configs[i]) | {"stage1_rank": ids.index(i) + 1}
            for name, D in periods.items():
                score = entry_score(D, *configs[i][:3])
                sig = entry_signals(D, *configs[i], config.ALLOW_SHORT, score=score)
                pos, full = positions_live(sig, score, D, config.ALLOW_SHORT, **live)
                m = evaluate(pos, D, cost, opened=full)      # trades = full-size entries
                mnet = monthly_net(pos * D["ret"] - np.abs(np.diff(np.r_[0.0, pos])) * cost, D)
                m |= {f"net_{mn}": v for mn, v in zip(D["month_names"], mnet)}
                all_entries = (pos != 0) & (np.r_[0.0, pos[:-1]] == 0)
                m["fallback_trades"] = int(all_entries.sum() - full.sum())
                m["all_trades_per_month"] = float(np.bincount(D["month"][all_entries], minlength=D["n_months"])
                                                  [D["full_months"]].mean())
                row |= {f"{name}_{k}": v for k, v in m.items()}
            rows.append(row | {"_id": i})
        L = pd.DataFrame(rows)
        print(f"done in {time.time() - t_start:.0f}s")
        live_path = args.out.replace(".csv", "_live.csv")
        L.drop(columns="_id").to_csv(live_path, index=False)
        print(f"live results -> {live_path}")
        L_ok = keep(L)
        if L_ok.empty:
            raise SystemExit("no configuration passes the filters in the live stage; relax them")
        top = L_ok.sort_values(rank_col, ascending=False).head(args.top)
        print(f"\nreport metrics for the top {len(top)} (backtest with config.py costs and weight)...")
        reports = []
        for rank, (_, r) in enumerate(top.iterrows(), 1):
            c = configs[int(r["_id"])]
            row = {"rank": rank} | describe(*c) | live
            months, mets = {}, {}
            for name, D in periods.items():
                score = entry_score(D, *c[:3])
                pos = positions_live(entry_signals(D, *c, config.ALLOW_SHORT, score=score), score, D,
                                     config.ALLOW_SHORT, **live)[0]
                met, mon = report(pos, D, config, args.asset)
                months |= {f"{name} net {k}": v for k, v in mon.items()}
                mets |= {f"{name} {k}": v for k, v in met.items()}
            reports.append(row | months | mets | {"_id": r["_id"]})
        top = pd.DataFrame(reports)
        top.drop(columns="_id").to_csv(top_path, index=False)
        shown = ("Total Return", "Annual Return", "Annual Volatility", "Sharpe Ratio",
                 "Max Drawdown", "Win Rate", "Number of Trades")
        lcols = ["rank", "rules", "params", "min_score", "vol_high",
                 *[c for c in top.columns if " net " in c],
                 *[f"val {m}" for m in shown], *[f"test {m}" for m in shown]]
        with pd.option_context("display.width", 300, "display.max_colwidth", 60):
            print(f"\nTop {args.top} by {rank_col}, live simulation: net return per month and report "
                  f"metrics (test for reference only; all columns in {top_path})")
            print(top[lcols].round(3).to_string(index=False))
        best = configs[int(top.iloc[0]["_id"])]
        expected = L.loc[L["_id"] == top.iloc[0]["_id"], "val_net"].iloc[0]
        best_live = True

    # ---- stage 2: exits on the best entry configs -----------------------------------
    if args.exit_top > 0 and not best_live:     # only with --live-top 0
        exits = [dict(zip(EXIT_GRID, v)) for v in itertools.product(*EXIT_GRID.values())]
        entry_ids = list(ranked.index[:args.exit_top])
        print(f"\nstage 2: {len(exits)} exit settings x top {len(entry_ids)} entry configs")
        t_start, rows = time.time(), []
        for rank, i in enumerate(entry_ids, 1):
            sigs = {n: entry_signals(D, *configs[i], config.ALLOW_SHORT) for n, D in periods.items()}
            for ex in exits:
                row = {"entry_rank": rank} | describe(*configs[i]) | ex
                for name, D in periods.items():
                    pos = positions_exits(sigs[name], D["open"], D["atr"], **ex)
                    row |= {f"{name}_{k}": v for k, v in evaluate(pos, D, cost).items()}
                rows.append(row | {"_id": i})
        E = pd.DataFrame(rows)
        print(f"done in {time.time() - t_start:.0f}s")
        exits_path = args.out.replace(".csv", "_exits.csv")
        E.drop(columns="_id").to_csv(exits_path, index=False)
        print(f"all exit results -> {exits_path}")

        E_ok = keep(E)
        print("\nmedian Sharpe over these entry configs for each exit setting (validation | test):")
        for col in EXIT_GRID:
            g = E_ok.groupby(col)[["val_sharpe", "test_sharpe"]].median()
            print(f"  {col:17s} " + "   ".join(f"{v}: {a:+.2f} | {b:+.2f}" for v, (a, b) in g.iterrows()))
        top_e = E_ok.sort_values(rank_col, ascending=False).head(args.top)
        ecols = ["entry_rank", "rules", "params", "min_score", "vol_high", *EXIT_GRID,
                 "val_trades_per_month", "val_net", "val_sharpe", "val_min_month_net",
                 "test_trades_per_month", "test_net", "test_sharpe", "test_min_month_net",
                 "test_pos_months"]
        with pd.option_context("display.width", 300, "display.max_colwidth", 50):
            print(f"\nTop {args.top} entry + exit by {rank_col} (test for reference only):")
            print(top_e[ecols].round(3).to_string(index=False))
        b = top_e.iloc[0]
        best = configs[int(b["_id"])]
        best_exit = {"horizon": int(b["horizon"]), "exit_on_opposite": bool(b["exit_on_opposite"]),
                     "stop_loss_atr": float(b["stop_loss_atr"]),
                     "take_profit_atr": float(b["take_profit_atr"])}
        expected = b["val_net"]

    # check the fast simulation against the real strategy class for the chosen config
    subset, sgn, ps, ms, vh = best
    exits = live if best_live else best_exit
    st = VotingStrategy(weights={r: float(s) for r, s in zip(subset, sgn)}, min_score=ms,
                        params={r: p for r, p in zip(subset, ps) if p},
                        use_vol_model=vh is not None, vol_high=vh or 0.0,
                        allow_short=config.ALLOW_SHORT, **exits)
    D = periods["val"]
    real = np.array([st.step(a, b, i, o, t)[1] for a, b, i, o, t in
                     zip(D["p_up"], D["p_down"], D["ind"], D["open"], D["index"])])
    assert np.isclose(evaluate(real, D, cost)["net"], expected), \
        "fast simulation differs from VotingStrategy"

    print(f"\nChosen config ({'live simulation' if best_live else 'full-size trades only'}), "
          f"net return per month:")
    for name, D in periods.items():
        score = entry_score(D, subset, sgn, ps)
        sig = entry_signals(D, *best, config.ALLOW_SHORT, score=score)
        if best_live:
            pos = positions_live(sig, score, D, config.ALLOW_SHORT, **live)[0]
        else:
            pos = positions_exits(sig, D["open"], D["atr"], **best_exit).astype(float)
        mn = monthly_net(pos * D["ret"] - np.abs(np.diff(np.r_[0.0, pos])) * cost, D)
        print(f"  {name:5s} " + "  ".join(f"{m}: {v:+.2%}" for m, v in zip(D["month_names"], mn)))

    print("\nBest-on-validation as config.py settings:")
    print(f"STRATEGY = \"voting\"\nUSE_VOL_MODEL = {vh is not None}")
    if vh is not None:
        print(f"VOL_HIGH = {vh}")
    if not best_live:
        for k, v in best_exit.items():
            print(f"{k.upper()} = {v}")
    else:
        print("# exits and fallback: as in config.py (HORIZON, PARTIAL_TP_ATR, DAILY_FALLBACK, ...)")
    print(f"RULE_WEIGHTS = {{{', '.join(f'{r!r}: {float(s)}' for r, s in zip(subset, sgn))}}}")
    print(f"RULE_PARAMS = {{{', '.join(f'{r!r}: {p}' for r, p in zip(subset, ps) if p)}}}")
    print(f"MIN_SCORE = {float(ms)}")


if __name__ == "__main__":
    main()

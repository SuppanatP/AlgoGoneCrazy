"""Train the up / down 12-bar models (same setup as model/model_h12_1sd.ipynb) on a chosen
date range, using src.features.build_features so live and training features match.

Labels: y = (open[t+12] - open[t]) / open[t] * 100; up = y > +1 SD, down = y < -1 SD,
with SD from the TRAIN rows only. Features at row t use data through bar t-1.
The last HORIZON rows of train and val are dropped (their labels look into the next split),
and no bar after --val-end is used.

Usage:
  python run_train_models.py --train-end 2026-05-31 --val-end 2026-07-31
  python run_train_models.py --train-end 2026-05-31 --val-end 2026-07-31 --tag 202607
Writes models/btc_up_<tag>.txt, models/btc_down_<tag>.txt, models/btc_models_<tag>_meta.json
"""
from __future__ import annotations

import argparse
import contextlib
import io
import json

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

import config
from src.data_source import fetch_bars
from src.features import build_features

HORIZON = 12
LGB_PARAMS = dict(objective="binary", n_estimators=5000, learning_rate=0.02, num_leaves=31,
                  min_child_samples=200, subsample=0.8, subsample_freq=1, colsample_bytree=0.7,
                  reg_lambda=1.0, random_state=42, verbose=-1)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--asset", default=next(iter(config.ASSETS)))
    ap.add_argument("--train-start", default="2023-09-01")
    ap.add_argument("--train-end", required=True, help="last day of training (inclusive)")
    ap.add_argument("--val-end", required=True, help="last day of early-stopping validation")
    ap.add_argument("--tag", help="file name suffix (default: val-end as YYYYMM)")
    ap.add_argument("--n-trees", type=int, help="fixed number of trees (no early stopping; "
                                               "val is then only reported)")
    args = ap.parse_args(argv)

    day = pd.Timedelta(days=1)
    t0 = pd.Timestamp(args.train_start, tz="UTC")
    t1 = pd.Timestamp(args.train_end, tz="UTC") + day          # exclusive
    t2 = pd.Timestamp(args.val_end, tz="UTC") + day            # exclusive; nothing after is used
    tag = args.tag or (t2 - day).strftime("%Y%m")
    spec = config.ASSETS[args.asset]

    bars = fetch_bars(t0, t2, spec["csv"], config.TIME_COL, spec.get("symbol"),
                      getattr(config, "SYMBOL_COL", "symbol"))
    with contextlib.redirect_stdout(io.StringIO()):
        feats = build_features(bars, args.asset)
    feats = feats[[c for c in feats.columns if not c.startswith("winrate_")]]   # old alias
    y = (bars["open"].shift(-HORIZON) / bars["open"] - 1) * 100             # NaN past t2

    data = feats.assign(y=y).dropna()
    train = data[data.index < t1].iloc[:-HORIZON]
    val = data[data.index >= t1].iloc[:-HORIZON]
    sd = float(train["y"].std())
    cols = list(feats.columns)
    print(f"{args.asset}: train {train.index[0]} -> {train.index[-1]} ({len(train):,} rows), "
          f"val {val.index[0]} -> {val.index[-1]} ({len(val):,} rows), 1 SD = {sd:.4f}%")

    meta = {"asset": args.asset, "horizon_bars": HORIZON, "label_threshold_pct": sd,
            "train": [str(train.index[0]), str(train.index[-1])],
            "val": [str(val.index[0]), str(val.index[-1])], "features": cols, "models": {}}
    for side, label in (("up", lambda s: s > sd), ("down", lambda s: s < -sd)):
        ytr, yva = label(train["y"]).astype(int), label(val["y"]).astype(int)
        if args.n_trees:
            m = lgb.LGBMClassifier(**(LGB_PARAMS | {"n_estimators": args.n_trees}))
            m.fit(train[cols], ytr)
            n_iter = args.n_trees
        else:
            m = lgb.LGBMClassifier(**LGB_PARAMS)
            m.fit(train[cols], ytr, eval_set=[(val[cols], yva)], eval_metric="auc",
                  callbacks=[lgb.early_stopping(200, verbose=False)])
            n_iter = m.best_iteration_
        auc = roc_auc_score(yva, m.predict_proba(val[cols], num_iteration=n_iter)[:, 1])
        path = config.ROOT / "models" / f"{args.asset.lower()}_{side}_{tag}.txt"
        m.booster_.save_model(str(path), num_iteration=n_iter)
        meta["models"][side] = {"file": path.name, "best_iteration": int(n_iter),
                                "val_auc": float(auc), "val_positive_rate": float(yva.mean())}
        print(f"  {side:4s}: {n_iter} trees, val AUC {auc:.4f}, "
              f"val positive rate {yva.mean():.1%} -> {path.name}")
    meta_path = config.ROOT / "models" / f"{args.asset.lower()}_models_{tag}_meta.json"
    meta_path.write_text(json.dumps(meta, indent=2))
    print(f"  meta -> {meta_path.name}")


if __name__ == "__main__":
    main()

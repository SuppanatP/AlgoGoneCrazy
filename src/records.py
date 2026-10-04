"""Persistent records for the daily runner.

records/
  state.json      portfolio + strategy state, last processed bar (overwritten each run)
  decisions.csv   one row per 30-min bar: probs, signal, target position, action, equity
  fills.csv       every executed order
  trades.csv      every closed round-trip trade
  equity.csv      portfolio value at each bar close
"""
from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Optional

import pandas as pd

STATE_FILE = "state.json"
CSV_FILES = {"decisions": "decisions.csv", "fills": "fills.csv",
             "trades": "trades.csv", "equity": "equity.csv"}


def load_state(records_dir: str | Path) -> Optional[dict]:
    path = Path(records_dir) / STATE_FILE
    if not path.exists():
        return None
    return json.loads(path.read_text())


def save_state(records_dir: str | Path, state: dict):
    """Write to a temp file then rename, so a crash never leaves a half-written state."""
    records_dir = Path(records_dir)
    records_dir.mkdir(parents=True, exist_ok=True)
    tmp = records_dir / (STATE_FILE + ".tmp")
    tmp.write_text(json.dumps(state, indent=2, default=str))
    os.replace(tmp, records_dir / STATE_FILE)


def append_csv(records_dir: str | Path, name: str, df: pd.DataFrame):
    if df is None or df.empty:
        return
    path = Path(records_dir) / CSV_FILES[name]
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, mode="a", header=not path.exists(), index=False)


def read_records(records_dir: str | Path) -> dict[str, pd.DataFrame]:
    """Load all CSV records with proper dtypes (UTC timestamps, timedeltas)."""
    out = {}
    for name, fname in CSV_FILES.items():
        path = Path(records_dir) / fname
        out[name] = pd.read_csv(path) if path.exists() else pd.DataFrame()

    for name in ("decisions", "equity", "fills"):
        df = out[name]
        if not df.empty:
            df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
            out[name] = df.set_index("timestamp").sort_index()

    tr = out["trades"]
    if not tr.empty:
        tr["entry_time"] = pd.to_datetime(tr["entry_time"], utc=True)
        tr["exit_time"] = pd.to_datetime(tr["exit_time"], utc=True)
        tr["holding_period"] = tr["exit_time"] - tr["entry_time"]
    return out

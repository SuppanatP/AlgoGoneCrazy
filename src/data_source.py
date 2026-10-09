"""Where the daily runner gets 30-min bars from.

fetch_bars() currently reads the local CSV. To run on fresh data every day, replace the
body with a call to your data pipeline (exchange API / the dev-data code) that returns
the same shape: UTC DatetimeIndex of bar open times, columns open/high/low/close/volume.
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd


def fetch_bars(start, end, csv_path: str | Path, time_col: str = "bar_open_time_utc",
               symbol: str | None = None, symbol_col: str = "symbol") -> pd.DataFrame:
    """30-min bars with open time in [start, end), sorted, UTC index.

    One CSV per asset: leave `symbol` as None. One CSV with all assets: pass `symbol`
    and the rows where `symbol_col` == symbol are returned.
    """
    start, end = pd.Timestamp(start), pd.Timestamp(end)
    start = start.tz_localize("UTC") if start.tzinfo is None else start.tz_convert("UTC")
    end = end.tz_localize("UTC") if end.tzinfo is None else end.tz_convert("UTC")

    bars = _load(Path(csv_path), time_col.lower(), symbol, symbol_col.lower())

    missing = {"open", "close"} - set(bars.columns)
    if missing:
        raise ValueError(f"bars are missing columns {missing}; rename them in fetch_bars")
    return bars[(bars.index >= start) & (bars.index < end)].copy()


_CACHE: dict = {}


def _load(path: Path, time_col: str, symbol, symbol_col: str) -> pd.DataFrame:
    """Read and parse the CSV once; re-read only if the file changes on disk."""
    key = (str(path), path.stat().st_mtime, symbol)
    if key not in _CACHE:
        bars = pd.read_csv(path)
        bars.columns = [c.lower() for c in bars.columns]
        if symbol is not None:
            bars = bars[bars[symbol_col] == symbol].drop(columns=symbol_col)
        bars.index = pd.to_datetime(bars.pop(time_col), utc=True)
        bars = bars.sort_index()
        _CACHE[key] = bars[~bars.index.duplicated(keep="last")]
    return _CACHE[key]

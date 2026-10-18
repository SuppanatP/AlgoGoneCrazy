"""Where the daily runner gets 30-min bars from.

fetch_bars() currently reads the local CSV. To run on fresh data every day, replace the
body with a call to your data pipeline (exchange API / the dev-data code) that returns
the same shape: UTC DatetimeIndex of bar open times, columns open/high/low/close/volume.
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd


def fetch_bars(start, end, csv_path, time_col: str = "bar_open_time_utc",
               symbol: str | None = None, symbol_col: str = "symbol") -> pd.DataFrame:
    """30-min bars with open time in [start, end), sorted, UTC index.

    One CSV per asset: leave `symbol` as None. One CSV with all assets: pass `symbol`
    and the rows where `symbol_col` == symbol are returned.
    `csv_path` can also be a list of CSVs (e.g. history + live): they are combined, and
    where the same bar appears twice the later file wins.
    """
    start, end = pd.Timestamp(start), pd.Timestamp(end)
    start = start.tz_localize("UTC") if start.tzinfo is None else start.tz_convert("UTC")
    end = end.tz_localize("UTC") if end.tzinfo is None else end.tz_convert("UTC")

    paths = [csv_path] if isinstance(csv_path, (str, Path)) else list(csv_path)
    parts = [_load(Path(p), time_col.lower(), symbol, symbol_col.lower()) for p in paths]
    bars = parts[0] if len(parts) == 1 else pd.concat(parts)
    bars = bars[~bars.index.duplicated(keep="last")].sort_index()

    missing = {"open", "close"} - set(bars.columns)
    if missing:
        raise ValueError(f"bars are missing columns {missing}; rename them in fetch_bars")
    return bars[(bars.index >= start) & (bars.index < end)].copy()


_CACHE: dict = {}


def _load(path: Path, time_col: str, symbol, symbol_col: str) -> pd.DataFrame:
    """Read and parse the CSV once; re-read only if the file changes on disk."""
    key = (str(path), path.stat().st_mtime, symbol)
    if key not in _CACHE:
        bars = pd.read_csv(path, low_memory=False)
        bars.columns = [c.lower() for c in bars.columns]
        if symbol is not None:
            bars = bars[bars[symbol_col] == symbol].drop(columns=symbol_col)
        bars.index = pd.to_datetime(bars.pop(time_col), utc=True)
        bars = bars.sort_index()
        _CACHE[key] = bars[~bars.index.duplicated(keep="last")]
    return _CACHE[key]


def history_problem(bars: pd.DataFrame, first_bar, days: float, freq: str = "30min") -> str | None:
    """Why `bars` cannot be used to trade from `first_bar`, or None if it can.

    Requires every `freq` bar in the `days` before `first_bar` to be present, so the
    features (moving averages, EMAs, RSI, ...) are computed on complete history.
    """
    first_bar = pd.Timestamp(first_bar)
    need = pd.date_range(first_bar - pd.Timedelta(days=days), first_bar, freq=freq, inclusive="left")
    missing = need.difference(bars.index)
    if missing.empty:
        return None
    if len(missing) == len(need):
        return f"no bars at all in {need[0]} -> {need[-1]}"
    return (f"{len(missing)} of {len(need)} bars missing between {need[0]} and {need[-1]}, "
            f"first missing {missing[0]}, last missing {missing[-1]}")

"""Labels and time-based splits. Move your feature-engineering functions here too,
so every notebook builds features the same way."""
from __future__ import annotations

import pandas as pd


def sd_labels(y_reg: pd.Series, threshold: float) -> tuple[pd.Series, pd.Series]:
    """Up / down labels: forward return above +threshold, or below -threshold.

    Compute `threshold` (e.g. 1 SD of y_reg) on the TRAIN split only.
    """
    return (y_reg > threshold).astype(int), (y_reg < -threshold).astype(int)


def time_split(df: pd.DataFrame, train_frac: float = 0.70, val_frac: float = 0.15,
               purge: int = 12) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Chronological train / val / test split.

    The last `purge` rows of train and val are dropped because their target windows
    overlap the next split (set purge = the target horizon in bars).
    """
    n = len(df)
    train_end, val_end = int(n * train_frac), int(n * (train_frac + val_frac))
    train = df.iloc[:train_end - purge] if purge else df.iloc[:train_end]
    val = df.iloc[train_end:val_end - purge] if purge else df.iloc[train_end:val_end]
    test = df.iloc[val_end:]
    return train, val, test


def build_features(bars: pd.DataFrame, asset: str | None = None,
                   all_bars: dict | None = None) -> pd.DataFrame:
    """Feature matrix for one asset: row t must use data through bar t-1 ONLY.

    bars     : this asset's 30-min bars (history + the day being simulated)
    asset    : its name, in case features differ per asset
    all_bars : {asset: bars} for every asset, for cross-asset features
               (e.g. BTC returns as a feature for ETH); ignore if not needed

    !!! EXAMPLE ONLY !!! Replace the body with the exact feature code used to train
    your models (move it here from data_preprocess.ipynb), so live features match
    training features column for column. The runner selects the columns the model
    was trained on (booster.feature_name()), so every one of them must be produced here.
    """
    import numpy as np

    c = bars["close"]
    r = np.log(c).diff()
    f = pd.DataFrame(index=bars.index)
    f["ret_1"] = r
    f["ret_6"] = r.rolling(6).sum()
    f["ret_12"] = r.rolling(12).sum()
    f["ret_48"] = r.rolling(48).sum()
    f["vol_12"] = r.rolling(12).std()
    f["vol_48"] = r.rolling(48).std()
    if {"high", "low"} <= set(bars.columns):
        f["range_1"] = (bars["high"] - bars["low"]) / c
    if "volume" in bars.columns:
        f["vol_ratio_48"] = bars["volume"] / bars["volume"].rolling(48).mean()
    return f.shift(1)  # leakage guard: row t sees nothing from bar t onwards

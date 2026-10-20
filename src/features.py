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

    This is the shared feature implementation used by model training, standalone
    replay, and target export. The runner selects the columns recorded in each
    model (booster.feature_name()), so training and execution use the same named
    inputs column for column.
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

    o, h, l, c, v = bars["open"], bars["high"], bars["low"], bars["close"], bars["volume"]
    qv = bars["quote_volume"]
    tc = bars["trade_count"]

    feat = pd.DataFrame(index=bars.index)
    ret1 = np.log(c / c.shift(1))

    # --- lagged log returns ---
    for lag in [1, 2, 3, 6, 12, 24, 48, 96]:
        feat[f"ret_lag_{lag}"] = np.log(c / c.shift(lag))

    # --- rolling return moments ---
    for w in [6, 12, 24, 48, 96, 192]:
        feat[f"ret_mean_{w}"] = ret1.rolling(w).mean()
        feat[f"ret_std_{w}"] = ret1.rolling(w).std()
        feat[f"ret_skew_{w}"] = ret1.rolling(w).skew()

    # --- moving-average deviation ---
    for w in [5, 10, 20, 50, 100, 200]:
        sma = c.rolling(w).mean()
        ema = c.ewm(span=w, adjust=False).mean()
        feat[f"close_over_sma_{w}"] = c / sma - 1
        feat[f"close_over_ema_{w}"] = c / ema - 1

    # --- MACD ---
    ema12 = c.ewm(span=12, adjust=False).mean()
    ema26 = c.ewm(span=26, adjust=False).mean()
    macd = ema12 - ema26
    macd_signal = macd.ewm(span=9, adjust=False).mean()
    feat["macd"] = macd / c
    feat["macd_signal"] = macd_signal / c
    feat["macd_hist"] = (macd - macd_signal) / c


    def rsi(series, w=14):
        delta = series.diff()
        gain = delta.clip(lower=0)
        loss = -delta.clip(upper=0)
        avg_gain = gain.ewm(alpha=1 / w, adjust=False).mean()
        avg_loss = loss.ewm(alpha=1 / w, adjust=False).mean()
        rs = avg_gain / avg_loss.replace(0, np.nan)
        return 100 - (100 / (1 + rs))


    for w in [7, 14, 21]:
        feat[f"rsi_{w}"] = rsi(c, w)

    # --- Bollinger Bands ---
    for w in [20, 50]:
        sma = c.rolling(w).mean()
        std = c.rolling(w).std()
        upper, lower = sma + 2 * std, sma - 2 * std
        feat[f"bb_pctb_{w}"] = (c - lower) / (upper - lower)
        feat[f"bb_width_{w}"] = (upper - lower) / sma

    # --- ATR ---
    prev_close = c.shift(1)
    tr = pd.concat([h - l, (h - prev_close).abs(), (l - prev_close).abs()], axis=1).max(axis=1)
    for w in [14, 28]:
        atr = tr.ewm(alpha=1 / w, adjust=False).mean()
        feat[f"atr_{w}"] = atr / c

    # --- stochastic oscillator ---
    for w in [14]:
        lowest, highest = l.rolling(w).min(), h.rolling(w).max()
        k = 100 * (c - lowest) / (highest - lowest)
        feat[f"stoch_k_{w}"] = k
        feat[f"stoch_d_{w}"] = k.rolling(3).mean()

    # --- candle shape ---
    feat["body"] = (c - o) / o
    feat["upper_wick"] = (h - np.maximum(o, c)) / o
    feat["lower_wick"] = (np.minimum(o, c) - l) / o
    feat["hl_range"] = (h - l) / c

    # --- volume / activity anomalies ---
    vol_ma = v.rolling(20).mean()
    feat["vol_ratio_20"] = v / vol_ma
    feat["vol_chg_1"] = np.log(v.replace(0, np.nan) / v.shift(1).replace(0, np.nan))
    feat["vol_z_20"] = (v - vol_ma) / v.rolling(20).std()
    feat["dollar_vol_ratio_20"] = qv / qv.rolling(20).mean()
    feat["trade_count_z_20"] = (tc - tc.rolling(20).mean()) / tc.rolling(20).std()

    # --- pct change ---
    for w in [3, 6, 12, 24]:
        feat[f"roc_{w}"] = c.pct_change(w)

    # --- rolling up-rate ---
    up_bar = (c > o).astype(float)
    for w in [10, 20, 50]:
        feat[f"uprate_{w}"] = up_bar.rolling(w).mean()
        feat[f"winrate_{w}"] = feat[f"uprate_{w}"]   # old name used by the saved models

    # --- cyclical time encodings ---
    hour = bars.index.hour + bars.index.minute / 60
    feat["hour_sin"] = np.sin(2 * np.pi * hour / 24)
    feat["hour_cos"] = np.cos(2 * np.pi * hour / 24)
    dow = bars.index.dayofweek
    feat["dow_sin"] = np.sin(2 * np.pi * dow / 7)
    feat["dow_cos"] = np.cos(2 * np.pi * dow / 7)

    return feat.shift(1)  # leakage guard: row t sees nothing from bar t onwards

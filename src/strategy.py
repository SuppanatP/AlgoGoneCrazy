"""Turn model probabilities into position signals (+1 long, -1 short, 0 flat).

Timing convention: the value at index t is the position to hold DURING bar t,
executed at bar t's open. It must only use information available before that open
(your features are built from data through bar t-1, so model outputs at t are fine).
Choose thresholds on the validation set, never on test.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

NAN = float("nan")


def signals_from_probs(p_up: pd.Series, p_down: pd.Series, th_up: float = 0.15,
                       th_down: float = 0.15, min_edge: float = 0.0,
                       allow_short: bool = True) -> pd.Series:
    """Entry signals from the up / down classifiers.

    Long when p_up clears its threshold and beats p_down by `min_edge`; short likewise.
    `min_edge` matters because both models also respond to volatility: requiring
    p_up - p_down > edge filters out bars where the models only expect a big move.
    """
    p_up, p_down = p_up.align(p_down, join="inner")
    long_ = (p_up >= th_up) & (p_up - p_down >= min_edge)
    short_ = (p_down >= th_down) & (p_down - p_up >= min_edge) & allow_short
    sig = pd.Series(0, index=p_up.index, dtype=int)
    sig[long_] = 1
    sig[short_ & ~long_] = -1
    print(sig)
    return sig


def hold_signals(signal: pd.Series, horizon: int = 12,
                 exit_on_opposite: bool = True) -> pd.Series:
    """Convert entry signals into positions held for `horizon` bars.

    A new signal in the same direction resets the holding clock; an opposite signal
    flips the position if `exit_on_opposite`, otherwise it is ignored until flat.
    """
    out = np.zeros(len(signal), dtype=int)
    pos, left = 0, 0
    for i, s in enumerate(signal.to_numpy()):
        if s != 0 and (pos == 0 or s == pos or exit_on_opposite):
            pos, left = int(s), horizon
        out[i] = pos
        if pos != 0:
            left -= 1
            if left == 0:
                pos = 0
    return pd.Series(out, index=signal.index, name="position")


class HoldStrategy:
    """Bar-by-bar version of signals_from_probs + hold_signals, with savable state.

    Used by the daily runner: call step(p_up, p_down) once per 30-min bar, in order.
    Its state (current position, bars left to hold) is saved between daily runs so a
    position opened late on one day is still held correctly the next day.
    Produces exactly the same positions as hold_signals(signals_from_probs(...)).
    """

    def __init__(self, th_up: float = 0.15, th_down: float = 0.15, min_edge: float = 0.0,
                 horizon: int = 12, allow_short: bool = True, exit_on_opposite: bool = True,
                 position: int = 0, bars_left: int = 0):
        self.th_up, self.th_down, self.min_edge = th_up, th_down, min_edge
        self.horizon, self.allow_short, self.exit_on_opposite = horizon, allow_short, exit_on_opposite
        self.position, self.bars_left = position, bars_left

    def signal(self, p_up: float, p_down: float, indicators: dict | None = None) -> int:
        """Probabilities only; `indicators` is ignored here (see RegimeStrategy)."""
        if p_up >= self.th_up and p_up - p_down >= self.min_edge:
            return 1
        if self.allow_short and p_down >= self.th_down and p_down - p_up >= self.min_edge:
            return -1
        return 0

    def step(self, p_up: float, p_down: float,
             indicators: dict | None = None) -> tuple[int, int]:
        """Returns (entry signal, position to hold during this bar)."""
        s = self.signal(p_up, p_down, indicators)
        if s != 0 and (self.position == 0 or s == self.position or self.exit_on_opposite):
            self.position, self.bars_left = s, self.horizon
        held = self.position
        if self.position != 0:
            self.bars_left -= 1
            if self.bars_left == 0:
                self.position = 0
        return s, held

    def to_state(self) -> dict:
        return dict(vars(self))

    @classmethod
    def from_state(cls, state: dict) -> "HoldStrategy":
        return cls(**state)


class RegimeStrategy(HoldStrategy):
    """Momentum / mean-reversion strategy that uses the models only as a volatility gauge.

    The up / down models mostly predict the SIZE of the next move, not its direction, so
    vol = p_up + p_down (chance of a >1 SD move either way within the horizon) picks the regime:
      vol >= vol_high  -> momentum: follow the trend (price vs SMA20/SMA50 + RSI)
      vol <= vol_low   -> mean reversion: fade extremes (Bollinger %B + RSI)
      in between       -> no new entry (an open position keeps its holding clock)
    Direction always comes from the indicators. Holding / flipping works as in HoldStrategy.
    NaN indicators (warm-up) never trigger an entry because NaN comparisons are False.
    """

    def __init__(self, vol_high: float = 0.40, vol_low: float = 0.25,
                 mom_rsi: float = 55.0, mr_rsi: float = 30.0, mr_bb: float = 0.0,
                 horizon: int = 12, allow_short: bool = True, exit_on_opposite: bool = True,
                 position: int = 0, bars_left: int = 0):
        self.vol_high, self.vol_low = vol_high, vol_low
        self.mom_rsi, self.mr_rsi, self.mr_bb = mom_rsi, mr_rsi, mr_bb
        self.horizon, self.allow_short, self.exit_on_opposite = horizon, allow_short, exit_on_opposite
        self.position, self.bars_left = position, bars_left

    # --- indicator signals: +1 long, -1 short, 0 none (NaN during warm-up -> 0) ---
    def momentum_signal(self, ind: dict) -> int:
        """Trend following: price above both SMAs with strong RSI -> long, mirror -> short."""
        rsi = ind.get("rsi_14", NAN)
        sma20, sma50 = ind.get("close_over_sma_20", NAN), ind.get("close_over_sma_50", NAN)
        if sma20 > 0 and sma50 > 0 and rsi >= self.mom_rsi:
            return 1
        if sma20 < 0 and sma50 < 0 and rsi <= 100 - self.mom_rsi:
            return -1
        return 0

    def mean_reversion_signal(self, ind: dict) -> int:
        """Fade extremes: below lower Bollinger band and oversold -> long, mirror -> short."""
        rsi, pctb = ind.get("rsi_14", NAN), ind.get("bb_pctb_20", NAN)
        if pctb <= self.mr_bb and rsi <= self.mr_rsi:
            return 1
        if pctb >= 1 - self.mr_bb and rsi >= 100 - self.mr_rsi:
            return -1
        return 0

    # --- combine with the volatility forecast ---
    def signal(self, p_up: float, p_down: float, indicators: dict | None = None) -> int:
        ind = indicators or {}
        vol = p_up + p_down
        if vol >= self.vol_high:
            s = self.momentum_signal(ind)
        elif vol <= self.vol_low:
            s = self.mean_reversion_signal(ind)
        else:
            s = 0
        if s == -1 and not self.allow_short:
            return 0
        return s

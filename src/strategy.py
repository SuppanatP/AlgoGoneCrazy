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
      vol >= vol_high  -> momentum: RSI <= mom_rsi -> long, RSI >= 100 - mom_rsi -> short
      vol <= vol_low   -> mean reversion: fade extremes (Bollinger %B + RSI)
      in between       -> no new entry (an open position keeps its holding clock)
    use_vol_model=False skips the regime gate: the momentum rules are applied on every bar.
    Direction always comes from the indicators.
    Exits (see step): max `horizon` bars, opposite signal (if exit_on_opposite), and optional
    stop-loss / take-profit in multiples of the ATR at entry, checked at each bar's open.
    NaN indicators (warm-up) never trigger an entry because NaN comparisons are False.
    """

    def __init__(self, vol_high: float = 0.40, vol_low: float = 0.25, use_vol_model: bool = True,
                 mom_rsi: float = 30.0, mr_rsi: float = 30.0, mr_bb: float = 0.0,
                 horizon: int = 12, allow_short: bool = True, exit_on_opposite: bool = True,
                 stop_loss_atr: float = 0.0, take_profit_atr: float = 0.0,
                 partial_tp_atr: float = 0.0, partial_sl_atr: float = 0.0,
                 partial_fraction: float = 0.5,
                 daily_fallback: bool = False, fallback_hour: int = 20,
                 fallback_size: float = 0.1, fallback_horizon: int = 12, **state):
        self.vol_high, self.vol_low, self.use_vol_model = vol_high, vol_low, use_vol_model
        self.mom_rsi, self.mr_rsi, self.mr_bb = mom_rsi, mr_rsi, mr_bb
        self.horizon, self.allow_short = horizon, allow_short
        self.partial_tp_atr, self.partial_sl_atr = partial_tp_atr, partial_sl_atr
        self.partial_fraction = partial_fraction
        self.fallback_horizon = fallback_horizon
        self._init_exits(exit_on_opposite, stop_loss_atr, take_profit_atr,
                         daily_fallback, fallback_hour, fallback_size, state)

    # running state saved between daily runs (with the parameters, see to_state)
    STATE = {"position": 0, "bars_left": 0, "entry_price": 0.0, "entry_atr": 0.0,
             "size": 1.0, "last_entry_day": "", "partial_done": False}

    def _init_exits(self, exit_on_opposite, stop_loss_atr, take_profit_atr,
                    daily_fallback, fallback_hour, fallback_size, state):
        self.exit_on_opposite = exit_on_opposite
        self.stop_loss_atr, self.take_profit_atr = stop_loss_atr, take_profit_atr
        self.daily_fallback, self.fallback_hour = daily_fallback, fallback_hour
        self.fallback_size = fallback_size
        unknown = set(state) - set(self.STATE)
        if unknown:
            raise TypeError(f"unknown arguments {sorted(unknown)}")
        for k, v in self.STATE.items():
            setattr(self, k, state.get(k, v))

    # --- indicator signals: +1 long, -1 short, 0 none (NaN during warm-up -> 0) ---
    def momentum_signal(self, ind: dict) -> int:
        """High-volatility rule on RSI: very low RSI -> long, very high RSI -> short."""
        rsi = ind.get("rsi_14", NAN)
        if rsi <= self.mom_rsi:
            return 1
        if rsi >= 100 - self.mom_rsi:
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
        if not self.use_vol_model:                     # ignore the models: momentum on every bar
            s = self.momentum_signal(ind)
        elif vol >= self.vol_high:
            s = self.momentum_signal(ind)
        elif vol <= self.vol_low:
            s = self.mean_reversion_signal(ind)
        else:
            s = 0
        if s == -1 and not self.allow_short:
            return 0
        return s

    def step(self, p_up: float, p_down: float, indicators: dict | None = None,
             price: float | None = None, ts=None) -> tuple[int, float]:
        """Returns (entry signal, position to hold during this bar, e.g. 1, -1, 0.1).

        `price` is this bar's open (where orders fill); needed for stop-loss / take-profit.
        `ts` is the bar's open time (UTC); needed for the daily fallback.
        flat + signal               -> enter at this open with the size from entry() (1 for a
                                       full signal, smaller for a partial one), <= `horizon` bars
        flat, no entry yet today and bar hour >= fallback_hour (daily_fallback on)
                                    -> enter `fallback_size` in the direction of lean()
        holding, open beyond stop / target (k x ATR at entry from the entry price)
                                    -> exit at this open (no re-entry on the same bar)
        holding, open beyond partial_tp_atr in profit or partial_sl_atr in loss
                                    -> reduce the size by partial_fraction (once per trade)
        holding + opposite signal   -> exit if exit_on_opposite, else ignore
        holding + signal with a bigger size (e.g. fallback or partial -> full)
                                    -> re-enter at that size in the signal's direction
        holding + same signal       -> keep holding, clock NOT reset
        `horizon` bars held         -> exit (horizon 0: no time limit, only the other exits;
                                       fallback trades use fallback_horizon instead)
        """
        s, size = self.entry(p_up, p_down, indicators)
        day = pd.Timestamp(ts).date().isoformat() if ts is not None else ""
        if self.position == 0:
            if s != 0:
                self._enter(s, size, indicators, price, day)
            elif (self.daily_fallback and ts is not None and day != self.last_entry_day
                  and pd.Timestamp(ts).hour >= self.fallback_hour):
                d = 1 if self.lean(p_up, p_down, indicators) >= 0 else -1
                if d == -1 and not self.allow_short:
                    d = 1
                self._enter(d, self.fallback_size, indicators, price, day, self.fallback_horizon)
        elif s != 0 and size > self.size:             # stronger signal replaces a smaller trade
            self._enter(s, size, indicators, price, day)
        elif self._hit_stop_or_target(price):
            self.position, self.bars_left = 0, 0
        elif s == -self.position and self.exit_on_opposite:
            self.position, self.bars_left = 0, 0
        elif self._hit_partial(price):
            self.size *= 1 - self.partial_fraction
            self.partial_done = True
        held = self.position * self.size
        if self.position != 0 and self.bars_left > 0:   # bars_left 0 = no time limit
            self.bars_left -= 1
            if self.bars_left == 0:
                self.position = 0
        return s, held

    def _enter(self, direction, size, indicators, price, day, horizon=None):
        atr = (indicators or {}).get("atr_14", NAN)
        self.position, self.size = direction, size
        self.bars_left = self.horizon if horizon is None else horizon    # 0 = no time limit
        self.entry_price = price if price is not None else 0.0
        self.entry_atr = atr if atr == atr else 0.0                 # NaN -> 0 (stops off)
        self.last_entry_day, self.partial_done = day, False

    def entry(self, p_up: float, p_down: float, indicators: dict | None = None) -> tuple[int, float]:
        """(direction, size) of an entry signal on this bar; size 0 when there is none."""
        s = self.signal(p_up, p_down, indicators)
        return s, (1.0 if s else 0.0)

    def lean(self, p_up: float, p_down: float, indicators: dict | None = None) -> float:
        """Direction for a fallback trade when there is no signal: the models' lean."""
        return p_up - p_down

    def _move_atr(self, price: float | None) -> float | None:
        """Open-to-entry move in the position's favour, in ATRs at entry (None if unknown)."""
        if price is None or not self.entry_price or not self.entry_atr:
            return None
        return self.position * (price / self.entry_price - 1) / self.entry_atr

    def _hit_stop_or_target(self, price: float | None) -> bool:
        move = self._move_atr(price)
        if move is None:
            return False
        return ((self.stop_loss_atr > 0 and move <= -self.stop_loss_atr) or
                (self.take_profit_atr > 0 and move >= self.take_profit_atr))

    def _hit_partial(self, price: float | None) -> bool:
        move = self._move_atr(price)
        if move is None or self.partial_done:
            return False
        return ((self.partial_tp_atr > 0 and move >= self.partial_tp_atr) or
                (self.partial_sl_atr > 0 and move <= -self.partial_sl_atr))


# --- voting strategy -----------------------------------------------------------------
# Each rule is deliberately loose and returns a vote: +1 long, -1 short, 0 no opinion.
# `ind` holds the indicator values for the bar (data through t-1); NaN gives 0 because
# every comparison with NaN is False.
def _vote(long_: bool, short_: bool) -> int:
    return 1 if long_ else -1 if short_ else 0


def rule_rsi(ind, p_up, p_down, lo=40.0):          # reversal: RSI <= lo long, >= 100-lo short
    rsi = ind.get("rsi_14", NAN)
    return _vote(rsi <= lo, rsi >= 100 - lo)


def rule_bb(ind, p_up, p_down, lo=0.2):            # reversal: %B <= lo long, >= 1-lo short
    b = ind.get("bb_pctb_20", NAN)
    return _vote(b <= lo, b >= 1 - lo)


def rule_stoch(ind, p_up, p_down, lo=20.0):        # reversal: stochastic %K <= lo / >= 100-lo
    k = ind.get("stoch_k_14", NAN)
    return _vote(k <= lo, k >= 100 - lo)


def rule_trend(ind, p_up, p_down):                 # trend: price above / below SMA20 and SMA50
    a, b = ind.get("close_over_sma_20", NAN), ind.get("close_over_sma_50", NAN)
    return _vote(a > 0 and b > 0, a < 0 and b < 0)


def rule_macd(ind, p_up, p_down):                  # trend: MACD histogram sign
    h = ind.get("macd_hist", NAN)
    return _vote(h > 0, h < 0)


def rule_ret(ind, p_up, p_down, lag=12):           # momentum: sign of the last `lag`-bar return
    r = ind.get(f"ret_lag_{lag}", NAN)
    return _vote(r > 0, r < 0)


def rule_volume(ind, p_up, p_down, z=1.0):         # volume z-score >= z confirms last bar's direction
    vz, body = ind.get("vol_z_20", NAN), ind.get("body", NAN)
    return _vote(vz >= z and body > 0, vz >= z and body < 0)


def rule_model(ind, p_up, p_down, edge=0.02):      # models' lean between big up and big down move
    return _vote(p_up - p_down >= edge, p_down - p_up >= edge)


RULES = {"rsi": rule_rsi, "bb": rule_bb, "stoch": rule_stoch, "trend": rule_trend,
         "macd": rule_macd, "ret": rule_ret, "volume": rule_volume, "model": rule_model}
VOTING_INDICATORS = ("rsi_14", "bb_pctb_20", "stoch_k_14", "close_over_sma_20",
                     "close_over_sma_50", "macd_hist", "ret_lag_6", "ret_lag_12", "ret_lag_24",
                     "vol_z_20", "body", "atr_14")


class VotingStrategy(RegimeStrategy):
    """Combine loose rules by weighted vote.

    score = sum(weight[r] * vote[r]); long if score >= min_score, short if <= -min_score.
    weight 0 switches a rule off; a negative weight flips it (e.g. use RSI as momentum).
    params = {rule: {arg: value}} overrides a rule's thresholds (e.g. {"rsi": {"lo": 30}}).
    With use_vol_model the vote only counts when the models expect a big move
    (p_up + p_down >= vol_high). Exits are the same as RegimeStrategy.step.
    """

    def __init__(self, weights: dict | None = None, min_score: float = 2.0,
                 params: dict | None = None, use_vol_model: bool = True, vol_high: float = 0.35,
                 partial_min_score: float = 0.0, partial_vol_high: float | None = None,
                 partial_entry_size: float = 0.5,
                 horizon: int = 12, allow_short: bool = True, exit_on_opposite: bool = True,
                 stop_loss_atr: float = 0.0, take_profit_atr: float = 0.0,
                 partial_tp_atr: float = 0.0, partial_sl_atr: float = 0.0,
                 partial_fraction: float = 0.5,
                 daily_fallback: bool = False, fallback_hour: int = 20,
                 fallback_size: float = 0.1, fallback_horizon: int = 12, **state):
        self.weights = dict(weights if weights is not None else {r: 1.0 for r in RULES})
        self.params = {r: dict(p) for r, p in (params or {}).items()}
        unknown = (set(self.weights) | set(self.params)) - set(RULES)
        if unknown:
            raise ValueError(f"unknown rules {sorted(unknown)}; available: {sorted(RULES)}")
        self.min_score, self.use_vol_model, self.vol_high = min_score, use_vol_model, vol_high
        self.partial_min_score, self.partial_entry_size = partial_min_score, partial_entry_size
        self.partial_vol_high = vol_high if partial_vol_high is None else partial_vol_high
        self.horizon, self.allow_short = horizon, allow_short
        self.partial_tp_atr, self.partial_sl_atr = partial_tp_atr, partial_sl_atr
        self.partial_fraction = partial_fraction
        self.fallback_horizon = fallback_horizon
        self._init_exits(exit_on_opposite, stop_loss_atr, take_profit_atr,
                         daily_fallback, fallback_hour, fallback_size, state)

    def votes(self, p_up: float, p_down: float, indicators: dict | None = None) -> dict:
        ind = indicators or {}
        return {r: RULES[r](ind, p_up, p_down, **self.params.get(r, {}))
                for r, w in self.weights.items() if w != 0}

    def lean(self, p_up: float, p_down: float, indicators: dict | None = None) -> float:
        """Weighted vote sum (no threshold, no volatility filter); models' lean on a tie."""
        score = sum(self.weights[r] * v for r, v in self.votes(p_up, p_down, indicators).items())
        return score if score != 0 else p_up - p_down

    def signal(self, p_up: float, p_down: float, indicators: dict | None = None) -> int:
        """Full-size entry: vol >= vol_high and |score| >= min_score."""
        return self._tier(p_up, p_down, indicators, self.vol_high, self.min_score)

    def entry(self, p_up: float, p_down: float, indicators: dict | None = None) -> tuple[int, float]:
        """Full signal -> size 1. Otherwise, if partial entries are on (partial_min_score > 0),
        an easier signal (vol >= partial_vol_high and |score| >= partial_min_score)
        -> size partial_entry_size."""
        s = self.signal(p_up, p_down, indicators)
        if s:
            return s, 1.0
        if self.partial_min_score > 0:
            s = self._tier(p_up, p_down, indicators, self.partial_vol_high, self.partial_min_score)
            if s:
                return s, self.partial_entry_size
        return 0, 0.0

    def _tier(self, p_up, p_down, indicators, vol_high, min_score) -> int:
        if self.use_vol_model and p_up + p_down < vol_high:
            return 0
        score = sum(self.weights[r] * v for r, v in self.votes(p_up, p_down, indicators).items())
        s = _vote(score >= min_score, score <= -min_score)
        return 0 if s == -1 and not self.allow_short else s

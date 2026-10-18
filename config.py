"""Settings for run_daily.py and report_metrics.py. Edit here, not in the scripts."""
from pathlib import Path

ROOT = Path(__file__).resolve().parent

# --- assets ------------------------------------------------------------------
# Standalone BTC strategy settings used by run_daily.py and report_metrics.py.
#   csv          30-min bars for this asset (or one shared CSV + "symbol" below); can be a
#                list of CSVs (history + live), combined in order, later files win
#   symbol       value in the CSV's instrument column when all assets share one CSV
#   model_up/down  LightGBM models (point all three to the same files for a pooled model)
#   weight       share of equity used per unit of signal (weights summing to 1 = no leverage)
# Optional per-asset strategy overrides: use_vol_model, vol_high, vol_low, mom_rsi, mr_rsi, mr_bb, horizon, allow_short
BTC_TRAINING_VALIDATION_FILE = (
    ROOT / "data" / "canonical" / "market_bars_30m_training_validation.csv"
)
BTC_LIVE_SIMULATION_FILE = (
    ROOT / "data" / "canonical" / "market_bars_30m_simulation_live_to_date.csv"
)
BTC_INPUT_FILES = [BTC_TRAINING_VALIDATION_FILE]
if BTC_LIVE_SIMULATION_FILE.exists():
    BTC_INPUT_FILES.append(BTC_LIVE_SIMULATION_FILE)

ASSETS = {
    "BTC": {"csv": BTC_INPUT_FILES,
            "symbol": "btc_usdt",
            "model_up": ROOT / "models" / "btc_up_202607_n100.txt",      # trained to 2026-05-31
            "model_down": ROOT / "models" / "btc_down_202607_n100.txt",  # (run_train_models.py)
            "weight": 1.0},
}
TIME_COL = "bar_open_time_utc"
SYMBOL_COL = "instrument_id"
BAR_FREQ = "30min"
LOOKBACK_DAYS = 30                  # history loaded before each day to compute features
WARMUP_DAYS = 14                    # gap-free history required before trading (EMA200 ~ 4 days x 3)

# --- outputs -------------------------------------------------------------------
RECORDS_DIR = ROOT / "records"
RESULTS_DIR = ROOT / "results"

# --- portfolio & transaction costs --------------------------------------------
INITIAL_CASH = 100_000.0
LEVERAGE = 1.0
FEE_BPS = 10.0                      # commission: 10 bps = 0.10% of notional per order
SLIPPAGE_BPS = 0.0                 # half spread + market impact, per order
MIN_FEE = 0.0                       # minimum commission per order
COST_OVERRIDES = {}                 # e.g. {"SOL": {"slippage_bps": 5.0}}

# --- strategy defaults (choose on the VALIDATION set) ------------------------------
# p_up + p_down is used as a volatility gauge (see src/strategy.py)
STRATEGY = "voting"                 # "regime" (RegimeStrategy) or "voting" (VotingStrategy)
USE_VOL_MODEL = True                 # False: ignore p_up/p_down, trade momentum rules on every bar
VOL_HIGH = 0.35                    # vol >= this -> momentum regime
VOL_LOW = 0                      # vol <= this -> mean-reversion regime (between: no entry)
MOM_RSI = 30.0                      # momentum: long if RSI <= this, short if RSI >= 100 - this
MR_RSI = 30.0                       # mean reversion: long if RSI <= this, short if RSI >= 100 - this
MR_BB = 0.0                         # mean reversion: long if %B <= this, short if %B >= 1 - this
HORIZON = 12                        # max bars to hold a position; 0 = no time limit
EXIT_ON_OPPOSITE = True             # exit when the signal points the other way
STOP_LOSS_ATR = 0.0                 # exit if the open is this many ATR(14) against entry; 0 = off
TAKE_PROFIT_ATR = 0.0               # exit if the open is this many ATR(14) in favour; 0 = off
PARTIAL_TP_ATR = 2.0                # sell PARTIAL_FRACTION once this many ATR in profit; 0 = off
PARTIAL_SL_ATR = 0.0                # cut PARTIAL_FRACTION once this many ATR in loss; 0 = off
PARTIAL_FRACTION = 0.5              # share of the position closed by a partial exit
DAILY_FALLBACK = True               # no entry yet today by FALLBACK_HOUR and flat -> small trade
FALLBACK_HOUR = 20                  # UTC hour from which the fallback may enter
FALLBACK_SIZE = 0.10                # fallback position size (fraction of a full position)
FALLBACK_HORIZON = 12               # fallback trades close after this many bars (keeps daily trading)
ALLOW_SHORT = True

# --- voting strategy (STRATEGY = "voting"; uses USE_VOL_MODEL, VOL_HIGH, HORIZON, ALLOW_SHORT) --
# weight per rule in src/strategy.py RULES: 0 = off, negative = flip the rule's vote
RULE_WEIGHTS = {"rsi": 0.0, "bb": 1.0, "stoch": 1.0, "trend": 0.0,
                "macd": 0.0, "ret": 1.0, "volume": 1.0, "model": 0.0}
# rule thresholds (defaults in src/strategy.py); run_rule_search.py finds good values
RULE_PARAMS = {"rsi": {"lo": 30.0}, "bb": {"lo": 0.1}, "stoch": {"lo": 30.0},
               "ret": {"lag": 12}, "volume": {"z": 1.0}, "model": {"edge": 0.05}}
MIN_SCORE = 2.0                     # long if weighted vote sum >= this, short if <= -this
# partial entry: an easier signal opens a smaller position (scaled up if a full signal follows)
PARTIAL_MIN_SCORE = 0.0             # |vote sum| needed for a partial entry; 0 = partial entries off
PARTIAL_VOL_HIGH = 0.35             # p_up + p_down needed for a partial entry; None = VOL_HIGH
PARTIAL_ENTRY_SIZE = 0.25           # partial entry size (fraction of a full position)

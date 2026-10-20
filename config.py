"""Runtime settings for the standalone BTC workflow.

Tracked strategy, model, data, and cost settings come from
``config/btc_strategy.json`` so the reproducibility manifest and executable
configuration cannot drift independently.
"""
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent
BTC_MANIFEST_PATH = ROOT / "config" / "btc_strategy.json"
BTC_MANIFEST = json.loads(BTC_MANIFEST_PATH.read_text())
_INSTRUMENT = BTC_MANIFEST["instrument"]
_DATA = BTC_MANIFEST["data"]
_MODELS = BTC_MANIFEST["models"]
_STRATEGY = BTC_MANIFEST["strategy"]
_EXECUTION = BTC_MANIFEST["execution"]

# --- assets ------------------------------------------------------------------
# Standalone BTC strategy settings used by run_daily.py and report_metrics.py.
#   csv          30-min bars for this asset (or one shared CSV + "symbol" below); can be a
#                list of CSVs (history + live), combined in order, later files win
#   symbol       value in the CSV's instrument column when all assets share one CSV
#   model_up/down  LightGBM models (point all three to the same files for a pooled model)
#   weight       share of equity used per unit of signal (weights summing to 1 = no leverage)
# Optional per-asset strategy overrides: use_vol_model, vol_high, vol_low, mom_rsi, mr_rsi, mr_bb, horizon, allow_short
BTC_TRAINING_VALIDATION_FILE = (
    ROOT / _DATA["training_validation_file"]
)
BTC_LIVE_SIMULATION_FILE = (
    ROOT / _DATA["live_simulation_file"]
)
BTC_INPUT_FILES = [BTC_TRAINING_VALIDATION_FILE]
if BTC_LIVE_SIMULATION_FILE.exists():
    BTC_INPUT_FILES.append(BTC_LIVE_SIMULATION_FILE)

ASSETS = {
    _INSTRUMENT["runtime_asset_id"]: {
        "csv": BTC_INPUT_FILES,
        "symbol": _INSTRUMENT["canonical_instrument_id"],
        "model_up": ROOT / _MODELS["up"]["file"],
        "model_down": ROOT / _MODELS["down"]["file"],
        "weight": float(_EXECUTION["strategy_weight"]),
    },
}
TIME_COL = _DATA["timestamp_column"]
SYMBOL_COL = _DATA["instrument_column"]
BAR_FREQ = _INSTRUMENT["bar_interval"]
LOOKBACK_DAYS = int(_DATA["lookback_days"])
WARMUP_DAYS = int(_DATA["warmup_days"])

# --- outputs -------------------------------------------------------------------
RECORDS_DIR = ROOT / "records"
RESULTS_DIR = ROOT / "results"

# --- portfolio & transaction costs --------------------------------------------
INITIAL_CASH = float(_EXECUTION["initial_cash_usd"])
LEVERAGE = float(_EXECUTION["leverage"])
FEE_BPS = float(_EXECUTION["commission_bps_per_order"])
SLIPPAGE_BPS = float(_EXECUTION["slippage_bps_per_order"])
MIN_FEE = float(_EXECUTION["minimum_commission_usd"])
COST_OVERRIDES = {}                 # e.g. {"SOL": {"slippage_bps": 5.0}}

# --- strategy defaults (choose on the VALIDATION set) ------------------------------
# p_up + p_down is used as a volatility gauge (see src/strategy.py)
STRATEGY = {
    "VotingStrategy": "voting",
    "RegimeStrategy": "regime",
}[_STRATEGY["class"]]
USE_VOL_MODEL = bool(_STRATEGY["use_volatility_model"])
VOL_HIGH = float(_STRATEGY["volatility_high_threshold"])
VOL_LOW = 0                      # vol <= this -> mean-reversion regime (between: no entry)
MOM_RSI = 30.0                      # momentum: long if RSI <= this, short if RSI >= 100 - this
MR_RSI = 30.0                       # mean reversion: long if RSI <= this, short if RSI >= 100 - this
MR_BB = 0.0                         # mean reversion: long if %B <= this, short if %B >= 1 - this
HORIZON = int(_STRATEGY["maximum_holding_bars"])
EXIT_ON_OPPOSITE = bool(_STRATEGY["exit_on_opposite"])
STOP_LOSS_ATR = float(_STRATEGY["stop_loss_atr"])
TAKE_PROFIT_ATR = float(_STRATEGY["take_profit_atr"])
PARTIAL_TP_ATR = float(_STRATEGY["partial_take_profit_atr"])
PARTIAL_SL_ATR = float(_STRATEGY["partial_stop_loss_atr"])
PARTIAL_FRACTION = float(_STRATEGY["partial_fraction"])
DAILY_FALLBACK = bool(_STRATEGY["daily_fallback"])
FALLBACK_HOUR = int(_STRATEGY["fallback_hour_utc"])
FALLBACK_SIZE = float(_STRATEGY["fallback_size"])
FALLBACK_HORIZON = int(_STRATEGY["fallback_holding_bars"])
ALLOW_SHORT = bool(_STRATEGY["allow_short"])

# --- voting strategy (STRATEGY = "voting"; uses USE_VOL_MODEL, VOL_HIGH, HORIZON, ALLOW_SHORT) --
# weight per rule in src/strategy.py RULES: 0 = off, negative = flip the rule's vote
RULE_WEIGHTS = dict(_STRATEGY["rule_weights"])
# rule thresholds (defaults in src/strategy.py); run_rule_search.py finds good values
RULE_PARAMS = dict(_STRATEGY["rule_parameters"])
MIN_SCORE = float(_STRATEGY["minimum_vote_score"])
# partial entry: an easier signal opens a smaller position (scaled up if a full signal follows)
PARTIAL_MIN_SCORE = float(_STRATEGY["partial_entry_minimum_score"])
PARTIAL_VOL_HIGH = float(_STRATEGY["partial_entry_volatility_threshold"])
PARTIAL_ENTRY_SIZE = float(_STRATEGY["partial_entry_size"])


def verify_selected_model_hashes() -> None:
    """Stop standalone replay/export if selected model contents have drifted."""
    for side, path in (
        ("up", ASSETS[_INSTRUMENT["runtime_asset_id"]]["model_up"]),
        ("down", ASSETS[_INSTRUMENT["runtime_asset_id"]]["model_down"]),
    ):
        expected = str(_MODELS[side]["sha256"])
        actual = hashlib.sha256(path.read_bytes()).hexdigest()
        if actual != expected:
            raise ValueError(
                f"Selected BTC {side} model hash does not match "
                f"{BTC_MANIFEST_PATH}: {path}"
            )

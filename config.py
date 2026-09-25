"""Settings for run_daily.py and report_metrics.py. Edit here, not in the scripts."""
from pathlib import Path

ROOT = Path(__file__).resolve().parent

# --- assets ------------------------------------------------------------------
# One entry per asset. Rename BTC/ETH/SOL to your three assets.
#   csv          30-min bars for this asset (or one shared CSV + "symbol" below)
#   symbol       value in the CSV's "symbol" column, only if all assets share one CSV
#   model_up/down  LightGBM models (point all three to the same files for a pooled model)
#   weight       share of equity used per unit of signal (weights summing to 1 = no leverage)
# Optional per-asset strategy overrides: th_up, th_down, min_edge, horizon, allow_short
ASSETS = {
    "BTC": {"csv": ROOT / "data" / "market_bars_30m_with_backfill.csv",
            "model_up": ROOT / "models" / "btc_up.txt",
            "model_down": ROOT / "models" / "btc_down.txt",
            "weight": 1 / 3},
#     "ETH": {"csv": ROOT / "data" / "raw" / "eth_30m.csv",
#             "model_up": ROOT / "models" / "eth_up.txt",
#             "model_down": ROOT / "models" / "eth_down.txt",
#             "weight": 1 / 3},
#     "SOL": {"csv": ROOT / "data" / "raw" / "sol_30m.csv",
#             "model_up": ROOT / "models" / "sol_up.txt",
#             "model_down": ROOT / "models" / "sol_down.txt",
#             "weight": 1 / 3},
}
TIME_COL = "bar_open_time_utc"
SYMBOL_COL = "symbol"
BAR_FREQ = "30min"
LOOKBACK_DAYS = 30                  # history loaded before each day to compute features

# --- outputs -------------------------------------------------------------------
RECORDS_DIR = ROOT / "records"
RESULTS_DIR = ROOT / "results"

# --- portfolio & transaction costs --------------------------------------------
INITIAL_CASH = 100_000.0
LEVERAGE = 1.0
FEE_BPS = 10.0                      # commission: 10 bps = 0.10% of notional per order
SLIPPAGE_BPS = 2.0                  # half spread + market impact, per order
MIN_FEE = 0.0                       # minimum commission per order
COST_OVERRIDES = {}                 # e.g. {"SOL": {"slippage_bps": 5.0}}

# --- strategy defaults (choose on the VALIDATION set) ------------------------------
TH_UP = 0.15
TH_DOWN = 0.15
MIN_EDGE = 0.02
HORIZON = 12                        # bars to hold after an entry signal
ALLOW_SHORT = True

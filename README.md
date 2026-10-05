# FE5209 Group Project - Market Data Pipeline

This first project phase extracts and cleans completed five-minute source bars
for:

- `SPY` from Alpaca Market Data
- `TLT` from Alpaca Market Data
- `SG.ES3` from Moomoo OpenAPI when entitled, or recent `ES3.SI` from Yahoo
- `BTCUSDT` from Binance Spot

It also provides separate steps that derive 30-minute OHLCV bars and apply a
small set of conservative backfill methods. A separate strategy package
generates momentum and mean-reversion signals, and a historical backtesting
entry point connects those signals to the shared trade tracker. Live execution,
parameter-search experiments, and PostgreSQL remain later phases.

## Design

Extraction and cleaning are separate modules but run in one application:

```text
Alpaca raw + adjusted / Moomoo OpenD / Binance
      -> source-specific extraction
      -> common standardization
      -> structural validation
      -> exchange-session validation
      -> accepted, rejected, missing-bar, and coverage files

Accepted 5m bars + expected market calendar
      -> session-aligned 30m aggregation
      -> complete 30m bars + incomplete-window records

Accepted 5m bars + complete 30m bars + incomplete-window records
      -> conservative backfill rules
      -> backfilled bars + combined usable bars + unresolved windows

Selected final 30m file from each asset
      -> source-selection and duplicate checks
      -> one canonical long-format 30m CSV

Configuration-driven orchestrator
      -> waits for all ingestions
      -> waits for all aggregations
      -> waits for all backfills
      -> runs consolidation last

Canonical 30m bars
      -> adjusted-close strategy features
      -> momentum or mean-reversion signal
      -> target long/cash position
      -> delay until the next complete real bar
      -> raw-open simulated execution
      -> trades, equity curve, costs, and comparison metrics
```

Five minutes is the source resolution. It preserves enough detail to construct
15-minute, 30-minute, hourly, or two-hour bars later without committing the
strategy to one aggregation schedule now.

For Alpaca instruments, ingestion requests each timestamp range twice. The
`raw` response supplies executable OHLCV values, while the `all` corporate-action
adjustment response supplies `adjusted_close`. Raw and adjusted timestamps must
match exactly or ingestion stops instead of silently mixing observations.

## Project layout

```text
src/market_data/
  config.py              Instrument metadata and ingestion settings
  extractors/alpaca.py   SPY/TLT raw/adjusted REST extraction and pagination
  extractors/moomoo.py   SG.ES3 OpenD extraction and pagination
  extractors/binance.py  BTCUSDT extraction and pagination
  extractors/yahoo.py    Selectable Yahoo fallback extraction
  standardize.py         Common provider-neutral columns
  validate.py            Shared quality rules
  sessions.py            Expected bars from XNYS, XSES, and 24/7 calendars
  coverage.py            Off-session and missing-expected-bar checks
  pipeline.py            Command-line orchestration and outputs
  aggregation.py         Complete 30m OHLCV construction and gap reporting
  backfill.py            Conservative 30m gap handling and provenance
  consolidation.py       Selected-source canonical long-format construction
  orchestration.py       Strict sequential workflow orchestration
src/trading_strategy/
  common.py              Shared strategy input checks and signal eligibility
  momentum.py            Long/cash EMA and trailing-return momentum rule
  mean_reversion.py      Long/cash rolling z-score mean-reversion rule
src/strategy_backtest.py Strategy-to-backtester adapter and output writer
src/backtest.py          Reused bar-by-bar backtest engine
src/tracker.py           Reused cash, position, fill, trade, and cost ledger
src/metrics.py           Reused performance and transaction-cost metrics
config/
  market_data_pipeline.json  Selected assets, providers, ranges, and paths
test/                    Offline tests using synthetic data
run_ingestion.py         Convenient entry point
run_aggregation.py       Separate 30m aggregation entry point
run_backfill.py          Separate non-destructive backfill entry point
run_consolidation.py     Canonical four-asset CSV entry point
run_market_data_pipeline.py  One-command sequential workflow entry point
run_strategy_backtest.py SPY/ES3 momentum and mean-reversion backtest entry point
run_parameter_experiments.py Pre-test parameter comparison entry point
```

## Strategy modules

The strategy package accepts canonical long-format bars and processes every
`instrument_id` independently. It uses `adjusted_close` for features and returns
signals only; it does not execute orders or calculate portfolio P&L.

Both strategies use the same output conventions:

| Column | Meaning |
|---|---|
| `strategy_name` | `momentum` or `mean_reversion` |
| `signal` | `BUY`, `SELL`, or `HOLD` |
| `target_position` | `1` for long and `0` for cash |
| `signal_strength` | Trailing return for momentum or negative z-score for mean reversion |
| `signal_reason` | Rule or data-quality reason behind the signal |
| `signal_eligible` | Whether the strategy may change position on that bar |

Signals may change only on final, complete aggregated bars. Approved backfilled
bars remain in the feature history for continuity, but a strategy does not enter
or exit on them. A later execution module must delay a signal and use a raw,
tradable price such as the next bar's `open`.

Example usage:

```python
import pandas as pd

from trading_strategy import (
    MeanReversionParameters,
    MomentumParameters,
    generate_mean_reversion_signals,
    generate_momentum_signals,
)

bars = pd.read_csv("data/canonical/market_bars_30m.csv")

momentum = generate_momentum_signals(
    bars,
    MomentumParameters(
        fast_window=13,
        slow_window=52,
        momentum_lookback=26,
    ),
)

mean_reversion = generate_mean_reversion_signals(
    bars,
    MeanReversionParameters(
        window=26,
        entry_z_score=-1.5,
        exit_z_score=0.0,
        maximum_holding_bars=52,
    ),
)
```

## Strategy backtesting

The strategy backtest runs SPY and ES3.SI independently so USD and SGD are not
mixed. Indicators use `adjusted_close`; a target produced after one bar closes
is delayed and executed at the next complete real bar's raw `open`. Positions
are valued using raw `close`. Backfilled bars preserve indicator continuity but
cannot trigger an execution. Any remaining position is force-closed using the
final complete real bar's raw `close` so each run ends with realized P&L.

By default, the runner executes momentum, mean reversion, and buy-and-hold for
both instruments using `config/cost_assumptions.json`. SPY and ES3 positions are
rounded to whole units. Slippage defaults to zero and can be supplied explicitly.

```bash
python run_strategy_backtest.py
```

Run one combination or select a historical range:

```bash
python run_strategy_backtest.py \
  --instrument SPY \
  --strategy momentum \
  --start 2024-01-01 \
  --end 2025-01-01 \
  --slippage-bps 2
```

Use a different canonical CSV or override project commissions with a flat fee:

```bash
python run_strategy_backtest.py \
  --input-file data/canonical/market_bars_30m.csv \
  --flat-fee-bps 10 \
  --output-dir results/strategy_backtests
```

The default output directory contains `summary.csv` and separate signal,
position, fill, trade, and equity CSVs for each run. `parameters.json` records
the parameters and execution assumptions. `summary.csv` is the first file to
use for comparing the four strategy/instrument combinations.

This is a historical simulation, not realized trading. The current raw-price
ledger does not add separate dividend cash flows, so dividend-paying assets and
their buy-and-hold benchmarks are compared consistently on a price-return basis.

### Pre-test parameter comparison

The parameter experiment uses only bars before 1 October 2026. It compares
three momentum and three mean-reversion configurations over chronological
development and validation periods. October and later data remain excluded for
the official test period.

```bash
python run_parameter_experiments.py
```

The default outputs are `results/parameter_experiments/parameter_comparison.csv`
and `experiment_manifest.json`. The manifest records the cutoff, periods, costs,
and every parameter set so the comparison can be reproduced.

## Setup

From the repository root:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

Create the local environment file:

```bash
cp .env.example .env
```

Add the Alpaca credentials generated in the Alpaca dashboard:

```text
ALPACA_API_KEY=your_key
ALPACA_SECRET_KEY=your_secret
ALPACA_DATA_FEED=sip
```

The `.env` file is ignored by Git. Do not place credentials directly in source
code or commit them.

Moomoo does not use the Alpaca-style key and secret for this workflow. Install
Moomoo OpenD, log in, confirm that the account has Singapore quote rights, and
leave OpenD running. The default local connection is configured in `.env`:

```text
MOOMOO_OPEND_HOST=127.0.0.1
MOOMOO_OPEND_PORT=11111
```

Binance public market-data downloads do not require credentials.

## Commands and options

Run commands from the repository root with the virtual environment activated.
Every command provides built-in help:

```bash
python run_ingestion.py --help
python run_aggregation.py --help
python run_backfill.py --help
python run_consolidation.py --help
python run_market_data_pipeline.py --help
python run_strategy_backtest.py --help
python run_parameter_experiments.py --help
```

### Ingestion options

| Argument | Meaning |
|---|---|
| `--provider NAME` | Select `alpaca`, `binance`, `yahoo`, or `moomoo`; canonical provider names also work |
| `--symbol SYMBOL` | Select `SPY`, `TLT`, `BTCUSDT`, `ES3.SI`, or `SG.ES3` |
| `--list-instruments` | List every configured provider-symbol pair and exit |
| `--start TIMESTAMP` | Inclusive UTC start |
| `--end TIMESTAMP` | Exclusive UTC end |
| `--lookback-days N` | Lookback used when `--start` is omitted; default 1,095 days |
| `--output-dir PATH` | Override the automatically selected staging directory |

Targeted runs are written under `data/staging/<provider>/<symbol>/`. A bare
`python run_ingestion.py` uses the default Alpaca, Moomoo, and Binance set and
writes to `data/staging/combined/`; avoid that command while Moomoo Singapore
data is unavailable.

### Aggregation and backfill options

| Command | Argument | Meaning |
|---|---|---|
| `run_aggregation.py` | `--input-dir PATH` | Ingestion output directory; default `data/staging/combined` |
| `run_aggregation.py` | `--output-dir PATH` | Aggregation output directory; default `data/staging/combined` |
| `run_backfill.py` | `--input-dir PATH` | Ingestion and aggregation directory; default `data/staging/combined` |
| `run_backfill.py` | `--output-dir PATH` | Backfill output directory; default `data/staging/combined` |

### Consolidation options

| Argument | Meaning |
|---|---|
| Repeated `--input-dir PATH` | Replace the four default selected-source directories |
| `--output-file PATH` | Canonical CSV; default `data/canonical/market_bars_30m.csv` |
| `--summary-file PATH` | JSON summary; default `data/canonical/consolidation_summary.json` |
| `--allow-partial` | Permit consolidation when fewer than all four instruments are supplied |

### Orchestrator option

| Argument | Meaning |
|---|---|
| `--config PATH` | Workflow JSON; default `config/market_data_pipeline.json` |

## Test provider connections

List the available combinations, then use short targeted runs. These commands
also test standardization, validation, and CSV writing:

```bash
python run_ingestion.py --list-instruments
python run_ingestion.py --provider alpaca --symbol SPY --lookback-days 7
python run_ingestion.py --provider alpaca --symbol TLT --lookback-days 7
python run_ingestion.py --provider binance --symbol BTCUSDT --lookback-days 2
python run_ingestion.py --provider yahoo --symbol ES3.SI --lookback-days 7
```

If Moomoo Singapore entitlement is obtained later:

```bash
python run_ingestion.py --provider moomoo --symbol SG.ES3 --lookback-days 7
```

Yahoo five-minute data is only a recent ES3 fallback; it cannot provide the
intended three-to-five-year intraday history.

## Recommended one-command workflow

Edit `config/market_data_pipeline.json` to choose each provider, symbol, and
date range, then run:

```bash
python run_market_data_pipeline.py
```

The entry point is a pipeline orchestrator. It uses strict stage barriers:

1. Every configured asset must finish ingestion.
2. Every configured asset must finish aggregation.
3. Every configured asset must finish backfill.
4. Consolidation runs only after all backfills finish.

An exception stops the process immediately, prints `Pipeline stopped`, and
prevents all later stages from being called. Files completed before the failure
remain available for diagnosis.

Each item in the configuration supports one of two range forms:

```json
{
  "provider": "alpaca",
  "symbol": "SPY",
  "start": "2023-09-02T00:00:00Z",
  "end": "2026-09-02T00:00:00Z"
}
```

or a rolling lookback:

```json
{
  "provider": "alpaca",
  "symbol": "SPY",
  "lookback_days": 1095
}
```

| Configuration field | Meaning |
|---|---|
| `assets` | Non-empty list of provider/symbol requests |
| `provider` | Provider alias or canonical provider name |
| `symbol` | Provider symbol configured in `config.py` |
| `start`, `end` | Exact half-open UTC range; both are required together |
| `lookback_days` | Rolling range alternative to `start`; `end` is optional |
| `output_dir` | Optional per-asset staging-directory override |
| `output_root` | Default staging root when `output_dir` is omitted |
| `canonical_output_file` | Final long-format canonical CSV path |
| `canonical_summary_file` | Consolidation summary JSON path |

Only one provider can be configured for each internal `instrument_id`. Source
resolution remains fixed at five minutes and aggregation remains fixed at 30
minutes in this phase; the configured timeframe controls the historical date
range, not the bar interval.

## Manual equivalent workflow

Use one exact half-open UTC range for all providers. Keep the temporary shared
range within Yahoo's recent intraday limit. These dates are an example:

```bash
DATA_START="2026-07-06T00:00:00Z"
DATA_END="2026-09-02T00:00:00Z"

python run_ingestion.py --provider alpaca --symbol SPY \
  --start "$DATA_START" --end "$DATA_END"
python run_ingestion.py --provider alpaca --symbol TLT \
  --start "$DATA_START" --end "$DATA_END"
python run_ingestion.py --provider binance --symbol BTCUSDT \
  --start "$DATA_START" --end "$DATA_END"
python run_ingestion.py --provider yahoo --symbol ES3.SI \
  --start "$DATA_START" --end "$DATA_END"
```

Aggregate each staging directory independently:

```bash
python run_aggregation.py --input-dir data/staging/alpaca_market_data/SPY \
  --output-dir data/staging/alpaca_market_data/SPY
python run_aggregation.py --input-dir data/staging/alpaca_market_data/TLT \
  --output-dir data/staging/alpaca_market_data/TLT
python run_aggregation.py --input-dir data/staging/binance_spot/BTCUSDT \
  --output-dir data/staging/binance_spot/BTCUSDT
python run_aggregation.py --input-dir data/staging/yahoo_finance/ES3_SI \
  --output-dir data/staging/yahoo_finance/ES3_SI
```

Apply backfill independently so no provider can fill another provider's bars:

```bash
python run_backfill.py --input-dir data/staging/alpaca_market_data/SPY \
  --output-dir data/staging/alpaca_market_data/SPY
python run_backfill.py --input-dir data/staging/alpaca_market_data/TLT \
  --output-dir data/staging/alpaca_market_data/TLT
python run_backfill.py --input-dir data/staging/binance_spot/BTCUSDT \
  --output-dir data/staging/binance_spot/BTCUSDT
python run_backfill.py --input-dir data/staging/yahoo_finance/ES3_SI \
  --output-dir data/staging/yahoo_finance/ES3_SI
```

Finally, consolidate the four selected sources:

```bash
python run_consolidation.py
```

The default consolidation selects Alpaca SPY and TLT, Binance BTCUSDT, and
Yahoo ES3.SI. To replace ES3 with another provider later, pass all selected
directories explicitly:

```bash
python run_consolidation.py \
  --input-dir data/staging/alpaca_market_data/SPY \
  --input-dir data/staging/alpaca_market_data/TLT \
  --input-dir data/staging/binance_spot/BTCUSDT \
  --input-dir path/to/the/replacement/es3/directory
```

Consolidation rejects duplicate bar keys and rejects more than one provider for
the same `instrument_id`. It unions the rows in long format and does not create
artificial observations while SPY, TLT, or ES3 is closed.

## Outputs

The `data/` directory is ignored by Git. Each ingestion replaces five stable
files inside its selected staging run directory:

```text
<staging-run>/market_bars_5m.csv
<staging-run>/rejected_market_bars_5m.csv
<staging-run>/missing_market_bars_5m.csv
<staging-run>/market_session_coverage.csv
<staging-run>/ingestion_summary.json
```

`market_bars_5m.csv` contains completed bars that passed validation. Plausible
but unusual rows remain present with `quality_status=warning` and an explanation
in `quality_flags`.

For Alpaca rows, raw OHLCV and adjusted close are joined by UTC bar timestamp.
The adjustment affects analysis only: raw prices remain available for orders and
position valuation. No separate dividend cash-flow file is created in this data
phase, so adjusted-return analysis must not add dividends again.

`rejected_market_bars_5m.csv` retains invalid, duplicate, or incomplete rows and
records the reason. It also retains a provider row if its timestamp falls
outside the instrument's expected regular session.

`missing_market_bars_5m.csv` contains timestamps where a completed five-minute
bar was expected while the market was open but no accepted bar was available.
These are data-quality gaps and are not filled with an earlier price.

`market_session_coverage.csv` gives one row per local calendar date and
instrument. `session_status` distinguishes an open session, weekend,
holiday/scheduled closure, or continuous 24/7 trading. It also reports expected,
accepted, and missing bars plus the coverage ratio.

`ingestion_summary.json` provides compact counts and coverage statistics for
each instrument. Its `price_policy` records that OHLCV is raw, Alpaca adjusted
close uses all supported corporate-action adjustments, and dividend cash flows
are not tracked by this phase.

The separate aggregation run replaces three additional files:

```text
<staging-run>/market_bars_30m.csv
<staging-run>/incomplete_market_bars_30m.csv
<staging-run>/aggregation_summary.json
```

`market_bars_30m.csv` contains only windows with all six expected five-minute
bars. OHLCV follows standard candlestick aggregation: first open, maximum high,
minimum low, last close, and summed volume, quote volume, and trade count.

`incomplete_market_bars_30m.csv` records an otherwise valid 30-minute window
when one or more expected source bars are absent. It stores expected and actual
counts, coverage ratio, and the missing timestamps, but does not create
potentially misleading partial OHLCV values.

`aggregation_summary.json` reports complete and incomplete counts by instrument
and explicitly records that no refetch was attempted. The backfill module uses
the incomplete-window file as its work list.

Partial 30-minute windows caused only by the requested start/end boundary or by
a bar that has not closed yet are not treated as missing-data failures. They are
simply not eligible for aggregation yet.

The backfill run replaces four additional files:

```text
<staging-run>/backfilled_market_bars_30m.csv
<staging-run>/market_bars_30m_with_backfill.csv
<staging-run>/unresolved_market_bars_30m.csv
<staging-run>/backfill_summary.json
```

`backfilled_market_bars_30m.csv` contains only incomplete windows resolved by
the approved methods. `market_bars_30m_with_backfill.csv` combines those rows
with the original complete bars and is the convenient input for later feature
work. `unresolved_market_bars_30m.csv` retains every unsupported case and its
reason. `backfill_summary.json` gives compact method and outcome counts.

The consolidation run replaces two canonical outputs:

```text
data/canonical/market_bars_30m.csv
data/canonical/consolidation_summary.json
```

`market_bars_30m.csv` is the long-format union of the selected primary source
for each instrument. It retains `source`, `bar_origin`, backfill provenance, and
quality columns. Rows are not aligned into a wide matrix at this stage because
the four assets do not all share the same trading session. The summary records
row counts, date coverage, source symbols, and bar origins for each instrument.

## Backfill rules

A 30-minute window contains six expected five-minute bars, labelled `t0` to
`t5`. Backfill is attempted only when at least four are observed.

- If only middle bars are missing, open and close remain the observed `t0` and
  `t5` values. High is the maximum observed high, low is the minimum observed
  low, and volume and trade counts sum the observed bars only.
- If `t0` is missing, open uses the immediately preceding accepted five-minute
  bar's close. The bar must be exactly five minutes earlier and, for equities,
  in the same trading session.
- If `t5` is missing, close uses the immediately following accepted five-minute
  bar's open. It must start exactly at the 30-minute window close and, for
  equities, be in the same trading session. Its adjusted close is calculated by
  applying that following bar's adjusted-to-raw price factor to the borrowed raw
  open.
- If both boundaries are missing, both adjacent observations must be available.
  A boundary value is never borrowed across an overnight close, weekend,
  holiday, or the SGX lunch break. BTCUSDT is continuous and may cross a UTC
  date boundary.

The range and volume of any resolved incomplete window are explicitly marked
as partial. Provenance columns distinguish complete derived bars, partial
observed bars, and synthetic boundary backfills. No standard deviation is
calculated here; volatility belongs in the later feature module. Cases outside
these rules remain unresolved for the group to decide rather than being filled
silently.

## Standard columns

| Column | Description |
|---|---|
| `instrument_id` | Stable internal instrument identifier |
| `source` | Provider identifier, including `alpaca_market_data`, `moomoo_openapi`, `yahoo_finance`, or `binance_spot` |
| `source_symbol` | Provider symbol |
| `asset_class` | Instrument class |
| `exchange` | Trading venue |
| `bar_interval` | Always `5m` in this phase |
| `bar_open_time_utc` | Inclusive bar start in UTC |
| `bar_close_time_utc` | Exclusive bar end in UTC |
| `session_date_local` | Date in the instrument's local session timezone |
| `open`, `high`, `low`, `close` | Unadjusted source OHLC prices |
| `adjusted_close` | Corporate-action-adjusted analytical close for Alpaca and Yahoo when supplied; equals raw close for BTC and unadjusted sources |
| `volume` | Source-native base volume |
| `volume_unit` | `shares`, `units`, or `BTC` |
| `quote_volume` | Moomoo SGD turnover or Binance USDT turnover; blank for Alpaca |
| `quote_currency` | USD, SGD, or USDT |
| `trade_count` | Alpaca or Binance trade count; blank when Moomoo omits it |
| `is_final` | Whether the source bar has closed |
| `retrieved_at_utc` | Extraction timestamp |
| `quality_status` | `pass` or `warning` for accepted rows |
| `quality_flags` | Semicolon-separated warning explanations |

The same schema does not make volume directly comparable. SPY and TLT volume is
shares, SG.ES3 volume is ETF units, and BTCUSDT volume is bitcoin.

Use `adjusted_close` for historical returns, volatility, momentum, RSI, and
other close-based model features. Use raw `open`/`close` for simulated or live
fills and raw `close` for actual position market value. An adjusted-return
portfolio is a total-return index; it is not a cash-and-share ledger and does
not expose dividends as spendable cash.

## Validation rules

Rows are rejected when they contain:

- missing timestamps or prices
- incomplete bars
- non-positive prices
- impossible OHLC relationships
- negative volume or trade counts
- duplicate instrument/source/interval/open-time keys
- bars outside the expected regular trading session

Zero volume and an intrabar range above 20% are retained as warnings.

The session check uses the `XNYS` US-equity calendar for SPY and TLT, `XSES`
for SG.ES3, and a continuous UTC calendar for BTCUSDT. Regular session hours,
exchange breaks, weekends, holidays, and early closes determine which bars
should exist. A weekend or holiday produces no rejected bars because no market
data was expected. A missing bar during an open session is written to
`missing_market_bars_5m.csv` instead.

The SGX regular-session break from 12:00 to 13:00 Singapore time is configured
explicitly because the calendar package currently supplies XSES holiday and
session dates but does not include that break in its schedule.

The cleaner never forward-fills or silently invents prices. This means a later
feature module can calculate intraday returns within `session_date_local` and
treat overnight movements separately, without learning artificial zero returns
while an equity market was closed.

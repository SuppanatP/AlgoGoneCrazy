# FE5209 Group Project - Market Data Ingestion

This first project phase extracts and cleans completed five-minute source bars
for:

- `SPY` from Yahoo Finance
- `ES3.SI` from Yahoo Finance
- `BTCUSDT` from Binance Spot

It intentionally does not aggregate bars, calculate indicators, generate
signals, write to PostgreSQL, or simulate trades. Those belong in later modules.

## Design

Extraction and cleaning are separate modules but run in one application:

```text
Yahoo / Binance
      -> source-specific extraction
      -> common standardization
      -> common validation
      -> accepted and rejected CSV files
```

Five minutes is the source resolution. It preserves enough detail to construct
15-minute, 30-minute, hourly, or two-hour bars later without committing the
strategy to one aggregation schedule now.

## Project layout

```text
src/market_data/
  config.py              Instrument metadata and ingestion settings
  extractors/yahoo.py    SPY and ES3.SI extraction
  extractors/binance.py  BTCUSDT extraction and pagination
  standardize.py         Common provider-neutral columns
  validate.py            Shared quality rules
  pipeline.py            Command-line orchestration and outputs
test/                    Offline tests using synthetic data
run_ingestion.py         Convenient entry point
```

## Setup

From the repository root:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

Yahoo and Binance public market-data downloads do not require API keys.

## Run

Download the latest 59 days, which stays within Yahoo's intraday-history limit:

```bash
python run_ingestion.py
```

Use a shorter lookback:

```bash
python run_ingestion.py --lookback-days 14
```

Or provide an exact half-open UTC range (`start <= bar open < end`):

```bash
python run_ingestion.py \
  --start 2026-08-01T00:00:00Z \
  --end 2026-08-28T00:00:00Z
```

Yahoo intraday requests cannot extend beyond the latest 60 days. The CLI limits
automatic lookback to 59 days; manually supplied dates must respect the same
provider restriction.

## Outputs

The `data/` directory is ignored by Git. Each run replaces three stable files:

```text
data/market_bars_5m.csv
data/rejected_market_bars_5m.csv
data/ingestion_summary.json
```

`market_bars_5m.csv` contains completed bars that passed validation. Plausible
but unusual rows remain present with `quality_status=warning` and an explanation
in `quality_flags`.

`rejected_market_bars_5m.csv` retains invalid, duplicate, or incomplete rows and
records the reason. The cleaner never forward-fills or silently invents prices.

## Standard columns

| Column | Description |
|---|---|
| `instrument_id` | Stable internal instrument identifier |
| `source` | `yahoo_finance` or `binance_spot` |
| `source_symbol` | Provider symbol |
| `asset_class` | Instrument class |
| `exchange` | Trading venue |
| `bar_interval` | Always `5m` in this phase |
| `bar_open_time_utc` | Inclusive bar start in UTC |
| `bar_close_time_utc` | Exclusive bar end in UTC |
| `session_date_local` | Date in the instrument's local session timezone |
| `open`, `high`, `low`, `close` | Unadjusted source OHLC prices |
| `adjusted_close` | Yahoo adjusted close when available; BTC equals close |
| `volume` | Source-native base volume |
| `volume_unit` | `shares`, `units`, or `BTC` |
| `quote_volume` | Binance USDT turnover; blank for Yahoo |
| `quote_currency` | USD, SGD, or USDT |
| `trade_count` | Binance trade count; blank for Yahoo |
| `is_final` | Whether the source bar has closed |
| `retrieved_at_utc` | Extraction timestamp |
| `quality_status` | `pass` or `warning` for accepted rows |
| `quality_flags` | Semicolon-separated warning explanations |

The same schema does not make volume directly comparable. SPY volume is shares,
ES3.SI volume is ETF units, and BTCUSDT volume is bitcoin.

## Validation rules

Rows are rejected when they contain:

- missing timestamps or prices
- incomplete bars
- non-positive prices
- impossible OHLC relationships
- negative volume or trade counts
- duplicate instrument/source/interval/open-time keys

Zero volume and an intrabar range above 20% are retained as warnings. Missing
bars are not manufactured because exchange closures and genuine no-trade
intervals must remain distinguishable from provider failures.

## Test without downloading data

```bash
python -m pytest -q
```

The tests use synthetic Yahoo and Binance responses and make no network calls.

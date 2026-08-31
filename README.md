# FE5209 Group Project - Market Data Pipeline

This first project phase extracts and cleans completed five-minute source bars
for:

- `SPY` from Yahoo Finance
- `ES3.SI` from Yahoo Finance
- `BTCUSDT` from Binance Spot

It also provides separate steps that derive 30-minute OHLCV bars and apply a
small set of conservative backfill methods. It intentionally does not calculate
indicators, generate signals, write to PostgreSQL, or simulate trades. Those
belong in later modules.

## Design

Extraction and cleaning are separate modules but run in one application:

```text
Yahoo / Binance
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
  sessions.py            Expected bars from XNYS, XSES, and 24/7 calendars
  coverage.py            Off-session and missing-expected-bar checks
  pipeline.py            Command-line orchestration and outputs
  aggregation.py         Complete 30m OHLCV construction and gap reporting
  backfill.py            Conservative 30m gap handling and provenance
test/                    Offline tests using synthetic data
run_ingestion.py         Convenient entry point
run_aggregation.py       Separate 30m aggregation entry point
run_backfill.py          Separate non-destructive backfill entry point
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

After ingestion finishes, generate the 30-minute bars:

```bash
python run_aggregation.py
```

Use different input and output directories if required:

```bash
python run_aggregation.py --input-dir data --output-dir data
```

The aggregation command only reads existing ingestion outputs. It does not call
Yahoo or Binance and does not attempt to refetch a missing five-minute interval.

Then apply the currently approved backfill methods:

```bash
python run_backfill.py
```

The normal sequence is therefore ingestion, aggregation, then backfill. The
backfill command reads the existing CSV files, does not call a provider, and
does not overwrite either aggregation input file.

## Outputs

The `data/` directory is ignored by Git. Each run replaces five stable files:

```text
data/market_bars_5m.csv
data/rejected_market_bars_5m.csv
data/missing_market_bars_5m.csv
data/market_session_coverage.csv
data/ingestion_summary.json
```

`market_bars_5m.csv` contains completed bars that passed validation. Plausible
but unusual rows remain present with `quality_status=warning` and an explanation
in `quality_flags`.

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
each instrument.

The separate aggregation run replaces three additional files:

```text
data/market_bars_30m.csv
data/incomplete_market_bars_30m.csv
data/aggregation_summary.json
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
data/backfilled_market_bars_30m.csv
data/market_bars_30m_with_backfill.csv
data/unresolved_market_bars_30m.csv
data/backfill_summary.json
```

`backfilled_market_bars_30m.csv` contains only incomplete windows resolved by
the approved methods. `market_bars_30m_with_backfill.csv` combines those rows
with the original complete bars and is the convenient input for later feature
work. `unresolved_market_bars_30m.csv` retains every unsupported case and its
reason. `backfill_summary.json` gives compact method and outcome counts.

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
  equities, be in the same trading session.
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
- bars outside the expected regular trading session

Zero volume and an intrabar range above 20% are retained as warnings.

The session check uses the `XNYS` calendar for SPY, `XSES` for ES3.SI, and a
continuous UTC calendar for BTCUSDT. Regular session hours, exchange breaks,
weekends, holidays, and early closes determine which bars should exist. A
weekend or holiday produces no rejected bars because no market data was
expected. A missing bar during an open session is written to
`missing_market_bars_5m.csv` instead.

The SGX regular-session break from 12:00 to 13:00 Singapore time is configured
explicitly because the calendar package currently supplies XSES holiday and
session dates but does not include that break in its schedule.

The cleaner never forward-fills or silently invents prices. This means a later
feature module can calculate intraday returns within `session_date_local` and
treat overnight movements separately, without learning artificial zero returns
while an equity market was closed.

## Test without downloading data

```bash
python -m pytest -q
```

The tests use synthetic Yahoo and Binance responses and make no network calls.

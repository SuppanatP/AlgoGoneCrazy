# FE5209 Group Project - Market Data Pipeline

This README documents the retained strategy-selection workflow, the frozen
SPY/STI selection, the standalone BTC strategy, and the final three-asset
portfolio policy. Generated selection outputs remain local and are not part
of the tracked repository.

This first project phase extracts and cleans completed five-minute source bars
for:

- `SPY` from Alpaca Market Data
- `SG.ES3` from Moomoo OpenAPI when entitled, or recent `ES3.SI` from Yahoo
- `BTCUSDT` from Binance Spot

It also provides separate steps that derive 30-minute OHLCV bars and apply a
small set of conservative backfill methods. A separate strategy package
generates momentum and mean-reversion signals, and a historical backtesting
entry point connects those signals to the shared trade tracker. A reproducible
pre-cutoff grid-search workflow selects the retained SPY/STI parameters. A
strategy-agnostic sizing layer can produce capped
inverse-volatility or equal-weight risk budgets before signals are applied.
Live execution and PostgreSQL remain later phases.

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

Pre-cutoff canonical bars
      -> exact SPY momentum and STI mean-reversion parameter grids
      -> training-period screening with the final execution and cost model
      -> joint pre-cutoff validation under the selection-time sizing policy
      -> reviewed parameters frozen in the tracked strategy configuration

Standalone SPY/STI/BTC strategy targets
      -> daily after-cost strategy return histories before the run starts
      -> inverse strategy-return volatility sleeves
      -> cap STI at 10%, SPY/BTC at 75%, and reserve 5% cash
      -> apply the configured materially-active capital-scaling policy

Selected SPY/STI strategies + stateful BTC target export
      -> delay SPY/STI to the next eligible raw open; execute BTC at its listed open
      -> apply frozen risk sleeves and point-in-time USD conversion
      -> execute through one USD cash and position ledger
      -> decisions, fills, trades, equity, costs, and per-asset P&L
```

### Configuration authority

The configuration files have distinct roles. They are not interchangeable:

| File | Authoritative role |
|---|---|
| `config/selected_strategies.json` | Frozen SPY/STI rules and the BTC target-file interface |
| `config/btc_strategy.json` | Standalone BTC models, voting, state, timing, shorting, and cost assumptions |
| `config/reproduction_manifest.json` | Final selection command, input fingerprints, verified environment, and expected SPY/STI winners |
| `config/sti_mean_reversion_expanded_optimization.json` | Exact SPY/STI search grids, training windows, validation windows, and cutoff |
| `config/strategy_search_baseline.json` | Neutral comparator supplied to the grid search; not a selected result |
| `config/portfolio_sizing.json` | Historical history-confidence sizing used only to reproduce SPY/STI pair selection |
| `config/three_asset_portfolio.json` | Final integrated SPY/STI/BTC sizing, caps, cash reserve, and active-capital policy |
| `config/cost_assumptions.json` | Broker commission assumptions used by selection and simulation |
| `config/risk_free_rate.json` | Risk-free benchmark and daily performance-reporting convention |

The final simulation therefore combines the signals in
`config/selected_strategies.json` with the portfolio policy in
`config/three_asset_portfolio.json`. The older selection-time sizing remains
tracked only because it is needed to reproduce how the SPY/STI pair was chosen.

### Information timing and look-ahead controls

- SPY and STI features include the current completed decision bar. Their
  resulting targets execute only at the next eligible underlying bar open,
  using raw tradable prices.
- BTC feature row `t` is shifted to use market observations only through
  `t-1`; its exported target therefore applies directly at the raw open of
  bar `t` and must not receive another delay.
- Training and validation periods are chronological. Forward BTC labels that
  overlap the next split are purged, and the label threshold is estimated from
  training rows only.
- Portfolio risk estimates and sleeve weights use observations strictly before
  the sizing timestamp. Evaluation-period returns cannot enter the initial
  allocation.
- SGD/USD and Treasury observations become usable only after their configured
  availability lag. Treasury rates affect Sharpe and Sortino reporting only;
  simulated cash does not earn interest.
- A backfilled bar may preserve feature continuity, but only final bars with
  `bar_origin=derived_complete` may change a SPY/STI strategy position. A
  synthetic boundary bar therefore cannot directly create an executable
  signal.

Adjusted closes are used for analytical features, while raw opens and closes
are used for simulated execution and valuation. This is the project's stated
corporate-action convention; separate dividend cash flows and point-in-time
corporate-action snapshots are not modelled.

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
  extractors/alpaca.py   Alpaca raw/adjusted REST extraction and pagination
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
src/strategy_optimizer/
  two_week.py            Grid generation, pre-cutoff screening, and joint validation
src/portfolio_sizing/
  inverse_volatility.py  Daily volatility, equal/inverse-vol capped schedules
  fx.py                  Point-in-time conversion into the USD portfolio base
  fred_csv.py            Downloaded FRED SGD/USD CSV standardization
  targets.py             Signal gating and base-currency target quantities
src/portfolio_simulation.py Shared signals, USD execution, and portfolio replay
src/strategy_backtest.py Strategy-to-backtester adapter and output writer
src/backtest.py          Reused bar-by-bar backtest engine
src/tracker.py           Reused cash, position, fill, trade, and cost ledger
src/metrics.py           Daily report risk and intraday drawdown/cost metrics
src/risk_free.py         Official Treasury risk-free-rate CSV loader
src/features.py          Lagged BTC features shared by training and standalone replay
src/strategy.py          Stateful BTC ML/regime/voting strategies
config/
  market_data_pipeline.json  Selected assets, providers, ranges, and paths
  reproduction_manifest.json Final selection recipe and input fingerprints
  sti_mean_reversion_expanded_optimization.json Exact SPY/STI grids and periods
  strategy_search_baseline.json Neutral grid-search input comparator
  portfolio_sizing.json      Historical SPY/STI selection sizing configuration
  three_asset_portfolio.json Final three-asset sleeves, caps, cash, and quantities
  risk_free_rate.json        Daily risk metric and Treasury benchmark convention
  selected_strategies.json   Frozen SPY/STI rules and BTC target interface
  btc_strategy.json          Existing standalone BTC strategy reproducibility manifest
test/                    Offline tests using synthetic data
run_ingestion.py         Convenient entry point
run_aggregation.py       Separate 30m aggregation entry point
run_backfill.py          Separate non-destructive backfill entry point
run_consolidation.py     Canonical three-asset CSV entry point
run_market_data_pipeline.py  One-command sequential workflow entry point
run_strategy_backtest.py SPY/ES3 momentum and mean-reversion backtest entry point
run_two_week_strategy_optimization.py Reproducible SPY/STI selection entry point
run_fx_ingestion.py       Downloaded FRED SGD/USD standardization entry point
run_portfolio_weights.py  Rolling inverse-volatility/equal-weight entry point
run_portfolio_simulation.py Shared-USD portfolio replay entry point
run_daily.py              Stateful standalone BTC replay entry point
report_metrics.py         Standalone BTC metrics and benchmark report
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

### Reproduce the selected SPY/STI parameters

The retained workflow searches SPY momentum and ES3.SI mean-reversion
parameters using only observations before the exclusive `2026-10-01T00:00:00Z`
boundary. All timestamps are UTC; every period uses an inclusive start and
exclusive end. BTC model selection is separate and is documented by
`config/btc_strategy.json`.

The complete and exact search space is tracked in
`config/sti_mean_reversion_expanded_optimization.json`:

| Asset | Training period | Search grid | Candidates |
|---|---|---|---:|
| SPY momentum | 18 Sep 2023–1 Oct 2025 | Decision interval 30/60/120m; fast horizon 1/2 sessions; slow horizon 4/6 sessions; return lookback 2/4 sessions; entry threshold 0.25%/0.50%; exit threshold 0% | 48 |
| ES3.SI mean reversion | 6 Jul–1 Sep 2026 | Decision interval 120m; lookback 2/3/4/5/6/8/10 sessions; entry z-score -1.25/-1.50/-1.75/-2.00/-2.25/-2.50; exit z-score -0.50/-0.25/0/0.25; minimum hold 1 session; cooldown 1 session; maximum hold 5 sessions | 168 |

Lookback horizons are converted from trading sessions to decision bars, so
changing the decision interval does not silently change the economic horizon.
Each candidate uses adjusted closes for features, the next eligible raw open
for execution, raw closes for valuation, historical SGD/USD conversion, whole
shares, and `config/cost_assumptions.json` commissions.

The workflow first screens each full grid over its configured training period.
It then reruns shortlisted candidates over independent rolling 14-session
windows. Trade frequency is reported but is not an eligibility or ranking
criterion. Finally, shortlisted SPY/STI pairs are compared jointly from
`2026-09-11T00:00:00Z` to the exclusive cutoff using the history-confidence
inverse-volatility policy in `config/portfolio_sizing.json`. The pair with the
highest after-cost joint validation return is the selected pair.

Run the reproducible selection:

```bash
python run_two_week_strategy_optimization.py \
  --input-file data/canonical/market_bars_30m_training_validation.csv \
  --fx-file data/fx/fx_rates_to_usd.csv \
  --config-file config/sti_mean_reversion_expanded_optimization.json \
  --selected-strategies config/strategy_search_baseline.json \
  --sizing-config config/portfolio_sizing.json \
  --cost-config config/cost_assumptions.json \
  --output-dir results/strategy_selection \
  --progress-every 10
```

The neutral baseline is an input comparator; it does not feed the frozen
selection back into its own search. The command writes generated files only to
the ignored output directory and never edits `config/selected_strategies.json`.
The expected winner is SPY candidate `spy_usd__momentum_0041` paired with STI
candidate `sti_es3_sgd__mean_reversion_0052`. Their complete reviewed parameter
values are frozen in `config/selected_strategies.json`.

`config/reproduction_manifest.json` is the tracked recipe for this run. It
records the same command, the SHA-256 fingerprints of the separately supplied
market and FX data, all authoritative input configurations, the verified Python
environment, and the expected winners. A fingerprint mismatch means the input
snapshot or configuration is different and the result need not reproduce.

Important generated outputs are:

- `candidate_screening.csv`: every grid candidate's full-period result;
- `two_week_independent_windows.csv`: reset-window shortlist diagnostics;
- `shortlisted_candidate_review.csv`: per-asset shortlist ranking;
- `joint_diagnostic_summary.csv`: joint pair results used for final selection;
- `optimization_manifest.json`: periods, grids, execution, costs, and sizing snapshot.

For a quick workflow check only, add `--max-candidates 2`. Add
`--skip-joint-diagnostic` only when checking the per-asset stages; a run without
joint validation cannot reproduce the final pair.

### Portfolio sizing

Portfolio sizing is separate from signal generation. SPY momentum and STI mean
reversion supply long-or-cash targets. BTC supplies its existing stateful target
between `-1` and `1`, including full, partial, fallback, flat, and short values.
Each target scales only that strategy's sleeve; a full BTC target no longer
means using the entire three-asset portfolio.

The combined portfolio reserves 5% strategic cash and caps STI at 10% because
it is intentionally the smallest exposure. SPY and BTC may each use at most
75%. The remaining allocation is determined by inverse volatility of each
standalone strategy's daily after-cost returns, using 60 observations with a
minimum of 40. This compares realised strategy risk rather than assuming equal
dollars create equal risk. No Sharpe or return forecast enters the calculation.

Sleeve weights are calculated once at the simulation start from observations
strictly before that timestamp and then frozen. Holdout-period performance
therefore cannot influence the allocation. The integration layer applies a
bounded active-capital rule. A strategy qualifies for the scaling gate only
when its absolute target reaches the configured materiality threshold: 1.0 for
SPY and STI and 0.5 for BTC. Small BTC fallback targets of 0.10 or 0.05 still
trade normally but do not unlock portfolio-wide expansion. With fewer than two
qualifying strategies, unused capital remains cash; with at least two, all
nonzero signed targets may be scaled while preserving their direction and
relative intensity. No target may exceed twice its raw sleeve target, the asset
cap, or the portfolio's 95% gross limit. STI therefore remains capped at 10%.
Target changes smaller than five percentage points are ignored to limit
resizing costs. Whole shares are used for SPY and ES3.SI; BTC uses fractional
units.

`config/three_asset_portfolio.json` is the sole retained integration policy. It
enables bounded active-capital scaling and allows thresholded resizing so that
the portfolio targets can take effect.

The portfolio simulation starts with USD 500,000 in cash and no positions.
There are no later external deposits or withdrawals. Sale proceeds, dividends,
and other investment cash flows would remain inside the portfolio rather than
being treated as capital injections. The current source snapshot does not yet
contain separate dividend cash flows.

Download the maximum-history CSV for official FRED series `DEXSIUS` from
`https://fred.stlouisfed.org/series/DEXSIUS` and save the unmodified file at:

```text
data/fx/raw/DEXSIUS.csv
```

Standardize the downloaded snapshot:

```bash
python run_fx_ingestion.py
```

No FRED account or API key is used by the code. The standardizer reads the
downloaded `observation_date,DEXSIUS` file, discards FRED's missing-value rows,
and stores `1 / DEXSIUS` as USD per SGD. The configured one-day availability
lag prevents a same-day Federal Reserve fixing from being treated as known
earlier that day. Re-running the command deterministically replaces the
standardized output from the downloaded snapshot. Optional `--start` and
`--end` arguments apply inclusive-start/exclusive-end filtering.

The downloaded CSV, standardized FX CSV, and FX summary JSON are data
artifacts beneath `data/`; they are ignored by Git and must not be committed.

Build weights for the three configured assets:

```bash
python run_portfolio_weights.py \
  --fx-file data/fx/fx_rates_to_usd.csv
```

Build the final integration sleeves after generating the BTC target file:

```bash
python run_portfolio_weights.py \
  --config-file config/three_asset_portfolio.json \
  --input-file data/canonical/market_bars_30m_training_validation.csv \
  --start 2026-09-01 \
  --end 2026-09-02 \
  --fx-file data/fx/fx_rates_to_usd.csv \
  --output-dir results/three_asset_portfolio_weights
```

Build the equal-weight schedule instead of the configured inverse-volatility
default:

```bash
python run_portfolio_weights.py \
  --sizing-method capped_equal_weight \
  --fx-file data/fx/fx_rates_to_usd.csv
```

Build a three-asset diagnostic schedule from the backtest snapshot:

```bash
python run_portfolio_weights.py \
  --input-file data/canonical/market_bars_30m_training_validation.csv \
  --instrument SPY ES3.SI BTCUSDT \
  --fx-file data/fx/fx_rates_to_usd.csv
```

Generated `portfolio_weights.csv`, `volatility_diagnostics.csv`,
`currency_conversion.csv`, and `portfolio_sizing_manifest.json` files are written beneath
`results/portfolio_sizing/` and ignored by Git.

USD is the portfolio base and final reporting currency. USD assets use a rate of
1, USDT is treated as USD at a fixed 1:1 rate for this project, and SGD assets
must use timestamped SGD-to-USD rates. Prices are converted before volatility
and weights are calculated, so the risk calculation and eventual portfolio
totals use one currency. Raw market prices retain their native `quote_currency`;
the sizing command verifies that metadata against its configuration. The
conversion audit records both `quote_currency` and `base_currency`.

The FX file is long format. `rate_to_base` means USD per one unit of the quote
currency; for example, `0.78` means SGD 1 = USD 0.78:

```csv
rate_time_utc,quote_currency,base_currency,rate_to_base,source,source_series,source_observation_date,is_forward_filled
2026-09-29T00:00:00Z,SGD,USD,0.781,fred_download,DEXSIUS,2026-09-28,false
2026-09-30T00:00:00Z,SGD,USD,0.780,fred_download,DEXSIUS,2026-09-29,false
```

The converter uses the latest rate available on or before each price date and
never fills backwards from a future rate. An FX file is not needed when every
selected asset is USD or USDT, for example `--instrument SPY BTCUSDT`.
Historical FX coverage is required when STI is selected; it should cover the
volatility lookback as well as the reporting period.

The reusable `target_quantities` function requires base-currency prices,
preventing accidental addition of USD, SGD, and USDT notionals. The portfolio
simulation uses the same sizing assumptions when it converts strategy targets
into actual quantities in one USD ledger.

The currently selected SPY and ES3.SI parameters are stored in
`config/selected_strategies.json`; the optimiser never changes that file
automatically. BTCUSDT uses a reproducible target export from the tracked models
and voting configuration.

| Asset | Signal method | Decision interval | Frozen selection |
|---|---|---:|---|
| SPY | Momentum | 120 minutes | `spy_usd__momentum_0041` |
| ES3.SI | Mean reversion | 120 minutes | `sti_es3_sgd__mean_reversion_0052` |
| BTCUSDT | LightGBM gate plus stateful voting | 30 minutes | `config/btc_strategy.json` |

Generate BTC targets from the same canonical snapshot used by the portfolio.
Start early enough to supply the pre-period strategy-risk history:

```bash
python export_btc_targets.py \
  --input-file data/canonical/market_bars_30m_training_validation.csv \
  --start 2026-06-01 \
  --end 2026-09-30 \
  --output records/btc_target_exports/btc_targets.csv
```

The generated file is ignored. Its signed fractional
`execution_target_position` applies at the listed BTC bar open because all BTC
features already end at the preceding bar; the portfolio simulator does not
delay BTC again. SPY and STI decisions still execute at the next eligible open.
BTC strategy state advances sequentially during export. The standalone
`state.json` portfolio cash is not imported into the shared ledger.

Run the integrated policy over a pre-cutoff period after generating BTC
targets:

```bash
python run_portfolio_simulation.py \
  --input-file data/canonical/market_bars_30m_training_validation.csv \
  --start 2026-09-01 \
  --end 2026-10-01 \
  --sizing-config config/three_asset_portfolio.json \
  --output-dir results/material_active_scaling_pre_holdout
```

The output directory is ignored. Review `performance_summary.csv`,
`asset_summary.csv`, and the raw-versus-scaled target columns in
`decisions.csv`. A missing BTC target file stops a complete run;
`--allow-missing-external` is only a temporary connectivity mode and must not
be used for policy selection. The simulator uses one USD 500,000
ledger, historical
SGD/USD rates, USDT/USD at 1:1, whole units for SPY/ES3.SI, fractional BTC,
configured broker commissions, no separate FX transaction fee, and the next
eligible raw open. It freezes sleeve weights at the run's start. Under the
default configuration, bounded scaling can deploy some otherwise idle cash
only when at least two strategies are materially active. The generated
`decisions.csv` records both the number of nonzero targets and the number that
qualify for the scaling gate so the policy remains auditable.

### Portfolio risk metrics and risk-free rate

Execution and valuation remain intraday. Report-level annualised volatility,
Sharpe, and Sortino use complete UTC-day changes in total portfolio equity so
that adjacent 30-minute returns are not assumed to be independent. The final
incomplete UTC day, if any, is omitted only from those daily risk statistics;
total return, fees, positions, and intraday drawdown still use the complete
simulation. Because BTC can change portfolio value every calendar day, daily
risk metrics use 365 periods per year.

The Sharpe benchmark is the official U.S. Treasury 3-month constant-maturity
yield from the [U.S. Department of the Treasury daily par yield curve](https://home.treasury.gov/resource-center/data-chart-center/interest-rates/TextView?type=daily_treasury_yield_curve).
The tracked convention is recorded in `config/risk_free_rate.json`. Download
the 2026 official CSV before running portfolio simulations:

```bash
mkdir -p data/risk_free/raw
curl -L --fail \
  'https://home.treasury.gov/resource-center/data-chart-center/interest-rates/daily-treasury-rates.csv/2026/all?type=daily_treasury_yield_curve&field_tdr_date_value=2026&page&_format=csv' \
  -o data/risk_free/raw/us_treasury_yield_curve_2026.csv
```

The downloaded CSV is ignored by Git and is supplied with the project data.
Each published yield is made available on the following UTC calendar date, so
weekends and holidays use the latest previously known rate without look-ahead.
Cash does not earn interest in the simulation; the Treasury rate is used only
as the opportunity-cost benchmark for Sharpe and Sortino.
For daily portfolio return `r_d` and annual Treasury yield `y_d`:

```text
risk_free_daily_d = (1 + y_d)^(1/365) - 1
annual_volatility = sample_std(r_d) * sqrt(365)
annual_sharpe = mean(r_d - risk_free_daily_d) / sample_std(r_d) * sqrt(365)
```

Annual return is elapsed-calendar-time CAGR. Maximum drawdown continues to use
the full intraday equity curve. The generated `simulation_manifest.json`
records the source, maturity, lag, annualisation and partial-day conventions.

Evaluation-period market data and generated targets are supplied separately in
ignored paths. Do not rerun the optimiser or revise frozen parameters after
the `2026-10-01T00:00:00Z` selection boundary.

Outputs beneath `results/portfolio_simulation/` include `decisions.csv`,
`fills.csv`, `trades.csv`, `equity_curve.csv`, `performance_summary.csv`,
`asset_summary.csv`, risk and FX diagnostics, and a reproducibility manifest.
Positions remain open at the final close to match a live-period snapshot. Raw
prices are used for execution and valuation; separate dividend cash flows are
not yet present in the source data and are recorded as excluded in the manifest.

## Standalone BTC strategy baseline

The existing BTC workflow is separate from the SPY momentum and STI
mean-reversion modules. It loads the tracked LightGBM up/down models, calculates
lagged features, combines technical-rule votes, and maintains a stateful target
through holding periods, partial exits, and a daily fallback. The reproducibility
manifest is `config/btc_strategy.json`. The executable `config.py` loads its BTC
model, data, strategy, and cost values from that manifest instead of maintaining
a second independent copy. Standalone replay and target export also verify the
selected model-file hashes before processing any bars. Runtime `state.json`
files remain ignored because they describe a particular run rather than the
strategy methodology.

The runner reads `btc_usdt` rows from the frozen canonical training/validation
file. When `data/canonical/market_bars_30m_simulation.csv` exists,
it is appended as the later source and wins on duplicate timestamps. Both files
are ignored and supplied separately from the repository.

BTC timing differs from the SPY/STI rule modules. A feature row at timestamp
`t` uses market data only through `t-1`; the resulting target is executed at the
raw open of bar `t`. Targets may be full, partial, flat, or short. This standalone
baseline must not be passed through an additional next-bar delay during later
portfolio integration.

On macOS, LightGBM also requires the OpenMP runtime:

```bash
brew install libomp
python -m pip install -r requirements.txt
```

Run an isolated short replay without overwriting another run:

```bash
python run_daily.py \
  --records-dir records/btc_short \
  --start 2026-09-01 \
  --end 2026-09-03

python report_metrics.py \
  --records records/btc_short \
  --output-dir results/btc_short \
  --no-plot
```

Run the pre-holdout standalone diagnostic used for integration review:

```bash
python run_daily.py \
  --records-dir records/btc_pre_holdout \
  --start 2026-06-01 \
  --end 2026-09-30

python report_metrics.py \
  --records records/btc_pre_holdout \
  --output-dir results/btc_pre_holdout \
  --no-plot
```

The records and reports are generated beneath ignored `records/` and `results/`
directories. They are not committed. Use a new records directory for each clean
replay; `--reset` deletes only the selected records directory.

Export the identical stateful target path without creating a portfolio, cash
ledger, fills, or fees:

```bash
python export_btc_targets.py \
  --start 2026-09-01 \
  --end 2026-09-03 \
  --output records/btc_target_exports/btc_targets_2026-09-01_to_03.csv
```

The export contains the model probabilities, active individual rule votes,
weighted vote score, entry signal, stateful target, and reason for each bar.
`execution_target_position` applies at `bar_open_time_utc`: its features already
end at the preceding bar, so downstream portfolio code must not shift or delay
the BTC target again. Negative values are valid short targets under the current
configuration; they are never silently converted to zero.

The standalone baseline therefore models BTC as short-capable exposure rather
than spot-only holdings. It caps the strategy at 1.0x gross notional leverage,
charges the configured 10 bps commission on each order, and currently assumes
zero borrowing and funding costs. The team accepted this short-capable treatment
for the combined project simulation. Margin requirements, liquidation, borrowing
costs, and exchange funding payments remain unmodeled limitations that must be
disclosed with the results. Changing to spot-only exposure would require BTC's
own pre-holdout selection to be rerun with shorting disabled.

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
ALPACA_DATA_FEED=iex
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
python run_two_week_strategy_optimization.py --help
python run_fx_ingestion.py --help
python run_portfolio_weights.py --help
python run_portfolio_simulation.py --help
python run_daily.py --help
python report_metrics.py --help
```

### Ingestion options

| Argument | Meaning |
|---|---|
| `--provider NAME` | Select `alpaca`, `binance`, `yahoo`, or `moomoo`; canonical provider names also work |
| `--symbol SYMBOL` | Select `SPY`, `BTCUSDT`, `ES3.SI`, or `SG.ES3` |
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
| Repeated `--input-dir PATH` | Replace the three default selected-source directories |
| `--output-file PATH` | Canonical CSV; default `data/canonical/market_bars_30m.csv` |
| `--summary-file PATH` | JSON summary; default `data/canonical/consolidation_summary.json` |
| `--allow-partial` | Permit consolidation when fewer than all three instruments are supplied |

### Orchestrator option

| Argument | Meaning |
|---|---|
| `--config PATH` | Workflow JSON; default `config/market_data_pipeline.json` |

### Training, validation, and live data separation

`data/canonical/market_bars_30m_training_validation.csv` is the frozen
pre-holdout dataset used for strategy training and validation. Its ES3.SI rows
come from
`data/staging/combined_history/yahoo_finance/ES3_SI/market_bars_30m_with_backfill.csv`,
The combined STI history merges the earlier and later Yahoo batches, removes
duplicate overlaps, and covers 6 July 2026 through 30 September 2026. The exact
inputs and row ranges are recorded in
`data/canonical/consolidation_summary_training_validation.json`.

Regenerate this exact three-asset canonical layout from the supplied staging
directories with:

```bash
python run_consolidation.py \
  --input-dir data/staging/alpaca_market_data/SPY \
  --input-dir data/staging/backtest_cutoff/alpaca_market_data/SPY \
  --input-dir data/staging/binance_spot/BTCUSDT \
  --input-dir data/staging/combined_history/yahoo_finance/ES3_SI \
  --output-file data/canonical/market_bars_30m_training_validation.csv \
  --summary-file data/canonical/consolidation_summary_training_validation.json
```

Evaluation-period data, pipeline settings, consolidated files and reports are
kept in ignored local paths and supplied separately from the repository. They
must not overwrite the frozen training/validation file or be used to retrain or
reselect the frozen strategy parameters.

## Test provider connections

List the available combinations, then use short targeted runs. These commands
also test standardization, validation, and CSV writing:

```bash
python run_ingestion.py --list-instruments
python run_ingestion.py --provider alpaca --symbol SPY --lookback-days 7
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

An explicit start with no end uses the latest completed delayed bar:

```json
{
  "provider": "alpaca",
  "symbol": "SPY",
  "start": "2026-09-01T00:00:00Z"
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
python run_ingestion.py --provider binance --symbol BTCUSDT \
  --start "$DATA_START" --end "$DATA_END"
python run_ingestion.py --provider yahoo --symbol ES3.SI \
  --start "$DATA_START" --end "$DATA_END"
```

Aggregate each staging directory independently:

```bash
python run_aggregation.py --input-dir data/staging/alpaca_market_data/SPY \
  --output-dir data/staging/alpaca_market_data/SPY
python run_aggregation.py --input-dir data/staging/binance_spot/BTCUSDT \
  --output-dir data/staging/binance_spot/BTCUSDT
python run_aggregation.py --input-dir data/staging/yahoo_finance/ES3_SI \
  --output-dir data/staging/yahoo_finance/ES3_SI
```

Apply backfill independently so no provider can fill another provider's bars:

```bash
python run_backfill.py --input-dir data/staging/alpaca_market_data/SPY \
  --output-dir data/staging/alpaca_market_data/SPY
python run_backfill.py --input-dir data/staging/binance_spot/BTCUSDT \
  --output-dir data/staging/binance_spot/BTCUSDT
python run_backfill.py --input-dir data/staging/yahoo_finance/ES3_SI \
  --output-dir data/staging/yahoo_finance/ES3_SI
```

Finally, consolidate the three selected sources:

```bash
python run_consolidation.py
```

The default consolidation selects Alpaca SPY, Binance BTCUSDT, and Yahoo
ES3.SI. To replace ES3 with another provider later, pass all selected
directories explicitly:

```bash
python run_consolidation.py \
  --input-dir data/staging/alpaca_market_data/SPY \
  --input-dir data/staging/binance_spot/BTCUSDT \
  --input-dir path/to/the/replacement/es3/directory
```

Consolidation rejects duplicate bar keys and rejects more than one provider for
the same `instrument_id`. It unions the rows in long format and does not create
artificial observations while SPY or ES3 is closed.

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
the three assets do not all share the same trading session. The summary records
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

The same schema does not make volume directly comparable. SPY volume is shares,
SG.ES3 volume is ETF units, and BTCUSDT volume is bitcoin.

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

The session check uses the `XNYS` US-equity calendar for SPY, `XSES`
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

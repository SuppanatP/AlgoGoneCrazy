# FE5209 Group Project - Market Data Pipeline

The tracked experiment history, retained configurations, final SPY/STI
selection, and final-period data policy are summarized in `EXPERIMENTS.md`.

This first project phase extracts and cleans completed five-minute source bars
for:

- `SPY` from Alpaca Market Data
- `SG.ES3` from Moomoo OpenAPI when entitled, or recent `ES3.SI` from Yahoo
- `BTCUSDT` from Binance Spot

It also provides separate steps that derive 30-minute OHLCV bars and apply a
small set of conservative backfill methods. A separate strategy package
generates momentum and mean-reversion signals, and a historical backtesting
entry point connects those signals to the shared trade tracker. A pre-test
parameter-comparison workflow and a training-only parameter optimiser are
also available. A strategy-agnostic sizing layer can produce capped
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

Development-period canonical bars
      -> configured SPY momentum and STI mean-reversion parameter grids
      -> chronological fold backtests with the same execution and cost model
      -> reliability filters and robustness ranking
      -> locally saved selected parameters

Canonical adjusted closes
      -> one daily close per local market session
      -> trailing annualized volatility known before each rebalance date
      -> inverse volatility multiplied by available-history confidence
      -> allocate a 95% strategic budget subject to one common 75% cap
      -> retain every inactive signal's allocation as cash

Selected SPY/STI strategies + external BTC signal interface
      -> delay decisions to each asset's next eligible raw open
      -> apply weekly risk budgets and point-in-time USD conversion
      -> execute through one USD cash and position ledger
      -> decisions, fills, trades, equity, costs, and per-asset P&L
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
  search_space.py        Valid, deterministic strategy parameter grids
  evaluation.py          Chronological fold evaluation through the backtester
  selection.py           Reliability filters, robust scoring, and ranking
  two_week.py            Cost-aware two-week screening and joint diagnostics
src/portfolio_sizing/
  inverse_volatility.py  Daily volatility, equal/inverse-vol capped schedules
  fx.py                  Point-in-time conversion into the USD portfolio base
  fred_csv.py            Downloaded FRED SGD/USD CSV standardization
  targets.py             Signal gating and base-currency target quantities
src/portfolio_simulation.py Shared signals, USD execution, and portfolio replay
src/strategy_backtest.py Strategy-to-backtester adapter and output writer
src/backtest.py          Reused bar-by-bar backtest engine
src/tracker.py           Reused cash, position, fill, trade, and cost ledger
src/metrics.py           Reused performance and transaction-cost metrics
config/
  market_data_pipeline.json  Selected assets, providers, ranges, and paths
  october_live_data_pipeline.json  Live extraction plus history/live consolidation
  strategy_optimization.json Search spaces, training folds, and constraints
  two_week_strategy_optimization.json Short-horizon frequencies and compact grids
  portfolio_sizing.json      Frozen history-confidence sizing and quantity steps
  selected_strategies.json   Frozen SPY/STI rules and external BTC interface
test/                    Offline tests using synthetic data
run_ingestion.py         Convenient entry point
run_aggregation.py       Separate 30m aggregation entry point
run_backfill.py          Separate non-destructive backfill entry point
run_consolidation.py     Canonical four-asset CSV entry point
run_market_data_pipeline.py  One-command sequential workflow entry point
run_strategy_backtest.py SPY/ES3 momentum and mean-reversion backtest entry point
run_parameter_experiments.py Pre-test parameter comparison entry point
run_strategy_optimizer.py Development-only strategy optimisation entry point
run_two_week_strategy_optimization.py Two-week SPY/STI review entry point
run_fx_ingestion.py       Downloaded FRED SGD/USD standardization entry point
run_portfolio_weights.py  Rolling inverse-volatility/equal-weight entry point
run_portfolio_simulation.py Shared-USD portfolio replay entry point
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
training and validation periods. October and later data remain excluded for
the official test period.

```bash
python run_parameter_experiments.py
```

The default outputs are `results/parameter_experiments/parameter_comparison.csv`
and `experiment_manifest.json`. The manifest records the cutoff, periods, costs,
and every parameter set so the comparison can be reproduced.

### Strategy parameter optimisation

The optimiser searches momentum parameters for SPY and mean-reversion
parameters for ES3.SI. Every candidate is evaluated on chronological training
folds using the existing next-bar execution, portfolio ledger, and project cost
model. Validation observations never affect candidate ranking. BTC strategy
development is outside this optimiser and connects through the external signal
interface described below.

All boundaries are UTC and use an inclusive start and exclusive end. The
official test period starts at `2026-10-01T00:00:00Z`, so no training or
validation run includes an October 2026 observation.

| Instrument | Training folds used for selection | Separate validation period |
|---|---|---|
| SPY | 18 Sep 2023–1 Apr 2024; 1 Apr–1 Oct 2024; 1 Oct 2024–1 Apr 2025; 1 Apr–1 Oct 2025 | 1 Oct 2025–1 Oct 2026 |
| ES3.SI | 6–21 Jul 2026; 21 Jul–4 Aug 2026; 4–18 Aug 2026; 18 Aug–1 Sep 2026 | 1 Sep–1 Oct 2026 |

The search is a deliberately small discrete grid rather than a continuous or
every-integer search. Values represent materially different horizons while
keeping the results understandable and limiting overfitting.

| Strategy | Values searched | Valid candidates |
|---|---|---:|
| SPY momentum | Fast EMA: 8, 13, 26 bars; slow EMA: 26, 52, 104 bars; return lookback: 13, 26, 52 bars; entry return: 0%, 0.25%, 0.50%; exit return: -0.25%, 0% | 144 |
| ES3.SI mean reversion | Rolling window: 20, 26, 52 bars; entry z-score: -1.25, -1.50, -2.00; exit z-score: -0.25, 0; maximum holding period: 26, 52, 104 bars | 54 |

This produces 792 fold backtests: `144 x 4` for SPY and `54 x 4` for ES3.SI.
On the current project machine, the complete training search took about two
minutes. Allow roughly two to five minutes depending on the computer and other
workload. `--validate-selected` repeats the training search before adding the
two validation runs, so it takes approximately the same amount of time.

Run the full training search:

```bash
python run_strategy_optimizer.py
```

Run a small connection and workflow check without completing the full grid:

```bash
python run_strategy_optimizer.py --max-candidates 2
```

After the training search has selected candidates, validation can be run
explicitly. It is reported separately and is not fed back into the ranking:

```bash
python run_strategy_optimizer.py --validate-selected
```

Generated files are written beneath `results/strategy_optimizer/` and are
ignored by Git:

- `fold_results.csv`: one row per candidate and training fold
- `candidate_ranking.csv`: reliability checks, aggregate metrics, and ranking
- `selected_parameters.json`: the training-only winners
- `validation_summary.csv`: optional untouched validation results
- `optimization_manifest.json`: data, configuration, costs, and run settings

If the group approves the winners, copy only the frozen parameter values into a
tracked strategy configuration. Do not commit generated optimisation outputs or
use official test-period observations to revise the parameters.

### Two-week strategy review

The short-horizon workflow complements the original fold optimiser. It compares
30-, 60-, and 120-minute decision frequencies while continuing to execute at
the next underlying 30-minute raw open. Lookbacks are expressed as trading-day
horizons and converted to the appropriate number of decision bars, so changing
frequency does not silently change the intended economic horizon.

Every per-asset screen uses the final USD mechanics: adjusted closes for
signals, raw prices for execution and valuation, historical SGD/USD rates,
whole shares, the common concentration cap, and broker commissions. FX transaction
fees are outside the project scope; exchange rates are still required for USD
conversion and reporting. The inexpensive first pass runs every candidate once
and derives rolling 14-session diagnostics stepped every five sessions. This
matches the roughly 14 equity sessions in the 1-20 October live period. The
windows overlap and are therefore robustness checks rather than independent
statistical observations. Only the shortlist is then rerun with cash and
positions independently reset for every 14-session window.

The final diagnostic combines shortlisted SPY and STI candidates in the actual
shared USD portfolio under both weekly capped inverse-volatility and capped
equal weighting. A separate rolling comparison holds the top strategy pair
fixed, so differences are caused by sizing rather than different signals. BTC
has a zero signal while its external signal file is unavailable. Because STI
history begins in July and sizing requires 40 daily returns, only the available
full pre-October windows are included and the output records the actual count.
The latest 20-calendar-day
pre-October period, 11-30 September 2026, is labelled as a known diagnostic
because earlier September results have already influenced the redesign; it is
not described as untouched validation. No October 2026 data is used, and the
command never overwrites `config/selected_strategies.json`.

Run the full compact search and diagnostic:

```bash
python run_two_week_strategy_optimization.py
```

Run only a limited workflow check, or omit both joint sizing comparisons:

```bash
python run_two_week_strategy_optimization.py --max-candidates 2
python run_two_week_strategy_optimization.py --skip-joint-diagnostic
```

Run the expanded pre-October SPY/STI search without changing the original
optimisation outputs or the frozen selected strategy:

```bash
python run_two_week_strategy_optimization.py \
  --config-file config/sti_mean_reversion_expanded_optimization.json \
  --output-dir results/sti_mean_reversion_expanded_optimization \
  --progress-every 10
```

This run screens the full SPY momentum grid and the expanded 120-minute STI
mean-reversion grid on their configured training windows. It does not reject a
candidate merely because some historical 14-session windows contain no order;
trade-window coverage remains visible as a diagnostic. Shortlisted SPY/STI
pairs are then compared jointly on the September pre-October validation period.
The run never reads October observations and never updates a selected-strategy
configuration automatically.

Run the same expanded grid under the separate confidence/performance-adjusted
sizing experiment:

```bash
python run_two_week_strategy_optimization.py \
  --input-file data/canonical/market_bars_30m_training_validation.csv \
  --config-file config/sti_mean_reversion_expanded_optimization.json \
  --selected-strategies config/selected_parameters_expanded_sti.json \
  --sizing-config config/portfolio_sizing_performance_adjusted.json \
  --output-dir results/sti_mean_reversion_expanded_performance_adjusted \
  --progress-every 10
```

This writes to a new result directory and does not replace either the original
expanded-grid results or the frozen official configuration. The current
pre-October joint diagnostic selected STI candidate `0026` from the expanded
grid: a 12-bar lookback, entry Z-score `-1.25`, exit Z-score `-0.25`, four-bar
minimum hold and cooldown, and 20-bar maximum hold. That manually recorded
experimental selection is stored in
`config/selected_parameters_expanded_sti_performance_adjusted.json`.

Replay the separate selection on the available October data without requiring
the still-pending BTC signal file:

```bash
python run_portfolio_simulation.py \
  --input-file data/canonical/market_bars_30m_simulation_live_to_date.csv \
  --strategy-config config/selected_parameters_expanded_sti_performance_adjusted.json \
  --sizing-config config/portfolio_sizing_performance_adjusted.json \
  --start 2026-10-01 \
  --end 2026-10-13 \
  --allow-missing-external \
  --output-dir results/portfolio_simulation_october_live_to_date_expanded_sti_performance_adjusted
```

The result directory is separate and ignored by Git. This is a
post-October-observation experiment, not untouched validation and not an
automatic replacement for `config/selected_strategies.json`. BTC remains an
unchanged external-signal interface and receives no orders in this temporary
run.

To test lower STI participation without imposing an STI-specific fixed cap,
run the same expanded grid with inactive strategy budgets retained as cash:

```bash
python run_two_week_strategy_optimization.py \
  --input-file data/canonical/market_bars_30m_training_validation.csv \
  --config-file config/sti_mean_reversion_expanded_optimization.json \
  --selected-strategies config/selected_parameters_expanded_sti.json \
  --sizing-config config/portfolio_sizing_performance_adjusted_cash_retention.json \
  --output-dir results/sti_mean_reversion_expanded_performance_adjusted_cash_retention \
  --progress-every 10
```

The best known-September joint diagnostic retained expanded STI candidate
`0052`; no newly tested candidate improved the joint result. Its separately
labelled replay configuration is
`config/selected_parameters_expanded_sti_cash_retention.json`. Replay it on the
available October snapshot with:

```bash
python run_portfolio_simulation.py \
  --input-file data/canonical/market_bars_30m_simulation_live_to_date.csv \
  --strategy-config config/selected_parameters_expanded_sti_cash_retention.json \
  --sizing-config config/portfolio_sizing_performance_adjusted_cash_retention.json \
  --start 2026-10-01 \
  --end 2026-10-13 \
  --allow-missing-external \
  --output-dir results/portfolio_simulation_october_live_to_date_expanded_sti_cash_retention
```

This policy multiplies each strategic risk weight by its signal and leaves the
unused portion in cash. It does not renormalize SPY, STI, or BTC to consume the
unused allocation. The output remains a separate post-observation diagnostic,
not untouched validation.

The strategy-performance sizing experiment replaces the underlying asset's
recent Sharpe multiplier with the configured strategy's own trailing,
after-commission daily-return Sharpe. Run the expanded diagnostic with:

```bash
python run_two_week_strategy_optimization.py \
  --input-file data/canonical/market_bars_30m_training_validation.csv \
  --config-file config/sti_mean_reversion_expanded_optimization.json \
  --selected-strategies config/selected_parameters_expanded_sti.json \
  --sizing-config config/portfolio_sizing_strategy_performance_cash_retention.json \
  --output-dir results/sti_mean_reversion_expanded_strategy_performance_cash_retention \
  --progress-every 10
```

Each candidate's weekly score is its inverse volatility multiplied by its
sample-history confidence and a clipped strategy-performance multiplier. The
multiplier is `clip(1 + strategy Sharpe / 2, 0.5, 1.5)`, calculated from the
last 60 available strategy-session returns strictly before the rebalance. The
return history uses executable positions, adjusted-price returns, the project
broker commissions, and a USD 100,000 reference sleeve for minimum/per-share
fee calculation. The resulting scores are allocated within the 95% strategic
budget and common 75% cap; inactive signal allocations subsequently remain in
cash.

The best known-September diagnostic again retained expanded STI candidate
`0052`. Its separate replay configuration is
`config/selected_parameters_expanded_sti_strategy_performance.json`. Run the
October snapshot with:

```bash
python run_portfolio_simulation.py \
  --input-file data/canonical/market_bars_30m_simulation_live_to_date.csv \
  --strategy-config config/selected_parameters_expanded_sti_strategy_performance.json \
  --sizing-config config/portfolio_sizing_strategy_performance_cash_retention.json \
  --start 2026-10-01 \
  --end 2026-10-13 \
  --allow-missing-external \
  --output-dir results/portfolio_simulation_october_live_to_date_strategy_performance_cash_retention
```

The simulation additionally writes `strategy_return_history.csv` for auditing
the values available to each weekly sizing decision. This remains a
post-October-observation comparison and does not replace the official
configuration automatically.

The simpler history-confidence experiment deliberately removes every return or
Sharpe multiplier. It uses only inverse volatility and the amount of history
available before each rebalance:

```text
allocation score = min(1, prior daily observations / 252) / volatility
```

It also keeps inactive allocations as cash and locks each position's quantity
from entry until the strategy exits. Weekly weight changes therefore affect the
next entry but do not generate resizing orders for an already-open position.
Run the no-trade-filter training and validation experiment with:

```bash
python run_two_week_strategy_optimization.py \
  --input-file data/canonical/market_bars_30m_training_validation.csv \
  --config-file config/sti_mean_reversion_expanded_optimization.json \
  --selected-strategies config/strategy_search_baseline.json \
  --sizing-config config/portfolio_sizing_history_confidence_cash_retention.json \
  --output-dir results/pre_october_joint_history_confidence_optimization \
  --progress-every 10
```

The highest after-cost joint validation result selects SPY momentum candidate
`0041` and expanded STI mean-reversion candidate `0052`. The reviewed and
frozen selection is stored in `config/selected_strategies.json`. The baseline
file is intentionally separate, so the selected pair is not supplied as an
input to its own search. The final evaluation-period simulation is reported
separately after that period is complete.

The ignored directory `results/two_week_strategy_optimization/` contains:

- `candidate_screening.csv`: full-period and inexpensive two-week candidate metrics
- `two_week_continuous_windows.csv`: first-pass rolling 14-session diagnostics
- `two_week_independent_windows.csv`: exact reset-window results for the shortlist
- `shortlisted_candidate_review.csv`: shortlist comparison used for discussion
- `sizing_method_comparison_windows.csv`: same top strategy pair under both sizing methods in rolling pre-October windows
- `sizing_method_comparison_summary.csv`: concise inverse-volatility versus equal-weight comparison
- `joint_diagnostic_summary.csv`: known September shared-portfolio comparison
- `optimization_manifest.json`: data, periods, costs, sizing, and grid snapshot

The sizing comparison ranks methods for review but does not change the default
automatically. The final default in `config/portfolio_sizing.json` is
history-confidence inverse volatility with inactive allocations held as cash.

There is no minimum trade-window eligibility condition. Completed candidates
are ranked by median after-cost return, profitable-window rate,
worst-window return, and finally cost rate. Trade frequency is reported but is
not a ranking variable. This remains a shortlist aid rather than an automatic
strategy-selection rule.

### Portfolio sizing

Portfolio sizing is separate from signal generation. Momentum, mean reversion,
or another teammate's strategy supplies a standardized target signal: `1` for
long and `0` for cash. The final sizing score is each asset's inverse
volatility multiplied by `min(1, prior daily observations / 252)`. These scores
share a 95% strategic budget subject to one common 75% maximum. If an asset's
signal is inactive, its strategic allocation remains cash rather than being
transferred to another asset. The configured 5% is therefore only the minimum
cash reserve when every strategy is active; actual cash can be higher.

The tracked defaults use 60 daily return observations, permit a minimum of 40
while STI history remains short, rebalance weekly, reserve at least 5% cash,
apply a 5% annualized volatility floor, and apply the same 75% maximum to SPY,
ES3.SI, and BTCUSDT. Existing position quantities are not resized merely
because a weekly target weight changes; a new weight is applied on the next
strategy entry. TLT is not part of the current portfolio. Every weight uses
adjusted closing prices strictly before its rebalance date, so current or
future prices cannot leak into the sizing decision.

The separate experimental configuration
`config/portfolio_sizing_performance_adjusted.json` keeps the same
inverse-volatility foundation, then multiplies each asset's score by two
strictly backward-looking terms: a history-confidence factor and a clipped
60-session adjusted-price Sharpe factor. It uses a common 75% cap and ignores
target-weight changes smaller than five percentage points unless the signal
itself changes. These choices are intended to favor assets with more evidence
and stronger recent risk-adjusted performance. They remain an alternative
experiment rather than the final sizing method.

The additional experimental configuration
`config/portfolio_sizing_performance_adjusted_cash_retention.json` changes only
the inactive-allocation policy. With `hold_as_cash`, an inactive strategy's
budget stays in cash rather than being transferred to another active strategy.
For probabilistic signals between zero and one, the same rule scales the
strategic weight proportionally. Cash retention is also used by the final
history-confidence configuration.

`config/portfolio_sizing_strategy_performance_cash_retention.json` retains the
same cash policy but bases the performance multiplier on the strategy rather
than the underlying asset. This distinction matters: a rising ETF does not
receive a performance bonus unless its configured signal would also have
earned a positive trailing return. Candidate-specific returns are rebuilt in
the joint optimiser, so every candidate is evaluated using its own prior
signal history without future observations.

`config/portfolio_sizing_history_confidence_cash_retention.json` preserves a
named copy of the final rule for audit. The same rule is now the default in
`config/portfolio_sizing.json`. A full year of prior observations receives
confidence one; shorter histories are linearly shrunk. As of 1 October this
produced strategic weights of approximately 63.6% SPY, 14.8% STI, and 16.6%
BTC before signals were applied. It uses no return or Sharpe forecast and no
asset-specific preference.

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

After refreshing the canonical file through 30 September 2026, build the
initial official-period weights using information available before 1 October:

```bash
python run_portfolio_weights.py \
  --start 2026-10-01 \
  --end 2026-10-02 \
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
2026-09-30T00:00:00Z,SGD,USD,0.78,fred_download,DEXSIUS,2026-09-29,false
2026-10-01T00:00:00Z,SGD,USD,0.779,fred_download,DEXSIUS,2026-09-30,false
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
automatically. BTCUSDT is configured as an external signal source.

| Asset | Signal method | Decision interval | Frozen selection |
|---|---|---:|---|
| SPY | Momentum | 120 minutes | `spy_usd__momentum_0041` |
| ES3.SI | Mean reversion | 120 minutes | `sti_es3_sgd__mean_reversion_0052` |
| BTCUSDT | Teammate external signal | 30 minutes | Awaiting signal CSV |

BTC's CSV must contain `bar_open_time_utc` and `target_position`, where the
target is `1` for long or `0` for cash and is decided after that 30-minute bar
closes. Timestamps must match canonical BTC 30-minute bar opens. Sparse changes
are allowed and are carried forward until the next supplied decision. The
simulator delays each decision to the next eligible BTC bar open. SPY and STI
retain their latest 120-minute decisions between their own decision times, so
all three assets can share one portfolio despite their mixed frequencies.

Run the September portfolio evaluation while the BTC signal file is unavailable:

```bash
python run_portfolio_simulation.py \
  --start 2026-09-01 \
  --end 2026-10-01 \
  --allow-missing-external
```

In that temporary mode BTC receives no orders and its strategic allocation
remains cash. SPY and STI keep their own history-confidence weights rather than
absorbing BTC's allocation. For the complete run, save the teammate's signal file at
`data/signals/btc_signals.csv` and omit `--allow-missing-external`; a missing BTC
file then stops the run. The simulator uses one USD 500,000 ledger, historical
SGD/USD rates, USDT/USD at 1:1, whole units for SPY/ES3.SI, fractional BTC,
configured broker commissions, no separate FX transaction fee, and the next
eligible raw open. It refreshes risk weights weekly, keeps inactive allocations
as cash, and does not resize an existing position solely for a weight update.

Use equal weighting for a direct portfolio replay comparison:

```bash
python run_portfolio_simulation.py \
  --start 2026-09-01 \
  --end 2026-10-01 \
  --allow-missing-external \
  --sizing-method capped_equal_weight \
  --output-dir results/portfolio_simulation_equal_weight
```

Equal weighting is retained as a benchmark; history-confidence inverse
volatility with cash retention is the configured final method. Once the BTC
signal file and market data through 20 October are available, refresh the live
pipeline and run the frozen official-period simulation once:

```bash
python run_portfolio_simulation.py \
  --start 2026-10-01 \
  --end 2026-10-21
```

The end timestamp is exclusive, so `2026-10-21` includes observations through
20 October. Do not rerun the optimiser with October data or revise the frozen
parameters after inspecting this result.

Outputs beneath `results/portfolio_simulation/` include `decisions.csv`,
`fills.csv`, `trades.csv`, `equity_curve.csv`, `performance_summary.csv`,
`asset_summary.csv`, risk and FX diagnostics, and a reproducibility manifest.
Positions remain open at the final close to match a live-period snapshot. Raw
prices are used for execution and valuation; separate dividend cash flows are
not yet present in the source data and are recorded as excluded in the manifest.

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
python run_parameter_experiments.py --help
python run_strategy_optimizer.py --help
python run_two_week_strategy_optimization.py --help
python run_fx_ingestion.py --help
python run_portfolio_weights.py --help
python run_portfolio_simulation.py --help
```

### Ingestion options

| Argument | Meaning |
|---|---|
| `--provider NAME` | Select `alpaca`, `binance`, `yahoo`, or `moomoo`; canonical provider names also work |
| `--symbol SYMBOL` | Select `SPY`, `BTCUSDT`, `ES3.SI`, `SG.ES3`, or optional `TLT` |
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

Download, aggregate, backfill, and consolidate the evaluation-period data
without modifying the frozen training/validation files:

```bash
python run_market_data_pipeline.py \
  --config config/october_live_data_pipeline.json
```

This writes provider-specific files below `data/staging/october_live/` and the
live-only canonical file to
`data/canonical/market_bars_30m_october_live.csv`. It then combines those live
outputs with the configured historical staging directories and refreshes
`data/canonical/market_bars_30m_simulation_live_to_date.csv`, which is the
default input for portfolio sizing and simulation. Its start is fixed at
1 October 2026 and its omitted end means the latest completed delayed bar at
the time of each run. Rerunning the command does not overwrite
`market_bars_30m_training_validation.csv`.

### Training, validation, and live data separation

`data/canonical/market_bars_30m_training_validation.csv` is the frozen
pre-October dataset used for strategy training and validation. Its ES3.SI rows
come from
`data/staging/combined_history/yahoo_finance/ES3_SI/market_bars_30m_with_backfill.csv`,
not from the October-live staging directory. That combined STI history merges
the earlier and later Yahoo batches, removes duplicate overlaps, and covers
6 July 2026 through 30 September 2026. The exact inputs and row ranges are
recorded in
`data/canonical/consolidation_summary_training_validation.json`.

`data/canonical/market_bars_30m_october_live.csv` contains only evaluation-period
bars from 1 October onward. The separate
`data/canonical/market_bars_30m_simulation_live_to_date.csv` appends historical
bars before the October bars solely so the strategies and sizing engine have
enough warm-up observations. Portfolio simulation must still be run with
`--start 2026-10-01`, so no pre-October trade, position, cash flow, or P&L is
included in the evaluation result. October data must not be used to retrain or
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
  "start": "2026-10-01T00:00:00Z"
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

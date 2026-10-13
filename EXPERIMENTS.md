# Experiment record

This file is the compact, tracked record of the strategy and portfolio
experiments. Generated CSV, JSON, and chart directories under `results/` are
working outputs and are not committed. The configuration files named below,
the final selected strategy, and this record are the reproducibility artifacts
that should be committed and pushed.

All timestamps and period boundaries are UTC. Period ends are exclusive.

## Data boundaries

| Role | Period | File used locally | Treatment |
|---|---|---|---|
| SPY training | 18 Sep 2023 to 1 Oct 2025 | `data/canonical/market_bars_30m_training_validation.csv` | Candidate screening only |
| STI training | 6 Jul 2026 to 1 Sep 2026 | `data/canonical/market_bars_30m_training_validation.csv` | Candidate screening only |
| Joint validation | 11 Sep 2026 to 1 Oct 2026 | `data/canonical/market_bars_30m_training_validation.csv` | Final SPY/STI pair selection |
| Official evaluation | 1 Oct 2026 to 21 Oct 2026 | Final October canonical dataset | Never used to tune or select parameters |

The training/validation file's STI rows come from the merged Yahoo history in
`data/staging/combined_history/yahoo_finance/ES3_SI/`. It covers 6 July through
30 September 2026. The October simulation input contains earlier bars only as
indicator and sizing warm-up; simulation P&L begins on 1 October.

The market-data file is submitted separately rather than committed. A rerun is
the same experiment only when the supplied files have these checksums:

| File | Data rows | SHA-256 |
|---|---:|---|
| `market_bars_30m_training_validation.csv` | 73,746 | `9fbd584b75467353be424cb76dfd8904653cb675751248386c3504b0519227e7` |
| `fx_rates_to_usd.csv` | 11,480 | `51ec9fde366f48e472d369448ef2d95e606c7f5832e91981ce88e59f1c7aa046` |

## Experiments and configurations

| Experiment | Purpose | Configuration retained |
|---|---|---|
| Initial chronological fold search | Establish baseline momentum and mean-reversion parameters | `config/strategy_optimization.json` |
| Short-window robustness search | Compare 30-, 60-, and 120-minute decisions over rolling 14-session windows | `config/two_week_strategy_optimization.json` |
| Expanded STI search | Search a wider 120-minute STI lookback and z-score grid alongside the full SPY grid | `config/sti_mean_reversion_expanded_optimization.json` |
| Confidence plus price-performance sizing | Test inverse volatility adjusted by data-history confidence and underlying-price Sharpe | `config/portfolio_sizing_performance_adjusted.json` |
| Performance sizing with cash retention | Test the same performance adjustment without redistributing inactive allocations | `config/portfolio_sizing_performance_adjusted_cash_retention.json` |
| Strategy-performance sizing | Replace underlying-price Sharpe with trailing after-cost strategy Sharpe | `config/portfolio_sizing_strategy_performance_cash_retention.json` |
| History-confidence sizing | Remove return/Sharpe multipliers; use history confidence and inverse volatility, retaining inactive allocations as cash | `config/portfolio_sizing_history_confidence_cash_retention.json` |
| Final portfolio sizing | Frozen copy of the selected history-confidence rule | `config/portfolio_sizing.json` |

The corresponding experimental strategy snapshots are retained as
`config/selected_parameters_*.json`. They are audit records, not active
defaults. `config/selected_strategies.json` is the only active final strategy
selection.

The retained snapshot files are:

- `config/selected_parameters_expanded_sti.json`
- `config/selected_parameters_expanded_sti_performance_adjusted.json`
- `config/selected_parameters_expanded_sti_cash_retention.json`
- `config/selected_parameters_expanded_sti_strategy_performance.json`
- `config/selected_parameters_expanded_sti_history_confidence.json`
- `config/selected_parameters_pre_october_history_confidence.json`

Several snapshots contain the same parameter values because the experiment
changed sizing rather than signal generation. They remain separately named so
the input combination used by each recorded experiment is unambiguous.

## Experiment sequence for presentation

Present strategy selection and portfolio construction as related but separate
steps:

1. Generate the SPY momentum and STI mean-reversion parameter grids.
2. Screen each asset's candidates over its training windows using the same
   execution timing, raw-price fills, FX conversion, and commission model.
3. Rank robustness and shortlist candidates. Trade frequency remains a
   diagnostic, not a ranking objective.
4. Compare portfolio sizing designs: equal weighting, inverse volatility,
   performance-adjusted inverse volatility, and history-confidence inverse
   volatility. This is the portfolio-construction experiment, after signal
   screening and before the final joint comparison.
5. Freeze history-confidence inverse-volatility sizing because it is the
   simpler rule, discounts STI's short history, retains inactive allocations
   as cash, and avoids adding a return/Sharpe forecast multiplier.
6. Evaluate shortlisted SPY/STI pairs together on the validation period under
   that frozen sizing rule.
7. Select the highest after-cost joint validation pair and freeze both the
   strategy parameters and sizing configuration before final evaluation.

The sizing experiment should normally be one methods slide or an appendix
table. It is important because the final pair is selected from a shared
portfolio result, but the presentation does not need every exploratory sizing
configuration on the main strategy-selection slide.

## Final training and validation search

The final search evaluated 48 SPY momentum candidates and 168 STI
mean-reversion candidates. Candidate screening used rolling 14-session windows
and ranked after-cost results by median return, profitable-window rate,
worst-window return, and cost rate. Trade frequency was diagnostic only; there
was no minimum trade-window filter.

The shortlisted SPY and STI candidates were then combined in a shared USD
portfolio for the September validation period using the final history-confidence
inverse-volatility sizing assumptions. The selected pair produced the highest
after-cost joint validation return among the shortlisted pairs.

### Frozen strategy pair

| Asset | Candidate | Decision interval | Parameters |
|---|---|---:|---|
| SPY | `spy_usd__momentum_0041` | 120 minutes | Fast EMA 8 bars; slow EMA 16 bars; return lookback 8 bars; entry threshold 0.25%; exit threshold 0% |
| ES3.SI | `sti_es3_sgd__mean_reversion_0052` | 120 minutes | Window 16 bars; entry z-score -1.25; exit z-score 0.25; minimum hold 4 bars; cooldown 4 bars; maximum hold 20 bars |
| BTCUSDT | External teammate signal | 30 minutes | Probability/target position from 0 to 1; no BTC model was selected by this search |

The frozen pair is stored in `config/selected_strategies.json`.

### Exact reproduction command

Place the separately submitted data at the paths shown below, then run:

```bash
python run_two_week_strategy_optimization.py \
  --input-file data/canonical/market_bars_30m_training_validation.csv \
  --fx-file data/fx/fx_rates_to_usd.csv \
  --config-file config/sti_mean_reversion_expanded_optimization.json \
  --selected-strategies config/strategy_search_baseline.json \
  --sizing-config config/portfolio_sizing_history_confidence_cash_retention.json \
  --cost-config config/cost_assumptions.json \
  --output-dir results/final_training_validation_selection \
  --progress-every 10
```

`config/strategy_search_baseline.json` is deliberately different from the
frozen output selection, preventing the final pair from being supplied to its
own search. The first row of `joint_diagnostic_summary.csv` should identify
SPY candidate `0041` and STI candidate `0052`. The command does not overwrite
`config/selected_strategies.json`; that file is the reviewed, frozen record of
the winning row.

### Training and validation evidence

| Result | SPY 0041 | STI 0052 |
|---|---:|---:|
| Candidate-screening full-period net return | 21.76% | 2.76% |
| Independent 14-session windows | 100 | 6 |
| Median independent-window net return | 0.62% | 1.13% |
| Profitable independent-window rate | 64% | 100% |
| Worst independent-window net return | -4.81% | 0.42% |

The combined 11-30 September validation result was:

| Metric | Result |
|---|---:|
| Net return | 0.3167% |
| Net profit on USD 500,000 | USD 1,583.64 |
| Maximum drawdown | -1.0338% |
| Orders | 8 |
| Completed round trips | 4 |
| Transaction costs | USD 198.20 |
| Average cash weight | 69.22% |

STI has a much shorter history than SPY, so its six rolling windows are weak
evidence and must not be presented as equivalent to SPY's 100 windows.

## Frozen sizing rule

The selected sizing configuration is `config/portfolio_sizing.json`:

- Starting capital is USD 500,000 with no later cash injection.
- Each active asset's score is its history confidence divided by annualized
  volatility, with a 5% volatility floor.
- History confidence is `min(1, prior daily observations / 252)`.
- The strategic gross budget is 95%, with the same 75% maximum weight for every
  asset.
- Inactive signal allocations remain cash and are not reassigned.
- Existing positions are not resized solely because weekly risk weights change;
  the newest weight applies on the next entry.
- Portfolio valuation and reporting are in USD. USD and USDT are treated as
  one-to-one; SGD uses the recorded historical SGD/USD series.
- Signals use adjusted prices; simulated execution and valuation use raw market
  prices; broker commissions are included.

## Final evaluation and retention policy

At the end of 20 October:

1. Refresh the October pipeline through the last completed bar.
2. Run the frozen simulation from `2026-10-01` to the exclusive end
   `2026-10-21` without changing either frozen configuration.
3. Add the completed evaluation-period results to this file as a separate final
   simulation section; do not feed them back into strategy or sizing selection.
4. Retain and commit the final October canonical market-data CSV, its
   consolidation summary, the submitted daily-closing-price CSV, and the final
   compact performance summary/manifest.
5. Do not commit raw provider downloads, intermediate aggregation/backfill
   files, full result directories, charts, or temporary comparison outputs.

The final October canonical data is a deliberate exception to the normal data
ignore policy because Yahoo intraday data expires and the exact evaluation
period may not be downloadable later. The frozen pre-October training file and
full generated outputs should be archived outside Git with the group handoff.

Because `data/` and `results/` are ignored by default, add only the approved
final files explicitly with `git add -f`; do not unignore their parent
directories. This prevents intermediate downloads and rerun outputs from being
committed accidentally.

# Strategy Parameter Experiments

## Purpose

This experiment compares a deliberately small set of momentum and
mean-reversion parameters before the official test period. It is intended for
parameter selection, not as the final October-to-Week-11 performance report.

No market data from 1 October 2026 onward was used. The canonical input ends on
30 September 2026:

`data/canonical/market_bars_30m_backtest_2026-09-30.csv`

## Experimental design

| Instrument | Development period | Validation period | Available history warning |
|---|---|---|---|
| SPY | 18 Sep 2023 to 30 Sep 2025 | 1 Oct 2025 to 30 Sep 2026 | Approximately three years are available |
| ES3.SI | 21 Jul 2026 to 31 Aug 2026 | 1 Sep 2026 to 30 Sep 2026 | The Yahoo five-minute history is short, so results are provisional |

Each run starts with 100,000 units of the instrument's native currency. Signals
use adjusted closes, trades execute at the next eligible real bar's raw open,
and positions are valued using raw closes. Project broker commissions are
included, slippage is zero, positions use whole units, and remaining positions
are closed at the final real bar close. Dividends are not added separately to
the raw-price ledger.

## Parameter sets

| Parameter set | Definition |
|---|---|
| Momentum short | Fast EMA 8, slow EMA 26, return lookback 13 |
| Momentum baseline | Fast EMA 13, slow EMA 52, return lookback 26 |
| Momentum slow | Fast EMA 26, slow EMA 104, return lookback 52 |
| Mean reversion sensitive | Window 20, enter at z <= -1.25, exit at z >= 0, maximum 40 bars |
| Mean reversion baseline | Window 26, enter at z <= -1.5, exit at z >= 0, maximum 52 bars |
| Mean reversion conservative | Window 52, enter at z <= -2.0, exit at z >= -0.25, maximum 104 bars |

## SPY results

| Parameter set | Development return | Development Sharpe | Development max drawdown | Validation return | Validation Sharpe | Validation max drawdown | Validation trades | Validation costs (USD) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Buy and hold | 50.19% | 1.38 | -19.51% | 14.89% | 1.13 | -9.64% | 1 | 1.05 |
| Momentum short | 15.83% | 0.85 | -12.35% | 10.06% | 1.35 | -5.75% | 107 | 109.20 |
| Momentum baseline | 17.83% | 0.96 | -12.42% | 4.24% | 0.60 | -10.39% | 89 | 88.37 |
| Momentum slow | 24.34% | 1.35 | -12.04% | 8.98% | 1.23 | -7.28% | 55 | 56.42 |
| Mean reversion sensitive | 17.52% | 0.82 | -14.08% | 7.49% | 0.86 | -5.73% | 89 | 90.82 |
| Mean reversion baseline | 16.95% | 0.85 | -11.47% | 5.58% | 0.68 | -4.58% | 65 | 64.30 |
| Mean reversion conservative | 5.38% | 0.34 | -12.44% | 13.52% | 1.66 | -3.26% | 31 | 32.43 |

The slow momentum configuration is the most consistent momentum candidate. It
has the strongest development-period momentum result and remains competitive
on validation with materially lower volatility, drawdown, turnover, and costs
than short momentum. Conservative mean reversion performs very well on
validation but poorly during development, indicating stronger regime
sensitivity; it should not be selected solely because of its validation result.

## ES3.SI results

| Parameter set | Development return | Development Sharpe | Development max drawdown | Validation return | Validation Sharpe | Validation max drawdown | Validation trades | Validation costs (SGD) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Buy and hold | 3.60% | 2.50 | -3.17% | -1.20% | -1.29 | -3.51% | 1 | 159.03 |
| Momentum short | 1.08% | 1.27 | -1.95% | -2.10% | -4.21 | -2.81% | 12 | 1,894.93 |
| Momentum baseline | -0.04% | -0.01 | -3.30% | -1.33% | -2.55 | -1.49% | 9 | 1,425.85 |
| Momentum slow | -4.62% | -4.12 | -4.94% | -1.70% | -2.67 | -2.79% | 7 | 1,113.06 |
| Mean reversion sensitive | 1.57% | 2.36 | -0.85% | -1.81% | -3.24 | -1.99% | 9 | 1,426.91 |
| Mean reversion baseline | 1.81% | 3.09 | -0.82% | -1.95% | -3.30 | -2.28% | 5 | 792.08 |
| Mean reversion conservative | 2.00% | 5.04 | -0.35% | -0.27% | -0.57 | -1.03% | 2 | 319.25 |

Conservative mean reversion is the strongest ES3.SI candidate. It trades less,
incurs much lower costs, and has the smallest drawdown in both periods. It also
loses less than buy and hold during validation. All ES3.SI conclusions remain
provisional because both the development and validation samples are short and
annualized ratios are unstable over such limited periods.

## Provisional choice for the official test period

| Instrument | Provisional strategy | Reason |
|---|---|---|
| SPY | Momentum slow | Most consistent momentum performance across both periods, with lower turnover than the other momentum configurations |
| ES3.SI | Mean reversion conservative | Lowest turnover and transaction-cost drag, smallest drawdowns, and best validation result among the active ES3.SI strategies |

These parameters should be frozen before October test observations are used.
October-to-Week-11 performance must be reported separately and should not be
used to revise the parameters retrospectively.

## Reproduction

Run from the repository root using the project Python environment:

```bash
/Users/tsai/Desktop/Kai/py/base-venv/bin/python3 run_parameter_experiments.py
```

Detailed machine-readable results are written to:

- `results/parameter_experiments/parameter_comparison.csv`
- `results/parameter_experiments/experiment_manifest.json`

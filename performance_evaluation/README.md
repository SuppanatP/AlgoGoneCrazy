# Performance evaluation package

> **This folder is separate from the trading pipeline.** Nothing in `src/`,
> `config/`, `test/` or the `run_*.py` / `export_*.py` scripts at the
> repository root reads from or writes to `performance_evaluation/`. It holds
> the analysis behind the report and presentation, together with the frozen
> files that analysis uses. Unlike the rest of the repository, its inputs and
> generated outputs (CSV, JSON, XLSX, PNG, HTML) are committed on purpose, so
> the results can be read and rerun without rebuilding the data pipeline.
>
> Market data is **not** included here. It is submitted separately (see
> [Data](#data-submitted-separately)).

## Contents

```text
performance_evaluation/
├── README.md                                This file
├── portfolio_performance_evaluation.ipynb   Out-of-sample evaluation, 1–20 Oct 2026
├── results/
│   ├── portfolio_simulation/                Frozen simulation outputs (notebook inputs)
│   ├── btc_target_exports/btc_targets.csv   Frozen BTC strategy targets (notebook input)
│   ├── eda_notebook/                        Charts, tables and dashboard written by the notebook
│   └── AlgoGoneCrazy_Performance_Evaluation_Report.docx   Written report
└── visualization/                           Pre-holdout exploratory analysis (EDA)
    ├── eda_spy_si_2026/
    │   ├── eda_spy_si_2026.py               SPY momentum vs STI mean-reversion EDA
    │   ├── slide_charts.py                  Presentation charts built on the EDA
    │   ├── output/                          Charts and stats written by eda_spy_si_2026.py
    │   └── slide_charts/                    Charts written by slide_charts.py
    └── eda_output/                          Earlier EDA run, kept for reference
```

The two parts answer different questions over different periods:

| | `visualization/` (EDA) | `portfolio_performance_evaluation.ipynb` |
|---|---|---|
| Question | Does SPY trend and ES3.SI mean-revert, supporting the chosen rules? | How did the three-asset portfolio perform? |
| Period | 1 Jan – 30 Sep 2026 (before the holdout) | 1 – 20 Oct 2026 (out-of-sample) |
| Assets | SPY, STI (ES3.SI) | SPY, STI, BTC |

## How this folder relates to the rest of the repository

The notebook only **reads** from the repository root. It never modifies it.

| Used from the repository root | Purpose |
|---|---|
| `src/metrics.py` | Daily returns, risk metrics and drawdowns |
| `src/risk_free.py` | Treasury risk-free-rate loader |
| `src/trading_strategy/` | Rebuilds the SPY and STI signals for the charts in section 4 |
| `config/selected_strategies.json` | Frozen strategy parameters and decision intervals |

The frozen files in `results/` were produced by the pipeline and then copied
here:

- `results/portfolio_simulation/` is the output of `run_portfolio_simulation.py`
  for 2026-10-01 to 2026-10-21 (end exclusive). Its `simulation_manifest.json`
  records the inputs and settings. Rerunning the simulation with the repository
  code at commit `39a8fd6` reproduces the equity curve, decisions, fills,
  trades and risk weights exactly. The notebook reads eight of its files;
  `asset_summary.csv`, `currency_conversion.csv`, `execution_signals.csv` and
  `strategy_return_history.csv` are kept as part of the complete simulation
  output.
- `results/btc_target_exports/btc_targets.csv` is the output of
  `export_btc_targets.py` (originally written to `records/btc_target_exports/`).

## Data (submitted separately)

Place the separately submitted `Data` folder **next to** the repository folder,
and the Treasury yield file inside the repository's `data/` folder:

```text
<parent folder>/
├── Data/
│   ├── asset_daily_closing_prices.xlsx            notebook
│   ├── market_bars_30m_simulation.xlsx            notebook
│   └── market_bars_30m_training_validation.xlsx   EDA scripts
└── AlgoGoneCrazy/                                 this repository
    ├── data/risk_free/raw/us_treasury_yield_curve_2026.csv   notebook
    └── performance_evaluation/
```

If `data/canonical/market_bars_30m_simulation_live_to_date.csv` exists in the
repository, the notebook reads 30-minute bars from it instead of
`market_bars_30m_simulation.xlsx`. The EDA scripts search upward from their
own folder for `Data/market_bars_30m_training_validation.xlsx`, so they find it
in the layout above.

## Setup

Complete the [Setup](../README.md#setup) steps in the main README, then install
the additional packages used here:

```bash
python -m pip install matplotlib plotly openpyxl ipykernel scipy statsmodels
```

API credentials are not needed for anything in this folder.

## Running the notebook

Open `performance_evaluation/portfolio_performance_evaluation.ipynb` in VS Code
or Jupyter, select the `.venv` kernel, and run all cells from top to bottom.
The notebook works whether the working directory is this folder (the VS Code
and Jupyter default) or the repository root.

The data-loading cell (section 2.1) prints a summary. With the submitted files
it should read:

```text
Evaluation window : 01 Oct 2026 00:00 to 20 Oct 2026 23:30 UTC (960 bars)
Equity            : $500,000 -> $507,084 (+1.42%)
Activity          : 57 orders, 24 completed round trips
Risk-free rate    : 4.23% average (U.S. Treasury 3-month)
```

Charts are saved as PNGs to `results/eda_notebook/` when `SAVE_FIGURES = True`
(the default). That folder also receives `performance_metric_summary.csv`,
`per_asset_scorecard.csv`, `metrics_comparison.csv`,
`key_performance_metrics.csv` and the interactive
`portfolio_interactive_dashboard.html`. Rerunning overwrites these files.

## Running the EDA scripts

From `performance_evaluation/visualization/eda_spy_si_2026/`:

```bash
python eda_spy_si_2026.py    # writes output/
python slide_charts.py       # runs the EDA first, then writes slide_charts/
```

Both can also be run cell by cell in VS Code (`# %%` cells). `eda_output/`
holds charts from an earlier version of the EDA. No current script
regenerates it.

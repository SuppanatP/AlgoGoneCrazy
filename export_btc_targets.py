"""Export the existing stateful BTC strategy's targets without trading a portfolio.

The exported execution_target_position at timestamp t applies at bar t's open.
It already uses features lagged through t-1 and must not receive another bar delay.
"""
from __future__ import annotations

import argparse
from datetime import timedelta
from pathlib import Path

import pandas as pd

import config
from run_daily import strategy_class, strategy_params
from src.btc_targets import generate_target_row, load_model_pair, prediction_frame
from src.data_source import fetch_bars, history_problem


def export_targets(
    start: str,
    end: str,
    output: str | Path,
    cfg=config,
    input_file: str | Path | None = None,
) -> pd.DataFrame:
    """Export targets for inclusive UTC dates ``start`` through ``end``."""
    if list(cfg.ASSETS) != ["BTC"]:
        raise ValueError("BTC target export requires config.ASSETS to contain only BTC")

    asset = "BTC"
    spec = cfg.ASSETS[asset]
    models = load_model_pair(spec["model_up"], spec["model_down"])
    strategy = strategy_class(cfg)(**strategy_params(cfg, asset))
    start_day = pd.Timestamp(start, tz="UTC")
    one_day = timedelta(days=1)
    end_exclusive = pd.Timestamp(end, tz="UTC") + one_day
    rows: list[dict] = []

    for day_start in pd.date_range(start_day, end_exclusive, freq="D", inclusive="left"):
        day_end = day_start + one_day
        history = fetch_bars(
            day_start - timedelta(days=cfg.LOOKBACK_DAYS),
            day_end,
            input_file if input_file is not None else spec["csv"],
            cfg.TIME_COL,
            spec.get("symbol"),
            getattr(cfg, "SYMBOL_COL", "symbol"),
        )
        bars = history[history.index >= day_start]
        if bars.empty:
            continue
        problem = history_problem(history, bars.index[0], cfg.WARMUP_DAYS, cfg.BAR_FREQ)
        if problem:
            raise SystemExit(
                f"{asset}: not enough history to export from {bars.index[0]} "
                f"(need {cfg.WARMUP_DAYS} days without gaps): {problem}"
            )
        predictions = prediction_frame(history, bars, asset, models, {asset: history})
        for timestamp, row in predictions.iterrows():
            rows.append(
                {
                    "bar_open_time_utc": timestamp,
                    "instrument_id": spec["symbol"],
                    **generate_target_row(strategy, row, timestamp),
                }
            )

    result = pd.DataFrame(rows)
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(output, index=False)
    print(f"Exported {len(result):,} BTC targets -> {output}")
    print("Timing: execution_target_position applies at the listed bar open; do not delay it again.")
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--start", required=True, help="first UTC date, inclusive")
    parser.add_argument("--end", required=True, help="last UTC date, inclusive")
    parser.add_argument(
        "--input-file",
        type=Path,
        help=(
            "canonical long-format file; defaults to the canonical files in config.py"
        ),
    )
    parser.add_argument(
        "--output",
        default="records/btc_target_exports/btc_targets.csv",
        help="ignored output CSV path",
    )
    args = parser.parse_args(argv)
    export_targets(args.start, args.end, args.output, input_file=args.input_file)


if __name__ == "__main__":
    main()

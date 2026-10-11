"""Build a rolling capped inverse-volatility weight schedule."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from src.portfolio_sizing import (  # noqa: E402
    AssetRiskSettings,
    InverseVolatilitySettings,
    build_weight_schedule,
    convert_daily_prices_to_base,
    daily_adjusted_closes,
    load_fx_rates,
    rebalance_dates,
)


INSTRUMENT_ALIASES = {
    "spy": "spy_usd",
    "spy_usd": "spy_usd",
    "tlt": "tlt_usd",
    "tlt_usd": "tlt_usd",
    "es3": "sti_es3_sgd",
    "es3.si": "sti_es3_sgd",
    "sg.es3": "sti_es3_sgd",
    "sti_es3_sgd": "sti_es3_sgd",
    "btc": "btc_usdt",
    "btcusdt": "btc_usdt",
    "btc_usdt": "btc_usdt",
}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input-file",
        type=Path,
        default=Path("data/canonical/market_bars_30m.csv"),
        help="Canonical long-format 30-minute market bars.",
    )
    parser.add_argument(
        "--config-file",
        type=Path,
        default=Path("config/portfolio_sizing.json"),
    )
    parser.add_argument(
        "--fx-file",
        type=Path,
        help=(
            "Long-format point-in-time FX rates. Required when a selected asset "
            "is not covered by a static rate in the sizing configuration."
        ),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("results/portfolio_sizing"),
        help="Ignored directory for weights, diagnostics, and the run manifest.",
    )
    parser.add_argument(
        "--instrument",
        nargs="+",
        help="Optional subset such as SPY ES3.SI. Defaults to every configured asset.",
    )
    parser.add_argument(
        "--start",
        help="Inclusive UTC sizing date; defaults to the latest first asset date.",
    )
    parser.add_argument(
        "--end",
        help="Exclusive UTC sizing date; defaults to one day after common data ends.",
    )
    return parser


def _read_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(f"Configuration not found: {path}")
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as error:
        raise ValueError(f"Invalid JSON in {path}: {error}") from error
    if not isinstance(value, dict):
        raise ValueError(f"Configuration in {path} must be a JSON object")
    return value


def _resolve_instruments(
    requested: list[str] | None,
    configured: dict[str, Any],
) -> list[str]:
    if requested is None:
        return list(configured)
    selected: list[str] = []
    for value in requested:
        normalized = value.strip().lower()
        try:
            instrument_id = INSTRUMENT_ALIASES[normalized]
        except KeyError as error:
            choices = ", ".join(sorted(INSTRUMENT_ALIASES))
            raise ValueError(
                f"Unsupported instrument {value!r}; choose one of: {choices}"
            ) from error
        if instrument_id not in configured:
            raise ValueError(f"No sizing configuration for {instrument_id}")
        if instrument_id not in selected:
            selected.append(instrument_id)
    return selected


def _utc_date(value: object, label: str) -> pd.Timestamp:
    try:
        timestamp = pd.Timestamp(value)
    except (TypeError, ValueError) as error:
        raise ValueError(f"{label} must be a valid timestamp") from error
    if timestamp.tzinfo is None:
        timestamp = timestamp.tz_localize("UTC")
    else:
        timestamp = timestamp.tz_convert("UTC")
    return timestamp.normalize()


def _default_range(prices: pd.DataFrame) -> tuple[pd.Timestamp, pd.Timestamp]:
    first_dates = [prices[column].first_valid_index() for column in prices]
    last_dates = [prices[column].last_valid_index() for column in prices]
    if any(value is None for value in first_dates + last_dates):
        raise ValueError("At least one selected asset has no daily prices")
    start = max(first_dates)
    end = min(last_dates) + pd.Timedelta(days=1)
    if end <= start:
        raise ValueError("Selected assets have no overlapping daily date range")
    return pd.Timestamp(start), pd.Timestamp(end)


def _validate_quote_currencies(
    canonical: pd.DataFrame,
    expected: dict[str, str],
) -> None:
    """Require canonical currency metadata to agree with sizing configuration."""
    if "quote_currency" not in canonical.columns:
        raise ValueError("Canonical data is missing quote_currency")
    selected = canonical.loc[
        canonical["instrument_id"].isin(expected),
        ["instrument_id", "quote_currency"],
    ].copy()
    selected["quote_currency"] = (
        selected["quote_currency"].astype("string").str.strip().str.upper()
    )
    for instrument_id, configured_currency in expected.items():
        observed = set(
            selected.loc[
                selected["instrument_id"] == instrument_id,
                "quote_currency",
            ].dropna()
        )
        required = configured_currency.strip().upper()
        if observed != {required}:
            raise ValueError(
                f"{instrument_id} quote_currency is {sorted(observed)} in the "
                f"canonical data but {required} in the sizing configuration"
            )


def main() -> None:
    args = build_parser().parse_args()
    config = _read_json(args.config_file)
    configured_assets = config.get("assets")
    if not isinstance(configured_assets, dict) or not configured_assets:
        raise ValueError("portfolio_sizing configuration must contain assets")
    instruments = _resolve_instruments(args.instrument, configured_assets)

    policy = InverseVolatilitySettings(
        lookback_observations=int(config.get("lookback_observations", 60)),
        minimum_observations=int(config.get("minimum_observations", 40)),
        target_gross_exposure=float(config.get("target_gross_exposure", 0.95)),
        volatility_floor=float(config.get("volatility_floor", 0.05)),
        rebalance_frequency=str(config.get("rebalance_frequency", "weekly")),
    )
    asset_settings = {
        instrument_id: AssetRiskSettings(
            annualization_periods=int(
                configured_assets[instrument_id]["annualization_periods"]
            ),
            maximum_weight=float(
                configured_assets[instrument_id]["maximum_weight"]
            ),
            quantity_step=float(configured_assets[instrument_id]["quantity_step"]),
        )
        for instrument_id in instruments
    }
    policy.validate()
    for instrument_id, settings in asset_settings.items():
        settings.validate(instrument_id)

    if not args.input_file.is_file():
        raise FileNotFoundError(f"Canonical data not found: {args.input_file}")
    base_currency = str(config.get("base_currency", "")).strip().upper()
    if not base_currency:
        raise ValueError("portfolio_sizing configuration must set base_currency")
    initial_cash_base = float(config.get("initial_cash_base", 100_000.0))
    if initial_cash_base <= 0:
        raise ValueError("initial_cash_base must be positive")
    starting_positions = str(config.get("starting_positions", "none")).strip().lower()
    if starting_positions != "none":
        raise ValueError("Only a zero-position portfolio start is currently supported")
    external_cash_flow_policy = str(
        config.get(
            "external_cash_flow_policy",
            "no_external_injections_or_withdrawals",
        )
    ).strip()
    if external_cash_flow_policy != "no_external_injections_or_withdrawals":
        raise ValueError(
            "Only a fixed-capital simulation without external cash flows is supported"
        )
    quote_currencies = {
        instrument_id: str(configured_assets[instrument_id]["quote_currency"])
        for instrument_id in instruments
    }
    static_rates = config.get("static_rates_to_base", {})
    if not isinstance(static_rates, dict):
        raise ValueError("static_rates_to_base must be a JSON object")

    canonical = pd.read_csv(args.input_file, low_memory=False)
    _validate_quote_currencies(canonical, quote_currencies)
    daily_native_prices = daily_adjusted_closes(canonical, instruments)
    fx_rates = (
        load_fx_rates(args.fx_file, base_currency)
        if args.fx_file is not None
        else None
    )
    daily_prices, conversion_audit = convert_daily_prices_to_base(
        daily_native_prices,
        quote_currencies,
        base_currency,
        static_rates_to_base=static_rates,
        fx_rates=fx_rates,
    )
    default_start, default_end = _default_range(daily_prices[instruments])
    start = _utc_date(args.start, "start") if args.start else default_start
    end = _utc_date(args.end, "end") if args.end else default_end
    dates = rebalance_dates(start, end, policy.rebalance_frequency)
    weights, diagnostics = build_weight_schedule(
        daily_prices[instruments],
        dates,
        policy,
        asset_settings,
    )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    weight_path = args.output_dir / "inverse_volatility_weights.csv"
    diagnostic_path = args.output_dir / "volatility_diagnostics.csv"
    conversion_path = args.output_dir / "currency_conversion.csv"
    weight_output = weights.copy()
    if not weight_output.empty:
        weight_output["allocated_weight"] = weight_output.sum(axis=1)
        weight_output["cash_weight"] = 1.0 - weight_output["allocated_weight"]
    weight_output.reset_index().to_csv(weight_path, index=False)
    diagnostics.to_csv(diagnostic_path, index=False)
    conversion_audit.to_csv(conversion_path, index=False)

    manifest = {
        "generated_at_utc": pd.Timestamp.now(tz="UTC").isoformat(),
        "input_file": str(args.input_file),
        "fx_file": str(args.fx_file) if args.fx_file is not None else None,
        "config_file": str(args.config_file),
        "output_directory": str(args.output_dir),
        "instruments": instruments,
        "start_utc": start.isoformat(),
        "end_utc_exclusive": end.isoformat(),
        "rebalance_dates_requested": len(dates),
        "rebalance_dates_ready": len(weights),
        "uses_prices_strictly_before_rebalance_date": True,
        "signals_applied": False,
        "quantity_conversion_performed": False,
        "base_currency": base_currency,
        "reported_currency": base_currency,
        "initial_cash_base": initial_cash_base,
        "starting_positions": starting_positions,
        "external_cash_flow_policy": external_cash_flow_policy,
        "usdt_policy": "USDT is treated as USD at a fixed 1:1 rate",
        "volatility_currency_basis": config.get("volatility_currency_basis"),
        "configuration_snapshot": config,
    }
    (args.output_dir / "portfolio_sizing_manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n",
        encoding="utf-8",
    )

    if weights.empty:
        raise ValueError(
            "No rebalance date had sufficient history for every selected asset; "
            f"inspect {diagnostic_path}"
        )

    latest_date = weights.index[-1]
    latest = weights.iloc[-1]
    print(f"\nLatest ready inverse-volatility weights ({latest_date.date()})")
    for instrument_id in instruments:
        print(f"  {instrument_id}: {latest[instrument_id]:.2%}")
    print(f"  allocated: {latest.sum():.2%}")
    print(f"  cash reserve/unallocated: {1.0 - latest.sum():.2%}")
    print(f"\nOutputs -> {args.output_dir}")
    print(f"All sizing prices were converted to {base_currency} before calculation.")
    print(
        f"Simulation policy: start with {initial_cash_base:,.2f} {base_currency}, "
        "no positions, and no later external cash injections or withdrawals."
    )
    print(
        "This command produces risk weights only; signal combination and target "
        "quantity execution remain separate steps."
    )


if __name__ == "__main__":
    main()

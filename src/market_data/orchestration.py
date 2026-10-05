"""Configuration-driven orchestration of the complete market-data workflow."""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pandas as pd

from market_data.aggregation import run_aggregation
from market_data.backfill import run_backfill
from market_data.config import (
    DEFAULT_DATA_DELAY_MINUTES,
    PANDAS_BAR_FREQUENCY,
    Instrument,
    select_instruments,
)
from market_data.consolidation import ConsolidationResult, run_consolidation
from market_data.pipeline import run_ingestion


@dataclass(frozen=True)
class AssetRun:
    """One selected provider instrument and its requested UTC range."""

    instrument: Instrument
    start_utc: datetime
    end_utc: datetime
    output_directory: Path


@dataclass(frozen=True)
class WorkflowConfig:
    """Validated paths and asset requests for one workflow execution."""

    assets: tuple[AssetRun, ...]
    canonical_output_file: Path
    canonical_summary_file: Path


def _parse_utc(value: object, field_name: str) -> datetime:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{field_name} must be a non-empty ISO timestamp")
    timestamp = pd.Timestamp(value)
    if timestamp.tzinfo is None:
        timestamp = timestamp.tz_localize("UTC")
    else:
        timestamp = timestamp.tz_convert("UTC")
    return timestamp.to_pydatetime()


def _default_end_utc() -> datetime:
    now = pd.Timestamp.now(tz="UTC")
    return (
        now.floor(PANDAS_BAR_FREQUENCY)
        - pd.Timedelta(minutes=DEFAULT_DATA_DELAY_MINUTES)
    ).to_pydatetime()


def _resolve_range(
    item: dict[str, object],
    shared_default_end_utc: datetime,
    label: str,
) -> tuple[datetime, datetime]:
    start_value = item.get("start")
    end_value = item.get("end")
    lookback_value = item.get("lookback_days")

    if start_value is not None:
        if lookback_value is not None:
            raise ValueError(
                f"{label} cannot provide both start and lookback_days"
            )
        if end_value is None:
            raise ValueError(f"{label} must provide end when start is used")
        start_utc = _parse_utc(start_value, f"{label}.start")
        end_utc = _parse_utc(end_value, f"{label}.end")
    else:
        if lookback_value is None:
            raise ValueError(
                f"{label} must provide either start/end or lookback_days"
            )
        try:
            lookback_days = int(lookback_value)
        except (TypeError, ValueError) as error:
            raise ValueError(f"{label}.lookback_days must be an integer") from error
        if lookback_days < 1:
            raise ValueError(f"{label}.lookback_days must be at least 1")
        end_utc = (
            _parse_utc(end_value, f"{label}.end")
            if end_value is not None
            else shared_default_end_utc
        )
        start_utc = end_utc - timedelta(days=lookback_days)

    if end_utc <= start_utc:
        raise ValueError(f"{label}.end must be later than start")
    return start_utc, end_utc


def load_workflow_config(path: Path) -> WorkflowConfig:
    """Read and validate one JSON market-data workflow configuration."""
    if not path.is_file():
        raise FileNotFoundError(f"Workflow configuration not found: {path}")
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("Workflow configuration must contain one JSON object")

    asset_items = payload.get("assets")
    if not isinstance(asset_items, list) or not asset_items:
        raise ValueError("Workflow configuration must contain a non-empty assets list")

    output_root = Path(str(payload.get("output_root", "data/staging")))
    shared_default_end = _default_end_utc()
    assets: list[AssetRun] = []
    seen_instrument_ids: set[str] = set()
    seen_output_directories: set[Path] = set()

    for index, raw_item in enumerate(asset_items):
        label = f"assets[{index}]"
        if not isinstance(raw_item, dict):
            raise ValueError(f"{label} must be a JSON object")
        provider = raw_item.get("provider")
        symbol = raw_item.get("symbol")
        if not isinstance(provider, str) or not provider.strip():
            raise ValueError(f"{label}.provider must be a non-empty string")
        if not isinstance(symbol, str) or not symbol.strip():
            raise ValueError(f"{label}.symbol must be a non-empty string")

        selected = select_instruments(provider, symbol)
        if len(selected) != 1:
            raise ValueError(f"{label} must resolve to exactly one instrument")
        instrument = selected[0]
        if instrument.instrument_id in seen_instrument_ids:
            raise ValueError(
                "Only one provider may be selected for each instrument_id; "
                f"duplicate: {instrument.instrument_id}"
            )
        seen_instrument_ids.add(instrument.instrument_id)

        start_utc, end_utc = _resolve_range(
            raw_item,
            shared_default_end,
            label,
        )
        configured_output = raw_item.get("output_dir")
        if configured_output is not None:
            output_directory = Path(str(configured_output))
        else:
            safe_symbol = instrument.source_symbol.replace(".", "_")
            output_directory = (
                output_root / instrument.provider / safe_symbol
            )
        if output_directory in seen_output_directories:
            raise ValueError(
                f"More than one asset uses output directory {output_directory}"
            )
        seen_output_directories.add(output_directory)
        assets.append(
            AssetRun(
                instrument=instrument,
                start_utc=start_utc,
                end_utc=end_utc,
                output_directory=output_directory,
            )
        )

    return WorkflowConfig(
        assets=tuple(assets),
        canonical_output_file=Path(
            str(
                payload.get(
                    "canonical_output_file",
                    "data/canonical/market_bars_30m.csv",
                )
            )
        ),
        canonical_summary_file=Path(
            str(
                payload.get(
                    "canonical_summary_file",
                    "data/canonical/consolidation_summary.json",
                )
            )
        ),
    )


def _asset_label(asset: AssetRun) -> str:
    return f"{asset.instrument.provider}:{asset.instrument.source_symbol}"


def run_market_data_workflow(config: WorkflowConfig) -> ConsolidationResult:
    """Run strict stage barriers and stop immediately when any call fails."""
    asset_count = len(config.assets)

    print("Stage 1/4: ingestion")
    for index, asset in enumerate(config.assets, start=1):
        print(f"[{index}/{asset_count}] Ingesting {_asset_label(asset)}")
        run_ingestion(
            asset.start_utc,
            asset.end_utc,
            asset.output_directory,
            instruments=(asset.instrument,),
        )

    print("Stage 2/4: aggregation")
    for index, asset in enumerate(config.assets, start=1):
        print(f"[{index}/{asset_count}] Aggregating {_asset_label(asset)}")
        run_aggregation(asset.output_directory, asset.output_directory)

    print("Stage 3/4: backfill")
    for index, asset in enumerate(config.assets, start=1):
        print(f"[{index}/{asset_count}] Backfilling {_asset_label(asset)}")
        run_backfill(asset.output_directory, asset.output_directory)

    print("Stage 4/4: consolidation")
    return run_consolidation(
        [asset.output_directory for asset in config.assets],
        config.canonical_output_file,
        config.canonical_summary_file,
        required_instruments={
            asset.instrument.instrument_id for asset in config.assets
        },
    )

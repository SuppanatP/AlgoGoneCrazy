"""Compare accepted source bars with expected market-calendar bars."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime

import pandas as pd

from market_data.config import INSTRUMENTS, Instrument
from market_data.schema import (
    COVERAGE_GAP_COLUMNS,
    REJECTED_BAR_COLUMNS,
    SESSION_COVERAGE_COLUMNS,
    STANDARD_BAR_COLUMNS,
)
from market_data.sessions import build_market_schedule


@dataclass(frozen=True)
class CoverageAudit:
    """Results from exchange-session and expected-bar checks."""

    accepted: pd.DataFrame
    rejected: pd.DataFrame
    missing_bars: pd.DataFrame
    sessions: pd.DataFrame


def _empty_rejected() -> pd.DataFrame:
    return pd.DataFrame(columns=REJECTED_BAR_COLUMNS)


def _empty_missing() -> pd.DataFrame:
    return pd.DataFrame(columns=COVERAGE_GAP_COLUMNS)


def _instrument_audit(
    accepted: pd.DataFrame,
    instrument: Instrument,
    start_utc: datetime,
    end_utc: datetime,
    retrieved_at_utc: datetime,
) -> CoverageAudit:
    schedule = build_market_schedule(
        instrument, start_utc, end_utc, retrieved_at_utc
    )
    actual = accepted.loc[
        accepted["instrument_id"] == instrument.instrument_id
    ].copy()
    expected = schedule.expected_bars.copy()

    expected_times = pd.DatetimeIndex(expected["bar_open_time_utc"])
    in_session = actual["bar_open_time_utc"].isin(expected_times)
    valid = actual.loc[in_session, STANDARD_BAR_COLUMNS].copy()

    outside = actual.loc[~in_session, STANDARD_BAR_COLUMNS].copy()
    if outside.empty:
        session_rejected = _empty_rejected()
    else:
        outside["quality_status"] = "reject"
        outside["rejection_reason"] = "outside_expected_market_session"
        session_rejected = outside.reindex(columns=REJECTED_BAR_COLUMNS)

    actual_times = pd.DatetimeIndex(valid["bar_open_time_utc"])
    missing = expected.loc[
        ~expected["bar_open_time_utc"].isin(actual_times)
    ].copy()
    if missing.empty:
        missing_bars = _empty_missing()
    else:
        missing_bars = missing.rename(
            columns={
                "bar_open_time_utc": "expected_bar_open_time_utc",
                "bar_close_time_utc": "expected_bar_close_time_utc",
            }
        )
        missing_bars["gap_reason"] = "missing_expected_bar"
        missing_bars["detected_at_utc"] = pd.Timestamp(retrieved_at_utc)
        missing_bars = missing_bars.reindex(columns=COVERAGE_GAP_COLUMNS)

    sessions = schedule.sessions.copy()
    accepted_counts = valid.groupby("session_date_local").size()
    missing_counts = missing.groupby("session_date_local").size()
    sessions["accepted_bar_count"] = sessions["session_date_local"].map(
        accepted_counts
    ).fillna(0).astype(int)
    sessions["missing_bar_count"] = sessions["session_date_local"].map(
        missing_counts
    ).fillna(0).astype(int)
    expected_count = pd.to_numeric(sessions["expected_bar_count"])
    sessions["coverage_ratio"] = sessions["accepted_bar_count"].div(
        expected_count
    ).where(expected_count > 0)
    sessions = sessions.reindex(columns=SESSION_COVERAGE_COLUMNS)

    return CoverageAudit(valid, session_rejected, missing_bars, sessions)


def audit_market_coverage(
    accepted: pd.DataFrame,
    start_utc: datetime,
    end_utc: datetime,
    retrieved_at_utc: datetime,
) -> CoverageAudit:
    """Reject off-session rows and report expected bars that are absent."""
    audits = [
        _instrument_audit(
            accepted,
            instrument,
            start_utc,
            end_utc,
            retrieved_at_utc,
        )
        for instrument in INSTRUMENTS
    ]

    valid = pd.concat(
        [audit.accepted for audit in audits], ignore_index=True
    ).reindex(columns=STANDARD_BAR_COLUMNS)
    rejected = pd.concat(
        [audit.rejected for audit in audits], ignore_index=True
    ).reindex(columns=REJECTED_BAR_COLUMNS)
    missing = pd.concat(
        [audit.missing_bars for audit in audits], ignore_index=True
    ).reindex(columns=COVERAGE_GAP_COLUMNS)
    sessions = pd.concat(
        [audit.sessions for audit in audits], ignore_index=True
    ).reindex(columns=SESSION_COVERAGE_COLUMNS)

    valid = valid.sort_values(
        ["instrument_id", "bar_open_time_utc"]
    ).reset_index(drop=True)
    rejected = rejected.sort_values(
        ["instrument_id", "bar_open_time_utc"]
    ).reset_index(drop=True)
    missing = missing.sort_values(
        ["instrument_id", "expected_bar_open_time_utc"]
    ).reset_index(drop=True)
    sessions = sessions.sort_values(
        ["instrument_id", "session_date_local"]
    ).reset_index(drop=True)
    return CoverageAudit(valid, rejected, missing, sessions)

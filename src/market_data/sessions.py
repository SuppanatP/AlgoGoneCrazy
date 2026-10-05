"""Build expected bars from exchange calendars and configured session rules."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime

import exchange_calendars as xcals
import pandas as pd

from market_data.config import (
    BAR_DURATION,
    BAR_INTERVAL,
    PANDAS_BAR_FREQUENCY,
    Instrument,
)
from market_data.schema import EXPECTED_BAR_COLUMNS, SESSION_COVERAGE_COLUMNS


@dataclass(frozen=True)
class MarketSchedule:
    """Expected completed bars and daily session descriptions."""

    expected_bars: pd.DataFrame
    sessions: pd.DataFrame


def _as_utc_timestamp(value: datetime) -> pd.Timestamp:
    timestamp = pd.Timestamp(value)
    if timestamp.tzinfo is None:
        return timestamp.tz_localize("UTC")
    return timestamp.tz_convert("UTC")


def _empty_expected_bars() -> pd.DataFrame:
    return pd.DataFrame(columns=EXPECTED_BAR_COLUMNS)


def _empty_sessions() -> pd.DataFrame:
    return pd.DataFrame(columns=SESSION_COVERAGE_COLUMNS)


def _bar_opens_for_segment(
    segment_open: pd.Timestamp,
    segment_close: pd.Timestamp,
) -> pd.DatetimeIndex:
    """Return starts of complete bars contained within one trading segment."""
    last_open = segment_close - pd.Timedelta(BAR_DURATION)
    if last_open < segment_open:
        return pd.DatetimeIndex([], tz="UTC")
    return pd.date_range(segment_open, last_open, freq=PANDAS_BAR_FREQUENCY)


def _expected_rows(
    instrument: Instrument,
    opens: pd.DatetimeIndex,
) -> pd.DataFrame:
    if opens.empty:
        return _empty_expected_bars()

    closes = opens + pd.Timedelta(BAR_DURATION)
    return pd.DataFrame(
        {
            "instrument_id": instrument.instrument_id,
            "source_symbol": instrument.source_symbol,
            "exchange": instrument.exchange,
            "calendar_name": instrument.calendar_name,
            "bar_interval": BAR_INTERVAL,
            "bar_open_time_utc": opens,
            "bar_close_time_utc": closes,
            "session_date_local": opens.tz_convert(
                instrument.session_timezone
            ).date,
        }
    ).reindex(columns=EXPECTED_BAR_COLUMNS)


def _session_times(
    instrument: Instrument,
    session_date: object,
    row: pd.Series,
) -> tuple[pd.Timestamp, pd.Timestamp, pd.Timestamp, pd.Timestamp]:
    """Read calendar times and apply a configured regular-session break."""
    market_open = pd.Timestamp(row["open"])
    market_close = pd.Timestamp(row["close"])
    break_start = pd.Timestamp(row["break_start"])
    break_end = pd.Timestamp(row["break_end"])

    configured_start = instrument.regular_break_start_local
    configured_end = instrument.regular_break_end_local
    if (
        (pd.isna(break_start) or pd.isna(break_end))
        and configured_start is not None
        and configured_end is not None
    ):
        local_midnight = pd.Timestamp(
            session_date, tz=instrument.session_timezone
        )
        candidate_start = local_midnight + pd.Timedelta(
            hours=configured_start.hour,
            minutes=configured_start.minute,
        )
        candidate_end = local_midnight + pd.Timedelta(
            hours=configured_end.hour,
            minutes=configured_end.minute,
        )
        candidate_start = candidate_start.tz_convert("UTC")
        candidate_end = candidate_end.tz_convert("UTC")
        if market_open < candidate_start and candidate_end < market_close:
            break_start = candidate_start
            break_end = candidate_end

    return market_open, break_start, break_end, market_close


def _continuous_schedule(
    instrument: Instrument,
    start: pd.Timestamp,
    end: pd.Timestamp,
    retrieved: pd.Timestamp,
) -> MarketSchedule:
    first_open = start.ceil(PANDAS_BAR_FREQUENCY)
    last_requested_open = (end - pd.Timedelta(nanoseconds=1)).floor(
        PANDAS_BAR_FREQUENCY
    )
    last_completed_open = (retrieved - pd.Timedelta(BAR_DURATION)).floor(
        PANDAS_BAR_FREQUENCY
    )
    last_open = min(last_requested_open, last_completed_open)
    if last_open < first_open:
        opens = pd.DatetimeIndex([], tz="UTC")
    else:
        opens = pd.date_range(first_open, last_open, freq=PANDAS_BAR_FREQUENCY)

    expected = _expected_rows(instrument, opens)
    coverage_horizon = min(end, retrieved)
    last_included = coverage_horizon - pd.Timedelta(nanoseconds=1)
    if last_included < start:
        return MarketSchedule(expected, _empty_sessions())

    dates = pd.date_range(
        start.tz_convert(instrument.session_timezone).date(),
        last_included.tz_convert(instrument.session_timezone).date(),
        freq="D",
    )
    expected_counts = expected.groupby("session_date_local").size()
    session_rows: list[dict[str, object]] = []
    for day in dates:
        local_midnight = pd.Timestamp(day.date(), tz=instrument.session_timezone)
        session_open = local_midnight.tz_convert("UTC")
        session_close = (local_midnight + pd.Timedelta(days=1)).tz_convert("UTC")
        session_rows.append(
            {
                "instrument_id": instrument.instrument_id,
                "source_symbol": instrument.source_symbol,
                "exchange": instrument.exchange,
                "calendar_name": instrument.calendar_name,
                "session_date_local": day.date(),
                "session_status": "continuous_24_7",
                "session_open_time_utc": session_open,
                "break_start_time_utc": pd.NaT,
                "break_end_time_utc": pd.NaT,
                "session_close_time_utc": session_close,
                "expected_bar_count": int(expected_counts.get(day.date(), 0)),
                "accepted_bar_count": 0,
                "missing_bar_count": 0,
                "coverage_ratio": pd.NA,
            }
        )

    sessions = pd.DataFrame(session_rows).reindex(columns=SESSION_COVERAGE_COLUMNS)
    return MarketSchedule(expected, sessions)


def _exchange_schedule(
    instrument: Instrument,
    start: pd.Timestamp,
    end: pd.Timestamp,
    retrieved: pd.Timestamp,
) -> MarketSchedule:
    coverage_horizon = min(end, retrieved)
    last_included = coverage_horizon - pd.Timedelta(nanoseconds=1)
    if last_included < start:
        return MarketSchedule(_empty_expected_bars(), _empty_sessions())

    first_local_date = start.tz_convert(instrument.session_timezone).date()
    last_local_date = last_included.tz_convert(instrument.session_timezone).date()
    calendar_end = (pd.Timestamp(last_local_date) + pd.Timedelta(days=1)).date()
    calendar = xcals.get_calendar(
        instrument.calendar_name,
        start=first_local_date,
        end=calendar_end,
        side="left",
    )
    schedule = calendar.schedule.loc[str(first_local_date) : str(last_local_date)]

    expected_frames: list[pd.DataFrame] = []
    schedule_by_date: dict[object, pd.Series] = {}
    for session_label, row in schedule.iterrows():
        session_date = pd.Timestamp(session_label).date()
        schedule_by_date[session_date] = row

        market_open, break_start, break_end, market_close = _session_times(
            instrument, session_date, row
        )

        if pd.isna(break_start) or pd.isna(break_end):
            opens = _bar_opens_for_segment(market_open, market_close)
        else:
            opens = _bar_opens_for_segment(
                market_open, break_start
            ).append(_bar_opens_for_segment(break_end, market_close))

        frame = _expected_rows(instrument, opens)
        frame = frame.loc[
            (frame["bar_open_time_utc"] >= start)
            & (frame["bar_open_time_utc"] < end)
            & (frame["bar_close_time_utc"] <= retrieved)
        ]
        if not frame.empty:
            expected_frames.append(frame)

    expected = (
        pd.concat(expected_frames, ignore_index=True)
        if expected_frames
        else _empty_expected_bars()
    )
    expected_counts = expected.groupby("session_date_local").size()

    session_rows: list[dict[str, object]] = []
    for day in pd.date_range(first_local_date, last_local_date, freq="D"):
        session_date = day.date()
        row = schedule_by_date.get(session_date)
        if row is None:
            status = (
                "weekend"
                if day.dayofweek >= 5
                else "holiday_or_scheduled_closure"
            )
            market_open = break_start = break_end = market_close = pd.NaT
        else:
            status = "open"
            market_open, break_start, break_end, market_close = _session_times(
                instrument, session_date, row
            )

        session_rows.append(
            {
                "instrument_id": instrument.instrument_id,
                "source_symbol": instrument.source_symbol,
                "exchange": instrument.exchange,
                "calendar_name": instrument.calendar_name,
                "session_date_local": session_date,
                "session_status": status,
                "session_open_time_utc": market_open,
                "break_start_time_utc": break_start,
                "break_end_time_utc": break_end,
                "session_close_time_utc": market_close,
                "expected_bar_count": int(expected_counts.get(session_date, 0)),
                "accepted_bar_count": 0,
                "missing_bar_count": 0,
                "coverage_ratio": pd.NA,
            }
        )

    sessions = pd.DataFrame(session_rows).reindex(columns=SESSION_COVERAGE_COLUMNS)
    return MarketSchedule(expected, sessions)


def build_market_schedule(
    instrument: Instrument,
    start_utc: datetime,
    end_utc: datetime,
    retrieved_at_utc: datetime,
) -> MarketSchedule:
    """Build expected completed bars and daily session status for one asset."""
    start = _as_utc_timestamp(start_utc)
    end = _as_utc_timestamp(end_utc)
    retrieved = _as_utc_timestamp(retrieved_at_utc)
    if instrument.calendar_name == "24/7":
        return _continuous_schedule(instrument, start, end, retrieved)
    return _exchange_schedule(instrument, start, end, retrieved)

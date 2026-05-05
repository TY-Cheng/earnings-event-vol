from __future__ import annotations

from collections.abc import Iterable
from datetime import date, datetime, time
from zoneinfo import ZoneInfo

import pandas as pd

from earnings_event_vol.schemas import AnnouncementTiming, EarningsEvent, EventWindow, UnderlyingBar

NEW_YORK_TZ = ZoneInfo("America/New_York")
REGULAR_CLOSE = time(hour=16, minute=0)


def _sorted_dates(dates: Iterable[date]) -> list[date]:
    return sorted(set(dates))


def previous_trading_date(trading_dates: Iterable[date], target: date) -> date | None:
    prior = [day for day in _sorted_dates(trading_dates) if day < target]
    return prior[-1] if prior else None


def next_trading_date(trading_dates: Iterable[date], target: date) -> date | None:
    future = [day for day in _sorted_dates(trading_dates) if day > target]
    return future[0] if future else None


def regular_close_timestamp(day: date) -> datetime:
    return datetime.combine(day, REGULAR_CLOSE, tzinfo=NEW_YORK_TZ)


def align_event_window(event: EarningsEvent, trading_dates: Iterable[date]) -> EventWindow:
    dates = _sorted_dates(trading_dates)
    entry_date: date | None
    exit_date: date | None
    if event.announcement_timing == AnnouncementTiming.AMC:
        entry_date = event.announcement_date
        exit_date = next_trading_date(dates, event.announcement_date)
    elif event.announcement_timing == AnnouncementTiming.BMO:
        entry_date = previous_trading_date(dates, event.announcement_date)
        exit_date = event.announcement_date if event.announcement_date in dates else None
    else:
        return EventWindow(
            ticker=event.ticker,
            announcement_date=event.announcement_date,
            announcement_timing=event.announcement_timing,
            entry_date=event.announcement_date,
            exit_date=event.announcement_date,
            feature_cutoff_date=event.announcement_date,
            event_entry_timestamp=regular_close_timestamp(event.announcement_date),
            source=event.source,
            sector=event.sector,
            exclusion_reason="non_bmo_amc",
        )
    if entry_date is None or exit_date is None:
        fallback = event.announcement_date
        return EventWindow(
            ticker=event.ticker,
            announcement_date=event.announcement_date,
            announcement_timing=event.announcement_timing,
            entry_date=entry_date or fallback,
            exit_date=exit_date or fallback,
            feature_cutoff_date=entry_date or fallback,
            event_entry_timestamp=regular_close_timestamp(entry_date or fallback),
            source=event.source,
            sector=event.sector,
            exclusion_reason="missing_entry_or_exit_date",
        )
    return EventWindow(
        ticker=event.ticker,
        announcement_date=event.announcement_date,
        announcement_timing=event.announcement_timing,
        entry_date=entry_date,
        exit_date=exit_date,
        feature_cutoff_date=entry_date,
        event_entry_timestamp=regular_close_timestamp(entry_date),
        source=event.source,
        sector=event.sector,
    )


def rvar_prices_for_window(
    window: EventWindow, bars: dict[tuple[str, date], UnderlyingBar]
) -> tuple[float, float]:
    before = bars[(window.ticker, window.entry_date)].close
    after = bars[(window.ticker, window.exit_date)].close
    return before, after


def has_ex_dividend_between(
    ex_dividend_dates: Iterable[date],
    *,
    start: date,
    end: date,
) -> bool:
    return any(start <= ex_date <= end for ex_date in ex_dividend_dates)


def is_halted_or_proxy_halted(bar: UnderlyingBar) -> tuple[bool, str | None]:
    if bar.vendor_halt_flag is True:
        return True, "vendor_halt_flag"
    if bar.proxy_halt_flag:
        return True, "proxy_zero_volume_unchanged_ohlc"
    return False, None


def validate_calendar_frame(frame: pd.DataFrame) -> pd.DataFrame:
    required = {"ticker", "announcement_date", "announcement_timing", "source"}
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(f"earnings calendar missing required columns: {missing}")
    out = frame.copy()
    out["announcement_date"] = pd.to_datetime(out["announcement_date"]).dt.date
    out["announcement_timing"] = out["announcement_timing"].map(
        lambda value: (
            EarningsEvent(
                ticker="CHECK",
                announcement_date=date(2000, 1, 1),
                announcement_timing=value,
                source="validation",
            ).announcement_timing.value
        )
    )
    out["is_main_sample_timing"] = out["announcement_timing"].isin(
        [AnnouncementTiming.BMO.value, AnnouncementTiming.AMC.value]
    )
    return out

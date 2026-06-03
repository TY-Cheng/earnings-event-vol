from __future__ import annotations

from collections.abc import Iterable, Mapping
from datetime import date, datetime, time, timedelta
from zoneinfo import ZoneInfo

import pandas as pd

from earnings_event_vol.schemas import AnnouncementTiming, EarningsEvent, EventWindow, UnderlyingBar

NEW_YORK_TZ = ZoneInfo("America/New_York")
REGULAR_CLOSE = time(hour=16, minute=0)
EARLY_CLOSE = time(hour=13, minute=0)


def _sorted_dates(dates: Iterable[date]) -> list[date]:
    return sorted(set(dates))


def _thanksgiving_day(year: int) -> date:
    first = date(year, 11, 1)
    days_until_thursday = (3 - first.weekday()) % 7
    return first + timedelta(days=days_until_thursday + 21)


def _nth_weekday(year: int, month: int, weekday: int, occurrence: int) -> date:
    first = date(year, month, 1)
    days_until_weekday = (weekday - first.weekday()) % 7
    return first + timedelta(days=days_until_weekday + 7 * (occurrence - 1))


def _last_weekday(year: int, month: int, weekday: int) -> date:
    last = date(year, month + 1, 1) - timedelta(days=1) if month < 12 else date(year, 12, 31)
    return last - timedelta(days=(last.weekday() - weekday) % 7)


def _observed_fixed_holiday(year: int, month: int, day: int) -> date:
    holiday = date(year, month, day)
    if holiday.weekday() == 5:
        return holiday - timedelta(days=1)
    if holiday.weekday() == 6:
        return holiday + timedelta(days=1)
    return holiday


def _easter_sunday(year: int) -> date:
    a = year % 19
    b = year // 100
    c = year % 100
    d = b // 4
    e = b % 4
    f = (b + 8) // 25
    g = (b - f + 1) // 3
    h = (19 * a + b - d - g + 15) % 30
    i = c // 4
    k = c % 4
    month_correction = (32 + 2 * e + 2 * i - h - k) % 7
    leap_correction = (a + 11 * h + 22 * month_correction) // 451
    month = (h + month_correction - 7 * leap_correction + 114) // 31
    day = ((h + month_correction - 7 * leap_correction + 114) % 31) + 1
    return date(year, month, day)


def us_equity_market_holidays(year: int) -> set[date]:
    holidays = {
        _observed_fixed_holiday(year, 1, 1),
        _nth_weekday(year, 1, 0, 3),
        _nth_weekday(year, 2, 0, 3),
        _easter_sunday(year) - timedelta(days=2),
        _last_weekday(year, 5, 0),
        _observed_fixed_holiday(year, 7, 4),
        _nth_weekday(year, 9, 0, 1),
        _thanksgiving_day(year),
        _observed_fixed_holiday(year, 12, 25),
    }
    if year >= 2022:
        holidays.add(_observed_fixed_holiday(year, 6, 19))
    holidays.update(
        {
            date(2018, 12, 5),
            date(2025, 1, 9),
            date(2012, 10, 29),
            date(2012, 10, 30),
        }
    )
    return holidays


def is_us_equity_trading_day(day: date) -> bool:
    if day.weekday() >= 5:
        return False
    return day not in us_equity_market_holidays(day.year)


def previous_us_equity_trading_day(target: date, *, include_target: bool = False) -> date:
    candidate = target if include_target else target - timedelta(days=1)
    while not is_us_equity_trading_day(candidate):
        candidate -= timedelta(days=1)
    return candidate


def next_us_equity_trading_day(target: date, *, include_target: bool = False) -> date:
    candidate = target if include_target else target + timedelta(days=1)
    while not is_us_equity_trading_day(candidate):
        candidate += timedelta(days=1)
    return candidate


def _scheduled_early_close_time(day: date) -> time | None:
    """Return scheduled NYSE-style half-day close used by the active proxy route."""
    if not is_us_equity_trading_day(day):
        return None
    if day == _thanksgiving_day(day.year) + timedelta(days=1):
        return EARLY_CLOSE
    if day.month == 7 and day.day == 3 and date(day.year, 7, 4).weekday() in {1, 2, 3, 4}:
        return EARLY_CLOSE
    if day.month == 12 and day.day == 24 and date(day.year, 12, 25).weekday() in {1, 2, 3, 4}:
        return EARLY_CLOSE
    return None


def previous_trading_date(trading_dates: Iterable[date], target: date) -> date | None:
    prior = [day for day in _sorted_dates(trading_dates) if day < target]
    return prior[-1] if prior else None


def next_trading_date(trading_dates: Iterable[date], target: date) -> date | None:
    future = [day for day in _sorted_dates(trading_dates) if day > target]
    return future[0] if future else None


def market_close_timestamp(
    day: date,
    *,
    early_closes: Mapping[date, time] | None = None,
) -> datetime:
    close_time = (early_closes or {}).get(day) or _scheduled_early_close_time(day) or REGULAR_CLOSE
    return datetime.combine(day, close_time, tzinfo=NEW_YORK_TZ)


def market_close_timestamp_utc(
    day: date,
    *,
    early_closes: Mapping[date, time] | None = None,
) -> datetime:
    return market_close_timestamp(day, early_closes=early_closes).astimezone(ZoneInfo("UTC"))


def regular_close_timestamp(day: date) -> datetime:
    return market_close_timestamp(day)


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

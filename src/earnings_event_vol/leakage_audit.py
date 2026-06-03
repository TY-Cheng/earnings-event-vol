from __future__ import annotations

import re
from collections.abc import Iterable
from dataclasses import dataclass
from datetime import datetime, time
from zoneinfo import ZoneInfo

import pandas as pd

DEFAULT_BLOCKED_PATTERNS = (
    re.compile(r"(^|_)s_after($|_)"),
    re.compile(r"(^|_)close_after($|_)"),
    re.compile(r"(^|_)open_after($|_)"),
    re.compile(r"(^|_)rvar_event($|_)"),
    re.compile(r"rvar_"),
    re.compile(r"(^|_)r_event_"),
    re.compile(r"gross_.*proxy_pnl"),
    re.compile(r"haircut_pnl"),
    re.compile(r"net_proxy_pnl"),
    re.compile(r"exit_option_value"),
    re.compile(r"exit_intrinsic"),
    re.compile(r"exit_option_vwap"),
    re.compile(r"post_open_option_vwap"),
    re.compile(r"c2o_"),
    re.compile(r"reaction_o2c"),
    re.compile(r"(^|_)realized(_|$)"),
    re.compile(r"(^|_)post_event(_|$)"),
    re.compile(r"(^|_)future(_|$)"),
    re.compile(r"same_event_return"),
    re.compile(r"preannouncement"),
    re.compile(r"prior_guidance"),
)
DEFAULT_VENDOR_FORECAST_PATTERNS = (
    re.compile(r"vendor_.*forecast"),
    re.compile(r"forecasted_iv"),
    re.compile(r"predicted_"),
)
NEW_YORK_TZ = ZoneInfo("America/New_York")


@dataclass(frozen=True)
class LeakageAuditResult:
    ok: bool
    asof_violations: pd.DataFrame
    blocked_columns: list[str]
    vendor_forecast_columns: list[str]


def _matching_columns(columns: Iterable[str], patterns: Iterable[re.Pattern[str]]) -> list[str]:
    return sorted(
        {
            column
            for column in columns
            for pattern in patterns
            if pattern.search(str(column).lower())
        }
    )


def _timestamp_tz_state(values: pd.Series) -> str:
    states: set[str] = set()
    for value in values.dropna():
        timestamp = pd.Timestamp(value)
        states.add("aware" if timestamp.tzinfo is not None else "naive")
    if not states:
        return "empty"
    if len(states) > 1:
        return "mixed"
    return next(iter(states))


def audit_feature_leakage(
    frame: pd.DataFrame,
    *,
    asof_col: str = "feature_asof_timestamp",
    entry_col: str = "event_entry_timestamp",
    vendor_forecast_whitelist: Iterable[str] = (),
) -> LeakageAuditResult:
    if asof_col not in frame.columns or entry_col not in frame.columns:
        raise ValueError(f"feature frame must include {asof_col} and {entry_col}")
    asof_state = _timestamp_tz_state(frame[asof_col])
    entry_state = _timestamp_tz_state(frame[entry_col])
    if "mixed" in {asof_state, entry_state} or (
        "empty" not in {asof_state, entry_state} and asof_state != entry_state
    ):
        asof_violations = frame.copy()
        asof_violations["leakage_audit_reason"] = "timezone_mismatch"
    else:
        use_utc = asof_state == "aware" or entry_state == "aware"
        asof = pd.to_datetime(frame[asof_col], errors="coerce", utc=use_utc)
        entry = pd.to_datetime(frame[entry_col], errors="coerce", utc=use_utc)
        asof_violations = frame.loc[asof.isna() | entry.isna() | (asof > entry)].copy()
    blocked_columns = _matching_columns(frame.columns, DEFAULT_BLOCKED_PATTERNS)
    vendor_forecasts = _matching_columns(frame.columns, DEFAULT_VENDOR_FORECAST_PATTERNS)
    whitelist = set(vendor_forecast_whitelist)
    vendor_forecasts = [column for column in vendor_forecasts if column not in whitelist]
    return LeakageAuditResult(
        ok=asof_violations.empty and not blocked_columns and not vendor_forecasts,
        asof_violations=asof_violations,
        blocked_columns=blocked_columns,
        vendor_forecast_columns=vendor_forecasts,
    )


def make_feature_timestamps(date_value: str, close_hour: int = 16) -> datetime:
    parsed = pd.Timestamp(date_value)
    return datetime.combine(
        parsed.date(), time(hour=close_hour, minute=0, second=0), tzinfo=NEW_YORK_TZ
    )

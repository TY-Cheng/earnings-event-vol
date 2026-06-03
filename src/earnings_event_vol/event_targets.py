from __future__ import annotations

import math
from datetime import date
from typing import Any, cast

import numpy as np
import pandas as pd

FORECAST_FLOOR = 1e-6
OPEN_STATUS_VENDOR_REGULAR_OHLC_ASSUMED = "vendor_regular_ohlc_assumed"
OPEN_STATUS_UNAVAILABLE = "unavailable"


def _safe_float(value: object) -> float | None:
    if value is None or pd.isna(value):
        return None
    out = float(cast(Any, value))
    return out if math.isfinite(out) and out > 0 else None


def _log_return(before: float | None, after: float | None) -> float | None:
    if before is None or after is None:
        return None
    return float(math.log(after / before))


def _variance_from_return(value: float | None) -> float | None:
    return None if value is None else float(value * value)


def _date(value: object) -> date | None:
    if value is None or pd.isna(value):
        return None
    return cast(date, pd.Timestamp(value).date())


def _excluded_target_row() -> dict[str, object]:
    return {
        "close_before": None,
        "open_after": None,
        "close_after": None,
        "s_before": None,
        "s_after": None,
        "r_event_jump_c2o": None,
        "RVAR_event_jump_c2o": None,
        "r_event_day_c2c": None,
        "RVAR_event_day_c2c": None,
        "r_event_reaction_o2c": None,
        "RVAR_event_reaction_o2c": None,
        "return_decomposition_residual": None,
        "RVAR_cross_term": None,
        "RVAR_day_reconstructed": None,
        "cross_term_share_raw": None,
        "cross_term_share_floored": None,
        "rvar_event": None,
        "primary_target_id": "jump_c2o",
        "open_after_status": OPEN_STATUS_UNAVAILABLE,
        "open_after_source": None,
        "open_after_is_regular_session": None,
        "open_after_is_adjusted": None,
        "open_after_volume_available": False,
    }


def add_event_return_targets(
    windows: pd.DataFrame,
    bars: pd.DataFrame,
    *,
    forecast_floor: float = FORECAST_FLOOR,
) -> pd.DataFrame:
    """Add C2O/C2C/O2C realized-variance targets and open audit fields.

    `rvar_event` remains a backward-compatible close-to-close alias.
    """
    out = windows.copy()
    if bars.empty:
        bar_lookup: dict[tuple[str, date], dict[str, object]] = {}
    else:
        frame = bars.copy()
        frame["ticker"] = frame["ticker"].astype(str).str.upper()
        frame["date"] = pd.to_datetime(frame["date"], errors="coerce").dt.date
        bar_lookup = {
            (str(row["ticker"]).upper(), row["date"]): row for row in frame.to_dict("records")
        }

    rows: list[dict[str, object]] = []
    for record in out.to_dict("records"):
        exclusion_reason = record.get("exclusion_reason")
        if exclusion_reason is not None and not pd.isna(exclusion_reason) and str(exclusion_reason):
            rows.append(_excluded_target_row())
            continue
        ticker = str(record.get("ticker", "")).upper()
        entry_date = _date(record.get("entry_date"))
        exit_date = _date(record.get("exit_date"))
        before_bar = bar_lookup.get((ticker, entry_date)) if entry_date else None
        after_bar = bar_lookup.get((ticker, exit_date)) if exit_date else None
        close_before = _safe_float(None if before_bar is None else before_bar.get("close"))
        open_after = _safe_float(None if after_bar is None else after_bar.get("open"))
        close_after = _safe_float(None if after_bar is None else after_bar.get("close"))

        r_jump = _log_return(close_before, open_after)
        r_day = _log_return(close_before, close_after)
        r_reaction = _log_return(open_after, close_after)
        rvar_jump = _variance_from_return(r_jump)
        rvar_day = _variance_from_return(r_day)
        rvar_reaction = _variance_from_return(r_reaction)
        cross_term = (
            None if r_jump is None or r_reaction is None else float(2.0 * r_jump * r_reaction)
        )
        reconstructed = (
            None
            if rvar_jump is None or rvar_reaction is None or cross_term is None
            else float(rvar_jump + rvar_reaction + cross_term)
        )
        residual = (
            None
            if r_day is None or r_jump is None or r_reaction is None
            else float(r_day - r_jump - r_reaction)
        )
        denominator = reconstructed if reconstructed is not None else np.nan
        share_raw = (
            None
            if cross_term is None or denominator is None or not np.isfinite(denominator)
            else float(abs(cross_term) / denominator)
            if denominator != 0
            else np.nan
        )
        share_floored = (
            None
            if cross_term is None
            else float(abs(cross_term) / max(abs(denominator), forecast_floor))
            if denominator is not None and np.isfinite(denominator)
            else None
        )
        open_available = open_after is not None
        rows.append(
            {
                "close_before": close_before,
                "open_after": open_after,
                "close_after": close_after,
                "s_before": close_before,
                "s_after": close_after,
                "r_event_jump_c2o": r_jump,
                "RVAR_event_jump_c2o": rvar_jump,
                "r_event_day_c2c": r_day,
                "RVAR_event_day_c2c": rvar_day,
                "r_event_reaction_o2c": r_reaction,
                "RVAR_event_reaction_o2c": rvar_reaction,
                "return_decomposition_residual": residual,
                "RVAR_cross_term": cross_term,
                "RVAR_day_reconstructed": reconstructed,
                "cross_term_share_raw": share_raw,
                "cross_term_share_floored": share_floored,
                "rvar_event": rvar_day,
                "primary_target_id": "jump_c2o",
                "open_after_status": OPEN_STATUS_VENDOR_REGULAR_OHLC_ASSUMED
                if open_available
                else OPEN_STATUS_UNAVAILABLE,
                "open_after_source": "underlying_day_aggs_open" if open_available else None,
                "open_after_is_regular_session": None,
                "open_after_is_adjusted": None,
                "open_after_volume_available": bool(
                    after_bar is not None
                    and _safe_float(after_bar.get("volume")) is not None
                    and float(cast(Any, after_bar.get("volume"))) > 0
                ),
            }
        )

    additions = pd.DataFrame(rows, index=out.index)
    for column in additions.columns:
        out[column] = additions[column]
    return out


def available_target_columns(frame: pd.DataFrame) -> dict[str, str]:
    candidates = {
        "jump_c2o": "RVAR_event_jump_c2o",
        "day_c2c": "RVAR_event_day_c2c",
        "reaction_o2c": "RVAR_event_reaction_o2c",
    }
    return {target_id: column for target_id, column in candidates.items() if column in frame}


def target_label_column(target_id: str, frame: pd.DataFrame) -> str:
    if target_id == "jump_c2o":
        return "RVAR_event_jump_c2o"
    if target_id == "day_c2c":
        return "RVAR_event_day_c2c" if "RVAR_event_day_c2c" in frame.columns else "rvar_event"
    if target_id == "reaction_o2c":
        return "RVAR_event_reaction_o2c"
    raise ValueError(f"unsupported target_id: {target_id}")

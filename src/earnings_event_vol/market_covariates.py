from __future__ import annotations

from bisect import bisect_left, bisect_right
from datetime import date
from typing import Literal, cast

import numpy as np
import pandas as pd

FRED_VIXCLS_URL = "https://fred.stlouisfed.org/graph/fredgraph.csv?id=VIXCLS"
MARKET_COVARIATE_SCHEMA_VERSION = "v1.0"
VIX_ALIGNMENT_PRIOR_CLOSE: Literal["prior_close_default"] = "prior_close_default"
VIX_ALIGNMENT_SAME_DAY_AMC: Literal["same_day_close_for_amc"] = "same_day_close_for_amc"
VixAlignment = Literal["prior_close_default", "same_day_close_for_amc"]

VIX_FEATURE_COLUMNS = [
    "resolved_vix_date",
    "vix_lag_days",
    "vix_level",
    "vix_change_1d",
    "vix_change_5d",
    "vix_percentile_252d",
    "vix_regime_tercile",
    "vix_above_30",
    "vix_available",
    "vix_alignment",
    "max_vix_lag_days",
]


def _coerce_date(value: object) -> date | None:
    timestamp = pd.Timestamp(value)
    if pd.isna(timestamp):
        return None
    return cast(date, timestamp.date())


def normalize_fred_vixcls_csv(
    raw: pd.DataFrame,
    *,
    source_snapshot_date: date,
    source_url: str = FRED_VIXCLS_URL,
    schema_version: str = MARKET_COVARIATE_SCHEMA_VERSION,
) -> pd.DataFrame:
    """Normalize a FRED VIXCLS CSV response into a reproducible silver table."""
    if "DATE" in raw.columns:
        date_column = "DATE"
    elif "observation_date" in raw.columns:
        date_column = "observation_date"
    elif "date" in raw.columns:
        date_column = "date"
    else:
        raise ValueError("FRED VIXCLS CSV requires DATE column")

    if "VIXCLS" in raw.columns:
        value_column = "VIXCLS"
    elif "vix_close" in raw.columns:
        value_column = "vix_close"
    else:
        raise ValueError("FRED VIXCLS CSV requires VIXCLS column")

    out = pd.DataFrame(
        {
            "date": pd.to_datetime(raw[date_column], errors="coerce").dt.date,
            "vix_close": pd.to_numeric(
                raw[value_column].replace(".", np.nan),
                errors="coerce",
            ),
        }
    )
    out = out.dropna(subset=["date"]).sort_values("date").drop_duplicates("date", keep="last")
    out["vix_close_date"] = out["date"]
    out["is_holiday_or_missing"] = out["vix_close"].isna()
    out["source_dataset"] = "fred_vixcls"
    out["source_url"] = source_url
    out["source_snapshot_date"] = source_snapshot_date.isoformat()
    out["schema_version"] = schema_version
    return out.reset_index(drop=True)


def _valid_vix_observations(vix_observations: pd.DataFrame) -> pd.DataFrame:
    if "vix_close" not in vix_observations.columns:
        raise ValueError("VIX observations require vix_close column")
    date_column = "vix_close_date" if "vix_close_date" in vix_observations.columns else "date"
    if date_column not in vix_observations.columns:
        raise ValueError("VIX observations require date or vix_close_date column")
    out = pd.DataFrame(
        {
            "vix_date": pd.to_datetime(vix_observations[date_column], errors="coerce").dt.date,
            "vix_close": pd.to_numeric(vix_observations["vix_close"], errors="coerce"),
        }
    )
    return (
        out.dropna(subset=["vix_date", "vix_close"])
        .sort_values("vix_date")
        .drop_duplicates("vix_date", keep="last")
        .reset_index(drop=True)
    )


def _regime_from_percentile(percentile: float) -> str:
    if percentile <= 1.0 / 3.0:
        return "low"
    if percentile <= 2.0 / 3.0:
        return "mid"
    return "high"


def build_vix_features(
    vix_observations: pd.DataFrame,
    feature_frame: pd.DataFrame,
    *,
    feature_asof_date_col: str = "feature_asof_date",
    timing_col: str = "announcement_timing",
    alignment: VixAlignment = VIX_ALIGNMENT_PRIOR_CLOSE,
    max_lag_days: int = 5,
    percentile_window: int = 252,
    min_regime_observations: int = 40,
) -> pd.DataFrame:
    """Resolve VIX features with no-leakage as-of alignment.

    Change features use valid-observation lags, not calendar-day lags.
    Percentile/regime cutpoints use only observations before the resolved VIX date.
    """
    if alignment not in {VIX_ALIGNMENT_PRIOR_CLOSE, VIX_ALIGNMENT_SAME_DAY_AMC}:
        raise ValueError(f"unsupported VIX alignment: {alignment}")
    if feature_asof_date_col not in feature_frame.columns:
        raise ValueError(f"feature frame requires {feature_asof_date_col}")
    if max_lag_days < 0:
        raise ValueError("max_lag_days must be non-negative")
    if percentile_window <= 0:
        raise ValueError("percentile_window must be positive")

    valid = _valid_vix_observations(vix_observations)
    dates = valid["vix_date"].tolist()
    values = valid["vix_close"].to_numpy(dtype=float)
    out = feature_frame.copy()
    rows: list[dict[str, object]] = []

    for record in out.to_dict("records"):
        feature_date = _coerce_date(record.get(feature_asof_date_col))
        row: dict[str, object] = {
            "resolved_vix_date": pd.NaT,
            "vix_lag_days": np.nan,
            "vix_level": np.nan,
            "vix_change_1d": np.nan,
            "vix_change_5d": np.nan,
            "vix_percentile_252d": np.nan,
            "vix_regime_tercile": pd.NA,
            "vix_above_30": pd.NA,
            "vix_available": False,
            "vix_alignment": alignment,
            "max_vix_lag_days": max_lag_days,
        }
        if feature_date is None or not dates:
            rows.append(row)
            continue

        timing = str(record.get(timing_col, "")).upper()
        allow_same_day = alignment == VIX_ALIGNMENT_SAME_DAY_AMC and timing == "AMC"
        position = (
            bisect_right(dates, feature_date) - 1
            if allow_same_day
            else bisect_left(dates, feature_date) - 1
        )
        if position < 0:
            rows.append(row)
            continue

        resolved_date = dates[position]
        lag_days = (feature_date - resolved_date).days
        if lag_days > max_lag_days:
            rows.append(row)
            continue

        current = float(values[position])
        row.update(
            {
                "resolved_vix_date": resolved_date,
                "vix_lag_days": lag_days,
                "vix_level": current,
                "vix_change_1d": current - float(values[position - 1]) if position >= 1 else np.nan,
                "vix_change_5d": current - float(values[position - 5]) if position >= 5 else np.nan,
                "vix_above_30": bool(current > 30.0),
                "vix_available": True,
            }
        )
        history_start = max(0, position - percentile_window)
        history = values[history_start:position]
        if len(history) >= min_regime_observations:
            percentile = float(np.mean(history <= current))
            row["vix_percentile_252d"] = percentile
            row["vix_regime_tercile"] = _regime_from_percentile(percentile)
        rows.append(row)

    features = pd.DataFrame(rows, index=out.index)
    for column in VIX_FEATURE_COLUMNS:
        out[column] = (
            features[column].astype("boolean") if column == "vix_above_30" else features[column]
        )
    return out

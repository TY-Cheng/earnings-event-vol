from __future__ import annotations

import math
from collections.abc import Iterable, Sequence
from datetime import date
from typing import cast

import numpy as np
import pandas as pd

from earnings_event_vol.events import market_close_timestamp

FEATURE_SCHEMA_V2_SEC_XBRL = "fe_v2_sec_xbrl"
DEFAULT_FEATURE_SCHEMA_VERSION = FEATURE_SCHEMA_V2_SEC_XBRL
FEATURE_SCHEMA_VERSIONS = (FEATURE_SCHEMA_V2_SEC_XBRL,)

FEATURE_SCHEMA_COLUMNS = [
    "feature_name",
    "family",
    "source",
    "as_of_rule",
    "model_feature",
    "reason",
    "feature_schema_version",
    "coverage",
]

V2_EXACT_EXCLUDE_COLUMNS = {
    "cik",
    "event_year",
    "event_month",
    "sic",
    "source_id",
    "source_timestamp",
    "feature_asof_timestamp",
    "event_entry_timestamp",
    "signal_timestamp",
    "split",
    "target_id",
    "target_label_column",
    "feature_schema_version",
    "paper_grade",
    "execution_confidence_band",
    "execution_confidence_score",
    "quote_execution_paper_grade",
    "required_quote_marks",
    "ok_quote_marks",
    "missing_quote_marks",
    "invalid_quote_marks",
    "stale_quote_marks",
    "wide_spread_marks",
    "max_quote_age_seconds",
    "median_spread_over_mid",
    "expiry_candidate_count",
    "surface_pair_count",
    "paper_grade_quote_ivar_mid",
    "paper_grade_quote_ivar_ask",
    "paper_grade_quote_iv_mid",
    "paper_grade_quote_iv_bidask",
    "paper_grade_quote_iv_surface_pair",
    "paper_grade_quote_surface_ivar_mid",
    "paper_grade_quote_surface_ivar_bid",
    "paper_grade_quote_surface_ivar_ask",
    "used_intrinsic_fallback",
    "xbrl_dropped_same_day_filed_rows",
}

V2_EXCLUDE_PATTERNS = (
    "accession",
    "primary_document",
    "sec_items",
    "source_id",
    "rvar_event",
    "rvar_",
    "r_event_",
    "edge_var_realized",
    "mispricing_realized",
    "preannouncement",
    "prior_guidance",
    "s_after",
    "close_after",
    "open_after",
    "return_decomposition",
    "cross_term",
    "gross_proxy_pnl_usd",
    "gross_exit_option_vwap",
    "gross_c2o",
    "gross_post_open",
    "gross_reaction_o2c",
    "exit_option_value",
    "exit_intrinsic",
    "exit_option_vwap",
    "post_open_option_vwap",
    "open_option_vwap",
    "option_proxy_decomposition",
    "reaction_o2c_option_vwap",
    "c2o_exit_intrinsic",
    "c2o_haircut",
    "c2o_proxy_pnl",
    "haircut_pnl_usd",
    "net_proxy_pnl_usd",
    "quote_execution",
    "quote_iv",
    "quote_ivar",
    "quote_bid_iv",
    "quote_mid_iv",
    "quote_ask_iv",
    "quote_mid_ivar",
    "quote_ask_ivar",
    "quote_surface",
    "quote_status",
    "quote_score",
    "surface_pair",
    "spread_over_mid",
    "return_on_premium_realized",
    "realized",
    "post_event",
    "future",
)


def validate_feature_schema_version(version: str) -> str:
    normalized = str(version).strip()
    if normalized not in FEATURE_SCHEMA_VERSIONS:
        raise ValueError(f"unsupported feature_schema_version: {version}")
    return normalized


def iv_butterfly_25d(*, iv_25p: float, iv_atm: float, iv_25c: float) -> float:
    return float(iv_25p - 2.0 * iv_atm + iv_25c)


def has_required_sequence_history(
    history_dates: Iterable[date],
    *,
    entry_date: date,
    required_trading_days: int = 20,
) -> bool:
    prior_dates = sorted({day for day in history_dates if day < entry_date})
    return len(prior_dates) >= required_trading_days


def sequence_eligibility_reason(
    history_dates: Iterable[date],
    *,
    entry_date: date,
    required_trading_days: int = 20,
) -> str | None:
    return (
        None
        if has_required_sequence_history(
            history_dates, entry_date=entry_date, required_trading_days=required_trading_days
        )
        else f"insufficient_{required_trading_days}_day_sequence"
    )


def universe_by_trailing_option_dollar_volume(
    rows: Sequence[dict[str, object]],
    *,
    top_n: int = 50,
) -> list[str]:
    totals: dict[str, float] = {}
    counts: dict[str, int] = {}
    for row in rows:
        ticker = str(row["ticker"])
        raw_dollar_volume = cast(float | int | str | None, row.get("option_dollar_volume", 0.0))
        dollar_volume = float(raw_dollar_volume if raw_dollar_volume is not None else 0.0)
        totals[ticker] = totals.get(ticker, 0.0) + dollar_volume
        counts[ticker] = counts.get(ticker, 0) + 1
    ranked = sorted(
        totals,
        key=lambda ticker: (totals[ticker] / max(1, counts[ticker]), ticker),
        reverse=True,
    )
    return ranked[:top_n]


def _event_date_column(frame: pd.DataFrame) -> str:
    for column in ("announcement_date", "event_date", "entry_date"):
        if column in frame.columns:
            return column
    raise ValueError("frame requires announcement_date, event_date, or entry_date")


def _numeric_or_nan(frame: pd.DataFrame, column: str) -> pd.Series:
    if column not in frame.columns:
        return pd.Series(pd.NA, index=frame.index, dtype="Float64")
    return pd.to_numeric(frame[column], errors="coerce")


def _timestamp_series(frame: pd.DataFrame, column: str) -> pd.Series:
    if column not in frame.columns:
        return pd.Series(pd.NaT, index=frame.index, dtype="datetime64[ns, UTC]")
    return pd.to_datetime(frame[column], errors="coerce", utc=True)


def _entry_timestamp_series(frame: pd.DataFrame) -> pd.Series:
    if "event_entry_timestamp" in frame.columns:
        entry_ts = pd.to_datetime(frame["event_entry_timestamp"], errors="coerce", utc=True)
    elif "entry_date" in frame.columns:
        entry_ts = pd.Series(
            [
                pd.Timestamp(market_close_timestamp(pd.Timestamp(value).date()))
                if value is not None and not pd.isna(value)
                else pd.NaT
                for value in frame["entry_date"]
            ],
            index=frame.index,
        )
        entry_ts = pd.to_datetime(entry_ts, errors="coerce", utc=True)
    else:
        raise ValueError("event panel requires event_entry_timestamp or entry_date")
    if entry_ts.isna().any():
        raise ValueError("event panel has missing event_entry_timestamp values")
    return entry_ts


def _history_stats(values: pd.Series, *, prefix: str) -> dict[str, object]:
    clean = pd.to_numeric(values, errors="coerce").dropna()
    if clean.empty:
        return {
            f"{prefix}_median": np.nan,
            f"{prefix}_std": np.nan,
            f"{prefix}_p75": np.nan,
            f"{prefix}_p90": np.nan,
            f"{prefix}_max": np.nan,
            f"{prefix}_ewm": np.nan,
        }
    weights = np.exp(np.linspace(-1.5, 0.0, len(clean)))
    return {
        f"{prefix}_median": float(clean.median()),
        f"{prefix}_std": float(clean.std(ddof=0)) if len(clean) > 1 else 0.0,
        f"{prefix}_p75": float(clean.quantile(0.75)),
        f"{prefix}_p90": float(clean.quantile(0.90)),
        f"{prefix}_max": float(clean.max()),
        f"{prefix}_ewm": float(np.average(clean.to_numpy(dtype=float), weights=weights)),
    }


def _normalization_fit_mask(split: pd.Series, fit_split: str) -> pd.Series:
    if fit_split == "train_validation":
        return split.isin(["train", "validation"])
    return split.eq(fit_split)


def add_rolling_earnings_history(frame: pd.DataFrame) -> pd.DataFrame:
    """Add point-in-time same-ticker earnings history features.

    A row may only use same-ticker rows whose event entry timestamp is strictly
    earlier than the current row's timestamp. This is intentionally slower than
    a vectorized rolling call because it makes same-day/tie handling explicit.
    """
    if "ticker" not in frame.columns:
        return frame.copy()
    out = frame.copy()
    order_ts = _timestamp_series(out, "event_entry_timestamp")
    if order_ts.isna().all():
        order_ts = pd.to_datetime(out.get("event_date"), errors="coerce", utc=True)
    out["_history_order_ts"] = order_ts
    target_sources = {
        "day_c2c": "RVAR_event_day_c2c" if "RVAR_event_day_c2c" in out else "rvar_event",
        "jump_c2o": "RVAR_event_jump_c2o",
        "reaction_o2c": "RVAR_event_reaction_o2c",
    }
    for target_name, source in target_sources.items():
        if source in out.columns:
            out[f"_hist_{target_name}_rvar"] = _numeric_or_nan(out, source)
            out[f"_hist_{target_name}_rv_iv_spread"] = out[
                f"_hist_{target_name}_rvar"
            ] - _numeric_or_nan(out, "ivar_event")
    history_rows: dict[int, dict[str, object]] = {}
    sort_columns = ["ticker", "_history_order_ts"]
    if "event_id" in out.columns:
        sort_columns.append("event_id")
    for _ticker, group in out.sort_values(sort_columns).groupby("ticker", dropna=False):
        for idx, row in group.iterrows():
            ts = row["_history_order_ts"]
            prior = group.iloc[0:0] if pd.isna(ts) else group.loc[group["_history_order_ts"] < ts]
            record: dict[str, object] = {
                "prior_earnings_count": int(len(prior)),
                "prior_earnings_observed": bool(len(prior) > 0),
            }
            if len(prior) and not pd.isna(ts):
                last_ts = prior["_history_order_ts"].max()
                record["prior_days_since_earnings"] = float(
                    (ts - last_ts).total_seconds() / 86400.0
                )
            else:
                record["prior_days_since_earnings"] = np.nan
            for target_name in target_sources:
                rvar_col = f"_hist_{target_name}_rvar"
                spread_col = f"_hist_{target_name}_rv_iv_spread"
                if rvar_col in prior:
                    record.update(
                        _history_stats(prior[rvar_col], prefix=f"prior_{target_name}_rvar")
                    )
                if spread_col in prior:
                    record.update(
                        _history_stats(
                            prior[spread_col],
                            prefix=f"prior_{target_name}_rv_iv_spread",
                        )
                    )
                    spread = pd.to_numeric(prior[spread_col], errors="coerce").dropna()
                    record[f"prior_{target_name}_rvar_gt_ivar_hit_rate"] = (
                        float((spread > 0).mean()) if len(spread) else np.nan
                    )
            history_rows[int(idx)] = record
    history = pd.DataFrame.from_dict(history_rows, orient="index")
    out = out.join(history)
    drop_cols = [column for column in out.columns if column.startswith("_hist_")]
    drop_cols.append("_history_order_ts")
    return out.drop(columns=drop_cols, errors="ignore")


def add_train_fit_normalized_features(
    frame: pd.DataFrame,
    *,
    columns: Sequence[str],
    feature_schema_version: str,
    fit_split: str = "train",
) -> tuple[pd.DataFrame, dict[str, object]]:
    """Add train-fitted z-score/rank features and return audit parameters."""
    out = frame.copy()
    validate_feature_schema_version(feature_schema_version)
    split = out["split"].astype(str) if "split" in out else pd.Series(fit_split, index=out.index)
    fit_mask = _normalization_fit_mask(split, fit_split)
    column_params: dict[str, object] = {}
    params: dict[str, object] = {
        "feature_schema_version": feature_schema_version,
        "fit_split": fit_split,
        "columns": column_params,
        "test_distribution_used": False,
    }
    for column in columns:
        if column not in out.columns or not (
            pd.api.types.is_numeric_dtype(out[column]) or pd.api.types.is_bool_dtype(out[column])
        ):
            continue
        values = pd.to_numeric(out[column], errors="coerce")
        train_values = values.loc[fit_mask & values.notna()]
        if train_values.empty:
            continue
        center = float(train_values.median())
        scale = float(train_values.std(ddof=0))
        if not np.isfinite(scale) or scale <= 1e-12:
            scale = 1.0
        sorted_train = np.sort(train_values.to_numpy(dtype=float))
        rank_bins = {str(q): float(train_values.quantile(q)) for q in (0.1, 0.25, 0.5, 0.75, 0.9)}
        z_col = f"{column}_train_z"
        rank_col = f"{column}_train_rank"
        out[z_col] = (values - center) / scale
        ranks = np.searchsorted(sorted_train, values.to_numpy(dtype=float), side="right")
        out[rank_col] = np.where(values.notna(), ranks / max(len(sorted_train), 1), np.nan)
        column_params[column] = {
            "center": center,
            "scale": scale,
            "n_fit": int(len(train_values)),
            "rank_bins": rank_bins,
            "z_feature": z_col,
            "rank_feature": rank_col,
        }
    return out, params


def normalization_params_only(
    frame: pd.DataFrame,
    *,
    columns: Sequence[str],
    feature_schema_version: str,
    fit_split: str,
) -> dict[str, object]:
    """Return normalization audit parameters without materializing features."""
    _, params = add_train_fit_normalized_features(
        frame,
        columns=columns,
        feature_schema_version=feature_schema_version,
        fit_split=fit_split,
    )
    for column_params in cast(dict[str, object], params["columns"]).values():
        if isinstance(column_params, dict):
            column_params.pop("z_feature", None)
            column_params.pop("rank_feature", None)
    return params


def _v2_model_feature(column: str, series: pd.Series) -> tuple[bool, str]:
    lower = column.lower()
    if column in V2_EXACT_EXCLUDE_COLUMNS or lower in V2_EXACT_EXCLUDE_COLUMNS:
        return False, "fe_v2_exact_exclusion"
    if lower.endswith("_cik") or lower.startswith("cik_") or lower.startswith("sec_raw_"):
        return False, "fe_v2_raw_identifier_exclusion"
    if lower.startswith("forecast_"):
        return False, "prediction_column"
    if lower.startswith("prior_"):
        if pd.api.types.is_numeric_dtype(series) or pd.api.types.is_bool_dtype(series):
            return True, "fe_v2_point_in_time_prior_history"
        return False, "non_numeric"
    if any(pattern in lower for pattern in V2_EXCLUDE_PATTERNS):
        return False, "fe_v2_leakage_or_identifier_exclusion"
    if lower.startswith("seq_t"):
        return False, "raw_pivot_sequence_column"
    if not (pd.api.types.is_numeric_dtype(series) or pd.api.types.is_bool_dtype(series)):
        return False, "non_numeric"
    return True, "fe_v2_allowlisted_numeric_or_bool"


def _feature_family(column: str) -> str:
    if column.startswith("seqagg_"):
        return "sequence_aggregate"
    if column.startswith("prior_"):
        return "rolling_earnings_history"
    if column.endswith("_train_z") or column.endswith("_train_rank"):
        return "train_fit_normalization"
    if column.startswith("xbrl_"):
        return "sec_xbrl_companyfacts"
    if column.startswith("sector_sic_"):
        return "sector_sic_coarse_control"
    if "delta_grid_proxy" in column or "rnd_proxy" in column:
        return "single_name_surface_proxy"
    if "runup" in column or "underlying_pre_event_" in column or "delta_" in column:
        return "single_name_runup_proxy"
    if column.startswith(("vix_", "spy_", "qqq_", "market_second_")):
        return "market_covariate"
    if "premium" in column or "cost" in column or "volume" in column or "transactions" in column:
        return "liquidity_execution_proxy"
    return "event_level"


def build_feature_schema_report(
    frame: pd.DataFrame,
    *,
    feature_schema_version: str = DEFAULT_FEATURE_SCHEMA_VERSION,
) -> pd.DataFrame:
    version = validate_feature_schema_version(feature_schema_version)
    rows: list[dict[str, object]] = []
    for column in frame.columns:
        model_feature, reason = _v2_model_feature(column, frame[column])
        coverage = float(frame[column].notna().mean()) if len(frame) else 0.0
        rows.append(
            {
                "feature_name": column,
                "family": _feature_family(column),
                "source": "generated_research_feature_matrix",
                "as_of_rule": "signal_timestamp_event_entry"
                if model_feature
                else "not_model_feature",
                "model_feature": bool(model_feature),
                "reason": reason,
                "feature_schema_version": version,
                "coverage": coverage,
            }
        )
    return pd.DataFrame(rows, columns=FEATURE_SCHEMA_COLUMNS)


def feature_columns_from_schema_report(
    schema_report: pd.DataFrame,
    *,
    frame: pd.DataFrame | None = None,
    include_sequence_aggregates: bool = True,
) -> list[str]:
    required = {"feature_name", "model_feature"}
    missing = required - set(schema_report.columns)
    if missing:
        raise ValueError(f"feature schema report missing columns: {sorted(missing)}")
    selected = schema_report.loc[schema_report["model_feature"].astype(bool)].copy()
    if not include_sequence_aggregates:
        selected = selected.loc[~selected["feature_name"].astype(str).str.startswith("seqagg_")]
    columns = [str(value) for value in selected["feature_name"].tolist()]
    if frame is not None:
        columns = [column for column in columns if column in frame.columns]
    return columns


def build_model_feature_matrix(
    event_panel: pd.DataFrame,
    *,
    straddle_diagnostics: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Build the leakage-aware event-level modeling table.

    The target is `rvar_event`. The market baseline is `ivar_event`. Realized
    mispricing is retained only as a label/evaluation column, not a feature.
    """
    required = {"ticker", "rvar_event", "ivar_event"}
    missing = sorted(required - set(event_panel.columns))
    if missing:
        raise ValueError(f"event panel missing required columns: {missing}")
    out = event_panel.copy()
    date_col = _event_date_column(out)
    out["event_date"] = pd.to_datetime(out[date_col], errors="coerce").dt.date
    out["ticker"] = out["ticker"].astype(str).str.upper()
    if "RVAR_event_day_c2c" in out.columns:
        out["rvar_event"] = _numeric_or_nan(out, "RVAR_event_day_c2c")
    else:
        out["rvar_event"] = _numeric_or_nan(out, "rvar_event")
    out["ivar_event"] = _numeric_or_nan(out, "ivar_event")
    out["edge_var_realized"] = out["rvar_event"] - out["ivar_event"]
    if "RVAR_event_jump_c2o" in out.columns:
        out["edge_var_realized_jump_c2o"] = (
            _numeric_or_nan(out, "RVAR_event_jump_c2o") - out["ivar_event"]
        )
    if "RVAR_event_day_c2c" in out.columns:
        out["edge_var_realized_day_c2c"] = (
            _numeric_or_nan(out, "RVAR_event_day_c2c") - out["ivar_event"]
        )
    if "RVAR_event_reaction_o2c" in out.columns:
        out["edge_var_realized_reaction_o2c"] = (
            _numeric_or_nan(out, "RVAR_event_reaction_o2c") - out["ivar_event"]
        )
    if "dte_1" in out.columns:
        dte = pd.to_numeric(out["dte_1"], errors="coerce")
    elif "dte" in out.columns:
        dte = pd.to_numeric(out["dte"], errors="coerce")
    else:
        dte = pd.Series(pd.NA, index=out.index, dtype="Float64")
    out["is_main_dte_5_14"] = dte.between(5, 14, inclusive="both")
    out["is_robustness_dte_3_21"] = dte.between(3, 21, inclusive="both")
    out["dte_bucket"] = pd.cut(
        dte,
        bins=[-float("inf"), 4, 14, 21, float("inf")],
        labels=["lt_5", "main_5_14", "robust_15_21", "ivar_support_gt_21"],
    ).astype(str)
    if "announcement_timing" in out.columns:
        out["is_bmo"] = out["announcement_timing"].astype(str).eq("BMO")
        out["is_amc"] = out["announcement_timing"].astype(str).eq("AMC")
    if "universe_rank" in out.columns:
        out["universe_rank"] = pd.to_numeric(out["universe_rank"], errors="coerce")
        out["liquidity_rank_score"] = 1.0 / out["universe_rank"].clip(lower=1)
    if "entry_date" in out.columns:
        entry = pd.to_datetime(out["entry_date"], errors="coerce")
        out["event_year"] = entry.dt.year
        out["event_month"] = entry.dt.month
        month = pd.to_numeric(out["event_month"], errors="coerce")
        month_angle = 2.0 * math.pi * (month - 1.0) / 12.0
        out["event_month_sin"] = np.sin(month_angle)
        out["event_month_cos"] = np.cos(month_angle)
        out["regime"] = out["event_year"].map(
            lambda year: "covid_shock" if year == 2020 else "steady_proxy"
        )
    event_entry_timestamp = _entry_timestamp_series(out)
    out["event_entry_timestamp"] = event_entry_timestamp
    out["signal_timestamp"] = event_entry_timestamp
    out["feature_asof_timestamp"] = event_entry_timestamp
    if straddle_diagnostics is not None and not straddle_diagnostics.empty:
        keep = [
            column
            for column in [
                "event_id",
                "entry_premium_usd",
                "entry_price_method",
                "exit_option_value_usd",
                "exit_option_vwap_preclose_15m_value_usd",
                "gross_exit_option_vwap_preclose_15m_proxy_pnl_usd",
                "exit_option_vwap_preclose_15m_haircut_pnl_usd",
                "exit_option_vwap_preclose_15m_status",
                "exit_option_vwap_preclose_15m_source",
                "exit_option_vwap_preclose_15m_volume",
                "exit_option_vwap_preclose_15m_transactions",
                "exit_option_vwap_preclose_15m_rows",
                "exit_intrinsic_usd",
                "open_after",
                "c2o_exit_intrinsic_usd",
                "gross_c2o_intrinsic_proxy_pnl_usd",
                "c2o_haircut_pnl_usd",
                "c2o_proxy_pnl_source",
                "c2o_proxy_pnl_status",
                "post_open_option_vwap_0_5_exit_usd",
                "gross_post_open_option_vwap_0_5_proxy_pnl_usd",
                "post_open_option_vwap_0_5_haircut_pnl_usd",
                "post_open_option_vwap_0_5_status",
                "post_open_option_vwap_0_5_source",
                "post_open_option_vwap_0_5_volume",
                "post_open_option_vwap_0_5_transactions",
                "post_open_option_vwap_0_5_rows",
                "open_option_vwap_0_5_anchor_usd",
                "gross_reaction_o2c_option_vwap_0_5_to_c2c_exit_proxy_pnl_usd",
                "reaction_o2c_option_vwap_0_5_haircut_pnl_usd",
                "reaction_o2c_option_vwap_0_5_status",
                "reaction_o2c_option_vwap_0_5_source",
                "option_proxy_decomposition_residual_0_5_usd",
                "post_open_option_vwap_5_15_exit_usd",
                "gross_post_open_option_vwap_5_15_proxy_pnl_usd",
                "post_open_option_vwap_5_15_haircut_pnl_usd",
                "post_open_option_vwap_5_15_status",
                "post_open_option_vwap_5_15_source",
                "post_open_option_vwap_5_15_volume",
                "post_open_option_vwap_5_15_transactions",
                "post_open_option_vwap_5_15_rows",
                "open_option_vwap_5_15_anchor_usd",
                "gross_reaction_o2c_option_vwap_5_15_to_c2c_exit_proxy_pnl_usd",
                "reaction_o2c_option_vwap_5_15_haircut_pnl_usd",
                "reaction_o2c_option_vwap_5_15_status",
                "reaction_o2c_option_vwap_5_15_source",
                "option_proxy_decomposition_residual_5_15_usd",
                "gross_proxy_pnl_usd",
                "haircut_pnl_usd",
                "proxy_volume_window",
                "proxy_transactions_window",
                "option_exit_price_status",
                "used_intrinsic_fallback",
            ]
            if column in straddle_diagnostics.columns
        ]
        if "event_id" in out.columns and "event_id" in keep:
            diagnostics = straddle_diagnostics[keep].copy()
            duplicated = diagnostics["event_id"].duplicated(keep=False)
            if duplicated.any():
                duplicate_ids = sorted(diagnostics.loc[duplicated, "event_id"].astype(str).unique())
                raise ValueError(
                    "straddle diagnostics must contain at most one row per event_id: "
                    + ", ".join(duplicate_ids[:5])
                )
            out = out.merge(diagnostics, on="event_id", how="left", validate="one_to_one")
            out["net_proxy_pnl_usd"] = _numeric_or_nan(out, "haircut_pnl_usd")
            out["estimated_transaction_cost_usd"] = (
                _numeric_or_nan(out, "gross_proxy_pnl_usd")
                - _numeric_or_nan(out, "haircut_pnl_usd")
            ).clip(lower=0)
            out["return_on_premium_realized"] = _numeric_or_nan(
                out, "net_proxy_pnl_usd"
            ) / _numeric_or_nan(out, "entry_premium_usd").replace(0, pd.NA)
    return add_rolling_earnings_history(out)


def build_option_surface_sequence_matrix(
    rows: pd.DataFrame,
    *,
    event_id_col: str = "event_id",
    day_index_col: str = "day_index",
    value_columns: Sequence[str] = ("atm_iv", "option_volume", "spread_over_mid"),
    lookback_days: int = 20,
) -> pd.DataFrame:
    """Pivot long pre-event surface rows into `seq_tXX_feature` columns."""
    required = {event_id_col, day_index_col, *value_columns}
    missing = sorted(required - set(rows.columns))
    if missing:
        raise ValueError(f"sequence rows missing required columns: {missing}")
    frame = rows.copy()
    frame[day_index_col] = pd.to_numeric(frame[day_index_col], errors="coerce").astype("Int64")
    frame = frame.loc[frame[day_index_col].between(-lookback_days, -1, inclusive="both")].copy()
    if frame.empty:
        return pd.DataFrame(columns=[event_id_col])
    out = pd.DataFrame({event_id_col: sorted(frame[event_id_col].dropna().unique())})
    for offset in range(-lookback_days, 0):
        day = frame.loc[frame[day_index_col].eq(offset)]
        for value_col in value_columns:
            series = (
                day.groupby(event_id_col)[value_col]
                .mean()
                .rename(f"seq_t{offset + lookback_days:02d}_{value_col}")
                .reset_index()
            )
            out = out.merge(series, on=event_id_col, how="left")
    return out

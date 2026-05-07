from __future__ import annotations

from collections.abc import Iterable, Sequence
from datetime import date
from typing import cast

import pandas as pd


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
        out["regime"] = out["event_year"].map(
            lambda year: "covid_shock" if year == 2020 else "steady_proxy"
        )
    if straddle_diagnostics is not None and not straddle_diagnostics.empty:
        keep = [
            column
            for column in [
                "event_id",
                "entry_premium_usd",
                "exit_option_value_usd",
                "exit_intrinsic_usd",
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
            out = out.merge(straddle_diagnostics[keep], on="event_id", how="left")
            out["net_proxy_pnl_usd"] = _numeric_or_nan(out, "haircut_pnl_usd")
            out["estimated_transaction_cost_usd"] = (
                _numeric_or_nan(out, "gross_proxy_pnl_usd")
                - _numeric_or_nan(out, "haircut_pnl_usd")
            ).clip(lower=0)
            out["return_on_premium_realized"] = _numeric_or_nan(
                out, "net_proxy_pnl_usd"
            ) / _numeric_or_nan(out, "entry_premium_usd").replace(0, pd.NA)
    return out


def build_option_surface_sequence_matrix(
    rows: pd.DataFrame,
    *,
    event_id_col: str = "event_id",
    day_index_col: str = "day_index",
    value_columns: Sequence[str] = ("atm_iv", "option_volume", "spread_over_mid"),
    lookback_days: int = 20,
) -> pd.DataFrame:
    """Pivot long pre-event surface rows into `seq_tXX_feature` columns for Mamba."""
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

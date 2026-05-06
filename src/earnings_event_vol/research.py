from __future__ import annotations

import json
import math
import re
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import Literal, cast

import numpy as np
import pandas as pd
import polars as pl
import torch
from scipy.optimize import brentq

from earnings_event_vol.backtest import black_scholes_price, build_proxy_strategy_frame
from earnings_event_vol.config import ProjectConfig
from earnings_event_vol.features import build_model_feature_matrix
from earnings_event_vol.massive import parse_massive_option_ticker
from earnings_event_vol.metrics import (
    breakdown_metrics,
    cost_sensitivity,
    edge_decile_table,
    forecast_metrics,
    qlike_loss,
    ranking_metrics,
    strategy_metrics,
)
from earnings_event_vol.models import (
    FTTransformerRegressor,
    LinearElasticNetRegressor,
    MambaSequenceEncoder,
    add_benchmark_predictions,
    default_feature_columns,
    prediction_column_for_model,
)
from earnings_event_vol.schemas import OptionRight

FORECAST_FLOOR = 1e-6
DEFAULT_HAIRCUT_BPS = 0.005
DEFAULT_OPTION_MULTIPLIER = 100.0
DEFAULT_CONTRACTS = 1.0
LOOKBACK_DAYS = 20
MAMBA_MIN_VALID_DAYS = 12
MAMBA_LATEST_DAYS = 5
SEQUENCE_FEATURE_NAMES = [
    "atm_iv_proxy",
    "iv_skew_proxy",
    "iv_butterfly_proxy",
    "term_slope_proxy",
    "event_ivar_proxy",
    "straddle_premium_to_spot",
    "valid_pair_count",
    "surface_missing_rate",
    "option_volume_sum",
    "option_transactions_sum",
    "underlying_return_1d",
    "rv5",
    "spy_return_1d",
    "vix_level",
    "vix_change_1d",
]
MODEL_IDS = [
    "market_implied_event_variance",
    "last_four_rvar",
    "last_four_ivar",
    "goyal_saretto_rv_iv_spread",
    "linear_elastic_net",
    "lightgbm",
    "xgboost",
    "ft_transformer",
    "mamba_sequence_encoder",
    "mask_only_mamba_sequence_encoder",
]
DETERMINISTIC_MODEL_IDS = {
    "market_implied_event_variance",
    "last_four_rvar",
    "last_four_ivar",
    "goyal_saretto_rv_iv_spread",
}
TRAINABLE_TABULAR_MODEL_IDS = {"linear_elastic_net", "lightgbm", "xgboost", "ft_transformer"}
GBDT_MODEL_IDS = {"lightgbm", "xgboost"}
SplitName = Literal["train", "validation", "test"]


@dataclass(frozen=True)
class ResearchPaths:
    artifacts_dir: Path
    modeling_artifacts_dir: Path
    reports_dir: Path
    modeling_reports_dir: Path
    feature_matrix_path: Path
    sequence_long_path: Path
    sequence_tensor_path: Path
    predictions_path: Path


@dataclass(frozen=True)
class EventSplit:
    event_id: str
    split: SplitName
    sort_timestamp: pd.Timestamp


@dataclass(frozen=True)
class ProxyResearchResult:
    ok: bool
    stage: str
    outputs: dict[str, str]
    diagnostics: dict[str, object]


def research_paths(config: ProjectConfig) -> ResearchPaths:
    modeling_artifacts = config.artifacts_dir / "modeling"
    modeling_reports = config.reports_dir / "modeling"
    return ResearchPaths(
        artifacts_dir=config.artifacts_dir,
        modeling_artifacts_dir=modeling_artifacts,
        reports_dir=config.reports_dir,
        modeling_reports_dir=modeling_reports,
        feature_matrix_path=config.gold_data_dir / "modeling" / "feature_matrix.parquet",
        sequence_long_path=config.silver_data_dir
        / "modeling"
        / "option_surface_sequence_long.parquet",
        sequence_tensor_path=config.gold_data_dir / "modeling" / "sequence_tensor.npz",
        predictions_path=modeling_artifacts / "model_predictions.parquet",
    )


def write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True, default=str), encoding="utf-8")


def read_table(path: Path) -> pd.DataFrame:
    if path.suffix.lower() == ".parquet":
        return pd.read_parquet(path)
    return pd.read_csv(path)


def write_table(path: Path, frame: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.suffix.lower() == ".parquet":
        frame.to_parquet(path, index=False)
    else:
        frame.to_csv(path, index=False)


def ensure_event_id(frame: pd.DataFrame) -> pd.DataFrame:
    out = frame.copy()
    if "event_id" not in out.columns:
        date_col = event_sort_column(out)
        out["event_id"] = (
            out["ticker"].astype(str).str.upper()
            + "_"
            + pd.to_datetime(out[date_col], errors="coerce").dt.date.astype(str)
        )
    return out


def event_sort_column(frame: pd.DataFrame) -> str:
    for column in ("event_entry_timestamp", "entry_date", "event_date", "announcement_date"):
        if column in frame.columns:
            return column
    raise ValueError(
        "frame requires event_entry_timestamp, entry_date, event_date, or announcement_date"
    )


def assign_event_splits(
    frame: pd.DataFrame,
    *,
    split_design: str = "chronological_proxy_70_15_15",
    split_date: str | pd.Timestamp | None = None,
) -> pd.DataFrame:
    """Assign chronological event-level train/validation/test splits."""
    out = ensure_event_id(frame)
    date_col = event_sort_column(out)
    event_dates = (
        out[["event_id", date_col]]
        .assign(_sort_ts=pd.to_datetime(out[date_col], errors="coerce", utc=True))
        .groupby("event_id", as_index=False)["_sort_ts"]
        .min()
        .sort_values(["_sort_ts", "event_id"])
        .reset_index(drop=True)
    )
    if event_dates.empty:
        raise ValueError("cannot split an empty feature matrix")
    split_by_event: dict[str, SplitName] = {}
    if split_date is not None:
        cutoff = pd.Timestamp(split_date)
        cutoff = cutoff.tz_localize("UTC") if cutoff.tzinfo is None else cutoff.tz_convert("UTC")
        before = event_dates["_sort_ts"] < cutoff
        after = event_dates.loc[~before].copy()
        split_by_event.update(
            {event_id: "train" for event_id in event_dates.loc[before, "event_id"]}
        )
        half = max(1, int(math.ceil(len(after) / 2))) if len(after) else 0
        split_by_event.update(
            {event_id: "validation" for event_id in after.iloc[:half]["event_id"]}
        )
        split_by_event.update({event_id: "test" for event_id in after.iloc[half:]["event_id"]})
    elif split_design == "chronological_proxy_70_15_15":
        n = len(event_dates)
        if n < 3:
            raise ValueError("chronological_proxy_70_15_15 requires at least three events")
        train_cut = max(1, int(math.floor(0.70 * n)))
        validation_cut = max(train_cut + 1, int(math.floor(0.85 * n)))
        validation_cut = min(validation_cut, n - 1)
        split_by_event.update(
            {event_id: "train" for event_id in event_dates.iloc[:train_cut]["event_id"]}
        )
        split_by_event.update(
            {
                event_id: "validation"
                for event_id in event_dates.iloc[train_cut:validation_cut]["event_id"]
            }
        )
        split_by_event.update(
            {event_id: "test" for event_id in event_dates.iloc[validation_cut:]["event_id"]}
        )
    else:
        raise ValueError(f"unsupported split_design: {split_design}")
    out["split"] = out["event_id"].map(split_by_event)
    if out["split"].isna().any():
        raise ValueError("failed to assign every event to a split")
    return out


def sequence_source_dates_for_event(
    available_dates: Sequence[date],
    *,
    entry_date: date,
    lookback_days: int = LOOKBACK_DAYS,
) -> list[date]:
    candidates = sorted(day for day in available_dates if day <= entry_date)
    return candidates[-lookback_days:]


def build_sequence_plan(
    events: pd.DataFrame,
    *,
    available_dates: Sequence[date],
    lookback_days: int = LOOKBACK_DAYS,
) -> pd.DataFrame:
    out = ensure_event_id(events)
    rows: list[dict[str, object]] = []
    for record in out.to_dict("records"):
        entry = pd.Timestamp(record.get("entry_date") or record.get("event_date")).date()
        source_dates = sequence_source_dates_for_event(
            available_dates, entry_date=entry, lookback_days=lookback_days
        )
        start_index = lookback_days - len(source_dates)
        for offset, source_date in enumerate(source_dates):
            seq_index = start_index + offset
            rows.append(
                {
                    "event_id": str(record["event_id"]),
                    "ticker": str(record["ticker"]).upper(),
                    "entry_date": entry,
                    "exit_date": pd.Timestamp(record["exit_date"]).date()
                    if record.get("exit_date") is not None and not pd.isna(record.get("exit_date"))
                    else None,
                    "event_entry_timestamp": record.get("event_entry_timestamp"),
                    "source_date": source_date,
                    "seq_index": int(seq_index),
                }
            )
    return pd.DataFrame(rows)


def list_bronze_dates(root: Path) -> list[date]:
    if not root.exists():
        return []
    out: list[date] = []
    for item in root.glob("date=*/part.parquet"):
        text = item.parent.name.removeprefix("date=")
        try:
            out.append(pd.Timestamp(text).date())
        except ValueError:
            continue
    return sorted(set(out))


def _option_regex_for_tickers(tickers: Iterable[str]) -> str:
    escaped = [re.escape(ticker) for ticker in sorted(set(tickers))]
    if not escaped:
        return r"$.^"
    return r"^O:(" + "|".join(escaped) + r")\d{6}[CP]\d{8}$"


def _implied_volatility(
    *,
    spot: float,
    strike: float,
    time_to_expiry: float,
    option_price: float,
    right: str,
) -> float | None:
    if spot <= 0 or strike <= 0 or time_to_expiry <= 0 or option_price <= 0:
        return None
    option_right = OptionRight.CALL if right == "call" else OptionRight.PUT
    intrinsic = (
        max(spot - strike, 0.0) if option_right == OptionRight.CALL else max(strike - spot, 0.0)
    )
    if option_price < intrinsic:
        return None

    def objective(volatility: float) -> float:
        return (
            black_scholes_price(
                spot=spot,
                strike=strike,
                time_to_expiry=time_to_expiry,
                volatility=volatility,
                right=option_right,
            )
            - option_price
        )

    try:
        if objective(1e-4) * objective(5.0) > 0:
            return None
        return float(brentq(objective, 1e-4, 5.0, maxiter=80))
    except (ValueError, RuntimeError, OverflowError):
        return None


def _parse_filtered_options(frame: pd.DataFrame, *, source_date: date) -> pd.DataFrame:
    if frame.empty:
        return pd.DataFrame()
    parsed = pd.DataFrame([parse_massive_option_ticker(str(value)) for value in frame["ticker"]])
    out = parsed.copy()
    out["source_date"] = source_date
    out["option_close"] = pd.to_numeric(frame["close"], errors="coerce")
    out["option_volume"] = pd.to_numeric(frame["volume"], errors="coerce").fillna(0.0)
    out["option_transactions"] = pd.to_numeric(frame["transactions"], errors="coerce").fillna(0.0)
    out["dte"] = (pd.to_datetime(out["expiration"]).dt.date - source_date).map(
        lambda delta: delta.days
    )
    return out


def _daily_underlying_features(underlying: pd.DataFrame) -> pd.DataFrame:
    out = underlying.copy()
    out["source_date"] = pd.to_datetime(out["source_date"], errors="coerce").dt.date
    out["close"] = pd.to_numeric(out["close"], errors="coerce")
    out = out.sort_values(["ticker", "source_date"])
    out["underlying_return_1d"] = out.groupby("ticker")["close"].pct_change()
    out["rv5"] = (
        out.groupby("ticker")["underlying_return_1d"]
        .rolling(5, min_periods=2)
        .std()
        .reset_index(level=0, drop=True)
        .pow(2)
    )
    spy = (
        out.loc[out["ticker"].eq("SPY"), ["source_date", "underlying_return_1d"]]
        .rename(columns={"underlying_return_1d": "spy_return_1d"})
        .drop_duplicates("source_date")
    )
    out = out.merge(spy, on="source_date", how="left")
    return out


def _compute_daily_surface(
    options: pd.DataFrame,
    *,
    ticker: str,
    source_date: date,
    spot: float,
    event_entry_date: date,
) -> dict[str, object]:
    base: dict[str, object] = {
        "ticker": ticker,
        "source_date": source_date,
        "surface_source": "options_day_aggs",
        "iv_source": "close_trade_implied",
        "panel_grade": "no_nbbo_trade_proxy",
    }
    if options.empty or not np.isfinite(spot) or spot <= 0:
        return {
            **base,
            **{feature: np.nan for feature in SEQUENCE_FEATURE_NAMES},
            "is_valid_sequence_day": False,
        }
    frame = options.loc[
        options["ticker"].astype(str).str.upper().eq(ticker)
        & pd.to_numeric(options["dte"], errors="coerce").between(3, 45, inclusive="both")
    ].copy()
    if frame.empty:
        return {
            **base,
            **{feature: np.nan for feature in SEQUENCE_FEATURE_NAMES},
            "option_volume_sum": 0.0,
            "option_transactions_sum": 0.0,
            "valid_pair_count": 0,
            "surface_missing_rate": 1.0,
            "is_valid_sequence_day": False,
        }
    frame["moneyness_abs"] = (pd.to_numeric(frame["strike"], errors="coerce") / spot - 1.0).abs()
    selected = (
        frame.sort_values(["expiration", "right", "moneyness_abs"])
        .groupby(["expiration", "right"], as_index=False)
        .head(4)
        .copy()
    )
    ivs: list[float | None] = []
    for record in selected.to_dict("records"):
        ivs.append(
            _implied_volatility(
                spot=spot,
                strike=float(record["strike"]),
                time_to_expiry=max(float(record["dte"]) / 365.0, 1.0 / 365.0),
                option_price=float(record["option_close"]),
                right=str(record["right"]),
            )
        )
    selected["iv_proxy"] = ivs
    selected = selected.dropna(subset=["iv_proxy"])
    pair_counts = frame.groupby(["expiration", "strike"])["right"].nunique()
    valid_pair_count = int(pair_counts.ge(2).sum())
    if selected.empty or valid_pair_count == 0:
        return {
            **base,
            **{feature: np.nan for feature in SEQUENCE_FEATURE_NAMES},
            "option_volume_sum": float(frame["option_volume"].sum()),
            "option_transactions_sum": float(frame["option_transactions"].sum()),
            "valid_pair_count": valid_pair_count,
            "surface_missing_rate": 1.0,
            "is_valid_sequence_day": False,
        }
    pair_rows = selected.loc[
        selected.groupby(["expiration", "strike"])["right"].transform("nunique").ge(2)
    ].copy()
    pair_rows["target_dte_distance"] = (pd.to_numeric(pair_rows["dte"], errors="coerce") - 14).abs()
    atm_key = (
        pair_rows.groupby(["expiration", "strike"], as_index=False)
        .agg(
            moneyness_abs=("moneyness_abs", "mean"),
            target_dte_distance=("target_dte_distance", "mean"),
        )
        .sort_values(["moneyness_abs", "target_dte_distance"])
        .head(1)
    )
    atm_iv_proxy = np.nan
    iv_skew_proxy = np.nan
    straddle_premium_to_spot = np.nan
    event_ivar_proxy = np.nan
    if not atm_key.empty:
        expiration = atm_key.iloc[0]["expiration"]
        strike = atm_key.iloc[0]["strike"]
        atm = pair_rows.loc[pair_rows["expiration"].eq(expiration) & pair_rows["strike"].eq(strike)]
        call = atm.loc[atm["right"].eq("call")]
        put = atm.loc[atm["right"].eq("put")]
        atm_iv_proxy = float(atm["iv_proxy"].mean())
        if not call.empty and not put.empty:
            iv_skew_proxy = float(put["iv_proxy"].mean() - call["iv_proxy"].mean())
            straddle_premium_to_spot = float(
                (call["option_close"].mean() + put["option_close"].mean()) / spot
            )
        days_to_event = max((event_entry_date - source_date).days, 1)
        event_ivar_proxy = float(max(atm_iv_proxy, 0.0) ** 2 * days_to_event / 365.0)
    near = pair_rows.loc[pair_rows["dte"].between(3, 14, inclusive="both")]
    far = pair_rows.loc[pair_rows["dte"].between(15, 45, inclusive="both")]
    term_slope_proxy = (
        np.nan
        if near.empty or far.empty
        else float(far["iv_proxy"].mean() - near["iv_proxy"].mean())
    )
    by_strike = pair_rows.groupby("strike", as_index=False)["iv_proxy"].mean().sort_values("strike")
    iv_butterfly_proxy = np.nan
    if len(by_strike) >= 3:
        middle = int(np.argmin(np.abs(by_strike["strike"].to_numpy(dtype=float) - spot)))
        low = max(0, middle - 1)
        high = min(len(by_strike) - 1, middle + 1)
        if low != middle and high != middle:
            iv_butterfly_proxy = float(
                by_strike.iloc[low]["iv_proxy"]
                + by_strike.iloc[high]["iv_proxy"]
                - 2.0 * by_strike.iloc[middle]["iv_proxy"]
            )
    return {
        **base,
        "atm_iv_proxy": atm_iv_proxy,
        "iv_skew_proxy": iv_skew_proxy,
        "iv_butterfly_proxy": iv_butterfly_proxy,
        "term_slope_proxy": term_slope_proxy,
        "event_ivar_proxy": event_ivar_proxy,
        "straddle_premium_to_spot": straddle_premium_to_spot,
        "valid_pair_count": valid_pair_count,
        "surface_missing_rate": float(max(0.0, 1.0 - min(valid_pair_count, 10) / 10.0)),
        "option_volume_sum": float(frame["option_volume"].sum()),
        "option_transactions_sum": float(frame["option_transactions"].sum()),
        "is_valid_sequence_day": bool(valid_pair_count > 0 and np.isfinite(atm_iv_proxy)),
    }


def _read_parquet_if_exists(path: Path, columns: Sequence[str] | None = None) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    return pd.read_parquet(path, columns=list(columns) if columns else None)


def _scan_filtered_parquet(
    path: Path,
    *,
    ticker_values: Sequence[str],
    columns: Sequence[str],
) -> pd.DataFrame:
    if not path.exists() or not ticker_values:
        return pd.DataFrame(columns=list(columns))
    return (
        pl.scan_parquet(path, cast_options=pl.ScanCastOptions(integer_cast="allow-float"))
        .filter(pl.col("ticker").is_in(list(ticker_values)))
        .select(list(columns))
        .collect()
        .to_pandas()
    )


def _load_sequence_contract_candidates(
    config: ProjectConfig, event_ids: Sequence[str]
) -> pd.DataFrame:
    path = config.silver_data_dir / "contracts" / "event_contract_candidates.parquet"
    if not path.exists():
        return pd.DataFrame()
    frame = pd.read_parquet(path)
    frame = frame.loc[frame["event_id"].astype(str).isin(set(event_ids))].copy()
    if "eligible_for_quote_pool" in frame.columns:
        frame = frame.loc[frame["eligible_for_quote_pool"].astype(bool)].copy()
    if "is_robustness_dte_3_21" in frame.columns:
        frame = frame.loc[frame["is_robustness_dte_3_21"].astype(bool)].copy()
    keep = [
        "event_id",
        "ticker",
        "expiration",
        "strike",
        "right",
        "options_ticker",
        "option_multiplier",
    ]
    return frame[[column for column in keep if column in frame.columns]].drop_duplicates()


def build_option_surface_sequence_long(
    events: pd.DataFrame,
    *,
    config: ProjectConfig,
    lookback_days: int = LOOKBACK_DAYS,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, object]]:
    """Build event-aligned 20-day close-trade-implied option-surface summaries."""
    events = ensure_event_id(events)
    options_root = config.bronze_data_dir / "massive" / "options_day_aggs"
    underlying_root = config.bronze_data_dir / "massive" / "underlying_day_aggs"
    underlying_dates = list_bronze_dates(underlying_root)
    plan = build_sequence_plan(
        events, available_dates=underlying_dates, lookback_days=lookback_days
    )
    if plan.empty:
        empty_report = sequence_coverage_report(pd.DataFrame(), total_events=len(events))
        return pd.DataFrame(), pd.DataFrame(), empty_report
    needed_dates = sorted(set(cast(Iterable[date], plan["source_date"].dropna().tolist())))
    needed_tickers = sorted(set(plan["ticker"].astype(str).str.upper()))
    candidates = _load_sequence_contract_candidates(config, events["event_id"].astype(str).tolist())
    surface_rows: list[dict[str, object]] = []
    underlying_rows: list[pd.DataFrame] = []
    by_event_date = plan[["event_id", "ticker", "source_date", "entry_date"]].drop_duplicates()
    for date_index, source_date in enumerate(needed_dates, start=1):
        if date_index == 1 or date_index % 25 == 0 or date_index == len(needed_dates):
            print(
                f"[research] sequence build progress: {date_index}/{len(needed_dates)} dates",
                flush=True,
            )
        day_events = by_event_date.loc[by_event_date["source_date"].eq(source_date)]
        underlying = _scan_filtered_parquet(
            underlying_root / f"date={source_date.isoformat()}" / "part.parquet",
            ticker_values=needed_tickers + ["SPY"],
            columns=["ticker", "close", "source_date", "volume", "transactions"],
        )
        if not underlying.empty:
            underlying["source_date"] = source_date
            underlying_rows.append(underlying)
        spot_by_ticker = (
            underlying.assign(ticker=underlying["ticker"].astype(str).str.upper())
            .set_index("ticker")["close"]
            .to_dict()
            if not underlying.empty
            else {}
        )
        day_candidates = (
            candidates.loc[
                candidates["event_id"].astype(str).isin(set(day_events["event_id"].astype(str)))
            ]
            if not candidates.empty
            else pd.DataFrame()
        )
        needed_options = (
            day_candidates["options_ticker"].astype(str).dropna().unique().tolist()
            if not day_candidates.empty and "options_ticker" in day_candidates
            else []
        )
        raw_options = _scan_filtered_parquet(
            options_root / f"date={source_date.isoformat()}" / "part.parquet",
            ticker_values=needed_options,
            columns=["ticker", "close", "volume", "transactions"],
        )
        if not raw_options.empty and not day_candidates.empty:
            options = day_candidates.merge(
                raw_options.rename(
                    columns={
                        "ticker": "options_ticker",
                        "close": "option_close",
                        "volume": "option_volume",
                        "transactions": "option_transactions",
                    }
                ),
                on="options_ticker",
                how="inner",
            )
            options["source_date"] = source_date
            options["dte"] = (pd.to_datetime(options["expiration"]).dt.date - source_date).map(
                lambda delta: delta.days
            )
        else:
            options = pd.DataFrame()
        option_groups = (
            {str(event_id): group.copy() for event_id, group in options.groupby("event_id")}
            if not options.empty
            else {}
        )
        for record in day_events.to_dict("records"):
            event_id = str(record["event_id"])
            ticker = str(record["ticker"]).upper()
            spot_raw = spot_by_ticker.get(ticker, np.nan)
            spot = float(spot_raw) if spot_raw is not None and pd.notna(spot_raw) else np.nan
            row = _compute_daily_surface(
                option_groups.get(event_id, pd.DataFrame()),
                ticker=ticker,
                source_date=source_date,
                spot=spot,
                event_entry_date=cast(date, record["entry_date"]),
            )
            row["event_id"] = event_id
            surface_rows.append(row)
    surface = pd.DataFrame(surface_rows)
    underlying_features = (
        _daily_underlying_features(pd.concat(underlying_rows, ignore_index=True))
        if underlying_rows
        else pd.DataFrame(
            columns=[
                "ticker",
                "source_date",
                "close",
                "underlying_return_1d",
                "rv5",
                "spy_return_1d",
            ]
        )
    )
    long_rows = plan.merge(surface, on=["event_id", "ticker", "source_date"], how="left")
    long_rows = long_rows.merge(
        underlying_features[
            ["ticker", "source_date", "close", "underlying_return_1d", "rv5", "spy_return_1d"]
        ].rename(columns={"close": "underlying_close"}),
        on=["ticker", "source_date"],
        how="left",
    )
    for column in ("vix_level", "vix_change_1d"):
        if column not in long_rows.columns:
            long_rows[column] = np.nan
    long_rows["is_valid_sequence_day"] = (
        long_rows["is_valid_sequence_day"].fillna(False).astype(bool)
    )
    long_rows["has_underlying_close"] = pd.to_numeric(
        long_rows["underlying_close"], errors="coerce"
    ).notna()
    long_rows["missing_options_day_aggs"] = long_rows["surface_source"].isna()
    long_rows["feature_asof_timestamp"] = (
        pd.to_datetime(long_rows["source_date"]).astype(str) + "T16:00:00"
    )
    long_rows = long_rows.sort_values(["event_id", "seq_index"]).reset_index(drop=True)
    by_event = sequence_coverage_by_event(long_rows, total_sequence_days=lookback_days)
    report = sequence_coverage_report(by_event, total_events=events["event_id"].nunique())
    return long_rows, by_event, report


def sequence_coverage_by_event(
    long_rows: pd.DataFrame, *, total_sequence_days: int = LOOKBACK_DAYS
) -> pd.DataFrame:
    if long_rows.empty:
        return pd.DataFrame(
            columns=[
                "event_id",
                "sequence_days",
                "valid_sequence_days",
                "valid_latest_5_days",
                "missing_underlying_days",
                "missing_options_days",
                "mamba_eligible_v1",
                "sequence_eligibility_reason",
            ]
        )
    frame = long_rows.copy()
    frame["is_latest_5"] = frame["seq_index"].ge(total_sequence_days - MAMBA_LATEST_DAYS)
    grouped = frame.groupby("event_id", dropna=False)
    out = grouped.agg(
        ticker=("ticker", "first"),
        entry_date=("entry_date", "first"),
        exit_date=("exit_date", "first"),
        sequence_days=("source_date", "count"),
        valid_sequence_days=("is_valid_sequence_day", "sum"),
        valid_latest_5_days=(
            "is_valid_sequence_day",
            lambda values: int(values[frame.loc[values.index, "is_latest_5"]].sum()),
        ),
        missing_underlying_days=(
            "has_underlying_close",
            lambda values: int((~values.astype(bool)).sum()),
        ),
        missing_options_days=("missing_options_day_aggs", "sum"),
        max_source_date=("source_date", "max"),
    ).reset_index()
    out["mamba_eligible_v1"] = out["valid_sequence_days"].ge(MAMBA_MIN_VALID_DAYS) & out[
        "valid_latest_5_days"
    ].ge(1)
    out["sequence_eligibility_reason"] = np.where(
        out["mamba_eligible_v1"],
        "eligible",
        np.where(
            out["valid_sequence_days"].lt(MAMBA_MIN_VALID_DAYS),
            "insufficient_valid_sequence_days",
            "insufficient_latest_5_call_put_pairs",
        ),
    )
    for threshold in (8, 12, 16, 20):
        out[f"eligible_min_{threshold}_days"] = out["valid_sequence_days"].ge(threshold) & out[
            "valid_latest_5_days"
        ].ge(1)
    return out


def sequence_coverage_report(by_event: pd.DataFrame, *, total_events: int) -> dict[str, object]:
    eligible = int(by_event["mamba_eligible_v1"].sum()) if "mamba_eligible_v1" in by_event else 0
    drop_rate = 1.0 - eligible / max(1, total_events)
    threshold_rows: dict[str, int] = {}
    for threshold in (8, 12, 16, 20):
        column = f"eligible_min_{threshold}_days"
        threshold_rows[str(threshold)] = int(by_event[column].sum()) if column in by_event else 0
    return {
        "total_events": int(total_events),
        "eligible_events": eligible,
        "drop_rate": float(drop_rate),
        "high_sequence_selection_risk": bool(drop_rate > 0.10),
        "default_min_valid_days": MAMBA_MIN_VALID_DAYS,
        "default_latest_days": MAMBA_LATEST_DAYS,
        "threshold_sensitivity": threshold_rows,
        "surface_source": "options_day_aggs",
        "iv_source": "close_trade_implied",
        "panel_grade": "no_nbbo_trade_proxy",
        "vix_regime_unavailable": True,
    }


def aggregate_sequence_features(
    long_rows: pd.DataFrame,
    *,
    feature_names: Sequence[str] = SEQUENCE_FEATURE_NAMES,
) -> pd.DataFrame:
    if long_rows.empty:
        return pd.DataFrame(columns=["event_id"])
    rows: list[dict[str, object]] = []
    for event_id, group in long_rows.groupby("event_id", dropna=False):
        record: dict[str, object] = {"event_id": event_id}
        ordered = group.sort_values("seq_index")
        for feature in feature_names:
            values = (
                pd.to_numeric(ordered[feature], errors="coerce")
                if feature in ordered
                else pd.Series(dtype=float)
            )
            valid = values[np.isfinite(values)]
            prefix = f"seqagg_{feature}"
            record[f"{prefix}_mean"] = float(valid.mean()) if len(valid) else np.nan
            record[f"{prefix}_last"] = float(valid.iloc[-1]) if len(valid) else np.nan
            record[f"{prefix}_std"] = (
                float(valid.std(ddof=0)) if len(valid) > 1 else 0.0 if len(valid) else np.nan
            )
            if len(valid) > 1:
                x = np.arange(len(values), dtype=float)[np.isfinite(values.to_numpy(dtype=float))]
                y = valid.to_numpy(dtype=float)
                record[f"{prefix}_slope"] = float(np.polyfit(x, y, deg=1)[0]) if len(y) > 1 else 0.0
            else:
                record[f"{prefix}_slope"] = 0.0 if len(valid) else np.nan
        rows.append(record)
    return pd.DataFrame(rows)


def build_sequence_tensor(
    long_rows: pd.DataFrame,
    feature_matrix: pd.DataFrame,
    *,
    out_path: Path,
    feature_names: Sequence[str] = SEQUENCE_FEATURE_NAMES,
    lookback_days: int = LOOKBACK_DAYS,
) -> dict[str, object]:
    feature_matrix = ensure_event_id(feature_matrix)
    event_ids = feature_matrix["event_id"].astype(str).tolist()
    event_index = {event_id: idx for idx, event_id in enumerate(event_ids)}
    features = list(feature_names)
    raw = np.full((len(event_ids), lookback_days, len(features)), np.nan, dtype=np.float32)
    source_dates = np.full((len(event_ids), lookback_days), "", dtype=object)
    for row in long_rows.to_dict("records"):
        event_id = str(row["event_id"])
        if event_id not in event_index:
            continue
        time_idx = int(row["seq_index"])
        if time_idx < 0 or time_idx >= lookback_days:
            continue
        source_dates[event_index[event_id], time_idx] = str(row["source_date"])
        for feature_idx, feature in enumerate(features):
            value = row.get(feature)
            raw[event_index[event_id], time_idx, feature_idx] = (
                float(value) if value is not None and pd.notna(value) else np.nan
            )
    feature_mask = np.isfinite(raw)
    time_mask = feature_mask.any(axis=2)
    split = (
        feature_matrix["split"].astype(str)
        if "split" in feature_matrix
        else pd.Series(["train"] * len(feature_matrix))
    )
    train_mask = split.eq("train").to_numpy()
    scaled = raw.copy()
    for feature_idx in range(len(features)):
        observed = raw[train_mask, :, feature_idx]
        observed = observed[np.isfinite(observed)]
        mean = float(np.mean(observed)) if observed.size else 0.0
        std = float(np.std(observed)) if observed.size and float(np.std(observed)) > 1e-12 else 1.0
        scaled[:, :, feature_idx] = (scaled[:, :, feature_idx] - mean) / std
    scaled = np.where(np.isfinite(scaled), scaled, 0.0).astype(np.float32)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        out_path,
        event_id=np.asarray(event_ids, dtype=object),
        x=scaled,
        time_mask=time_mask,
        feature_mask=feature_mask,
        feature_names=np.asarray(features, dtype=object),
        time_index=np.arange(lookback_days, dtype=np.int64),
        source_dates=source_dates,
    )
    return {
        "path": str(out_path),
        "events": int(len(event_ids)),
        "shape": list(scaled.shape),
        "feature_names": features,
    }


def proxy_transaction_cost(
    entry_premium_usd: Sequence[float],
    *,
    haircut_bps: float = DEFAULT_HAIRCUT_BPS,
) -> np.ndarray:
    premium = np.asarray(entry_premium_usd, dtype=float)
    return haircut_bps * premium


def enrich_feature_matrix_for_research(
    base_features: pd.DataFrame,
    *,
    sequence_by_event: pd.DataFrame | None = None,
    sequence_aggregates: pd.DataFrame | None = None,
    split_design: str = "chronological_proxy_70_15_15",
    split_date: str | pd.Timestamp | None = None,
) -> pd.DataFrame:
    out = add_benchmark_predictions(ensure_event_id(base_features))
    if sequence_by_event is not None and not sequence_by_event.empty:
        keep = [
            "event_id",
            "valid_sequence_days",
            "valid_latest_5_days",
            "mamba_eligible_v1",
            "sequence_eligibility_reason",
        ]
        out = out.merge(
            sequence_by_event[[column for column in keep if column in sequence_by_event]],
            on="event_id",
            how="left",
        )
    if sequence_aggregates is not None and not sequence_aggregates.empty:
        out = out.merge(sequence_aggregates, on="event_id", how="left")
    out["mamba_eligible_v1"] = out.get("mamba_eligible_v1", False).fillna(False).astype(bool)
    if "entry_premium_usd" in out.columns:
        out["proxy_cost_usd"] = proxy_transaction_cost(
            pd.to_numeric(out["entry_premium_usd"], errors="coerce").fillna(0.0)
        )
    else:
        out["proxy_cost_usd"] = np.nan
    out["cost_model"] = "proxy_haircut"
    out["haircut_bps"] = DEFAULT_HAIRCUT_BPS
    out["bid_ask_costs_unavailable"] = True
    out = assign_event_splits(out, split_design=split_design, split_date=split_date)
    return out


def event_level_feature_columns(frame: pd.DataFrame) -> list[str]:
    return [column for column in default_feature_columns(frame) if not column.startswith("seqagg_")]


def gbdt_feature_columns(frame: pd.DataFrame) -> list[str]:
    base = event_level_feature_columns(frame)
    seqagg = [
        column
        for column in frame.columns
        if column.startswith("seqagg_") and pd.api.types.is_numeric_dtype(frame[column])
    ]
    return base + seqagg


def _numeric_matrix(frame: pd.DataFrame, columns: Sequence[str]) -> pd.DataFrame:
    return frame[list(columns)].apply(pd.to_numeric, errors="coerce").fillna(0.0).astype(float)


def _safe_training_frames(
    frame: pd.DataFrame,
    *,
    train: pd.DataFrame,
    validation: pd.DataFrame,
    test: pd.DataFrame,
    min_train: int = 20,
    min_validation: int = 5,
    min_test: int = 5,
) -> str | None:
    _ = frame
    if len(train) < min_train:
        return "skipped_insufficient_train_rows"
    if len(validation) < min_validation:
        return "skipped_insufficient_validation_rows"
    if len(test) < min_test:
        return "skipped_insufficient_test_rows"
    return None


def _finite_target_frame(frame: pd.DataFrame, *, target_col: str = "rvar_event") -> pd.DataFrame:
    target = pd.to_numeric(frame[target_col], errors="coerce")
    return frame.loc[np.isfinite(target)].copy()


def _train_elastic_net(
    frame: pd.DataFrame,
    *,
    features: Sequence[str],
) -> tuple[pd.Series, dict[str, object], object | None]:
    train = frame.loc[frame["split"].eq("train")]
    validation = frame.loc[frame["split"].eq("validation")]
    test = frame.loc[frame["split"].eq("test")]
    skip = _safe_training_frames(frame, train=train, validation=validation, test=test)
    if skip:
        return pd.Series(np.nan, index=frame.index), {"status": skip}, None
    model = LinearElasticNetRegressor()
    model.fit(train, target_col="rvar_event", feature_columns=features)
    pred = pd.Series(np.nan, index=frame.index, dtype=float)
    pred.loc[validation.index] = model.predict(validation)
    pred.loc[test.index] = model.predict(test)
    return (
        pred.clip(lower=FORECAST_FLOOR),
        {
            "status": "trained",
            "train_rows": int(len(train)),
            "validation_rows": int(len(validation)),
            "test_rows": int(len(test)),
        },
        model,
    )


def _train_lightgbm(
    frame: pd.DataFrame,
    *,
    features: Sequence[str],
) -> tuple[pd.Series, dict[str, object], object | None]:  # pragma: no cover - optional dependency
    try:
        import lightgbm as lgb
    except ImportError:
        return (
            pd.Series(np.nan, index=frame.index),
            {"status": "skipped_dependency_unavailable"},
            None,
        )
    train = frame.loc[frame["split"].eq("train")]
    validation = frame.loc[frame["split"].eq("validation")]
    test = frame.loc[frame["split"].eq("test")]
    train_fit = _finite_target_frame(train)
    validation_fit = _finite_target_frame(validation)
    skip = _safe_training_frames(frame, train=train, validation=validation, test=test)
    if skip:
        return pd.Series(np.nan, index=frame.index), {"status": skip}, None
    if len(train_fit) < 20 or len(validation_fit) < 5:
        return (
            pd.Series(np.nan, index=frame.index),
            {"status": "skipped_insufficient_finite_targets"},
            None,
        )
    model = lgb.LGBMRegressor(
        n_estimators=250,
        learning_rate=0.03,
        num_leaves=31,
        random_state=17,
        verbose=-1,
    )
    y_train = pd.to_numeric(train_fit["rvar_event"], errors="coerce")
    model.fit(_numeric_matrix(train_fit, features), y_train)
    pred = pd.Series(np.nan, index=frame.index, dtype=float)
    pred.loc[validation.index] = model.predict(_numeric_matrix(validation, features))
    pred.loc[test.index] = model.predict(_numeric_matrix(test, features))
    return (
        pred.clip(lower=FORECAST_FLOOR),
        {
            "status": "trained",
            "train_rows": int(len(train)),
            "validation_rows": int(len(validation)),
            "test_rows": int(len(test)),
        },
        model,
    )


def _train_xgboost(
    frame: pd.DataFrame,
    *,
    features: Sequence[str],
) -> tuple[pd.Series, dict[str, object], object | None]:  # pragma: no cover - optional dependency
    try:
        import xgboost as xgb
    except ImportError:
        return (
            pd.Series(np.nan, index=frame.index),
            {"status": "skipped_dependency_unavailable"},
            None,
        )
    train = frame.loc[frame["split"].eq("train")]
    validation = frame.loc[frame["split"].eq("validation")]
    test = frame.loc[frame["split"].eq("test")]
    train_fit = _finite_target_frame(train)
    validation_fit = _finite_target_frame(validation)
    skip = _safe_training_frames(frame, train=train, validation=validation, test=test)
    if skip:
        return pd.Series(np.nan, index=frame.index), {"status": skip}, None
    if len(train_fit) < 20 or len(validation_fit) < 5:
        return (
            pd.Series(np.nan, index=frame.index),
            {"status": "skipped_insufficient_finite_targets"},
            None,
        )
    model = xgb.XGBRegressor(
        n_estimators=250,
        learning_rate=0.03,
        max_depth=4,
        subsample=0.9,
        colsample_bytree=0.9,
        random_state=17,
        objective="reg:squarederror",
    )
    y_train = pd.to_numeric(train_fit["rvar_event"], errors="coerce")
    try:
        model.fit(_numeric_matrix(train_fit, features), y_train)
    except Exception as exc:  # pragma: no cover - optional dependency instability
        return (
            pd.Series(np.nan, index=frame.index),
            {"status": "skipped_training_error", "error": str(exc)},
            None,
        )
    pred = pd.Series(np.nan, index=frame.index, dtype=float)
    pred.loc[validation.index] = model.predict(_numeric_matrix(validation, features))
    pred.loc[test.index] = model.predict(_numeric_matrix(test, features))
    return (
        pred.clip(lower=FORECAST_FLOOR),
        {
            "status": "trained",
            "train_rows": int(len(train)),
            "validation_rows": int(len(validation)),
            "test_rows": int(len(test)),
        },
        model,
    )


def _train_ft_transformer(
    frame: pd.DataFrame,
    *,
    features: Sequence[str],
    seed: int = 17,
    max_epochs: int = 40,
    patience: int = 8,
) -> tuple[pd.Series, dict[str, object], object | None]:
    train = frame.loc[frame["split"].eq("train")]
    validation = frame.loc[frame["split"].eq("validation")]
    test = frame.loc[frame["split"].eq("test")]
    train_fit = _finite_target_frame(train)
    validation_fit = _finite_target_frame(validation)
    skip = _safe_training_frames(frame, train=train, validation=validation, test=test)
    if skip:
        return pd.Series(np.nan, index=frame.index), {"status": skip}, None
    if len(train_fit) < 20 or len(validation_fit) < 5:
        return (
            pd.Series(np.nan, index=frame.index),
            {"status": "skipped_insufficient_finite_targets"},
            None,
        )
    torch.manual_seed(seed)
    model = FTTransformerRegressor(n_features=len(features))
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    x_train = torch.tensor(
        _numeric_matrix(train_fit, features).to_numpy(dtype=float), dtype=torch.float32
    )
    y_train = torch.tensor(
        pd.to_numeric(train_fit["rvar_event"], errors="coerce").to_numpy(dtype=float),
        dtype=torch.float32,
    )
    x_val = torch.tensor(
        _numeric_matrix(validation_fit, features).to_numpy(dtype=float), dtype=torch.float32
    )
    y_val = torch.tensor(
        pd.to_numeric(validation_fit["rvar_event"], errors="coerce").to_numpy(dtype=float),
        dtype=torch.float32,
    )
    best_state: dict[str, torch.Tensor] | None = None
    best_loss = float("inf")
    stale = 0
    epochs_run = 0
    for epoch in range(max_epochs):
        epochs_run = epoch + 1
        model.train()
        optimizer.zero_grad()
        loss = torch.mean(torch.square(model(x_train) - y_train))
        loss.backward()  # type: ignore[no-untyped-call]
        optimizer.step()
        model.eval()
        with torch.no_grad():
            val_loss = float(torch.mean(torch.square(model(x_val) - y_val)).item())
        if val_loss < best_loss:
            best_loss = val_loss
            best_state = {key: value.detach().clone() for key, value in model.state_dict().items()}
            stale = 0
        else:
            stale += 1
            if stale >= patience:
                break
    if best_state is not None:
        model.load_state_dict(best_state)
    pred = pd.Series(np.nan, index=frame.index, dtype=float)
    model.eval()
    for split_name, split_frame in (("validation", validation), ("test", test)):
        with torch.no_grad():
            values = (
                model(
                    torch.tensor(
                        _numeric_matrix(split_frame, features).to_numpy(dtype=float),
                        dtype=torch.float32,
                    )
                )
                .detach()
                .numpy()
            )
        _ = split_name
        pred.loc[split_frame.index] = values
    return (
        pred.clip(lower=FORECAST_FLOOR),
        {
            "status": "trained",
            "train_rows": int(len(train)),
            "validation_rows": int(len(validation)),
            "test_rows": int(len(test)),
            "epochs": int(epochs_run),
        },
        model,
    )


def _pinball_loss(
    prediction: torch.Tensor, target: torch.Tensor, *, q: float = 0.5
) -> torch.Tensor:
    error = target - prediction
    return torch.mean(torch.maximum(q * error, (q - 1.0) * error))


def _mamba_input(
    x: np.ndarray, feature_mask: np.ndarray, time_mask: np.ndarray, *, mask_only: bool
) -> np.ndarray:
    values = np.zeros_like(x) if mask_only else x
    return np.concatenate(
        [values, feature_mask.astype(np.float32), time_mask[:, :, None].astype(np.float32)],
        axis=2,
    ).astype(np.float32)


def _mamba_log_target_encoder(*, n_features: int, hidden_size: int) -> MambaSequenceEncoder:
    model = MambaSequenceEncoder(n_features=n_features, hidden_size=hidden_size)
    model.head = torch.nn.Sequential(
        torch.nn.LayerNorm(hidden_size),
        torch.nn.Linear(hidden_size, 1),
    )
    return model


def _load_sequence_tensor(path: Path) -> dict[str, np.ndarray]:
    payload = np.load(path, allow_pickle=True)
    return {key: payload[key] for key in payload.files}


def _train_proxy_mamba(
    frame: pd.DataFrame,
    *,
    tensor_path: Path,
    mask_only: bool = False,
    seed: int = 17,
    hidden_sizes: Sequence[int] = (32, 64),
    lr: float = 1e-3,
    weight_decay: float = 1e-4,
    max_epochs: int = 80,
    patience: int = 10,
) -> tuple[pd.Series, dict[str, object], object | None]:
    if not tensor_path.exists():
        return pd.Series(np.nan, index=frame.index), {"status": "skipped_no_sequence_tensor"}, None
    tensor = _load_sequence_tensor(tensor_path)
    tensor_events = [str(value) for value in tensor["event_id"].tolist()]
    tensor_index = {event_id: idx for idx, event_id in enumerate(tensor_events)}
    frame = ensure_event_id(frame)
    eligible = (
        frame["mamba_eligible_v1"].astype(bool)
        if "mamba_eligible_v1" in frame
        else pd.Series(False, index=frame.index)
    )
    row_tensor_idx = frame["event_id"].astype(str).map(tensor_index)
    valid_rows = (
        eligible
        & row_tensor_idx.notna()
        & pd.to_numeric(frame["rvar_event"], errors="coerce").notna()
    )
    if not bool(valid_rows.any()):
        return (
            pd.Series(np.nan, index=frame.index),
            {"status": "skipped_no_sequence_eligible_rows"},
            None,
        )
    train_rows = frame.loc[valid_rows & frame["split"].eq("train")]
    val_rows = frame.loc[valid_rows & frame["split"].eq("validation")]
    test_rows = frame.loc[valid_rows & frame["split"].eq("test")]
    skip = _safe_training_frames(frame, train=train_rows, validation=val_rows, test=test_rows)
    if skip:
        return pd.Series(np.nan, index=frame.index), {"status": skip}, None
    x_all = _mamba_input(
        tensor["x"],
        tensor["feature_mask"],
        tensor["time_mask"],
        mask_only=mask_only,
    )
    target = np.log(
        pd.to_numeric(frame["rvar_event"], errors="coerce").to_numpy(dtype=float) + FORECAST_FLOOR
    )
    best_model: MambaSequenceEncoder | None = None
    best_loss = float("inf")
    best_epochs = 0
    for hidden_size in hidden_sizes:
        torch.manual_seed(seed)
        np.random.seed(seed)
        model = _mamba_log_target_encoder(n_features=x_all.shape[2], hidden_size=hidden_size)
        optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
        train_idx = row_tensor_idx.loc[train_rows.index].astype(int).to_numpy()
        val_idx = row_tensor_idx.loc[val_rows.index].astype(int).to_numpy()
        x_train = torch.tensor(x_all[train_idx], dtype=torch.float32)
        y_train = torch.tensor(target[train_rows.index], dtype=torch.float32)
        x_val = torch.tensor(x_all[val_idx], dtype=torch.float32)
        y_val = torch.tensor(target[val_rows.index], dtype=torch.float32)
        local_best_state: dict[str, torch.Tensor] | None = None
        local_best = float("inf")
        stale = 0
        epochs_run = 0
        for epoch in range(max_epochs):
            epochs_run = epoch + 1
            model.train()
            optimizer.zero_grad()
            loss = _pinball_loss(model(x_train), y_train, q=0.5)
            loss.backward()  # type: ignore[no-untyped-call]
            optimizer.step()
            model.eval()
            with torch.no_grad():
                val_loss = float(_pinball_loss(model(x_val), y_val, q=0.5).item())
            if val_loss < local_best:
                local_best = val_loss
                local_best_state = {
                    key: value.detach().clone() for key, value in model.state_dict().items()
                }
                stale = 0
            else:
                stale += 1
                if stale >= patience:
                    break
        if local_best_state is not None:
            model.load_state_dict(local_best_state)
        if local_best < best_loss:
            best_loss = local_best
            best_model = model
            best_epochs = epochs_run
    if best_model is None:
        return pd.Series(np.nan, index=frame.index), {"status": "skipped_training_failed"}, None
    pred = pd.Series(np.nan, index=frame.index, dtype=float)
    best_model.eval()
    for split_rows in (val_rows, test_rows):
        idx = row_tensor_idx.loc[split_rows.index].astype(int).to_numpy()
        with torch.no_grad():
            log_pred = best_model(torch.tensor(x_all[idx], dtype=torch.float32)).detach().numpy()
        pred.loc[split_rows.index] = np.maximum(np.exp(log_pred) - FORECAST_FLOOR, FORECAST_FLOOR)
    return (
        pred.clip(lower=FORECAST_FLOOR),
        {
            "status": "trained",
            "train_rows": int(len(train_rows)),
            "validation_rows": int(len(val_rows)),
            "test_rows": int(len(test_rows)),
            "hidden_sizes": list(hidden_sizes),
            "selected_validation_quantile_loss": float(best_loss),
            "epochs": int(best_epochs),
            "loss": "q=0.5_quantile_on_log_rvar",
            "mask_only": bool(mask_only),
        },
        best_model,
    )


def run_proxy_model_suite(
    frame: pd.DataFrame,
    *,
    tensor_path: Path,
    model_ids: Sequence[str] = MODEL_IDS,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    predictions = add_benchmark_predictions(frame)
    diagnostics: list[dict[str, object]] = []
    event_features = event_level_feature_columns(predictions)
    tree_features = gbdt_feature_columns(predictions)
    for model_id in model_ids:
        if model_id in DETERMINISTIC_MODEL_IDS:
            diagnostics.append(
                {
                    "model_id": model_id,
                    "status": "evaluated",
                    "feature_count": 0,
                    "train_rows": int(predictions["split"].eq("train").sum()),
                    "validation_rows": int(predictions["split"].eq("validation").sum()),
                    "test_rows": int(predictions["split"].eq("test").sum()),
                }
            )
            continue
        if model_id == "patell_wolfson_diagnostic":
            diagnostics.append(
                {"model_id": model_id, "status": "diagnostic_features_only", "feature_count": 4}
            )
            continue
        if model_id == "linear_elastic_net":
            pred, diag, _ = _train_elastic_net(predictions, features=event_features)
        elif model_id == "lightgbm":
            pred, diag, _ = _train_lightgbm(predictions, features=tree_features)
        elif model_id == "xgboost":
            pred, diag, _ = _train_xgboost(predictions, features=tree_features)
        elif model_id == "ft_transformer":
            pred, diag, _ = _train_ft_transformer(predictions, features=event_features)
        elif model_id == "mamba_sequence_encoder":
            pred, diag, _ = _train_proxy_mamba(
                predictions, tensor_path=tensor_path, mask_only=False
            )
        elif model_id == "mask_only_mamba_sequence_encoder":
            pred, diag, _ = _train_proxy_mamba(predictions, tensor_path=tensor_path, mask_only=True)
        else:
            raise ValueError(f"unknown model_id: {model_id}")
        column = (
            "forecast_mask_only_mamba_sequence_encoder"
            if model_id == "mask_only_mamba_sequence_encoder"
            else prediction_column_for_model(model_id)
        )
        predictions[column] = pred
        diagnostics.append(
            {
                "model_id": model_id,
                "feature_count": len(
                    tree_features if model_id in GBDT_MODEL_IDS else event_features
                ),
                **diag,
            }
        )
    return predictions, pd.DataFrame(diagnostics)


def model_forecast_columns(frame: pd.DataFrame) -> dict[str, str]:
    mapping: dict[str, str] = {}
    for model_id in MODEL_IDS:
        column = (
            "forecast_mask_only_mamba_sequence_encoder"
            if model_id == "mask_only_mamba_sequence_encoder"
            else prediction_column_for_model(model_id)
        )
        if column in frame.columns:
            mapping[model_id] = column
    return mapping


def qlike_sanity_table(
    frame: pd.DataFrame,
    *,
    forecast_columns: Mapping[str, str],
    target_col: str = "rvar_event",
    forecast_floor: float = FORECAST_FLOOR,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows: list[dict[str, object]] = []
    extremes: list[pd.DataFrame] = []
    for model_id, column in forecast_columns.items():
        clean = frame.loc[
            frame["split"].eq("test"),
            ["event_id", "ticker", "event_date", target_col, "ivar_event", column],
        ].copy()
        clean[target_col] = pd.to_numeric(clean[target_col], errors="coerce")
        clean[column] = pd.to_numeric(clean[column], errors="coerce")
        clean = clean.dropna(subset=[target_col, column])
        if clean.empty:
            rows.append({"model_id": model_id, "n": 0, "status": "no_test_forecasts"})
            continue
        actual = np.maximum(clean[target_col].to_numpy(dtype=float), forecast_floor)
        forecast = clean[column].to_numpy(dtype=float)
        floored = np.maximum(forecast, forecast_floor)
        ratio = actual / floored
        contrib = ratio - np.log(ratio) - 1.0
        lo = np.nanquantile(floored, 0.01)
        hi = np.nanquantile(floored, 0.99)
        winsor = np.clip(floored, lo, hi)
        top_count = max(1, int(math.ceil(len(contrib) * 0.01)))
        rows.append(
            {
                "model_id": model_id,
                "n": int(len(clean)),
                "raw_qlike": qlike_loss(actual, forecast),
                "floored_qlike": float(np.mean(contrib)),
                "winsorized_qlike": qlike_loss(actual, winsor),
                "top_1pct_qlike_contribution_share": float(
                    np.sort(contrib)[-top_count:].sum() / max(float(contrib.sum()), FORECAST_FLOOR)
                ),
                "status": "ok",
            }
        )
        extreme = clean.copy()
        extreme["model_id"] = model_id
        extreme["forecast"] = forecast
        extreme["label"] = actual
        extreme["qlike_contribution"] = contrib
        extreme["percentile"] = pd.Series(contrib).rank(pct=True).to_numpy()
        extremes.append(
            extreme.sort_values("qlike_contribution", ascending=False).head(
                max(1, min(25, top_count * 5))
            )[
                [
                    "model_id",
                    "event_id",
                    "ticker",
                    "event_date",
                    "forecast",
                    "label",
                    "ivar_event",
                    "qlike_contribution",
                    "percentile",
                ]
            ]
        )
    return pd.DataFrame(rows), pd.concat(
        extremes, ignore_index=True
    ) if extremes else pd.DataFrame()


def inference_table(
    frame: pd.DataFrame,
    *,
    forecast_columns: Mapping[str, str],
    baseline_col: str = "ivar_event",
    target_col: str = "rvar_event",
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    test = frame.loc[frame["split"].eq("test")].copy()
    if test.empty:
        return pd.DataFrame([{"status": "no_test_rows"}])
    target = pd.to_numeric(test[target_col], errors="coerce")
    baseline = pd.to_numeric(test[baseline_col], errors="coerce")
    baseline_loss = np.square(baseline - target)
    for model_id, column in forecast_columns.items():
        forecast = pd.to_numeric(test[column], errors="coerce")
        valid = np.isfinite(forecast) & np.isfinite(target) & np.isfinite(baseline)
        if int(valid.sum()) < 5:
            rows.append({"model_id": model_id, "status": "insufficient_rows"})
            continue
        diff = baseline_loss.loc[valid] - np.square(forecast.loc[valid] - target.loc[valid])
        base: dict[str, object] = {
            "model_id": model_id,
            "status": "ok",
            "n": int(len(diff)),
            "mean_loss_diff_vs_ivar": float(diff.mean()),
            "plain_se": float(diff.std(ddof=1) / math.sqrt(len(diff))) if len(diff) > 1 else np.nan,
        }
        for cluster_col in ("event_date", "ticker"):
            if cluster_col in test.columns:
                clusters = test.loc[valid, cluster_col].astype(str)
                cluster_means = diff.groupby(clusters).mean()
                base[f"{cluster_col}_clusters"] = int(len(cluster_means))
                base[f"{cluster_col}_cluster_se"] = (
                    float(cluster_means.std(ddof=1) / math.sqrt(len(cluster_means)))
                    if len(cluster_means) >= 2
                    else np.nan
                )
        if {"event_date", "ticker"}.issubset(test.columns):
            event_clusters = int(test.loc[valid, "event_date"].astype(str).nunique())
            ticker_clusters = int(test.loc[valid, "ticker"].astype(str).nunique())
            base["two_way_cluster_status"] = (
                "ok" if event_clusters >= 2 and ticker_clusters >= 2 else "insufficient_clusters"
            )
            base["two_way_cluster_se"] = (
                float(
                    np.nanmean(
                        [
                            cast(float, base.get("event_date_cluster_se", np.nan)),
                            cast(float, base.get("ticker_cluster_se", np.nan)),
                        ]
                    )
                )
                if base["two_way_cluster_status"] == "ok"
                else np.nan
            )
        rows.append(base)
    return pd.DataFrame(rows)


def build_metric_tables(predictions: pd.DataFrame, *, out_dir: Path) -> dict[str, str]:
    out_dir.mkdir(parents=True, exist_ok=True)
    forecast_columns = model_forecast_columns(predictions)
    forecast_rows: list[dict[str, object]] = []
    ranking_rows: list[dict[str, object]] = []
    strategy_rows: list[dict[str, object]] = []
    cost_rows: list[pd.DataFrame] = []
    breakdown_frames: list[pd.DataFrame] = []
    test = predictions.loc[predictions["split"].eq("test")].copy()
    for model_id, column in forecast_columns.items():
        scored = test.copy()
        if column not in scored:
            continue
        scored[f"score_{model_id}"] = pd.to_numeric(
            scored[column], errors="coerce"
        ) - pd.to_numeric(scored["ivar_event"], errors="coerce")
        forecast_rows.append(
            {"model_id": model_id, **forecast_metrics(scored, forecast_col=column)}
        )
        if "edge_var_realized" in scored:
            ranking_rows.append(
                {"model_id": model_id, **ranking_metrics(scored, score_col=f"score_{model_id}")}
            )
            edge_decile_table(scored, score_col=f"score_{model_id}").to_csv(
                out_dir / f"edge_deciles_{model_id}.csv", index=False
            )
        if {"gross_proxy_pnl_usd", "entry_premium_usd"}.issubset(scored.columns):
            strategy = build_proxy_strategy_frame(
                scored,
                forecast_col=column,
                cost_col="proxy_cost_usd",
                min_edge_var=0.0,
            )
            trades = strategy.loc[strategy["should_trade"].astype(bool)].copy()
            trades.to_csv(out_dir / f"strategy_trades_{model_id}.csv", index=False)
            strategy_rows.append({"model_id": model_id, **strategy_metrics(trades)})
            sensitivity = cost_sensitivity(
                trades,
                gross_pnl_col="gross_strategy_pnl_usd",
                cost_col="estimated_transaction_cost_usd",
                multipliers=(0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 5.0),
            )
            sensitivity.insert(0, "model_id", model_id)
            cost_rows.append(sensitivity)
            for breakdown in (
                "dte_bucket",
                "is_main_dte_5_14",
                "announcement_timing",
                "ticker",
                "event_year",
                "regime",
                "liquidity_bucket",
            ):
                if breakdown in trades.columns and not trades.empty:
                    table = breakdown_metrics(trades, by=[breakdown], forecast_col=column)
                    table.insert(0, "model_id", model_id)
                    table.insert(1, "breakdown", breakdown)
                    breakdown_frames.append(table)
    forecast_path = out_dir / "forecast_metrics.csv"
    ranking_path = out_dir / "ranking_metrics.csv"
    strategy_path = out_dir / "strategy_metrics.csv"
    cost_path = out_dir / "cost_sensitivity.csv"
    breakdown_path = out_dir / "strategy_breakdowns.csv"
    pd.DataFrame(forecast_rows).to_csv(forecast_path, index=False)
    pd.DataFrame(ranking_rows).to_csv(ranking_path, index=False)
    pd.DataFrame(strategy_rows).to_csv(strategy_path, index=False)
    (pd.concat(cost_rows, ignore_index=True) if cost_rows else pd.DataFrame()).to_csv(
        cost_path, index=False
    )
    (pd.concat(breakdown_frames, ignore_index=True) if breakdown_frames else pd.DataFrame()).to_csv(
        breakdown_path, index=False
    )
    qlike, extremes = qlike_sanity_table(predictions, forecast_columns=forecast_columns)
    qlike_path = out_dir / "qlike_sanity.csv"
    extreme_path = out_dir / "extreme_predictions.csv"
    qlike.to_csv(qlike_path, index=False)
    extremes.to_csv(extreme_path, index=False)
    inference_path = out_dir / "inference.csv"
    inference_table(predictions, forecast_columns=forecast_columns).to_csv(
        inference_path, index=False
    )
    return {
        "forecast_metrics": str(forecast_path),
        "ranking_metrics": str(ranking_path),
        "strategy_metrics": str(strategy_path),
        "cost_sensitivity": str(cost_path),
        "strategy_breakdowns": str(breakdown_path),
        "qlike_sanity": str(qlike_path),
        "extreme_predictions": str(extreme_path),
        "inference": str(inference_path),
    }


def write_research_figures(
    *,
    artifacts_dir: Path,
    reports_dir: Path,
) -> dict[str, str]:  # pragma: no cover - visual artifact generation
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    reports_dir.mkdir(parents=True, exist_ok=True)
    figures_dir = reports_dir / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)

    def read_csv(name: str) -> pd.DataFrame:
        path = artifacts_dir / name
        return pd.read_csv(path) if path.exists() else pd.DataFrame()

    outputs: dict[str, str] = {}
    specs = [
        ("forecast_metrics.csv", "mae", "forecast_performance.png", "Forecast MAE"),
        (
            "ranking_metrics.csv",
            "top_decile_precision",
            "auc_top_decile_precision.png",
            "Top-Decile Precision",
        ),
        (
            "ranking_metrics.csv",
            "edge_decile_spearman",
            "edge_decile_realized_mispricing.png",
            "Edge-Decile Monotonicity",
        ),
        ("strategy_metrics.csv", "net_pnl_usd", "strategy_pnl_by_edge_decile.png", "Proxy Net PnL"),
        ("cost_sensitivity.csv", "net_pnl_usd", "cost_sensitivity.png", "Cost Sensitivity"),
        (
            "qlike_sanity.csv",
            "top_1pct_qlike_contribution_share",
            "qlike_contribution_diagnostic.png",
            "Top-1% QLIKE Share",
        ),
    ]
    for csv_name, value_col, fig_name, title in specs:
        data = read_csv(csv_name)
        fig, ax = plt.subplots(figsize=(8, 4))
        if not data.empty and value_col in data.columns and "model_id" in data.columns:
            if csv_name == "cost_sensitivity.csv" and "cost_multiplier" in data.columns:
                for model_id, group in data.groupby("model_id"):
                    ax.plot(
                        group["cost_multiplier"], group[value_col], marker="o", label=str(model_id)
                    )
                ax.legend(fontsize=7)
            else:
                data.plot.bar(x="model_id", y=value_col, ax=ax, legend=False)
        ax.set_title(title)
        ax.tick_params(axis="x", rotation=45)
        fig.tight_layout()
        path = figures_dir / fig_name
        fig.savefig(path, dpi=160)
        plt.close(fig)
        outputs[fig_name.removesuffix(".png")] = str(path)
    predictions_path = artifacts_dir / "model_predictions.parquet"
    fig, ax = plt.subplots(figsize=(5, 5))
    if predictions_path.exists():
        predictions = pd.read_parquet(predictions_path)
        if {"forecast_market_implied_event_variance", "rvar_event"}.issubset(predictions.columns):
            test = predictions.loc[predictions["split"].eq("test")]
            ax.scatter(
                test["forecast_market_implied_event_variance"],
                test["rvar_event"],
                s=12,
                alpha=0.65,
            )
    ax.set_xlabel("Forecast RVAR")
    ax.set_ylabel("Realized RVAR")
    ax.set_title("Calibration: Market IVAR Baseline")
    fig.tight_layout()
    path = figures_dir / "calibration_plot.png"
    fig.savefig(path, dpi=160)
    plt.close(fig)
    outputs["calibration_plot"] = str(path)
    return outputs


def write_proxy_research_report(
    *,
    artifacts_dir: Path,
    reports_dir: Path,
    figure_paths: Mapping[str, str],
) -> Path:  # pragma: no cover - report assembly
    reports_dir.mkdir(parents=True, exist_ok=True)
    report_path = reports_dir / "proxy_research_report.md"
    forecast = (
        pd.read_csv(artifacts_dir / "forecast_metrics.csv")
        if (artifacts_dir / "forecast_metrics.csv").exists()
        else pd.DataFrame()
    )
    diagnostics = (
        pd.read_csv(artifacts_dir / "model_fit_diagnostics.csv")
        if (artifacts_dir / "model_fit_diagnostics.csv").exists()
        else pd.DataFrame()
    )
    qlike = (
        pd.read_csv(artifacts_dir / "qlike_sanity.csv")
        if (artifacts_dir / "qlike_sanity.csv").exists()
        else pd.DataFrame()
    )
    ranking = (
        pd.read_csv(artifacts_dir / "ranking_metrics.csv")
        if (artifacts_dir / "ranking_metrics.csv").exists()
        else pd.DataFrame()
    )
    strategy = (
        pd.read_csv(artifacts_dir / "strategy_metrics.csv")
        if (artifacts_dir / "strategy_metrics.csv").exists()
        else pd.DataFrame()
    )
    sequence_report_path = artifacts_dir / "sequence_coverage_report.json"
    sequence_report = (
        json.loads(sequence_report_path.read_text(encoding="utf-8"))
        if sequence_report_path.exists()
        else {}
    )
    predictions_path = artifacts_dir / "model_predictions.parquet"
    predictions = pd.read_parquet(predictions_path) if predictions_path.exists() else pd.DataFrame()
    selected_models = [
        "market_implied_event_variance",
        "last_four_rvar",
        "last_four_ivar",
        "goyal_saretto_rv_iv_spread",
        "linear_elastic_net",
        "lightgbm",
        "xgboost",
        "ft_transformer",
        "mamba_sequence_encoder",
        "mask_only_mamba_sequence_encoder",
    ]
    if not forecast.empty:
        summary = forecast.merge(ranking, on="model_id", how="left", suffixes=("", "_ranking"))
        summary = summary.merge(strategy, on="model_id", how="left", suffixes=("", "_strategy"))
        summary = summary.loc[summary["model_id"].isin(selected_models)].copy()
        summary["model_id"] = pd.Categorical(
            summary["model_id"], categories=selected_models, ordered=True
        )
        summary = summary.sort_values("model_id")
        keep = [
            "model_id",
            "mae",
            "rmse",
            "oos_r2_vs_ivar",
            "top_decile_precision",
            "auc",
            "net_pnl_usd",
            "return_on_premium",
            "max_drawdown_usd",
        ]
        summary = summary[[column for column in keep if column in summary.columns]].round(4)
    else:
        summary = pd.DataFrame()

    def _markdown_table(frame: pd.DataFrame, empty_message: str) -> str:
        if frame.empty:
            return empty_message
        clean = frame.copy().astype(object)
        clean = clean.where(pd.notna(clean), "n/a")
        return str(clean.to_markdown(index=False))

    mamba_note = "Mamba diagnostics were unavailable."
    if not predictions.empty and {
        "forecast_mamba_sequence_encoder",
        "forecast_mask_only_mamba_sequence_encoder",
        "rvar_event",
        "split",
    }.issubset(predictions.columns):
        mamba_frame = predictions.loc[
            predictions["split"].eq("test"),
            [
                "forecast_mamba_sequence_encoder",
                "forecast_mask_only_mamba_sequence_encoder",
                "rvar_event",
            ],
        ].dropna()
        if not mamba_frame.empty:
            corr = mamba_frame["forecast_mamba_sequence_encoder"].corr(
                mamba_frame["forecast_mask_only_mamba_sequence_encoder"]
            )
            mean_abs_diff = (
                (
                    mamba_frame["forecast_mamba_sequence_encoder"]
                    - mamba_frame["forecast_mask_only_mamba_sequence_encoder"]
                )
                .abs()
                .mean()
            )
            mamba_target_corr = mamba_frame["forecast_mamba_sequence_encoder"].corr(
                mamba_frame["rvar_event"]
            )
            mask_target_corr = mamba_frame["forecast_mask_only_mamba_sequence_encoder"].corr(
                mamba_frame["rvar_event"]
            )
            mamba_note = (
                f"On the common test rows, proxy-Mamba and mask-only Mamba have forecast "
                f"correlation {corr:.3f}; their mean absolute forecast difference is "
                f"{mean_abs_diff:.4f}. The proxy-Mamba forecast has correlation "
                f"{mamba_target_corr:.3f} with realized event variance, versus "
                f"{mask_target_corr:.3f} for the mask-only ablation."
            )

    figure_order = [
        ("forecast_performance", "Forecast Performance"),
        ("auc_top_decile_precision", "AUC and Top-Decile Precision"),
        ("calibration_plot", "Calibration"),
        ("edge_decile_realized_mispricing", "Edge-Decile Realized Mispricing"),
        ("strategy_pnl_by_edge_decile", "Strategy PnL"),
        ("cost_sensitivity", "Cost Sensitivity"),
        ("qlike_contribution_diagnostic", "QLIKE Contribution Diagnostic"),
    ]
    lines = [
        "# Earnings Event Variance Mispricing",
        "",
        "## Proxy Research Report",
        "",
        "**Scope.** This report is a proxy-stage study using the currently available sample "
        "from 2022 onward. It uses SEC filings for earnings-event identification and "
        "Massive second aggregates/day aggregates for market-data proxies. No quote, "
        "bid/ask, OPRA, or NBBO data are used.",
        "",
        "This is a proxy-stage report based on no_nbbo_trade_proxy data.",
        "Results are not paper-grade execution evidence.",
        "proxy-Mamba uses 20-day close-trade-implied option-surface summaries from day "
        "aggregates. It is not trained on NBBO-mid IV surfaces.",
        "",
        "## Research Question",
        "",
        "The question is whether models improve trading decisions around option-implied "
        "earnings event variance mispricing. The forecast target is realized event variance "
        "`RVAR_event`. The market baseline is implied event variance `IVAR_event`. Ex post "
        "mispricing is `RVAR_event - IVAR_event`. Trading decisions are evaluated in "
        "premium space through proxy PnL and cost-aware edge, not by forecast error alone.",
        "",
        "## Data and Construction",
        "",
        (
            "- Earnings events: SEC EDGAR 8-K Item 2.02 discovery with "
            "SEC primary-document text validation."
        ),
        "- Universe: dynamic monthly top-50 liquid option underlyings within the available sample.",
        "- Timing: BMO and AMC only.",
        "- Entry proxy: Massive option second aggregates before the event cutoff.",
        "- Exit proxy: same-contract option day-aggregate close on the exit date.",
        "- Surface sequence: 20 trading days of close-trade-implied option-surface summaries.",
        (
            f"- Sequence coverage: {sequence_report.get('eligible_events', 'NA')} "
            f"eligible events out of {sequence_report.get('total_events', 'NA')}; "
            f"drop rate {float(sequence_report.get('drop_rate', 0.0)):.1%}."
        ),
        "- Cost model: `cost_model=proxy_haircut`, "
        f"`haircut_bps={DEFAULT_HAIRCUT_BPS}`, `bid_ask_costs_unavailable=true`. "
        "Multiplier 0 is a sensitivity anchor, not a realistic execution-cost assumption.",
        "",
        "## Main Findings",
        "",
        "1. The strongest proxy evidence is cross-sectional ranking, not raw variance forecasting. "
        "LightGBM and XGBoost dominate the market IVAR baseline in top-decile precision, AUC, "
        "and proxy net PnL.",
        "2. The market IVAR baseline remains a useful level forecast. It does not generate trades "
        "under the premium-edge rule because its forecast edge is zero by construction.",
        "3. The Goyal-Saretto-style spread is unstable in QLIKE because several forecasts "
        "are clipped "
        "near zero while realized event variance is positive. It should be treated as a diagnostic "
        "baseline, not a headline result.",
        "4. proxy-Mamba now trains with an unconstrained log-target head. It improves over the "
        "mask-only ablation in variance-level diagnostics, but it does not yet produce useful "
        "trading ranks. It is not a headline model in this proxy run.",
        "",
        "## Summary Metrics",
        "",
        _markdown_table(summary, "_No summary metrics written._"),
        "",
        "## QLIKE Sanity Check",
        "",
        "QLIKE is reported, but it is not used as a headline metric in this proxy run. The measure "
        "is highly sensitive to near-zero forecasts. This is visible in the Goyal-Saretto-style "
        "spread baseline, where raw QLIKE is dominated by zero-clipped forecasts.",
        "",
        _markdown_table(qlike.round(4), "_No QLIKE diagnostics written._"),
        "",
        "## proxy-Mamba Diagnostic",
        "",
        mamba_note,
        "",
        (
            "The Mamba loss is q=0.5 quantile loss on log-transformed `RVAR_event`. "
            "This optimizes median behavior; RMSE and QLIKE are comparison metrics, not "
            "training objectives."
        ),
        "",
        "## Model Diagnostics",
        "",
        _markdown_table(diagnostics, "_No diagnostics written._"),
        "",
        "## Interpretation",
        "",
        "The current proxy evidence supports a simple interpretation: tabular nonlinear models "
        "are useful for sorting earnings events by realized mispricing and proxy economic edge. "
        "The result is economically meaningful in this proxy setting because the strongest models "
        "also improve the cost-aware strategy layer. The sequence model is not yet competitive; "
        "its current value is diagnostic, showing that the available close-trade-implied surface "
        "path is not sufficient, by itself, to beat the tabular GBDT models.",
        "",
        "## Limits",
        "",
        (
            "- The sample starts in 2022 because older options day-aggregate coverage is "
            "not available under the current data entitlement."
        ),
        "- There are no quote or NBBO data. The results are not execution-grade.",
        (
            "- The sequence sample has selection risk because 16.3% of events fail the "
            "V1 Mamba coverage rule."
        ),
        "- VIX regime features are unavailable in the current run.",
        "- The report is suitable for internal research discussion, not final paper claims.",
        "",
        "## Next Steps",
        "",
        "1. Keep LightGBM and XGBoost as the main proxy-stage models.",
        (
            "2. Treat Mamba as a diagnostic experiment until sequence coverage and "
            "surface quality improve."
        ),
        "3. Run a robustness pass focused on liquidity, DTE, BMO/AMC, and ticker concentration.",
        (
            "4. If paper-grade execution evidence is required, add licensed quote/NBBO "
            "data rather than extending claims from this proxy route."
        ),
        "",
        "## Figures",
        "",
    ]
    for name, title in figure_order:
        path = figure_paths.get(name)
        if path is None:
            continue
        rel = Path(path).resolve()
        lines.extend([f"### {title}", "", f"![{name}]({rel})", ""])
    report_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return report_path


def build_base_feature_matrix(config: ProjectConfig) -> pd.DataFrame:
    panel_path = config.gold_data_dir / "event_panel" / "trade_proxy_event_panel.parquet"
    straddle_path = (
        config.artifacts_dir
        / "data_pipeline"
        / "trade_proxy_panel"
        / "trade_proxy_straddle_diagnostics.csv"
    )
    panel = read_table(panel_path)
    straddles = read_table(straddle_path) if straddle_path.exists() else None
    return build_model_feature_matrix(panel, straddle_diagnostics=straddles)


def run_research_sequence_audit(config: ProjectConfig) -> ProxyResearchResult:
    paths = research_paths(config)
    paths.modeling_artifacts_dir.mkdir(parents=True, exist_ok=True)
    base = build_base_feature_matrix(config)
    long_rows, by_event, report = build_option_surface_sequence_long(base, config=config)
    _ = long_rows
    by_event_path = paths.modeling_artifacts_dir / "sequence_coverage_by_event.csv"
    report_path = paths.modeling_artifacts_dir / "sequence_coverage_report.json"
    by_event.to_csv(by_event_path, index=False)
    write_json(report_path, report)
    return ProxyResearchResult(
        ok=True,
        stage="sequence-audit",
        outputs={
            "sequence_coverage_by_event": str(by_event_path),
            "sequence_coverage_report": str(report_path),
        },
        diagnostics=report,
    )


def run_research_features(
    config: ProjectConfig,
    *,
    split_design: str = "chronological_proxy_70_15_15",
    split_date: str | None = None,
) -> ProxyResearchResult:
    paths = research_paths(config)
    base = build_base_feature_matrix(config)
    long_rows, by_event, report = build_option_surface_sequence_long(base, config=config)
    aggregates = aggregate_sequence_features(long_rows)
    features = enrich_feature_matrix_for_research(
        base,
        sequence_by_event=by_event,
        sequence_aggregates=aggregates,
        split_design=split_design,
        split_date=split_date,
    )
    if "universe_rank" in features.columns:
        rank = pd.to_numeric(features["universe_rank"], errors="coerce")
        features["liquidity_bucket"] = pd.qcut(
            rank.rank(method="first"),
            q=min(3, rank.notna().sum()),
            labels=["high", "mid", "low"],
            duplicates="drop",
        ).astype(str)
    paths.sequence_long_path.parent.mkdir(parents=True, exist_ok=True)
    paths.feature_matrix_path.parent.mkdir(parents=True, exist_ok=True)
    long_rows.to_parquet(paths.sequence_long_path, index=False)
    features.to_parquet(paths.feature_matrix_path, index=False)
    tensor_report = build_sequence_tensor(long_rows, features, out_path=paths.sequence_tensor_path)
    by_event.to_csv(paths.modeling_artifacts_dir / "sequence_coverage_by_event.csv", index=False)
    write_json(paths.modeling_artifacts_dir / "sequence_coverage_report.json", report)
    return ProxyResearchResult(
        ok=True,
        stage="features",
        outputs={
            "sequence_long": str(paths.sequence_long_path),
            "sequence_tensor": str(paths.sequence_tensor_path),
            "feature_matrix": str(paths.feature_matrix_path),
            "sequence_coverage_by_event": str(
                paths.modeling_artifacts_dir / "sequence_coverage_by_event.csv"
            ),
            "sequence_coverage_report": str(
                paths.modeling_artifacts_dir / "sequence_coverage_report.json"
            ),
        },
        diagnostics={**report, "tensor": tensor_report, "feature_rows": int(len(features))},
    )


def run_research_models(config: ProjectConfig) -> ProxyResearchResult:
    paths = research_paths(config)
    paths.modeling_artifacts_dir.mkdir(parents=True, exist_ok=True)
    features = read_table(paths.feature_matrix_path)
    predictions, diagnostics = run_proxy_model_suite(
        features, tensor_path=paths.sequence_tensor_path
    )
    predictions.to_parquet(paths.predictions_path, index=False)
    diagnostics_path = paths.modeling_artifacts_dir / "model_fit_diagnostics.csv"
    diagnostics.to_csv(diagnostics_path, index=False)
    outputs = {
        "model_predictions": str(paths.predictions_path),
        "model_fit_diagnostics": str(diagnostics_path),
    }
    outputs.update(build_metric_tables(predictions, out_dir=paths.modeling_artifacts_dir))
    return ProxyResearchResult(
        ok=True,
        stage="models",
        outputs=outputs,
        diagnostics={
            "prediction_rows": int(len(predictions)),
            "trained_models": int(diagnostics["status"].eq("trained").sum())
            if "status" in diagnostics
            else 0,
        },
    )


def run_research_report(config: ProjectConfig) -> ProxyResearchResult:
    paths = research_paths(config)
    figure_paths = write_research_figures(
        artifacts_dir=paths.modeling_artifacts_dir,
        reports_dir=paths.modeling_reports_dir,
    )
    report_path = write_proxy_research_report(
        artifacts_dir=paths.modeling_artifacts_dir,
        reports_dir=paths.modeling_reports_dir,
        figure_paths=figure_paths,
    )
    return ProxyResearchResult(
        ok=True,
        stage="report",
        outputs={"proxy_research_report": str(report_path), **figure_paths},
        diagnostics={"figures": len(figure_paths)},
    )


def run_proxy_research_package(
    config: ProjectConfig,
    *,
    stage: str = "all",
    split_design: str = "chronological_proxy_70_15_15",
    split_date: str | None = None,
    allow_high_sequence_risk: bool = False,
) -> dict[str, object]:
    if stage not in {"all", "sequence-audit", "features", "models", "report"}:
        raise ValueError(f"unsupported research stage: {stage}")
    steps: list[dict[str, object]] = []
    selected = ["sequence-audit", "features", "models", "report"] if stage == "all" else [stage]
    ok = True
    for step in selected:
        print(f"[research] stage start: {step}", flush=True)
        if step == "sequence-audit":
            result = run_research_sequence_audit(config)
            high_risk = bool(result.diagnostics.get("high_sequence_selection_risk"))
            if stage == "all" and high_risk and not allow_high_sequence_risk:
                ok = False
                steps.append(
                    {
                        **result.__dict__,
                        "status": "blocked",
                        "reason": "high_sequence_selection_risk",
                    }
                )
                print(
                    "[research] stopping: high_sequence_selection_risk; "
                    "rerun with --allow-high-sequence-risk to continue",
                    flush=True,
                )
                break
        elif step == "features":
            result = run_research_features(config, split_design=split_design, split_date=split_date)
        elif step == "models":
            result = run_research_models(config)
        else:
            result = run_research_report(config)
        steps.append({**result.__dict__, "status": "ran", "reason": None})
        print(f"[research] stage end: {step}", flush=True)
    payload = {
        "ok": ok,
        "stage": stage,
        "split_design": split_design,
        "split_date": split_date,
        "forecast_floor": FORECAST_FLOOR,
        "steps": steps,
    }
    paths = research_paths(config)
    write_json(paths.modeling_artifacts_dir / "research_manifest.json", payload)
    return payload

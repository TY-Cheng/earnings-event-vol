from __future__ import annotations

import hashlib
import json
import math
import re
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import Any, Literal, cast

import numpy as np
import pandas as pd
import polars as pl
import torch
from scipy.optimize import brentq

from earnings_event_vol.backtest import black_scholes_price, build_proxy_strategy_frame
from earnings_event_vol.config import ProjectConfig
from earnings_event_vol.event_targets import available_target_columns, target_label_column
from earnings_event_vol.features import build_model_feature_matrix
from earnings_event_vol.market_covariates import (
    VIX_ALIGNMENT_PRIOR_CLOSE,
    build_vix_features,
)
from earnings_event_vol.market_index_proxy import MARKET_INDEX_DAILY_SURFACE_FEATURES
from earnings_event_vol.massive import parse_massive_option_ticker
from earnings_event_vol.metrics import (
    auc_score,
    breakdown_metrics,
    cost_sensitivity,
    edge_decile_table,
    forecast_metrics,
    qlike_loss,
    ranking_metrics,
    strategy_metrics,
)
from earnings_event_vol.models import (
    AttentionPoolingSequenceEncoder,
    BiGRUSequenceEncoder,
    DilatedCNNSequenceEncoder,
    FTTransformerRegressor,
    LinearElasticNetRegressor,
    MambaSSMSequenceEncoder,
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
HYBRID_DAILY_STEPS = 19
HYBRID_INTRADAY_STEPS = 12
HYBRID_STEPS = HYBRID_DAILY_STEPS + HYBRID_INTRADAY_STEPS
SEQUENCE_MIN_VALID_DAYS = 12
SEQUENCE_LATEST_DAYS = 5
BASE_OPTION_SURFACE_FEATURE_NAMES = [
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
]
MARKET_INDEX_DAILY_SEQUENCE_FEATURE_NAMES = [
    f"{symbol.lower()}_{feature}"
    for symbol in ("SPY", "QQQ")
    for feature in MARKET_INDEX_DAILY_SURFACE_FEATURES
]
SEQUENCE_FEATURE_NAMES = [
    *BASE_OPTION_SURFACE_FEATURE_NAMES,
    "spy_return_1d",
    "qqq_return_1d",
    *MARKET_INDEX_DAILY_SEQUENCE_FEATURE_NAMES,
    "vix_level",
    "vix_change_1d",
    "vix_change_5d",
    "vix_percentile_252d",
    "vix_above_30",
]
HYBRID_SEQUENCE_FEATURE_NAMES = [
    "atm_iv_proxy",
    "event_ivar_proxy",
    "term_slope_proxy",
    "skew_proxy",
    "butterfly_proxy",
    "straddle_premium_to_spot",
    "valid_pair_count",
    "surface_missing_rate",
    "option_volume_sum",
    "option_transactions_sum",
    "underlying_return_in_bin",
    "underlying_volume_sum",
    "latest_option_trade_bar_age_seconds",
    "underlying_bar_age_seconds",
    "is_intraday_bin",
    "step_type_intraday",
    "log_delta_minutes_from_prev_step",
    "normalized_time_to_entry",
    "hours_until_announcement_proxy",
    "iv_extraction_source_daily_close_trade",
    "iv_extraction_source_intraday_5min_last_trade",
]
HYBRID_SURFACE_VALUE_FEATURE_NAMES = [
    feature
    for feature in HYBRID_SEQUENCE_FEATURE_NAMES
    if feature
    not in {
        "is_intraday_bin",
        "step_type_intraday",
        "log_delta_minutes_from_prev_step",
        "normalized_time_to_entry",
        "hours_until_announcement_proxy",
        "iv_extraction_source_daily_close_trade",
        "iv_extraction_source_intraday_5min_last_trade",
    }
]
TARGET_IDS = ["jump_c2o", "day_c2c", "reaction_o2c"]
PHASE1_TARGET_IDS = TARGET_IDS
PHASE2_TARGET_IDS = TARGET_IDS
RETIRED_MAMBA_MODEL_IDS = [
    "daily_mamba_20step",
    "hybrid_mamba_31step",
    "intraday_only_mamba_12step",
    "mask_only_hybrid_mamba",
]
MODEL_IDS = [
    "market_implied_event_variance",
    "last_four_rvar",
    "last_four_ivar",
    "goyal_saretto_rv_iv_spread",
    "linear_elastic_net",
    "lightgbm",
    "xgboost",
    "lightgbm_xgboost_mean_ensemble",
    "ft_transformer",
    "ridge_flat_aggregates_sequence",
    "bigru_sequence",
    "mamba_ssm_sequence",
    "mask_only_sequence",
    "time_shuffle_sequence",
]
TUNED_TABULAR_MODEL_IDS = [
    "linear_elastic_net_tuned",
    "lightgbm_tuned",
    "xgboost_tuned",
    "ft_transformer_tuned",
]
SEQUENCE_ENSEMBLE_MODEL_IDS = [
    "bigru_sequence_5seed",
    "mamba_ssm_sequence_5seed",
]
PHASE2_SEQUENCE_MODEL_IDS = [
    "attention_pooling_sequence",
    "dilated_cnn_sequence",
]
DETERMINISTIC_MODEL_IDS = {
    "market_implied_event_variance",
    "last_four_rvar",
    "last_four_ivar",
    "goyal_saretto_rv_iv_spread",
}
TRAINABLE_TABULAR_MODEL_IDS = {
    "linear_elastic_net",
    "lightgbm",
    "xgboost",
    "ft_transformer",
    *TUNED_TABULAR_MODEL_IDS,
}
GBDT_MODEL_IDS = {"lightgbm", "xgboost", "lightgbm_tuned", "xgboost_tuned"}
SEQUENCE_MODEL_IDS = {
    "ridge_flat_aggregates_sequence",
    "attention_pooling_sequence",
    "bigru_sequence",
    "dilated_cnn_sequence",
    "mamba_ssm_sequence",
    "bigru_sequence_5seed",
    "mamba_ssm_sequence_5seed",
    "mask_only_sequence",
    "time_shuffle_sequence",
}
SEQUENCE_CONTROL_MODEL_IDS = {"mask_only_sequence", "time_shuffle_sequence"}
REAL_SEQUENCE_MODEL_IDS = (
    SEQUENCE_MODEL_IDS - SEQUENCE_CONTROL_MODEL_IDS - {"ridge_flat_aggregates_sequence"}
)
SplitName = Literal["train", "validation", "test"]
TuningProfile = Literal["untuned", "tuned_phase1"]
TUNING_PROFILES = {"untuned", "tuned_phase1"}
TUNING_SELECTION_TARGET_ID = "jump_c2o"
TUNING_LIGHTGBM_TRIALS = 50
TUNING_XGBOOST_TRIALS = 50
TUNING_FT_TRANSFORMER_TRIALS = 30
TUNING_SEQUENCE_ENSEMBLE_SEEDS = (17, 42, 123, 456, 789)


@dataclass
class TuningState:
    profile: TuningProfile = "untuned"
    seed: int = 17
    selected_params: dict[str, dict[str, object]] | None = None
    trial_records: list[dict[str, object]] | None = None

    def __post_init__(self) -> None:
        if self.selected_params is None:
            self.selected_params = {}
        if self.trial_records is None:
            self.trial_records = []

    @property
    def selected(self) -> dict[str, dict[str, object]]:
        if self.selected_params is None:
            self.selected_params = {}
        return self.selected_params

    @property
    def trials(self) -> list[dict[str, object]]:
        if self.trial_records is None:
            self.trial_records = []
        return self.trial_records


@dataclass(frozen=True)
class ResearchPaths:
    artifacts_dir: Path
    modeling_artifacts_dir: Path
    reports_dir: Path
    modeling_reports_dir: Path
    feature_matrix_path: Path
    sequence_long_path: Path
    sequence_tensor_path: Path
    hybrid_sequence_long_path: Path
    hybrid_sequence_tensor_path: Path
    hybrid_sequence_tensor_v2_path: Path
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
        hybrid_sequence_long_path=config.silver_data_dir
        / "modeling"
        / "option_proxy_surface_hybrid_sequence_long.parquet",
        hybrid_sequence_tensor_path=config.gold_data_dir
        / "modeling"
        / "hybrid_sequence_tensor.npz",
        hybrid_sequence_tensor_v2_path=config.gold_data_dir
        / "modeling"
        / "hybrid_sequence_tensor_v2.npz",
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


def _safe_second_partition_value(value: str) -> str:
    return "".join(char if char.isalnum() else "_" for char in value)


def _option_second_agg_cache_path(
    config: ProjectConfig,
    *,
    option_ticker: str,
    entry_date: date,
    cutoff_timestamp: pd.Timestamp,
    buffer_minutes: int = 60,
) -> Path:
    cutoff = cutoff_timestamp.tz_convert("America/New_York").strftime("%H%M")
    return (
        config.bronze_data_dir
        / "massive"
        / "options_second_aggs"
        / f"date={entry_date.isoformat()}"
        / f"cutoff={cutoff}"
        / f"buffer_minutes={buffer_minutes}"
        / f"options_ticker={_safe_second_partition_value(option_ticker)}"
        / "part.parquet"
    )


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
    qqq = (
        out.loc[out["ticker"].eq("QQQ"), ["source_date", "underlying_return_1d"]]
        .rename(columns={"underlying_return_1d": "qqq_return_1d"})
        .drop_duplicates("source_date")
    )
    out = out.merge(spy, on="source_date", how="left")
    out = out.merge(qqq, on="source_date", how="left")
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
            **{feature: np.nan for feature in BASE_OPTION_SURFACE_FEATURE_NAMES},
            "is_valid_sequence_day": False,
        }
    frame = options.loc[
        options["ticker"].astype(str).str.upper().eq(ticker)
        & pd.to_numeric(options["dte"], errors="coerce").between(3, 45, inclusive="both")
    ].copy()
    if frame.empty:
        return {
            **base,
            **{feature: np.nan for feature in BASE_OPTION_SURFACE_FEATURE_NAMES},
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
            **{feature: np.nan for feature in BASE_OPTION_SURFACE_FEATURE_NAMES},
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


def _read_market_covariates(config: ProjectConfig) -> pd.DataFrame:
    path = config.silver_data_dir / "market_covariates" / "daily_market_covariates.parquet"
    if not path.exists() or path.stat().st_size <= 0:
        return pd.DataFrame()
    return pd.read_parquet(path)


def _read_market_second_covariates(config: ProjectConfig) -> pd.DataFrame:
    path = config.silver_data_dir / "market_covariates" / "market_second_covariates.parquet"
    if not path.exists() or path.stat().st_size <= 0:
        return pd.DataFrame()
    return pd.read_parquet(path)


def _vix_columns_for_merge() -> list[str]:
    return [
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


def _scan_market_index_options(
    path: Path,
    *,
    symbols: Sequence[str],
    columns: Sequence[str],
) -> pd.DataFrame:
    if not path.exists() or not symbols:
        return pd.DataFrame(columns=list(columns))
    pattern = r"^O:(" + "|".join(re.escape(symbol) for symbol in symbols) + r")\d{6}[CP]\d{8}$"
    return (
        pl.scan_parquet(path, cast_options=pl.ScanCastOptions(integer_cast="allow-float"))
        .filter(pl.col("ticker").str.contains(pattern))
        .select(list(columns))
        .collect()
        .to_pandas()
    )


def _market_index_daily_surface_rows(
    raw_options: pd.DataFrame,
    *,
    source_date: date,
    spots: Mapping[str, float],
) -> dict[str, dict[str, object]]:
    if raw_options.empty:
        return {}
    parsed = _parse_filtered_options(raw_options, source_date=source_date)
    out: dict[str, dict[str, object]] = {}
    for symbol in ("SPY", "QQQ"):
        spot = spots.get(symbol, np.nan)
        surface = _compute_daily_surface(
            parsed,
            ticker=symbol,
            source_date=source_date,
            spot=float(spot) if spot is not None and pd.notna(spot) else np.nan,
            event_entry_date=source_date,
        )
        prefixed: dict[str, object] = {}
        for feature in MARKET_INDEX_DAILY_SURFACE_FEATURES:
            prefixed[f"{symbol.lower()}_{feature}"] = surface.get(feature, np.nan)
        out[symbol] = prefixed
    return out


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
            ticker_values=needed_tickers + ["SPY", "QQQ"],
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
        market_index_raw_options = _scan_market_index_options(
            options_root / f"date={source_date.isoformat()}" / "part.parquet",
            symbols=["SPY", "QQQ"],
            columns=["ticker", "close", "volume", "transactions"],
        )
        market_index_surfaces = _market_index_daily_surface_rows(
            market_index_raw_options,
            source_date=source_date,
            spots={
                symbol: float(spot_by_ticker.get(symbol, np.nan))
                for symbol in ("SPY", "QQQ")
                if spot_by_ticker.get(symbol) is not None
            },
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
            for surface in market_index_surfaces.values():
                row.update(surface)
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
                "qqq_return_1d",
            ]
        )
    )
    long_rows = plan.merge(surface, on=["event_id", "ticker", "source_date"], how="left")
    long_rows = long_rows.merge(
        underlying_features[
            [
                "ticker",
                "source_date",
                "close",
                "underlying_return_1d",
                "rv5",
                "spy_return_1d",
                "qqq_return_1d",
            ]
        ].rename(columns={"close": "underlying_close"}),
        on=["ticker", "source_date"],
        how="left",
    )
    market_covariates = _read_market_covariates(config)
    if not market_covariates.empty:
        vix_lookup = (
            long_rows.reset_index(names="_vix_row_id")[["_vix_row_id", "source_date"]]
            .rename(columns={"source_date": "feature_asof_date"})
            .copy()
        )
        vix_features = build_vix_features(
            market_covariates,
            vix_lookup,
            alignment=VIX_ALIGNMENT_PRIOR_CLOSE,
        ).set_index("_vix_row_id")
        for column in _vix_columns_for_merge():
            long_rows[column] = vix_features[column].reindex(range(len(long_rows))).to_numpy()
    for column in _vix_columns_for_merge():
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
    report["vix_available"] = bool(
        "vix_available" in long_rows and long_rows["vix_available"].fillna(False).astype(bool).any()
    )
    report["vix_regime_unavailable"] = not bool(
        "vix_regime_tercile" in long_rows and long_rows["vix_regime_tercile"].notna().any()
    )
    report["vix_alignment"] = VIX_ALIGNMENT_PRIOR_CLOSE
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
                "sequence_eligible_v2",
                "sequence_eligibility_reason",
            ]
        )
    frame = long_rows.copy()
    frame["is_latest_5"] = frame["seq_index"].ge(total_sequence_days - SEQUENCE_LATEST_DAYS)
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
    out["sequence_eligible_v2"] = out["valid_sequence_days"].ge(SEQUENCE_MIN_VALID_DAYS) & out[
        "valid_latest_5_days"
    ].ge(1)
    out["sequence_eligibility_reason"] = np.where(
        out["sequence_eligible_v2"],
        "eligible",
        np.where(
            out["valid_sequence_days"].lt(SEQUENCE_MIN_VALID_DAYS),
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
    eligible = (
        int(by_event["sequence_eligible_v2"].sum()) if "sequence_eligible_v2" in by_event else 0
    )
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
        "default_min_valid_days": SEQUENCE_MIN_VALID_DAYS,
        "default_latest_days": SEQUENCE_LATEST_DAYS,
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


def _event_timestamp(value: object) -> pd.Timestamp | None:
    if value is None or pd.isna(value):
        return None
    timestamp = pd.Timestamp(value)
    if timestamp.tzinfo is None:
        timestamp = timestamp.tz_localize("America/New_York")
    return timestamp.tz_convert("America/New_York")


def _announcement_proxy_timestamp(record: Mapping[str, object]) -> tuple[pd.Timestamp | None, str]:
    value = record.get("announcement_proxy_timestamp") or record.get("source_timestamp")
    timestamp = _event_timestamp(value)
    if timestamp is None:
        return None, "unavailable"
    source = str(record.get("announcement_proxy_source") or "sec_acceptance_timestamp")
    return timestamp, source


def _calendar_minutes(later: pd.Timestamp, earlier: pd.Timestamp) -> float:
    return float((later - earlier).total_seconds() / 60.0)


def _compute_hybrid_time_features(frame: pd.DataFrame) -> pd.DataFrame:
    if frame.empty:
        return frame
    out = frame.copy()
    out["source_timestamp"] = pd.to_datetime(out["source_timestamp"], errors="coerce", utc=True)
    out["event_entry_timestamp"] = pd.to_datetime(
        out["event_entry_timestamp"], errors="coerce", utc=True
    )
    rows: list[pd.DataFrame] = []
    for _event_id, group in out.sort_values(["event_id", "seq_index"]).groupby("event_id"):
        part = group.copy()
        source_ts = part["source_timestamp"]
        previous = source_ts.shift(1)
        delta_minutes = (source_ts - previous).dt.total_seconds() / 60.0
        delta_minutes = delta_minutes.fillna(1440.0).clip(lower=1.0)
        part["log_delta_minutes_from_prev_step"] = np.log1p(delta_minutes)
        entry_ts = part["event_entry_timestamp"]
        minutes_to_entry = (source_ts - entry_ts).dt.total_seconds() / 60.0
        part["normalized_time_to_entry"] = minutes_to_entry / float(20 * 24 * 60)
        rows.append(part)
    return pd.concat(rows, ignore_index=True)


def _daily_hybrid_rows(daily_long: pd.DataFrame, events: pd.DataFrame) -> pd.DataFrame:
    if daily_long.empty:
        return pd.DataFrame()
    event_meta = ensure_event_id(events).set_index("event_id").to_dict("index")
    rows: list[dict[str, object]] = []
    for event_id, group in daily_long.groupby("event_id", dropna=False):
        meta = event_meta.get(str(event_id), {})
        entry_date = pd.Timestamp(meta.get("entry_date") or group["entry_date"].iloc[0]).date()
        eligible = group.loc[pd.to_datetime(group["source_date"]).dt.date < entry_date].copy()
        eligible = eligible.sort_values("source_date").tail(HYBRID_DAILY_STEPS)
        start = HYBRID_DAILY_STEPS - len(eligible)
        proxy_ts, proxy_source = _announcement_proxy_timestamp(meta)
        for offset, record in enumerate(eligible.to_dict("records")):
            seq_index = start + offset
            source_date = pd.Timestamp(record["source_date"]).date()
            source_timestamp = pd.Timestamp(source_date, tz="America/New_York") + pd.Timedelta(
                hours=16
            )
            hours_until = (
                None
                if proxy_ts is None
                else float((proxy_ts - source_timestamp).total_seconds() / 3600.0)
            )
            rows.append(
                {
                    **record,
                    "seq_index": int(seq_index),
                    "source_timestamp": source_timestamp.isoformat(),
                    "is_intraday_bin": 0.0,
                    "step_type": "daily",
                    "step_type_intraday": 0.0,
                    "iv_extraction_source": "daily_close_trade",
                    "iv_extraction_source_daily_close_trade": 1.0,
                    "iv_extraction_source_intraday_5min_last_trade": 0.0,
                    "skew_proxy": record.get("iv_skew_proxy"),
                    "butterfly_proxy": record.get("iv_butterfly_proxy"),
                    "underlying_return_in_bin": record.get("underlying_return_1d"),
                    "underlying_volume_sum": np.nan,
                    "latest_option_trade_bar_age_seconds": np.nan,
                    "underlying_bar_age_seconds": np.nan,
                    "hours_until_announcement_proxy": hours_until,
                    "announcement_proxy_source": proxy_source,
                    "hybrid_valid_step": bool(record.get("is_valid_sequence_day", False)),
                    "intraday_window_spec": "daily_prior_19",
                    "underlying_spot_source": "daily_close",
                }
            )
    return pd.DataFrame(rows)


def _load_second_bars_for_contracts(
    config: ProjectConfig,
    *,
    contracts: pd.DataFrame,
    event: Mapping[str, object],
    buffer_minutes: int = 60,
) -> pd.DataFrame:
    if contracts.empty:
        return pd.DataFrame()
    entry_date = pd.Timestamp(event["entry_date"]).date()
    cutoff = _event_timestamp(event.get("event_entry_timestamp"))
    if cutoff is None:
        return pd.DataFrame()
    frames: list[pd.DataFrame] = []
    for row in contracts.to_dict("records"):
        path = _option_second_agg_cache_path(
            config,
            option_ticker=str(row["options_ticker"]),
            entry_date=entry_date,
            cutoff_timestamp=cutoff,
            buffer_minutes=buffer_minutes,
        )
        if not path.exists() or path.stat().st_size <= 0:
            continue
        try:
            bars = pd.read_parquet(path)
        except Exception:
            continue
        for column, value in row.items():
            if column not in bars.columns:
                bars[column] = value
        frames.append(bars)
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def _intraday_bin_surface(
    bars: pd.DataFrame,
    *,
    spot: float,
    source_date: date,
    bin_start: pd.Timestamp,
    bin_end: pd.Timestamp,
) -> dict[str, object]:
    base = {
        "surface_source": "options_second_aggs",
        "iv_source": "intraday_5min_last_trade",
        "panel_grade": "no_nbbo_trade_proxy",
        "underlying_spot_source": "s_before_fallback",
        "underlying_bar_age_seconds": np.nan,
        "underlying_return_in_bin": np.nan,
        "underlying_volume_sum": np.nan,
    }
    if bars.empty or not np.isfinite(spot) or spot <= 0:
        return {
            **base,
            **{feature: np.nan for feature in HYBRID_SURFACE_VALUE_FEATURE_NAMES},
            "valid_pair_count": 0,
            "surface_missing_rate": 1.0,
            "hybrid_valid_step": False,
        }
    frame = bars.copy()
    frame["timestamp_et"] = pd.to_datetime(frame["timestamp_et"], errors="coerce")
    if frame["timestamp_et"].dt.tz is None:
        frame["timestamp_et"] = frame["timestamp_et"].dt.tz_localize("America/New_York")
    frame["timestamp_et"] = frame["timestamp_et"].dt.tz_convert("America/New_York")
    frame = frame.loc[frame["timestamp_et"].between(bin_start, bin_end, inclusive="both")].copy()
    if frame.empty:
        return {
            **base,
            **{feature: np.nan for feature in HYBRID_SURFACE_VALUE_FEATURE_NAMES},
            "valid_pair_count": 0,
            "surface_missing_rate": 1.0,
            "hybrid_valid_step": False,
        }
    frame = (
        frame.sort_values("timestamp_et").groupby("options_ticker", as_index=False).tail(1).copy()
    )
    frame["strike"] = pd.to_numeric(frame["strike"], errors="coerce")
    frame["option_price"] = pd.to_numeric(
        frame.get("option_vwap", frame.get("option_close")), errors="coerce"
    )
    frame["moneyness_abs"] = (frame["strike"] / spot - 1.0).abs()
    frame = frame.loc[frame["moneyness_abs"].le(0.05) & frame["option_price"].gt(0)].copy()
    if frame.empty:
        return {
            **base,
            **{feature: np.nan for feature in HYBRID_SURFACE_VALUE_FEATURE_NAMES},
            "valid_pair_count": 0,
            "surface_missing_rate": 1.0,
            "hybrid_valid_step": False,
        }
    frame["dte"] = (pd.to_datetime(frame["expiration"]).dt.date - source_date).map(
        lambda delta: delta.days
    )
    ivs: list[float | None] = []
    for record in frame.to_dict("records"):
        ivs.append(
            _implied_volatility(
                spot=spot,
                strike=float(record["strike"]),
                time_to_expiry=max(float(record["dte"]) / 365.0, 1.0 / 365.0),
                option_price=float(record["option_price"]),
                right=str(record["right"]),
            )
        )
    frame["iv_proxy"] = ivs
    frame = frame.dropna(subset=["iv_proxy"])
    pair_rows = frame.loc[
        frame.groupby(["expiration", "strike"])["right"].transform("nunique").ge(2)
    ].copy()
    valid_pair_count = int(pair_rows.groupby(["expiration", "strike"]).ngroups)
    if pair_rows.empty:
        return {
            **base,
            **{feature: np.nan for feature in HYBRID_SURFACE_VALUE_FEATURE_NAMES},
            "option_volume_sum": float(frame.get("volume", pd.Series(dtype=float)).sum()),
            "option_transactions_sum": float(
                frame.get("transactions", pd.Series(dtype=float)).sum()
            ),
            "valid_pair_count": 0,
            "surface_missing_rate": 1.0,
            "hybrid_valid_step": False,
        }
    atm_key = (
        pair_rows.groupby(["expiration", "strike"], as_index=False)
        .agg(moneyness_abs=("moneyness_abs", "mean"))
        .sort_values("moneyness_abs")
        .head(1)
    )
    expiration = atm_key.iloc[0]["expiration"]
    strike = atm_key.iloc[0]["strike"]
    atm = pair_rows.loc[pair_rows["expiration"].eq(expiration) & pair_rows["strike"].eq(strike)]
    call = atm.loc[atm["right"].astype(str).eq("call")]
    put = atm.loc[atm["right"].astype(str).eq("put")]
    atm_iv_proxy = float(atm["iv_proxy"].mean())
    skew_proxy = (
        np.nan
        if call.empty or put.empty
        else float(put["iv_proxy"].mean() - call["iv_proxy"].mean())
    )
    straddle_premium_to_spot = (
        np.nan
        if call.empty or put.empty
        else float((call["option_price"].mean() + put["option_price"].mean()) / spot)
    )
    near = pair_rows.loc[pair_rows["dte"].between(3, 14, inclusive="both")]
    far = pair_rows.loc[pair_rows["dte"].between(15, 45, inclusive="both")]
    term_slope_proxy = (
        np.nan
        if near.empty or far.empty
        else float(far["iv_proxy"].mean() - near["iv_proxy"].mean())
    )
    by_strike = pair_rows.groupby("strike", as_index=False)["iv_proxy"].mean().sort_values("strike")
    butterfly_proxy = np.nan
    if len(by_strike) >= 3:
        middle = int(np.argmin(np.abs(by_strike["strike"].to_numpy(dtype=float) - spot)))
        low = max(0, middle - 1)
        high = min(len(by_strike) - 1, middle + 1)
        if low != middle and high != middle:
            butterfly_proxy = float(
                by_strike.iloc[low]["iv_proxy"]
                + by_strike.iloc[high]["iv_proxy"]
                - 2.0 * by_strike.iloc[middle]["iv_proxy"]
            )
    event_ivar_proxy = float(max(atm_iv_proxy, 0.0) ** 2 / 365.0)
    age_seconds = float((bin_end - frame["timestamp_et"].max()).total_seconds())
    return {
        **base,
        "atm_iv_proxy": atm_iv_proxy,
        "event_ivar_proxy": event_ivar_proxy,
        "term_slope_proxy": term_slope_proxy,
        "skew_proxy": skew_proxy,
        "butterfly_proxy": butterfly_proxy,
        "straddle_premium_to_spot": straddle_premium_to_spot,
        "valid_pair_count": valid_pair_count,
        "surface_missing_rate": float(max(0.0, 1.0 - min(valid_pair_count, 10) / 10.0)),
        "option_volume_sum": float(pd.to_numeric(frame["volume"], errors="coerce").sum()),
        "option_transactions_sum": float(
            pd.to_numeric(frame["transactions"], errors="coerce").sum()
        ),
        "latest_option_trade_bar_age_seconds": age_seconds,
        "hybrid_valid_step": bool(valid_pair_count > 0 and np.isfinite(atm_iv_proxy)),
    }


def _intraday_hybrid_rows(
    events: pd.DataFrame,
    *,
    config: ProjectConfig,
    buffer_minutes: int = 60,
) -> pd.DataFrame:
    events = ensure_event_id(events)
    candidates = _load_sequence_contract_candidates(config, events["event_id"].astype(str).tolist())
    if candidates.empty:
        return pd.DataFrame()
    rows: list[dict[str, object]] = []
    candidates_by_event = {
        str(event_id): group.copy() for event_id, group in candidates.groupby("event_id")
    }
    for event in events.to_dict("records"):
        event_id = str(event["event_id"])
        cutoff = _event_timestamp(event.get("event_entry_timestamp"))
        entry_date = pd.Timestamp(event.get("entry_date")).date()
        if cutoff is None:
            continue
        contracts = candidates_by_event.get(event_id, pd.DataFrame())
        bars = _load_second_bars_for_contracts(
            config,
            contracts=contracts,
            event=event,
            buffer_minutes=buffer_minutes,
        )
        spot = float(event.get("s_before", np.nan))
        proxy_ts, proxy_source = _announcement_proxy_timestamp(event)
        for idx in range(HYBRID_INTRADAY_STEPS):
            bin_start = cutoff - pd.Timedelta(minutes=60 - idx * 5)
            bin_end = bin_start + pd.Timedelta(minutes=5)
            if idx == HYBRID_INTRADAY_STEPS - 1:
                bin_end = cutoff
            surface = _intraday_bin_surface(
                bars,
                spot=spot,
                source_date=entry_date,
                bin_start=bin_start,
                bin_end=bin_end,
            )
            hours_until = (
                None if proxy_ts is None else float((proxy_ts - bin_end).total_seconds() / 3600.0)
            )
            rows.append(
                {
                    "event_id": event_id,
                    "ticker": str(event.get("ticker", "")).upper(),
                    "entry_date": entry_date,
                    "exit_date": pd.Timestamp(event.get("exit_date")).date()
                    if event.get("exit_date") is not None and not pd.isna(event.get("exit_date"))
                    else None,
                    "event_entry_timestamp": cutoff.isoformat(),
                    "source_date": entry_date,
                    "source_timestamp": bin_end.isoformat(),
                    "seq_index": HYBRID_DAILY_STEPS + idx,
                    "is_intraday_bin": 1.0,
                    "step_type": "intraday",
                    "step_type_intraday": 1.0,
                    "iv_extraction_source": "intraday_5min_last_trade",
                    "iv_extraction_source_daily_close_trade": 0.0,
                    "iv_extraction_source_intraday_5min_last_trade": 1.0,
                    "hours_until_announcement_proxy": hours_until,
                    "announcement_proxy_source": proxy_source,
                    "intraday_window_spec": "preclose_60_0",
                    **surface,
                }
            )
    return pd.DataFrame(rows)


def build_hybrid_proxy_sequence_long(
    daily_long: pd.DataFrame,
    events: pd.DataFrame,
    *,
    config: ProjectConfig,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, object]]:
    daily = _daily_hybrid_rows(daily_long, events)
    intraday = _intraday_hybrid_rows(events, config=config)
    hybrid = pd.concat([daily, intraday], ignore_index=True, sort=False)
    if hybrid.empty:
        by_event = pd.DataFrame(columns=["event_id"])
        return hybrid, by_event, hybrid_sequence_report(by_event, total_events=len(events))
    hybrid = _compute_hybrid_time_features(hybrid)
    hybrid = hybrid.sort_values(["event_id", "seq_index"]).reset_index(drop=True)
    by_event = hybrid_sequence_coverage_by_event(hybrid)
    report = hybrid_sequence_report(
        by_event, total_events=ensure_event_id(events)["event_id"].nunique()
    )
    return hybrid, by_event, report


def hybrid_sequence_coverage_by_event(hybrid: pd.DataFrame) -> pd.DataFrame:
    if hybrid.empty:
        return pd.DataFrame(columns=["event_id"])
    numeric_features = [feature for feature in HYBRID_SEQUENCE_FEATURE_NAMES if feature in hybrid]
    rows: list[dict[str, object]] = []
    for event_id, group in hybrid.groupby("event_id", dropna=False):
        intraday = group.loc[group["is_intraday_bin"].fillna(0).astype(float).gt(0)]
        valid_intraday = int(intraday["hybrid_valid_step"].fillna(False).astype(bool).sum())
        feature_values = group[numeric_features].apply(pd.to_numeric, errors="coerce")
        density = (
            float(np.isfinite(feature_values.to_numpy(dtype=float)).mean())
            if not feature_values.empty
            else 0.0
        )
        rows.append(
            {
                "event_id": event_id,
                "hybrid_steps": int(len(group)),
                "intraday_valid_bin_count": valid_intraday,
                "latest_5min_valid_surface": bool(
                    not intraday.empty
                    and bool(intraday.sort_values("seq_index").tail(1)["hybrid_valid_step"].iloc[0])
                ),
                "hybrid_feature_mask_density": density,
                "hybrid_sequence_eligible_v2": bool(valid_intraday >= 8 and density >= 0.50),
            }
        )
    return pd.DataFrame(rows)


def hybrid_sequence_report(by_event: pd.DataFrame, *, total_events: int) -> dict[str, object]:
    if by_event.empty:
        return {
            "total_events": int(total_events),
            "events_with_8_valid_intraday_bins": 0,
            "median_hybrid_feature_mask_density": 0.0,
            "hybrid_sequence_too_sparse": True,
            "intraday_window_spec": "preclose_60_0",
            "surface_wording": "trade_aggregate_proxy",
        }
    events_with_8 = int(by_event["intraday_valid_bin_count"].ge(8).sum())
    median_density = float(by_event["hybrid_feature_mask_density"].median())
    sparse = events_with_8 < math.ceil(0.70 * max(1, total_events)) or median_density < 0.50
    return {
        "total_events": int(total_events),
        "events_with_8_valid_intraday_bins": events_with_8,
        "median_hybrid_feature_mask_density": median_density,
        "hybrid_sequence_too_sparse": bool(sparse),
        "intraday_window_spec": "preclose_60_0",
        "closing_auction_caveat": (
            "The final 30 minutes may contain MOC, benchmark, and closing-auction "
            "microstructure unrelated to earnings positioning."
        ),
        "surface_wording": "trade_aggregate_proxy",
    }


def proxy_surface_distribution_audit(long_rows: pd.DataFrame) -> pd.DataFrame:
    if long_rows.empty or "iv_extraction_source" not in long_rows:
        return pd.DataFrame()
    metrics = [
        "atm_iv_proxy",
        "event_ivar_proxy",
        "term_slope_proxy",
        "skew_proxy",
        "butterfly_proxy",
        "straddle_premium_to_spot",
        "valid_pair_count",
        "surface_missing_rate",
        "option_volume_sum",
        "option_transactions_sum",
        "latest_option_trade_bar_age_seconds",
    ]
    rows: list[dict[str, object]] = []
    for source, group in long_rows.groupby("iv_extraction_source", dropna=False):
        for metric in metrics:
            if metric not in group:
                continue
            values = pd.to_numeric(group[metric], errors="coerce").dropna()
            rows.append(
                {
                    "iv_extraction_source": source,
                    "metric": metric,
                    "n": int(len(values)),
                    "mean": float(values.mean()) if len(values) else np.nan,
                    "std": float(values.std(ddof=0)) if len(values) else np.nan,
                    "p01": float(values.quantile(0.01)) if len(values) else np.nan,
                    "p50": float(values.quantile(0.50)) if len(values) else np.nan,
                    "p99": float(values.quantile(0.99)) if len(values) else np.nan,
                    "missing_rate": float(group[metric].isna().mean()),
                }
            )
    return pd.DataFrame(rows)


def build_sequence_tensor(
    long_rows: pd.DataFrame,
    feature_matrix: pd.DataFrame,
    *,
    out_path: Path,
    feature_names: Sequence[str] = SEQUENCE_FEATURE_NAMES,
    lookback_days: int = LOOKBACK_DAYS,
    per_step_type_scaling: bool = False,
) -> dict[str, object]:
    feature_matrix = ensure_event_id(feature_matrix)
    event_ids = feature_matrix["event_id"].astype(str).tolist()
    event_index = {event_id: idx for idx, event_id in enumerate(event_ids)}
    features = list(feature_names)
    raw = np.full((len(event_ids), lookback_days, len(features)), np.nan, dtype=np.float32)
    source_dates = np.full((len(event_ids), lookback_days), "", dtype=object)
    step_type = np.full((len(event_ids), lookback_days), "daily", dtype=object)
    for row in long_rows.to_dict("records"):
        event_id = str(row["event_id"])
        if event_id not in event_index:
            continue
        time_idx = int(row["seq_index"])
        if time_idx < 0 or time_idx >= lookback_days:
            continue
        source_dates[event_index[event_id], time_idx] = str(row["source_date"])
        step_type[event_index[event_id], time_idx] = str(row.get("step_type") or "daily")
        for feature_idx, feature in enumerate(features):
            value = row.get(feature)
            if (
                value is not None
                and pd.notna(value)
                and any(token in feature for token in ("volume", "count", "age", "transactions"))
            ):
                value = math.log1p(max(float(value), 0.0))
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
        step_types = sorted(set(step_type.reshape(-1))) if per_step_type_scaling else ["all"]
        for current_step_type in step_types:
            type_mask = (
                np.ones(step_type.shape, dtype=bool)
                if current_step_type == "all"
                else step_type == current_step_type
            )
            observed = raw[:, :, feature_idx][train_mask, :]
            observed_mask = type_mask[train_mask, :]
            observed = observed[observed_mask & np.isfinite(observed)]
            center = float(np.median(observed)) if observed.size else 0.0
            if observed.size:
                q75, q25 = np.quantile(observed, [0.75, 0.25])
                scale = float(q75 - q25)
            else:
                scale = 1.0
            if scale <= 1e-12:
                scale = 1.0
            scaled[:, :, feature_idx] = np.where(
                type_mask,
                (scaled[:, :, feature_idx] - center) / scale,
                scaled[:, :, feature_idx],
            )
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
        step_type=step_type,
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
    hybrid_by_event: pd.DataFrame | None = None,
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
            "sequence_eligible_v2",
            "sequence_eligibility_reason",
        ]
        out = out.merge(
            sequence_by_event[[column for column in keep if column in sequence_by_event]],
            on="event_id",
            how="left",
        )
    if sequence_aggregates is not None and not sequence_aggregates.empty:
        out = out.merge(sequence_aggregates, on="event_id", how="left")
    if "sequence_eligible_v2" not in out.columns:
        out["sequence_eligible_v2"] = False
    out["sequence_eligible_v2"] = out["sequence_eligible_v2"].fillna(False).astype(bool)
    if hybrid_by_event is not None and not hybrid_by_event.empty:
        keep = [
            "event_id",
            "intraday_valid_bin_count",
            "latest_5min_valid_surface",
            "hybrid_feature_mask_density",
            "hybrid_sequence_eligible_v2",
        ]
        out = out.merge(
            hybrid_by_event[[column for column in keep if column in hybrid_by_event]],
            on="event_id",
            how="left",
        )
    if "hybrid_sequence_eligible_v2" not in out.columns:
        out["hybrid_sequence_eligible_v2"] = False
    out["hybrid_sequence_eligible_v2"] = (
        out["hybrid_sequence_eligible_v2"].fillna(False).astype(bool)
    )
    if "entry_premium_usd" in out.columns:
        out["proxy_cost_usd"] = proxy_transaction_cost(
            pd.to_numeric(out["entry_premium_usd"], errors="coerce").fillna(0.0)
        )
    else:
        out["proxy_cost_usd"] = np.nan
    for label in ("0_5", "5_15"):
        premium_col = f"open_option_vwap_{label}_anchor_usd"
        cost_col = f"open_option_vwap_{label}_proxy_cost_usd"
        if premium_col in out.columns:
            out[cost_col] = proxy_transaction_cost(
                pd.to_numeric(out[premium_col], errors="coerce").fillna(0.0)
            )
        else:
            out[cost_col] = np.nan
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


def prepare_target_frame(base: pd.DataFrame, *, target_id: str) -> pd.DataFrame:
    out = base.copy()
    label_col = target_label_column(target_id, out)
    if label_col not in out.columns:
        raise ValueError(f"target {target_id} requires missing column {label_col}")
    out["target_id"] = target_id
    out["target_label_column"] = label_col
    out["rvar_event"] = pd.to_numeric(out[label_col], errors="coerce")
    out["edge_var_realized"] = out["rvar_event"] - pd.to_numeric(out["ivar_event"], errors="coerce")
    out[f"edge_var_realized_{target_id}"] = out["edge_var_realized"]
    out["target_has_strategy_pnl"] = target_id == "day_c2c"
    out["target_has_diagnostic_c2o_proxy_pnl"] = target_id == "jump_c2o"
    if target_id == "reaction_o2c":
        out["ivar_baseline_interpretation"] = "weak_comparator_only"
    elif target_id == "jump_c2o":
        out["ivar_baseline_interpretation"] = "conservative_full_event_ivar_benchmark"
    else:
        out["ivar_baseline_interpretation"] = "c2c_literature_compatible"
    return out


def o2c_scale_diagnostic(frame: pd.DataFrame) -> pd.DataFrame:
    target = frame.copy()
    if "target_id" in target.columns:
        target = target.loc[target["target_id"].astype(str).eq("reaction_o2c")].copy()
    required = {"rvar_event", "ivar_event"}
    if not required.issubset(target.columns):
        paired = pd.DataFrame()
    else:
        paired = target.dropna(subset=["rvar_event", "ivar_event"]).copy()
    if paired.empty:
        sd_rvar = np.nan
        sd_ivar = np.nan
        mean_rvar = np.nan
        mean_ivar = np.nan
    else:
        sd_rvar = float(pd.to_numeric(paired["rvar_event"], errors="coerce").std())
        sd_ivar = float(pd.to_numeric(paired["ivar_event"], errors="coerce").std())
        mean_rvar = float(pd.to_numeric(paired["rvar_event"], errors="coerce").mean())
        mean_ivar = float(pd.to_numeric(paired["ivar_event"], errors="coerce").mean())
    return pd.DataFrame(
        [
            {
                "target_id": "reaction_o2c",
                "paired_rows": int(len(paired)),
                "sd_rvar_reaction_o2c": sd_rvar,
                "sd_ivar_event": sd_ivar,
                "sd_ratio_o2c_to_ivar": sd_rvar / sd_ivar
                if np.isfinite(sd_rvar) and np.isfinite(sd_ivar) and sd_ivar != 0
                else np.nan,
                "mean_ratio_o2c_to_ivar": mean_rvar / mean_ivar
                if np.isfinite(mean_rvar) and np.isfinite(mean_ivar) and mean_ivar != 0
                else np.nan,
                "ivar_baseline_interpretation": "weak_full_event_comparator_only",
            }
        ]
    )


def research_prediction_column(model_id: str) -> str:
    mapping = {
        "lightgbm_with_hybrid_aggregates": "forecast_lightgbm_with_hybrid_aggregates",
    }
    return mapping.get(model_id, prediction_column_for_model(model_id))


def _numeric_matrix(frame: pd.DataFrame, columns: Sequence[str]) -> pd.DataFrame:
    return frame[list(columns)].apply(pd.to_numeric, errors="coerce").fillna(0.0).astype(float)


def _combined_train_validation(frame: pd.DataFrame) -> pd.DataFrame:
    return frame.loc[frame["split"].isin(["train", "validation"])].copy()


def _fit_splits(frame: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    train = frame.loc[frame["split"].eq("train")]
    validation = frame.loc[frame["split"].eq("validation")]
    test = frame.loc[frame["split"].eq("test")]
    return train, validation, test


def _validation_tuning_metrics(
    validation: pd.DataFrame,
    *,
    forecast: np.ndarray,
) -> dict[str, float | int | None]:
    scored = validation.copy()
    scored["_forecast_tuned"] = np.maximum(np.asarray(forecast, dtype=float), FORECAST_FLOOR)
    scored["_score_tuned"] = pd.to_numeric(
        scored["_forecast_tuned"], errors="coerce"
    ) - pd.to_numeric(scored["ivar_event"], errors="coerce")
    forecast_values = forecast_metrics(scored, forecast_col="_forecast_tuned")
    ranking_values = ranking_metrics(scored, score_col="_score_tuned")
    return {
        "validation_n": int(forecast_values.get("n") or 0),
        "validation_mae": cast(float | None, forecast_values.get("mae")),
        "validation_rmse": cast(float | None, forecast_values.get("rmse")),
        "validation_auc": cast(float | None, ranking_values.get("auc")),
        "validation_top_decile_precision": cast(
            float | None, ranking_values.get("top_decile_precision")
        ),
    }


def _finite_or_default(value: object, default: float) -> float:
    try:
        candidate = float(cast(float, value))
    except (TypeError, ValueError):
        return default
    return candidate if np.isfinite(candidate) else default


def _tuning_sort_key(metrics: Mapping[str, object]) -> tuple[float, float, float]:
    auc = _finite_or_default(metrics.get("validation_auc"), -1.0)
    top_decile = _finite_or_default(metrics.get("validation_top_decile_precision"), -1.0)
    rmse = _finite_or_default(metrics.get("validation_rmse"), float("inf"))
    return auc, top_decile, -rmse


def _tuning_objective_value(metrics: Mapping[str, object]) -> float:
    auc, top_decile, neg_rmse = _tuning_sort_key(metrics)
    if auc < 0:
        return -1e6
    return float(auc + 1e-3 * max(top_decile, 0.0) + 1e-6 * neg_rmse)


def _require_optuna() -> Any:
    try:
        import optuna
    except ImportError as exc:  # pragma: no cover - optional dependency
        raise RuntimeError("optuna is required for --tuning-profile tuned_phase1") from exc
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    return optuna


def _best_completed_trial(study: Any) -> Any | None:
    trials = [trial for trial in study.trials if trial.value is not None]
    if not trials:
        return None
    return max(trials, key=lambda trial: _tuning_sort_key(trial.user_attrs))


def _record_optuna_trials(
    *,
    state: TuningState,
    study: Any,
    model_id: str,
    target_id: str,
    selected_number: int | None,
) -> None:
    for trial in study.trials:
        attrs = trial.user_attrs
        state.trials.append(
            {
                "model_id": model_id,
                "target_id": target_id,
                "trial_number": int(trial.number),
                "selected": selected_number is not None and int(trial.number) == selected_number,
                "seed": state.seed,
                "params_json": json.dumps(trial.params, sort_keys=True),
                "validation_n": attrs.get("validation_n"),
                "validation_mae": attrs.get("validation_mae"),
                "validation_rmse": attrs.get("validation_rmse"),
                "validation_auc": attrs.get("validation_auc"),
                "validation_top_decile_precision": attrs.get("validation_top_decile_precision"),
                "objective_value": trial.value,
            }
        )


def _selected_key(model_id: str) -> str:
    return model_id


def _cache_selected_params(
    *,
    state: TuningState,
    model_id: str,
    target_id: str,
    params: Mapping[str, object],
    metrics: Mapping[str, object],
) -> dict[str, object]:
    payload: dict[str, object] = {
        "model_id": model_id,
        "selection_target_id": target_id,
        "selection_protocol": "train_validation_only",
        "refit_protocol": "train_plus_validation",
        "primary_metric": "validation_jump_c2o_predicted_edge_auc"
        if target_id == TUNING_SELECTION_TARGET_ID
        else f"validation_{target_id}_predicted_edge_auc_fallback",
        "params": dict(params),
        "validation_metrics": dict(metrics),
    }
    state.selected[_selected_key(model_id)] = payload
    return payload


def _cached_params(state: TuningState, model_id: str) -> dict[str, object] | None:
    payload = state.selected.get(_selected_key(model_id))
    params = None if payload is None else payload.get("params")
    return cast(dict[str, object], params) if isinstance(params, dict) else None


def _param_float(params: Mapping[str, object], key: str) -> float:
    return float(cast(Any, params[key]))


def _param_int(params: Mapping[str, object], key: str, default: int | None = None) -> int:
    raw = params.get(key, default) if default is not None else params[key]
    return int(cast(Any, raw))


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


def _train_elastic_net_tuned(
    frame: pd.DataFrame,
    *,
    features: Sequence[str],
    target_id: str,
    tuning_state: TuningState,
) -> tuple[pd.Series, dict[str, object], object | None]:
    try:
        from sklearn.linear_model import ElasticNet, ElasticNetCV
        from sklearn.model_selection import TimeSeriesSplit
        from sklearn.pipeline import Pipeline
        from sklearn.preprocessing import StandardScaler
    except ImportError:
        return (
            pd.Series(np.nan, index=frame.index),
            {"status": "skipped_dependency_unavailable"},
            None,
        )
    train, validation, test = _fit_splits(frame)
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
    params = _cached_params(tuning_state, "linear_elastic_net_tuned")
    selection_target = cast(
        str,
        tuning_state.selected.get("linear_elastic_net_tuned", {}).get(
            "selection_target_id", target_id
        ),
    )
    if params is None:
        cv_splits = min(5, max(2, len(train_fit) - 1))
        cv_model = Pipeline(
            [
                ("scaler", StandardScaler()),
                (
                    "elastic_net",
                    ElasticNetCV(
                        l1_ratio=[0.1, 0.5, 0.7, 0.9, 0.95, 0.99, 1.0],
                        n_alphas=100,
                        cv=TimeSeriesSplit(n_splits=cv_splits),
                        max_iter=10_000,
                        random_state=tuning_state.seed,
                    ),
                ),
            ]
        )
        cv_model.fit(
            _numeric_matrix(train_fit, features),
            pd.to_numeric(train_fit["rvar_event"], errors="coerce").to_numpy(dtype=float),
        )
        val_pred = cv_model.predict(_numeric_matrix(validation_fit, features))
        metrics = _validation_tuning_metrics(validation_fit, forecast=val_pred)
        elastic = cast(Any, cv_model.named_steps["elastic_net"])
        params = {
            "alpha": float(elastic.alpha_),
            "l1_ratio": float(elastic.l1_ratio_),
            "max_iter": 10_000,
        }
        _cache_selected_params(
            state=tuning_state,
            model_id="linear_elastic_net_tuned",
            target_id=target_id,
            params=params,
            metrics=metrics,
        )
        tuning_state.trials.append(
            {
                "model_id": "linear_elastic_net_tuned",
                "target_id": target_id,
                "trial_number": 0,
                "selected": True,
                "seed": tuning_state.seed,
                "params_json": json.dumps(params, sort_keys=True),
                **metrics,
                "objective_value": _tuning_objective_value(metrics),
            }
        )
        selection_target = target_id
    train_validation_fit = _finite_target_frame(_combined_train_validation(frame))
    final_model = Pipeline(
        [
            ("scaler", StandardScaler()),
            (
                "elastic_net",
                ElasticNet(
                    alpha=_param_float(params, "alpha"),
                    l1_ratio=_param_float(params, "l1_ratio"),
                    max_iter=_param_int(params, "max_iter", 10_000),
                    random_state=tuning_state.seed,
                ),
            ),
        ]
    )
    final_model.fit(
        _numeric_matrix(train_validation_fit, features),
        pd.to_numeric(train_validation_fit["rvar_event"], errors="coerce").to_numpy(dtype=float),
    )
    pred = pd.Series(np.nan, index=frame.index, dtype=float)
    for split_frame in (validation, test):
        pred.loc[split_frame.index] = final_model.predict(_numeric_matrix(split_frame, features))
    return (
        pred.clip(lower=FORECAST_FLOOR),
        {
            "status": "trained",
            "train_rows": int(len(train)),
            "validation_rows": int(len(validation)),
            "test_rows": int(len(test)),
            "tuning_profile": tuning_state.profile,
            "selection_target_id": selection_target,
            "tuned_alpha": _param_float(params, "alpha"),
            "tuned_l1_ratio": _param_float(params, "l1_ratio"),
            "refit_rows": int(len(train_validation_fit)),
            "implementation": "sklearn_elastic_net_cv",
        },
        final_model,
    )


def _train_lightgbm_tuned(
    frame: pd.DataFrame,
    *,
    features: Sequence[str],
    target_id: str,
    tuning_state: TuningState,
) -> tuple[pd.Series, dict[str, object], object | None]:  # pragma: no cover - optional dependency
    try:
        import lightgbm as lgb
    except ImportError:
        return (
            pd.Series(np.nan, index=frame.index),
            {"status": "skipped_dependency_unavailable"},
            None,
        )
    try:
        optuna = _require_optuna()
    except RuntimeError as exc:
        return pd.Series(np.nan, index=frame.index), {"status": str(exc)}, None
    train, validation, test = _fit_splits(frame)
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
    params = _cached_params(tuning_state, "lightgbm_tuned")
    selection_target = cast(
        str, tuning_state.selected.get("lightgbm_tuned", {}).get("selection_target_id", target_id)
    )
    if params is None:
        x_train = _numeric_matrix(train_fit, features)
        y_train = pd.to_numeric(train_fit["rvar_event"], errors="coerce")
        x_val = _numeric_matrix(validation_fit, features)
        y_val = pd.to_numeric(validation_fit["rvar_event"], errors="coerce")

        def objective(trial: Any) -> float:
            trial_params = {
                "num_leaves": trial.suggest_int("num_leaves", 7, 63),
                "min_data_in_leaf": trial.suggest_int("min_data_in_leaf", 5, 80),
                "learning_rate": trial.suggest_float("learning_rate", 0.005, 0.08, log=True),
                "feature_fraction": trial.suggest_float("feature_fraction", 0.55, 1.0),
                "bagging_fraction": trial.suggest_float("bagging_fraction", 0.55, 1.0),
                "lambda_l1": trial.suggest_float("lambda_l1", 1e-8, 10.0, log=True),
                "lambda_l2": trial.suggest_float("lambda_l2", 1e-8, 10.0, log=True),
            }
            model = lgb.LGBMRegressor(
                **trial_params,
                n_estimators=2000,
                random_state=tuning_state.seed,
                bagging_seed=tuning_state.seed,
                bagging_freq=1,
                feature_fraction_seed=tuning_state.seed,
                objective="regression",
                verbose=-1,
            )
            model.fit(
                x_train,
                y_train,
                eval_set=[(x_val, y_val)],
                eval_metric="rmse",
                callbacks=[lgb.early_stopping(50, verbose=False)],
            )
            forecast = model.predict(x_val)
            metrics = _validation_tuning_metrics(validation_fit, forecast=forecast)
            trial.set_user_attr("best_iteration", int(getattr(model, "best_iteration_", 0) or 0))
            for key, value in metrics.items():
                trial.set_user_attr(key, value)
            return _tuning_objective_value(metrics)

        study = optuna.create_study(
            direction="maximize",
            sampler=optuna.samplers.TPESampler(seed=tuning_state.seed),
        )
        study.optimize(objective, n_trials=TUNING_LIGHTGBM_TRIALS, show_progress_bar=False)
        best = _best_completed_trial(study)
        if best is None:
            return (
                pd.Series(np.nan, index=frame.index),
                {"status": "skipped_tuning_failed"},
                None,
            )
        best_iteration = int(best.user_attrs.get("best_iteration") or 2000)
        params = {**best.params, "best_iteration": max(1, best_iteration), "bagging_freq": 1}
        _record_optuna_trials(
            state=tuning_state,
            study=study,
            model_id="lightgbm_tuned",
            target_id=target_id,
            selected_number=int(best.number),
        )
        _cache_selected_params(
            state=tuning_state,
            model_id="lightgbm_tuned",
            target_id=target_id,
            params=params,
            metrics=best.user_attrs,
        )
        selection_target = target_id
    train_validation_fit = _finite_target_frame(_combined_train_validation(frame))
    final_params = cast(
        dict[str, Any],
        {key: value for key, value in params.items() if key != "best_iteration"},
    )
    final_model = lgb.LGBMRegressor(
        **final_params,
        n_estimators=_param_int(params, "best_iteration", 2000),
        random_state=tuning_state.seed,
        bagging_seed=tuning_state.seed,
        feature_fraction_seed=tuning_state.seed,
        objective="regression",
        verbose=-1,
    )
    final_model.fit(
        _numeric_matrix(train_validation_fit, features),
        pd.to_numeric(train_validation_fit["rvar_event"], errors="coerce"),
    )
    pred = pd.Series(np.nan, index=frame.index, dtype=float)
    for split_frame in (validation, test):
        pred.loc[split_frame.index] = final_model.predict(_numeric_matrix(split_frame, features))
    return (
        pred.clip(lower=FORECAST_FLOOR),
        {
            "status": "trained",
            "train_rows": int(len(train)),
            "validation_rows": int(len(validation)),
            "test_rows": int(len(test)),
            "tuning_profile": tuning_state.profile,
            "selection_target_id": selection_target,
            "best_iteration": _param_int(params, "best_iteration", 2000),
            "refit_rows": int(len(train_validation_fit)),
        },
        final_model,
    )


def _train_xgboost_tuned(
    frame: pd.DataFrame,
    *,
    features: Sequence[str],
    target_id: str,
    tuning_state: TuningState,
) -> tuple[pd.Series, dict[str, object], object | None]:  # pragma: no cover - optional dependency
    try:
        import xgboost as xgb
    except ImportError:
        return (
            pd.Series(np.nan, index=frame.index),
            {"status": "skipped_dependency_unavailable"},
            None,
        )
    try:
        optuna = _require_optuna()
    except RuntimeError as exc:
        return pd.Series(np.nan, index=frame.index), {"status": str(exc)}, None
    train, validation, test = _fit_splits(frame)
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
    params = _cached_params(tuning_state, "xgboost_tuned")
    selection_target = cast(
        str, tuning_state.selected.get("xgboost_tuned", {}).get("selection_target_id", target_id)
    )
    if params is None:
        x_train = _numeric_matrix(train_fit, features)
        y_train = pd.to_numeric(train_fit["rvar_event"], errors="coerce")
        x_val = _numeric_matrix(validation_fit, features)
        y_val = pd.to_numeric(validation_fit["rvar_event"], errors="coerce")

        def objective(trial: Any) -> float:
            trial_params = {
                "max_depth": trial.suggest_int("max_depth", 2, 6),
                "min_child_weight": trial.suggest_float("min_child_weight", 1.0, 20.0),
                "learning_rate": trial.suggest_float("learning_rate", 0.005, 0.08, log=True),
                "subsample": trial.suggest_float("subsample", 0.55, 1.0),
                "colsample_bytree": trial.suggest_float("colsample_bytree", 0.55, 1.0),
                "gamma": trial.suggest_float("gamma", 1e-8, 10.0, log=True),
                "reg_alpha": trial.suggest_float("reg_alpha", 1e-8, 10.0, log=True),
                "reg_lambda": trial.suggest_float("reg_lambda", 1e-8, 10.0, log=True),
            }
            model = xgb.XGBRegressor(
                **trial_params,
                n_estimators=2000,
                objective="reg:squarederror",
                random_state=tuning_state.seed,
                eval_metric="rmse",
                early_stopping_rounds=50,
                verbosity=0,
            )
            model.fit(x_train, y_train, eval_set=[(x_val, y_val)], verbose=False)
            forecast = model.predict(x_val)
            metrics = _validation_tuning_metrics(validation_fit, forecast=forecast)
            best_iteration = getattr(model, "best_iteration", None)
            trial.set_user_attr(
                "best_iteration",
                int(best_iteration) + 1 if best_iteration is not None else 2000,
            )
            for key, value in metrics.items():
                trial.set_user_attr(key, value)
            return _tuning_objective_value(metrics)

        study = optuna.create_study(
            direction="maximize",
            sampler=optuna.samplers.TPESampler(seed=tuning_state.seed),
        )
        study.optimize(objective, n_trials=TUNING_XGBOOST_TRIALS, show_progress_bar=False)
        best = _best_completed_trial(study)
        if best is None:
            return (
                pd.Series(np.nan, index=frame.index),
                {"status": "skipped_tuning_failed"},
                None,
            )
        params = {**best.params, "best_iteration": int(best.user_attrs.get("best_iteration", 2000))}
        _record_optuna_trials(
            state=tuning_state,
            study=study,
            model_id="xgboost_tuned",
            target_id=target_id,
            selected_number=int(best.number),
        )
        _cache_selected_params(
            state=tuning_state,
            model_id="xgboost_tuned",
            target_id=target_id,
            params=params,
            metrics=best.user_attrs,
        )
        selection_target = target_id
    train_validation_fit = _finite_target_frame(_combined_train_validation(frame))
    final_params = {key: value for key, value in params.items() if key != "best_iteration"}
    final_model = xgb.XGBRegressor(
        **final_params,
        n_estimators=_param_int(params, "best_iteration", 2000),
        objective="reg:squarederror",
        random_state=tuning_state.seed,
        verbosity=0,
    )
    final_model.fit(
        _numeric_matrix(train_validation_fit, features),
        pd.to_numeric(train_validation_fit["rvar_event"], errors="coerce"),
    )
    pred = pd.Series(np.nan, index=frame.index, dtype=float)
    for split_frame in (validation, test):
        pred.loc[split_frame.index] = final_model.predict(_numeric_matrix(split_frame, features))
    return (
        pred.clip(lower=FORECAST_FLOOR),
        {
            "status": "trained",
            "train_rows": int(len(train)),
            "validation_rows": int(len(validation)),
            "test_rows": int(len(test)),
            "tuning_profile": tuning_state.profile,
            "selection_target_id": selection_target,
            "best_iteration": _param_int(params, "best_iteration", 2000),
            "refit_rows": int(len(train_validation_fit)),
        },
        final_model,
    )


def _fit_ft_transformer_once(
    train_fit: pd.DataFrame,
    *,
    features: Sequence[str],
    seed: int,
    d_token: int,
    n_heads: int,
    n_layers: int,
    dropout: float,
    lr: float,
    weight_decay: float,
    epochs: int,
) -> FTTransformerRegressor:
    torch.manual_seed(seed)
    model = FTTransformerRegressor(
        n_features=len(features),
        d_token=d_token,
        n_heads=n_heads,
        n_layers=n_layers,
        dropout=dropout,
    )
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    x_train = torch.tensor(
        _numeric_matrix(train_fit, features).to_numpy(dtype=float), dtype=torch.float32
    )
    y_train = torch.tensor(
        pd.to_numeric(train_fit["rvar_event"], errors="coerce").to_numpy(dtype=float),
        dtype=torch.float32,
    )
    for _ in range(max(1, epochs)):
        model.train()
        optimizer.zero_grad()
        loss = torch.mean(torch.square(model(x_train) - y_train))
        loss.backward()  # type: ignore[no-untyped-call]
        optimizer.step()
    return model


def _train_ft_transformer_tuned(
    frame: pd.DataFrame,
    *,
    features: Sequence[str],
    target_id: str,
    tuning_state: TuningState,
) -> tuple[pd.Series, dict[str, object], object | None]:
    try:
        optuna = _require_optuna()
    except RuntimeError as exc:
        return pd.Series(np.nan, index=frame.index), {"status": str(exc)}, None
    train, validation, test = _fit_splits(frame)
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
    params = _cached_params(tuning_state, "ft_transformer_tuned")
    selection_target = cast(
        str,
        tuning_state.selected.get("ft_transformer_tuned", {}).get("selection_target_id", target_id),
    )
    if params is None:
        x_val = torch.tensor(
            _numeric_matrix(validation_fit, features).to_numpy(dtype=float), dtype=torch.float32
        )

        def objective(trial: Any) -> float:
            d_token = trial.suggest_categorical("d_token", [16, 32, 48])
            n_heads = trial.suggest_categorical("n_heads", [2, 4])
            if int(d_token) % int(n_heads) != 0:
                raise optuna.TrialPruned()
            trial_params = {
                "d_token": int(d_token),
                "n_heads": int(n_heads),
                "n_layers": int(trial.suggest_categorical("n_layers", [1, 2])),
                "lr": float(trial.suggest_float("lr", 1e-4, 1e-3, log=True)),
                "weight_decay": float(
                    trial.suggest_categorical("weight_decay", [1e-5, 1e-4, 1e-3])
                ),
                "dropout": float(trial.suggest_categorical("dropout", [0.0, 0.1, 0.2])),
            }
            torch.manual_seed(tuning_state.seed)
            model = FTTransformerRegressor(
                n_features=len(features),
                d_token=int(trial_params["d_token"]),
                n_heads=int(trial_params["n_heads"]),
                n_layers=int(trial_params["n_layers"]),
                dropout=float(trial_params["dropout"]),
            )
            optimizer = torch.optim.AdamW(
                model.parameters(),
                lr=float(trial_params["lr"]),
                weight_decay=float(trial_params["weight_decay"]),
            )
            x_train = torch.tensor(
                _numeric_matrix(train_fit, features).to_numpy(dtype=float), dtype=torch.float32
            )
            y_train = torch.tensor(
                pd.to_numeric(train_fit["rvar_event"], errors="coerce").to_numpy(dtype=float),
                dtype=torch.float32,
            )
            y_val = torch.tensor(
                pd.to_numeric(validation_fit["rvar_event"], errors="coerce").to_numpy(dtype=float),
                dtype=torch.float32,
            )
            best_state: dict[str, torch.Tensor] | None = None
            best_loss = float("inf")
            epochs_run = 0
            stale = 0
            for epoch in range(40):
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
                    best_state = {
                        key: value.detach().clone() for key, value in model.state_dict().items()
                    }
                    stale = 0
                else:
                    stale += 1
                    if stale >= 8:
                        break
            if best_state is not None:
                model.load_state_dict(best_state)
            model.eval()
            with torch.no_grad():
                forecast = model(x_val).detach().numpy()
            metrics = _validation_tuning_metrics(validation_fit, forecast=forecast)
            trial.set_user_attr("epochs", int(epochs_run))
            for key, value in metrics.items():
                trial.set_user_attr(key, value)
            return _tuning_objective_value(metrics)

        study = optuna.create_study(
            direction="maximize",
            sampler=optuna.samplers.TPESampler(seed=tuning_state.seed),
        )
        study.optimize(objective, n_trials=TUNING_FT_TRANSFORMER_TRIALS, show_progress_bar=False)
        best = _best_completed_trial(study)
        if best is None:
            return (
                pd.Series(np.nan, index=frame.index),
                {"status": "skipped_tuning_failed"},
                None,
            )
        params = {**best.params, "epochs": int(best.user_attrs.get("epochs", 40))}
        _record_optuna_trials(
            state=tuning_state,
            study=study,
            model_id="ft_transformer_tuned",
            target_id=target_id,
            selected_number=int(best.number),
        )
        _cache_selected_params(
            state=tuning_state,
            model_id="ft_transformer_tuned",
            target_id=target_id,
            params=params,
            metrics=best.user_attrs,
        )
        selection_target = target_id
    train_validation_fit = _finite_target_frame(_combined_train_validation(frame))
    final_model = _fit_ft_transformer_once(
        train_validation_fit,
        features=features,
        seed=tuning_state.seed,
        d_token=_param_int(params, "d_token"),
        n_heads=_param_int(params, "n_heads"),
        n_layers=_param_int(params, "n_layers"),
        dropout=_param_float(params, "dropout"),
        lr=_param_float(params, "lr"),
        weight_decay=_param_float(params, "weight_decay"),
        epochs=_param_int(params, "epochs", 40),
    )
    pred = pd.Series(np.nan, index=frame.index, dtype=float)
    final_model.eval()
    for split_frame in (validation, test):
        with torch.no_grad():
            values = (
                final_model(
                    torch.tensor(
                        _numeric_matrix(split_frame, features).to_numpy(dtype=float),
                        dtype=torch.float32,
                    )
                )
                .detach()
                .numpy()
            )
        pred.loc[split_frame.index] = values
    return (
        pred.clip(lower=FORECAST_FLOOR),
        {
            "status": "trained",
            "train_rows": int(len(train)),
            "validation_rows": int(len(validation)),
            "test_rows": int(len(test)),
            "tuning_profile": tuning_state.profile,
            "selection_target_id": selection_target,
            "epochs": _param_int(params, "epochs", 40),
            "d_token": _param_int(params, "d_token"),
            "n_heads": _param_int(params, "n_heads"),
            "n_layers": _param_int(params, "n_layers"),
            "dropout": _param_float(params, "dropout"),
            "lr": _param_float(params, "lr"),
            "weight_decay": _param_float(params, "weight_decay"),
            "refit_rows": int(len(train_validation_fit)),
        },
        final_model,
    )


def _load_sequence_tensor(path: Path) -> dict[str, np.ndarray]:
    payload = np.load(path, allow_pickle=True)
    return {key: payload[key] for key in payload.files}


def _deterministic_permutation(event_id: str, *, length: int, seed: int) -> np.ndarray:
    digest = hashlib.sha256(f"{event_id}:{seed}".encode()).digest()
    local_seed = int.from_bytes(digest[:8], byteorder="little", signed=False) % (2**32)
    rng = np.random.default_rng(local_seed)
    return rng.permutation(length)


def _sequence_input(
    tensor: Mapping[str, np.ndarray],
    *,
    mask_only: bool = False,
    time_shuffle: bool = False,
    seed: int = 17,
) -> tuple[np.ndarray, np.ndarray]:
    x_values = tensor["x"].astype(np.float32).copy()
    feature_mask = tensor["feature_mask"].astype(bool).copy()
    time_mask = tensor["time_mask"].astype(bool).copy()
    if time_shuffle:
        event_ids = [str(value) for value in tensor["event_id"].tolist()]
        for row_idx, event_id in enumerate(event_ids):
            order = _deterministic_permutation(event_id, length=x_values.shape[1], seed=seed)
            x_values[row_idx] = x_values[row_idx, order, :]
            feature_mask[row_idx] = feature_mask[row_idx, order, :]
            time_mask[row_idx] = time_mask[row_idx, order]
    values = np.zeros_like(x_values) if mask_only else x_values
    x_all = np.concatenate(
        [values, feature_mask.astype(np.float32), time_mask[:, :, None].astype(np.float32)],
        axis=2,
    ).astype(np.float32)
    return x_all, time_mask


def _pairwise_ranking_loss(
    predicted_edge: torch.Tensor, realized_edge: torch.Tensor
) -> torch.Tensor:
    y_diff = realized_edge[:, None] - realized_edge[None, :]
    sign = torch.sign(y_diff)
    valid = sign.ne(0)
    if int(valid.sum().item()) == 0:
        return torch.zeros((), dtype=predicted_edge.dtype, device=predicted_edge.device)
    pred_diff = predicted_edge[:, None] - predicted_edge[None, :]
    return torch.nn.functional.softplus(-sign[valid] * pred_diff[valid]).mean()


def _sequence_losses(
    log_prediction: torch.Tensor,
    log_target: torch.Tensor,
    predicted_edge: torch.Tensor,
    realized_edge: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    huber = torch.nn.functional.huber_loss(log_prediction, log_target, delta=0.5)
    ranking = _pairwise_ranking_loss(predicted_edge, realized_edge)
    return huber, ranking


def _loss_weights_from_scale(huber: float, ranking: float) -> tuple[float, float, str]:
    if not np.isfinite(huber) or not np.isfinite(ranking) or huber <= 0 or ranking <= 0:
        return 0.5, 0.5, "default_invalid_scale"
    larger = max(huber, ranking)
    smaller = min(huber, ranking)
    if larger / smaller <= 10.0:
        return 0.5, 0.5, "default_within_10x"
    inv_huber = 1.0 / huber
    inv_ranking = 1.0 / ranking
    total = inv_huber + inv_ranking
    return inv_huber / total, inv_ranking / total, "inverse_scale_rebalanced"


def _make_sequence_encoder(
    model_id: str,
    *,
    n_features: int,
    hidden_size: int,
    n_layers: int,
) -> torch.nn.Module:
    if model_id in {"bigru_sequence", "mask_only_sequence", "time_shuffle_sequence"}:
        return BiGRUSequenceEncoder(
            n_features=n_features,
            hidden_size=hidden_size,
            n_layers=n_layers,
            dropout=0.15,
        )
    if model_id == "attention_pooling_sequence":
        return AttentionPoolingSequenceEncoder(n_features=n_features, hidden_size=hidden_size)
    if model_id == "dilated_cnn_sequence":
        return DilatedCNNSequenceEncoder(n_features=n_features)
    if model_id == "mamba_ssm_sequence":
        return MambaSSMSequenceEncoder(
            n_features=n_features,
            hidden_size=hidden_size,
            n_layers=n_layers,
            dropout=0.15,
        )
    raise ValueError(f"unsupported sequence model_id: {model_id}")


def _sequence_row_indices(
    frame: pd.DataFrame,
    *,
    tensor_path: Path,
    eligibility_col: str = "hybrid_sequence_eligible_v2",
) -> tuple[dict[str, np.ndarray], pd.Series, pd.Series]:
    if not tensor_path.exists():
        raise FileNotFoundError(str(tensor_path))
    tensor = _load_sequence_tensor(tensor_path)
    tensor_events = [str(value) for value in tensor["event_id"].tolist()]
    tensor_index = {event_id: idx for idx, event_id in enumerate(tensor_events)}
    frame = ensure_event_id(frame)
    eligible = (
        frame[eligibility_col].astype(bool)
        if eligibility_col in frame
        else pd.Series(False, index=frame.index)
    )
    row_tensor_idx = frame["event_id"].astype(str).map(tensor_index)
    valid_rows = (
        eligible
        & row_tensor_idx.notna()
        & pd.to_numeric(frame["rvar_event"], errors="coerce").notna()
    )
    return tensor, row_tensor_idx, valid_rows


def _train_sequence_model(
    frame: pd.DataFrame,
    *,
    tensor_path: Path,
    model_id: str,
    mask_only: bool = False,
    time_shuffle: bool = False,
    eligibility_col: str = "hybrid_sequence_eligible_v2",
    seed: int = 17,
    hidden_sizes: Sequence[int] = (16, 32, 64),
    layers: Sequence[int] = (1, 2),
    lr: float = 1e-3,
    weight_decay: float = 1e-4,
    max_epochs: int = 60,
    patience: int = 8,
) -> tuple[pd.Series, dict[str, object], object | None]:
    try:
        tensor, row_tensor_idx, valid_rows = _sequence_row_indices(
            frame, tensor_path=tensor_path, eligibility_col=eligibility_col
        )
    except FileNotFoundError:
        return pd.Series(np.nan, index=frame.index), {"status": "skipped_no_sequence_tensor"}, None
    if not bool(valid_rows.any()):
        return pd.Series(np.nan, index=frame.index), {"status": "skipped_no_sequence_rows"}, None
    train_rows = frame.loc[valid_rows & frame["split"].eq("train")]
    val_rows = frame.loc[valid_rows & frame["split"].eq("validation")]
    test_rows = frame.loc[valid_rows & frame["split"].eq("test")]
    skip = _safe_training_frames(frame, train=train_rows, validation=val_rows, test=test_rows)
    if skip:
        return pd.Series(np.nan, index=frame.index), {"status": skip}, None
    x_all, time_mask = _sequence_input(
        tensor, mask_only=mask_only, time_shuffle=time_shuffle, seed=seed
    )
    target = np.log(pd.to_numeric(frame["rvar_event"], errors="coerce") + FORECAST_FLOOR)
    ivar = pd.to_numeric(frame["ivar_event"], errors="coerce")
    target_values = frame["target_id"].dropna().astype(str) if "target_id" in frame else pd.Series()
    edge_column = (
        f"edge_var_realized_{target_values.iloc[0]}"
        if not target_values.empty and f"edge_var_realized_{target_values.iloc[0]}" in frame.columns
        else "edge_var_realized"
    )
    realized_edge = pd.to_numeric(frame[edge_column], errors="coerce")
    best_model: torch.nn.Module | None = None
    best_loss = float("inf")
    best_epochs = 0
    best_hidden_size = 0
    best_layers = 0
    best_huber_weight = 0.5
    best_ranking_weight = 0.5
    best_rebalance_status = "unavailable"
    torch.manual_seed(seed)
    np.random.seed(seed)
    for hidden_size in hidden_sizes:
        for n_layers in layers:
            try:
                model = _make_sequence_encoder(
                    model_id,
                    n_features=x_all.shape[2],
                    hidden_size=hidden_size,
                    n_layers=n_layers,
                )
            except RuntimeError as exc:
                return (
                    pd.Series(np.nan, index=frame.index),
                    {"status": "skipped_dependency_unavailable", "error": str(exc)},
                    None,
                )
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            model = model.to(device)
            optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
            train_idx = row_tensor_idx.loc[train_rows.index].astype(int).to_numpy()
            val_idx = row_tensor_idx.loc[val_rows.index].astype(int).to_numpy()
            x_train = torch.tensor(x_all[train_idx], dtype=torch.float32, device=device)
            mask_train = torch.tensor(time_mask[train_idx], dtype=torch.bool, device=device)
            y_train = torch.tensor(
                target.loc[train_rows.index].to_numpy(dtype=float),
                dtype=torch.float32,
                device=device,
            )
            edge_train = torch.tensor(
                realized_edge.loc[train_rows.index].to_numpy(dtype=float),
                dtype=torch.float32,
                device=device,
            )
            ivar_train = torch.tensor(
                ivar.loc[train_rows.index].to_numpy(dtype=float),
                dtype=torch.float32,
                device=device,
            )
            x_val = torch.tensor(x_all[val_idx], dtype=torch.float32, device=device)
            mask_val = torch.tensor(time_mask[val_idx], dtype=torch.bool, device=device)
            y_val = torch.tensor(
                target.loc[val_rows.index].to_numpy(dtype=float),
                dtype=torch.float32,
                device=device,
            )
            edge_val = torch.tensor(
                realized_edge.loc[val_rows.index].to_numpy(dtype=float),
                dtype=torch.float32,
                device=device,
            )
            ivar_val = torch.tensor(
                ivar.loc[val_rows.index].to_numpy(dtype=float),
                dtype=torch.float32,
                device=device,
            )
            with torch.no_grad():
                init_log = cast(torch.Tensor, model(x_train, mask_train))
                init_pred = torch.exp(init_log).clamp_min(FORECAST_FLOOR)
                init_huber, init_ranking = _sequence_losses(
                    init_log,
                    y_train,
                    init_pred - ivar_train,
                    edge_train,
                )
            huber_weight, ranking_weight, rebalance_status = _loss_weights_from_scale(
                float(init_huber.item()),
                float(init_ranking.item()),
            )
            local_best_state: dict[str, torch.Tensor] | None = None
            local_best = float("inf")
            stale = 0
            epochs_run = 0
            for epoch in range(max_epochs):
                epochs_run = epoch + 1
                model.train()
                optimizer.zero_grad()
                log_pred = cast(torch.Tensor, model(x_train, mask_train))
                pred = torch.exp(log_pred).clamp_min(FORECAST_FLOOR)
                huber, ranking = _sequence_losses(log_pred, y_train, pred - ivar_train, edge_train)
                loss = huber_weight * huber + ranking_weight * ranking
                loss.backward()  # type: ignore[no-untyped-call]
                optimizer.step()
                model.eval()
                with torch.no_grad():
                    val_log = cast(torch.Tensor, model(x_val, mask_val))
                    val_pred = torch.exp(val_log).clamp_min(FORECAST_FLOOR)
                    val_huber, val_ranking = _sequence_losses(
                        val_log,
                        y_val,
                        val_pred - ivar_val,
                        edge_val,
                    )
                    val_loss = float(
                        (huber_weight * val_huber + ranking_weight * val_ranking).item()
                    )
                if val_loss < local_best:
                    local_best = val_loss
                    local_best_state = {
                        key: value.detach().cpu().clone()
                        for key, value in model.state_dict().items()
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
                best_hidden_size = hidden_size
                best_layers = n_layers
                best_huber_weight = huber_weight
                best_ranking_weight = ranking_weight
                best_rebalance_status = rebalance_status
    if best_model is None:
        return pd.Series(np.nan, index=frame.index), {"status": "skipped_training_failed"}, None
    pred = pd.Series(np.nan, index=frame.index, dtype=float)
    best_model.eval()
    for split_rows in (val_rows, test_rows):
        idx = row_tensor_idx.loc[split_rows.index].astype(int).to_numpy()
        with torch.no_grad():
            log_values = (
                cast(
                    torch.Tensor,
                    best_model(
                        torch.tensor(x_all[idx], dtype=torch.float32, device=device),
                        torch.tensor(time_mask[idx], dtype=torch.bool, device=device),
                    ),
                )
                .detach()
                .cpu()
                .numpy()
            )
        pred.loc[split_rows.index] = np.maximum(np.exp(log_values) - FORECAST_FLOOR, FORECAST_FLOOR)
    return (
        pred.clip(lower=FORECAST_FLOOR),
        {
            "status": "trained",
            "train_rows": int(len(train_rows)),
            "validation_rows": int(len(val_rows)),
            "test_rows": int(len(test_rows)),
            "hidden_sizes": list(hidden_sizes),
            "selected_hidden_size": int(best_hidden_size),
            "layers": list(layers),
            "selected_layers": int(best_layers),
            "selected_validation_sequence_loss": float(best_loss),
            "epochs": int(best_epochs),
            "loss": "huber_log_rvar_plus_pairwise_edge_ranking",
            "ranking_edge_column": edge_column,
            "huber_weight": float(best_huber_weight),
            "ranking_weight": float(best_ranking_weight),
            "loss_rebalance_status": best_rebalance_status,
            "device": device.type,
            "mask_only": bool(mask_only),
            "time_shuffle": bool(time_shuffle),
            "claim_scope": "diagnostic",
            "headline_eligible": False,
        },
        best_model,
    )


def _sequence_flat_aggregate_frame(
    frame: pd.DataFrame,
    *,
    tensor_path: Path,
    eligibility_col: str = "hybrid_sequence_eligible_v2",
) -> tuple[pd.DataFrame, list[str]]:
    tensor, row_tensor_idx, valid_rows = _sequence_row_indices(
        frame, tensor_path=tensor_path, eligibility_col=eligibility_col
    )
    x_values = tensor["x"].astype(float)
    feature_mask = tensor["feature_mask"].astype(bool)
    step_type = tensor.get("step_type", np.full(x_values.shape[:2], "all", dtype=object)).astype(
        str
    )
    feature_names = [str(value) for value in tensor["feature_names"].tolist()]
    rows: list[dict[str, object]] = []
    for row_index, _row in frame.iterrows():
        record: dict[str, object] = {"_row_index": row_index}
        tensor_idx_value = row_tensor_idx.loc[row_index]
        if not bool(valid_rows.loc[row_index]) or pd.isna(tensor_idx_value):
            rows.append(record)
            continue
        tensor_idx = int(tensor_idx_value)
        for branch_name, branch_mask in (
            ("daily", step_type[tensor_idx] == "daily"),
            ("intraday", step_type[tensor_idx] == "intraday"),
        ):
            if not bool(branch_mask.any()):
                branch_mask = np.ones(x_values.shape[1], dtype=bool)
            for feature_idx, feature in enumerate(feature_names):
                observed_mask = branch_mask & feature_mask[tensor_idx, :, feature_idx]
                values = x_values[tensor_idx, :, feature_idx][observed_mask]
                prefix = f"seqflat_{branch_name}_{feature}"
                if values.size:
                    record[f"{prefix}_mean"] = float(np.mean(values))
                    record[f"{prefix}_std"] = float(np.std(values)) if values.size > 1 else 0.0
                    record[f"{prefix}_min"] = float(np.min(values))
                    record[f"{prefix}_max"] = float(np.max(values))
                    record[f"{prefix}_last_value"] = float(values[-1])
                    if values.size > 1:
                        record[f"{prefix}_simple_slope"] = float(
                            np.polyfit(np.arange(values.size, dtype=float), values, deg=1)[0]
                        )
                    else:
                        record[f"{prefix}_simple_slope"] = 0.0
                else:
                    for suffix in ("mean", "std", "min", "max", "last_value", "simple_slope"):
                        record[f"{prefix}_{suffix}"] = np.nan
        rows.append(record)
    aggregate = pd.DataFrame(rows).set_index("_row_index")
    features = [column for column in aggregate.columns if column.startswith("seqflat_")]
    out = pd.concat([frame.copy(), aggregate], axis=1)
    return out, features


def _train_ridge_flat_sequence(
    frame: pd.DataFrame,
    *,
    tensor_path: Path,
    eligibility_col: str = "hybrid_sequence_eligible_v2",
) -> tuple[pd.Series, dict[str, object], object | None]:
    try:
        flat_frame, flat_features = _sequence_flat_aggregate_frame(
            frame, tensor_path=tensor_path, eligibility_col=eligibility_col
        )
    except FileNotFoundError:
        return pd.Series(np.nan, index=frame.index), {"status": "skipped_no_sequence_tensor"}, None
    if not flat_features:
        return (
            pd.Series(np.nan, index=frame.index),
            {"status": "skipped_no_sequence_features"},
            None,
        )
    train = flat_frame.loc[flat_frame["split"].eq("train")]
    validation = flat_frame.loc[flat_frame["split"].eq("validation")]
    test = flat_frame.loc[flat_frame["split"].eq("test")]
    skip = _safe_training_frames(flat_frame, train=train, validation=validation, test=test)
    if skip:
        return pd.Series(np.nan, index=frame.index), {"status": skip}, None
    model = LinearElasticNetRegressor(alpha=0.01, l1_ratio=0.0)
    model.fit(train, target_col="rvar_event", feature_columns=flat_features)
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
            "feature_count": int(len(flat_features)),
            "claim_scope": "diagnostic",
            "headline_eligible": False,
        },
        model,
    )


def _train_lightgbm_xgboost_ensemble(
    predictions: pd.DataFrame,
) -> tuple[pd.Series, dict[str, object], object | None]:
    required = ["forecast_lightgbm", "forecast_xgboost"]
    if any(column not in predictions.columns for column in required):
        return pd.Series(np.nan, index=predictions.index), {"status": "skipped_missing_base"}, None
    lgbm = pd.to_numeric(predictions["forecast_lightgbm"], errors="coerce")
    xgboost = pd.to_numeric(predictions["forecast_xgboost"], errors="coerce")
    raw_mean = (lgbm + xgboost) / 2.0
    rank_average = (lgbm.rank(pct=True) + xgboost.rank(pct=True)) / 2.0
    pred = pd.Series(np.nan, index=predictions.index, dtype=float)
    valid = raw_mean.notna() & rank_average.notna()
    if bool(valid.any()):
        pred.loc[valid] = np.quantile(
            raw_mean.loc[valid].to_numpy(dtype=float),
            rank_average.loc[valid].clip(0.0, 1.0).to_numpy(dtype=float),
        )
    return (
        pred.clip(lower=FORECAST_FLOOR),
        {
            "status": "evaluated",
            "ensemble_method": "equal_weight_rank_average",
            "train_rows": int(predictions["split"].eq("train").sum()),
            "validation_rows": int(predictions["split"].eq("validation").sum()),
            "test_rows": int(predictions["split"].eq("test").sum()),
        },
        None,
    )


def _train_sequence_seed_ensemble(
    frame: pd.DataFrame,
    *,
    tensor_path: Path,
    model_id: str,
    base_model_id: str,
    seeds: Sequence[int],
) -> tuple[pd.Series, dict[str, object], object | None]:
    seed_predictions: list[pd.Series] = []
    seed_diagnostics: list[dict[str, object]] = []
    for seed in seeds:
        seed_pred, seed_diag, _ = _train_sequence_model(
            frame,
            tensor_path=tensor_path,
            model_id=base_model_id,
            seed=seed,
        )
        seed_predictions.append(seed_pred)
        seed_diagnostics.append(seed_diag)
    pred = pd.concat(seed_predictions, axis=1).mean(axis=1)
    trained_count = sum(str(diag.get("status")) == "trained" for diag in seed_diagnostics)
    first = seed_diagnostics[0] if seed_diagnostics else {}
    status = "trained" if trained_count else str(first.get("status", "skipped_training_failed"))
    return (
        pred.clip(lower=FORECAST_FLOOR),
        {
            **first,
            "status": status,
            "model_id": model_id,
            "base_model_id": base_model_id,
            "seed_count": int(len(seeds)),
            "trained_seed_count": int(trained_count),
            "seed_list": ",".join(str(seed) for seed in seeds),
            "seed_statuses": ",".join(str(diag.get("status")) for diag in seed_diagnostics),
            "claim_scope": "diagnostic",
            "headline_eligible": False,
        },
        None,
    )


def _train_model_dispatch(
    model_id: str,
    predictions: pd.DataFrame,
    *,
    event_features: Sequence[str],
    tree_features: Sequence[str],
    tensor_path: Path,
    hybrid_tensor_path: Path,
    mamba_backend: str,
    mamba_seeds: Sequence[int],
    tuning_state: TuningState,
    target_id: str,
) -> tuple[pd.Series, dict[str, object], object | None]:
    if model_id == "linear_elastic_net":
        return _train_elastic_net(predictions, features=event_features)
    if model_id == "linear_elastic_net_tuned":
        return _train_elastic_net_tuned(
            predictions,
            features=event_features,
            target_id=target_id,
            tuning_state=tuning_state,
        )
    if model_id == "lightgbm":
        return _train_lightgbm(predictions, features=tree_features)
    if model_id == "lightgbm_tuned":
        return _train_lightgbm_tuned(
            predictions,
            features=tree_features,
            target_id=target_id,
            tuning_state=tuning_state,
        )
    if model_id == "xgboost":
        return _train_xgboost(predictions, features=tree_features)
    if model_id == "xgboost_tuned":
        return _train_xgboost_tuned(
            predictions,
            features=tree_features,
            target_id=target_id,
            tuning_state=tuning_state,
        )
    if model_id == "lightgbm_xgboost_mean_ensemble":
        return _train_lightgbm_xgboost_ensemble(predictions)
    if model_id == "ft_transformer":
        return _train_ft_transformer(predictions, features=event_features)
    if model_id == "ft_transformer_tuned":
        return _train_ft_transformer_tuned(
            predictions,
            features=event_features,
            target_id=target_id,
            tuning_state=tuning_state,
        )
    if model_id == "ridge_flat_aggregates_sequence":
        return _train_ridge_flat_sequence(predictions, tensor_path=hybrid_tensor_path)
    if model_id == "bigru_sequence_5seed":
        return _train_sequence_seed_ensemble(
            predictions,
            tensor_path=hybrid_tensor_path,
            model_id=model_id,
            base_model_id="bigru_sequence",
            seeds=TUNING_SEQUENCE_ENSEMBLE_SEEDS,
        )
    if model_id == "mamba_ssm_sequence_5seed":
        _ = mamba_backend
        return _train_sequence_seed_ensemble(
            predictions,
            tensor_path=hybrid_tensor_path,
            model_id=model_id,
            base_model_id="mamba_ssm_sequence",
            seeds=TUNING_SEQUENCE_ENSEMBLE_SEEDS,
        )
    if model_id in {
        "attention_pooling_sequence",
        "bigru_sequence",
        "dilated_cnn_sequence",
        "mamba_ssm_sequence",
        "mask_only_sequence",
        "time_shuffle_sequence",
    }:
        if model_id == "mamba_ssm_sequence":
            seed_predictions: list[pd.Series] = []
            seed_diagnostics: list[dict[str, object]] = []
            for seed in mamba_seeds:
                seed_pred, seed_diag, _ = _train_sequence_model(
                    predictions,
                    tensor_path=hybrid_tensor_path,
                    model_id=model_id,
                    seed=seed,
                )
                seed_predictions.append(seed_pred)
                seed_diagnostics.append(seed_diag)
            pred = pd.concat(seed_predictions, axis=1).mean(axis=1)
            diag = {
                **seed_diagnostics[0],
                "mamba_backend": mamba_backend,
                "mamba_seeds": ",".join(str(seed) for seed in mamba_seeds),
                "seed_count": len(mamba_seeds),
                "seed_statuses": ",".join(str(item.get("status")) for item in seed_diagnostics),
            }
            return pred.clip(lower=FORECAST_FLOOR), diag, None
        return _train_sequence_model(
            predictions,
            tensor_path=hybrid_tensor_path,
            model_id=model_id,
            mask_only=model_id == "mask_only_sequence",
            time_shuffle=model_id == "time_shuffle_sequence",
        )
    if model_id == "lightgbm_with_hybrid_aggregates":
        return _train_lightgbm(predictions, features=tree_features)
    raise ValueError(f"unknown model_id: {model_id}")


def _prediction_availability_diagnostics(
    pred: pd.Series,
    frame: pd.DataFrame,
) -> dict[str, object]:
    numeric = pd.to_numeric(pred, errors="coerce")
    finite = pd.Series(np.isfinite(numeric.to_numpy(dtype=float)), index=pred.index)
    if "split" not in frame.columns:
        return {
            "prediction_finite_rows": int(finite.sum()),
            "validation_prediction_finite_rows": 0,
            "test_prediction_finite_rows": 0,
        }
    validation_mask = frame["split"].astype(str).eq("validation")
    test_mask = frame["split"].astype(str).eq("test")
    return {
        "prediction_finite_rows": int(finite.sum()),
        "validation_prediction_finite_rows": int(finite.loc[validation_mask].sum()),
        "test_prediction_finite_rows": int(finite.loc[test_mask].sum()),
    }


def _validated_model_diagnostics(
    diag: Mapping[str, object],
    *,
    pred: pd.Series,
    frame: pd.DataFrame,
) -> dict[str, object]:
    availability = _prediction_availability_diagnostics(pred, frame)
    out: dict[str, object] = {**diag, **availability}
    validation_finite = cast(int, availability["validation_prediction_finite_rows"])
    test_finite = cast(int, availability["test_prediction_finite_rows"])
    if str(diag.get("status")) == "trained" and validation_finite == 0 and test_finite == 0:
        out["raw_status"] = diag.get("status")
        out["status"] = "invalid_no_usable_predictions"
    return out


def run_proxy_model_suite(
    frame: pd.DataFrame,
    *,
    tensor_path: Path,
    hybrid_tensor_path: Path | None = None,
    model_ids: Sequence[str] = MODEL_IDS,
    mamba_backend: str = "mamba_ssm",
    mamba_seeds: Sequence[int] = (17,),
    tuning_state: TuningState | None = None,
    target_id: str = TUNING_SELECTION_TARGET_ID,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    if mamba_backend != "mamba_ssm":
        raise ValueError(f"unsupported mamba_backend: {mamba_backend}")
    if not mamba_seeds:
        raise ValueError("mamba_seeds must include at least one seed")
    predictions = add_benchmark_predictions(frame)
    diagnostics: list[dict[str, object]] = []
    event_features = event_level_feature_columns(predictions)
    tree_features = gbdt_feature_columns(predictions)
    active_tuning_state = tuning_state or TuningState()
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
        pred, diag, _ = _train_model_dispatch(
            model_id,
            predictions,
            event_features=event_features,
            tree_features=tree_features,
            tensor_path=tensor_path,
            hybrid_tensor_path=hybrid_tensor_path or tensor_path,
            mamba_backend=mamba_backend,
            mamba_seeds=mamba_seeds,
            tuning_state=active_tuning_state,
            target_id=target_id,
        )
        diag = _validated_model_diagnostics(diag, pred=pred, frame=predictions)
        hybrid_sparse = (
            bool(predictions["hybrid_sequence_too_sparse"].any())
            if "hybrid_sequence_too_sparse" in predictions
            else False
        )
        if hybrid_sparse:
            diag = {**diag, "status_label": "high_missingness_diagnostic"}
        column = research_prediction_column(model_id)
        predictions[column] = pred
        diagnostics.append(
            {
                "model_id": model_id,
                "feature_count": len(
                    tree_features
                    if model_id in GBDT_MODEL_IDS
                    or model_id
                    in {
                        "lightgbm_with_hybrid_aggregates",
                        "lightgbm_xgboost_mean_ensemble",
                    }
                    else event_features
                ),
                **diag,
            }
        )
    return predictions, pd.DataFrame(diagnostics)


def model_forecast_columns(frame: pd.DataFrame) -> dict[str, str]:
    mapping: dict[str, str] = {}
    for model_id in [
        *MODEL_IDS,
        *TUNED_TABULAR_MODEL_IDS,
        *SEQUENCE_ENSEMBLE_MODEL_IDS,
        *PHASE2_SEQUENCE_MODEL_IDS,
    ]:
        column = research_prediction_column(model_id)
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


def append_day_c2c_additive_naive_diagnostics(predictions: pd.DataFrame) -> pd.DataFrame:
    if "target_id" not in predictions.columns:
        return predictions
    out = predictions.copy()
    forecast_columns = model_forecast_columns(out)
    keys = ["event_id"]
    if "target_id" not in out or "event_id" not in out:
        return out
    for model_id, column in forecast_columns.items():
        if column not in out.columns:
            continue
        jump = out.loc[out["target_id"].eq("jump_c2o"), keys + [column]].rename(
            columns={column: "_jump_forecast"}
        )
        reaction = out.loc[out["target_id"].eq("reaction_o2c"), keys + [column]].rename(
            columns={column: "_reaction_forecast"}
        )
        additive = jump.merge(reaction, on=keys, how="inner")
        additive[f"forecast_day_c2c_additive_naive_{model_id}"] = pd.to_numeric(
            additive["_jump_forecast"], errors="coerce"
        ) + pd.to_numeric(additive["_reaction_forecast"], errors="coerce")
        out = out.merge(
            additive[keys + [f"forecast_day_c2c_additive_naive_{model_id}"]],
            on=keys,
            how="left",
        )
        out.loc[
            ~out["target_id"].eq("day_c2c"),
            f"forecast_day_c2c_additive_naive_{model_id}",
        ] = np.nan
    return out


MODEL_LEVEL_CSV_ARTIFACT_GLOBS = (
    "edge_deciles_*.csv",
    "strategy_trades_*.csv",
    "c2o_option_vwap_5_15_strategy_trades_*.csv",
    "c2o_option_vwap_0_5_strategy_trades_*.csv",
    "c2o_intrinsic_strategy_trades_*.csv",
    "o2c_option_vwap_5_15_strategy_trades_*.csv",
    "o2c_option_vwap_0_5_strategy_trades_*.csv",
)


def remove_model_level_csv_artifacts(out_dir: Path) -> list[Path]:
    removed: list[Path] = []
    if not out_dir.exists():
        return removed
    for pattern in MODEL_LEVEL_CSV_ARTIFACT_GLOBS:
        for path in sorted(out_dir.glob(pattern)):
            if path.is_file():
                path.unlink()
                removed.append(path)
    return removed


def write_retired_model_manifest(out_dir: Path) -> Path:
    path = out_dir / "retired_model_ids.json"
    write_json(
        path,
        {
            "retired_model_ids": RETIRED_MAMBA_MODEL_IDS,
            "reason": "in-repo gated-RNN, not official Mamba",
            "replacement": "mamba_ssm_sequence",
            "records": [
                {
                    "model_id": model_id,
                    "reason": "retired_in_repo_gated_rnn_not_official_mamba_ssm",
                    "replacement": "mamba_ssm_sequence",
                    "claim_scope": "retired",
                }
                for model_id in RETIRED_MAMBA_MODEL_IDS
            ],
        },
    )
    return path


def build_sequence_v2_quality(
    features: pd.DataFrame,
    *,
    tensor_path: Path,
) -> pd.DataFrame:
    features = ensure_event_id(features)
    if not tensor_path.exists():
        return pd.DataFrame(
            columns=[
                "event_id",
                "ticker",
                "split",
                "valid_len",
                "daily_valid_len",
                "intraday_valid_len",
                "missing_rate",
                "high_quality_sequence",
                "common_row_eligible",
                "sequence_gate_reason",
            ]
        )
    tensor = _load_sequence_tensor(tensor_path)
    event_ids = [str(value) for value in tensor["event_id"].tolist()]
    time_mask = tensor["time_mask"].astype(bool)
    feature_mask = tensor["feature_mask"].astype(bool)
    step_type = tensor.get("step_type", np.full(time_mask.shape, "all", dtype=object)).astype(str)
    feature_lookup = features.set_index(features["event_id"].astype(str), drop=False)
    rows: list[dict[str, object]] = []
    for idx, event_id in enumerate(event_ids):
        feature_row = feature_lookup.loc[event_id] if event_id in feature_lookup.index else None
        daily_mask = step_type[idx] == "daily"
        intraday_mask = step_type[idx] == "intraday"
        valid_len = int(time_mask[idx].sum())
        daily_valid = int(time_mask[idx, daily_mask].sum()) if bool(daily_mask.any()) else 0
        intraday_valid = (
            int(time_mask[idx, intraday_mask].sum()) if bool(intraday_mask.any()) else 0
        )
        missing_rate = 1.0 - float(feature_mask[idx].mean()) if feature_mask[idx].size else 1.0
        high_quality = bool(intraday_valid >= 8 and missing_rate <= 0.50)
        rows.append(
            {
                "event_id": event_id,
                "ticker": None if feature_row is None else str(feature_row.get("ticker", "")),
                "split": None if feature_row is None else str(feature_row.get("split", "")),
                "valid_len": valid_len,
                "daily_valid_len": daily_valid,
                "intraday_valid_len": intraday_valid,
                "missing_rate": missing_rate,
                "high_quality_sequence": high_quality,
                "common_row_eligible": bool(high_quality and feature_row is not None),
                "sequence_gate_reason": "eligible" if high_quality else "low_quality_or_sparse",
            }
        )
    return pd.DataFrame(rows)


def _bootstrap_auc_lift(
    clean: pd.DataFrame,
    *,
    score_a: str,
    score_b: str,
    realized_edge_col: str = "edge_var_realized",
    cluster_col: str = "event_id",
    n_iter: int = 200,
    seed: int = 17,
) -> dict[str, float | int | None]:
    if clean.empty or n_iter <= 0:
        return {"bootstrap_iter": 0, "auc_lift_ci_low": None, "auc_lift_ci_high": None}
    if score_a == score_b:
        return {"bootstrap_iter": int(n_iter), "auc_lift_ci_low": 0.0, "auc_lift_ci_high": 0.0}
    required = [cluster_col, realized_edge_col, score_a, score_b]
    if any(column not in clean.columns for column in required):
        return {"bootstrap_iter": 0, "auc_lift_ci_low": None, "auc_lift_ci_high": None}
    base = clean[required].copy()
    base[realized_edge_col] = pd.to_numeric(base[realized_edge_col], errors="coerce")
    base[score_a] = pd.to_numeric(base[score_a], errors="coerce")
    base[score_b] = pd.to_numeric(base[score_b], errors="coerce")
    base = base.dropna()
    if base.empty:
        return {"bootstrap_iter": 0, "auc_lift_ci_low": None, "auc_lift_ci_high": None}
    rng = np.random.default_rng(seed)
    clusters = base[cluster_col].astype(str).to_numpy()
    unique = np.asarray(sorted(pd.unique(pd.Series(clusters))), dtype=object)
    cluster_indices = {cluster: np.flatnonzero(clusters == cluster) for cluster in unique}
    edge = base[realized_edge_col].to_numpy(dtype=float) > 0
    values_a = base[score_a].to_numpy(dtype=float)
    values_b = base[score_b].to_numpy(dtype=float)
    lifts: list[float] = []
    for _ in range(n_iter):
        sampled = rng.choice(unique, size=len(unique), replace=True)
        sample_idx = np.concatenate([cluster_indices[str(cluster)] for cluster in sampled])
        auc_a = auc_score(edge[sample_idx], values_a[sample_idx])
        auc_b = auc_score(edge[sample_idx], values_b[sample_idx])
        if auc_a is not None and auc_b is not None:
            lifts.append(float(auc_a) - float(auc_b))
    if not lifts:
        return {"bootstrap_iter": 0, "auc_lift_ci_low": None, "auc_lift_ci_high": None}
    return {
        "bootstrap_iter": int(len(lifts)),
        "auc_lift_ci_low": float(np.quantile(lifts, 0.025)),
        "auc_lift_ci_high": float(np.quantile(lifts, 0.975)),
    }


def build_common_row_diagnostics(
    predictions: pd.DataFrame,
    *,
    out_dir: Path,
    bootstrap_iter: int = 200,
) -> dict[str, str]:
    out_dir.mkdir(parents=True, exist_ok=True)
    forecast_columns = model_forecast_columns(predictions)
    universe_rows: list[dict[str, object]] = []
    pair_rows: list[dict[str, object]] = []
    bootstrap_rows: list[dict[str, object]] = []
    incremental_rows: list[dict[str, object]] = []
    sequence_diag_rows: list[dict[str, object]] = []
    for target_id, group in predictions.groupby("target_id", dropna=False):
        target_id = str(target_id)
        test = group.loc[group["split"].eq("test")].copy()
        validation = group.loc[group["split"].eq("validation")].copy()
        for model_id, column in forecast_columns.items():
            if column not in group.columns:
                continue
            available = pd.to_numeric(group[column], errors="coerce").notna()
            for _, row in group.loc[available, ["event_id", "split"]].iterrows():
                universe_rows.append(
                    {
                        "target_id": target_id,
                        "event_id": row["event_id"],
                        "split": row["split"],
                        "model_id": model_id,
                        "forecast_available": True,
                    }
                )
        score_columns: dict[str, str] = {}
        for model_id, column in forecast_columns.items():
            score_col = f"_score_{model_id}"
            if column in test:
                test[score_col] = pd.to_numeric(test[column], errors="coerce") - pd.to_numeric(
                    test["ivar_event"], errors="coerce"
                )
                validation[score_col] = pd.to_numeric(
                    validation.get(column, pd.Series(np.nan, index=validation.index)),
                    errors="coerce",
                ) - pd.to_numeric(validation["ivar_event"], errors="coerce")
                score_columns[model_id] = score_col
        model_ids = sorted(score_columns)
        for left_idx, model_a in enumerate(model_ids):
            for model_b in model_ids[left_idx + 1 :]:
                score_a = score_columns[model_a]
                score_b = score_columns[model_b]
                keep = ["event_id", "edge_var_realized", score_a, score_b]
                clean = test[keep].dropna().copy()
                if clean.empty:
                    continue
                metrics_a = ranking_metrics(clean, score_col=score_a)
                metrics_b = ranking_metrics(clean, score_col=score_b)
                auc_a = metrics_a.get("auc")
                auc_b = metrics_b.get("auc")
                lift = None if auc_a is None or auc_b is None else float(auc_a) - float(auc_b)
                ci = _bootstrap_auc_lift(
                    clean,
                    score_a=score_a,
                    score_b=score_b,
                    n_iter=bootstrap_iter,
                )
                row = {
                    "target_id": target_id,
                    "model_a": model_a,
                    "model_b": model_b,
                    "common_rows": int(len(clean)),
                    "auc_a": auc_a,
                    "auc_b": auc_b,
                    "auc_lift": lift,
                    **ci,
                }
                pair_rows.append(row)
                bootstrap_rows.append(row)
        baseline_col: str | None = None
        for candidate in (
            "forecast_lightgbm_xgboost_mean_ensemble",
            "forecast_lightgbm",
            "forecast_xgboost",
        ):
            if candidate in group.columns:
                baseline_col = candidate
                break
        if baseline_col is not None:
            for model_id in sorted(SEQUENCE_MODEL_IDS):
                sequence_column = forecast_columns.get(model_id)
                if sequence_column is None or sequence_column not in group:
                    continue
                val_clean = validation[[baseline_col, sequence_column, "rvar_event"]].dropna()
                test_clean = test[
                    [
                        "event_id",
                        baseline_col,
                        sequence_column,
                        "rvar_event",
                        "ivar_event",
                        "edge_var_realized",
                    ]
                ].dropna()
                if len(val_clean) < 5 or len(test_clean) < 5:
                    continue
                x_val = np.column_stack(
                    [
                        np.ones(len(val_clean)),
                        pd.to_numeric(val_clean[baseline_col], errors="coerce").to_numpy(
                            dtype=float
                        ),
                        pd.to_numeric(val_clean[sequence_column], errors="coerce").to_numpy(
                            dtype=float
                        ),
                    ]
                )
                y_val = pd.to_numeric(val_clean["rvar_event"], errors="coerce").to_numpy(
                    dtype=float
                )
                coef, *_ = np.linalg.lstsq(x_val, y_val, rcond=None)
                x_test = np.column_stack(
                    [
                        np.ones(len(test_clean)),
                        pd.to_numeric(test_clean[baseline_col], errors="coerce").to_numpy(
                            dtype=float
                        ),
                        pd.to_numeric(test_clean[sequence_column], errors="coerce").to_numpy(
                            dtype=float
                        ),
                    ]
                )
                stacked = x_test @ coef
                score_base = pd.to_numeric(
                    test_clean[baseline_col], errors="coerce"
                ) - pd.to_numeric(test_clean["ivar_event"], errors="coerce")
                score_stacked = stacked - pd.to_numeric(test_clean["ivar_event"], errors="coerce")
                eval_frame = test_clean.assign(_score_base=score_base, _score_stacked=score_stacked)
                auc_base = ranking_metrics(eval_frame, score_col="_score_base").get("auc")
                auc_stacked = ranking_metrics(eval_frame, score_col="_score_stacked").get("auc")
                incremental_rows.append(
                    {
                        "target_id": target_id,
                        "sequence_model_id": model_id,
                        "baseline_forecast_col": baseline_col,
                        "validation_rows": int(len(val_clean)),
                        "test_rows": int(len(test_clean)),
                        "auc_base": auc_base,
                        "auc_stacked": auc_stacked,
                        "auc_lift": None
                        if auc_base is None or auc_stacked is None
                        else float(auc_stacked) - float(auc_base),
                        "meta_model": "validation_ols_locked_test",
                    }
                )
        mask_auc: float | None = None
        shuffle_auc: float | None = None
        best_non_mamba_auc: float | None = None
        mask_score = score_columns.get("mask_only_sequence")
        if mask_score is not None:
            mask_auc = ranking_metrics(test.dropna(subset=[mask_score]), score_col=mask_score).get(
                "auc"
            )
        shuffle_score = score_columns.get("time_shuffle_sequence")
        if shuffle_score is not None:
            shuffle_auc = ranking_metrics(
                test.dropna(subset=[shuffle_score]), score_col=shuffle_score
            ).get("auc")
        non_mamba_candidates = [
            model_id
            for model_id in score_columns
            if model_id in REAL_SEQUENCE_MODEL_IDS and model_id != "mamba_ssm_sequence"
        ]
        non_mamba_auc_values: list[float] = []
        for model_id in non_mamba_candidates:
            value = ranking_metrics(
                test.dropna(subset=[score_columns[model_id]]),
                score_col=score_columns[model_id],
            ).get("auc")
            if value is not None:
                non_mamba_auc_values.append(float(value))
        if non_mamba_auc_values:
            best_non_mamba_auc = max(non_mamba_auc_values)
        for model_id in sorted(SEQUENCE_MODEL_IDS):
            score = score_columns.get(model_id)
            if score is None:
                continue
            metrics = ranking_metrics(test.dropna(subset=[score]), score_col=score)
            auc = metrics.get("auc")
            mask_lift = None if auc is None or mask_auc is None else float(auc) - float(mask_auc)
            shuffle_lift = (
                None if auc is None or shuffle_auc is None else float(auc) - float(shuffle_auc)
            )
            best_non_mamba_lift = (
                None
                if auc is None or best_non_mamba_auc is None
                else float(auc) - float(best_non_mamba_auc)
            )
            mask_ci_low = None
            mask_ci_high = None
            if mask_score is not None and score == mask_score:
                mask_ci_low = 0.0
                mask_ci_high = 0.0
            elif mask_score is not None:
                clean_mask = test[["event_id", "edge_var_realized", score, mask_score]].dropna()
                mask_ci = _bootstrap_auc_lift(
                    clean_mask,
                    score_a=score,
                    score_b=mask_score,
                    n_iter=bootstrap_iter,
                )
                mask_ci_low = mask_ci.get("auc_lift_ci_low")
                mask_ci_high = mask_ci.get("auc_lift_ci_high")
            shuffle_ci_low = None
            shuffle_ci_high = None
            if shuffle_score is not None and score == shuffle_score:
                shuffle_ci_low = 0.0
                shuffle_ci_high = 0.0
            elif shuffle_score is not None:
                clean_shuffle = test[
                    ["event_id", "edge_var_realized", score, shuffle_score]
                ].dropna()
                shuffle_ci = _bootstrap_auc_lift(
                    clean_shuffle,
                    score_a=score,
                    score_b=shuffle_score,
                    n_iter=bootstrap_iter,
                )
                shuffle_ci_low = shuffle_ci.get("auc_lift_ci_low")
                shuffle_ci_high = shuffle_ci.get("auc_lift_ci_high")
            control_lifts = [value for value in (mask_lift, shuffle_lift) if value is not None]
            control_ci_lows = [
                float(value) for value in (mask_ci_low, shuffle_ci_low) if value is not None
            ]
            control_ci_highs = [
                float(value) for value in (mask_ci_high, shuffle_ci_high) if value is not None
            ]
            diagnostic = bool(
                model_id in REAL_SEQUENCE_MODEL_IDS
                and mask_lift is not None
                and shuffle_lift is not None
                and mask_lift >= 0.05
                and shuffle_lift >= 0.05
                and control_ci_lows
                and min(control_ci_lows) > 0
            )
            coverage = int(metrics.get("n") or 0)
            available_test_rows = int(len(test))
            sequence_diag_rows.append(
                {
                    "model_id": model_id,
                    "target_id": target_id,
                    "coverage": coverage,
                    "drop_rate": None
                    if available_test_rows == 0
                    else 1.0 - (coverage / available_test_rows),
                    "auc_lift": None if not control_lifts else min(control_lifts),
                    "auc_lift_ci_low": None if not control_ci_lows else min(control_ci_lows),
                    "auc_lift_ci_high": None if not control_ci_highs else min(control_ci_highs),
                    "mask_only_lift": mask_lift,
                    "time_shuffle_lift": shuffle_lift,
                    "best_non_mamba_lift": best_non_mamba_lift,
                    "headline_eligible": False,
                    "claim_scope": "diagnostic_grade_signal_indication"
                    if diagnostic
                    else "diagnostic",
                    "fail_reason": None if diagnostic else "gate_not_passed_or_control_missing",
                }
            )
    paths = {
        "common_row_universe": out_dir / "common_row_universe.csv",
        "common_row_pairwise_metrics": out_dir / "common_row_pairwise_metrics.csv",
        "incremental_value_diagnostics": out_dir / "incremental_value_diagnostics.csv",
        "clustered_bootstrap_ci": out_dir / "clustered_bootstrap_ci.csv",
        "sequence_model_fit_diagnostics": out_dir / "sequence_model_fit_diagnostics.csv",
    }
    pd.DataFrame(universe_rows).to_csv(paths["common_row_universe"], index=False)
    pd.DataFrame(pair_rows).to_csv(paths["common_row_pairwise_metrics"], index=False)
    pd.DataFrame(incremental_rows).to_csv(paths["incremental_value_diagnostics"], index=False)
    pd.DataFrame(bootstrap_rows).to_csv(paths["clustered_bootstrap_ci"], index=False)
    pd.DataFrame(sequence_diag_rows).to_csv(paths["sequence_model_fit_diagnostics"], index=False)
    return {key: str(value) for key, value in paths.items()}


def build_metric_tables(predictions: pd.DataFrame, *, out_dir: Path) -> dict[str, str]:
    out_dir.mkdir(parents=True, exist_ok=True)
    remove_model_level_csv_artifacts(out_dir)
    forecast_columns = model_forecast_columns(predictions)
    forecast_rows: list[dict[str, object]] = []
    ranking_rows: list[dict[str, object]] = []
    strategy_rows: list[dict[str, object]] = []
    cost_rows: list[pd.DataFrame] = []
    breakdown_frames: list[pd.DataFrame] = []
    test_all = predictions.loc[predictions["split"].eq("test")].copy()
    groups = (
        list(test_all.groupby("target_id", dropna=False))
        if "target_id" in test_all.columns
        else [("day_c2c", test_all)]
    )
    for target_id, test in groups:
        target_id = str(target_id)
        for model_id, column in forecast_columns.items():
            scored = test.copy()
            if column not in scored:
                continue
            scored[f"score_{model_id}"] = pd.to_numeric(
                scored[column], errors="coerce"
            ) - pd.to_numeric(scored["ivar_event"], errors="coerce")
            forecast_rows.append(
                {
                    "target_id": target_id,
                    "model_id": model_id,
                    **forecast_metrics(scored, forecast_col=column),
                }
            )
            if "edge_var_realized" in scored:
                ranking_rows.append(
                    {
                        "target_id": target_id,
                        "model_id": model_id,
                        **ranking_metrics(scored, score_col=f"score_{model_id}"),
                    }
                )
                edge_decile_table(scored, score_col=f"score_{model_id}").to_csv(
                    out_dir / f"edge_deciles_{target_id}_{model_id}.csv", index=False
                )
            strategy_specs = {
                "day_c2c": [
                    {
                        "realized_col": "gross_proxy_pnl_usd",
                        "proxy_kind": "day_c2c_exit_preclose_15m_proxy",
                        "headline_eligible": True,
                        "trade_prefix": "strategy_trades",
                    }
                ],
                "jump_c2o": [
                    {
                        "realized_col": "gross_post_open_option_vwap_5_15_proxy_pnl_usd",
                        "proxy_kind": "post_open_option_vwap_5_15_proxy",
                        "headline_eligible": False,
                        "trade_prefix": "c2o_option_vwap_5_15_strategy_trades",
                    },
                    {
                        "realized_col": "gross_post_open_option_vwap_0_5_proxy_pnl_usd",
                        "proxy_kind": "post_open_option_vwap_0_5_proxy",
                        "headline_eligible": False,
                        "trade_prefix": "c2o_option_vwap_0_5_strategy_trades",
                    },
                    {
                        "realized_col": "gross_c2o_intrinsic_proxy_pnl_usd",
                        "proxy_kind": "c2o_intrinsic_open_diagnostic",
                        "headline_eligible": False,
                        "trade_prefix": "c2o_intrinsic_strategy_trades",
                    },
                ],
                "reaction_o2c": [
                    {
                        "realized_col": (
                            "gross_reaction_o2c_option_vwap_5_15_to_c2c_exit_proxy_pnl_usd"
                        ),
                        "proxy_kind": "reaction_o2c_option_vwap_5_15_to_c2c_exit_proxy",
                        "headline_eligible": False,
                        "trade_prefix": "o2c_option_vwap_5_15_strategy_trades",
                        "entry_premium_col": "open_option_vwap_5_15_anchor_usd",
                        "cost_col": "open_option_vwap_5_15_proxy_cost_usd",
                    },
                    {
                        "realized_col": (
                            "gross_reaction_o2c_option_vwap_0_5_to_c2c_exit_proxy_pnl_usd"
                        ),
                        "proxy_kind": "reaction_o2c_option_vwap_0_5_to_c2c_exit_proxy",
                        "headline_eligible": False,
                        "trade_prefix": "o2c_option_vwap_0_5_strategy_trades",
                        "entry_premium_col": "open_option_vwap_0_5_anchor_usd",
                        "cost_col": "open_option_vwap_0_5_proxy_cost_usd",
                    },
                ],
            }.get(target_id, [])
            if not strategy_specs:
                continue
            for strategy_spec in strategy_specs:
                realized_col = str(strategy_spec["realized_col"])
                entry_premium_col = str(strategy_spec.get("entry_premium_col", "entry_premium_usd"))
                cost_col = str(strategy_spec.get("cost_col", "proxy_cost_usd"))
                if {realized_col, entry_premium_col}.issubset(scored.columns):
                    proxy_kind = str(strategy_spec["proxy_kind"])
                    headline_eligible = bool(strategy_spec["headline_eligible"])
                    trade_prefix = str(strategy_spec["trade_prefix"])
                    strategy = build_proxy_strategy_frame(
                        scored,
                        forecast_col=column,
                        realized_long_pnl_col=realized_col,
                        entry_premium_col=entry_premium_col,
                        cost_col=cost_col,
                        min_edge_var=0.0,
                    )
                    strategy["strategy_proxy_kind"] = proxy_kind
                    strategy["pnl_headline_eligible"] = headline_eligible
                    trades = strategy.loc[strategy["should_trade"].astype(bool)].copy()
                    trades.to_csv(out_dir / f"{trade_prefix}_{model_id}.csv", index=False)
                    strategy_rows.append(
                        {
                            "target_id": target_id,
                            "model_id": model_id,
                            "strategy_proxy_kind": proxy_kind,
                            "pnl_headline_eligible": headline_eligible,
                            **strategy_metrics(trades, gross_pnl_col="gross_strategy_pnl_usd"),
                        }
                    )
                    sensitivity = cost_sensitivity(
                        trades,
                        gross_pnl_col="gross_strategy_pnl_usd",
                        cost_col="estimated_transaction_cost_usd",
                        multipliers=(0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 5.0),
                    )
                    sensitivity.insert(0, "target_id", target_id)
                    sensitivity.insert(1, "model_id", model_id)
                    sensitivity.insert(2, "strategy_proxy_kind", proxy_kind)
                    sensitivity.insert(3, "pnl_headline_eligible", headline_eligible)
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
                            table.insert(0, "target_id", target_id)
                            table.insert(1, "model_id", model_id)
                            table.insert(2, "strategy_proxy_kind", proxy_kind)
                            table.insert(3, "pnl_headline_eligible", headline_eligible)
                            table.insert(4, "breakdown", breakdown)
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
    o2c_scale_path = out_dir / "o2c_scale_diagnostic.csv"
    o2c_scale_diagnostic(predictions).to_csv(o2c_scale_path, index=False)
    qlike_frames: list[pd.DataFrame] = []
    extreme_frames: list[pd.DataFrame] = []
    inference_frames: list[pd.DataFrame] = []
    for target_id, group in (
        list(predictions.groupby("target_id", dropna=False))
        if "target_id" in predictions
        else [("day_c2c", predictions)]
    ):
        qlike_one, extremes_one = qlike_sanity_table(group, forecast_columns=forecast_columns)
        qlike_one.insert(0, "target_id", str(target_id))
        if not extremes_one.empty:
            extremes_one.insert(0, "target_id", str(target_id))
        inference_one = inference_table(group, forecast_columns=forecast_columns)
        inference_one.insert(0, "target_id", str(target_id))
        qlike_frames.append(qlike_one)
        extreme_frames.append(extremes_one)
        inference_frames.append(inference_one)
    qlike = pd.concat(qlike_frames, ignore_index=True) if qlike_frames else pd.DataFrame()
    extremes = pd.concat(extreme_frames, ignore_index=True) if extreme_frames else pd.DataFrame()
    qlike_path = out_dir / "qlike_sanity.csv"
    extreme_path = out_dir / "extreme_predictions.csv"
    qlike.to_csv(qlike_path, index=False)
    extremes.to_csv(extreme_path, index=False)
    inference_path = out_dir / "inference.csv"
    (pd.concat(inference_frames, ignore_index=True) if inference_frames else pd.DataFrame()).to_csv(
        inference_path, index=False
    )
    return {
        "forecast_metrics": str(forecast_path),
        "ranking_metrics": str(ranking_path),
        "strategy_metrics": str(strategy_path),
        "cost_sensitivity": str(cost_path),
        "strategy_breakdowns": str(breakdown_path),
        "o2c_scale_diagnostic": str(o2c_scale_path),
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

    figure_labels = {
        "market_implied_event_variance": "Market IVAR",
        "last_four_rvar": "Last-four RVAR",
        "last_four_ivar": "Last-four IVAR",
        "goyal_saretto_rv_iv_spread": "Goyal-Saretto spread",
        "linear_elastic_net": "Elastic Net",
        "linear_elastic_net_tuned": "Elastic Net tuned",
        "lightgbm": "LightGBM",
        "lightgbm_tuned": "LightGBM tuned",
        "xgboost": "XGBoost",
        "xgboost_tuned": "XGBoost tuned",
        "lightgbm_xgboost_mean_ensemble": "LightGBM/XGBoost ensemble",
        "ft_transformer": "FT-Transformer",
        "ft_transformer_tuned": "FT-Transformer tuned",
        "ridge_flat_aggregates_sequence": "Ridge-flat sequence",
        "bigru_sequence": "BiGRU sequence",
        "bigru_sequence_5seed": "BiGRU 5-seed",
        "mamba_ssm_sequence": "Official mamba-ssm",
        "mamba_ssm_sequence_5seed": "Official mamba-ssm 5-seed",
        "mask_only_sequence": "Mask-only sequence",
        "time_shuffle_sequence": "Time-shuffle sequence",
        "SD RVAR reaction_o2c": "SD RVAR reaction_o2c",
        "SD IVAR event": "SD IVAR_event",
    }

    def figure_label(value: object) -> str:
        raw = str(value)
        return figure_labels.get(raw, raw.replace("_", " "))

    def figure_height(row_count: int, *, series_count: int = 1) -> float:
        grouped_extra = max(series_count - 1, 0) * 0.12
        return min(9.0, max(3.6, 1.4 + row_count * (0.34 + grouped_extra)))

    def with_figure_labels(data: pd.DataFrame, x_col: str) -> pd.DataFrame:
        plot_data = data.copy()
        plot_data["_figure_label"] = plot_data[x_col].map(figure_label)
        return plot_data

    model_plot_groups = {
        "market_implied_event_variance": "Benchmarks",
        "last_four_rvar": "Benchmarks",
        "last_four_ivar": "Benchmarks",
        "goyal_saretto_rv_iv_spread": "Benchmarks",
        "linear_elastic_net": "Tabular ML",
        "linear_elastic_net_tuned": "Tabular ML",
        "lightgbm": "Tabular ML",
        "lightgbm_tuned": "Tabular ML",
        "xgboost": "Tabular ML",
        "xgboost_tuned": "Tabular ML",
        "lightgbm_xgboost_mean_ensemble": "Tabular ML",
        "ft_transformer": "Deep/sequence",
        "ft_transformer_tuned": "Deep/sequence",
        "bigru_sequence": "Deep/sequence",
        "bigru_sequence_5seed": "Deep/sequence",
        "mamba_ssm_sequence": "Deep/sequence",
        "mamba_ssm_sequence_5seed": "Deep/sequence",
        "ridge_flat_aggregates_sequence": "Sequence controls",
        "mask_only_sequence": "Sequence controls",
        "time_shuffle_sequence": "Sequence controls",
    }
    model_plot_order = {
        model_id: index
        for index, model_id in enumerate(
            [
                "market_implied_event_variance",
                "last_four_rvar",
                "last_four_ivar",
                "goyal_saretto_rv_iv_spread",
                "linear_elastic_net",
                "linear_elastic_net_tuned",
                "lightgbm",
                "lightgbm_tuned",
                "xgboost",
                "xgboost_tuned",
                "lightgbm_xgboost_mean_ensemble",
                "ft_transformer",
                "ft_transformer_tuned",
                "bigru_sequence",
                "bigru_sequence_5seed",
                "mamba_ssm_sequence",
                "mamba_ssm_sequence_5seed",
                "ridge_flat_aggregates_sequence",
                "mask_only_sequence",
                "time_shuffle_sequence",
            ]
        )
    }
    model_family_colors = {
        "market_implied_event_variance": "#7f7f7f",
        "last_four_rvar": "#9467bd",
        "last_four_ivar": "#8c564b",
        "goyal_saretto_rv_iv_spread": "#17becf",
        "linear_elastic_net": "#1f77b4",
        "lightgbm": "#2ca02c",
        "xgboost": "#ff7f0e",
        "lightgbm_xgboost_mean_ensemble": "#111827",
        "ft_transformer": "#9467bd",
        "bigru_sequence": "#d62728",
        "mamba_ssm_sequence": "#17becf",
        "ridge_flat_aggregates_sequence": "#4b5563",
        "mask_only_sequence": "#8c564b",
        "time_shuffle_sequence": "#bcbd22",
    }

    def model_plot_group(model_id: object) -> str:
        return model_plot_groups.get(str(model_id), "Other")

    def model_plot_family(model_id: object) -> str:
        raw = str(model_id)
        if raw.endswith("_tuned"):
            return raw.removesuffix("_tuned")
        if raw.endswith("_5seed"):
            return raw.removesuffix("_5seed")
        return raw

    def model_plot_style(model_id: object) -> dict[str, object]:
        raw = str(model_id)
        family = model_plot_family(raw)
        style: dict[str, object] = {
            "color": model_family_colors.get(family, "#6b7280"),
            "linestyle": "-",
            "marker": "o",
            "linewidth": 1.7,
            "markersize": 3.8,
        }
        if raw.endswith("_tuned"):
            style.update({"linestyle": "--", "marker": "s"})
        elif raw.endswith("_5seed"):
            style.update({"linestyle": ":", "marker": "D"})
        return style

    def model_plot_order_key(model_id: object) -> tuple[int, str]:
        raw = str(model_id)
        return (model_plot_order.get(raw, len(model_plot_order)), raw)

    outputs: dict[str, str] = {}

    def write_cost_sensitivity_figure(
        data: pd.DataFrame,
        *,
        value_col: str,
        fig_name: str,
        title: str,
    ) -> None:
        fig, axes_grid = plt.subplots(2, 2, figsize=(11.0, 7.2), sharex=True)
        axes = list(np.ravel(axes_grid))
        plot_data = data.copy()
        plot_data["cost_multiplier"] = pd.to_numeric(plot_data["cost_multiplier"], errors="coerce")
        plot_data[value_col] = pd.to_numeric(plot_data[value_col], errors="coerce")
        plot_data["_trade_n"] = (
            pd.to_numeric(plot_data["n"], errors="coerce").fillna(0)
            if "n" in plot_data.columns
            else pd.Series(np.ones(len(plot_data)), index=plot_data.index)
        )
        plot_data["_plot_group"] = plot_data["model_id"].map(model_plot_group)
        plot_data = plot_data.loc[plot_data["cost_multiplier"].notna()].copy()
        for ax, group_name in zip(
            axes,
            ["Benchmarks", "Tabular ML", "Deep/sequence", "Sequence controls"],
            strict=True,
        ):
            group = plot_data.loc[plot_data["_plot_group"].eq(group_name)].copy()
            skipped: list[str] = []
            plotted = 0
            for model_id in sorted(group["model_id"].dropna().unique(), key=model_plot_order_key):
                model_rows = group.loc[group["model_id"].astype(str).eq(str(model_id))].sort_values(
                    "cost_multiplier"
                )
                max_n = int(pd.to_numeric(model_rows["_trade_n"], errors="coerce").max())
                valid = model_rows[value_col].notna()
                if max_n <= 0 or not bool(valid.any()):
                    skipped.append(figure_label(model_id))
                    continue
                ax.plot(
                    model_rows.loc[valid, "cost_multiplier"],
                    model_rows.loc[valid, value_col],
                    label=f"{figure_label(model_id)} (n={max_n})",
                    **model_plot_style(model_id),
                )
                plotted += 1
            ax.axhline(0, color="#6b7280", linewidth=0.8, alpha=0.75)
            ax.set_title(group_name, fontsize=10)
            ax.grid(axis="y", alpha=0.18)
            ax.grid(axis="x", alpha=0.22)
            ax.set_axisbelow(True)
            if plotted:
                ax.legend(
                    fontsize=6.1 if plotted > 5 else 6.8,
                    frameon=False,
                    loc="best",
                    ncol=2 if plotted > 5 else 1,
                    handlelength=1.7,
                    columnspacing=0.8,
                    labelspacing=0.24,
                    borderaxespad=0.25,
                )
            if skipped:
                ax.text(
                    0.02,
                    0.03,
                    "No trades: " + ", ".join(skipped),
                    transform=ax.transAxes,
                    fontsize=6.8,
                    color="#4b5563",
                    va="bottom",
                )
        fig.suptitle(title, fontsize=12)
        fig.supxlabel("Cost multiplier", fontsize=9)
        fig.supylabel("Net PnL (USD)", fontsize=9)
        fig.tight_layout(rect=(0.02, 0.02, 1.0, 0.95))
        path = figures_dir / fig_name
        fig.savefig(path, dpi=160)
        plt.close(fig)
        outputs[fig_name.removesuffix(".png")] = str(path)

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
        if "target_id" in data.columns:
            if csv_name in {"forecast_metrics.csv", "ranking_metrics.csv", "qlike_sanity.csv"}:
                data = data.loc[data["target_id"].astype(str).eq("jump_c2o")].copy()
            elif csv_name in {"strategy_metrics.csv", "cost_sensitivity.csv"}:
                data = data.loc[data["target_id"].astype(str).eq("day_c2c")].copy()
        if not data.empty and value_col in data.columns and "model_id" in data.columns:
            if csv_name == "cost_sensitivity.csv" and "cost_multiplier" in data.columns:
                write_cost_sensitivity_figure(
                    data,
                    value_col=value_col,
                    fig_name=fig_name,
                    title=title,
                )
                continue
            else:
                plot_data = with_figure_labels(data, "model_id").sort_values(
                    value_col, ascending=True
                )
                fig, ax = plt.subplots(figsize=(8.8, figure_height(len(plot_data))))
                plot_data.plot.barh(x="_figure_label", y=value_col, ax=ax, legend=False)
        else:
            fig, ax = plt.subplots(figsize=(8.8, 3.8))
        ax.set_title(title)
        ax.set_ylabel("")
        ax.grid(axis="x", alpha=0.25)
        ax.set_axisbelow(True)
        fig.tight_layout()
        path = figures_dir / fig_name
        fig.savefig(path, dpi=160)
        plt.close(fig)
        outputs[fig_name.removesuffix(".png")] = str(path)

    def write_bar_figure(
        data: pd.DataFrame,
        *,
        value_cols: str | list[str],
        fig_name: str,
        title: str,
        x_col: str = "model_id",
    ) -> None:
        cols = [value_cols] if isinstance(value_cols, str) else value_cols
        if not data.empty and x_col in data.columns and all(col in data.columns for col in cols):
            plot_cols: str | list[str] = cols[0] if len(cols) == 1 else cols
            plot_data = with_figure_labels(data, x_col)
            if cols[0] in plot_data.columns and x_col != "series":
                plot_data = plot_data.sort_values(cols[0], ascending=True)
            fig, ax = plt.subplots(
                figsize=(8.8, figure_height(len(plot_data), series_count=len(cols)))
            )
            plot_data.plot.barh(x="_figure_label", y=plot_cols, ax=ax, legend=len(cols) > 1)
            if len(cols) > 1:
                ax.legend(fontsize=8, frameon=False)
        else:
            fig, ax = plt.subplots(figsize=(8.8, 3.8))
        ax.set_title(title)
        ax.set_ylabel("")
        ax.grid(axis="x", alpha=0.25)
        ax.set_axisbelow(True)
        fig.tight_layout()
        path = figures_dir / fig_name
        fig.savefig(path, dpi=160)
        plt.close(fig)
        outputs[fig_name.removesuffix(".png")] = str(path)

    o2c_forecast = read_csv("forecast_metrics.csv")
    if "target_id" in o2c_forecast.columns:
        o2c_forecast = o2c_forecast.loc[
            o2c_forecast["target_id"].astype(str).eq("reaction_o2c")
        ].copy()
    write_bar_figure(
        o2c_forecast,
        value_cols="mae",
        fig_name="o2c_forecast_performance.png",
        title="O2C Forecast MAE",
    )

    o2c_ranking = read_csv("ranking_metrics.csv")
    if "target_id" in o2c_ranking.columns:
        o2c_ranking = o2c_ranking.loc[
            o2c_ranking["target_id"].astype(str).eq("reaction_o2c")
        ].copy()
    write_bar_figure(
        o2c_ranking,
        value_cols=["auc", "top_decile_precision"],
        fig_name="o2c_auc_top_decile_precision.png",
        title="O2C Ranking Quality",
    )

    o2c_strategy = read_csv("strategy_metrics.csv")
    if {"target_id", "strategy_proxy_kind"}.issubset(o2c_strategy.columns):
        o2c_strategy = o2c_strategy.loc[
            o2c_strategy["target_id"].astype(str).eq("reaction_o2c")
            & o2c_strategy["strategy_proxy_kind"]
            .astype(str)
            .eq("reaction_o2c_option_vwap_5_15_to_c2c_exit_proxy")
        ].copy()
    write_bar_figure(
        o2c_strategy,
        value_cols="net_pnl_usd",
        fig_name="o2c_strategy_proxy_pnl.png",
        title="O2C 5-15m Diagnostic Proxy Net PnL",
    )

    o2c_scale = read_csv("o2c_scale_diagnostic.csv")
    if {"sd_rvar_reaction_o2c", "sd_ivar_event"}.issubset(o2c_scale.columns):
        scale_plot = pd.DataFrame(
            {
                "series": ["SD RVAR reaction_o2c", "SD IVAR event"],
                "value": [
                    float(o2c_scale["sd_rvar_reaction_o2c"].iloc[0]),
                    float(o2c_scale["sd_ivar_event"].iloc[0]),
                ],
            }
        )
    else:
        scale_plot = pd.DataFrame(columns=["series", "value"])
    write_bar_figure(
        scale_plot,
        value_cols="value",
        fig_name="o2c_scale_diagnostic.png",
        title="O2C Scale Mismatch",
        x_col="series",
    )
    predictions_path = artifacts_dir / "model_predictions.parquet"
    fig, ax = plt.subplots(figsize=(5, 5))
    if predictions_path.exists():
        predictions = pd.read_parquet(predictions_path)
        if "target_id" in predictions.columns:
            predictions = predictions.loc[predictions["target_id"].astype(str).eq("jump_c2o")]
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
    cost = (
        pd.read_csv(artifacts_dir / "cost_sensitivity.csv")
        if (artifacts_dir / "cost_sensitivity.csv").exists()
        else pd.DataFrame()
    )
    o2c_scale = (
        pd.read_csv(artifacts_dir / "o2c_scale_diagnostic.csv")
        if (artifacts_dir / "o2c_scale_diagnostic.csv").exists()
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
    tuning_selected_path = artifacts_dir / "tuning_selected_params.json"
    tuning_selected = (
        json.loads(tuning_selected_path.read_text(encoding="utf-8"))
        if tuning_selected_path.exists()
        else {}
    )
    tuning_profile = str(tuning_selected.get("tuning_profile", "untuned"))
    selected_models = [
        "market_implied_event_variance",
        "last_four_rvar",
        "last_four_ivar",
        "goyal_saretto_rv_iv_spread",
        "linear_elastic_net",
        "linear_elastic_net_tuned",
        "lightgbm",
        "lightgbm_tuned",
        "xgboost",
        "xgboost_tuned",
        "lightgbm_xgboost_mean_ensemble",
        "ft_transformer",
        "ft_transformer_tuned",
        "ridge_flat_aggregates_sequence",
        "bigru_sequence",
        "bigru_sequence_5seed",
        "mamba_ssm_sequence",
        "mamba_ssm_sequence_5seed",
        "mask_only_sequence",
        "time_shuffle_sequence",
        "attention_pooling_sequence",
        "dilated_cnn_sequence",
    ]
    forecast_main = (
        forecast.loc[forecast["target_id"].astype(str).eq("jump_c2o")].copy()
        if "target_id" in forecast
        else forecast.copy()
    )
    ranking_main = (
        ranking.loc[ranking["target_id"].astype(str).eq("jump_c2o")].copy()
        if "target_id" in ranking
        else ranking.copy()
    )
    strategy_main = (
        strategy.loc[strategy["target_id"].astype(str).eq("day_c2c")].copy()
        if "target_id" in strategy
        else strategy.copy()
    )
    c2o_strategy_diag = (
        strategy.loc[strategy["target_id"].astype(str).eq("jump_c2o")].copy()
        if "target_id" in strategy
        else pd.DataFrame()
    )
    c2o_strategy_primary = (
        c2o_strategy_diag.loc[
            c2o_strategy_diag.get("strategy_proxy_kind", pd.Series(dtype=str))
            .astype(str)
            .eq("post_open_option_vwap_5_15_proxy")
        ].copy()
        if not c2o_strategy_diag.empty and "strategy_proxy_kind" in c2o_strategy_diag
        else pd.DataFrame()
    )
    o2c_forecast = (
        forecast.loc[forecast["target_id"].astype(str).eq("reaction_o2c")].copy()
        if "target_id" in forecast
        else pd.DataFrame()
    )
    o2c_ranking = (
        ranking.loc[ranking["target_id"].astype(str).eq("reaction_o2c")].copy()
        if "target_id" in ranking
        else pd.DataFrame()
    )
    o2c_strategy_diag = (
        strategy.loc[strategy["target_id"].astype(str).eq("reaction_o2c")].copy()
        if "target_id" in strategy
        else pd.DataFrame()
    )
    o2c_strategy_primary = (
        o2c_strategy_diag.loc[
            o2c_strategy_diag.get("strategy_proxy_kind", pd.Series(dtype=str))
            .astype(str)
            .eq("reaction_o2c_option_vwap_5_15_to_c2c_exit_proxy")
        ].copy()
        if not o2c_strategy_diag.empty and "strategy_proxy_kind" in o2c_strategy_diag
        else pd.DataFrame()
    )
    if not forecast_main.empty:
        summary = forecast_main.merge(
            ranking_main, on="model_id", how="left", suffixes=("", "_ranking")
        )
        summary = summary.merge(
            strategy_main, on="model_id", how="left", suffixes=("", "_strategy")
        )
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
        summary = summary.rename(
            columns={
                "mae": "mae_jump_c2o",
                "rmse": "rmse_jump_c2o",
                "oos_r2_vs_ivar": "oos_r2_vs_ivar_jump_c2o",
                "top_decile_precision": "top_decile_precision_jump_c2o",
                "auc": "auc_jump_c2o",
                "net_pnl_usd": "net_pnl_usd_day_c2c",
                "return_on_premium": "return_on_premium_day_c2c",
                "max_drawdown_usd": "max_drawdown_usd_day_c2c",
            }
        )
    else:
        summary = pd.DataFrame()

    def _markdown_table(frame: pd.DataFrame, empty_message: str) -> str:
        if frame.empty:
            return empty_message
        clean = frame.copy().astype(object)
        clean = clean.where(pd.notna(clean), "n/a")
        return str(clean.to_markdown(index=False))

    def _label(model_id: object) -> str:
        return str(model_id).replace("_", " ")

    def _value(frame: pd.DataFrame, model_id: str, metric: str) -> float | None:
        if frame.empty or metric not in frame.columns or "model_id" not in frame.columns:
            return None
        rows = frame.loc[frame["model_id"].astype(str).eq(model_id), metric]
        if rows.empty:
            return None
        value = pd.to_numeric(rows, errors="coerce").dropna()
        if value.empty:
            return None
        return float(value.iloc[0])

    def _fmt(value: float | None, *, pct: bool = False, money: bool = False) -> str:
        if value is None or not np.isfinite(value):
            return "n/a"
        if money:
            return f"${value:,.0f}"
        if pct:
            return f"{value:.1%}"
        return f"{value:.4f}"

    def _best(
        frame: pd.DataFrame,
        metric: str,
        *,
        higher_is_better: bool = True,
    ) -> tuple[str, float] | None:
        if frame.empty or "model_id" not in frame.columns or metric not in frame.columns:
            return None
        values = pd.to_numeric(frame[metric], errors="coerce")
        valid = frame.loc[values.notna(), ["model_id"]].copy()
        valid[metric] = values.loc[values.notna()]
        if valid.empty:
            return None
        idx = valid[metric].idxmax() if higher_is_better else valid[metric].idxmin()
        return str(valid.loc[idx, "model_id"]), float(valid.loc[idx, metric])

    def _append_bullets(lines: list[str], bullets: Sequence[str]) -> None:
        lines.extend([f"- {bullet}" for bullet in bullets if bullet])
        lines.append("")

    def _figure_path(name: str) -> str | None:
        path = figure_paths.get(name)
        if path is None:
            return None
        resolved = Path(path).resolve()
        try:
            return resolved.relative_to(report_path.parent.resolve()).as_posix()
        except ValueError:
            return str(resolved)

    def _figure_block(
        lines: list[str],
        *,
        name: str,
        title: str,
        bullets: Sequence[str],
        level: str = "####",
    ) -> None:
        path = _figure_path(name)
        if path is None:
            return
        lines.extend([f"{level} {title}", "", f"![{name}]({path})", ""])
        _append_bullets(lines, bullets)

    sequence_note = "Sequence diagnostics were unavailable."
    sequence_target = (
        predictions.loc[predictions["target_id"].astype(str).eq("jump_c2o")].copy()
        if "target_id" in predictions
        else predictions.copy()
    )
    if not sequence_target.empty and {
        "forecast_mamba_ssm_sequence",
        "forecast_mask_only_sequence",
        "rvar_event",
        "split",
    }.issubset(sequence_target.columns):
        sequence_frame = sequence_target.loc[
            sequence_target["split"].eq("test"),
            [
                "forecast_mamba_ssm_sequence",
                "forecast_mask_only_sequence",
                "rvar_event",
            ],
        ].dropna()
        if not sequence_frame.empty:
            corr = sequence_frame["forecast_mamba_ssm_sequence"].corr(
                sequence_frame["forecast_mask_only_sequence"]
            )
            mean_abs_diff = (
                (
                    sequence_frame["forecast_mamba_ssm_sequence"]
                    - sequence_frame["forecast_mask_only_sequence"]
                )
                .abs()
                .mean()
            )
            sequence_target_corr = sequence_frame["forecast_mamba_ssm_sequence"].corr(
                sequence_frame["rvar_event"]
            )
            mask_target_corr = sequence_frame["forecast_mask_only_sequence"].corr(
                sequence_frame["rvar_event"]
            )
            sequence_note = (
                "On common C2O test rows, official mamba-ssm sequence and mask-only "
                f"sequence forecasts have correlation {corr:.3f}; their mean absolute "
                f"forecast difference is {mean_abs_diff:.4f}. The mamba-ssm sequence "
                "forecast has correlation "
                f"{sequence_target_corr:.3f} with realized `jump_c2o` variance, versus "
                f"{mask_target_corr:.3f} for the mask-only ablation."
            )

    best_mae = _best(forecast_main, "mae", higher_is_better=False)
    best_oos = _best(forecast_main, "oos_r2_vs_ivar", higher_is_better=True)
    best_auc = _best(ranking_main, "auc", higher_is_better=True)
    best_top_decile = _best(ranking_main, "top_decile_precision", higher_is_better=True)
    best_edge_monotone = _best(ranking_main, "edge_decile_spearman", higher_is_better=True)
    best_net = _best(strategy_main, "net_pnl_usd", higher_is_better=True)
    best_return = _best(strategy_main, "return_on_premium", higher_is_better=True)
    best_c2o_diag_net = _best(c2o_strategy_primary, "net_pnl_usd", higher_is_better=True)
    best_o2c_diag_net = _best(o2c_strategy_primary, "net_pnl_usd", higher_is_better=True)
    best_o2c_auc = _best(o2c_ranking, "auc", higher_is_better=True)
    qlike_worst = _best(qlike, "raw_qlike", higher_is_better=True)
    qlike_share_worst = _best(
        qlike,
        "top_1pct_qlike_contribution_share",
        higher_is_better=True,
    )
    mamba_auc = _value(ranking_main, "mamba_ssm_sequence", "auc")
    mask_auc = _value(ranking_main, "mask_only_sequence", "auc")
    mamba_net = _value(strategy_main, "mamba_ssm_sequence", "net_pnl_usd")
    mask_net = _value(strategy_main, "mask_only_sequence", "net_pnl_usd")
    sequence_drop_rate = float(sequence_report.get("drop_rate", 0.0))
    if best_auc and best_top_decile:
        ranking_winner_text = (
            f"{_label(best_auc[0])} is the clearest ranking winner in this proxy run."
            if best_auc[0] == best_top_decile[0]
            else (
                f"{_label(best_auc[0])} leads AUC, while {_label(best_top_decile[0])} "
                "leads top-decile precision."
            )
        )
    else:
        ranking_winner_text = "Ranking comparison was unavailable."

    cost_main = (
        cost.loc[cost["target_id"].astype(str).eq("day_c2c")].copy()
        if "target_id" in cost
        else cost.copy()
    )
    if not cost_main.empty and {"cost_multiplier", "net_pnl_usd", "model_id"}.issubset(
        cost_main.columns
    ):
        cost_snapshot = cost_main.loc[
            pd.to_numeric(cost_main["cost_multiplier"], errors="coerce").isin([0.0, 1.0, 3.0, 5.0])
            & cost_main["model_id"]
            .astype(str)
            .isin(
                [
                    "lightgbm",
                    "lightgbm_tuned",
                    "xgboost",
                    "xgboost_tuned",
                    "linear_elastic_net",
                    "linear_elastic_net_tuned",
                    "mamba_ssm_sequence",
                    "mamba_ssm_sequence_5seed",
                ]
            )
        ].copy()
        cost_snapshot = cost_snapshot[
            ["model_id", "cost_multiplier", "n", "net_pnl_usd", "hit_rate", "max_drawdown_usd"]
        ].round(4)
    else:
        cost_snapshot = pd.DataFrame()

    c2o_strategy_table = pd.DataFrame()
    if not c2o_strategy_diag.empty:
        c2o_strategy_table = (
            c2o_strategy_diag.loc[c2o_strategy_diag["model_id"].isin(selected_models)]
            .copy()
            .sort_values(["strategy_proxy_kind", "model_id"])
        )
        keep = [
            "strategy_proxy_kind",
            "model_id",
            "n",
            "net_pnl_usd",
            "return_on_premium",
            "sharpe",
            "max_drawdown_usd",
        ]
        c2o_strategy_table = c2o_strategy_table[
            [column for column in keep if column in c2o_strategy_table.columns]
        ].round(4)
    o2c_summary_table = pd.DataFrame()
    if not o2c_forecast.empty:
        o2c_summary_table = o2c_forecast.merge(
            o2c_ranking, on=["target_id", "model_id"], how="left", suffixes=("", "_ranking")
        )
        o2c_summary_table = o2c_summary_table.loc[
            o2c_summary_table["model_id"].isin(selected_models)
        ].copy()
        o2c_summary_table["model_id"] = pd.Categorical(
            o2c_summary_table["model_id"], categories=selected_models, ordered=True
        )
        o2c_summary_table = o2c_summary_table.sort_values("model_id")
        keep = [
            "model_id",
            "n",
            "mae",
            "rmse",
            "oos_r2_vs_ivar",
            "top_decile_precision",
            "auc",
            "edge_decile_spearman",
        ]
        o2c_summary_table = o2c_summary_table[
            [column for column in keep if column in o2c_summary_table.columns]
        ].round(4)

    o2c_strategy_table = pd.DataFrame()
    if not o2c_strategy_diag.empty:
        o2c_strategy_table = (
            o2c_strategy_diag.loc[o2c_strategy_diag["model_id"].isin(selected_models)]
            .copy()
            .sort_values(["strategy_proxy_kind", "model_id"])
        )
        keep = [
            "strategy_proxy_kind",
            "model_id",
            "n",
            "net_pnl_usd",
            "return_on_premium",
            "sharpe",
            "max_drawdown_usd",
            "pnl_headline_eligible",
        ]
        o2c_strategy_table = o2c_strategy_table[
            [column for column in keep if column in o2c_strategy_table.columns]
        ].round(4)

    o2c_scale_table = (
        o2c_scale[
            [
                column
                for column in [
                    "paired_rows",
                    "sd_rvar_reaction_o2c",
                    "sd_ivar_event",
                    "sd_ratio_o2c_to_ivar",
                    "mean_ratio_o2c_to_ivar",
                    "ivar_baseline_interpretation",
                ]
                if column in o2c_scale.columns
            ]
        ].round(4)
        if not o2c_scale.empty
        else pd.DataFrame()
    )

    diagnostics_model_id = diagnostics.get("model_id", pd.Series(dtype=str)).astype(str)
    diagnostics_snapshot = diagnostics.loc[diagnostics_model_id.isin(selected_models)].copy()
    if not diagnostics_snapshot.empty:
        diagnostics_snapshot = diagnostics_snapshot[
            [
                column
                for column in [
                    "target_id",
                    "model_id",
                    "status",
                    "feature_count",
                    "train_rows",
                    "validation_rows",
                    "test_rows",
                    "epochs",
                    "hidden_sizes",
                    "device",
                    "mamba_backend",
                    "mamba_seeds",
                    "seed_count",
                    "seed_list",
                    "loss",
                    "mask_only",
                    "tuning_profile",
                    "selection_target_id",
                    "best_iteration",
                    "tuned_alpha",
                    "tuned_l1_ratio",
                    "dropout",
                ]
                if column in diagnostics_snapshot.columns
            ]
        ]

    lines = [
        "# Earnings Event Variance Mispricing",
        "",
        "## Intro",
        "",
        "**Scope.** This report is a proxy-stage study using the currently available sample "
        "from 2022 onward. It uses SEC filings for earnings-event identification and "
        "Massive second aggregates/day aggregates for market-data proxies. No quote, "
        "bid/ask, OPRA, or NBBO data are used.",
        "",
        "This is a proxy-stage report based on no_nbbo_trade_proxy data.",
        "Results are not paper-grade execution evidence.",
        f"Tuning profile: `{tuning_profile}`. Tuned rows, when present, use train/validation "
        "selection only; locked test metrics are excluded from tuning artifacts and are "
        "evaluated once after train+validation refit.",
        "The sequence diagnostics use pre-entry proxy-surface paths, including a "
        "31-step hybrid tensor with 19 daily steps plus 12 entry-day five-minute "
        "trade-aggregate proxy bins. They are not trained on NBBO-mid IV surfaces.",
        "",
        "### Research Question",
        "",
        "The question is whether models improve trading decisions around option-implied "
        "earnings event variance mispricing. The target system has three realized-variance "
        "labels: `jump_c2o` is the primary scientific target for close-to-open earnings "
        "jump variance; `day_c2c` is the literature-compatible target and the only V1 "
        "proxy-PnL headline; `reaction_o2c` is a diagnostic target for post-open digestion. "
        "The market baseline is implied event variance `IVAR_event`. C2C ex post "
        "mispricing is `RVAR_event_day_c2c - IVAR_event`; C2O is reported as "
        "forecast/ranking evidence plus post-open option-VWAP proxy diagnostics. Trading "
        "decisions are evaluated in premium space through proxy PnL and cost-aware edge, "
        "not by forecast error alone.",
        "The unified option open anchor is same-contract option VWAP from 5 to 15 "
        "minutes after the regular-session open. It is the primary C2O exit proxy "
        "and the O2C realized-decomposition entry proxy; O2C is not a V1 "
        "model-driven strategy headline because no post-open residual-IV baseline is "
        "estimated.",
        "",
        "## Materials: Data",
        "",
        (
            "- Earnings events: SEC EDGAR 8-K Item 2.02 discovery with "
            "SEC primary-document text validation."
        ),
        "- Universe: dynamic monthly top-50 liquid option underlyings within the available sample.",
        "- Timing: BMO and AMC only.",
        "- Entry proxy: Massive option second aggregates before the event cutoff.",
        (
            "- C2C exit proxy: same-contract option second aggregates over the "
            "final 15 minutes before the exit-date close; option day-aggregate "
            "close is fallback/diagnostic only."
        ),
        "- Daily sequence: 20 trading days of close-trade-implied option-surface summaries.",
        "- Hybrid sequence: 31 steps, with 19 prior daily proxy-surface states and 12 "
        "entry-day five-minute trade-aggregate proxy bins.",
        (
            f"- Sequence coverage: {sequence_report.get('eligible_events', 'NA')} "
            f"eligible events out of {sequence_report.get('total_events', 'NA')}; "
            f"drop rate {float(sequence_report.get('drop_rate', 0.0)):.1%}."
        ),
        "- Data coverage and selection-risk diagnostics are summarized in Appendix A.",
        "- Cost model: `cost_model=proxy_haircut`, "
        f"`haircut_bps={DEFAULT_HAIRCUT_BPS}`, `bid_ask_costs_unavailable=true`. "
        "Multiplier 0 is a sensitivity anchor, not a realistic execution-cost assumption.",
        "",
        "## Methods: Models and Configuration",
        "",
        (
            "- Split: chronological event-level `70/15/15`; all rows for the same "
            "`event_id` remain in one split."
        ),
        (
            "- Baselines: market-implied IVAR, last-four RVAR, last-four IVAR, and "
            "Goyal-Saretto-style RV-IV spread."
        ),
        (
            "- Tabular models: Elastic Net, LightGBM, XGBoost, a simple GBDT ensemble, "
            "and FT-Transformer use event-level features; GBDT models also receive "
            "sequence aggregates."
        ),
        (
            f"- Sequence diagnostics use a 31 x {len(HYBRID_SEQUENCE_FEATURE_NAMES)} "
            "mixed-clock hybrid tensor. Ridge-flat, BiGRU, official mamba-ssm, "
            "mask-only, and time-shuffle are the phase-1 diagnostic suite."
        ),
        (
            "- Sequence neural losses combine Huber loss on `log(RVAR_event)` with "
            "pairwise ranking loss on realized event-variance edge."
        ),
        "- Full fit status and hyperparameter diagnostics are in Appendix B.",
        "",
        "## Results Snapshot",
        "",
        "### Main Results",
        "",
        "#### Main Result Table",
        "",
        "Forecast and ranking columns use the primary `jump_c2o` target. Strategy and "
        "PnL columns use `day_c2c`, the only V1 proxy-PnL headline.",
        "",
        _markdown_table(summary, "_No summary metrics written._"),
        "",
    ]
    _append_bullets(
        lines,
        [
            (
                f"For `jump_c2o`, {_label(best_oos[0])} has the best OOS R2 versus IVAR "
                f"({_fmt(best_oos[1])}), while {_label(best_mae[0])} has the lowest MAE "
                f"({_fmt(best_mae[1])})."
                if best_oos and best_mae
                else "Forecast metrics were not available for model comparison."
            ),
            (
                f"For `jump_c2o`, {_label(best_auc[0])} leads ranking quality with AUC "
                f"{_fmt(best_auc[1])}; "
                f"{_label(best_top_decile[0])} has top-decile precision "
                f"{_fmt(best_top_decile[1], pct=True)}."
                if best_auc and best_top_decile
                else "Ranking metrics were not available for model comparison."
            ),
            (
                f"For `day_c2c`, {_label(best_net[0])} has the strongest proxy net PnL "
                f"({_fmt(best_net[1], money=True)}), and {_label(best_return[0])} has the best "
                f"return on premium ({_fmt(best_return[1], pct=True)})."
                if best_net and best_return
                else "Strategy metrics were not available for model comparison."
            ),
            (
                "The market IVAR baseline remains the level benchmark, but it does not create "
                "trades under the zero-edge premium rule."
            ),
        ],
    )
    _figure_block(
        lines,
        name="forecast_performance",
        title="Forecast Performance",
        bullets=[
            (
                f"For `jump_c2o`, best variance-level forecast improvement is "
                f"{_label(best_oos[0])} "
                f"with OOS R2 {_fmt(best_oos[1])} versus IVAR."
                if best_oos
                else "Forecast comparison was unavailable."
            ),
            (
                f"For `jump_c2o`, best absolute-error model is {_label(best_mae[0])} with MAE "
                f"{_fmt(best_mae[1])}; this is separate from ranking quality."
                if best_mae
                else "MAE comparison was unavailable."
            ),
            (
                "QLIKE is not used as the headline forecast metric because near-zero forecasts "
                "dominate some raw values; see Appendix C."
            ),
        ],
    )
    _figure_block(
        lines,
        name="auc_top_decile_precision",
        title="Ranking and Top-Decile Precision",
        bullets=[
            ranking_winner_text,
            (
                "This is the main sellable `jump_c2o` result: the task is not only level "
                "forecasting, but sorting event-jump variance opportunities."
            ),
            (
                "The deterministic IVAR baseline is intentionally neutral in ranking because "
                "its forecast edge is zero by construction."
            ),
        ],
    )
    _figure_block(
        lines,
        name="edge_decile_realized_mispricing",
        title="Edge-Decile Realized Mispricing",
        bullets=[
            (
                f"{_label(best_edge_monotone[0])} has the strongest edge-decile monotonicity "
                f"(Spearman {_fmt(best_edge_monotone[1])})."
                if best_edge_monotone
                else "Edge-decile monotonicity was unavailable."
            ),
            (
                "For `jump_c2o`, a useful model should concentrate positive realized "
                "jump-variance mispricing in high predicted-edge deciles."
            ),
            "This figure is closer to the paper's economic question than MAE alone.",
        ],
    )
    _figure_block(
        lines,
        name="strategy_pnl_by_edge_decile",
        title="Strategy PnL by Edge Decile",
        bullets=[
            (
                f"For `day_c2c`, {_label(best_net[0])} produces the best net proxy PnL "
                f"in the current test sample ({_fmt(best_net[1], money=True)})."
                if best_net
                else "Strategy PnL comparison was unavailable."
            ),
            (
                "The strategy layer evaluates premium-space C2C outcomes and proxy costs, "
                "which is the intended V1 economic target for selling the project."
            ),
            (
                "O2C option PnL uses the same 5-15 minute open anchor as a realized "
                "decomposition diagnostic, but it is not converted into an IVAR-based "
                "strategy in V1."
            ),
            "These are still no-NBBO proxy economics, not paper-grade execution evidence.",
        ],
    )
    lines.extend(
        [
            "### C2O Post-Open Proxy PnL",
            "",
            (
                "The primary C2O option proxy uses same-contract option VWAP from "
                "5 to 15 minutes after the regular-session open. The 0 to 5 minute "
                "VWAP is an opening-microstructure stress test, and the intrinsic-open "
                "mark `abs(open_after - strike) * 100` remains a jump diagnostic only. "
                "The same 5 to 15 minute option VWAP is also the O2C diagnostic entry "
                "anchor for realized decomposition to the primary C2C exit mark. All "
                "marks are no-NBBO trade-aggregate proxies, not executable NBBO PnL."
            ),
            "",
            _markdown_table(
                c2o_strategy_table,
                "C2O post-open proxy PnL was unavailable.",
            ),
            "",
            (
                f"Best C2O 5-15 minute option-VWAP proxy net PnL is "
                f"{_label(best_c2o_diag_net[0])} at {_fmt(best_c2o_diag_net[1], money=True)}."
                if best_c2o_diag_net
                else "C2O 5-15 minute option-VWAP proxy strategy metrics were unavailable."
            ),
            (
                "Interpret the 5-15 minute mark as the V1 C2O trade-aggregate comparison "
                "against C2C; the intrinsic mark is not an option-price exit."
            ),
            "",
        ]
    )
    lines.extend(
        [
            "### O2C Post-Open Diagnostic Proxy PnL",
            "",
            (
                "`reaction_o2c` is now modeled in Phase 1 as a diagnostic target. "
                "Its realized variance is post-open only, while `IVAR_event` is a "
                "full-event implied-variance comparator; this makes O2C evidence "
                "suitable for ranking and directional diagnostics, not level-calibrated "
                "mispricing claims."
            ),
            "",
            "#### O2C Forecast and Ranking",
            "",
            _markdown_table(o2c_summary_table, "O2C forecast/ranking metrics were unavailable."),
            "",
            "#### O2C Premium-Space Diagnostic Strategy",
            "",
            _markdown_table(
                o2c_strategy_table,
                "O2C premium-space diagnostic strategy metrics were unavailable.",
            ),
            "",
            "#### O2C Scale Diagnostic",
            "",
            _markdown_table(o2c_scale_table, "O2C scale diagnostics were unavailable."),
            "",
        ]
    )
    _figure_block(
        lines,
        name="o2c_forecast_performance",
        title="O2C Forecast Performance",
        bullets=[
            (
                "This figure repeats the forecast comparison for `reaction_o2c`, the "
                "post-open digestion target."
            ),
            (
                "O2C level-fit metrics are diagnostic because full-event `IVAR_event` is "
                "only a weak comparator for post-open realized variance."
            ),
        ],
    )
    _figure_block(
        lines,
        name="o2c_auc_top_decile_precision",
        title="O2C Ranking and Top-Decile Precision",
        bullets=[
            (
                f"For `reaction_o2c`, best ranking AUC is {_label(best_o2c_auc[0])} at "
                f"{_fmt(best_o2c_auc[1])}."
                if best_o2c_auc
                else "O2C ranking metrics were unavailable."
            ),
            "Treat O2C ranking as a third-window diagnostic, not as a headline claim.",
        ],
    )
    _figure_block(
        lines,
        name="o2c_strategy_proxy_pnl",
        title="O2C Diagnostic Proxy PnL",
        bullets=[
            (
                f"Best O2C 5-15 minute diagnostic net PnL is "
                f"{_label(best_o2c_diag_net[0])} at {_fmt(best_o2c_diag_net[1], money=True)}."
                if best_o2c_diag_net
                else "O2C 5-15 minute diagnostic proxy metrics were unavailable."
            ),
            (
                "All O2C strategy rows use post-open option VWAP anchors and remain "
                "`pnl_headline_eligible=false`."
            ),
        ],
    )
    _figure_block(
        lines,
        name="o2c_scale_diagnostic",
        title="O2C Scale Diagnostic",
        bullets=[
            (
                "The scale figure compares post-open realized variance against full-event "
                "`IVAR_event` and visualizes why IVAR is a weak O2C comparator."
            ),
            "Use O2C evidence for ranking/direction diagnostics, not calibrated mispricing.",
        ],
    )
    lines.extend(
        [
            (
                f"Best O2C 5-15 minute diagnostic net PnL is "
                f"{_label(best_o2c_diag_net[0])} at {_fmt(best_o2c_diag_net[1], money=True)}."
                if best_o2c_diag_net
                else "O2C 5-15 minute diagnostic proxy metrics were unavailable."
            ),
            (
                f"Best O2C ranking AUC is {_label(best_o2c_auc[0])} at {_fmt(best_o2c_auc[1])}."
                if best_o2c_auc
                else "O2C ranking metrics were unavailable."
            ),
            (
                "All O2C strategy rows are `pnl_headline_eligible=false` and remain "
                "no-NBBO trade-aggregate proxy diagnostics."
            ),
            "",
        ]
    )
    lines.extend(["### Other Results and Diagnostics", ""])
    _figure_block(
        lines,
        name="calibration_plot",
        title="Calibration",
        bullets=[
            (
                "Calibration checks whether `jump_c2o` forecasted event-jump variance is on "
                "the same scale as realized jump variance, not just correctly ranked."
            ),
            (
                "The current proxy evidence is stronger for ranking than for perfectly "
                "calibrated variance levels."
            ),
            (
                "This plot is a guardrail against over-selling high AUC models as fully "
                "calibrated probability or variance forecasters."
            ),
        ],
    )
    _figure_block(
        lines,
        name="cost_sensitivity",
        title="Cost Sensitivity",
        bullets=[
            (
                "The default cost model is a proxy haircut: "
                "`proxy_cost_usd = 0.005 * entry_premium_usd`."
            ),
            (
                "Multiplier 0 is shown only as an anchor; multiplier 1 is the default proxy "
                "cost assumption."
            ),
            (
                "Persistence across higher multipliers is the relevant robustness check "
                "because true bid/ask costs are unavailable."
            ),
        ],
    )
    lines.extend(
        [
            "##### Cost Sensitivity Snapshot",
            "",
            _markdown_table(cost_snapshot, "_No cost sensitivity table written._"),
            "",
        ]
    )
    _append_bullets(
        lines,
        [
            (
                "The snapshot keeps the main tabular contenders and official mamba-ssm at "
                "multipliers 0, 1, 3, and 5."
            ),
            (
                "Use this as a stress-test table rather than an execution-cost estimate; "
                "bid/ask costs are unavailable in the current route."
            ),
        ],
    )
    _figure_block(
        lines,
        name="qlike_contribution_diagnostic",
        title="QLIKE Contribution Diagnostic",
        bullets=[
            (
                f"Raw QLIKE is most distorted for {_label(qlike_worst[0])}; the worst top-1% "
                f"contribution share is {_fmt(qlike_share_worst[1], pct=True)} for "
                f"{_label(qlike_share_worst[0])}."
                if qlike_worst and qlike_share_worst
                else "QLIKE diagnostics were unavailable."
            ),
            (
                "Raw QLIKE is a diagnostic, not the headline result, when forecasts are "
                "clipped near zero."
            ),
            "The full raw/floored/winsorized table is in Appendix C.",
        ],
    )
    lines.extend(["### Sequence Diagnostic Result", "", sequence_note, ""])
    _append_bullets(
        lines,
        [
            (
                f"Official mamba-ssm `jump_c2o` AUC is {_fmt(mamba_auc)} versus mask-only "
                f"AUC {_fmt(mask_auc)}; `day_c2c` net PnL is {_fmt(mamba_net, money=True)} versus "
                f"{_fmt(mask_net, money=True)}."
            ),
            "Sequence models are diagnostic-grade unless they pass common-row, bootstrap, "
            "simple-sequence-baseline, and premium-space economics gates.",
            (
                "The sequence result should be read as diagnostic because the sequence coverage "
                "drop rate is above 10%."
            ),
        ],
    )
    lines.extend(
        [
            "## Interpretation",
            "",
            (
                "The current proxy evidence supports a simple interpretation: tabular nonlinear "
                "models are useful for sorting earnings events by realized mispricing and proxy "
                "economic edge. The result is economically meaningful in this proxy setting "
                "because the strongest models also improve the cost-aware strategy layer. "
                "The sequence suite is a diagnostic test of whether ordered pre-event "
                "proxy-surface paths add information beyond tabular aggregates."
            ),
            "",
            "## Appendix",
            "",
            "### Appendix A: Data Coverage and Selection Risk",
            "",
            (
                f"- Sequence coverage: {sequence_report.get('eligible_events', 'NA')} eligible "
                f"events out of {sequence_report.get('total_events', 'NA')} total events."
            ),
            f"- Default sequence eligibility drop rate: {sequence_drop_rate:.1%}.",
            f"- Threshold sensitivity: `{sequence_report.get('threshold_sensitivity', {})}`.",
            (
                "- `high_sequence_selection_risk="
                f"{sequence_report.get('high_sequence_selection_risk', 'NA')}`."
            ),
            f"- `vix_regime_unavailable={sequence_report.get('vix_regime_unavailable', 'NA')}`.",
            "",
            "### Appendix B: Model Configuration and Fit Diagnostics",
            "",
            _markdown_table(diagnostics_snapshot, "_No diagnostics written._"),
            "",
            "### Appendix C: QLIKE Sanity Check",
            "",
            (
                "QLIKE is reported, but it is not used as a headline metric in this proxy run. "
                "The measure is highly sensitive to near-zero forecasts. This is visible in the "
                "Goyal-Saretto-style spread baseline, where raw QLIKE is dominated by "
                "zero-clipped forecasts."
            ),
            "",
            _markdown_table(qlike.round(4), "_No QLIKE diagnostics written._"),
            "",
            "### Appendix D: Limits and Next Steps",
            "",
            (
                "- The sample starts in 2022 because older options day-aggregate coverage is "
                "not available under the current data entitlement."
            ),
            "- There are no quote or NBBO data. The results are not execution-grade.",
            (
                "- The sequence sample has selection risk because 16.3% of events fail the "
                "V1 sequence coverage rule."
            ),
            "- VIX regime features are unavailable in the current run.",
            "- The report is suitable for internal research discussion, not final paper claims.",
            "",
            "Next steps:",
            "",
            "1. Keep LightGBM and XGBoost as the main proxy-stage models.",
            (
                "2. Treat sequence/Mamba-SSM as a diagnostic experiment until sequence "
                "coverage and surface quality improve."
            ),
            (
                "3. Run a robustness pass focused on liquidity, DTE, BMO/AMC, and "
                "ticker concentration."
            ),
            (
                "4. If paper-grade execution evidence is required, add licensed quote/NBBO "
                "data rather than extending claims from this proxy route."
            ),
            "",
        ]
    )
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
    hybrid_long, hybrid_by_event, hybrid_report = build_hybrid_proxy_sequence_long(
        long_rows,
        base,
        config=config,
    )
    by_event_path = paths.modeling_artifacts_dir / "sequence_coverage_by_event.csv"
    report_path = paths.modeling_artifacts_dir / "sequence_coverage_report.json"
    hybrid_by_event_path = paths.modeling_artifacts_dir / "hybrid_sequence_coverage_by_event.csv"
    hybrid_report_path = paths.modeling_artifacts_dir / "hybrid_sequence_coverage_report.json"
    distribution_audit_path = paths.modeling_artifacts_dir / "proxy_surface_distribution_audit.csv"
    by_event.to_csv(by_event_path, index=False)
    hybrid_by_event.to_csv(hybrid_by_event_path, index=False)
    proxy_surface_distribution_audit(hybrid_long).to_csv(distribution_audit_path, index=False)
    retired_path = write_retired_model_manifest(paths.modeling_artifacts_dir)
    write_json(report_path, report)
    write_json(hybrid_report_path, hybrid_report)
    return ProxyResearchResult(
        ok=True,
        stage="sequence-audit",
        outputs={
            "sequence_coverage_by_event": str(by_event_path),
            "sequence_coverage_report": str(report_path),
            "hybrid_sequence_coverage_by_event": str(hybrid_by_event_path),
            "hybrid_sequence_coverage_report": str(hybrid_report_path),
            "proxy_surface_distribution_audit": str(distribution_audit_path),
            "retired_model_ids": str(retired_path),
        },
        diagnostics={**report, "hybrid": hybrid_report},
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
    hybrid_long, hybrid_by_event, hybrid_report = build_hybrid_proxy_sequence_long(
        long_rows,
        base,
        config=config,
    )
    aggregates = aggregate_sequence_features(long_rows)
    features = enrich_feature_matrix_for_research(
        base,
        sequence_by_event=by_event,
        hybrid_by_event=hybrid_by_event,
        sequence_aggregates=aggregates,
        split_design=split_design,
        split_date=split_date,
    )
    features["hybrid_sequence_too_sparse"] = bool(hybrid_report.get("hybrid_sequence_too_sparse"))
    market_covariates = _read_market_covariates(config)
    if not market_covariates.empty and "entry_date" in features.columns:
        vix_input_columns = ["event_id", "entry_date"]
        if "announcement_timing" in features.columns:
            vix_input_columns.append("announcement_timing")
        vix_input = features[vix_input_columns].rename(columns={"entry_date": "feature_asof_date"})
        vix_features = build_vix_features(
            market_covariates,
            vix_input,
            alignment=VIX_ALIGNMENT_PRIOR_CLOSE,
        )
        merge_columns = ["event_id", *_vix_columns_for_merge()]
        features = features.drop(
            columns=[column for column in _vix_columns_for_merge() if column in features.columns]
        ).merge(vix_features[merge_columns], on="event_id", how="left")
    market_second = _read_market_second_covariates(config)
    market_second_columns = 0
    if not market_second.empty and "event_id" in market_second.columns:
        keep = [
            column
            for column in market_second.columns
            if column == "event_id"
            or column.startswith("spy_second_")
            or column.startswith("qqq_second_")
            or column.startswith("market_second_")
        ]
        market_second_columns = len(keep) - 1
        features = features.drop(
            columns=[column for column in keep if column != "event_id" and column in features]
        ).merge(market_second[keep].drop_duplicates("event_id"), on="event_id", how="left")
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
    hybrid_long.to_parquet(paths.hybrid_sequence_long_path, index=False)
    features.to_parquet(paths.feature_matrix_path, index=False)
    tensor_report = build_sequence_tensor(long_rows, features, out_path=paths.sequence_tensor_path)
    hybrid_tensor_report = build_sequence_tensor(
        hybrid_long,
        features,
        out_path=paths.hybrid_sequence_tensor_path,
        feature_names=HYBRID_SEQUENCE_FEATURE_NAMES,
        lookback_days=HYBRID_STEPS,
        per_step_type_scaling=True,
    )
    hybrid_tensor_v2_report = build_sequence_tensor(
        hybrid_long,
        features,
        out_path=paths.hybrid_sequence_tensor_v2_path,
        feature_names=HYBRID_SEQUENCE_FEATURE_NAMES,
        lookback_days=HYBRID_STEPS,
        per_step_type_scaling=True,
    )
    sequence_quality = build_sequence_v2_quality(
        features,
        tensor_path=paths.hybrid_sequence_tensor_v2_path,
    )
    sequence_quality_path = paths.modeling_artifacts_dir / "sequence_v2_quality.csv"
    sequence_quality.to_csv(sequence_quality_path, index=False)
    by_event.to_csv(paths.modeling_artifacts_dir / "sequence_coverage_by_event.csv", index=False)
    hybrid_by_event.to_csv(
        paths.modeling_artifacts_dir / "hybrid_sequence_coverage_by_event.csv", index=False
    )
    proxy_surface_distribution_audit(hybrid_long).to_csv(
        paths.modeling_artifacts_dir / "proxy_surface_distribution_audit.csv", index=False
    )
    retired_path = write_retired_model_manifest(paths.modeling_artifacts_dir)
    write_json(paths.modeling_artifacts_dir / "sequence_coverage_report.json", report)
    write_json(paths.modeling_artifacts_dir / "hybrid_sequence_coverage_report.json", hybrid_report)
    return ProxyResearchResult(
        ok=True,
        stage="features",
        outputs={
            "sequence_long": str(paths.sequence_long_path),
            "sequence_tensor": str(paths.sequence_tensor_path),
            "hybrid_sequence_long": str(paths.hybrid_sequence_long_path),
            "hybrid_sequence_tensor": str(paths.hybrid_sequence_tensor_path),
            "hybrid_sequence_tensor_v2": str(paths.hybrid_sequence_tensor_v2_path),
            "feature_matrix": str(paths.feature_matrix_path),
            "sequence_v2_quality": str(sequence_quality_path),
            "sequence_coverage_by_event": str(
                paths.modeling_artifacts_dir / "sequence_coverage_by_event.csv"
            ),
            "sequence_coverage_report": str(
                paths.modeling_artifacts_dir / "sequence_coverage_report.json"
            ),
            "hybrid_sequence_coverage_by_event": str(
                paths.modeling_artifacts_dir / "hybrid_sequence_coverage_by_event.csv"
            ),
            "hybrid_sequence_coverage_report": str(
                paths.modeling_artifacts_dir / "hybrid_sequence_coverage_report.json"
            ),
            "proxy_surface_distribution_audit": str(
                paths.modeling_artifacts_dir / "proxy_surface_distribution_audit.csv"
            ),
            "retired_model_ids": str(retired_path),
        },
        diagnostics={
            **report,
            "hybrid": hybrid_report,
            "tensor": tensor_report,
            "hybrid_tensor": hybrid_tensor_report,
            "hybrid_tensor_v2": hybrid_tensor_v2_report,
            "feature_rows": int(len(features)),
            "market_second_covariate_columns": int(market_second_columns),
        },
    )


def _model_ids_for_sequence_suite(
    sequence_suite: str,
    *,
    tuning_profile: TuningProfile = "untuned",
) -> list[str]:
    if sequence_suite == "none":
        model_ids = [model_id for model_id in MODEL_IDS if model_id not in SEQUENCE_MODEL_IDS]
    elif sequence_suite == "phase1":
        model_ids = MODEL_IDS.copy()
    elif sequence_suite == "phase2":
        model_ids = [*MODEL_IDS, *PHASE2_SEQUENCE_MODEL_IDS]
    else:
        raise ValueError(f"unsupported sequence_suite: {sequence_suite}")
    if tuning_profile == "tuned_phase1":
        model_ids = [*model_ids, *TUNED_TABULAR_MODEL_IDS]
        if sequence_suite != "none":
            model_ids = [*model_ids, *SEQUENCE_ENSEMBLE_MODEL_IDS]
    return model_ids


def _target_ids_for_sequence_suite(sequence_suite: str) -> list[str]:
    return PHASE2_TARGET_IDS if sequence_suite == "phase2" else PHASE1_TARGET_IDS


def _write_tuning_artifacts(
    out_dir: Path,
    *,
    tuning_state: TuningState,
) -> dict[str, str]:
    trials_path = out_dir / "tuning_trials.csv"
    selected_path = out_dir / "tuning_selected_params.json"
    trial_columns = [
        "model_id",
        "target_id",
        "trial_number",
        "selected",
        "seed",
        "params_json",
        "validation_n",
        "validation_mae",
        "validation_rmse",
        "validation_auc",
        "validation_top_decile_precision",
        "objective_value",
    ]
    pd.DataFrame(tuning_state.trials, columns=trial_columns).to_csv(trials_path, index=False)
    write_json(
        selected_path,
        {
            "tuning_profile": tuning_state.profile,
            "tuning_seed": tuning_state.seed,
            "selection_target_id": TUNING_SELECTION_TARGET_ID,
            "test_metrics_used_for_selection": False,
            "selected_params": tuning_state.selected,
        },
    )
    return {
        "tuning_trials": str(trials_path),
        "tuning_selected_params": str(selected_path),
    }


def run_research_models(
    config: ProjectConfig,
    *,
    sequence_suite: str = "phase1",
    mamba_backend: str = "mamba_ssm",
    mamba_seeds: Sequence[int] = (17,),
    bootstrap_iter: int = 200,
    tuning_profile: TuningProfile = "untuned",
    tuning_seed: int = 17,
) -> ProxyResearchResult:
    paths = research_paths(config)
    paths.modeling_artifacts_dir.mkdir(parents=True, exist_ok=True)
    features = read_table(paths.feature_matrix_path)
    prediction_frames: list[pd.DataFrame] = []
    diagnostic_frames: list[pd.DataFrame] = []
    available_targets = available_target_columns(features)
    if "day_c2c" not in available_targets and "rvar_event" in features:
        available_targets["day_c2c"] = "rvar_event"
    if tuning_profile not in TUNING_PROFILES:
        raise ValueError(f"unsupported tuning_profile: {tuning_profile}")
    model_ids = _model_ids_for_sequence_suite(sequence_suite, tuning_profile=tuning_profile)
    tuning_state = TuningState(profile=tuning_profile, seed=tuning_seed)
    for target_id in _target_ids_for_sequence_suite(sequence_suite):
        if target_id not in available_targets:
            continue
        target_frame = prepare_target_frame(features, target_id=target_id)
        predictions_one, diagnostics_one = run_proxy_model_suite(
            target_frame,
            tensor_path=paths.sequence_tensor_path,
            hybrid_tensor_path=paths.hybrid_sequence_tensor_v2_path,
            model_ids=model_ids,
            mamba_backend=mamba_backend,
            mamba_seeds=mamba_seeds,
            tuning_state=tuning_state,
            target_id=target_id,
        )
        predictions_one["target_id"] = target_id
        diagnostics_one["target_id"] = target_id
        prediction_frames.append(predictions_one)
        diagnostic_frames.append(diagnostics_one)
    if not prediction_frames:
        raise ValueError("no available target columns for research modeling")
    predictions = pd.concat(prediction_frames, ignore_index=True)
    diagnostics = pd.concat(diagnostic_frames, ignore_index=True)
    predictions = append_day_c2c_additive_naive_diagnostics(predictions)
    predictions.to_parquet(paths.predictions_path, index=False)
    diagnostics_path = paths.modeling_artifacts_dir / "model_fit_diagnostics.csv"
    diagnostics.to_csv(diagnostics_path, index=False)
    outputs = {
        "model_predictions": str(paths.predictions_path),
        "model_fit_diagnostics": str(diagnostics_path),
        "retired_model_ids": str(write_retired_model_manifest(paths.modeling_artifacts_dir)),
    }
    outputs.update(_write_tuning_artifacts(paths.modeling_artifacts_dir, tuning_state=tuning_state))
    outputs.update(build_metric_tables(predictions, out_dir=paths.modeling_artifacts_dir))
    outputs.update(
        build_common_row_diagnostics(
            predictions,
            out_dir=paths.modeling_artifacts_dir,
            bootstrap_iter=bootstrap_iter,
        )
    )
    return ProxyResearchResult(
        ok=True,
        stage="models",
        outputs=outputs,
        diagnostics={
            "prediction_rows": int(len(predictions)),
            "trained_models": int(diagnostics["status"].eq("trained").sum())
            if "status" in diagnostics
            else 0,
            "sequence_suite": sequence_suite,
            "mamba_backend": mamba_backend,
            "mamba_seeds": ",".join(str(seed) for seed in mamba_seeds),
            "bootstrap_iter": bootstrap_iter,
            "tuning_profile": tuning_profile,
            "tuning_seed": tuning_seed,
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
    sequence_suite: str = "phase1",
    mamba_backend: str = "mamba_ssm",
    mamba_seeds: Sequence[int] = (17,),
    bootstrap_iter: int = 200,
    tuning_profile: TuningProfile = "untuned",
    tuning_seed: int = 17,
) -> dict[str, object]:
    if stage not in {"all", "sequence-audit", "features", "models", "report"}:
        raise ValueError(f"unsupported research stage: {stage}")
    if sequence_suite not in {"none", "phase1", "phase2"}:
        raise ValueError(f"unsupported sequence_suite: {sequence_suite}")
    if mamba_backend != "mamba_ssm":
        raise ValueError(f"unsupported mamba_backend: {mamba_backend}")
    if not mamba_seeds:
        raise ValueError("mamba_seeds must include at least one seed")
    if tuning_profile not in TUNING_PROFILES:
        raise ValueError(f"unsupported tuning_profile: {tuning_profile}")
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
            result = run_research_models(
                config,
                sequence_suite=sequence_suite,
                mamba_backend=mamba_backend,
                mamba_seeds=mamba_seeds,
                bootstrap_iter=bootstrap_iter,
                tuning_profile=tuning_profile,
                tuning_seed=tuning_seed,
            )
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
        "sequence_suite": sequence_suite,
        "mamba_backend": mamba_backend,
        "mamba_seeds": ",".join(str(seed) for seed in mamba_seeds),
        "bootstrap_iter": bootstrap_iter,
        "tuning_profile": tuning_profile,
        "tuning_seed": tuning_seed,
        "steps": steps,
    }
    paths = research_paths(config)
    write_json(paths.modeling_artifacts_dir / "research_manifest.json", payload)
    return payload

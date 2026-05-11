from __future__ import annotations

import importlib
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any, Protocol, cast

import numpy as np
import pandas as pd
import torch
from torch import nn

TARGET_COL = "rvar_event"
MARKET_BASELINE_COL = "ivar_event"
PREDICTION_PREFIX = "forecast_"
DEFAULT_FEATURE_EXCLUDE_PATTERNS = (
    "rvar_event",
    "rvar_",
    "r_event_",
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
    "realized",
    "post_event",
    "future",
)


class TabularPredictor(Protocol):
    def predict(self, frame: pd.DataFrame) -> np.ndarray: ...


@dataclass(frozen=True)
class ModelSpec:
    model_id: str
    role: str
    implemented: bool
    justification: str
    risk: str


@dataclass
class ModelFitResult:
    model_id: str
    predictions: pd.DataFrame
    feature_columns: list[str]
    diagnostics: dict[str, object]
    model: object | None = None


MODEL_REGISTRY: dict[str, ModelSpec] = {
    "market_implied_event_variance": ModelSpec(
        model_id="market_implied_event_variance",
        role="baseline",
        implemented=True,
        justification="Uses IVAR_event as the market forecast to beat.",
        risk="Can dominate if earnings option prices are already efficient after costs.",
    ),
    "last_four_rvar": ModelSpec(
        model_id="last_four_rvar",
        role="baseline",
        implemented=True,
        justification="Same-ticker average realized variance over the prior four earnings events.",
        risk="No cross-sectional information and weak for regime changes.",
    ),
    "last_four_ivar": ModelSpec(
        model_id="last_four_ivar",
        role="baseline",
        implemented=True,
        justification=(
            "Same-ticker average implied event variance over the prior four earnings events."
        ),
        risk="Can inherit historical option-market bias.",
    ),
    "patell_wolfson_diagnostic": ModelSpec(
        model_id="patell_wolfson_diagnostic",
        role="diagnostic",
        implemented=True,
        justification=(
            "Patell-Wolfson-style diagnostic features based on pre-event implied-volatility "
            "behavior and prior event variance history."
        ),
        risk="Diagnostic feature set, not a standalone modern ML model.",
    ),
    "goyal_saretto_rv_iv_spread": ModelSpec(
        model_id="goyal_saretto_rv_iv_spread",
        role="feature_baseline",
        implemented=True,
        justification="Trailing RV-IV spread adjustment to the market-implied event variance.",
        risk="Useful benchmark, not a full replication of the original portfolio design.",
    ),
    "linear_elastic_net": ModelSpec(
        model_id="linear_elastic_net",
        role="model",
        implemented=True,
        justification="Transparent semi-structural tabular benchmark.",
        risk="Limited nonlinear interaction capacity.",
    ),
    "linear_elastic_net_tuned": ModelSpec(
        model_id="linear_elastic_net_tuned",
        role="model_tuned",
        implemented=True,
        justification="Sklearn ElasticNetCV tuned on train/validation in tuned_phase1.",
        risk=(
            "Available through the research tuned_phase1 protocol, not the legacy train-models CLI."
        ),
    ),
    "lightgbm": ModelSpec(
        model_id="lightgbm",
        role="model",
        implemented=True,
        justification="Strong tabular ML baseline before deep models.",
        risk="Depends on optional LightGBM package availability.",
    ),
    "lightgbm_tuned": ModelSpec(
        model_id="lightgbm_tuned",
        role="model_tuned",
        implemented=True,
        justification="LightGBM with validation-only Optuna tuning and early stopping.",
        risk="Depends on optional LightGBM, scikit-learn, and Optuna package availability.",
    ),
    "xgboost": ModelSpec(
        model_id="xgboost",
        role="model",
        implemented=True,
        justification="Strong tree-boosting benchmark paired with LightGBM.",
        risk="Depends on optional XGBoost package availability.",
    ),
    "xgboost_tuned": ModelSpec(
        model_id="xgboost_tuned",
        role="model_tuned",
        implemented=True,
        justification="XGBoost with validation-only Optuna tuning and early stopping.",
        risk="Depends on optional XGBoost, scikit-learn, and Optuna package availability.",
    ),
    "ft_transformer": ModelSpec(
        model_id="ft_transformer",
        role="deep_model",
        implemented=True,
        justification="Deep tabular architecture for mixed event features.",
        risk="May not beat GBDT on small tabular panels.",
    ),
    "ft_transformer_tuned": ModelSpec(
        model_id="ft_transformer_tuned",
        role="deep_model_tuned",
        implemented=True,
        justification="FT-Transformer architecture/dropout/lr tuning in tuned_phase1.",
        risk="May overfit small event panels despite validation-only selection.",
    ),
    "lightgbm_xgboost_mean_ensemble": ModelSpec(
        model_id="lightgbm_xgboost_mean_ensemble",
        role="ensemble",
        implemented=True,
        justification="Equal-weight LightGBM/XGBoost ensemble sanity check.",
        risk="Should not be tuned on the locked test sample.",
    ),
    "ridge_flat_aggregates_sequence": ModelSpec(
        model_id="ridge_flat_aggregates_sequence",
        role="sequence_baseline",
        implemented=True,
        justification=(
            "Flattens daily and intraday sequence channels into simple aggregates before "
            "a ridge-style linear fit."
        ),
        risk="Tests whether ordering adds value beyond coarse path summaries.",
    ),
    "attention_pooling_sequence": ModelSpec(
        model_id="attention_pooling_sequence",
        role="sequence_baseline",
        implemented=True,
        justification="Learned-query attention pooling over pre-entry sequence tokens.",
        risk="Diagnostic; may overfit on small sequence samples.",
    ),
    "bigru_sequence": ModelSpec(
        model_id="bigru_sequence",
        role="sequence_baseline",
        implemented=True,
        justification="LayerNorm BiGRU baseline for ordered pre-entry proxy-surface paths.",
        risk="Diagnostic; sequence sample has selection risk.",
    ),
    "bigru_sequence_5seed": ModelSpec(
        model_id="bigru_sequence_5seed",
        role="sequence_baseline_tuned",
        implemented=True,
        justification="Five-seed BiGRU ensemble diagnostic for tuned_phase1.",
        risk="Diagnostic only unless it clears mask-only/time-shuffle/common-row gates.",
    ),
    "dilated_cnn_sequence": ModelSpec(
        model_id="dilated_cnn_sequence",
        role="sequence_baseline",
        implemented=True,
        justification="Non-causal dilated 1D CNN baseline for short pre-entry paths.",
        risk="Diagnostic; non-causal only because all tokens are pre-entry.",
    ),
    "mamba_ssm_sequence": ModelSpec(
        model_id="mamba_ssm_sequence",
        role="sequence_challenger",
        implemented=True,
        justification=(
            "Official mamba-ssm selective state-space block used as a diagnostic challenger."
        ),
        risk="Requires Linux/NVIDIA/CUDA; skipped when mamba-ssm is unavailable.",
    ),
    "mamba_ssm_sequence_5seed": ModelSpec(
        model_id="mamba_ssm_sequence_5seed",
        role="sequence_challenger_tuned",
        implemented=True,
        justification="Five-seed official mamba-ssm ensemble diagnostic for tuned_phase1.",
        risk="Requires Linux/NVIDIA/CUDA; diagnostic only unless it clears sequence controls.",
    ),
    "mask_only_sequence": ModelSpec(
        model_id="mask_only_sequence",
        role="sequence_control",
        implemented=True,
        justification="Sequence control with values zeroed and masks retained.",
        risk="High performance indicates missingness/selection signal rather than path values.",
    ),
    "time_shuffle_sequence": ModelSpec(
        model_id="time_shuffle_sequence",
        role="sequence_control",
        implemented=True,
        justification="Deterministic within-event time-order ablation for sequence models.",
        risk="Tests ordering, not sample selection.",
    ),
}


def get_model_spec(model_id: str) -> ModelSpec:
    return MODEL_REGISTRY[model_id]


def unimplemented_model_message(model_id: str) -> str:
    spec = get_model_spec(model_id)
    if spec.implemented:
        return f"{model_id} is implemented and available in the research protocol."
    return f"{model_id} is registered for the protocol but not implemented in v1."


def _event_date_column(frame: pd.DataFrame) -> str:
    for column in ("announcement_date", "event_date", "entry_date"):
        if column in frame.columns:
            return column
    raise ValueError("frame requires announcement_date, event_date, or entry_date")


def _sorted_events(frame: pd.DataFrame) -> pd.DataFrame:
    date_col = _event_date_column(frame)
    out = frame.copy()
    out[date_col] = pd.to_datetime(out[date_col], errors="coerce")
    return out.sort_values([date_col, "ticker"] if "ticker" in out.columns else [date_col]).copy()


def _prior_rolling_mean(
    frame: pd.DataFrame,
    column: str,
    *,
    window: int = 4,
    fallback_col: str = MARKET_BASELINE_COL,
) -> pd.Series:
    if column not in frame.columns:
        raise ValueError(f"frame must include {column}")
    out = _sorted_events(frame)
    values = pd.to_numeric(out[column], errors="coerce")
    by_ticker = (
        values.groupby(out["ticker"].astype(str)).transform(
            lambda series: series.shift(1).rolling(window, min_periods=1).mean()
        )
        if "ticker" in out.columns
        else values.shift(1).rolling(window, min_periods=1).mean()
    )
    global_prior = values.shift(1).expanding(min_periods=1).mean()
    fallback = (
        pd.to_numeric(out[fallback_col], errors="coerce") if fallback_col in out else global_prior
    )
    result = by_ticker.fillna(global_prior).fillna(fallback).fillna(values)
    return result.reindex(frame.index)


def add_benchmark_predictions(frame: pd.DataFrame) -> pd.DataFrame:
    required = {TARGET_COL, MARKET_BASELINE_COL}
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(f"feature frame missing required columns: {missing}")
    out = frame.copy()
    market = pd.to_numeric(out[MARKET_BASELINE_COL], errors="coerce")
    rvar_last4 = _prior_rolling_mean(out, TARGET_COL).clip(lower=0)
    ivar_last4 = _prior_rolling_mean(out, MARKET_BASELINE_COL).clip(lower=0)
    trailing_spread = _prior_rolling_mean(
        out.assign(_rv_iv_spread=pd.to_numeric(out[TARGET_COL], errors="coerce") - market),
        "_rv_iv_spread",
        fallback_col=MARKET_BASELINE_COL,
    ).fillna(0.0)
    out["forecast_market_implied_event_variance"] = market
    out["forecast_last_four_rvar"] = rvar_last4
    out["forecast_last_four_ivar"] = ivar_last4
    out["forecast_goyal_saretto_rv_iv_spread"] = (market + trailing_spread).clip(lower=0)
    out["patell_wolfson_prior_rvar_mean"] = rvar_last4
    out["patell_wolfson_prior_ivar_mean"] = ivar_last4
    out["patell_wolfson_prior_rv_iv_spread"] = rvar_last4 - ivar_last4
    out["patell_wolfson_iv_runup_proxy"] = market / ivar_last4.replace(0, np.nan) - 1.0
    out["mispricing_realized"] = pd.to_numeric(out[TARGET_COL], errors="coerce") - market
    return out


def default_feature_columns(frame: pd.DataFrame) -> list[str]:
    columns: list[str] = []
    for column in frame.columns:
        lower = column.lower()
        if any(pattern in lower for pattern in DEFAULT_FEATURE_EXCLUDE_PATTERNS):
            continue
        if lower.startswith(PREDICTION_PREFIX):
            continue
        if pd.api.types.is_numeric_dtype(frame[column]) or pd.api.types.is_bool_dtype(
            frame[column]
        ):
            columns.append(column)
    return columns


def _numeric_feature_frame(frame: pd.DataFrame, features: Sequence[str]) -> pd.DataFrame:
    return frame[list(features)].apply(pd.to_numeric, errors="coerce").fillna(0.0).astype(float)


def _finite_target_training_frame(
    frame: pd.DataFrame, target_col: str
) -> tuple[pd.DataFrame, np.ndarray]:
    target = pd.to_numeric(frame[target_col], errors="coerce")
    valid = np.isfinite(target)
    if not bool(valid.any()):
        raise ValueError(f"no finite {target_col} rows for model fit")
    return frame.loc[valid].copy(), target.loc[valid].to_numpy(dtype=float)


class LinearElasticNetRegressor:
    def __init__(self, *, alpha: float = 0.01, l1_ratio: float = 0.15, max_iter: int = 500):
        if alpha < 0:
            raise ValueError("alpha must be nonnegative")
        if not 0 <= l1_ratio <= 1:
            raise ValueError("l1_ratio must be in [0, 1]")
        self.alpha = alpha
        self.l1_ratio = l1_ratio
        self.max_iter = max_iter
        self.feature_columns: list[str] = []
        self.mean_: np.ndarray | None = None
        self.scale_: np.ndarray | None = None
        self.coef_: np.ndarray | None = None
        self.intercept_: float = 0.0

    def _matrix(self, frame: pd.DataFrame) -> np.ndarray:
        if not self.feature_columns:
            raise ValueError("model is not fit")
        x = (
            frame[self.feature_columns]
            .apply(pd.to_numeric, errors="coerce")
            .fillna(0.0)
            .to_numpy(dtype=float)
        )
        mean = self.mean_
        scale = self.scale_
        if mean is None or scale is None:
            raise ValueError("model is not fit")
        return np.asarray((x - mean) / scale, dtype=float)

    def fit(self, frame: pd.DataFrame, *, target_col: str, feature_columns: Sequence[str]) -> None:
        self.feature_columns = list(feature_columns)
        x_raw = (
            frame[self.feature_columns]
            .apply(pd.to_numeric, errors="coerce")
            .fillna(0.0)
            .to_numpy(dtype=float)
        )
        y = pd.to_numeric(frame[target_col], errors="coerce").to_numpy(dtype=float)
        valid = np.isfinite(y)
        if not bool(valid.any()):
            raise ValueError("no finite target rows for elastic-net fit")
        x_raw = x_raw[valid]
        y = y[valid]
        self.mean_ = x_raw.mean(axis=0)
        self.scale_ = np.where(x_raw.std(axis=0) <= 1e-12, 1.0, x_raw.std(axis=0))
        x = (x_raw - self.mean_) / self.scale_
        self.intercept_ = float(y.mean())
        y_centered = y - self.intercept_
        beta = np.zeros(x.shape[1], dtype=float)
        l1 = self.alpha * self.l1_ratio
        l2 = self.alpha * (1.0 - self.l1_ratio)
        for _ in range(self.max_iter):
            old = beta.copy()
            for j in range(x.shape[1]):
                residual = y_centered - x @ beta + x[:, j] * beta[j]
                rho = float(np.dot(x[:, j], residual) / len(y_centered))
                denom = float(np.dot(x[:, j], x[:, j]) / len(y_centered) + l2)
                beta[j] = np.sign(rho) * max(abs(rho) - l1, 0.0) / max(denom, 1e-12)
            if float(np.max(np.abs(beta - old))) < 1e-8:
                break
        self.coef_ = beta

    def predict(self, frame: pd.DataFrame) -> np.ndarray:
        coef = self.coef_
        if coef is None:
            raise ValueError("model is not fit")
        return np.asarray(
            np.maximum(self.intercept_ + self._matrix(frame) @ coef, 0.0), dtype=float
        )


def temporal_train_test_split(
    frame: pd.DataFrame,
    *,
    split_date: str | pd.Timestamp | None = None,
    train_fraction: float = 0.7,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    out = _sorted_events(frame)
    date_col = _event_date_column(out)
    if split_date is not None:
        split = pd.Timestamp(split_date)
        train = out.loc[out[date_col] < split].copy()
        test = out.loc[out[date_col] >= split].copy()
    else:
        cut = max(1, int(len(out) * train_fraction))
        train = out.iloc[:cut].copy()
        test = out.iloc[cut:].copy()
    if train.empty or test.empty:
        raise ValueError("temporal split produced an empty train or test set")
    return train, test


def fit_linear_elastic_net(
    frame: pd.DataFrame,
    *,
    target_col: str = TARGET_COL,
    feature_columns: Sequence[str] | None = None,
    split_date: str | pd.Timestamp | None = None,
) -> ModelFitResult:
    features = list(feature_columns or default_feature_columns(frame))
    train, test = temporal_train_test_split(frame, split_date=split_date)
    model = LinearElasticNetRegressor()
    model.fit(train, target_col=target_col, feature_columns=features)
    predictions = test.copy()
    predictions["forecast_linear_elastic_net"] = model.predict(test)
    return ModelFitResult(
        model_id="linear_elastic_net",
        predictions=predictions,
        feature_columns=features,
        diagnostics={"train_rows": int(len(train)), "test_rows": int(len(test))},
        model=model,
    )


def fit_lightgbm(
    frame: pd.DataFrame,
    *,
    target_col: str = TARGET_COL,
    feature_columns: Sequence[str] | None = None,
    split_date: str | pd.Timestamp | None = None,
) -> ModelFitResult:  # pragma: no cover - optional dependency exercised in integration environments
    try:
        import lightgbm as lgb
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError("lightgbm extra is required for model_id=lightgbm") from exc
    features = list(feature_columns or default_feature_columns(frame))
    train, test = temporal_train_test_split(frame, split_date=split_date)
    try:
        model = lgb.LGBMRegressor(
            n_estimators=200,
            learning_rate=0.03,
            max_depth=-1,
            num_leaves=31,
            random_state=17,
            verbose=-1,
        )
        train_fit, y_train = _finite_target_training_frame(train, target_col)
        model.fit(_numeric_feature_frame(train_fit, features), y_train)
    except Exception as exc:  # pragma: no cover
        message = str(exc)
        if "scikit-learn is required" in message or "No module named 'sklearn'" in message:
            raise RuntimeError("lightgbm extra requires scikit-learn") from exc
        raise
    predictions = test.copy()
    predictions["forecast_lightgbm"] = np.maximum(
        model.predict(_numeric_feature_frame(test, features)), 0.0
    )
    return ModelFitResult(
        model_id="lightgbm",
        predictions=predictions,
        feature_columns=features,
        diagnostics={"train_rows": int(len(train)), "test_rows": int(len(test))},
        model=model,
    )


def fit_xgboost(
    frame: pd.DataFrame,
    *,
    target_col: str = TARGET_COL,
    feature_columns: Sequence[str] | None = None,
    split_date: str | pd.Timestamp | None = None,
) -> ModelFitResult:  # pragma: no cover - optional dependency exercised in integration environments
    try:
        xgb = importlib.import_module("xgboost")
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError("xgboost extra is required for model_id=xgboost") from exc
    features = list(feature_columns or default_feature_columns(frame))
    train, test = temporal_train_test_split(frame, split_date=split_date)
    try:
        model = xgb.XGBRegressor(
            n_estimators=200,
            learning_rate=0.03,
            max_depth=4,
            subsample=0.9,
            colsample_bytree=0.9,
            objective="reg:squarederror",
            random_state=17,
        )
        train_fit, y_train = _finite_target_training_frame(train, target_col)
        model.fit(_numeric_feature_frame(train_fit, features), y_train)
    except Exception as exc:  # pragma: no cover
        message = str(exc)
        if "No module named 'sklearn'" in message or "scikit-learn" in message:
            raise RuntimeError("xgboost extra requires scikit-learn") from exc
        raise
    predictions = test.copy()
    predictions["forecast_xgboost"] = np.maximum(
        model.predict(_numeric_feature_frame(test, features)), 0.0
    )
    return ModelFitResult(
        model_id="xgboost",
        predictions=predictions,
        feature_columns=features,
        diagnostics={"train_rows": int(len(train)), "test_rows": int(len(test))},
        model=model,
    )


class FTTransformerRegressor(nn.Module):
    def __init__(
        self,
        *,
        n_features: int,
        d_token: int = 32,
        n_heads: int = 4,
        n_layers: int = 2,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.feature_projection = nn.Linear(1, d_token)
        self.feature_embedding = nn.Parameter(torch.zeros(n_features, d_token))
        self.cls_token = nn.Parameter(torch.zeros(1, 1, d_token))
        layer = nn.TransformerEncoderLayer(
            d_model=d_token,
            nhead=n_heads,
            dim_feedforward=d_token * 4,
            dropout=dropout,
            batch_first=True,
            activation="gelu",
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=n_layers)
        self.head = nn.Sequential(
            nn.LayerNorm(d_token),
            nn.Dropout(dropout),
            nn.Linear(d_token, 1),
            nn.Softplus(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        tokens = self.feature_projection(x.unsqueeze(-1)) + self.feature_embedding.unsqueeze(0)
        cls = self.cls_token.expand(x.shape[0], -1, -1)
        encoded = self.encoder(torch.cat([cls, tokens], dim=1))
        return cast(torch.Tensor, self.head(encoded[:, 0]).squeeze(-1))


def _validate_sequence(sequence: torch.Tensor) -> None:
    if sequence.ndim != 3:
        raise ValueError("sequence must have shape [batch, time, features]")


def _masked_mean(sequence: torch.Tensor, mask: torch.Tensor | None) -> torch.Tensor:
    if mask is None:
        return sequence.mean(dim=1)
    weights = mask.to(dtype=sequence.dtype).unsqueeze(-1)
    denom = weights.sum(dim=1).clamp_min(1.0)
    return (sequence * weights).sum(dim=1) / denom


def _masked_max(sequence: torch.Tensor, mask: torch.Tensor | None) -> torch.Tensor:
    if mask is None:
        return torch.max(sequence, dim=1).values
    masked = sequence.masked_fill(~mask.unsqueeze(-1), -1e9)
    values = torch.max(masked, dim=1).values
    return torch.where(torch.isfinite(values), values, torch.zeros_like(values))


class AttentionPoolingSequenceEncoder(nn.Module):
    """Small learned-query attention pooler for event-level sequence forecasts."""

    def __init__(
        self,
        *,
        n_features: int,
        hidden_size: int = 32,
        n_heads: int = 4,
    ):
        super().__init__()
        if hidden_size % n_heads != 0:
            raise ValueError("hidden_size must be divisible by n_heads")
        self.input_projection = nn.Linear(n_features, hidden_size)
        self.query = nn.Parameter(torch.zeros(1, 1, hidden_size))
        self.attention = nn.MultiheadAttention(hidden_size, n_heads, batch_first=True)
        self.head = nn.Sequential(nn.LayerNorm(hidden_size), nn.Linear(hidden_size, 1))
        self.last_attention_weights: torch.Tensor | None = None

    def forward(self, sequence: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
        _validate_sequence(sequence)
        tokens = self.input_projection(sequence)
        query = self.query.expand(sequence.shape[0], -1, -1)
        key_padding_mask = None if mask is None else ~mask.bool()
        pooled, weights = self.attention(
            query,
            tokens,
            tokens,
            key_padding_mask=key_padding_mask,
            need_weights=True,
            average_attn_weights=True,
        )
        self.last_attention_weights = weights.detach()
        return cast(torch.Tensor, self.head(pooled[:, 0, :]).squeeze(-1))


class BiGRUSequenceEncoder(nn.Module):
    """LayerNorm bidirectional GRU for short pre-entry option-surface paths."""

    def __init__(
        self,
        *,
        n_features: int,
        hidden_size: int = 32,
        n_layers: int = 1,
        dropout: float = 0.15,
    ):
        super().__init__()
        self.input_norm = nn.LayerNorm(n_features)
        self.gru = nn.GRU(
            input_size=n_features,
            hidden_size=hidden_size,
            num_layers=n_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if n_layers > 1 else 0.0,
        )
        self.head = nn.Sequential(
            nn.LayerNorm(hidden_size * 4),
            nn.Dropout(dropout),
            nn.Linear(hidden_size * 4, 1),
        )

    def forward(self, sequence: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
        _validate_sequence(sequence)
        encoded, _ = self.gru(self.input_norm(sequence))
        pooled = torch.cat([_masked_mean(encoded, mask), _masked_max(encoded, mask)], dim=1)
        return cast(torch.Tensor, self.head(pooled).squeeze(-1))


class DilatedCNNSequenceEncoder(nn.Module):
    """Non-causal dilated CNN for offline pre-entry sequence encoding."""

    def __init__(
        self,
        *,
        n_features: int,
        channels: Sequence[int] = (32, 64, 64),
        dilations: Sequence[int] = (1, 2, 4),
        dropout: float = 0.15,
    ):
        super().__init__()
        if len(channels) != len(dilations):
            raise ValueError("channels and dilations must have equal length")
        layers: list[nn.Module] = []
        in_channels = n_features
        for out_channels, dilation in zip(channels, dilations, strict=True):
            padding = dilation
            layers.extend(
                [
                    nn.Conv1d(
                        in_channels,
                        out_channels,
                        kernel_size=3,
                        dilation=dilation,
                        padding=padding,
                    ),
                    nn.GELU(),
                    nn.Dropout(dropout),
                ]
            )
            in_channels = out_channels
        self.encoder = nn.Sequential(*layers)
        self.head = nn.Sequential(nn.LayerNorm(in_channels), nn.Linear(in_channels, 1))

    def forward(self, sequence: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
        _validate_sequence(sequence)
        encoded = self.encoder(sequence.transpose(1, 2)).transpose(1, 2)
        return cast(torch.Tensor, self.head(_masked_mean(encoded, mask)).squeeze(-1))


class MambaSSMSequenceEncoder(nn.Module):
    """Bidirectional wrapper around official `mamba_ssm.Mamba`.

    This is a non-causal encoder over a completed pre-entry path. It is not a causal
    event-after-entry predictor and therefore does not use post-entry information.
    """

    def __init__(
        self,
        *,
        n_features: int,
        hidden_size: int = 32,
        n_layers: int = 1,
        dropout: float = 0.15,
    ):
        super().__init__()
        if not torch.cuda.is_available():
            raise RuntimeError("mamba_ssm_sequence requires CUDA-enabled torch")
        try:
            module = importlib.import_module("mamba_ssm")
            mamba_cls = cast(Any, module.Mamba)
        except (ImportError, AttributeError) as exc:
            raise RuntimeError("mamba_ssm package with Mamba class is required") from exc
        self.input_projection = nn.Linear(n_features, hidden_size)
        self.forward_layers = nn.ModuleList(
            [mamba_cls(d_model=hidden_size) for _ in range(n_layers)]
        )
        self.reverse_layers = nn.ModuleList(
            [mamba_cls(d_model=hidden_size) for _ in range(n_layers)]
        )
        self.head = nn.Sequential(
            nn.LayerNorm(hidden_size * 4),
            nn.Dropout(dropout),
            nn.Linear(hidden_size * 4, 1),
        )

    def _encode(self, tokens: torch.Tensor, layers: nn.ModuleList) -> torch.Tensor:
        out = tokens
        for layer in layers:
            out = cast(torch.Tensor, layer(out))
        return out

    def forward(self, sequence: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
        _validate_sequence(sequence)
        tokens = self.input_projection(sequence)
        forward = self._encode(tokens, self.forward_layers)
        reverse = torch.flip(
            self._encode(torch.flip(tokens, dims=[1]), self.reverse_layers), dims=[1]
        )
        encoded = torch.cat([forward, reverse], dim=2)
        pooled = torch.cat([_masked_mean(encoded, mask), _masked_max(encoded, mask)], dim=1)
        return cast(torch.Tensor, self.head(pooled).squeeze(-1))


def _torch_matrix(frame: pd.DataFrame, features: Sequence[str]) -> torch.Tensor:
    data = _numeric_feature_frame(frame, features).to_numpy(dtype=float)
    return torch.tensor(data, dtype=torch.float32)


def fit_ft_transformer(
    frame: pd.DataFrame,
    *,
    target_col: str = TARGET_COL,
    feature_columns: Sequence[str] | None = None,
    split_date: str | pd.Timestamp | None = None,
    epochs: int = 40,
) -> ModelFitResult:
    features = list(feature_columns or default_feature_columns(frame))
    train, test = temporal_train_test_split(frame, split_date=split_date)
    model = FTTransformerRegressor(n_features=len(features))
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    train_fit, y_train_array = _finite_target_training_frame(train, target_col)
    x_train = _torch_matrix(train_fit, features)
    y_train = torch.tensor(y_train_array, dtype=torch.float32)
    for _ in range(max(1, epochs)):
        optimizer.zero_grad()
        loss = torch.mean(torch.square(model(x_train) - y_train))
        loss.backward()  # type: ignore[no-untyped-call]
        optimizer.step()
    with torch.no_grad():
        pred = model(_torch_matrix(test, features)).detach().numpy()
    predictions = test.copy()
    predictions["forecast_ft_transformer"] = pred
    return ModelFitResult(
        model_id="ft_transformer",
        predictions=predictions,
        feature_columns=features,
        diagnostics={"train_rows": int(len(train)), "test_rows": int(len(test)), "epochs": epochs},
        model=model,
    )


def fit_model(
    model_id: str,
    frame: pd.DataFrame,
    *,
    target_col: str = TARGET_COL,
    feature_columns: Sequence[str] | None = None,
    split_date: str | pd.Timestamp | None = None,
) -> ModelFitResult:
    if model_id == "linear_elastic_net":
        return fit_linear_elastic_net(
            frame, target_col=target_col, feature_columns=feature_columns, split_date=split_date
        )
    if model_id == "lightgbm":
        return fit_lightgbm(
            frame, target_col=target_col, feature_columns=feature_columns, split_date=split_date
        )
    if model_id == "xgboost":
        return fit_xgboost(
            frame, target_col=target_col, feature_columns=feature_columns, split_date=split_date
        )
    if model_id == "ft_transformer":
        return fit_ft_transformer(
            frame, target_col=target_col, feature_columns=feature_columns, split_date=split_date
        )
    raise ValueError(f"{model_id} is not a trainable tabular model")


def prediction_column_for_model(model_id: str) -> str:
    mapping = {
        "market_implied_event_variance": "forecast_market_implied_event_variance",
        "last_four_rvar": "forecast_last_four_rvar",
        "last_four_ivar": "forecast_last_four_ivar",
        "goyal_saretto_rv_iv_spread": "forecast_goyal_saretto_rv_iv_spread",
        "linear_elastic_net": "forecast_linear_elastic_net",
        "linear_elastic_net_tuned": "forecast_linear_elastic_net_tuned",
        "lightgbm": "forecast_lightgbm",
        "lightgbm_tuned": "forecast_lightgbm_tuned",
        "xgboost": "forecast_xgboost",
        "xgboost_tuned": "forecast_xgboost_tuned",
        "ft_transformer": "forecast_ft_transformer",
        "ft_transformer_tuned": "forecast_ft_transformer_tuned",
        "lightgbm_xgboost_mean_ensemble": "forecast_lightgbm_xgboost_mean_ensemble",
        "ridge_flat_aggregates_sequence": "forecast_ridge_flat_aggregates_sequence",
        "attention_pooling_sequence": "forecast_attention_pooling_sequence",
        "bigru_sequence": "forecast_bigru_sequence",
        "bigru_sequence_5seed": "forecast_bigru_sequence_5seed",
        "dilated_cnn_sequence": "forecast_dilated_cnn_sequence",
        "mamba_ssm_sequence": "forecast_mamba_ssm_sequence",
        "mamba_ssm_sequence_5seed": "forecast_mamba_ssm_sequence_5seed",
        "mask_only_sequence": "forecast_mask_only_sequence",
        "time_shuffle_sequence": "forecast_time_shuffle_sequence",
        "lightgbm_with_hybrid_aggregates": "forecast_lightgbm_with_hybrid_aggregates",
    }
    return mapping[model_id]


def sequence_feature_columns(frame: pd.DataFrame) -> list[str]:
    return sorted(column for column in frame.columns if column.startswith("seq_t"))


def sequence_tensor_from_frame(frame: pd.DataFrame, columns: Sequence[str]) -> torch.Tensor:
    """Build [event, time, feature] tensor from columns named `seq_tXX_feature`."""
    if not columns:
        raise ValueError("at least one sequence column is required")
    parsed: list[tuple[int, str, str]] = []
    for column in columns:
        parts = column.split("_", 2)
        if len(parts) != 3 or not parts[1].startswith("t"):
            raise ValueError(f"invalid sequence column name: {column}")
        parsed.append((int(parts[1][1:]), parts[2], column))
    times = sorted({item[0] for item in parsed})
    features = sorted({item[1] for item in parsed})
    arrays: list[np.ndarray] = []
    for time_index in times:
        feature_arrays: list[np.ndarray] = []
        for feature in features:
            matched_column = next(
                (raw for t, f, raw in parsed if t == time_index and f == feature),
                None,
            )
            if matched_column is None:
                feature_arrays.append(np.zeros(len(frame), dtype=float))
            else:
                feature_arrays.append(
                    pd.to_numeric(frame[matched_column], errors="coerce")
                    .fillna(0.0)
                    .to_numpy(dtype=float)
                )
        arrays.append(np.stack(feature_arrays, axis=1))
    return torch.tensor(np.stack(arrays, axis=1), dtype=torch.float32)


def run_model_suite(
    frame: pd.DataFrame,
    *,
    model_ids: Sequence[str],
    split_date: str | pd.Timestamp | None = None,
) -> tuple[pd.DataFrame, list[ModelFitResult]]:
    base = add_benchmark_predictions(frame)
    trainable = {
        "linear_elastic_net",
        "lightgbm",
        "xgboost",
        "ft_transformer",
    }
    results: list[ModelFitResult] = []
    predictions = base.copy()
    for model_id in model_ids:
        if model_id in {
            "market_implied_event_variance",
            "last_four_rvar",
            "last_four_ivar",
            "goyal_saretto_rv_iv_spread",
            "patell_wolfson_diagnostic",
        }:
            results.append(
                ModelFitResult(
                    model_id=model_id,
                    predictions=pd.DataFrame(),
                    feature_columns=[],
                    diagnostics={
                        "status": "diagnostic_features_only"
                        if model_id == "patell_wolfson_diagnostic"
                        else "evaluated"
                    },
                )
            )
            continue
        if model_id in trainable:
            try:
                result = fit_model(model_id, base, split_date=split_date)
            except RuntimeError as exc:
                results.append(
                    ModelFitResult(
                        model_id=model_id,
                        predictions=pd.DataFrame(),
                        feature_columns=[],
                        diagnostics={"status": "skipped_dependency_unavailable", "error": str(exc)},
                    )
                )
                continue
        else:
            raise ValueError(f"unknown model_id: {model_id}")
        results.append(result)
        column = prediction_column_for_model(model_id)
        if not result.predictions.empty and column in result.predictions.columns:
            predictions.loc[result.predictions.index, column] = result.predictions[column]
    return predictions, results


def model_diagnostics_as_frame(results: Sequence[ModelFitResult]) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for result in results:
        rows.append(
            {
                "model_id": result.model_id,
                "feature_count": len(result.feature_columns),
                **result.diagnostics,
            }
        )
    return pd.DataFrame(rows)

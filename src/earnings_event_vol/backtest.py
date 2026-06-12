from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass
from datetime import date
from typing import Protocol, cast

import numpy as np
import pandas as pd
from scipy.stats import norm

from earnings_event_vol.metrics import max_drawdown, strategy_metrics
from earnings_event_vol.schemas import (
    OptionQuote,
    OptionRight,
    OptionSide,
    SignalRecord,
    StrategyTrade,
    TradeLeg,
)

DEFAULT_STRATEGY_THRESHOLD_MULTIPLIERS = (0.5, 1.0, 1.5, 2.0, 3.0)
DEFAULT_STRATEGY_MIN_EDGE_VARS = (0.0, 0.0005, 0.001)
DEFAULT_STRATEGY_TOP_K_VALUES = (None, 5, 10, 20)
DEFAULT_STRATEGY_MIN_VALIDATION_TRADES = 3
DEFAULT_STRATEGY_DRAWDOWN_PENALTY = 0.10
DEFAULT_QUOTE_MAX_AGE_SECONDS = 60.0
DEFAULT_QUOTE_MAX_SPREAD_OVER_MID = 0.25


@dataclass(frozen=True)
class StrategyPolicy:
    threshold_multiplier: float = 1.5
    min_edge_var: float = 0.0
    top_k: int | None = None
    allowed_liquidity_buckets: tuple[str, ...] = ()
    require_main_dte_5_14: bool | None = None
    allowed_dte_buckets: tuple[str, ...] = ()
    allowed_execution_confidence_bands: tuple[str, ...] = ()
    min_execution_confidence_score: float | None = None
    max_median_spread_over_mid: float | None = None
    max_quote_age_seconds: float | None = None
    quote_filter_status: str = "unavailable"
    selection_split: str = "validation"
    selection_score_col: str = "expected_strategy_edge_usd"
    objective_name: str = "net_pnl_minus_drawdown_penalty"


def strategy_policy_to_dict(policy: StrategyPolicy) -> dict[str, object]:
    return {
        "threshold_multiplier": float(policy.threshold_multiplier),
        "min_edge_var": float(policy.min_edge_var),
        "top_k": policy.top_k,
        "allowed_liquidity_buckets": "|".join(policy.allowed_liquidity_buckets),
        "require_main_dte_5_14": policy.require_main_dte_5_14,
        "allowed_dte_buckets": "|".join(policy.allowed_dte_buckets),
        "allowed_execution_confidence_bands": "|".join(policy.allowed_execution_confidence_bands),
        "min_execution_confidence_score": policy.min_execution_confidence_score,
        "max_median_spread_over_mid": policy.max_median_spread_over_mid,
        "max_quote_age_seconds": policy.max_quote_age_seconds,
        "quote_filter_status": policy.quote_filter_status,
        "selection_split": policy.selection_split,
        "selection_score_col": policy.selection_score_col,
        "objective_name": policy.objective_name,
    }


class EventJumpDistribution(Protocol):
    def support(self, variance: float) -> tuple[np.ndarray, np.ndarray]:
        """Return event log-return points and probabilities."""


@dataclass(frozen=True)
class GaussianEventJumpDistribution:
    nodes: int = 21

    def support(self, variance: float) -> tuple[np.ndarray, np.ndarray]:
        if variance < 0:
            raise ValueError("variance must be nonnegative")
        if variance == 0:
            return np.array([0.0]), np.array([1.0])
        hermite_nodes, hermite_weights = np.polynomial.hermite.hermgauss(self.nodes)
        returns = np.sqrt(2.0 * variance) * hermite_nodes
        probabilities = hermite_weights / np.sqrt(np.pi)
        return returns, probabilities


@dataclass(frozen=True)
class SymmetricTwoPointJumpDistribution:
    def support(self, variance: float) -> tuple[np.ndarray, np.ndarray]:
        if variance < 0:
            raise ValueError("variance must be nonnegative")
        jump = float(np.sqrt(variance))
        return np.array([-jump, jump]), np.array([0.5, 0.5])


def black_scholes_price(
    *,
    spot: float,
    strike: float,
    time_to_expiry: float,
    volatility: float,
    right: OptionRight,
    rate: float = 0.0,
) -> float:
    if spot <= 0 or strike <= 0:
        raise ValueError("spot and strike must be positive")
    if time_to_expiry <= 0 or volatility <= 0:
        intrinsic = (
            max(spot - strike, 0.0) if right == OptionRight.CALL else max(strike - spot, 0.0)
        )
        return float(intrinsic)
    sqrt_t = np.sqrt(time_to_expiry)
    d1 = (np.log(spot / strike) + (rate + 0.5 * volatility**2) * time_to_expiry) / (
        volatility * sqrt_t
    )
    d2 = d1 - volatility * sqrt_t
    if right == OptionRight.CALL:
        return float(spot * norm.cdf(d1) - strike * np.exp(-rate * time_to_expiry) * norm.cdf(d2))
    return float(strike * np.exp(-rate * time_to_expiry) * norm.cdf(-d2) - spot * norm.cdf(-d1))


def option_payoff(terminal_spot: np.ndarray, *, strike: float, right: OptionRight) -> np.ndarray:
    if right == OptionRight.CALL:
        return np.maximum(terminal_spot - strike, 0.0)
    return np.maximum(strike - terminal_spot, 0.0)


def expected_strategy_value_usd(
    *,
    spot: float,
    forecast_rvar_event: float,
    legs: Sequence[TradeLeg],
    distribution: EventJumpDistribution | None = None,
) -> float:
    """Value an event strategy using terminal intrinsic payoff after the event jump.

    This v1 smoke valuation deliberately ignores discounting and any residual post-event time
    value. Paper-grade runs should either trade contracts whose post-event residual value is
    negligible or replace this with a marked-to-post-event option-pricing layer.
    """
    distribution = distribution or GaussianEventJumpDistribution()
    returns, probabilities = distribution.support(forecast_rvar_event)
    terminal_spot = spot * np.exp(returns)
    total = np.zeros_like(terminal_spot)
    for leg in legs:
        side = 1.0 if leg.side == OptionSide.LONG else -1.0
        total += (
            side
            * float(leg.contracts)
            * float(leg.option_multiplier)
            * option_payoff(terminal_spot, strike=leg.strike, right=leg.right)
        )
    return float(np.sum(total * probabilities))


def market_entry_cost_usd(legs: Sequence[TradeLeg]) -> float:
    total = 0.0
    for leg in legs:
        sign = 1.0 if leg.side == OptionSide.LONG else -1.0
        total += sign * float(leg.contracts) * float(leg.option_multiplier) * leg.filled_price
    return float(total)


def estimated_transaction_cost_usd(
    quotes: Sequence[OptionQuote], *, contracts: float = 1.0
) -> float:
    return float(
        sum(
            ((quote.ask - quote.bid) / 2.0) * float(contracts) * quote.option_multiplier
            for quote in quotes
        )
    )


def premium_space_signal(
    *,
    ticker: str,
    event_date: date,
    strategy: str,
    forecast_rvar_event: float,
    ivar_event: float,
    expected_value_usd: float,
    entry_cost_usd: float,
    transaction_cost_usd: float,
    threshold_multiplier: float = 1.5,
) -> SignalRecord:
    expected_edge = expected_value_usd - entry_cost_usd
    return SignalRecord(
        ticker=ticker,
        event_date=event_date,
        strategy=strategy,
        forecast_rvar_event=forecast_rvar_event,
        ivar_event=ivar_event,
        edge_var=forecast_rvar_event - ivar_event,
        expected_strategy_value_usd=expected_value_usd,
        market_entry_cost_usd=entry_cost_usd,
        expected_strategy_edge_usd=expected_edge,
        estimated_transaction_cost_usd=transaction_cost_usd,
        threshold_multiplier=threshold_multiplier,
    )


def build_proxy_strategy_frame(
    frame: pd.DataFrame,
    *,
    forecast_col: str,
    ivar_col: str = "ivar_event",
    realized_long_pnl_col: str = "gross_proxy_pnl_usd",
    entry_premium_col: str = "entry_premium_usd",
    cost_col: str = "estimated_transaction_cost_usd",
    min_edge_var: float = 0.0,
    threshold_multiplier: float = 1.5,
) -> pd.DataFrame:
    """Evaluate cost-aware proxy straddle trades from event-level forecasts.

    Positive forecast edge buys the proxy straddle only when the premium-space edge clears
    the configured transaction-cost threshold. Negative variance-edge rows are kept as
    diagnostics, but this proxy route does not open naked short straddles because their risk
    is not defined by the observed entry premium.
    """
    if threshold_multiplier <= 0:
        raise ValueError("threshold_multiplier must be positive.")
    required = {forecast_col, ivar_col, realized_long_pnl_col, entry_premium_col}
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(f"strategy frame missing required columns: {missing}")
    out = frame.copy()
    forecast = pd.to_numeric(out[forecast_col], errors="coerce")
    ivar = pd.to_numeric(out[ivar_col], errors="coerce")
    premium = pd.to_numeric(out[entry_premium_col], errors="coerce")
    long_pnl = pd.to_numeric(out[realized_long_pnl_col], errors="coerce")
    cost = (
        pd.to_numeric(out[cost_col], errors="coerce")
        if cost_col in out.columns
        else (0.10 * premium).fillna(0.0)
    )
    edge_var = forecast - ivar
    expected_edge_usd = edge_var / ivar.replace(0.0, np.nan) * premium
    threshold_usd = threshold_multiplier * cost
    finite_signal = (
        np.isfinite(forecast)
        & np.isfinite(ivar)
        & np.isfinite(premium)
        & np.isfinite(cost)
        & ivar.gt(0)
        & premium.gt(0)
    )
    entry_signal = finite_signal & edge_var.gt(min_edge_var) & expected_edge_usd.gt(threshold_usd)
    direction = np.where(entry_signal, 1, 0)
    out["forecast_edge_var"] = edge_var
    out["expected_strategy_edge_usd"] = expected_edge_usd
    out["premium_space_threshold_usd"] = threshold_usd
    out["threshold_multiplier"] = float(threshold_multiplier)
    out["realized_pnl_available"] = np.isfinite(long_pnl)
    out["trade_direction"] = np.where(
        direction > 0, "long_straddle", np.where(edge_var < -min_edge_var, "no_trade", "no_trade")
    )
    out["estimated_transaction_cost_usd"] = cost
    out["gross_strategy_pnl_usd"] = direction * long_pnl
    out["net_proxy_pnl_usd"] = out["gross_strategy_pnl_usd"] - np.abs(direction) * cost
    out["capital_at_risk_usd"] = premium.abs()
    out["return_on_premium"] = out["net_proxy_pnl_usd"] / premium.abs().replace(0.0, np.nan)
    out["should_trade"] = entry_signal
    return out


def _policy_bool_filter(frame: pd.DataFrame, policy: StrategyPolicy) -> pd.Series:
    passed = pd.Series(True, index=frame.index, dtype=bool)
    if policy.allowed_liquidity_buckets:
        if "liquidity_bucket" not in frame.columns:
            return pd.Series(False, index=frame.index, dtype=bool)
        passed &= frame["liquidity_bucket"].astype(str).isin(policy.allowed_liquidity_buckets)
    if policy.require_main_dte_5_14 is not None:
        if "is_main_dte_5_14" not in frame.columns:
            return pd.Series(False, index=frame.index, dtype=bool)
        passed &= (
            frame["is_main_dte_5_14"].fillna(False).astype(bool).eq(policy.require_main_dte_5_14)
        )
    if policy.allowed_dte_buckets:
        if "dte_bucket" not in frame.columns:
            return pd.Series(False, index=frame.index, dtype=bool)
        passed &= frame["dte_bucket"].astype(str).isin(policy.allowed_dte_buckets)
    if policy.allowed_execution_confidence_bands:
        if "execution_confidence_band" not in frame.columns:
            return pd.Series(False, index=frame.index, dtype=bool)
        passed &= (
            frame["execution_confidence_band"]
            .fillna("missing")
            .astype(str)
            .isin(policy.allowed_execution_confidence_bands)
        )
    if policy.min_execution_confidence_score is not None:
        if "execution_confidence_score" not in frame.columns:
            return pd.Series(False, index=frame.index, dtype=bool)
        score = pd.to_numeric(frame["execution_confidence_score"], errors="coerce")
        passed &= score.ge(float(policy.min_execution_confidence_score)).fillna(False)
    if policy.max_median_spread_over_mid is not None:
        if "median_spread_over_mid" not in frame.columns:
            return pd.Series(False, index=frame.index, dtype=bool)
        spread = pd.to_numeric(frame["median_spread_over_mid"], errors="coerce")
        passed &= spread.le(float(policy.max_median_spread_over_mid)).fillna(False)
    if policy.max_quote_age_seconds is not None:
        if "max_quote_age_seconds" not in frame.columns:
            return pd.Series(False, index=frame.index, dtype=bool)
        age = pd.to_numeric(frame["max_quote_age_seconds"], errors="coerce")
        passed &= age.le(float(policy.max_quote_age_seconds)).fillna(False)
    return passed


def _top_k_mask(
    frame: pd.DataFrame,
    *,
    candidate: pd.Series,
    top_k: int | None,
    split_col: str,
    score_col: str,
) -> tuple[pd.Series, pd.Series]:
    ranks = pd.Series(np.nan, index=frame.index, dtype=float)
    if top_k is None:
        return pd.Series(True, index=frame.index, dtype=bool), ranks
    if top_k <= 0:
        raise ValueError("top_k must be positive when provided.")
    selected = pd.Series(False, index=frame.index, dtype=bool)
    scores = pd.to_numeric(frame[score_col], errors="coerce")
    group_values = (
        frame[split_col].fillna("_all").astype(str)
        if split_col in frame.columns
        else pd.Series("_all", index=frame.index, dtype=str)
    )
    for _, group_index in frame.loc[candidate].groupby(group_values.loc[candidate]).groups.items():
        group_scores = scores.loc[group_index].dropna().sort_values(ascending=False)
        if group_scores.empty:
            continue
        ordered = group_scores.index
        ranks.loc[ordered] = np.arange(1, len(ordered) + 1, dtype=float)
        selected.loc[ordered[:top_k]] = True
    return selected, ranks


def apply_strategy_policy(
    frame: pd.DataFrame,
    policy: StrategyPolicy,
    *,
    split_col: str = "split",
) -> pd.DataFrame:
    """Apply validation-selected trade filters to a proxy strategy frame."""
    required = {"should_trade", "gross_strategy_pnl_usd", "estimated_transaction_cost_usd"}
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(f"strategy frame missing required columns: {missing}")
    if policy.threshold_multiplier <= 0:
        raise ValueError("threshold_multiplier must be positive.")
    out = frame.copy()
    base_signal = out["should_trade"].fillna(False).astype(bool)
    filter_pass = _policy_bool_filter(out, policy)
    candidate = base_signal & filter_pass
    selection_score_col = (
        policy.selection_score_col
        if policy.selection_score_col in out.columns
        else "expected_strategy_edge_usd"
    )
    top_k_pass, ranks = _top_k_mask(
        out,
        candidate=candidate,
        top_k=policy.top_k,
        split_col=split_col,
        score_col=selection_score_col,
    )
    final_signal = candidate & top_k_pass
    policy_fields = strategy_policy_to_dict(policy)
    for key, value in policy_fields.items():
        out[f"strategy_policy_{key}"] = value
    out["strategy_policy_base_signal"] = base_signal
    out["strategy_policy_passes_filters"] = filter_pass
    out["strategy_policy_effective_selection_score_col"] = selection_score_col
    out["strategy_policy_rank"] = ranks
    out["strategy_policy_selected_top_k"] = top_k_pass if policy.top_k is not None else candidate
    out["should_trade"] = final_signal
    out["trade_direction"] = np.where(final_signal, "long_straddle", "no_trade")
    cost = pd.to_numeric(out["estimated_transaction_cost_usd"], errors="coerce").fillna(0.0)
    out.loc[~final_signal, "gross_strategy_pnl_usd"] = 0.0
    out["net_proxy_pnl_usd"] = out["gross_strategy_pnl_usd"] - final_signal.astype(float) * cost
    premium = (
        pd.to_numeric(out["capital_at_risk_usd"], errors="coerce")
        if "capital_at_risk_usd" in out.columns
        else pd.Series(np.nan, index=out.index, dtype=float)
    )
    out["return_on_premium"] = out["net_proxy_pnl_usd"] / premium.abs().replace(0.0, np.nan)
    return out


def _dedupe_modes(modes: Sequence[dict[str, object]]) -> list[dict[str, object]]:
    seen: set[tuple[tuple[str, str], ...]] = set()
    out: list[dict[str, object]] = []
    for mode in modes:
        key = tuple(sorted((str(k), str(v)) for k, v in mode.items()))
        if key in seen:
            continue
        seen.add(key)
        out.append(mode)
    return out


def _liquidity_modes(frame: pd.DataFrame) -> list[tuple[str, ...]]:
    modes: list[tuple[str, ...]] = [()]
    if "liquidity_bucket" not in frame.columns:
        return modes
    values = set(frame["liquidity_bucket"].dropna().astype(str))
    if {"high", "mid"} & values:
        modes.append(tuple(value for value in ("high", "mid") if value in values))
    if "high" in values:
        modes.append(("high",))
    return list(dict.fromkeys(modes))


def _dte_modes(frame: pd.DataFrame) -> list[dict[str, object]]:
    modes: list[dict[str, object]] = [{"require_main_dte_5_14": None, "allowed_dte_buckets": ()}]
    if "is_main_dte_5_14" in frame.columns:
        modes.append({"require_main_dte_5_14": True, "allowed_dte_buckets": ()})
    elif "dte_bucket" in frame.columns and frame["dte_bucket"].astype(str).eq("main_5_14").any():
        modes.append({"require_main_dte_5_14": None, "allowed_dte_buckets": ("main_5_14",)})
    return _dedupe_modes(modes)


def _quote_modes(frame: pd.DataFrame) -> list[dict[str, object]]:
    if "execution_confidence_band" not in frame.columns:
        return [
            {
                "allowed_execution_confidence_bands": (),
                "min_execution_confidence_score": None,
                "max_median_spread_over_mid": None,
                "max_quote_age_seconds": None,
                "quote_filter_status": "unavailable_missing_execution_confidence_band",
            }
        ]
    bands = frame["execution_confidence_band"].fillna("missing").astype(str)
    usable = bands.isin({"high", "medium"})
    if not bool(usable.any()):
        return [
            {
                "allowed_execution_confidence_bands": (),
                "min_execution_confidence_score": None,
                "max_median_spread_over_mid": None,
                "max_quote_age_seconds": None,
                "quote_filter_status": "unavailable_no_validation_high_medium_quotes",
            }
        ]
    has_score = "execution_confidence_score" in frame.columns
    has_spread = "median_spread_over_mid" in frame.columns
    has_age = "max_quote_age_seconds" in frame.columns
    modes = [
        {
            "allowed_execution_confidence_bands": ("high", "medium"),
            "min_execution_confidence_score": 0.5 if has_score else None,
            "max_median_spread_over_mid": DEFAULT_QUOTE_MAX_SPREAD_OVER_MID if has_spread else None,
            "max_quote_age_seconds": DEFAULT_QUOTE_MAX_AGE_SECONDS if has_age else None,
            "quote_filter_status": "required_high_medium_fresh_nonwide",
        },
        {
            "allowed_execution_confidence_bands": ("high",),
            "min_execution_confidence_score": 0.8 if has_score else None,
            "max_median_spread_over_mid": 0.15 if has_spread else None,
            "max_quote_age_seconds": DEFAULT_QUOTE_MAX_AGE_SECONDS if has_age else None,
            "quote_filter_status": "required_high_strict_fresh_tight",
        },
    ]
    return _dedupe_modes(modes)


def _strategy_objective(
    trades: pd.DataFrame,
    *,
    min_validation_trades: int,
    drawdown_penalty: float,
) -> dict[str, object]:
    metrics = strategy_metrics(trades, gross_pnl_col="gross_strategy_pnl_usd")
    net = metrics.get("net_pnl_usd")
    drawdown = metrics.get("max_drawdown_usd")
    n = int(metrics.get("n") or 0)
    net_value = float(net) if net is not None and np.isfinite(float(net)) else float("-inf")
    drawdown_value = (
        float(drawdown) if drawdown is not None and np.isfinite(float(drawdown)) else 0.0
    )
    objective = net_value - float(drawdown_penalty) * abs(min(drawdown_value, 0.0))
    if not np.isfinite(objective):
        objective = -1e18
    return {
        **metrics,
        "objective_value": float(objective),
        "meets_min_validation_trades": bool(n >= int(min_validation_trades)),
    }


def _strategy_objective_from_mask(
    frame: pd.DataFrame,
    selected: pd.Series,
    *,
    min_validation_trades: int,
    drawdown_penalty: float,
) -> dict[str, object]:
    selected_index = selected.loc[selected].index
    if len(selected_index) == 0:
        return {
            **strategy_metrics(frame.iloc[0:0], gross_pnl_col="gross_strategy_pnl_usd"),
            "objective_value": -1e18,
            "meets_min_validation_trades": False,
        }
    if "event_date" in frame.columns:
        selected_index = frame.loc[selected_index].sort_values("event_date").index
    net = pd.to_numeric(frame.loc[selected_index, "net_proxy_pnl_usd"], errors="coerce")
    valid = np.isfinite(net)
    net = net.loc[valid]
    selected_index = net.index
    if net.empty:
        return {
            **strategy_metrics(frame.iloc[0:0], gross_pnl_col="gross_strategy_pnl_usd"),
            "objective_value": -1e18,
            "meets_min_validation_trades": False,
        }
    gross = pd.to_numeric(
        frame.loc[selected_index, "gross_strategy_pnl_usd"],
        errors="coerce",
    )
    premium = pd.to_numeric(frame.loc[selected_index, "entry_premium_usd"], errors="coerce").abs()
    losses = net.loc[net < 0]
    wins = net.loc[net > 0]
    std = float(net.std(ddof=1)) if len(net) > 1 else 0.0
    downside = float(losses.std(ddof=1)) if len(losses) > 1 else 0.0
    premium_sum = float(premium.sum())
    drawdown = max_drawdown(net)
    n = int(len(net))
    metrics: dict[str, object] = {
        "n": n,
        "gross_pnl_usd": None if gross.dropna().empty else float(gross.sum()),
        "net_pnl_usd": float(net.sum()),
        "return_on_premium": None
        if not np.isfinite(premium_sum) or premium_sum <= 1e-12
        else float(net.sum() / premium_sum),
        "return_on_capital": None
        if not np.isfinite(premium_sum) or premium_sum <= 1e-12
        else float(net.sum() / premium_sum),
        "sharpe": None if std <= 1e-12 else float(net.mean() / std * np.sqrt(len(net))),
        "sortino": None if downside <= 1e-12 else float(net.mean() / downside * np.sqrt(len(net))),
        "max_drawdown_usd": drawdown,
        "hit_rate": float(net.gt(0).mean()),
        "avg_win_usd": None if wins.empty else float(wins.mean()),
        "avg_loss_usd": None if losses.empty else float(losses.mean()),
        "tail_loss_5pct_usd": float(net.quantile(0.05)),
        "turnover": n,
    }
    objective = float(net.sum()) - float(drawdown_penalty) * abs(min(float(drawdown), 0.0))
    metrics["objective_value"] = float(objective)
    metrics["meets_min_validation_trades"] = bool(n >= int(min_validation_trades))
    return metrics


def _top_k_single_group_mask(
    *,
    candidate: pd.Series,
    scores: pd.Series,
    top_k: int | None,
) -> pd.Series:
    if top_k is None:
        return candidate
    if top_k <= 0:
        raise ValueError("top_k must be positive when provided.")
    selected = pd.Series(False, index=candidate.index, dtype=bool)
    ordered = scores.loc[candidate].dropna().sort_values(ascending=False).index
    selected.loc[ordered[:top_k]] = True
    return selected


def tune_strategy_policy_validation_only(
    frame: pd.DataFrame,
    *,
    forecast_col: str,
    selection_score_col: str | None = None,
    ivar_col: str = "ivar_event",
    realized_long_pnl_col: str = "gross_proxy_pnl_usd",
    entry_premium_col: str = "entry_premium_usd",
    cost_col: str = "estimated_transaction_cost_usd",
    split_col: str = "split",
    validation_split: str = "validation",
    threshold_multipliers: Sequence[float] = DEFAULT_STRATEGY_THRESHOLD_MULTIPLIERS,
    min_edge_vars: Sequence[float] = DEFAULT_STRATEGY_MIN_EDGE_VARS,
    top_k_values: Sequence[int | None] = DEFAULT_STRATEGY_TOP_K_VALUES,
    min_validation_trades: int = DEFAULT_STRATEGY_MIN_VALIDATION_TRADES,
    drawdown_penalty: float = DEFAULT_STRATEGY_DRAWDOWN_PENALTY,
) -> tuple[StrategyPolicy, pd.DataFrame]:
    """Select a strategy policy using validation rows only.

    Forecast/ranking models remain independent of this layer. The selected policy can
    then be applied to locked-test rows without inspecting locked-test PnL.
    """
    if min_validation_trades < 0:
        raise ValueError("min_validation_trades must be nonnegative.")
    if drawdown_penalty < 0:
        raise ValueError("drawdown_penalty must be nonnegative.")
    if split_col not in frame.columns:
        validation = frame.copy()
    else:
        validation = frame.loc[frame[split_col].astype(str).eq(validation_split)].copy()
    if validation.empty:
        policy = StrategyPolicy(
            selection_split=validation_split,
            quote_filter_status="no_validation_rows",
        )
        return policy, pd.DataFrame([{**strategy_policy_to_dict(policy), "selected": True}])

    liquidity_modes = _liquidity_modes(validation)
    dte_modes = _dte_modes(validation)
    quote_modes = _quote_modes(validation)
    filter_modes: list[tuple[tuple[str, ...], dict[str, object], dict[str, object], pd.Series]] = []
    for liquidity in liquidity_modes:
        for dte_mode in dte_modes:
            for quote_mode in quote_modes:
                filter_policy = StrategyPolicy(
                    allowed_liquidity_buckets=liquidity,
                    require_main_dte_5_14=cast(bool | None, dte_mode["require_main_dte_5_14"]),
                    allowed_dte_buckets=cast(tuple[str, ...], dte_mode["allowed_dte_buckets"]),
                    allowed_execution_confidence_bands=cast(
                        tuple[str, ...],
                        quote_mode["allowed_execution_confidence_bands"],
                    ),
                    min_execution_confidence_score=cast(
                        float | None,
                        quote_mode["min_execution_confidence_score"],
                    ),
                    max_median_spread_over_mid=cast(
                        float | None,
                        quote_mode["max_median_spread_over_mid"],
                    ),
                    max_quote_age_seconds=cast(
                        float | None,
                        quote_mode["max_quote_age_seconds"],
                    ),
                    quote_filter_status=str(quote_mode["quote_filter_status"]),
                    selection_split=validation_split,
                )
                filter_modes.append(
                    (
                        liquidity,
                        dte_mode,
                        quote_mode,
                        _policy_bool_filter(validation, filter_policy),
                    )
                )
    rows: list[dict[str, object]] = []
    policies: dict[int, StrategyPolicy] = {}
    candidate_id = 0
    for threshold in threshold_multipliers:
        if float(threshold) <= 0:
            continue
        for min_edge in min_edge_vars:
            try:
                base = build_proxy_strategy_frame(
                    validation,
                    forecast_col=forecast_col,
                    ivar_col=ivar_col,
                    realized_long_pnl_col=realized_long_pnl_col,
                    entry_premium_col=entry_premium_col,
                    cost_col=cost_col,
                    min_edge_var=float(min_edge),
                    threshold_multiplier=float(threshold),
                )
            except ValueError:
                continue
            base_signal = base["should_trade"].fillna(False).astype(bool)
            effective_selection_score_col = (
                selection_score_col
                if selection_score_col is not None and selection_score_col in base.columns
                else "expected_strategy_edge_usd"
            )
            scores = pd.to_numeric(base[effective_selection_score_col], errors="coerce")
            for top_k in top_k_values:
                for liquidity, dte_mode, quote_mode, filter_pass in filter_modes:
                    policy = StrategyPolicy(
                        threshold_multiplier=float(threshold),
                        min_edge_var=float(min_edge),
                        top_k=top_k,
                        allowed_liquidity_buckets=liquidity,
                        require_main_dte_5_14=cast(bool | None, dte_mode["require_main_dte_5_14"]),
                        allowed_dte_buckets=cast(tuple[str, ...], dte_mode["allowed_dte_buckets"]),
                        allowed_execution_confidence_bands=cast(
                            tuple[str, ...],
                            quote_mode["allowed_execution_confidence_bands"],
                        ),
                        min_execution_confidence_score=cast(
                            float | None, quote_mode["min_execution_confidence_score"]
                        ),
                        max_median_spread_over_mid=cast(
                            float | None, quote_mode["max_median_spread_over_mid"]
                        ),
                        max_quote_age_seconds=cast(
                            float | None, quote_mode["max_quote_age_seconds"]
                        ),
                        quote_filter_status=str(quote_mode["quote_filter_status"]),
                        selection_split=validation_split,
                        selection_score_col=effective_selection_score_col,
                    )
                    candidate = base_signal & filter_pass
                    selected = _top_k_single_group_mask(
                        candidate=candidate,
                        scores=scores,
                        top_k=policy.top_k,
                    )
                    objective = _strategy_objective_from_mask(
                        base,
                        selected,
                        min_validation_trades=min_validation_trades,
                        drawdown_penalty=drawdown_penalty,
                    )
                    policies[candidate_id] = policy
                    rows.append(
                        {
                            "candidate_id": candidate_id,
                            **strategy_policy_to_dict(policy),
                            "selected": False,
                            "validation_min_trades_required": int(min_validation_trades),
                            "validation_drawdown_penalty": float(drawdown_penalty),
                            **{f"validation_{key}": value for key, value in objective.items()},
                        }
                    )
                    candidate_id += 1
    search = pd.DataFrame(rows)
    if search.empty:
        policy = StrategyPolicy(
            selection_split=validation_split,
            quote_filter_status="no_valid_candidates",
        )
        return policy, pd.DataFrame([{**strategy_policy_to_dict(policy), "selected": True}])
    finite = search.loc[np.isfinite(pd.to_numeric(search["validation_objective_value"]))]
    eligible = finite.loc[finite["validation_meets_min_validation_trades"].astype(bool)]
    pool = eligible if not eligible.empty else finite
    if pool.empty:
        pool = search
    best = pool.sort_values(
        ["validation_objective_value", "validation_net_pnl_usd", "validation_n"],
        ascending=[False, False, False],
        kind="mergesort",
    ).iloc[0]
    selected_id = int(best["candidate_id"])
    search.loc[search["candidate_id"].eq(selected_id), "selected"] = True
    return policies[selected_id], search


def integer_contract_count(*, target_max_loss_usd: float, max_loss_per_contract_usd: float) -> int:
    if max_loss_per_contract_usd <= 0:
        raise ValueError("max_loss_per_contract_usd must be positive")
    return int(np.floor(target_max_loss_usd / max_loss_per_contract_usd))


def apply_portfolio_caps(
    trades: Sequence[StrategyTrade],
    *,
    nav_usd: float,
    per_event_loss_fraction: float = 0.01,
    event_date_loss_fraction: float = 0.10,
    sector_event_date_loss_fraction: float = 0.03,
) -> list[StrategyTrade]:
    if nav_usd <= 0:
        raise ValueError("nav_usd must be positive")
    capped: list[StrategyTrade] = []
    for trade in trades:
        max_loss = min(trade.max_theoretical_loss_usd, nav_usd * per_event_loss_fraction)
        scale = max_loss / trade.max_theoretical_loss_usd
        capped.append(_scale_trade(trade, scale))

    capped = _cap_groups(
        capped,
        nav_usd=nav_usd,
        cap_fraction=event_date_loss_fraction,
        key=lambda trade: str(trade.event_date),
    )
    capped = _cap_groups(
        capped,
        nav_usd=nav_usd,
        cap_fraction=sector_event_date_loss_fraction,
        key=lambda trade: f"{trade.event_date}|{trade.sector or 'UNKNOWN'}",
    )
    return capped


def _cap_groups(
    trades: Sequence[StrategyTrade],
    *,
    nav_usd: float,
    cap_fraction: float,
    key: Callable[[StrategyTrade], str],
) -> list[StrategyTrade]:
    out = list(trades)
    groups: dict[str, list[int]] = {}
    for idx, trade in enumerate(out):
        groups.setdefault(str(key(trade)), []).append(idx)
    for indices in groups.values():
        group_loss = sum(out[idx].max_theoretical_loss_usd for idx in indices)
        cap = nav_usd * cap_fraction
        if group_loss <= cap or group_loss <= 0:
            continue
        positive_edges = [max(out[idx].expected_net_edge_usd, 0.0) for idx in indices]
        edge_total = sum(positive_edges)
        if edge_total <= 0:
            scale = cap / group_loss
            for idx in indices:
                out[idx] = _scale_trade(out[idx], scale)
            continue
        for idx, edge in zip(indices, positive_edges, strict=True):
            current_loss = out[idx].max_theoretical_loss_usd
            if current_loss <= 0:
                continue
            allocated_loss = cap * edge / edge_total
            scale = min(1.0, allocated_loss / current_loss)
            out[idx] = _scale_trade(out[idx], scale)
    return out


def _scale_trade(trade: StrategyTrade, scale: float) -> StrategyTrade:
    scaled_legs = tuple(
        leg.model_copy(update={"contracts": float(leg.contracts) * scale}) for leg in trade.legs
    )
    return trade.model_copy(
        update={
            "max_theoretical_loss_usd": trade.max_theoretical_loss_usd * scale,
            "expected_net_edge_usd": trade.expected_net_edge_usd * scale,
            "legs": scaled_legs,
        }
    )

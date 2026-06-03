from __future__ import annotations

from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from typing import Any, cast

import numpy as np
import pandas as pd

EPSILON = 1e-12
QLIKE_FLOOR = 1e-6


@dataclass(frozen=True)
class EvaluationBundle:
    forecast: dict[str, float | int | None]
    ranking: dict[str, float | int | None]
    strategy: dict[str, float | int | None]
    breakdowns: dict[str, pd.DataFrame]


def _finite_pair(frame: pd.DataFrame, left: str, right: str) -> pd.DataFrame:
    if left not in frame.columns or right not in frame.columns:
        raise ValueError(f"frame must include {left} and {right}")
    out = frame[[left, right]].copy()
    out[left] = pd.to_numeric(out[left], errors="coerce")
    out[right] = pd.to_numeric(out[right], errors="coerce")
    return out.loc[np.isfinite(out[left]) & np.isfinite(out[right])].copy()


def qlike_loss(actual: Sequence[float], forecast: Sequence[float]) -> float:
    y = np.maximum(np.asarray(actual, dtype=float), QLIKE_FLOOR)
    f = np.maximum(np.asarray(forecast, dtype=float), QLIKE_FLOOR)
    ratio = y / f
    return float(np.mean(ratio - np.log(ratio) - 1.0))


def forecast_metrics(
    frame: pd.DataFrame,
    *,
    forecast_col: str,
    target_col: str = "rvar_event",
    baseline_col: str = "ivar_event",
) -> dict[str, float | int | None]:
    clean = _finite_pair(frame, forecast_col, target_col)
    forecast_target_n = int(len(clean))
    if baseline_col in frame.columns and not clean.empty:
        clean[baseline_col] = frame.loc[clean.index, baseline_col].pipe(
            pd.to_numeric, errors="coerce"
        )
        clean = clean.loc[np.isfinite(clean[baseline_col])].copy()
    if clean.empty:
        return {
            "n": 0,
            "n_forecast_target": forecast_target_n,
            "n_oos_r2": 0,
            "mae": None,
            "rmse": None,
            "qlike": None,
            "oos_r2_vs_ivar": None,
        }
    error = clean[forecast_col] - clean[target_col]
    result: dict[str, float | int | None] = {
        "n": int(len(clean)),
        "n_forecast_target": forecast_target_n,
        "n_oos_r2": int(len(clean)) if baseline_col in clean.columns else 0,
        "mae": float(error.abs().mean()),
        "rmse": float(np.sqrt(np.mean(np.square(error)))),
        "qlike": qlike_loss(clean[target_col], clean[forecast_col]),
        "oos_r2_vs_ivar": None,
    }
    if baseline_col in clean.columns:
        y = clean[target_col]
        f = clean[forecast_col]
        b = clean[baseline_col]
        denominator = float(np.sum(np.square(y - b)))
        result["oos_r2_vs_ivar"] = (
            None if denominator <= EPSILON else float(1.0 - np.sum(np.square(y - f)) / denominator)
        )
    return result


def _rank_probability(scores: pd.Series) -> pd.Series:
    ranks = scores.rank(method="average", pct=True)
    return ranks.clip(lower=EPSILON, upper=1.0 - EPSILON)


def auc_score(outcome: Sequence[bool | int | float], score: Sequence[float]) -> float | None:
    y = np.asarray(outcome, dtype=float)
    s = np.asarray(score, dtype=float)
    valid = np.isfinite(y) & np.isfinite(s)
    y = y[valid] > 0
    s = s[valid]
    positives = int(np.sum(y))
    negatives = int(len(y) - positives)
    if positives == 0 or negatives == 0:
        return None
    ranks = pd.Series(s).rank(method="average").to_numpy()
    pos_rank_sum = float(np.sum(ranks[y]))
    return float((pos_rank_sum - positives * (positives + 1) / 2) / (positives * negatives))


def brier_score(
    outcome: Sequence[bool | int | float], probability: Sequence[float]
) -> float | None:
    y = np.asarray(outcome, dtype=float)
    p = np.asarray(probability, dtype=float)
    valid = np.isfinite(y) & np.isfinite(p)
    if not bool(valid.any()):
        return None
    y = (y[valid] > 0).astype(float)
    p = np.clip(p[valid], EPSILON, 1.0 - EPSILON)
    return float(np.mean(np.square(p - y)))


def calibration_table(
    frame: pd.DataFrame,
    *,
    score_col: str,
    outcome_col: str,
    bins: int = 10,
) -> pd.DataFrame:
    if score_col not in frame.columns or outcome_col not in frame.columns:
        raise ValueError(f"frame must include {score_col} and {outcome_col}")
    clean = frame[[score_col, outcome_col]].copy()
    clean[score_col] = pd.to_numeric(clean[score_col], errors="coerce")
    clean[outcome_col] = pd.to_numeric(clean[outcome_col], errors="coerce")
    clean = clean.dropna()
    if clean.empty:
        return pd.DataFrame(columns=["bucket", "n", "mean_score", "event_rate"])
    q = min(bins, len(clean))
    clean["bucket"] = pd.qcut(
        clean[score_col].rank(method="first"),
        q=q,
        labels=False,
        duplicates="drop",
    )
    return (
        clean.groupby("bucket", dropna=False)
        .agg(
            n=(outcome_col, "size"),
            mean_score=(score_col, "mean"),
            event_rate=(outcome_col, "mean"),
        )
        .reset_index()
    )


def ranking_metrics(
    frame: pd.DataFrame,
    *,
    score_col: str,
    realized_edge_col: str = "edge_var_realized",
    bins: int = 10,
) -> dict[str, float | int | None]:
    if score_col not in frame.columns or realized_edge_col not in frame.columns:
        raise ValueError(f"frame must include {score_col} and {realized_edge_col}")
    clean = frame[[score_col, realized_edge_col]].copy()
    clean[score_col] = pd.to_numeric(clean[score_col], errors="coerce")
    clean[realized_edge_col] = pd.to_numeric(clean[realized_edge_col], errors="coerce")
    clean = clean.loc[np.isfinite(clean[score_col]) & np.isfinite(clean[realized_edge_col])].copy()
    if clean.empty:
        return {
            "n": 0,
            "top_decile_precision": None,
            "auc": None,
            "rank_probability_brier": None,
            "edge_decile_spearman": None,
            "edge_decile_adjacent_up_share": None,
        }
    outcome = clean[realized_edge_col].gt(0)
    probability = _rank_probability(clean[score_col])
    top_count = max(1, int(np.ceil(len(clean) / bins)))
    if clean[score_col].nunique(dropna=True) <= 1:
        top = clean
    else:
        top = clean.sort_values(score_col, ascending=False).head(top_count)
    deciles = edge_decile_table(
        clean, score_col=score_col, realized_edge_col=realized_edge_col, bins=bins
    )
    means = deciles["mean_realized_edge"].to_numpy(dtype=float)
    adjacent = np.diff(means)
    up_share = None if len(adjacent) == 0 else float(np.mean(adjacent >= 0))
    spearman = None
    if len(deciles) > 1 and float(np.std(means)) > EPSILON:
        spearman = float(
            pd.Series(deciles["edge_decile"]).corr(pd.Series(means), method="spearman")
        )
    return {
        "n": int(len(clean)),
        "top_decile_precision": float(top[realized_edge_col].gt(0).mean()),
        "auc": auc_score(outcome, clean[score_col]),
        "rank_probability_brier": brier_score(outcome, probability),
        "edge_decile_spearman": spearman,
        "edge_decile_adjacent_up_share": up_share,
    }


def edge_decile_table(
    frame: pd.DataFrame,
    *,
    score_col: str,
    realized_edge_col: str = "edge_var_realized",
    bins: int = 10,
) -> pd.DataFrame:
    clean = frame[[score_col, realized_edge_col]].copy()
    clean[score_col] = pd.to_numeric(clean[score_col], errors="coerce")
    clean[realized_edge_col] = pd.to_numeric(clean[realized_edge_col], errors="coerce")
    clean = clean.loc[np.isfinite(clean[score_col]) & np.isfinite(clean[realized_edge_col])].copy()
    if clean.empty:
        return pd.DataFrame(columns=["edge_decile", "n", "mean_score", "mean_realized_edge"])
    if clean[score_col].nunique(dropna=True) <= 1:
        clean["edge_decile"] = 0
        return (
            clean.groupby("edge_decile", dropna=False)
            .agg(
                n=(realized_edge_col, "size"),
                mean_score=(score_col, "mean"),
                mean_realized_edge=(realized_edge_col, "mean"),
            )
            .reset_index()
        )
    q = min(bins, len(clean))
    clean["edge_decile"] = pd.qcut(
        clean[score_col].rank(method="average"),
        q=q,
        labels=False,
        duplicates="drop",
    )
    return (
        clean.groupby("edge_decile", dropna=False)
        .agg(
            n=(realized_edge_col, "size"),
            mean_score=(score_col, "mean"),
            mean_realized_edge=(realized_edge_col, "mean"),
        )
        .reset_index()
        .sort_values("edge_decile")
    )


def max_drawdown(values: Sequence[float]) -> float:
    pnl = np.asarray(values, dtype=float)
    if pnl.size == 0:
        return 0.0
    curve = np.cumsum(np.nan_to_num(pnl, nan=0.0))
    peak = np.maximum.accumulate(curve)
    return float(np.min(curve - peak))


def strategy_metrics(
    frame: pd.DataFrame,
    *,
    net_pnl_col: str = "net_proxy_pnl_usd",
    gross_pnl_col: str = "gross_strategy_pnl_usd",
    premium_col: str = "entry_premium_usd",
    capital_col: str | None = None,
    date_col: str = "event_date",
) -> dict[str, float | int | None]:
    if net_pnl_col not in frame.columns:
        raise ValueError(f"frame must include {net_pnl_col}")
    out = frame.copy()
    if date_col in out.columns:
        out = out.sort_values(date_col)
    net_all = pd.to_numeric(out[net_pnl_col], errors="coerce")
    valid = np.isfinite(net_all)
    out = out.loc[valid].copy()
    net = net_all.loc[valid]
    if net.empty:
        return {
            "n": 0,
            "gross_pnl_usd": None,
            "net_pnl_usd": None,
            "return_on_premium": None,
            "return_on_capital": None,
            "sharpe": None,
            "sortino": None,
            "max_drawdown_usd": None,
            "hit_rate": None,
            "avg_win_usd": None,
            "avg_loss_usd": None,
            "tail_loss_5pct_usd": None,
            "turnover": 0,
        }
    resolved_gross_col = (
        gross_pnl_col
        if gross_pnl_col in out.columns
        else "gross_proxy_pnl_usd"
        if "gross_proxy_pnl_usd" in out.columns
        else ""
    )
    gross = pd.to_numeric(out[resolved_gross_col], errors="coerce") if resolved_gross_col else net
    premium = (
        pd.to_numeric(out[premium_col], errors="coerce").abs().sum()
        if premium_col in out.columns
        else np.nan
    )
    capital = (
        pd.to_numeric(out[capital_col], errors="coerce").abs().sum()
        if capital_col and capital_col in out.columns
        else premium
    )
    losses = net.loc[net < 0]
    wins = net.loc[net > 0]
    std = float(net.std(ddof=1)) if len(net) > 1 else 0.0
    downside = float(losses.std(ddof=1)) if len(losses) > 1 else 0.0
    return {
        "n": int(len(net)),
        "gross_pnl_usd": None if gross.dropna().empty else float(gross.sum()),
        "net_pnl_usd": float(net.sum()),
        "return_on_premium": None
        if not np.isfinite(premium) or premium <= EPSILON
        else float(net.sum() / premium),
        "return_on_capital": None
        if not np.isfinite(capital) or capital <= EPSILON
        else float(net.sum() / capital),
        "sharpe": None if std <= EPSILON else float(net.mean() / std * np.sqrt(len(net))),
        "sortino": None
        if downside <= EPSILON
        else float(net.mean() / downside * np.sqrt(len(net))),
        "max_drawdown_usd": max_drawdown(net),
        "hit_rate": float(net.gt(0).mean()),
        "avg_win_usd": None if wins.empty else float(wins.mean()),
        "avg_loss_usd": None if losses.empty else float(losses.mean()),
        "tail_loss_5pct_usd": float(net.quantile(0.05)),
        "turnover": int(len(net)),
    }


def cost_sensitivity(
    frame: pd.DataFrame,
    *,
    gross_pnl_col: str = "gross_strategy_pnl_usd",
    cost_col: str = "estimated_transaction_cost_usd",
    multipliers: Iterable[float] = (0.0, 0.5, 1.0, 1.5, 2.0),
) -> pd.DataFrame:
    resolved_gross_col = (
        gross_pnl_col
        if gross_pnl_col in frame.columns
        else "gross_proxy_pnl_usd"
        if "gross_proxy_pnl_usd" in frame.columns
        else ""
    )
    if not resolved_gross_col or cost_col not in frame.columns:
        raise ValueError(f"frame must include {gross_pnl_col} and {cost_col}")
    rows: list[dict[str, float | int]] = []
    gross = pd.to_numeric(frame[resolved_gross_col], errors="coerce")
    cost = pd.to_numeric(frame[cost_col], errors="coerce")
    valid = np.isfinite(gross) & np.isfinite(cost)
    if "net_proxy_pnl_usd" in frame.columns:
        net_proxy = pd.to_numeric(frame["net_proxy_pnl_usd"], errors="coerce")
        valid &= np.isfinite(net_proxy)
    gross = gross.loc[valid]
    cost = cost.loc[valid]
    for multiplier in multipliers:
        net = gross - float(multiplier) * cost
        rows.append(
            {
                "cost_multiplier": float(multiplier),
                "n": int(len(net)),
                "net_pnl_usd": float(net.sum()),
                "hit_rate": float(net.gt(0).mean()) if len(net) else 0.0,
                "max_drawdown_usd": max_drawdown(net),
            }
        )
    return pd.DataFrame(rows)


def breakdown_metrics(
    frame: pd.DataFrame,
    *,
    by: Sequence[str],
    net_pnl_col: str = "net_proxy_pnl_usd",
    gross_pnl_col: str = "gross_strategy_pnl_usd",
    forecast_col: str | None = None,
    target_col: str = "rvar_event",
) -> pd.DataFrame:
    missing = [column for column in by if column not in frame.columns]
    if missing:
        raise ValueError(f"frame missing breakdown columns: {missing}")
    rows: list[dict[str, Any]] = []
    for keys, group in frame.groupby(list(by), dropna=False):
        key_values = keys if isinstance(keys, tuple) else (keys,)
        base = dict(zip(by, key_values, strict=True))
        pnl = (
            strategy_metrics(group, net_pnl_col=net_pnl_col, gross_pnl_col=gross_pnl_col)
            if net_pnl_col in group.columns
            else {}
        )
        forecast = (
            forecast_metrics(group, forecast_col=forecast_col, target_col=target_col)
            if forecast_col is not None
            and forecast_col in group.columns
            and target_col in group.columns
            else {}
        )
        rows.append({**base, **{f"strategy_{k}": v for k, v in pnl.items()}, **forecast})
    return pd.DataFrame(rows)


def evaluate_prediction_bundle(
    frame: pd.DataFrame,
    *,
    forecast_col: str,
    score_col: str | None = None,
    target_col: str = "rvar_event",
    baseline_col: str = "ivar_event",
    realized_edge_col: str = "edge_var_realized",
    breakdown_columns: Sequence[str] = (),
) -> EvaluationBundle:
    effective_score = score_col or forecast_col
    forecast = forecast_metrics(
        frame, forecast_col=forecast_col, target_col=target_col, baseline_col=baseline_col
    )
    ranking = cast(
        dict[str, float | int | None],
        ranking_metrics(frame, score_col=effective_score, realized_edge_col=realized_edge_col)
        if realized_edge_col in frame.columns
        else {"n": 0},
    )
    strategy = cast(
        dict[str, float | int | None],
        strategy_metrics(frame) if "net_proxy_pnl_usd" in frame.columns else {"n": 0},
    )
    breakdowns = {
        column: breakdown_metrics(
            frame,
            by=[column],
            forecast_col=forecast_col,
            net_pnl_col="net_proxy_pnl_usd",
        )
        for column in breakdown_columns
        if column in frame.columns
    }
    return EvaluationBundle(
        forecast=forecast, ranking=ranking, strategy=strategy, breakdowns=breakdowns
    )

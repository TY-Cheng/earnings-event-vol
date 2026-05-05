from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass
from datetime import date
from typing import Protocol

import numpy as np
from scipy.stats import norm

from earnings_event_vol.schemas import (
    OptionQuote,
    OptionRight,
    OptionSide,
    SignalRecord,
    StrategyTrade,
    TradeLeg,
)


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

from __future__ import annotations

from collections.abc import Iterable, Sequence
from datetime import date
from typing import cast


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

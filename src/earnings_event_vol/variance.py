from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from datetime import date, timedelta
from math import log

import pandas as pd

from earnings_event_vol.schemas import IVARFailureReason, TimeConvention

IVAR_FAILURE_REASONS: dict[str, str] = {
    IVARFailureReason.NO_TWO_EVENT_COVERING_EXPIRIES.value: (
        "Fewer than two expiries cover the event."
    ),
    IVARFailureReason.NONPOSITIVE_TIME_GAP.value: "The selected expiries do not have T2 > T1.",
    IVARFailureReason.NEGATIVE_EXTRACTED_IVAR.value: (
        "Two-expiry formula produced negative event variance."
    ),
    IVARFailureReason.STALE_OR_MISSING_IV.value: (
        "One or both selected IV observations are missing or stale."
    ),
    IVARFailureReason.NONPOSITIVE_TOTAL_VARIANCE.value: (
        "One or both total variances are nonpositive."
    ),
    IVARFailureReason.NONMONOTONE_TOTAL_VARIANCE.value: (
        "Later-expiry total variance is below earlier-expiry total variance."
    ),
}


@dataclass(frozen=True)
class TotalVariancePoint:
    expiration: date
    iv: float | None
    dte_days: int
    time_convention: TimeConvention = TimeConvention.ACT_365
    stale: bool = False
    moneyness: float | None = None
    spread_over_mid: float | None = None
    vix_regime: str | None = None


@dataclass(frozen=True)
class IVARExtractionResult:
    ivar_event: float | None
    failure_reason: IVARFailureReason | None
    t1: float | None = None
    t2: float | None = None
    w1: float | None = None
    w2: float | None = None
    expiration_gap_days: int | None = None
    iv_used_for_extraction_1: float | None = None
    iv_used_for_extraction_2: float | None = None
    dte_1: int | None = None
    dte_2: int | None = None
    expiration_1: date | None = None
    expiration_2: date | None = None
    spread_over_mid_1: float | None = None
    spread_over_mid_2: float | None = None

    @property
    def expiry_gap_days(self) -> int | None:
        return self.expiration_gap_days


def year_fraction(dte_days: int, convention: TimeConvention = TimeConvention.ACT_365) -> float:
    if dte_days <= 0:
        return 0.0
    if convention == TimeConvention.ACT_365:
        return dte_days / 365.0
    if convention == TimeConvention.TRADING_252:
        return dte_days / 252.0
    raise ValueError("Vendor time convention must be resolved before IVAR extraction.")


def realized_event_variance(s_before: float, s_after: float) -> float:
    if s_before <= 0 or s_after <= 0:
        raise ValueError("S_before and S_after must be positive.")
    return log(s_after / s_before) ** 2


def _result_with_selected_points(
    ivar_event: float | None,
    failure_reason: IVARFailureReason | None,
    *,
    first: TotalVariancePoint,
    second: TotalVariancePoint,
    t1: float | None = None,
    t2: float | None = None,
    w1: float | None = None,
    w2: float | None = None,
) -> IVARExtractionResult:
    return IVARExtractionResult(
        ivar_event,
        failure_reason,
        t1=t1,
        t2=t2,
        w1=w1,
        w2=w2,
        expiration_gap_days=(second.expiration - first.expiration).days,
        iv_used_for_extraction_1=first.iv,
        iv_used_for_extraction_2=second.iv,
        dte_1=first.dte_days,
        dte_2=second.dte_days,
        expiration_1=first.expiration,
        expiration_2=second.expiration,
        spread_over_mid_1=first.spread_over_mid,
        spread_over_mid_2=second.spread_over_mid,
    )


def extract_implied_event_variance(
    points: Sequence[TotalVariancePoint],
    *,
    event_date: date,
    event_exit_date: date | None = None,
    convention: TimeConvention = TimeConvention.ACT_365,
) -> IVARExtractionResult:
    """Extract scheduled-event variance from two expiries covering the realized event window.

    `event_exit_date` should be the close-to-close realized move end date. For AMC events this
    is normally the next trading day; for BMO events it is normally the announcement date. If a
    caller has not built event windows yet, the conservative fallback excludes same-date expiry.
    """
    if any(point.time_convention != convention for point in points):
        raise ValueError("Mixed time-to-expiry conventions are forbidden within one IVAR run.")
    coverage_date = event_exit_date or event_date + timedelta(days=1)
    covering = sorted(
        [point for point in points if point.expiration >= coverage_date],
        key=lambda point: point.expiration,
    )
    if len(covering) < 2:
        return IVARExtractionResult(None, IVARFailureReason.NO_TWO_EVENT_COVERING_EXPIRIES)
    first, second = covering[0], covering[1]
    if first.iv is None or second.iv is None or first.stale or second.stale:
        return _result_with_selected_points(
            None, IVARFailureReason.STALE_OR_MISSING_IV, first=first, second=second
        )
    t1 = year_fraction(first.dte_days, convention)
    t2 = year_fraction(second.dte_days, convention)
    if t2 <= t1 or t1 <= 0:
        return _result_with_selected_points(
            None,
            IVARFailureReason.NONPOSITIVE_TIME_GAP,
            first=first,
            second=second,
            t1=t1,
            t2=t2,
        )
    w1 = first.iv**2 * t1
    w2 = second.iv**2 * t2
    if w1 <= 0 or w2 <= 0:
        return _result_with_selected_points(
            None,
            IVARFailureReason.NONPOSITIVE_TOTAL_VARIANCE,
            first=first,
            second=second,
            t1=t1,
            t2=t2,
            w1=w1,
            w2=w2,
        )
    if w2 < w1:
        return _result_with_selected_points(
            None,
            IVARFailureReason.NONMONOTONE_TOTAL_VARIANCE,
            first=first,
            second=second,
            t1=t1,
            t2=t2,
            w1=w1,
            w2=w2,
        )
    ivar = (t2 * w1 - t1 * w2) / (t2 - t1)
    if ivar < 0:
        return _result_with_selected_points(
            None,
            IVARFailureReason.NEGATIVE_EXTRACTED_IVAR,
            first=first,
            second=second,
            t1=t1,
            t2=t2,
            w1=w1,
            w2=w2,
        )
    return _result_with_selected_points(
        ivar, None, first=first, second=second, t1=t1, t2=t2, w1=w1, w2=w2
    )


def edge_variance(forecast_rvar_event: float, ivar_event: float) -> float:
    return float(forecast_rvar_event - ivar_event)


def negative_ivar_diagnostics(rows: Sequence[dict[str, object]]) -> pd.DataFrame:
    columns = [
        "ticker",
        "event_date",
        "expiration_1",
        "expiration_2",
        "dte_1",
        "dte_2",
        "expiry_gap_days",
        "expiration_gap_days",
        "iv_used_for_extraction_1",
        "iv_used_for_extraction_2",
        "spread_over_mid_1",
        "spread_over_mid_2",
        "moneyness",
        "spread_over_mid",
        "iv_1",
        "iv_2",
        "w1",
        "w2",
        "failure_reason",
        "vix_regime",
    ]
    return pd.DataFrame(list(rows), columns=columns)

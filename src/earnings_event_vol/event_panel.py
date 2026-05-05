from __future__ import annotations

from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from datetime import date
from math import isfinite
from typing import cast

import pandas as pd

from earnings_event_vol.events import has_ex_dividend_between

CONTRACT_STATUS_OK = "ok"
CONTRACT_STATUS_NON_STANDARD_EXCLUDED = "non_standard_excluded"
CONTRACT_STATUS_OUTSIDE_DTE_RANGE = "outside_dte_range"
CONTRACT_STATUS_DOES_NOT_COVER_EVENT = "does_not_cover_event_window"
CONTRACT_STATUS_MISSING_METADATA = "missing_contract_metadata"

FORWARD_SOURCE_PUT_CALL_PARITY = "put_call_parity"
FORWARD_SOURCE_SPOT_FALLBACK = "spot_fallback"
ATM_METHOD_ATMF = "ATMF"
ATM_METHOD_NEAREST_SPOT = "nearest_spot_atm"


@dataclass(frozen=True)
class ForwardSelection:
    forward_source: str
    forward_price: float
    atm_selection_method: str
    american_forward_caveat_flag: bool


def _require_columns(frame: pd.DataFrame, required: Iterable[str], *, name: str) -> None:
    missing = sorted(set(required) - set(frame.columns))
    if missing:
        raise ValueError(f"{name} missing required columns: {missing}")


def _date_series(frame: pd.DataFrame, column: str) -> pd.Series:
    return pd.to_datetime(frame[column], errors="coerce").dt.date


def _to_date(value: object) -> date | None:
    if value is None:
        return None
    try:
        if bool(pd.isna(value)):
            return None
    except TypeError:
        pass
    return cast(date, pd.Timestamp(value).date())


def _normalize_deliverable(value: object) -> str:
    if value is None:
        return "standard"
    try:
        if bool(pd.isna(value)):
            return "standard"
    except TypeError:
        pass
    text = str(value).strip().lower().replace("-", "_").replace(" ", "_")
    if text in {"", "nan", "none", "standard", "regular", "normal", "100_share", "100_shares"}:
        return "standard"
    return text


def _boolean_flag(value: object) -> bool:
    if isinstance(value, bool):
        return value
    if value is None:
        return False
    try:
        if bool(pd.isna(value)):
            return False
    except TypeError:
        pass
    return str(value).strip().lower() in {"1", "true", "yes", "y", "corporate_action"}


def _default_event_ids(events: pd.DataFrame) -> pd.Series:
    if "event_id" in events.columns:
        return events["event_id"].astype(str)
    return events["ticker"].astype(str).str.upper() + "_" + events["entry_date"].astype(str)


def discover_option_contracts(
    events: pd.DataFrame,
    contracts: pd.DataFrame,
    *,
    dte_min: int = 5,
    dte_max: int = 14,
) -> pd.DataFrame:
    """Map event rows to candidate option contracts and hard-filter OCC odd lots.

    Missing multiplier/deliverable metadata defaults to a standard 100-share equity option because
    Massive day-aggregate contract symbols do not always carry OCC deliverable details. When a
    reference source supplies different values, non-standard contracts are excluded before quotes
    enter the pool.
    """
    _require_columns(events, {"ticker", "entry_date"}, name="event frame")
    _require_columns(contracts, {"ticker", "expiration", "strike", "right"}, name="contract frame")
    if dte_min < 0 or dte_max < dte_min:
        raise ValueError("DTE range must satisfy 0 <= dte_min <= dte_max.")

    event_frame = events.copy()
    event_frame["ticker"] = event_frame["ticker"].astype(str).str.upper()
    event_frame["entry_date"] = _date_series(event_frame, "entry_date")
    if "exit_date" not in event_frame.columns:
        event_frame["exit_date"] = event_frame["entry_date"]
    else:
        event_frame["exit_date"] = _date_series(event_frame, "exit_date")
    event_frame["event_id"] = _default_event_ids(event_frame)

    contract_frame = contracts.copy()
    contract_frame["ticker"] = contract_frame["ticker"].astype(str).str.upper()
    contract_frame["expiration"] = _date_series(contract_frame, "expiration")
    contract_frame["strike"] = pd.to_numeric(contract_frame["strike"], errors="coerce")
    contract_frame["right"] = contract_frame["right"].astype(str).str.lower()
    if "options_ticker" not in contract_frame.columns:
        if "option_symbol" in contract_frame.columns:
            contract_frame["options_ticker"] = contract_frame["option_symbol"]
        else:
            contract_frame["options_ticker"] = pd.NA
    if "option_multiplier" not in contract_frame.columns:
        contract_frame["option_multiplier"] = 100
    if "contract_size" not in contract_frame.columns:
        contract_frame["contract_size"] = contract_frame["option_multiplier"]
    if "deliverable_status" not in contract_frame.columns:
        contract_frame["deliverable_status"] = "standard"
    if "corporate_action_flag" not in contract_frame.columns:
        contract_frame["corporate_action_flag"] = False

    out = event_frame.merge(contract_frame, on="ticker", how="left", suffixes=("", "_contract"))
    out["option_multiplier"] = pd.to_numeric(out["option_multiplier"], errors="coerce").fillna(100)
    out["contract_size"] = pd.to_numeric(out["contract_size"], errors="coerce").fillna(
        out["option_multiplier"]
    )
    out["deliverable_status"] = out["deliverable_status"].map(_normalize_deliverable)
    out["corporate_action_flag"] = out["corporate_action_flag"].map(_boolean_flag)
    out["dte"] = (
        pd.to_datetime(out["expiration"], errors="coerce")
        - pd.to_datetime(out["entry_date"], errors="coerce")
    ).dt.days
    out["covers_event_window"] = out["expiration"] >= out["exit_date"]

    has_contract = out["expiration"].notna() & out["strike"].notna() & out["right"].notna()
    is_standard = (
        out["option_multiplier"].eq(100)
        & out["contract_size"].eq(100)
        & out["deliverable_status"].eq("standard")
        & ~out["corporate_action_flag"]
    )
    in_dte = out["dte"].between(dte_min, dte_max, inclusive="both")
    covers = out["covers_event_window"].fillna(False)

    out["contract_discovery_status"] = CONTRACT_STATUS_OK
    out.loc[~has_contract, "contract_discovery_status"] = CONTRACT_STATUS_MISSING_METADATA
    out.loc[has_contract & ~is_standard, "contract_discovery_status"] = (
        CONTRACT_STATUS_NON_STANDARD_EXCLUDED
    )
    out.loc[has_contract & is_standard & ~in_dte, "contract_discovery_status"] = (
        CONTRACT_STATUS_OUTSIDE_DTE_RANGE
    )
    out.loc[has_contract & is_standard & in_dte & ~covers, "contract_discovery_status"] = (
        CONTRACT_STATUS_DOES_NOT_COVER_EVENT
    )
    out["eligible_for_quote_pool"] = out["contract_discovery_status"].eq(CONTRACT_STATUS_OK)

    preferred = [
        "event_id",
        "ticker",
        "entry_date",
        "exit_date",
        "expiration",
        "strike",
        "right",
        "options_ticker",
        "dte",
        "covers_event_window",
        "option_multiplier",
        "contract_size",
        "deliverable_status",
        "corporate_action_flag",
        "contract_discovery_status",
        "eligible_for_quote_pool",
    ]
    rest = [column for column in out.columns if column not in preferred]
    return out[preferred + rest]


def _strict_quote_frame(
    quotes: pd.DataFrame,
    *,
    entry_date: date,
    spot: float,
    dte_min: int,
    dte_max: int,
    max_spread_over_mid: float,
    atm_moneyness_tolerance: float,
) -> pd.DataFrame:
    _require_columns(quotes, {"expiration", "strike", "right", "bid", "ask"}, name="quote frame")
    out = quotes.copy()
    out["expiration"] = _date_series(out, "expiration")
    out["strike"] = pd.to_numeric(out["strike"], errors="coerce")
    out["right"] = out["right"].astype(str).str.lower()
    out["bid"] = pd.to_numeric(out["bid"], errors="coerce")
    out["ask"] = pd.to_numeric(out["ask"], errors="coerce")
    out["mid"] = (out["bid"] + out["ask"]) / 2.0
    if "spread_over_mid" in out.columns:
        out["spread_over_mid"] = pd.to_numeric(out["spread_over_mid"], errors="coerce")
    else:
        out["spread_over_mid"] = (out["ask"] - out["bid"]) / out["mid"]
    out["dte"] = (pd.to_datetime(out["expiration"]) - pd.Timestamp(entry_date)).dt.days
    out["moneyness_abs"] = (out["strike"] / spot - 1.0).abs()
    return out.loc[
        (out["dte"].between(dte_min, dte_max, inclusive="both"))
        & out["bid"].gt(0)
        & out["ask"].gt(out["bid"])
        & out["mid"].gt(0)
        & out["spread_over_mid"].le(max_spread_over_mid)
        & out["moneyness_abs"].le(atm_moneyness_tolerance)
    ].copy()


def select_forward_and_atm(
    quotes: pd.DataFrame,
    *,
    entry_date: date,
    spot: float,
    second_ivar_expiry: date,
    ex_dividend_dates: Sequence[date] = (),
    dte_min: int = 5,
    dte_max: int = 14,
    max_spread_over_mid: float = 0.30,
    atm_moneyness_tolerance: float = 0.10,
) -> ForwardSelection:
    """Select a forward/ATM convention for American single-name options.

    Put-call parity is used only as a short-DTE, no-dividend, near-ATM approximation. If the
    required pair is weak or the dividend window is dirty, v1 falls back to nearest spot ATM and
    records the fallback source.
    """
    if spot <= 0:
        raise ValueError("spot must be positive.")
    if has_ex_dividend_between(ex_dividend_dates, start=entry_date, end=second_ivar_expiry):
        return ForwardSelection(
            forward_source=FORWARD_SOURCE_SPOT_FALLBACK,
            forward_price=float(spot),
            atm_selection_method=ATM_METHOD_NEAREST_SPOT,
            american_forward_caveat_flag=False,
        )
    if quotes.empty:
        return ForwardSelection(
            forward_source=FORWARD_SOURCE_SPOT_FALLBACK,
            forward_price=float(spot),
            atm_selection_method=ATM_METHOD_NEAREST_SPOT,
            american_forward_caveat_flag=False,
        )

    eligible = _strict_quote_frame(
        quotes,
        entry_date=entry_date,
        spot=spot,
        dte_min=dte_min,
        dte_max=dte_max,
        max_spread_over_mid=max_spread_over_mid,
        atm_moneyness_tolerance=atm_moneyness_tolerance,
    )
    if eligible.empty:
        return ForwardSelection(
            forward_source=FORWARD_SOURCE_SPOT_FALLBACK,
            forward_price=float(spot),
            atm_selection_method=ATM_METHOD_NEAREST_SPOT,
            american_forward_caveat_flag=False,
        )

    calls = eligible.loc[eligible["right"].isin(["call", "c"])].copy()
    puts = eligible.loc[eligible["right"].isin(["put", "p"])].copy()
    pairs = calls.merge(
        puts,
        on=["expiration", "strike"],
        suffixes=("_call", "_put"),
    )
    if pairs.empty:
        return ForwardSelection(
            forward_source=FORWARD_SOURCE_SPOT_FALLBACK,
            forward_price=float(spot),
            atm_selection_method=ATM_METHOD_NEAREST_SPOT,
            american_forward_caveat_flag=False,
        )
    pairs["distance"] = (pairs["strike"] / spot - 1.0).abs()
    pairs = pairs.sort_values(["distance", "spread_over_mid_call", "spread_over_mid_put"])
    best = pairs.iloc[0]
    forward = float(best["strike"] + best["mid_call"] - best["mid_put"])
    if not isfinite(forward) or forward <= 0:
        return ForwardSelection(
            forward_source=FORWARD_SOURCE_SPOT_FALLBACK,
            forward_price=float(spot),
            atm_selection_method=ATM_METHOD_NEAREST_SPOT,
            american_forward_caveat_flag=False,
        )
    return ForwardSelection(
        forward_source=FORWARD_SOURCE_PUT_CALL_PARITY,
        forward_price=forward,
        atm_selection_method=ATM_METHOD_ATMF,
        american_forward_caveat_flag=True,
    )


def flag_possible_preannouncement_or_prior_guidance(
    frame: pd.DataFrame,
    *,
    high_ivar_quantile: float = 0.75,
    max_rvar_to_ivar_ratio: float = 0.25,
) -> pd.DataFrame:
    """Flag events where realized movement is very small relative to rich event IV."""
    _require_columns(frame, {"rvar_event", "ivar_event"}, name="event panel")
    if not 0 <= high_ivar_quantile <= 1:
        raise ValueError("high_ivar_quantile must be in [0, 1].")
    out = frame.copy()
    ivar = pd.to_numeric(out["ivar_event"], errors="coerce")
    rvar = pd.to_numeric(out["rvar_event"], errors="coerce")
    valid_ivar = ivar.loc[ivar.gt(0)]
    high_threshold = float(valid_ivar.quantile(high_ivar_quantile)) if not valid_ivar.empty else 0.0
    realized_edge = rvar - ivar
    out["possible_preannouncement_or_prior_guidance"] = (
        ivar.gt(0)
        & ivar.ge(high_threshold)
        & rvar.ge(0)
        & rvar.le(ivar * max_rvar_to_ivar_ratio)
        & realized_edge.lt(0)
    )
    return out


def _event_ex_dividends(
    ex_dividends: pd.DataFrame | None,
    *,
    ticker: str,
) -> list[date]:
    if ex_dividends is None or ex_dividends.empty:
        return []
    _require_columns(ex_dividends, {"ticker", "ex_dividend_date"}, name="ex-dividend frame")
    frame = ex_dividends.copy()
    frame["ticker"] = frame["ticker"].astype(str).str.upper()
    frame["ex_dividend_date"] = _date_series(frame, "ex_dividend_date")
    return [
        value
        for value in frame.loc[frame["ticker"].eq(ticker.upper()), "ex_dividend_date"].tolist()
        if isinstance(value, date)
    ]


def _second_expiry_for_event(event: pd.Series, quotes: pd.DataFrame, *, entry_date: date) -> date:
    for column in ("expiration_2", "second_ivar_expiry", "second_expiry"):
        if column in event.index:
            parsed = _to_date(event[column])
            if parsed is not None:
                return parsed
    if quotes.empty:
        return entry_date
    expiries = sorted(
        {
            parsed
            for parsed in (_to_date(value) for value in quotes["expiration"].tolist())
            if parsed is not None
        }
    )
    return expiries[1] if len(expiries) >= 2 else expiries[0]


def build_event_panel(
    events: pd.DataFrame,
    quotes: pd.DataFrame,
    *,
    ex_dividends: pd.DataFrame | None = None,
    dte_min: int = 5,
    dte_max: int = 14,
) -> pd.DataFrame:
    """Attach v1 ATM/forward diagnostics and preannouncement-review flags to event rows."""
    _require_columns(events, {"ticker", "entry_date"}, name="event frame")
    out = events.copy()
    out["ticker"] = out["ticker"].astype(str).str.upper()
    out["entry_date"] = _date_series(out, "entry_date")
    if "event_id" not in out.columns:
        out["event_id"] = _default_event_ids(out)
    if "spot" not in out.columns:
        if "s_before" in out.columns:
            out["spot"] = out["s_before"]
        elif "S_before" in out.columns:
            out["spot"] = out["S_before"]
        else:
            raise ValueError("event frame missing spot or s_before column.")

    quote_frame = quotes.copy()
    if not quote_frame.empty and "ticker" in quote_frame.columns:
        quote_frame["ticker"] = quote_frame["ticker"].astype(str).str.upper()
    selections: list[dict[str, object]] = []
    for idx, event in out.iterrows():
        ticker = str(event["ticker"]).upper()
        entry_date = _to_date(event["entry_date"])
        if entry_date is None:
            raise ValueError(f"missing entry_date for event row {idx}")
        event_quotes = quote_frame
        if "event_id" in quote_frame.columns:
            event_quotes = event_quotes.loc[
                quote_frame["event_id"].astype(str).eq(str(event["event_id"]))
            ]
        elif "ticker" in quote_frame.columns:
            event_quotes = event_quotes.loc[quote_frame["ticker"].eq(ticker)]
        second_expiry = _second_expiry_for_event(event, event_quotes, entry_date=entry_date)
        selection = select_forward_and_atm(
            event_quotes,
            entry_date=entry_date,
            spot=float(event["spot"]),
            second_ivar_expiry=second_expiry,
            ex_dividend_dates=_event_ex_dividends(ex_dividends, ticker=ticker),
            dte_min=dte_min,
            dte_max=dte_max,
        )
        selections.append(
            {
                "forward_source": selection.forward_source,
                "forward_price": selection.forward_price,
                "atm_selection_method": selection.atm_selection_method,
                "american_forward_caveat_flag": selection.american_forward_caveat_flag,
            }
        )
    selection_frame = pd.DataFrame(selections, index=out.index)
    out = pd.concat([out, selection_frame], axis=1)
    if {"rvar_event", "ivar_event"}.issubset(out.columns):
        out = flag_possible_preannouncement_or_prior_guidance(out)
    else:
        out["possible_preannouncement_or_prior_guidance"] = False
    return out

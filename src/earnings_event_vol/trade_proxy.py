from __future__ import annotations

import json
import urllib.parse
from dataclasses import dataclass
from datetime import date, datetime
from pathlib import Path
from typing import Any, cast

import httpx
import pandas as pd
from scipy.optimize import brentq

from earnings_event_vol.backtest import black_scholes_price
from earnings_event_vol.config import ProjectConfig
from earnings_event_vol.massive import read_secret_file
from earnings_event_vol.schemas import OptionRight
from earnings_event_vol.variance import (
    TotalVariancePoint,
    extract_implied_event_variance,
)

TRADE_PROXY_PANEL_GRADE = "no_nbbo_trade_proxy"
TRADE_PROXY_ROUTE_SECOND_AGGS = "massive_rest_second_aggs"
TRADE_PROXY_STATUS_OK = "ok"
TRADE_PROXY_STATUS_NO_TRADE_IN_WINDOW = "no_trade_in_cutoff_window"
TRADE_PROXY_STATUS_MISSING_CONTRACT = "missing_contract"
TRADE_PROXY_STATUS_FETCH_FAILED = "fetch_failed"
UNDERLYING_EXIT_PRICE_SOURCE = "day_aggs_close"
OPTION_EXIT_PRICE_SOURCE_DAY_AGGS = "options_day_aggs_close"
OPTION_EXIT_PAYOFF_FALLBACK_INTRINSIC = "intrinsic_value_at_underlying_exit"
OPTION_EXIT_STATUS_OK = "ok"
OPTION_EXIT_STATUS_MISSING_DAY_AGG = "missing_exit_option_close"
OPTION_EXIT_STATUS_EXPIRATION_AT_EXIT = "expiration_at_exit_intrinsic"


@dataclass(frozen=True)
class ProxyPriceSelection:
    status: str
    proxy_price: float | None = None
    proxy_timestamp: datetime | None = None
    proxy_age_seconds: float | None = None
    proxy_volume: int = 0
    proxy_transactions: int = 0
    proxy_rows_in_window: int = 0
    price_field: str = "option_vwap"


def _to_date(value: object) -> date:
    return cast(date, pd.Timestamp(value).date())


def _to_datetime(value: object) -> datetime:
    timestamp = pd.Timestamp(value)
    if timestamp.tzinfo is None:
        raise ValueError("timestamp must be timezone-aware")
    return cast(datetime, timestamp.to_pydatetime())


def _as_float(value: object) -> float:
    return float(cast(Any, value))


def _as_int(value: object) -> int:
    return int(cast(Any, value))


def _aware_et_timestamp(value: object, *, name: str) -> pd.Timestamp:
    timestamp = pd.Timestamp(value)
    if timestamp.tzinfo is None:
        raise ValueError(f"{name} must be timezone-aware")
    return timestamp.tz_convert("America/New_York")


def filter_pre_cutoff_buffer(
    bars: pd.DataFrame,
    *,
    cutoff_timestamp: datetime,
    buffer_minutes: int,
) -> pd.DataFrame:
    """Keep only bars in the pre-cutoff buffer used for entry pricing/features."""
    if buffer_minutes <= 0:
        raise ValueError("buffer_minutes must be positive.")
    if bars.empty:
        return bars.copy()
    if "timestamp_et" not in bars.columns:
        raise ValueError("bar frame missing timestamp_et column.")
    cutoff = _aware_et_timestamp(cutoff_timestamp, name="cutoff_timestamp")
    start = cutoff - pd.Timedelta(minutes=buffer_minutes)
    frame = bars.copy()
    frame["timestamp_et"] = pd.to_datetime(frame["timestamp_et"])
    if frame["timestamp_et"].dt.tz is None:
        raise ValueError("timestamp_et must be timezone-aware")
    frame["timestamp_et"] = frame["timestamp_et"].dt.tz_convert("America/New_York")
    return frame.loc[frame["timestamp_et"].between(start, cutoff, inclusive="both")].copy()


def _api_key(config: ProjectConfig) -> str:
    secret = read_secret_file(config.massive_api_key_file)
    if not secret:
        raise ValueError("MASSIVE_API_KEY_FILE is not configured or empty.")
    return secret


def fetch_massive_option_second_aggregates(
    config: ProjectConfig,
    *,
    option_ticker: str,
    trade_date: date,
    limit: int = 50_000,
    timeout_seconds: float | None = None,
) -> pd.DataFrame:
    """Fetch one option contract's historical 1-second trade aggregates from Massive REST.

    The returned bars are trade aggregates, not quotes. They contain no bid/ask or NBBO fields,
    so downstream panels are explicitly marked as no-NBBO trade proxies.
    """
    encoded_ticker = urllib.parse.quote(option_ticker, safe="")
    url = (
        f"{config.massive_base_url.rstrip('/')}/v2/aggs/ticker/{encoded_ticker}"
        f"/range/1/second/{trade_date.isoformat()}/{trade_date.isoformat()}"
    )
    params: dict[str, str | int | bool] = {
        "adjusted": "false",
        "sort": "desc",
        "limit": limit,
        "apiKey": _api_key(config),
    }
    with httpx.Client(timeout=timeout_seconds or config.massive_request_timeout_seconds) as client:
        response = client.get(url, params=params)
        response.raise_for_status()
        payload = response.json()
    return pd.DataFrame(payload.get("results", []))


def normalize_second_aggregates(raw: pd.DataFrame, *, option_ticker: str) -> pd.DataFrame:
    """Normalize Massive second aggregate fields to the project trade-proxy schema."""
    columns = [
        "options_ticker",
        "timestamp_utc",
        "timestamp_et",
        "option_open",
        "option_high",
        "option_low",
        "option_close",
        "option_vwap",
        "volume",
        "transactions",
        "source_dataset",
    ]
    if raw.empty:
        return pd.DataFrame(columns=columns)
    required = {"t", "o", "h", "l", "c", "v", "vw", "n"}
    missing = sorted(required - set(raw.columns))
    if missing:
        raise ValueError(f"second aggregate frame missing required columns: {missing}")
    out = pd.DataFrame(
        {
            "options_ticker": option_ticker,
            "timestamp_utc": pd.to_datetime(raw["t"], unit="ms", utc=True),
            "option_open": pd.to_numeric(raw["o"], errors="coerce"),
            "option_high": pd.to_numeric(raw["h"], errors="coerce"),
            "option_low": pd.to_numeric(raw["l"], errors="coerce"),
            "option_close": pd.to_numeric(raw["c"], errors="coerce"),
            "option_vwap": pd.to_numeric(raw["vw"], errors="coerce"),
            "volume": pd.to_numeric(raw["v"], errors="coerce").fillna(0).astype(int),
            "transactions": pd.to_numeric(raw["n"], errors="coerce").fillna(0).astype(int),
            "source_dataset": TRADE_PROXY_ROUTE_SECOND_AGGS,
        }
    )
    out["timestamp_et"] = out["timestamp_utc"].dt.tz_convert("America/New_York")
    return out[columns]


def select_latest_proxy_price(
    bars: pd.DataFrame,
    *,
    cutoff_timestamp: datetime,
    lookback_seconds: int = 900,
    price_field: str = "option_vwap",
) -> ProxyPriceSelection:
    if lookback_seconds <= 0:
        raise ValueError("lookback_seconds must be positive.")
    if price_field not in {"option_vwap", "option_close"}:
        raise ValueError("price_field must be option_vwap or option_close.")
    if bars.empty:
        return ProxyPriceSelection(status=TRADE_PROXY_STATUS_NO_TRADE_IN_WINDOW)
    if "timestamp_et" not in bars.columns:
        raise ValueError("bar frame missing timestamp_et column.")
    cutoff = _aware_et_timestamp(cutoff_timestamp, name="cutoff_timestamp")
    start = cutoff - pd.Timedelta(seconds=lookback_seconds)
    frame = bars.copy()
    frame["timestamp_et"] = pd.to_datetime(frame["timestamp_et"])
    if frame["timestamp_et"].dt.tz is None:
        raise ValueError("timestamp_et must be timezone-aware")
    frame["timestamp_et"] = frame["timestamp_et"].dt.tz_convert("America/New_York")
    frame[price_field] = pd.to_numeric(frame[price_field], errors="coerce")
    eligible = frame.loc[
        frame["timestamp_et"].between(start, cutoff, inclusive="both") & frame[price_field].gt(0)
    ].copy()
    if eligible.empty:
        return ProxyPriceSelection(status=TRADE_PROXY_STATUS_NO_TRADE_IN_WINDOW)
    selected = eligible.sort_values("timestamp_et").iloc[-1]
    timestamp = pd.Timestamp(selected["timestamp_et"]).to_pydatetime()
    return ProxyPriceSelection(
        status=TRADE_PROXY_STATUS_OK,
        proxy_price=float(selected[price_field]),
        proxy_timestamp=timestamp,
        proxy_age_seconds=(cutoff.to_pydatetime() - timestamp).total_seconds(),
        proxy_volume=int(eligible["volume"].sum()) if "volume" in eligible.columns else 0,
        proxy_transactions=int(eligible["transactions"].sum())
        if "transactions" in eligible.columns
        else 0,
        proxy_rows_in_window=int(len(eligible)),
        price_field=price_field,
    )


def build_trade_proxy_price_frame(
    contracts: pd.DataFrame,
    bar_frames: dict[str, pd.DataFrame],
    *,
    lookback_seconds: int = 900,
    price_field: str = "option_vwap",
) -> pd.DataFrame:
    """Attach cutoff-proxy prices to candidate contracts from pre-fetched aggregate bars."""
    required = {
        "event_id",
        "ticker",
        "entry_date",
        "exit_date",
        "expiration",
        "strike",
        "right",
        "options_ticker",
        "dte",
        "moneyness_abs",
    }
    missing = sorted(required - set(contracts.columns))
    if missing:
        raise ValueError(f"contract frame missing required columns: {missing}")
    rows: list[dict[str, object]] = []
    for contract in contracts.to_dict("records"):
        option_ticker = str(contract["options_ticker"])
        cutoff_raw = contract.get("event_entry_timestamp")
        cutoff = (
            _to_datetime(cutoff_raw)
            if cutoff_raw is not None and not pd.isna(cutoff_raw)
            else pd.Timestamp(
                _to_date(contract["entry_date"]).isoformat() + " 16:00:00",
                tz="America/New_York",
            ).to_pydatetime()
        )
        selection = select_latest_proxy_price(
            bar_frames.get(option_ticker, pd.DataFrame()),
            cutoff_timestamp=cutoff,
            lookback_seconds=lookback_seconds,
            price_field=price_field,
        )
        rows.append(
            {
                **contract,
                "proxy_price": selection.proxy_price,
                "proxy_timestamp": selection.proxy_timestamp,
                "proxy_age_seconds": selection.proxy_age_seconds,
                "proxy_volume_window": selection.proxy_volume,
                "proxy_transactions_window": selection.proxy_transactions,
                "proxy_rows_in_window": selection.proxy_rows_in_window,
                "proxy_price_field": selection.price_field,
                "proxy_status": selection.status,
                "quote_route": TRADE_PROXY_ROUTE_SECOND_AGGS,
                "quote_status": TRADE_PROXY_PANEL_GRADE,
                "panel_grade": TRADE_PROXY_PANEL_GRADE,
            }
        )
    return pd.DataFrame(rows)


def _implied_volatility(
    *,
    spot: float,
    strike: float,
    time_to_expiry: float,
    option_price: float,
    right: str,
) -> tuple[float | None, str]:
    if spot <= 0 or strike <= 0 or time_to_expiry <= 0 or option_price <= 0:
        return None, "invalid_iv_inputs"
    option_right = OptionRight.CALL if right == "call" else OptionRight.PUT
    intrinsic = (
        max(spot - strike, 0.0) if option_right == OptionRight.CALL else max(strike - spot, 0.0)
    )
    if option_price < intrinsic:
        return None, "price_below_intrinsic"

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
        low = objective(1e-4)
        high = objective(5.0)
        if low * high > 0:
            return None, "iv_root_not_bracketed"
        return float(brentq(objective, 1e-4, 5.0, maxiter=100)), "ok"
    except (ValueError, RuntimeError, OverflowError):
        return None, "iv_solver_failed"


def attach_trade_proxy_local_iv(proxy_prices: pd.DataFrame, windows: pd.DataFrame) -> pd.DataFrame:
    spot_by_event = windows.set_index("event_id")["s_before"].to_dict()
    out = proxy_prices.copy()
    local_ivs: list[float | None] = []
    statuses: list[str] = []
    for row in out.to_dict("records"):
        if row.get("proxy_status") != TRADE_PROXY_STATUS_OK or pd.isna(row.get("proxy_price")):
            local_ivs.append(None)
            statuses.append(str(row.get("proxy_status") or "missing_proxy_price"))
            continue
        spot = float(spot_by_event[row["event_id"]])
        dte = int(row["dte"])
        iv, status = _implied_volatility(
            spot=spot,
            strike=float(row["strike"]),
            time_to_expiry=dte / 365.0,
            option_price=float(row["proxy_price"]),
            right=str(row["right"]),
        )
        local_ivs.append(iv)
        statuses.append(status)
    out["local_iv"] = local_ivs
    out["local_iv_status"] = statuses
    return out


def build_trade_proxy_ivar_inputs(
    iv_contracts: pd.DataFrame, windows: pd.DataFrame
) -> pd.DataFrame:
    """Build event-expiry ATM IV inputs, requiring paired call/put proxy prices."""
    columns = [
        "event_id",
        "ticker",
        "event_date",
        "event_exit_date",
        "entry_date",
        "expiration",
        "iv",
        "dte_days",
        "strike",
        "moneyness",
        "volume",
        "transactions",
        "atm_selection_method",
        "quote_route",
        "quote_status",
        "panel_grade",
    ]
    if iv_contracts.empty:
        return pd.DataFrame(columns=columns)
    rows: list[dict[str, object]] = []
    spot_by_event = windows.set_index("event_id")["s_before"].to_dict()
    for (event_id, expiration), group in iv_contracts.groupby(["event_id", "expiration"]):
        valid = group.loc[group["local_iv"].notna()].copy()
        if valid.empty:
            continue
        spot = float(spot_by_event[event_id])
        pairs: list[dict[str, object]] = []
        for strike, strike_group in valid.groupby("strike"):
            rights = set(strike_group["right"].astype(str))
            if {"call", "put"}.issubset(rights):
                pairs.append(
                    {
                        "strike": float(strike),
                        "iv": float(strike_group["local_iv"].mean()),
                        "volume": int(strike_group["proxy_volume_window"].sum()),
                        "transactions": int(strike_group["proxy_transactions_window"].sum()),
                        "moneyness_abs": abs(float(strike) / spot - 1.0),
                    }
                )
        if not pairs:
            continue
        selected = sorted(
            pairs,
            key=lambda item: (_as_float(item["moneyness_abs"]), -_as_int(item["volume"])),
        )[0]
        event_row = windows.loc[windows["event_id"].eq(event_id)].iloc[0]
        entry_date = _to_date(event_row["entry_date"])
        expiry = _to_date(expiration)
        selected_iv = _as_float(selected["iv"])
        selected_moneyness_abs = _as_float(selected["moneyness_abs"])
        rows.append(
            {
                "event_id": event_id,
                "ticker": event_row["ticker"],
                "event_date": event_row["announcement_date"],
                "event_exit_date": event_row["exit_date"],
                "entry_date": entry_date,
                "expiration": expiry,
                "iv": selected_iv,
                "dte_days": (expiry - entry_date).days,
                "strike": selected["strike"],
                "moneyness": 1.0 + selected_moneyness_abs,
                "volume": selected["volume"],
                "transactions": selected["transactions"],
                "atm_selection_method": "trade_proxy_call_put_average",
                "quote_route": TRADE_PROXY_ROUTE_SECOND_AGGS,
                "quote_status": TRADE_PROXY_PANEL_GRADE,
                "panel_grade": TRADE_PROXY_PANEL_GRADE,
            }
        )
    return pd.DataFrame(rows, columns=columns)


def extract_trade_proxy_event_panel(
    ivar_inputs: pd.DataFrame, windows: pd.DataFrame
) -> pd.DataFrame:
    grouped = ivar_inputs.groupby("event_id") if not ivar_inputs.empty else []
    input_by_event = {event_id: group for event_id, group in grouped}
    rows: list[dict[str, object]] = []
    for event in windows.to_dict("records"):
        event_id = event["event_id"]
        group = input_by_event.get(event_id, pd.DataFrame())
        group_records = (
            []
            if group.empty or "expiration" not in group.columns
            else group.sort_values("expiration").to_dict("records")
        )
        points = [
            TotalVariancePoint(
                expiration=_to_date(row["expiration"]),
                iv=float(row["iv"]) if pd.notna(row["iv"]) else None,
                dte_days=int(row["dte_days"]),
                moneyness=float(row["moneyness"]) if pd.notna(row["moneyness"]) else None,
            )
            for row in group_records
        ]
        extraction = extract_implied_event_variance(
            points,
            event_date=_to_date(event["announcement_date"]),
            event_exit_date=_to_date(event["exit_date"]),
        )
        failure_reason = (
            extraction.failure_reason.value if extraction.failure_reason is not None else None
        )
        rows.append(
            {
                **event,
                "trade_proxy_ivar_event": extraction.ivar_event,
                "ivar_event": extraction.ivar_event,
                "ivar_failure_reason": failure_reason,
                "edge_var_realized": None
                if extraction.ivar_event is None
                else float(event["rvar_event"]) - extraction.ivar_event,
                "t1": extraction.t1,
                "t2": extraction.t2,
                "w1": extraction.w1,
                "w2": extraction.w2,
                "expiry_gap_days": extraction.expiry_gap_days,
                "iv_used_for_extraction_1": extraction.iv_used_for_extraction_1,
                "iv_used_for_extraction_2": extraction.iv_used_for_extraction_2,
                "dte_1": extraction.dte_1,
                "dte_2": extraction.dte_2,
                "expiration_1": extraction.expiration_1,
                "expiration_2": extraction.expiration_2,
                "quote_route": TRADE_PROXY_ROUTE_SECOND_AGGS,
                "quote_status": TRADE_PROXY_PANEL_GRADE,
                "panel_grade": TRADE_PROXY_PANEL_GRADE,
                "paper_grade": False,
            }
        )
    return pd.DataFrame(rows)


def _option_exit_close_lookup(
    option_exit_prices: pd.DataFrame | None,
) -> dict[tuple[str, date], float]:
    if option_exit_prices is None or option_exit_prices.empty:
        return {}
    required = {"options_ticker", "date", "option_close"}
    missing = sorted(required - set(option_exit_prices.columns))
    if missing:
        raise ValueError(f"option exit price frame missing required columns: {missing}")
    lookup: dict[tuple[str, date], float] = {}
    for row in option_exit_prices.to_dict("records"):
        close = row.get("option_close")
        if close is None or pd.isna(close) or float(close) <= 0:
            continue
        lookup[(str(row["options_ticker"]), _to_date(row["date"]))] = float(close)
    return lookup


def build_proxy_straddle_diagnostics(
    iv_contracts: pd.DataFrame,
    windows: pd.DataFrame,
    *,
    option_exit_prices: pd.DataFrame | None = None,
    haircut_fraction: float = 0.10,
) -> pd.DataFrame:
    """Compute one-contract long ATM straddle proxy PnL using entry proxy prices.

    Exit value uses the same contracts' exit-date option day-aggregate close when available.
    Intrinsic value is a fallback only when the exit option close is missing or the contract
    expires on the exit date. This remains a gross screening diagnostic, not a quote-executable
    strategy result.
    """
    columns = [
        "event_id",
        "ticker",
        "entry_date",
        "exit_date",
        "expiration",
        "strike",
        "entry_premium_usd",
        "exit_option_value_usd",
        "exit_intrinsic_usd",
        "underlying_exit_price_source",
        "option_exit_price_source",
        "option_exit_price_status",
        "used_intrinsic_fallback",
        "gross_proxy_pnl_usd",
        "haircut_pnl_usd",
        "proxy_volume_window",
        "proxy_transactions_window",
        "panel_grade",
    ]
    if iv_contracts.empty:
        return pd.DataFrame(columns=columns)
    exit_close_by_contract = _option_exit_close_lookup(option_exit_prices)
    rows: list[dict[str, object]] = []
    seen_events: set[object] = set()
    grouped = iv_contracts.sort_values(["event_id", "expiration"]).groupby(
        ["event_id", "expiration"]
    )
    for (event_id, expiration), group in grouped:
        if event_id in seen_events:
            continue
        event = windows.loc[windows["event_id"].eq(event_id)]
        if event.empty:
            continue
        event_row = event.iloc[0]
        spot = float(event_row["s_before"])
        candidates: list[dict[str, object]] = []
        valid = group.loc[group["proxy_status"].eq(TRADE_PROXY_STATUS_OK)].copy()
        for strike, strike_group in valid.groupby("strike"):
            rights = set(strike_group["right"].astype(str))
            if {"call", "put"}.issubset(rights):
                call = strike_group.loc[strike_group["right"].astype(str).eq("call")].iloc[0]
                put = strike_group.loc[strike_group["right"].astype(str).eq("put")].iloc[0]
                candidates.append(
                    {
                        "strike": float(strike),
                        "call_price": float(call["proxy_price"]),
                        "put_price": float(put["proxy_price"]),
                        "call_options_ticker": str(call["options_ticker"]),
                        "put_options_ticker": str(put["options_ticker"]),
                        "moneyness_abs": abs(float(strike) / spot - 1.0),
                        "volume": int(strike_group["proxy_volume_window"].sum()),
                        "transactions": int(strike_group["proxy_transactions_window"].sum()),
                    }
                )
        if not candidates:
            continue
        selected = sorted(
            candidates,
            key=lambda item: (_as_float(item["moneyness_abs"]), -_as_int(item["volume"])),
        )[0]
        strike = _as_float(selected["strike"])
        entry_premium_usd = (
            _as_float(selected["call_price"]) + _as_float(selected["put_price"])
        ) * 100.0
        exit_date = _to_date(event_row["exit_date"])
        expiry = _to_date(expiration)
        terminal_spot = float(event_row["s_after"])
        exit_intrinsic_usd = abs(terminal_spot - strike) * 100.0
        call_exit_close = exit_close_by_contract.get(
            (str(selected["call_options_ticker"]), exit_date)
        )
        put_exit_close = exit_close_by_contract.get(
            (str(selected["put_options_ticker"]), exit_date)
        )
        use_intrinsic = expiry <= exit_date or call_exit_close is None or put_exit_close is None
        if expiry <= exit_date:
            exit_status = OPTION_EXIT_STATUS_EXPIRATION_AT_EXIT
        elif call_exit_close is None or put_exit_close is None:
            exit_status = OPTION_EXIT_STATUS_MISSING_DAY_AGG
        else:
            exit_status = OPTION_EXIT_STATUS_OK
        if use_intrinsic:
            exit_option_value_usd = exit_intrinsic_usd
        else:
            assert call_exit_close is not None
            assert put_exit_close is not None
            exit_option_value_usd = (float(call_exit_close) + float(put_exit_close)) * 100.0
        gross_pnl = exit_option_value_usd - entry_premium_usd
        rows.append(
            {
                "event_id": event_id,
                "ticker": event_row["ticker"],
                "entry_date": event_row["entry_date"],
                "exit_date": exit_date,
                "expiration": expiry,
                "strike": strike,
                "entry_premium_usd": entry_premium_usd,
                "exit_option_value_usd": exit_option_value_usd,
                "exit_intrinsic_usd": exit_intrinsic_usd,
                "underlying_exit_price_source": UNDERLYING_EXIT_PRICE_SOURCE,
                "option_exit_price_source": OPTION_EXIT_PRICE_SOURCE_DAY_AGGS,
                "option_exit_price_status": exit_status,
                "used_intrinsic_fallback": use_intrinsic,
                "gross_proxy_pnl_usd": gross_pnl,
                "haircut_pnl_usd": gross_pnl - haircut_fraction * entry_premium_usd,
                "proxy_volume_window": selected["volume"],
                "proxy_transactions_window": selected["transactions"],
                "panel_grade": TRADE_PROXY_PANEL_GRADE,
            }
        )
        seen_events.add(event_id)
    return pd.DataFrame(rows, columns=columns)


def summarize_trade_proxy_panel(
    *,
    panel: pd.DataFrame,
    proxy_prices: pd.DataFrame,
    straddle_diagnostics: pd.DataFrame,
    lookback_seconds: int,
    price_field: str,
) -> dict[str, object]:
    status_counts = (
        proxy_prices["proxy_status"].fillna("missing").value_counts().to_dict()
        if not proxy_prices.empty
        else {}
    )
    return {
        "panel_grade": TRADE_PROXY_PANEL_GRADE,
        "paper_grade": False,
        "quote_route": TRADE_PROXY_ROUTE_SECOND_AGGS,
        "lookback_seconds": lookback_seconds,
        "price_field": price_field,
        "events": int(len(panel)),
        "events_with_rvar": int(panel["rvar_event"].notna().sum()) if "rvar_event" in panel else 0,
        "events_with_trade_proxy_ivar": int(panel["trade_proxy_ivar_event"].notna().sum())
        if "trade_proxy_ivar_event" in panel
        else 0,
        "ivar_failure_counts": panel["ivar_failure_reason"].fillna("ok").value_counts().to_dict()
        if "ivar_failure_reason" in panel
        else {},
        "proxy_contracts": int(len(proxy_prices)),
        "proxy_contract_status_counts": status_counts,
        "proxy_contracts_with_local_iv": int(proxy_prices["local_iv"].notna().sum())
        if "local_iv" in proxy_prices
        else 0,
        "straddle_diagnostics_rows": int(len(straddle_diagnostics)),
        "mean_gross_proxy_pnl_usd": float(straddle_diagnostics["gross_proxy_pnl_usd"].mean())
        if not straddle_diagnostics.empty
        else None,
        "mean_haircut_pnl_usd": float(straddle_diagnostics["haircut_pnl_usd"].mean())
        if not straddle_diagnostics.empty
        else None,
        "limitations": [
            "Second aggregates are trade-price OHLCV bars, not bid/ask or NBBO quotes.",
            "Proxy PnL is for signal screening only and is not full-spread executable.",
            "Paper-grade transaction-cost tables still require historical bid/ask or NBBO.",
        ],
    }


def edge_decile_diagnostics(panel: pd.DataFrame) -> pd.DataFrame:
    if panel.empty or "edge_var_realized" not in panel.columns:
        return pd.DataFrame(columns=["edge_decile", "mean_expost_mispricing", "count"])
    frame = panel.loc[pd.to_numeric(panel["edge_var_realized"], errors="coerce").notna()].copy()
    if frame.empty:
        return pd.DataFrame(columns=["edge_decile", "mean_expost_mispricing", "count"])
    bins = min(10, len(frame))
    frame["edge_decile"] = pd.qcut(
        frame["edge_var_realized"].rank(method="first"),
        q=bins,
        labels=False,
        duplicates="drop",
    )
    return (
        frame.groupby("edge_decile", dropna=False)
        .agg(mean_expost_mispricing=("edge_var_realized", "mean"), count=("event_id", "count"))
        .reset_index()
    )


def write_trade_proxy_metadata(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")

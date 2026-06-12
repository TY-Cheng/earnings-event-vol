from __future__ import annotations

import json
import re
import time as time_module
import urllib.parse
from collections.abc import Mapping
from dataclasses import dataclass
from datetime import date, datetime
from pathlib import Path
from typing import Any, cast

import httpx
import pandas as pd
from scipy.optimize import brentq

from earnings_event_vol.backtest import black_scholes_price
from earnings_event_vol.config import ProjectConfig
from earnings_event_vol.events import market_close_timestamp
from earnings_event_vol.massive import read_secret_file
from earnings_event_vol.rate_limit import throttle_requests_per_minute
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
OPTION_EXIT_PAYOFF_FALLBACK_INTRINSIC = "intrinsic_value_at_underlying_exit"
OPTION_EXIT_STATUS_MISSING_PRECLOSE_VWAP = "missing_exit_preclose_vwap_intrinsic_fallback"
OPTION_EXIT_STATUS_EXPIRATION_AT_EXIT = "expiration_at_exit_intrinsic"
EXIT_PRECLOSE_OPTION_VWAP_SOURCE = "exit_preclose_15m_option_second_agg_vwap"
EXIT_PRECLOSE_OPTION_VWAP_STATUS_OK = "ok"
EXIT_PRECLOSE_OPTION_VWAP_STATUS_MISSING_LEG = "missing_leg_vwap"
C2O_PROXY_PNL_SOURCE_INTRINSIC_OPEN = "underlying_open_intrinsic_diagnostic_not_option_vwap"
C2O_PROXY_PNL_STATUS_OK = "vendor_open_intrinsic_proxy"
C2O_PROXY_PNL_STATUS_MISSING_OPEN = "missing_open_after"
ENTRY_PRICE_METHOD_PRECLOSE_WINDOW_VWAP = "preclose_15m_option_second_agg_vwap"
POST_OPEN_OPTION_VWAP_SOURCE = "post_open_option_second_agg_vwap"
POST_OPEN_OPTION_VWAP_STATUS_OK = "ok"
POST_OPEN_OPTION_VWAP_STATUS_MISSING_LEG = "missing_leg_vwap"
REACTION_O2C_OPTION_VWAP_SOURCE = "post_open_option_vwap_to_c2c_exit_proxy"
REACTION_O2C_OPTION_VWAP_STATUS_OK = "ok"
REACTION_O2C_OPTION_VWAP_STATUS_MISSING_OPEN_ANCHOR = "missing_open_option_vwap"
POST_OPEN_OPTION_VWAP_WINDOWS: tuple[tuple[str, int, int], ...] = (
    ("0_5", 0, 5),
    ("5_15", 5, 15),
)
_SECRET_QUERY_PATTERN = re.compile(r"(?i)((?:apiKey|api_key)=)[^&\s)]+")
QueryParamValue = str | int | float | bool | None


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
    price_method: str = "window_vwap"
    window_start: datetime | None = None
    window_end: datetime | None = None


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


def _optional_positive_float(value: object) -> float | None:
    if value is None or pd.isna(value):
        return None
    out = float(cast(Any, value))
    return out if pd.notna(out) and out > 0 else None


def redact_secret_query_params(text: str) -> str:
    return _SECRET_QUERY_PATTERN.sub(r"\1<redacted>", text)


def safe_exception_text(exc: Exception, *, max_chars: int = 300) -> str:
    if isinstance(exc, httpx.HTTPStatusError):
        body = redact_secret_query_params(" ".join(exc.response.text.strip().split()))
        suffix = f": {body[:200]}" if body else ""
        return redact_secret_query_params(f"HTTP {exc.response.status_code}{suffix}")[:max_chars]
    return redact_secret_query_params(str(exc))[:max_chars]


def _retryable_http_error(exc: httpx.HTTPStatusError) -> bool:
    return exc.response.status_code in {429, 500, 502, 503, 504}


def _get_json_with_retries(
    client: httpx.Client,
    url: str,
    *,
    params: Mapping[str, QueryParamValue],
    max_retries: int,
    backoff_seconds: float,
    requests_per_minute: int | None = None,
) -> dict[str, object]:
    attempts = max(1, int(max_retries) + 1)
    last_exc: Exception | None = None
    for attempt in range(attempts):
        try:
            throttle_requests_per_minute(requests_per_minute)
            response = client.get(url, params=dict(params))
            response.raise_for_status()
            payload = response.json()
            return payload if isinstance(payload, dict) else {"results": []}
        except httpx.HTTPStatusError as exc:
            if not _retryable_http_error(exc):
                raise
            last_exc = exc
        except (httpx.TransportError, httpx.TimeoutException) as exc:
            last_exc = exc
        if attempt < attempts - 1 and backoff_seconds > 0:
            time_module.sleep(backoff_seconds * (2**attempt))
    assert last_exc is not None
    raise last_exc


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
    try:
        with httpx.Client(
            timeout=timeout_seconds or config.massive_request_timeout_seconds
        ) as client:
            payload = _get_json_with_retries(
                client,
                url,
                params=params,
                max_retries=config.massive_max_retries,
                backoff_seconds=config.massive_retry_backoff_seconds,
                requests_per_minute=config.massive_requests_per_minute,
            )
    except httpx.HTTPError as exc:
        raise RuntimeError(safe_exception_text(exc)) from None
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


def select_option_window_vwap(
    bars: pd.DataFrame,
    *,
    window_start: datetime,
    window_end: datetime,
    price_field: str = "option_vwap",
    include_end: bool = True,
    price_method: str = "window_vwap",
) -> ProxyPriceSelection:
    if price_field not in {"option_vwap", "option_close"}:
        raise ValueError("price_field must be option_vwap or option_close.")
    start = _aware_et_timestamp(window_start, name="window_start")
    end = _aware_et_timestamp(window_end, name="window_end")
    if end <= start:
        raise ValueError("window_end must be after window_start.")
    if bars.empty:
        return ProxyPriceSelection(
            status=TRADE_PROXY_STATUS_NO_TRADE_IN_WINDOW,
            price_field=price_field,
            price_method=price_method,
            window_start=start.to_pydatetime(),
            window_end=end.to_pydatetime(),
        )
    if "timestamp_et" not in bars.columns:
        raise ValueError("bar frame missing timestamp_et column.")
    frame = bars.copy()
    frame["timestamp_et"] = pd.to_datetime(frame["timestamp_et"])
    if frame["timestamp_et"].dt.tz is None:
        raise ValueError("timestamp_et must be timezone-aware")
    frame["timestamp_et"] = frame["timestamp_et"].dt.tz_convert("America/New_York")
    frame[price_field] = pd.to_numeric(frame[price_field], errors="coerce")
    if "volume" in frame.columns:
        frame["volume"] = pd.to_numeric(frame["volume"], errors="coerce").fillna(0)
    else:
        frame["volume"] = 0
    if include_end:
        in_window = frame["timestamp_et"].between(start, end, inclusive="both")
    else:
        in_window = frame["timestamp_et"].ge(start) & frame["timestamp_et"].lt(end)
    eligible = frame.loc[in_window & frame[price_field].gt(0) & frame["volume"].gt(0)].copy()
    if eligible.empty:
        return ProxyPriceSelection(
            status=TRADE_PROXY_STATUS_NO_TRADE_IN_WINDOW,
            price_field=price_field,
            price_method=price_method,
            window_start=start.to_pydatetime(),
            window_end=end.to_pydatetime(),
        )
    eligible = eligible.sort_values("timestamp_et")
    volume = pd.to_numeric(eligible["volume"], errors="coerce").fillna(0.0)
    total_volume = float(volume.sum())
    if total_volume <= 0:
        return ProxyPriceSelection(
            status=TRADE_PROXY_STATUS_NO_TRADE_IN_WINDOW,
            price_field=price_field,
            price_method=price_method,
            window_start=start.to_pydatetime(),
            window_end=end.to_pydatetime(),
        )
    window_vwap = float((pd.to_numeric(eligible[price_field], errors="coerce") * volume).sum())
    window_vwap /= total_volume
    selected = eligible.iloc[-1]
    timestamp = pd.Timestamp(selected["timestamp_et"]).to_pydatetime()
    return ProxyPriceSelection(
        status=TRADE_PROXY_STATUS_OK,
        proxy_price=window_vwap,
        proxy_timestamp=timestamp,
        proxy_age_seconds=(end.to_pydatetime() - timestamp).total_seconds(),
        proxy_volume=int(total_volume),
        proxy_transactions=int(eligible["transactions"].sum())
        if "transactions" in eligible.columns
        else 0,
        proxy_rows_in_window=int(len(eligible)),
        price_field=price_field,
        price_method=price_method,
        window_start=start.to_pydatetime(),
        window_end=end.to_pydatetime(),
    )


def select_preclose_entry_proxy_price(
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
        return ProxyPriceSelection(
            status=TRADE_PROXY_STATUS_NO_TRADE_IN_WINDOW,
            price_field=price_field,
            price_method=ENTRY_PRICE_METHOD_PRECLOSE_WINDOW_VWAP,
        )
    if "timestamp_et" not in bars.columns:
        raise ValueError("bar frame missing timestamp_et column.")
    cutoff = _aware_et_timestamp(cutoff_timestamp, name="cutoff_timestamp")
    start = cutoff - pd.Timedelta(seconds=lookback_seconds)
    return select_option_window_vwap(
        bars,
        window_start=start.to_pydatetime(),
        window_end=cutoff.to_pydatetime(),
        price_field=price_field,
        include_end=True,
        price_method=ENTRY_PRICE_METHOD_PRECLOSE_WINDOW_VWAP,
    )


def build_trade_proxy_price_frame(
    contracts: pd.DataFrame,
    bar_frames: Mapping[Any, Any],
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
            else market_close_timestamp(_to_date(contract["entry_date"]))
        )
        entry_date = _to_date(contract["entry_date"])
        cutoff_key = pd.Timestamp(cutoff).tz_convert("America/New_York").isoformat()
        bars = bar_frames.get((option_ticker, entry_date, cutoff_key))
        if bars is None:
            bars = bar_frames.get((option_ticker, entry_date))
        if bars is None:
            bars = bar_frames.get(option_ticker)
        if bars is None:
            bars = pd.DataFrame()
        selection = select_preclose_entry_proxy_price(
            bars,
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
                "proxy_price_method": selection.price_method,
                "proxy_window_start": selection.window_start,
                "proxy_window_end": selection.window_end,
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


def build_post_open_option_vwap_frame(
    selected_straddles: pd.DataFrame,
    bar_frames: dict[tuple[str, date], pd.DataFrame],
    *,
    price_field: str = "option_vwap",
) -> pd.DataFrame:
    """Compute post-open option VWAP windows for selected straddle legs.

    These are trade-aggregate VWAP marks, not bid/ask or NBBO executable exits.
    """
    columns = [
        "event_id",
        "options_ticker",
        "exit_date",
        "window_label",
        "window_start",
        "window_end",
        "option_exit_vwap",
        "volume",
        "transactions",
        "rows_in_window",
        "last_trade_timestamp",
        "last_trade_age_seconds",
        "status",
        "source",
        "panel_grade",
    ]
    if selected_straddles.empty:
        return pd.DataFrame(columns=columns)
    rows: list[dict[str, object]] = []
    for straddle in selected_straddles.to_dict("records"):
        event_id = straddle["event_id"]
        exit_date = _to_date(straddle["exit_date"])
        open_ts = pd.Timestamp(
            f"{exit_date.isoformat()} 09:30:00", tz="America/New_York"
        ).to_pydatetime()
        for option_column in ("call_options_ticker", "put_options_ticker"):
            option_ticker_raw = straddle.get(option_column)
            if option_ticker_raw is None or pd.isna(option_ticker_raw):
                continue
            option_ticker = str(option_ticker_raw)
            bars = bar_frames.get((option_ticker, exit_date), pd.DataFrame())
            for label, start_minute, end_minute in POST_OPEN_OPTION_VWAP_WINDOWS:
                start = (pd.Timestamp(open_ts) + pd.Timedelta(minutes=start_minute)).to_pydatetime()
                end = (pd.Timestamp(open_ts) + pd.Timedelta(minutes=end_minute)).to_pydatetime()
                selection = select_option_window_vwap(
                    bars,
                    window_start=start,
                    window_end=end,
                    price_field=price_field,
                    include_end=True,
                    price_method=f"post_open_{label}_window_vwap",
                )
                rows.append(
                    {
                        "event_id": event_id,
                        "options_ticker": option_ticker,
                        "exit_date": exit_date,
                        "window_label": label,
                        "window_start": start,
                        "window_end": end,
                        "option_exit_vwap": selection.proxy_price,
                        "volume": selection.proxy_volume,
                        "transactions": selection.proxy_transactions,
                        "rows_in_window": selection.proxy_rows_in_window,
                        "last_trade_timestamp": selection.proxy_timestamp,
                        "last_trade_age_seconds": selection.proxy_age_seconds,
                        "status": selection.status,
                        "source": POST_OPEN_OPTION_VWAP_SOURCE,
                        "panel_grade": TRADE_PROXY_PANEL_GRADE,
                    }
                )
    return pd.DataFrame(rows, columns=columns)


def build_exit_preclose_option_vwap_frame(
    selected_straddles: pd.DataFrame,
    bar_frames: dict[tuple[str, date], pd.DataFrame],
    *,
    price_field: str = "option_vwap",
    lookback_seconds: int = 900,
) -> pd.DataFrame:
    """Compute exit-day preclose option VWAP for selected straddle legs.

    These are trade-aggregate VWAP marks, not bid/ask or NBBO executable exits.
    """
    columns = [
        "event_id",
        "options_ticker",
        "exit_date",
        "window_start",
        "window_end",
        "option_exit_vwap",
        "volume",
        "transactions",
        "rows_in_window",
        "last_trade_timestamp",
        "last_trade_age_seconds",
        "status",
        "source",
        "panel_grade",
    ]
    if selected_straddles.empty:
        return pd.DataFrame(columns=columns)
    rows: list[dict[str, object]] = []
    for straddle in selected_straddles.to_dict("records"):
        event_id = straddle["event_id"]
        exit_date = _to_date(straddle["exit_date"])
        close_ts = market_close_timestamp(exit_date)
        start_ts = (pd.Timestamp(close_ts) - pd.Timedelta(seconds=lookback_seconds)).to_pydatetime()
        for option_column in ("call_options_ticker", "put_options_ticker"):
            option_ticker_raw = straddle.get(option_column)
            if option_ticker_raw is None or pd.isna(option_ticker_raw):
                continue
            option_ticker = str(option_ticker_raw)
            bars = bar_frames.get((option_ticker, exit_date), pd.DataFrame())
            selection = select_option_window_vwap(
                bars,
                window_start=start_ts,
                window_end=close_ts,
                price_field=price_field,
                include_end=True,
                price_method=EXIT_PRECLOSE_OPTION_VWAP_SOURCE,
            )
            rows.append(
                {
                    "event_id": event_id,
                    "options_ticker": option_ticker,
                    "exit_date": exit_date,
                    "window_start": start_ts,
                    "window_end": close_ts,
                    "option_exit_vwap": selection.proxy_price,
                    "volume": selection.proxy_volume,
                    "transactions": selection.proxy_transactions,
                    "rows_in_window": selection.proxy_rows_in_window,
                    "last_trade_timestamp": selection.proxy_timestamp,
                    "last_trade_age_seconds": selection.proxy_age_seconds,
                    "status": selection.status,
                    "source": EXIT_PRECLOSE_OPTION_VWAP_SOURCE,
                    "panel_grade": TRADE_PROXY_PANEL_GRADE,
                }
            )
    return pd.DataFrame(rows, columns=columns)


def _post_open_option_vwap_lookup(
    post_open_option_prices: pd.DataFrame | None,
) -> dict[tuple[object, str, str], dict[str, object]]:
    if post_open_option_prices is None or post_open_option_prices.empty:
        return {}
    required = {
        "event_id",
        "options_ticker",
        "window_label",
        "option_exit_vwap",
        "volume",
        "transactions",
        "rows_in_window",
        "status",
    }
    missing = sorted(required - set(post_open_option_prices.columns))
    if missing:
        raise ValueError(f"post-open option price frame missing required columns: {missing}")
    lookup: dict[tuple[object, str, str], dict[str, object]] = {}
    for row in post_open_option_prices.to_dict("records"):
        key = (row["event_id"], str(row["options_ticker"]), str(row["window_label"]))
        lookup[key] = row
    return lookup


def _exit_preclose_option_vwap_lookup(
    exit_preclose_option_prices: pd.DataFrame | None,
) -> dict[tuple[object, str], dict[str, object]]:
    if exit_preclose_option_prices is None or exit_preclose_option_prices.empty:
        return {}
    required = {
        "event_id",
        "options_ticker",
        "option_exit_vwap",
        "volume",
        "transactions",
        "rows_in_window",
        "status",
    }
    missing = sorted(required - set(exit_preclose_option_prices.columns))
    if missing:
        raise ValueError(f"exit preclose option price frame missing required columns: {missing}")
    lookup: dict[tuple[object, str], dict[str, object]] = {}
    for row in exit_preclose_option_prices.to_dict("records"):
        lookup[(row["event_id"], str(row["options_ticker"]))] = row
    return lookup


def _exit_preclose_window_metrics(
    *,
    lookup: dict[tuple[object, str], dict[str, object]],
    event_id: object,
    call_options_ticker: str,
    put_options_ticker: str,
    entry_premium_usd: float,
    haircut_fraction: float,
) -> dict[str, object]:
    call = lookup.get((event_id, call_options_ticker))
    put = lookup.get((event_id, put_options_ticker))
    call_price_raw = None if call is None else call.get("option_exit_vwap")
    put_price_raw = None if put is None else put.get("option_exit_vwap")
    call_price = (
        float(cast(Any, call_price_raw))
        if call_price_raw is not None and pd.notna(call_price_raw)
        else None
    )
    put_price = (
        float(cast(Any, put_price_raw))
        if put_price_raw is not None and pd.notna(put_price_raw)
        else None
    )
    if call_price is not None and put_price is not None and call_price > 0 and put_price > 0:
        exit_usd = (call_price + put_price) * 100.0
        gross = exit_usd - entry_premium_usd
        haircut = gross - haircut_fraction * entry_premium_usd
        status = EXIT_PRECLOSE_OPTION_VWAP_STATUS_OK
    else:
        exit_usd = None
        gross = None
        haircut = None
        status = EXIT_PRECLOSE_OPTION_VWAP_STATUS_MISSING_LEG
    call_volume = 0 if call is None else int(cast(Any, call.get("volume") or 0))
    put_volume = 0 if put is None else int(cast(Any, put.get("volume") or 0))
    call_transactions = 0 if call is None else int(cast(Any, call.get("transactions") or 0))
    put_transactions = 0 if put is None else int(cast(Any, put.get("transactions") or 0))
    call_rows = 0 if call is None else int(cast(Any, call.get("rows_in_window") or 0))
    put_rows = 0 if put is None else int(cast(Any, put.get("rows_in_window") or 0))
    return {
        "exit_option_vwap_preclose_15m_value_usd": exit_usd,
        "gross_exit_option_vwap_preclose_15m_proxy_pnl_usd": gross,
        "exit_option_vwap_preclose_15m_haircut_pnl_usd": haircut,
        "exit_option_vwap_preclose_15m_status": status,
        "exit_option_vwap_preclose_15m_source": EXIT_PRECLOSE_OPTION_VWAP_SOURCE,
        "exit_option_vwap_preclose_15m_volume": call_volume + put_volume,
        "exit_option_vwap_preclose_15m_transactions": call_transactions + put_transactions,
        "exit_option_vwap_preclose_15m_rows": call_rows + put_rows,
    }


def _post_open_window_metrics(
    *,
    lookup: dict[tuple[object, str, str], dict[str, object]],
    event_id: object,
    call_options_ticker: str,
    put_options_ticker: str,
    label: str,
    entry_premium_usd: float,
    haircut_fraction: float,
) -> dict[str, object]:
    prefix = f"post_open_option_vwap_{label}"
    call = lookup.get((event_id, call_options_ticker, label))
    put = lookup.get((event_id, put_options_ticker, label))
    call_price_raw = None if call is None else call.get("option_exit_vwap")
    put_price_raw = None if put is None else put.get("option_exit_vwap")
    call_price = (
        float(cast(Any, call_price_raw))
        if call_price_raw is not None and pd.notna(call_price_raw)
        else None
    )
    put_price = (
        float(cast(Any, put_price_raw))
        if put_price_raw is not None and pd.notna(put_price_raw)
        else None
    )
    if call_price is not None and put_price is not None and call_price > 0 and put_price > 0:
        exit_usd = (call_price + put_price) * 100.0
        gross = exit_usd - entry_premium_usd
        haircut = gross - haircut_fraction * entry_premium_usd
        status = POST_OPEN_OPTION_VWAP_STATUS_OK
    else:
        exit_usd = None
        gross = None
        haircut = None
        status = POST_OPEN_OPTION_VWAP_STATUS_MISSING_LEG
    call_volume = 0 if call is None else int(cast(Any, call.get("volume") or 0))
    put_volume = 0 if put is None else int(cast(Any, put.get("volume") or 0))
    call_transactions = 0 if call is None else int(cast(Any, call.get("transactions") or 0))
    put_transactions = 0 if put is None else int(cast(Any, put.get("transactions") or 0))
    call_rows = 0 if call is None else int(cast(Any, call.get("rows_in_window") or 0))
    put_rows = 0 if put is None else int(cast(Any, put.get("rows_in_window") or 0))
    volume = call_volume + put_volume
    transactions = call_transactions + put_transactions
    rows = call_rows + put_rows
    return {
        f"{prefix}_exit_usd": exit_usd,
        f"gross_{prefix}_proxy_pnl_usd": gross,
        f"{prefix}_haircut_pnl_usd": haircut,
        f"{prefix}_status": status,
        f"{prefix}_source": POST_OPEN_OPTION_VWAP_SOURCE,
        f"{prefix}_volume": volume,
        f"{prefix}_transactions": transactions,
        f"{prefix}_rows": rows,
    }


def _reaction_o2c_window_metrics(
    *,
    label: str,
    post_open_metrics: dict[str, object],
    exit_option_value_usd: float,
    gross_c2c_pnl_usd: float,
    haircut_fraction: float,
) -> dict[str, object]:
    """Compute O2C proxy PnL using the same option-VWAP anchor as C2O exit."""
    prefix = f"reaction_o2c_option_vwap_{label}"
    c2o_prefix = f"post_open_option_vwap_{label}"
    open_anchor = _optional_positive_float(post_open_metrics.get(f"{c2o_prefix}_exit_usd"))
    c2o_gross = post_open_metrics.get(f"gross_{c2o_prefix}_proxy_pnl_usd")
    if open_anchor is None:
        gross = None
        haircut = None
        residual = None
        status = REACTION_O2C_OPTION_VWAP_STATUS_MISSING_OPEN_ANCHOR
    else:
        gross = exit_option_value_usd - open_anchor
        haircut = gross - haircut_fraction * open_anchor
        c2o_value = _as_float(c2o_gross) if c2o_gross is not None and pd.notna(c2o_gross) else None
        residual = None if c2o_value is None else gross_c2c_pnl_usd - c2o_value - gross
        status = REACTION_O2C_OPTION_VWAP_STATUS_OK
    return {
        f"open_option_vwap_{label}_anchor_usd": open_anchor,
        f"gross_{prefix}_to_c2c_exit_proxy_pnl_usd": gross,
        f"{prefix}_haircut_pnl_usd": haircut,
        f"{prefix}_status": status,
        f"{prefix}_source": REACTION_O2C_OPTION_VWAP_SOURCE,
        f"option_proxy_decomposition_residual_{label}_usd": residual,
    }


def build_proxy_straddle_diagnostics(
    iv_contracts: pd.DataFrame,
    windows: pd.DataFrame,
    *,
    exit_preclose_option_prices: pd.DataFrame | None = None,
    post_open_option_prices: pd.DataFrame | None = None,
    haircut_fraction: float = 0.10,
) -> pd.DataFrame:
    """Compute one-contract long ATM straddle proxy PnL using entry proxy prices.

    Primary C2C exit value uses the same contracts' option VWAP over the final 15
    minutes before the exit-date close. Intrinsic value is used only when this
    mark is missing or the contract expires on the exit date. This remains a
    gross screening diagnostic, not a quote-executable strategy result.
    """
    columns = [
        "event_id",
        "ticker",
        "entry_date",
        "exit_date",
        "expiration",
        "strike",
        "call_options_ticker",
        "put_options_ticker",
        "entry_premium_usd",
        "entry_price_method",
        "exit_option_value_usd",
        "exit_option_vwap_preclose_15m_value_usd",
        "gross_exit_option_vwap_preclose_15m_proxy_pnl_usd",
        "exit_option_vwap_preclose_15m_haircut_pnl_usd",
        "exit_option_vwap_preclose_15m_status",
        "exit_option_vwap_preclose_15m_source",
        "exit_option_vwap_preclose_15m_volume",
        "exit_option_vwap_preclose_15m_transactions",
        "exit_option_vwap_preclose_15m_rows",
        "exit_intrinsic_usd",
        "open_after",
        "c2o_exit_intrinsic_usd",
        "gross_c2o_intrinsic_proxy_pnl_usd",
        "c2o_haircut_pnl_usd",
        "c2o_proxy_pnl_source",
        "c2o_proxy_pnl_status",
        "post_open_option_vwap_0_5_exit_usd",
        "gross_post_open_option_vwap_0_5_proxy_pnl_usd",
        "post_open_option_vwap_0_5_haircut_pnl_usd",
        "post_open_option_vwap_0_5_status",
        "post_open_option_vwap_0_5_source",
        "post_open_option_vwap_0_5_volume",
        "post_open_option_vwap_0_5_transactions",
        "post_open_option_vwap_0_5_rows",
        "open_option_vwap_0_5_anchor_usd",
        "gross_reaction_o2c_option_vwap_0_5_to_c2c_exit_proxy_pnl_usd",
        "reaction_o2c_option_vwap_0_5_haircut_pnl_usd",
        "reaction_o2c_option_vwap_0_5_status",
        "reaction_o2c_option_vwap_0_5_source",
        "option_proxy_decomposition_residual_0_5_usd",
        "post_open_option_vwap_5_15_exit_usd",
        "gross_post_open_option_vwap_5_15_proxy_pnl_usd",
        "post_open_option_vwap_5_15_haircut_pnl_usd",
        "post_open_option_vwap_5_15_status",
        "post_open_option_vwap_5_15_source",
        "post_open_option_vwap_5_15_volume",
        "post_open_option_vwap_5_15_transactions",
        "post_open_option_vwap_5_15_rows",
        "open_option_vwap_5_15_anchor_usd",
        "gross_reaction_o2c_option_vwap_5_15_to_c2c_exit_proxy_pnl_usd",
        "reaction_o2c_option_vwap_5_15_haircut_pnl_usd",
        "reaction_o2c_option_vwap_5_15_status",
        "reaction_o2c_option_vwap_5_15_source",
        "option_proxy_decomposition_residual_5_15_usd",
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
    exit_preclose_lookup = _exit_preclose_option_vwap_lookup(exit_preclose_option_prices)
    post_open_lookup = _post_open_option_vwap_lookup(post_open_option_prices)
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
                        "entry_price_method": str(
                            call.get("proxy_price_method")
                            or put.get("proxy_price_method")
                            or ENTRY_PRICE_METHOD_PRECLOSE_WINDOW_VWAP
                        ),
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
        call_options_ticker = str(selected["call_options_ticker"])
        put_options_ticker = str(selected["put_options_ticker"])
        post_open_metrics: dict[str, object] = {}
        for label, _, _ in POST_OPEN_OPTION_VWAP_WINDOWS:
            post_open_metrics.update(
                _post_open_window_metrics(
                    lookup=post_open_lookup,
                    event_id=event_id,
                    call_options_ticker=call_options_ticker,
                    put_options_ticker=put_options_ticker,
                    label=label,
                    entry_premium_usd=entry_premium_usd,
                    haircut_fraction=haircut_fraction,
                )
            )
        exit_preclose_metrics = _exit_preclose_window_metrics(
            lookup=exit_preclose_lookup,
            event_id=event_id,
            call_options_ticker=call_options_ticker,
            put_options_ticker=put_options_ticker,
            entry_premium_usd=entry_premium_usd,
            haircut_fraction=haircut_fraction,
        )
        exit_date = _to_date(event_row["exit_date"])
        expiry = _to_date(expiration)
        terminal_spot = float(event_row["s_after"])
        exit_intrinsic_usd = abs(terminal_spot - strike) * 100.0
        open_after = _optional_positive_float(event_row.get("open_after"))
        if open_after is None:
            c2o_exit_intrinsic_usd = None
            gross_c2o_intrinsic_proxy_pnl = None
            c2o_haircut_pnl = None
            c2o_status = C2O_PROXY_PNL_STATUS_MISSING_OPEN
        else:
            c2o_exit_intrinsic_usd = abs(open_after - strike) * 100.0
            gross_c2o_intrinsic_proxy_pnl = c2o_exit_intrinsic_usd - entry_premium_usd
            c2o_haircut_pnl = gross_c2o_intrinsic_proxy_pnl - haircut_fraction * entry_premium_usd
            c2o_status = C2O_PROXY_PNL_STATUS_OK
        exit_preclose_value = _optional_positive_float(
            exit_preclose_metrics.get("exit_option_vwap_preclose_15m_value_usd")
        )
        if expiry <= exit_date:
            exit_option_value_usd = exit_intrinsic_usd
            primary_exit_status = OPTION_EXIT_STATUS_EXPIRATION_AT_EXIT
            primary_exit_source = OPTION_EXIT_PAYOFF_FALLBACK_INTRINSIC
            primary_used_intrinsic = True
        elif exit_preclose_value is not None:
            exit_option_value_usd = exit_preclose_value
            primary_exit_status = EXIT_PRECLOSE_OPTION_VWAP_STATUS_OK
            primary_exit_source = EXIT_PRECLOSE_OPTION_VWAP_SOURCE
            primary_used_intrinsic = False
        else:
            exit_option_value_usd = exit_intrinsic_usd
            primary_exit_status = (
                OPTION_EXIT_STATUS_EXPIRATION_AT_EXIT
                if expiry <= exit_date
                else OPTION_EXIT_STATUS_MISSING_PRECLOSE_VWAP
            )
            primary_exit_source = OPTION_EXIT_PAYOFF_FALLBACK_INTRINSIC
            primary_used_intrinsic = True
        gross_pnl = exit_option_value_usd - entry_premium_usd
        reaction_o2c_metrics: dict[str, object] = {}
        for label, _, _ in POST_OPEN_OPTION_VWAP_WINDOWS:
            reaction_o2c_metrics.update(
                _reaction_o2c_window_metrics(
                    label=label,
                    post_open_metrics=post_open_metrics,
                    exit_option_value_usd=exit_option_value_usd,
                    gross_c2c_pnl_usd=gross_pnl,
                    haircut_fraction=haircut_fraction,
                )
            )
        rows.append(
            {
                "event_id": event_id,
                "ticker": event_row["ticker"],
                "entry_date": event_row["entry_date"],
                "exit_date": exit_date,
                "expiration": expiry,
                "strike": strike,
                "call_options_ticker": call_options_ticker,
                "put_options_ticker": put_options_ticker,
                "entry_premium_usd": entry_premium_usd,
                "entry_price_method": selected["entry_price_method"],
                "exit_option_value_usd": exit_option_value_usd,
                **exit_preclose_metrics,
                "exit_intrinsic_usd": exit_intrinsic_usd,
                "open_after": open_after,
                "c2o_exit_intrinsic_usd": c2o_exit_intrinsic_usd,
                "gross_c2o_intrinsic_proxy_pnl_usd": gross_c2o_intrinsic_proxy_pnl,
                "c2o_haircut_pnl_usd": c2o_haircut_pnl,
                "c2o_proxy_pnl_source": C2O_PROXY_PNL_SOURCE_INTRINSIC_OPEN,
                "c2o_proxy_pnl_status": c2o_status,
                **post_open_metrics,
                **reaction_o2c_metrics,
                "underlying_exit_price_source": UNDERLYING_EXIT_PRICE_SOURCE,
                "option_exit_price_source": primary_exit_source,
                "option_exit_price_status": primary_exit_status,
                "used_intrinsic_fallback": primary_used_intrinsic,
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
        "main_dte_5_14_contracts": int(proxy_prices["is_main_dte_5_14"].sum())
        if "is_main_dte_5_14" in proxy_prices
        else 0,
        "robustness_dte_3_21_contracts": int(proxy_prices["is_robustness_dte_3_21"].sum())
        if "is_robustness_dte_3_21" in proxy_prices
        else 0,
        "proxy_contracts_with_local_iv": int(proxy_prices["local_iv"].notna().sum())
        if "local_iv" in proxy_prices
        else 0,
        "straddle_diagnostics_rows": int(len(straddle_diagnostics)),
        "mean_gross_proxy_pnl_usd": float(straddle_diagnostics["gross_proxy_pnl_usd"].mean())
        if not straddle_diagnostics.empty
        else None,
        "mean_gross_exit_option_vwap_preclose_15m_proxy_pnl_usd": float(
            straddle_diagnostics["gross_exit_option_vwap_preclose_15m_proxy_pnl_usd"].mean()
        )
        if not straddle_diagnostics.empty
        and "gross_exit_option_vwap_preclose_15m_proxy_pnl_usd" in straddle_diagnostics
        else None,
        "mean_haircut_pnl_usd": float(straddle_diagnostics["haircut_pnl_usd"].mean())
        if not straddle_diagnostics.empty
        else None,
        "mean_gross_c2o_intrinsic_proxy_pnl_usd": float(
            straddle_diagnostics["gross_c2o_intrinsic_proxy_pnl_usd"].mean()
        )
        if not straddle_diagnostics.empty
        and "gross_c2o_intrinsic_proxy_pnl_usd" in straddle_diagnostics
        else None,
        "mean_gross_post_open_option_vwap_0_5_proxy_pnl_usd": float(
            straddle_diagnostics["gross_post_open_option_vwap_0_5_proxy_pnl_usd"].mean()
        )
        if not straddle_diagnostics.empty
        and "gross_post_open_option_vwap_0_5_proxy_pnl_usd" in straddle_diagnostics
        else None,
        "mean_gross_post_open_option_vwap_5_15_proxy_pnl_usd": float(
            straddle_diagnostics["gross_post_open_option_vwap_5_15_proxy_pnl_usd"].mean()
        )
        if not straddle_diagnostics.empty
        and "gross_post_open_option_vwap_5_15_proxy_pnl_usd" in straddle_diagnostics
        else None,
        "mean_gross_reaction_o2c_option_vwap_5_15_to_c2c_exit_proxy_pnl_usd": float(
            straddle_diagnostics[
                "gross_reaction_o2c_option_vwap_5_15_to_c2c_exit_proxy_pnl_usd"
            ].mean()
        )
        if not straddle_diagnostics.empty
        and "gross_reaction_o2c_option_vwap_5_15_to_c2c_exit_proxy_pnl_usd" in straddle_diagnostics
        else None,
        "mean_option_proxy_decomposition_residual_5_15_usd": float(
            straddle_diagnostics["option_proxy_decomposition_residual_5_15_usd"].mean()
        )
        if not straddle_diagnostics.empty
        and "option_proxy_decomposition_residual_5_15_usd" in straddle_diagnostics
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

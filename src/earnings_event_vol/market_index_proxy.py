from __future__ import annotations

import urllib.parse
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import date, datetime
from typing import Any, cast

import httpx
import numpy as np
import pandas as pd
from scipy.optimize import brentq

from earnings_event_vol.backtest import black_scholes_price
from earnings_event_vol.config import ProjectConfig
from earnings_event_vol.massive import parse_massive_option_ticker, read_secret_file
from earnings_event_vol.schemas import OptionRight
from earnings_event_vol.trade_proxy import (
    TRADE_PROXY_PANEL_GRADE,
    _get_json_with_retries,
    filter_pre_cutoff_buffer,
    safe_exception_text,
    select_preclose_entry_proxy_price,
)

MARKET_INDEX_SYMBOLS = ("SPY", "QQQ")
MARKET_INDEX_SECOND_ROUTE = "massive_rest_second_aggs"
MARKET_INDEX_SECOND_SCHEMA_VERSION = "v1.0"
MARKET_INDEX_TARGET_DTES = (7, 14, 30)
MARKET_INDEX_MAX_STRIKES_PER_EXPIRY = 3

MARKET_INDEX_DAILY_SURFACE_FEATURES = [
    "atm_iv_proxy",
    "iv_skew_proxy",
    "iv_butterfly_proxy",
    "term_slope_proxy",
    "straddle_premium_to_spot",
    "valid_pair_count",
    "surface_missing_rate",
    "option_volume_sum",
    "option_transactions_sum",
]


@dataclass(frozen=True)
class UnderlyingSecondFeatures:
    status: str
    close: float | None
    vwap: float | None
    return_in_buffer: float | None
    volume_sum: int
    transactions_sum: int
    rows_in_buffer: int


def _api_key(config: ProjectConfig) -> str:  # pragma: no cover
    secret = read_secret_file(config.massive_api_key_file)
    if not secret:
        raise ValueError("MASSIVE_API_KEY_FILE is not configured or empty.")
    return secret


def _to_datetime(value: object) -> datetime:  # pragma: no cover
    timestamp = pd.Timestamp(value)
    if timestamp.tzinfo is None:
        raise ValueError("timestamp must be timezone-aware")
    return cast(datetime, timestamp.to_pydatetime())


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
    except (OverflowError, RuntimeError, ValueError):
        return None


def fetch_massive_underlying_second_aggregates(
    config: ProjectConfig,
    *,
    ticker: str,
    trade_date: date,
    limit: int = 50_000,
    timeout_seconds: float | None = None,
) -> pd.DataFrame:  # pragma: no cover
    """Fetch 1-second trade aggregates for an underlying ticker.

    These are trade OHLCV bars, not quotes or NBBO.
    """
    encoded_ticker = urllib.parse.quote(ticker.upper(), safe="")
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
            )
    except httpx.HTTPError as exc:
        raise RuntimeError(safe_exception_text(exc)) from None
    return pd.DataFrame(payload.get("results", []))


def normalize_underlying_second_aggregates(raw: pd.DataFrame, *, ticker: str) -> pd.DataFrame:
    columns = [
        "ticker",
        "timestamp_utc",
        "timestamp_et",
        "underlying_open",
        "underlying_high",
        "underlying_low",
        "underlying_close",
        "underlying_vwap",
        "volume",
        "transactions",
        "source_dataset",
    ]
    if raw.empty:
        return pd.DataFrame(columns=columns)
    required = {"t", "o", "h", "l", "c", "v", "vw", "n"}
    missing = sorted(required - set(raw.columns))
    if missing:
        raise ValueError(f"underlying second aggregate frame missing required columns: {missing}")
    out = pd.DataFrame(
        {
            "ticker": ticker.upper(),
            "timestamp_utc": pd.to_datetime(raw["t"], unit="ms", utc=True),
            "underlying_open": pd.to_numeric(raw["o"], errors="coerce"),
            "underlying_high": pd.to_numeric(raw["h"], errors="coerce"),
            "underlying_low": pd.to_numeric(raw["l"], errors="coerce"),
            "underlying_close": pd.to_numeric(raw["c"], errors="coerce"),
            "underlying_vwap": pd.to_numeric(raw["vw"], errors="coerce"),
            "volume": pd.to_numeric(raw["v"], errors="coerce").fillna(0).astype(int),
            "transactions": pd.to_numeric(raw["n"], errors="coerce").fillna(0).astype(int),
            "source_dataset": MARKET_INDEX_SECOND_ROUTE,
        }
    )
    out["timestamp_et"] = out["timestamp_utc"].dt.tz_convert("America/New_York")
    return out[columns]


def select_underlying_second_features(
    bars: pd.DataFrame,
    *,
    cutoff_timestamp: datetime,
    buffer_minutes: int,
    lookback_seconds: int,
) -> UnderlyingSecondFeatures:
    if bars.empty:
        return UnderlyingSecondFeatures("no_second_bars", None, None, None, 0, 0, 0)
    filtered = filter_pre_cutoff_buffer(
        bars,
        cutoff_timestamp=cutoff_timestamp,
        buffer_minutes=buffer_minutes,
    )
    if filtered.empty:
        return UnderlyingSecondFeatures("no_bars_in_cutoff_buffer", None, None, None, 0, 0, 0)
    cutoff = pd.Timestamp(cutoff_timestamp).tz_convert("America/New_York")
    start = cutoff - pd.Timedelta(seconds=lookback_seconds)
    frame = filtered.copy()
    frame["timestamp_et"] = pd.to_datetime(frame["timestamp_et"]).dt.tz_convert("America/New_York")
    frame = frame.loc[frame["timestamp_et"].between(start, cutoff, inclusive="both")].copy()
    if frame.empty:
        return UnderlyingSecondFeatures("no_bars_in_lookback", None, None, None, 0, 0, 0)
    ordered = frame.sort_values("timestamp_et")
    first = ordered.iloc[0]
    last = ordered.iloc[-1]
    first_close = float(cast(Any, first["underlying_close"]))
    last_close = float(cast(Any, last["underlying_close"]))
    return UnderlyingSecondFeatures(
        status="ok",
        close=last_close,
        vwap=float(cast(Any, last["underlying_vwap"])),
        return_in_buffer=(last_close / first_close - 1.0) if first_close > 0 else None,
        volume_sum=int(pd.to_numeric(ordered["volume"], errors="coerce").fillna(0).sum()),
        transactions_sum=int(
            pd.to_numeric(ordered["transactions"], errors="coerce").fillna(0).sum()
        ),
        rows_in_buffer=int(len(ordered)),
    )


def _parse_index_options(
    day_options: pd.DataFrame, *, symbol: str, source_date: date
) -> pd.DataFrame:
    if day_options.empty:
        return pd.DataFrame()
    ticker_text = day_options["ticker"].astype(str)
    mask = ticker_text.str.match(rf"^O:{symbol}\d{{6}}[CP]\d{{8}}$")
    frame = day_options.loc[mask].copy()
    if frame.empty:
        return pd.DataFrame()
    parsed = pd.DataFrame([parse_massive_option_ticker(str(value)) for value in frame["ticker"]])
    out = parsed.copy()
    if "options_ticker" not in out.columns and "option_symbol" in out.columns:
        out["options_ticker"] = out["option_symbol"]
    out["option_close"] = pd.to_numeric(frame["close"], errors="coerce").to_numpy()
    out["option_volume"] = pd.to_numeric(frame["volume"], errors="coerce").fillna(0.0).to_numpy()
    transactions = (
        frame["transactions"]
        if "transactions" in frame.columns
        else pd.Series([0] * len(frame), index=frame.index)
    )
    out["option_transactions"] = pd.to_numeric(transactions, errors="coerce").fillna(0.0).to_numpy()
    out["source_date"] = source_date
    out["dte"] = (pd.to_datetime(out["expiration"]).dt.date - source_date).map(
        lambda delta: delta.days
    )
    return out


def select_market_index_option_candidates(
    day_options: pd.DataFrame,
    *,
    symbol: str,
    source_date: date,
    spot: float,
    dte_min: int = 3,
    dte_max: int = 45,
    target_dtes: Sequence[int] = MARKET_INDEX_TARGET_DTES,
    max_strikes_per_expiry: int = MARKET_INDEX_MAX_STRIKES_PER_EXPIRY,
) -> pd.DataFrame:
    """Select a small ATM/wing option set for an index ETF proxy surface."""
    parsed = _parse_index_options(day_options, symbol=symbol.upper(), source_date=source_date)
    if parsed.empty or spot <= 0 or not np.isfinite(spot):
        return pd.DataFrame()
    frame = parsed.loc[
        pd.to_numeric(parsed["dte"], errors="coerce").between(dte_min, dte_max, inclusive="both")
        & pd.to_numeric(parsed["option_close"], errors="coerce").gt(0)
    ].copy()
    if frame.empty:
        return pd.DataFrame()
    frame["moneyness_abs"] = (pd.to_numeric(frame["strike"], errors="coerce") / spot - 1.0).abs()
    selected_keys: set[tuple[date, float, str]] = set()
    rows: list[pd.DataFrame] = []
    expiries = (
        frame[["expiration", "dte"]]
        .drop_duplicates()
        .assign(_dte=lambda data: pd.to_numeric(data["dte"], errors="coerce"))
    )
    for target in target_dtes:
        expiry_rows = expiries.assign(_distance=(expiries["_dte"] - target).abs()).sort_values(
            ["_distance", "_dte"]
        )
        if expiry_rows.empty:
            continue
        expiry = expiry_rows.iloc[0]["expiration"]
        expiry_frame = frame.loc[frame["expiration"].eq(expiry)].copy()
        pair_counts = expiry_frame.groupby("strike")["right"].nunique()
        paired_strikes = pair_counts.loc[pair_counts.ge(2)].index.tolist()
        if not paired_strikes:
            continue
        strike_frame = (
            pd.DataFrame({"strike": paired_strikes})
            .assign(
                _distance=lambda data: (pd.to_numeric(data["strike"], errors="coerce") - spot).abs()
            )
            .sort_values("_distance")
            .head(max_strikes_per_expiry)
        )
        for strike in strike_frame["strike"].tolist():
            subset = expiry_frame.loc[expiry_frame["strike"].eq(strike)].copy()
            for record in subset.to_dict("records"):
                key = (
                    cast(date, pd.Timestamp(record["expiration"]).date()),
                    float(record["strike"]),
                    str(record["right"]),
                )
                if key not in selected_keys:
                    selected_keys.add(key)
                    rows.append(pd.DataFrame([record]))
    if not rows:
        return pd.DataFrame()
    out = pd.concat(rows, ignore_index=True).drop_duplicates("options_ticker")
    out["index_symbol"] = symbol.upper()
    return out.reset_index(drop=True)


def market_index_surface_features(
    candidates: pd.DataFrame,
    bar_frames: dict[str, pd.DataFrame],
    *,
    symbol: str,
    spot: float | None,
    cutoff_timestamp: datetime,
    lookback_seconds: int,
    price_field: str = "option_vwap",
) -> dict[str, object]:
    """Build event-time index ETF option-surface proxy features from second aggregates."""
    base: dict[str, object] = {
        "index_symbol": symbol.upper(),
        "market_surface_source": MARKET_INDEX_SECOND_ROUTE,
        "market_surface_panel_grade": TRADE_PROXY_PANEL_GRADE,
        "market_surface_candidate_contracts": int(len(candidates)),
        "market_surface_priced_contracts": 0,
        "market_surface_status": "no_candidates" if candidates.empty else "no_prices",
        "market_atm_iv_proxy": np.nan,
        "market_term_slope_proxy": np.nan,
        "market_skew_proxy": np.nan,
        "market_butterfly_proxy": np.nan,
        "market_straddle_premium_to_spot": np.nan,
        "market_valid_pair_count": 0,
        "market_surface_missing_rate": 1.0,
        "market_option_volume_sum": 0,
        "market_option_transactions_sum": 0,
        "market_option_second_rows": 0,
    }
    if candidates.empty or spot is None or not np.isfinite(spot) or spot <= 0:
        return base
    priced_rows: list[dict[str, object]] = []
    for record in candidates.to_dict("records"):
        option_ticker = str(record["options_ticker"])
        selection = select_preclose_entry_proxy_price(
            bar_frames.get(option_ticker, pd.DataFrame()),
            cutoff_timestamp=cutoff_timestamp,
            lookback_seconds=lookback_seconds,
            price_field=price_field,
        )
        if selection.proxy_price is None:
            continue
        row = dict(record)
        row["option_proxy_price"] = selection.proxy_price
        row["proxy_volume"] = selection.proxy_volume
        row["proxy_transactions"] = selection.proxy_transactions
        row["proxy_rows_in_window"] = selection.proxy_rows_in_window
        row["iv_proxy"] = _implied_volatility(
            spot=float(spot),
            strike=float(record["strike"]),
            time_to_expiry=max(float(record["dte"]) / 365.0, 1.0 / 365.0),
            option_price=float(selection.proxy_price),
            right=str(record["right"]),
        )
        priced_rows.append(row)
    if not priced_rows:
        return base
    priced = pd.DataFrame(priced_rows).dropna(subset=["iv_proxy"])
    base["market_surface_priced_contracts"] = int(len(priced))
    base["market_option_volume_sum"] = int(
        pd.to_numeric(priced["proxy_volume"], errors="coerce").sum()
    )
    base["market_option_transactions_sum"] = int(
        pd.to_numeric(priced["proxy_transactions"], errors="coerce").sum()
    )
    base["market_option_second_rows"] = int(
        pd.to_numeric(priced["proxy_rows_in_window"], errors="coerce").sum()
    )
    if priced.empty:
        return base
    pair_counts = priced.groupby(["expiration", "strike"])["right"].nunique()
    valid_pair_count = int(pair_counts.ge(2).sum())
    base["market_valid_pair_count"] = valid_pair_count
    base["market_surface_missing_rate"] = float(
        1.0
        - min(valid_pair_count, max(len(candidates) / 2.0, 1.0)) / max(len(candidates) / 2.0, 1.0)
    )
    pair_rows = priced.loc[
        priced.groupby(["expiration", "strike"])["right"].transform("nunique").ge(2)
    ].copy()
    if pair_rows.empty:
        return base
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
    if not atm_key.empty:
        expiration = atm_key.iloc[0]["expiration"]
        strike = atm_key.iloc[0]["strike"]
        atm = pair_rows.loc[pair_rows["expiration"].eq(expiration) & pair_rows["strike"].eq(strike)]
        call = atm.loc[atm["right"].eq("call")]
        put = atm.loc[atm["right"].eq("put")]
        base["market_atm_iv_proxy"] = float(atm["iv_proxy"].mean())
        if not call.empty and not put.empty:
            base["market_skew_proxy"] = float(put["iv_proxy"].mean() - call["iv_proxy"].mean())
            base["market_straddle_premium_to_spot"] = float(
                (call["option_proxy_price"].mean() + put["option_proxy_price"].mean()) / float(spot)
            )
    near = pair_rows.loc[pair_rows["dte"].between(3, 14, inclusive="both")]
    far = pair_rows.loc[pair_rows["dte"].between(15, 45, inclusive="both")]
    if not near.empty and not far.empty:
        base["market_term_slope_proxy"] = float(far["iv_proxy"].mean() - near["iv_proxy"].mean())
    by_strike = pair_rows.groupby("strike", as_index=False)["iv_proxy"].mean().sort_values("strike")
    if len(by_strike) >= 3:
        middle = int(np.argmin(np.abs(by_strike["strike"].to_numpy(dtype=float) - float(spot))))
        low = max(0, middle - 1)
        high = min(len(by_strike) - 1, middle + 1)
        if low != middle and high != middle:
            base["market_butterfly_proxy"] = float(
                by_strike.iloc[low]["iv_proxy"]
                + by_strike.iloc[high]["iv_proxy"]
                - 2.0 * by_strike.iloc[middle]["iv_proxy"]
            )
    base["market_surface_status"] = "ok"
    return base


def prefix_market_index_features(features: dict[str, object], *, symbol: str) -> dict[str, object]:
    prefix = symbol.lower()
    return {
        f"{prefix}_second_{key}": value for key, value in features.items() if key != "index_symbol"
    }

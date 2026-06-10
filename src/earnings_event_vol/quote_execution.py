from __future__ import annotations

import gzip
import hashlib
import io
import json
import re
import subprocess
import time as time_module
import urllib.parse
from collections.abc import Iterable, Iterator, Mapping, Sequence
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from datetime import date, timedelta
from pathlib import Path
from typing import Any, cast

import httpx
import numpy as np
import pandas as pd
from scipy.optimize import brentq

from earnings_event_vol.backtest import black_scholes_price
from earnings_event_vol.config import ProjectConfig
from earnings_event_vol.events import market_close_timestamp
from earnings_event_vol.massive import (
    massive_flat_file_aws_env,
    option_quotes_flat_file_key,
    read_secret_file,
)
from earnings_event_vol.schemas import IVARFailureReason, OptionRight
from earnings_event_vol.variance import year_fraction

CONTRACT_MULTIPLIER = 100.0
QUOTE_SOURCE_REST = "rest"
QUOTE_SOURCE_FLAT_FILE = "flat-file"
QUOTE_ROUTE_REST_TARGETED = "massive_quotes_v3_rest_targeted"
QUOTE_ROUTE_FLAT_FILE_FILTERED = "massive_quotes_v1_flat_file_filtered"
QUOTE_ROUTE_CSV_FIXTURE = "local_quotes_csv_filtered"
QUOTE_IVAR_METHOD = "entry_straddle_premium_total_variance_proxy"
QUOTE_IVAR_CLAIM_SCOPE = "diagnostic_quote_premium_proxy_not_model_feature"
QUOTE_IV_SURFACE_METHOD = "entry_quote_black_scholes_iv_surface"
QUOTE_IV_SURFACE_CLAIM_SCOPE = "bounded_quote_iv_surface_diagnostic_not_full_nbbo_surface"
QUOTE_SURFACE_IVAR_METHOD = "entry_quote_iv_surface_total_variance_extraction"
QUOTE_SURFACE_IVAR_CLAIM_SCOPE = "bounded_quote_iv_surface_ivar_diagnostic_not_full_nbbo_surface"
QUOTE_STATUS_OK = "ok"
QUOTE_STATUS_MISSING = "missing_quote"
QUOTE_STATUS_INVALID = "invalid_bid_ask"
QUOTE_STATUS_STALE = "stale_quote"
QUOTE_STATUS_WIDE = "wide_spread"
QueryParamValue = str | int | float | bool | None
_SECRET_QUERY_PATTERN = re.compile(r"(?i)((?:apiKey|api_key)=)[^&\s)]+")


@dataclass(frozen=True)
class QuoteExtractionReport:
    ok: bool
    route: str
    metadata_only: bool
    raw_full_day_files_written: bool
    request_rows: int
    event_count: int
    quote_rows_scanned: int
    quote_rows_matched: int
    dates: tuple[str, ...]
    output_paths: dict[str, str]
    quote_workers: int = 1
    event_offset: int = 0
    batch_label: str | None = None

    def as_dict(self) -> dict[str, object]:
        return {
            "ok": self.ok,
            "route": self.route,
            "metadata_only": self.metadata_only,
            "raw_full_day_files_written": self.raw_full_day_files_written,
            "request_rows": self.request_rows,
            "event_count": self.event_count,
            "quote_rows_scanned": self.quote_rows_scanned,
            "quote_rows_matched": self.quote_rows_matched,
            "dates": list(self.dates),
            "output_paths": self.output_paths,
            "quote_workers": self.quote_workers,
            "event_offset": self.event_offset,
            "batch_label": self.batch_label,
        }


def _to_datetime_et(value: object) -> pd.Timestamp:
    timestamp = pd.Timestamp(value)
    if timestamp.tzinfo is None:
        timestamp = timestamp.tz_localize("America/New_York")
    return timestamp.tz_convert("America/New_York")


def _to_date(value: object) -> date:
    parsed = pd.Timestamp(value)
    return date(parsed.year, parsed.month, parsed.day)


def _api_key(config: ProjectConfig) -> str:
    secret = read_secret_file(config.massive_api_key_file)
    if not secret:
        raise ValueError("MASSIVE_API_KEY_FILE is not configured or empty.")
    return secret


def _redact_secret_query_params(text: str) -> str:
    return _SECRET_QUERY_PATTERN.sub(r"\1<redacted>", text)


def _safe_exception_text(exc: Exception, *, max_chars: int = 300) -> str:
    if isinstance(exc, httpx.HTTPStatusError):
        body = _redact_secret_query_params(" ".join(exc.response.text.strip().split()))
        suffix = f": {body[:200]}" if body else ""
        return _redact_secret_query_params(f"HTTP {exc.response.status_code}{suffix}")[:max_chars]
    return _redact_secret_query_params(str(exc))[:max_chars]


def _retryable_http_error(exc: httpx.HTTPStatusError) -> bool:
    return exc.response.status_code in {429, 500, 502, 503, 504}


def _get_json_with_retries(
    client: httpx.Client,
    url: str,
    *,
    params: Mapping[str, QueryParamValue],
    config: ProjectConfig,
) -> dict[str, Any]:
    attempts = max(1, int(config.massive_max_retries) + 1)
    last_exc: Exception | None = None
    for attempt in range(attempts):
        try:
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
        if attempt < attempts - 1 and config.massive_retry_backoff_seconds > 0:
            time_module.sleep(float(config.massive_retry_backoff_seconds) * (2**attempt))
    assert last_exc is not None
    raise last_exc


def _rest_timestamp_param(value: object) -> str:
    return str(_to_datetime_et(value).tz_convert("UTC").isoformat()).replace("+00:00", "Z")


def _safe_cache_component(value: object) -> str:
    text = str(value)
    return "".join(ch if ch.isalnum() else "_" for ch in text).strip("_") or "blank"


def _rest_quote_cache_path(
    cache_dir: Path,
    *,
    options_ticker: str,
    window_start: object,
    window_end: object,
) -> Path:
    start = _to_datetime_et(window_start)
    end = _to_datetime_et(window_end)
    digest = hashlib.sha256(
        f"{options_ticker}|{start.isoformat()}|{end.isoformat()}".encode()
    ).hexdigest()[:20]
    return (
        cache_dir
        / f"quote_date={start.date().isoformat()}"
        / f"options_ticker={_safe_cache_component(options_ticker)}"
        / f"window={digest}.parquet"
    )


def _load_cached_normalized_quotes(path: Path) -> pd.DataFrame | None:
    if not path.exists() or path.stat().st_size <= 0:
        return None
    try:
        cached = pd.read_parquet(path)
    except (OSError, ValueError):
        return None
    return cached if isinstance(cached, pd.DataFrame) else None


def _write_cached_normalized_quotes(path: Path, frame: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_parquet(path, index=False)


def fetch_massive_option_quote_window_rest(
    client: httpx.Client,
    config: ProjectConfig,
    *,
    options_ticker: str,
    window_start: object,
    window_end: object,
    limit: int = 50_000,
) -> pd.DataFrame:
    encoded_ticker = urllib.parse.quote(options_ticker, safe="")
    url = f"{config.massive_base_url.rstrip('/')}/v3/quotes/{encoded_ticker}"
    api_key = _api_key(config)
    params: dict[str, QueryParamValue] = {
        "timestamp.gte": _rest_timestamp_param(window_start),
        "timestamp.lte": _rest_timestamp_param(window_end),
        "sort": "timestamp",
        "order": "asc",
        "limit": int(limit),
        "apiKey": api_key,
    }
    rows: list[dict[str, Any]] = []
    page_count = 0
    while True:
        page_count += 1
        if page_count > 1000:
            raise RuntimeError(
                f"Massive quote REST pagination exceeded safety limit: {options_ticker}"
            )
        try:
            payload = _get_json_with_retries(client, url, params=params, config=config)
        except httpx.HTTPError as exc:
            raise RuntimeError(_safe_exception_text(exc)) from None
        results = payload.get("results", [])
        if isinstance(results, list):
            rows.extend(cast(list[dict[str, Any]], results))
        next_url = payload.get("next_url")
        if not next_url:
            break
        url = str(next_url)
        if url.startswith("/"):
            url = f"{config.massive_base_url.rstrip('/')}{url}"
        params = {"apiKey": api_key}
    raw = pd.DataFrame(rows)
    if raw.empty:
        return _empty_normalized_quotes_frame()
    raw["ticker"] = options_ticker
    return normalize_option_quote_rows(raw)


def _load_or_fetch_rest_quote_window(
    client: httpx.Client,
    config: ProjectConfig,
    *,
    request: pd.Series,
    cache_dir: Path,
    limit: int,
) -> pd.DataFrame:
    options_ticker = str(request["options_ticker"])
    cache_path = _rest_quote_cache_path(
        cache_dir,
        options_ticker=options_ticker,
        window_start=request["window_start"],
        window_end=request["window_end"],
    )
    cached = _load_cached_normalized_quotes(cache_path)
    if cached is not None:
        return cached
    fetched = fetch_massive_option_quote_window_rest(
        client,
        config,
        options_ticker=options_ticker,
        window_start=request["window_start"],
        window_end=request["window_end"],
        limit=limit,
    )
    _write_cached_normalized_quotes(cache_path, fetched)
    return fetched


def _empty_normalized_quotes_frame() -> pd.DataFrame:
    return pd.DataFrame(
        columns=[
            "options_ticker",
            "quote_date",
            "quote_timestamp_et",
            "bid",
            "ask",
            "mid",
            "spread",
            "spread_over_mid",
        ]
    )


def _empty_quote_window_marks_frame() -> pd.DataFrame:
    return pd.DataFrame(
        columns=[
            "event_id",
            "ticker",
            "announcement_date",
            "announcement_timing",
            "entry_date",
            "exit_date",
            "s_before",
            "close_before",
            "spot",
            "options_ticker",
            "quote_date",
            "window_label",
            "window_start",
            "window_end",
            "expiration",
            "strike",
            "right",
            "quote_count",
            "quote_timestamp_et",
            "quote_age_seconds",
            "bid",
            "ask",
            "mid",
            "spread",
            "spread_over_mid",
            "stale_quote",
            "wide_spread",
            "quote_status",
            "quote_score",
        ]
    )


def _empty_quote_execution_legs_frame() -> pd.DataFrame:
    return pd.DataFrame(
        columns=[
            "event_id",
            "ticker",
            "announcement_date",
            "announcement_timing",
            "entry_date",
            "exit_date",
            "expiration",
            "strike",
            "right",
            "options_ticker",
            "window_label",
            "execution_side",
            "quote_status",
            "quote_score",
            "quote_timestamp_et",
            "quote_age_seconds",
            "bid",
            "ask",
            "mid",
            "spread_over_mid",
            "contract_multiplier",
            "long_fill_price",
            "long_fill_value_usd",
            "mid_fill_value_usd",
        ]
    )


def _empty_quote_straddle_execution_frame() -> pd.DataFrame:
    return pd.DataFrame(
        columns=[
            "event_id",
            "ticker",
            "announcement_date",
            "announcement_timing",
            "entry_date",
            "exit_date",
            "expiration",
            "strike",
            "spot",
            "entry_call_status",
            "entry_put_status",
            "exit_call_status",
            "exit_put_status",
            "entry_ask_cost_usd",
            "entry_mid_cost_usd",
            "entry_bid_cost_usd",
            "exit_bid_value_usd",
            "exit_mid_value_usd",
            "exit_ask_value_usd",
            "quote_bidask_pnl_usd",
            "quote_mid_pnl_usd",
            "quote_entry_mid_premium_pct_spot",
            "quote_entry_ask_premium_pct_spot",
            "quote_premium_var_mid",
            "quote_premium_var_ask",
            "complete_bidask_pair",
            "complete_mid_pair",
            "paper_grade_execution",
        ]
    )


def _empty_quote_ivar_event_frame() -> pd.DataFrame:
    return pd.DataFrame(
        columns=[
            "event_id",
            "ticker",
            "announcement_date",
            "announcement_timing",
            "entry_date",
            "exit_date",
            "quote_ivar_method",
            "quote_ivar_claim_scope",
            "expiry_candidate_count",
            "quote_mid_ivar_event",
            "quote_ask_ivar_event",
            "quote_mid_ivar_failure_reason",
            "quote_ask_ivar_failure_reason",
            "t1",
            "t2",
            "dte_1",
            "dte_2",
            "expiration_1",
            "expiration_2",
            "expiry_gap_days",
            "mid_total_variance_1",
            "mid_total_variance_2",
            "ask_total_variance_1",
            "ask_total_variance_2",
            "strike_1",
            "strike_2",
            "spot_1",
            "spot_2",
            "mid_complete_pair_1",
            "mid_complete_pair_2",
            "bidask_complete_pair_1",
            "bidask_complete_pair_2",
            "paper_grade_quote_ivar_mid",
            "paper_grade_quote_ivar_ask",
        ]
    )


def _empty_quote_iv_surface_frame() -> pd.DataFrame:
    return pd.DataFrame(
        columns=[
            "event_id",
            "ticker",
            "announcement_date",
            "announcement_timing",
            "entry_date",
            "expiration",
            "dte",
            "time_to_expiry",
            "strike",
            "right",
            "options_ticker",
            "spot",
            "quote_status",
            "quote_timestamp_et",
            "quote_age_seconds",
            "bid",
            "ask",
            "mid",
            "spread_over_mid",
            "quote_bid_iv",
            "quote_mid_iv",
            "quote_ask_iv",
            "quote_bid_iv_failure_reason",
            "quote_mid_iv_failure_reason",
            "quote_ask_iv_failure_reason",
            "paper_grade_quote_iv_mid",
            "paper_grade_quote_iv_bidask",
            "quote_iv_surface_method",
            "quote_iv_surface_claim_scope",
        ]
    )


def _empty_quote_iv_surface_summary_frame() -> pd.DataFrame:
    return pd.DataFrame(
        columns=[
            "event_id",
            "ticker",
            "announcement_date",
            "announcement_timing",
            "entry_date",
            "expiration",
            "dte",
            "time_to_expiry",
            "strike",
            "spot",
            "call_status",
            "put_status",
            "call_mid_iv",
            "put_mid_iv",
            "call_bid_iv",
            "put_bid_iv",
            "call_ask_iv",
            "put_ask_iv",
            "mean_mid_iv",
            "mean_bid_iv",
            "mean_ask_iv",
            "skew_put_minus_call_mid_iv",
            "quote_mid_total_variance",
            "quote_bid_total_variance",
            "quote_ask_total_variance",
            "complete_mid_surface_pair",
            "complete_bidask_surface_pair",
            "paper_grade_quote_iv_surface_pair",
            "quote_iv_surface_method",
            "quote_iv_surface_claim_scope",
        ]
    )


def _empty_quote_surface_ivar_event_frame() -> pd.DataFrame:
    return pd.DataFrame(
        columns=[
            "event_id",
            "ticker",
            "announcement_date",
            "announcement_timing",
            "entry_date",
            "quote_surface_ivar_method",
            "quote_surface_ivar_claim_scope",
            "surface_pair_count",
            "quote_surface_mid_ivar_event",
            "quote_surface_bid_ivar_event",
            "quote_surface_ask_ivar_event",
            "quote_surface_mid_ivar_failure_reason",
            "quote_surface_bid_ivar_failure_reason",
            "quote_surface_ask_ivar_failure_reason",
            "t1",
            "t2",
            "dte_1",
            "dte_2",
            "expiration_1",
            "expiration_2",
            "expiry_gap_days",
            "mid_total_variance_1",
            "mid_total_variance_2",
            "bid_total_variance_1",
            "bid_total_variance_2",
            "ask_total_variance_1",
            "ask_total_variance_2",
            "strike_1",
            "strike_2",
            "spot_1",
            "spot_2",
            "mid_complete_pair_1",
            "mid_complete_pair_2",
            "bidask_complete_pair_1",
            "bidask_complete_pair_2",
            "paper_grade_quote_surface_ivar_mid",
            "paper_grade_quote_surface_ivar_bid",
            "paper_grade_quote_surface_ivar_ask",
        ]
    )


def _timestamp_from_sip(value: object) -> pd.Timestamp:
    if value is None or pd.isna(value):
        return pd.NaT
    text = str(value).strip()
    if text == "":
        return pd.NaT
    try:
        numeric = float(text)
    except ValueError:
        return pd.Timestamp(text, tz="UTC").tz_convert("America/New_York")
    if not np.isfinite(numeric):
        return pd.NaT
    if numeric > 1e17:
        timestamp = pd.to_datetime(int(numeric), unit="ns", utc=True)
    elif numeric > 1e14:
        timestamp = pd.to_datetime(int(numeric), unit="us", utc=True)
    elif numeric > 1e11:
        timestamp = pd.to_datetime(int(numeric), unit="ms", utc=True)
    else:
        timestamp = pd.to_datetime(numeric, unit="s", utc=True)
    return pd.Timestamp(timestamp).tz_convert("America/New_York")


def normalize_option_quote_rows(
    raw: pd.DataFrame, *, quote_date: date | None = None
) -> pd.DataFrame:
    """Normalize Massive quotes_v1-like rows to the project quote mark schema."""
    if raw.empty:
        return _empty_normalized_quotes_frame()
    ticker_col = "ticker" if "ticker" in raw.columns else "options_ticker"
    bid_col = "bid_price" if "bid_price" in raw.columns else "bid"
    ask_col = "ask_price" if "ask_price" in raw.columns else "ask"
    ts_col = (
        "sip_timestamp"
        if "sip_timestamp" in raw.columns
        else "quote_timestamp"
        if "quote_timestamp" in raw.columns
        else "timestamp"
        if "timestamp" in raw.columns
        else "quote_timestamp_et"
    )
    missing = [name for name in (ticker_col, bid_col, ask_col, ts_col) if name not in raw.columns]
    if missing:
        raise ValueError(f"quote frame missing required columns: {missing}")
    out = pd.DataFrame(
        {
            "options_ticker": raw[ticker_col].astype(str),
            "bid": pd.to_numeric(raw[bid_col], errors="coerce"),
            "ask": pd.to_numeric(raw[ask_col], errors="coerce"),
        }
    )
    if ts_col == "sip_timestamp":
        out["quote_timestamp_et"] = raw[ts_col].map(_timestamp_from_sip)
    else:
        out["quote_timestamp_et"] = pd.to_datetime(raw[ts_col], errors="coerce")
        out["quote_timestamp_et"] = out["quote_timestamp_et"].map(
            lambda value: pd.NaT if pd.isna(value) else _to_datetime_et(value)
        )
    if "quote_date" in raw.columns:
        out["quote_date"] = pd.to_datetime(raw["quote_date"], errors="coerce").dt.date
    elif quote_date is not None:
        out["quote_date"] = quote_date
    else:
        out["quote_date"] = out["quote_timestamp_et"].map(
            lambda value: None if pd.isna(value) else pd.Timestamp(value).date()
        )
    out["mid"] = (out["bid"] + out["ask"]) / 2.0
    out["spread"] = out["ask"] - out["bid"]
    out["spread_over_mid"] = out["spread"] / out["mid"].replace(0.0, np.nan)
    return out


def build_quote_window_requests(
    contracts: pd.DataFrame,
    windows: pd.DataFrame,
    *,
    entry_lookback_seconds: int = 900,
    exit_lookback_seconds: int = 900,
    max_events: int | None = None,
    event_offset: int = 0,
) -> pd.DataFrame:
    """Build event/contract/window requests for quote extraction."""
    if event_offset < 0:
        raise ValueError("event_offset must be non-negative")
    required_contracts = {"event_id", "options_ticker", "entry_date", "exit_date"}
    required_windows = {"event_id"}
    missing_contracts = sorted(required_contracts - set(contracts.columns))
    missing_windows = sorted(required_windows - set(windows.columns))
    if missing_contracts:
        raise ValueError(f"contract frame missing required columns: {missing_contracts}")
    if missing_windows:
        raise ValueError(f"window frame missing required columns: {missing_windows}")
    event_ids = windows["event_id"].astype(str).drop_duplicates().tolist()
    if event_offset:
        event_ids = event_ids[int(event_offset) :]
    if max_events is not None:
        event_ids = event_ids[: int(max_events)]
    event_filter = set(event_ids)
    base = contracts.loc[contracts["event_id"].astype(str).isin(event_filter)].copy()
    if "eligible_for_quote_pool" in base.columns:
        base = base.loc[base["eligible_for_quote_pool"].astype(bool)].copy()
    if base.empty:
        return pd.DataFrame(
            columns=[
                "event_id",
                "ticker",
                "entry_date",
                "exit_date",
                "options_ticker",
                "quote_date",
                "window_label",
                "window_start",
                "window_end",
                "expiration",
                "strike",
                "right",
            ]
        )
    window_cols = [
        col
        for col in (
            "event_id",
            "ticker",
            "announcement_date",
            "announcement_timing",
            "event_entry_timestamp",
            "s_before",
            "close_before",
            "spot",
        )
        if col in windows.columns
    ]
    merged = base.merge(
        windows[window_cols].drop_duplicates("event_id"),
        on="event_id",
        how="left",
        suffixes=("", "_window"),
    )
    rows: list[dict[str, object]] = []
    for row in merged.to_dict("records"):
        event_entry_raw = row.get("event_entry_timestamp")
        entry_end = (
            _to_datetime_et(event_entry_raw)
            if event_entry_raw is not None and not pd.isna(event_entry_raw)
            else market_close_timestamp(_to_date(row["entry_date"]))
        )
        entry_start = entry_end - pd.Timedelta(seconds=int(entry_lookback_seconds))
        exit_end = market_close_timestamp(_to_date(row["exit_date"]))
        exit_start = exit_end - pd.Timedelta(seconds=int(exit_lookback_seconds))
        common = {
            "event_id": str(row["event_id"]),
            "ticker": row.get("ticker") or row.get("ticker_window"),
            "announcement_date": row.get("announcement_date"),
            "announcement_timing": row.get("announcement_timing"),
            "entry_date": row.get("entry_date"),
            "exit_date": row.get("exit_date"),
            "s_before": row.get("s_before"),
            "close_before": row.get("close_before"),
            "spot": row.get("spot"),
            "options_ticker": str(row["options_ticker"]),
            "expiration": row.get("expiration"),
            "strike": row.get("strike"),
            "right": row.get("right"),
        }
        rows.append(
            {
                **common,
                "quote_date": entry_end.date(),
                "window_label": "entry_preclose_15m",
                "window_start": entry_start,
                "window_end": entry_end,
            }
        )
        rows.append(
            {
                **common,
                "quote_date": exit_end.date(),
                "window_label": "exit_preclose_15m",
                "window_start": exit_start,
                "window_end": exit_end,
            }
        )
    out = pd.DataFrame(rows)
    out["quote_date"] = pd.to_datetime(out["quote_date"]).dt.date
    return out


def _quote_score(status: str) -> float:
    if status == QUOTE_STATUS_OK:
        return 1.0
    if status in {QUOTE_STATUS_STALE, QUOTE_STATUS_WIDE}:
        return 0.5
    return 0.0


def quote_confidence_band(score: float | int | None) -> str:
    if score is None or not np.isfinite(float(score)):
        return "missing"
    value = float(score)
    if value >= 0.8:
        return "high"
    if value >= 0.5:
        return "medium"
    if value > 0:
        return "low"
    return "missing"


def build_quote_window_marks(
    quotes: pd.DataFrame,
    requests: pd.DataFrame,
    *,
    stale_seconds: int = 60,
    wide_spread_threshold: float = 0.25,
) -> pd.DataFrame:
    """Select the latest valid quote in each requested event/contract/window."""
    if requests.empty:
        return _empty_quote_window_marks_frame()
    normalized = quotes.copy()
    if "quote_timestamp_et" in normalized.columns:
        normalized["quote_timestamp_et"] = normalized["quote_timestamp_et"].map(
            lambda value: pd.NaT if pd.isna(value) else _to_datetime_et(value)
        )
    rows: list[dict[str, object]] = []
    for request in requests.to_dict("records"):
        option_ticker = str(request["options_ticker"])
        start = _to_datetime_et(request["window_start"])
        end = _to_datetime_et(request["window_end"])
        quote_date = _to_date(request["quote_date"])
        subset = normalized.loc[
            normalized["options_ticker"].astype(str).eq(option_ticker)
            & pd.to_datetime(normalized["quote_date"], errors="coerce").dt.date.eq(quote_date)
        ].copy()
        if not subset.empty:
            ts = subset["quote_timestamp_et"].map(lambda value: _to_datetime_et(value))
            subset = subset.loc[ts.ge(start) & ts.le(end)].copy()
            subset["quote_timestamp_et"] = ts.loc[subset.index]
        valid = subset.loc[
            np.isfinite(pd.to_numeric(subset.get("bid"), errors="coerce"))
            & np.isfinite(pd.to_numeric(subset.get("ask"), errors="coerce"))
            & pd.to_numeric(subset.get("bid"), errors="coerce").ge(0)
            & pd.to_numeric(subset.get("ask"), errors="coerce").gt(0)
            & pd.to_numeric(subset.get("ask"), errors="coerce").ge(
                pd.to_numeric(subset.get("bid"), errors="coerce")
            )
        ].copy()
        base = dict(request)
        base["window_start"] = start
        base["window_end"] = end
        base["quote_count"] = int(len(subset))
        if subset.empty:
            rows.append(
                {
                    **base,
                    "quote_timestamp_et": pd.NaT,
                    "quote_age_seconds": np.nan,
                    "bid": np.nan,
                    "ask": np.nan,
                    "mid": np.nan,
                    "spread": np.nan,
                    "spread_over_mid": np.nan,
                    "stale_quote": False,
                    "wide_spread": False,
                    "quote_status": QUOTE_STATUS_MISSING,
                    "quote_score": 0.0,
                }
            )
            continue
        if valid.empty:
            rows.append(
                {
                    **base,
                    "quote_timestamp_et": pd.NaT,
                    "quote_age_seconds": np.nan,
                    "bid": np.nan,
                    "ask": np.nan,
                    "mid": np.nan,
                    "spread": np.nan,
                    "spread_over_mid": np.nan,
                    "stale_quote": False,
                    "wide_spread": False,
                    "quote_status": QUOTE_STATUS_INVALID,
                    "quote_score": 0.0,
                }
            )
            continue
        selected = valid.sort_values("quote_timestamp_et").iloc[-1]
        bid = float(selected["bid"])
        ask = float(selected["ask"])
        mid = (bid + ask) / 2.0
        spread = ask - bid
        spread_over_mid = np.nan if mid <= 0 else spread / mid
        age = (end - _to_datetime_et(selected["quote_timestamp_et"])).total_seconds()
        stale = bool(age > float(stale_seconds))
        wide = bool(np.isfinite(spread_over_mid) and spread_over_mid > wide_spread_threshold)
        status = QUOTE_STATUS_OK
        if stale:
            status = QUOTE_STATUS_STALE
        elif wide:
            status = QUOTE_STATUS_WIDE
        rows.append(
            {
                **base,
                "quote_timestamp_et": selected["quote_timestamp_et"],
                "quote_age_seconds": float(age),
                "bid": bid,
                "ask": ask,
                "mid": mid,
                "spread": spread,
                "spread_over_mid": spread_over_mid,
                "stale_quote": stale,
                "wide_spread": wide,
                "quote_status": status,
                "quote_score": _quote_score(status),
            }
        )
    return pd.DataFrame(rows)


def build_execution_confidence_panel(
    marks: pd.DataFrame,
    *,
    quote_execution_route: str = QUOTE_ROUTE_FLAT_FILE_FILTERED,
) -> pd.DataFrame:
    """Aggregate leg/window quote diagnostics to event-level execution confidence."""
    if marks.empty:
        return pd.DataFrame(
            columns=[
                "event_id",
                "quote_execution_route",
                "required_quote_marks",
                "ok_quote_marks",
                "execution_confidence_score",
                "execution_confidence_band",
            ]
        )
    rows: list[dict[str, object]] = []
    for event_id, group in marks.groupby("event_id", dropna=False):
        status = group["quote_status"].astype(str)
        score = pd.to_numeric(group["quote_score"], errors="coerce").fillna(0.0)
        ages = pd.to_numeric(group.get("quote_age_seconds"), errors="coerce")
        spreads = pd.to_numeric(group.get("spread_over_mid"), errors="coerce")
        confidence = float(score.mean()) if len(score) else 0.0
        ticker = group["ticker"].dropna().iloc[0] if group["ticker"].notna().any() else None
        rows.append(
            {
                "event_id": event_id,
                "ticker": ticker,
                "announcement_date": group["announcement_date"].dropna().iloc[0]
                if "announcement_date" in group and group["announcement_date"].notna().any()
                else None,
                "announcement_timing": group["announcement_timing"].dropna().iloc[0]
                if "announcement_timing" in group and group["announcement_timing"].notna().any()
                else None,
                "quote_execution_route": quote_execution_route,
                "required_quote_marks": int(len(group)),
                "ok_quote_marks": int(status.eq(QUOTE_STATUS_OK).sum()),
                "missing_quote_marks": int(status.eq(QUOTE_STATUS_MISSING).sum()),
                "invalid_quote_marks": int(status.eq(QUOTE_STATUS_INVALID).sum()),
                "stale_quote_marks": int(status.eq(QUOTE_STATUS_STALE).sum()),
                "wide_spread_marks": int(status.eq(QUOTE_STATUS_WIDE).sum()),
                "max_quote_age_seconds": None if ages.dropna().empty else float(ages.max()),
                "median_spread_over_mid": None
                if spreads.dropna().empty
                else float(spreads.median()),
                "execution_confidence_score": confidence,
                "execution_confidence_band": quote_confidence_band(confidence),
                "paper_grade": False,
            }
        )
    return pd.DataFrame(rows)


def build_quote_execution_leg_panel(marks: pd.DataFrame) -> pd.DataFrame:
    """Build leg-level long-straddle execution fills from quote-window marks."""
    if marks.empty:
        return _empty_quote_execution_legs_frame()
    frame = marks.copy()
    frame["window_label"] = frame["window_label"].astype(str)
    frame["execution_side"] = np.where(
        frame["window_label"].str.startswith("entry"),
        "buy_to_open",
        np.where(frame["window_label"].str.startswith("exit"), "sell_to_close", "unknown"),
    )
    frame["contract_multiplier"] = CONTRACT_MULTIPLIER
    bid = pd.to_numeric(frame.get("bid"), errors="coerce")
    ask = pd.to_numeric(frame.get("ask"), errors="coerce")
    mid = pd.to_numeric(frame.get("mid"), errors="coerce")
    frame["long_fill_price"] = np.where(frame["execution_side"].eq("buy_to_open"), ask, bid)
    frame.loc[frame["execution_side"].eq("unknown"), "long_fill_price"] = np.nan
    frame["long_fill_value_usd"] = (
        pd.to_numeric(frame["long_fill_price"], errors="coerce") * CONTRACT_MULTIPLIER
    )
    frame["mid_fill_value_usd"] = mid * CONTRACT_MULTIPLIER
    columns = [column for column in _empty_quote_execution_legs_frame().columns if column in frame]
    return frame[columns].copy()


def _first_non_null(group: pd.DataFrame, columns: Sequence[str]) -> object | None:
    for column in columns:
        if column in group.columns:
            values = group[column].dropna()
            if not values.empty:
                return cast(object, values.iloc[0])
    return None


def _mark_by_right(group: pd.DataFrame, *, window_label: str, right: str) -> pd.Series | None:
    subset = group.loc[
        group["window_label"].astype(str).eq(window_label)
        & group["right"].astype(str).str.lower().eq(right)
    ]
    if subset.empty:
        return None
    return subset.iloc[0]


def _numeric_cell(row: pd.Series | None, column: str) -> float:
    if row is None or column not in row:
        return float("nan")
    value = pd.to_numeric(pd.Series([row[column]]), errors="coerce").iloc[0]
    return float(value) if np.isfinite(value) else float("nan")


def _status_cell(row: pd.Series | None) -> str:
    if row is None:
        return QUOTE_STATUS_MISSING
    return str(row.get("quote_status", QUOTE_STATUS_MISSING))


def _spot_for_group(group: pd.DataFrame) -> float:
    spot_raw = _first_non_null(group, ("spot", "s_before", "close_before"))
    if spot_raw is None:
        return float("nan")
    value = pd.to_numeric(pd.Series([spot_raw]), errors="coerce").iloc[0]
    return float(value) if np.isfinite(value) and float(value) > 0 else float("nan")


def _premium_var(premium_usd: float, spot: float) -> float:
    if not np.isfinite(premium_usd) or not np.isfinite(spot) or spot <= 0:
        return float("nan")
    premium_per_share = premium_usd / CONTRACT_MULTIPLIER
    return float((premium_per_share / spot) ** 2)


def _implied_vol_from_quote(
    *,
    spot: float,
    strike: float,
    time_to_expiry: float,
    option_price: float,
    right: str,
) -> tuple[float | None, str | None]:
    if not np.isfinite(spot) or not np.isfinite(strike) or spot <= 0 or strike <= 0:
        return None, "nonpositive_spot_or_strike"
    if not np.isfinite(time_to_expiry) or time_to_expiry <= 0:
        return None, "nonpositive_time_to_expiry"
    if not np.isfinite(option_price) or option_price <= 0:
        return None, "missing_or_nonpositive_option_price"
    option_right = OptionRight.CALL if str(right).lower() == "call" else OptionRight.PUT
    intrinsic = (
        max(spot - strike, 0.0) if option_right == OptionRight.CALL else max(strike - spot, 0.0)
    )
    if option_price < intrinsic:
        return None, "option_price_below_intrinsic"

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
            return None, "implied_vol_root_not_bracketed"
        return float(brentq(objective, 1e-4, 5.0, maxiter=80)), None
    except (ValueError, RuntimeError, OverflowError):
        return None, "implied_vol_solver_failed"


def _mean_if_all_finite(values: Sequence[float]) -> float:
    arr = np.array(values, dtype=float)
    return float(arr.mean()) if bool(np.isfinite(arr).all()) else float("nan")


def _date_or_none(value: object) -> date | None:
    if value is None or pd.isna(value):
        return None
    parsed = pd.Timestamp(value)
    return date(parsed.year, parsed.month, parsed.day)


def _event_exit_fallback(
    *,
    announcement_date: date | None,
    announcement_timing: object,
    exit_date: date | None,
) -> date | None:
    if exit_date is not None:
        return exit_date
    if announcement_date is None:
        return None
    timing = str(announcement_timing or "").upper()
    return announcement_date if timing == "BMO" else announcement_date + timedelta(days=1)


def build_quote_iv_surface_panel(marks: pd.DataFrame) -> pd.DataFrame:
    """Build bounded entry-window Black-Scholes IV diagnostics from quote marks."""

    if marks.empty:
        return _empty_quote_iv_surface_frame()
    required = {"event_id", "expiration", "strike", "right", "window_label", "spot", "mid"}
    missing = sorted(required - set(marks.columns))
    if missing:
        raise ValueError(f"quote marks missing required columns for IV surface: {missing}")
    frame = marks.loc[marks["window_label"].astype(str).eq("entry_preclose_15m")].copy()
    if frame.empty:
        return _empty_quote_iv_surface_frame()
    rows: list[dict[str, object]] = []
    for _, row in frame.iterrows():
        entry_date = _date_or_none(row.get("entry_date")) or _date_or_none(
            row.get("announcement_date")
        )
        expiration = _date_or_none(row.get("expiration"))
        dte = None if entry_date is None or expiration is None else (expiration - entry_date).days
        time_to_expiry = year_fraction(dte) if dte is not None and dte > 0 else float("nan")
        spot = _numeric_cell(row, "spot")
        if not np.isfinite(spot):
            spot = _numeric_cell(row, "s_before")
        if not np.isfinite(spot):
            spot = _numeric_cell(row, "close_before")
        strike = _numeric_cell(row, "strike")
        right = str(row.get("right", "")).lower()
        quote_status = str(row.get("quote_status", QUOTE_STATUS_MISSING))
        bid_iv, bid_reason = _implied_vol_from_quote(
            spot=spot,
            strike=strike,
            time_to_expiry=time_to_expiry,
            option_price=_numeric_cell(row, "bid"),
            right=right,
        )
        mid_iv, mid_reason = _implied_vol_from_quote(
            spot=spot,
            strike=strike,
            time_to_expiry=time_to_expiry,
            option_price=_numeric_cell(row, "mid"),
            right=right,
        )
        ask_iv, ask_reason = _implied_vol_from_quote(
            spot=spot,
            strike=strike,
            time_to_expiry=time_to_expiry,
            option_price=_numeric_cell(row, "ask"),
            right=right,
        )
        if quote_status == QUOTE_STATUS_MISSING:
            bid_reason = bid_reason or QUOTE_STATUS_MISSING
            mid_reason = mid_reason or QUOTE_STATUS_MISSING
            ask_reason = ask_reason or QUOTE_STATUS_MISSING
        elif quote_status == QUOTE_STATUS_INVALID:
            bid_reason = bid_reason or QUOTE_STATUS_INVALID
            mid_reason = mid_reason or QUOTE_STATUS_INVALID
            ask_reason = ask_reason or QUOTE_STATUS_INVALID
        elif quote_status == QUOTE_STATUS_STALE:
            bid_reason = bid_reason or QUOTE_STATUS_STALE
            mid_reason = mid_reason or QUOTE_STATUS_STALE
            ask_reason = ask_reason or QUOTE_STATUS_STALE
        rows.append(
            {
                "event_id": row.get("event_id"),
                "ticker": row.get("ticker"),
                "announcement_date": row.get("announcement_date"),
                "announcement_timing": row.get("announcement_timing"),
                "entry_date": entry_date,
                "expiration": expiration,
                "dte": dte,
                "time_to_expiry": time_to_expiry,
                "strike": strike,
                "right": right,
                "options_ticker": row.get("options_ticker"),
                "spot": spot,
                "quote_status": quote_status,
                "quote_timestamp_et": row.get("quote_timestamp_et"),
                "quote_age_seconds": row.get("quote_age_seconds"),
                "bid": row.get("bid"),
                "ask": row.get("ask"),
                "mid": row.get("mid"),
                "spread_over_mid": row.get("spread_over_mid"),
                "quote_bid_iv": bid_iv,
                "quote_mid_iv": mid_iv,
                "quote_ask_iv": ask_iv,
                "quote_bid_iv_failure_reason": bid_reason,
                "quote_mid_iv_failure_reason": mid_reason,
                "quote_ask_iv_failure_reason": ask_reason,
                "paper_grade_quote_iv_mid": quote_status == QUOTE_STATUS_OK and mid_reason is None,
                "paper_grade_quote_iv_bidask": quote_status == QUOTE_STATUS_OK
                and bid_reason is None
                and ask_reason is None,
                "quote_iv_surface_method": QUOTE_IV_SURFACE_METHOD,
                "quote_iv_surface_claim_scope": QUOTE_IV_SURFACE_CLAIM_SCOPE,
            }
        )
    return pd.DataFrame(rows, columns=_empty_quote_iv_surface_frame().columns)


def build_quote_iv_surface_summary_panel(surface: pd.DataFrame) -> pd.DataFrame:
    """Aggregate entry leg IVs to event/expiration/strike call-put surface pairs."""

    if surface.empty:
        return _empty_quote_iv_surface_summary_frame()
    required = {"event_id", "expiration", "strike", "right", "quote_mid_iv"}
    missing = sorted(required - set(surface.columns))
    if missing:
        raise ValueError(f"quote IV surface frame missing required columns: {missing}")
    rows: list[dict[str, object]] = []
    for (event_id, expiration, strike), group in surface.groupby(
        ["event_id", "expiration", "strike"], dropna=False
    ):
        call_subset = group.loc[group["right"].astype(str).str.lower().eq("call")]
        put_subset = group.loc[group["right"].astype(str).str.lower().eq("put")]
        call_row = call_subset.iloc[0] if not call_subset.empty else None
        put_row = put_subset.iloc[0] if not put_subset.empty else None
        call_mid = _numeric_cell(call_row, "quote_mid_iv")
        put_mid = _numeric_cell(put_row, "quote_mid_iv")
        call_bid = _numeric_cell(call_row, "quote_bid_iv")
        put_bid = _numeric_cell(put_row, "quote_bid_iv")
        call_ask = _numeric_cell(call_row, "quote_ask_iv")
        put_ask = _numeric_cell(put_row, "quote_ask_iv")
        mean_mid = _mean_if_all_finite([call_mid, put_mid])
        mean_bid = _mean_if_all_finite([call_bid, put_bid])
        mean_ask = _mean_if_all_finite([call_ask, put_ask])
        reference_row = call_row if call_row is not None else put_row
        time_to_expiry = _numeric_cell(reference_row, "time_to_expiry")
        complete_mid = np.isfinite(mean_mid) and np.isfinite(time_to_expiry)
        complete_bidask = (
            np.isfinite(mean_bid) and np.isfinite(mean_ask) and np.isfinite(time_to_expiry)
        )
        call_status = _status_cell(call_row)
        put_status = _status_cell(put_row)
        rows.append(
            {
                "event_id": event_id,
                "ticker": _first_non_null(group, ("ticker",)),
                "announcement_date": _first_non_null(group, ("announcement_date",)),
                "announcement_timing": _first_non_null(group, ("announcement_timing",)),
                "entry_date": _first_non_null(group, ("entry_date",)),
                "expiration": expiration,
                "dte": _first_non_null(group, ("dte",)),
                "time_to_expiry": time_to_expiry,
                "strike": strike,
                "spot": _first_non_null(group, ("spot",)),
                "call_status": call_status,
                "put_status": put_status,
                "call_mid_iv": call_mid,
                "put_mid_iv": put_mid,
                "call_bid_iv": call_bid,
                "put_bid_iv": put_bid,
                "call_ask_iv": call_ask,
                "put_ask_iv": put_ask,
                "mean_mid_iv": mean_mid,
                "mean_bid_iv": mean_bid,
                "mean_ask_iv": mean_ask,
                "skew_put_minus_call_mid_iv": put_mid - call_mid
                if np.isfinite(put_mid) and np.isfinite(call_mid)
                else np.nan,
                "quote_mid_total_variance": mean_mid**2 * time_to_expiry
                if complete_mid
                else np.nan,
                "quote_bid_total_variance": mean_bid**2 * time_to_expiry
                if np.isfinite(mean_bid) and np.isfinite(time_to_expiry)
                else np.nan,
                "quote_ask_total_variance": mean_ask**2 * time_to_expiry
                if np.isfinite(mean_ask) and np.isfinite(time_to_expiry)
                else np.nan,
                "complete_mid_surface_pair": bool(complete_mid),
                "complete_bidask_surface_pair": bool(complete_bidask),
                "paper_grade_quote_iv_surface_pair": bool(
                    call_status == QUOTE_STATUS_OK
                    and put_status == QUOTE_STATUS_OK
                    and complete_bidask
                ),
                "quote_iv_surface_method": QUOTE_IV_SURFACE_METHOD,
                "quote_iv_surface_claim_scope": QUOTE_IV_SURFACE_CLAIM_SCOPE,
            }
        )
    return pd.DataFrame(rows, columns=_empty_quote_iv_surface_summary_frame().columns)


def build_quote_surface_ivar_event_panel(surface_summary: pd.DataFrame) -> pd.DataFrame:
    """Extract event IVAR from bounded quote-IV surface total variance pairs."""

    if surface_summary.empty:
        return _empty_quote_surface_ivar_event_frame()
    required = {
        "event_id",
        "expiration",
        "strike",
        "quote_mid_total_variance",
        "quote_bid_total_variance",
        "quote_ask_total_variance",
        "complete_mid_surface_pair",
        "complete_bidask_surface_pair",
    }
    missing = sorted(required - set(surface_summary.columns))
    if missing:
        raise ValueError(f"quote IV surface summary missing required columns: {missing}")
    frame = surface_summary.copy()
    frame["expiration"] = pd.to_datetime(frame["expiration"], errors="coerce").dt.date
    if "entry_date" in frame:
        frame["entry_date"] = pd.to_datetime(frame["entry_date"], errors="coerce").dt.date
    if "announcement_date" in frame:
        frame["announcement_date"] = pd.to_datetime(
            frame["announcement_date"], errors="coerce"
        ).dt.date
    rows: list[dict[str, object]] = []
    for event_id, group in frame.groupby("event_id", dropna=False):
        group = group.copy()
        group["_spot_numeric"] = pd.to_numeric(group.get("spot"), errors="coerce")
        group["_strike_numeric"] = pd.to_numeric(group["strike"], errors="coerce")
        group["_moneyness_abs"] = (group["_strike_numeric"] / group["_spot_numeric"] - 1.0).abs()
        selected = (
            group.sort_values(
                ["complete_mid_surface_pair", "complete_bidask_surface_pair", "_moneyness_abs"],
                ascending=[False, False, True],
            )
            .groupby("expiration", dropna=True)
            .head(1)
            .sort_values("expiration")
            .reset_index(drop=True)
        )
        entry_date = _date_or_none(_first_non_null(group, ("entry_date",)))
        first = selected.iloc[0] if len(selected) else None
        second = selected.iloc[1] if len(selected) > 1 else None
        if len(selected) < 2 or entry_date is None:
            failure = IVARFailureReason.NO_TWO_EVENT_COVERING_EXPIRIES.value
            rows.append(
                {
                    "event_id": event_id,
                    "ticker": _first_non_null(group, ("ticker",)),
                    "announcement_date": _first_non_null(group, ("announcement_date",)),
                    "announcement_timing": _first_non_null(group, ("announcement_timing",)),
                    "entry_date": entry_date,
                    "quote_surface_ivar_method": QUOTE_SURFACE_IVAR_METHOD,
                    "quote_surface_ivar_claim_scope": QUOTE_SURFACE_IVAR_CLAIM_SCOPE,
                    "surface_pair_count": int(len(selected)),
                    "quote_surface_mid_ivar_event": None,
                    "quote_surface_bid_ivar_event": None,
                    "quote_surface_ask_ivar_event": None,
                    "quote_surface_mid_ivar_failure_reason": failure,
                    "quote_surface_bid_ivar_failure_reason": failure,
                    "quote_surface_ask_ivar_failure_reason": failure,
                    "expiration_1": None if first is None else first.get("expiration"),
                    "expiration_2": None if second is None else second.get("expiration"),
                    "strike_1": None if first is None else first.get("strike"),
                    "strike_2": None if second is None else second.get("strike"),
                    "spot_1": None if first is None else first.get("spot"),
                    "spot_2": None if second is None else second.get("spot"),
                    "paper_grade_quote_surface_ivar_mid": False,
                    "paper_grade_quote_surface_ivar_bid": False,
                    "paper_grade_quote_surface_ivar_ask": False,
                }
            )
            continue
        first_row = cast(pd.Series, first)
        second_row = cast(pd.Series, second)
        exp1 = _date_or_none(first_row["expiration"])
        exp2 = _date_or_none(second_row["expiration"])
        dte1 = None if exp1 is None else (exp1 - entry_date).days
        dte2 = None if exp2 is None else (exp2 - entry_date).days
        mid_ivar: float | None
        bid_ivar: float | None
        ask_ivar: float | None
        mid_reason: str | None
        bid_reason: str | None
        ask_reason: str | None
        if dte1 is None or dte2 is None or dte1 <= 0 or dte2 <= 0:
            t1 = None
            t2 = None
            mid_ivar, mid_reason = None, IVARFailureReason.NONPOSITIVE_TIME_GAP.value
            bid_ivar, bid_reason = None, IVARFailureReason.NONPOSITIVE_TIME_GAP.value
            ask_ivar, ask_reason = None, IVARFailureReason.NONPOSITIVE_TIME_GAP.value
        else:
            t1 = year_fraction(dte1)
            t2 = year_fraction(dte2)
            mid_ivar, mid_reason = _total_variance_ivar(
                first_row,
                second_row,
                total_variance_col="quote_mid_total_variance",
                complete_col="complete_mid_surface_pair",
                t1=t1,
                t2=t2,
            )
            bid_ivar, bid_reason = _total_variance_ivar(
                first_row,
                second_row,
                total_variance_col="quote_bid_total_variance",
                complete_col="complete_bidask_surface_pair",
                t1=t1,
                t2=t2,
            )
            ask_ivar, ask_reason = _total_variance_ivar(
                first_row,
                second_row,
                total_variance_col="quote_ask_total_variance",
                complete_col="complete_bidask_surface_pair",
                t1=t1,
                t2=t2,
            )
        rows.append(
            {
                "event_id": event_id,
                "ticker": _first_non_null(group, ("ticker",)),
                "announcement_date": _first_non_null(group, ("announcement_date",)),
                "announcement_timing": _first_non_null(group, ("announcement_timing",)),
                "entry_date": entry_date,
                "quote_surface_ivar_method": QUOTE_SURFACE_IVAR_METHOD,
                "quote_surface_ivar_claim_scope": QUOTE_SURFACE_IVAR_CLAIM_SCOPE,
                "surface_pair_count": int(len(selected)),
                "quote_surface_mid_ivar_event": mid_ivar,
                "quote_surface_bid_ivar_event": bid_ivar,
                "quote_surface_ask_ivar_event": ask_ivar,
                "quote_surface_mid_ivar_failure_reason": mid_reason,
                "quote_surface_bid_ivar_failure_reason": bid_reason,
                "quote_surface_ask_ivar_failure_reason": ask_reason,
                "t1": t1,
                "t2": t2,
                "dte_1": dte1,
                "dte_2": dte2,
                "expiration_1": exp1,
                "expiration_2": exp2,
                "expiry_gap_days": None if exp1 is None or exp2 is None else (exp2 - exp1).days,
                "mid_total_variance_1": first_row.get("quote_mid_total_variance"),
                "mid_total_variance_2": second_row.get("quote_mid_total_variance"),
                "bid_total_variance_1": first_row.get("quote_bid_total_variance"),
                "bid_total_variance_2": second_row.get("quote_bid_total_variance"),
                "ask_total_variance_1": first_row.get("quote_ask_total_variance"),
                "ask_total_variance_2": second_row.get("quote_ask_total_variance"),
                "strike_1": first_row.get("strike"),
                "strike_2": second_row.get("strike"),
                "spot_1": first_row.get("spot"),
                "spot_2": second_row.get("spot"),
                "mid_complete_pair_1": first_row.get("complete_mid_surface_pair"),
                "mid_complete_pair_2": second_row.get("complete_mid_surface_pair"),
                "bidask_complete_pair_1": first_row.get("complete_bidask_surface_pair"),
                "bidask_complete_pair_2": second_row.get("complete_bidask_surface_pair"),
                "paper_grade_quote_surface_ivar_mid": mid_reason is None,
                "paper_grade_quote_surface_ivar_bid": bid_reason is None,
                "paper_grade_quote_surface_ivar_ask": ask_reason is None,
            }
        )
    return pd.DataFrame(rows, columns=_empty_quote_surface_ivar_event_frame().columns)


def _total_variance_ivar(
    first: pd.Series,
    second: pd.Series,
    *,
    total_variance_col: str,
    complete_col: str,
    t1: float,
    t2: float,
) -> tuple[float | None, str | None]:
    if not bool(first.get(complete_col, False)) or not bool(second.get(complete_col, False)):
        return None, IVARFailureReason.STALE_OR_MISSING_IV.value
    w1 = _numeric_cell(first, total_variance_col)
    w2 = _numeric_cell(second, total_variance_col)
    if not np.isfinite(w1) or not np.isfinite(w2):
        return None, IVARFailureReason.STALE_OR_MISSING_IV.value
    if t2 <= t1 or t1 <= 0:
        return None, IVARFailureReason.NONPOSITIVE_TIME_GAP.value
    if w1 <= 0 or w2 <= 0:
        return None, IVARFailureReason.NONPOSITIVE_TOTAL_VARIANCE.value
    if w2 < w1:
        return None, IVARFailureReason.NONMONOTONE_TOTAL_VARIANCE.value
    ivar = float((t2 * w1 - t1 * w2) / (t2 - t1))
    if ivar < 0:
        return None, IVARFailureReason.NEGATIVE_EXTRACTED_IVAR.value
    return ivar, None


def build_quote_straddle_execution_panel(marks: pd.DataFrame) -> pd.DataFrame:
    """Aggregate call/put quote marks to event-strike long-straddle execution rows."""
    if marks.empty:
        return _empty_quote_straddle_execution_frame()
    required = {"event_id", "expiration", "strike", "right", "window_label"}
    missing = sorted(required - set(marks.columns))
    if missing:
        raise ValueError(f"quote marks missing required columns: {missing}")
    rows: list[dict[str, object]] = []
    group_cols = ["event_id", "expiration", "strike"]
    for (event_id, expiration, strike), group in marks.groupby(group_cols, dropna=False):
        entry_call = _mark_by_right(group, window_label="entry_preclose_15m", right="call")
        entry_put = _mark_by_right(group, window_label="entry_preclose_15m", right="put")
        exit_call = _mark_by_right(group, window_label="exit_preclose_15m", right="call")
        exit_put = _mark_by_right(group, window_label="exit_preclose_15m", right="put")
        entry_ask = (
            _numeric_cell(entry_call, "ask") + _numeric_cell(entry_put, "ask")
        ) * CONTRACT_MULTIPLIER
        entry_mid = (
            _numeric_cell(entry_call, "mid") + _numeric_cell(entry_put, "mid")
        ) * CONTRACT_MULTIPLIER
        entry_bid = (
            _numeric_cell(entry_call, "bid") + _numeric_cell(entry_put, "bid")
        ) * CONTRACT_MULTIPLIER
        exit_bid = (
            _numeric_cell(exit_call, "bid") + _numeric_cell(exit_put, "bid")
        ) * CONTRACT_MULTIPLIER
        exit_mid = (
            _numeric_cell(exit_call, "mid") + _numeric_cell(exit_put, "mid")
        ) * CONTRACT_MULTIPLIER
        exit_ask = (
            _numeric_cell(exit_call, "ask") + _numeric_cell(exit_put, "ask")
        ) * CONTRACT_MULTIPLIER
        spot = _spot_for_group(group)
        statuses = {
            "entry_call_status": _status_cell(entry_call),
            "entry_put_status": _status_cell(entry_put),
            "exit_call_status": _status_cell(exit_call),
            "exit_put_status": _status_cell(exit_put),
        }
        complete_bidask_pair = all(status == QUOTE_STATUS_OK for status in statuses.values())
        complete_mid_pair = all(np.isfinite(value) for value in (entry_mid, exit_mid)) and all(
            status != QUOTE_STATUS_MISSING for status in statuses.values()
        )
        rows.append(
            {
                "event_id": event_id,
                "ticker": _first_non_null(group, ("ticker",)),
                "announcement_date": _first_non_null(group, ("announcement_date",)),
                "announcement_timing": _first_non_null(group, ("announcement_timing",)),
                "entry_date": _first_non_null(group, ("entry_date",)),
                "exit_date": _first_non_null(group, ("exit_date",)),
                "expiration": expiration,
                "strike": strike,
                "spot": spot,
                **statuses,
                "entry_ask_cost_usd": entry_ask,
                "entry_mid_cost_usd": entry_mid,
                "entry_bid_cost_usd": entry_bid,
                "exit_bid_value_usd": exit_bid,
                "exit_mid_value_usd": exit_mid,
                "exit_ask_value_usd": exit_ask,
                "quote_bidask_pnl_usd": exit_bid - entry_ask,
                "quote_mid_pnl_usd": exit_mid - entry_mid,
                "quote_entry_mid_premium_pct_spot": np.sqrt(_premium_var(entry_mid, spot))
                if np.isfinite(_premium_var(entry_mid, spot))
                else np.nan,
                "quote_entry_ask_premium_pct_spot": np.sqrt(_premium_var(entry_ask, spot))
                if np.isfinite(_premium_var(entry_ask, spot))
                else np.nan,
                "quote_premium_var_mid": _premium_var(entry_mid, spot),
                "quote_premium_var_ask": _premium_var(entry_ask, spot),
                "complete_bidask_pair": complete_bidask_pair,
                "complete_mid_pair": complete_mid_pair,
                "paper_grade_execution": bool(complete_bidask_pair and np.isfinite(spot)),
            }
        )
    return pd.DataFrame(rows)


def build_quote_ivar_event_panel(straddles: pd.DataFrame) -> pd.DataFrame:
    """Build event-level quote-premium IVAR diagnostics from straddle quote marks.

    The quote entry premium is treated as a total-variance proxy. This is a
    diagnostic artifact for quote-aware robustness, not a replacement for a full
    implied-volatility surface or NBBO execution study.
    """
    if straddles.empty:
        return _empty_quote_ivar_event_frame()
    required = {
        "event_id",
        "expiration",
        "strike",
        "spot",
        "quote_premium_var_mid",
        "quote_premium_var_ask",
        "complete_mid_pair",
        "complete_bidask_pair",
    }
    missing = sorted(required - set(straddles.columns))
    if missing:
        raise ValueError(f"quote straddle frame missing required columns: {missing}")
    frame = straddles.copy()
    frame["expiration"] = pd.to_datetime(frame["expiration"], errors="coerce").dt.date
    if "entry_date" in frame:
        frame["entry_date"] = pd.to_datetime(frame["entry_date"], errors="coerce").dt.date
    if "exit_date" in frame:
        frame["exit_date"] = pd.to_datetime(frame["exit_date"], errors="coerce").dt.date
    if "announcement_date" in frame:
        frame["announcement_date"] = pd.to_datetime(
            frame["announcement_date"], errors="coerce"
        ).dt.date

    rows: list[dict[str, object]] = []
    for event_id, group in frame.groupby("event_id", dropna=False):
        group = group.copy()
        group["_spot_numeric"] = pd.to_numeric(group["spot"], errors="coerce")
        group["_strike_numeric"] = pd.to_numeric(group["strike"], errors="coerce")
        group["_moneyness_abs"] = (group["_strike_numeric"] / group["_spot_numeric"] - 1.0).abs()
        selected_by_expiry = (
            group.sort_values(
                ["complete_mid_pair", "complete_bidask_pair", "_moneyness_abs"],
                ascending=[False, False, True],
            )
            .groupby("expiration", dropna=True)
            .head(1)
            .sort_values("expiration")
            .reset_index(drop=True)
        )
        announcement_date = _date_or_none(_first_non_null(group, ("announcement_date",)))
        announcement_timing = _first_non_null(group, ("announcement_timing",))
        entry_date = _date_or_none(_first_non_null(group, ("entry_date",)))
        exit_date = _event_exit_fallback(
            announcement_date=announcement_date,
            announcement_timing=announcement_timing,
            exit_date=_date_or_none(_first_non_null(group, ("exit_date",))),
        )
        coverage_date = exit_date or (
            announcement_date + timedelta(days=1) if announcement_date is not None else None
        )
        selected = selected_by_expiry
        if coverage_date is not None:
            selected = selected.loc[
                pd.to_datetime(selected["expiration"], errors="coerce").dt.date.ge(coverage_date)
            ].copy()
        selected = selected.sort_values("expiration").reset_index(drop=True)
        first = selected.iloc[0] if len(selected) else None
        second = selected.iloc[1] if len(selected) > 1 else None
        failure = IVARFailureReason.NO_TWO_EVENT_COVERING_EXPIRIES.value
        if len(selected) < 2 or entry_date is None:
            rows.append(
                {
                    "event_id": event_id,
                    "ticker": _first_non_null(group, ("ticker",)),
                    "announcement_date": announcement_date,
                    "announcement_timing": announcement_timing,
                    "entry_date": entry_date,
                    "exit_date": exit_date,
                    "quote_ivar_method": QUOTE_IVAR_METHOD,
                    "quote_ivar_claim_scope": QUOTE_IVAR_CLAIM_SCOPE,
                    "expiry_candidate_count": int(len(selected)),
                    "quote_mid_ivar_event": None,
                    "quote_ask_ivar_event": None,
                    "quote_mid_ivar_failure_reason": failure,
                    "quote_ask_ivar_failure_reason": failure,
                    "t1": None,
                    "t2": None,
                    "dte_1": None,
                    "dte_2": None,
                    "expiration_1": None if first is None else first.get("expiration"),
                    "expiration_2": None if second is None else second.get("expiration"),
                    "expiry_gap_days": None,
                    "mid_total_variance_1": None
                    if first is None
                    else first.get("quote_premium_var_mid"),
                    "mid_total_variance_2": None
                    if second is None
                    else second.get("quote_premium_var_mid"),
                    "ask_total_variance_1": None
                    if first is None
                    else first.get("quote_premium_var_ask"),
                    "ask_total_variance_2": None
                    if second is None
                    else second.get("quote_premium_var_ask"),
                    "strike_1": None if first is None else first.get("strike"),
                    "strike_2": None if second is None else second.get("strike"),
                    "spot_1": None if first is None else first.get("spot"),
                    "spot_2": None if second is None else second.get("spot"),
                    "mid_complete_pair_1": None
                    if first is None
                    else first.get("complete_mid_pair"),
                    "mid_complete_pair_2": None
                    if second is None
                    else second.get("complete_mid_pair"),
                    "bidask_complete_pair_1": None
                    if first is None
                    else first.get("complete_bidask_pair"),
                    "bidask_complete_pair_2": None
                    if second is None
                    else second.get("complete_bidask_pair"),
                    "paper_grade_quote_ivar_mid": False,
                    "paper_grade_quote_ivar_ask": False,
                }
            )
            continue

        first_row = cast(pd.Series, first)
        second_row = cast(pd.Series, second)
        exp1 = _date_or_none(first_row["expiration"])
        exp2 = _date_or_none(second_row["expiration"])
        dte1 = None if exp1 is None else (exp1 - entry_date).days
        dte2 = None if exp2 is None else (exp2 - entry_date).days
        t1: float | None
        t2: float | None
        mid_ivar: float | None
        ask_ivar: float | None
        mid_reason: str | None
        ask_reason: str | None
        if dte1 is None or dte2 is None or dte1 <= 0 or dte2 <= 0:
            t1 = None
            t2 = None
            mid_ivar, mid_reason = None, IVARFailureReason.NONPOSITIVE_TIME_GAP.value
            ask_ivar, ask_reason = None, IVARFailureReason.NONPOSITIVE_TIME_GAP.value
        else:
            t1 = year_fraction(dte1)
            t2 = year_fraction(dte2)
            mid_ivar, mid_reason = _total_variance_ivar(
                first_row,
                second_row,
                total_variance_col="quote_premium_var_mid",
                complete_col="complete_mid_pair",
                t1=t1,
                t2=t2,
            )
            ask_ivar, ask_reason = _total_variance_ivar(
                first_row,
                second_row,
                total_variance_col="quote_premium_var_ask",
                complete_col="complete_bidask_pair",
                t1=t1,
                t2=t2,
            )
        rows.append(
            {
                "event_id": event_id,
                "ticker": _first_non_null(group, ("ticker",)),
                "announcement_date": announcement_date,
                "announcement_timing": announcement_timing,
                "entry_date": entry_date,
                "exit_date": exit_date,
                "quote_ivar_method": QUOTE_IVAR_METHOD,
                "quote_ivar_claim_scope": QUOTE_IVAR_CLAIM_SCOPE,
                "expiry_candidate_count": int(len(selected)),
                "quote_mid_ivar_event": mid_ivar,
                "quote_ask_ivar_event": ask_ivar,
                "quote_mid_ivar_failure_reason": mid_reason,
                "quote_ask_ivar_failure_reason": ask_reason,
                "t1": t1,
                "t2": t2,
                "dte_1": dte1,
                "dte_2": dte2,
                "expiration_1": exp1,
                "expiration_2": exp2,
                "expiry_gap_days": None if exp1 is None or exp2 is None else (exp2 - exp1).days,
                "mid_total_variance_1": first_row.get("quote_premium_var_mid"),
                "mid_total_variance_2": second_row.get("quote_premium_var_mid"),
                "ask_total_variance_1": first_row.get("quote_premium_var_ask"),
                "ask_total_variance_2": second_row.get("quote_premium_var_ask"),
                "strike_1": first_row.get("strike"),
                "strike_2": second_row.get("strike"),
                "spot_1": first_row.get("spot"),
                "spot_2": second_row.get("spot"),
                "mid_complete_pair_1": first_row.get("complete_mid_pair"),
                "mid_complete_pair_2": second_row.get("complete_mid_pair"),
                "bidask_complete_pair_1": first_row.get("complete_bidask_pair"),
                "bidask_complete_pair_2": second_row.get("complete_bidask_pair"),
                "paper_grade_quote_ivar_mid": mid_reason is None,
                "paper_grade_quote_ivar_ask": ask_reason is None,
            }
        )
    return pd.DataFrame(rows, columns=_empty_quote_ivar_event_frame().columns)


def _read_quote_csv_chunks(paths: Sequence[Path], *, chunksize: int) -> Iterator[pd.DataFrame]:
    for path in paths:
        yield from pd.read_csv(path, chunksize=chunksize)


def _stream_massive_quote_chunks(
    config: ProjectConfig,
    *,
    date_value: date,
    chunksize: int,
    aws_executable: str = "aws",
) -> Iterator[pd.DataFrame]:  # pragma: no cover - exercised through integration runs
    key = option_quotes_flat_file_key(
        config, year=date_value.year, month=date_value.month, date=date_value.isoformat()
    )
    command = [
        aws_executable,
        "s3",
        "cp",
        f"s3://{config.massive_flat_file_bucket}/{key}",
        "-",
        "--endpoint-url",
        config.massive_flat_file_endpoint_url,
    ]
    process = subprocess.Popen(
        command,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        env=massive_flat_file_aws_env(config),
    )
    if process.stdout is None:
        raise RuntimeError("failed to open Massive quote stream")
    try:
        with gzip.GzipFile(fileobj=process.stdout) as gz_file:
            text_stream = io.TextIOWrapper(gz_file, encoding="utf-8")
            yield from pd.read_csv(text_stream, chunksize=chunksize)
    finally:
        stderr = b"" if process.stderr is None else process.stderr.read()
        return_code = process.wait()
        if return_code != 0:
            raise RuntimeError(
                f"Massive quote stream failed for {key}: {stderr.decode('utf-8', errors='replace')}"
            )


def _filter_normalized_quotes(normalized: pd.DataFrame, requests: pd.DataFrame) -> pd.DataFrame:
    if normalized.empty or requests.empty:
        return normalized.iloc[0:0].copy()
    tickers = set(requests["options_ticker"].astype(str))
    dates = set(pd.to_datetime(requests["quote_date"], errors="coerce").dt.date)
    filtered = normalized.loc[
        normalized["options_ticker"].astype(str).isin(tickers)
        & pd.to_datetime(normalized["quote_date"], errors="coerce").dt.date.isin(dates)
    ].copy()
    if filtered.empty:
        return filtered
    min_start = min(_to_datetime_et(value) for value in requests["window_start"])
    max_end = max(_to_datetime_et(value) for value in requests["window_end"])
    ts = filtered["quote_timestamp_et"].map(lambda value: _to_datetime_et(value))
    return filtered.loc[ts.ge(min_start) & ts.le(max_end)].copy()


def _load_rest_quote_window_for_worker(
    config: ProjectConfig,
    *,
    request: pd.Series,
    cache_dir: Path,
    limit: int,
) -> pd.DataFrame:
    with httpx.Client(timeout=config.massive_request_timeout_seconds) as client:
        return _load_or_fetch_rest_quote_window(
            client,
            config,
            request=request,
            cache_dir=cache_dir,
            limit=limit,
        )


def extract_quote_execution_panel(
    *,
    config: ProjectConfig,
    contracts: pd.DataFrame,
    windows: pd.DataFrame,
    out_dir: Path,
    quote_csv_paths: Sequence[Path] = (),
    dates: Sequence[date] = (),
    metadata_only: bool = False,
    chunksize: int = 250_000,
    entry_lookback_seconds: int = 900,
    exit_lookback_seconds: int = 900,
    stale_seconds: int = 60,
    wide_spread_threshold: float = 0.25,
    max_events: int | None = None,
    aws_executable: str = "aws",
    quote_source: str = QUOTE_SOURCE_FLAT_FILE,
    quote_cache_dir: Path | None = None,
    rest_limit: int = 50_000,
    quote_workers: int = 1,
    event_offset: int = 0,
    batch_label: str | None = None,
) -> QuoteExtractionReport:
    out_dir.mkdir(parents=True, exist_ok=True)
    if quote_source not in {QUOTE_SOURCE_REST, QUOTE_SOURCE_FLAT_FILE}:
        raise ValueError(f"unsupported quote_source: {quote_source}")
    if quote_workers <= 0:
        raise ValueError("quote_workers must be positive")
    if event_offset < 0:
        raise ValueError("event_offset must be non-negative")
    requests = build_quote_window_requests(
        contracts,
        windows,
        entry_lookback_seconds=entry_lookback_seconds,
        exit_lookback_seconds=exit_lookback_seconds,
        max_events=max_events,
        event_offset=event_offset,
    )
    if dates:
        date_set = set(dates)
        quote_dates = pd.to_datetime(requests["quote_date"], errors="coerce").dt.date
        touched_events = set(requests.loc[quote_dates.isin(date_set), "event_id"].astype(str))
        requests = requests.loc[requests["event_id"].astype(str).isin(touched_events)].copy()
    requests_path = out_dir / "quote_window_requests.csv"
    quotes_path = out_dir / "quote_window_quotes.csv"
    marks_path = out_dir / "quote_window_marks.csv"
    legs_path = out_dir / "quote_execution_legs.csv"
    straddle_path = out_dir / "quote_straddle_execution.csv"
    quote_ivar_path = out_dir / "quote_ivar_event.csv"
    quote_iv_surface_path = out_dir / "quote_iv_surface.csv"
    quote_iv_surface_summary_path = out_dir / "quote_iv_surface_summary.csv"
    quote_surface_ivar_path = out_dir / "quote_surface_ivar_event.csv"
    confidence_path = out_dir / "quote_execution_confidence.csv"
    report_path = out_dir / "quote_execution_report.json"
    requests.to_csv(requests_path, index=False)
    route = (
        QUOTE_ROUTE_CSV_FIXTURE
        if quote_csv_paths
        else QUOTE_ROUTE_REST_TARGETED
        if quote_source == QUOTE_SOURCE_REST
        else QUOTE_ROUTE_FLAT_FILE_FILTERED
    )
    if metadata_only:
        _empty_normalized_quotes_frame().to_csv(quotes_path, index=False)
        _empty_quote_window_marks_frame().to_csv(marks_path, index=False)
        _empty_quote_execution_legs_frame().to_csv(legs_path, index=False)
        _empty_quote_straddle_execution_frame().to_csv(straddle_path, index=False)
        _empty_quote_ivar_event_frame().to_csv(quote_ivar_path, index=False)
        _empty_quote_iv_surface_frame().to_csv(quote_iv_surface_path, index=False)
        _empty_quote_iv_surface_summary_frame().to_csv(quote_iv_surface_summary_path, index=False)
        _empty_quote_surface_ivar_event_frame().to_csv(quote_surface_ivar_path, index=False)
        build_execution_confidence_panel(pd.DataFrame(), quote_execution_route=route).to_csv(
            confidence_path, index=False
        )
        report = QuoteExtractionReport(
            ok=True,
            route=route,
            metadata_only=True,
            raw_full_day_files_written=False,
            request_rows=int(len(requests)),
            event_count=int(requests["event_id"].nunique()) if "event_id" in requests else 0,
            quote_rows_scanned=0,
            quote_rows_matched=0,
            dates=tuple(sorted(str(value) for value in requests["quote_date"].dropna().unique())),
            output_paths={
                "quote_window_requests": str(requests_path),
                "quote_window_quotes": str(quotes_path),
                "quote_window_marks": str(marks_path),
                "quote_execution_legs": str(legs_path),
                "quote_straddle_execution": str(straddle_path),
                "quote_ivar_event": str(quote_ivar_path),
                "quote_iv_surface": str(quote_iv_surface_path),
                "quote_iv_surface_summary": str(quote_iv_surface_summary_path),
                "quote_surface_ivar_event": str(quote_surface_ivar_path),
                "quote_execution_confidence": str(confidence_path),
                "quote_execution_report": str(report_path),
            },
            quote_workers=int(quote_workers),
            event_offset=int(event_offset),
            batch_label=batch_label,
        )
        report_path.write_text(
            json.dumps(report.as_dict(), indent=2, sort_keys=True), encoding="utf-8"
        )
        return report

    scanned = 0
    matched = 0
    matched_chunks: list[pd.DataFrame] = []
    if quote_csv_paths:
        chunk_iterable: Iterable[pd.DataFrame] = _read_quote_csv_chunks(
            quote_csv_paths, chunksize=chunksize
        )
        for chunk in chunk_iterable:
            scanned += int(len(chunk))
            normalized = normalize_option_quote_rows(chunk)
            filtered = _filter_normalized_quotes(normalized, requests)
            matched += int(len(filtered))
            if not filtered.empty:
                matched_chunks.append(filtered)
    elif quote_source == QUOTE_SOURCE_REST:
        cache_dir = quote_cache_dir or (out_dir / "quote_rest_cache")
        if not requests.empty:
            request_groups = requests.drop_duplicates(
                ["options_ticker", "window_start", "window_end"]
            ).reset_index(drop=True)
            rest_results: list[tuple[int, pd.Series, pd.DataFrame]] = []
            if quote_workers == 1:
                with httpx.Client(timeout=config.massive_request_timeout_seconds) as client:
                    for request_index, request in request_groups.iterrows():
                        normalized = _load_or_fetch_rest_quote_window(
                            client,
                            config,
                            request=request,
                            cache_dir=cache_dir,
                            limit=rest_limit,
                        )
                        rest_results.append((int(request_index), request, normalized))
            else:
                max_workers = min(int(quote_workers), max(1, len(request_groups)))
                with ThreadPoolExecutor(max_workers=max_workers) as executor:
                    futures = {
                        executor.submit(
                            _load_rest_quote_window_for_worker,
                            config,
                            request=request,
                            cache_dir=cache_dir,
                            limit=rest_limit,
                        ): (int(request_index), request)
                        for request_index, request in request_groups.iterrows()
                    }
                    for future in as_completed(futures):
                        request_index, request = futures[future]
                        rest_results.append((request_index, request, future.result()))
                rest_results.sort(key=lambda item: item[0])
            for _, request, normalized in rest_results:
                same_request = requests.loc[
                    requests["options_ticker"].astype(str).eq(str(request["options_ticker"]))
                    & requests["window_start"].astype(str).eq(str(request["window_start"]))
                    & requests["window_end"].astype(str).eq(str(request["window_end"]))
                ].copy()
                scanned += int(len(normalized))
                filtered = _filter_normalized_quotes(normalized, same_request)
                matched += int(len(filtered))
                if not filtered.empty:
                    matched_chunks.append(filtered)
    else:
        extraction_dates = sorted(
            pd.to_datetime(requests["quote_date"], errors="coerce").dt.date.dropna().unique()
        )
        for date_value in extraction_dates:
            date_requests = requests.loc[
                pd.to_datetime(requests["quote_date"], errors="coerce").dt.date.eq(date_value)
            ].copy()
            for chunk in _stream_massive_quote_chunks(
                config,
                date_value=date_value,
                chunksize=chunksize,
                aws_executable=aws_executable,
            ):
                scanned += int(len(chunk))
                normalized = normalize_option_quote_rows(chunk, quote_date=date_value)
                filtered = _filter_normalized_quotes(normalized, date_requests)
                matched += int(len(filtered))
                if not filtered.empty:
                    matched_chunks.append(filtered)
    quotes = pd.concat(matched_chunks, ignore_index=True) if matched_chunks else pd.DataFrame()
    marks = build_quote_window_marks(
        quotes,
        requests,
        stale_seconds=stale_seconds,
        wide_spread_threshold=wide_spread_threshold,
    )
    legs = build_quote_execution_leg_panel(marks)
    straddles = build_quote_straddle_execution_panel(marks)
    quote_ivar = build_quote_ivar_event_panel(straddles)
    quote_iv_surface = build_quote_iv_surface_panel(marks)
    quote_iv_surface_summary = build_quote_iv_surface_summary_panel(quote_iv_surface)
    quote_surface_ivar = build_quote_surface_ivar_event_panel(quote_iv_surface_summary)
    confidence = build_execution_confidence_panel(marks, quote_execution_route=route)
    if quotes.empty:
        quotes = _empty_normalized_quotes_frame()
    quotes.to_csv(quotes_path, index=False)
    marks.to_csv(marks_path, index=False)
    legs.to_csv(legs_path, index=False)
    straddles.to_csv(straddle_path, index=False)
    quote_ivar.to_csv(quote_ivar_path, index=False)
    quote_iv_surface.to_csv(quote_iv_surface_path, index=False)
    quote_iv_surface_summary.to_csv(quote_iv_surface_summary_path, index=False)
    quote_surface_ivar.to_csv(quote_surface_ivar_path, index=False)
    confidence.to_csv(confidence_path, index=False)
    report = QuoteExtractionReport(
        ok=True,
        route=route,
        metadata_only=False,
        raw_full_day_files_written=False,
        request_rows=int(len(requests)),
        event_count=int(requests["event_id"].nunique()) if "event_id" in requests else 0,
        quote_rows_scanned=scanned,
        quote_rows_matched=matched,
        dates=tuple(sorted(str(value) for value in requests["quote_date"].dropna().unique())),
        output_paths={
            "quote_window_requests": str(requests_path),
            "quote_window_quotes": str(quotes_path),
            "quote_window_marks": str(marks_path),
            "quote_execution_legs": str(legs_path),
            "quote_straddle_execution": str(straddle_path),
            "quote_ivar_event": str(quote_ivar_path),
            "quote_iv_surface": str(quote_iv_surface_path),
            "quote_iv_surface_summary": str(quote_iv_surface_summary_path),
            "quote_surface_ivar_event": str(quote_surface_ivar_path),
            "quote_execution_confidence": str(confidence_path),
            "quote_execution_report": str(report_path),
        },
        quote_workers=int(quote_workers),
        event_offset=int(event_offset),
        batch_label=batch_label,
    )
    report_path.write_text(json.dumps(report.as_dict(), indent=2, sort_keys=True), encoding="utf-8")
    return report

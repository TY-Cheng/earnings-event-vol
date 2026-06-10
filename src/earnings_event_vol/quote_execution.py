from __future__ import annotations

import gzip
import io
import json
import subprocess
from collections.abc import Iterable, Iterator, Sequence
from dataclasses import dataclass
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd

from earnings_event_vol.config import ProjectConfig
from earnings_event_vol.events import market_close_timestamp
from earnings_event_vol.massive import (
    massive_flat_file_aws_env,
    option_quotes_flat_file_key,
)

QUOTE_ROUTE_FLAT_FILE_FILTERED = "massive_quotes_v1_flat_file_filtered"
QUOTE_ROUTE_CSV_FIXTURE = "local_quotes_csv_filtered"
QUOTE_STATUS_OK = "ok"
QUOTE_STATUS_MISSING = "missing_quote"
QUOTE_STATUS_INVALID = "invalid_bid_ask"
QUOTE_STATUS_STALE = "stale_quote"
QUOTE_STATUS_WIDE = "wide_spread"


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
        }


def _to_datetime_et(value: object) -> pd.Timestamp:
    timestamp = pd.Timestamp(value)
    if timestamp.tzinfo is None:
        timestamp = timestamp.tz_localize("America/New_York")
    return timestamp.tz_convert("America/New_York")


def _to_date(value: object) -> date:
    parsed = pd.Timestamp(value)
    return date(parsed.year, parsed.month, parsed.day)


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
) -> pd.DataFrame:
    """Build event/contract/window requests for quote extraction."""
    required_contracts = {"event_id", "options_ticker", "entry_date", "exit_date"}
    required_windows = {"event_id"}
    missing_contracts = sorted(required_contracts - set(contracts.columns))
    missing_windows = sorted(required_windows - set(windows.columns))
    if missing_contracts:
        raise ValueError(f"contract frame missing required columns: {missing_contracts}")
    if missing_windows:
        raise ValueError(f"window frame missing required columns: {missing_windows}")
    event_ids = windows["event_id"].astype(str).drop_duplicates().tolist()
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
        return pd.DataFrame()
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


def build_execution_confidence_panel(marks: pd.DataFrame) -> pd.DataFrame:
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
                "quote_execution_route": QUOTE_ROUTE_FLAT_FILE_FILTERED,
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
) -> QuoteExtractionReport:
    out_dir.mkdir(parents=True, exist_ok=True)
    requests = build_quote_window_requests(
        contracts,
        windows,
        entry_lookback_seconds=entry_lookback_seconds,
        exit_lookback_seconds=exit_lookback_seconds,
        max_events=max_events,
    )
    if dates:
        date_set = set(dates)
        requests = requests.loc[
            pd.to_datetime(requests["quote_date"], errors="coerce").dt.date.isin(date_set)
        ].copy()
    requests_path = out_dir / "quote_window_requests.csv"
    marks_path = out_dir / "quote_window_marks.csv"
    confidence_path = out_dir / "quote_execution_confidence.csv"
    report_path = out_dir / "quote_execution_report.json"
    requests.to_csv(requests_path, index=False)
    route = QUOTE_ROUTE_CSV_FIXTURE if quote_csv_paths else QUOTE_ROUTE_FLAT_FILE_FILTERED
    if metadata_only:
        pd.DataFrame().to_csv(marks_path, index=False)
        build_execution_confidence_panel(pd.DataFrame()).to_csv(confidence_path, index=False)
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
                "quote_window_marks": str(marks_path),
                "quote_execution_confidence": str(confidence_path),
                "quote_execution_report": str(report_path),
            },
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
    else:
        extraction_dates = (
            sorted(set(dates))
            if dates
            else sorted(
                pd.to_datetime(requests["quote_date"], errors="coerce").dt.date.dropna().unique()
            )
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
    confidence = build_execution_confidence_panel(marks)
    marks.to_csv(marks_path, index=False)
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
            "quote_window_marks": str(marks_path),
            "quote_execution_confidence": str(confidence_path),
            "quote_execution_report": str(report_path),
        },
    )
    report_path.write_text(json.dumps(report.as_dict(), indent=2, sort_keys=True), encoding="utf-8")
    return report

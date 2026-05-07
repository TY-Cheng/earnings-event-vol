from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from collections.abc import Mapping
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import date
from pathlib import Path
from typing import Literal

import pandas as pd
import polars as pl

from earnings_event_vol.config import ProjectConfig, load_project_config
from earnings_event_vol.trade_proxy import (
    POST_OPEN_OPTION_VWAP_WINDOWS,
    TRADE_PROXY_PANEL_GRADE,
    TRADE_PROXY_STATUS_FETCH_FAILED,
    attach_trade_proxy_local_iv,
    build_exit_preclose_option_vwap_frame,
    build_post_open_option_vwap_frame,
    build_proxy_straddle_diagnostics,
    build_trade_proxy_ivar_inputs,
    build_trade_proxy_price_frame,
    edge_decile_diagnostics,
    extract_trade_proxy_event_panel,
    fetch_massive_option_second_aggregates,
    filter_pre_cutoff_buffer,
    normalize_second_aggregates,
    summarize_trade_proxy_panel,
)

PARQUET_COMPRESSION: Literal["zstd"] = "zstd"
SECOND_AGG_REQUIRED_COLUMNS = {
    "options_ticker",
    "timestamp_utc",
    "timestamp_et",
    "option_close",
    "option_vwap",
    "volume",
    "transactions",
}


def _progress(message: str) -> None:
    print(f"[trade-proxy] {message}", file=sys.stderr, flush=True)


def _write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")


def _json_params_match(path: Path, expected: Mapping[str, object]) -> bool:
    if not path.exists() or path.stat().st_size <= 0:
        return False
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return False
    return bool(payload.get("pipeline_params") == dict(expected))


def _write_parquet(path: Path, frame: pd.DataFrame | pl.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    pl_frame = pl.from_pandas(frame) if isinstance(frame, pd.DataFrame) else frame
    pl_frame.write_parquet(path, compression=PARQUET_COMPRESSION)


def _read_parquet(path: Path) -> pd.DataFrame:
    return pl.read_parquet(path).to_pandas()


def _parquet_is_usable(path: Path, *, required_columns: set[str]) -> bool:
    if not path.exists() or path.stat().st_size <= 0:
        return False
    try:
        schema = pl.scan_parquet(path).collect_schema()
    except Exception:
        return False
    return required_columns.issubset(set(schema.names()))


def _safe_partition_value(value: str) -> str:
    return "".join(char if char.isalnum() else "_" for char in value)


def _second_agg_bronze_path(
    config: ProjectConfig,
    *,
    option_ticker: str,
    entry_date: pd.Timestamp,
    cutoff_timestamp: pd.Timestamp,
    buffer_minutes: int,
) -> Path:
    day = entry_date.date().isoformat()
    safe_ticker = _safe_partition_value(option_ticker)
    cutoff = cutoff_timestamp.tz_convert("America/New_York").strftime("%H%M")
    return (
        config.bronze_data_dir
        / "massive"
        / "options_second_aggs"
        / f"date={day}"
        / f"cutoff={cutoff}"
        / f"buffer_minutes={buffer_minutes}"
        / f"options_ticker={safe_ticker}"
        / "part.parquet"
    )


def _post_open_second_agg_bronze_path(
    config: ProjectConfig,
    *,
    option_ticker: str,
    exit_date: pd.Timestamp,
    window_minutes: int,
) -> Path:
    day = exit_date.date().isoformat()
    safe_ticker = _safe_partition_value(option_ticker)
    return (
        config.bronze_data_dir
        / "massive"
        / "options_second_aggs_post_open"
        / f"date={day}"
        / f"window_minutes={window_minutes}"
        / f"options_ticker={safe_ticker}"
        / "part.parquet"
    )


def _exit_preclose_second_agg_bronze_path(
    config: ProjectConfig,
    *,
    option_ticker: str,
    exit_date: pd.Timestamp,
    lookback_seconds: int,
) -> Path:
    day = exit_date.date().isoformat()
    safe_ticker = _safe_partition_value(option_ticker)
    return (
        config.bronze_data_dir
        / "massive"
        / "options_second_aggs_exit_preclose"
        / f"date={day}"
        / f"lookback_seconds={lookback_seconds}"
        / f"options_ticker={safe_ticker}"
        / "part.parquet"
    )


def _fetch_one_contract(
    config: ProjectConfig,
    *,
    option_ticker: str,
    entry_date: pd.Timestamp,
    limit: int,
) -> tuple[str, pd.DataFrame, dict[str, object]]:
    try:
        raw = fetch_massive_option_second_aggregates(
            config,
            option_ticker=option_ticker,
            trade_date=entry_date.date(),
            limit=limit,
        )
        normalized = normalize_second_aggregates(raw, option_ticker=option_ticker)
        return (
            option_ticker,
            normalized,
            {
                "options_ticker": option_ticker,
                "status": "ok",
                "rows": int(len(normalized)),
                "entry_date": entry_date.date(),
            },
        )
    except Exception as exc:
        return (
            option_ticker,
            pd.DataFrame(),
            {
                "options_ticker": option_ticker,
                "status": TRADE_PROXY_STATUS_FETCH_FAILED,
                "rows": 0,
                "entry_date": entry_date.date(),
                "error": str(exc)[:300],
            },
        )


def _fetch_or_load_one_contract(
    config: ProjectConfig,
    *,
    option_ticker: str,
    entry_date: pd.Timestamp,
    cutoff_timestamp: pd.Timestamp,
    limit: int,
    buffer_minutes: int,
    force: bool,
) -> tuple[str, pd.DataFrame, dict[str, object]]:
    bronze_path = _second_agg_bronze_path(
        config,
        option_ticker=option_ticker,
        entry_date=entry_date,
        cutoff_timestamp=cutoff_timestamp,
        buffer_minutes=buffer_minutes,
    )
    repaired = False
    if not force and _parquet_is_usable(
        bronze_path,
        required_columns=SECOND_AGG_REQUIRED_COLUMNS,
    ):
        cached = _read_parquet(bronze_path)
        return (
            option_ticker,
            cached,
            {
                "options_ticker": option_ticker,
                "status": "ok",
                "rows": int(len(cached)),
                "entry_date": entry_date.date(),
                "bronze_path": str(bronze_path),
                "cutoff_timestamp": cutoff_timestamp.isoformat(),
                "buffer_minutes": buffer_minutes,
                "raw_rows": None,
                "cache_status": "hit",
            },
        )
    if not force and bronze_path.exists():
        repaired = True
        bronze_path.unlink(missing_ok=True)

    option_ticker, normalized, report = _fetch_one_contract(
        config,
        option_ticker=option_ticker,
        entry_date=entry_date,
        limit=limit,
    )
    buffered = filter_pre_cutoff_buffer(
        normalized,
        cutoff_timestamp=cutoff_timestamp.to_pydatetime(),
        buffer_minutes=buffer_minutes,
    )
    report["bronze_path"] = str(bronze_path)
    report["cutoff_timestamp"] = cutoff_timestamp.isoformat()
    report["buffer_minutes"] = buffer_minutes
    report["raw_rows"] = int(len(normalized))
    report["rows"] = int(len(buffered))
    if report["status"] == "ok":
        _write_parquet(bronze_path, buffered)
        report["cache_status"] = "repaired" if repaired else "written"
    else:
        report["cache_status"] = "repair_failed" if repaired else "miss"
    return option_ticker, buffered, report


def _filter_post_open_window(
    bars: pd.DataFrame,
    *,
    exit_date: pd.Timestamp,
    window_minutes: int,
) -> pd.DataFrame:
    if bars.empty:
        return bars.copy()
    frame = bars.copy()
    frame["timestamp_et"] = pd.to_datetime(frame["timestamp_et"])
    if frame["timestamp_et"].dt.tz is None:
        raise ValueError("timestamp_et must be timezone-aware")
    frame["timestamp_et"] = frame["timestamp_et"].dt.tz_convert("America/New_York")
    open_ts = pd.Timestamp(f"{exit_date.date().isoformat()} 09:30:00", tz="America/New_York")
    end = open_ts + pd.Timedelta(minutes=window_minutes)
    return frame.loc[frame["timestamp_et"].between(open_ts, end, inclusive="both")].copy()


def _filter_exit_preclose_window(
    bars: pd.DataFrame,
    *,
    exit_date: pd.Timestamp,
    lookback_seconds: int,
) -> pd.DataFrame:
    if bars.empty:
        return bars.copy()
    frame = bars.copy()
    frame["timestamp_et"] = pd.to_datetime(frame["timestamp_et"])
    if frame["timestamp_et"].dt.tz is None:
        raise ValueError("timestamp_et must be timezone-aware")
    frame["timestamp_et"] = frame["timestamp_et"].dt.tz_convert("America/New_York")
    close_ts = pd.Timestamp(f"{exit_date.date().isoformat()} 16:00:00", tz="America/New_York")
    start = close_ts - pd.Timedelta(seconds=lookback_seconds)
    return frame.loc[frame["timestamp_et"].between(start, close_ts, inclusive="both")].copy()


def _fetch_or_load_one_post_open_contract(
    config: ProjectConfig,
    *,
    option_ticker: str,
    exit_date: pd.Timestamp,
    limit: int,
    window_minutes: int,
    force: bool,
) -> tuple[tuple[str, pd.Timestamp], pd.DataFrame, dict[str, object]]:
    bronze_path = _post_open_second_agg_bronze_path(
        config,
        option_ticker=option_ticker,
        exit_date=exit_date,
        window_minutes=window_minutes,
    )
    repaired = False
    if not force and _parquet_is_usable(
        bronze_path,
        required_columns=SECOND_AGG_REQUIRED_COLUMNS,
    ):
        cached = _read_parquet(bronze_path)
        return (
            (option_ticker, exit_date),
            cached,
            {
                "options_ticker": option_ticker,
                "status": "ok",
                "rows": int(len(cached)),
                "exit_date": exit_date.date(),
                "bronze_path": str(bronze_path),
                "window_minutes": window_minutes,
                "raw_rows": None,
                "cache_status": "hit",
            },
        )
    if not force and bronze_path.exists():
        repaired = True
        bronze_path.unlink(missing_ok=True)

    option_ticker, normalized, report = _fetch_one_contract(
        config,
        option_ticker=option_ticker,
        entry_date=exit_date,
        limit=limit,
    )
    buffered = _filter_post_open_window(
        normalized,
        exit_date=exit_date,
        window_minutes=window_minutes,
    )
    report["exit_date"] = exit_date.date()
    report["bronze_path"] = str(bronze_path)
    report["window_minutes"] = window_minutes
    report["raw_rows"] = int(len(normalized))
    report["rows"] = int(len(buffered))
    if report["status"] == "ok":
        _write_parquet(bronze_path, buffered)
        report["cache_status"] = "repaired" if repaired else "written"
    else:
        report["cache_status"] = "repair_failed" if repaired else "miss"
    return (option_ticker, exit_date), buffered, report


def _fetch_or_load_one_exit_preclose_contract(
    config: ProjectConfig,
    *,
    option_ticker: str,
    exit_date: pd.Timestamp,
    limit: int,
    lookback_seconds: int,
    force: bool,
) -> tuple[tuple[str, pd.Timestamp], pd.DataFrame, dict[str, object]]:
    bronze_path = _exit_preclose_second_agg_bronze_path(
        config,
        option_ticker=option_ticker,
        exit_date=exit_date,
        lookback_seconds=lookback_seconds,
    )
    repaired = False
    if not force and _parquet_is_usable(
        bronze_path,
        required_columns=SECOND_AGG_REQUIRED_COLUMNS,
    ):
        cached = _read_parquet(bronze_path)
        return (
            (option_ticker, exit_date),
            cached,
            {
                "options_ticker": option_ticker,
                "status": "ok",
                "rows": int(len(cached)),
                "exit_date": exit_date.date(),
                "bronze_path": str(bronze_path),
                "lookback_seconds": lookback_seconds,
                "raw_rows": None,
                "cache_status": "hit",
            },
        )
    if not force and bronze_path.exists():
        repaired = True
        bronze_path.unlink(missing_ok=True)

    option_ticker, normalized, report = _fetch_one_contract(
        config,
        option_ticker=option_ticker,
        entry_date=exit_date,
        limit=limit,
    )
    buffered = _filter_exit_preclose_window(
        normalized,
        exit_date=exit_date,
        lookback_seconds=lookback_seconds,
    )
    report["exit_date"] = exit_date.date()
    report["bronze_path"] = str(bronze_path)
    report["lookback_seconds"] = lookback_seconds
    report["raw_rows"] = int(len(normalized))
    report["rows"] = int(len(buffered))
    if report["status"] == "ok":
        _write_parquet(bronze_path, buffered)
        report["cache_status"] = "repaired" if repaired else "written"
    else:
        report["cache_status"] = "repair_failed" if repaired else "miss"
    return (option_ticker, exit_date), buffered, report


def _fetch_second_aggregate_bars(
    config: ProjectConfig,
    contracts: pd.DataFrame,
    *,
    jobs: int,
    limit: int,
    buffer_minutes: int,
    force: bool,
) -> tuple[dict[str, pd.DataFrame], pd.DataFrame]:
    requests = (
        contracts[["options_ticker", "entry_date", "event_entry_timestamp"]]
        .dropna()
        .drop_duplicates()
        .sort_values(["entry_date", "event_entry_timestamp", "options_ticker"])
    )
    bar_frames: dict[str, pd.DataFrame] = {}
    reports: list[dict[str, object]] = []
    report_columns = [
        "options_ticker",
        "status",
        "rows",
        "entry_date",
        "bronze_path",
        "cutoff_timestamp",
        "buffer_minutes",
        "raw_rows",
        "cache_status",
        "failure_reason",
    ]
    total = int(len(requests))
    if requests.empty:
        _progress("second-agg fetch: no candidate contracts")
        return bar_frames, pd.DataFrame(columns=report_columns)
    _progress(
        f"second-agg fetch start: contracts={total} jobs={jobs} "
        f"buffer_minutes={buffer_minutes} force={force}"
    )
    counts: Counter[str] = Counter()
    checkpoint = max(1, total // 10)

    def note_progress(completed: int, option_ticker: str, report: dict[str, object]) -> None:
        cache_status = str(report.get("cache_status") or "unknown")
        status = str(report.get("status") or "unknown")
        counts[f"{cache_status}:{status}"] += 1
        if completed == total or completed % checkpoint == 0:
            _progress(
                f"second-agg fetch {completed}/{total}: latest={option_ticker} "
                f"rows={report.get('rows')} raw_rows={report.get('raw_rows')} "
                f"counts={dict(counts)}"
            )

    if jobs <= 1:
        for completed, row in enumerate(requests.to_dict("records"), start=1):
            option_ticker, frame, report = _fetch_or_load_one_contract(
                config,
                option_ticker=str(row["options_ticker"]),
                entry_date=pd.Timestamp(row["entry_date"]),
                cutoff_timestamp=pd.Timestamp(row["event_entry_timestamp"]),
                limit=limit,
                buffer_minutes=buffer_minutes,
                force=force,
            )
            bar_frames[option_ticker] = frame
            reports.append(report)
            note_progress(completed, option_ticker, report)
        return (
            bar_frames,
            pd.DataFrame(reports, columns=report_columns).sort_values(
                ["entry_date", "options_ticker"]
            ),
        )

    with ThreadPoolExecutor(max_workers=jobs) as executor:
        futures = [
            executor.submit(
                _fetch_or_load_one_contract,
                config,
                option_ticker=str(row["options_ticker"]),
                entry_date=pd.Timestamp(row["entry_date"]),
                cutoff_timestamp=pd.Timestamp(row["event_entry_timestamp"]),
                limit=limit,
                buffer_minutes=buffer_minutes,
                force=force,
            )
            for row in requests.to_dict("records")
        ]
        for completed, future in enumerate(as_completed(futures), start=1):
            option_ticker, frame, report = future.result()
            bar_frames[option_ticker] = frame
            reports.append(report)
            note_progress(completed, option_ticker, report)
    return (
        bar_frames,
        pd.DataFrame(reports, columns=report_columns).sort_values(["entry_date", "options_ticker"]),
    )


def _fetch_post_open_second_aggregate_bars(
    config: ProjectConfig,
    selected_straddles: pd.DataFrame,
    *,
    jobs: int,
    limit: int,
    window_minutes: int,
    force: bool,
) -> tuple[dict[tuple[str, date], pd.DataFrame], pd.DataFrame]:
    report_columns = [
        "options_ticker",
        "status",
        "rows",
        "exit_date",
        "bronze_path",
        "window_minutes",
        "raw_rows",
        "cache_status",
        "failure_reason",
    ]
    if selected_straddles.empty:
        _progress("post-open second-agg fetch: no selected straddles")
        return {}, pd.DataFrame(columns=report_columns)
    requests: list[dict[str, object]] = []
    for row in selected_straddles.to_dict("records"):
        exit_date = pd.Timestamp(row["exit_date"])
        for column in ("call_options_ticker", "put_options_ticker"):
            value = row.get(column)
            if value is not None and not pd.isna(value):
                requests.append({"options_ticker": str(value), "exit_date": exit_date})
    request_frame = (
        pd.DataFrame(requests).drop_duplicates().sort_values(["exit_date", "options_ticker"])
        if requests
        else pd.DataFrame(columns=["options_ticker", "exit_date"])
    )
    total = int(len(request_frame))
    if request_frame.empty:
        _progress("post-open second-agg fetch: no option legs")
        return {}, pd.DataFrame(columns=report_columns)
    _progress(
        f"post-open second-agg fetch start: contracts={total} jobs={jobs} "
        f"window_minutes={window_minutes} force={force}"
    )
    bar_frames: dict[tuple[str, date], pd.DataFrame] = {}
    reports: list[dict[str, object]] = []
    counts: Counter[str] = Counter()
    checkpoint = max(1, total // 10)

    def note_progress(completed: int, option_ticker: str, report: dict[str, object]) -> None:
        cache_status = str(report.get("cache_status") or "unknown")
        status = str(report.get("status") or "unknown")
        counts[f"{cache_status}:{status}"] += 1
        if completed == total or completed % checkpoint == 0:
            _progress(
                f"post-open second-agg fetch {completed}/{total}: latest={option_ticker} "
                f"rows={report.get('rows')} raw_rows={report.get('raw_rows')} "
                f"counts={dict(counts)}"
            )

    if jobs <= 1:
        for completed, row in enumerate(request_frame.to_dict("records"), start=1):
            (option_ticker, exit_date), frame, report = _fetch_or_load_one_post_open_contract(
                config,
                option_ticker=str(row["options_ticker"]),
                exit_date=pd.Timestamp(row["exit_date"]),
                limit=limit,
                window_minutes=window_minutes,
                force=force,
            )
            bar_frames[(option_ticker, exit_date.date())] = frame
            reports.append(report)
            note_progress(completed, option_ticker, report)
    else:
        with ThreadPoolExecutor(max_workers=jobs) as executor:
            futures = [
                executor.submit(
                    _fetch_or_load_one_post_open_contract,
                    config,
                    option_ticker=str(row["options_ticker"]),
                    exit_date=pd.Timestamp(row["exit_date"]),
                    limit=limit,
                    window_minutes=window_minutes,
                    force=force,
                )
                for row in request_frame.to_dict("records")
            ]
            for completed, future in enumerate(as_completed(futures), start=1):
                (option_ticker, exit_date), frame, report = future.result()
                bar_frames[(option_ticker, exit_date.date())] = frame
                reports.append(report)
                note_progress(completed, option_ticker, report)
    return (
        bar_frames,
        pd.DataFrame(reports, columns=report_columns).sort_values(["exit_date", "options_ticker"]),
    )


def _fetch_exit_preclose_second_aggregate_bars(
    config: ProjectConfig,
    selected_straddles: pd.DataFrame,
    *,
    jobs: int,
    limit: int,
    lookback_seconds: int,
    force: bool,
) -> tuple[dict[tuple[str, date], pd.DataFrame], pd.DataFrame]:
    report_columns = [
        "options_ticker",
        "status",
        "rows",
        "exit_date",
        "bronze_path",
        "lookback_seconds",
        "raw_rows",
        "cache_status",
        "failure_reason",
    ]
    if selected_straddles.empty:
        _progress("exit-preclose second-agg fetch: no selected straddles")
        return {}, pd.DataFrame(columns=report_columns)
    requests: list[dict[str, object]] = []
    for row in selected_straddles.to_dict("records"):
        exit_date = pd.Timestamp(row["exit_date"])
        for column in ("call_options_ticker", "put_options_ticker"):
            value = row.get(column)
            if value is not None and not pd.isna(value):
                requests.append({"options_ticker": str(value), "exit_date": exit_date})
    request_frame = (
        pd.DataFrame(requests).drop_duplicates().sort_values(["exit_date", "options_ticker"])
        if requests
        else pd.DataFrame(columns=["options_ticker", "exit_date"])
    )
    total = int(len(request_frame))
    if request_frame.empty:
        _progress("exit-preclose second-agg fetch: no option legs")
        return {}, pd.DataFrame(columns=report_columns)
    _progress(
        f"exit-preclose second-agg fetch start: contracts={total} jobs={jobs} "
        f"lookback_seconds={lookback_seconds} force={force}"
    )
    bar_frames: dict[tuple[str, date], pd.DataFrame] = {}
    reports: list[dict[str, object]] = []
    counts: Counter[str] = Counter()
    checkpoint = max(1, total // 10)

    def note_progress(completed: int, option_ticker: str, report: dict[str, object]) -> None:
        cache_status = str(report.get("cache_status") or "unknown")
        status = str(report.get("status") or "unknown")
        counts[f"{cache_status}:{status}"] += 1
        if completed == total or completed % checkpoint == 0:
            _progress(
                f"exit-preclose second-agg fetch {completed}/{total}: latest={option_ticker} "
                f"rows={report.get('rows')} raw_rows={report.get('raw_rows')} "
                f"counts={dict(counts)}"
            )

    if jobs <= 1:
        for completed, row in enumerate(request_frame.to_dict("records"), start=1):
            (option_ticker, exit_date), frame, report = _fetch_or_load_one_exit_preclose_contract(
                config,
                option_ticker=str(row["options_ticker"]),
                exit_date=pd.Timestamp(row["exit_date"]),
                limit=limit,
                lookback_seconds=lookback_seconds,
                force=force,
            )
            bar_frames[(option_ticker, exit_date.date())] = frame
            reports.append(report)
            note_progress(completed, option_ticker, report)
    else:
        with ThreadPoolExecutor(max_workers=jobs) as executor:
            futures = [
                executor.submit(
                    _fetch_or_load_one_exit_preclose_contract,
                    config,
                    option_ticker=str(row["options_ticker"]),
                    exit_date=pd.Timestamp(row["exit_date"]),
                    limit=limit,
                    lookback_seconds=lookback_seconds,
                    force=force,
                )
                for row in request_frame.to_dict("records")
            ]
            for completed, future in enumerate(as_completed(futures), start=1):
                (option_ticker, exit_date), frame, report = future.result()
                bar_frames[(option_ticker, exit_date.date())] = frame
                reports.append(report)
                note_progress(completed, option_ticker, report)
    return (
        bar_frames,
        pd.DataFrame(reports, columns=report_columns).sort_values(["exit_date", "options_ticker"]),
    )


def build_trade_proxy_panel(
    *,
    config: ProjectConfig,
    out_root: Path,
    force: bool,
    max_events: int | None,
    max_contracts: int | None,
    lookback_seconds: int,
    second_agg_buffer_minutes: int,
    price_field: str,
    jobs: int,
    rest_limit: int,
    haircut_fraction: float,
    refresh_bronze: bool,
) -> dict[str, object]:
    windows_path = config.silver_data_dir / "event_windows" / "event_windows.parquet"
    contracts_path = config.silver_data_dir / "contracts" / "event_contract_candidates.parquet"
    if not windows_path.exists() or not contracts_path.exists():
        raise FileNotFoundError(
            "trade-proxy-panel requires an existing pilot-panel run with event windows and "
            "candidate contracts."
        )

    silver_proxy_dir = config.silver_data_dir / "trade_proxy"
    gold_panel_dir = config.gold_data_dir / "event_panel"
    proxy_prices_path = silver_proxy_dir / "trade_proxy_option_prices.parquet"
    iv_estimates_path = silver_proxy_dir / "trade_proxy_contract_iv_estimates.parquet"
    ivar_inputs_path = silver_proxy_dir / "trade_proxy_ivar_inputs.parquet"
    exit_preclose_prices_path = silver_proxy_dir / "exit_preclose_option_vwap_prices.parquet"
    post_open_exit_prices_path = silver_proxy_dir / "post_open_option_vwap_exit_prices.parquet"
    panel_path = gold_panel_dir / "trade_proxy_event_panel.parquet"
    diagnostics_path = out_root / "trade_proxy_panel" / "trade_proxy_panel_report.json"
    params = {
        "stage": "trade-proxy-panel",
        "max_events": max_events,
        "max_contracts": max_contracts,
        "lookback_seconds": lookback_seconds,
        "second_agg_buffer_minutes": second_agg_buffer_minutes,
        "price_field": price_field,
        "entry_price_method": "preclose_15m_option_second_agg_vwap",
        "c2c_exit_price_method": "exit_preclose_15m_option_second_agg_vwap",
        "post_open_option_vwap_windows": [window[0] for window in POST_OPEN_OPTION_VWAP_WINDOWS],
    }
    if not force and panel_path.exists() and _json_params_match(diagnostics_path, params):
        return {
            "status": "skipped",
            "reason": "outputs_exist_params_match",
            "panel_grade": TRADE_PROXY_PANEL_GRADE,
            "outputs": {
                "trade_proxy_event_panel": str(panel_path),
                "trade_proxy_panel_report": str(diagnostics_path),
            },
        }

    windows = _read_parquet(windows_path)
    contracts = _read_parquet(contracts_path)
    if max_events is not None:
        keep_events = windows["event_id"].head(max_events).tolist()
        windows = windows.loc[windows["event_id"].isin(keep_events)].copy()
        contracts = contracts.loc[contracts["event_id"].isin(keep_events)].copy()
    contracts = contracts.loc[contracts["eligible_for_quote_pool"].astype(bool)].copy()
    contracts = contracts.merge(
        windows[["event_id", "event_entry_timestamp", "s_before", "s_after", "rvar_event"]],
        on="event_id",
        how="inner",
    )
    if max_contracts is not None:
        contracts = contracts.head(max_contracts).copy()

    bar_frames, fetch_report = _fetch_second_aggregate_bars(
        config,
        contracts,
        jobs=jobs,
        limit=rest_limit,
        buffer_minutes=second_agg_buffer_minutes,
        force=refresh_bronze,
    )
    proxy_prices = build_trade_proxy_price_frame(
        contracts,
        bar_frames,
        lookback_seconds=lookback_seconds,
        price_field=price_field,
    )
    iv_estimates = attach_trade_proxy_local_iv(proxy_prices, windows)
    ivar_inputs = build_trade_proxy_ivar_inputs(iv_estimates, windows)
    panel = extract_trade_proxy_event_panel(ivar_inputs, windows)
    selected_straddles = build_proxy_straddle_diagnostics(
        iv_estimates,
        windows,
        haircut_fraction=haircut_fraction,
    )
    exit_preclose_bar_frames, exit_preclose_fetch_report = (
        _fetch_exit_preclose_second_aggregate_bars(
            config,
            selected_straddles,
            jobs=jobs,
            limit=rest_limit,
            lookback_seconds=lookback_seconds,
            force=refresh_bronze,
        )
    )
    exit_preclose_prices = build_exit_preclose_option_vwap_frame(
        selected_straddles,
        exit_preclose_bar_frames,
        price_field=price_field,
        lookback_seconds=lookback_seconds,
    )
    post_open_window_minutes = max(window[2] for window in POST_OPEN_OPTION_VWAP_WINDOWS)
    post_open_bar_frames, post_open_fetch_report = _fetch_post_open_second_aggregate_bars(
        config,
        selected_straddles,
        jobs=jobs,
        limit=rest_limit,
        window_minutes=post_open_window_minutes,
        force=refresh_bronze,
    )
    post_open_exit_prices = build_post_open_option_vwap_frame(
        selected_straddles,
        post_open_bar_frames,
        price_field=price_field,
    )
    straddle_diagnostics = build_proxy_straddle_diagnostics(
        iv_estimates,
        windows,
        exit_preclose_option_prices=exit_preclose_prices,
        post_open_option_prices=post_open_exit_prices,
        haircut_fraction=haircut_fraction,
    )
    edge_deciles = edge_decile_diagnostics(panel)

    _write_parquet(proxy_prices_path, proxy_prices)
    _write_parquet(iv_estimates_path, iv_estimates)
    _write_parquet(ivar_inputs_path, ivar_inputs)
    _write_parquet(exit_preclose_prices_path, exit_preclose_prices)
    _write_parquet(post_open_exit_prices_path, post_open_exit_prices)
    _write_parquet(panel_path, panel)
    report_dir = out_root / "trade_proxy_panel"
    report_dir.mkdir(parents=True, exist_ok=True)
    fetch_report.to_csv(report_dir / "second_aggregate_fetch_report.csv", index=False)
    exit_preclose_fetch_report.to_csv(
        report_dir / "exit_preclose_second_aggregate_fetch_report.csv", index=False
    )
    post_open_fetch_report.to_csv(
        report_dir / "post_open_second_aggregate_fetch_report.csv", index=False
    )
    straddle_diagnostics.to_csv(report_dir / "trade_proxy_straddle_diagnostics.csv", index=False)
    edge_deciles.to_csv(report_dir / "trade_proxy_edge_deciles.csv", index=False)

    report = summarize_trade_proxy_panel(
        panel=panel,
        proxy_prices=iv_estimates,
        straddle_diagnostics=straddle_diagnostics,
        lookback_seconds=lookback_seconds,
        price_field=price_field,
    )
    report["pipeline_params"] = params
    report["second_agg_buffer_minutes"] = second_agg_buffer_minutes
    report["bronze_second_aggregate_cache"] = {
        "dataset": "options_second_aggs",
        "root": str(config.bronze_data_dir / "massive" / "options_second_aggs"),
        "files": int(fetch_report["bronze_path"].nunique()) if "bronze_path" in fetch_report else 0,
        "cache_status_counts": fetch_report["cache_status"].value_counts().to_dict()
        if "cache_status" in fetch_report
        else {},
    }
    report["bronze_post_open_second_aggregate_cache"] = {
        "dataset": "options_second_aggs_post_open",
        "root": str(config.bronze_data_dir / "massive" / "options_second_aggs_post_open"),
        "files": int(post_open_fetch_report["bronze_path"].nunique())
        if "bronze_path" in post_open_fetch_report
        else 0,
        "cache_status_counts": post_open_fetch_report["cache_status"].value_counts().to_dict()
        if "cache_status" in post_open_fetch_report
        else {},
    }
    report["bronze_exit_preclose_second_aggregate_cache"] = {
        "dataset": "options_second_aggs_exit_preclose",
        "root": str(config.bronze_data_dir / "massive" / "options_second_aggs_exit_preclose"),
        "files": int(exit_preclose_fetch_report["bronze_path"].nunique())
        if "bronze_path" in exit_preclose_fetch_report
        else 0,
        "cache_status_counts": exit_preclose_fetch_report["cache_status"].value_counts().to_dict()
        if "cache_status" in exit_preclose_fetch_report
        else {},
    }
    report["outputs"] = {
        "trade_proxy_second_aggs_bronze_root": str(
            config.bronze_data_dir / "massive" / "options_second_aggs"
        ),
        "trade_proxy_option_prices": str(proxy_prices_path),
        "trade_proxy_contract_iv_estimates": str(iv_estimates_path),
        "trade_proxy_ivar_inputs": str(ivar_inputs_path),
        "exit_preclose_option_vwap_prices": str(exit_preclose_prices_path),
        "post_open_option_vwap_exit_prices": str(post_open_exit_prices_path),
        "trade_proxy_event_panel": str(panel_path),
        "second_aggregate_fetch_report": str(report_dir / "second_aggregate_fetch_report.csv"),
        "exit_preclose_second_aggregate_fetch_report": str(
            report_dir / "exit_preclose_second_aggregate_fetch_report.csv"
        ),
        "post_open_second_aggregate_fetch_report": str(
            report_dir / "post_open_second_aggregate_fetch_report.csv"
        ),
        "trade_proxy_straddle_diagnostics": str(
            report_dir / "trade_proxy_straddle_diagnostics.csv"
        ),
        "trade_proxy_edge_deciles": str(report_dir / "trade_proxy_edge_deciles.csv"),
        "trade_proxy_panel_report": str(diagnostics_path),
    }
    _write_json(diagnostics_path, report)
    return report


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-root", type=Path, default=Path("artifacts/data_pipeline"))
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--max-events", type=int)
    parser.add_argument("--max-contracts", type=int)
    parser.add_argument("--lookback-seconds", type=int, default=900)
    parser.add_argument("--second-agg-buffer-minutes", type=int, default=60)
    parser.add_argument(
        "--price-field", choices=["option_vwap", "option_close"], default="option_vwap"
    )
    parser.add_argument("--jobs", type=int, default=1)
    parser.add_argument("--rest-limit", type=int, default=50_000)
    parser.add_argument("--haircut-fraction", type=float, default=0.10)
    parser.add_argument(
        "--refresh-bronze",
        action="store_true",
        help="Re-fetch second-agg bronze caches instead of reusing valid cached Parquet.",
    )
    args = parser.parse_args()

    if args.jobs <= 0:
        raise ValueError("--jobs must be positive.")
    if args.lookback_seconds <= 0:
        raise ValueError("--lookback-seconds must be positive.")
    if args.second_agg_buffer_minutes <= 0:
        raise ValueError("--second-agg-buffer-minutes must be positive.")
    if args.haircut_fraction < 0:
        raise ValueError("--haircut-fraction must be nonnegative.")
    config = load_project_config()
    report = build_trade_proxy_panel(
        config=config,
        out_root=args.out_root,
        force=args.force,
        max_events=args.max_events,
        max_contracts=args.max_contracts,
        lookback_seconds=args.lookback_seconds,
        second_agg_buffer_minutes=args.second_agg_buffer_minutes,
        price_field=args.price_field,
        jobs=args.jobs,
        rest_limit=args.rest_limit,
        haircut_fraction=args.haircut_fraction,
        refresh_bronze=args.refresh_bronze,
    )
    print(json.dumps(report, indent=2, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

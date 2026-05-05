from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Literal

import pandas as pd
import polars as pl

from earnings_event_vol.config import ProjectConfig, load_project_config
from earnings_event_vol.massive import (
    _run_head_object_command,
    build_download_file_command,
    massive_flat_file_aws_env,
    option_flat_file_key,
)
from earnings_event_vol.trade_proxy import (
    TRADE_PROXY_PANEL_GRADE,
    TRADE_PROXY_STATUS_FETCH_FAILED,
    attach_trade_proxy_local_iv,
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
OPTIONS_DAY_AGG_REQUIRED_COLUMNS = {"ticker", "close"}


def _progress(message: str) -> None:
    print(f"[trade-proxy] {message}", file=sys.stderr, flush=True)


def _write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")


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


def _tmp_flat_file_path(config: ProjectConfig, *, dataset: str, day: pd.Timestamp) -> Path:
    return (
        config.bronze_data_dir / "_tmp" / "massive" / dataset / f"{day.date().isoformat()}.csv.gz"
    )


def _options_day_agg_path(config: ProjectConfig, day: pd.Timestamp) -> Path:
    return (
        config.bronze_data_dir
        / "massive"
        / "options_day_aggs"
        / f"date={day.date().isoformat()}"
        / "part.parquet"
    )


def _ensure_options_day_agg_file(config: ProjectConfig, day: pd.Timestamp) -> str | None:
    destination = _options_day_agg_path(config, day)
    repaired = False
    if _parquet_is_usable(destination, required_columns=OPTIONS_DAY_AGG_REQUIRED_COLUMNS):
        return "hit"
    if destination.exists():
        repaired = True
        destination.unlink(missing_ok=True)
    key = option_flat_file_key(
        config,
        year=day.year,
        month=day.month,
        date=day.date().isoformat(),
    )
    tmp_path = _tmp_flat_file_path(config, dataset="options_day_aggs", day=day)
    tmp_path.parent.mkdir(parents=True, exist_ok=True)
    result = _run_head_object_command(
        build_download_file_command(config, key=key, destination=tmp_path),
        massive_flat_file_aws_env(config),
        config.massive_request_timeout_seconds * 8,
    )
    if result.returncode != 0:
        tmp_path.unlink(missing_ok=True)
        return None
    frame = pl.read_csv(tmp_path).with_columns(
        [
            pl.lit(day.date().isoformat()).alias("source_date"),
            pl.lit("options_day_aggs").alias("source_dataset"),
            pl.lit(key).alias("source_key"),
        ]
    )
    destination.parent.mkdir(parents=True, exist_ok=True)
    tmp_parquet = destination.with_name(f".{destination.stem}.tmp.parquet")
    frame.write_parquet(tmp_parquet, compression=PARQUET_COMPRESSION)
    tmp_parquet.replace(destination)
    tmp_path.unlink(missing_ok=True)
    return "repaired" if repaired else "downloaded"


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


def _load_option_exit_prices(config: ProjectConfig, contracts: pd.DataFrame) -> pd.DataFrame:
    columns = ["options_ticker", "date", "option_close", "source_dataset"]
    if contracts.empty:
        _progress("exit day-agg prices: no contracts")
        return pd.DataFrame(columns=columns)
    rows: list[pd.DataFrame] = []
    exit_groups = list(contracts.groupby("exit_date"))
    _progress(
        f"exit day-agg prices start: exit_dates={len(exit_groups)} contract_rows={len(contracts)}"
    )
    for index, (exit_date, group) in enumerate(exit_groups, start=1):
        day = pd.Timestamp(exit_date)
        tickers = sorted(set(group["options_ticker"].astype(str)))
        cache_status = _ensure_options_day_agg_file(config, day)
        if cache_status is None:
            _progress(
                f"exit day-agg prices {index}/{len(exit_groups)}: "
                f"date={day.date().isoformat()} status=missing contracts={len(tickers)}"
            )
            continue
        path = _options_day_agg_path(config, day)
        frame = (
            pl.scan_parquet(path)
            .filter(pl.col("ticker").is_in(tickers))
            .select(
                [
                    pl.col("ticker").alias("options_ticker"),
                    pl.lit(day.date()).alias("date"),
                    pl.col("close").cast(pl.Float64, strict=False).alias("option_close"),
                    pl.lit("options_day_aggs").alias("source_dataset"),
                ]
            )
            .collect()
        )
        _progress(
            f"exit day-agg prices {index}/{len(exit_groups)}: "
            f"date={day.date().isoformat()} status={cache_status} "
            f"contracts={len(tickers)} matched={frame.height}"
        )
        if frame.height:
            rows.append(frame.to_pandas())
    if not rows:
        _progress("exit day-agg prices done: matched_rows=0")
        return pd.DataFrame(columns=columns)
    out = pd.concat(rows, ignore_index=True)[columns]
    _progress(f"exit day-agg prices done: matched_rows={len(out)}")
    return out


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
    panel_path = gold_panel_dir / "trade_proxy_event_panel.parquet"
    diagnostics_path = out_root / "trade_proxy_panel" / "trade_proxy_panel_report.json"
    if not force and panel_path.exists() and diagnostics_path.exists():
        return {
            "status": "skipped",
            "reason": "outputs_exist",
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
        force=force,
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
    option_exit_prices = _load_option_exit_prices(config, contracts)
    straddle_diagnostics = build_proxy_straddle_diagnostics(
        iv_estimates,
        windows,
        option_exit_prices=option_exit_prices,
        haircut_fraction=haircut_fraction,
    )
    edge_deciles = edge_decile_diagnostics(panel)

    _write_parquet(proxy_prices_path, proxy_prices)
    _write_parquet(iv_estimates_path, iv_estimates)
    _write_parquet(ivar_inputs_path, ivar_inputs)
    _write_parquet(panel_path, panel)
    report_dir = out_root / "trade_proxy_panel"
    report_dir.mkdir(parents=True, exist_ok=True)
    fetch_report.to_csv(report_dir / "second_aggregate_fetch_report.csv", index=False)
    straddle_diagnostics.to_csv(report_dir / "trade_proxy_straddle_diagnostics.csv", index=False)
    edge_deciles.to_csv(report_dir / "trade_proxy_edge_deciles.csv", index=False)

    report = summarize_trade_proxy_panel(
        panel=panel,
        proxy_prices=iv_estimates,
        straddle_diagnostics=straddle_diagnostics,
        lookback_seconds=lookback_seconds,
        price_field=price_field,
    )
    report["second_agg_buffer_minutes"] = second_agg_buffer_minutes
    report["bronze_second_aggregate_cache"] = {
        "dataset": "options_second_aggs",
        "root": str(config.bronze_data_dir / "massive" / "options_second_aggs"),
        "files": int(fetch_report["bronze_path"].nunique()) if "bronze_path" in fetch_report else 0,
        "cache_status_counts": fetch_report["cache_status"].value_counts().to_dict()
        if "cache_status" in fetch_report
        else {},
    }
    report["outputs"] = {
        "trade_proxy_second_aggs_bronze_root": str(
            config.bronze_data_dir / "massive" / "options_second_aggs"
        ),
        "trade_proxy_option_prices": str(proxy_prices_path),
        "trade_proxy_contract_iv_estimates": str(iv_estimates_path),
        "trade_proxy_ivar_inputs": str(ivar_inputs_path),
        "trade_proxy_event_panel": str(panel_path),
        "second_aggregate_fetch_report": str(report_dir / "second_aggregate_fetch_report.csv"),
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
    )
    print(json.dumps(report, indent=2, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

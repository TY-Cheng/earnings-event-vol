from __future__ import annotations

import json
import subprocess
import sys
import time
from collections import Counter, deque
from collections.abc import Callable, Mapping, Sequence
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from datetime import date, timedelta
from pathlib import Path

import httpx
import pandas as pd
import polars as pl

from earnings_event_vol.config import ProjectConfig
from earnings_event_vol.contract_reference import (
    CONTRACT_REFERENCE_SCHEMA_VERSION,
    REFERENCE_STATUS_VALIDATED,
    apply_contract_reference_validation,
    fetch_massive_option_contract_reference,
)
from earnings_event_vol.data_audit import audit_data_fields
from earnings_event_vol.earnings_calendar import build_earnings_calendar_candidates
from earnings_event_vol.event_panel import build_event_panel, discover_option_contracts
from earnings_event_vol.event_window_panel import build_event_window_panel
from earnings_event_vol.market_covariates import (
    MARKET_COVARIATE_SCHEMA_VERSION,
    normalize_fred_vixcls_csv,
)
from earnings_event_vol.market_index_proxy import (
    MARKET_INDEX_SECOND_SCHEMA_VERSION,
    MARKET_INDEX_SYMBOLS,
    fetch_massive_underlying_second_aggregates,
    market_index_surface_features,
    normalize_underlying_second_aggregates,
    prefix_market_index_features,
    select_market_index_option_candidates,
    select_underlying_second_features,
)
from earnings_event_vol.massive import (
    MassiveCommandResult,
    _run_head_object_command,
    build_download_file_command,
    build_massive_day_agg_sample,
    massive_flat_file_aws_env,
    massive_flat_file_manifest,
    option_flat_file_key,
    underlying_flat_file_key,
)
from earnings_event_vol.trade_proxy import (
    fetch_massive_option_second_aggregates,
    filter_pre_cutoff_buffer,
    normalize_second_aggregates,
)
from earnings_event_vol.universe import (
    ELIGIBLE_EQUITY_RULE_VERSION,
    build_eligible_equity_tickers,
    build_monthly_liquid_universe,
    build_ticker_month_liquidity,
    eligible_equity_cache_matches_rule,
)

DEFAULT_STATIC_TICKERS: tuple[str, ...] = (
    "AAPL",
    "MSFT",
    "NVDA",
    "AMZN",
    "TSLA",
    "META",
    "GOOGL",
    "AVGO",
    "AMD",
    "NFLX",
    "JPM",
    "XOM",
    "UNH",
    "BAC",
    "GS",
    "COST",
    "WMT",
    "HD",
    "LLY",
    "MRK",
    "PFE",
    "BA",
    "CAT",
)

SUPPORTED_DATA_STAGES = {
    "all",
    "proxy-all",
    "fixture-audit",
    "massive-probe",
    "options-day-aggs-bulk",
    "market-covariates",
    "market-second-covariates",
    "universe",
    "dynamic-calendar",
    "event-window-panel",
    "contracts",
    "contract-reference-validation",
    "panel",
    "trade-proxy-panel",
}


@dataclass(frozen=True)
class DataPipelineStep:
    name: str
    status: str
    outputs: tuple[Path, ...] = ()
    reason: str | None = None
    metadata: dict[str, object] = field(default_factory=dict)

    def as_dict(self) -> dict[str, object]:
        return {
            "name": self.name,
            "status": self.status,
            "outputs": [str(path) for path in self.outputs],
            "reason": self.reason,
            "metadata": self.metadata,
        }


def parse_text_list(values: Sequence[str] | str | None) -> list[str]:
    if values is None:
        return []
    raw_values = [values] if isinstance(values, str) else list(values)
    items: list[str] = []
    for value in raw_values:
        items.extend(part.strip() for part in value.replace(",", " ").split() if part.strip())
    return items


def _complete(paths: Sequence[Path]) -> bool:
    return bool(paths) and all(path.exists() and path.stat().st_size > 0 for path in paths)


def _json_params_match(path: Path, expected: Mapping[str, object]) -> bool:
    if not path.exists() or path.stat().st_size <= 0:
        return False
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return False
    return bool(payload.get("pipeline_params") == dict(expected))


def _complete_with_params(
    paths: Sequence[Path],
    *,
    params_path: Path,
    expected_params: Mapping[str, object],
) -> bool:
    return _complete(paths) and _json_params_match(params_path, expected_params)


def _bulk_day_aggs_complete_with_params(
    outputs: Sequence[Path],
    *,
    manifest_path: Path,
    expected_params: Mapping[str, object],
) -> bool:
    if not _complete(outputs) or not _json_params_match(manifest_path, expected_params):
        return False
    try:
        payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return False
    status_counts = payload.get("status_counts")
    if not isinstance(status_counts, dict) or int(status_counts.get("failed") or 0) > 0:
        return False
    dataset_counts = payload.get("dataset_counts")
    if not isinstance(dataset_counts, dict):
        return False
    options_counts = dataset_counts.get("options_day_aggs")
    if not isinstance(options_counts, dict):
        return False
    success = sum(
        int(options_counts.get(status) or 0) for status in ("hit", "downloaded", "repaired")
    )
    return success > 0


def _progress(message: str) -> None:
    print(f"[data] {message}", file=sys.stderr, flush=True)


def _run_command_with_progress(
    command: Sequence[str],
    *,
    cwd: Path,
    label: str,
) -> subprocess.CompletedProcess[str]:
    tail: deque[str] = deque(maxlen=80)
    _progress(f"{label}: command {' '.join(command)}")
    process = subprocess.Popen(
        list(command),
        cwd=cwd,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
    )
    assert process.stdout is not None
    for line in process.stdout:
        tail.append(line)
        print(f"[{label}] {line}", end="", file=sys.stderr, flush=True)
    return subprocess.CompletedProcess(
        list(command),
        process.wait(),
        stdout="".join(tail),
        stderr="",
    )


def _write_manifest(out_root: Path, steps: Sequence[DataPipelineStep]) -> None:
    out_root.mkdir(parents=True, exist_ok=True)
    payload = {
        "steps": [step.as_dict() for step in steps],
        "ok": all(step.status in {"ran", "skipped"} for step in steps),
    }
    (out_root / "data_pipeline_manifest.json").write_text(
        json.dumps(payload, indent=2, default=str),
        encoding="utf-8",
    )


def _read_universe_source(path: Path) -> pd.DataFrame:
    if path.is_dir():
        parquet_files = sorted(path.rglob("*.parquet"))
        csv_files = sorted(path.rglob("*.csv"))
        if parquet_files:
            return pl.scan_parquet([str(file) for file in parquet_files]).collect().to_pandas()
        if csv_files:
            return pd.concat((pd.read_csv(file) for file in csv_files), ignore_index=True)
        raise FileNotFoundError(f"no parquet or csv files found under {path}")
    if path.suffix == ".parquet":
        return pl.read_parquet(path).to_pandas()
    return pd.read_csv(path)


def _write_parquet(path: Path, frame: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    pl.from_pandas(frame).write_parquet(path, compression="zstd")


def _market_covariates_step(
    config: ProjectConfig,
    *,
    out_root: Path,
    force: bool,
) -> DataPipelineStep:
    raw_path = config.bronze_data_dir / "market_covariates" / "fred_vixcls.csv"
    silver_path = config.silver_data_dir / "market_covariates" / "daily_market_covariates.parquet"
    manifest_path = out_root / "market_covariates" / "market_covariates_manifest.json"
    outputs = (raw_path, silver_path, manifest_path)
    params = {
        "stage": "market-covariates",
        "source_dataset": "fred_vixcls",
        "source_url": config.fred_vixcls_url,
        "schema_version": MARKET_COVARIATE_SCHEMA_VERSION,
    }
    if not force and _complete_with_params(
        outputs,
        params_path=manifest_path,
        expected_params=params,
    ):
        return DataPipelineStep(
            "market-covariates",
            "skipped",
            outputs,
            reason="outputs_exist_params_match",
        )

    raw_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        if force or not raw_path.exists() or raw_path.stat().st_size <= 0:
            with httpx.Client(timeout=config.massive_request_timeout_seconds) as client:
                response = client.get(config.fred_vixcls_url)
                response.raise_for_status()
                raw_path.write_bytes(response.content)
        raw = pd.read_csv(raw_path)
        snapshot = date.today()
        silver = normalize_fred_vixcls_csv(
            raw,
            source_snapshot_date=snapshot,
            source_url=config.fred_vixcls_url,
        )
        _write_parquet(silver_path, silver)
    except Exception as exc:
        manifest_path.write_text(
            json.dumps(
                {
                    "pipeline_params": params,
                    "status": "blocked",
                    "reason": "market_covariates_failed",
                    "error": str(exc),
                },
                indent=2,
                default=str,
            ),
            encoding="utf-8",
        )
        return DataPipelineStep(
            "market-covariates",
            "blocked",
            outputs,
            reason="market_covariates_failed",
            metadata={"error": str(exc)},
        )

    payload = {
        "pipeline_params": params,
        "status": "ran",
        "rows": int(len(silver)),
        "vix_rows": int(pd.to_numeric(silver["vix_close"], errors="coerce").notna().sum()),
        "missing_rows": int(silver["is_holiday_or_missing"].sum()),
        "source_snapshot_date": snapshot.isoformat(),
        "outputs": {
            "bronze_fred_vixcls_csv": str(raw_path),
            "daily_market_covariates": str(silver_path),
        },
    }
    manifest_path.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")
    return DataPipelineStep(
        "market-covariates",
        "ran",
        outputs,
        metadata={
            "rows": int(len(silver)),
            "vix_rows": int(pd.to_numeric(silver["vix_close"], errors="coerce").notna().sum()),
            "missing_rows": int(silver["is_holiday_or_missing"].sum()),
        },
    )


MARKET_INDEX_OPTION_SECOND_COLUMNS = {
    "options_ticker",
    "timestamp_et",
    "option_close",
    "option_vwap",
    "volume",
    "transactions",
}
MARKET_INDEX_UNDERLYING_SECOND_COLUMNS = {
    "ticker",
    "timestamp_et",
    "underlying_close",
    "underlying_vwap",
    "volume",
    "transactions",
}


def _safe_timestamp(value: object) -> pd.Timestamp | None:  # pragma: no cover
    timestamp = pd.Timestamp(value)
    if pd.isna(timestamp):
        return None
    if timestamp.tzinfo is None:
        return None
    return timestamp.tz_convert("America/New_York")


def _cutoff_partition(cutoff_timestamp: pd.Timestamp) -> str:  # pragma: no cover
    return str(cutoff_timestamp.strftime("%H%M%S"))


def _safe_ticker_partition(value: str) -> str:  # pragma: no cover
    return value.replace(":", "_").replace("/", "_")


def _market_index_option_second_path(
    config: ProjectConfig,
    *,
    option_ticker: str,
    entry_date: date,
    cutoff_timestamp: pd.Timestamp,
    buffer_minutes: int,
) -> Path:  # pragma: no cover
    return (
        config.bronze_data_dir
        / "massive"
        / "market_index_options_second_aggs"
        / f"date={entry_date.isoformat()}"
        / f"cutoff={_cutoff_partition(cutoff_timestamp)}"
        / f"buffer_minutes={buffer_minutes}"
        / f"options_ticker={_safe_ticker_partition(option_ticker)}"
        / "part.parquet"
    )


def _market_index_underlying_second_path(
    config: ProjectConfig,
    *,
    symbol: str,
    entry_date: date,
    cutoff_timestamp: pd.Timestamp,
    buffer_minutes: int,
) -> Path:  # pragma: no cover
    return (
        config.bronze_data_dir
        / "massive"
        / "market_index_underlying_second_aggs"
        / f"date={entry_date.isoformat()}"
        / f"cutoff={_cutoff_partition(cutoff_timestamp)}"
        / f"buffer_minutes={buffer_minutes}"
        / f"index_symbol={symbol.upper()}"
        / "part.parquet"
    )


def _read_cached_parquet(  # pragma: no cover
    path: Path, required_columns: set[str]
) -> pd.DataFrame | None:
    if not _parquet_has_columns(path, required_columns):
        return None
    try:
        return pd.read_parquet(path)
    except Exception:
        return None


def _write_pandas_parquet(path: Path, frame: pd.DataFrame) -> None:  # pragma: no cover
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_parquet(path, index=False)


def _option_index_symbol(option_ticker: str) -> str:  # pragma: no cover
    parsed = option_ticker.removeprefix("O:")
    for symbol in MARKET_INDEX_SYMBOLS:
        if parsed.startswith(symbol):
            return symbol
    return "UNKNOWN"


def _read_market_index_spots(  # pragma: no cover
    config: ProjectConfig, entry_date: date
) -> dict[str, float]:
    path = _bronze_day_agg_path(config, dataset="underlying_day_aggs", date_value=entry_date)
    if not path.exists():
        return {}
    try:
        frame = (
            pl.scan_parquet(str(path), cast_options=pl.ScanCastOptions(integer_cast="allow-float"))
            .filter(pl.col("ticker").is_in(list(MARKET_INDEX_SYMBOLS)))
            .select(["ticker", "close"])
            .collect()
            .to_pandas()
        )
    except Exception:
        return {}
    if frame.empty:
        return {}
    frame["ticker"] = frame["ticker"].astype(str).str.upper()
    frame["close"] = pd.to_numeric(frame["close"], errors="coerce")
    return {
        str(row["ticker"]): float(row["close"])
        for row in frame.dropna(subset=["close"]).to_dict("records")
    }


def _read_market_index_options_day_aggs(
    config: ProjectConfig,
    entry_date: date,
) -> pd.DataFrame:  # pragma: no cover
    path = _bronze_day_agg_path(config, dataset="options_day_aggs", date_value=entry_date)
    if not path.exists():
        return pd.DataFrame()
    pattern = r"^O:(" + "|".join(MARKET_INDEX_SYMBOLS) + r")\d{6}[CP]\d{8}$"
    try:
        schema = pl.scan_parquet(str(path)).collect_schema()
        columns = [
            column
            for column in ("ticker", "close", "volume", "transactions")
            if column in schema.names()
        ]
        frame = (
            pl.scan_parquet(str(path), cast_options=pl.ScanCastOptions(integer_cast="allow-float"))
            .filter(pl.col("ticker").str.contains(pattern))
            .select(columns)
            .collect()
            .to_pandas()
        )
    except Exception:
        return pd.DataFrame()
    if "transactions" not in frame.columns:
        frame["transactions"] = 0
    return frame


def _fetch_or_load_market_underlying_second_bars(
    config: ProjectConfig,
    *,
    symbol: str,
    entry_date: date,
    cutoff_timestamp: pd.Timestamp,
    buffer_minutes: int,
    refresh_bronze: bool,
) -> tuple[pd.DataFrame, dict[str, object]]:  # pragma: no cover
    path = _market_index_underlying_second_path(
        config,
        symbol=symbol,
        entry_date=entry_date,
        cutoff_timestamp=cutoff_timestamp,
        buffer_minutes=buffer_minutes,
    )
    cached = (
        None
        if refresh_bronze
        else _read_cached_parquet(path, MARKET_INDEX_UNDERLYING_SECOND_COLUMNS)
    )
    if cached is not None:
        return cached, {
            "ticker": symbol,
            "status": "ok",
            "cache_status": "hit",
            "rows": int(len(cached)),
            "bronze_path": str(path),
        }
    if path.exists():
        path.unlink(missing_ok=True)
    try:
        raw = fetch_massive_underlying_second_aggregates(
            config,
            ticker=symbol,
            trade_date=entry_date,
        )
        normalized = normalize_underlying_second_aggregates(raw, ticker=symbol)
        buffered = filter_pre_cutoff_buffer(
            normalized,
            cutoff_timestamp=cutoff_timestamp.to_pydatetime(),
            buffer_minutes=buffer_minutes,
        )
        _write_pandas_parquet(path, buffered)
        return buffered, {
            "ticker": symbol,
            "status": "ok",
            "cache_status": "written",
            "rows": int(len(buffered)),
            "raw_rows": int(len(normalized)),
            "bronze_path": str(path),
        }
    except Exception as exc:
        return pd.DataFrame(), {
            "ticker": symbol,
            "status": "fetch_failed",
            "cache_status": "miss",
            "rows": 0,
            "bronze_path": str(path),
            "error": str(exc)[:300],
        }


def _fetch_or_load_market_option_second_bars(
    config: ProjectConfig,
    *,
    option_ticker: str,
    entry_date: date,
    cutoff_timestamp: pd.Timestamp,
    buffer_minutes: int,
    refresh_bronze: bool,
) -> tuple[str, pd.DataFrame, dict[str, object]]:  # pragma: no cover
    path = _market_index_option_second_path(
        config,
        option_ticker=option_ticker,
        entry_date=entry_date,
        cutoff_timestamp=cutoff_timestamp,
        buffer_minutes=buffer_minutes,
    )
    cached = (
        None if refresh_bronze else _read_cached_parquet(path, MARKET_INDEX_OPTION_SECOND_COLUMNS)
    )
    if cached is not None:
        return (
            option_ticker,
            cached,
            {
                "options_ticker": option_ticker,
                "index_symbol": _option_index_symbol(option_ticker),
                "status": "ok",
                "cache_status": "hit",
                "rows": int(len(cached)),
                "bronze_path": str(path),
            },
        )
    if path.exists():
        path.unlink(missing_ok=True)
    try:
        raw = fetch_massive_option_second_aggregates(
            config,
            option_ticker=option_ticker,
            trade_date=entry_date,
        )
        normalized = normalize_second_aggregates(raw, option_ticker=option_ticker)
        buffered = filter_pre_cutoff_buffer(
            normalized,
            cutoff_timestamp=cutoff_timestamp.to_pydatetime(),
            buffer_minutes=buffer_minutes,
        )
        _write_pandas_parquet(path, buffered)
        return (
            option_ticker,
            buffered,
            {
                "options_ticker": option_ticker,
                "index_symbol": _option_index_symbol(option_ticker),
                "status": "ok",
                "cache_status": "written",
                "rows": int(len(buffered)),
                "raw_rows": int(len(normalized)),
                "bronze_path": str(path),
            },
        )
    except Exception as exc:
        return (
            option_ticker,
            pd.DataFrame(),
            {
                "options_ticker": option_ticker,
                "index_symbol": _option_index_symbol(option_ticker),
                "status": "fetch_failed",
                "cache_status": "miss",
                "rows": 0,
                "bronze_path": str(path),
                "error": str(exc)[:300],
            },
        )


def _market_second_event_row(
    config: ProjectConfig,
    *,
    event: Mapping[str, object],
    options_day_aggs: pd.DataFrame,
    spots: Mapping[str, float],
    jobs: int,
    lookback_seconds: int,
    buffer_minutes: int,
    price_field: str,
    refresh_bronze: bool,
) -> tuple[dict[str, object], list[dict[str, object]]]:  # pragma: no cover
    event_id = str(event.get("event_id") or "")
    entry_date = pd.Timestamp(event.get("entry_date")).date()
    cutoff_timestamp = _safe_timestamp(event.get("event_entry_timestamp"))
    base: dict[str, object] = {
        "event_id": event_id,
        "entry_date": entry_date,
        "event_entry_timestamp": event.get("event_entry_timestamp"),
        "market_second_route": "massive_rest_second_aggs",
        "market_second_panel_grade": "no_nbbo_trade_proxy",
        "market_second_schema_version": MARKET_INDEX_SECOND_SCHEMA_VERSION,
    }
    reports: list[dict[str, object]] = []
    if cutoff_timestamp is None:
        base["market_second_status"] = "invalid_event_entry_timestamp"
        return base, reports
    if options_day_aggs.empty:
        base["market_second_status"] = "missing_options_day_aggs"
        return base, reports
    base["market_second_status"] = "ok"
    for symbol in MARKET_INDEX_SYMBOLS:
        spot = spots.get(symbol)
        if spot is None or not pd.notna(spot):
            base[f"{symbol.lower()}_second_underlying_status"] = "missing_underlying_day_aggs"
            continue
        underlying_bars, underlying_report = _fetch_or_load_market_underlying_second_bars(
            config,
            symbol=symbol,
            entry_date=entry_date,
            cutoff_timestamp=cutoff_timestamp,
            buffer_minutes=buffer_minutes,
            refresh_bronze=refresh_bronze,
        )
        underlying_report.update({"event_id": event_id, "dataset": "underlying_second_aggs"})
        reports.append(underlying_report)
        underlying_features = select_underlying_second_features(
            underlying_bars,
            cutoff_timestamp=cutoff_timestamp.to_pydatetime(),
            buffer_minutes=buffer_minutes,
            lookback_seconds=lookback_seconds,
        )
        prefix = symbol.lower()
        base.update(
            {
                f"{prefix}_second_underlying_status": underlying_features.status,
                f"{prefix}_second_underlying_close": underlying_features.close,
                f"{prefix}_second_underlying_vwap": underlying_features.vwap,
                f"{prefix}_second_underlying_return_in_buffer": (
                    underlying_features.return_in_buffer
                ),
                f"{prefix}_second_underlying_volume_sum": underlying_features.volume_sum,
                f"{prefix}_second_underlying_transactions_sum": (
                    underlying_features.transactions_sum
                ),
                f"{prefix}_second_underlying_rows": underlying_features.rows_in_buffer,
            }
        )
        candidates = select_market_index_option_candidates(
            options_day_aggs,
            symbol=symbol,
            source_date=entry_date,
            spot=float(spot),
        )
        option_frames: dict[str, pd.DataFrame] = {}
        requests = (
            sorted(candidates["options_ticker"].astype(str).unique().tolist())
            if not candidates.empty
            else []
        )
        if requests:
            if jobs <= 1:
                results = [
                    _fetch_or_load_market_option_second_bars(
                        config,
                        option_ticker=option_ticker,
                        entry_date=entry_date,
                        cutoff_timestamp=cutoff_timestamp,
                        buffer_minutes=buffer_minutes,
                        refresh_bronze=refresh_bronze,
                    )
                    for option_ticker in requests
                ]
            else:
                with ThreadPoolExecutor(max_workers=max(1, jobs)) as executor:
                    futures = [
                        executor.submit(
                            _fetch_or_load_market_option_second_bars,
                            config,
                            option_ticker=option_ticker,
                            entry_date=entry_date,
                            cutoff_timestamp=cutoff_timestamp,
                            buffer_minutes=buffer_minutes,
                            refresh_bronze=refresh_bronze,
                        )
                        for option_ticker in requests
                    ]
                    results = [future.result() for future in as_completed(futures)]
            for option_ticker, frame, report in results:
                option_frames[option_ticker] = frame
                report.update({"event_id": event_id, "dataset": "options_second_aggs"})
                reports.append(report)
        surface = market_index_surface_features(
            candidates,
            option_frames,
            symbol=symbol,
            spot=float(spot),
            cutoff_timestamp=cutoff_timestamp.to_pydatetime(),
            lookback_seconds=lookback_seconds,
            price_field=price_field,
        )
        base.update(prefix_market_index_features(surface, symbol=symbol))
    return base, reports


def _market_second_covariates_step(
    config: ProjectConfig,
    *,
    out_root: Path,
    force: bool,
    refresh_bronze: bool,
    jobs: int,
    max_events: int | None,
    lookback_seconds: int,
    second_agg_buffer_minutes: int,
    price_field: str,
) -> DataPipelineStep:  # pragma: no cover
    panel_path = config.gold_data_dir / "event_panel" / "trade_proxy_event_panel.parquet"
    out = out_root / "market_second_covariates"
    silver_path = config.silver_data_dir / "market_covariates" / "market_second_covariates.parquet"
    report_path = out / "market_second_covariates_fetch_report.csv"
    manifest_path = out / "market_second_covariates_manifest.json"
    outputs = (silver_path, report_path, manifest_path)
    params = {
        "stage": "market-second-covariates",
        "input_panel": _path_signature(panel_path),
        "symbols": list(MARKET_INDEX_SYMBOLS),
        "max_events": max_events,
        "lookback_seconds": lookback_seconds,
        "second_agg_buffer_minutes": second_agg_buffer_minutes,
        "price_field": price_field,
        "schema_version": MARKET_INDEX_SECOND_SCHEMA_VERSION,
        "refresh_bronze": refresh_bronze,
    }
    if not panel_path.exists():
        return DataPipelineStep(
            "market-second-covariates",
            "blocked",
            outputs,
            reason=f"requires configured event panel at {panel_path}",
        )
    if (
        not force
        and not refresh_bronze
        and _complete_with_params(
            outputs,
            params_path=manifest_path,
            expected_params=params,
        )
    ):
        return DataPipelineStep(
            "market-second-covariates",
            "skipped",
            outputs,
            reason="outputs_exist_params_match",
        )
    panel = pd.read_parquet(panel_path)
    required = {"event_id", "entry_date", "event_entry_timestamp"}
    missing = sorted(required - set(panel.columns))
    if missing:
        return DataPipelineStep(
            "market-second-covariates",
            "blocked",
            outputs,
            reason=f"panel_missing_columns:{','.join(missing)}",
        )
    events = (
        panel[list(required)]
        .dropna(subset=["event_id", "entry_date"])
        .drop_duplicates("event_id")
        .sort_values(["entry_date", "event_id"])
        .reset_index(drop=True)
    )
    if max_events is not None:
        events = events.head(max_events).copy()
    _progress(
        "market-second-covariates: "
        f"events={len(events)} symbols={','.join(MARKET_INDEX_SYMBOLS)} jobs={jobs}"
    )
    rows: list[dict[str, object]] = []
    reports: list[dict[str, object]] = []
    for index, event in enumerate(events.to_dict("records"), start=1):
        entry_date = pd.Timestamp(event["entry_date"]).date()
        options_day = _read_market_index_options_day_aggs(config, entry_date)
        spots = _read_market_index_spots(config, entry_date)
        row, event_reports = _market_second_event_row(
            config,
            event=event,
            options_day_aggs=options_day,
            spots=spots,
            jobs=jobs,
            lookback_seconds=lookback_seconds,
            buffer_minutes=second_agg_buffer_minutes,
            price_field=price_field,
            refresh_bronze=refresh_bronze,
        )
        rows.append(row)
        reports.extend(event_reports)
        if index == len(events) or index % max(1, len(events) // 10) == 0:
            progress_counts = Counter(str(item.get("status")) for item in reports)
            _progress(
                "market-second-covariates progress: "
                f"{index}/{len(events)} events reports={dict(progress_counts)}"
            )
    frame = pd.DataFrame(rows)
    report = pd.DataFrame(reports)
    _write_pandas_parquet(silver_path, frame)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report.to_csv(report_path, index=False)
    status_counts = _frame_value_counts(frame, "market_second_status")
    fetch_counts = _frame_value_counts(report, "status")
    manifest = {
        "pipeline_params": params,
        "status": "ran",
        "rows": int(len(frame)),
        "fetch_rows": int(len(report)),
        "status_counts": status_counts,
        "fetch_status_counts": fetch_counts,
        "outputs": {
            "market_second_covariates": str(silver_path),
            "fetch_report": str(report_path),
        },
    }
    manifest_path.write_text(json.dumps(manifest, indent=2, default=str), encoding="utf-8")
    return DataPipelineStep(
        "market-second-covariates",
        "ran",
        outputs,
        metadata={
            "rows": int(len(frame)),
            "fetch_rows": int(len(report)),
            "status_counts": status_counts,
            "fetch_status_counts": fetch_counts,
        },
    )


def _sec_company_ticker_rows(payload: object) -> list[dict[str, object]]:
    if isinstance(payload, list):
        return [row for row in payload if isinstance(row, dict)]
    if not isinstance(payload, dict):
        return []

    fields = payload.get("fields")
    data = payload.get("data")
    if isinstance(fields, list) and isinstance(data, list):
        rows: list[dict[str, object]] = []
        field_names = [str(field) for field in fields]
        for values in data:
            if not isinstance(values, list):
                continue
            rows.append(
                {
                    field: values[index] if index < len(values) else None
                    for index, field in enumerate(field_names)
                }
            )
        return rows

    return [value for value in payload.values() if isinstance(value, dict)]


def _read_valid_eligible_equity_cache(path: Path) -> pd.DataFrame | None:
    if not path.exists() or path.stat().st_size <= 0:
        return None
    try:
        frame = pl.read_parquet(path).to_pandas()
    except Exception:
        return None
    if not eligible_equity_cache_matches_rule(frame):
        return None
    return frame


def _load_or_build_eligible_equity_cache(
    config: ProjectConfig,
    *,
    path: Path,
    source_snapshot_date: date,
) -> tuple[pd.DataFrame, str]:
    cached = _read_valid_eligible_equity_cache(path)
    if cached is not None:
        return cached, "hit"

    with httpx.Client(timeout=config.massive_request_timeout_seconds) as client:
        response = client.get(
            config.sec_company_tickers_url,
            headers={"User-Agent": config.sec_user_agent},
        )
        response.raise_for_status()
        rows = _sec_company_ticker_rows(response.json())
    frame = build_eligible_equity_tickers(
        rows,
        source_snapshot_date=source_snapshot_date,
    )
    if frame.empty:
        raise ValueError("SEC company ticker metadata produced no eligible-equity rows")
    _write_parquet(path, frame)
    return frame, "written"


def _eligible_ticker_set(frame: pd.DataFrame) -> set[str]:
    if frame.empty or "ticker" not in frame.columns or "eligible" not in frame.columns:
        return set()
    eligible = frame["eligible"]
    if pd.api.types.is_bool_dtype(eligible):
        mask = eligible.astype(bool)
    elif pd.api.types.is_numeric_dtype(eligible):
        mask = pd.to_numeric(eligible, errors="coerce").fillna(0).ne(0)
    else:
        mask = eligible.astype(str).str.lower().isin({"1", "true", "yes"})
    return {
        str(ticker).upper()
        for ticker in frame.loc[mask, "ticker"].dropna().tolist()
        if str(ticker).strip()
    }


def _filter_reason_counts(frame: pd.DataFrame) -> dict[str, int]:
    if frame.empty or "filter_reason" not in frame.columns:
        return {}
    counts = frame["filter_reason"].astype(str).value_counts().to_dict()
    return {str(key): int(value) for key, value in counts.items()}


def _month_start(value: date) -> date:
    return date(value.year, value.month, 1)


def _add_months(value: date, months: int) -> date:
    month_index = value.year * 12 + value.month - 1 + months
    return date(month_index // 12, month_index % 12 + 1, 1)


def _universe_lookback_start(start_date: date, trailing_months: int) -> date:
    return _add_months(_month_start(start_date), -trailing_months)


def _weekday_dates(start_date: date, end_date: date) -> list[date]:
    days: list[date] = []
    current = start_date
    while current <= end_date:
        if current.weekday() < 5:
            days.append(current)
        current += timedelta(days=1)
    return days


def _date_partition_value(path: Path) -> date | None:
    for part in path.parts:
        if part.startswith("date="):
            try:
                return date.fromisoformat(part.split("=", 1)[1])
            except ValueError:
                return None
    return None


def _bronze_day_agg_path(config: ProjectConfig, *, dataset: str, date_value: date) -> Path:
    return (
        config.bronze_data_dir
        / "massive"
        / dataset
        / f"date={date_value.isoformat()}"
        / "part.parquet"
    )


def _day_agg_key(config: ProjectConfig, *, dataset: str, date_value: date) -> str:
    date_text = date_value.isoformat()
    if dataset == "options_day_aggs":
        return option_flat_file_key(
            config,
            year=date_value.year,
            month=date_value.month,
            date=date_text,
        )
    if dataset == "underlying_day_aggs":
        return underlying_flat_file_key(config, date=date_text)
    raise ValueError(f"unsupported bulk day-agg dataset: {dataset}")


def _parquet_has_columns(path: Path, required_columns: set[str]) -> bool:
    if not path.exists() or path.stat().st_size <= 0:
        return False
    try:
        schema = pl.scan_parquet(str(path)).collect_schema()
    except Exception:
        return False
    return required_columns.issubset(set(schema.names()))


def _bulk_required_columns(dataset: str) -> set[str]:
    if dataset in {"options_day_aggs", "underlying_day_aggs"}:
        return {"ticker", "close", "volume"}
    raise ValueError(f"unsupported bulk day-agg dataset: {dataset}")


def _download_error_status(result: MassiveCommandResult) -> str:
    if result.returncode in {124, 127}:
        return "failed"
    text = f"{result.stderr}\n{result.stdout}".lower()
    missing_markers = ("nosuchkey", "not found", "404", "does not exist", "not exist")
    return "missing_flat_file" if any(marker in text for marker in missing_markers) else "failed"


def _download_error_text(result: MassiveCommandResult) -> str:
    text = (result.stderr or result.stdout or "").strip()
    if not text:
        return f"aws command failed with exit code {result.returncode}"
    return text.splitlines()[-1][:500]


def _normalize_bulk_day_agg_frame(
    frame: pl.DataFrame,
    *,
    dataset: str,
    date_value: date,
    source_key: str,
) -> pl.DataFrame:
    if "ticker" not in frame.columns:
        raise ValueError("flat file missing ticker column")
    if "volume" not in frame.columns:
        raise ValueError("flat file missing volume column")
    if "close" not in frame.columns:
        raise ValueError("flat file missing close column")
    out = frame.with_columns(
        [
            pl.col("ticker").cast(pl.Utf8),
            pl.col("volume").cast(pl.Float64, strict=False),
            pl.col("close").cast(pl.Float64, strict=False),
            pl.lit(date_value.isoformat()).alias("source_date"),
            pl.lit(dataset).alias("source_dataset"),
            pl.lit(source_key).alias("source_key"),
        ]
    )
    if "vwap" in out.columns:
        out = out.with_columns(pl.col("vwap").cast(pl.Float64, strict=False))
    return out


def _ensure_bulk_day_agg_partition(
    config: ProjectConfig,
    *,
    dataset: str,
    date_value: date,
    refresh_bronze: bool,
    runner: Callable[
        [Sequence[str], Mapping[str, str], float], MassiveCommandResult
    ] = _run_head_object_command,
) -> dict[str, object]:
    destination = _bronze_day_agg_path(config, dataset=dataset, date_value=date_value)
    required_columns = _bulk_required_columns(dataset)
    if not refresh_bronze and _parquet_has_columns(destination, required_columns):
        return {
            "date": date_value.isoformat(),
            "dataset": dataset,
            "status": "hit",
            "path": str(destination),
        }

    had_existing = destination.exists()
    destination.parent.mkdir(parents=True, exist_ok=True)
    if had_existing:
        destination.unlink()

    key = _day_agg_key(config, dataset=dataset, date_value=date_value)
    csv_path = destination.parent / "download.csv.gz"
    if csv_path.exists():
        csv_path.unlink()
    command = build_download_file_command(config, key=key, destination=csv_path)
    try:
        env = massive_flat_file_aws_env(config)
    except Exception as exc:
        return {
            "date": date_value.isoformat(),
            "dataset": dataset,
            "status": "failed",
            "path": str(destination),
            "key": key,
            "error": str(exc),
        }
    result = runner(command, env, config.massive_request_timeout_seconds)
    if result.returncode != 0:
        if csv_path.exists():
            csv_path.unlink()
        return {
            "date": date_value.isoformat(),
            "dataset": dataset,
            "status": _download_error_status(result),
            "path": str(destination),
            "key": key,
            "error": _download_error_text(result),
        }

    tmp_path = destination.with_suffix(".parquet.tmp")
    if tmp_path.exists():
        tmp_path.unlink()
    try:
        raw = pl.read_csv(csv_path, infer_schema_length=10000)
        normalized = _normalize_bulk_day_agg_frame(
            raw,
            dataset=dataset,
            date_value=date_value,
            source_key=key,
        )
        normalized.write_parquet(tmp_path, compression="zstd")
        tmp_path.replace(destination)
    except Exception as exc:
        if tmp_path.exists():
            tmp_path.unlink()
        return {
            "date": date_value.isoformat(),
            "dataset": dataset,
            "status": "failed",
            "path": str(destination),
            "key": key,
            "error": str(exc),
        }
    finally:
        if csv_path.exists():
            csv_path.unlink()

    return {
        "date": date_value.isoformat(),
        "dataset": dataset,
        "status": "repaired" if had_existing and not refresh_bronze else "downloaded",
        "path": str(destination),
        "key": key,
        "rows": int(normalized.height),
    }


def _option_day_agg_monthly_liquidity_from_parquet_dir(
    path: Path,
    *,
    start_date: date,
    end_date: date,
    trailing_months: int,
    source_snapshot_date: date,
) -> pd.DataFrame:
    files = sorted(path.rglob("*.parquet"))
    if not files:
        raise FileNotFoundError(f"no parquet files found under {path}")
    liquidity_start = _universe_lookback_start(start_date, trailing_months)
    option_ticker_pattern = r"^O:([A-Z0-9.]+?)\d{6}[CP]\d{8}$"
    month_frames: list[pl.DataFrame] = []
    for file in files:
        date_value = _date_partition_value(file)
        if date_value is None or date_value < liquidity_start or date_value > end_date:
            continue
        try:
            schema = pl.scan_parquet(str(file)).collect_schema()
        except Exception:
            continue
        columns = set(schema.names())
        if not {"ticker", "volume"}.issubset(columns):
            continue
        price_columns = [
            column
            for column in ("option_vwap", "vwap", "option_close", "close")
            if column in columns
        ]
        if not price_columns:
            continue
        price_exprs = [
            pl.when(pl.col(column).cast(pl.Float64, strict=False) > 0)
            .then(pl.col(column).cast(pl.Float64, strict=False))
            .otherwise(None)
            for column in price_columns
        ]
        month_value = date(date_value.year, date_value.month, 1)
        try:
            daily = (
                pl.scan_parquet(str(file))
                .select(
                    [
                        pl.coalesce(
                            [
                                pl.col("ticker")
                                .cast(pl.Utf8)
                                .str.to_uppercase()
                                .str.extract(option_ticker_pattern, 1),
                                pl.col("ticker").cast(pl.Utf8).str.to_uppercase(),
                            ]
                        ).alias("ticker"),
                        pl.col("volume").cast(pl.Float64, strict=False).alias("volume"),
                        pl.coalesce(price_exprs).alias("premium_price"),
                    ]
                )
                .filter(
                    pl.col("ticker").is_not_null()
                    & (pl.col("ticker") != "")
                    & (pl.col("volume") > 0)
                    & (pl.col("premium_price") > 0)
                )
                .with_columns(
                    [
                        pl.lit(month_value).alias("month"),
                        (pl.col("premium_price") * pl.col("volume") * 100.0).alias(
                            "option_premium_dollar_volume"
                        ),
                    ]
                )
                .group_by(["month", "ticker"])
                .agg(
                    [
                        pl.col("option_premium_dollar_volume").sum(),
                        pl.col("volume").sum().alias("option_contract_volume"),
                        pl.len().alias("option_day_rows"),
                    ]
                )
                .collect()
            )
        except Exception:
            continue
        if daily.height > 0:
            month_frames.append(daily)

    if not month_frames:
        return pd.DataFrame(
            columns=[
                "month",
                "ticker",
                "option_premium_dollar_volume",
                "option_contract_volume",
                "option_day_rows",
                "source_snapshot_date",
                "rule_version",
                "source_dataset",
            ]
        )
    grouped = (
        pl.concat(month_frames)
        .group_by(["month", "ticker"])
        .agg(
            [
                pl.col("option_premium_dollar_volume").sum(),
                pl.col("option_contract_volume").sum(),
                pl.col("option_day_rows").sum(),
            ]
        )
        .sort(["month", "ticker"])
        .with_columns(
            [
                pl.lit(source_snapshot_date.isoformat()).alias("source_snapshot_date"),
                pl.lit("v1.0").alias("rule_version"),
                pl.lit("massive_options_day_aggs").alias("source_dataset"),
            ]
        )
    )
    return grouped.to_pandas()


def _fixture_audit_step(config: ProjectConfig, *, out_root: Path, force: bool) -> DataPipelineStep:
    out = out_root / "fixture_audit"
    outputs = (
        out / "required_fields_report.json",
        out / "field_coverage.csv",
        out / "vendor_local_iv_diff.csv",
        out / "quote_source_report.csv",
    )
    if not force and _complete(outputs):
        return DataPipelineStep("fixture-audit", "skipped", outputs, reason="outputs_exist")

    fixtures = config.repo_root / "tests" / "fixtures"
    result = audit_data_fields(
        options=pd.read_csv(fixtures / "option_quotes.csv"),
        underlying=pd.read_csv(fixtures / "underlying_bars.csv"),
        earnings=pd.read_csv(fixtures / "earnings_calendar.csv"),
        source_paths=[
            fixtures / "option_quotes.csv",
            fixtures / "underlying_bars.csv",
            fixtures / "earnings_calendar.csv",
        ],
    )
    out.mkdir(parents=True, exist_ok=True)
    outputs[0].write_text(
        json.dumps(result.required_fields_report, indent=2, default=str),
        encoding="utf-8",
    )
    result.field_coverage.to_csv(outputs[1], index=False)
    result.vendor_local_iv_diff.to_csv(outputs[2], index=False)
    result.quote_source_report.to_csv(outputs[3], index=False)
    return DataPipelineStep(
        "fixture-audit",
        "ran",
        outputs,
        metadata={"ok": bool(result.required_fields_report["ok"])},
    )


def _massive_probe_one_date(
    config: ProjectConfig,
    *,
    out_root: Path,
    probe_date: date,
    force: bool,
    download_samples: bool,
) -> DataPipelineStep:
    out = out_root / "massive_probe" / probe_date.isoformat()
    outputs = (
        (out / "massive_sample_schema_report.json")
        if download_samples
        else (out / "massive_flat_file_manifest.json"),
    )
    if not force and _complete(outputs):
        return DataPipelineStep(
            f"massive-probe:{probe_date.isoformat()}",
            "skipped",
            outputs,
            reason="outputs_exist",
        )

    out.mkdir(parents=True, exist_ok=True)
    metadata: dict[str, object]
    if download_samples:
        report = build_massive_day_agg_sample(config, date_value=probe_date, out_dir=out)
        metadata = {"sample_rows": int(report["sample_rows"])}
    else:
        manifest = massive_flat_file_manifest(config, date_value=probe_date, run_head=True)
        (out / "massive_flat_file_manifest.json").write_text(
            json.dumps(manifest, indent=2, default=str),
            encoding="utf-8",
        )
        pd.DataFrame(manifest["objects"]).to_csv(out / "massive_flat_file_objects.csv", index=False)
        metadata = {"objects": len(manifest["objects"])}
    return DataPipelineStep(
        f"massive-probe:{probe_date.isoformat()}",
        "ran",
        outputs,
        metadata=metadata,
    )


def _universe_step(
    config: ProjectConfig,
    *,
    out_root: Path,
    options_day_aggs_path: Path | None,
    start_date: date,
    end_date: date,
    top_n: int,
    trailing_months: int,
    force: bool,
) -> DataPipelineStep:
    out = out_root / "universe"
    eligible_cache_path = out / "eligible_equity_tickers.parquet"
    liquidity_path = out / "ticker_month_liquidity.parquet"
    universe_path = out / "monthly_top50_universe.parquet"
    manifest_path = out / "universe_manifest.json"
    outputs = (
        eligible_cache_path,
        liquidity_path,
        universe_path,
        manifest_path,
    )
    source_path = options_day_aggs_path or (config.bronze_data_dir / "massive" / "options_day_aggs")
    params = {
        "stage": "universe",
        "options_day_aggs_path": str(source_path),
        "start_date": start_date.isoformat(),
        "end_date": end_date.isoformat(),
        "top_n": top_n,
        "trailing_months": trailing_months,
        "eligible_equity_rule_version": ELIGIBLE_EQUITY_RULE_VERSION,
        "eligible_equity_source_url": config.sec_company_tickers_url,
    }
    if options_day_aggs_path is None:
        options_day_aggs_path = source_path
    if not options_day_aggs_path.exists():
        return DataPipelineStep(
            "universe",
            "blocked",
            outputs,
            reason=(
                "requires options day aggregates from options-day-aggs-bulk or --options-day-aggs"
            ),
        )
    cached_eligible = _read_valid_eligible_equity_cache(eligible_cache_path)
    if (
        not force
        and _complete_with_params(
            outputs,
            params_path=manifest_path,
            expected_params=params,
        )
        and cached_eligible is not None
    ):
        return DataPipelineStep(
            "universe",
            "skipped",
            outputs,
            reason="outputs_exist_params_match",
        )

    try:
        if cached_eligible is None:
            eligible_frame, eligible_cache_status = _load_or_build_eligible_equity_cache(
                config,
                path=eligible_cache_path,
                source_snapshot_date=date.today(),
            )
        else:
            eligible_frame = cached_eligible
            eligible_cache_status = "hit"
    except Exception as exc:
        return DataPipelineStep(
            "universe",
            "blocked",
            outputs,
            reason=f"eligible_equity_cache_unavailable: {exc}",
        )
    eligible_tickers = _eligible_ticker_set(eligible_frame)
    if not eligible_tickers:
        return DataPipelineStep(
            "universe",
            "blocked",
            outputs,
            reason="eligible_equity_cache_has_no_eligible_tickers",
            metadata={"eligible_equity_cache_status": eligible_cache_status},
        )

    if options_day_aggs_path.is_dir() and list(options_day_aggs_path.rglob("*.parquet")):
        liquidity = _option_day_agg_monthly_liquidity_from_parquet_dir(
            options_day_aggs_path,
            start_date=start_date,
            end_date=end_date,
            trailing_months=trailing_months,
            source_snapshot_date=end_date,
        )
    else:
        option_day_aggs = _read_universe_source(options_day_aggs_path)
        liquidity = build_ticker_month_liquidity(
            option_day_aggs,
            source_snapshot_date=end_date,
        )
    if liquidity.empty:
        return DataPipelineStep(
            "universe",
            "blocked",
            outputs,
            reason="no_liquidity_rows_from_options_day_aggs",
        )
    universe = build_monthly_liquid_universe(
        liquidity,
        start_month=start_date,
        end_month=end_date,
        top_n=top_n,
        trailing_months=trailing_months,
        eligible_tickers=sorted(eligible_tickers),
    )
    if universe.empty:
        return DataPipelineStep(
            "universe",
            "blocked",
            outputs,
            reason="no_eligible_liquidity_rows_after_single_name_filter",
            metadata={
                "ticker_month_rows": int(len(liquidity)),
                "eligible_equity_tickers": int(len(eligible_tickers)),
                "eligible_equity_cache_status": eligible_cache_status,
            },
        )
    _write_parquet(liquidity_path, liquidity)
    _write_parquet(universe_path, universe)
    liquidity_tickers = {str(ticker).upper() for ticker in liquidity["ticker"].dropna().unique()}
    excluded_liquidity_tickers = sorted(liquidity_tickers - eligible_tickers)
    manifest = {
        "pipeline_params": params,
        "eligible_equity_cache": _path_signature(eligible_cache_path),
        "eligible_equity_cache_status": eligible_cache_status,
        "eligible_equity_rule_version": ELIGIBLE_EQUITY_RULE_VERSION,
        "eligible_equity_rows": int(len(eligible_frame)),
        "eligible_equity_tickers": int(len(eligible_tickers)),
        "eligible_equity_filter_reason_counts": _filter_reason_counts(eligible_frame),
        "ticker_month_rows": int(len(liquidity)),
        "liquidity_tickers": int(len(liquidity_tickers)),
        "eligible_liquidity_tickers": int(len(liquidity_tickers & eligible_tickers)),
        "excluded_liquidity_tickers": int(len(excluded_liquidity_tickers)),
        "excluded_liquidity_ticker_examples": excluded_liquidity_tickers[:50],
        "universe_rows": int(len(universe)),
        "universe_months": int(universe["universe_month"].nunique())
        if "universe_month" in universe
        else 0,
    }
    manifest_path.write_text(json.dumps(manifest, indent=2, default=str), encoding="utf-8")
    return DataPipelineStep(
        "universe",
        "ran",
        outputs,
        metadata={
            "ticker_month_rows": int(len(liquidity)),
            "universe_rows": int(len(universe)),
            "top_n": top_n,
            "trailing_months": trailing_months,
            "eligible_equity_tickers": int(len(eligible_tickers)),
            "eligible_equity_cache_status": eligible_cache_status,
            "excluded_liquidity_tickers": int(len(excluded_liquidity_tickers)),
        },
    )


def _massive_probe_steps(
    config: ProjectConfig,
    *,
    out_root: Path,
    dates: Sequence[date],
    force: bool,
    jobs: int,
    download_samples: bool,
) -> list[DataPipelineStep]:
    if not dates:
        return [
            DataPipelineStep(
                "massive-probe",
                "blocked",
                reason="requires at least one date via --dates",
            )
        ]
    max_workers = max(1, jobs)
    if max_workers == 1 or len(dates) == 1:
        return [
            _massive_probe_one_date(
                config,
                out_root=out_root,
                probe_date=probe_date,
                force=force,
                download_samples=download_samples,
            )
            for probe_date in dates
        ]

    steps: list[DataPipelineStep] = []
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = [
            executor.submit(
                _massive_probe_one_date,
                config,
                out_root=out_root,
                probe_date=probe_date,
                force=force,
                download_samples=download_samples,
            )
            for probe_date in dates
        ]
        for future in as_completed(futures):
            steps.append(future.result())
    return sorted(steps, key=lambda step: step.name)


def _options_day_aggs_bulk_step(
    config: ProjectConfig,
    *,
    out_root: Path,
    start_date: date,
    end_date: date,
    trailing_months: int,
    force: bool,
    refresh_bronze: bool,
    jobs: int,
) -> DataPipelineStep:
    out = out_root / "options_day_aggs_bulk"
    outputs = (out / "day_agg_fetch_report.csv", out / "options_day_aggs_bulk_manifest.json")
    lookback_start = _universe_lookback_start(start_date, trailing_months)
    datasets = ("options_day_aggs", "underlying_day_aggs")
    params = {
        "stage": "options-day-aggs-bulk",
        "start_date": start_date.isoformat(),
        "end_date": end_date.isoformat(),
        "universe_lookback_start": lookback_start.isoformat(),
        "universe_trailing_months": trailing_months,
        "datasets": list(datasets),
        "refresh_bronze": refresh_bronze,
    }
    if (
        not force
        and not refresh_bronze
        and _bulk_day_aggs_complete_with_params(
            outputs,
            manifest_path=outputs[1],
            expected_params=params,
        )
    ):
        return DataPipelineStep(
            "options-day-aggs-bulk",
            "skipped",
            outputs,
            reason="outputs_exist_params_match",
        )

    dates = _weekday_dates(lookback_start, end_date)
    tasks = [(date_value, dataset) for date_value in dates for dataset in datasets]
    _progress(
        "options-day-aggs-bulk: "
        f"{len(dates)} weekdays, {len(tasks)} dataset-date partitions, jobs={jobs}"
    )
    out.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, object]] = []
    completed = 0
    with ThreadPoolExecutor(max_workers=max(1, jobs)) as executor:
        futures = {
            executor.submit(
                _ensure_bulk_day_agg_partition,
                config,
                dataset=str(dataset),
                date_value=date_value,
                refresh_bronze=refresh_bronze,
            ): (date_value, dataset)
            for date_value, dataset in tasks
        }
        for future in as_completed(futures):
            date_value, dataset = futures[future]
            try:
                row = future.result()
            except Exception as exc:
                row = {
                    "date": date_value.isoformat(),
                    "dataset": str(dataset),
                    "status": "failed",
                    "error": str(exc),
                }
            rows.append(row)
            completed += 1
            if completed == len(tasks) or completed % 25 == 0:
                counts = Counter(str(item.get("status")) for item in rows)
                _progress(
                    "options-day-aggs-bulk progress: "
                    f"{completed}/{len(tasks)} "
                    + ", ".join(f"{key}={value}" for key, value in sorted(counts.items()))
                )

    frame = pd.DataFrame(rows).sort_values(["date", "dataset"]).reset_index(drop=True)
    frame.to_csv(outputs[0], index=False)
    status_counts = Counter(frame["status"].astype(str)) if not frame.empty else Counter()
    dataset_counts: dict[str, dict[str, int]] = {}
    if not frame.empty:
        for dataset, group in frame.groupby("dataset"):
            dataset_counts[str(dataset)] = {
                str(status): int(count)
                for status, count in group["status"].astype(str).value_counts().items()
            }
    manifest = {
        "pipeline_params": params,
        "status_counts": dict(status_counts),
        "dataset_counts": dataset_counts,
        "date_range": {"start": start_date.isoformat(), "end": end_date.isoformat()},
        "universe_lookback_start": lookback_start.isoformat(),
        "outputs": [str(path) for path in outputs],
    }
    outputs[1].write_text(json.dumps(manifest, indent=2, default=str), encoding="utf-8")
    success_statuses = {"hit", "downloaded", "repaired"}
    options_success = int(
        frame.loc[
            frame["dataset"].astype(str).eq("options_day_aggs")
            & frame["status"].astype(str).isin(success_statuses)
        ].shape[0]
    )
    failed = int(frame["status"].astype(str).eq("failed").sum()) if not frame.empty else 0
    if failed:
        return DataPipelineStep(
            "options-day-aggs-bulk",
            "blocked",
            outputs,
            reason="bulk_day_agg_failures",
            metadata={
                "status_counts": dict(status_counts),
                "dataset_counts": dataset_counts,
            },
        )
    if options_success == 0:
        return DataPipelineStep(
            "options-day-aggs-bulk",
            "blocked",
            outputs,
            reason="no_options_day_aggs_available",
            metadata={
                "status_counts": dict(status_counts),
                "dataset_counts": dataset_counts,
            },
        )
    return DataPipelineStep(
        "options-day-aggs-bulk",
        "ran",
        outputs,
        metadata={
            "status_counts": dict(status_counts),
            "dataset_counts": dataset_counts,
            "weekdays": len(dates),
        },
    )


def _path_signature(path: Path) -> dict[str, object]:
    if not path.exists():
        return {"path": str(path), "exists": False}
    stat = path.stat()
    return {
        "path": str(path),
        "exists": True,
        "size": stat.st_size,
        "mtime_ns": stat.st_mtime_ns,
    }


def _event_month(value: object) -> date | None:
    parsed = pd.to_datetime(value, errors="coerce")
    if pd.isna(parsed):
        return None
    return date(int(parsed.year), int(parsed.month), 1)


def _frame_value_counts(frame: pd.DataFrame, column: str) -> dict[str, int]:
    if frame.empty or column not in frame.columns:
        return {}
    counts = frame[column].astype(str).value_counts().to_dict()
    return {str(key): int(value) for key, value in counts.items()}


def _apply_dynamic_universe_membership(
    calendar: pd.DataFrame,
    universe: pd.DataFrame,
) -> tuple[pd.DataFrame, dict[str, int]]:
    if calendar.empty:
        out = calendar.copy()
        out["universe_month"] = pd.Series(dtype="object")
        out["universe_rank"] = pd.Series(dtype="Int64")
        out["in_universe"] = pd.Series(dtype="bool")
        out["universe_filter_status"] = pd.Series(dtype="object")
        return out, {"no_universe_membership": 0, "bad_event_month": 0}

    required = {"ticker", "universe_month", "rank"}
    missing = sorted(required - set(universe.columns))
    if missing:
        raise ValueError(f"monthly universe missing required columns: {missing}")

    membership: dict[str, list[tuple[date, int]]] = {}
    for row in universe.to_dict("records"):
        ticker = str(row.get("ticker") or "").upper()
        month = _event_month(row.get("universe_month"))
        if not ticker or month is None:
            continue
        membership.setdefault(ticker, []).append((month, int(row.get("rank") or 0)))
    for values in membership.values():
        values.sort(key=lambda item: item[0])

    universe_months: list[date | None] = []
    ranks: list[int | None] = []
    statuses: list[str] = []
    counts: Counter[str] = Counter()
    for row in calendar.to_dict("records"):
        ticker = str(row.get("ticker") or "").upper()
        event_month = _event_month(row.get("announcement_date"))
        if event_month is None:
            universe_months.append(None)
            ranks.append(None)
            statuses.append("bad_event_month")
            counts["bad_event_month"] += 1
            continue
        candidates = [
            (month, rank) for month, rank in membership.get(ticker, []) if month <= event_month
        ]
        if not candidates:
            universe_months.append(None)
            ranks.append(None)
            statuses.append("no_universe_membership")
            counts["no_universe_membership"] += 1
            continue
        month, rank = candidates[-1]
        universe_months.append(month)
        ranks.append(rank)
        statuses.append("in_universe")
        counts["in_universe"] += 1

    out = calendar.copy()
    out["universe_month"] = universe_months
    out["universe_rank"] = pd.Series(ranks, dtype="Int64")
    out["in_universe"] = [status == "in_universe" for status in statuses]
    out["universe_filter_status"] = statuses
    return out, {str(key): int(value) for key, value in counts.items()}


def _dynamic_calendar_step(
    config: ProjectConfig,
    *,
    out_root: Path,
    start_date: date,
    end_date: date,
    sec_submissions_dir: Path | None,
    massive_8k_text_dir: Path | None,
    validate_with_massive: bool,
    force: bool,
) -> DataPipelineStep:
    out = out_root / "dynamic_calendar"
    universe_path = out_root / "universe" / "monthly_top50_universe.parquet"
    outputs = (
        out / "earnings_calendar_candidates.csv",
        out / "earnings_calendar_candidates.parquet",
        out / "earnings_calendar_report.json",
    )
    if not universe_path.exists():
        return DataPipelineStep(
            "dynamic-calendar",
            "blocked",
            outputs,
            reason="requires universe/monthly_top50_universe.parquet",
        )
    universe = pl.read_parquet(universe_path).to_pandas()
    tickers = sorted({str(ticker).upper() for ticker in universe.get("ticker", []) if ticker})
    if not tickers:
        return DataPipelineStep(
            "dynamic-calendar",
            "blocked",
            outputs,
            reason="monthly universe has no tickers",
        )
    params = {
        "stage": "dynamic-calendar",
        "universe": _path_signature(universe_path),
        "start_date": start_date.isoformat(),
        "end_date": end_date.isoformat(),
        "sec_submissions_dir": str(sec_submissions_dir) if sec_submissions_dir else None,
        "massive_8k_text_dir": str(massive_8k_text_dir) if massive_8k_text_dir else None,
        "validate_with_massive": validate_with_massive,
    }
    if not force and _complete_with_params(
        outputs,
        params_path=outputs[2],
        expected_params=params,
    ):
        return DataPipelineStep(
            "dynamic-calendar",
            "skipped",
            outputs,
            reason="outputs_exist_params_match",
        )

    frame, report = build_earnings_calendar_candidates(
        config=config,
        tickers=tickers,
        start_date=start_date,
        end_date=end_date,
        sec_submissions_dir=sec_submissions_dir,
        massive_8k_text_dir=massive_8k_text_dir,
        validate_with_massive=validate_with_massive,
        fail_on_missing_tickers=False,
    )
    annotated, universe_counts = _apply_dynamic_universe_membership(frame, universe)
    filtered = annotated.loc[annotated["in_universe"].astype(bool)].copy()
    out.mkdir(parents=True, exist_ok=True)
    filtered.to_csv(outputs[0], index=False)
    _write_parquet(outputs[1], filtered)
    report.update(
        {
            "pipeline_params": params,
            "pre_universe_filter_rows": int(len(frame)),
            "row_count": int(len(filtered)),
            "main_sample_candidate_rows": int(
                filtered["is_main_sample_candidate"].sum()
                if "is_main_sample_candidate" in filtered
                else 0
            ),
            "rows_by_ticker": _frame_value_counts(filtered, "ticker"),
            "timing_counts": _frame_value_counts(filtered, "announcement_timing"),
            "text_validation_counts": _frame_value_counts(filtered, "text_validation_status"),
            "text_validation_source_counts": _frame_value_counts(
                filtered, "text_validation_source"
            ),
            "universe_filter_counts": universe_counts,
            "universe_ticker_count": len(tickers),
            "universe_month_count": int(universe["universe_month"].nunique())
            if "universe_month" in universe
            else 0,
        }
    )
    outputs[2].write_text(json.dumps(report, indent=2, default=str), encoding="utf-8")
    return DataPipelineStep(
        "dynamic-calendar",
        "ran",
        outputs,
        metadata={
            "rows": int(len(filtered)),
            "pre_universe_filter_rows": int(len(frame)),
            "universe_ticker_count": len(tickers),
            "universe_filter_counts": universe_counts,
        },
    )


def _contracts_step(
    *,
    out_root: Path,
    events_path: Path | None,
    contracts_path: Path | None,
    dte_min: int,
    dte_max: int,
    force: bool,
) -> DataPipelineStep:
    out = out_root / "contracts"
    outputs = (out / "event_contract_candidates.csv",)
    if events_path is None or contracts_path is None:
        return DataPipelineStep(
            "contracts",
            "blocked",
            outputs,
            reason="requires --events and --contracts",
        )
    if not force and _complete(outputs):
        return DataPipelineStep("contracts", "skipped", outputs, reason="outputs_exist")

    frame = discover_option_contracts(
        pd.read_csv(events_path),
        pd.read_csv(contracts_path),
        dte_min=dte_min,
        dte_max=dte_max,
    )
    out.mkdir(parents=True, exist_ok=True)
    frame.to_csv(outputs[0], index=False)
    return DataPipelineStep(
        "contracts",
        "ran",
        outputs,
        metadata={
            "rows": int(len(frame)),
            "eligible_for_quote_pool": int(frame["eligible_for_quote_pool"].sum()),
            "non_standard_excluded": int(
                frame["contract_discovery_status"].eq("non_standard_excluded").sum()
            ),
        },
    )


def _panel_step(
    *,
    out_root: Path,
    events_path: Path | None,
    quotes_path: Path | None,
    ex_dividends_path: Path | None,
    dte_min: int,
    dte_max: int,
    force: bool,
) -> DataPipelineStep:
    out = out_root / "event_panel"
    outputs = (out / "event_panel.csv",)
    if events_path is None or quotes_path is None:
        return DataPipelineStep(
            "panel",
            "blocked",
            outputs,
            reason="requires --events and --quotes",
        )
    if not force and _complete(outputs):
        return DataPipelineStep("panel", "skipped", outputs, reason="outputs_exist")

    ex_dividends = pd.read_csv(ex_dividends_path) if ex_dividends_path else None
    frame = build_event_panel(
        pd.read_csv(events_path),
        pd.read_csv(quotes_path),
        ex_dividends=ex_dividends,
        dte_min=dte_min,
        dte_max=dte_max,
    )
    out.mkdir(parents=True, exist_ok=True)
    frame.to_csv(outputs[0], index=False)
    return DataPipelineStep(
        "panel",
        "ran",
        outputs,
        metadata={
            "rows": int(len(frame)),
            "spot_fallback_rows": int(frame["forward_source"].eq("spot_fallback").sum()),
            "put_call_parity_rows": int(frame["forward_source"].eq("put_call_parity").sum()),
        },
    )


def _event_window_panel_step(
    config: ProjectConfig,
    *,
    out_root: Path,
    force: bool,
    dte_min: int,
    dte_max: int,
    max_events: int | None,
    calendar_path: Path | None = None,
) -> DataPipelineStep:
    effective_calendar_path = calendar_path or (
        out_root / "dynamic_calendar" / "earnings_calendar_candidates.csv"
    )
    report_path = out_root / "event_window_panel" / "event_window_panel_report.json"
    outputs = (
        config.silver_data_dir / "event_windows" / "event_windows.parquet",
        config.silver_data_dir / "contracts" / "event_contract_candidates.parquet",
        report_path,
    )
    if not effective_calendar_path.exists():
        return DataPipelineStep(
            "event-window-panel",
            "blocked",
            outputs,
            reason="requires dynamic-calendar earnings_calendar_candidates.csv",
        )
    second_expiry_dte_max = max(dte_max + 14, 28)
    params = {
        "stage": "event-window-panel",
        "calendar": str(effective_calendar_path),
        "dte_min": dte_min,
        "dte_max": dte_max,
        "ivar_support_dte_max": second_expiry_dte_max,
        "max_events": max_events,
    }
    if not force and _complete_with_params(
        outputs,
        params_path=report_path,
        expected_params=params,
    ):
        return DataPipelineStep(
            "event-window-panel",
            "skipped",
            outputs,
            reason="outputs_exist_params_match",
        )
    try:
        report = build_event_window_panel(
            config=config,
            calendar_path=effective_calendar_path,
            out_root=out_root,
            dte_min=dte_min,
            dte_max=dte_max,
            strikes_per_expiry=3,
            max_events=max_events,
        )
    except Exception as exc:
        return DataPipelineStep(
            "event-window-panel",
            "blocked",
            outputs,
            reason="event_window_panel_failed",
            metadata={"error": str(exc)},
        )
    metadata: dict[str, object] = {
        key: int(value) if isinstance(value, int | float | str) else 0
        for key, value in {
            "events": report.get("events"),
            "contracts": report.get("contracts"),
            "quote_pool_contracts": report.get("quote_pool_contracts"),
        }.items()
    }
    return DataPipelineStep(
        "event-window-panel",
        "ran" if _complete(outputs) else "blocked",
        outputs,
        reason=None if _complete(outputs) else "event_window_panel_outputs_missing",
        metadata=metadata,
    )


def _contract_reference_validation_step(
    config: ProjectConfig,
    *,
    out_root: Path,
    force: bool,
    jobs: int,
    max_contracts: int | None,
    refresh_bronze: bool,
) -> DataPipelineStep:
    candidate_path = config.silver_data_dir / "contracts" / "event_contract_candidates.parquet"
    reference_path = config.silver_data_dir / "contracts" / "contract_reference_validation.parquet"
    out = out_root / "contract_reference_validation"
    report_path = out / "contract_reference_fetch_report.csv"
    manifest_path = out / "contract_reference_validation_manifest.json"
    outputs = (candidate_path, reference_path, report_path, manifest_path)
    if not candidate_path.exists():
        return DataPipelineStep(
            "contract-reference-validation",
            "blocked",
            outputs,
            reason="requires event_contract_candidates.parquet",
        )

    params = {
        "stage": "contract-reference-validation",
        "candidate_path": str(candidate_path),
        "max_contracts": max_contracts,
        "refresh_bronze": refresh_bronze,
        "schema_version": CONTRACT_REFERENCE_SCHEMA_VERSION,
    }
    required_candidate_columns = {
        "contract_reference_status",
        "contract_reference_validated",
        "contract_reference_source_dataset",
    }
    if (
        not force
        and not refresh_bronze
        and _complete_with_params(
            outputs,
            params_path=manifest_path,
            expected_params=params,
        )
        and _parquet_has_columns(candidate_path, required_candidate_columns)
    ):
        return DataPipelineStep(
            "contract-reference-validation",
            "skipped",
            outputs,
            reason="outputs_exist_params_match",
        )

    candidates = pd.read_parquet(candidate_path)
    if "options_ticker" not in candidates.columns:
        return DataPipelineStep(
            "contract-reference-validation",
            "blocked",
            outputs,
            reason="candidate_contracts_missing_options_ticker",
        )
    unique_tickers = (
        candidates["options_ticker"]
        .dropna()
        .astype(str)
        .str.strip()
        .str.upper()
        .loc[lambda series: series.ne("")]
        .drop_duplicates()
        .sort_values()
        .tolist()
    )
    if max_contracts is not None:
        unique_tickers = unique_tickers[:max_contracts]
    if not unique_tickers:
        return DataPipelineStep(
            "contract-reference-validation",
            "blocked",
            outputs,
            reason="no_candidate_option_tickers",
        )

    _progress(
        "contract-reference-validation: "
        f"contracts={len(unique_tickers)} jobs={jobs} refresh_bronze={refresh_bronze}"
    )
    out.mkdir(parents=True, exist_ok=True)
    cache_root = config.bronze_data_dir / "massive" / "options_contract_reference"
    rows: list[dict[str, object]] = []
    completed = 0
    with (
        httpx.Client(timeout=config.massive_request_timeout_seconds) as client,
        ThreadPoolExecutor(max_workers=max(1, jobs)) as executor,
    ):
        futures = {
            executor.submit(
                fetch_massive_option_contract_reference,
                client,
                config,
                options_ticker=option_ticker,
                cache_root=cache_root,
                refresh_bronze=refresh_bronze,
            ): option_ticker
            for option_ticker in unique_tickers
        }
        for future in as_completed(futures):
            option_ticker = futures[future]
            try:
                row = future.result().report_row()
            except Exception as exc:
                row = {
                    "options_ticker": option_ticker,
                    "fetch_status": "failed",
                    "contract_reference_status": "fetch_failed",
                    "contract_reference_error": str(exc),
                }
            rows.append(row)
            completed += 1
            if completed == len(unique_tickers) or completed % 25 == 0:
                counts = Counter(str(item.get("fetch_status")) for item in rows)
                _progress(
                    "contract-reference-validation progress: "
                    f"{completed}/{len(unique_tickers)} "
                    + ", ".join(f"{key}={value}" for key, value in sorted(counts.items()))
                )

    reference_report = pd.DataFrame(rows).sort_values("options_ticker").reset_index(drop=True)
    validated = apply_contract_reference_validation(candidates, reference_report)
    _write_parquet(candidate_path, validated)
    _write_parquet(reference_path, reference_report)
    reference_report.to_csv(report_path, index=False)
    fetch_counts = (
        reference_report["fetch_status"].astype(str).value_counts().to_dict()
        if "fetch_status" in reference_report
        else {}
    )
    status_counts = (
        reference_report["contract_reference_status"].astype(str).value_counts().to_dict()
        if "contract_reference_status" in reference_report
        else {}
    )
    non_standard_rows = int(
        validated["contract_discovery_status"].astype(str).eq("non_standard_excluded").sum()
    )
    manifest = {
        "pipeline_params": params,
        "status_counts": {str(key): int(value) for key, value in status_counts.items()},
        "fetch_status_counts": {str(key): int(value) for key, value in fetch_counts.items()},
        "candidate_rows": int(len(candidates)),
        "candidate_rows_after_validation": int(len(validated)),
        "reference_contracts_requested": int(len(unique_tickers)),
        "validated_contracts": int(
            reference_report["contract_reference_status"]
            .astype(str)
            .eq(REFERENCE_STATUS_VALIDATED)
            .sum()
        )
        if "contract_reference_status" in reference_report
        else 0,
        "non_standard_excluded_rows": non_standard_rows,
        "outputs": [str(path) for path in outputs],
    }
    manifest_path.write_text(json.dumps(manifest, indent=2, default=str), encoding="utf-8")
    return DataPipelineStep(
        "contract-reference-validation",
        "ran",
        outputs,
        metadata={
            "reference_contracts_requested": int(len(unique_tickers)),
            "status_counts": manifest["status_counts"],
            "fetch_status_counts": manifest["fetch_status_counts"],
            "non_standard_excluded_rows": non_standard_rows,
        },
    )


def _trade_proxy_panel_step(
    config: ProjectConfig,
    *,
    out_root: Path,
    force: bool,
    max_events: int | None,
    max_contracts: int | None,
    jobs: int,
    lookback_seconds: int,
    second_agg_buffer_minutes: int,
    price_field: str,
    refresh_bronze: bool,
) -> DataPipelineStep:
    outputs = (
        config.gold_data_dir / "event_panel" / "trade_proxy_event_panel.parquet",
        out_root / "trade_proxy_panel" / "trade_proxy_panel_report.json",
    )
    params = {
        "stage": "trade-proxy-panel",
        "max_events": max_events,
        "max_contracts": max_contracts,
        "lookback_seconds": lookback_seconds,
        "second_agg_buffer_minutes": second_agg_buffer_minutes,
        "price_field": price_field,
    }
    outputs_complete = _complete(outputs)
    if not force and _complete_with_params(
        outputs,
        params_path=outputs[1],
        expected_params=params,
    ):
        return DataPipelineStep(
            "trade-proxy-panel",
            "skipped",
            outputs,
            reason="outputs_exist_params_match",
        )
    command = [
        sys.executable,
        str(config.repo_root / "scripts" / "build_trade_proxy_panel.py"),
        "--out-dir",
        str(out_root),
        "--jobs",
        str(jobs),
        "--lookback-seconds",
        str(lookback_seconds),
        "--second-agg-buffer-minutes",
        str(second_agg_buffer_minutes),
        "--price-field",
        price_field,
    ]
    if force or outputs_complete:
        command.append("--force")
    if refresh_bronze:
        command.append("--refresh-bronze")
    if max_events is not None:
        command.extend(["--max-events", str(max_events)])
    if max_contracts is not None:
        command.extend(["--max-contracts", str(max_contracts)])
    result = _run_command_with_progress(
        command,
        cwd=config.repo_root,
        label="trade-proxy-panel",
    )
    metadata: dict[str, object] = {"returncode": result.returncode}
    if result.stdout.strip():
        metadata["stdout_tail"] = result.stdout[-1000:]
    if result.stderr.strip():
        metadata["stderr_tail"] = result.stderr[-1000:]
    return DataPipelineStep(
        "trade-proxy-panel",
        "ran" if result.returncode == 0 and _complete(outputs) else "blocked",
        outputs,
        reason=None if result.returncode == 0 else "trade_proxy_panel_script_failed",
        metadata=metadata,
    )


def _dry_run_estimate(
    *,
    out_root: Path,
    stage: str,
    tickers: Sequence[str],
    start_date: date,
    end_date: date,
    dte_min: int,
    dte_max: int,
    max_events: int | None,
    max_contracts: int | None,
    lookback_seconds: int,
    second_agg_buffer_minutes: int,
    price_field: str,
    universe_top_n: int,
    universe_trailing_months: int,
) -> dict[str, object]:
    normalized_tickers = sorted({ticker.upper() for ticker in tickers if ticker.strip()})
    lookback_start = _universe_lookback_start(start_date, universe_trailing_months)
    month_count = (end_date.year - start_date.year) * 12 + end_date.month - start_date.month + 1
    study_calendar_days = (end_date - start_date).days + 1
    bulk_calendar_days = (end_date - lookback_start).days + 1
    estimated_trading_days = int(round(bulk_calendar_days * 252 / 365))
    estimated_years = max(study_calendar_days / 365.25, 1 / 365.25)
    dynamic_ticker_months = month_count * universe_top_n
    estimated_unique_dynamic_tickers = max(universe_top_n, min(dynamic_ticker_months, 250))
    ticker_count = (
        estimated_unique_dynamic_tickers if stage == "proxy-all" else len(normalized_tickers)
    )
    estimated_events = (
        max_events if max_events is not None else int(round(ticker_count * 4 * estimated_years))
    )
    estimated_contracts = max_contracts if max_contracts is not None else int(estimated_events * 18)
    exclusions = {
        "non_bmo_amc_timing": "estimated_after_calendar_stage",
        "missing_sec_or_text_validation": "estimated_after_calendar_stage",
        "no_universe_membership": "estimated_after_universe_assignment",
        "missing_underlying_exit_price": "estimated_after_event_windows",
        "missing_contract_metadata": "estimated_after_contract_metadata",
        f"no_dte_{dte_min}_{dte_max}_contracts": "estimated_after_contract_discovery",
        "no_dte_5_14_main_eligible_contracts": "estimated_after_contract_discovery",
        "missing_second_agg_entry_window": "estimated_after_second_agg_fetch",
        "missing_option_day_agg_exit_price": "estimated_after_exit_price_join",
    }
    payload: dict[str, object] = {
        "ok": True,
        "dry_run": True,
        "stage": stage,
        "out_dir": str(out_root),
        "date_range": {"start": start_date.isoformat(), "end": end_date.isoformat()},
        "bulk_day_aggs_date_range": {
            "start": lookback_start.isoformat(),
            "end": end_date.isoformat(),
        },
        "planned_stages": [
            "options-day-aggs-bulk",
            "universe",
            "dynamic-calendar",
            "event-window-panel",
            "contract-reference-validation",
            "trade-proxy-panel",
        ]
        if stage == "proxy-all"
        else [stage],
        "estimated_counts": {
            "study_calendar_days": study_calendar_days,
            "bulk_calendar_days": bulk_calendar_days,
            "trading_days": estimated_trading_days,
            "months": month_count,
            "universe_ticker_months": dynamic_ticker_months if stage == "proxy-all" else None,
            "tickers": ticker_count,
            "events": estimated_events,
            "contracts": estimated_contracts,
            "contract_reference_rest_calls": estimated_contracts,
            "second_agg_rest_calls": estimated_contracts,
            "bulk_day_agg_partitions": estimated_trading_days * 2 if stage == "proxy-all" else None,
        },
        "parameters": {
            "dte_min": dte_min,
            "dte_max": dte_max,
            "lookback_seconds": lookback_seconds,
            "second_agg_buffer_minutes": second_agg_buffer_minutes,
            "price_field": price_field,
            "universe_top_n": universe_top_n,
            "universe_trailing_months": universe_trailing_months,
        },
        "exclusion_estimate": exclusions,
        "writes_data_outputs": False,
    }
    return payload


def run_data_pipeline(
    config: ProjectConfig,
    *,
    stage: str,
    out_root: Path,
    force: bool = False,
    jobs: int = 1,
    tickers: Sequence[str] = DEFAULT_STATIC_TICKERS,
    start_date: date = date(2013, 1, 1),
    end_date: date = date(2025, 12, 31),
    dates: Sequence[date] = (),
    events_path: Path | None = None,
    contracts_path: Path | None = None,
    quotes_path: Path | None = None,
    options_day_aggs_path: Path | None = None,
    ex_dividends_path: Path | None = None,
    sec_submissions_dir: Path | None = None,
    massive_8k_text_dir: Path | None = None,
    validate_with_massive: bool = True,
    dte_min: int = 5,
    dte_max: int = 14,
    max_events: int | None = None,
    max_contracts: int | None = None,
    download_samples: bool = False,
    lookback_seconds: int = 900,
    second_agg_buffer_minutes: int = 60,
    price_field: str = "option_vwap",
    dry_run: bool = False,
    universe_top_n: int = 50,
    universe_trailing_months: int = 6,
    refresh_bronze: bool = False,
) -> dict[str, object]:
    normalized_stage = stage.strip().lower()
    if normalized_stage not in SUPPORTED_DATA_STAGES:
        raise ValueError(f"unsupported data stage: {stage}")
    if jobs <= 0:
        raise ValueError("jobs must be positive.")
    if start_date > end_date:
        raise ValueError("start_date must be <= end_date.")
    if second_agg_buffer_minutes <= 0:
        raise ValueError("second_agg_buffer_minutes must be positive.")
    if universe_top_n <= 0:
        raise ValueError("universe_top_n must be positive.")
    if universe_trailing_months <= 0:
        raise ValueError("universe_trailing_months must be positive.")
    if dry_run:
        return _dry_run_estimate(
            out_root=out_root,
            stage=normalized_stage,
            tickers=tickers,
            start_date=start_date,
            end_date=end_date,
            dte_min=dte_min,
            dte_max=dte_max,
            max_events=max_events,
            max_contracts=max_contracts,
            lookback_seconds=lookback_seconds,
            second_agg_buffer_minutes=second_agg_buffer_minutes,
            price_field=price_field,
            universe_top_n=universe_top_n,
            universe_trailing_months=universe_trailing_months,
        )

    if normalized_stage == "all":
        stages = [
            "fixture-audit",
            "massive-probe",
            "options-day-aggs-bulk",
            "universe",
            "dynamic-calendar",
            "contracts",
            "panel",
            "event-window-panel",
            "contract-reference-validation",
        ]
    elif normalized_stage == "proxy-all":
        stages = [
            "options-day-aggs-bulk",
            "universe",
            "dynamic-calendar",
            "event-window-panel",
            "contract-reference-validation",
            "trade-proxy-panel",
        ]
    else:
        stages = [normalized_stage]
    normalized_tickers = sorted({ticker.upper() for ticker in tickers if ticker.strip()})
    if not normalized_tickers:
        normalized_tickers = list(DEFAULT_STATIC_TICKERS)

    stage_force = force
    step_builders: dict[str, Callable[[], list[DataPipelineStep]]] = {
        "fixture-audit": lambda: [
            _fixture_audit_step(config, out_root=out_root, force=stage_force)
        ],
        "massive-probe": lambda: _massive_probe_steps(
            config,
            out_root=out_root,
            dates=dates,
            force=stage_force,
            jobs=jobs,
            download_samples=download_samples,
        ),
        "market-covariates": lambda: [
            _market_covariates_step(config, out_root=out_root, force=stage_force)
        ],
        "market-second-covariates": lambda: [
            _market_second_covariates_step(
                config,
                out_root=out_root,
                force=stage_force,
                refresh_bronze=refresh_bronze,
                jobs=jobs,
                max_events=max_events,
                lookback_seconds=lookback_seconds,
                second_agg_buffer_minutes=second_agg_buffer_minutes,
                price_field=price_field,
            )
        ],
        "options-day-aggs-bulk": lambda: [
            _options_day_aggs_bulk_step(
                config,
                out_root=out_root,
                start_date=start_date,
                end_date=end_date,
                trailing_months=universe_trailing_months,
                force=stage_force,
                refresh_bronze=refresh_bronze,
                jobs=jobs,
            )
        ],
        "universe": lambda: [
            _universe_step(
                config,
                out_root=out_root,
                options_day_aggs_path=options_day_aggs_path,
                start_date=start_date,
                end_date=end_date,
                top_n=universe_top_n,
                trailing_months=universe_trailing_months,
                force=stage_force,
            )
        ],
        "dynamic-calendar": lambda: [
            _dynamic_calendar_step(
                config,
                out_root=out_root,
                start_date=start_date,
                end_date=end_date,
                sec_submissions_dir=sec_submissions_dir,
                massive_8k_text_dir=massive_8k_text_dir,
                validate_with_massive=validate_with_massive,
                force=stage_force,
            )
        ],
        "contracts": lambda: [
            _contracts_step(
                out_root=out_root,
                events_path=events_path,
                contracts_path=contracts_path,
                dte_min=dte_min,
                dte_max=dte_max,
                force=stage_force,
            )
        ],
        "panel": lambda: [
            _panel_step(
                out_root=out_root,
                events_path=events_path,
                quotes_path=quotes_path,
                ex_dividends_path=ex_dividends_path,
                dte_min=dte_min,
                dte_max=dte_max,
                force=stage_force,
            )
        ],
        "event-window-panel": lambda: [
            _event_window_panel_step(
                config,
                out_root=out_root,
                force=stage_force,
                dte_min=dte_min,
                dte_max=dte_max,
                max_events=max_events,
                calendar_path=(
                    out_root / "dynamic_calendar" / "earnings_calendar_candidates.csv"
                    if normalized_stage in {"proxy-all", "event-window-panel"}
                    else None
                ),
            )
        ],
        "contract-reference-validation": lambda: [
            _contract_reference_validation_step(
                config,
                out_root=out_root,
                force=stage_force,
                jobs=jobs,
                max_contracts=max_contracts,
                refresh_bronze=refresh_bronze,
            )
        ],
        "trade-proxy-panel": lambda: [
            _trade_proxy_panel_step(
                config,
                out_root=out_root,
                force=stage_force,
                max_events=max_events,
                max_contracts=max_contracts,
                jobs=jobs,
                lookback_seconds=lookback_seconds,
                second_agg_buffer_minutes=second_agg_buffer_minutes,
                price_field=price_field,
                refresh_bronze=refresh_bronze,
            )
        ],
    }

    steps: list[DataPipelineStep] = []
    force_downstream = False
    for index, selected_stage in enumerate(stages, start=1):
        stage_force = force or force_downstream
        started_at = time.perf_counter()
        _progress(f"stage {index}/{len(stages)} start: {selected_stage}")
        stage_steps = step_builders[selected_stage]()
        steps.extend(stage_steps)
        if any(step.status == "ran" for step in stage_steps):
            force_downstream = True
        elapsed = time.perf_counter() - started_at
        status_summary = ", ".join(f"{step.name}={step.status}" for step in stage_steps)
        _progress(
            f"stage {index}/{len(stages)} end: {selected_stage} ({status_summary}, {elapsed:.1f}s)"
        )
        if any(step.status not in {"ran", "skipped"} for step in stage_steps):
            _progress(f"stopping after {selected_stage}: downstream stages are blocked")
            break
    _write_manifest(out_root, steps)
    _progress(f"manifest written: {out_root / 'data_pipeline_manifest.json'}")
    return {
        "ok": all(step.status in {"ran", "skipped"} for step in steps),
        "out_dir": str(out_root),
        "steps": [step.as_dict() for step in steps],
    }

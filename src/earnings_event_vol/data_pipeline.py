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
from typing import cast

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
from earnings_event_vol.earnings_calendar import (
    build_earnings_calendar_candidates,
    fetch_sec_submission_payloads,
    fetch_sec_ticker_map,
)
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
from earnings_event_vol.quote_execution import (
    QUOTE_SOURCE_FLAT_FILE,
    QUOTE_SOURCE_REST,
    extract_quote_execution_panel,
)
from earnings_event_vol.trade_proxy import (
    ENTRY_PRICE_METHOD_PRECLOSE_WINDOW_VWAP,
    EXIT_PRECLOSE_OPTION_VWAP_SOURCE,
    POST_OPEN_OPTION_VWAP_WINDOWS,
    fetch_massive_option_second_aggregates,
    filter_pre_cutoff_buffer,
    normalize_second_aggregates,
    safe_exception_text,
)
from earnings_event_vol.universe import (
    ELIGIBLE_EQUITY_RULE_VERSION,
    build_eligible_equity_tickers,
    build_monthly_liquid_universe,
    build_ticker_month_liquidity,
    eligible_equity_cache_matches_rule,
)

TARGET_WINDOW_START = date(2013, 1, 1)
TARGET_WINDOW_END = date(2026, 6, 5)

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
DEFAULT_TRADE_PROXY_REST_LIMIT = 50_000
DEFAULT_TRADE_PROXY_HAIRCUT_FRACTION = 0.10

SUPPORTED_DATA_STAGES = {
    "all",
    "fixture-audit",
    "lake-quality-audit",
    "massive-probe",
    "options-day-aggs-bulk",
    "market-covariates",
    "market-second-covariates",
    "sec-companyfacts",
    "universe",
    "dynamic-calendar",
    "event-window-panel",
    "contract-reference-validation",
    "trade-proxy-panel",
    "quote-execution-panel",
    "quote-execution-merge",
}

ACTIVE_PROXY_DATA_DAG = (
    "options-day-aggs-bulk",
    "universe",
    "dynamic-calendar",
    "sec-companyfacts",
    "event-window-panel",
    "contract-reference-validation",
    "trade-proxy-panel",
    "quote-execution-panel",
)


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


def _write_parquet_from_csv(csv_path: Path, parquet_path: Path) -> int:
    frame = pd.read_csv(csv_path)
    _write_parquet(parquet_path, frame)
    return int(len(frame))


QUOTE_EXECUTION_DATASET_FILES: dict[str, tuple[str, str, str]] = {
    "bronze_quote_window_requests": (
        "bronze",
        "quote_window_requests.parquet",
        "quote_window_requests.csv",
    ),
    "bronze_quote_window_quotes": (
        "bronze",
        "quote_window_quotes.parquet",
        "quote_window_quotes.csv",
    ),
    "silver_quote_window_marks": (
        "silver",
        "quote_window_marks.parquet",
        "quote_window_marks.csv",
    ),
    "silver_quote_execution_legs": (
        "silver",
        "quote_execution_legs.parquet",
        "quote_execution_legs.csv",
    ),
    "gold_quote_straddle_execution": (
        "gold",
        "quote_straddle_execution.parquet",
        "quote_straddle_execution.csv",
    ),
    "gold_quote_ivar_event": (
        "gold",
        "quote_ivar_event.parquet",
        "quote_ivar_event.csv",
    ),
    "gold_quote_iv_surface": (
        "gold",
        "quote_iv_surface.parquet",
        "quote_iv_surface.csv",
    ),
    "gold_quote_iv_surface_summary": (
        "gold",
        "quote_iv_surface_summary.parquet",
        "quote_iv_surface_summary.csv",
    ),
    "gold_quote_surface_ivar_event": (
        "gold",
        "quote_surface_ivar_event.parquet",
        "quote_surface_ivar_event.csv",
    ),
    "gold_quote_execution_confidence": (
        "gold",
        "quote_execution_confidence.parquet",
        "quote_execution_confidence.csv",
    ),
}

QUOTE_EXECUTION_DEDUPE_KEYS: dict[str, tuple[str, ...]] = {
    "bronze_quote_window_requests": (
        "event_id",
        "options_ticker",
        "window_label",
        "window_start",
        "window_end",
    ),
    "bronze_quote_window_quotes": (
        "event_id",
        "options_ticker",
        "window_label",
        "quote_timestamp_et",
        "bid",
        "ask",
    ),
    "silver_quote_window_marks": ("event_id", "options_ticker", "window_label"),
    "silver_quote_execution_legs": ("event_id", "options_ticker", "window_label"),
    "gold_quote_straddle_execution": ("event_id", "expiration", "strike"),
    "gold_quote_ivar_event": ("event_id",),
    "gold_quote_iv_surface": ("event_id", "options_ticker", "expiration", "strike", "right"),
    "gold_quote_iv_surface_summary": ("event_id", "expiration", "strike"),
    "gold_quote_surface_ivar_event": ("event_id",),
    "gold_quote_execution_confidence": ("event_id",),
}


def _normalize_quote_batch_label(label: str | None) -> str | None:
    if label is None:
        return None
    normalized = str(label).strip()
    if not normalized:
        return None
    if not all(ch.isalnum() or ch in {"_", "-"} for ch in normalized):
        raise ValueError("quote_batch_label must contain only letters, numbers, '_' or '-'.")
    return normalized


def _quote_execution_lake_paths(
    config: ProjectConfig, *, batch_label: str | None = None
) -> dict[str, Path]:
    normalized_batch_label = _normalize_quote_batch_label(batch_label)
    roots = {
        "bronze": config.bronze_data_dir / "massive" / "quotes_v1_target_windows",
        "silver": config.silver_data_dir / "quote_execution",
        "gold": config.gold_data_dir / "quote_execution",
    }
    if normalized_batch_label is not None:
        batch_partition = f"batch={normalized_batch_label}"
        roots = {layer: root / "batches" / batch_partition for layer, root in roots.items()}
    return {
        dataset_id: roots[layer] / parquet_name
        for dataset_id, (layer, parquet_name, _csv_name) in QUOTE_EXECUTION_DATASET_FILES.items()
    }


def _quote_execution_artifact_paths(
    out_root: Path, *, batch_label: str | None = None
) -> dict[str, Path]:
    normalized_batch_label = _normalize_quote_batch_label(batch_label)
    root = out_root / "quote_execution_panel"
    if normalized_batch_label is not None:
        root = root / "batches" / normalized_batch_label
    return {
        dataset_id: root / csv_name
        for dataset_id, (_layer, _parquet_name, csv_name) in QUOTE_EXECUTION_DATASET_FILES.items()
    }


def _quote_execution_artifact_root(out_root: Path, *, batch_label: str | None = None) -> Path:
    normalized_batch_label = _normalize_quote_batch_label(batch_label)
    root = out_root / "quote_execution_panel"
    if normalized_batch_label is not None:
        root = root / "batches" / normalized_batch_label
    return root


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


XBRL_CONCEPTS = {
    "Assets": "assets",
    "Liabilities": "liabilities",
    "CashAndCashEquivalentsAtCarryingValue": "cash",
    "CashAndCashEquivalentsAndShortTermInvestments": "cash",
    "CurrentAssets": "current_assets",
    "AssetsCurrent": "current_assets",
    "CurrentLiabilities": "current_liabilities",
    "LiabilitiesCurrent": "current_liabilities",
    "Revenues": "revenue",
    "RevenueFromContractWithCustomerExcludingAssessedTax": "revenue",
    "SalesRevenueNet": "revenue",
    "NetIncomeLoss": "net_income",
    "OperatingIncomeLoss": "operating_income",
}


def _submission_acceptance_lookup(payload: Mapping[str, object]) -> dict[str, str]:
    out: dict[str, str] = {}
    filings = payload.get("filings") if isinstance(payload.get("filings"), dict) else {}
    blocks: list[Mapping[str, object]] = []
    recent = filings.get("recent") if isinstance(filings, dict) else None
    if isinstance(recent, Mapping):
        blocks.append(recent)
    archives = payload.get("archive_payloads")
    if isinstance(archives, list):
        blocks.extend(archive for archive in archives if isinstance(archive, Mapping))
    for block in blocks:
        accns = block.get("accessionNumber")
        acceptances = block.get("acceptanceDateTime")
        if not isinstance(accns, list) or not isinstance(acceptances, list):
            continue
        for accn, acceptance in zip(accns, acceptances, strict=False):
            if accn and acceptance:
                out[str(accn)] = str(acceptance)
    return out


def _normalize_companyfacts_payload(
    *,
    ticker: str,
    cik: int,
    payload: Mapping[str, object],
    acceptance_lookup: Mapping[str, str],
) -> pd.DataFrame:
    facts = payload.get("facts") if isinstance(payload.get("facts"), Mapping) else {}
    us_gaap = facts.get("us-gaap") if isinstance(facts, Mapping) else {}
    rows: list[dict[str, object]] = []
    if not isinstance(us_gaap, Mapping):
        return pd.DataFrame()
    for sec_concept, feature_concept in XBRL_CONCEPTS.items():
        concept_payload = us_gaap.get(sec_concept)
        if not isinstance(concept_payload, Mapping):
            continue
        units = concept_payload.get("units")
        if not isinstance(units, Mapping):
            continue
        for unit, entries in units.items():
            if str(unit).upper() != "USD" or not isinstance(entries, list):
                continue
            for entry in entries:
                if not isinstance(entry, Mapping):
                    continue
                accn = str(entry.get("accn") or "")
                rows.append(
                    {
                        "ticker": ticker,
                        "cik": int(cik),
                        "sec_concept": sec_concept,
                        "feature_concept": feature_concept,
                        "unit": unit,
                        "val": entry.get("val"),
                        "start": entry.get("start"),
                        "end": entry.get("end"),
                        "fy": entry.get("fy"),
                        "fp": entry.get("fp"),
                        "form": entry.get("form"),
                        "filed": entry.get("filed"),
                        "frame": entry.get("frame"),
                        "accn": accn,
                        "acceptance_datetime": acceptance_lookup.get(accn),
                    }
                )
    return pd.DataFrame(rows)


def _sec_companyfacts_step(
    config: ProjectConfig,
    *,
    out_root: Path,
    force: bool,
) -> DataPipelineStep:
    calendar_path = out_root / "dynamic_calendar" / "earnings_calendar_candidates.csv"
    out = out_root / "sec_companyfacts"
    raw_dir = config.bronze_data_dir / "sec" / "companyfacts"
    silver_path = config.silver_data_dir / "sec" / "companyfacts.parquet"
    diagnostics_path = out / "sec_companyfacts_diagnostics.csv"
    manifest_path = out / "sec_companyfacts_manifest.json"
    outputs = (silver_path, diagnostics_path, manifest_path)
    if not calendar_path.exists():
        return DataPipelineStep(
            "sec-companyfacts",
            "blocked",
            outputs,
            reason="requires dynamic-calendar earnings_calendar_candidates.csv",
        )
    calendar = pd.read_csv(calendar_path)
    tickers = sorted({str(ticker).upper() for ticker in calendar.get("ticker", []) if ticker})
    params = {
        "stage": "sec-companyfacts",
        "calendar": _path_signature(calendar_path),
        "endpoint": config.sec_companyfacts_url_template,
        "request_interval_seconds": 0.125,
    }
    if not force and _complete_with_params(
        outputs,
        params_path=manifest_path,
        expected_params=params,
    ):
        return DataPipelineStep(
            "sec-companyfacts",
            "skipped",
            outputs,
            reason="outputs_exist_params_match",
        )

    out.mkdir(parents=True, exist_ok=True)
    raw_dir.mkdir(parents=True, exist_ok=True)
    rows: list[pd.DataFrame] = []
    diagnostics: list[dict[str, object]] = []
    try:
        with httpx.Client(timeout=config.massive_request_timeout_seconds) as client:
            ticker_map = fetch_sec_ticker_map(client, config)
            submissions = fetch_sec_submission_payloads(
                tickers=tickers,
                config=config,
                client=client,
                archive_cache_dir=config.bronze_data_dir / "sec" / "submissions",
                fail_on_missing_tickers=False,
                request_interval_seconds=0.125,
            )
            last_request_at = 0.0
            for ticker in tickers:
                cik = ticker_map.get(ticker)
                if cik is None:
                    diagnostics.append({"ticker": ticker, "status": "missing_cik"})
                    continue
                try:
                    cache_path = raw_dir / f"CIK{int(cik):010d}.json"
                    status = "cache_hit"
                    if force or not cache_path.exists() or cache_path.stat().st_size <= 0:
                        elapsed = time.perf_counter() - last_request_at
                        if elapsed < 0.125:
                            time.sleep(0.125 - elapsed)
                        response = client.get(
                            config.sec_companyfacts_url_template.format(cik=int(cik)),
                            headers={"User-Agent": config.sec_user_agent},
                        )
                        last_request_at = time.perf_counter()
                        response.raise_for_status()
                        payload = response.json()
                        cache_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
                        status = "fetched"
                    else:
                        payload = json.loads(cache_path.read_text(encoding="utf-8"))
                    acceptance_lookup = _submission_acceptance_lookup(submissions.get(ticker, {}))
                    normalized = _normalize_companyfacts_payload(
                        ticker=ticker,
                        cik=int(cik),
                        payload=payload,
                        acceptance_lookup=acceptance_lookup,
                    )
                    if not normalized.empty:
                        rows.append(normalized)
                    diagnostics.append(
                        {
                            "ticker": ticker,
                            "cik": int(cik),
                            "status": status,
                            "fact_rows": int(len(normalized)),
                            "mapped_acceptance_rows": int(
                                normalized["acceptance_datetime"].notna().sum()
                            )
                            if not normalized.empty
                            else 0,
                            "fallback_filed_rows": int(
                                normalized["acceptance_datetime"].isna().sum()
                            )
                            if not normalized.empty
                            else 0,
                        }
                    )
                except Exception as exc:
                    diagnostics.append(
                        {
                            "ticker": ticker,
                            "cik": int(cik),
                            "status": "ticker_failed_graceful_degradation",
                            "error": safe_exception_text(exc),
                        }
                    )
    except Exception as exc:
        error = safe_exception_text(exc)
        _write_parquet(silver_path, pd.DataFrame(columns=["ticker", "cik", "feature_concept"]))
        pd.DataFrame([{"status": "http_or_parse_degraded", "error": error}]).to_csv(
            diagnostics_path, index=False
        )
        manifest_path.write_text(
            json.dumps(
                {
                    "pipeline_params": params,
                    "status": "degraded",
                    "reason": "sec_companyfacts_failed_graceful_degradation",
                    "error": error,
                },
                indent=2,
                default=str,
            ),
            encoding="utf-8",
        )
        return DataPipelineStep(
            "sec-companyfacts",
            "ran",
            outputs,
            reason="sec_companyfacts_failed_graceful_degradation",
            metadata={"error": error},
        )

    facts = pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()
    if not facts.empty:
        facts["val"] = pd.to_numeric(facts["val"], errors="coerce")
        facts["fy"] = pd.to_numeric(facts["fy"], errors="coerce")
        _write_parquet(silver_path, facts)
    else:
        _write_parquet(silver_path, pd.DataFrame(columns=["ticker", "cik", "feature_concept"]))
    diag = pd.DataFrame(diagnostics)
    diag.to_csv(diagnostics_path, index=False)
    mapped_rows = (
        int(facts["acceptance_datetime"].notna().sum()) if "acceptance_datetime" in facts else 0
    )
    fallback_rows = (
        int(facts["acceptance_datetime"].isna().sum()) if "acceptance_datetime" in facts else 0
    )
    payload = {
        "pipeline_params": params,
        "status": "ran",
        "tickers": int(len(tickers)),
        "fact_rows": int(len(facts)),
        "mapped_acceptance_rows": mapped_rows,
        "mapped_acceptance_share": float(mapped_rows / max(1, len(facts))),
        "fallback_filed_rows": fallback_rows,
        "fallback_filed_share": float(fallback_rows / max(1, len(facts))),
        "missing_cik_share": float(
            diag["status"].eq("missing_cik").mean() if "status" in diag else 0.0
        ),
        "request_interval_seconds": 0.125,
        "outputs": {
            "companyfacts": str(silver_path),
            "diagnostics": str(diagnostics_path),
        },
    }
    manifest_path.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")
    return DataPipelineStep(
        "sec-companyfacts",
        "ran",
        outputs,
        metadata={"fact_rows": int(len(facts)), "tickers": int(len(tickers))},
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
            "error": safe_exception_text(exc),
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
                "error": safe_exception_text(exc),
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

    destination.parent.mkdir(parents=True, exist_ok=True)
    had_existing = destination.exists()

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


def _parquet_row_count(path: Path) -> int | None:
    if not path.exists() or path.stat().st_size <= 0:
        return None
    try:
        return int(pl.scan_parquet(str(path)).select(pl.len()).collect().item())
    except Exception:
        return None


def _extract_date_bounds(
    frame: pd.DataFrame,
    candidate_columns: Sequence[str],
) -> tuple[str | None, date | None, date | None, int]:
    for column in candidate_columns:
        if column not in frame.columns:
            continue
        values = pd.to_datetime(frame[column], errors="coerce", utc=True)
        valid = values.dropna()
        if valid.empty:
            continue
        return (
            column,
            valid.min().date(),
            valid.max().date(),
            int(valid.notna().sum()),
        )
    return None, None, None, 0


def _target_coverage_status(
    *,
    exists: bool,
    first_date: date | None,
    last_date: date | None,
    target_start: date,
    target_end: date,
    available_partitions: int | None = None,
    expected_partitions: int | None = None,
) -> tuple[str, str | None, float | None]:
    if not exists:
        return "missing", "dataset_path_missing", 0.0
    if expected_partitions is None or expected_partitions <= 0 or available_partitions is None:
        partition_coverage = None
    else:
        partition_coverage = float(available_partitions / max(1, expected_partitions))
    if first_date is None or last_date is None:
        return "exists_no_date_bounds", "no_auditable_date_column", partition_coverage
    if first_date <= target_start and last_date >= target_end:
        if partition_coverage is not None and partition_coverage < 0.95:
            return (
                "span_ok_partition_gap",
                "date_span_covers_target_but_partitions_incomplete",
                partition_coverage,
            )
        return "target_span_covered", None, partition_coverage
    missing: list[str] = []
    if first_date > target_start:
        missing.append("history_starts_after_target")
    if last_date < target_end:
        missing.append("history_ends_before_target")
    return "target_span_incomplete", ",".join(missing), partition_coverage


def _lake_dataset_row(
    *,
    dataset_id: str,
    layer: str,
    path: Path,
    target_start: date,
    target_end: date,
    date_columns: Sequence[str],
    required_for_target_window: bool,
    paper_grade_requirement: str,
    note: str,
    partitioned_by_date: bool = False,
    expected_weekday_partitions: int | None = None,
) -> tuple[dict[str, object], pd.DataFrame]:
    exists = path.exists()
    row_count: int | None = None
    event_count: int | None = None
    ticker_count: int | None = None
    partition_count: int | None = None
    first_date: date | None = None
    last_date: date | None = None
    date_column: str | None = None
    date_non_null_rows = 0
    year_rows: list[dict[str, object]] = []
    read_status = "missing" if not exists else "ok"
    read_error: str | None = None

    if exists and partitioned_by_date:
        files = sorted(path.rglob("*.parquet")) if path.is_dir() else [path]
        partition_dates = sorted(
            {value for file in files if (value := _date_partition_value(file)) is not None}
        )
        partition_count = len(partition_dates)
        if partition_dates:
            first_date = partition_dates[0]
            last_date = partition_dates[-1]
            date_column = "date_partition"
        available_by_year = Counter(value.year for value in partition_dates)
        expected_by_year = Counter(value.year for value in _weekday_dates(target_start, target_end))
        for year in range(target_start.year, target_end.year + 1):
            expected = int(expected_by_year.get(year, 0))
            available = int(available_by_year.get(year, 0))
            year_rows.append(
                {
                    "dataset_id": dataset_id,
                    "year": year,
                    "available_rows_or_partitions": available,
                    "expected_weekday_partitions": expected,
                    "coverage_share": float(available / max(1, expected)) if expected else None,
                }
            )
    elif exists:
        try:
            if path.suffix == ".parquet":
                row_count = _parquet_row_count(path)
                frame = pd.read_parquet(path)
            elif path.suffix == ".csv":
                frame = pd.read_csv(path)
                row_count = int(len(frame))
            else:
                frame = pd.DataFrame()
                read_status = "exists_unread_date_not_applicable"
            if not frame.empty:
                if row_count is None:
                    row_count = int(len(frame))
                date_column, first_date, last_date, date_non_null_rows = _extract_date_bounds(
                    frame,
                    date_columns,
                )
                if "event_id" in frame.columns:
                    event_count = int(frame["event_id"].nunique(dropna=True))
                if "ticker" in frame.columns:
                    ticker_count = int(frame["ticker"].nunique(dropna=True))
                if date_column is not None:
                    dates = pd.to_datetime(frame[date_column], errors="coerce", utc=True).dropna()
                    counts = Counter(int(value.year) for value in dates)
                    for year in range(target_start.year, target_end.year + 1):
                        year_rows.append(
                            {
                                "dataset_id": dataset_id,
                                "year": year,
                                "available_rows_or_partitions": int(counts.get(year, 0)),
                                "expected_weekday_partitions": None,
                                "coverage_share": None,
                            }
                        )
        except Exception as exc:
            read_status = "read_failed"
            read_error = safe_exception_text(exc)

    expected_partitions = expected_weekday_partitions if partitioned_by_date else None
    coverage_status, gap_reason, partition_coverage = _target_coverage_status(
        exists=exists,
        first_date=first_date,
        last_date=last_date,
        target_start=target_start,
        target_end=target_end,
        available_partitions=partition_count,
        expected_partitions=expected_partitions,
    )
    row = {
        "dataset_id": dataset_id,
        "layer": layer,
        "path": str(path),
        "exists": bool(exists),
        "read_status": read_status,
        "read_error": read_error,
        "row_count": row_count,
        "event_count": event_count,
        "ticker_count": ticker_count,
        "partition_count": partition_count,
        "date_column": date_column,
        "date_non_null_rows": date_non_null_rows,
        "first_date": first_date.isoformat() if first_date else None,
        "last_date": last_date.isoformat() if last_date else None,
        "target_start": target_start.isoformat(),
        "target_end": target_end.isoformat(),
        "target_coverage_status": coverage_status,
        "gap_reason": gap_reason,
        "required_for_target_window": bool(required_for_target_window),
        "required_for_2013_2025": bool(required_for_target_window),
        "paper_grade_requirement": paper_grade_requirement,
        "expected_weekday_partitions": expected_partitions,
        "partition_coverage_share": partition_coverage,
        "note": note,
    }
    return row, pd.DataFrame(year_rows)


def _lake_quality_audit_step(
    config: ProjectConfig,
    *,
    out_root: Path,
    force: bool,
    target_start: date,
    target_end: date,
) -> DataPipelineStep:
    out = out_root / "lake_quality_audit"
    coverage_path = out / "lake_dataset_coverage.csv"
    year_path = out / "lake_year_coverage.csv"
    report_path = out / "lake_quality_report.json"
    outputs = (coverage_path, year_path, report_path)
    params = {
        "stage": "lake-quality-audit",
        "target_start": target_start.isoformat(),
        "target_end": target_end.isoformat(),
        "data_dir": str(config.data_dir),
        "artifacts_dir": str(config.artifacts_dir),
    }
    if not force and _complete_with_params(
        outputs,
        params_path=report_path,
        expected_params=params,
    ):
        return DataPipelineStep(
            "lake-quality-audit",
            "skipped",
            outputs,
            reason="outputs_exist_params_match",
        )

    expected_weekdays = len(_weekday_dates(target_start, target_end))
    dataset_specs = [
        {
            "dataset_id": "bronze_options_day_aggs",
            "layer": "bronze",
            "path": config.bronze_data_dir / "massive" / "options_day_aggs",
            "date_columns": ("source_date", "date", "quote_date"),
            "required_for_target_window": True,
            "paper_grade_requirement": (
                "historical option chain daily aggregates for universe, contract discovery, "
                "proxy IV surfaces"
            ),
            "note": (
                "Partition-level audit; row counts intentionally skipped to keep the audit fast."
            ),
            "partitioned_by_date": True,
            "expected_weekday_partitions": expected_weekdays,
        },
        {
            "dataset_id": "bronze_underlying_day_aggs",
            "layer": "bronze",
            "path": config.bronze_data_dir / "massive" / "underlying_day_aggs",
            "date_columns": ("source_date", "date"),
            "required_for_target_window": True,
            "paper_grade_requirement": (
                "underlying daily bars for event returns and universe construction"
            ),
            "note": (
                "Partition-level audit; row counts intentionally skipped to keep the audit fast."
            ),
            "partitioned_by_date": True,
            "expected_weekday_partitions": expected_weekdays,
        },
        {
            "dataset_id": "bronze_quote_target_windows",
            "layer": "bronze",
            "path": config.bronze_data_dir
            / "massive"
            / "quotes_v1_target_windows"
            / "quote_window_quotes.parquet",
            "date_columns": ("quote_date", "quote_timestamp_et"),
            "required_for_target_window": True,
            "paper_grade_requirement": (
                "matched targeted quote rows for quote-aware execution diagnostics"
            ),
            "note": "Target-window normalized quote subset only; not full-day raw quote storage.",
        },
        {
            "dataset_id": "silver_earnings_calendar",
            "layer": "silver",
            "path": config.silver_data_dir / "earnings_calendar" / "main_sample.parquet",
            "date_columns": ("announcement_date", "entry_date"),
            "required_for_target_window": True,
            "paper_grade_requirement": (
                "SEC/text-validated event calendar over the target study window"
            ),
            "note": "Main-sample events after dynamic universe and timing filters.",
        },
        {
            "dataset_id": "silver_event_windows",
            "layer": "silver",
            "path": config.silver_data_dir / "event_windows" / "event_windows.parquet",
            "date_columns": ("entry_date", "announcement_date", "event_entry_timestamp"),
            "required_for_target_window": True,
            "paper_grade_requirement": "event-aligned entry/exit windows and contract candidates",
            "note": "One event row per retained earnings event.",
        },
        {
            "dataset_id": "silver_contract_candidates",
            "layer": "silver",
            "path": config.silver_data_dir / "contracts" / "event_contract_candidates.parquet",
            "date_columns": ("entry_date", "announcement_date", "quote_date"),
            "required_for_target_window": True,
            "paper_grade_requirement": (
                "validated option candidate contracts for every retained event"
            ),
            "note": "Contract-level candidate table after DTE and reference validation gates.",
        },
        {
            "dataset_id": "silver_trade_proxy_prices",
            "layer": "silver",
            "path": config.silver_data_dir / "trade_proxy" / "trade_proxy_option_prices.parquet",
            "date_columns": ("entry_date", "quote_date", "timestamp_et", "event_entry_timestamp"),
            "required_for_target_window": False,
            "paper_grade_requirement": (
                "diagnostic no-NBBO trade-proxy entry marks, not quote execution"
            ),
            "note": "Useful for proxy research, insufficient for bid/ask or NBBO claims.",
        },
        {
            "dataset_id": "silver_quote_window_marks",
            "layer": "silver",
            "path": config.silver_data_dir / "quote_execution" / "quote_window_marks.parquet",
            "date_columns": ("quote_date", "quote_timestamp_et", "window_start"),
            "required_for_target_window": True,
            "paper_grade_requirement": "selected bid/ask quote marks for every event leg/window",
            "note": "Bounded quote slice should be nonzero; full sample remains the target.",
        },
        {
            "dataset_id": "silver_quote_execution_legs",
            "layer": "silver",
            "path": config.silver_data_dir / "quote_execution" / "quote_execution_legs.parquet",
            "date_columns": ("quote_date", "quote_timestamp_et", "window_start"),
            "required_for_target_window": True,
            "paper_grade_requirement": "leg-level bid/ask execution diagnostics",
            "note": "Bounded quote slice should be nonzero; full sample remains the target.",
        },
        {
            "dataset_id": "gold_trade_proxy_event_panel",
            "layer": "gold",
            "path": config.gold_data_dir / "event_panel" / "trade_proxy_event_panel.parquet",
            "date_columns": ("entry_date", "announcement_date", "event_entry_timestamp"),
            "required_for_target_window": False,
            "paper_grade_requirement": (
                "no-NBBO proxy event panel; paper-grade execution requires quote/NBBO replacement"
            ),
            "note": "Current canonical modeling input is explicitly a no-NBBO trade proxy.",
        },
        {
            "dataset_id": "gold_feature_matrix",
            "layer": "gold",
            "path": config.gold_data_dir / "modeling" / "feature_matrix.parquet",
            "date_columns": ("entry_date", "announcement_date", "event_entry_timestamp"),
            "required_for_target_window": True,
            "paper_grade_requirement": "modeling features over the final target event sample",
            "note": "Analysis-only quote fields must remain excluded from model features.",
        },
        {
            "dataset_id": "gold_quote_straddle_execution",
            "layer": "gold",
            "path": config.gold_data_dir / "quote_execution" / "quote_straddle_execution.parquet",
            "date_columns": (
                "entry_quote_date",
                "exit_quote_date",
                "quote_date",
                "entry_date",
                "announcement_date",
                "exit_date",
            ),
            "required_for_target_window": True,
            "paper_grade_requirement": "straddle-level bid/ask execution diagnostics",
            "note": "Bounded quote slice should be nonzero; full sample remains the target.",
        },
        {
            "dataset_id": "gold_quote_ivar_event",
            "layer": "gold",
            "path": config.gold_data_dir / "quote_execution" / "quote_ivar_event.parquet",
            "date_columns": ("entry_date", "announcement_date", "quote_date"),
            "required_for_target_window": True,
            "paper_grade_requirement": (
                "quote-based event IVAR or explicitly diagnostic quote premium proxy"
            ),
            "note": "Current quote-IVAR remains a diagnostic premium-total-variance proxy.",
        },
        {
            "dataset_id": "gold_quote_iv_surface",
            "layer": "gold",
            "path": config.gold_data_dir / "quote_execution" / "quote_iv_surface.parquet",
            "date_columns": ("entry_date", "announcement_date", "expiration"),
            "required_for_target_window": True,
            "paper_grade_requirement": (
                "bounded quote-derived leg-level Black-Scholes IV diagnostics"
            ),
            "note": (
                "Bounded targeted diagnostic; not a complete historical NBBO-equivalent surface."
            ),
        },
        {
            "dataset_id": "gold_quote_iv_surface_summary",
            "layer": "gold",
            "path": config.gold_data_dir / "quote_execution" / "quote_iv_surface_summary.parquet",
            "date_columns": ("entry_date", "announcement_date", "expiration"),
            "required_for_target_window": True,
            "paper_grade_requirement": "call-put quote-IV surface-pair diagnostics",
            "note": (
                "Pairs entry-window quote IVs by event, expiry, and strike for bounded diagnostics."
            ),
        },
        {
            "dataset_id": "gold_quote_surface_ivar_event",
            "layer": "gold",
            "path": config.gold_data_dir / "quote_execution" / "quote_surface_ivar_event.parquet",
            "date_columns": ("entry_date", "announcement_date", "expiration_1", "expiration_2"),
            "required_for_target_window": True,
            "paper_grade_requirement": "event-IVAR extracted from quote-IV total variance pairs",
            "note": (
                "Surface-IVAR diagnostic over bounded targeted quotes; full historical coverage "
                "and NBBO validation remain required."
            ),
        },
        {
            "dataset_id": "gold_quote_execution_confidence",
            "layer": "gold",
            "path": config.gold_data_dir / "quote_execution" / "quote_execution_confidence.parquet",
            "date_columns": ("entry_date", "announcement_date", "quote_date"),
            "required_for_target_window": True,
            "paper_grade_requirement": (
                "event-level execution-confidence bands over the final target event sample"
            ),
            "note": "Used for quote-confidence stratified strategy and casebook diagnostics.",
        },
    ]
    rows: list[dict[str, object]] = []
    year_frames: list[pd.DataFrame] = []
    for spec in dataset_specs:
        row, years = _lake_dataset_row(
            dataset_id=str(spec["dataset_id"]),
            layer=str(spec["layer"]),
            path=cast(Path, spec["path"]),
            target_start=target_start,
            target_end=target_end,
            date_columns=cast(Sequence[str], spec["date_columns"]),
            required_for_target_window=bool(spec["required_for_target_window"]),
            paper_grade_requirement=str(spec["paper_grade_requirement"]),
            note=str(spec["note"]),
            partitioned_by_date=bool(spec.get("partitioned_by_date", False)),
            expected_weekday_partitions=cast(int | None, spec.get("expected_weekday_partitions")),
        )
        rows.append(row)
        if not years.empty:
            year_frames.append(years)

    coverage = pd.DataFrame(rows)
    year_coverage = (
        pd.concat(year_frames, ignore_index=True)
        if year_frames
        else pd.DataFrame(
            columns=[
                "dataset_id",
                "year",
                "available_rows_or_partitions",
                "expected_weekday_partitions",
                "coverage_share",
            ]
        )
    )
    out.mkdir(parents=True, exist_ok=True)
    coverage.to_csv(coverage_path, index=False)
    year_coverage.to_csv(year_path, index=False)
    required = coverage.loc[coverage["required_for_target_window"].astype(bool)].copy()
    complete_statuses = {"target_span_covered"}
    incomplete_required = required.loc[
        ~required["target_coverage_status"].astype(str).isin(complete_statuses)
    ]
    quote_rows = coverage.loc[coverage["dataset_id"].astype(str).str.contains("quote")]
    report = {
        "pipeline_params": params,
        "status": "ran",
        "ok": bool(incomplete_required.empty),
        "target_window": {"start": target_start.isoformat(), "end": target_end.isoformat()},
        "datasets": int(len(coverage)),
        "required_datasets": int(len(required)),
        "incomplete_required_datasets": int(len(incomplete_required)),
        "incomplete_required_dataset_ids": incomplete_required["dataset_id"].astype(str).tolist(),
        "coverage_status_counts": _frame_value_counts(coverage, "target_coverage_status"),
        "quote_dataset_rows": int(len(quote_rows)),
        "paper_grade_execution_ready": False,
        "paper_grade_execution_blocker": (
            "requires full-window quote/NBBO or equivalent coverage and quote-IVAR beyond "
            "the current bounded diagnostic slice"
        ),
        "outputs": {
            "lake_dataset_coverage": str(coverage_path),
            "lake_year_coverage": str(year_path),
        },
    }
    report_path.write_text(json.dumps(report, indent=2, default=str), encoding="utf-8")
    return DataPipelineStep(
        "lake-quality-audit",
        "ran",
        outputs,
        metadata={
            "ok": bool(report["ok"]),
            "incomplete_required_datasets": int(report["incomplete_required_datasets"]),
            "coverage_status_counts": report["coverage_status_counts"],
        },
    )


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

    if not refresh_bronze and reference_path.exists() and reference_path.stat().st_size > 0:
        try:
            reference_report = pd.read_parquet(reference_path)
            if "options_ticker" in reference_report.columns and not reference_report.empty:
                out.mkdir(parents=True, exist_ok=True)
                reference_report = reference_report.drop_duplicates(
                    "options_ticker", keep="last"
                ).copy()
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
                    reference_report["contract_reference_status"]
                    .astype(str)
                    .value_counts()
                    .to_dict()
                    if "contract_reference_status" in reference_report
                    else {}
                )
                non_standard_rows = int(
                    validated["contract_discovery_status"]
                    .astype(str)
                    .eq("non_standard_excluded")
                    .sum()
                )
                proxy_usable_rows = int(
                    validated.get(
                        "contract_reference_proxy_usable",
                        pd.Series(False, index=validated.index),
                    )
                    .fillna(False)
                    .astype(bool)
                    .sum()
                )
                manifest = {
                    "pipeline_params": params,
                    "status_counts": {str(key): int(value) for key, value in status_counts.items()},
                    "fetch_status_counts": {
                        str(key): int(value) for key, value in fetch_counts.items()
                    },
                    "candidate_rows": int(len(candidates)),
                    "candidate_rows_after_validation": int(len(validated)),
                    "reference_contracts_requested": int(len(reference_report)),
                    "validated_contracts": int(
                        reference_report["contract_reference_status"]
                        .astype(str)
                        .eq(REFERENCE_STATUS_VALIDATED)
                        .sum()
                    )
                    if "contract_reference_status" in reference_report
                    else 0,
                    "proxy_usable_contract_rows": proxy_usable_rows,
                    "non_standard_excluded_rows": non_standard_rows,
                    "reused_reference_report": True,
                    "outputs": [str(path) for path in outputs],
                }
                manifest_path.write_text(
                    json.dumps(manifest, indent=2, default=str), encoding="utf-8"
                )
                return DataPipelineStep(
                    "contract-reference-validation",
                    "ran",
                    outputs,
                    metadata={
                        "reference_contracts_requested": int(len(reference_report)),
                        "status_counts": manifest["status_counts"],
                        "fetch_status_counts": manifest["fetch_status_counts"],
                        "proxy_usable_contract_rows": proxy_usable_rows,
                        "non_standard_excluded_rows": non_standard_rows,
                        "reused_reference_report": True,
                    },
                )
        except Exception as exc:
            _progress(
                "contract-reference-validation: existing reference report could not be reused; "
                f"falling back to fetch ({safe_exception_text(exc)})"
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
                    "contract_reference_error": safe_exception_text(exc),
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
    proxy_usable_rows = int(
        validated.get(
            "contract_reference_proxy_usable",
            pd.Series(False, index=validated.index),
        )
        .fillna(False)
        .astype(bool)
        .sum()
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
        "proxy_usable_contract_rows": proxy_usable_rows,
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
            "proxy_usable_contract_rows": proxy_usable_rows,
            "non_standard_excluded_rows": non_standard_rows,
        },
    )


def _quote_execution_panel_step(
    config: ProjectConfig,
    *,
    out_root: Path,
    force: bool,
    dates: Sequence[date],
    max_events: int | None,
    metadata_only: bool,
    allow_all_dates: bool,
    chunksize: int,
    entry_lookback_seconds: int,
    exit_lookback_seconds: int,
    stale_seconds: int,
    wide_spread_threshold: float,
    aws_executable: str,
    quote_source: str,
    quote_cache_dir: Path | None,
    rest_limit: int,
    quote_workers: int,
    event_offset: int,
    batch_label: str | None,
) -> DataPipelineStep:
    contracts_path = config.silver_data_dir / "contracts" / "event_contract_candidates.parquet"
    windows_path = config.silver_data_dir / "event_windows" / "event_windows.parquet"
    normalized_batch_label = _normalize_quote_batch_label(batch_label)
    batch_mode = normalized_batch_label is not None
    out = out_root / "quote_execution_panel"
    if normalized_batch_label is not None:
        out = out / "batches" / normalized_batch_label
    report_path = out / "quote_execution_report.json"
    manifest_path = out / "quote_execution_panel_manifest.json"
    artifact_paths = {
        "quote_window_requests_csv": out / "quote_window_requests.csv",
        "quote_window_quotes_csv": out / "quote_window_quotes.csv",
        "quote_window_marks_csv": out / "quote_window_marks.csv",
        "quote_execution_legs_csv": out / "quote_execution_legs.csv",
        "quote_straddle_execution_csv": out / "quote_straddle_execution.csv",
        "quote_ivar_event_csv": out / "quote_ivar_event.csv",
        "quote_iv_surface_csv": out / "quote_iv_surface.csv",
        "quote_iv_surface_summary_csv": out / "quote_iv_surface_summary.csv",
        "quote_surface_ivar_event_csv": out / "quote_surface_ivar_event.csv",
        "quote_execution_confidence_csv": out / "quote_execution_confidence.csv",
        "quote_execution_report": report_path,
    }
    bronze_quote_root = config.bronze_data_dir / "massive" / "quotes_v1_target_windows"
    silver_quote_root = config.silver_data_dir / "quote_execution"
    gold_quote_root = config.gold_data_dir / "quote_execution"
    if normalized_batch_label is not None:
        batch_partition = f"batch={normalized_batch_label}"
        bronze_quote_root = bronze_quote_root / "batches" / batch_partition
        silver_quote_root = silver_quote_root / "batches" / batch_partition
        gold_quote_root = gold_quote_root / "batches" / batch_partition
    lake_paths = {
        "bronze_quote_window_requests": bronze_quote_root / "quote_window_requests.parquet",
        "bronze_quote_window_quotes": bronze_quote_root / "quote_window_quotes.parquet",
        "silver_quote_window_marks": silver_quote_root / "quote_window_marks.parquet",
        "silver_quote_execution_legs": silver_quote_root / "quote_execution_legs.parquet",
        "gold_quote_straddle_execution": gold_quote_root / "quote_straddle_execution.parquet",
        "gold_quote_ivar_event": gold_quote_root / "quote_ivar_event.parquet",
        "gold_quote_iv_surface": gold_quote_root / "quote_iv_surface.parquet",
        "gold_quote_iv_surface_summary": gold_quote_root / "quote_iv_surface_summary.parquet",
        "gold_quote_surface_ivar_event": gold_quote_root / "quote_surface_ivar_event.parquet",
        "gold_quote_execution_confidence": gold_quote_root / "quote_execution_confidence.parquet",
    }
    outputs = (
        contracts_path,
        windows_path,
        *artifact_paths.values(),
        *lake_paths.values(),
        manifest_path,
    )
    if not contracts_path.exists() or not windows_path.exists():
        return DataPipelineStep(
            "quote-execution-panel",
            "blocked",
            outputs,
            reason="requires event_contract_candidates.parquet and event_windows.parquet",
        )
    if not metadata_only and not dates and not allow_all_dates:
        out.mkdir(parents=True, exist_ok=True)
        blocked_params: dict[str, object] = {
            "stage": "quote-execution-panel",
            "contracts": _path_signature(contracts_path),
            "windows": _path_signature(windows_path),
            "dates": [],
            "max_events": max_events,
            "event_offset": event_offset,
            "metadata_only": metadata_only,
            "allow_all_dates": allow_all_dates,
            "quote_source": quote_source,
            "quote_workers": quote_workers,
            "quote_batch_label": normalized_batch_label,
        }
        manifest_path.write_text(
            json.dumps(
                {
                    "pipeline_params": blocked_params,
                    "status": "blocked",
                    "reason": "requires_quote_dates_or_allow_all_dates_for_quote_stream",
                    "raw_full_day_files_written": False,
                },
                indent=2,
                default=str,
            ),
            encoding="utf-8",
        )
        return DataPipelineStep(
            "quote-execution-panel",
            "blocked",
            outputs,
            reason="requires_quote_dates_or_allow_all_dates_for_quote_stream",
        )

    normalized_dates = tuple(sorted(dates))
    normalized_cache_dir = (
        quote_cache_dir
        if quote_cache_dir is not None
        else config.bronze_data_dir / "massive" / "quotes_v3_rest_target_windows" / "cache"
    )
    params: dict[str, object] = {
        "stage": "quote-execution-panel",
        "contracts": _path_signature(contracts_path),
        "windows": _path_signature(windows_path),
        "dates": [value.isoformat() for value in normalized_dates],
        "max_events": max_events,
        "event_offset": event_offset,
        "metadata_only": metadata_only,
        "allow_all_dates": allow_all_dates,
        "chunksize": chunksize,
        "entry_lookback_seconds": entry_lookback_seconds,
        "exit_lookback_seconds": exit_lookback_seconds,
        "stale_seconds": stale_seconds,
        "wide_spread_threshold": wide_spread_threshold,
        "quote_source": quote_source,
        "quote_cache_dir": str(normalized_cache_dir),
        "rest_limit": rest_limit,
        "quote_workers": quote_workers,
        "quote_batch_label": normalized_batch_label,
    }
    if not force and _complete_with_params(
        outputs,
        params_path=manifest_path,
        expected_params=params,
    ):
        return DataPipelineStep(
            "quote-execution-panel",
            "skipped",
            outputs,
            reason="outputs_exist_params_match",
        )

    try:
        contracts = pd.read_parquet(contracts_path)
        windows = pd.read_parquet(windows_path)
        report = extract_quote_execution_panel(
            config=config,
            contracts=contracts,
            windows=windows,
            out_dir=out,
            dates=normalized_dates,
            metadata_only=metadata_only,
            chunksize=chunksize,
            entry_lookback_seconds=entry_lookback_seconds,
            exit_lookback_seconds=exit_lookback_seconds,
            stale_seconds=stale_seconds,
            wide_spread_threshold=wide_spread_threshold,
            max_events=max_events,
            aws_executable=aws_executable,
            quote_source=quote_source,
            quote_cache_dir=normalized_cache_dir,
            rest_limit=rest_limit,
            quote_workers=quote_workers,
            event_offset=event_offset,
            batch_label=normalized_batch_label,
        )
        row_counts = {
            "bronze_quote_window_requests": _write_parquet_from_csv(
                artifact_paths["quote_window_requests_csv"],
                lake_paths["bronze_quote_window_requests"],
            ),
            "bronze_quote_window_quotes": _write_parquet_from_csv(
                artifact_paths["quote_window_quotes_csv"],
                lake_paths["bronze_quote_window_quotes"],
            ),
            "silver_quote_window_marks": _write_parquet_from_csv(
                artifact_paths["quote_window_marks_csv"],
                lake_paths["silver_quote_window_marks"],
            ),
            "silver_quote_execution_legs": _write_parquet_from_csv(
                artifact_paths["quote_execution_legs_csv"],
                lake_paths["silver_quote_execution_legs"],
            ),
            "gold_quote_straddle_execution": _write_parquet_from_csv(
                artifact_paths["quote_straddle_execution_csv"],
                lake_paths["gold_quote_straddle_execution"],
            ),
            "gold_quote_ivar_event": _write_parquet_from_csv(
                artifact_paths["quote_ivar_event_csv"],
                lake_paths["gold_quote_ivar_event"],
            ),
            "gold_quote_iv_surface": _write_parquet_from_csv(
                artifact_paths["quote_iv_surface_csv"],
                lake_paths["gold_quote_iv_surface"],
            ),
            "gold_quote_iv_surface_summary": _write_parquet_from_csv(
                artifact_paths["quote_iv_surface_summary_csv"],
                lake_paths["gold_quote_iv_surface_summary"],
            ),
            "gold_quote_surface_ivar_event": _write_parquet_from_csv(
                artifact_paths["quote_surface_ivar_event_csv"],
                lake_paths["gold_quote_surface_ivar_event"],
            ),
            "gold_quote_execution_confidence": _write_parquet_from_csv(
                artifact_paths["quote_execution_confidence_csv"],
                lake_paths["gold_quote_execution_confidence"],
            ),
        }
    except Exception as exc:
        out.mkdir(parents=True, exist_ok=True)
        manifest_path.write_text(
            json.dumps(
                {
                    "pipeline_params": params,
                    "status": "blocked",
                    "reason": "quote_execution_panel_failed",
                    "error": safe_exception_text(exc),
                    "raw_full_day_files_written": False,
                },
                indent=2,
                default=str,
            ),
            encoding="utf-8",
        )
        return DataPipelineStep(
            "quote-execution-panel",
            "blocked",
            outputs,
            reason="quote_execution_panel_failed",
            metadata={"error": safe_exception_text(exc)},
        )

    manifest = {
        "pipeline_params": params,
        "status": "ran",
        "report": report.as_dict(),
        "lake_output_rows": row_counts,
        "artifact_outputs": {key: str(path) for key, path in artifact_paths.items()},
        "lake_outputs": {key: str(path) for key, path in lake_paths.items()},
        "lake_policy": {
            "bronze": "target-window request table plus matched normalized quote subset only",
            "silver": "selected quote marks and leg-level bid/ask execution diagnostics",
            "gold": (
                "event/straddle execution diagnostics, diagnostic quote-IVAR proxy, "
                "bounded quote-IV surface diagnostics, and execution confidence"
            ),
            "quote_source": quote_source,
            "rest_cache": str(normalized_cache_dir),
            "rest_workers": quote_workers,
            "batch_mode": batch_mode,
            "batch_label": normalized_batch_label,
            "canonical_outputs_updated": not batch_mode,
            "raw_full_day_quote_files_in_repo": False,
        },
        "raw_full_day_files_written": report.raw_full_day_files_written,
    }
    manifest_path.write_text(json.dumps(manifest, indent=2, default=str), encoding="utf-8")
    complete = _complete(outputs)
    return DataPipelineStep(
        "quote-execution-panel",
        "ran" if report.ok and complete else "blocked",
        outputs,
        reason=None if report.ok and complete else "quote_execution_outputs_missing",
        metadata={
            "route": report.route,
            "metadata_only": report.metadata_only,
            "request_rows": report.request_rows,
            "event_count": report.event_count,
            "quote_rows_scanned": report.quote_rows_scanned,
            "quote_rows_matched": report.quote_rows_matched,
            "event_offset": event_offset,
            "quote_batch_label": normalized_batch_label,
            "lake_output_rows": row_counts,
        },
    )


def _discover_quote_batch_labels(config: ProjectConfig, out_root: Path) -> list[str]:
    roots = [
        config.bronze_data_dir / "massive" / "quotes_v1_target_windows" / "batches",
        config.silver_data_dir / "quote_execution" / "batches",
        config.gold_data_dir / "quote_execution" / "batches",
        out_root / "quote_execution_panel" / "batches",
    ]
    labels: set[str] = set()
    for root in roots:
        if not root.exists():
            continue
        for child in root.iterdir():
            if not child.is_dir():
                continue
            label = (
                child.name.removeprefix("batch=") if child.name.startswith("batch=") else child.name
            )
            normalized = _normalize_quote_batch_label(label)
            if normalized is not None:
                labels.add(normalized)
    return sorted(labels)


def _dedupe_quote_execution_frame(dataset_id: str, frame: pd.DataFrame) -> pd.DataFrame:
    if frame.empty:
        return frame.reset_index(drop=True)
    keys = [key for key in QUOTE_EXECUTION_DEDUPE_KEYS[dataset_id] if key in frame.columns]
    if keys:
        return frame.drop_duplicates(keys, keep="last").reset_index(drop=True)
    return frame.drop_duplicates(keep="last").reset_index(drop=True)


def _merge_quote_execution_sources(
    *,
    config: ProjectConfig,
    out_root: Path,
    batch_labels: Sequence[str],
    include_canonical: bool,
) -> tuple[dict[str, pd.DataFrame], dict[str, list[Path]], dict[str, dict[str, int]]]:
    source_paths: dict[str, list[Path]] = {
        dataset_id: [] for dataset_id in QUOTE_EXECUTION_DATASET_FILES
    }
    source_rows: dict[str, dict[str, int]] = {
        dataset_id: {} for dataset_id in QUOTE_EXECUTION_DATASET_FILES
    }
    merged: dict[str, pd.DataFrame] = {}
    canonical_paths = _quote_execution_lake_paths(config)
    batch_path_sets = {
        label: _quote_execution_lake_paths(config, batch_label=label) for label in batch_labels
    }
    for dataset_id in QUOTE_EXECUTION_DATASET_FILES:
        frames: list[pd.DataFrame] = []
        if include_canonical and canonical_paths[dataset_id].exists():
            path = canonical_paths[dataset_id]
            frame = pd.read_parquet(path)
            frames.append(frame)
            source_paths[dataset_id].append(path)
            source_rows[dataset_id]["canonical"] = int(len(frame))
        for label in batch_labels:
            path = batch_path_sets[label][dataset_id]
            if not path.exists():
                continue
            frame = pd.read_parquet(path)
            frames.append(frame)
            source_paths[dataset_id].append(path)
            source_rows[dataset_id][label] = int(len(frame))
        if frames:
            merged[dataset_id] = _dedupe_quote_execution_frame(
                dataset_id, pd.concat(frames, ignore_index=True, sort=False)
            )
        else:
            merged[dataset_id] = pd.DataFrame()
    return merged, source_paths, source_rows


def _quote_execution_merge_step(
    config: ProjectConfig,
    *,
    out_root: Path,
    force: bool,
    batch_labels: Sequence[str],
    include_canonical: bool,
) -> DataPipelineStep:
    normalized_labels = sorted(
        {
            label
            for label in (_normalize_quote_batch_label(value) for value in batch_labels)
            if label is not None
        }
    )
    if not normalized_labels:
        normalized_labels = _discover_quote_batch_labels(config, out_root)
    artifact_root = _quote_execution_artifact_root(out_root)
    manifest_path = artifact_root / "quote_execution_panel_manifest.json"
    report_path = artifact_root / "quote_execution_report.json"
    lake_paths = _quote_execution_lake_paths(config)
    artifact_csv_paths = _quote_execution_artifact_paths(out_root)
    outputs = (*lake_paths.values(), *artifact_csv_paths.values(), report_path, manifest_path)
    if not normalized_labels and not include_canonical:
        return DataPipelineStep(
            "quote-execution-merge",
            "blocked",
            outputs,
            reason="requires_quote_batch_labels_or_canonical_input",
        )
    merged, source_paths, source_rows = _merge_quote_execution_sources(
        config=config,
        out_root=out_root,
        batch_labels=normalized_labels,
        include_canonical=include_canonical,
    )
    input_signatures = {
        dataset_id: [_path_signature(path) for path in paths]
        for dataset_id, paths in source_paths.items()
    }
    params: dict[str, object] = {
        "stage": "quote-execution-merge",
        "batch_labels": normalized_labels,
        "include_canonical": include_canonical,
        "input_signatures": input_signatures,
    }
    has_any_input = any(paths for paths in source_paths.values())
    if not has_any_input:
        artifact_root.mkdir(parents=True, exist_ok=True)
        manifest_path.write_text(
            json.dumps(
                {
                    "pipeline_params": params,
                    "status": "blocked",
                    "reason": "requires_existing_quote_execution_lake_sources",
                    "raw_full_day_files_written": False,
                },
                indent=2,
                default=str,
            ),
            encoding="utf-8",
        )
        return DataPipelineStep(
            "quote-execution-merge",
            "blocked",
            outputs,
            reason="requires_existing_quote_execution_lake_sources",
        )
    if not force and _complete_with_params(
        outputs,
        params_path=manifest_path,
        expected_params=params,
    ):
        return DataPipelineStep(
            "quote-execution-merge",
            "skipped",
            outputs,
            reason="outputs_exist_params_match",
        )

    row_counts: dict[str, int] = {}
    artifact_root.mkdir(parents=True, exist_ok=True)
    for dataset_id, frame in merged.items():
        _write_parquet(lake_paths[dataset_id], frame)
        artifact_csv_paths[dataset_id].parent.mkdir(parents=True, exist_ok=True)
        frame.to_csv(artifact_csv_paths[dataset_id], index=False)
        row_counts[dataset_id] = int(len(frame))
    final_input_signatures = {
        dataset_id: [_path_signature(path) for path in paths]
        for dataset_id, paths in source_paths.items()
    }
    final_params = {**params, "input_signatures": final_input_signatures}

    confidence = merged["gold_quote_execution_confidence"]
    event_count = (
        int(confidence["event_id"].astype(str).nunique())
        if not confidence.empty and "event_id" in confidence.columns
        else 0
    )
    report = {
        "ok": True,
        "route": "quote_batch_consolidation",
        "metadata_only": False,
        "event_count": event_count,
        "request_rows": row_counts["bronze_quote_window_requests"],
        "quote_rows_scanned": 0,
        "quote_rows_matched": row_counts["bronze_quote_window_quotes"],
        "raw_full_day_files_written": False,
        "dates": [],
        "batch_labels": normalized_labels,
        "include_canonical": include_canonical,
        "output_paths": {key: str(path) for key, path in artifact_csv_paths.items()},
    }
    report_path.write_text(json.dumps(report, indent=2, default=str), encoding="utf-8")
    manifest = {
        "pipeline_params": final_params,
        "status": "ran",
        "report": report,
        "lake_output_rows": row_counts,
        "source_rows": source_rows,
        "artifact_outputs": {key: str(path) for key, path in artifact_csv_paths.items()},
        "lake_outputs": {key: str(path) for key, path in lake_paths.items()},
        "lake_policy": {
            "batch_consolidation": True,
            "batch_labels": normalized_labels,
            "canonical_outputs_updated": True,
            "raw_full_day_quote_files_in_repo": False,
        },
        "raw_full_day_files_written": False,
    }
    manifest_path.write_text(json.dumps(manifest, indent=2, default=str), encoding="utf-8")
    complete = _complete(outputs)
    return DataPipelineStep(
        "quote-execution-merge",
        "ran" if complete else "blocked",
        outputs,
        reason=None if complete else "quote_execution_merge_outputs_missing",
        metadata={
            "batch_labels": normalized_labels,
            "include_canonical": include_canonical,
            "event_count": event_count,
            "lake_output_rows": row_counts,
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
        "rest_limit": DEFAULT_TRADE_PROXY_REST_LIMIT,
        "haircut_fraction": DEFAULT_TRADE_PROXY_HAIRCUT_FRACTION,
        "entry_price_method": ENTRY_PRICE_METHOD_PRECLOSE_WINDOW_VWAP,
        "c2c_exit_price_method": EXIT_PRECLOSE_OPTION_VWAP_SOURCE,
        "post_open_option_vwap_windows": [window[0] for window in POST_OPEN_OPTION_VWAP_WINDOWS],
    }
    outputs_complete = _complete(outputs)
    if (
        not force
        and not refresh_bronze
        and _complete_with_params(
            outputs,
            params_path=outputs[1],
            expected_params=params,
        )
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
        "--rest-limit",
        str(DEFAULT_TRADE_PROXY_REST_LIMIT),
        "--haircut-fraction",
        str(DEFAULT_TRADE_PROXY_HAIRCUT_FRACTION),
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
    quote_metadata_only: bool,
    quote_source: str,
    quote_rest_limit: int,
    quote_workers: int,
    quote_event_offset: int,
    quote_batch_label: str | None,
    quote_merge_batch_labels: Sequence[str],
    quote_merge_include_canonical: bool,
) -> dict[str, object]:
    normalized_quote_batch_label = _normalize_quote_batch_label(quote_batch_label)
    normalized_tickers = sorted({ticker.upper() for ticker in tickers if ticker.strip()})
    lookback_start = _universe_lookback_start(start_date, universe_trailing_months)
    month_count = (end_date.year - start_date.year) * 12 + end_date.month - start_date.month + 1
    study_calendar_days = (end_date - start_date).days + 1
    bulk_calendar_days = (end_date - lookback_start).days + 1
    estimated_trading_days = int(round(bulk_calendar_days * 252 / 365))
    estimated_years = max(study_calendar_days / 365.25, 1 / 365.25)
    dynamic_ticker_months = month_count * universe_top_n
    estimated_unique_dynamic_tickers = max(universe_top_n, min(dynamic_ticker_months, 250))
    active_proxy_dag = stage == "all"
    ticker_count = estimated_unique_dynamic_tickers if active_proxy_dag else len(normalized_tickers)
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
        "planned_stages": list(ACTIVE_PROXY_DATA_DAG) if active_proxy_dag else [stage],
        "estimated_counts": {
            "study_calendar_days": study_calendar_days,
            "bulk_calendar_days": bulk_calendar_days,
            "trading_days": estimated_trading_days,
            "months": month_count,
            "universe_ticker_months": dynamic_ticker_months if active_proxy_dag else None,
            "tickers": ticker_count,
            "events": estimated_events,
            "contracts": estimated_contracts,
            "contract_reference_rest_calls": estimated_contracts,
            "second_agg_rest_calls": estimated_contracts,
            "quote_window_requests": estimated_contracts * 2,
            "quote_rest_window_calls": (
                0
                if quote_metadata_only or quote_source != QUOTE_SOURCE_REST
                else estimated_contracts * 2
            ),
            "quote_rest_workers": (
                0 if quote_metadata_only or quote_source != QUOTE_SOURCE_REST else quote_workers
            ),
            "quote_flat_file_full_day_scans": (
                0
                if quote_metadata_only or quote_source != QUOTE_SOURCE_FLAT_FILE
                else "requires_dates"
            ),
            "bulk_day_agg_partitions": estimated_trading_days * 2 if active_proxy_dag else None,
        },
        "parameters": {
            "dte_min": dte_min,
            "dte_max": dte_max,
            "lookback_seconds": lookback_seconds,
            "second_agg_buffer_minutes": second_agg_buffer_minutes,
            "price_field": price_field,
            "universe_top_n": universe_top_n,
            "universe_trailing_months": universe_trailing_months,
            "quote_metadata_only": quote_metadata_only,
            "quote_source": quote_source,
            "quote_rest_limit": quote_rest_limit,
            "quote_workers": quote_workers,
            "quote_event_offset": quote_event_offset,
            "quote_batch_label": normalized_quote_batch_label,
            "quote_batch_mode": normalized_quote_batch_label is not None,
            "quote_merge_batch_labels": list(quote_merge_batch_labels),
            "quote_merge_include_canonical": quote_merge_include_canonical,
            "quote_ingestion_policy": "targeted_windows_no_full_day_raw_files",
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
    start_date: date = TARGET_WINDOW_START,
    end_date: date = TARGET_WINDOW_END,
    dates: Sequence[date] = (),
    options_day_aggs_path: Path | None = None,
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
    quote_dates: Sequence[date] = (),
    quote_metadata_only: bool = True,
    quote_allow_all_dates: bool = False,
    quote_chunksize: int = 250_000,
    quote_entry_lookback_seconds: int = 900,
    quote_exit_lookback_seconds: int = 900,
    quote_stale_seconds: int = 60,
    quote_wide_spread_threshold: float = 0.25,
    quote_aws_executable: str = "aws",
    quote_source: str = QUOTE_SOURCE_REST,
    quote_cache_dir: Path | None = None,
    quote_rest_limit: int = 50_000,
    quote_workers: int = 1,
    quote_event_offset: int = 0,
    quote_batch_label: str | None = None,
    quote_merge_batch_labels: Sequence[str] = (),
    quote_merge_include_canonical: bool = True,
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
    if quote_chunksize <= 0:
        raise ValueError("quote_chunksize must be positive.")
    if quote_entry_lookback_seconds <= 0:
        raise ValueError("quote_entry_lookback_seconds must be positive.")
    if quote_exit_lookback_seconds <= 0:
        raise ValueError("quote_exit_lookback_seconds must be positive.")
    if quote_stale_seconds <= 0:
        raise ValueError("quote_stale_seconds must be positive.")
    if quote_wide_spread_threshold <= 0:
        raise ValueError("quote_wide_spread_threshold must be positive.")
    if quote_source not in {QUOTE_SOURCE_REST, QUOTE_SOURCE_FLAT_FILE}:
        raise ValueError(f"unsupported quote_source: {quote_source}")
    if quote_rest_limit <= 0:
        raise ValueError("quote_rest_limit must be positive.")
    if quote_workers <= 0:
        raise ValueError("quote_workers must be positive.")
    if quote_event_offset < 0:
        raise ValueError("quote_event_offset must be non-negative.")
    normalized_quote_batch_label = _normalize_quote_batch_label(quote_batch_label)
    normalized_quote_merge_batch_labels = [
        label
        for label in (_normalize_quote_batch_label(value) for value in quote_merge_batch_labels)
        if label is not None
    ]
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
            quote_metadata_only=quote_metadata_only,
            quote_source=quote_source,
            quote_rest_limit=quote_rest_limit,
            quote_workers=quote_workers,
            quote_event_offset=quote_event_offset,
            quote_batch_label=normalized_quote_batch_label,
            quote_merge_batch_labels=normalized_quote_merge_batch_labels,
            quote_merge_include_canonical=quote_merge_include_canonical,
        )

    active_proxy_dag = normalized_stage == "all"
    stages = list(ACTIVE_PROXY_DATA_DAG) if active_proxy_dag else [normalized_stage]
    normalized_tickers = sorted({ticker.upper() for ticker in tickers if ticker.strip()})
    if not normalized_tickers:
        normalized_tickers = list(DEFAULT_STATIC_TICKERS)
    normalized_quote_dates = tuple(quote_dates or dates)

    stage_force = force
    step_builders: dict[str, Callable[[], list[DataPipelineStep]]] = {
        "fixture-audit": lambda: [
            _fixture_audit_step(config, out_root=out_root, force=stage_force)
        ],
        "lake-quality-audit": lambda: [
            _lake_quality_audit_step(
                config,
                out_root=out_root,
                force=stage_force,
                target_start=start_date,
                target_end=end_date,
            )
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
        "sec-companyfacts": lambda: [
            _sec_companyfacts_step(config, out_root=out_root, force=stage_force)
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
                    if normalized_stage in {"all", "event-window-panel"}
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
        "quote-execution-panel": lambda: [
            _quote_execution_panel_step(
                config,
                out_root=out_root,
                force=stage_force,
                dates=normalized_quote_dates,
                max_events=max_events,
                metadata_only=quote_metadata_only,
                allow_all_dates=quote_allow_all_dates,
                chunksize=quote_chunksize,
                entry_lookback_seconds=quote_entry_lookback_seconds,
                exit_lookback_seconds=quote_exit_lookback_seconds,
                stale_seconds=quote_stale_seconds,
                wide_spread_threshold=quote_wide_spread_threshold,
                aws_executable=quote_aws_executable,
                quote_source=quote_source,
                quote_cache_dir=quote_cache_dir,
                rest_limit=quote_rest_limit,
                quote_workers=quote_workers,
                event_offset=quote_event_offset,
                batch_label=normalized_quote_batch_label,
            )
        ],
        "quote-execution-merge": lambda: [
            _quote_execution_merge_step(
                config,
                out_root=out_root,
                force=stage_force,
                batch_labels=normalized_quote_merge_batch_labels,
                include_canonical=quote_merge_include_canonical,
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

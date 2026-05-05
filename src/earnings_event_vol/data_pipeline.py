from __future__ import annotations

import json
import subprocess
import sys
import time
from collections import deque
from collections.abc import Callable, Sequence
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from datetime import date
from pathlib import Path

import pandas as pd
import polars as pl

from earnings_event_vol.config import ProjectConfig
from earnings_event_vol.data_audit import audit_data_fields
from earnings_event_vol.earnings_calendar import build_earnings_calendar_candidates
from earnings_event_vol.event_panel import build_event_panel, discover_option_contracts
from earnings_event_vol.massive import build_massive_day_agg_sample, massive_flat_file_manifest
from earnings_event_vol.universe import (
    build_monthly_liquid_universe,
    build_ticker_month_liquidity,
)

DEFAULT_PILOT_TICKERS: tuple[str, ...] = (
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
    "universe",
    "calendar-pilot",
    "contracts",
    "panel",
    "pilot-panel",
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
    outputs = (out / "ticker_month_liquidity.parquet", out / "monthly_top50_universe.parquet")
    if options_day_aggs_path is None:
        return DataPipelineStep(
            "universe",
            "blocked",
            outputs,
            reason="requires --options-day-aggs pointing to normalized option day aggregates",
        )
    if not force and _complete(outputs):
        return DataPipelineStep("universe", "skipped", outputs, reason="outputs_exist")

    option_day_aggs = _read_universe_source(options_day_aggs_path)
    liquidity = build_ticker_month_liquidity(
        option_day_aggs,
        source_snapshot_date=end_date,
    )
    universe = build_monthly_liquid_universe(
        liquidity,
        start_month=start_date,
        end_month=end_date,
        top_n=top_n,
        trailing_months=trailing_months,
    )
    _write_parquet(outputs[0], liquidity)
    _write_parquet(outputs[1], universe)
    return DataPipelineStep(
        "universe",
        "ran",
        outputs,
        metadata={
            "ticker_month_rows": int(len(liquidity)),
            "universe_rows": int(len(universe)),
            "top_n": top_n,
            "trailing_months": trailing_months,
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


def _calendar_pilot_step(
    config: ProjectConfig,
    *,
    out_root: Path,
    tickers: Sequence[str],
    start_date: date,
    end_date: date,
    sec_submissions_dir: Path | None,
    massive_8k_text_dir: Path | None,
    validate_with_massive: bool,
    force: bool,
) -> DataPipelineStep:
    out = out_root / "earnings_calendar_pilot"
    outputs = (out / "earnings_calendar_candidates.csv", out / "earnings_calendar_report.json")
    if not force and _complete(outputs):
        return DataPipelineStep("calendar-pilot", "skipped", outputs, reason="outputs_exist")

    frame, report = build_earnings_calendar_candidates(
        config=config,
        tickers=tickers,
        start_date=start_date,
        end_date=end_date,
        sec_submissions_dir=sec_submissions_dir,
        massive_8k_text_dir=massive_8k_text_dir,
        validate_with_massive=validate_with_massive,
    )
    out.mkdir(parents=True, exist_ok=True)
    frame.to_csv(outputs[0], index=False)
    outputs[1].write_text(json.dumps(report, indent=2, default=str), encoding="utf-8")
    return DataPipelineStep(
        "calendar-pilot",
        "ran",
        outputs,
        metadata={
            "rows": int(report["row_count"]),
            "main_sample_candidate_rows": int(report["main_sample_candidate_rows"]),
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


def _pilot_panel_step(
    config: ProjectConfig,
    *,
    out_root: Path,
    force: bool,
    dte_min: int,
    dte_max: int,
    max_events: int | None,
) -> DataPipelineStep:
    outputs = (
        config.gold_data_dir / "event_panel" / "pilot_event_panel.parquet",
        out_root / "event_panel" / "pilot_panel_report.json",
    )
    if not force and _complete(outputs):
        return DataPipelineStep("pilot-panel", "skipped", outputs, reason="outputs_exist")
    command = [
        sys.executable,
        str(config.repo_root / "scripts" / "build_pilot_panel.py"),
        "--out-root",
        str(out_root),
        "--dte-min",
        str(dte_min),
        "--dte-max",
        str(dte_max),
    ]
    if force:
        command.append("--force")
    if max_events is not None:
        command.extend(["--max-events", str(max_events)])
    result = _run_command_with_progress(
        command,
        cwd=config.repo_root,
        label="pilot-panel",
    )
    metadata: dict[str, object] = {"returncode": result.returncode}
    if result.stdout.strip():
        metadata["stdout_tail"] = result.stdout[-1000:]
    if result.stderr.strip():
        metadata["stderr_tail"] = result.stderr[-1000:]
    return DataPipelineStep(
        "pilot-panel",
        "ran" if result.returncode == 0 and _complete(outputs) else "blocked",
        outputs,
        reason=None if result.returncode == 0 else "pilot_panel_script_failed",
        metadata=metadata,
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
) -> DataPipelineStep:
    outputs = (
        config.gold_data_dir / "event_panel" / "trade_proxy_event_panel.parquet",
        out_root / "trade_proxy_panel" / "trade_proxy_panel_report.json",
    )
    if not force and _complete(outputs):
        return DataPipelineStep("trade-proxy-panel", "skipped", outputs, reason="outputs_exist")
    command = [
        sys.executable,
        str(config.repo_root / "scripts" / "build_trade_proxy_panel.py"),
        "--out-root",
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
    if force:
        command.append("--force")
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
    month_count = (end_date.year - start_date.year) * 12 + end_date.month - start_date.month + 1
    calendar_days = (end_date - start_date).days + 1
    estimated_trading_days = int(round(calendar_days * 252 / 365))
    estimated_events = max_events if max_events is not None else len(normalized_tickers) * 4
    estimated_contracts = max_contracts if max_contracts is not None else int(estimated_events * 18)
    exclusions = {
        "non_bmo_amc_timing": "estimated_after_calendar_stage",
        "missing_sec_or_massive_validation": "estimated_after_calendar_stage",
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
        "out_root": str(out_root),
        "date_range": {"start": start_date.isoformat(), "end": end_date.isoformat()},
        "estimated_counts": {
            "calendar_days": calendar_days,
            "trading_days": estimated_trading_days,
            "months": month_count,
            "tickers": len(normalized_tickers),
            "events": estimated_events,
            "contracts": estimated_contracts,
            "second_agg_rest_calls": estimated_contracts,
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
    tickers: Sequence[str] = DEFAULT_PILOT_TICKERS,
    start_date: date = date(2025, 1, 1),
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
            "universe",
            "calendar-pilot",
            "contracts",
            "panel",
            "pilot-panel",
        ]
    elif normalized_stage == "proxy-all":
        stages = ["calendar-pilot", "pilot-panel", "trade-proxy-panel"]
    else:
        stages = [normalized_stage]
    normalized_tickers = sorted({ticker.upper() for ticker in tickers if ticker.strip()})
    if not normalized_tickers:
        normalized_tickers = list(DEFAULT_PILOT_TICKERS)

    step_builders: dict[str, Callable[[], list[DataPipelineStep]]] = {
        "fixture-audit": lambda: [_fixture_audit_step(config, out_root=out_root, force=force)],
        "massive-probe": lambda: _massive_probe_steps(
            config,
            out_root=out_root,
            dates=dates,
            force=force,
            jobs=jobs,
            download_samples=download_samples,
        ),
        "universe": lambda: [
            _universe_step(
                out_root=out_root,
                options_day_aggs_path=options_day_aggs_path,
                start_date=start_date,
                end_date=end_date,
                top_n=universe_top_n,
                trailing_months=universe_trailing_months,
                force=force,
            )
        ],
        "calendar-pilot": lambda: [
            _calendar_pilot_step(
                config,
                out_root=out_root,
                tickers=normalized_tickers,
                start_date=start_date,
                end_date=end_date,
                sec_submissions_dir=sec_submissions_dir,
                massive_8k_text_dir=massive_8k_text_dir,
                validate_with_massive=validate_with_massive,
                force=force,
            )
        ],
        "contracts": lambda: [
            _contracts_step(
                out_root=out_root,
                events_path=events_path,
                contracts_path=contracts_path,
                dte_min=dte_min,
                dte_max=dte_max,
                force=force,
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
                force=force,
            )
        ],
        "pilot-panel": lambda: [
            _pilot_panel_step(
                config,
                out_root=out_root,
                force=force,
                dte_min=dte_min,
                dte_max=dte_max,
                max_events=max_events,
            )
        ],
        "trade-proxy-panel": lambda: [
            _trade_proxy_panel_step(
                config,
                out_root=out_root,
                force=force,
                max_events=max_events,
                max_contracts=max_contracts,
                jobs=jobs,
                lookback_seconds=lookback_seconds,
                second_agg_buffer_minutes=second_agg_buffer_minutes,
                price_field=price_field,
            )
        ],
    }

    steps: list[DataPipelineStep] = []
    for index, selected_stage in enumerate(stages, start=1):
        started_at = time.perf_counter()
        _progress(f"stage {index}/{len(stages)} start: {selected_stage}")
        stage_steps = step_builders[selected_stage]()
        steps.extend(stage_steps)
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
        "out_root": str(out_root),
        "steps": [step.as_dict() for step in steps],
    }

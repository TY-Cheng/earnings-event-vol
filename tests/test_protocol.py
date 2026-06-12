from __future__ import annotations

import argparse
import gzip
import json
import subprocess
import sys
import urllib.parse
from collections.abc import Mapping, Sequence
from dataclasses import replace
from datetime import date, datetime, time
from pathlib import Path
from typing import Any, cast

import httpx
import numpy as np
import pandas as pd
import polars as pl
import pytest
import torch
from pydantic import ValidationError

import earnings_event_vol.cli as cli_module
import earnings_event_vol.contract_reference as contract_reference
import earnings_event_vol.data_pipeline as data_pipeline
import earnings_event_vol.event_window_panel as event_window_panel_module
import earnings_event_vol.massive as massive_module
import earnings_event_vol.trade_proxy as trade_proxy_module
import scripts.build_trade_proxy_panel as trade_proxy_panel_script
from earnings_event_vol.backtest import (
    GaussianEventJumpDistribution,
    StrategyPolicy,
    SymmetricTwoPointJumpDistribution,
    apply_portfolio_caps,
    apply_strategy_policy,
    black_scholes_price,
    build_proxy_strategy_frame,
    estimated_transaction_cost_usd,
    expected_strategy_value_usd,
    integer_contract_count,
    market_entry_cost_usd,
    premium_space_signal,
    tune_strategy_policy_validation_only,
)
from earnings_event_vol.cli import build_parser, main
from earnings_event_vol.config import load_project_config
from earnings_event_vol.contract_reference import apply_contract_reference_validation
from earnings_event_vol.data_audit import audit_data_fields, vendor_local_iv_comparison
from earnings_event_vol.data_pipeline import (
    TARGET_WINDOW_END,
    TARGET_WINDOW_START,
    DataPipelineStep,
    parse_text_list,
    run_data_pipeline,
)
from earnings_event_vol.earnings_calendar import (
    apply_official_then_aux_text_validation,
    apply_text_validation,
    build_earnings_calendar_candidates,
    build_earnings_calendar_report,
    classify_8k_text,
    fetch_massive_8k_text_payloads,
    fetch_sec_primary_document_texts,
    fetch_sec_submission_payloads,
    fetch_sec_ticker_map,
    infer_timing_from_acceptance_timestamp,
    load_json_payloads_from_dir,
    massive_text_by_accession,
    normalize_sec_submission_candidates,
    parse_aware_timestamp,
)
from earnings_event_vol.event_panel import (
    CONTRACT_STATUS_DOES_NOT_COVER_EVENT,
    CONTRACT_STATUS_MISSING_METADATA,
    CONTRACT_STATUS_NON_STANDARD_EXCLUDED,
    CONTRACT_STATUS_OK,
    CONTRACT_STATUS_OUTSIDE_DTE_RANGE,
    FORWARD_SOURCE_PUT_CALL_PARITY,
    FORWARD_SOURCE_SPOT_FALLBACK,
    build_event_panel,
    discover_option_contracts,
    flag_possible_preannouncement_or_prior_guidance,
    select_forward_and_atm,
)
from earnings_event_vol.event_targets import (
    add_event_return_targets,
    available_target_columns,
    target_label_column,
)
from earnings_event_vol.events import (
    align_event_window,
    has_ex_dividend_between,
    is_halted_or_proxy_halted,
    is_us_equity_trading_day,
    market_close_timestamp,
    market_close_timestamp_utc,
    next_us_equity_trading_day,
    previous_us_equity_trading_day,
    regular_close_timestamp,
    rvar_prices_for_window,
    validate_calendar_frame,
)
from earnings_event_vol.features import (
    FEATURE_SCHEMA_V1_LEGACY,
    FEATURE_SCHEMA_V2_SEC_XBRL,
    add_rolling_earnings_history,
    add_train_fit_normalized_features,
    build_feature_schema_report,
    build_model_feature_matrix,
    build_option_surface_sequence_matrix,
    feature_columns_from_schema_report,
    has_required_sequence_history,
    iv_butterfly_25d,
    normalization_params_only,
    sequence_eligibility_reason,
    universe_by_trailing_option_dollar_volume,
    validate_feature_schema_version,
)
from earnings_event_vol.leakage_audit import audit_feature_leakage, make_feature_timestamps
from earnings_event_vol.market_covariates import (
    FRED_VIXCLS_URL,
    VIX_ALIGNMENT_PRIOR_CLOSE,
    VIX_ALIGNMENT_SAME_DAY_AMC,
    build_vix_features,
    normalize_fred_vixcls_csv,
)
from earnings_event_vol.market_index_proxy import (
    _implied_volatility as market_index_implied_volatility,
)
from earnings_event_vol.market_index_proxy import (
    market_index_surface_features,
    normalize_underlying_second_aggregates,
    prefix_market_index_features,
    select_market_index_option_candidates,
    select_underlying_second_features,
)
from earnings_event_vol.massive import (
    MassiveCommandResult,
    build_head_object_command,
    build_massive_day_agg_sample,
    flat_file_object_specs,
    head_flat_file_objects,
    massive_flat_file_manifest,
    normalize_option_day_aggs,
    normalize_underlying_day_aggs,
    option_flat_file_key,
    option_quotes_flat_file_key,
    parse_flat_file_key_text,
    parse_massive_option_ticker,
    probe_key_file,
    read_secret_file,
    underlying_flat_file_key,
)
from earnings_event_vol.metrics import (
    auc_score,
    breakdown_metrics,
    brier_score,
    calibration_table,
    cost_sensitivity,
    edge_decile_table,
    evaluate_prediction_bundle,
    forecast_metrics,
    max_drawdown,
    qlike_loss,
    ranking_metrics,
    strategy_metrics,
)
from earnings_event_vol.models import (
    MODEL_REGISTRY,
    AttentionPoolingSequenceEncoder,
    DilatedCNNSequenceEncoder,
    FTTransformerRegressor,
    LinearElasticNetRegressor,
    RidgeRegressor,
    add_benchmark_predictions,
    fit_model,
    get_model_spec,
    model_diagnostics_as_frame,
    prediction_column_for_model,
    run_model_suite,
    sequence_feature_columns,
    sequence_tensor_from_frame,
    temporal_train_test_split,
    unimplemented_model_message,
)
from earnings_event_vol.research import (
    CANONICAL_BACK_TRANSFORM,
    CANONICAL_EVALUATION_SPACE,
    CANONICAL_TARGET_TRANSFORM,
    CANONICAL_TRAINING_SPACE,
    DEFAULT_TUNING_PROFILE,
    ENSEMBLE_RANK_SIGNAL_COL,
    FORECAST_FLOOR,
    HYBRID_SEQUENCE_FEATURE_NAMES,
    HYBRID_STEPS,
    SEQUENCE_FEATURE_NAMES,
    TARGET_IDS,
    TUNING_SELECTION_TARGET_ID,
    TuningState,
    _best_completed_trial,
    _latest_xbrl_values_for_event,
    _load_reusable_tuning_state,
    _log_rvar_to_variance,
    _model_ids_for_sequence_suite,
    _sequence_losses,
    _target_ids_for_sequence_suite,
    _target_to_log_rvar,
    _torch_log_rvar_to_variance,
    _write_tuning_artifacts,
    aggregate_sequence_features,
    assign_event_splits,
    build_common_row_diagnostics,
    build_completion_gap_audit,
    build_metric_tables,
    build_quote_diagnostic_tables,
    build_robustness_summary_table,
    build_sequence_tensor,
    build_sequence_v2_quality,
    enrich_feature_matrix_for_research,
    hybrid_sequence_coverage_by_event,
    inference_table,
    o2c_scale_diagnostic,
    prepare_target_frame,
    proxy_surface_distribution_audit,
    proxy_transaction_cost,
    qlike_sanity_table,
    research_paths,
    run_proxy_model_suite,
    run_proxy_research_package,
    run_research_models,
    run_research_report,
    sequence_coverage_by_event,
    sequence_coverage_report,
    write_proxy_research_report,
    write_retired_model_manifest,
)
from earnings_event_vol.schemas import (
    AnnouncementTiming,
    EarningsEvent,
    FeatureRow,
    IVARFailureReason,
    OptionQuote,
    OptionRight,
    OptionSide,
    SignalRecord,
    StrategyTrade,
    TimeConvention,
    TradeLeg,
    UnderlyingBar,
)
from earnings_event_vol.trade_proxy import (
    EXIT_PRECLOSE_OPTION_VWAP_STATUS_OK,
    OPTION_EXIT_STATUS_MISSING_PRECLOSE_VWAP,
    TRADE_PROXY_PANEL_GRADE,
    TRADE_PROXY_STATUS_NO_TRADE_IN_WINDOW,
    TRADE_PROXY_STATUS_OK,
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
    select_option_window_vwap,
    select_preclose_entry_proxy_price,
    summarize_trade_proxy_panel,
    write_trade_proxy_metadata,
)
from earnings_event_vol.universe import (
    ELIGIBLE_EQUITY_RULE_VERSION,
    PHASE1_COVID_SHOCK_BUCKET,
    PHASE1_STEADY_PROXY_BUCKET,
    TICKER_MAPPING_AMBIGUOUS,
    TICKER_MAPPING_OK,
    TICKER_NOT_FOUND,
    build_eligible_equity_tickers,
    build_monthly_liquid_universe,
    build_ticker_month_liquidity,
    eligible_equity_cache_matches_rule,
    phase1_telemetry_bucket,
    ticker_mapping_diagnostics,
)
from earnings_event_vol.variance import (
    TotalVariancePoint,
    extract_implied_event_variance,
    negative_ivar_diagnostics,
    realized_event_variance,
    year_fraction,
)

FIXTURES = Path(__file__).resolve().parent / "fixtures"


def _pipeline_steps(payload: dict[str, object]) -> list[dict[str, object]]:
    return cast(list[dict[str, object]], payload["steps"])


def _write_eligible_equity_cache(
    out_root: Path,
    rows: Sequence[Mapping[str, object]],
) -> Path:
    path = out_root / "universe" / "eligible_equity_tickers.parquet"
    path.parent.mkdir(parents=True, exist_ok=True)
    frame = build_eligible_equity_tickers(rows, source_snapshot_date=date(2026, 5, 6))
    pl.from_pandas(frame).write_parquet(path)
    return path


def _sec_submissions_payload(rows: list[dict[str, str]]) -> dict[str, object]:
    fields: dict[str, list[str]] = {
        "accessionNumber": [],
        "filingDate": [],
        "reportDate": [],
        "acceptanceDateTime": [],
        "form": [],
        "items": [],
        "primaryDocument": [],
        "primaryDocDescription": [],
    }
    for row in rows:
        for key in fields:
            fields[key].append(row.get(key, ""))
    return {"filings": {"recent": fields}}


def _massive_text_payload(rows: list[dict[str, str]]) -> dict[str, object]:
    return {"status": "OK", "results": rows}


def test_config_defaults_and_massive_helpers(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    secret = tmp_path / "massive_key"
    secret.write_text("redacted\n", encoding="utf-8")
    missing = tmp_path / "missing_key"
    data_dir = tmp_path / "data"

    monkeypatch.setenv("DATA_DIR", str(data_dir))
    monkeypatch.setenv("BRONZE_DATA_DIR", "")
    monkeypatch.setenv("SILVER_DATA_DIR", "")
    monkeypatch.setenv("GOLD_DATA_DIR", "")
    monkeypatch.setenv("MASSIVE_API_KEY_FILE", str(secret))
    monkeypatch.setenv("MASSIVE_FLAT_FILE_KEY_FILE", str(missing))
    monkeypatch.setenv("MASSIVE_REQUEST_TIMEOUT_SECONDS", "")
    monkeypatch.setenv("MASSIVE_MAX_RETRIES", "")
    monkeypatch.setenv("MASSIVE_REQUESTS_PER_MINUTE", "")

    config = load_project_config()

    assert config.data_dir == data_dir.resolve()
    assert config.bronze_data_dir == (data_dir / "bronze").resolve()
    assert config.massive_request_timeout_seconds == 30.0
    assert config.massive_max_retries == 3
    assert config.massive_requests_per_minute is None
    assert config.as_dict()["data_dir"] == str(data_dir.resolve())
    assert read_secret_file(None) is None
    assert read_secret_file(secret) == "redacted"
    assert probe_key_file(None).configured is False
    assert probe_key_file(secret).exists is True
    assert probe_key_file(missing).exists is False
    assert (
        option_flat_file_key(config, year=2026, month=2, date="2026-02-05")
        == "us_options_opra/day_aggs_v1/2026/02/2026-02-05.csv.gz"
    )
    assert (
        option_quotes_flat_file_key(config, year=2026, month=2, date="2026-02-05")
        == "us_options_opra/quotes_v1/2026/02/2026-02-05.csv.gz"
    )
    assert (
        underlying_flat_file_key(config, date="2026-02-05")
        == "us_stocks_sip/day_aggs_v1/2026/02/2026-02-05.csv.gz"
    )
    assert parse_flat_file_key_text("access\nsecret\n") == ("access", "secret")
    specs = flat_file_object_specs(config, date_value=date(2026, 2, 5))
    assert [spec.dataset for spec in specs] == [
        "options_day_aggs",
        "options_quotes",
        "underlying_day_aggs",
    ]
    assert specs[1].sample_allowed is False
    command = build_head_object_command(config, key=specs[0].key)
    assert command[:3] == ["aws", "s3api", "head-object"]
    assert specs[0].key in command


def test_massive_flat_file_manifest_uses_safe_head_metadata(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    secret = tmp_path / "massive_flat_file_key"
    secret.write_text("access\nsecret\n", encoding="utf-8")
    monkeypatch.setenv("MASSIVE_FLAT_FILE_KEY_FILE", str(secret))
    monkeypatch.setenv(
        "MASSIVE_UNDERLYING_FLAT_FILE_KEY_TEMPLATE",
        "us_stocks_sip/{dataset}/{year}/{month}/{date}.csv.gz",
    )
    config = load_project_config()

    def fake_runner(
        command: Sequence[str],
        env: Mapping[str, str],
        timeout: float,
    ) -> MassiveCommandResult:
        assert "AWS_ACCESS_KEY_ID" in env
        assert "AWS_SECRET_ACCESS_KEY" in env
        assert "--endpoint-url" in command
        assert timeout == config.massive_request_timeout_seconds
        return MassiveCommandResult(
            returncode=0,
            stdout=json.dumps(
                {
                    "ContentLength": 123,
                    "LastModified": "2025-02-06T11:00:00+00:00",
                    "ETag": '"abc"',
                }
            ),
            stderr="",
        )

    manifest = massive_flat_file_manifest(
        config,
        date_value=date(2025, 2, 5),
        runner=fake_runner,
    )

    assert manifest["date"] == "2025-02-05"
    assert manifest["head_object_ran"] is True
    assert len(manifest["objects"]) == 3
    assert all(item["ok"] is True for item in manifest["objects"])
    assert "manifest_hash" in manifest

    manifest_without_head = massive_flat_file_manifest(
        config,
        date_value=date(2025, 2, 5),
        run_head=False,
    )
    assert manifest_without_head["head_object_ran"] is False
    assert all(item["ok"] is None for item in manifest_without_head["objects"])


def test_massive_low_level_error_helpers(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    assert read_secret_file(None) is None
    with pytest.raises(ValueError, match="access key and secret key"):
        parse_flat_file_key_text("access-only\n")
    with pytest.raises(ValueError, match="MASSIVE_FLAT_FILE_KEY_FILE"):
        massive_module.read_flat_file_credentials(None)
    assert massive_module._ls_metadata_from_stdout(
        "2025-02-07 20:00:02 94947767406 2025-02-05.csv.gz\n"
    ) == (94947767406, "2025-02-07T20:00:02")
    assert massive_module._ls_metadata_from_stdout("not an s3 ls row") is None
    assert (
        massive_module._safe_error_text(MassiveCommandResult(returncode=7, stdout="", stderr=""))
        == "aws command failed with exit code 7"
    )

    def missing_run(*args: object, **kwargs: object) -> subprocess.CompletedProcess[str]:
        _ = args, kwargs
        raise FileNotFoundError("aws missing")

    monkeypatch.setattr("earnings_event_vol.massive.subprocess.run", missing_run)
    missing = massive_module._run_head_object_command(["aws"], {}, 1.0)
    assert missing.returncode == 127
    assert "aws missing" in missing.stderr

    def timeout_run(*args: object, **kwargs: object) -> subprocess.CompletedProcess[str]:
        _ = args, kwargs
        raise subprocess.TimeoutExpired(cmd=["aws"], timeout=1.0, output=b"out", stderr=b"err")

    monkeypatch.setattr("earnings_event_vol.massive.subprocess.run", timeout_run)
    timed_out = massive_module._run_head_object_command(["aws"], {}, 1.0)
    assert timed_out.returncode == 124
    assert timed_out.stdout == "out"
    assert timed_out.stderr == "err"

    key_file = tmp_path / "flat_file_key"
    key_file.write_text("access\nsecret\n", encoding="utf-8")
    config = replace(load_project_config(), massive_flat_file_key_file=key_file)

    def failing_runner(
        command: Sequence[str], env: Mapping[str, str], timeout: float
    ) -> MassiveCommandResult:
        _ = command, env, timeout
        return MassiveCommandResult(returncode=1, stdout="", stderr="download failed")

    with pytest.raises(RuntimeError, match="download failed"):
        massive_module.download_sample_allowed_flat_files(
            config,
            date_value=date(2025, 2, 5),
            out_dir=tmp_path / "downloads",
            runner=failing_runner,
        )


def test_massive_head_object_command_success_and_failed_fallback(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def completed_run(*args: object, **kwargs: object) -> subprocess.CompletedProcess[str]:
        _ = args, kwargs
        return subprocess.CompletedProcess(["aws"], 0, stdout='{"ok": true}', stderr="")

    monkeypatch.setattr("earnings_event_vol.massive.subprocess.run", completed_run)
    completed = massive_module._run_head_object_command(["aws"], {}, 1.0)
    assert completed.returncode == 0
    assert completed.stdout == '{"ok": true}'

    secret = tmp_path / "massive_flat_file_key"
    secret.write_text("access\nsecret\n", encoding="utf-8")
    monkeypatch.setenv("MASSIVE_FLAT_FILE_KEY_FILE", str(secret))
    config = load_project_config()

    def failing_runner(
        command: Sequence[str],
        env: Mapping[str, str],
        timeout: float,
    ) -> MassiveCommandResult:
        _ = env, timeout
        if command[1:3] == ["s3api", "head-object"]:
            return MassiveCommandResult(returncode=1, stdout="", stderr="head denied")
        return MassiveCommandResult(returncode=1, stdout="", stderr="ls denied")

    results = head_flat_file_objects(
        config,
        date_value=date(2025, 2, 5),
        runner=failing_runner,
    )

    assert len(results) == 3
    assert all(result.ok is False for result in results)
    assert all(result.error == "head denied" for result in results)


def test_massive_head_fallback_to_s3_ls(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    secret = tmp_path / "massive_flat_file_key"
    secret.write_text("access\nsecret\n", encoding="utf-8")
    monkeypatch.setenv("MASSIVE_FLAT_FILE_KEY_FILE", str(secret))
    config = load_project_config()

    def fake_runner(
        command: Sequence[str],
        env: Mapping[str, str],
        timeout: float,
    ) -> MassiveCommandResult:
        assert env["AWS_ACCESS_KEY_ID"] == "access"
        assert timeout == config.massive_request_timeout_seconds
        if command[1:3] == ["s3api", "head-object"]:
            return MassiveCommandResult(returncode=1, stdout="", stderr="Forbidden")
        return MassiveCommandResult(
            returncode=0,
            stdout="2025-02-07 20:00:02 94947767406 2025-02-05.csv.gz\n",
            stderr="",
        )

    results = head_flat_file_objects(
        config,
        date_value=date(2025, 2, 5),
        runner=fake_runner,
    )

    assert len(results) == 3
    assert all(result.ok for result in results)
    assert {result.metadata_source for result in results} == {"s3_ls"}


def test_massive_option_ticker_parsing_and_day_agg_normalization() -> None:
    parsed = parse_massive_option_ticker("O:AAPL250221C00145000")
    assert parsed["ticker"] == "AAPL"
    assert parsed["expiration"] == date(2025, 2, 21)
    assert parsed["right"] == "call"
    assert parsed["strike"] == pytest.approx(145.0)
    with pytest.raises(ValueError, match="unsupported Massive option ticker"):
        parse_massive_option_ticker("AAPL")

    raw_options = pd.DataFrame(
        {
            "ticker": ["O:A250221C00145000"],
            "volume": [16],
            "open": [5.0],
            "close": [5.0],
            "high": [5.2],
            "low": [5.0],
            "window_start": [1738731600000000000],
            "transactions": [7],
        }
    )
    normalized_options = normalize_option_day_aggs(raw_options, quote_date=date(2025, 2, 5))
    assert normalized_options.loc[0, "ticker"] == "A"
    assert normalized_options.loc[0, "quote_date"] == date(2025, 2, 5)
    assert "bid" not in normalized_options.columns
    assert "open_interest" not in normalized_options.columns

    raw_underlying = pd.DataFrame(
        {
            "ticker": ["A"],
            "volume": [1348074],
            "open": [147.89],
            "close": [147.99],
            "high": [148.71],
            "low": [146.31],
            "window_start": [1738731600000000000],
            "transactions": [21077],
        }
    )
    normalized_underlying = normalize_underlying_day_aggs(raw_underlying, bar_date=date(2025, 2, 5))
    assert list(normalized_underlying.columns) == [
        "ticker",
        "date",
        "open",
        "high",
        "low",
        "close",
        "volume",
        "source_dataset",
    ]


def test_build_massive_day_agg_sample_reports_v1_gaps(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    secret = tmp_path / "massive_flat_file_key"
    secret.write_text("access\nsecret\n", encoding="utf-8")
    monkeypatch.setenv("MASSIVE_FLAT_FILE_KEY_FILE", str(secret))
    config = load_project_config()

    def fake_runner(
        command: Sequence[str],
        env: Mapping[str, str],
        timeout: float,
    ) -> MassiveCommandResult:
        if command[1:3] == ["s3api", "head-object"]:
            return MassiveCommandResult(
                returncode=0,
                stdout=json.dumps(
                    {
                        "ContentLength": 100,
                        "LastModified": "2025-02-06T11:00:00+00:00",
                        "ETag": '"abc"',
                    }
                ),
                stderr="",
            )
        assert command[1:3] == ["s3", "cp"]
        destination = Path(command[4])
        destination.parent.mkdir(parents=True, exist_ok=True)
        if "us_options_opra" in command[3]:
            payload = (
                "ticker,volume,open,close,high,low,window_start,transactions\n"
                "O:A250221C00145000,16,5,5,5.2,5,1738731600000000000,7\n"
            )
        else:
            payload = (
                "ticker,volume,open,close,high,low,window_start,transactions\n"
                "A,1348074,147.89,147.99,148.71,146.31,1738731600000000000,21077\n"
            )
        with gzip.open(destination, "wt", encoding="utf-8") as handle:
            handle.write(payload)
        return MassiveCommandResult(returncode=0, stdout="", stderr="")

    report = build_massive_day_agg_sample(
        config,
        date_value=date(2025, 2, 5),
        out_dir=tmp_path / "sample",
        runner=fake_runner,
    )

    assert report["v1_readiness"]["day_aggs_support_contract_parsing"] is True
    assert report["v1_readiness"]["day_aggs_support_bid_ask_costs"] is False
    assert "bid" in report["v1_readiness"]["missing_quote_fields"]
    assert (tmp_path / "sample" / "massive_sample_schema_report.json").exists()


def test_earnings_calendar_candidates_from_sec_and_massive_fixture_dirs(
    tmp_path: Path,
) -> None:
    sec_dir = tmp_path / "sec"
    massive_dir = tmp_path / "massive"
    sec_dir.mkdir()
    massive_dir.mkdir()

    (sec_dir / "AAPL.json").write_text(
        json.dumps(
            _sec_submissions_payload(
                [
                    {
                        "accessionNumber": "0000320193-26-000005",
                        "filingDate": "2026-01-29",
                        "reportDate": "2025-12-27",
                        "acceptanceDateTime": "2026-01-29T21:30:33.000Z",
                        "form": "8-K",
                        "items": "2.02,9.01",
                        "primaryDocument": "aapl-20260129.htm",
                    },
                    {
                        "accessionNumber": "0000320193-26-000006",
                        "filingDate": "2026-01-30",
                        "acceptanceDateTime": "2026-01-30T21:30:33.000Z",
                        "form": "8-K",
                        "items": "5.02",
                    },
                ]
            )
        ),
        encoding="utf-8",
    )
    (sec_dir / "MSFT.json").write_text(
        json.dumps(
            _sec_submissions_payload(
                [
                    {
                        "accessionNumber": "0001193125-26-191457",
                        "filingDate": "2026-04-29",
                        "acceptanceDateTime": "2026-04-29T12:15:00.000Z",
                        "form": "8-K",
                        "items": "2.02,9.01",
                    },
                    {
                        "accessionNumber": "0001193125-26-191458",
                        "filingDate": "2026-04-30",
                        "acceptanceDateTime": "2026-04-30T15:00:00.000Z",
                        "form": "8-K/A",
                        "items": "2.02,9.01",
                    },
                ]
            )
        ),
        encoding="utf-8",
    )
    (sec_dir / "TSLA.json").write_text(
        json.dumps(
            _sec_submissions_payload(
                [
                    {
                        "accessionNumber": "0001628280-26-022956",
                        "filingDate": "2026-04-02",
                        "acceptanceDateTime": "2026-04-02T13:07:13.000Z",
                        "form": "8-K",
                        "items": "2.02,9.01",
                    }
                ]
            )
        ),
        encoding="utf-8",
    )
    earnings_text = (
        "Item 2.02 Results of Operations and Financial Condition. "
        "The company announced financial results for the fiscal quarter ended March 31."
    )
    (massive_dir / "AAPL.json").write_text(
        json.dumps(
            _massive_text_payload(
                [{"accession_number": "0000320193-26-000005", "items_text": earnings_text}]
            )
        ),
        encoding="utf-8",
    )
    (massive_dir / "MSFT.json").write_text(
        json.dumps(
            _massive_text_payload(
                [
                    {
                        "accession_number": "0001193125-26-191457",
                        "items_text": earnings_text,
                    },
                    {
                        "accession_number": "0001193125-26-191458",
                        "items_text": earnings_text,
                    },
                ]
            )
        ),
        encoding="utf-8",
    )
    (massive_dir / "TSLA.json").write_text(
        json.dumps(
            _massive_text_payload(
                [
                    {
                        "accession_number": "0001628280-26-022956",
                        "items_text": (
                            "Item 2.02 Results of Operations and Financial Condition. "
                            "Tesla published production and deliveries for the quarter."
                        ),
                    }
                ]
            )
        ),
        encoding="utf-8",
    )

    frame, report = build_earnings_calendar_candidates(
        config=load_project_config(),
        tickers=["AAPL", "MSFT", "TSLA"],
        start_date=date(2026, 1, 1),
        end_date=date(2026, 12, 31),
        sec_submissions_dir=sec_dir,
        massive_8k_text_dir=massive_dir,
    )

    assert len(frame) == 3
    assert "8-K/A" not in set(frame["form_type"])
    assert report["main_sample_candidate_rows"] == 0
    assert report["timing_counts"] == {"UNKNOWN": 3}
    assert report["acceptance_inferred_timing_counts"] == {"AMC": 1, "BMO": 2}
    assert (
        frame.loc[frame["source_id"] == "0001628280-26-022956", "text_validation_status"].iloc[0]
        == "non_earnings_item_2_02"
    )
    validated_calendar = validate_calendar_frame(
        frame[["ticker", "announcement_date", "announcement_timing", "source"]]
    )
    assert int(validated_calendar["is_main_sample_timing"].sum()) == 0


def test_earnings_calendar_http_fetch_path_uses_official_and_massive_sources(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    api_key = tmp_path / "massive_api_key"
    api_key.write_text("redacted", encoding="utf-8")
    monkeypatch.setenv("MASSIVE_API_KEY_FILE", str(api_key))
    monkeypatch.setenv("MASSIVE_BASE_URL", "https://massive.example")
    monkeypatch.setenv("MASSIVE_MAX_RETRIES", "1")
    monkeypatch.setenv("MASSIVE_RETRY_BACKOFF_SECONDS", "0")
    monkeypatch.setenv("SEC_COMPANY_TICKERS_URL", "https://sec.example/tickers.json")
    monkeypatch.setenv(
        "SEC_SUBMISSIONS_URL_TEMPLATE",
        "https://sec.example/submissions/CIK{cik:010d}.json",
    )
    config = replace(load_project_config(), bronze_data_dir=tmp_path / "bronze")

    sec_payload = _sec_submissions_payload(
        [
            {
                "accessionNumber": "0000320193-26-000005",
                "filingDate": "2026-01-29",
                "acceptanceDateTime": "2026-01-29T21:30:33.000Z",
                "form": "8-K",
                "items": "2.02,9.01",
                "primaryDocument": "aapl-20260129.htm",
            }
        ]
    )
    massive_calls = {"count": 0}

    def handler(request: httpx.Request) -> httpx.Response:
        if str(request.url) == "https://sec.example/tickers.json":
            return httpx.Response(200, json={"0": {"ticker": "AAPL", "cik_str": 320193}})
        if str(request.url) == "https://sec.example/submissions/CIK0000320193.json":
            return httpx.Response(200, json=sec_payload)
        if str(request.url) == (
            "https://www.sec.gov/Archives/edgar/data/320193/000032019326000005/aapl-20260129.htm"
        ):
            return httpx.Response(
                200,
                text=(
                    "Item 2.02 Results of Operations and Financial Condition. "
                    "Apple issued financial results for its fiscal quarter."
                ),
            )
        if request.url.host == "massive.example":
            massive_calls["count"] += 1
            if massive_calls["count"] == 1:
                raise httpx.RemoteProtocolError("server disconnected")
            assert request.url.params["apiKey"] == "redacted"
            return httpx.Response(
                200,
                json=_massive_text_payload(
                    [
                        {
                            "accession_number": "0000320193-26-000005",
                            "filing_date": "2026-01-29",
                            "items_text": (
                                "Item 2.02 Results of Operations and Financial Condition. "
                                "Apple issued financial results for its fiscal quarter."
                            ),
                        }
                    ]
                ),
            )
        return httpx.Response(404)

    with httpx.Client(transport=httpx.MockTransport(handler)) as client:
        frame, report = build_earnings_calendar_candidates(
            config=config,
            tickers=["AAPL"],
            start_date=date(2026, 1, 1),
            end_date=date(2026, 12, 31),
            http_client=client,
        )

    assert frame["announcement_timing"].tolist() == ["UNKNOWN"]
    assert frame["acceptance_inferred_timing"].tolist() == ["AMC"]
    assert frame["is_main_sample_candidate"].tolist() == [False]
    assert frame["text_validation_source"].tolist() == ["sec_primary_document_text"]
    assert report["validation_route"] == "sec_edgar_http+sec_primary_document_text"
    assert report["massive_8k_fetch_failed"] == 0
    assert massive_calls["count"] == 0


def test_earnings_calendar_uses_massive_only_as_auxiliary_fallback(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    api_key = tmp_path / "massive_api_key"
    api_key.write_text("redacted", encoding="utf-8")
    monkeypatch.setenv("MASSIVE_API_KEY_FILE", str(api_key))
    monkeypatch.setenv("MASSIVE_BASE_URL", "https://massive.example")
    monkeypatch.setenv("SEC_COMPANY_TICKERS_URL", "https://sec.example/tickers.json")
    monkeypatch.setenv(
        "SEC_SUBMISSIONS_URL_TEMPLATE",
        "https://sec.example/submissions/CIK{cik:010d}.json",
    )
    config = replace(load_project_config(), bronze_data_dir=tmp_path / "bronze")
    sec_payload = _sec_submissions_payload(
        [
            {
                "accessionNumber": "0000320193-26-000005",
                "filingDate": "2026-01-29",
                "acceptanceDateTime": "2026-01-29T21:30:33.000Z",
                "form": "8-K",
                "items": "2.02,9.01",
                "primaryDocument": "aapl-20260129.htm",
            }
        ]
    )
    massive_calls = {"count": 0}

    def handler(request: httpx.Request) -> httpx.Response:
        if str(request.url) == "https://sec.example/tickers.json":
            return httpx.Response(200, json={"0": {"ticker": "AAPL", "cik_str": 320193}})
        if str(request.url) == "https://sec.example/submissions/CIK0000320193.json":
            return httpx.Response(200, json=sec_payload)
        if str(request.url) == (
            "https://www.sec.gov/Archives/edgar/data/320193/000032019326000005/aapl-20260129.htm"
        ):
            return httpx.Response(200, text="Item 2.02 Results of Operations.")
        if request.url.host == "massive.example":
            massive_calls["count"] += 1
            assert request.url.params["apiKey"] == "redacted"
            return httpx.Response(
                200,
                json=_massive_text_payload(
                    [
                        {
                            "accession_number": "0000320193-26-000005",
                            "filing_date": "2026-01-29",
                            "items_text": (
                                "Item 2.02 Results of Operations and Financial Condition. "
                                "Apple issued financial results for its fiscal quarter."
                            ),
                        }
                    ]
                ),
            )
        return httpx.Response(404)

    with httpx.Client(transport=httpx.MockTransport(handler)) as client:
        frame, report = build_earnings_calendar_candidates(
            config=config,
            tickers=["AAPL"],
            start_date=date(2026, 1, 1),
            end_date=date(2026, 12, 31),
            http_client=client,
        )

    assert frame["text_validation_source"].tolist() == ["massive_8k_text_fallback"]
    assert frame["text_validation_aux_status"].tolist() == ["validated_earnings_release"]
    assert frame["is_main_sample_candidate"].tolist() == [False]
    assert report["validation_route"] == (
        "sec_edgar_http+sec_primary_document_text+massive_8k_text_http_auxiliary"
    )
    assert massive_calls["count"] == 1


def test_earnings_calendar_soft_fails_when_auxiliary_massive_key_missing(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("SEC_COMPANY_TICKERS_URL", "https://sec.example/tickers.json")
    monkeypatch.setenv(
        "SEC_SUBMISSIONS_URL_TEMPLATE",
        "https://sec.example/submissions/CIK{cik:010d}.json",
    )
    config = replace(
        load_project_config(),
        bronze_data_dir=tmp_path / "bronze",
        massive_api_key_file=tmp_path / "missing_massive_key",
    )
    sec_payload = _sec_submissions_payload(
        [
            {
                "accessionNumber": "0000320193-26-000005",
                "filingDate": "2026-01-29",
                "acceptanceDateTime": "2026-01-29T21:30:33.000Z",
                "form": "8-K",
                "items": "2.02,9.01",
                "primaryDocument": "aapl-20260129.htm",
            }
        ]
    )

    def handler(request: httpx.Request) -> httpx.Response:
        if str(request.url) == "https://sec.example/tickers.json":
            return httpx.Response(200, json={"0": {"ticker": "AAPL", "cik_str": 320193}})
        if str(request.url) == "https://sec.example/submissions/CIK0000320193.json":
            return httpx.Response(200, json=sec_payload)
        if str(request.url) == (
            "https://www.sec.gov/Archives/edgar/data/320193/000032019326000005/aapl-20260129.htm"
        ):
            return httpx.Response(200, text="Item 2.02 Results of Operations.")
        raise AssertionError(f"Massive fallback should stop at missing key: {request.url}")

    with httpx.Client(transport=httpx.MockTransport(handler)) as client:
        frame, report = build_earnings_calendar_candidates(
            config=config,
            tickers=["AAPL"],
            start_date=date(2026, 1, 1),
            end_date=date(2026, 12, 31),
            http_client=client,
        )

    assert frame["text_validation_source"].tolist() == ["sec_primary_document_text"]
    assert frame["text_validation_status"].tolist() == ["ambiguous_item_2_02_text"]
    assert report["massive_8k_aux_status"] == "unavailable_missing_key"


def test_earnings_calendar_soft_fails_when_auxiliary_massive_unauthorized(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    api_key = tmp_path / "massive_api_key"
    api_key.write_text("redacted", encoding="utf-8")
    monkeypatch.setenv("MASSIVE_API_KEY_FILE", str(api_key))
    monkeypatch.setenv("MASSIVE_BASE_URL", "https://massive.example")
    monkeypatch.setenv("SEC_COMPANY_TICKERS_URL", "https://sec.example/tickers.json")
    monkeypatch.setenv(
        "SEC_SUBMISSIONS_URL_TEMPLATE",
        "https://sec.example/submissions/CIK{cik:010d}.json",
    )
    config = replace(load_project_config(), bronze_data_dir=tmp_path / "bronze")
    sec_payload = _sec_submissions_payload(
        [
            {
                "accessionNumber": "0000320193-26-000005",
                "filingDate": "2026-01-29",
                "acceptanceDateTime": "2026-01-29T21:30:33.000Z",
                "form": "8-K",
                "items": "2.02,9.01",
                "primaryDocument": "aapl-20260129.htm",
            }
        ]
    )

    def handler(request: httpx.Request) -> httpx.Response:
        if str(request.url) == "https://sec.example/tickers.json":
            return httpx.Response(200, json={"0": {"ticker": "AAPL", "cik_str": 320193}})
        if str(request.url) == "https://sec.example/submissions/CIK0000320193.json":
            return httpx.Response(200, json=sec_payload)
        if str(request.url) == (
            "https://www.sec.gov/Archives/edgar/data/320193/000032019326000005/aapl-20260129.htm"
        ):
            return httpx.Response(200, text="Item 2.02 Results of Operations.")
        if request.url.host == "massive.example":
            return httpx.Response(401, json={"error": "unauthorized"})
        return httpx.Response(404)

    with httpx.Client(transport=httpx.MockTransport(handler)) as client:
        frame, report = build_earnings_calendar_candidates(
            config=config,
            tickers=["AAPL"],
            start_date=date(2026, 1, 1),
            end_date=date(2026, 12, 31),
            http_client=client,
        )

    assert frame["text_validation_source"].tolist() == ["sec_primary_document_text"]
    assert frame["text_validation_status"].tolist() == ["ambiguous_item_2_02_text"]
    assert report["massive_8k_aux_status"] == "unavailable_http_401"


def test_sec_primary_document_fetch_cache_and_failure_paths(tmp_path: Path) -> None:
    config = replace(load_project_config(), massive_max_retries=0)
    with httpx.Client(transport=httpx.MockTransport(lambda request: httpx.Response(404))) as client:
        empty_texts, empty_failures = fetch_sec_primary_document_texts(
            candidates=pd.DataFrame(),
            config=config,
            client=client,
            cache_dir=tmp_path / "cache",
            request_interval_seconds=0,
        )
    assert empty_texts == {}
    assert empty_failures == []

    candidates = pd.DataFrame(
        [
            {
                "ticker": "AAPL",
                "source_id": "0000320193-26-000005",
                "cik": 320193,
                "primary_document": "ok.htm",
            },
            {
                "ticker": "AAPL",
                "source_id": "0000320193-26-000005",
                "cik": 320193,
                "primary_document": "ok.htm",
            },
            {
                "ticker": "MSFT",
                "source_id": "0000789019-26-000001",
                "cik": 789019,
                "primary_document": "missing.htm",
            },
            {
                "ticker": "NVDA",
                "source_id": "0001045810-26-000001",
                "cik": None,
                "primary_document": "",
            },
            {
                "ticker": "",
                "source_id": "",
                "cik": 1,
                "primary_document": "ignored.htm",
            },
        ]
    )
    calls: list[str] = []

    def handler(request: httpx.Request) -> httpx.Response:
        calls.append(str(request.url))
        if str(request.url).endswith("/ok.htm"):
            return httpx.Response(200, text="Item 2.02 quarterly results.")
        return httpx.Response(404)

    with httpx.Client(transport=httpx.MockTransport(handler)) as client:
        texts, failures = fetch_sec_primary_document_texts(
            candidates=candidates,
            config=config,
            client=client,
            cache_dir=tmp_path / "cache",
            request_interval_seconds=0,
        )

    assert texts[("AAPL", "0000320193-26-000005")] == "Item 2.02 quarterly results."
    assert len([url for url in calls if url.endswith("/ok.htm")]) == 1
    assert {failure["reason"] for failure in failures} == {
        "missing_cik_or_primary_document",
        "sec_primary_document_fetch_failed",
    }

    def no_network_handler(request: httpx.Request) -> httpx.Response:
        raise AssertionError(f"unexpected cache-miss fetch: {request.url}")

    with httpx.Client(transport=httpx.MockTransport(no_network_handler)) as client:
        cached_texts, cached_failures = fetch_sec_primary_document_texts(
            candidates=candidates.iloc[[0]],
            config=config,
            client=client,
            cache_dir=tmp_path / "cache",
            request_interval_seconds=0,
        )
    assert cached_texts == texts
    assert cached_failures == []

    unresolved = apply_official_then_aux_text_validation(
        pd.DataFrame(
            [
                {
                    "ticker": "AAPL",
                    "source_id": "0000320193-26-000005",
                    "is_main_sample_timing": True,
                }
            ]
        ),
        sec_text_by_accession={
            ("AAPL", "0000320193-26-000005"): "Item 2.02 Results of Operations."
        },
        aux_text_by_accession={},
    )
    assert unresolved["text_validation_source"].iloc[0] == "sec_primary_document_text"
    assert unresolved["text_validation_aux_status"].iloc[0] == "missing_text"
    assert "Massive auxiliary status: missing_text" in unresolved["text_validation_reason"].iloc[0]


def test_sec_archived_submission_files_are_cached_normalized_and_deduped(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("SEC_COMPANY_TICKERS_URL", "https://sec.example/tickers.json")
    monkeypatch.setenv(
        "SEC_SUBMISSIONS_URL_TEMPLATE",
        "https://data.sec.gov/submissions/CIK{cik:010d}.json",
    )
    config = load_project_config()
    archive_name = "CIK0000320193-submissions-001.json"
    recent_payload = _sec_submissions_payload(
        [
            {
                "accessionNumber": "duplicate",
                "filingDate": "2013-01-24",
                "acceptanceDateTime": "2013-01-24T21:30:00.000Z",
                "form": "8-K",
                "items": "2.02",
            }
        ]
    )
    recent_payload["filings"]["files"] = [{"name": archive_name}]  # type: ignore[index]
    archive_payload = {
        "accessionNumber": ["duplicate", "archive-only"],
        "filingDate": ["2013-01-24", "2013-04-23"],
        "acceptanceDateTime": ["2013-01-24T21:30:00.000Z", "2013-04-23T12:15:00.000Z"],
        "form": ["8-K", "8-K"],
        "items": ["2.02", "2.02"],
        "reportDate": ["", ""],
        "primaryDocument": ["", ""],
        "primaryDocDescription": ["", ""],
    }

    def handler(request: httpx.Request) -> httpx.Response:
        if str(request.url) == "https://sec.example/tickers.json":
            return httpx.Response(200, json={"0": {"ticker": "AAPL", "cik_str": 320193}})
        if str(request.url) == "https://data.sec.gov/submissions/CIK0000320193.json":
            return httpx.Response(200, json=recent_payload)
        if str(request.url) == f"https://data.sec.gov/submissions/{archive_name}":
            return httpx.Response(200, json=archive_payload)
        return httpx.Response(404)

    cache_dir = tmp_path / "sec_cache"
    with httpx.Client(transport=httpx.MockTransport(handler)) as client:
        payloads = fetch_sec_submission_payloads(
            tickers=["AAPL"],
            config=config,
            client=client,
            archive_cache_dir=cache_dir,
            request_interval_seconds=0,
        )
    normalized = normalize_sec_submission_candidates(
        ticker="AAPL",
        payload=payloads["AAPL"],
        start_date=date(2013, 1, 1),
        end_date=date(2013, 12, 31),
    )
    assert (cache_dir / "ticker=AAPL" / archive_name).exists()
    assert normalized["source_id"].tolist() == ["duplicate", "archive-only"]
    assert normalized["source"].tolist() == [
        "sec_edgar_submissions_recent",
        "sec_edgar_submissions_archive",
    ]


def test_sec_submission_archives_cache_failure_and_missing_ticker(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("SEC_COMPANY_TICKERS_URL", "https://sec.example/tickers.json")
    monkeypatch.setenv(
        "SEC_SUBMISSIONS_URL_TEMPLATE", "https://data.sec.gov/submissions/CIK{cik:010d}.json"
    )
    config = load_project_config()
    cache_dir = tmp_path / "sec_cache"
    corrupt_archive = cache_dir / "ticker=AAPL" / "bad.json"
    cached_archive = cache_dir / "ticker=AAPL" / "cached.json"
    corrupt_archive.parent.mkdir(parents=True)
    corrupt_archive.write_text("{bad json", encoding="utf-8")
    cached_archive.write_text(
        json.dumps(
            {
                "accessionNumber": ["cached"],
                "filingDate": ["2025-01-03"],
                "acceptanceDateTime": ["2025-01-03T20:00:00.000Z"],
                "form": ["8-K"],
                "items": ["2.02"],
            }
        ),
        encoding="utf-8",
    )

    def handler(request: httpx.Request) -> httpx.Response:
        url = str(request.url)
        if url == "https://sec.example/tickers.json":
            return httpx.Response(200, json={"0": {"ticker": "AAPL", "cik_str": 320193}})
        if url == "https://data.sec.gov/submissions/CIK0000320193.json":
            return httpx.Response(
                200,
                json={
                    "filings": {
                        "recent": {},
                        "files": [
                            "bad-item",
                            {},
                            {"name": ""},
                            {"name": "bad.json"},
                            {"name": "cached.json"},
                        ],
                    }
                },
            )
        if url == "https://data.sec.gov/submissions/bad.json":
            return httpx.Response(503, json={"error": "temporary"})
        raise AssertionError(f"unexpected URL {url}")

    with httpx.Client(transport=httpx.MockTransport(handler)) as client:
        payloads = fetch_sec_submission_payloads(
            tickers=["AAPL", "MSFT"],
            config=config,
            client=client,
            archive_cache_dir=cache_dir,
            fail_on_missing_tickers=False,
            request_interval_seconds=0,
        )

    assert payloads["MSFT"]["sec_fetch_status"] == "ticker_not_found"
    assert payloads["AAPL"]["sec_fetch_status"] == "ok"
    assert len(payloads["AAPL"]["archive_payloads"]) == 1
    assert payloads["AAPL"]["archive_payloads"][0]["accessionNumber"] == ["cached"]
    failures = payloads["AAPL"]["sec_archive_fetch_failed"]
    assert [failure["name"] for failure in failures] == ["bad.json", "bad.json"]


def test_earnings_calendar_timing_and_text_classification() -> None:
    assert infer_timing_from_acceptance_timestamp("2026-04-02T13:07:13.000Z") == "BMO"
    assert infer_timing_from_acceptance_timestamp("2026-04-02T15:00:00.000Z") == "DMH"
    assert infer_timing_from_acceptance_timestamp("2026-04-02T20:05:00.000Z") == "AMC"
    assert infer_timing_from_acceptance_timestamp("2026-04-02T20:05:00") == "UNKNOWN"
    assert parse_aware_timestamp(None) is None
    assert parse_aware_timestamp("") is None
    assert parse_aware_timestamp("not-a-timestamp") is None

    status, _ = classify_8k_text(
        "Item 2.02 Results of Operations and Financial Condition. "
        "The company reported quarterly results for the quarter ended March 31."
    )
    assert status == "validated_earnings_release"
    assert classify_8k_text("Item 8.01 Other Events.")[0] == "not_item_2_02_text"
    status, _ = classify_8k_text(
        "Item 2.02 Results of Operations and Financial Condition. "
        "The company reported production and deliveries for the quarter."
    )
    assert status == "non_earnings_item_2_02"
    status, _ = classify_8k_text(
        "Item 2.02 Results of Operations and Financial Condition. "
        "Quarterly results include production and deliveries metrics."
    )
    assert status == "non_earnings_item_2_02"
    assert classify_8k_text("Item 2.02 Results of Operations. Press release attached.")[0] == (
        "ambiguous_item_2_02_text"
    )
    assert classify_8k_text(None)[0] == "missing_text"

    normalized = normalize_sec_submission_candidates(
        ticker="AAPL",
        payload=_sec_submissions_payload(
            [
                {
                    "accessionNumber": "outside",
                    "filingDate": "2025-12-31",
                    "acceptanceDateTime": "2025-12-31T21:00:00.000Z",
                    "form": "8-K",
                    "items": "2.02,9.01",
                }
            ]
        ),
        start_date=date(2026, 1, 1),
        end_date=date(2026, 12, 31),
    )
    assert normalized.empty

    mismatch = normalize_sec_submission_candidates(
        ticker="AI",
        payload=_sec_submissions_payload(
            [
                {
                    "accessionNumber": "0001833214-25-000101",
                    "filingDate": "2025-08-11",
                    "acceptanceDateTime": "2025-08-08T23:32:37.000Z",
                    "form": "8-K",
                    "items": "2.02,9.01",
                },
                {
                    "accessionNumber": "0001833214-25-000102",
                    "filingDate": "2025-08-12",
                    "acceptanceDateTime": "2025-08-12T21:00:00.000Z",
                    "form": "8-K/A",
                    "items": "2.02,9.01",
                },
            ]
        ),
        start_date=date(2025, 8, 1),
        end_date=date(2025, 8, 31),
    )
    assert len(mismatch) == 1
    assert mismatch["source_id"].tolist() == ["0001833214-25-000101"]
    assert mismatch["announcement_date"].iloc[0] == "2025-08-08"
    assert mismatch["filing_date"].iloc[0] == "2025-08-11"
    assert mismatch["acceptance_local_date"].iloc[0] == "2025-08-08"
    assert mismatch["acceptance_inferred_timing"].iloc[0] == "AMC"
    assert mismatch["announcement_date_source"].iloc[0] == "sec_acceptance_local_date_proxy"
    assert bool(mismatch["filing_acceptance_date_mismatch"].iloc[0]) is True

    assert normalize_sec_submission_candidates(
        ticker="AAPL",
        payload={"filings": "bad"},
        start_date=date(2026, 1, 1),
        end_date=date(2026, 12, 31),
    ).empty
    assert normalize_sec_submission_candidates(
        ticker="AAPL",
        payload={"filings": {"recent": {"form": "8-K"}}},
        start_date=date(2026, 1, 1),
        end_date=date(2026, 12, 31),
    ).empty
    assert normalize_sec_submission_candidates(
        ticker="AAPL",
        payload=_sec_submissions_payload(
            [
                {"form": "10-K", "items": "2.02", "filingDate": "2026-01-29"},
                {"form": "8-K", "items": "2.02", "filingDate": "not-a-date"},
            ]
        ),
        start_date=date(2026, 1, 1),
        end_date=date(2026, 12, 31),
    ).empty
    assert normalize_sec_submission_candidates(
        ticker="AAPL",
        payload={
            "filings": {
                "recent": {
                    "form": ["8-K", "8-K"],
                    "items": ["2.02", "2.02"],
                    "filingDate": [None, ""],
                }
            }
        },
        start_date=date(2026, 1, 1),
        end_date=date(2026, 12, 31),
    ).empty


def test_earnings_calendar_fail_closed_edge_cases(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    with pytest.raises(ValueError, match="missing fixture payload"):
        load_json_payloads_from_dir(["AAPL"], tmp_path)

    bad_dir = tmp_path / "bad"
    bad_dir.mkdir()
    (bad_dir / "AAPL.json").write_text("[]", encoding="utf-8")
    with pytest.raises(ValueError, match="JSON object"):
        load_json_payloads_from_dir(["AAPL"], bad_dir)

    assert (
        massive_text_by_accession(
            {
                "AAPL": {"results": "bad"},
                "MSFT": {"results": ["bad", {}, {"accession_number": "", "items_text": "x"}]},
            }
        )
        == {}
    )

    skipped = apply_text_validation(
        pd.DataFrame(
            [
                {
                    "ticker": "AAPL",
                    "source_id": "accession",
                    "is_main_sample_timing": True,
                }
            ]
        ),
        None,
    )
    assert skipped["text_validation_status"].iloc[0] == "validation_skipped"

    empty_report = build_earnings_calendar_report(
        frame=pd.DataFrame(),
        tickers=["AAPL"],
        start_date=date(2026, 1, 1),
        end_date=date(2026, 12, 31),
        validation_route="test",
    )
    assert empty_report["rows_by_ticker"] == {}

    config = load_project_config()
    with pytest.raises(ValueError, match="at least one ticker"):
        build_earnings_calendar_candidates(
            config=config,
            tickers=[],
            start_date=date(2026, 1, 1),
            end_date=date(2026, 12, 31),
        )
    with pytest.raises(ValueError, match="start_date"):
        build_earnings_calendar_candidates(
            config=config,
            tickers=["AAPL"],
            start_date=date(2026, 12, 31),
            end_date=date(2026, 1, 1),
        )

    sec_dir = tmp_path / "sec_skip"
    sec_dir.mkdir()
    (sec_dir / "AAPL.json").write_text(
        json.dumps(
            _sec_submissions_payload(
                [
                    {
                        "accessionNumber": "0000320193-26-000005",
                        "filingDate": "2026-01-29",
                        "acceptanceDateTime": "2026-01-29T21:30:33.000Z",
                        "form": "8-K",
                        "items": "2.02,9.01",
                    }
                ]
            )
        ),
        encoding="utf-8",
    )
    skipped_frame, skipped_report = build_earnings_calendar_candidates(
        config=config,
        tickers=["AAPL"],
        start_date=date(2026, 1, 1),
        end_date=date(2026, 12, 31),
        sec_submissions_dir=sec_dir,
        validate_with_massive=False,
    )
    assert skipped_frame["text_validation_status"].iloc[0] == "validation_skipped"
    assert skipped_report["validation_route"] == "fixture_dir+skipped"

    monkeypatch.delenv("MASSIVE_API_KEY_FILE", raising=False)
    with (
        httpx.Client(transport=httpx.MockTransport(lambda request: httpx.Response(200))) as client,
        pytest.raises(ValueError, match="MASSIVE_API_KEY_FILE"),
    ):
        fetch_massive_8k_text_payloads(
            tickers=["AAPL"],
            config=load_project_config(),
            client=client,
            start_date=date(2026, 1, 1),
            end_date=date(2026, 12, 31),
        )


def test_earnings_calendar_http_fetch_fail_closed_branches(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    api_key = tmp_path / "massive_api_key"
    api_key.write_text("redacted", encoding="utf-8")
    monkeypatch.setenv("MASSIVE_API_KEY_FILE", str(api_key))
    monkeypatch.setenv("MASSIVE_BASE_URL", "https://massive.example")
    monkeypatch.setenv("SEC_COMPANY_TICKERS_URL", "https://sec.example/tickers.json")
    config = load_project_config()

    with (
        httpx.Client(
            transport=httpx.MockTransport(lambda request: httpx.Response(200, json=[]))
        ) as client,
        pytest.raises(ValueError, match="expected JSON object"),
    ):
        fetch_sec_ticker_map(client, config)

    def sec_missing_handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            json={
                "0": ["bad"],
                "1": {"ticker": "BAD", "cik_str": "not-int"},
            },
        )

    with httpx.Client(transport=httpx.MockTransport(sec_missing_handler)) as client:
        assert fetch_sec_ticker_map(client, config) == {}
        with pytest.raises(ValueError, match="SEC ticker map missing tickers"):
            fetch_sec_submission_payloads(tickers=["AAPL"], config=config, client=client)

    def massive_handler(request: httpx.Request) -> httpx.Response:
        if request.url.params["form_type"] == "8-K/A":
            return httpx.Response(200, json={"results": "bad"})
        return httpx.Response(
            200,
            json={
                "results": [
                    "bad",
                    {"filing_date": "bad"},
                    {"filing_date": "2025-01-01", "accession_number": "old"},
                    {"filing_date": "2026-01-29", "accession_number": "keep"},
                ],
                "next_url": "",
            },
        )

    with httpx.Client(transport=httpx.MockTransport(massive_handler)) as client:
        payloads = fetch_massive_8k_text_payloads(
            tickers=["AAPL"],
            config=config,
            client=client,
            start_date=date(2026, 1, 1),
            end_date=date(2026, 12, 31),
        )

    assert payloads["AAPL"]["results"] == [
        {"filing_date": "2026-01-29", "accession_number": "keep"}
    ]

    monkeypatch.setenv("MASSIVE_MAX_RETRIES", "1")
    monkeypatch.setenv("MASSIVE_RETRY_BACKOFF_SECONDS", "0")
    retry_config = load_project_config()

    def disconnecting_massive_handler(request: httpx.Request) -> httpx.Response:
        raise httpx.RemoteProtocolError("server disconnected")

    with httpx.Client(transport=httpx.MockTransport(disconnecting_massive_handler)) as client:
        failed_payloads = fetch_massive_8k_text_payloads(
            tickers=["AAPL"],
            config=retry_config,
            client=client,
            start_date=date(2026, 1, 1),
            end_date=date(2026, 12, 31),
        )

    assert failed_payloads["AAPL"]["results"] == [
        {
            "fetch_failure": "massive_8k_text_transport_retry_exhausted",
            "ticker": "AAPL",
            "form_type": "8-K",
            "error": "server disconnected",
        }
    ]

    monkeypatch.setenv("MASSIVE_MAX_RETRIES", "0")
    retry_config = load_project_config()

    def retryable_http_handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(503, json={"error": "try later"})

    with httpx.Client(transport=httpx.MockTransport(retryable_http_handler)) as client:
        http_failed_payloads = fetch_massive_8k_text_payloads(
            tickers=["AAPL"],
            config=retry_config,
            client=client,
            start_date=date(2026, 1, 1),
            end_date=date(2026, 12, 31),
        )

    assert {item["fetch_failure"] for item in http_failed_payloads["AAPL"]["results"]} == {
        "massive_8k_text_http_retry_exhausted"
    }


def test_act_365_default_and_mixed_convention_rejected() -> None:
    assert year_fraction(365) == 1.0
    assert year_fraction(0) == 0.0
    assert year_fraction(252, TimeConvention.TRADING_252) == 1.0
    with pytest.raises(ValueError, match="Vendor time convention"):
        year_fraction(30, TimeConvention.VENDOR)

    points = [
        TotalVariancePoint(date(2026, 2, 13), 0.80, 10, TimeConvention.ACT_365),
        TotalVariancePoint(date(2026, 2, 20), 0.60, 17, TimeConvention.TRADING_252),
    ]
    with pytest.raises(ValueError, match="Mixed time-to-expiry"):
        extract_implied_event_variance(points, event_date=date(2026, 2, 5))


def test_rvar_and_ivar_formula_and_failure_codes() -> None:
    rvar = realized_event_variance(100.0, 110.0)
    assert rvar == pytest.approx(0.0090840304)
    with pytest.raises(ValueError, match="S_before and S_after"):
        realized_event_variance(0.0, 110.0)

    points = [
        TotalVariancePoint(date(2026, 2, 13), 0.80, 10),
        TotalVariancePoint(date(2026, 2, 20), 0.68, 17),
    ]
    result = extract_implied_event_variance(points, event_date=date(2026, 2, 5))
    t1 = 10 / 365
    t2 = 17 / 365
    w1 = 0.80**2 * t1
    w2 = 0.68**2 * t2
    assert result.failure_reason is None
    assert result.ivar_event == pytest.approx((t2 * w1 - t1 * w2) / (t2 - t1))

    nonmonotone = extract_implied_event_variance(
        [
            TotalVariancePoint(date(2026, 2, 13), 1.00, 10),
            TotalVariancePoint(date(2026, 2, 20), 0.40, 17),
        ],
        event_date=date(2026, 2, 5),
    )
    assert nonmonotone.failure_reason == IVARFailureReason.NONMONOTONE_TOTAL_VARIANCE

    negative = extract_implied_event_variance(
        [
            TotalVariancePoint(date(2026, 2, 13), 0.30, 10),
            TotalVariancePoint(date(2026, 2, 20), 0.50, 17),
        ],
        event_date=date(2026, 2, 5),
    )
    assert negative.failure_reason == IVARFailureReason.NEGATIVE_EXTRACTED_IVAR


def test_ivar_additional_failure_codes_and_diagnostics() -> None:
    event_date = date(2026, 2, 5)
    too_few = extract_implied_event_variance(
        [TotalVariancePoint(date(2026, 2, 13), 0.80, 8)], event_date=event_date
    )
    assert too_few.failure_reason == IVARFailureReason.NO_TWO_EVENT_COVERING_EXPIRIES

    stale = extract_implied_event_variance(
        [
            TotalVariancePoint(date(2026, 2, 13), 0.80, 8, stale=True),
            TotalVariancePoint(date(2026, 2, 20), 0.70, 15),
        ],
        event_date=event_date,
    )
    assert stale.failure_reason == IVARFailureReason.STALE_OR_MISSING_IV

    nonpositive_gap = extract_implied_event_variance(
        [
            TotalVariancePoint(date(2026, 2, 13), 0.80, 0),
            TotalVariancePoint(date(2026, 2, 20), 0.70, 15),
        ],
        event_date=event_date,
    )
    assert nonpositive_gap.failure_reason == IVARFailureReason.NONPOSITIVE_TIME_GAP

    nonpositive_total_variance = extract_implied_event_variance(
        [
            TotalVariancePoint(date(2026, 2, 13), 0.0, 8),
            TotalVariancePoint(date(2026, 2, 20), 0.70, 15),
        ],
        event_date=event_date,
    )
    assert nonpositive_total_variance.failure_reason == IVARFailureReason.NONPOSITIVE_TOTAL_VARIANCE

    diagnostics = negative_ivar_diagnostics(
        [
            {
                "ticker": "ABC",
                "event_date": event_date,
                "expiration_1": date(2026, 2, 13),
                "expiration_2": date(2026, 2, 20),
                "failure_reason": "negative_extracted_ivar",
            }
        ]
    )
    assert list(diagnostics.columns)[0] == "ticker"
    assert diagnostics["failure_reason"].iloc[0] == "negative_extracted_ivar"


def test_ivar_failure_report_keeps_raw_ivs() -> None:
    event_date = date(2026, 2, 5)
    result = extract_implied_event_variance(
        [
            TotalVariancePoint(
                date(2026, 2, 13),
                None,
                8,
                spread_over_mid=0.10,
            ),
            TotalVariancePoint(
                date(2026, 2, 20),
                0.70,
                15,
                spread_over_mid=0.12,
            ),
        ],
        event_date=event_date,
    )

    assert result.failure_reason == IVARFailureReason.STALE_OR_MISSING_IV
    assert result.iv_used_for_extraction_1 is None
    assert result.iv_used_for_extraction_2 == pytest.approx(0.70)
    assert result.dte_1 == 8
    assert result.dte_2 == 15
    assert result.expiration_1 == date(2026, 2, 13)
    assert result.expiration_2 == date(2026, 2, 20)
    assert result.spread_over_mid_1 == pytest.approx(0.10)
    assert result.spread_over_mid_2 == pytest.approx(0.12)
    assert result.expiry_gap_days == 7

    diagnostics = negative_ivar_diagnostics(
        [
            {
                "ticker": "ABC",
                "event_date": event_date,
                "expiration_1": result.expiration_1,
                "expiration_2": result.expiration_2,
                "dte_1": result.dte_1,
                "dte_2": result.dte_2,
                "iv_used_for_extraction_1": result.iv_used_for_extraction_1,
                "iv_used_for_extraction_2": result.iv_used_for_extraction_2,
                "spread_over_mid_1": result.spread_over_mid_1,
                "spread_over_mid_2": result.spread_over_mid_2,
                "expiry_gap_days": result.expiry_gap_days,
                "failure_reason": result.failure_reason.value,
            }
        ]
    )
    assert diagnostics["iv_used_for_extraction_2"].iloc[0] == pytest.approx(0.70)
    assert diagnostics["expiry_gap_days"].iloc[0] == 7


def test_ivar_default_excludes_event_date_expiry() -> None:
    event_date = date(2026, 2, 5)
    points = [
        TotalVariancePoint(event_date, 0.95, 1),
        TotalVariancePoint(date(2026, 2, 12), 0.80, 7),
        TotalVariancePoint(date(2026, 2, 19), 0.70, 14),
    ]

    result = extract_implied_event_variance(points, event_date=event_date)

    assert result.failure_reason is None
    assert result.expiration_gap_days == 7
    assert result.t1 == pytest.approx(7 / 365)


def test_non_standard_contracts_excluded() -> None:
    events = pd.DataFrame(
        {
            "event_id": ["ABC_2026Q1"],
            "ticker": ["ABC"],
            "entry_date": ["2026-02-05"],
            "exit_date": ["2026-02-06"],
        }
    )
    contracts = pd.DataFrame(
        {
            "ticker": ["ABC", "ABC", "ABC"],
            "expiration": ["2026-02-13", "2026-02-13", "2026-02-13"],
            "strike": [100, 105, 110],
            "right": ["call", "call", "put"],
            "options_ticker": [
                "O:ABC260213C00100000",
                "O:ABC260213C00105000",
                "O:ABC260213P00110000",
            ],
            "option_multiplier": [150, 100, 100],
            "contract_size": [100, 100, 100],
            "deliverable_status": ["standard", "non_standard", "standard"],
            "corporate_action_flag": [False, False, True],
        }
    )

    discovered = discover_option_contracts(events, contracts)

    assert discovered["contract_discovery_status"].tolist() == [
        CONTRACT_STATUS_NON_STANDARD_EXCLUDED,
        CONTRACT_STATUS_NON_STANDARD_EXCLUDED,
        CONTRACT_STATUS_NON_STANDARD_EXCLUDED,
    ]
    assert discovered["eligible_for_quote_pool"].eq(False).all()


def test_standard_contracts_enter_quote_pool() -> None:
    events = pd.DataFrame(
        {
            "event_id": ["ABC_2026Q1"],
            "ticker": ["ABC"],
            "entry_date": ["2026-02-05"],
            "exit_date": ["2026-02-06"],
        }
    )
    contracts = pd.DataFrame(
        {
            "ticker": ["ABC"],
            "expiration": ["2026-02-13"],
            "strike": [100],
            "right": ["call"],
            "options_ticker": ["O:ABC260213C00100000"],
            "option_multiplier": [100],
            "contract_size": [100],
            "deliverable_status": ["standard"],
            "corporate_action_flag": [False],
        }
    )

    discovered = discover_option_contracts(events, contracts)

    assert discovered["contract_discovery_status"].tolist() == [CONTRACT_STATUS_OK]
    assert discovered["eligible_for_quote_pool"].tolist() == [True]
    assert discovered["dte"].iloc[0] == 8
    assert discovered["is_main_dte_5_14"].tolist() == [True]
    assert discovered["is_robustness_dte_3_21"].tolist() == [True]
    assert bool(discovered["covers_event_window"].iloc[0]) is True


def test_contract_reference_validation_excludes_adjusted_deliverables() -> None:
    candidates = pd.DataFrame(
        {
            "event_id": ["evt1", "evt1", "evt2"],
            "options_ticker": [
                "O:ABC260213C00100000",
                "O:ABC260213P00100000",
                "O:XYZ260213C00050000",
            ],
            "option_multiplier": [100, 100, 100],
            "contract_size": [100, 100, 100],
            "deliverable_status": ["standard", "standard", "standard"],
            "corporate_action_flag": [False, False, False],
            "contract_discovery_status": ["ok", "ok", "ok"],
            "eligible_for_quote_pool": [True, True, True],
            "is_main_dte_5_14": [True, True, True],
            "is_robustness_dte_3_21": [True, True, True],
        }
    )
    reference = pd.DataFrame(
        {
            "options_ticker": [
                "O:ABC260213C00100000",
                "O:ABC260213P00100000",
                "O:XYZ260213C00050000",
            ],
            "contract_reference_status": ["validated", "validated", "fetch_failed"],
            "contract_reference_error": [None, None, "HTTP 503"],
            "contract_reference_shares_per_contract": [100, 150, pd.NA],
            "contract_reference_additional_underlyings_count": [0, 1, 0],
            "contract_reference_has_adjusted_deliverable": [False, True, False],
            "contract_reference_exercise_style": ["american", "american", pd.NA],
            "contract_reference_correction": [0, 1, pd.NA],
        }
    )

    validated = apply_contract_reference_validation(candidates, reference)

    assert validated.loc[0, "option_multiplier"] == 100
    assert bool(validated.loc[0, "contract_reference_validated"]) is True
    assert bool(validated.loc[0, "eligible_for_quote_pool"]) is True
    assert validated.loc[1, "option_multiplier"] == 150
    assert validated.loc[1, "contract_size"] == 150
    assert validated.loc[1, "deliverable_status"] == "non_standard"
    assert validated.loc[1, "contract_discovery_status"] == CONTRACT_STATUS_NON_STANDARD_EXCLUDED
    assert bool(validated.loc[1, "eligible_for_quote_pool"]) is False
    assert bool(validated.loc[1, "is_main_dte_5_14"]) is False
    assert validated.loc[2, "contract_reference_status"] == "fetch_failed"
    assert bool(validated.loc[2, "contract_reference_validated"]) is False
    assert bool(validated.loc[2, "eligible_for_quote_pool"]) is False
    assert bool(validated.loc[2, "is_main_dte_5_14"]) is False
    assert validated.loc[2, "contract_discovery_status"] == (
        "contract_reference_unvalidated_excluded"
    )


def test_contract_reference_helpers_parse_cache_and_fallback_endpoint(
    tmp_path: Path,
) -> None:
    payload = {
        "results": [
            {"ticker": "O:OTHER", "shares_per_contract": 100},
            {
                "ticker": "O:ABC260213C00100000",
                "shares_per_contract": 150,
                "additional_underlyings": [{"ticker": "XYZ"}],
                "exercise_style": "american",
                "correction": 1,
            },
        ]
    }
    result = contract_reference.ContractReferenceFetchResult(
        options_ticker="O:ABC260213C00100000",
        fetch_status="hit",
        contract_reference_status="validated",
        payload=payload,
        error=None,
        cache_path="cache.json",
    )

    row = result.report_row()

    assert row["contract_reference_shares_per_contract"] == 150
    assert row["contract_reference_additional_underlyings_count"] == 1
    assert row["contract_reference_has_adjusted_deliverable"] is True
    assert (
        contract_reference.contract_reference_cache_path(tmp_path, "O:ABC260213C00100000")
        .as_posix()
        .endswith("options_ticker=O_ABC260213C00100000/reference.json")
    )
    assert (
        contract_reference.extract_contract_reference_fields("O:ABC", {})[
            "contract_reference_has_adjusted_deliverable"
        ]
        is False
    )

    config = replace(
        load_project_config(),
        massive_api_key_file=tmp_path / "massive_api_key.txt",
        massive_base_url="https://api.massive.test",
    )
    assert config.massive_api_key_file is not None
    config.massive_api_key_file.write_text("key", encoding="utf-8")
    seen_urls: list[str] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen_urls.append(str(request.url))
        if request.url.path.rstrip("/") != "/v3/reference/options/contracts":
            return httpx.Response(404, json={"status": "NOT_FOUND"})
        return httpx.Response(200, json=payload)

    with httpx.Client(transport=httpx.MockTransport(handler)) as client:
        fetched = contract_reference.fetch_massive_option_contract_reference(
            client,
            config,
            options_ticker="O:ABC260213C00100000",
            cache_root=tmp_path / "cache",
        )

    assert fetched.fetch_status == "downloaded"
    assert fetched.contract_reference_status == "validated"
    assert any("/v3/reference/options/contracts?" in url for url in seen_urls)
    fallback_url = next(url for url in seen_urls if "/v3/reference/options/contracts?" in url)
    parsed_query = urllib.parse.parse_qs(urllib.parse.urlparse(fallback_url).query)
    assert parsed_query["ticker"] == ["O:ABC260213C00100000"]
    assert parsed_query["expired"] == ["true"]
    assert parsed_query["as_of"] == ["2026-02-13"]
    assert Path(str(fetched.cache_path)).exists()

    with httpx.Client(transport=httpx.MockTransport(handler)) as client:
        cached = contract_reference.fetch_massive_option_contract_reference(
            client,
            config,
            options_ticker="O:ABC260213C00100000",
            cache_root=tmp_path / "cache",
        )
    assert cached.fetch_status == "hit"


def test_contract_reference_handles_missing_parse_and_failed_fetch(
    tmp_path: Path,
) -> None:
    assert (
        contract_reference.extract_contract_reference_fields("O:ABC", None)[
            "contract_reference_has_adjusted_deliverable"
        ]
        is False
    )
    missing_fields = contract_reference.extract_contract_reference_fields("O:ABC", {"results": []})
    missing_shares = cast(float, missing_fields["contract_reference_shares_per_contract"])
    assert np.isnan(missing_shares)
    direct = contract_reference.extract_contract_reference_fields(
        "O:ABC",
        {"shares_per_contract": 100, "additional_underlyings": {"cash": 1}},
    )
    assert direct["contract_reference_additional_underlyings_count"] == 1
    mismatched_list = contract_reference.extract_contract_reference_fields(
        "O:ABC",
        {"results": [{"ticker": "O:OTHER", "shares_per_contract": 100}]},
    )
    assert np.isnan(cast(float, mismatched_list["contract_reference_shares_per_contract"]))
    assert contract_reference._expiration_from_options_ticker("bad") is None
    assert contract_reference._safe_exception_text(RuntimeError("apiKey=key failed")) == (
        "apiKey=<redacted> failed"
    )
    assert contract_reference._additional_underlyings_count(None) == 0
    assert contract_reference._additional_underlyings_count([]) == 0
    assert contract_reference._additional_underlyings_count("[]") == 0
    assert contract_reference._additional_underlyings_count("cash") == 1
    assert contract_reference._additional_underlyings_count(1) == 1

    key_file = tmp_path / "massive_api_key.txt"
    key_file.write_text("key", encoding="utf-8")
    config = replace(
        load_project_config(),
        massive_api_key_file=key_file,
        massive_base_url="https://api.massive.test",
    )
    cache_root = tmp_path / "cache"
    missing_cache = contract_reference.contract_reference_cache_path(cache_root, "O:MISS")
    missing_cache.parent.mkdir(parents=True)
    missing_cache.write_text('{"results": []}', encoding="utf-8")
    parse_cache = contract_reference.contract_reference_cache_path(cache_root, "O:PARSE")
    parse_cache.parent.mkdir(parents=True)
    parse_cache.write_text('{"results": {"ticker": "O:PARSE"}}', encoding="utf-8")
    bad_cache = contract_reference.contract_reference_cache_path(cache_root, "O:BAD")
    bad_cache.parent.mkdir(parents=True)
    bad_cache.write_text("{bad-json", encoding="utf-8")

    with httpx.Client(transport=httpx.MockTransport(lambda request: httpx.Response(401))) as client:
        missing = contract_reference.fetch_massive_option_contract_reference(
            client,
            config,
            options_ticker="O:MISS",
            cache_root=cache_root,
        )
        parse_failed = contract_reference.fetch_massive_option_contract_reference(
            client,
            config,
            options_ticker="O:PARSE",
            cache_root=cache_root,
        )
        failed = contract_reference.fetch_massive_option_contract_reference(
            client,
            config,
            options_ticker="O:BAD",
            cache_root=cache_root,
        )

    assert missing.contract_reference_status == "missing_reference"
    assert parse_failed.contract_reference_status == "parse_failed"
    assert failed.contract_reference_status == "fetch_failed"

    def secret_error_handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            403,
            text="denied for apiKey=key",
            request=request,
        )

    with httpx.Client(transport=httpx.MockTransport(secret_error_handler)) as client:
        failed_secret = contract_reference.fetch_massive_option_contract_reference(
            client,
            config,
            options_ticker="O:SECRET260213C00100000",
            cache_root=cache_root,
            refresh_bronze=True,
        )
    assert failed_secret.contract_reference_status == "fetch_failed"
    assert failed_secret.error is not None
    assert "apiKey=<redacted>" in failed_secret.error
    assert "apiKey=key" not in failed_secret.error

    no_key_config = replace(load_project_config(), massive_api_key_file=None)
    ok_transport = httpx.MockTransport(lambda request: httpx.Response(200))
    with (
        pytest.raises(ValueError, match="MASSIVE_API_KEY_FILE"),
        httpx.Client(transport=ok_transport) as client,
    ):
        contract_reference.fetch_massive_option_contract_reference(
            client,
            no_key_config,
            options_ticker="O:NOKEY",
            cache_root=tmp_path / "no_key_cache",
            refresh_bronze=True,
        )


def test_contract_reference_validation_empty_and_invalid_reports() -> None:
    candidates = pd.DataFrame({"options_ticker": ["O:ABC260213C00100000"]})

    empty = apply_contract_reference_validation(candidates, pd.DataFrame())

    assert empty["contract_reference_status"].tolist() == ["not_requested"]
    assert empty["contract_discovery_status"].isna().all()
    assert "eligible_for_quote_pool" not in empty.columns
    gated_empty = apply_contract_reference_validation(
        pd.DataFrame(
            {
                "options_ticker": ["O:ABC260213C00100000"],
                "contract_discovery_status": ["ok"],
                "eligible_for_quote_pool": [True],
                "is_main_dte_5_14": [True],
                "is_robustness_dte_3_21": [True],
            }
        ),
        pd.DataFrame(),
    )
    assert bool(gated_empty["eligible_for_quote_pool"].iloc[0]) is False
    assert bool(gated_empty["is_main_dte_5_14"].iloc[0]) is False
    assert gated_empty["contract_discovery_status"].iloc[0] == (
        "contract_reference_unvalidated_excluded"
    )
    minimal = apply_contract_reference_validation(
        candidates,
        pd.DataFrame(
            {
                "options_ticker": ["O:ABC260213C00100000"],
                "contract_reference_status": ["validated"],
                "contract_reference_shares_per_contract": [100],
            }
        ),
    )
    assert minimal["deliverable_status"].tolist() == ["standard"]
    assert minimal["corporate_action_flag"].tolist() == [False]
    missing_reference_standard = apply_contract_reference_validation(
        pd.DataFrame(
            {
                "options_ticker": ["O:ABC260213C00100000"],
                "contract_discovery_status": ["ok"],
                "eligible_for_quote_pool": [True],
                "is_main_dte_5_14": [True],
                "is_robustness_dte_3_21": [True],
                "option_multiplier": [100],
                "contract_size": [100],
            }
        ),
        pd.DataFrame(
            {
                "options_ticker": ["O:ABC260213C00100000"],
                "contract_reference_status": ["missing_reference"],
                "contract_reference_has_adjusted_deliverable": [False],
            }
        ),
    )
    assert bool(missing_reference_standard["contract_reference_validated"].iloc[0]) is False
    assert bool(missing_reference_standard["contract_reference_proxy_usable"].iloc[0]) is True
    assert bool(missing_reference_standard["eligible_for_quote_pool"].iloc[0]) is True
    failed_minimal = apply_contract_reference_validation(
        pd.DataFrame(
            {
                "options_ticker": ["O:ABC260213C00100000"],
                "contract_discovery_status": ["ok"],
                "eligible_for_quote_pool": [True],
            }
        ),
        pd.DataFrame(
            {
                "options_ticker": ["O:ABC260213C00100000"],
                "contract_reference_status": ["fetch_failed"],
            }
        ),
    )
    assert bool(failed_minimal["eligible_for_quote_pool"].iloc[0]) is False
    with pytest.raises(ValueError, match="missing columns"):
        apply_contract_reference_validation(
            candidates,
            pd.DataFrame({"options_ticker": ["O:ABC260213C00100000"]}),
        )
    with pytest.raises(ValueError, match="missing options_ticker"):
        apply_contract_reference_validation(pd.DataFrame({"ticker": ["ABC"]}), pd.DataFrame())


def test_contract_reference_validation_pipeline_step_updates_candidates(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = replace(
        load_project_config(),
        silver_data_dir=tmp_path / "silver",
        bronze_data_dir=tmp_path / "bronze",
    )
    candidate_path = config.silver_data_dir / "contracts" / "event_contract_candidates.parquet"
    candidate_path.parent.mkdir(parents=True)
    candidates = pd.DataFrame(
        {
            "event_id": ["evt1", "evt1"],
            "options_ticker": ["O:AAA260213C00100000", "O:AAA260213P00100000"],
            "option_multiplier": [100, 100],
            "contract_size": [100, 100],
            "deliverable_status": ["standard", "standard"],
            "corporate_action_flag": [False, False],
            "contract_discovery_status": ["ok", "ok"],
            "eligible_for_quote_pool": [True, True],
            "is_main_dte_5_14": [True, True],
            "is_robustness_dte_3_21": [True, True],
        }
    )
    pl.from_pandas(candidates).write_parquet(candidate_path)

    def fake_fetch(
        client: httpx.Client,
        config: object,
        *,
        options_ticker: str,
        cache_root: Path,
        refresh_bronze: bool = False,
    ) -> contract_reference.ContractReferenceFetchResult:
        del client, config, cache_root, refresh_bronze
        shares = 150 if options_ticker.endswith("P00100000") else 100
        return contract_reference.ContractReferenceFetchResult(
            options_ticker=options_ticker,
            fetch_status="downloaded",
            contract_reference_status="validated",
            payload={
                "results": {
                    "ticker": options_ticker,
                    "shares_per_contract": shares,
                    "additional_underlyings": [{"ticker": "CASH"}] if shares != 100 else [],
                    "exercise_style": "american",
                    "correction": 0,
                }
            },
        )

    monkeypatch.setattr(
        "earnings_event_vol.data_pipeline.fetch_massive_option_contract_reference",
        fake_fetch,
    )

    step = data_pipeline._contract_reference_validation_step(
        config,
        out_root=tmp_path / "artifacts",
        force=False,
        jobs=2,
        max_contracts=None,
        refresh_bronze=False,
    )

    assert step.status == "ran"
    validated = pd.read_parquet(candidate_path)
    assert validated["contract_reference_validated"].tolist() == [True, True]
    assert validated["option_multiplier"].tolist() == [100, 150]
    assert validated["contract_discovery_status"].tolist() == [
        "ok",
        CONTRACT_STATUS_NON_STANDARD_EXCLUDED,
    ]
    assert validated["eligible_for_quote_pool"].tolist() == [True, False]
    assert (
        tmp_path
        / "artifacts"
        / "contract_reference_validation"
        / "contract_reference_validation_manifest.json"
    ).exists()

    resume = data_pipeline._contract_reference_validation_step(
        config,
        out_root=tmp_path / "artifacts",
        force=False,
        jobs=2,
        max_contracts=None,
        refresh_bronze=False,
    )
    assert resume.status == "skipped"


def test_contract_reference_validation_reuses_existing_reference_report(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = replace(
        load_project_config(),
        silver_data_dir=tmp_path / "silver",
        bronze_data_dir=tmp_path / "bronze",
    )
    candidate_path = config.silver_data_dir / "contracts" / "event_contract_candidates.parquet"
    reference_path = config.silver_data_dir / "contracts" / "contract_reference_validation.parquet"
    candidate_path.parent.mkdir(parents=True)
    candidates = pd.DataFrame(
        {
            "event_id": ["evt1"],
            "options_ticker": ["O:AAA260213C00100000"],
            "option_multiplier": [100],
            "contract_size": [100],
            "contract_discovery_status": ["ok"],
            "eligible_for_quote_pool": [True],
        }
    )
    reference_report = pd.DataFrame(
        {
            "options_ticker": ["O:AAA260213C00100000"],
            "fetch_status": ["hit"],
            "contract_reference_status": ["missing_reference"],
            "contract_reference_has_adjusted_deliverable": [False],
        }
    )
    pl.from_pandas(candidates).write_parquet(candidate_path)
    pl.from_pandas(reference_report).write_parquet(reference_path)

    def fail_fetch(*args: object, **kwargs: object) -> object:
        raise AssertionError("existing reference report should be reused")

    monkeypatch.setattr(
        "earnings_event_vol.data_pipeline.fetch_massive_option_contract_reference",
        fail_fetch,
    )

    step = data_pipeline._contract_reference_validation_step(
        config,
        out_root=tmp_path / "artifacts",
        force=False,
        jobs=1,
        max_contracts=None,
        refresh_bronze=False,
    )

    assert step.status == "ran"
    validated = pd.read_parquet(candidate_path)
    assert bool(validated["contract_reference_proxy_usable"].iloc[0]) is True
    assert bool(validated["eligible_for_quote_pool"].iloc[0]) is True
    assert step.metadata["reused_reference_report"] is True


def test_contract_reference_validation_pipeline_step_blocks_without_candidates(
    tmp_path: Path,
) -> None:
    config = replace(load_project_config(), silver_data_dir=tmp_path / "silver")

    step = data_pipeline._contract_reference_validation_step(
        config,
        out_root=tmp_path / "artifacts",
        force=False,
        jobs=1,
        max_contracts=None,
        refresh_bronze=False,
    )

    assert step.status == "blocked"
    assert step.reason == "requires event_contract_candidates.parquet"


def test_forward_parity_requires_no_dividend_window() -> None:
    quotes = pd.DataFrame(
        {
            "expiration": ["2026-02-13", "2026-02-13"],
            "strike": [100, 100],
            "right": ["call", "put"],
            "bid": [4.9, 4.4],
            "ask": [5.1, 4.6],
        }
    )

    selection = select_forward_and_atm(
        quotes,
        entry_date=date(2026, 2, 5),
        spot=100.0,
        second_ivar_expiry=date(2026, 2, 20),
        ex_dividend_dates=[date(2026, 2, 10)],
    )

    assert selection.forward_source == FORWARD_SOURCE_SPOT_FALLBACK
    assert selection.forward_price == pytest.approx(100.0)
    assert selection.american_forward_caveat_flag is False


def test_forward_spot_fallback_flagged() -> None:
    quotes = pd.DataFrame(
        {
            "expiration": ["2026-02-13"],
            "strike": [100],
            "right": ["call"],
            "bid": [4.9],
            "ask": [5.1],
        }
    )

    selection = select_forward_and_atm(
        quotes,
        entry_date=date(2026, 2, 5),
        spot=100.0,
        second_ivar_expiry=date(2026, 2, 20),
    )

    assert selection.forward_source == FORWARD_SOURCE_SPOT_FALLBACK
    assert selection.atm_selection_method == "nearest_spot_atm"


def test_forward_parity_records_american_caveat_when_used() -> None:
    quotes = pd.DataFrame(
        {
            "expiration": ["2026-02-13", "2026-02-13"],
            "strike": [100, 100],
            "right": ["call", "put"],
            "bid": [4.9, 4.4],
            "ask": [5.1, 4.6],
        }
    )

    selection = select_forward_and_atm(
        quotes,
        entry_date=date(2026, 2, 5),
        spot=100.0,
        second_ivar_expiry=date(2026, 2, 20),
    )

    assert selection.forward_source == FORWARD_SOURCE_PUT_CALL_PARITY
    assert selection.forward_price == pytest.approx(100.5)
    assert selection.atm_selection_method == "ATMF"
    assert selection.american_forward_caveat_flag is True


def test_possible_preannouncement_tag() -> None:
    frame = pd.DataFrame(
        {
            "ticker": ["AAA", "BBB"],
            "rvar_event": [0.001, 0.009],
            "ivar_event": [0.04, 0.010],
        }
    )

    flagged = flag_possible_preannouncement_or_prior_guidance(frame)

    assert flagged["possible_preannouncement_or_prior_guidance"].tolist() == [True, False]
    assert len(flagged) == 2


def test_build_event_panel_outputs_forward_and_preannouncement_fields() -> None:
    events = pd.DataFrame(
        {
            "event_id": ["ABC_2026Q1"],
            "ticker": ["ABC"],
            "entry_date": ["2026-02-05"],
            "spot": [100.0],
            "rvar_event": [0.001],
            "ivar_event": [0.04],
        }
    )
    quotes = pd.DataFrame(
        {
            "event_id": ["ABC_2026Q1", "ABC_2026Q1"],
            "ticker": ["ABC", "ABC"],
            "expiration": ["2026-02-13", "2026-02-13"],
            "strike": [100, 100],
            "right": ["call", "put"],
            "bid": [4.9, 4.4],
            "ask": [5.1, 4.6],
        }
    )

    panel = build_event_panel(events, quotes)

    assert panel["forward_source"].iloc[0] == FORWARD_SOURCE_PUT_CALL_PARITY
    assert panel["atm_selection_method"].iloc[0] == "ATMF"
    assert bool(panel["american_forward_caveat_flag"].iloc[0]) is True
    assert bool(panel["possible_preannouncement_or_prior_guidance"].iloc[0]) is True


def test_event_panel_fail_closed_and_fallback_branches() -> None:
    base_event = pd.DataFrame({"ticker": ["ABC"], "entry_date": ["2026-02-05"]})
    standard_contract = pd.DataFrame(
        {
            "ticker": ["ABC"],
            "expiration": ["2026-02-13"],
            "strike": [100],
            "right": ["call"],
            "option_symbol": ["O:ABC260213C00100000"],
        }
    )

    discovered = discover_option_contracts(base_event, standard_contract)

    assert discovered["event_id"].iloc[0] == "ABC_2026-02-05"
    assert discovered["options_ticker"].iloc[0] == "O:ABC260213C00100000"
    assert discovered["option_multiplier"].iloc[0] == 100
    assert discovered["contract_size"].iloc[0] == 100
    assert discovered["deliverable_status"].iloc[0] == "standard"
    assert bool(discovered["corporate_action_flag"].iloc[0]) is False

    with pytest.raises(ValueError, match="event_id values must be non-null"):
        discover_option_contracts(
            pd.DataFrame(
                {
                    "event_id": [""],
                    "ticker": ["ABC"],
                    "entry_date": ["2026-02-05"],
                }
            ),
            standard_contract,
        )
    with pytest.raises(ValueError, match="event_id values must be unique"):
        discover_option_contracts(
            pd.DataFrame(
                {
                    "event_id": ["ABC_DUP", "ABC_DUP"],
                    "ticker": ["ABC", "ABC"],
                    "entry_date": ["2026-02-05", "2026-02-06"],
                }
            ),
            standard_contract,
        )
    with pytest.raises(ValueError, match="event_id values must be unique"):
        discover_option_contracts(
            pd.DataFrame(
                {
                    "ticker": ["ABC", "ABC"],
                    "entry_date": ["2026-02-05", "2026-02-05"],
                }
            ),
            standard_contract,
        )

    status_cases = discover_option_contracts(
        pd.DataFrame(
            {
                "event_id": ["ABC_SHORT", "ABC_LONG", "DEF_MISSING"],
                "ticker": ["ABC", "ABC", "DEF"],
                "entry_date": ["2026-02-05", "2026-02-05", "2026-02-05"],
                "exit_date": ["2026-02-06", "2026-02-20", "2026-02-06"],
            }
        ),
        pd.DataFrame(
            {
                "ticker": ["ABC", "ABC"],
                "expiration": ["2026-03-20", "2026-02-13"],
                "strike": [100, 100],
                "right": ["call", "put"],
                "deliverable_status": [None, pd.NA],
                "corporate_action_flag": [None, "no"],
            }
        ),
    )
    assert status_cases["contract_discovery_status"].tolist() == [
        CONTRACT_STATUS_OUTSIDE_DTE_RANGE,
        CONTRACT_STATUS_OK,
        CONTRACT_STATUS_OUTSIDE_DTE_RANGE,
        CONTRACT_STATUS_DOES_NOT_COVER_EVENT,
        CONTRACT_STATUS_MISSING_METADATA,
    ]

    with pytest.raises(ValueError, match="DTE range"):
        discover_option_contracts(base_event, standard_contract, dte_min=10, dte_max=5)
    with pytest.raises(ValueError, match="contract frame missing required columns"):
        discover_option_contracts(base_event, pd.DataFrame({"ticker": ["ABC"]}))
    with pytest.raises(ValueError, match="spot must be positive"):
        select_forward_and_atm(
            pd.DataFrame(),
            entry_date=date(2026, 2, 5),
            spot=0,
            second_ivar_expiry=date(2026, 2, 20),
        )

    empty_selection = select_forward_and_atm(
        pd.DataFrame(),
        entry_date=date(2026, 2, 5),
        spot=100,
        second_ivar_expiry=date(2026, 2, 20),
    )
    assert empty_selection.forward_source == FORWARD_SOURCE_SPOT_FALLBACK

    wide_quote_selection = select_forward_and_atm(
        pd.DataFrame(
            {
                "expiration": ["2026-02-13", "2026-02-13"],
                "strike": [100, 100],
                "right": ["call", "put"],
                "bid": [1.0, 1.0],
                "ask": [3.0, 3.0],
                "spread_over_mid": [1.0, 1.0],
            }
        ),
        entry_date=date(2026, 2, 5),
        spot=100,
        second_ivar_expiry=date(2026, 2, 20),
    )
    assert wide_quote_selection.forward_source == FORWARD_SOURCE_SPOT_FALLBACK

    invalid_forward_selection = select_forward_and_atm(
        pd.DataFrame(
            {
                "expiration": ["2026-02-13", "2026-02-13"],
                "strike": [100, 100],
                "right": ["call", "put"],
                "bid": [0.09, 100.9],
                "ask": [0.11, 101.1],
            }
        ),
        entry_date=date(2026, 2, 5),
        spot=100,
        second_ivar_expiry=date(2026, 2, 20),
    )
    assert invalid_forward_selection.forward_source == FORWARD_SOURCE_SPOT_FALLBACK

    asof_selection = select_forward_and_atm(
        pd.DataFrame(
            {
                "expiration": ["2026-02-13", "2026-02-13", "2026-02-13", "2026-02-13"],
                "strike": [100, 100, 100, 100],
                "right": ["call", "put", "call", "put"],
                "bid": [4.9, 4.4, 6.9, 6.4],
                "ask": [5.1, 4.6, 7.1, 6.6],
                "quote_timestamp": [
                    "2026-02-05T20:55:00Z",
                    "2026-02-05T20:55:00Z",
                    "2026-02-06T20:55:00Z",
                    "2026-02-06T20:55:00Z",
                ],
                "quote_date": [
                    "2026-02-05",
                    "2026-02-05",
                    "2026-02-06",
                    "2026-02-06",
                ],
            }
        ),
        entry_date=date(2026, 2, 5),
        spot=100,
        second_ivar_expiry=date(2026, 2, 20),
    )
    assert asof_selection.forward_source == FORWARD_SOURCE_PUT_CALL_PARITY
    assert asof_selection.forward_price == pytest.approx(100.5)

    with pytest.raises(ValueError, match="Merge keys are not unique"):
        select_forward_and_atm(
            pd.DataFrame(
                {
                    "expiration": ["2026-02-13", "2026-02-13", "2026-02-13"],
                    "strike": [100, 100, 100],
                    "right": ["call", "call", "put"],
                    "bid": [4.9, 4.95, 4.4],
                    "ask": [5.1, 5.15, 4.6],
                }
            ),
            entry_date=date(2026, 2, 5),
            spot=100,
            second_ivar_expiry=date(2026, 2, 20),
        )

    no_label_panel = build_event_panel(
        pd.DataFrame(
            {
                "ticker": ["ABC"],
                "entry_date": ["2026-02-05"],
                "s_before": [100.0],
                "second_expiry": ["2026-02-20"],
            }
        ),
        pd.DataFrame(
            {
                "ticker": ["ABC", "ABC"],
                "expiration": ["2026-02-13", "2026-02-13"],
                "strike": [100, 100],
                "right": ["call", "put"],
                "bid": [4.9, 4.4],
                "ask": [5.1, 4.6],
            }
        ),
        ex_dividends=pd.DataFrame({"ticker": ["ABC"], "ex_dividend_date": ["2026-02-10"]}),
    )
    assert no_label_panel["forward_source"].iloc[0] == FORWARD_SOURCE_SPOT_FALLBACK
    assert bool(no_label_panel["possible_preannouncement_or_prior_guidance"].iloc[0]) is False

    with pytest.raises(ValueError, match="spot or s_before"):
        build_event_panel(base_event, pd.DataFrame())
    with pytest.raises(ValueError, match="event_id values must be unique"):
        build_event_panel(
            pd.DataFrame(
                {
                    "event_id": ["DUP", "DUP"],
                    "ticker": ["ABC", "ABC"],
                    "entry_date": ["2026-02-05", "2026-02-06"],
                    "spot": [100.0, 101.0],
                }
            ),
            pd.DataFrame(),
        )
    with pytest.raises(ValueError, match="spot values must be finite"):
        build_event_panel(
            pd.DataFrame(
                {
                    "event_id": ["BAD_SPOT"],
                    "ticker": ["ABC"],
                    "entry_date": ["2026-02-05"],
                    "spot": [np.nan],
                }
            ),
            pd.DataFrame(),
        )
    with pytest.raises(ValueError, match="high_ivar_quantile"):
        flag_possible_preannouncement_or_prior_guidance(
            pd.DataFrame({"rvar_event": [0.0], "ivar_event": [0.0]}),
            high_ivar_quantile=1.5,
        )
    empty_ivar_flags = flag_possible_preannouncement_or_prior_guidance(
        pd.DataFrame({"rvar_event": [0.0], "ivar_event": [0.0]})
    )
    assert bool(empty_ivar_flags["possible_preannouncement_or_prior_guidance"].iloc[0]) is False


def test_data_pipeline_resume_force_and_stage_outputs(tmp_path: Path) -> None:
    config = load_project_config()
    out_root = tmp_path / "pipeline"

    assert parse_text_list(["AAPL,MSFT", "NVDA TSLA"]) == ["AAPL", "MSFT", "NVDA", "TSLA"]
    assert parse_text_list(None) == []

    fixture_run = run_data_pipeline(config, stage="fixture-audit", out_root=out_root)
    assert fixture_run["ok"] is True
    assert _pipeline_steps(fixture_run)[0]["status"] == "ran"
    assert (out_root / "fixture_audit" / "required_fields_report.json").exists()

    fixture_resume = run_data_pipeline(config, stage="fixture-audit", out_root=out_root)
    assert _pipeline_steps(fixture_resume)[0]["status"] == "skipped"
    assert _pipeline_steps(fixture_resume)[0]["reason"] == "outputs_exist"

    fixture_force = run_data_pipeline(config, stage="fixture-audit", out_root=out_root, force=True)
    assert _pipeline_steps(fixture_force)[0]["status"] == "ran"

    lake_config = replace(
        config,
        bronze_data_dir=tmp_path / "lake_bronze",
        silver_data_dir=tmp_path / "lake_silver",
        gold_data_dir=tmp_path / "lake_gold",
        artifacts_dir=tmp_path / "lake_artifacts",
    )
    option_partition = (
        lake_config.bronze_data_dir
        / "massive"
        / "options_day_aggs"
        / "date=2022-12-01"
        / "part.parquet"
    )
    underlying_partition = (
        lake_config.bronze_data_dir
        / "massive"
        / "underlying_day_aggs"
        / "date=2022-12-01"
        / "part.parquet"
    )
    option_partition.parent.mkdir(parents=True)
    underlying_partition.parent.mkdir(parents=True)
    pl.DataFrame(
        {"ticker": ["O:AAA230120C00100000"], "close": [1.0], "volume": [10]}
    ).write_parquet(
        option_partition,
    )
    pl.DataFrame({"ticker": ["AAA"], "close": [100.0], "volume": [1000]}).write_parquet(
        underlying_partition,
    )
    calendar_path = lake_config.silver_data_dir / "earnings_calendar" / "main_sample.parquet"
    calendar_path.parent.mkdir(parents=True)
    pd.DataFrame(
        {
            "event_id": ["AAA_2022Q4"],
            "ticker": ["AAA"],
            "announcement_date": ["2022-12-01"],
        }
    ).to_parquet(calendar_path, index=False)
    lake_run = run_data_pipeline(
        lake_config,
        stage="lake-quality-audit",
        out_root=out_root,
        start_date=date(2013, 1, 1),
        end_date=date(2025, 12, 31),
    )
    assert lake_run["ok"] is True
    lake_step = _pipeline_steps(lake_run)[0]
    assert lake_step["status"] == "ran"
    report = json.loads(
        (out_root / "lake_quality_audit" / "lake_quality_report.json").read_text(encoding="utf-8")
    )
    assert report["ok"] is False
    assert "bronze_options_day_aggs" in report["incomplete_required_dataset_ids"]
    coverage = pd.read_csv(out_root / "lake_quality_audit" / "lake_dataset_coverage.csv")
    options_row = coverage.loc[coverage["dataset_id"].eq("bronze_options_day_aggs")].iloc[0]
    assert options_row["target_coverage_status"] == "target_span_incomplete"
    assert "history_starts_after_target" in str(options_row["gap_reason"])
    lake_resume = run_data_pipeline(
        lake_config,
        stage="lake-quality-audit",
        out_root=out_root,
        start_date=date(2013, 1, 1),
        end_date=date(2025, 12, 31),
    )
    assert _pipeline_steps(lake_resume)[0]["status"] == "skipped"

    massive_out = out_root / "massive_probe" / "2025-02-05"
    massive_out.mkdir(parents=True)
    (massive_out / "massive_flat_file_manifest.json").write_text("{}")
    massive_skip = run_data_pipeline(
        config,
        stage="massive-probe",
        out_root=out_root,
        dates=[date(2025, 2, 5)],
        jobs=2,
    )
    assert _pipeline_steps(massive_skip)[0]["status"] == "skipped"

    universe_source = tmp_path / "option_day_aggs.csv"
    universe_source.write_text(
        "\n".join(
            [
                "ticker,quote_date,option_close,volume",
                "AAA,2020-01-15,1.0,100",
                "BBB,2020-02-15,4.0,100",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    _write_eligible_equity_cache(
        out_root,
        [
            {"ticker": "AAA", "exchange": "NASDAQ", "title": "AAA Corporation"},
            {"ticker": "BBB", "exchange": "NYSE", "title": "BBB Corporation"},
        ],
    )
    universe_run = run_data_pipeline(
        config,
        stage="universe",
        out_root=out_root,
        options_day_aggs_path=universe_source,
        start_date=date(2020, 3, 1),
        end_date=date(2020, 3, 31),
        universe_top_n=1,
        universe_trailing_months=2,
    )
    assert universe_run["ok"] is True
    universe_step = _pipeline_steps(universe_run)[0]
    assert universe_step["status"] == "ran"
    assert (out_root / "universe" / "monthly_top50_universe.parquet").exists()
    universe_resume = run_data_pipeline(
        config,
        stage="universe",
        out_root=out_root,
        options_day_aggs_path=universe_source,
        start_date=date(2020, 3, 1),
        end_date=date(2020, 3, 31),
        universe_top_n=1,
        universe_trailing_months=2,
    )
    assert _pipeline_steps(universe_resume)[0]["status"] == "skipped"
    universe_blocked = run_data_pipeline(
        replace(config, bronze_data_dir=tmp_path / "empty_bronze"),
        stage="universe",
        out_root=tmp_path / "blocked",
    )
    assert universe_blocked["ok"] is False
    assert _pipeline_steps(universe_blocked)[0]["status"] == "blocked"

    market_config = replace(
        config,
        bronze_data_dir=tmp_path / "market_bronze",
        silver_data_dir=tmp_path / "market_silver",
    )
    market_raw = market_config.bronze_data_dir / "market_covariates" / "fred_vixcls.csv"
    market_raw.parent.mkdir(parents=True)
    market_raw.write_text("DATE,VIXCLS\n2024-01-02,12.5\n2024-01-03,.\n", encoding="utf-8")
    market_run = run_data_pipeline(
        market_config,
        stage="market-covariates",
        out_root=out_root,
    )
    assert market_run["ok"] is True
    assert _pipeline_steps(market_run)[0]["status"] == "ran"
    market_silver = (
        market_config.silver_data_dir / "market_covariates" / "daily_market_covariates.parquet"
    )
    assert market_silver.exists()
    market_resume = run_data_pipeline(
        market_config,
        stage="market-covariates",
        out_root=out_root,
    )
    assert _pipeline_steps(market_resume)[0]["status"] == "skipped"

    with pytest.raises(ValueError, match="unsupported data stage"):
        run_data_pipeline(config, stage="bad-stage", out_root=out_root)
    for removed_stage in ("contracts", "panel"):
        with pytest.raises(ValueError, match="unsupported data stage"):
            run_data_pipeline(config, stage=removed_stage, out_root=out_root)
    with pytest.raises(ValueError, match="jobs must be positive"):
        run_data_pipeline(config, stage="fixture-audit", out_root=out_root, jobs=0)
    with pytest.raises(ValueError, match="start_date"):
        run_data_pipeline(
            config,
            stage="dynamic-calendar",
            out_root=out_root,
            start_date=date(2026, 12, 31),
            end_date=date(2026, 1, 1),
        )
    with pytest.raises(ValueError, match="universe_top_n"):
        run_data_pipeline(config, stage="universe", out_root=out_root, universe_top_n=0)
    with pytest.raises(ValueError, match="universe_trailing_months"):
        run_data_pipeline(
            config,
            stage="universe",
            out_root=out_root,
            universe_trailing_months=0,
        )
    with pytest.raises(ValueError, match="quote_workers"):
        run_data_pipeline(config, stage="quote-execution-panel", out_root=out_root, quote_workers=0)
    with pytest.raises(ValueError, match="quote_event_offset"):
        run_data_pipeline(
            config,
            stage="quote-execution-panel",
            out_root=out_root,
            quote_event_offset=-1,
        )
    with pytest.raises(ValueError, match="quote_batch_label"):
        run_data_pipeline(
            config,
            stage="quote-execution-panel",
            out_root=out_root,
            quote_batch_label="../bad",
        )

    dry_run = run_data_pipeline(
        config,
        stage="all",
        out_root=out_root,
        tickers=["AAPL", "MSFT"],
        start_date=date(2026, 1, 1),
        end_date=date(2026, 3, 31),
        max_events=8,
        max_contracts=80,
        dry_run=True,
        quote_workers=4,
        quote_event_offset=8,
        quote_batch_label="offset8_size8",
        quote_merge_batch_labels=["offset8_size8", "offset16_size8"],
        quote_merge_include_canonical=False,
    )
    assert dry_run["dry_run"] is True
    assert dry_run["writes_data_outputs"] is False
    assert cast(dict[str, object], dry_run["estimated_counts"])["second_agg_rest_calls"] == 80
    assert cast(dict[str, object], dry_run["estimated_counts"])["quote_rest_workers"] == 0
    assert cast(dict[str, object], dry_run["parameters"])["quote_workers"] == 4
    assert cast(dict[str, object], dry_run["parameters"])["quote_event_offset"] == 8
    assert cast(dict[str, object], dry_run["parameters"])["quote_batch_label"] == "offset8_size8"
    assert cast(dict[str, object], dry_run["parameters"])["quote_batch_mode"] is True
    assert cast(dict[str, object], dry_run["parameters"])["quote_merge_batch_labels"] == [
        "offset8_size8",
        "offset16_size8",
    ]
    assert cast(dict[str, object], dry_run["parameters"])["quote_merge_include_canonical"] is False
    assert "missing_option_day_agg_exit_price" in cast(
        dict[str, object], dry_run["exclusion_estimate"]
    )
    empty_label_dry_run = run_data_pipeline(
        config,
        stage="quote-execution-panel",
        out_root=out_root,
        dry_run=True,
        quote_batch_label="  ",
    )
    assert cast(dict[str, object], empty_label_dry_run["parameters"])["quote_batch_label"] is None
    assert cast(dict[str, object], empty_label_dry_run["parameters"])["quote_batch_mode"] is False
    assert dry_run["planned_stages"] == [
        "options-day-aggs-bulk",
        "universe",
        "dynamic-calendar",
        "sec-companyfacts",
        "event-window-panel",
        "contract-reference-validation",
        "trade-proxy-panel",
        "quote-execution-panel",
    ]

    assert cast(dict[str, object], dry_run["bulk_day_aggs_date_range"])["start"] == "2025-07-01"


def test_bulk_day_aggs_bronze_statuses_and_refresh(tmp_path: Path) -> None:
    key_file = tmp_path / "flat_file_keys"
    key_file.write_text("access\nsecret\n", encoding="utf-8")
    config = replace(
        load_project_config(),
        bronze_data_dir=tmp_path / "bronze",
        massive_flat_file_key_file=key_file,
    )

    hit_path = data_pipeline._bronze_day_agg_path(
        config, dataset="options_day_aggs", date_value=date(2025, 1, 2)
    )
    hit_path.parent.mkdir(parents=True, exist_ok=True)
    pl.DataFrame(
        {"ticker": ["O:AAPL250117C00100000"], "close": [1.0], "volume": [10]}
    ).write_parquet(hit_path)
    hit = data_pipeline._ensure_bulk_day_agg_partition(
        config,
        dataset="options_day_aggs",
        date_value=date(2025, 1, 2),
        refresh_bronze=False,
    )
    assert hit["status"] == "hit"

    def ok_runner(
        command: Sequence[str], env: Mapping[str, str], timeout: float
    ) -> MassiveCommandResult:
        destination = Path(command[4])
        destination.parent.mkdir(parents=True, exist_ok=True)
        with gzip.open(destination, "wt", encoding="utf-8") as file:
            file.write("ticker,close,volume,vwap\nO:AAPL250117C00100000,1.5,10,1.4\n")
        return MassiveCommandResult(returncode=0, stdout="", stderr="")

    corrupt_path = data_pipeline._bronze_day_agg_path(
        config, dataset="options_day_aggs", date_value=date(2025, 1, 3)
    )
    corrupt_path.parent.mkdir(parents=True, exist_ok=True)
    corrupt_path.write_text("not parquet", encoding="utf-8")
    repaired = data_pipeline._ensure_bulk_day_agg_partition(
        config,
        dataset="options_day_aggs",
        date_value=date(2025, 1, 3),
        refresh_bronze=False,
        runner=ok_runner,
    )
    assert repaired["status"] == "repaired"

    downloaded = data_pipeline._ensure_bulk_day_agg_partition(
        config,
        dataset="underlying_day_aggs",
        date_value=date(2025, 1, 6),
        refresh_bronze=False,
        runner=ok_runner,
    )
    assert downloaded["status"] == "downloaded"

    refreshed = data_pipeline._ensure_bulk_day_agg_partition(
        config,
        dataset="options_day_aggs",
        date_value=date(2025, 1, 2),
        refresh_bronze=True,
        runner=ok_runner,
    )
    assert refreshed["status"] == "downloaded"
    before_failed_refresh = pl.read_parquet(hit_path)

    def failed_refresh_runner(
        command: Sequence[str], env: Mapping[str, str], timeout: float
    ) -> MassiveCommandResult:
        return MassiveCommandResult(returncode=127, stdout="", stderr="aws missing")

    failed_refresh = data_pipeline._ensure_bulk_day_agg_partition(
        config,
        dataset="options_day_aggs",
        date_value=date(2025, 1, 2),
        refresh_bronze=True,
        runner=failed_refresh_runner,
    )
    assert failed_refresh["status"] == "failed"
    assert hit_path.exists()
    assert pl.read_parquet(hit_path).equals(before_failed_refresh)

    def missing_runner(
        command: Sequence[str], env: Mapping[str, str], timeout: float
    ) -> MassiveCommandResult:
        return MassiveCommandResult(returncode=1, stdout="", stderr="NoSuchKey: not found")

    missing = data_pipeline._ensure_bulk_day_agg_partition(
        config,
        dataset="options_day_aggs",
        date_value=date(2025, 1, 7),
        refresh_bronze=False,
        runner=missing_runner,
    )
    assert missing["status"] == "missing_flat_file"

    def failed_runner(
        command: Sequence[str], env: Mapping[str, str], timeout: float
    ) -> MassiveCommandResult:
        return MassiveCommandResult(returncode=127, stdout="", stderr="aws missing")

    failed = data_pipeline._ensure_bulk_day_agg_partition(
        config,
        dataset="options_day_aggs",
        date_value=date(2025, 1, 8),
        refresh_bronze=False,
        runner=failed_runner,
    )
    assert failed["status"] == "failed"


def test_data_pipeline_completion_and_bulk_helper_edges(tmp_path: Path) -> None:
    output = tmp_path / "out.txt"
    manifest = tmp_path / "manifest.json"
    step = DataPipelineStep(
        "unit",
        "ran",
        outputs=(output,),
        reason="because",
        metadata={"rows": 1},
    )
    assert step.as_dict() == {
        "name": "unit",
        "status": "ran",
        "outputs": [str(output)],
        "reason": "because",
        "metadata": {"rows": 1},
    }
    assert parse_text_list(None) == []
    assert parse_text_list("AAPL, MSFT\nNVDA") == ["AAPL", "MSFT", "NVDA"]
    assert data_pipeline._complete([]) is False
    assert data_pipeline._complete([output]) is False
    assert data_pipeline._json_params_match(manifest, {"x": 1}) is False
    manifest.write_text("{bad json", encoding="utf-8")
    assert data_pipeline._json_params_match(manifest, {"x": 1}) is False
    output.write_text("ok", encoding="utf-8")
    manifest.write_text(json.dumps({"pipeline_params": {"x": 1}}), encoding="utf-8")
    assert data_pipeline._complete_with_params(
        [output], params_path=manifest, expected_params={"x": 1}
    )

    manifest.write_text(
        json.dumps(
            {
                "pipeline_params": {"x": 1},
                "status_counts": {"failed": 1},
                "dataset_counts": {"options_day_aggs": {"downloaded": 1}},
            }
        ),
        encoding="utf-8",
    )
    assert (
        data_pipeline._bulk_day_aggs_complete_with_params(
            [output], manifest_path=manifest, expected_params={"x": 1}
        )
        is False
    )
    manifest.write_text(
        json.dumps(
            {
                "pipeline_params": {"x": 1},
                "status_counts": {"failed": 0},
                "dataset_counts": {"options_day_aggs": {"hit": 1}},
            }
        ),
        encoding="utf-8",
    )
    assert data_pipeline._bulk_day_aggs_complete_with_params(
        [output], manifest_path=manifest, expected_params={"x": 1}
    )
    manifest.write_text("{bad json", encoding="utf-8")
    assert (
        data_pipeline._bulk_day_aggs_complete_with_params(
            [output], manifest_path=manifest, expected_params={"x": 1}
        )
        is False
    )
    manifest.write_text(
        json.dumps(
            {
                "pipeline_params": {"x": 1},
                "status_counts": {"failed": 0},
            }
        ),
        encoding="utf-8",
    )
    assert (
        data_pipeline._bulk_day_aggs_complete_with_params(
            [output], manifest_path=manifest, expected_params={"x": 1}
        )
        is False
    )
    manifest.write_text(
        json.dumps(
            {
                "pipeline_params": {"x": 1},
                "status_counts": {"failed": 0},
                "dataset_counts": {"underlying_day_aggs": {"hit": 1}},
            }
        ),
        encoding="utf-8",
    )
    assert (
        data_pipeline._bulk_day_aggs_complete_with_params(
            [output], manifest_path=manifest, expected_params={"x": 1}
        )
        is False
    )

    assert data_pipeline._date_partition_value(
        tmp_path / "date=2025-01-02" / "part.parquet"
    ) == date(2025, 1, 2)
    assert data_pipeline._date_partition_value(tmp_path / "date=bad" / "part.parquet") is None
    assert data_pipeline._date_partition_value(tmp_path / "part.parquet") is None
    with pytest.raises(ValueError, match="unsupported bulk day-agg dataset"):
        data_pipeline._day_agg_key(
            load_project_config(), dataset="bad", date_value=date(2025, 1, 2)
        )
    assert data_pipeline._parquet_has_columns(tmp_path / "missing.parquet", {"ticker"}) is False
    corrupt = tmp_path / "corrupt.parquet"
    corrupt.write_text("not parquet", encoding="utf-8")
    assert data_pipeline._parquet_has_columns(corrupt, {"ticker"}) is False
    with pytest.raises(ValueError, match="unsupported bulk day-agg dataset"):
        data_pipeline._bulk_required_columns("bad")
    with pytest.raises(ValueError, match="flat file missing ticker column"):
        data_pipeline._normalize_bulk_day_agg_frame(
            pl.DataFrame({"close": [1.0], "volume": [2.0]}),
            dataset="options_day_aggs",
            date_value=date(2025, 1, 2),
            source_key="key",
        )
    normalized = data_pipeline._normalize_bulk_day_agg_frame(
        pl.DataFrame({"ticker": ["A"], "close": [1], "volume": [2], "vwap": [1.1]}),
        dataset="options_day_aggs",
        date_value=date(2025, 1, 2),
        source_key="key",
    )
    assert normalized["source_date"].to_list() == ["2025-01-02"]


def test_lake_quality_audit_helper_edges(tmp_path: Path) -> None:
    target_start = date(2025, 1, 1)
    target_end = date(2025, 1, 3)
    assert data_pipeline._parquet_row_count(tmp_path / "missing.parquet") is None
    dated = tmp_path / "dated.parquet"
    pd.DataFrame(
        {
            "event_id": ["E1", "E2"],
            "ticker": ["AAA", "BBB"],
            "entry_date": ["2025-01-01", "2025-01-03"],
        }
    ).to_parquet(dated, index=False)
    assert data_pipeline._parquet_row_count(dated) == 2
    assert data_pipeline._extract_date_bounds(
        pd.DataFrame({"bad": ["x"], "entry_date": ["2025-01-02"]}),
        ("missing", "bad", "entry_date"),
    ) == ("entry_date", date(2025, 1, 2), date(2025, 1, 2), 1)
    assert data_pipeline._extract_date_bounds(
        pd.DataFrame({"bad": ["x"]}),
        ("missing", "bad"),
    ) == (None, None, None, 0)

    assert data_pipeline._target_coverage_status(
        exists=False,
        first_date=None,
        last_date=None,
        target_start=target_start,
        target_end=target_end,
    ) == ("missing", "dataset_path_missing", 0.0)
    assert data_pipeline._target_coverage_status(
        exists=True,
        first_date=None,
        last_date=None,
        target_start=target_start,
        target_end=target_end,
    ) == ("exists_no_date_bounds", "no_auditable_date_column", None)
    assert data_pipeline._target_coverage_status(
        exists=True,
        first_date=target_start,
        last_date=target_end,
        target_start=target_start,
        target_end=target_end,
        available_partitions=1,
        expected_partitions=3,
    ) == ("span_ok_partition_gap", "date_span_covers_target_but_partitions_incomplete", 1 / 3)
    assert data_pipeline._target_coverage_status(
        exists=True,
        first_date=target_start,
        last_date=target_end,
        target_start=target_start,
        target_end=target_end,
        available_partitions=3,
        expected_partitions=3,
    ) == ("target_span_covered", None, 1.0)

    missing_row, missing_years = data_pipeline._lake_dataset_row(
        dataset_id="missing",
        layer="bronze",
        path=tmp_path / "not_there",
        target_start=target_start,
        target_end=target_end,
        date_columns=("entry_date",),
        required_for_target_window=True,
        paper_grade_requirement="required",
        note="missing path",
    )
    assert missing_row["target_coverage_status"] == "missing"
    assert missing_row["required_for_target_window"] is True
    assert missing_row["required_for_2013_2025"] is True
    assert missing_years.empty

    partitioned = tmp_path / "partitioned"
    for value in (target_start, target_end):
        part = partitioned / f"date={value.isoformat()}" / "part.parquet"
        part.parent.mkdir(parents=True)
        pd.DataFrame({"ticker": ["AAA"], "close": [1.0], "volume": [1]}).to_parquet(
            part,
            index=False,
        )
    partition_row, partition_years = data_pipeline._lake_dataset_row(
        dataset_id="partitioned",
        layer="bronze",
        path=partitioned,
        target_start=target_start,
        target_end=target_end,
        date_columns=("source_date",),
        required_for_target_window=True,
        paper_grade_requirement="required",
        note="partitioned",
        partitioned_by_date=True,
        expected_weekday_partitions=3,
    )
    assert partition_row["partition_count"] == 2
    assert partition_row["target_coverage_status"] == "span_ok_partition_gap"
    assert int(partition_years["available_rows_or_partitions"].sum()) == 2

    csv_path = tmp_path / "dated.csv"
    pd.DataFrame({"event_id": ["E1"], "entry_date": ["2025-01-02"]}).to_csv(
        csv_path,
        index=False,
    )
    csv_row, csv_years = data_pipeline._lake_dataset_row(
        dataset_id="csv",
        layer="silver",
        path=csv_path,
        target_start=target_start,
        target_end=target_end,
        date_columns=("entry_date",),
        required_for_target_window=True,
        paper_grade_requirement="required",
        note="csv",
    )
    assert csv_row["row_count"] == 1
    assert csv_row["event_count"] == 1
    assert csv_row["target_coverage_status"] == "target_span_incomplete"
    assert int(csv_years["available_rows_or_partitions"].sum()) == 1

    text_path = tmp_path / "metadata.txt"
    text_path.write_text("ok", encoding="utf-8")
    text_row, _ = data_pipeline._lake_dataset_row(
        dataset_id="text",
        layer="bronze",
        path=text_path,
        target_start=target_start,
        target_end=target_end,
        date_columns=("entry_date",),
        required_for_target_window=False,
        paper_grade_requirement="metadata",
        note="text",
    )
    assert text_row["read_status"] == "exists_unread_date_not_applicable"

    corrupt = tmp_path / "bad.parquet"
    corrupt.write_text("not parquet", encoding="utf-8")
    bad_row, _ = data_pipeline._lake_dataset_row(
        dataset_id="bad",
        layer="bronze",
        path=corrupt,
        target_start=target_start,
        target_end=target_end,
        date_columns=("entry_date",),
        required_for_target_window=False,
        paper_grade_requirement="bad",
        note="bad",
    )
    assert bad_row["read_status"] == "read_failed"

    full_config = replace(
        load_project_config(),
        data_dir=tmp_path / "full_lake",
        bronze_data_dir=tmp_path / "full_lake" / "bronze",
        silver_data_dir=tmp_path / "full_lake" / "silver",
        gold_data_dir=tmp_path / "full_lake" / "gold",
        artifacts_dir=tmp_path / "full_artifacts",
    )

    def write_parquet(path: Path, rows: Sequence[Mapping[str, object]]) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(rows).to_parquet(path, index=False)

    for value in (date(2025, 1, 1), date(2025, 1, 2), date(2025, 1, 3)):
        for dataset in ("options_day_aggs", "underlying_day_aggs"):
            write_parquet(
                full_config.bronze_data_dir
                / "massive"
                / dataset
                / f"date={value.isoformat()}"
                / "part.parquet",
                [{"ticker": "AAA", "close": 1.0, "volume": 1}],
            )
    spanning_rows = [
        {"event_id": "E1", "ticker": "AAA", "entry_date": "2025-01-01"},
        {"event_id": "E2", "ticker": "AAA", "entry_date": "2025-01-03"},
    ]
    write_parquet(
        full_config.bronze_data_dir
        / "massive"
        / "quotes_v1_target_windows"
        / "quote_window_quotes.parquet",
        [
            {"options_ticker": "O:AAA250117C00100000", "quote_date": "2025-01-01"},
            {"options_ticker": "O:AAA250117P00100000", "quote_date": "2025-01-03"},
        ],
    )
    write_parquet(
        full_config.silver_data_dir / "earnings_calendar" / "main_sample.parquet",
        spanning_rows,
    )
    write_parquet(
        full_config.silver_data_dir / "event_windows" / "event_windows.parquet",
        spanning_rows,
    )
    write_parquet(
        full_config.silver_data_dir / "contracts" / "event_contract_candidates.parquet",
        spanning_rows,
    )
    write_parquet(
        full_config.silver_data_dir / "quote_execution" / "quote_window_marks.parquet",
        [
            {"event_id": "E1", "ticker": "AAA", "quote_date": "2025-01-01"},
            {"event_id": "E2", "ticker": "AAA", "quote_date": "2025-01-03"},
        ],
    )
    write_parquet(
        full_config.silver_data_dir / "quote_execution" / "quote_execution_legs.parquet",
        [
            {"event_id": "E1", "ticker": "AAA", "quote_date": "2025-01-01"},
            {"event_id": "E2", "ticker": "AAA", "quote_date": "2025-01-03"},
        ],
    )
    write_parquet(
        full_config.gold_data_dir / "modeling" / "feature_matrix.parquet",
        spanning_rows,
    )
    write_parquet(
        full_config.gold_data_dir / "quote_execution" / "quote_straddle_execution.parquet",
        spanning_rows,
    )
    write_parquet(
        full_config.gold_data_dir / "quote_execution" / "quote_ivar_event.parquet",
        spanning_rows,
    )
    write_parquet(
        full_config.gold_data_dir / "quote_execution" / "quote_iv_surface.parquet",
        [
            {
                "event_id": "E1",
                "ticker": "AAA",
                "entry_date": "2025-01-01",
                "expiration": "2025-01-17",
            },
            {
                "event_id": "E2",
                "ticker": "AAA",
                "entry_date": "2025-01-03",
                "expiration": "2025-01-17",
            },
        ],
    )
    write_parquet(
        full_config.gold_data_dir / "quote_execution" / "quote_iv_surface_summary.parquet",
        [
            {
                "event_id": "E1",
                "ticker": "AAA",
                "entry_date": "2025-01-01",
                "expiration": "2025-01-17",
            },
            {
                "event_id": "E2",
                "ticker": "AAA",
                "entry_date": "2025-01-03",
                "expiration": "2025-01-17",
            },
        ],
    )
    write_parquet(
        full_config.gold_data_dir / "quote_execution" / "quote_surface_ivar_event.parquet",
        [
            {
                "event_id": "E1",
                "ticker": "AAA",
                "entry_date": "2025-01-01",
                "expiration_1": "2025-01-17",
                "expiration_2": "2025-02-21",
            },
            {
                "event_id": "E2",
                "ticker": "AAA",
                "entry_date": "2025-01-03",
                "expiration_1": "2025-01-17",
                "expiration_2": "2025-02-21",
            },
        ],
    )
    write_parquet(
        full_config.gold_data_dir / "quote_execution" / "quote_execution_confidence.parquet",
        spanning_rows,
    )
    full_step = data_pipeline._lake_quality_audit_step(
        full_config,
        out_root=tmp_path / "full_audit",
        force=False,
        target_start=target_start,
        target_end=target_end,
    )
    assert full_step.status == "ran"
    full_report = json.loads(
        (tmp_path / "full_audit" / "lake_quality_audit" / "lake_quality_report.json").read_text(
            encoding="utf-8"
        )
    )
    assert full_report["ok"] is True
    assert full_report["incomplete_required_datasets"] == 0
    skipped_step = data_pipeline._lake_quality_audit_step(
        full_config,
        out_root=tmp_path / "full_audit",
        force=False,
        target_start=target_start,
        target_end=target_end,
    )
    assert skipped_step.status == "skipped"
    assert skipped_step.reason == "outputs_exist_params_match"

    normalized = data_pipeline._normalize_bulk_day_agg_frame(
        pl.DataFrame({"ticker": ["A"], "close": [1], "volume": [2], "vwap": [1.1]}),
        dataset="options_day_aggs",
        date_value=date(2025, 1, 2),
        source_key="key",
    )
    assert normalized["vwap"].dtype == pl.Float64
    assert (
        data_pipeline._download_error_status(
            MassiveCommandResult(returncode=1, stdout="", stderr="NoSuchKey")
        )
        == "missing_flat_file"
    )
    assert (
        data_pipeline._download_error_status(
            MassiveCommandResult(returncode=124, stdout="", stderr="timeout")
        )
        == "failed"
    )
    assert (
        data_pipeline._download_error_text(MassiveCommandResult(returncode=2, stdout="", stderr=""))
        == "aws command failed with exit code 2"
    )


def test_market_covariates_step_success_skip_and_failure(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = replace(
        load_project_config(),
        bronze_data_dir=tmp_path / "bronze",
        silver_data_dir=tmp_path / "silver",
        fred_vixcls_url="https://example.test/vix.csv",
        massive_request_timeout_seconds=1.0,
    )
    out_root = tmp_path / "artifacts"

    class FakeResponse:
        content = b"DATE,VIXCLS\n2025-01-02,18.5\n"

        def raise_for_status(self) -> None:
            return None

    class FakeClient:
        def __init__(self, *, timeout: float) -> None:
            assert timeout == 1.0

        def __enter__(self) -> FakeClient:
            return self

        def __exit__(
            self,
            exc_type: type[BaseException] | None,
            exc: BaseException | None,
            tb: object,
        ) -> None:
            _ = exc_type, exc, tb
            return None

        def get(self, url: str) -> FakeResponse:
            assert url == "https://example.test/vix.csv"
            return FakeResponse()

    def fake_normalize(
        raw: pd.DataFrame,
        *,
        source_snapshot_date: date,
        source_url: str,
    ) -> pd.DataFrame:
        assert raw.loc[0, "VIXCLS"] == 18.5
        assert source_snapshot_date <= date.today()
        assert source_url == "https://example.test/vix.csv"
        return pd.DataFrame(
            {
                "date": [date(2025, 1, 2)],
                "vix_close": [18.5],
                "is_holiday_or_missing": [False],
            }
        )

    monkeypatch.setattr("earnings_event_vol.data_pipeline.httpx.Client", FakeClient)
    monkeypatch.setattr(data_pipeline, "normalize_fred_vixcls_csv", fake_normalize)
    step = data_pipeline._market_covariates_step(config, out_root=out_root, force=False)
    assert step.status == "ran"
    skipped = data_pipeline._market_covariates_step(config, out_root=out_root, force=False)
    assert skipped.status == "skipped"

    def failing_normalize(
        raw: pd.DataFrame,
        *,
        source_snapshot_date: date,
        source_url: str,
    ) -> pd.DataFrame:
        _ = raw, source_snapshot_date, source_url
        raise ValueError("bad vix")

    monkeypatch.setattr(data_pipeline, "normalize_fred_vixcls_csv", failing_normalize)
    failed = data_pipeline._market_covariates_step(config, out_root=out_root, force=True)
    assert failed.status == "blocked"
    assert failed.reason == "market_covariates_failed"


def test_data_pipeline_source_readers_and_massive_probe_parallel(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    completed = data_pipeline._run_command_with_progress(
        [sys.executable, "-c", "print('hello-progress')"],
        cwd=tmp_path,
        label="progress-test",
    )
    assert completed.returncode == 0
    assert "hello-progress" in completed.stdout

    csv_dir = tmp_path / "csv_dir"
    csv_dir.mkdir()
    (csv_dir / "part.csv").write_text("ticker,quote_date,option_close,volume\nA,2025-01-01,1,1\n")
    assert data_pipeline._read_universe_source(csv_dir)["ticker"].tolist() == ["A"]

    parquet_dir = tmp_path / "parquet_dir"
    parquet_dir.mkdir()
    pl.DataFrame(
        {"ticker": ["B"], "quote_date": ["2025-01-01"], "option_close": [2.0], "volume": [2]}
    ).write_parquet(parquet_dir / "part.parquet")
    assert data_pipeline._read_universe_source(parquet_dir)["ticker"].tolist() == ["B"]

    parquet_file = tmp_path / "single.parquet"
    pl.DataFrame(
        {"ticker": ["C"], "quote_date": ["2025-01-01"], "option_close": [3.0], "volume": [3]}
    ).write_parquet(parquet_file)
    assert data_pipeline._read_universe_source(parquet_file)["ticker"].tolist() == ["C"]

    (tmp_path / "empty_dir").mkdir()
    with pytest.raises(FileNotFoundError):
        data_pipeline._read_universe_source(tmp_path / "empty_dir")

    blocked = data_pipeline._massive_probe_steps(
        load_project_config(),
        out_root=tmp_path / "probe",
        dates=[],
        force=False,
        jobs=2,
        download_samples=False,
    )
    assert blocked[0].status == "blocked"

    def fake_probe_one_date(
        config: object,
        *,
        out_root: Path,
        probe_date: date,
        force: bool,
        download_samples: bool,
    ) -> DataPipelineStep:
        return DataPipelineStep(f"massive-probe:{probe_date.isoformat()}", "ran")

    monkeypatch.setattr(
        "earnings_event_vol.data_pipeline._massive_probe_one_date", fake_probe_one_date
    )
    parallel = data_pipeline._massive_probe_steps(
        load_project_config(),
        out_root=tmp_path / "probe",
        dates=[date(2025, 1, 2), date(2025, 1, 3)],
        force=False,
        jobs=2,
        download_samples=False,
    )
    assert [step.name for step in parallel] == [
        "massive-probe:2025-01-02",
        "massive-probe:2025-01-03",
    ]


def test_bulk_day_aggs_stage_manifest_resume_and_blocking(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config = replace(load_project_config(), bronze_data_dir=tmp_path / "bronze")
    out_root = tmp_path / "pipeline"

    def successful_partition(
        config: object,
        *,
        dataset: str,
        date_value: date,
        refresh_bronze: bool,
    ) -> dict[str, object]:
        status = "hit" if dataset == "options_day_aggs" else "missing_flat_file"
        return {
            "date": date_value.isoformat(),
            "dataset": dataset,
            "status": status,
            "path": f"/fake/{dataset}/{date_value.isoformat()}",
        }

    monkeypatch.setattr(
        "earnings_event_vol.data_pipeline._ensure_bulk_day_agg_partition",
        successful_partition,
    )
    step = data_pipeline._options_day_aggs_bulk_step(
        config,
        out_root=out_root,
        start_date=date(2025, 1, 1),
        end_date=date(2025, 1, 3),
        trailing_months=1,
        force=False,
        refresh_bronze=False,
        jobs=3,
    )
    assert step.status == "ran"
    assert (out_root / "options_day_aggs_bulk" / "day_agg_fetch_report.csv").exists()
    assert isinstance(step.metadata["weekdays"], int)
    assert step.metadata["weekdays"] > 0

    skipped = data_pipeline._options_day_aggs_bulk_step(
        config,
        out_root=out_root,
        start_date=date(2025, 1, 1),
        end_date=date(2025, 1, 3),
        trailing_months=1,
        force=False,
        refresh_bronze=False,
        jobs=1,
    )
    assert skipped.status == "skipped"

    manifest_path = out_root / "options_day_aggs_bulk" / "options_day_aggs_bulk_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["status_counts"]["failed"] = 1
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    rerun_after_failed_manifest = data_pipeline._options_day_aggs_bulk_step(
        config,
        out_root=out_root,
        start_date=date(2025, 1, 1),
        end_date=date(2025, 1, 3),
        trailing_months=1,
        force=False,
        refresh_bronze=False,
        jobs=1,
    )
    assert rerun_after_failed_manifest.status == "ran"

    def failed_partition(
        config: object,
        *,
        dataset: str,
        date_value: date,
        refresh_bronze: bool,
    ) -> dict[str, object]:
        return {"date": date_value.isoformat(), "dataset": dataset, "status": "failed"}

    monkeypatch.setattr(
        "earnings_event_vol.data_pipeline._ensure_bulk_day_agg_partition",
        failed_partition,
    )
    failed = data_pipeline._options_day_aggs_bulk_step(
        config,
        out_root=tmp_path / "failed",
        start_date=date(2025, 1, 1),
        end_date=date(2025, 1, 1),
        trailing_months=1,
        force=False,
        refresh_bronze=False,
        jobs=2,
    )
    assert failed.status == "blocked"
    assert failed.reason == "bulk_day_agg_failures"

    def missing_partition(
        config: object,
        *,
        dataset: str,
        date_value: date,
        refresh_bronze: bool,
    ) -> dict[str, object]:
        return {
            "date": date_value.isoformat(),
            "dataset": dataset,
            "status": "missing_flat_file",
        }

    monkeypatch.setattr(
        "earnings_event_vol.data_pipeline._ensure_bulk_day_agg_partition",
        missing_partition,
    )
    missing = data_pipeline._options_day_aggs_bulk_step(
        config,
        out_root=tmp_path / "missing",
        start_date=date(2025, 1, 1),
        end_date=date(2025, 1, 1),
        trailing_months=1,
        force=False,
        refresh_bronze=False,
        jobs=2,
    )
    assert missing.status == "blocked"
    assert missing.reason == "no_options_day_aggs_available"


def test_universe_stage_reads_partitioned_options_day_aggs(tmp_path: Path) -> None:
    config = replace(load_project_config(), bronze_data_dir=tmp_path / "bronze")
    options_dir = tmp_path / "bronze" / "massive" / "options_day_aggs"
    first = options_dir / "date=2025-01-15" / "part.parquet"
    second = options_dir / "date=2025-02-14" / "part.parquet"
    outside = options_dir / "date=2025-04-01" / "part.parquet"
    for path, frame in [
        (
            first,
            pl.DataFrame(
                {
                    "ticker": [
                        "O:AAPL250117C00100000",
                        "O:MSFT250117C00100000",
                        "O:SPY250117C00500000",
                    ],
                    "volume": [10.0, 2.0, 10000.0],
                    "vwap": [2.0, 1.0, 20.0],
                    "close": [1.8, 1.1, 20.0],
                }
            ),
        ),
        (
            second,
            pl.DataFrame(
                {
                    "ticker": ["O:MSFT250321C00300000"],
                    "volume": [20.0],
                    "option_close": [3.0],
                    "close": [3.0],
                }
            ),
        ),
        (
            outside,
            pl.DataFrame(
                {
                    "ticker": ["O:TSLA250516C00200000"],
                    "volume": [1000.0],
                    "vwap": [10.0],
                    "close": [10.0],
                }
            ),
        ),
    ]:
        path.parent.mkdir(parents=True, exist_ok=True)
        frame.write_parquet(path)

    _write_eligible_equity_cache(
        tmp_path / "pipeline",
        [
            {"ticker": "AAPL", "exchange": "NASDAQ", "title": "Apple Inc."},
            {"ticker": "MSFT", "exchange": "NASDAQ", "title": "Microsoft Corporation"},
            {"ticker": "SPY", "exchange": "NYSE ArCA", "title": "SPDR S&P 500 ETF Trust"},
        ],
    )
    result = run_data_pipeline(
        config,
        stage="universe",
        out_root=tmp_path / "pipeline",
        options_day_aggs_path=options_dir,
        start_date=date(2025, 3, 1),
        end_date=date(2025, 3, 31),
        universe_top_n=1,
        universe_trailing_months=2,
    )
    assert result["ok"] is True
    universe = pl.read_parquet(
        tmp_path / "pipeline" / "universe" / "monthly_top50_universe.parquet"
    ).to_pandas()
    assert universe["ticker"].tolist() == ["MSFT"]
    assert universe["rank"].tolist() == [1]


def test_dynamic_calendar_filters_by_latest_prior_universe_month(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config = replace(load_project_config(), bronze_data_dir=tmp_path / "bronze")
    out_root = tmp_path / "pipeline"
    universe_path = out_root / "universe" / "monthly_top50_universe.parquet"
    universe_path.parent.mkdir(parents=True, exist_ok=True)
    pl.DataFrame(
        {
            "universe_month": ["2025-01-01", "2025-02-01"],
            "ticker": ["AAPL", "MSFT"],
            "rank": [1, 2],
            "trailing_months": [6, 6],
            "top_n": [50, 50],
            "trailing_option_premium_dollar_volume": [1000.0, 900.0],
            "telemetry_bucket": ["steady_proxy", "steady_proxy"],
        }
    ).write_parquet(universe_path)

    def fake_calendar(**kwargs: object) -> tuple[pd.DataFrame, dict[str, object]]:
        assert kwargs["tickers"] == ["AAPL", "MSFT"]
        assert kwargs["fail_on_missing_tickers"] is False
        return (
            pd.DataFrame(
                [
                    {
                        "ticker": "AAPL",
                        "announcement_date": "2025-01-30",
                        "announcement_timing": "AMC",
                        "source": "sec",
                        "text_validation_status": "validated_earnings_release",
                        "text_validation_source": "sec_primary_document_text",
                        "is_main_sample_candidate": True,
                    },
                    {
                        "ticker": "MSFT",
                        "announcement_date": "2025-03-01",
                        "announcement_timing": "BMO",
                        "source": "sec",
                        "text_validation_status": "validated_earnings_release",
                        "text_validation_source": "sec_primary_document_text",
                        "is_main_sample_candidate": True,
                    },
                    {
                        "ticker": "TSLA",
                        "announcement_date": "2025-01-30",
                        "announcement_timing": "AMC",
                        "source": "sec",
                        "text_validation_status": "validated_earnings_release",
                        "text_validation_source": "sec_primary_document_text",
                        "is_main_sample_candidate": True,
                    },
                ]
            ),
            {"row_count": 3, "main_sample_candidate_rows": 3},
        )

    monkeypatch.setattr(
        "earnings_event_vol.data_pipeline.build_earnings_calendar_candidates",
        fake_calendar,
    )
    step = data_pipeline._dynamic_calendar_step(
        config,
        out_root=out_root,
        start_date=date(2025, 1, 1),
        end_date=date(2025, 12, 31),
        sec_submissions_dir=None,
        massive_8k_text_dir=None,
        validate_with_massive=True,
        force=False,
    )
    assert step.status == "ran"
    output = pd.read_csv(out_root / "dynamic_calendar" / "earnings_calendar_candidates.csv")
    assert output["ticker"].tolist() == ["AAPL", "MSFT"]
    assert output["universe_rank"].tolist() == [1, 2]
    assert output["universe_filter_status"].eq("in_universe").all()
    report = json.loads(
        (out_root / "dynamic_calendar" / "earnings_calendar_report.json").read_text()
    )
    assert report["row_count"] == 2
    assert report["main_sample_candidate_rows"] == 2
    assert report["rows_by_ticker"] == {"AAPL": 1, "MSFT": 1}
    assert report["timing_counts"] == {"AMC": 1, "BMO": 1}
    assert report["text_validation_counts"] == {"validated_earnings_release": 2}
    assert report["text_validation_source_counts"] == {"sec_primary_document_text": 2}

    skipped = data_pipeline._dynamic_calendar_step(
        config,
        out_root=out_root,
        start_date=date(2025, 1, 1),
        end_date=date(2025, 12, 31),
        sec_submissions_dir=None,
        massive_8k_text_dir=None,
        validate_with_massive=True,
        force=False,
    )
    assert skipped.status == "skipped"


def test_dynamic_calendar_blocks_and_membership_edge_cases(tmp_path: Path) -> None:
    config = load_project_config()
    assert data_pipeline._path_signature(tmp_path / "does_not_exist") == {
        "path": str(tmp_path / "does_not_exist"),
        "exists": False,
    }
    missing_universe = data_pipeline._dynamic_calendar_step(
        config,
        out_root=tmp_path / "missing",
        start_date=date(2025, 1, 1),
        end_date=date(2025, 12, 31),
        sec_submissions_dir=None,
        massive_8k_text_dir=None,
        validate_with_massive=True,
        force=False,
    )
    assert missing_universe.status == "blocked"
    assert missing_universe.reason == "requires universe/monthly_top50_universe.parquet"

    empty_root = tmp_path / "empty"
    empty_path = empty_root / "universe" / "monthly_top50_universe.parquet"
    empty_path.parent.mkdir(parents=True, exist_ok=True)
    pl.DataFrame({"universe_month": ["2025-01-01"], "ticker": [""], "rank": [1]}).write_parquet(
        empty_path
    )
    empty_step = data_pipeline._dynamic_calendar_step(
        config,
        out_root=empty_root,
        start_date=date(2025, 1, 1),
        end_date=date(2025, 12, 31),
        sec_submissions_dir=None,
        massive_8k_text_dir=None,
        validate_with_massive=True,
        force=False,
    )
    assert empty_step.status == "blocked"
    assert empty_step.reason == "monthly universe has no tickers"

    empty_membership, empty_counts = data_pipeline._apply_dynamic_universe_membership(
        pd.DataFrame(),
        pd.DataFrame({"ticker": ["AAPL"], "universe_month": ["2025-01-01"], "rank": [1]}),
    )
    assert empty_membership.empty
    assert empty_counts == {"no_universe_membership": 0, "bad_event_month": 0}

    with pytest.raises(ValueError, match="monthly universe missing required columns"):
        data_pipeline._apply_dynamic_universe_membership(
            pd.DataFrame([{"ticker": "AAPL", "announcement_date": "2025-01-01"}]),
            pd.DataFrame({"ticker": ["AAPL"]}),
        )

    annotated, counts = data_pipeline._apply_dynamic_universe_membership(
        pd.DataFrame(
            [
                {"ticker": "AAPL", "announcement_date": "not-a-date"},
                {"ticker": "MSFT", "announcement_date": "2025-01-30"},
            ]
        ),
        pd.DataFrame({"ticker": ["AAPL"], "universe_month": ["2025-01-01"], "rank": [1]}),
    )
    assert annotated["universe_filter_status"].tolist() == [
        "bad_event_month",
        "no_universe_membership",
    ]
    assert counts == {"bad_event_month": 1, "no_universe_membership": 1}


def test_event_window_panel_step_uses_dynamic_calendar_outputs(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config = replace(
        load_project_config(),
        silver_data_dir=tmp_path / "data" / "silver",
    )
    out_root = tmp_path / "pipeline"
    calendar = out_root / "dynamic_calendar" / "earnings_calendar_candidates.csv"
    calendar.parent.mkdir(parents=True)
    calendar.write_text("ticker,announcement_date,is_main_sample_candidate\nAAPL,2025-01-30,true\n")

    def fake_event_window_panel(**kwargs: object) -> dict[str, object]:
        assert kwargs["calendar_path"] == calendar
        windows = config.silver_data_dir / "event_windows" / "event_windows.parquet"
        contracts = config.silver_data_dir / "contracts" / "event_contract_candidates.parquet"
        report = out_root / "event_window_panel" / "event_window_panel_report.json"
        windows.parent.mkdir(parents=True, exist_ok=True)
        contracts.parent.mkdir(parents=True, exist_ok=True)
        report.parent.mkdir(parents=True, exist_ok=True)
        windows.write_text("parquet-placeholder", encoding="utf-8")
        contracts.write_text("parquet-placeholder", encoding="utf-8")
        report.write_text(
            json.dumps(
                {
                    "pipeline_params": {
                        "stage": "event-window-panel",
                        "calendar": str(calendar),
                        "dte_min": 3,
                        "dte_max": 21,
                        "ivar_support_dte_max": 35,
                        "max_events": 7,
                    }
                }
            ),
            encoding="utf-8",
        )
        return {"events": 1, "contracts": 2, "quote_pool_contracts": 2}

    monkeypatch.setattr(
        "earnings_event_vol.data_pipeline.build_event_window_panel",
        fake_event_window_panel,
    )
    step = data_pipeline._event_window_panel_step(
        config,
        out_root=out_root,
        dte_min=3,
        dte_max=21,
        max_events=7,
        calendar_path=calendar,
        force=False,
    )
    assert step.status == "ran"
    assert step.name == "event-window-panel"

    skipped = data_pipeline._event_window_panel_step(
        config,
        out_root=out_root,
        dte_min=3,
        dte_max=21,
        max_events=7,
        calendar_path=calendar,
        force=False,
    )
    assert skipped.status == "skipped"


def test_data_pipeline_event_window_panel_stage_uses_lake_outputs_and_max_events(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = replace(
        load_project_config(),
        repo_root=tmp_path,
        silver_data_dir=tmp_path / "data" / "silver",
    )
    out_root = tmp_path / "artifacts" / "data_pipeline"
    calendar = out_root / "dynamic_calendar" / "earnings_calendar_candidates.csv"
    calendar.parent.mkdir(parents=True, exist_ok=True)
    calendar.write_text("ticker,announcement_date,is_main_sample_candidate\nAAPL,2025-01-30,true\n")

    def fake_event_window_panel(**kwargs: object) -> dict[str, object]:
        assert kwargs["calendar_path"] == calendar
        assert kwargs["dte_min"] == 3
        assert kwargs["dte_max"] == 21
        assert kwargs["max_events"] == 7
        windows = config.silver_data_dir / "event_windows" / "event_windows.parquet"
        contracts = config.silver_data_dir / "contracts" / "event_contract_candidates.parquet"
        report = out_root / "event_window_panel" / "event_window_panel_report.json"
        windows.parent.mkdir(parents=True, exist_ok=True)
        contracts.parent.mkdir(parents=True, exist_ok=True)
        report.parent.mkdir(parents=True, exist_ok=True)
        windows.write_text("parquet-placeholder", encoding="utf-8")
        contracts.write_text("parquet-placeholder", encoding="utf-8")
        report.write_text(
            json.dumps(
                {
                    "pipeline_params": {
                        "stage": "event-window-panel",
                        "calendar": str(calendar),
                        "dte_min": 3,
                        "dte_max": 21,
                        "ivar_support_dte_max": 35,
                        "max_events": 7,
                    }
                }
            ),
            encoding="utf-8",
        )
        return {"events": 1, "contracts": 2, "quote_pool_contracts": 2}

    monkeypatch.setattr(
        "earnings_event_vol.data_pipeline.build_event_window_panel",
        fake_event_window_panel,
    )

    result = run_data_pipeline(
        config,
        stage="event-window-panel",
        out_root=out_root,
        force=True,
        dte_min=3,
        dte_max=21,
        max_events=7,
    )

    assert result["ok"] is True
    step = _pipeline_steps(result)[0]
    assert step["status"] == "ran"
    outputs = cast(list[str], step["outputs"])
    assert str(config.silver_data_dir / "event_windows" / "event_windows.parquet") in outputs
    assert (
        str(config.silver_data_dir / "contracts" / "event_contract_candidates.parquet") in outputs
    )

    skipped = run_data_pipeline(
        config,
        stage="event-window-panel",
        out_root=out_root,
        dte_min=3,
        dte_max=21,
        max_events=7,
    )
    assert _pipeline_steps(skipped)[0]["status"] == "skipped"


def test_event_window_contract_discovery_requires_exact_underlying(tmp_path: Path) -> None:
    options_path = tmp_path / "options_day_aggs.parquet"
    pl.DataFrame(
        {
            "ticker": [
                "O:A260220C00100000",
                "O:AAPL260220C00100000",
                "O:A260220P00100000",
            ],
            "close": [4.0, 99.0, 3.5],
            "volume": [10.0, 10.0, 8.0],
            "transactions": [2.0, 2.0, 1.0],
        }
    ).write_parquet(options_path)
    event = pd.Series(
        {
            "event_id": "A_2026Q1",
            "ticker": "A",
            "entry_date": date(2026, 2, 5),
            "exit_date": date(2026, 2, 6),
            "s_before": 100.0,
            "requested_dte_max": 21,
        }
    )

    contracts = event_window_panel_module._load_option_day_contracts(
        options_path,
        event=event,
        dte_min=3,
        dte_max=28,
    )

    assert {row["options_ticker"] for row in contracts} == {
        "O:A260220C00100000",
        "O:A260220P00100000",
    }
    assert all(row["ticker"] == "A" for row in contracts)


def test_event_window_contract_discovery_keeps_short_dte_ivar_support(
    tmp_path: Path,
) -> None:
    options_path = tmp_path / "options_day_aggs.parquet"
    pl.DataFrame(
        {
            "ticker": [
                "O:ABC260206C00100000",
                "O:ABC260206P00100000",
                "O:ABC260220C00100000",
                "O:ABC260220P00100000",
            ],
            "close": [2.0, 1.5, 4.0, 3.5],
            "volume": [10.0, 10.0, 8.0, 8.0],
            "transactions": [2.0, 2.0, 1.0, 1.0],
        }
    ).write_parquet(options_path)
    event = pd.Series(
        {
            "event_id": "ABC_2026Q1",
            "ticker": "ABC",
            "entry_date": date(2026, 2, 5),
            "exit_date": date(2026, 2, 6),
            "s_before": 100.0,
            "requested_dte_min": 3,
            "requested_dte_max": 21,
        }
    )

    contracts = pd.DataFrame(
        event_window_panel_module._load_option_day_contracts(
            options_path,
            event=event,
            dte_min=3,
            dte_max=28,
        )
    )

    short_dte = contracts.loc[contracts["expiration"].eq(date(2026, 2, 6))]
    assert len(short_dte) == 2
    assert short_dte["eligible_for_quote_pool"].eq(False).all()
    assert short_dte["is_ivar_support_only"].eq(True).all()
    assert set(contracts["expiration"]) == {date(2026, 2, 6), date(2026, 2, 20)}


def test_event_window_near_atm_selection_prefers_valid_call_put_pairs() -> None:
    contracts = pd.DataFrame(
        {
            "event_id": ["evt"] * 3,
            "expiration": [date(2026, 2, 20)] * 3,
            "strike": [100.0, 101.0, 101.0],
            "right": ["call", "call", "put"],
            "moneyness_abs": [0.0, 0.01, 0.01],
            "options_ticker": [
                "O:ABC260220C00100000",
                "O:ABC260220C00101000",
                "O:ABC260220P00101000",
            ],
        }
    )

    selected = event_window_panel_module._select_near_atm_contracts(
        contracts,
        strikes_per_expiry=1,
    )

    assert set(selected["strike"]) == {101.0}
    assert set(selected["right"]) == {"call", "put"}


def test_event_window_entry_timestamp_uses_scheduled_early_close(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = replace(
        load_project_config(),
        bronze_data_dir=tmp_path / "bronze",
        silver_data_dir=tmp_path / "silver",
    )
    calendar_path = tmp_path / "calendar.csv"
    pd.DataFrame(
        [
            {
                "ticker": "A",
                "announcement_date": "2026-11-27",
                "announcement_timing": "AMC",
                "source": "sec",
                "source_timestamp": "2026-11-27T21:05:00.000Z",
                "source_id": "half-day",
                "timing_confidence": "proxy",
                "text_validation_status": "validated_earnings_release",
                "is_main_sample_candidate": True,
            }
        ]
    ).to_csv(calendar_path, index=False)

    trading_days = {date(2026, 11, 27), date(2026, 11, 30)}

    def fake_ensure_underlying_file(config: object, day: date) -> bool:
        return day in trading_days

    def fake_load_underlying_bars(
        path: Path,
        tickers: set[str],
        day: date,
    ) -> list[dict[str, object]]:
        assert tickers == {"A"}
        return [
            {
                "ticker": "A",
                "date": day,
                "open": 100.0,
                "high": 101.0,
                "low": 99.0,
                "close": 100.0 if day == date(2026, 11, 27) else 104.0,
                "volume": 1000,
                "source_dataset": "underlying_day_aggs",
            }
        ]

    monkeypatch.setattr(
        event_window_panel_module,
        "_ensure_underlying_file",
        fake_ensure_underlying_file,
    )
    monkeypatch.setattr(
        event_window_panel_module, "_load_underlying_bars", fake_load_underlying_bars
    )
    monkeypatch.setattr(event_window_panel_module, "_ensure_options_file", lambda *args: False)

    event_window_panel_module.build_event_window_panel(
        config=config,
        calendar_path=calendar_path,
        out_root=tmp_path / "out",
        dte_min=3,
        dte_max=21,
        strikes_per_expiry=3,
        max_events=None,
    )

    windows = pl.read_parquet(
        config.silver_data_dir / "event_windows" / "event_windows.parquet"
    ).to_pandas()
    assert pd.Timestamp(windows["entry_date"].iloc[0]).date() == date(2026, 11, 27)
    assert pd.Timestamp(windows["exit_date"].iloc[0]).date() == date(2026, 11, 30)
    assert windows["event_entry_timestamp"].iloc[0].startswith("2026-11-27T13:00:00")


def test_event_window_panel_excludes_non_trading_announcement_date(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = replace(
        load_project_config(),
        bronze_data_dir=tmp_path / "bronze",
        silver_data_dir=tmp_path / "silver",
    )
    calendar_path = tmp_path / "calendar.csv"
    pd.DataFrame(
        [
            {
                "ticker": "A",
                "announcement_date": "2026-11-28",
                "announcement_timing": "BMO",
                "source": "manual",
                "source_timestamp": "2026-11-28T13:00:00.000Z",
                "source_id": "weekend-bmo",
                "timing_confidence": "explicit",
                "text_validation_status": "validated_earnings_release",
                "is_main_sample_candidate": True,
            }
        ]
    ).to_csv(calendar_path, index=False)

    monkeypatch.setattr(event_window_panel_module, "_ensure_underlying_file", lambda *args: False)
    monkeypatch.setattr(event_window_panel_module, "_ensure_options_file", lambda *args: False)

    event_window_panel_module.build_event_window_panel(
        config=config,
        calendar_path=calendar_path,
        out_root=tmp_path / "out",
        dte_min=3,
        dte_max=21,
        strikes_per_expiry=3,
        max_events=None,
    )

    windows = pl.read_parquet(
        config.silver_data_dir / "event_windows" / "event_windows.parquet"
    ).to_pandas()
    assert pd.isna(windows["entry_date"].iloc[0])
    assert pd.isna(windows["exit_date"].iloc[0])
    assert windows["exclusion_reason"].iloc[0] == "announcement_date_not_trading_day"


def test_event_window_panel_excludes_source_timestamp_timing_mismatch(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = replace(
        load_project_config(),
        bronze_data_dir=tmp_path / "bronze",
        silver_data_dir=tmp_path / "silver",
    )
    calendar_path = tmp_path / "calendar.csv"
    pd.DataFrame(
        [
            {
                "ticker": "A",
                "announcement_date": "2026-02-05",
                "announcement_timing": "AMC",
                "source": "manual",
                "source_timestamp": "2026-02-05T17:00:00Z",
                "source_id": "dmh-mislabeled",
                "timing_confidence": "explicit",
                "text_validation_status": "validated_earnings_release",
                "is_main_sample_candidate": True,
            }
        ]
    ).to_csv(calendar_path, index=False)

    monkeypatch.setattr(event_window_panel_module, "_ensure_underlying_file", lambda *args: False)
    monkeypatch.setattr(event_window_panel_module, "_ensure_options_file", lambda *args: False)

    event_window_panel_module.build_event_window_panel(
        config=config,
        calendar_path=calendar_path,
        out_root=tmp_path / "out",
        dte_min=3,
        dte_max=21,
        strikes_per_expiry=3,
        max_events=None,
    )

    windows = pl.read_parquet(
        config.silver_data_dir / "event_windows" / "event_windows.parquet"
    ).to_pandas()
    assert pd.isna(windows["entry_date"].iloc[0])
    assert pd.isna(windows["exit_date"].iloc[0])
    assert windows["exclusion_reason"].iloc[0] == "source_timestamp_timing_mismatch"
    assert pd.isna(windows["rvar_event"].iloc[0])


def test_data_pipeline_trade_proxy_panel_stage_uses_single_entrypoint(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = replace(
        load_project_config(),
        repo_root=tmp_path,
        gold_data_dir=tmp_path / "data" / "gold",
    )
    out_root = tmp_path / "artifacts" / "data_pipeline"
    captured: dict[str, object] = {}

    def fake_run(
        command: Sequence[str],
        *,
        cwd: Path,
        label: str,
    ) -> subprocess.CompletedProcess[str]:
        captured["command"] = list(command)
        captured["cwd"] = cwd
        captured["label"] = label
        gold_output = config.gold_data_dir / "event_panel" / "trade_proxy_event_panel.parquet"
        report = out_root / "trade_proxy_panel" / "trade_proxy_panel_report.json"
        gold_output.parent.mkdir(parents=True, exist_ok=True)
        report.parent.mkdir(parents=True, exist_ok=True)
        gold_output.write_text("parquet-placeholder", encoding="utf-8")
        report.write_text(
            json.dumps(
                {
                    "pipeline_params": {
                        "stage": "trade-proxy-panel",
                        "max_events": 2,
                        "max_contracts": 12,
                        "lookback_seconds": 600,
                        "second_agg_buffer_minutes": 60,
                        "price_field": "option_close",
                        "rest_limit": data_pipeline.DEFAULT_TRADE_PROXY_REST_LIMIT,
                        "haircut_fraction": data_pipeline.DEFAULT_TRADE_PROXY_HAIRCUT_FRACTION,
                        "entry_price_method": "preclose_15m_option_second_agg_vwap",
                        "c2c_exit_price_method": "exit_preclose_15m_option_second_agg_vwap",
                        "post_open_option_vwap_windows": ["0_5", "5_15"],
                    }
                }
            ),
            encoding="utf-8",
        )
        return subprocess.CompletedProcess(command, 0, stdout="trade proxy ok", stderr="")

    monkeypatch.setattr("earnings_event_vol.data_pipeline._run_command_with_progress", fake_run)

    result = run_data_pipeline(
        config,
        stage="trade-proxy-panel",
        out_root=out_root,
        force=True,
        jobs=3,
        max_events=2,
        max_contracts=12,
        lookback_seconds=600,
        price_field="option_close",
    )

    assert result["ok"] is True
    step = _pipeline_steps(result)[0]
    assert step["status"] == "ran"
    command = cast(list[str], captured["command"])
    assert "build_trade_proxy_panel.py" in command[1]
    assert "--jobs" in command
    assert "--lookback-seconds" in command
    assert "--second-agg-buffer-minutes" in command
    assert "--price-field" in command
    assert "--rest-limit" in command
    assert "--haircut-fraction" in command
    assert "--max-events" in command
    assert "--max-contracts" in command
    assert captured["cwd"] == tmp_path
    assert captured["label"] == "trade-proxy-panel"

    skipped = run_data_pipeline(
        config,
        stage="trade-proxy-panel",
        out_root=out_root,
        max_events=2,
        max_contracts=12,
        lookback_seconds=600,
        price_field="option_close",
    )
    assert _pipeline_steps(skipped)[0]["status"] == "skipped"


def test_trade_proxy_panel_requires_contract_reference_validation(tmp_path: Path) -> None:
    config = replace(
        load_project_config(),
        silver_data_dir=tmp_path / "silver",
        gold_data_dir=tmp_path / "gold",
    )
    windows_path = config.silver_data_dir / "event_windows" / "event_windows.parquet"
    contracts_path = config.silver_data_dir / "contracts" / "event_contract_candidates.parquet"
    windows_path.parent.mkdir(parents=True, exist_ok=True)
    contracts_path.parent.mkdir(parents=True, exist_ok=True)
    pl.DataFrame(
        {
            "event_id": ["ABC_2026Q1"],
            "ticker": ["ABC"],
            "event_entry_timestamp": ["2026-02-05T16:00:00-05:00"],
            "s_before": [100.0],
            "s_after": [101.0],
            "rvar_event": [0.01],
        }
    ).write_parquet(windows_path)
    pl.DataFrame(
        {
            "event_id": ["ABC_2026Q1"],
            "options_ticker": ["O:ABC260213C00100000"],
            "eligible_for_quote_pool": [True],
        }
    ).write_parquet(contracts_path)

    with pytest.raises(ValueError, match="contract_reference_validated"):
        trade_proxy_panel_script.build_trade_proxy_panel(
            config=config,
            out_root=tmp_path / "out",
            force=True,
            max_events=None,
            max_contracts=None,
            lookback_seconds=900,
            second_agg_buffer_minutes=60,
            price_field="option_vwap",
            jobs=1,
            rest_limit=100,
            haircut_fraction=0.10,
            refresh_bronze=False,
        )


def test_trade_proxy_reference_proxy_mask_allows_standard_missing_reference() -> None:
    contracts = pd.DataFrame(
        {
            "contract_reference_validated": [True, False, False, False, False, False],
            "contract_reference_status": [
                "validated",
                "missing_reference",
                "missing_reference",
                "missing_reference",
                "fetch_failed",
                "not_requested",
            ],
            "contract_discovery_status_pre_reference": [
                "ok",
                "ok",
                "ok",
                "non_standard_excluded",
                "ok",
                "ok",
            ],
            "contract_reference_has_adjusted_deliverable": [
                False,
                False,
                True,
                False,
                False,
                False,
            ],
            "option_multiplier": [100, 100, 100, 100, 100, pd.NA],
            "contract_size": [100, 100, 100, 100, 100, 100],
        }
    )

    assert trade_proxy_panel_script._contract_reference_proxy_mask(contracts).tolist() == [
        True,
        True,
        False,
        False,
        False,
        True,
    ]


def test_trade_proxy_panel_keeps_ivar_support_contracts_outside_trade_pool(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = replace(
        load_project_config(),
        silver_data_dir=tmp_path / "silver",
        gold_data_dir=tmp_path / "gold",
    )
    windows_path = config.silver_data_dir / "event_windows" / "event_windows.parquet"
    contracts_path = config.silver_data_dir / "contracts" / "event_contract_candidates.parquet"
    windows_path.parent.mkdir(parents=True, exist_ok=True)
    contracts_path.parent.mkdir(parents=True, exist_ok=True)
    pl.DataFrame(
        {
            "event_id": ["ABC_2026Q1"],
            "ticker": ["ABC"],
            "announcement_date": [date(2026, 2, 5)],
            "entry_date": [date(2026, 2, 5)],
            "exit_date": [date(2026, 2, 6)],
            "event_entry_timestamp": ["2026-02-05T16:00:00-05:00"],
            "s_before": [100.0],
            "s_after": [101.0],
            "rvar_event": [0.01],
        }
    ).write_parquet(windows_path)
    pl.DataFrame(
        {
            "event_id": ["ABC_2026Q1"] * 4,
            "ticker": ["ABC"] * 4,
            "entry_date": [date(2026, 2, 5)] * 4,
            "exit_date": [date(2026, 2, 6)] * 4,
            "expiration": [
                date(2026, 2, 6),
                date(2026, 2, 6),
                date(2026, 2, 20),
                date(2026, 2, 20),
            ],
            "strike": [100.0] * 4,
            "right": ["call", "put", "call", "put"],
            "options_ticker": [
                "O:ABC260206C00100000",
                "O:ABC260206P00100000",
                "O:ABC260220C00100000",
                "O:ABC260220P00100000",
            ],
            "dte": [1, 1, 15, 15],
            "moneyness_abs": [0.0] * 4,
            "eligible_for_quote_pool": [False, False, True, True],
            "is_ivar_support_only": [True, True, False, False],
            "contract_reference_validated": [True] * 4,
            "is_main_dte_5_14": [False, False, False, False],
            "is_robustness_dte_3_21": [False, False, True, True],
        }
    ).write_parquet(contracts_path)
    captured: dict[str, object] = {"straddle_sets": []}

    def fake_fetch(
        config: object,
        contracts: pd.DataFrame,
        *,
        jobs: int,
        limit: int,
        buffer_minutes: int,
        force: bool,
    ) -> tuple[dict[tuple[str, date, str], pd.DataFrame], pd.DataFrame]:
        del config, jobs, limit, buffer_minutes, force
        captured["fetched"] = contracts.copy()
        return {}, pd.DataFrame(columns=["bronze_path", "cache_status"])

    def fake_price_frame(
        contracts: pd.DataFrame,
        bar_frames: Mapping[object, object],
        *,
        lookback_seconds: int = 900,
        price_field: str = "option_vwap",
    ) -> pd.DataFrame:
        del bar_frames, lookback_seconds, price_field
        return contracts.assign(
            proxy_status=TRADE_PROXY_STATUS_OK,
            proxy_price=1.0,
            proxy_volume_window=1,
            proxy_transactions_window=1,
            panel_grade=TRADE_PROXY_PANEL_GRADE,
        )

    def fake_attach(proxy_prices: pd.DataFrame, windows: pd.DataFrame) -> pd.DataFrame:
        del windows
        return proxy_prices.assign(local_iv=0.5, local_iv_status="ok")

    def fake_straddles(
        iv_estimates: pd.DataFrame,
        windows: pd.DataFrame,
        **kwargs: object,
    ) -> pd.DataFrame:
        del windows, kwargs
        cast(list[set[str]], captured["straddle_sets"]).append(
            set(iv_estimates["options_ticker"].astype(str))
        )
        return pd.DataFrame()

    monkeypatch.setattr(trade_proxy_panel_script, "_fetch_second_aggregate_bars", fake_fetch)
    monkeypatch.setattr(trade_proxy_panel_script, "build_trade_proxy_price_frame", fake_price_frame)
    monkeypatch.setattr(trade_proxy_panel_script, "attach_trade_proxy_local_iv", fake_attach)
    monkeypatch.setattr(
        trade_proxy_panel_script,
        "build_trade_proxy_ivar_inputs",
        lambda iv_estimates, windows: iv_estimates[["event_id", "expiration", "local_iv"]].rename(
            columns={"local_iv": "iv"}
        ),
    )
    monkeypatch.setattr(
        trade_proxy_panel_script,
        "extract_trade_proxy_event_panel",
        lambda ivar_inputs, windows: windows.assign(
            trade_proxy_ivar_event=0.01,
            ivar_event=0.01,
            ivar_failure_reason=None,
        ),
    )
    monkeypatch.setattr(
        trade_proxy_panel_script, "build_proxy_straddle_diagnostics", fake_straddles
    )
    monkeypatch.setattr(
        trade_proxy_panel_script,
        "edge_decile_diagnostics",
        lambda panel: pd.DataFrame(),
    )

    trade_proxy_panel_script.build_trade_proxy_panel(
        config=config,
        out_root=tmp_path / "out",
        force=True,
        max_events=None,
        max_contracts=None,
        lookback_seconds=900,
        second_agg_buffer_minutes=60,
        price_field="option_vwap",
        jobs=1,
        rest_limit=100,
        haircut_fraction=0.10,
        refresh_bronze=False,
    )

    fetched = cast(pd.DataFrame, captured["fetched"])
    assert set(fetched["options_ticker"]) == {
        "O:ABC260206C00100000",
        "O:ABC260206P00100000",
        "O:ABC260220C00100000",
        "O:ABC260220P00100000",
    }
    assert cast(list[set[str]], captured["straddle_sets"]) == [
        {"O:ABC260220C00100000", "O:ABC260220P00100000"},
        {"O:ABC260220C00100000", "O:ABC260220P00100000"},
    ]


def test_data_pipeline_all_orchestrates_dynamic_top50_proxy_dag(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = replace(
        load_project_config(),
        repo_root=tmp_path,
        gold_data_dir=tmp_path / "data" / "gold",
    )
    out_root = tmp_path / "artifacts" / "data_pipeline"
    captured_commands: list[list[str]] = []

    def fake_bulk(*args: object, **kwargs: object) -> DataPipelineStep:
        return DataPipelineStep("options-day-aggs-bulk", "ran")

    def fake_universe(*args: object, **kwargs: object) -> DataPipelineStep:
        return DataPipelineStep("universe", "ran")

    def fake_dynamic_calendar(*args: object, **kwargs: object) -> DataPipelineStep:
        dynamic_out = out_root / "dynamic_calendar"
        dynamic_out.mkdir(parents=True, exist_ok=True)
        csv_path = dynamic_out / "earnings_calendar_candidates.csv"
        parquet_path = dynamic_out / "earnings_calendar_candidates.parquet"
        report_path = dynamic_out / "earnings_calendar_report.json"
        pd.DataFrame(
            [
                {
                    "ticker": "AAPL",
                    "announcement_date": "2025-01-30",
                    "announcement_timing": "AMC",
                    "source": "sec_edgar_submissions_archive",
                    "is_main_sample_candidate": True,
                    "universe_month": "2025-01-01",
                    "universe_rank": 1,
                    "in_universe": True,
                    "universe_filter_status": "in_universe",
                }
            ]
        ).to_csv(csv_path, index=False)
        parquet_path.write_text("parquet-placeholder", encoding="utf-8")
        report_path.write_text("{}", encoding="utf-8")
        return DataPipelineStep(
            "dynamic-calendar",
            "ran",
            (csv_path, parquet_path, report_path),
        )

    def fake_sec_companyfacts(*args: object, **kwargs: object) -> DataPipelineStep:
        return DataPipelineStep("sec-companyfacts", "ran")

    def fake_event_window_panel(*args: object, **kwargs: object) -> DataPipelineStep:
        assert kwargs["calendar_path"] == out_root / "dynamic_calendar" / (
            "earnings_calendar_candidates.csv"
        )
        return DataPipelineStep("event-window-panel", "ran")

    def fake_contract_reference(*args: object, **kwargs: object) -> DataPipelineStep:
        return DataPipelineStep("contract-reference-validation", "ran")

    def fake_quote_execution(*args: object, **kwargs: object) -> DataPipelineStep:
        return DataPipelineStep("quote-execution-panel", "ran")

    def fake_run(
        command: Sequence[str],
        *,
        cwd: Path,
        label: str,
    ) -> subprocess.CompletedProcess[str]:
        assert cwd == tmp_path
        captured_commands.append(list(command))
        gold_output = config.gold_data_dir / "event_panel" / "trade_proxy_event_panel.parquet"
        report = out_root / "trade_proxy_panel" / "trade_proxy_panel_report.json"
        gold_output.parent.mkdir(parents=True, exist_ok=True)
        report.parent.mkdir(parents=True, exist_ok=True)
        gold_output.write_text("parquet-placeholder", encoding="utf-8")
        report.write_text("{}", encoding="utf-8")
        return subprocess.CompletedProcess(command, 0, stdout="ok", stderr="")

    monkeypatch.setattr("earnings_event_vol.data_pipeline._options_day_aggs_bulk_step", fake_bulk)
    monkeypatch.setattr("earnings_event_vol.data_pipeline._universe_step", fake_universe)
    monkeypatch.setattr(
        "earnings_event_vol.data_pipeline._dynamic_calendar_step", fake_dynamic_calendar
    )
    monkeypatch.setattr(
        "earnings_event_vol.data_pipeline._sec_companyfacts_step", fake_sec_companyfacts
    )
    monkeypatch.setattr(
        "earnings_event_vol.data_pipeline._event_window_panel_step", fake_event_window_panel
    )
    monkeypatch.setattr(
        "earnings_event_vol.data_pipeline._contract_reference_validation_step",
        fake_contract_reference,
    )
    monkeypatch.setattr(
        "earnings_event_vol.data_pipeline._quote_execution_panel_step",
        fake_quote_execution,
    )
    monkeypatch.setattr("earnings_event_vol.data_pipeline._run_command_with_progress", fake_run)

    result = run_data_pipeline(
        config,
        stage="all",
        out_root=out_root,
        jobs=3,
        dte_min=3,
        dte_max=21,
        max_events=5,
        max_contracts=90,
    )

    steps = _pipeline_steps(result)
    assert result["ok"] is True
    assert [step["name"] for step in steps] == [
        "options-day-aggs-bulk",
        "universe",
        "dynamic-calendar",
        "sec-companyfacts",
        "event-window-panel",
        "contract-reference-validation",
        "trade-proxy-panel",
        "quote-execution-panel",
    ]
    assert [step["status"] for step in steps] == [
        "ran",
        "ran",
        "ran",
        "ran",
        "ran",
        "ran",
        "ran",
        "ran",
    ]
    assert len(captured_commands) == 1
    trade_command = captured_commands[0]
    assert "build_trade_proxy_panel.py" in trade_command[1]
    assert "--max-contracts" in trade_command


def test_sec_companyfacts_stage_writes_silver_and_diagnostics(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = replace(
        load_project_config(),
        bronze_data_dir=tmp_path / "bronze",
        silver_data_dir=tmp_path / "silver",
        sec_companyfacts_url_template="https://example.test/CIK{cik:010d}.json",
        massive_request_timeout_seconds=1.0,
    )
    out_root = tmp_path / "artifacts" / "data_pipeline"
    dynamic_out = out_root / "dynamic_calendar"
    dynamic_out.mkdir(parents=True)
    pd.DataFrame([{"ticker": "AAPL"}]).to_csv(
        dynamic_out / "earnings_calendar_candidates.csv", index=False
    )
    companyfacts_payload: dict[str, object] = {
        "facts": {
            "us-gaap": {
                "Assets": {
                    "units": {
                        "USD": [
                            {
                                "accn": "0000320193-25-000001",
                                "val": 100.0,
                                "filed": "2025-01-03",
                                "end": "2024-12-31",
                                "fy": 2024,
                                "fp": "FY",
                                "form": "10-K",
                            }
                        ]
                    }
                }
            }
        }
    }

    def fake_ticker_map(client: object, cfg: object) -> dict[str, int]:
        _ = client, cfg
        return {"AAPL": 320193}

    def fake_submissions(
        *,
        tickers: Sequence[str],
        config: object,
        client: object,
        archive_cache_dir: Path,
        fail_on_missing_tickers: bool,
        request_interval_seconds: float,
    ) -> dict[str, dict[str, object]]:
        _ = tickers, config, client, archive_cache_dir, fail_on_missing_tickers
        assert request_interval_seconds == pytest.approx(0.125)
        return {
            "AAPL": {
                "filings": {
                    "recent": {
                        "accessionNumber": ["0000320193-25-000001"],
                        "acceptanceDateTime": ["2025-01-03T20:00:00.000Z"],
                    }
                }
            }
        }

    class FakeResponse:
        def raise_for_status(self) -> None:
            return None

        def json(self) -> dict[str, object]:
            return companyfacts_payload

    class FakeClient:
        def __init__(self, *, timeout: float) -> None:
            self.timeout = timeout
            self.urls: list[str] = []

        def __enter__(self) -> FakeClient:
            return self

        def __exit__(
            self,
            exc_type: type[BaseException] | None,
            exc: BaseException | None,
            tb: object,
        ) -> None:
            _ = exc_type, exc, tb
            return None

        def get(self, url: str, *, headers: Mapping[str, str]) -> FakeResponse:
            assert headers["User-Agent"] == config.sec_user_agent
            self.urls.append(url)
            return FakeResponse()

    monkeypatch.setattr("earnings_event_vol.data_pipeline.fetch_sec_ticker_map", fake_ticker_map)
    monkeypatch.setattr(
        "earnings_event_vol.data_pipeline.fetch_sec_submission_payloads", fake_submissions
    )
    monkeypatch.setattr("earnings_event_vol.data_pipeline.httpx.Client", FakeClient)

    step = data_pipeline._sec_companyfacts_step(config, out_root=out_root, force=True)

    assert step.status == "ran"
    facts = pd.read_parquet(config.silver_data_dir / "sec" / "companyfacts.parquet")
    assert facts.loc[0, "ticker"] == "AAPL"
    assert facts.loc[0, "cik"] == 320193
    assert facts.loc[0, "feature_concept"] == "assets"
    assert facts.loc[0, "acceptance_datetime"] == "2025-01-03T20:00:00.000Z"
    diagnostics = pd.read_csv(out_root / "sec_companyfacts" / "sec_companyfacts_diagnostics.csv")
    assert diagnostics.loc[0, "status"] == "fetched"
    manifest = json.loads(
        (out_root / "sec_companyfacts" / "sec_companyfacts_manifest.json").read_text(
            encoding="utf-8"
        )
    )
    assert manifest["mapped_acceptance_rows"] == 1
    assert manifest["fallback_filed_rows"] == 0


def test_sec_companyfacts_missing_calendar_and_payload_edges(tmp_path: Path) -> None:
    config = replace(
        load_project_config(),
        bronze_data_dir=tmp_path / "bronze",
        silver_data_dir=tmp_path / "silver",
    )
    out_root = tmp_path / "artifacts" / "data_pipeline"

    blocked = data_pipeline._sec_companyfacts_step(config, out_root=out_root, force=True)
    assert blocked.status == "blocked"
    assert blocked.reason == "requires dynamic-calendar earnings_calendar_candidates.csv"

    calendar_path = out_root / "dynamic_calendar" / "earnings_calendar_candidates.csv"
    calendar_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame([{"ticker": "AAA"}]).to_csv(calendar_path, index=False)
    silver_path = config.silver_data_dir / "sec" / "companyfacts.parquet"
    diagnostics_path = out_root / "sec_companyfacts" / "sec_companyfacts_diagnostics.csv"
    manifest_path = out_root / "sec_companyfacts" / "sec_companyfacts_manifest.json"
    silver_path.parent.mkdir(parents=True, exist_ok=True)
    diagnostics_path.parent.mkdir(parents=True, exist_ok=True)
    silver_path.write_text("cached", encoding="utf-8")
    diagnostics_path.write_text("ticker,status\nAAA,cached\n", encoding="utf-8")
    manifest_path.write_text(
        json.dumps(
            {
                "pipeline_params": {
                    "stage": "sec-companyfacts",
                    "calendar": data_pipeline._path_signature(calendar_path),
                    "endpoint": config.sec_companyfacts_url_template,
                    "request_interval_seconds": 0.125,
                }
            }
        ),
        encoding="utf-8",
    )
    skipped = data_pipeline._sec_companyfacts_step(config, out_root=out_root, force=False)
    assert skipped.status == "skipped"
    assert skipped.reason == "outputs_exist_params_match"

    acceptance = data_pipeline._submission_acceptance_lookup(
        {
            "filings": {
                "recent": {
                    "accessionNumber": ["RECENT"],
                    "acceptanceDateTime": ["2025-01-02T20:00:00.000Z"],
                }
            },
            "archive_payloads": [
                {
                    "accessionNumber": ["ARCHIVE", None],
                    "acceptanceDateTime": ["2025-01-03T22:00:00.000Z", ""],
                },
                "not-a-block",
            ],
        }
    )
    assert acceptance == {
        "RECENT": "2025-01-02T20:00:00.000Z",
        "ARCHIVE": "2025-01-03T22:00:00.000Z",
    }
    assert data_pipeline._normalize_companyfacts_payload(
        ticker="AAA",
        cik=1,
        payload={"facts": {"us-gaap": []}},
        acceptance_lookup={},
    ).empty
    normalized = data_pipeline._normalize_companyfacts_payload(
        ticker="AAA",
        cik=1,
        payload={
            "facts": {
                "us-gaap": {
                    "Assets": {
                        "units": {
                            "shares": [{"accn": "SHARES", "val": 10}],
                            "USD": [
                                "not-a-row",
                                {
                                    "accn": "ARCHIVE",
                                    "val": 100,
                                    "filed": "2025-01-04",
                                    "end": "2024-12-31",
                                },
                            ],
                        }
                    },
                    "Revenues": {"units": []},
                    "NetIncomeLoss": {},
                }
            }
        },
        acceptance_lookup=acceptance,
    )
    assert normalized["feature_concept"].tolist() == ["assets"]
    assert normalized["acceptance_datetime"].tolist() == ["2025-01-03T22:00:00.000Z"]


def test_sec_companyfacts_stage_keeps_successful_rows_when_one_ticker_fails(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = replace(
        load_project_config(),
        bronze_data_dir=tmp_path / "bronze",
        silver_data_dir=tmp_path / "silver",
    )
    out_root = tmp_path / "artifacts" / "data_pipeline"
    dynamic_out = out_root / "dynamic_calendar"
    dynamic_out.mkdir(parents=True)
    pd.DataFrame([{"ticker": "GOOD"}, {"ticker": "BAD"}]).to_csv(
        dynamic_out / "earnings_calendar_candidates.csv", index=False
    )
    raw_dir = config.bronze_data_dir / "sec" / "companyfacts"
    raw_dir.mkdir(parents=True)
    (raw_dir / "CIK0000000001.json").write_text(
        json.dumps(
            {
                "facts": {
                    "us-gaap": {
                        "Assets": {
                            "units": {
                                "USD": [
                                    {
                                        "accn": "GOOD-1",
                                        "val": 100.0,
                                        "filed": "2025-01-03",
                                        "end": "2024-12-31",
                                        "fy": 2024,
                                        "fp": "FY",
                                        "form": "10-K",
                                    }
                                ]
                            }
                        }
                    }
                }
            }
        ),
        encoding="utf-8",
    )
    bad_cache = raw_dir / "CIK0000000002.json"
    bad_cache.write_text("{bad-json", encoding="utf-8")

    class FakeClient:
        def __init__(self, *, timeout: float) -> None:
            _ = timeout

        def __enter__(self) -> FakeClient:
            return self

        def __exit__(
            self,
            exc_type: type[BaseException] | None,
            exc: BaseException | None,
            tb: object,
        ) -> None:
            _ = exc_type, exc, tb
            return None

    monkeypatch.setattr("earnings_event_vol.data_pipeline.httpx.Client", FakeClient)
    monkeypatch.setattr(
        "earnings_event_vol.data_pipeline.fetch_sec_ticker_map",
        lambda client, cfg: {"GOOD": 1, "BAD": 2},
    )
    monkeypatch.setattr(
        "earnings_event_vol.data_pipeline.fetch_sec_submission_payloads",
        lambda **kwargs: {},
    )

    step = data_pipeline._sec_companyfacts_step(config, out_root=out_root, force=False)

    assert step.status == "ran"
    facts = pd.read_parquet(config.silver_data_dir / "sec" / "companyfacts.parquet")
    assert facts["ticker"].tolist() == ["GOOD"]
    diagnostics = pd.read_csv(out_root / "sec_companyfacts" / "sec_companyfacts_diagnostics.csv")
    assert set(diagnostics["status"]) == {"cache_hit", "ticker_failed_graceful_degradation"}


def test_sec_companyfacts_stage_graceful_degradation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = replace(
        load_project_config(),
        bronze_data_dir=tmp_path / "bronze",
        silver_data_dir=tmp_path / "silver",
    )
    out_root = tmp_path / "artifacts" / "data_pipeline"
    dynamic_out = out_root / "dynamic_calendar"
    dynamic_out.mkdir(parents=True)
    pd.DataFrame([{"ticker": "AAPL"}]).to_csv(
        dynamic_out / "earnings_calendar_candidates.csv", index=False
    )

    class FakeClient:
        def __init__(self, *, timeout: float) -> None:
            _ = timeout

        def __enter__(self) -> FakeClient:
            return self

        def __exit__(
            self,
            exc_type: type[BaseException] | None,
            exc: BaseException | None,
            tb: object,
        ) -> None:
            _ = exc_type, exc, tb
            return None

    def failing_ticker_map(client: object, cfg: object) -> dict[str, int]:
        _ = client, cfg
        raise RuntimeError("sec unavailable")

    monkeypatch.setattr("earnings_event_vol.data_pipeline.httpx.Client", FakeClient)
    monkeypatch.setattr("earnings_event_vol.data_pipeline.fetch_sec_ticker_map", failing_ticker_map)

    step = data_pipeline._sec_companyfacts_step(config, out_root=out_root, force=True)

    assert step.status == "ran"
    assert step.reason == "sec_companyfacts_failed_graceful_degradation"
    facts = pd.read_parquet(config.silver_data_dir / "sec" / "companyfacts.parquet")
    assert list(facts.columns) == ["ticker", "cik", "feature_concept"]
    diagnostics = pd.read_csv(out_root / "sec_companyfacts" / "sec_companyfacts_diagnostics.csv")
    assert diagnostics.loc[0, "status"] == "http_or_parse_degraded"
    manifest = json.loads(
        (out_root / "sec_companyfacts" / "sec_companyfacts_manifest.json").read_text(
            encoding="utf-8"
        )
    )
    assert manifest["status"] == "degraded"
    assert manifest["reason"] == "sec_companyfacts_failed_graceful_degradation"


def test_data_pipeline_all_stops_after_blocked_upstream_stage(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = replace(load_project_config(), repo_root=tmp_path)
    out_root = tmp_path / "artifacts" / "data_pipeline"

    def fake_bulk(*args: object, **kwargs: object) -> DataPipelineStep:
        return DataPipelineStep("options-day-aggs-bulk", "ran")

    def blocked_universe(*args: object, **kwargs: object) -> DataPipelineStep:
        return DataPipelineStep("universe", "blocked", reason="missing bulk artifact")

    monkeypatch.setattr("earnings_event_vol.data_pipeline._options_day_aggs_bulk_step", fake_bulk)
    monkeypatch.setattr("earnings_event_vol.data_pipeline._universe_step", blocked_universe)

    result = run_data_pipeline(
        config,
        stage="all",
        out_root=out_root,
        force=False,
    )

    steps = _pipeline_steps(result)
    assert result["ok"] is False
    assert [step["name"] for step in steps] == ["options-day-aggs-bulk", "universe"]
    assert [step["status"] for step in steps] == ["ran", "blocked"]


def test_trade_proxy_second_aggregates_are_cached_in_bronze_parquet(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = replace(load_project_config(), bronze_data_dir=tmp_path / "data" / "bronze")
    option_ticker = "O:ABC260213C00100000"
    contracts = pd.DataFrame(
        {
            "options_ticker": [option_ticker, option_ticker],
            "entry_date": ["2026-02-05", "2026-02-06"],
            "event_entry_timestamp": [
                pd.Timestamp("2026-02-05 16:00:00", tz="America/New_York"),
                pd.Timestamp("2026-02-06 16:00:00", tz="America/New_York"),
            ],
            "event_id": ["ABC_1", "ABC_2"],
            "ticker": ["ABC", "ABC"],
            "exit_date": ["2026-02-06", "2026-02-09"],
            "expiration": [date(2026, 2, 13), date(2026, 2, 13)],
            "strike": [100.0, 100.0],
            "right": ["call", "call"],
            "dte": [8, 7],
            "moneyness_abs": [0.0, 0.0],
        }
    )
    fetch_calls = 0

    def fake_fetch_one_contract(
        config: object,
        *,
        option_ticker: str,
        entry_date: pd.Timestamp,
        limit: int,
    ) -> tuple[str, pd.DataFrame, dict[str, object]]:
        nonlocal fetch_calls
        fetch_calls += 1
        price = 4.05 if entry_date.date() == date(2026, 2, 5) else 9.05
        normalized = pd.DataFrame(
            {
                "options_ticker": [option_ticker],
                "timestamp_utc": [
                    pd.Timestamp(
                        f"{entry_date.date().isoformat()} 20:59:58",
                        tz="UTC",
                    )
                ],
                "timestamp_et": [
                    pd.Timestamp(
                        f"{entry_date.date().isoformat()} 15:59:58",
                        tz="America/New_York",
                    )
                ],
                "option_open": [price - 0.05],
                "option_high": [price + 0.15],
                "option_low": [price - 0.15],
                "option_close": [price + 0.05],
                "option_vwap": [price],
                "volume": [3],
                "transactions": [2],
                "source_dataset": ["massive_rest_second_aggs"],
            }
        )
        return (
            option_ticker,
            normalized,
            {
                "options_ticker": option_ticker,
                "status": "ok",
                "rows": len(normalized),
                "entry_date": entry_date.date(),
            },
        )

    monkeypatch.setattr(
        trade_proxy_panel_script,
        "_fetch_one_contract",
        fake_fetch_one_contract,
    )

    first_frames, first_report = trade_proxy_panel_script._fetch_second_aggregate_bars(
        config,
        contracts,
        jobs=1,
        limit=100,
        buffer_minutes=60,
        force=False,
    )
    first_key = (
        option_ticker,
        date(2026, 2, 5),
        pd.Timestamp("2026-02-05 16:00:00", tz="America/New_York").isoformat(),
    )
    second_key = (
        option_ticker,
        date(2026, 2, 6),
        pd.Timestamp("2026-02-06 16:00:00", tz="America/New_York").isoformat(),
    )
    assert fetch_calls == 2
    assert first_report["cache_status"].tolist() == ["written", "written"]
    assert Path(first_report["bronze_path"].iloc[0]).exists()
    assert set(first_frames) == {first_key, second_key}
    assert first_frames[first_key]["option_vwap"].iloc[0] == pytest.approx(4.05)
    assert first_frames[second_key]["option_vwap"].iloc[0] == pytest.approx(9.05)
    proxy_prices = build_trade_proxy_price_frame(contracts, first_frames)
    assert proxy_prices["proxy_price"].tolist() == pytest.approx([4.05, 9.05])

    second_frames, second_report = trade_proxy_panel_script._fetch_second_aggregate_bars(
        config,
        contracts,
        jobs=1,
        limit=100,
        buffer_minutes=60,
        force=False,
    )
    assert fetch_calls == 2
    assert second_report["cache_status"].tolist() == ["hit", "hit"]
    assert second_frames[first_key]["option_close"].iloc[0] == pytest.approx(4.1)
    assert second_frames[second_key]["option_close"].iloc[0] == pytest.approx(9.1)

    Path(first_report["bronze_path"].iloc[0]).write_text("not a parquet file", encoding="utf-8")
    repaired_frames, repaired_report = trade_proxy_panel_script._fetch_second_aggregate_bars(
        config,
        contracts,
        jobs=1,
        limit=100,
        buffer_minutes=60,
        force=False,
    )
    assert fetch_calls == 3
    assert repaired_report["cache_status"].tolist() == ["repaired", "hit"]
    assert repaired_frames[first_key]["option_close"].iloc[0] == pytest.approx(4.1)
    assert repaired_frames[second_key]["option_close"].iloc[0] == pytest.approx(9.1)


def test_second_aggregates_trade_proxy_panel_requires_pre_cutoff_trades() -> None:
    raw = pd.DataFrame(
        {
            "t": [
                pd.Timestamp("2026-02-05 20:59:50Z").value // 1_000_000,
                pd.Timestamp("2026-02-05 20:59:58Z").value // 1_000_000,
                pd.Timestamp("2026-02-05 21:00:01Z").value // 1_000_000,
            ],
            "o": [4.0, 4.2, 4.8],
            "h": [4.0, 4.2, 4.8],
            "l": [4.0, 4.2, 4.8],
            "c": [4.0, 4.2, 4.8],
            "vw": [4.0, 4.2, 4.8],
            "v": [1, 2, 99],
            "n": [1, 1, 1],
        }
    )
    bars = normalize_second_aggregates(raw, option_ticker="O:ABC260213C00100000")
    selected = select_preclose_entry_proxy_price(
        bars,
        cutoff_timestamp=pd.Timestamp("2026-02-05 16:00:00", tz="America/New_York").to_pydatetime(),
        lookback_seconds=30,
    )
    assert selected.status == TRADE_PROXY_STATUS_OK
    assert selected.proxy_price == pytest.approx((4.0 * 1 + 4.2 * 2) / 3)
    assert selected.price_method == "preclose_15m_option_second_agg_vwap"
    assert selected.proxy_volume == 3
    assert selected.proxy_rows_in_window == 2

    stale = select_preclose_entry_proxy_price(
        bars,
        cutoff_timestamp=pd.Timestamp("2026-02-05 16:00:00", tz="America/New_York").to_pydatetime(),
        lookback_seconds=1,
    )
    assert stale.status == TRADE_PROXY_STATUS_NO_TRADE_IN_WINDOW
    buffered = filter_pre_cutoff_buffer(
        bars,
        cutoff_timestamp=pd.Timestamp("2026-02-05 16:00:00", tz="America/New_York").to_pydatetime(),
        buffer_minutes=1,
    )
    assert buffered["option_vwap"].tolist() == [4.0, 4.2]
    early_close = market_close_timestamp(
        date(2026, 11, 27), early_closes={date(2026, 11, 27): time(13)}
    )
    assert is_us_equity_trading_day(date(2026, 11, 27))
    assert not is_us_equity_trading_day(date(2026, 11, 26))
    assert not is_us_equity_trading_day(date(2022, 6, 20))
    assert not is_us_equity_trading_day(date(2025, 1, 9))
    assert previous_us_equity_trading_day(date(2025, 1, 10)) == date(2025, 1, 8)
    assert previous_us_equity_trading_day(date(2026, 11, 27)) == date(2026, 11, 25)
    assert next_us_equity_trading_day(date(2026, 11, 26)) == date(2026, 11, 27)
    assert early_close.hour == 13
    assert market_close_timestamp(date(2026, 11, 27)).hour == 13
    assert market_close_timestamp(date(2026, 12, 24)).hour == 13
    assert regular_close_timestamp(date(2026, 2, 5)).hour == 16
    assert (
        market_close_timestamp_utc(
            date(2026, 11, 27), early_closes={date(2026, 11, 27): time(13)}
        ).tzinfo
        is not None
    )

    half_day_ticker = "O:ABC261218C00100000"
    half_day_proxy = build_trade_proxy_price_frame(
        pd.DataFrame(
            {
                "event_id": ["ABC_half_day"],
                "ticker": ["ABC"],
                "entry_date": [date(2026, 11, 27)],
                "exit_date": [date(2026, 11, 30)],
                "expiration": [date(2026, 12, 18)],
                "strike": [100.0],
                "right": ["call"],
                "options_ticker": [half_day_ticker],
                "dte": [21],
                "moneyness_abs": [0.0],
            }
        ),
        {
            half_day_ticker: pd.DataFrame(
                {
                    "options_ticker": [half_day_ticker] * 3,
                    "timestamp_et": [
                        pd.Timestamp("2026-11-27 12:50:00", tz="America/New_York"),
                        pd.Timestamp("2026-11-27 12:59:00", tz="America/New_York"),
                        pd.Timestamp("2026-11-27 15:59:00", tz="America/New_York"),
                    ],
                    "option_vwap": [5.0, 7.0, 100.0],
                    "option_close": [5.0, 7.0, 100.0],
                    "volume": [1, 3, 99],
                    "transactions": [1, 3, 1],
                }
            )
        },
    )
    assert half_day_proxy["proxy_price"].iloc[0] == pytest.approx((5.0 * 1 + 7.0 * 3) / 4)
    assert half_day_proxy["proxy_status"].iloc[0] == TRADE_PROXY_STATUS_OK


def test_post_open_option_vwap_windows_use_trade_weighted_prices() -> None:
    exit_date = date(2026, 2, 6)
    selected = pd.DataFrame(
        {
            "event_id": ["ABC_2026Q1"],
            "exit_date": [exit_date],
            "call_options_ticker": ["O:ABC260220C00100000"],
            "put_options_ticker": ["O:ABC260220P00100000"],
        }
    )

    def bars(ticker: str, first: float, second: float, third: float) -> pd.DataFrame:
        return pd.DataFrame(
            {
                "options_ticker": [ticker, ticker, ticker],
                "timestamp_et": [
                    pd.Timestamp("2026-02-06 09:31:00", tz="America/New_York"),
                    pd.Timestamp("2026-02-06 09:36:00", tz="America/New_York"),
                    pd.Timestamp("2026-02-06 09:40:00", tz="America/New_York"),
                ],
                "option_vwap": [first, second, third],
                "option_close": [first, second, third],
                "volume": [1, 2, 3],
                "transactions": [1, 2, 3],
            }
        )

    call_ticker = "O:ABC260220C00100000"
    put_ticker = "O:ABC260220P00100000"
    out = build_post_open_option_vwap_frame(
        selected,
        {
            (call_ticker, exit_date): bars(call_ticker, 5.0, 6.0, 7.0),
            (put_ticker, exit_date): bars(put_ticker, 4.0, 5.0, 6.0),
        },
    )
    assert set(out["window_label"]) == {"0_5", "5_15"}
    assert set(out["panel_grade"]) == {TRADE_PROXY_PANEL_GRADE}
    call_5_15 = out.loc[
        out["options_ticker"].eq(call_ticker) & out["window_label"].eq("5_15"),
        "option_exit_vwap",
    ].iloc[0]
    assert call_5_15 == pytest.approx((6.0 * 2 + 7.0 * 3) / 5)

    direct = select_option_window_vwap(
        bars(call_ticker, 5.0, 6.0, 7.0),
        window_start=pd.Timestamp("2026-02-06 09:30:00", tz="America/New_York").to_pydatetime(),
        window_end=pd.Timestamp("2026-02-06 09:35:00", tz="America/New_York").to_pydatetime(),
        include_end=False,
    )
    assert direct.proxy_price == pytest.approx(5.0)
    assert direct.proxy_volume == 1

    exit_out = build_exit_preclose_option_vwap_frame(
        selected,
        {
            (call_ticker, exit_date): pd.DataFrame(
                {
                    "options_ticker": [call_ticker, call_ticker, call_ticker],
                    "timestamp_et": [
                        pd.Timestamp("2026-02-06 15:44:00", tz="America/New_York"),
                        pd.Timestamp("2026-02-06 15:46:00", tz="America/New_York"),
                        pd.Timestamp("2026-02-06 15:59:00", tz="America/New_York"),
                    ],
                    "option_vwap": [4.0, 6.0, 8.0],
                    "option_close": [4.0, 6.0, 8.0],
                    "volume": [99, 2, 3],
                    "transactions": [1, 2, 3],
                }
            ),
            (put_ticker, exit_date): pd.DataFrame(
                {
                    "options_ticker": [put_ticker, put_ticker],
                    "timestamp_et": [
                        pd.Timestamp("2026-02-06 15:50:00", tz="America/New_York"),
                        pd.Timestamp("2026-02-06 15:58:00", tz="America/New_York"),
                    ],
                    "option_vwap": [5.0, 7.0],
                    "option_close": [5.0, 7.0],
                    "volume": [1, 3],
                    "transactions": [1, 3],
                }
            ),
        },
    )
    call_exit = exit_out.loc[exit_out["options_ticker"].eq(call_ticker), "option_exit_vwap"].iloc[0]
    put_exit = exit_out.loc[exit_out["options_ticker"].eq(put_ticker), "option_exit_vwap"].iloc[0]
    assert set(exit_out["panel_grade"]) == {TRADE_PROXY_PANEL_GRADE}
    assert call_exit == pytest.approx((6.0 * 2 + 8.0 * 3) / 5)
    assert put_exit == pytest.approx((5.0 * 1 + 7.0 * 3) / 4)

    half_day = date(2026, 11, 27)
    half_call = "O:ABC261218C00100000"
    half_put = "O:ABC261218P00100000"
    half_day_out = build_exit_preclose_option_vwap_frame(
        pd.DataFrame(
            {
                "event_id": ["ABC_half_day"],
                "exit_date": [half_day],
                "call_options_ticker": [half_call],
                "put_options_ticker": [half_put],
            }
        ),
        {
            (half_call, half_day): pd.DataFrame(
                {
                    "options_ticker": [half_call] * 4,
                    "timestamp_et": [
                        pd.Timestamp("2026-11-27 12:44:00", tz="America/New_York"),
                        pd.Timestamp("2026-11-27 12:46:00", tz="America/New_York"),
                        pd.Timestamp("2026-11-27 12:59:00", tz="America/New_York"),
                        pd.Timestamp("2026-11-27 15:59:00", tz="America/New_York"),
                    ],
                    "option_vwap": [100.0, 6.0, 8.0, 100.0],
                    "option_close": [100.0, 6.0, 8.0, 100.0],
                    "volume": [99, 2, 3, 99],
                    "transactions": [1, 2, 3, 1],
                }
            ),
            (half_put, half_day): pd.DataFrame(
                {
                    "options_ticker": [half_put, half_put],
                    "timestamp_et": [
                        pd.Timestamp("2026-11-27 12:50:00", tz="America/New_York"),
                        pd.Timestamp("2026-11-27 12:58:00", tz="America/New_York"),
                    ],
                    "option_vwap": [5.0, 7.0],
                    "option_close": [5.0, 7.0],
                    "volume": [1, 3],
                    "transactions": [1, 3],
                }
            ),
        },
    )
    half_call_exit = half_day_out.loc[
        half_day_out["options_ticker"].eq(half_call), "option_exit_vwap"
    ].iloc[0]
    assert half_call_exit == pytest.approx((6.0 * 2 + 8.0 * 3) / 5)


def test_trade_proxy_window_validation_edge_cases() -> None:
    start = pd.Timestamp("2026-02-06 09:30:00", tz="America/New_York").to_pydatetime()
    end = pd.Timestamp("2026-02-06 09:35:00", tz="America/New_York").to_pydatetime()
    cutoff = pd.Timestamp("2026-02-05 16:00:00", tz="America/New_York").to_pydatetime()

    with pytest.raises(ValueError, match="buffer_minutes"):
        filter_pre_cutoff_buffer(pd.DataFrame(), cutoff_timestamp=cutoff, buffer_minutes=0)
    with pytest.raises(ValueError, match="timestamp_et"):
        filter_pre_cutoff_buffer(
            pd.DataFrame({"option_vwap": [1.0]}),
            cutoff_timestamp=cutoff,
            buffer_minutes=5,
        )
    with pytest.raises(ValueError, match="timezone-aware"):
        filter_pre_cutoff_buffer(
            pd.DataFrame({"timestamp_et": ["2026-02-05 15:59:00"], "option_vwap": [1.0]}),
            cutoff_timestamp=cutoff,
            buffer_minutes=5,
        )

    with pytest.raises(ValueError, match="price_field"):
        select_option_window_vwap(
            pd.DataFrame(), window_start=start, window_end=end, price_field="bad"
        )
    with pytest.raises(ValueError, match="window_end"):
        select_option_window_vwap(pd.DataFrame(), window_start=start, window_end=start)
    with pytest.raises(ValueError, match="timestamp_et"):
        select_option_window_vwap(
            pd.DataFrame({"option_vwap": [1.0], "volume": [1]}),
            window_start=start,
            window_end=end,
        )
    with pytest.raises(ValueError, match="timezone-aware"):
        select_option_window_vwap(
            pd.DataFrame(
                {"timestamp_et": ["2026-02-06 09:31:00"], "option_vwap": [1.0], "volume": [1]}
            ),
            window_start=start,
            window_end=end,
        )

    assert build_post_open_option_vwap_frame(pd.DataFrame(), {}).empty
    assert build_exit_preclose_option_vwap_frame(pd.DataFrame(), {}).empty
    with pytest.raises(ValueError, match="post-open option price frame"):
        trade_proxy_module._post_open_option_vwap_lookup(pd.DataFrame({"event_id": ["x"]}))
    with pytest.raises(ValueError, match="exit preclose option price frame"):
        trade_proxy_module._exit_preclose_option_vwap_lookup(pd.DataFrame({"event_id": ["x"]}))


def test_trade_proxy_validation_branches_and_summaries(tmp_path: Path) -> None:
    empty_normalized = normalize_second_aggregates(pd.DataFrame(), option_ticker="O:ABC")
    assert list(empty_normalized.columns)[0] == "options_ticker"
    with pytest.raises(ValueError, match="missing required columns"):
        normalize_second_aggregates(pd.DataFrame({"t": [1]}), option_ticker="O:ABC")
    with pytest.raises(ValueError, match="lookback_seconds"):
        select_preclose_entry_proxy_price(
            pd.DataFrame(), cutoff_timestamp=datetime(2026, 2, 5), lookback_seconds=0
        )
    with pytest.raises(ValueError, match="price_field"):
        select_preclose_entry_proxy_price(
            pd.DataFrame(), cutoff_timestamp=datetime(2026, 2, 5), price_field="bad"
        )
    with pytest.raises(ValueError, match="timestamp_et"):
        select_preclose_entry_proxy_price(
            pd.DataFrame({"option_vwap": [1.0]}),
            cutoff_timestamp=datetime(2026, 2, 5),
        )
    assert (
        select_preclose_entry_proxy_price(
            pd.DataFrame(),
            cutoff_timestamp=datetime(2026, 2, 5),
        ).status
        == TRADE_PROXY_STATUS_NO_TRADE_IN_WINDOW
    )
    with pytest.raises(ValueError, match="timezone-aware"):
        select_preclose_entry_proxy_price(
            pd.DataFrame(
                {
                    "timestamp_et": ["2026-02-05 15:59:59"],
                    "option_vwap": [1.25],
                    "volume": [1],
                    "transactions": [1],
                }
            ),
            cutoff_timestamp=datetime(2026, 2, 5, 16, 0),
        )
    with pytest.raises(ValueError, match="contract frame missing"):
        build_trade_proxy_price_frame(pd.DataFrame({"event_id": ["x"]}), {})

    windows = pd.DataFrame({"event_id": ["ABC_2026Q1"], "ticker": ["ABC"], "s_before": [100.0]})
    invalid_proxy_iv = attach_trade_proxy_local_iv(
        pd.DataFrame(
            {
                "event_id": ["ABC_2026Q1", "ABC_2026Q1", "ABC_2026Q1", "ABC_2026Q1"],
                "proxy_status": [
                    TRADE_PROXY_STATUS_NO_TRADE_IN_WINDOW,
                    TRADE_PROXY_STATUS_OK,
                    TRADE_PROXY_STATUS_OK,
                    TRADE_PROXY_STATUS_OK,
                ],
                "proxy_price": [None, 1.0, 0.0, 10_000.0],
                "strike": [100.0, 120.0, 100.0, 100.0],
                "dte": [8, 8, 8, 8],
                "right": ["call", "put", "call", "call"],
            }
        ),
        windows,
    )
    assert invalid_proxy_iv["local_iv_status"].tolist() == [
        TRADE_PROXY_STATUS_NO_TRADE_IN_WINDOW,
        "price_below_intrinsic",
        "invalid_iv_inputs",
        "iv_root_not_bracketed",
    ]

    empty_inputs = build_trade_proxy_ivar_inputs(pd.DataFrame(), windows)
    assert empty_inputs.empty
    no_pair_inputs = build_trade_proxy_ivar_inputs(
        pd.DataFrame(
            {
                "event_id": ["ABC_2026Q1"],
                "expiration": [date(2026, 2, 13)],
                "right": ["call"],
                "strike": [100.0],
                "local_iv": [0.5],
                "proxy_volume_window": [1],
                "proxy_transactions_window": [1],
            }
        ),
        pd.DataFrame({"event_id": ["ABC_2026Q1"], "s_before": [100.0]}),
    )
    assert no_pair_inputs.empty
    no_valid_inputs = build_trade_proxy_ivar_inputs(
        pd.DataFrame(
            {
                "event_id": ["ABC_2026Q1"],
                "expiration": [date(2026, 2, 13)],
                "right": ["call"],
                "strike": [100.0],
                "local_iv": [None],
                "proxy_volume_window": [1],
                "proxy_transactions_window": [1],
            }
        ),
        windows,
    )
    assert no_valid_inputs.empty

    assert build_proxy_straddle_diagnostics(pd.DataFrame(), windows).empty
    assert build_proxy_straddle_diagnostics(
        pd.DataFrame(
            {
                "event_id": ["missing"],
                "expiration": [date(2026, 2, 13)],
                "proxy_status": [TRADE_PROXY_STATUS_OK],
                "right": ["call"],
                "strike": [100.0],
            }
        ),
        windows,
    ).empty
    assert build_proxy_straddle_diagnostics(
        pd.DataFrame(
            {
                "event_id": ["ABC_2026Q1"],
                "expiration": [date(2026, 2, 13)],
                "proxy_status": [TRADE_PROXY_STATUS_OK],
                "right": ["call"],
                "strike": [100.0],
            }
        ),
        windows.assign(s_after=101.0, entry_date=date(2026, 2, 5), exit_date=date(2026, 2, 6)),
    ).empty

    exit_preclose_straddle = build_proxy_straddle_diagnostics(
        pd.DataFrame(
            {
                "event_id": ["ABC_2026Q1", "ABC_2026Q1"],
                "expiration": [date(2026, 2, 20), date(2026, 2, 20)],
                "options_ticker": ["O:ABC260220C00100000", "O:ABC260220P00100000"],
                "proxy_status": [TRADE_PROXY_STATUS_OK, TRADE_PROXY_STATUS_OK],
                "right": ["call", "put"],
                "strike": [100.0, 100.0],
                "proxy_price": [5.0, 4.0],
                "proxy_volume_window": [10, 20],
                "proxy_transactions_window": [1, 2],
            }
        ),
        windows.assign(
            s_after=101.0,
            open_after=105.0,
            entry_date=date(2026, 2, 5),
            exit_date=date(2026, 2, 6),
        ),
        exit_preclose_option_prices=pd.DataFrame(
            {
                "event_id": ["ABC_2026Q1", "ABC_2026Q1"],
                "options_ticker": ["O:ABC260220C00100000", "O:ABC260220P00100000"],
                "option_exit_vwap": [5.8, 4.4],
                "volume": [30, 40],
                "transactions": [3, 4],
                "rows_in_window": [2, 2],
                "status": ["ok", "ok"],
            }
        ),
        post_open_option_prices=pd.DataFrame(
            {
                "event_id": ["ABC_2026Q1"] * 4,
                "options_ticker": [
                    "O:ABC260220C00100000",
                    "O:ABC260220P00100000",
                    "O:ABC260220C00100000",
                    "O:ABC260220P00100000",
                ],
                "window_label": ["0_5", "0_5", "5_15", "5_15"],
                "option_exit_vwap": [5.5, 3.5, 6.0, 4.0],
                "volume": [10, 10, 20, 20],
                "transactions": [1, 1, 2, 2],
                "rows_in_window": [1, 1, 2, 2],
                "status": ["ok"] * 4,
            }
        ),
    )
    assert (
        exit_preclose_straddle["option_exit_price_status"].iloc[0]
        == EXIT_PRECLOSE_OPTION_VWAP_STATUS_OK
    )
    assert bool(exit_preclose_straddle["used_intrinsic_fallback"].iloc[0]) is False
    assert exit_preclose_straddle["exit_option_value_usd"].iloc[0] == pytest.approx(1020.0)
    assert exit_preclose_straddle["gross_proxy_pnl_usd"].iloc[0] == pytest.approx(120.0)
    assert exit_preclose_straddle["gross_exit_option_vwap_preclose_15m_proxy_pnl_usd"].iloc[
        0
    ] == pytest.approx(120.0)
    assert exit_preclose_straddle["c2o_exit_intrinsic_usd"].iloc[0] == pytest.approx(500.0)
    assert exit_preclose_straddle["gross_c2o_intrinsic_proxy_pnl_usd"].iloc[0] == pytest.approx(
        -400.0
    )
    assert exit_preclose_straddle["c2o_proxy_pnl_status"].iloc[0] == "vendor_open_intrinsic_proxy"
    assert (
        exit_preclose_straddle["c2o_proxy_pnl_source"].iloc[0]
        == "underlying_open_intrinsic_diagnostic_not_option_vwap"
    )
    assert exit_preclose_straddle["gross_post_open_option_vwap_0_5_proxy_pnl_usd"].iloc[
        0
    ] == pytest.approx(0.0)
    assert exit_preclose_straddle["gross_post_open_option_vwap_5_15_proxy_pnl_usd"].iloc[
        0
    ] == pytest.approx(100.0)
    assert exit_preclose_straddle["post_open_option_vwap_5_15_status"].iloc[0] == "ok"
    assert exit_preclose_straddle["open_option_vwap_5_15_anchor_usd"].iloc[0] == pytest.approx(
        1000.0
    )
    assert exit_preclose_straddle[
        "gross_reaction_o2c_option_vwap_5_15_to_c2c_exit_proxy_pnl_usd"
    ].iloc[0] == pytest.approx(20.0)
    assert exit_preclose_straddle["option_proxy_decomposition_residual_5_15_usd"].iloc[
        0
    ] == pytest.approx(0.0)
    assert exit_preclose_straddle[
        "gross_reaction_o2c_option_vwap_0_5_to_c2c_exit_proxy_pnl_usd"
    ].iloc[0] == pytest.approx(120.0)
    assert exit_preclose_straddle["option_proxy_decomposition_residual_0_5_usd"].iloc[
        0
    ] == pytest.approx(0.0)

    expiration_at_exit = build_proxy_straddle_diagnostics(
        pd.DataFrame(
            {
                "event_id": ["ABC_expiry", "ABC_expiry"],
                "expiration": [date(2026, 2, 6), date(2026, 2, 6)],
                "options_ticker": ["O:ABC260206C00100000", "O:ABC260206P00100000"],
                "proxy_status": [TRADE_PROXY_STATUS_OK, TRADE_PROXY_STATUS_OK],
                "right": ["call", "put"],
                "strike": [100.0, 100.0],
                "proxy_price": [5.0, 4.0],
                "proxy_volume_window": [10, 20],
                "proxy_transactions_window": [1, 2],
            }
        ),
        pd.DataFrame(
            {
                "event_id": ["ABC_expiry"],
                "ticker": ["ABC"],
                "s_before": [100.0],
                "s_after": [101.0],
                "open_after": [105.0],
                "entry_date": [date(2026, 2, 5)],
                "exit_date": [date(2026, 2, 6)],
            }
        ),
        exit_preclose_option_prices=pd.DataFrame(
            {
                "event_id": ["ABC_expiry", "ABC_expiry"],
                "options_ticker": ["O:ABC260206C00100000", "O:ABC260206P00100000"],
                "option_exit_vwap": [20.0, 20.0],
                "volume": [30, 40],
                "transactions": [3, 4],
                "rows_in_window": [2, 2],
                "status": ["ok", "ok"],
            }
        ),
    )
    assert expiration_at_exit["exit_option_value_usd"].iloc[0] == pytest.approx(100.0)
    assert expiration_at_exit["option_exit_price_status"].iloc[0] == "expiration_at_exit_intrinsic"
    assert bool(expiration_at_exit["used_intrinsic_fallback"].iloc[0]) is True

    missing_exit_preclose = build_proxy_straddle_diagnostics(
        pd.DataFrame(
            {
                "event_id": ["ABC_2026Q1", "ABC_2026Q1"],
                "expiration": [date(2026, 2, 20), date(2026, 2, 20)],
                "options_ticker": ["O:ABC260220C00100000", "O:ABC260220P00100000"],
                "proxy_status": [TRADE_PROXY_STATUS_OK, TRADE_PROXY_STATUS_OK],
                "right": ["call", "put"],
                "strike": [100.0, 100.0],
                "proxy_price": [5.0, 4.0],
                "proxy_volume_window": [10, 20],
                "proxy_transactions_window": [1, 2],
            }
        ),
        windows.assign(s_after=101.0, entry_date=date(2026, 2, 5), exit_date=date(2026, 2, 6)),
    )
    assert (
        missing_exit_preclose["option_exit_price_status"].iloc[0]
        == OPTION_EXIT_STATUS_MISSING_PRECLOSE_VWAP
    )
    assert bool(missing_exit_preclose["used_intrinsic_fallback"].iloc[0]) is True
    assert missing_exit_preclose["c2o_proxy_pnl_status"].iloc[0] == "missing_open_after"

    empty_panel = pd.DataFrame()
    empty_summary = summarize_trade_proxy_panel(
        panel=empty_panel,
        proxy_prices=pd.DataFrame(),
        straddle_diagnostics=pd.DataFrame(),
        lookback_seconds=900,
        price_field="option_vwap",
    )
    assert empty_summary["paper_grade"] is False
    assert "bid/ask" in cast(list[str], empty_summary["limitations"])[0]
    assert edge_decile_diagnostics(empty_panel).empty
    assert edge_decile_diagnostics(pd.DataFrame({"edge_var_realized": [None]})).empty
    deciles = edge_decile_diagnostics(
        pd.DataFrame(
            {
                "event_id": ["a", "b", "c"],
                "edge_var_realized": [0.1, -0.2, 0.3],
            }
        )
    )
    assert int(deciles["count"].sum()) == 3

    metadata_path = tmp_path / "trade_proxy" / "metadata.json"
    write_trade_proxy_metadata(metadata_path, {"ok": True, "date": date(2026, 2, 5)})
    assert json.loads(metadata_path.read_text(encoding="utf-8"))["ok"] is True


def test_massive_second_aggregates_fetch_uses_encoded_contract_and_api_key(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    secret = tmp_path / "massive_api_key"
    secret.write_text("secret-key\n", encoding="utf-8")
    config = replace(
        load_project_config(),
        massive_api_key_file=secret,
        massive_base_url="https://api.massive.test",
        massive_request_timeout_seconds=12.0,
    )
    captured: dict[str, object] = {}

    class FakeResponse:
        def raise_for_status(self) -> None:
            return None

        def json(self) -> dict[str, object]:
            return {"results": [{"t": 1, "o": 2, "h": 2, "l": 2, "c": 2, "v": 1, "vw": 2, "n": 1}]}

    class FakeClient:
        def __init__(self, *, timeout: float) -> None:
            captured["timeout"] = timeout

        def __enter__(self) -> FakeClient:
            return self

        def __exit__(self, *args: object) -> None:
            return None

        def get(self, url: str, *, params: dict[str, object]) -> FakeResponse:
            captured["url"] = url
            captured["params"] = params
            return FakeResponse()

    monkeypatch.setattr("earnings_event_vol.trade_proxy.httpx.Client", FakeClient)
    result = fetch_massive_option_second_aggregates(
        config,
        option_ticker="O:ABC260213C00100000",
        trade_date=date(2026, 2, 5),
    )
    assert len(result) == 1
    assert "O%3AABC260213C00100000" in str(captured["url"])
    assert cast(dict[str, object], captured["params"])["apiKey"] == "secret-key"
    assert captured["timeout"] == 12.0

    retry_calls = {"count": 0}

    class RetryResponse:
        def __init__(self, request: httpx.Request, status_code: int) -> None:
            self.request = request
            self.status_code = status_code
            self.response = httpx.Response(
                status_code,
                json={"results": []}
                if status_code >= 400
                else {
                    "results": [{"t": 1, "o": 3, "h": 3, "l": 3, "c": 3, "v": 1, "vw": 3, "n": 1}]
                },
                request=request,
            )

        def raise_for_status(self) -> None:
            if self.status_code >= 400:
                raise httpx.HTTPStatusError(
                    "retryable",
                    request=self.request,
                    response=self.response,
                )

        def json(self) -> dict[str, object]:
            return cast(dict[str, object], self.response.json())

    class RetryClient:
        def __init__(self, *, timeout: float) -> None:
            self.timeout = timeout

        def __enter__(self) -> RetryClient:
            return self

        def __exit__(self, *args: object) -> None:
            return None

        def get(
            self,
            url: str,
            *,
            params: dict[str, str | int | float | bool | None],
        ) -> RetryResponse:
            retry_calls["count"] += 1
            request = httpx.Request("GET", url, params=params)
            return RetryResponse(request, 503 if retry_calls["count"] == 1 else 200)

    monkeypatch.setattr("earnings_event_vol.trade_proxy.httpx.Client", RetryClient)
    retry_result = fetch_massive_option_second_aggregates(
        replace(config, massive_max_retries=1, massive_retry_backoff_seconds=0),
        option_ticker="O:ABC260213C00100000",
        trade_date=date(2026, 2, 5),
    )
    assert retry_calls["count"] == 2
    assert retry_result["c"].tolist() == [3]

    class FailingResponse:
        def __init__(self, request: httpx.Request) -> None:
            self.request = request
            self.response = httpx.Response(
                403,
                text="denied for apiKey=secret-key",
                request=request,
            )

        def raise_for_status(self) -> None:
            raise httpx.HTTPStatusError(
                "blocked request https://api.massive.test?apiKey=secret-key",
                request=self.request,
                response=self.response,
            )

    class FailingClient:
        def __init__(self, *, timeout: float) -> None:
            self.timeout = timeout

        def __enter__(self) -> FailingClient:
            return self

        def __exit__(self, *args: object) -> None:
            return None

        def get(self, url: str, *, params: dict[str, object]) -> FailingResponse:
            query = urllib.parse.urlencode(params)
            return FailingResponse(httpx.Request("GET", f"{url}?{query}"))

    monkeypatch.setattr("earnings_event_vol.trade_proxy.httpx.Client", FailingClient)
    with pytest.raises(RuntimeError) as excinfo:
        fetch_massive_option_second_aggregates(
            config,
            option_ticker="O:ABC260213C00100000",
            trade_date=date(2026, 2, 5),
        )
    error_text = str(excinfo.value)
    assert "HTTP 403" in error_text
    assert "secret-key" not in error_text
    assert "apiKey=<redacted>" in error_text

    empty_secret = tmp_path / "empty_key"
    empty_secret.write_text("", encoding="utf-8")
    empty_key_config = replace(config, massive_api_key_file=empty_secret)
    with pytest.raises(ValueError, match="not configured"):
        fetch_massive_option_second_aggregates(
            empty_key_config,
            option_ticker="O:ABC260213C00100000",
            trade_date=date(2026, 2, 5),
        )

    empty_key_config = replace(config, massive_api_key_file=tmp_path / "missing")
    with pytest.raises(FileNotFoundError):
        fetch_massive_option_second_aggregates(
            empty_key_config,
            option_ticker="O:ABC260213C00100000",
            trade_date=date(2026, 2, 5),
        )


def test_trade_proxy_ivar_and_gross_straddle_diagnostics_are_no_nbbo() -> None:
    windows = pd.DataFrame(
        {
            "event_id": ["ABC_2026Q1"],
            "ticker": ["ABC"],
            "announcement_date": [date(2026, 2, 5)],
            "entry_date": [date(2026, 2, 5)],
            "exit_date": [date(2026, 2, 6)],
            "event_entry_timestamp": [
                pd.Timestamp("2026-02-05 16:00:00", tz="America/New_York").to_pydatetime()
            ],
            "s_before": [100.0],
            "s_after": [108.0],
            "rvar_event": [0.00616],
        }
    )
    contracts = pd.DataFrame(
        {
            "event_id": ["ABC_2026Q1"] * 4,
            "ticker": ["ABC"] * 4,
            "entry_date": [date(2026, 2, 5)] * 4,
            "exit_date": [date(2026, 2, 6)] * 4,
            "event_entry_timestamp": windows["event_entry_timestamp"].iloc[0],
            "expiration": [
                date(2026, 2, 13),
                date(2026, 2, 13),
                date(2026, 2, 20),
                date(2026, 2, 20),
            ],
            "strike": [100.0, 100.0, 100.0, 100.0],
            "right": ["call", "put", "call", "put"],
            "options_ticker": [
                "O:ABC260213C00100000",
                "O:ABC260213P00100000",
                "O:ABC260220C00100000",
                "O:ABC260220P00100000",
            ],
            "dte": [8, 8, 15, 15],
            "moneyness_abs": [0.0, 0.0, 0.0, 0.0],
        }
    )
    bar_frames = {
        ticker: pd.DataFrame(
            {
                "options_ticker": [ticker],
                "timestamp_et": [pd.Timestamp("2026-02-05 15:59:55", tz="America/New_York")],
                "option_vwap": [price],
                "option_close": [price],
                "volume": [10],
                "transactions": [3],
            }
        )
        for ticker, price in zip(
            contracts["options_ticker"].tolist(),
            [6.0, 5.8, 7.0, 6.5],
            strict=True,
        )
    }

    proxy_prices = build_trade_proxy_price_frame(contracts, bar_frames)
    iv_estimates = attach_trade_proxy_local_iv(proxy_prices, windows)
    ivar_inputs = build_trade_proxy_ivar_inputs(iv_estimates, windows)
    panel = extract_trade_proxy_event_panel(ivar_inputs, windows)
    straddles = build_proxy_straddle_diagnostics(iv_estimates, windows)
    empty_input_panel = extract_trade_proxy_event_panel(
        build_trade_proxy_ivar_inputs(pd.DataFrame(), windows),
        windows,
    )

    assert proxy_prices["panel_grade"].eq(TRADE_PROXY_PANEL_GRADE).all()
    assert ivar_inputs["atm_selection_method"].eq("trade_proxy_call_put_average").all()
    assert panel["panel_grade"].iloc[0] == TRADE_PROXY_PANEL_GRADE
    assert panel["ivar_failure_reason"].iloc[0] is None
    assert panel["trade_proxy_ivar_event"].iloc[0] > 0
    assert empty_input_panel["trade_proxy_ivar_event"].isna().all()
    assert empty_input_panel["ivar_failure_reason"].iloc[0] == "no_two_event_covering_expiries"
    assert straddles["gross_proxy_pnl_usd"].iloc[0] == pytest.approx(-380.0)
    assert straddles["option_exit_price_status"].iloc[0] == OPTION_EXIT_STATUS_MISSING_PRECLOSE_VWAP
    assert bool(straddles["used_intrinsic_fallback"].iloc[0]) is True
    assert straddles["haircut_pnl_usd"].iloc[0] < straddles["gross_proxy_pnl_usd"].iloc[0]


def test_bmo_amc_close_to_close_alignment_and_halt_proxy() -> None:
    trading_dates = [date(2026, 2, 4), date(2026, 2, 5), date(2026, 2, 6)]
    amc = align_event_window(
        EarningsEvent(
            ticker="ABC",
            announcement_date=date(2026, 2, 5),
            announcement_timing=AnnouncementTiming.AMC,
            source="fixture",
        ),
        trading_dates,
    )
    assert amc.entry_date == date(2026, 2, 5)
    assert amc.exit_date == date(2026, 2, 6)
    assert amc.event_entry_timestamp == regular_close_timestamp(date(2026, 2, 5))
    assert str(amc.event_entry_timestamp.tzinfo) == "America/New_York"

    bmo = align_event_window(
        EarningsEvent(
            ticker="ABC",
            announcement_date=date(2026, 2, 5),
            announcement_timing=AnnouncementTiming.BMO,
            source="fixture",
        ),
        trading_dates,
    )
    assert bmo.entry_date == date(2026, 2, 4)
    assert bmo.exit_date == date(2026, 2, 5)

    missing_window = align_event_window(
        EarningsEvent(
            ticker="ABC",
            announcement_date=date(2026, 2, 4),
            announcement_timing=AnnouncementTiming.BMO,
            source="fixture",
        ),
        trading_dates,
    )
    assert missing_window.exclusion_reason == "missing_entry_or_exit_date"

    dmh = align_event_window(
        EarningsEvent(
            ticker="ABC",
            announcement_date=date(2026, 2, 5),
            announcement_timing=AnnouncementTiming.DMH,
            source="fixture",
        ),
        trading_dates,
    )
    assert dmh.exclusion_reason == "non_bmo_amc"

    before, after = rvar_prices_for_window(
        bmo,
        {
            ("ABC", date(2026, 2, 4)): UnderlyingBar(
                ticker="ABC", date=date(2026, 2, 4), open=99, high=101, low=98, close=100, volume=1
            ),
            ("ABC", date(2026, 2, 5)): UnderlyingBar(
                ticker="ABC",
                date=date(2026, 2, 5),
                open=101,
                high=106,
                low=100,
                close=105,
                volume=1,
            ),
        },
    )
    assert before == 100
    assert after == 105
    assert has_ex_dividend_between([date(2026, 2, 5)], start=bmo.entry_date, end=bmo.exit_date)
    assert not has_ex_dividend_between([date(2026, 2, 6)], start=bmo.entry_date, end=bmo.exit_date)

    vendor_halted, vendor_reason = is_halted_or_proxy_halted(
        UnderlyingBar(
            ticker="ABC",
            date=date(2026, 2, 5),
            open=10,
            high=11,
            low=9,
            close=10,
            volume=100,
            vendor_halt_flag=True,
        )
    )
    assert vendor_halted is True
    assert vendor_reason == "vendor_halt_flag"

    halted, reason = is_halted_or_proxy_halted(
        UnderlyingBar(
            ticker="ABC", date=date(2026, 2, 5), open=10, high=10, low=10, close=10, volume=0
        )
    )
    assert halted is True
    assert reason == "proxy_zero_volume_unchanged_ohlc"

    active, active_reason = is_halted_or_proxy_halted(
        UnderlyingBar(
            ticker="ABC", date=date(2026, 2, 5), open=10, high=11, low=9, close=10, volume=1
        )
    )
    assert active is False
    assert active_reason is None


def test_leakage_audit_blocks_late_asof_and_vendor_forecasts() -> None:
    frame = pd.DataFrame(
        {
            "ticker": ["ABC"],
            "feature_asof_timestamp": [datetime(2026, 2, 5, 16, 1)],
            "event_entry_timestamp": [datetime(2026, 2, 5, 16, 0)],
            "vendor_alpha_forecast": [1.0],
            "same_event_return": [0.1],
            "gross_proxy_pnl_usd": [10.0],
            "exit_option_value_usd": [500.0],
        }
    )
    result = audit_feature_leakage(frame)
    assert result.ok is False
    assert len(result.asof_violations) == 1
    assert "vendor_alpha_forecast" in result.vendor_forecast_columns
    assert "same_event_return" in result.blocked_columns
    assert "gross_proxy_pnl_usd" in result.blocked_columns
    assert "exit_option_value_usd" in result.blocked_columns


def test_leakage_audit_flags_timezone_mismatch() -> None:
    frame = pd.DataFrame(
        {
            "ticker": ["ABC"],
            "feature_asof_timestamp": [make_feature_timestamps("2026-02-05").isoformat()],
            "event_entry_timestamp": ["2026-02-05T16:00:00"],
        }
    )

    result = audit_feature_leakage(frame)

    assert result.ok is False
    assert result.asof_violations["leakage_audit_reason"].iloc[0] == "timezone_mismatch"


def test_leakage_and_calendar_fail_closed_on_bad_inputs() -> None:
    with pytest.raises(ValueError, match="feature frame must include"):
        audit_feature_leakage(pd.DataFrame({"ticker": ["ABC"]}))

    mixed_tz = pd.DataFrame(
        {
            "feature_asof_timestamp": [
                "2026-02-05T15:59:00-05:00",
                "2026-02-05T15:59:00",
            ],
            "event_entry_timestamp": [
                "2026-02-05T16:00:00-05:00",
                "2026-02-05T16:00:00-05:00",
            ],
        }
    )
    result = audit_feature_leakage(mixed_tz)
    assert result.ok is False
    assert result.asof_violations["leakage_audit_reason"].eq("timezone_mismatch").all()

    with pytest.raises(ValueError, match="earnings calendar missing required columns"):
        validate_calendar_frame(pd.DataFrame({"ticker": ["ABC"]}))


def test_data_audit_detects_fields_and_vendor_local_iv_difference() -> None:
    options = pd.read_csv(FIXTURES / "option_quotes.csv")
    underlying = pd.read_csv(FIXTURES / "underlying_bars.csv")
    earnings = pd.read_csv(FIXTURES / "earnings_calendar.csv")
    result = audit_data_fields(options=options, underlying=underlying, earnings=earnings)
    assert result.required_fields_report["ok"] is True
    assert not result.vendor_local_iv_diff.empty
    assert vendor_local_iv_comparison(options)["mean_abs_vendor_local_iv_diff"].iloc[0] > 0
    missing_quote_source = audit_data_fields(
        options=options.drop(columns=["quote_source"]),
        underlying=underlying,
        earnings=earnings,
    )
    assert missing_quote_source.quote_source_report["quote_source"].iloc[0] == "missing"


def test_vendor_local_iv_comparison_derives_or_reports_bucket_inputs() -> None:
    options = pd.DataFrame(
        {
            "ticker": ["ABC"],
            "quote_date": ["2026-02-05"],
            "expiration": ["2026-02-12"],
            "strike": [100.0],
            "spot": [100.0],
            "vendor_iv": [0.80],
            "local_iv": [0.78],
        }
    )

    derived = vendor_local_iv_comparison(options)

    assert derived["status"].iloc[0] == "ok"
    assert derived["n"].iloc[0] == 1

    skipped = vendor_local_iv_comparison(options.drop(columns=["spot"]))
    assert skipped["status"].iloc[0] == "skipped"
    assert skipped["reason"].iloc[0] == "missing_required_columns"
    assert "moneyness" in skipped["missing_columns"].iloc[0]

    no_complete_rows = vendor_local_iv_comparison(options.assign(vendor_iv=None))
    assert no_complete_rows["status"].iloc[0] == "skipped"
    assert no_complete_rows["reason"].iloc[0] == "no_complete_vendor_local_iv_rows"


def test_features_universe_and_sequence_rules() -> None:
    assert iv_butterfly_25d(iv_25p=0.70, iv_atm=0.60, iv_25c=0.68) == pytest.approx(0.18)
    assert has_required_sequence_history(
        [date(2026, 1, day) for day in range(1, 22)],
        entry_date=date(2026, 1, 22),
        required_trading_days=20,
    )
    assert not has_required_sequence_history(
        [date(2026, 1, day) for day in range(15, 22)],
        entry_date=date(2026, 1, 22),
        required_trading_days=20,
    )
    universe = universe_by_trailing_option_dollar_volume(
        [
            {"ticker": "AAA", "option_dollar_volume": 10},
            {"ticker": "BBB", "option_dollar_volume": 100},
            {"ticker": "AAA", "option_dollar_volume": 20},
        ],
        top_n=1,
    )
    assert universe == ["BBB"]
    assert (
        len(
            universe_by_trailing_option_dollar_volume(
                [{"ticker": f"T{idx:03d}", "option_dollar_volume": idx} for idx in range(80)]
            )
        )
        == 50
    )


def test_eligible_equity_cache_version_and_ticker_mapping_soft_fail() -> None:
    eligible = build_eligible_equity_tickers(
        [
            {"ticker": "", "exchange": "NASDAQ", "title": "Blank Corporation"},
            {"ticker": "ABC", "exchange": "NASDAQ", "title": "ABC Corporation"},
            {"ticker": "SPY", "exchange": "NYSE ArCA", "title": "SPDR S&P 500 ETF Trust"},
            {"ticker": "OTC", "exchange": "OTC", "title": "OTC Company"},
        ],
        source_snapshot_date=date(2026, 5, 5),
    )

    assert eligible_equity_cache_matches_rule(eligible)
    assert not eligible_equity_cache_matches_rule(eligible, expected_rule_version="v9")
    assert eligible.loc[eligible["ticker"].eq("ABC"), "filter_reason"].iloc[0] == (
        "eligible_common_equity"
    )
    assert bool(eligible.loc[eligible["ticker"].eq("SPY"), "eligible"].iloc[0]) is False
    assert eligible["rule_version"].iloc[0] == ELIGIBLE_EQUITY_RULE_VERSION

    diagnostics = ticker_mapping_diagnostics(
        ["META", "FB", "GOOG"],
        ["META", "GOOGL", "GOOG"],
        aliases={"FB": ["META"], "GOOG": ["GOOG", "GOOGL"]},
    )
    assert diagnostics.loc[diagnostics["ticker"].eq("META"), "mapping_status"].iloc[0] == (
        TICKER_MAPPING_OK
    )
    assert diagnostics.loc[diagnostics["ticker"].eq("FB"), "mapped_ticker"].iloc[0] == "META"
    assert diagnostics.loc[diagnostics["ticker"].eq("GOOG"), "mapping_status"].iloc[0] == (
        TICKER_MAPPING_AMBIGUOUS
    )
    missing = ticker_mapping_diagnostics(["XYZ"], ["ABC"])
    assert missing["mapping_status"].iloc[0] == TICKER_NOT_FOUND


def test_data_pipeline_eligible_equity_cache_helpers(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    assert data_pipeline._sec_company_ticker_rows("bad") == []
    assert data_pipeline._sec_company_ticker_rows([{"ticker": "ABC"}, "bad"]) == [{"ticker": "ABC"}]
    assert data_pipeline._sec_company_ticker_rows(
        {
            "fields": ["ticker", "title", "exchange"],
            "data": [["ABC", "ABC Corporation", "NASDAQ"], "bad"],
        }
    ) == [{"ticker": "ABC", "title": "ABC Corporation", "exchange": "NASDAQ"}]

    assert data_pipeline._eligible_ticker_set(
        pd.DataFrame({"ticker": ["abc", "SPY", ""], "eligible": ["true", "false", "true"]})
    ) == {"ABC"}
    assert data_pipeline._eligible_ticker_set(
        pd.DataFrame({"ticker": ["ABC", "SPY"], "eligible": [1, 0]})
    ) == {"ABC"}
    assert data_pipeline._eligible_ticker_set(pd.DataFrame({"ticker": ["ABC"]})) == set()
    assert data_pipeline._filter_reason_counts(
        pd.DataFrame({"filter_reason": ["eligible_common_equity", "eligible_common_equity"]})
    ) == {"eligible_common_equity": 2}
    assert data_pipeline._filter_reason_counts(pd.DataFrame()) == {}

    missing_path = tmp_path / "missing.parquet"
    assert data_pipeline._read_valid_eligible_equity_cache(missing_path) is None
    corrupt = tmp_path / "corrupt.parquet"
    corrupt.write_text("not parquet", encoding="utf-8")
    assert data_pipeline._read_valid_eligible_equity_cache(corrupt) is None
    stale = tmp_path / "stale.parquet"
    pl.from_pandas(
        build_eligible_equity_tickers(
            [{"ticker": "ABC", "title": "ABC Corporation"}],
            source_snapshot_date=date(2026, 5, 6),
            rule_version="old",
        )
    ).write_parquet(stale)
    assert data_pipeline._read_valid_eligible_equity_cache(stale) is None

    class FakeResponse:
        def raise_for_status(self) -> None:
            return None

        def json(self) -> dict[str, dict[str, object]]:
            return {
                "0": {"ticker": "ABC", "title": "ABC Corporation", "cik_str": 1},
                "1": {"ticker": "SPY", "title": "SPDR S&P 500 ETF Trust", "cik_str": 2},
            }

    class FakeClient:
        def __init__(self, *args: object, **kwargs: object) -> None:
            self.requests: list[tuple[str, dict[str, str]]] = []

        def __enter__(self) -> FakeClient:
            return self

        def __exit__(self, *args: object) -> None:
            return None

        def get(self, url: str, *, headers: dict[str, str]) -> FakeResponse:
            self.requests.append((url, headers))
            return FakeResponse()

    monkeypatch.setattr(httpx, "Client", FakeClient)
    cache_path = tmp_path / "eligible.parquet"
    frame, status = data_pipeline._load_or_build_eligible_equity_cache(
        load_project_config(),
        path=cache_path,
        source_snapshot_date=date(2026, 5, 6),
    )
    assert status == "written"
    assert data_pipeline._eligible_ticker_set(frame) == {"ABC"}
    cached, cached_status = data_pipeline._load_or_build_eligible_equity_cache(
        load_project_config(),
        path=cache_path,
        source_snapshot_date=date(2026, 5, 6),
    )
    assert cached_status == "hit"
    assert data_pipeline._eligible_ticker_set(cached) == {"ABC"}

    class EmptyResponse(FakeResponse):
        def json(self) -> dict[str, dict[str, object]]:
            return {}

    class EmptyClient(FakeClient):
        def get(self, url: str, *, headers: dict[str, str]) -> EmptyResponse:
            return EmptyResponse()

    monkeypatch.setattr(httpx, "Client", EmptyClient)
    with pytest.raises(ValueError, match="no eligible-equity rows"):
        data_pipeline._load_or_build_eligible_equity_cache(
            load_project_config(),
            path=tmp_path / "empty.parquet",
            source_snapshot_date=date(2026, 5, 6),
        )


def test_monthly_universe_uses_trailing_option_premium_volume_only() -> None:
    liquidity = build_ticker_month_liquidity(
        pd.DataFrame(
            {
                "ticker": ["AAA", "AAA", "BBB", "CCC", "CCC"],
                "quote_date": [
                    date(2020, 1, 15),
                    date(2020, 2, 14),
                    date(2020, 2, 14),
                    date(2020, 3, 16),
                    date(2020, 4, 16),
                ],
                "option_vwap": [1.0, None, 5.0, 2.0, 100.0],
                "option_close": [1.1, 3.0, 4.0, 2.5, 100.0],
                "volume": [100, 10, 30, 40, 1],
            }
        ),
        source_snapshot_date=date(2026, 5, 5),
    )
    assert set(
        [
            "source_snapshot_date",
            "rule_version",
            "source_dataset",
            "option_premium_dollar_volume",
        ]
    ).issubset(liquidity.columns)
    feb_aaa = liquidity.loc[liquidity["ticker"].eq("AAA") & liquidity["month"].eq(date(2020, 2, 1))]
    assert feb_aaa["option_premium_dollar_volume"].iloc[0] == pytest.approx(3_000.0)

    universe = build_monthly_liquid_universe(
        liquidity,
        start_month=date(2020, 3, 1),
        end_month=date(2020, 5, 1),
        top_n=2,
        trailing_months=2,
        eligible_tickers=["AAA", "BBB", "CCC"],
    )
    march = universe.loc[universe["universe_month"].eq(date(2020, 3, 1))]
    assert march["ticker"].tolist() == ["BBB", "AAA"]
    may = universe.loc[universe["universe_month"].eq(date(2020, 5, 1))]
    assert may["ticker"].tolist() == ["CCC"]
    assert phase1_telemetry_bucket(date(2020, 3, 31)) == PHASE1_COVID_SHOCK_BUCKET
    assert phase1_telemetry_bucket(date(2020, 10, 1)) == PHASE1_STEADY_PROXY_BUCKET

    close_only = build_ticker_month_liquidity(
        pd.DataFrame(
            {
                "ticker": ["O:DDD200117C00100000", "O:BAD"],
                "source_date": [date(2020, 1, 2), date(2020, 1, 2)],
                "close": [2.0, None],
                "vwap": [None, 1.0],
                "volume": [5, 1],
            }
        ),
        source_snapshot_date=date(2026, 5, 5),
    )
    assert close_only["ticker"].iloc[0] == "DDD"
    assert close_only["ticker"].iloc[1] == "O:BAD"
    assert close_only["option_premium_dollar_volume"].iloc[0] == pytest.approx(1_000.0)
    with pytest.raises(ValueError, match="missing required columns"):
        build_ticker_month_liquidity(
            pd.DataFrame({"ticker": ["AAA"]}),
            source_snapshot_date=date(2026, 5, 5),
        )
    with pytest.raises(ValueError, match="requires vwap/option_vwap"):
        build_ticker_month_liquidity(
            pd.DataFrame({"ticker": ["AAA"], "quote_date": [date(2020, 1, 1)], "volume": [1]}),
            source_snapshot_date=date(2026, 5, 5),
        )
    with pytest.raises(ValueError, match="requires quote_date"):
        build_ticker_month_liquidity(
            pd.DataFrame({"ticker": ["AAA"], "option_close": [1.0], "volume": [1]}),
            source_snapshot_date=date(2026, 5, 5),
        )
    with pytest.raises(ValueError, match="top_n"):
        build_monthly_liquid_universe(
            liquidity,
            start_month=date(2020, 1, 1),
            end_month=date(2020, 1, 1),
            top_n=0,
        )
    with pytest.raises(ValueError, match="trailing_months"):
        build_monthly_liquid_universe(
            liquidity,
            start_month=date(2020, 1, 1),
            end_month=date(2020, 1, 1),
            trailing_months=0,
        )
    with pytest.raises(ValueError, match="missing required columns"):
        build_monthly_liquid_universe(
            pd.DataFrame({"ticker": ["AAA"]}),
            start_month=date(2020, 1, 1),
            end_month=date(2020, 1, 1),
        )


def test_premium_space_threshold_and_swappable_distribution() -> None:
    timestamp = datetime(2026, 2, 5, 16)
    legs = (
        TradeLeg(
            ticker="ABC",
            expiration=date(2026, 2, 13),
            strike=100,
            right=OptionRight.CALL,
            side=OptionSide.LONG,
            contracts=1.0,
            filled_price=4.0,
            filled_timestamp=timestamp,
        ),
        TradeLeg(
            ticker="ABC",
            expiration=date(2026, 2, 13),
            strike=100,
            right=OptionRight.PUT,
            side=OptionSide.LONG,
            contracts=1.0,
            filled_price=3.8,
            filled_timestamp=timestamp,
        ),
    )
    expected_gaussian = expected_strategy_value_usd(
        spot=100,
        forecast_rvar_event=0.04,
        legs=legs,
        distribution=GaussianEventJumpDistribution(nodes=11),
    )
    expected_twopoint = expected_strategy_value_usd(
        spot=100,
        forecast_rvar_event=0.04,
        legs=legs,
        distribution=SymmetricTwoPointJumpDistribution(),
    )
    assert expected_gaussian > 0
    assert expected_twopoint > 0
    entry_cost = market_entry_cost_usd(legs)
    signal = premium_space_signal(
        ticker="ABC",
        event_date=date(2026, 2, 5),
        strategy="long_atm_straddle",
        forecast_rvar_event=0.04,
        ivar_event=0.01,
        expected_value_usd=expected_gaussian,
        entry_cost_usd=entry_cost,
        transaction_cost_usd=20.0,
    )
    assert signal.edge_var == pytest.approx(0.03)
    assert signal.expected_strategy_edge_usd == pytest.approx(expected_gaussian - entry_cost)
    assert isinstance(signal.should_trade, bool)


def test_black_scholes_and_model_registry_protocol() -> None:
    call = black_scholes_price(
        spot=100,
        strike=100,
        time_to_expiry=30 / 365,
        volatility=0.30,
        right=OptionRight.CALL,
    )
    put = black_scholes_price(
        spot=100,
        strike=100,
        time_to_expiry=30 / 365,
        volatility=0.30,
        right=OptionRight.PUT,
    )
    intrinsic = black_scholes_price(
        spot=110,
        strike=100,
        time_to_expiry=0,
        volatility=0,
        right=OptionRight.CALL,
    )
    put_intrinsic = black_scholes_price(
        spot=90,
        strike=100,
        time_to_expiry=0,
        volatility=0,
        right=OptionRight.PUT,
    )
    assert call > 0
    assert put > 0
    assert intrinsic == pytest.approx(10.0)
    assert put_intrinsic == pytest.approx(10.0)
    with pytest.raises(ValueError, match="spot and strike"):
        black_scholes_price(
            spot=0,
            strike=100,
            time_to_expiry=30 / 365,
            volatility=0.30,
            right=OptionRight.CALL,
        )
    with pytest.raises(ValueError, match="variance must be nonnegative"):
        GaussianEventJumpDistribution().support(-0.01)
    with pytest.raises(ValueError, match="variance must be nonnegative"):
        SymmetricTwoPointJumpDistribution().support(-0.01)
    zero_returns, zero_probabilities = GaussianEventJumpDistribution().support(0.0)
    assert zero_returns.tolist() == [0.0]
    assert zero_probabilities.tolist() == [1.0]
    assert MODEL_REGISTRY["patell_wolfson_diagnostic"].role == "diagnostic"
    patell_text = MODEL_REGISTRY["patell_wolfson_diagnostic"].justification
    assert "diagnostic features" in patell_text
    assert "pre-event implied-volatility behavior" in patell_text
    assert "prior event variance history" in patell_text
    goyal_text = MODEL_REGISTRY["goyal_saretto_rv_iv_spread"].justification
    assert "Trailing RV-IV spread" in goyal_text
    assert get_model_spec("market_implied_event_variance").implemented is True
    assert get_model_spec("last_four_rvar").implemented is True
    assert get_model_spec("last_four_ivar").implemented is True
    assert get_model_spec("patell_wolfson_diagnostic").implemented is True
    assert get_model_spec("xgboost_tuned").implemented is True
    assert get_model_spec("lightgbm_xgboost_forecast_ensemble").implemented is True
    assert get_model_spec("lightgbm_xgboost_rank_ensemble").implemented is False
    for removed_model_id in [
        "linear_elastic_net",
        "lightgbm",
        "xgboost",
        "bigru_sequence",
        "mamba_ssm_sequence",
        "bigru_sequence_5seed",
        "mamba_ssm_sequence_5seed",
    ]:
        assert removed_model_id not in MODEL_REGISTRY
    assert "implemented as a deterministic baseline" in unimplemented_model_message(
        "market_implied_event_variance"
    ) or "implemented and available" in unimplemented_model_message("market_implied_event_variance")
    for retired_id in [
        "daily_mamba_20step",
        "hybrid_mamba_31step",
        "intraday_only_mamba_12step",
        "mask_only_hybrid_mamba",
        "mamba_sequence_encoder",
    ]:
        assert retired_id not in MODEL_REGISTRY
    for sequence_id in [
        "ridge_flat_aggregates_sequence",
        "attention_pooling_sequence",
        "dilated_cnn_sequence",
        "mask_only_sequence",
        "time_shuffle_sequence",
    ]:
        assert MODEL_REGISTRY[sequence_id].role.startswith("sequence")
    with pytest.raises(KeyError):
        get_model_spec("mamba_sequence_encoder")


def test_patell_wolfson_registry_text() -> None:
    spec = MODEL_REGISTRY["patell_wolfson_diagnostic"]

    assert spec.role == "diagnostic"
    assert spec.implemented is True
    assert "diagnostic features" in spec.justification
    assert "pre-event implied-volatility behavior" in spec.justification
    assert "prior event variance history" in spec.justification


def _synthetic_tuning_frame() -> pd.DataFrame:
    n_rows = 40
    idx = np.arange(n_rows, dtype=float)
    signal = np.sin(idx)
    ivar = 0.02 + 0.001 * (idx % 5)
    rvar = np.maximum(ivar + 0.004 * signal, 0.001)
    return pd.DataFrame(
        {
            "event_id": [f"EV_{idx_int:02d}" for idx_int in range(n_rows)],
            "ticker": ["AAA" if idx_int % 2 == 0 else "BBB" for idx_int in range(n_rows)],
            "announcement_date": pd.date_range("2024-01-01", periods=n_rows, freq="7D"),
            "event_date": pd.date_range("2024-01-02", periods=n_rows, freq="7D"),
            "split": ["train"] * 28 + ["validation"] * 6 + ["test"] * 6,
            "ivar_event": ivar,
            "rvar_event": rvar,
            "edge_var_realized": rvar - ivar,
            "feature_signal": signal,
            "feature_trend": idx / n_rows,
            "feature_cycle": idx % 3,
        }
    )


def test_log_rvar_target_helpers_round_trip_and_floor() -> None:
    raw = np.asarray([0.0, 0.01, 0.2])
    log_values = _target_to_log_rvar(raw)
    restored = _log_rvar_to_variance(log_values)

    assert restored[0] == pytest.approx(FORECAST_FLOOR)
    assert restored[1:] == pytest.approx(raw[1:])
    assert _log_rvar_to_variance([-1000.0])[0] == pytest.approx(FORECAST_FLOOR)


def test_sequence_pairwise_edge_uses_variance_space_after_log_back_transform() -> None:
    log_prediction = torch.log(torch.tensor([FORECAST_FLOOR, 0.02 + FORECAST_FLOOR]))
    log_target = torch.log(torch.tensor([FORECAST_FLOOR, 0.03 + FORECAST_FLOOR]))
    ivar = torch.tensor([0.01, 0.01])
    realized_edge = torch.tensor([-0.01, 0.02])

    predicted_variance = _torch_log_rvar_to_variance(log_prediction)
    _huber, ranking = _sequence_losses(
        log_prediction,
        log_target,
        predicted_variance - ivar,
        realized_edge,
    )
    _bad_huber, bad_ranking = _sequence_losses(
        log_prediction,
        log_target,
        log_prediction - ivar,
        realized_edge,
    )

    assert predicted_variance.tolist() == pytest.approx([FORECAST_FLOOR, 0.02])
    assert float(ranking.item()) != pytest.approx(float(bad_ranking.item()))


def test_ft_transformer_log_mode_uses_unconstrained_head() -> None:
    default_model = FTTransformerRegressor(n_features=3)
    log_model = FTTransformerRegressor(n_features=3, positive_output=False)

    assert default_model.positive_output is True
    assert any(isinstance(module, torch.nn.Softplus) for module in default_model.head.modules())
    assert log_model.positive_output is False
    assert not any(isinstance(module, torch.nn.Softplus) for module in log_model.head.modules())
    with torch.no_grad():
        raw_log_output = log_model(torch.zeros((2, 3), dtype=torch.float32)).numpy()
    assert (_log_rvar_to_variance(raw_log_output) >= FORECAST_FLOOR).all()


def test_research_tuning_cli_and_model_ids() -> None:
    parser = build_parser()
    args = parser.parse_args(
        [
            "research",
            "--tuning-profile",
            "tuned_phase1_day_c2c_rank_log_rvar",
            "--tuning-seed",
            "123",
            "--reuse-tuning-params",
        ]
    )

    assert args.tuning_profile == DEFAULT_TUNING_PROFILE
    assert args.tuning_seed == 123
    assert args.reuse_tuning_params is True
    assert args.feature_schema_version == FEATURE_SCHEMA_V2_SEC_XBRL
    assert not hasattr(args, "mamba_backend")
    assert not hasattr(args, "mamba_seeds")
    for function in (run_proxy_model_suite, run_research_models, run_proxy_research_package):
        defaults = function.__kwdefaults__
        assert defaults is not None
        assert "mamba_backend" not in defaults
        assert "mamba_seeds" not in defaults
    with pytest.raises(SystemExit):
        parser.parse_args(["research", "--tuning-profile", "bad_profile"])

    default_ids = _model_ids_for_sequence_suite("all")
    tuned_ids = _model_ids_for_sequence_suite(
        "all", tuning_profile="tuned_phase1_day_c2c_rank_log_rvar"
    )
    tuned_no_sequence_ids = _model_ids_for_sequence_suite(
        "none",
        tuning_profile="tuned_phase1_day_c2c_rank_log_rvar",
    )
    with pytest.raises(ValueError, match="unsupported sequence_suite"):
        _model_ids_for_sequence_suite("phase1")

    assert "linear_elastic_net" not in default_ids
    assert "lightgbm" not in default_ids
    assert "xgboost" not in default_ids
    assert "ft_transformer" in default_ids
    assert "linear_elastic_net_tuned" in tuned_ids
    assert "lightgbm_tuned" in tuned_ids
    assert "xgboost_tuned" in tuned_ids
    assert "lightgbm_xgboost_forecast_ensemble" in tuned_ids
    assert "lightgbm_xgboost_rank_ensemble" not in tuned_ids
    assert "ft_transformer" in tuned_ids
    assert "bigru_sequence_5seed" not in tuned_ids
    assert "mamba_ssm_sequence_5seed" not in tuned_ids
    assert "attention_pooling_sequence" in tuned_ids
    assert "dilated_cnn_sequence" in tuned_ids
    assert "bigru_sequence_5seed" not in tuned_no_sequence_ids
    assert "mamba_ssm_sequence_5seed" not in tuned_no_sequence_ids
    assert "attention_pooling_sequence" not in tuned_no_sequence_ids
    assert "dilated_cnn_sequence" not in tuned_no_sequence_ids
    assert default_ids == tuned_ids
    assert TUNING_SELECTION_TARGET_ID == "day_c2c"
    assert _target_ids_for_sequence_suite("all")[0] == TUNING_SELECTION_TARGET_ID


def test_data_cli_defaults_target_rebuild_window() -> None:
    parser = build_parser()
    args = parser.parse_args(["data"])

    assert args.start == TARGET_WINDOW_START.isoformat()
    assert args.end == TARGET_WINDOW_END.isoformat()


def test_lightgbm_xgboost_forecast_ensemble_uses_variance_units(tmp_path: Path) -> None:
    frame = _synthetic_tuning_frame().assign(
        forecast_lightgbm_tuned=np.linspace(0.01, 0.04, 40),
        forecast_xgboost_tuned=np.linspace(0.02, 0.08, 40)[::-1],
    )

    predictions, diagnostics = run_proxy_model_suite(
        frame,
        tensor_path=tmp_path / "unused_sequence_tensor.npz",
        model_ids=["lightgbm_xgboost_forecast_ensemble"],
    )

    forecast = predictions["forecast_lightgbm_xgboost_forecast_ensemble"]
    expected = (frame["forecast_lightgbm_tuned"] + frame["forecast_xgboost_tuned"]) / 2.0
    assert np.allclose(forecast, expected)
    assert ENSEMBLE_RANK_SIGNAL_COL in predictions.columns
    edges = pd.DataFrame(
        {
            "lightgbm": frame["forecast_lightgbm_tuned"] - frame["ivar_event"],
            "xgboost": frame["forecast_xgboost_tuned"] - frame["ivar_event"],
        }
    )
    split_groups = frame["split"].astype(str)
    expected_rank_signal = pd.DataFrame(
        {
            "lightgbm": edges.groupby(split_groups)["lightgbm"].rank(method="average", pct=True),
            "xgboost": edges.groupby(split_groups)["xgboost"].rank(method="average", pct=True),
        }
    ).mean(axis=1)
    assert np.allclose(predictions[ENSEMBLE_RANK_SIGNAL_COL], expected_rank_signal)
    assert not np.allclose(
        forecast.rank(method="average", pct=True),
        predictions[ENSEMBLE_RANK_SIGNAL_COL],
    )
    row = diagnostics.loc[diagnostics["model_id"].eq("lightgbm_xgboost_forecast_ensemble")].iloc[0]
    assert row["ensemble_method"] == "equal_weight_forecast_average"
    assert row["prediction_scale"] == "variance_units"
    assert row["ranking_signal_col"] == ENSEMBLE_RANK_SIGNAL_COL


def test_best_completed_trial_respects_penalized_objective() -> None:
    class FakeTrial:
        def __init__(self, value: float, auc: float, top_decile: float, rmse: float):
            self.value = value
            self.user_attrs = {
                "validation_auc": auc,
                "validation_top_decile_precision": top_decile,
                "validation_rmse": rmse,
            }

    class FakeStudy:
        trials = [
            FakeTrial(0.61, auc=0.66, top_decile=0.5, rmse=0.01),
            FakeTrial(0.62, auc=0.62, top_decile=0.3, rmse=0.02),
        ]

    assert _best_completed_trial(FakeStudy()) is FakeStudy.trials[1]


def test_tuned_elastic_net_records_validation_only_artifacts(tmp_path: Path) -> None:
    pytest.importorskip("sklearn")
    frame = _synthetic_tuning_frame()
    tuning_state = TuningState(profile="tuned_phase1_day_c2c_rank_log_rvar", seed=17)

    predictions, diagnostics = run_proxy_model_suite(
        frame,
        tensor_path=tmp_path / "unused_sequence_tensor.npz",
        model_ids=["linear_elastic_net_tuned"],
        tuning_state=tuning_state,
        target_id=TUNING_SELECTION_TARGET_ID,
    )
    outputs = _write_tuning_artifacts(tmp_path, tuning_state=tuning_state)
    selected_payload = json.loads(Path(outputs["tuning_selected_params"]).read_text())
    trials = pd.read_csv(outputs["tuning_trials"])

    assert "forecast_linear_elastic_net" not in predictions
    assert "forecast_linear_elastic_net_tuned" in predictions
    assert (
        predictions.loc[predictions["split"].eq("test"), "forecast_linear_elastic_net_tuned"]
        .notna()
        .all()
    )
    assert (
        diagnostics.loc[diagnostics["model_id"].eq("linear_elastic_net_tuned"), "status"].iloc[0]
        == "trained"
    )
    assert selected_payload["test_metrics_used_for_selection"] is False
    assert selected_payload["selection_target_id"] == TUNING_SELECTION_TARGET_ID
    assert selected_payload["target_transform"] == CANONICAL_TARGET_TRANSFORM
    assert selected_payload["training_space"] == CANONICAL_TRAINING_SPACE
    assert selected_payload["evaluation_space"] == CANONICAL_EVALUATION_SPACE
    assert selected_payload["back_transform"] == CANONICAL_BACK_TRANSFORM
    selected = selected_payload["selected_params"]["linear_elastic_net_tuned"]
    assert selected["selection_target_id"] == TUNING_SELECTION_TARGET_ID
    assert selected["target_transform"] == CANONICAL_TARGET_TRANSFORM
    assert selected["training_space"] == CANONICAL_TRAINING_SPACE
    assert selected["evaluation_space"] == CANONICAL_EVALUATION_SPACE
    assert selected["back_transform"] == CANONICAL_BACK_TRANSFORM
    assert (
        selected["primary_metric"] == f"validation_{TUNING_SELECTION_TARGET_ID}_predicted_edge_auc"
    )
    assert selected["selection_protocol"] == "train_validation_only"
    assert selected["refit_protocol"] == "train_plus_validation"
    assert set(selected["params"]) >= {"alpha", "l1_ratio"}
    assert not any(key.startswith("test") for key in selected["validation_metrics"])
    assert not any(column.startswith("test") for column in trials.columns)
    assert trials["feature_schema_version"].eq(FEATURE_SCHEMA_V2_SEC_XBRL).all()
    assert trials["tuning_profile"].eq("tuned_phase1_day_c2c_rank_log_rvar").all()
    assert trials["target_id"].eq(TUNING_SELECTION_TARGET_ID).all()
    assert trials["target_transform"].eq(CANONICAL_TARGET_TRANSFORM).all()
    assert trials["evaluation_space"].eq(CANONICAL_EVALUATION_SPACE).all()
    assert {"validation_auc", "validation_top_decile_precision", "validation_rmse"}.issubset(
        trials.columns
    )


def test_reusable_tuning_state_requires_matching_schema_profile_and_seed(tmp_path: Path) -> None:
    tuning_state = TuningState(profile="tuned_phase1_day_c2c_rank_log_rvar", seed=17)
    cache_params: dict[str, object] = {"alpha": 0.01, "l1_ratio": 0.1, "max_iter": 100}
    _cache_payload: dict[str, object] = {
        "model_id": "linear_elastic_net_tuned",
        "selection_target_id": TUNING_SELECTION_TARGET_ID,
        "selection_protocol": "train_validation_only",
        "refit_protocol": "train_plus_validation",
        "primary_metric": f"validation_{TUNING_SELECTION_TARGET_ID}_predicted_edge_auc",
        "target_transform": CANONICAL_TARGET_TRANSFORM,
        "training_space": CANONICAL_TRAINING_SPACE,
        "evaluation_space": CANONICAL_EVALUATION_SPACE,
        "back_transform": CANONICAL_BACK_TRANSFORM,
        "forecast_floor": FORECAST_FLOOR,
        "params": cache_params,
        "validation_metrics": {"validation_auc": 0.5},
    }
    tuning_state.selected["linear_elastic_net_tuned"] = _cache_payload
    tuning_state.trials.append(
        {
            "model_id": "linear_elastic_net_tuned",
            "target_id": TUNING_SELECTION_TARGET_ID,
            "trial_number": 0,
            "selected": True,
            "seed": 17,
            "params_json": json.dumps(cache_params, sort_keys=True),
            "validation_n": 10,
            "validation_mae": 0.1,
            "validation_rmse": 0.2,
            "validation_auc": 0.5,
            "validation_top_decile_precision": 0.3,
            "objective_value": 0.5,
        }
    )
    _write_tuning_artifacts(
        tmp_path,
        tuning_state=tuning_state,
        feature_schema_version=FEATURE_SCHEMA_V2_SEC_XBRL,
    )

    loaded, reused, source = _load_reusable_tuning_state(
        tmp_path,
        tuning_profile="tuned_phase1_day_c2c_rank_log_rvar",
        tuning_seed=17,
        feature_schema_version=FEATURE_SCHEMA_V2_SEC_XBRL,
    )

    assert reused is True
    assert source == str(tmp_path / "tuning_selected_params.json")
    assert loaded.selected["linear_elastic_net_tuned"]["params"] == _cache_payload["params"]
    assert loaded.trials[0]["model_id"] == "linear_elastic_net_tuned"

    fallback, reused, reason = _load_reusable_tuning_state(
        tmp_path,
        tuning_profile="tuned_phase1_day_c2c_rank_log_rvar",
        tuning_seed=123,
        feature_schema_version=FEATURE_SCHEMA_V2_SEC_XBRL,
    )

    assert reused is False
    assert reason == "tuning_seed_mismatch"
    assert fallback.selected == {}

    selected_payload = json.loads((tmp_path / "tuning_selected_params.json").read_text())
    selected_payload["target_transform"] = "raw_rvar"
    (tmp_path / "tuning_selected_params.json").write_text(json.dumps(selected_payload))
    fallback, reused, reason = _load_reusable_tuning_state(
        tmp_path,
        tuning_profile="tuned_phase1_day_c2c_rank_log_rvar",
        tuning_seed=17,
        feature_schema_version=FEATURE_SCHEMA_V2_SEC_XBRL,
    )

    assert reused is False
    assert reason == "target_transform_mismatch"
    assert fallback.selected == {}

    selected_payload["target_transform"] = CANONICAL_TARGET_TRANSFORM
    selected_payload["selection_target_id"] = "jump_c2o"
    (tmp_path / "tuning_selected_params.json").write_text(json.dumps(selected_payload))
    fallback, reused, reason = _load_reusable_tuning_state(
        tmp_path,
        tuning_profile="tuned_phase1_day_c2c_rank_log_rvar",
        tuning_seed=17,
        feature_schema_version=FEATURE_SCHEMA_V2_SEC_XBRL,
    )

    assert reused is False
    assert reason == "selection_target_id_mismatch"
    assert fallback.selected == {}


def test_trained_model_with_no_usable_predictions_is_invalid(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    frame = _synthetic_tuning_frame()

    def fake_train_ft_transformer(
        frame: pd.DataFrame,
        *,
        features: Sequence[str],
        target_id: str,
        tuning_state: TuningState,
    ) -> tuple[pd.Series, dict[str, object], object | None]:
        _ = features, target_id, tuning_state
        return (
            pd.Series(np.nan, index=frame.index),
            {"status": "trained", "train_rows": 28, "validation_rows": 6, "test_rows": 6},
            None,
        )

    monkeypatch.setattr(
        "earnings_event_vol.research._train_ft_transformer",
        fake_train_ft_transformer,
    )

    predictions, diagnostics = run_proxy_model_suite(
        frame,
        tensor_path=tmp_path / "unused_sequence_tensor.npz",
        model_ids=["ft_transformer"],
    )

    assert predictions["forecast_ft_transformer"].isna().all()
    row = diagnostics.loc[diagnostics["model_id"].eq("ft_transformer")].iloc[0]
    assert row["status"] == "invalid_no_usable_predictions"
    assert row["raw_status"] == "trained"
    assert row["validation_prediction_finite_rows"] == 0
    assert row["test_prediction_finite_rows"] == 0


def test_removed_sequence_ensembles_are_not_runnable(tmp_path: Path) -> None:
    frame = _synthetic_tuning_frame()
    active_ids = _model_ids_for_sequence_suite("all")

    for removed_model_id in ("bigru_sequence_5seed", "mamba_ssm_sequence_5seed"):
        assert removed_model_id not in active_ids
        assert removed_model_id not in MODEL_REGISTRY
        with pytest.raises(KeyError):
            prediction_column_for_model(removed_model_id)
        with pytest.raises(ValueError, match="unknown model_id"):
            run_proxy_model_suite(
                frame,
                tensor_path=tmp_path / "unused_sequence_tensor.npz",
                model_ids=[removed_model_id],
            )


def test_feature_matrix_benchmarks_models_and_metrics() -> None:
    panel = pd.DataFrame(
        {
            "event_id": [f"ABC_{idx}" for idx in range(6)],
            "ticker": ["ABC", "ABC", "ABC", "XYZ", "XYZ", "XYZ"],
            "announcement_date": pd.date_range("2025-01-01", periods=6, freq="30D"),
            "entry_date": pd.date_range("2025-01-01", periods=6, freq="30D"),
            "announcement_timing": ["AMC", "AMC", "BMO", "AMC", "BMO", "AMC"],
            "rvar_event": [0.05, 0.03, 0.07, 0.02, 0.04, 0.06],
            "ivar_event": [0.04, 0.04, 0.05, 0.03, 0.03, 0.05],
            "dte_1": [8, 9, 16, 20, 12, 7],
            "universe_rank": [1, 2, 3, 4, 5, 6],
            "s_before": [100, 101, 102, 50, 51, 52],
            "paper_grade": [False] * 6,
        }
    )
    straddles = pd.DataFrame(
        {
            "event_id": [f"ABC_{idx}" for idx in range(6)],
            "entry_premium_usd": [400, 420, 430, 250, 260, 270],
            "gross_proxy_pnl_usd": [80, -40, 100, -30, 50, 70],
            "haircut_pnl_usd": [40, -82, 57, -55, 24, 43],
            "proxy_volume_window": [10, 11, 12, 13, 14, 15],
            "proxy_transactions_window": [3, 4, 5, 6, 7, 8],
        }
    )

    features = add_benchmark_predictions(
        build_model_feature_matrix(panel, straddle_diagnostics=straddles)
    )

    assert features["is_main_dte_5_14"].tolist() == [True, True, False, False, True, True]
    assert "forecast_last_four_rvar" in features
    assert "forecast_goyal_saretto_rv_iv_spread" in features
    assert features["mispricing_realized"].iloc[0] == pytest.approx(0.01)

    with pytest.raises(ValueError, match="not a trainable"):
        fit_model("linear_elastic_net", features, split_date="2025-05-01")

    forecast = forecast_metrics(
        features,
        forecast_col="forecast_market_implied_event_variance",
    )
    assert forecast["n"] == 6
    assert forecast["mae"] is not None

    ranking = ranking_metrics(
        features.assign(
            score=features["forecast_goyal_saretto_rv_iv_spread"] - features["ivar_event"]
        ),
        score_col="score",
    )
    assert ranking["top_decile_precision"] is not None
    assert (
        edge_decile_table(
            features.assign(score=features["edge_var_realized"]), score_col="score"
        ).empty
        is False
    )

    strategy = build_proxy_strategy_frame(
        features,
        forecast_col="forecast_goyal_saretto_rv_iv_spread",
    )
    trades = strategy.loc[strategy["should_trade"].astype(bool)]
    metrics = strategy_metrics(trades)
    assert metrics["turnover"] == len(trades)
    assert cost_sensitivity(trades).shape[0] == 5
    if not trades.empty:
        assert trades["trade_direction"].eq("long_straddle").all()
        assert (
            trades["expected_strategy_edge_usd"]
            > trades["threshold_multiplier"] * trades["estimated_transaction_cost_usd"]
        ).all()

    strategy_threshold = build_proxy_strategy_frame(
        pd.DataFrame(
            {
                "forecast": [0.052, 0.020, 0.080],
                "ivar_event": [0.050, 0.050, 0.050],
                "gross_proxy_pnl_usd": [10.0, 20.0, np.nan],
                "entry_premium_usd": [100.0, 100.0, 100.0],
                "estimated_transaction_cost_usd": [5.0, 5.0, 5.0],
            }
        ),
        forecast_col="forecast",
    )
    assert strategy_threshold["should_trade"].tolist() == [False, False, True]
    assert strategy_threshold["trade_direction"].tolist() == [
        "no_trade",
        "no_trade",
        "long_straddle",
    ]
    assert strategy_threshold["expected_strategy_edge_usd"].iloc[1] < 0
    with pytest.raises(ValueError, match="threshold_multiplier must be positive"):
        build_proxy_strategy_frame(
            strategy_threshold,
            forecast_col="forecast",
            threshold_multiplier=0,
        )


def test_validation_only_strategy_policy_filters_quote_quality_and_top_k() -> None:
    validation_edges = [0.020, 0.019, 0.018, 0.017, 0.016, 0.015, 0.014]
    validation_gross = [101.0, 91.0, 81.0, 71.0, 61.0, -199.0, -199.0]
    filtered_edges = [0.030, 0.029, 0.028]
    filtered_gross = [5001.0, 5001.0, 5001.0]
    test_edges = [0.020, 0.019, 0.030, 0.029, 0.028]
    test_gross = [31.0, 21.0, 5001.0, 5001.0, 5001.0]
    frame = pd.DataFrame(
        {
            "split": ["validation"] * 10 + ["test"] * 5,
            "forecast": [
                *(0.05 + np.asarray(validation_edges)),
                *(0.05 + np.asarray(filtered_edges)),
                *(0.05 + np.asarray(test_edges)),
            ],
            "ivar_event": [0.05] * 15,
            "gross_proxy_pnl_usd": [*validation_gross, *filtered_gross, *test_gross],
            "entry_premium_usd": [100.0] * 15,
            "estimated_transaction_cost_usd": [1.0] * 15,
            "liquidity_bucket": ["high"] * 15,
            "is_main_dte_5_14": [True] * 15,
            "execution_confidence_band": [
                *["high"] * 7,
                "low",
                "high",
                "high",
                "high",
                "medium",
                "low",
                "high",
                "high",
            ],
            "execution_confidence_score": [
                *[0.95] * 7,
                0.25,
                0.95,
                0.95,
                0.95,
                0.70,
                0.25,
                0.95,
                0.95,
            ],
            "median_spread_over_mid": [
                *[0.05] * 8,
                0.30,
                0.05,
                0.05,
                0.05,
                0.05,
                0.30,
                0.05,
            ],
            "max_quote_age_seconds": [
                *[10.0] * 9,
                70.0,
                10.0,
                10.0,
                10.0,
                10.0,
                70.0,
            ],
        }
    )

    policy, search = tune_strategy_policy_validation_only(frame, forecast_col="forecast")

    assert not search.empty
    assert int(search["selected"].sum()) == 1
    assert policy.top_k == 5
    assert "low" not in policy.allowed_execution_confidence_bands
    assert policy.max_median_spread_over_mid is not None
    assert policy.max_quote_age_seconds == pytest.approx(60.0)
    assert policy.quote_filter_status.startswith("required_")

    base = build_proxy_strategy_frame(
        frame,
        forecast_col="forecast",
        min_edge_var=policy.min_edge_var,
        threshold_multiplier=policy.threshold_multiplier,
    )
    tuned = apply_strategy_policy(base, policy)
    validation_trades = tuned.loc[
        tuned["split"].eq("validation") & tuned["should_trade"].astype(bool)
    ]
    test_trades = tuned.loc[tuned["split"].eq("test") & tuned["should_trade"].astype(bool)]

    assert len(validation_trades) == 5
    assert validation_trades["net_proxy_pnl_usd"].sum() == pytest.approx(400.0)
    assert len(test_trades) == 2
    assert set(test_trades["execution_confidence_band"]) <= {"high", "medium"}
    assert test_trades["median_spread_over_mid"].le(0.25).all()
    assert test_trades["max_quote_age_seconds"].le(60.0).all()


def test_strategy_policy_top_k_can_use_separate_selection_score() -> None:
    frame = pd.DataFrame(
        {
            "event_id": ["V_forecast", "V_rank", "T_forecast", "T_rank"],
            "split": ["validation", "validation", "test", "test"],
            "forecast": [0.08, 0.07, 0.08, 0.07],
            "ivar_event": [0.05, 0.05, 0.05, 0.05],
            "rank_signal": [0.10, 0.90, 0.20, 0.80],
            "gross_proxy_pnl_usd": [-10.0, 100.0, -10.0, 100.0],
            "entry_premium_usd": [100.0, 100.0, 100.0, 100.0],
            "estimated_transaction_cost_usd": [1.0, 1.0, 1.0, 1.0],
        }
    )

    policy, search = tune_strategy_policy_validation_only(
        frame,
        forecast_col="forecast",
        selection_score_col="rank_signal",
        threshold_multipliers=(0.5,),
        min_edge_vars=(0.0,),
        top_k_values=(1,),
        min_validation_trades=1,
        drawdown_penalty=0.0,
    )
    base = build_proxy_strategy_frame(
        frame,
        forecast_col="forecast",
        min_edge_var=policy.min_edge_var,
        threshold_multiplier=policy.threshold_multiplier,
    )
    tuned = apply_strategy_policy(base, policy)

    assert not search.empty
    assert policy.selection_score_col == "rank_signal"
    assert tuned["strategy_policy_effective_selection_score_col"].eq("rank_signal").all()
    assert tuned.loc[
        tuned["split"].eq("validation") & tuned["should_trade"], "event_id"
    ].tolist() == ["V_rank"]
    assert tuned.loc[tuned["split"].eq("test") & tuned["should_trade"], "event_id"].tolist() == [
        "T_rank"
    ]
    selected_test = tuned.loc[tuned["event_id"].eq("T_rank")].iloc[0]
    assert selected_test["expected_strategy_edge_usd"] == pytest.approx(40.0)


def test_strategy_policy_defensive_branches() -> None:
    frame = pd.DataFrame(
        {
            "split": ["validation", "test"],
            "forecast": [0.08, 0.09],
            "ivar_event": [0.05, 0.05],
            "gross_proxy_pnl_usd": [100.0, -10.0],
            "entry_premium_usd": [100.0, 100.0],
            "estimated_transaction_cost_usd": [1.0, 1.0],
            "expected_strategy_edge_usd": [60.0, 80.0],
            "should_trade": [True, True],
            "gross_strategy_pnl_usd": [100.0, -10.0],
            "net_proxy_pnl_usd": [99.0, -11.0],
            "capital_at_risk_usd": [100.0, 100.0],
        }
    )

    with pytest.raises(ValueError, match="strategy frame missing required columns"):
        build_proxy_strategy_frame(frame.drop(columns=["forecast"]), forecast_col="forecast")
    with pytest.raises(ValueError, match="threshold_multiplier must be positive"):
        apply_strategy_policy(frame, StrategyPolicy(threshold_multiplier=0.0))
    with pytest.raises(ValueError, match="top_k must be positive"):
        apply_strategy_policy(frame, StrategyPolicy(top_k=0))

    missing_field_policies = [
        StrategyPolicy(allowed_liquidity_buckets=("high",)),
        StrategyPolicy(require_main_dte_5_14=True),
        StrategyPolicy(allowed_dte_buckets=("main_5_14",)),
        StrategyPolicy(allowed_execution_confidence_bands=("high",)),
        StrategyPolicy(min_execution_confidence_score=0.8),
        StrategyPolicy(max_median_spread_over_mid=0.25),
        StrategyPolicy(max_quote_age_seconds=60.0),
    ]
    for policy in missing_field_policies:
        filtered = apply_strategy_policy(frame, policy)
        assert not filtered["should_trade"].any()

    policy, search = tune_strategy_policy_validation_only(
        frame.loc[frame["split"].eq("test")],
        forecast_col="forecast",
    )
    assert policy.quote_filter_status == "no_validation_rows"
    assert bool(search["selected"].iloc[0])

    with pytest.raises(ValueError, match="min_validation_trades must be nonnegative"):
        tune_strategy_policy_validation_only(
            frame,
            forecast_col="forecast",
            min_validation_trades=-1,
        )
    with pytest.raises(ValueError, match="drawdown_penalty must be nonnegative"):
        tune_strategy_policy_validation_only(frame, forecast_col="forecast", drawdown_penalty=-0.1)


def test_feature_schema_allowlist_blocks_raw_ids_and_outcomes() -> None:
    frame = pd.DataFrame(
        {
            "cik": [320193],
            "company_cik": [320193],
            "event_year": [2025],
            "event_month": [2],
            "event_month_sin": [0.5],
            "ivar_event": [0.04],
            "entry_premium_usd": [400.0],
            "exit_option_value_usd": [300.0],
            "exit_intrinsic_usd": [100.0],
            "gross_proxy_pnl_usd": [10.0],
            "prior_day_c2c_rvar_median": [0.03],
            "xbrl_log_assets": [10.0],
            "xbrl_dropped_same_day_filed_rows": [1],
            "feature_schema_version": [FEATURE_SCHEMA_V2_SEC_XBRL],
        }
    )

    schema = build_feature_schema_report(frame, feature_schema_version=FEATURE_SCHEMA_V2_SEC_XBRL)
    selected = set(feature_columns_from_schema_report(schema, frame=frame))

    assert "cik" not in selected
    assert "company_cik" not in selected
    assert "event_year" not in selected
    assert "event_month" not in selected
    assert "exit_option_value_usd" not in selected
    assert "exit_intrinsic_usd" not in selected
    assert "gross_proxy_pnl_usd" not in selected
    assert "xbrl_dropped_same_day_filed_rows" not in selected
    assert {
        "event_month_sin",
        "ivar_event",
        "entry_premium_usd",
        "prior_day_c2c_rvar_median",
        "xbrl_log_assets",
    } <= selected


def test_feature_schema_versions_and_selector_branches() -> None:
    with pytest.raises(ValueError, match="unsupported feature_schema_version"):
        validate_feature_schema_version("bad")
    frame = pd.DataFrame(
        {
            "ivar_event": [0.04],
            "prior_day_c2c_rvar_median": [0.03],
            "prior_text": ["old"],
            "xbrl_log_assets": [10.0],
            "runup_5d_atm_iv_proxy_mean_proxy": [0.2],
            "delta_grid_proxy_curvature": [0.01],
            "rnd_proxy_tail_asymmetry": [-0.5],
            "vix_level": [18.0],
            "entry_cost_to_premium": [0.02],
            "seqagg_surface_missing_rate_mean": [0.1],
            "seq_t00_atm_iv_proxy": [0.2],
            "company_cik": [320193],
            "forecast_xgboost": [0.05],
            "post_event_outcome": [1.0],
            "event_month_sin": [0.5],
            "ivar_event_train_z": [0.0],
            "ticker_text": ["AAPL"],
        }
    )

    legacy = build_feature_schema_report(frame, feature_schema_version=FEATURE_SCHEMA_V1_LEGACY)
    legacy_selected = set(feature_columns_from_schema_report(legacy, frame=frame))
    assert "ivar_event" in legacy_selected
    assert "prior_day_c2c_rvar_median" not in legacy_selected
    assert "xbrl_log_assets" not in legacy_selected
    assert "event_month_sin" not in legacy_selected
    assert "ivar_event_train_z" not in legacy_selected
    assert "ticker_text" not in legacy_selected

    v2 = build_feature_schema_report(frame, feature_schema_version=FEATURE_SCHEMA_V2_SEC_XBRL)
    v2_selected = set(
        feature_columns_from_schema_report(v2, frame=frame, include_sequence_aggregates=False)
    )
    assert {
        "prior_day_c2c_rvar_median",
        "xbrl_log_assets",
        "runup_5d_atm_iv_proxy_mean_proxy",
        "delta_grid_proxy_curvature",
        "rnd_proxy_tail_asymmetry",
        "vix_level",
        "entry_cost_to_premium",
    } <= v2_selected
    assert "seqagg_surface_missing_rate_mean" not in v2_selected
    assert "seq_t00_atm_iv_proxy" not in v2_selected
    assert "company_cik" not in v2_selected
    assert "forecast_xgboost" not in v2_selected
    assert "post_event_outcome" not in v2_selected
    assert "prior_text" not in v2_selected
    with pytest.raises(ValueError, match="feature schema report missing columns"):
        feature_columns_from_schema_report(pd.DataFrame({"feature_name": ["x"]}))


def test_rolling_history_no_peeking_on_same_ticker_timestamps() -> None:
    frame = pd.DataFrame(
        {
            "event_id": ["OLD", "TIE", "NEW"],
            "ticker": ["AAA", "AAA", "AAA"],
            "event_entry_timestamp": [
                "2025-01-01T21:00:00Z",
                "2025-02-01T21:00:00Z",
                "2025-02-01T21:00:00Z",
            ],
            "event_date": ["2025-01-01", "2025-02-01", "2025-02-01"],
            "rvar_event": [0.01, 0.50, 0.09],
            "RVAR_event_day_c2c": [0.01, 0.50, 0.09],
            "RVAR_event_jump_c2o": [0.02, 0.60, 0.08],
            "ivar_event": [0.02, 0.03, 0.04],
        }
    )

    out = add_rolling_earnings_history(frame)

    assert out.loc[out["event_id"].eq("OLD"), "prior_earnings_count"].iloc[0] == 0
    assert out.loc[out["event_id"].eq("TIE"), "prior_earnings_count"].iloc[0] == 1
    new = out.loc[out["event_id"].eq("NEW")].iloc[0]
    assert new["prior_earnings_count"] == 1
    assert new["prior_day_c2c_rvar_median"] == pytest.approx(0.01)
    assert new["prior_day_c2c_rvar_median"] != pytest.approx(0.50)


def test_rolling_history_fallbacks_without_ticker_or_entry_timestamp() -> None:
    no_ticker = pd.DataFrame({"event_date": ["2025-01-01"], "rvar_event": [0.01]})
    assert add_rolling_earnings_history(no_ticker).equals(no_ticker)
    frame = pd.DataFrame(
        {
            "ticker": ["AAA", "AAA"],
            "event_date": ["2025-01-01", "2025-02-01"],
            "rvar_event": [0.02, 0.04],
        }
    )

    out = add_rolling_earnings_history(frame)

    assert out.loc[0, "prior_earnings_count"] == 0
    assert out.loc[1, "prior_earnings_count"] == 1
    assert out.loc[1, "prior_day_c2c_rvar_median"] == pytest.approx(0.02)
    assert "prior_day_c2c_rv_iv_spread_median" in out.columns


def test_train_fit_normalizer_does_not_use_test_distribution() -> None:
    frame = pd.DataFrame(
        {
            "split": ["train", "train", "validation", "test"],
            "ivar_event": [1.0, 3.0, 100.0, 1000.0],
        }
    )

    out, params = add_train_fit_normalized_features(
        frame,
        columns=["ivar_event"],
        feature_schema_version=FEATURE_SCHEMA_V2_SEC_XBRL,
    )

    assert params["test_distribution_used"] is False
    normalizer_columns = cast(dict[str, Any], params["columns"])
    ivar_params = cast(dict[str, Any], normalizer_columns["ivar_event"])
    ivar_rank_bins = cast(dict[str, float], ivar_params["rank_bins"])
    assert ivar_params["center"] == pytest.approx(2.0)
    assert ivar_params["scale"] == pytest.approx(1.0)
    assert ivar_rank_bins["0.5"] == pytest.approx(2.0)
    assert out.loc[3, "ivar_event_train_z"] == pytest.approx(998.0)


def test_normalizer_params_only_train_validation_and_skips() -> None:
    frame = pd.DataFrame(
        {
            "split": ["train", "validation", "test"],
            "constant_feature": [2.0, 2.0, 99.0],
            "empty_feature": [np.nan, np.nan, 1.0],
            "text_feature": ["a", "b", "c"],
        }
    )

    params = normalization_params_only(
        frame,
        columns=["constant_feature", "empty_feature", "text_feature", "missing_feature"],
        feature_schema_version=FEATURE_SCHEMA_V2_SEC_XBRL,
        fit_split="train_validation",
    )

    columns = cast(dict[str, Any], params["columns"])
    constant = cast(dict[str, Any], columns["constant_feature"])
    assert params["fit_split"] == "train_validation"
    assert constant["n_fit"] == 2
    assert constant["scale"] == pytest.approx(1.0)
    assert "z_feature" not in constant
    assert "rank_feature" not in constant
    assert "empty_feature" not in columns
    assert "text_feature" not in columns


def test_sec_xbrl_asof_gate_prefers_acceptance_timestamp_and_drops_same_day_filed() -> None:
    payload = {
        "filings": {
            "recent": {
                "accessionNumber": ["A", "B"],
                "acceptanceDateTime": ["2025-01-02T20:00:00.000Z", "2025-01-03T22:00:00.000Z"],
            }
        }
    }
    acceptance = data_pipeline._submission_acceptance_lookup(payload)
    facts = pd.DataFrame(
        {
            "ticker": ["AAA", "AAA", "AAA"],
            "feature_concept": ["assets", "assets", "revenue"],
            "val": [100.0, 999.0, 50.0],
            "filed": ["2025-01-02", "2025-01-03", "2025-01-03"],
            "end": ["2024-12-31", "2024-12-31", "2024-12-31"],
            "accn": ["A", "B", "MISSING"],
            "acceptance_datetime": [acceptance["A"], acceptance["B"], None],
        }
    )
    event = pd.Series(
        {
            "ticker": "AAA",
            "feature_asof_timestamp": pd.Timestamp("2025-01-03T21:00:00Z"),
            "event_entry_timestamp": pd.Timestamp("2025-01-03T21:00:00Z"),
        }
    )

    features = _latest_xbrl_values_for_event(facts, event)

    assert features["xbrl_available"] is True
    assert features["xbrl_log_assets"] == pytest.approx(np.log1p(100.0))
    assert np.isnan(float(cast(float, features["xbrl_log_revenue"])))
    assert features["xbrl_mapped_acceptance_rows"] == 1


def test_event_target_decomposition_amc_bmo_and_open_audit() -> None:
    windows = pd.DataFrame(
        {
            "event_id": ["AMC", "BMO"],
            "ticker": ["AAA", "BBB"],
            "announcement_timing": ["AMC", "BMO"],
            "entry_date": [date(2026, 2, 5), date(2026, 2, 4)],
            "exit_date": [date(2026, 2, 6), date(2026, 2, 5)],
        }
    )
    bars = pd.DataFrame(
        {
            "ticker": ["AAA", "AAA", "BBB", "BBB"],
            "date": [date(2026, 2, 5), date(2026, 2, 6), date(2026, 2, 4), date(2026, 2, 5)],
            "open": [99.0, 108.0, 49.0, 55.0],
            "close": [100.0, 110.0, 50.0, 54.0],
            "volume": [10, 11, 12, 13],
        }
    )
    out = add_event_return_targets(windows, bars)
    assert out.loc[0, "close_before"] == pytest.approx(100.0)
    assert out.loc[0, "open_after"] == pytest.approx(108.0)
    assert out.loc[0, "close_after"] == pytest.approx(110.0)
    assert out.loc[1, "close_before"] == pytest.approx(50.0)
    assert out.loc[1, "open_after"] == pytest.approx(55.0)
    np.testing.assert_allclose(out["return_decomposition_residual"], 0.0, atol=1e-7)
    np.testing.assert_allclose(out["RVAR_day_reconstructed"], out["RVAR_event_day_c2c"])
    assert out["rvar_event"].equals(out["RVAR_event_day_c2c"])
    assert set(out["open_after_status"]) == {"vendor_regular_ohlc_assumed"}

    missing = add_event_return_targets(windows.iloc[[0]], bars.iloc[[0]])
    assert missing["open_after_status"].iloc[0] == "unavailable"
    assert pd.isna(missing["RVAR_event_jump_c2o"].iloc[0])

    missing_dates = add_event_return_targets(
        pd.DataFrame(
            {
                "event_id": ["NO_DATES"],
                "ticker": ["AAA"],
                "announcement_timing": ["AMC"],
                "entry_date": [None],
                "exit_date": [None],
            }
        ),
        pd.DataFrame(),
    )
    assert missing_dates["open_after_status"].iloc[0] == "unavailable"
    assert pd.isna(missing_dates["rvar_event"].iloc[0])

    excluded = add_event_return_targets(
        windows.iloc[[0]].assign(exclusion_reason=["non_bmo_amc"]),
        bars,
    )
    assert excluded["open_after_status"].iloc[0] == "unavailable"
    assert pd.isna(excluded["rvar_event"].iloc[0])

    features = build_model_feature_matrix(
        out.assign(
            ivar_event=[0.01, 0.02],
            dte_1=[7, 8],
            universe_rank=[1, 2],
        )
    )
    assert features["rvar_event"].equals(features["RVAR_event_day_c2c"])
    assert "edge_var_realized_jump_c2o" in features
    assert "edge_var_realized_day_c2c" in features
    assert "edge_var_realized_reaction_o2c" in features

    targets = available_target_columns(out)
    assert targets == {
        "jump_c2o": "RVAR_event_jump_c2o",
        "day_c2c": "RVAR_event_day_c2c",
        "reaction_o2c": "RVAR_event_reaction_o2c",
    }
    assert target_label_column("day_c2c", out) == "RVAR_event_day_c2c"
    assert target_label_column("day_c2c", pd.DataFrame({"rvar_event": [0.1]})) == "rvar_event"
    with pytest.raises(ValueError, match="unsupported target_id"):
        target_label_column("bad_target", out)


def test_sequence_matrix_and_torch_sequence_models() -> None:
    rows = pd.DataFrame(
        {
            "event_id": ["E1", "E1", "E2", "E2"],
            "day_index": [-2, -1, -2, -1],
            "atm_iv": [0.30, 0.32, 0.40, 0.42],
            "option_volume": [100, 120, 200, 220],
            "spread_over_mid": [0.10, 0.09, 0.08, 0.07],
        }
    )
    sequence = build_option_surface_sequence_matrix(rows, lookback_days=2)
    columns = [column for column in sequence.columns if column.startswith("seq_t")]
    tensor = sequence_tensor_from_frame(sequence, columns)

    assert tensor.shape == (2, 2, 3)
    mask = torch.tensor([[True, False], [True, True]])
    ft = FTTransformerRegressor(n_features=3)
    ft_dropout = FTTransformerRegressor(n_features=3, dropout=0.2)
    attention = AttentionPoolingSequenceEncoder(n_features=3, hidden_size=4, n_heads=2)
    cnn = DilatedCNNSequenceEncoder(
        n_features=3,
        channels=(4, 4),
        dilations=(1, 2),
        dropout=0.0,
    )
    assert ft(tensor[:, 0, :]).shape == (2,)
    assert ft_dropout(tensor[:, 0, :]).shape == (2,)
    assert attention(tensor, mask).shape == (2,)
    assert attention.last_attention_weights is not None
    assert attention.last_attention_weights.shape == (2, 1, 2)
    assert cnn(tensor, mask).shape == (2,)
    with pytest.raises(ValueError, match="hidden_size must be divisible"):
        AttentionPoolingSequenceEncoder(n_features=3, hidden_size=5, n_heads=2)
    with pytest.raises(ValueError, match="channels and dilations"):
        DilatedCNNSequenceEncoder(n_features=3, channels=(4,), dilations=(1, 2))


def test_research_metrics_cover_empty_and_breakdown_paths() -> None:
    frame = pd.DataFrame(
        {
            "event_date": pd.date_range("2025-01-01", periods=5, freq="D"),
            "ticker": ["AAA", "AAA", "BBB", "BBB", "CCC"],
            "forecast": [0.06, 0.02, 0.07, 0.01, 0.05],
            "rvar_event": [0.05, 0.03, 0.08, 0.02, 0.04],
            "ivar_event": [0.04, 0.04, 0.05, 0.03, 0.05],
            "edge_var_realized": [0.01, -0.01, 0.03, -0.01, -0.01],
            "net_proxy_pnl_usd": [100, -50, 200, -20, 0],
            "gross_proxy_pnl_usd": [120, -30, 230, -10, 5],
            "entry_premium_usd": [500, 400, 600, 300, 450],
            "estimated_transaction_cost_usd": [20, 20, 30, 10, 5],
            "announcement_timing": ["AMC", "BMO", "AMC", "BMO", "AMC"],
        }
    )

    assert qlike_loss([0.05, 0.10], [0.04, 0.11]) >= 0
    assert (
        forecast_metrics(
            pd.DataFrame({"forecast": [np.nan], "rvar_event": [np.nan]}),
            forecast_col="forecast",
        )["n"]
        == 0
    )
    assert (
        forecast_metrics(frame.drop(columns=["ivar_event"]), forecast_col="forecast")[
            "oos_r2_vs_ivar"
        ]
        is None
    )
    mixed_sample = forecast_metrics(
        pd.DataFrame(
            {
                "forecast": [0.05, 0.06],
                "rvar_event": [0.04, 0.07],
                "ivar_event": [0.04, np.nan],
            }
        ),
        forecast_col="forecast",
    )
    assert mixed_sample["n_forecast_target"] == 2
    assert mixed_sample["n"] == 1
    assert mixed_sample["n_oos_r2"] == 1
    assert auc_score([1, 1], [0.2, 0.3]) is None
    assert brier_score([np.nan], [np.nan]) is None
    assert max_drawdown([]) == 0.0

    calibration = calibration_table(
        frame.assign(outcome=frame["edge_var_realized"].gt(0).astype(int)),
        score_col="forecast",
        outcome_col="outcome",
        bins=3,
    )
    assert calibration["n"].sum() == len(frame)
    assert calibration_table(
        pd.DataFrame({"forecast": [np.nan], "outcome": [np.nan]}),
        score_col="forecast",
        outcome_col="outcome",
    ).empty
    with pytest.raises(ValueError, match="frame must include"):
        calibration_table(frame, score_col="missing", outcome_col="outcome")

    assert (
        ranking_metrics(
            pd.DataFrame({"score": [np.nan], "edge_var_realized": [np.nan]}),
            score_col="score",
        )["n"]
        == 0
    )
    assert edge_decile_table(
        pd.DataFrame({"score": [np.nan], "edge_var_realized": [np.nan]}), score_col="score"
    ).empty
    tied = pd.DataFrame({"score": [1.0, 1.0, 1.0], "edge_var_realized": [1.0, -1.0, 1.0]})
    assert ranking_metrics(tied, score_col="score")["top_decile_precision"] == pytest.approx(2 / 3)
    assert edge_decile_table(tied, score_col="score")["n"].tolist() == [3]
    assert strategy_metrics(pd.DataFrame({"net_proxy_pnl_usd": [np.nan]}))["turnover"] == 0
    mixed_trades = pd.DataFrame(
        {
            "net_proxy_pnl_usd": [90.0, np.nan],
            "gross_strategy_pnl_usd": [100.0, 200.0],
            "entry_premium_usd": [1000.0, 5000.0],
            "estimated_transaction_cost_usd": [10.0, 20.0],
        }
    )
    mixed_metrics = strategy_metrics(mixed_trades)
    assert mixed_metrics["n"] == 1
    assert mixed_metrics["gross_pnl_usd"] == pytest.approx(100.0)
    assert mixed_metrics["return_on_premium"] == pytest.approx(0.09)
    sensitivity = cost_sensitivity(mixed_trades)
    assert sensitivity["n"].tolist() == [1, 1, 1, 1, 1]
    with pytest.raises(ValueError, match="frame must include"):
        strategy_metrics(frame.drop(columns=["net_proxy_pnl_usd"]))
    with pytest.raises(ValueError, match="frame must include"):
        cost_sensitivity(frame.drop(columns=["gross_proxy_pnl_usd"]))
    with pytest.raises(ValueError, match="breakdown columns"):
        breakdown_metrics(frame, by=["missing"], forecast_col="forecast")

    by_timing = breakdown_metrics(frame, by=["announcement_timing"], forecast_col="forecast")
    assert set(by_timing["announcement_timing"]) == {"AMC", "BMO"}
    bundle = evaluate_prediction_bundle(
        frame,
        forecast_col="forecast",
        score_col="forecast",
        breakdown_columns=["announcement_timing"],
    )
    assert bundle.forecast["n"] == len(frame)
    assert bundle.ranking["n"] == len(frame)
    assert bundle.strategy["turnover"] == len(frame)
    assert "announcement_timing" in bundle.breakdowns


def test_quote_diagnostic_tables_write_confidence_artifacts(tmp_path: Path) -> None:
    predictions = pd.DataFrame(
        {
            "event_id": ["E1", "E1", "E2", "E3"],
            "target_id": ["jump_c2o", "day_c2c", "jump_c2o", "jump_c2o"],
            "split": ["test", "test", "test", "train"],
            "ticker": ["AAA", "AAA", "BBB", "CCC"],
            "execution_confidence_band": ["high", "high", "medium", "missing"],
            "execution_confidence_score": [0.95, 0.95, 0.72, np.nan],
            "max_quote_age_seconds": [12.0, 12.0, 45.0, np.nan],
            "median_spread_over_mid": [0.04, 0.04, 0.18, np.nan],
            "quote_mid_ivar_event": [0.012, 0.012, 0.015, np.nan],
            "quote_ask_ivar_event": [0.013, 0.013, np.nan, np.nan],
            "paper_grade_quote_ivar_mid": [True, True, True, False],
            "paper_grade_quote_ivar_ask": [True, True, False, False],
        }
    )
    strategy_breakdowns = pd.DataFrame(
        {
            "target_id": ["day_c2c"],
            "model_id": ["goyal_saretto_rv_iv_spread"],
            "strategy_proxy_kind": ["day_c2c_exit_preclose_15m_proxy"],
            "breakdown": ["execution_confidence_band"],
            "execution_confidence_band": ["high"],
            "strategy_n": [1],
            "strategy_net_pnl_usd": [100.0],
        }
    )
    ivar_defeat_breakdowns = pd.DataFrame(
        {
            "target_id": ["jump_c2o"],
            "model_id": ["goyal_saretto_rv_iv_spread"],
            "breakdown": ["execution_confidence_band"],
            "breakdown_value": ["high"],
            "n": [1],
            "model_beats_ivar_abs_rate": [1.0],
        }
    )
    casebook_events = pd.DataFrame(
        {
            "execution_confidence_band": ["high", "medium"],
            "case_type": ["false_positive", "market_right_model_wrong"],
            "target_id": ["jump_c2o", "jump_c2o"],
            "model_id": ["xgboost_tuned", "xgboost_tuned"],
            "severity_score": [0.02, 0.01],
            "model_abs_error": [0.03, 0.02],
            "ivar_abs_error": [0.01, 0.01],
        }
    )

    paths = build_quote_diagnostic_tables(
        predictions,
        strategy_breakdowns=strategy_breakdowns,
        ivar_defeat_breakdowns=ivar_defeat_breakdowns,
        casebook_events=casebook_events,
        out_dir=tmp_path,
    )

    assert set(paths) == {
        "quote_confidence_prediction_coverage",
        "quote_ivar_summary",
        "quote_confidence_strategy_summary",
        "quote_confidence_ivar_defeat_summary",
        "quote_confidence_casebook_summary",
    }
    coverage = pd.read_csv(paths["quote_confidence_prediction_coverage"])
    high_jump = coverage.loc[
        coverage["execution_confidence_band"].eq("high") & coverage["target_id"].eq("jump_c2o")
    ].iloc[0]
    assert high_jump["n_target_rows"] == 1
    assert high_jump["n_events"] == 1
    quote_ivar = pd.read_csv(paths["quote_ivar_summary"])
    high = quote_ivar.loc[quote_ivar["execution_confidence_band"].eq("high")].iloc[0]
    assert high["n_events"] == 1
    assert high["mid_ivar_available_events"] == 1
    assert high["ask_ivar_available_events"] == 1
    casebook = pd.read_csv(paths["quote_confidence_casebook_summary"])
    assert set(casebook["case_type"]) == {"false_positive", "market_right_model_wrong"}


def test_completion_gap_audit_separates_quote_progress_from_paper_grade_gaps(
    tmp_path: Path,
) -> None:
    config = replace(
        load_project_config(),
        repo_root=tmp_path,
        data_dir=tmp_path / "data",
        bronze_data_dir=tmp_path / "data" / "bronze",
        silver_data_dir=tmp_path / "data" / "silver",
        gold_data_dir=tmp_path / "data" / "gold",
        artifacts_dir=tmp_path / "artifacts",
        reports_dir=tmp_path / "reports",
    )
    paths = research_paths(config)
    quote_dir = config.artifacts_dir / "data_pipeline" / "quote_execution_panel"
    lake_dir = config.artifacts_dir / "data_pipeline" / "lake_quality_audit"
    quote_dir.mkdir(parents=True)
    lake_dir.mkdir(parents=True)
    paths.modeling_artifacts_dir.mkdir(parents=True)

    (quote_dir / "quote_execution_panel_manifest.json").write_text(
        json.dumps(
            {
                "report": {
                    "ok": True,
                    "metadata_only": False,
                    "raw_full_day_files_written": False,
                    "quote_rows_matched": 42,
                    "route": "massive_quotes_v3_rest_targeted",
                },
                "lake_output_rows": {
                    "bronze_quote_window_quotes": 42,
                    "silver_quote_window_marks": 8,
                    "silver_quote_execution_legs": 8,
                    "gold_quote_straddle_execution": 4,
                    "gold_quote_ivar_event": 2,
                    "gold_quote_iv_surface": 4,
                    "gold_quote_iv_surface_summary": 2,
                    "gold_quote_surface_ivar_event": 1,
                    "gold_quote_execution_confidence": 2,
                },
                "lake_policy": {
                    "quote_source": "rest",
                    "raw_full_day_quote_files_in_repo": False,
                },
                "raw_full_day_files_written": False,
            }
        ),
        encoding="utf-8",
    )
    pd.DataFrame({"quote_mid_iv": [0.35]}).to_csv(quote_dir / "quote_iv_surface.csv", index=False)
    pd.DataFrame({"quote_mid_total_variance": [0.01]}).to_csv(
        quote_dir / "quote_iv_surface_summary.csv", index=False
    )
    pd.DataFrame({"quote_surface_mid_ivar_event": [0.008]}).to_csv(
        quote_dir / "quote_surface_ivar_event.csv", index=False
    )
    (lake_dir / "lake_quality_report.json").write_text(
        json.dumps(
            {
                "status": "ran",
                "ok": False,
                "target_window": {
                    "start": TARGET_WINDOW_START.isoformat(),
                    "end": TARGET_WINDOW_END.isoformat(),
                },
                "incomplete_required_datasets": 2,
                "incomplete_required_dataset_ids": [
                    "bronze_options_day_aggs",
                    "gold_quote_ivar_event",
                ],
                "paper_grade_execution_ready": False,
                "paper_grade_execution_blocker": "requires full-window quote/NBBO",
            }
        ),
        encoding="utf-8",
    )
    for name in [
        "quote_confidence_prediction_coverage.csv",
        "quote_ivar_summary.csv",
        "quote_confidence_strategy_summary.csv",
        "quote_confidence_ivar_defeat_summary.csv",
        "quote_confidence_casebook_summary.csv",
        "robustness_summary.csv",
    ]:
        pd.DataFrame({"x": [1]}).to_csv(paths.modeling_artifacts_dir / name, index=False)
    sequence_rows = [
        {
            "model_id": model_id,
            "target_id": target_id,
            "coverage": 10,
            "headline_eligible": False,
        }
        for model_id in [
            "ridge_flat_aggregates_sequence",
            "attention_pooling_sequence",
            "dilated_cnn_sequence",
            "mask_only_sequence",
            "time_shuffle_sequence",
        ]
        for target_id in TARGET_IDS
    ]
    pd.DataFrame(sequence_rows).to_csv(
        paths.modeling_artifacts_dir / "sequence_model_fit_diagnostics.csv",
        index=False,
    )

    outputs = build_completion_gap_audit(paths)

    audit = pd.read_csv(outputs["completion_gap_audit"])
    statuses = dict(zip(audit["requirement_id"], audit["status"], strict=True))
    assert statuses["bounded_quote_extraction_matched_rows"] == "complete"
    assert statuses["quote_marks_legs_straddles_confidence_populated"] == "complete"
    assert statuses["quote_confidence_stratified_results"] == "complete"
    assert statuses["bounded_quote_iv_surface_diagnostics_populated"] == "complete"
    assert statuses["sequence_diagnostics_full_suite_populated"] == "complete"
    assert statuses["full_historical_lake_quality_audit"] == "complete"
    assert statuses["quote_ivar_populated_but_not_surface"] == "diagnostic_only"
    assert statuses["sequence_headline_gate"] == "diagnostic_only"
    assert statuses["target_window_data_coverage"] == "incomplete"
    assert statuses["paper_grade_bid_ask_nbbo_execution"] == "incomplete"
    summary = json.loads(Path(outputs["completion_gap_audit_summary"]).read_text())
    assert summary["paper_grade_ready"] is False
    assert "paper_grade_bid_ask_nbbo_execution" in summary["blocking_requirement_ids"]


def test_robustness_summary_table_covers_dte_liquidity_and_vix_regime(tmp_path: Path) -> None:
    strategy_breakdowns = pd.DataFrame(
        {
            "target_id": ["day_c2c"] * 6,
            "model_id": ["goyal_saretto_rv_iv_spread"] * 6,
            "strategy_proxy_kind": ["day_c2c_exit_preclose_15m_proxy"] * 6,
            "pnl_headline_eligible": [True] * 6,
            "breakdown": [
                "dte_bucket",
                "dte_bucket",
                "liquidity_bucket",
                "liquidity_bucket",
                "vix_regime_tercile",
                "vix_regime_tercile",
            ],
            "dte_bucket": ["main_5_14", "lt_5", np.nan, np.nan, np.nan, np.nan],
            "liquidity_bucket": [np.nan, np.nan, "high", "low", np.nan, np.nan],
            "vix_regime_tercile": [np.nan, np.nan, np.nan, np.nan, "low", "high"],
            "strategy_n": [8, 4, 12, 3, 9, 7],
            "strategy_net_pnl_usd": [100.0, -50.0, 250.0, -20.0, 10.0, -5.0],
        }
    )
    ivar_defeat_breakdowns = pd.DataFrame(
        {
            "target_id": ["jump_c2o"] * 6,
            "model_id": ["lightgbm_tuned"] * 6,
            "breakdown": [
                "dte_bucket",
                "dte_bucket",
                "liquidity_bucket",
                "liquidity_bucket",
                "vix_regime_tercile",
                "vix_regime_tercile",
            ],
            "breakdown_value": ["main_5_14", "lt_5", "high", "low", "low", "high"],
            "n": [20, 6, 18, 5, 10, 9],
            "mae_lift_vs_ivar": [0.02, -0.01, 0.03, -0.02, 0.01, 0.00],
        }
    )

    paths = build_robustness_summary_table(
        strategy_breakdowns,
        ivar_defeat_breakdowns,
        out_dir=tmp_path,
        feature_schema_version=FEATURE_SCHEMA_V2_SEC_XBRL,
    )

    summary = pd.read_csv(paths["robustness_summary"])
    assert {"dte_bucket", "liquidity_bucket", "vix_regime_tercile"}.issubset(
        set(summary["breakdown"])
    )
    assert set(summary["source"]) == {"strategy", "ivar_defeat"}
    vix_strategy = summary.loc[
        summary["source"].eq("strategy") & summary["breakdown"].eq("vix_regime_tercile")
    ].iloc[0]
    assert vix_strategy["subgroup_count"] == 2
    assert vix_strategy["primary_metric"] == "strategy_net_pnl_usd"
    assert vix_strategy["claim_gate_status"] == "available_multi_bucket"
    dte_strategy = summary.loc[
        summary["source"].eq("strategy") & summary["breakdown"].eq("dte_bucket")
    ].iloc[0]
    assert dte_strategy["claim_gate_status"] == "exploratory_small_cells"
    dte_ivar = summary.loc[
        summary["source"].eq("ivar_defeat") & summary["breakdown"].eq("dte_bucket")
    ].iloc[0]
    assert dte_ivar["primary_metric"] == "mae_lift_vs_ivar"
    assert dte_ivar["positive_metric_rows"] == 1


def test_report_uses_sequence_gate_for_lightweight_sequence_suite(tmp_path: Path) -> None:
    artifacts = tmp_path / "artifacts"
    reports = tmp_path / "reports"
    artifacts.mkdir()
    (artifacts / "sequence_coverage_report.json").write_text(
        json.dumps(
            {
                "eligible_events": 20,
                "total_events": 25,
                "drop_rate": 0.2,
                "high_sequence_selection_risk": True,
            }
        ),
        encoding="utf-8",
    )
    pd.DataFrame(
        {
            "model_id": ["attention_pooling_sequence"],
            "target_id": ["jump_c2o"],
            "status": ["trained"],
            "feature_count": [243],
        }
    ).to_csv(artifacts / "model_fit_diagnostics.csv", index=False)
    pd.DataFrame(
        {
            "feature_schema_version": [FEATURE_SCHEMA_V2_SEC_XBRL] * 4,
            "tuning_profile": ["tuned_phase1_day_c2c_rank_log_rvar"] * 4,
            "target_id": ["jump_c2o"] * 4,
            "model_id": [
                "ridge_flat_aggregates_sequence",
                "attention_pooling_sequence",
                "dilated_cnn_sequence",
                "mask_only_sequence",
            ],
            "coverage": [12, 10, 10, 10],
            "drop_rate": [0.1, 0.2, 0.2, 0.2],
            "auc_lift": [0.03, -0.01, 0.01, 0.0],
            "auc_lift_ci_low": [-0.02, -0.04, -0.03, 0.0],
            "auc_lift_ci_high": [0.08, 0.02, 0.04, 0.0],
            "mask_only_lift": [0.03, -0.01, 0.01, 0.0],
            "time_shuffle_lift": [0.02, -0.02, 0.0, 0.0],
            "headline_eligible": [False, False, False, False],
            "claim_scope": ["diagnostic"] * 4,
            "fail_reason": ["gate_not_passed_or_control_missing"] * 4,
        }
    ).to_csv(artifacts / "sequence_model_fit_diagnostics.csv", index=False)

    report = write_proxy_research_report(
        artifacts_dir=artifacts,
        reports_dir=reports,
        figure_paths={},
    )

    text = report.read_text(encoding="utf-8")
    assert "Sequence diagnostics were unavailable" not in text
    assert "Current sequence diagnostics are populated" in text
    assert "mamba" not in text.lower()
    assert "sequence diagnostics as exploratory" in text
    assert "common-row/control/bootstrap/economics gates" in text


def test_proxy_research_split_tensor_models_and_sanity_tables(tmp_path: Path) -> None:
    events = pd.DataFrame(
        {
            "event_id": [f"E{idx:02d}" for idx in range(40)],
            "ticker": ["AAA" if idx % 2 == 0 else "BBB" for idx in range(40)],
            "event_date": pd.date_range("2023-01-01", periods=40, freq="14D"),
            "entry_date": pd.date_range("2023-01-01", periods=40, freq="14D"),
            "exit_date": pd.date_range("2023-01-02", periods=40, freq="14D"),
            "announcement_timing": ["AMC", "BMO"] * 20,
            "rvar_event": np.linspace(0.010, 0.050, 40),
            "ivar_event": np.linspace(0.012, 0.045, 40),
            "dte_1": [8, 16] * 20,
            "universe_rank": list(range(1, 41)),
            "entry_premium_usd": np.linspace(300, 500, 40),
            "open_option_vwap_0_5_anchor_usd": np.linspace(280, 480, 40),
            "open_option_vwap_5_15_anchor_usd": np.linspace(260, 460, 40),
            "gross_proxy_pnl_usd": np.linspace(-50, 80, 40),
            "gross_c2o_intrinsic_proxy_pnl_usd": np.linspace(-120, 60, 40),
            "gross_post_open_option_vwap_0_5_proxy_pnl_usd": np.linspace(-100, 70, 40),
            "gross_post_open_option_vwap_5_15_proxy_pnl_usd": np.linspace(-90, 90, 40),
            "gross_reaction_o2c_option_vwap_0_5_to_c2c_exit_proxy_pnl_usd": np.linspace(
                -80, 50, 40
            ),
            "gross_reaction_o2c_option_vwap_5_15_to_c2c_exit_proxy_pnl_usd": np.linspace(
                -70, 60, 40
            ),
            "haircut_pnl_usd": np.linspace(-55, 70, 40),
        }
    )
    assert "reaction_o2c" in TARGET_IDS
    rows: list[dict[str, object]] = []
    for event_idx, event_id in enumerate(events["event_id"]):
        for seq_idx in range(20):
            row: dict[str, object] = {
                "event_id": event_id,
                "ticker": events.loc[event_idx, "ticker"],
                "entry_date": events.loc[event_idx, "entry_date"].date(),
                "exit_date": events.loc[event_idx, "exit_date"].date(),
                "source_date": (
                    pd.Timestamp("2022-01-01") + pd.Timedelta(days=seq_idx + event_idx * 14)
                ).date(),
                "seq_index": seq_idx,
                "is_valid_sequence_day": True,
                "has_underlying_close": True,
                "missing_options_day_aggs": False,
            }
            for feature in SEQUENCE_FEATURE_NAMES:
                row[feature] = 0.01 * (event_idx + 1) + 0.001 * seq_idx
            rows.append(row)
    long_rows = pd.DataFrame(rows)
    by_event = sequence_coverage_by_event(long_rows)
    report = sequence_coverage_report(by_event, total_events=len(events))
    assert report["high_sequence_selection_risk"] is False
    split_only = assign_event_splits(events)
    assert split_only.groupby("event_id")["split"].nunique().max() == 1
    aggregates = aggregate_sequence_features(long_rows)
    features = enrich_feature_matrix_for_research(
        events,
        sequence_by_event=by_event,
        sequence_aggregates=aggregates,
    )
    features = features.assign(
        RVAR_event_jump_c2o=features["rvar_event"] * 0.8,
        RVAR_event_day_c2c=features["rvar_event"],
        RVAR_event_reaction_o2c=features["rvar_event"] * 0.2,
    )
    target_rows = pd.concat(
        [
            prepare_target_frame(
                features,
                target_id=target_id,
            )
            for target_id in ["jump_c2o", "day_c2c", "reaction_o2c"]
        ],
        ignore_index=True,
    )
    assert target_rows.groupby("event_id")["split"].nunique().max() == 1
    assert set(target_rows["target_id"]) == {"jump_c2o", "day_c2c", "reaction_o2c"}
    target_rows["forecast_market_implied_event_variance"] = target_rows["ivar_event"]
    target_rows["forecast_linear_elastic_net_tuned"] = target_rows["rvar_event"] * 1.02
    assert set(features["split"]) == {"train", "validation", "test"}
    assert features.groupby("event_id")["split"].nunique().max() == 1
    tensor_path = tmp_path / "sequence_tensor.npz"
    tensor_report = build_sequence_tensor(long_rows, features, out_path=tensor_path)
    payload = np.load(tensor_path, allow_pickle=True)
    assert tensor_report["shape"] == [40, 20, len(SEQUENCE_FEATURE_NAMES)]
    assert payload["time_mask"].shape == (40, 20)
    assert payload["feature_mask"].shape == (40, 20, len(SEQUENCE_FEATURE_NAMES))

    hybrid_rows: list[dict[str, object]] = []
    for event_idx, event_id in enumerate(events["event_id"]):
        for seq_idx in range(HYBRID_STEPS):
            is_intraday = seq_idx >= 19
            row = {
                "event_id": event_id,
                "ticker": events.loc[event_idx, "ticker"],
                "entry_date": events.loc[event_idx, "entry_date"].date(),
                "exit_date": events.loc[event_idx, "exit_date"].date(),
                "source_date": events.loc[event_idx, "entry_date"].date(),
                "source_timestamp": (
                    pd.Timestamp(events.loc[event_idx, "entry_date"])
                    + pd.Timedelta(minutes=seq_idx)
                ).isoformat(),
                "event_entry_timestamp": (
                    pd.Timestamp(events.loc[event_idx, "entry_date"]) + pd.Timedelta(hours=16)
                ).isoformat(),
                "seq_index": seq_idx,
                "is_intraday_bin": float(is_intraday),
                "step_type": "intraday" if is_intraday else "daily",
                "step_type_intraday": float(is_intraday),
                "hybrid_valid_step": True,
            }
            for feature in HYBRID_SEQUENCE_FEATURE_NAMES:
                row.setdefault(feature, 0.01 * (event_idx + 1) + 0.001 * seq_idx)
            hybrid_rows.append(row)
    hybrid_long = pd.DataFrame(hybrid_rows)
    hybrid_by_event = hybrid_sequence_coverage_by_event(hybrid_long)
    assert hybrid_by_event["intraday_valid_bin_count"].min() == 12
    features = features.merge(
        hybrid_by_event[
            [
                "event_id",
                "intraday_valid_bin_count",
                "latest_5min_valid_surface",
                "hybrid_feature_mask_density",
                "hybrid_sequence_eligible_v2",
            ]
        ],
        on="event_id",
        how="left",
        suffixes=("", "_hybrid"),
    )
    if "hybrid_sequence_eligible_v2_hybrid" in features:
        features["hybrid_sequence_eligible_v2"] = features["hybrid_sequence_eligible_v2_hybrid"]
    features["hybrid_sequence_too_sparse"] = False
    hybrid_tensor_path = tmp_path / "hybrid_sequence_tensor.npz"
    hybrid_tensor_report = build_sequence_tensor(
        hybrid_long,
        features,
        out_path=hybrid_tensor_path,
        feature_names=HYBRID_SEQUENCE_FEATURE_NAMES,
        lookback_days=HYBRID_STEPS,
        per_step_type_scaling=True,
    )
    hybrid_payload = np.load(hybrid_tensor_path, allow_pickle=True)
    assert hybrid_tensor_report["shape"] == [40, 31, len(HYBRID_SEQUENCE_FEATURE_NAMES)]
    assert hybrid_payload["time_mask"].shape == (40, 31)
    assert hybrid_payload["feature_mask"].shape == (40, 31, len(HYBRID_SEQUENCE_FEATURE_NAMES))
    assert set(hybrid_payload["step_type"][0].tolist()) == {"daily", "intraday"}
    sequence_quality = build_sequence_v2_quality(features, tensor_path=hybrid_tensor_path)
    assert {"valid_len", "missing_rate", "common_row_eligible"}.issubset(sequence_quality.columns)
    audit = proxy_surface_distribution_audit(hybrid_long.assign(iv_extraction_source="synthetic"))
    assert {"iv_extraction_source", "metric", "missing_rate"}.issubset(audit.columns)

    pytest.importorskip("sklearn")
    model_frame = prepare_target_frame(features, target_id="day_c2c")
    predictions, diagnostics = run_proxy_model_suite(
        model_frame,
        tensor_path=tensor_path,
        hybrid_tensor_path=hybrid_tensor_path,
        model_ids=[
            "market_implied_event_variance",
            "linear_elastic_net_tuned",
            "ridge_flat_aggregates_sequence",
            "attention_pooling_sequence",
        ],
    )
    assert (
        diagnostics.loc[diagnostics["model_id"].eq("market_implied_event_variance"), "status"].iloc[
            0
        ]
        == "evaluated"
    )
    assert (
        diagnostics.loc[
            diagnostics["model_id"].eq("ridge_flat_aggregates_sequence"), "status"
        ].iloc[0]
        == "trained"
    )
    assert (
        diagnostics.loc[diagnostics["model_id"].eq("attention_pooling_sequence"), "status"].iloc[0]
        == "trained"
    )
    ridge_forecast = predictions["forecast_ridge_flat_aggregates_sequence"].dropna()
    assert ridge_forecast.gt(0).all()
    assert ridge_forecast.std() > 0
    for retired_column in [
        "forecast_daily_mamba_20step",
        "forecast_hybrid_mamba_31step",
        "forecast_intraday_only_mamba_12step",
        "forecast_mask_only_hybrid_mamba",
    ]:
        assert retired_column not in predictions.columns

    qlike, extremes = qlike_sanity_table(
        predictions,
        forecast_columns={
            "ridge_flat_aggregates_sequence": "forecast_ridge_flat_aggregates_sequence"
        },
    )
    assert qlike["floored_qlike"].notna().all()
    assert set(
        [
            "model_id",
            "event_id",
            "ticker",
            "event_date",
            "forecast",
            "label",
            "ivar_event",
            "qlike_contribution",
            "percentile",
        ]
    ).issubset(extremes.columns)
    assert proxy_transaction_cost([100.0])[0] == pytest.approx(0.5)
    assert pytest.approx(1e-6) == FORECAST_FLOOR
    inference = inference_table(
        predictions,
        forecast_columns={
            "ridge_flat_aggregates_sequence": "forecast_ridge_flat_aggregates_sequence"
        },
    )
    assert inference["status"].iloc[0] in {"ok", "insufficient_rows", "insufficient_clusters"}
    diagnostics_paths = build_common_row_diagnostics(
        predictions,
        out_dir=tmp_path / "common",
        bootstrap_iter=10,
    )
    assert Path(diagnostics_paths["common_row_universe"]).exists()
    common_rows = pd.read_csv(diagnostics_paths["common_row_universe"])
    assert common_rows["feature_schema_version"].eq(FEATURE_SCHEMA_V2_SEC_XBRL).all()
    assert common_rows["tuning_profile"].eq("tuned_phase1_day_c2c_rank_log_rvar").all()
    retired_manifest = write_retired_model_manifest(tmp_path / "manifest")
    retired_payload = json.loads(retired_manifest.read_text())
    assert "daily_mamba_20step" in retired_payload["retired_model_ids"]
    assert retired_payload["reason"] == "legacy in-repo Mamba-style gated RNN models are retired"
    metrics_dir = tmp_path / "metrics"
    (metrics_dir / "edge_deciles_mamba_sequence_encoder.csv").parent.mkdir()
    (metrics_dir / "edge_deciles_mamba_sequence_encoder.csv").write_text("stale\n")
    (metrics_dir / "strategy_trades_mask_only_mamba_sequence_encoder.csv").write_text("stale\n")
    (metrics_dir / "o2c_option_vwap_5_15_strategy_trades_mamba_sequence_encoder.csv").write_text(
        "stale\n"
    )
    (metrics_dir / "o2c_option_vwap_0_5_strategy_trades_mamba_sequence_encoder.csv").write_text(
        "stale\n"
    )
    metric_paths = build_metric_tables(target_rows, out_dir=metrics_dir)
    assert not (metrics_dir / "edge_deciles_mamba_sequence_encoder.csv").exists()
    assert not (metrics_dir / "strategy_trades_mask_only_mamba_sequence_encoder.csv").exists()
    assert not (
        metrics_dir / "o2c_option_vwap_5_15_strategy_trades_mamba_sequence_encoder.csv"
    ).exists()
    assert not (
        metrics_dir / "o2c_option_vwap_0_5_strategy_trades_mamba_sequence_encoder.csv"
    ).exists()
    strategy = pd.read_csv(metric_paths["strategy_metrics"])
    for artifact_name in (
        "forecast_metrics",
        "ranking_metrics",
        "strategy_metrics",
        "strategy_policy_search",
        "strategy_selected_policies",
        "cost_sensitivity",
        "robustness_summary",
        "qlike_sanity",
        "inference",
    ):
        artifact = pd.read_csv(metric_paths[artifact_name])
        assert artifact["feature_schema_version"].eq(FEATURE_SCHEMA_V2_SEC_XBRL).all()
        assert artifact["tuning_profile"].eq("tuned_phase1_day_c2c_rank_log_rvar").all()
    selected_policies = pd.read_csv(metric_paths["strategy_selected_policies"])
    assert not selected_policies.empty
    assert selected_policies["selected"].astype(bool).all()
    policy_counts = selected_policies.groupby(
        ["target_id", "model_id", "strategy_proxy_kind"], dropna=False
    )["selected"].sum()
    assert policy_counts.eq(1).all()
    assert {"day_c2c", "jump_c2o", "reaction_o2c"}.issubset(set(strategy["target_id"]))
    c2o_strategy = strategy.loc[strategy["target_id"].eq("jump_c2o")]
    assert {
        "c2o_intrinsic_open_diagnostic",
        "post_open_option_vwap_0_5_proxy",
        "post_open_option_vwap_5_15_proxy",
    }.issubset(set(c2o_strategy["strategy_proxy_kind"]))
    assert c2o_strategy["pnl_headline_eligible"].astype(str).str.lower().eq("false").all()
    o2c_strategy = strategy.loc[strategy["target_id"].eq("reaction_o2c")]
    assert {
        "reaction_o2c_option_vwap_0_5_to_c2c_exit_proxy",
        "reaction_o2c_option_vwap_5_15_to_c2c_exit_proxy",
    }.issubset(set(o2c_strategy["strategy_proxy_kind"]))
    assert o2c_strategy["pnl_headline_eligible"].astype(str).str.lower().eq("false").all()
    o2c_trades = pd.read_csv(
        metrics_dir / "o2c_option_vwap_5_15_strategy_trades_linear_elastic_net_tuned.csv"
    )
    if not o2c_trades.empty:
        assert o2c_trades["feature_schema_version"].eq(FEATURE_SCHEMA_V2_SEC_XBRL).all()
        assert o2c_trades["tuning_profile"].eq("tuned_phase1_day_c2c_rank_log_rvar").all()
        expected_anchor = events[["event_id", "open_option_vwap_5_15_anchor_usd"]]
        anchored = o2c_trades.merge(expected_anchor, on="event_id", how="left")
        assert np.allclose(
            anchored["capital_at_risk_usd"],
            anchored["open_option_vwap_5_15_anchor_usd_y"],
        )
        assert np.allclose(
            o2c_trades["estimated_transaction_cost_usd"],
            0.005 * o2c_trades["capital_at_risk_usd"],
        )
    scale = pd.read_csv(metric_paths["o2c_scale_diagnostic"])
    assert np.isfinite(scale["sd_ratio_o2c_to_ivar"].iloc[0])
    assert np.isfinite(scale["mean_ratio_o2c_to_ivar"].iloc[0])
    direct_scale = o2c_scale_diagnostic(target_rows)
    assert int(direct_scale["paired_rows"].iloc[0]) == len(events)


def test_research_report_fails_when_model_artifacts_are_missing(tmp_path: Path) -> None:
    config = replace(
        load_project_config(),
        artifacts_dir=tmp_path / "artifacts",
        reports_dir=tmp_path / "reports",
        gold_data_dir=tmp_path / "gold",
        silver_data_dir=tmp_path / "silver",
    )
    result = run_research_report(config)
    assert result.ok is False
    missing = result.diagnostics["missing_required_artifacts"]
    assert isinstance(missing, list)
    assert "forecast_metrics.csv" in missing


def test_market_index_second_surface_and_underlying_features() -> None:
    cutoff = pd.Timestamp("2025-01-03 16:00:00", tz="America/New_York")
    assert (
        market_index_implied_volatility(
            spot=0.0,
            strike=100.0,
            time_to_expiry=0.1,
            option_price=1.0,
            right="call",
        )
        is None
    )
    assert (
        market_index_implied_volatility(
            spot=100.0,
            strike=90.0,
            time_to_expiry=0.1,
            option_price=1.0,
            right="call",
        )
        is None
    )
    assert (
        market_index_implied_volatility(
            spot=100.0,
            strike=100.0,
            time_to_expiry=0.1,
            option_price=1_000.0,
            right="put",
        )
        is None
    )
    assert normalize_underlying_second_aggregates(pd.DataFrame(), ticker="SPY").empty
    with pytest.raises(ValueError, match="missing required columns"):
        normalize_underlying_second_aggregates(pd.DataFrame({"t": [1]}), ticker="SPY")
    assert (
        select_underlying_second_features(
            pd.DataFrame(),
            cutoff_timestamp=cutoff.to_pydatetime(),
            buffer_minutes=60,
            lookback_seconds=900,
        ).status
        == "no_second_bars"
    )
    raw_underlying = pd.DataFrame(
        {
            "t": [
                int(pd.Timestamp("2025-01-03 20:40:00", tz="UTC").timestamp() * 1000),
                int(pd.Timestamp("2025-01-03 20:50:00", tz="UTC").timestamp() * 1000),
                int(pd.Timestamp("2025-01-03 20:59:00", tz="UTC").timestamp() * 1000),
            ],
            "o": [498.0, 499.0, 500.0],
            "h": [499.0, 500.0, 501.0],
            "l": [497.5, 498.5, 499.5],
            "c": [498.5, 499.5, 500.5],
            "v": [100, 200, 300],
            "vw": [498.4, 499.4, 500.4],
            "n": [10, 20, 30],
        }
    )
    underlying = normalize_underlying_second_aggregates(raw_underlying, ticker="SPY")
    selected_underlying = select_underlying_second_features(
        underlying,
        cutoff_timestamp=cutoff.to_pydatetime(),
        buffer_minutes=60,
        lookback_seconds=900,
    )
    assert selected_underlying.status == "ok"
    assert selected_underlying.close == pytest.approx(500.5)
    assert selected_underlying.volume_sum == 500
    assert selected_underlying.transactions_sum == 50
    assert (
        select_underlying_second_features(
            underlying.head(1),
            cutoff_timestamp=cutoff.to_pydatetime(),
            buffer_minutes=5,
            lookback_seconds=60,
        ).status
        == "no_bars_in_cutoff_buffer"
    )
    assert (
        select_underlying_second_features(
            underlying.head(1),
            cutoff_timestamp=cutoff.to_pydatetime(),
            buffer_minutes=60,
            lookback_seconds=60,
        ).status
        == "no_bars_in_lookback"
    )

    source_date = date(2025, 1, 3)
    spot = 500.0
    assert select_market_index_option_candidates(
        pd.DataFrame(), symbol="SPY", source_date=source_date, spot=spot
    ).empty
    assert select_market_index_option_candidates(
        pd.DataFrame({"ticker": ["O:ABC250110C00500000"], "close": [1], "volume": [1]}),
        symbol="SPY",
        source_date=source_date,
        spot=spot,
    ).empty
    assert select_market_index_option_candidates(
        pd.DataFrame({"ticker": ["O:SPY250110C00500000"], "close": [1], "volume": [1]}),
        symbol="SPY",
        source_date=source_date,
        spot=0.0,
    ).empty
    option_rows: list[dict[str, object]] = []
    for expiry_text, vol in [("250110", 0.20), ("250117", 0.22), ("250131", 0.25)]:
        expiry = pd.Timestamp(f"20{expiry_text[:2]}-{expiry_text[2:4]}-{expiry_text[4:6]}").date()
        tte = (expiry - source_date).days / 365.0
        for strike in (490.0, 500.0, 510.0):
            strike_text = f"{int(strike * 1000):08d}"
            for right_code, right in [("C", OptionRight.CALL), ("P", OptionRight.PUT)]:
                price = black_scholes_price(
                    spot=spot,
                    strike=strike,
                    time_to_expiry=tte,
                    volatility=vol,
                    right=right,
                )
                option_rows.append(
                    {
                        "ticker": f"O:SPY{expiry_text}{right_code}{strike_text}",
                        "close": price,
                        "volume": 100,
                        "transactions": 20,
                    }
                )
    day_options = pd.DataFrame(option_rows)
    candidates = select_market_index_option_candidates(
        day_options,
        symbol="SPY",
        source_date=source_date,
        spot=spot,
    )
    assert not candidates.empty
    bar_frames = {
        str(row["options_ticker"]): pd.DataFrame(
            {
                "timestamp_et": [pd.Timestamp("2025-01-03 15:55:00", tz="America/New_York")],
                "option_vwap": [float(row["option_close"])],
                "option_close": [float(row["option_close"])],
                "volume": [11],
                "transactions": [3],
            }
        )
        for row in candidates.to_dict("records")
    }
    surface = market_index_surface_features(
        candidates,
        bar_frames,
        symbol="SPY",
        spot=spot,
        cutoff_timestamp=cutoff.to_pydatetime(),
        lookback_seconds=900,
    )
    assert surface["market_surface_status"] == "ok"
    assert surface["market_atm_iv_proxy"] == pytest.approx(0.22, abs=0.03)
    assert cast(float, surface["market_straddle_premium_to_spot"]) > 0
    assert cast(int, surface["market_valid_pair_count"]) > 0
    assert cast(int, surface["market_option_transactions_sum"]) > 0
    assert (
        market_index_surface_features(
            candidates.head(0),
            {},
            symbol="SPY",
            spot=spot,
            cutoff_timestamp=cutoff.to_pydatetime(),
            lookback_seconds=900,
        )["market_surface_status"]
        == "no_candidates"
    )
    assert (
        market_index_surface_features(
            candidates,
            {},
            symbol="SPY",
            spot=spot,
            cutoff_timestamp=cutoff.to_pydatetime(),
            lookback_seconds=900,
        )["market_surface_status"]
        == "no_prices"
    )
    prefixed = prefix_market_index_features(surface, symbol="SPY")
    assert "spy_second_market_atm_iv_proxy" in prefixed
    assert "spy_second_index_symbol" not in prefixed


def test_vix_feature_construction_uses_valid_observation_lags_and_prior_history() -> None:
    dates = pd.bdate_range("2024-01-02", periods=60)
    raw = pd.DataFrame(
        {
            "DATE": [day.date().isoformat() for day in dates],
            "VIXCLS": [str(10.0 + idx) for idx in range(len(dates))],
        }
    )
    vix = normalize_fred_vixcls_csv(raw, source_snapshot_date=date(2024, 4, 1))
    feature_date = dates[45].date()
    resolved_date = dates[44].date()
    features = build_vix_features(
        vix,
        pd.DataFrame(
            {
                "event_id": ["E1"],
                "feature_asof_date": [feature_date],
                "announcement_timing": ["AMC"],
            }
        ),
        alignment=VIX_ALIGNMENT_PRIOR_CLOSE,
    )
    assert features["resolved_vix_date"].iloc[0] == resolved_date
    assert features["vix_change_1d"].iloc[0] == pytest.approx(1.0)
    assert features["vix_change_5d"].iloc[0] == pytest.approx(5.0)
    history = vix.loc[vix["vix_close_date"].lt(resolved_date), "vix_close"].tail(252)
    expected_percentile = float((history <= features["vix_level"].iloc[0]).mean())
    assert features["vix_percentile_252d"].iloc[0] == pytest.approx(expected_percentile)
    assert features["vix_regime_tercile"].iloc[0] == "high"
    assert bool(features["vix_above_30"].iloc[0]) is True


def test_vix_alignment_same_day_amc_only_and_stale_observations_are_missing() -> None:
    raw = pd.DataFrame(
        {
            "DATE": ["2024-01-02", "2024-01-03", "2024-01-04", "2024-01-05", "2024-01-08"],
            "VIXCLS": ["12", "13", "14", "15", "."],
        }
    )
    vix = normalize_fred_vixcls_csv(
        raw,
        source_snapshot_date=date(2024, 1, 9),
        source_url=FRED_VIXCLS_URL,
    )
    assert int(vix["is_holiday_or_missing"].sum()) == 1
    assert vix["source_dataset"].iloc[0] == "fred_vixcls"
    assert vix["source_url"].iloc[0] == FRED_VIXCLS_URL
    assert vix["source_snapshot_date"].iloc[0] == "2024-01-09"

    prior = build_vix_features(
        vix,
        pd.DataFrame(
            {
                "event_id": ["AMC", "BMO"],
                "feature_asof_date": [date(2024, 1, 5), date(2024, 1, 5)],
                "announcement_timing": ["AMC", "BMO"],
            }
        ),
        alignment=VIX_ALIGNMENT_PRIOR_CLOSE,
    )
    assert prior["resolved_vix_date"].tolist() == [date(2024, 1, 4), date(2024, 1, 4)]

    robustness = build_vix_features(
        vix,
        pd.DataFrame(
            {
                "event_id": ["AMC", "BMO"],
                "feature_asof_date": [date(2024, 1, 5), date(2024, 1, 5)],
                "announcement_timing": ["AMC", "BMO"],
            }
        ),
        alignment=VIX_ALIGNMENT_SAME_DAY_AMC,
    )
    assert robustness["resolved_vix_date"].tolist() == [date(2024, 1, 5), date(2024, 1, 4)]

    stale = build_vix_features(
        vix,
        pd.DataFrame({"event_id": ["STALE"], "feature_asof_date": [date(2024, 1, 12)]}),
        alignment=VIX_ALIGNMENT_PRIOR_CLOSE,
        max_lag_days=5,
    )
    assert bool(stale["vix_available"].iloc[0]) is False
    assert pd.isna(stale["resolved_vix_date"].iloc[0])
    assert str(stale["vix_above_30"].dtype) == "boolean"
    schema = build_feature_schema_report(stale, feature_schema_version=FEATURE_SCHEMA_V2_SEC_XBRL)
    selected = set(feature_columns_from_schema_report(schema, frame=stale))
    assert "vix_above_30" in selected


def test_vix_feature_edge_cases_and_validation_errors() -> None:
    lower = normalize_fred_vixcls_csv(
        pd.DataFrame({"observation_date": ["2024-01-02"], "vix_close": [22.0]}),
        source_snapshot_date=date(2024, 1, 3),
    )
    assert lower["vix_close"].iloc[0] == pytest.approx(22.0)

    with pytest.raises(ValueError, match="DATE"):
        normalize_fred_vixcls_csv(
            pd.DataFrame({"VIXCLS": [12]}),
            source_snapshot_date=date(2024, 1, 3),
        )
    with pytest.raises(ValueError, match="VIXCLS"):
        normalize_fred_vixcls_csv(
            pd.DataFrame({"DATE": ["2024-01-02"]}),
            source_snapshot_date=date(2024, 1, 3),
        )
    with pytest.raises(ValueError, match="vix_close"):
        build_vix_features(
            pd.DataFrame({"date": ["2024-01-02"]}),
            pd.DataFrame({"feature_asof_date": [date(2024, 1, 3)]}),
        )
    with pytest.raises(ValueError, match="date"):
        build_vix_features(
            pd.DataFrame({"vix_close": [12]}),
            pd.DataFrame({"feature_asof_date": [date(2024, 1, 3)]}),
        )
    with pytest.raises(ValueError, match="unsupported"):
        build_vix_features(
            lower,
            pd.DataFrame({"feature_asof_date": [date(2024, 1, 3)]}),
            alignment=cast(Any, "bad"),
        )
    with pytest.raises(ValueError, match="feature_asof_date"):
        build_vix_features(lower, pd.DataFrame({"event_id": ["E1"]}))
    with pytest.raises(ValueError, match="max_lag"):
        build_vix_features(
            lower,
            pd.DataFrame({"feature_asof_date": [date(2024, 1, 3)]}),
            max_lag_days=-1,
        )
    with pytest.raises(ValueError, match="percentile_window"):
        build_vix_features(
            lower,
            pd.DataFrame({"feature_asof_date": [date(2024, 1, 3)]}),
            percentile_window=0,
        )

    missing_date = build_vix_features(lower, pd.DataFrame({"feature_asof_date": [pd.NaT]}))
    assert bool(missing_date["vix_available"].iloc[0]) is False
    before_first = build_vix_features(
        lower,
        pd.DataFrame({"feature_asof_date": [date(2024, 1, 2)]}),
    )
    assert bool(before_first["vix_available"].iloc[0]) is False

    low_raw = pd.DataFrame(
        {
            "DATE": ["2024-01-02", "2024-01-03", "2024-01-04", "2024-01-05"],
            "VIXCLS": ["10", "20", "30", "15"],
        }
    )
    low = build_vix_features(
        normalize_fred_vixcls_csv(low_raw, source_snapshot_date=date(2024, 1, 6)),
        pd.DataFrame({"feature_asof_date": [date(2024, 1, 8)]}),
        min_regime_observations=3,
    )
    assert low["vix_regime_tercile"].iloc[0] == "low"

    mid_raw = pd.DataFrame(
        {
            "DATE": ["2024-01-02", "2024-01-03", "2024-01-04", "2024-01-05"],
            "VIXCLS": ["10", "20", "30", "25"],
        }
    )
    mid = build_vix_features(
        normalize_fred_vixcls_csv(mid_raw, source_snapshot_date=date(2024, 1, 6)),
        pd.DataFrame({"feature_asof_date": [date(2024, 1, 8)]}),
        min_regime_observations=3,
    )
    assert mid["vix_regime_tercile"].iloc[0] == "mid"


def test_research_model_suite_and_error_paths() -> None:
    feature_frame = pd.DataFrame(
        {
            "event_id": [f"E{idx}" for idx in range(8)],
            "ticker": ["AAA", "AAA", "AAA", "AAA", "BBB", "BBB", "BBB", "BBB"],
            "announcement_date": pd.date_range("2024-01-01", periods=8, freq="30D"),
            "rvar_event": [0.05, 0.03, 0.06, 0.08, 0.02, 0.04, 0.07, 0.09],
            "ivar_event": [0.04, 0.04, 0.05, 0.06, 0.03, 0.03, 0.05, 0.06],
            "feature_a": [1, 2, 3, 4, 1, 2, 3, 4],
            "feature_b": [True, False, True, False, True, False, True, False],
            "entry_premium_usd": [400, 410, 420, 430, 300, 310, 320, 330],
            "gross_proxy_pnl_usd": [40, -20, 60, 80, -10, 30, 70, 90],
            "haircut_pnl_usd": [5, -55, 23, 43, -40, 1, 38, 57],
        }
    )

    with pytest.raises(ValueError, match="announcement_date"):
        temporal_train_test_split(pd.DataFrame({"ticker": ["AAA"], "rvar_event": [0.1]}))
    with pytest.raises(ValueError, match="missing required"):
        add_benchmark_predictions(feature_frame.drop(columns=["ivar_event"]))
    with pytest.raises(ValueError, match="model is not fit"):
        LinearElasticNetRegressor().predict(feature_frame)
    with pytest.raises(ValueError, match="alpha"):
        LinearElasticNetRegressor(alpha=-0.1)
    with pytest.raises(ValueError, match="l1_ratio"):
        LinearElasticNetRegressor(l1_ratio=1.5)
    regressor = LinearElasticNetRegressor(max_iter=25)
    regressor.fit(
        feature_frame, target_col="rvar_event", feature_columns=["feature_a", "feature_b"]
    )
    regressor_predictions = regressor.predict(feature_frame)
    assert regressor_predictions.shape == (len(feature_frame),)
    assert np.isfinite(regressor_predictions).all()
    ridge = RidgeRegressor(alpha=0.01)
    ridge.fit(feature_frame, target_col="rvar_event", feature_columns=["feature_a", "feature_b"])
    ridge_predictions = ridge.predict(feature_frame)
    assert ridge_predictions.shape == (len(feature_frame),)
    assert np.isfinite(ridge_predictions).all()
    with pytest.raises(ValueError, match="alpha"):
        RidgeRegressor(alpha=-0.1)
    with pytest.raises(ValueError, match="model is not fit"):
        RidgeRegressor().predict(feature_frame)
    with pytest.raises(ValueError, match="no finite target"):
        regressor.fit(
            feature_frame.assign(rvar_event=np.nan),
            target_col="rvar_event",
            feature_columns=["feature_a", "feature_b"],
        )
    split_train, split_test = temporal_train_test_split(feature_frame, split_date="2024-06-01")
    assert len(split_train) == 6
    assert len(split_test) == 2
    frac_train, frac_test = temporal_train_test_split(feature_frame, train_fraction=0.5)
    assert len(frac_train) == 4
    assert len(frac_test) == 4

    predictions, results = run_model_suite(
        feature_frame,
        model_ids=[
            "market_implied_event_variance",
            "last_four_rvar",
            "last_four_ivar",
            "goyal_saretto_rv_iv_spread",
        ],
        split_date="2024-06-01",
    )
    assert "forecast_market_implied_event_variance" in predictions
    assert predictions["split"].tolist().count("test") == 2
    diagnostics = model_diagnostics_as_frame(results)
    assert set(diagnostics["model_id"]) == {
        "market_implied_event_variance",
        "last_four_rvar",
        "last_four_ivar",
        "goyal_saretto_rv_iv_spread",
    }
    assert (
        diagnostics.loc[diagnostics["model_id"].eq("market_implied_event_variance"), "status"].iloc[
            0
        ]
        == "evaluated"
    )
    seq_frame = feature_frame.assign(
        seq_t00_atm_iv=np.linspace(0.2, 0.3, len(feature_frame)),
        seq_t01_atm_iv=np.linspace(0.21, 0.31, len(feature_frame)),
        seq_t00_volume=np.linspace(10, 17, len(feature_frame)),
        seq_t01_volume=np.linspace(11, 18, len(feature_frame)),
    )
    assert sequence_feature_columns(seq_frame) == [
        "seq_t00_atm_iv",
        "seq_t00_volume",
        "seq_t01_atm_iv",
        "seq_t01_volume",
    ]
    assert (
        prediction_column_for_model("linear_elastic_net_tuned")
        == "forecast_linear_elastic_net_tuned"
    )
    with pytest.raises(KeyError):
        prediction_column_for_model("linear_elastic_net")
    with pytest.raises(ValueError, match="unknown model_id"):
        run_model_suite(feature_frame, model_ids=["not_a_model"])
    with pytest.raises(ValueError, match="not a trainable"):
        fit_model("not_a_trainable_model", feature_frame)
    with pytest.raises(ValueError, match="at least one sequence"):
        sequence_tensor_from_frame(seq_frame, [])
    with pytest.raises(ValueError, match="invalid sequence"):
        sequence_tensor_from_frame(seq_frame, ["bad_name"])

    same_time = pd.DataFrame(
        {
            "event_id": ["E1", "E2", "E3"],
            "ticker": ["AAA", "AAA", "AAA"],
            "event_entry_timestamp": pd.to_datetime(
                ["2025-01-02 21:00Z", "2025-01-02 21:00Z", "2025-02-01 21:00Z"]
            ),
            "rvar_event": [0.10, 0.20, 0.30],
            "ivar_event": [0.05, 0.06, 0.07],
        }
    )
    same_time_predictions = add_benchmark_predictions(same_time)
    assert same_time_predictions["forecast_last_four_rvar"].iloc[:2].tolist() == [
        pytest.approx(0.05),
        pytest.approx(0.06),
    ]
    assert same_time_predictions["forecast_last_four_rvar"].iloc[2] == pytest.approx(0.15)
    assert same_time_predictions["forecast_goyal_saretto_rv_iv_spread"].iloc[0] == pytest.approx(
        0.05
    )
    assert bool(same_time_predictions["goyal_saretto_fallback_spread_used"].iloc[0]) is True
    assert same_time_predictions["goyal_saretto_signed_rv_iv_spread"].iloc[2] == pytest.approx(
        0.095
    )
    negative_spread = add_benchmark_predictions(
        pd.DataFrame(
            {
                "event_id": ["N1", "N2"],
                "ticker": ["NEG", "NEG"],
                "event_date": pd.to_datetime(["2025-01-01", "2025-02-01"]),
                "rvar_event": [0.02, 0.04],
                "ivar_event": [0.08, 0.03],
            }
        )
    )
    assert negative_spread["goyal_saretto_signed_rv_iv_spread"].iloc[1] == pytest.approx(-0.06)
    assert negative_spread["forecast_goyal_saretto_rv_iv_spread"].iloc[1] == pytest.approx(0.0)
    with pytest.raises(ValueError, match="sequence must have shape"):
        DilatedCNNSequenceEncoder(n_features=2)(torch.zeros(2, 2))


def test_feature_matrix_edge_cases_and_sequence_eligibility() -> None:
    assert (
        sequence_eligibility_reason(
            [date(2025, 1, 1), date(2025, 1, 2)],
            entry_date=date(2025, 1, 3),
            required_trading_days=2,
        )
        is None
    )
    assert (
        sequence_eligibility_reason(
            [date(2025, 1, 1)],
            entry_date=date(2025, 1, 3),
            required_trading_days=2,
        )
        == "insufficient_2_day_sequence"
    )
    with pytest.raises(ValueError, match="missing required"):
        build_model_feature_matrix(pd.DataFrame({"ticker": ["AAA"], "rvar_event": [0.1]}))
    with pytest.raises(ValueError, match="requires announcement_date"):
        build_model_feature_matrix(
            pd.DataFrame({"ticker": ["AAA"], "rvar_event": [0.1], "ivar_event": [0.2]})
        )

    panel = pd.DataFrame(
        {
            "ticker": ["AAA"],
            "event_date": ["2025-01-01"],
            "entry_date": ["2025-01-01"],
            "rvar_event": [0.05],
            "ivar_event": [0.04],
            "dte": [22],
        }
    )
    features = build_model_feature_matrix(panel)
    assert features["dte_bucket"].iloc[0] == "ivar_support_gt_21"
    assert bool(features["is_robustness_dte_3_21"].iloc[0]) is False
    assert "event_entry_timestamp" in features
    assert "feature_asof_timestamp" in features
    with pytest.raises(ValueError, match="at most one row per event_id"):
        build_model_feature_matrix(
            panel.assign(event_id=["AAA_2025Q1"]),
            straddle_diagnostics=pd.DataFrame(
                {
                    "event_id": ["AAA_2025Q1", "AAA_2025Q1"],
                    "entry_premium_usd": [100.0, 101.0],
                }
            ),
        )

    with pytest.raises(ValueError, match="sequence rows missing"):
        build_option_surface_sequence_matrix(pd.DataFrame({"event_id": ["E1"]}))
    empty_sequence = build_option_surface_sequence_matrix(
        pd.DataFrame(
            {
                "event_id": ["E1"],
                "day_index": [0],
                "atm_iv": [0.2],
                "option_volume": [10],
                "spread_over_mid": [0.1],
            }
        )
    )
    assert empty_sequence.columns.tolist() == ["event_id"]


def test_transaction_cost_and_portfolio_caps_scale_trades() -> None:
    quotes = [
        OptionQuote(
            ticker="ABC",
            quote_date=date(2026, 2, 5),
            expiration=date(2026, 2, 13),
            strike=100,
            right=OptionRight.CALL,
            bid=4.0,
            ask=4.4,
        )
    ]
    assert estimated_transaction_cost_usd(quotes) == pytest.approx(20.0)
    assert quotes[0].mid == pytest.approx(4.2)
    assert quotes[0].spread == pytest.approx(0.4)
    assert quotes[0].spread_over_mid == pytest.approx(0.4 / 4.2)
    with pytest.raises(ValidationError, match="ask must be greater"):
        OptionQuote(
            ticker="ABC",
            quote_date=date(2026, 2, 5),
            expiration=date(2026, 2, 13),
            strike=100,
            right=OptionRight.CALL,
            bid=4.5,
            ask=4.4,
        )
    with pytest.raises(ValidationError, match="feature_asof_timestamp"):
        FeatureRow(
            ticker="ABC",
            event_date=date(2026, 2, 5),
            feature_asof_timestamp=datetime(2026, 2, 5, 16, 1),
            event_entry_timestamp=datetime(2026, 2, 5, 16, 0),
        )
    assert (
        FeatureRow(
            ticker="ABC",
            event_date=date(2026, 2, 5),
            feature_asof_timestamp=datetime(2026, 2, 5, 16, 0),
            event_entry_timestamp=datetime(2026, 2, 5, 16, 0),
        ).ticker
        == "ABC"
    )
    with pytest.raises(ValidationError, match="edge_var must equal"):
        SignalRecord(
            ticker="ABC",
            event_date=date(2026, 2, 5),
            strategy="long_atm_straddle",
            forecast_rvar_event=0.05,
            ivar_event=0.04,
            edge_var=0.02,
            expected_strategy_value_usd=120.0,
            market_entry_cost_usd=100.0,
            expected_strategy_edge_usd=20.0,
            estimated_transaction_cost_usd=5.0,
        )
    with pytest.raises(ValidationError, match="expected_strategy_edge_usd"):
        SignalRecord(
            ticker="ABC",
            event_date=date(2026, 2, 5),
            strategy="long_atm_straddle",
            forecast_rvar_event=0.05,
            ivar_event=0.04,
            edge_var=0.01,
            expected_strategy_value_usd=120.0,
            market_entry_cost_usd=100.0,
            expected_strategy_edge_usd=10.0,
            estimated_transaction_cost_usd=5.0,
        )
    with pytest.raises(ValueError, match="max_loss_per_contract_usd"):
        integer_contract_count(target_max_loss_usd=100, max_loss_per_contract_usd=0)
    assert integer_contract_count(target_max_loss_usd=250, max_loss_per_contract_usd=100) == 2
    with pytest.raises(ValueError, match="nav_usd"):
        apply_portfolio_caps([], nav_usd=0)

    timestamp = datetime(2026, 2, 5, 16)
    trade_a = StrategyTrade(
        ticker="AAA",
        event_date=date(2026, 2, 5),
        strategy="long_atm_straddle",
        sector="Information Technology",
        expected_net_edge_usd=500,
        max_theoretical_loss_usd=2_000,
        legs=(
            TradeLeg(
                ticker="AAA",
                expiration=date(2026, 2, 13),
                strike=100,
                right=OptionRight.CALL,
                side=OptionSide.LONG,
                contracts=2,
                filled_price=5,
                filled_timestamp=timestamp,
            ),
        ),
    )
    trade_b = trade_a.model_copy(
        update={
            "ticker": "BBB",
            "expected_net_edge_usd": 1500,
            "max_theoretical_loss_usd": 2_000,
        }
    )
    capped = apply_portfolio_caps(
        [trade_a, trade_b],
        nav_usd=100_000,
        per_event_loss_fraction=0.05,
    )
    assert sum(trade.max_theoretical_loss_usd for trade in capped) <= 3_000
    assert capped[1].max_theoretical_loss_usd > capped[0].max_theoretical_loss_usd

    zero_edge_capped = apply_portfolio_caps(
        [
            trade_a.model_copy(
                update={"expected_net_edge_usd": -1, "max_theoretical_loss_usd": 100}
            ),
            trade_b.model_copy(
                update={"expected_net_edge_usd": 0, "max_theoretical_loss_usd": 100}
            ),
        ],
        nav_usd=100,
        per_event_loss_fraction=1.0,
        event_date_loss_fraction=0.50,
        sector_event_date_loss_fraction=0.50,
    )
    assert [trade.max_theoretical_loss_usd for trade in zero_edge_capped] == [25.0, 25.0]


def test_portfolio_caps_handle_zero_scaled_trade_on_second_cap_pass() -> None:
    timestamp = datetime(2026, 2, 5, 16)
    base_leg = TradeLeg(
        ticker="AAA",
        expiration=date(2026, 2, 13),
        strike=100,
        right=OptionRight.CALL,
        side=OptionSide.LONG,
        contracts=1,
        filled_price=5,
        filled_timestamp=timestamp,
    )
    trades = [
        StrategyTrade(
            ticker=ticker,
            event_date=date(2026, 2, 5),
            strategy="long_atm_straddle",
            sector="Information Technology",
            expected_net_edge_usd=edge,
            max_theoretical_loss_usd=100,
            legs=(base_leg.model_copy(update={"ticker": ticker}),),
        )
        for ticker, edge in [("AAA", 100), ("BBB", 0), ("CCC", 100)]
    ]

    capped = apply_portfolio_caps(
        trades,
        nav_usd=100,
        per_event_loss_fraction=1.0,
        event_date_loss_fraction=0.50,
        sector_event_date_loss_fraction=0.25,
    )

    assert len(capped) == 3
    assert capped[1].max_theoretical_loss_usd == 0
    assert sum(trade.max_theoretical_loss_usd for trade in capped) <= 25


def test_cli_smoke_commands(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    assert main(["status"]) == 0
    assert cli_module._path_status(tmp_path)["exists"] is True

    audit_out = tmp_path / "audit"
    assert (
        main(
            [
                "audit-data",
                "--quotes",
                str(FIXTURES / "option_quotes.csv"),
                "--underlying",
                str(FIXTURES / "underlying_bars.csv"),
                "--earnings",
                str(FIXTURES / "earnings_calendar.csv"),
                "--out",
                str(audit_out),
            ]
        )
        == 0
    )
    assert (audit_out / "required_fields_report.json").exists()

    data_out = tmp_path / "data_pipeline_cli"
    assert (
        main(
            [
                "data",
                "--stage",
                "fixture-audit",
                "--out-dir",
                str(data_out),
            ]
        )
        == 0
    )
    assert (data_out / "data_pipeline_manifest.json").exists()

    monkeypatch.setenv("ARTIFACTS_DIR", str(tmp_path / "configured_artifacts"))
    assert main(["data", "--stage", "fixture-audit"]) == 0
    assert (
        tmp_path / "configured_artifacts" / "data_pipeline" / "data_pipeline_manifest.json"
    ).exists()

    calendar_out = tmp_path / "calendar.csv"
    assert (
        main(
            [
                "validate-calendar",
                "--input",
                str(FIXTURES / "earnings_calendar.csv"),
                "--out",
                str(calendar_out),
            ]
        )
        == 0
    )
    assert calendar_out.exists()

    calendar = validate_calendar_frame(pd.read_csv(FIXTURES / "earnings_calendar.csv"))
    assert int(calendar["is_main_sample_timing"].sum()) == 2

    sec_dir = tmp_path / "sec_cli"
    massive_dir = tmp_path / "massive_cli"
    sec_dir.mkdir()
    massive_dir.mkdir()
    (sec_dir / "AAPL.json").write_text(
        json.dumps(
            _sec_submissions_payload(
                [
                    {
                        "accessionNumber": "0000320193-26-000005",
                        "filingDate": "2026-01-29",
                        "acceptanceDateTime": "2026-01-29T21:30:33.000Z",
                        "form": "8-K",
                        "items": "2.02,9.01",
                    }
                ]
            )
        ),
        encoding="utf-8",
    )
    (massive_dir / "AAPL.json").write_text(
        json.dumps(
            _massive_text_payload(
                [
                    {
                        "accession_number": "0000320193-26-000005",
                        "items_text": (
                            "Item 2.02 Results of Operations and Financial Condition. "
                            "Apple announced financial results for its fiscal quarter."
                        ),
                    }
                ]
            )
        ),
        encoding="utf-8",
    )
    calendar_build_out = tmp_path / "calendar_build"
    assert (
        main(
            [
                "build-earnings-calendar",
                "--tickers",
                "AAPL",
                "--start",
                "2026-01-01",
                "--end",
                "2026-12-31",
                "--sec-submissions-dir",
                str(sec_dir),
                "--massive-8k-text-dir",
                str(massive_dir),
                "--out",
                str(calendar_build_out),
            ]
        )
        == 0
    )
    assert (calendar_build_out / "earnings_calendar_candidates.csv").exists()

    align_out = tmp_path / "windows.csv"
    assert (
        main(
            [
                "align-events",
                "--earnings",
                str(FIXTURES / "earnings_calendar.csv"),
                "--underlying",
                str(FIXTURES / "underlying_bars.csv"),
                "--out",
                str(align_out),
            ]
        )
        == 0
    )
    assert align_out.exists()

    variance_out = tmp_path / "variance.csv"
    assert (
        main(
            [
                "compute-variance",
                "--ivar-input",
                str(FIXTURES / "ivar_input.csv"),
                "--prices",
                str(FIXTURES / "event_prices.csv"),
                "--out",
                str(variance_out),
            ]
        )
        == 0
    )
    assert pd.read_csv(variance_out)["ivar_event"].notna().all()

    event_contracts_input = tmp_path / "event_contracts.csv"
    event_contracts_input.write_text(
        "\n".join(
            [
                "event_id,ticker,entry_date,exit_date",
                "ABC_2026Q1,ABC,2026-02-05,2026-02-06",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    contract_input = tmp_path / "contracts.csv"
    contract_input.write_text(
        "\n".join(
            [
                (
                    "ticker,expiration,strike,right,options_ticker,option_multiplier,"
                    "contract_size,deliverable_status,corporate_action_flag"
                ),
                "ABC,2026-02-13,100,call,O:ABC260213C00100000,100,100,standard,false",
                "ABC,2026-02-13,105,call,O:ABC260213C00105000,150,100,standard,false",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    discovered_out = tmp_path / "discovered_contracts.csv"
    assert (
        main(
            [
                "discover-option-contracts",
                "--events",
                str(event_contracts_input),
                "--contracts",
                str(contract_input),
                "--out",
                str(discovered_out),
            ]
        )
        == 0
    )
    discovered = pd.read_csv(discovered_out)
    assert discovered["eligible_for_quote_pool"].sum() == 1

    event_panel_input = tmp_path / "event_panel_events.csv"
    event_panel_input.write_text(
        "\n".join(
            [
                "event_id,ticker,entry_date,spot,rvar_event,ivar_event",
                "ABC_2026Q1,ABC,2026-02-05,100,0.001,0.04",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    event_quotes_input = tmp_path / "event_quotes.csv"
    event_quotes_input.write_text(
        "\n".join(
            [
                "event_id,ticker,expiration,strike,right,bid,ask",
                "ABC_2026Q1,ABC,2026-02-13,100,call,4.9,5.1",
                "ABC_2026Q1,ABC,2026-02-13,100,put,4.4,4.6",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    panel_out = tmp_path / "event_panel.csv"
    assert (
        main(
            [
                "build-event-panel",
                "--events",
                str(event_panel_input),
                "--quotes",
                str(event_quotes_input),
                "--out",
                str(panel_out),
            ]
        )
        == 0
    )
    panel = pd.read_csv(panel_out)
    assert panel["forward_source"].iloc[0] == FORWARD_SOURCE_PUT_CALL_PARITY

    leakage_out = tmp_path / "leakage"
    assert (
        main(
            [
                "leakage-audit",
                "--features",
                str(FIXTURES / "features_clean.csv"),
                "--out",
                str(leakage_out),
            ]
        )
        == 0
    )
    assert json.loads((leakage_out / "leakage_report.json").read_text())["ok"] is True

    backtest_out = tmp_path / "backtest"
    assert (
        main(
            [
                "backtest-smoke",
                "--legs",
                str(FIXTURES / "trade_legs.csv"),
                "--signals",
                str(FIXTURES / "signals.csv"),
                "--out",
                str(backtest_out),
            ]
        )
        == 0
    )
    assert (backtest_out / "backtest_smoke_signal.json").exists()

    feature_panel = tmp_path / "trade_proxy_panel.parquet"
    pd.DataFrame(
        {
            "event_id": ["E1", "E2", "E3", "E4"],
            "ticker": ["ABC", "ABC", "XYZ", "XYZ"],
            "announcement_date": ["2025-01-01", "2025-02-01", "2025-03-01", "2025-04-01"],
            "entry_date": ["2025-01-01", "2025-01-31", "2025-03-01", "2025-03-31"],
            "announcement_timing": ["AMC", "BMO", "AMC", "BMO"],
            "rvar_event": [0.05, 0.04, 0.03, 0.06],
            "ivar_event": [0.04, 0.05, 0.03, 0.04],
            "dte_1": [8, 9, 12, 16],
            "universe_rank": [1, 2, 3, 4],
        }
    ).to_parquet(feature_panel, index=False)
    feature_straddles = tmp_path / "straddles.csv"
    pd.DataFrame(
        {
            "event_id": ["E1", "E2", "E3", "E4"],
            "entry_premium_usd": [400, 420, 300, 350],
            "gross_proxy_pnl_usd": [80, -30, 20, 90],
            "haircut_pnl_usd": [40, -72, -10, 55],
        }
    ).to_csv(feature_straddles, index=False)
    features_out = tmp_path / "features.parquet"
    assert (
        main(
            [
                "build-feature-matrix",
                "--panel",
                str(feature_panel),
                "--straddles",
                str(feature_straddles),
                "--out",
                str(features_out),
            ]
        )
        == 0
    )
    models_out = tmp_path / "models"
    assert (
        main(
            [
                "train-models",
                "--features",
                str(features_out),
                "--out",
                str(models_out),
                "--models",
                "market_implied_event_variance,last_four_rvar,goyal_saretto_rv_iv_spread",
            ]
        )
        == 0
    )
    assert (models_out / "forecast_metrics.csv").exists()
    assert (models_out / "strategy_metrics.csv").exists()


def test_cli_private_command_branches(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    csv_path = tmp_path / "table.csv"
    cli_module._write_table(csv_path, pd.DataFrame({"x": [1]}))
    assert cli_module._read_table(csv_path)["x"].tolist() == [1]
    assert cli_module._model_ids_from_args([]) == [
        "market_implied_event_variance",
        "last_four_rvar",
        "last_four_ivar",
        "goyal_saretto_rv_iv_spread",
    ]
    assert cli_module._model_ids_from_args(["market_implied_event_variance,last_four_rvar"]) == [
        "market_implied_event_variance",
        "last_four_rvar",
    ]
    with pytest.raises(ValueError, match="unknown model ids"):
        cli_module._model_ids_from_args(["does_not_exist"])
    with pytest.raises(SystemExit):
        cli_module._source_probe("bad", load_project_config())

    manifest: dict[str, object] = {
        "date": "2025-02-05",
        "objects": [{"dataset": "options_day_aggs", "ok": True}],
    }

    def fake_manifest(*args: object, **kwargs: object) -> dict[str, object]:
        _ = args
        assert kwargs["run_head"] is False
        return manifest

    monkeypatch.setattr(cli_module, "massive_flat_file_manifest", fake_manifest)
    metadata_args = argparse.Namespace(
        date="2025-02-05",
        out=tmp_path / "metadata",
        metadata_only=True,
        no_head=True,
        aws_executable="aws",
        sample_rows=3,
    )
    assert cli_module._massive_flat_files(metadata_args, load_project_config()) == 0
    assert (tmp_path / "metadata" / "massive_flat_file_manifest.json").exists()

    def fake_sample(*args: object, **kwargs: object) -> dict[str, object]:
        _ = args
        assert kwargs["sample_rows"] == 3
        return {
            "manifest": {
                "date": "2025-02-05",
                "objects": [{"dataset": "options_day_aggs", "ok": False}],
            }
        }

    monkeypatch.setattr(cli_module, "build_massive_day_agg_sample", fake_sample)
    sample_args = argparse.Namespace(
        date="2025-02-05",
        out=tmp_path / "sample",
        metadata_only=False,
        no_head=False,
        aws_executable="aws",
        sample_rows=3,
    )
    assert cli_module._massive_flat_files(sample_args, load_project_config()) == 1

    def fake_research_package(*args: object, **kwargs: object) -> dict[str, object]:
        _ = args
        assert kwargs["feature_schema_version"] == FEATURE_SCHEMA_V2_SEC_XBRL
        assert "mamba_backend" not in kwargs
        assert "mamba_seeds" not in kwargs
        return {"ok": False, "stage": kwargs["stage"]}

    monkeypatch.setattr(cli_module, "run_proxy_research_package", fake_research_package)
    research_args = argparse.Namespace(
        stage="features",
        split_design="chronological_proxy_70_15_15",
        split_date=None,
        allow_high_sequence_risk=True,
        sequence_suite="all",
        bootstrap_iter=10,
        tuning_profile="tuned_phase1_day_c2c_rank_log_rvar",
        tuning_seed=17,
        feature_schema_version=FEATURE_SCHEMA_V2_SEC_XBRL,
        reuse_tuning_params=False,
    )
    assert cli_module._research(research_args, load_project_config()) == 1


def test_compute_variance_uses_event_exit_date_and_rejects_duplicates(tmp_path: Path) -> None:
    assert cli_module._parse_bool_cell(None) is False
    assert cli_module._parse_bool_cell("true") is True
    assert cli_module._parse_bool_cell("false") is False
    assert cli_module._parse_bool_cell("unexpected", default=True) is True

    ivar_input = tmp_path / "ivar_input.csv"
    ivar_input.write_text(
        "\n".join(
            [
                "ticker,event_date,event_exit_date,expiration,iv,dte_days,stale",
                "ABC,2026-02-05,2026-02-05,2026-02-05,0.80,1,false",
                "ABC,2026-02-05,2026-02-05,2026-02-12,0.70,8,false",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    prices = tmp_path / "event_prices.csv"
    prices.write_text("ticker,event_date,s_before,s_after\nABC,2026-02-05,100,101\n")
    out = tmp_path / "variance.csv"

    assert (
        main(
            [
                "compute-variance",
                "--ivar-input",
                str(ivar_input),
                "--prices",
                str(prices),
                "--out",
                str(out),
            ]
        )
        == 0
    )
    variance = pd.read_csv(out)
    assert variance["ivar_event"].notna().iloc[0]
    assert variance["t1"].iloc[0] == pytest.approx(1 / 365)

    duplicate_prices = tmp_path / "duplicate_event_prices.csv"
    duplicate_prices.write_text(
        "ticker,event_date,s_before,s_after\nABC,2026-02-05,100,101\nABC,2026-02-05,100,102\n"
    )
    with pytest.raises(ValueError, match="duplicate event price rows"):
        main(
            [
                "compute-variance",
                "--ivar-input",
                str(ivar_input),
                "--prices",
                str(duplicate_prices),
                "--out",
                str(tmp_path / "duplicate_variance.csv"),
            ]
        )

    missing_exit = tmp_path / "missing_exit_ivar_input.csv"
    missing_exit.write_text(
        "\n".join(
            [
                "ticker,event_date,event_exit_date,expiration,iv,dte_days,stale",
                "ABC,2026-02-05,,2026-02-05,0.80,1,false",
                "ABC,2026-02-05,,2026-02-12,0.70,8,false",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="missing event_exit_date"):
        main(
            [
                "compute-variance",
                "--ivar-input",
                str(missing_exit),
                "--prices",
                str(prices),
                "--out",
                str(tmp_path / "missing_exit_variance.csv"),
            ]
        )


def test_compute_variance_reports_missing_price_row(tmp_path: Path) -> None:
    prices = tmp_path / "event_prices.csv"
    prices.write_text("ticker,event_date,s_before,s_after\nZZZ,2026-02-05,100,101\n")

    with pytest.raises(ValueError, match="missing event price row for ABC 2026-02-05"):
        main(
            [
                "compute-variance",
                "--ivar-input",
                str(FIXTURES / "ivar_input.csv"),
                "--prices",
                str(prices),
                "--out",
                str(tmp_path / "variance.csv"),
            ]
        )

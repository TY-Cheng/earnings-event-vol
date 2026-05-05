from __future__ import annotations

import gzip
import importlib
import json
import subprocess
from collections.abc import Mapping, Sequence
from dataclasses import replace
from datetime import date, datetime, time
from pathlib import Path
from typing import Any, cast

import httpx
import pandas as pd
import pytest

import scripts.build_trade_proxy_panel as trade_proxy_panel_script
from earnings_event_vol.backtest import (
    GaussianEventJumpDistribution,
    SymmetricTwoPointJumpDistribution,
    apply_portfolio_caps,
    black_scholes_price,
    estimated_transaction_cost_usd,
    expected_strategy_value_usd,
    integer_contract_count,
    market_entry_cost_usd,
    premium_space_signal,
)
from earnings_event_vol.cli import main
from earnings_event_vol.config import load_project_config
from earnings_event_vol.data_audit import audit_data_fields, vendor_local_iv_comparison
from earnings_event_vol.data_pipeline import parse_text_list, run_data_pipeline
from earnings_event_vol.earnings_calendar import (
    apply_text_validation,
    build_earnings_calendar_candidates,
    build_earnings_calendar_report,
    classify_8k_text,
    fetch_massive_8k_text_payloads,
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
from earnings_event_vol.events import (
    align_event_window,
    has_ex_dividend_between,
    is_halted_or_proxy_halted,
    market_close_timestamp,
    market_close_timestamp_utc,
    regular_close_timestamp,
    rvar_prices_for_window,
    validate_calendar_frame,
)
from earnings_event_vol.features import (
    has_required_sequence_history,
    iv_butterfly_25d,
    universe_by_trailing_option_dollar_volume,
)
from earnings_event_vol.leakage_audit import audit_feature_leakage, make_feature_timestamps
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
from earnings_event_vol.models import MODEL_REGISTRY, get_model_spec, unimplemented_model_message
from earnings_event_vol.schemas import (
    AnnouncementTiming,
    EarningsEvent,
    IVARFailureReason,
    OptionQuote,
    OptionRight,
    OptionSide,
    StrategyTrade,
    TimeConvention,
    TradeLeg,
    UnderlyingBar,
)
from earnings_event_vol.trade_proxy import (
    OPTION_EXIT_STATUS_MISSING_DAY_AGG,
    OPTION_EXIT_STATUS_OK,
    TRADE_PROXY_PANEL_GRADE,
    TRADE_PROXY_STATUS_NO_TRADE_IN_WINDOW,
    TRADE_PROXY_STATUS_OK,
    attach_trade_proxy_local_iv,
    build_proxy_straddle_diagnostics,
    build_trade_proxy_ivar_inputs,
    build_trade_proxy_price_frame,
    edge_decile_diagnostics,
    extract_trade_proxy_event_panel,
    fetch_massive_option_second_aggregates,
    filter_pre_cutoff_buffer,
    normalize_second_aggregates,
    select_latest_proxy_price,
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

    assert len(frame) == 4
    assert report["main_sample_candidate_rows"] == 2
    assert report["timing_counts"] == {"AMC": 1, "BMO": 2, "DMH": 1}
    assert (
        frame.loc[frame["source_id"] == "0001628280-26-022956", "text_validation_status"].iloc[0]
        == "non_earnings_item_2_02"
    )
    validated_calendar = validate_calendar_frame(
        frame[["ticker", "announcement_date", "announcement_timing", "source"]]
    )
    assert int(validated_calendar["is_main_sample_timing"].sum()) == 3


def test_earnings_calendar_http_fetch_path_uses_official_and_massive_sources(
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
    config = load_project_config()

    sec_payload = _sec_submissions_payload(
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

    def handler(request: httpx.Request) -> httpx.Response:
        if str(request.url) == "https://sec.example/tickers.json":
            return httpx.Response(200, json={"0": {"ticker": "AAPL", "cik_str": 320193}})
        if str(request.url) == "https://sec.example/submissions/CIK0000320193.json":
            return httpx.Response(200, json=sec_payload)
        if request.url.host == "massive.example":
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

    assert frame["announcement_timing"].tolist() == ["AMC"]
    assert frame["is_main_sample_candidate"].tolist() == [True]
    assert report["validation_route"] == "sec_edgar_http+massive_8k_text_http"


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
    assert bool(discovered["covers_event_window"].iloc[0]) is True


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

    status_cases = discover_option_contracts(
        pd.DataFrame(
            {
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

    calendar_out = out_root / "earnings_calendar_pilot"
    calendar_out.mkdir(parents=True)
    (calendar_out / "earnings_calendar_candidates.csv").write_text("ticker\nAAPL\n")
    (calendar_out / "earnings_calendar_report.json").write_text("{}")
    calendar_skip = run_data_pipeline(
        config,
        stage="calendar-pilot",
        out_root=out_root,
        tickers=["AAPL"],
        start_date=date(2026, 1, 1),
        end_date=date(2026, 12, 31),
    )
    assert _pipeline_steps(calendar_skip)[0]["status"] == "skipped"

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

    blocked = run_data_pipeline(config, stage="contracts", out_root=out_root)
    assert blocked["ok"] is False
    assert _pipeline_steps(blocked)[0]["status"] == "blocked"

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
    universe_blocked = run_data_pipeline(config, stage="universe", out_root=tmp_path / "blocked")
    assert universe_blocked["ok"] is False
    assert _pipeline_steps(universe_blocked)[0]["status"] == "blocked"

    with pytest.raises(ValueError, match="unsupported data stage"):
        run_data_pipeline(config, stage="bad-stage", out_root=out_root)
    with pytest.raises(ValueError, match="jobs must be positive"):
        run_data_pipeline(config, stage="fixture-audit", out_root=out_root, jobs=0)
    with pytest.raises(ValueError, match="start_date"):
        run_data_pipeline(
            config,
            stage="calendar-pilot",
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

    dry_run = run_data_pipeline(
        config,
        stage="proxy-all",
        out_root=out_root,
        tickers=["AAPL", "MSFT"],
        start_date=date(2026, 1, 1),
        end_date=date(2026, 3, 31),
        max_events=8,
        max_contracts=80,
        dry_run=True,
    )
    assert dry_run["dry_run"] is True
    assert dry_run["writes_data_outputs"] is False
    assert cast(dict[str, object], dry_run["estimated_counts"])["second_agg_rest_calls"] == 80
    assert "missing_option_day_agg_exit_price" in cast(
        dict[str, object], dry_run["exclusion_estimate"]
    )


def test_data_pipeline_contracts_and_panel_stages(tmp_path: Path) -> None:
    config = load_project_config()
    events = tmp_path / "events.csv"
    events.write_text(
        "\n".join(
            [
                "event_id,ticker,entry_date,exit_date,spot,rvar_event,ivar_event",
                "ABC_2026Q1,ABC,2026-02-05,2026-02-06,100,0.001,0.04",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    contracts = tmp_path / "contracts.csv"
    contracts.write_text(
        "\n".join(
            [
                "ticker,expiration,strike,right,option_multiplier,contract_size,deliverable_status",
                "ABC,2026-02-13,100,call,100,100,standard",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    quotes = tmp_path / "quotes.csv"
    quotes.write_text(
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

    out_root = tmp_path / "pipeline"
    contracts_result = run_data_pipeline(
        config,
        stage="contracts",
        out_root=out_root,
        events_path=events,
        contracts_path=contracts,
    )
    assert contracts_result["ok"] is True
    contract_output = out_root / "contracts" / "event_contract_candidates.csv"
    assert contract_output.exists()
    assert pd.read_csv(contract_output)["eligible_for_quote_pool"].sum() == 1

    panel_result = run_data_pipeline(
        config,
        stage="panel",
        out_root=out_root,
        events_path=events,
        quotes_path=quotes,
    )
    assert panel_result["ok"] is True
    panel_output = out_root / "event_panel" / "event_panel.csv"
    assert panel_output.exists()
    assert pd.read_csv(panel_output)["forward_source"].iloc[0] == FORWARD_SOURCE_PUT_CALL_PARITY


def test_data_pipeline_pilot_panel_stage_uses_lake_outputs_and_max_events(
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
        gold_output = config.gold_data_dir / "event_panel" / "pilot_event_panel.parquet"
        report = out_root / "event_panel" / "pilot_panel_report.json"
        gold_output.parent.mkdir(parents=True, exist_ok=True)
        report.parent.mkdir(parents=True, exist_ok=True)
        gold_output.write_text("parquet-placeholder", encoding="utf-8")
        report.write_text("{}", encoding="utf-8")
        return subprocess.CompletedProcess(command, 0, stdout="pilot ok", stderr="")

    monkeypatch.setattr("earnings_event_vol.data_pipeline._run_command_with_progress", fake_run)

    result = run_data_pipeline(
        config,
        stage="pilot-panel",
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
    assert str(config.gold_data_dir / "event_panel" / "pilot_event_panel.parquet") in outputs
    command = cast(list[str], captured["command"])
    assert command[-3:] == ["--force", "--max-events", "7"]
    assert "--dte-min" in command
    assert "--dte-max" in command
    assert captured["cwd"] == tmp_path
    assert captured["label"] == "pilot-panel"

    skipped = run_data_pipeline(config, stage="pilot-panel", out_root=out_root)
    assert _pipeline_steps(skipped)[0]["status"] == "skipped"


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
        report.write_text("{}", encoding="utf-8")
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
    assert "--max-events" in command
    assert "--max-contracts" in command
    assert captured["cwd"] == tmp_path
    assert captured["label"] == "trade-proxy-panel"

    skipped = run_data_pipeline(config, stage="trade-proxy-panel", out_root=out_root)
    assert _pipeline_steps(skipped)[0]["status"] == "skipped"


def test_data_pipeline_proxy_all_orchestrates_calendar_pilot_and_trade_proxy(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = replace(
        load_project_config(),
        repo_root=tmp_path,
        gold_data_dir=tmp_path / "data" / "gold",
    )
    out_root = tmp_path / "artifacts" / "data_pipeline"
    calendar_out = out_root / "earnings_calendar_pilot"
    calendar_out.mkdir(parents=True)
    (calendar_out / "earnings_calendar_candidates.csv").write_text("ticker\nAAPL\n")
    (calendar_out / "earnings_calendar_report.json").write_text("{}")
    captured_commands: list[list[str]] = []

    def fake_run(
        command: Sequence[str],
        *,
        cwd: Path,
        label: str,
    ) -> subprocess.CompletedProcess[str]:
        assert cwd == tmp_path
        captured_commands.append(list(command))
        if "build_pilot_panel.py" in command[1]:
            gold_output = config.gold_data_dir / "event_panel" / "pilot_event_panel.parquet"
            report = out_root / "event_panel" / "pilot_panel_report.json"
        else:
            gold_output = config.gold_data_dir / "event_panel" / "trade_proxy_event_panel.parquet"
            report = out_root / "trade_proxy_panel" / "trade_proxy_panel_report.json"
        gold_output.parent.mkdir(parents=True, exist_ok=True)
        report.parent.mkdir(parents=True, exist_ok=True)
        gold_output.write_text("parquet-placeholder", encoding="utf-8")
        report.write_text("{}", encoding="utf-8")
        return subprocess.CompletedProcess(command, 0, stdout="ok", stderr="")

    monkeypatch.setattr("earnings_event_vol.data_pipeline._run_command_with_progress", fake_run)

    result = run_data_pipeline(
        config,
        stage="proxy-all",
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
        "calendar-pilot",
        "pilot-panel",
        "trade-proxy-panel",
    ]
    assert [step["status"] for step in steps] == ["skipped", "ran", "ran"]
    assert len(captured_commands) == 2
    pilot_command, trade_command = captured_commands
    assert "build_pilot_panel.py" in pilot_command[1]
    assert "build_trade_proxy_panel.py" in trade_command[1]
    assert "--dte-min" in pilot_command
    assert "--dte-max" in pilot_command
    assert "--max-contracts" in trade_command


def test_data_pipeline_proxy_all_stops_after_blocked_upstream_stage(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = replace(load_project_config(), repo_root=tmp_path)
    out_root = tmp_path / "artifacts" / "data_pipeline"
    calendar_out = out_root / "earnings_calendar_pilot"
    calendar_out.mkdir(parents=True)
    (calendar_out / "earnings_calendar_candidates.csv").write_text("ticker\nAAPL\n")
    (calendar_out / "earnings_calendar_report.json").write_text("{}", encoding="utf-8")
    captured_commands: list[list[str]] = []

    def fake_run(
        command: Sequence[str],
        *,
        cwd: Path,
        label: str,
    ) -> subprocess.CompletedProcess[str]:
        captured_commands.append(list(command))
        return subprocess.CompletedProcess(command, 1, stdout="", stderr="pilot failed")

    monkeypatch.setattr("earnings_event_vol.data_pipeline._run_command_with_progress", fake_run)

    result = run_data_pipeline(
        config,
        stage="proxy-all",
        out_root=out_root,
        force=False,
    )

    steps = _pipeline_steps(result)
    assert result["ok"] is False
    assert [step["name"] for step in steps] == ["calendar-pilot", "pilot-panel"]
    assert [step["status"] for step in steps] == ["skipped", "blocked"]
    assert len(captured_commands) == 1
    assert "build_pilot_panel.py" in captured_commands[0][1]


def test_trade_proxy_second_aggregates_are_cached_in_bronze_parquet(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = replace(load_project_config(), bronze_data_dir=tmp_path / "data" / "bronze")
    contracts = pd.DataFrame(
        {
            "options_ticker": ["O:ABC260213C00100000"],
            "entry_date": ["2026-02-05"],
            "event_entry_timestamp": [pd.Timestamp("2026-02-05 16:00:00", tz="America/New_York")],
        }
    )
    fetch_calls = 0
    normalized = pd.DataFrame(
        {
            "options_ticker": ["O:ABC260213C00100000"],
            "timestamp_utc": [pd.Timestamp("2026-02-05 20:59:58Z")],
            "timestamp_et": [pd.Timestamp("2026-02-05 15:59:58", tz="America/New_York")],
            "option_open": [4.0],
            "option_high": [4.2],
            "option_low": [3.9],
            "option_close": [4.1],
            "option_vwap": [4.05],
            "volume": [3],
            "transactions": [2],
            "source_dataset": ["massive_rest_second_aggs"],
        }
    )

    def fake_fetch_one_contract(
        config: object,
        *,
        option_ticker: str,
        entry_date: pd.Timestamp,
        limit: int,
    ) -> tuple[str, pd.DataFrame, dict[str, object]]:
        nonlocal fetch_calls
        fetch_calls += 1
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
    assert fetch_calls == 1
    assert first_report["cache_status"].tolist() == ["written"]
    assert Path(first_report["bronze_path"].iloc[0]).exists()
    assert first_frames["O:ABC260213C00100000"]["option_vwap"].iloc[0] == pytest.approx(4.05)

    second_frames, second_report = trade_proxy_panel_script._fetch_second_aggregate_bars(
        config,
        contracts,
        jobs=1,
        limit=100,
        buffer_minutes=60,
        force=False,
    )
    assert fetch_calls == 1
    assert second_report["cache_status"].tolist() == ["hit"]
    assert second_frames["O:ABC260213C00100000"]["option_close"].iloc[0] == pytest.approx(4.1)

    Path(first_report["bronze_path"].iloc[0]).write_text("not a parquet file", encoding="utf-8")
    repaired_frames, repaired_report = trade_proxy_panel_script._fetch_second_aggregate_bars(
        config,
        contracts,
        jobs=1,
        limit=100,
        buffer_minutes=60,
        force=False,
    )
    assert fetch_calls == 2
    assert repaired_report["cache_status"].tolist() == ["repaired"]
    assert repaired_frames["O:ABC260213C00100000"]["option_close"].iloc[0] == pytest.approx(4.1)


def test_trade_proxy_exit_day_aggs_repair_corrupt_cache(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = replace(load_project_config(), bronze_data_dir=tmp_path / "data" / "bronze")
    day = pd.Timestamp("2026-02-06")
    destination = trade_proxy_panel_script._options_day_agg_path(config, day)
    destination.parent.mkdir(parents=True)
    destination.write_text("not a parquet file", encoding="utf-8")

    def fake_download_command(config: object, *, key: str, destination: Path) -> list[str]:
        return ["fake-download", str(destination)]

    def fake_aws_env(config: object) -> dict[str, str]:
        return {}

    def fake_run_download(command: Sequence[str], env: Mapping[str, str], timeout: float) -> object:
        with gzip.open(command[-1], "wt", encoding="utf-8") as handle:
            handle.write("ticker,close\nO:ABC260213C00100000,4.75\n")
        return subprocess.CompletedProcess(list(command), 0)

    monkeypatch.setattr(
        trade_proxy_panel_script,
        "build_download_file_command",
        fake_download_command,
    )
    monkeypatch.setattr(trade_proxy_panel_script, "massive_flat_file_aws_env", fake_aws_env)
    monkeypatch.setattr(trade_proxy_panel_script, "_run_head_object_command", fake_run_download)

    status = trade_proxy_panel_script._ensure_options_day_agg_file(config, day)

    assert status == "repaired"
    repaired = pd.read_parquet(destination)
    assert repaired["ticker"].tolist() == ["O:ABC260213C00100000"]
    assert repaired["close"].tolist() == pytest.approx([4.75])


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
    selected = select_latest_proxy_price(
        bars,
        cutoff_timestamp=pd.Timestamp("2026-02-05 16:00:00", tz="America/New_York").to_pydatetime(),
        lookback_seconds=30,
    )
    assert selected.status == TRADE_PROXY_STATUS_OK
    assert selected.proxy_price == pytest.approx(4.2)
    assert selected.proxy_volume == 3
    assert selected.proxy_rows_in_window == 2

    stale = select_latest_proxy_price(
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
    assert early_close.hour == 13
    assert (
        market_close_timestamp_utc(
            date(2026, 11, 27), early_closes={date(2026, 11, 27): time(13)}
        ).tzinfo
        is not None
    )


def test_trade_proxy_validation_branches_and_summaries(tmp_path: Path) -> None:
    empty_normalized = normalize_second_aggregates(pd.DataFrame(), option_ticker="O:ABC")
    assert list(empty_normalized.columns)[0] == "options_ticker"
    with pytest.raises(ValueError, match="missing required columns"):
        normalize_second_aggregates(pd.DataFrame({"t": [1]}), option_ticker="O:ABC")
    with pytest.raises(ValueError, match="lookback_seconds"):
        select_latest_proxy_price(
            pd.DataFrame(), cutoff_timestamp=datetime(2026, 2, 5), lookback_seconds=0
        )
    with pytest.raises(ValueError, match="price_field"):
        select_latest_proxy_price(
            pd.DataFrame(), cutoff_timestamp=datetime(2026, 2, 5), price_field="bad"
        )
    with pytest.raises(ValueError, match="timestamp_et"):
        select_latest_proxy_price(
            pd.DataFrame({"option_vwap": [1.0]}),
            cutoff_timestamp=datetime(2026, 2, 5),
        )
    assert (
        select_latest_proxy_price(
            pd.DataFrame(),
            cutoff_timestamp=datetime(2026, 2, 5),
        ).status
        == TRADE_PROXY_STATUS_NO_TRADE_IN_WINDOW
    )
    with pytest.raises(ValueError, match="timezone-aware"):
        select_latest_proxy_price(
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

    pilot_panel_script = cast(Any, importlib.import_module("scripts.build_pilot_panel"))
    pilot_no_inputs = pilot_panel_script._extract_event_ivar(
        pd.DataFrame(),
        pd.DataFrame(
            {
                "event_id": ["ABC_2026Q1"],
                "ticker": ["ABC"],
                "announcement_date": [date(2026, 2, 5)],
                "exit_date": [date(2026, 2, 6)],
                "s_before": [100.0],
                "rvar_event": [0.01],
            }
        ),
    )
    assert pilot_no_inputs["ivar_event"].isna().all()
    assert pilot_no_inputs["ivar_failure_reason"].iloc[0] == (
        IVARFailureReason.NO_TWO_EVENT_COVERING_EXPIRIES.value
    )

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

    exit_close_straddle = build_proxy_straddle_diagnostics(
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
        option_exit_prices=pd.DataFrame(
            {
                "options_ticker": ["O:ABC260220C00100000", "O:ABC260220P00100000"],
                "date": [date(2026, 2, 6), date(2026, 2, 6)],
                "option_close": [6.0, 3.5],
            }
        ),
    )
    assert exit_close_straddle["option_exit_price_status"].iloc[0] == OPTION_EXIT_STATUS_OK
    assert bool(exit_close_straddle["used_intrinsic_fallback"].iloc[0]) is False
    assert exit_close_straddle["exit_option_value_usd"].iloc[0] == pytest.approx(950.0)
    assert exit_close_straddle["gross_proxy_pnl_usd"].iloc[0] == pytest.approx(50.0)

    missing_exit_close = build_proxy_straddle_diagnostics(
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
        option_exit_prices=pd.DataFrame(),
    )
    assert (
        missing_exit_close["option_exit_price_status"].iloc[0] == OPTION_EXIT_STATUS_MISSING_DAY_AGG
    )
    assert bool(missing_exit_close["used_intrinsic_fallback"].iloc[0]) is True

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
    assert straddles["option_exit_price_status"].iloc[0] == OPTION_EXIT_STATUS_MISSING_DAY_AGG
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
        }
    )
    result = audit_feature_leakage(frame)
    assert result.ok is False
    assert len(result.asof_violations) == 1
    assert "vendor_alpha_forecast" in result.vendor_forecast_columns
    assert "same_event_return" in result.blocked_columns


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
    assert "not a trainable model" in patell_text
    goyal_text = MODEL_REGISTRY["goyal_saretto_rv_iv_spread"].justification
    assert "RV-IV spread feature/baseline" in goyal_text
    assert "not a full replication" in goyal_text
    assert get_model_spec("market_implied_event_variance").implemented is True
    assert get_model_spec("last_four_rvar").implemented is False
    assert get_model_spec("last_four_ivar").implemented is False
    assert get_model_spec("patell_wolfson_diagnostic").implemented is False
    assert "implemented as a deterministic baseline" in unimplemented_model_message(
        "market_implied_event_variance"
    )
    assert "not implemented in v1" in unimplemented_model_message("mamba_sequence_encoder")


def test_patell_wolfson_registry_text() -> None:
    spec = MODEL_REGISTRY["patell_wolfson_diagnostic"]

    assert spec.role == "diagnostic"
    assert spec.implemented is False
    assert "diagnostic features" in spec.justification
    assert "pre-event implied-volatility behavior" in spec.justification
    assert "realized earnings move history" in spec.justification
    assert "post-event volatility compression diagnostics" in spec.justification


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


def test_cli_smoke_commands(tmp_path: Path) -> None:
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
                "--out-root",
                str(data_out),
            ]
        )
        == 0
    )
    assert (data_out / "data_pipeline_manifest.json").exists()

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


def test_compute_variance_uses_event_exit_date_and_rejects_duplicates(tmp_path: Path) -> None:
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

from __future__ import annotations

import json
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from earnings_event_vol import quote_execution as quote_execution_module
from earnings_event_vol.cli import main
from earnings_event_vol.config import load_project_config
from earnings_event_vol.features import FEATURE_SCHEMA_V2_SEC_XBRL, build_feature_schema_report
from earnings_event_vol.quote_execution import (
    QUOTE_STATUS_MISSING,
    QUOTE_STATUS_OK,
    QUOTE_STATUS_WIDE,
    build_execution_confidence_panel,
    build_quote_window_marks,
    build_quote_window_requests,
    extract_quote_execution_panel,
    normalize_option_quote_rows,
    quote_confidence_band,
)
from earnings_event_vol.research import build_metric_tables


def _quote_contracts() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "event_id": ["ABC_2026Q1", "ABC_2026Q1"],
            "ticker": ["ABC", "ABC"],
            "entry_date": [date(2026, 2, 5), date(2026, 2, 5)],
            "exit_date": [date(2026, 2, 6), date(2026, 2, 6)],
            "expiration": [date(2026, 2, 13), date(2026, 2, 13)],
            "strike": [100.0, 100.0],
            "right": ["call", "put"],
            "options_ticker": ["O:ABC260213C00100000", "O:ABC260213P00100000"],
            "eligible_for_quote_pool": [True, True],
        }
    )


def _quote_windows() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "event_id": ["ABC_2026Q1"],
            "ticker": ["ABC"],
            "announcement_date": [date(2026, 2, 5)],
            "announcement_timing": ["AMC"],
            "event_entry_timestamp": [pd.Timestamp("2026-02-05 16:00:00", tz="America/New_York")],
        }
    )


def test_quote_window_marks_and_execution_confidence_bands() -> None:
    requests = build_quote_window_requests(_quote_contracts(), _quote_windows())
    raw_quotes = pd.DataFrame(
        {
            "ticker": [
                "O:ABC260213C00100000",
                "O:ABC260213P00100000",
                "O:ABC260213C00100000",
                "O:ABC260213P00100000",
            ],
            "bid_price": [5.0, 4.8, 7.0, 1.0],
            "ask_price": [5.2, 5.0, 7.2, 3.0],
            "quote_timestamp_et": [
                "2026-02-05 15:59:50-05:00",
                "2026-02-05 15:59:45-05:00",
                "2026-02-06 15:59:40-05:00",
                "2026-02-06 15:59:40-05:00",
            ],
        }
    )
    quotes = normalize_option_quote_rows(raw_quotes)
    marks = build_quote_window_marks(quotes, requests, wide_spread_threshold=0.25)
    confidence = build_execution_confidence_panel(marks)

    assert set(marks["quote_status"]) == {QUOTE_STATUS_OK, QUOTE_STATUS_WIDE}
    assert len(marks) == 4
    assert confidence["execution_confidence_band"].iloc[0] == "high"
    assert confidence["execution_confidence_score"].iloc[0] == 0.875


def test_quote_normalization_and_timestamp_branches() -> None:
    assert normalize_option_quote_rows(pd.DataFrame()).empty
    with pytest.raises(ValueError, match="quote frame missing required columns"):
        normalize_option_quote_rows(pd.DataFrame({"ticker": ["O:ABC260213C00100000"]}))

    assert pd.isna(quote_execution_module._timestamp_from_sip(None))
    assert pd.isna(quote_execution_module._timestamp_from_sip(""))
    assert pd.isna(quote_execution_module._timestamp_from_sip(np.nan))
    assert quote_execution_module._timestamp_from_sip(1_770_316_790_000_000_000).year == 2026
    assert quote_execution_module._timestamp_from_sip(1_770_316_790_000_000).year == 2026
    assert quote_execution_module._timestamp_from_sip(1_770_316_790_000).year == 2026
    assert quote_execution_module._timestamp_from_sip(1_770_316_790).year == 2026
    assert quote_execution_module._timestamp_from_sip("2026-02-05 21:00:00").year == 2026

    raw = pd.DataFrame(
        {
            "options_ticker": ["O:ABC260213C00100000"],
            "bid": [5.0],
            "ask": [5.2],
            "timestamp": ["2026-02-05 15:59:50-05:00"],
            "quote_date": ["2026-02-05"],
        }
    )
    normalized = normalize_option_quote_rows(raw)
    assert normalized["quote_date"].iloc[0] == date(2026, 2, 5)
    assert normalized["mid"].iloc[0] == 5.1


def test_quote_request_validation_empty_and_max_events() -> None:
    with pytest.raises(ValueError, match="contract frame missing required columns"):
        build_quote_window_requests(pd.DataFrame({"event_id": ["x"]}), _quote_windows())
    with pytest.raises(ValueError, match="window frame missing required columns"):
        build_quote_window_requests(_quote_contracts(), pd.DataFrame({"ticker": ["ABC"]}))

    ineligible = _quote_contracts().assign(eligible_for_quote_pool=False)
    assert build_quote_window_requests(ineligible, _quote_windows()).empty

    windows = pd.concat(
        [
            _quote_windows(),
            _quote_windows().assign(event_id="XYZ_2026Q1", ticker="XYZ"),
        ],
        ignore_index=True,
    )
    contracts = pd.concat(
        [
            _quote_contracts(),
            _quote_contracts().assign(
                event_id="XYZ_2026Q1",
                ticker="XYZ",
                options_ticker=["O:XYZ260213C00100000", "O:XYZ260213P00100000"],
            ),
        ],
        ignore_index=True,
    )
    requests = build_quote_window_requests(contracts, windows, max_events=1)
    assert set(requests["event_id"]) == {"ABC_2026Q1"}


def test_quote_mark_edge_statuses_and_empty_confidence() -> None:
    assert build_quote_window_marks(pd.DataFrame(), pd.DataFrame()).empty
    assert build_execution_confidence_panel(pd.DataFrame()).empty
    assert quote_confidence_band(None) == "missing"
    assert quote_confidence_band(0.25) == "low"
    assert quote_confidence_band(0.5) == "medium"

    request = build_quote_window_requests(_quote_contracts().head(1), _quote_windows()).head(1)
    invalid_quotes = normalize_option_quote_rows(
        pd.DataFrame(
            {
                "ticker": ["O:ABC260213C00100000"],
                "bid_price": [5.5],
                "ask_price": [5.0],
                "quote_timestamp_et": ["2026-02-05 15:59:50-05:00"],
            }
        )
    )
    invalid_marks = build_quote_window_marks(invalid_quotes, request)
    assert invalid_marks["quote_status"].iloc[0] == "invalid_bid_ask"

    stale_quotes = normalize_option_quote_rows(
        pd.DataFrame(
            {
                "ticker": ["O:ABC260213C00100000"],
                "bid_price": [5.0],
                "ask_price": [5.2],
                "quote_timestamp_et": ["2026-02-05 15:58:00-05:00"],
            }
        )
    )
    stale_marks = build_quote_window_marks(stale_quotes, request, stale_seconds=60)
    assert stale_marks["quote_status"].iloc[0] == "stale_quote"


def test_quote_execution_panel_cli_uses_local_csv_without_raw_full_day_file(
    tmp_path: Path,
) -> None:
    contracts_path = tmp_path / "contracts.parquet"
    windows_path = tmp_path / "windows.parquet"
    quotes_path = tmp_path / "quotes.csv"
    out_dir = tmp_path / "quote_execution"
    _quote_contracts().to_parquet(contracts_path, index=False)
    _quote_windows().to_parquet(windows_path, index=False)
    pd.DataFrame(
        {
            "ticker": ["O:ABC260213C00100000"],
            "bid_price": [5.0],
            "ask_price": [5.2],
            "quote_timestamp_et": ["2026-02-05 15:59:50-05:00"],
        }
    ).to_csv(quotes_path, index=False)

    assert (
        main(
            [
                "quote-execution-panel",
                "--contracts",
                str(contracts_path),
                "--windows",
                str(windows_path),
                "--quotes-csv",
                str(quotes_path),
                "--out",
                str(out_dir),
            ]
        )
        == 0
    )

    report = json.loads((out_dir / "quote_execution_report.json").read_text())
    marks = pd.read_csv(out_dir / "quote_window_marks.csv")
    assert report["raw_full_day_files_written"] is False
    assert report["quote_rows_scanned"] == 1
    assert QUOTE_STATUS_MISSING in set(marks["quote_status"])


def test_quote_execution_metadata_only_writes_empty_outputs(tmp_path: Path) -> None:
    report = extract_quote_execution_panel(
        config=load_project_config(),
        contracts=_quote_contracts(),
        windows=_quote_windows(),
        out_dir=tmp_path,
        metadata_only=True,
    )
    assert report.metadata_only is True
    assert (tmp_path / "quote_window_requests.csv").exists()
    assert (tmp_path / "quote_execution_report.json").exists()


def test_quote_execution_date_filter_can_return_no_requests(tmp_path: Path) -> None:
    quotes_path = tmp_path / "quotes.csv"
    pd.DataFrame(
        {
            "ticker": ["O:ABC260213C00100000"],
            "bid_price": [5.0],
            "ask_price": [5.2],
            "quote_timestamp_et": ["2026-02-05 15:59:50-05:00"],
        }
    ).to_csv(quotes_path, index=False)
    report = extract_quote_execution_panel(
        config=load_project_config(),
        contracts=_quote_contracts(),
        windows=_quote_windows(),
        out_dir=tmp_path / "date_filtered",
        quote_csv_paths=[quotes_path],
        dates=[date(2030, 1, 1)],
    )
    assert report.request_rows == 0
    assert report.quote_rows_scanned == 1
    assert report.quote_rows_matched == 0


def test_ivar_defeat_and_casebook_artifacts_from_metric_tables(tmp_path: Path) -> None:
    predictions = pd.DataFrame(
        {
            "split": ["test", "test", "test"],
            "target_id": ["day_c2c", "day_c2c", "day_c2c"],
            "event_id": ["E1", "E2", "E3"],
            "ticker": ["AAA", "BBB", "CCC"],
            "event_date": ["2026-01-01", "2026-01-02", "2026-01-03"],
            "announcement_date": ["2026-01-01", "2026-01-02", "2026-01-03"],
            "announcement_timing": ["AMC", "BMO", "AMC"],
            "rvar_event": [0.08, 0.03, 0.07],
            "ivar_event": [0.04, 0.04, 0.04],
            "edge_var_realized": [0.04, -0.01, 0.03],
            "forecast_lightgbm_tuned": [0.07, 0.06, 0.03],
            "execution_confidence_score": [0.9, 0.4, 0.9],
            "execution_confidence_band": ["high", "low", "high"],
        }
    )

    paths = build_metric_tables(predictions, out_dir=tmp_path)
    defeat = pd.read_csv(paths["ivar_defeat_events"])
    casebook = pd.read_csv(paths["casebook_events"])
    metrics = pd.read_csv(paths["ivar_defeat_metrics"])

    assert {"false_positive", "false_negative", "model_corrected_market"}.issubset(
        set(casebook["case_type"])
    )
    assert "execution_fragile" in set(casebook["case_type"])
    assert bool(defeat.loc[defeat["event_id"].eq("E1"), "model_corrected_market"].iloc[0])
    assert metrics["model_beats_ivar_abs_rate"].iloc[0] > 0


def test_execution_confidence_fields_are_not_model_features() -> None:
    frame = pd.DataFrame(
        {
            "event_id": ["E1", "E2"],
            "rvar_event": [0.01, 0.02],
            "ivar_event": [0.015, 0.015],
            "execution_confidence_score": [1.0, 0.0],
            "execution_confidence_band": ["high", "missing"],
            "quote_execution_route": ["massive_quotes_v1_flat_file_filtered"] * 2,
            "spread_over_mid": [0.02, 0.50],
            "plain_pre_event_feature": [1.0, 2.0],
        }
    )
    schema = build_feature_schema_report(frame, feature_schema_version=FEATURE_SCHEMA_V2_SEC_XBRL)
    model_features = set(schema.loc[schema["model_feature"], "feature_name"])
    assert "plain_pre_event_feature" in model_features
    assert "execution_confidence_score" not in model_features
    assert "execution_confidence_band" not in model_features
    assert "quote_execution_route" not in model_features
    assert "spread_over_mid" not in model_features

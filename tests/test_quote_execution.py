from __future__ import annotations

import json
from dataclasses import replace
from datetime import date
from pathlib import Path
from typing import cast

import httpx
import numpy as np
import pandas as pd
import polars as pl
import pytest

from earnings_event_vol import data_pipeline as data_pipeline_module
from earnings_event_vol import quote_execution as quote_execution_module
from earnings_event_vol.cli import main
from earnings_event_vol.config import load_project_config
from earnings_event_vol.data_pipeline import run_data_pipeline
from earnings_event_vol.features import FEATURE_SCHEMA_V2_SEC_XBRL, build_feature_schema_report
from earnings_event_vol.quote_execution import (
    QUOTE_STATUS_INVALID,
    QUOTE_STATUS_MISSING,
    QUOTE_STATUS_OK,
    QUOTE_STATUS_STALE,
    QUOTE_STATUS_WIDE,
    build_execution_confidence_panel,
    build_quote_execution_leg_panel,
    build_quote_iv_surface_panel,
    build_quote_iv_surface_summary_panel,
    build_quote_ivar_event_panel,
    build_quote_straddle_execution_panel,
    build_quote_surface_ivar_event_panel,
    build_quote_window_marks,
    build_quote_window_requests,
    extract_quote_execution_panel,
    normalize_option_quote_rows,
    quote_confidence_band,
)
from earnings_event_vol.research import build_metric_tables, merge_quote_execution_diagnostics
from earnings_event_vol.schemas import IVARFailureReason


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


def _two_event_quote_contracts_and_windows() -> tuple[pd.DataFrame, pd.DataFrame]:
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
    return contracts, windows


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
    legs = build_quote_execution_leg_panel(marks)
    straddles = build_quote_straddle_execution_panel(marks)
    assert set(legs["execution_side"]) == {"buy_to_open", "sell_to_close"}
    assert straddles["entry_mid_cost_usd"].iloc[0] == pytest.approx(1000.0)
    assert straddles["quote_mid_pnl_usd"].iloc[0] == pytest.approx(-90.0)


def test_quote_iv_surface_and_surface_ivar_diagnostics() -> None:
    requests = build_quote_window_requests(_quote_contracts(), _quote_windows())
    raw_quotes = pd.DataFrame(
        {
            "ticker": [
                "O:ABC260213C00100000",
                "O:ABC260213P00100000",
                "O:ABC260213C00100000",
                "O:ABC260213P00100000",
            ],
            "bid_price": [2.0, 2.1, 2.2, 2.3],
            "ask_price": [2.2, 2.3, 2.4, 2.5],
            "quote_timestamp_et": [
                "2026-02-05 15:59:50-05:00",
                "2026-02-05 15:59:45-05:00",
                "2026-02-06 15:59:40-05:00",
                "2026-02-06 15:59:40-05:00",
            ],
        }
    )
    marks = build_quote_window_marks(
        normalize_option_quote_rows(raw_quotes),
        requests,
        wide_spread_threshold=0.25,
    )
    marks["s_before"] = 100.0

    surface = build_quote_iv_surface_panel(marks)
    summary = build_quote_iv_surface_summary_panel(surface)
    manual_summary = pd.DataFrame(
        {
            "event_id": ["ABC_2026Q1", "ABC_2026Q1"],
            "ticker": ["ABC", "ABC"],
            "announcement_date": [date(2026, 2, 5), date(2026, 2, 5)],
            "announcement_timing": ["AMC", "AMC"],
            "entry_date": [date(2026, 2, 5), date(2026, 2, 5)],
            "expiration": [date(2026, 2, 13), date(2026, 2, 20)],
            "strike": [100.0, 100.0],
            "spot": [100.0, 100.0],
            "quote_mid_total_variance": [0.01, 0.015],
            "quote_bid_total_variance": [0.009, 0.014],
            "quote_ask_total_variance": [0.011, 0.016],
            "complete_mid_surface_pair": [True, True],
            "complete_bidask_surface_pair": [True, True],
        }
    )
    surface_ivar = build_quote_surface_ivar_event_panel(manual_summary)

    assert len(surface) == 2
    assert set(surface["spot"]) == {100.0}
    assert surface["quote_mid_iv"].notna().all()
    assert surface["paper_grade_quote_iv_mid"].all()
    assert len(summary) == 1
    assert summary["mean_mid_iv"].iloc[0] > 0
    assert bool(summary["paper_grade_quote_iv_surface_pair"].iloc[0])
    assert surface_ivar["quote_surface_mid_ivar_event"].iloc[0] == pytest.approx(
        ((15 * 0.01) - (8 * 0.015)) / (15 - 8)
    )
    assert surface_ivar["quote_surface_mid_ivar_failure_reason"].isna().all()
    assert bool(surface_ivar["paper_grade_quote_surface_ivar_mid"].iloc[0])


def test_quote_iv_solver_and_surface_failure_branches(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    implied_vol = quote_execution_module._implied_vol_from_quote
    _, reason = implied_vol(
        spot=0.0,
        strike=100.0,
        time_to_expiry=0.1,
        option_price=2.0,
        right="call",
    )
    assert reason == "nonpositive_spot_or_strike"
    _, reason = implied_vol(
        spot=100.0,
        strike=100.0,
        time_to_expiry=0.0,
        option_price=2.0,
        right="call",
    )
    assert reason == "nonpositive_time_to_expiry"
    _, reason = implied_vol(
        spot=100.0,
        strike=100.0,
        time_to_expiry=0.1,
        option_price=0.0,
        right="call",
    )
    assert reason == "missing_or_nonpositive_option_price"
    _, reason = implied_vol(
        spot=100.0,
        strike=90.0,
        time_to_expiry=0.1,
        option_price=5.0,
        right="call",
    )
    assert reason == "option_price_below_intrinsic"
    _, reason = implied_vol(
        spot=100.0,
        strike=100.0,
        time_to_expiry=0.1,
        option_price=1_000.0,
        right="put",
    )
    assert reason == "implied_vol_root_not_bracketed"

    def broken_brentq(*args: object, **kwargs: object) -> float:
        _ = args, kwargs
        raise ValueError("solver failed")

    monkeypatch.setattr(quote_execution_module, "brentq", broken_brentq)
    _, reason = implied_vol(
        spot=100.0,
        strike=100.0,
        time_to_expiry=0.1,
        option_price=2.0,
        right="call",
    )
    assert reason == "implied_vol_solver_failed"

    with pytest.raises(ValueError, match="quote marks missing required columns"):
        build_quote_iv_surface_panel(pd.DataFrame({"event_id": ["E1"]}))
    assert build_quote_iv_surface_panel(
        pd.DataFrame(
            {
                "event_id": ["E1"],
                "expiration": [date(2026, 2, 13)],
                "strike": [100.0],
                "right": ["call"],
                "window_label": ["exit_preclose_15m"],
                "spot": [100.0],
                "mid": [2.0],
            }
        )
    ).empty


def test_quote_iv_surface_status_and_summary_failure_branches() -> None:
    marks = pd.DataFrame(
        {
            "event_id": ["invalid", "stale"],
            "ticker": ["ABC", "ABC"],
            "announcement_date": [date(2026, 2, 5), date(2026, 2, 5)],
            "announcement_timing": ["AMC", "AMC"],
            "entry_date": [date(2026, 2, 5), date(2026, 2, 5)],
            "expiration": [date(2026, 2, 13), date(2026, 2, 13)],
            "strike": [100.0, 100.0],
            "right": ["call", "call"],
            "options_ticker": ["O:ABC260213C00100000", "O:ABC260213C00100000"],
            "window_label": ["entry_preclose_15m", "entry_preclose_15m"],
            "spot": [np.nan, np.nan],
            "s_before": [100.0, np.nan],
            "close_before": [np.nan, 100.0],
            "quote_status": [QUOTE_STATUS_INVALID, QUOTE_STATUS_STALE],
            "quote_timestamp_et": [
                "2026-02-05 15:59:50-05:00",
                "2026-02-05 15:58:00-05:00",
            ],
            "quote_age_seconds": [10.0, 120.0],
            "bid": [2.0, 2.0],
            "ask": [2.2, 2.2],
            "mid": [2.1, 2.1],
            "spread_over_mid": [0.05, 0.05],
        }
    )
    surface = build_quote_iv_surface_panel(marks).set_index("event_id")

    assert surface.loc["invalid", "spot"] == pytest.approx(100.0)
    assert surface.loc["stale", "spot"] == pytest.approx(100.0)
    assert surface.loc["invalid", "quote_mid_iv_failure_reason"] == QUOTE_STATUS_INVALID
    assert surface.loc["stale", "quote_mid_iv_failure_reason"] == QUOTE_STATUS_STALE
    assert not bool(surface.loc["invalid", "paper_grade_quote_iv_mid"])
    assert not bool(surface.loc["stale", "paper_grade_quote_iv_mid"])

    with pytest.raises(ValueError, match="quote IV surface frame missing required columns"):
        build_quote_iv_surface_summary_panel(pd.DataFrame({"event_id": ["E1"]}))
    single_right_summary = build_quote_iv_surface_summary_panel(surface.reset_index())
    assert not bool(single_right_summary["complete_mid_surface_pair"].iloc[0])


def test_quote_surface_ivar_failure_branches() -> None:
    with pytest.raises(ValueError, match="quote IV surface summary missing required columns"):
        build_quote_surface_ivar_event_panel(pd.DataFrame({"event_id": ["E1"]}))

    base_row = {
        "event_id": "E1",
        "ticker": "ABC",
        "announcement_date": date(2026, 2, 5),
        "announcement_timing": "AMC",
        "entry_date": date(2026, 2, 5),
        "strike": 100.0,
        "spot": 100.0,
        "quote_mid_total_variance": 0.01,
        "quote_bid_total_variance": 0.009,
        "quote_ask_total_variance": 0.011,
        "complete_mid_surface_pair": True,
        "complete_bidask_surface_pair": True,
    }
    one_expiry = build_quote_surface_ivar_event_panel(
        pd.DataFrame([{**base_row, "expiration": date(2026, 2, 13)}])
    )
    assert one_expiry["quote_surface_mid_ivar_failure_reason"].iloc[0] == (
        IVARFailureReason.NO_TWO_EVENT_COVERING_EXPIRIES.value
    )

    nonpositive_time = build_quote_surface_ivar_event_panel(
        pd.DataFrame(
            [
                {**base_row, "entry_date": date(2026, 2, 13), "expiration": date(2026, 2, 13)},
                {
                    **base_row,
                    "entry_date": date(2026, 2, 13),
                    "expiration": date(2026, 2, 20),
                    "quote_mid_total_variance": 0.015,
                    "quote_bid_total_variance": 0.014,
                    "quote_ask_total_variance": 0.016,
                },
            ]
        )
    )
    assert nonpositive_time["quote_surface_mid_ivar_failure_reason"].iloc[0] == (
        IVARFailureReason.NONPOSITIVE_TIME_GAP.value
    )
    assert not bool(nonpositive_time["paper_grade_quote_surface_ivar_mid"].iloc[0])


def test_quote_ivar_event_panel_uses_two_expiry_premium_variance_proxy() -> None:
    straddles = pd.DataFrame(
        {
            "event_id": ["ABC_2026Q1", "ABC_2026Q1"],
            "ticker": ["ABC", "ABC"],
            "announcement_date": [date(2026, 2, 5), date(2026, 2, 5)],
            "announcement_timing": ["AMC", "AMC"],
            "entry_date": [date(2026, 2, 5), date(2026, 2, 5)],
            "exit_date": [date(2026, 2, 6), date(2026, 2, 6)],
            "expiration": [date(2026, 2, 13), date(2026, 2, 20)],
            "strike": [100.0, 100.0],
            "spot": [100.0, 100.0],
            "quote_premium_var_mid": [0.01, 0.015],
            "quote_premium_var_ask": [0.012, 0.018],
            "complete_mid_pair": [True, True],
            "complete_bidask_pair": [True, True],
        }
    )
    panel = build_quote_ivar_event_panel(straddles)

    assert len(panel) == 1
    assert panel["quote_mid_ivar_event"].iloc[0] == pytest.approx(
        ((15 * 0.01) - (8 * 0.015)) / (15 - 8)
    )
    assert panel["quote_mid_ivar_failure_reason"].isna().all()
    assert bool(panel["paper_grade_quote_ivar_mid"].iloc[0])
    assert panel["quote_ivar_claim_scope"].iloc[0] == (
        "diagnostic_quote_premium_proxy_not_model_feature"
    )


def test_quote_ivar_event_panel_failure_reasons() -> None:
    with pytest.raises(ValueError, match="quote straddle frame missing required columns"):
        build_quote_ivar_event_panel(pd.DataFrame({"event_id": ["missing"]}))

    def rows(
        event_id: str,
        *,
        expirations: tuple[date, date] = (date(2026, 2, 13), date(2026, 2, 20)),
        mid_vars: tuple[float, float] = (0.01, 0.015),
        ask_vars: tuple[float, float] = (0.012, 0.018),
        complete_mid: tuple[bool, bool] = (True, True),
        complete_bidask: tuple[bool, bool] = (True, True),
        entry_date: date = date(2026, 2, 5),
    ) -> list[dict[str, object]]:
        return [
            {
                "event_id": event_id,
                "ticker": "ABC",
                "announcement_date": date(2026, 2, 5),
                "announcement_timing": "AMC",
                "entry_date": entry_date,
                "exit_date": date(2026, 2, 6),
                "expiration": expiration,
                "strike": 100.0,
                "spot": 100.0,
                "quote_premium_var_mid": mid_var,
                "quote_premium_var_ask": ask_var,
                "complete_mid_pair": mid_complete,
                "complete_bidask_pair": bidask_complete,
            }
            for expiration, mid_var, ask_var, mid_complete, bidask_complete in zip(
                expirations,
                mid_vars,
                ask_vars,
                complete_mid,
                complete_bidask,
                strict=True,
            )
        ]

    frame = pd.DataFrame(
        [
            *rows("stale", complete_mid=(False, True)),
            *rows(
                "nonpositive_time",
                expirations=(date(2026, 2, 6), date(2026, 2, 20)),
                entry_date=date(2026, 2, 13),
            ),
            *rows("nonpositive_variance", mid_vars=(0.0, 0.015), ask_vars=(0.0, 0.018)),
            *rows("nonmonotone", mid_vars=(0.02, 0.015), ask_vars=(0.03, 0.018)),
            *rows("negative_extracted", mid_vars=(0.01, 0.1), ask_vars=(0.012, 0.12)),
            *rows("missing_second")[:1],
        ]
    )
    panel = build_quote_ivar_event_panel(frame).set_index("event_id")

    assert panel.loc["stale", "quote_mid_ivar_failure_reason"] == (
        IVARFailureReason.STALE_OR_MISSING_IV.value
    )
    assert panel.loc["nonpositive_time", "quote_mid_ivar_failure_reason"] == (
        IVARFailureReason.NONPOSITIVE_TIME_GAP.value
    )
    assert panel.loc["nonpositive_variance", "quote_mid_ivar_failure_reason"] == (
        IVARFailureReason.NONPOSITIVE_TOTAL_VARIANCE.value
    )
    assert panel.loc["nonmonotone", "quote_mid_ivar_failure_reason"] == (
        IVARFailureReason.NONMONOTONE_TOTAL_VARIANCE.value
    )
    assert panel.loc["negative_extracted", "quote_mid_ivar_failure_reason"] == (
        IVARFailureReason.NEGATIVE_EXTRACTED_IVAR.value
    )
    assert panel.loc["missing_second", "quote_mid_ivar_failure_reason"] == (
        IVARFailureReason.NO_TWO_EVENT_COVERING_EXPIRIES.value
    )
    assert not bool(panel.loc["negative_extracted", "paper_grade_quote_ivar_mid"])


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
    assert pd.isna(quote_execution_module._timestamp_from_sip(float("inf")))

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

    sip_normalized = normalize_option_quote_rows(
        pd.DataFrame(
            {
                "ticker": ["O:ABC260213C00100000"],
                "bid_price": [5.0],
                "ask_price": [5.2],
                "sip_timestamp": [1_770_316_790_000_000_000],
            }
        ),
        quote_date=date(2026, 2, 5),
    )
    assert sip_normalized["quote_date"].iloc[0] == date(2026, 2, 5)


def test_quote_request_validation_empty_and_max_events() -> None:
    with pytest.raises(ValueError, match="contract frame missing required columns"):
        build_quote_window_requests(pd.DataFrame({"event_id": ["x"]}), _quote_windows())
    with pytest.raises(ValueError, match="window frame missing required columns"):
        build_quote_window_requests(_quote_contracts(), pd.DataFrame({"ticker": ["ABC"]}))

    ineligible = _quote_contracts().assign(eligible_for_quote_pool=False)
    assert build_quote_window_requests(ineligible, _quote_windows()).empty

    contracts, windows = _two_event_quote_contracts_and_windows()
    requests = build_quote_window_requests(contracts, windows, max_events=1)
    assert set(requests["event_id"]) == {"ABC_2026Q1"}
    requests_without_eligibility = build_quote_window_requests(
        contracts.drop(columns=["eligible_for_quote_pool"]), windows, max_events=1
    )
    assert set(requests_without_eligibility["event_id"]) == {"ABC_2026Q1"}
    offset_requests = build_quote_window_requests(contracts, windows, max_events=1, event_offset=1)
    assert set(offset_requests["event_id"]) == {"XYZ_2026Q1"}
    assert build_quote_window_requests(contracts, windows, event_offset=99).empty
    with pytest.raises(ValueError, match="event_offset"):
        build_quote_window_requests(contracts, windows, event_offset=-1)


def test_quote_helper_edge_cases() -> None:
    assert (
        quote_execution_module._mark_by_right(
            pd.DataFrame({"window_label": ["entry_preclose_15m"], "right": ["call"]}),
            window_label="exit_preclose_15m",
            right="put",
        )
        is None
    )
    assert np.isnan(quote_execution_module._numeric_cell(None, "bid"))
    assert np.isnan(quote_execution_module._numeric_cell(pd.Series({"bid": "bad"}), "bid"))
    assert quote_execution_module._status_cell(None) == QUOTE_STATUS_MISSING
    assert np.isnan(quote_execution_module._spot_for_group(pd.DataFrame({"ticker": ["ABC"]})))
    assert np.isnan(quote_execution_module._spot_for_group(pd.DataFrame({"spot": [-1.0]})))
    assert np.isnan(quote_execution_module._premium_var(100.0, 0.0))
    assert quote_execution_module._date_or_none(None) is None
    assert (
        quote_execution_module._event_exit_fallback(
            announcement_date=None,
            announcement_timing="AMC",
            exit_date=None,
        )
        is None
    )
    assert quote_execution_module._event_exit_fallback(
        announcement_date=date(2026, 2, 5),
        announcement_timing="BMO",
        exit_date=None,
    ) == date(2026, 2, 5)

    with pytest.raises(ValueError, match="quote marks missing required columns"):
        build_quote_straddle_execution_panel(pd.DataFrame({"event_id": ["x"]}))

    normalized = normalize_option_quote_rows(
        pd.DataFrame(
            {
                "ticker": ["O:XYZ260213C00100000"],
                "bid_price": [5.0],
                "ask_price": [5.2],
                "quote_timestamp_et": ["2026-02-05 15:59:50-05:00"],
            }
        )
    )
    requests = build_quote_window_requests(_quote_contracts().head(1), _quote_windows()).head(1)
    assert quote_execution_module._filter_normalized_quotes(normalized, requests).empty


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
    legs = pd.read_csv(out_dir / "quote_execution_legs.csv")
    straddles = pd.read_csv(out_dir / "quote_straddle_execution.csv")
    assert report["raw_full_day_files_written"] is False
    assert report["quote_rows_scanned"] == 1
    assert "quote_window_quotes" in report["output_paths"]
    assert "quote_execution_legs" in report["output_paths"]
    assert "quote_straddle_execution" in report["output_paths"]
    assert "quote_ivar_event" in report["output_paths"]
    assert "quote_iv_surface" in report["output_paths"]
    assert "quote_iv_surface_summary" in report["output_paths"]
    assert "quote_surface_ivar_event" in report["output_paths"]
    assert QUOTE_STATUS_MISSING in set(marks["quote_status"])
    assert len(legs) == 4
    assert len(straddles) == 1


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
    assert list(pd.read_csv(tmp_path / "quote_window_marks.csv").columns)
    assert list(pd.read_csv(tmp_path / "quote_window_quotes.csv").columns)
    assert list(pd.read_csv(tmp_path / "quote_ivar_event.csv").columns)
    assert list(pd.read_csv(tmp_path / "quote_iv_surface.csv").columns)
    assert list(pd.read_csv(tmp_path / "quote_iv_surface_summary.csv").columns)
    assert list(pd.read_csv(tmp_path / "quote_surface_ivar_event.csv").columns)
    assert (tmp_path / "quote_execution_report.json").exists()


def test_quote_execution_panel_streams_fake_massive_quotes(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def fake_stream(*args: object, **kwargs: object) -> list[pd.DataFrame]:
        _ = args
        date_value = kwargs["date_value"]
        assert date_value in {date(2026, 2, 5), date(2026, 2, 6)}
        timestamp = (
            "2026-02-05 15:59:50-05:00"
            if date_value == date(2026, 2, 5)
            else "2026-02-06 15:59:50-05:00"
        )
        return [
            pd.DataFrame(
                {
                    "ticker": ["O:ABC260213C00100000", "O:ABC260213P00100000"],
                    "bid_price": [5.0, 4.8],
                    "ask_price": [5.2, 5.0],
                    "quote_timestamp_et": [timestamp, timestamp],
                }
            )
        ]

    monkeypatch.setattr(quote_execution_module, "_stream_massive_quote_chunks", fake_stream)
    report = extract_quote_execution_panel(
        config=load_project_config(),
        contracts=_quote_contracts(),
        windows=_quote_windows(),
        out_dir=tmp_path,
        dates=(date(2026, 2, 5),),
        metadata_only=False,
    )

    assert report.metadata_only is False
    assert report.raw_full_day_files_written is False
    assert report.request_rows == 4
    assert report.quote_rows_scanned == 4
    assert report.quote_rows_matched == 4
    assert json.loads((tmp_path / "quote_execution_report.json").read_text())["ok"] is True


def test_quote_execution_panel_fetches_targeted_rest_quotes_with_cache(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    key_path = tmp_path / "massive.key"
    key_path.write_text("test-key\n", encoding="utf-8")
    config = replace(
        load_project_config(),
        massive_api_key_file=key_path,
        massive_base_url="https://api.massive.test",
        massive_retry_backoff_seconds=0,
    )
    calls: list[tuple[str, dict[str, object]]] = []

    class FakeResponse:
        text = ""

        def __init__(self, payload: dict[str, object]) -> None:
            self._payload = payload

        def raise_for_status(self) -> None:
            return None

        def json(self) -> dict[str, object]:
            return self._payload

    class FakeClient:
        def __init__(self, *args: object, **kwargs: object) -> None:
            _ = args, kwargs

        def __enter__(self) -> FakeClient:
            return self

        def __exit__(self, *args: object) -> None:
            _ = args

        def get(self, url: str, params: dict[str, object]) -> FakeResponse:
            calls.append((url, dict(params)))
            ticker = "O:ABC260213C00100000" if "C00100000" in url else "O:ABC260213P00100000"
            timestamp_text = str(params["timestamp.lte"])
            stamp = pd.Timestamp(timestamp_text).tz_convert("UTC")
            bid = 5.0 if ticker.endswith("C00100000") else 4.8
            ask = 5.2 if ticker.endswith("C00100000") else 5.0
            return FakeResponse(
                {
                    "results": [
                        {
                            "sip_timestamp": int(stamp.value),
                            "bid_price": bid,
                            "ask_price": ask,
                        }
                    ]
                }
            )

    monkeypatch.setattr("earnings_event_vol.quote_execution.httpx.Client", FakeClient)
    cache_dir = tmp_path / "quote_cache"
    report = extract_quote_execution_panel(
        config=config,
        contracts=_quote_contracts(),
        windows=_quote_windows(),
        out_dir=tmp_path / "first",
        dates=(date(2026, 2, 5),),
        metadata_only=False,
        quote_source="rest",
        quote_cache_dir=cache_dir,
        quote_workers=2,
    )

    assert report.route == "massive_quotes_v3_rest_targeted"
    assert report.quote_workers == 2
    assert report.request_rows == 4
    assert report.quote_rows_scanned == 4
    assert report.quote_rows_matched == 4
    assert len(calls) == 4
    assert all(call[1]["apiKey"] == "test-key" for call in calls)
    confidence = pd.read_csv(tmp_path / "first" / "quote_execution_confidence.csv")
    assert set(confidence["quote_execution_route"]) == {"massive_quotes_v3_rest_targeted"}
    assert list(cache_dir.glob("quote_date=2026-02-05/options_ticker=*/window=*.parquet"))
    assert list(cache_dir.glob("quote_date=2026-02-06/options_ticker=*/window=*.parquet"))

    cached_report = extract_quote_execution_panel(
        config=config,
        contracts=_quote_contracts(),
        windows=_quote_windows(),
        out_dir=tmp_path / "second",
        dates=(date(2026, 2, 5),),
        metadata_only=False,
        quote_source="rest",
        quote_cache_dir=cache_dir,
        quote_workers=2,
    )

    assert cached_report.quote_rows_matched == 4
    assert cached_report.quote_workers == 2
    assert len(calls) == 4

    cache_files = sorted(cache_dir.glob("quote_date=*/options_ticker=*/window=*.parquet"))
    cache_files[0].write_bytes(b"not a parquet file")
    refetched_report = extract_quote_execution_panel(
        config=config,
        contracts=_quote_contracts(),
        windows=_quote_windows(),
        out_dir=tmp_path / "third",
        dates=(date(2026, 2, 5),),
        metadata_only=False,
        quote_source="rest",
        quote_cache_dir=cache_dir,
        quote_workers=2,
    )

    assert refetched_report.quote_rows_matched == 4
    assert len(calls) == 5
    pd.read_parquet(cache_files[0])

    pd.DataFrame({"unexpected": [1]}).to_parquet(cache_files[1], index=False)
    schema_refetched_report = extract_quote_execution_panel(
        config=config,
        contracts=_quote_contracts(),
        windows=_quote_windows(),
        out_dir=tmp_path / "fourth",
        dates=(date(2026, 2, 5),),
        metadata_only=False,
        quote_source="rest",
        quote_cache_dir=cache_dir,
        quote_workers=2,
    )

    assert schema_refetched_report.quote_rows_matched == 4
    assert len(calls) == 6
    assert "unexpected" not in pd.read_parquet(cache_files[1]).columns


def test_quote_rest_cache_loader_discards_empty_files(tmp_path: Path) -> None:
    cache_path = tmp_path / "empty.parquet"
    cache_path.touch()

    assert quote_execution_module._load_cached_normalized_quotes(cache_path) is None
    assert not cache_path.exists()


def test_quote_rest_cache_loader_discards_non_dataframe_cache(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    cache_path = tmp_path / "bad.parquet"
    cache_path.write_bytes(b"nonempty")
    monkeypatch.setattr("earnings_event_vol.quote_execution.pd.read_parquet", lambda _: ["bad"])

    assert quote_execution_module._load_cached_normalized_quotes(cache_path) is None
    assert not cache_path.exists()


def test_quote_rest_retries_paginates_and_redacts_errors(tmp_path: Path) -> None:
    key_path = tmp_path / "massive.key"
    key_path.write_text("test-key\n", encoding="utf-8")
    config = replace(
        load_project_config(),
        massive_api_key_file=key_path,
        massive_base_url="https://api.massive.test",
        massive_max_retries=1,
        massive_retry_backoff_seconds=0,
    )
    stamp = int(pd.Timestamp("2026-02-05 20:59:50Z").value)
    calls: list[tuple[str, dict[str, object]]] = []

    class RetryClient:
        def get(self, url: str, params: dict[str, object]) -> httpx.Response:
            calls.append((url, dict(params)))
            request = httpx.Request("GET", url)
            if len(calls) == 1:
                return httpx.Response(503, request=request, text="temporary apiKey=test-key")
            if len(calls) == 2:
                return httpx.Response(
                    200,
                    request=request,
                    json={
                        "results": [{"sip_timestamp": stamp, "bid_price": 5.0, "ask_price": 5.2}],
                        "next_url": "/v3/quotes/next",
                    },
                )
            return httpx.Response(
                200,
                request=request,
                json={"results": [{"sip_timestamp": stamp, "bid_price": 5.1, "ask_price": 5.3}]},
            )

    quotes = quote_execution_module.fetch_massive_option_quote_window_rest(
        cast(httpx.Client, RetryClient()),
        config,
        options_ticker="O:ABC260213C00100000",
        window_start="2026-02-05 15:45:00-05:00",
        window_end="2026-02-05 16:00:00-05:00",
    )

    assert len(quotes) == 2
    assert len(calls) == 3
    assert calls[-1][0] == "https://api.massive.test/v3/quotes/next"
    assert calls[-1][1] == {"apiKey": "test-key"}

    class BadRequestClient:
        def get(self, url: str, params: dict[str, object]) -> httpx.Response:
            _ = params
            return httpx.Response(
                400,
                request=httpx.Request("GET", url),
                text="bad apiKey=test-key token",
            )

    with pytest.raises(RuntimeError) as exc_info:
        quote_execution_module.fetch_massive_option_quote_window_rest(
            cast(httpx.Client, BadRequestClient()),
            config,
            options_ticker="O:ABC260213C00100000",
            window_start="2026-02-05 15:45:00-05:00",
            window_end="2026-02-05 16:00:00-05:00",
        )
    assert "test-key" not in str(exc_info.value)
    assert "<redacted>" in str(exc_info.value)

    class TransportClient:
        def get(self, url: str, params: dict[str, object]) -> httpx.Response:
            _ = url, params
            raise httpx.TransportError("temporary network failure")

    with pytest.raises(httpx.TransportError, match="temporary network failure"):
        quote_execution_module._get_json_with_retries(
            cast(httpx.Client, TransportClient()),
            "https://api.massive.test/v3/quotes/O:ABC",
            params={"apiKey": "test-key"},
            config=config,
        )


def test_quote_execution_misc_defensive_branches(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    assert (
        quote_execution_module._to_datetime_et("2026-02-05 15:45:00")
        .tz_convert("America/New_York")
        .hour
        == 15
    )
    assert "apiKey=<redacted>" in quote_execution_module._safe_exception_text(
        ValueError("failed url?apiKey=test-key")
    )

    empty_key_path = tmp_path / "empty.key"
    empty_key_path.write_text("\n", encoding="utf-8")
    missing_key_config = replace(
        load_project_config(),
        massive_api_key_file=empty_key_path,
    )
    with pytest.raises(ValueError, match="MASSIVE_API_KEY_FILE"):
        quote_execution_module._api_key(missing_key_config)

    key_path = tmp_path / "massive.key"
    key_path.write_text("test-key\n", encoding="utf-8")
    config = replace(
        load_project_config(),
        massive_api_key_file=key_path,
        massive_base_url="https://api.massive.test",
        massive_max_retries=1,
        massive_retry_backoff_seconds=0.5,
    )
    sleeps: list[float] = []
    monkeypatch.setattr("earnings_event_vol.quote_execution.time_module.sleep", sleeps.append)

    class EmptyRetryClient:
        def __init__(self) -> None:
            self.calls = 0

        def get(self, url: str, params: dict[str, object]) -> httpx.Response:
            _ = params
            self.calls += 1
            request = httpx.Request("GET", url)
            if self.calls == 1:
                return httpx.Response(503, request=request, text="temporary")
            return httpx.Response(200, request=request, json={"results": []})

    quotes = quote_execution_module.fetch_massive_option_quote_window_rest(
        cast(httpx.Client, EmptyRetryClient()),
        config,
        options_ticker="O:ABC260213C00100000",
        window_start="2026-02-05 15:45:00-05:00",
        window_end="2026-02-05 16:00:00-05:00",
    )
    assert quotes.empty
    assert sleeps == [0.5]

    with pytest.raises(ValueError, match="unsupported quote_source"):
        extract_quote_execution_panel(
            config=config,
            contracts=_quote_contracts(),
            windows=_quote_windows(),
            out_dir=tmp_path / "bad-source",
            quote_source="bad",
        )
    with pytest.raises(ValueError, match="quote_workers must be positive"):
        extract_quote_execution_panel(
            config=config,
            contracts=_quote_contracts(),
            windows=_quote_windows(),
            out_dir=tmp_path / "bad-workers",
            quote_workers=0,
        )
    with pytest.raises(ValueError, match="event_offset must be non-negative"):
        extract_quote_execution_panel(
            config=config,
            contracts=_quote_contracts(),
            windows=_quote_windows(),
            out_dir=tmp_path / "bad-offset",
            event_offset=-1,
        )


def test_data_pipeline_quote_execution_stage_writes_lake_artifacts(tmp_path: Path) -> None:
    config = replace(
        load_project_config(),
        bronze_data_dir=tmp_path / "bronze",
        silver_data_dir=tmp_path / "silver",
        gold_data_dir=tmp_path / "gold",
    )
    contracts_path = config.silver_data_dir / "contracts" / "event_contract_candidates.parquet"
    windows_path = config.silver_data_dir / "event_windows" / "event_windows.parquet"
    contracts_path.parent.mkdir(parents=True)
    windows_path.parent.mkdir(parents=True)
    contracts, windows = _two_event_quote_contracts_and_windows()
    pl.from_pandas(contracts).write_parquet(contracts_path)
    pl.from_pandas(windows).write_parquet(windows_path)

    result = run_data_pipeline(
        config,
        stage="quote-execution-panel",
        out_root=tmp_path / "artifacts" / "data_pipeline",
        max_events=1,
        quote_workers=3,
    )

    assert result["ok"] is True
    steps = cast(list[dict[str, object]], result["steps"])
    step = steps[0]
    assert step["status"] == "ran"
    metadata = cast(dict[str, object], step["metadata"])
    assert metadata["metadata_only"] is True
    requests = pd.read_parquet(
        config.bronze_data_dir
        / "massive"
        / "quotes_v1_target_windows"
        / "quote_window_requests.parquet"
    )
    assert len(requests) == 4
    assert pd.read_parquet(
        config.bronze_data_dir
        / "massive"
        / "quotes_v1_target_windows"
        / "quote_window_quotes.parquet"
    ).empty
    assert pd.read_parquet(
        config.silver_data_dir / "quote_execution" / "quote_window_marks.parquet"
    ).empty
    assert pd.read_parquet(
        config.gold_data_dir / "quote_execution" / "quote_execution_confidence.parquet"
    ).empty
    assert pd.read_parquet(
        config.gold_data_dir / "quote_execution" / "quote_ivar_event.parquet"
    ).empty
    assert pd.read_parquet(
        config.gold_data_dir / "quote_execution" / "quote_iv_surface.parquet"
    ).empty
    assert pd.read_parquet(
        config.gold_data_dir / "quote_execution" / "quote_iv_surface_summary.parquet"
    ).empty
    assert pd.read_parquet(
        config.gold_data_dir / "quote_execution" / "quote_surface_ivar_event.parquet"
    ).empty
    manifest = json.loads(
        (
            tmp_path
            / "artifacts"
            / "data_pipeline"
            / "quote_execution_panel"
            / "quote_execution_panel_manifest.json"
        ).read_text(encoding="utf-8")
    )
    assert manifest["pipeline_params"]["quote_workers"] == 3
    assert manifest["report"]["quote_workers"] == 3
    assert manifest["lake_policy"]["rest_workers"] == 3
    assert manifest["lake_policy"]["batch_mode"] is False
    assert manifest["lake_policy"]["canonical_outputs_updated"] is True
    assert manifest["lake_policy"]["raw_full_day_quote_files_in_repo"] is False

    batch = run_data_pipeline(
        config,
        stage="quote-execution-panel",
        out_root=tmp_path / "artifacts" / "data_pipeline",
        max_events=1,
        quote_workers=3,
        quote_event_offset=1,
        quote_batch_label="offset1_size1",
        force=True,
    )
    assert batch["ok"] is True
    batch_step = cast(list[dict[str, object]], batch["steps"])[0]
    batch_metadata = cast(dict[str, object], batch_step["metadata"])
    assert batch_metadata["event_offset"] == 1
    assert batch_metadata["quote_batch_label"] == "offset1_size1"

    batch_requests = pd.read_parquet(
        config.bronze_data_dir
        / "massive"
        / "quotes_v1_target_windows"
        / "batches"
        / "batch=offset1_size1"
        / "quote_window_requests.parquet"
    )
    assert len(batch_requests) == 4
    assert set(batch_requests["event_id"]) == {"XYZ_2026Q1"}
    canonical_requests = pd.read_parquet(
        config.bronze_data_dir
        / "massive"
        / "quotes_v1_target_windows"
        / "quote_window_requests.parquet"
    )
    assert set(canonical_requests["event_id"]) == {"ABC_2026Q1"}
    batch_manifest = json.loads(
        (
            tmp_path
            / "artifacts"
            / "data_pipeline"
            / "quote_execution_panel"
            / "batches"
            / "offset1_size1"
            / "quote_execution_panel_manifest.json"
        ).read_text(encoding="utf-8")
    )
    assert batch_manifest["pipeline_params"]["event_offset"] == 1
    assert batch_manifest["pipeline_params"]["quote_batch_label"] == "offset1_size1"
    assert batch_manifest["lake_policy"]["batch_mode"] is True
    assert batch_manifest["lake_policy"]["batch_label"] == "offset1_size1"
    assert batch_manifest["lake_policy"]["canonical_outputs_updated"] is False

    merged = run_data_pipeline(
        config,
        stage="quote-execution-merge",
        out_root=tmp_path / "artifacts" / "data_pipeline",
        quote_merge_batch_labels=["offset1_size1"],
        force=True,
    )
    assert merged["ok"] is True
    merge_step = cast(list[dict[str, object]], merged["steps"])[0]
    assert merge_step["status"] == "ran"
    merge_metadata = cast(dict[str, object], merge_step["metadata"])
    assert merge_metadata["batch_labels"] == ["offset1_size1"]
    merged_requests = pd.read_parquet(
        config.bronze_data_dir
        / "massive"
        / "quotes_v1_target_windows"
        / "quote_window_requests.parquet"
    )
    assert len(merged_requests) == 8
    assert set(merged_requests["event_id"]) == {"ABC_2026Q1", "XYZ_2026Q1"}
    merged_manifest = json.loads(
        (
            tmp_path
            / "artifacts"
            / "data_pipeline"
            / "quote_execution_panel"
            / "quote_execution_panel_manifest.json"
        ).read_text(encoding="utf-8")
    )
    assert merged_manifest["pipeline_params"]["stage"] == "quote-execution-merge"
    assert merged_manifest["pipeline_params"]["batch_labels"] == ["offset1_size1"]
    assert merged_manifest["lake_policy"]["batch_consolidation"] is True
    assert merged_manifest["lake_policy"]["canonical_outputs_updated"] is True
    assert merged_manifest["lake_output_rows"]["bronze_quote_window_requests"] == 8

    merge_resume = run_data_pipeline(
        config,
        stage="quote-execution-merge",
        out_root=tmp_path / "artifacts" / "data_pipeline",
        quote_merge_batch_labels=["offset1_size1"],
    )
    assert cast(list[dict[str, object]], merge_resume["steps"])[0]["status"] == "skipped"

    blocked = run_data_pipeline(
        config,
        stage="quote-execution-panel",
        out_root=tmp_path / "blocked",
        quote_metadata_only=False,
    )
    assert blocked["ok"] is False
    blocked_steps = cast(list[dict[str, object]], blocked["steps"])
    assert blocked_steps[0]["reason"] == "requires_quote_dates_or_allow_all_dates_for_quote_stream"


def test_quote_execution_merge_discovers_batches_and_dedupes(tmp_path: Path) -> None:
    config = replace(
        load_project_config(),
        bronze_data_dir=tmp_path / "bronze",
        silver_data_dir=tmp_path / "silver",
        gold_data_dir=tmp_path / "gold",
    )
    out_root = tmp_path / "artifacts" / "data_pipeline"
    assert str(
        data_pipeline_module._quote_execution_artifact_root(out_root, batch_label="disc1")
    ).endswith("quote_execution_panel/batches/disc1")

    early_block = run_data_pipeline(
        config,
        stage="quote-execution-merge",
        out_root=out_root / "early_block",
        quote_merge_include_canonical=False,
    )
    assert early_block["ok"] is False
    assert cast(list[dict[str, object]], early_block["steps"])[0]["reason"] == (
        "requires_quote_batch_labels_or_canonical_input"
    )
    no_input = run_data_pipeline(
        config,
        stage="quote-execution-merge",
        out_root=out_root / "no_input",
    )
    assert no_input["ok"] is False
    assert cast(list[dict[str, object]], no_input["steps"])[0]["reason"] == (
        "requires_existing_quote_execution_lake_sources"
    )

    row = {
        "event_id": "DISC_2026Q1",
        "ticker": "DISC",
        "options_ticker": "O:DISC260213C00100000",
        "window_label": "entry_preclose_15m",
        "window_start": "2026-02-05 15:45:00-05:00",
        "window_end": "2026-02-05 16:00:00-05:00",
        "quote_timestamp_et": "2026-02-05 15:59:50-05:00",
        "quote_date": "2026-02-05",
        "bid": 1.0,
        "ask": 1.2,
        "expiration": "2026-02-13",
        "strike": 100.0,
        "right": "call",
        "execution_confidence_band": "high",
        "revision": 1,
    }
    updated = {**row, "revision": 2}
    batch_paths = data_pipeline_module._quote_execution_lake_paths(config, batch_label="disc1")
    for path in batch_paths.values():
        path.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame([row, updated]).to_parquet(path, index=False)

    merged = run_data_pipeline(
        config,
        stage="quote-execution-merge",
        out_root=out_root,
        quote_merge_include_canonical=False,
    )
    assert merged["ok"] is True
    step = cast(list[dict[str, object]], merged["steps"])[0]
    assert step["status"] == "ran"
    metadata = cast(dict[str, object], step["metadata"])
    assert metadata["batch_labels"] == ["disc1"]
    assert metadata["event_count"] == 1
    row_counts = cast(dict[str, int], metadata["lake_output_rows"])
    assert set(row_counts.values()) == {1}

    canonical_requests = pd.read_parquet(
        config.bronze_data_dir
        / "massive"
        / "quotes_v1_target_windows"
        / "quote_window_requests.parquet"
    )
    assert len(canonical_requests) == 1
    assert canonical_requests["revision"].iloc[0] == 2
    report = json.loads(
        (out_root / "quote_execution_panel" / "quote_execution_report.json").read_text(
            encoding="utf-8"
        )
    )
    assert report["route"] == "quote_batch_consolidation"
    assert report["event_count"] == 1


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
            "required_quote_marks": [4, 4],
            "ok_quote_marks": [4, 0],
            "max_quote_age_seconds": [10.0, np.nan],
            "median_spread_over_mid": [0.02, 0.50],
            "quote_execution_route": ["massive_quotes_v1_flat_file_filtered"] * 2,
            "quote_execution_paper_grade": [False, False],
            "quote_mid_ivar_event": [0.01, 0.02],
            "quote_ivar_claim_scope": ["diagnostic_quote_premium_proxy_not_model_feature"] * 2,
            "quote_mid_iv": [0.30, 0.35],
            "quote_bid_iv": [0.29, 0.34],
            "quote_ask_iv": [0.31, 0.36],
            "quote_surface_mid_ivar_event": [0.011, 0.021],
            "quote_surface_ivar_claim_scope": [
                "bounded_quote_iv_surface_ivar_diagnostic_not_full_nbbo_surface"
            ]
            * 2,
            "surface_pair_count": [2, 2],
            "paper_grade_quote_iv_mid": [True, False],
            "paper_grade_quote_surface_ivar_mid": [True, False],
            "expiry_candidate_count": [2, 0],
            "spread_over_mid": [0.02, 0.50],
            "plain_pre_event_feature": [1.0, 2.0],
        }
    )
    schema = build_feature_schema_report(frame, feature_schema_version=FEATURE_SCHEMA_V2_SEC_XBRL)
    model_features = set(schema.loc[schema["model_feature"], "feature_name"])
    assert "plain_pre_event_feature" in model_features
    assert "execution_confidence_score" not in model_features
    assert "execution_confidence_band" not in model_features
    assert "required_quote_marks" not in model_features
    assert "ok_quote_marks" not in model_features
    assert "max_quote_age_seconds" not in model_features
    assert "median_spread_over_mid" not in model_features
    assert "quote_execution_route" not in model_features
    assert "quote_execution_paper_grade" not in model_features
    assert "quote_mid_ivar_event" not in model_features
    assert "quote_ivar_claim_scope" not in model_features
    assert "quote_mid_iv" not in model_features
    assert "quote_bid_iv" not in model_features
    assert "quote_ask_iv" not in model_features
    assert "quote_surface_mid_ivar_event" not in model_features
    assert "quote_surface_ivar_claim_scope" not in model_features
    assert "surface_pair_count" not in model_features
    assert "paper_grade_quote_iv_mid" not in model_features
    assert "paper_grade_quote_surface_ivar_mid" not in model_features
    assert "expiry_candidate_count" not in model_features
    assert "spread_over_mid" not in model_features


def test_merge_quote_execution_diagnostics_attaches_analysis_only_columns(
    tmp_path: Path,
) -> None:
    config = replace(load_project_config(), gold_data_dir=tmp_path / "gold")
    quote_root = config.gold_data_dir / "quote_execution"
    quote_root.mkdir(parents=True)
    pd.DataFrame(
        {
            "event_id": ["E1"],
            "quote_execution_route": ["massive_quotes_v3_rest_targeted"],
            "required_quote_marks": [4],
            "ok_quote_marks": [4],
            "execution_confidence_score": [1.0],
            "execution_confidence_band": ["high"],
            "paper_grade": [False],
        }
    ).to_parquet(quote_root / "quote_execution_confidence.parquet", index=False)
    pd.DataFrame(
        {
            "event_id": ["E1"],
            "quote_ivar_method": ["entry_straddle_premium_total_variance_proxy"],
            "quote_ivar_claim_scope": ["diagnostic_quote_premium_proxy_not_model_feature"],
            "expiry_candidate_count": [2],
            "quote_mid_ivar_event": [0.02],
            "quote_mid_ivar_failure_reason": [None],
        }
    ).to_parquet(quote_root / "quote_ivar_event.parquet", index=False)

    merged = merge_quote_execution_diagnostics(
        pd.DataFrame({"event_id": ["E1", "E2"], "plain_pre_event_feature": [1.0, 2.0]}),
        config,
    )

    assert merged.loc[merged["event_id"].eq("E1"), "execution_confidence_band"].iloc[0] == "high"
    assert merged.loc[merged["event_id"].eq("E2"), "execution_confidence_band"].iloc[0] == "missing"
    assert "quote_execution_paper_grade" in merged.columns
    schema = build_feature_schema_report(merged, feature_schema_version=FEATURE_SCHEMA_V2_SEC_XBRL)
    model_features = set(schema.loc[schema["model_feature"], "feature_name"])
    assert "plain_pre_event_feature" in model_features
    assert "required_quote_marks" not in model_features
    assert "quote_mid_ivar_event" not in model_features

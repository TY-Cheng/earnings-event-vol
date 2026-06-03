from __future__ import annotations

import json
import re
from datetime import date
from pathlib import Path
from typing import Any, Literal, cast

import pandas as pd
import polars as pl

from earnings_event_vol.config import ProjectConfig
from earnings_event_vol.earnings_calendar import infer_timing_from_acceptance_timestamp
from earnings_event_vol.event_targets import add_event_return_targets
from earnings_event_vol.events import (
    is_us_equity_trading_day,
    market_close_timestamp,
    next_us_equity_trading_day,
    previous_us_equity_trading_day,
)
from earnings_event_vol.massive import (
    _run_head_object_command,
    build_download_file_command,
    massive_flat_file_aws_env,
    option_flat_file_key,
    parse_massive_option_ticker,
    underlying_flat_file_key,
)
from earnings_event_vol.schemas import AnnouncementTiming

PARQUET_COMPRESSION: Literal["zstd"] = "zstd"


def _write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")


def _write_parquet(path: Path, frame: pd.DataFrame | pl.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    pl_frame = pl.from_pandas(frame) if isinstance(frame, pd.DataFrame) else frame
    pl_frame.write_parquet(path, compression=PARQUET_COMPRESSION)


def _event_id(row: pd.Series) -> str:
    source_id = str(row.get("source_id", "")).replace("/", "_")
    return f"{row['ticker']}_{row['announcement_date']}_{source_id}"


def _download_flat_file(config: ProjectConfig, *, key: str, destination: Path) -> bool:
    if destination.exists() and destination.stat().st_size > 0:
        return True
    destination.parent.mkdir(parents=True, exist_ok=True)
    result = _run_head_object_command(
        build_download_file_command(config, key=key, destination=destination),
        massive_flat_file_aws_env(config),
        config.massive_request_timeout_seconds * 8,
    )
    if result.returncode != 0:
        if destination.exists():
            destination.unlink()
        return False
    return True


def _bronze_massive_path(config: ProjectConfig, *, dataset: str, day: date) -> Path:
    return config.bronze_data_dir / "massive" / dataset / f"date={day.isoformat()}" / "part.parquet"


def _tmp_flat_file_path(config: ProjectConfig, *, dataset: str, day: date) -> Path:
    return config.bronze_data_dir / "_tmp" / "massive" / dataset / f"{day.isoformat()}.csv.gz"


def _convert_flat_file_to_bronze(
    *,
    tmp_path: Path,
    destination: Path,
    dataset: str,
    source_key: str,
    source_date: date,
) -> None:
    frame = pl.read_csv(tmp_path)
    frame = frame.with_columns(
        [
            pl.lit(source_date.isoformat()).alias("source_date"),
            pl.lit(dataset).alias("source_dataset"),
            pl.lit(source_key).alias("source_key"),
        ]
    )
    destination.parent.mkdir(parents=True, exist_ok=True)
    tmp_parquet = destination.with_name(f".{destination.stem}.tmp.parquet")
    frame.write_parquet(tmp_parquet, compression=PARQUET_COMPRESSION)
    tmp_parquet.replace(destination)
    tmp_path.unlink(missing_ok=True)


def _underlying_path(config: ProjectConfig, day: date) -> Path:
    return _bronze_massive_path(config, dataset="underlying_day_aggs", day=day)


def _options_path(config: ProjectConfig, day: date) -> Path:
    return _bronze_massive_path(config, dataset="options_day_aggs", day=day)


def _ensure_bronze_flat_file(
    config: ProjectConfig,
    *,
    dataset: str,
    key: str,
    destination: Path,
    day: date,
) -> bool:
    if destination.exists() and destination.stat().st_size > 0:
        return True
    tmp_path = _tmp_flat_file_path(config, dataset=dataset, day=day)
    if not _download_flat_file(config, key=key, destination=tmp_path):
        return False
    try:
        _convert_flat_file_to_bronze(
            tmp_path=tmp_path,
            destination=destination,
            dataset=dataset,
            source_key=key,
            source_date=day,
        )
    except Exception:
        destination.unlink(missing_ok=True)
        raise
    return True


def _ensure_underlying_file(config: ProjectConfig, day: date) -> bool:
    path = _underlying_path(config, day)
    key = underlying_flat_file_key(config, date=day.isoformat())
    return _ensure_bronze_flat_file(
        config,
        dataset="underlying_day_aggs",
        key=key,
        destination=path,
        day=day,
    )


def _ensure_options_file(config: ProjectConfig, day: date) -> bool:
    path = _options_path(config, day)
    key = option_flat_file_key(config, year=day.year, month=day.month, date=day.isoformat())
    return _ensure_bronze_flat_file(
        config,
        dataset="options_day_aggs",
        key=key,
        destination=path,
        day=day,
    )


def _find_trading_day(
    config: ProjectConfig,
    target: date,
    *,
    direction: int,
    include_target: bool,
    max_calendar_days: int = 10,
) -> date | None:
    del config, max_calendar_days
    if direction < 0:
        return previous_us_equity_trading_day(target, include_target=include_target)
    if direction > 0:
        return next_us_equity_trading_day(target, include_target=include_target)
    return target if include_target and is_us_equity_trading_day(target) else None


def _source_timestamp_timing(row: dict[str, object]) -> AnnouncementTiming:
    raw_inferred = row.get("acceptance_inferred_timing")
    if raw_inferred is not None and not pd.isna(raw_inferred):
        try:
            return AnnouncementTiming(str(raw_inferred))
        except ValueError:
            pass
    return infer_timing_from_acceptance_timestamp(row.get("source_timestamp"))


def _load_underlying_bars(path: Path, tickers: set[str], day: date) -> list[dict[str, object]]:
    frame = (
        pl.scan_parquet(path)
        .with_columns(pl.col("ticker").str.to_uppercase().alias("ticker"))
        .filter(pl.col("ticker").is_in(sorted(tickers)))
        .select(["ticker", "open", "high", "low", "close", "volume"])
        .collect()
    )
    return [
        {
            "ticker": str(row["ticker"]),
            "date": day,
            "open": float(row["open"]),
            "high": float(row["high"]),
            "low": float(row["low"]),
            "close": float(row["close"]),
            "volume": int(float(row["volume"])),
            "source_dataset": "underlying_day_aggs",
        }
        for row in frame.iter_rows(named=True)
    ]


def _load_option_day_contracts(
    path: Path,
    *,
    event: pd.Series,
    dte_min: int,
    dte_max: int,
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    ticker = str(event["ticker"]).upper()
    option_ticker_pattern = rf"^O:{re.escape(ticker)}\d{{6}}[CP]\d{{8}}$"
    entry_date = pd.Timestamp(event["entry_date"]).date()
    exit_date = pd.Timestamp(event["exit_date"]).date()
    spot = float(event["s_before"])
    frame = pl.read_parquet(path)
    if "transactions" not in frame.columns:
        frame = frame.with_columns(pl.lit(0).alias("transactions"))
    frame = frame.filter(pl.col("ticker").str.contains(option_ticker_pattern))
    for row in frame.iter_rows(named=True):
        option_symbol = str(row["ticker"])
        try:
            parsed = parse_massive_option_ticker(option_symbol)
        except ValueError:
            continue
        if str(parsed["ticker"]).upper() != ticker:
            continue
        expiration = parsed["expiration"]
        assert isinstance(expiration, date)
        dte = (expiration - entry_date).days
        requested_dte_min = int(event.get("requested_dte_min", dte_min))
        requested_dte_max = int(event.get("requested_dte_max", dte_max))
        if dte < 0 or dte > dte_max or expiration < exit_date:
            continue
        close_value = cast(Any, row["close"])
        volume_value = cast(Any, row["volume"])
        close = float(close_value) if close_value is not None else float("nan")
        volume = int(float(volume_value)) if volume_value is not None else 0
        if close <= 0 or volume <= 0:
            continue
        strike = float(cast(Any, parsed["strike"]))
        in_requested_dte = requested_dte_min <= dte <= requested_dte_max
        rows.append(
            {
                "event_id": event["event_id"],
                "ticker": ticker,
                "entry_date": entry_date,
                "exit_date": exit_date,
                "expiration": expiration,
                "strike": strike,
                "right": parsed["right"],
                "options_ticker": option_symbol,
                "option_close": close,
                "volume": volume,
                "transactions": int(float(cast(Any, row["transactions"]) or 0)),
                "dte": dte,
                "moneyness_abs": abs(strike / spot - 1.0),
                "covers_event_window": True,
                "option_multiplier": 100,
                "contract_size": 100,
                "deliverable_status": "standard",
                "corporate_action_flag": False,
                "contract_discovery_status": "ok" if in_requested_dte else "ivar_support_only",
                "eligible_for_quote_pool": bool(in_requested_dte),
                "is_main_dte_5_14": bool(in_requested_dte and 5 <= dte <= 14),
                "is_robustness_dte_3_21": bool(in_requested_dte and 3 <= dte <= 21),
                "is_ivar_support_only": bool(not in_requested_dte),
                "quote_route": "options_day_aggs_close_proxy",
            }
        )
    return rows


def _select_near_atm_contracts(
    contracts: pd.DataFrame,
    *,
    strikes_per_expiry: int,
) -> pd.DataFrame:
    selected: list[pd.DataFrame] = []
    for (_event_id, _expiration), group in contracts.groupby(["event_id", "expiration"]):
        pair_counts = group.groupby("strike")["right"].nunique()
        paired_strikes = pair_counts.loc[pair_counts.ge(2)].index
        candidate_group = (
            group.loc[group["strike"].isin(paired_strikes)].copy()
            if len(paired_strikes) > 0
            else group
        )
        strikes = (
            candidate_group[["strike", "moneyness_abs"]]
            .drop_duplicates()
            .sort_values(["moneyness_abs", "strike"])
            .head(strikes_per_expiry)["strike"]
        )
        selected.append(group.loc[group["strike"].isin(strikes)].copy())
    if not selected:
        return pd.DataFrame(columns=contracts.columns)
    return pd.concat(selected, ignore_index=True)


def build_event_window_panel(
    *,
    config: ProjectConfig,
    calendar_path: Path,
    out_root: Path,
    dte_min: int,
    dte_max: int,
    strikes_per_expiry: int,
    max_events: int | None,
) -> dict[str, object]:
    second_expiry_dte_max = max(dte_max + 14, 28)
    silver_calendar_dir = config.silver_data_dir / "earnings_calendar"
    silver_windows_dir = config.silver_data_dir / "event_windows"
    silver_contracts_dir = config.silver_data_dir / "contracts"

    calendar = pd.read_csv(calendar_path)
    calendar["announcement_date"] = pd.to_datetime(calendar["announcement_date"]).dt.date
    main = calendar.loc[calendar["is_main_sample_candidate"].astype(bool)].copy()
    main["event_id"] = main.apply(_event_id, axis=1)
    if max_events is not None:
        main = main.head(max_events).copy()

    audit_dir = out_root / "calendar_audit"
    audit_dir.mkdir(parents=True, exist_ok=True)
    _write_parquet(silver_calendar_dir / "main_sample.parquet", main)
    source_timestamp = main.get(
        "source_timestamp",
        pd.Series([pd.NaT] * len(main), index=main.index, dtype=object),
    )
    local_ts = pd.to_datetime(source_timestamp, utc=True, errors="coerce").dt.tz_convert(
        "America/New_York"
    )
    timing_confidence = (
        main["timing_confidence"].astype(str)
        if "timing_confidence" in main.columns
        else pd.Series([], dtype=str)
    )
    audit_report = {
        "candidate_rows": int(len(calendar)),
        "main_sample_rows": int(len(main)),
        "main_timing_counts": main["announcement_timing"].value_counts().to_dict(),
        "main_text_validation_counts": main["text_validation_status"].value_counts().to_dict(),
        "proxy_timing_rows": int(timing_confidence.str.startswith("proxy").sum()),
        "accepted_local_date_mismatch_rows": int(
            (local_ts.dt.date != main["announcement_date"]).sum()
        ),
        "timing_note": (
            "SEC acceptance timestamp is retained as a proxy audit field; this event-window "
            "setup is not a final first-release-time sample."
        ),
    }
    _write_json(audit_dir / "calendar_audit_report.json", audit_report)

    window_rows: list[dict[str, object]] = []
    for row in main.to_dict("records"):
        announcement_date = row["announcement_date"]
        timing = AnnouncementTiming(str(row["announcement_timing"]))
        source_timing = _source_timestamp_timing(row)
        exclusion_reason = None
        if source_timing == AnnouncementTiming.DMH or (
            source_timing in {AnnouncementTiming.BMO, AnnouncementTiming.AMC}
            and timing in {AnnouncementTiming.BMO, AnnouncementTiming.AMC}
            and source_timing != timing
        ):
            entry_date = None
            exit_date = None
            exclusion_reason = "source_timestamp_timing_mismatch"
        elif not is_us_equity_trading_day(announcement_date):
            entry_date = None
            exit_date = None
            exclusion_reason = "announcement_date_not_trading_day"
        elif timing == AnnouncementTiming.AMC:
            entry_date = _find_trading_day(
                config, announcement_date, direction=-1, include_target=True
            )
            exit_date = _find_trading_day(
                config, announcement_date, direction=1, include_target=False
            )
        elif timing == AnnouncementTiming.BMO:
            entry_date = _find_trading_day(
                config, announcement_date, direction=-1, include_target=False
            )
            exit_date = _find_trading_day(
                config, announcement_date, direction=-1, include_target=True
            )
        else:
            entry_date = None
            exit_date = None
        if exclusion_reason is None and (not entry_date or not exit_date):
            exclusion_reason = "missing_entry_or_exit_date"
        window_rows.append(
            {
                **row,
                "entry_date": entry_date,
                "exit_date": exit_date,
                "feature_cutoff_date": entry_date,
                "requested_dte_min": dte_min,
                "requested_dte_max": dte_max,
                "ivar_support_dte_max": second_expiry_dte_max,
                "event_entry_timestamp": market_close_timestamp(entry_date).isoformat()
                if entry_date
                else None,
                "exclusion_reason": exclusion_reason,
            }
        )
    window_columns = [
        *main.columns,
        "entry_date",
        "exit_date",
        "feature_cutoff_date",
        "requested_dte_min",
        "requested_dte_max",
        "ivar_support_dte_max",
        "event_entry_timestamp",
        "exclusion_reason",
    ]
    windows = pd.DataFrame(window_rows, columns=window_columns)
    _write_parquet(silver_windows_dir / "event_windows.parquet", windows)

    tickers = set(windows["ticker"].astype(str).str.upper())
    bar_rows: list[dict[str, object]] = []
    for day in sorted(
        {
            pd.Timestamp(value).date()
            for value in pd.concat([windows["entry_date"], windows["exit_date"]]).dropna()
        }
    ):
        if _ensure_underlying_file(config, day):
            bar_rows.extend(_load_underlying_bars(_underlying_path(config, day), tickers, day))
    bars = pd.DataFrame(bar_rows)
    _write_parquet(silver_windows_dir / "underlying_event_bars.parquet", bars)
    windows = add_event_return_targets(windows, bars)
    for column in ("s_before", "s_after"):
        if column not in windows.columns:
            windows[column] = pd.NA
    missing_underlying = windows["exclusion_reason"].isna() & (
        windows["s_before"].isna() | windows["s_after"].isna()
    )
    windows.loc[missing_underlying, "exclusion_reason"] = "missing_underlying_entry_or_exit_bar"
    _write_parquet(silver_windows_dir / "event_windows.parquet", windows)

    contract_rows: list[dict[str, object]] = []
    valid_windows = windows.loc[
        windows["exclusion_reason"].isna() & windows["s_before"].notna()
    ].copy()
    for entry_date, group in valid_windows.groupby("entry_date"):
        day = pd.Timestamp(entry_date).date()
        if not _ensure_options_file(config, day):
            continue
        options_path = _options_path(config, day)
        for _, event in group.iterrows():
            contract_rows.extend(
                _load_option_day_contracts(
                    options_path,
                    event=event,
                    dte_min=dte_min,
                    dte_max=second_expiry_dte_max,
                )
            )
    contract_columns = [
        "event_id",
        "ticker",
        "entry_date",
        "exit_date",
        "expiration",
        "strike",
        "right",
        "options_ticker",
        "option_close",
        "volume",
        "transactions",
        "dte",
        "moneyness_abs",
        "covers_event_window",
        "option_multiplier",
        "contract_size",
        "deliverable_status",
        "corporate_action_flag",
        "contract_discovery_status",
        "eligible_for_quote_pool",
        "is_main_dte_5_14",
        "is_robustness_dte_3_21",
        "is_ivar_support_only",
        "quote_route",
    ]
    contracts = pd.DataFrame(contract_rows, columns=contract_columns)
    if not contracts.empty:
        contracts = _select_near_atm_contracts(contracts, strikes_per_expiry=strikes_per_expiry)
    _write_parquet(silver_contracts_dir / "event_contract_candidates.parquet", contracts)

    quote_report = {
        "nbbo_quote_route": "blocked",
        "rest_quotes_status": "not_entitled_in_current_massive_plan",
        "s3_select_status": "method_not_allowed_on_massive_flat_file_endpoint",
        "flat_file_full_scan_status": "deferred_too_large_for_proxy_pipeline",
        "observed_quotes_v1_compressed_size_example_bytes": 115_992_798_867,
        "provisional_route_used": "options_day_aggs_close_proxy",
        "paper_grade": False,
        "lake_note": (
            "Downloaded Massive CSV.GZ day files are converted immediately into bronze "
            "Parquet and then removed."
        ),
    }
    _write_json(out_root / "quote_readiness" / "quote_route_report.json", quote_report)
    report_dir = out_root / "event_window_panel"
    report_dir.mkdir(parents=True, exist_ok=True)
    panel_report = {
        "pipeline_params": {
            "stage": "event-window-panel",
            "calendar": str(calendar_path),
            "dte_min": dte_min,
            "dte_max": dte_max,
            "ivar_support_dte_max": second_expiry_dte_max,
            "max_events": max_events,
        },
        "events": int(len(windows)),
        "events_with_rvar": int(windows["rvar_event"].notna().sum())
        if "rvar_event" in windows
        else 0,
        "events_with_entry_window": int(windows["entry_date"].notna().sum())
        if "entry_date" in windows
        else 0,
        "contracts": int(len(contracts)),
        "quote_pool_contracts": int(contracts["eligible_for_quote_pool"].sum())
        if "eligible_for_quote_pool" in contracts
        else 0,
        "main_dte_5_14_contracts": int(contracts["is_main_dte_5_14"].sum())
        if "is_main_dte_5_14" in contracts
        else 0,
        "robustness_dte_3_21_contracts": int(contracts["is_robustness_dte_3_21"].sum())
        if "is_robustness_dte_3_21" in contracts
        else 0,
        "ivar_support_only_contracts": int(contracts["is_ivar_support_only"].sum())
        if "is_ivar_support_only" in contracts
        else 0,
        "panel_grade": "event_window_contract_setup",
        "lake_layout": {
            "bronze": str(config.bronze_data_dir),
            "silver": str(config.silver_data_dir),
            "format": "parquet",
            "compression": PARQUET_COMPRESSION,
        },
        "outputs": {
            "calendar_main_sample": str(silver_calendar_dir / "main_sample.parquet"),
            "event_windows": str(silver_windows_dir / "event_windows.parquet"),
            "underlying_event_bars": str(silver_windows_dir / "underlying_event_bars.parquet"),
            "event_contract_candidates": str(
                silver_contracts_dir / "event_contract_candidates.parquet"
            ),
            "quote_route_report": str(out_root / "quote_readiness" / "quote_route_report.json"),
            "event_window_panel_report": str(report_dir / "event_window_panel_report.json"),
        },
    }
    _write_json(report_dir / "event_window_panel_report.json", panel_report)
    return panel_report

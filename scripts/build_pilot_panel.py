from __future__ import annotations

import argparse
import json
from collections.abc import Mapping
from datetime import date, timedelta
from pathlib import Path

import pandas as pd
import polars as pl
from scipy.optimize import brentq

from earnings_event_vol.backtest import black_scholes_price
from earnings_event_vol.config import ProjectConfig, load_project_config
from earnings_event_vol.event_targets import add_event_return_targets
from earnings_event_vol.events import regular_close_timestamp
from earnings_event_vol.massive import (
    _run_head_object_command,
    build_download_file_command,
    massive_flat_file_aws_env,
    option_flat_file_key,
    parse_massive_option_ticker,
    underlying_flat_file_key,
)
from earnings_event_vol.schemas import AnnouncementTiming, OptionRight
from earnings_event_vol.variance import (
    TotalVariancePoint,
    extract_implied_event_variance,
)

PARQUET_COMPRESSION = "zstd"


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
    offset = 0 if include_target else direction
    for _ in range(max_calendar_days + 1):
        candidate = target + timedelta(days=offset)
        if _ensure_underlying_file(config, candidate):
            return candidate
        offset += direction
    return None


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
    prefix = f"O:{ticker}"
    entry_date = pd.Timestamp(event["entry_date"]).date()
    exit_date = pd.Timestamp(event["exit_date"]).date()
    spot = float(event["s_before"])
    frame = pl.read_parquet(path)
    if "transactions" not in frame.columns:
        frame = frame.with_columns(pl.lit(0).alias("transactions"))
    frame = frame.filter(pl.col("ticker").str.starts_with(prefix))
    for row in frame.iter_rows(named=True):
        option_symbol = str(row["ticker"])
        try:
            parsed = parse_massive_option_ticker(option_symbol)
        except ValueError:
            continue
        expiration = parsed["expiration"]
        assert isinstance(expiration, date)
        dte = (expiration - entry_date).days
        if dte < dte_min or dte > dte_max or expiration < exit_date:
            continue
        close = float(row["close"]) if row["close"] is not None else float("nan")
        volume = int(float(row["volume"])) if row["volume"] is not None else 0
        if close <= 0 or volume <= 0:
            continue
        strike = float(parsed["strike"])
        in_requested_dte = dte <= int(event.get("requested_dte_max", dte_max))
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
                "transactions": int(float(row["transactions"] or 0)),
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
        strikes = (
            group[["strike", "moneyness_abs"]]
            .drop_duplicates()
            .sort_values(["moneyness_abs", "strike"])
            .head(strikes_per_expiry)["strike"]
        )
        selected.append(group.loc[group["strike"].isin(strikes)].copy())
    if not selected:
        return pd.DataFrame(columns=contracts.columns)
    return pd.concat(selected, ignore_index=True)


def _implied_volatility(
    *,
    spot: float,
    strike: float,
    time_to_expiry: float,
    option_price: float,
    right: str,
) -> tuple[float | None, str | None]:
    if spot <= 0 or strike <= 0 or time_to_expiry <= 0 or option_price <= 0:
        return None, "invalid_iv_inputs"
    option_right = OptionRight.CALL if right == "call" else OptionRight.PUT
    intrinsic = (
        max(spot - strike, 0.0) if option_right == OptionRight.CALL else max(strike - spot, 0.0)
    )
    if option_price < intrinsic:
        return None, "price_below_intrinsic"

    def objective(volatility: float) -> float:
        return (
            black_scholes_price(
                spot=spot,
                strike=strike,
                time_to_expiry=time_to_expiry,
                volatility=volatility,
                right=option_right,
            )
            - option_price
        )

    try:
        low = objective(1e-4)
        high = objective(5.0)
        if low * high > 0:
            return None, "iv_root_not_bracketed"
        return float(brentq(objective, 1e-4, 5.0, maxiter=100)), None
    except (ValueError, RuntimeError, OverflowError):
        return None, "iv_solver_failed"


def _attach_local_iv(contracts: pd.DataFrame, windows: pd.DataFrame) -> pd.DataFrame:
    spot_by_event = windows.set_index("event_id")["s_before"].to_dict()
    out = contracts.copy()
    local_ivs: list[float | None] = []
    statuses: list[str] = []
    for row in out.to_dict("records"):
        spot = float(spot_by_event[row["event_id"]])
        dte = int(row["dte"])
        iv, status = _implied_volatility(
            spot=spot,
            strike=float(row["strike"]),
            time_to_expiry=dte / 365.0,
            option_price=float(row["option_close"]),
            right=str(row["right"]),
        )
        local_ivs.append(iv)
        statuses.append(status or "ok")
    out["local_iv"] = local_ivs
    out["local_iv_status"] = statuses
    return out


def _event_expiry_ivar_inputs(iv_contracts: pd.DataFrame, windows: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    spot_by_event = windows.set_index("event_id")["s_before"].to_dict()
    for (event_id, expiration), group in iv_contracts.groupby(["event_id", "expiration"]):
        valid = group.loc[group["local_iv"].notna()].copy()
        if valid.empty:
            continue
        spot = float(spot_by_event[event_id])
        pair_rows: list[dict[str, object]] = []
        for strike, strike_group in valid.groupby("strike"):
            rights = set(strike_group["right"].astype(str))
            if {"call", "put"}.issubset(rights):
                pair_rows.append(
                    {
                        "strike": float(strike),
                        "atm_iv": float(strike_group["local_iv"].mean()),
                        "volume": int(strike_group["volume"].sum()),
                        "moneyness_abs": abs(float(strike) / spot - 1.0),
                        "selection_method": "call_put_average",
                    }
                )
        if pair_rows:
            selected = sorted(pair_rows, key=lambda item: (item["moneyness_abs"], -item["volume"]))[
                0
            ]
        else:
            fallback = valid.sort_values(["moneyness_abs", "volume"], ascending=[True, False]).iloc[
                0
            ]
            selected = {
                "strike": float(fallback["strike"]),
                "atm_iv": float(fallback["local_iv"]),
                "volume": int(fallback["volume"]),
                "moneyness_abs": float(fallback["moneyness_abs"]),
                "selection_method": f"single_{fallback['right']}",
            }
        event_row = windows.loc[windows["event_id"].eq(event_id)].iloc[0]
        rows.append(
            {
                "event_id": event_id,
                "ticker": event_row["ticker"],
                "event_date": event_row["announcement_date"],
                "event_exit_date": event_row["exit_date"],
                "entry_date": event_row["entry_date"],
                "expiration": expiration,
                "iv": selected["atm_iv"],
                "dte_days": (
                    pd.Timestamp(expiration).date() - pd.Timestamp(event_row["entry_date"]).date()
                ).days,
                "strike": selected["strike"],
                "moneyness": 1.0 + selected["moneyness_abs"],
                "spread_over_mid": None,
                "volume": selected["volume"],
                "atm_selection_method": selected["selection_method"],
                "quote_route": "options_day_aggs_close_proxy",
            }
        )
    return pd.DataFrame(rows)


def _extract_event_ivar(ivar_inputs: pd.DataFrame, windows: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    grouped = ivar_inputs.groupby("event_id") if not ivar_inputs.empty else []
    input_by_event = {event_id: group for event_id, group in grouped}
    for event in windows.to_dict("records"):
        event_id = event["event_id"]
        group = input_by_event.get(event_id, pd.DataFrame())
        group_records = (
            []
            if group.empty or "expiration" not in group.columns
            else group.sort_values("expiration").to_dict("records")
        )
        points = [
            TotalVariancePoint(
                expiration=pd.Timestamp(row["expiration"]).date(),
                iv=float(row["iv"]) if pd.notna(row["iv"]) else None,
                dte_days=int(row["dte_days"]),
                moneyness=float(row["moneyness"]) if pd.notna(row["moneyness"]) else None,
                spread_over_mid=None,
            )
            for row in group_records
        ]
        extraction = extract_implied_event_variance(
            points,
            event_date=pd.Timestamp(event["announcement_date"]).date(),
            event_exit_date=pd.Timestamp(event["exit_date"]).date(),
        )
        failure_reason = (
            extraction.failure_reason.value if extraction.failure_reason is not None else None
        )
        rows.append(
            {
                **event,
                "ivar_event": extraction.ivar_event,
                "ivar_failure_reason": failure_reason,
                "edge_var_realized": None
                if extraction.ivar_event is None
                else float(event["rvar_event"]) - extraction.ivar_event,
                "t1": extraction.t1,
                "t2": extraction.t2,
                "w1": extraction.w1,
                "w2": extraction.w2,
                "expiry_gap_days": extraction.expiry_gap_days,
                "iv_used_for_extraction_1": extraction.iv_used_for_extraction_1,
                "iv_used_for_extraction_2": extraction.iv_used_for_extraction_2,
                "dte_1": extraction.dte_1,
                "dte_2": extraction.dte_2,
                "expiration_1": extraction.expiration_1,
                "expiration_2": extraction.expiration_2,
                "quote_route": "options_day_aggs_close_proxy",
                "quote_status": "provisional_no_nbbo",
                "forward_source": "spot_fallback",
                "forward_price": event["s_before"],
                "american_forward_caveat_flag": False,
                "panel_grade": "provisional_no_nbbo",
            }
        )
    return pd.DataFrame(rows)


def build_pilot_panel(
    *,
    config: ProjectConfig,
    calendar_path: Path,
    out_root: Path,
    force: bool,
    dte_min: int,
    dte_max: int,
    strikes_per_expiry: int,
    max_events: int | None,
) -> dict[str, object]:
    second_expiry_dte_max = max(dte_max + 14, 28)
    silver_calendar_dir = config.silver_data_dir / "earnings_calendar"
    silver_windows_dir = config.silver_data_dir / "event_windows"
    silver_contracts_dir = config.silver_data_dir / "contracts"
    silver_ivar_dir = config.silver_data_dir / "ivar"
    gold_panel_dir = config.gold_data_dir / "event_panel"

    calendar = pd.read_csv(calendar_path)
    calendar["announcement_date"] = pd.to_datetime(calendar["announcement_date"]).dt.date
    main = calendar.loc[calendar["is_main_sample_candidate"].astype(bool)].copy()
    main["event_id"] = main.apply(_event_id, axis=1)
    if max_events is not None:
        main = main.head(max_events).copy()

    audit_dir = out_root / "calendar_audit"
    audit_dir.mkdir(parents=True, exist_ok=True)
    _write_parquet(silver_calendar_dir / "pilot_main_sample.parquet", main)
    local_ts = pd.to_datetime(main["source_timestamp"], utc=True, errors="coerce").dt.tz_convert(
        "America/New_York"
    )
    audit_report = {
        "candidate_rows": int(len(calendar)),
        "main_sample_rows": int(len(main)),
        "main_timing_counts": main["announcement_timing"].value_counts().to_dict(),
        "main_text_validation_counts": main["text_validation_status"].value_counts().to_dict(),
        "proxy_timing_rows": int(main["timing_confidence"].eq("proxy").sum()),
        "accepted_local_date_mismatch_rows": int(
            (local_ts.dt.date != main["announcement_date"]).sum()
        ),
        "timing_note": (
            "SEC acceptance timestamp is retained as a proxy audit field; this pilot panel is "
            "not yet a final first-release-time sample."
        ),
    }
    _write_json(audit_dir / "calendar_audit_report.json", audit_report)

    window_rows: list[dict[str, object]] = []
    for row in main.to_dict("records"):
        announcement_date = row["announcement_date"]
        timing = AnnouncementTiming(str(row["announcement_timing"]))
        if timing == AnnouncementTiming.AMC:
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
        exclusion_reason = None if entry_date and exit_date else "missing_entry_or_exit_date"
        window_rows.append(
            {
                **row,
                "entry_date": entry_date,
                "exit_date": exit_date,
                "feature_cutoff_date": entry_date,
                "requested_dte_min": dte_min,
                "requested_dte_max": dte_max,
                "ivar_support_dte_max": second_expiry_dte_max,
                "event_entry_timestamp": regular_close_timestamp(entry_date).isoformat()
                if entry_date
                else None,
                "exclusion_reason": exclusion_reason,
            }
        )
    windows = pd.DataFrame(window_rows)
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
    contracts = pd.DataFrame(contract_rows)
    if not contracts.empty:
        contracts = _select_near_atm_contracts(contracts, strikes_per_expiry=strikes_per_expiry)
    _write_parquet(silver_contracts_dir / "event_contract_candidates.parquet", contracts)

    iv_contracts = _attach_local_iv(contracts, windows) if not contracts.empty else contracts.copy()
    _write_parquet(silver_ivar_dir / "contract_iv_estimates.parquet", iv_contracts)
    ivar_inputs = (
        _event_expiry_ivar_inputs(iv_contracts, windows)
        if not iv_contracts.empty
        else pd.DataFrame()
    )
    _write_parquet(silver_ivar_dir / "ivar_inputs.parquet", ivar_inputs)
    panel = _extract_event_ivar(ivar_inputs, windows)
    panel_dir = out_root / "event_panel"
    panel_dir.mkdir(parents=True, exist_ok=True)
    _write_parquet(gold_panel_dir / "pilot_event_panel.parquet", panel)

    quote_report = {
        "nbbo_quote_route": "blocked",
        "rest_quotes_status": "not_entitled_in_current_massive_plan",
        "s3_select_status": "method_not_allowed_on_massive_flat_file_endpoint",
        "flat_file_full_scan_status": "deferred_too_large_for_pilot",
        "observed_quotes_v1_compressed_size_example_bytes": 115_992_798_867,
        "provisional_route_used": "options_day_aggs_close_proxy",
        "paper_grade": False,
        "lake_note": (
            "Downloaded Massive CSV.GZ day files are converted immediately into bronze "
            "Parquet and then removed."
        ),
    }
    _write_json(out_root / "quote_readiness" / "quote_route_report.json", quote_report)
    panel_report = {
        "pipeline_params": {
            "stage": "pilot-panel",
            "calendar": str(calendar_path),
            "dte_min": dte_min,
            "dte_max": dte_max,
            "ivar_support_dte_max": second_expiry_dte_max,
            "max_events": max_events,
        },
        "events": int(len(panel)),
        "events_with_rvar": int(panel["rvar_event"].notna().sum()),
        "events_with_ivar": int(panel["ivar_event"].notna().sum()),
        "ivar_failure_counts": panel["ivar_failure_reason"].fillna("ok").value_counts().to_dict(),
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
        "contracts_with_local_iv": int(iv_contracts["local_iv"].notna().sum())
        if not iv_contracts.empty
        else 0,
        "panel_grade": "provisional_no_nbbo",
        "lake_layout": {
            "bronze": str(config.bronze_data_dir),
            "silver": str(config.silver_data_dir),
            "gold": str(config.gold_data_dir),
            "format": "parquet",
            "compression": PARQUET_COMPRESSION,
        },
        "outputs": {
            "calendar_main_sample": str(silver_calendar_dir / "pilot_main_sample.parquet"),
            "event_windows": str(silver_windows_dir / "event_windows.parquet"),
            "underlying_event_bars": str(silver_windows_dir / "underlying_event_bars.parquet"),
            "event_contract_candidates": str(
                silver_contracts_dir / "event_contract_candidates.parquet"
            ),
            "contract_iv_estimates": str(silver_ivar_dir / "contract_iv_estimates.parquet"),
            "ivar_inputs": str(silver_ivar_dir / "ivar_inputs.parquet"),
            "pilot_event_panel": str(gold_panel_dir / "pilot_event_panel.parquet"),
            "quote_route_report": str(out_root / "quote_readiness" / "quote_route_report.json"),
            "pilot_panel_report": str(panel_dir / "pilot_panel_report.json"),
        },
    }
    _write_json(panel_dir / "pilot_panel_report.json", panel_report)
    return panel_report


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--calendar",
        type=Path,
        default=Path(
            "artifacts/data_pipeline/earnings_calendar_pilot/earnings_calendar_candidates.csv"
        ),
    )
    parser.add_argument("--out-root", type=Path, default=Path("artifacts/data_pipeline"))
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--dte-min", type=int, default=5)
    parser.add_argument("--dte-max", type=int, default=14)
    parser.add_argument("--strikes-per-expiry", type=int, default=3)
    parser.add_argument("--max-events", type=int)
    args = parser.parse_args()

    config = load_project_config()
    output = config.gold_data_dir / "event_panel" / "pilot_event_panel.parquet"
    report = args.out_root / "event_panel" / "pilot_panel_report.json"
    params = {
        "stage": "pilot-panel",
        "calendar": str(args.calendar),
        "dte_min": args.dte_min,
        "dte_max": args.dte_max,
        "max_events": args.max_events,
    }
    if not args.force and output.exists() and _json_params_match(report, params):
        print(
            json.dumps(
                {
                    "status": "skipped",
                    "reason": "outputs_exist_params_match",
                    "pilot_event_panel": str(output),
                    "pilot_panel_report": str(report),
                },
                indent=2,
            )
        )
        return 0
    panel_report = build_pilot_panel(
        config=config,
        calendar_path=args.calendar,
        out_root=args.out_root,
        force=args.force,
        dte_min=args.dte_min,
        dte_max=args.dte_max,
        strikes_per_expiry=args.strikes_per_expiry,
        max_events=args.max_events,
    )
    print(json.dumps(panel_report, indent=2, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

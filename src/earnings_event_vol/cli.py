from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, NoReturn, cast

import pandas as pd

from earnings_event_vol.backtest import (
    build_proxy_strategy_frame,
    expected_strategy_value_usd,
    market_entry_cost_usd,
    premium_space_signal,
)
from earnings_event_vol.config import ProjectConfig, load_project_config
from earnings_event_vol.data_audit import audit_data_fields
from earnings_event_vol.data_pipeline import (
    DEFAULT_STATIC_TICKERS,
    parse_text_list,
    run_data_pipeline,
)
from earnings_event_vol.earnings_calendar import build_earnings_calendar_candidates
from earnings_event_vol.event_panel import build_event_panel, discover_option_contracts
from earnings_event_vol.events import align_event_window, validate_calendar_frame
from earnings_event_vol.features import (
    DEFAULT_FEATURE_SCHEMA_VERSION,
    FEATURE_SCHEMA_VERSIONS,
    build_model_feature_matrix,
)
from earnings_event_vol.leakage_audit import audit_feature_leakage
from earnings_event_vol.massive import (
    build_massive_day_agg_sample,
    credential_probe,
    massive_flat_file_manifest,
)
from earnings_event_vol.metrics import (
    breakdown_metrics,
    edge_decile_table,
    forecast_metrics,
    ranking_metrics,
    strategy_metrics,
)
from earnings_event_vol.models import (
    MODEL_REGISTRY,
    add_benchmark_predictions,
    model_diagnostics_as_frame,
    prediction_column_for_model,
    run_model_suite,
)
from earnings_event_vol.research import remove_model_level_csv_artifacts, run_proxy_research_package
from earnings_event_vol.schemas import EarningsEvent, OptionRight, OptionSide, TradeLeg
from earnings_event_vol.variance import (
    TotalVariancePoint,
    edge_variance,
    extract_implied_event_variance,
    realized_event_variance,
)


def _path_status(path: Path | None) -> dict[str, object]:
    if path is None:
        return {"configured": False, "exists": False, "path": None}
    return {"configured": True, "exists": path.exists(), "path": str(path)}


def _print_json(payload: dict[str, object]) -> None:
    print(json.dumps(payload, indent=2, sort_keys=True, default=str))


def _status(config: ProjectConfig) -> int:
    payload: dict[str, object] = {
        "project_name": config.project_name,
        "repo_root": str(config.repo_root),
        "data_dir": str(config.data_dir),
        "reports_dir": str(config.reports_dir),
        "artifacts_dir": str(config.artifacts_dir),
        "massive": {
            "base_url": config.massive_base_url,
            "flat_file_endpoint_url": config.massive_flat_file_endpoint_url,
            "flat_file_bucket": config.massive_flat_file_bucket,
            "options_dataset": config.massive_options_flat_file_dataset,
            "api_key_file": _path_status(config.massive_api_key_file),
            "flat_file_key_file": _path_status(config.massive_flat_file_key_file),
        },
    }
    _print_json(payload)
    return 0


def _source_probe(source: str, config: ProjectConfig) -> int:
    if source not in {"all", "massive"}:
        _die(f"unsupported source: {source}")

    status = credential_probe(config)
    payload: dict[str, object] = {"source": "massive", "status": status}
    _print_json(payload)
    return 0 if status["ok"] else 1


def _read_csv(path: Path) -> pd.DataFrame:
    return pd.read_csv(path)


def _read_table(path: Path) -> pd.DataFrame:
    if path.suffix.lower() == ".parquet":
        return pd.read_parquet(path)
    return pd.read_csv(path)


def _write_table(path: Path, frame: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.suffix.lower() == ".parquet":
        frame.to_parquet(path, index=False)
    else:
        frame.to_csv(path, index=False)


def _write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True, default=str), encoding="utf-8")


def _audit_data(args: argparse.Namespace) -> int:
    quotes = _read_csv(args.quotes)
    underlying = _read_csv(args.underlying)
    earnings = _read_csv(args.earnings)
    out = Path(args.out)
    result = audit_data_fields(
        options=quotes,
        underlying=underlying,
        earnings=earnings,
        source_paths=[args.quotes, args.underlying, args.earnings],
    )
    out.mkdir(parents=True, exist_ok=True)
    _write_json(out / "required_fields_report.json", result.required_fields_report)
    result.field_coverage.to_csv(out / "field_coverage.csv", index=False)
    result.vendor_local_iv_diff.to_csv(out / "vendor_local_iv_diff.csv", index=False)
    result.quote_source_report.to_csv(out / "quote_source_report.csv", index=False)
    _print_json({"ok": result.required_fields_report["ok"], "out": str(out)})
    return 0 if bool(result.required_fields_report["ok"]) else 1


def _massive_flat_files(args: argparse.Namespace, config: ProjectConfig) -> int:
    date_value = pd.Timestamp(args.date).date()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    if args.metadata_only:
        manifest = massive_flat_file_manifest(
            config,
            date_value=date_value,
            run_head=not args.no_head,
            aws_executable=args.aws_executable,
        )
        sample_report: dict[str, Any] | None = None
    else:
        sample_report = build_massive_day_agg_sample(
            config,
            date_value=date_value,
            out_dir=out,
            sample_rows=args.sample_rows,
            aws_executable=args.aws_executable,
        )
        manifest = cast(dict[str, Any], sample_report["manifest"])
    _write_json(out / "massive_flat_file_manifest.json", manifest)
    pd.DataFrame(manifest["objects"]).to_csv(out / "massive_flat_file_objects.csv", index=False)
    objects = manifest["objects"]
    ok = all(bool(item["ok"]) for item in objects) if not args.no_head else True
    _print_json(
        {
            "ok": ok,
            "date": manifest["date"],
            "objects": len(objects),
            "sample_rows": 0 if sample_report is None else args.sample_rows,
            "out": str(out),
        }
    )
    return 0 if ok else 1


def _validate_calendar(args: argparse.Namespace) -> int:
    frame = validate_calendar_frame(_read_csv(args.input))
    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        frame.to_csv(args.out, index=False)
    _print_json(
        {
            "rows": int(len(frame)),
            "main_sample_timing_rows": int(frame["is_main_sample_timing"].sum()),
            "excluded_timing_rows": int((~frame["is_main_sample_timing"]).sum()),
        }
    )
    return 0


def _parse_tickers(values: list[str]) -> list[str]:
    tickers: list[str] = []
    for value in values:
        tickers.extend(part.strip().upper() for part in value.split(",") if part.strip())
    return sorted(set(tickers))


def _build_earnings_calendar(args: argparse.Namespace, config: ProjectConfig) -> int:
    tickers = _parse_tickers(args.tickers)
    start_date = pd.Timestamp(args.start).date()
    end_date = pd.Timestamp(args.end).date()
    out = Path(args.out)
    frame, report = build_earnings_calendar_candidates(
        config=config,
        tickers=tickers,
        start_date=start_date,
        end_date=end_date,
        sec_submissions_dir=args.sec_submissions_dir,
        massive_8k_text_dir=args.massive_8k_text_dir,
        validate_with_massive=not args.skip_massive_validation,
    )
    out.mkdir(parents=True, exist_ok=True)
    frame.to_csv(out / "earnings_calendar_candidates.csv", index=False)
    _write_json(out / "earnings_calendar_report.json", report)
    _print_json(
        {
            "rows": int(report["row_count"]),
            "main_sample_candidate_rows": int(report["main_sample_candidate_rows"]),
            "out": str(out),
        }
    )
    return 0


def _align_events(args: argparse.Namespace) -> int:
    earnings = validate_calendar_frame(_read_csv(args.earnings))
    underlying = _read_csv(args.underlying)
    underlying["date"] = pd.to_datetime(underlying["date"]).dt.date
    trading_dates = underlying["date"].tolist()
    windows = []
    for row in earnings.to_dict("records"):
        event = EarningsEvent(
            ticker=row["ticker"],
            announcement_date=row["announcement_date"],
            announcement_timing=row["announcement_timing"],
            source=row["source"],
            sector=row.get("sector"),
        )
        windows.append(align_event_window(event, trading_dates).model_dump())
    out = pd.DataFrame(windows)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(args.out, index=False)
    _print_json({"rows": int(len(out)), "out": str(args.out)})
    return 0


def _compute_variance(args: argparse.Namespace) -> int:
    iv = _read_csv(args.ivar_input)
    prices = _read_csv(args.prices)
    rows = []
    for (ticker, event_date_raw), group in iv.groupby(["ticker", "event_date"]):
        event_date = pd.Timestamp(event_date_raw).date()
        event_exit_date = (
            pd.Timestamp(group["event_exit_date"].dropna().iloc[0]).date()
            if "event_exit_date" in group.columns and not group["event_exit_date"].dropna().empty
            else None
        )
        points = [
            TotalVariancePoint(
                expiration=pd.Timestamp(row["expiration"]).date(),
                iv=None if pd.isna(row["iv"]) else float(row["iv"]),
                dte_days=int(row["dte_days"]),
                stale=bool(row.get("stale", False)),
            )
            for row in group.to_dict("records")
        ]
        extraction = extract_implied_event_variance(
            points, event_date=event_date, event_exit_date=event_exit_date
        )
        price_matches = prices.loc[
            (prices["ticker"] == ticker)
            & (pd.to_datetime(prices["event_date"]).dt.date == event_date)
        ]
        if price_matches.empty:
            raise ValueError(f"missing event price row for {ticker} {event_date}")
        if len(price_matches) > 1:
            raise ValueError(f"duplicate event price rows for {ticker} {event_date}")
        price_row = price_matches.iloc[0]
        rvar = realized_event_variance(float(price_row["s_before"]), float(price_row["s_after"]))
        ivar = extraction.ivar_event
        rows.append(
            {
                "ticker": ticker,
                "event_date": event_date,
                "rvar_event": rvar,
                "ivar_event": ivar,
                "edge_var": None if ivar is None else edge_variance(rvar, ivar),
                "failure_reason": None
                if extraction.failure_reason is None
                else extraction.failure_reason.value,
                "t1": extraction.t1,
                "t2": extraction.t2,
                "w1": extraction.w1,
                "w2": extraction.w2,
                "expiration_gap_days": extraction.expiration_gap_days,
                "expiry_gap_days": extraction.expiry_gap_days,
                "iv_used_for_extraction_1": extraction.iv_used_for_extraction_1,
                "iv_used_for_extraction_2": extraction.iv_used_for_extraction_2,
                "dte_1": extraction.dte_1,
                "dte_2": extraction.dte_2,
                "expiration_1": extraction.expiration_1,
                "expiration_2": extraction.expiration_2,
                "spread_over_mid_1": extraction.spread_over_mid_1,
                "spread_over_mid_2": extraction.spread_over_mid_2,
            }
        )
    out = pd.DataFrame(rows)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(args.out, index=False)
    _print_json({"rows": int(len(out)), "out": str(args.out)})
    return 0


def _discover_option_contracts(args: argparse.Namespace) -> int:
    events = _read_csv(args.events)
    contracts = _read_csv(args.contracts)
    out = discover_option_contracts(
        events,
        contracts,
        dte_min=args.dte_min,
        dte_max=args.dte_max,
    )
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(args.out, index=False)
    _print_json(
        {
            "rows": int(len(out)),
            "eligible_for_quote_pool": int(out["eligible_for_quote_pool"].sum()),
            "non_standard_excluded": int(
                out["contract_discovery_status"].eq("non_standard_excluded").sum()
            ),
            "out": str(args.out),
        }
    )
    return 0


def _build_event_panel(args: argparse.Namespace) -> int:
    events = _read_csv(args.events)
    quotes = _read_csv(args.quotes)
    ex_dividends = _read_csv(args.ex_dividends) if args.ex_dividends else None
    out = build_event_panel(
        events,
        quotes,
        ex_dividends=ex_dividends,
        dte_min=args.dte_min,
        dte_max=args.dte_max,
    )
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(args.out, index=False)
    _print_json(
        {
            "rows": int(len(out)),
            "put_call_parity_forward_rows": int(out["forward_source"].eq("put_call_parity").sum()),
            "spot_fallback_rows": int(out["forward_source"].eq("spot_fallback").sum()),
            "possible_preannouncement_or_prior_guidance": int(
                out["possible_preannouncement_or_prior_guidance"].sum()
            ),
            "out": str(args.out),
        }
    )
    return 0


def _data(args: argparse.Namespace, config: ProjectConfig) -> int:
    out_dir = args.out_dir or config.artifacts_dir / "data_pipeline"
    payload = run_data_pipeline(
        config,
        stage=args.stage,
        out_root=out_dir,
        force=args.force,
        jobs=args.jobs,
        tickers=parse_text_list(args.tickers) or list(DEFAULT_STATIC_TICKERS),
        start_date=pd.Timestamp(args.start).date(),
        end_date=pd.Timestamp(args.end).date(),
        dates=[pd.Timestamp(value).date() for value in parse_text_list(args.dates)],
        events_path=args.events,
        contracts_path=args.contracts,
        quotes_path=args.quotes,
        options_day_aggs_path=args.options_day_aggs,
        ex_dividends_path=args.ex_dividends,
        sec_submissions_dir=args.sec_submissions_dir,
        massive_8k_text_dir=args.massive_8k_text_dir,
        validate_with_massive=not args.skip_massive_validation,
        dte_min=args.dte_min,
        dte_max=args.dte_max,
        max_events=args.max_events,
        max_contracts=args.max_contracts,
        download_samples=args.download_samples,
        lookback_seconds=args.lookback_seconds,
        second_agg_buffer_minutes=args.second_agg_buffer_minutes,
        price_field=args.price_field,
        dry_run=args.dry_run,
        universe_top_n=args.universe_top_n,
        universe_trailing_months=args.universe_trailing_months,
        refresh_bronze=args.refresh_bronze,
    )
    _print_json(payload)
    return 0 if bool(payload["ok"]) else 1


def _build_feature_matrix(args: argparse.Namespace, config: ProjectConfig) -> int:
    panel_path = (
        args.panel or config.gold_data_dir / "event_panel" / "trade_proxy_event_panel.parquet"
    )
    straddles_path = (
        args.straddles
        or config.artifacts_dir
        / "data_pipeline"
        / "trade_proxy_panel"
        / "trade_proxy_straddle_diagnostics.csv"
    )
    out_path = args.out or config.gold_data_dir / "modeling" / "feature_matrix.parquet"
    panel = _read_table(panel_path)
    straddles = _read_table(straddles_path) if straddles_path.exists() else None
    features = build_model_feature_matrix(panel, straddle_diagnostics=straddles)
    features = add_benchmark_predictions(features)
    _write_table(out_path, features)
    payload = {
        "rows": int(len(features)),
        "columns": int(len(features.columns)),
        "out": str(out_path),
        "target": "rvar_event",
        "market_baseline": "ivar_event",
        "prediction_columns": [
            column for column in features.columns if column.startswith("forecast_")
        ],
    }
    _print_json(payload)
    return 0


def _model_ids_from_args(values: list[str]) -> list[str]:
    if not values or values == ["all"]:
        return [
            "market_implied_event_variance",
            "last_four_rvar",
            "last_four_ivar",
            "goyal_saretto_rv_iv_spread",
        ]
    parsed: list[str] = []
    for value in values:
        parsed.extend(part.strip() for part in value.split(",") if part.strip())
    unknown = sorted(set(parsed) - set(MODEL_REGISTRY))
    if unknown:
        raise ValueError(f"unknown model ids: {unknown}")
    return parsed


def _train_models(args: argparse.Namespace, config: ProjectConfig) -> int:
    features_path = args.features or config.gold_data_dir / "modeling" / "feature_matrix.parquet"
    features = _read_table(features_path)
    model_ids = _model_ids_from_args(args.models)
    predictions, fit_results = run_model_suite(
        features,
        model_ids=model_ids,
        split_date=args.split_date,
    )
    out = args.out or config.artifacts_dir / "modeling"
    out.mkdir(parents=True, exist_ok=True)
    remove_model_level_csv_artifacts(out)
    predictions_path = out / "model_predictions.parquet"
    predictions.to_parquet(predictions_path, index=False)
    diagnostics = model_diagnostics_as_frame(fit_results)
    diagnostics.to_csv(out / "model_fit_diagnostics.csv", index=False)

    forecast_rows: list[dict[str, object]] = []
    ranking_rows: list[dict[str, object]] = []
    strategy_rows: list[dict[str, object]] = []
    breakdown_frames: list[pd.DataFrame] = []
    for model_id in model_ids:
        if model_id == "patell_wolfson_diagnostic":
            continue
        column = prediction_column_for_model(model_id)
        if column not in predictions.columns:
            continue
        scored = predictions.copy()
        scored[f"score_{model_id}"] = pd.to_numeric(
            scored[column], errors="coerce"
        ) - pd.to_numeric(scored["ivar_event"], errors="coerce")
        forecast_rows.append(
            {
                "model_id": model_id,
                **forecast_metrics(scored, forecast_col=column),
            }
        )
        if "edge_var_realized" in scored.columns:
            ranking_rows.append(
                {
                    "model_id": model_id,
                    **ranking_metrics(scored, score_col=f"score_{model_id}"),
                }
            )
            edge_decile_table(
                scored,
                score_col=f"score_{model_id}",
            ).to_csv(out / f"edge_deciles_{model_id}.csv", index=False)
        if {"gross_proxy_pnl_usd", "entry_premium_usd"}.issubset(scored.columns):
            strategy_frame = build_proxy_strategy_frame(
                scored,
                forecast_col=column,
                min_edge_var=args.min_edge_var,
            )
            trades = strategy_frame.loc[strategy_frame["should_trade"].astype(bool)].copy()
            trades.to_csv(out / f"strategy_trades_{model_id}.csv", index=False)
            strategy_rows.append(
                {
                    "model_id": model_id,
                    **strategy_metrics(trades),
                }
            )
            for breakdown in (
                "is_main_dte_5_14",
                "announcement_timing",
                "event_year",
                "regime",
                "ticker",
            ):
                if breakdown in trades.columns and not trades.empty:
                    frame = breakdown_metrics(
                        trades,
                        by=[breakdown],
                        forecast_col=column,
                    )
                    frame.insert(0, "model_id", model_id)
                    frame.insert(1, "breakdown", breakdown)
                    breakdown_frames.append(frame)

    forecast_frame = pd.DataFrame(forecast_rows)
    ranking_frame = pd.DataFrame(ranking_rows)
    strategy_frame = pd.DataFrame(strategy_rows)
    forecast_frame.to_csv(out / "forecast_metrics.csv", index=False)
    ranking_frame.to_csv(out / "ranking_metrics.csv", index=False)
    strategy_frame.to_csv(out / "strategy_metrics.csv", index=False)
    if breakdown_frames:
        pd.concat(breakdown_frames, ignore_index=True).to_csv(
            out / "strategy_breakdowns.csv", index=False
        )
    payload = {
        "ok": True,
        "features": str(features_path),
        "out": str(out),
        "models": model_ids,
        "prediction_rows": int(len(predictions)),
        "outputs": {
            "predictions": str(predictions_path),
            "forecast_metrics": str(out / "forecast_metrics.csv"),
            "ranking_metrics": str(out / "ranking_metrics.csv"),
            "strategy_metrics": str(out / "strategy_metrics.csv"),
            "model_fit_diagnostics": str(out / "model_fit_diagnostics.csv"),
        },
    }
    _write_json(out / "modeling_report.json", payload)
    _print_json(payload)
    return 0


def _research(args: argparse.Namespace, config: ProjectConfig) -> int:
    payload = run_proxy_research_package(
        config,
        stage=args.stage,
        split_design=args.split_design,
        split_date=args.split_date,
        allow_high_sequence_risk=args.allow_high_sequence_risk,
        sequence_suite=args.sequence_suite,
        mamba_backend=args.mamba_backend,
        mamba_seeds=_parse_comma_ints(args.mamba_seeds),
        bootstrap_iter=args.bootstrap_iter,
        tuning_profile=args.tuning_profile,
        tuning_seed=args.tuning_seed,
        feature_schema_version=args.feature_schema_version,
    )
    _print_json(payload)
    return 0 if bool(payload["ok"]) else 1


def _parse_comma_ints(raw: str) -> list[int]:
    values: list[int] = []
    for piece in str(raw).split(","):
        value = piece.strip()
        if not value:
            continue
        values.append(int(value))
    if not values:
        raise ValueError("expected at least one integer seed")
    return values


def _leakage_audit(args: argparse.Namespace) -> int:
    frame = _read_csv(args.features)
    result = audit_feature_leakage(frame)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    result.asof_violations.to_csv(out / "asof_violations.csv", index=False)
    _write_json(
        out / "leakage_report.json",
        {
            "ok": result.ok,
            "blocked_columns": result.blocked_columns,
            "vendor_forecast_columns": result.vendor_forecast_columns,
        },
    )
    _print_json({"ok": result.ok, "out": str(out)})
    return 0 if result.ok else 1


def _backtest_smoke(args: argparse.Namespace) -> int:
    legs_frame = _read_csv(args.legs)
    signal_frame = _read_csv(args.signals)
    first_signal = signal_frame.iloc[0]
    legs = tuple(
        TradeLeg(
            ticker=row["ticker"],
            expiration=pd.Timestamp(row["expiration"]).date(),
            strike=float(row["strike"]),
            right=OptionRight(str(row["right"]).lower()),
            side=OptionSide(str(row["side"]).lower()),
            contracts=float(row["contracts"]),
            filled_price=float(row["filled_price"]),
            filled_timestamp=pd.Timestamp(row["filled_timestamp"]).to_pydatetime(),
        )
        for row in legs_frame.to_dict("records")
    )
    expected_value = expected_strategy_value_usd(
        spot=float(first_signal["spot"]),
        forecast_rvar_event=float(first_signal["forecast_rvar_event"]),
        legs=legs,
    )
    entry_cost = market_entry_cost_usd(legs)
    signal = premium_space_signal(
        ticker=str(first_signal["ticker"]),
        event_date=pd.Timestamp(first_signal["event_date"]).date(),
        strategy=str(first_signal["strategy"]),
        forecast_rvar_event=float(first_signal["forecast_rvar_event"]),
        ivar_event=float(first_signal["ivar_event"]),
        expected_value_usd=expected_value,
        entry_cost_usd=entry_cost,
        transaction_cost_usd=float(first_signal["estimated_transaction_cost_usd"]),
    )
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    _write_json(out / "backtest_smoke_signal.json", signal.model_dump())
    _print_json({"should_trade": signal.should_trade, "out": str(out)})
    return 0


def _die(message: str) -> NoReturn:
    raise SystemExit(message)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="earnings-event-vol")
    subparsers = parser.add_subparsers(dest="command", required=True)

    subparsers.add_parser("status", help="Print active project configuration without secrets.")

    source_probe = subparsers.add_parser(
        "source-probe", help="Check configured vendor credential file paths."
    )
    source_probe.add_argument("source", choices=["all", "massive"], nargs="?", default="all")

    audit_data = subparsers.add_parser("audit-data", help="Audit required data fields.")
    audit_data.add_argument("--quotes", type=Path, required=True)
    audit_data.add_argument("--underlying", type=Path, required=True)
    audit_data.add_argument("--earnings", type=Path, required=True)
    audit_data.add_argument("--out", type=Path, required=True)

    flat_files = subparsers.add_parser(
        "massive-flat-files", help="Probe Massive S3 flat-file object metadata."
    )
    flat_files.add_argument("--date", required=True, help="Trading date in YYYY-MM-DD format.")
    flat_files.add_argument("--out", type=Path, required=True)
    flat_files.add_argument("--aws-executable", default="aws")
    flat_files.add_argument("--sample-rows", type=int, default=25)
    flat_files.add_argument(
        "--no-head", action="store_true", help="Build manifest without S3 calls."
    )
    flat_files.add_argument(
        "--metadata-only", action="store_true", help="Skip small day-aggregate sample download."
    )

    validate_calendar = subparsers.add_parser("validate-calendar", help="Validate earnings timing.")
    validate_calendar.add_argument("--input", type=Path, required=True)
    validate_calendar.add_argument("--out", type=Path)

    build_calendar = subparsers.add_parser(
        "build-earnings-calendar",
        help=(
            "Build SEC-first earnings candidates with SEC primary-document text "
            "validation and optional Massive auxiliary text fallback."
        ),
    )
    build_calendar.add_argument("--tickers", nargs="+", required=True)
    build_calendar.add_argument("--start", required=True, help="Start date in YYYY-MM-DD format.")
    build_calendar.add_argument("--end", required=True, help="End date in YYYY-MM-DD format.")
    build_calendar.add_argument("--out", type=Path, required=True)
    build_calendar.add_argument("--sec-submissions-dir", type=Path)
    build_calendar.add_argument("--massive-8k-text-dir", type=Path)
    build_calendar.add_argument(
        "--skip-massive-validation",
        action="store_true",
        help=(
            "Skip Massive auxiliary text fallback; live SEC primary-document validation "
            "still runs when SEC HTTP sources are used."
        ),
    )

    align_events = subparsers.add_parser(
        "align-events", help="Align BMO/AMC events to EOD windows."
    )
    align_events.add_argument("--earnings", type=Path, required=True)
    align_events.add_argument("--underlying", type=Path, required=True)
    align_events.add_argument("--out", type=Path, required=True)

    compute_variance = subparsers.add_parser("compute-variance", help="Compute RVAR and IVAR.")
    compute_variance.add_argument("--ivar-input", type=Path, required=True)
    compute_variance.add_argument("--prices", type=Path, required=True)
    compute_variance.add_argument("--out", type=Path, required=True)

    discover_contracts = subparsers.add_parser(
        "discover-option-contracts",
        help="Map events to candidate contracts and filter non-standard OCC deliverables.",
    )
    discover_contracts.add_argument("--events", type=Path, required=True)
    discover_contracts.add_argument("--contracts", type=Path, required=True)
    discover_contracts.add_argument("--out", type=Path, required=True)
    discover_contracts.add_argument("--dte-min", type=int, default=5)
    discover_contracts.add_argument("--dte-max", type=int, default=14)

    event_panel = subparsers.add_parser(
        "build-event-panel",
        help="Attach forward/ATM diagnostics and preannouncement review flags.",
    )
    event_panel.add_argument("--events", type=Path, required=True)
    event_panel.add_argument("--quotes", type=Path, required=True)
    event_panel.add_argument("--out", type=Path, required=True)
    event_panel.add_argument("--ex-dividends", type=Path)
    event_panel.add_argument("--dte-min", type=int, default=5)
    event_panel.add_argument("--dte-max", type=int, default=14)

    data = subparsers.add_parser(
        "data",
        help="Run resumable data-engineering stages behind one manifest.",
    )
    data.add_argument(
        "--stage",
        choices=[
            "all",
            "proxy-all",
            "fixture-audit",
            "massive-probe",
            "market-covariates",
            "market-second-covariates",
            "sec-companyfacts",
            "options-day-aggs-bulk",
            "universe",
            "dynamic-calendar",
            "contracts",
            "contract-reference-validation",
            "panel",
            "event-window-panel",
            "trade-proxy-panel",
        ],
        default="proxy-all",
    )
    data.add_argument("--out-dir", "--out-root", dest="out_dir", type=Path)
    data.add_argument("--force", action="store_true")
    data.add_argument("--jobs", type=int, default=1)
    data.add_argument(
        "--tickers",
        nargs="*",
        default=list(DEFAULT_STATIC_TICKERS),
        help="Ticker list; accepts repeated, comma-separated, or space-separated values.",
    )
    data.add_argument("--start", default="2022-12-01")
    data.add_argument("--end", default="2025-12-31")
    data.add_argument("--dates", nargs="*", default=[])
    data.add_argument("--events", type=Path)
    data.add_argument("--contracts", type=Path)
    data.add_argument("--quotes", type=Path)
    data.add_argument("--options-day-aggs", type=Path)
    data.add_argument("--ex-dividends", type=Path)
    data.add_argument("--sec-submissions-dir", type=Path)
    data.add_argument("--massive-8k-text-dir", type=Path)
    data.add_argument(
        "--skip-massive-validation",
        action="store_true",
        help=(
            "Skip Massive auxiliary 8-K text fallback; live SEC primary-document "
            "validation remains the default calendar route."
        ),
    )
    data.add_argument("--dte-min", type=int, default=5)
    data.add_argument("--dte-max", type=int, default=14)
    data.add_argument("--max-events", type=int)
    data.add_argument("--max-contracts", type=int)
    data.add_argument("--download-samples", action="store_true")
    data.add_argument("--dry-run", action="store_true")
    data.add_argument(
        "--refresh-bronze",
        action="store_true",
        help="Re-fetch bronze second-agg caches instead of reusing valid cached Parquet.",
    )
    data.add_argument("--lookback-seconds", type=int, default=900)
    data.add_argument("--second-agg-buffer-minutes", type=int, default=60)
    data.add_argument("--universe-top-n", type=int, default=50)
    data.add_argument("--universe-trailing-months", type=int, default=6)
    data.add_argument(
        "--price-field",
        choices=["option_vwap", "option_close"],
        default="option_vwap",
        help="Trade-proxy option price field from second aggregates.",
    )

    build_features = subparsers.add_parser(
        "build-feature-matrix", help="Build event-level feature matrix for models."
    )
    build_features.add_argument(
        "--panel",
        type=Path,
    )
    build_features.add_argument(
        "--straddles",
        type=Path,
    )
    build_features.add_argument(
        "--out",
        type=Path,
    )

    train_models = subparsers.add_parser(
        "train-models", help="Train benchmarks/models and write forecast/ranking/strategy metrics."
    )
    train_models.add_argument(
        "--features",
        type=Path,
    )
    train_models.add_argument("--out", type=Path)
    train_models.add_argument("--models", nargs="*", default=["all"])
    train_models.add_argument("--split-date", default=None)
    train_models.add_argument("--min-edge-var", type=float, default=0.0)

    research = subparsers.add_parser(
        "research",
        help="Build the no-NBBO proxy research package without downloading market data.",
    )
    research.add_argument(
        "--stage",
        choices=["all", "sequence-audit", "features", "models", "report"],
        default="all",
    )
    research.add_argument(
        "--split-design",
        default="chronological_proxy_70_15_15",
        choices=["chronological_proxy_70_15_15"],
    )
    research.add_argument("--split-date", default=None)
    research.add_argument("--allow-high-sequence-risk", action="store_true")
    research.add_argument(
        "--sequence-suite",
        choices=["none", "all"],
        default="all",
        help="Sequence diagnostics to run; use none to skip sequence models.",
    )
    research.add_argument("--mamba-backend", choices=["mamba_ssm"], default="mamba_ssm")
    research.add_argument("--mamba-seeds", default="17")
    research.add_argument("--bootstrap-iter", type=int, default=200)
    research.add_argument(
        "--tuning-profile",
        choices=["tuned_phase1"],
        default="tuned_phase1",
        help=argparse.SUPPRESS,
    )
    research.add_argument("--tuning-seed", type=int, default=17)
    research.add_argument(
        "--feature-schema-version",
        choices=list(FEATURE_SCHEMA_VERSIONS),
        default=DEFAULT_FEATURE_SCHEMA_VERSION,
        help="Feature schema/allowlist version; defaults to the current FE V2 schema.",
    )

    leakage_audit = subparsers.add_parser("leakage-audit", help="Audit feature leakage.")
    leakage_audit.add_argument("--features", type=Path, required=True)
    leakage_audit.add_argument("--out", type=Path, required=True)

    backtest_smoke = subparsers.add_parser(
        "backtest-smoke", help="Run deterministic backtest smoke."
    )
    backtest_smoke.add_argument("--legs", type=Path, required=True)
    backtest_smoke.add_argument("--signals", type=Path, required=True)
    backtest_smoke.add_argument("--out", type=Path, required=True)

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    config = load_project_config()

    if args.command == "status":
        return _status(config)
    if args.command == "source-probe":
        return _source_probe(args.source, config)
    if args.command == "audit-data":
        return _audit_data(args)
    if args.command == "massive-flat-files":
        return _massive_flat_files(args, config)
    if args.command == "validate-calendar":
        return _validate_calendar(args)
    if args.command == "build-earnings-calendar":
        return _build_earnings_calendar(args, config)
    if args.command == "align-events":
        return _align_events(args)
    if args.command == "compute-variance":
        return _compute_variance(args)
    if args.command == "discover-option-contracts":
        return _discover_option_contracts(args)
    if args.command == "build-event-panel":
        return _build_event_panel(args)
    if args.command == "data":
        return _data(args, config)
    if args.command == "build-feature-matrix":
        return _build_feature_matrix(args, config)
    if args.command == "train-models":
        return _train_models(args, config)
    if args.command == "research":
        return _research(args, config)
    if args.command == "leakage-audit":
        return _leakage_audit(args)
    if args.command == "backtest-smoke":
        return _backtest_smoke(args)
    _die(f"unsupported command: {args.command}")


if __name__ == "__main__":
    raise SystemExit(main())

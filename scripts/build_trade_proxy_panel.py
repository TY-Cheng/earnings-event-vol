from __future__ import annotations

import argparse
import json
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import pandas as pd
import polars as pl

from earnings_event_vol.config import ProjectConfig, load_project_config
from earnings_event_vol.trade_proxy import (
    TRADE_PROXY_PANEL_GRADE,
    TRADE_PROXY_STATUS_FETCH_FAILED,
    attach_trade_proxy_local_iv,
    build_proxy_straddle_diagnostics,
    build_trade_proxy_ivar_inputs,
    build_trade_proxy_price_frame,
    edge_decile_diagnostics,
    extract_trade_proxy_event_panel,
    fetch_massive_option_second_aggregates,
    normalize_second_aggregates,
    summarize_trade_proxy_panel,
)

PARQUET_COMPRESSION = "zstd"


def _write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")


def _write_parquet(path: Path, frame: pd.DataFrame | pl.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    pl_frame = pl.from_pandas(frame) if isinstance(frame, pd.DataFrame) else frame
    pl_frame.write_parquet(path, compression=PARQUET_COMPRESSION)


def _read_parquet(path: Path) -> pd.DataFrame:
    return pl.read_parquet(path).to_pandas()


def _fetch_one_contract(
    config: ProjectConfig,
    *,
    option_ticker: str,
    entry_date: pd.Timestamp,
    limit: int,
) -> tuple[str, pd.DataFrame, dict[str, object]]:
    try:
        raw = fetch_massive_option_second_aggregates(
            config,
            option_ticker=option_ticker,
            trade_date=entry_date.date(),
            limit=limit,
        )
        normalized = normalize_second_aggregates(raw, option_ticker=option_ticker)
        return (
            option_ticker,
            normalized,
            {
                "options_ticker": option_ticker,
                "status": "ok",
                "rows": int(len(normalized)),
                "entry_date": entry_date.date(),
            },
        )
    except Exception as exc:
        return (
            option_ticker,
            pd.DataFrame(),
            {
                "options_ticker": option_ticker,
                "status": TRADE_PROXY_STATUS_FETCH_FAILED,
                "rows": 0,
                "entry_date": entry_date.date(),
                "error": str(exc)[:300],
            },
        )


def _fetch_second_aggregate_bars(
    config: ProjectConfig,
    contracts: pd.DataFrame,
    *,
    jobs: int,
    limit: int,
) -> tuple[dict[str, pd.DataFrame], pd.DataFrame]:
    requests = (
        contracts[["options_ticker", "entry_date"]]
        .dropna()
        .drop_duplicates()
        .sort_values(["entry_date", "options_ticker"])
    )
    bar_frames: dict[str, pd.DataFrame] = {}
    reports: list[dict[str, object]] = []
    if jobs <= 1:
        for row in requests.to_dict("records"):
            option_ticker, frame, report = _fetch_one_contract(
                config,
                option_ticker=str(row["options_ticker"]),
                entry_date=pd.Timestamp(row["entry_date"]),
                limit=limit,
            )
            bar_frames[option_ticker] = frame
            reports.append(report)
        return bar_frames, pd.DataFrame(reports)

    with ThreadPoolExecutor(max_workers=jobs) as executor:
        futures = [
            executor.submit(
                _fetch_one_contract,
                config,
                option_ticker=str(row["options_ticker"]),
                entry_date=pd.Timestamp(row["entry_date"]),
                limit=limit,
            )
            for row in requests.to_dict("records")
        ]
        for future in as_completed(futures):
            option_ticker, frame, report = future.result()
            bar_frames[option_ticker] = frame
            reports.append(report)
    return bar_frames, pd.DataFrame(reports).sort_values(["entry_date", "options_ticker"])


def build_trade_proxy_panel(
    *,
    config: ProjectConfig,
    out_root: Path,
    force: bool,
    max_events: int | None,
    max_contracts: int | None,
    lookback_seconds: int,
    price_field: str,
    jobs: int,
    rest_limit: int,
    haircut_fraction: float,
) -> dict[str, object]:
    windows_path = config.silver_data_dir / "event_windows" / "event_windows.parquet"
    contracts_path = config.silver_data_dir / "contracts" / "event_contract_candidates.parquet"
    if not windows_path.exists() or not contracts_path.exists():
        raise FileNotFoundError(
            "trade-proxy-panel requires an existing pilot-panel run with event windows and "
            "candidate contracts."
        )

    silver_proxy_dir = config.silver_data_dir / "trade_proxy"
    gold_panel_dir = config.gold_data_dir / "event_panel"
    proxy_prices_path = silver_proxy_dir / "trade_proxy_option_prices.parquet"
    iv_estimates_path = silver_proxy_dir / "trade_proxy_contract_iv_estimates.parquet"
    ivar_inputs_path = silver_proxy_dir / "trade_proxy_ivar_inputs.parquet"
    panel_path = gold_panel_dir / "trade_proxy_event_panel.parquet"
    diagnostics_path = out_root / "trade_proxy_panel" / "trade_proxy_panel_report.json"
    if not force and panel_path.exists() and diagnostics_path.exists():
        return {
            "status": "skipped",
            "reason": "outputs_exist",
            "panel_grade": TRADE_PROXY_PANEL_GRADE,
            "outputs": {
                "trade_proxy_event_panel": str(panel_path),
                "trade_proxy_panel_report": str(diagnostics_path),
            },
        }

    windows = _read_parquet(windows_path)
    contracts = _read_parquet(contracts_path)
    if max_events is not None:
        keep_events = windows["event_id"].head(max_events).tolist()
        windows = windows.loc[windows["event_id"].isin(keep_events)].copy()
        contracts = contracts.loc[contracts["event_id"].isin(keep_events)].copy()
    contracts = contracts.loc[contracts["eligible_for_quote_pool"].astype(bool)].copy()
    contracts = contracts.merge(
        windows[["event_id", "event_entry_timestamp", "s_before", "s_after", "rvar_event"]],
        on="event_id",
        how="inner",
    )
    if max_contracts is not None:
        contracts = contracts.head(max_contracts).copy()

    bar_frames, fetch_report = _fetch_second_aggregate_bars(
        config,
        contracts,
        jobs=jobs,
        limit=rest_limit,
    )
    proxy_prices = build_trade_proxy_price_frame(
        contracts,
        bar_frames,
        lookback_seconds=lookback_seconds,
        price_field=price_field,
    )
    iv_estimates = attach_trade_proxy_local_iv(proxy_prices, windows)
    ivar_inputs = build_trade_proxy_ivar_inputs(iv_estimates, windows)
    panel = extract_trade_proxy_event_panel(ivar_inputs, windows)
    straddle_diagnostics = build_proxy_straddle_diagnostics(
        iv_estimates,
        windows,
        haircut_fraction=haircut_fraction,
    )
    edge_deciles = edge_decile_diagnostics(panel)

    _write_parquet(proxy_prices_path, proxy_prices)
    _write_parquet(iv_estimates_path, iv_estimates)
    _write_parquet(ivar_inputs_path, ivar_inputs)
    _write_parquet(panel_path, panel)
    report_dir = out_root / "trade_proxy_panel"
    report_dir.mkdir(parents=True, exist_ok=True)
    fetch_report.to_csv(report_dir / "second_aggregate_fetch_report.csv", index=False)
    straddle_diagnostics.to_csv(report_dir / "trade_proxy_straddle_diagnostics.csv", index=False)
    edge_deciles.to_csv(report_dir / "trade_proxy_edge_deciles.csv", index=False)

    report = summarize_trade_proxy_panel(
        panel=panel,
        proxy_prices=iv_estimates,
        straddle_diagnostics=straddle_diagnostics,
        lookback_seconds=lookback_seconds,
        price_field=price_field,
    )
    report["outputs"] = {
        "trade_proxy_option_prices": str(proxy_prices_path),
        "trade_proxy_contract_iv_estimates": str(iv_estimates_path),
        "trade_proxy_ivar_inputs": str(ivar_inputs_path),
        "trade_proxy_event_panel": str(panel_path),
        "second_aggregate_fetch_report": str(report_dir / "second_aggregate_fetch_report.csv"),
        "trade_proxy_straddle_diagnostics": str(
            report_dir / "trade_proxy_straddle_diagnostics.csv"
        ),
        "trade_proxy_edge_deciles": str(report_dir / "trade_proxy_edge_deciles.csv"),
        "trade_proxy_panel_report": str(diagnostics_path),
    }
    _write_json(diagnostics_path, report)
    return report


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-root", type=Path, default=Path("artifacts/data_pipeline"))
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--max-events", type=int)
    parser.add_argument("--max-contracts", type=int)
    parser.add_argument("--lookback-seconds", type=int, default=900)
    parser.add_argument(
        "--price-field", choices=["option_vwap", "option_close"], default="option_vwap"
    )
    parser.add_argument("--jobs", type=int, default=1)
    parser.add_argument("--rest-limit", type=int, default=50_000)
    parser.add_argument("--haircut-fraction", type=float, default=0.10)
    args = parser.parse_args()

    if args.jobs <= 0:
        raise ValueError("--jobs must be positive.")
    if args.lookback_seconds <= 0:
        raise ValueError("--lookback-seconds must be positive.")
    if args.haircut_fraction < 0:
        raise ValueError("--haircut-fraction must be nonnegative.")
    config = load_project_config()
    report = build_trade_proxy_panel(
        config=config,
        out_root=args.out_root,
        force=args.force,
        max_events=args.max_events,
        max_contracts=args.max_contracts,
        lookback_seconds=args.lookback_seconds,
        price_field=args.price_field,
        jobs=args.jobs,
        rest_limit=args.rest_limit,
        haircut_fraction=args.haircut_fraction,
    )
    print(json.dumps(report, indent=2, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

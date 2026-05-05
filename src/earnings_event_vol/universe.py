from __future__ import annotations

from collections.abc import Mapping, Sequence
from datetime import date

import pandas as pd

from earnings_event_vol.massive import parse_massive_option_ticker

ELIGIBLE_EQUITY_RULE_VERSION = "v1.0"
ELIGIBLE_EQUITY_SOURCE_DATASET = "sec_company_tickers"
TICKER_MAPPING_OK = "ok"
TICKER_NOT_FOUND = "ticker_not_found"
TICKER_MAPPING_AMBIGUOUS = "ticker_mapping_ambiguous"
PHASE1_COVID_SHOCK_BUCKET = "covid_shock"
PHASE1_STEADY_PROXY_BUCKET = "steady_proxy"

_ALLOWED_EXCHANGES = {"nasdaq", "nyse", "nyse american", "nyse arca"}
_FUND_MARKERS = (" etf", " fund", " trust", " etn", " index")


def _month_start(value: object) -> date:
    timestamp = pd.Timestamp(value)
    return date(timestamp.year, timestamp.month, 1)


def _underlying_ticker(value: object) -> str:
    raw = str(value).upper().strip()
    if raw.startswith("O:"):
        try:
            return str(parse_massive_option_ticker(raw)["ticker"]).upper()
        except ValueError:
            return raw
    return raw


def phase1_telemetry_bucket(value: object) -> str:
    day = pd.Timestamp(value).date()
    if date(2020, 1, 1) <= day <= date(2020, 9, 30):
        return PHASE1_COVID_SHOCK_BUCKET
    return PHASE1_STEADY_PROXY_BUCKET


def build_eligible_equity_tickers(
    rows: Sequence[Mapping[str, object]],
    *,
    source_snapshot_date: date,
    rule_version: str = ELIGIBLE_EQUITY_RULE_VERSION,
) -> pd.DataFrame:
    out_rows: list[dict[str, object]] = []
    for row in rows:
        ticker = str(row.get("ticker") or "").upper().strip()
        name = str(row.get("title") or row.get("name") or "").lower()
        exchange = str(row.get("exchange") or "").lower().strip()
        if not ticker:
            continue
        if exchange and exchange not in _ALLOWED_EXCHANGES:
            eligible = False
            reason = "unsupported_exchange"
        elif any(marker in f" {name}" for marker in _FUND_MARKERS):
            eligible = False
            reason = "fund_or_index_like_name"
        else:
            eligible = True
            reason = "eligible_common_equity"
        out_rows.append(
            {
                "ticker": ticker,
                "eligible": eligible,
                "filter_reason": reason,
                "exchange": exchange.upper() if exchange else None,
                "source_snapshot_date": source_snapshot_date,
                "rule_version": rule_version,
                "source_dataset": ELIGIBLE_EQUITY_SOURCE_DATASET,
            }
        )
    columns = [
        "ticker",
        "eligible",
        "filter_reason",
        "exchange",
        "source_snapshot_date",
        "rule_version",
        "source_dataset",
    ]
    return pd.DataFrame(out_rows, columns=columns)


def eligible_equity_cache_matches_rule(
    frame: pd.DataFrame,
    *,
    expected_rule_version: str = ELIGIBLE_EQUITY_RULE_VERSION,
) -> bool:
    return (
        not frame.empty
        and "rule_version" in frame.columns
        and frame["rule_version"].astype(str).eq(expected_rule_version).all()
    )


def build_ticker_month_liquidity(
    option_day_aggs: pd.DataFrame,
    *,
    source_snapshot_date: date,
    source_dataset: str = "massive_options_day_aggs",
    rule_version: str = ELIGIBLE_EQUITY_RULE_VERSION,
) -> pd.DataFrame:
    date_column = next(
        (
            column
            for column in ("quote_date", "source_date", "date")
            if column in option_day_aggs.columns
        ),
        None,
    )
    required = {"ticker", "volume"}
    missing = sorted(required - set(option_day_aggs.columns))
    if missing:
        raise ValueError(f"option day-agg frame missing required columns: {missing}")
    if date_column is None:
        raise ValueError("option day-agg frame requires quote_date, source_date, or date")
    if (
        "option_vwap" not in option_day_aggs.columns
        and "vwap" not in option_day_aggs.columns
        and "option_close" not in option_day_aggs.columns
        and "close" not in option_day_aggs.columns
    ):
        raise ValueError("option day-agg frame requires vwap/option_vwap or close/option_close")
    frame = option_day_aggs.copy()
    frame["ticker"] = frame["ticker"].map(_underlying_ticker)
    frame["quote_date"] = pd.to_datetime(frame[date_column]).dt.date
    frame["volume"] = pd.to_numeric(frame["volume"], errors="coerce")
    close_column = "option_close" if "option_close" in frame.columns else "close"
    close_source = (
        frame[close_column]
        if close_column in frame.columns
        else pd.Series(pd.NA, index=frame.index)
    )
    close = pd.to_numeric(close_source, errors="coerce")
    vwap_column = "option_vwap" if "option_vwap" in frame.columns else "vwap"
    if vwap_column in frame.columns:
        vwap = pd.to_numeric(frame[vwap_column], errors="coerce")
        frame["premium_price"] = vwap.where(vwap.gt(0), close)
    else:
        frame["premium_price"] = close
    frame["option_premium_dollar_volume"] = frame["premium_price"] * frame["volume"] * 100.0
    frame = frame.loc[
        frame["ticker"].ne("")
        & frame["premium_price"].gt(0)
        & frame["volume"].gt(0)
        & frame["option_premium_dollar_volume"].gt(0)
    ].copy()
    frame["month"] = frame["quote_date"].map(_month_start)
    grouped = (
        frame.groupby(["month", "ticker"], as_index=False)
        .agg(
            option_premium_dollar_volume=("option_premium_dollar_volume", "sum"),
            option_contract_volume=("volume", "sum"),
            option_day_rows=("ticker", "size"),
        )
        .sort_values(["month", "ticker"])
        .reset_index(drop=True)
    )
    grouped["source_snapshot_date"] = source_snapshot_date
    grouped["rule_version"] = rule_version
    grouped["source_dataset"] = source_dataset
    return grouped[
        [
            "month",
            "ticker",
            "option_premium_dollar_volume",
            "option_contract_volume",
            "option_day_rows",
            "source_snapshot_date",
            "rule_version",
            "source_dataset",
        ]
    ]


def build_monthly_liquid_universe(
    ticker_month_liquidity: pd.DataFrame,
    *,
    start_month: date,
    end_month: date,
    top_n: int = 50,
    trailing_months: int = 6,
    eligible_tickers: Sequence[str] | None = None,
) -> pd.DataFrame:
    if top_n <= 0:
        raise ValueError("top_n must be positive")
    if trailing_months <= 0:
        raise ValueError("trailing_months must be positive")
    required = {"month", "ticker", "option_premium_dollar_volume"}
    missing = sorted(required - set(ticker_month_liquidity.columns))
    if missing:
        raise ValueError(f"ticker month liquidity frame missing required columns: {missing}")
    frame = ticker_month_liquidity.copy()
    frame["month"] = pd.to_datetime(frame["month"]).map(_month_start)
    frame["month_ts"] = pd.to_datetime(frame["month"])
    frame["ticker"] = frame["ticker"].astype(str).str.upper()
    frame["option_premium_dollar_volume"] = pd.to_numeric(
        frame["option_premium_dollar_volume"], errors="coerce"
    ).fillna(0.0)
    allowed = {ticker.upper() for ticker in eligible_tickers or []}
    if allowed:
        frame = frame.loc[frame["ticker"].isin(allowed)].copy()

    current = pd.Timestamp(_month_start(start_month))
    final = pd.Timestamp(_month_start(end_month))
    rows: list[dict[str, object]] = []
    while current <= final:
        trailing_start = current - pd.DateOffset(months=trailing_months)
        window = frame.loc[(frame["month_ts"] >= trailing_start) & (frame["month_ts"] < current)]
        ranked = (
            window.groupby("ticker", as_index=False)["option_premium_dollar_volume"]
            .sum()
            .sort_values(["option_premium_dollar_volume", "ticker"], ascending=[False, True])
            .head(top_n)
            .reset_index(drop=True)
        )
        for index, row in ranked.iterrows():
            rows.append(
                {
                    "universe_month": current.date(),
                    "ticker": row["ticker"],
                    "rank": int(index) + 1,
                    "trailing_months": trailing_months,
                    "top_n": top_n,
                    "trailing_option_premium_dollar_volume": float(
                        row["option_premium_dollar_volume"]
                    ),
                    "telemetry_bucket": phase1_telemetry_bucket(current.date()),
                }
            )
        current += pd.DateOffset(months=1)
    return pd.DataFrame(
        rows,
        columns=[
            "universe_month",
            "ticker",
            "rank",
            "trailing_months",
            "top_n",
            "trailing_option_premium_dollar_volume",
            "telemetry_bucket",
        ],
    )


def ticker_mapping_diagnostics(
    requested_tickers: Sequence[str],
    option_chain_tickers: Sequence[str],
    *,
    aliases: Mapping[str, Sequence[str]] | None = None,
) -> pd.DataFrame:
    available = {ticker.upper() for ticker in option_chain_tickers}
    alias_map = {
        key.upper(): [value.upper() for value in values] for key, values in (aliases or {}).items()
    }
    rows: list[dict[str, object]] = []
    for raw in requested_tickers:
        ticker = raw.upper()
        candidates = [ticker, *alias_map.get(ticker, [])]
        matches = sorted({candidate for candidate in candidates if candidate in available})
        if len(matches) == 1:
            status = TICKER_MAPPING_OK
            mapped_ticker = matches[0]
        elif len(matches) > 1:
            status = TICKER_MAPPING_AMBIGUOUS
            mapped_ticker = None
        else:
            status = TICKER_NOT_FOUND
            mapped_ticker = None
        rows.append(
            {
                "ticker": ticker,
                "mapped_ticker": mapped_ticker,
                "mapping_status": status,
                "candidate_tickers": ",".join(candidates),
            }
        )
    return pd.DataFrame(
        rows,
        columns=["ticker", "mapped_ticker", "mapping_status", "candidate_tickers"],
    )

from __future__ import annotations

import hashlib
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pandas as pd

REQUIRED_OPTION_FIELDS = [
    "ticker",
    "quote_date",
    "expiration",
    "strike",
    "right",
    "bid",
    "ask",
    "volume",
    "open_interest",
]
REQUIRED_UNDERLYING_FIELDS = ["ticker", "date", "open", "high", "low", "close", "volume"]
REQUIRED_EARNINGS_FIELDS = ["ticker", "announcement_date", "announcement_timing", "source"]


@dataclass(frozen=True)
class FieldAuditResult:
    required_fields_report: dict[str, object]
    field_coverage: pd.DataFrame
    vendor_local_iv_diff: pd.DataFrame
    quote_source_report: pd.DataFrame


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _coverage(frame: pd.DataFrame, *, dataset: str, required_fields: Iterable[str]) -> pd.DataFrame:
    rows = []
    for field in required_fields:
        available = field in frame.columns
        non_null_share = float(frame[field].notna().mean()) if available and len(frame) else 0.0
        rows.append(
            {
                "dataset": dataset,
                "field": field,
                "available": available,
                "non_null_share": non_null_share,
                "row_count": int(len(frame)),
            }
        )
    return pd.DataFrame(rows)


def _bucket_dte(values: pd.Series) -> pd.Series:
    return pd.cut(
        values,
        bins=[-np.inf, 7, 14, 30, 60, np.inf],
        labels=["0_7", "8_14", "15_30", "31_60", "61_plus"],
    )


def _bucket_moneyness(values: pd.Series) -> pd.Series:
    return pd.cut(
        values,
        bins=[-np.inf, 0.9, 0.97, 1.03, 1.1, np.inf],
        labels=["deep_put_wing", "put_wing", "atm", "call_wing", "deep_call_wing"],
    )


def _empty_vendor_local_iv_diff(
    *, reason: str, missing_columns: Iterable[str] = ()
) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "moneyness_bucket": None,
                "dte_bucket": None,
                "n": 0,
                "mean_abs_vendor_local_iv_diff": None,
                "status": "skipped",
                "reason": reason,
                "missing_columns": ",".join(sorted(missing_columns)),
            }
        ]
    )


def _options_with_audit_buckets(options: pd.DataFrame) -> pd.DataFrame:
    frame = options.copy()
    if "dte" not in frame.columns and {"expiration", "quote_date"}.issubset(frame.columns):
        expiration = pd.to_datetime(frame["expiration"], errors="coerce")
        quote_date = pd.to_datetime(frame["quote_date"], errors="coerce")
        frame["dte"] = (expiration - quote_date).dt.days
    if "moneyness" not in frame.columns:
        if {"strike", "underlying_close"}.issubset(frame.columns):
            frame["moneyness"] = pd.to_numeric(frame["strike"]) / pd.to_numeric(
                frame["underlying_close"]
            )
        elif {"strike", "spot"}.issubset(frame.columns):
            frame["moneyness"] = pd.to_numeric(frame["strike"]) / pd.to_numeric(frame["spot"])
    return frame


def vendor_local_iv_comparison(options: pd.DataFrame) -> pd.DataFrame:
    options = _options_with_audit_buckets(options)
    required = {"vendor_iv", "local_iv", "moneyness", "dte"}
    if not required.issubset(options.columns):
        return _empty_vendor_local_iv_diff(
            reason="missing_required_columns",
            missing_columns=required - set(options.columns),
        )
    frame = options.dropna(subset=["vendor_iv", "local_iv", "moneyness", "dte"]).copy()
    if frame.empty:
        return _empty_vendor_local_iv_diff(reason="no_complete_vendor_local_iv_rows")
    frame["moneyness_bucket"] = _bucket_moneyness(pd.to_numeric(frame["moneyness"]))
    frame["dte_bucket"] = _bucket_dte(pd.to_numeric(frame["dte"]))
    frame["abs_diff"] = (pd.to_numeric(frame["vendor_iv"]) - pd.to_numeric(frame["local_iv"])).abs()
    out = (
        frame.groupby(["moneyness_bucket", "dte_bucket"], observed=True)
        .agg(n=("abs_diff", "size"), mean_abs_vendor_local_iv_diff=("abs_diff", "mean"))
        .reset_index()
    )
    out["status"] = "ok"
    out["reason"] = None
    out["missing_columns"] = None
    return out


def quote_source_report(options: pd.DataFrame) -> pd.DataFrame:
    if "quote_source" not in options.columns:
        return pd.DataFrame(
            [
                {
                    "quote_source": "missing",
                    "rows": int(len(options)),
                    "share": 1.0 if len(options) else 0.0,
                }
            ]
        )
    counts = options["quote_source"].fillna("missing").astype(str).value_counts(dropna=False)
    total = int(counts.sum())
    return pd.DataFrame(
        [
            {
                "quote_source": source,
                "rows": int(rows),
                "share": float(rows / total) if total else 0.0,
            }
            for source, rows in counts.items()
        ]
    )


def audit_data_fields(
    *,
    options: pd.DataFrame,
    underlying: pd.DataFrame,
    earnings: pd.DataFrame,
    source_paths: Sequence[Path] = (),
) -> FieldAuditResult:
    coverage = pd.concat(
        [
            _coverage(options, dataset="option_quotes", required_fields=REQUIRED_OPTION_FIELDS),
            _coverage(
                underlying, dataset="underlying_bars", required_fields=REQUIRED_UNDERLYING_FIELDS
            ),
            _coverage(
                earnings, dataset="earnings_calendar", required_fields=REQUIRED_EARNINGS_FIELDS
            ),
        ],
        ignore_index=True,
    )
    missing = coverage.loc[~coverage["available"], ["dataset", "field"]].to_dict("records")
    report: dict[str, object] = {
        "ok": not missing,
        "missing_required_fields": missing,
        "created_at_utc": datetime.now(UTC).isoformat(timespec="seconds"),
        "source_files": [
            {"path": str(path), "sha256": file_sha256(path)}
            for path in source_paths
            if Path(path).exists()
        ],
    }
    return FieldAuditResult(
        required_fields_report=report,
        field_coverage=coverage,
        vendor_local_iv_diff=vendor_local_iv_comparison(options),
        quote_source_report=quote_source_report(options),
    )

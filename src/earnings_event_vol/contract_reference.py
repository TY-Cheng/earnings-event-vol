from __future__ import annotations

import json
import re
import time
from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import Any
from urllib.parse import quote

import httpx
import numpy as np
import pandas as pd

from earnings_event_vol.config import ProjectConfig
from earnings_event_vol.event_panel import CONTRACT_STATUS_NON_STANDARD_EXCLUDED
from earnings_event_vol.massive import parse_massive_option_ticker, read_secret_file

CONTRACT_REFERENCE_SCHEMA_VERSION = "v1"
CONTRACT_REFERENCE_SOURCE_DATASET = "massive_reference_options_contracts"
CONTRACT_STATUS_REFERENCE_UNVALIDATED_EXCLUDED = "contract_reference_unvalidated_excluded"
QueryParamValue = str | int | float | bool | None
_SECRET_QUERY_PATTERN = re.compile(r"(?i)((?:apiKey|api_key)=)[^&\s)]+")

REFERENCE_STATUS_VALIDATED = "validated"
REFERENCE_STATUS_FETCH_FAILED = "fetch_failed"
REFERENCE_STATUS_MISSING_REFERENCE = "missing_reference"
REFERENCE_STATUS_PARSE_FAILED = "parse_failed"
REFERENCE_STATUS_NOT_REQUESTED = "not_requested"
REFERENCE_PROXY_SOURCE_EXCLUDED = "excluded"
REFERENCE_PROXY_SOURCE_VALIDATED = "validated_reference"
REFERENCE_PROXY_SOURCE_MISSING_STANDARD_FALLBACK = "missing_reference_standard_contract_fallback"


@dataclass(frozen=True)
class ContractReferenceFetchResult:
    options_ticker: str
    fetch_status: str
    contract_reference_status: str
    payload: dict[str, Any] | None = None
    error: str | None = None
    cache_path: str | None = None

    def report_row(self) -> dict[str, object]:
        extracted = extract_contract_reference_fields(self.options_ticker, self.payload)
        return {
            "options_ticker": self.options_ticker,
            "fetch_status": self.fetch_status,
            "contract_reference_status": self.contract_reference_status,
            "contract_reference_error": self.error,
            "cache_path": self.cache_path,
            **extracted,
        }


def contract_reference_cache_path(root: Path, options_ticker: str) -> Path:
    safe = "".join(ch if ch.isalnum() else "_" for ch in options_ticker)
    return root / f"options_ticker={safe}" / "reference.json"


def _api_key(config: ProjectConfig) -> str:
    secret = read_secret_file(config.massive_api_key_file)
    if not secret:
        raise ValueError("MASSIVE_API_KEY_FILE is not configured.")
    return secret


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")


def _load_cached_payload(path: Path) -> dict[str, Any] | None:
    if not path.exists() or path.stat().st_size <= 0:
        return None
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    return payload if isinstance(payload, dict) else None


def _redact_secret_query_params(text: str) -> str:
    return _SECRET_QUERY_PATTERN.sub(r"\1<redacted>", text)


def _safe_exception_text(exc: Exception, *, max_chars: int = 300) -> str:
    if isinstance(exc, httpx.HTTPStatusError):
        body = _redact_secret_query_params(" ".join(exc.response.text.strip().split()))
        suffix = f": {body[:200]}" if body else ""
        return _redact_secret_query_params(f"HTTP {exc.response.status_code}{suffix}")[:max_chars]
    return _redact_secret_query_params(str(exc))[:max_chars]


def _extract_result(payload: dict[str, Any] | None, options_ticker: str) -> dict[str, Any] | None:
    if not isinstance(payload, dict):
        return None
    result = payload.get("results")
    if isinstance(result, dict):
        return result
    if isinstance(result, list):
        if not result:
            return None
        upper_ticker = options_ticker.upper()
        for item in result:
            if isinstance(item, dict) and str(item.get("ticker", "")).upper() == upper_ticker:
                return item
        return None
    if any(key in payload for key in ("shares_per_contract", "additional_underlyings", "ticker")):
        return payload
    return None


def _expiration_from_options_ticker(options_ticker: str) -> date | None:
    try:
        expiration = parse_massive_option_ticker(options_ticker)["expiration"]
    except ValueError:
        return None
    return expiration if isinstance(expiration, date) else None


def _additional_underlyings_count(value: Any) -> int:
    if value is None:
        return 0
    if isinstance(value, list | tuple | set):
        return len(value)
    if isinstance(value, dict):
        return len(value)
    if isinstance(value, str):
        return 0 if value.strip() in {"", "[]", "{}", "null", "None"} else 1
    return 1


def _reference_status(
    payload: dict[str, Any] | None,
    options_ticker: str,
) -> tuple[str, str | None]:
    result = _extract_result(payload, options_ticker)
    if result is None:
        return REFERENCE_STATUS_MISSING_REFERENCE, "missing reference result"
    shares = pd.to_numeric(result.get("shares_per_contract"), errors="coerce")
    if pd.isna(shares):
        return REFERENCE_STATUS_PARSE_FAILED, "missing shares_per_contract"
    return REFERENCE_STATUS_VALIDATED, None


def _get_json_with_retries(
    client: httpx.Client,
    url: str,
    *,
    params: dict[str, QueryParamValue],
    config: ProjectConfig,
) -> dict[str, Any]:
    attempts = max(1, int(config.massive_max_retries) + 1)
    retry_statuses = {429, 500, 502, 503, 504}
    last_error: Exception | None = None
    for attempt in range(attempts):
        try:
            response = client.get(url, params=params)
            if response.status_code in retry_statuses and attempt < attempts - 1:
                time.sleep(float(config.massive_retry_backoff_seconds) * (attempt + 1))
                continue
            response.raise_for_status()
            payload = response.json()
            if not isinstance(payload, dict):
                raise ValueError("Massive contract reference response is not a JSON object.")
            return payload
        except (httpx.HTTPError, ValueError) as exc:
            last_error = exc
            if isinstance(exc, httpx.HTTPStatusError):
                status = exc.response.status_code
                if status not in retry_statuses:
                    raise
            if attempt < attempts - 1:
                time.sleep(float(config.massive_retry_backoff_seconds) * (attempt + 1))
    assert last_error is not None
    raise last_error


def fetch_massive_option_contract_reference(
    client: httpx.Client,
    config: ProjectConfig,
    *,
    options_ticker: str,
    cache_root: Path,
    refresh_bronze: bool = False,
) -> ContractReferenceFetchResult:
    ticker = str(options_ticker).strip().upper()
    cache_path = contract_reference_cache_path(cache_root, ticker)
    if not refresh_bronze:
        cached = _load_cached_payload(cache_path)
        if cached is not None:
            status, error = _reference_status(cached, ticker)
            return ContractReferenceFetchResult(
                options_ticker=ticker,
                fetch_status="hit",
                contract_reference_status=status,
                payload=cached,
                error=error,
                cache_path=str(cache_path),
            )

    api_key = _api_key(config)
    base = config.massive_base_url.rstrip("/")
    encoded = quote(ticker, safe="")
    params: dict[str, QueryParamValue] = {"apiKey": api_key}
    urls = [
        f"{base}/v3/reference/options/contracts/{encoded}",
        f"{base}/v3/reference/options/contracts",
    ]
    fallback_params: dict[str, QueryParamValue] = {
        "apiKey": api_key,
        "ticker": ticker,
        "expired": "true",
        "limit": 1,
    }
    expiration = _expiration_from_options_ticker(ticker)
    if expiration is not None:
        fallback_params["as_of"] = expiration.isoformat()
    last_error: str | None = None
    for index, url in enumerate(urls):
        try:
            payload = _get_json_with_retries(
                client,
                url,
                params=params if index == 0 else fallback_params,
                config=config,
            )
            status, error = _reference_status(payload, ticker)
            _write_json(cache_path, payload)
            fetch_status = (
                "missing_reference"
                if status == REFERENCE_STATUS_MISSING_REFERENCE
                else "downloaded"
            )
            return ContractReferenceFetchResult(
                options_ticker=ticker,
                fetch_status=fetch_status,
                contract_reference_status=status,
                payload=payload,
                error=error,
                cache_path=str(cache_path),
            )
        except httpx.HTTPStatusError as exc:
            status_code = exc.response.status_code
            last_error = _safe_exception_text(exc)
            if status_code in {400, 404} and index == 0:
                continue
            break
        except Exception as exc:  # pragma: no cover - network defensive path
            last_error = _safe_exception_text(exc)
            break

    return ContractReferenceFetchResult(
        options_ticker=ticker,
        fetch_status="failed",
        contract_reference_status=REFERENCE_STATUS_FETCH_FAILED,
        payload=None,
        error=last_error or "contract reference fetch failed",
        cache_path=str(cache_path),
    )


def extract_contract_reference_fields(
    options_ticker: str,
    payload: dict[str, Any] | None,
) -> dict[str, object]:
    result = _extract_result(payload, options_ticker)
    if result is None:
        return {
            "contract_reference_shares_per_contract": np.nan,
            "contract_reference_additional_underlyings_count": 0,
            "contract_reference_has_adjusted_deliverable": False,
            "contract_reference_exercise_style": pd.NA,
            "contract_reference_correction": pd.NA,
        }
    shares = pd.to_numeric(result.get("shares_per_contract"), errors="coerce")
    additional_count = _additional_underlyings_count(result.get("additional_underlyings"))
    return {
        "contract_reference_shares_per_contract": float(shares) if not pd.isna(shares) else np.nan,
        "contract_reference_additional_underlyings_count": int(additional_count),
        "contract_reference_has_adjusted_deliverable": bool(additional_count > 0),
        "contract_reference_exercise_style": result.get("exercise_style", pd.NA),
        "contract_reference_correction": result.get("correction", pd.NA),
    }


def _bool_column(frame: pd.DataFrame, column: str, default: bool = False) -> pd.Series:
    if column not in frame.columns:
        return pd.Series(default, index=frame.index)
    return frame[column].fillna(default).astype(bool)


def _numeric_column(frame: pd.DataFrame, column: str) -> pd.Series:
    if column not in frame.columns:
        return pd.Series(pd.NA, index=frame.index, dtype="Float64")
    return pd.to_numeric(frame[column], errors="coerce")


def _contract_reference_proxy_mask(frame: pd.DataFrame) -> pd.Series:
    validated = _bool_column(frame, "contract_reference_validated")
    status = (
        frame["contract_reference_status"].astype(str)
        if "contract_reference_status" in frame.columns
        else pd.Series("", index=frame.index)
    )
    discovery_status = (
        frame["contract_discovery_status_pre_reference"].astype(str)
        if "contract_discovery_status_pre_reference" in frame.columns
        else frame.get("contract_discovery_status", pd.Series("", index=frame.index))
        .fillna("")
        .astype(str)
    )
    adjusted = _bool_column(frame, "contract_reference_has_adjusted_deliverable")
    standard_size = _numeric_column(frame, "option_multiplier").eq(100) | _numeric_column(
        frame, "contract_size"
    ).eq(100)
    reference_missing = status.isin(
        {REFERENCE_STATUS_MISSING_REFERENCE, REFERENCE_STATUS_NOT_REQUESTED}
    )
    fallback = reference_missing & discovery_status.eq("ok") & standard_size & ~adjusted
    return validated | fallback


def apply_contract_reference_validation(
    candidates: pd.DataFrame,
    reference_report: pd.DataFrame,
) -> pd.DataFrame:
    if "options_ticker" not in candidates.columns:
        raise ValueError("candidate contract frame missing options_ticker column.")
    out = candidates.copy()
    if "contract_discovery_status" in out.columns:
        out["contract_discovery_status_pre_reference"] = out["contract_discovery_status"]
    else:
        out["contract_discovery_status_pre_reference"] = pd.NA
        out["contract_discovery_status"] = pd.NA
    if reference_report.empty:
        out["contract_reference_status"] = REFERENCE_STATUS_NOT_REQUESTED
        out["contract_reference_validated"] = False
        out["contract_reference_proxy_usable"] = False
        out["contract_reference_proxy_source"] = REFERENCE_PROXY_SOURCE_EXCLUDED
        out["contract_reference_source_dataset"] = CONTRACT_REFERENCE_SOURCE_DATASET
        if "eligible_for_quote_pool" in out.columns:
            out.loc[:, "eligible_for_quote_pool"] = False
        if "is_main_dte_5_14" in out.columns:
            out.loc[:, "is_main_dte_5_14"] = False
        if "is_robustness_dte_3_21" in out.columns:
            out.loc[:, "is_robustness_dte_3_21"] = False
        out.loc[
            out["contract_discovery_status"].astype(str).eq("ok"),
            "contract_discovery_status",
        ] = CONTRACT_STATUS_REFERENCE_UNVALIDATED_EXCLUDED
        return out

    required = {"options_ticker", "contract_reference_status"}
    missing = required.difference(reference_report.columns)
    if missing:
        raise ValueError(f"contract reference report missing columns: {sorted(missing)}")

    report = reference_report.drop_duplicates("options_ticker", keep="last").copy()
    merge_columns = [
        column
        for column in [
            "options_ticker",
            "contract_reference_status",
            "contract_reference_error",
            "contract_reference_shares_per_contract",
            "contract_reference_additional_underlyings_count",
            "contract_reference_has_adjusted_deliverable",
            "contract_reference_exercise_style",
            "contract_reference_correction",
        ]
        if column in report.columns
    ]
    out = out.merge(report[merge_columns], on="options_ticker", how="left")
    out["contract_reference_status"] = out["contract_reference_status"].fillna(
        REFERENCE_STATUS_NOT_REQUESTED
    )
    out["contract_reference_validated"] = out["contract_reference_status"].eq(
        REFERENCE_STATUS_VALIDATED
    )
    proxy_usable = _contract_reference_proxy_mask(out)
    out["contract_reference_proxy_usable"] = proxy_usable
    out["contract_reference_proxy_source"] = REFERENCE_PROXY_SOURCE_EXCLUDED
    out.loc[
        out["contract_reference_validated"].astype(bool),
        "contract_reference_proxy_source",
    ] = REFERENCE_PROXY_SOURCE_VALIDATED
    out.loc[
        proxy_usable & ~out["contract_reference_validated"].astype(bool),
        "contract_reference_proxy_source",
    ] = REFERENCE_PROXY_SOURCE_MISSING_STANDARD_FALLBACK
    out["contract_reference_source_dataset"] = CONTRACT_REFERENCE_SOURCE_DATASET

    raw_shares = (
        out["contract_reference_shares_per_contract"]
        if "contract_reference_shares_per_contract" in out
        else pd.Series(pd.NA, index=out.index, dtype="Float64")
    )
    shares = pd.to_numeric(raw_shares, errors="coerce")
    validated = out["contract_reference_validated"].astype(bool)
    out["option_multiplier"] = pd.to_numeric(out.get("option_multiplier"), errors="coerce")
    out["contract_size"] = pd.to_numeric(out.get("contract_size"), errors="coerce")
    has_reference_shares = validated & shares.notna()
    if has_reference_shares.any():
        out.loc[has_reference_shares, "option_multiplier"] = shares.loc[has_reference_shares]
        out.loc[has_reference_shares, "contract_size"] = shares.loc[has_reference_shares]

    adjusted = out.get("contract_reference_has_adjusted_deliverable", False)
    adjusted = pd.Series(adjusted, index=out.index).fillna(False).astype(bool)
    non_standard = validated & (shares.ne(100) | adjusted)
    not_proxy_usable = ~proxy_usable
    if "deliverable_status" not in out.columns:
        out["deliverable_status"] = "standard"
    out.loc[validated & ~non_standard, "deliverable_status"] = "standard"
    out.loc[non_standard, "deliverable_status"] = "non_standard"
    if "corporate_action_flag" not in out.columns:
        out["corporate_action_flag"] = False
    out["corporate_action_flag"] = (
        out["corporate_action_flag"].fillna(False).astype(bool) | non_standard
    )

    out.loc[non_standard, "contract_discovery_status"] = CONTRACT_STATUS_NON_STANDARD_EXCLUDED
    if "eligible_for_quote_pool" in out.columns:
        out.loc[non_standard | not_proxy_usable, "eligible_for_quote_pool"] = False
    if "is_main_dte_5_14" in out.columns:
        out.loc[non_standard | not_proxy_usable, "is_main_dte_5_14"] = False
    if "is_robustness_dte_3_21" in out.columns:
        out.loc[non_standard | not_proxy_usable, "is_robustness_dte_3_21"] = False
    out.loc[
        not_proxy_usable & out["contract_discovery_status"].astype(str).eq("ok"),
        "contract_discovery_status",
    ] = CONTRACT_STATUS_REFERENCE_UNVALIDATED_EXCLUDED
    return out

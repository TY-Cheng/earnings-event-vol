from __future__ import annotations

import json
import re
import time as time_module
from collections.abc import Mapping, Sequence
from datetime import UTC, date, datetime, time
from pathlib import Path
from typing import Any

import httpx
import pandas as pd

from earnings_event_vol.config import ProjectConfig
from earnings_event_vol.events import NEW_YORK_TZ
from earnings_event_vol.massive import read_secret_file
from earnings_event_vol.schemas import AnnouncementTiming

CALENDAR_COLUMNS = [
    "ticker",
    "announcement_date",
    "announcement_timing",
    "source",
    "source_timestamp",
    "source_id",
    "filing_date",
    "acceptance_local_date",
    "acceptance_inferred_timing",
    "announcement_date_source",
    "filing_acceptance_date_mismatch",
    "cik",
    "form_type",
    "sec_items",
    "report_date",
    "primary_document",
    "primary_doc_description",
    "timing_source",
    "timing_confidence",
    "text_validation_status",
    "text_validation_reason",
    "text_validation_source",
    "text_validation_aux_status",
    "is_main_sample_timing",
    "is_validated_earnings_event",
    "is_main_sample_candidate",
]

_SEC_FORMS = {"8-K"}
_BMO_CUTOFF = time(9, 30)
_AMC_CUTOFF = time(16, 0)
_EARNINGS_MARKERS = (
    "financial results",
    "earnings results",
    "quarterly results",
    "results for the quarter",
    "results for its quarter",
    "results for its fiscal",
    "fiscal quarter",
    "quarter ended",
    "quarter and year ended",
)
_NON_EARNINGS_MARKERS = (
    "production and deliveries",
    "vehicle production",
    "deliveries and production",
    "annual meeting",
    "election of directors",
    "departure of directors",
    "compensatory arrangements",
    "submission of matters",
    "definitive merger agreement",
    "notes due",
)
_SEC_ARCHIVE_URL_TEMPLATE = "https://data.sec.gov/submissions/{name}"
_SEC_PRIMARY_DOCUMENT_URL_TEMPLATE = (
    "https://www.sec.gov/Archives/edgar/data/{cik}/{accession}/{primary_document}"
)
_SECRET_QUERY_PATTERN = re.compile(r"(?i)((?:apiKey|api_key)=)[^&\s)]+")


def _parse_date(value: object) -> date | None:
    if value is None:
        return None
    text = str(value).strip()
    if not text:
        return None
    try:
        return date.fromisoformat(text[:10])
    except ValueError:
        return None


def parse_aware_timestamp(value: object) -> datetime | None:
    if value is None:
        return None
    text = str(value).strip()
    if not text:
        return None
    try:
        timestamp = datetime.fromisoformat(text.replace("Z", "+00:00"))
    except ValueError:
        return None
    if timestamp.tzinfo is None:
        return None
    return timestamp


def infer_timing_from_acceptance_timestamp(value: object) -> AnnouncementTiming:
    timestamp = parse_aware_timestamp(value)
    if timestamp is None:
        return AnnouncementTiming.UNKNOWN
    local_time = timestamp.astimezone(NEW_YORK_TZ).time()
    if local_time < _BMO_CUTOFF:
        return AnnouncementTiming.BMO
    if local_time >= _AMC_CUTOFF:
        return AnnouncementTiming.AMC
    return AnnouncementTiming.DMH


def _acceptance_local_date(value: object) -> date | None:
    timestamp = parse_aware_timestamp(value)
    if timestamp is None:
        return None
    return timestamp.astimezone(NEW_YORK_TZ).date()


def classify_8k_text(items_text: str | None) -> tuple[str, str]:
    if not items_text:
        return "missing_text", "Filing text was not available for this accession."
    text = re.sub(r"\s+", " ", items_text).lower()
    if "item 2.02" not in text and "item 2.02." not in text:
        return "not_item_2_02_text", "The filing text does not contain Item 2.02."
    has_earnings_marker = any(marker in text for marker in _EARNINGS_MARKERS)
    has_non_earnings_marker = any(marker in text for marker in _NON_EARNINGS_MARKERS)
    if has_non_earnings_marker:
        return "non_earnings_item_2_02", "Item 2.02 text appears unrelated to quarterly earnings."
    if has_earnings_marker:
        return "validated_earnings_release", "Item 2.02 text describes quarterly results."
    return "ambiguous_item_2_02_text", "Item 2.02 is present but earnings-release wording is weak."


def _safe_exception_text(exc: Exception, *, max_chars: int = 300) -> str:
    text = str(exc)
    if isinstance(exc, httpx.HTTPStatusError):
        body = " ".join(exc.response.text.strip().split())
        text = (
            f"HTTP {exc.response.status_code}: {body[:200]}"
            if body
            else f"HTTP {exc.response.status_code}"
        )
    return _SECRET_QUERY_PATTERN.sub(r"\1<redacted>", text)[:max_chars]


def _recent_value(recent: Mapping[str, Any], key: str, index: int) -> object:
    values = recent.get(key)
    if not isinstance(values, list) or index >= len(values):
        return None
    return values[index]


def _normalize_sec_submission_block(
    *,
    ticker: str,
    block: Mapping[str, Any],
    cik: int | None,
    start_date: date,
    end_date: date,
    source: str,
) -> pd.DataFrame:
    forms = block.get("form")
    if not isinstance(forms, list):
        return pd.DataFrame(columns=CALENDAR_COLUMNS)

    rows: list[dict[str, object]] = []
    for index, form_value in enumerate(forms):
        form = str(form_value or "").strip()
        if form not in _SEC_FORMS:
            continue
        items = str(_recent_value(block, "items", index) or "")
        if "2.02" not in items:
            continue
        filing_date = _parse_date(_recent_value(block, "filingDate", index))
        acceptance = _recent_value(block, "acceptanceDateTime", index)
        acceptance_local_date = _acceptance_local_date(acceptance)
        announcement_date = acceptance_local_date or filing_date
        if (
            announcement_date is None
            or announcement_date < start_date
            or announcement_date > end_date
        ):
            continue
        inferred_timing = infer_timing_from_acceptance_timestamp(acceptance)
        timing = AnnouncementTiming.UNKNOWN
        rows.append(
            {
                "ticker": ticker.upper(),
                "announcement_date": announcement_date.isoformat(),
                "announcement_timing": timing.value,
                "source": source,
                "source_timestamp": acceptance,
                "source_id": _recent_value(block, "accessionNumber", index),
                "filing_date": filing_date.isoformat() if filing_date is not None else "",
                "acceptance_local_date": (
                    acceptance_local_date.isoformat() if acceptance_local_date is not None else ""
                ),
                "acceptance_inferred_timing": inferred_timing.value,
                "announcement_date_source": (
                    "sec_acceptance_local_date_proxy"
                    if acceptance_local_date is not None
                    else "sec_filing_date"
                ),
                "filing_acceptance_date_mismatch": bool(
                    filing_date is not None
                    and acceptance_local_date is not None
                    and filing_date != acceptance_local_date
                ),
                "cik": cik,
                "form_type": form,
                "sec_items": items,
                "report_date": _recent_value(block, "reportDate", index),
                "primary_document": _recent_value(block, "primaryDocument", index),
                "primary_doc_description": _recent_value(block, "primaryDocDescription", index),
                "timing_source": "sec_acceptance_timestamp_proxy_not_main_sample",
                "timing_confidence": "proxy_inconclusive",
                "text_validation_status": "pending_text_validation",
                "text_validation_reason": "",
                "text_validation_source": "",
                "text_validation_aux_status": "",
                "is_main_sample_timing": False,
                "is_validated_earnings_event": False,
                "is_main_sample_candidate": False,
            }
        )
    return pd.DataFrame(rows, columns=CALENDAR_COLUMNS)


def normalize_sec_submission_candidates(
    *,
    ticker: str,
    payload: Mapping[str, Any],
    start_date: date,
    end_date: date,
) -> pd.DataFrame:
    filings = payload.get("filings") if isinstance(payload.get("filings"), dict) else {}
    recent = filings.get("recent") if isinstance(filings, dict) else {}
    cik_value = payload.get("cik")
    cik = int(cik_value) if isinstance(cik_value, int) else None
    frames: list[pd.DataFrame] = []
    if isinstance(recent, dict):
        frames.append(
            _normalize_sec_submission_block(
                ticker=ticker,
                block=recent,
                cik=cik,
                start_date=start_date,
                end_date=end_date,
                source="sec_edgar_submissions_recent",
            )
        )
    archive_payloads = payload.get("archive_payloads")
    if isinstance(archive_payloads, list):
        for archive in archive_payloads:
            if isinstance(archive, dict):
                frames.append(
                    _normalize_sec_submission_block(
                        ticker=ticker,
                        block=archive,
                        cik=cik,
                        start_date=start_date,
                        end_date=end_date,
                        source="sec_edgar_submissions_archive",
                    )
                )
    if not frames:
        return pd.DataFrame(columns=CALENDAR_COLUMNS)
    out = pd.concat(frames, ignore_index=True)
    if out.empty:
        return pd.DataFrame(columns=CALENDAR_COLUMNS)
    return (
        out.drop_duplicates(subset=["ticker", "source_id"], keep="first")
        .sort_values(["announcement_date", "ticker", "source_id"], kind="stable")
        .reset_index(drop=True)
    )


def _get_json(
    client: httpx.Client,
    url: str,
    *,
    headers: Mapping[str, str] | None = None,
    params: Mapping[str, str] | None = None,
) -> dict[str, Any]:
    response = client.get(url, headers=dict(headers or {}), params=dict(params or {}))
    response.raise_for_status()
    payload = response.json()
    if not isinstance(payload, dict):
        raise ValueError(f"expected JSON object from {url}")
    return payload


def _get_text(
    client: httpx.Client,
    url: str,
    *,
    headers: Mapping[str, str] | None = None,
) -> str:
    response = client.get(url, headers=dict(headers or {}))
    response.raise_for_status()
    return response.text


def _retryable_http_status(exc: httpx.HTTPStatusError) -> bool:
    return exc.response.status_code in {429, 500, 502, 503, 504}


def _get_json_with_retries(
    client: httpx.Client,
    url: str,
    *,
    headers: Mapping[str, str] | None = None,
    params: Mapping[str, str] | None = None,
    max_retries: int,
    backoff_seconds: float,
) -> dict[str, Any]:
    attempts = max(1, max_retries + 1)
    last_exc: Exception | None = None
    for attempt in range(attempts):
        try:
            return _get_json(client, url, headers=headers, params=params)
        except httpx.HTTPStatusError as exc:
            if not _retryable_http_status(exc):
                raise
            last_exc = exc
        except (httpx.TransportError, httpx.TimeoutException) as exc:
            last_exc = exc
        if attempt < attempts - 1 and backoff_seconds > 0:
            time_module.sleep(backoff_seconds * (2**attempt))
    assert last_exc is not None
    raise last_exc


def _get_text_with_retries(
    client: httpx.Client,
    url: str,
    *,
    headers: Mapping[str, str] | None = None,
    max_retries: int,
    backoff_seconds: float,
) -> str:
    attempts = max(1, max_retries + 1)
    last_exc: Exception | None = None
    for attempt in range(attempts):
        try:
            return _get_text(client, url, headers=headers)
        except httpx.HTTPStatusError as exc:
            if not _retryable_http_status(exc):
                raise
            last_exc = exc
        except (httpx.TransportError, httpx.TimeoutException) as exc:
            last_exc = exc
        if attempt < attempts - 1 and backoff_seconds > 0:
            time_module.sleep(backoff_seconds * (2**attempt))
    assert last_exc is not None
    raise last_exc


def fetch_sec_ticker_map(client: httpx.Client, config: ProjectConfig) -> dict[str, int]:
    payload = _get_json(
        client,
        config.sec_company_tickers_url,
        headers={"User-Agent": config.sec_user_agent},
    )
    out: dict[str, int] = {}
    for value in payload.values():
        if not isinstance(value, dict):
            continue
        ticker = str(value.get("ticker") or "").upper()
        cik = value.get("cik_str")
        if ticker and isinstance(cik, int):
            out[ticker] = cik
    return out


def fetch_sec_submission_payloads(
    *,
    tickers: Sequence[str],
    config: ProjectConfig,
    client: httpx.Client,
    archive_cache_dir: Path | None = None,
    include_archives: bool = True,
    fail_on_missing_tickers: bool = True,
    request_interval_seconds: float = 0.11,
) -> dict[str, dict[str, Any]]:
    ticker_map = fetch_sec_ticker_map(client, config)
    payloads: dict[str, dict[str, Any]] = {}
    missing: list[str] = []
    last_request_at = 0.0

    def throttled_get_json(url: str) -> dict[str, Any]:
        nonlocal last_request_at
        elapsed = time_module.monotonic() - last_request_at
        if elapsed < request_interval_seconds:
            time_module.sleep(request_interval_seconds - elapsed)
        payload = _get_json(client, url, headers={"User-Agent": config.sec_user_agent})
        last_request_at = time_module.monotonic()
        return payload

    for ticker in tickers:
        normalized = ticker.upper()
        cik = ticker_map.get(normalized)
        if cik is None:
            missing.append(normalized)
            if not fail_on_missing_tickers:
                payloads[normalized] = {
                    "filings": {"recent": {}},
                    "sec_fetch_status": "ticker_not_found",
                }
            continue
        payload = throttled_get_json(config.sec_submissions_url_template.format(cik=cik))
        payload["cik"] = int(cik)
        payload["sec_fetch_status"] = "ok"
        archive_payloads: list[dict[str, Any]] = []
        archive_failures: list[dict[str, str]] = []
        files = (
            (payload.get("filings") or {}).get("files")
            if isinstance(payload.get("filings"), dict)
            else []
        )
        if include_archives and isinstance(files, list):
            for item in files:
                if not isinstance(item, dict):
                    continue
                name = str(item.get("name") or "").strip()
                if not name:
                    continue
                archive_payload: dict[str, Any] | None = None
                archive_path = (
                    archive_cache_dir / f"ticker={normalized}" / name
                    if archive_cache_dir is not None
                    else None
                )
                if archive_path is not None and archive_path.exists():
                    try:
                        parsed = json.loads(archive_path.read_text(encoding="utf-8"))
                        archive_payload = parsed if isinstance(parsed, dict) else None
                    except (OSError, json.JSONDecodeError) as exc:
                        archive_failures.append({"name": name, "error": str(exc)})
                if archive_payload is None:
                    try:
                        archive_payload = throttled_get_json(
                            _SEC_ARCHIVE_URL_TEMPLATE.format(name=name)
                        )
                        if archive_path is not None:
                            archive_path.parent.mkdir(parents=True, exist_ok=True)
                            archive_path.write_text(
                                json.dumps(archive_payload, indent=2),
                                encoding="utf-8",
                            )
                    except Exception as exc:
                        archive_failures.append({"name": name, "error": str(exc)})
                        continue
                archive_payloads.append(archive_payload)
        payload["archive_payloads"] = archive_payloads
        payload["sec_archive_fetch_failed"] = archive_failures
        payloads[normalized] = payload
    if missing and fail_on_missing_tickers:
        raise ValueError(f"SEC ticker map missing tickers: {', '.join(sorted(missing))}")
    return payloads


def fetch_massive_8k_text_payloads(
    *,
    tickers: Sequence[str],
    config: ProjectConfig,
    client: httpx.Client,
    start_date: date,
    end_date: date,
    max_pages_per_form: int = 25,
) -> dict[str, dict[str, Any]]:
    api_key = read_secret_file(config.massive_api_key_file)
    if not api_key:
        raise ValueError("MASSIVE_API_KEY_FILE is required for Massive 8-K text fallback.")

    payloads: dict[str, dict[str, Any]] = {}
    endpoint = config.massive_base_url.rstrip("/") + config.massive_8k_text_path
    for ticker in tickers:
        results: list[dict[str, Any]] = []
        for form_type in _SEC_FORMS:
            url: str | None = endpoint
            params: dict[str, str] | None = {
                "ticker": ticker.upper(),
                "form_type": form_type,
                "filing_date.gte": start_date.isoformat(),
                "filing_date.lte": end_date.isoformat(),
                "sort": "filing_date.desc",
                "limit": "100",
                "apiKey": api_key,
            }
            pages = 0
            while url and pages < max_pages_per_form:
                try:
                    payload = _get_json_with_retries(
                        client,
                        url,
                        params=params,
                        max_retries=config.massive_max_retries,
                        backoff_seconds=config.massive_retry_backoff_seconds,
                    )
                except httpx.HTTPStatusError as exc:
                    if not _retryable_http_status(exc):
                        raise
                    results.append(
                        {
                            "fetch_failure": "massive_8k_text_http_retry_exhausted",
                            "ticker": ticker.upper(),
                            "form_type": form_type,
                            "error": _safe_exception_text(exc),
                        }
                    )
                    break
                except (httpx.TransportError, httpx.TimeoutException) as exc:
                    results.append(
                        {
                            "fetch_failure": "massive_8k_text_transport_retry_exhausted",
                            "ticker": ticker.upper(),
                            "form_type": form_type,
                            "error": _safe_exception_text(exc),
                        }
                    )
                    break
                params = None
                pages += 1
                raw_results = payload.get("results")
                if isinstance(raw_results, list):
                    for item in raw_results:
                        if not isinstance(item, dict):
                            continue
                        filing_date = _parse_date(item.get("filing_date"))
                        if filing_date is None or not (start_date <= filing_date <= end_date):
                            continue
                        results.append(item)
                next_url = payload.get("next_url")
                url = next_url if isinstance(next_url, str) and next_url else None
        payloads[ticker.upper()] = {"results": results}
    return payloads


def _sec_primary_document_url(
    *,
    cik: object,
    accession: object,
    primary_document: object,
) -> str | None:
    try:
        cik_text = (
            str(int(cik))
            if isinstance(cik, float) and cik.is_integer()
            else str(int(str(cik).strip()))
        )
    except (TypeError, ValueError):
        return None
    accession_text = str(accession or "").strip()
    primary_document_text = str(primary_document or "").strip()
    if not accession_text or not primary_document_text:
        return None
    accession_path = accession_text.replace("-", "")
    return _SEC_PRIMARY_DOCUMENT_URL_TEMPLATE.format(
        cik=cik_text,
        accession=accession_path,
        primary_document=primary_document_text,
    )


def _sec_primary_document_cache_path(
    cache_dir: Path,
    *,
    ticker: str,
    accession: str,
    primary_document: str,
) -> Path:
    safe_document = Path(primary_document).name or "primary_document.txt"
    return cache_dir / f"ticker={ticker.upper()}" / f"accession={accession}" / safe_document


def fetch_sec_primary_document_texts(
    *,
    candidates: pd.DataFrame,
    config: ProjectConfig,
    client: httpx.Client,
    cache_dir: Path,
    request_interval_seconds: float = 0.11,
) -> tuple[dict[tuple[str, str], str], list[dict[str, str]]]:
    text_by_accession: dict[tuple[str, str], str] = {}
    failures: list[dict[str, str]] = []
    if candidates.empty:
        return text_by_accession, failures
    last_request_at = 0.0
    seen: set[tuple[str, str]] = set()
    for row in candidates.to_dict("records"):
        ticker = str(row.get("ticker") or "").upper()
        accession = str(row.get("source_id") or "").strip()
        primary_document = str(row.get("primary_document") or "").strip()
        if not ticker or not accession:
            continue
        key = (ticker, accession)
        if key in seen:
            continue
        seen.add(key)
        url = _sec_primary_document_url(
            cik=row.get("cik"),
            accession=accession,
            primary_document=primary_document,
        )
        if url is None:
            failures.append(
                {
                    "ticker": ticker,
                    "accession": accession,
                    "reason": "missing_cik_or_primary_document",
                }
            )
            continue
        cache_path = _sec_primary_document_cache_path(
            cache_dir,
            ticker=ticker,
            accession=accession,
            primary_document=primary_document,
        )
        if cache_path.exists():
            try:
                text_by_accession[key] = cache_path.read_text(encoding="utf-8")
                continue
            except OSError as exc:
                failures.append(
                    {
                        "ticker": ticker,
                        "accession": accession,
                        "reason": "cache_read_failed",
                        "error": str(exc),
                    }
                )
        elapsed = time_module.monotonic() - last_request_at
        if elapsed < request_interval_seconds:
            time_module.sleep(request_interval_seconds - elapsed)
        try:
            text = _get_text_with_retries(
                client,
                url,
                headers={"User-Agent": config.sec_user_agent},
                max_retries=config.massive_max_retries,
                backoff_seconds=config.massive_retry_backoff_seconds,
            )
            last_request_at = time_module.monotonic()
        except Exception as exc:
            failures.append(
                {
                    "ticker": ticker,
                    "accession": accession,
                    "reason": "sec_primary_document_fetch_failed",
                    "error": str(exc),
                }
            )
            continue
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        cache_path.write_text(text, encoding="utf-8")
        text_by_accession[key] = text
    return text_by_accession, failures


def load_json_payloads_from_dir(
    tickers: Sequence[str], directory: Path
) -> dict[str, dict[str, Any]]:
    payloads: dict[str, dict[str, Any]] = {}
    for ticker in tickers:
        normalized = ticker.upper()
        candidates = [directory / f"{normalized}.json", directory / f"{normalized.lower()}.json"]
        path = next((candidate for candidate in candidates if candidate.exists()), None)
        if path is None:
            raise ValueError(f"missing fixture payload for {normalized} in {directory}")
        payload = path.read_text(encoding="utf-8")
        parsed = json.loads(payload)
        if not isinstance(parsed, dict):
            raise ValueError(f"fixture payload must be a JSON object: {path}")
        payloads[normalized] = parsed
    return payloads


def massive_text_by_accession(
    payloads: Mapping[str, Mapping[str, Any]],
) -> dict[tuple[str, str], str]:
    out: dict[tuple[str, str], str] = {}
    for ticker, payload in payloads.items():
        results = payload.get("results")
        if not isinstance(results, list):
            continue
        for item in results:
            if not isinstance(item, dict):
                continue
            accession = str(item.get("accession_number") or "")
            if not accession:
                continue
            out[(ticker.upper(), accession)] = str(item.get("items_text") or "")
    return out


def _validation_is_decisive(status: str) -> bool:
    return status in {
        "validated_earnings_release",
        "non_earnings_item_2_02",
        "not_item_2_02_text",
    }


def apply_text_validation(
    candidates: pd.DataFrame,
    text_by_accession: Mapping[tuple[str, str], str] | None,
    *,
    source_label: str = "massive_8k_text",
) -> pd.DataFrame:
    out = candidates.copy()
    statuses: list[str] = []
    reasons: list[str] = []
    sources: list[str] = []
    aux_statuses: list[str] = []
    validated: list[bool] = []
    if text_by_accession is None:
        out["text_validation_status"] = "validation_skipped"
        out["text_validation_reason"] = "Text validation was skipped."
        out["text_validation_source"] = "skipped"
        out["text_validation_aux_status"] = ""
        out["is_validated_earnings_event"] = False
        out["is_main_sample_candidate"] = False
        return out

    for row in out.to_dict("records"):
        key = (str(row.get("ticker") or "").upper(), str(row.get("source_id") or ""))
        status, reason = classify_8k_text(text_by_accession.get(key))
        statuses.append(status)
        reasons.append(reason)
        sources.append(source_label)
        aux_statuses.append("")
        validated.append(status == "validated_earnings_release")

    out["text_validation_status"] = statuses
    out["text_validation_reason"] = reasons
    out["text_validation_source"] = sources
    out["text_validation_aux_status"] = aux_statuses
    out["is_validated_earnings_event"] = validated
    out["is_main_sample_candidate"] = out["is_main_sample_timing"].astype(bool) & out[
        "is_validated_earnings_event"
    ].astype(bool)
    return out


def apply_official_then_aux_text_validation(
    candidates: pd.DataFrame,
    *,
    sec_text_by_accession: Mapping[tuple[str, str], str],
    aux_text_by_accession: Mapping[tuple[str, str], str] | None = None,
) -> pd.DataFrame:
    out = candidates.copy()
    statuses: list[str] = []
    reasons: list[str] = []
    sources: list[str] = []
    aux_statuses: list[str] = []
    validated: list[bool] = []
    for row in out.to_dict("records"):
        key = (str(row.get("ticker") or "").upper(), str(row.get("source_id") or ""))
        sec_status, sec_reason = classify_8k_text(sec_text_by_accession.get(key))
        status = sec_status
        reason = sec_reason
        source = "sec_primary_document_text"
        aux_status = ""
        if not _validation_is_decisive(sec_status) and aux_text_by_accession is not None:
            massive_status, massive_reason = classify_8k_text(aux_text_by_accession.get(key))
            aux_status = massive_status
            if _validation_is_decisive(massive_status):
                status = massive_status
                reason = f"Massive auxiliary fallback: {massive_reason}"
                source = "massive_8k_text_fallback"
            else:
                reason = f"{sec_reason} Massive auxiliary status: {massive_status}."
        statuses.append(status)
        reasons.append(reason)
        sources.append(source)
        aux_statuses.append(aux_status)
        validated.append(status == "validated_earnings_release")
    out["text_validation_status"] = statuses
    out["text_validation_reason"] = reasons
    out["text_validation_source"] = sources
    out["text_validation_aux_status"] = aux_statuses
    out["is_validated_earnings_event"] = validated
    out["is_main_sample_candidate"] = out["is_main_sample_timing"].astype(bool) & out[
        "is_validated_earnings_event"
    ].astype(bool)
    return out


def _value_counts(frame: pd.DataFrame, column: str) -> dict[str, int]:
    if column not in frame.columns or frame.empty:
        return {}
    counts = frame[column].astype(str).value_counts().to_dict()
    return {str(key): int(value) for key, value in counts.items()}


def build_earnings_calendar_report(
    *,
    frame: pd.DataFrame,
    tickers: Sequence[str],
    start_date: date,
    end_date: date,
    validation_route: str,
) -> dict[str, Any]:
    return {
        "source_route": "sec_edgar_submissions_plus_sec_primary_document_text",
        "validation_route": validation_route,
        "sample_tickers": [ticker.upper() for ticker in tickers],
        "start_date": start_date.isoformat(),
        "end_date": end_date.isoformat(),
        "snapshot_timestamp_utc": datetime.now(UTC).isoformat(),
        "row_count": int(len(frame)),
        "main_sample_candidate_rows": int(frame["is_main_sample_candidate"].sum())
        if "is_main_sample_candidate" in frame
        else 0,
        "rows_by_ticker": _value_counts(frame, "ticker"),
        "timing_counts": _value_counts(frame, "announcement_timing"),
        "acceptance_inferred_timing_counts": _value_counts(frame, "acceptance_inferred_timing"),
        "text_validation_counts": _value_counts(frame, "text_validation_status"),
        "text_validation_source_counts": _value_counts(frame, "text_validation_source"),
        "filing_acceptance_date_mismatch_rows": int(frame["filing_acceptance_date_mismatch"].sum())
        if "filing_acceptance_date_mismatch" in frame
        else 0,
        "limitations": [
            (
                "SEC acceptance time is a regulatory timestamp, not guaranteed first public "
                "release time."
            ),
            (
                "SEC primary filing text is the default validation source; Massive 8-K text "
                "is auxiliary fallback only when enabled and available."
            ),
            "DMH and UNKNOWN events are outside the first main sample.",
        ],
    }


def build_earnings_calendar_candidates(
    *,
    config: ProjectConfig,
    tickers: Sequence[str],
    start_date: date,
    end_date: date,
    sec_submissions_dir: Path | None = None,
    massive_8k_text_dir: Path | None = None,
    validate_with_massive: bool = True,
    http_client: httpx.Client | None = None,
    fail_on_missing_tickers: bool = True,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    normalized_tickers = sorted({ticker.upper() for ticker in tickers if ticker.strip()})
    if not normalized_tickers:
        raise ValueError("at least one ticker is required")
    if start_date > end_date:
        raise ValueError("start_date must be on or before end_date")

    owns_client = http_client is None
    client = http_client or httpx.Client(timeout=config.massive_request_timeout_seconds)
    try:
        if sec_submissions_dir is not None:
            sec_payloads = load_json_payloads_from_dir(normalized_tickers, sec_submissions_dir)
            sec_route = "fixture_dir"
            archive_cache_dir = None
        else:
            archive_cache_dir = config.bronze_data_dir / "sec" / "submissions"
            sec_payloads = fetch_sec_submission_payloads(
                tickers=normalized_tickers,
                config=config,
                client=client,
                archive_cache_dir=archive_cache_dir,
                fail_on_missing_tickers=fail_on_missing_tickers,
            )
            sec_route = "sec_edgar_http"

        frames: list[pd.DataFrame] = []
        for ticker, payload in sec_payloads.items():
            normalized = normalize_sec_submission_candidates(
                ticker=ticker,
                payload=payload,
                start_date=start_date,
                end_date=end_date,
            )
            frames.append(normalized)
            if archive_cache_dir is not None:
                summary_path = archive_cache_dir / f"ticker={ticker}" / "normalized_submissions.csv"
                summary_path.parent.mkdir(parents=True, exist_ok=True)
                normalized.to_csv(summary_path, index=False)
        candidates = (
            pd.concat(frames, ignore_index=True)
            if frames
            else pd.DataFrame(columns=CALENDAR_COLUMNS)
        )

        massive_payloads: dict[str, dict[str, Any]] | None = None
        sec_text_failures: list[dict[str, str]] = []
        sec_text_by_accession: dict[tuple[str, str], str] | None = None
        if sec_submissions_dir is None:
            sec_text_by_accession, sec_text_failures = fetch_sec_primary_document_texts(
                candidates=candidates,
                config=config,
                client=client,
                cache_dir=config.bronze_data_dir / "sec" / "primary_documents",
            )
            validation_route = "sec_primary_document_text"
        else:
            validation_route = "sec_primary_document_text_unavailable"

        should_fetch_massive_aux = (
            validate_with_massive
            and not candidates.empty
            and (
                sec_text_by_accession is None
                or candidates.apply(
                    lambda row: (
                        not _validation_is_decisive(
                            classify_8k_text(
                                sec_text_by_accession.get(
                                    (
                                        str(row.get("ticker") or "").upper(),
                                        str(row.get("source_id") or ""),
                                    )
                                )
                            )[0]
                        )
                    ),
                    axis=1,
                ).any()
            )
        )
        massive_aux_status = "not_requested"
        if should_fetch_massive_aux:
            if massive_8k_text_dir is not None:
                massive_payloads = load_json_payloads_from_dir(
                    normalized_tickers, massive_8k_text_dir
                )
                massive_aux_status = "fixture_dir"
            else:
                try:
                    massive_payloads = fetch_massive_8k_text_payloads(
                        tickers=normalized_tickers,
                        config=config,
                        client=client,
                        start_date=start_date,
                        end_date=end_date,
                    )
                    massive_aux_status = "massive_8k_text_http"
                except httpx.HTTPStatusError as exc:
                    massive_payloads = None
                    massive_aux_status = f"unavailable_http_{exc.response.status_code}"
                except (OSError, ValueError):
                    massive_payloads = None
                    massive_aux_status = "unavailable_missing_key"
                except httpx.HTTPError:
                    massive_payloads = None
                    massive_aux_status = "unavailable_fetch_failed"

        if sec_text_by_accession is not None:
            validated = apply_official_then_aux_text_validation(
                candidates,
                sec_text_by_accession=sec_text_by_accession,
                aux_text_by_accession=massive_text_by_accession(massive_payloads)
                if massive_payloads is not None
                else None,
            )
            if massive_aux_status not in {"not_requested", "unavailable_missing_key"}:
                validation_route = f"{validation_route}+{massive_aux_status}_auxiliary"
        else:
            if validate_with_massive and massive_payloads is not None:
                validation_route = massive_aux_status
                validated = apply_text_validation(
                    candidates,
                    massive_text_by_accession(massive_payloads),
                    source_label="massive_8k_text",
                )
            else:
                validation_route = "skipped"
                validated = apply_text_validation(candidates, None)
    finally:
        if owns_client:
            client.close()

    report = build_earnings_calendar_report(
        frame=validated,
        tickers=normalized_tickers,
        start_date=start_date,
        end_date=end_date,
        validation_route=f"{sec_route}+{validation_route}",
    )
    report["sec_primary_document_fetch_failed"] = len(sec_text_failures)
    report["massive_8k_aux_status"] = massive_aux_status
    report["sec_fetch_status_counts"] = {
        str(status): int(count)
        for status, count in pd.Series(
            [str(payload.get("sec_fetch_status") or "fixture") for payload in sec_payloads.values()]
        )
        .value_counts()
        .items()
    }
    report["sec_archive_payload_count"] = int(
        sum(
            len(payload.get("archive_payloads") or [])
            for payload in sec_payloads.values()
            if isinstance(payload.get("archive_payloads"), list)
        )
    )
    report["sec_archive_fetch_failed"] = int(
        sum(
            len(payload.get("sec_archive_fetch_failed") or [])
            for payload in sec_payloads.values()
            if isinstance(payload.get("sec_archive_fetch_failed"), list)
        )
    )
    report["massive_8k_fetch_failed"] = int(
        sum(
            1
            for payload in (massive_payloads or {}).values()
            for item in (payload.get("results") or [])
            if isinstance(item, dict) and item.get("fetch_failure")
        )
    )
    return validated, report

from __future__ import annotations

import json
import re
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
    "form_type",
    "sec_items",
    "report_date",
    "primary_document",
    "primary_doc_description",
    "timing_source",
    "timing_confidence",
    "text_validation_status",
    "text_validation_reason",
    "is_main_sample_timing",
    "is_validated_earnings_event",
    "is_main_sample_candidate",
]

_SEC_FORMS = {"8-K", "8-K/A"}
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


def classify_8k_text(items_text: str | None) -> tuple[str, str]:
    if not items_text:
        return "missing_text", "Massive 8-K text was not available for this accession."
    text = re.sub(r"\s+", " ", items_text).lower()
    if "item 2.02" not in text and "item 2.02." not in text:
        return "not_item_2_02_text", "The filing text does not contain Item 2.02."
    has_earnings_marker = any(marker in text for marker in _EARNINGS_MARKERS)
    has_non_earnings_marker = any(marker in text for marker in _NON_EARNINGS_MARKERS)
    if has_earnings_marker:
        return "validated_earnings_release", "Item 2.02 text describes quarterly results."
    if has_non_earnings_marker:
        return "non_earnings_item_2_02", "Item 2.02 text appears unrelated to quarterly earnings."
    return "ambiguous_item_2_02_text", "Item 2.02 is present but earnings-release wording is weak."


def _recent_value(recent: Mapping[str, Any], key: str, index: int) -> object:
    values = recent.get(key)
    if not isinstance(values, list) or index >= len(values):
        return None
    return values[index]


def normalize_sec_submission_candidates(
    *,
    ticker: str,
    payload: Mapping[str, Any],
    start_date: date,
    end_date: date,
) -> pd.DataFrame:
    recent = (
        (payload.get("filings") or {}).get("recent")
        if isinstance(payload.get("filings"), dict)
        else {}
    )
    if not isinstance(recent, dict):
        return pd.DataFrame(columns=CALENDAR_COLUMNS)

    forms = recent.get("form")
    if not isinstance(forms, list):
        return pd.DataFrame(columns=CALENDAR_COLUMNS)

    rows: list[dict[str, object]] = []
    for index, form_value in enumerate(forms):
        form = str(form_value or "").strip()
        if form not in _SEC_FORMS:
            continue
        items = str(_recent_value(recent, "items", index) or "")
        if "2.02" not in items:
            continue
        filing_date = _parse_date(_recent_value(recent, "filingDate", index))
        if filing_date is None or filing_date < start_date or filing_date > end_date:
            continue
        acceptance = _recent_value(recent, "acceptanceDateTime", index)
        timing = infer_timing_from_acceptance_timestamp(acceptance)
        rows.append(
            {
                "ticker": ticker.upper(),
                "announcement_date": filing_date.isoformat(),
                "announcement_timing": timing.value,
                "source": "sec_edgar_submissions",
                "source_timestamp": acceptance,
                "source_id": _recent_value(recent, "accessionNumber", index),
                "form_type": form,
                "sec_items": items,
                "report_date": _recent_value(recent, "reportDate", index),
                "primary_document": _recent_value(recent, "primaryDocument", index),
                "primary_doc_description": _recent_value(recent, "primaryDocDescription", index),
                "timing_source": "sec_acceptance_timestamp",
                "timing_confidence": "proxy",
                "text_validation_status": "pending_massive_text",
                "text_validation_reason": "",
                "is_main_sample_timing": timing in {AnnouncementTiming.BMO, AnnouncementTiming.AMC},
                "is_validated_earnings_event": False,
                "is_main_sample_candidate": False,
            }
        )
    return pd.DataFrame(rows, columns=CALENDAR_COLUMNS)


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
) -> dict[str, dict[str, Any]]:
    ticker_map = fetch_sec_ticker_map(client, config)
    payloads: dict[str, dict[str, Any]] = {}
    missing: list[str] = []
    for ticker in tickers:
        normalized = ticker.upper()
        cik = ticker_map.get(normalized)
        if cik is None:
            missing.append(normalized)
            continue
        payloads[normalized] = _get_json(
            client,
            config.sec_submissions_url_template.format(cik=cik),
            headers={"User-Agent": config.sec_user_agent},
        )
    if missing:
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
        raise ValueError("MASSIVE_API_KEY_FILE is required for Massive 8-K text validation.")

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
                payload = _get_json(client, url, params=params)
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


def apply_text_validation(
    candidates: pd.DataFrame,
    text_by_accession: Mapping[tuple[str, str], str] | None,
) -> pd.DataFrame:
    out = candidates.copy()
    statuses: list[str] = []
    reasons: list[str] = []
    validated: list[bool] = []
    if text_by_accession is None:
        out["text_validation_status"] = "validation_skipped"
        out["text_validation_reason"] = "Massive text validation was skipped."
        out["is_validated_earnings_event"] = False
        out["is_main_sample_candidate"] = False
        return out

    for row in out.to_dict("records"):
        key = (str(row.get("ticker") or "").upper(), str(row.get("source_id") or ""))
        status, reason = classify_8k_text(text_by_accession.get(key))
        statuses.append(status)
        reasons.append(reason)
        validated.append(status == "validated_earnings_release")

    out["text_validation_status"] = statuses
    out["text_validation_reason"] = reasons
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
        "source_route": "sec_edgar_submissions_plus_massive_8k_text",
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
        "text_validation_counts": _value_counts(frame, "text_validation_status"),
        "limitations": [
            (
                "SEC acceptance time is a regulatory timestamp, not guaranteed first public "
                "release time."
            ),
            (
                "Massive 8-K text is used for text validation and is not treated as the "
                "timestamp source."
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
        else:
            sec_payloads = fetch_sec_submission_payloads(
                tickers=normalized_tickers,
                config=config,
                client=client,
            )
            sec_route = "sec_edgar_http"

        frames = [
            normalize_sec_submission_candidates(
                ticker=ticker,
                payload=payload,
                start_date=start_date,
                end_date=end_date,
            )
            for ticker, payload in sec_payloads.items()
        ]
        candidates = (
            pd.concat(frames, ignore_index=True)
            if frames
            else pd.DataFrame(columns=CALENDAR_COLUMNS)
        )

        if validate_with_massive:
            if massive_8k_text_dir is not None:
                massive_payloads = load_json_payloads_from_dir(
                    normalized_tickers, massive_8k_text_dir
                )
                validation_route = "fixture_dir"
            else:
                massive_payloads = fetch_massive_8k_text_payloads(
                    tickers=normalized_tickers,
                    config=config,
                    client=client,
                    start_date=start_date,
                    end_date=end_date,
                )
                validation_route = "massive_8k_text_http"
            validated = apply_text_validation(
                candidates, massive_text_by_accession(massive_payloads)
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
    return validated, report

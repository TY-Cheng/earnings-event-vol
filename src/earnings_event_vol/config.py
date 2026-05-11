from __future__ import annotations

import os
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[2]
EXTERNAL_VOLUME_ROOT = Path("/Volumes/ExternalSSD")
EXTERNAL_DATA_DIR = EXTERNAL_VOLUME_ROOT / "data" / "earnings-event-vol"


def _expanded_path(value: str | Path) -> Path:
    return Path(os.path.expandvars(str(value))).expanduser().resolve()


def _repo_is_cloud_synced() -> bool:
    parts = set(REPO_ROOT.parts)
    return "CloudStorage" in parts or any(
        part.startswith(("OneDrive-", "GoogleDrive-", "Dropbox")) for part in parts
    )


def _default_data_dir() -> Path:
    if EXTERNAL_VOLUME_ROOT.exists():
        return EXTERNAL_DATA_DIR
    if _repo_is_cloud_synced():
        raise RuntimeError(
            "DATA_DIR is unset and /Volumes/ExternalSSD is not mounted; set DATA_DIR "
            "to an external path instead of creating repo-local data."
        )
    return REPO_ROOT / "data"


def _path_from_env(name: str, default: str | Path) -> Path:
    raw = os.getenv(name)
    return _expanded_path(raw if raw else default)


def _optional_path_from_env(name: str) -> Path | None:
    raw = os.getenv(name)
    if not raw:
        return None
    return _expanded_path(raw)


@dataclass(frozen=True)
class ProjectConfig:
    project_name: str
    repo_root: Path
    data_dir: Path
    bronze_data_dir: Path
    silver_data_dir: Path
    gold_data_dir: Path
    reports_dir: Path
    artifacts_dir: Path
    massive_api_key_file: Path | None
    massive_flat_file_key_file: Path | None
    massive_base_url: str
    massive_flat_file_endpoint_url: str
    massive_flat_file_bucket: str
    massive_options_flat_file_dataset: str
    massive_options_quotes_flat_file_dataset: str
    massive_options_flat_file_template: str
    massive_option_flat_file_key_template: str
    massive_option_quotes_flat_file_key_template: str
    massive_underlying_flat_file_template: str
    massive_underlying_flat_file_key_template: str
    massive_request_timeout_seconds: float
    massive_max_retries: int
    massive_retry_backoff_seconds: float
    massive_requests_per_minute: int | None
    sec_company_tickers_url: str
    sec_submissions_url_template: str
    sec_user_agent: str
    massive_8k_text_path: str
    fred_vixcls_url: str

    def as_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        for key, value in payload.items():
            if isinstance(value, Path):
                payload[key] = str(value)
        return payload


def _float_env(name: str, default: float) -> float:
    raw = os.getenv(name)
    if raw is None or raw == "":
        return default
    return float(raw)


def _int_env(name: str, default: int) -> int:
    raw = os.getenv(name)
    if raw is None or raw == "":
        return default
    return int(raw)


def _optional_int_env(name: str) -> int | None:
    raw = os.getenv(name)
    if raw is None or raw == "":
        return None
    return int(raw)


def load_project_config() -> ProjectConfig:
    data_dir = _path_from_env("DATA_DIR", _default_data_dir())
    return ProjectConfig(
        project_name=os.getenv("PROJECT_NAME", "earnings-event-vol"),
        repo_root=_path_from_env("PROJECT_ROOT", REPO_ROOT),
        data_dir=data_dir,
        bronze_data_dir=_path_from_env("BRONZE_DATA_DIR", data_dir / "bronze"),
        silver_data_dir=_path_from_env("SILVER_DATA_DIR", data_dir / "silver"),
        gold_data_dir=_path_from_env("GOLD_DATA_DIR", data_dir / "gold"),
        reports_dir=_path_from_env("REPORTS_DIR", REPO_ROOT / "reports"),
        artifacts_dir=_path_from_env("ARTIFACTS_DIR", REPO_ROOT / "artifacts"),
        massive_api_key_file=_optional_path_from_env("MASSIVE_API_KEY_FILE"),
        massive_flat_file_key_file=_optional_path_from_env("MASSIVE_FLAT_FILE_KEY_FILE"),
        massive_base_url=os.getenv("MASSIVE_BASE_URL", "https://api.massive.com"),
        massive_flat_file_endpoint_url=os.getenv(
            "MASSIVE_FLAT_FILE_ENDPOINT_URL", "https://files.massive.com"
        ),
        massive_flat_file_bucket=os.getenv("MASSIVE_FLAT_FILE_BUCKET", "flatfiles"),
        massive_options_flat_file_dataset=os.getenv(
            "MASSIVE_OPTIONS_FLAT_FILE_DATASET", "day_aggs_v1"
        ),
        massive_options_quotes_flat_file_dataset=os.getenv(
            "MASSIVE_OPTIONS_QUOTES_FLAT_FILE_DATASET", "quotes_v1"
        ),
        massive_options_flat_file_template=os.getenv(
            "MASSIVE_OPTIONS_FLAT_FILE_TEMPLATE",
            "s3://flatfiles/us_options_opra/day_aggs_v1/{year}/{month}/{date}.csv.gz",
        ),
        massive_option_flat_file_key_template=os.getenv(
            "MASSIVE_OPTION_FLAT_FILE_KEY_TEMPLATE",
            "us_options_opra/{dataset}/{year}/{month}/{date}.csv.gz",
        ),
        massive_option_quotes_flat_file_key_template=os.getenv(
            "MASSIVE_OPTION_QUOTES_FLAT_FILE_KEY_TEMPLATE",
            "us_options_opra/{dataset}/{year}/{month}/{date}.csv.gz",
        ),
        massive_underlying_flat_file_template=os.getenv(
            "MASSIVE_UNDERLYING_FLAT_FILE_TEMPLATE",
            "s3://flatfiles/us_stocks_sip/day_aggs_v1/{year}/{month}/{date}.csv.gz",
        ),
        massive_underlying_flat_file_key_template=os.getenv(
            "MASSIVE_UNDERLYING_FLAT_FILE_KEY_TEMPLATE",
            "us_stocks_sip/{dataset}/{year}/{month}/{date}.csv.gz",
        ),
        massive_request_timeout_seconds=_float_env("MASSIVE_REQUEST_TIMEOUT_SECONDS", 30.0),
        massive_max_retries=_int_env("MASSIVE_MAX_RETRIES", 3),
        massive_retry_backoff_seconds=_float_env("MASSIVE_RETRY_BACKOFF_SECONDS", 2.0),
        massive_requests_per_minute=_optional_int_env("MASSIVE_REQUESTS_PER_MINUTE"),
        sec_company_tickers_url=os.getenv(
            "SEC_COMPANY_TICKERS_URL", "https://www.sec.gov/files/company_tickers.json"
        ),
        sec_submissions_url_template=os.getenv(
            "SEC_SUBMISSIONS_URL_TEMPLATE",
            "https://data.sec.gov/submissions/CIK{cik:010d}.json",
        ),
        sec_user_agent=os.getenv(
            "SEC_USER_AGENT",
            "earnings-event-vol research contact no-reply@example.com",
        ),
        massive_8k_text_path=os.getenv("MASSIVE_8K_TEXT_PATH", "/stocks/filings/8-K/vX/text"),
        fred_vixcls_url=os.getenv(
            "FRED_VIXCLS_URL",
            "https://fred.stlouisfed.org/graph/fredgraph.csv?id=VIXCLS",
        ),
    )

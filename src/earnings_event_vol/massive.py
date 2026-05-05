from __future__ import annotations

import hashlib
import json
import os
import re
import subprocess
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from datetime import UTC, date, datetime
from pathlib import Path
from typing import Any

import pandas as pd

from earnings_event_vol.config import ProjectConfig

MASSIVE_FIELD_MAP: Mapping[str, tuple[str, ...]] = {
    "option_day_aggs": ("ticker", "volume", "open", "close", "high", "low", "window_start"),
    "option_quotes_flat_file": ("ticker", "bid_price", "ask_price", "sip_timestamp"),
    "quoted_bid_ask": ("bid", "ask"),
    "nbbo_bid_ask": ("nbbo_bid", "nbbo_ask"),
    "underlying_ohlcv": ("open", "high", "low", "close", "volume"),
    "option_contract": ("ticker", "expiration", "strike", "right"),
    "vendor_iv_greeks": ("implied_volatility", "delta", "gamma", "vega"),
    "local_iv_greeks": ("local_iv", "local_delta", "local_gamma", "local_vega"),
    "liquidity": ("volume", "open_interest"),
    "corporate_actions": ("split_ratio", "dividend_amount", "ex_dividend_date"),
    "earnings_calendar": ("announcement_date", "announcement_timing", "source"),
}


@dataclass(frozen=True)
class CredentialProbeResult:
    configured: bool
    exists: bool
    path: str | None


@dataclass(frozen=True)
class MassiveCommandResult:
    returncode: int
    stdout: str
    stderr: str


@dataclass(frozen=True)
class FlatFileObjectSpec:
    dataset: str
    key: str
    sample_allowed: bool = True


@dataclass(frozen=True)
class FlatFileHeadResult:
    dataset: str
    key: str
    ok: bool
    sample_allowed: bool = True
    size_bytes: int | None = None
    last_modified: str | None = None
    etag: str | None = None
    metadata_source: str | None = None
    error: str | None = None

    def as_dict(self) -> dict[str, object]:
        return {
            "dataset": self.dataset,
            "key": self.key,
            "ok": self.ok,
            "sample_allowed": self.sample_allowed,
            "size_bytes": self.size_bytes,
            "last_modified": self.last_modified,
            "etag": self.etag,
            "metadata_source": self.metadata_source,
            "error": self.error,
        }


MassiveCommandRunner = Callable[[Sequence[str], Mapping[str, str], float], MassiveCommandResult]


def read_secret_file(path: Path | None) -> str | None:
    if path is None:
        return None
    return path.expanduser().read_text(encoding="utf-8").strip()


def parse_flat_file_key_text(text: str) -> tuple[str, str]:
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    if len(lines) < 2:
        raise ValueError("Massive flat-file key file must contain access key and secret key.")
    return lines[0], lines[1]


def read_flat_file_credentials(path: Path | None) -> tuple[str, str]:
    text = read_secret_file(path)
    if text is None:
        raise ValueError("MASSIVE_FLAT_FILE_KEY_FILE is not configured.")
    return parse_flat_file_key_text(text)


def probe_key_file(path: Path | None) -> CredentialProbeResult:
    if path is None:
        return CredentialProbeResult(configured=False, exists=False, path=None)
    expanded = path.expanduser().resolve()
    return CredentialProbeResult(configured=True, exists=expanded.exists(), path=str(expanded))


def credential_probe(config: ProjectConfig) -> dict[str, object]:
    api_key = probe_key_file(config.massive_api_key_file)
    flat_file_key = probe_key_file(config.massive_flat_file_key_file)
    return {
        "ok": api_key.configured
        and api_key.exists
        and flat_file_key.configured
        and flat_file_key.exists,
        "api_key_file": api_key.__dict__,
        "flat_file_key_file": flat_file_key.__dict__,
        "base_url": config.massive_base_url,
        "flat_file_endpoint_url": config.massive_flat_file_endpoint_url,
        "field_map_keys": sorted(MASSIVE_FIELD_MAP),
    }


def option_flat_file_key(config: ProjectConfig, *, year: int, month: int, date: str) -> str:
    return config.massive_option_flat_file_key_template.format(
        dataset=config.massive_options_flat_file_dataset,
        year=f"{int(year):04d}",
        month=f"{int(month):02d}",
        date=date,
    )


def option_quotes_flat_file_key(config: ProjectConfig, *, year: int, month: int, date: str) -> str:
    return config.massive_option_quotes_flat_file_key_template.format(
        dataset=config.massive_options_quotes_flat_file_dataset,
        year=f"{int(year):04d}",
        month=f"{int(month):02d}",
        date=date,
    )


def underlying_flat_file_key(config: ProjectConfig, *, date: str) -> str:
    parsed = datetime.strptime(date, "%Y-%m-%d").date()
    return config.massive_underlying_flat_file_key_template.format(
        dataset="day_aggs_v1",
        year=f"{parsed.year:04d}",
        month=f"{parsed.month:02d}",
        date=date,
    )


def flat_file_object_specs(
    config: ProjectConfig, *, date_value: date
) -> tuple[FlatFileObjectSpec, ...]:
    date_text = date_value.isoformat()
    return (
        FlatFileObjectSpec(
            dataset="options_day_aggs",
            key=option_flat_file_key(
                config, year=date_value.year, month=date_value.month, date=date_text
            ),
        ),
        FlatFileObjectSpec(
            dataset="options_quotes",
            key=option_quotes_flat_file_key(
                config, year=date_value.year, month=date_value.month, date=date_text
            ),
            sample_allowed=False,
        ),
        FlatFileObjectSpec(
            dataset="underlying_day_aggs",
            key=underlying_flat_file_key(config, date=date_text),
        ),
    )


def massive_flat_file_aws_env(config: ProjectConfig) -> dict[str, str]:
    access_key_id, secret_access_key = read_flat_file_credentials(config.massive_flat_file_key_file)
    env = dict(os.environ)
    env.update(
        {
            "AWS_ACCESS_KEY_ID": access_key_id,
            "AWS_SECRET_ACCESS_KEY": secret_access_key,
            "AWS_DEFAULT_REGION": "us-east-1",
            "AWS_EC2_METADATA_DISABLED": "true",
        }
    )
    return env


def build_head_object_command(
    config: ProjectConfig, *, key: str, aws_executable: str = "aws"
) -> list[str]:
    return [
        aws_executable,
        "s3api",
        "head-object",
        "--bucket",
        config.massive_flat_file_bucket,
        "--key",
        key,
        "--endpoint-url",
        config.massive_flat_file_endpoint_url,
        "--no-cli-pager",
    ]


def build_ls_object_command(
    config: ProjectConfig, *, key: str, aws_executable: str = "aws"
) -> list[str]:
    return [
        aws_executable,
        "s3",
        "ls",
        f"s3://{config.massive_flat_file_bucket}/{key}",
        "--endpoint-url",
        config.massive_flat_file_endpoint_url,
        "--no-cli-pager",
    ]


def build_download_file_command(
    config: ProjectConfig, *, key: str, destination: Path, aws_executable: str = "aws"
) -> list[str]:
    return [
        aws_executable,
        "s3",
        "cp",
        f"s3://{config.massive_flat_file_bucket}/{key}",
        str(destination),
        "--endpoint-url",
        config.massive_flat_file_endpoint_url,
        "--no-cli-pager",
    ]


def _run_head_object_command(
    command: Sequence[str], env: Mapping[str, str], timeout_seconds: float
) -> MassiveCommandResult:
    try:
        completed = subprocess.run(
            list(command),
            env=dict(env),
            text=True,
            capture_output=True,
            timeout=timeout_seconds,
            check=False,
        )
    except FileNotFoundError as exc:
        return MassiveCommandResult(returncode=127, stdout="", stderr=str(exc))
    except subprocess.TimeoutExpired as exc:
        stdout = exc.stdout.decode() if isinstance(exc.stdout, bytes) else exc.stdout or ""
        stderr = exc.stderr.decode() if isinstance(exc.stderr, bytes) else exc.stderr or ""
        return MassiveCommandResult(
            returncode=124,
            stdout=stdout,
            stderr=stderr or f"timed out after {timeout_seconds} seconds",
        )
    return MassiveCommandResult(
        returncode=completed.returncode,
        stdout=completed.stdout,
        stderr=completed.stderr,
    )


_LS_OBJECT_PATTERN = re.compile(
    r"^(?P<date>\d{4}-\d{2}-\d{2})\s+"
    r"(?P<time>\d{2}:\d{2}:\d{2})\s+"
    r"(?P<size>\d+)\s+"
    r"(?P<name>.+)$"
)


def _ls_metadata_from_stdout(stdout: str) -> tuple[int, str] | None:
    for line in stdout.splitlines():
        match = _LS_OBJECT_PATTERN.match(line.strip())
        if match:
            size = int(match.group("size"))
            last_modified = f"{match.group('date')}T{match.group('time')}"
            return size, last_modified
    return None


def _safe_error_text(command_result: MassiveCommandResult) -> str:
    text = (command_result.stderr or command_result.stdout or "").strip()
    if not text:
        return f"aws command failed with exit code {command_result.returncode}"
    return text.splitlines()[-1][:300]


def head_flat_file_objects(
    config: ProjectConfig,
    *,
    date_value: date,
    aws_executable: str = "aws",
    runner: MassiveCommandRunner = _run_head_object_command,
) -> tuple[FlatFileHeadResult, ...]:
    env = massive_flat_file_aws_env(config)
    results: list[FlatFileHeadResult] = []
    for spec in flat_file_object_specs(config, date_value=date_value):
        command = build_head_object_command(config, key=spec.key, aws_executable=aws_executable)
        command_result = runner(command, env, config.massive_request_timeout_seconds)
        if command_result.returncode != 0:
            fallback = runner(
                build_ls_object_command(config, key=spec.key, aws_executable=aws_executable),
                env,
                config.massive_request_timeout_seconds,
            )
            ls_metadata = (
                _ls_metadata_from_stdout(fallback.stdout) if fallback.returncode == 0 else None
            )
            if ls_metadata is not None:
                size_bytes, last_modified = ls_metadata
                results.append(
                    FlatFileHeadResult(
                        dataset=spec.dataset,
                        key=spec.key,
                        ok=True,
                        sample_allowed=spec.sample_allowed,
                        size_bytes=size_bytes,
                        last_modified=last_modified,
                        metadata_source="s3_ls",
                    )
                )
                continue
            results.append(
                FlatFileHeadResult(
                    dataset=spec.dataset,
                    key=spec.key,
                    ok=False,
                    sample_allowed=spec.sample_allowed,
                    error=_safe_error_text(command_result),
                )
            )
            continue
        payload = json.loads(command_result.stdout)
        results.append(
            FlatFileHeadResult(
                dataset=spec.dataset,
                key=spec.key,
                ok=True,
                sample_allowed=spec.sample_allowed,
                size_bytes=int(payload["ContentLength"]),
                last_modified=str(payload.get("LastModified"))
                if payload.get("LastModified")
                else None,
                etag=str(payload.get("ETag")) if payload.get("ETag") else None,
                metadata_source="head_object",
            )
        )
    return tuple(results)


def massive_flat_file_manifest(
    config: ProjectConfig,
    *,
    date_value: date,
    run_head: bool = True,
    aws_executable: str = "aws",
    runner: MassiveCommandRunner = _run_head_object_command,
) -> dict[str, Any]:
    specs = flat_file_object_specs(config, date_value=date_value)
    if run_head:
        objects = [
            result.as_dict()
            for result in head_flat_file_objects(
                config,
                date_value=date_value,
                aws_executable=aws_executable,
                runner=runner,
            )
        ]
    else:
        objects = [
            {
                "dataset": spec.dataset,
                "key": spec.key,
                "ok": None,
                "sample_allowed": spec.sample_allowed,
                "size_bytes": None,
                "last_modified": None,
                "etag": None,
                "metadata_source": None,
                "error": None,
            }
            for spec in specs
        ]
    manifest: dict[str, Any] = {
        "date": date_value.isoformat(),
        "endpoint_url": config.massive_flat_file_endpoint_url,
        "bucket": config.massive_flat_file_bucket,
        "head_object_ran": run_head,
        "snapshot_timestamp_utc": datetime.now(UTC).isoformat(),
        "objects": objects,
    }
    stable_payload = json.dumps(
        {key: value for key, value in manifest.items() if key != "snapshot_timestamp_utc"},
        sort_keys=True,
        default=str,
    )
    manifest["manifest_hash"] = hashlib.sha256(stable_payload.encode("utf-8")).hexdigest()
    return manifest


def parse_massive_option_ticker(option_ticker: str) -> dict[str, object]:
    match = re.match(
        r"^O:(?P<underlying>.+?)(?P<yy>\d{2})(?P<mm>\d{2})(?P<dd>\d{2})"
        r"(?P<right>[CP])(?P<strike>\d{8})$",
        option_ticker,
    )
    if match is None:
        raise ValueError(f"unsupported Massive option ticker: {option_ticker}")
    expiration = date(
        2000 + int(match.group("yy")),
        int(match.group("mm")),
        int(match.group("dd")),
    )
    return {
        "option_symbol": option_ticker,
        "ticker": match.group("underlying"),
        "expiration": expiration,
        "right": "call" if match.group("right") == "C" else "put",
        "strike": int(match.group("strike")) / 1000.0,
    }


def normalize_option_day_aggs(raw: pd.DataFrame, *, quote_date: date) -> pd.DataFrame:
    parsed = pd.DataFrame([parse_massive_option_ticker(str(value)) for value in raw["ticker"]])
    out = parsed.copy()
    out["quote_date"] = quote_date
    out["option_open"] = pd.to_numeric(raw["open"], errors="coerce")
    out["option_high"] = pd.to_numeric(raw["high"], errors="coerce")
    out["option_low"] = pd.to_numeric(raw["low"], errors="coerce")
    out["option_close"] = pd.to_numeric(raw["close"], errors="coerce")
    out["volume"] = pd.to_numeric(raw["volume"], errors="coerce")
    out["transactions"] = pd.to_numeric(raw["transactions"], errors="coerce")
    out["window_start"] = pd.to_numeric(raw["window_start"], errors="coerce")
    out["source_dataset"] = "options_day_aggs"
    return out


def normalize_underlying_day_aggs(raw: pd.DataFrame, *, bar_date: date) -> pd.DataFrame:
    out = raw.copy()
    out["date"] = bar_date
    out["source_dataset"] = "underlying_day_aggs"
    return out[["ticker", "date", "open", "high", "low", "close", "volume", "source_dataset"]]


def download_sample_allowed_flat_files(
    config: ProjectConfig,
    *,
    date_value: date,
    out_dir: Path,
    aws_executable: str = "aws",
    runner: MassiveCommandRunner = _run_head_object_command,
) -> dict[str, Path]:
    env = massive_flat_file_aws_env(config)
    out_dir.mkdir(parents=True, exist_ok=True)
    paths: dict[str, Path] = {}
    for spec in flat_file_object_specs(config, date_value=date_value):
        if not spec.sample_allowed:
            continue
        destination = out_dir / f"{spec.dataset}_{date_value.isoformat()}.csv.gz"
        command = build_download_file_command(
            config, key=spec.key, destination=destination, aws_executable=aws_executable
        )
        command_result = runner(command, env, config.massive_request_timeout_seconds * 4)
        if command_result.returncode != 0:
            raise RuntimeError(f"failed to download {spec.key}: {_safe_error_text(command_result)}")
        paths[spec.dataset] = destination
    return paths


def build_massive_day_agg_sample(
    config: ProjectConfig,
    *,
    date_value: date,
    out_dir: Path,
    sample_rows: int = 25,
    aws_executable: str = "aws",
    runner: MassiveCommandRunner = _run_head_object_command,
) -> dict[str, Any]:
    manifest = massive_flat_file_manifest(
        config,
        date_value=date_value,
        run_head=True,
        aws_executable=aws_executable,
        runner=runner,
    )
    raw_dir = out_dir / "raw"
    raw_paths = download_sample_allowed_flat_files(
        config,
        date_value=date_value,
        out_dir=raw_dir,
        aws_executable=aws_executable,
        runner=runner,
    )
    raw_options = pd.read_csv(raw_paths["options_day_aggs"], nrows=sample_rows)
    raw_underlying = pd.read_csv(raw_paths["underlying_day_aggs"], nrows=sample_rows)
    options = normalize_option_day_aggs(raw_options, quote_date=date_value)
    underlying = normalize_underlying_day_aggs(raw_underlying, bar_date=date_value)

    out_dir.mkdir(parents=True, exist_ok=True)
    raw_options.to_csv(out_dir / "options_day_aggs_raw_head.csv", index=False)
    raw_underlying.to_csv(out_dir / "underlying_day_aggs_raw_head.csv", index=False)
    options.to_csv(out_dir / "options_day_aggs_normalized_head.csv", index=False)
    underlying.to_csv(out_dir / "underlying_day_aggs_normalized_head.csv", index=False)

    missing_quote_fields = sorted({"bid", "ask", "open_interest"} - set(options.columns))
    missing_iv_fields = sorted({"vendor_iv", "delta", "gamma", "vega"} - set(options.columns))
    report: dict[str, Any] = {
        "date": date_value.isoformat(),
        "sample_rows": sample_rows,
        "raw_paths": {dataset: str(path) for dataset, path in raw_paths.items()},
        "raw_columns": {
            "options_day_aggs": list(raw_options.columns),
            "underlying_day_aggs": list(raw_underlying.columns),
        },
        "normalized_columns": {
            "options_day_aggs": list(options.columns),
            "underlying_day_aggs": list(underlying.columns),
        },
        "row_counts": {
            "options_day_aggs": int(len(options)),
            "underlying_day_aggs": int(len(underlying)),
        },
        "v1_readiness": {
            "day_aggs_support_contract_parsing": not options.empty,
            "day_aggs_support_underlying_close": "close" in underlying.columns,
            "day_aggs_support_bid_ask_costs": False,
            "day_aggs_support_ivar_extraction": False,
            "missing_quote_fields": missing_quote_fields,
            "missing_iv_greek_fields": missing_iv_fields,
            "required_next_dataset": (
                "options_quotes_v1 plus IV/Greeks/OI source or local IV solver"
            ),
        },
        "manifest": manifest,
    }
    (out_dir / "massive_sample_schema_report.json").write_text(
        json.dumps(report, indent=2, sort_keys=True, default=str),
        encoding="utf-8",
    )
    return report

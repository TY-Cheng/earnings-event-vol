from __future__ import annotations

import json
import tomllib
from pathlib import Path

import pytest
import yaml

from earnings_event_vol.cli import main
from earnings_event_vol.config import load_project_config

REPO_ROOT = Path(__file__).resolve().parents[1]


class _MkDocsYamlLoader(yaml.SafeLoader):
    pass


def _construct_python_name(loader: yaml.Loader, suffix: str, node: yaml.Node) -> str:
    return suffix


_MkDocsYamlLoader.add_multi_constructor(  # type: ignore[no-untyped-call]
    "tag:yaml.org,2002:python/name:",
    _construct_python_name,
)


def _read(path: str) -> str:
    return (REPO_ROOT / path).read_text(encoding="utf-8")


def test_project_identity_uses_earnings_event_vol() -> None:
    pyproject = tomllib.loads(_read("pyproject.toml"))
    readme = _read("README.md")
    env_example = _read(".env.example")

    assert pyproject["project"]["name"] == "earnings-event-vol"
    assert "# Earnings Event Vol" in readme
    assert "log" + "-iv" not in readme.lower()
    assert 'UV_PROJECT_ENVIRONMENT="${HOME}/.venvs/earnings-event-vol"' in env_example
    assert 'PROJECT_NAME="earnings-event-vol"' in env_example


def test_docs_front_door_matches_project() -> None:
    mkdocs = yaml.load(_read("mkdocs.yml"), Loader=_MkDocsYamlLoader)
    home = _read("docs/index.md")

    assert mkdocs["site_name"] == "Earnings Event Vol"
    assert mkdocs["repo_name"] == "TY-Cheng/earnings-event-vol"
    assert home.strip().endswith('--8<-- "README.md:docs-home"')
    assert mkdocs["nav"] == [
        {"Home": "index.md"},
        {"Results Snapshot": "results_snapshot.md"},
        {"Paper Plan": "paper_plan.md"},
        {
            "Audit Prompts": [
                {"Development Audit": "development_audit_prompt.md"},
                {"Manuscript Audit": "manuscript_audit_prompt.md"},
            ]
        },
        {"Future Work": "future_work.md"},
    ]


def test_config_loads_project_identity(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("PROJECT_NAME", "earnings-event-vol")
    monkeypatch.setenv("MASSIVE_REQUESTS_PER_MINUTE", "3000")

    config = load_project_config()

    assert config.project_name == "earnings-event-vol"
    assert config.massive_requests_per_minute == 3000
    assert config.as_dict()["project_name"] == "earnings-event-vol"


def test_cli_status_reports_project_without_secret_values(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.setenv("PROJECT_NAME", "earnings-event-vol")
    monkeypatch.delenv("MASSIVE_API_KEY_FILE", raising=False)
    monkeypatch.delenv("MASSIVE_FLAT_FILE_KEY_FILE", raising=False)

    assert main(["status"]) == 0
    output = capsys.readouterr().out
    payload = json.loads(output)

    assert payload["project_name"] == "earnings-event-vol"
    assert "massive" in payload
    assert "secret" not in output.lower()


def test_cli_source_probe_detects_missing_massive_files(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    api_key = tmp_path / "massive_api_key"
    flat_file_key = tmp_path / "massive_flat_file_key"
    api_key.write_text("redacted", encoding="utf-8")
    flat_file_key.write_text("redacted", encoding="utf-8")
    monkeypatch.setenv("MASSIVE_API_KEY_FILE", str(api_key))
    monkeypatch.setenv("MASSIVE_FLAT_FILE_KEY_FILE", str(flat_file_key))

    assert main(["source-probe", "massive"]) == 0
    payload = json.loads(capsys.readouterr().out)

    assert payload["status"]["ok"] is True
    assert payload["status"]["api_key_file"]["exists"] is True

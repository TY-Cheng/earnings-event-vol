set dotenv-load := true
set shell := ["bash", "-cu"]

cli := "PYTHONPATH=src uv run --env-file .env python -m earnings_event_vol.cli"
active_src := "src/earnings_event_vol"
active_tests := "tests"
format_paths := "src/earnings_event_vol tests scripts"

default:
    @just --list

_require-external-uv-env:
    @python3 -c 'import os, sys; from pathlib import Path; raw = os.environ.get("UV_PROJECT_ENVIRONMENT", ""); repo = Path("{{ justfile_directory() }}").resolve(); expanded = Path(os.path.expanduser(os.path.expandvars(raw))) if raw else Path(); missing = not raw; relative = bool(raw) and not expanded.is_absolute(); inside = False if missing or relative else expanded.resolve().is_relative_to(repo); reason = "is required" if missing else "must be an absolute path" if relative else "must be outside the repo" if inside else "ok"; print("UV_PROJECT_ENVIRONMENT=" + (raw or "<unset>")); sys.exit(0 if reason == "ok" else (print("error: UV_PROJECT_ENVIRONMENT " + reason, file=sys.stderr) or 1))'

_sync: _require-external-uv-env
    uv sync --all-extras --dev

_format: _sync
    uv run ruff format {{ format_paths }}
    uv run ruff check --fix {{ format_paths }}

check: _format
    uv run mypy {{ active_src }} {{ active_tests }}
    uv run pytest
    uv run mkdocs build --strict --clean
    {{ cli }} status
    {{ cli }} source-probe all

status: _require-external-uv-env
    {{ cli }} status

audit date="": _format
    @probe_date="{{ date }}"; probe_date="${probe_date#date=}"; if [[ -n "$probe_date" ]]; then {{ cli }} massive-flat-files --date "$probe_date" --out artifacts/massive_flat_file_probe; else {{ cli }} audit-data --quotes tests/fixtures/option_quotes.csv --underlying tests/fixtures/underlying_bars.csv --earnings tests/fixtures/earnings_calendar.csv --out artifacts/audit_data_fixtures; fi

data stage="proxy-all" args="": _require-external-uv-env
    @stage='{{ stage }}'; extra='{{ args }}'; if [[ "$stage" == args=* ]]; then extra="${stage#args=}"; stage="proxy-all"; elif [[ "$stage" == --* ]]; then extra="$stage ${extra#args=}"; stage="proxy-all"; else extra="${extra#args=}"; fi; defaults=(); if [[ "$stage" == "proxy-all" ]]; then defaults=(--start 2022-12-01 --end 2025-12-31 --jobs 4 --lookback-seconds 900 --second-agg-buffer-minutes 60 --price-field option_vwap --dte-min 3 --dte-max 21 --universe-top-n 50 --universe-trailing-months 6); elif [[ "$stage" == "trade-proxy-panel" || "$stage" == "market-second-covariates" ]]; then defaults=(--jobs 4 --lookback-seconds 900 --second-agg-buffer-minutes 60 --price-field option_vwap); fi; read -r -a extra_args <<< "$extra"; {{ cli }} data --stage "$stage" "${defaults[@]}" "${extra_args[@]}"

research args="": _sync
    @extra='{{ args }}'; extra="${extra#args=}"; read -r -a extra_args <<< "$extra"; {{ cli }} research "${extra_args[@]}"

docs port="8000": _format
    uv run mkdocs build --strict --clean
    @port=$(python3 -c 'import socket, sys; host = "127.0.0.1"; start = int(sys.argv[1]); print(next(p for p in range(start, start + 100) if socket.socket().connect_ex((host, p))))' "{{ port }}"); echo "Serving docs at http://127.0.0.1:${port}"; uv run mkdocs serve -a 127.0.0.1:${port}

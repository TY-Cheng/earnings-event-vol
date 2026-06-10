set dotenv-load := true
set shell := ["bash", "-cu"]
export PATH := env_var_or_default("HOME", "") + "/.local/bin:" + env_var_or_default("PATH", "")

cli := "PYTHONPATH=src uv run --env-file .env python -m earnings_event_vol.cli"
active_src := "src/earnings_event_vol"
active_tests := "tests"
format_paths := "src/earnings_event_vol tests scripts"

default:
    @just --list

_require-external-uv-env:
    @python3 -c 'import os, sys; from pathlib import Path; raw = os.environ.get("UV_PROJECT_ENVIRONMENT", ""); repo = Path("{{ justfile_directory() }}").resolve(); expanded = Path(os.path.expanduser(os.path.expandvars(raw))) if raw else Path(); missing = not raw; relative = bool(raw) and not expanded.is_absolute(); inside = False if missing or relative else expanded.resolve().is_relative_to(repo); reason = "is required" if missing else "must be an absolute path" if relative else "must be outside the repo" if inside else "ok"; print("UV_PROJECT_ENVIRONMENT=" + (raw or "<unset>")); sys.exit(0 if reason == "ok" else (print("error: UV_PROJECT_ENVIRONMENT " + reason, file=sys.stderr) or 1))'

_require-external-data:
    @python3 -c 'import os, sys; from pathlib import Path; repo = Path("{{ justfile_directory() }}").resolve(); raw = os.environ.get("DATA_DIR", ""); path = Path(os.path.expanduser(os.path.expandvars(raw))) if raw else Path(); cloud = "CloudStorage" in path.parts or any(part.startswith(("OneDrive-", "GoogleDrive-", "Dropbox")) for part in path.parts); bad = (not raw) or (not path.is_absolute()) or path.resolve().is_relative_to(repo) or cloud or (not path.exists()); print("execution_root=" + str(repo)); print("DATA_DIR=" + (str(path) if raw else "<unset>")); sys.exit((print("error: DATA_DIR must be an existing absolute path outside repo/cloud storage", file=sys.stderr) or 1) if bad else 0)'

_sync: _require-external-uv-env
    uv sync --all-extras --dev --inexact

mamba-install: _sync
    @uv run python -c 'import shutil; nvcc = shutil.which("nvcc"); print("nvcc=" + str(nvcc or "unavailable")); print("mamba-install uses prebuilt wheels; nvcc is optional for this recipe")'
    uv pip install --python "$UV_PROJECT_ENVIRONMENT/bin/python" --torch-backend cu130 "torch==2.11.0"
    uv pip install --python "$UV_PROJECT_ENVIRONMENT/bin/python" --only-binary :all: "https://github.com/Dao-AILab/causal-conv1d/releases/download/v1.6.1.post4/causal_conv1d-1.6.1%2Bcu13torch2.10cxx11abiTRUE-cp313-cp313-linux_x86_64.whl" "https://github.com/state-spaces/mamba/releases/download/v2.3.1/mamba_ssm-2.3.1%2Bcu13torch2.10cxx11abiTRUE-cp313-cp313-linux_x86_64.whl"

mamba-doctor: _require-external-uv-env
    @uv run python -c 'import importlib.util, platform, shutil, torch; probe=lambda name: "available version=" + str(getattr(__import__(name), "__version__", "unknown")) if importlib.util.find_spec(name) else "unavailable"; print(f"os={platform.system()} {platform.release()} machine={platform.machine()}"); print(f"torch={torch.__version__}"); print(f"torch_cuda_available={torch.cuda.is_available()}"); print(f"torch_cuda_version={torch.version.cuda}"); print(f"cuda_device_count={torch.cuda.device_count()}"); print("nvcc=" + str(shutil.which("nvcc") or "unavailable")); print("mamba_ssm=" + probe("mamba_ssm")); print("causal_conv1d=" + probe("causal_conv1d"))'

_format: _sync
    uv run ruff format {{ format_paths }}
    uv run ruff check --fix {{ format_paths }}

_sync-doc-figures:
    @mkdir -p docs/assets/images/modeling
    @reports_dir="${REPORTS_DIR:-reports}"; shopt -s nullglob; figures=("${reports_dir}"/modeling/figures/*.png); if (( ${#figures[@]} == 0 )); then echo "No ${reports_dir}/modeling/figures/*.png to sync"; exit 0; fi; cp "${figures[@]}" docs/assets/images/modeling/

_check-doc-figures:
    @reports_dir="${REPORTS_DIR:-reports}"; shopt -s nullglob; figures=("${reports_dir}"/modeling/figures/*.png); if (( ${#figures[@]} == 0 )); then echo "No ${reports_dir}/modeling/figures/*.png to check"; exit 0; fi; failures=0; for src in "${figures[@]}"; do dest="docs/assets/images/modeling/$(basename "$src")"; if [[ ! -f "$dest" ]]; then echo "missing synced docs figure: $dest"; failures=1; elif ! cmp -s "$src" "$dest"; then echo "stale synced docs figure: $dest"; failures=1; fi; done; docs_figures=(docs/assets/images/modeling/*.png); for dest in "${docs_figures[@]}"; do src="${reports_dir}/modeling/figures/$(basename "$dest")"; if [[ ! -f "$src" ]]; then echo "docs figure has no report source: $dest"; failures=1; fi; done; exit "$failures"

check: _format
    uv run mypy {{ active_src }} {{ active_tests }}
    TMPDIR=/tmp uv run pytest
    uv run mkdocs build --strict --clean
    just _check-doc-figures
    {{ cli }} status
    {{ cli }} source-probe all

status: _require-external-uv-env
    {{ cli }} status

audit date="": _format
    @probe_date="{{ date }}"; probe_date="${probe_date#date=}"; if [[ -n "$probe_date" ]]; then {{ cli }} massive-flat-files --date "$probe_date" --out artifacts/massive_flat_file_probe; else {{ cli }} audit-data --quotes tests/fixtures/option_quotes.csv --underlying tests/fixtures/underlying_bars.csv --earnings tests/fixtures/earnings_calendar.csv --out artifacts/audit_data_fixtures; fi

data stage="all" args="": _require-external-uv-env _require-external-data
    @stage='{{ stage }}'; extra='{{ args }}'; if [[ "$stage" == args=* ]]; then extra="${stage#args=}"; stage="all"; elif [[ "$stage" == --* ]]; then extra="$stage ${extra#args=}"; stage="all"; else extra="${extra#args=}"; fi; defaults=(); if [[ "$stage" == "all" ]]; then defaults=(--start 2013-01-01 --end 2025-12-31 --jobs 4 --lookback-seconds 900 --second-agg-buffer-minutes 60 --price-field option_vwap --dte-min 3 --dte-max 21 --universe-top-n 50 --universe-trailing-months 6); elif [[ "$stage" == "event-window-panel" ]]; then defaults=(--dte-min 3 --dte-max 21); elif [[ "$stage" == "trade-proxy-panel" || "$stage" == "market-second-covariates" ]]; then defaults=(--jobs 4 --lookback-seconds 900 --second-agg-buffer-minutes 60 --price-field option_vwap); elif [[ "$stage" == "contract-reference-validation" ]]; then defaults=(--jobs 4); fi; read -r -a extra_args <<< "$extra"; {{ cli }} data --stage "$stage" "${defaults[@]}" "${extra_args[@]}"

research args="": _require-external-data _sync
    @extra='{{ args }}'; extra="${extra#args=}"; if [[ -z "$extra" ]]; then extra="--stage all --sequence-suite all --allow-high-sequence-risk --bootstrap-iter 200"; fi; read -r -a extra_args <<< "$extra"; {{ cli }} research "${extra_args[@]}"
    just _sync-doc-figures

research-fast: _require-external-data _sync
    {{ cli }} research --stage models --sequence-suite none --allow-high-sequence-risk --bootstrap-iter 50 --reuse-tuning-params
    {{ cli }} research --stage report --sequence-suite none --allow-high-sequence-risk --bootstrap-iter 50 --reuse-tuning-params
    just _sync-doc-figures

research-report: _require-external-data _sync
    {{ cli }} research --stage report --sequence-suite all --allow-high-sequence-risk --bootstrap-iter 200
    just _sync-doc-figures

docs port="8000": _format
    uv run mkdocs build --strict --clean
    @port=$(python3 -c 'import socket, sys; host = "127.0.0.1"; start = int(sys.argv[1]); print(next(p for p in range(start, start + 100) if socket.socket().connect_ex((host, p))))' "{{ port }}"); echo "Serving docs at http://127.0.0.1:${port}"; uv run mkdocs serve -a 127.0.0.1:${port}

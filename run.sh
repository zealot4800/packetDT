#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT_DIR"

PYTHON_BIN="${PYTHON:-python}"
LOG_DIR="$ROOT_DIR/logs"
mkdir -p "$LOG_DIR"

configs=(
  "configs/cic-ids-2017-c10-bo.yml"
  "configs/cic-ids-2018-c10-bo.yml"
  "configs/cic-iot-2023-c4-bo.yml"
  "configs/cic-iot-2023-c32-bo.yml"
  "configs/cic-iomt-2024-c19-bo.yml"
)

usage() {
  cat <<'EOF'
Usage:
  ./run.sh splidt [config]
  ./run.sh leo [config]
  ./run.sh netbeacon [config]
  ./run.sh iisy [config]
  ./run.sh all-models [config]
  ./run.sh all-datasets
  ./run.sh all

Defaults:
  config = configs/cic-ids-2017-c10-bo.yml

Notes:
  all-datasets runs SpliDT/CAP for every local config.
  all runs SpliDT/CAP, LEO, NetBeacon, and IIsy for every local config.
  IIsy is skipped automatically when dataset_df_p0.pkl is unavailable.
EOF
}

dataset_dir_for_config() {
  "$PYTHON_BIN" - "$1" <<'PY'
import os
import sys
import yaml

with open(sys.argv[1], "r") as config_file:
    config = yaml.safe_load(config_file)

dataset = config["dataset"]
print(os.path.join(dataset["path"], dataset["name"], dataset["destination"]))
PY
}

run_model() {
  local label="$1"
  local script="$2"
  local config="$3"
  local name
  local timestamp
  local log_file

  name="$(basename "$config" .yml)"
  timestamp="$(date +%Y%m%d-%H%M%S)"
  log_file="$LOG_DIR/${name}-${label}-${timestamp}.log"

  echo "=== Running ${label} with ${config} ==="
  echo "Log: $log_file"
  "$PYTHON_BIN" "$script" --config "$config" 2>&1 | tee "$log_file"
}

run_iisy_if_available() {
  local config="$1"
  local dataset_dir

  dataset_dir="$(dataset_dir_for_config "$config")"
  if [[ -f "$dataset_dir/dataset_df_p0.pkl" ]]; then
    run_model "iisy" "src/iisy.py" "$config"
  else
    echo "Skipping iisy for ${config}: ${dataset_dir}/dataset_df_p0.pkl is not available."
  fi
}

run_all_models_for_config() {
  local config="$1"

  run_model "splidt" "src/train.py" "$config"
  run_model "leo" "src/leo.py" "$config"
  run_model "netbeacon" "src/netbeacon.py" "$config"
  run_iisy_if_available "$config"
}

mode="${1:-}"
config="${2:-configs/cic-ids-2017-c10-bo.yml}"

case "$mode" in
  splidt)
    run_model "splidt" "src/train.py" "$config"
    ;;
  leo)
    run_model "leo" "src/leo.py" "$config"
    ;;
  netbeacon)
    run_model "netbeacon" "src/netbeacon.py" "$config"
    ;;
  iisy)
    run_iisy_if_available "$config"
    ;;
  all-models)
    run_all_models_for_config "$config"
    ;;
  all-datasets)
    for config in "${configs[@]}"; do
      run_model "splidt" "src/train.py" "$config"
    done
    ;;
  all)
    for config in "${configs[@]}"; do
      run_all_models_for_config "$config"
    done
    ;;
  -h|--help|help|"")
    usage
    ;;
  *)
    echo "Unknown mode: $mode" >&2
    usage >&2
    exit 2
    ;;
esac

echo "Run completed."

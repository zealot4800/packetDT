#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT_DIR"

if [[ -n "${PYTHON:-}" ]]; then
  PYTHON_BIN="$PYTHON"
elif [[ -x "$ROOT_DIR/.venv/bin/python" ]]; then
  PYTHON_BIN="$ROOT_DIR/.venv/bin/python"
elif command -v python >/dev/null 2>&1; then
  PYTHON_BIN="python"
else
  PYTHON_BIN="python3"
fi
COMMAND="${1:-}"

if [[ -z "$COMMAND" ]]; then
  "$PYTHON_BIN" -m src.main --help
  exit 0
fi

if [[ "$COMMAND" == "all-flows" ]]; then
  DATASET_CONFIGS=("$ROOT_DIR"/configs/datasets/*.yaml)
  if [[ ! -e "${DATASET_CONFIGS[0]}" ]]; then
    echo "No dataset configurations found under configs/datasets/" >&2
    exit 1
  fi

  RUN_STATUS=0
  for CONFIG_PATH in "${DATASET_CONFIGS[@]}"; do
    echo "Running all models and flow counts with ${CONFIG_PATH#"$ROOT_DIR"/}"
    if ! "$PYTHON_BIN" -m src.main degradation --config "$CONFIG_PATH"; then
      echo "Flow-count run failed for ${CONFIG_PATH#"$ROOT_DIR"/}" >&2
      RUN_STATUS=1
    fi
  done
  exit "$RUN_STATUS"
elif [[ "$COMMAND" == "example" ]]; then
  "$PYTHON_BIN" -m src.main example
else
  CONFIG="${2:-configs/datasets/cic_ids_2017.yaml}"
  "$PYTHON_BIN" -m src.main "$COMMAND" --config "$CONFIG"
fi

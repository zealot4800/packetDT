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

if [[ "$COMMAND" == "example" ]]; then
  "$PYTHON_BIN" -m src.main example
else
  CONFIG="${2:-configs/datasets/cic_ids_2017.yaml}"
  "$PYTHON_BIN" -m src.main "$COMMAND" --config "$CONFIG"
fi

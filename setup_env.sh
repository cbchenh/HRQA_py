#!/usr/bin/env bash
# One-shot environment setup for HRQA.
# Usage:
#   bash setup_env.sh           # uses conda if available, else venv
#   bash setup_env.sh --venv    # force venv
#   bash setup_env.sh --conda   # force conda
set -euo pipefail

MODE="auto"
if [[ "${1:-}" == "--venv" ]]; then MODE="venv"; fi
if [[ "${1:-}" == "--conda" ]]; then MODE="conda"; fi

if [[ "$MODE" == "auto" ]]; then
  if command -v conda >/dev/null 2>&1; then MODE="conda"; else MODE="venv"; fi
fi

cd "$(dirname "$0")"

if [[ "$MODE" == "conda" ]]; then
  echo "[hrqa] Creating conda env 'hrqa' from environment.yml ..."
  conda env create -f environment.yml -y || conda env update -f environment.yml --prune
  echo "[hrqa] Installing package in editable mode ..."
  conda run -n hrqa pip install -e .
  echo
  echo "Done. Activate with:  conda activate hrqa"
else
  echo "[hrqa] Creating venv at .venv ..."
  python3 -m venv .venv
  # shellcheck disable=SC1091
  source .venv/bin/activate
  pip install --upgrade pip
  pip install -r requirements.txt
  pip install -e .
  echo
  echo "Done. Activate with:  source .venv/bin/activate"
fi

echo
echo "Quick test:"
echo "  pytest tests/"
echo "  python demos/demo_signal.py"

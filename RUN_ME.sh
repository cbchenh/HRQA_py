#!/usr/bin/env bash
# One-shot setup + run. Works on Linux / WSL / macOS.
set -e
cd "$(dirname "$0")"

echo "=== HRQA Python — one-shot setup ==="

# 1. venv
if [ ! -d ".venv" ]; then
  echo "[1/4] Creating virtual environment (.venv)..."
  python3 -m venv .venv
else
  echo "[1/4] Reusing existing .venv"
fi

# 2. activate
# shellcheck disable=SC1091
source .venv/bin/activate

# 3. install
echo "[2/4] Installing dependencies (this may take a minute)..."
pip install --quiet --upgrade pip
pip install --quiet numpy scipy scikit-learn numba joblib pandas matplotlib pytest
pip install --quiet -e .

# 4. demo
echo
echo "[3/4] Running demo_signal.py ..."
echo "----------------------------------------"
python demos/demo_signal.py
echo "----------------------------------------"

# 5. tests
echo
echo "[4/4] Running test suite ..."
echo "----------------------------------------"
pytest tests/ -q
echo "----------------------------------------"

echo
echo "All done. To use the env later:"
echo "    source .venv/bin/activate"
echo "    python demos/demo_signal.py"
echo "    python demos/demo_sequence.py"
echo "    python demos/benchmark.py"

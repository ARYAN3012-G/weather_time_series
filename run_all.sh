#!/usr/bin/env bash
# v3 drought forecasting — full run on the lab desktop (Linux / macOS / Git Bash)
# Run from the project root:   bash run_all.sh
# Each step can be re-run safely; finished model runs are skipped.
set -euo pipefail
cd "$(dirname "$0")"

# Use the project virtual environment if it exists (Windows: .venv/Scripts, Linux/macOS: .venv/bin)
if   [ -x .venv/Scripts/python.exe ]; then PY=.venv/Scripts/python.exe
elif [ -x .venv/bin/python ];         then PY=.venv/bin/python
else PY=python; fi

echo "[0/7] Installing requirements";             "$PY" -m pip install -r requirements.txt
echo "[1/7] Audit + dedupe + SPEI recalibration";  "$PY" src/dataprep/prepare_data.py
echo "[2/7] Integrity / leakage tests";            "$PY" -m pytest tests -q
echo "[3/7] Smoke test";                           "$PY" src/experiments/run_experiments.py --quick --horizons 7 --seeds 0
echo "[4/7] MAIN STUDY (temporal)";                "$PY" src/experiments/run_experiments.py --split temporal
echo "[5/7] Leave-one-cluster-out";                "$PY" src/experiments/run_experiments.py --split loco --feature-sets weather weather_spei
echo "[6/7] Marusov et al. protocol";              "$PY" src/experiments/marusov_protocol.py
echo "[7/7] Aggregate";                            "$PY" src/experiments/aggregate.py
echo "DONE. See results/v3/tables/REPORT_*.md and results/v3/marusov_protocol/MARUSOV_PROTOCOL.md"

#!/usr/bin/env bash
set -euo pipefail

# Full reproducible benchmark pipeline (LOSO, both devices, 2-class + 3-class).
# Run from repository root:
#   bash run_all.sh

PYTHON_BIN="${PYTHON_BIN:-python}"

echo "[1/6] ML LOSO (2-class)"
"${PYTHON_BIN}" src/training.py --approach loso --device both --n-classes 2

echo "[2/6] ML LOSO (3-class)"
"${PYTHON_BIN}" src/training.py --approach loso --device both --n-classes 3

echo "[3/6] DL LOSO (all architectures, binary + 3-class)"
"${PYTHON_BIN}" src/dl_training.py --arch all --approach loso --classes both --device both

echo "[4/6] Optional raw-signal DL baseline (set RUN_RAW_BASELINE=1)"
if [[ "${RUN_RAW_BASELINE:-0}" == "1" ]]; then
  "${PYTHON_BIN}" src/raw_dl_training.py --classes both --device both
else
  echo "Skip raw baseline. Set RUN_RAW_BASELINE=1 to enable."
fi

echo "[5/6] Build tracked benchmark summaries"
"${PYTHON_BIN}" src/build_results_summary.py
"${PYTHON_BIN}" src/build_device_ablation_summary.py || true

echo "[6/6] Run tests"
"${PYTHON_BIN}" -m pytest -q

echo "Done."

#!/usr/bin/env bash
set -euo pipefail

# Full reproducible benchmark pipeline (LOSO, both devices, 2-class + 3-class).
# Run from repository root:
#   bash run_all.sh

PYTHON_BIN="${PYTHON_BIN:-python}"

echo "[1/5] ML LOSO (2-class)"
"${PYTHON_BIN}" src/training.py --approach loso --device both --n-classes 2

echo "[2/5] ML LOSO (3-class)"
"${PYTHON_BIN}" src/training.py --approach loso --device both --n-classes 3

echo "[3/5] DL LOSO (all architectures, binary + 3-class)"
"${PYTHON_BIN}" src/dl_training.py --arch all --approach loso --classes both --device both

echo "[4/5] Build tracked benchmark summaries"
"${PYTHON_BIN}" src/build_results_summary.py

echo "[5/5] Run tests"
"${PYTHON_BIN}" -m pytest -q

echo "Done."

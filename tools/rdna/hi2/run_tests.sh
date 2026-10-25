#!/usr/bin/env bash
# HI2 unit tests on one idle V620:
#   bash tools/rdna/hi2/run_tests.sh <hip_device> [pytest args...]
set -euo pipefail
DEV=${1:?usage: run_tests.sh <hip_device> [pytest args]}; shift
source "$(dirname "$0")/env.sh" "$DEV"
cd "$PYTHONPATH"
exec "$PY" -m pytest -q -p no:cacheprovider tests/kernels/quantization/test_hi2_rdna2.py "$@"

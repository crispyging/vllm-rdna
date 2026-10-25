#!/usr/bin/env bash
# Serving benchmark of rdna_hint_flex vs its dense fp16 export (V620s,
# FULL_AND_PIECEWISE cudagraphs via RECIPE=full), the four standard cells,
# through tools/rdna/port_v031/serve_validate.sh.
#   GPU=5 [TP=1] [CENSUS_DIR=dir] bash tools/rdna/rdna_hint_flex/bench_serve.sh \
#       tag=/model/path ...
# CENSUS_DIR records the GEMM shapes without a TunableOp row
# (PYTORCH_TUNABLEOP_RECORD_UNTUNED) into CENSUS_DIR/<tag>/untuned<rank>.csv.
# TUNABLEOP=0 runs on rocBLAS heuristics (no rows at all). SERVE_OVERRIDES adds
# recipe KEY=value overrides (e.g. VLLM_USE_RDNA2_FA=0 for head size 64).
set -uo pipefail
TREE=$(cd "$(dirname "$0")/../../.." && pwd)
GPU=${GPU:?GPU (HIP indices among ours) required}
TP=${TP:-1}
ROOT=${ROOT:-$HOME/w4a8_runs/hi2-support}
CELLS=${CELLS:-"1024:512:1 1024:512:8:32 16384:1024:1:4 16384:1024:8:16"}
PORT=${PORT:-18731}
for arm in "$@"; do
  tag=${arm%%=*}
  model=${arm#*=}
  echo "[$(date +%H:%M:%S)] arm $tag $model TP=$TP GPU=$GPU"
  if [[ -n ${CENSUS_DIR:-} ]]; then
    mkdir -p "$CENSUS_DIR/$tag"
    export PYTORCH_TUNABLEOP_RECORD_UNTUNED=1
    export PYTORCH_TUNABLEOP_UNTUNED_FILENAME=$CENSUS_DIR/$tag/untuned.csv
  fi
  # shellcheck disable=SC2086
  VENV=${VENV:-$HOME/Apps/vllm/venv-7.14.0_0.31.0-final} RECIPE=full MODEL=$model \
    GPUS=$GPU PORT=$PORT TAG=$tag OUT=$ROOT/serve-$tag CACHE=$ROOT/cache-$tag \
    CELLS="$CELLS" CONC_PROBE=${CONC_PROBE:-1} PREFIX_PROBE=0 \
    bash "$TREE/tools/rdna/port_v031/serve_validate.sh" TP="$TP" MAXLEN=32768 \
    SEQS=8 KV="${KV:-8000000000}" TUNABLEOP="${TUNABLEOP:-1}" ${SERVE_OVERRIDES:-}
  echo "[$(date +%H:%M:%S)] arm $tag rc=$?"
done

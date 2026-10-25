#!/usr/bin/env bash
# Serving benchmark of rdna_hint_flex vs its dense fp16 export (one V620,
# TP=1, FULL_AND_PIECEWISE cudagraphs via RECIPE=full), the four standard
# cells, through tools/rdna/port_v031/serve_validate.sh.
#   GPU=5 bash tools/rdna/rdna_hint_flex/bench_serve.sh tag=/model/path ...
set -uo pipefail
TREE=$(cd "$(dirname "$0")/../../.." && pwd)
GPU=${GPU:?GPU (one HIP index among ours) required}
ROOT=${ROOT:-$HOME/w4a8_runs/hi2-support}
CELLS=${CELLS:-"1024:512:1 1024:512:8:32 16384:1024:1:4 16384:1024:8:16"}
PORT=${PORT:-18731}
for arm in "$@"; do
  tag=${arm%%=*}
  model=${arm#*=}
  echo "[$(date +%H:%M:%S)] arm $tag $model"
  VENV=${VENV:-$HOME/Apps/vllm/venv-7.14.0_0.31.0-final} RECIPE=full MODEL=$model \
    GPUS=$GPU PORT=$PORT TAG=$tag OUT=$ROOT/serve-$tag CACHE=$ROOT/cache-$tag \
    CELLS="$CELLS" CONC_PROBE=${CONC_PROBE:-1} PREFIX_PROBE=0 \
    bash "$TREE/tools/rdna/port_v031/serve_validate.sh" TP=1 MAXLEN=32768 \
    SEQS=8 KV="${KV:-8000000000}"
  echo "[$(date +%H:%M:%S)] arm $tag rc=$?"
done

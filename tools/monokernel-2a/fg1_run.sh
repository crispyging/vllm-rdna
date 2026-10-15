#!/usr/bin/env bash
# One lock-protected serve + bench on HIP 6-9 for the FG=1 skew investigation.
#   TAG=f1-on-t ARM=on-t FG=1 bash tools/monokernel-2a/fg1_run.sh [KEY=v ...]
# Acquires ~/w4a8_runs/port-v031/GPU69.lockdir (shared with the MonoKernel
# agent), waits for rocm-smi GPU[0-3] VRAM to read 0, samples sysfs/KFD state
# during the serve (fg1_sample.sh), runs serve_2a.sh from this tree, releases
# the lock as soon as the server is down. Results: ~/w4a8_runs/fg1-skew/serve-$TAG/.
set -uo pipefail
HERE=$(cd "$(dirname "$0")" && pwd)
: "${TAG:?}" "${ARM:?}"
FG=${FG:-1}
RUN=${RUN:-$HOME/w4a8_runs/fg1-skew}
LOCK=$HOME/w4a8_runs/port-v031/GPU69.lockdir
mkdir -p "$RUN"
log() { echo "[$(date +%T)] $*"; }

until mkdir "$LOCK" 2>/dev/null; do sleep "${LOCK_POLL:-3}"; done
echo "fg1-skew: $TAG" > "$LOCK/owner"
log "lock acquired ($TAG)"
release() { rm -rf "$LOCK"; log "lock released"; }
trap release EXIT

for _ in $(seq 1 180); do
    busy=$(rocm-smi --showmemuse 2>/dev/null | awk '/GPU\[[0-3]\].*VRAM%/{s+=$NF} END{print s+0}')
    [[ $busy -eq 0 ]] && break
    sleep 10
done
log "GPUs idle (VRAM% sum $busy)"

# SAMPLE=1 opts into the sysfs sampler (off by default: two host power cuts
# on 2026-10-15 coincided with it running during model load; unproven, but
# the per-rank GPU timers are the primary evidence and do not need it).
SPID=
if [[ ${SAMPLE:-0} == 1 ]]; then
    SAMP=$RUN/sample-$TAG.csv
    setsid nohup bash "$HERE/fg1_sample.sh" "$SAMP" 5 > "$RUN/sample-$TAG.log" 2>&1 < /dev/null &
    SPID=$!
    echo "$SPID" > "$RUN/sample-$TAG.pid"
fi

export RUN GPUS=${GPUS:-6,7,8,9} PORT=${PORT:-18800} RECIPE=${RECIPE:-flashnext-mtp0}
export CACHE=${CACHE:-$RUN/cache-$ARM}
export CELLS=${CELLS:-"1024:512:1 1024:512:8"} BENCH_EXTRA=${BENCH_EXTRA:-"--temperature 0"}
export MONO_CFG=${MONO_CFG:-'"rdna_mono_ffn_max_rows":8'}
log "serve ARM=$ARM FG=$FG RECIPE=$RECIPE CELLS=$CELLS MONO_CFG=$MONO_CFG extra=$*"
ARM=$ARM TAG=$TAG bash "$HERE/serve_2a.sh" HSA_FORCE_FINE_GRAIN_PCIE="$FG" "$@"
rc=$?
[[ -n $SPID ]] && { kill -- -"$SPID" 2>/dev/null; kill "$SPID" 2>/dev/null; }
log "serve done rc=$rc"
grep -E "READY|bench-|PASS|FAIL" "$RUN/serve-$TAG/summary.txt" 2>/dev/null | grep -vE "^\s+(PASS|FAIL)  prompt" | cut -c1-220

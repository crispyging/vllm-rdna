#!/usr/bin/env bash
# The DRAM-bound QSA indexer scoring kernel (mono_qsa_mqa_rdna2 mode 2,
# mqa_dram_k) on HIP 6 under the shared HIP 6-9 lock: the correctness tests,
# then qsa_mqa_bench.py --bounded, warm and --cold, own (MTP0) and mtp (MTP2
# verify) layouts. Logs under $OUT. Run from the built tree:
#   OUT=~/w4a8_runs/mqa3 bash tools/rdna/mono/qsa_mqa_dram.sh [pytest|bench|all]
set -uo pipefail
HERE=$(cd "$(dirname "$0")" && pwd)
TREE=$(cd "$HERE/../../.." && pwd)
OUT=${OUT:-$HOME/w4a8_runs/mqa3}
WHAT=${1:-all}
VENV=${VENV:-$HOME/Apps/vllm/venv-7.14.0_0.31.0-final}
LOCKDIR=${LOCKDIR:-$HOME/w4a8_runs/port-v031/GPU69.lockdir}
mkdir -p "$OUT"
HELD=0
unlock() { [[ $HELD == 1 ]] && rm -rf "$LOCKDIR"; HELD=0; return 0; }
trap unlock EXIT
until mkdir "$LOCKDIR" 2>/dev/null; do sleep 30; done
HELD=1
echo "qsa-mqa-dram: $WHAT" > "$LOCKDIR/owner"
for _ in $(seq 1 120); do  # rocm-smi GPU[0-3] are HIP 6-9
    busy=$(rocm-smi -d 0 1 2 3 --showmemuse 2>/dev/null | awk '/VRAM%/{s+=$NF} END{print s+0}')
    [[ $busy -eq 0 ]] && break
    sleep 10
done
SITE=$VENV/lib/python3.12/site-packages
export LD_LIBRARY_PATH=$SITE/_rocm_sdk_libraries/lib:$SITE/_rocm_sdk_core/lib/host-math/lib:$SITE/_rocm_sdk_core/lib/rocm_sysdeps/lib:$SITE/_rocm_sdk_core/lib/core/lib:$SITE/torch/lib
export HIP_VISIBLE_DEVICES=6 PYTHONPATH=$TREE VLLM_ROCM_USE_AITER=0 VLLM_RDNA_FORCE_FP16=1
export HSA_ENABLE_SDMA=0 GPU_MAX_HW_QUEUES=4 ROCPROFILER_REGISTER_ENABLED=0 HF_HUB_OFFLINE=1
export TRITON_CACHE_DIR=$OUT/cache/triton VLLM_CACHE_ROOT=$OUT/cache/vllm
PY=$VENV/bin/python
cd "$TREE"

if [[ $WHAT == pytest || $WHAT == all ]]; then
    "$PY" -m pytest -p no:cacheprovider -q -rs \
        tests/models/qwen4_exp/test_amd_mono_rdna2_kernels.py -k "qsa_mqa" \
        > "$OUT/pytest.log" 2>&1
    echo "pytest exit $? ($(tail -1 "$OUT/pytest.log"))"
fi
[[ $WHAT == pytest ]] && exit 0

B=tools/rdna/mono/qsa_mqa_bench.py
# MTP2 verify: mqa_dram_k shapes (lanes per key x keys per lane, slots auto /
# per window, streamed keys) next to today's best group shapes (q in
# registers, q in LDS) and the G=1 DRAM shape
MTP=g3w8l2u1m2,g3w8l2u2m2,g3w8l4u2m2,g3w8l4u1m2,g3w4l2u1m2
MTP+=,g3w8l2u1c9m2,g3w8l2u1c36m2,g3w8l2u1c72m2,g3w8l2u1m2n,g3w8l1u1m2n
MTP+=,g3w8l16u2t0,g3w8l8u1q,g1w8l16u2t0
OWN=g1w8l2u1m2,g1w8l4u1m2,g1w8l2u1m2n,g1w8l16u2t0,g1w4l4u1t8
for temp in warm cold; do
    cold=()
    [[ $temp == cold ]] && cold=(--cold)
    "$PY" "$B" --bounded "${cold[@]}" --layouts mtp --ctx 16384,32768,65536,131072 \
        --rows 3,6,8,12,24 --variants "$MTP" > "$OUT/mtp-$temp.log" 2>&1
    echo "mtp $temp exit $? ($(tail -1 "$OUT/mtp-$temp.log"))"
    "$PY" "$B" --bounded "${cold[@]}" --layouts own --ctx 16384,65536,131072 \
        --rows 1,4,8,16,24 --variants "$OWN" > "$OUT/own-$temp.log" 2>&1
    echo "own $temp exit $? ($(tail -1 "$OUT/own-$temp.log"))"
done

#!/usr/bin/env bash
# FG=1 HW-queue A/B on HIP 6-9: Q=2 kernel-off, Q=4 kernel-off, Q=4 all-on
# (FFN + K1 + E1' + E3), MTP0 and MTP2, four cells each, production env (no
# timers). One lock per arm (fg1_run.sh), 90 s back-off after each release,
# dmesg "Runlist is getting oversubscribed" count per arm.
#   bash tools/monokernel-2a/fg1_ab.sh [arm ...]   (default: all six)
set -uo pipefail
HERE=$(cd "$(dirname "$0")" && pwd)
RUN=${RUN:-$HOME/w4a8_runs/fg1-skew}
CELLS=${CELLS:-"1024:512:1 1024:512:8 16384:1024:1 16384:1024:8"}
ALLON='"rdna_mono_ffn":true,"rdna_mono_ffn_max_rows":8,"rdna_mono_hc":true,"rdna_mono_gdn":true,"rdna_mono_qsa":true'
ARMS=${*:-"mtp0:q2:off mtp0:q4:off mtp0:q4:on mtp2:q2:off mtp2:q4:off mtp2:q4:on"}
rl() { sudo -n dmesg 2>/dev/null | grep -c "Runlist is getting oversubscribed"; }
for spec in $ARMS; do
    IFS=: read -r mtp q arm <<< "$spec"
    tag=ab-$mtp-$q-$arm
    nq=${q#q}
    before=$(rl)
    echo "== $(date +%T) $tag (GPU_MAX_HW_QUEUES=$nq) runlist-oversubscribed before=$before"
    if [[ $arm == on ]]; then
        MONO_CFG=$ALLON ARM=cfg CACHE=$RUN/cache-cfg TAG=$tag RECIPE=flashnext-$mtp CELLS="$CELLS" \
            bash "$HERE/fg1_run.sh" GPU_MAX_HW_QUEUES="$nq" > "$RUN/$tag.out" 2>&1
    else
        ARM=off CACHE=$RUN/cache-off TAG=$tag RECIPE=flashnext-$mtp CELLS="$CELLS" \
            bash "$HERE/fg1_run.sh" GPU_MAX_HW_QUEUES="$nq" > "$RUN/$tag.out" 2>&1
    fi
    after=$(rl)
    echo "== $(date +%T) $tag done; runlist-oversubscribed after=$after (delta $((after - before)))"
    grep -E "READY|bench-|PASS|FAIL" "$RUN/serve-$tag/summary.txt" 2>/dev/null \
        | grep -vE "^\s+(PASS|FAIL)  prompt" | cut -c1-200
    sleep 90  # back-off after the lock release
done

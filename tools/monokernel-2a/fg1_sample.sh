#!/usr/bin/env bash
# Per-GPU sysfs sampler for the FG=1 skew investigation (par1-cs25, HIP 6-9 =
# BDF 07/0a/12/15). Appends one CSV line per card per interval to $1; every
# $SNAP_EVERY samples it also dumps the KFD runlist / HQD state (sudo -n) to
# $1.kfd.<n>.
#   bash fg1_sample.sh OUT.csv [interval_s]
set -u
OUT=${1:?out csv}; IV=${2:-1}; SNAP_EVERY=${SNAP_EVERY:-30}
BDFS=${BDFS:-"0000:07:00.0 0000:0a:00.0 0000:12:00.0 0000:15:00.0"}
echo "ts,bdf,mclk,sclk,fclk,socclk,busy,membusy,power_uW,temp_mC,vram_used,pcie_replay" >> "$OUT"
n=0
while :; do
    ts=$(date +%s.%N)
    for b in $BDFS; do
        d=/sys/bus/pci/devices/$b
        hw=$(ls -d $d/hwmon/hwmon* 2>/dev/null | head -1)
        cur() { grep '\*' "$d/$1" 2>/dev/null | sed -E 's/^[0-9]+: ([0-9]+)Mhz.*/\1/' | head -1; }
        echo "$ts,$b,$(cur pp_dpm_mclk),$(cur pp_dpm_sclk),$(cur pp_dpm_fclk),$(cur pp_dpm_socclk),$(cat $d/gpu_busy_percent 2>/dev/null),$(cat $d/mem_busy_percent 2>/dev/null),$(cat $hw/power1_average 2>/dev/null),$(cat $hw/temp1_input 2>/dev/null),$(cat $d/mem_info_vram_used 2>/dev/null),$(cat $d/pcie_replay_count 2>/dev/null)" >> "$OUT"
    done
    # KFD debugfs snapshots (rls/hqds read HQD registers over MMIO) only with KFD=1
    if [[ ${KFD:-0} == 1 ]] && (( n % SNAP_EVERY == 0 )); then
        { echo "== $(date +%T) rls"; sudo -n cat /sys/kernel/debug/kfd/rls 2>&1; echo "== hqds"; sudo -n cat /sys/kernel/debug/kfd/hqds 2>&1; echo "== threads"; pids=$(rocm-smi --showpids --csv 2>/dev/null | sed -nE 's/^"PID([0-9]+)".*/\1/p' | sort -u | tr '\n' ' '); [[ -n $pids ]] && ps -L -o pid,tid,psr,pcpu,stat,comm -p $pids 2>&1; } > "$OUT.kfd.$n" 2>&1
    fi
    n=$((n + 1))
    sleep "$IV"
done

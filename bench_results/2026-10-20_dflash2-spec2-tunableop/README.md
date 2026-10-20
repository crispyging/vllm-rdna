# DFlash2 two-token BF16 tuning (2026-10-20)

Four Radeon PRO V620s (gfx1030), HIP 6–9 on `par1-cs25`, ROCm
7.14.60850, PyTorch 2.12.0+rocm7.14.0, rocBLAS 5.5.0.cd957402.
The required `librocblas.so.5` SHA256 is
`f30bb442e9b5a3c479f6aa60c4d0d0740f67db2957fc0a5c416253e1bf320262`.
The AWQ target stays FP16; DFlash2 stays BF16 to avoid FP16 residual overflow.

## Shapes and isolated measurements

[shapes.csv](shapes.csv) contains 36 regular BF16 TN keys missing from the
1824-row canonical profile. Its SHA256 is
`abec4e024e074cbc9be73ec247de06af4238efe2ec2d9a24b2ac17ef45d551c6`.
The two-token FULL graph query widths are M=3/6/12/24, and candidate/head
widths are M=2/4/8/16. Variable auxiliary-FC and mixed-prefill context widths
are outside this set; four strided-batched BF16 keys still use the heuristic.

[measurements.csv](measurements.csv) records all 144 rank/shape measurements
and selected solver IDs. Each GPU tuned its own rows with numerical checking
(rtol=0.01, atol=0.01), at least 10 iterations and a 25 ms solver budget.
Canonical lookup, heuristic, and scratch lookup were measured in separate
processes with five warmups and 30 repetitions (GPU-event median).
`gain_vs_best = min(canonical_ms, heuristic_ms) / candidate_ms`.
All 144 gains exceed 3%: minimum 2.354x, median 3.417x, maximum 5.347x.
The BF16 head at M=2/4/8/16 falls from about 8.8–8.9 ms to 1.65–1.87 ms.
These are isolated GEMM gains, not serving throughput multipliers.

Run the existing tuner once per rank, with the exact runtime/library above:

```bash
# VENV, WORK, RANK and HIP_VISIBLE_DEVICES are deployment-specific.
"$VENV/bin/python" tools/rdna2_028/tune_prod_shapes.py \
  --shape-source bench_results/2026-10-20_dflash2-spec2-tunableop/shapes.csv \
  --dtype BFloat16 --no-small-m --mode tune --device 0 \
  --rank "$RANK" --scratch "$WORK/rank$RANK/scratch" \
  --iterations 10 --max-ms 25 --out "$WORK/rank$RANK/tune.json"
```

Measure the same shape source with `--mode measure --warmup 5 --reps 30`
and either `--rows <canonical-or-scratch-rank-file>` or `--heuristic`.
Exclude the tuner's extra warmup key when merging. Existing canonical rows
must remain unchanged; each candidate appends only these 36 BF16 keys.
The candidate contains 1860 rows/rank (1766 FP16 + 94 BF16), and a replay
confirmed 36/36 candidate lookup hits on each GPU.

Raw tuning and serving artifacts remain under
`~/w4a8_runs/port-v031/` on `par1-cs25` (`dflash2-spec2-bf16-tune36`,
`dflash2-spec2-tuned-registry`, and `serve-tune36-*`).

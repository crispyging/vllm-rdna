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

## Shipped profile and serving status

The 36 rows per rank are now appended to the existing
[`tunableop/rocm7.14-rocblas5.5/`](../../tunableop/rocm7.14-rocblas5.5/)
profile. Every previous row and validator is preserved; the registry reports
1860 rows/rank and retains its separate ROCm 10 profile. The new rows are
rank-specific and require the exact rocBLAS library hash above.

## Same-boot serving A/B

The complete comparison used the preserved 1824-row profile from
`ccb38749e2` and the expanded 1860-row profile from `c667dc80b8`.
Both arms ran on boot `d100f920-0655-4319-bddd-a430b7d2d282` with identical
source, target snapshot `0939125b929543a783ce700c90e36dd1a575c00c`,
FP16 AWQ target, BF16 DFlash2 with two draft tokens, BF16 PLE, FA-RDNA2,
MonoKernel, TP4/EP4 on HIP 6–9, FULL_AND_PIECEWISE captures [3,6,12,24],
max batched tokens 2048, max sequences 8 and KV bytes 4026531840.
Profiling and online tuning were off. Each cell reset prefix caching and used
temperature 0, seed 12345 and forced output length. Each arm completed 80
requests: eight per c=1 cell and 32 per c=8 cell.

The initial attempt was interrupted by a host reboot of undetermined cause.
All cells below were rerun after reboot. [serving_ab.csv](serving_ab.csv)
contains the exact metrics, acceptance and artifact paths;
[serving_audit.json](serving_audit.json) contains the independent compact audit.

| Profile | Workload | Prefill tok/s/req (agg estimate) | Decode tok/s/req (agg output) | Mean TTFT | ITL median / mean |
| --- | --- | ---: | ---: | ---: | ---: |
| old1824 | 1,024 → 512, c=1 | 2036.2 (1950.1) | 42.62 (40.99) | 0.50 s | 49.6 / 49.7 ms |
| new1860 | 1,024 → 512, c=1 | 2160.9 (2094.7) | 68.60 (64.62) | 0.47 s | 31.2 / 31.3 ms |
| old1824 | 1,024 → 512, c=8 | 929.6 (2100.6) | 23.83 (173.83) | 1.10 s | 81.8 / 93.0 ms |
| new1860 | 1,024 → 512, c=8 | 963.2 (2154.2) | 29.67 (208.78) | 1.06 s | 63.4 / 74.0 ms |
| old1824 | 16,384 → 1,024, c=1 | 1991.8 (1975.4) | 46.09 (33.66) | 8.23 s | 51.2 / 51.2 ms |
| new1860 | 16,384 → 1,024, c=1 | 2051.1 (2038.6) | 65.44 (43.35) | 7.99 s | 32.7 / 32.7 ms |
| old1824 | 16,384 → 1,024, c=8 | 897.2 (1957.8) | 12.01 (77.28) | 18.26 s | 85.2 / 177.0 ms |
| new1860 | 16,384 → 1,024, c=8 | 978.8 (2020.4) | 13.31 (85.60) | 16.74 s | 67.3 / 163.5 ms |

Aggregate output increased **57.7%, 20.1%, 28.8% and 10.8%** for short c=1,
short c=8, long c=1 and long c=8. Median ITL fell **37.2%, 22.6%, 36.1%
and 21.0%**, respectively. These are one paired run per cell; the modest
long c=8 throughput gain has no repeat-derived precision estimate.
Speculative acceptance varies: long c=1 acceptance length fell from 2.359
to 2.140 while throughput improved, and short-cell acceptance changed by
about 1%. The CSV includes draft verifications per full-run second to help
interpret throughput alongside acceptance; it includes prefill and queueing.

Prefill per request is mean input tokens / mean TTFT. The aggregate prefill
estimate is min(concurrency, completed) × mean input / p99 TTFT; it is a
wave proxy including latency effects. Decode per request is 1000 / mean
TPOT in ms. Aggregate output is measured output tokens / total run duration.

### Correctness and provenance

Both arms passed greedy, perplexity (4.81 old / 4.87 new), prefix-cache,
normal-EOS concurrency (0 hard failures / 48 checks each), and after-load
probes. All eight prefix resets passed. Both serve logs and amdgpu dmesg
deltas contained zero fault markers. API arguments matched except port;
rank files matched the intended snapshots with unchanged original rows.

Outputs are not bit-identical. Same-boot old→new has 14/48 exact sequences;
all first changed tokens were in the old top five, with maximum margin
0.297 nats. Seven stopping differences followed earlier token divergences;
none was a matching-prefix premature stop. The unchanged old profile's
pre/post-reboot control itself had 18/48 exact sequences and six stopping
differences. The full saved-token/logprob audit remains with the raw artifacts.
Worker filename/provenance checks confirm the selected files; process-title
rewriting limits what `/proc` exposes, as documented in the fork's port notes.

The recipe's default depth remains three while the fresh short-context
control and the separate native MTP2 comparison are completed.

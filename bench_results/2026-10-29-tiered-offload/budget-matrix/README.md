# Expert/KV byte-budget comparison

Source: `5dccb11e7a`; Flash-Next AWQ INT4, HIP 6–7, TP2/EP2, MTP2,
FULL_AND_PIECEWISE, native expert copy and K2 gate 8. Live KV paging, route
recording and transfer timing were disabled. CPU KV budget: 16 GiB total.
Both GPU and CPU prefix caches reset per cell. Expert history carries between
cells. Fixed workload seeds/order match across launches. Random performance
outputs force EOS continuation and are not semantic quality evidence.

`baseline` and `budget88` are first passes, including startup shape effects.
`*-repeat` are warmed repeats. This is one sequential pair plus its repeats;
MTP/routing trajectories and expert history can differ. Use the observations
as workload-specific evidence, not an isolated causal or universal speedup.

## Controlled memory

Per rank, uniform96 uses 11,767,037,952 expert/map GPU bytes plus
4,026,531,840 configured GPU KV bytes. The 88% plan uses 10,337,021,952
expert/map GPU bytes plus 5,456,547,840 configured GPU KV bytes.
Both equal **15,793,569,792 bytes**. Other model/graph/runtime allocations
are outside this controlled sum. Page rounding leaves small unused KV slack.
GPU physical pages increase 139 → 188; CPU chunks stay 296/rank.
Expert RAM stays 31,378,685,952 bytes/rank. The plan uses 48–160 expert slots
per layer; uniform96 uses 96 everywhere.

The owned HIP 6–7 workers map to SMI 0–1 / PCI 0000:07:00.0 and
0000:0a:00.0 on this boot. The sampler reads only these PCI sysfs paths.
Sampled device-wide VRAM peaks are 30,750,601,216 and 30,717,046,784 bytes
out of 32,195,477,504 each, leaving roughly 1.45 GB; this includes allocator
reservations and contexts and does not prove all remaining bytes are safe KV
headroom. Sampling covers part of the candidate warmed pass, not every stage.

## Warmed results

| Cell | Output tok/s, uniform → 88% | Mean TTFT s | Mean ITL ms |
| --- | --- | --- | --- |
| 1k/512 c1 | 44.41 → 30.27 | 1.16 → 1.23 | 58.9 → 73.3 |
| 1k/512 c8 | 69.17 → 62.87 | 5.51 → 5.70 | 190.8 → 216.4 |
| 16k/1024 c1 | 23.45 → 19.03 | 16.17 → 21.45 | 69.3 → 79.5 |
| 16k/1024 c8 | 31.20 → 31.13 | 96.18 → 79.40 | 307.6 → 400.9 |

Long c8 admits eight requests together in the candidate versus six running
and two waiting in the baseline. Candidate mean/p99 TTFT improve 17.4%/34.3%,
and sampled waiting request-seconds fall about 24.8%. Throughput is unchanged
(−0.2%), mean ITL worsens 30.3%, and expert-copy bytes rise 22.6%.
All cells completed with zero recompute preemptions.

Keep uniform96 as the general default. The 88% plan is an optional admission/TTFT
tradeoff, not a throughput improvement. Initial-pass comparisons reach the same
conclusion: long c8 throughput 31.67 → 31.36, TTFT 94.96 → 81.24 s, ITL
298.9 → 421.9 ms. Candidate quality after both matrices passed PPL 4.83 and
48 concurrency checks with zero hard failures.

## Bottlenecks and next route

Reduced-capacity layers add more transfers than enlarged layers save. In the
repeated short c8 cell, excess misses are largest in layers 31 (48 slots),
27, 33 and 25 (64 slots). In repeated long c8, layers 11, 10, 27 and 39
lead the excess traffic. These are observed counter differences, not isolated
layer latency measurements.

The repeated short c1 outlier also changes MTP mean acceptance from 2.909 to
2.388 and increases misses in some enlarged layers. Its entire slowdown cannot
be assigned to capacity alone. The 61-call decode-only planning trace did not
exercise frequency decay and is insufficient evidence for permanent allocation.
Retain the expert budget for the next nonuniform-plan experiment; gather longer
representative c1/c8 prefill and decode traces first. Evaluate live request KV
spilling separately: these no-preemption results do not establish its speedup.

`summary.json` contains exact metrics, byte deltas, MTP acceptance and sampled
admission totals; `summarize-tiered-budget.py.txt` reproduces it. Each arm retains
raw benchmark results, per-layer before/after receipts and one-second metrics.
Server/client logs are gzip-compressed; `log-scan.json` records server hashes
and numeric/fault markers. The baseline has one output-handler EngineDeadError
traceback during deliberate shutdown after all cells and the idle layout RPC;
no serving numeric/GPU-fault marker was found. Kernel-ring access was unavailable.

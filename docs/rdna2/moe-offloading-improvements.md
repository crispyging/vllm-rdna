# RDNA2 MoE offloading improvement backlog

This draft tracks follow-up work after #146. Runtime behavior is unchanged.
The qualified baseline and receipts live in `moe-offloading-target.md` and
`bench_results/2026-10-28-moe-offload/`.

## Baseline and constraints

TP2/EP2 uses 96 GPU expert slots per layer, LFU/LRU with decay 128,
segmented FULL_AND_PIECEWISE graphs, MTP2, prefix caching, and an optional
16 GiB aggregate CPU KV pool. Cold expert weights remain in pinned RAM.
Active experts use the same GPU slots; there is no separate active-weight pool.

Per GPU, model-load allocation is 21.39 GiB, including 10.959 GiB of expert
slots. The explicit GPU KV pool is 3.75 GiB. Capture-era allocation grows by
2.12 GiB; this is not a pure graph-memory census or a peak-headroom guarantee.
Across TP2, host expert banks occupy 58.447 GiB, pinned INT4 PLE tables about
29.802 GiB, plus the configured 16 GiB CPU KV pool. Budgets are fixed at startup.

Each GPU owns 256 whole experts out of 512. Dense projections are tensor
parallel; routed experts are expert parallel, with partial outputs reduced.
The default K2 row gate is 8, so MTP2 concurrency 8 verification at 24 rows
uses the cached modular MoE fallback. Native K2 support alone does not enable it.

Measured 16k/1k aggregate output throughput is 19.767 tok/s at c=1 and
26.718 tok/s at c=8, including prefill. Mean TTFT rises from 17.307 s to
106.260 s, and mean TPOT from 33.721 ms to 148.441 ms. These observations
identify a scaling limitation, not its causal attribution. Hosted CI was
skipped; focused local and dedicated GPU validation passed for #146.

## Ordered work

- [ ] Capture CPU/HIP timelines and per-rank expert hits, misses, transfer bytes,
  readback waits, collective time, and peak memory. Distinguish queue time from
  prefill execution; do not infer bottleneck percentages from wall time alone.
- [ ] A/B the 24-row K2 gate with the current 8-row default. Coordinate with
  #110 rather than duplicate its wide-K2 implementation.
- [ ] Batch the six payload copy submissions per missed expert and measure
  submission overhead versus transfer time, preserving pinned-buffer lifetime.
- [ ] Reduce blocking route readback and host planning overhead without changing
  routing, eviction protection, overflow behavior, or graph replay correctness.
- [ ] Evaluate route-informed prefetch and overlapping transfers with computation.
  Bound wasted bytes and protect demand transfers from speculative traffic.
- [ ] Measure per-layer demand and rebalance expert slots within a fixed budget.
  One additional slot across all 48 layers costs 116.895 MiB per GPU; adding
  32 slots costs about 3.653 GiB per GPU before other transient allocations.
- [ ] Optimize deterministic dense accumulation while retaining established
  parity. Isolated 4096-row QKVZ cost rose from 8.844 to 14.173 ms; a whole-model
  A/B is required before claiming an end-to-end improvement.
- [ ] Measure collective crossover and rank imbalance, coordinating with #111
  and #129. Current custom all-reduce eligibility ends at 64 KiB; a 24-row
  FP16 hidden-state tensor of width 2560 exceeds this bound.
- [ ] Evaluate coordinated expert/KV transfer priorities and memory budgets.
  Current native KV offload restores eligible prefix/hybrid state; it is not
  arbitrary active-context paging. Keep raw QSA-ring restore restrictions.
- [ ] Extend correctness and regression coverage to affected platform paths and
  reproducible mixed workloads, including prefix reuse and cache churn.

## Acceptance gates

Every runtime optimization needs an isolated commit, eager and graph parity,
TP2 cache-churn coverage, MTP2 and prefix-cache checks, and normal-EOS semantic
probes. Keep forced-length throughput probes separate from semantic evaluation.
Report prefill tok/s, decode tok/s per request, TTFT, ITL and TPOT for 16k/1k
and 1k/512 at c=1 and c=8, plus peak GPU/host memory and expert transfer counters.
Use identical seeds, launch configuration and warmed tuning rows for A/B runs.

Only physical GPUs 6, 7 and 8 are allocated. Full-resident TP2 needs about
39.655 GiB per GPU before KV and graphs, so it is not a feasible control on
these V620 devices. Use a feasible matched control and state its limits.

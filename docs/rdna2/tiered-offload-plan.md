# Tiered offloading implementation and qualification

Branch: `codex/rdna2-tiered-offload`, based on merged PR #147.
GPU allocation: HIP 6–7 only. Local source is authoritative.

The objective is request throughput and latency under a fixed memory budget,
including long-context concurrency eight. Cache hit rate alone is not acceptance.
The existing CPU KV connector supports prefix reuse; live request paging is a
separate requirement and must not be advertised before its resume gate passes.

## Execution order

1. **Per-layer measurements.** Keep cheap demand/miss/copy-byte counters; expose
   per-layer slot and payload/map byte counts. Opt-in timing must distinguish CPU
   planning, pinned-map source waits, GPU copy dependency waits, transfer duration,
   and exposed compute-stream waits. Bound diagnostic event retention. Gather route
   traces from existing CPU routing results, without an extra device readback.
2. **Static byte-budgeted slot plans.** Replay traces with the production eviction
   policy. Select per-rank/per-layer capacities, accounting for exact expert bytes
   and fixed map bytes. Reject incomplete plans. Allocate once at model load so
   graphs retain stable addresses. The replay heuristic is not a global optimum,
   and an empty-cache trace does not reproduce the complete serving history.
3. **Expert/KV budget comparison.** Hold total GPU use and CPU allocations fixed.
   Sweep the expert budget, leaving the saved GPU bytes for KV allocation. Compare
   all four 1k/512 and 16k/1024, c1/c8 cells. Record actual KV capacity/occupancy,
   running/waiting requests, completion throughput, TTFT, ITL, transfer bytes and
   MTP acceptance. Reset both external CPU and GPU prefix caches for cold runs;
   also test explicit warm-prefix reuse. Reject regressions in admission/latency.
4. **Known-route overlap.** Start with resident-hit computation while known misses
   copy into disjoint, reserved slots of the existing pool. Keep demand priority,
   one final accumulation cast, stable graph buffers and map publication after
   completion. Cancellation invalidates overwritten victims. Benchmark the extra
   pass/map costs; keep disabled unless net benefit is established. Exact next
   layer routing depends on current output; prediction is a later optional hint.
5. **Live KV RAM spill/resume.** Reuse native pinned CPU pages and transfer engine,
   adding protected request snapshots and a two-phase scheduler transition. Store
   complete hybrid state, including raw QSA rings and unaligned GDN/conv/PLE tails,
   before freeing any GPU block. Resume without prefix-boundary truncation or
   recomputation. See `tiered-kv-spill-design.md`. LRU may select inactive requests;
   it must never evict the sole live snapshot. CPU budget exhaustion must fall
   back safely to the existing scheduler behavior.
6. **Prediction, only after measurement.** Bound speculative bytes and pending
   work; record useful, unused and cancelled bytes and demand delays. Never change
   router results. An improved hit percentage does not establish a speedup.

## Release gates

- Focused CPU contracts and applicable hooks pass for each implementation slice.
- GPU 7 cache-policy/storage/copy tests pass; TP2 uses only GPUs 6–7.
- Full-and-piecewise graphs and eager: routing/churn/overflow parity, MTP and
  accepted-index safety, greedy/PPL/semantic gates, no numeric or GPU faults.
- Hybrid KV: resume after nonaligned token counts; all state tensors and logits
  agree with an uninterrupted run, including TP2, concurrent cancellation,
  snapshot eviction pressure, reset, partial transfer failure and stream reuse.
- Performance receipts include exact source/runtime/configuration, cache budgets,
  cold/warm state, route/miss/byte counters and model quality/MTP acceptance.
- Keep experimental controls opt-in. Commit validated slices separately; do not
  merge live paging on the strength of prefix-cache tests.

## Progress

- [x] Audit native KV connector and exact-route overlap feasibility.
- [x] Implement deterministic per-rank byte-budget planner (26 CPU tests).
- [x] Add per-layer diagnostic telemetry and lightweight route traces.
- [x] Wire and validate startup per-layer slot selection.
- [x] Collect GPU 6–7 diagnostics and select same-budget and 88% expert-budget candidates.
- [x] Implement known-route overlap; GPU packing/parity and cleanup tests pass.
- [x] Qualify graph-mode overlap quality and run ABBA throughput comparison;
  no consistent gain, so overlap remains disabled by default. Eager coverage remains.
- [x] Reuse native CPU pool for retained, allocator-backed physical page transfers.
- [x] Implement typed hybrid snapshots, native CPU leases and opt-in V2 live paging;
  792 regression tests pass on GPU 7.
- [x] Qualify the supported hybrid live KV configuration: graph PPL/concurrency,
  repeated nonaligned 25k–29k-token resume, graph disconnect cleanup, eager
  PPL/56 checks, eager prefix reuse, and combined eager overlap/48 checks.
- [x] Run matched first/warmed four-cell matrices and retain uniform96 as the
  general default. The 88% plan improves long c8 TTFT but not throughput and
  worsens ITL and shorter cells; see the budget-matrix receipts.
- [x] Finish eager live-paging and combined overlap qualification.
- [ ] Measure normal-threshold live paging with the uniform expert budget;
  forced-zero correctness runs do not establish its serving benefit.
- [ ] Before promoting a nonuniform default, collect longer representative
  prefill/decode c1/c8 traces that exercise frequency decay and retrain the plan.

# Tiered offload qualification receipts — 2026-10-29

Target: Flash-Next AWQ INT4, gfx1030, TP=2/EP=2, MTP2,
FULL_AND_PIECEWISE graphs. Only HIP GPUs 6–7 were used for serving;
physical-page and regression tests used GPU 7.

## Verified so far

- `restore-fence-regression-tests.txt`: latest 801-test suite passes after the
  admission-order/readiness fixes and restore-destination prefix-reader fence.
- `live-pre-serve-tests.txt`: 792 regression tests passed, including typed hybrid
  snapshots, CPU leases, connector transactions, cancellation, prefix caching and
  allocator-backed GPU → RAM → GPU physical page copies.
- `graphs-initial`: uniform 96-slot quality qualification; greedy, PPL 4.83,
  prefix reuse and 48 normal-EOS concurrency checks passed.
- `overlap-quality`: known-demand overlap PPL 4.80 and concurrency quality passed.
- `overlap-abba/results.json`: off/on/on/off aggregate output throughput was
  51.25 / 52.93 / 47.85 / 51.83 tokens/s. Expert history was retained and MTP
  trajectories differed. This does not establish a consistent overlap benefit;
  overlap stays disabled by default.

## Diagnostic and modeled results

`profile` contains instrumented measurements, not clean throughput benchmarks.
Transfer intervals include enqueue/scheduling gaps, and a large rank-0 layer-16
wait outlier must not be interpreted as isolated DMA service time.
Optional KV transfer counters absent from the server are recorded as unavailable.
The kernel ring could not be read; log scans do not establish hardware-ring health.
`profile/server-info.json` is sanitized configuration only.

`policy-replay.json` replays 61 decode calls per rank/layer from empty caches.
Frequency decay at 128 calls was not exercised. Prefill and previous residency
are excluded. Slot plans are candidates, not measured speedups or global optima.

- `slot-plan.json`: same expert byte budget, 48–160 slots per layer; about 7.2%
  fewer modeled decode copy bytes than uniform 96.
- `slot-plan-0.88.json`: releases 1.33180618 GiB/GPU for about 2.6% more modeled
  decode copy bytes. The launch recipe explicitly caps GPU KV at 3.75 GiB;
  using those freed bytes requires `KV=5456547840` (about 5.08 GiB).

The initial live-paging model run passed PPL 4.87, 48 normal concurrency checks
and 56 checks with eight long contexts (zero hard failures). Logs show repeated
spill/resume at identical nonaligned frontiers between 25.9k and 29k tokens.
There were 31 paired spill/resumes, all at equal token frontiers; the only
unpaired spill belongs to the intentionally disconnected request. Eight surviving
requests finished after that disconnect, post-abort PPL passed, and native cache
reset succeeded. The 12k-word prefix probe measured 11.67 s cold versus 2.30 s
warm TTFT. The 3k-word probe showed no benefit (3.25 s versus 3.16 s) and is
retained as a failed timing gate. See `live-qualification/`.
The matched expert/KV matrix is complete in `budget-matrix/`: the warmed 88%
plan changes long c8 throughput 31.20 → 31.13 output tok/s, mean TTFT
96.18 → 79.40 s, and mean ITL 307.6 → 400.9 ms. Shorter cells regress;
uniform96 remains the general default. `eager-live-qualification/` adds PPL 4.80,
56 checks with zero hard failures and 28 matched-frontier spill/resumes. Combined
eager overlap/live paging passed PPL 4.86 and 48 checks; all 802,067,884,800
known-demand prefetch bytes were adopted, with zero unused bytes or failures.
These are correctness receipts, not an overlap performance claim.
Live paging uses a forced zero watermark during correctness qualification only;
that setting is not a recommended serving default.

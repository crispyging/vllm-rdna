# Live Flash-Next KV spill into the native CPU pool

Status: design only. Prefix-cache offloading exists; whole-request live paging
is not implemented or qualified. This design does not extend deferred PR #23.
Only physical HIP devices 6 and 7 are authorized for subsequent GPU validation.
This audit performed no remote or GPU work.

## Boundary and smallest useful slice

Start with an explicit, same-engine suspend/resume operation for one live,
text-only Flash-Next request at a fully processed target/drafter step boundary:
TP2/EP2, MTP2, fixed model/layout, native CPU backend, blocks_per_chunk=1,
worker-private storage, and a verified single packed GPU manager-page backing.
Keep existing prefix offloading and its defaults unchanged when paging is off.

The request retains its CPU token history, sampler/grammar state and worker
request object. Save its allocated GPU pages, release its references only after
both ranks finish storing, then allocate replacement GPU pages and restore
before allowing another model step. This is whole-request residency paging:
the request must fit the GPU pool while running. It is not per-layer demand
paging, does not run arbitrary active contexts directly from RAM, and does not
make a context larger than the GPU pool executable.

Do not start with automatic victim selection, cross-engine migration, CPU-tier
eviction of suspended requests, vision inputs, TP changes, DCP/PCP/PP, canonical
layout conversion, mmap deduplication, or partially in-flight MTP verification.
An explicit manual operation establishes correctness before scheduling policy.

## What existing code does

- `vllm/v1/core/sched/scheduler.py::_preempt_request` immediately frees/detaches
  GPU blocks, resets num_computed_tokens to zero, clears speculative IDs and
  queues PREEMPTED. Admission then finds cached prefixes and recomputes any
  missing suffix. This is not a save-before-release live paging transition.
- `vllm/distributed/kv_transfer/kv_connector/v1/offloading/scheduler.py` stores
  content-hashed prefix chunks and materialized recurrent checkpoints. Its
  lookup is bounded/aligned; it is not an exact arbitrary live frontier.
- `vllm/distributed/kv_transfer/kv_connector/v1/offloading/config.py` selects
  prefix-cacheable groups. `CircularBufferSpec.prefix_cacheable` is false, so
  QSA raw rings are not represented as ordinary offload keys.
- `vllm/v1/kv_offload/cpu/gpu_worker.py` already has pinned host storage, batched
  copies, descriptor lifetime retention, compute-stream dependencies, pooled
  events and completion polling. Real loads block admission until completed;
  `wait_for_layer_load` is intentionally unnecessary for this async contract.
- `offloading/common.py::OffloadingWorkerMetadata` and scheduler job pending
  counts aggregate completion across workers. Existing source-reuse fences and
  jobs_to_flush remain necessary alongside a new paging transaction.
- LRU/ARC in `vllm/v1/kv_offload/cpu/policies/` choose evictable CPU cache chunks.
  They do not choose GPU request victims or guarantee a suspended request keeps
  its only copy. `CPUOffloadingManager.complete_store` normally makes a ready
  chunk evictable; transfer pinning lasts only through prepare/complete_load.

## Save the complete state, not just a token prefix

The snapshot manifest must inventory every allocated group, including sparse
logical block-table positions, null sentinels and any repeated physical IDs:

- QSA main K/V pages and compressed index keys, including the exact partial
  final page and quantization metadata stored in those pages.
- QSA raw-key circular buffer, including cached MRoPE positions if enabled.
  The initial text-only gate excludes MRoPE/vision, but must reject rather than
  silently omit an unrecognized ring format. See
  `vllm/models/qwen4_exp/common/qsa_cache.py::QSAKeyStateCache` and
  `QSACompressedKeyCache`, and `amd/indexer_qsa.py`.
- GDN convolution history and recurrent state, in every non-null live Mamba
  slot needed by align mode, acceptance and the next MTP step. GDN is TP-sharded.
- PLE short/dilated convolution state, including speculative history. This
  state is replicated across TP, but both worker-private copies must restore.
  See `vllm/v1/kv_cache_interface.py::MambaSpec`, Mamba state calculators and
  `vllm/models/qwen4_exp/amd/ple_layer.py`.
- Drafter KV and recurrent state, if its groups are separately allocated.
  Prefix offloading's volatile-tail exclusion must not be reused for a live
  snapshot. Save the exact live bytes or explicitly normalize and regenerate
  pending drafts under a separately validated resume contract.

PLE embedding tables are immutable model storage, not request state. N-gram
context can be reconstructed from retained token history and the exact frontier
by `gpu_model_runner.py::_prepare_ngram_context`; it must not be reconstructed
from prompt length alone. PLE IPC output/doorbell buffers are transient: drain
existing operations before suspension and rebuild normal per-step inputs.

Also retain scheduler and manager state: accepted/computed frontier, token and
output counters, pending input/speculative IDs, request RNG/grammar state, and
per-group bookkeeping. `MambaManager` keeps last_state_block_idx, retired-block
counts, allocation flags and checkpoint/partial-tail metadata beyond its block
table. `pop_blocks_for_free` removes these. A raw table transplant is insufficient.
Do not serialize arbitrary manager __dict__ objects: add typed detach/reattach
hooks and explicitly validate each supported manager/spec.

## Reuse packed physical pages when actually available

`offloading/worker.py::register_kv_caches` already detects interleaved manager
pages when a layer's stride exceeds its page size. It registers one full-page
CanonicalKVCacheTensor and identical whole-page refs for selected groups.

For that layout only, deduplicate all non-null physical block IDs in the
request's manifest and copy each complete manager page once. This preserves
raw rings and partial hybrid state without making them prefix-hashable or
inventing a tensor serializer. Preserve the group's sparse logical tables in
the manifest and replace old physical IDs through a restore map.

Registration must prove a single packed backing, bounds/stride/extent validity,
full-page refs with no fragment/canonical conversion, supported group specs,
and matching TP/layout/dtype/model identities. Actual Flash-Next worker layout
must be inspected and tested; the presence of this code path alone is not proof
that a given serve uses it. Reject unsupported layouts before reserving or
freeing anything. Do not change generic get_offloading_group_ids to include
rings: that would break positional prefix planning.

An internal physical-page transfer spec can translate to existing
GPULoadStoreSpec with the complete page-ID list assigned to one verified
full-page ref slot and zero counts in other slots. CPU chunk allocation and
copy execution stay in the existing native pool. Explicitly validate this
translation; do not rely on an incidental first-group ordering.

## Transaction and ownership

State transitions:

1. RUNNING -> QUIESCING: stop scheduling this request; retain GPU blocks.
2. Drain its in-flight scheduler output and acceptance corrections. Require
   num_in_flight_tokens=0, processed scheduling fences satisfied, and a worker
   event after target, drafter and state postprocessing. Wait on that event
   from the store stream. Host output arrival alone is not a GPU fence.
3. QUIESCING -> STORING: freeze the accepted frontier and manifest. Resolve or
   flush existing prefix stores that reference these blocks. Atomically reserve
   all snapshot CPU chunks across both ranks; admission filtering must not
   silently skip snapshot pages. If capacity is unavailable, retain GPU state
   and refuse/defer suspension without changing request semantics.
4. STORING -> HOST_RESIDENT: only after both ranks acknowledge every page,
   publish the immutable snapshot and detach/release request GPU references.
   Shared prefix pages may remain GPU-resident for other requests; releasing
   this request does not guarantee all those physical pages become free.
5. HOST_RESIDENT -> LOADING: reserve replacement GPU pages atomically, construct
   sparse per-group tables and restore into them. Preserve nulls/aliasing.
   Fence initialization/zeroing before H2D; do not expose tables to computation.
6. LOADING -> RUNNING: after both ranks finish restoration and typed manager
   metadata is reattached, restore the exact computed frontier and request
   state, rebuild batch metadata, and permit the next decode operation. Bypass
   prefix lookup and do not reset the frontier to zero. Preserve or explicitly
   invalidate request-indexed scratch/top-k reuse; batch-slot reuse is not state.

Use request-private keys domain-separated from prefix content keys, containing
engine generation, request generation, snapshot ID and physical-page ordinal.
A snapshot holds a persistent CPU residency lease until successful resume or
cancellation. Transient load refs are additional to that lease. Store completion
must transfer directly into the lease without an evictable interval. LRU/ARC may
evict best-effort prefix entries to make room; neither may discard a suspended
request's only state. Refuse capacity pressure rather than silently recompute.
Do not count a snapshot insertion/pin as a new frequency hit on every poll.

The scheduler needs explicit blocked/live-paging states or an equivalent paging
state plus queues. These must remain unfinished requests and count against
appropriate admission limits. If RequestStatus gains members, place them before
PREEMPTED or update is_finished: the existing enum treats later values as finished.
Existing WAITING_FOR_REMOTE_KVS can represent the loading phase, but cannot
represent durable HOST_RESIDENT state or outbound storage by itself.

## Proposed APIs

Names are a proposal, not existing interfaces:

- Scheduler: request_suspend(req_id) -> operation_id; request_resume(req_id).
  Suspension becomes effective only at a processed step boundary.
- KVCacheManager/coordinator: export_live_state(req_id) -> typed manager/table
  manifest; detach_live_state(req_id, manifest); allocate_live_restore(manifest)
  -> replacement tables; commit_live_restore(req_id, manifest, tables).
- Native connector: prepare_live_snapshot(req_id, frontier, manifest) -> handle
  or backpressure; prepare_live_restore(handle, replacement_tables) -> job_ids;
  release_live_snapshot(handle). Metadata carries a distinct job kind and
  generation so completions cannot enter prefix-store bookkeeping accidentally.
- CPU manager: atomic snapshot reservation plus acquire/release persistent lease.
  Reserve the complete page set before launch. Keep prefix admission/policy
  behavior unchanged for existing calls.

First implementation: explicit suspend/resume of one quiescent request,
worker-private packed pages, whole-request capacity reservation and exact
restore. No background automatic swapping policy until this passes qualification.

## Failure, cancellation and reset

- Before committed storage, failure retains the authoritative GPU state. Drain
  any started copies before releasing CPU reservations; restore RUNNING or fail
  the operation. Never claim HOST_RESIDENT after partial rank completion.
- After GPU release, failed restore keeps the CPU lease; discard failed GPU
  allocations only after their H2D jobs stop. Retry or explicitly fail the
  request. Do not silently fall back to prefix recomputation.
- Cancellation in any phase is idempotent. Retain each source/destination until
  its last GPU event and all-rank completion are accounted for, then free both
  resources. Reject late completions from an older request/snapshot generation.
- Existing reset_cache clears all policies regardless of refs and drops job
  state after scheduling flushes. It cannot invalidate live snapshots. Refuse
  connector reset while durable snapshots exist in the first slice, unless an
  explicit abort-all operation drains and releases every snapshot first.
- Process restart recovery is excluded. GPU loss fails affected requests; a
  same-process CPU pool is not durable storage. Bound stuck operations and
  report failure without unsafe reclamation of DMA-visible memory.

## Verification and acceptance gates

Extend nearby tests, rather than replacing the existing prefix suites:

- CPU manager LRU/ARC tests: atomic reservation, lease/pin transitions, full
  capacity backpressure, no eviction window, cancellation and reset rejection.
- Scheduler tests: two-phase free, no extra model step after quiesce, late and
  duplicate rank completion, async stale output, blocked-state counts, sparse
  table remapping and exact non-aligned frontier preservation.
- Native worker test: full packed-page byte parity including QSA raw ring,
  compressed KV, GDN and PLE/MTP slots; different destination IDs; dirty/NaN
  destination memory; writes queued before snapshot and zeroing before restore.
- TP2/MTP2 real model: pause mid-decode at compression/block edges and arbitrary
  acceptance counts, overwrite/reuse released GPU pages with a second request,
  then resume under eager and FULL_AND_PIECEWISE graphs. Compare continuation
  tokens/logprobs against a no-suspend control, repeated cycles and prefix hits.
  Prove no already-computed prefix rows executed again, and report saved bytes,
  reclaimed GPU refs/pages, CPU lease bytes and suspend/resume latency.

Existing reusable tests and commands (not run during this design audit):

```bash
.venv/bin/python -m pytest tests/v1/kv_offload/cpu/test_manager.py \
  tests/v1/kv_offload/cpu/policies -q
.venv/bin/python -m pytest \
  tests/v1/kv_connector/unit/offloading_connector/test_scheduler.py \
  tests/v1/core/prefix_cache/test_mamba_eagle_resume_checkpoint.py -q
HIP_VISIBLE_DEVICES=6,7 .venv/bin/python -m pytest \
  tests/v1/kv_offload/cpu/test_gpu_worker.py \
  tests/v1/kv_connector/unit/offloading_connector/test_worker.py -q
```

Specifically preserve test_request_preemption, test_reset_cache,
test_async_preempt_readmit_before_transfer_output_is_deferred,
test_native_hybrid_cpu_roundtrip_is_byte_identical, and
test_load_waits_for_pending_compute_stream_writes. Those existing tests do not
qualify live paging: add the transaction, raw-ring and model continuation cases
above. No merge/readiness claim before those new gates pass on the actual layout.

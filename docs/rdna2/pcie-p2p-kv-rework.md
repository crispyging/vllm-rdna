# PCIe P2P KV handoff: review and rework target

## Required outcome

Qualify Flash-Next disaggregated prefill/decode with complete QSA/GDN/PLE
state, eager and FULL_AND_PIECEWISE graphs, prefix reuse and MTP2. Conventional
full-attention-only qualification is insufficient. PR #23 remains an unqualified
draft; this review does not enable its connector or change runtime defaults.

Reviewed original head: `bbf8c76bae1885c441576527e92dc41e5c869175`.
The draft targets the archived `rdna_extra/v0.28.0` base. The rework must target
current `rdna_extras` and preserve other platforms and existing launchers.

## Confirmed problems in the original draft

- The consumer uses prompt length instead of exported `remote_num_tokens`.
  A reproducer exports 16 tokens and observes 127 claimed matches.
- Block lists are truncated with `min()` and missing groups are skipped, then
  the transfer is acknowledged as complete. A reproducer requests two blocks,
  copies one and still produces an ACK.
- The block axis is inferred from the first dimension equal to capacity.
  For `[2, 2, 16, 4, 128]`, capacity 2, this chooses the K/V axis.
- Producer truncation for recurrent h(N-1) and speculative lookahead is absent.
  Sending h(N), then recomputing token N on decode, is not a valid handoff.
- Consumer ACKs use the local request ID instead of `remote_request_id`.
  Full local prefix hits return early without releasing producer ownership.
- The producer has no write-completion publication barrier; consumer stream
  synchronization establishes only completion of its own reads.
- IPC export omits allocation-base offsets and extents. Tensor views and
  caching-allocator suballocations cannot assume mapped allocation base equals
  tensor data pointer. Shared handles must be mapped/closed once.
- Peer-access setup uses the producer's ordinal in the consumer's visibility
  namespace. Separate visibility lists require identity-safe IPC handling.
- Layout compatibility, group completeness, upper block bounds, engine
  identity and TP compatibility are not validated sufficiently.
- Socket waits are unbounded. Handle rollback, thread/context shutdown,
  cancellation and restart/generation-safe ownership are missing.
- `kv_both` takes the producer-only branches and cannot consume.
- Calling HIP device-to-device memcpy does not prove SDMA selection, peer
  coherence or cross-root performance. Those need real transport evidence.

The original five CPU tests do not exercise the scheduler, worker, HIP calls
or any hybrid state. Reproducers and frozen original source are retained in
`bench_results/2026-10-28-pcie-p2p-review/`.

## State that must travel together

- Main QSA K/V pages (`FullAttentionSpec`).
- Compressed QSA keys (`MLAAttentionSpec`, compression-aware token geometry).
- Raw QSA compressor ring (`CircularBufferSpec`), including required
  speculative scratch. Native CPU prefix offload deliberately omits this
  ring; that exclusion is unsuitable for immediate P/D handoff.
- TP-sharded GDN convolution and recurrent state (`MambaSpec`).
- TP-replicated PLE convolution state (`MambaSpec`).

## Preferred implementation direction

Reuse the existing NIXL pull scheduler and worker, including request truncation,
all-group geometry, prefix alignment, TP mapping, leases, notifications,
completion aggregation and failures. Prove the ROCm UCX IPC transport works
before introducing another transport or copying its scheduler.

Repository evidence:

- `docker/Dockerfile.rocm` builds pinned UCX with `--with-rocm` and NIXL with
  `wheel_variant=rocm`.
- `test_nixl_rocm_gpu_mem_diag.py` specifically covers UCX `rocm_ipc`
  registration behavior. The original draft's blanket assertion that NIXL/UCX
  does not use ROCm IPC is unsupported by the current tree.
- `nixl/base_scheduler.py` supplies producer truncation, recurrent backoff and
  SSM-slot clipping.
- `nixl/base_worker.py` supplies explicit CSA-linear main/compressed/ring,
  GDN and PLE geometry, physical/logical mapping and per-group prefix slicing.

If the existing transport qualifies, replace the bespoke implementation with
reproducible setup, launcher integration and qualification evidence. If it
cannot satisfy the required transfer behavior, implement a scoped HIP-IPC
transport adapter while retaining these planner/state-machine semantics.
Such an adapter must implement the wrapper's registration, descriptors,
metadata, transfers/status, notifications and cleanup contracts; it is not a
one-function memcpy substitution. Avoid global monkeypatches.

## Validation status and gates

- [x] Reproduce token-count, partial-copy ACK and block-axis failures on CPU.
- [x] Existing Flash-Next NIXL geometry selection: 6 passed.
- [x] Existing recurrent N-1 handoff and SSM clipping: 7 passed.
- [x] Inspect isolated remote dependencies: neither `nixl_rocm` nor a UCX tool
  installation was found; a registry dry run found no `nixl-rocm` package.
- [x] Build pinned ROCm UCX/NIXL privately; inspect `rocm_ipc`/`rocm_copy`.
- [x] Diagnostic real IPC: offset regions sharing one allocation, separate
  visibility lists, nondefault producer stream, exact guard bytes, completion
  notification and explicit transfer/list/registration cleanup.
- [ ] Verify missing groups, out-of-range blocks, failure/cancel/restart,
  local prefix hits and remote request identity without false success ACKs.
- [ ] Byte-compare every Flash-Next state kind after transfer.
- [ ] Run normal-EOS decoding and prefix reuse, eager and graphs, MTP0/MTP2.
- [ ] Compare integrated P/D behavior with a matched monolithic baseline.
- [ ] Prove cross-PLX topology and selected transfer protocol; report throughput,
  TTFT, ITL/TPOT, KV transfer bytes/time and memory overhead.

Only physical GPUs 6-8 are authorized unless the user expands the allocation.
TP2 prefill plus TP2 decode requires four GPUs and cannot be fully qualified
within that allocation. ROCm's coarse topology matrix alone does not establish
which pair crosses PLX switches; use PCIe ancestry and per-device bus identity.

## Transport experiment results

Privately built UCX `96e58a16039f6d7d213bc967b8069238742c5194` and NIXL
`231d56753047c989062a5cb2ac703a1ad761c7d2` (1.4.0), matching the repository's
Dockerfile pins. Neither shared venvs nor installed serving extensions changed.

A six-region, 173,953-byte READ probe used two processes, each with one visible
GPU and local ordinal 0. Producer HIP 6 resolved to V620 gfx1030 PCI
0000:07:00.0; consumer HIP 8 resolved to V620 gfx1030 PCI 0000:12:00.0.
ROCm-SMI device numbers differ from these HIP ordinals; do not interchange them.
Both devices are behind the same root/switch complex. This is not cross-PLX
qualification and is not a model-state replay or throughput benchmark.

With UCX TLS `rocm_ipc,rocm_copy,self,sm,tcp` and default NIXL error mode
`peer`, exact bytes passed, but UCX protocol logs selected TCP software
emulation for the GPU READ. Removing TCP from TLS caused backend creation to
fail because no available active-message transport supported peer failure
handling. Merely listing `rocm_ipc` is not evidence that it carried payloads.

The diagnostic explicitly created the UCX backend with
`ucx_error_handling_mode=none`. UCX then selected `rocm_ipc` zero-copy for
READs of at least 128 bytes. All six synthetic regions, including an unaligned
513-byte region, and all untouched guard bytes matched exactly; the producer
received the completion notification and both sides released registrations.
This establishes a feasible IPC data path, not complete Flash-Next semantics.

Do not silently turn peer-error handling off in production. The next rework
must provide explicit opt-in configuration, bounded failure detection and safe
ownership through cancellation/process failure, and qualify that behavior
before full-model serving. Keep the mature hybrid scheduler and descriptor
logic. A generic duplicate scheduler is not warranted by the transport result.

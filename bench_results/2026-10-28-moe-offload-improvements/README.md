# MoE offload follow-up receipts

Target: Flash-Next AWQ INT4, gfx1030/HIP 7.14, TP2/EP2 on HIP GPUs 6/7.
Kernel/cache checks use GPU8. No GPU9 allocation was used.

## Wider K2 gate

`wide-k2/arm.sh.txt` records the serving recipe. Both arms use tensor copies,
96 expert slots, LFU/LRU decay 128, MTP2, segmented FULL_AND_PIECEWISE graphs,
3.75 GiB GPU KV per rank and a 16 GiB aggregate native CPU KV pool. TunableOp
uses the same committed lookup rows. Prefix state resets before each benchmark.

`wide-k2/comparison.json` contains aggregate output throughput, TTFT, TPOT, ITL,
MTP acceptance and forced-output warnings. Aggregate output includes prefill:

- 1k/512 c1: 15.986 -> 19.076 tok/s (+19.33%).
- 1k/512 c8: 56.012 -> 48.149 tok/s (-14.04%).
- 16k/1k c1: 22.305 -> 21.954 tok/s (-1.57%).
- 16k/1k c8: 28.763 -> 27.279 tok/s (-5.16%).

Keep gate 8. Each arm ran once, serially; expert-cache histories and MTP
acceptance differ, so these numbers do not isolate a kernel speedup. Greedy,
PPL, prefix and concurrency checks passed in both arms. PPL was 4.81/4.79;
prefix cold/warm TTFT was 12.64/2.32 s and 12.34/2.31 s. Each concurrency run
passed 48 checks with zero hard failures, retaining near-tie/soft differences.
Both summaries report zero new GPU fault markers.

Detailed benchmark outputs are preserved. Random-prompt benchmarks force their
output lengths; repetition/garbage screening warnings are recorded separately
and must not be treated as normal-EOS semantic correctness evidence.

## Native copies

`native-cache-tests.txt`: 29 integrated scratch cache tests passed on GPU8,
including native copies of all six payloads, consumer-stream changes, stable
addresses and invalid inputs rejected before writes. `native-receipts/`
preserves standalone compilation/test drivers. The forced native-loop fallback
also passed on HIP 7.14; this is not older-HIP runtime qualification.

`copy-api-probe.jsonl` and its driver are a HIP API prototype, not the integrated
C++ backend or a full-model speedup. CPU enqueue medians include descriptor
construction where labeled `batch_rebuild`; `batch_prebuilt` excludes it.
GPU event intervals can include host starvation and are not pure DMA time.
The 16-miss tensor-copy timing needs broader repetition before generalization.

Model-level native-copy quality and same-process performance qualification are
pending. Tensor copies remain the default. Pool sizes and tensor addresses are
unchanged by selecting the native submission backend; HIP command metadata and
host descriptor allocations are not a measured zero-overhead claim.

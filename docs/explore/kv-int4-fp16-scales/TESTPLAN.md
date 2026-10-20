# Test plan: INT4 + FP16 block scales for the RDNA2 KV cache

**Status**: explore. Nothing has been measured. Every cell in this document is
empty on purpose, and a number only goes in a cell when it comes from a run
that is recorded in the PR body.

The format under test is specified in `README.md`. The reference codec is
`reference_codec.py`; it needs no GPU and its output is already in the README.

## What has to be true for this format to be worth building

1. **H1 - accuracy.** Block-granular FP16 scales must beat the shipped
   `int4_per_token_head` scale by a useful margin on *real* KV activations,
   not on synthetic rows. The synthetic gap is 3.76 dB; the real gap is
   unknown.
2. **H2 - compute path.** Dequantising to FP16 must put the dot on
   `v_dot2_f32_f16` and not on the scalar FMA path, and that must be visible
   in decode throughput.
3. **H3 - it has to be reachable.** `RdnaAttentionBackend.supported_kv_cache_dtypes`
   is `["auto", "float16"]` today. Either the RDNA2 backend grows the dtype or
   the design is re-scoped to the Triton path.
4. **H4 - no global scale.** If the global scale turns out to matter on real
   activations, the format needs a per-tensor side buffer and the change
   stops being a kernel-sized change.

If H1 or H2 fails, the format is not worth a new dtype and the shipped
`int4_per_token_head` stays.

## Step 1 - capture real KV activations

The synthetic rows in the README are a guess. The capture is the part that
decides H1.

- **Where**: the RDNA2 attention kernel, after the K and V projections and
  before the cache write. `vllm/v1/attention/ops/int4_per_token_head.py`
  already has the pre-quantisation tensors in hand at the reshape kernel.
- **What**: a sample of real K and V rows, per layer, per head, with the
  layer id and head id kept, at a context long enough for the outliers to
  show up (16k tokens is the fork's standard prompt length).
- **Size**: enough rows to see the tail - a few hundred rows per layer, not a
  few. The format question is entirely about what happens in the worst
  channel of a row.
- **Storage**: raw FP16 dumps, so the reference codec can be re-run offline
  against them without a GPU.

| Capture | Model | Layers | Rows/layer | Context | File |
| --- | --- | --- | --- | --- | --- |
| K | Qwen3.8-Flash-Next | | | 16k | |
| V | Qwen3.8-Flash-Next | | | 16k | |
| K | DeepSeek-V4-Flash | | | 16k | |
| V | DeepSeek-V4-Flash | | | 16k | |

## Step 2 - offline codec comparison (no GPU)

Run the reference codec over the captured rows. This is the H1 check and it
needs no GPU, only the dumps from Step 1.

| Scheme | SNR dB (real K) | SNR dB (real V) | bits/value |
| --- | --- | --- | --- |
| `int4_per_token_head` (shipped) | | | 4.250 |
| `int4_fp16_block` k=64 | | | 4.250 |
| `int4_fp16_block` k=32 | | | 4.500 |
| `int4_fp16_block` k=16 | | | 5.000 |
| `int4_fp16_block` k=32, asymmetric | | | 4.625 |
| `int4_fp16_block` k=32 + global | | | 4.500 |
| `int4_e4m3_block` k=16, no global | | | 4.500 |
| `int4_e4m3_block` k=16 + global | | | 4.500 |
| NVFP4 (E2M1 + E4M3/16 + global) | | | 4.500 |

Acceptance for H1: the `k=32` row beats the shipped row by at least 2 dB on
both K and V. If it does not, stop here.

## Step 3 - memory budget check (no GPU)

Byte budget per token per layer, `head_size=128`:

| Scheme | Element bytes | Scale bytes | Total bytes | bits/value |
| --- | --- | --- | --- | --- |
| `int4_per_token_head` | 64 | 4 (zp in the scale) | 68 | 4.250 |
| `int4_fp16_block` k=32 | 64 | 8 | 72 | 4.500 |

The scale bytes sit beside the data in the same page, so the page size has to
grow by 4 bytes per head per token for the proposal. That is the whole
memory argument and it is small; the argument for the format is accuracy and
the compute path, not the bytes.

## Step 4 - kernel, then decode A/B (GPU)

Only after Steps 1-3 hold does a kernel get written.

- **Kernel**: a reshape (write) kernel and a split-dot (read) kernel next to
  `vllm/v1/attention/ops/int4_per_token_head.py`. Write path: FP16 block
  scale per `k` values, nibble-packed elements. Read path: unpack to FP16 and
  dot with `v_dot2_f32_f16`.
- **Correctness**: round-trip against the reference codec, and the existing
  per-token-head test suite extended to the new mode
  (`tests/quantization/test_per_token_kv_cache.py`).
- **Wave32**: `num_warps` <= 4 and RDNA-sized tiles, per the fork's RDNA code
  rules.
- **Launch**: current stream, per the fork's RDNA code rules.
- **Cudagraphs**: correct under `FULL_AND_PIECEWISE` and under
  `--enforce-eager`.

Serve through the in-tree launcher; do not hand-write a `vllm serve` argv:

```bash
MODEL=/path/to/checkpoint VENV=/path/to/venv \
  bash tools/rdna/serve_rdna.sh RECIPE=flashnext
```

Decode matrix, one cell per row, all four numbers per cell as the fork
requires:

| Arm | KV dtype | prefill tok/s | decode tok/s per req | TTFT (s) | ITL (ms) | 16k/1k c=1 | 16k/1k c=8 | 1k/512 c=1 | 1k/512 c=8 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| A | fp16 | | | | | | | | |
| B | `int4_per_token_head` | | | | | | | | |
| C | `int4_fp16_block` k=32 | | | | | | | | |

Acceptance for H2: arm C's decode tok/s per request beats arm B's at c=1 and
c=8, with prefill within noise and identical greedy output.

## Step 5 - output quality

If the format changes the numbers the model produces, it needs a quality
result, not just a speed result. The fork's standard is output-token parity
against the FP16 baseline, and `tools/rdna/port_v031/ppl_probe.py` for a
perplexity number on the pruned DeepSeek-V4 checkpoint.

| Arm | Perplexity / probe | Parity vs fp16 | Notes |
| --- | --- | --- | --- |
| A (fp16) | | - | baseline |
| B (`int4_per_token_head`) | | | |
| C (`int4_fp16_block` k=32) | | | |

## Records

Nothing goes in a cell above without a run log next to it. When a cell is
filled, the PR body gets the command, the box, the model, the context length
and the raw output.

| Run | Date | Box | Recipe | Log |
| --- | --- | --- | --- | --- |
| | | | | |

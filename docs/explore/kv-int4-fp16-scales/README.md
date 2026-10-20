# Explore: INT4 + FP16 block scales for the RDNA2 KV cache

**Status**: explore. This is a format contract plus a reference codec, not an
implementation. No dtype is added, no backend changes, nothing changes by
default. The kernels, the backend hook and the measurement on the V620 are all
still to be written.

The question this document answers: *if the fork wants a 4-bit KV cache on
gfx1030, what should the scale format be?* The short version is that the
scale format the fork already ships is the wrong granularity, that NVFP4's
narrow block scale is a Blackwell constraint rather than an accuracy optimum,
and that the second (global) level NVFP4 carries buys nothing for activation
data, so it can be dropped.

## The question

`--kv-cache-dtype int4_per_token_head` already exists in this fork
(`vllm/config/cache.py:53`). It is the only 4-bit KV cache dtype here, and on
gfx1030 it is the wrong shape for two independent reasons.

**Its granularity is a whole head row.** The scale is
`(max - min) / 15` over all `head_size` values of a row, with a 4-bit
zero-point hidden in the low mantissa bits of the FP32 scale
(`vllm/v1/attention/ops/int4_per_token_head.py:158`, write side; `:550`, read
side). One channel with a large magnitude therefore sets the step size for
every other channel in the row. The reference codec measures the cost of that
on synthetic rows with two channels scaled 40x: **14.57 dB**, against
**20.03 dB** for the same rows without outliers.

**Its dequant lands on the wrong hardware path.** The cached values are
unpacked to FP32 and multiplied against an FP32 query
(`int4_per_token_head.py:440`, `:527`, and the dot at `:571`). gfx1030 has no
FP32 matrix instruction, so an FP32 `tl.dot` cannot use the packed path. The
established RDNA2 inner product in this fork is `v_dot2_f32_f16` with FP32
accumulate (`docs/rdna/monokernel/README.md:104`). An FP32 dot is the
scalar-`v_fma_f32` fallback, which that document describes as the path kept
only for deployments this fork does not target.

## The proposal

A new KV dtype, `int4_fp16_block`:

| Field | Value |
| --- | --- |
| Element | INT4, signed two's complement, symmetric (no zero point) |
| Packing | 2 elements per byte, low nibble first (same as `int4_per_token_head`) |
| Block scale | FP16 (E5M10), one per `k` values along the head dimension |
| Global scale | none (see *Two levels* below) |
| Bits/value | `4 + 16/k`; at `k=32` that is **4.500** |
| Dequant | `x_hat = q * scale`, in FP16 |

The scale cache would be `[num_blocks, block_size, num_kv_heads,
head_size/k]` FP16, mirroring the shape of the existing
`[num_blocks, block_size, num_kv_heads]` FP32 scale cache
(`vllm/v1/attention/backends/triton_attn.py:405`, `_ensure_scale_caches`).

`k=32` is the proposal's baseline because it is the granularity KIVI measured
as best for the value tensor, and because it costs exactly the same
4.5 bits/value as NVFP4 while giving the scale 10 mantissa bits instead of
E4M3's 3.

### Two levels

NVFP4 is a two-level format: an E4M3 scale per 16 elements, times an FP32
scale per tensor. The global is folded into the stored block scale at write
time and multiplied back at read time
(`csrc/libtorch_stable/nvfp4_kv_cache_kernels.cu`,
`cvt_warp_fp16_to_fp4` in `csrc/libtorch_stable/quantization/fp4/nvfp4_utils.cuh`).

That second level is not free of consequences. E4M3 is a narrow format: its
normal range runs from `2^-6` to 448, a ratio of about 2.9e4, so a block scale
has to fit inside that once the global has been divided out. FP16's normal
range runs from `2^-14` to 65504, a ratio of about 1.1e9 - roughly 37000x
wider.

The reference codec measures what the second level is worth. The global is
modelled at its most favourable value, the tensor maximum:

| | FP16 block scale | E4M3 block scale |
| --- | --- | --- |
| SNR, no global (outlier rows) | 18.326 dB | 19.984 dB |
| SNR, with a global (outlier rows) | 18.326 dB | 20.043 dB |
| SNR, no global (clean rows) | 20.341 dB | 21.331 dB |
| SNR, with a global (clean rows) | 20.341 dB | 21.369 dB |
| Blocks whose stored scale went subnormal, of 512 (with a global) | 0 | 15 |

So for an FP16 block scale the global scale **buys nothing**: the SNR is the
same to three decimals with and without it, and no block scale underflows.
For an E4M3 block scale it buys 0.06 dB, and it is what pushes 15 of the 512
block scales into E4M3's subnormal range, where they lose mantissa bits.

The reason NVFP4 carries a global at all is that its global is a *checkpoint*
constant (`layer._k_scale`), not the tensor maximum. Dividing by a calibration
constant that is far from the tensor maximum is exactly the case where a
narrow E4M3 block scale underflows, and where the second level earns its
keep. A KV cache has no checkpoint scale. Its rows are activations, and the
natural global - a running maximum - would invalidate every block already
written, because the blocks were divided by the old value.

That is the concrete reason this proposal drops the second level: not because
two-level scaling is wrong, but because for activation data it has nothing to
calibrate against, and a single FP16 block scale has the range to not need it.

## What the reference codec says

`reference_codec.py` in this directory implements all of the above in pure
Python and prints the table below. The numbers are arithmetic on synthetic
rows - 64 rows of `N(0, 0.5)`, `head_size=128`, with two channels scaled 40x
in the first case. **They are not V620 measurements.**

| Scheme | SNR dB (outliers) | SNR dB (clean) | bits/value |
| --- | --- | --- | --- |
| `int4_per_token_head` (shipped) | 14.570 | 20.035 | 4.250 |
| `int4_fp16_block` k=64 | 15.475 | 19.495 | 4.250 |
| **`int4_fp16_block` k=32 (proposal)** | **18.326** | **20.341** | **4.500** |
| `int4_fp16_block` k=16 | 20.322 | 21.442 | 5.000 |
| `int4_fp16_block` k=32, asymmetric | 20.341 | 21.867 | 4.625 |
| `int4_fp16_block` k=32 + global | 18.326 | 20.341 | 4.500 |
| `int4_e4m3_block` k=16, no global | 19.984 | 21.331 | 4.500 |
| `int4_e4m3_block` k=16 + global | 20.043 | 21.369 | 4.500 |
| NVFP4 (E2M1 + E4M3/16 + global) | 21.724 | 20.472 | 4.500 |

Reading it:

- **The proposal beats the shipped format by 3.76 dB on outlier-heavy rows**
  for 5.9 % more memory (4.500 vs 4.250 bits/value). It is a replacement
  candidate for `int4_per_token_head` on RDNA2, not an additional dtype.
  The shipped format's zero-point is hidden in the low mantissa bits of its
  FP32 scale, so it costs no extra bytes and its true cost is 4.250.
- **`k=64` costs exactly what ships today** (4.250 bits/value) and buys
  +0.90 dB on outlier rows, but it is 0.54 dB *worse* than the shipped
  format on clean rows. The accuracy that block granularity buys arrives
  sharply between `k=64` and `k=32`. `k=16` is the other end of the
  frontier and costs 5.000 bits/value.
- **NVFP4's own element format beats the proposal on outlier rows** (21.724 vs
  18.326 dB at the same 4.5 bits/value): E2M1's non-uniform levels - 0.5, 1,
  1.5, 2, 3, 4, 6 - spend their precision near zero, which is where a
  zero-centred activation distribution lives. On clean rows the ordering
  reverses and the proposal is ahead of NVFP4 (20.341 vs 20.472 dB). Whether
  the element type should be INT4 or E2M1 is therefore an open question, not
  a settled one.
- **The asymmetric variant is the accuracy winner on clean rows** (21.867 dB
  at 4.625 bits/value) but costs more memory than the proposal.

## Hypotheses and kill criteria

| # | Hypothesis | Check | Holds if | If not |
| --- | --- | --- | --- | --- |
| H1 | Block-granular FP16 scales beat the shipped per-row scale on real KV | capture real K/V activations from a V620 decode run, run the reference codec over them | SNR gain over `int4_per_token_head` >= 2 dB | stop: the format is not worth a new dtype, keep the shipped one |
| H2 | Dequantising to FP16 puts the dot on `v_dot2_f32_f16` instead of the scalar FMA path | A/B the decode kernel on the V620, FP16 operands against FP32 operands | decode tok/s at c=1 improves, prefill within noise | keep FP16 for the memory and accuracy, drop the speed claim |
| H3 | The format lands in the RDNA2 backend at all | wire `int4_fp16_block` into `RdnaAttentionBackend.supported_kv_cache_dtypes` (`vllm/v1/attention/backends/rdna_attn.py:583`, currently `["auto", "float16"]`) | a decode run completes with the new dtype and the cache size matches the byte-budget arithmetic | stop: the Triton path is the only viable target and the design must be re-scoped for it |
| H4 | An FP16 block scale does not need a global scale | replay the H1 capture with the global on and off | SNR identical to within 0.01 dB | the global becomes mandatory and the format needs a per-tensor side buffer, which is a much larger change |

## Where it would plug in

Nothing below exists yet. This is the surface a future implementation would
have to touch.

| Piece | File | What changes |
| --- | --- | --- |
| dtype literal | `vllm/config/cache.py:39` | add `"int4_fp16_block"` to `CacheDType` |
| dtype to torch dtype | `vllm/utils/torch_utils.py:43` | add the `torch.uint8` entry |
| quantised-predicate | `vllm/utils/torch_utils.py:78` | `is_quantized_kv_cache` already returns True for any `*per_token_head`; a new dtype needs its own branch |
| mode enum | `vllm/v1/kv_cache_interface.py:39` | add a `KVQuantMode` value and map it in `get_kv_quant_mode` (`:84`) |
| page size | `vllm/v1/kv_cache_interface.py:86` | the scale bytes are part of the page, so the spec has to account for `head_size/k` FP16 scales per head |
| attention ops | `vllm/v1/attention/ops/` | a new module next to `int4_per_token_head.py`, owning nibble pack/unpack, the reshape (write) kernel and the split-dot (read) kernel |
| backend opt-in | `vllm/v1/attention/backends/rdna_attn.py:583` | `supported_kv_cache_dtypes` |

The Triton backend already lists `int4_per_token_head`
(`vllm/v1/attention/backends/triton_attn.py:320`), so H3 is the fork's own
question: whether the RDNA2 backend is the target, or whether the Triton path
is.

## What this PR contains

- `docs/explore/kv-int4-fp16-scales/README.md` - this document.
- `docs/explore/kv-int4-fp16-scales/reference_codec.py` - the reference codec
  and comparison harness. Pure Python, no dependencies, self-checking.
- `docs/explore/kv-int4-fp16-scales/TESTPLAN.md` - the measurement plan, with
  the V620 cells left empty.

## What this PR does not contain

- No dtype, no backend change, no kernel. Nothing in the serving path
  changes.
- No performance claim. `TESTPLAN.md` has empty cells; the only numbers in
  this document come from `reference_codec.py` and are labelled as arithmetic
  on synthetic data.
- No element-format decision. INT4 and E2M1 are both in the table above and
  the choice is open.

## Reproducing the numbers

```bash
python3 docs/explore/kv-int4-fp16-scales/reference_codec.py
```

The script runs its own self-check first (FP16 and E4M3 round-trip, INT4
packing, and the scale-format invariants) and prints the two tables above.

## AI assistance

This draft was written with AI assistance. Every file:line reference above was
checked against the tree, and the numbers come from the committed script
rather than from the author. No hardware measurement was taken for this PR:
the V620 cells in `TESTPLAN.md` are empty on purpose.

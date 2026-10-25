# rdna_hint_flex weights on RDNA2 (HadamardINT2-FP16 spec v3.2)

`quant_method: rdna_hint_flex` (alias `hadamard_int2`) loads checkpoints
written by the separate `tools/hi2` quantizer (spec:
`docs/rdna2/hadamard-int2-v3.2-spec.md` on the `feat/hadamard-int2-weights`
branch). This page covers the vLLM side: the loader, the gfx1030 kernels,
the decisions taken where the spec is open, and how it was checked.

## Name

The canonical `quant_method` is `rdna_hint_flex`; `hadamard_int2` (the name
the first quantizer releases write) is an accepted alias. Both live in
`vllm/model_executor/layers/quantization/rdna_hint_flex_names.py`, the only
place to change them. The checkpoint tensor prefix stays `hi2_`.

## Format, as read

Each quantized `<name>.weight` is replaced by flat per-row buffers
`<name>.hi2_<field>` (spec 2.4):

| field | dtype | shape |
| --- | --- | --- |
| `codes` | int32 | sum over rows of `K * nbits / 32` words, codes LSB-first |
| `code_off` | int32 | `R + 1`, first word of each row |
| `blk_params` | uint8 | `2 * sum K / B`, `(lo_u8, step_u8)` per block |
| `param_off` | int32 | `R + 1`, first block of each row |
| `row_consts` | fp16 | `[R, 3]`: `lo_min, lo_scale, step_scale` |
| `row_nbits` | uint8 | `R`: 2, 4 or 8 |
| `row_block` | uint8 | `R`: 16, 32 or 64 |
| `had` | uint8 | `K / 16` Hadamard flags |
| `col_scale` | fp16 | `K`, optional smoothing vector |

Decode (spec 2.2): `lo = fp16(fma_f32(lo_u8, lo_scale, lo_min))`,
`step = fp16(step_u8 * step_scale)`, `w = fp16(lo + code * step)`. The stored
basis is `H (W * s)`, so `y = H(x / s) . W_stored^T` (spec 2.3).

Which modules are quantized is read from the checkpoint index: a module stored
as `.hi2_*` uses this format, one stored as `.weight` stays dense (vision
tower, embeddings, head, GDN `conv1d`, ...). No module list is needed in
`quantization_config`.

## Decisions where the spec is open (for the spec to match)

1. **Offsets.** `code_off` and `param_off` are **int32** (the spec table says
   uint32), each with `R + 1` entries starting at 0. `code_off` counts
   **32-bit words** of `codes`, `param_off` counts **blocks** (one block is two
   bytes of `blk_params`, so the byte offset is `2 * param_off`). The loader
   checks `code_off[r+1] - code_off[r] == K * nbits_r / 32`,
   `param_off[r+1] - param_off[r] == K / B_r` and that the last entries match
   the buffer lengths.
2. **INT8 rows** (spec 11.6) are accepted with **any** block size of 16, 32 or
   64 (the spec lists 32 or 64). Four codes per word, one byte each, LSB first,
   same ladder and parameters as INT2/INT4.
3. **INT8 rows in I8 mode.** Codes 0..255 do not fit the signed operand of
   `v_dot4_i32_i8` (and `v_dot4_i32_iu8` does not exist on gfx1030, spec 7):
   they enter the dot as `code - 128` and `128 * sumq` is added back to the
   integer sum, which is exact.
4. **I8 activation.** Per token and 16-column group (spec 5, A = 16), from the
   fp32 transformed activation (not its fp16 rounding): `sx = fp16(absmax /
   127)`, `xq = clamp(rint(x / sx), -127, 127)` with rint rounding half to even,
   `sumq = sum xq` (stored int32). A group whose absmax is 0 gets `sx = 0` and
   `xq = 0`.
5. **F16 activation.** The transform (`x / s` then the order-16 Hadamard) runs
   in fp32 and is rounded once to fp16 before `v_dot2_f32_f16`.
6. **Routed experts.** Each expert is its own tensor with its own smoothing
   vector and flags; **gate and up of one expert must share them** (they see
   the same tokens). A checkpoint where they differ is rejected.
7. **Row-parallel TP** needs the per-rank K to be a multiple of 64 (no word,
   block or Hadamard group is cut).

## Files

| file | content |
| --- | --- |
| `csrc/rocm/rdna_hint_flex_rdna2.cu` | kernels and host wrappers (gfx1030 build only) |
| `vllm/model_executor/layers/quantization/rdna_hint_flex.py` | `RdnaHintFlexConfig`, `RdnaHintFlexLinearMethod` |
| `vllm/model_executor/layers/quantization/rdna_hint_flex_moe.py` | `RdnaHintFlexMoEMethod`, dense MoE reference |
| `vllm/model_executor/layers/quantization/rdna_hint_flex_names.py` | the name and its aliases |
| `vllm/model_executor/layers/quantization/utils/rdna_hint_flex_utils.py` | host helpers: validate, slice rows/columns, concatenate, PyTorch dequantize, transform and the I8 reference |
| `tests/kernels/quantization/test_rdna_hint_flex_rdna2.py` | reference port of `tools/hi2/hi2/format.py`, CPU and gfx1030 tests |
| `tools/rdna/rdna_hint_flex/` | env, sync/build, test runner, end-to-end check, kernel timing, serving benchmark |

Outside the new files: op declarations (`csrc/rocm/ops.h`), registration in
the `VLLM_ROCM_GFX1030` block of `csrc/rocm/torch_bindings.cpp`, the gfx1030
source list in `CMakeLists.txt`, wrappers and fake kernels in
`vllm/_custom_ops.py`, the registry in `quantization/__init__.py` (it also
appends the names to `RocmPlatform.supported_quantization`), and two hooks in
`fused_moe/routed_experts.py` (expert `hi2_*` tensors go to the `w13_weight` /
`w2_weight` loader and to the method's `absorb()`, as for EXL3). Nothing in
`fa_rdna2` or the attention backends changed.

## Compute modes

`compute_mode` in `quantization_config`, overridden by the
`VLLM_RDNA_HINT_FLEX_COMPUTE_MODE` environment variable (read when the config
is built, so it reaches the workers with it):

* `f16` (default, spec 6.1): fp16 activations, `v_dot2_f32_f16`.
* `i8` (spec 6.2): int8 activations and `v_dot4_i32_i8` for M <= 48 (decode),
  the f16 prefill path above.
* `i8_all`: the I8 kernel for every M (slow prefill; for accuracy runs).

Routed experts follow the same mode (I8 for batches up to 48 tokens with
`i8`, always with `i8_all`).

## Kernels

* `rdna_hint_flex_act_transform_rdna2`: `x' = H16(x / s)` on the flagged
  groups. One thread per (token, group).
* `rdna_hint_flex_gemv_rdna2` (F16 decode, M <= 48): one wave32 per row (or 2
  to 4 waves per row with an LDS reduction for tensors under 8192 rows), lanes
  take consecutive words, two in flight per lane. A word never crosses a block:
  one `lo`/`step` per word, code to fp16 by `half(0x6400 | c) - 1024` (exact up
  to 1023, so INT8 too), one `v_pk_fma_f16`, `v_dot2_f32_f16`, fp32
  accumulation, wave reduction.
* `rdna_hint_flex_dequant_rdna2`: stored-basis fp16 `W [R, K]`, one 16-column
  group per thread, 16-byte stores.
* `rdna_hint_flex_act_quant_rdna2` + `rdna_hint_flex_gemv_i8_rdna2` (I8):
  transform and quantization in one thread per (token, group); the GEMV spreads
  one group's codes into four byte-packed dwords and takes four
  `v_dot4_i32_i8`, then `sx * (step * S + lo * sumq)` in fp32.
* `rdna_hint_flex_linear_rdna2`: transform, then GEMV for M <= 48, otherwise
  dequant into scratch and a rocBLAS GEMM. Chosen inside the op at execution
  time, so torch.compile cannot fix it to the warmup shape.
* `rdna_hint_flex_moe_rdna2` (routed experts, F16): experts stacked row-wise
  into one ragged tensor per stage with `[E, K]` / `[E, K/16]` transform tables;
  a transform per (token, expert) pair, a GEMV per pair for gate/up, SiLU and
  mul, and the down GEMV with the routing-weighted top-k sum inside the kernel
  (no atomics, fixed shapes, cudagraph-capturable). Pairs whose expert is not
  on this rank (`-1`) add nothing.

All launches use the current torch stream and `C10_CUDA_KERNEL_LAUNCH_CHECK`.
The kernels trust the offsets; the loader validates them on the host first.

Tried and dropped (slower on V620, measured with `bench_kernel.py`): staging
the activation in LDS, lanes owning 16-column groups and reusing the activation
in registers across 2-8 rows, and fusing the transform into the GEMV (VGPRs 43
to 126, occupancy 16 to 8 waves/SIMD; 1.8x to 3x slower at M=1).

## Loader and tensor parallelism

The loader collects each checkpoint tensor per output shard (`None`, an int, a
tuple such as Qwen3.5's `in_proj_qkv -> (0, 1, 2)`, or `q`/`k`/`v`),
splits fused checkpoint tensors into the layer's partitions, slices rows for
column-parallel layers (GQA kv replication included) and 64-aligned columns for
row-parallel layers, validates, and concatenates consecutive partitions that
share `col_scale` and `had` into one tensor, so fused q/k/v and gate/up are one
kernel call (in the tool's checkpoints they always share them). Experts slice
gate/up rows and down columns the same way.

Where the kernels are not built (not gfx10x, or not ROCm), dense layers are
dequantized once to fp16 and run through `F.linear` after the PyTorch transform;
routed experts need the kernels.

Activations must be fp16 (`--dtype float16`): RDNA2 has no bf16 dot.

## Quantizing routed experts (tool side)

No HI2 MoE checkpoint exists yet; the method is tested on a synthetic one. For
`tools/hi2` to produce one:

* Per-expert 2-D tensors (`mlp.experts.E.{gate,up,down}_proj.weight`) are
  already eligible (2-D, K a multiple of 64) and the router is excluded by
  `DEFAULT_EXCLUDE`; they are written as `mlp.experts.E.<proj>.hi2_*`, which is
  what vLLM loads.
* Checkpoints that store experts fused in 3-D (`experts.gate_up_proj [E, 2N, K]`)
  are skipped by `is_eligible` (2-D only); the writer must split them per expert
  under the names above.
* Gate and up of one expert must get the same smoothing vector and flags
  (decision 6); deriving `s` from the expert's routed-token statistics, shared
  by both, gives that.
* Calibration per expert: rarely routed experts see few tokens (the spec's
  data-free fallback applies and must be logged), one Hessian per expert input
  (gate/up share it) is `E * K^2` floats per layer, and the sensitivity probe's
  one forward per tensor multiplies by the expert count; probing experts in
  groups is needed to keep it practical.

## Checks

Unit tests: `bash tools/rdna/rdna_hint_flex/run_tests.sh <hip_device>`,
**163 passed** on a V620. The dequant kernel and the I8 activation quantization
are bit-identical to the reference; `rdna_hint_flex_linear_rdna2` matches
`reference_linear` (float64) within 2e-3 relative error for M = 1..37 (F16),
the I8 GEMV matches the integer-exact I8 reference within 1e-3; the synthetic
4-expert MoE matches the dense reference within 1e-2 (F16) and 3e-2 (I8); the fused prefill GEMM matches reference_linear within 2e-3.

End to end (`tools/rdna/rdna_hint_flex/e2e_check.py`: teacher-forced log-probs
on a fixed 137-token text and 4 greedy prompts of 48 tokens, against the
checkpoint's `hi2 export-dense`; 3.0 bpw checkpoints, sensitivity allocation):

| model | mode | ppl ours / dense | mean / max abs log-prob gap | greedy identical | weights in GPU |
| --- | --- | --- | --- | --- | --- |
| Qwen2.5-0.5B-Instruct | F16, eager | 24.454 / 24.471 | 0.005 / 0.031 | 4/4 | 0.39 vs 0.93 GiB |
| Qwen2.5-0.5B-Instruct | F16, F&P graphs | 24.457 / 24.471 | 0.005 / 0.034 | 4/4 | 0.39 GiB |
| Qwen3.5-0.8B | F16, eager | 27.954 / 27.947 | 0.004 / 0.017 | 4/4 | 0.78 vs 1.53 GiB |
| Qwen3.5-4B | F16, F&P graphs | 10.410 / 10.405 | 0.003 / 0.024 | 4/4 | 3.42 vs 8.81 GiB |
| Qwen3.5-4B | F16, F&P graphs, **TP=2** vs TP=1 | 10.417 / 10.410 | 0.003 / 0.019 | 4/4 | 1.79 GiB per GPU |

I8 accuracy against F16 (same checkpoints, eager; `i8_all` runs the I8 kernel
in prefill too, so the teacher-forced numbers measure it; `i8` only changes
decode, so its teacher-forced numbers equal F16 and only greedy can move):

| model | i8_all ppl vs F16 | mean / max abs log-prob gap vs F16 | greedy identical vs F16 (`i8_all` / `i8`) |
| --- | --- | --- | --- |
| Qwen2.5-0.5B | 24.479 vs 24.454 (+0.10%) | 0.015 / 0.081 | 3/4 (one prompt from token 4) / 4/4 |
| Qwen3.5-0.8B | 27.919 vs 27.954 (-0.13%) | 0.013 / 0.087 | 3/4 (one from token 36) / 3/4 (from token 36) |
| Qwen3.5-4B | 10.401 vs 10.410 (-0.08%) | 0.009 / 0.045 | 4/4 / 4/4 |

So I8 costs about three times the log-prob noise of F16 (F16 against the dense
export: 0.003-0.005) and no measurable perplexity on this short text.

## Performance

### Prefill: fused decode-in-GEMM against dequant + rocBLAS (decided: keep dequant + GEMM)

`rdna_hint_flex_gemm_rdna2` (128 x 128 tiles, 64-column K steps, the next
step's activation, weight words and block parameters prefetched into registers,
the ladder decoded on the LDS store, `v_dot2_f32_f16` on 8 x 8 micro-tiles) is
built and tested, and selectable with `RDNA_HINT_FLEX_PREFILL=1`. It does not
beat rocBLAS at prefill sizes, so the default stays dequant + rocBLAS GEMM.

Isolated (`tools/rdna/rdna_hint_flex/bench_prefill.py`, one V620, tuned
TunableOp rows), time relative to the dense fp16 `F.linear` of the same shape:

| shape (R x K) | path | M=128 | M=512 | M=2048 |
| --- | --- | --- | --- | --- |
| gate_up 18432 x 2560 | dequant + GEMM | 1.15 | 1.11 | 1.03 |
| gate_up 18432 x 2560 | fused | **0.71** | 1.17 | 1.16 |
| down 2560 x 9216 | dequant + GEMM | 1.08 | 1.08 | 1.06 |
| down 2560 x 9216 | fused | **0.83** | 1.09 | 1.19 |
| in_proj_qkvz 12288 x 2560 | dequant + GEMM | 1.04 | 1.12 | 1.03 |
| in_proj_qkvz 12288 x 2560 | fused | **0.25** | 1.17 | 1.21 |
| out_proj 2560 x 4096 | dequant + GEMM | 1.45 | 0.99 | 1.06 |
| out_proj 2560 x 4096 | fused | 1.14 | 1.20 | 1.09 |

The fused kernel reaches 19-21 TF/s at M = 2048 against rocBLAS's 21-24
TF/s; it only wins at M around 128, where rocBLAS has no good solution (a size
that serving rarely produces: decode goes to the GEMV, prefill chunks are 512 to
2048 tokens). Earlier variants (no prefetch, 32-column K steps, 64 x 64 and
128 x 64 tiles) were slower still (13-17 TF/s). In serving (Qwen3.5-4B, TP=1,
tuned rows), the fused path lost: 1k/512 c=1 prefill 1809 vs 1932 tok/s, 16k/1k
c=1 1334 vs 1407 tok/s (full table under "Serving").

### I8 for prefill (not worth it on gfx1030 with this activation grouping)

The spec's per-16-column activation scale (spec 5) forces a float rescale every
16 columns: per 16 multiply-adds an I8 GEMM issues 4 `v_dot4_i32_i8` plus a
convert, a multiply and two FMAs (8 instructions), the same as F16's 8
`v_dot2_f32_f16`; it would not be faster than the F16 GEMM, only less accurate.
The existing `i8_all` mode runs prefill through the I8 GEMV in 8-token tiles and
reaches 6 TF/s (3.4-4x slower than dense at M = 512-2048,
`bench_prefill.py --mode 2`); it is kept for accuracy evaluation. A per-token
activation scale would let the int32 sum run over a whole weight block (16-64
columns) and could roughly halve the instruction count, but spec 5 measured it
costing about 9% more output error on INT4 rows; that is a spec decision, not
done here.

### Kernel timing (decode)

`tools/rdna/rdna_hint_flex/bench_kernel.py`, one V620, CUDA graph of 50 calls;
random tensors with 30% INT4 rows and mixed blocks; microseconds per call
including the activation transform:

| shape (R x K) | F16 M=1 | I8 M=1 | F16 M=3 | F16 M=8 | F16 M=32 |
| --- | --- | --- | --- | --- | --- |
| gate_up 18432 x 2560 | 40.1 (475 GB/s) | 33.9 | 71.5 | 138.5 | 535.9 |
| down 2560 x 9216 | 28.1 (338 GB/s) | 18.5 | 41.5 | 74.8 | 294.3 |
| in_proj_qkvz 12288 x 2560 | 30.5 (415 GB/s) | 28.6 | 47.4 | 93.5 | 366.9 |
| out_proj 2560 x 4096 | 15.6 (273 GB/s) | 12.6 | 27.9 | 36.8 | 127.0 |

### TunableOp

The rocBLAS paths (prefill GEMM after the dequant, the dense arm's layers,
`lm_head` and the remaining dense layers) were tuned per the profile policy:
census during the bench cells (`bench_serve.sh CENSUS_DIR`), then
`tools/rdna/rdna_hint_flex/tunableop_campaign.sh` (tune, measure heuristic /
current / new, curate at 3 %, additive merge into all four rank files of
`tunableop/rocm7.14-rocblas5.5/`, 1860 -> 2594 rows, lookup hits 2500/2500).
605 Qwen3.5-4B and 136 Qwen2.5-0.5B rows were adopted (median 1.60x and 1.26x
over the heuristic per shape, 1.21x for the 4B's M >= 512 prefill shapes).
Details: `tunableop/rocm7.14-rocblas5.5/README.md` and `provenance.json`.

### Serving

`tools/rdna/rdna_hint_flex/bench_serve.sh` (vLLM OpenAI server, `RECIPE=full` =
FULL_AND_PIECEWISE cudagraphs, `vllm bench serve` random dataset with
`--ignore-eos`, n = 4 / 32 / 4 / 16 requests). Qwen3.5-4B, 3.0 bpw
rdna_hint_flex against its dense fp16 export, TP=1 on HIP 2, TP=2 on HIP 4-5,
TP=4 on HIP 2-5. Prefill and decode per request (aggregate in brackets).

With the tuned rows:

| cell | arm | prefill tok/s | decode tok/s | TTFT s | ITL ms (median) |
| --- | --- | --- | --- | --- | --- |
| 1k/512 c=1 | TP=1 ours | 1932 (1910) | 78.4 (72.6) | 0.53 | 12.8 |
| 1k/512 c=1 | TP=1 dense | 1961 (1941) | 38.1 (36.7) | 0.52 | 26.3 |
| 1k/512 c=1 | TP=2 ours | 3436 (3399) | 102.7 (97.1) | 0.30 | 9.7 |
| 1k/512 c=1 | TP=2 dense | 3418 (3374) | 61.4 (59.4) | 0.30 | 16.3 |
| 1k/512 c=1 | TP=4 ours | 6541 (6529) | 113.9 (110.2) | 0.16 | 8.8 |
| 1k/512 c=1 | TP=4 dense | 6686 (6631) | 93.3 (90.9) | 0.15 | 10.7 |
| 1k/512 c=8 | TP=1 ours | 410 (2537) | 35.4 (242.0) | 2.50 | 26.8 |
| 1k/512 c=8 | TP=1 dense | 416 (2570) | 27.8 (196.2) | 2.46 | 34.6 |
| 1k/512 c=8 | TP=2 ours | 759 (4639) | 55.7 (389.2) | 1.35 | 17.2 |
| 1k/512 c=8 | TP=2 dense | 788 (5064) | 46.4 (332.3) | 1.30 | 20.7 |
| 1k/512 c=8 | TP=4 ours | 1492 (7518) | 71.3 (521.3) | 0.69 | 13.2 |
| 1k/512 c=8 | TP=4 dense | 1514 (7629) | 70.2 (514.1) | 0.68 | 13.5 |
| 16k/1k c=1 | TP=1 ours | 1407 (1402) | 57.8 (34.9) | 11.65 | 17.3 |
| 16k/1k c=1 | TP=1 dense | 1416 (1411) | 32.6 (23.9) | 11.57 | 30.7 |
| 16k/1k c=1 | TP=2 ours | 2647 (2626) | 79.0 (53.5) | 6.19 | 12.7 |
| 16k/1k c=1 | TP=2 dense | 2647 (2633) | 52.3 (39.8) | 6.19 | 19.1 |
| 16k/1k c=1 | TP=4 ours | 4724 (4705) | 98.5 (73.9) | 3.47 | 10.2 |
| 16k/1k c=1 | TP=4 dense | 4756 (4729) | 82.4 (64.6) | 3.45 | 12.2 |
| 16k/1k c=8 | TP=1 ours | 583 (2668) | 10.1 (63.3) | 28.10 | 57.6 |
| 16k/1k c=8 | TP=1 dense | 589 (2697) | 9.5 (60.0) | 27.83 | 65.2 |
| 16k/1k c=8 | TP=2 ours | 1149 (5170) | 18.3 (116.1) | 14.26 | 33.1 |
| 16k/1k c=8 | TP=2 dense | 1135 (5211) | 17.6 (112.0) | 14.44 | 36.0 |
| 16k/1k c=8 | TP=4 ours | 1528 (4795) | 27.0 (167.8) | 10.72 | 21.0 |
| 16k/1k c=8 | TP=4 dense | 1569 (4838) | 27.0 (168.6) | 10.44 | 21.0 |

Before tuning (the profile's 1860 rows, none for these shapes):

UN| cell | arm | prefill tok/s | decode tok/s | TTFT s | ITL ms (median) |
| --- | --- | --- | --- | --- | --- |
| 1k/512 c=1 | TP=1 ours | 1932 (1910) | 78.4 (72.6) | 0.53 | 12.8 |
| 1k/512 c=1 | TP=1 dense | 1961 (1941) | 38.1 (36.7) | 0.52 | 26.3 |
| 1k/512 c=1 | TP=2 ours | 3436 (3399) | 102.7 (97.1) | 0.30 | 9.7 |
| 1k/512 c=1 | TP=2 dense | 3418 (3374) | 61.4 (59.4) | 0.30 | 16.3 |
| 1k/512 c=1 | TP=4 ours | 6541 (6529) | 113.9 (110.2) | 0.16 | 8.8 |
| 1k/512 c=1 | TP=4 dense | 6686 (6631) | 93.3 (90.9) | 0.15 | 10.7 |
| 1k/512 c=8 | TP=1 ours | 410 (2537) | 35.4 (242.0) | 2.50 | 26.8 |
| 1k/512 c=8 | TP=1 dense | 416 (2570) | 27.8 (196.2) | 2.46 | 34.6 |
| 1k/512 c=8 | TP=2 ours | 759 (4639) | 55.7 (389.2) | 1.35 | 17.2 |
| 1k/512 c=8 | TP=2 dense | 788 (5064) | 46.4 (332.3) | 1.30 | 20.7 |
| 1k/512 c=8 | TP=4 ours | 1492 (7518) | 71.3 (521.3) | 0.69 | 13.2 |
| 1k/512 c=8 | TP=4 dense | 1514 (7629) | 70.2 (514.1) | 0.68 | 13.5 |
| 16k/1k c=1 | TP=1 ours | 1407 (1402) | 57.8 (34.9) | 11.65 | 17.3 |
| 16k/1k c=1 | TP=1 dense | 1416 (1411) | 32.6 (23.9) | 11.57 | 30.7 |
| 16k/1k c=1 | TP=2 ours | 2647 (2626) | 79.0 (53.5) | 6.19 | 12.7 |
| 16k/1k c=1 | TP=2 dense | 2647 (2633) | 52.3 (39.8) | 6.19 | 19.1 |
| 16k/1k c=1 | TP=4 ours | 4724 (4705) | 98.5 (73.9) | 3.47 | 10.2 |
| 16k/1k c=1 | TP=4 dense | 4756 (4729) | 82.4 (64.6) | 3.45 | 12.2 |
| 16k/1k c=8 | TP=1 ours | 583 (2668) | 10.1 (63.3) | 28.10 | 57.6 |
| 16k/1k c=8 | TP=1 dense | 589 (2697) | 9.5 (60.0) | 27.83 | 65.2 |
| 16k/1k c=8 | TP=2 ours | 1149 (5170) | 18.3 (116.1) | 14.26 | 33.1 |
| 16k/1k c=8 | TP=2 dense | 1135 (5211) | 17.6 (112.0) | 14.44 | 36.0 |
| 16k/1k c=8 | TP=4 ours | 1528 (4795) | 27.0 (167.8) | 10.72 | 21.0 |
| 16k/1k c=8 | TP=4 dense | 1569 (4838) | 27.0 (168.6) | 10.44 | 21.0 |

Tuned against untuned, per request: prefill +13% (ours) / +11% (dense) at
1k/512 c=1 TP=1 (TTFT 0.60 -> 0.53 s), +105% / +98% at TP=2 (TTFT 0.61 -> 0.30
s), +4% at TP=4; 16k/1k prefill and decode move by less than the run-to-run
spread (the TP=1/TP=2 tuned arms ran while other benches used the box's other
GPUs). Decode is unchanged (the decode path is the GEMV, not rocBLAS).

Ours against dense (tuned): decode per request 2.06x at TP=1 (1k/512 c=1),
1.67x at TP=2, 1.22x at TP=4; at c=8 1.27x / 1.20x / 1.02x; prefill within 4%
of dense everywhere (the same rocBLAS GEMM after the dequant). Every Qwen3.5-4B
arm passed the greedy, perplexity and concurrency probes with no fault markers.

Qwen2.5-0.5B-Instruct, TP=1, tuned rows, against its dense fp16 export and the
existing AWQ 4-bit g128 checkpoint (the AWQ arm uses the fork's RDNA2 W4A16
kernel). Head size 64 is not supported by RDNA_ATTN, so these arms run with
`VLLM_USE_RDNA2_FA=0` (Triton attention), which dominates the 16k cells:

| cell | arm | prefill tok/s | decode tok/s | TTFT s | ITL ms (median) |
| --- | --- | --- | --- | --- | --- |
| 1k/512 c=1 | ours 3.0 bpw | 13505 (13032) | 176.7 (172.5) | 0.08 | 5.7 |
| 1k/512 c=1 | dense fp16 | 13784 (13387) | 163.7 (160.2) | 0.07 | 6.1 |
| 1k/512 c=1 | AWQ 4-bit g128 | 14139 (13572) | 114.8 (113.1) | 0.07 | 8.7 |
| 1k/512 c=8 | ours 3.0 bpw | 3151 (14754) | 139.7 (1027.6) | 0.32 | 6.8 |
| 1k/512 c=8 | dense fp16 | 3002 (14600) | 149.5 (1088.8) | 0.34 | 6.3 |
| 1k/512 c=8 | AWQ 4-bit g128 | 3133 (15604) | 109.2 (817.4) | 0.33 | 8.8 |
| 16k/1k c=1 | ours 3.0 bpw | 3045 (3035) | 25.4 (22.4) | 5.38 | 39.4 |
| 16k/1k c=1 | dense fp16 | 3053 (3045) | 25.1 (22.2) | 5.37 | 39.8 |
| 16k/1k c=1 | AWQ 4-bit g128 | 3067 (3058) | 23.6 (21.0) | 5.34 | 42.4 |
| 16k/1k c=8 | ours 3.0 bpw | 934 (4201) | 15.7 (98.2) | 17.54 | 40.7 |
| 16k/1k c=8 | dense fp16 | 947 (4343) | 15.7 (98.8) | 17.30 | 40.0 |
| 16k/1k c=8 | AWQ 4-bit g128 | 942 (4258) | 15.3 (96.3) | 17.39 | 42.7 |

The 0.5B model fails the exact-answer greedy probe in its 3.0 bpw and dense
forms (model quality; AWQ passes). The tuned dense arm failed one of 48
concurrency checks (one prompt diverged at token 15 at c=8 with a 2.3-nat
margin; the untuned dense arm and all other arms passed); not investigated.

## Checks: held-out perplexity

`tools/rdna/rdna_hint_flex/acc_long.sh` / `acc_tp.sh`: 8 x 2048 tokens of
`study_text_all.txt` after token 800000 (the tools/hi2 eval protocol; the
unquantized ppl matches the spec: 10.49 / 10.87 / 7.067), eager unless noted:

| model | unquantized | dense export | ours F16 | ours I8 (`i8_all`) | I8 / F16 | mean / max abs log-prob gap I8 vs F16 |
| --- | --- | --- | --- | --- | --- | --- |
| Qwen2.5-0.5B (3.0 bpw) | 10.4932 | 12.8013 | 12.8018 | 12.8097 | 1.00061 | 0.0128 / 0.267 |
| Qwen3.5-0.8B (3.0 bpw) | 10.8736 | 13.1276 | 13.1283 | 13.1327 | 1.00033 | 0.0114 / 0.279 |
| Qwen3.5-4B (3.0 bpw) | 7.0671 | 7.7916 | 7.7911 | 7.8001 | 1.00115 | 0.0095 / 5.46 |

AWQ 4-bit g128 Qwen2.5-0.5B: 12.0334 (4.25 bpw against our 3.0).

Tensor parallelism (Qwen3.5-4B, F16, full cudagraphs): ppl 7.7913 (TP=1),
7.7915 (TP=2), 7.7912 (TP=4); log-prob gap to TP=1 mean 0.0025 (TP=2) and
0.0026 (TP=4), the same as TP=1 graphs against TP=1 eager (0.0025).

## Routed experts: candidates on the box

No small unquantized MoE model is present (nothing downloaded). Found:
Qwen3.6-35B-A3B only as GPTQ-Int4 / MLX (no bf16 source); Qwen3.8-Flash-Next
only quantized (AWQ, AutoRound, MLX, EXL3); DeepSeek-V4-Flash fp8 (89-108 GB);
`Nemotron-derived-fp16-890ab464-20261021` (fp16, 65.8 GB, 128 experts top-6,
per-expert 2-D `backbone.layers.N.mixer.experts.E.{up,down}_proj.weight`), whose
experts are ungated (relu^2, no gate projection), which `RdnaHintFlexMoEMethod`
does not implement (it assumes SiLU-gated gate/up). A suitable small source would
be `allenai/OLMoE-1B-7B-0924` (bf16, about 14 GB, 64 experts top-8, per-expert
2-D gate/up/down, SiLU) or `Qwen/Qwen1.5-MoE-A2.7B` (about 29 GB, with a shared
expert).

## Not done

* A prefill kernel faster than dequant + rocBLAS (the fused kernel is opt-in;
  see above). I8 prefill (only the GEMV path exists, for accuracy).
* Routed experts were not run end to end (no suitable checkpoint); the forward is
  a per-pair GEMV (prefill re-reads each expert per token).
* The tuned dense Qwen2.5-0.5B concurrency failure was not investigated.

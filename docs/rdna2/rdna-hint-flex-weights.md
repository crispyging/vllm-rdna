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
**157 passed** on a V620. The dequant kernel and the I8 activation quantization
are bit-identical to the reference; `rdna_hint_flex_linear_rdna2` matches
`reference_linear` (float64) within 2e-3 relative error for M = 1..37 (F16),
the I8 GEMV matches the integer-exact I8 reference within 1e-3; the synthetic
4-expert MoE matches the dense reference within 1e-2.

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

Kernel timing (`tools/rdna/rdna_hint_flex/bench_kernel.py`, one V620, CUDA
graph of 50 calls; random tensors with 30% INT4 rows and mixed blocks, so the
bytes are representative but not a real checkpoint's), microseconds per call
including the activation transform:

| shape (R x K) | F16 M=1 | I8 M=1 | F16 M=3 | F16 M=8 | F16 M=32 |
| --- | --- | --- | --- | --- | --- |
| gate_up 18432 x 2560 | 40.1 (475 GB/s) | 33.9 | 71.5 | 138.5 | 535.9 |
| down 2560 x 9216 | 28.1 (338 GB/s) | 18.5 | 41.5 | 74.8 | 294.3 |
| in_proj_qkvz 12288 x 2560 | 30.5 (415 GB/s) | 28.6 | 47.4 | 93.5 | 366.9 |
| out_proj 2560 x 4096 | 15.6 (273 GB/s) | 12.6 | 27.9 | 36.8 | 127.0 |

The V620's peak is 512 GB/s; gate_up is at 93% of it. Large-K tensors stay
lower because every row re-reads the whole activation (int8 activations halve
that traffic, which is why I8 is faster there).

Serving (`tools/rdna/rdna_hint_flex/bench_serve.sh`: vLLM OpenAI server,
`RECIPE=full` = FULL_AND_PIECEWISE cudagraphs, TP=1 on one V620, `vllm bench
serve` random dataset with `--ignore-eos`, n = 4 / 32 / 4 / 16 requests).
Qwen3.5-4B, 3.0 bpw checkpoint against its dense fp16 export. Prefill and
decode are per request (aggregate in brackets):

| cell | arm | prefill tok/s | decode tok/s | TTFT s | ITL ms (median) |
| --- | --- | --- | --- | --- | --- |
| 1k/512 c=1 | rdna_hint_flex F16 | 1712 (1689) | **79.4** (72.8) | 0.60 | **12.6** |
| 1k/512 c=1 | rdna_hint_flex I8 | 1706 (1685) | 76.8 (70.6) | 0.60 | 13.0 |
| 1k/512 c=1 | dense fp16 | 1765 (1749) | 38.3 (36.8) | 0.58 | 26.1 |
| 1k/512 c=8 | rdna_hint_flex F16 | 404 (2502) | **36.1** (245.2) | 2.54 | **26.2** |
| 1k/512 c=8 | rdna_hint_flex I8 | 394 (2342) | 35.6 (241.2) | 2.60 | 26.7 |
| 1k/512 c=8 | dense fp16 | 416 (2574) | 28.0 (197.8) | 2.46 | 34.2 |
| 16k/1k c=1 | rdna_hint_flex F16 | 1444 (1435) | **58.9** (35.7) | 11.35 | **17.0** |
| 16k/1k c=1 | rdna_hint_flex I8 | 1426 (1420) | 57.6 (35.0) | 11.49 | 17.4 |
| 16k/1k c=1 | dense fp16 | 1460 (1453) | 33.0 (24.2) | 11.22 | 30.4 |
| 16k/1k c=8 | rdna_hint_flex F16 | 598 (2740) | **10.4** (65.0) | 27.38 | **55.9** |
| 16k/1k c=8 | rdna_hint_flex I8 | 592 (2685) | 10.4 (64.7) | 27.70 | 56.3 |
| 16k/1k c=8 | dense fp16 | 593 (2657) | 9.8 (61.6) | 27.64 | 63.7 |

Decode per request is 2.07x the dense model at c=1 (1k) and 1.79x at 16k;
prefill is within 3% of dense (same rocBLAS GEMM after the dequant). All arms
passed the greedy probe, the perplexity probe and the concurrency probe (48 of
48 greedy outputs identical between c=8, churned order and solo), with no fault
markers. I8 is not faster in serving despite the kernel numbers: its two extra
launches and buffers per layer eat the saving at these sizes. (The I8 arm
ran while another job booted a server on other GPUs of the box, so CPU load may
have cost it a little; the F16 and dense arms ran on a quiet box.)

No W4A16/GPTQ checkpoint of Qwen3.5-4B exists on the box, so that comparison
was not run.

## Not done

* A real prefill kernel: prefill above M = 48 is dequant into scratch plus the
  rocBLAS GEMM, so prefill throughput equals the dense model's, not better.
* I8 for prefill (only the GEMV is I8) and for routed experts.
* Routed experts were not run end to end (no checkpoint); the method is unit
  tested only, and its forward is a per-pair GEMV (prefill re-reads each
  expert per token).
* Tensor parallelism above 2 GPUs was not run.

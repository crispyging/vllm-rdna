# HI2 (HadamardINT2-FP16 v3.2) weights on RDNA2

`quant_method: hadamard_int2` loads checkpoints written by the separate
`tools/hi2` quantizer (spec: `docs/rdna2/hadamard-int2-v3.2-spec.md` on the
`feat/hadamard-int2-weights` branch). This page covers the vLLM side: the
loader, the gfx1030 kernels, and how they were checked.

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

Which modules are HI2 is read from the checkpoint index: a module stored as
`.hi2_*` is HI2, one stored as `.weight` stays dense (vision tower, embeddings,
head, GDN `conv1d`, ...). No module list is needed in `quantization_config`.

## Files

| file | content |
| --- | --- |
| `csrc/rocm/hi2_rdna2.cu` | kernels and host wrappers (gfx1030 build only) |
| `vllm/model_executor/layers/quantization/hi2.py` | `HadamardInt2Config`, `HadamardInt2LinearMethod` |
| `vllm/model_executor/layers/quantization/utils/hi2_utils.py` | host helpers: validate, slice rows/columns, concatenate, pure-PyTorch dequantize and activation transform |
| `tests/kernels/quantization/test_hi2_rdna2.py` | reference port of `tools/hi2/hi2/format.py`, CPU and gfx1030 tests |
| `tools/rdna/hi2/` | in-tree env, test runner, end-to-end check, kernel timing |

Outside the new files: the op declarations (`csrc/rocm/ops.h`), their
registration inside the `VLLM_ROCM_GFX1030` block of
`csrc/rocm/torch_bindings.cpp`, the gfx1030 source list in `CMakeLists.txt`,
wrappers and fake kernels in `vllm/_custom_ops.py`, the method name in
`quantization/__init__.py` and in `RocmPlatform.supported_quantization`.
Nothing in `fa_rdna2` or the attention backends changed.

## Kernels (F16 mode, spec 6.1)

* `hi2_act_transform_rdna2`: `x' = H16(x / s)` on the flagged groups, fp32
  math, one rounding to fp16. One thread per (token, group).
* `hi2_gemv_rdna2` (decode): one wave32 per output row, lanes stride over
  the row's words, so the weight stream is coalesced. A word never crosses a
  block, so each word decodes one `lo`/`step` pair; codes become fp16 through
  `half(0x6400 | c) - 1024` (exact up to 1023, so INT8 rows work too), then one
  `v_pk_fma_f16` and `v_dot2_f32_f16` against `x'`, fp32 accumulation and a
  wave reduction. The row's width is wave-uniform, so mixed INT2/INT4/INT8
  rows do not diverge. Up to 8 tokens per launch row tile.
* `hi2_dequant_rdna2`: stored-basis fp16 `W [R, K]`.
* `hi2_linear_rdna2`: transform, then GEMV for `M <= 8`, otherwise dequant
  into a scratch buffer and a rocBLAS GEMM (the simple prefill path). The
  choice happens inside the op at execution time, so torch.compile cannot fix
  it to the warmup shape.

All launches use the current torch stream and `C10_CUDA_KERNEL_LAUNCH_CHECK`.
The kernels trust the offsets; the loader validates them on the host first.

## Loader

`create_weights` registers placeholder parameters per field;
the loader collects each checkpoint tensor per output shard (`None`, an int,
a tuple such as Qwen3.5's `in_proj_qkv -> (0, 1, 2)`, or `q`/`k`/`v`).
`process_weights_after_loading` splits fused checkpoint tensors into the
layer's partitions, slices rows for column-parallel layers (with the GQA kv
replication rule) and 64-aligned columns for row-parallel layers, validates,
and concatenates consecutive partitions that share `col_scale` and `had` into
one tensor, so fused q/k/v and gate/up are one kernel call. In the tool's
checkpoints q/k/v and gate/up always share them.

Where the kernels are not built (not gfx10x, or not ROCm), the weights are
dequantized once to fp16 and run through `F.linear` after the PyTorch
activation transform: correct, without the memory saving.

Activations must be fp16 (`--dtype float16`): RDNA2 has no bf16 dot.

## Checks

Unit tests (`bash tools/rdna/hi2/run_tests.sh <hip_device>`): 133 pass on a
V620. The dequant kernel is bit-identical to the reference decode for mixed
INT2/INT4/INT8 rows and blocks 16/32/64; `hi2_linear_rdna2` matches
`reference_linear` (float64) within 2e-3 relative error for M = 1..37 with
and without smoothing and rotation; the GEMV and GEMM paths agree.

End to end (`tools/rdna/hi2/e2e_check.py`, HI2 checkpoint against its
`hi2 export-dense`, greedy, fp16, one V620):

| model (3.0 bpw, sens) | mode | teacher-forced ppl HI2 / dense | greedy tokens identical | weights in GPU memory HI2 / dense |
| --- | --- | --- | --- | --- |
| Qwen2.5-0.5B-Instruct | eager | 24.454 / 24.471 | 4 of 4 prompts, 48/48 | 0.39 / 0.93 GiB |
| Qwen2.5-0.5B-Instruct | compile + FULL_AND_PIECEWISE | 24.457 / 24.471 | 4 of 4, 48/48 | 0.39 GiB |
| Qwen3.5-0.8B | eager | 27.954 / 27.947 | 4 of 4, 48/48 | 0.78 / 1.53 GiB |
| Qwen3.5-4B | compile + FULL_AND_PIECEWISE | 10.410 / 10.405 | 4 of 4, 48/48 | 3.42 / 8.81 GiB |

Per-token log-probability gap to the dense export: mean 0.003 to 0.005, max
0.034. The remaining difference is the fp16 rounding of the transformed
activation and of the dense export's original-basis weights.

Serving throughput has not been measured. A kernel-only timing is in
`tools/rdna/hi2/bench_kernel.py` (decode GEMV at M=1 streams HI2 bytes at
300 to 470 GB/s on a V620; the prefill path is untuned).

## Not done

* I8 compute mode (spec 6.2, `v_dot4_i32_i8` with int8 activations).
* HI2 MoE experts (only `LinearBase` layers are handled).
* Tensor parallelism is implemented and unit-tested on the host (row and
  column slicing), but not run on GPUs.
* Kernel tuning: the GEMV time grows about linearly with M (each wave handles
  all tokens), and the prefill path is dequant + GEMM.

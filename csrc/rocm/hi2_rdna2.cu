// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project
//
// HadamardINT2-FP16 (HI2, spec v3.2) weight kernels for RDNA2 (gfx1030),
// F16 compute mode (spec 6.1), fp16 only.
//
// Storage (one tensor W[R, K], spec 2.4; all buffers flat, per-row ragged):
//   codes      int32 [sum_r K*nbits_r/32]  codes LSB-first, word aligned
//                                          (16 INT2, 8 INT4 or 4 INT8 per word)
//   code_off   int32 [R+1]   first word of each row
//   blk_params uint8 [2*sum_r K/B_r]  (lo_u8, step_u8) per block
//   param_off  int32 [R+1]   first block of each row
//   row_consts fp16  [R, 3]  lo_min, lo_scale, step_scale
//   row_nbits  uint8 [R]     2, 4 or 8
//   row_block  uint8 [R]     16, 32 or 64
//   had        uint8 [K/16]  order-16 Walsh-Hadamard flag per column group
//   col_scale  fp16  [K]     optional smoothing vector s (numel 0 = none)
//
// Decode (spec 2.2):
//   lo   = fp16(fma_f32(lo_u8, lo_scale, lo_min))
//   step = fp16(step_u8 * step_scale)
//   w    = fp16(lo + code * step)       one rounding: v_fma_f16
// The stored basis is H(W * s); the runtime activation is x' = H(x / s) on
// the flagged groups (spec 2.3), so y = x' . W_stored^T.
//
// Kernels:
//   hi2_act_transform : x' = H16(x / s), fp32 math, one fp16 rounding.
//   hi2_gemv          : decode path (small M). One wave32 per output row,
//                       lanes stride over the row's words (coalesced), code ->
//                       fp16 by the 0x6400 magic, v_pk_fma_f16 with the block
//                       lo/step, v_dot2_f32_f16 against x', wave reduction.
//   hi2_dequant       : stored-basis fp16 W [R, K]; the prefill path is
//                       dequant + rocBLAS GEMM (correct and simple, not tuned).
//   hi2_linear_rdna2  : composite (transform + gemv or dequant+GEMM), chosen
//                       at execution time from M so torch.compile cannot bake
//                       the warmup shape into the dispatch.
//
// All launches go to the current torch stream and are followed by
// C10_CUDA_KERNEL_LAUNCH_CHECK.

#include <algorithm>
#include <cstdint>

#include <torch/all.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAException.h>
#include <ATen/cuda/CUDAContext.h>

#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>

#if defined(__HIPCC__) && defined(__gfx1030__)
  #define __HIP__RDNA2__
#endif

namespace vllm {
namespace hi2 {

constexpr int kGroup = 16;        // Hadamard group (spec 2.3)
constexpr int kWave = 32;         // RDNA2 wave32
constexpr int kRowsPerBlock = 8;  // 8 waves x 32 lanes = 256 threads
constexpr int kThreads = kRowsPerBlock * kWave;

#if defined(__HIP__RDNA2__) || !defined(__HIP_DEVICE_COMPILE__)

__device__ __forceinline__ void decode_params(uint32_t lo_u8, uint32_t step_u8,
                                              float lo_min, float lo_scale,
                                              float step_scale, half& lo,
                                              half& step) {
  lo = __float2half_rn(__fmaf_rn(static_cast<float>(lo_u8), lo_scale, lo_min));
  // u8 * fp16 has at most 19 significant bits: exact in fp32, so this is one
  // rounding (to fp16), as in the spec.
  step = __float2half_rn(static_cast<float>(step_u8) * step_scale);
}

// Two codes (< 1024) -> half2 exactly: half(0x6400 | c) = 1024 + c.
__device__ __forceinline__ half2 codes_to_half2(uint32_t c0, uint32_t c1) {
  union {
    uint32_t u;
    half2 h;
  } v;
  v.u = (0x6400u | c0) | ((0x6400u | c1) << 16);
  return __hsub2(v.h, __float2half2_rn(1024.0f));
}

__device__ __forceinline__ half code_to_half(uint32_t c) {
  union {
    uint16_t u;
    half h;
  } v;
  v.u = static_cast<uint16_t>(0x6400u | c);
  return __hsub(v.h, __float2half(1024.0f));
}

__device__ __forceinline__ int log2_block(int b) {
  return b == 16 ? 4 : (b == 32 ? 5 : 6);
}

// ---------------------------------------------------------------------------
// Activation transform: y[m, g*16 + i] = (H16 (x / s))_i on flagged groups.
// One thread per (row, group).
// ---------------------------------------------------------------------------
__global__ void hi2_act_transform_kernel(const half* __restrict__ x,
                                         const int64_t ldx,
                                         const half* __restrict__ col_scale,
                                         const uint8_t* __restrict__ had,
                                         half* __restrict__ y, const int M,
                                         const int K) {
  const int groups = K / kGroup;
  const int64_t g = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (g >= static_cast<int64_t>(M) * groups) return;
  const int m = static_cast<int>(g / groups);
  const int gi = static_cast<int>(g % groups);
  const half* xp = x + m * ldx + gi * kGroup;
  float v[kGroup];
  #pragma unroll
  for (int i = 0; i < kGroup; ++i) v[i] = __half2float(xp[i]);
  if (col_scale != nullptr) {
  #pragma unroll
    for (int i = 0; i < kGroup; ++i)
      v[i] = v[i] / __half2float(col_scale[gi * kGroup + i]);
  }
  if (had[gi]) {
    // In-place butterflies give the Sylvester (natural) order, i.e. the
    // symmetric matrix built by [[h, h], [h, -h]] in the reference.
  #pragma unroll
    for (int len = 1; len < kGroup; len <<= 1) {
  #pragma unroll
      for (int i = 0; i < kGroup; ++i) {
        if ((i & len) == 0) {
          const float a = v[i], b = v[i + len];
          v[i] = a + b;
          v[i + len] = a - b;
        }
      }
    }
  #pragma unroll
    for (int i = 0; i < kGroup; ++i) v[i] *= 0.25f;  // 1/sqrt(16)
  }
  half* yp = y + static_cast<int64_t>(m) * K + gi * kGroup;
  #pragma unroll
  for (int i = 0; i < kGroup; ++i) yp[i] = __float2half_rn(v[i]);
}

// ---------------------------------------------------------------------------
// GEMV: one wave per output row. NB = code width of the row.
// ---------------------------------------------------------------------------
template <int MT, int NB>
__device__ __forceinline__ void row_dot(
    const half* __restrict__ x, const int K, const int m0, const int M,
    const uint32_t* __restrict__ words, const uint8_t* __restrict__ params,
    const int lb, const float lo_min, const float lo_scale,
    const float step_scale, const int lane, float (&acc)[MT]) {
  constexpr int CPW = 32 / NB;  // codes per word; a word never crosses a block
  constexpr uint32_t MASK = (1u << NB) - 1u;
  const int nwords = K / CPW;
  for (int w = lane; w < nwords; w += kWave) {
    const uint32_t q = words[w];
    const int k0 = w * CPW;
    const int blk = k0 >> lb;
    half lo, step;
    decode_params(params[2 * blk], params[2 * blk + 1], lo_min, lo_scale,
                  step_scale, lo, step);
    const half2 lo2 = __half2half2(lo);
    const half2 st2 = __half2half2(step);
    half2 wv[CPW / 2];
  #pragma unroll
    for (int i = 0; i < CPW / 2; ++i) {
      const uint32_t c0 = (q >> (NB * (2 * i))) & MASK;
      const uint32_t c1 = (q >> (NB * (2 * i + 1))) & MASK;
      wv[i] = __hfma2(codes_to_half2(c0, c1), st2, lo2);
    }
  #pragma unroll
    for (int m = 0; m < MT; ++m) {
      if (m0 + m < M) {
        const half2* xp = reinterpret_cast<const half2*>(
            x + static_cast<int64_t>(m0 + m) * K + k0);
        float a = acc[m];
  #pragma unroll
        for (int i = 0; i < CPW / 2; ++i)
          a = __builtin_amdgcn_fdot2(wv[i], xp[i], a, /*clamp=*/false);
        acc[m] = a;
      }
    }
  }
}

template <int MT>
__global__ void __launch_bounds__(kThreads)
    hi2_gemv_kernel(const half* __restrict__ x, const int M, const int K,
                    const int32_t* __restrict__ codes,
                    const int32_t* __restrict__ code_off,
                    const uint8_t* __restrict__ blk_params,
                    const int32_t* __restrict__ param_off,
                    const half* __restrict__ row_consts,
                    const uint8_t* __restrict__ row_nbits,
                    const uint8_t* __restrict__ row_block,
                    half* __restrict__ out, const int64_t ldo, const int R) {
  const int wave = threadIdx.x / kWave;
  const int lane = threadIdx.x % kWave;
  const int r = blockIdx.x * kRowsPerBlock + wave;
  if (r >= R) return;  // wave-uniform; no barriers below
  const int m0 = blockIdx.y * MT;
  const int nb = row_nbits[r];
  const int lb = log2_block(row_block[r]);
  const float lo_min = __half2float(row_consts[3 * r + 0]);
  const float lo_scale = __half2float(row_consts[3 * r + 1]);
  const float step_scale = __half2float(row_consts[3 * r + 2]);
  const uint32_t* words =
      reinterpret_cast<const uint32_t*>(codes) + code_off[r];
  const uint8_t* params = blk_params + 2 * static_cast<int64_t>(param_off[r]);

  float acc[MT];
  #pragma unroll
  for (int m = 0; m < MT; ++m) acc[m] = 0.0f;
  switch (nb) {
    case 2:
      row_dot<MT, 2>(x, K, m0, M, words, params, lb, lo_min, lo_scale,
                     step_scale, lane, acc);
      break;
    case 4:
      row_dot<MT, 4>(x, K, m0, M, words, params, lb, lo_min, lo_scale,
                     step_scale, lane, acc);
      break;
    default:
      row_dot<MT, 8>(x, K, m0, M, words, params, lb, lo_min, lo_scale,
                     step_scale, lane, acc);
      break;
  }
  #pragma unroll
  for (int m = 0; m < MT; ++m) {
    float v = acc[m];
  #pragma unroll
    for (int off = kWave / 2; off > 0; off >>= 1)
      v += __shfl_xor(v, off, kWave);
    if (lane == 0 && m0 + m < M)
      out[static_cast<int64_t>(m0 + m) * ldo + r] = __float2half_rn(v);
  }
}

// ---------------------------------------------------------------------------
// Dequant: W_stored[r, :] in fp16. grid.x = rows, grid.y strides the words.
// ---------------------------------------------------------------------------
__global__ void __launch_bounds__(kThreads) hi2_dequant_kernel(
    const int32_t* __restrict__ codes, const int32_t* __restrict__ code_off,
    const uint8_t* __restrict__ blk_params,
    const int32_t* __restrict__ param_off, const half* __restrict__ row_consts,
    const uint8_t* __restrict__ row_nbits,
    const uint8_t* __restrict__ row_block, half* __restrict__ w, const int K) {
  const int r = blockIdx.x;
  const int nb = row_nbits[r];
  const int lb = log2_block(row_block[r]);
  const int cpw = 32 / nb;
  const uint32_t mask = (1u << nb) - 1u;
  const float lo_min = __half2float(row_consts[3 * r + 0]);
  const float lo_scale = __half2float(row_consts[3 * r + 1]);
  const float step_scale = __half2float(row_consts[3 * r + 2]);
  const uint32_t* words =
      reinterpret_cast<const uint32_t*>(codes) + code_off[r];
  const uint8_t* params = blk_params + 2 * static_cast<int64_t>(param_off[r]);
  half* wr = w + static_cast<int64_t>(r) * K;
  const int nwords = K / cpw;
  for (int wi = blockIdx.y * blockDim.x + threadIdx.x; wi < nwords;
       wi += gridDim.y * blockDim.x) {
    const uint32_t q = words[wi];
    const int k0 = wi * cpw;
    const int blk = k0 >> lb;
    half lo, step;
    decode_params(params[2 * blk], params[2 * blk + 1], lo_min, lo_scale,
                  step_scale, lo, step);
    for (int i = 0; i < cpw; ++i) {
      const uint32_t c = (q >> (nb * i)) & mask;
      wr[k0 + i] = __hfma(code_to_half(c), step, lo);
    }
  }
}

#else  // non-RDNA2 device pass: empty kernels for symbol parity.

__global__ void hi2_act_transform_kernel(const half*, const int64_t,
                                         const half*, const uint8_t*, half*,
                                         const int, const int) {}
template <int MT>
__global__ void hi2_gemv_kernel(const half*, const int, const int,
                                const int32_t*, const int32_t*, const uint8_t*,
                                const int32_t*, const half*, const uint8_t*,
                                const uint8_t*, half*, const int64_t,
                                const int) {}
__global__ void hi2_dequant_kernel(const int32_t*, const int32_t*,
                                   const uint8_t*, const int32_t*, const half*,
                                   const uint8_t*, const uint8_t*, half*,
                                   const int) {}

#endif  // __HIP__RDNA2__ || !__HIP_DEVICE_COMPILE__

}  // namespace hi2
}  // namespace vllm

// ---------------------------------------------------------------------------
// Host side.
// ---------------------------------------------------------------------------
namespace {

struct Hi2Weight {
  const torch::Tensor& codes;
  const torch::Tensor& code_off;
  const torch::Tensor& blk_params;
  const torch::Tensor& param_off;
  const torch::Tensor& row_consts;
  const torch::Tensor& row_nbits;
  const torch::Tensor& row_block;
};

void check_dev(const torch::Tensor& t, const char* name, at::ScalarType dt) {
  TORCH_CHECK(t.is_cuda(), "hi2: ", name, " must be a CUDA/HIP tensor");
  TORCH_CHECK(t.scalar_type() == dt, "hi2: ", name, " must be ", dt, ", got ",
              t.scalar_type());
  TORCH_CHECK(t.is_contiguous(), "hi2: ", name, " must be contiguous");
}

// Shape/dtype contract. The per-row offsets are validated on the host when
// the checkpoint is loaded (vllm/model_executor/layers/quantization/hi2.py);
// reading them back here would sync the stream.
int64_t check_weight(const Hi2Weight& w) {
  check_dev(w.codes, "codes", torch::kInt32);
  check_dev(w.code_off, "code_off", torch::kInt32);
  check_dev(w.blk_params, "blk_params", torch::kUInt8);
  check_dev(w.param_off, "param_off", torch::kInt32);
  check_dev(w.row_consts, "row_consts", torch::kHalf);
  check_dev(w.row_nbits, "row_nbits", torch::kUInt8);
  check_dev(w.row_block, "row_block", torch::kUInt8);
  const int64_t R = w.row_nbits.numel();
  TORCH_CHECK(w.row_block.numel() == R, "hi2: row_block must have R entries");
  TORCH_CHECK(w.code_off.numel() == R + 1 && w.param_off.numel() == R + 1,
              "hi2: code_off/param_off must have R+1 entries");
  TORCH_CHECK(w.row_consts.numel() == 3 * R, "hi2: row_consts must be [R, 3]");
  TORCH_CHECK(R < (int64_t{1} << 31) / vllm::hi2::kRowsPerBlock,
              "hi2: too many rows");
  return R;
}

void check_act(const torch::Tensor& x, int64_t K) {
  TORCH_CHECK(x.is_cuda() && x.scalar_type() == torch::kHalf,
              "hi2: activations must be fp16 CUDA/HIP tensors (RDNA2 has no "
              "bf16 dot)");
  TORCH_CHECK(x.dim() == 2 && x.size(1) == K, "hi2: x must be [M, ", K, "]");
  TORCH_CHECK(x.stride(1) == 1, "hi2: x must be row-major");
  TORCH_CHECK(K % 64 == 0 && K > 0, "hi2: K must be a positive multiple of 64");
}

void launch_transform(const torch::Tensor& x, const torch::Tensor& col_scale,
                      const torch::Tensor& had, torch::Tensor& y,
                      cudaStream_t stream) {
  const int64_t M = x.size(0), K = x.size(1);
  const half* s = nullptr;
  if (col_scale.numel() > 0) {
    check_dev(col_scale, "col_scale", torch::kHalf);
    TORCH_CHECK(col_scale.numel() == K, "hi2: col_scale must have K entries");
    s = reinterpret_cast<const half*>(col_scale.data_ptr());
  }
  check_dev(had, "had", torch::kUInt8);
  TORCH_CHECK(had.numel() == K / vllm::hi2::kGroup,
              "hi2: had must have K/16 entries");
  const int64_t n = M * (K / vllm::hi2::kGroup);
  if (n == 0) return;
  const int threads = 256;
  const int64_t blocks = (n + threads - 1) / threads;
  vllm::hi2::hi2_act_transform_kernel<<<blocks, threads, 0, stream>>>(
      reinterpret_cast<const half*>(x.data_ptr()), x.stride(0), s,
      had.data_ptr<uint8_t>(), reinterpret_cast<half*>(y.data_ptr()),
      static_cast<int>(M), static_cast<int>(K));
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

template <int MT>
void launch_gemv_mt(const torch::Tensor& x, const Hi2Weight& w,
                    torch::Tensor& out, int64_t R, cudaStream_t stream) {
  const int64_t M = x.size(0), K = x.size(1);
  dim3 grid((R + vllm::hi2::kRowsPerBlock - 1) / vllm::hi2::kRowsPerBlock,
            (M + MT - 1) / MT);
  vllm::hi2::hi2_gemv_kernel<MT><<<grid, vllm::hi2::kThreads, 0, stream>>>(
      reinterpret_cast<const half*>(x.data_ptr()), static_cast<int>(M),
      static_cast<int>(K), w.codes.data_ptr<int32_t>(),
      w.code_off.data_ptr<int32_t>(), w.blk_params.data_ptr<uint8_t>(),
      w.param_off.data_ptr<int32_t>(),
      reinterpret_cast<const half*>(w.row_consts.data_ptr()),
      w.row_nbits.data_ptr<uint8_t>(), w.row_block.data_ptr<uint8_t>(),
      reinterpret_cast<half*>(out.data_ptr()), out.stride(0),
      static_cast<int>(R));
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

void launch_gemv(const torch::Tensor& x, const Hi2Weight& w, torch::Tensor& out,
                 int64_t R, cudaStream_t stream) {
  const int64_t M = x.size(0);
  if (M == 0 || R == 0) return;
  TORCH_CHECK(x.is_contiguous(), "hi2: gemv input must be contiguous");
  if (M == 1) {
    launch_gemv_mt<1>(x, w, out, R, stream);
  } else if (M == 2) {
    launch_gemv_mt<2>(x, w, out, R, stream);
  } else if (M <= 4) {
    launch_gemv_mt<4>(x, w, out, R, stream);
  } else {
    launch_gemv_mt<8>(x, w, out, R, stream);
  }
}

void launch_dequant(const Hi2Weight& w, torch::Tensor& out, int64_t R,
                    int64_t K, cudaStream_t stream) {
  if (R == 0) return;
  // Enough y-blocks that each thread decodes a handful of words.
  const int64_t max_words = K / 4;  // INT8 rows have the most words
  const int64_t ny = std::max<int64_t>(
      1, (max_words + vllm::hi2::kThreads - 1) / vllm::hi2::kThreads);
  dim3 grid(static_cast<unsigned>(R), static_cast<unsigned>(ny));
  vllm::hi2::hi2_dequant_kernel<<<grid, vllm::hi2::kThreads, 0, stream>>>(
      w.codes.data_ptr<int32_t>(), w.code_off.data_ptr<int32_t>(),
      w.blk_params.data_ptr<uint8_t>(), w.param_off.data_ptr<int32_t>(),
      reinterpret_cast<const half*>(w.row_consts.data_ptr()),
      w.row_nbits.data_ptr<uint8_t>(), w.row_block.data_ptr<uint8_t>(),
      reinterpret_cast<half*>(out.data_ptr()), static_cast<int>(K));
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

}  // namespace

// x [M, K] fp16 -> out [M, K] fp16 = H16(x / s) on flagged groups.
void hi2_act_transform_rdna2(torch::Tensor x, torch::Tensor col_scale,
                             torch::Tensor had, torch::Tensor out) {
  check_act(x, x.size(1));
  check_dev(out, "out", torch::kHalf);
  TORCH_CHECK(out.sizes() == x.sizes(), "hi2: out must match x");
  const at::cuda::OptionalCUDAGuard device_guard(device_of(x));
  auto stream = at::cuda::getCurrentCUDAStream();
  launch_transform(x, col_scale, had, out, stream);
}

// x [M, K] fp16 (already transformed) -> out [M, R] fp16 (written, not
// accumulated; out may be a row-strided view with unit column stride).
void hi2_gemv_rdna2(torch::Tensor x, torch::Tensor codes,
                    torch::Tensor code_off, torch::Tensor blk_params,
                    torch::Tensor param_off, torch::Tensor row_consts,
                    torch::Tensor row_nbits, torch::Tensor row_block,
                    torch::Tensor out) {
  const Hi2Weight w{codes,      code_off,  blk_params, param_off,
                    row_consts, row_nbits, row_block};
  const int64_t R = check_weight(w);
  check_act(x, x.size(1));
  TORCH_CHECK(out.is_cuda() && out.scalar_type() == torch::kHalf &&
                  out.dim() == 2 && out.size(0) == x.size(0) &&
                  out.size(1) == R && out.stride(1) == 1,
              "hi2: out must be fp16 [M, R] with unit column stride");
  const at::cuda::OptionalCUDAGuard device_guard(device_of(x));
  auto stream = at::cuda::getCurrentCUDAStream();
  launch_gemv(x.contiguous(), w, out, R, stream);
}

// Stored-basis reconstruction W [R, K] fp16.
void hi2_dequant_rdna2(torch::Tensor codes, torch::Tensor code_off,
                       torch::Tensor blk_params, torch::Tensor param_off,
                       torch::Tensor row_consts, torch::Tensor row_nbits,
                       torch::Tensor row_block, torch::Tensor out) {
  const Hi2Weight w{codes,      code_off,  blk_params, param_off,
                    row_consts, row_nbits, row_block};
  const int64_t R = check_weight(w);
  check_dev(out, "out", torch::kHalf);
  TORCH_CHECK(out.dim() == 2 && out.size(0) == R && out.size(1) % 64 == 0,
              "hi2: out must be [R, K] with K a multiple of 64");
  const at::cuda::OptionalCUDAGuard device_guard(device_of(out));
  auto stream = at::cuda::getCurrentCUDAStream();
  launch_dequant(w, out, R, out.size(1), stream);
}

// y = x . W^T for an HI2 tensor (transform included when `transform`).
// M <= decode_max_m: GEMV kernel; otherwise dequant + rocBLAS GEMM.
torch::Tensor hi2_linear_rdna2(torch::Tensor x, torch::Tensor col_scale,
                               torch::Tensor had, torch::Tensor codes,
                               torch::Tensor code_off, torch::Tensor blk_params,
                               torch::Tensor param_off,
                               torch::Tensor row_consts,
                               torch::Tensor row_nbits, torch::Tensor row_block,
                               bool transform, int64_t decode_max_m) {
  const Hi2Weight w{codes,      code_off,  blk_params, param_off,
                    row_consts, row_nbits, row_block};
  const int64_t R = check_weight(w);
  const int64_t K = x.size(1);
  check_act(x, K);
  const int64_t M = x.size(0);
  const at::cuda::OptionalCUDAGuard device_guard(device_of(x));
  auto stream = at::cuda::getCurrentCUDAStream();
  auto out = torch::empty({M, R}, x.options());
  if (M == 0 || R == 0) return out;

  torch::Tensor xt;
  if (transform) {
    xt = torch::empty({M, K}, x.options());
    launch_transform(x, col_scale, had, xt, stream);
  } else {
    xt = x.contiguous();
  }
  if (M <= decode_max_m) {
    launch_gemv(xt, w, out, R, stream);
  } else {
    auto wd = torch::empty({R, K}, x.options());
    launch_dequant(w, wd, R, K, stream);
    at::mm_out(out, xt, wd.t());
  }
  return out;
}

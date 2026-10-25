// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project
//
// rdna_hint_flex (HadamardINT2-FP16 spec v3.2) weight kernels for RDNA2
// (gfx1030), F16 compute mode (spec 6.1), fp16 only.
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
//   act_transform : x' = H16(x / s), fp32 math, one fp16 rounding.
//   gemv          : decode path (small M). One wave32 per output row,
//                       lanes stride over the row's words (coalesced), code ->
//                       fp16 by the 0x6400 magic, v_pk_fma_f16 with the block
//                       lo/step, v_dot2_f32_f16 against x', wave reduction.
//   dequant       : stored-basis fp16 W [R, K]; the prefill path is
//                       dequant + rocBLAS GEMM (correct and simple, not tuned).
//   rdna_hint_flex_linear_rdna2  : composite (transform + gemv or
//   dequant+GEMM), chosen
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
namespace rdna_hint_flex {

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
__global__ void act_transform_kernel(const half* __restrict__ x,
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
    gemv_kernel(const half* __restrict__ x, const int M, const int K,
                const int32_t* __restrict__ codes,
                const int32_t* __restrict__ code_off,
                const uint8_t* __restrict__ blk_params,
                const int32_t* __restrict__ param_off,
                const half* __restrict__ row_consts,
                const uint8_t* __restrict__ row_nbits,
                const uint8_t* __restrict__ row_block, half* __restrict__ out,
                const int64_t ldo, const int R) {
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
__global__ void __launch_bounds__(kThreads) dequant_kernel(
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

// ---------------------------------------------------------------------------
// I8 compute mode (spec 6.2).
//
// Activation: per token and 16-column group g of the transformed activation
//   sx_g = fp16(absmax / 127), xq = clamp(rint(x / sx_g), -127, 127),
//   sumq_g = sum xq. The transform and the quantization group are the same
//   16 columns, so one thread does both.
// Weights: per 16-column chunk of a row, S_g = sum code * xq through four
//   v_dot4_i32_i8; y += sx_g * (step * S_g + lo * sumq_g) in fp32.
// INT8 codes (0..255) do not fit a signed byte: they enter the dot as
//   code - 128 and S_g gets 128 * sumq_g back, exactly, in int32.
// ---------------------------------------------------------------------------
__global__ void act_quant_kernel(const half* __restrict__ x, const int64_t ldx,
                                 const half* __restrict__ col_scale,
                                 const uint8_t* __restrict__ had,
                                 int8_t* __restrict__ xq, half* __restrict__ sx,
                                 int32_t* __restrict__ sumq, const int M,
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
  if (had != nullptr && had[gi]) {
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
    for (int i = 0; i < kGroup; ++i) v[i] *= 0.25f;
  }
  float amax = 0.0f;
  #pragma unroll
  for (int i = 0; i < kGroup; ++i) amax = fmaxf(amax, fabsf(v[i]));
  const half sh = __float2half_rn(amax / 127.0f);
  const float sf = __half2float(sh);
  union {
    int8_t b[kGroup];
    uint4 u;
  } q;
  int sum = 0;
  #pragma unroll
  for (int i = 0; i < kGroup; ++i) {
    float r = sf > 0.0f ? rintf(v[i] / sf) : 0.0f;
    r = fminf(fmaxf(r, -127.0f), 127.0f);
    q.b[i] = static_cast<int8_t>(r);
    sum += q.b[i];
  }
  *reinterpret_cast<uint4*>(xq + static_cast<int64_t>(m) * K + gi * kGroup) =
      q.u;
  sx[static_cast<int64_t>(m) * groups + gi] = sh;
  sumq[static_cast<int64_t>(m) * groups + gi] = sum;
}

// Four consecutive codes as the bytes of one dword (code j in byte j).
__device__ __forceinline__ int spread2(uint32_t x) {
  return static_cast<int>((x & 0x03u) | ((x & 0x0Cu) << 6) |
                          ((x & 0x30u) << 12) | ((x & 0xC0u) << 18));
}
__device__ __forceinline__ int spread4(uint32_t x) {
  return static_cast<int>((x & 0x000Fu) | ((x & 0x00F0u) << 4) |
                          ((x & 0x0F00u) << 8) | ((x & 0xF000u) << 12));
}

template <int NB>
__device__ __forceinline__ void group_codes(const uint32_t* __restrict__ w,
                                            int (&d)[4]) {
  if constexpr (NB == 2) {
    const uint32_t q = w[0];
    d[0] = spread2(q);
    d[1] = spread2(q >> 8);
    d[2] = spread2(q >> 16);
    d[3] = spread2(q >> 24);
  } else if constexpr (NB == 4) {
    const uint2 q = *reinterpret_cast<const uint2*>(w);
    d[0] = spread4(q.x);
    d[1] = spread4(q.x >> 16);
    d[2] = spread4(q.y);
    d[3] = spread4(q.y >> 16);
  } else {
    const uint4 q = *reinterpret_cast<const uint4*>(w);
    d[0] = static_cast<int>(q.x ^ 0x80808080u);
    d[1] = static_cast<int>(q.y ^ 0x80808080u);
    d[2] = static_cast<int>(q.z ^ 0x80808080u);
    d[3] = static_cast<int>(q.w ^ 0x80808080u);
  }
}

template <int MT, int NB>
__device__ __forceinline__ void row_dot_i8(
    const int8_t* __restrict__ xq, const half* __restrict__ sx,
    const int32_t* __restrict__ sumq, const int K, const int m0, const int M,
    const uint32_t* __restrict__ words, const uint8_t* __restrict__ params,
    const int lb, const float lo_min, const float lo_scale,
    const float step_scale, const int lane, float (&acc)[MT]) {
  constexpr int WPG = NB / 2;  // words per 16-column group
  const int groups = K / kGroup;
  for (int g = lane; g < groups; g += kWave) {
    int d[4];
    group_codes<NB>(words + g * WPG, d);
    const int blk = (g * kGroup) >> lb;
    half lo, step;
    decode_params(params[2 * blk], params[2 * blk + 1], lo_min, lo_scale,
                  step_scale, lo, step);
    const float lof = __half2float(lo), stf = __half2float(step);
  #pragma unroll
    for (int m = 0; m < MT; ++m) {
      if (m0 + m < M) {
        const int64_t row = static_cast<int64_t>(m0 + m);
        const int4 xv =
            *reinterpret_cast<const int4*>(xq + row * K + g * kGroup);
        const int sq = sumq[row * groups + g];
        int S = __builtin_amdgcn_sdot4(d[0], xv.x, 0, false);
        S = __builtin_amdgcn_sdot4(d[1], xv.y, S, false);
        S = __builtin_amdgcn_sdot4(d[2], xv.z, S, false);
        S = __builtin_amdgcn_sdot4(d[3], xv.w, S, false);
        if constexpr (NB == 8) S += 128 * sq;
        const float sxf = __half2float(sx[row * groups + g]);
        acc[m] +=
            sxf * (stf * static_cast<float>(S) + lof * static_cast<float>(sq));
      }
    }
  }
}

template <int MT>
__global__ void __launch_bounds__(kThreads)
    gemv_i8_kernel(const int8_t* __restrict__ xq, const half* __restrict__ sx,
                   const int32_t* __restrict__ sumq, const int M, const int K,
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
  if (r >= R) return;
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
      row_dot_i8<MT, 2>(xq, sx, sumq, K, m0, M, words, params, lb, lo_min,
                        lo_scale, step_scale, lane, acc);
      break;
    case 4:
      row_dot_i8<MT, 4>(xq, sx, sumq, K, m0, M, words, params, lb, lo_min,
                        lo_scale, step_scale, lane, acc);
      break;
    default:
      row_dot_i8<MT, 8>(xq, sx, sumq, K, m0, M, words, params, lb, lo_min,
                        lo_scale, step_scale, lane, acc);
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
// Routed experts (F16 mode). Experts are stacked row-wise: expert e owns rows
// [e * n_out, (e + 1) * n_out) of one ragged tensor, and has its own
// smoothing vector and Hadamard flags (the quantizer calibrates each expert
// on the tokens routed to it). A (token, k) pair is p = t * top_k + k.
// ---------------------------------------------------------------------------
__device__ __forceinline__ void transform16(float (&v)[kGroup],
                                            const half* __restrict__ s,
                                            const bool rotate) {
  if (s != nullptr) {
  #pragma unroll
    for (int i = 0; i < kGroup; ++i) v[i] = v[i] / __half2float(s[i]);
  }
  if (rotate) {
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
    for (int i = 0; i < kGroup; ++i) v[i] *= 0.25f;
  }
}

// y[p] = transform_{expert[p]}(x[p / src_div]); rows of unrouted pairs
// (expert < 0) are zero.
__global__ void moe_act_transform_kernel(const half* __restrict__ x,
                                         const int64_t ldx, const int src_div,
                                         const int32_t* __restrict__ expert,
                                         const half* __restrict__ col_scale,
                                         const uint8_t* __restrict__ had,
                                         half* __restrict__ y, const int P,
                                         const int K) {
  const int groups = K / kGroup;
  const int64_t g = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (g >= static_cast<int64_t>(P) * groups) return;
  const int p = static_cast<int>(g / groups);
  const int gi = static_cast<int>(g % groups);
  const int e = expert[p];
  half* yp = y + static_cast<int64_t>(p) * K + gi * kGroup;
  if (e < 0) {
  #pragma unroll
    for (int i = 0; i < kGroup; ++i) yp[i] = __float2half(0.0f);
    return;
  }
  const half* xp = x + (p / src_div) * ldx + gi * kGroup;
  float v[kGroup];
  #pragma unroll
  for (int i = 0; i < kGroup; ++i) v[i] = __half2float(xp[i]);
  const half* s = col_scale != nullptr
                      ? col_scale + static_cast<int64_t>(e) * K + gi * kGroup
                      : nullptr;
  const bool rot = had != nullptr && had[static_cast<int64_t>(e) * groups + gi];
  transform16(v, s, rot);
  #pragma unroll
  for (int i = 0; i < kGroup; ++i) yp[i] = __float2half_rn(v[i]);
}

// Per-lane partial dot of one fp16 activation row with weight row gr.
__device__ __forceinline__ float row_dot1(
    const half* __restrict__ x, const int K, const int gr,
    const int32_t* __restrict__ codes, const int32_t* __restrict__ code_off,
    const uint8_t* __restrict__ blk_params,
    const int32_t* __restrict__ param_off, const half* __restrict__ row_consts,
    const uint8_t* __restrict__ row_nbits,
    const uint8_t* __restrict__ row_block, const int lane) {
  const int nb = row_nbits[gr];
  const int lb = log2_block(row_block[gr]);
  const float lo_min = __half2float(row_consts[3 * gr + 0]);
  const float lo_scale = __half2float(row_consts[3 * gr + 1]);
  const float step_scale = __half2float(row_consts[3 * gr + 2]);
  const uint32_t* words =
      reinterpret_cast<const uint32_t*>(codes) + code_off[gr];
  const uint8_t* params = blk_params + 2 * static_cast<int64_t>(param_off[gr]);
  float acc[1] = {0.0f};
  switch (nb) {
    case 2:
      row_dot<1, 2>(x, K, 0, 1, words, params, lb, lo_min, lo_scale, step_scale,
                    lane, acc);
      break;
    case 4:
      row_dot<1, 4>(x, K, 0, 1, words, params, lb, lo_min, lo_scale, step_scale,
                    lane, acc);
      break;
    default:
      row_dot<1, 8>(x, K, 0, 1, words, params, lb, lo_min, lo_scale, step_scale,
                    lane, acc);
      break;
  }
  return acc[0];
}

// REDUCE=false: out[p, r] = x[p] . W_{expert[p]}[r]            (gate/up)
// REDUCE=true:  out[t, r] = sum_k w[t, k] x[t*top_k+k] . W_e[r] (down)
template <bool REDUCE>
__global__ void __launch_bounds__(kThreads)
    moe_gemv_kernel(const half* __restrict__ x, const int K, const int top_k,
                    const int32_t* __restrict__ expert,
                    const float* __restrict__ topk_w,
                    const int32_t* __restrict__ codes,
                    const int32_t* __restrict__ code_off,
                    const uint8_t* __restrict__ blk_params,
                    const int32_t* __restrict__ param_off,
                    const half* __restrict__ row_consts,
                    const uint8_t* __restrict__ row_nbits,
                    const uint8_t* __restrict__ row_block, const int n_out,
                    half* __restrict__ out, const int64_t ldo) {
  const int wave = threadIdx.x / kWave;
  const int lane = threadIdx.x % kWave;
  const int r = blockIdx.x * kRowsPerBlock + wave;
  if (r >= n_out) return;
  const int row = blockIdx.y;
  float acc = 0.0f;
  if constexpr (REDUCE) {
    for (int k = 0; k < top_k; ++k) {
      const int p = row * top_k + k;
      const int e = expert[p];
      if (e < 0) continue;
      acc += topk_w[p] * row_dot1(x + static_cast<int64_t>(p) * K, K,
                                  e * n_out + r, codes, code_off, blk_params,
                                  param_off, row_consts, row_nbits, row_block,
                                  lane);
    }
  } else {
    const int e = expert[row];
    if (e >= 0)
      acc = row_dot1(x + static_cast<int64_t>(row) * K, K, e * n_out + r, codes,
                     code_off, blk_params, param_off, row_consts, row_nbits,
                     row_block, lane);
  }
  #pragma unroll
  for (int off = kWave / 2; off > 0; off >>= 1)
    acc += __shfl_xor(acc, off, kWave);
  if (lane == 0)
    out[static_cast<int64_t>(row) * ldo + r] = __float2half_rn(acc);
}

#else  // non-RDNA2 device pass: empty kernels for symbol parity.

__global__ void moe_act_transform_kernel(const half*, const int64_t, const int,
                                         const int32_t*, const half*,
                                         const uint8_t*, half*, const int,
                                         const int) {}
template <bool REDUCE>
__global__ void moe_gemv_kernel(const half*, const int, const int,
                                const int32_t*, const float*, const int32_t*,
                                const int32_t*, const uint8_t*, const int32_t*,
                                const half*, const uint8_t*, const uint8_t*,
                                const int, half*, const int64_t) {}

__global__ void act_quant_kernel(const half*, const int64_t, const half*,
                                 const uint8_t*, int8_t*, half*, int32_t*,
                                 const int, const int) {}
template <int MT>
__global__ void gemv_i8_kernel(const int8_t*, const half*, const int32_t*,
                               const int, const int, const int32_t*,
                               const int32_t*, const uint8_t*, const int32_t*,
                               const half*, const uint8_t*, const uint8_t*,
                               half*, const int64_t, const int) {}

__global__ void act_transform_kernel(const half*, const int64_t, const half*,
                                     const uint8_t*, half*, const int,
                                     const int) {}
template <int MT>
__global__ void gemv_kernel(const half*, const int, const int, const int32_t*,
                            const int32_t*, const uint8_t*, const int32_t*,
                            const half*, const uint8_t*, const uint8_t*, half*,
                            const int64_t, const int) {}
__global__ void dequant_kernel(const int32_t*, const int32_t*, const uint8_t*,
                               const int32_t*, const half*, const uint8_t*,
                               const uint8_t*, half*, const int) {}

#endif  // __HIP__RDNA2__ || !__HIP_DEVICE_COMPILE__

}  // namespace rdna_hint_flex
}  // namespace vllm

// ---------------------------------------------------------------------------
// Host side.
// ---------------------------------------------------------------------------
namespace {

namespace hf = vllm::rdna_hint_flex;

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
  TORCH_CHECK(R < (int64_t{1} << 31) / hf::kRowsPerBlock, "hi2: too many rows");
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
  TORCH_CHECK(had.numel() == K / hf::kGroup, "hi2: had must have K/16 entries");
  const int64_t n = M * (K / hf::kGroup);
  if (n == 0) return;
  const int threads = 256;
  const int64_t blocks = (n + threads - 1) / threads;
  hf::act_transform_kernel<<<blocks, threads, 0, stream>>>(
      reinterpret_cast<const half*>(x.data_ptr()), x.stride(0), s,
      had.data_ptr<uint8_t>(), reinterpret_cast<half*>(y.data_ptr()),
      static_cast<int>(M), static_cast<int>(K));
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

template <int MT>
void launch_gemv_mt(const torch::Tensor& x, const Hi2Weight& w,
                    torch::Tensor& out, int64_t R, cudaStream_t stream) {
  const int64_t M = x.size(0), K = x.size(1);
  dim3 grid((R + hf::kRowsPerBlock - 1) / hf::kRowsPerBlock, (M + MT - 1) / MT);
  hf::gemv_kernel<MT><<<grid, hf::kThreads, 0, stream>>>(
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
  const int64_t ny =
      std::max<int64_t>(1, (max_words + hf::kThreads - 1) / hf::kThreads);
  dim3 grid(static_cast<unsigned>(R), static_cast<unsigned>(ny));
  hf::dequant_kernel<<<grid, hf::kThreads, 0, stream>>>(
      w.codes.data_ptr<int32_t>(), w.code_off.data_ptr<int32_t>(),
      w.blk_params.data_ptr<uint8_t>(), w.param_off.data_ptr<int32_t>(),
      reinterpret_cast<const half*>(w.row_consts.data_ptr()),
      w.row_nbits.data_ptr<uint8_t>(), w.row_block.data_ptr<uint8_t>(),
      reinterpret_cast<half*>(out.data_ptr()), static_cast<int>(K));
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

// I8 activation: xq [M, K] int8, sx [M, K/16] fp16, sumq [M, K/16] int32.
void launch_act_quant(const torch::Tensor& x, const torch::Tensor& col_scale,
                      const torch::Tensor& had, bool transform,
                      torch::Tensor& xq, torch::Tensor& sx, torch::Tensor& sumq,
                      cudaStream_t stream) {
  const int64_t M = x.size(0), K = x.size(1);
  const half* s = nullptr;
  const uint8_t* h = nullptr;
  if (transform) {
    if (col_scale.numel() > 0) {
      check_dev(col_scale, "col_scale", torch::kHalf);
      TORCH_CHECK(col_scale.numel() == K, "hi2: col_scale must have K entries");
      s = reinterpret_cast<const half*>(col_scale.data_ptr());
    }
    check_dev(had, "had", torch::kUInt8);
    TORCH_CHECK(had.numel() == K / hf::kGroup,
                "hi2: had must have K/16 entries");
    h = had.data_ptr<uint8_t>();
  }
  const int64_t n = M * (K / hf::kGroup);
  if (n == 0) return;
  const int threads = 256;
  hf::act_quant_kernel<<<(n + threads - 1) / threads, threads, 0, stream>>>(
      reinterpret_cast<const half*>(x.data_ptr()), x.stride(0), s, h,
      xq.data_ptr<int8_t>(), reinterpret_cast<half*>(sx.data_ptr()),
      sumq.data_ptr<int32_t>(), static_cast<int>(M), static_cast<int>(K));
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

template <int MT>
void launch_gemv_i8_mt(const torch::Tensor& xq, const torch::Tensor& sx,
                       const torch::Tensor& sumq, const Hi2Weight& w,
                       torch::Tensor& out, int64_t R, cudaStream_t stream) {
  const int64_t M = xq.size(0), K = xq.size(1);
  dim3 grid((R + hf::kRowsPerBlock - 1) / hf::kRowsPerBlock, (M + MT - 1) / MT);
  hf::gemv_i8_kernel<MT><<<grid, hf::kThreads, 0, stream>>>(
      xq.data_ptr<int8_t>(), reinterpret_cast<const half*>(sx.data_ptr()),
      sumq.data_ptr<int32_t>(), static_cast<int>(M), static_cast<int>(K),
      w.codes.data_ptr<int32_t>(), w.code_off.data_ptr<int32_t>(),
      w.blk_params.data_ptr<uint8_t>(), w.param_off.data_ptr<int32_t>(),
      reinterpret_cast<const half*>(w.row_consts.data_ptr()),
      w.row_nbits.data_ptr<uint8_t>(), w.row_block.data_ptr<uint8_t>(),
      reinterpret_cast<half*>(out.data_ptr()), out.stride(0),
      static_cast<int>(R));
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

void launch_gemv_i8(const torch::Tensor& xq, const torch::Tensor& sx,
                    const torch::Tensor& sumq, const Hi2Weight& w,
                    torch::Tensor& out, int64_t R, cudaStream_t stream) {
  const int64_t M = xq.size(0);
  if (M == 0 || R == 0) return;
  if (M == 1) {
    launch_gemv_i8_mt<1>(xq, sx, sumq, w, out, R, stream);
  } else if (M == 2) {
    launch_gemv_i8_mt<2>(xq, sx, sumq, w, out, R, stream);
  } else if (M <= 4) {
    launch_gemv_i8_mt<4>(xq, sx, sumq, w, out, R, stream);
  } else {
    launch_gemv_i8_mt<8>(xq, sx, sumq, w, out, R, stream);
  }
}

void launch_moe_transform(const torch::Tensor& x, int src_div,
                          const torch::Tensor& expert,
                          const torch::Tensor& col_scale,
                          const torch::Tensor& had, bool transform,
                          torch::Tensor& y, cudaStream_t stream) {
  const int64_t P = y.size(0), K = y.size(1);
  const int64_t n = P * (K / hf::kGroup);
  if (n == 0) return;
  const half* s = transform && col_scale.numel() > 0
                      ? reinterpret_cast<const half*>(col_scale.data_ptr())
                      : nullptr;
  const uint8_t* h =
      transform && had.numel() > 0 ? had.data_ptr<uint8_t>() : nullptr;
  const int threads = 256;
  hf::moe_act_transform_kernel<<<(n + threads - 1) / threads, threads, 0,
                                 stream>>>(
      reinterpret_cast<const half*>(x.data_ptr()), x.stride(0), src_div,
      expert.data_ptr<int32_t>(), s, h, reinterpret_cast<half*>(y.data_ptr()),
      static_cast<int>(P), static_cast<int>(K));
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

template <bool REDUCE>
void launch_moe_gemv(const torch::Tensor& x, int top_k,
                     const torch::Tensor& expert, const torch::Tensor& topk_w,
                     const Hi2Weight& w, int64_t n_out, torch::Tensor& out,
                     cudaStream_t stream) {
  const int64_t rows = out.size(0);
  if (rows == 0 || n_out == 0) return;
  dim3 grid((n_out + hf::kRowsPerBlock - 1) / hf::kRowsPerBlock, rows);
  hf::moe_gemv_kernel<REDUCE><<<grid, hf::kThreads, 0, stream>>>(
      reinterpret_cast<const half*>(x.data_ptr()), static_cast<int>(x.size(1)),
      top_k, expert.data_ptr<int32_t>(), topk_w.data_ptr<float>(),
      w.codes.data_ptr<int32_t>(), w.code_off.data_ptr<int32_t>(),
      w.blk_params.data_ptr<uint8_t>(), w.param_off.data_ptr<int32_t>(),
      reinterpret_cast<const half*>(w.row_consts.data_ptr()),
      w.row_nbits.data_ptr<uint8_t>(), w.row_block.data_ptr<uint8_t>(),
      static_cast<int>(n_out), reinterpret_cast<half*>(out.data_ptr()),
      out.stride(0));
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

}  // namespace

// x [M, K] fp16 -> out [M, K] fp16 = H16(x / s) on flagged groups.
void rdna_hint_flex_act_transform_rdna2(torch::Tensor x,
                                        torch::Tensor col_scale,
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
void rdna_hint_flex_gemv_rdna2(torch::Tensor x, torch::Tensor codes,
                               torch::Tensor code_off, torch::Tensor blk_params,
                               torch::Tensor param_off,
                               torch::Tensor row_consts,
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
void rdna_hint_flex_dequant_rdna2(torch::Tensor codes, torch::Tensor code_off,
                                  torch::Tensor blk_params,
                                  torch::Tensor param_off,
                                  torch::Tensor row_consts,
                                  torch::Tensor row_nbits,
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

// Routed experts: out[t] = sum_k w[t,k] * down_e(silu(gate_e(x_t)) * up_e(x_t))
// with e = topk_ids[t, k] (local expert id, < 0 = not on this rank). w13 rows
// of expert e: [e*2N, e*2N+N) gate, [e*2N+N, (e+1)*2N) up; w2 rows of expert
// e: [e*H, (e+1)*H). Per-expert col_scale [E, K] (or empty) and had [E, K/16].
torch::Tensor rdna_hint_flex_moe_rdna2(
    torch::Tensor x, torch::Tensor topk_ids, torch::Tensor topk_weights,
    torch::Tensor w13_codes, torch::Tensor w13_code_off,
    torch::Tensor w13_blk_params, torch::Tensor w13_param_off,
    torch::Tensor w13_row_consts, torch::Tensor w13_row_nbits,
    torch::Tensor w13_row_block, torch::Tensor w13_col_scale,
    torch::Tensor w13_had, bool w13_transform, torch::Tensor w2_codes,
    torch::Tensor w2_code_off, torch::Tensor w2_blk_params,
    torch::Tensor w2_param_off, torch::Tensor w2_row_consts,
    torch::Tensor w2_row_nbits, torch::Tensor w2_row_block,
    torch::Tensor w2_col_scale, torch::Tensor w2_had, bool w2_transform,
    int64_t num_experts, int64_t intermediate) {
  const Hi2Weight w13{w13_codes,     w13_code_off,   w13_blk_params,
                      w13_param_off, w13_row_consts, w13_row_nbits,
                      w13_row_block};
  const Hi2Weight w2{w2_codes,      w2_code_off,  w2_blk_params, w2_param_off,
                     w2_row_consts, w2_row_nbits, w2_row_block};
  const int64_t R13 = check_weight(w13), R2 = check_weight(w2);
  const int64_t T = x.size(0), H = x.size(1), N = intermediate;
  check_act(x, H);
  TORCH_CHECK(N % 64 == 0,
              "rdna_hint_flex: intermediate must be a multiple of 64");
  TORCH_CHECK(R13 == num_experts * 2 * N && R2 == num_experts * H,
              "rdna_hint_flex: stacked expert rows do not match E, N, H");
  check_dev(topk_ids, "topk_ids", torch::kInt32);
  check_dev(topk_weights, "topk_weights", torch::kFloat);
  TORCH_CHECK(topk_ids.dim() == 2 && topk_ids.size(0) == T &&
                  topk_weights.sizes() == topk_ids.sizes(),
              "rdna_hint_flex: topk_ids/topk_weights must be [T, top_k]");
  const int top_k = static_cast<int>(topk_ids.size(1));
  const int64_t P = T * top_k;
  const at::cuda::OptionalCUDAGuard device_guard(device_of(x));
  auto stream = at::cuda::getCurrentCUDAStream();
  auto out = torch::empty({T, H}, x.options());
  if (T == 0) return out;
  auto xp = torch::empty({P, H}, x.options());
  launch_moe_transform(x, top_k, topk_ids, w13_col_scale, w13_had,
                       w13_transform, xp, stream);
  auto g13 = torch::empty({P, 2 * N}, x.options());
  launch_moe_gemv<false>(xp, top_k, topk_ids, topk_weights, w13, 2 * N, g13,
                         stream);
  auto h = at::silu(g13.narrow(1, 0, N)) * g13.narrow(1, N, N);
  auto hp = torch::empty({P, N}, x.options());
  launch_moe_transform(h, 1, topk_ids, w2_col_scale, w2_had, w2_transform, hp,
                       stream);
  launch_moe_gemv<true>(hp, top_k, topk_ids, topk_weights, w2, H, out, stream);
  return out;
}

// I8 activation quantization (transform included when `transform`).
void rdna_hint_flex_act_quant_rdna2(torch::Tensor x, torch::Tensor col_scale,
                                    torch::Tensor had, bool transform,
                                    torch::Tensor xq, torch::Tensor sx,
                                    torch::Tensor sumq) {
  check_act(x, x.size(1));
  const int64_t M = x.size(0), K = x.size(1);
  check_dev(xq, "xq", torch::kInt8);
  check_dev(sx, "sx", torch::kHalf);
  check_dev(sumq, "sumq", torch::kInt32);
  TORCH_CHECK(xq.numel() == M * K && sx.numel() == M * K / 16 &&
                  sumq.numel() == M * K / 16,
              "hi2: xq/sx/sumq must be [M, K], [M, K/16], [M, K/16]");
  const at::cuda::OptionalCUDAGuard device_guard(device_of(x));
  auto stream = at::cuda::getCurrentCUDAStream();
  launch_act_quant(x, col_scale, had, transform, xq, sx, sumq, stream);
}

// I8 GEMV from a quantized activation -> out [M, R] fp16.
void rdna_hint_flex_gemv_i8_rdna2(
    torch::Tensor xq, torch::Tensor sx, torch::Tensor sumq, torch::Tensor codes,
    torch::Tensor code_off, torch::Tensor blk_params, torch::Tensor param_off,
    torch::Tensor row_consts, torch::Tensor row_nbits, torch::Tensor row_block,
    torch::Tensor out) {
  const Hi2Weight w{codes,      code_off,  blk_params, param_off,
                    row_consts, row_nbits, row_block};
  const int64_t R = check_weight(w);
  check_dev(xq, "xq", torch::kInt8);
  check_dev(sx, "sx", torch::kHalf);
  check_dev(sumq, "sumq", torch::kInt32);
  TORCH_CHECK(xq.dim() == 2 && xq.size(1) % 64 == 0, "hi2: xq must be [M, K]");
  TORCH_CHECK(out.is_cuda() && out.scalar_type() == torch::kHalf &&
                  out.dim() == 2 && out.size(0) == xq.size(0) &&
                  out.size(1) == R && out.stride(1) == 1,
              "hi2: out must be fp16 [M, R] with unit column stride");
  const at::cuda::OptionalCUDAGuard device_guard(device_of(xq));
  auto stream = at::cuda::getCurrentCUDAStream();
  launch_gemv_i8(xq, sx, sumq, w, out, R, stream);
}

// y = x . W^T for an rdna_hint_flex tensor (transform included when
// `transform`). mode 0 (F16): M <= decode_max_m runs the F16 GEMV, otherwise
// dequant + rocBLAS GEMM. mode 1 (I8 decode): M <= decode_max_m runs the I8
// path, larger M as mode 0. mode 2 (I8 everywhere): the I8 path for every M.
torch::Tensor rdna_hint_flex_linear_rdna2(
    torch::Tensor x, torch::Tensor col_scale, torch::Tensor had,
    torch::Tensor codes, torch::Tensor code_off, torch::Tensor blk_params,
    torch::Tensor param_off, torch::Tensor row_consts, torch::Tensor row_nbits,
    torch::Tensor row_block, bool transform, int64_t decode_max_m,
    int64_t mode) {
  const Hi2Weight w{codes,      code_off,  blk_params, param_off,
                    row_consts, row_nbits, row_block};
  const int64_t R = check_weight(w);
  const int64_t K = x.size(1);
  check_act(x, K);
  TORCH_CHECK(mode >= 0 && mode <= 2, "hi2: mode must be 0, 1 or 2");
  const int64_t M = x.size(0);
  const at::cuda::OptionalCUDAGuard device_guard(device_of(x));
  auto stream = at::cuda::getCurrentCUDAStream();
  auto out = torch::empty({M, R}, x.options());
  if (M == 0 || R == 0) return out;

  if (mode == 2 || (mode == 1 && M <= decode_max_m)) {
    const auto opts = x.options();
    auto xq = torch::empty({M, K}, opts.dtype(torch::kInt8));
    auto sx = torch::empty({M, K / 16}, opts);
    auto sumq = torch::empty({M, K / 16}, opts.dtype(torch::kInt32));
    launch_act_quant(x, col_scale, had, transform, xq, sx, sumq, stream);
    launch_gemv_i8(xq, sx, sumq, w, out, R, stream);
    return out;
  }

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

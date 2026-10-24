// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project
//
// Packed INT4 + FP16 block-scale KV cache layout for RDNA2 (gfx1030).
//
// Record layout (one (block, slot, head) record):
//   [ D/2 payload bytes , D/32 little-endian uint16 scale words ]
//   * payload: two INT4 codes per byte, low nibble first
//   * scale word: (fp16_bits(scale) & 0xfff0) | zp
//     - the high 12 bits are an FP16 scale with its low 4 mantissa bits
//       cleared so the scale always lands on a 6-bit-mantissa grid
//     - the low 4 bits are a 4-bit zero-point
//   * one scale covers 32 contiguous head-dimension channels
//     (groups = D/32)
//   * dequant: value = (code - zp) * scale, clamped to the FP16 range
//
// The scale words are NOT a separate allocation: they are a view onto the
// tail of the same physical pages, so the descriptor carries the payload
// pointer and the scale pointer with independent byte strides.
//
// Used by:
//   * csrc/rocm/fa_rdna2.cu            (RDNA_ATTN decode + prefill)
//   * csrc/rocm/qsa_rdna2.cu            (QSA sparse decode)
//   * csrc/rocm/mono_qsa_decode_rdna2.cu (MonoKernel QSA decode)
//
// Every consumer must identify the layout positively (the descriptor says
// which format it was handed); never infer it from a dtype. A kernel that
// reads these bytes as FP16 produces finite, plausible, silently wrong
// numbers rather than a fault.

#ifndef _INT4_KV_LAYOUT_RDNA2_CUH
#define _INT4_KV_LAYOUT_RDNA2_CUH

#include <cstdint>
#include <type_traits>

#include <hip/hip_fp16.h>

namespace vllm {
namespace int4_kv {

// ---------------------------------------------------------------------------
// Layout descriptor
// ---------------------------------------------------------------------------

// Byte strides, per side (0 = K, 1 = V), per dimension:
//   [0] block   [1] head   [2] slot   [3] channel
// `scales[side]` points at the first scale word of side `side`.
struct Int4KvLayout {
  bool vector8;  // 8-channel (uint4) loads are legal
  const uint8_t* scales[2];
  int64_t data_stride[2][4];   // payload bytes
  int64_t scale_stride[2][4];  // scale bytes
};

// Empty placeholder so a kernel can take the descriptor as a defaulted
// template argument without carrying the INT4 layout when it is unused.
struct NoInt4KvLayout {};

template <bool Enabled>
using Int4KvArg = std::conditional_t<Enabled, Int4KvLayout, NoInt4KvLayout>;

// ---------------------------------------------------------------------------
// Dequant helpers
//
// The payload and the scale tail live in different allocations only in the
// sense that the scale is a window into the same page: the row helpers below
// therefore take the two row bases separately and apply the channel strides
// themselves.
// ---------------------------------------------------------------------------

// One channel of one record, dequantized to half.
__forceinline__ __device__ half int4_load_half(const uint8_t* cache,
                                               const Int4KvLayout& layout,
                                               int side, int block, int head,
                                               int slot, int d) {
  const auto& ds = layout.data_stride[side];
  const auto& ss = layout.scale_stride[side];
  const int64_t data_offset = int64_t(block) * ds[0] + int64_t(head) * ds[1] +
                              int64_t(slot) * ds[2] + int64_t(d / 2) * ds[3];
  const int64_t scale_offset = int64_t(block) * ss[0] + int64_t(head) * ss[1] +
                               int64_t(slot) * ss[2] + int64_t(d / 32) * ss[3];
  const uint8_t* sp = layout.scales[side] + scale_offset;
  const uint16_t word = uint16_t(sp[0]) | (uint16_t(sp[1]) << 8);
  const uint16_t clean_scale = word & 0xfff0;
  half block_scale;
  __builtin_memcpy(&block_scale, &clean_scale, sizeof(half));
  const int q = (cache[data_offset] >> ((d & 1) * 4)) & 15;
  const float value = float(q - int(word & 15)) * __half2float(block_scale);
  return __float2half_rn(fminf(65504.0f, fmaxf(-65504.0f, value)));
}

// Two adjacent channels of one record, dequantized to half2.
// `data_row` / `scale_row` are the record bases; the channel strides carry
// the per-channel step inside the record.
__forceinline__ __device__ half2 int4_pair_from_rows(
    const uint8_t* data_row, const uint8_t* scale_row,
    int64_t data_channel_stride, int64_t scale_channel_stride, int d) {
  // Shared integer row bases otherwise lose global address-space provenance.
  using GlobalByte = const uint8_t __attribute__((address_space(1)));
  using GlobalWord = const uint16_t __attribute__((address_space(1)));
  const auto* data =
      reinterpret_cast<GlobalByte*>(reinterpret_cast<uint64_t>(data_row) +
                                    int64_t(d / 2) * data_channel_stride);
  const auto* sp =
      reinterpret_cast<GlobalWord*>(reinterpret_cast<uint64_t>(scale_row) +
                                    int64_t(d / 32) * scale_channel_stride);
  // Metadata is a 16-bit tensor view, so every scale address is aligned.
  const uint16_t word = *sp;
  const uint16_t clean_scale = word & 0xfff0;
  half block_scale;
  __builtin_memcpy(&block_scale, &clean_scale, sizeof(half));
  const float scale = __half2float(block_scale);
  const int zp = word & 15;
  const uint8_t packed = *data;
  const float lo = float(int(packed & 15) - zp) * scale;
  const float hi = float(int(packed >> 4) - zp) * scale;
  return __floats2half2_rn(fminf(65504.0f, fmaxf(-65504.0f, lo)),
                           fminf(65504.0f, fmaxf(-65504.0f, hi)));
}

// Eight adjacent channels of one record, dequantized to a uint4 of halves.
__forceinline__ __device__ uint4
int4_eight_from_rows(const uint8_t* data_row, const uint8_t* scale_row,
                     int64_t scale_channel_stride, int d) {
  using GlobalDword = const uint32_t __attribute__((address_space(1)));
  using GlobalWord = const uint16_t __attribute__((address_space(1)));
  const auto* data = reinterpret_cast<GlobalDword*>(
      reinterpret_cast<uint64_t>(data_row) + d / 2);
  const auto* sp =
      reinterpret_cast<GlobalWord*>(reinterpret_cast<uint64_t>(scale_row) +
                                    int64_t(d / 32) * scale_channel_stride);
  const uint32_t packed = *data;
  const uint16_t word = *sp;
  const uint16_t clean_scale = word & 0xfff0;
  half block_scale;
  __builtin_memcpy(&block_scale, &clean_scale, sizeof(half));
  const float scale = __half2float(block_scale);
  const int zp = word & 15;
  uint32_t pairs[4];
#pragma unroll
  for (int pair = 0; pair < 4; ++pair) {
    const float lo = float(int((packed >> (pair * 8)) & 15) - zp) * scale;
    const float hi = float(int((packed >> (pair * 8 + 4)) & 15) - zp) * scale;
    const half2 value =
        __floats2half2_rn(fminf(65504.0f, fmaxf(-65504.0f, lo)),
                          fminf(65504.0f, fmaxf(-65504.0f, hi)));
    __builtin_memcpy(&pairs[pair], &value, sizeof(half2));
  }
  return make_uint4(pairs[0], pairs[1], pairs[2], pairs[3]);
}

}  // namespace int4_kv
}  // namespace vllm

#endif  // _INT4_KV_LAYOUT_RDNA2_CUH

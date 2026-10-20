# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Reference codec for the proposed INT4 + FP16 block-scale KV cache format.

This file is the executable half of ``docs/explore/kv-int4-fp16-scales/README.md``.
It is pure Python with no third-party dependency so that it runs anywhere the
repo's docs run, and so that every number quoted in the README can be
regenerated with::

    python3 docs/explore/kv-int4-fp16-scales/reference_codec.py

It compares scale strategies for a 4-bit KV cache, all at the same element
width:

``int4_per_token_head``
    What the fork ships today. Asymmetric INT4, one FP32 scale per
    (token, head) row, with a 4-bit zero-point hidden in the scale's low
    mantissa bits. Mirrors ``vllm/v1/attention/ops/int4_per_token_head.py``.

``int4_fp16_block``
    The proposal. Symmetric INT4, one FP16 scale per block of ``k`` values
    along the head dimension, single level (no global scale).

``int4_fp16_block_g``
    The proposal with NVFP4's second level: FP16 block scale divided by a
    per-tensor FP32 global scale, which is multiplied back at read time.

``int4_e4m3_block_g``
    NVFP4's scale geometry with an integer element type: symmetric INT4,
    one E4M3 scale per block, divided by a per-tensor FP32 global scale.

``nvfp4``
    NVIDIA NVFP4 itself: E2M1 elements, one E4M3 scale per 16 values, divided
    by a per-tensor FP32 global scale.

The global scale is modelled the way NVFP4 actually does it: the block scale
stored in the cache is ``effective_scale / global``, and reconstruction
multiplies the global back in. That detail matters, because E4M3 has a narrow
exponent range and a block whose effective scale is far below the tensor
maximum underflows it, while FP16 does not.

The script reports round-trip SNR and the true bits/value of each scheme,
including scale overhead, so the cost of the scale format is visible next to
the accuracy it buys.

These are arithmetic results on synthetic data. They are not measurements on a
V620; ``TESTPLAN.md`` carries the empty cells for those.
"""

from __future__ import annotations

import bisect
import math
import random
import struct
from collections.abc import Sequence
from dataclasses import dataclass

__all__ = [
    "e4m3_encode",
    "e4m3_decode",
    "f16_round",
    "pack_int4",
    "unpack_int4",
    "quantize_int4_per_token_head",
    "quantize_int4_fp16_block",
    "quantize_int4_fp16_block_g",
    "quantize_int4_e4m3_block_g",
    "quantize_nvfp4",
    "snr_db",
    "bits_per_value",
]

#: Largest finite value of the OCP E4M3 element type.
E4M3_MAX = 448.0
#: Largest finite value of FP16.
FP16_MAX = 65504.0
#: Smallest positive normal FP16.
FP16_MIN_NORMAL = 6.103515625e-05
#: Smallest positive normal E4M3.
E4M3_MIN_NORMAL = 2.0**-6


def f16_round(x: float) -> float:
    """Round ``x`` to the nearest FP16 value, ties to even.

    Args:
        x: The value to round.

    Returns:
        The FP16-representable value nearest to ``x``. Values beyond the FP16
        range saturate to +/- ``FP16_MAX`` rather than raising.

    """
    if x != x:  # NaN
        return float("nan")
    if x > FP16_MAX:
        return FP16_MAX
    if x < -FP16_MAX:
        return -FP16_MAX
    return struct.unpack("<e", struct.pack("<e", x))[0]


def f16_is_normal(x: float) -> bool:
    """Return True when ``x`` is a normal (non-zero, non-subnormal) FP16."""
    return x != 0.0 and abs(x) >= FP16_MIN_NORMAL


def e4m3_is_normal(x: float) -> bool:
    """Return True when ``x`` is a normal (non-zero, non-subnormal) E4M3."""
    return x != 0.0 and abs(x) >= E4M3_MIN_NORMAL


def _e4m3_code_to_float(code: int) -> float | None:
    """Decode one E4M3 code. Returns None for the two NaN encodings."""
    sign = -1.0 if code & 0x80 else 1.0
    exp = (code >> 3) & 0x0F
    mant = code & 0x07
    if exp == 0x0F and mant == 0x07:
        return None  # NaN
    if exp == 0:
        return sign * (mant / 8.0) * 2.0**-6
    return sign * (1.0 + mant / 8.0) * 2.0 ** (exp - 7)


#: All finite E4M3 values, sorted, with their codes and mantissa parity.
_E4M3_TABLE: list[tuple[float, int, bool]] = sorted(
    (v, code, (code & 0x07) % 2 == 0)
    for code in range(256)
    if (v := _e4m3_code_to_float(code)) is not None
)
_E4M3_VALUES = [entry[0] for entry in _E4M3_TABLE]


def e4m3_decode(code: int) -> float:
    """Decode one E4M3 code to a float. NaN codes decode to NaN."""
    value = _e4m3_code_to_float(code & 0xFF)
    return float("nan") if value is None else value


def e4m3_encode(x: float) -> int:
    """Round ``x`` to the nearest E4M3 value and return its code.

    Ties go to the neighbour whose mantissa LSB is zero, matching the
    round-to-nearest-even behaviour of the hardware cast.

    Args:
        x: The value to encode.

    Returns:
        The 8-bit E4M3 code. NaN and infinities encode to the NaN code 0x7F.

    """
    if x != x or math.isinf(x):
        return 0x7F
    if x > E4M3_MAX:
        x = E4M3_MAX
    if x < -E4M3_MAX:
        x = -E4M3_MAX
    idx = bisect.bisect_left(_E4M3_VALUES, x)
    if idx == 0:
        return _E4M3_TABLE[0][1]
    if idx == len(_E4M3_VALUES):
        return _E4M3_TABLE[-1][1]
    lo = _E4M3_TABLE[idx - 1]
    hi = _E4M3_TABLE[idx]
    d_lo = x - lo[0]
    d_hi = hi[0] - x
    if d_lo < d_hi:
        return lo[1]
    if d_hi < d_lo:
        return hi[1]
    # Exact tie: prefer the even mantissa.
    return lo[1] if lo[2] else hi[1]


# ---------------------------------------------------------------------------
# INT4 packing
# ---------------------------------------------------------------------------


def pack_int4(codes: Sequence[int]) -> bytes:
    """Pack unsigned 4-bit codes, two per byte, low nibble first.

    Args:
        codes: Values in ``[0, 15]``.

    Returns:
        The packed bytes.

    """
    out = bytearray()
    for i in range(0, len(codes), 2):
        lo = codes[i] & 0x0F
        hi = (codes[i + 1] & 0x0F) if i + 1 < len(codes) else 0
        out.append(lo | (hi << 4))
    return bytes(out)


def unpack_int4(data: bytes, count: int) -> list[int]:
    """Unpack ``count`` 4-bit codes from ``data``, low nibble first."""
    out: list[int] = []
    for i in range(count):
        byte = data[i >> 1]
        out.append((byte >> 4) if (i & 1) else (byte & 0x0F))
    return out


def _round_half_away(x: float) -> int:
    """Round half away from zero, matching the shipped INT4 kernel path."""
    return int(math.floor(x + 0.5)) if x >= 0 else -int(math.floor(-x + 0.5))


def _round_half_even(x: float) -> int:
    """Round half to even, matching a hardware float-to-int cast."""
    return int(round(x))


# ---------------------------------------------------------------------------
# Result container
# ---------------------------------------------------------------------------


@dataclass
class TensorResult:
    """Round-trip result for a whole KV tensor.

    Attributes:
        reconstructed: The dequantized values, one list per row.
        scales: The stored block scales, one list per row.
        global_scale: The per-tensor global scale, or None for single-level
            schemes.
        underflowed_blocks: Number of blocks whose stored scale fell below
            the normal range of its scale format.

    """

    reconstructed: list[list[float]]
    scales: list[list[float]]
    global_scale: float | None = None
    underflowed_blocks: int = 0


def _blocks(row: Sequence[float], k: int):
    """Yield (start, block) pairs over a row."""
    for start in range(0, len(row), k):
        yield start, row[start : start + k]


# ---------------------------------------------------------------------------
# Scheme A: int4_per_token_head (shipped today)
# ---------------------------------------------------------------------------


def quantize_int4_per_token_head(rows: Sequence[Sequence[float]]) -> TensorResult:
    """Quantize rows the way ``int4_per_token_head`` does today.

    Asymmetric: the scale covers ``(max - min) / 15`` and a 4-bit zero-point
    absorbs the offset. The shipped kernel keeps the zero-point in the low
    mantissa bits of the FP32 scale; this reference keeps them apart, which
    changes no arithmetic.

    Args:
        rows: The rows to quantize.

    Returns:
        The round-trip result.

    """
    reconstructed: list[list[float]] = []
    scales: list[list[float]] = []
    for row in rows:
        lo = min(row)
        hi = max(row)
        scale = max((hi - lo) / 15.0, 1e-6)
        zp = max(0, min(15, _round_half_away(-lo / scale)))
        out = []
        for value in row:
            code = max(0, min(15, _round_half_away(value / scale) + zp))
            out.append((code - zp) * scale)
        reconstructed.append(out)
        scales.append([scale])
    return TensorResult(reconstructed, scales)


# ---------------------------------------------------------------------------
# Scheme B: int4_fp16_block (the proposal, single level)
# ---------------------------------------------------------------------------


def quantize_int4_fp16_block(
    rows: Sequence[Sequence[float]],
    k: int = 32,
    symmetric: bool = True,
) -> TensorResult:
    """Quantize rows with a block-granular FP16 scale and no global scale.

    Args:
        rows: The rows to quantize.
        k: Block size along the head dimension.
        symmetric: When True use a symmetric scale and no zero-point.

    Returns:
        The round-trip result.

    """
    reconstructed: list[list[float]] = []
    scales: list[list[float]] = []
    underflowed = 0
    for row in rows:
        out: list[float] = []
        row_scales: list[float] = []
        for _, block in _blocks(row, k):
            if symmetric:
                scale = max(max(abs(v) for v in block) / 7.0, 1e-6)
                zp = 0
                codes = [max(-8, min(7, _round_half_even(v / scale))) for v in block]
            else:
                lo = min(block)
                hi = max(block)
                scale = max((hi - lo) / 15.0, 1e-6)
                zp = max(0, min(15, _round_half_away(-lo / scale)))
                codes = [
                    max(0, min(15, _round_half_away(v / scale) + zp)) for v in block
                ]
            scale16 = f16_round(scale)
            if not f16_is_normal(scale16):
                underflowed += 1
            row_scales.append(scale16)
            if symmetric:
                out.extend(code * scale16 for code in codes)
            else:
                out.extend((code - zp) * scale16 for code in codes)
        reconstructed.append(out)
        scales.append(row_scales)
    return TensorResult(reconstructed, scales, underflowed_blocks=underflowed)


# ---------------------------------------------------------------------------
# Scheme C: int4_fp16_block_g (the proposal with NVFP4's second level)
# ---------------------------------------------------------------------------


def quantize_int4_fp16_block_g(
    rows: Sequence[Sequence[float]],
    k: int = 32,
) -> TensorResult:
    """Quantize rows with an FP16 block scale divided by a global scale.

    The global scale is the largest effective block scale in the tensor, so
    the stored block scales land at or below 1.0 and the global is multiplied
    back at read time. This is NVFP4's structure with an FP16 block scale.

    Args:
        rows: The rows to quantize.
        k: Block size along the head dimension.

    Returns:
        The round-trip result.

    """
    effective: list[list[float]] = []
    for row in rows:
        row_effective = []
        for _, block in _blocks(row, k):
            row_effective.append(max(max(abs(v) for v in block) / 7.0, 1e-6))
        effective.append(row_effective)
    global_scale = max(max(row) for row in effective)

    reconstructed: list[list[float]] = []
    scales: list[list[float]] = []
    underflowed = 0
    for row, row_effective in zip(rows, effective):
        out: list[float] = []
        row_scales: list[float] = []
        for (start, block), eff in zip(_blocks(row, k), row_effective):
            stored = f16_round(eff / global_scale)
            if not f16_is_normal(stored):
                underflowed += 1
            row_scales.append(stored)
            scale = stored * global_scale
            codes = [max(-8, min(7, _round_half_even(v / scale))) for v in block]
            out.extend(code * scale for code in codes)
        reconstructed.append(out)
        scales.append(row_scales)
    return TensorResult(
        reconstructed, scales, global_scale=global_scale, underflowed_blocks=underflowed
    )


# ---------------------------------------------------------------------------
# Scheme D: int4_e4m3_block_g (NVFP4 scale geometry, integer elements)
# ---------------------------------------------------------------------------


def quantize_int4_e4m3_block_g(
    rows: Sequence[Sequence[float]],
    k: int = 16,
) -> TensorResult:
    """Quantize rows with an E4M3 block scale divided by a global scale.

    Mirrors NVFP4's scale geometry (E4M3 per block, FP32 global) but keeps
    uniform INT4 elements instead of E2M1.

    Args:
        rows: The rows to quantize.
        k: Block size along the head dimension.

    Returns:
        The round-trip result.

    """
    effective: list[list[float]] = []
    for row in rows:
        effective.append(
            [
                max(max(abs(v) for v in block) / 7.0, 1e-6)
                for _, block in _blocks(row, k)
            ]
        )
    global_scale = max(max(row) for row in effective)

    reconstructed: list[list[float]] = []
    scales: list[list[float]] = []
    underflowed = 0
    for row, row_effective in zip(rows, effective):
        out: list[float] = []
        row_scales: list[float] = []
        for (start, block), eff in zip(_blocks(row, k), row_effective):
            stored = e4m3_decode(e4m3_encode(eff / global_scale))
            if not e4m3_is_normal(stored):
                underflowed += 1
            row_scales.append(stored)
            scale = stored * global_scale
            codes = [max(-8, min(7, _round_half_even(v / scale))) for v in block]
            out.extend(code * scale for code in codes)
        reconstructed.append(out)
        scales.append(row_scales)
    return TensorResult(
        reconstructed, scales, global_scale=global_scale, underflowed_blocks=underflowed
    )


def quantize_int4_e4m3_block(
    rows: Sequence[Sequence[float]],
    k: int = 16,
) -> TensorResult:
    """Quantize rows with an E4M3 block scale and no global scale.

    The single-level counterpart of :func:`quantize_int4_e4m3_block_g`, kept
    so the cost of NVFP4's second level can be measured rather than asserted.

    Args:
        rows: The rows to quantize.
        k: Block size along the head dimension.

    Returns:
        The round-trip result.

    """
    reconstructed: list[list[float]] = []
    scales: list[list[float]] = []
    underflowed = 0
    for row in rows:
        out: list[float] = []
        row_scales: list[float] = []
        for _, block in _blocks(row, k):
            eff = max(max(abs(v) for v in block) / 7.0, 1e-6)
            stored = e4m3_decode(e4m3_encode(eff))
            if not e4m3_is_normal(stored):
                underflowed += 1
            row_scales.append(stored)
            codes = [max(-8, min(7, _round_half_even(v / stored))) for v in block]
            out.extend(code * stored for code in codes)
        reconstructed.append(out)
        scales.append(row_scales)
    return TensorResult(reconstructed, scales, underflowed_blocks=underflowed)


# ---------------------------------------------------------------------------
# Scheme E: NVFP4 (E2M1 elements, E4M3 scale per 16)
# ---------------------------------------------------------------------------

#: E2M1 magnitudes. E2M1 has no negative zero; both zeros decode to 0.
_E2M1_VALUES = [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0]


def _e2m1_decode(code: int) -> float:
    """Decode a 4-bit E2M1 code to its value."""
    sign = -1.0 if code & 0x08 else 1.0
    return sign * _E2M1_VALUES[code & 0x07]


def _e2m1_encode(x: float) -> int:
    """Round ``x`` to the nearest E2M1 code."""
    sign_bit = 0x08 if x < 0 else 0x00
    mag = abs(x)
    best_code = 0
    best_err = math.inf
    for idx, value in enumerate(_E2M1_VALUES):
        err = abs(mag - value)
        if err < best_err:
            best_err = err
            best_code = idx
    return best_code | sign_bit


def quantize_nvfp4(rows: Sequence[Sequence[float]], k: int = 16) -> TensorResult:
    """Quantize rows with E2M1 elements and an E4M3 scale per block.

    This is the NVFP4 geometry the fork already has for Blackwell
    (``csrc/libtorch_stable/nvfp4_kv_cache_kernels.cu``), reproduced here in
    Python so its cost can be compared on the same data. The global scale is
    modelled the way the kernel models it: folded into the stored block scale
    and multiplied back at read time.

    Args:
        rows: The rows to quantize.
        k: Block size along the head dimension.

    Returns:
        The round-trip result.

    """
    effective: list[list[float]] = []
    for row in rows:
        effective.append(
            [
                max(max(abs(v) for v in block) / 6.0, 1e-6)
                for _, block in _blocks(row, k)
            ]
        )
    global_scale = max(max(row) for row in effective)

    reconstructed: list[list[float]] = []
    scales: list[list[float]] = []
    underflowed = 0
    for row, row_effective in zip(rows, effective):
        out: list[float] = []
        row_scales: list[float] = []
        for (start, block), eff in zip(_blocks(row, k), row_effective):
            stored = e4m3_decode(e4m3_encode(eff / global_scale))
            if not e4m3_is_normal(stored):
                underflowed += 1
            row_scales.append(stored)
            scale = stored * global_scale
            codes = [_e2m1_encode(v / scale) for v in block]
            out.extend(_e2m1_decode(code) * scale for code in codes)
        reconstructed.append(out)
        scales.append(row_scales)
    return TensorResult(
        reconstructed, scales, global_scale=global_scale, underflowed_blocks=underflowed
    )


# ---------------------------------------------------------------------------
# Accounting
# ---------------------------------------------------------------------------


def snr_db(reference: Sequence[float], test: Sequence[float]) -> float:
    """Signal-to-noise ratio in dB between two equal-length sequences."""
    num = 0.0
    den = 0.0
    for ref, tst in zip(reference, test):
        num += ref * ref
        den += (ref - tst) ** 2
    if den == 0.0:
        return math.inf
    return 10.0 * math.log10(num / den)


def bits_per_value(
    element_bits: int,
    scale_bits: int,
    block: int,
    zero_point_bits: int = 0,
) -> float:
    """Total bits per stored value including the scale overhead.

    A per-tensor global scale is not included: it is one number for the whole
    tensor and costs no meaningful number of bits.

    Args:
        element_bits: Bits per element payload.
        scale_bits: Bits per stored scale.
        block: Elements sharing one scale.
        zero_point_bits: Bits per stored zero-point. The shipped
            ``int4_per_token_head`` format hides its zero-point in the low
            mantissa bits of the FP32 scale, so it passes 0 here.

    Returns:
        The total bits per value.

    """
    return element_bits + scale_bits / block + zero_point_bits / block


# ---------------------------------------------------------------------------
# Harness
# ---------------------------------------------------------------------------


@dataclass
class SchemeResult:
    """One scheme's outcome on one tensor.

    Attributes:
        name: The scheme name.
        snr: Round-trip SNR in dB, averaged over rows.
        bits: Total bits per value including scale overhead.
        notes: Free-text caveats.
        underflowed: Blocks whose stored scale fell out of the normal range.

    """

    name: str
    snr: float
    bits: float
    notes: str
    underflowed: int = 0


def _make_row(rng: random.Random, head_size: int, outlier: bool) -> list[float]:
    """Build one synthetic KV row, optionally with channel outliers."""
    row = [rng.gauss(0.0, 0.5) for _ in range(head_size)]
    if outlier:
        # KIVI's observation: a few channels carry much larger values.
        for channel in (3, 17):
            if channel < head_size:
                row[channel] *= 40.0
    return row


def run_comparison(
    head_size: int = 128,
    outlier: bool = True,
    seed: int = 42,
) -> list[SchemeResult]:
    """Compare the scale strategies on synthetic KV rows.

    Args:
        head_size: Values per row.
        outlier: Whether to scale two channels up by 40x.
        seed: RNG seed, so the numbers are reproducible.

    Returns:
        One result per scheme.

    """
    rng = random.Random(seed)
    rows = [_make_row(rng, head_size, outlier) for _ in range(64)]

    def measure(name, quantize, bits, notes) -> SchemeResult:
        result = quantize(rows)
        snrs = [snr_db(row, recon) for row, recon in zip(rows, result.reconstructed)]
        return SchemeResult(
            name,
            sum(snrs) / len(snrs),
            bits,
            notes,
            underflowed=result.underflowed_blocks,
        )

    return [
        measure(
            "int4_per_token_head (shipped)",
            quantize_int4_per_token_head,
            bits_per_value(4, 32, head_size),
            "asymmetric, one FP32 scale per row, zp in the scale mantissa",
        ),
        measure(
            "int4_fp16_block k=64",
            lambda r: quantize_int4_fp16_block(r, k=64),
            bits_per_value(4, 16, 64),
            "symmetric, one FP16 scale per 64 values, single level",
        ),
        measure(
            "int4_fp16_block k=32 (proposal)",
            lambda r: quantize_int4_fp16_block(r, k=32),
            bits_per_value(4, 16, 32),
            "symmetric, one FP16 scale per 32 values, single level",
        ),
        measure(
            "int4_fp16_block k=16",
            lambda r: quantize_int4_fp16_block(r, k=16),
            bits_per_value(4, 16, 16),
            "symmetric, one FP16 scale per 16 values, single level",
        ),
        measure(
            "int4_fp16_block k=32 asym",
            lambda r: quantize_int4_fp16_block(r, k=32, symmetric=False),
            bits_per_value(4, 16, 32, zero_point_bits=4),
            "asymmetric, one FP16 scale + 4-bit zp per 32 values",
        ),
        measure(
            "int4_fp16_block_g k=32 (NVFP4 structure)",
            quantize_int4_fp16_block_g,
            bits_per_value(4, 16, 32),
            "symmetric, FP16 block scale / FP32 per-tensor global",
        ),
        measure(
            "int4_e4m3_block k=16 (no global)",
            quantize_int4_e4m3_block,
            bits_per_value(4, 8, 16),
            "symmetric, one E4M3 scale per 16 values, single level",
        ),
        measure(
            "int4_e4m3_block_g k=16",
            quantize_int4_e4m3_block_g,
            bits_per_value(4, 8, 16),
            "symmetric, E4M3 block scale / FP32 global (NVFP4 geometry)",
        ),
        measure(
            "nvfp4 (E2M1 + E4M3/16)",
            quantize_nvfp4,
            bits_per_value(4, 8, 16),
            "NVIDIA NVFP4: E2M1 elements, E4M3 scale per 16, FP32 global",
        ),
    ]


def _self_check() -> None:
    """Assert the primitives behave as the README claims."""
    assert f16_round(0.1) == struct.unpack("<e", struct.pack("<e", 0.1))[0]
    assert f16_round(FP16_MAX * 2) == FP16_MAX

    assert e4m3_decode(0x7E) == E4M3_MAX
    assert e4m3_encode(E4M3_MAX) == 0x7E
    assert e4m3_encode(1e9) == 0x7E
    assert math.isnan(e4m3_decode(0x7F))

    # E4M3 round-trip is exact for every finite code. Negative zero is
    # canonicalised to positive zero by the encoder, which is harmless for
    # a KV cache.
    for code in range(256):
        value = e4m3_decode(code)
        if math.isnan(value):
            continue
        if value == 0.0:
            assert e4m3_decode(e4m3_encode(value)) == 0.0
            continue
        assert e4m3_encode(value) == code, f"E4M3 round-trip failed at {code:#04x}"

    codes = list(range(16))
    assert unpack_int4(pack_int4(codes), 16) == codes

    # The proposal's stored scale really is FP16-representable.
    rows = [[0.5 * i for i in range(64)]]
    result = quantize_int4_fp16_block(rows, k=32)
    assert len(result.scales[0]) == 2
    for scale in result.scales[0]:
        assert f16_round(scale) == scale

    # The global-scale variant reconstructs the same values as the single
    # level one when the global is the tensor maximum.
    single = quantize_int4_fp16_block(rows, k=32)
    with_global = quantize_int4_fp16_block_g(rows, k=32)
    assert with_global.global_scale == max(single.scales[0])


def main() -> None:
    """Run the self-check and print the comparison tables."""
    _self_check()
    head_size = 128

    for outlier in (True, False):
        label = "with two x40 outlier channels" if outlier else "no outlier channels"
        results = run_comparison(head_size=head_size, outlier=outlier)
        print(f"=== head_size={head_size}, {label} ===")
        print("64 rows, N(0, 0.5). SNR is round-trip, averaged over rows.")
        print("These are arithmetic results, not V620 measurements.\n")
        header = f"{'scheme':<38} {'SNR dB':>8} {'bits/val':>10} {'subnorm':>9}"
        print(header)
        print("-" * len(header))
        for result in results:
            print(
                f"{result.name:<38} {result.snr:>8.3f} {result.bits:>10.3f} "
                f"{result.underflowed:>9d}"
            )
        print()
        for result in results:
            print(f"  {result.name}: {result.notes}")
        print()

    print("Self-check passed.")


if __name__ == "__main__":
    main()

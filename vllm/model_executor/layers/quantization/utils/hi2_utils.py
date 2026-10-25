# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Host-side helpers for HadamardINT2-FP16 (HI2, spec v3.2) tensors.

An HI2 tensor ``W[R, K]`` is a dict of flat buffers (spec 2.4), keyed by the
checkpoint field names without the ``hi2_`` prefix:

==============  =======  ===================================================
field           dtype    shape / meaning
==============  =======  ===================================================
``codes``       int32    ``[sum_r K*nbits_r/32]``, codes LSB-first per word
``code_off``    int32    ``[R+1]`` first word of each row
``blk_params``  uint8    ``[2*sum_r K/B_r]`` ``(lo_u8, step_u8)`` per block
``param_off``   int32    ``[R+1]`` first block of each row
``row_consts``  fp16     ``[R, 3]`` ``lo_min, lo_scale, step_scale``
``row_nbits``   uint8    ``[R]`` 2, 4 or 8 (INT8 row class, spec 11.6)
``row_block``   uint8    ``[R]`` 16, 32 or 64
``had``         uint8    ``[K/16]`` order-16 Hadamard flag per column group
``col_scale``   fp16     ``[K]`` optional smoothing vector ``s``
==============  =======  ===================================================

The stored basis is ``H (W * s)`` and the activation ``H (x / s)`` (spec 2.3).
These helpers slice and concatenate tensors for tensor parallelism and fused
layers, validate them, and give a pure-PyTorch dequantization that is the
fallback where the RDNA2 kernels are not built.
"""

import math

import torch

GROUP = 16
BLOCK_SIZES = (16, 32, 64)
NBITS = (2, 4, 8)
K_MULTIPLE = 64

FIELDS = (
    "codes",
    "code_off",
    "blk_params",
    "param_off",
    "row_consts",
    "row_nbits",
    "row_block",
    "had",
)
OPTIONAL_FIELDS = ("col_scale",)
ALL_FIELDS = FIELDS + OPTIONAL_FIELDS

Hi2 = dict[str, torch.Tensor]


def num_rows(t: Hi2) -> int:
    return int(t["row_nbits"].numel())


def num_cols(t: Hi2) -> int:
    return int(t["had"].numel()) * GROUP


def canonicalize(t: Hi2) -> Hi2:
    """Copy into the dtypes and flat shapes the kernels expect (CPU)."""
    out: Hi2 = {
        "codes": t["codes"].reshape(-1).to(torch.int32),
        "code_off": t["code_off"].reshape(-1).to(torch.int64).to(torch.int32),
        "blk_params": t["blk_params"].reshape(-1).to(torch.uint8),
        "param_off": t["param_off"].reshape(-1).to(torch.int64).to(torch.int32),
        "row_consts": t["row_consts"].reshape(-1, 3).to(torch.float16),
        "row_nbits": t["row_nbits"].reshape(-1).to(torch.uint8),
        "row_block": t["row_block"].reshape(-1).to(torch.uint8),
        "had": t["had"].reshape(-1).to(torch.uint8),
    }
    if t.get("col_scale") is not None:
        out["col_scale"] = t["col_scale"].reshape(-1).to(torch.float16)
    return out


def validate(t: Hi2, k: int | None = None) -> None:
    """Check the buffers against each other. The kernels trust the offsets,
    so this must pass before a tensor reaches the device."""
    r = num_rows(t)
    kk = num_cols(t)
    if k is not None and kk != k:
        raise ValueError(f"HI2 tensor has K={kk}, layer expects K={k}")
    if kk % K_MULTIPLE:
        raise ValueError(f"HI2: K={kk} is not a multiple of {K_MULTIPLE}")
    nb = t["row_nbits"].to(torch.int64)
    bl = t["row_block"].to(torch.int64)
    if t["row_block"].numel() != r or t["row_consts"].shape != (r, 3):
        raise ValueError("HI2: row_block/row_consts do not have R rows")
    if not bool(torch.isin(nb, torch.tensor(NBITS)).all()):
        raise ValueError(f"HI2: row_nbits outside {NBITS}: {sorted(set(nb.tolist()))}")
    if not bool(torch.isin(bl, torch.tensor(BLOCK_SIZES)).all()):
        raise ValueError(
            f"HI2: row_block outside {BLOCK_SIZES}: {sorted(set(bl.tolist()))}"
        )
    for name, per_row, buf, scale in (
        ("code_off", kk * nb // 32, t["codes"], 1),
        ("param_off", kk // bl, t["blk_params"], 2),
    ):
        off = t[name].to(torch.int64)
        if off.numel() != r + 1 or int(off[0]) != 0:
            raise ValueError(f"HI2: {name} must have R+1 entries starting at 0")
        if not torch.equal(off[1:] - off[:-1], per_row):
            raise ValueError(f"HI2: {name} does not match row_nbits/row_block")
        if int(off[-1]) * scale != buf.numel():
            raise ValueError(f"HI2: {name}[-1] does not match the buffer length")
    if t.get("col_scale") is not None and t["col_scale"].numel() != kk:
        raise ValueError("HI2: col_scale must have K entries")


def select_rows(t: Hi2, r0: int, r1: int) -> Hi2:
    """Rows ``[r0, r1)``: a contiguous span of every row buffer."""
    co = t["code_off"].to(torch.int64)
    po = t["param_off"].to(torch.int64)
    out = dict(t)
    out["codes"] = t["codes"][int(co[r0]) : int(co[r1])]
    out["code_off"] = (co[r0 : r1 + 1] - co[r0]).to(torch.int32)
    out["blk_params"] = t["blk_params"][2 * int(po[r0]) : 2 * int(po[r1])]
    out["param_off"] = (po[r0 : r1 + 1] - po[r0]).to(torch.int32)
    for f in ("row_consts", "row_nbits", "row_block"):
        out[f] = t[f][r0:r1]
    return out


def _same_transform(a: Hi2, b: Hi2) -> bool:
    ca, cb = a.get("col_scale"), b.get("col_scale")
    if (ca is None) != (cb is None):
        return False
    if ca is not None and not torch.equal(ca, cb):
        return False
    return torch.equal(a["had"], b["had"])


def concat_rows(parts: list[Hi2]) -> Hi2:
    """Stack tensors with the same K and the same transform along R."""
    if len(parts) == 1:
        return dict(parts[0])
    first = parts[0]
    for p in parts[1:]:
        if num_cols(p) != num_cols(first) or not _same_transform(first, p):
            raise ValueError("HI2: concat_rows needs equal K and transform")
    out = dict(first)
    for f, off in (("codes", "code_off"), ("blk_params", "param_off")):
        bases, total = [], 0
        for p in parts:
            bases.append(p[off].to(torch.int64)[:-1] + total)
            total += int(p[off][-1])
        out[off] = torch.cat(bases + [torch.tensor([total])]).to(torch.int32)
        out[f] = torch.cat([p[f] for p in parts])
    for f in ("row_consts", "row_nbits", "row_block"):
        out[f] = torch.cat([p[f] for p in parts])
    return out


def select_cols(t: Hi2, k0: int, k1: int) -> Hi2:
    """Columns ``[k0, k1)`` of every row (row-parallel tensor parallelism).
    Both bounds must be multiples of 64, so no word, block or Hadamard group
    is cut."""
    if k0 % K_MULTIPLE or k1 % K_MULTIPLE or not 0 <= k0 < k1 <= num_cols(t):
        raise ValueError(f"HI2: column slice [{k0}, {k1}) is not 64-aligned")
    r = num_rows(t)
    nb = t["row_nbits"].to(torch.int64)
    bl = t["row_block"].to(torch.int64)
    co = t["code_off"].to(torch.int64)
    po = t["param_off"].to(torch.int64)
    w_lo, w_n = k0 * nb // 32, (k1 - k0) * nb // 32
    b_lo, b_n = k0 // bl, (k1 - k0) // bl
    new_co = torch.zeros(r + 1, dtype=torch.int64)
    new_co[1:] = torch.cumsum(w_n, 0)
    new_po = torch.zeros(r + 1, dtype=torch.int64)
    new_po[1:] = torch.cumsum(b_n, 0)
    codes = torch.empty(int(new_co[-1]), dtype=t["codes"].dtype)
    params = torch.empty(2 * int(new_po[-1]), dtype=t["blk_params"].dtype)
    pv = t["blk_params"].view(-1, 2)
    nv = params.view(-1, 2)
    for nbits in sorted(set(nb.tolist())):
        for bs in sorted(set(bl.tolist())):
            rows = ((nb == nbits) & (bl == bs)).nonzero().flatten()
            if rows.numel() == 0:
                continue
            wn, bn = int(w_n[rows[0]]), int(b_n[rows[0]])
            src = (co[rows] + w_lo[rows]).unsqueeze(1) + torch.arange(wn)
            dst = new_co[rows].unsqueeze(1) + torch.arange(wn)
            codes[dst.reshape(-1)] = t["codes"][src.reshape(-1)]
            src = (po[rows] + b_lo[rows]).unsqueeze(1) + torch.arange(bn)
            dst = new_po[rows].unsqueeze(1) + torch.arange(bn)
            nv[dst.reshape(-1)] = pv[src.reshape(-1)]
    out = dict(t)
    out["codes"] = codes
    out["code_off"] = new_co.to(torch.int32)
    out["blk_params"] = params
    out["param_off"] = new_po.to(torch.int32)
    out["had"] = t["had"][k0 // GROUP : k1 // GROUP]
    if t.get("col_scale") is not None:
        out["col_scale"] = t["col_scale"][k0:k1]
    return out


def needs_transform(t: Hi2) -> bool:
    return t.get("col_scale") is not None or bool(t["had"].any())


def same_transform(a: Hi2, b: Hi2) -> bool:
    return _same_transform(a, b)


def _hadamard16(dtype: torch.dtype, device) -> torch.Tensor:
    h = torch.ones(1, 1, dtype=torch.float64)
    while h.shape[0] < GROUP:
        h = torch.cat([torch.cat([h, h], 1), torch.cat([h, -h], 1)], 0)
    return (h / math.sqrt(GROUP)).to(dtype=dtype, device=device)


def act_transform(x: torch.Tensor, t: Hi2) -> torch.Tensor:
    """``H16 (x / s)`` on the flagged groups, fp32 math, result in x.dtype."""
    k = x.shape[-1]
    xf = x.float()
    s = t.get("col_scale")
    if s is not None and s.numel() > 0:
        xf = xf / s.to(x.device).float()
    xg = xf.reshape(*x.shape[:-1], k // GROUP, GROUP)
    rot = xg @ _hadamard16(torch.float32, x.device)
    flags = t["had"].to(x.device).bool().reshape(-1, 1)
    return torch.where(flags, rot, xg).reshape(x.shape).to(x.dtype)


def unpack_codes(words: torch.Tensor, nbits: int) -> torch.Tensor:
    per_word = 32 // nbits
    w = words.to(torch.int64) & 0xFFFFFFFF
    mask = (1 << nbits) - 1
    c = torch.stack([(w >> (nbits * i)) & mask for i in range(per_word)], dim=-1)
    return c.reshape(*words.shape[:-1], -1)


def dequantize_stored(t: Hi2) -> torch.Tensor:
    """fp16 ``[R, K]`` reconstruction in the stored basis, bit-identical to
    the spec decode (2.2): float64 holds every intermediate exactly before the
    single roundings."""
    r, k = num_rows(t), num_cols(t)
    out = torch.zeros(r, k, dtype=torch.float16)
    nb = t["row_nbits"].to(torch.int64)
    bl = t["row_block"].to(torch.int64)
    co = t["code_off"].to(torch.int64)
    po = t["param_off"].to(torch.int64)
    pv = t["blk_params"].view(-1, 2)
    for nbits in sorted(set(nb.tolist())):
        for bs in sorted(set(bl.tolist())):
            rows = ((nb == nbits) & (bl == bs)).nonzero().flatten()
            if rows.numel() == 0:
                continue
            wr, nblk = k * nbits // 32, k // bs
            words = t["codes"][co[rows].unsqueeze(1) + torch.arange(wr)]
            codes = unpack_codes(words, nbits).reshape(rows.numel(), nblk, bs)
            p = pv[po[rows].unsqueeze(1) + torch.arange(nblk)]  # [r, nblk, 2]
            rc = t["row_consts"][rows].unsqueeze(1)  # [r, 1, 3]
            lo = (
                (p[..., 0].double() * rc[..., 1].double() + rc[..., 0].double())
                .float()
                .to(torch.float16)
            )
            step = (p[..., 1].float() * rc[..., 2].float()).to(torch.float16)
            w = lo.double().unsqueeze(-1) + codes.double() * step.double().unsqueeze(-1)
            out[rows] = w.to(torch.float16).reshape(-1, k)
    return out

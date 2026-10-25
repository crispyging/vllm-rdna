# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Kernel-level timing of hi2_linear_rdna2 against a dense fp16 F.linear of
the same shape (random HI2 tensors, ~30% INT4 rows, mixed blocks). This is a
kernel microbenchmark, not a serving benchmark.

  python tools/rdna/hi2/bench_kernel.py
"""

import functools

import torch
import torch.nn.functional as F

from vllm import _custom_ops as ops
from vllm.model_executor.layers.quantization.utils import hi2_utils as hu

SHAPES = {  # Qwen3.5-4B projections (R, K)
    "gate_up": (2 * 9216, 2560),
    "down": (2560, 9216),
    "in_proj_qkvz": (12288, 2560),
}


def make(r: int, k: int, int4_frac: float = 0.3, seed: int = 0) -> dict:
    g = torch.Generator().manual_seed(seed)
    nb = torch.where(torch.rand(r, generator=g) < int4_frac, 4, 2)
    bl = torch.tensor([16, 32, 64])[torch.randint(3, (r,), generator=g)]
    words = k * nb // 32
    blocks = k // bl
    co = torch.zeros(r + 1, dtype=torch.int64)
    co[1:] = torch.cumsum(words, 0)
    po = torch.zeros(r + 1, dtype=torch.int64)
    po[1:] = torch.cumsum(blocks, 0)
    return {
        "codes": torch.randint(-(2**31), 2**31 - 1, (int(co[-1]),), dtype=torch.int32),
        "code_off": co.to(torch.int32),
        "blk_params": torch.randint(0, 256, (2 * int(po[-1]),), dtype=torch.uint8),
        "param_off": po.to(torch.int32),
        "row_consts": (torch.rand(r, 3) * 1e-2).to(torch.float16),
        "row_nbits": nb.to(torch.uint8),
        "row_block": bl.to(torch.uint8),
        "had": torch.ones(k // 16, dtype=torch.uint8),
        "col_scale": (0.5 + torch.rand(k)).to(torch.float16),
    }


def timeit(fn, iters: int = 200) -> float:
    for _ in range(10):
        fn()
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(iters):
        fn()
    end.record()
    torch.accelerator.synchronize()
    return start.elapsed_time(end) / iters * 1e3  # us


def main() -> None:
    for name, (r, k) in SHAPES.items():
        t = make(r, k)
        hu.validate(t)
        d = {f: v.cuda() for f, v in t.items()}
        qbytes = sum(v.numel() * v.element_size() for v in d.values())
        w = torch.randn(r, k, dtype=torch.float16, device="cuda")
        for m in (1, 4, 8, 64):
            x = torch.randn(m, k, dtype=torch.float16, device="cuda")

            args = (
                x, d["col_scale"], d["had"], d["codes"], d["code_off"],
                d["blk_params"], d["param_off"], d["row_consts"],
                d["row_nbits"], d["row_block"], True, 8,
            )  # fmt: skip
            t_q = timeit(functools.partial(ops.hi2_linear_rdna2, *args))
            t_d = timeit(functools.partial(F.linear, x, w))
            print(
                f"{name:13s} R={r:6d} K={k:5d} M={m:3d}  hi2 {t_q:8.1f} us "
                f"({qbytes / t_q / 1e3:6.1f} GB/s of HI2 bytes)  dense fp16 "
                f"{t_d:8.1f} us ({r * k * 2 / t_d / 1e3:6.1f} GB/s)"
            )


if __name__ == "__main__":
    main()

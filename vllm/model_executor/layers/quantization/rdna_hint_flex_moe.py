# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""rdna_hint_flex routed experts.

Each expert's gate, up and down projection is its own rdna_hint_flex tensor
(``experts.E.{gate,up,down}_proj.hi2_<field>``). After loading, the experts
are stacked row-wise into one ragged tensor per stage (gate/up: expert e owns
rows ``[2Ne, 2N(e+1))``, gate first; down: ``[He, H(e+1))``), and their
smoothing vectors and Hadamard flags into ``[E, K]`` / ``[E, K/16]`` tables,
because the quantizer calibrates each expert on its own routed tokens.

Gate and up of one expert must share the transform (they see the same
tokens, so the quantizer derives the same vector); a checkpoint where they
differ is rejected.

The forward is ``rdna_hint_flex_moe_rdna2``: one transform per (token, expert)
pair, a GEMV per pair for gate/up, SiLU-and-mul, the down-projection GEMV with
the routing-weighted top-k sum done in the kernel (no atomics). Fixed shapes,
so it is cudagraph-capturable. F16 and I8 compute modes (the config's
compute_mode).
"""

import os
from typing import TYPE_CHECKING

import torch
import torch.nn.functional as F

from vllm import _custom_ops as ops
from vllm.model_executor.layers.fused_moe.config import FusedMoEQuantConfig
from vllm.model_executor.layers.fused_moe.fused_moe_method_base import (
    FusedMoEMethodBase,
)
from vllm.model_executor.layers.quantization.utils import rdna_hint_flex_utils as hu
from vllm.model_executor.utils import set_weight_attrs

if TYPE_CHECKING:
    from vllm.model_executor.layers.fused_moe.routed_experts import RoutedExperts

_SHARDS = ("w1", "w3", "w2")
# Batches above this many tokens use the grouped (per-expert GEMM) path;
# RDNA_HINT_FLEX_MOE_GROUPED_MIN overrides it (A/B only).
GROUPED_MIN_TOKENS = int(os.environ.get("RDNA_HINT_FLEX_MOE_GROUPED_MIN", "16"))
MOE_BLOCK_M = 64  # must match kMoeBlockM in rdna_hint_flex_rdna2.cu


def stack_experts(
    experts: list[list[hu.Hi2]],
    k: int,
    device: torch.device | str,
) -> tuple[hu.Hi2, bool]:
    """Stack per-expert tensors (each a list of row blocks, in order) into one
    ragged tensor plus ``col_scale [E, K]`` (empty if no expert has one) and
    ``had [E, K/16]``; returns it with the "needs a transform" flag."""
    blocks = [b for e in experts for b in e]
    t = hu.concat_rows(blocks, check_transform=False)
    firsts = [e[0] for e in experts]
    has_scale = any(f.get("col_scale") is not None for f in firsts)
    if has_scale:
        scales = [
            f["col_scale"]
            if f.get("col_scale") is not None
            else torch.ones(k, dtype=torch.float16)
            for f in firsts
        ]
        col_scale = torch.stack(scales).contiguous()
    else:
        col_scale = torch.empty(0, dtype=torch.float16)
    had = torch.stack([f["had"] for f in firsts]).contiguous()
    out = {f: t[f].contiguous().to(device) for f in hu.FIELDS if f != "had"}
    out["col_scale"] = col_scale.to(device)
    out["had"] = had.to(device)
    return out, has_scale or bool(had.any())


class RdnaHintFlexMoEMethod(FusedMoEMethodBase):
    def __init__(self, moe, compute_mode: int = 0) -> None:
        super().__init__(moe)
        # 0 = F16, 1 = I8 for decode-sized batches, 2 = I8 always (see
        # COMPUTE_MODES in rdna_hint_flex.py).
        self.compute_mode = compute_mode

    def get_fused_moe_quant_config(
        self, layer: "RoutedExperts"
    ) -> FusedMoEQuantConfig | None:
        return None

    def create_weights(
        self,
        layer: "RoutedExperts",
        num_experts: int,
        hidden_size: int,
        intermediate_size_per_partition: int,
        params_dtype: torch.dtype,
        **extra_weight_attrs,
    ):
        del params_dtype
        if hidden_size % hu.K_MULTIPLE or intermediate_size_per_partition % 64:
            raise ValueError(
                "rdna_hint_flex MoE needs hidden and intermediate (per rank) "
                f"multiples of 64, got {hidden_size}, "
                f"{intermediate_size_per_partition}"
            )
        loader = extra_weight_attrs.get("weight_loader")
        # The ragged buffers are collected by absorb(); these placeholders own
        # the loader, as for EXL3.
        for name in ("w13_weight", "w2_weight"):
            dummy = torch.nn.Parameter(
                torch.empty(num_experts, 1, dtype=torch.uint8), requires_grad=False
            )
            if loader is not None:
                set_weight_attrs(dummy, {"weight_loader": loader})
            layer.register_parameter(name, dummy)
        layer._rdna_hint_flex_moe_loaded = {}
        self._num_experts = num_experts
        self._hidden = hidden_size
        self._inter = intermediate_size_per_partition

    def absorb(self, layer, loaded, weight_name, shard_id, expert_id) -> None:
        field = weight_name.rsplit(".", 1)[-1].removeprefix("hi2_")
        if field not in hu.ALL_FIELDS:
            raise ValueError(f"rdna_hint_flex MoE: unexpected tensor {weight_name}")
        entry = layer._rdna_hint_flex_moe_loaded.setdefault((expert_id, shard_id), {})
        entry[field] = loaded.detach().to("cpu").clone()

    def process_weights_after_loading(self, layer: "RoutedExperts") -> None:
        loaded = layer._rdna_hint_flex_moe_loaded
        tp = int(layer.moe_config.tp_size)
        rank = int(layer.moe_config.tp_rank)
        n, h = self._inter, self._hidden
        device = layer.w13_weight.device
        w13_experts: list[list[hu.Hi2]] = []
        w2_experts: list[list[hu.Hi2]] = []
        for e in range(self._num_experts):
            got = {}
            for shard in _SHARDS:
                buf = loaded.get((e, shard))
                if buf is None or set(hu.FIELDS) - set(buf):
                    raise ValueError(
                        f"rdna_hint_flex MoE: expert {e} shard {shard} incomplete"
                    )
                t = hu.canonicalize(buf)
                hu.validate(t)
                got[shard] = t
            gate, up, down = got["w1"], got["w3"], got["w2"]
            if not hu.same_transform(gate, up):
                raise ValueError(
                    f"rdna_hint_flex MoE: expert {e} gate and up have different "
                    "smoothing/rotation"
                )
            if hu.num_rows(gate) != n * tp or hu.num_rows(up) != n * tp:
                raise ValueError(f"rdna_hint_flex MoE: expert {e} gate/up rows")
            if tp > 1:
                gate = hu.select_rows(gate, rank * n, (rank + 1) * n)
                up = hu.select_rows(up, rank * n, (rank + 1) * n)
                down = hu.select_cols(down, rank * n, (rank + 1) * n)
            hu.validate(gate, h)
            hu.validate(down, n)
            if hu.num_rows(down) != h:
                raise ValueError(f"rdna_hint_flex MoE: expert {e} down rows")
            w13_experts.append([gate, up])
            w2_experts.append([down])
        layer._rdna_hint_flex_w13, layer._rdna_hint_flex_w13_tr = stack_experts(
            w13_experts, h, device
        )
        layer._rdna_hint_flex_w2, layer._rdna_hint_flex_w2_tr = stack_experts(
            w2_experts, n, device
        )
        del layer._rdna_hint_flex_moe_loaded

    def apply(
        self,
        layer: "RoutedExperts",
        x: torch.Tensor,
        topk_weights: torch.Tensor,
        topk_ids: torch.Tensor,
        shared_experts=None,
        shared_experts_input=None,
    ) -> torch.Tensor:
        del shared_experts, shared_experts_input
        act = str(getattr(layer, "activation", "silu")).lower()
        if "silu" not in act and "swiglu" not in act:
            raise NotImplementedError(f"rdna_hint_flex MoE: activation {act}")
        x2 = x.reshape(-1, x.shape[-1]).to(torch.float16)
        m = x2.shape[0]
        ids = topk_ids.reshape(m, -1).long()
        emap = getattr(layer, "expert_map", None)
        if m > GROUPED_MIN_TOKENS and self.compute_mode == 0:
            return self._apply_grouped(layer, x2, topk_ids.reshape(m, -1),
                                       topk_weights.reshape(m, -1), emap)  # fmt: skip
        if emap is not None:
            ids = emap[ids]
        return ops.rdna_hint_flex_moe_rdna2(
            x2,
            ids.to(torch.int32).contiguous(),
            topk_weights.reshape(m, -1).float().contiguous(),
            layer._rdna_hint_flex_w13,
            layer._rdna_hint_flex_w13_tr,
            layer._rdna_hint_flex_w2,
            layer._rdna_hint_flex_w2_tr,
            self._num_experts,
            self._inter,
            self.compute_mode,
        )

    def _apply_grouped(self, layer, x2, topk_ids, topk_weights, emap):
        """Prefill: pairs grouped by expert (moe_align_block_size, block 64) so
        each expert's weights are decoded once per 64 pairs, through the fused
        decode-in-GEMM kernel; per-pair transforms; weighted top-k sum."""
        from vllm.model_executor.layers.fused_moe.moe_align_block_size import (
            moe_align_block_size,
        )

        m, top_k = topk_ids.shape
        p = m * top_k
        n, h = self._inter, self._hidden
        flat = topk_ids.reshape(-1)
        num_experts = int(emap.shape[0]) if emap is not None else self._num_experts
        sorted_ids, expert_ids, ntpp = moe_align_block_size(
            flat.view(p, 1),
            MOE_BLOCK_M,
            num_experts,
            expert_map=emap,
            pad_sorted_ids=True,
            ignore_invalid_experts=True,
        )
        local = (emap[flat.long()] if emap is not None else flat).to(torch.int32)
        w13, w2 = layer._rdna_hint_flex_w13, layer._rdna_hint_flex_w2
        xp = torch.empty(p, h, dtype=torch.float16, device=x2.device)
        ops.rdna_hint_flex_moe_transform_rdna2(
            x2, top_k, local, w13, layer._rdna_hint_flex_w13_tr, xp
        )
        g13 = torch.empty(p, 2 * n, dtype=torch.float16, device=x2.device)
        ops.rdna_hint_flex_moe_gemm_rdna2(
            xp, sorted_ids, expert_ids, ntpp, w13, 2 * n, g13
        )
        hmid = F.silu(g13[:, :n]) * g13[:, n:]
        hp = torch.empty(p, n, dtype=torch.float16, device=x2.device)
        ops.rdna_hint_flex_moe_transform_rdna2(
            hmid, 1, local, w2, layer._rdna_hint_flex_w2_tr, hp
        )
        y = torch.empty(p, h, dtype=torch.float16, device=x2.device)
        ops.rdna_hint_flex_moe_gemm_rdna2(hp, sorted_ids, expert_ids, ntpp, w2, h, y)
        wts = topk_weights.float() * (local.view(m, top_k) >= 0)
        return (y.view(m, top_k, h).float() * wts.unsqueeze(-1)).sum(1).to(x2.dtype)


def moe_reference(
    x: torch.Tensor,
    topk_ids: torch.Tensor,
    topk_weights: torch.Tensor,
    gate: list[hu.Hi2],
    up: list[hu.Hi2],
    down: list[hu.Hi2],
) -> torch.Tensor:
    """Dense float64 reference: each expert dequantized to its original basis
    (smoothing and rotation undone), SiLU-and-mul, routing-weighted sum."""

    def dense(t: hu.Hi2) -> torch.Tensor:
        w = hu.dequantize_stored(t).double()
        k = w.shape[1]
        wg = w.reshape(-1, k // hu.GROUP, hu.GROUP)
        h16 = hu._hadamard16(torch.float64, "cpu")
        rot = wg @ h16
        w = torch.where(t["had"].bool().reshape(-1, 1), rot, wg).reshape(-1, k)
        if t.get("col_scale") is not None:
            w = w / t["col_scale"].double()
        return w

    xd = x.double().cpu()
    out = torch.zeros_like(xd)
    cache: dict[int, tuple[torch.Tensor, torch.Tensor, torch.Tensor]] = {}
    for tok in range(x.shape[0]):
        for k in range(topk_ids.shape[1]):
            e = int(topk_ids[tok, k])
            if e < 0:
                continue
            if e not in cache:
                cache[e] = (dense(gate[e]), dense(up[e]), dense(down[e]))
            wg, wu, wd = cache[e]
            hmid = F.silu(xd[tok] @ wg.T) * (xd[tok] @ wu.T)
            out[tok] += float(topk_weights[tok, k]) * (hmid @ wd.T)
    return out

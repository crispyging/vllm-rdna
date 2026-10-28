# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Opt-in worker extension for quiescent expert-copy A/Bs and cache counters.

Load with --worker-extension-cls tools.rdna.mono.expert_cache_worker.ExpertCacheWorker.
The development collective_rpc endpoint can then call rdna_expert_cache_control
with an optional copy_mode string (tensor/native). Call only between workloads.
Captured compute graphs, expert residency and all payload pointers stay intact.
"""

from typing import Any

import torch


class ExpertCacheWorker:
    """Development-only mixin for the existing worker-extension interface."""

    def rdna_expert_cache_control(self: Any, copy_mode: str = "") -> dict[str, Any]:
        """Inspect successful expert passes, optionally selecting their copy backend."""
        if copy_mode not in ("", "tensor", "native"):
            raise ValueError("copy_mode must be tensor or native")
        caches = []
        seen = set()
        for module in self.get_model().modules():
            cache = getattr(module, "_rdna_expert_cache", None)
            if cache is not None and id(cache) not in seen:
                seen.add(id(cache))
                caches.append(cache)
        if not caches:
            raise ValueError("this worker has no offloaded expert caches")
        op = None
        if copy_mode == "native":
            if torch.version.hip is None or any(
                cache.device.type != "cuda" for cache in caches
            ):
                raise ValueError("native expert copies require HIP")
            op = getattr(torch.ops._rocm_C, "rdna2_expert_cache_copy", None)
            if op is None:
                raise RuntimeError(
                    "rebuild the ROCm extension for native expert copies"
                )
        if copy_mode:
            if any(cache.policy._active is not None for cache in caches):
                raise RuntimeError("expert-copy switching requires released leases")
            torch.accelerator.synchronize()
            for cache in caches:
                cache._copy_op = op
        return {
            "caches": len(caches),
            "copy_modes": sorted(
                {
                    "native" if cache._copy_op is not None else "tensor"
                    for cache in caches
                }
            ),
            # Demands are distinct experts per successful bounded pass, not
            # routed token/expert pairs or a token-level cache hit rate.
            "expert_demands": sum(cache.expert_demands for cache in caches),
            "expert_misses": sum(cache.expert_misses for cache in caches),
            "committed_passes": sum(cache.committed_passes for cache in caches),
            "copied_bytes": sum(cache.copied_bytes for cache in caches),
            "native_copy_calls": sum(cache.native_copy_calls for cache in caches),
            "batch_copy_calls": sum(cache.batch_copy_calls for cache in caches),
            "map_uploads": sum(cache.map_uploads for cache in caches),
            "device_bytes": sum(cache.device_bytes for cache in caches),
            "host_bytes": sum(cache.host_bytes for cache in caches),
        }

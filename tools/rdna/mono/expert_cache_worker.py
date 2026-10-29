# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Opt-in worker extension for quiescent expert-copy A/Bs and cache counters.

Load with --worker-extension-cls tools.rdna.mono.expert_cache_worker.ExpertCacheWorker.
The development collective_rpc endpoint can then call rdna_expert_cache_control
with an optional copy_mode string (tensor/native). Call only between workloads.
Captured compute graphs, expert residency and all payload pointers stay intact.
"""

import sys
from typing import Any

import torch


class ExpertCacheWorker:
    """Development-only mixin for the existing worker-extension interface."""

    def rdna_kv_snapshot_layout(self: Any) -> dict[str, Any]:
        """Read layout capabilities only; no transfer or device synchronization."""
        state = sys.modules.get("vllm.distributed.kv_transfer.kv_transfer_state")
        connector = getattr(state, "_KV_CONNECTOR_AGENT", None)
        wrapper = getattr(connector, "connector_worker", None)
        worker = getattr(wrapper, "worker", None)
        config = getattr(getattr(self, "model_runner", None), "kv_cache_config", None)
        groups = getattr(config, "kv_cache_groups", ())
        spec = getattr(wrapper, "spec", None)
        physical = bool(getattr(worker, "supports_physical_page_snapshots", False))
        page_bytes = (
            getattr(spec, "cpu_page_size_per_worker", None)
            if physical and getattr(spec, "blocks_per_chunk", None) == 1
            else None
        )
        gpu_pages = getattr(worker, "_physical_page_count", 0)
        cpu_chunks = getattr(worker, "_physical_cpu_chunk_count", 0)
        live_configured = bool(getattr(wrapper, "_live_snapshots_enabled", False))
        return {
            "connector": type(connector).__name__,
            "worker": type(worker).__name__,
            "physical_page_transport": physical,
            "physical_page_count": gpu_pages,
            "cpu_chunk_count": cpu_chunks,
            "physical_page_bytes": page_bytes,
            "gpu_physical_bytes": None
            if page_bytes is None
            else gpu_pages * page_bytes,
            "cpu_physical_bytes": None
            if page_bytes is None
            else cpu_chunks * page_bytes,
            "blocks_per_chunk": getattr(spec, "blocks_per_chunk", None),
            "groups": [
                {
                    "spec": type(group.kv_cache_spec).__name__,
                    "layers": list(group.layer_names),
                }
                for group in groups
            ],
            "live_request_paging_configured": live_configured,
            "live_request_paging": live_configured and physical,
        }

    def rdna_expert_cache_control(
        self: Any,
        copy_mode: str = "",
        finite_checks: str = "",
        post_step_checks: str = "",
        transfer_timing: str = "",
        overlap: str = "",
    ) -> dict[str, Any]:
        """Inspect successful expert passes, optionally selecting their copy backend."""
        if copy_mode not in ("", "tensor", "native"):
            raise ValueError("copy_mode must be tensor or native")
        if finite_checks not in ("", "on", "off"):
            raise ValueError("finite_checks must be on or off")
        if post_step_checks not in ("", "collective", "wait", "off"):
            raise ValueError("post_step_checks must be collective, wait or off")
        if transfer_timing not in ("", "on", "off"):
            raise ValueError("transfer_timing must be on or off")
        if overlap not in ("", "on", "off"):
            raise ValueError("overlap must be on or off")
        options = getattr(
            sys.modules.get("vllm.models.qwen4_exp.amd.mono.rdna2"), "OPTS", None
        )
        if finite_checks and options is None:
            raise RuntimeError("Mono options are not loaded")
        failstop = sys.modules.get("vllm.models.qwen4_exp.amd.mono.rdna2.failstop")
        runner = getattr(self, "model_runner", None)
        if post_step_checks and (
            runner is None
            or failstop is None
            or not hasattr(failstop, "_USE_TP_COLLECTIVE")
        ):
            raise RuntimeError("worker has no post-step diagnostic controls")
        caches = []
        seen = set()
        names = {}
        model = self.get_model()
        modules = (
            model.named_modules()
            if hasattr(model, "named_modules")
            else ((str(index), module) for index, module in enumerate(model.modules()))
        )
        for name, module in modules:
            cache = getattr(module, "_rdna_expert_cache", None)
            if cache is not None and id(cache) not in seen:
                seen.add(id(cache))
                caches.append(cache)
                names[id(cache)] = name
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
        if copy_mode or finite_checks or post_step_checks or transfer_timing or overlap:
            if any(
                cache.policy._active is not None or cache._preload is not None
                for cache in caches
            ):
                raise RuntimeError("expert-copy switching requires released leases")
            torch.accelerator.synchronize()
            if copy_mode:
                for cache in caches:
                    cache._copy_op = op
            if transfer_timing:
                for cache in caches:
                    cache.set_transfer_timing(transfer_timing == "on")
            if overlap:
                for cache in caches:
                    cache.overlap = overlap == "on"
            if finite_checks:
                # Existing debug mode: excludes this run from performance claims.
                options.timing = finite_checks == "on"
            if post_step_checks:
                runner._mono_mtp_failstop = post_step_checks != "off"
                failstop._USE_TP_COLLECTIVE = post_step_checks != "wait"
        post_mode = None
        if runner is not None and failstop is not None:
            post_mode = (
                "collective"
                if getattr(failstop, "_USE_TP_COLLECTIVE", True)
                else "wait"
            )
            if not runner._mono_mtp_failstop:
                post_mode = "off"
        rank = getattr(
            self, "rank", getattr(getattr(self, "model_runner", None), "rank", None)
        )
        layers = [
            {
                "module": names[id(cache)],
                "rank": rank,
                "expert_bytes": cache._expert_bytes,
                "map_bytes": cache.slot_map.numel() * cache.slot_map.element_size(),
                "slots": cache.slot_count,
                "overlap": cache.overlap,
                "prefetch_bytes": cache.prefetch_bytes,
                "adopted_prefetch_bytes": cache.adopted_prefetch_bytes,
                "unused_prefetch_bytes": cache.unused_prefetch_bytes,
                "prefetch_failures": cache.prefetch_failures,
                "expert_demands": cache.expert_demands,
                "expert_misses": cache.expert_misses,
                "committed_passes": cache.committed_passes,
                "copied_bytes": cache.copied_bytes,
                "map_uploads": cache.map_uploads,
                "native_copy_calls": cache.native_copy_calls,
                "batch_copy_calls": cache.batch_copy_calls,
                "transfer_timing": cache.transfer_timing(),
            }
            for cache in caches
        ]
        return {
            "layers": layers,
            "caches": len(caches),
            "debug_timing": None if options is None else bool(options.timing),
            "post_step_checks": post_mode,
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

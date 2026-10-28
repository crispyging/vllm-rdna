# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Packed-bank contracts without loading the platform-dependent vLLM package."""

import ast
import builtins
import importlib.util
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch


@pytest.fixture
def cache_module(monkeypatch):
    root = Path(__file__).resolve().parents[4]
    directory = root / "vllm/model_executor/layers/fused_moe"
    policy_name = "vllm.model_executor.layers.fused_moe.rdna_expert_cache_policy"
    for name, path in (
        (policy_name, directory / "rdna_expert_cache_policy.py"),
        ("rdna_expert_cache_storage_test", directory / "rdna_expert_cache.py"),
    ):
        spec = importlib.util.spec_from_file_location(name, path)
        assert spec is not None and spec.loader is not None
        module = importlib.util.module_from_spec(spec)
        monkeypatch.setitem(sys.modules, name, module)
        spec.loader.exec_module(module)
    return module


def _bank(module, experts=256):
    return {
        name: torch.arange(experts * 6, dtype=torch.int32).reshape(experts, 2, 3)
        + offset * 10000
        for offset, name in enumerate(module.PAYLOAD_NAMES)
    }


@pytest.mark.parametrize("capacity", [1, 2, 3])
@pytest.mark.parametrize("history", [(0, 1, 2), (2, 1, 0)])
def test_cached_reduction_rounds_once_after_all_passes(
    cache_module, monkeypatch, capacity, history
):
    """FP16 group sums lose the half-unit; integer accumulation preserves it."""
    source = (
        Path(__file__).resolve().parents[4]
        / "vllm/model_executor/layers/quantization/compressed_tensors"
        / "compressed_tensors_moe/compressed_tensors_moe_wna16_rdna2.py"
    )
    tree = ast.parse(source.read_text())
    function = next(
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef) and node.name == "_apply_cached_moe"
    )
    function.decorator_list = []
    monkeypatch.setitem(
        sys.modules,
        "vllm.model_executor.layers.fused_moe.rdna_expert_cache",
        cache_module,
    )
    contributions = torch.tensor(
        [[2048 << 32], [-(2048 << 32)], [1 << 31]], dtype=torch.int64
    )

    def run_fixed_pass(x, weights, remapped, *, routed_accumulator, **kwargs):
        active = remapped.flatten().ge(0)
        assert routed_accumulator.dtype == torch.int64
        routed_accumulator[:, :-1].add_(contributions[active].sum(0))
        return routed_accumulator

    def cast(accumulator, output):
        output.copy_((accumulator[:, :-1].float() / (1 << 32)).to(output.dtype))

    namespace = {
        "torch": torch,
        "_rdna2_fused_moe": run_fixed_pass,
        "ops": SimpleNamespace(moe_fixedpoint_to_half_rdna2=cast),
    }
    exec(
        compile(ast.Module(body=[function], type_ignores=[]), str(source), "exec"),
        namespace,
    )
    cache = cache_module.PackedExpertCache(_bank(cache_module, 3), capacity, "cpu")
    for expert in history:
        cache.release(cache.ensure([expert]))
    layer = SimpleNamespace(
        _rdna_expert_cache=cache, expert_map=None, apply_router_weight_on_input=False
    )
    x = torch.zeros(1, 1, dtype=torch.half)
    ids = torch.tensor([[0, 1, 2]], dtype=torch.int32)
    result = torch.empty_like(x)
    namespace["_apply_cached_moe"](result, layer, x, torch.ones(1, 3), ids, None)
    assert result.item() == 0.5


def test_six_payloads_move_together_and_slots_keep_storage(cache_module):
    bank = _bank(cache_module)
    cache = cache_module.PackedExpertCache(bank, 2, "cpu")
    pointers = {name: t.data_ptr() for name, t in cache.payload.items()}
    for experts in ([255, 2], [2, 9], [17, 255]):
        lease = cache.ensure(experts)
        for expert, slot, _ in lease.bindings:
            for name in bank:
                assert torch.equal(cache.payload[name][slot], bank[name][expert])
        assert {name: t.data_ptr() for name, t in cache.payload.items()} == pointers
        assert cache.slot_map.ge(0).sum().item() == 2
        cache.release(lease)
    assert cache.logical_num_experts == 256
    assert cache.slot_count == 2
    expected = sum(t.numel() * t.element_size() for t in bank.values())
    assert cache.host_bytes == expected + 256 * 4
    assert cache.device_bytes == expected // 128 + 256 * 4


def test_hit_has_no_payload_copy_and_map_excludes_inactive_residents(cache_module):
    cache = cache_module.PackedExpertCache(_bank(cache_module), 2, "cpu")
    first = cache.ensure([3, 5])
    cache.release(first)
    copied = cache.copied_bytes
    second = cache.ensure([5])
    assert cache.copied_bytes == copied
    assert cache.slot_map[3].item() == -1
    assert cache.slot_map[5].item() >= 0
    assert cache.policy.mapping[3] >= 0
    cache.release(second)
    maps = cache.map_uploads
    third = cache.ensure([5])
    assert cache.map_uploads == maps
    cache.release(third)


def test_global_local_slot_mapping_preserves_padding_and_pass_coverage(cache_module):
    cache = cache_module.PackedExpertCache(_bank(cache_module, 4), 2, "cpu")
    expert_map = torch.tensor([-1, 0, -1, 1, 2, -1, 3], dtype=torch.int32)
    routed = torch.tensor([[1, 3, 4], [6, 1, -1], [0, 8, 3]])
    local = cache.local_ids(routed, expert_map)
    assert local.tolist() == [[0, 1, 2], [3, 0, -1], [-1, -1, 1]]
    ids = local.flatten().tolist()
    cache.policy.observe(ids)
    appearances = torch.zeros_like(local)
    for experts in cache.policy.plan_passes(ids):
        lease = cache.ensure(experts)
        remapped = cache.remap(local, lease)
        appearances += remapped.ge(0)
        for expert, slot, _ in lease.bindings:
            assert torch.equal(remapped.eq(slot), local.eq(expert))
        cache.release(lease)
    assert torch.equal(appearances, local.ge(0).long())
    assert cache.policy.frequencies == (2, 2, 1, 1)


def test_failed_copy_invalidates_victim_before_retry(cache_module, monkeypatch):
    cache = cache_module.PackedExpertCache(_bank(cache_module, 3), 1, "cpu")
    first = cache.ensure([0])
    cache.release(first)
    original = cache._copy

    def fail(bindings):
        original(bindings)
        raise RuntimeError("copy failed after overwriting the victim")

    monkeypatch.setattr(cache, "_copy", fail)
    with pytest.raises(RuntimeError, match="copy failed"):
        cache.ensure([1])
    assert cache.policy.mapping == {}
    monkeypatch.setattr(cache, "_copy", original)
    retry = cache.ensure([0])
    assert retry.bindings == ((0, 0, True),)
    cache.release(retry)


def test_host_preparation_keeps_shuffle_bounded_to_one_expert(cache_module):
    experts = 5
    packed = torch.arange(experts * 16 * 8, dtype=torch.int32).reshape(experts, 16, 8)
    scales = torch.arange(experts * 16 * 8, dtype=torch.float16).reshape(experts, 16, 8)
    layer = SimpleNamespace(
        w13_weight_packed=packed,
        w2_weight_packed=packed.clone(),
        w13_weight_scale=scales,
        w2_weight_scale=scales.clone(),
    )
    calls = []

    def shuffle(row, bits):
        assert row.shape == (8, 16)
        assert row.is_contiguous()
        calls.append(bits)
        row.add_(1)

    host = cache_module.prepare_host_payload(layer, 8, "cpu", shuffle)
    assert calls == [4] * (experts * 2)
    assert torch.equal(host["w13_weight_packed"], packed.transpose(1, 2) + 1)
    assert torch.equal(host["w2_weight_scale"], scales.transpose(1, 2))
    assert host["w13_qzeros"].shape == (experts, 8, 2)
    assert host["w13_qzeros"].eq(0x77777777).all()
    assert set(host) == set(cache_module.PAYLOAD_NAMES)
    assert all(t.device.type == "cpu" and t.is_contiguous() for t in host.values())
    assert packed[0, 0, 0].item() == 0


@pytest.mark.parametrize(
    ("active", "raw_capture", "expected"),
    [
        (None, False, False),
        (None, True, True),
        (SimpleNamespace(capturing_segment=False), True, False),
        (SimpleNamespace(capturing_segment=True), False, True),
    ],
)
def test_capture_guard_distinguishes_eager_breaks_from_raw_graphs(
    cache_module, monkeypatch, active, raw_capture, expected
):
    original = builtins.__import__

    def import_breakable(name, *args, **kwargs):
        if name == "vllm.compilation.breakable_cudagraph":
            return SimpleNamespace(
                BreakableCUDAGraphCapture=SimpleNamespace(current=lambda: active)
            )
        return original(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", import_breakable)
    monkeypatch.setattr(torch.cuda, "is_current_stream_capturing", lambda: raw_capture)
    assert cache_module.is_capturing_expert_work() is expected


def _loading_config(monkeypatch, slots):
    config = SimpleNamespace(
        additional_config={
            "rocm_mono_decode": {
                "expert_cache_slots": slots,
                "expert_cache_policy": "lru",
                "expert_cache_decay": 64,
            }
        },
        kernel_config=SimpleNamespace(monokernel=True),
        model_config=SimpleNamespace(
            architectures=["Qwen4ExpForConditionalGeneration"]
        ),
    )
    modules = {
        "vllm.config": SimpleNamespace(get_current_vllm_config_or_none=lambda: config),
        "vllm.platforms": SimpleNamespace(
            current_platform=SimpleNamespace(is_rocm=lambda: True)
        ),
        "vllm.platforms.rocm": SimpleNamespace(on_gfx10x=lambda: True),
    }
    original = builtins.__import__

    def import_config(name, *args, **kwargs):
        return modules[name] if name in modules else original(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", import_config)
    monkeypatch.setattr(torch, "get_default_device", lambda: torch.device("cuda:0"))
    cls = type("CompressedTensorsWNA16RDNA2MoEMethod", (), {})
    method = cls()
    method.symmetric = True
    method.num_bits = 4
    method.moe = SimpleNamespace(
        in_dtype=torch.float16,
        has_bias=False,
        is_lora_enabled=False,
        moe_parallel_config=SimpleNamespace(enable_eplb=False),
    )
    return config, method


def test_loading_context_stages_only_enabled_native_experts(cache_module, monkeypatch):
    config, method = _loading_config(monkeypatch, 2)
    layer = SimpleNamespace()
    with cache_module.cpu_weight_context(method, layer):
        assert torch.empty(1).device.type == "cpu"
    assert layer._rdna_expert_cache_slots == 2
    assert layer._rdna_expert_cache_device == torch.device("cuda:0")
    assert layer._rdna_expert_cache_policy == "lru"
    assert layer._rdna_expert_cache_decay == 64
    assert method.requires_device_loading is False
    config.additional_config["rocm_mono_decode"]["expert_cache_slots"] = 0
    untouched = SimpleNamespace()
    with cache_module.cpu_weight_context(method, untouched):
        pass
    assert vars(untouched) == {}


@pytest.mark.parametrize("unsupported", ["symmetric", "dtype", "eplb", "model"])
def test_loading_context_rejects_unsupported_expert_contracts(
    cache_module, monkeypatch, unsupported
):
    config, method = _loading_config(monkeypatch, 2)
    if unsupported == "symmetric":
        method.symmetric = False
    elif unsupported == "dtype":
        method.moe.in_dtype = torch.bfloat16
    elif unsupported == "eplb":
        method.moe.moe_parallel_config.enable_eplb = True
    else:
        config.model_config.architectures = ["OtherMoEForCausalLM"]
    layer = SimpleNamespace()
    with pytest.raises(ValueError), cache_module.cpu_weight_context(method, layer):
        pytest.fail("unsupported expert loading must not begin")
    assert vars(layer) == {}


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA/HIP")
def test_device_slot_reuse_waits_for_previous_consumer(cache_module):
    bank = _bank(cache_module, 4)
    cache = cache_module.PackedExpertCache(bank, 1, "cuda")
    assert all(t.is_pinned() for t in cache.host_payload.values())
    streams = (torch.cuda.Stream(), torch.cuda.Stream())
    copies = []
    maps = []
    pointers = {name: t.data_ptr() for name, t in cache.payload.items()}
    experts = (0, 1, 3, 0, 2, 1)
    for index, expert in enumerate(experts):
        with torch.cuda.stream(streams[index % 2]):
            lease = cache.ensure([expert])
            copies.append(cache.payload[cache_module.PAYLOAD_NAMES[0]][0].clone())
            maps.append(cache.slot_map.clone())
            cache.release(lease)
    for stream in streams:
        stream.synchronize()
    for result, mapping, expert in zip(copies, maps, experts, strict=True):
        assert torch.equal(result.cpu(), bank[cache_module.PAYLOAD_NAMES[0]][expert])
        expected = torch.full((4,), -1, dtype=torch.int32)
        expected[expert] = 0
        assert torch.equal(mapping.cpu(), expected)
    assert {name: t.data_ptr() for name, t in cache.payload.items()} == pointers

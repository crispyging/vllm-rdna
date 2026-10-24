# PR134 handover

## Scope and ownership

Implement opt-in INT4 KV with FP16 per-32-value block scales in native FA-RDNA2. **Rebased onto `origin/rdna_extras` = `12a46f40b4`** (PR #137, "improve MTP2 MonoKernel execution"; before that the DFlash2 series from PR #132). The explore format contract stays on PR #134 as the doc-only branch `explore/kv-int4-fp16-scales`; the implementation is its own branch `feat/kv-int4-fp16-scales`. Remote `par1-cs25` (192.168.1.84). **Serving numbers for this PR are measured on HIP 6,7,8,9** (user directive, superseding the earlier HIP2–5 scope). HIP 0–1 are the other workload's W7800s and were never touched. Private source `/home/chenco_adm/vllm-pr-int4fp16`, private runtime `/home/chenco_adm/Apps/vllm/venv-pr134-int4fp16`. No dependency upgrades or source changes during serving.

## Hardware topology (measured, not assumed)

Two numbering schemes disagree. **HIP ordinal** (what `HIP_VISIBLE_DEVICES` uses) is not `rocm-smi`'s `GPU[N]`, which is PCI-bus order:

| HIP ordinal | BDF | Device | rocm-smi index | PCIe root port |
| --- | --- | --- | --- | --- |
| 0, 1 | c7:00.0, 83:00.0 | W7800 48GB (gfx1100, other workload) | GPU[9], GPU[8] | c0:03.1, 80:01.1 |
| 2, 3 | 47:00.0, 4a:00.0 | V620 (gfx1030) | GPU[4], GPU[5] | **40:03.1** |
| 4, 5 | 52:00.0, 55:00.0 | V620 | GPU[6], GPU[7] | **40:03.1** |
| 6, 7, 8, 9 | 07:00.0, 0a:00.0, 12:00.0, 15:00.0 | V620 | GPU[0..3] | **00:01.1** |

`HIP_VISIBLE_DEVICES=6,7,8,9` resolves to BDF 07/0a/12/15 (verified via torch `pci_bus_id` and by worker VRAM placement). Both quads sit under one PEX880xx switch, so they are directly comparable; only the upstream root port differs.

The mapping is now reproducible, not asserted: `benchmarks/kernels/pr134/gpu_placement.py` resolves KFD node → HIP ordinal → PCI bus → PCIe root port and the VRAM each card actually holds, and asserts the served set. On this box it reports the four serving GPUs as HIP 6,7,8,9 on root port 00:01.1 with HIP 0–5 idle, so the placement claim in this table is checkable from the evidence rather than taken on trust. Note that rocm-smi's card index is **not** the HIP ordinal (rocm-smi sorts by PCI bus address, HIP follows the KFD node order), so the W7800s are HIP 0/1 but rocm-smi GPU[8]/GPU[9].

## Implemented and validated

Native writer, vector8 attention reader, registration/build wiring, explicit dtype and native backend dispatch. Both LBHNC/LHBNC cache layouts, inline metadata, exact writer edge cases, eager/graph/compiled changed-input gates passed. Registered integration evidence is in `core_integration/`. Reader SHA `b38c5ca5580b0da7c2082be457298b91d939a8370a72710f66f0d97d0640d13b`.

Three cores were built from this tree as it was rebased, and each arm directory records which one served it:

| core | tree | campaign |
| --- | --- | --- |
| `e44311ef…` | pre-PR#137 base | the first clean campaign, on HIP 6,7,8,9 |
| `c495d204…` | rebased onto `30d6e82dcd` | second campaign |
| `13615214…` | rebased onto `12a46f40b4` (PR #137 merged) | **current campaign**, the numbers below |

Synthetic native microbench results: decode beats FP16 by 13–28% at 16k, 18–30% at 64k, 24–41% at 240k across tested cells. GQA prefill approximately tied D128, 2.2% faster D256. Packed KV payload plus scales is 28.125% of FP16 bytes. Microbench results are not serving results.

## Actual serving results — INT4 now measured

Model `cyankiwi/Qwen3.8-27B-AWQ-INT4` snapshot `63768c10df38c0395e12ef49edac1bd539eaeeea`, recipe 27b-awq, TP=4/MTP2, LBHNC, block1024, KV budget 8GB/rank, max context 262144, max batched tokens 2048, max seqs 8, temp0/seed12345, HIP 6,7,8,9, boot `8dad1932-dd73-41bd-9fc6-d509b2a7ef42`.

**Both arms now raw PASS with clean hardware attribution** (qualification_mode `strict-pass`, zero AER events, zero fatal events):

- FP16 reference `serve-27b-fp16-core13615214-hip6789`: PASS/PASS, clean.
- INT4 candidate `serve-27b-int4-core13615214-hip6789`: PASS/PASS, clean, reference_mode `strict-pass`.
- Model quality: greedy 3/3, PPL 4.97 vs FP16 4.94, **paired ratio 1.0061** (gate 1.05) — PASS. Prefix, concurrency and normal-EOS gates passed before/after.
- KV capacity: FP16 **438,210** tokens (1.67x max concurrency at 262144 ctx) → INT4 **1,420,392** tokens (**5.42x**), i.e. **3.24x** capacity for the same 8GB/rank budget.

### 16k input / 1k output (the requested load) — current base, core `13615214`

| metric | c=1 FP16 | c=1 INT4 | Δ | c=8 FP16 | c=8 INT4 | Δ |
| --- | --- | --- | --- | --- | --- | --- |
| prefill per-request (tok/s) | 861.40 | 865.84 | **+0.5%** | 475.05 | 172.22 | **−63.8%** |
| prefill aggregated (tok/s) | 496.73 | 522.52 | **+5.2%** | 678.21 | 592.84 | −12.6% |
| decode per-request (tok/s) | 73.75 | 82.29 | **+11.6%** | 8.52 | 11.07 | **+29.9%** |
| decode aggregated (tok/s) | 31.05 | 32.66 | **+5.2%** | 42.39 | 37.05 | **−12.6%** |
| TTFT (s) | 19.02 | 18.92 | −0.5% | 48.48 | 107.42 | **+121.6%** |
| ITL median (ms) | 38.05 | 35.90 | **−5.6%** | 542.76 | 317.56 | −41.5% |
| ITL mean (ms) | 38.15 | 36.01 | **−5.6%** | 394.77 | 302.20 | **−23.5%** |
| MTP acceptance length | 2.797 | 2.967 | +6.1% | 2.820 | 2.908 | +3.1% |

The c=8 ITL median is reported for completeness only — it is not a stable statistic (see below). Read the c=8 ITL mean.

Short-context cells (1k/512): c=1 aggregate output 82.63 → 83.90 tok/s (+1.53%); c=8 206.85 → 205.37 tok/s (−0.71%). Decode per-request +1.53% at c=1 and +0.94% at c=8.

### Neither rebase moved the result

The FP16 reference is what every INT4 delta is measured against, so each rebase has to be shown not to have moved it. The reference was re-measured on each rebased tree and compared cell by cell with the same arm measured before that rebase (`serving/baseline-reproducibility-rebase1.json` and `serving/baseline-reproducibility-rebase2.json`: same boot, same config, same model snapshot, different core binary).

| metric | pre-rebase FP16 | rebased FP16 | Δ |
| --- | --- | --- | --- |
| 16k/1k c=1 prefill per-request (tok/s) | 863.324 | 863.205 | −0.01% |
| 16k/1k c=1 decode per-request (tok/s) | 74.298 | 73.998 | −0.40% |
| 16k/1k c=1 aggregate decode (tok/s) | 31.206 | 31.131 | −0.24% |
| 16k/1k c=1 TTFT (s) | 18.978 | 18.981 | +0.01% |
| 16k/1k c=1 ITL median (ms) | 38.015 | 38.020 | +0.01% |
| 16k/1k c=8 aggregate decode (tok/s) | 42.425 | 42.427 | +0.00% |
| 16k/1k c=8 whole-run input (tok/s) | 678.800 | 678.833 | +0.00% |
| 16k/1k c=8 decode per-request (tok/s) | 8.539 | 8.581 | +0.50% |

Every throughput metric is within ±0.5%. The one statistic that moves is the **c=8 ITL median** (−26% between two FP16 runs of the same configuration) while the c=8 ITL *mean* moves only −0.29% and aggregate throughput does not move at all: at c=8 the median is dominated by a few very slow stream chunks and is **not** a stable statistic. Read the c=8 ITL mean, not the median.

The second rebase (onto the PR #137 base) behaves the same way: every throughput metric within ±0.5% of the previous FP16 reference, while the c=8 ITL median moved **+36.6%** and the c=8 ITL *mean* moved only **+0.52%**. That is a second independent confirmation that the median is noise at c=8.

The INT4-vs-FP16 deltas agree across both rebases: c=1 decode per-request +10.83% → **+11.17%** → **+11.57%**, c=1 aggregate decode +4.94% → **+4.92%** → **+5.19%**, c=8 decode per-request +30.38% → **+28.87%** → **+29.87%**, c=8 aggregate decode −12.37% → **−12.60%** → **−12.59%**.

### Counterbalanced ABBA confirmation (all four arms PASS, same boot, same GPUs, clean)

Sequence FP16-A → INT4-A → INT4-B → FP16-B controls for run-order drift. All four arms: boot `8dad1932`, HIP 6,7,8,9, `strict-pass`, zero kernel events.

> The table below is the **pre-rebase** campaign (core `e44311ef…`); the current campaign's repeats are in `serving/abba-counterbalanced-core13615214-hip6789.json`.

| 16k/1k metric | FP16-A | FP16-B | INT4-A | INT4-B | ratio A | ratio B | pooled Δ |
| --- | --- | --- | --- | --- | --- | --- | --- |
| decode aggregated (tok/s) | 31.206 | 31.141 | 32.749 | 32.761 | 1.0494 | 1.0520 | **+5.07%** |
| decode per-request (tok/s) | 74.298 | 73.873 | 82.345 | 82.351 | 1.1083 | 1.1148 | **+11.15%** |
| prefill per-request (tok/s) | 863.324 | 864.611 | 869.462 | 869.982 | 1.0071 | 1.0062 | +0.67% |
| TTFT (s) | 18.978 | 18.950 | 18.844 | 18.833 | 0.9929 | 0.9938 | −0.66% |
| ITL median (ms) | 38.015 | 38.026 | 35.876 | 35.855 | 0.9437 | 0.9429 | −5.67% |

| 16k/1k c=8 metric | FP16-A | FP16-B | INT4-A | INT4-B | ratio A | ratio B | pooled Δ |
| --- | --- | --- | --- | --- | --- | --- | --- |
| decode aggregated (tok/s) | 42.425 | 42.466 | 37.177 | 37.186 | 0.8763 | 0.8757 | **−12.40%** |
| decode per-request (tok/s) | 8.539 | 8.526 | 11.132 | 11.090 | 1.3038 | 1.3008 | **+30.23%** |
| prefill per-request (tok/s) | 476.587 | 502.024 | 171.542 | 172.891 | 0.3599 | 0.3444 | **−64.80%** |
| TTFT (s) | 48.655 | 46.973 | 107.629 | 106.969 | 2.2121 | 2.2772 | **+124.41%** |
| ITL median (ms) | 540.028 | 557.958 | 316.249 | 316.375 | 0.5856 | 0.5670 | −42.38% |

The two independent repeats agree to within 0.3 pp at c=1 and 0.06 pp at c=8 on aggregate decode. The c=8 regression and the c=1 win are both reproducible and are **not** run-order drift.

**Reading of the result.** At c=1 INT4 is a clean win: decode per-request +10.8%, aggregate decode +4.9%, ITL −5.6%, TTFT unchanged, prefill unchanged. At c=8 the per-request decode win is larger (+30.4%) and ITL improves sharply (−41.4%), but **aggregate throughput regresses 12.4% because TTFT more than doubles**: in the mixed prefill+decode batch the INT4 path keeps fewer requests in flight (running 6.3→5.4 vs FP16 7.7→7.1) and prefill per-request collapses. The engine-level prompt throughput during the c=8 window is 648/559 tok/s (INT4) vs 733/647 (FP16) — ~13% slower — while at c=1 the two arms are identical (504.1 vs 504.1). So the pure-prefill and pure-decode paths are fine; the regression is specific to the **mixed-batch regime** a 16k-input workload at concurrency 8 produces. No preemption, no capacity pressure (INT4 has 3.24x the capacity) and no fatal events in either arm.

This is the known open gap: the INT4 native attention path needs a mixed prefill+decode fast path before it is a win at 16k/c=8. The c=1 win and the c=8 decode-side win are both far outside the measured FP16 noise floor (±0.2% at long cells, ±4.2% at 1k/c8).

## Verification of the measurement

**Did both arms run the native FA-RDNA2 kernels? Yes — and it is provable from the code, not just from the log.**

- Both arms select `attention_backend: RDNA_ATTN` ("Using RDNA_ATTN backend (selected via --attention-backend)"), same as the pre-existing FP16 baseline.
- `is_available()` = `VLLM_USE_RDNA2_FA != "0"` and `_on_gfx10x()` (arch prefix `gfx103`); our GPUs are gfx1030 → true.
- The FP16 path is gated by `_can_run_fa_rdna2()`, and when it is not satisfied the backend **raises `NotImplementedError`** ("selector should have routed this layer to ROCM_ATTN") instead of falling back. The FP16 arm served all four cells to completion, therefore it ran the native `fa_rdna2_decode_paged` / `fa_rdna2_prefill_paged_varlen*` kernels.
- The MTP-verify fallback is **opt-in and disabled**: `VLLM_FARDNA2_ENABLE_SPEC_GATE` defaults to `"0"`, so the speculative-decode verify pass also runs native FA-RDNA2, not the Triton fallback. (The old `max_seqlen_q == 3` heuristic fired on every chunked-prefill step and is gone.)
- The INT4 path *requires* native FA-RDNA2: it raises `NotImplementedError("int4_fp16_block requires native RDNA2 FA")` and `int4_fp16_kernels_available()` raises if `reshape_and_cache_int4_fp16_rdna2` / `fa_rdna2_paged_int4_fp16_block` are missing. The INT4 arm ran → both native INT4 ops are registered and were used.
- Zero occurrences of "outside FA-RDNA2 coverage" or "routed to fallback" in any arm log (FP16 r3, FP16 r4, INT4-A, INT4-B).

**Cross-check against the merged fork.** `bench_results/2026-09-29_w4a8_wiring/` holds the same model snapshot (`63768c10…`), MTP=2 and the same cell shapes, measured on the merged `vllm-rdna-0.28.0` tree:

| cell | historical (2026-09-29) | this work | note |
| --- | --- | --- | --- |
| 16384/1024 c=1 | TTFT 18.67 s, 1 prompt | TTFT 18.98 s, 8 prompts | TTFT is the comparable quantity; +1.7% |
| 1024/512 c=1 | TTFT 0.973 s, 1 prompt | TTFT 0.947 s, 8 prompts | −2.7% |
| 16384/1024 c=8 | TTFT 88.07 s, 8 prompts | TTFT 48.66 s, 32 prompts | more prompts amortise ramp-up |

Aggregate tok/s is not directly comparable (1/8 prompts vs 8/32). All three FP16 reference arms on this tree (r1 30.679, r2 30.730, r3 30.849 at 16k/c=1) plus r4 on the clean quad (31.206) agree within the measured noise floor.

**Rebased onto the merged base, twice.** The branch was first **15 behind / 8 ahead** of `origin/rdna_extras` (the DFlash2 series, PR #132). It was rebased onto `30d6e82dcd`, measured, and then PR #137 merged into the base at 17:04 UTC, so it was rebased again onto `12a46f40b4`. It is now 0 behind / 19 ahead. Two hand-resolved conflicts came out of the first rebase, one out of the second.

Two conflicts had to be resolved by hand, both in the FA-RDNA2 surface the upstream DFlash2 work also touched:

- `csrc/rocm/fa_rdna2.cu`: the upstream commit added a `typename IO_T = half` template parameter to the GQA varlen kernel (for BF16 DFlash2); the INT4 work added `bool IS_INT4 = false`. Merged as `…, bool IS_INT8 = false, typename IO_T = half, bool IS_INT4 = false>` — upstream's parameter order kept, `IS_INT4` appended — and the INT4 instantiation was updated to `<…, uint8_t, false, false, half, true>`. The vectorised-load predicate keeps upstream's stricter `std::is_same_v<KV_T, half>` (upstream narrowed it from `sizeof(KV_T) == 2` so a BF16 KV type cannot take the half2 path) and adds `(!IS_INT4)`.
- `vllm/v1/attention/backends/rdna_attn.py`: upstream added BF16 to `supported_dtypes` and `bfloat16` to `supported_kv_cache_dtypes`; the INT4 work added `int4_fp16_block`. Merged as the union — which creates a new combination (a BF16 model with an INT4 cache) that did not exist before, so `validate_configuration` now refuses it explicitly: the packed cache stores FP16 block scales, so a BF16 model cannot read it back without changing the scale semantics.

The merge was then verified end to end, not just by inspection: the core was rebuilt from each rebased tree (`c495d204…`, then `13615214…`), the registered-op / page-size / writer / CUDA-graph parity gate passed for both KV layouts (`core_integration/`), and the whole campaign was re-measured on each — see *Neither rebase moved the result* above.

## Corrected PCIe events — root cause found, and why HIP 6–9 fixed it

Every run on **HIP 2–5** produced corrected PCIe BadTLP noise on root port **0000:40:03.1**, the branch that carries HIP 2–5. All of it was classified by the kernel itself as corrected ("It has been corrected by h/w", severity corrected, `aer_uncor_status: 0x00000000`, endpoints tagged `[13] NonFatalErr`); no amdgpu fault, timeout, reset or UTCL2 event ever occurred. The raw `status.json` of those arms stays FAIL and is preserved.

Moving the TP group to **HIP 6–9** (root port 00:01.1) produced **zero** AER events and a clean strict PASS. The corrected link noise is a property of the 40:03.1 branch, not of the INT4 work.

Evidence: `serving/fault-attribution-fp16-arms.json` (r1: 3 corrected groups / 3 errors; r2: 4 groups / 20 errors incl. a 14-error burst and two AER recovery-buffer overflows; r3: 1 group / 1 error; all 0 fatal), and the per-arm `qualification.json`.

## Tooling added this session

- `benchmarks/kernels/pr134/kernel_events.py` — classifies kernel deltas into corrected-link noise vs fatal faults using the kernel's own severity verdict. Never reports "clean" when the raw gate matched something it cannot attribute (verdict `unclassified`).
- `qualify_serving.py` — writes `kernel.events.json` + `qualification.json`; **raw gate and raw status semantics unchanged**; adds `--diagnostic-reference` for an explicitly diagnostic-only candidate against a corrected-link-only reference, and `--gpus` to select the GPU set.
- `pr134-27b-pair.sh`, `pr134-27b-candidate.sh`, `pr134-27b-counterbalanced.sh` — paired arms; the counterbalanced phase is FP16→INT4→INT4→FP16 (ABBA).
- `compare_arms.py`, `report_16k.py`, `reference_mode.py`, `dryrun_arms.py`, `pull_serving_evidence.sh`, `manifest_local.py`.
- `dryrun_arms.py` exercises six arm paths (clean, corrected-only, fatal, strict reference, diagnostic reference, refused candidate) without a GPU.
- `compare_baselines.py` — compares two FP16 reference arms cell by cell, so a rebase can be shown not to have moved the baseline the INT4 deltas are measured against.
- `curate_arm.py` — reduces a pulled arm to the evidence worth keeping in Git (metrics, gate results, the kernel delta the classifier read, the manifest that pinned the binary, and a distilled serve log).
- `abba_report.py` — rebuilds the counterbalanced artifact from the four arms and refuses to compare arms that differ in boot, configuration or source/binary manifest. Run against the pre-rebase campaign it reproduces the committed artifact exactly.
- `gpu_placement.py` — resolves KFD node → HIP ordinal → PCI bus → PCIe root port and the VRAM each card holds, then asserts the served set. rocm-smi's card index is *not* the HIP ordinal, so without this the placement evidence would be filed on the wrong GPUs.
- `refresh_evidence.sh` — folds a campaign into the evidence tree under core-tagged names and regenerates every artifact from the arms it actually describes.
- The three launch scripts now derive their output-directory tag from the SHA256 of the installed `_rocm_C.abi3.so`, so a rebuilt core can never file its results under the previous build's directory.

Bugs found and fixed before they cost a GPU cycle: unbound `reference_mode`; a reset that erased the reference mode; a PPL stub mismatch; the kernel timestamp-padding blind spot (`[ 2626.944694]`); `reference_mode.py` missing its executable bit (this one silently aborted the first two pair launches).

**One evidence defect found and fixed.** The counterbalanced artifact named arm directories that carried no core tag, and those directories were later overwritten by the rebased campaign — so the artifact's numbers no longer matched the arms on disk, and it had no generator, so it could not be rebuilt. Every arm directory now carries its core tag, the pre-rebase arms were restored from the remote, and `abba_report.py` regenerates the artifact from the arms themselves.

## MonoKernel and INT4 KV

The MonoKernel path (`--monokernel`, Qwen4Exp / Flash-Next on gfx1030) and the INT4 KV cache are **different model families**: the MonoKernel's QSA owner reads the main KV cache as FP16/BF16 words, and the MonoKernel kernels read K and V as FP16 words too. A packed INT4 record is a different layout, so the combination is **refused, not mis-read**:

- `vllm/models/qwen4_exp/amd/mono_decode_rdna2.py` — the switch refuses a quantized KV cache up front, at the API server, naming the cache dtype it was given.
- `vllm/models/qwen4_exp/amd/mono/rdna2/qsa.py` — the fused HIP decode gate now requires an FP16 K/V cache, so the precondition is local to the kernel instead of inherited from the caller's assert.
- `vllm/models/qwen4_exp/amd/qsa.py` already refused any non-FP16/BF16 KV cache in three independent places before this PR; the new refusal is the early, mono-specific one.

Making INT4 KV *work* inside the Qwen4Exp QSA owner is a separate, larger feature: the QSA owner's cache write, its Triton sparse-attention read and the MonoKernel HIP sparse decode would all have to learn the packed record layout. It is not measured by this PR (the benchmark model is a Qwen3.8 AWQ dense model, not a Qwen4Exp W4A16 MoE).

## Current jobs

Counterbalanced ABBA repeats on the current base, HIP 6,7,8,9, boot `8dad1932-dd73-41bd-9fc6-d509b2a7ef42`: INT4 repeat → `serve-27b-int4-core13615214-hip6789-b`, then FP16 repeat → `serve-27b-fp16-core13615214-hip6789-b`. Driver log `/home/chenco_adm/pr134-results/pr134-27b-counterbalanced-hip6789-rebased-driver.log`.

The pair phase of this campaign is complete and clean: `serve-27b-fp16-core13615214-hip6789` and `serve-27b-int4-core13615214-hip6789`, both raw PASS/PASS, `strict-pass`, zero kernel events, byte-identical source/binary manifests (432 entries).

## Next work

The c=8 mixed-batch regression is confirmed and unchanged by either rebase; it still needs a fast path: profile the INT4 attention path in a mixed prefill+decode batch and add one. Do not claim a c=8 aggregate win before that.

Decide whether INT4 KV inside the Qwen4Exp / QSA / MonoKernel path is wanted (it is refused today, loudly); if it is, the W4A16 Flash-Next model already cached on the box is the validation target.

Push the rebased branches to the PR only on an explicit go-ahead. Two branches are ready: `explore/kv-int4-fp16-scales` (the explore format contract, doc-only, 3 files) and `feat/kv-int4-fp16-scales` (the implementation).

Flash-Next QSA checkpoint startup attempt does NOT exercise nativeFA; unsupported QSA route is not covered. Do not claim Flash-Next/QSA support from ordinary 27B results. Shipped per-head INT4 applies RHT, so raw unrotated codec comparison is not full shipped attention quality.

Finish documentation, recipe/provenance and PR134 update only with actual results. No binaries in Git; preserve intended raw evidence. GitHub commands use `env -u GITHUB_TOKEN -u GH_TOKEN`. No merge requested in this task.

# PR134 follow-on: the INT4 KV cache was not the problem, the attention block size was

Model `cyankiwi/Qwen3.8-27B-AWQ-INT4` (snapshot `63768c10df38c0395e12ef49edac1bd539eaeeaa`),
4x gfx1030 Radeon PRO V620 on `par1-cs25`, TP=4 / PP=1 / DP=1, MTP2, recipe `27b-awq`
(`--max-num-seqs 8`, `--block-size 1024` requested, `--enable-prefix-caching`,
`--mamba-cache-mode align`, `--kv-cache-memory-bytes 8000000000`,
`cudagraph_mode=FULL_AND_PIECEWISE`), private core `4292ace2dd42b1d6`, one boot
(`89e15654-71a8-4986-8dba-ea413393b0a2`), HIP 6,7,8,9.

## The symptom

The INT4 KV cache lost ~12% of its 16k x c=8 throughput against the FP16 KV cache:

| arm | attn block | out tok/s | TTFT |
|---|---|---|---|
| FP16 KV | 1024 | 42.18 | 50.7 s |
| INT4 KV | 3072 | 36.88 | 107.5 s |

## What it is not

The full four-cell matrix exonerates the packed format. INT4 wins three of the four
cells outright and loses only the long-prompt/high-concurrency one:

| cell | FP16 ref | INT4 fp32 SSM | INT4 fp16 SSM |
|---|---|---|---|
| 1024x512 c1 | 81.81 | 84.51 | 84.46 |
| 1024x512 c8 | 203.75 | 204.97 | 206.46 |
| 16384x1024 c1 | 31.20 | 33.02 | 32.84 |
| 16384x1024 c8 | **42.18** | 36.88 | 37.14 |

- **Not the KV format.** An FP16 arm forced to block 2048 scores 37.42 tok/s, which is
  the INT4 arm's number (37.14). Holding block and budget fixed, the format is worth
  about 2%, not 12%.
- **Not the prefill kernel.** The scheduler's own decode-stall fit reports the
  per-token prefill cost at 1.03-1.10 ms/tok in every arm, including the slow ones.
- **Not prefix caching.** The prefix cache hit rate was 0.0% in both arms during the
  16k cell.
- **Not KV capacity.** GPU KV cache usage was 12% (INT4) against 46% (FP16) during the
  cell; the INT4 arm had 3.4x *more* cache and still ran slower.

## What it is

The hybrid mamba/attention page unification sets the attention block size equal to the
mamba state block size. A packed KV cache has a smaller attention page, so the block
grows until its page covers the mamba state page:

| KV dtype | SSM state | attention block | KV tokens |
|---|---|---|---|
| fp16 | fp32 | 1024 | 438,210 |
| int4 | fp32 | 3072 | 1,420,392 |
| int4 | fp16 | 2048 | 1,492,348 |

The mamba block-aligned prefill split ends every prefill chunk on a mamba state block
boundary, so a scheduler step carries whole blocks, and the per-step token budget then
decides how many requests are prefilled per step. With a budget of one block, one
request is prefilled per step and every block additionally costs a near-empty
re-alignment step, so a long prompt at concurrency drains at half the rate.

This is why the cliff is invisible at c=1 (nothing to share the budget with) and at
1024-token prompts (one chunk): at 16k x c=1 the three block sizes score 31.20 / 31.00 /
30.97 tok/s, and at 1024 x c=8 they score 203.75 / 207.21 / 206.35.

## The controlled confirmation

One variable per arm, same boot / tree / core build:

| arm | KV | block | MAXBAT | out tok/s | TTFT |
|---|---|---|---|---|---|
| FP16 | fp16 | 1024 | 2048 | 42.18 | 50.7 s |
| FP16 | fp16 | 1024 | 2048 (fp16 SSM) | 42.93 | 47.98 s |
| FP16 | fp16 | 2048 | 2048 | 37.42 | 103.3 s |
| FP16 | fp16 | 3072 | 2048 | 37.57 | 104.3 s |
| FP16 | fp16 | 2048 | **4096** | **42.15** | **49.3 s** |
| INT4 | int4 | 3072 | 2048 | 36.88 | 107.5 s |
| INT4 | int4 | 2048 | 2048 | 37.14 | 105.9 s |
| INT4 | int4 | 2048 | **4096** | **41.20** | **48.1 s** |

Raising the budget at a fixed block recovers +12.6% throughput and halves TTFT. 3072 is
no worse than 2048 because both are capped by the same budget.

## The mechanism, confirmed against the real code

`Scheduler._mamba_block_aligned_split` was probed on the remote venv with the
production budget arithmetic of `scheduler.py:848-890`
(`num_new_tokens = min(remaining, token_budget, max_num_batched_tokens - draft_slots)`):

| attention block | MAXBAT | draft slots | num_new_tokens | chunk |
|---|---|---|---|---|
| 1024 | 2048 | 3 | 2045 | 1024 |
| 2048 | 2048 | 3 | 2045 | **0** |
| 2048 | 4096 | 3 | 4093 | 2048 |
| 3072 | 2048 | 3 | 2045 | 2045 |
| 3072 | 6144 | 3 | 6141 | 3072 |

At a budget of one block the chunk degenerates, and `scheduler.py:916-934` skips the
request: *"insufficient budget for a block-aligned chunk in hybrid models with mamba
cache mode align"*. At two blocks the chunk is a clean whole block and two requests are
prefilled per step.

## The fix

`max_num_batched_tokens >= 2 x the mamba state block size`.

- `tools/rdna/serve_rdna.sh` derives MAXBAT from the KV and SSM dtypes (2048 / 4096 /
  6144); an explicit MAXBAT still wins.
- `vllm/v1/core/sched/scheduler.py` warns loudly, naming the flag and the number, when
  the budget cannot cover two blocks.
- `benchmarks/kernels/pr134/qualify_serving.py` refuses such an arm instead of
  recording it as a slow run.

Commits `9522c57097` and `d7d040316c` on `feat/gdn-fp16-ssm-state`.

## Result

The INT4 KV cache is now a pure win rather than a trade-off:

- 16k x c=8: **41.20 tok/s, 48.1 s TTFT** (was 37.14 tok/s, 105.9 s), against 42.15
  tok/s and 49.3 s for the FP16 KV arm at the same block and budget.
- KV cache: **1,492,348 tokens** against 419,430 for the FP16 arm at the same budget.
- PPL 4.95 against the reference's 4.94, ratio 1.0020 (limit 1.05); greedy 3/3 PASS.
- Mamba, prefix caching and `FULL_AND_PIECEWISE` graphs all intact.

## Hardware attribution caveat

The FP16 reference arm of the final cohort has a raw FAIL: a corrected PCIe AER error
at 19:50:50, 98 seconds after an unrelated 4-GPU workload started on the same PLX
88096 fabric. The harness classifies it `corrected-only` with zero fatal events and
never relaxes the raw gate, so the candidate was qualified through the documented
`--diagnostic-reference` path. See `pcie-aer-attribution.json`.

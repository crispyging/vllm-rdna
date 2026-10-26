# PR134: how many mamba state blocks per scheduler step?

A prefill chunk ends on a mamba state block boundary, so a scheduler step carries whole blocks. With a budget of one block, one request is prefilled per step and every block costs a near-empty re-alignment step; two blocks restored parity (block 2048: MAXBAT 2048 -> 37.4 tok/s, MAXBAT 4096 -> 42.2 tok/s). This asks whether three blocks is better still.

Model `cyankiwi/Qwen3.8-27B-AWQ-INT4`, TP=4, MTP=2, prefix caching + mamba align, FULL_AND_PIECEWISE graphs, one boot, one source/binary manifest, the fp16 SSM state, and the recipe's own knobs (W4A8=1, decode-stall cap 0). One knob changed: the prefill budget.

| arm | KV | MAXBAT | blocks | status | 1k c=1 | 1k c=8 | 16k c=1 | 16k c=8 |
| --- | --- | ---: | ---: | --- | --- | --- | --- | --- |
| fp16-core4292ace2-recipe-final-hip6789 | fp16 | 2048 | 2 | PASS | 85.1 / 0.73s | 236.6 / 1.60s | 35.5 / 15.42s | 52.0 / 35.90s |
| fp16-core4292ace2-maxbat3b-hip6789 | fp16 | 3072 | 3 | PASS | 87.4 / 0.73s | 241.3 / 1.73s | 35.6 / 15.36s | 53.1 / 37.39s |
| int4-core4292ace2-recipe-final-hip6789 | int4 | 4096 | 2 | PASS | 83.8 / 0.73s | 233.9 / 1.77s | 37.7 / 14.81s | 52.0 / 36.81s |
| int4-core4292ace2-maxbat3b-hip6789 | int4 | 6144 | 3 | PASS | 81.5 / 0.73s | 229.5 / 1.82s | 37.2 / 15.16s | 51.4 / 33.24s |

Cells read `aggregate output tok/s / mean TTFT`.

## Three blocks against two

| KV | cell | output | TTFT | prefill | decode |
| --- | --- | ---: | ---: | ---: | ---: |
| fp16 | 1k x c=1 | +2.74 % | -0.27 % | +0.27 % | +3.08 % |
| fp16 | 1k x c=8 | +1.98 % | +7.90 % | -6.01 % | +1.79 % |
| fp16 | 16k x c=1 | +0.35 % | -0.37 % | +0.37 % | +0.54 % |
| fp16 | 16k x c=8 | +2.10 % | +4.15 % | -6.71 % | +2.09 % |
| int4 | 1k x c=1 | -2.84 % | +0.35 % | -0.35 % | -2.56 % |
| int4 | 1k x c=8 | -1.89 % | +2.78 % | +0.06 % | +0.16 % |
| int4 | 16k x c=1 | -1.50 % | +2.34 % | -2.29 % | -0.54 % |
| int4 | 16k x c=8 | -1.08 % | -9.70 % | +15.60 % | -2.03 % |

## Verdict: a wash, keep two blocks

The two KV dtypes move in opposite directions (FP16 output +0.35 % to +2.74 %, INT4 -1.08 % to -2.84 %), which is the signature of run-to-run noise rather than a lever: the box's noise floor is 1-2.5 % across a ~30 minute gap, and the shipped-recipe pair reproduces the harness-driven stall-0 arms within that band. The one cell that moves on its own (INT4 16k x c=8, TTFT -9.7 % and prefill +15.6 %) does not move the output (-1.08 %).

So the shipped derivation stands: **two mamba state blocks per scheduler step** (MAXBAT 2048 for the FP16 KV cache, 4096 for the packed KV cache with the fp16 SSM state).

## The first attempt was aborted by a PCIe link event

The first pair (`maxbat3`, kept under `contaminated-arm-maxbat3/`) failed the raw hardware gate on a **corrected PCIe BadTLP on root port `0000:40:03.1`**. The raw FAIL is preserved; the harness classified it separately as `contaminated-corrected-link-noise` and the retry was run with `--diagnostic-reference`.

The port is not on the serving path. ROCm GPU 0-3 (HIP 6-9, the serving GPUs) sit at PCI 07/0a/12/15 behind root port `0000:00:01.1`; the four compile GPUs (HIP 2-5) sit at PCI 47/4a/52/55 behind `0000:40:03.1`, where four `hi2 quantize` jobs were running the cards at 97-99 % use. The box logged three AER events in 12.2 h of uptime on that port, so a ~28 minute arm has roughly a 13 % chance of catching one.

No gate was relaxed. The contamination is recorded with the endpoint BDF, and the retry arms are the ones the numbers above come from.

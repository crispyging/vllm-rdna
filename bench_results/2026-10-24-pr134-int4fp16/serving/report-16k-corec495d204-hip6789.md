# 16k input / 1k output - serving report

- FP16 reference: `/Users/kletorch/Projects/infrastructure/gfx1030_optimized/vllm-pr-int4fp16/bench_results/2026-10-24-pr134-int4fp16/serving/27b-fp16-corec495d204-hip6789`
- INT4 candidate: `/Users/kletorch/Projects/infrastructure/gfx1030_optimized/vllm-pr-int4fp16/bench_results/2026-10-24-pr134-int4fp16/serving/27b-int4-corec495d204-hip6789`

## c=1 (bench-16384x1024-c1)

| metric | FP16 | INT4 | ratio | delta |
| --- | --- | --- | --- | --- |
| prefill per-request (tok/s) | 863.205 | 866.175 | 1.0034 | +0.34% |
| prefill aggregated (tok/s) | 498.097 | 522.588 | 1.0492 | +4.92% |
| prefill aggregated proxy (tok/s) | 858.581 | 863.869 | 1.0062 | +0.62% |
| decode per-request (tok/s) | 73.998 | 82.266 | 1.1117 | +11.17% |
| decode aggregated (tok/s) | 31.131 | 32.662 | 1.0492 | +4.92% |
| TTFT (s) | 18.981 | 18.916 | 0.9966 | -0.34% |
| ITL median (ms) | 38.020 | 35.907 | 0.9444 | -5.56% |
| ITL mean (ms) | 38.128 | 36.019 | 0.9447 | -5.53% |
| MTP acceptance length | 2.806 | 2.967 | 1.0575 | +5.75% |

## c=8 (bench-16384x1024-c8)

| metric | FP16 | INT4 | ratio | delta |
| --- | --- | --- | --- | --- |
| prefill per-request (tok/s) | 457.260 | 172.547 | 0.3773 | -62.27% |
| prefill aggregated (tok/s) | 678.833 | 593.271 | 0.8740 | -12.60% |
| prefill aggregated proxy (tok/s) | 805.542 | 676.964 | 0.8404 | -15.96% |
| decode per-request (tok/s) | 8.581 | 11.059 | 1.2887 | +28.87% |
| decode aggregated (tok/s) | 42.427 | 37.079 | 0.8740 | -12.60% |
| TTFT (s) | 49.934 | 107.240 | 2.1476 | +114.76% |
| ITL median (ms) | 397.292 | 317.034 | 0.7980 | -20.20% |
| ITL mean (ms) | 392.718 | 301.846 | 0.7686 | -23.14% |
| MTP acceptance length | 2.838 | 2.905 | 1.0235 | +2.35% |

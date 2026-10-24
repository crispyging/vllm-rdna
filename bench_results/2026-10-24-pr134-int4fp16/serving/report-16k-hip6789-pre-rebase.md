# 16k input / 1k output - serving report

- FP16 reference: `bench_results/2026-10-24-pr134-int4fp16/serving/27b-fp16-hip6789`
- INT4 candidate: `bench_results/2026-10-24-pr134-int4fp16/serving/27b-int4-hip6789`

## c=1 (bench-16384x1024-c1)

| metric | FP16 | INT4 | ratio | delta |
| --- | --- | --- | --- | --- |
| prefill per-request (tok/s) | 863.324 | 869.462 | 1.0071 | +0.71% |
| prefill aggregated (tok/s) | 499.294 | 523.979 | 1.0494 | +4.94% |
| prefill aggregated proxy (tok/s) | 858.070 | 863.163 | 1.0059 | +0.59% |
| decode per-request (tok/s) | 74.298 | 82.345 | 1.1083 | +10.83% |
| decode aggregated (tok/s) | 31.206 | 32.749 | 1.0494 | +4.94% |
| TTFT (s) | 18.978 | 18.844 | 0.9929 | -0.71% |
| ITL median (ms) | 38.015 | 35.876 | 0.9437 | -5.63% |
| ITL mean (ms) | 38.128 | 35.972 | 0.9434 | -5.66% |
| MTP acceptance length | 2.822 | 2.966 | 1.0508 | +5.08% |

## c=8 (bench-16384x1024-c8)

| metric | FP16 | INT4 | ratio | delta |
| --- | --- | --- | --- | --- |
| prefill per-request (tok/s) | 476.587 | 171.542 | 0.3599 | -64.01% |
| prefill aggregated (tok/s) | 678.800 | 594.831 | 0.8763 | -12.37% |
| prefill aggregated proxy (tok/s) | 804.999 | 677.696 | 0.8419 | -15.81% |
| decode per-request (tok/s) | 8.539 | 11.132 | 1.3038 | +30.38% |
| decode aggregated (tok/s) | 42.425 | 37.177 | 0.8763 | -12.37% |
| TTFT (s) | 48.655 | 107.629 | 2.2121 | +121.21% |
| ITL median (ms) | 540.028 | 316.249 | 0.5856 | -41.44% |
| ITL mean (ms) | 393.869 | 300.930 | 0.7640 | -23.60% |
| MTP acceptance length | 2.821 | 2.922 | 1.0357 | +3.57% |

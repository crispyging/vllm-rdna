# PR134 serving levers

Model `cyankiwi/Qwen3.8-27B-AWQ-INT4`, TP=4, MTP=2, prefix caching +
mamba align, FULL_AND_PIECEWISE graphs, one boot, one source/binary manifest.

| arm | KV | W4A8 | stall ms | status | 1k c=1 | 1k c=8 | 16k c=1 | 16k c=8 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| fp16-core4292ace2-base-ssm16-real-hip6789 | fp16 | 0 | recipe | PASS | 82.7 / 0.9s | 207.9 / 2.6s | 31.1 / 19.1s | 42.5 / 48.8s |
| int4-core4292ace2-base-ssm16-real-hip6789 | int4_fp16_block | 0 | recipe | PASS | 84.6 / 0.9s | 202.4 / 3.2s | 33.3 / 18.3s | 41.2 / 48.4s |
| fp16-core4292ace2-w4a8-ssm16-hip6789 | fp16 | 1 | recipe | PASS | 87.3 / 0.7s | 234.2 / 2.0s | 36.1 / 15.2s | 52.4 / 35.6s |
| fp16-core4292ace2-w4a8-ssm16-real-hip6789 | fp16 | 1 | recipe | PASS | 87.6 / 0.7s | 233.3 / 2.0s | 36.1 / 15.2s | 52.3 / 34.4s |
| int4-core4292ace2-w4a8-ssm16-real-hip6789 | int4_fp16_block | 1 | recipe | PASS | 81.6 / 0.7s | 230.0 / 2.1s | 37.5 / 15.0s | 50.5 / 39.5s |
| fp16-core4292ace2-stall0-w4a8-ssm16-real-hip6789 | fp16 | 1 | 0 | PASS | 87.3 / 0.7s | 239.1 / 1.6s | 36.1 / 15.2s | 52.7 / 34.5s |
| int4-core4292ace2-stall0-w4a8-ssm16-real-hip6789 | int4_fp16_block | 1 | 0 | PASS | 81.8 / 0.7s | 231.3 / 1.7s | 37.3 / 14.9s | 51.7 / 34.9s |
| fp16-core4292ace2-recipe-final-hip6789 | fp16 | recipe | recipe | PASS | 85.1 / 0.7s | 236.7 / 1.6s | 35.5 / 15.4s | 52.0 / 35.9s |
| int4-core4292ace2-recipe-final-hip6789 | int4_fp16_block | recipe | recipe | PASS | 83.8 / 0.7s | 233.9 / 1.8s | 37.7 / 14.8s | 52.0 / 36.8s |

Cells read `aggregate output tok/s / mean TTFT`.

A knob of `recipe` means the recipe's own value: the 27b-awq recipe sets W4A8=1 and the decode-stall cap to 0, so those arms are the shipped configuration.

## Lever deltas

One knob changed, everything else held fixed, same boot.

### fp16 KV, W4A8 0 -> 1 (stall 250)

| cell | output | TTFT | prefill | decode |
| --- | --- | --- | --- | --- |
| 1k x c=1 | +5.88 % | -23.16 % | +30.86 % | +1.94 % |
| 1k x c=8 | +12.20 % | -24.32 % | +26.04 % | +9.08 % |
| 16k x c=1 | +16.05 % | -20.00 % | +24.94 % | +5.00 % |
| 16k x c=8 | +23.12 % | -29.47 % | +52.76 % | +13.89 % |

### fp16 KV, decode-stall cap 250 -> 0 (W4A8 1)

| cell | output | TTFT | prefill | decode |
| --- | --- | --- | --- | --- |
| 1k x c=1 | +0.05 % | +0.00 % | -0.49 % | +0.12 % |
| 1k x c=8 | +2.11 % | -17.35 % | +22.84 % | +0.77 % |
| 16k x c=1 | +0.03 % | +0.07 % | -0.06 % | +0.17 % |
| 16k x c=8 | +0.61 % | -2.92 % | +1.16 % | -0.71 % |

### int4 KV, W4A8 0 -> 1 (stall 250)

| cell | output | TTFT | prefill | decode |
| --- | --- | --- | --- | --- |
| 1k x c=1 | -3.48 % | -22.34 % | +29.14 % | -6.17 % |
| 1k x c=8 | +13.66 % | -34.59 % | +46.67 % | +5.56 % |
| 16k x c=1 | +12.74 % | -18.15 % | +22.16 % | +1.21 % |
| 16k x c=8 | +22.44 % | -18.26 % | +22.43 % | +14.18 % |

### int4 KV, decode-stall cap 250 -> 0 (W4A8 1)

| cell | output | TTFT | prefill | decode |
| --- | --- | --- | --- | --- |
| 1k x c=1 | +0.22 % | +0.00 % | -0.16 % | +0.14 % |
| 1k x c=8 | +0.56 % | -19.23 % | +32.53 % | +2.30 % |
| 16k x c=1 | -0.64 % | -0.13 % | +0.13 % | -1.42 % |
| 16k x c=8 | +2.48 % | -11.74 % | +17.66 % | -0.32 % |

## Environment

- Host: par1-cs25 (192.168.1.84), 8x Radeon PRO V620 (gfx1030)
- Serving GPUs: HIP 6,7,8,9 (cards 0-3, the clean PLX 88096 root port); compile GPUs: HIP 2,3,4,5 (cards 4-7)
- Boot id: 9d2621ae-2838-4db6-951d-43ccd261f397

### The Hugging Face snapshot directory of the served model became unopenable mid-campaign (the parent directory still listed it, but the inode could not be resolved: a stale XFS directory entry after the host reboot). The blobs it pointed at stayed intact

The model was rebuilt from the intact blobs into /home/chenco_adm/pr134-model-real/qwen3.8-27b-awq (a plain copy, same content) and served from there for the rest of the campaign, with the pair driver taking the model directory as an override so both arms of a pair always served the same copy. A bind mount over the original snapshot path was also installed and verified to resolve for pre-existing sessions, so the path string kept working while the copy was being made.

None on the measurements: every arm after the repair loaded the same model weights from the same blobs, and the arm command lines are identical apart from the model directory. The two arms measured before the repair (the first FP16 reference) bracket the repaired ones: they differ by 0.4 % or less in every cell. The wider spread shows up across the campaign rather than within a pair: the shipped-recipe reference, re-measured about half an hour after the stall-0 FP16 arm on the same configuration, is 1.0-2.5 % lower in every cell. Treat single-digit percentages as the box's run-to-run noise and only the larger deltas as signal.

### fp16 KV, fp16-core4292ace2-base-ssm16-real-hip6789 -> fp16-core4292ace2-w4a8-ssm16-hip6789

| cell | output | TTFT | prefill | decode |
| --- | --- | --- | --- | --- |
| 1k x c=1 | +5.51 % | -23.16 % | +30.72 % | +1.57 % |
| 1k x c=8 | +12.62 % | -24.32 % | +27.11 % | +9.58 % |
| 16k x c=1 | +16.11 % | -20.05 % | +25.06 % | +4.98 % |
| 16k x c=8 | +23.23 % | -27.12 % | +46.03 % | +14.94 % |

### fp16 KV, fp16-core4292ace2-base-ssm16-real-hip6789 -> fp16-core4292ace2-w4a8-ssm16-real-hip6789

| cell | output | TTFT | prefill | decode |
| --- | --- | --- | --- | --- |
| 1k x c=1 | +5.88 % | -23.16 % | +30.86 % | +1.94 % |
| 1k x c=8 | +12.20 % | -24.32 % | +26.04 % | +9.08 % |
| 16k x c=1 | +16.05 % | -20.00 % | +24.94 % | +5.00 % |
| 16k x c=8 | +23.12 % | -29.47 % | +52.76 % | +13.89 % |

### fp16 KV, fp16-core4292ace2-base-ssm16-real-hip6789 -> fp16-core4292ace2-stall0-w4a8-ssm16-real-hip6789

| cell | output | TTFT | prefill | decode |
| --- | --- | --- | --- | --- |
| 1k x c=1 | +5.56 % | -23.16 % | +30.08 % | +1.69 % |
| 1k x c=8 | +14.99 % | -37.45 % | +56.14 % | +10.42 % |
| 16k x c=1 | +16.14 % | -20.00 % | +24.99 % | +5.16 % |
| 16k x c=8 | +23.99 % | -29.25 % | +47.73 % | +14.12 % |

### fp16 KV, fp16-core4292ace2-base-ssm16-real-hip6789 -> fp16-core4292ace2-recipe-final-hip6789

| cell | output | TTFT | prefill | decode |
| --- | --- | --- | --- | --- |
| 1k x c=1 | +2.88 % | -23.16 % | +30.58 % | -1.32 % |
| 1k x c=8 | +13.81 % | -38.22 % | +57.91 % | +9.64 % |
| 16k x c=1 | +14.12 % | -19.06 % | +23.53 % | +2.63 % |
| 16k x c=8 | +22.46 % | -26.42 % | +39.59 % | +13.42 % |

### fp16 KV, fp16-core4292ace2-w4a8-ssm16-hip6789 -> fp16-core4292ace2-w4a8-ssm16-real-hip6789

| cell | output | TTFT | prefill | decode |
| --- | --- | --- | --- | --- |
| 1k x c=1 | +0.34 % | +0.00 % | +0.11 % | +0.36 % |
| 1k x c=8 | -0.38 % | +0.00 % | -0.84 % | -0.46 % |
| 16k x c=1 | -0.06 % | +0.07 % | -0.09 % | +0.01 % |
| 16k x c=8 | -0.10 % | -3.23 % | +4.61 % | -0.91 % |

### fp16 KV, fp16-core4292ace2-w4a8-ssm16-hip6789 -> fp16-core4292ace2-stall0-w4a8-ssm16-real-hip6789

| cell | output | TTFT | prefill | decode |
| --- | --- | --- | --- | --- |
| 1k x c=1 | +0.05 % | +0.00 % | -0.49 % | +0.12 % |
| 1k x c=8 | +2.11 % | -17.35 % | +22.84 % | +0.77 % |
| 16k x c=1 | +0.03 % | +0.07 % | -0.06 % | +0.17 % |
| 16k x c=8 | +0.61 % | -2.92 % | +1.16 % | -0.71 % |

### fp16 KV, fp16-core4292ace2-w4a8-ssm16-hip6789 -> fp16-core4292ace2-recipe-final-hip6789

| cell | output | TTFT | prefill | decode |
| --- | --- | --- | --- | --- |
| 1k x c=1 | -2.50 % | +0.00 % | -0.11 % | -2.85 % |
| 1k x c=8 | +1.05 % | -18.37 % | +24.23 % | +0.06 % |
| 16k x c=1 | -1.72 % | +1.25 % | -1.23 % | -2.25 % |
| 16k x c=8 | -0.63 % | +0.96 % | -4.41 % | -1.32 % |

### fp16 KV, fp16-core4292ace2-w4a8-ssm16-real-hip6789 -> fp16-core4292ace2-stall0-w4a8-ssm16-real-hip6789

| cell | output | TTFT | prefill | decode |
| --- | --- | --- | --- | --- |
| 1k x c=1 | -0.30 % | +0.00 % | -0.60 % | -0.24 % |
| 1k x c=8 | +2.49 % | -17.35 % | +23.88 % | +1.23 % |
| 16k x c=1 | +0.08 % | +0.00 % | +0.04 % | +0.15 % |
| 16k x c=8 | +0.71 % | +0.32 % | -3.30 % | +0.20 % |

### fp16 KV, fp16-core4292ace2-w4a8-ssm16-real-hip6789 -> fp16-core4292ace2-recipe-final-hip6789

| cell | output | TTFT | prefill | decode |
| --- | --- | --- | --- | --- |
| 1k x c=1 | -2.83 % | +0.00 % | -0.21 % | -3.20 % |
| 1k x c=8 | +1.44 % | -18.37 % | +25.28 % | +0.52 % |
| 16k x c=1 | -1.66 % | +1.18 % | -1.13 % | -2.26 % |
| 16k x c=8 | -0.54 % | +4.33 % | -8.62 % | -0.41 % |

### fp16 KV, fp16-core4292ace2-stall0-w4a8-ssm16-real-hip6789 -> fp16-core4292ace2-recipe-final-hip6789

| cell | output | TTFT | prefill | decode |
| --- | --- | --- | --- | --- |
| 1k x c=1 | -2.54 % | +0.00 % | +0.39 % | -2.97 % |
| 1k x c=8 | -1.03 % | -1.23 % | +1.14 % | -0.71 % |
| 16k x c=1 | -1.74 % | +1.18 % | -1.17 % | -2.41 % |
| 16k x c=8 | -1.23 % | +4.00 % | -5.51 % | -0.61 % |

### int4 KV, int4-core4292ace2-base-ssm16-real-hip6789 -> int4-core4292ace2-w4a8-ssm16-real-hip6789

| cell | output | TTFT | prefill | decode |
| --- | --- | --- | --- | --- |
| 1k x c=1 | -3.48 % | -22.34 % | +29.14 % | -6.17 % |
| 1k x c=8 | +13.66 % | -34.59 % | +46.67 % | +5.56 % |
| 16k x c=1 | +12.74 % | -18.15 % | +22.16 % | +1.21 % |
| 16k x c=8 | +22.44 % | -18.26 % | +22.43 % | +14.18 % |

### int4 KV, int4-core4292ace2-base-ssm16-real-hip6789 -> int4-core4292ace2-stall0-w4a8-ssm16-real-hip6789

| cell | output | TTFT | prefill | decode |
| --- | --- | --- | --- | --- |
| 1k x c=1 | -3.26 % | -22.34 % | +28.94 % | -6.04 % |
| 1k x c=8 | +14.30 % | -47.17 % | +94.37 % | +7.99 % |
| 16k x c=1 | +12.02 % | -18.26 % | +22.32 % | -0.23 % |
| 16k x c=8 | +25.47 % | -27.85 % | +44.06 % | +13.81 % |

### int4 KV, int4-core4292ace2-base-ssm16-real-hip6789 -> int4-core4292ace2-recipe-final-hip6789

| cell | output | TTFT | prefill | decode |
| --- | --- | --- | --- | --- |
| 1k x c=1 | -0.86 % | -22.34 % | +28.54 % | -3.86 % |
| 1k x c=8 | +15.58 % | -44.34 % | +86.60 % | +6.98 % |
| 16k x c=1 | +13.41 % | -19.03 % | +23.45 % | +1.22 % |
| 16k x c=8 | +26.10 % | -23.88 % | +32.99 % | +14.67 % |

### int4 KV, int4-core4292ace2-w4a8-ssm16-real-hip6789 -> int4-core4292ace2-stall0-w4a8-ssm16-real-hip6789

| cell | output | TTFT | prefill | decode |
| --- | --- | --- | --- | --- |
| 1k x c=1 | +0.22 % | +0.00 % | -0.16 % | +0.14 % |
| 1k x c=8 | +0.56 % | -19.23 % | +32.53 % | +2.30 % |
| 16k x c=1 | -0.64 % | -0.13 % | +0.13 % | -1.42 % |
| 16k x c=8 | +2.48 % | -11.74 % | +17.66 % | -0.32 % |

### int4 KV, int4-core4292ace2-w4a8-ssm16-real-hip6789 -> int4-core4292ace2-recipe-final-hip6789

| cell | output | TTFT | prefill | decode |
| --- | --- | --- | --- | --- |
| 1k x c=1 | +2.71 % | +0.00 % | -0.46 % | +2.47 % |
| 1k x c=8 | +1.69 % | -14.90 % | +27.23 % | +1.34 % |
| 16k x c=1 | +0.59 % | -1.07 % | +1.05 % | +0.01 % |
| 16k x c=8 | +2.99 % | -6.88 % | +8.62 % | +0.43 % |

### int4 KV, int4-core4292ace2-stall0-w4a8-ssm16-real-hip6789 -> int4-core4292ace2-recipe-final-hip6789

| cell | output | TTFT | prefill | decode |
| --- | --- | --- | --- | --- |
| 1k x c=1 | +2.48 % | +0.00 % | -0.31 % | +2.33 % |
| 1k x c=8 | +1.12 % | +5.36 % | -4.00 % | -0.94 % |
| 16k x c=1 | +1.23 % | -0.94 % | +0.92 % | +1.45 % |
| 16k x c=8 | +0.50 % | +5.50 % | -7.68 % | +0.75 % |

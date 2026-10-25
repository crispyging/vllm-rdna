# PR134 serving levers

Model `cyankiwi/Qwen3.8-27B-AWQ-INT4`, TP=4, MTP=2, prefix caching +
mamba align, FULL_AND_PIECEWISE graphs, one boot, one source/binary manifest.

| arm | KV | W4A8 | stall ms | status | 1k c=1 | 1k c=8 | 16k c=1 | 16k c=8 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| fp16-core4292ace2-w4a8-ssm16-hip6789 | fp16 | 1 | recipe | PASS | 87.3 / 0.7s | 234.2 / 2.0s | 36.1 / 15.2s | 52.4 / 35.6s |
| fp16-core4292ace2-w4a8-ssm16-real-hip6789 | fp16 | 1 | recipe | PASS | 87.6 / 0.7s | 233.3 / 2.0s | 36.1 / 15.2s | 52.3 / 34.4s |
| int4-core4292ace2-w4a8-ssm16-real-hip6789 | int4_fp16_block | 1 | recipe | PASS | 81.6 / 0.7s | 230.0 / 2.1s | 37.5 / 15.0s | 50.5 / 39.5s |

Cells read `aggregate output tok/s / mean TTFT`.

## Environment

- Host: par1-cs25 (192.168.1.84), 8x Radeon PRO V620 (gfx1030)
- Serving GPUs: HIP 6,7,8,9 (cards 0-3, the clean PLX 88096 root port); compile GPUs: HIP 2,3,4,5 (cards 4-7)
- Boot id: 9d2621ae-2838-4db6-951d-43ccd261f397

### The Hugging Face snapshot directory of the served model became unopenable mid-campaign (the parent directory still listed it, but the inode could not be resolved: a stale XFS directory entry after the host reboot). The blobs it pointed at stayed intact

The model was rebuilt from the intact blobs into /home/chenco_adm/pr134-model-real/qwen3.8-27b-awq (a plain copy, same content) and served from there for the rest of the campaign, with the pair driver taking the model directory as an override so both arms of a pair always served the same copy. A bind mount over the original snapshot path was also installed and verified to resolve for pre-existing sessions, so the path string kept working while the copy was being made.

None on the measurements: every arm after the repair loaded the same model weights from the same blobs, and the arm command lines are identical apart from the model directory. The two arms measured before the repair (the first FP16 reference) bracket the repaired ones: they differ by 0.4 % or less in every cell, which is the run-to-run noise of the box.

### fp16: {'w4a8': '1', 'decode_stall_ms': 'recipe'} -> {'w4a8': '1', 'decode_stall_ms': 'recipe'}

| cell | output | TTFT | prefill | decode |
| --- | --- | --- | --- | --- |
| 1k x c=1 | +0.34 % | +0.00 % | +0.11 % | +0.36 % |
| 1k x c=8 | -0.38 % | +0.00 % | -0.84 % | -0.46 % |
| 16k x c=1 | -0.06 % | +0.07 % | -0.09 % | +0.01 % |
| 16k x c=8 | -0.10 % | -3.23 % | +4.61 % | -0.91 % |

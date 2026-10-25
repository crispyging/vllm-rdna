# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Markdown table of serve_validate.sh bench results: all four numbers per cell.

python summarize.py ROOT tag1 tag2 ...   (ROOT/serve-<tag>/bench-*.json)
"""

import json
import sys
from pathlib import Path

CELLS = [
    ("1k/512 c=1", "bench-1024x512-c1"),
    ("1k/512 c=8", "bench-1024x512-c8"),
    ("16k/1k c=1", "bench-16384x1024-c1"),
    ("16k/1k c=8", "bench-16384x1024-c8"),
]


def cell(path: Path) -> str:
    if not path.exists():
        return "| n/a | n/a | n/a | n/a |"
    d = json.loads(path.read_text())
    done = max(int(d.get("completed", 0)), 1)
    in_len = d.get("total_input_tokens", 0) / done
    ttft = d.get("mean_ttft_ms", 0.0) / 1000
    p99 = d.get("p99_ttft_ms", 0.0) / 1000
    wave = min(int(d.get("max_concurrency") or 1), done)
    pre = in_len / ttft if ttft else 0
    pre_agg = wave * in_len / p99 if p99 else 0
    tpot = d.get("mean_tpot_ms", 0.0)
    dec = 1000 / tpot if tpot else 0
    return (
        f"| {pre:.0f} ({pre_agg:.0f}) | {dec:.1f} ({d.get('output_throughput', 0):.1f})"
        f" | {ttft:.2f} | {d.get('median_itl_ms', 0):.1f} |"
    )


def main() -> None:
    root = Path(sys.argv[1])
    print("| cell | arm | prefill tok/s | decode tok/s | TTFT s | ITL ms (median) |")
    print("| --- | --- | --- | --- | --- | --- |")
    for label, name in CELLS:
        for tag in sys.argv[2:]:
            print(f"| {label} | {tag} " + cell(root / f"serve-{tag}" / f"{name}.json"))


if __name__ == "__main__":
    main()

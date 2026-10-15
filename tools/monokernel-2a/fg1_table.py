#!/usr/bin/env python3
"""Per-rank table of the in-serve timers (mono_ffn.py log lines) for one or
more serve dirs: python3 fg1_table.py ~/w4a8_runs/fg1-skew/serve-* [--widths 1,8]

Columns (us, per layer): region (FFN block + AR), kernel (mono), rest (outside
the FFN region), attn (HC preamble + attention + its AR), hc, AR (RowParallel
all-reduce incl. wait), qsa/gdn core (HC end -> AR end), gap8 (empty-kernel
dispatch gap), lat9 (64-step dependent L2 chain, shader cycles), sclk.
"""

import re
import sys
from pathlib import Path

RE = re.compile(
    r"\(Worker_TP(?P<rank>\d)[^)]*\).*rdna_mono_ffn timing dev\d w(?P<w>\d+): "
    r"region n=(?P<rn>\d+) (?P<region>[\d.]+) us \| kernel n=(?P<kn>\d+) total (?P<kernel>[\d.]+)"
    r".*?rest n=\d+ (?P<rest>[\d.]+) us \| attn n=\d+ (?P<attn>[\d.]+) us \(hc (?P<hc>[\d.]+)\)"
    r" \| linear AR n=\d+ (?P<ar>[\d.]+) us \| core qsa n=\d+ (?P<qsa>[\d.]+) gdn n=\d+ (?P<gdn>[\d.]+)"
    r" \| sclk attn (?P<sclk>\d+)"
    r"(?:.*?gap8 n=\d+ (?P<gap8>[\d.]+) us \| lat9 n=\d+ (?P<lat9>\d+) cyc)?"
)


def main(argv):
    widths = {1, 8}
    dirs = []
    for a in argv:
        if a.startswith("--widths"):
            widths = {int(x) for x in a.split("=", 1)[1].split(",")}
        else:
            dirs.append(Path(a))
    for d in dirs:
        log = d / "serve.log" if d.is_dir() else d
        if not log.exists():
            continue
        last = {}
        fg = itl = None
        for line in log.read_text(errors="replace").splitlines():
            if fg is None and "HSA_FORCE_FINE_GRAIN_PCIE=" in line:
                m = re.search(r"HSA_FORCE_FINE_GRAIN_PCIE=(\d)", line)
                fg = m.group(1) if m else None
            m = RE.search(line)
            if m and int(m["w"]) in widths:
                last[(int(m["w"]), int(m["rank"]))] = m.groupdict()
        summ = d / "summary.txt" if d.is_dir() else None
        if summ and summ.exists():
            for line in summ.read_text().splitlines():
                if "bench-1024x512-c1 " in line:
                    mm = re.search(r"ITL med\s+([\d.]+)", line)
                    itl = mm.group(1) if mm else None
        print(f"## {d.name} FG={fg} ITL(c1)={itl} ms")
        print("  w rank  region kernel   rest   attn    hc    AR   qsa   gdn  gap8  lat9  sclk")
        for (w, r), g in sorted(last.items()):
            print(
                f"  {w:>1} dev{r}  {float(g['region']):6.1f} {float(g['kernel']):6.1f} "
                f"{float(g['rest']):6.1f} {float(g['attn']):6.1f} {float(g['hc']):5.1f} "
                f"{float(g['ar']):5.1f} {float(g['qsa']):5.1f} {float(g['gdn']):5.1f} "
                f"{float(g['gap8'] or 0):5.2f} {int(g['lat9'] or 0):5d}  {g['sclk']}"
            )


if __name__ == "__main__":
    main(sys.argv[1:])

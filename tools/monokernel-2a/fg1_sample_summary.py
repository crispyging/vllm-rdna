#!/usr/bin/env python3
"""Summarise fg1_sample.sh CSVs: per BDF, the distribution of MCLK/SCLK/FCLK
states, mean power/temp/busy, over samples whose gpu_busy >= 50 (the serve's
busy window). python3 fg1_sample_summary.py sample-*.csv"""

import collections
import csv
import sys


def main(paths):
    for p in paths:
        rows = collections.defaultdict(list)
        with open(p) as f:
            for r in csv.DictReader(f):
                if r.get("bdf") == "bdf":
                    continue
                try:
                    if int(r["busy"] or 0) < 50:
                        continue
                except ValueError:
                    continue
                rows[r["bdf"]].append(r)
        print(f"## {p}")
        for bdf, rs in sorted(rows.items()):
            n = len(rs)
            if not n:
                continue
            mclk = collections.Counter(r["mclk"] for r in rs)
            fclk = collections.Counter(r["fclk"] for r in rs)
            sclk = [int(r["sclk"]) for r in rs if r["sclk"]]
            pw = [int(r["power_uW"]) for r in rs if r["power_uW"]]
            tp = [int(r["temp_mC"]) for r in rs if r["temp_mC"]]
            mb = [int(r["membusy"]) for r in rs if r["membusy"]]
            rp = sorted({r["pcie_replay"] for r in rs if r["pcie_replay"]})
            print(
                f"  {bdf} n={n:4d} mclk={dict(mclk)} fclk={dict(fclk)} "
                f"sclk~{sum(sclk)/len(sclk) if sclk else 0:.0f} "
                f"power~{sum(pw)/len(pw)/1e6 if pw else 0:.0f}W "
                f"temp~{sum(tp)/len(tp)/1e3 if tp else 0:.0f}C membusy~{sum(mb)/len(mb) if mb else 0:.0f} "
                f"pcie_replay={rp}"
            )


if __name__ == "__main__":
    main(sys.argv[1:])

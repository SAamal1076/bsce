#!/usr/bin/env python3
"""Mean +- std over training seeds from an evaluate.py result.

  python scripts/aggregate_seeds.py --csv outputs/SV_ULA/nmse_vs_snr.csv

Columns '<tag>' (seed 0, the normal run) and '<tag>_s1', '<tag>_s2', ... are one method trained
with different seeds (train.py --tag <tag>_s1 --set train.seed=1). Writes
nmse_vs_snr_seeds.csv (mean, std, n per method and SNR) and nmse_vs_snr_seeds.png (mean +- std,
values printed) next to the input.
"""
import argparse
import csv
import re
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from bsce import naming  # noqa: E402


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--csv", required=True)
    a = ap.parse_args()
    f = Path(a.csv)
    with open(f, encoding="utf-8") as fh:
        rows = list(csv.DictReader(fh))
    snr = [float(r["SNR_dB"]) for r in rows]
    groups = defaultdict(list)
    for c in rows[0]:
        if c == "SNR_dB":
            continue
        # run_suite names: <tag>_s1 (one network), <tag>_s1_ft_<edges>dB_<k> (fine-tuned; v23: <tag>_s1-ft[edges]):
        # drop the seed in all of them (bsce/naming.py)
        groups[naming.drop_seed(c)].append(c)
    out, stats = [], {}
    for g, cols in groups.items():
        v = np.array([[float(r[c]) for c in cols] for r in rows])          # SNR x seeds
        m, s = v.mean(1), (v.std(1, ddof=1) if v.shape[1] > 1 else np.zeros(len(rows)))
        stats[g] = (m, s, v.shape[1])
        for i, x in enumerate(snr):
            out.append(dict(method=g, SNR_dB=x, mean_db=round(m[i], 4), std_db=round(s[i], 4), n_seeds=v.shape[1]))
    o = f.with_name(f.stem + "_seeds.csv")
    with open(o, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(out[0])); w.writeheader(); w.writerows(out)
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(10, 6))
    for (g, (m, s, n)), c in zip(stats.items(), plt.cm.tab20(np.linspace(0, 1, max(len(stats), 2)))):
        ax.errorbar(snr, m, yerr=s, color=c, marker="o", ms=3, lw=1.3, capsize=2,
                    label=f"{g} (n={n}): {m[0]:.2f} / {m[len(m) // 2]:.2f} / {m[-1]:.2f} dB")
        ax.annotate(f"{m[-1]:.2f}±{s[-1]:.2f}", (snr[-1], m[-1]), fontsize=6, color=c, xytext=(4, 0),
                    textcoords="offset points", va="center")
    ax.set_xlabel("SNR (dB)"); ax.set_ylabel("NMSE (dB)"); ax.grid(alpha=.3)
    ax.set_title(f"{f.parent.name}: mean ± std over seeds (legend: first / middle / last SNR)", fontsize=10)
    ax.legend(frameon=False, fontsize=7)
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    fig.tight_layout(); fig.savefig(o.with_suffix(".png"), dpi=160)
    print(f"written {o} and {o.with_suffix('.png')}")
    for g, (m, s, n) in stats.items():
        print(f"{g:32s} n={n}  " + "  ".join(f"{x:.0f}dB {mm:.2f}±{ss:.2f}" for x, mm, ss in list(zip(snr, m, s))[::5]))


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Paper figures and tables from evaluated run folders (one panel per scenario, numbers printed).

  python scripts/make_figures.py --runs outputs/SV_ULA outputs/SV_UPA outputs/DeepMIMO_ULA outputs/DeepMIMO_UPA
  python scripts/make_figures.py --runs outputs/SV_UPA_user outputs/SV_UPA_user_2band \
         --methods "GM-LAMP-authors=GM-LAMP" "LADMM-untitled2-better=LADMM (one net)" "LADMM-untitled2-better-ft=LADMM (2 bands)" \
         --ref GM-LAMP --snr-table 0 10 20 --out outputs/figures/sv_upa_ladmm
  options: --methods tag[=label] ...   which columns of nmse_vs_snr.csv, in this order, with a display name
                                       (default: every learned method + amp; tags may use * as a wildcard)
           --ref <label>               adds 'gain over <label>' (mean over the test SNRs, dB) to the tables
           --snr-table 0 10 20         SNRs of the table (default 0 10 20)
           --titles "SV ULA" ...       panel titles (default: folder names)    --ncols 2    --out outputs/figures

Input : <run folder>/nmse_vs_snr.csv from evaluate.py. Seeds: columns <tag>_s1, <tag>_s2, ... are averaged with
        <tag> (mean +- std; std shown as a band and in the tables).
Output: <out>/nmse_panels.png and .pdf (NMSE vs SNR, one panel per run folder, value printed at every table SNR),
        <out>/table.csv, table.md, table.tex (methods x scenario x table SNR, mean +- std when seeds exist,
        gain over --ref), <out>/figures_info.json.
"""
import argparse
import csv
import fnmatch
import json
import re
import sys
from collections import OrderedDict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
PAL = ["#2a78d6", "#c23b6a", "#1f9e6e", "#eb6834", "#8c5ab4", "#b08a1a", "#00a3c4", "#e34948", "#52514e", "#6250d6",
       "#008300", "#a05a00"]
sys.path.insert(0, str(ROOT))
from bsce.classical import METHODS as CLASSICAL  # noqa: E402  (v31: one list)


def read_csv(folder: Path):
    f = folder / "nmse_vs_snr.csv"
    if not f.exists():
        sys.exit(f"missing {f}: run scripts/evaluate.py --runs {folder} first")
    rows = list(csv.DictReader(open(f, encoding="utf-8")))
    snr = np.array([float(r["SNR_dB"]) for r in rows])
    cols = {k: np.array([float(r[k]) if r[k] not in ("", None) else np.nan for r in rows]) for k in rows[0] if k != "SNR_dB"}
    groups = OrderedDict()
    for k in cols:
        groups.setdefault(re.sub(r"_s\d+$", "", k), []).append(cols[k])
    stats = {g: (np.nanmean(v, 0), np.nanstd(v, 0, ddof=1) if len(v) > 1 else np.zeros(len(snr)), len(v))
             for g, v in groups.items()}
    return snr, stats


def select(stats, specs):
    """specs: ['tag=label', 'LADMM*'] -> OrderedDict label -> tag."""
    if not specs:
        learned = [g for g in stats if g not in CLASSICAL]
        return OrderedDict((g, g) for g in learned + [g for g in ("amp",) if g in stats])
    out = OrderedDict()
    for sp in specs:
        tag, _, label = sp.partition("=")
        hits = [g for g in stats if fnmatch.fnmatch(g, tag)]
        for h in hits:
            out[label if (label and len(hits) == 1) else h] = h
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--runs", nargs="+", required=True)
    ap.add_argument("--methods", nargs="*", default=None)
    ap.add_argument("--ref", default=None)
    ap.add_argument("--snr-table", nargs="+", type=float, default=[0, 10, 20])
    ap.add_argument("--titles", nargs="*", default=None)
    ap.add_argument("--ncols", type=int, default=2)
    ap.add_argument("--out", default="outputs/figures")
    a = ap.parse_args()
    out = Path(a.out) if Path(a.out).is_absolute() else ROOT / a.out
    out.mkdir(parents=True, exist_ok=True)
    folders = [Path(r) if Path(r).is_absolute() else ROOT / r for r in a.runs]
    titles = a.titles or [f.name for f in folders]
    data = [read_csv(f) for f in folders]
    labels_all = []
    sels = []
    for snr, st in data:
        sel = select(st, a.methods)
        sels.append(sel)
        for lab in sel:
            if lab not in labels_all:
                labels_all.append(lab)
    color = {lab: PAL[i % len(PAL)] for i, lab in enumerate(labels_all)}

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    n = len(folders)
    nc = min(a.ncols, n)
    nr = int(np.ceil(n / nc))
    fig, axs = plt.subplots(nr, nc, figsize=(7.2 * nc, 5.0 * nr), squeeze=False)
    rows_tab, info = [], {"runs": [str(f) for f in folders], "methods": {}}
    for k, ((snr, st), sel, ttl) in enumerate(zip(data, sels, titles)):
        ax = axs[k // nc][k % nc]
        ref = st[sels[k][a.ref]][0] if a.ref and a.ref in sel else None
        for j, (lab, tag) in enumerate(sel.items()):
            m, s, ns = st[tag]
            c = color[lab]
            ls = "--" if tag in CLASSICAL or "authors" in tag else "-"
            gain = f", {np.nanmean(ref - m):+.2f} dB vs {a.ref}" if ref is not None and lab != a.ref else ""
            ax.plot(snr, m, ls, color=c, marker="o", ms=3.5, lw=1.6,
                    label=f"{lab}{f' ({ns} seeds)' if ns > 1 else ''}{gain}")
            if ns > 1:
                ax.fill_between(snr, m - s, m + s, color=c, alpha=0.15, lw=0)
            for t in a.snr_table:
                i = int(np.argmin(np.abs(snr - t)))
                ax.annotate(f"{m[i]:.2f}", (snr[i], m[i]), fontsize=6.5, color=c, ha="center",
                            xytext=(0, 6 if j % 2 == 0 else -11), textcoords="offset points")
            row = {"scenario": ttl, "method": lab, "n_seeds": ns}
            for t in a.snr_table:
                i = int(np.argmin(np.abs(snr - t)))
                row[f"{t:g}dB"] = f"{m[i]:.2f}" + (f" ± {s[i]:.2f}" if ns > 1 else "")
            row["mean_0_20"] = f"{np.nanmean(m):.2f}"
            row[f"gain_vs_{a.ref}"] = f"{np.nanmean(ref - m):+.2f}" if ref is not None else ""
            rows_tab.append(row)
            info["methods"].setdefault(lab, {})[ttl] = dict(tag=tag, n_seeds=ns, nmse_db=[round(float(v), 4) for v in m])
        ax.set_title(ttl, fontsize=10)
        ax.set_xlabel("SNR (dB)"); ax.set_ylabel("NMSE (dB)"); ax.grid(alpha=0.3)
        ax.legend(frameon=False, fontsize=7)
        for sp in ("top", "right"):
            ax.spines[sp].set_visible(False)
    for k in range(n, nr * nc):
        axs[k // nc][k % nc].axis("off")
    fig.suptitle(f"NMSE vs SNR (values printed at {', '.join(f'{t:g}' for t in a.snr_table)} dB; same test channels "
                 "and noise for every method)", fontsize=10)
    fig.tight_layout()
    fig.savefig(out / "nmse_panels.png", dpi=180)
    fig.savefig(out / "nmse_panels.pdf")
    keys = ["scenario", "method", "n_seeds"] + [f"{t:g}dB" for t in a.snr_table] + ["mean_0_20"] + (
        [f"gain_vs_{a.ref}"] if a.ref else [])
    with open(out / "table.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=keys, extrasaction="ignore"); w.writeheader(); w.writerows(rows_tab)
    md = ["| " + " | ".join(keys) + " |", "|" + "---|" * len(keys)]
    md += ["| " + " | ".join(str(r.get(k, "")) for k in keys) + " |" for r in rows_tab]
    (out / "table.md").write_text("\n".join(md) + "\n", encoding="utf-8")
    tex = ["\\begin{tabular}{ll" + "c" * (len(keys) - 2) + "}", "\\hline",
           " & ".join(k.replace("_", "\\_").replace("±", "$\\pm$") for k in keys) + " \\\\", "\\hline"]
    tex += [" & ".join(str(r.get(k, "")).replace("±", "$\\pm$").replace("_", "\\_") for k in keys) + " \\\\" for r in rows_tab]
    tex += ["\\hline", "\\end{tabular}"]
    (out / "table.tex").write_text("\n".join(tex) + "\n", encoding="utf-8")
    (out / "figures_info.json").write_text(json.dumps(info, indent=1), encoding="utf-8")
    print("\n".join(md))
    print(f"\nwritten {out / 'nmse_panels.png'} (.pdf), table.csv / .md / .tex")


if __name__ == "__main__":
    main()

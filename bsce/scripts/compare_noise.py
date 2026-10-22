#!/usr/bin/env python3
"""Colored against white measurement noise: every network on both test noises, trained on either (cross tests).

  %run scripts/compare_noise.py --scenarios sv_ula --tags NF-LADMM-S-T8 LADMM-T6-ft GM-LAMP-T8-ft
  %run scripts/compare_noise.py --scenarios sv_ula sv_upa --tags "NF-LADMM*" --nf-matched --name noise_nf

Needs the networks trained with both noise models under the same names:
  outputs/<SCENARIO>/<tag>          run_suite.py ...                        (colored, y = A (x + n), the default)
  outputs/<SCENARIO>_white/<tag>    run_suite.py ... --noise-model white    (white, y = A x + w, same mean noise power)
(a tag present in one folder only is still shown, for the noise model it has). The authors' published GM-LAMP and LAMP
(trained with colored noise) are added from outputs/<SCENARIO>/ for SV / DeepMIMO (import_gmlamp_weights.py if missing;
--no-authors to leave them out).

For every scenario and test noise (colored, white), with the same 2000 test channels and the same noise draw (the white
test noise is the colored draw whitened, bsce/data/noise.py), evaluate.py is run on both folders:
  trained on the test's noise model  -> solid line ("matched training")
  trained on the other noise model   -> dashed line ("cross": what a network loses when the noise model changes)
  --nf-matched: NF-LADMM networks of the other folder once more with the exact data step of the test noise (same
                weights, evaluate.py --nf-step matched) -> dotted line
Writes results/<name>/:
  noise_comparison.png / .pdf  one row per scenario: NMSE vs SNR on colored | on white test noise (value at the last
                               SNR next to each line) | mismatch loss at every SNR (cross - matched training, dB)
  noise_summary.csv / .md      every line: method, trained on, data step, tested on, NMSE at every SNR, mean
  <SCENARIO>/train-<a>_test-<b>[_nfmatched]/   the evaluate.py outputs
"""
import argparse
import csv
import fnmatch
import subprocess
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from bsce.config import load_yaml, resolve, scenario_run_name  # noqa: E402
from bsce.classical import METHODS as CLASSICAL  # noqa: E402  (v31: one list)

SCEN = {"sv_ula": "configs/scenarios/sv_ula.yaml", "sv_upa": "configs/scenarios/sv_upa.yaml",
        "deepmimo_ula": "configs/scenarios/deepmimo_ula.yaml", "deepmimo_upa": "configs/scenarios/deepmimo_upa.yaml",
        "3gpp_upa": "configs/scenarios/3gpp_upa.yaml", "3gpp_ula": "configs/scenarios/3gpp_ula.yaml"}
# the reference categorical palette, fixed order (colour follows the method, never its rank)
PALETTE = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948"]
NM = ("colored", "white")


def run(cmd):
    print("\n>>> " + " ".join(str(c) for c in cmd[1:]), flush=True)
    r = subprocess.run([str(c) for c in cmd], cwd=ROOT)
    if r.returncode:
        sys.exit(f"step failed (exit {r.returncode})")


def matching(folder, pats, exclude=None):
    if not folder.is_dir():
        return []
    return sorted(d.name for d in folder.iterdir() if d.is_dir() and any(f.name.startswith(d.name + "_") and f.suffix == ".pt"
                  for f in d.iterdir()) and (not pats or any(fnmatch.fnmatchcase(d.name, p) for p in pats))
                  and not (exclude and any(fnmatch.fnmatchcase(d.name, p) for p in exclude)))


def nf_step(folder):
    """The data step ('colored' / 'white') of a run folder whose networks are NF-LADMM, else None."""
    import torch
    f = next(iter(sorted(folder.glob(f"{folder.name}_*.pt"))), None)
    try:
        cfg = torch.load(f, map_location="cpu", weights_only=False)["cfg"] if f is not None else {}
    except Exception:                                   # noqa: BLE001 - an unreadable file is simply not switched
        return None
    return cfg.get("noise_model", "colored") if cfg.get("model") == "NF-LADMM" else None


def read_csv(path):
    with open(path, encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    snr = np.array([float(r["SNR_dB"]) for r in rows])
    return snr, {k: np.array([float(r[k]) for r in rows]) for k in rows[0] if k != "SNR_dB"}


def repel(vals, gap):
    order = np.argsort(vals)
    pos = np.array(vals, float)
    for i in range(1, len(order)):
        pos[order[i]] = max(pos[order[i]], pos[order[i - 1]] + gap)
    return pos - (pos.mean() - np.mean(vals))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--scenarios", nargs="+", required=True)
    ap.add_argument("--tags", nargs="*", default=None, help="run folders (names or patterns); default: all")
    ap.add_argument("--exclude", nargs="*", default=None, help="leave out the run folders matching these patterns")
    ap.add_argument("--nf-matched", action="store_true",
                    help="also NF-LADMM trained on the other noise with the exact data step of the test noise")
    ap.add_argument("--no-authors", action="store_true")
    ap.add_argument("--classical", default="amp")
    ap.add_argument("--name", default="noise_comparison")
    ap.add_argument("--outputs", default="outputs")
    ap.add_argument("--results", default="results")
    ap.add_argument("--device", default="auto")
    ap.add_argument("--smoke", action="store_true")
    a = ap.parse_args()
    py = sys.executable
    res = resolve(a.results) / a.name
    res.mkdir(parents=True, exist_ok=True)
    lines, figs = [], []
    for s in a.scenarios:
        sp = SCEN.get(s.lower(), s if s.endswith(".yaml") else f"configs/scenarios/{s.lower()}.yaml")
        sc = load_yaml(sp)
        folders = {nm: resolve(a.outputs) / scenario_run_name(dict(sc, noise_model=nm), a.smoke) for nm in NM}
        published = sc["dataset"] in ("SV", "DeepMIMO") and not a.no_authors
        if published and folders["colored"].is_dir() and not (folders["colored"] / "GM-LAMP-authors").exists():
            run([py, "scripts/import_gmlamp_weights.py", "--scenario", sp, "--runs", folders["colored"]])
        tags = {nm: matching(folders[nm], a.tags, a.exclude) for nm in NM}
        if published:
            tags["colored"] += [t for t in ("GM-LAMP-authors", "LAMP-authors")
                                if t not in tags["colored"] and (folders["colored"] / t).is_dir()]
        if not tags["colored"] and not tags["white"]:
            print(f"[noise] {sc['name']}: no matching networks in {folders['colored']} or {folders['white']}, skipped")
            continue
        print(f"[noise] {sc['name']}: colored-trained {tags['colored']}\n        white-trained {tags['white']}")
        curves = []                                      # (method, trained, step, test, snr, values)
        for test in NM:
            first = True
            for trained in NM:
                if not tags[trained]:
                    continue
                variants = [("trained", "", tags[trained])]
                nf = [t for t in tags[trained] if nf_step(folders[trained] / t) not in (None, test)]   # step differs
                if a.nf_matched and trained != test and nf:
                    variants.append(("matched", "_nfmatched", nf))
                for step, sfx, tg in variants:
                    out = res / scenario_run_name(sc) / f"train-{trained}_test-{test}{sfx}"
                    cl = a.classical if (first and step == "trained") else ""
                    cmd = [py, "scripts/evaluate.py", "--scenario", sp, "--runs", folders[trained], "--out", out,
                           "--noise-model", test, "--classical", cl, "--device", a.device, "--tags", *tg]
                    if step == "matched":
                        cmd += ["--nf-step", "matched"]
                    run(cmd)
                    snr, val = read_csv(out / "nmse_vs_snr.csv")
                    for m, v in val.items():
                        if m in CLASSICAL:
                            curves.append((m, "-", "-", test, snr, v))
                        else:
                            curves.append((m, trained, step, test, snr, v))
                    first = False
        figs.append((sc["name"], curves))
        for m, tr, st, te, snr, v in curves:
            lines.append([sc["name"], m, tr, st, te] + [f"{x:.4f}" for x in v] + [f"{v.mean():.4f}"])
    if not figs:
        sys.exit("nothing evaluated")
    snr_cols = [f"{x:g}dB" for x in figs[0][1][0][4]]
    with open(res / "noise_summary.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["scenario", "method", "trained_on", "data_step", "tested_on"] + snr_cols + ["mean"])
        w.writerows(lines)
    md = ["# Colored against white measurement noise (NMSE in dB; the same test channels and noise draw)", "",
          "trained on = the noise model of the training data; data step 'matched' = NF-LADMM run with the exact data step "
          "of the test noise (same weights).", ""]
    for name, curves in figs:
        snr = curves[0][4]
        idx = [int(np.argmin(abs(snr - q))) for q in (0, 10, 20)]
        md += [f"## {name}", "", "| method | trained on | data step | colored test 0 / 10 / 20 dB | mean | "
               "white test 0 / 10 / 20 dB | mean |", "|---|---|---|---|---|---|---|"]
        keys = list(dict.fromkeys((m, tr, st) for m, tr, st, _, _, _ in curves))
        for k in keys:
            cell = []
            for te in NM:
                v = next((c[5] for c in curves if (c[0], c[1], c[2]) == k and c[3] == te), None)
                cell += ["–", "–"] if v is None else [" / ".join(f"{v[i]:.2f}" for i in idx), f"{v.mean():.2f}"]
            md.append(f"| {k[0]} | {k[1]} | {k[2]} | " + " | ".join(cell) + " |")
        md.append("")
    (res / "noise_summary.md").write_text("\n".join(md), encoding="utf-8")
    plot(figs, res)
    print(f"\nwritten {res}: noise_comparison.png/pdf, noise_summary.csv/md")


def plot(figs, res):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    methods = list(dict.fromkeys(c[0] for _, cs in figs for c in cs if c[0] not in CLASSICAL))
    color = {m: PALETTE[i % len(PALETTE)] for i, m in enumerate(methods)}
    color.update({"amp": "#7a7a7a", "omp": "#5a5a5a", "sd": "#3a3a3a", "admm": "#9a9a9a", "l1": "#b0b0b0",
                  "grwl1": "#4a4a4a"})
    fig, axs = plt.subplots(len(figs), 3, figsize=(24, 6.4 * len(figs)), squeeze=False,
                            gridspec_kw=dict(width_ratios=[1.25, 1.25, 0.85]))
    for r, (name, curves) in enumerate(figs):
        for j, te in enumerate(NM):
            ax = axs[r, j]
            cs = [c for c in curves if c[3] == te]
            labs, last = [], []
            for m, tr, st, _, snr, v in cs:
                ls = "-" if tr in ("-", te) else (":" if st == "matched" else "--")
                lab = m if tr == "-" else f"{m} (trained {tr}" + (", step of the test)" if st == "matched" else ")")
                ax.plot(snr, v, ls=ls, color=color[m], lw=2, marker="o" if ls == "-" else None, ms=4, label=lab)
                labs.append((lab, color[m])); last.append(v[-1])
            span = (max(max(c[5]) for c in cs) - min(min(c[5]) for c in cs)) or 1
            pos = repel(last, 0.032 * span)
            x1 = cs[0][4][-1]
            for (lab, c), v, p in zip(labs, last, pos):
                ax.annotate(f"{v:.2f}  {lab}", xy=(x1, v), xytext=(x1 + 0.6, p), fontsize=6.6, color="#2b2b2b",
                            va="center", arrowprops=dict(arrowstyle="-", color=c, lw=0.6))
            ax.set_xlim(cs[0][4][0] - 0.5, x1 + 9.5)
            ax.set_title(f"{name}: {te} test noise" + (" y = A(x + n)" if te == "colored" else " y = Ax + w"), fontsize=10)
            ax.set_xlabel("SNR (dB)"); ax.set_ylabel("NMSE (dB)"); ax.grid(alpha=.25)
            for sp in ("top", "right"):
                ax.spines[sp].set_visible(False)
        bx = axs[r, 2]                                   # mismatch loss: cross - matched training, per method
        for m in methods:
            for te in NM:
                other = "white" if te == "colored" else "colored"
                own = next((c for c in curves if c[0] == m and c[1] == te and c[3] == te), None)
                for st, ls in (("trained", "--"), ("matched", ":")):
                    cr = next((c for c in curves if c[0] == m and c[1] == other and c[2] == st and c[3] == te), None)
                    if own is None or cr is None:
                        continue
                    d = cr[5] - own[5]
                    bx.plot(own[4], d, ls=ls, color=color[m], lw=2, marker="s" if te == "white" else "^", ms=5,
                            label=f"{m}: trained {other}, tested {te}" + (" (step matched)" if st == "matched" else ""))
                    for xi, di in zip(own[4][::2], d[::2]):
                        bx.annotate(f"{di:+.1f}", (xi, di), textcoords="offset points", xytext=(0, 6), ha="center",
                                    fontsize=6.3, color="#2b2b2b")
        bx.axhline(0, color="#7a7a7a", lw=1)
        bx.set_title("mismatch loss (dB): trained on the other noise − trained on the test's", fontsize=9)
        bx.set_xlabel("SNR (dB)"); bx.set_ylabel("loss (dB), > 0 = worse"); bx.grid(alpha=.25)
        if bx.lines[1:]:
            bx.legend(fontsize=6.6, frameon=False, loc="upper center", bbox_to_anchor=(.5, -.12), ncol=1)
        for sp in ("top", "right"):
            bx.spines[sp].set_visible(False)
    fig.suptitle("Solid: trained on the test's noise model.  Dashed: trained on the other one.  Dotted: NF-LADMM trained on "
                 "the other one, run with the test's exact data step.  Grey: classical.", fontsize=10)
    fig.tight_layout()
    fig.savefig(res / "noise_comparison.png", dpi=140)
    fig.savefig(res / "noise_comparison.pdf")


if __name__ == "__main__":
    main()

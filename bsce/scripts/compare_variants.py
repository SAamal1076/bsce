#!/usr/bin/env python3
"""Train model variants under one protocol on several scenarios, evaluate them on the same test
data and plot the comparison.

  python scripts/compare_variants.py --scenarios q1k_sv_upa q1k_sv_ula \
      --variants scalar:model.threshold=scalar group:model.threshold=group group_k:model.threshold=group_k \
      --set model.T=2 train.maxit=1000 train.better_wait=100000000 model.precision=float32 model.init_precision=float32 \
      --protocol gmlamp_best --out outputs_cmp --seeds 0

Every variant 'name:override ...' (overrides separated by ',') is trained with train.py (--tag LADMM-<name>[_s<k>]),
then evaluate.py runs on outputs_cmp/<scenario>/ (same test channels and noise for all), and
compare.png / compare.csv / compare.json summarise NMSE vs SNR and the difference to the first
variant (mean over seeds when several).
"""
import argparse
import csv
import json
import subprocess
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent


def run(cmd, log):
    print(">>>", " ".join(cmd), flush=True)
    with open(log, "w", encoding="utf-8") as f:
        r = subprocess.run(cmd, cwd=ROOT, stdout=f, stderr=subprocess.STDOUT, text=True)
    print(f"<<< exit {r.returncode} (log {log})", flush=True)
    return r.returncode == 0


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--scenarios", nargs="+", required=True, help="names in configs/scenarios (without .yaml)")
    ap.add_argument("--variants", nargs="+", required=True)
    ap.add_argument("--train", default="configs/train/ladmm_original.yaml")
    ap.add_argument("--protocol", default="gmlamp_best")
    ap.add_argument("--set", nargs="*", default=[])
    ap.add_argument("--seeds", nargs="+", type=int, default=[0])
    ap.add_argument("--device", default="auto")
    ap.add_argument("--out", default="outputs_cmp")
    ap.add_argument("--skip-train", action="store_true")
    ap.add_argument("--refs", nargs="*", default=[],
                    help="other model folders in the runs folder to draw as references (e.g. LAMP-authors GM-LAMP-authors)")
    a = ap.parse_args()
    py = sys.executable
    variants = [(v.split(":", 1)[0], [o for o in v.split(":", 1)[1].split(",") if o] if ":" in v else []) for v in a.variants]
    out = ROOT / a.out
    summary = {}
    for sc in a.scenarios:
        scf = f"configs/scenarios/{sc}.yaml"
        sys.path.insert(0, str(ROOT))
        from bsce.config import load_yaml
        name = load_yaml(ROOT / scf)["name"]
        logs = out / name / "logs"
        logs.mkdir(parents=True, exist_ok=True)
        if not a.skip_train:
            for vname, ov in variants:
                for s in a.seeds:
                    tag = f"LADMM-{vname}" + (f"_s{s}" if s else "")
                    if (out / name / tag / "train_log.json").exists():
                        print(f"skip {name}/{tag} (done)")
                        continue
                    run([py, "scripts/train.py", "--scenario", scf, "--train", a.train, "--protocol", a.protocol,
                         "--device", a.device, "--out", a.out, "--tag", tag, "--seed", str(s), "--set"] + a.set + ov,
                        logs / f"train_{tag}.log")
        run([py, "scripts/evaluate.py", "--scenario", scf, "--runs", f"{a.out}/{name}", "--classical", "amp",
             "--device", a.device], logs / "evaluate.log")
        rows = list(csv.DictReader(open(out / name / "nmse_vs_snr.csv", encoding="utf-8")))
        snr = [float(r["SNR_dB"]) for r in rows]
        res = {"AMP": [float(r["amp"]) for r in rows]}
        for ref in a.refs:
            if ref in rows[0]:
                res["ref:" + ref] = [float(r[ref]) for r in rows]
        for vname, _ in variants:
            cols = [c for c in rows[0] if c == f"LADMM-{vname}" or c.startswith(f"LADMM-{vname}_s")]
            if cols:
                v = np.array([[float(r[c]) for c in cols] for r in rows])
                res[vname] = v.mean(1).tolist()
                res[vname + "_std"] = (v.std(1, ddof=1) if v.shape[1] > 1 else np.zeros(len(rows))).tolist()
                res[vname + "_n"] = v.shape[1]
        summary[name] = dict(snr=snr, nmse=res)
    json.dump(summary, open(out / "compare.json", "w"), indent=1)
    base = variants[0][0]
    with open(out / "compare.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["scenario", "variant", "n_seeds"] + [f"{s:g}dB" for s in next(iter(summary.values()))["snr"]]
                   + ["mean_gain_vs_" + base + "_dB"])
        for name, d in summary.items():
            for vname, _ in variants:
                if vname in d["nmse"]:
                    v = np.array(d["nmse"][vname]); b = np.array(d["nmse"][base])
                    w.writerow([name, vname, d["nmse"][vname + "_n"]] + [f"{x:.3f}" for x in v] + [f"{np.mean(b - v):.3f}"])
            for ref in a.refs:
                if "ref:" + ref in d["nmse"]:
                    v = np.array(d["nmse"]["ref:" + ref]); b = np.array(d["nmse"][base])
                    w.writerow([name, ref, 1] + [f"{x:.3f}" for x in v] + [f"{np.mean(b - v):.3f}"])
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    n = len(summary)
    fig, axs = plt.subplots(2, n, figsize=(5.2 * n, 8.4), squeeze=False, gridspec_kw=dict(height_ratios=[2.2, 1]))
    cols = ["#2a78d6", "#eb6834", "#1f9e6e", "#8c5ab4", "#c23b6a"]
    for j, (name, d) in enumerate(summary.items()):
        ax, ax2 = axs[0, j], axs[1, j]
        snr = d["snr"]
        ax.plot(snr, d["nmse"]["AMP"], ":", color="#999999", marker="x", ms=3, label=f"AMP: {d['nmse']['AMP'][0]:.2f} / {d['nmse']['AMP'][-1]:.2f}")
        b = np.array(d["nmse"][base])
        for k, (ref, rc) in enumerate(zip(a.refs, ("#555555", "#b08a1a", "#c23b6a"))):
            if "ref:" + ref in d["nmse"]:
                v = np.array(d["nmse"]["ref:" + ref])
                ax.plot(snr, v, "--", color=rc, marker="sd^"[k], ms=3, lw=1.3, label=f"{ref}: {v[0]:.2f} / {v[-1]:.2f} dB")
                for s_, u in list(zip(snr, v))[1::2]:
                    ax.annotate(f"{u:.1f}", (s_, u), fontsize=6, color=rc, xytext=(0, -10), textcoords="offset points", ha="center")
                g = b - v
                ax2.plot(snr, g, "--", color=rc, ms=3, lw=1.1, marker="sd^"[k], label=f"{ref} - gain mean {g.mean():+.2f} dB")
        for (vname, _), c in zip(variants, cols):
            if vname not in d["nmse"]:
                continue
            v = np.array(d["nmse"][vname]); sd = np.array(d["nmse"][vname + "_std"])
            ax.plot(snr, v, "-o", color=c, ms=3.5, lw=1.6, label=f"{vname} (n={d['nmse'][vname + '_n']}): {v[0]:.2f} / {v[-1]:.2f} dB")
            if sd.any():
                ax.fill_between(snr, v - sd, v + sd, color=c, alpha=0.15)
            for s_, u in list(zip(snr, v))[::2]:
                ax.annotate(f"{u:.1f}", (s_, u), fontsize=6, color=c, xytext=(0, 5), textcoords="offset points", ha="center")
            if vname != base:
                g = b - v
                ax2.plot(snr, g, "-o", color=c, ms=3, lw=1.4, label=f"{vname} - gain mean {g.mean():+.2f} dB")
                for s_, u in list(zip(snr, g))[::2]:
                    ax2.annotate(f"{u:+.2f}", (s_, u), fontsize=6, color=c, xytext=(0, 5), textcoords="offset points", ha="center")
        ax.set_title(name, fontsize=10); ax.set_xlabel("SNR (dB)"); ax.set_ylabel("test NMSE (dB)")
        ax.grid(alpha=.3); ax.legend(frameon=False, fontsize=7)
        ax2.axhline(0, color="#555555", lw=0.8)
        ax2.set_xlabel("SNR (dB)"); ax2.set_ylabel(f"gain over {base} (dB)"); ax2.grid(alpha=.3)
        ax2.legend(frameon=False, fontsize=7)
        for x in (ax, ax2):
            for s_ in ("top", "right"):
                x.spines[s_].set_visible(False)
    fig.suptitle(f"variants under one protocol ({a.protocol}; {' '.join(a.set)})", fontsize=9)
    fig.tight_layout(); fig.savefig(out / "compare.png", dpi=150)
    print("written", out / "compare.png", out / "compare.csv", out / "compare.json")


if __name__ == "__main__":
    main()

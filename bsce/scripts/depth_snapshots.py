#!/usr/bin/env python3
"""Depth curve on the TEST set: NMSE of the network saved when each layer was finished (train.py --save-layers).

  python scripts/depth_snapshots.py --runs outputs/DeepMIMO_UPA/LADMM-best-T12 outputs/DeepMIMO_ULA/LADMM-best-T12
  python scripts/depth_snapshots.py --runs "outputs/*/LADMM-T12" "outputs/*/LAMP-T16" --name depth_ladmm_lamp
  (--runs accepts patterns: * matches any scenario or name; every match that has a layers/ folder is used)
  options: --band 0to20dB (default: the only/first band found)   --tol 0.1 (dB, "flat" threshold)
           --snr 0 10 20 (SNRs drawn as curves; the mean is over all test SNRs of the scenario)
           --device auto   --name depth (results/<name>/)   --out <dir> (instead of results/<name>)

For every run folder: reads <run>/layers/T<t>_<tag>_<band>.pt and the scenario stored in <run>/train_log.json,
evaluates every T on the scenario's test channels with exactly the noise of evaluate.py
(NumpyMatlabRNG(seed + 1000 + i) at test SNR i; white for a run trained with --noise-model white), and writes
  depth_curve.png / .pdf   one column per run: (top) test NMSE vs T at the chosen SNRs + the mean over all test SNRs,
                           (bottom) validation NMSE logged when layer T was finished + gain of each added layer
  depth_curve.csv          run, T, n_params, validation, test NMSE at every SNR, mean
  depth_curve.md           the same as a table, with the flat point of every run
Flat point: the smallest T whose mean test NMSE is within --tol dB of the best T of that run.
"""
import argparse
import json
import re
import sys
from pathlib import Path

import numpy as np
import scipy.io as sio
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from bsce.config import resolve, scenario_cs_matrix, scenario_test_dir  # noqa: E402
from bsce.device import get_device, describe  # noqa: E402
from bsce.export import load_model  # noqa: E402
from bsce.data import measurements  # noqa: E402
from bsce.data import noise as noise_models  # noqa: E402
from bsce.rng import NumpyMatlabRNG  # noqa: E402

PAT = re.compile(r"^T(\d+)_(.+)_([^_]+dB)\.pt$")


def nmse_db(xh, x):
    return float(10 * np.log10(np.sum(np.abs(xh - x) ** 2) / np.sum(np.abs(x) ** 2)))


def snapshots(run, band):
    found = {}
    for p in sorted((run / "layers").glob("T*_*.pt")):
        m = PAT.match(p.name)
        if m:
            found.setdefault(m.group(3), {})[int(m.group(1))] = p
    if not found:
        sys.exit(f"no layer snapshots in {run / 'layers'} (train with --save-layers)")
    b = band if band else sorted(found)[0]
    if b not in found:
        sys.exit(f"band {b} not in {run / 'layers'}: {sorted(found)}")
    return b, dict(sorted(found[b].items()))


def evaluate_run(run, band, dev):
    log = json.loads((run / "train_log.json").read_text(encoding="utf-8"))
    sc = log["scenario"]
    b, snaps = snapshots(run, band)
    pre = f"{sc['dataset']}_{sc['array']}"
    A = sio.loadmat(scenario_cs_matrix(sc))["A"].astype(np.float64)
    x = sio.loadmat(scenario_test_dir(sc, False) / f"{pre}_test_channel.mat")["x"]
    snr = [float(s) for s in sc["test_snr_db"]]
    nm = noise_models.of_scenario(sc)                   # test noise = the noise the run was trained with
    ys = [measurements.gen_measurements_matlab(x, A, s, NumpyMatlabRNG(int(sc["seed"]) + 1000 + i), noise_model=nm)
          for i, s in enumerate(snr)]
    rows = []
    for t, p in snaps.items():
        net = load_model(p).to(dev).eval()
        ck = torch.load(p, map_location="cpu", weights_only=False)
        row = dict(run=run.name, scenario=sc["name"], band=b, T=t, n_params=net.n_params(),
                   val_db=float(ck["data_info"]["val_nmse_db"]))
        with torch.no_grad():
            for s, y in zip(snr, ys):
                yt = torch.as_tensor(y).to(dev)
                if getattr(net, "needs_noise_var", False):
                    xh = net(yt, s2=np.full(y.shape[1], 10.0 ** (-s / 10)))
                else:
                    xh = net(yt)
                row[f"{s:g}dB"] = nmse_db(xh.cpu().numpy(), x)
        row["mean"] = float(np.mean([row[f"{s:g}dB"] for s in snr]))
        rows.append(row)
        print(f"  {sc['name']}/{run.name:18s} T={t:2d}  val {row['val_db']:8.3f}  test mean {row['mean']:8.3f}  "
              + "  ".join(f"{s:g}:{row[f'{s:g}dB']:.2f}" for s in snr))
    return sc, b, snr, rows


def flat_point(rows, tol):
    m = np.array([r["mean"] for r in rows])
    ok = [r["T"] for r, v in zip(rows, m) if v <= m.min() + tol]
    return min(ok), rows[int(np.argmin(m))]["T"]


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--runs", nargs="+", required=True)
    ap.add_argument("--band", default=None)
    ap.add_argument("--tol", type=float, default=0.1)
    ap.add_argument("--snr", nargs="+", type=float, default=[0, 10, 20])
    ap.add_argument("--device", default="auto")
    ap.add_argument("--out", default=None)
    ap.add_argument("--name", default="depth")
    a = ap.parse_args()
    dev = get_device(a.device)
    print(f"[bsce] device: {describe(dev)}")
    import glob
    runs = []
    for r in a.runs:
        hits = sorted(glob.glob(str(resolve(r)))) if any(c in r for c in "*?[") else [str(resolve(r))]
        runs += [Path(h) for h in hits if (Path(h) / "layers").is_dir() and Path(h) not in runs]
    if not runs:
        sys.exit(f"no run folder with a layers/ sub-folder matches {a.runs} (train with --save-layers)")
    print("runs:", *[f"{r.parent.name}/{r.name}" for r in runs], sep="\n  ")
    out = resolve(a.out) if a.out else resolve("results") / a.name
    out.mkdir(parents=True, exist_ok=True)
    res = [(r, *evaluate_run(r, a.band, dev)) for r in runs]

    allrows = [row for *_, rows in res for row in rows]
    cols = list(dict.fromkeys(k for row in allrows for k in row))
    with open(out / "depth_curve.csv", "w", encoding="utf-8") as f:
        f.write(",".join(cols) + "\n")
        for row in allrows:
            f.write(",".join(f"{row.get(c, '')}" if not isinstance(row.get(c), float) else f"{row[c]:.4f}"
                             for c in cols) + "\n")
    md = [f"# Depth curve (test NMSE in dB; flat point = smallest T within {a.tol:g} dB of the best mean)\n"]
    for r, sc, b, snr, rows in res:
        fT, bT = flat_point(rows, a.tol)
        md.append(f"\n## {sc['name']} / {r.name} / band {b}: best T = {bT}, flat from T = {fT}\n")
        md.append("| T | params | validation | " + " | ".join(f"{s:g} dB" for s in snr) + " | mean |")
        md.append("|---" * (len(snr) + 4) + "|")
        for row in rows:
            md.append(f"| {row['T']} | {row['n_params']:,} | {row['val_db']:.3f} | "
                      + " | ".join(f"{row[f'{s:g}dB']:.2f}" for s in snr) + f" | {row['mean']:.3f} |")
    (out / "depth_curve.md").write_text("\n".join(md) + "\n", encoding="utf-8")

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    colors = ["#c23b6a", "#e8743b", "#1f9e6e", "#7b3fb5", "#b8860b"]
    fig, axs = plt.subplots(2, len(res), figsize=(6.6 * len(res), 9), squeeze=False)
    for j, (r, sc, b, snr, rows) in enumerate(res):
        T = [row["T"] for row in rows]
        fT, bT = flat_point(rows, a.tol)
        ax, bx = axs[0, j], axs[1, j]
        series = [(f"{s:g} dB", [row[f"{s:g}dB"] for row in rows], colors[i % len(colors)])
                  for i, s in enumerate(a.snr) if f"{s:g}dB" in rows[0]]
        series.append((f"mean {snr[0]:g}-{snr[-1]:g} dB", [row["mean"] for row in rows], "#2a78d6"))
        for k, (lab, v, c) in enumerate(series):
            ax.plot(T, v, "-o", ms=4, color=c, lw=2.2 if lab.startswith("mean") else 1.4, label=lab)
            for t_, v_ in zip(T, v):
                ax.annotate(f"{v_:.2f}", (t_, v_), xytext=(0, 6 if k % 2 == 0 else -11), textcoords="offset points",
                            ha="center", fontsize=6.5, color=c)
        ax.axvline(fT, color="k", ls=":", lw=1)
        ax.text(fT, ax.get_ylim()[1], f" flat from T={fT}\n (within {a.tol:g} dB of best T={bT})",
                va="top", fontsize=8)
        ax.set_title(f"{sc['name']} / {r.name}: test NMSE of the network saved after layer T", fontsize=9.5)
        ax.set_xlabel("layers T"); ax.set_ylabel("test NMSE (dB)"); ax.set_xticks(T)
        ax.legend(fontsize=8, frameon=False, loc="best"); ax.grid(alpha=.3)
        v = [row["val_db"] for row in rows]
        bx.plot(T, v, "-o", ms=4, color="#2a78d6", label=f"validation, band {b}")
        for t_, v_ in zip(T, v):
            bx.annotate(f"{v_:.2f}", (t_, v_), xytext=(0, 6), textcoords="offset points", ha="center", fontsize=7)
        g = [np.nan] + [v[i - 1] - v[i] for i in range(1, len(v))]
        b2 = bx.twinx()
        b2.bar(T, g, width=.5, color="#1f9e6e", alpha=.35, label="gain of the added layer (dB)")
        for t_, g_ in zip(T[1:], g[1:]):
            b2.annotate(f"{g_:+.2f}", (t_, g_), xytext=(0, 2), textcoords="offset points", ha="center",
                        fontsize=6.5, color="#1f9e6e")
        b2.axhline(0, color="#1f9e6e", lw=.8)
        b2.set_ylabel("gain of the added layer (dB)", color="#1f9e6e")
        for a_ in (bx, b2):
            a_.set_xlim(min(T) - .6, max(T) + .6)
        ax.set_xlim(min(T) - .6, max(T) + .6)
        bx.set_title("validation NMSE logged when layer T was finished", fontsize=10)
        bx.set_xlabel("layers T"); bx.set_ylabel("validation NMSE (dB)"); bx.set_xticks(T); bx.grid(alpha=.3)
        h1, l1 = bx.get_legend_handles_labels(); h2, l2 = b2.get_legend_handles_labels()
        bx.legend(h1 + h2, l1 + l2, fontsize=8, frameon=False, loc="upper right")
    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(out / f"depth_curve.{ext}", dpi=160)
    print(f"written {out / 'depth_curve.png'}, depth_curve.csv, depth_curve.md")
    for r, sc, b, snr, rows in res:
        fT, bT = flat_point(rows, a.tol)
        print(f"  {sc['name']}/{r.name}: best T = {bT}, flat from T = {fT} (tol {a.tol:g} dB)")


if __name__ == "__main__":
    main()

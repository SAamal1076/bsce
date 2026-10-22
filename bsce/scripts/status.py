#!/usr/bin/env python3
"""Checklist of the experiment plan (RUNBOOK.md): which steps are done, which are next.

  python scripts/status.py            -> prints the checklist and writes STATUS.md
  python scripts/status.py --data     -> also checks the data fingerprints (values the review used)

A step is 'done' when its expected files exist. Model folders are looked up by tag
(outputs/<scenario>/<tag>/train_log.json); evaluation by outputs/<scenario>/nmse_vs_snr.csv.
"""
import argparse
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SC = ["SV_ULA", "SV_UPA", "DeepMIMO_ULA", "DeepMIMO_UPA"]
MAIN = ["LAMP_gmlamp", "GM-LAMP-spread_gmlamp", "LISTA_gmlamp", "ALISTA_gmlamp", "LVAMP_gmlamp", "ADMM-Net_gmlamp",
        "LADMM-full_gmlamp", "LADMM-diagprior_gmlamp", "LADMM-psd_gmlamp", "NF-LADMM-S", "NF-LADMM-C"]
SEED_METHODS = ["LAMP_gmlamp", "GM-LAMP-spread_gmlamp", "LISTA_gmlamp", "LADMM-full_gmlamp", "LADMM-diagprior_gmlamp"]
SCALABLE = ["LADMM-diag_gmlamp", "LADMM-lowrank_gmlamp", "LADMM-kron_gmlamp", "LADMM-kronprior_gmlamp"]

# fingerprints of the data used in the review (test x[0,0], sum |x|^2 of the test set, training x[5,7])
FP = {"SV_ULA": (0.009743 + 0.210479j, 508185.925, -0.899453 - 0.536304j),
      "SV_UPA": (-0.045484 + 0.167336j, 505719.42, -0.003831 + 0.01958j),
      "DeepMIMO_ULA": (0.027395 + 0.037722j, 512000.0, 0.000864 - 0.003146j),
      "DeepMIMO_UPA": (-0.037769 + 0.034738j, 512000.0, -0.012252 + 0.042254j)}


def run_done(sc, tag):
    return (ROOT / "outputs" / sc / tag / "train_log.json").exists()


def steps():
    S = []
    S.append(("S0a", "data ready (4 scenarios, measurements)", [
        ROOT / "data" / s / f"{s}_training_128_measurements_10to20dB.mat" for s in SC]))
    S.append(("S0b", "speed test: device and precision", [ROOT / "outputs/speed_test.json"]))
    S.append(("S1", "sanity: LAMP + GM-LAMP retrained on SV_ULA", [
        ROOT / "outputs/SV_ULA" / t / "train_log.json" for t in ("LAMP_gmlamp", "GM-LAMP-spread_gmlamp")]))
    S.append(("S2", "depth of LADMM and NF-LADMM on SV_ULA, SV_UPA (select_depth.py)", [
        ROOT / "outputs/depth_select/SV_ULA/depth_select.json",
        ROOT / "outputs/depth_select/SV_UPA/depth_select.json"]))
    for s in SC:
        S.append((f"S3-{s}", f"main comparison {s} (9 learned + NF-LADMM-S/C + AMP/OMP) + evaluation",
                  [ROOT / "outputs" / s / t / "train_log.json" for t in MAIN] + [ROOT / "outputs" / s / "nmse_vs_snr.csv"]))
    for s in SC:
        S.append((f"S4-{s}", f"seeds 1, 2 for 5 methods on {s} + aggregate",
                  [ROOT / "outputs" / s / f"{t}_s{k}" / "train_log.json" for t in SEED_METHODS for k in (1, 2)]
                  + [ROOT / "outputs" / s / "nmse_vs_snr_seeds.csv"]))
    for s in ("SV_UPA", "DeepMIMO_UPA"):
        S.append((f"S5-{s}", f"scalable LADMM operators on {s}",
                  [ROOT / "outputs" / s / t / "train_log.json" for t in SCALABLE]))
    for k, d in (("S6", "complexity: params / FLOPs / latency (script provided at this step)"),
                 ("S7", "robustness: new combiner, SNR mismatch (script provided at this step)"),
                 ("S8", "cross-dataset SV <-> DeepMIMO (script provided at this step)"),
                 ("S9", "per-channel NMSE CDF and sum-rate (script provided at this step)"),
                 ("S10", "ablations alpha=0, tied, positivity (script provided at this step)")):
        S.append((k, d, [ROOT / "outputs" / f"{k}_done.flag"]))
    return S


def check_data():
    import numpy as np
    import scipy.io as sio
    print("\ndata fingerprints (must match the review):")
    for s, (a, b, c) in FP.items():
        try:
            x = sio.loadmat(ROOT / "data" / s / f"{s}_test_channel.mat")["x"]
            t = sio.loadmat(ROOT / "data" / s / f"{s}_training_channel.mat")["x"]
            ok = abs(x[0, 0] - a) < 1e-5 and abs(np.sum(abs(x) ** 2) - b) < 1e-2 and abs(t[5, 7] - c) < 1e-5
            print(f"  {s:13s} {'OK' if ok else 'DIFFERENT'}   test x[0,0]={x[0, 0]:.6f}  sum|x|^2={np.sum(abs(x) ** 2):.3f}  "
                  f"train x[5,7]={t[5, 7]:.6f}")
        except Exception as e:  # noqa: BLE001
            print(f"  {s:13s} MISSING ({e})")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", action="store_true")
    a = ap.parse_args()
    lines = ["# STATUS (scripts/status.py)", "", "| step | task | done | missing files |", "|---|---|---|---|"]
    nxt = None
    for sid, desc, files in steps():
        miss = [f for f in files if not f.exists()]
        done = len(files) - len(miss)
        mark = "DONE" if not miss else ("partial" if done else "todo")
        if miss and nxt is None:
            nxt = sid
        short = ", ".join(str(f.relative_to(ROOT)).replace("\\", "/") for f in miss[:3]) + (" ..." if len(miss) > 3 else "")
        lines.append(f"| {sid} | {desc} | {mark} ({done}/{len(files)}) | {short} |")
        print(f"[{'x' if not miss else ' '}] {sid:16s} {mark:8s} {done}/{len(files)}  {desc}")
    lines += ["", f"next step: {nxt}"]
    (ROOT / "STATUS.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"\nnext step: {nxt}   (STATUS.md written)")
    if a.data:
        check_data()


if __name__ == "__main__":
    main()

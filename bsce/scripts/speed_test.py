#!/usr/bin/env python3
"""Short timed trainings to choose device and precision before the long runs.

  python scripts/speed_test.py                     (SV_ULA, band 0to10dB, ~5-15 min in total)
  python scripts/speed_test.py --iters 500 --configs gmlamp_spread ladmm_original

For each config x device (cuda, cpu) x precision (float64, float32) a 2-layer network is trained
for a fixed number of iterations per stage (no early stop), with --protocol gmlamp, and the wall
time is recorded. Writes outputs/speed_test.json / .csv / .png and prints the fastest setting per
method. The precision only changes speed and round-off; the protocol is the same.
"""
import argparse
import csv
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--scenario", default="configs/scenarios/sv_ula.yaml")
    ap.add_argument("--configs", nargs="+", default=["lamp_original", "gmlamp_spread", "ladmm_original", "lista"])
    ap.add_argument("--iters", type=int, default=300)
    ap.add_argument("--devices", nargs="+", default=["cuda", "cpu"])
    a = ap.parse_args()
    import torch
    devs = [d for d in a.devices if d != "cuda" or torch.cuda.is_available()]
    res = []
    for c in a.configs:
        for d in devs:
            for p in ("float64", "float32"):
                tag = f"speed_{c}_{d}_{p}"
                cmd = [sys.executable, "scripts/train.py", "--scenario", a.scenario, "--train", f"configs/train/{c}.yaml",
                       "--protocol", "gmlamp", "--device", d, "--bands", "0to10dB", "--tag", tag, "--out", "outputs/speed",
                       "--set", "model.T=2", f"train.maxit={a.iters}", "train.better_wait=100000000",
                       f"model.precision={p}"]
                if "ladmm" in c:
                    cmd.append(f"model.init_precision={p}")
                print(">>>", " ".join(cmd), flush=True)
                r = subprocess.run(cmd, cwd=ROOT, capture_output=True, text=True, encoding="utf-8", errors="replace")
                if r.returncode != 0:
                    print(r.stdout[-1500:], r.stderr[-1500:])
                    res.append(dict(config=c, device=d, precision=p, seconds=None, val_nmse_db=None))
                    continue
                log = json.loads(next((ROOT / "outputs/speed").glob(f"*/{tag}/train_log.json")).read_text(encoding="utf-8"))
                b = log["bands"]["0to10dB"]
                res.append(dict(config=c, device=d, precision=p, seconds=round(b["train_seconds"], 1),
                                val_nmse_db=round(b["final_val_nmse_db"], 3)))
                print(f"    {c:16s} {d:5s} {p}: {b['train_seconds']:.1f} s", flush=True)
    out = ROOT / "outputs"
    (out / "speed_test.json").write_text(json.dumps(res, indent=1), encoding="utf-8")
    with open(out / "speed_test.csv", "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(res[0])); w.writeheader(); w.writerows(res)
    print("\nfastest per method:")
    for c in a.configs:
        ok = [r for r in res if r["config"] == c and r["seconds"]]
        if ok:
            b = min(ok, key=lambda r: r["seconds"])
            print(f"  {c:16s} -> --device {b['device']}  model.precision={b['precision']}  ({b['seconds']} s; "
                  + ", ".join(f"{r['device']}/{r['precision']} {r['seconds']} s" for r in ok) + ")")
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    labels = [f"{r['config']}\n{r['device']}/{r['precision'][5:]}" for r in res if r["seconds"]]
    vals = [r["seconds"] for r in res if r["seconds"]]
    fig, ax = plt.subplots(figsize=(max(8, 0.7 * len(vals)), 4.5))
    bars = ax.bar(range(len(vals)), vals, color="#2a78d6")
    for i, v in enumerate(vals):
        ax.text(i, v, f"{v:.0f}s", ha="center", va="bottom", fontsize=7)
    ax.set_xticks(range(len(vals))); ax.set_xticklabels(labels, fontsize=6.5)
    ax.set_ylabel("seconds (same number of iterations)"); ax.set_title("speed test", fontsize=10)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    fig.tight_layout(); fig.savefig(out / "speed_test.png", dpi=150)
    print(f"written {out / 'speed_test.json'}, speed_test.csv, speed_test.png")


if __name__ == "__main__":
    main()

"""Figure of smoke test 3: validation NMSE per epoch of the short UTRR trainings (solid), NMSE on 2,000 training
channels (dashed, where logged), values printed, the paper's Table II values as reference lines.
usage: python plot_smoke3.py <folder with the json files> <out.png>"""
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

runs = [json.loads(f.read_text()) for f in sorted(Path(sys.argv[1]).glob("*.json"))]
runs.sort(key=lambda r: (r["args"]["sign"] != "correct", r["args"]["bn"], -r["args"]["ntrain"]))
pal = {(0, "correct", 80000): "#1f9e6e", (0, "correct", 20000): "#2a78d6", (1, "correct", 20000): "#c23b6a",
       (0, "paper", 20000): "#7a7a7a", (1, "paper", 20000): "#9a9a9a"}
fig, ax = plt.subplots(1, 2, figsize=(17, 5.6), gridspec_kw=dict(width_ratios=[1.35, 1]))
paper = {20: -26.93, 10: -18.92}
lo, hi = -30, 12
for r in runs:
    a = r["args"]
    k = 0 if a["snr"] == 20 else 1
    c = pal.get((a["bn"], a["sign"], a["ntrain"]), "#555")
    ep = [l["epoch"] for l in r["log"]]
    v = [l["val_pooled_dB"] for l in r["log"]]
    lab = (f"BN {'on' if a['bn'] else 'off'}, {'printed' if a['sign'] == 'paper' else 'eq. 16'} sign, "
           f"{a['ntrain']:,} channels: test {r['test_pooled_dB']:.2f} dB ({r['log'][-1]['epoch']} epochs, {r['minutes']} min)")
    vc = [min(max(x, lo + 0.3), hi - 0.3) for x in v]
    ax[k].plot(ep, vc, "-o", ms=2.8, color=c, lw=1.5, label=lab)
    tr = [(l["epoch"], l["train2k_pooled_dB"]) for l in r["log"] if l.get("train2k_pooled_dB") is not None]
    if tr:
        ax[k].plot([e for e, _ in tr], [max(t, lo + 0.3) for _, t in tr], "--", color=c, lw=1.0, alpha=.8)
        ax[k].annotate(f"train {tr[-1][1]:.2f}", tr[-1], textcoords="offset points", xytext=(4, -10), fontsize=7.2, color=c)
    ax[k].annotate(f"{v[-1]:.2f}", (ep[-1], vc[-1]), textcoords="offset points", xytext=(4, 3), fontsize=7.6, color=c,
                   fontweight="bold")
    if v[0] > hi:
        ax[k].annotate(f"starts at {v[0]:+.1f} dB", (ep[0], hi - 0.3), textcoords="offset points", xytext=(16, -14),
                       fontsize=7.2, color=c)
for k, snr in enumerate((20, 10)):
    ax[k].axhline(paper[snr], color="#222", ls="--", lw=1.1)
    ax[k].text(0.5, paper[snr] + 0.4, f"paper, Table II (K = 16, full training): {paper[snr]} dB", fontsize=8)
    ax[k].set_ylim(lo, hi if k == 0 else 0)
    ax[k].set_title(f"SNR {snr} dB, noise after the combiner (the paper's setting): solid = validation, dashed = 2,000 "
                    "training channels", fontsize=8.5)
    ax[k].set_xlabel("epoch (0 = untrained: 30 layers of the unrolled ITRR)"); ax[k].set_ylabel("NMSE (dB)")
    ax[k].grid(alpha=.3); ax[k].legend(frameon=False, fontsize=7.4, loc="upper right")
fig.suptitle("Smoke test 3: short UTRR trainings on a CPU (K = 16, 30 layers; the paper: 80,000 channels, up to 300 epochs "
             "on a GPU)", fontsize=10)
fig.tight_layout(); fig.savefig(sys.argv[2], dpi=150)
print("written", sys.argv[2])

"""Figure: the first layer retrained on our data (bsce, paper protocol) against the authors' training logs, and the
spread that the 10 random chunk SNRs of the authors' data recipe cause by themselves."""
import json, sys
from pathlib import Path
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
B = Path(sys.argv[1])                     # bsce folder
sys.path.insert(0, str(B / "scripts"))
import check_vs_authors as C
spread = {json.loads(l)["scenario"]: json.loads(l) for l in open(sys.argv[2])}
fig, axs = plt.subplots(1, 3, figsize=(18, 5.6), gridspec_kw=dict(width_ratios=[1.2, 1.2, 1]))
names = ["linear", "lin. r0", "lin. r1", "lin. r2", "non-lin.", "nl. r0", "nl. r1", "nl. r2"]
for ax, sc in zip(axs[:2], ["DeepMIMO_UPA", "DeepMIMO_ULA"]):
    ours = C.stage_log_of(B / f"outputs_e4/{sc}/LAMP-T1_paper", "0to10dB")
    _, _, la = C.authors_state(B / "reference/gmlamp/Train/Trained_Networks", False, sc, "0to10dB")
    _, _, ga = C.authors_state(B / "reference/gmlamp/Train/Trained_Networks", True, sc, "0to10dB")
    k = np.arange(1, 9)
    o, a, g = [s["nmse_db"] for s in ours], [s["nmse_db"] for s in la[:8]], [s["nmse_db"] for s in ga[:4]]
    ax.plot(k, a, "o-", color="#B3412E", label="authors, LAMP run (their log)")
    ax.plot(k[:4], g, "^--", color="#D99A00", label="authors, GM-LAMP run (same linear stages)")
    ax.plot(k, o, "s-", color="#2F62C8", label="bsce on our data, paper protocol")
    for i in range(8):
        ax.annotate(f"{o[i]:.2f}\n({ours[i]['iters'] // 1000}k)", (k[i], o[i]), textcoords="offset points", xytext=(0, 8), ha="center", fontsize=7.5, color="#2F62C8")
        ax.annotate(f"{a[i]:.2f}\n({la[i]['iters'] // 1000}k)", (k[i], a[i]), textcoords="offset points", xytext=(0, -24), ha="center", fontsize=7.5, color="#B3412E")
    for i in range(4):
        ax.annotate(f"{g[i]:.2f}", (k[i], g[i]), textcoords="offset points", xytext=(14, 2), ha="left", fontsize=7.5, color="#B07A00")
    ax.set_xticks(k, names, fontsize=8)
    ax.margins(y=0.18)
    ax.set_title(f"{sc}, LAMP 0-10 dB, first layer: validation NMSE after each stage (iterations)", fontsize=9.5)
    ax.set_ylabel("validation NMSE at 10 dB (dB)")
    ax.grid(alpha=0.3); ax.legend(fontsize=8, loc="lower left")
ax = axs[2]
for sc, c in (("DeepMIMO_UPA", "#2F62C8"), ("DeepMIMO_ULA", "#B3412E")):
    s = spread[sc]
    ax.scatter(s["other_mean_snr"], s["other_draws_db"], color=c, s=22, label=f"{sc}: 12 other draws (sd {s['sd_db']:.2f} dB)")
    ax.scatter([s["ours_mean_chunk_snr"]], [s["ours_db"]], color=c, marker="*", s=180, edgecolor="k", label=f"{sc}: our files ({s['ours_db']:.2f})")
    for x_, y_ in zip(s["other_mean_snr"], s["other_draws_db"]):
        ax.annotate(f"{y_:.2f}", (x_, y_), textcoords="offset points", xytext=(4, 3), fontsize=6.5, color=c)
ax.set_xlabel("mean of the 10 chunk SNRs of the training file (dB)")
ax.set_ylabel("best linear first stage (least squares), validation NMSE (dB)")
ax.set_title("The authors' recipe draws 10 SNRs per file: that alone moves stage 1 by ± 0.2 dB", fontsize=9.5)
ax.grid(alpha=0.3); ax.legend(fontsize=7.5, loc="lower left")
fig.tight_layout(); fig.savefig(sys.argv[3], dpi=125)

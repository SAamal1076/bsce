"""Figure and tables of the NF-LADMM noise smoke test, from compare_noise.py's noise_summary.csv.

Lines per scenario and test noise (NF-LADMM-SR, T = 4, 3,350 / 8,982 parameters, 4,000 CPU steps, one network 0-20 dB):
  NF colored : trained on colored data, colored (exact) data step
  NF white   : trained on white data, white (exact) data step
  NF white, colored step : trained on white data with the colored step (the mismatch on purpose)
  NF white, adaptive     : trained on white data, white step with the SNR-adaptive penalty (model.white_step=adaptive)
  NF colored, white step : the colored-trained network run with the white step (same weights; on white test only)
  NF white, colored step at test : the white-trained network run with the colored step (same weights; colored test)
  GM-LAMP published (2 networks, trained on colored noise), AMP
usage: python plot_nf_noise_smoke.py <noise_summary.csv> <out folder>
"""
import csv
import json
import sys
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

src, out = Path(sys.argv[1]), Path(sys.argv[2])
out.mkdir(parents=True, exist_ok=True)
rows = list(csv.DictReader(open(src, encoding="utf-8")))
snr_cols = [k for k in rows[0] if k.endswith("dB")]
snr = np.array([float(k[:-2]) for k in snr_cols])
T = "NF-LADMM-SR-T4_paper_steps-4000"
TM = T + "_noise-model-colored"
# (label, colour, line style, selector(row) -> bool)
LINES = [
    ("NF-LADMM, trained colored (colored step)", "#2a78d6", "-", lambda r: r["method"] == T and r["trained_on"] == "colored" and r["data_step"] == "trained"),
    ("NF-LADMM, trained white (white step)", "#eb6834", "-", lambda r: r["method"] == T and r["trained_on"] == "white" and r["data_step"] == "trained"),
    ("NF-LADMM, trained white with the colored step", "#1baf7a", "-", lambda r: r["method"] == TM and r["data_step"] == "trained"),
    ("NF-LADMM, trained white (white step, SNR-adaptive penalty)", "#e87ba4", "-",
     lambda r: r["method"] == T + "_white-step-adaptive" and r["data_step"] == "trained"),
    ("NF-LADMM trained colored, run with the white step", "#2a78d6", ":", lambda r: r["method"] == T and r["trained_on"] == "colored" and r["data_step"] == "matched"),
    ("NF-LADMM trained white, run with the colored step", "#eb6834", ":", lambda r: r["method"] == T and r["trained_on"] == "white" and r["data_step"] == "matched"),
    ("GM-LAMP published (trained colored, 2 networks)", "#6250d6", "--", lambda r: r["method"] == "GM-LAMP-authors"),
    ("AMP", "#7a7a7a", "--", lambda r: r["method"] == "amp"),
]
scen = list(dict.fromkeys(r["scenario"] for r in rows))
data = {}
for s in scen:
    for te in ("colored", "white"):
        for lab, c, ls, sel in LINES:
            r = [r for r in rows if r["scenario"] == s and r["tested_on"] == te and sel(r)]
            if r:
                data[(s, te, lab)] = np.array([float(r[0][k]) for k in snr_cols])
(out / "nf_noise_smoke.json").write_text(json.dumps({f"{k[0]} | test {k[1]} | {k[2]}": v.round(4).tolist()
                                                     for k, v in data.items()} | {"snr_db": snr.tolist()}, indent=1))

PTS = [0, 10, 20]
idx = [int(np.argmin(abs(snr - q))) for q in PTS]
fig, axs = plt.subplots(len(scen), 2, figsize=(17, 6.0 * len(scen)), squeeze=False)
for i, s in enumerate(scen):
    for j, te in enumerate(("colored", "white")):
        ax = axs[i, j]
        present = [(lab, c, ls) for lab, c, ls, _ in LINES if (s, te, lab) in data]
        for lab, c, ls in present:
            v = data[(s, te, lab)]
            ax.plot(snr, v, ls=ls, color=c, lw=2, marker="o" if ls == "-" else None, ms=4, label=lab)
        for q in idx:                                    # values at 0 / 10 / 20 dB, right of the points, pushed apart
            items = sorted([(data[(s, te, lab)][q], c) for lab, c, ls in present], reverse=True)
            ys = [items[0][0]]
            for v_, _ in items[1:]:
                ys.append(min(v_, ys[-1] - 0.85))
            for (v_, c), y_ in zip(items, ys):
                ax.text(snr[q] + 0.35, y_, f"{v_:.2f}", color="#2b2b2b", fontsize=7, va="center", ha="left",
                        bbox=dict(fc="white", ec=c, lw=0.8, pad=1.2, alpha=0.9))
        ax.set_xlim(-0.6, 22.6)
        lo = min(min(data[(s, te, lab)]) for lab, c, ls in present)
        ax.set_ylim(lo - 0.9 * len(present) * 0.35 - 0.5, None)        # room for the pushed-apart labels at 20 dB
        ax.set_title(f"{s}: {te} test noise" + (" y = A(x + n)" if te == "colored" else " y = Ax + w"), fontsize=10)
        ax.set_xlabel("SNR (dB)"); ax.set_ylabel("NMSE (dB), 2,000 test channels"); ax.grid(alpha=.25)
        ax.legend(frameon=False, fontsize=7.5, loc="upper right")
        for sp in ("top", "right"):
            ax.spines[sp].set_visible(False)
fig.suptitle("NF-LADMM-SR (T = 4, one network for 0–20 dB, 4,000 CPU steps) under colored and white measurement noise: "
             "the same channels, the same noise draw", fontsize=10.5)
fig.tight_layout(); fig.savefig(out / "nf_noise_smoke.png", dpi=150)

# markdown tables (0 / 10 / 20 dB and the mean over the 11 test SNRs)
md = []
for s in scen:
    md += [f"### {s}", "", "| network | colored test 0 / 10 / 20 dB | mean | white test 0 / 10 / 20 dB | mean |", "|---|---|---|---|---|"]
    for lab, c, ls, _ in LINES:
        cells = []
        for te in ("colored", "white"):
            v = data.get((s, te, lab))
            cells += ["–", "–"] if v is None else [" / ".join(f"{v[q]:.2f}" for q in idx), f"{v.mean():.2f}"]
        md.append(f"| {lab} | " + " | ".join(cells) + " |")
    md.append("")
(out / "nf_noise_smoke_tables.md").write_text("\n".join(md), encoding="utf-8")
print("\n".join(md))
print("written", out)

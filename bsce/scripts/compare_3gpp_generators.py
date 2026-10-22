#!/usr/bin/env python3
"""The same 38.901 setting from several generators (Sionna, QuaDRiGa with and without the elevation fixes): beamspace
statistics per array, LOS fraction, LMMSE NMSE at 10 dB. Every figure prints its numbers.

  python scripts/compare_3gpp_generators.py --case UPA Sionna "raw/sionna/H_3gpp_upa_part*.mat"
                                            --case UPA QuaDRiGa "raw/quadriga/H_3gpp_upa_part*.mat" ...
--case PANEL NAME PATTERN   one line of one panel; PANEL = UPA, ULA_Y or ULA (also UPA_v19 ...: the array is read from
                            the files); the first NAME of a panel is drawn solid, the next dashed, then dotted.
Writes <out>/generators.png, generators.json, generators.md.
"""
import argparse
import glob
import json
import sys
from pathlib import Path

import numpy as np
import scipy.io as sio

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))
from check_3gpp import energy_curve, lmmse_nmse, to_beamspace  # noqa: E402
from make_dataset import load_3gpp  # noqa: E402

COL = ["#2a78d6", "#eb6834", "#1f9e6e", "#c23b6a"]
LS = ["-", "--", ":", "-."]


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--case", nargs=3, action="append", metavar=("PANEL", "NAME", "PATTERN"), required=True)
    ap.add_argument("--n", type=int, default=10000, help="the first n channels of every case (in generation order)")
    ap.add_argument("--out", default="docs/3gpp_check")
    a = ap.parse_args()
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    A = sio.loadmat(ROOT / "reference/gmlamp/CE_SV/CSmatrix256128.mat")["A"].astype(np.float64)
    panels = list(dict.fromkeys(c[0] for c in a.case))
    fig, axs = plt.subplots(2, len(panels), figsize=(5.8 * len(panels), 8.6), squeeze=False)
    res = {}
    for j, pan in enumerate(panels):
        cases = [c for c in a.case if c[0] == pan]
        res[pan] = {}
        curves = []
        for i, (_, name, pat) in enumerate(cases):
            files = sorted(glob.glob(pat)) or [pat]
            g = sio.loadmat(files[0], variable_names=["params"])["params"]
            geo = tuple(int(np.squeeze(g[k][0, 0])) for k in ("num_ant_x", "num_ant_y", "num_ant_z"))
            arr = "UPA" if geo == (1, 16, 16) else "ULA"
            raw, params = load_3gpp(files, arr)
            H = raw["H"][:, :a.n]
            los = raw["los"][:a.n, 0] > 0.5
            x = to_beamspace(H, arr)
            curve, k95 = energy_curve(x)
            K = x.shape[1]
            n_cov = min(6000, K - 2000)
            lm = float(lmmse_nmse(x[:, :n_cov], x[:, K - 2000:], A, [10.0])[0]) if n_cov >= 1000 else None
            r = dict(implementation=str(params.get("implementation", params.get("generator", "?"))), n=int(K),
                     axes=params.get("array_axes"), los=float(los.mean()), beams95_median=float(np.median(k95)),
                     beams95_p10=float(np.percentile(k95, 10)), beams95_p90=float(np.percentile(k95, 90)),
                     beams95_los=float(np.median(k95[los])), beams95_nlos=float(np.median(k95[~los])),
                     top1=float(curve[0]), top16=float(curve[15]), lmmse_10dB=lm, lmmse_n_cov=int(n_cov))
            res[pan][name] = r
            curves.append(curve)
            axs[0, j].semilogx(np.arange(1, 257), curve, LS[i % 4], color=COL[i % 4], lw=2,
                               label=f"{name}: 95 % in {r['beams95_median']:.0f} beams ({r['beams95_p10']:.0f}-"
                                     f"{r['beams95_p90']:.0f}); LOS / NLOS {r['beams95_los']:.0f} / {r['beams95_nlos']:.0f}")
            axs[1, j].plot(np.mean(np.abs(x) ** 2, 1), LS[i % 4], color=COL[i % 4], lw=1.1,
                           label=f"{name}: LOS {r['los']:.3f}, top-1 {r['top1']:.2f}"
                                 + (f", LMMSE 10 dB {lm:.2f} dB" if lm is not None else ""))
        axs[0, j].axhline(0.95, color="#999", lw=0.8)
        axs[0, j].set_title(f"{pan}: energy in the k strongest beams", fontsize=9)
        axs[0, j].set_xlabel("k strongest beams"); axs[0, j].set_ylabel("mean fraction of the energy")
        axs[1, j].set_title(f"{pan}: mean power per beam", fontsize=9)
        axs[1, j].set_xlabel("beam index" + (" (UPA: vertical index runs fastest)" if pan.startswith("UPA") else ""))
        axs[1, j].set_ylabel("E|x_b|^2 (sum 256)")
        for ax in axs[:, j]:
            ax.legend(frameon=False, fontsize=7); ax.grid(alpha=0.3)
            for s_ in ("top", "right"):
                ax.spines[s_].set_visible(False)
    fig.suptitle(f"3GPP TR 38.901 UMi, 28 GHz, the same setting from several generators (first {a.n:,} channels "
                 "of each)", fontsize=10)
    fig.tight_layout()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    fig.savefig(out / "generators.png", dpi=140)
    (out / "generators.json").write_text(json.dumps(res, indent=1), encoding="utf-8")
    md = ["| array | generator | LOS | beams for 95 % (median, p10-p90) | LOS / NLOS | top-1 | LMMSE 10 dB |",
          "|---|---|---|---|---|---|---|"]
    for pan, rr in res.items():
        for name, r in rr.items():
            lm = "" if r["lmmse_10dB"] is None else f"{r['lmmse_10dB']:.2f} dB"
            md.append(f"| {pan} | {name} | {r['los']:.3f} | {r['beams95_median']:.0f} ({r['beams95_p10']:.0f}-"
                      f"{r['beams95_p90']:.0f}) | {r['beams95_los']:.0f} / {r['beams95_nlos']:.0f} | {r['top1']:.3f} | {lm} |")
    (out / "generators.md").write_text("\n".join(md) + "\n", encoding="utf-8")
    print("\n".join(md))


if __name__ == "__main__":
    main()

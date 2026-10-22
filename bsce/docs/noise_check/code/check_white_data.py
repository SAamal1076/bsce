"""Check of the two data sets (colored and white measurement noise) made by bsce v23 for one scenario.

Makes the white measurement files of the scenario's bands with make_sv_measurements.py --noise-model white (from the
colored files of the same draw, e.g. the authors' SV_ULA files), then checks, per band and per file:
  1. paired: y_white - A x = T (y_colored - A x), T = sqrt(c) (AA^T)^(-1/2)  (max abs difference)
  2. the same SNR: per chunk of the file (the authors' files: 10 chunks of equal SNR), the SNR measured from the data,
     10 log10(sum |Ax|^2 / sum |y - Ax|^2), colored against white
  3. the noise colour: eigenvalues of the empirical noise covariance (one chunk, divided by its noise variance s2):
     colored -> the eigenvalues of AA^T (spread ~ 0.2 .. 5.6); white -> all equal to c = tr(AA^T)/M
Writes <out>/white_data_check.json and white_data_check.png (values printed).
usage: python check_white_data.py <bsce> <scenario yaml> <out>
"""
import json
import subprocess
import sys
from pathlib import Path

import matplotlib
import numpy as np
import scipy.io as sio

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

root, yml, out = Path(sys.argv[1]), sys.argv[2], Path(sys.argv[3])
out.mkdir(parents=True, exist_ok=True)
sys.path.insert(0, str(root))
from bsce.config import band_file_names, load_yaml, scenario_cs_matrix, scenario_data_dir  # noqa: E402
from bsce.data import noise  # noqa: E402

subprocess.run([sys.executable, "scripts/make_sv_measurements.py", "--scenario", yml, "--noise-model", "white"],
               cwd=root, check=True)
sc = load_yaml(root / yml)
data = scenario_data_dir(sc)
A = sio.loadmat(scenario_cs_matrix(sc))["A"].astype(np.float64)
M, N = A.shape
T = noise.whitening_map(A)
c = noise.white_scale(A)
lamA = np.sort(np.linalg.eigvalsh(A @ A.T))[::-1]
pre = f"{sc['dataset']}_{sc['array']}"
res = {"scenario": sc["name"], "c": c, "eig_AAt": [float(lamA.min()), float(lamA.max())], "files": {}}
eig = {}
for band in sc["bands"]:
    for kind, fc, fw in zip(("training", "validation"), band_file_names(sc, band),
                            band_file_names(dict(sc, noise_model="white"), band)):
        x = sio.loadmat(data / f"{pre}_{kind}_channel.mat")["x"]
        yc, yw = sio.loadmat(data / fc)["y"], sio.loadmat(data / fw)["y"]
        sx = A @ x
        dc, dw = yc - sx, yw - sx
        pair = float(np.max(np.abs(dw - T @ dc)))
        n_ch = int(band.get("n_snr_chunks", sc.get("n_snr_chunks", 10))) if kind == "training" else len(band.get("val_snr", [1]))
        K = x.shape[1] // n_ch
        snr_c = [float(10 * np.log10(np.sum(np.abs(sx[:, k * K:(k + 1) * K]) ** 2) / np.sum(np.abs(dc[:, k * K:(k + 1) * K]) ** 2)))
                 for k in range(n_ch)]
        snr_w = [float(10 * np.log10(np.sum(np.abs(sx[:, k * K:(k + 1) * K]) ** 2) / np.sum(np.abs(dw[:, k * K:(k + 1) * K]) ** 2)))
                 for k in range(n_ch)]
        res["files"][fw] = dict(colored_twin=fc, max_abs_pairing_error=pair, chunks=n_ch, snr_colored_db=snr_c,
                                snr_white_db=snr_w, max_abs_snr_difference_db=float(np.max(np.abs(np.subtract(snr_c, snr_w)))))
        print(fw, f"paired to {pair:.2e}", "SNR (colored / white):",
              " ".join(f"{a:.2f}/{b:.2f}" for a, b in zip(snr_c, snr_w)))
        if kind == "training" and band is sc["bands"][0]:
            d0c, d0w = dc[:, :K], dw[:, :K]
            s2 = 10 ** (-snr_c[0] / 10) * np.mean(np.sum(np.abs(sx[:, :K]) ** 2, 0)) / np.trace(A @ A.T)  # per-entry s2
            eig["colored"] = np.sort(np.linalg.eigvalsh((d0c @ d0c.conj().T).real / K))[::-1] / s2
            eig["white"] = np.sort(np.linalg.eigvalsh((d0w @ d0w.conj().T).real / K))[::-1] / s2
            eig["band"], eig["snr"] = band["name"], snr_c[0]
res["noise_eigen_spread"] = {k: [float(eig[k].min()), float(eig[k].max()), float(eig[k].max() / eig[k].min())]
                             for k in ("colored", "white")}
(out / "white_data_check.json").write_text(json.dumps(res, indent=1), encoding="utf-8")

fig, ax = plt.subplots(1, 2, figsize=(15, 5.2))
ax[0].plot(eig["colored"], color="#eb6834", lw=2, label="colored files: y = A(x + n)")
ax[0].plot(eig["white"], color="#2a78d6", lw=2, label="white files: y = Ax + w")
ax[0].plot(lamA, color="#7a7a7a", lw=1, ls="--", label="eigenvalues of AAᵀ (theory, colored)")
ax[0].axhline(c, color="#2a78d6", lw=1, ls=":", label=f"c = tr(AAᵀ)/M = {c:.2f} (theory, white)")
for k, col in (("colored", "#eb6834"), ("white", "#2a78d6")):
    e = eig[k]
    ax[0].annotate(f"max {e.max():.2f}, min {e.min():.2f} (ratio {e.max() / e.min():.1f})", (M * 0.45, e[int(M * 0.45)]),
                   textcoords="offset points", xytext=(0, 10 if k == "white" else -16), fontsize=8, color="#2b2b2b")
ax[0].set_title(f"{sc['name']}: eigenvalues of the noise covariance / σ², one chunk of the {eig['band']} training file "
                f"({eig['snr']:.2f} dB)", fontsize=9)
ax[0].set_xlabel("index (sorted)"); ax[0].set_ylabel("eigenvalue / σ²"); ax[0].grid(alpha=.25)
ax[0].legend(frameon=False, fontsize=8)
f0 = next(k for k in res["files"] if "training" in k)
d = res["files"][f0]
xs = np.arange(len(d["snr_colored_db"]))
ax[1].bar(xs - 0.2, d["snr_colored_db"], 0.38, color="#eb6834", label="colored")
ax[1].bar(xs + 0.2, d["snr_white_db"], 0.38, color="#2a78d6", label="white")
for i, (a_, b_) in enumerate(zip(d["snr_colored_db"], d["snr_white_db"])):
    ax[1].text(i - 0.2, a_ + 0.1, f"{a_:.2f}", ha="center", fontsize=6.5, rotation=90, va="bottom")
    ax[1].text(i + 0.2, b_ + 0.1, f"{b_:.2f}", ha="center", fontsize=6.5, rotation=90, va="bottom")
ax[1].set_title(f"SNR measured in every chunk of {f0} and its colored twin", fontsize=9)
ax[1].set_xlabel("chunk"); ax[1].set_ylabel("SNR (dB) = 10 log10(Σ|Ax|² / Σ|y − Ax|²)"); ax[1].legend(frameon=False, fontsize=8)
ax[1].set_ylim(0, max(d["snr_colored_db"] + d["snr_white_db"]) + 4)
for a_ in ax:
    for sp in ("top", "right"):
        a_.spines[sp].set_visible(False)
fig.tight_layout(); fig.savefig(out / "white_data_check.png", dpi=150)
print("written", out)

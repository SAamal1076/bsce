"""Independent check of the measurement files against the authors' generate_measurement_data.py:
y = A (x + n), n ~ CN(0, sigma^2 I_N), sigma^2 = 10^(-SNR/10); training file = 10 equal chunks, one SNR ~ U(band)
each; validation file = one chunk at 10 dB; A = the authors' matrix of the data set (SV: CSmatrix256128,
DeepMIMO: CSmatrix_128256). The noise power of every chunk is estimated from the residual y - A x:
E||A n||^2 = sigma^2 trace(A A^T) = sigma^2 N (entries +-1/sqrt(M))."""
import json, sys
import numpy as np, scipy.io as sio
from pathlib import Path
O, D, out = Path(sys.argv[1]), Path(sys.argv[2]), Path(sys.argv[3])
A_of = {"SV": sio.loadmat(O / "CE for SV Channel Model/CSmatrix256128.mat")["A"].astype(float),
        "DeepMIMO": sio.loadmat(O / "CE for DeepMIMO Dataset/CSmatrix_128256.mat")["A"].astype(float)}
res = {}
for sc in ["SV_ULA", "SV_UPA", "DeepMIMO_ULA", "DeepMIMO_UPA"]:
    ch = sc.split("_")[0]
    A, Aw = A_of[ch], A_of["DeepMIMO" if ch == "SV" else "SV"]
    N = A.shape[1]
    res[sc] = {}
    for part in ["training", "validation"]:
        x = sio.loadmat(D / sc / f"{sc}_{part}_channel.mat")["x"]
        for band, (lo, hi) in (("0to10dB", (0, 10)), ("10to20dB", (10, 20))):
            f = sio.loadmat(D / sc / f"{sc}_{part}_128_measurements_{band}.mat")
            y, stored = f["y"], np.asarray(f["snr_db"], float).ravel()
            K = x.shape[1]
            nch = 10 if part == "training" else 1
            kc = K // nch
            est, sto, wrong = [], [], []
            for k in range(nch):
                s = slice(k * kc, (k + 1) * kc)
                r = y[:, s] - A @ x[:, s]
                est.append(-10 * np.log10(np.sum(np.abs(r) ** 2) / (N * kc)))
                sto.append(float(np.unique(np.round(stored[s] if stored.size == K else stored[k:k + 1], 9))[0]))
                rw = y[:, s] - Aw @ x[:, s]
                wrong.append(-10 * np.log10(np.sum(np.abs(rw) ** 2) / (N * kc)))
            est, sto = np.array(est), np.array(sto)
            ok_band = bool(np.all((sto >= lo - 1e-9) & (sto <= hi + 1e-9))) if part == "training" else bool(np.allclose(sto, 10))
            res[sc][f"{part} {band}"] = dict(columns=K, chunks=nch, stored_snr=np.round(sto, 3).tolist(),
                                            estimated_snr=np.round(est, 3).tolist(), max_abs_diff_db=round(float(np.max(np.abs(est - sto))), 4),
                                            snr_in_band=ok_band, residual_snr_with_the_other_A=round(float(np.mean(wrong)), 2))
            print(f"{sc:13s} {part:10s} {band:9s} K={K:6d} chunks={nch:2d} stored SNR {np.round(sto, 2)} | estimated - stored max "
                  f"{np.max(np.abs(est - sto)):.3f} dB | in band {ok_band} | with the other A: {np.mean(wrong):+.1f} dB")
    xv = sio.loadmat(D / sc / f"{sc}_validation_channel.mat")["x"]
    y1 = sio.loadmat(D / sc / f"{sc}_validation_128_measurements_0to10dB.mat")["y"]
    y2 = sio.loadmat(D / sc / f"{sc}_validation_128_measurements_10to20dB.mat")["y"]
    res[sc]["validation files of the two bands identical"] = bool(np.array_equal(y1, y2))
    print(f"{sc:13s} the two validation files (both 10 dB) identical: {np.array_equal(y1, y2)}")
out.mkdir(parents=True, exist_ok=True)
(out / "measurement_check.json").write_text(json.dumps(res, indent=1))

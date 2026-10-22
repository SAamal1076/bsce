"""How much do the 10 random chunk SNRs of the authors' measurement recipe move the first training stage?
First stage = the linear estimator B y; its least-squares optimum on the training set is evaluated on the validation
file (10 dB), for our training file and for 12 other draws of the 10 chunk SNRs (same channels)."""
import json, sys
import numpy as np, scipy.io as sio
O, D, sc = sys.argv[1], sys.argv[2], sys.argv[3]
A = sio.loadmat(O + ("/CE for DeepMIMO Dataset/CSmatrix_128256.mat" if sc.startswith("DeepMIMO") else "/CE for SV Channel Model/CSmatrix256128.mat"))["A"].astype(float)
xt = sio.loadmat(f"{D}/{sc}/{sc}_training_channel.mat")["x"]; xv = sio.loadmat(f"{D}/{sc}/{sc}_validation_channel.mat")["x"]
yv = sio.loadmat(f"{D}/{sc}/{sc}_validation_128_measurements_0to10dB.mat")["y"]
def ls_B(y, x):
    Yr, Xr = np.concatenate([y.real, y.imag], 1), np.concatenate([x.real, x.imag], 1)
    return np.linalg.solve(Yr @ Yr.T, Yr @ Xr.T).T
def nm(B):
    return 10 * np.log10(np.mean(np.sum(np.abs(B @ yv - xv) ** 2, 0)) / np.mean(np.sum(np.abs(xv) ** 2, 0)))
f = sio.loadmat(f"{D}/{sc}/{sc}_training_128_measurements_0to10dB.mat")
ours = nm(ls_B(f["y"], xt))
r = np.random.default_rng(0); K = xt.shape[1]; kc = K // 10; vals, means = [], []
for d in range(12):
    snr = r.uniform(0, 10, 10); y = np.zeros((A.shape[0], K), complex)
    for k in range(10):
        s = slice(k * kc, (k + 1) * kc)
        y[:, s] = A @ (xt[:, s] + np.sqrt(10 ** (-snr[k] / 10)) * (r.standard_normal((A.shape[1], kc)) + 1j * r.standard_normal((A.shape[1], kc))) / np.sqrt(2))
    vals.append(nm(ls_B(y, xt))); means.append(snr.mean())
res = dict(scenario=sc, ours_db=round(ours, 3), ours_mean_chunk_snr=round(float(np.mean(np.unique(np.round(np.asarray(f["snr_db"]).ravel(), 6)))), 2),
           other_draws_db=np.round(vals, 3).tolist(), other_mean_snr=np.round(means, 2).tolist(), sd_db=round(float(np.std(vals)), 3),
           corr_with_mean_snr=round(float(np.corrcoef(vals, means)[0, 1]), 2))
print(json.dumps(res))

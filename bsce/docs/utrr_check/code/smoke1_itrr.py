"""Smoke test 1: ITRR (Algorithm 1) on the paper's Sec. III-C experiment, over many samples.
Sparse samples: SV ULA beamspace channels of bsce (data/SV_ULA test set) with all but their 16 largest-magnitude
entries set to zero; real part; noiseless y = Phi x with GM-LAMP's CS matrix (M = 128, +-1/sqrt(M)).
ITRR with the fixed step 1/(k + 2 rho) and with BB steps (K = 16; rho = 0.1 and 1), against PGD-BB for Lasso and
ridge (the paper's Figs. 2-4); and ITRR with the printed sign of grad F.
usage: python smoke1_itrr.py <bsce folder> <out folder> [n_samples]"""
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import scipy.io as sio

sys.path.insert(0, str(Path(__file__).parent))
from utrr_ref import itrr  # noqa: E402

root, out = Path(sys.argv[1]), Path(sys.argv[2])
ns = int(sys.argv[3]) if len(sys.argv) > 3 else 40
out.mkdir(parents=True, exist_ok=True)
Phi = sio.loadmat(root / "reference/gmlamp/CE_SV/CSmatrix256128.mat")["A"].astype(float)
H = sio.loadmat(root / "data/SV_ULA/SV_ULA_test_channel.mat")["x"][:, :ns]
X = np.zeros((H.shape[0], ns))
for j in range(ns):
    keep = np.argsort(-np.abs(H[:, j]))[:16]
    X[keep, j] = H[keep, j].real
Y = Phi @ X
N = Phi.shape[1]
n_it = 4000


def pgd_bb(Y, reg, lam, n_iter):
    """Proximal gradient with BB steps for Lasso (reg='l1') and ridge (reg='l2'), the paper's PGD-BB, per column."""
    xk = Phi.T @ Y
    xp = gp = None
    L = np.linalg.eigvalsh(Phi.T @ Phi).max()
    errs = []
    for t in range(n_iter):
        g = Phi.T @ (Phi @ xk - Y) + (2 * lam * xk if reg == "l2" else 0)
        if xp is None:
            a = np.full(Y.shape[1], 1 / L)
        else:
            dz, dg = xk - xp, g - gp
            den = np.sum(dz * dg, 0)
            a = np.where(den > 0, np.sum(dz * dz, 0) / np.where(den > 0, den, 1), 1 / L)
        w = xk - a * g
        if reg == "l1":
            w = np.sign(w) * np.maximum(np.abs(w) - a * lam, 0)
        xp, gp = xk, g
        xk = xk + (t / (t + 3) if t else 1) * (w - xk)
        errs.append(np.sum((xk - X) ** 2, 0) / np.sum(X ** 2, 0))
    return np.array(errs)


runs = {}
for step in ("fixed", "bb"):
    for rho in (0.1, 1.0):
        _, h = itrr(Y, Phi, K=16, rho=rho, n_iter=n_it, step=step, x_true=X)
        runs[f"ITRR{'-BB' if step == 'bb' else ''}, rho={rho:g}"] = np.array(h["err"])
runs["PGD-BB Lasso (lambda 1e-3)"] = pgd_bb(Y, "l1", 1e-3, n_it)
runs["PGD-BB ridge (lambda 1e-3)"] = pgd_bb(Y, "l2", 1e-3, n_it)
_, h = itrr(Y, Phi, K=16, rho=1.0, n_iter=60, step="fixed", sign="paper", x_true=X)
runs["ITRR, printed sign of grad F"] = np.array(h["err"])

summ = {}
for k, e in runs.items():
    f = e[-1]
    summ[k] = dict(iterations=len(e), median_final=float(np.median(f)), best=float(f.min()), worst=float(f.max()),
                   exact_below_1e20=int(np.sum(f < 1e-20)), n=int(f.size))
(out / "smoke1_itrr.json").write_text(json.dumps(summ, indent=1))

cols = {"ITRR, rho=0.1": "#9ec1ee", "ITRR, rho=1": "#2a78d6", "ITRR-BB, rho=0.1": "#93d4b9", "ITRR-BB, rho=1": "#1f9e6e",
        "PGD-BB Lasso (lambda 1e-3)": "#eb6834", "PGD-BB ridge (lambda 1e-3)": "#c23b6a",
        "ITRR, printed sign of grad F": "#7a7a7a"}
j = int(np.argmax(runs["ITRR, rho=1"][-1] < 1e-20)) if np.any(runs["ITRR, rho=1"][-1] < 1e-20) else 0
fig, ax = plt.subplots(1, 3, figsize=(20, 4.8))
for k, e in runs.items():
    v = np.maximum(e[:, j], 1e-35)
    ax[0].semilogy(np.arange(1, len(v) + 1), v, color=cols[k], lw=1.5, label=f"{k}: {v[-1]:.1e}")
ax[0].set_title(f"sample {j} (the first one ITRR rho=1 recovers): error per iteration", fontsize=9)
ax[0].set_xlabel("iteration"); ax[0].set_ylabel("||x^ - x||^2 / ||x||^2"); ax[0].set_ylim(1e-35, 1e3)
for k, e in runs.items():
    if "printed" in k:
        continue
    f = np.sort(np.maximum(e[-1], 1e-35))
    ax[1].semilogx(f, np.arange(1, f.size + 1) / f.size, color=cols[k], lw=1.6,
                   label=f"{k}: median {np.median(e[-1]):.1e}, exact {np.sum(e[-1] < 1e-20)}/{f.size}")
ax[1].set_title(f"final error over {ns} samples ({n_it} iterations; exact = below 1e-20)", fontsize=9)
ax[1].set_xlabel("||x^ - x||^2 / ||x||^2"); ax[1].set_ylabel("CDF over samples")
ax[2].stem(np.arange(N), X[:, j], linefmt="#9aa5ad", markerfmt="o", basefmt=" ", label="sparse sample (real part)")
xb, _ = itrr(Y[:, j], Phi, K=16, rho=1.0, n_iter=n_it, step="fixed")
ax[2].plot(np.arange(N), xb[:, 0], "x", color="#2a78d6", ms=6, label=f"ITRR rho=1: error {runs['ITRR, rho=1'][-1, j]:.1e}")
ax[2].set_title(f"sample {j}: reconstruction (as the paper's Fig. 2)", fontsize=9); ax[2].set_xlabel("beam index")
for a in ax:
    a.grid(alpha=.3); a.legend(frameon=False, fontsize=7.2)
fig.tight_layout(); fig.savefig(out / "smoke1_itrr.png", dpi=150)
print(json.dumps(summ, indent=1))

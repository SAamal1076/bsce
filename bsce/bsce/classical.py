"""Classical baselines - numpy ports of the MATLAB files used in the LADMM project.

AMP.m (complex AMP, soft threshold alf=1.1402, T=10), OMP.m (per-column, S iterations),
SD.m (support detection of Gao et al. 2017, per column), ADMM_vectorized.m (ADMM-LASSO).
Verified against the .m files in Octave (tests/test_classical_vs_octave.py).
"""
from __future__ import annotations

import numpy as np

# v31: the names of the classical methods in evaluate.py outputs (one list for every script)
METHODS = ("amp", "omp", "sd", "admm", "l1", "grwl1")

EPS = np.finfo(float).eps


def amp(y, A, T: int = 10, alf: float = 1.1402):
    M, N = A.shape
    S = y.shape[1]
    Bmf = A.conj().T
    xhat = np.zeros((N, S), dtype=complex)
    v = y.astype(complex)
    for _ in range(T):
        rhat = xhat + Bmf @ v
        rvar = np.sum(np.abs(v) ** 2, 0) / M
        lam = alf * np.sqrt(rvar)
        xhat = np.exp(1j * np.angle(rhat)) * np.maximum(np.abs(rhat) - lam[None, :], 0)
        on = (xhat != 0)
        g = (1 / M) * np.sum(on * (1 - lam[None, :] / (2 * np.abs(rhat) + EPS)), 0)
        c = (1 / M) * np.sum(on * (0.5 * lam[None, :] * np.sqrt(rhat) * (1. / ((np.conj(rhat)) ** (3 / 2) + EPS))), 0)
        v = y - A @ xhat + v * g[None, :] + np.conj(v) * c[None, :]
    return xhat


def omp_column(y, A, t: int):
    M, N = A.shape
    A = A.astype(complex).copy()
    xhat = np.zeros(N, dtype=complex)
    At = np.zeros((M, t), dtype=complex)
    pos_xhat = np.zeros(t, dtype=int)
    r = y.copy()
    xls = None
    for i in range(t):
        product = A.conj().T @ r
        pos = int(np.argmax(np.abs(product)))
        At[:, i] = A[:, pos]
        pos_xhat[i] = pos
        A[:, pos] = 0
        Ai = At[:, :i + 1]
        xls = np.linalg.inv(Ai.conj().T @ Ai) @ Ai.conj().T @ y
        r = y - Ai @ xls
    xhat[pos_xhat] = xls
    return xhat


def omp(Y, A, t: int = 24):
    return np.stack([omp_column(Y[:, i], A, t) for i in range(Y.shape[1])], 1)


def sd_column(x, Phi, L: int = 3, V: int = 8):
    n = Phi.shape[1]
    x_temp = x.copy()
    select = np.zeros((L, V), dtype=int)                  # 1-based like MATLAB
    for l in range(L):
        yv = Phi.conj().T @ x_temp
        energy = np.sum(np.abs(yv[:, None]) ** 2, 1)
        order = int(np.argsort(-energy, kind="stable")[0]) + 1
        if V % 2 == 0:
            sel = np.arange(order - V // 2, order + (V - 2) // 2 + 1)
        else:
            sel = np.arange(order - (V - 1) // 2, order + (V - 1) // 2 + 1)
        sel = np.where(sel > n, sel - n, sel)
        sel = np.where(sel < 1, sel + n, sel)
        select[l] = sel
        Phi2 = Phi[:, sel - 1]
        h_hat2 = np.linalg.inv(Phi2.conj().T @ Phi2) @ Phi2.conj().T @ x_temp
        temp = np.zeros(n, dtype=complex)
        temp[sel - 1] = h_hat2
        x_temp = x_temp - Phi @ temp
    sel_final = np.unique(select.reshape(-1, order="F"))
    Pf = Phi[:, sel_final - 1]
    est = np.linalg.inv(Pf.conj().T @ Pf) @ Pf.conj().T @ x
    h = np.zeros(n, dtype=complex)
    h[sel_final - 1] = est
    return h


def sd(Y, Phi, L: int = 3, V: int = 8):
    return np.stack([sd_column(Y[:, i], Phi, L, V) for i in range(Y.shape[1])], 1)


def soft_c(V, thr):
    mag = np.abs(V)
    shr = np.maximum(mag - thr, 0)
    Z = shr * np.exp(1j * np.angle(V))
    Z[shr == 0] = 0
    return Z


def admm(Y, A, lam: float, rho: float = 1.0, max_iter: int = 500, tol: float = 1e-5):
    """ADMM_vectorized.m: returns Z and the number of iterations run."""
    M, N = A.shape
    K = Y.shape[1]
    AHA = A.conj().T @ A
    AHY = A.conj().T @ Y
    Hm = np.linalg.inv(AHA + rho * np.eye(N))
    H = np.zeros((N, K), dtype=complex)
    Z = np.zeros((N, K), dtype=complex)
    U = np.zeros((N, K), dtype=complex)
    it = 0
    for it in range(1, max_iter + 1):
        Z_old = Z
        H = Hm @ (AHY + rho * Z - U)
        V = H + U / rho
        Z = soft_c(V, lam / rho)
        U = U + rho * (H - Z)
        primal = np.mean(np.sqrt(np.sum(np.abs(H - Z) ** 2, 0)))
        dual = rho * np.mean(np.sqrt(np.sum(np.abs(Z - Z_old) ** 2, 0)))
        if primal < tol and dual < tol:
            break
    return Z, it


# -- v31: graph-reweighted l1 (the uploaded graph_reweighted_l1 package) and its fixed-iteration l1 ADMM ------------------
def pooled_amplitude(Z, array: str = "UPA", N1: int = 16, N2: int = 16):
    """sqrt of the mean of |z|^2 over the closed neighbourhood of every beam: UPA (N1 x N2 torus, index N2 a + b) the
    beam and its 4 neighbours; ULA (ring) the beam and the beams at +-1, +-2. Z: (N, K)."""
    P = np.abs(Z) ** 2
    N = Z.shape[0]
    if array.upper() == "UPA":
        Q = P.reshape(N1, N2, -1)
        S = Q + np.roll(Q, 1, 0) + np.roll(Q, -1, 0) + np.roll(Q, 1, 1) + np.roll(Q, -1, 1)
        S = S.reshape(N, -1)
    else:
        S = P + np.roll(P, 1, 0) + np.roll(P, -1, 0) + np.roll(P, 2, 0) + np.roll(P, -2, 0)
    return np.sqrt(S / 5.0)


def graph_weights(Z, eps: float, array: str = "UPA", N1: int = 16, N2: int = 16):
    """The package's weights: a = pooled amplitude / max (per channel), w = 1 / (a + eps), normalised to mean 1."""
    a = pooled_amplitude(Z, array, N1, N2)
    a = a / np.maximum(a.max(0, keepdims=True), 1e-12)
    w = 1.0 / (a + eps)
    return w / w.mean(0, keepdims=True)


def l1_admm(Y, A, lam: float, rho: float = 1.0, T: int = 300, wts=None, P=None):
    """min 1/2 ||y - A x||^2 + lam sum_i w_i |x_i| (w = 1: plain l1) with T ADMM iterations (scaled dual), as the
    package's admm_weighted_l1: x = (A^T A + rho I)^-1 (A^T y + rho (z - u)), z = soft(x + u, lam w / rho), u += x - z."""
    A = np.asarray(A, np.float64)
    n = A.shape[1]
    P = np.linalg.inv(A.T @ A + rho * np.eye(n)) if P is None else P
    Aty = A.T @ Y
    Z = np.zeros((n, Y.shape[1]), complex)
    U = np.zeros_like(Z)
    thr = lam / rho if wts is None else (lam / rho) * wts
    for _ in range(T):
        X = P @ (Aty + rho * (Z - U))
        V = X + U
        m = np.abs(V)
        Z = V * np.maximum(1 - thr / np.maximum(m, 1e-300), 0)
        U = U + X - Z
    return Z


def grwl1(Y, A, lam: float, eps: float, rounds: int = 2, rho: float = 1.0, T: int = 300, array: str = "UPA",
          N1: int = 16, N2: int = 16):
    """Graph-reweighted l1 (classical, no training): round 0 plain l1, then `rounds` rounds with graph_weights of the
    previous estimate (the package's graph_reweighted_l1; T ADMM iterations per round)."""
    A = np.asarray(A, np.float64)
    P = np.linalg.inv(A.T @ A + rho * np.eye(A.shape[1]))
    Z = l1_admm(Y, A, lam, rho, T, None, P)
    for _ in range(rounds):
        Z = l1_admm(Y, A, lam, rho, T, graph_weights(Z, eps, array, N1, N2), P)
    return Z

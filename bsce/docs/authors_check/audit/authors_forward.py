"""Independent NumPy port of the authors' network (Train Network/define_networks.build_LAMP + define_shrinkage),
complex128, used only for this audit (not bsce code)."""
import numpy as np


def soft(r, rvar, theta):
    lam = np.abs(theta[0] * np.sqrt(rvar))
    ar = np.abs(r)
    ampl = np.maximum(ar - lam, 0)
    xhat = r * (ampl / ar)
    on = (ar - lam) > 0
    dxdr = np.mean(on * (1 - lam / (2 * ar)), 0)
    dxdr1 = np.mean(on * (0.5 * lam * np.sqrt(r) * np.power(np.conj(r), -1.5)), 0)
    return xhat, dxdr, dxdr1


def gm(corr, rvar, theta, K=4):
    th = -40.0
    P = np.exp(theta[0]) / np.sum(np.exp(theta[0]))
    mu = theta[1]
    var = np.exp(np.maximum(th, theta[2]))
    U = V = dU = dV = dU1 = dV1 = 0
    for k in range(K):
        tmu = (mu[k] * rvar + corr * var[k]) / (rvar + var[k])
        pdf = np.exp(np.maximum(th, np.real(-np.conj(corr - mu[k]) * (corr - mu[k]) / 2 / (rvar + var[k])))) / 2 / np.pi / np.real(var[k] + rvar)
        u = tmu * P[k] * pdf
        du = var[k] / (rvar + var[k]) * P[k] * pdf + u * (-np.conj(corr - mu[k]) / 2 / (rvar + var[k]))
        du1 = u * (-(corr - mu[k]) / 2 / (rvar + var[k]))
        v = P[k] * pdf
        dv = v * (-np.conj(corr - mu[k]) / 2 / (rvar + var[k]))
        dv1 = v * (-(corr - mu[k]) / 2 / (rvar + var[k]))
        U, dU, dU1, V, dV, dV1 = U + u, dU + du, dU1 + du1, V + v, dV + dv, dV1 + dv1
    xhat = U / V
    dxdr = np.mean((dU * V - U * dV) / (V * np.conj(V)), 0)
    dxdr1 = np.mean((dU1 * V - U * dV1) / (V * np.conj(V)), 0)
    return xhat, dxdr, dxdr1


def lamp_forward(y, A, Bs, thetas, shrink):
    """Bs, thetas: lists (one per layer) as in the .mat / .npz files. Returns xhat after the last layer."""
    M, N = A.shape
    eta = soft if shrink == "soft" else gm
    rvar = np.sum(np.abs(y) ** 2, 0) / M
    xhat, d, d1 = eta(Bs[0] @ y, rvar, thetas[0])
    v = y
    for t in range(1, len(Bs)):
        b, c = N / M * d, N / M * d1
        v = y - A @ xhat + b * v + c * np.conj(v)
        rvar = np.sum(np.abs(v) ** 2, 0) / M
        xhat, d, d1 = eta(xhat + Bs[t] @ v, rvar, thetas[t])
    return xhat


def nmse_db(xh, x):
    return 10 * np.log10(np.mean(np.sum(np.abs(xh - x) ** 2, 0)) / np.mean(np.sum(np.abs(x) ** 2, 0)))

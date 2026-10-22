"""Reference implementation of ITRR and UTRR (Wu, Cheng, Eldar, Cioffi, "Learned Trimmed-Ridge Regression for Channel
Estimation in Millimeter-Wave Massive MIMO", IEEE TCOM 73(2):1128-1141, 2025), written from the paper only (no code
was published). Real-valued, as in the paper: the real and imaginary parts of a beamspace channel are two independent
real samples, y = Phi x + n (eq. 8-9).

ITRR (Algorithm 1, Sec. III):
    split x = u - v, u = (x)+, v = (-x)+, z = [u; v] >= 0, A = [Phi, -Phi]
    grad F(z) = A^T (A z - y) + 2 rho (z - zbar_K)          zbar_K = z with only its K largest entries kept
    fixed step : z <- ( z - grad F(z) / (k + 2 rho) )+,       k >= lambda_max(A^T A)                 (eq. 19-20)
    BB step    : w <- ( z - alpha_t grad F(z) )+, z <- z + beta_t (w - z), alpha_t of eq. 22, beta_t = t / (t + 3)  (eq. 21)
    x = u - v
  Sign: the paper prints grad F(z) = A^T (y - A z) + ...; that is the gradient of -1/2||y - Az||^2 + ..., and a step
  along it would move away from the data. Eq. 16 (DC programming) gives A^T (A z - y), which is what is used here
  (and what makes Figs. 2-4 reproducible, see smoke test 1). Option sign="paper" keeps the printed form.

UTRR (Sec. IV, Fig. 6, Algorithm 2):
    x0 = Phi^T y,  z0 = [(x0)+; (-x0)+],  [BN]
    layer t = 0..L-1:   g = A_t^T (A_t z - y) + rho_t (z - zbar_K^(t))
                        w = ReLU( [BN_t]( z - alpha_t g ) )
                        z = (w + z) / 2                                         (beta = 1/2, residual unit)
    xhat = u^(L) - v^(L)
    trainable: A_t in R^{M x 2N} (init [Phi, -Phi]), rho_t (init 1.0), alpha_t (init 0.1): L (2MN + 2) parameters
    RCC (Sec. IV-B): K = 0 in the layers 0..L-2, K only in the last layer (eq. 23)
    BN: Fig. 6 draws a batch-normalisation layer after z0 and inside every layer (before the ReLU); Algorithm 2 and
    the parameter count L(2MN + 2) leave it out. Option bn=True|False.
    loss: E ||x - xhat||^2 (eq. 24); 30 layers; K in {0, 16, 64, 256}; ensemble UTRR-E = mean of the four models.
"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn


# ---------------------------------------------------------------------------------------------------------------- ITRR
def topk_keep(z: np.ndarray, K: int) -> np.ndarray:
    """zbar_K: z with only its K largest entries kept (z >= 0, so largest = largest magnitude), per column."""
    if K <= 0:
        return np.zeros_like(z)
    out = np.zeros_like(z)
    idx = np.argpartition(-z, K - 1, axis=0)[:K]
    np.put_along_axis(out, idx, np.take_along_axis(z, idx, 0), 0)
    return out


def itrr(y, Phi, K, rho=1.0, n_iter=4000, step="bb", sign="correct", eps=0.0, x_true=None, F_track=False):
    """Algorithm 1 (step='fixed') or its BB variant (step='bb', eq. 21-22). y: (M,) or (M, B) real.
    Returns x (N, B) and a history dict (normalised squared error per iteration if x_true is given)."""
    y = np.atleast_2d(np.asarray(y, float).T).T
    M, N = Phi.shape
    A = np.hstack([Phi, -Phi])
    k = np.linalg.eigvalsh(A.T @ A).max()
    s = 1.0 if sign == "correct" else -1.0

    def grad(z):
        return s * (A.T @ (A @ z - y)) + 2 * rho * (z - topk_keep(z, K))

    def F(z):
        x = z[:N] - z[N:]
        zs = np.sort(z, 0)[::-1]
        return 0.5 * np.sum((y - Phi @ x) ** 2, 0) + rho * (np.sum(z ** 2, 0) - np.sum(zs[:K] ** 2, 0))

    if x_true is not None:
        x_true = np.asarray(x_true, float).reshape(N, -1)
    x0 = Phi.T @ y
    z = np.vstack([np.maximum(x0, 0), np.maximum(-x0, 0)])
    hist = {"err": [], "F": []}
    zp, gp = None, None
    for t in range(n_iter):
        g = grad(z)
        if step == "fixed":
            zn = np.maximum(z - g / (k + 2 * rho), 0)
        else:
            if zp is None:
                a = 1.0 / (k + 2 * rho)
            else:
                dz, dg = z - zp, g - gp
                den = np.sum(dz * dg, 0)
                a = np.where(den > 0, np.sum(dz * dz, 0) / np.where(den > 0, den, 1.0), 1.0 / (k + 2 * rho))
            w = np.maximum(z - a * g, 0)
            beta = t / (t + 3.0) if t > 0 else 1.0
            zn = z + beta * (w - z)
            zp, gp = z, g
        if x_true is not None:
            x = zn[:N] - zn[N:]
            hist["err"].append(np.sum((x - x_true) ** 2, 0) / np.sum(x_true ** 2, 0))     # per column
        if F_track:
            hist["F"].append(F(zn))
        if eps and np.linalg.norm(zn - z) <= eps:
            z = zn
            break
        z = zn
    return z[:N] - z[N:], hist


# ---------------------------------------------------------------------------------------------------------------- UTRR
class BN(nn.Module):
    """Batch normalisation over the batch (columns) for each of the 2N features (rows); batch statistics in training
    with gradients enabled, running statistics otherwise (validation, test)."""

    def __init__(self, n, momentum=0.1, eps=1e-5, dtype=torch.float32):
        super().__init__()
        self.gamma = nn.Parameter(torch.ones(n, 1, dtype=dtype))
        self.beta = nn.Parameter(torch.zeros(n, 1, dtype=dtype))
        self.register_buffer("mean", torch.zeros(n, 1, dtype=dtype))
        self.register_buffer("var", torch.ones(n, 1, dtype=dtype))
        self.momentum, self.eps = momentum, eps

    def forward(self, z):
        if self.training and torch.is_grad_enabled():
            m = z.mean(1, keepdim=True)
            v = z.var(1, unbiased=False, keepdim=True)
            with torch.no_grad():
                n = z.shape[1]
                self.mean.mul_(1 - self.momentum).add_(self.momentum * m)
                self.var.mul_(1 - self.momentum).add_(self.momentum * v * n / max(n - 1, 1))
        else:
            m, v = self.mean, self.var
        return self.gamma * (z - m) / torch.sqrt(v + self.eps) + self.beta


def topk_keep_t(z: torch.Tensor, K: int) -> torch.Tensor:
    if K <= 0:
        return torch.zeros_like(z)
    v, i = torch.topk(z, K, dim=0)
    return torch.zeros_like(z).scatter(0, i, v)


class UTRRNet(nn.Module):
    def __init__(self, Phi, L=30, K=16, rcc=True, bn=True, rho0=1.0, alpha0=0.1, sign="correct",
                 dtype=torch.float32):
        super().__init__()
        Phi = np.asarray(Phi, float)
        self.M, self.N = Phi.shape
        self.L, self.K, self.rcc, self.bn, self.sign = L, K, rcc, bn, sign
        self.register_buffer("Phi", torch.tensor(Phi, dtype=dtype))
        A0 = torch.tensor(np.hstack([Phi, -Phi]), dtype=dtype)
        self.A = nn.ParameterList([nn.Parameter(A0.clone()) for _ in range(L)])
        self.rho = nn.ParameterList([nn.Parameter(torch.tensor([rho0], dtype=dtype)) for _ in range(L)])
        self.alpha = nn.ParameterList([nn.Parameter(torch.tensor([alpha0], dtype=dtype)) for _ in range(L)])
        if bn:
            self.bn0 = BN(2 * self.N, dtype=dtype)
            self.bns = nn.ModuleList([BN(2 * self.N, dtype=dtype) for _ in range(L)])

    def k_of(self, t, last):
        return self.K if (not self.rcc or t == last) else 0

    def forward(self, y, upto=None):
        """y: (M, B) real -> xhat (N, B). upto: index of the last layer used (default L-1); the top-K term sits in
        that layer (RCC), so the output equals a network of upto+1 layers."""
        last = self.L - 1 if upto is None else upto
        x0 = self.Phi.T @ y
        z = torch.cat([torch.relu(x0), torch.relu(-x0)], 0)
        if self.bn:
            z = self.bn0(z)
        s = 1.0 if self.sign == "correct" else -1.0
        for t in range(last + 1):
            A = self.A[t]
            g = s * (A.T @ (A @ z - y)) + self.rho[t] * (z - topk_keep_t(z, self.k_of(t, last)))
            q = z - self.alpha[t] * g
            if self.bn:
                q = self.bns[t](q)
            z = (torch.relu(q) + z) / 2
        return z[: self.N] - z[self.N:]

    def forward_complex(self, y):
        """bsce convention: complex y (M, B) -> complex xhat (N, B); real and imaginary parts as 2B real samples."""
        B = y.shape[1]
        xr = self.forward(torch.cat([y.real, y.imag], 1))
        return torch.complex(xr[:, :B], xr[:, B:])

    def n_params(self):
        return sum(p.numel() for p in self.parameters())

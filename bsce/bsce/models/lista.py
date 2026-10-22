"""LISTA and ALISTA (Chen, Liu, Wang, Yin) - ports of the ALISTA repository (TF 1.12).

LISTA  (models/LISTA.py):  x_{t} = soft(W_t x_{t-1} + B y, theta_t),  B = A^T/s, W = I - B A,
                           s = 1.001 ||A||_2^2, theta init lam/s; tied (one W) or untied (W_t).
ALISTA (models/ALISTA.py): z = x + alpha_t W^T (y - A x);  x = shrink_ss(z, theta_t, p_t),
                           W analytic (CalculateW.m), p_t = min((t+1) p, maxp) % support selection.

``complex_mode=True`` (beamspace use): real weights act on Re/Im (as in GM-LAMP) and the
shrinkage acts on |.| with the phase kept; it equals the real version for real inputs.
``complex_mode=False`` reproduces the original real-valued code (used by the TF tests).
"""
from __future__ import annotations

from typing import List, Optional

import numpy as np
import torch

from .common import Stage, UnrolledNet, rmm
from .shrinkage import shrink_ss_complex, shrink_ss_real, soft_complex, soft_real


def alista_W(A: np.ndarray, eta: Optional[float] = None, maxit: int = 1000, tol: float = 1e-12, verbose=False):
    """Port of matlabs/CalculateW.m: min ||W^T A||_F^2 (off-diag) s.t. diag(W^T A) = 1.
    v31: the default step is min(0.1, 0.6 / ||A||_2^2): 0.1, the original, for every scenario combiner (||A||_2^2 < 6);
    the fixed 0.1 diverged (NaN W) on the correlated test combiner (||A||_2^2 = 44.7). The projection is the original
    one (exact for unit-norm columns). An exact projection for other column norms was tried and not kept: with it, the
    trained ALISTA was worse on the Gaussian (-8.97 against -9.53 dB) and partially connected (+17.3 against -6.1 dB)
    test combiners, since its learned steps go with the original W."""
    D = np.asarray(A, np.float64)
    m, n = D.shape
    if eta is None:
        eta = min(0.1, 0.6 / float(np.linalg.norm(D, 2) ** 2))

    def func(W):
        res = D.T @ W - np.eye(n)
        Q = np.ones((n, n)) + np.eye(n) * (-1)
        res = res * np.sqrt(Q)
        return np.sum(np.sum(res * res))

    def proj(W):
        aw = np.diag(D.T @ W)
        aw = np.tile(aw[None, :], (D.shape[0], 1))
        return W + (1 - aw) * D

    W = D.copy()
    f = func(D)
    for t in range(1, maxit + 1):
        res = D.T @ W - np.eye(n)
        gra = D @ res
        W_next = W - eta * gra
        W_next = proj(W_next)
        f_next = func(W_next)
        if abs(f - f_next) / f < tol:
            break
        W = W_next
        f = f_next
        if verbose and t % 50 == 0:
            print(f"t: {t}\t, func: {f:f}")
    return W


class _ISTAFamily(UnrolledNet):
    def __init__(self, A, complex_mode, precision, device):
        super().__init__(A, precision)
        self.complex_mode = complex_mode
        self.scale = 1.001 * np.linalg.norm(self.A_np, ord=2) ** 2

    def _in(self, y):
        if self.complex_mode:
            return self.as_complex(y)
        y = y if torch.is_tensor(y) else torch.as_tensor(np.asarray(y))
        return y.to(device=self.A.device, dtype=self.rdtype)

    def _mm(self, W, z):
        return rmm(W, z) if self.complex_mode else W @ z


class LISTA(_ISTAFamily):
    family = "lista"

    def __init__(self, A, T: int = 16, lam: float = 0.4, untied: bool = True, coord: bool = False,
                 complex_mode: bool = True, scope: str = "LISTA", precision: str = "float64", device=None):
        super().__init__(A, complex_mode, precision, device)
        self.T, self.untied, self.coord, self.scope = T, untied, coord, scope
        self.cfg = dict(model="LISTA", T=T, lam=lam, untied=untied, coord=coord, complex_mode=complex_mode,
                        precision=precision)
        A_ = self.A_np
        N = self.N
        theta = lam / self.scale
        if coord:
            theta = np.ones((N, 1)) * theta
        B = A_.T / self.scale
        W = np.eye(N) - B @ A_
        self._p("B", B)
        if not untied:
            self._p("W", W)
        for t in range(T):
            self._p(f"theta_{t + 1}", theta)
            if untied:
                self._p(f"W_{t + 1}", W)
        if device is not None:
            self.to(device)

    def _W(self, t):
        return self.P(f"W_{t + 1}") if self.untied else self.P("W")

    def stage_plan(self) -> List[Stage]:
        out, allv = [], []
        for t in range(self.T):
            new = ["B", f"W_{t + 1}" if self.untied else "W", f"theta_{t + 1}"]
            allv += [n for n in new if n not in allv]
            out.append(Stage(f"{self.scope} T={t + 1}", "layer", new, list(allv), t))
        return out

    def forward_stages(self, y, upto: Optional[int] = None):
        y = self._in(y)
        n_st = self.T if upto is None else upto + 1
        By = self._mm(self.P("B"), y)
        xh = torch.zeros((self.N, y.shape[-1]), dtype=y.dtype, device=y.device)
        shrink = soft_complex if self.complex_mode else soft_real
        outs = []
        for t in range(n_st):
            xh = shrink(self._mm(self._W(t), xh) + By, self.P(f"theta_{t + 1}"))
            outs.append(xh)
        return outs


class ALISTA(_ISTAFamily):
    family = "alista"

    def __init__(self, A, T: int = 16, lam: float = 0.4, W=None, percent: float = 1.2,
                 max_percent: float = 13.0, coord: bool = False, complex_mode: bool = True,
                 scope: str = "ALISTA", precision: str = "float64", device=None):
        super().__init__(A, complex_mode, precision, device)
        self.T, self.scope, self.p, self.maxp = T, scope, percent, max_percent
        if W is None:
            W = alista_W(self.A_np)
        W = np.asarray(W, np.float64)
        self.cfg = dict(model="ALISTA", T=T, lam=lam, percent=percent, max_percent=max_percent,
                        coord=coord, complex_mode=complex_mode, precision=precision)
        self.register_buffer("W", torch.tensor(W, dtype=self.rdtype))
        theta = lam / self.scale
        if coord:
            theta = np.ones((self.N, 1)) * theta
        self.ps = np.clip([(t + 1) * percent for t in range(T)], 0.0, max_percent)
        for t in range(T):
            self._p(f"alpha_{t + 1}", 1.0)
            self._p(f"theta_{t + 1}", theta)
        if device is not None:
            self.to(device)

    def set_combiner(self, A):
        """v31: use another combiner A (M' x N, any M'): A and the analytic weight W = alista_W(A) are recomputed (the
        learned steps and thresholds are kept; they were learned with the W of the training combiner)."""
        A = self._combiner_array(A)
        self.M, self.A_np = A.shape[0], A
        self._put_buffer("A", A)
        self._put_buffer("W", alista_W(A))
        self.scale = 1.001 * np.linalg.norm(A, ord=2) ** 2

    def stage_plan(self) -> List[Stage]:
        out, allv = [], []
        for t in range(self.T):
            new = [f"alpha_{t + 1}", f"theta_{t + 1}"]
            allv += new
            out.append(Stage(f"{self.scope} T={t + 1}", "layer", new, list(allv), t))
        return out

    def forward_stages(self, y, upto: Optional[int] = None):
        y = self._in(y)
        n_st = self.T if upto is None else upto + 1
        xh = torch.zeros((self.N, y.shape[-1]), dtype=y.dtype, device=y.device)
        shrink = shrink_ss_complex if self.complex_mode else shrink_ss_real
        Wt = self.W.T
        outs = []
        for t in range(n_st):
            res = y - self._mm(self.A, xh)
            zh = xh + self.P(f"alpha_{t + 1}") * self._mm(Wt, res)
            xh = shrink(zh, self.P(f"theta_{t + 1}"), float(self.ps[t]))
            outs.append(xh)
        return outs

    def mat_dict(self):
        d = super().mat_dict()
        d["W"] = self.W.cpu().numpy()
        d["ps"] = np.asarray(self.ps, np.float64)
        return d

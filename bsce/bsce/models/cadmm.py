"""Complex ADMM-Net with untied matrices (CADMM), after Johnston, Li, Lops, Wang, "ADMM-Net for communication
interference removal in stepped-frequency radar", IEEE TSP 69:2818-2832, 2021 (single-penalty version of its
eqs. (51)-(60)). A baseline of the LADMM family (v28):

    x_t  = M1_t y + M2_t (z_{t-1} - u_{t-1})            M1_t: N x M, M2_t: N x N, real, untied
    xi_t = a_t x_t + (1 - a_t) z_{t-1}                  over-relaxation
    z_t  = S_{lam_t}(xi_t + u_{t-1})                    complex soft threshold, lam_t = softplus(lam_raw_t)
    u_t  = u_{t-1} + eta_t (xi_t - z_t)                 scaled dual
    output z_T

Init: P = (A^T A + I)^-1, M1 = P A^T, M2 = P, a = eta = 1, lam = 1: the first layers are ADMM-LASSO with rho = 1.
LADMM-F is the tied special case M1_t = H_t A^T, M2_t = rho H_t, a_t = 1, when rho is the same in every layer.
Because M1_t acts on y itself, it can also learn the whitened data step A^+ y of the folded noise.
Parameters: T (M N + N^2 + 3): 589,842 for T = 6, N = 256, M = 128. The last eta_T is never used (as LADMM's alpha_T).
Training (layer-wise, like LADMM): stage 'linear' trains M1_t, M2_t on x_t; stage 'nonlinear' trains lam_t, a_t on z_t.
"""
from __future__ import annotations

from typing import List, Optional

import numpy as np
import numpy.linalg as la
import torch
import torch.nn.functional as F

from .common import Stage, UnrolledNet, rmm
from .shrinkage import csoft_ladmm


def _softplus_inv(v: float) -> float:
    return float(np.log(np.expm1(v)))


class CADMM(UnrolledNet):
    family = "cadmm"

    def __init__(self, A, T: int = 6, rho0: float = 1.0, precision: str = "float32", device=None):
        super().__init__(A, precision)
        self.T, self.rho0 = int(T), float(rho0)
        self.cfg = dict(model="CADMM", T=self.T, rho0=self.rho0, precision=precision)
        N = self.N
        Anp = self.A_np
        P = la.inv(Anp.T @ Anp + self.rho0 * np.eye(N))
        M1, M2 = P @ Anp.T, self.rho0 * P
        for t in range(self.T):
            self._p(f"M1_{t}", M1)
            self._p(f"M2_{t}", M2)
            self._p(f"lam_raw_{t}", _softplus_inv(1.0))
            self._p(f"a_{t}", 1.0)
            self._p(f"eta_{t}", 1.0)
        if device is not None:
            self.to(device)

    def lam(self, t):
        return F.softplus(self.P(f"lam_raw_{t}"))

    def stage_plan(self) -> List[Stage]:
        stages, allv = [], []
        for t in range(self.T):
            lin = [f"M1_{t}", f"M2_{t}"]
            allv += lin
            stages.append(Stage(f"CADMM x-update T={t + 1}", "linear", list(lin), list(allv)))
            nl = [f"lam_raw_{t}", f"a_{t}"]
            allv += nl
            stages.append(Stage(f"CADMM z-update T={t + 1}", "nonlinear", list(nl), list(allv)))
            allv.append(f"eta_{t}")
        for i, st in enumerate(stages):
            st.index = i
        return stages

    def forward_stages(self, y, upto: Optional[int] = None):
        y = self.as_complex(y)
        n_st = 2 * self.T if upto is None else upto + 1
        B = y.shape[1]
        Z = torch.zeros((self.N, B), dtype=self.cdtype, device=y.device)
        U = torch.zeros_like(Z)
        outs = []
        for t in range(self.T):
            X = rmm(self.P(f"M1_{t}"), y) + rmm(self.P(f"M2_{t}"), Z - U)
            outs.append(X)
            if len(outs) >= n_st:
                return outs
            a = self.P(f"a_{t}").to(self.cdtype)
            Xi = a * X + (1 - a) * Z
            Z = csoft_ladmm(Xi + U, self.lam(t))
            outs.append(Z)
            if len(outs) >= n_st:
                return outs
            U = U + self.P(f"eta_{t}").to(self.cdtype) * (Xi - Z)
        return outs

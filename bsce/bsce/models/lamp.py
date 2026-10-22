"""LAMP and GM-LAMP for complex beamspace channels (Wei, Hu & Dai, TCOM 2021).

Port of build_LAMP() in the authors' define_networks.py (training graph) and of
LAMP.m / GM_LAMP.m (MATLAB inference). Layer t (t >= 1):

    v_t   = y - A x_{t-1} + b_t v_{t-1} + c_t conj(v_{t-1})      (widely-linear Onsager)
    rvar  = ||v_t||^2 / M
    r_t   = x_{t-1} + B_t v_t
    x_t, dxdr, dxdr1 = eta(r_t, rvar, theta_t);  b = (N/M) dxdr,  c = (N/M) dxdr1

Parameters (and their order) follow the TF graph: B_0, theta_0, theta_1, B_1, theta_2, B_2, ...
Exported .mat names follow load_trained_network.py: B_0..B_{T-1}, theta0..theta{T-1}.
"""
from __future__ import annotations

from typing import List, Optional

import numpy as np
import numpy.linalg as la
import torch

from .common import Stage, UnrolledNet, rmm
from .shrinkage import gm_wei, soft_wei


def gm_theta_init(K: int, mode: str = "original", spread: float = 1e-3) -> np.ndarray:
    """Initial GM parameters [logits; means; log-variances] of each layer.

    'original' : get_shrinkage_function('gm') of define_shrinkage.py - all K components
                 identical. The gradient that separates the components is then exactly zero
                 in exact arithmetic, so in the original training the components split only
                 through amplified round-off (see the verification report).
    'spread'   : the same values plus a deterministic offset spread*linspace(-1, 1, K) on the
                 means and log-variances, so that symmetry breaking is driven by the data and
                 not by round-off (reproducible across frameworks / CPU / GPU).
    """
    th = np.array([np.ones(K) / K, np.zeros(K), -np.ones(K) / 1000000000]).astype(np.float64)
    if mode == "spread":
        off = np.linspace(-1.0, 1.0, K) * spread
        th[1] += off
        th[2] += off
    elif mode != "original":
        raise ValueError(mode)
    return th


class LAMP(UnrolledNet):
    family = "lamp"

    def __init__(self, A, T: int = 8, shrink: str = "soft", untied: bool = True, K: int = 4,
                 precision: str = "float64", device=None, gm_init: str = "original",
                 gm_spread: float = 1e-3, theta_init=None):
        super().__init__(A, precision)
        assert shrink in ("soft", "gm")
        self.T, self.shrink, self.untied, self.K = T, shrink, untied, K
        self.cfg = dict(model="GM-LAMP" if shrink == "gm" else "LAMP", T=T, shrink=shrink,
                        untied=untied, K=K, precision=precision)
        if shrink == "gm" and gm_init != "original":
            self.cfg.update(gm_init=gm_init, gm_spread=gm_spread)
        M, N = self.M, self.N
        B = self.A_np.T / (1.01 * la.norm(self.A_np, 2) ** 2)
        if theta_init is not None:
            th0 = np.asarray(theta_init, dtype=np.float64)
        elif shrink == "soft":
            th0 = np.array([1.0, 1.0])
        else:
            th0 = gm_theta_init(K, gm_init, gm_spread)
        self._p("B_0", B)
        self._p("theta_0", th0)
        for t in range(1, T):
            self._p(f"theta_{t}", th0)
            if untied:
                self._p(f"B_{t}", B)
        if device is not None:
            self.to(device)

    def identical_gm_components(self, names=None) -> List[str]:
        """v31.1: the GM parameters theta_t (of `names`, default all) whose K mixture components are identical."""
        if self.shrink != "gm" or self.K < 2:
            return []
        names = self.param_names() if names is None else names
        out = []
        for n in names:
            if n.startswith("theta_"):
                th = self.P(n)
                if th.dim() == 2 and th.shape[1] == self.K and bool(torch.all(th == th[:, :1])):
                    out.append(n)
        return out

    def break_gm_symmetry(self, names, spread: Optional[float] = None) -> List[str]:
        """v31.1 (bsce_code_31_1, fix of 18 Oct): give the identical mixture components of the GM parameters in `names`
        the deterministic offset of gm_init='spread' (means and log-variances + spread * linspace(-1, 1, K)). Returns the
        names changed.

        Why: GM-LAMP's original init has K identical components. PyTorch then computes bit-identical gradients for them
        (checked on the CPU in float32 and float64; a GPU training log shows the same plateau), so Adam moves them
        together and they never separate. The mixture stays one Gaussian (every entry scaled by the same factor, no use of
        the sparsity), and the network stays near the linear-estimator level: about -2.9 dB after layer 1 on SV ULA at
        10 dB, against -9.2 dB in the authors' log. The authors' TensorFlow code splits them through round-off. The
        trainer calls this before a stage trains such a parameter (train.gm_symmetry='break', the default)."""
        hit = self.identical_gm_components(names)
        if not hit:
            return []
        sp = float(self.cfg.get("gm_spread", 1e-3) if spread is None else spread)
        off = torch.as_tensor(np.linspace(-1.0, 1.0, self.K) * sp, dtype=self.rdtype, device=self.A.device)
        with torch.no_grad():
            for n in hit:
                th = self.P(n)
                th[1].add_(off)
                th[2].add_(off)
        return hit

    def _B(self, t):
        return self.P(f"B_{t}") if self.untied else self.P("B_0")

    def _eta(self, r, rvar, theta):
        if self.shrink == "soft":
            return soft_wei(r, rvar, theta)
        return gm_wei(r, rvar, theta, self.K)

    # ---------------------------------------------------------------------------------
    def stage_plan(self) -> List[Stage]:
        s = self.shrink
        # theta_0 is created before the T=1 layer tuples are appended in build_LAMP, so the
        # var_all of "linear T=1" already contains it (it simply receives no gradient there).
        stages, allv = [], ["B_0", "theta_0"]
        stages.append(Stage(f"LAMP-{s} linear T=1", "linear", ["B_0"], list(allv)))
        stages.append(Stage(f"LAMP-{s} non-linear T=1", "nonlinear", ["theta_0"], list(allv)))
        for t in range(1, self.T):
            allv.append(f"theta_{t}")
            if self.untied:
                allv.append(f"B_{t}")
                stages.append(Stage(f"LAMP-{s} linear T={t + 1}", "linear", [f"B_{t}"], list(allv)))
            stages.append(Stage(f"LAMP-{s} non-linear T={t + 1}", "nonlinear", [f"theta_{t}"], list(allv)))
        for i, st in enumerate(stages):
            st.index = i
        return stages

    def forward_stages(self, y, upto: Optional[int] = None):
        y = self.as_complex(y)
        M, N = self.M, self.N
        n_st = len(self.stage_plan()) if upto is None else upto + 1
        outs = []
        OneOverM = 1.0 / M
        NOverM = torch.tensor(float(N) / M, dtype=self.cdtype, device=y.device)
        rvar = OneOverM * torch.sum(torch.square(torch.abs(y)), 0)
        By = rmm(self._B(0), y)
        outs.append(By)
        if len(outs) >= n_st:
            return outs
        xhat, dxdr, dxdr1 = self._eta(By, rvar, self.P("theta_0"))
        outs.append(xhat)
        v = y
        for t in range(1, self.T):
            if len(outs) >= n_st:
                return outs
            b = NOverM * dxdr
            c = NOverM * dxdr1
            Ax = rmm(self.A, xhat)
            v = y - Ax + b * v + c * torch.conj(v)
            tmp = torch.abs(v)
            rvar = OneOverM * torch.sum(tmp * tmp, 0)
            rhat = xhat + rmm(self._B(t), v)
            if self.untied:
                outs.append(rhat)
                if len(outs) >= n_st:
                    return outs
            xhat, dxdr, dxdr1 = self._eta(rhat, rvar, self.P(f"theta_{t}"))
            outs.append(xhat)
        return outs

    # ---------------------------------------------------------------------------------
    def mat_dict(self):
        d = {}
        for t in range(self.T):
            d[f"B_{t}"] = self._B(t).detach().cpu().numpy()
            d[f"theta{t}"] = self.P(f"theta_{t}").detach().cpu().numpy()
        return d

    def load_mat_dict(self, d):
        with torch.no_grad():
            for t in range(self.T):
                th = self.P(f"theta_{t}")
                key = f"theta{t}" if f"theta{t}" in d else f"theta_{t}"
                th.copy_(torch.as_tensor(np.asarray(d[key], np.float64).reshape(th.shape), dtype=self.rdtype))
                if self.untied or t == 0:
                    B = self.P(f"B_{t}")
                    B.copy_(torch.as_tensor(np.asarray(d[f"B_{t}"], np.float64), dtype=self.rdtype))

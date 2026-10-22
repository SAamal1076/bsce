"""ADMM-Net of Klioui (THADMM-Net repository, models/admmnet.py), PyTorch original.

Layer t (scaled dual v, output x of the last layer):
    tt = max(rho_t, 0)
    x  = (W_t + tt I)^-1 (tt (z - v) + A^H y)        W_t complex N x N, init A^H A (Gram)
    z  = exp(j angle(x + v)) * max(|x + v| - beta_t, 0)
    v  = v + x - z
Here A is the (real) beamspace sensing matrix, so A^H A = A^T A. The port reproduces the
original forward exactly (tests/test_admmnet_vs_thadmm.py imports the original class).
"""
from __future__ import annotations

from typing import List, Optional

import numpy as np
import torch

from .common import Stage, UnrolledNet


class ADMMNet(UnrolledNet):
    family = "admmnet"

    def __init__(self, A, T: int = 15, beta: float = 0.1, rho: float = 1.0,
                 precision: str = "float64", device=None):
        super().__init__(A, precision)
        self.T = T
        self.cfg = dict(model="ADMM-Net", T=T, beta=beta, rho=rho, precision=precision)
        A_ = self.A_np.astype(np.complex128)
        G = A_.conj().T @ A_
        for t in range(T):
            self.register_parameter(f"W_{t}", torch.nn.Parameter(torch.tensor(G, dtype=self.cdtype)))
            self._param_order.append(f"W_{t}")
            self._p(f"rho_{t}", np.array([rho]))
            self._p(f"beta_{t}", np.array([beta]))
        if device is not None:
            self.to(device)

    def stage_plan(self) -> List[Stage]:
        allv = []
        out = []
        for t in range(self.T):
            new = [f"W_{t}", f"rho_{t}", f"beta_{t}"]
            allv += new
            out.append(Stage(f"ADMM-Net T={t + 1}", "layer", new, list(allv), t))
        return out

    def forward_stages(self, y, upto: Optional[int] = None):
        y = self.as_complex(y)
        N = self.N
        n_st = self.T if upto is None else upto + 1
        Ac = self.A.to(self.cdtype)
        yf = Ac.T.conj() @ y
        x = torch.zeros((N, y.shape[1]), dtype=self.cdtype, device=y.device)
        z = torch.zeros_like(x)
        v = torch.zeros_like(x)
        I = torch.eye(N, dtype=self.cdtype, device=y.device)
        outs = []
        for t in range(n_st):
            tt = torch.max(self.P(f"rho_{t}"), torch.tensor([0], device=y.device))
            W_tmp = torch.linalg.inv(self.P(f"W_{t}") + tt * I)
            x = W_tmp @ (tt * (z - v) + yf)
            u = x + v
            zeros = torch.zeros(u.size(), device=y.device)
            z = torch.exp(1j * u.angle()) * torch.max(torch.abs(u) - self.P(f"beta_{t}"), zeros)
            v = v + x - z
            outs.append(x)
        return outs

    def mat_dict(self):
        d = {}
        with torch.no_grad():
            for t in range(self.T):
                W = self.P(f"W_{t}")
                tt = max(float(self.P(f"rho_{t}")), 0.0)
                d[f"W_{t}"] = W.cpu().numpy()
                d[f"rho_{t}"] = np.array(float(self.P(f"rho_{t}")))
                d[f"beta_{t}"] = np.array(float(self.P(f"beta_{t}")))
                # materialised x-update operator for MATLAB: x = Winv_t*(rho_t*(z-v) + A'*y)
                d[f"Winv_{t}"] = torch.linalg.inv(W + tt * torch.eye(self.N, dtype=W.dtype, device=W.device)).cpu().numpy()
        d["T"] = np.array(self.T)
        return d

    def load_mat_dict(self, d):
        with torch.no_grad():
            for t in range(self.T):
                self.P(f"W_{t}").copy_(torch.as_tensor(np.asarray(d[f"W_{t}"]), dtype=self.cdtype))
                self.P(f"rho_{t}").copy_(torch.as_tensor(np.asarray(d[f"rho_{t}"], np.float64).reshape(1), dtype=self.rdtype))
                self.P(f"beta_{t}").copy_(torch.as_tensor(np.asarray(d[f"beta_{t}"], np.float64).reshape(1), dtype=self.rdtype))

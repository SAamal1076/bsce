"""UTRR: unfolded trimmed-ridge regression (Wu, Cheng, Eldar, Cioffi, "Learned Trimmed-Ridge Regression for Channel
Estimation in Millimeter-Wave Massive MIMO", IEEE TCOM 73(2):1128-1141, 2025), written from the paper (no code was
published). Checked against the standalone reference tests/proposals/utrr_ref.py (tests/test_utrr.py).

Real-valued, as in the paper: the real and imaginary parts of every column are two real samples (eq. 8-9), and the
sensing matrix is real. With bsce's y = A (x + n) the real and imaginary parts stay separable, so the model is the
paper's network applied to [Re y, Im y].

    x0 = A^T y,  z0 = [(x0)+; (-x0)+],  [BN_0]                                    (initialising layer)
    layer t:  g = A_t^T (A_t z - y) + rho_t (z - zbar_K^(t))                        (eq. 23 / Algorithm 2)
              z <- ( ReLU( [BN_t]( z - alpha_t g ) ) + z ) / 2                      (beta = 1/2)
    xhat = u - v,  z = [u; v]
  zbar_K: z with only its K largest entries kept. RCC (Sec. IV-B, default): K = 0 in every layer but the last.
  A_t in R^{M x 2N} (init [A, -A]), rho_t (init 1), alpha_t (init 0.1): L (2MN + 2) parameters, the paper's count.
  BN (bn=True): the batch normalisation drawn in Fig. 6 (not in Algorithm 2 nor in the parameter count), 2 x 2N
  parameters per BN layer; batch statistics while training (gradients on), running statistics otherwise. Default off:
  in the smoke tests of docs/UTRR.md it trained far worse than without.
  Sign: the paper prints A^T (y - A z); its eq. 16 gives A^T (A z - y), used here (the printed form diverges; see
  docs/UTRR.md).
  RCC and depth: the top-K term sits in the LAST layer used. forward_stages(y, upto=t)[-1] is therefore the output of
  the (t+1)-layer network (it equals the truncated network exactly), which is what layer-wise training and
  --save-layers need. The earlier entries of the returned list are the outputs of the layers inside that network.
"""
from __future__ import annotations

from typing import List, Optional

import numpy as np
import torch

from .common import Stage, UnrolledNet


def topk_keep(z: torch.Tensor, K: int) -> torch.Tensor:
    if K <= 0:
        return torch.zeros_like(z)
    v, i = torch.topk(z, min(K, z.shape[0]), dim=0)
    return torch.zeros_like(z).scatter(0, i, v)


class UTRR(UnrolledNet):
    family = "utrr"

    def __init__(self, A, T: int = 30, K: int = 16, rcc: bool = True, bn: bool = False, rho: float = 1.0,
                 alpha: float = 0.1, bn_momentum: float = 0.1, precision: str = "float32", device=None):
        super().__init__(A, precision)
        self.T, self.K, self.rcc, self.bn = int(T), int(K), bool(rcc), bool(bn)
        self.bn_momentum, self.bn_eps = float(bn_momentum), 1e-5
        self.cfg = dict(model="UTRR", T=self.T, K=self.K, rcc=self.rcc, bn=self.bn, rho=rho, alpha=alpha,
                        bn_momentum=self.bn_momentum, precision=precision)
        A2 = np.hstack([self.A_np, -self.A_np])
        n2 = 2 * self.N
        if self.bn:
            self._bn_params("bn0", n2)
        for t in range(self.T):
            self._p(f"A_{t}", A2)
            self._p(f"rho_{t}", np.array([rho]))
            self._p(f"alpha_{t}", np.array([alpha]))
            if self.bn:
                self._bn_params(f"bn{t + 1}", n2)
        if device is not None:
            self.to(device)

    def _bn_params(self, name, n):
        self._p(f"{name}_gamma", np.ones((n, 1)))
        self._p(f"{name}_beta", np.zeros((n, 1)))
        self.register_buffer(f"{name}_mean", torch.zeros((n, 1), dtype=self.rdtype))
        self.register_buffer(f"{name}_var", torch.ones((n, 1), dtype=self.rdtype))

    def _norm(self, name, z):
        mean, var = getattr(self, f"{name}_mean"), getattr(self, f"{name}_var")
        if self.training and torch.is_grad_enabled():
            m = z.mean(1, keepdim=True)
            v = z.var(1, unbiased=False, keepdim=True)
            with torch.no_grad():
                n = z.shape[1]
                mean.mul_(1 - self.bn_momentum).add_(self.bn_momentum * m)
                var.mul_(1 - self.bn_momentum).add_(self.bn_momentum * v * n / max(n - 1, 1))
        else:
            m, v = mean, var
        return self.P(f"{name}_gamma") * (z - m) / torch.sqrt(v + self.bn_eps) + self.P(f"{name}_beta")

    def stage_plan(self) -> List[Stage]:
        allv, out = [], []
        for t in range(self.T):
            new = (["bn0_gamma", "bn0_beta"] if (t == 0 and self.bn) else []) + [f"A_{t}", f"rho_{t}", f"alpha_{t}"]
            if self.bn:
                new += [f"bn{t + 1}_gamma", f"bn{t + 1}_beta"]
            allv += new
            out.append(Stage(f"UTRR T={t + 1}", "layer", new, list(allv), t))
        return out

    def forward_real(self, yr: torch.Tensor, upto: Optional[int] = None) -> List[torch.Tensor]:
        """The paper's real-valued network: yr (M, R) real samples -> the output of every layer, (N, R) each."""
        N = self.N
        last = self.T - 1 if upto is None else int(upto)
        x0 = self.A.T @ yr
        z = torch.cat([torch.relu(x0), torch.relu(-x0)], 0)
        if self.bn:
            z = self._norm("bn0", z)
        outs = []
        for t in range(last + 1):
            At = self.P(f"A_{t}")
            K = self.K if (not self.rcc or t == last) else 0
            g = At.T @ (At @ z - yr) + self.P(f"rho_{t}") * (z - topk_keep(z, K))
            q = z - self.P(f"alpha_{t}") * g
            if self.bn:
                q = self._norm(f"bn{t + 1}", q)
            z = (torch.relu(q) + z) / 2
            outs.append(z[:N] - z[N:])
        return outs

    def forward_stages(self, y, upto: Optional[int] = None):
        """Complex columns (bsce): their real and imaginary parts are 2B real samples of one batch."""
        y = self.as_complex(y)
        B = y.shape[1]
        yr = torch.cat([y.real, y.imag], 1).to(self.rdtype)
        return [torch.complex(xr[:, :B], xr[:, B:]) for xr in self.forward_real(yr, upto)]

    # -- export: the BN running statistics travel with the parameters ------------------------------------------------
    def _bn_buffer_names(self):
        names = (["bn0"] if self.bn else []) + ([f"bn{t + 1}" for t in range(self.T)] if self.bn else [])
        return [f"{n}_{s}" for n in names for s in ("mean", "var")]

    def mat_dict(self):
        d = super().mat_dict()
        for b in self._bn_buffer_names():
            d[b] = getattr(self, b).detach().cpu().numpy()
        return d

    def load_mat_dict(self, d):
        super().load_mat_dict(d)
        with torch.no_grad():
            for b in self._bn_buffer_names():
                if b in d:
                    getattr(self, b).copy_(torch.as_tensor(np.asarray(d[b], np.float64).reshape(getattr(self, b).shape),
                                                           dtype=self.rdtype))

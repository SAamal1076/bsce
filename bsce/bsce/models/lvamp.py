"""Learned VAMP with the SVD parameterisation (Borgerding, Schniter & Rangan, TSP 2017).

Port of build_LVAMP() in tools/networks.py. Learns the measurement-noise log-variance
(logyvar) and the shrinkage parameters theta_t; the LMMSE stage uses the SVD of A.

``complex_mode=True`` extends it to complex beamspace channels (A real, so U, s, V are
real and act on Re/Im; the Bernoulli-Gaussian denoiser becomes its circular-complex
version with the Wirtinger divergence). ``complex_mode=False`` = the original real code.

``noise_model`` (v27 option): the noise the LMMSE stage assumes.
* 'white' (default: Borgerding's LVAMP, and bsce up to v26): y = A x + w, w white. In the SVD domain
  ytilde = S^-1 U^T y = V^T x + S^-1 U^T w, so component i has noise variance yvar / s_i^2.
* 'colored' (folded noise, y = A (x + n): GM-LAMP's model and bsce's default data): ytilde = V^T (x + n), so every
  component has the same noise variance yvar; the LMMSE weights are uniform in the SVD domain.
'colored' is the statistically matched stage for bsce's data, but trained it was worse than 'white' on SV ULA
(T6 end to end, 2 seeds: test mean over 0-20 dB -7.83 vs -8.41 dB; T2 with its own layer-wise protocol: -5.66 vs
-5.76 dB; better at low SNR, worse above 6-8 dB, by up to 1.4 dB): the single
learned yvar cannot follow the per-sample noise level of a 0-20 dB band, and the 1/s_i^2 weights of 'white' damp the
poorly conditioned directions. So 'white' stays the default; a network saved without the key loads as 'white'.
"""
from __future__ import annotations

import math
from typing import List, Optional

import numpy as np
import numpy.linalg as la
import torch

from .common import Stage, UnrolledNet, rmm
from .shrinkage import bg_complex, bg_real


class LVAMP(UnrolledNet):
    family = "lvamp"

    def __init__(self, A, T: int = 6, shrink: str = "bg", complex_mode: bool = True,
                 precision: str = "float64", device=None, noise_model: str = "white"):
        super().__init__(A, precision)
        assert shrink == "bg", "only the Bernoulli-Gaussian shrinkage is ported"
        from ..data.noise import canonical
        self.noise_model = canonical(noise_model)
        self.T, self.shrink, self.complex_mode = T, shrink, complex_mode
        self.cfg = dict(model="LVAMP", T=T, shrink=shrink, complex_mode=complex_mode, precision=precision,
                        noise_model=self.noise_model)
        for k, v in self._svd_buffers(self.A_np).items():
            self.register_buffer(k, torch.tensor(v, dtype=self.rdtype))
        theta0 = np.array([1.0, math.log(1 / .1 - 1)])
        self._p("logyvar", 0.0)
        for t in range(T):
            self._p(f"theta_{t}", theta0)
        if device is not None:
            self.to(device)

    @staticmethod
    def _svd_buffers(A_):
        s2, U = la.eigh(A_ @ A_.T)
        s = np.sqrt(s2)
        return dict(V=(A_.T @ U) / s, rS2=np.reshape(1 / (s * s), (-1, 1)), UsT=(U / s).T)

    def set_combiner(self, A):
        """v31: use another combiner A (M' x N, any M'). LVAMP learns logyvar and two thresholds per layer, nothing
        tied to A; the SVD buffers V, rS2, UsT are recomputed."""
        A = self._combiner_array(A)
        self.M, self.A_np = A.shape[0], A
        self._put_buffer("A", A)
        for k, v in self._svd_buffers(A).items():
            self._put_buffer(k, v)

    def stage_plan(self) -> List[Stage]:
        out, allv = [], ["logyvar"]
        for t in range(self.T):
            allv.append(f"theta_{t}")
            new = [] if t == 0 else [f"theta_{t}"]     # t == 0: learnvars=None ("all")
            out.append(Stage(f"LVAMP-{self.shrink} T={t + 1}", "nonlinear", new, list(allv), t))
        return out

    def forward_stages(self, y, upto: Optional[int] = None):
        cm = self.complex_mode
        if cm:
            y = self.as_complex(y)
            mm = rmm
            sq = lambda z: torch.square(torch.abs(z))
            eta = bg_complex
        else:
            y = (y if torch.is_tensor(y) else torch.as_tensor(np.asarray(y))).to(self.A.device, self.rdtype)
            mm = lambda W, z: W @ z
            sq = lambda z: z * z
            eta = bg_real
        n_st = self.T if upto is None else upto + 1
        N = self.N
        rj = torch.zeros((N, y.shape[-1]), dtype=y.dtype, device=y.device)
        taurj = torch.sum(sq(y), 0) / N
        yvar = torch.exp(self.P("logyvar"))
        ytilde = mm(self.UsT, y)
        Vt = self.V.T
        outs = []
        w = self.rS2 if self.noise_model == "white" else torch.ones_like(self.rS2)   # noise variance per component / yvar
        for t in range(n_st):
            varRat = torch.reshape(yvar / taurj, (1, -1))
            scale_each = 1 / (1 + w * varRat)
            zetai = N / torch.sum(scale_each, 0)
            diff = ytilde - mm(Vt, rj)
            adjust = (scale_each.to(y.dtype) * diff) * zetai.to(y.dtype)
            ri = rj + mm(self.V, adjust)
            tauri = taurj * (zetai - 1)
            xhat, dxdr = eta(ri, tauri, self.P(f"theta_{t}"))
            outs.append(xhat)
            zetaj = 1 / (1 - dxdr)
            rj = (xhat - dxdr.to(y.dtype) * ri) * zetaj.to(y.dtype)
            taurj = tauri * (zetaj - 1)
        return outs

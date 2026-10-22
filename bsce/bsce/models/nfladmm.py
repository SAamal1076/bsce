"""NF-LADMM: noise-folding-aware learned ADMM (proposal NF-LADMM_proposal.md, 14 Oct 2026), as a bsce model.

Measurement model y = A (x + n), n ~ CN(0, s2 I). Exact likelihood y | x ~ CN(Ax, s2 AA^T). Per layer t (v = z - u):
    r_t   = y - A v
    x_t   = v + beta_t A^+ r_t                       A^+ = A^T (AA^T)^-1, beta_t = sigmoid(rho_raw_t) in (0, 1)
  The x-step is the exact proximal map of the likelihood, x = v + A^T (AA^T + rho~ Sigma~)^-1 r, with the noise
  covariance normalised by s2 (Sigma~ = Sigma / s2) and rho~_t = 1/beta_t - 1:
    noise_model 'colored' (default): Sigma~ = AA^T   -> x_t = v + beta_t A^+ r_t          (the form above)
    noise_model 'white'            : Sigma~ = c I    -> x_t = v + A^T U diag(1 / (lam_i + c rho~_t)) U^T r_t
                                     (AA^T = U diag(lam) U^T, c = tr(AA^T)/M: y = A x + w, w ~ CN(0, c s2 I);
                                      bsce/data/noise.py). The same parameters; only the data step changes, so a
                                      network can also be run with the other step (evaluate.py --nf-step matched).
    w_t   = x_t + u
    s_t^2 = exp(m_t) ||r_t||^2 / N + exp(k_t) s2    (sigma='known'; sigma='residual' drops the s2 term)
    z_t   = s_t psi_t(|w_t|/s_t) w_t/|w_t|  [+ gamma_t s_t C(w_t/s_t, z_shrink/s_t; log s_t)   (denoiser 'conv')]
    u_t   = u + alpha_t (x_t - z_t)                  alpha_t = golden * sigmoid(alpha_raw_t)   (update='dual')
  or the suggested form (update='relax', NF-LADMM_theory.md / NF-LADMM_convergence.md):
    x_t  <- lam_t x_t + (1 - lam_t) z_{t-1}         lam_t = 2 sigmoid(lam_raw_t) in (0, 2), over-relaxation
    w_t   = x_t + u,  z_t = D_t(w_t),  u_t = w_t - z_t  (= relaxed Douglas-Rachford in zeta = z + u)
    output z_T
psi_t: monotone piecewise-linear radial shrinkage (16 knots on [0, 6], slopes in (0, 1)), init ~ soft threshold at the
first knot >= thr0 (knots 0, 0.4, 0.8, 1.2, ...: thr0 = 1 gives 1.2 s_t)
(slope logits +-slope_init; 6 = original, saturated: the shape hardly trains; 2 = suggested).
C: shift-equivariant circular CNN on the beam grid (1-D kernel 9 for the ULA, 2-D 5x5 on the N1 x N2 grid for the UPA),
3 layers, `hidden` channels, FiLM on log s_t; shared over layers (share_cnn=True).
  NF-LADMM-C = denoiser 'radial' (certified, 160 parameters at T = 8), NF-LADMM-S = denoiser 'conv'.

The equations and the parameterisation are those of the proposal's nf_ladmm.py (reference/proposals/nf_ladmm.py);
tests/test_nfladmm.py checks that this class gives the same output as that module with the same weights.
bsce conventions: y is (M, B), outputs are (N, B); A is fixed by the scenario (the network does not learn it);
the model needs the noise variance per column (forward(y, s2=...)) when sigma='known'.
"""
from __future__ import annotations

import math
from typing import Dict, List, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .common import Stage, UnrolledNet

GOLD = (1 + 5 ** 0.5) / 2


class RadialShrink(nn.Module):
    """psi(r) = sum_k inc_k relu(r - b_k), cumulative slopes sigmoid(slope_logit) in (0, 1)."""

    def __init__(self, K=16, rmax=6.0, thr0=1.0, slope_init=6.0):
        super().__init__()
        self.register_buffer("knots", torch.linspace(0, rmax, K))
        s = torch.full((K,), -float(slope_init))
        s[self.knots >= thr0] = float(slope_init)
        self.slope_logit = nn.Parameter(s)

    def forward(self, r):
        slopes = torch.sigmoid(self.slope_logit)
        inc = torch.diff(slopes, prepend=slopes.new_zeros(1))
        return (inc * F.relu(r.unsqueeze(-1) - self.knots)).sum(-1)


class CircConvResidual(nn.Module):
    """Shift-equivariant residual on the beam grid; complex -> 2 real channels (+ shrinkage output)."""

    def __init__(self, array, hidden=16, k=9, N1=16, N2=16):
        super().__init__()
        self.array, self.k, self.N1, self.N2 = array, k, N1, N2
        conv = nn.Conv1d if array == "ULA" else nn.Conv2d
        self.c1 = conv(4, hidden, k, padding=0)
        self.c2 = conv(hidden, hidden, k, padding=0)
        self.c3 = conv(hidden, 2, k, padding=0)
        self.film = nn.Linear(1, 2 * hidden)
        nn.init.normal_(self.c3.weight, std=1e-3)
        nn.init.zeros_(self.c3.bias)

    def _pad(self, h):
        p = self.k // 2
        return F.pad(h, (p, p) if self.array == "ULA" else (p, p, p, p), mode="circular")

    def forward(self, v, zs, logsig):              # v, zs: (B, N) complex
        B, N = v.shape
        h = torch.stack([v.real, v.imag, zs.real, zs.imag], 1)
        if self.array == "UPA":
            h = h.view(B, 4, self.N1, self.N2)     # beam index i1*N2 + i2 (kron(U_N1, U_N2))
        g, b = self.film(logsig.view(B, 1)).chunk(2, -1)
        shp = (B, -1, 1) if self.array == "ULA" else (B, -1, 1, 1)
        h = F.gelu(self.c1(self._pad(h)) * (1 + g.view(shp)) + b.view(shp))
        h = F.gelu(self.c2(self._pad(h)))
        h = self.c3(self._pad(h)).reshape(B, 2, N)
        return torch.complex(h[:, 0], h[:, 1])


class NFLADMM(UnrolledNet):
    family = "nfladmm"
    needs_noise_var = True

    def __init__(self, A, T: int = 8, array: str = "ULA", denoiser: str = "conv", share_cnn: bool = True,
                 thr0: float = 1.0, hidden: int = 16, kernel: Optional[int] = None, sigma: str = "known",
                 N1: int = 16, N2: int = 16, update: str = "dual", slope_init: float = 6.0,
                 noise_model: str = "colored", white_step: str = "normalized", precision: str = "float32", device=None,
                 seed: Optional[int] = None):
        super().__init__(A, precision)
        assert denoiser in ("radial", "conv"), denoiser
        assert sigma in ("known", "residual"), sigma
        assert update in ("dual", "relax"), update
        array = array.upper()
        kernel = kernel or (9 if array == "ULA" else 5)
        self.T, self.array, self.denoiser, self.sigma, self.update = T, array, denoiser, sigma, update
        self.needs_noise_var = sigma == "known"
        from ..data.noise import canonical
        self.noise_model = canonical(noise_model)      # the data step (can be switched for evaluation)
        assert white_step in ("normalized", "adaptive"), white_step
        self.white_step = white_step
        self.cfg = dict(model="NF-LADMM", T=T, array=array, denoiser=denoiser, share_cnn=share_cnn, thr0=thr0,
                        hidden=hidden, kernel=kernel, sigma=sigma, N1=N1, N2=N2, update=update,
                        slope_init=float(slope_init), noise_model=self.noise_model, white_step=white_step,
                        precision=precision)
        if seed is not None:
            torch.manual_seed(seed)
        Ad = torch.as_tensor(self.A_np, dtype=torch.float64)
        self.register_buffer("Apinv", (Ad.T @ torch.linalg.inv(Ad @ Ad.T)).to(self.rdtype))   # (N, M)
        # white data step (not saved: computed from A, so files of earlier versions load unchanged)
        lam, U = torch.linalg.eigh(Ad @ Ad.T)
        self.register_buffer("eig_U", U.to(self.rdtype), persistent=False)                     # (M, M)
        self.register_buffer("eig_AtU", (Ad.T @ U).to(self.rdtype), persistent=False)          # (N, M)
        self.register_buffer("eig_lam", lam.to(self.rdtype), persistent=False)                 # (M,)
        self.c_white = float(torch.sum(Ad * Ad) / Ad.shape[0])
        self.rho_raw = nn.Parameter(torch.zeros(T, dtype=self.rdtype))
        if update == "dual":
            self.alpha_raw = nn.Parameter(torch.full((T,), 0.4, dtype=self.rdtype))
        else:
            self.lam_raw = nn.Parameter(torch.zeros(T, dtype=self.rdtype))      # lam_t = 1 at init
        self.m = nn.Parameter(torch.zeros(T, dtype=self.rdtype))
        self.k = nn.Parameter(torch.full((T,), -3.0, dtype=self.rdtype))
        self.shrink = nn.ModuleList([RadialShrink(thr0=thr0, slope_init=slope_init) for _ in range(T)])
        if denoiser == "conv":
            n_cnn = 1 if share_cnn else T
            self.cnn = nn.ModuleList([CircConvResidual(array, hidden=hidden, k=kernel, N1=N1, N2=N2)
                                      for _ in range(n_cnn)])
            self.gamma = nn.Parameter(torch.full((T,), 0.1, dtype=self.rdtype))
        self.to(self.rdtype)
        if device is not None:
            self.to(device)

    def set_combiner(self, A):
        """v31: use another combiner A (M' x N, any M'). Nothing learned depends on A (the scalars, the radial
        shrinkages and the CNN act in the beam domain); A, A^+, the eigen-decomposition of A A^T and c_white are
        recomputed, so the network equals one built on A with the same weights."""
        A = self._combiner_array(A)
        Ad = torch.as_tensor(A, dtype=torch.float64)
        lam, U = torch.linalg.eigh(Ad @ Ad.T)
        self.M, self.A_np = A.shape[0], A
        self._put_buffer("A", A)
        self._put_buffer("Apinv", (Ad.T @ torch.linalg.inv(Ad @ Ad.T)).numpy())
        self._put_buffer("eig_U", U.numpy())
        self._put_buffer("eig_AtU", (Ad.T @ U).numpy())
        self._put_buffer("eig_lam", lam.numpy())
        self.c_white = float(torch.sum(Ad * Ad) / Ad.shape[0])

    # -- bsce interface ------------------------------------------------------------
    def P(self, name):
        return dict(self.named_parameters())[name]

    def param_names(self) -> List[str]:
        return [n for n, _ in self.named_parameters()]

    def stage_plan(self) -> List[Stage]:
        names = self.param_names()
        # every layer: all parameters (the CNN is shared and rho/alpha/m/k/gamma are per-layer vectors), so a layer-wise
        # schedule trains all of them at lr, then refines them; end-to-end schedules use param_names() directly
        return [Stage(f"NF-LADMM layer T={t + 1}", "layer", list(names), list(names), index=t) for t in range(self.T)]

    def forward(self, y, s2=None):
        return self.forward_stages(y, s2=s2)[-1]

    def forward_stages(self, y, upto: Optional[int] = None, s2=None):
        y = self.as_complex(y).T                    # (B, M)
        B, N = y.shape[0], self.N
        if self.sigma == "known":
            if s2 is None:
                raise ValueError("NF-LADMM with sigma='known' needs the noise variance s2 of every column")
            s2 = torch.as_tensor(s2, dtype=self.rdtype, device=y.device).reshape(-1)
            if s2.numel() == 1:
                s2 = s2.expand(B)
        Ac = self.A.to(self.cdtype)
        z = y.new_zeros(B, N)
        u = y.new_zeros(B, N)
        outs = []
        n_out = self.T if upto is None else upto + 1
        for t in range(n_out):
            v = z - u
            res = y - v @ Ac.T
            x = self.data_step(v, res, t, s2)
            if self.update == "relax":
                lam = 2 * torch.sigmoid(self.lam_raw[t])
                x = lam * x + (1 - lam) * z
            w = x + u
            s_eff = torch.exp(self.m[t]) * res.abs().pow(2).sum(1) / N
            if self.sigma == "known":
                s_eff = s_eff + torch.exp(self.k[t]) * s2
            sig = torch.sqrt(s_eff).view(B, 1)
            r = w.abs()
            zs = (self.shrink[t](r / sig) * sig) * w / r.clamp_min(1e-12)
            if self.denoiser == "conv":
                cnn = self.cnn[0] if len(self.cnn) == 1 else self.cnn[t]
                zs = zs + self.gamma[t] * sig * cnn(w / sig, zs / sig, torch.log(sig))
            z = zs
            if self.update == "relax":
                u = w - z
            else:
                u = u + GOLD * torch.sigmoid(self.alpha_raw[t]) * (x - z)
            outs.append(z.T)
        return outs

    S2_REF = 0.1          # white_step 'adaptive': the noise variance at which kappa_t = c (1/beta_t - 1) (10 dB)

    def data_step(self, v, res, t: int, s2=None):
        """x-step of layer t (rows are samples: v (B, N), res = y - A v (B, M)): the exact proximal map
        x = v + A^T (AA^T + rho~_t Sigma~)^-1 res, rho~_t = 1/beta_t - 1, Sigma~ = AA^T (colored) or c I (white).
        white_step 'normalized' (default): kappa_t = c rho~_t for every SNR (the penalty scales with 1/s2, as the
        colored step); 'adaptive': kappa_t,b = c rho~_t s2_b / S2_REF, the exact prox of an ADMM with a FIXED penalty
        (kappa -> 0 at high SNR: the step tends to the projection A^+; needs s2)."""
        beta = torch.sigmoid(self.rho_raw[t])
        if self.noise_model == "white":
            Ue, AtU = self.eig_U.to(self.cdtype), self.eig_AtU.to(self.cdtype)
            kappa = self.c_white * (1 / beta - 1)
            if self.white_step == "adaptive":
                if s2 is None:
                    raise ValueError("white_step 'adaptive' needs the noise variance s2 of every column")
                s2 = torch.as_tensor(s2, dtype=self.rdtype, device=res.device).reshape(-1, 1)
                kappa = kappa * (s2 / self.S2_REF)
            return v + ((res @ Ue) / (self.eig_lam + kappa)) @ AtU.T
        return v + beta * (res @ self.Apinv.to(self.cdtype).T)

    # -- export: every parameter, dots -> '__' (MATLAB-safe names) ---------------------
    def mat_dict(self) -> Dict[str, np.ndarray]:
        return {n.replace(".", "__"): p.detach().cpu().numpy() for n, p in self.named_parameters()}

    def load_mat_dict(self, d) -> None:
        with torch.no_grad():
            for n, p in self.named_parameters():
                v = np.asarray(d[n.replace(".", "__")], dtype=np.float64).reshape(tuple(p.shape))
                p.copy_(torch.as_tensor(v, dtype=p.dtype))

"""NF-LADMM: noise-folding-aware learned ADMM for lens-array beamspace channel estimation.

Measurement model (GM-LAMP / LADMM setting):  y = A (x + n),  n ~ CN(0, s2 I),  A in R^{M x N}.
Exact likelihood y | x ~ CN(Ax, s2 AA^T). Scaled-form ADMM on
      min_x  1/(2 s2) || (AA^T)^{-1/2} (y - A x) ||^2 + g(x)
gives per layer t (v = z - u):
      x_t = v + beta_t * A^+ (y - A v)            A^+ = A^T (AA^T)^{-1},  beta_t = 1/(1 + rho_t s2) in (0,1)
      z_t = D_t(x_t + u ; sigma_t)                  learned prior step (prox of g, or a denoiser)
      x_t <- lam_t x_t + (1 - lam_t) z_{t-1}         over-relaxation, lam_t in (0, 2)
      u_t = w_t - z_t,  w_t = x_t + u                  (= relaxed Douglas-Rachford in zeta = z + u; THEORY.md)
The x-step is the exact proximal map of the whitened data term: no N x N inverse, O(MN) per layer,
and A enters as an INPUT (the network is not tied to one combiner).

Denoisers
  'radial'  : learned monotone radial shrinkage z = psi(|v|/sigma) * sigma * v/|v|, psi piecewise
              linear with slopes in [0,1], psi(0)=0  ->  prox of a convex function (NF-LADMM-C).
  'conv'    : radial + gamma_t * circular-conv residual (1-D circular for ULA, 2-D circular for UPA,
              equivariant to beam shifts), conditioned on log(sigma) (NF-LADMM-S).
sigma_t^2 = exp(m_t)||y - A v||^2/N + exp(k_t) s2 is the effective noise level of the denoiser input
(residual estimate as in LAMP plus the known folded-noise floor; m_t, k_t learned).

Complex tensors are used throughout (torch.cfloat). Shapes: y (B,M), A (M,N) or (B,M,N), s2 (B,).
"""
import math, torch, torch.nn as nn, torch.nn.functional as F

GOLD = (1 + 5 ** 0.5) / 2


class RadialShrink(nn.Module):
    """psi(r) = sum_k w_k * relu(r - b_k) with cumulative slopes in [0,1] (monotone, nonexpansive)."""
    def __init__(self, K=16, rmax=6.0, thr0=1.0):
        super().__init__()
        self.register_buffer("knots", torch.linspace(0, rmax, K))
        # init ~ soft threshold at thr0: slope 0 below thr0, 1 above
        s = torch.full((K,), -6.0); s[self.knots >= thr0] = 6.0
        self.slope_logit = nn.Parameter(s)

    def forward(self, r):                         # r >= 0, any shape
        slopes = torch.sigmoid(self.slope_logit)  # slope on [b_k, b_{k+1}] in (0,1)
        inc = torch.diff(slopes, prepend=slopes.new_zeros(1))
        return (inc * F.relu(r.unsqueeze(-1) - self.knots)).sum(-1)


class CircConvResidual(nn.Module):
    """Shift-equivariant residual on the beam grid; complex -> 2 real channels (+ psi output)."""
    def __init__(self, array, hidden=16, k=9):
        super().__init__()
        self.array, self.k = array, k
        conv = nn.Conv1d if array == "ULA" else nn.Conv2d
        self.c1 = conv(4, hidden, k, padding=0)
        self.c2 = conv(hidden, hidden, k, padding=0)
        self.c3 = conv(hidden, 2, k, padding=0)
        self.film = nn.Linear(1, 2 * hidden)         # FiLM on log(sigma)
        nn.init.normal_(self.c3.weight, std=1e-3); nn.init.zeros_(self.c3.bias)  # near-zero, not zero: all params get gradient

    def _pad(self, h):
        p = self.k // 2
        return F.pad(h, (p, p) if self.array == "ULA" else (p, p, p, p), mode="circular")

    def forward(self, v, zs, logsig):              # v, zs: (B,N) complex
        B, N = v.shape
        h = torch.stack([v.real, v.imag, zs.real, zs.imag], 1)
        if self.array == "UPA":
            n1 = int(round(N ** 0.5)); h = h.view(B, 4, n1, n1)
        g, b = self.film(logsig.view(B, 1)).chunk(2, -1)
        shp = (B, -1, 1) if self.array == "ULA" else (B, -1, 1, 1)
        h = F.gelu(self.c1(self._pad(h)) * (1 + g.view(shp)) + b.view(shp))
        h = F.gelu(self.c2(self._pad(h)))
        h = self.c3(self._pad(h)).reshape(B, 2, N)
        return torch.complex(h[:, 0], h[:, 1])


class NFLADMM(nn.Module):
    def __init__(self, T=8, array="ULA", denoiser="conv", share_cnn=True, thr0=1.0):
        super().__init__()
        self.T, self.denoiser = T, denoiser
        self.rho_raw = nn.Parameter(torch.zeros(T))           # beta_t = sigmoid(rho_raw)
        self.alpha_raw = nn.Parameter(torch.full((T,), 0.4))  # alpha_t = GOLD*sigmoid -> ~0.97 init
        # sigma_t^2 = exp(m_t) ||y - A v||^2 / N + exp(k_t) s2   (residual estimate + folded-noise floor)
        self.m = nn.Parameter(torch.zeros(T))
        self.k = nn.Parameter(torch.full((T,), -3.0))
        self.shrink = nn.ModuleList([RadialShrink(thr0=thr0) for _ in range(T)])
        if denoiser == "conv":
            n_cnn = 1 if share_cnn else T
            self.cnn = nn.ModuleList([CircConvResidual(array, k=9 if array == 'ULA' else 5) for _ in range(n_cnn)])
            self.gamma = nn.Parameter(torch.full((T,), 0.1))   # not 0: with c3 = 0 both grads would vanish

    @staticmethod
    def pinv(A):
        At = A.transpose(-1, -2)
        return At @ torch.linalg.inv(A @ At)                  # (.., N, M), real

    def forward(self, y, A, s2, return_all=False):
        cdt = y.dtype
        Ap = self.pinv(A).to(cdt); Ac = A.to(cdt)
        mv = (lambda Mx, v: v @ Mx.transpose(-1, -2)) if A.dim() == 2 else \
             (lambda Mx, v: torch.einsum("bij,bj->bi", Mx, v))
        B, N = y.shape[0], A.shape[-1]
        z = y.new_zeros(B, N); u = y.new_zeros(B, N); outs = []
        for t in range(self.T):
            beta = torch.sigmoid(self.rho_raw[t]); alpha = GOLD * torch.sigmoid(self.alpha_raw[t])
            v = z - u
            res = y - mv(Ac, v)
            x = v + beta * mv(Ap, res)
            w = x + u
            sig = torch.sqrt(torch.exp(self.m[t]) * res.abs().pow(2).sum(1) / N
                             + torch.exp(self.k[t]) * s2).view(B, 1)
            r = w.abs(); mag = self.shrink[t](r / sig) * sig
            zs = mag * w / r.clamp_min(1e-12)
            if self.denoiser == "conv":
                cnn = self.cnn[0] if len(self.cnn) == 1 else self.cnn[t]
                zs = zs + self.gamma[t] * sig * cnn(w / sig, zs / sig, torch.log(sig))
            z = zs
            u = u + alpha * (x - z)
            outs.append(z)
        return outs if return_all else z


class LAMP(nn.Module):
    """Borgerding LAMP (soft threshold, untied B_t init A^T, lambda_t = a_t ||v||/sqrt(M))."""
    def __init__(self, A, T=8):
        super().__init__()
        M, N = A.shape; self.M = M
        self.register_buffer("A", A)
        self.B = nn.ParameterList([nn.Parameter(A.t().clone()) for _ in range(T)])
        self.a = nn.Parameter(torch.full((T,), 1.1))

    def forward(self, y, A=None, s2=None):
        Ac = self.A.to(y.dtype); x = y.new_zeros(y.shape[0], self.A.shape[1]); v = y; bprev = 0
        for t, Bt in enumerate(self.B):
            r = x + v @ Bt.to(y.dtype).t()
            lam = self.a[t] * v.norm(dim=1, keepdim=True) / math.sqrt(self.M)
            m = r.abs(); x = r * F.relu(1 - lam / m.clamp_min(1e-12))
            b = (m > lam).float().sum(1, keepdim=True) / self.M
            v = y - x @ Ac.t() + b * v
        return x


def nparams(m): return sum(p.numel() for p in m.parameters())

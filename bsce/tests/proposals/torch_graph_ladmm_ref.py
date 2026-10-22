"""
Graph-LADMM in PyTorch: unrolled scaled ADMM on the whitened folded model with a MAGNETIC BEAM GRAPH prior.

    y = A (x + n),  W = (A A^T)^(-1/2),  y_w = W y = A_w x + n_w,  n_w ~ CN(0, s2 I)

Graph: beam ring (ULA) / 16x16 torus (UPA) with wrap edges. Edge e = (i -> j) along axis d carries a phase phi_c
(one per class c = (axis, interior|wrap)); (D x)_e = x_j - e^{j phi_c} x_i. phi_c comes from the leakage tails of
training channels (estimate_edge_phases), e.g. -180/n deg with an element-0 array reference, ~180 deg with a centred
one. It can also be trained (learn_phase=True).

Layer t (T layers, one network for all SNRs; thresholds scale with the known sigma):
    x   = (A_w^T A_w + rho_t (I + D^H D))^-1 (A_w^T y_w + rho_t (z1 - u1) + rho_t D^H (z2 - u2))      [xstep='analytic']
        = H_t (same right-hand side), H_t a free complex N x N matrix initialised at the analytic inverse  [xstep='free']
    w   = x + u1
    z1  = soft(w, k1_t sigma)                                   zstep in {'l1', 'gtv'}
        = w * sigmoid(b_t (sqrt(e_bar(w)) / sigma - tau_t))     zstep in {'gate', 'gtv_gate'}   (bias-free energy gate)
        = w * (1 - k1_t sigma / sqrt(e_bar(w)))_+              zstep == 'gsoft'                  (energy shrinkage)
    z2  = soft(D x + u2, k2_t sigma)                            zstep in {'gtv', 'gtv_gate'}     (graph TV on edges)
    u1 += e_t (x - z1);   u2 += e_t (D x - z2)
    e_bar_i = (|w_i|^2 + a1_t S1|w|^2 + a2_t S2|w|^2) / (1 + 2 a1_t + 2 a2_t)    S1/S2: sums over the two neighbours at
              offset 1 and 2 (ULA) or along axis 1 and axis 2 (UPA): phase-free pooled energy.
All scalars go through softplus. Output: z1 of the last layer. 'l1' (no graph) is the graph-free twin.
Checked against the JAX pilot (graph_unrolled_compare.py / gtv_unrolled_pilot.py): test_torch_graph_ladmm.py.
"""
import math

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


# ------------------------------------------------------------------ graph
def beam_edges(array, ring=True):
    """List of (i, j, cls) with cls = (axis, wrap). UPA index = 16 a + b (a: first kron factor, b: second)."""
    if array == "ULA":
        n = 256
        return [(i, (i + 1) % n, (0, i == n - 1)) for i in range(n if ring else n - 1)]
    nb = 16 if ring else 15
    e = [(16 * a + b, 16 * a + (b + 1) % 16, (1, b == 15)) for a in range(16) for b in range(nb)]
    e += [(16 * a + b, 16 * ((a + 1) % 16) + b, (0, a == 15)) for a in range(nb) for b in range(16)]
    return e


def estimate_edge_phases(X, array, min_wrap=20):
    """Edge phase per class from clean channels X (N x K complex numpy): circular mean of angle(x_j x_i^*) on the
    leakage tails (edges at offsets (+1,+2), (+2,+3), (-2,-1), (-3,-2) from each channel's strongest beam; the two
    edges bracketing the peak carry 0 or pi and are excluded). Returns {cls: (phase_rad, coherence, n_samples)}."""
    k = np.argmax(np.abs(X), 0)
    ball = np.arange(X.shape[1])
    n = 256 if array == "ULA" else 16
    acc = {}
    for dim in ([0] if array == "ULA" else [0, 1]):
        if array == "ULA":
            pos, idx = k, (lambda p_, b_: p_)
        elif dim == 1:
            pos, idx = k % 16, (lambda p_, b_: (k[b_] // 16) * 16 + p_)
        else:
            pos, idx = k // 16, (lambda p_, b_: p_ * 16 + k[b_] % 16)
        for o in (1, 2, -2, -3):
            pi_, pj_ = (pos + o) % n, (pos + o + 1) % n
            z = X[idx(pj_, ball), ball] * np.conj(X[idx(pi_, ball), ball])
            z = z / np.maximum(np.abs(z), 1e-300)
            for wv in (False, True):
                acc.setdefault((dim, wv), []).append(z[(pi_ == n - 1) == wv])
    out = {}
    for c, zs in acc.items():
        z = np.concatenate(zs)
        out[c] = (float(np.angle(np.mean(z))) if len(z) else 0.0, float(np.abs(np.mean(z))) if len(z) else 0.0, len(z))
    for (dim, wv) in list(out):
        if wv and out[(dim, True)][2] < min_wrap:                  # too few wrap samples: use the interior value
            out[(dim, True)] = (out[(dim, False)][0], out[(dim, False)][1], out[(dim, True)][2])
    return out


def whitening(A):
    ev, EV = np.linalg.eigh(A @ A.T)
    return EV @ np.diag(ev ** -0.5) @ EV.T


# ------------------------------------------------------------------ model
class GraphLADMM(nn.Module):
    PARAMS = {"l1": ("rho", "k1", "e"), "gtv": ("rho", "k1", "k2", "e"), "gsoft": ("rho", "k1", "e", "a1", "a2"),
              "gate": ("rho", "b", "tau", "e", "a1", "a2"), "gtv_gate": ("rho", "b", "tau", "k2", "e", "a1", "a2")}
    INIT = {"l1": dict(rho=1.0, k1=0.5, e=1.0), "gtv": dict(rho=1.0, k1=0.2, k2=0.5, e=1.0),
            "gsoft": dict(rho=1.0, k1=0.5, e=1.0, a1=0.5, a2=0.2),
            "gate": dict(rho=1.0, b=2.0, tau=1.5, e=1.0, a1=0.5, a2=0.2),
            "gtv_gate": dict(rho=1.0, b=2.0, tau=1.5, k2=0.5, e=1.0, a1=0.5, a2=0.2)}   # UPA: a2 = 0.5

    def __init__(self, A, array, T=16, zstep="gtv_gate", xstep="analytic", phases=None, learn_phase=False,
                 ring=True, init=None, dtype=torch.complex128):
        super().__init__()
        assert zstep in self.PARAMS and xstep in ("analytic", "free")
        self.array, self.T, self.zstep, self.xstep = array, T, zstep, xstep
        self.cdt = dtype
        self.rdt = torch.float64 if dtype == torch.complex128 else torch.float32
        A = np.asarray(A, float)
        self.M, self.N = A.shape
        W = whitening(A)
        Aw = W @ A
        self.register_buffer("W", torch.tensor(W, dtype=self.rdt))
        self.register_buffer("Aw", torch.tensor(Aw, dtype=self.rdt))
        self.register_buffer("P", torch.tensor(Aw.T @ Aw, dtype=self.rdt))
        E = beam_edges(array, ring)
        classes = sorted({c for _, _, c in E})
        self.classes = classes
        self.register_buffer("I", torch.tensor([e[0] for e in E]))
        self.register_buffer("J", torch.tensor([e[1] for e in E]))
        self.register_buffer("cls", torch.tensor([classes.index(e[2]) for e in E]))
        n = 256 if array == "ULA" else 16
        ph0 = [(phases or {}).get(c, (-math.pi / n,))[0] for c in classes]
        ph0 = torch.tensor(ph0, dtype=self.rdt)
        if learn_phase:
            self.phase = nn.Parameter(ph0)
        else:
            self.register_buffer("phase", ph0)
        init0 = dict(self.INIT[zstep])
        if array == "UPA" and "a2" in init0:
            init0["a2"] = 0.5
        init = {**init0, **(init or {})}
        isp = lambda v: math.log(math.expm1(v))
        self.raw = nn.ParameterDict({k: nn.Parameter(torch.full((T,), isp(init[k]), dtype=self.rdt))
                                     for k in self.PARAMS[zstep]})
        if xstep == "free":                                       # H_t initialised at the analytic inverse
            with torch.no_grad():
                H0 = torch.linalg.inv(self._xmat(F.softplus(self.raw["rho"][0])))
            self.H = nn.Parameter(H0.unsqueeze(0).repeat(T, 1, 1))

    # graph operators -------------------------------------------------
    def _w(self):
        return torch.exp(1j * self.phase.to(self.cdt))[self.cls]        # (E,)

    def Dop(self, x):                                                     # x (B, N) -> (B, E)
        return x[:, self.J] - self._w() * x[:, self.I]

    def DHop(self, v):                                                    # v (B, E) -> (B, N)
        out = torch.zeros(v.shape[0], self.N, dtype=v.dtype, device=v.device)
        out = out.index_add(1, self.J, v)
        return out.index_add(1, self.I, -torch.conj(self._w()) * v)

    def _xmat(self, rho):
        use_g = self.zstep in ("gtv", "gtv_gate")
        G = torch.eye(self.N, dtype=self.cdt, device=self.P.device)
        if use_g:
            D = torch.zeros(len(self.I), self.N, dtype=self.cdt, device=self.P.device)
            ar = torch.arange(len(self.I), device=self.P.device)
            D[ar, self.J] = 1
            D[ar, self.I] = -self._w()
            G = G + D.conj().T @ D
        return self.P.to(self.cdt) + rho * G

    def pooled_energy(self, w, a1, a2):
        e = w.abs() ** 2                                                  # (B, N) real
        if self.array == "ULA":
            s1 = torch.roll(e, 1, 1) + torch.roll(e, -1, 1)
            s2 = torch.roll(e, 2, 1) + torch.roll(e, -2, 1)
        else:
            g = e.reshape(-1, 16, 16)
            s1 = (torch.roll(g, 1, 1) + torch.roll(g, -1, 1)).reshape(-1, self.N)
            s2 = (torch.roll(g, 1, 2) + torch.roll(g, -1, 2)).reshape(-1, self.N)
        return (e + a1 * s1 + a2 * s2) / (1 + 2 * a1 + 2 * a2)

    @staticmethod
    def soft(v, t):
        mag = v.abs()
        return v * torch.clamp(1 - t / torch.clamp(mag, min=1e-30), min=0)

    def set_combiner(self, A):
        """Swap in another combiner A (M x N real): re-whitens; the analytic x-step follows it (combiner-agnostic)."""
        A = np.asarray(A.detach().real.cpu() if torch.is_tensor(A) else A, float)
        W = whitening(A)
        Aw = W @ A
        self.W.copy_(torch.tensor(W, dtype=self.rdt))
        self.Aw.copy_(torch.tensor(Aw, dtype=self.rdt))
        self.P.copy_(torch.tensor(Aw.T @ Aw, dtype=self.rdt))

    def forward(self, y, sigma, whitened=False, A=None):
        """y: (B, M) complex measurements (bsce convention y = A(x+n)); sigma: (B,) noise std. Returns x_hat (B, N).
        A: optional other combiner for this call (analytic x-step only; the model's own combiner is restored)."""
        if A is not None:
            saved = (self.W.clone(), self.Aw.clone(), self.P.clone())
            self.set_combiner(A)
            try:
                return self.forward(y, sigma, whitened)
            finally:
                self.W.copy_(saved[0]); self.Aw.copy_(saved[1]); self.P.copy_(saved[2])
        y = y.to(self.cdt)
        sig = sigma.to(self.rdt).unsqueeze(1)
        yw = y if whitened else y @ self.W.T.to(self.cdt)
        AtY = yw @ self.Aw.to(self.cdt)                                   # (B, N) = (A_w^T y_w)^T
        B = y.shape[0]
        z1 = torch.zeros(B, self.N, dtype=self.cdt, device=y.device)
        u1 = torch.zeros_like(z1)
        tv = self.zstep in ("gtv", "gtv_gate")
        if tv:
            z2 = torch.zeros(B, len(self.I), dtype=self.cdt, device=y.device)
            u2 = torch.zeros_like(z2)
        p = {k: F.softplus(v) for k, v in self.raw.items()}
        for t in range(self.T):
            rho = p["rho"][t]
            rhs = AtY + rho * (z1 - u1)
            if tv:
                rhs = rhs + rho * self.DHop(z2 - u2)
            if self.xstep == "free":
                x = rhs @ self.H[t].T
            elif tv:
                x = torch.linalg.solve(self._xmat(rho), rhs.T).T
            else:                                                         # (P + rho I)^-1, P a projection
                x = (rhs - (rhs @ self.P.to(self.cdt)) / (1 + rho)) / rho
            w = x + u1
            if self.zstep in ("l1", "gtv"):
                z1 = self.soft(w, p["k1"][t] * sig)
            elif self.zstep == "gsoft":
                eb = self.pooled_energy(w, p["a1"][t], p["a2"][t])
                z1 = w * torch.clamp(1 - p["k1"][t] * sig / torch.sqrt(eb + 1e-30), min=0)
            else:
                eb = self.pooled_energy(w, p["a1"][t], p["a2"][t])
                z1 = w * torch.sigmoid(p["b"][t] * (torch.sqrt(eb) / sig - p["tau"][t]))
            if tv:
                Dx = self.Dop(x)
                z2 = self.soft(Dx + u2, p["k2"][t] * sig)
                u2 = u2 + p["e"][t] * (Dx - z2)
            u1 = u1 + p["e"][t] * (x - z1)
        return z1

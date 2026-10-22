"""Graph-LADMM (v29): unrolled ADMM on the whitened folded model with a magnetic beam-graph prior.

    y = A (x + n),  W = (A A^T)^-1/2,  y_w = W y = A_w x + n_w,  n_w ~ CN(0, s2 I),  A^+ = A^T (A A^T)^-1 = A_w^T W

Graph: the N beams, edges between axis neighbours, with the wrap-around edge (ULA: ring of N beams; UPA: N1 x N2 torus,
4 neighbours, index = N2 a + b as bsce's kron(U_N1, U_N2)). Edge e = (i -> j) of class c = (axis, wrap) carries a phase
phi_c: (D x)_e = x_j - e^{j phi_c} x_i. phi_c is the phase step of a path's leakage between neighbouring beams; it is
estimated from the training channels (init_from_data, the circular mean on the leakage tails): about pi for the
centred array index of GM-LAMP's SV data, about -pi/n for an element-0 reference.

Layer t (scaled dual; "+ TV" terms only for the zsteps gtv, gtv_gate, gtv_gm):

    x_t  = H_t (A^+ y + rho_t (z1 - u1) + rho_t D^H (z2 - u2))
           xstep 'analytic': H_t = (P + rho_t (I + D^H D))^-1, P = A^T (A A^T)^-1 A   (any combiner: set_combiner)
           xstep 'free'    : H_t a real N x N matrix, initialised at that inverse (one combiner, like LADMM-F)
    w    = x_t + u1
    z1   = l1      : soft(w, k1_t s)                         s = noise level (option sigma)
           gm      : GM-MMSE(w, (c_t s)^2)                   K-component complex Gaussian mixture, learned per layer
           gate    : w * sigmoid(b_t (sqrt(e_bar(w)) / s - tau_t))       bias-free energy gate on the graph
           gsoft   : w * (1 - k1_t s / sqrt(e_bar(w)))_+                energy shrinkage (the gsoft patch)
           gtv     : soft(w, k1_t s)              + TV
           gtv_gate: the gate                     + TV
           gtv_gm  : GM-MMSE                      + TV
           gm_graph: graph-coupled GM-MMSE: the mixture weights of beam i are softmax_k(l_k + beta_k g_i),
                     g_i = log(1 + e_nb_i / (S_nb s^2)), e_nb = the pooled energy of the neighbours only (E16 "PC-GM")
           gtv_gm_graph: gm_graph + TV
           rwl1    : graph-reweighted l1 (v31; the unrolled form of the uploaded graph_reweighted_l1 package):
                     soft(w, k1_t s om_i), om from the previous estimate z1_{t-1}:
                     a_i = sqrt(e_bar(z1_{t-1}))_i / max_j (...)_j, om_i = 1 / (a_i + eps_t), om <- om / mean(om)
                     (per channel; layer 1: z1 = 0, om = 1, plain l1). a1 = a2 = 1 at init = the package's equal
                     closed-neighbourhood mean (beam + 4 torus neighbours on the UPA, + 2 and +-2 on the ULA)
           gtv_rwl1: rwl1                         + TV
    z2   = soft(D x_t + u2, k2_t s)                          graph total variation on the magnetic edges
    u1  += e_t (x_t - z1),   u2 += e_t (D x_t - z2)
    output z1 of the last layer
v31: no parameter that cannot reach the output is created, so every stored parameter is trained (_dead_names):
e (and k2 with TV) of the last layer (their updates would feed a layer T + 1; the last layer stops at its z1), eps, a1,
a2 of layer 1 of the reweighted z-steps (z1 = 0 there: weights 1), rho of layer 1 with a free H (it multiplies
z - u = 0), and the pooling weights without graph pooling. Older checkpoints load (those entries are ignored; they never
changed the output). gladmm T = 8: 54 parameters (56 stored before v31, of which 54 were trained).
Inference (no gradient, TV analytic x-step): H_t = (P + rho_t (I + D^H D))^-1 is computed once per combiner and per set of
rho_t (cache_inference, default on) instead of 8 LU factorisations in every call; I + D^H D is built from the edges.
Beam prior (option beam_prior='gate', gate z-steps): the gate logit gets + beta_t l_i, l_i the log of beam i's mean
power relative to the mean (init_from_data, clipped at log 1e-3), learned and shared by the layers, beta_t per layer
(init 0: the plain network at init).
Data step (option data_step, v31): 'whitened' (default) as above; 'plain' the package's unwhitened least squares
1/2 ||y - A x||^2: A^+ y -> A^T y and P -> A^T A, i.e. x_t = (A^T A + rho_t (I [+ D^H D]))^-1 (A^T y + ...), through
the thin SVD of A without TV. The noise level s is the whitened residual in both.
    e_bar_i = (|w_i|^2 + a1_t S1|w|^2 + a2_t S2|w|^2) / (1 + 2 a1_t + 2 a2_t): pooled neighbour energy, phase-free
              (S1, S2: the two neighbours at offset 1 and 2 on the ULA; on the UPA, S1 along axis 0 (slow index a) and S2
              along axis 1 (fast index b), as in beam_edges)

Noise level s (option sigma): 'residual' (default) s^2 = ||y_w - A_w z1_{t-1}||^2 / M per channel (layer 1: ||y_w||^2/M);
'blind' s = median_i |w_i| / sqrt(ln 2); 'known' s^2 = the true noise variance (needs_noise_var, as NF-LADMM).
Every scalar is softplus(lr_mult * raw): with Adam the scalars then move lr_mult times faster than H_t.
With zstep 'l1', xstep 'analytic' and tied parameters the layer is ADMM for the fused LASSO
1/2 ||y_w - A_w x||^2 + lam ||x||_1 + mu ||D x||_1 (convergent); the gate and GM are learned, non-proximal steps.
Port of torch_graph_ladmm.py (graph-choice study, 16 Oct 2026) to bsce's conventions (columns, stages, export).
"""
from __future__ import annotations

import math
from typing import Dict, List, Optional

import numpy as np
import torch
import torch.nn.functional as F

from .common import Stage, UnrolledNet, rmm
from .shrinkage import LN2, gm_mmse_shrink, gm_mmse_shrink_beam

ZSTEPS = ("l1", "gm", "gate", "gsoft", "gtv", "gtv_gate", "gtv_gm", "gm_graph", "gtv_gm_graph", "rwl1", "gtv_rwl1")
TV = ("gtv", "gtv_gate", "gtv_gm", "gtv_gm_graph", "gtv_rwl1")
RW = ("rwl1", "gtv_rwl1")
GM = ("gm", "gtv_gm", "gm_graph", "gtv_gm_graph")
ZPARAMS = {"l1": ("k1",), "gm": ("c",), "gate": ("b", "tau", "a1", "a2"), "gsoft": ("k1", "a1", "a2"),
           "gtv": ("k1", "k2"), "gtv_gate": ("b", "tau", "a1", "a2", "k2"), "gtv_gm": ("c", "k2"),
           "gm_graph": ("c", "a1", "a2"), "gtv_gm_graph": ("c", "a1", "a2", "k2"),
           "rwl1": ("k1", "eps", "a1", "a2"), "gtv_rwl1": ("k1", "eps", "a1", "a2", "k2")}
INIT = dict(rho=1.0, e=1.0, k1=0.5, k2=0.5, c=1.0, b=2.0, tau=1.5, a1=0.5, a2=0.2)
RW_INIT = dict(k1=1.0, eps=0.2, a1=1.0, a2=1.0)       # rwl1: equal neighbourhood mean, eps of the package
DATA_STEPS = ("whitened", "plain")
SIGMAS = ("residual", "blind", "known")


def beam_edges(array: str, N1: int, N2: int, ring: bool = True):
    """[(i, j, (axis, wrap))]. ULA: N = N1 N2 beams on a ring. UPA: index N2 a + b; axis 1 = b (fast), axis 0 = a."""
    if array == "ULA":
        n = N1 * N2
        return [(i, (i + 1) % n, (0, i == n - 1)) for i in range(n if ring else n - 1)]
    e = [(N2 * a + b, N2 * a + (b + 1) % N2, (1, b == N2 - 1)) for a in range(N1) for b in range(N2 if ring else N2 - 1)]
    e += [(N2 * a + b, N2 * ((a + 1) % N1) + b, (0, a == N1 - 1)) for a in range(N1 if ring else N1 - 1) for b in range(N2)]
    return e


def estimate_edge_phases(X: np.ndarray, array: str, N1: int, N2: int, min_wrap: int = 20) -> Dict:
    """Edge phase per class from clean channels X (N x K complex): circular mean of angle(x_j conj(x_i)) on the leakage
    tails (offsets +1, +2, -2, -3 from each channel's strongest beam; the two edges around the peak are left out).
    Returns {class: (phase_rad, coherence, n_samples)}; a wrap class with too few samples takes the interior value."""
    X = np.asarray(X)
    k = np.argmax(np.abs(X), 0)
    cols = np.arange(X.shape[1])
    acc = {}
    dims = [0] if array == "ULA" else [0, 1]
    for dim in dims:
        if array == "ULA":
            n, pos = N1 * N2, k
            idx = lambda p_: p_                                            # noqa: E731
        elif dim == 1:
            n, pos = N2, k % N2
            idx = lambda p_: (k // N2) * N2 + p_                           # noqa: E731
        else:
            n, pos = N1, k // N2
            idx = lambda p_: p_ * N2 + k % N2                              # noqa: E731
        for o in (1, 2, -2, -3):
            pi_, pj_ = (pos + o) % n, (pos + o + 1) % n
            z = X[idx(pj_), cols] * np.conj(X[idx(pi_), cols])
            z = z / np.maximum(np.abs(z), 1e-300)
            for wv in (False, True):
                acc.setdefault((dim, wv), []).append(z[(pi_ == n - 1) == wv])
    out = {}
    for c, zs in acc.items():
        z = np.concatenate(zs)
        out[c] = (float(np.angle(np.mean(z))) if len(z) else 0.0, float(np.abs(np.mean(z))) if len(z) else 0.0, len(z))
    for (dim, wv) in list(out):
        if wv and out[(dim, True)][2] < min_wrap:
            out[(dim, True)] = (out[(dim, False)][0], out[(dim, False)][1], out[(dim, True)][2])
    return out


def _isp(v: float) -> float:
    return float(math.log(math.expm1(v)))


class GraphLADMM(UnrolledNet):
    family = "gladmm"

    def __init__(self, A, T: int = 8, zstep: str = "gtv_gate", xstep: str = "analytic", array: str = "ULA",
                 N1: int = 16, N2: int = 16, sigma: str = "residual", ring: bool = True, edge_phase="data",
                 learn_phase: bool = False, gm_K: int = 4, lr_mult: float = 10.0, graph_pool: bool = True,
                 data_step: str = "whitened", beam_prior: str = "none", cache_inference: bool = True,
                 precision: str = "float32", device=None):
        super().__init__(A, precision)
        assert zstep in ZSTEPS, zstep
        assert xstep in ("analytic", "free"), xstep
        assert sigma in SIGMAS, sigma
        if data_step not in DATA_STEPS:
            raise ValueError(f"data_step {data_step!r}: one of {DATA_STEPS}")
        if data_step == "plain" and xstep != "analytic":
            raise ValueError("data_step=plain needs xstep=analytic (a free H_t has no data step to change)")
        self.data_step = data_step
        if beam_prior not in ("none", "gate"):
            raise ValueError(f"beam_prior {beam_prior!r}: 'none' or 'gate'")
        if beam_prior == "gate" and zstep not in ("gate", "gtv_gate"):
            raise ValueError("beam_prior=gate needs a gate z-step (gate, gtv_gate)")
        self.beam_prior = beam_prior
        self.cache_inference = bool(cache_inference)       # not part of cfg: it does not change the network
        self._hcache = None
        self._comb_ver = 0
        self.T, self.zstep, self.xstep, self.array = int(T), zstep, xstep, array.upper()
        self.N1, self.N2, self.sigma, self.ring = int(N1), int(N2), sigma, bool(ring)
        self.edge_phase, self.learn_phase, self.gm_K, self.lr_mult = edge_phase, bool(learn_phase), int(gm_K), float(lr_mult)
        self.graph_pool = bool(graph_pool)          # False (ablation): the gate / gsoft / gm_graph see only the beam itself
        self.tv = zstep in TV
        self.needs_noise_var = sigma == "known"
        if self.N1 * self.N2 != self.N:
            raise ValueError(f"N1 x N2 = {self.N1 * self.N2} but the combiner has N = {self.N} columns")
        self.cfg = dict(model="Graph-LADMM", T=self.T, zstep=zstep, xstep=xstep, array=self.array, N1=self.N1,
                        N2=self.N2, sigma=sigma, ring=self.ring, edge_phase=edge_phase, learn_phase=self.learn_phase,
                        gm_K=self.gm_K, lr_mult=self.lr_mult, precision=precision)
        if not self.graph_pool:
            self.cfg["graph_pool"] = False
        if data_step != "whitened":
            self.cfg["data_step"] = data_step          # v31; absent = whitened (the cfg of older networks unchanged)
        if beam_prior != "none":
            self.cfg["beam_prior"] = beam_prior
        self._set_buffers(self.A_np, register=True)
        E = beam_edges(self.array, self.N1, self.N2, self.ring)
        self.classes = sorted({c for _, _, c in E})
        self.register_buffer("I_e", torch.tensor([e[0] for e in E], dtype=torch.long))
        self.register_buffer("J_e", torch.tensor([e[1] for e in E], dtype=torch.long))
        self.register_buffer("cls_e", torch.tensor([self.classes.index(e[2]) for e in E], dtype=torch.long))
        ph0 = np.array([self._phase_default(ax) for ax, _ in self.classes], dtype=float)
        if self.learn_phase:
            self._p("phase", ph0)
        else:
            self.register_buffer("phase", torch.tensor(ph0, dtype=self.rdtype))
        m = self.lr_mult
        dead = self._dead_names()
        if self.beam_prior == "gate":
            self._p("beam_ell", np.zeros(self.N))                              # init_from_data: the power profile
        for t in range(self.T):
            if self.xstep == "free":
                self._p(f"H_{t}", np.zeros((self.N, self.N)))
            if f"rho_raw_{t}" not in dead:
                self._p(f"rho_raw_{t}", _isp(INIT["rho"]) / m)
            for k in ZPARAMS[zstep]:
                if f"{k}_raw_{t}" in dead:
                    continue
                if zstep in RW and k in RW_INIT:
                    v = RW_INIT[k]
                else:
                    v = INIT[k] if not (k == "a2" and self.array == "UPA") else 0.5
                self._p(f"{k}_raw_{t}", _isp(v) / m)
            if zstep in GM:
                from .ladmm import gm_init
                lw, ln = gm_init(self.gm_K)
                self._p(f"gm_logit_{t}", lw)
                self._p(f"gm_lognu_{t}", ln)
                if zstep in ("gm_graph", "gtv_gm_graph") and f"gm_beta_{t}" not in dead:
                    self._p(f"gm_beta_{t}", np.zeros(self.gm_K))          # 0: the plain GM at init
            if self.beam_prior == "gate":
                self._p(f"beam_beta_{t}", 0.0)                                 # 0: the plain gate at init
            if f"e_raw_{t}" not in dead:
                self._p(f"e_raw_{t}", _isp(INIT["e"]) / m)
        if self.xstep == "free":
            self._init_H()
        if device is not None:
            self.to(device)

    def _dead_names(self) -> set:
        """v31: the parameters that cannot reach the output and are therefore not created (see the module doc)."""
        T, zp = self.T, ZPARAMS[self.zstep]
        d = {f"e_raw_{T - 1}"}                             # the last dual step would feed a layer T + 1
        if self.tv:
            d.add(f"k2_raw_{T - 1}")                       # so would the last z2
        if self.zstep in RW:
            d |= {f"{k}_raw_0" for k in ("eps", "a1", "a2")}  # layer 1: z1 = 0, weights 1
        if self.xstep == "free":
            d.add("rho_raw_0")                             # layer 1: rho multiplies z - u = 0
        if not self.graph_pool:
            d |= {f"{k}_raw_{t}" for t in range(T) for k in ("a1", "a2") if k in zp}
            if self.zstep in ("gm_graph", "gtv_gm_graph"):
                d |= {f"gm_beta_{t}" for t in range(T)}
        return d

    def _load_from_state_dict(self, state_dict, prefix, local_metadata, strict, missing_keys, unexpected_keys,
                              error_msgs):
        for n in self._dead_names():                       # checkpoints before v31 store them (never trained)
            state_dict.pop(prefix + n, None)
        self._hcache = None
        super()._load_from_state_dict(state_dict, prefix, local_metadata, strict, missing_keys, unexpected_keys,
                                      error_msgs)

    # -- combiner -------------------------------------------------------------------------------------------------------
    def _set_buffers(self, A: np.ndarray, register: bool = False):
        A = np.asarray(A, np.float64)
        ev, U = np.linalg.eigh(A @ A.T)
        W = (U / np.sqrt(ev)) @ U.T
        Apinv = A.T @ (U / ev) @ U.T
        P = Apinv @ A
        P = 0.5 * (P + P.T)
        vals = dict(Wm=W, Aw=W @ A, Apinv=Apinv, Pm=P)
        if getattr(self, "data_step", "whitened") == "plain":          # v31: 1/2 ||y - A x||^2, unwhitened
            _, sv, Vt = np.linalg.svd(A, full_matrices=False)
            AtA = A.T @ A
            vals.update(Apinv=A.T.copy(), Pm=0.5 * (AtA + AtA.T), Vr=Vt.T.copy(), sv2=sv ** 2)
        for k, v in vals.items():
            if register:
                self.register_buffer(k, torch.tensor(v, dtype=self.rdtype))
            else:
                self._put_buffer(k, v)                  # v30: a new M registers the buffer again
        self._comb_ver = getattr(self, "_comb_ver", 0) + 1
        self._hcache = None

    def set_combiner(self, A):
        """Use another combiner A (M' x N): the analytic x-step follows it (combiner-agnostic). v30: any number of RF
        chains M' (no learned parameter depends on M; W, A_w, A^+ and P are recomputed, the noise level is the RMS of
        the M'-dim residual). A 'free' H_t was learned for one combiner, so a free network keeps its own (ValueError)."""
        if self.xstep == "free":
            raise ValueError("Graph-LADMM with xstep=free has H_t learned for one combiner")
        A = np.asarray(A.detach().cpu().numpy() if torch.is_tensor(A) else A, np.float64)
        if A.ndim != 2 or A.shape[1] != self.N:
            raise ValueError(f"combiner of shape {A.shape}: it must have N = {self.N} columns (beams)")
        if np.linalg.matrix_rank(A) < A.shape[0]:
            raise ValueError(f"combiner of shape {A.shape} has dependent rows (A A^T is singular)")
        self.M, self.A_np = A.shape[0], A
        self._put_buffer("A", A)
        self._set_buffers(A)

    # -- graph ----------------------------------------------------------------------------------------------------------
    def _phase_default(self, axis: int = 1) -> float:
        ep = self.edge_phase
        if ep in ("blind", None):
            return 0.0
        if isinstance(ep, (int, float)) and not isinstance(ep, bool):
            return math.radians(float(ep))
        n = self.N if self.array == "ULA" else (self.N1 if axis == 0 else self.N2)
        return -math.pi / n                                    # 'theory' (element-0 reference); 'data' until estimated

    def init_from_data(self, X):
        """Edge phases from clean training channels X (N x K) when edge_phase='data'; then H_0 (xstep free); the beam
        prior's power profile when beam_prior='gate'. Returns {class: (degrees, coherence, samples)}."""
        if self.beam_prior == "gate":
            pw = np.mean(np.abs(np.asarray(X)) ** 2, axis=1)
            ell = np.log(np.maximum(pw / max(float(pw.mean()), 1e-300), 1e-3))
            with torch.no_grad():
                self.P("beam_ell").copy_(torch.tensor(ell, dtype=self.rdtype))
        if self.edge_phase != "data":
            return {}
        est = estimate_edge_phases(np.asarray(X), self.array, self.N1, self.N2)
        with torch.no_grad():
            self.phase.copy_(torch.tensor([est[c][0] for c in self.classes], dtype=self.rdtype))
        if self.xstep == "free":
            self._init_H()
        return {f"axis{c[0]}_{'wrap' if c[1] else 'interior'}": (round(math.degrees(v[0]), 3), round(v[1], 4), v[2])
                for c, v in est.items()}

    def _wedge(self):
        return torch.exp(1j * self.phase.to(self.cdtype))[self.cls_e]           # (E,)

    def Dop(self, x):                                                            # (N, B) -> (E, B)
        return x[self.J_e] - self._wedge().unsqueeze(1) * x[self.I_e]

    def DHop(self, v):                                                           # (E, B) -> (N, B)
        out = torch.zeros(self.N, v.shape[1], dtype=v.dtype, device=v.device)
        out = out.index_add(0, self.J_e, v)
        return out.index_add(0, self.I_e, -torch.conj(self._wedge()).unsqueeze(1) * v)

    def _G(self):
        """I + D^H D (N x N, complex) for the TV z-steps. v31: built from the edges, O(E): row e of D is
        x_j - w_e x_i, so D^H D gets 1 at (j, j), |w_e|^2 at (i, i), -w_e at (j, i) and -conj(w_e) at (i, j)."""
        w = self._wedge()
        G = torch.eye(self.N, dtype=self.cdtype, device=self.A.device)
        rows = torch.cat([self.J_e, self.I_e, self.J_e, self.I_e])
        cols = torch.cat([self.J_e, self.I_e, self.I_e, self.J_e])
        vals = torch.cat([torch.ones_like(w), w * w.conj(), -w, -w.conj()])
        return G.index_put((rows, cols), vals, accumulate=True)

    def _cached_H(self):
        """v31, inference: H_t = (P + rho_t G)^-1 for every layer, computed in complex128 once per combiner, edge
        phases and rho_t (the key), then reused. Same output as the per-call solve to the arithmetic precision."""
        rhos = tuple(float(self.s("rho", t)) for t in range(self.T))
        key = (self._comb_ver, str(self.A.device), str(self.cdtype), rhos, self.phase.detach().cpu().numpy().tobytes())
        if self._hcache is not None and self._hcache[0] == key:
            return self._hcache[1]
        P = self.Pm.to(torch.complex128)
        G = self._G().to(torch.complex128)
        H = [torch.linalg.inv(P + r * G).to(self.cdtype) for r in rhos]
        self._hcache = (key, H)
        return H

    def clear_cache(self):
        self._hcache = None

    def _init_H(self):
        with torch.no_grad():
            P = self.Pm.to(self.cdtype)
            G = self._G() if self.tv else torch.eye(self.N, dtype=self.cdtype, device=self.A.device)
            H0 = torch.linalg.inv(P + G).real                                    # rho_0 = 1
            for t in range(self.T):
                self.P(f"H_{t}").copy_(H0)

    def pooled_energy(self, w, t: int):
        """e_bar of layer t: (|w_i|^2 + a1_t S1|w|^2 + a2_t S2|w|^2) / (1 + 2 a1_t + 2 a2_t); |w|^2 without pooling."""
        e = w.real ** 2 + w.imag ** 2                                            # (N, B)
        if not self.graph_pool:
            return e
        a1, a2 = self.s("a1", t), self.s("a2", t)
        if self.array == "ULA":
            s1 = torch.roll(e, 1, 0) + torch.roll(e, -1, 0)
            s2 = torch.roll(e, 2, 0) + torch.roll(e, -2, 0)
        else:
            g = e.reshape(self.N1, self.N2, -1)
            s1 = (torch.roll(g, 1, 0) + torch.roll(g, -1, 0)).reshape(self.N, -1)
            s2 = (torch.roll(g, 1, 1) + torch.roll(g, -1, 1)).reshape(self.N, -1)
        return (e + a1 * s1 + a2 * s2) / (1 + 2 * a1 + 2 * a2)

    def rw_weights(self, z, t: int):
        """Graph-reweighted l1 weights of layer t from an estimate z (N, B): a = sqrt(e_bar(z)) / max a,
        om = 1 / (a + eps_t), normalised to mean 1 per channel (the package's graph_weights with a1 = a2 = 1)."""
        amp = torch.sqrt(self.pooled_energy(z, t) + 1e-30)
        amp = amp / torch.clamp(amp.max(0, keepdim=True).values, min=1e-12)
        om = 1.0 / (amp + self.s("eps", t))
        return om / om.mean(0, keepdim=True)          # layer 1 (t = 0) is never asked: its weights are 1

    @staticmethod
    def soft(v, t):
        return v * torch.clamp(1 - t / torch.clamp(torch.abs(v), min=1e-30), min=0).to(v.dtype)

    # -- parameters -----------------------------------------------------------------------------------------------------
    def s(self, name: str, t: int):
        """A scalar of layer t: softplus(lr_mult * raw)."""
        return F.softplus(self.lr_mult * self.P(f"{name}_raw_{t}"))

    def _layer_names(self, t: int):
        have = set(self.param_names())
        x = ([f"H_{t}"] if self.xstep == "free" else []) + [f"rho_raw_{t}"]
        z = [f"{k}_raw_{t}" for k in ZPARAMS[self.zstep]]
        if self.zstep in GM:
            z += [f"gm_logit_{t}", f"gm_lognu_{t}"]
        if self.zstep in ("gm_graph", "gtv_gm_graph"):
            z += [f"gm_beta_{t}"]
        if self.beam_prior == "gate":
            z += (["beam_ell"] if t == 0 else []) + [f"beam_beta_{t}"]
        keep = lambda names: [n for n in names if n in have]                   # noqa: E731
        return keep(x), keep(z), keep([f"e_raw_{t}"])

    def stage_plan(self) -> List[Stage]:
        stages, allv = [], (["phase"] if self.learn_phase else [])
        for t in range(self.T):
            x, z, e = self._layer_names(t)
            allv += x
            stages.append(Stage(f"Graph-LADMM x-update T={t + 1}", "linear", list(x), list(allv)))
            allv += z
            stages.append(Stage(f"Graph-LADMM z-update T={t + 1}", "nonlinear", list(z), list(allv)))
            allv += e
        for i, st in enumerate(stages):
            st.index = i
        return stages

    # -- forward --------------------------------------------------------------------------------------------------------
    def forward_stages(self, y, upto: Optional[int] = None, s2=None):
        y = self.as_complex(y)
        n_st = 2 * self.T if upto is None else upto + 1
        B = y.shape[1]
        b = rmm(self.Apinv, y)                                                  # A^+ y = A_w^T y_w
        yw = rmm(self.Wm, y) if self.sigma == "residual" else None
        if self.sigma == "known":
            if s2 is None:
                raise ValueError("Graph-LADMM with sigma=known needs the noise variance of every column (s2)")
            sk = torch.sqrt(torch.as_tensor(np.asarray(s2, float) if not torch.is_tensor(s2) else s2,
                                            dtype=self.rdtype, device=y.device)).reshape(1, -1)
        z1 = torch.zeros((self.N, B), dtype=self.cdtype, device=y.device)
        u1 = torch.zeros_like(z1)
        if self.tv:
            n_e = self.I_e.shape[0]
            z2 = torch.zeros((n_e, B), dtype=self.cdtype, device=y.device)
            u2 = torch.zeros_like(z2)
        Hc = (self._cached_H() if self.tv and self.xstep == "analytic" and self.cache_inference
              and not torch.is_grad_enabled() else None)                         # v31: inference cache
        if self.tv:
            G = self._G() if (self.xstep == "analytic" and Hc is None) else None
        P = self.Pm.to(self.cdtype)
        outs = []
        for t in range(self.T):
            rho = self.s("rho", t) if (self.xstep == "analytic" or t > 0) else None
            rc = None if rho is None else rho.to(self.cdtype)
            if t == 0:
                rhs = b                                                         # z1 = u1 = z2 = u2 = 0
            else:
                rhs = b + rc * (z1 - u1)
                if self.tv:
                    rhs = rhs + rc * self.DHop(z2 - u2)
            if self.xstep == "free":
                x = rmm(self.P(f"H_{t}"), rhs)
            elif self.tv:
                x = Hc[t] @ rhs if Hc is not None else torch.linalg.solve(P + rc * G, rhs)
            elif self.data_step == "plain":                                     # (A^T A + rho I)^-1, thin SVD of A
                g = (1.0 / (self.sv2 + rho) - 1.0 / rho).to(self.cdtype).unsqueeze(1)
                x = rhs / rc + rmm(self.Vr, g * rmm(self.Vr.T, rhs))
            else:                                                               # (P + rho I)^-1, P a projection
                x = (rhs - (P @ rhs) / (1 + rc)) / rc
            outs.append(x)
            if len(outs) >= n_st:
                return outs
            w = x + u1
            if self.sigma == "known":
                sg = sk
            elif self.sigma == "blind":
                sg = torch.median(torch.abs(w), dim=0, keepdim=True).values / math.sqrt(LN2)
            else:
                r = yw - rmm(self.Aw, z1)
                sg = torch.sqrt(torch.mean(r.real ** 2 + r.imag ** 2, dim=0, keepdim=True) + 1e-30)
            zs = self.zstep
            if zs in ("l1", "gtv"):
                z1 = self.soft(w, self.s("k1", t) * sg)
            elif zs in ("gm", "gtv_gm"):
                z1 = gm_mmse_shrink(w, (self.s("c", t) * sg) ** 2, self.P(f"gm_logit_{t}"), self.P(f"gm_lognu_{t}"))
            elif zs in ("gm_graph", "gtv_gm_graph"):
                logits = self.P(f"gm_logit_{t}").reshape(-1, 1, 1)
                if self.graph_pool:                                              # without pooling g = 0
                    a1, a2 = self.s("a1", t), self.s("a2", t)
                    enb = (1 + 2 * a1 + 2 * a2) * self.pooled_energy(w, t) - (w.real ** 2 + w.imag ** 2)
                    g = torch.log1p(enb / ((2 * a1 + 2 * a2) * sg ** 2 + 1e-30))                  # (N, B)
                    logits = logits + self.P(f"gm_beta_{t}").reshape(-1, 1, 1) * g.unsqueeze(0)
                z1 = gm_mmse_shrink_beam(w, (self.s("c", t) * sg) ** 2, logits, self.P(f"gm_lognu_{t}"))
            elif zs in RW:                                                      # v31: graph-reweighted l1
                th = self.s("k1", t) * sg                                       # layer 1: z1 = 0, weights 1
                z1 = self.soft(w, th if t == 0 else th * self.rw_weights(z1, t))  # weights from z1_{t-1}
            elif zs == "gsoft":
                eb = self.pooled_energy(w, t)
                z1 = w * torch.clamp(1 - self.s("k1", t) * sg / torch.sqrt(eb + 1e-30), min=0).to(w.dtype)
            else:                                                               # gate, gtv_gate
                eb = self.pooled_energy(w, t)
                logit = self.s("b", t) * (torch.sqrt(eb) / sg - self.s("tau", t))
                if self.beam_prior == "gate":
                    logit = logit + self.P(f"beam_beta_{t}") * self.P("beam_ell").unsqueeze(1)
                z1 = w * torch.sigmoid(logit).to(w.dtype)
            if t == self.T - 1:                                                 # v31: the rest of the layer would
                outs.append(z1)                                                 # only feed a layer T + 1
                return outs
            ec = self.s("e", t).to(self.cdtype)
            if self.tv:
                Dx = self.Dop(x)
                z2 = self.soft(Dx + u2, self.s("k2", t) * sg)
                u2 = u2 + ec * (Dx - z2)
            u1 = u1 + ec * (x - z1)
            outs.append(z1)
            if len(outs) >= n_st:
                return outs
        return outs

    def forward(self, y, s2=None):
        return self.forward_stages(y, s2=s2)[-1]

    def mat_dict(self):
        d = {n: self.P(n).detach().cpu().numpy() for n in self.param_names()}
        with torch.no_grad():
            have = set(self.param_names())
            for t in range(self.T):
                for k in ["rho", "e"] + list(ZPARAMS[self.zstep]):
                    if f"{k}_raw_{t}" in have:
                        d[f"{k}_{t}"] = np.array(float(self.s(k, t)))
            d["edge_phase_rad"] = self.phase.detach().cpu().numpy()
        return d

    def load_mat_dict(self, d):
        raise ValueError("a Graph-LADMM network is loaded from its .pt file")

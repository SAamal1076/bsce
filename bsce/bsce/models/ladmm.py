"""Learned ADMM (LADMM) for beamspace channel estimation.

Faithful ports of the LADMM TF training scripts (operator = 'full' | 'diag' | 'lowrank' |
'kron') and of LADMM_test.m (MATLAB inference). Layer t:

    x_t = H_t (A^T y + rho_t z_{t-1} - u_{t-1})          (H-update, learned operator H_t)
    z_t = S_{theta_t / rho_t}(x_t + u_{t-1} / rho_t)      (complex soft threshold)
    u_t = u_{t-1} + alpha_t rho_t (x_t - z_t)             (dual update)
    output z_T

Operators (names of the TF variables are kept):
  full    : H_t = H_matrix_t (N x N), init inv(A^T A + I)       [LADMM_train.py, A_LADMM_*]
  diag    : H_t = diag(d_t) W, W = inv(A^T A + I) fixed           [A_*_DiagonalScaling.py]
  lowrank : H_t = W + P_t Q_t^T (rank r), P,Q from SVD of W x 0.01 [B_*_LowRank.py]
  kron    : H_t x computed exactly like kronecker_matmul() of C_*_Kronecker.py, i.e.
            reshape(x,[N2,N1,B]) -> H2 . X . H1^T -> reshape, which is (H2 kron H1) x for
            row-major vec (the script's docstring says H1 kron H2; the code is reproduced).

Proposed variants (not in the original code), keeping exact ADMM semantics:
  psd       : H_t = (G_t + rho_t I)^-1, G_t = L_t L_t^T, L_t (N x N) init [A^T | 0]   (G=A^T A)
  kronprior : H_t = (A^T A + rho_t I + (L1 L1^T) kron (L2 L2^T))^-1   (UPA prior, 2*N1^2 params)
  diagprior : H_t = (A^T A + rho_t I + diag(softplus(d_t)))^-1
  For these, rho_t = softplus(rho_raw_t) > 0 and theta_t = softplus(theta_raw_t) >= 0.

Threshold (option, not in the original code; default 'scalar' = the scripts):
  scalar   : Z = S_{theta_t/rho_t}(V)                    one threshold per layer
  group    : Z_i = S_{(theta_t/rho_t) w_i}(V_i),  w_i = 1 / (1 + kappa g_i(V)),
             g_i = sqrt(local mean of |V|^2 in a window around beam i) / mean_i(g)
             (ADMM-Improved group threshold; window 3 x 3 on the 16 x 16 UPA beam grid, 3 beams for
             the ULA, circular); kappa = 3 fixed, uniform window -> no extra parameters
  group_k  : kappa_t learned per layer (softplus, init 3)                    +1 parameter / layer
  group_kc : kappa_t and the window weights c_t learned (softplus, normalised) +1+win^2 (ULA 1+win)
  adaptive : Z = S_{c_t sigma_hat}(V), sigma_hat = median_i |V_i| / sqrt(ln 2) for every column (blind noise
             level: with a sparse channel most entries of V are noise); c_t = softplus(theta_raw_t), init 1
             (v28; the noise-adaptive threshold of LAMP / TISTA)              no extra parameter
  firm     : firm (MCP) shrinkage: 0 below t1 = c_t sigma_hat, a ramp, identity above t2 = t1 (1 + softplus(k_t)),
             so the strong beams are not biased (init t2 = 2 t1)            +1 parameter / layer
  gm       : MMSE shrinkage of a gm_K-component complex Gaussian mixture (exact density exp(-|v|^2/s)/(pi s),
             zero means) in noise of variance (c_t sigma_hat)^2; weights and variances learned per layer, init
             w = .80/.12/.05/.03, nu = 1e-3/.5/5/50 (channels of mean power 1, as every bsce data set)
                                                                             +2 gm_K parameters / layer
  With adaptive / firm / gm, theta_t is a softplus scale for every operator (it cannot become negative).
  These three run in Python only (LADMM_test.m knows the soft threshold).

Data step (option, v28; default 'plain' = the scripts):
  plain    : x_t = H_t (A^T y + rho_t z - u)    the exact ADMM step for white noise y = A x + w
  whitened : x_t = H_t (A^+ y + rho_t z - u)    the exact step for the folded noise y = A (x + n) (GM-LAMP's):
             with P = A^T (A A^T)^-1 A it is (P + rho_t I)^-1 (A^+ y + rho_t z - u). Every A^T A of the operators
             becomes P: init of full / diag / lowrank / kron (P + I)^-1, psd L_0 = [A^T (A A^T)^-1/2 | 0],
             kronprior / diagprior / analytic use P. No extra parameter. A free H cannot reproduce this step from
             A^T y (12-40 % step error on SV ULA), so the option changes the model class.

Analytic operators (v28; combiner-agnostic, no matrix is learned):
  analytic  : H_t = (G + rho_t I)^-1, G = A^T A (plain) or P (whitened), applied through the thin SVD of A
              (Woodbury); rho_t, theta_t softplus; alpha_t = phi sigmoid(a_t) in (0, golden ratio), init 1
              -> 3 parameters per layer (LADMM-A)
  analytic1 : the same with alpha_t = 1 (the ADMM-CE-Net form) -> 2 parameters per layer

Every model exports MATLAB names H_matrix_t, rho_t, theta_t, alpha_t (structured operators
are also materialised into H_matrix_t), so LADMM_test.m runs them unchanged.
"""
from __future__ import annotations

from typing import List, Optional

import numpy as np
import numpy.linalg as la
import torch
import torch.nn.functional as F

from .common import Stage, UnrolledNet, rmm
from .shrinkage import beam_noise_level, csoft_ladmm, firm_shrink, gm_mmse_shrink

FAITHFUL = ("full", "diag", "lowrank", "kron")
THRESHOLDS = ("scalar", "group", "group_k", "group_kc", "adaptive", "firm", "gm")
NOISE_THRESHOLDS = ("adaptive", "firm", "gm")       # v28: threshold scale x blind noise level
ANALYTIC = ("analytic", "analytic1")                # v28: (G + rho I)^-1, nothing learned in the operator
PROPOSED = ("psd", "kronprior", "diagprior") + ANALYTIC
DATA_STEPS = ("plain", "whitened")
GOLD = (1 + 5 ** 0.5) / 2


def gm_init(K: int):
    """Initial mixture weights and variances of threshold='gm' (pilot values for K = 4)."""
    if K == 4:
        return np.log([0.80, 0.12, 0.05, 0.03]), np.log([1e-3, 0.5, 5.0, 50.0])
    w = np.r_[0.8, np.full(K - 1, 0.2 / max(K - 1, 1))][:K]
    return np.log(w / w.sum()), np.log(np.logspace(-3, np.log10(50.0), K))


def kron_factors_init(W_full: np.ndarray, N1: int, N2: int):
    """init_kronecker_factors() of the Kronecker script (rank-1 rearrangement SVD)."""
    R = W_full.reshape(N1, N2, N1, N2).transpose(0, 2, 1, 3).reshape(N1 * N1, N2 * N2)
    U, S, Vt = la.svd(R, full_matrices=False)
    H1 = np.sqrt(S[0]) * U[:, 0].reshape(N1, N1)
    H2 = np.sqrt(S[0]) * Vt[0, :].reshape(N2, N2)
    return H1, H2


def _softplus_inv(v: float) -> float:
    return float(np.log(np.expm1(v)))


class LADMM(UnrolledNet):
    family = "ladmm"

    def __init__(self, A, T: int = 6, operator: str = "full", untied: bool = True, rank: int = 16,
                 N1: int = 16, N2: int = 16, init_precision: Optional[str] = None,
                 lowrank_init_scale: float = 0.01, prior_init: float = 0.1,
                 threshold: str = "scalar", group_kappa: float = 3.0, group_win: int = 3, array: str = "UPA",
                 data_step: str = "plain", gm_K: int = 4, precision: str = "float64", device=None):
        super().__init__(A, precision)
        assert operator in FAITHFUL + PROPOSED, operator
        assert threshold in THRESHOLDS, threshold
        assert data_step in DATA_STEPS, data_step
        self.data_step, self.gm_K = data_step, int(gm_K)
        self._rho_soft = operator in PROPOSED
        self._theta_soft = operator in PROPOSED or threshold in NOISE_THRESHOLDS
        self.threshold, self.group_kappa, self.group_win, self.array = threshold, group_kappa, group_win, array.upper()
        self.T, self.operator, self.untied, self.rank, self.N1, self.N2 = T, operator, untied, rank, N1, N2
        self.cfg = dict(model="LADMM", T=T, operator=operator, untied=untied, rank=rank, N1=N1, N2=N2,
                        precision=precision, init_precision=init_precision,
                        lowrank_init_scale=lowrank_init_scale, prior_init=prior_init,
                        threshold=threshold, group_kappa=group_kappa, group_win=group_win, array=self.array)
        if data_step != "plain":                 # v28 keys only when used: configs of older runs stay as they were
            self.cfg["data_step"] = data_step
        if threshold == "gm":
            self.cfg["gm_K"] = self.gm_K
        N, M = self.N, self.M
        Anp = self.A_np
        white = data_step == "whitened"
        if white:                                 # P = A^T (A A^T)^-1 A, A^+ = A^T (A A^T)^-1, (A A^T)^-1/2
            ev, Ue = la.eigh(Anp @ Anp.T)
            Apinv = Anp.T @ (Ue / ev) @ Ue.T
            Gram = Apinv @ Anp
            Gram = 0.5 * (Gram + Gram.T)
            iSq = (Ue / np.sqrt(ev)) @ Ue.T
            self.register_buffer("Apinv", torch.tensor(Apinv, dtype=self.rdtype))
        else:
            Gram = Anp.T @ Anp
        # inv(A^T A + I) computed like the scripts: float32 script -> float32 numpy, else float64
        ip = init_precision or precision
        if white:
            W0 = la.inv(Gram + np.eye(N))
        elif ip == "float32":
            A32 = Anp.astype(np.float32)
            W0 = la.inv(A32.T @ A32 + np.eye(N, dtype=np.float32)).astype(np.float32)
        elif ip == "untitled2":      # user's untitled2.py: float32 A^T A + float64 I, inverse in float64, cast to float32
            A32 = Anp.astype(np.float32)
            W0 = la.inv(A32.T @ A32 + np.eye(N)).astype(np.float32)
        else:
            W0 = la.inv(Anp.T @ Anp + np.eye(N))
        self.W0 = W0
        self.register_buffer("Wfix", torch.tensor(np.asarray(W0, np.float64), dtype=self.rdtype))
        self.register_buffer("AtA", torch.tensor(Gram, dtype=self.rdtype))      # the Gram of the data step
        if operator in ANALYTIC:                  # thin SVD: A^T A = V diag(s^2) V^T, P = V V^T
            _, sv, Vt = la.svd(Anp, full_matrices=False)
            self.register_buffer("Vs", torch.tensor(Vt.T, dtype=self.rdtype))
            self.register_buffer("gs", torch.tensor(np.ones(M) if white else sv ** 2, dtype=self.rdtype))
        if operator == "lowrank":
            Uf, Sf, Vtf = la.svd(W0, full_matrices=False)
            P0 = Uf[:, :rank] * np.sqrt(Sf[:rank])[None, :] * lowrank_init_scale
            Q0 = Vtf[:rank, :].T * np.sqrt(Sf[:rank])[None, :] * lowrank_init_scale
        if operator == "kron":
            K1, K2 = kron_factors_init(np.asarray(W0, np.float64), N1, N2)
        self._op_names: List[List[str]] = []
        for t in range(T):
            names = []
            if operator in ANALYTIC:
                pass
            elif untied or t == 0:
                if operator == "full":
                    self._p(f"H_matrix_{t}", W0); names = [f"H_matrix_{t}"]
                elif operator == "diag":
                    self._p(f"d_{t}", np.ones(N)); names = [f"d_{t}"]
                elif operator == "lowrank":
                    self._p(f"P_{t}", P0); self._p(f"Q_{t}", Q0); names = [f"P_{t}", f"Q_{t}"]
                elif operator == "kron":
                    self._p(f"H1_{t}", K1); self._p(f"H2_{t}", K2); names = [f"H1_{t}", f"H2_{t}"]
                elif operator == "psd":
                    L0 = np.zeros((N, N)); L0[:, :M] = Anp.T @ iSq if white else Anp.T
                    self._p(f"L_{t}", L0); names = [f"L_{t}"]
                elif operator == "kronprior":
                    self._p(f"L1_{t}", prior_init * np.eye(N1)); self._p(f"L2_{t}", prior_init * np.eye(N2))
                    names = [f"L1_{t}", f"L2_{t}"]
                elif operator == "diagprior":
                    self._p(f"d_{t}", np.full(N, _softplus_inv(prior_init))); names = [f"d_{t}"]
            self._op_names.append(names if (names or operator in ANALYTIC) else self._op_names[0])
            self._p(f"rho_raw_{t}" if self._rho_soft else f"rho_{t}", _softplus_inv(1.0) if self._rho_soft else 1.0)
            if self._theta_soft:
                self._p(f"theta_raw_{t}", np.array([_softplus_inv(1.0)]))
            else:
                self._p(f"theta_{t}", np.array([1.0]))
            if operator == "analytic":
                self._p(f"alpha_raw_{t}", float(np.log((1 / GOLD) / (1 - 1 / GOLD))))   # phi sigmoid(.) = 1
            elif operator != "analytic1":
                self._p(f"alpha_{t}", 1.0)
            if threshold == "firm":
                self._p(f"firm_raw_{t}", _softplus_inv(1.0))                       # t2 = 2 t1
            if threshold == "gm":
                lw, ln = gm_init(self.gm_K)
                self._p(f"gm_logit_{t}", lw)
                self._p(f"gm_lognu_{t}", ln)
            if threshold in ("group_k", "group_kc"):
                self._p(f"kappa_raw_{t}", _softplus_inv(group_kappa))
            if threshold == "group_kc":
                shape = (group_win, group_win) if self.array == "UPA" else (group_win,)
                self._p(f"kern_raw_{t}", np.full(shape, _softplus_inv(1.0)))
        if device is not None:
            self.to(device)

    # -- combiner (v29) ----------------------------------------------------------------
    def set_combiner(self, A):
        """Use another combiner A (M' x N). Only the analytic operators follow it (nothing in them was learned for one
        combiner); every other operator was learned for its own A, so it raises ValueError. v30: any number of RF
        chains M' (the buffers of another shape are registered again)."""
        if self.operator not in ANALYTIC:
            raise ValueError(f"LADMM operator '{self.operator}' was learned for one combiner")
        A = np.asarray(A.detach().cpu().numpy() if torch.is_tensor(A) else A, np.float64)
        if A.ndim != 2 or A.shape[1] != self.N:
            raise ValueError(f"combiner of shape {A.shape}: it must have N = {self.N} columns (beams)")
        if np.linalg.matrix_rank(A) < A.shape[0]:
            raise ValueError(f"combiner of shape {A.shape} has dependent rows (A A^T is singular)")
        white = self.data_step == "whitened"
        put = self._put_buffer
        self.M, self.A_np = A.shape[0], A
        put("A", A)
        if white:
            ev, Ue = la.eigh(A @ A.T)
            Apinv = A.T @ (Ue / ev) @ Ue.T
            Gram = Apinv @ A
            put("Apinv", Apinv)
            put("AtA", 0.5 * (Gram + Gram.T))
        else:
            put("AtA", A.T @ A)
        _, sv, Vt = la.svd(A, full_matrices=False)
        put("Vs", Vt.T)
        put("gs", np.ones(self.M) if white else sv ** 2)

    # -- per-layer quantities -------------------------------------------------------
    def rho(self, t):
        return F.softplus(self.P(f"rho_raw_{t}")) if self._rho_soft else self.P(f"rho_{t}")

    def theta(self, t):
        return F.softplus(self.P(f"theta_raw_{t}")) if self._theta_soft else self.P(f"theta_{t}")

    def alpha(self, t):
        if self.operator == "analytic":
            return GOLD * torch.sigmoid(self.P(f"alpha_raw_{t}"))
        if self.operator == "analytic1":
            return torch.ones((), dtype=self.rdtype, device=self.A.device)
        return self.P(f"alpha_{t}")

    def _alpha_name(self, t):
        return {"analytic": f"alpha_raw_{t}", "analytic1": None}.get(self.operator, f"alpha_{t}")

    def _shrink(self, t: int, V: torch.Tensor, rho: torch.Tensor) -> torch.Tensor:
        if self.threshold in NOISE_THRESHOLDS:
            c = self.theta(t).reshape(-1)[0]
            sig = beam_noise_level(V)                               # (1, B)
            if self.threshold == "adaptive":
                return csoft_ladmm(V, c * sig)
            if self.threshold == "firm":
                t1 = c * sig
                return firm_shrink(V, t1, t1 * (1 + F.softplus(self.P(f"firm_raw_{t}"))))
            return gm_mmse_shrink(V, (c * sig) ** 2, self.P(f"gm_logit_{t}"), self.P(f"gm_lognu_{t}"))
        thr = self.theta(t).reshape(-1)[0] / rho
        if self.threshold != "scalar":
            thr = thr * self._group_weights(t, V)
        return csoft_ladmm(V, thr)

    def _group_weights(self, t: int, V: torch.Tensor) -> torch.Tensor:
        """w_i = 1/(1 + kappa g_i): low threshold inside energy clusters, full threshold elsewhere."""
        B = V.shape[1]
        E = torch.abs(V) ** 2
        r = (self.group_win - 1) // 2
        offs = range(-r, r + 1)
        if self.threshold == "group_kc":
            c = F.softplus(self.P(f"kern_raw_{t}"))
            c = c / c.sum()
        if self.array == "UPA":
            E = E.reshape(self.N1, self.N2, B)       # kron(U_N1, U_N2): N2 index fastest
            S = torch.zeros_like(E)
            for a, da in enumerate(offs):
                for b, db in enumerate(offs):
                    wgt = c[a, b] if self.threshold == "group_kc" else 1.0 / self.group_win ** 2
                    S = S + wgt * torch.roll(E, shifts=(da, db), dims=(0, 1))
            S = S.reshape(self.N, B)
        else:
            S = torch.zeros_like(E)
            for a, da in enumerate(offs):
                wgt = c[a] if self.threshold == "group_kc" else 1.0 / self.group_win
                S = S + wgt * torch.roll(E, shifts=da, dims=0)
        g = torch.sqrt(S + 1e-12)
        g = g / (g.mean(dim=0, keepdim=True) + 1e-12)
        kappa = F.softplus(self.P(f"kappa_raw_{t}")) if self.threshold in ("group_k", "group_kc") else self.group_kappa
        return 1.0 / (1.0 + kappa * g)

    def _apply_op(self, t: int, inp: torch.Tensor) -> torch.Tensor:
        op, n = self.operator, self._op_names[t]
        if op == "full":
            return rmm(self.P(n[0]), inp)
        if op == "diag":
            d = self.P(n[0]).reshape(self.N, 1)
            return torch.complex(d * (self.Wfix @ inp.real), d * (self.Wfix @ inp.imag))
        if op == "lowrank":
            P, Q = self.P(n[0]), self.P(n[1])
            re, im = inp.real, inp.imag
            return torch.complex(self.Wfix @ re + P @ (Q.T @ re), self.Wfix @ im + P @ (Q.T @ im))
        if op == "kron":
            H1, H2 = self.P(n[0]), self.P(n[1])
            return torch.complex(self._kron_tf(H1, H2, inp.real), self._kron_tf(H1, H2, inp.imag))
        if op in ANALYTIC:                       # (G + rho I)^-1 v = (v - V diag(g/(g+rho)) V^T v) / rho
            rho = self.rho(t)
            c = rmm(self.Vs.T, inp)
            c = c * (self.gs / (self.gs + rho)).reshape(-1, 1).to(c.dtype)
            return (inp - rmm(self.Vs, c)) / rho.to(inp.dtype)
        # proposed: exact regularised LS solve
        return rmm(self.H_matrix(t), inp)

    def _kron_tf(self, H1, H2, x):
        B = x.shape[1]
        X = x.reshape(self.N2, self.N1, B)
        s1 = torch.einsum("ij,jkb->ikb", H2, X)
        s2 = torch.einsum("ikb,lk->ilb", s1, H1)
        return s2.reshape(self.N1 * self.N2, B)

    def H_matrix(self, t: int) -> torch.Tensor:
        """Materialised N x N operator of layer t (used for export / MATLAB)."""
        op, n = self.operator, self._op_names[t]
        N = self.N
        I = torch.eye(N, dtype=self.rdtype, device=self.A.device)
        if op == "full":
            return self.P(n[0])
        if op == "diag":
            return self.P(n[0]).reshape(N, 1) * self.Wfix
        if op == "lowrank":
            return self.Wfix + self.P(n[0]) @ self.P(n[1]).T
        if op == "kron":
            return torch.kron(self.P(n[1]), self.P(n[0]))   # (H2 kron H1), see module docstring
        if op == "psd":
            L = self.P(n[0])
            return torch.linalg.inv(L @ L.T + self.rho(t) * I)
        if op == "kronprior":
            L1, L2 = self.P(n[0]), self.P(n[1])
            Lam = torch.kron(L1 @ L1.T, L2 @ L2.T)
            return torch.linalg.inv(self.AtA + self.rho(t) * I + Lam)
        if op == "diagprior":
            return torch.linalg.inv(self.AtA + self.rho(t) * I + torch.diag(F.softplus(self.P(n[0]))))
        if op in ANALYTIC:
            return torch.linalg.inv(self.AtA + self.rho(t) * I)
        raise ValueError(op)

    # -- training plan / forward ----------------------------------------------------
    def stage_plan(self) -> List[Stage]:
        stages, allv = [], []
        for t in range(self.T):
            new = list(self._op_names[t])      # tied: TF var_list still holds the layer-0 operator
            rho_name = f"rho_raw_{t}" if self._rho_soft else f"rho_{t}"
            th_name = f"theta_raw_{t}" if self._theta_soft else f"theta_{t}"
            allv += [n for n in new if n not in allv] + [rho_name]
            stages.append(Stage(f"LADMM-soft H-update T={t + 1}", "linear", new + [rho_name], list(allv)))
            extra = ([f"kappa_raw_{t}"] if self.threshold in ("group_k", "group_kc") else []) + (
                [f"kern_raw_{t}"] if self.threshold == "group_kc" else []) + (
                [f"firm_raw_{t}"] if self.threshold == "firm" else []) + (
                [f"gm_logit_{t}", f"gm_lognu_{t}"] if self.threshold == "gm" else [])
            allv += [th_name] + extra
            stages.append(Stage(f"LADMM-soft Z-update T={t + 1}", "nonlinear", [th_name] + extra, list(allv)))
            if self._alpha_name(t):
                allv.append(self._alpha_name(t))
        for i, st in enumerate(stages):
            st.index = i
        return stages

    def forward_stages(self, y, upto: Optional[int] = None):
        y = self.as_complex(y)
        n_st = 2 * self.T if upto is None else upto + 1
        B = y.shape[1]
        Ay = rmm(self.Apinv, y) if self.data_step == "whitened" else rmm(self.A.T, y)
        Z = torch.zeros((self.N, B), dtype=self.cdtype, device=y.device)
        U = torch.zeros_like(Z)
        outs = []
        for t in range(self.T):
            rho = self.rho(t)
            rho_c = rho.to(self.cdtype)
            inp = Ay + rho_c * Z - U
            H = self._apply_op(t, inp)
            outs.append(H)
            if len(outs) >= n_st:
                return outs
            V = H + U / rho_c
            Z = self._shrink(t, V, rho)
            outs.append(Z)
            if len(outs) >= n_st:
                return outs
            U = U + self.alpha(t).to(self.cdtype) * rho_c * (H - Z)
        return outs

    # -- export -----------------------------------------------------------------------
    def mat_dict(self):
        d = {}
        with torch.no_grad():
            for t in range(self.T):
                d[f"H_matrix_{t}"] = self.H_matrix(t).cpu().numpy()
                d[f"rho_{t}"] = np.array(float(self.rho(t)))
                d[f"theta_{t}"] = self.theta(t).cpu().numpy().reshape(1)
                d[f"alpha_{t}"] = np.array(float(self.alpha(t)))
            for n in self.param_names():          # raw learned parameters too
                if n not in d:
                    d[n] = self.P(n).detach().cpu().numpy()
        d["T"] = np.array(self.T)
        d["operator"] = self.operator
        return d

    def load_mat_dict(self, d):
        with torch.no_grad():
            for n in self.param_names():
                if n in d:
                    v = np.asarray(d[n], np.float64).reshape(tuple(self.P(n).shape))
                    self.P(n).copy_(torch.as_tensor(v, dtype=self.rdtype))
                elif n.startswith("H_matrix_") or n.startswith("rho_") or n.startswith("theta_") or n.startswith("alpha_"):
                    raise KeyError(n)

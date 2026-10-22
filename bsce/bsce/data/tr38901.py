"""3GPP TR 38.901 channels for the bsce setting, in NumPy: the TR procedure as written.

The setting of docs/3GPP_DATASET.md:
- UMi street canyon (Table 7.5-6 v16.1, or the Rel-19 values v19.2);
- one BS at the origin, height h_bs, boresight +x. Single-polarised (V) panel of 38.901 elements (Table 7.3-1),
  or isotropic elements; lambda/2 spacing. UPA 16 x 16 on y-z, ULA 256 on y, or on z;
- UEs with one isotropic V-polarised antenna, not rotated, outdoor, height h_ut. Uniform in the 120-degree hexagonal
  sector cell (ISD), d2D >= d_min;
- narrowband at the carrier (h = sum of the ray coefficients), plane waves, no path loss / shadowing in h (bsce
  normalises ||x||^2 = N per channel anyway).

Steps (TR 38.901 §7.4.2, §7.5; numbers of the equations of v16.1):
  LOS state       Table 7.4.2-1, per user
  step 4          LSPs [DS, ASD, ASA, SF, K, ZSA, ZSD], log-normal (Table 7.5-6). Spatially correlated between users of
                  the same state, exp(-d/D_X). Cross-correlated by the Cholesky factor in the TR's order
                  [SF, K, DS, ASD, ASA, ZSD, ZSA]. ASD, ASA <= 104 deg; ZSD, ZSA <= 52 deg
  step 5          cluster delays (7.5-1, 7.5-2), only for the order of the clusters and their powers
  step 6          cluster powers (7.5-5, 7.5-6), LOS powers for the angles (7.5-8), clusters < -25 dB removed
  step 7          AOD (7.5-9 to 7.5-13) and ZOD (7.5-14 to 7.5-20). C_phi, C_theta of Tables 7.5-2 and 7.5-4 with the
                  K-factor terms. ZOD: the NLOS offset of Table 7.5-6 and the ray spread (3/8) 10^mu_lgZSD
  step 8          AOD and ZOD rays coupled at random within each cluster (within the sub-clusters of Table 7.5-5 for
                  the two strongest clusters)
  steps 10-11     random initial phases; channel coefficient (7.5-22) with the LOS ray (7.5-29, 7.5-30)
Because the UE antenna is isotropic, V-polarised and not rotated, only the departure side enters h: F_rx = [1, 0].
The arrival angles, their ray coupling and the XPR (steps 7-9 for AOA/ZOA, step 9) therefore do not change h and are not
drawn. The term F_tx,theta e^{j Phi_theta-theta} is the only one left of the polarisation matrix.
Checked against Sionna 2.2 (sionna.phy.channel.tr38901.UMi) in docs/3GPP_DATASET.md.
"""
from __future__ import annotations

import time
from typing import Callable, Optional

import numpy as np

# Table 7.5-6 Part-1, UMi street canyon: mu = a log10(1 + fc) + c, sigma = a' log10(1 + fc) + c' (log10 of s / deg);
# K, SF in dB. The values are those of Sionna 2.2's UMi_LoS / UMi_NLoS json files (v16_1, v19_2).
_COMMON = {
    "LOS": dict(N=12, r_tau=3.0, zeta=3.0, cASD=3.0, cASA=17.0, cZSA=7.0, C_phi=1.146, C_theta=1.104, K=(9.0, 5.0),
                SF=4.0, ZSD_sigma=0.35,
                corr=dict(ASDvsDS=0.5, ASAvsDS=0.8, ASAvsSF=-0.4, ASDvsSF=-0.5, DSvsSF=-0.4, ASDvsASA=0.4, ASDvsK=-0.2,
                          ASAvsK=-0.3, DSvsK=-0.7, SFvsK=0.5, ZSDvsSF=0.0, ZSAvsSF=0.0, ZSDvsK=0.0, ZSAvsK=0.0,
                          ZSDvsDS=0.0, ZSAvsDS=0.2, ZSDvsASD=0.5, ZSAvsASD=0.3, ZSDvsASA=0.0, ZSAvsASA=0.0,
                          ZSDvsZSA=0.0),
                dist=dict(DS=7, ASD=8, ASA=8, SF=10, K=15, ZSA=12, ZSD=12)),
    "NLOS": dict(N=19, r_tau=2.1, zeta=3.0, cASD=10.0, cASA=22.0, cZSA=7.0, C_phi=1.273, C_theta=1.184, K=(0.0, 0.0),
                 SF=7.82, ZSD_sigma=0.35,
                 corr=dict(ASDvsDS=0.0, ASAvsDS=0.4, ASAvsSF=-0.4, ASDvsSF=0.0, DSvsSF=-0.7, ASDvsASA=0.0, ASDvsK=0.0,
                           ASAvsK=0.0, DSvsK=0.0, SFvsK=0.0, ZSDvsSF=0.0, ZSAvsSF=0.0, ZSDvsK=0.0, ZSAvsK=0.0,
                           ZSDvsDS=-0.5, ZSAvsDS=0.0, ZSDvsASD=0.5, ZSAvsASD=0.5, ZSDvsASA=0.0, ZSAvsASA=0.2,
                           ZSDvsZSA=0.0),
                 dist=dict(DS=10, ASD=10, ASA=9, SF=13, K=1, ZSA=10, ZSD=10)),
}
# (mu_a, mu_c, sigma_a, sigma_c) per LSP
_LSP = {
    "16.1": {"LOS": dict(DS=(-0.24, -7.14, 0.0, 0.38), ASD=(-0.05, 1.21, 0.0, 0.41), ASA=(-0.08, 1.73, 0.014, 0.28),
                         ZSA=(-0.1, 0.73, -0.04, 0.34)),
             "NLOS": dict(DS=(-0.24, -6.83, 0.16, 0.28), ASD=(-0.23, 1.53, 0.11, 0.33), ASA=(-0.08, 1.81, 0.05, 0.3),
                          ZSA=(-0.04, 0.92, -0.07, 0.41))},
    "19.2": {"LOS": dict(DS=(-0.18, -7.28, 0.0, 0.39), ASD=(-0.05, 1.21, 0.08, 0.29), ASA=(-0.07, 1.66, 0.021, 0.26),
                         ZSA=(-0.11, 0.81, -0.03, 0.29)),
             "NLOS": dict(DS=(-0.22, -6.87, 0.19, 0.22), ASD=(-0.24, 1.54, 0.10, 0.33), ASA=(-0.07, 1.76, 0.05, 0.27),
                          ZSA=(-0.03, 0.92, -0.05, 0.35))},
}
ORDER = ("DS", "ASD", "ASA", "SF", "K", "ZSA", "ZSD")          # internal LSP order
TR_ORDER = ("SF", "K", "DS", "ASD", "ASA", "ZSD", "ZSA")       # Cholesky order of step 4
# Table 7.5-3 (ray offsets) and Table 7.5-5 (sub-clusters, 0-based rays)
RAY_OFFSETS = np.array([0.0447, -0.0447, 0.1413, -0.1413, 0.2492, -0.2492, 0.3715, -0.3715, 0.5129, -0.5129,
                        0.6797, -0.6797, 0.8844, -0.8844, 1.1481, -1.1481, 1.5195, -1.5195, 2.1551, -2.1551])
SUBCLUSTERS = (np.array([0, 1, 2, 3, 4, 5, 6, 7, 18, 19]), np.array([8, 9, 10, 11, 16, 17]), np.array([12, 13, 14, 15]))
GEOMETRY = {"UPA": (16, 16), "ULA_Y": (1, 256), "ULA": (256, 1)}      # (vertical z, horizontal y) elements
SCENARIOS = ("umi", "umi_los", "umi_nlos")


def array_positions(array: str) -> np.ndarray:
    """Element positions (N x 3) in wavelengths, lambda/2 spacing, centred; horizontal (y) index slow, vertical (z)
    fast: the element order of scripts/sionna_make_H.py and matlab/quadriga_make_H.m."""
    nv, nh = GEOMETRY[array]
    z = (np.arange(nv) - (nv - 1) / 2) * 0.5
    y = (np.arange(nh) - (nh - 1) / 2) * 0.5
    Y, Z = np.meshgrid(y, z, indexing="ij")               # (nh, nv): y slow, z fast after ravel
    return np.stack([np.zeros(Y.size), Y.ravel(), Z.ravel()], 1)


def element_field(theta_deg: np.ndarray, phi_deg: np.ndarray, element: str = "38.901") -> np.ndarray:
    """F_theta of a V-polarised BS element in the GCS (no array rotation), Table 7.3-1; isotropic for 'omni'."""
    if element == "omni":
        return np.ones_like(theta_deg)
    av = -np.minimum(12.0 * ((theta_deg - 90.0) / 65.0) ** 2, 30.0)
    ah = -np.minimum(12.0 * (phi_deg / 65.0) ** 2, 30.0)
    a_db = 8.0 - np.minimum(-(av + ah), 30.0)
    return np.sqrt(10.0 ** (a_db / 10.0))


def p_los_umi(d2d: np.ndarray) -> np.ndarray:
    """Table 7.4.2-1, UMi street canyon."""
    d = np.maximum(np.asarray(d2d, float), 1e-9)
    return np.where(d <= 18.0, 1.0, 18.0 / d + np.exp(-d / 36.0) * (1.0 - 18.0 / d))


def hex_sector(rng: np.random.Generator, n: int, isd: float = 200.0, d_min: float = 10.0) -> np.ndarray:
    """n points uniform in the hexagonal sector cell with the BS at a corner (120 deg around +x), d2D >= d_min."""
    r = isd / 3
    out = np.zeros((2, 0))
    while out.shape[1] < n:
        m = 4 * n
        c = np.vstack([2 * r * rng.random(m), (2 * rng.random(m) - 1) * r * np.sqrt(3) / 2])
        q = np.abs(c - np.array([[r], [0.0]]))
        ok = (q[1] <= r * np.sqrt(3) / 2) & (np.sqrt(3) * q[0] + q[1] <= np.sqrt(3) * r) & (np.sum(c ** 2, 0) >= d_min ** 2)
        out = np.hstack([out, c[:, ok]])
    return out[:, :n]


def mu_lg_zsd(los: bool, d2d: np.ndarray, h_bs: float, h_ut: float) -> np.ndarray:
    """Mean of lgZSD, Table 7.5-6 (UMi)."""
    if los:
        return np.maximum(-0.21, -14.8 * d2d / 1000 + 0.01 * abs(h_ut - h_bs) + 0.83)
    return np.maximum(-0.5, -3.1 * d2d / 1000 + 0.01 * max(h_ut - h_bs, 0.0) + 0.2)


def _lsp_mean_std(spec: str, state: str, fc_ghz: float, d2d: np.ndarray, h_bs: float, h_ut: float):
    """(mu, sigma), each (n, 7) in ORDER, log10 domain (K and SF: dB / 10)."""
    fc = max(fc_ghz, 2.0)                                      # Table 7.5-6, note: UMi with fc < 2 GHz uses 2 GHz
    L = np.log10(1.0 + fc)
    t, c = _LSP[spec][state], _COMMON[state]
    n = d2d.size
    mu, sd = np.zeros((n, 7)), np.zeros((n, 7))
    for i, name in enumerate(ORDER):
        if name in t:
            a, b, sa, sb = t[name]
            mu[:, i], sd[:, i] = a * L + b, sa * L + sb
    mu[:, 3], sd[:, 3] = 0.0, c["SF"] / 10.0
    mu[:, 4], sd[:, 4] = c["K"][0] / 10.0, c["K"][1] / 10.0
    mu[:, 6], sd[:, 6] = mu_lg_zsd(state == "LOS", d2d, h_bs, h_ut), c["ZSD_sigma"]
    return mu, sd


def _cross_chol(state: str) -> np.ndarray:
    """Cholesky factor of the 7 x 7 cross-correlation matrix, in TR_ORDER (step 4)."""
    c = _COMMON[state]["corr"]
    C = np.eye(7)
    idx = {k: i for i, k in enumerate(TR_ORDER)}
    for key, v in c.items():
        a, b = key.split("vs")
        C[idx[a], idx[b]] = C[idx[b], idx[a]] = v
    if state == "NLOS":                                       # no K-factor in NLOS: decoupled
        C[idx["K"], :] = C[:, idx["K"]] = 0.0
        C[idx["K"], idx["K"]] = 1.0
    return np.linalg.cholesky(C)


def _spatial_sqrt(xy: np.ndarray, dist: float) -> np.ndarray:
    """Square root (lower Cholesky) of exp(-d/D) between the users of one state."""
    if xy.shape[1] <= 1 or dist <= 0:
        return np.eye(xy.shape[1])
    d = np.sqrt(((xy[:, :, None] - xy[:, None, :]) ** 2).sum(0))
    R = np.exp(-d / dist)
    try:
        return np.linalg.cholesky(R + 1e-10 * np.eye(R.shape[0]))
    except np.linalg.LinAlgError:
        w, V = np.linalg.eigh(R)
        return V * np.sqrt(np.maximum(w, 0.0))


def draw_lsps(rng: np.random.Generator, xy: np.ndarray, state: str, spec: str, fc_ghz: float, h_bs: float,
              h_ut: float) -> np.ndarray:
    """Step 4 for the users of one state: (n, 7) linear LSPs in ORDER (DS [s], ASD, ASA, SF [lin], K [lin], ZSA, ZSD)."""
    n = xy.shape[1]
    d2d = np.hypot(xy[0], xy[1])
    mu, sd = _lsp_mean_std(spec, state, fc_ghz, d2d, h_bs, h_ut)
    s = rng.standard_normal((7, n))                                   # one spatial field per LSP
    dist = _COMMON[state]["dist"]
    for i, name in enumerate(ORDER):
        s[i] = _spatial_sqrt(xy, dist[name]) @ s[i]
    perm = [ORDER.index(k) for k in TR_ORDER]
    st = _cross_chol(state) @ s[perm]                                 # cross-correlation, TR order
    s = np.empty_like(st)
    s[perm] = st
    lg = mu + sd * s.T
    v = 10.0 ** lg
    v[:, 1] = np.minimum(v[:, 1], 104.0)
    v[:, 2] = np.minimum(v[:, 2], 104.0)
    v[:, 5] = np.minimum(v[:, 5], 52.0)
    v[:, 6] = np.minimum(v[:, 6], 52.0)
    return v


def _wrap_zenith(t: np.ndarray) -> np.ndarray:
    t = np.mod(t, 360.0)
    return np.where(t > 180.0, 360.0 - t, t)


def _wrap_azimuth(p: np.ndarray) -> np.ndarray:
    p = np.mod(p, 360.0)
    return np.where(p > 180.0, p - 360.0, p)


def drop_channels(rng: np.random.Generator, n: int, array: str = "UPA", fc: float = 28e9, spec: str = "16.1",
                  scenario: str = "umi", element: str = "38.901", isd: float = 200.0, d_min: float = 10.0,
                  h_bs: float = 10.0, h_ut: float = 1.5, chunk: int = 50) -> dict:
    """One drop of n users. Returns H (N x n complex), loc (n x 3), los (n,), lsp (n x 7: DS [s], ASD, ASA, ZSD, ZSA
    [deg], K [dB], SF [dB])."""
    if scenario not in SCENARIOS:
        raise ValueError(f"scenario must be one of {SCENARIOS}")
    if spec not in _LSP:
        raise ValueError(f"spec must be one of {tuple(_LSP)}")
    fc_ghz = fc / 1e9
    xy = hex_sector(rng, n, isd, d_min)
    d2d = np.hypot(xy[0], xy[1])
    if scenario == "umi":
        los = rng.random(n) < p_los_umi(d2d)
    else:
        los = np.full(n, scenario == "umi_los")
    lsp = np.zeros((n, 7))
    for state, m in (("LOS", los), ("NLOS", ~los)):
        if m.any():
            lsp[m] = draw_lsps(rng, xy[:, m], state, spec, fc_ghz, h_bs, h_ut)
    N = array_positions(array).shape[0]
    H = np.zeros((N, n), complex)
    dz = h_ut - h_bs
    d3d = np.sqrt(d2d ** 2 + dz ** 2)
    phi_los = np.degrees(np.arctan2(xy[1], xy[0]))
    theta_los = np.degrees(np.arccos(dz / d3d))
    nv, nh = GEOMETRY[array]
    for k0 in range(0, n, chunk):
        ks = np.arange(k0, min(n, k0 + chunk))
        G, TH, PH = [], [], []                                        # per user: ray weights and angles
        for k in ks:
            st = "LOS" if los[k] else "NLOS"
            c = _COMMON[st]
            C = c["N"]
            ds, asd, zsd, K = lsp[k, 0], lsp[k, 1], lsp[k, 6], lsp[k, 4]
            # step 5: delays (unscaled; their LOS scaling 7.5-3/7.5-4 only shifts delays, not used narrowband)
            x = rng.random(C) * (1 - 1e-6) + 1e-6
            tau = -c["r_tau"] * ds * np.log(x)
            tau = np.sort(tau - tau.min())
            # step 6: powers
            z = rng.standard_normal(C) * c["zeta"]
            p = np.exp(-tau * (c["r_tau"] - 1.0) / (c["r_tau"] * ds)) * 10.0 ** (-z / 10.0)
            p = p / p.sum()
            p = np.where(p >= p.max() * 10 ** -2.5, p, 0.0)            # remove clusters < -25 dB
            if st == "LOS":
                pa = p / (K + 1.0)
                pa[0] += K / (K + 1.0)                                # (7.5-8): only for the angles
                K_db = 10 * np.log10(K)
                c_phi = c["C_phi"] * (1.1035 - 0.028 * K_db - 0.002 * K_db ** 2 + 0.0001 * K_db ** 3)
                c_th = c["C_theta"] * (1.3086 + 0.0339 * K_db - 0.0077 * K_db ** 2 + 0.0002 * K_db ** 3)
            else:
                pa, c_phi, c_th = p, c["C_phi"], c["C_theta"]
            zz = np.clip(pa / pa.max(), 1e-6, 1.0)
            # step 7: AOD
            ph_p = 2.0 * (asd / 1.4) * np.sqrt(-np.log(zz)) / c_phi
            X = rng.choice([-1.0, 1.0], C)
            Y = rng.standard_normal(C) * asd / 7.0
            ph = X * ph_p + Y + phi_los[k]
            # step 7: ZOD
            th_p = -zsd * np.log(zz) / c_th
            Xz = rng.choice([-1.0, 1.0], C)
            Yz = rng.standard_normal(C) * zsd / 7.0
            th = Xz * th_p + Yz
            if st == "LOS":
                ph = ph - (X[0] * ph_p[0] + Y[0])
                th = th + theta_los[k] - (Xz[0] * th_p[0] + Yz[0])
            else:
                off = -10.0 ** (-1.5 * np.log10(max(10.0, d2d[k])) + 3.3)
                th = th + theta_los[k] + off
            # rays: offsets, step 8 coupling (AOD rays fixed, ZOD rays permuted; sub-clusters for the 2 strongest)
            czsd = 3.0 / 8.0 * 10.0 ** mu_lg_zsd(st == "LOS", np.array([d2d[k]]), h_bs, h_ut)[0]
            perm = np.argsort(rng.random((C, 20)), axis=1)
            strongest = np.argsort(-p)[:2]
            for s_ in strongest:
                for grp in SUBCLUSTERS:
                    perm[s_, grp] = grp[np.argsort(rng.random(grp.size))]
            ph_r = _wrap_azimuth(ph[:, None] + c["cASD"] * RAY_OFFSETS[None, :])
            th_r = _wrap_zenith(th[:, None] + czsd * RAY_OFFSETS[perm])
            # steps 10-11: random phase, field, power
            Phi = rng.uniform(-np.pi, np.pi, (C, 20))
            g = np.sqrt(p[:, None] / 20.0) * element_field(th_r, ph_r, element) * np.exp(1j * Phi)
            g, th_r, ph_r = g.ravel(), th_r.ravel(), ph_r.ravel()
            if st == "LOS":
                g = np.concatenate([g / np.sqrt(K + 1.0),
                                    [np.sqrt(K / (K + 1.0)) * element_field(np.array([theta_los[k]]),
                                                                            np.array([phi_los[k]]), element)[0]
                                     * np.exp(-2j * np.pi * d3d[k] * fc / 299792458.0)]])
                th_r = np.concatenate([th_r, [theta_los[k]]])
                ph_r = np.concatenate([ph_r, [phi_los[k]]])
            G.append(g); TH.append(th_r); PH.append(ph_r)
        for j, k in enumerate(ks):
            t, f = np.radians(TH[j]), np.radians(PH[j])
            r = np.stack([np.sin(t) * np.cos(f), np.sin(t) * np.sin(f), np.cos(t)])   # (3, R)
            H[:, k] = _array_sum(r, G[j], nv, nh)
    out = np.zeros((n, 7))
    out[:, 0], out[:, 1], out[:, 2] = lsp[:, 0], lsp[:, 1], lsp[:, 2]
    out[:, 3], out[:, 4] = lsp[:, 6], lsp[:, 5]
    out[:, 5] = np.where(los, 10 * np.log10(np.maximum(lsp[:, 4], 1e-10)), -100.0)
    out[:, 6] = 10 * np.log10(lsp[:, 3])
    loc = np.c_[xy.T, np.full(n, h_ut)]
    return dict(H=H, loc=loc, los=los, lsp=out)


def _array_sum(r: np.ndarray, g: np.ndarray, nv: int, nh: int) -> np.ndarray:
    """sum_r g_r exp(j 2 pi r_hat_r . d_n / lambda) for every element n (y slow, z fast), done as a product of two
    small factors: UPA a = kron(a_y, a_z); a 256-element ULA as 16 x 16 sub-blocks (n = 16 a + b)."""
    if nv > 1 and nh > 1:
        yv = (np.arange(nh) - (nh - 1) / 2) * 0.5
        zv = (np.arange(nv) - (nv - 1) / 2) * 0.5
        f1 = np.exp(2j * np.pi * np.outer(yv, r[1]))
        f2 = np.exp(2j * np.pi * np.outer(zv, r[2]))
    else:
        L = max(nv, nh)
        u = r[1] if nh > 1 else r[2]
        q = int(round(np.sqrt(L)))
        if q * q != L:
            return np.exp(2j * np.pi * np.outer((np.arange(L) - (L - 1) / 2) * 0.5, u)) @ g
        f1 = np.exp(2j * np.pi * np.outer(np.arange(q) * q * 0.5, u))
        f2 = np.exp(2j * np.pi * np.outer((np.arange(q) - (L - 1) / 2) * 0.5, u))
    return ((f1 * g) @ f2.T).ravel()


def generate(n: int, array: str = "UPA", seed: int = 2026, drop_users: int = 250, first_drop: int = 1,
             progress: Optional[Callable[[int, int], None]] = None, **kw) -> dict:
    """n channels in drops of drop_users users; drop d (1-based, counted from first_drop) uses the seed (seed, d), so
    any range of drops can be made alone. Returns H (N x n), loc (n x 3), los, indoor, drop (n,), lsp (n x 7)."""
    if array not in GEOMETRY:
        raise ValueError(f"array must be one of {tuple(GEOMETRY)}")
    parts = {k: [] for k in ("H", "loc", "los", "lsp", "drop")}
    done, d = 0, first_drop
    while done < n:
        m = min(drop_users, n - done)
        r = drop_channels(np.random.default_rng([int(seed), int(d)]), m, array, **kw)
        for k in ("H", "loc", "los", "lsp"):
            parts[k].append(r[k])
        parts["drop"].append(np.full(m, d, float))
        done += m
        d += 1
        if progress:
            progress(done, n)
    H = np.concatenate(parts["H"], 1)
    return dict(H=H, loc=np.concatenate(parts["loc"]), los=np.concatenate(parts["los"]),
                indoor=np.zeros(n, bool), drop=np.concatenate(parts["drop"]), lsp=np.concatenate(parts["lsp"]))


def params_of(array: str, n: int, seed: int, drop_users: int, fc: float = 28e9, spec: str = "16.1",
              scenario: str = "umi", element: str = "38.901", isd: float = 200.0, d_min: float = 10.0,
              h_bs: float = 10.0, h_ut: float = 1.5) -> dict:
    """The params struct of the raw files (as scripts/sionna_make_H.py)."""
    nv, nh = GEOMETRY[array]
    return dict(generator="bsce bsce/data/tr38901.py", implementation=f"bsce built-in (NumPy), 3GPP TR 38.901 v{spec}",
                array=array, num_ant_x=1, num_ant_y=nh, num_ant_z=nv, ant_spacing=0.5,
                scenario={"umi": "3GPP_38.901_UMi", "umi_los": "3GPP_38.901_UMi_LOS",
                          "umi_nlos": "3GPP_38.901_UMi_NLOS"}[scenario], fc=fc,
                element="3gpp" if element == "38.901" else "omni", isd=isd, d_min=d_min, h_bs=h_bs, h_ut=h_ut,
                indoor_frac=0.0, drop_users=drop_users, seed=seed, K_total=n, narrowband=1, ue_antenna="omni",
                spec=spec, table="v16" if spec == "16.1" else "v19", zsd_fix=1, realized=0)


def save_raw(path, res: dict, params: dict, sl: slice = slice(None)) -> None:
    """Write channels in the raw-file format of scripts/sionna_make_H.py (read by make_dataset.py --input and
    check_3gpp.py --raw)."""
    import scipy.io as sio
    K = res["H"][:, sl].shape[1]
    sio.savemat(path, dict(H=res["H"][:, sl], loc=res["loc"][sl], los=res["los"][sl, None].astype(float),
                           indoor=res["indoor"][sl, None].astype(float), drop=res["drop"][sl, None],
                           lsp=res["lsp"][sl], lsp_real=np.full((K, 5), np.nan),
                           lsp_names=np.array(["DS_s", "ASD_deg", "ASA_deg", "ZSD_deg", "ZSA_deg", "KF_dB", "SF_dB"],
                                              dtype=object), params=params))


class Timer:
    def __init__(self, label: str):
        self.label, self.t0 = label, time.time()

    def __call__(self, done: int, n: int):
        if done == n or done % 5000 < 250:
            el = time.time() - self.t0
            print(f"  {self.label}: {done:,} / {n:,} channels, {el:.0f} s"
                  + (f", about {el / done * (n - done) / 60:.0f} min left" if done < n else ""), flush=True)

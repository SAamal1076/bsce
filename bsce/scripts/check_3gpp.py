#!/usr/bin/env python3
"""Check the 3GPP TR 38.901 channels of matlab/quadriga_make_H.m against the standard and compare them with SV and DeepMIMO.

  %run scripts/check_3gpp.py --builtin UPA ULA_Y              the built-in generator (as make_3gpp_data.py), 10,000 channels
  %run scripts/check_3gpp.py --raw UPA="raw3gpp/H_3gpp_upa_part*.mat" ULA_Y="raw3gpp/H_3gpp_ulay_part*.mat"
  (raw files of sionna_make_H.py / quadriga_make_H.m; ULA="..." adds the vertical ULA; KEY = UPA, ULA_Y or ULA)

1. LOS probability against 38.901 Table 7.4.2-1 (UMi street canyon), per 2-D distance bin.
2. The large-scale parameters drawn per user (DS, ASD, ASA, ZSD, ZSA, K) against 38.901 Table 7.5-6 (UMi street
   canyon) at the carrier of the files: median and robust log-std (IQR / 1.349), LOS and NLOS users apart. The spreads
   of the generated paths are shown too when the raw files have them (quadriga_make_H(..., 'realized', true)).
3. Beamspace x = U0^H h, ||x||^2 = N (as make_dataset.py): energy in the k strongest beams and the strongest-beam
   histogram, against bsce's SV and DeepMIMO test sets (data/SV_<ARR>, data/DeepMIMO_<ARR>).
4. Difficulty: NMSE of the LMMSE estimator built from the sample covariance of --n-cov training channels (the same
   number for every data set), on 2000 test channels, y = A(x + n), every data set with its own CS matrix.
5. Transfer: the authors' published GM-LAMP networks (trained on SV or DeepMIMO, reference/gmlamp/Train/Trained_Networks)
   on their own test set and on the 3GPP test set, at 5 dB (0-10 dB network) and 15 dB (10-20 dB network).
The 3GPP test set is the held-out-drop test set of make_dataset.py (the last --n-test users generated), its training
channels the first users. Writes results/<name>/check_3gpp.{json,md} and lsp_vs_38901.png, los_k_vs_38901.png,
beamspace_3datasets.png, beamspace_upa_maps.png, lmmse_3datasets.png, transfer_gmlamp.png.
"""
import argparse
import glob
import json
import sys
from pathlib import Path

import numpy as np
import scipy.io as sio
from scipy.stats import norm

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))
from bsce.config import resolve  # noqa: E402
from bsce.data import measurements  # noqa: E402
from bsce.rng import NumpyMatlabRNG  # noqa: E402

LSP = ["DS", "ASD", "ASA", "ZSD", "ZSA"]
UNIT = {"DS": "ns", "ASD": "deg", "ASA": "deg", "ZSD": "deg", "ZSA": "deg"}
# 38.901 Table 7.5-6 Part-1, UMi street canyon: mu = a log10(1 + fc) + b, sigma = c log10(1 + fc) + d (log10 of s / deg)
# v16: 38.901 v16.1 (QuaDRiGa v2.8.1); v19: the Rel-19 update of DS, ASD, ASA, ZSA (38.901 v19; Sionna 2.2 'v19_2')
UMI = {"LOS": {"DS": (-0.24, -7.14, 0.0, 0.38), "ASD": (-0.05, 1.21, 0.0, 0.41), "ASA": (-0.08, 1.73, 0.014, 0.28),
               "ZSA": (-0.1, 0.73, -0.04, 0.34), "K": (9.0, 5.0)},
       "NLOS": {"DS": (-0.24, -6.83, 0.16, 0.28), "ASD": (-0.23, 1.53, 0.11, 0.33), "ASA": (-0.08, 1.81, 0.05, 0.3),
                "ZSA": (-0.04, 0.92, -0.07, 0.41)}}
UMI_V19 = {"LOS": {"DS": (-0.18, -7.28, 0.0, 0.39), "ASD": (-0.05, 1.21, 0.08, 0.29), "ASA": (-0.07, 1.66, 0.021, 0.26),
                   "ZSA": (-0.11, 0.81, -0.03, 0.29), "K": (9.0, 5.0)},
           "NLOS": {"DS": (-0.22, -6.87, 0.19, 0.22), "ASD": (-0.24, 1.54, 0.10, 0.33), "ASA": (-0.07, 1.76, 0.05, 0.27),
                    "ZSA": (-0.03, 0.92, -0.05, 0.35)}}
TABLES = {"v16": UMI, "v19": UMI_V19}
PAL = {"SV": "#7a7a7a", "DeepMIMO": "#eb6834", "3GPP UPA": "#2a78d6", "3GPP ULA_Y": "#2a78d6", "3GPP ULA": "#1f9e6e"}


def p_los_umi(d):
    d = np.asarray(d, float)
    return np.where(d <= 18, 1.0, 18 / np.maximum(d, 1e-9) + np.exp(-d / 36) * (1 - 18 / np.maximum(d, 1e-9)))


def table(fc_ghz, state, name, d2d, h_bs, h_ut, version="v16"):
    """(mu, sigma) of log10(value) per user; ZSD with the distance law of Table 7.5-6 (UMi, the same in v16 and v19)."""
    if name == "ZSD":
        if state == "LOS":
            mu = np.maximum(-0.21, -14.8 * d2d / 1000 + 0.01 * abs(h_ut - h_bs) + 0.83)
        else:
            mu = np.maximum(-0.5, -3.1 * d2d / 1000 + 0.01 * max(h_ut - h_bs, 0) + 0.2)
        return mu, np.full_like(mu, 0.35)
    a, b, c, dd = TABLES[version][state][name]
    L = np.log10(1 + fc_ghz)
    return np.full(d2d.shape, a * L + b), np.full(d2d.shape, c * L + dd)


def rstd(v):
    q = np.percentile(v, [25, 75])
    return float((q[1] - q[0]) / 1.349)


def load_raw(pattern):
    from make_dataset import load_3gpp
    files = sorted(glob.glob(pattern)) or [pattern]
    d0 = sio.loadmat(files[0], variable_names=["params"])["params"]
    g = tuple(int(np.squeeze(d0[k][0, 0])) for k in ("num_ant_x", "num_ant_y", "num_ant_z"))
    arr = "UPA" if g == (1, 16, 16) else "ULA"
    raw, params = load_3gpp(files, arr)
    params["files"] = files
    return raw, params, arr, g


def u0_lens(arr):
    from deepmimo_convert import u0_1d
    return u0_1d(256) if arr == "ULA" else np.kron(u0_1d(16), u0_1d(16))


def to_beamspace(H, arr):
    x = u0_lens(arr).conj().T @ H
    e = np.sum(np.abs(x) ** 2, 0)
    return x * np.sqrt(256 / e)[None, :]


def energy_curve(x):
    p = np.sort(np.abs(x) ** 2, 0)[::-1]
    c = np.cumsum(p, 0) / np.sum(p, 0)
    return c.mean(1), np.argmax(c >= 0.95, 0) + 1


def lmmse_nmse(x_cov, x_test, A, snrs, seed=2026):
    R = x_cov @ x_cov.conj().T / x_cov.shape[1]
    out = []
    for i, s in enumerate(snrs):
        y = measurements.gen_measurements_matlab(x_test, A, float(s), NumpyMatlabRNG(seed + 1000 + i))
        s2 = 10 ** (-s / 10)
        C = A @ (R + s2 * np.eye(R.shape[0])) @ A.T
        xh = R @ A.T @ np.linalg.solve(C, y)
        out.append(10 * np.log10(np.sum(np.abs(xh - x_test) ** 2) / np.sum(np.abs(x_test) ** 2)))
    return out


def style(ax):
    ax.grid(alpha=0.3)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--raw", nargs="+", default=None, help='KEY=pattern, KEY = UPA, ULA_Y or ULA')
    ap.add_argument("--builtin", nargs="+", choices=["UPA", "ULA_Y", "ULA"], default=None,
                    help="check the built-in generator (bsce/data/tr38901.py) for these arrays instead of raw files")
    ap.add_argument("--n", type=int, default=10000, help="--builtin: channels per array")
    ap.add_argument("--spec", choices=["16.1", "19.2"], default="16.1", help="--builtin: 38.901 tables")
    ap.add_argument("--channel", choices=["umi", "umi_los", "umi_nlos"], default="umi", help="--builtin")
    ap.add_argument("--element", choices=["38.901", "omni"], default="38.901", help="--builtin")
    ap.add_argument("--seed", type=int, default=2026, help="--builtin")
    ap.add_argument("--n-test", type=int, default=2000)
    ap.add_argument("--n-cov", type=int, default=6000, help="training channels for the LMMSE sample covariance")
    ap.add_argument("--snr", nargs="+", type=float, default=[0, 5, 10, 15, 20])
    ap.add_argument("--no-transfer", action="store_true", help="skip the GM-LAMP transfer test")
    ap.add_argument("--authors", default="reference/gmlamp/Train/Trained_Networks")
    ap.add_argument("--name", default="check_3gpp")
    ap.add_argument("--results", default="results")
    a = ap.parse_args()
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.ticker  # noqa: F401

    out = resolve(a.results) / a.name
    out.mkdir(parents=True, exist_ok=True)
    if not a.raw and not a.builtin:
        sys.exit("give --raw KEY=pattern ... or --builtin UPA ULA_Y ...")
    sets = {}
    for key in a.builtin or []:
        from bsce.data import tr38901
        r = tr38901.generate(a.n, key, seed=a.seed, drop_users=250, spec=a.spec, scenario=a.channel, element=a.element,
                             progress=tr38901.Timer(f"built-in {key}"))
        raw = dict(H=r["H"], los=r["los"][:, None].astype(float), indoor=r["indoor"][:, None].astype(float),
                   drop=r["drop"][:, None], loc=r["loc"], lsp=r["lsp"], lsp_real=np.full((a.n, 5), np.nan))
        params = tr38901.params_of(key, a.n, a.seed, 250, spec=a.spec, scenario=a.channel, element=a.element)
        sets[key] = dict(raw=raw, params=params, arr="UPA" if key == "UPA" else "ULA")
        print(f"{key}: {a.n:,} built-in channels, LOS {r['los'].mean():.3f}")
    for item in a.raw or []:
        key, _, pat = item.partition("=")
        key, pat = key.strip().upper(), pat.strip().strip("'\"")
        if key not in ("UPA", "ULA_Y", "ULA") or not pat:
            sys.exit(f"--raw {item}: use KEY=pattern with KEY = UPA, ULA_Y or ULA")
        raw, params, arr, g = load_raw(pat)
        want = {"UPA": (1, 16, 16), "ULA_Y": (1, 256, 1), "ULA": (1, 1, 256)}[key]
        if g != want:
            sys.exit(f"--raw {key}: the files hold the (x, y, z) = {g} array, {want} expected")
        sets[key] = dict(raw=raw, params=params, arr=arr)
        print(f"{key}: {raw['H'].shape[1]:,} channels, LOS {raw['los'].mean():.3f}, {params.get('scenario')}, "
              f"{float(params['fc']) / 1e9:.1f} GHz, element {params.get('element')}")
    rep = {"sets": {k: {kk: (vv if not isinstance(vv, np.generic) else vv.item()) for kk, vv in v["params"].items()}
                    for k, v in sets.items()}}

    # ---------------- 1-2. the standard: LOS probability, K-factor, LSPs ----------------
    pool = {k: np.concatenate([s["raw"][k] for s in sets.values()]) for k in ("loc", "los", "lsp", "lsp_real", "indoor")}
    # the arrays of one seed hold the same users (same drops, positions and LSPs): count every user once
    key_ = np.round(np.c_[pool["loc"], pool["lsp"][:, :5] * np.array([1e9, 1, 1, 1, 1])], 6)
    _, first = np.unique(key_, axis=0, return_index=True)
    first = np.sort(first)
    if first.size < key_.shape[0]:
        print(f"standard check: {first.size:,} distinct users ({key_.shape[0] - first.size:,} repeated across the arrays)")
    pool = {k: v[first] for k, v in pool.items()}
    p0 = next(iter(sets.values()))["params"]
    fc, h_bs, h_ut = float(p0["fc"]) / 1e9, float(p0["h_bs"]), float(p0["h_ut"])
    out_door = pool["indoor"][:, 0] < 0.5
    d2d = np.hypot(pool["loc"][:, 0], pool["loc"][:, 1])
    los = pool["los"][:, 0] > 0.5
    scen = str(p0.get("scenario"))
    ver = str(p0.get("table", "v16"))
    if len({str(s["params"].get("table", "v16")) for s in sets.values()}) > 1:
        sys.exit("--raw: the arrays were made with different 'table' versions; check them apart")
    impl = str(p0.get("implementation", p0.get("generator", "?")))
    std = {"scenario": scen, "fc_GHz": fc, "h_bs": h_bs, "h_ut": h_ut, "n_users": int(los.size), "table": ver,
           "implementation": impl, "fixes": {k: p0.get(k) for k in ("zsd_fix", "czsd_fix", "zod_offset") if k in p0}}
    fig, ax = plt.subplots(1, 2, figsize=(12, 4.4))
    edges = np.linspace(10, max(d2d.max(), 20), 13)
    mid = 0.5 * (edges[1:] + edges[:-1])
    rows = []
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (d2d >= lo) & (d2d < hi) & out_door
        if m.sum():
            rows.append((0.5 * (lo + hi), float(los[m].mean()), float(np.sqrt(los[m].mean() * (1 - los[m].mean()) / m.sum())),
                         float(p_los_umi(d2d[m]).mean()), int(m.sum())))
    r = np.array(rows)
    exp_all = float(p_los_umi(d2d[out_door]).mean())
    obs_all = float(los[out_door].mean())
    sd_all = float(np.sqrt(exp_all * (1 - exp_all) / out_door.sum()))
    std["los_probability"] = dict(expected=exp_all, observed=obs_all, binomial_sd=sd_all,
                                  z=(obs_all - exp_all) / sd_all if sd_all > 0 else None,
                                  bins=[dict(d_mid=b[0], observed=b[1], sd=b[2], expected=b[3], n=b[4]) for b in rows])
    if "UMi" in scen and "LOS" not in scen.replace("UMi", ""):
        dd = np.linspace(10, edges[-1], 200)
        ax[0].plot(dd, p_los_umi(dd), color="#222", lw=1.6, label="38.901 Table 7.4.2-1 (UMi)")
        ax[0].errorbar(r[:, 0], r[:, 1], yerr=2 * r[:, 2], fmt="o", color="#2a78d6", ms=5, capsize=3,
                       label="generated (±2 sd)")
        for b in rows:
            ax[0].annotate(f"{b[1]:.2f}", (b[0], b[1]), textcoords="offset points", xytext=(0, 9), ha="center", fontsize=7)
        ax[0].set_title(f"LOS probability: {obs_all:.3f} generated, {exp_all:.3f} expected "
                        f"(z = {(obs_all - exp_all) / sd_all:+.2f})", fontsize=9)
    else:
        ax[0].text(0.5, 0.5, f"{scen}: LOS state fixed by the scenario\nLOS fraction {obs_all:.3f}", ha="center",
                   transform=ax[0].transAxes)
    ax[0].set_xlabel("2-D BS-UE distance (m)"); ax[0].set_ylabel("fraction of LOS users"); ax[0].set_ylim(0, 1.08)
    ax[0].legend(frameon=False, fontsize=8, loc="upper right"); style(ax[0])
    kf = pool["lsp"][los, 5]
    if kf.size:
        mu_k, sg_k = UMI["LOS"]["K"]
        v = np.sort(kf)
        ax[1].plot(v, np.arange(1, v.size + 1) / v.size, color="#2a78d6", lw=2,
                   label=f"generated: median {np.median(kf):.2f} dB, sd {rstd(kf):.2f} dB (n = {kf.size:,})")
        g_ = np.linspace(v[0], v[-1], 200)
        ax[1].plot(g_, norm.cdf(g_, mu_k, sg_k), "--", color="#222", lw=1.4,
                   label=f"38.901: N({mu_k:.0f} dB, {sg_k:.0f} dB)")
        std["K_LOS_dB"] = dict(median=float(np.median(kf)), rstd=rstd(kf), table_mu=mu_k, table_sigma=sg_k, n=int(kf.size))
    ax[1].set_xlabel("Ricean K-factor of LOS users (dB)"); ax[1].set_ylabel("CDF"); style(ax[1])
    ax[1].legend(frameon=False, fontsize=8, loc="lower right")
    ax[1].set_title("K-factor against 38.901 Table 7.5-6 (UMi LOS)", fontsize=9)
    fig.suptitle(f"3GPP {scen}, {fc:.0f} GHz, {impl}: LOS state and K-factor ({los.size:,} users)", fontsize=10)
    fig.tight_layout(); fig.savefig(out / "los_k_vs_38901.png", dpi=150); plt.close(fig)

    fig, axs = plt.subplots(2, 5, figsize=(21, 8.2))
    lsp_rows = []
    have_real = np.isfinite(pool["lsp_real"]).all(1)
    for ri, state in enumerate(["LOS", "NLOS"]):
        m = (los if state == "LOS" else ~los) & out_door
        for ci, name in enumerate(LSP):
            ax = axs[ri, ci]
            col = LSP.index(name)
            v = pool["lsp"][m, col] * (1e9 if name == "DS" else 1)
            if v.size < 10:
                ax.set_visible(False); continue
            lg = np.log10(pool["lsp"][m, col])
            mu, sg = table(fc, state, name, d2d[m], h_bs, h_ut, ver)
            mu_u = mu + (9 if name == "DS" else 0)               # log10 of ns
            res = lg - mu                                       # residual against the table, per user
            grid = np.linspace(np.percentile(np.log10(v), 0.2), np.percentile(np.log10(v), 99.8), 300)
            cdf_t = norm.cdf((grid[:, None] - mu_u[None, :]) / sg[None, :]).mean(1)
            med_t = float(10 ** np.interp(0.5, cdf_t, grid))
            vs = np.sort(v)
            ax.semilogx(vs, np.arange(1, vs.size + 1) / vs.size, color="#2a78d6", lw=2,
                        label=f"drawn: median {np.median(v):.3g} {UNIT[name]}")
            ax.semilogx(10 ** grid, cdf_t, "--", color="#222", lw=1.4, label=f"38.901 {ver}: median {med_t:.3g} {UNIT[name]}")
            row = dict(state=state, lsp=name, n=int(v.size), median=float(np.median(v)), table_median=med_t,
                       log_offset=float(np.median(res)), log_rstd=rstd(res if name == "ZSD" else lg),
                       table_log_sigma=float(np.median(sg)))
            rr = have_real[m]
            if rr.sum() > 10:
                vr = pool["lsp_real"][m][rr, col] * (1e9 if name == "DS" else 1)
                vr = np.sort(vr[vr > 0])
                ax.semilogx(vr, np.arange(1, vr.size + 1) / vr.size, ":", color="#c23b6a", lw=1.8,
                            label=f"generated paths: median {np.median(vr):.3g}")
                row["realized_median"] = float(np.median(vr))
            lsp_rows.append(row)
            ax.set_title(f"{state} {name}: median offset {row['log_offset']:+.3f} dec, "
                         f"log-sd {row['log_rstd']:.2f} / {row['table_log_sigma']:.2f}", fontsize=8.5)
            ax.xaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
            ax.set_xlabel(f"{name} ({UNIT[name]})"); ax.set_ylabel("CDF over users")
            ax.legend(frameon=False, fontsize=7.5, loc="upper left"); style(ax)
    std["lsp"] = lsp_rows
    fig.suptitle(f"Large-scale parameters against 38.901 {ver} Table 7.5-6 (UMi street canyon, {fc:.0f} GHz, "
                 f"h_BS {h_bs:g} m, h_UT {h_ut:g} m), {impl}; 'dec' = decades (log10), log-sd = IQR/1.349 (drawn / table)",
                 fontsize=10)
    fig.tight_layout(); fig.savefig(out / "lsp_vs_38901.png", dpi=140); plt.close(fig)
    rep["standard"] = std

    # ---------------- 3. beamspace against SV and DeepMIMO ----------------
    def test_x(scn, arr):
        f = ROOT / "data" / f"{scn}_{arr}" / f"{scn}_{arr}_test_channel.mat"
        return sio.loadmat(f)["x"] if f.exists() else None

    def train_x(scn, arr, n):
        f = ROOT / "data" / f"{scn}_{arr}" / f"{scn}_{arr}_training_channel.mat"
        return sio.loadmat(f)["x"][:, :n] if f.exists() else None

    groups = {"UPA": ["UPA"], "ULA": [k for k in ("ULA_Y", "ULA") if k in sets]}
    groups = {g: ks for g, ks in groups.items() if any(k in sets for k in ks)}
    data = {}
    for g, ks in groups.items():
        for scn in ("SV", "DeepMIMO"):
            xt = test_x(scn, g)
            if xt is not None:
                data[(g, scn)] = dict(test=xt, train=train_x(scn, g, a.n_cov), label=f"{scn} {g}"
                                      + (" (z axis)" if scn == "DeepMIMO" and g == "ULA" else ""), color=PAL[scn])
        for k in ks:
            if k not in sets:
                continue
            raw = sets[k]["raw"]
            K = raw["H"].shape[1]
            if K < a.n_test + a.n_cov:
                print(f"NOTE {k}: {K} channels < n_test + n_cov = {a.n_test + a.n_cov}; covariance from {K - a.n_test}")
            xt = to_beamspace(raw["H"][:, K - a.n_test:], sets[k]["arr"])
            xc = to_beamspace(raw["H"][:, :min(a.n_cov, K - a.n_test)], sets[k]["arr"])
            lab = {"UPA": "3GPP UPA", "ULA_Y": "3GPP ULA (y axis)", "ULA": "3GPP ULA (z axis)"}[k]
            data[(g, f"3GPP {k}")] = dict(test=xt, train=xc, label=lab, color=PAL[f"3GPP {k}"],
                                          ls="-" if k != "ULA" else "-.")
    fig, axs = plt.subplots(len(groups), 2, figsize=(14, 4.4 * len(groups)), squeeze=False)
    bs = {}
    for gi, g in enumerate(groups):
        for (gg, key), d in data.items():
            if gg != g:
                continue
            curve, k95 = energy_curve(d["test"])
            bs[d["label"]] = dict(beams95_median=float(np.median(k95)), beams95_p10=float(np.percentile(k95, 10)),
                                  beams95_p90=float(np.percentile(k95, 90)), top1_energy=float(curve[0]),
                                  top16_energy=float(curve[15]), n=int(d["test"].shape[1]))
            ls = d.get("ls", "-")
            axs[gi, 0].semilogx(np.arange(1, 257), curve, ls, color=d["color"], lw=2,
                                label=f"{d['label']}: 95 % in {np.median(k95):.0f} beams (p10-p90 "
                                      f"{np.percentile(k95, 10):.0f}-{np.percentile(k95, 90):.0f}), top-1 {curve[0]:.2f}")
            pk = np.bincount(np.argmax(np.abs(d["test"]), 0), minlength=256) / d["test"].shape[1]
            axs[gi, 1].plot(pk, ls, color=d["color"], lw=1.1, label=d["label"])
        axs[gi, 0].axhline(0.95, color="#999", lw=0.8)
        axs[gi, 0].set_xlabel("k strongest beams"); axs[gi, 0].set_ylabel("mean fraction of the energy")
        axs[gi, 0].set_title(f"{g}: energy in the k strongest beams (test sets, x = U0^H h, ||x||^2 = N)", fontsize=9)
        axs[gi, 1].set_xlabel("index of the strongest beam"); axs[gi, 1].set_ylabel("fraction of channels")
        axs[gi, 1].set_title(f"{g}: where the strongest beam is", fontsize=9)
        for ax in axs[gi]:
            ax.legend(frameon=False, fontsize=7.5); style(ax)
    fig.tight_layout(); fig.savefig(out / "beamspace_3datasets.png", dpi=140); plt.close(fig)
    rep["beamspace"] = bs
    upa = [d for (gg, key), d in data.items() if gg == "UPA"]
    if upa:                                    # where the strongest beam is, on the 16 x 16 beam grid
        fig, axs = plt.subplots(1, len(upa), figsize=(4.6 * len(upa), 4.4), squeeze=False)
        for ax, d in zip(axs[0], upa):
            pk = np.bincount(np.argmax(np.abs(d["test"]), 0), minlength=256) / d["test"].shape[1]
            img = pk.reshape(16, 16)           # row = slow index, column = fast index (3GPP: fast = vertical z)
            im = ax.imshow(img, origin="lower", cmap="Blues", vmin=0)
            top = np.argsort(pk)[::-1][:3]
            ax.set_title(f"{d['label']}: top beams hold {pk[top].sum():.2f} of the channels", fontsize=8.5)
            for t in top:
                ax.text(t % 16, t // 16, f"{pk[t]:.2f}", ha="center", va="center", fontsize=6.5, color="#c23b6a")
            ax.set_xlabel("fast beam index" + (" (vertical, z)" if d["label"].startswith("3GPP") else ""))
            ax.set_ylabel("slow beam index" + (" (horizontal, y)" if d["label"].startswith("3GPP") else ""))
            fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        fig.suptitle("UPA: fraction of the test channels whose strongest beam is at each grid point", fontsize=10)
        fig.tight_layout(); fig.savefig(out / "beamspace_upa_maps.png", dpi=150); plt.close(fig)

    # ---------------- 4. LMMSE difficulty ----------------
    A_sv = sio.loadmat(ROOT / "reference/gmlamp/CE_SV/CSmatrix256128.mat")["A"].astype(np.float64)
    A_dm = sio.loadmat(ROOT / "reference/gmlamp/CE_DeepMIMO/CSmatrix_128256.mat")["A"].astype(np.float64)
    fig, axs = plt.subplots(1, len(groups), figsize=(7 * len(groups), 4.6), squeeze=False)
    lm = {}
    for gi, g in enumerate(groups):
        ax = axs[0, gi]
        k_ = -1
        for (gg, key), d in data.items():
            if gg != g or d["train"] is None:
                continue
            k_ += 1
            A = A_dm if key == "DeepMIMO" else A_sv
            v = lmmse_nmse(d["train"], d["test"], A, a.snr)
            lm[d["label"]] = dict(snr_db=a.snr, nmse_db=v, n_cov=int(d["train"].shape[1]))
            ax.plot(a.snr, v, "o" + d.get("ls", "-"), color=d["color"], lw=1.8, ms=5, label=d["label"])
            for s, y in zip(a.snr, v):
                ax.annotate(f"{y:.1f}", (s, y), textcoords="offset points", xytext=(0, 6 if k_ % 2 == 0 else -12),
                            ha="center", fontsize=7, color=d["color"])
        ax.set_xlabel("SNR (dB)"); ax.set_ylabel("NMSE (dB)")
        ax.set_title(f"{g}: LMMSE with the sample covariance of {a.n_cov:,} training channels", fontsize=9)
        ax.legend(frameon=False, fontsize=8); style(ax)
    fig.tight_layout(); fig.savefig(out / "lmmse_3datasets.png", dpi=150); plt.close(fig)
    rep["lmmse"] = lm

    # ---------------- 5. transfer of the authors' GM-LAMP ----------------
    tr = {}
    authors = resolve(a.authors)
    if not a.no_transfer and authors.exists():
        from check_vs_authors import authors_net, authors_state
        fig, axs = plt.subplots(1, len(groups), figsize=(6.6 * len(groups), 4.2), squeeze=False)
        for gi, g in enumerate(groups):
            ax = axs[0, gi]
            tests = [(d["label"], d) for (gg, key), d in data.items() if gg == g]
            rows, mat = [], []
            for src in ("SV", "DeepMIMO"):
                A = A_dm if src == "DeepMIMO" else A_sv
                for snr, band in ((5.0, "0to10dB"), (15.0, "10to20dB")):
                    try:
                        Bs, th, _ = authors_state(authors, True, f"{src}_{g}", band)
                    except FileNotFoundError:
                        continue
                    row = []
                    for lab, d in tests:
                        x = d["test"]
                        y = measurements.gen_measurements_matlab(x, A, snr, NumpyMatlabRNG(2026 + int(snr)))
                        xh = authors_net(y, A, Bs, th, True)
                        v = 10 * np.log10(np.sum(np.abs(xh - x) ** 2) / np.sum(np.abs(x) ** 2))
                        tr[f"GM-LAMP({src} {g}, {band}) on {lab} at {snr:g} dB"] = float(v)
                        row.append(v)
                    rows.append(f"{src}-trained, {band.replace('to', '-').replace('dB', ' dB')}\ntested at {snr:g} dB")
                    mat.append(row)
            if not mat:
                ax.set_visible(False); continue
            M_ = np.array(mat)
            im = ax.imshow(M_, cmap="RdYlBu_r", vmin=-26, vmax=2, aspect="auto")
            for i in range(M_.shape[0]):
                for j in range(M_.shape[1]):
                    ax.text(j, i, f"{M_[i, j]:.1f}", ha="center", va="center", fontsize=9,
                            color="white" if M_[i, j] < -18 else "#222")
            ax.set_xticks(range(len(tests))); ax.set_xticklabels([lab.replace(" (", "\n(") for lab, _ in tests], fontsize=8)
            ax.set_yticks(range(len(rows))); ax.set_yticklabels(rows, fontsize=7.5)
            ax.set_xlabel("test set (2000 channels)")
            ax.set_title(f"{g}: NMSE (dB) of the authors' GM-LAMP networks", fontsize=9)
            fig.colorbar(im, ax=ax, fraction=0.046, pad=0.03, label="NMSE (dB)")
        fig.tight_layout(); fig.savefig(out / "transfer_gmlamp.png", dpi=150); plt.close(fig)
    rep["transfer_gmlamp"] = tr

    (out / "check_3gpp.json").write_text(json.dumps(rep, indent=1, default=str), encoding="utf-8")
    md = [f"# 3GPP data check ({scen}, {fc:.0f} GHz, 38.901 {ver} tables)\n", f"Generator: {impl}.\n",
          "## Standard conformance\n",
          f"LOS probability: generated {obs_all:.3f}, expected {exp_all:.3f} (binomial sd {sd_all:.3f}).\n"]
    paths = any(np.isfinite(r_.get("realized_median", np.nan)) for r_ in lsp_rows)   # QuaDRiGa: spreads of the paths
    md += ["| state | LSP | n | median drawn | median 38.901 | offset (dec) | log-sd drawn | log-sd 38.901 |"
           + (" median of paths |" if paths else ""), "|---|---|---|---|---|---|---|---|" + ("---|" if paths else "")]
    for r_ in lsp_rows:
        md.append(f"| {r_['state']} | {r_['lsp']} ({UNIT[r_['lsp']]}) | {r_['n']} | {r_['median']:.3g} | {r_['table_median']:.3g} | "
                  f"{r_['log_offset']:+.3f} | {r_['log_rstd']:.3f} | {r_['table_log_sigma']:.3f} |"
                  + (f" {r_.get('realized_median', float('nan')):.3g} |" if paths else ""))
    if "K_LOS_dB" in std:
        k_ = std["K_LOS_dB"]
        md.append(f"\nK-factor (LOS): median {k_['median']:.2f} dB, sd {k_['rstd']:.2f} dB; 38.901: 9 dB, 5 dB.\n")
    md += ["## Beamspace (test sets)\n", "| data set | beams for 95 % (median, p10-p90) | top-1 energy | top-16 energy |",
           "|---|---|---|---|"]
    for k_, v_ in bs.items():
        md.append(f"| {k_} | {v_['beams95_median']:.0f} ({v_['beams95_p10']:.0f}-{v_['beams95_p90']:.0f}) | "
                  f"{v_['top1_energy']:.3f} | {v_['top16_energy']:.3f} |")
    md += ["\n## LMMSE NMSE (dB)\n", "| data set | " + " | ".join(f"{s:g} dB" for s in a.snr) + " |",
           "|---|" + "---|" * len(a.snr)]
    for k_, v_ in lm.items():
        md.append(f"| {k_} | " + " | ".join(f"{y:.2f}" for y in v_["nmse_db"]) + " |")
    if tr:
        md += ["\n## Transfer of the authors' GM-LAMP (NMSE, dB)\n", "| network, test set, SNR | NMSE |", "|---|---|"]
        md += [f"| {k_} | {v_:.2f} |" for k_, v_ in tr.items()]
    (out / "check_3gpp.md").write_text("\n".join(md) + "\n", encoding="utf-8")
    print("\n".join(md))
    print(f"\nwritten: {out}")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Raw 3GPP TR 38.901 channels with Sionna (the standard's procedure as written), saved in parts as H (N x K complex).

  python scripts/sionna_make_H.py --array UPA   --n 100000 --out data/raw3gpp/H_3gpp_upa
  python scripts/sionna_make_H.py --array ULA_Y --n 100000 --out data/raw3gpp/H_3gpp_ulay
  then (bsce environment):
  python scripts/make_dataset.py --dataset 3GPP --array UPA --input "data/raw3gpp/H_3gpp_upa_part*.mat" ...

Needs Sionna 2.2 without its ray tracer, best in its own environment (PyTorch >= 2.9.1; the CPU build is enough):
  conda create -n sionna python=3.12 -y  &  conda activate sionna  &  pip install sionna-no-rt==2.2.0 scipy
This script imports only numpy, scipy, torch and sionna (no bsce module), so it runs in that environment.

The setting is that of matlab/quadriga_make_H.m and docs/3GPP_DATASET.md:
  model     38.901 UMi street canyon (sionna.phy.channel.tr38901.UMi), tables of --spec 16.1 (default) or 19.2;
            LOS/NLOS per user by Table 7.4.2-1 (--scenario umi), or all LOS / all NLOS (umi_los, umi_nlos)
  carrier   --fc 28e9 (= DeepMIMO O1_28)
  BS        height 10 m at the origin, boresight +x, single-polarised (V) panel with the 38.901 element of Table 7.3-1
            (--element omni = isotropic), lambda/2: UPA 16 x 16 (y-z), ULA_Y 1 x 256 (y), ULA 256 x 1 (z)
  UE        one isotropic antenna, 1.5 m, outdoor (--indoor = fraction of O2I users, low-loss model)
  users     uniform in the 120-degree hexagonal sector cell of the site (ISD 200 m), 2-D distance >= 10 m
  channel   narrowband at the carrier: h = sum of the path coefficients; plane waves; no path loss and no shadowing
            here (bsce normalises ||x||^2 = N per channel anyway)
Element order of H: horizontal (y) index slow, vertical (z) index fast, as quadriga_make_H.m.
Drops of --drop-users users (new topology and LSPs per drop); drop d uses the seed (--seed, d), so any part can be made
alone (--parts 1-10 here, 11-20 on another computer). Existing parts made with the same settings are kept (resume).

Saved per part (<out>_part001.mat, ...): H (N x Kp), loc (Kp x 3), los, indoor, drop (Kp x 1),
lsp (Kp x 7: DS [s], ASD, ASA, ZSD, ZSA [deg], K-factor [dB], SF [dB], as drawn by Sionna), lsp_real (NaN), lsp_names,
params (the settings; num_ant_x/y/z as matlab/deepmimo_make_H.m).
"""
import argparse
import sys
import time
from pathlib import Path

import numpy as np
import scipy.io as sio

GEOM = {"UPA": (16, 16, (1, 16, 16)), "ULA_Y": (1, 256, (1, 256, 1)), "ULA": (256, 1, (1, 1, 256))}
SAME = ("array", "scenario", "fc", "element", "isd", "d_min", "h_bs", "h_ut", "indoor_frac", "drop_users", "part_users",
        "seed", "spec")


def hex_sector(rng, n, isd, d_min):
    """n points uniform in the hexagonal sector cell with the BS at a corner (120 degrees around +x), d2D >= d_min."""
    r = isd / 3
    out = np.zeros((2, 0))
    while out.shape[1] < n:
        m = 4 * n
        c = np.vstack([2 * r * rng.random(m), (2 * rng.random(m) - 1) * r * np.sqrt(3) / 2])
        q = np.abs(c - np.array([[r], [0.0]]))
        ok = (q[1] <= r * np.sqrt(3) / 2) & (np.sqrt(3) * q[0] + q[1] <= np.sqrt(3) * r) & (np.sum(c ** 2, 0) >= d_min ** 2)
        out = np.hstack([out, c[:, ok]])
    return out[:, :n]


def parse_parts(text, n_parts):
    if not text:
        return list(range(1, n_parts + 1))
    out = []
    for tok in str(text).replace(",", " ").split():
        lo, _, hi = tok.partition("-")
        out += list(range(int(lo), int(hi or lo) + 1))
    return [p for p in out if 1 <= p <= n_parts]


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--array", choices=list(GEOM), required=True)
    ap.add_argument("--n", type=int, default=100000)
    ap.add_argument("--out", required=True, help="prefix of the part files, e.g. data/raw3gpp/H_3gpp_upa")
    ap.add_argument("--scenario", choices=["umi", "umi_los", "umi_nlos"], default="umi")
    ap.add_argument("--spec", choices=["16.1", "19.2"], default="16.1",
                    help="38.901 tables: 16.1 (as QuaDRiGa 2.8.1) or 19.2 (the Rel-19 update; Sionna's own default)")
    ap.add_argument("--fc", type=float, default=28e9)
    ap.add_argument("--element", choices=["38.901", "omni"], default="38.901")
    ap.add_argument("--isd", type=float, default=200.0)
    ap.add_argument("--d-min", type=float, default=10.0)
    ap.add_argument("--h-bs", type=float, default=10.0)
    ap.add_argument("--h-ut", type=float, default=1.5)
    ap.add_argument("--indoor", type=float, default=0.0, help="fraction of O2I users (38.901 calibration: 0.8)")
    ap.add_argument("--drop-users", type=int, default=250)
    ap.add_argument("--part-users", type=int, default=5000)
    ap.add_argument("--seed", type=int, default=2026)
    ap.add_argument("--parts", default="", help="e.g. 1-10 or '1 3 5' (default: all)")
    ap.add_argument("--device", default="cpu", help="cpu or cuda")
    ap.add_argument("--threads", type=int, default=0, help="CPU threads for PyTorch (0 = PyTorch's default)")
    a = ap.parse_args()
    if a.part_users % a.drop_users:
        sys.exit("--part-users must be a multiple of --drop-users")
    import torch
    if a.threads:
        torch.set_num_threads(a.threads)
    try:
        import sionna
        from sionna.phy import config
        from sionna.phy.channel.tr38901 import UMi, PanelArray
    except ImportError as e:
        sys.exit(f"Sionna is not installed in this environment ({e}).\n  pip install sionna-no-rt==2.2.0 scipy  "
                 "(PyTorch >= 2.9.1; see the help of this script)")
    rows, cols, geo = GEOM[a.array]
    bs = PanelArray(num_rows_per_panel=rows, num_cols_per_panel=cols, polarization="single", polarization_type="V",
                    antenna_pattern=a.element, carrier_frequency=a.fc, precision="double", device=a.device)
    ut = PanelArray(num_rows_per_panel=1, num_cols_per_panel=1, polarization="single", polarization_type="V",
                    antenna_pattern="omni", carrier_frequency=a.fc, precision="double", device=a.device)
    pos = bs.ant_pos.cpu().numpy()
    order = np.lexsort((pos[:, 2], pos[:, 1]))          # y slow, z fast (the element order of quadriga_make_H.m)
    def new_model():      # a new model per drop: Sionna keeps random-generator state inside a model between calls
        return UMi(carrier_frequency=a.fc, o2i_model="low", ut_array=ut, bs_array=bs, direction="downlink",
                   enable_pathloss=False, enable_shadow_fading=False, precision="double", device=a.device,
                   spec_version=a.spec)
    los_ctl = {"umi": None, "umi_los": True, "umi_nlos": False}[a.scenario]
    N = rows * cols
    n_parts = int(np.ceil(a.n / a.part_users))
    params = dict(generator="bsce scripts/sionna_make_H.py", implementation=f"Sionna {sionna.__version__} "
                  f"(sionna.phy.channel.tr38901.UMi, 3GPP TR 38.901 v{a.spec})", array=a.array,
                  num_ant_x=geo[0], num_ant_y=geo[1], num_ant_z=geo[2], ant_spacing=0.5,
                  scenario={"umi": "3GPP_38.901_UMi", "umi_los": "3GPP_38.901_UMi_LOS", "umi_nlos": "3GPP_38.901_UMi_NLOS"}[a.scenario],
                  fc=a.fc, element="3gpp" if a.element == "38.901" else "omni", isd=a.isd, d_min=a.d_min, h_bs=a.h_bs,
                  h_ut=a.h_ut, indoor_frac=a.indoor, drop_users=a.drop_users, part_users=a.part_users, seed=a.seed,
                  K_total=a.n, narrowband=1, ue_antenna="omni", spec=a.spec, table="v16" if a.spec == "16.1" else "v19",
                  zsd_fix=1, realized=0)
    lsp_names = np.array(["DS_s", "ASD_deg", "ASA_deg", "ZSD_deg", "ZSA_deg", "KF_dB", "SF_dB"], dtype=object)
    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    parts = parse_parts(a.parts, n_parts)
    print(f"3GPP UMi ({a.scenario}, 38.901 v{a.spec}, Sionna {sionna.__version__}), {a.array}, {a.fc / 1e9:.1f} GHz, "
          f"{a.element} elements: {a.n:,} channels in {n_parts} part(s) of {a.part_users} (drops of {a.drop_users})")
    t_all, made = time.time(), 0
    for p in parts:
        f = Path(f"{out}_part{p:03d}.mat")
        if f.exists():
            old = sio.loadmat(f, variable_names=["params"])["params"]
            bad = []
            for k in SAME:
                v = np.squeeze(old[k][0, 0]) if k in old.dtype.names else "(not stored)"
                v = v.item() if np.ndim(v) == 0 and hasattr(v, "item") else str(v)
                if str(v) != str(params[k]) and not (isinstance(v, float) and np.isclose(v, float(params[k]))):
                    bad.append(f"{k} = {v} (now {params[k]})")
            if bad:
                sys.exit(f"{f} was made with other settings: {'; '.join(bad)}. Delete the old parts or use another --out.")
            print(f"part {p}: {f} exists, kept")
            continue
        first = (p - 1) * a.part_users
        Kp = min(a.part_users, a.n - first)
        if Kp <= 0:
            continue
        t0 = time.time()
        H = np.zeros((N, Kp), complex)
        loc, los, ind, drop = np.zeros((Kp, 3)), np.zeros(Kp), np.zeros(Kp), np.zeros(Kp)
        lsp = np.full((Kp, 7), np.nan)
        for k0 in range(0, Kp, a.drop_users):
            n = min(a.drop_users, Kp - k0)
            d = (first + k0) // a.drop_users + 1                       # global drop number
            rng = np.random.default_rng([a.seed, d])
            config.seed = int(a.seed) * 1_000_003 + d
            model = new_model()
            xy = hex_sector(rng, n, a.isd, a.d_min)
            indoor = rng.random(n) < a.indoor
            ut_loc = torch.tensor(np.vstack([xy, np.full(n, a.h_ut)]).T[None], dtype=torch.float64, device=a.device)
            z_ut = torch.zeros(1, n, 3, dtype=torch.float64, device=a.device)
            model.set_topology(ut_loc=ut_loc, bs_loc=torch.tensor([[[0.0, 0.0, a.h_bs]]], dtype=torch.float64, device=a.device),
                               ut_orientations=z_ut, bs_orientations=torch.zeros(1, 1, 3, dtype=torch.float64, device=a.device),
                               ut_velocities=z_ut, in_state=torch.tensor(indoor[None], device=a.device), los=los_ctl)
            coef = model(1, 1.0)[0]                       # [batch, UT, UT ant, BS, BS ant, paths, time]
            h = coef[0, :, 0, 0, :, :, 0].sum(-1).cpu().numpy()
            ii = slice(k0, k0 + n)
            H[:, ii] = h[:, order].T
            loc[ii] = ut_loc[0].cpu().numpy()
            los[ii] = model._scenario.los[0, 0].cpu().numpy()
            ind[ii] = indoor
            drop[ii] = d
            L = model._lsp
            g = lambda t: t[0, 0].detach().cpu().numpy().astype(float)  # noqa: E731
            lsp[ii] = np.c_[g(L.ds), g(L.asd), g(L.asa), g(L.zsd), g(L.zsa),
                            10 * np.log10(np.maximum(g(L.k_factor), 1e-10)), 10 * np.log10(np.maximum(g(L.sf), 1e-30))]
            lsp[ii][~(los[ii] > 0.5), 5] = -100.0                       # no K-factor for NLOS users
        sio.savemat(f, dict(H=H, loc=loc, los=los[:, None], indoor=ind[:, None], drop=drop[:, None], lsp=lsp,
                            lsp_real=np.full((Kp, 5), np.nan), lsp_names=lsp_names, params=params))
        made += Kp
        dt = time.time() - t0
        left = sum(1 for q in parts if q > p) * a.part_users
        print(f"part {p} / {n_parts}: {Kp} channels, LOS {los.mean():.2f}, {dt:.0f} s ({1000 * dt / Kp:.1f} ms per channel)"
              f" -> {f}" + (f"\n          about {(time.time() - t_all) / made * left / 60:.0f} min left" if left else ""))


if __name__ == "__main__":
    main()

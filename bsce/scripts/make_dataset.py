#!/usr/bin/env python3
"""ONE command per data set: channels (train / val / test) + measurement files for any SNR bands + a scenario file.

SV (Python only, ~1 min for 200k):
  python scripts/make_dataset.py --dataset SV --array UPA --n 200000 --snr 0 20 --name SV_UPA_200k
DeepMIMO (after MATLAB has written the raw channels H, see matlab/deepmimo_make_H_chunks.m):
  python scripts/make_dataset.py --dataset DeepMIMO --array UPA --input H_upa_part*.mat --n 200000 --snr 0 20 --name DeepMIMO_UPA_200k
3GPP TR 38.901 UMi, 28 GHz (docs/3GPP_DATASET.md; scripts/make_3gpp_data.py makes the standard 3GPP_UPA / 3GPP_ULA):
  python scripts/make_dataset.py --dataset 3GPP --array UPA --sizes sv --authors-bands --name 3GPP_UPA_80k
         (built-in generator, Python only, ~1.5 min per 100k)
  python scripts/make_dataset.py --dataset 3GPP --array UPA --input "raw3gpp/H_3gpp_upa_part*.mat" --n-train 96000
         --n-val 2000 --n-test 2000 --authors-bands --name 3GPP_UPA_100k
         (raw channels of scripts/sionna_make_H.py or matlab/quadriga_make_H.m instead)

Options
  --n 200000 --split 0.90 0.05 0.05     total number of channels and train/val/test fractions (default 0.90/0.05/0.05)
  --n-train / --n-val / --n-test         exact sizes instead (override --n/--split)
  --sizes sv|deepmimo|dong2019|100k      a preset of (train, val, test): sv 80000/2000/2000 (GM-LAMP's SV), deepmimo
                                         50000/2000/2000 (GM-LAMP's DeepMIMO), dong2019 81000/9000/19000 (Dong et al.,
                                         IEEE JSTSP 2019: 3GPP TR 38.901 CDL, 28 GHz), 100k 96000/2000/2000
  --test standard|random                 standard (default): the 2000 test channels every bsce/GM-LAMP result uses
                                         (SV: bsce's SV test set; DeepMIMO: GM-LAMP's CE_DeepMIMO test file, whose users
                                         are removed from train/val) -> results comparable with the paper tables.
                                         3GPP: held-out drops (no drop shared between the sets). Built-in generator:
                                         the test users are the same for any --n-train / --n-val (fixed drops).
                                         random: a fresh test split of size n_test from the same pool
  --snr 0 20 | 0 10 20 | -5 0 10 20 30   SNR band edges (dB): one measurement file pair (train + val) per band,
                                         100 SNR chunks ~ U(band); the SNR of every column is stored ('snr_db')
  --authors-bands                        instead of --snr: the bands of sv_*.yaml / deepmimo_*.yaml (GM-LAMP's protocol):
                                         0-10 and 10-20 dB, 10 SNR chunks per training file, validation at 10 dB
  --no-measurements                      channels only (train later with --noise online, or add bands later with
                                         scripts/make_sv_measurements.py --scenario <written yaml> --snr ...)
  --test-measurements                    also write test measurement files at the scenario's test SNRs (0:2:20 dB), with
                                         exactly the noise evaluate.py uses (for MATLAB tests)
  --noise-model colored|white|both       measurement noise of the files: colored y = A (x + n) (default), white
                                         y = A x + w (files <name>_white.mat, used by --noise-model white runs), or both
                                         (paired: the white files are the colored ones with the same draw whitened)
  --seed 2026  --out-dir data/<name>  --force
3GPP with the built-in generator (no --input; bsce/data/tr38901.py, the TR 38.901 procedure as written):
  --ula-axis y|z                         ULA on y (horizontal, default) or z (vertical, the geometry of GM-LAMP's
                                         DeepMIMO ULA); the UPA is 16 x 16 on y-z
  --spec 16.1|19.2  --channel umi|umi_los|umi_nlos  --element 38.901|omni  --drop-users 250
  --raw-out PREFIX                       also write the raw channels (<PREFIX>_part001.mat, ...; check_3gpp.py --raw)

Writes data/<name>/
  <D>_<ARR>_{training,validation,test}_channel.mat     x (N x K), ||x||^2 = N on average (SV) / exactly (DeepMIMO)
  <D>_<ARR>_{training,validation}_128_measurements_<band>.mat    y (M x K) + snr_db (1 x K)
  CSmatrix.mat (GM-LAMP's A: CSmatrix256128 for SV and 3GPP, CSmatrix_128256 for DeepMIMO), dataset_info.json,
  dataset_check.png
and configs/scenarios/<name lower case>.yaml -> train with
  python scripts/train.py --scenario configs/scenarios/<name>.yaml --train configs/train/ladmm_best.yaml
(add --snr <edges> to train other bands than the ones made here; missing files are then made automatically).
"""
import argparse
import glob
import json
import sys
import time
from pathlib import Path

import numpy as np
import scipy.io as sio
import yaml

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))
from bsce.config import resolve, band_file_names  # noqa: E402
from bsce.data import sv, measurements  # noqa: E402
from bsce.data.files import bands_from_edges, ensure_band_files  # noqa: E402
from bsce.rng import NumpyMatlabRNG  # noqa: E402

REF = ROOT / "reference" / "gmlamp"
CS = {"SV": REF / "CE_SV" / "CSmatrix256128.mat", "DeepMIMO": REF / "CE_DeepMIMO" / "CSmatrix_128256.mat",
      "3GPP": REF / "CE_SV" / "CSmatrix256128.mat"}
AUTHORS_BANDS = [dict(name="0to10dB", train=[0, 10], val_snr=[10], use_for=[0, 10]),
                 dict(name="10to20dB", train=[10, 20], val_snr=[10], use_for=[10, 20.01])]
STD_TEST = {"SV": 2000, "DeepMIMO": 2000}
SIZES = {"sv": (80000, 2000, 2000), "deepmimo": (50000, 2000, 2000), "dong2019": (81000, 9000, 19000),
         "100k": (96000, 2000, 2000)}
# built-in 3GPP generator: every set has its own drops (drop d uses the seed (seed, d)), so the validation and test users
# do not depend on the other sizes; order train < validation < test, as the held-out split of raw files
DROP_START = {"training": 1, "validation": 1_000_001, "test": 2_000_001}
TEST_SNR = [0, 2, 4, 6, 8, 10, 12, 14, 16, 18, 20]


def need(path: Path, what: str):
    if not path.exists():
        sys.exit(f"missing {path}\n  ({what})\n  Extract bsce_reference_part1.zip, part2.zip and part3.zip where you extracted "
                 f"bsce_code.zip (paths start with bsce/), or copy reference/gmlamp from an older bsce folder, and run again.")


def sizes(a, pool=None):
    if getattr(a, "sizes", None):
        return SIZES[a.sizes]
    if a.n_train:
        n_tr, n_va, n_te = a.n_train, a.n_val or 0, a.n_test or 0
    else:
        f = np.asarray(a.split, float)
        if f.size != 3 or abs(f.sum() - 1) > 1e-6:
            sys.exit("--split needs three fractions that sum to 1, e.g. 0.90 0.05 0.05")
        n_tr, n_va = int(round(a.n * f[0])), int(round(a.n * f[1]))
        n_te = a.n - n_tr - n_va
    return n_tr, n_va, n_te


def make_sv(a, n_tr, n_va, n_te):
    arr = a.array
    gen = lambda seed, K: sv.beamspace_channels(NumpyMatlabRNG(seed), K, arr, 3, 16, 16)
    t0 = time.time()
    # bsce's SV test set = generate_data.py with seed 2026, part index 2 (GM-LAMP generator, MATLAB random order)
    if a.test == "standard":
        test = gen(2026 + 2, STD_TEST["SV"])
        test_note = "standard bsce SV test set (seed 2028, 2000 channels; same as data/SV_<ARR>/SV_<ARR>_test_channel.mat)"
        base = a.seed + 10_000                 # train/val streams independent of the test stream
    else:
        test = None
        base = a.seed
    tr = gen(base + 0, n_tr)
    va = gen(base + 1, n_va)
    if test is None:
        test = gen(base + 2, n_te)
        test_note = f"random test set (seed {base + 2}, {n_te} channels)"
    print(f"SV {arr}: {n_tr:,} / {n_va:,} / {test.shape[1]:,} channels in {time.time() - t0:.0f} s")
    return {"training": tr, "validation": va, "test": test}, {"test": test_note, "train_seed": base, "val_seed": base + 1}


def make_deepmimo(a, n_tr, n_va, n_te):
    from deepmimo_convert import load_channels, u0_1d, match_reference
    files = sorted({f for pat in a.input for f in (glob.glob(pat) or [pat])})
    if not files:
        sys.exit("--input: no DeepMIMO channel file found")
    N = 256
    H = np.concatenate([load_channels(f, None, N) for f in files], axis=1)
    U0 = u0_1d(N) if a.array == "ULA" else np.kron(u0_1d(16), u0_1d(16))
    x = U0.conj().T @ H                                  # GM-LAMP's DeepMIMO beamspace (deepmimo_data_gmlamp_equivalence)
    del H
    e = np.sum(np.abs(x) ** 2, 0)
    x = x[:, e > 0] * np.sqrt(N / e[e > 0])[None, :]
    # duplicate users (overlapping row ranges) -> keep one copy
    key = np.round(np.concatenate([x[:8].real, x[:8].imag]).T, 8)          # (K, 16) fingerprint of each user
    _, uniq = np.unique(key, axis=0, return_index=True)
    x = x[:, np.sort(uniq)]
    print(f"DeepMIMO {a.array}: {x.shape[1]:,} distinct users from {len(files)} file(s)")
    rng = np.random.default_rng(a.seed)
    info = {"input_files": files, "users": int(x.shape[1])}
    if a.test == "standard":
        ref_f = REF / "CE_DeepMIMO" / f"{a.array}_test_channel.mat"
        need(ref_f, "GM-LAMP's DeepMIMO test channels")
        ref = sio.loadmat(ref_f)["x"]
        idx, fid, rel = match_reference(ref, x)
        ok = rel < 1e-3
        info["gmlamp_test_found"] = f"{int(ok.sum())} / {ref.shape[1]}"
        print(f"  GM-LAMP test channels found among the users: {int(ok.sum())} / {ref.shape[1]}")
        if ok.mean() < 0.99:
            sys.exit("GM-LAMP's test users are not in the input: the input must contain rows 1000-1300 of O1_28, BS 3 "
                     "(matlab/deepmimo_make_H_chunks.m), with the array of GM-LAMP (ULA on the z axis, UPA y-z). "
                     "Use --test random for other data.")
        pool = np.setdiff1d(np.arange(x.shape[1]), np.unique(idx))
        test = ref
        info["test"] = "GM-LAMP CE_DeepMIMO test file (identical channels); those users are excluded from train/val"
    else:
        pool = np.arange(x.shape[1])
        test = None
    perm = pool[rng.permutation(pool.size)]
    want = n_tr + n_va + (0 if test is not None else n_te)
    if perm.size < want:
        sys.exit(f"only {perm.size:,} usable users for {want:,} requested channels. Generate more rows in MATLAB "
                 f"(181 users per row: rows 1000-{1000 + int(np.ceil((want + (2000 if test is not None else 0)) / 181)) + 4}; "
                 f"matlab/deepmimo_make_H_chunks.m does this), "
                 "or ask for fewer (--n), or train with --noise online (fresh noise every batch).")
    parts = {"training": x[:, perm[:n_tr]], "validation": x[:, perm[n_tr:n_tr + n_va]]}
    parts["test"] = test if test is not None else x[:, perm[n_tr + n_va:want]]
    if test is None:
        info["test"] = f"random split ({n_te} users)"
    return parts, info


def load_3gpp(files, arr):
    """Raw channels of scripts/sionna_make_H.py / matlab/quadriga_make_H.m: H (N x K) and the per-user fields, in
    generation (drop) order."""
    keys = ("H", "los", "indoor", "drop", "loc", "lsp", "lsp_real")
    got = {k: [] for k in keys}
    geo = set()
    for f in files:
        d = sio.loadmat(f)
        if "H" not in d:
            sys.exit(f"{f}: no variable H (not a file of sionna_make_H.py / quadriga_make_H.m)")
        H = np.asarray(d["H"], np.complex128)
        if H.shape[0] != 256:
            sys.exit(f"{f}: H has {H.shape[0]} rows, 256 expected")
        K = H.shape[1]
        got["H"].append(H)
        for k in keys[1:]:
            v = np.asarray(d[k]) if k in d else np.full((K, 1), np.nan)
            got[k].append(v.reshape(K, -1).astype(float))
        pm = d.get("params")
        if pm is not None:
            g = tuple(int(np.squeeze(pm[k][0, 0])) for k in ("num_ant_x", "num_ant_y", "num_ant_z"))
            geo.add(g)
            p0 = {k: np.squeeze(pm[k][0, 0]).item() if np.size(pm[k][0, 0]) == 1 else str(np.squeeze(pm[k][0, 0]))
                  for k in pm.dtype.names}
    if len(geo) > 1:
        sys.exit(f"--input mixes array geometries {sorted(geo)}: one array per data set")
    g = geo.pop() if geo else None
    want = {"UPA": [(1, 16, 16)], "ULA": [(1, 1, 256), (1, 256, 1)]}[arr]
    if g is not None and g not in want:
        sys.exit(f"--array {arr} but the raw files hold the (x, y, z) = {g} array")
    out = {k: np.concatenate(v, axis=1 if k == "H" else 0) for k, v in got.items()}
    order = np.argsort(out["drop"][:, 0], kind="stable")       # generation order: drop 1, 2, ...
    out = {k: (v[:, order] if k == "H" else v[order]) for k, v in out.items()}
    params = {k: (v.strip() if isinstance(v, str) else v) for k, v in (p0.items() if g is not None else [])}
    params["array_axes"] = {(1, 16, 16): "UPA y-z", (1, 1, 256): "ULA z (vertical)", (1, 256, 1): "ULA y (horizontal)"}.get(g, "?")
    return out, params


def make_3gpp(a, n_tr, n_va, n_te):
    from deepmimo_convert import u0_1d
    files = sorted({f for pat in a.input for f in (glob.glob(pat) or [pat])})
    if not files:
        sys.exit("--input: no 3GPP channel file found (scripts/sionna_make_H.py and matlab/quadriga_make_H.m write "
                 "<prefix>_part001.mat, ...)")
    raw, params = load_3gpp(files, a.array)
    N = 256
    U0 = u0_1d(N) if a.array == "ULA" else np.kron(u0_1d(16), u0_1d(16))
    x = U0.conj().T @ raw.pop("H")                       # the lens convention of the DeepMIMO data (GM-LAMP)
    e = np.sum(np.abs(x) ** 2, 0)
    ok = e > 0
    x = x[:, ok] * np.sqrt(N / e[ok])[None, :]           # ||x||^2 = N per channel (removes path loss + shadowing)
    raw = {k: v[ok] for k, v in raw.items()}
    K = x.shape[1]
    print(f"3GPP {a.array} ({params.get('array_axes')}): {K:,} channels from {len(files)} file(s), "
          f"{params.get('scenario', '?')}, {float(params.get('fc', np.nan)) / 1e9:.1f} GHz, LOS {raw['los'].mean():.3f}")
    want = n_tr + n_va + n_te
    if K < want:
        sys.exit(f"only {K:,} channels for {want:,} requested: generate more parts (matlab/quadriga_make_H.m, K = {want}) "
                 f"or ask for fewer (--n-train / --n-val / --n-test)")
    if a.test == "standard":                             # held-out drops: test = the last drops, val before them
        idx = {"training": np.arange(n_tr), "validation": np.arange(K - n_te - n_va, K - n_te),
               "test": np.arange(K - n_te, K)}
        how = "held-out drops (validation and test are the last drops generated; training the first users)"
    else:
        perm = np.random.default_rng(a.seed).permutation(K)
        idx = {"training": perm[:n_tr], "validation": perm[n_tr:n_tr + n_va], "test": perm[n_tr + n_va:want]}
        how = "random users"
    drops = {k: set(raw["drop"][v, 0].astype(int).tolist()) for k, v in idx.items()}
    shared = {f"{p}-{q}": len(drops[p] & drops[q]) for p, q in (("training", "validation"), ("training", "test"),
                                                                ("validation", "test"))}
    parts = {k: x[:, v] for k, v in idx.items()}
    info = {"input_files": files, "channels": int(K), "generator": params, "split": how,
            "drops_shared_between_sets": shared,
            "los_fraction": {k: float(raw["los"][v].mean()) for k, v in idx.items()},
            "indoor_fraction": {k: float(np.nanmean(raw["indoor"][v])) for k, v in idx.items()},
            "test": f"{how}; {n_te} channels"}
    print(f"  split: {how}; drops shared {shared}")
    return parts, info


def make_3gpp_builtin(a, n_tr, n_va, n_te):
    """3GPP channels with the built-in generator (bsce/data/tr38901.py), held-out drops per set."""
    from deepmimo_convert import u0_1d
    from bsce.data import tr38901
    arr = "UPA" if a.array == "UPA" else ("ULA_Y" if a.ula_axis == "y" else "ULA")
    kw = dict(spec=a.spec, scenario=a.channel, element=a.element)
    N = 256
    U0 = u0_1d(N) if a.array == "ULA" else np.kron(u0_1d(16), u0_1d(16))
    parts, meta, raw_all = {}, {}, []
    t0 = time.time()
    sets = (("training", n_tr), ("validation", n_va), ("test", n_te))
    if a.test == "random":
        n_all = n_tr + n_va + n_te
        r = tr38901.generate(n_all, arr, seed=a.seed, drop_users=a.drop_users, first_drop=1,
                             progress=tr38901.Timer(f"3GPP {arr}"), **kw)
        perm = np.random.default_rng(a.seed).permutation(n_all)
        cuts = np.cumsum([0, n_tr, n_va, n_te])
        res = {k: {f: (v[:, perm[cuts[i]:cuts[i + 1]]] if f == "H" else v[perm[cuts[i]:cuts[i + 1]]])
                   for f, v in r.items()} for i, (k, _) in enumerate(sets)}
        how = "random users of one generated pool"
    else:
        res = {k: tr38901.generate(n, arr, seed=a.seed, drop_users=a.drop_users, first_drop=DROP_START[k],
                                   progress=tr38901.Timer(f"3GPP {arr} {k}"), **kw) for k, n in sets}
        how = (f"held-out drops: training from drop {DROP_START['training']}, validation from drop "
               f"{DROP_START['validation']:,}, test from drop {DROP_START['test']:,} (the same users for any size)")
    for k, _ in sets:
        x = U0.conj().T @ res[k]["H"]
        x *= np.sqrt(N / np.sum(np.abs(x) ** 2, 0))[None, :]
        parts[k] = x
        for f in ("loc", "los", "drop", "lsp"):
            meta[f"{k}_{f}"] = np.asarray(res[k][f], float)
        raw_all.append(res[k])
    params = tr38901.params_of(arr, n_tr + n_va + n_te, a.seed, a.drop_users, **{"spec": a.spec, "scenario": a.channel,
                                                                                   "element": a.element})
    if a.raw_out:
        cat = {f: (np.concatenate([r["H"] for r in raw_all], 1) if f == "H" else np.concatenate([r[f] for r in raw_all]))
               for f in ("H", "loc", "los", "indoor", "drop", "lsp")}
        Path(a.raw_out).parent.mkdir(parents=True, exist_ok=True)
        K = cat["H"].shape[1]
        for i, k0 in enumerate(range(0, K, 5000)):
            tr38901.save_raw(f"{a.raw_out}_part{i + 1:03d}.mat", cat, params, slice(k0, k0 + 5000))
        print(f"  raw channels written: {a.raw_out}_part001.mat ... ({K:,} channels)")
    drops = {k: set(np.asarray(res[k]["drop"]).astype(int).tolist()) for k, _ in sets}
    shared = {f"{p}-{q}": len(drops[p] & drops[q]) for p, q in (("training", "validation"), ("training", "test"),
                                                                ("validation", "test"))}
    print(f"3GPP {arr}: {n_tr:,} / {n_va:,} / {n_te:,} channels in {time.time() - t0:.0f} s; "
          f"LOS {np.mean(res['training']['los']):.3f} (training); split: {how}")
    info = {"generator": params, "split": how, "drops_shared_between_sets": shared,
            "los_fraction": {k: float(np.mean(res[k]["los"])) for k, _ in sets}, "test": f"{how}; {n_te} channels",
            "meta_file": f"3GPP_{a.array}_meta.mat (loc, los, drop, lsp of every channel, per set)"}
    info["_meta"] = meta
    return parts, info


def check_figure(parts, out, title, band_files):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(1, 3, figsize=(16, 4.4))
    pal = {"training": "#2a78d6", "validation": "#1f9e6e", "test": "#c23b6a"}
    for k, x in parts.items():
        e = np.sum(np.abs(x) ** 2, 0)
        lab = f"{k}: {x.shape[1]:,} ch., mean {e.mean():.1f}, std {e.std():.1f}"
        if np.ptp(e) < 1e-6 * max(e.mean(), 1):          # normalised channels (DeepMIMO): one value
            ax[0].axvline(e.mean(), color=pal[k], lw=1.4, label=lab)
        else:
            ax[0].hist(e, bins=60, histtype="step", color=pal[k], lw=1.4, density=True, label=lab)
        pk = np.bincount(np.argmax(np.abs(x), 0), minlength=x.shape[0]) / x.shape[1]
        ax[1].plot(pk, color=pal[k], lw=1, label=f"{k}")
    tv = 0.5 * np.abs(np.bincount(np.argmax(np.abs(parts["training"]), 0), minlength=256) / parts["training"].shape[1]
                      - np.bincount(np.argmax(np.abs(parts["test"]), 0), minlength=256) / parts["test"].shape[1]).sum()
    ax[0].set_xlabel("||x||^2"); ax[0].set_title("channel energy (N = 256)", fontsize=9)
    ax[1].set_xlabel("index of the strongest beam")
    ax[1].set_title(f"strongest-beam histogram (TV distance train-test = {tv:.3f})", fontsize=9)
    for f, lab in band_files:
        d = sio.loadmat(f, variable_names=["snr_db"])
        if "snr_db" in d:
            s = np.ravel(d["snr_db"])
            ax[2].hist(s, bins=40, histtype="step", lw=1.4, label=f"{lab}: {s.min():.1f} .. {s.max():.1f} dB")
    ax[2].set_xlabel("SNR of the column (dB)"); ax[2].set_title("SNR of the measurement files", fontsize=9)
    for a_ in ax:
        a_.grid(alpha=0.3); a_.legend(frameon=False, fontsize=7.5)
        for s_ in ("top", "right"):
            a_.spines[s_].set_visible(False)
    fig.suptitle(title, fontsize=10); fig.tight_layout(); fig.savefig(out / "dataset_check.png", dpi=150)
    return float(tv)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dataset", choices=["SV", "DeepMIMO", "3GPP"], required=True)
    ap.add_argument("--array", choices=["ULA", "UPA"], required=True)
    ap.add_argument("--n", type=int, default=200000)
    ap.add_argument("--split", nargs=3, type=float, default=[0.90, 0.05, 0.05])
    ap.add_argument("--n-train", type=int, default=None)
    ap.add_argument("--n-val", type=int, default=None)
    ap.add_argument("--n-test", type=int, default=None)
    ap.add_argument("--sizes", choices=list(SIZES), default=None,
                    help="(train, val, test) preset: " + ", ".join(f"{k} {'/'.join(map(str, v))}" for k, v in SIZES.items()))
    ap.add_argument("--test", choices=["standard", "random"], default="standard")
    ap.add_argument("--snr", nargs="+", type=float, default=[0, 20])
    ap.add_argument("--authors-bands", action="store_true",
                    help="bands 0-10 / 10-20 dB, 10 SNR chunks, validation at 10 dB (sv_*.yaml, deepmimo_*.yaml)")
    ap.add_argument("--no-measurements", action="store_true")
    ap.add_argument("--test-measurements", action="store_true")
    ap.add_argument("--noise-model", choices=["colored", "white", "both"], default="colored",
                    help="measurement files: colored y = A (x + n) (default, as GM-LAMP), white y = A x + w (files "
                         "<name>_white.mat), or both (the white files are the colored ones with the same noise draw "
                         "whitened, so the two sets are paired; bsce/data/noise.py)")
    ap.add_argument("--input", nargs="+", default=None, help="DeepMIMO / 3GPP: raw channel .mat files (wildcards allowed)")
    ap.add_argument("--name", default=None)
    ap.add_argument("--out-dir", default=None)
    ap.add_argument("--seed", type=int, default=2026)
    ap.add_argument("--force", action="store_true")
    g3 = ap.add_argument_group("3GPP, built-in generator (no --input)")
    g3.add_argument("--ula-axis", choices=["y", "z"], default="y", help="ULA on y (horizontal) or z (vertical)")
    g3.add_argument("--spec", choices=["16.1", "19.2"], default="16.1", help="38.901 tables")
    g3.add_argument("--channel", choices=["umi", "umi_los", "umi_nlos"], default="umi",
                    help="LOS/NLOS by Table 7.4.2-1, or all LOS / all NLOS")
    g3.add_argument("--element", choices=["38.901", "omni"], default="38.901", help="BS element")
    g3.add_argument("--drop-users", type=int, default=250, help="users per drop")
    g3.add_argument("--raw-out", default=None, help="also write the raw channels to <PREFIX>_part001.mat, ...")
    a = ap.parse_args(argv)
    a.array = a.array.upper()
    n_tr, n_va, n_te = sizes(a)
    name = a.name or f"{a.dataset}_{a.array}_{(n_tr + n_va + n_te) // 1000}k"
    out = resolve(a.out_dir or f"data/{name}")
    pre = f"{a.dataset}_{a.array}"
    if (out / f"{pre}_training_channel.mat").exists() and not a.force:
        sys.exit(f"{out} already has channels; use --force to overwrite (or another --name)")
    need(CS[a.dataset], "GM-LAMP's CS matrix")
    if a.dataset == "DeepMIMO" and not a.input:
        sys.exit("DeepMIMO needs --input <raw channel files from matlab/deepmimo_make_H_chunks.m>")
    out.mkdir(parents=True, exist_ok=True)
    if a.force:
        # new channels make every old measurement file of this folder invalid (its y belongs to the old x), also for
        # bands not in this --snr; remove them so train.py remakes them instead of silently reusing them
        old = sorted(out.glob(f"{pre}_*_measurements_*.mat")) + sorted(out.glob(f"{pre}_*_measurements_*.mat.snr.json"))
        for f in old:
            f.unlink()
        if old:
            print(f"--force: removed {len(old)} old measurement file(s) of {out.name} (they belonged to the old channels)")
    if a.test == "standard" and a.dataset != "3GPP":
        print(f"train {n_tr:,} / val {n_va:,}; test = the standard 2000 channels (comparable with every earlier result). "
              f"For a {n_te:,}-channel test split from the same pool use --test random.")
    if a.dataset == "3GPP" and not a.input:
        parts, info = make_3gpp_builtin(a, n_tr, n_va, n_te)
    else:
        parts, info = {"SV": make_sv, "DeepMIMO": make_deepmimo, "3GPP": make_3gpp}[a.dataset](a, n_tr, n_va, n_te)
    meta = info.pop("_meta", None)
    if meta is not None:
        sio.savemat(out / f"{pre}_meta.mat", meta)
    A = sio.loadmat(CS[a.dataset])["A"].astype(np.float64)
    sio.savemat(out / "CSmatrix.mat", {"A": A})
    for k, x in parts.items():
        sio.savemat(out / f"{pre}_{k}_channel.mat", {"x": x})
        print(f"saved {pre}_{k}_channel.mat  {x.shape}")
    bands = [dict(b) for b in AUTHORS_BANDS] if a.authors_bands else bands_from_edges(a.snr)
    sc = dict(name=name, dataset=a.dataset, array=a.array, N1=16, N2=16, L=3, M=int(A.shape[0]),
              cs_matrix=str(CS[a.dataset].relative_to(ROOT)).replace("\\", "/"),
              data_dir=str(out.relative_to(ROOT)).replace("\\", "/") if out.is_relative_to(ROOT) else str(out),
              seed=a.seed, n_train=int(parts["training"].shape[1]), n_val=int(parts["validation"].shape[1]),
              n_test=int(parts["test"].shape[1]), n_snr_chunks=10 if a.authors_bands else 100, bands=bands,
              test_snr_db=TEST_SNR)
    if a.dataset == "3GPP":
        g = info.get("generator", {})
        sc["channel_model"] = {k: g.get(k) for k in ("implementation", "scenario", "fc", "array", "element", "isd", "d_min",
                                                      "h_bs", "h_ut", "drop_users") if k in g}
        sc["channel_model"]["axes"] = {(1, 16, 16): "UPA y-z", (1, 256, 1): "ULA y (horizontal)",
                                       (1, 1, 256): "ULA z (vertical)"}.get(
            tuple(int(g.get(k, 0)) for k in ("num_ant_x", "num_ant_y", "num_ant_z")), "?")
        sc["channel_model"]["split"] = info.get("split")
    band_files = []
    nms = ["colored", "white"] if a.noise_model == "both" else [a.noise_model]
    if not a.no_measurements:
        for nm in nms:                                  # colored first: the white files are made from them
            scn = dict(sc, noise_model=nm) if nm == "white" else sc
            for bi, b in enumerate(bands):
                fn = band_file_names(scn, b)
                ensure_band_files(scn, b, bi, out, A, fn, seed=a.seed, force=True)
                band_files += [(out / fn[0], f"train {b['name']}" + (" white" if nm == "white" else "")),
                               (out / fn[1], f"val {b['name']}" + (" white" if nm == "white" else ""))]
    if a.test_measurements:
        for nm in nms:
            sf = "_white" if nm == "white" else ""
            for i, s in enumerate(TEST_SNR):   # exactly the noise of evaluate.py (--noise-model)
                y = measurements.gen_measurements_matlab(parts["test"], A, float(s), NumpyMatlabRNG(a.seed + 1000 + i),
                                                         noise_model=nm)
                sio.savemat(out / f"{pre}_test_{sc['M']}_measurements_{s}dB{sf}.mat",
                            {"y": y, "snr_db": float(s), "noise_model": nm})
            print(f"saved test measurement files ({nm} noise) at {TEST_SNR} dB")
    tv = check_figure(parts, out, f"{name}: {sc['n_train']:,} / {sc['n_val']:,} / {sc['n_test']:,} channels, "
                                  f"bands {[b['name'] for b in bands]}", band_files)
    info.update(scenario=sc, strongest_beam_tv_train_test=tv, made=time.strftime("%Y-%m-%d %H:%M"))
    (out / "dataset_info.json").write_text(json.dumps(info, indent=1, default=str), encoding="utf-8")
    yml = ROOT / "configs" / "scenarios" / f"{name.lower()}.yaml"
    head = (f"# made by scripts/make_dataset.py ({time.strftime('%Y-%m-%d')}): {a.dataset} {a.array}, "
            f"{sc['n_train']} / {sc['n_val']} / {sc['n_test']} channels, test = {a.test}\n")
    if a.dataset == "3GPP":
        head += ("# 3GPP TR 38.901 UMi street canyon, 28 GHz; lens x = U0^H h, ||x||^2 = N per channel; "
                 "docs/3GPP_DATASET.md\n")
    yml.write_text(head + yaml.safe_dump(sc, sort_keys=False, default_flow_style=None), encoding="utf-8")
    print(f"\nscenario written: {yml.relative_to(ROOT)}")
    print(f"check: {out / 'dataset_check.png'}  (train-test strongest-beam TV {tv:.3f})")
    print(f"train: python scripts/train.py --scenario {yml.relative_to(ROOT).as_posix()} --train configs/train/ladmm_best.yaml")


if __name__ == "__main__":
    main()

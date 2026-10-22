#!/usr/bin/env python3
"""Turn raw DeepMIMO spatial channels h into GM-LAMP's beamspace data sets.

  python scripts/deepmimo_convert.py --scenario configs/scenarios/deepmimo_upa.yaml --input H_upa.mat
  then:  python scripts/generate_data.py --scenario configs/scenarios/deepmimo_upa.yaml --measurements-only

Raw channels: matlab/deepmimo_make_H.m (DeepMIMO v1.3, O1_28, BS 3, rows 1000-1300, 3 paths,
1 subcarrier; ULA = 256 elements on the z axis, UPA = 16 x 16 on the y, z axes).

GM-LAMP's beamspace convention for DeepMIMO (identified on its CE_DeepMIMO test files: every one
of its 2000 UPA test channels is found among these users with complex fidelity ~1 - 4e-11):
    x = U0^H h,   U0(n, k) = exp(j 2 pi n (k - (N-1)/2) / N) / sqrt(N),  n, k = 0..N-1
(element index from 0, symmetric beam grid; UPA: kron(U0_16, U0_16) on DeepMIMO's element order),
then ||x||^2 = N per channel. This differs from the SV lens matrix (centred element index), so
--lens gmlamp (default) is required to obtain GM-LAMP's data; --lens bsce gives x = U.' h.

Split (--test reference, default): the test set is GM-LAMP's own test file (identical channels),
its users are removed from the pool, and training / validation are drawn (seeded) from the rest:
50000 / 2000. --test random draws all three sets from the pool.
Outputs (data_dir): DeepMIMO_<ARR>_{training,validation,test}_channel.mat (x, N x K),
deepmimo_stats.json (match with GM-LAMP, statistics), deepmimo_stats.png.
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import scipy.io as sio

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from bsce.config import load_yaml, resolve, apply_overrides  # noqa: E402
from bsce.data import sv  # noqa: E402


def load_channels(path, var, N):
    d = sio.loadmat(path)
    if var is None:   # largest complex array
        cands = [(k, v) for k, v in d.items() if not k.startswith("__") and isinstance(v, np.ndarray)
                 and np.iscomplexobj(v)]
        if not cands:
            raise SystemExit(f"{path}: no complex array found; pass --var")
        var, H = max(cands, key=lambda kv: kv[1].size)
    else:
        H = d[var]
    H = np.squeeze(np.asarray(H, dtype=np.complex128))
    if H.ndim != 2:
        raise SystemExit(f"{path}:{var} has shape {H.shape}; expected N x K after squeezing")
    if H.shape[0] != N and H.shape[1] == N:
        H = H.T
    if H.shape[0] != N:
        raise SystemExit(f"{path}:{var} has shape {H.shape}; neither dimension is N = {N}")
    print(f"{Path(path).name}: variable '{var}', {H.shape[1]} channels of length {N}")
    return H


def beams95(x):
    p = np.sort(np.abs(x) ** 2, 0)[::-1]
    c = np.cumsum(p, 0) / np.sum(p, 0)
    return np.argmax(c >= 0.95, 0) + 1


def stats(x):
    b = beams95(x)
    return dict(beams95_median=float(np.median(b)), beams95_p10=float(np.percentile(b, 10)),
                beams95_p90=float(np.percentile(b, 90)), energy_mean=float(np.mean(np.sum(np.abs(x) ** 2, 0))),
                peak_median=float(np.median(np.argmax(np.abs(x), 0))))


def peak_tv(x, ref_x):
    """Total-variation distance between the strongest-beam histograms of x and ref_x."""
    h = lambda z: np.bincount(np.argmax(np.abs(z), 0), minlength=z.shape[0]) / z.shape[1]
    return float(0.5 * np.abs(h(x) - h(ref_x)).sum())


def u0_1d(n):
    """GM-LAMP's DeepMIMO lens matrix: element index 0..n-1, beam grid k - (n-1)/2."""
    k = np.arange(n) - (n - 1) / 2
    return np.exp(2j * np.pi * np.outer(np.arange(n), k) / n) / np.sqrt(n)


def match_reference(ref_x, x, chunk=500):
    """For every reference channel: best user in x by |<ref, x>| / (|ref||x|) (global phase free)."""
    R = ref_x / np.linalg.norm(ref_x, axis=0, keepdims=True)
    X = x / np.linalg.norm(x, axis=0, keepdims=True)
    idx = np.empty(R.shape[1], int); fid = np.empty(R.shape[1])
    for c in range(0, R.shape[1], chunk):
        C = np.abs(R[:, c:c + chunk].conj().T @ X)
        idx[c:c + chunk] = C.argmax(1); fid[c:c + chunk] = C.max(1)
    rel = np.linalg.norm(x[:, idx] - ref_x, axis=0) / np.linalg.norm(ref_x, axis=0)   # no phase alignment
    return idx, fid, rel


def transpose_upa(H, n1, n2):
    """Swap the two array axes of every channel (element order n1-major <-> n2-major)."""
    K = H.shape[1]
    return H.reshape(n1, n2, K).transpose(1, 0, 2).reshape(n1 * n2, K)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--scenario", required=True)
    ap.add_argument("--input", nargs="+", required=True, help=".mat file(s) with raw channels h")
    ap.add_argument("--var", default=None, help="variable name (default: largest complex array)")
    ap.add_argument("--n-train", type=int, default=None, help="default: scenario n_train or 50000")
    ap.add_argument("--n-val", type=int, default=None, help="default: scenario n_val or 2000")
    ap.add_argument("--n-test", type=int, default=None, help="default: scenario n_test or 2000 (--test random)")
    ap.add_argument("--lens", choices=["gmlamp", "bsce"], default="gmlamp",
                    help="gmlamp: x = U0^H h (GM-LAMP's DeepMIMO data); bsce: x = U.' h (SV lens matrix)")
    ap.add_argument("--conj", action="store_true", help="--lens bsce only: use conj(h)")
    ap.add_argument("--upa-order", choices=["as_is", "transpose"], default="as_is")
    ap.add_argument("--test", choices=["reference", "random"], default="reference",
                    help="reference: test set = GM-LAMP's test file, its users excluded from training")
    ap.add_argument("--reference", default=None, help="GM-LAMP beamspace test file "
                    "(default reference/gmlamp/CE_DeepMIMO/<ARR>_test_channel.mat)")
    ap.add_argument("--set", nargs="*", default=[])
    a = ap.parse_args()
    sc = apply_overrides(load_yaml(a.scenario), a.set)
    a.n_train = a.n_train or int(sc.get("n_train", 50000))
    a.n_val = a.n_val or int(sc.get("n_val", 2000))
    a.n_test = a.n_test or int(sc.get("n_test", 2000))
    arr, n1, n2 = sc["array"].upper(), int(sc["N1"]), int(sc["N2"])
    N = n1 * n2
    H = np.concatenate([load_channels(f, a.var, N) for f in a.input], axis=1)
    for f in a.input:            # array geometry stored by matlab/deepmimo_make_H.m
        try:
            pm = sio.loadmat(f, variable_names=["params"]).get("params")
            if pm is not None:
                g = tuple(int(np.squeeze(pm[k][0, 0])) for k in ("num_ant_x", "num_ant_y", "num_ant_z"))
                print(f"{Path(f).name}: DeepMIMO array (x, y, z) = {g}")
                want = (1, 1, N) if arr == "ULA" else (1, n1, n2)
                if g != want:
                    print(f"WARNING: GM-LAMP's {arr} is (x, y, z) = {want}; this file has {g}. "
                          "Regenerate with the current matlab/deepmimo_make_H.m.")
        except Exception:
            pass
    if arr == "UPA" and a.upa_order == "transpose":
        H = transpose_upa(H, n1, n2)
    if a.lens == "gmlamp":
        U0 = u0_1d(N) if arr == "ULA" else np.kron(u0_1d(n1), u0_1d(n2))
        x = U0.conj().T @ H
    else:
        x = sv.dft_matrix(arr, n1, n2).T @ (H.conj() if a.conj else H)
    e = np.sum(np.abs(x) ** 2, 0)
    keep = e > 0
    dropped = int(np.sum(~keep))
    x = x[:, keep] * np.sqrt(N / e[keep])[None, :]

    ref_file = Path(a.reference) if a.reference else ROOT / "reference/gmlamp/CE_DeepMIMO" / f"{arr}_test_channel.mat"
    ref_x = sio.loadmat(ref_file)["x"] if ref_file.exists() else None
    report = {"reference_file": str(ref_file), "lens": a.lens, "conj": bool(a.conj), "upa_order": a.upa_order if arr == "UPA" else None,
              "n_channels": int(x.shape[1]), "dropped_zero": dropped, "converted": stats(x),
              "reference": stats(ref_x) if ref_x is not None else None}
    matched = None
    if ref_x is not None:
        print("matching GM-LAMP's test channels with the converted users ...")
        idx, fid, rel = match_reference(ref_x, x)
        ok = rel < 1e-3      # matches ~1e-5 (DeepMIMO version round-off); neighbouring users differ by >= 2e-2
        report["match"] = dict(n_reference=int(ref_x.shape[1]), n_exact=int(ok.sum()), frac_exact=float(ok.mean()),
                               fidelity_median=float(np.median(fid)), fidelity_min=float(fid.min()),
                               rel_err_median=float(np.median(rel)), rel_err_max_exact=float(rel[ok].max()) if ok.any() else None,
                               n_distinct_users=int(len(np.unique(idx[ok]))),
                               peak_tv_to_reference=peak_tv(x, ref_x))
        m = report["match"]
        print(f"  {m['n_exact']} / {m['n_reference']} GM-LAMP test channels found (rel. error < 1e-3; "
              f"median {m['rel_err_median']:.1e}); fidelity median {m['fidelity_median']:.12f}")
        if m["frac_exact"] < 0.99:
            print("WARNING: the converted channels are not GM-LAMP's data set (ULA: generate it with "
                  "deepmimo_make_H(..., 'ULA'), i.e. 256 elements on the z axis; keep --lens gmlamp).")
        else:
            matched = np.unique(idx)     # best match of every test channel (exact or nearest) leaves the pool
    if a.test == "reference":
        if ref_x is None:
            raise SystemExit(f"GM-LAMP's test file not found:\n  {ref_file}\nExtract bsce_reference_mat_part*.zip into the "
                             "bsce folder (it must contain reference/gmlamp/CE_DeepMIMO/), or pass --reference <file>.")
        if matched is None:
            raise SystemExit("GM-LAMP's test channels were not found among the input users (see WARNING above): "
                             "the input is not GM-LAMP's data set. (--test random converts it anyway.)")
        pool = np.setdiff1d(np.arange(x.shape[1]), matched)
        perm = pool[np.random.default_rng(int(sc["seed"])).permutation(pool.size)]
        if perm.size < a.n_train + a.n_val:
            raise SystemExit(f"only {perm.size} channels outside GM-LAMP's test set, {a.n_train + a.n_val} requested")
        parts = {"training": x[:, perm[:a.n_train]], "validation": x[:, perm[a.n_train:a.n_train + a.n_val]],
                 "test": ref_x}
        report["split"] = dict(test="GM-LAMP test file (identical channels)", excluded_users=int(matched.size),
                               training_validation_pool=int(pool.size))
    else:
        need = a.n_train + a.n_val + a.n_test
        if x.shape[1] < need:
            raise SystemExit(f"only {x.shape[1]} usable channels, {need} requested (--n-train/--n-val/--n-test)")
        perm = np.random.default_rng(int(sc["seed"])).permutation(x.shape[1])
        parts = {"training": x[:, perm[:a.n_train]], "validation": x[:, perm[a.n_train:a.n_train + a.n_val]],
                 "test": x[:, perm[a.n_train + a.n_val:need]]}
        report["split"] = dict(test="random")
    out = resolve(sc["data_dir"])
    out.mkdir(parents=True, exist_ok=True)
    pre = f"{sc['dataset']}_{sc['array']}"
    for part, xs in parts.items():
        f = out / f"{pre}_{part}_channel.mat"
        sio.savemat(f, {"x": xs})
        print(f"saved {f}  {xs.shape}")
    (out / "deepmimo_stats.json").write_text(json.dumps(report, indent=1), encoding="utf-8")
    print(json.dumps({k: report[k] for k in ("reference", "converted") if report[k] is not None}, indent=1))
    plot(parts["training"], ref_file, out / "deepmimo_stats.png", arr)
    print("next: python scripts/generate_data.py --scenario", a.scenario, "--measurements-only")


def plot(x, ref_file, path, arr):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(1, 2, figsize=(11, 4.2))
    series = [("converted DeepMIMO (training set)", x, "#2a78d6")]
    rx = sio.loadmat(ref_file)["x"] if ref_file.exists() else None
    if rx is not None:
        series.append(("GM-LAMP CE_DeepMIMO", rx, "#eb6834"))
    for lab, xx, c in series:
        b = beams95(xx)
        v, cnt = np.unique(b, return_counts=True)
        ax[0].plot(v, np.cumsum(cnt) / cnt.sum(), color=c, lw=2, label=f"{lab}: median {np.median(b):.0f}")
        peak = np.argmax(np.abs(xx), 0)
        h, _ = np.histogram(peak, bins=np.arange(xx.shape[0] + 1))
        tv = f", TV to GM-LAMP {peak_tv(xx, rx):.3f}" if rx is not None and xx is x else ""
        ax[1].plot(h / h.sum(), color=c, lw=1.2, label=f"{lab}: median {np.median(peak):.0f}{tv}")
    ax[0].set_xlabel("beams holding 95 % of the energy"); ax[0].set_ylabel("CDF over channels")
    ax[1].set_xlabel("index of the strongest beam"); ax[1].set_ylabel("fraction of channels")
    for a_ in ax:
        a_.grid(alpha=0.3); a_.legend(frameon=False, fontsize=8)
        for s in ("top", "right"):
            a_.spines[s].set_visible(False)
    fig.suptitle(f"DeepMIMO {arr}: converted channels vs GM-LAMP's data set", fontsize=10)
    fig.tight_layout()
    fig.savefig(path, dpi=160)


if __name__ == "__main__":
    main()

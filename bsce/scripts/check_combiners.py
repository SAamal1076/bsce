#!/usr/bin/env python3
"""Transfer check (v30): every network of a runs folder on a set of standard test combiners it never saw in training.

  %run scripts/check_combiners.py --scenario configs/scenarios/sv_ula.yaml --runs outputs/SV_ULA
  %run scripts/check_combiners.py --scenario configs/scenarios/sv_ula.yaml --runs outputs/SV_ULA --tags "G-LADMM*" GM-LAMP-authors
  %run scripts/check_combiners.py --scenario configs/scenarios/sv_ula.yaml --runs outputs/SV_ULA --combiners own pm1 dct M64
  %run scripts/check_combiners.py ... --combiner-files my_A1.mat my_A2.mat        (your own combiners, 'A' = M x N)

The test combiners (real M x N, N and M = M0 from the scenario; deterministic, written to <out>/combiners/):
  own        the scenario's combiner (the one the networks were trained with)
  pm1        random +-1/sqrt(M0), a new draw (seed 11)
  gauss      i.i.d. Gaussian / sqrt(M0)
  corr       Gaussian rows with correlation 0.9 between neighbouring RF chains (badly conditioned A A^T)
  sparse     partially connected: every RF chain sees 16 random beams (+-1/4), the rest 0
  dct        M0 random rows of the orthonormal DCT-II (a structured, deterministic transform subset)
  M<k>       random +-1/sqrt(k) with another number of RF chains: k = 1.25 M0, 0.75 M0, 0.5 M0 (160, 96, 64 for M0 = 128);
             in --combiners also by their aliases M+ M- M-- (whatever M0 is)
For every combiner, evaluate.py tests the networks on the scenario's test channels with the same noise:
  - networks that can follow a new combiner use it, any M: Graph-LADMM analytic, LADMM operator=analytic, and from v31
    NF-LADMM, LVAMP and ALISTA (nothing they learn depends on A);
  - the others keep the combiner they were learned for (that is the transfer test); with another M they cannot run.
  v31: a combiner on which evaluate.py fails, or where no method can run, is noted and the check goes on.
Writes <out>/ (default results/combiner_check_<SCENARIO>_<runs folder>):
  combiner_check.md / .csv   mean NMSE over the test SNRs, network x combiner, and the change against 'own'
  combiner_check.png         the same as a table, every value printed (red: worse than estimating zero)
  combiners.csv              M, cond(A A^T) and the mutual coherence of every test combiner
  <combiner>/                evaluate.py output (nmse_vs_snr.csv at every SNR, eval_log.json: who followed)
Options: --classical amp (default; '' = none), --graph-phase own|target (another data set's edge phases),
--n-test N (fewer test channels, quick look), --device.
"""
import argparse
import csv
import fnmatch
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import scipy.io as sio

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from bsce.config import load_yaml, resolve, scenario_cs_matrix, scenario_run_name  # noqa: E402
from bsce.classical import METHODS as CLASSICAL  # noqa: E402  (v31: one list)

LABEL = {"own": "the scenario's combiner (training)", "pm1": "random ±1, new draw", "gauss": "Gaussian",
         "corr": "correlated rows (0.9)", "sparse": "partially connected (16 beams per RF chain)",
         "dct": "DCT-II row subset (structured)"}

def make_combiners(A0: np.ndarray):
    """The standard test combiners for a scenario combiner A0 (M0 x N), always the same ones."""
    from scipy.fft import dct
    M0, N = A0.shape
    g = np.random.default_rng(2026)                      # draws in a fixed order: gauss, sparse, dct, corr
    pm = lambda M, s: ((np.random.default_rng(s).random((M, N)) > 0.5) * 2 - 1) / np.sqrt(M)  # noqa: E731
    C = {"own": A0, "pm1": pm(M0, 11), "gauss": g.standard_normal((M0, N)) / np.sqrt(M0)}
    S = np.zeros((M0, N))
    k = min(16, N)
    for m in range(M0):
        idx = g.choice(N, k, replace=False)
        S[m, idx] = (g.random(k) > 0.5) * 2 - 1
    C["sparse"] = S / np.sqrt(k)
    D = dct(np.eye(N), norm="ortho", axis=0)
    C["dct"] = D[np.sort(g.choice(N, M0, replace=False))]
    R = np.zeros((M0, N)); R[0] = g.standard_normal(N)
    for m in range(1, M0):
        R[m] = 0.9 * R[m - 1] + np.sqrt(1 - 0.81) * g.standard_normal(N)
    C["corr"] = R / np.sqrt(M0)
    for key, f, s in (("M+", 1.25, 14), ("M-", 0.75, 12), ("M--", 0.5, 13)):
        M = int(round(f * M0))
        C[f"M{M}"] = pm(M, s)
        LABEL[f"M{M}"] = f"random ±1, M = {M}"
    return C


def properties(A):
    An = A / np.linalg.norm(A, axis=0, keepdims=True)
    G = np.abs(An.T @ An)
    np.fill_diagonal(G, 0)
    return dict(M=A.shape[0], N=A.shape[1], cond_AAt=float(np.linalg.cond(A @ A.T)), coherence=float(G.max()))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--scenario", required=True)
    ap.add_argument("--runs", required=True, help="runs folder, e.g. outputs/SV_ULA")
    ap.add_argument("--tags", nargs="*", default=None, help="run folders (names or patterns); default: all")
    ap.add_argument("--exclude", nargs="*", default=None)
    ap.add_argument("--combiners", nargs="*", default=None,
                    help="subset of: own pm1 gauss corr sparse dct M+ M- M-- (or the M<k> names; default: all)")
    ap.add_argument("--combiner-files", nargs="*", default=[], help="your own combiners (.mat with A, M x N)")
    ap.add_argument("--classical", default="amp")
    ap.add_argument("--graph-phase", choices=["own", "target"], default="own")
    ap.add_argument("--n-test", type=int, default=None)
    ap.add_argument("--device", default="auto")
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    sc = load_yaml(a.scenario)
    runs = resolve(a.runs)
    if not runs.is_dir():
        sys.exit(f"{runs} is not a folder")
    smoke = runs.name.endswith("_smoke")                # as evaluate.py: the smoke data set goes with *_smoke runs
    A0 = sio.loadmat(scenario_cs_matrix(sc, smoke))["A"].astype(np.float64)
    out = resolve(a.out) if a.out else resolve("results") / f"combiner_check_{scenario_run_name(sc, smoke)}_{runs.name}"
    (out / "combiners").mkdir(parents=True, exist_ok=True)
    C = make_combiners(A0)
    alias = {al: f"M{int(round(f * A0.shape[0]))}" for al, f in (("M+", 1.25), ("M-", 0.75), ("M--", 0.5))}
    if a.combiners:
        a.combiners = list(dict.fromkeys(alias.get(c, c) for c in a.combiners))
        unknown = [c for c in a.combiners if c not in C]
        if unknown:
            sys.exit(f"unknown combiners {unknown}; known: {', '.join(C)}")
        C = {k: C[k] for k in a.combiners}
    for f in a.combiner_files:
        p = resolve(f)
        A = sio.loadmat(p)["A"].astype(np.float64)
        if A.ndim != 2 or A.shape[1] != A0.shape[1]:
            sys.exit(f"{p}: A must be M x {A0.shape[1]}, it is {A.shape}")
        name = f"file_{p.stem}"
        if name in C:
            sys.exit(f"two combiner files are named {p.stem}")
        C[name], LABEL[name] = A, f"{p.name}, M = {A.shape[0]}"
    props = []
    for k, A in C.items():
        sio.savemat(out / "combiners" / f"{k}.mat", {"A": A})
        props.append(dict(combiner=k, label=LABEL.get(k, k), **properties(A)))
    with open(out / "combiners.csv", "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(props[0])); w.writeheader(); w.writerows(props)
    for p in props:
        print(f"[check] {p['combiner']:8s} {p['M']:4d} x {p['N']}  cond(AA^T) = {p['cond_AAt']:9.1f}  "
              f"coherence = {p['coherence']:.3f}  ({p['label']})")

    res = {}
    for k in C:
        d = out / k
        cmd = [sys.executable, "scripts/evaluate.py", "--scenario", a.scenario, "--runs", str(runs), "--out", str(d),
               "--classical", a.classical, "--device", a.device, "--graph-phase", a.graph_phase]
        if k != "own":
            cmd += ["--combiner", str(out / "combiners" / f"{k}.mat")]
        if a.tags:
            cmd += ["--tags", *a.tags]
        if a.exclude:
            cmd += ["--exclude", *a.exclude]
        if a.n_test:
            cmd += ["--n-test", str(a.n_test)]
        print(f"\n>>> combiner {k} ({LABEL.get(k, k)})", flush=True)
        r = subprocess.run(cmd, cwd=ROOT)
        if r.returncode or not (d / "nmse_vs_snr.csv").exists():   # v31: note it and go on with the next combiner
            print(f"[check] evaluate.py failed on combiner {k} (exit {r.returncode}); its cells are '–'")
            res[k] = ({}, {"_failed": f"evaluate.py failed (exit {r.returncode})"})
            continue
        rows = list(csv.DictReader(open(d / "nmse_vs_snr.csv", encoding="utf-8")))
        notes = json.loads((d / "eval_log.json").read_text(encoding="utf-8")).get("network_combiner", {})
        val = {m: float(np.mean([float(x[m]) for x in rows])) for m in (rows[0] if rows else {}) if m != "SNR_dB"}
        if not val:
            print(f"[check] combiner {k}: no method could run (see the notes)")
        res[k] = (val, notes)

    nets = list(dict.fromkeys(m for v, n in res.values() for m in list(v) + list(n) if m != "_failed"))
    classical = [m for m in nets if m in CLASSICAL]
    nets = [m for m in nets if m not in classical] + classical
    if a.tags:                                           # keep the order of --tags where it applies
        order = [m for t in a.tags for m in nets if fnmatch.fnmatchcase(m, t)]
        nets = list(dict.fromkeys(order + nets))
    own = res.get("own", (None, {}))[0]
    lines_csv = []
    head = "| network | " + " | ".join(LABEL.get(k, k) for k in C) + " |"     # the M<k> labels name their M
    md = ["# Mean NMSE (dB) over the test SNRs on every test combiner, without retraining", "",
          f"Scenario {sc['name']}, runs {runs}. 'cannot run': learned for another number of RF chains. "
          "In brackets: the change against the scenario's own combiner (negative = worse).", "",
          head, "|---" * (len(C) + 1) + "|"]
    for m in nets:
        cells = []
        for k in C:
            val, notes = res[k]
            if m in val:
                v = val[m]
                d = "" if own is None or m not in own or k == "own" else f" ({own[m] - v:+.2f})"
                cells.append(f"{v:.2f}{d}")
                lines_csv.append(dict(network=m, combiner=k, mean_nmse_db=round(v, 4),
                                      change_vs_own_db=None if own is None or m not in own else round(own[m] - v, 4),
                                      note=notes.get(m, "classical" if m in classical else "own")))
            else:
                cells.append("cannot run" if "cannot run" in str(notes.get(m, "")) else "–")
                lines_csv.append(dict(network=m, combiner=k, mean_nmse_db=None, change_vs_own_db=None,
                                      note=notes.get(m, "not evaluated")))
        md.append(f"| {m} | " + " | ".join(cells) + " |")
    (out / "combiner_check.md").write_text("\n".join(md) + "\n", encoding="utf-8")
    if not lines_csv:
        sys.exit("[check] nothing could be evaluated on any combiner (see the messages above)")
    with open(out / "combiner_check.csv", "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(lines_csv[0])); w.writeheader(); w.writerows(lines_csv)
    print("\n".join(md))

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    M = np.array([[res[k][0].get(m, np.nan) for k in C] for m in nets], float)
    fig, ax = plt.subplots(figsize=(2.0 + 1.25 * len(C), 1.6 + 0.42 * len(nets)))
    cm = plt.get_cmap("viridis_r")
    fin = M[np.isfinite(M) & (M < 0)]
    lo, hi = (fin.min(), fin.max()) if fin.size else (-1.0, 0.0)
    for i, m in enumerate(nets):
        for j, k in enumerate(C):
            v = M[i, j]
            if not np.isfinite(v):
                t = "cannot run" if "cannot run" in str(res[k][1].get(m, "")) else "–"
                ax.add_patch(plt.Rectangle((j, i), 0.96, 0.92, facecolor="#f1f0ec", edgecolor="none"))
                ax.text(j + 0.48, i + 0.46, t, ha="center", va="center", fontsize=6.6, color="#777777")
                continue
            f = (v - lo) / max(hi - lo, 1e-9)
            ax.add_patch(plt.Rectangle((j, i), 0.96, 0.92, edgecolor="none",
                                       facecolor="#c23b3b" if v >= 0 else cm(0.05 + 0.85 * f)))
            ax.text(j + 0.48, i + 0.46, f"{v:.2f}" if abs(v) < 1000 else f"{v:.0f}", ha="center", va="center",
                    fontsize=7.4, fontweight="bold", color="white" if (v >= 0 or f > 0.55) else "#111111")
    ax.set_xlim(0, len(C)); ax.set_ylim(len(nets), 0)
    ax.set_xticks(np.arange(len(C)) + 0.48)
    ax.set_xticklabels([LABEL.get(k, k) for k in C], fontsize=7.2, rotation=25, ha="left"); ax.xaxis.tick_top()
    ax.set_yticks(np.arange(len(nets)) + 0.46); ax.set_yticklabels(nets, fontsize=7.4)
    for sp in ax.spines.values():
        sp.set_visible(False)
    ax.tick_params(length=0)
    ax.set_title(f"{sc['name']}: mean NMSE (dB) on combiners never seen in training "
                 "(red = worse than estimating zero)", fontsize=9.5, pad=70)
    fig.savefig(out / "combiner_check.png", dpi=150, bbox_inches="tight")
    print(f"\nwritten {out / 'combiner_check.md'}, .csv, .png, combiners.csv")


if __name__ == "__main__":
    main()

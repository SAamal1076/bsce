#!/usr/bin/env python3
"""Tune the classical l1 ADMM and graph-reweighted l1 (v31) per test SNR on validation channels, for evaluate.py.

  %run scripts/tune_classical.py --scenario configs/scenarios/sv_upa.yaml
  %run scripts/evaluate.py --scenario configs/scenarios/sv_upa.yaml --runs outputs/SV_UPA --classical amp,l1,grwl1 \
       --classical-params results/classical_params_SV_UPA.json

Per test SNR of the scenario (as the uploaded graph_reweighted_l1 package tunes, with the true SNR):
  l1     lam on 13 log points 10^-2.5 .. 10^0.5, then 0.65x and 1.5x the best;
  grwl1  lam_rw in {0.5, 1, 2, 3} x lam_l1 and eps in {0.05, 0.2, 0.6}; round 0 + 2 reweighted rounds.
Every solve is 300 ADMM iterations with rho = 1 (bsce.classical.l1_admm / grwl1). Validation: the first --n-val
channels of <data>/<scenario>_validation_channel.mat, noise rng NumpyMatlabRNG(seed + 9000 + i) (i = SNR index; the test
noise of evaluate.py uses seed + 1000 + i). Writes a JSON {snr: {lam_l1, val_l1_db, lam_rw, eps, val_rw_db}}.
About 2-3 min per SNR on one CPU core with 300 channels.
"""
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import scipy.io as sio

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from bsce import classical  # noqa: E402
from bsce.config import load_yaml, resolve, scenario_cs_matrix, scenario_data_dir  # noqa: E402
from bsce.data import measurements  # noqa: E402
from bsce.data import noise as noise_models  # noqa: E402
from bsce.rng import NumpyMatlabRNG  # noqa: E402

LAM = list(np.logspace(-2.5, 0.5, 13))


def nm(Z, X):
    return float(np.sum(np.abs(Z - X) ** 2) / np.sum(np.abs(X) ** 2))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--scenario", required=True)
    ap.add_argument("--snr", nargs="*", type=float, default=None, help="test SNRs to tune (default: the scenario's)")
    ap.add_argument("--n-val", type=int, default=300)
    ap.add_argument("--methods", default="l1,grwl1")
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--out", default=None, help="default results/classical_params_<SCENARIO>.json (_white.json for white)")
    ap.add_argument("--noise-model", default="colored", choices=["colored", "coloured", "folded", "white"],
                    help="validation noise (v31.3): tune for the noise model the comparison uses")
    a = ap.parse_args()
    sc = load_yaml(a.scenario)
    A = sio.loadmat(scenario_cs_matrix(sc, a.smoke))["A"].astype(np.float64)
    pre = f"{sc['dataset']}_{sc['array']}"                      # as evaluate.py
    xv = sio.loadmat(scenario_data_dir(sc, a.smoke) / f"{pre}_validation_channel.mat")["x"][:, :a.n_val]
    array, N1, N2 = sc.get("array", "ULA"), int(sc.get("N1", 16)), int(sc.get("N2", 16))
    snrs = sc["test_snr_db"] if a.snr is None else a.snr
    methods = [m for m in a.methods.split(",") if m]
    noise = noise_models.canonical(a.noise_model)
    out = resolve(a.out) if a.out else resolve("results") / (f"classical_params_{sc['name']}"
                                                            f"{'_white' if noise == 'white' else ''}{'_smoke' if a.smoke else ''}.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    res = json.loads(out.read_text()) if out.exists() else {}
    P = np.linalg.inv(A.T @ A + np.eye(A.shape[1]))
    for snr in snrs:
        t0 = time.time()
        i = list(sc["test_snr_db"]).index(snr) if snr in sc["test_snr_db"] else int(round(snr))
        yv = measurements.gen_measurements_matlab(xv, A, snr, NumpyMatlabRNG(int(sc["seed"]) + 9000 + i), noise_model=noise)
        sco = sorted((nm(classical.l1_admm(yv, A, l_, P=P), xv), l_) for l_ in LAM)
        l0 = sco[0][1]
        sco += [(nm(classical.l1_admm(yv, A, l0 * f, P=P), xv), l0 * f) for f in (0.65, 1.5)]
        v1, lam1 = min(sco)
        r = dict(lam_l1=lam1, val_l1_db=10 * np.log10(v1))
        if "grwl1" in methods:
            best = (9e9, None, None)
            for f in (0.5, 1.0, 2.0, 3.0):
                for eps in (0.05, 0.2, 0.6):
                    v = nm(classical.grwl1(yv, A, lam1 * f, eps, array=array, N1=N1, N2=N2), xv)
                    if v < best[0]:
                        best = (v, lam1 * f, eps)
            r.update(lam_rw=best[1], eps=best[2], val_rw_db=10 * np.log10(best[0]))
        res[f"{snr:g}"] = r
        out.write_text(json.dumps(res, indent=1), encoding="utf-8")
        print(f"SNR {snr:5.1f} dB: " + ", ".join(f"{k} {v:.4g}" for k, v in r.items()) + f"  [{time.time() - t0:.0f} s]",
              flush=True)
    print("written", out)


if __name__ == "__main__":
    main()

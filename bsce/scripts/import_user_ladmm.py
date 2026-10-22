#!/usr/bin/env python3
"""Put a LADMM checkpoint of your own scripts (untitled2.py .pt) into a runs folder as a bsce model, so evaluate.py
tests it on the scenario's test set with the same noise as every other method.

  python scripts/import_user_ladmm.py --scenario configs/scenarios/sv_upa_user.yaml --runs outputs/SV_UPA_user \
         --pt Trained_LADMM_for_SV_UPA_2561280to20dB_1_11dB_better.pt --tag LADMM-untitled2-better [--band 0to20dB]
-> <runs>/<tag>/<tag>_<band>.pt/.mat
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import scipy.io as sio

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from bsce.config import load_yaml, resolve, scenario_cs_matrix  # noqa: E402
from bsce.export import save_model  # noqa: E402
from bsce.models.user_import import ladmm_from_untitled2  # noqa: E402


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--scenario", required=True)
    ap.add_argument("--runs", required=True)
    ap.add_argument("--pt", required=True)
    ap.add_argument("--tag", required=True)
    ap.add_argument("--band", default="0to20dB")
    a = ap.parse_args()
    sc = load_yaml(a.scenario)
    A = sio.loadmat(scenario_cs_matrix(sc))["A"].astype(np.float64)
    m, ck = ladmm_from_untitled2(a.pt, A)
    log = str(ck.get("log", ""))
    p, _ = save_model(m, resolve(a.runs) / a.tag / f"{a.tag}_{a.band}",
                      train_state={"done": list(ck.get("done", [])), "log": log.splitlines(),
                                   "layer_nmse_db": [float(v) for v in np.ravel(ck.get("layernmse", []))]},
                      data_info={"source": str(a.pt), "note": "imported from the user's untitled2.py checkpoint"})
    print("written", p)


if __name__ == "__main__":
    main()

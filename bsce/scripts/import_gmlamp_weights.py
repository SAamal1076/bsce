#!/usr/bin/env python3
"""Put GM-LAMP's published trained networks (LAMP, GM-LAMP; T = 8) into a runs folder as bsce models,
so evaluate.py tests them with the same test channels and noise as the other models.

  python scripts/import_gmlamp_weights.py --scenario configs/scenarios/sv_upa.yaml --runs outputs/SV_UPA

Creates <runs>/LAMP-authors/LAMP-authors_<band>.pt and <runs>/GM-LAMP-authors/... for the bands
0to10dB and 10to20dB from reference/gmlamp/CE_SV or CE_DeepMIMO/Trained_*_for_<D>_<ARR>_256128<band>.mat.
The scenario's CS matrix must be GM-LAMP's (CSmatrix256128 for SV, CSmatrix_128256 for DeepMIMO).
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import scipy.io as sio

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from bsce.config import load_yaml, resolve, scenario_cs_matrix  # noqa: E402
from bsce.models.lamp import LAMP  # noqa: E402
from bsce.export import save_model  # noqa: E402


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--scenario", required=True)
    ap.add_argument("--runs", required=True)
    a = ap.parse_args()
    sc = load_yaml(a.scenario)
    ds, arr = sc["dataset"], sc["array"].upper()
    folder = ROOT / "reference/gmlamp" / ("CE_SV" if ds == "SV" else "CE_DeepMIMO")
    A = sio.loadmat(scenario_cs_matrix(sc))["A"].astype(np.float64)
    Aref = sio.loadmat(folder / ("CSmatrix256128.mat" if ds == "SV" else "CSmatrix_128256.mat"))["A"].astype(np.float64)
    if np.abs(A - Aref).max() > 1e-6:
        sys.exit("the scenario's CS matrix is not GM-LAMP's: the published weights do not apply")
    for name, pre, shr in (("LAMP-authors", "Trained_LAMP", "soft"), ("GM-LAMP-authors", "Trained_GM_LAMP", "gm")):
        for band in ({"name": "0to10dB"}, {"name": "10to20dB"}):   # the authors published these two bands only
            f = folder / f"{pre}_for_{ds}_{arr}_256128{band['name']}.mat"
            if not f.exists():
                print(f"missing {f}"); continue
            m = LAMP(A, T=8, shrink=shr, untied=True, K=4, precision="float64")
            m.load_mat_dict(sio.loadmat(f))
            p, _ = save_model(m, resolve(a.runs) / name / f"{name}_{band['name']}",
                              data_info={"source": str(f.relative_to(ROOT))})
            print("written", p)


if __name__ == "__main__":
    main()

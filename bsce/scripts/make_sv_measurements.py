#!/usr/bin/env python3
"""Make the measurement files of a scenario from its channel files, like generate_SV_data.py / generate_measurement_data.py:
y = A (x + n), the K columns split into n_snr_chunks equal chunks, chunk k at SNR_k ~ U(lo, hi) dB, sigma^2 = 10^(-SNR/10).

  python scripts/make_sv_measurements.py --scenario configs/scenarios/sv_ula_user.yaml
  python scripts/make_sv_measurements.py --scenario configs/scenarios/sv_upa_user_2band.yaml
  python scripts/make_sv_measurements.py --scenario configs/scenarios/sv_upa.yaml --snr 0 5 10 15 20
  options: --snr <edges> (any SNR bands)  --force (overwrite existing files)   --seed 0   --only train|val
           --noise-model white   white files y = A x + w (<name>_white.mat): the colored file of the same draw with its
                                 noise whitened (made from the colored file when it exists; bsce/data/noise.py)
           --snr-dist fair [--snr-step 0.2]   (v26) every training column its own SNR, spread evenly over the band
                                 (<name>_fair.mat / _fair0p2.mat); validation over a range at equally spaced SNRs
                                 (<name>_fair.mat): the files train.py --snr-dist fair uses

For every band of the scenario:
  training   <data_dir>/<D>_<ARR>_training_<M>_measurements_<train_name or name>.mat   SNR ~ U(band.train)
  validation <data_dir>/<D>_<ARR>_validation_<M>_measurements_<val_name or name>.mat SNR ~ U(band.val_range)
             (band.val_range missing -> the band's val_snr points, one chunk each)
Existing files are kept (your SV_UPA_*_0to20dB / _1_11dB files are not touched). Each file also stores the SNR of
every column ('snr_db', needed by NF-LADMM in file mode); the chunk SNRs are in <file>.snr.json.
train.py makes missing files itself; this script is for making them ahead of time.
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import scipy.io as sio

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from bsce.config import load_yaml, scenario_cs_matrix, scenario_data_dir, band_file_names  # noqa: E402
from bsce.data.files import apply_band_options, bands_from_edges, ensure_band_files, snr_dist_suffix  # noqa: E402


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--scenario", required=True)
    ap.add_argument("--snr", nargs="+", type=float, default=None,
                    help="SNR band edges instead of the scenario's bands: 0 20 -> 0to20dB; 0 10 20 -> 0to10dB, 10to20dB")
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--only", choices=["train", "val"], default=None)
    ap.add_argument("--noise-model", default=None, choices=["colored", "coloured", "folded", "white"],
                    help="colored y = A (x + n) (default: the scenario's) or white y = A x + w")
    ap.add_argument("--snr-dist", choices=["chunks", "fair"], default="chunks",
                    help="chunks (default, as before) or fair: one SNR per training column, spread evenly over the band")
    ap.add_argument("--snr-step", type=float, default=None, help="with --snr-dist fair: the SNR grid step in dB")
    a = ap.parse_args()
    try:
        fair = bool(snr_dist_suffix(a.snr_dist, a.snr_step))
    except ValueError as e:
        sys.exit(str(e))
    sc = load_yaml(a.scenario)
    if a.noise_model:
        sc["noise_model"] = a.noise_model
    if a.snr:
        sc["bands"] = bands_from_edges(a.snr)
    if fair:
        sc["bands"] = apply_band_options(sc["bands"], default_chunks=int(sc.get("n_snr_chunks", 10)), snr_dist="fair",
                                         snr_step=a.snr_step)
    data = scenario_data_dir(sc)
    A = sio.loadmat(scenario_cs_matrix(sc))["A"].astype(np.float64)
    for bi, band in enumerate(sc["bands"]):
        ensure_band_files(sc, band, bi, data, A, band_file_names(sc, band), seed=a.seed, force=a.force, only=a.only)


if __name__ == "__main__":
    main()

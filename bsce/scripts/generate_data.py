#!/usr/bin/env python3
"""Generate beamspace channels + measurements for a scenario (same files as the originals).

  python scripts/generate_data.py --scenario configs/scenarios/sv_ula.yaml
  python scripts/generate_data.py --scenario ... --smoke            (tiny sizes, quick check)
  python scripts/generate_data.py --scenario ... --measurements-only (channels already there,
                                                                       e.g. DeepMIMO)
Files (data_dir):
  <D>_<ARR>_{training,validation,test}_channel.mat               x  (N x K)
  <D>_<ARR>_training_<M>_measurements_<band>.mat                 y  (M x K)
  <D>_<ARR>_validation_<M>_measurements_<band>.mat               y
  CSmatrix.mat                                                   A  (M x N)
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import scipy.io as sio

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from bsce.config import load_yaml, resolve, apply_overrides  # noqa: E402
from bsce.data import sv, sensing, measurements  # noqa: E402
from bsce.rng import NumpyMatlabRNG  # noqa: E402


def prefix(sc):
    return f"{sc['dataset']}_{sc['array']}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scenario", required=True)
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--measurements-only", action="store_true")
    ap.add_argument("--set", nargs="*", default=[], help="overrides, e.g. n_train=1000")
    a = ap.parse_args()
    sc = apply_overrides(load_yaml(a.scenario), a.set)
    if a.smoke:
        sc.update(n_train=1000, n_val=200, n_test=200, data_dir=sc["data_dir"] + "_smoke")
    out = resolve(sc["data_dir"])
    out.mkdir(parents=True, exist_ok=True)
    pre = prefix(sc)
    seed = int(sc["seed"])
    # sensing matrix
    if sc.get("cs_matrix"):
        A = sio.loadmat(resolve(sc["cs_matrix"]))["A"].astype(np.float64)
    else:
        A = sensing.cs_matrix(NumpyMatlabRNG(seed + 99), sc["M"], sc["N1"] * sc["N2"])
    sio.savemat(out / "CSmatrix.mat", {"A": A})
    # channels
    xs = {}
    for i, part in enumerate(("training", "validation", "test")):
        f = out / f"{pre}_{part}_channel.mat"
        if a.measurements_only or (sc["dataset"] != "SV"):
            xs[part] = sio.loadmat(f)["x"]
            continue
        K = int(sc[{"training": "n_train", "validation": "n_val", "test": "n_test"}[part]])
        xs[part] = sv.beamspace_channels(NumpyMatlabRNG(seed + i), K, sc["array"], sc["L"], sc["N1"], sc["N2"])
        sio.savemat(f, {"x": xs[part]})
        print(f"saved {f.name}  {xs[part].shape}")
    # measurements per band (genMeasurements semantics)
    info = {"scenario": sc, "files": []}
    for j, band in enumerate(sc["bands"]):
        rs = np.random.RandomState(seed + 100 + j)
        snrs = measurements.draw_snr_list(rs, band["train"][0], band["train"][1], int(sc["n_snr_chunks"]))
        y = measurements.gen_measurements_py(xs["training"], A, snrs, rs, strict=True)
        f = out / f"{pre}_training_{sc['M']}_measurements_{band['name']}.mat"
        sio.savemat(f, {"y": y, "snr_db": snrs})
        if "val_snr" in band:
            vs = np.asarray(band["val_snr"], float)
        else:                           # val_range: one SNR per chunk, drawn like the training file
            vs = measurements.draw_snr_list(rs, band["val_range"][0], band["val_range"][1],
                                            int(band.get("val_n_snr_chunks", sc["n_snr_chunks"])))
        yv = measurements.gen_measurements_py(xs["validation"], A, vs, rs, strict=True)
        fv = out / f"{pre}_validation_{sc['M']}_measurements_{band['name']}.mat"
        sio.savemat(fv, {"y": yv, "snr_db": vs})
        info["files"] += [f.name, fv.name]
        print(f"saved {f.name}, {fv.name}")
    (out / "dataset_info.json").write_text(json.dumps(info, indent=1, default=str), encoding="utf-8")


if __name__ == "__main__":
    main()

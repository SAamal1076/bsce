#!/usr/bin/env python3
"""Check every trained model of a run folder: .pt, .mat (bsce_infer.m in Octave/MATLAB) and,
where it exists, _orig.mat must give the same estimate.

  python scripts/check_exports.py --runs outputs/SV_ULA_smoke --scenario configs/scenarios/sv_ula.yaml
  (uses Octave from BSCE_OCTAVE / PATH; --matlab "C:/Program Files/MATLAB/R2024b/bin/matlab.exe" uses MATLAB)
Writes <runs>/export_check.json and prints one line per model file.
"""
import argparse
import json
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np
import scipy.io as sio
import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from bsce.config import load_yaml, resolve, scenario_cs_matrix, scenario_test_dir  # noqa: E402
from bsce.export import load_model  # noqa: E402
from bsce.data import measurements  # noqa: E402
from bsce.rng import NumpyMatlabRNG  # noqa: E402


def rel(a, b):
    return float(np.linalg.norm(a - b) / max(np.linalg.norm(b), 1e-300))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", required=True)
    ap.add_argument("--scenario", required=True)
    ap.add_argument("--n", type=int, default=50)
    ap.add_argument("--octave", default=os.environ.get("BSCE_OCTAVE") or shutil.which("octave-cli") or shutil.which("octave"),
                    help="octave-cli executable (default: BSCE_OCTAVE or PATH)")
    ap.add_argument("--matlab", default=None, help="use MATLAB instead: path to matlab(.exe), run with -batch")
    a = ap.parse_args()
    sc = load_yaml(a.scenario)
    runs = resolve(a.runs)
    smoke = runs.name.endswith("_smoke")
    A = sio.loadmat(scenario_cs_matrix(sc, smoke))["A"].astype(np.float64)
    x = sio.loadmat(scenario_test_dir(sc, smoke) / f"{sc['dataset']}_{sc['array']}_test_channel.mat")["x"][:, :a.n]
    y = measurements.gen_measurements_matlab(x, A, 10.0, NumpyMatlabRNG(1))
    results = {}
    for pt in sorted(runs.glob("*/*.pt")):
        net = load_model(pt)
        if getattr(net, "family", "") == "nfladmm":
            print(f"{pt.relative_to(runs)}: NF-LADMM has no MATLAB inference (Python only) -> skipped")
            continue
        with torch.no_grad():
            ref = net(torch.as_tensor(y)).cpu().numpy()
            # the same trained weights evaluated in float64 (MATLAB/Octave compute in double)
            ref64 = load_model(pt, precision="float64")(torch.as_tensor(y)).cpu().numpy()
            r = {"precision": net.cfg.get("precision"),
                 "mat_python_rel": rel(load_model(pt.with_suffix(".mat"))(torch.as_tensor(y)).cpu().numpy(), ref)}
        if a.octave or a.matlab:
            with tempfile.TemporaryDirectory() as td:
                tdp = Path(td).as_posix()            # forward slashes work in Octave and MATLAB on Windows
                sio.savemat(Path(td) / "in.mat", {"y": y})
                code = (f"addpath('{(ROOT / 'matlab').as_posix()}'); load('{tdp}/in.mat'); "
                        f"net = bsce_load('{pt.with_suffix('.mat').as_posix()}'); xh = bsce_infer(net, y); ")
                if a.matlab:
                    cmd = [a.matlab, "-batch", code + f"save('{tdp}/out.mat', 'xh');"]
                else:
                    cmd = [a.octave, "--no-gui", "--quiet", "--eval", code + f"save('-v7', '{tdp}/out.mat', 'xh');"]
                subprocess.run(cmd, check=True, capture_output=True)
                xo = sio.loadmat(Path(td) / "out.mat")["xh"]
                r["octave_vs_torch_native"] = rel(xo, ref)
                r["octave_vs_torch_f64"] = rel(xo, ref64)
        orig = pt.parent / (pt.stem + "_orig.mat")
        r["has_orig_mat"] = orig.exists()
        results[str(pt.relative_to(runs))] = r
        print(f"{str(pt.relative_to(runs)):48s} " + "  ".join((f"{k}={v:.1e}" if isinstance(v, float) else f"{k}={v}") for k, v in r.items()))
    (runs / "export_check.json").write_text(json.dumps(results, indent=1), encoding="utf-8")


if __name__ == "__main__":
    main()

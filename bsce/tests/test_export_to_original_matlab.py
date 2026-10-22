"""Trained-model files written by bsce are drop-in replacements for the ORIGINAL weight files.

save_model() writes <prefix>_orig.mat containing only the variables the original MATLAB
code reads. Here a PyTorch model with generic (perturbed) weights is exported, the file is
copied under the name the unchanged original function load()s, and the original function
(LAMP.m, GM_LAMP.m of the GM-LAMP code; LADMM_test.m of the LADMM project) is run in Octave.
Its estimate must equal the PyTorch estimate to machine precision.
"""
import shutil

import numpy as np
import pytest
import scipy.io as sio
import torch

from harness.octave import REFERENCE, compare, octave_available, run_octave, tmpdir
from bsce.data import sv, measurements
from bsce.rng import NumpyMatlabRNG
from bsce.models import build_model
from bsce.export import save_model

CE_SV = REFERENCE / "gmlamp" / "CE_SV"
ULA_TEST = REFERENCE / "ladmm_user" / "ULA_test_code"
pytestmark = pytest.mark.skipif(not octave_available(), reason="octave not installed")

CASES = [  # cfg, original function call, expected weight-file name, .m files needed
    (dict(model="LAMP", T=8, precision="float64"), "xh = LAMP(y, A, 1, 1);",
     "Trained_LAMP_for_SV_ULA_2561280to10dB.mat", [CE_SV / "LAMP.m"]),
    (dict(model="GM-LAMP", T=8, K=4, precision="float64"), "xh = GM_LAMP(y, A, 1, 2);",
     "Trained_GM_LAMP_for_SV_ULA_25612810to20dB.mat", [CE_SV / "GM_LAMP.m", CE_SV / "eta_gm.m"]),
    (dict(model="LADMM", T=6, operator="full", precision="float64"), "[xh, nm] = LADMM_test(y, A, 1, 1, x);",
     "Trained_LADMM_for_SV_ULA_2561280to5dB_1dB_T=6_it=15k_12.mat", [ULA_TEST / "LADMM_test.m"]),
    (dict(model="LADMM", T=6, operator="kron", precision="float64"), "[xh, nm] = LADMM_test(y, A, 1, 1, x);",
     "Trained_LADMM_for_SV_ULA_2561280to5dB_1dB_T=6_it=15k_12.mat", [ULA_TEST / "LADMM_test.m"]),
    (dict(model="LADMM", T=6, operator="kronprior", precision="float64"), "[xh, nm] = LADMM_test(y, A, 1, 1, x);",
     "Trained_LADMM_for_SV_ULA_2561280to5dB_1dB_T=6_it=15k_12.mat", [ULA_TEST / "LADMM_test.m"]),
]


@pytest.mark.parametrize("cfg,call,fname,mfiles", CASES, ids=lambda v: v.get("operator", v["model"]) if isinstance(v, dict) else None)
@pytest.mark.parametrize("seed", [0, 1])
def test_export_runs_in_original_matlab(cfg, call, fname, mfiles, seed, record):
    missing = [str(m) for m in mfiles if not m.exists()]
    if missing:
        pytest.skip(f"needs the original MATLAB code {missing} (reference/ladmm_user, bsce_reference zips)")
    A = sio.loadmat(CE_SV / "CSmatrix256128.mat")["A"].astype(np.float64)
    x = sv.beamspace_channels(NumpyMatlabRNG(seed), 64, "ULA")
    y = measurements.gen_measurements_matlab(x, A, 5.0 + 5 * seed, NumpyMatlabRNG(seed + 3))
    torch.manual_seed(seed)
    model = build_model(cfg, A)
    g = np.random.default_rng(seed)
    with torch.no_grad():
        for n in model.param_names():
            p = model.P(n)
            p.add_(torch.as_tensor(g.standard_normal(p.shape) * 0.02 * max(float(p.abs().max()), 1e-3), dtype=p.dtype))
    wd = tmpdir("bsce_orig_")
    save_model(model, wd / "net")
    shutil.copy(wd / "net_orig.mat", wd / fname)
    for m in mfiles:
        shutil.copy(m, wd)
    sio.savemat(wd / "in.mat", {"y": y, "A": A, "x": x})
    run_octave(f"load in.mat; {call} save('-v7', 'out.mat', 'xh');", wd)
    with torch.no_grad():
        xp = model(torch.as_tensor(y)).numpy()
    r = compare(xp, sio.loadmat(wd / "out.mat")["xh"], f"{cfg} original MATLAB", 1e-12)
    record("export/original_matlab", model=cfg["model"] + ("-" + cfg["operator"] if "operator" in cfg else ""),
           function=call.split("=")[1].split("(")[0].strip(), seed=seed, rel=r["rel"])
    shutil.rmtree(wd)

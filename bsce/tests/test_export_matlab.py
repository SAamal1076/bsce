"""Export round trip: every learned model -> .pt and .mat; reload in Python (both formats)
and run the MATLAB inference (matlab/bsce_infer.m) in Octave; all must give the same
estimate as the PyTorch model (complex beamspace channels, perturbed weights, ULA & UPA).
"""
import shutil
from pathlib import Path

import numpy as np
import pytest
import scipy.io as sio
import torch

from harness.octave import REPO, compare, octave_available, run_octave, tmpdir
from bsce.data import sv, measurements
from bsce.rng import NumpyMatlabRNG
from bsce.models import build_model
from bsce.export import save_model, load_model

MODELS = [
    dict(model="LAMP", T=4),
    dict(model="GM-LAMP", T=4, K=4),
    dict(model="LADMM", T=4, operator="full"),
    dict(model="LADMM", T=4, operator="diag"),
    dict(model="LADMM", T=4, operator="lowrank", rank=8),
    dict(model="LADMM", T=4, operator="kron"),
    dict(model="LADMM", T=4, operator="psd"),
    dict(model="LADMM", T=4, operator="kronprior"),
    dict(model="LADMM", T=4, operator="diagprior"),
    dict(model="LISTA", T=5),
    dict(model="ALISTA", T=5, percent=3.0, max_percent=12.0),
    dict(model="LVAMP", T=4),
    dict(model="ADMM-Net", T=4),
]


def _id(c):
    return c["model"] + ("-" + c["operator"] if "operator" in c else "")


@pytest.mark.parametrize("cfg", MODELS, ids=[_id(c) for c in MODELS])
@pytest.mark.parametrize("seed", [0, 1])
def test_export_roundtrip_and_matlab(cfg, seed, record):
    g = np.random.default_rng(seed)
    A = ((g.random((128, 256)) > 0.5) * 2 - 1) / np.sqrt(128)
    array = "ULA" if seed == 0 else "UPA"
    x = sv.beamspace_channels(NumpyMatlabRNG(seed), 40, array)
    y = measurements.gen_measurements_matlab(x, A, 8.0, NumpyMatlabRNG(seed + 1))
    m = build_model({**cfg, "precision": "float64"}, A)
    with torch.no_grad():   # generic weights
        for n in m.param_names():
            p = m.P(n)
            noise = g.standard_normal(p.shape) + (1j * g.standard_normal(p.shape) if p.is_complex() else 0)
            p.add_(torch.as_tensor(0.01 * noise * max(float(p.abs().max()), 0.1), dtype=p.dtype))
        ref = m(torch.as_tensor(y)).numpy()
    wd = tmpdir("bsce_exp_")
    pt, mat = save_model(m, wd / "net")
    for f in (pt, mat):
        m2 = load_model(f, precision="float64")
        with torch.no_grad():
            r = compare(m2(torch.as_tensor(y)).numpy(), ref, f"reload {f.suffix}", 1e-13)
    out = {"py_pt_rel": 0.0}
    if octave_available():
        sio.savemat(wd / "in.mat", {"y": y})
        run_octave(f"net = bsce_load('net.mat'); load in.mat; xh = bsce_infer(net, y); save('-v7','out.mat','xh');",
                   wd, addpaths=[REPO / "matlab"])
        xm = sio.loadmat(wd / "out.mat")["xh"]
        rm = compare(xm, ref, f"MATLAB {_id(cfg)}", 1e-11)
        out["matlab_rel"] = rm["rel"]
    record("export/roundtrip", model=_id(cfg), seed=seed, mat_reload_rel=r["rel"], **out)
    shutil.rmtree(wd)

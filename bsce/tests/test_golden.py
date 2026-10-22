"""Golden numbers: short deterministic trainings of every learned model, PyTorch only.

The reference numbers in tests/golden/golden.json were produced on CPU (float64) with the
implementation verified against the original code (see VERIFICATION_REPORT.md). Running this
file on another machine or on the GPU (device chosen by BSCE_DEVICE, default 'auto') checks
that the installation reproduces them. Needs neither Octave nor TensorFlow (~2 min on CPU).

  python -m pytest tests/test_golden.py -q                 # check
  BSCE_UPDATE_GOLDEN=1 python -m pytest tests/test_golden.py   # (re)write the golden file
"""
import json
import os
from pathlib import Path

import numpy as np
import pytest
import torch

from bsce.data import sv, measurements
from bsce.device import get_device
from bsce.models import build_model, alista_W
from bsce.rng import NumpyMatlabRNG
from bsce.train.trainer import Trainer, TrainConfig

GOLDEN = Path(__file__).resolve().parent / "golden" / "golden.json"
DEVICE = get_device(os.environ.get("BSCE_DEVICE", "auto"))
TOL = 1e-9 if DEVICE.type == "cpu" else 1e-7      # GPU kernels sum in a different order

CASES = {
    "LAMP": (dict(model="LAMP", T=2, precision="float64"), dict(warm_start=True)),
    "GM-LAMP-spread": (dict(model="GM-LAMP", T=2, K=4, precision="float64", gm_init="spread"), dict(warm_start=True)),
    "LADMM-full": (dict(model="LADMM", T=2, operator="full", precision="float64", init_precision="float64"), {}),
    "LADMM-kron": (dict(model="LADMM", T=2, operator="kron", precision="float64", init_precision="float64"), {}),
    "LADMM-diag": (dict(model="LADMM", T=2, operator="diag", precision="float64", init_precision="float64"), {}),
    "LADMM-lowrank": (dict(model="LADMM", T=2, operator="lowrank", rank=8, precision="float64", init_precision="float64"), {}),
    "LADMM-kronprior": (dict(model="LADMM", T=2, operator="kronprior", precision="float64"), {}),
    "LADMM-psd": (dict(model="LADMM", T=2, operator="psd", precision="float64"), {}),
    "LADMM-diagprior": (dict(model="LADMM", T=2, operator="diagprior", precision="float64"), {}),
    "LISTA": (dict(model="LISTA", T=3, precision="float64"), dict(schedule="alista", loss="l2", val_metric="nmse_l2", val_round=None, check_mult=10)),
    "ALISTA": (dict(model="ALISTA", T=3, percent=1.2, max_percent=13.0, precision="float64"), dict(schedule="alista", loss="l2", val_metric="nmse_l2", val_round=None, check_mult=10)),
    "LVAMP": (dict(model="LVAMP", T=2, precision="float64"), dict(loss="l2", val_metric="nmse_l2", val_round=None, lr=1e-4, refine_name="value")),
    "ADMM-Net": (dict(model="ADMM-Net", T=3, precision="float64"), dict(schedule="end2end", epochs=2, optimizer="adam_torch", val_round=None)),
}


@pytest.fixture(scope="module")
def data():
    g = np.random.default_rng(0)
    A = ((g.random((128, 256)) > 0.5) * 2 - 1) / np.sqrt(128)
    xt = sv.beamspace_channels(NumpyMatlabRNG(0), 400, "ULA")
    xv = sv.beamspace_channels(NumpyMatlabRNG(1), 100, "ULA")
    yt = measurements.gen_measurements_matlab(xt, A, 10.0, NumpyMatlabRNG(2))
    yv = measurements.gen_measurements_matlab(xv, A, 10.0, NumpyMatlabRNG(3))
    return A, xt, yt, xv, yv


@pytest.fixture(scope="module")
def golden():
    ref = json.loads(GOLDEN.read_text(encoding="utf-8")) if GOLDEN.exists() else {}
    new = {}
    yield ref, new
    if os.environ.get("BSCE_UPDATE_GOLDEN") and new:
        GOLDEN.parent.mkdir(exist_ok=True)
        GOLDEN.write_text(json.dumps({**ref, **new}, indent=1, sort_keys=True), encoding="utf-8")


@pytest.mark.parametrize("name", list(CASES))
def test_golden(name, data, golden, record):
    A, xt, yt, xv, yv = data
    mcfg, tkw = CASES[name]
    mcfg = dict(mcfg)
    if mcfg["model"] == "ALISTA":
        mcfg["W"] = alista_W(A)
    torch.manual_seed(0)
    model = build_model(mcfg, A, device=DEVICE)
    cfg = TrainConfig(**{**dict(lr=1e-3, maxit=20, better_wait=10, ivl=10, batch_size=64, loss="nmse",
                                 val_round=5, seed=123, verbose=False, lr_decay=(0.2, 0.02)), **tkw})
    state = Trainer(model, cfg, xt, yt, xv, yv).run()
    with torch.no_grad():
        xh = model(torch.as_tensor(yv).to(DEVICE)).cpu().numpy()
    nmse = float(np.sum(np.abs(xh - xv) ** 2) / np.sum(np.abs(xv) ** 2))
    psum = float(sum(p.detach().abs().sum().item() for p in model.parameters()))
    out = dict(val_nmse=nmse, param_abs_sum=psum, n_stages=len(state["done"]))
    ref, new = golden
    new[name] = out
    record("golden", model=name, device=str(DEVICE), **out,
           rel_nmse=abs(nmse - ref[name]["val_nmse"]) / ref[name]["val_nmse"] if name in ref else None)
    if os.environ.get("BSCE_UPDATE_GOLDEN") or name not in ref:
        pytest.skip("golden value written / not available")
    assert out["n_stages"] == ref[name]["n_stages"]
    if name == "LADMM-lowrank":
        # The low-rank init (B_*_LowRank.py) takes the top-r singular vectors of W = (A^T A + I)^-1, whose
        # largest singular value 1 has multiplicity N - M = 128: the chosen vectors are arbitrary and
        # depend on the LAPACK build (MKL on Windows vs OpenBLAS on Linux). Same-machine runs are exact
        # (test_gpu_vs_cpu.py); across machines only statistical agreement can be expected.
        assert abs(nmse - ref[name]["val_nmse"]) <= 0.02 * ref[name]["val_nmse"]
        return
    assert abs(nmse - ref[name]["val_nmse"]) <= TOL * ref[name]["val_nmse"]
    assert abs(psum - ref[name]["param_abs_sum"]) <= TOL * ref[name]["param_abs_sum"]

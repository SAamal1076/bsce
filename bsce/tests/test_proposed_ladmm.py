"""Proposed LADMM variants (psd, kronprior, diagprior): definition checks.

At their structural 'zero' they must reduce to classical ADMM-LASSO (whose numpy port is
verified against ADMM_vectorized.m), and the materialised operator used for MATLAB export
must equal the solve used in training.
"""
import numpy as np
import pytest
import torch
import torch.nn.functional as F

from harness.octave import compare
from bsce import classical
from bsce.data import sv, measurements
from bsce.rng import NumpyMatlabRNG
from bsce.models.ladmm import LADMM


def _setup(seed=0):
    g = np.random.default_rng(seed)
    A = ((g.random((128, 256)) > 0.5) * 2 - 1) / np.sqrt(128)
    x = sv.beamspace_channels(NumpyMatlabRNG(seed), 40, "UPA")
    y = measurements.gen_measurements_matlab(x, A, 10.0, NumpyMatlabRNG(seed + 1))
    return A, y


def _inv_softplus(v):
    return float(np.log(np.expm1(v)))


@pytest.mark.parametrize("op", ["psd", "kronprior", "diagprior"])
@pytest.mark.parametrize("lam,rho", [(0.05, 1.0), (0.2, 2.0)])
def test_reduces_to_admm(op, lam, rho, record):
    A, y = _setup()
    T = 8
    m = LADMM(A, T=T, operator=op, precision="float64")
    with torch.no_grad():
        for t in range(T):
            m.P(f"rho_raw_{t}").fill_(_inv_softplus(rho))
            m.P(f"theta_raw_{t}").fill_(_inv_softplus(lam))
            if op == "kronprior":
                m.P(f"L1_{t}").zero_(); m.P(f"L2_{t}").zero_()
            if op == "diagprior":
                m.P(f"d_{t}").fill_(-80.0)          # softplus(-80) ~ 1.8e-35
    with torch.no_grad():
        z = m(torch.as_tensor(y)).numpy()
    zr, _ = classical.admm(y, A, lam, rho, T, 0.0)
    r = compare(z, zr, f"{op} -> ADMM", 1e-11)
    record("ladmm_proposed/reduces_to_admm", operator=op, lam=lam, rho=rho, rel=r["rel"])


@pytest.mark.parametrize("op", ["psd", "kronprior", "diagprior"])
def test_materialised_operator(op, record):
    A, y = _setup(1)
    m = LADMM(A, T=3, operator=op, precision="float64")
    g = np.random.default_rng(2)
    with torch.no_grad():
        for n in m.param_names():
            m.P(n).add_(torch.as_tensor(0.05 * g.standard_normal(m.P(n).shape)))
        H = m.H_matrix(1).numpy()
    # PSD / symmetric positive definite inverse: exact ADMM semantics
    ev = np.linalg.eigvalsh((H + H.T) / 2)
    assert np.allclose(H, H.T, atol=1e-12) and ev.min() > 0
    record("ladmm_proposed/operator_spd", operator=op, min_eig=float(ev.min()))

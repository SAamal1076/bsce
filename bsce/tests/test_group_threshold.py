"""LADMM group threshold (option threshold='group' | 'group_k' | 'group_kc').

The fixed variant must equal the group weights of ADMM-Improved (local_energy_weight in
ADMM_improved.m / group_weights in admm_lib.py), reproduced here as an independent NumPy reference:
UPA = 3 x 3 circular box on the 16 x 16 beam grid (N2 index fastest), ULA = grid (256, 1).
"""
import numpy as np
import pytest
import torch

from bsce.models import build_model


def ref_weights(V, N1, N2, win=3, mix=3.0):
    N, K = V.shape
    E = (np.abs(V) ** 2).reshape(N1, N2, K)
    r = (win - 1) // 2
    S0 = sum(np.roll(E, s, axis=0) for s in range(-r, r + 1))
    S = sum(np.roll(S0, s, axis=1) for s in range(-r, r + 1))
    g = np.sqrt(S / win ** 2).reshape(N, K)
    g = g / (g.mean(axis=0, keepdims=True) + 1e-12)
    return 1.0 / (1.0 + mix * g)


@pytest.fixture(scope="module")
def A():
    return ((np.random.default_rng(0).random((128, 256)) > 0.5) * 2 - 1) / np.sqrt(128)


@pytest.mark.parametrize("array,grid", [("UPA", (16, 16)), ("ULA", (256, 1))])
def test_fixed_group_equals_admm_improved(A, array, grid, record):
    g = np.random.default_rng(1)
    V = g.standard_normal((256, 9)) + 1j * g.standard_normal((256, 9))
    m = build_model(dict(model="LADMM", T=1, threshold="group", array=array, precision="float64"), A)
    w = m._group_weights(0, torch.as_tensor(V)).numpy()
    err = float(np.abs(w - ref_weights(V, *grid)).max())
    record("ladmm/group_threshold", array=array, max_abs_err=err)
    assert err < 1e-10


def test_learned_variants_start_at_fixed(A):
    """group_k / group_kc are initialised at kappa = 3 and a uniform window: same output as 'group'."""
    g = np.random.default_rng(2)
    y = torch.as_tensor(g.standard_normal((128, 6)) + 1j * g.standard_normal((128, 6)))
    outs = []
    for th in ("group", "group_k", "group_kc"):
        torch.manual_seed(0)
        m = build_model(dict(model="LADMM", T=2, threshold=th, array="UPA", precision="float64"), A)
        with torch.no_grad():
            outs.append(m(y).numpy())
    assert np.abs(outs[1] - outs[0]).max() < 1e-10
    assert np.abs(outs[2] - outs[0]).max() < 1e-10


def test_extra_parameters_and_gradients(A):
    y = torch.randn(128, 4, dtype=torch.complex128)
    n0 = build_model(dict(model="LADMM", T=2, precision="float64"), A).n_params()
    for th, extra in (("group", 0), ("group_k", 2), ("group_kc", 2 * (1 + 9))):
        m = build_model(dict(model="LADMM", T=2, threshold=th, array="UPA", precision="float64"), A)
        assert m.n_params() == n0 + extra
        m(y).abs().sum().backward()
        for n in m.param_names():
            if n == "alpha_1":          # last-layer alpha never reaches the output (known, as in the scripts)
                continue
            assert m.P(n).grad is not None and torch.isfinite(m.P(n).grad).all(), n

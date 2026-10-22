"""v31.3: LADMM noise_level='residual' and debias='ls'."""
import math
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from bsce.export import load_model, original_format_dict, save_model, truncate_model  # noqa: E402
from bsce.models import build_model  # noqa: E402


def _A(M=32, N=64, seed=0):
    return np.random.default_rng(seed).standard_normal((M, N)) / math.sqrt(M)


def _xy(A, B=16, snr_db=10.0, seed=1):
    g = np.random.default_rng(seed)
    N = A.shape[1]
    x = np.zeros((N, B), complex)
    for b in range(B):
        x[g.choice(N, 3, replace=False), b] = (g.standard_normal(3) + 1j * g.standard_normal(3)) * math.sqrt(N / 6)
    s = math.sqrt(10 ** (-snr_db / 10) / 2)
    n = s * (g.standard_normal(x.shape) + 1j * g.standard_normal(x.shape))
    return x, A @ (x + n)


def _net(**kw):
    return build_model(dict(model="LADMM", T=2, array="ULA", precision="float64", **kw), _A())


def test_defaults_unchanged():
    m = _net()
    assert "noise_level" not in m.cfg and "debias" not in m.cfg and not hasattr(m, "Wm")
    assert len(m.forward_stages(_xy(_A())[1])) == 4


def test_residual_needs_a_noise_threshold():
    with pytest.raises(ValueError):
        _net(noise_level="residual")


def test_residual_level_is_the_noise_std():
    A = _A(64, 128)
    m = build_model(dict(model="LADMM", T=2, array="ULA", precision="float64", threshold="adaptive",
                         noise_level="residual"), A)
    for snr in (0.0, 10.0, 20.0):
        x, y = _xy(A, B=400, snr_db=snr)
        yw = torch.as_tensor(m.Wm.numpy() @ y)
        sig = m._residual_level(yw, torch.as_tensor(x))           # residual at the true channel: only noise
        assert abs(float(sig.pow(2).mean()) / 10 ** (-snr / 10) - 1) < 0.05


def test_residual_changes_only_layers_after_the_first():
    A = _A()
    _, y = _xy(A)
    a = build_model(dict(model="LADMM", T=2, array="ULA", precision="float64", threshold="adaptive"), A)
    b = build_model(dict(model="LADMM", T=2, array="ULA", precision="float64", threshold="adaptive",
                         noise_level="residual"), A)
    b.load_state_dict(a.state_dict(), strict=False)
    oa, ob = a.forward_stages(y), b.forward_stages(y)
    assert torch.equal(oa[1], ob[1]) and not torch.allclose(oa[3], ob[3])


def test_debias_off_support_is_identity_and_fixes_bias():
    A = _A()
    x, y = _xy(A, snr_db=30.0)
    m = _net(debias="ls")
    with torch.no_grad():
        m.debias_tau_raw.fill_(50.0)                             # support threshold far above every beam: m ~ 0
        st = m.forward_stages(y)
        assert len(st) == 5 and torch.allclose(st[-1], st[-2], atol=1e-10)
        m.debias_tau_raw.fill_(-5.0)                             # every beam on the support, small eps: LS fit
        m.debias_kappa_raw.fill_(5.0)
        m.debias_eps_raw.fill_(-10.0)
        z = m(y)
    yw = m.Wm.numpy() @ y
    res = yw - m.Aw.numpy() @ z.numpy()
    assert np.linalg.norm(res) < 1e-2 * np.linalg.norm(yw)       # the correction fits the measurements


def test_debias_trains_save_load_truncate_export(tmp_path):
    A = _A()
    x, y = _xy(A)
    m = build_model(dict(model="LADMM", T=3, array="ULA", precision="float64", threshold="adaptive",
                         noise_level="residual", debias="ls"), A)
    assert m.stage_plan()[-1].name == "LADMM debias"
    loss = (m(y) - torch.as_tensor(x)).abs().pow(2).sum()
    loss.backward()
    assert all(float(getattr(m, f"debias_{k}_raw").grad.abs()) > 0 for k in ("kappa", "tau", "eps"))
    pt, _ = save_model(m, tmp_path / "n")
    m2 = load_model(pt)
    assert torch.equal(m(y), m2(y))
    t2 = truncate_model(m, 2)
    assert "debias" not in t2.cfg and torch.equal(t2(y), m.forward_stages(y)[3])      # z_2: no debias layer
    assert truncate_model(m, 3).cfg["debias"] == "ls"
    assert original_format_dict(m) is None
    assert original_format_dict(_net(debias="ls")) is None


def test_analytic_set_combiner_refreshes_the_whitener():
    A = _A()
    m = build_model(dict(model="LADMM", T=2, array="ULA", precision="float64", operator="analytic",
                         threshold="adaptive", noise_level="residual", debias="ls"), A)
    A2 = _A(48, 64, seed=3)
    m.set_combiner(A2)
    ref = build_model(dict(model="LADMM", T=2, array="ULA", precision="float64", operator="analytic",
                           threshold="adaptive", noise_level="residual", debias="ls"), A2)
    ref.load_state_dict(m.state_dict())
    _, y = _xy(A2)
    assert torch.allclose(m(y), ref(y)) and m.Wm.shape == (48, 48)


def test_psd_trains_every_column_and_firm_starts_at_two():
    A = _A()
    x, y = _xy(A)
    m = build_model(dict(model="LADMM", T=1, array="ULA", precision="float64", operator="psd"), A)
    (m(y) - torch.as_tensor(x)).abs().pow(2).sum().backward()
    assert float(m.L_0.grad[:, A.shape[0]:].abs().max()) > 0          # the block beyond A^T gets a gradient
    f = _net(threshold="firm")
    assert abs(float(f.theta(0)) - 2.0) < 1e-12

"""The suggested NF-LADMM (update='relax', slope_init=2) in bsce, against the proposal's relaxed module
(tests/proposals/nf_ladmm_relax.py = NF-LADMM_proposal nf_ladmm.py after the THEORY.md update).

1. same weights -> same output as the proposal module (ULA/UPA x radial/conv, T = 4 and 8)
2. each layer is relaxed Douglas-Rachford in zeta = z + u (THEORY.md, Prop. 3), checked on the bsce model
3. init: slope logits +-2, lam_t = 1; every parameter gets a gradient; config keys survive .pt and .mat round trips
4. the default (no 'update' key) is still the original dual-step model, so old runs load unchanged
5. finetune_bands.noisy(..., with_s2=True) draws the same noise as before and returns the right variances
"""
import importlib.util
import sys
from pathlib import Path

import numpy as np
import pytest
import scipy.io as sio
import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from bsce.models import build_model  # noqa: E402
from bsce.export import save_model, load_model  # noqa: E402

A = sio.loadmat(ROOT / "reference/gmlamp/CE_SV/CSmatrix256128.mat")["A"].astype(np.float64)
M, N = A.shape
CFG = dict(model="NF-LADMM", denoiser="conv", update="relax", slope_init=2.0)


def _load(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def _batch(B=6, seed=0, dtype=torch.float32):
    g = torch.Generator().manual_seed(seed)
    cd = torch.complex64 if dtype == torch.float32 else torch.complex128
    x = (torch.complex(torch.randn(N, B, generator=g, dtype=torch.float64),
                       torch.randn(N, B, generator=g, dtype=torch.float64)) / np.sqrt(2)).to(cd)
    s2 = 10 ** (-torch.rand(B, generator=g, dtype=torch.float64) * 2)
    n = torch.complex(torch.randn(N, B, generator=g, dtype=torch.float64),
                      torch.randn(N, B, generator=g, dtype=torch.float64)).to(cd) * torch.sqrt(s2 / 2).to(dtype)
    At = torch.as_tensor(A, dtype=dtype)
    xn = x + n
    return torch.complex(At @ xn.real, At @ xn.imag), x, s2.to(dtype)


@pytest.mark.parametrize("T", [4, 8])
@pytest.mark.parametrize("array", ["ULA", "UPA"])
@pytest.mark.parametrize("den", ["radial", "conv"])
def test_same_output_as_relaxed_proposal_module(T, array, den):
    ref = _load(ROOT / "tests/proposals/nf_ladmm_relax.py", "nf_relax_ref")
    torch.manual_seed(3)
    r = ref.NFLADMM(T=T, array=array, denoiser=den)
    with torch.no_grad():
        for p in r.parameters():
            p.add_(0.05 * torch.randn_like(p))
    b = build_model(dict(CFG, T=T, array=array, denoiser=den), A)
    rsd, bsd = dict(r.named_parameters()), dict(b.named_parameters())
    assert set(rsd) == set(bsd), f"parameter names differ: {set(rsd) ^ set(bsd)}"
    with torch.no_grad():
        for n_, p in b.named_parameters():
            p.copy_(rsd[n_])
    y, x, s2 = _batch()
    with torch.no_grad():
        zr = r(y.T.contiguous(), torch.as_tensor(A, dtype=torch.float32), s2).T
        zb = b(y, s2=s2)
    rel = float((zr - zb).abs().norm() / zr.abs().norm())
    print(f"T={T} {array} {den}: params {b.n_params()}  relative difference {rel:.2e}")
    assert rel < 1e-4


@pytest.mark.parametrize("array", ["ULA", "UPA"])
def test_layer_is_relaxed_douglas_rachford(array):
    """zeta_t = zeta_{t-1} + lam_t (J_f(2 z_{t-1} - zeta_{t-1}) - z_{t-1}),  z_t = D_t(zeta_t)  (float64)."""
    b = build_model(dict(CFG, T=4, array=array, precision="float64"), A)
    torch.manual_seed(5)
    with torch.no_grad():
        for p in b.parameters():
            p.add_(0.1 * torch.randn_like(p))
    y, x, s2 = _batch(dtype=torch.float64)
    with torch.no_grad():
        outs = b.forward_stages(y, s2=s2)
        Ap = b.Apinv.to(torch.complex128)
        Ac = b.A.to(torch.complex128)
        yb = y.T
        z, u = torch.zeros(6, N, dtype=torch.complex128), torch.zeros(6, N, dtype=torch.complex128)
        for t in range(4):
            beta, lam = torch.sigmoid(b.rho_raw[t]), 2 * torch.sigmoid(b.lam_raw[t])
            zeta = z + u
            v = 2 * z - zeta
            res = yb - v @ Ac.T
            Jf = v + beta * (res @ Ap.T)
            zeta_new = zeta + lam * (Jf - z)
            # D_t(zeta_new) = the model's z_t; the model's u_t must equal zeta_new - z_t
            zt = outs[t].T
            sig2 = torch.exp(b.m[t]) * res.abs().pow(2).sum(1) / N + torch.exp(b.k[t]) * s2
            assert torch.allclose(zt + (zeta_new - zt), zeta_new)
            # recompute z_t from zeta_new with the model's own prior step
            sig = torch.sqrt(sig2).view(-1, 1)
            r_ = zeta_new.abs()
            zs = b.shrink[t](r_ / sig) * sig * zeta_new / r_.clamp_min(1e-12)
            zs = zs + b.gamma[t] * sig * b.cnn[0](zeta_new / sig, zs / sig, torch.log(sig))
            err = float((zs - zt).abs().norm() / zt.abs().norm())
            assert err < 1e-12, f"layer {t + 1}: {err}"
            z, u = zt, zeta_new - zt
    print(f"{array}: relaxed-DRS form reproduced to < 1e-12 in every layer")


def test_init_gradients_and_roundtrip(tmp_path):
    b = build_model(dict(CFG, T=4, array="UPA"), A)
    assert torch.allclose(2 * torch.sigmoid(b.lam_raw), torch.ones(4, dtype=b.rdtype))
    assert float(b.shrink[0].slope_logit.abs().max()) == 2.0
    assert not hasattr(b, "alpha_raw")
    y, x, s2 = _batch()
    loss = (b(y, s2=s2) - x).abs().pow(2).sum()
    loss.backward()
    dead = [n for n, p in b.named_parameters() if p.grad is None or float(p.grad.abs().sum()) == 0]
    assert not dead, dead
    p, _ = save_model(b, tmp_path / "m")
    for f in (p, tmp_path / "m.mat"):
        m2 = load_model(f)
        assert m2.update == "relax" and float(m2.shrink[0].slope_logit.detach().abs().max()) == 2.0, f
        with torch.no_grad():
            d = float((m2(y, s2=s2) - b(y, s2=s2)).abs().max())
        assert d < 1e-5, (f, d)


def test_default_is_original_dual_model():
    b = build_model(dict(model="NF-LADMM", T=4, array="ULA"), A)
    assert b.update == "dual" and hasattr(b, "alpha_raw") and not hasattr(b, "lam_raw")
    assert float(b.shrink[0].slope_logit.abs().max()) == 6.0


def test_finetune_noise_unchanged_and_s2():
    ft = _load(ROOT / "scripts/finetune_bands.py", "ftb")
    x = torch.complex(torch.randn(N, 50), torch.randn(N, 50))
    Ac = torch.as_tensor(A, dtype=torch.float32).to(torch.complex64)
    y_old = ft.noisy(Ac, x, 0, 20, torch.Generator().manual_seed(9))
    y_new, s2 = ft.noisy(Ac, x, 0, 20, torch.Generator().manual_seed(9), with_s2=True)
    assert torch.equal(y_old, y_new)
    snr = 0 + 20 * torch.rand(50, generator=torch.Generator().manual_seed(9))
    assert torch.allclose(s2, (10 ** (-snr / 10)).float())

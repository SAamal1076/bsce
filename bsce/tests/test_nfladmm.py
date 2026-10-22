"""NF-LADMM in bsce (bsce/models/nfladmm.py) against the proposal's module (tests/proposals/nf_ladmm.py).

1. same weights -> same output as the proposal module (ULA/UPA x radial/conv, float32)
2. the x-step is the exact prox of the whitened data term (closed form vs linear solve, float64)
3. every parameter gets a gradient; save_model / load_model round trip
4. Trainer: schedule 'steps' with online noise lowers the validation NMSE; s2 is required with sigma='known'
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
from bsce.train.trainer import Trainer, TrainConfig  # noqa: E402
from bsce.data.files import online_validation  # noqa: E402

A = sio.loadmat(ROOT / "reference/gmlamp/CE_SV/CSmatrix256128.mat")["A"].astype(np.float64)
M, N = A.shape


def _ref():
    spec = importlib.util.spec_from_file_location("nf_ladmm_ref", ROOT / "tests/proposals/nf_ladmm.py")
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def _batch(B=6, seed=0):
    g = torch.Generator().manual_seed(seed)
    x = torch.complex(torch.randn(N, B, generator=g), torch.randn(N, B, generator=g)) / np.sqrt(2)
    s2 = 10 ** (-torch.rand(B, generator=g, dtype=torch.float64) * 2)            # 0..20 dB
    n = torch.complex(torch.randn(N, B, generator=g), torch.randn(N, B, generator=g)) * torch.sqrt(s2 / 2).float()
    At = torch.as_tensor(A, dtype=torch.float32)
    xn = x + n
    y = torch.complex(At @ xn.real, At @ xn.imag)
    return y, x, s2.float()


@pytest.mark.parametrize("array", ["ULA", "UPA"])
@pytest.mark.parametrize("den", ["radial", "conv"])
def test_same_output_as_proposal_module(array, den):
    ref = _ref()
    torch.manual_seed(3)
    r = ref.NFLADMM(T=8, array=array, denoiser=den)
    with torch.no_grad():                         # move every parameter away from its init
        for p in r.parameters():
            p.add_(0.05 * torch.randn_like(p))
    b = build_model(dict(model="NF-LADMM", T=8, array=array, denoiser=den), A)
    rsd = dict(r.named_parameters())
    with torch.no_grad():
        for n_, p in b.named_parameters():
            p.copy_(rsd[n_])
    assert set(rsd) == set(dict(b.named_parameters())), "parameter names differ"
    y, x, s2 = _batch()
    with torch.no_grad():
        zr = r(y.T.contiguous(), torch.as_tensor(A, dtype=torch.float32), s2).T
        zb = b(y, s2=s2)
    rel = float((zr - zb).abs().norm() / zr.abs().norm())
    print(f"{array} {den}: params {b.n_params()}  relative difference to the proposal module {rel:.2e}")
    assert rel < 1e-4


def test_xstep_is_exact_prox():
    Ad = torch.as_tensor(A)
    Ap = Ad.T @ torch.linalg.inv(Ad @ Ad.T)
    rho, s = 0.7, 0.1
    g = torch.Generator().manual_seed(1)
    v = torch.complex(torch.randn(N, generator=g, dtype=torch.float64), torch.randn(N, generator=g, dtype=torch.float64))
    yy = torch.complex(torch.randn(M, generator=g, dtype=torch.float64), torch.randn(M, generator=g, dtype=torch.float64))
    G = (Ap @ Ad).to(torch.cdouble)
    xi = torch.linalg.solve(G / s + rho * torch.eye(N, dtype=torch.cdouble), Ap.to(torch.cdouble) @ yy / s + rho * v)
    xc = v + Ap.to(torch.cdouble) @ (yy - Ad.to(torch.cdouble) @ v) / (1 + rho * s)
    assert float((xi - xc).abs().norm() / xi.abs().norm()) < 1e-12


def test_gradients_and_export(tmp_path):
    b = build_model(dict(model="NF-LADMM", T=4, array="UPA", denoiser="conv"), A)
    y, x, s2 = _batch()
    loss = (b(y, s2=s2) - x).abs().pow(2).sum()
    loss.backward()
    dead = [n for n, p in b.named_parameters() if p.grad is None or float(p.grad.abs().sum()) == 0]
    assert not dead, dead
    save_model(b, tmp_path / "nf")
    b2 = load_model(tmp_path / "nf.pt")
    with torch.no_grad():
        assert torch.allclose(b(y, s2=s2), b2(y, s2=s2))
    with pytest.raises(ValueError):
        b(y)                                       # sigma='known' needs s2


def test_steps_schedule_online_noise():
    xs = sio.loadmat(ROOT / "data/SV_UPA_smoke/SV_UPA_training_channel.mat")["x"] if (
        ROOT / "data/SV_UPA_smoke/SV_UPA_training_channel.mat").exists() else _batch(400, 5)[1].numpy()
    xt, xv = xs[:, :300], xs[:, -100:]
    yv, s2v, _ = online_validation(xv, A, 0, 20, 11)
    torch.manual_seed(0)
    b = build_model(dict(model="NF-LADMM", T=3, array="UPA", denoiser="radial"), A)
    cfg = TrainConfig(schedule="steps", steps=60, lr=1e-2, val_every=20, batch_size=32, loss="log_nse",
                      val_round=None, restore_best=True, optimizer="adam_torch", noise="online", snr_range=[0, 20],
                      grad_clip=1.0, verbose=False)
    tr = Trainer(b, cfg, xt, None, xv, yv, s2_val=s2v)
    v0 = tr.validate(b.T - 1)
    tr.run()
    v1 = tr.validate(b.T - 1)
    print(f"online steps: val {10 * np.log10(v0):.3f} -> {10 * np.log10(v1):.3f} dB")
    assert v1 < v0

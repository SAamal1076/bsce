"""--save-layers: truncate_model(model, t) is the first t layers of the network: its output equals the output
validated when layer t is finished (the last output index of layer t in forward_stages), for every learned model."""
import numpy as np
import pytest
import torch

from bsce.export import truncate_model
from bsce.models import build_model, alista_W

RNG = np.random.default_rng(0)
A = RNG.standard_normal((32, 64)) / 8
CFGS = [dict(model="LADMM", T=5, operator="full"), dict(model="LAMP", T=4), dict(model="GM-LAMP", T=3, precision="float64"),
        dict(model="LISTA", T=4), dict(model="ALISTA", T=4), dict(model="LVAMP", T=3), dict(model="ADMM-Net", T=4),
        dict(model="NF-LADMM", T=3, array="ULA", N1=64, N2=1, denoiser="conv"),
        dict(model="NF-LADMM", T=3, array="ULA", N1=64, N2=1, denoiser="radial"),
        dict(model="NF-LADMM", T=3, array="ULA", N1=64, N2=1, denoiser="conv", update="relax", slope_init=2.0)]


@pytest.mark.parametrize("cfg", CFGS, ids=lambda c: f"{c['model']}-{c.get('denoiser', '')}{'-relax' if c.get('update') == 'relax' else ''}")
def test_truncate_matches_prefix_output(cfg):
    cfg = dict(cfg)
    if cfg["model"] == "ALISTA":
        cfg["W"] = alista_W(A)
    m = build_model(cfg, A)
    torch.manual_seed(1)
    with torch.no_grad():
        for p in m.parameters():
            p.add_(0.01 * torch.randn_like(p))
    y = torch.as_tensor(RNG.standard_normal((32, 7)) + 1j * RNG.standard_normal((32, 7)))
    kw = {"s2": np.full(7, 0.1)} if getattr(m, "needs_noise_var", False) else {}
    per = len(m.stage_plan()) // m.T
    with torch.no_grad():
        outs = m.forward_stages(y, **kw)
        for t in range(1, m.T + 1):
            s = truncate_model(m, t)
            assert s.T == t
            ref = outs[per * t - 1]
            assert torch.allclose(s(y, **kw), ref, rtol=1e-12, atol=1e-12), (cfg["model"], t)

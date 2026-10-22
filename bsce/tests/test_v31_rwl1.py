"""v31: Graph-LADMM's graph-reweighted l1 z-step (rwl1, gtv_rwl1) and its plain (unwhitened) data step.

* the weights equal the uploaded package's graph_weights (numpy formula written out here) with a1 = a2 = 1, on the UPA
  torus and on the ULA ring;
* layer 1 of rwl1 (z = 0, weights 1) is the plain l1 layer;
* data_step=plain: the x-step equals (A^T A + rho I [+ rho D^H D])^-1 (A^T y + ...), solved explicitly;
* every stored parameter gets a gradient (v31 does not create eps, a1, a2 of layer 1, nor e (and k2) of the last layer);
* save / reload, another combiner (any M), and the cfg of the whitened networks unchanged (older checkpoints load).
"""
import numpy as np
import pytest
import torch

from bsce.data import measurements, sv
from bsce.export import load_model, save_model
from bsce.models import build_model
from bsce.models.graph_ladmm import RW, ZPARAMS
from bsce.rng import NumpyMatlabRNG


def _A(seed=0, M=128, N=256):
    g = np.random.default_rng(seed)
    return ((g.random((M, N)) > 0.5) * 2 - 1) / np.sqrt(M)


def _cfg(**kw):
    return dict(dict(model="Graph-LADMM", T=3, array="UPA", N1=16, N2=16, precision="float64", zstep="rwl1"), **kw)


def _package_weights(Z, eps, array):
    """The package's graph_weights (rw_l1_graph.py), written out: closed-neighbourhood mean of |z|^2 (UPA: beam + 4
    torus neighbours; ULA: beam + offsets +-1, +-2), sqrt, / max, 1 / (a + eps), / mean."""
    P = np.abs(Z) ** 2
    if array == "UPA":
        P = P.reshape(16, 16, -1)
        S = P + np.roll(P, 1, 0) + np.roll(P, -1, 0) + np.roll(P, 1, 1) + np.roll(P, -1, 1)
        a = np.sqrt(S / 5.0).reshape(256, -1)
    else:
        S = P + np.roll(P, 1, 0) + np.roll(P, -1, 0) + np.roll(P, 2, 0) + np.roll(P, -2, 0)
        a = np.sqrt(S / 5.0)
    a = a / np.maximum(a.max(0, keepdims=True), 1e-12)
    w = 1.0 / (a + eps)
    return w / w.mean(0, keepdims=True)


@pytest.mark.parametrize("array", ["UPA", "ULA"])
def test_weights_equal_package(array, record):
    m = build_model(_cfg(array=array), _A(0))
    Z = sv.beamspace_channels(NumpyMatlabRNG(1), 7, array)
    for t in range(1, m.T):                       # layer 1 has no weights (z = 0: they are 1; v31 stores no eps there)
        eps = float(m.s("eps", t))
        assert abs(float(m.s("a1", t)) - 1) < 1e-12 and abs(float(m.s("a2", t)) - 1) < 1e-12
        with torch.no_grad():
            om = m.rw_weights(torch.as_tensor(Z), t).numpy()
        err = np.abs(om - _package_weights(Z, eps, array)).max()
        assert err < 1e-9, err
    record("rwl1_weights_equal_package", array=array, max_abs_err=float(err))


def test_layer1_is_plain_l1():
    A = _A(0)
    x = sv.beamspace_channels(NumpyMatlabRNG(1), 9, "UPA")
    y = torch.as_tensor(measurements.gen_measurements_matlab(x, A, 5.0, NumpyMatlabRNG(2)))
    for ds in ("whitened", "plain"):
        a, b = build_model(_cfg(data_step=ds), A), build_model(_cfg(zstep="l1", data_step=ds), A)
        with torch.no_grad():
            for t in range(a.T):
                b.P(f"k1_raw_{t}").copy_(a.P(f"k1_raw_{t}"))
            za, zb = a.forward_stages(y, upto=1)[-1], b.forward_stages(y, upto=1)[-1]
        assert torch.allclose(za, zb, atol=1e-12, rtol=0), ds


@pytest.mark.parametrize("zstep", ["rwl1", "gtv_rwl1"])
def test_plain_data_step_is_unwhitened_least_squares(zstep):
    A = _A(3)
    m = build_model(_cfg(zstep=zstep, data_step="plain", T=2), A)
    x = sv.beamspace_channels(NumpyMatlabRNG(4), 5, "UPA")
    y = measurements.gen_measurements_matlab(x, A, 10.0, NumpyMatlabRNG(5))
    with torch.no_grad():
        x1 = m.forward_stages(torch.as_tensor(y), upto=0)[-1].numpy()         # z = u = 0: x = (A^T A + rho G)^-1 A^T y
        rho = float(m.s("rho", 0))
        G = m._G().numpy() if m.tv else np.eye(256)
    ref = np.linalg.solve(A.T @ A + rho * G, A.T @ y)
    assert np.abs(x1 - ref).max() / np.abs(ref).max() < 1e-10
    assert m.cfg["data_step"] == "plain"


@pytest.mark.parametrize("zstep,ds", [("rwl1", "plain"), ("rwl1", "whitened"), ("gtv_rwl1", "whitened")])
def test_gradients_reach_all_but_structural(zstep, ds):
    A = _A(0)
    m = build_model(_cfg(zstep=zstep, data_step=ds, precision="float32"), A)
    x = sv.beamspace_channels(NumpyMatlabRNG(6), 32, "UPA")
    y = measurements.gen_measurements_matlab(x, A, 10.0, NumpyMatlabRNG(7))
    out = m(torch.as_tensor(y))
    (torch.abs(out - torch.as_tensor(x, dtype=out.dtype)) ** 2).sum().backward()
    dead = {n for n, p in m.named_parameters() if p.grad is None or float(p.grad.abs().max()) == 0.0}
    assert dead == set(), dead                     # v31: the structural ones are not created
    last = m.T - 1
    gone = {"eps_raw_0", "a1_raw_0", "a2_raw_0", f"e_raw_{last}"} | ({f"k2_raw_{last}"} if m.tv else set())
    assert gone == m._dead_names() and not gone & {n for n, _ in m.named_parameters()}
    assert sum(p.numel() for p in m.parameters()) == m.T * (2 + len(ZPARAMS[zstep])) - len(gone)


@pytest.mark.parametrize("ds", ["plain", "whitened"])
def test_save_reload_and_any_M(ds, tmp_path):
    A0, A1 = _A(0), _A(9, M=96)
    m = build_model(_cfg(data_step=ds, T=2), A0)
    g = torch.Generator().manual_seed(1)
    with torch.no_grad():
        for p in m.parameters():
            p.add_(0.05 * torch.randn(p.shape, generator=g, dtype=p.dtype))
    x = sv.beamspace_channels(NumpyMatlabRNG(2), 6, "UPA")
    y0 = torch.as_tensor(measurements.gen_measurements_matlab(x, A0, 8.0, NumpyMatlabRNG(3)))
    save_model(m, tmp_path / "net")
    r = load_model(tmp_path / "net.pt")
    with torch.no_grad():
        assert torch.allclose(m(y0), r(y0), atol=1e-12, rtol=0)
    f = build_model(_cfg(data_step=ds, T=2), A1)
    with torch.no_grad():
        for k, v in m.named_parameters():
            f.get_parameter(k).copy_(v)
    m.set_combiner(A1)
    y1 = torch.as_tensor(measurements.gen_measurements_matlab(x, A1, 8.0, NumpyMatlabRNG(3)))
    with torch.no_grad():
        assert torch.allclose(m(y1), f(y1), atol=1e-10, rtol=0)


def test_whitened_cfg_and_buffers_unchanged():
    """A whitened network (every network before v31) has no data_step key and no new buffer, so older checkpoints load."""
    m = build_model(dict(model="Graph-LADMM", T=2, array="ULA"), _A(0))
    assert "data_step" not in m.cfg
    assert not {"Vr", "sv2"} & {n for n, _ in m.named_buffers()}
    assert RW == ("rwl1", "gtv_rwl1")


def test_refusals():
    with pytest.raises(ValueError):
        build_model(_cfg(data_step="white"), _A(0))
    with pytest.raises(ValueError):
        build_model(_cfg(data_step="plain", xstep="free"), _A(0))

"""v29: Graph-LADMM (magnetic graph TV, energy gate, GM-MMSE; analytic or free x-step) and combiner-agnostic networks.

* the bsce port equals the graph-choice study's PyTorch layer (tests/proposals/torch_graph_ladmm_ref.py) for every
  z-step, ULA and UPA;
* zstep gtv with tied parameters is ADMM for the fused LASSO on the magnetic graph (an independent numpy ADMM);
* free H_t at init gives the analytic step; the edge phases are estimated as the array convention says;
* set_combiner: an analytic network on a new combiner equals a network built on that combiner with the same weights;
  networks learned for one combiner refuse;
* train.combiner=random trains, and leaves the network (and its checkpoint) on the scenario's combiner;
* evaluate.py --combiner / another data set: the network follows or keeps its combiner, as eval_log.json says.
"""
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

from bsce.data import measurements, sv
from bsce.export import load_model, save_model, truncate_model
from bsce.models import build_model
from bsce.models.graph_ladmm import ZSTEPS, estimate_edge_phases
from bsce.rng import NumpyMatlabRNG
from bsce.train.trainer import Trainer, TrainConfig

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "tests" / "proposals"))


def _A(seed=0, M=128, N=256):
    g = np.random.default_rng(seed)
    return ((g.random((M, N)) > 0.5) * 2 - 1) / np.sqrt(M)


@pytest.fixture(scope="module")
def ula():
    A = _A(0)
    x = sv.beamspace_channels(NumpyMatlabRNG(0), 24, "ULA")
    y = measurements.gen_measurements_matlab(x, A, 10.0, NumpyMatlabRNG(1))
    return A, x, y


@pytest.mark.parametrize("array", ["ULA", "UPA"])
@pytest.mark.parametrize("z", ["l1", "gsoft", "gate", "gtv", "gtv_gate"])
def test_equals_reference_layer(array, z, record):
    from torch_graph_ladmm_ref import GraphLADMM as Ref
    A = _A(1)
    x = sv.beamspace_channels(NumpyMatlabRNG(2), 16, array)
    y = measurements.gen_measurements_matlab(x, A, 10.0, NumpyMatlabRNG(3))
    s2 = np.full(16, 0.1)
    ref = Ref(A, array, T=3, zstep=z)
    m = build_model(dict(model="Graph-LADMM", T=3, zstep=z, array=array, sigma="known", edge_phase="theory",
                         precision="float64"), A)
    g = np.random.default_rng(4)
    with torch.no_grad():
        for k in ref.raw:
            v = g.normal(0, 0.7, 3)
            ref.raw[k].copy_(torch.tensor(v))
            for t in range(3):
                if f"{k}_raw_{t}" in m.param_names():  # v31: e / k2 of the last layer are not stored (no effect)
                    m.P(f"{k}_raw_{t}").fill_(v[t] / m.lr_mult)
        xr = ref(torch.tensor(y.T), torch.tensor(np.sqrt(s2))).numpy().T
        xm = m(torch.tensor(y), s2=s2).numpy()
    rel = np.linalg.norm(xm - xr) / np.linalg.norm(xr)
    record("v29/graph_vs_reference", array=array, zstep=z, rel=float(rel))
    assert rel < 1e-12


def test_gtv_is_fused_lasso_admm(ula, record):
    """Tied rho, lambda/rho = k1 s, mu/rho = k2 s, e = 1: the network = ADMM for 1/2||y_w - A_w x||^2 + lam||x||_1 +
    mu||Dx||_1 with the split [I; D] x = [z1; z2] (scaled dual), computed here in numpy."""
    A, x, y = ula
    T, rho, lam, mu, s2 = 12, 0.7, 0.05, 0.03, 0.1
    m = build_model(dict(model="Graph-LADMM", T=T, zstep="gtv", array="ULA", sigma="known", edge_phase=180.0,
                         precision="float64"), A)
    sig = np.sqrt(s2)
    isp = lambda v: np.log(np.expm1(v)) / m.lr_mult  # noqa: E731
    with torch.no_grad():
        for t in range(T):
            m.P(f"rho_raw_{t}").fill_(isp(rho))
            m.P(f"k1_raw_{t}").fill_(isp(lam / rho / sig))
            if t < m.T - 1:                     # v31: the last layer has no k2 / e (they never reached the output)
                m.P(f"k2_raw_{t}").fill_(isp(mu / rho / sig))
                m.P(f"e_raw_{t}").fill_(isp(1.0))
        z = m(torch.tensor(y), s2=np.full(y.shape[1], s2)).numpy()
    ev, U = np.linalg.eigh(A @ A.T)
    W = (U / np.sqrt(ev)) @ U.T
    Aw, yw = W @ A, W @ y
    N = A.shape[1]
    D = np.zeros((N, N))
    for i in range(N):
        D[i, (i + 1) % N] = 1.0
        D[i, i] = 1.0                               # x_j - e^{j pi} x_i = x_j + x_i
    soft = lambda v, t: v * np.maximum(1 - t / np.maximum(np.abs(v), 1e-30), 0)  # noqa: E731
    K = np.linalg.inv(Aw.T @ Aw + rho * (np.eye(N) + D.T @ D))
    z1 = np.zeros_like(x); u1 = np.zeros_like(x); z2 = np.zeros_like(x); u2 = np.zeros_like(x)
    for _ in range(T):
        xx = K @ (Aw.T @ yw + rho * (z1 - u1) + rho * D.T @ (z2 - u2))
        z1 = soft(xx + u1, lam / rho)
        Dx = D @ xx
        z2 = soft(Dx + u2, mu / rho)
        u1 = u1 + xx - z1
        u2 = u2 + Dx - z2
    rel = np.linalg.norm(z - z1) / np.linalg.norm(z1)
    record("v29/gtv_vs_fused_lasso_admm", rel=float(rel))
    assert rel < 1e-10


@pytest.mark.parametrize("z", ["l1", "gtv_gm"])
def test_free_H_at_init_is_the_analytic_step(z, ula):
    A, _, y = ula
    a = build_model(dict(model="Graph-LADMM", T=3, zstep=z, xstep="analytic", array="ULA", edge_phase=180.0,
                         precision="float64"), A)
    f = build_model(dict(model="Graph-LADMM", T=3, zstep=z, xstep="free", array="ULA", edge_phase=180.0,
                         precision="float64"), A)
    with torch.no_grad():
        xa, xf = a.forward_stages(torch.tensor(y), upto=0)[0], f.forward_stages(torch.tensor(y), upto=0)[0]
    assert torch.allclose(xa, xf, atol=1e-10)
    assert f.n_params() == a.n_params() + 3 * 256 * 256 - 1      # v31: with a free H, rho of layer 1 is not stored


def test_edge_phase_follows_the_array_convention():
    """GM-LAMP's SV generator uses a centred array index: neighbouring beams alternate sign (about 180 degrees)."""
    X = sv.beamspace_channels(NumpyMatlabRNG(5), 3000, "ULA")
    est = estimate_edge_phases(X, "ULA", 16, 16)
    ph, coh, n = est[(0, False)]
    assert abs(abs(np.degrees(ph)) - 180) < 3 and coh > 0.8 and n > 1000
    m = build_model(dict(model="Graph-LADMM", T=2, array="ULA"), _A(0))
    info = m.init_from_data(X)
    assert abs(abs(info["axis0_interior"][0]) - 180) < 3


def test_set_combiner(ula):
    A, x, _ = ula
    A2 = _A(9)
    y2 = A2 @ (x + 0.1 * np.random.default_rng(7).standard_normal(x.shape))
    for cfg in [dict(model="Graph-LADMM", T=3, zstep="gtv_gm", array="ULA", edge_phase=180.0),
                dict(model="LADMM", T=3, operator="analytic", data_step="whitened"),
                dict(model="LADMM", T=3, operator="analytic1", threshold="gm")]:
        cfg = dict(cfg, precision="float64")
        m = build_model(cfg, A)
        g = np.random.default_rng(6)
        with torch.no_grad():
            for p in m.parameters():
                p.add_(torch.as_tensor(0.05 * g.standard_normal(p.shape)))
        ref = build_model(cfg, A2)
        ref.load_state_dict({k: v for k, v in m.state_dict().items()
                             if k not in ("A", "Wm", "Aw", "Apinv", "Pm", "AtA", "Vs", "gs", "Wfix")}, strict=False)
        m.set_combiner(A2)
        with torch.no_grad():
            assert torch.allclose(m(torch.tensor(y2)), ref(torch.tensor(y2)), atol=1e-10), cfg
    for cfg in [dict(model="Graph-LADMM", T=2, xstep="free"), dict(model="LADMM", T=2), dict(model="CADMM", T=2)]:
        m = build_model(cfg, A)
        with pytest.raises((ValueError, AttributeError)):
            m.set_combiner(A2)


def test_every_zstep_trains_saves_loads_truncates(tmp_path, record):
    A = _A(0)
    xt = sv.beamspace_channels(NumpyMatlabRNG(0), 200, "ULA")
    xv = sv.beamspace_channels(NumpyMatlabRNG(1), 60, "ULA")
    yv = measurements.gen_measurements_py(xv, A, [0.0, 10.0, 20.0], np.random.RandomState(3))
    for z in ZSTEPS:
        for xs in ("analytic", "free"):
            torch.manual_seed(0)
            m = build_model(dict(model="Graph-LADMM", T=2, zstep=z, xstep=xs, array="ULA", precision="float64"), A)
            m.init_from_data(xt)
            cfg = TrainConfig(schedule="steps", steps=20, val_every=10, lr=1e-3, batch_size=32, noise="online",
                              snr_range=[0, 20], val_round=None, restore_best=True, verbose=False)
            st = Trainer(m, cfg, xt, None, xv, yv).run()
            v = [r["nmse_db"] for r in st["log"]]
            assert all(np.isfinite(v))
            p, _ = save_model(m, tmp_path / f"{z}_{xs}")
            m2 = load_model(p)
            yy = torch.as_tensor(yv[:, :5])
            with torch.no_grad():
                assert torch.allclose(m(yy), m2(yy), atol=1e-12)
                assert torch.allclose(truncate_model(m, 1)(yy), m.forward_stages(yy)[1], atol=1e-12)
            record("v29/train", zstep=z, xstep=xs, nmse_db=v[-1])


def test_random_combiner_training(tmp_path):
    A = _A(0)
    xt = sv.beamspace_channels(NumpyMatlabRNG(0), 200, "ULA")
    xv = sv.beamspace_channels(NumpyMatlabRNG(1), 40, "ULA")
    yv = measurements.gen_measurements_py(xv, A, [0.0, 20.0], np.random.RandomState(3))
    m = build_model(dict(model="Graph-LADMM", T=2, zstep="gm", array="ULA", precision="float64"), A)
    cfg = TrainConfig(schedule="steps", steps=20, val_every=10, lr=1e-3, batch_size=16, noise="online",
                      snr_range=[0, 20], val_round=None, restore_best=True, verbose=False, combiner="random",
                      checkpoint_iters=5)
    tr = Trainer(m, cfg, xt, None, xv, yv, checkpoint=str(tmp_path / "ck"))
    tr.run()
    assert np.array_equal(m.A_np, A) and torch.allclose(m.A, torch.as_tensor(A))
    ck = torch.load(str(tmp_path / "ck") + ".pt", weights_only=False)
    assert torch.allclose(ck["model_state"]["A"], torch.as_tensor(A))
    assert tr._nb == 20                                      # one new combiner per minibatch
    with pytest.raises(ValueError):                          # measurement files are for one combiner
        Trainer(m, TrainConfig(noise="file", combiner="random"), xt, np.zeros((128, 200), complex), xv, yv)
    with pytest.raises(ValueError):                          # a free H_t cannot follow a combiner
        Trainer(build_model(dict(model="LADMM", T=2), A), cfg, xt, None, xv, yv)


@pytest.mark.skipif(not (ROOT / "data/SV_ULA/SV_ULA_validation_channel.mat").exists() or
                    not (ROOT / "data/DeepMIMO_ULA/DeepMIMO_ULA_test_channel.mat").exists(), reason="needs SV/DeepMIMO data")
def test_evaluate_other_combiner_and_data_set(tmp_path):
    import scipy.io as sio
    A = sio.loadmat(ROOT / "data/SV_ULA/CSmatrix.mat")["A"]
    for tag, cfg in [("G-an", dict(model="Graph-LADMM", T=2, array="ULA")),
                     ("G-free", dict(model="Graph-LADMM", T=2, array="ULA", xstep="free")),
                     ("L-F", dict(model="LADMM", T=2))]:
        save_model(build_model(dict(cfg, precision="float32"), A), tmp_path / "runs" / tag / f"{tag}_0to20dB")
    run = lambda *x: subprocess.run([sys.executable, "scripts/evaluate.py", "--runs", str(tmp_path / "runs"),  # noqa: E731
                                     "--classical", "", "--n-test", "50", "--device", "cpu", *x],
                                    cwd=ROOT, capture_output=True, text=True)
    r = run("--scenario", "configs/scenarios/sv_ula.yaml", "--combiner", "random:3", "--out", str(tmp_path / "o1"))
    assert r.returncode == 0, r.stderr[-1500:]
    note = json.loads((tmp_path / "o1" / "eval_log.json").read_text())["network_combiner"]
    assert note["G-an"] == "follows" and note["G-free"].startswith("fixed") and note["L-F"].startswith("fixed")
    r = run("--scenario", "configs/scenarios/deepmimo_ula.yaml", "--graph-phase", "target", "--out", str(tmp_path / "o2"))
    assert r.returncode == 0, r.stderr[-1500:]
    note = json.loads((tmp_path / "o2" / "eval_log.json").read_text())["network_combiner"]
    assert note["G-an"] == "follows, edge phases of this data set" and note["L-F"].startswith("fixed")
    assert note["G-free"] == "fixed (learned for its own combiner)"     # free H_t keeps the phases it was learned with


def test_suite_keys():
    r = subprocess.run([sys.executable, "scripts/run_suite.py", "--scenarios", "sv_ula", "--methods", "gladmm",
                        "gladmmf", "gladmmany", "--protocol", "paper", "--dry-run"], cwd=ROOT, capture_output=True, text=True)
    assert r.returncode == 0
    for t in ("G-LADMM-T8_paper", "G-LADMM-F-T6_paper", "G-LADMM-anyA-T8_paper", "configs/train/gladmm_anyA.yaml"):
        assert t in r.stdout
    r = subprocess.run([sys.executable, "scripts/run_suite.py", "--scenarios", "sv_ula", "--methods", "all", "--dry-run"],
                       cwd=ROOT, capture_output=True, text=True)
    assert "G-LADMM" not in r.stdout


@pytest.mark.parametrize("pair", [("gm", "gm_graph"), ("gtv_gm", "gtv_gm_graph")])
def test_graph_coupled_gm_starts_as_gm(pair, ula):
    """beta_k = 0 at init: the graph-coupled GM is the plain GM (the graph enters only through what is learned)."""
    A, _, y = ula
    a = build_model(dict(model="Graph-LADMM", T=3, zstep=pair[0], array="ULA", precision="float64"), A)
    b = build_model(dict(model="Graph-LADMM", T=3, zstep=pair[1], array="ULA", precision="float64"), A)
    sd = b.state_dict()
    sd.update({k: v for k, v in a.state_dict().items() if k in sd})
    b.load_state_dict(sd)
    with torch.no_grad():
        assert torch.allclose(a(torch.tensor(y)), b(torch.tensor(y)), atol=1e-12)
        beta = b.P("gm_beta_1")                     # the same beta for every component would not change the softmax
        beta.copy_(torch.linspace(-0.7, 0.7, beta.numel(), dtype=beta.dtype))
        assert not torch.allclose(a(torch.tensor(y)), b(torch.tensor(y)), atol=1e-6)

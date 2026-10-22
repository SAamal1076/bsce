"""v31: stored = trained parameters, the inference cache, the beam prior, set_combiner for NF-LADMM / LVAMP / ALISTA,
the combiner mixture in training, the classical graph-reweighted l1, robust evaluation plots, the cost module, names.

* Graph-LADMM creates no parameter that cannot reach the output: for every z-step, x-step, array, data step, pooling and
  the beam prior, every stored parameter gets a gradient; files written before v31 (with those entries) load and give
  the same output;
* I + D^H D built from the edges equals the dense product; the cached inference H_t equals the per-call solve and follows
  a new combiner, new edge phases and new rho;
* the beam prior: beta = 0 is the plain network, the profile comes from the training channels, gradients reach it;
* NF-LADMM, LVAMP and ALISTA follow a new combiner of any M (= a fresh network on it), and a saved moved network reloads;
* ALISTA's W solver: the default step is 0.1 on the scenario combiners and stays finite on a large ||A||;
* train.combiner=mix: the families, sizes and checks, and a short training;
* classical l1 / grwl1: the l1 ADMM converges to bsce's ADMM, grwl1 with a huge eps is l1, the weights equal the
  Graph-LADMM reweighting; tune_classical.py + evaluate.py --classical l1,grwl1;
* evaluate.py draws diverged (NaN) networks; check_combiners.py goes on when no method can run;
* bsce/cost.py: LISTA's untrained W_1, LAMP's MACs; the new suite keys and their folders.
"""
import csv
import json
import math
import re
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
import scipy.io as sio
import torch

from bsce import classical
from bsce.cost import macs_per_channel, trained_parameters
from bsce.data import measurements, sv
from bsce.export import load_model, save_model
from bsce.models import build_model
from bsce.models.graph_ladmm import ZSTEPS
from bsce.models.lista import alista_W
from bsce.rng import NumpyMatlabRNG

ROOT = Path(__file__).resolve().parent.parent
SV_A = ROOT / "data" / "SV_ULA" / "CSmatrix.mat"


def _A(seed=0, M=128, N=256):
    g = np.random.default_rng(seed)
    return ((g.random((M, N)) > 0.5) * 2 - 1) / np.sqrt(M)


def _y(A, array="ULA", K=12, snr=10.0, seed=1):
    x = sv.beamspace_channels(NumpyMatlabRNG(seed), K, array)
    return x, measurements.gen_measurements_matlab(x, A, snr, NumpyMatlabRNG(seed + 1))


GL_CASES = ([dict(zstep=z, array=a) for z in ZSTEPS for a in ("ULA", "UPA")]
            + [dict(zstep="gate", xstep="free"), dict(zstep="gtv_gate", xstep="free"), dict(zstep="rwl1", data_step="plain"),
               dict(zstep="gtv_rwl1", data_step="plain"), dict(zstep="gate", graph_pool=False),
               dict(zstep="gm_graph", graph_pool=False), dict(zstep="rwl1", graph_pool=False),
               dict(zstep="gtv_gate", beam_prior="gate"), dict(zstep="gate", beam_prior="gate", array="UPA")])


@pytest.mark.parametrize("case", GL_CASES, ids=lambda c: "-".join(f"{v}" for v in c.values()))
def test_graph_ladmm_stores_only_trained(case):
    cfg = dict(dict(model="Graph-LADMM", T=3, array="ULA", precision="float64"), **case)
    m = build_model(cfg, _A(0))
    if case.get("beam_prior"):
        m.init_from_data(sv.beamspace_channels(NumpyMatlabRNG(5), 400, cfg["array"]))
    g = torch.Generator().manual_seed(1)
    with torch.no_grad():                    # off the init saddles (gm_beta = 0, beam_beta = 0 hide a1, a2, beam_ell)
        for q in m.parameters():
            q.add_(0.05 * torch.randn(q.shape, generator=g, dtype=q.dtype))
        for t in range(m.T):                 # a low reweighted threshold: no layer output all zero by chance
            if f"k1_raw_{t}" in m.param_names() and m.zstep in ("rwl1", "gtv_rwl1"):
                m.P(f"k1_raw_{t}").fill_(math.log(math.expm1(0.3)) / m.lr_mult)
    tr, st, dead = trained_parameters(m, K=96)
    assert dead == [] and tr == st, dead
    assert not m._dead_names() & set(m.param_names())


def test_pre_v31_files_load_with_the_same_output(tmp_path):
    """A v30 file stores e and k2 of the last layer: v31 ignores them, and they never changed the output."""
    A = _A(1)
    m = build_model(dict(model="Graph-LADMM", T=3, array="ULA", precision="float64"), A)
    g = torch.Generator().manual_seed(2)
    with torch.no_grad():
        for p in m.parameters():
            p.add_(0.1 * torch.randn(p.shape, generator=g, dtype=p.dtype))
    save_model(m, tmp_path / "net")
    ck = torch.load(tmp_path / "net.pt", weights_only=False)
    ck["state_dict"]["e_raw_2"] = torch.tensor(0.0541, dtype=torch.float64)          # what v30 wrote
    ck["state_dict"]["k2_raw_2"] = torch.tensor(-0.0433, dtype=torch.float64)
    torch.save(ck, tmp_path / "old.pt")
    old = load_model(tmp_path / "old.pt")
    _, y = _y(A)
    with torch.no_grad():
        assert torch.equal(old(torch.tensor(y)), m(torch.tensor(y)))
    m.load_state_dict(ck["state_dict"])                              # also through load_state_dict itself
    assert len(m.forward_stages(torch.tensor(y))) == 2 * m.T


@pytest.mark.parametrize("array,ring", [("ULA", True), ("ULA", False), ("UPA", True), ("UPA", False)])
def test_sparse_G_equals_dense(array, ring):
    m = build_model(dict(model="Graph-LADMM", T=2, array=array, ring=ring, precision="float64"), _A(0))
    with torch.no_grad():
        m.phase.copy_(torch.tensor(np.linspace(-2.5, 1.7, len(m.classes))))
        n_e = m.I_e.shape[0]
        D = torch.zeros(n_e, m.N, dtype=m.cdtype)
        D[torch.arange(n_e), m.J_e] = 1
        D[torch.arange(n_e), m.I_e] = -m._wedge()
        dense = torch.eye(m.N, dtype=m.cdtype) + D.conj().T @ D
    assert float((m._G() - dense).abs().max()) < 1e-12


def test_inference_cache_equals_solve_and_follows_changes():
    A0, A1 = _A(0), _A(7, M=96)
    m = build_model(dict(model="Graph-LADMM", T=4, array="ULA", precision="float64"), A0)
    g = torch.Generator().manual_seed(3)
    with torch.no_grad():
        for p in m.parameters():
            p.add_(0.1 * torch.randn(p.shape, generator=g, dtype=p.dtype))

    def both(A):
        _, y = _y(A)
        with torch.no_grad():
            m.cache_inference = True
            a = m(torch.tensor(y))
            m.cache_inference = False
            b = m(torch.tensor(y))
            m.cache_inference = True
        return float((a - b).abs().max() / b.abs().max())

    assert both(A0) < 1e-10 and m._hcache is not None
    key0 = m._hcache[0]
    with torch.no_grad():
        m.P("rho_raw_1").add_(0.05)                                   # new rho: a new key
    assert both(A0) < 1e-10 and m._hcache[0] != key0
    with torch.no_grad():
        m.phase.add_(0.3)                                             # new edge phases
    assert both(A0) < 1e-10
    m.set_combiner(A1)                                                # another combiner, another M
    assert m._hcache is None and both(A1) < 1e-10
    x = torch.zeros(1, requires_grad=True)
    with torch.enable_grad():                                         # training: no cache, gradients flow to rho
        _, y = _y(A1)
        out = m(torch.tensor(y))
        out.abs().sum().backward()
    assert m.P("rho_raw_0").grad is not None and float(m.P("rho_raw_0").grad.abs()) > 0
    del x


def test_beam_prior():
    A = _A(0)
    xt = sv.beamspace_channels(NumpyMatlabRNG(4), 500, "ULA")
    p = build_model(dict(model="Graph-LADMM", T=3, array="ULA", precision="float64"), A)
    b = build_model(dict(model="Graph-LADMM", T=3, array="ULA", beam_prior="gate", precision="float64"), A)
    for n, v in p.named_parameters():
        b.get_parameter(n).data.copy_(v.data)
    b.init_from_data(xt)
    p.init_from_data(xt)                                              # the same edge phases
    pw = np.mean(np.abs(xt) ** 2, 1)
    assert np.allclose(b.P("beam_ell").detach().numpy(), np.log(np.maximum(pw / pw.mean(), 1e-3)))
    _, y = _y(A)
    with torch.no_grad():
        assert torch.equal(p(torch.tensor(y)), b(torch.tensor(y)))   # beta = 0: the plain network
    assert b.cfg["beam_prior"] == "gate" and "beam_prior" not in p.cfg
    assert sum(q.numel() for q in b.parameters()) == sum(q.numel() for q in p.parameters()) + 256 + 3
    with pytest.raises(ValueError):
        build_model(dict(model="Graph-LADMM", T=2, zstep="l1", beam_prior="gate"), A)
    b.set_combiner(_A(9, M=64))                                       # the prior is in the beam domain


NF_CFGS = [dict(model="NF-LADMM", T=2, denoiser="radial", precision="float64"),
           dict(model="NF-LADMM", T=2, denoiser="conv", precision="float64", seed=0),
           dict(model="NF-LADMM", T=2, denoiser="radial", noise_model="white", precision="float64"),
           dict(model="LVAMP", T=3, precision="float64"),
           dict(model="ALISTA", T=3, precision="float64")]


@pytest.mark.parametrize("M", [64, 96, 160])
@pytest.mark.parametrize("cfg", NF_CFGS, ids=lambda c: f"{c['model']}-{c.get('denoiser', '')}{c.get('noise_model', '')}")
def test_set_combiner_nf_lvamp_alista(cfg, M, tmp_path):
    A0, A1 = _A(0), _A(7, M=M)
    a, b = build_model(dict(cfg), A0), build_model(dict(cfg), A1)
    g = torch.Generator().manual_seed(3)
    with torch.no_grad():
        for p in a.parameters():
            p.add_(0.05 * torch.randn(p.shape, generator=g, dtype=p.dtype))
        for k, v in a.named_parameters():
            b.get_parameter(k).copy_(v)
    a.set_combiner(A1)
    assert a.M == M and tuple(a.A.shape) == (M, 256)
    x, y = _y(A1, K=8)
    s2 = np.full(8, 0.1)
    run = lambda n: n(torch.tensor(y), s2=s2) if getattr(n, "needs_noise_var", False) else n(torch.tensor(y))  # noqa: E731
    with torch.no_grad():
        ya, yb = run(a), run(b)
    assert float((ya - yb).abs().max() / yb.abs().max()) < 1e-10
    save_model(a, tmp_path / "moved")
    r = load_model(tmp_path / "moved.pt")
    with torch.no_grad():
        assert float((run(r) - ya).abs().max() / ya.abs().max()) < 1e-10
    with pytest.raises(ValueError):
        a.set_combiner(np.vstack([A1[:10], A1[:1]]))                  # dependent rows
    with pytest.raises(ValueError):
        a.set_combiner(_A(1, M=64, N=128))                            # another N


def test_alista_W_step():
    for f in sorted((ROOT / "data").glob("*/CSmatrix.mat"))[:3]:
        A = sio.loadmat(f)["A"].astype(np.float64)
        assert np.linalg.norm(A, 2) ** 2 < 6.0                       # so the default step is the original 0.1
    A = _A(0, M=32, N=64)
    assert np.array_equal(alista_W(A), alista_W(A, eta=0.1))         # norm^2 < 6: the original step
    sys.path.insert(0, str(ROOT / "scripts"))
    import check_combiners as cc
    R = cc.make_combiners(sio.loadmat(SV_A)["A"].astype(np.float64))["corr"]          # ||R||_2^2 = 44.7
    with np.errstate(all="ignore"):
        assert not np.isfinite(alista_W(R, eta=0.1)).all()           # the v30 step diverged here
    assert np.linalg.norm(R, 2) ** 2 > 40 and np.isfinite(alista_W(R)).all()


def test_combiner_mix_families_and_checks():
    from bsce.train.trainer import COMBINER_TYPES, make_train_combiner
    g = torch.Generator().manual_seed(0)
    for kind in COMBINER_TYPES:
        for M in (64, 160):
            C = make_train_combiner(kind, M, 256, g)
            assert C.shape == (M, 256) and np.linalg.matrix_rank(C) == M
    S = make_train_combiner("sparse", 32, 256, g)
    assert (np.count_nonzero(S, 1) == 16).all()
    D = make_train_combiner("dct", 32, 256, g)
    assert np.allclose(D @ D.T, np.eye(32))
    with pytest.raises(ValueError):
        make_train_combiner("nope", 8, 16, g)


def _train(tmp_path, *sets):
    return subprocess.run([sys.executable, "scripts/train.py", "--scenario", "sv_upa", "--smoke", "--train",
                           "configs/train/gladmm.yaml", "--protocol", "paper", "--snr", "0", "20", "--device", "cpu",
                           "--out", str(tmp_path), "--tag", "mix", "--set", "train.steps=6", "train.val_every=3",
                           "train.batch_size=16", "model.T=2", *sets], cwd=ROOT, capture_output=True, text=True)


def test_combiner_mix_training(tmp_path):
    if not (ROOT / "data" / "SV_UPA_smoke").exists():
        pytest.skip("no SV_UPA smoke data set")
    r = _train(tmp_path, "train.combiner=mix", "train.combiner_M=[64,96,160]", "train.combiner_types=[pm1,dct,sparse]")
    assert r.returncode == 0, (r.stdout + r.stderr)[-3000:]
    f = next(tmp_path.rglob("mix_0to20dB.pt"))
    net = load_model(f)
    A0 = sio.loadmat(ROOT / "data" / "SV_UPA_smoke" / "CSmatrix.mat")["A"]
    assert np.array_equal(net.A_np, A0)                               # saved on the scenario's combiner
    r = _train(tmp_path / "b", "train.combiner=mix", "train.combiner_M=[300]")
    assert r.returncode != 0 and "combiner_M" in (r.stdout + r.stderr)
    r = _train(tmp_path / "c", "train.combiner=random", "train.combiner_types=[dct]")
    assert r.returncode != 0 and "combiner=mix" in (r.stdout + r.stderr)


def test_classical_l1_and_grwl1():
    A = _A(3, M=48, N=96)
    x = np.zeros((96, 6), complex)
    g = np.random.default_rng(1)
    for k in range(6):
        x[g.choice(96, 3, replace=False), k] = g.standard_normal(3) + 1j * g.standard_normal(3)
    y = A @ (x + 0.05 * (g.standard_normal(x.shape) + 1j * g.standard_normal(x.shape)))
    z1 = classical.l1_admm(y, A, 0.05, T=3000)
    z2, _ = classical.admm(y, A, 0.05, 1.0, 20000, 1e-12)
    assert np.abs(z1 - z2).max() < 1e-6 * np.abs(z2).max()
    zr = classical.grwl1(y, A, 0.05, 1e12, T=200, array="ULA")
    assert np.abs(zr - classical.l1_admm(y, A, 0.05, T=200)).max() < 1e-9
    for array in ("ULA", "UPA"):                                      # the same weights as Graph-LADMM's rwl1
        m = build_model(dict(model="Graph-LADMM", T=3, array=array, zstep="rwl1", precision="float64"), _A(0))
        Z = sv.beamspace_channels(NumpyMatlabRNG(2), 5, array)
        with torch.no_grad():
            om = m.rw_weights(torch.tensor(Z), 1).numpy()
        assert np.abs(om - classical.graph_weights(Z, float(m.s("eps", 1)), array)).max() < 1e-10
    assert {"l1", "grwl1", "amp"} <= set(classical.METHODS)


def test_tune_and_evaluate_classical(tmp_path):
    if not (ROOT / "data" / "SV_UPA_smoke").exists():
        pytest.skip("no SV_UPA smoke data set")
    js = tmp_path / "par.json"
    r = subprocess.run([sys.executable, "scripts/tune_classical.py", "--scenario", "configs/scenarios/sv_upa.yaml",
                        "--smoke", "--snr", "0", "10", "--n-val", "12", "--out", str(js)], cwd=ROOT, capture_output=True,
                       text=True)
    assert r.returncode == 0, r.stderr[-2000:]
    par = json.loads(js.read_text())
    assert set(par) == {"0", "10"} and {"lam_l1", "lam_rw", "eps"} <= set(par["0"])
    runs = tmp_path / "SV_UPA_smoke"
    runs.mkdir()
    ev = lambda *x: subprocess.run([sys.executable, "scripts/evaluate.py", "--scenario",  # noqa: E731
                                    "configs/scenarios/sv_upa.yaml", "--runs", str(runs), "--n-test", "10", "--device",
                                    "cpu", "--out", str(tmp_path / "ev"), *x], cwd=ROOT, capture_output=True, text=True)
    r = ev("--classical", "l1")
    assert r.returncode != 0 and "--classical-params" in (r.stdout + r.stderr)
    r = ev("--classical", "amp,l1,grwl1", "--classical-params", str(js))
    assert r.returncode != 0 and "no values for the test SNRs" in (r.stdout + r.stderr)   # only 0 and 10 dB tuned


def test_plot_survives_diverged_networks(tmp_path):
    sys.path.insert(0, str(ROOT / "scripts"))
    import evaluate as ev
    rows = [{"SNR_dB": s, "good": -s / 2.0, "bad": (float("nan") if s > 4 else 3.0)} for s in (0, 2, 4, 6, 8)]
    ev.plot(rows, ["good", "bad"], tmp_path / "nmse_vs_snr.png", "test")
    assert (tmp_path / "nmse_vs_snr.png").exists() and (tmp_path / "nmse_table.png").exists()


def test_check_combiners_goes_on_without_methods(tmp_path):
    runs = tmp_path / "runs"
    A = sio.loadmat(SV_A)["A"].astype(np.float64)
    save_model(build_model(dict(model="LADMM", T=2, precision="float32"), A), runs / "L-F" / "L-F_0to20dB")
    out = tmp_path / "cc"
    r = subprocess.run([sys.executable, "scripts/check_combiners.py", "--scenario", "configs/scenarios/sv_ula.yaml",
                        "--runs", str(runs), "--combiners", "own", "M--", "--classical", "", "--n-test", "20",
                        "--device", "cpu", "--out", str(out)], cwd=ROOT, capture_output=True, text=True)
    assert r.returncode == 0, (r.stdout + r.stderr)[-2000:]
    cell = {(x["network"], x["combiner"]): x for x in csv.DictReader(open(out / "combiner_check.csv"))}
    assert cell[("L-F", "M64")]["note"].startswith("cannot run") and cell[("L-F", "own")]["mean_nmse_db"] != ""


def test_cost_module():
    A = sio.loadmat(SV_A)["A"].astype(np.float64)
    li = build_model(dict(model="LISTA", T=3, precision="float32"), A)
    tr, st, dead = trained_parameters(li)
    assert st - tr == 256 * 256 and dead == ["W_1 (65536)"]          # W_1 multiplies x_0 = 0
    lp = build_model(dict(model="LAMP", T=4, shrink="soft", precision="float32"), A)
    macs, _ = macs_per_channel(lp)
    assert macs == 2 * 256 * 128 + 3 * 4 * 256 * 128                 # B_0 y, then A x and B_t v per layer
    gl = build_model(dict(model="Graph-LADMM", T=8, array="ULA", precision="float32"), A)
    macs, _ = macs_per_channel(gl)                                   # cached H_t: 8 x (4 N^2 + 2 M N) + A^+ y, W y
    assert macs == 8 * (4 * 256 ** 2 + 2 * 128 * 256) + 2 * 256 * 128 + 2 * 128 * 128


def _suite(*x):
    r = subprocess.run([sys.executable, "scripts/run_suite.py", "--scenarios", "sv_upa", "--dry-run", *x], cwd=ROOT,
                       capture_output=True, text=True)
    assert r.returncode == 0, r.stderr[-2000:]
    return [ln for ln in r.stdout.splitlines() if ln.startswith(">>> scripts/train.py")]


def test_new_suite_keys():
    cmds = _suite("--methods", "gladmmrw", "gladmmrwtv", "gladmmbp", "gladmmmix")
    tag = {re.search(r"--tag (\S+)", c).group(1): c for c in cmds}
    assert set(tag) == {"G-LADMM-RW-T8_paper", "G-LADMM-RW-TV-T8_paper", "G-LADMM-BP-T8_paper", "G-LADMM-mixA-T8_paper"}
    for c in cmds:
        assert "--protocol" not in c                                  # their own end-to-end settings
    if (ROOT / "app" / "bsce_app.py").exists():                      # the Spyder edition has no workbench
        sys.path.insert(0, str(ROOT / "app"))
        import bsce_app
        for k in ("gladmmrw", "gladmmrwtv", "gladmmbp", "gladmmmix"):
            assert k in bsce_app.METHOD_INFO
    import yaml
    for f, n in (("gladmm_rwl1", 44), ("gladmm_rwl1_tv", 51), ("gladmm_bp", 318), ("gladmm_mixA", 54), ("gladmm", 54)):
        c = yaml.safe_load(open(ROOT / "configs" / "train" / f"{f}.yaml"))["model"]
        c["array"] = "UPA"
        assert sum(p.numel() for p in build_model(c, _A(0)).parameters()) == n, f


# -- v31 fix of 18 Oct (GM-LAMP trained to the linear level): the identical mixture components -----------------------
def _gm_data():
    g = np.random.default_rng(0)
    A = ((g.random((128, 256)) > 0.5) * 2 - 1) / np.sqrt(128)
    xt = sv.beamspace_channels(NumpyMatlabRNG(0), 400, "ULA")
    xv = sv.beamspace_channels(NumpyMatlabRNG(1), 100, "ULA")
    yt = measurements.gen_measurements_py(xt, A, [10.0], np.random.RandomState(2))
    yv = measurements.gen_measurements_py(xv, A, [10.0], np.random.RandomState(3))
    return A, xt, yt, xv, yv


@pytest.mark.parametrize("precision", ["float32", "float64"])
def test_gmlamp_identical_components_and_their_break(precision):
    """The cause: with GM-LAMP's original init the K components get bit-identical gradients, so Adam never separates
    them. break_gm_symmetry gives exactly the 'spread' init, once, and only to GM parameters."""
    from bsce.models.lamp import gm_theta_init
    A = _gm_data()[0]
    m = build_model(dict(model="GM-LAMP", T=2, K=4, precision=precision), A)
    x, y = _y(A, K=64)
    out = m(torch.tensor(y))
    (torch.abs(out - torch.as_tensor(x, dtype=out.dtype)) ** 2).sum().backward()
    g = m.P("theta_1").grad
    assert bool(torch.all(g == g[:, :1]))                            # identical across the components
    assert m.break_gm_symmetry(["B_0", "theta_0", "theta_1"]) == ["theta_0", "theta_1"]
    sp = torch.as_tensor(gm_theta_init(4, "spread"), dtype=m.rdtype)
    for n in ("theta_0", "theta_1"):                                 # = gm_init='spread' (to float32 rounding)
        assert torch.allclose(m.P(n).detach(), sp, rtol=0, atol=1e-7 if precision == "float32" else 0)
    assert m.break_gm_symmetry(["theta_0"]) == []                    # already separated: unchanged
    assert build_model(dict(model="LAMP", T=2, precision=precision), A).break_gm_symmetry(["theta_0"]) == []


def test_gmlamp_training_separates_the_components():
    from bsce.train.trainer import Trainer, TrainConfig
    A, xt, yt, xv, yv = _gm_data()
    res = {}
    for mode in ("break", "keep"):
        torch.manual_seed(0)
        m = build_model(dict(model="GM-LAMP", T=1, K=4, precision="float64"), A)
        cfg = TrainConfig(lr=1e-3, maxit=150, better_wait=1000, ivl=10, check_mult=5, batch_size=64, refinements=[],
                          warm_start=True, gm_symmetry=mode, verbose=False)
        st = Trainer(m, cfg, xt, yt, xv, yv).run()
        th = m.P("theta_0").detach()
        res[mode] = (bool(torch.all(th == th[:, :1])), st.get("gm_symmetry_broken", []))
    assert res["break"] == (False, ["theta_0"])                      # separated: a real mixture
    assert res["keep"] == (True, [])                                 # the bug: still one Gaussian after training
    with pytest.raises(ValueError):
        Trainer(build_model(dict(model="GM-LAMP", T=1, precision="float64"), A),
                TrainConfig(gm_symmetry="nope", verbose=False), xt, yt, xv, yv)


def test_gmlamp_pre_v31_1_checkpoint_is_not_resumed(tmp_path, capsys):
    """A checkpoint of the old training (identical components after their stages) is set aside: the same command then
    gives exactly the network of a fresh v31.1 run. A v31.1 checkpoint resumes as before."""
    from bsce.train.trainer import Trainer, TrainConfig
    A, xt, yt, xv, yv = _gm_data()

    def cfg(mode):
        return TrainConfig(lr=1e-3, maxit=60, better_wait=1000, ivl=10, check_mult=5, batch_size=64, refinements=[0.5],
                           warm_start=True, gm_symmetry=mode, verbose=False)

    def run(mode, ck):
        torch.manual_seed(0)
        m = build_model(dict(model="GM-LAMP", T=1, K=4, precision="float64"), A)
        st = Trainer(m, cfg(mode), xt, yt, xv, yv, checkpoint=str(ck)).run()
        return m, st

    old = tmp_path / "old" / "ck"
    m0, _ = run("keep", old)                                         # the pre-v31.1 training
    assert m0.identical_gm_components() == ["theta_0"] and Path(str(old) + ".pt").exists()
    capsys.readouterr()
    m1, st1 = run("break", old)                                      # the same command with v31.1
    assert "written before v31.1" in capsys.readouterr().out
    assert (old.parent / "ck.pre_v31_1.pt").exists()
    m2, st2 = run("break", tmp_path / "new" / "ck")                  # a fresh v31.1 run
    assert m1.identical_gm_components() == [] and st1["gm_symmetry_broken"] == ["theta_0"]
    for k, v in m2.state_dict().items():
        assert torch.equal(v, m1.state_dict()[k]), k
    capsys.readouterr()
    m3, _ = run("break", tmp_path / "new" / "ck")                    # a v31.1 checkpoint: resumed, nothing retrained
    assert "written before v31.1" not in capsys.readouterr().out
    for k, v in m2.state_dict().items():
        assert torch.equal(v, m3.state_dict()[k]), k


def test_check_gm_components_script(tmp_path):
    """scripts/check_gm_components.py: an original-init GM-LAMP network is AFFECTED (exit 1), a spread one and a soft
    LAMP are not (exit 0), and a training checkpoint with identical components is listed apart."""
    A = _gm_data()[0]

    def run(*runs):
        return subprocess.run([sys.executable, "scripts/check_gm_components.py", "--runs", *map(str, runs)], cwd=ROOT,
                              capture_output=True, text=True)

    for name, cfg in (("orig", dict(model="GM-LAMP", T=2, K=4, precision="float64")),
                      ("spread", dict(model="GM-LAMP", T=2, K=4, precision="float64", gm_init="spread")),
                      ("lamp", dict(model="LAMP", T=2, precision="float64"))):
        (tmp_path / name).mkdir()
        save_model(build_model(cfg, A), tmp_path / name / f"{name}_0to20dB")
    r = run(tmp_path / "orig")
    assert r.returncode == 1 and "AFFECTED" in r.stdout and "theta_0, theta_1" in r.stdout, r.stdout + r.stderr
    r = run(tmp_path / "spread", tmp_path / "lamp")
    assert r.returncode == 0 and "AFFECTED" not in r.stdout and "no affected" in r.stdout, r.stdout + r.stderr
    m = build_model(dict(model="GM-LAMP", T=1, K=4, precision="float64"), A)
    (tmp_path / "ck" / "checkpoints").mkdir(parents=True)
    torch.save({"model_state": m.state_dict(), "cfg_model": m.cfg, "train_state": {"done": []}},
               tmp_path / "ck" / "checkpoints" / "x.pt")
    r = run(tmp_path / "ck")
    assert r.returncode == 0 and "old ckpt" in r.stdout, r.stdout + r.stderr


def test_suite_save_layers_with_end_to_end_methods():
    """v31.2: --save-layers with a mixed list goes on; only the layer-wise methods get --save-layers. A list in which
    no method trains layer by layer is still refused before any training."""
    cmds = _suite("--methods", "ladmm4", "gladmm8", "gladmmrw8", "nfladmmsr4", "--ft-snr", "none", "--save-layers")
    flag = {re.search(r"--tag (\S+)", c).group(1): "--save-layers" in c.split() for c in cmds}
    assert flag == {"LADMM-T4": True, "G-LADMM-T8_paper": False, "G-LADMM-RW-T8_paper": False, "NF-LADMM-SR-T4": True}
    r = subprocess.run([sys.executable, "scripts/run_suite.py", "--scenarios", "sv_upa", "--dry-run", "--methods",
                        "gladmm8", "gladmmrw8", "--ft-snr", "none", "--save-layers"], cwd=ROOT, capture_output=True,
                       text=True)
    assert r.returncode != 0 and "nothing would be saved per layer" in r.stderr

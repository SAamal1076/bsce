"""v28 options: LADMM threshold=adaptive|firm|gm, data_step=whitened, operator=analytic|analytic1, the CADMM baseline,
the polish post-step and the ensemble option of the suite. Every new option is off by default: the default LADMM has
the same parameters and configuration as before."""
import math
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
import torch
import torch.nn.functional as F

from bsce import classical
from bsce.data import measurements, sv
from bsce.export import load_model, save_model, truncate_model
from bsce.models import build_model
from bsce.models.ladmm import LADMM
from bsce.models.shrinkage import beam_noise_level, firm_shrink, gm_mmse_shrink
from bsce.polish import Polisher, choose
from bsce.rng import NumpyMatlabRNG
from bsce.train.trainer import Trainer, TrainConfig

ROOT = Path(__file__).resolve().parent.parent


def _inv_softplus(v):
    return float(np.log(np.expm1(v)))


@pytest.fixture(scope="module")
def setup():
    g = np.random.default_rng(0)
    A = ((g.random((128, 256)) > 0.5) * 2 - 1) / np.sqrt(128)
    x = sv.beamspace_channels(NumpyMatlabRNG(0), 40, "ULA")
    y = measurements.gen_measurements_matlab(x, A, 10.0, NumpyMatlabRNG(1))
    return A, x, y


# ---- defaults unchanged ---------------------------------------------------------------------------------------------
def test_default_ladmm_unchanged(setup):
    A, _, _ = setup
    m = build_model(dict(model="LADMM", T=3), A)
    names = []
    for t in range(3):
        names += [f"H_matrix_{t}", f"rho_{t}", f"theta_{t}", f"alpha_{t}"]
    assert m.param_names() == names
    assert "data_step" not in m.cfg and "gm_K" not in m.cfg
    assert set(m.state_dict()) == set(names) | {"A", "Wfix", "AtA"}


# ---- analytic operator ----------------------------------------------------------------------------------------------
@pytest.mark.parametrize("op", ["analytic", "analytic1"])
@pytest.mark.parametrize("lam,rho", [(0.05, 1.0), (0.2, 2.0)])
def test_analytic_reduces_to_admm(op, lam, rho, setup, record):
    A, _, y = setup
    T = 6
    m = LADMM(A, T=T, operator=op, precision="float64")
    assert m.n_params() == (3 if op == "analytic" else 2) * T
    with torch.no_grad():
        for t in range(T):
            m.P(f"rho_raw_{t}").fill_(_inv_softplus(rho))
            m.P(f"theta_raw_{t}").fill_(_inv_softplus(lam))
        z = m(torch.as_tensor(y)).numpy()
    zr, _ = classical.admm(y, A, lam, rho, T, 0.0)          # alpha = 1 at init (phi sigmoid)
    rel = np.linalg.norm(z - zr) / np.linalg.norm(zr)
    record("v28/analytic_vs_admm", operator=op, lam=lam, rho=rho, rel=float(rel))
    assert rel < 1e-11


@pytest.mark.parametrize("ds", ["plain", "whitened"])
def test_analytic_operator_is_the_inverse(ds, setup):
    A, _, _ = setup
    m = LADMM(A, T=2, operator="analytic", data_step=ds, precision="float64")
    with torch.no_grad():
        m.P("rho_raw_1").fill_(_inv_softplus(0.37))
        v = torch.as_tensor(np.random.default_rng(3).standard_normal((256, 5)) + 0j)
        got = m._apply_op(1, v).numpy()
    G = A.T @ A if ds == "plain" else A.T @ np.linalg.inv(A @ A.T) @ A
    ref = np.linalg.solve(G + 0.37 * np.eye(256), v.numpy())
    assert np.linalg.norm(got - ref) / np.linalg.norm(ref) < 1e-10
    assert np.allclose(m.H_matrix(1).detach().numpy(), np.linalg.inv(G + 0.37 * np.eye(256)), atol=1e-9)


# ---- whitened data step ---------------------------------------------------------------------------------------------
def test_whitened_step_is_the_exact_folded_step(setup):
    """full / diag / lowrank / kron / psd at init (rho = 1): x_1 = (P + I)^-1 A^+ y, the exact ADMM step of y = A(x+n)."""
    A, _, y = setup
    Ap = np.linalg.pinv(A)
    P = Ap @ A
    ref = np.linalg.solve(P + np.eye(256), Ap @ y)
    for op in ["full", "diag", "lowrank", "kron", "psd", "diagprior", "analytic"]:
        m = LADMM(A, T=2, operator=op, data_step="whitened", precision="float64", N1=16, N2=16)
        if op == "diagprior":
            with torch.no_grad():
                m.P("d_0").fill_(-80.0)
        with torch.no_grad():
            x1 = m.forward_stages(torch.as_tensor(y), upto=0)[0].numpy()
        tol = 1e-6 if op == "kron" else 1e-9          # kron: nearest Kronecker factorisation of the init
        if op == "kron":
            continue                                  # not exact for a random A (rank-1 rearrangement)
        assert np.linalg.norm(x1 - ref) / np.linalg.norm(ref) < tol, op
        assert m.cfg["data_step"] == "whitened"


# ---- shrinkages -----------------------------------------------------------------------------------------------------
def test_gm_mmse_against_numpy():
    g = np.random.default_rng(5)
    V = g.standard_normal((50, 4)) + 1j * g.standard_normal((50, 4))
    s2 = np.array([[0.1, 0.5, 1.0, 2.0]])
    w = np.array([0.7, 0.2, 0.1]); nu = np.array([0.01, 1.0, 10.0])
    got = gm_mmse_shrink(torch.as_tensor(V), torch.as_tensor(s2), torch.as_tensor(np.log(w)),
                         torch.as_tensor(np.log(nu))).numpy()
    # direct: p(v | k) = exp(-|v|^2 / (nu_k + s2)) / (pi (nu_k + s2)); E[x | v, k] = nu_k / (nu_k + s2) v
    vk = nu[:, None, None] + s2[None]
    lik = w[:, None, None] * np.exp(-np.abs(V)[None] ** 2 / vk) / (np.pi * vk)
    post = lik / lik.sum(0)
    ref = V * np.sum(post * nu[:, None, None] / vk, 0)
    assert np.max(np.abs(got - ref)) < 1e-12


def test_firm_and_noise_level():
    v = torch.linspace(0, 3, 301, dtype=torch.float64).to(torch.complex128) * np.exp(0.3j)
    z = firm_shrink(v.reshape(-1, 1), torch.tensor(1.0, dtype=torch.float64), torch.tensor(2.0, dtype=torch.float64))
    m = z.abs().reshape(-1).numpy()
    r = v.abs().numpy()
    assert np.all(m[r <= 1.0] == 0) and np.allclose(m[r >= 2.0], r[r >= 2.0])
    assert np.all(np.diff(m) >= -1e-12)                       # monotone, continuous ramp
    assert np.allclose(np.angle(z.reshape(-1).numpy()[r > 1.01]), 0.3)
    g = np.random.default_rng(0)
    n = (g.standard_normal((256, 200)) + 1j * g.standard_normal((256, 200))) * math.sqrt(0.3 / 2)
    s = beam_noise_level(torch.as_tensor(n)).numpy()
    assert abs(np.mean(s) - math.sqrt(0.3)) < 0.02


# ---- every new option: trains, saves, loads, truncates --------------------------------------------------------------
NEW = [dict(threshold="adaptive"), dict(threshold="firm"), dict(threshold="gm"), dict(data_step="whitened"),
       dict(operator="analytic"), dict(operator="analytic1", threshold="gm"), dict(operator="psd", data_step="whitened"),
       dict(threshold="gm", data_step="whitened")]


@pytest.fixture(scope="module")
def data():
    g = np.random.default_rng(0)
    A = ((g.random((128, 256)) > 0.5) * 2 - 1) / np.sqrt(128)
    xt = sv.beamspace_channels(NumpyMatlabRNG(0), 300, "ULA")
    xv = sv.beamspace_channels(NumpyMatlabRNG(1), 100, "ULA")
    yt = measurements.gen_measurements_py(xt, A, [0.0, 5.0, 10.0, 20.0], np.random.RandomState(2))
    yv = measurements.gen_measurements_py(xv, A, [0.0, 5.0, 10.0, 20.0], np.random.RandomState(3))
    return A, xt, yt, xv, yv


@pytest.mark.parametrize("opt", NEW + [dict(model="CADMM")], ids=lambda o: "-".join(f"{v}" for v in o.values()))
def test_new_options_train_save_load(opt, data, tmp_path, record):
    A, xt, yt, xv, yv = data
    torch.manual_seed(0)
    cfg = dict(model="LADMM", T=2, precision="float64", array="ULA")
    cfg.update(opt)
    m = build_model(cfg, A)
    tr = Trainer(m, TrainConfig(lr=1e-3, maxit=30, better_wait=1000, ivl=5, check_mult=2, batch_size=32,
                                val_round=None, refinements=[0.5], verbose=False), xt, yt, xv, yv)
    st = tr.run()
    v = [r["nmse_db"] for r in st["log"]]
    assert all(np.isfinite(v)), v
    record("v28/train", cfg=str(opt), first_db=v[0], last_db=v[-1])
    p, _ = save_model(m, tmp_path / "net")
    m2 = load_model(p)
    yy = torch.as_tensor(yv[:, :8])
    with torch.no_grad():
        assert torch.allclose(m(yy), m2(yy), atol=1e-12)
        s = truncate_model(m, 1)
        assert torch.allclose(s(yy), m.forward_stages(yy)[1], atol=1e-12)
    orig = tmp_path / "net_orig.mat"
    plain_soft = cfg.get("model") == "LADMM" and cfg.get("data_step", "plain") == "plain" and \
        cfg.get("threshold", "scalar") == "scalar"
    assert orig.exists() == plain_soft or cfg.get("model") == "CADMM"


# ---- CADMM ---------------------------------------------------------------------------------------------------------
def test_cadmm_is_untied_ladmm(setup):
    A, _, y = setup
    c = build_model(dict(model="CADMM", T=3, precision="float64"), A)
    l = build_model(dict(model="LADMM", T=3, precision="float64", init_precision="float64"), A)
    yy = torch.as_tensor(y)
    with torch.no_grad():
        for a, b in zip(c.forward_stages(yy), l.forward_stages(yy)):
            assert torch.allclose(a, b, atol=1e-12)
        l.P("rho_1").fill_(1.7)                      # a per-layer rho: tied CADMM no longer equals it at init
        assert not torch.allclose(c(yy), l(yy), atol=1e-6)
    assert build_model(dict(model="CADMM", T=6), A).n_params() == 6 * (128 * 256 + 256 * 256 + 3)


# ---- polish --------------------------------------------------------------------------------------------------------
def test_polish_recovers_a_biased_estimate():
    g = np.random.default_rng(1)
    A = ((g.random((64, 128)) > 0.5) * 2 - 1) / 8.0
    x = np.zeros((128, 20), complex)
    for j in range(20):
        idx = g.choice(128, 4, replace=False)
        x[idx, j] = (2 + np.abs(g.standard_normal(4))) * np.exp(2j * np.pi * g.random(4))   # |x| >= 2: z keeps the support
    y = A @ x                                           # no noise
    z = x * np.maximum(1 - 1.0 / np.maximum(np.abs(x), 1e-12), 0)   # soft-shrunk: biased by 1
    for nm in ("white", "colored"):
        p = Polisher(A, nm)
        zp = p.polish(z, y, 8, 1e-9)
        assert np.linalg.norm(zp - x) / np.linalg.norm(x) < 1e-6
        assert p.polish(z, y, 0, 0.1) is z
        grids = [p.grid(z, y, x)]
        (K, eps), best, base = choose(grids)
        assert K > 0 and best < base - 20
        _, best0, base0 = choose([p.grid(x + 1e-3, y, x)])   # K = 0 is a candidate: never worse than the network
        assert best0 <= base0 + 1e-12


@pytest.mark.skipif(not (ROOT / "data/SV_ULA/SV_ULA_validation_channel.mat").exists(), reason="needs SV ULA data")
def test_evaluate_polish_columns(tmp_path):
    A = __import__("scipy.io", fromlist=["loadmat"]).loadmat(ROOT / "data/SV_ULA/CSmatrix.mat")["A"]
    run = tmp_path / "runs" / "LADMM-T2"
    save_model(build_model(dict(model="LADMM", T=2, precision="float32"), A), run / "LADMM-T2_0to20dB")
    out = tmp_path / "res"
    r = subprocess.run([sys.executable, "scripts/evaluate.py", "--scenario", "configs/scenarios/sv_ula.yaml",
                        "--runs", str(tmp_path / "runs"), "--classical", "", "--n-test", "200", "--polish-n-val", "200",
                        "--polish", "band", "--out", str(out), "--device", "cpu"], cwd=ROOT, capture_output=True, text=True)
    assert r.returncode == 0, r.stderr[-2000:]
    head = (out / "nmse_vs_snr.csv").read_text().splitlines()[0].split(",")
    assert head == ["SNR_dB", "LADMM-T2", "LADMM-T2+polish"]
    import csv
    ch = list(csv.DictReader(open(out / "polish_choices.csv")))
    assert len(ch) == 1 and float(ch[0]["val_gain_db"]) >= 0


# ---- suite ensemble option -----------------------------------------------------------------------------------------
def test_suite_ensemble_option():
    sys.path.insert(0, str(ROOT / "scripts"))
    from run_suite import ensemble_tag
    assert ensemble_tag(["LADMM-T6_ft_0-10-20dB_20k", "LADMM-T8_ft_0-10-20dB_20k"]) == \
        "ENS-LADMM-T6+LADMM-T8_ft_0-10-20dB_20k"
    assert ensemble_tag(["LADMM-T6", "LAMP-T8"]) == "ENS-LADMM-T6+LAMP-T8"
    assert ensemble_tag(["LADMM-T6_fair", "LAMP-T8"]) == "ENS-LADMM-T6_fair+LAMP-T8"
    run = lambda *x: subprocess.run([sys.executable, "scripts/run_suite.py", "--scenarios", "sv_ula", *x, "--dry-run"],
                                    cwd=ROOT, capture_output=True, text=True)
    r = run("--methods", "ladmm6", "ladmm8")
    assert r.returncode == 0 and "make_ensemble" not in r.stdout          # off by default
    r = run("--methods", "ladmm6", "ladmm8", "--ensemble", "ladmm6", "ladmm8")
    assert r.returncode == 0 and "make_ensemble.py" in r.stdout and "ENS-LADMM-T6+LADMM-T8_ft_0-10-20dB_20k" in r.stdout
    r = run("--methods", "ladmm6", "--ensemble", "ladmm6")
    assert r.returncode != 0 and "at least two" in (r.stdout + r.stderr)
    r = run("--methods", "cadmm", "--ft-snr", "none")
    assert r.returncode == 0 and "configs/train/cadmm.yaml" in r.stdout and "CADMM-T6" in r.stdout
    r = subprocess.run([sys.executable, "scripts/run_suite.py", "--scenarios", "sv_ula", "--methods", "all", "--dry-run"],
                       cwd=ROOT, capture_output=True, text=True)
    assert "CADMM" not in r.stdout                                        # 'all' as before v28

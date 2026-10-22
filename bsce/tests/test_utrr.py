"""UTRR (bsce/models/utrr.py; Wu, Cheng, Eldar, Cioffi, IEEE TCOM 2025) and the trainer options it uses.

1. the bsce model equals the standalone reference tests/proposals/utrr_ref.py (written from the paper alone) for
   perturbed parameters, with and without batch normalisation, with batch and with running statistics
2. at initialisation (no BN) the network is the paper's unrolled iteration (eq. 23: K = 0 in the inner layers, top-K
   in the last, beta = 1/2), computed here with NumPy
3. parameter count T (2MN + 2) without BN (the paper's count)
4. forward_stages(y, upto=t)[-1] = the (t+1)-layer network (truncate_model), and .pt / .mat round trips keep the
   output, BN running statistics included; the stage plan covers every parameter once
5. ITRR (the paper's Algorithm 1) recovers an exactly sparse vector to machine precision; with the printed sign of
   grad F it diverges
6. trainer: end2end with lr_list (one stage per rate, one epoch budget) and patience (early stopping); end2end without
   them is unchanged (ADMM-Net); the 'mse' loss
7. run_suite knows the method key utrr (T = 30)
8. the model-averaging ensemble (UTRR-E) and scripts/make_ensemble.py
No data or GPU needed.
"""
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tests" / "proposals"))
from bsce.export import load_model, save_model, truncate_model  # noqa: E402
from bsce.models import build_model  # noqa: E402
from bsce.train.losses import LOSSES  # noqa: E402
from bsce.train.trainer import Trainer, TrainConfig, build_stages  # noqa: E402
from utrr_ref import UTRRNet, itrr, topk_keep  # noqa: E402

M, N = 32, 64


def setup(seed=0, B=40):
    r = np.random.default_rng(seed)
    A = r.choice([-1.0, 1.0], (M, N)) / np.sqrt(M)
    X = np.zeros((N, B), complex)
    for b in range(B):
        i = r.choice(N, 4, replace=False)
        X[i, b] = r.standard_normal(4) + 1j * r.standard_normal(4)
    Y = A @ (X + 0.05 * (r.standard_normal(X.shape) + 1j * r.standard_normal(X.shape)))
    return A, X, Y


def perturb(m, seed=3):
    g = torch.Generator().manual_seed(seed)
    with torch.no_grad():
        for n in m.param_names():
            p = m.P(n)
            p.add_(0.02 * torch.randn(p.shape, generator=g, dtype=p.dtype))


@pytest.mark.parametrize("bn", [False, True])
def test_matches_reference(bn):
    A, X, Y = setup()
    m = build_model(dict(model="UTRR", T=5, K=6, bn=bn, precision="float64"), A)
    perturb(m)
    r = UTRRNet(A, L=5, K=6, bn=bn, dtype=torch.float64)
    with torch.no_grad():
        for t in range(5):
            r.A[t].copy_(m.P(f"A_{t}")); r.rho[t].copy_(m.P(f"rho_{t}")); r.alpha[t].copy_(m.P(f"alpha_{t}"))
            if bn:
                r.bns[t].gamma.copy_(m.P(f"bn{t + 1}_gamma")); r.bns[t].beta.copy_(m.P(f"bn{t + 1}_beta"))
        if bn:
            r.bn0.gamma.copy_(m.P("bn0_gamma")); r.bn0.beta.copy_(m.P("bn0_beta"))
    y = torch.tensor(Y)
    assert torch.allclose(m(y), r.forward_complex(y), atol=1e-12, rtol=0)            # batch statistics (training)
    with torch.no_grad():
        a = m(y)
        r.eval()
        b = r.forward_complex(y)
    assert torch.allclose(a, b, atol=1e-12, rtol=0)                                  # running statistics


def test_init_is_the_paper_iteration():
    A, X, Y = setup(1)
    T, K = 9, 5
    m = build_model(dict(model="UTRR", T=T, K=K, bn=False, precision="float64"), A)
    yr = np.concatenate([Y.real, Y.imag], 1)
    A2 = np.hstack([A, -A])
    x0 = A.T @ yr
    z = np.vstack([np.maximum(x0, 0), np.maximum(-x0, 0)])
    for t in range(T):
        g = A2.T @ (A2 @ z - yr) + 1.0 * (z - topk_keep(z, K if t == T - 1 else 0))
        z = (np.maximum(z - 0.1 * g, 0) + z) / 2
    xr = z[:N] - z[N:]
    with torch.no_grad():
        out = m(torch.tensor(Y)).numpy()
    B = Y.shape[1]
    assert np.allclose(out, xr[:, :B] + 1j * xr[:, B:], atol=1e-12, rtol=0)


def test_parameter_count():
    A, _, _ = setup()
    T = 30
    assert build_model(dict(model="UTRR", T=T), A).n_params() == T * (2 * M * N + 2)          # default: no BN
    assert build_model(dict(model="UTRR", T=T, bn=False), A).n_params() == T * (2 * M * N + 2)
    assert build_model(dict(model="UTRR", T=T, bn=True), A).n_params() == T * (2 * M * N + 2) + (T + 1) * 2 * 2 * N


def test_truncation_save_load_and_stage_plan(tmp_path):
    A, X, Y = setup(2)
    m = build_model(dict(model="UTRR", T=6, K=6, bn=True, precision="float64"), A)
    perturb(m, 5)
    y = torch.tensor(Y)
    for _ in range(3):                                                       # move the running statistics
        m(y)
    with torch.no_grad():
        for t in (0, 2, 5):
            s = truncate_model(m, t + 1)
            assert torch.allclose(m.forward_stages(y, upto=t)[-1], s(y), atol=1e-12, rtol=0)
        ref = m(y)
    pt, mat = save_model(m, tmp_path / "u")
    for f in (pt, mat):
        with torch.no_grad():
            assert torch.allclose(load_model(f)(y), ref, atol=1e-10, rtol=0), f
    plan = m.stage_plan()
    assert len(plan) == 6 and plan[0].new[:2] == ["bn0_gamma", "bn0_beta"]
    assert sorted(plan[-1].all) == sorted(m.param_names()) and len(set(plan[-1].all)) == len(plan[-1].all)


def test_itrr_exact_recovery_and_printed_sign():
    r = np.random.default_rng(7)
    Mi, Ni, s = 64, 128, 6
    Phi = r.choice([-1.0, 1.0], (Mi, Ni)) / np.sqrt(Mi)
    x = np.zeros(Ni)
    x[r.choice(Ni, s, replace=False)] = r.standard_normal(s) + np.sign(r.standard_normal(s))
    xh, h = itrr(Phi @ x, Phi, K=s, rho=1.0, n_iter=3000, step="fixed", x_true=x)
    assert h["err"][-1][0] < 1e-20
    _, h2 = itrr(Phi @ x, Phi, K=s, rho=1.0, n_iter=40, step="fixed", sign="paper", x_true=x)
    assert h2["err"][-1][0] > 1e3


def test_trainer_lr_list_and_patience():
    A, X, Y = setup(4, B=120)
    m = build_model(dict(model="UTRR", T=3, K=4, bn=True), A)
    cfg = TrainConfig(schedule="end2end", lr_list=[1e-2, 1e-3, 1e-4], epochs=5, patience=2, batch_size=16, loss="mse",
                      val_round=None, optimizer="adam_torch", restore_best=True, verbose=False)
    st = build_stages(m, cfg)
    assert [s["name"] for s in st] == ["end2end lr=0.01", "end2end lr=0.001", "end2end lr=0.0001"]
    tr = Trainer(m, cfg, X[:, :100], Y[:, :100], X[:, 100:], Y[:, 100:])
    state = tr.run()
    assert state["epochs_done"] <= 5 and set(state["done"]) == {s["name"] for s in st}
    assert all(e["stage"].startswith("end2end lr=") for e in state["log"])
    # end2end without lr_list: as before (ADMM-Net), every stage runs `epochs` epochs
    a = build_model(dict(model="ADMM-Net", T=2), A)
    cfg2 = TrainConfig(schedule="end2end", lr=1e-4, epochs=2, batch_size=32, val_round=None, optimizer="adam_torch",
                       verbose=False)
    s2 = Trainer(a, cfg2, X[:, :100], Y[:, :100], X[:, 100:], Y[:, 100:]).run()
    assert [e["stage"] for e in s2["log"]] == ["epoch 1", "epoch 2"] and "epochs_done" not in s2
    xh, x = torch.tensor(X[:, :3]), torch.tensor(X[:, 3:6])
    assert torch.isclose(LOSSES["mse"](xh, x), torch.mean(torch.sum(torch.abs(xh - x) ** 2, 0)))


def test_run_suite_knows_utrr():
    r = subprocess.run([sys.executable, str(ROOT / "scripts/run_suite.py"), "--scenarios", "sv_ula", "--methods", "utrr",
                        "--dry-run"], capture_output=True, text=True, cwd=ROOT)
    assert r.returncode == 0, r.stdout + r.stderr
    assert "--train configs/train/utrr.yaml" in r.stdout and "--tag UTRR-T30" in r.stdout and "model.T=30" in r.stdout


def test_ensemble_and_make_ensemble(tmp_path):
    A, X, Y = setup(6)
    ms = []
    for i, K in enumerate((0, 6, 12)):
        m = build_model(dict(model="UTRR", T=3, K=K, bn=False, precision="float64"), A)
        perturb(m, 10 + i)
        ms.append(m)
        run = tmp_path / f"U{K}"
        save_model(m, run / f"U{K}_0to20dB")
    from bsce.models.ensemble import Ensemble
    y = torch.tensor(Y)
    e = Ensemble.from_models(ms)
    with torch.no_grad():
        mean = torch.stack([m(y) for m in ms]).mean(0)
        assert torch.allclose(e(y), mean, atol=1e-12, rtol=0)
    r = subprocess.run([sys.executable, str(ROOT / "scripts/make_ensemble.py"), "--runs", *[str(tmp_path / f"U{K}") for K in (0, 6, 12)],
                        "--tag", "U-E"], capture_output=True, text=True, cwd=ROOT)
    assert r.returncode == 0, r.stdout + r.stderr
    f = tmp_path / "U-E" / "U-E_0to20dB.pt"
    with torch.no_grad():
        assert torch.allclose(load_model(f)(y), mean, atol=1e-12, rtol=0)
    assert (tmp_path / "U-E" / "README.txt").exists()
    save_model(ms[0], tmp_path / "V" / "V_0to10dB")
    r = subprocess.run([sys.executable, str(ROOT / "scripts/make_ensemble.py"), "--runs", str(tmp_path / "U0"), str(tmp_path / "V"),
                        "--tag", "X"], capture_output=True, text=True, cwd=ROOT)
    assert r.returncode != 0 and "no band file in common" in (r.stdout + r.stderr)

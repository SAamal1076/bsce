"""LISTA / ALISTA (ALISTA repo) and LVAMP (Borgerding) vs the ORIGINAL TF code.

Real-valued problems (the originals are real): every layer output, gradients and Adam
updates of every training stage of the original setup functions
(utils.train.setup_sc_training / tools.train.setup_training), in
  * float64 copies of the references (machine precision), and
  * the unchanged float32 references (float32 tolerance).
Also: ALISTA's analytic W (CalculateW.m in Octave vs Python port), and complex-mode
consistency (complex code with real inputs == real code).
"""
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
import scipy.io as sio
import torch

from harness.octave import REFERENCE, compare, octave_available, run_octave, tf_available, tmpdir

needs_tf = pytest.mark.skipif(not tf_available("sklearn"), reason="tensorflow / scikit-learn not installed (requirements-verify.txt)")
from bsce.models.lista import ALISTA, LISTA, alista_W
from bsce.models.lvamp import LVAMP
from bsce.train.trainer import TrainConfig, build_stages
from bsce.train.losses import l2_loss
from bsce.train.optim import AdamTF1

HERE = Path(__file__).resolve().parent
RUNNER = HERE / "harness" / "ref_tf_ista.py"
LR = 2.0 ** -12          # exactly representable in float32 (TF casts lr tensors to float32)


def _problem(seed, M=64, N=128, B=40):
    g = np.random.default_rng(seed)
    A = ((g.random((M, N)) > 0.5) * 2 - 1) / np.sqrt(M)
    x = (g.random((N, B)) < 0.1) * g.standard_normal((N, B))
    y = A @ x + 0.01 * g.standard_normal((M, B))
    return A, x, y


def _run(**kw):
    wd = tmpdir("bsce_tf_")
    np.savez(wd / "in.npz", **kw)
    subprocess.run([sys.executable, str(RUNNER), str(wd / "in.npz"), str(wd / "out.npz")], check=True,
                   capture_output=True)
    return dict(np.load(wd / "out.npz", allow_pickle=True))


def _perturb(model, seed):
    g = np.random.default_rng(seed + 77)
    with torch.no_grad():
        for n in model.param_names():
            p = model.P(n)
            p.add_(torch.as_tensor(g.standard_normal(p.shape) * 0.02 * max(float(p.abs().max()), 0.1), dtype=p.dtype))
    return {n: model.P(n).detach().numpy().copy() for n in model.param_names()}


def _check(model, ref, stages, w, x, y, tol, gtol, nsteps):
    yt, xt = torch.as_tensor(y, dtype=model.rdtype), torch.as_tensor(x, dtype=model.rdtype)
    with torch.no_grad():
        outs = model.forward_stages(yt)
    frel = max(compare(o.numpy(), ref[f"out_{i}"], f"out {i}", tol)["rel"] for i, o in enumerate(outs))
    gmax = amax = 0.0
    for si, st in enumerate(stages):
        model.load_mat_dict(w)
        model.zero_grad(set_to_none=True)
        l2_loss(model.forward_stages(yt, upto=st["out"])[-1], xt).backward()
        for n in st["params"]:
            if f"grad_{si}_{n}" in ref:
                gmax = max(gmax, compare(model.P(n).grad.numpy(), ref[f"grad_{si}_{n}"], f"grad {si} {n}", gtol)["rel"])
        scale = {id(model.P(n)): s for n, s in st["scale"].items()} if st["scale"] else None
        opt = AdamTF1([model.P(n) for n in st["params"]], lr=st["lr"], grad_scale=scale)
        for _ in range(nsteps):
            opt.zero_grad(set_to_none=True)
            l2_loss(model.forward_stages(yt, upto=st["out"])[-1], xt).backward()
            opt.step()
        for n in st["params"]:
            amax = max(amax, compare(model.P(n).detach().numpy() - w[n], ref[f"after_{si}_{n}"] - w[n],
                                     f"Adam {si} {n}", gtol)["rel"])
    return frel, gmax, amax


@needs_tf
@pytest.mark.parametrize("net", ["LISTA", "ALISTA"])
@pytest.mark.parametrize("prec", ["float64", "float32"])
@pytest.mark.parametrize("seed", [0, 1])
def test_lista_alista_vs_tf(net, prec, seed, record):
    A, x, y = _problem(seed)
    T, lam = 5, 0.4
    W = alista_W(A)
    kw = dict(percent=4.0, max_percent=12.0)
    if net == "LISTA":
        model = LISTA(A, T=T, lam=lam, untied=True, complex_mode=False, precision=prec)
    else:
        model = ALISTA(A, T=T, lam=lam, W=W, complex_mode=False, precision=prec, **kw)
    if prec == "float32":   # the original casts A (and W) to float32 before everything
        model.A.copy_(torch.as_tensor(A.astype(np.float32)))
    w = _perturb(model, seed)
    cfg = TrainConfig(schedule="alista", lr=LR, lr_decay=(0.25, 0.0625), decay_rate=0.5)
    stages = build_stages(model, cfg)
    refdir = REFERENCE / ("alista_f64" if prec == "float64" else "alista")
    ref = _run(net=net, refdir=str(refdir), A=A, y=y, x=x, T=T, lam=lam, untied=True, W=W, init_lr=LR,
               decay_rate=0.5, lr_decay=np.array([0.25, 0.0625]), refinements=np.array([0.5]), nsteps=4,
               **{f"w_{k}": v for k, v in w.items()}, **kw)
    assert [",".join(s["params"]) for s in stages] == list(ref["stage_vars"])
    if prec == "float64":
        for n in model.param_names():
            pass
    tol, gtol = (1e-12, 1e-10) if prec == "float64" else (2e-5, 2e-3)
    frel, gmax, amax = _check(model, ref, stages, w, x, y, tol, gtol, 4)
    record("ista/vs_tf", net=net, precision=prec, seed=seed, n_stages=len(stages), fwd_rel=frel,
           grad_rel=gmax, adam_rel=amax)


@needs_tf
@pytest.mark.parametrize("prec", ["float64", "float32"])
@pytest.mark.parametrize("seed", [0, 1, 2])
def test_lvamp_vs_tf(prec, seed, record):
    A, x, y = _problem(seed)
    T = 5
    model = LVAMP(A, T=T, complex_mode=False, precision=prec)
    w = _perturb(model, seed)
    cfg = TrainConfig(schedule="borgerding", lr=LR, refinements=(0.5, 0.25, 0.125), loss="l2",
                      refine_name="value")
    stages = build_stages(model, cfg)
    refdir = REFERENCE / ("borgerding_py3_f64" if prec == "float64" else "borgerding_py3")
    ref = _run(net="LVAMP", refdir=str(refdir), A=A, y=y, x=x, T=T, init_lr=LR,
               refinements=np.array([0.5, 0.25, 0.125]), nsteps=4, **{f"w_{k}": v for k, v in w.items()})
    assert [s["name"] for s in stages] == list(ref["stage_names"])
    assert [",".join(s["params"]) for s in stages] == list(ref["stage_vars"])
    tol, gtol = (1e-12, 1e-10) if prec == "float64" else (2e-5, 2e-3)
    frel, gmax, amax = _check(model, ref, stages, w, x, y, tol, gtol, 4)
    record("lvamp/vs_tf", precision=prec, seed=seed, n_stages=len(stages), fwd_rel=frel, grad_rel=gmax,
           adam_rel=amax)


@pytest.mark.skipif(not octave_available(), reason="octave not installed")
@pytest.mark.skipif(not (REFERENCE / "alista").exists(), reason="needs reference/alista (bsce_reference zips)")
@pytest.mark.parametrize("seed", [0, 1, 2])
def test_alista_W_vs_octave(seed, record):
    """matlabs/CalculateW.m run unchanged (loads ./D.mat, saves W.mat)."""
    g = np.random.default_rng(seed)
    D = g.standard_normal((40, 80))
    D = D / np.linalg.norm(D, axis=0)
    wd = tmpdir()
    # CalculateW.m is a script with local functions at the end (MATLAB >= R2016b). Octave
    # needs them as separate files, so the file is split mechanically (no line is changed).
    src = (REFERENCE / "alista" / "matlabs" / "CalculateW.m").read_text(encoding="utf-8").splitlines()
    first_fn = next(i for i, l in enumerate(src) if l.startswith("function "))
    (wd / "CalculateW_body.m").write_text("\n".join(src[:first_fn]) + "\n", encoding="utf-8")
    chunk, name = [], None
    for l in src[first_fn:] + ["function __end__"]:
        if l.startswith("function "):
            if name:
                (wd / f"{name}.m").write_text("\n".join(chunk) + "\n", encoding="utf-8")
            name = l.split("=")[-1].split("(")[0].replace("function", "").strip()
            chunk = []
        chunk.append(l)
    sio.savemat(wd / "D.mat", {"D": D})
    try:
        run_octave("set(0,'defaultfigurevisible','off'); CalculateW_body", wd)
    except RuntimeError:
        pass   # visualization() may fail headless; W.mat is saved before it
    Wref = sio.loadmat(wd / "W.mat")["W"]
    r = compare(alista_W(D), Wref, f"ALISTA W seed{seed}", 1e-10)
    record("alista/W_vs_octave", seed=seed, rel=r["rel"], max_abs=r["max_abs"])
    shutil.rmtree(wd)


@pytest.mark.parametrize("cls", ["LISTA", "ALISTA", "LVAMP_soft_consistency"])
def test_complex_mode_consistency(cls, record):
    """Complex-mode network on real inputs reproduces the real network (soft shrinkages)."""
    A, x, y = _problem(5)
    if cls == "LISTA":
        mr, mc = LISTA(A, T=6, complex_mode=False), LISTA(A, T=6, complex_mode=True)
    elif cls == "ALISTA":
        W = alista_W(A)
        mr, mc = ALISTA(A, T=6, W=W, complex_mode=False, percent=4, max_percent=12), \
            ALISTA(A, T=6, W=W, complex_mode=True, percent=4, max_percent=12)
    else:
        pytest.skip("complex BG prior is a different (circular) model by design")
    with torch.no_grad():
        a = mr(torch.as_tensor(y)).numpy()
        b = mc(torch.as_tensor(y + 0j)).numpy()
    r = compare(b.real, a, f"{cls} complex vs real", 1e-14)
    assert np.abs(b.imag).max() == 0
    record("ista/complex_consistency", net=cls, rel=r["rel"])

"""LADMM family: PyTorch vs the ORIGINAL code.

1. MATLAB inference: LADMM_test.m / LADMM_UPA_test.m (unchanged, Octave) with the trained
   DeepMIMO LADMM weights, 100 channels, several SNRs/seeds (also the per-layer NMSE output).
2. Classical ADMM: LADMM with analytic parameters == ADMM_vectorized.m for T iterations.
3. TF training graphs of the LADMM scripts (full float32 & float64, diag, low-rank, Kronecker):
   initial values, all stage outputs, gradients, Adam updates.
"""
import os
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
import scipy.io as sio
import torch

from harness.octave import REFERENCE, compare, octave_available, run_octave, tf_available, tmpdir
from bsce.data import sv, measurements
from bsce.rng import NumpyMatlabRNG
from bsce.models.ladmm import LADMM
from bsce.train.trainer import TrainConfig, build_stages
from bsce.train.losses import nmse_loss
from bsce.train.optim import AdamTF1

HERE = Path(__file__).resolve().parent
USER = REFERENCE / "ladmm_user"
# trained DeepMIMO LADMM weights (your upload), copied unchanged; BSCE_LADMM_MODELS overrides
MODELS = Path(os.environ.get("BSCE_LADMM_MODELS", str(USER / "models")))
needs_oct = pytest.mark.skipif(not octave_available(), reason="octave not installed")


def _A_double():
    return sio.loadmat(REFERENCE / "gmlamp" / "CE_DeepMIMO" / "CSmatrix_128256.mat")["A"].astype(np.float64)


def _x(array, seed, K=100):
    x = sio.loadmat(REFERENCE / "gmlamp" / "CE_DeepMIMO" / f"{array}_test_channel.mat")["x"]
    g = np.random.default_rng(seed)
    return x[:, g.choice(x.shape[1], K, replace=False)]


# ------------------------------------------------------------------------------------
# 1. MATLAB inference
# ------------------------------------------------------------------------------------
ULA_NAMES = {1: "0to5dB_1dB_T=6_it=15k_12", 2: "5to10dB_6dB_T=6_it=15k_12",
             3: "10to15dB_11dB_T=6_it=15k_12", 4: "15to20dB_16dB_T=6_it=15k_12"}
UPA_NAMES = {1: "0to5dB_1dB", 2: "5to10dB_6dB", 3: "10to15dB_11dB", 4: "15to20dB_16dB"}
UPA_T = {1: 5, 2: 4, 3: 5, 4: 5}   # depths hard-coded in LADMM_UPA_test.m


@needs_oct
@pytest.mark.skipif(not MODELS.exists(), reason="trained LADMM weights not available")
@pytest.mark.parametrize("array", ["ULA", "UPA"])
@pytest.mark.parametrize("snr", [2.0, 8.0, 12.0, 17.0])
@pytest.mark.parametrize("seed", [0, 1])
def test_ladmm_vs_matlab(array, snr, seed, record):
    A = _A_double()
    x = _x(array, seed)
    y = measurements.gen_measurements_matlab(x, A, snr, NumpyMatlabRNG(500 + seed))
    rng_idx = 1 if snr < 5 else 2 if snr < 10 else 3 if snr < 15 else 4
    src = MODELS / (f"Trained_LADMM_for_DeepMIMO_{array}_1282560to10dB_1dB.mat" if snr < 10
                    else f"Trained_LADMM_for_DeepMIMO_{array}_12825610to20dB_11dB.mat")
    wd = tmpdir()
    if array == "ULA":
        fn, typ, T, func = f"Trained_LADMM_for_SV_ULA_256128{ULA_NAMES[rng_idx]}.mat", 1, 6, "LADMM_test"
        shutil.copy(USER / "ULA_test_code" / "LADMM_test.m", wd)
    else:
        fn, typ, T, func = f"Trained_LADMM_for_SV_UPA_256128{UPA_NAMES[rng_idx]}.mat", 2, UPA_T[rng_idx], "LADMM_UPA_test"
        # LADMM_UPA_test.m sets T=5 *before* load(...) for range 1; a weight file that
        # contains a variable 'T' silently overrides it (the DeepMIMO files store T=6).
        Wf = sio.loadmat(src)
        if rng_idx == 1 and "T" in Wf:
            T = int(np.asarray(Wf["T"]).ravel()[0])
        shutil.copy(USER / "UPA_test_code" / "LADMM_UPA_test.m", wd)
    shutil.copy(src, wd / fn)       # the unchanged .m loads this file name
    sio.savemat(wd / "in.mat", {"y": y, "A": A, "x": x})
    run_octave(f"load in.mat; [xh, nm] = {func}(y, A, {typ}, {rng_idx}, x); save('-v7','out.mat','xh','nm');", wd)
    ref = sio.loadmat(wd / "out.mat")
    model = LADMM(A, T=T, operator="full", precision="float64")
    model.load_mat_dict(sio.loadmat(src))
    with torch.no_grad():
        outs = model.forward_stages(torch.as_tensor(y))
    r = compare(outs[-1].numpy(), ref["xh"], f"LADMM {array} {snr}dB", 1e-12)
    nm_py = [10 * np.log10(np.mean(np.sum(np.abs(outs[2 * t + 1].numpy() - x) ** 2, 0)) /
                           np.mean(np.sum(np.abs(x) ** 2, 0))) for t in range(T)]
    r2 = compare(np.array(nm_py), ref["nm"].ravel(), "per-layer NMSE", 1e-12)
    record("ladmm/vs_matlab", array=array, snr=snr, seed=seed, T=T, rel=r["rel"], max_abs=r["max_abs"],
           layer_nmse_rel=r2["rel"], nmse_db_last=nm_py[-1])
    shutil.rmtree(wd)


# ------------------------------------------------------------------------------------
# 2. Analytic LADMM == classical ADMM (ADMM_vectorized.m)
# ------------------------------------------------------------------------------------
@needs_oct
@pytest.mark.skipif(not USER.exists(), reason="needs reference/ladmm_user (bsce_reference zips)")
@pytest.mark.parametrize("lam,rho,T", [(0.05, 1.0, 6), (0.2, 1.0, 10), (0.1, 2.0, 15)])
def test_ladmm_reduces_to_admm(lam, rho, T, record):
    g = np.random.default_rng(3)
    A = ((g.random((128, 256)) > 0.5) * 2 - 1) / np.sqrt(128)
    x = sv.beamspace_channels(NumpyMatlabRNG(4), 50, "UPA")
    y = measurements.gen_measurements_matlab(x, A, 10.0, NumpyMatlabRNG(5))
    wd = tmpdir()
    shutil.copy(USER / "ULA_test_code" / "ADMM_vectorized.m", wd)
    sio.savemat(wd / "in.mat", {"y": y, "A": A})
    run_octave(f"load in.mat; [h, c] = ADMM_vectorized(y, A, {lam}, {rho}, {T}, 0); save('-v7','out.mat','h');", wd)
    ref = sio.loadmat(wd / "out.mat")["h"]
    model = LADMM(A, T=T, operator="full", precision="float64")
    W = np.linalg.inv(A.T @ A + rho * np.eye(256))
    d = {}
    for t in range(T):
        d[f"H_matrix_{t}"], d[f"rho_{t}"], d[f"theta_{t}"], d[f"alpha_{t}"] = W, rho, np.array([lam]), 1.0
    model.load_mat_dict(d)
    with torch.no_grad():
        z = model(torch.as_tensor(y)).numpy()
    r = compare(z, ref, f"ADMM T={T}", 1e-11)
    record("ladmm/equals_admm", lam=lam, rho=rho, T=T, rel=r["rel"], max_abs=r["max_abs"])
    shutil.rmtree(wd)


# ------------------------------------------------------------------------------------
# 3. TF training graphs
# ------------------------------------------------------------------------------------
TF_CASES = [
    # script, operator, precision, kwargs for build_LADMM, refinements
    ("LADMM_train.py", "full", "float32", {"shrink": "soft", "untied": True}, (.5, .1, .01)),
    ("A_LADMM_UPA_SV_0_20dB.py", "full", "float64", {"shrink": "soft", "untied": True}, (.5, .1, .01)),
    ("A_UPA_DeepMIMO_0_10dB_DiagonalScaling.py", "diag", "float64", {"shrink": "soft", "untied": True}, (.5, .1, .01)),
    ("B_UPA_DeepMIMO_0_10dB_LowRank.py", "lowrank", "float64", {"shrink": "soft", "untied": True, "rank": 16}, (.5, .1, .01)),
    ("C_UPA_DeepMIMO_0_10dB_Kronecker.py", "kron", "float64", {"N1": 16, "N2": 16, "shrink": "soft", "untied": True}, (.5, .1, .01)),
]


@pytest.mark.skipif(not tf_available(), reason="tensorflow not installed (requirements-verify.txt)")
@pytest.mark.parametrize("script,operator,precision,kw,refs", TF_CASES)
@pytest.mark.parametrize("seed", [0, 1])
def test_ladmm_vs_tf(script, operator, precision, kw, refs, seed, record):
    g = np.random.default_rng(seed)
    A = ((g.random((128, 256)) > 0.5) * 2 - 1) / np.sqrt(128)
    if precision == "float32":
        A = A.astype(np.float32)
    x = sv.beamspace_channels(NumpyMatlabRNG(seed), 48, "UPA")
    y = measurements.gen_measurements_matlab(x, A.astype(np.float64), 6.0 + 4 * seed, NumpyMatlabRNG(seed + 1))
    T, lr, nsteps = 6, 1e-3, 4
    model = LADMM(A, T=T, operator=operator, rank=kw.get("rank", 16), precision=precision,
                  init_precision=precision)
    init = {n: model.P(n).detach().numpy().copy() for n in model.param_names()}
    with torch.no_grad():
        for n in model.param_names():
            p = model.P(n)
            scale = 0.02 * float(p.abs().max()) if p.numel() > 1 else 0.05
            p.add_(torch.as_tensor(g.standard_normal(p.shape) * scale, dtype=p.dtype))
    weights = {f"w_{n}": model.P(n).detach().numpy().copy() for n in model.param_names()}
    cfg = TrainConfig(lr=lr, refinements=refs)
    stages = build_stages(model, cfg)
    test_stages = sorted({0, 1, 3, 4, 7, 12, 17, len(stages) - 5, len(stages) - 1})
    wd = tmpdir("bsce_tf_")
    np.savez(wd / "in.npz", script=str(USER / script), A=A, y=y, x=x, T=T, kwargs=np.array(kw, dtype=object),
             lr=lr, refinements=np.array(refs), nsteps=nsteps, test_stages=np.array(test_stages), **weights)
    subprocess.run([sys.executable, str(HERE / "harness" / "ref_tf_ladmm.py"), str(wd / "in.npz"),
                    str(wd / "out.npz")], check=True, capture_output=True)
    ref = dict(np.load(wd / "out.npz", allow_pickle=True))
    assert [s["name"] for s in stages] == list(ref["stage_names"]), "stage names differ"
    assert [",".join(s["params"]) for s in stages] == list(ref["stage_vars"]), "stage variable lists differ"
    tol = 1e-5 if precision == "float32" else 1e-12
    irel = max(compare(init[n], ref[f"init_{n}"], f"init {n}", tol)["rel"] for n in init)
    yt, xt = torch.as_tensor(y), torch.as_tensor(x)
    with torch.no_grad():
        outs = model.forward_stages(yt)
    frel = max(compare(o.numpy(), ref[f"out_{i}"], f"out {i}", tol)["rel"] for i, o in enumerate(outs))
    gmax = amax = 0.0
    gtol = 1e-4 if precision == "float32" else 1e-9
    for si in test_stages:
        st = stages[si]
        model.load_mat_dict({n: weights[f"w_{n}"] for n in model.param_names()})
        model.zero_grad(set_to_none=True)
        nmse_loss(model.forward_stages(yt, upto=st["out"])[-1], xt).backward()
        for n in model.param_names():
            key = f"grad_{si}_{n}"
            if key in ref:
                gmax = max(gmax, compare(model.P(n).grad.numpy(), ref[key], f"grad {st['name']} {n}", gtol)["rel"])
        opt = AdamTF1([model.P(n) for n in st["params"]], lr=st["lr"])
        for _ in range(nsteps):
            opt.zero_grad(set_to_none=True)
            nmse_loss(model.forward_stages(yt, upto=st["out"])[-1], xt).backward()
            opt.step()
        for n in st["params"]:
            w0 = weights[f"w_{n}"]
            amax = max(amax, compare(model.P(n).detach().numpy() - w0, ref[f"after_{si}_{n}"] - w0,
                                     f"Adam {st['name']} {n}", gtol)["rel"])
    record("ladmm/vs_tf", script=script, operator=operator, precision=precision, seed=seed,
           init_rel=irel, fwd_rel=frel, grad_rel=gmax, adam_rel=amax, n_stages=len(stages))

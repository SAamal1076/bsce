"""LAMP / GM-LAMP: PyTorch vs the ORIGINAL TF training graph (authors' define_networks.py).

Checks, with identical (randomly perturbed) weights and data:
  * every layer output of the graph (all 16 stage points for T=8),
  * gradients of the stage losses wrt all trainable variables,
  * parameters after several AdamOptimizer steps (TF1 Adam vs bsce AdamTF1),
for the soft-threshold (LAMP) and Gaussian-mixture (GM-LAMP) shrinkage, several seeds.
"""
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

from harness.octave import compare, tf_available, tmpdir
from bsce.data import sv, measurements
from bsce.rng import NumpyMatlabRNG
from bsce.models.lamp import LAMP
from bsce.train.trainer import TrainConfig, build_stages
from bsce.train.losses import nmse_loss
from bsce.train.optim import AdamTF1

HERE = Path(__file__).resolve().parent
RUNNER = HERE / "harness" / "ref_tf_gmlamp.py"
pytestmark = pytest.mark.skipif(not tf_available(), reason="tensorflow not installed (requirements-verify.txt)")
TOL_FWD, TOL_GRAD, TOL_ADAM = 1e-12, 1e-10, 1e-10


def _problem(seed, B=64):
    g = np.random.default_rng(seed)
    A = ((g.random((128, 256)) > 0.5) * 2 - 1) / np.sqrt(128)
    x = sv.beamspace_channels(NumpyMatlabRNG(seed), B, "ULA" if seed % 2 == 0 else "UPA")
    y = measurements.gen_measurements_matlab(x, A, 5.0 + 5 * (seed % 3), NumpyMatlabRNG(seed + 9))
    return A, x, y


def _run_ref(**kw):
    wd = tmpdir("bsce_tf_")
    np.savez(wd / "in.npz", **kw)
    subprocess.run([sys.executable, str(RUNNER), str(wd / "in.npz"), str(wd / "out.npz")], check=True,
                   capture_output=True)
    return dict(np.load(wd / "out.npz", allow_pickle=True))


def _perturbed(model, seed):
    g = np.random.default_rng(100 + seed)
    with torch.no_grad():   # generic weights: perturb the original initialisation
        for n in model.param_names():
            p = model.P(n)
            p.add_(torch.as_tensor(g.standard_normal(p.shape) * (0.02 if n.startswith("B") else 0.1)))
    return {f"w_{n}": model.P(n).detach().numpy().copy() for n in model.param_names()}


@pytest.mark.parametrize("shrink", ["soft", "gm"])
@pytest.mark.parametrize("seed", [0, 1, 2, 3])
def test_lamp_forward_T8(shrink, seed, record):
    """All 16 stage outputs of the full T=8 network (graph built by build_LAMP)."""
    A, x, y = _problem(seed)
    model = LAMP(A, T=8, shrink=shrink, untied=True, K=4, precision="float64")
    weights = _perturbed(model, seed)
    ref = _run_ref(mode="forward", A=A, y=y, x=x, shrink=shrink, K=4, T=8, lr=1e-3,
                   refinements=np.array((0.5, 0.1, 0.01)), nsteps=0, test_stages=np.array([], int), **weights)
    with torch.no_grad():
        outs = model.forward_stages(torch.as_tensor(y))
    assert len(outs) == len([k for k in ref if k.startswith("out_")])
    fwd = max(compare(o.numpy(), ref[f"out_{i}"], f"out {i}", TOL_FWD)["rel"] for i, o in enumerate(outs))
    record("lamp/vs_tf_forward", shrink=shrink, seed=seed, T=8, n_outputs=len(outs), fwd_rel=fwd)


@pytest.mark.parametrize("shrink,T", [("soft", 4), ("gm", 3)])
@pytest.mark.parametrize("seed", [0, 1, 2])
def test_lamp_training_graph(shrink, T, seed, record):
    """Original setup_training(): stage list, gradients and Adam updates of EVERY stage."""
    A, x, y = _problem(seed)
    K, lr, refs, nsteps = 4, 1e-3, (0.5, 0.1, 0.01), 5
    model = LAMP(A, T=T, shrink=shrink, untied=True, K=K, precision="float64")
    weights = _perturbed(model, seed)
    cfg = TrainConfig(lr=lr, refinements=refs, loss="nmse")
    stages = build_stages(model, cfg)
    test_stages = list(range(len(stages)))
    ref = _run_ref(mode="train", A=A, y=y, x=x, shrink=shrink, K=K, T=T, lr=lr, refinements=np.array(refs),
                   nsteps=nsteps, test_stages=np.array(test_stages), **weights)
    assert [s["name"] for s in stages] == list(ref["stage_names"]), "stage list differs from original"
    assert [",".join(s["params"]) for s in stages] == list(ref["stage_vars"]), "stage variable lists differ"
    # forward
    yt, xt = torch.as_tensor(y), torch.as_tensor(x)
    with torch.no_grad():
        outs = model.forward_stages(yt)
    fwd = max(compare(o.numpy(), ref[f"out_{i}"], f"out {i}", TOL_FWD)["rel"] for i, o in enumerate(outs))
    # gradients and Adam steps
    gmax, amax = 0.0, 0.0
    for si in test_stages:
        st = stages[si]
        model.load_mat_dict({n: weights[f"w_{n}"] for n in model.param_names()})
        model.zero_grad(set_to_none=True)
        loss = nmse_loss(model.forward_stages(yt, upto=st["out"])[-1], xt)
        loss.backward()
        for n in model.param_names():
            key = f"grad_{si}_{n}"
            if key in ref:
                gmax = max(gmax, compare(model.P(n).grad.numpy(), ref[key], f"grad {st['name']} {n}", TOL_GRAD)["rel"])
        params = [model.P(n) for n in st["params"]]
        opt = AdamTF1(params, lr=st["lr"])
        for _ in range(nsteps):
            opt.zero_grad(set_to_none=True)
            nmse_loss(model.forward_stages(yt, upto=st["out"])[-1], xt).backward()
            opt.step()
        for n in st["params"]:   # compare the parameter *updates* (after - before)
            w0 = weights[f"w_{n}"]
            amax = max(amax, compare(model.P(n).detach().numpy() - w0, ref[f"after_{si}_{n}"] - w0,
                                     f"Adam update {st['name']} {n}", TOL_ADAM)["rel"])
    record("lamp/vs_tf", shrink=shrink, seed=seed, n_stages=len(stages), tested_stages=len(test_stages),
           fwd_rel=fwd, grad_rel=gmax, adam_rel=amax, adam_steps=nsteps)

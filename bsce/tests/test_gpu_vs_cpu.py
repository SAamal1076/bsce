"""PyTorch on the GPU vs PyTorch on the CPU, for every learned model.

The rest of the suite verifies PyTorch-on-CPU against the ORIGINAL code (MATLAB / TF / THADMM-Net,
worst float64 error 1.7e-13). This file closes the chain on the GPU machine: the same models,
weights, inputs and minibatches are run on the CPU and on the GPU in one process, so
|GPU - original| <= |GPU - CPU| + |CPU - original| is bounded by the numbers below.

Checked per model, in float64 (exact up to round-off) and float32 (round-off level):
  * every stage output of the network,
  * the gradient of every stage loss w.r.t. every parameter,
  * 5 TF-1 Adam steps on every stage (parameter updates),
  * a short layer-wise training run with identical minibatches: iteration count (early stop)
    and validation NMSE of every stage, final parameters,
  * the exported .pt / .mat of the GPU-trained model reloaded on the CPU.

  python -m pytest tests/test_gpu_vs_cpu.py -q          # on the GPU machine (skips without CUDA)
  BSCE_GPU_DEVICE=cpu python -m pytest tests/test_gpu_vs_cpu.py -q   # dry run of the code path
No Octave / TensorFlow needed. Results go to tests/results/equivalence.jsonl (test 'gpu/vs_cpu');
`python scripts/summarize_tests.py` adds them to the summary table.
"""
import copy
import os

import numpy as np
import pytest
import torch

from bsce.data import sv, measurements
from bsce.device import get_device, describe
from bsce.export import save_model, load_model
from bsce.models import build_model, alista_W
from bsce.rng import NumpyMatlabRNG
from bsce.train.losses import nmse_loss
from bsce.train.optim import make_optimizer
from bsce.train.trainer import Trainer, TrainConfig
from harness.octave import tmpdir

_REQ = os.environ.get("BSCE_GPU_DEVICE", "cuda")
if _REQ.startswith("cuda") and not torch.cuda.is_available():
    pytest.skip("CUDA not available (set BSCE_GPU_DEVICE=cpu for a dry run)", allow_module_level=True)
DEV = get_device(_REQ)          # also disables TF32 on CUDA

BORG = dict(schedule="borgerding", loss="nmse", val_metric="nmse", val_round=5)
ISTA = dict(schedule="alista", loss="l2", val_metric="nmse_l2", val_round=None, lr_decay=(0.2, 0.02))
CASES = {  # name: (model cfg, training cfg, exact training expected)
    "LAMP": (dict(model="LAMP", T=3), dict(BORG, warm_start=True), True),
    "GM-LAMP": (dict(model="GM-LAMP", T=2, K=4), dict(BORG, warm_start=True), False),  # round-off driven (report §7)
    "GM-LAMP-spread": (dict(model="GM-LAMP", T=2, K=4, gm_init="spread"), dict(BORG, warm_start=True), True),
    "LADMM-full": (dict(model="LADMM", T=3, operator="full"), dict(BORG), True),
    "LADMM-diag": (dict(model="LADMM", T=3, operator="diag"), dict(BORG), True),
    "LADMM-lowrank": (dict(model="LADMM", T=3, operator="lowrank", rank=8), dict(BORG), True),
    "LADMM-kron": (dict(model="LADMM", T=3, operator="kron"), dict(BORG), True),
    "LADMM-psd": (dict(model="LADMM", T=3, operator="psd"), dict(BORG), True),
    "LADMM-kronprior": (dict(model="LADMM", T=3, operator="kronprior"), dict(BORG), True),
    "LADMM-diagprior": (dict(model="LADMM", T=3, operator="diagprior"), dict(BORG), True),
    "LISTA": (dict(model="LISTA", T=4), dict(ISTA, check_mult=10), True),
    "ALISTA": (dict(model="ALISTA", T=4, percent=1.2, max_percent=13.0), dict(ISTA, check_mult=10), True),
    "LVAMP": (dict(model="LVAMP", T=3), dict(BORG, loss="l2", val_metric="nmse_l2", val_round=None, refine_name="value"), True),
    "ADMM-Net": (dict(model="ADMM-Net", T=4), dict(schedule="end2end", epochs=2, optimizer="adam_torch", val_round=None), True),
}
TOL = {"float64": dict(fwd=1e-12, grad=1e-10, adam=1e-9, train=1e-8, export=1e-12),
       "float32": dict(fwd=1e-4, grad=1e-2, adam=5e-2, train=None, export=1e-4)}


def npy(t):
    return t.detach().cpu().numpy()


def rel(a, b):
    """Relative error computed in float64; non-finite values give inf (so a test can never pass
    on NaN), an all-zero reference gives the absolute error."""
    a = np.asarray(a).astype(np.complex128 if np.iscomplexobj(a) or np.iscomplexobj(b) else np.float64).ravel()
    b = np.asarray(b).astype(a.dtype).ravel()
    if not (np.all(np.isfinite(a)) and np.all(np.isfinite(b))):
        return float("inf")
    nb = np.linalg.norm(b)
    return float(np.linalg.norm(a - b) / nb) if nb > 0 else float(np.linalg.norm(a - b))


def rel_group(da, db):
    """Relative error of a group of arrays (e.g. all gradients of one stage) taken together."""
    keys = sorted(db)
    return rel(np.concatenate([np.asarray(da[k]).ravel() for k in keys]),
               np.concatenate([np.asarray(db[k]).ravel() for k in keys]))


@pytest.fixture(scope="module")
def data():
    g = np.random.default_rng(0)
    A = ((g.random((128, 256)) > 0.5) * 2 - 1) / np.sqrt(128)
    xt = sv.beamspace_channels(NumpyMatlabRNG(0), 300, "UPA")
    xv = sv.beamspace_channels(NumpyMatlabRNG(1), 80, "UPA")
    yt = measurements.gen_measurements_matlab(xt, A, 10.0, NumpyMatlabRNG(2))
    yv = measurements.gen_measurements_matlab(xv, A, 10.0, NumpyMatlabRNG(3))
    return A, xt, yt, xv, yv, alista_W(A)


def _build(name, precision, data):
    A, *_, W = data
    cfg = dict(CASES[name][0], precision=precision)
    if cfg["model"] == "ALISTA":
        cfg["W"] = W
    if cfg["model"] == "LADMM" and cfg["operator"] in ("full", "diag", "lowrank", "kron"):
        cfg["init_precision"] = precision
    torch.manual_seed(0)
    m = build_model(cfg, A)
    g = np.random.default_rng(11)
    with torch.no_grad():   # generic weights: perturb the initial values
        for n in m.param_names():
            p = m.P(n)
            s = 0.02 * max(float(p.detach().abs().max()), 1e-3)
            noise = g.standard_normal(tuple(p.shape)) + (1j * g.standard_normal(tuple(p.shape)) if p.is_complex() else 0)
            p.add_(torch.as_tensor(noise * s, dtype=p.dtype))
    return m


@pytest.mark.parametrize("precision", ["float64", "float32"])
@pytest.mark.parametrize("name", list(CASES))
def test_gpu_vs_cpu(name, precision, data, record):
    A, xt, yt, xv, yv, _ = data
    tol = TOL[precision]
    cpu = _build(name, precision, data)
    gpu = copy.deepcopy(cpu).to(DEV)
    w_cpu = copy.deepcopy(cpu.state_dict())
    w_gpu = copy.deepcopy(gpu.state_dict())
    y = torch.as_tensor(yt[:, :64])
    x = torch.as_tensor(xt[:, :64])

    # 1. forward: every stage output
    with torch.no_grad():
        oc, og = cpu.forward_stages(y), gpu.forward_stages(y)
    fwd = max(rel(npy(b), npy(a)) for a, b in zip(oc, og))

    # 2./3. gradients of every stage loss and 5 Adam steps per stage
    grad = adam = 0.0
    for st in cpu.stage_plan():
        names = list(st.all) or cpu.param_names()
        res = {}
        for m, w in ((cpu, w_cpu), (gpu, w_gpu)):
            m.load_state_dict(w)
            xd = x.to(m.A.device, m.cdtype)
            m.zero_grad(set_to_none=True)
            nmse_loss(m.forward_stages(y, upto=st.index)[-1], xd).backward()
            gr = {n: npy(m.P(n).grad) for n in names if m.P(n).grad is not None}
            opt = make_optimizer("adam_tf1", [m.P(n) for n in names], 1e-3)
            for _ in range(5):
                opt.zero_grad(set_to_none=True)
                nmse_loss(m.forward_stages(y, upto=st.index)[-1], xd).backward()
                opt.step()
            upd = {n: npy(m.P(n)) - npy(w[n]) for n in names}
            res[m is gpu] = (gr, upd)
        (gc, uc), (gg, ug) = res[False], res[True]
        assert gc.keys() == gg.keys()
        grad = max(grad, rel_group(gg, gc))
        adam = max(adam, rel_group(ug, uc))
    cpu.load_state_dict(w_cpu)
    gpu.load_state_dict(w_gpu)

    # 4. short training run, identical minibatches on both devices
    mcfg, tkw, exact = CASES[name]
    rs = np.random.RandomState(5)
    batches = [rs.choice(xt.shape[1], size=64) for _ in range(4000)]
    states, models = [], []
    for m in (cpu, gpu):
        it = iter(batches)
        cfg = TrainConfig(**{**dict(lr=1e-3, maxit=30, better_wait=5, ivl=1, check_mult=10, batch_size=64,
                                    seed=3, verbose=False, optimizer="adam_tf1"), **tkw})
        states.append(Trainer(m, cfg, xt, yt, xv, yv, sampler=lambda: next(it)).run())
        models.append(m)
    lc, lg = states[0]["log"], states[1]["log"]
    assert [s["stage"] for s in lc] == [s["stage"] for s in lg]
    iters_equal = all(a.get("iters") == b.get("iters") for a, b in zip(lc, lg))
    dnmse = max(abs(a["nmse_db"] - b["nmse_db"]) if np.isfinite(a["nmse_db"] - b["nmse_db"]) else float("inf")
                for a, b in zip(lc, lg))
    tparam = rel_group({n: npy(gpu.P(n)) for n in cpu.param_names()}, {n: npy(cpu.P(n)) for n in cpu.param_names()})

    # 5. export of the GPU-trained model, reloaded on the CPU (.pt and .mat)
    wd = tmpdir("bsce_gpu_")
    pt, mat = save_model(gpu, wd / "net")
    with torch.no_grad():
        ref = npy(gpu(y))
        exp_pt = rel(npy(load_model(pt, device="cpu")(y)), ref)
        exp_mat = rel(npy(load_model(mat, device="cpu")(y)), ref)

    record("gpu/vs_cpu", model=name, precision=precision, device=describe(DEV), fwd_rel=fwd, grad_rel=grad,
           adam_rel=adam, train_param_rel=tparam, train_dnmse_db=dnmse, train_iters_equal=iters_equal,
           export_pt_rel=exp_pt, export_mat_rel=exp_mat, n_stages=len(lc))
    assert fwd < tol["fwd"], f"forward {fwd:.2e}"
    assert grad < tol["grad"], f"gradients {grad:.2e}"
    assert adam < tol["adam"], f"Adam updates {adam:.2e}"
    # float32: the reloaded model runs on the CPU, the reference on the GPU -> single-precision round-off
    assert exp_pt < tol["export"] and exp_mat < tol["export"]
    if precision == "float64" and exact:
        assert iters_equal, "early-stop decisions differ"
        assert dnmse < 1e-6 and tparam < tol["train"], f"training: dNMSE {dnmse:.1e} dB, params {tparam:.1e}"
    else:   # float32, or GM-LAMP from identical components: statistical agreement only
        assert dnmse < 0.1, f"training: validation NMSE differs by {dnmse:.3f} dB"


@pytest.mark.parametrize("precision", ["float64", "float32"])
def test_gmlamp_original_init_on_this_device(precision, data, record):
    """v31.1: GM-LAMP from its original init (identical mixture components) on this device.

    Records whether the K components get identical gradients here (on the CPU they always do; that was the cause of the
    bug fixed in v31.1), and checks that the trainer separates them (train.gm_symmetry='break', the default)."""
    A, xt, yt, xv, yv, _ = data
    torch.manual_seed(0)
    m = build_model(dict(model="GM-LAMP", T=2, K=4, precision=precision), A).to(DEV)
    y = torch.as_tensor(yt[:, :64]).to(DEV)
    nmse_loss(m(y), torch.as_tensor(xt[:, :64]).to(DEV, m.cdtype)).backward()
    g = m.P("theta_1").grad
    same = bool(torch.all(g == g[:, :1]))
    gap = float((g - g[:, :1]).abs().max())
    rs = np.random.RandomState(5)
    cfg = TrainConfig(lr=1e-3, maxit=60, better_wait=1000, ivl=10, check_mult=5, batch_size=64, refinements=[],
                      warm_start=True, verbose=False, optimizer="adam_tf1")
    torch.manual_seed(0)
    m2 = build_model(dict(model="GM-LAMP", T=1, K=4, precision=precision), A).to(DEV)
    st = Trainer(m2, cfg, xt, yt, xv, yv, sampler=lambda: rs.choice(xt.shape[1], size=64)).run()
    record("gpu/gmlamp_original_init", precision=precision, device=describe(DEV), identical_gradients=same,
           max_gradient_gap=gap, broken=st.get("gm_symmetry_broken", []))
    assert st.get("gm_symmetry_broken") == ["theta_0"]
    assert m2.identical_gm_components() == []

"""End-to-end training-loop reproduction: bsce Trainer vs the ORIGINAL do_training().

The original training scripts (GM-LAMP's define_networks.py and the LADMM scripts) are run
unchanged on small data with small maxit / better_wait; the minibatch indices they draw
with np.random.choice are recorded and replayed into bsce.train.Trainer, and the parameters
they save after every stage are kept. Everything else (stage list, a fresh Adam per stage,
validation every iv1 steps before the step, round(nmse, 5), early stop every 100*iv1 steps,
GM-LAMP's warm start of each new layer from the previous one) is the PyTorch code. Compared
after EVERY stage: number of iterations (early-stop decisions), the logged validation NMSE,
and all parameters.

GM-LAMP with the original initialisation is special: all K mixture components start
identical, so the gradient that separates them is exactly zero in exact arithmetic and the
components split only through round-off, amplified exponentially (e-folding in a few Adam
steps). Both implementations agree to machine precision in every parameter until the
identical-component layer-1 mixture is trained through layer 2 (stage 'linear T=2
trainrate=0': its logits then get round-off-only gradients that do not change the output),
have identical iteration counts and NMSE until the warm-started layer 2 mixture is trained
(stage 'non-linear T=2'), and agree only statistically afterwards. The same test with a
slightly spread initial value (both codes given the same value) is the control experiment.
Set BSCE_REF_CACHE=<dir> to reuse reference runs (off by default: the originals are re-run).
"""
import hashlib
import os
import re
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from harness.octave import tf_available, tmpdir
from bsce.data import sv, measurements
from bsce.rng import NumpyMatlabRNG
from bsce.models.lamp import LAMP, gm_theta_init
from bsce.models.ladmm import LADMM
from bsce.train.trainer import Trainer, TrainConfig

HERE = Path(__file__).resolve().parent
RUNNER = HERE / "harness" / "ref_tf_dotraining.py"
pytestmark = pytest.mark.skipif(not tf_available(), reason="tensorflow not installed (requirements-verify.txt)")
USER = HERE.parent / "reference" / "ladmm_user"
LOG_RE = re.compile(r"\n(.+?) nmse=(-?[\d.]+|nan|-?inf) dB in (\d+) iterations")
REFS = (0.5, 0.1, 0.01)


def _data(seed, array, n_train=600, n_val=100, snr=10.0, A_dtype=np.float64):
    g = np.random.default_rng(seed)
    A = ((g.random((128, 256)) > 0.5) * 2 - 1) / np.sqrt(128)
    xt = sv.beamspace_channels(NumpyMatlabRNG(seed), n_train, array)
    xv = sv.beamspace_channels(NumpyMatlabRNG(seed + 50), n_val, array)
    yt = measurements.gen_measurements_matlab(xt, A, snr, NumpyMatlabRNG(seed + 7))
    yv = measurements.gen_measurements_matlab(xv, A, snr, NumpyMatlabRNG(seed + 8))
    return A.astype(A_dtype), xt, yt, xv, yv


def _run_ref(**kw):
    """Run the original loop in a subprocess. Optional cache (BSCE_REF_CACHE=<dir>, off by
    default): key = hash of the inputs, the runner and the original script."""
    wd = tmpdir("bsce_dotrain_")
    np.savez(wd / "in.npz", **kw)
    cache = os.environ.get("BSCE_REF_CACHE")
    key = None
    if cache:
        h = hashlib.sha1((wd / "in.npz").read_bytes() + RUNNER.read_bytes())
        h.update(Path(str(kw.get("script", RUNNER))).read_bytes())
        key = Path(cache) / f"dotrain_{h.hexdigest()}.npz"
    if key is not None and key.exists():
        shutil.copy(key, wd / "out.npz")
    else:
        r = subprocess.run([sys.executable, str(RUNNER), str(wd / "in.npz"), str(wd / "out.npz")],
                           capture_output=True, text=True, encoding="utf-8", errors="replace")
        (wd / "stdout.txt").write_text(r.stdout + "\n" + r.stderr, encoding="utf-8")
        assert r.returncode == 0, r.stderr[-3000:]
        if key is not None:
            key.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy(wd / "out.npz", key)
    ref = dict(np.load(wd / "out.npz", allow_pickle=True))
    ref["stages"] = [(m.group(1), float(m.group(2)), int(m.group(3))) for m in LOG_RE.finditer(str(ref["log"]))]
    return ref


def _rel(a, b):
    return float(np.linalg.norm(a - b) / max(np.linalg.norm(b), 1e-300))


def _train(model, ref, data, cycle=1, **cfg_kw):
    """bsce Trainer fed with the recorded minibatches; returns state, #batches used, snapshots."""
    _, xt, yt, xv, yv = data
    batches = iter(list(ref["idx"]) * cycle)
    used, snaps = [0], []

    def sampler():
        used[0] += 1
        return next(batches)

    def cb(st, m):
        snaps.append({n: m.P(n).detach().cpu().numpy().copy() for n in m.param_names()})

    cfg = TrainConfig(schedule="borgerding", loss="nmse", val_metric="nmse", val_round=5,
                      optimizer="adam_tf1", check_mult=100, verbose=False, refinements=REFS, **cfg_kw)
    state = Trainer(model, cfg, xt, yt, xv, yv, sampler=sampler, stage_callback=cb).run()
    return state, used[0], snaps


def _compare(model, ref, state, used, snaps):
    names = [s[0] for s in ref["stages"]]
    assert [s["stage"] for s in state["log"]] == names, "stage sequence differs"
    assert len(snaps) == int(ref["n_snapshots"]) - 1   # the original saves once more at the very end
    it_eq = [s["iters"] == r[2] for s, r in zip(state["log"], ref["stages"])]
    dn = [abs(s["nmse_db"] - r[1]) for s, r in zip(state["log"], ref["stages"])]
    worst = [max((_rel(sn[n].reshape(ref[f"snap{k}_{n}"].shape), ref[f"snap{k}_{n}"]), n) for n in sn)
             for k, sn in enumerate(snaps)]
    prel = [w[0] for w in worst]
    return dict(n_stages=len(names), total_steps=int(ref["idx"].shape[0]), steps_used=used,
                stage_names=names, stage_iters=[r[2] for r in ref["stages"]], stage_iters_equal=it_eq,
                stage_dnmse_db=dn, stage_param_rel=prel, stage_worst_param=[w[1] for w in worst],
                final_nmse_db_tf=ref["stages"][-1][1], final_nmse_db_torch=state["log"][-1]["nmse_db"])


def _init_rel(model, ref):
    return max(_rel(model.P(n).detach().cpu().numpy().reshape(ref[f"init_{n}"].shape), ref[f"init_{n}"])
               for n in model.param_names())


def _assert_exact(res, upto=None, ptol=1e-9, param_upto=None):
    n = res["n_stages"] if upto is None else upto
    npar = n if param_upto is None else param_upto
    assert all(res["stage_iters_equal"][:n]), "early-stop iteration counts differ"
    assert max(res["stage_dnmse_db"][:n]) < 2e-6       # the log prints the NMSE with 6 decimals
    assert max(res["stage_param_rel"][:npar]) < ptol


# ------------------------------------------------------------------------------------
LAMP_CASES = [  # id, shrink, gm_init, T, array, iv1, maxit, better_wait, seed
    ("GM-LAMP original init", "gm", "original", 2, "ULA", 1, 200, 30, 0),
    ("GM-LAMP spread init", "gm", "spread", 2, "UPA", 2, 300, 40, 1),
    ("LAMP", "soft", None, 3, "ULA", 1, 200, 30, 2),
]


@pytest.mark.parametrize("case,shrink,gm_init,T,array,iv1,maxit,bw,seed", LAMP_CASES)
def test_gmlamp_do_training(case, shrink, gm_init, T, array, iv1, maxit, bw, seed, record):
    """Train_GM-LAMP.py / Train_LAMP.py loop (define_networks.do_training, float64, batch 128)."""
    data = _data(seed, array)
    A, lr = data[0], 1e-3
    extra = {}
    if shrink == "gm" and gm_init != "original":
        extra["gm_theta"] = gm_theta_init(4, gm_init)
    ref = _run_ref(net="lamp", shrink=shrink, K=4, T=T, A=A, xt=data[1], yt=data[2], xv=data[3], yv=data[4],
                   lr=lr, refinements=np.array(REFS), iv1=iv1, maxit=maxit, better_wait=bw, seed=1000 + seed, **extra)
    model = LAMP(A, T=T, shrink=shrink, untied=True, K=4, precision="float64", gm_init=gm_init or "original")
    irel = _init_rel(model, ref)
    state, used, snaps = _train(model, ref, data, lr=lr, maxit=maxit, better_wait=bw, ivl=iv1,
                                warm_start=True, batch_size=128, cycle=2, gm_symmetry="keep")  # v31: the original arithmetic
    res = _compare(model, ref, state, used, snaps)
    names = res["stage_names"]
    if case == "GM-LAMP original init":
        # parameters: bit-level agreement until the identical-component layer-1 mixture is trained
        # through layer 2 (its logits then receive round-off-only gradients); iteration counts and
        # NMSE: until the warm-started identical-component layer 2 is trained (see docstring)
        n_par, n_exact = names.index("LAMP-gm linear T=2 trainrate=0"), names.index("LAMP-gm non-linear T=2")
    else:
        n_par = n_exact = res["n_stages"]
    record("train/do_training", case=case, script="define_networks.py", T=T, array=array, iv1=iv1, maxit=maxit,
           better_wait=bw, seed=seed, precision="float64", init_rel=irel, n_exact_stages=n_exact,
           n_param_exact_stages=n_par, **res)
    assert irel < 1e-15
    _assert_exact(res, upto=n_exact, param_upto=n_par)
    if n_exact == res["n_stages"]:
        assert res["steps_used"] == res["total_steps"]


LADMM_CASES = [  # script, operator, kwargs, T, iv1, maxit, better_wait, seed
    ("A_LADMM_UPA_SV_0_20dB.py", "full", {"shrink": "soft", "untied": True}, 2, 1, 200, 30, 0),
    ("C_UPA_DeepMIMO_0_10dB_Kronecker.py", "kron", {"N1": 16, "N2": 16, "shrink": "soft", "untied": True}, 2, 1, 200, 30, 1),
    ("A_UPA_DeepMIMO_0_10dB_DiagonalScaling.py", "diag", {"shrink": "soft", "untied": True}, 2, 1, 200, 30, 2),
]


@pytest.mark.parametrize("script,operator,kw,T,iv1,maxit,bw,seed", LADMM_CASES)
def test_ladmm_do_training(script, operator, kw, T, iv1, maxit, bw, seed, record):
    """float64 LADMM scripts: build_LADMM + setup_training + do_training (batch 256)."""
    data = _data(seed, "UPA")
    A, lr = data[0], 1e-3
    ref = _run_ref(net="ladmm", script=str(USER / script), kwargs=np.array(kw, dtype=object), T=T, A=A,
                   xt=data[1], yt=data[2], xv=data[3], yv=data[4], lr=lr, refinements=np.array(REFS),
                   iv1=iv1, maxit=maxit, better_wait=bw, seed=2000 + seed)
    model = LADMM(A, T=T, operator=operator, rank=kw.get("rank", 16), precision="float64", init_precision="float64")
    irel = _init_rel(model, ref)
    state, used, snaps = _train(model, ref, data, lr=lr, maxit=maxit, better_wait=bw, ivl=iv1,
                                warm_start=False, batch_size=256)
    res = _compare(model, ref, state, used, snaps)
    record("train/do_training", case="LADMM-" + operator, script=script, T=T, array="UPA", iv1=iv1, maxit=maxit,
           better_wait=bw, seed=seed, precision="float64", init_rel=irel, n_exact_stages=res["n_stages"],
           n_param_exact_stages=res["n_stages"], **res)
    assert irel < 1e-12
    assert res["steps_used"] == res["total_steps"]
    _assert_exact(res, ptol=1e-8)


@pytest.mark.parametrize("seed", [0])
def test_ladmm_train_float32(seed, record):
    """LADMM_train.py itself (float32/complex64, batch_size argument): statistical agreement.

    In single precision the summation order of TF and PyTorch kernels differs (~1e-7 per op),
    so trajectories drift; the minibatches are identical and the stage-wise validation NMSE
    must agree to a small fraction of a dB."""
    data = _data(seed, "ULA", A_dtype=np.float32)
    A, lr, T, maxit, bw, bs = data[0], 1e-3, 2, 200, 30, 256
    ref = _run_ref(net="ladmm", script=str(USER / "LADMM_train.py"),
                   kwargs=np.array({"shrink": "soft", "untied": True}, dtype=object),
                   T=T, A=A, xt=data[1], yt=data[2], xv=data[3], yv=data[4], lr=lr, refinements=np.array(REFS),
                   iv1=1, maxit=maxit, better_wait=bw, batch_size=bs, seed=3000 + seed)
    model = LADMM(A, T=T, operator="full", precision="float32", init_precision="float32")
    irel = _init_rel(model, ref)
    # an early-stop decision may differ -> replay cyclically so the recorded batches never run out
    state, used, snaps = _train(model, ref, data, lr=lr, maxit=maxit, better_wait=bw, ivl=1,
                                warm_start=False, batch_size=bs, cycle=3)
    res = _compare(model, ref, state, used, snaps)
    record("train/do_training_f32", case="LADMM-full float32", script="LADMM_train.py", T=T, seed=seed,
           precision="float32", init_rel=irel, **res)
    assert irel < 1e-6
    assert max(res["stage_dnmse_db"]) < 0.05

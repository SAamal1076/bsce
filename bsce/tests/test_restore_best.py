"""Options beyond the original code: restore_best and grouped validation (val_groups).

* restore_best: after every stage the weights are those of the best validation value; validating
  the returned model reproduces the logged best NMSE, and it is never worse than the last value.
* val_groups: validation value = geometric mean of the per-group NMSE (= mean in dB).
* defaults (both off) are covered by test_golden.py / test_training_loop_vs_tf.py (unchanged numbers).
"""
import math

import numpy as np
import pytest
import torch

from bsce.data import sv, measurements
from bsce.models import build_model
from bsce.rng import NumpyMatlabRNG
from bsce.train.trainer import Trainer, TrainConfig


@pytest.fixture(scope="module")
def data():
    g = np.random.default_rng(0)
    A = ((g.random((128, 256)) > 0.5) * 2 - 1) / np.sqrt(128)
    xt = sv.beamspace_channels(NumpyMatlabRNG(0), 400, "ULA")
    xv = sv.beamspace_channels(NumpyMatlabRNG(1), 100, "ULA")
    yt = measurements.gen_measurements_py(xt, A, [0.0, 5.0, 10.0, 2.0], np.random.RandomState(2))
    yv = measurements.gen_measurements_py(xv, A, [0.0, 2.5, 5.0, 7.5, 10.0], np.random.RandomState(3))
    return A, xt, yt, xv, yv


@pytest.mark.parametrize("model", ["LAMP", "LADMM"])
def test_restore_best(model, data, record):
    A, xt, yt, xv, yv = data
    torch.manual_seed(0)
    m = build_model(dict(model=model, T=2, precision="float64", operator="full", init_precision="float64"), A)
    cfg = TrainConfig(lr=3e-3, maxit=60, better_wait=1000, ivl=5, check_mult=2, batch_size=32, val_round=None,
                      restore_best=True, verbose=False)
    tr = Trainer(m, cfg, xt, yt, xv, yv)
    st = tr.run()
    last = st["log"][-1]
    v_now = 10 * math.log10(tr.validate(m.stage_plan()[-1].index))
    record("train/restore_best", model=model, logged_best_db=last["nmse_db"], revalidated_db=v_now)
    assert abs(v_now - last["nmse_db"]) < 1e-9
    assert last["nmse_db"] <= 10 * math.log10(last["nmse_hist"][-1]) + 1e-12      # best <= last value


def test_val_groups(data):
    A, xt, yt, xv, yv = data
    torch.manual_seed(0)
    m = build_model(dict(model="LAMP", T=2, precision="float64"), A)
    tr = Trainer(m, TrainConfig(val_groups=5, val_round=None, verbose=False), xt, yt, xv, yv)
    out = m.stage_plan()[-1].index
    v = tr.validate(out)
    with torch.no_grad():
        xh = m.forward_stages(torch.as_tensor(yv), upto=out)[-1].numpy()
    per = [np.mean(np.sum(np.abs(xh[:, i * 20:(i + 1) * 20] - xv[:, i * 20:(i + 1) * 20]) ** 2, 0))
           / np.mean(np.sum(np.abs(xv[:, i * 20:(i + 1) * 20]) ** 2, 0)) for i in range(5)]
    assert abs(10 * math.log10(v) - np.mean(10 * np.log10(per))) < 1e-10

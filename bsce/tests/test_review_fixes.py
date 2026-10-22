"""Regression tests for the code review of 15 Oct 2026 (docs/changes/CHANGES_2026-10-15_review_fixes.md).

1. resuming a layer-wise run from its checkpoint gives exactly the weights of an uninterrupted run (random state saved)
2. every common protocol trains every method with the same training settings (grad_clip included)
3. n_params counts a complex parameter as two real ones
Synthetic data only (no reference/ or data/ needed).
"""
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from bsce.config import load_yaml, merge  # noqa: E402
from bsce.models import build_model  # noqa: E402
from bsce.train.trainer import Trainer, TrainConfig  # noqa: E402

rng = np.random.default_rng(0)
M, N = 32, 64
A = rng.choice([-1.0, 1.0], (M, N)) / np.sqrt(M)


def _data(K, seed):
    r = np.random.default_rng(seed)
    x = np.zeros((N, K), complex)
    for k in range(K):
        idx = r.choice(N, 3, replace=False)
        x[idx, k] = (r.standard_normal(3) + 1j * r.standard_normal(3)) / np.sqrt(2) * np.sqrt(N / 3)
    n = (r.standard_normal((N, K)) + 1j * r.standard_normal((N, K))) * np.sqrt(0.05 / 2)
    return x, A @ (x + n)


class _Stop(Exception):
    pass


def test_resume_is_exact(tmp_path):
    cfg = merge(load_yaml(ROOT / "configs/train/ladmm_original.yaml"), load_yaml(ROOT / "configs/train/ladmm_best_protocol.yaml"))
    cfg["train"].update(maxit=20, better_wait=10, verbose=False, batch_size=32)
    mc = dict(cfg["model"], T=2, array="ULA", N1=N, N2=1, init_precision=None)
    xt, yt = _data(400, 1)
    xv, yv = _data(100, 2)

    def run(ck, stop_at=None):
        torch.manual_seed(0)
        net = build_model(mc, A)
        n = [0]

        def cb(st, model):
            n[0] += 1
            if stop_at and n[0] == stop_at:
                raise _Stop

        try:
            Trainer(net, TrainConfig.from_dict(cfg["train"]), xt, yt, xv, yv, checkpoint=str(ck), stage_callback=cb).run()
        except _Stop:
            return None
        return net

    a = run(tmp_path / "a" / "run")
    run(tmp_path / "b" / "run", stop_at=3)
    b = run(tmp_path / "b" / "run")
    sa, sb = a.state_dict(), b.state_dict()
    diff = max(float((sa[k] - sb[k]).abs().max()) for k in sa)
    print(f"max |uninterrupted - resumed| = {diff}")
    assert diff == 0.0


@pytest.mark.parametrize("protocol", ["ladmm_best", "gmlamp", "gmlamp_best", "fair", "e2e"])
def test_protocol_settings_are_the_same_for_every_method(protocol):
    methods = ["ladmm_original", "lamp_original", "gmlamp_original", "lista", "alista", "lvamp", "admmnet",
               "nfladmm_s", "nfladmm_c", "nfladmm_sr"]
    keys = ["schedule", "lr", "batch_size", "loss", "noise", "grad_clip", "restore_best", "warm_start", "optimizer"]
    rows = {}
    for m in methods:
        c = merge(load_yaml(ROOT / f"configs/train/{m}.yaml"), load_yaml(ROOT / f"configs/train/{protocol}_protocol.yaml"))
        rows[m] = tuple(c["train"].get(k) for k in keys)
    distinct = set(rows.values())
    assert len(distinct) == 1, {m: dict(zip(keys, v)) for m, v in rows.items()}


def test_n_params_counts_complex_twice():
    m = build_model(dict(model="ADMM-Net", T=2), A)
    real = sum(p.numel() * (2 if p.is_complex() else 1) for p in m.parameters())
    assert m.n_params() == real and any(p.is_complex() for p in m.parameters())

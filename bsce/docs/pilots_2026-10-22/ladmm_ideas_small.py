"""Small experiment: LADMM before / after noise_level='residual' (+ adaptive threshold) and debias='ls'.

SV UPA (GM-LAMP setting: N = 256, M = 128, L = 3, GM-LAMP's CS matrix), y = A (x + n).
  training   n_train new channels (default 500), SNR ~ U(0, 20) dB in 100 chunks
  validation 500 new channels, SNR ~ U(0, 20) dB
  test       the standard 2000 SV_UPA test channels with evaluate.py's test noise (seed 2026 + 1000 + i)
Every model: the same data, the same layer-wise protocol and budget, seeds 0, 1, 2 for the minibatches.
usage: python ladmm_ideas_small.py <bsce folder> <T> <out.json> [n_train (500)] [maxit per stage (5000)]
       [<folder>/<prefix> of a data set, e.g. .../3GPP_UPA/3GPP_UPA: its training / validation / test channels
        instead of new SV channels and the SV_UPA test set]
"""
import json
import sys
import time
from pathlib import Path

import numpy as np
import scipy.io as sio
import torch

ROOT = Path(sys.argv[1]).resolve()
sys.path.insert(0, str(ROOT))
from bsce.data import measurements, sv  # noqa: E402
from bsce.models import build_model  # noqa: E402
from bsce.rng import NumpyMatlabRNG  # noqa: E402
from bsce.train.trainer import Trainer, TrainConfig  # noqa: E402

T = int(sys.argv[2])
OUT = Path(sys.argv[3])
N_TRAIN = int(sys.argv[4]) if len(sys.argv) > 4 else 500
MAXIT = int(sys.argv[5]) if len(sys.argv) > 5 else 5000
torch.set_num_threads(4)

A = sio.loadmat(ROOT / "reference/gmlamp/CE_SV/CSmatrix256128.mat")["A"].astype(np.float64)
DATA = sys.argv[6] if len(sys.argv) > 6 else None
if DATA:
    xt = sio.loadmat(DATA + "_training_channel.mat")["x"][:, :N_TRAIN]
    xv = sio.loadmat(DATA + "_validation_channel.mat")["x"][:, :500]
else:
    xt = sv.beamspace_channels(NumpyMatlabRNG(7001), N_TRAIN, "UPA", 3, 16, 16)
    xv = sv.beamspace_channels(NumpyMatlabRNG(7002), 500, "UPA", 3, 16, 16)
rs = np.random.RandomState(7003)
yt = measurements.gen_measurements_py(xt, A, measurements.draw_snr_list(rs, 0, 20, 100), rs)
yv = measurements.gen_measurements_py(xv, A, measurements.draw_snr_list(rs, 0, 20, 100), rs)
xte = sio.loadmat(DATA + "_test_channel.mat" if DATA else ROOT / "data/SV_UPA/SV_UPA_test_channel.mat")["x"]
TEST_SNR = [0, 4, 8, 12, 16, 20]
yte = {s: measurements.gen_measurements_matlab(xte, A, s, NumpyMatlabRNG(2026 + 1000 + i))
       for i, s in enumerate([0, 2, 4, 6, 8, 10, 12, 14, 16, 18, 20]) if s in TEST_SNR}

VARIANTS = {
    1: {"LADMM (original)": {},
        "+ idea 1: noise-adaptive threshold": dict(threshold="adaptive"),
        "+ idea 2: debias layer": dict(debias="ls"),
        "+ ideas 1 and 2": dict(threshold="adaptive", debias="ls")},
    2: {"LADMM (original)": {},
        "adaptive, median noise level": dict(threshold="adaptive"),
        "+ idea 1: residual noise level": dict(threshold="adaptive", noise_level="residual"),
        "+ ideas 1 and 2": dict(threshold="adaptive", noise_level="residual", debias="ls")},
}[T]


def nmse_db(x_hat, x):
    return 10 * np.log10(np.sum(np.abs(x_hat - x) ** 2) / np.sum(np.abs(x) ** 2))


def run(opts, seed):
    torch.manual_seed(seed)
    m = build_model(dict(model="LADMM", T=T, array="UPA", N1=16, N2=16, precision="float32", **opts), A)
    cfg = TrainConfig(schedule="borgerding", lr=1e-3, refinements=(0.5, 0.1, 0.01), maxit=MAXIT, better_wait=500,
                      ivl=10, batch_size=100, loss="nmse", val_round=None, seed=seed, optimizer="adam_tf1",
                      verbose=False)
    t0 = time.time()
    Trainer(m, cfg, xt, yt, xv, yv).run()
    with torch.no_grad():
        res = {f"{s} dB": nmse_db(m(yte[s]).cpu().numpy(), xte) for s in TEST_SNR}
        res["train"] = nmse_db(m(yt).cpu().numpy(), xt)
        res["validation"] = nmse_db(m(yv).cpu().numpy(), xv)
    res["params"] = sum(p.numel() for p in m.parameters())
    res["seconds"] = time.time() - t0
    return res


out = json.loads(OUT.read_text())["results"] if OUT.exists() else {}      # resume after a restart
for name, opts in VARIANTS.items():
    out.setdefault(name, [])
    for s in range(len(out[name]), 3):
        out[name].append(run(opts, s))
        OUT.write_text(json.dumps(dict(T=T, variants=VARIANTS, results=out), indent=1))
        print(f"  {name} seed {s}: " + "  ".join(f"{k}={v:.2f}" for k, v in out[name][-1].items() if k != "params"), flush=True)
    r = out[name]
    print(f"{name:40s} " + "  ".join(f"{k}={np.mean([x[k] for x in r]):7.2f}±{np.std([x[k] for x in r], ddof=1):.2f}"
                                       for k in [f"{s} dB" for s in TEST_SNR] + ["train"]) +
          f"  params={r[0]['params']}  {np.mean([x['seconds'] for x in r]):.0f}s", flush=True)
OUT.write_text(json.dumps(dict(T=T, variants=VARIANTS, results=out), indent=1))

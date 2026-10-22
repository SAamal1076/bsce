"""Small experiment: cold vs warm start (train.warm_start) for LADMM and LADMM-AT.

SV UPA (GM-LAMP setting: N = 256, M = 128, L = 3, GM-LAMP's CS matrix), y = A (x + n); the data of ladmm_ideas_small.py:
  training   n_train new channels (default 1000), SNR ~ U(0, 20) dB in 100 chunks
  validation 500 new channels, SNR ~ U(0, 20) dB
  test       the standard 2000 SV_UPA test channels with evaluate.py's test noise (seed 2026 + 1000 + i)
T = 3, layer-wise (borgerding) at most 2000 iterations per stage, seed 0; the only difference between runs is warm_start.
usage: python ladmm_warm_small.py <bsce folder> <out.json> [n_train (1000)]
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

OUT = Path(sys.argv[2])
T, MAXIT, SEED = 3, 2000, 0
N_TRAIN = int(sys.argv[3]) if len(sys.argv) > 3 else 1000
torch.set_num_threads(4)

A = sio.loadmat(ROOT / "reference/gmlamp/CE_SV/CSmatrix256128.mat")["A"].astype(np.float64)
xt = sv.beamspace_channels(NumpyMatlabRNG(7001), N_TRAIN, "UPA", 3, 16, 16)
xv = sv.beamspace_channels(NumpyMatlabRNG(7002), 500, "UPA", 3, 16, 16)
rs = np.random.RandomState(7003)
yt = measurements.gen_measurements_py(xt, A, measurements.draw_snr_list(rs, 0, 20, 100), rs)
yv = measurements.gen_measurements_py(xv, A, measurements.draw_snr_list(rs, 0, 20, 100), rs)
xte = sio.loadmat(ROOT / "data/SV_UPA/SV_UPA_test_channel.mat")["x"]
TEST_SNR = [0, 4, 8, 12, 16, 20]
yte = {s: measurements.gen_measurements_matlab(xte, A, s, NumpyMatlabRNG(2026 + 1000 + i))
       for i, s in enumerate([0, 2, 4, 6, 8, 10, 12, 14, 16, 18, 20]) if s in TEST_SNR}

RUNS = {"LADMM, cold start": ({}, False),
        "LADMM, warm start": ({}, True),
        "LADMM-AT, cold start": (dict(threshold="adaptive"), False),
        "LADMM-AT, warm start": (dict(threshold="adaptive"), True)}


def nmse_db(x_hat, x):
    return 10 * np.log10(np.sum(np.abs(x_hat - x) ** 2) / np.sum(np.abs(x) ** 2))


def run(opts, warm):
    torch.manual_seed(SEED)
    m = build_model(dict(model="LADMM", T=T, array="UPA", N1=16, N2=16, precision="float32", **opts), A)
    cfg = TrainConfig(schedule="borgerding", lr=1e-3, refinements=(0.5, 0.1, 0.01), maxit=MAXIT, better_wait=500,
                      ivl=10, batch_size=100, loss="nmse", val_round=None, seed=SEED, optimizer="adam_tf1",
                      warm_start=warm, verbose=False)
    t0 = time.time()
    st = Trainer(m, cfg, xt, yt, xv, yv).run()
    with torch.no_grad():
        res = {f"{s} dB": nmse_db(m(yte[s]).cpu().numpy(), xte) for s in TEST_SNR}
        res["train"] = nmse_db(m(yt).cpu().numpy(), xt)
        res["validation"] = nmse_db(m(yv).cpu().numpy(), xv)
    res["val_per_output"] = st["layer_nmse_db"]            # x_1, z_1, x_2, z_2, x_3, z_3 when each was finished
    res["iterations"] = sum(e.get("iters") or 0 for e in st["log"])
    res["rho"] = [float(m.rho(t).detach()) for t in range(T)]
    res["seconds"] = time.time() - t0
    return res


out = json.loads(OUT.read_text()) if OUT.exists() else {}             # resume after a restart
for name, (opts, warm) in RUNS.items():
    if name not in out:
        out[name] = run(opts, warm)
        OUT.write_text(json.dumps(out, indent=1))
    r = out[name]
    print(f"{name:22s} " + "  ".join(f"{k}={r[k]:7.2f}" for k in [f"{s} dB" for s in TEST_SNR] + ["train", "validation"])
          + f"  iters={r['iterations']}  rho={[round(v, 3) for v in r['rho']]}  {r['seconds']:.0f}s", flush=True)

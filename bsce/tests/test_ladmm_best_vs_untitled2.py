"""bsce LADMM + configs/train/ladmm_best.yaml reproduces the user's untitled2.py training run.

Both run on the same small SV UPA set, with the same minibatch indices: untitled2.py draws them with
torch.randint on the global generator; bsce gets the same sequence through `sampler`. The run uses T = 2,
maxit 1500 and better_wait 300, so the early-stopping test is exercised.

Checked:
  * the same stage names in the same order
  * the same iteration count at every stage stop
  * stage-end validation NMSE equal within 1e-3 dB
  * final weights equal within float32 accumulation noise

Run: python -m pytest -q tests/test_ladmm_best_vs_untitled2.py   (CPU, about 1-2 min)
"""
import importlib.util
import math
import re
import sys
from pathlib import Path

import numpy as np
import pytest
import scipy.io as sio
import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from bsce.config import load_yaml  # noqa: E402
from bsce.models import build_model  # noqa: E402
from bsce.train.trainer import Trainer, TrainConfig  # noqa: E402
from bsce.data.measurements import draw_snr_list, gen_measurements_py  # noqa: E402

U2 = ROOT / "reference/user/untitled2.py"
DATA = ROOT / "data/SV_UPA"
K_TR, K_VA, T, MAXIT, BW, BATCH, SEED = 3000, 500, 2, 1500, 300, 64, 5


def _untitled2():
    spec = importlib.util.spec_from_file_location("untitled2_user", U2)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


@pytest.mark.skipif(not U2.exists() or not (DATA / "SV_UPA_training_channel.mat").exists(), reason="needs data")
def test_ladmm_best_matches_untitled2(tmp_path):
    A = sio.loadmat(ROOT / "reference/gmlamp/CE_SV/CSmatrix256128.mat")["A"].astype(np.float64)
    xt = sio.loadmat(DATA / "SV_UPA_training_channel.mat")["x"][:, :K_TR]
    xv = sio.loadmat(DATA / "SV_UPA_validation_channel.mat")["x"][:, :K_VA]
    rng = np.random.RandomState(0)
    yt = gen_measurements_py(xt, A, draw_snr_list(rng, 0, 20, 10), rng)
    yv = gen_measurements_py(xv, A, draw_snr_list(rng, 0, 20, 10), rng)
    files = {}
    for k, v in dict(tc=("x", xt), tm=("y", yt), vc=("x", xv), vm=("y", yv)).items():
        files[k] = str(tmp_path / f"{k}.mat")
        sio.savemat(files[k], {v[0]: v[1]})

    # ---- user's script
    u = _untitled2()
    torch.manual_seed(SEED)
    um = u.LADMM(A.astype(np.float32), T=T, shrink="soft", untied=True)
    u.do_training(um, str(tmp_path / "u.pt"), files["tc"], files["tm"], files["vc"], files["vm"],
                  trinit=5e-4, refinements=(.5, .2, .1, .01), iv1=10, maxit=MAXIT, better_wait=BW, batch_size=BATCH)
    ck = torch.load(tmp_path / "u.pt", weights_only=False)
    ulog = [(m.group(1), float(m.group(2)), int(m.group(3)))
            for m in re.finditer(r"\n(.+?) nmse=(-?[\d.]+) dB in (\d+) iterations", ck["log"])]

    # ---- bsce with the same protocol
    cfg = load_yaml(ROOT / "configs/train/ladmm_best.yaml")
    cfg["model"]["T"] = T
    cfg["train"].update(maxit=MAXIT, better_wait=BW, batch_size=BATCH, verbose=False)
    bm = build_model(cfg["model"], A)
    g = torch.Generator().manual_seed(SEED)
    tr = Trainer(bm, TrainConfig.from_dict(cfg["train"]), xt, yt, xv, yv,
                 sampler=lambda: torch.randint(0, K_TR, (BATCH,), generator=g))
    st = tr.run()
    blog = [(d["stage"], d["nmse_db"], d["iters"]) for d in st["log"]]

    assert [s for s, _, _ in ulog] == [s for s, _, _ in blog], "stage names / order differ"
    for (s, nu, iu), (_, nb, ib) in zip(ulog, blog):
        assert iu == ib, f"{s}: stopped at {iu} (untitled2) vs {ib} (bsce)"
        assert abs(nu - nb) < 1e-3, f"{s}: {nu:.6f} vs {nb:.6f} dB"
    ms = ck["model_state"]
    worst = 0.0
    for t in range(T):
        for un, bn in ((f"H_matrices.{t}", f"H_matrix_{t}"), (f"rhos.{t}", f"rho_{t}"),
                       (f"thetas.{t}", f"theta_{t}"), (f"alphas.{t}", f"alpha_{t}")):
            a, b = ms[un].double().numpy().ravel(), bm.P(bn).detach().double().numpy().ravel()
            worst = max(worst, float(np.abs(a - b).max() / (np.abs(a).max() + 1e-12)))
    print(f"\n{len(blog)} stages identical; last stage {blog[-1][1]:.6f} dB; max relative weight difference {worst:.2e}")
    assert worst < 1e-3

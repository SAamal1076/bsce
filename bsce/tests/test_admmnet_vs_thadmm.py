"""ADMM-Net (Klioui): bsce port vs the ORIGINAL THADMM-Net repository class (PyTorch)."""
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

from harness.octave import compare, tmpdir
from bsce.data import sv, measurements
from bsce.rng import NumpyMatlabRNG
from bsce.models.admmnet import ADMMNet
from bsce.train.losses import nmse_loss

HERE = Path(__file__).resolve().parent
THADMM = HERE.parent / "reference" / "thadmmnet"


@pytest.mark.skipif(not THADMM.exists(), reason="needs reference/thadmmnet (bsce_reference zips)")
@pytest.mark.parametrize("seed", [0, 1, 2])
def test_admmnet_vs_original(seed, record):
    g = np.random.default_rng(seed)
    A = ((g.random((128, 256)) > 0.5) * 2 - 1) / np.sqrt(128)
    x = sv.beamspace_channels(NumpyMatlabRNG(seed), 30, "ULA")
    y = measurements.gen_measurements_matlab(x, A, 10.0, NumpyMatlabRNG(seed + 3))
    T = 5
    m = ADMMNet(A, T=T, beta=0.1, rho=1.0, precision="float64")
    w = {}
    with torch.no_grad():
        for t in range(T):
            W = m.P(f"W_{t}")
            W.add_(torch.as_tensor(0.01 * (g.standard_normal(W.shape) + 1j * g.standard_normal(W.shape))))
            m.P(f"rho_{t}").fill_(float(1 + 0.2 * g.standard_normal()))
            # the original stores beta as float32: use float32-representable values
            m.P(f"beta_{t}").fill_(float(np.float32(0.1 + 0.02 * g.standard_normal())))
            w[f"W_{t}"] = W.numpy().copy()
            w[f"rho_{t}"] = m.P(f"rho_{t}").numpy().copy()
            w[f"beta_{t}"] = m.P(f"beta_{t}").numpy().astype(np.float32)
    wd = tmpdir("bsce_th_")
    lr, nsteps = 1e-3, 3
    np.savez(wd / "in.npz", A=A, y=y, x=x, T=T, beta=0.1, rho=1.0, lr=lr, nsteps=nsteps, **w)
    subprocess.run([sys.executable, str(HERE / "harness" / "ref_thadmm.py"), str(wd / "in.npz"),
                    str(wd / "out.npz")], check=True, capture_output=True)
    ref = dict(np.load(wd / "out.npz", allow_pickle=True))
    yt, xt = torch.as_tensor(y), torch.as_tensor(x)
    with torch.no_grad():
        f = compare(m(yt).numpy(), ref["x_est"], "ADMM-Net forward", 1e-12)
    m.zero_grad()
    nmse_loss(m(yt), xt).backward()
    gmax = 0.0
    for t in range(T):
        for k in ("W", "rho", "beta"):
            key = f"grad_{k}_{t}"
            g_ = m.P(f"{k}_{t}").grad
            if key not in ref:
                assert g_ is None or float(g_.abs().max()) == 0.0
                continue
            # the original keeps beta as a float32 Parameter -> its gradient is rounded to float32
            gmax = max(gmax, compare(g_.numpy().reshape(ref[key].shape), ref[key], key,
                                     1e-6 if k == "beta" else 1e-8)["rel"])
    # Adam (torch.optim.Adam, as in the original training loop)
    m.load_mat_dict({**{k: v for k, v in w.items()}})
    opt = torch.optim.Adam(m.parameters(), lr)
    for _ in range(nsteps):
        loss = nmse_loss(m(yt), xt)
        m.zero_grad()
        loss.backward()
        opt.step()
    amax = max(compare(m.P(f"{k}_{t}").detach().numpy().reshape(ref[f"after_{k}_{t}"].shape) - w[f"{k}_{t}"].reshape(ref[f"after_{k}_{t}"].shape),
                       ref[f"after_{k}_{t}"] - w[f"{k}_{t}"].reshape(ref[f"after_{k}_{t}"].shape), f"Adam {k}_{t}", 1e-6)["rel"]
               for t in range(T) for k in ("W", "rho") )
    record("admmnet/vs_original", seed=seed, T=T, fwd_rel=f["rel"], grad_rel=gmax, adam_rel=amax,
           beta_dtype=str(ref["beta_dtype"]))

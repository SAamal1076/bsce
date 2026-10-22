#!/usr/bin/env python3
"""GM-LAMP symmetry experiment: round-off sensitivity of training from the original init.

The original GM initial value has K identical mixture components. Training layer 2 of GM-LAMP
from a warm start (theta_1 <- theta_0, as do_training does) is run twice with identical
minibatches: once as is, once with theta_1 perturbed by one part in 1e15. With identical
components the perturbation (like round-off) is amplified exponentially; with the 'spread'
initial value (bsce gm_init='spread') it stays at round-off level.

  python tests/diagnostics/gm_symmetry.py      -> tests/results/gm_symmetry.{png,json}
"""
import json
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from bsce.data import sv, measurements  # noqa: E402
from bsce.rng import NumpyMatlabRNG  # noqa: E402
from bsce.models.lamp import LAMP  # noqa: E402
from bsce.train.trainer import Trainer, TrainConfig, build_stages  # noqa: E402
from bsce.train.losses import nmse_loss  # noqa: E402
from bsce.train.optim import AdamTF1  # noqa: E402

OUT = ROOT / "tests" / "results"
STEPS = 400


def data(seed=0, n_train=600, n_val=100, snr=10.0):
    g = np.random.default_rng(seed)
    A = ((g.random((128, 256)) > 0.5) * 2 - 1) / np.sqrt(128)
    xt = sv.beamspace_channels(NumpyMatlabRNG(seed), n_train, "ULA")
    xv = sv.beamspace_channels(NumpyMatlabRNG(seed + 50), n_val, "ULA")
    yt = measurements.gen_measurements_matlab(xt, A, snr, NumpyMatlabRNG(seed + 7))
    yv = measurements.gen_measurements_matlab(xv, A, snr, NumpyMatlabRNG(seed + 8))
    return A, xt, yt, xv, yv


def layer2_runs(gm_init, D):
    A, xt, yt, xv, yv = D
    model = LAMP(A, T=2, shrink="gm", K=4, precision="float64", gm_init=gm_init)
    cfg = TrainConfig(lr=1e-3, maxit=100, better_wait=30, ivl=1, warm_start=True, batch_size=128,
                      loss="nmse", val_round=5, seed=1, verbose=False)
    stages = build_stages(model, cfg)
    k = [s["name"] for s in stages].index("LAMP-gm non-linear T=2")
    Trainer(model, cfg, xt, yt, xv, yv).run(stages[:k])          # layer 1 + linear part of layer 2
    with torch.no_grad():
        model.P("theta_1").copy_(model.P("theta_0"))              # do_training's warm start
    start = {n: model.P(n).detach().clone() for n in model.param_names()}
    g = torch.Generator().manual_seed(7)
    batches = [torch.randint(0, xt.shape[1], (128,), generator=g) for _ in range(STEPS)]
    Yt, Xt, Yv, Xv = (torch.as_tensor(v) for v in (yt, xt, yv, xv))
    runs = []
    for eps in (0.0, 1e-15):
        with torch.no_grad():
            for n, v in start.items():
                model.P(n).copy_(v)
            model.P("theta_1")[1, 0] *= (1 + eps)                # one part in 1e15 on one mean
        opt = AdamTF1([model.P("theta_1")], lr=1e-3)
        traj, nmse = [], []
        for i, idx in enumerate(batches):
            traj.append(model.P("theta_1").detach().numpy().copy())
            if i % 20 == 0:
                with torch.no_grad():
                    nmse.append(10 * np.log10(float(nmse_loss(model.forward_stages(Yv, upto=3)[-1], Xv))))
            opt.zero_grad()
            nmse_loss(model.forward_stages(Yt[:, idx], upto=3)[-1], Xt[:, idx]).backward()
            opt.step()
        traj.append(model.P("theta_1").detach().numpy().copy())
        runs.append((np.array(traj), np.array(nmse)))
    (t0, n0), (t1, n1) = runs
    diff = np.array([np.linalg.norm(a - b) / np.linalg.norm(b) for a, b in zip(t1, t0)])
    split = np.array([np.ptp(t[1]) for t in t0])                 # spread of the means
    return dict(diff=diff, split=split, nmse0=n0, nmse1=n1, theta_final0=t0[-1], theta_final1=t1[-1])


def main():
    D = data()
    res = {m: layer2_runs(m, D) for m in ("original", "spread")}
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(1, 2, figsize=(11, 4.2))
    col = {"original": "#e34948", "spread": "#2a78d6"}
    lab = {"original": "original init (K identical components)", "spread": "spread init (gm_init='spread')"}
    s = np.arange(STEPS + 1)
    summary = {}
    for m, r in res.items():
        d = np.maximum(r["diff"], 1e-18)
        ax[0].semilogy(s, d, color=col[m], lw=2, label=lab[m])
        ax[0].annotate(f"{d[-1]:.1e}", (s[-1], d[-1]), textcoords="offset points", xytext=(-40, 6), color=col[m], fontsize=8)
        sp = np.maximum(r["split"], 1e-18)
        ax[1].semilogy(s, sp, color=col[m], lw=2, label=lab[m])
        txt = f"{r['split'][-1]:.1e}" if r["split"][-1] > 0 else "exactly 0 at every step (plotted at 1e-18)"
        ax[1].annotate(txt, (s[-1], sp[-1]), textcoords="offset points", xytext=(-200 if r["split"][-1] == 0 else -40, 6),
                       color=col[m], fontsize=8)
        summary[m] = dict(final_rel_diff=float(r["diff"][-1]), final_mean_spread=float(r["split"][-1]),
                          final_val_nmse_db=[float(r["nmse0"][-1]), float(r["nmse1"][-1])],
                          theta1_final_unperturbed=r["theta_final0"].tolist(),
                          theta1_final_perturbed=r["theta_final1"].tolist())
    for m, r in res.items():
        ax[0].text(170, 1e-12 if m == "original" else 1.5e-13, f"final val. NMSE {lab[m].split(' (')[0]}: "
                   f"{r['nmse0'][-1]:.4f} / {r['nmse1'][-1]:.4f} dB (unpert./pert.)", color=col[m], fontsize=7)
    ax[0].set_title("theta_1: perturbed (1e-15) vs unperturbed run", fontsize=9)
    ax[0].set_xlabel("Adam step of stage 'non-linear T=2'"); ax[0].set_ylabel("relative difference")
    ax[1].set_title("spread of the 4 component means of theta_1 (unperturbed run)", fontsize=9)
    ax[1].set_xlabel("Adam step"); ax[1].set_ylabel("max - min of means")
    for a in ax:
        a.grid(alpha=0.3); a.legend(frameon=False, fontsize=8)
        for sp_ in ("top", "right"):
            a.spines[sp_].set_visible(False)
    fig.suptitle("GM-LAMP: training from identical mixture components amplifies round-off", fontsize=10)
    fig.tight_layout()
    OUT.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT / "gm_symmetry.png", dpi=180)
    (OUT / "gm_symmetry.json").write_text(json.dumps(summary, indent=1), encoding="utf-8")
    np.savez(OUT / "gm_symmetry_curves.npz", **{f"{m}_{k}": v for m, r in res.items() for k, v in r.items()})
    print(json.dumps({m: {k: v for k, v in d.items() if not k.startswith("theta")} for m, d in summary.items()}, indent=1))


if __name__ == "__main__":
    main()

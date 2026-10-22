"""Smoke test 3: a short UTRR training in the paper's own setting (noise after the combiner).
Data: bsce's SV ULA channels (GM-LAMP's generator; test = the 2,000 standard test channels), CS matrix of GM-LAMP
(M = 128). Measurements z = Phi h + n with SNR = ||Phi h||^2 / ||n||^2 per channel (the paper's definition), the noise
drawn once (a fixed data set, as the paper's D). Real and imaginary parts are two real samples.
Training as Sec. V-A, shortened for a CPU: Adam, loss E||x - xhat||^2, batch 128 real samples, learning rates
0.005 -> 0.001 -> 0.0005 -> 0.0002 -> 0.0001 with early stopping per rate (patience in epochs), at most --epochs epochs.
usage: python smoke3_train.py <bsce> <out> --snr 20 --K 16 --bn 1 --sign correct --ntrain 20000 --epochs 20 --patience 3
"""
import argparse
import json
import math
import sys
import time
from pathlib import Path

import numpy as np
import scipy.io as sio
import torch

sys.path.insert(0, str(Path(__file__).parent))
from utrr_ref import UTRRNet  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("root"); ap.add_argument("out")
ap.add_argument("--snr", type=float, default=20); ap.add_argument("--K", type=int, default=16)
ap.add_argument("--L", type=int, default=30); ap.add_argument("--bn", type=int, default=1)
ap.add_argument("--sign", default="correct"); ap.add_argument("--ntrain", type=int, default=20000)
ap.add_argument("--epochs", type=int, default=20); ap.add_argument("--patience", type=int, default=3)
ap.add_argument("--lrs", type=float, nargs="+", default=[0.005, 0.001, 0.0005, 0.0002, 0.0001])
ap.add_argument("--tag", default=None); ap.add_argument("--seed", type=int, default=0)
a = ap.parse_args()
root, out = Path(a.root), Path(a.out)
out.mkdir(parents=True, exist_ok=True)
tag = a.tag or f"utrr_snr{a.snr:g}_K{a.K}_bn{a.bn}_{a.sign}_n{a.ntrain}"
torch.manual_seed(a.seed)
rng = np.random.default_rng(1000 + int(a.snr))

Phi = sio.loadmat(root / "reference/gmlamp/CE_SV/CSmatrix256128.mat")["A"].astype(np.float64)
M, N = Phi.shape


def load(k, n=None):
    x = sio.loadmat(root / f"data/SV_ULA/SV_ULA_{k}_channel.mat")["x"]
    return x if n is None else x[:, :n]


def measure(x):
    """z = Phi x + n, SNR = ||Phi x||^2 / ||n||^2 per channel (complex), fixed noise."""
    s = Phi @ x
    p = np.sum(np.abs(s) ** 2, 0) / M
    s2 = p / 10 ** (a.snr / 10)
    n = np.sqrt(s2 / 2) * (rng.standard_normal(s.shape) + 1j * rng.standard_normal(s.shape))
    return s + n


def real(c):
    return torch.tensor(np.concatenate([c.real, c.imag], 1), dtype=torch.float32)


xt, xv, xs = load("training", a.ntrain), load("validation"), load("test")
Xt, Yt = real(xt), real(measure(xt))
Xv, Yv = real(xv), real(measure(xv))
Xs, Ys = real(xs), real(measure(xs))
net = UTRRNet(Phi, L=a.L, K=a.K, rcc=True, bn=bool(a.bn), sign=a.sign)


@torch.no_grad()
def nmse(X, Y):
    net.eval()
    xh = torch.cat([net(Y[:, i:i + 1000]) for i in range(0, Y.shape[1], 1000)], 1)
    net.train()
    B = X.shape[1] // 2
    e = (xh[:, :B] - X[:, :B]) ** 2 + (xh[:, B:] - X[:, B:]) ** 2
    p = X[:, :B] ** 2 + X[:, B:] ** 2
    pooled = float(e.sum() / p.sum())
    per = float((e.sum(0) / p.sum(0)).mean())
    return 10 * math.log10(pooled), 10 * math.log10(per)


log = []
v0 = nmse(Xv, Yv)
log.append(dict(epoch=0, lr=None, val_pooled_dB=v0[0], val_per_sample_dB=v0[1], sec=0))
print(f"[{tag}] init: val NMSE {v0[0]:.2f} dB (per-sample mean {v0[1]:.2f} dB)", flush=True)
best = (v0[0], {k: v.clone() for k, v in net.state_dict().items()})
ep_total, t0 = 0, time.time()
R = Xt.shape[1]
for lr in a.lrs:
    opt = torch.optim.Adam(net.parameters(), lr=lr)
    stale = 0
    while ep_total < a.epochs and stale < a.patience:
        perm = torch.randperm(R)
        for s in range(0, R, 128):
            idx = perm[s:s + 128]
            opt.zero_grad(set_to_none=True)
            loss = torch.sum((net(Yt[:, idx]) - Xt[:, idx]) ** 2) / idx.numel()
            loss.backward()
            opt.step()
        ep_total += 1
        v = nmse(Xv, Yv)
        vt = nmse(Xt[:, list(range(2000)) + list(range(R // 2, R // 2 + 2000))],
                  Yt[:, list(range(2000)) + list(range(R // 2, R // 2 + 2000))])     # 2,000 training channels
        if not math.isfinite(v[0]):
            print(f"[{tag}] NaN at epoch {ep_total}", flush=True)
            break
        log.append(dict(epoch=ep_total, lr=lr, val_pooled_dB=v[0], val_per_sample_dB=v[1], train2k_pooled_dB=vt[0],
                        sec=round(time.time() - t0)))
        if v[0] < best[0] - 1e-4:
            best, stale = (v[0], {k: t.clone() for k, t in net.state_dict().items()}), 0
        else:
            stale += 1
        print(f"[{tag}] epoch {ep_total:3d} lr {lr:.4f}: val {v[0]:.2f} dB (per-sample {v[1]:.2f}), train-2k {vt[0]:.2f}, best {best[0]:.2f}, "
              f"{time.time() - t0:.0f} s", flush=True)
    net.load_state_dict(best[1])                    # early stopping: back to the best weights before the next rate
    if ep_total >= a.epochs:
        break
ts = nmse(Xs, Ys)
res = dict(tag=tag, args=vars(a), params=net.n_params(), log=log, test_pooled_dB=ts[0], test_per_sample_dB=ts[1],
           best_val_dB=best[0], minutes=round((time.time() - t0) / 60, 1),
           paper_table2=dict(snr10={"K0": -18.92, "K16": -18.92, "K64": -19.00, "K256": -18.77, "E": -22.06},
                             snr20={"K0": -26.97, "K16": -26.93, "K64": -26.73, "K256": -26.94, "E": -30.18}))
(out / f"{tag}.json").write_text(json.dumps(res, indent=1))
torch.save(net.state_dict(), out / f"{tag}.pt")
print(f"[{tag}] TEST NMSE {ts[0]:.2f} dB pooled, {ts[1]:.2f} dB per-sample mean; {res['minutes']} min", flush=True)

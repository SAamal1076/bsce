#!/usr/bin/env python3
"""Reproduce UTRR's Table II (Wu, Cheng, Eldar, Cioffi, IEEE TCOM 2025) in the paper's OWN setting, before UTRR is
compared with the other methods in bsce's setting.

The paper's setting differs from bsce's (= GM-LAMP's) in one point: the noise is added AFTER the combiner,
z = Phi h + n, with SNR = ||Phi h||^2 / ||n||^2 (bsce: y = A (x + n)). Everything else is bsce's SV ULA data set
(GM-LAMP's generator, 80,000 / 2,000 / 2,000 channels, M = 128, the CS matrix of GM-LAMP). One model per SNR and top-K,
trained as Sec. V-A: Adam, loss E||x - xhat||^2, batch 128 real samples, learning rates 0.005 -> 0.001 -> 0.0005 ->
0.0002 -> 0.0001, early stopping after 10 epochs without improvement per rate, at most 300 epochs, best weights kept.

  %run scripts/utrr_reproduce.py --snr 20 --K 16 --device cuda                 one model (Table II: -26.93 dB)
  %run scripts/utrr_reproduce.py --snr 20 --K 0 16 64 256 --device cuda        the four models, then UTRR-E (-30.18 dB)
  %run scripts/utrr_reproduce.py --snr 10 --K 0 16 64 256 --device cuda        Table II at 10 dB (-18.92 / -22.06 dB)
  options: --bn 1 (the batch normalisation of Fig. 6)  --ntrain 20000 (a shorter check)  --epochs 300  --patience 10

A stopped run resumes from its last epoch (results/utrr_reproduce/<tag>.ckpt.pt). Results: results/utrr_reproduce/
<tag>.json (validation per epoch, test NMSE), <tag>.pt, and table.md / table_2.png with every model and the ensemble.
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

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from bsce.device import describe, get_device  # noqa: E402
from bsce.models.utrr import UTRR  # noqa: E402

PAPER = {10: {0: -18.92, 16: -18.92, 64: -19.00, 256: -18.77, "E": -22.06},
         20: {0: -26.97, 16: -26.93, 64: -26.73, 256: -26.94, "E": -30.18}}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--snr", type=float, default=20)
    ap.add_argument("--K", type=int, nargs="+", default=[16])
    ap.add_argument("--T", type=int, default=30)
    ap.add_argument("--bn", type=int, default=0, help="1: the batch normalisation of Fig. 6 (default 0, docs/UTRR.md)")
    ap.add_argument("--ntrain", type=int, default=80000)
    ap.add_argument("--epochs", type=int, default=300)
    ap.add_argument("--patience", type=int, default=10)
    ap.add_argument("--lrs", type=float, nargs="+", default=[0.005, 0.001, 0.0005, 0.0002, 0.0001])
    ap.add_argument("--batch", type=int, default=128, help="real samples per batch (the paper: 128)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default="auto")
    ap.add_argument("--data", default="data/SV_ULA")
    ap.add_argument("--out", default="results/utrr_reproduce")
    a = ap.parse_args()
    dev = get_device(a.device)
    print(f"[utrr] device: {describe(dev)}")
    out = ROOT / a.out
    out.mkdir(parents=True, exist_ok=True)
    data = ROOT / a.data
    Phi = sio.loadmat(data / "CSmatrix.mat")["A"].astype(np.float64)
    M = Phi.shape[0]
    rng = np.random.default_rng(1000 + int(a.snr))          # the same noisy data set for every K (as the paper's D)

    def load(k, n=None):
        x = sio.loadmat(data / f"{data.name}_{k}_channel.mat")["x"]
        return x if n is None else x[:, :n]

    def measure(x):
        s = Phi @ x
        s2 = np.sum(np.abs(s) ** 2, 0) / M / 10 ** (a.snr / 10)
        return s + np.sqrt(s2 / 2) * (rng.standard_normal(s.shape) + 1j * rng.standard_normal(s.shape))

    def real(c):
        return torch.tensor(np.concatenate([c.real, c.imag], 1), dtype=torch.float32, device=dev)

    sets = {k: load(k, a.ntrain if k == "training" else None) for k in ("training", "validation", "test")}
    T_ = {k: (real(x), real(measure(x))) for k, x in sets.items()}
    xh_test = {}
    rows = []
    for K in a.K:
        tag = f"UTRR_snr{a.snr:g}_K{K}_T{a.T}_bn{a.bn}" + ("" if a.ntrain == 80000 else f"_n{a.ntrain}")
        torch.manual_seed(a.seed + K)
        net = UTRR(Phi, T=a.T, K=K, bn=bool(a.bn), precision="float32").to(dev)
        Xt, Yt = T_["training"]
        Xv, Yv = T_["validation"]

        @torch.no_grad()
        def predict(Y):
            return torch.cat([real_forward(net, Y[:, i:i + 2000]) for i in range(0, Y.shape[1], 2000)], 1)

        def nmse_db(X, Y):
            xh = predict(Y)
            B = X.shape[1] // 2
            e = (xh[:, :B] - X[:, :B]) ** 2 + (xh[:, B:] - X[:, B:]) ** 2
            p = X[:, :B] ** 2 + X[:, B:] ** 2
            return 10 * math.log10(float(e.sum() / p.sum())), 10 * math.log10(float((e.sum(0) / p.sum(0)).mean())), xh

        ck = out / f"{tag}.ckpt.pt"
        st = dict(epoch=0, lr_i=0, stale=0, best=float("inf"), log=[])
        opt_state = None
        if ck.exists():
            c = torch.load(ck, map_location=dev, weights_only=False)
            net.load_state_dict(c["model"]); st = c["state"]; opt_state = c["opt"]; best_sd = c["best_sd"]
            print(f"[utrr] {tag}: resumed at epoch {st['epoch']} (rate {a.lrs[st['lr_i']]:g})")
        else:
            v0 = nmse_db(Xv, Yv)
            st["log"].append(dict(epoch=0, lr=None, val_dB=v0[0], val_per_sample_dB=v0[1]))
            best_sd = {k: v.clone() for k, v in net.state_dict().items()}
            print(f"[utrr] {tag}: {net.n_params():,} parameters; initial validation NMSE {v0[0]:.2f} dB")
        t0 = time.time()
        R = Xt.shape[1]
        g = torch.Generator(device="cpu").manual_seed(a.seed + 7 * K + st["epoch"])
        while st["lr_i"] < len(a.lrs) and st["epoch"] < a.epochs:
            lr = a.lrs[st["lr_i"]]
            opt = torch.optim.Adam(net.parameters(), lr=lr)
            if opt_state is not None:
                opt.load_state_dict(opt_state); opt_state = None
            while st["epoch"] < a.epochs and st["stale"] < a.patience:
                perm = torch.randperm(R, generator=g).to(dev)
                net.train()
                for s in range(0, R, a.batch):
                    idx = perm[s:s + a.batch]
                    opt.zero_grad(set_to_none=True)
                    loss = torch.sum((real_forward(net, Yt[:, idx]) - Xt[:, idx]) ** 2) / idx.numel()
                    loss.backward()
                    opt.step()
                st["epoch"] += 1
                v = nmse_db(Xv, Yv)
                st["log"].append(dict(epoch=st["epoch"], lr=lr, val_dB=v[0], val_per_sample_dB=v[1]))
                if v[0] < st["best"] - 1e-4:
                    st["best"], st["stale"] = v[0], 0
                    best_sd = {k: t.detach().clone() for k, t in net.state_dict().items()}
                else:
                    st["stale"] += 1
                print(f"[utrr] {tag} epoch {st['epoch']:3d} lr {lr:g}: val {v[0]:.2f} dB (best {st['best']:.2f}), "
                      f"{time.time() - t0:.0f} s", flush=True)
                torch.save(dict(model=net.state_dict(), state=st, opt=opt.state_dict(), best_sd=best_sd), ck)
            if st["stale"] < a.patience:                       # the epoch budget ended the run, not early stopping
                break
            net.load_state_dict(best_sd)                       # early stopping: the best weights go on
            st["lr_i"] += 1
            st["stale"] = 0
            torch.save(dict(model=net.state_dict(), state=st, opt=None, best_sd=best_sd), ck)
        net.load_state_dict(best_sd)
        te = nmse_db(*T_["test"])
        xh_test[K] = te[2].cpu()
        torch.save(dict(cfg=net.cfg, state_dict=net.state_dict(), A=Phi), out / f"{tag}.pt")
        r = dict(tag=tag, K=K, snr=a.snr, test_dB=te[0], test_per_sample_dB=te[1], best_val_dB=st["best"],
                 epochs=st["epoch"], paper_dB=PAPER.get(int(a.snr), {}).get(K), log=st["log"], args=vars(a))
        (out / f"{tag}.json").write_text(json.dumps(r, indent=1), encoding="utf-8")
        rows.append(r)
        print(f"[utrr] {tag}: TEST NMSE {te[0]:.2f} dB (per-sample mean {te[1]:.2f} dB); paper {r['paper_dB']} dB")
    if len(a.K) > 1:
        Xs = T_["test"][0].cpu()
        xm = torch.stack([xh_test[K] for K in a.K]).mean(0)
        B = Xs.shape[1] // 2
        e = (xm[:, :B] - Xs[:, :B]) ** 2 + (xm[:, B:] - Xs[:, B:]) ** 2
        p = Xs[:, :B] ** 2 + Xs[:, B:] ** 2
        ens = 10 * math.log10(float(e.sum() / p.sum()))
        rows.append(dict(tag=f"UTRR-E (K = {', '.join(map(str, a.K))})", K="E", test_dB=ens,
                         paper_dB=PAPER.get(int(a.snr), {}).get("E")))
        print(f"[utrr] UTRR-E: TEST NMSE {ens:.2f} dB; paper {rows[-1]['paper_dB']} dB")
    md = [f"# UTRR in the paper's setting (noise after the combiner), SNR {a.snr:g} dB\n",
          "| model | test NMSE (dB) | paper, Table II (dB) | epochs |", "|---|---|---|---|"]
    md += [f"| {r['tag']} | {r['test_dB']:.2f} | {r['paper_dB'] if r['paper_dB'] is not None else '-'} | "
           f"{r.get('epochs', '-')} |" for r in rows]
    (out / f"table_snr{a.snr:g}.md").write_text("\n".join(md) + "\n", encoding="utf-8")
    print("\n".join(md))


def real_forward(net, Yr):
    """The paper's real-valued network on real samples (the columns of Yr: real and imaginary parts)."""
    return net.forward_real(Yr)[-1]


if __name__ == "__main__":
    main()

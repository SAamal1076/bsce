"""Is the authors' validation set like a random 2000-channel draw from our channels? For each data set, the authors'
published 10-20 dB networks (LAMP, GM-LAMP) are run at 10 dB on ALL our training + validation channels; the NMSE of
5000 random 2000-channel subsets gives the spread a validation set has by chance. The authors' logged validation NMSE
(their validation set, 10 dB) and the NMSE on the test set are placed in that distribution."""
import json, re, sys
import numpy as np, scipy.io as sio
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))
from authors_forward import lamp_forward
O, D, out = Path(sys.argv[1]), Path(sys.argv[2]), Path(sys.argv[3])
A_of = {"SV": sio.loadmat(O / "CE for SV Channel Model/CSmatrix256128.mat")["A"].astype(float),
        "DeepMIMO": sio.loadmat(O / "CE for DeepMIMO Dataset/CSmatrix_128256.mat")["A"].astype(float)}
res = {}
for sc in ["SV_ULA", "SV_UPA", "DeepMIMO_ULA", "DeepMIMO_UPA"]:
    ch, arr = sc.split("_")
    A = A_of[ch]
    pool = np.concatenate([sio.loadmat(D / sc / f"{sc}_training_channel.mat")["x"], sio.loadmat(D / sc / f"{sc}_validation_channel.mat")["x"]], 1)
    pool = pool[:, np.random.default_rng(5).choice(pool.shape[1], 20000, replace=False)]   # 20,000 random channels of the pool
    xt = sio.loadmat(D / sc / f"{sc}_test_channel.mat")["x"]
    for net in ["LAMP", "GM_LAMP"]:
        f = O / "Train Network/Trained_Networks" / (f"Trained_{net}_for_{sc}_25612810to20dB_10dB" + ("_K=4" if net == "GM_LAMP" else "") + ".npz")
        d = np.load(f, allow_pickle=True)
        logged = float(re.findall(r"nmse=(-?[\d.]+) dB in", str(d["log"]))[-1])
        Bs, ths = [d[f"B_{t}:0"] for t in range(8)], [d[f"theta_{t}:0"] for t in range(8)]
        shrink = "gm" if net == "GM_LAMP" else "soft"
        def per_user(x, seed):
            r = np.random.default_rng(seed)
            n = np.sqrt(0.1) * (r.standard_normal(x.shape) + 1j * r.standard_normal(x.shape)) / np.sqrt(2)
            e = []
            for s in range(0, x.shape[1], 4000):
                xs = x[:, s:s + 4000]
                e.append(np.sum(np.abs(lamp_forward(A @ (xs + n[:, s:s + 4000]), A, Bs, ths, shrink) - xs) ** 2, 0))
            return np.concatenate(e), np.sum(np.abs(x) ** 2, 0)
        e, en = per_user(pool, 11)
        et, ent = per_user(xt, 12)
        rb = np.random.default_rng(3)
        sub = np.array([10 * np.log10(e[i].sum() / en[i].sum()) for i in (rb.choice(e.size, 2000, replace=False) for _ in range(5000))])
        test = 10 * np.log10(et.sum() / ent.sum())
        pct = lambda v: round(float(np.mean(sub <= v) * 100), 1)
        res[f"{net.replace('_', '-')} {sc}"] = dict(pool=int(e.size), subsets_mean=round(float(sub.mean()), 3), subsets_sd=round(float(sub.std()), 3),
                                                    p2_5=round(float(np.percentile(sub, 2.5)), 3), p97_5=round(float(np.percentile(sub, 97.5)), 3),
                                                    authors_val=logged, authors_val_percentile=pct(logged), test=round(float(test), 3),
                                                    test_percentile=pct(test), subsets=np.round(sub, 4).tolist())
        r = res[f"{net.replace('_', '-')} {sc}"]
        print(f"{net:8s} {sc:13s} our 2000-subsets {r['subsets_mean']:.3f} +- {r['subsets_sd']:.3f} (95% {r['p2_5']:.3f} .. {r['p97_5']:.3f}) | "
              f"authors' val {logged:.3f} (percentile {r['authors_val_percentile']}) | test {test:.3f} (percentile {r['test_percentile']})", flush=True)
out.mkdir(parents=True, exist_ok=True)
(out / "validation_percentiles.json").write_text(json.dumps(res))

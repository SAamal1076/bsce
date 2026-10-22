"""DeepMIMO ULA: where do our validation users and the authors' validation set fall among random 2000-user sets?"""
import re, sys
import numpy as np, scipy.io as sio
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))
from authors_forward import lamp_forward
O, D = Path(sys.argv[1]), Path(sys.argv[2])
sc = sys.argv[3] if len(sys.argv) > 3 else "DeepMIMO_ULA"
A = sio.loadmat(O / "CE for DeepMIMO Dataset/CSmatrix_128256.mat")["A"].astype(float)
d = np.load(O / f"Train Network/Trained_Networks/Trained_LAMP_for_{sc}_25612810to20dB_10dB.npz", allow_pickle=True)
logged = float(re.findall(r"nmse=(-?[\d.]+) dB in", str(d["log"]))[-1])
Bs, ths = [d[f"B_{t}:0"] for t in range(8)], [d[f"theta_{t}:0"] for t in range(8)]
xt = sio.loadmat(D / sc / f"{sc}_training_channel.mat")["x"]
xv = sio.loadmat(D / sc / f"{sc}_validation_channel.mat")["x"]
x = np.concatenate([xt, xv], 1)
out = {}
for seed in (11, 12):
    r = np.random.default_rng(seed)
    n = np.sqrt(0.1) * (r.standard_normal(x.shape) + 1j * r.standard_normal(x.shape)) / np.sqrt(2)
    e = np.concatenate([np.sum(np.abs(lamp_forward(A @ (x[:, s:s + 5000] + n[:, s:s + 5000]), A, Bs, ths, "soft") - x[:, s:s + 5000]) ** 2, 0)
                        for s in range(0, x.shape[1], 5000)])
    en = np.sum(np.abs(x) ** 2, 0)
    rb = np.random.default_rng(3)
    sub = np.array([10 * np.log10(e[i].sum() / en[i].sum()) for i in (rb.choice(e.size, 2000, replace=False) for _ in range(5000))])
    ours = 10 * np.log10(e[-2000:].sum() / en[-2000:].sum())
    print(f"noise seed {seed}: random 2000-sets {sub.mean():.3f} +- {sub.std():.3f}; our validation users {ours:.3f} "
          f"(percentile {np.mean(sub <= ours) * 100:.1f}); authors' logged {logged:.3f} (percentile {np.mean(sub <= logged) * 100:.1f})")

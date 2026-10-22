"""Are the authors' training states (Train Network/Trained_Networks/*.npz, with their training logs) the same
networks as the published .mat files used by Main.m (CE for SV / CE for DeepMIMO)?"""
import glob, re, sys
import numpy as np, scipy.io as sio
from pathlib import Path
O = Path(sys.argv[1])
rows = []
for f in sorted((O / "Train Network/Trained_Networks").glob("*.npz")):
    m = re.match(r"Trained_(GM_LAMP|LAMP)_for_(SV|DeepMIMO)_(ULA|UPA)_256128(0to10dB|10to20dB)_10dB", f.name)
    net, ch, arr, band = m.groups()
    mat = O / ("CE for SV Channel Model" if ch == "SV" else "CE for DeepMIMO Dataset") / f"Trained_{net}_for_{ch}_{arr}_256128{band}.mat"
    d = np.load(f, allow_pickle=True)
    if not mat.exists():
        rows.append((f.name, "no .mat")); continue
    w = sio.loadmat(mat)
    err = 0.0
    for t in range(8):
        B = d[f"B_{t}:0"]; th = d[f"theta_{t}:0"]
        Bm = w[f"B_{t}"]; thm = w[f"theta{t}"]
        err = max(err, np.abs(B - Bm).max() / np.abs(B).max(), np.abs(th - thm.reshape(th.shape)).max() / max(np.abs(th).max(), 1e-12))
    rows.append((f.name, f"max rel diff {err:.1e}"))
for r in rows: print(*r)

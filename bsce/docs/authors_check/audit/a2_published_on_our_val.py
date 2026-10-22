"""The authors' 16 published networks on OUR validation sets (10 dB, the validation SNR of the authors), against the
validation NMSE the authors logged at the end of training (their own validation set, same SNR). Also: the same
networks on the test sets at 0, 10, 20 dB (DeepMIMO: the authors' own test file)."""
import json, re, sys
import numpy as np, scipy.io as sio
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))
from authors_forward import lamp_forward, nmse_db
O, D = Path(sys.argv[1]), Path(sys.argv[2])          # authors' package, bsce data folder
out = Path(sys.argv[3]); out.mkdir(parents=True, exist_ok=True)
A_of = {"SV": sio.loadmat(O / "CE for SV Channel Model/CSmatrix256128.mat")["A"].astype(float),
        "DeepMIMO": sio.loadmat(O / "CE for DeepMIMO Dataset/CSmatrix_128256.mat")["A"].astype(float)}
rows = []
for f in sorted((O / "Train Network/Trained_Networks").glob("*.npz")):
    net, ch, arr, band = re.match(r"Trained_(GM_LAMP|LAMP)_for_(SV|DeepMIMO)_(ULA|UPA)_256128(0to10dB|10to20dB)_10dB", f.name).groups()
    d = np.load(f, allow_pickle=True)
    logged = float(re.findall(r"nmse=(-?[\d.]+) dB in", str(d["log"]))[-1])
    Bs = [d[f"B_{t}:0"] for t in range(8)]
    ths = [d[f"theta_{t}:0"] for t in range(8)]
    sc = f"{ch}_{arr}"
    A = A_of[ch]
    xv = sio.loadmat(D / sc / f"{sc}_validation_channel.mat")["x"]
    fv = sio.loadmat(D / sc / f"{sc}_validation_128_measurements_{band}.mat")
    yv, snr = fv["y"], np.unique(np.round(np.asarray(fv["snr_db"]).ravel(), 6))
    xh = lamp_forward(yv, A, Bs, ths, "gm" if net == "GM_LAMP" else "soft")
    ours = nmse_db(xh, xv)
    e, en = np.sum(np.abs(xh - xv) ** 2, 0), np.sum(np.abs(xv) ** 2, 0)
    rb = np.random.default_rng(7)
    boot = [10 * np.log10(e[i].mean() / en[i].mean()) for i in (rb.integers(0, e.size, e.size) for _ in range(2000))]
    sd = float(np.std(boot))     # sampling std of a 2000-channel validation NMSE (bootstrap)
    # test set, fixed noise (seeded here; all networks of a scenario see the same noise)
    xt = sio.loadmat(D / sc / f"{sc}_test_channel.mat")["x"]
    test = {}
    for s in ((0, 5) if band == "0to10dB" else (10, 15, 20)):
        r = np.random.default_rng(1000 + s)
        n = np.sqrt(10 ** (-s / 10)) * (r.standard_normal(xt.shape) + 1j * r.standard_normal(xt.shape)) / np.sqrt(2)
        test[s] = round(nmse_db(lamp_forward(A @ (xt + n), A, Bs, ths, "gm" if net == "GM_LAMP" else "soft"), xt), 3)
    rows.append(dict(network=f"{net.replace('_', '-')} {sc} {band}", logged_val_db=logged, our_val_db=round(ours, 3),
                     diff_db=round(ours - logged, 3), boot_sd_db=round(sd, 3), z=round((ours - logged) / (np.sqrt(2) * sd), 2), val_snr=snr.tolist(), test_db=test))
    print(f"{rows[-1]['network']:32s} authors' val {logged:8.3f}  ours {ours:8.3f}  diff {ours - logged:+.3f}  sd {sd:.3f}  z {(ours - logged) / (np.sqrt(2) * sd):+.2f}  (val SNR {snr})  test {test}")
(out / "published_on_our_validation.json").write_text(json.dumps(rows, indent=1))

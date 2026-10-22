"""scripts/make_dataset.py: SV data set with standard test set, any SNR bands, scenario file usable by train.py."""
import subprocess
import sys
from pathlib import Path

import numpy as np
import scipy.io as sio

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from bsce.config import load_yaml  # noqa: E402
from bsce.data import sv  # noqa: E402
from bsce.rng import NumpyMatlabRNG  # noqa: E402


def test_make_sv_dataset(tmp_path):
    name = "tst_SV_ULA_3k"
    yml = ROOT / "configs/scenarios" / f"{name.lower()}.yaml"
    try:
        r = subprocess.run([sys.executable, str(ROOT / "scripts/make_dataset.py"), "--dataset", "SV", "--array", "ULA",
                            "--n", "3000", "--split", "0.8", "0.1", "0.1", "--snr", "-5", "5", "25", "--name", name,
                            "--out-dir", str(tmp_path / name)], capture_output=True, text=True, cwd=ROOT)
        assert r.returncode == 0, r.stdout + r.stderr
        d = tmp_path / name
        x = {k: sio.loadmat(d / f"SV_ULA_{k}_channel.mat")["x"] for k in ("training", "validation", "test")}
        assert x["training"].shape == (256, 2400) and x["validation"].shape == (256, 300)
        std = sv.beamspace_channels(NumpyMatlabRNG(2028), 2000, "ULA", 3, 16, 16)
        assert np.array_equal(x["test"], std)                       # the standard bsce SV test set
        sc = load_yaml(yml)
        assert [b["name"] for b in sc["bands"]] == ["m5to5dB", "5to25dB"]
        for b, (lo, hi) in zip(sc["bands"], ((-5, 5), (5, 25))):
            m = sio.loadmat(d / f"SV_ULA_training_128_measurements_{b['name']}.mat")
            assert m["y"].shape == (128, 2400)
            s = m["snr_db"].ravel()
            assert s.size == 2400 and lo <= s.min() and s.max() <= hi
        assert (d / "dataset_check.png").exists() and (d / "CSmatrix.mat").exists()
    finally:
        yml.unlink(missing_ok=True)

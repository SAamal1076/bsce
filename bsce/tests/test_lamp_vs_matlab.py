"""LAMP / GM-LAMP: PyTorch forward vs the ORIGINAL MATLAB inference (LAMP.m, GM_LAMP.m,
eta_gm.m) run in Octave with the authors' trained networks, on 100 channels.

Covers: SV (ULA/UPA) and DeepMIMO (ULA/UPA) weight sets, both SNR ranges, several seeds/SNRs.
"""
import shutil

import numpy as np
import pytest
import scipy.io as sio
import torch

from harness.octave import REFERENCE, compare, octave_available, run_octave, tmpdir
from bsce.rng import NumpyMatlabRNG
from bsce.data import sv, measurements
from bsce.models.lamp import LAMP

pytestmark = pytest.mark.skipif(not octave_available(), reason="octave not installed")
TOL = 1e-12
K = 100


def _weights(dataset, net, array, rng_label):
    folder = REFERENCE / "gmlamp" / ("CE_SV" if dataset == "SV" else "CE_DeepMIMO")
    pre = "Trained_LAMP" if net == "LAMP" else "Trained_GM_LAMP"
    return folder, sio.loadmat(folder / f"{pre}_for_{dataset}_{array}_256128{rng_label}.mat")


def _channels(dataset, array, seed):
    if dataset == "SV":
        return sv.beamspace_channels(NumpyMatlabRNG(seed), K, array)
    x = sio.loadmat(REFERENCE / "gmlamp" / "CE_DeepMIMO" / f"{array}_test_channel.mat")["x"]
    g = np.random.default_rng(seed)
    return x[:, g.choice(x.shape[1], K, replace=False)]


CASES = []
for dataset in ("SV", "DeepMIMO"):
    for array in ("ULA", "UPA"):
        for snr in (2.0, 7.0, 13.0, 18.0):
            for seed in (0, 1):
                CASES.append((dataset, array, snr, seed))


@pytest.mark.parametrize("dataset,array,snr,seed", CASES)
def test_lamp_gmlamp_vs_octave(dataset, array, snr, seed, record):
    typ = 1 if array == "ULA" else 2
    snr_range, label = (1, "0to10dB") if snr < 10 else (2, "10to20dB")
    folder = REFERENCE / "gmlamp" / ("CE_SV" if dataset == "SV" else "CE_DeepMIMO")
    Afile = "CSmatrix256128.mat" if dataset == "SV" else "CSmatrix_128256.mat"
    # NOTE: CSmatrix_128256.mat stores A as single; MATLAB would then run A*x in single
    # precision (~1e-7). We pass A as double to both sides (the .m code is unchanged).
    A = sio.loadmat(folder / Afile)["A"].astype(np.float64)
    x = _channels(dataset, array, seed)
    y = measurements.gen_measurements_matlab(x, A, snr, NumpyMatlabRNG(1000 + seed))
    wd = tmpdir()
    sio.savemat(wd / "in.mat", {"y": y, "A": A})
    code = f"""cd('{folder.as_posix()}');
S = load('{(wd / 'in.mat').as_posix()}'); y = S.y; A = S.A;
xl = LAMP(y, A, {typ}, {snr_range});
xg = GM_LAMP(y, A, {typ}, {snr_range});
save('-v7', '{(wd / 'out.mat').as_posix()}', 'xl', 'xg');"""
    run_octave(code, wd)
    ref = sio.loadmat(wd / "out.mat")
    for net, key, shrink in (("LAMP", "xl", "soft"), ("GM-LAMP", "xg", "gm")):
        _, W = _weights(dataset, "LAMP" if net == "LAMP" else "GM", array, label)
        model = LAMP(A, T=8, shrink=shrink, untied=True, K=4, precision="float64")
        model.load_mat_dict(W)
        with torch.no_grad():
            xh = model(torch.as_tensor(y)).numpy()
        r = compare(xh, ref[key], f"{net} {dataset} {array} {snr}dB seed{seed}", TOL)
        nm_py = 10 * np.log10(np.mean(np.sum(np.abs(xh - x) ** 2, 0)) / np.mean(np.sum(np.abs(x) ** 2, 0)))
        nm_m = 10 * np.log10(np.mean(np.sum(np.abs(ref[key] - x) ** 2, 0)) / np.mean(np.sum(np.abs(x) ** 2, 0)))
        record("lamp/vs_matlab", net=net, dataset=dataset, array=array, snr=snr, seed=seed, K=K,
               rel=r["rel"], max_abs=r["max_abs"], nmse_torch_db=nm_py, nmse_matlab_db=nm_m)
    shutil.rmtree(wd)

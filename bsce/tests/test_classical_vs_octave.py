"""Classical baselines: numpy ports vs the ORIGINAL .m files (AMP, OMP, SD, ADMM) in Octave."""
import shutil

import numpy as np
import pytest
import scipy.io as sio

from harness.octave import REFERENCE, compare, octave_available, run_octave, tmpdir
from bsce import classical
from bsce.data import sv, measurements
from bsce.rng import NumpyMatlabRNG

CODE = REFERENCE / "ladmm_user" / "ULA_test_code"
pytestmark = [pytest.mark.skipif(not octave_available(), reason="octave not installed"),
              pytest.mark.skipif(not CODE.exists(), reason="needs reference/ladmm_user (bsce_reference zips)")]


@pytest.mark.parametrize("array,snr,seed", [("ULA", 0.0, 0), ("ULA", 15.0, 1), ("UPA", 5.0, 2), ("UPA", 20.0, 3)])
def test_classical(array, snr, seed, record):
    g = np.random.default_rng(seed)
    A = ((g.random((128, 256)) > 0.5) * 2 - 1) / np.sqrt(128)
    x = sv.beamspace_channels(NumpyMatlabRNG(seed), 30, array)
    y = measurements.gen_measurements_matlab(x, A, snr, NumpyMatlabRNG(seed + 1))
    lam = 0.05 + 0.02 * seed
    wd = tmpdir()
    for f in ("AMP.m", "OMP.m", "SD.m", "ADMM_vectorized.m"):
        shutil.copy(CODE / f, wd)
    sio.savemat(wd / "in.mat", {"y": y, "A": A})
    code = f"""load in.mat; K = size(y,2);
xa = AMP(y, A);
xo = []; for i=1:K, xo = [xo, OMP(y(:,i), A, 24)]; end
xs = []; for i=1:K, [h, ~] = SD(y(:,i), A, 3, 8); xs = [xs, h]; end
[xd, conv] = ADMM_vectorized(y, A, {lam}, 1.0, 200, 1e-5); nit = numel(conv.objective);
save('-v7','out.mat','xa','xo','xs','xd','nit');"""
    run_octave(code, wd)
    ref = sio.loadmat(wd / "out.mat")
    r = {}
    r["amp"] = compare(classical.amp(y, A), ref["xa"], "AMP", 1e-12)["rel"]
    r["omp"] = compare(classical.omp(y, A, 24), ref["xo"], "OMP", 1e-10)["rel"]
    r["sd"] = compare(classical.sd(y, A, 3, 8), ref["xs"], "SD", 1e-10)["rel"]
    zd, nit = classical.admm(y, A, lam, 1.0, 200, 1e-5)
    r["admm"] = compare(zd, ref["xd"], "ADMM", 1e-11)["rel"]
    assert nit == int(ref["nit"].ravel()[0]), f"ADMM iterations {nit} vs {int(ref['nit'].ravel()[0])}"
    record("classical/vs_octave", array=array, snr=snr, seed=seed, admm_iters=nit, **{f"{k}_rel": v for k, v in r.items()})
    shutil.rmtree(wd)

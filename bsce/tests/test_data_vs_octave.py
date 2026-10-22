"""Data generation: Python ports vs the ORIGINAL MATLAB/Python files (unchanged).

Random numbers are replayed into Octave through shims, so every number Octave consumes
is the number Python consumes. Several seeds, ULA and UPA.
"""
import importlib.util
import shutil

import numpy as np
import pytest
import scipy.io as sio

from harness.octave import REFERENCE, compare, make_streams, octave_available, run_octave, tmpdir
from bsce.rng import ReplayRNG
from bsce.data import sv, sensing, measurements

pytestmark = pytest.mark.skipif(not octave_available(), reason="octave not installed")
GEN = REFERENCE / "gmlamp" / "CE_SV"          # generate_channel.m used by Main.m (authors)
TRAIN = REFERENCE / "gmlamp" / "Train"        # data scripts of the authors
USER = REFERENCE / "ladmm_user"               # data scripts used in the LADMM project
TOL = 1e-13

# DFT + loop lines copied verbatim from Main.m / generate_channel_data_for_SV_channel_model.m
WRAP = """
N1=16; N2=16; N=N1*N2; L=3; d=0.5; type={type}; K={K};
if type==1
    UN=(1/sqrt(N))*exp(1i*2*pi*[-(N-1):2:N-1]'/2*d*[-(N-1):2:N-1]*(1/N)); % the DFT matrix
end
if type==2
    UN1=(1/sqrt(N1))*exp(1i*2*pi*[-(N1-1):2:N1-1]'/2*d*[-(N1-1):2:N1-1]*(1/N1));
    UN2=(1/sqrt(N2))*exp(1i*2*pi*[-(N2-1):2:N2-1]'/2*d*[-(N2-1):2:N2-1]*(1/N2));
    UN=kron(UN1,UN2); % the DFT matrix
end
x=zeros(N,K);
for k=1:K
    x(:,k)=UN.'*generate_channel(N1,N2,L,type);
end
save('-v7','out.mat','x','UN');
"""


@pytest.mark.parametrize("array,typ", [("ULA", 1), ("UPA", 2)])
@pytest.mark.parametrize("seed", [0, 1, 2, 3, 4])
def test_beamspace_channels(array, typ, seed, record):
    K, L = 60, 3
    per = 2 * L + (L if typ == 1 else 2 * L)
    u, n = make_streams(seed, K * per, K * per)
    wd = tmpdir()
    run_octave(WRAP.format(type=typ, K=K), wd, addpaths=[GEN], replay=(u, n))
    ref = sio.loadmat(wd / "out.mat")
    x_py = sv.beamspace_channels(ReplayRNG(u, n), K, array, L)
    r1 = compare(sv.dft_matrix(array), ref["UN"], f"DFT {array}", TOL)
    r2 = compare(x_py, ref["x"], f"x {array} seed{seed}", TOL)
    record("data/beamspace", array=array, seed=seed, K=K, dft_rel=r1["rel"], x_rel=r2["rel"],
           x_max_abs=r2["max_abs"])
    shutil.rmtree(wd)


@pytest.mark.parametrize("seed", [0, 7, 11])
def test_cs_matrix_original_script(seed, record):
    """Run generate_CS_matrix.m unchanged (it saves CSmatrix256128.mat)."""
    M, N = 128, 256
    u, n = make_streams(seed, M * N, 1)
    wd = tmpdir()
    shutil.copy(TRAIN / "generate_CS_matrix.m", wd)
    run_octave("generate_CS_matrix", wd, replay=(u, n))
    A_ref = sio.loadmat(wd / "CSmatrix256128.mat")["A"]
    A_py = sensing.cs_matrix(ReplayRNG(u, n), M, N)
    r = compare(A_py, A_ref, f"CS matrix seed{seed}", 0.0)
    record("data/cs_matrix", seed=seed, rel=r["rel"], max_abs=r["max_abs"])
    shutil.rmtree(wd)


@pytest.mark.parametrize("snr_db", [0, 7.5, 20])
@pytest.mark.parametrize("seed", [3, 5])
def test_measurements_matlab_style(snr_db, seed, record):
    """Noise + y lines copied verbatim from Main.m."""
    K, N = 40, 256
    rng0 = np.random.default_rng(100 + seed)
    x = rng0.standard_normal((N, K)) + 1j * rng0.standard_normal((N, K))
    A = ((rng0.random((128, N)) > 0.5) * 2 - 1) / np.sqrt(128)
    u, n = make_streams(seed, 1, 2 * N * K)
    wd = tmpdir()
    sio.savemat(wd / "in.mat", {"x": x, "A": A})
    code = f"""load in.mat
SNR_dB={snr_db}; SNR_linear=10.^(SNR_dB/10.); iS=1; N={N}; K={K};
    sigma2=1/SNR_linear(iS);
    noise = sqrt(sigma2)*(randn(N,K)+1i*randn(N,K))/sqrt(2);
    y=A*(x+noise);
save('-v7','out.mat','y','noise');"""
    run_octave(code, wd, replay=(u, n))
    ref = sio.loadmat(wd / "out.mat")
    y_py, noise_py = measurements.gen_measurements_matlab(x, A, snr_db, ReplayRNG(u, n), return_noise=True)
    r1 = compare(noise_py, ref["noise"], "noise", TOL)
    r2 = compare(y_py, ref["y"], "y", TOL)
    record("data/measurements_matlab", seed=seed, snr_db=snr_db, noise_rel=r1["rel"], y_rel=r2["rel"])
    shutil.rmtree(wd)


def _load_original_genmeas():
    spec = importlib.util.spec_from_file_location("orig_genmeas", TRAIN / "generate_measurement_data.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.genMeasurements


@pytest.mark.parametrize("seed", [0, 1, 2])
@pytest.mark.parametrize("band", [(0, 10), (10, 20), (0, 20)])
def test_measurements_python_original(seed, band, record):
    """genMeasurements (original Python, global np.random) vs port (RandomState)."""
    genMeasurements = _load_original_genmeas()
    rng0 = np.random.default_rng(seed + 50)
    N, M, K, nch = 256, 128, 500, 10
    x = rng0.standard_normal((N, K)) + 1j * rng0.standard_normal((N, K))
    A = ((rng0.random((M, N)) > 0.5) * 2 - 1) / np.sqrt(M)
    np.random.seed(seed)
    snr_o = np.random.uniform(band[0], band[1], size=(nch))
    y_o = genMeasurements(x, A, snr_o)
    rs = np.random.RandomState(seed)
    snr_p = measurements.draw_snr_list(rs, band[0], band[1], nch)
    y_p = measurements.gen_measurements_py(x, A, snr_p, rs)
    compare(snr_p, snr_o, "snr list", 0.0)
    r = compare(y_p, y_o, "y (python original)", 0.0)
    record("data/measurements_python", seed=seed, band=list(band), rel=r["rel"], max_abs=r["max_abs"])


@pytest.mark.slow
def test_full_original_training_script_authors(record):
    """Authors' generate_channel_data_for_SV_channel_model.m run UNCHANGED (ULA, S=80000)."""
    S, L = 80000, 3
    u, n = make_streams(2024, S * L, S * 2 * L)
    wd = tmpdir()
    shutil.copy(TRAIN / "generate_channel_data_for_SV_channel_model.m", wd)
    shutil.copy(TRAIN / "generate_channel.m", wd)
    run_octave("generate_channel_data_for_SV_channel_model", wd, replay=(u, n))
    x_ref = sio.loadmat(wd / "SV_ULA_training_channel.mat")["x"]
    x_py = sv.beamspace_channels(ReplayRNG(u, n), S, "ULA", L)
    r = compare(x_py, x_ref, "authors' full script ULA S=80000", TOL)
    record("data/full_script_authors", S=S, rel=r["rel"], max_abs=r["max_abs"])
    shutil.rmtree(wd)


@pytest.mark.slow
@pytest.mark.skipif(not USER.exists(), reason="needs reference/ladmm_user (bsce_reference zips)")
def test_full_original_training_script_ladmm_project(record):
    """LADMM-project script run UNCHANGED: set=1:2 (100000/10000) x type=1:2 (ULA/UPA)."""
    L = 3
    sizes = [(100000, "ULA"), (100000, "UPA"), (10000, "ULA"), (10000, "UPA")]  # loop order
    n_u = sum(S * (L if a == "ULA" else 2 * L) for S, a in sizes)
    n_n = sum(S * 2 * L for S, _ in sizes)
    u, n = make_streams(77, n_u, n_n)
    wd = tmpdir()
    shutil.copy(USER / "generate_channel_data_for_SV_channel_model.m", wd)
    shutil.copy(USER / "generate_channel.m", wd)
    run_octave("generate_channel_data_for_SV_channel_model", wd, replay=(u, n))
    rng = ReplayRNG(u, n)
    files = {(100000, "ULA"): "SV_ULA_training_channel.mat", (100000, "UPA"): "SV_UPA_training_channel.mat",
             (10000, "ULA"): "SV_ULA_validation_channel.mat", (10000, "UPA"): "SV_UPA_validation_channel.mat"}
    for S, a in sizes:
        x_py = sv.beamspace_channels(rng, S, a, L)
        x_ref = sio.loadmat(wd / files[(S, a)])["x"]
        r = compare(x_py, x_ref, f"LADMM-project script {a} S={S}", TOL)
        record("data/full_script_ladmm_project", array=a, S=S, rel=r["rel"], max_abs=r["max_abs"])
    shutil.rmtree(wd)

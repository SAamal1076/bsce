"""The noise-model option (bsce/data/noise.py): colored y = A (x + n) and white y = A x + w.

1. the whitening map: T A A^T T^T = c I, c = tr(AA^T)/M; white noise has the same mean power as the colored noise
2. colored output is unchanged (the same object / the same file as before); white = A x + T (y_colored - A x)
3. measurement generators: paired draws, empirical covariance white, the original RNG stream untouched
4. measurement files: <name>_white.mat made from the colored twin (paired), snr_db copied; names unchanged for colored
5. NF-LADMM: the white data step is the exact proximal map of the white likelihood; colored step unchanged;
   no new saved state (files of earlier versions load)
6. trainer online noise, and the whole pipeline: train.py --noise-model white -> outputs/<scenario>_white_smoke, its
   white files, evaluate.py (auto = white for that folder, --nf-step matched), compare_noise.py
"""
import json
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
import scipy.io as sio
import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from bsce.config import band_file_names, scenario_run_name  # noqa: E402
from bsce.data import measurements, noise, sensing, sv  # noqa: E402
from bsce.data.files import ensure_band_files, online_validation  # noqa: E402
from bsce.models import build_model  # noqa: E402
from bsce.rng import NumpyMatlabRNG  # noqa: E402
from bsce.train.trainer import Trainer, TrainConfig  # noqa: E402

# self-contained data (no data/ or reference/ needed): GM-LAMP's +-1/sqrt(M) combiner and SV UPA channels
A = sensing.cs_matrix(NumpyMatlabRNG(2), 128, 256)
M, N = A.shape


@pytest.fixture(scope="module")
def tiny(tmp_path_factory):
    """A tiny SV UPA data folder <tmp>/SVT_smoke (CSmatrix.mat + training / validation / test channels)."""
    d = tmp_path_factory.mktemp("noise") / "SVT_smoke"
    d.mkdir()
    sio.savemat(d / "CSmatrix.mat", {"A": A})
    for k, (name, K) in enumerate((("training", 1000), ("validation", 200), ("test", 200))):
        sio.savemat(d / f"SV_UPA_{name}_channel.mat", {"x": sv.beamspace_channels(NumpyMatlabRNG(10 + k), K, "UPA")})
    return d


def test_whitening_map():
    T = noise.whitening_map(A)
    c = noise.white_scale(A)
    assert abs(c - np.trace(A @ A.T) / M) < 1e-12 and abs(c - N / M) < 1e-9      # +-1/sqrt(M) combiner: c = N/M
    np.testing.assert_allclose(T @ A @ A.T @ T.T, c * np.eye(M), atol=1e-10)
    np.testing.assert_allclose(T, T.T, atol=1e-12)
    assert np.trace(T @ A @ A.T @ T.T) == pytest.approx(np.trace(A @ A.T), rel=1e-10)   # same mean noise power


def test_names_and_aliases():
    assert [noise.canonical(k) for k in (None, "colored", "Coloured", "folded", "white")] == \
        ["colored", "colored", "colored", "colored", "white"]
    with pytest.raises(ValueError):
        noise.canonical("pink")
    sc = dict(name="SV_UPA", dataset="SV", array="UPA", M=128)
    b = dict(name="0to20dB")
    assert band_file_names(sc, b) == ("SV_UPA_training_128_measurements_0to20dB.mat",
                                      "SV_UPA_validation_128_measurements_0to20dB.mat")      # colored: as before
    assert band_file_names(dict(sc, noise_model="white"), b) == ("SV_UPA_training_128_measurements_0to20dB_white.mat",
                                                                "SV_UPA_validation_128_measurements_0to20dB_white.mat")
    assert scenario_run_name(sc) == "SV_UPA" and scenario_run_name(dict(sc, noise_model="white"), True) == "SV_UPA_white_smoke"


def test_generators_paired_and_white(tiny):
    x = sio.loadmat(tiny / "SV_UPA_validation_channel.mat")["x"]
    snr = [10.0, 20.0]
    yc = measurements.gen_measurements_py(x, A, snr, np.random.RandomState(5))
    yc2 = measurements.gen_measurements_py(x, A, snr, np.random.RandomState(5), noise_model="colored")
    yw = measurements.gen_measurements_py(x, A, snr, np.random.RandomState(5), noise_model="white")
    assert np.array_equal(yc, yc2)
    T = noise.whitening_map(A)
    np.testing.assert_allclose(yw - A @ x, T @ (yc - A @ x), atol=1e-12)              # the same draw, whitened
    assert noise.apply(yc, x, A, "colored") is yc
    # test noise (evaluate.py): paired, and the MATLAB-style draw itself is unchanged
    tc, nc = measurements.gen_measurements_matlab(x, A, 10.0, NumpyMatlabRNG(7), return_noise=True)
    tw, nw = measurements.gen_measurements_matlab(x, A, 10.0, NumpyMatlabRNG(7), return_noise=True, noise_model="white")
    assert np.array_equal(nc, nw) and np.array_equal(tc, A @ (x + nc))
    np.testing.assert_allclose(tw, A @ x + T @ A @ nc, atol=1e-12)


def test_white_noise_statistics():
    rng = np.random.default_rng(0)
    K, s2 = 40000, 0.1
    x = np.zeros((N, K))
    n = np.sqrt(s2 / 2) * (rng.standard_normal((N, K)) + 1j * rng.standard_normal((N, K)))
    w = noise.apply(A @ (x + n), x, A, "white")
    C = (w @ w.conj().T) / K
    c = noise.white_scale(A)
    assert np.max(np.abs(np.diag(C).real / (s2 * c) - 1)) < 0.05                       # variance c s2 on every entry
    assert np.max(np.abs(C - np.diag(np.diag(C)))) / (s2 * c) < 0.05                    # uncorrelated
    Cc = ((A @ n) @ (A @ n).conj().T) / K
    lam = np.linalg.eigvalsh(Cc.real)
    assert lam.max() / lam.min() > 10                                                   # the colored one is not white
    assert np.trace(C).real == pytest.approx(np.trace(Cc).real, rel=0.02)              # same mean power


def test_online_validation_paired(tiny):
    x = sio.loadmat(tiny / "SV_UPA_validation_channel.mat")["x"]
    yc, s2c, snc = online_validation(x, A, 0, 20, 11)
    yw, s2w, snw = online_validation(x, A, 0, 20, 11, noise_model="white")
    assert np.array_equal(s2c, s2w) and np.array_equal(snc, snw)
    np.testing.assert_allclose(yw - A @ x, noise.whitening_map(A) @ (yc - A @ x), atol=1e-12)


def test_band_files_white_from_colored_twin(tmp_path, tiny):
    for f in ("CSmatrix.mat", "SV_UPA_training_channel.mat", "SV_UPA_validation_channel.mat"):
        shutil.copy(tiny / f, tmp_path / f)
    sc = dict(name="SV_UPA", dataset="SV", array="UPA", M=128, n_snr_chunks=10, noise_model="white")
    band = dict(name="0to20dB", train=[0, 20], val_range=[0, 20], n_snr_chunks=100)
    sc_c0 = {k: v for k, v in sc.items() if k != "noise_model"}
    ensure_band_files(sc_c0, band, 0, tmp_path, A, band_file_names(sc_c0, band), verbose=False, only="train")
    fn = band_file_names(sc, band)
    written = ensure_band_files(sc, band, 0, tmp_path, A, fn, verbose=False)
    assert [p.name for p in written] == list(fn)
    x = sio.loadmat(tmp_path / "SV_UPA_training_channel.mat")["x"]
    dc = sio.loadmat(tmp_path / "SV_UPA_training_128_measurements_0to20dB.mat")
    dw = sio.loadmat(tmp_path / fn[0])
    np.testing.assert_allclose(dw["y"], noise.apply(dc["y"], x, A, "white"), atol=1e-10)   # twin of the existing file
    assert np.array_equal(dw["snr_db"], dc["snr_db"]) and str(dw["noise_model"][0]) == "white"
    assert (tmp_path / (fn[0] + ".snr.json")).exists()
    # validation: no colored twin -> generated with the same RNG as the colored file would be (paired with it)
    xv = sio.loadmat(tmp_path / "SV_UPA_validation_channel.mat")["x"]
    sc_c = {k: v for k, v in sc.items() if k != "noise_model"}
    fc = band_file_names(sc_c, band)
    ensure_band_files(sc_c, band, 0, tmp_path, A, fc, verbose=False, only="val")
    vc, vw = sio.loadmat(tmp_path / fc[1])["y"], sio.loadmat(tmp_path / fn[1])["y"]
    np.testing.assert_allclose(vw, noise.apply(vc, xv, A, "white"), atol=1e-10)


def _nf(noise_model, T=2, **kw):
    torch.manual_seed(0)
    return build_model(dict(model="NF-LADMM", T=T, array="UPA", denoiser="radial", noise_model=noise_model,
                            precision="float64", **kw), A)


def test_nfladmm_white_step_is_exact_prox():
    net = _nf("white")
    with torch.no_grad():
        net.rho_raw.copy_(torch.tensor([0.3, -1.1], dtype=torch.float64))
    g = torch.Generator().manual_seed(1)
    v = torch.complex(torch.randn(5, N, generator=g, dtype=torch.float64), torch.randn(5, N, generator=g, dtype=torch.float64))
    y = torch.complex(torch.randn(5, M, generator=g, dtype=torch.float64), torch.randn(5, M, generator=g, dtype=torch.float64))
    At = torch.as_tensor(A)
    res = y - v @ At.T.to(v.dtype)
    for t in range(2):
        beta = torch.sigmoid(net.rho_raw[t])
        rho = 1 / beta - 1
        c = noise.white_scale(A)
        x = net.data_step(v, res, t)
        # closed form x = v + A^T (AA^T + rho c I)^-1 r
        ref = v + (torch.linalg.solve((At @ At.T + rho * c * torch.eye(M, dtype=torch.float64)).to(res.dtype), res.T).T
                   @ At.to(res.dtype))
        torch.testing.assert_close(x, ref, atol=1e-10, rtol=0)
        # optimality of argmin 1/(2c) ||y - A x||^2 + rho/2 ||x - v||^2  (white, normalised by s2)
        yv = res + v @ At.T.to(v.dtype)
        grad = -(yv - x @ At.T.to(x.dtype)) @ At.to(x.dtype) / c + rho * (x - v)
        assert float(grad.detach().abs().max()) < 1e-9
    col = _nf("colored")
    with torch.no_grad():
        col.rho_raw.copy_(net.rho_raw)
    x = col.data_step(v, res, 0)
    rho = 1 / torch.sigmoid(col.rho_raw[0]) - 1
    yv = res + v @ At.T.to(v.dtype)
    W = torch.linalg.inv(At @ At.T).to(res.dtype)                     # colored: Sigma~ = AA^T
    grad = -((yv - x @ At.T.to(x.dtype)) @ W) @ At.to(x.dtype) + rho * (x - v)
    assert float(grad.detach().abs().max()) < 1e-9


def test_nfladmm_state_unchanged_and_switchable():
    w, c = _nf("white"), _nf("colored")
    assert set(w.state_dict()) == set(c.state_dict()) and not any(k.startswith("eig_") for k in w.state_dict())
    assert w.n_params() == c.n_params() and w.cfg["noise_model"] == "white" and c.cfg["noise_model"] == "colored"
    old = {k: v for k, v in c.state_dict().items()}
    cfg = {k: v for k, v in c.cfg.items() if k != "noise_model"}              # a v22 file: no noise_model in cfg
    m = build_model(cfg, A)
    m.load_state_dict({k: v.to(m.state_dict()[k].dtype) for k, v in old.items()})
    assert m.noise_model == "colored"
    y = torch.complex(torch.randn(M, 4, dtype=torch.float64), torch.randn(M, 4, dtype=torch.float64))
    with torch.no_grad():
        a0 = c(y, s2=0.1)
        c.noise_model = "white"                                               # evaluate.py --nf-step matched
        a1 = c(y, s2=0.1)
        w.load_state_dict(c.state_dict())
        a2 = w(y, s2=0.1)
    assert not torch.allclose(a0, a1) and torch.allclose(a1, a2)


def test_trainer_online_white(tiny):
    xt = sio.loadmat(tiny / "SV_UPA_training_channel.mat")["x"][:, :64]
    xv = xt[:, :16]
    net = _nf("white", T=1)
    yv, s2v, _ = online_validation(xv, A, 0, 20, 3, noise_model="white")
    cfg = TrainConfig.from_dict(dict(schedule="steps", steps=1, noise="online", snr_range=[0, 20], batch_size=8, seed=4))
    out = {}
    for nm in ("colored", "white"):
        tr = Trainer(net, cfg, xt, None, xv, yv, s2_val=s2v, noise_model=nm)
        out[nm] = tr._batch()
    (yc, x, s2), (yw, x2, s2b) = out["colored"], out["white"]
    assert torch.equal(x, x2) and torch.equal(s2, s2b)                       # the same channels and SNRs
    Ax = torch.complex(net.A @ x.real, net.A @ x.imag)
    T = torch.as_tensor(noise.whitening_map(A), dtype=net.rdtype)
    d = yc - Ax
    torch.testing.assert_close(yw - Ax, torch.complex(T @ d.real, T @ d.imag), atol=1e-6, rtol=1e-6)


@pytest.mark.slow
def test_pipeline_white(tmp_path, tiny):
    """train.py --noise-model white (smoke) -> <scenario>_white_smoke; evaluate.py auto / --nf-step matched; compare_noise."""
    data = tmp_path / "SVT_smoke"
    shutil.copytree(tiny, data)
    yml = tmp_path / "svt.yaml"
    yml.write_text(f"name: SVT\ndataset: SV\narray: UPA\nN1: 16\nN2: 16\nL: 3\nM: 128\ndata_dir: {tmp_path / 'SVT'}\n"
                   "seed: 2026\nn_snr_chunks: 10\nbands:\n  - {name: 0to20dB, train: [0, 20], val_range: [0, 20], "
                   "n_snr_chunks: 100, use_for: [0, 20.01]}\ntest_snr_db: [0, 10, 20]\n", encoding="utf-8")
    out = tmp_path / "outputs"
    py = sys.executable
    for nm in ("colored", "white"):
        subprocess.run([py, "scripts/train.py", "--scenario", yml, "--train", "configs/train/nfladmm_c.yaml", "--smoke",
                        "--tag", "NF-C", "--out", out, "--device", "cpu", "--noise-model", nm, "--noise", "file", "--set", "model.T=2"],
                       cwd=ROOT, check=True, capture_output=True, text=True)
    assert (out / "SVT_smoke/NF-C/NF-C_0to20dB.pt").exists() and (out / "SVT_white_smoke/NF-C/NF-C_0to20dB.pt").exists()
    for f in ("training", "validation"):                                        # file mode: both sets, paired
        fc, fw = (data / f"SV_UPA_{f}_128_measurements_0to20dB{s}.mat" for s in ("", "_white"))
        x = sio.loadmat(data / f"SV_UPA_{f}_channel.mat")["x"]
        np.testing.assert_allclose(sio.loadmat(fw)["y"], noise.apply(sio.loadmat(fc)["y"], x, A, "white"), atol=1e-10)
    log = json.loads((out / "SVT_white_smoke/NF-C/train_log.json").read_text())
    assert log["scenario"]["noise_model"] == "white" and log["bands"]["0to20dB"]["noise_model"] == "white"
    assert "noise_model" not in json.loads((out / "SVT_smoke/NF-C/train_log.json").read_text())["scenario"]
    ck = torch.load(out / "SVT_white_smoke/NF-C/NF-C_0to20dB.pt", weights_only=False)
    assert ck["cfg"]["noise_model"] == "white"
    r = subprocess.run([py, "scripts/evaluate.py", "--scenario", yml, "--runs", out / "SVT_white_smoke", "--classical", "",
                        "--device", "cpu"], cwd=ROOT, check=True, capture_output=True, text=True)
    assert "test noise: white" in r.stdout
    assert json.loads((out / "SVT_white_smoke/eval_log.json").read_text())["noise_model"] == "white"
    subprocess.run([py, "scripts/compare_noise.py", "--scenarios", yml, "--outputs", out, "--results", tmp_path / "res",
                    "--smoke", "--nf-matched", "--no-authors", "--device", "cpu"], cwd=ROOT, check=True,
                   capture_output=True, text=True)
    rows = list(__import__("csv").DictReader(open(tmp_path / "res/noise_comparison/noise_summary.csv")))
    combos = {(r["method"], r["trained_on"], r["data_step"], r["tested_on"]) for r in rows}
    for tr in ("colored", "white"):
        for te in ("colored", "white"):
            assert ("NF-C", tr, "trained", te) in combos
    assert ("NF-C", "colored", "matched", "white") in combos and ("NF-C", "white", "matched", "colored") in combos
    assert (tmp_path / "res/noise_comparison/noise_comparison.png").exists()

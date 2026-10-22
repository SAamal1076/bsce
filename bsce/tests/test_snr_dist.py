"""--snr-dist fair (v26): every training sample its own SNR, spread evenly over the band, instead of SNR chunks.

1. fair_snr: stratified (one SNR in each of K equal parts, random order) and on a grid (equal counts);
2. gen_measurements_percol: the noise of every column has the power of its SNR (white: the same draw whitened);
3. the measurement files: names <band>_fair[<step>].mat, one SNR per column, validation at equally spaced SNRs;
   without the option every name and band is as before;
4. the online minibatches of the trainer (and of the fine-tune) spread evenly; the default stream is unchanged;
5. train.py / finetune_bands.py / run_suite.py / the workbench: the option, the names, the errors.
"""
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
import scipy.io as sio
import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from bsce.config import band_file_names  # noqa: E402
from bsce.data.files import apply_band_options, bands_from_edges, ensure_band_files, fair_validation, snr_dist_suffix  # noqa: E402
from bsce.data.measurements import evenly_spaced_snr, fair_snr, gen_measurements_percol  # noqa: E402
from bsce.data import noise as noise_models  # noqa: E402
from bsce.models import build_model  # noqa: E402
from bsce.train.trainer import Trainer, TrainConfig  # noqa: E402

M, N = 32, 64
A = np.random.default_rng(0).choice([-1.0, 1.0], (M, N)) / np.sqrt(M)


def _x(K, seed):
    r = np.random.default_rng(seed)
    x = np.zeros((N, K), complex)
    for k in range(K):
        idx = r.choice(N, 3, replace=False)
        x[idx, k] = (r.standard_normal(3) + 1j * r.standard_normal(3)) / np.sqrt(2) * np.sqrt(N / 3)
    return x


# ------------------------------------------------------------------------------------------------ 1. the draw
def test_fair_snr_stratified():
    K, lo, hi = 1000, 0.0, 20.0
    s = fair_snr(np.random.RandomState(3), lo, hi, K)
    ss = np.sort(s)
    k = np.arange(K)
    assert np.all(ss >= lo + (hi - lo) * k / K) and np.all(ss < lo + (hi - lo) * (k + 1) / K)   # one per part
    assert np.array_equal(np.histogram(s, bins=20, range=(lo, hi))[0], np.full(20, 50))       # flat histogram
    assert np.unique(s).size == K and not np.all(np.diff(s) > 0)                              # continuous, random order
    assert np.array_equal(s, fair_snr(np.random.RandomState(3), lo, hi, K))                   # reproducible
    s2 = fair_snr(np.random.RandomState(0), -10, 30, 80000)
    assert s2.min() >= -10 and s2.max() < 30
    assert np.array_equal(np.histogram(s2, bins=400, range=(-10, 30))[0], np.full(400, 200))  # 0.1 dB bins: all 200


def test_fair_snr_grid():
    s = fair_snr(np.random.RandomState(1), 0, 20, 80000, step=0.2)
    v, c = np.unique(np.round(s, 9), return_counts=True)
    assert v.size == 101 and v[0] == 0 and abs(v[-1] - 20) < 1e-9 and np.allclose(np.diff(v), 0.2)
    assert c.max() - c.min() <= 1 and c.sum() == 80000                 # 80000 / 101: 792 or 793 each
    s = fair_snr(np.random.RandomState(1), 0, 10, 2000, step=0.5)      # 21 values, 2000 not divisible
    _, c = np.unique(np.round(s, 9), return_counts=True)
    assert c.size == 21 and c.max() - c.min() == 1
    with pytest.raises(ValueError):
        fair_snr(np.random.RandomState(1), 10, 0, 100)


def test_evenly_spaced():
    assert np.allclose(evenly_spaced_snr(0, 20, 4), [2.5, 7.5, 12.5, 17.5])
    assert np.allclose(evenly_spaced_snr(-10, 30, 100)[[0, -1]], [-9.8, 29.8])


# ------------------------------------------------------------------------------------------------ 2. per column
def test_percol_noise_power_and_white():
    K = 20000
    x = np.zeros((N, K), complex)
    snr = np.repeat([0.0, 10.0, 20.0, 30.0], K // 4)
    y = gen_measurements_percol(x, A, snr, np.random.RandomState(0))
    fro = np.sum(A ** 2)                                     # E ||A n||^2 = sigma2 ||A||_F^2
    for j, s in enumerate([0.0, 10.0, 20.0, 30.0]):
        p = np.mean(np.sum(np.abs(y[:, j * K // 4:(j + 1) * K // 4]) ** 2, 0)) / fro
        assert p == pytest.approx(10 ** (-s / 10), rel=0.03)
    # white: exactly the colored draw whitened, block by block
    x = _x(500, 1)
    s = fair_snr(np.random.RandomState(2), 0, 20, 500)
    yc = gen_measurements_percol(x, A, s, np.random.RandomState(5), block=128)
    yw = gen_measurements_percol(x, A, s, np.random.RandomState(5), noise_model="white", block=128)
    assert np.allclose(yw, noise_models.apply(yc, x, A, "white"))
    assert np.allclose(yc, gen_measurements_percol(x, A, s, np.random.RandomState(5), block=4000))   # block-free


# ------------------------------------------------------------------------------------------------ 3. the files
def _scenario(tmp_path, Kt=2000, Kv=400):
    d = tmp_path / "data"
    d.mkdir()
    sio.savemat(d / "SV_ULA_training_channel.mat", {"x": _x(Kt, 10)})
    sio.savemat(d / "SV_ULA_validation_channel.mat", {"x": _x(Kv, 11)})
    return dict(name="T", dataset="SV", array="ULA", M=M, n_snr_chunks=10, seed=3), d


def test_band_names_unchanged_by_default():
    for edges in ([0, 20], [0, 10, 20], [-10, 30]):
        b = bands_from_edges(edges)
        assert apply_band_options(b, snr_dist=None) == b and apply_band_options(b, snr_dist="chunks") == b
    sc = dict(dataset="SV", array="ULA", M=128)
    assert band_file_names(sc, bands_from_edges([0, 20])[0]) == ("SV_ULA_training_128_measurements_0to20dB.mat",
                                                                  "SV_ULA_validation_128_measurements_0to20dB.mat")
    assert snr_dist_suffix(None) == "" and snr_dist_suffix("fair") == "fair" and snr_dist_suffix("fair", 0.2) == "fair0p2"
    with pytest.raises(ValueError):
        snr_dist_suffix("chunks", 0.2)
    with pytest.raises(ValueError):
        apply_band_options(bands_from_edges([0, 20]), chunks=50, snr_dist="fair")


def test_fair_files(tmp_path):
    sc, d = _scenario(tmp_path)
    band = apply_band_options(bands_from_edges([-10, 30]), snr_dist="fair")[0]
    ftr, fva = band_file_names(sc, band)
    assert ftr == f"SV_ULA_training_{M}_measurements_m10to30dB_fair.mat"
    assert fva == f"SV_ULA_validation_{M}_measurements_m10to30dB_fair.mat"
    ensure_band_files(sc, band, 0, d, A, (ftr, fva), verbose=False)
    st = sio.loadmat(d / ftr)["snr_db"].ravel()
    assert st.size == 2000 and np.unique(st).size == 2000 and st.min() >= -10 and st.max() < 30
    assert np.array_equal(np.histogram(st, bins=40, range=(-10, 30))[0], np.full(40, 50))     # 1 dB bins: 50 each
    info = json.loads(Path(str(d / ftr) + ".snr.json").read_text())
    assert info["snr_dist"] == "fair" and info["distinct"] == 2000 and info["counts_20_equal_bins"] == [100] * 20
    sv = sio.loadmat(d / fva)["snr_db"].ravel()                     # validation: 100 equally spaced SNRs, 4 each
    assert np.allclose(np.unique(sv), evenly_spaced_snr(-10, 30, 100)) and np.all(np.unique(sv, return_counts=True)[1] == 4)
    assert np.allclose(sv[:4], -9.8) and np.allclose(sv[-4:], 29.8)              # equal chunks, in order (val_groups)
    # the grid
    bg = apply_band_options(bands_from_edges([0, 20]), snr_dist="fair", snr_step=0.5)[0]
    fg = band_file_names(sc, bg)
    assert fg[0].endswith("0to20dB_fair0p5.mat") and fg[1].endswith("0to20dB_fair.mat")
    ensure_band_files(sc, bg, 0, d, A, fg, verbose=False, only="train")
    v, c = np.unique(np.round(sio.loadmat(d / fg[0])["snr_db"].ravel(), 9), return_counts=True)
    assert v.size == 41 and c.max() - c.min() <= 1
    # white: the colored fair file whitened, the same SNRs
    scw = dict(sc, noise_model="white")
    fw = band_file_names(scw, band)
    assert fw[0].endswith("_fair_white.mat")
    ensure_band_files(scw, band, 0, d, A, fw, verbose=False)
    dw, dc = sio.loadmat(d / fw[0]), sio.loadmat(d / ftr)
    xt = sio.loadmat(d / "SV_ULA_training_channel.mat")["x"]
    assert np.array_equal(dw["snr_db"], dc["snr_db"]) and np.allclose(dw["y"], noise_models.apply(dc["y"], xt, A, "white"))


def test_fair_with_margin_and_authors_bands():
    b = apply_band_options(bands_from_edges([0, 10, 20]), margin=2, snr_dist="fair")
    assert [x["train_name"] for x in b] == ["m2to12dB_fair", "8to22dB_fair"]
    assert [x["val_name"] for x in b] == ["0to10dB_fair", "10to20dB_fair"]
    authors = [dict(name="0to10dB", train=[0, 10], val_snr=[10], n_snr_chunks=10)]       # validation at 10 dB: kept
    a = apply_band_options(authors, snr_dist="fair", default_chunks=10)[0]
    assert a["train_name"] == "0to10dB_fair" and "val_name" not in a and a["val_snr"] == [10]
    a = apply_band_options(authors, val="range", snr_dist="fair", default_chunks=10)[0]
    assert a["val_name"] == "0to10dB_valrange_fair"


def test_fair_validation():
    x = _x(300, 2)
    y, s2, snr = fair_validation(x, A, 0, 20, 7, n=100)
    assert np.unique(snr).size == 100 and np.allclose(np.unique(snr), evenly_spaced_snr(0, 20, 100))
    y, s2, snr = fair_validation(x, A, 0, 20, 7, n=7)              # 300 not divisible by 7 -> 6 SNRs of 50
    assert np.unique(snr).size == 6


# ------------------------------------------------------------------------------------------------ 4. online
def _trainer(**kw):
    tc = TrainConfig.from_dict(dict(schedule="steps", steps=2, noise="online", snr_range=[0, 20], verbose=False, **kw))
    x = _x(50, 1)
    return Trainer(build_model(dict(model="NF-LADMM", T=1, array="ULA", N1=N, N2=1, denoiser="radial"), A), tc,
                   x, None, x, A @ x, s2_val=np.full(50, 0.1))


def test_online_batches():
    t = _trainer(snr_dist="fair")
    s = t._batch_snr(0.0, 20.0, 400).numpy()
    assert np.array_equal(np.histogram(s, bins=400, range=(0, 20))[0], np.ones(400))        # one SNR per 0.05 dB
    assert not np.all(np.diff(s) > 0)
    t = _trainer(snr_dist="fair", snr_step=0.2)
    v, c = np.unique(np.round(t._batch_snr(0.0, 20.0, 400).numpy(), 9), return_counts=True)
    assert set(np.round(v / 0.2).astype(int)) <= set(range(101)) and c.max() - c.min() <= 1 and c.sum() == 400
    t = _trainer()                                                # default: the stream of before (i.i.d. U)
    g = torch.Generator(device="cpu").manual_seed(0)
    assert torch.equal(t._batch_snr(0.0, 20.0, 64), 20.0 * torch.rand(64, generator=g, dtype=torch.float64))


# ------------------------------------------------------------------------------------------------ 5. scripts
def _yaml(tmp_path):
    d = tmp_path / "data" / "T"
    d.mkdir(parents=True)
    sio.savemat(d / "CSmatrix.mat", {"A": A})
    for k, (part, K) in enumerate((("training", 400), ("validation", 100), ("test", 100))):
        sio.savemat(d / f"SV_ULA_{part}_channel.mat", {"x": _x(K, 20 + k)})
    y = tmp_path / "t.yaml"
    y.write_text(f"name: T\ndataset: SV\narray: ULA\nN1: {N}\nN2: 1\nL: 3\nM: {M}\ndata_dir: {d}\nseed: 3\n"
                 "n_snr_chunks: 10\nbands:\n  - {name: 0to20dB, train: [0, 20], val_range: [0, 20], n_snr_chunks: 100, "
                 "use_for: [0, 20.01]}\ntest_snr_db: [0, 10, 20]\n", encoding="utf-8")
    return y, d


def _run(*args, check=True):
    return subprocess.run([sys.executable, *map(str, args)], cwd=ROOT, capture_output=True, text=True, check=check)


def test_train_py_and_finetune_fair(tmp_path):
    y, d = _yaml(tmp_path)
    out = tmp_path / "outputs"
    r = _run("scripts/train.py", "--scenario", y, "--method", "ladmm2", "--snr-dist", "fair", "--snr-step", "0.5",
             "--out", out, "--device", "cpu", "--set", "train.maxit=20", "train.better_wait=10", "model.init_precision=null")
    run = out / "T" / "LADMM-T2_fair0p5_maxit-20_better-wait-10_init-precision-null"
    assert (run / f"{run.name}_0to20dB.pt").exists(), r.stdout[-2000:]
    assert "one SNR per sample, spread evenly on a 0.5 dB grid" in r.stdout
    assert (d / f"SV_ULA_training_{M}_measurements_0to20dB_fair0p5.mat").exists()
    assert (d / f"SV_ULA_validation_{M}_measurements_0to20dB_fair.mat").exists()
    assert not (d / f"SV_ULA_training_{M}_measurements_0to20dB.mat").exists()            # the chunk file not needed
    assert "fair: one SNR per sample, spread evenly over the band, grid step 0.5 dB" in (run / "README.txt").read_text()
    r = _run("scripts/make_sv_measurements.py", "--scenario", y, "--snr", "0", "20", "--snr-dist", "fair", "--snr-step", "0.5")
    assert r.stdout.count("keep") == 2                                # the same files as train.py (made ahead)
    # online noise (NF-LADMM-C, its own protocol): fair minibatches, equally spaced validation
    r = _run("scripts/train.py", "--scenario", y, "--method", "nfladmmc1", "--protocol", "paper", "--snr-dist", "fair",
             "--out", out, "--device", "cpu", "--set", "train.steps=20", "train.val_every=10")
    assert "spread evenly in every minibatch" in r.stdout and "100 equally spaced SNRs in [0,20] dB" in r.stdout
    assert (out / "T" / "NF-LADMM-C-T1_paper_fair_steps-20_val-every-10").exists()
    # fine-tune
    r = _run("scripts/finetune_bands.py", "--from", run, "--iters", "30", "--batch", "16", "--val-every", "10",
             "--device", "cpu", "--snr-dist", "fair")
    ft = out / "T" / f"{run.name}_ft_0-10-20dB_30it_batch16_valevery10_fair"
    assert (ft / f"{ft.name}_10to20dB.pt").exists(), r.stdout[-2000:]
    assert "100 equally spaced SNRs in [0,10] dB" in r.stdout
    lg = json.loads((ft / "finetune_log.json").read_text())
    assert lg["setting"]["snr_dist"] == "fair"
    assert "spread evenly over the band in every minibatch" in (ft / "README.txt").read_text()
    r = _run("scripts/finetune_bands.py", "--from", run, "--snr-step", "0.2", "--dry-name", check=False)
    assert r.returncode != 0 and "--snr-step needs --snr-dist fair" in r.stderr


def test_suite_fair(tmp_path):
    r = _run("scripts/run_suite.py", "--scenarios", "sv_upa", "--methods", "ladmm6", "--train-snr", "-10", "30",
             "--ft-snr", "-10", "0", "10", "20", "30", "--snr-dist", "fair", "--dry-run")
    assert "--snr-dist fair --tag LADMM-T6_trm10-30_fair " in r.stdout
    assert "LADMM-T6_trm10-30_fair_ft_m10-0-10-20-30dB_20k_fair" in r.stdout
    assert r.stdout.count("--snr-dist fair") >= 2                                        # training and fine-tune
    r = _run("scripts/run_suite.py", "--scenarios", "sv_upa", "--methods", "ladmm6", "--snr-dist", "fair",
             "--snr-chunks", "50", "--dry-run", check=False)
    assert r.returncode != 0 and "no effect with --snr-dist fair" in r.stderr
    r = _run("scripts/run_suite.py", "--scenarios", "sv_upa", "--methods", "ladmm6", "--snr-step", "0.2", "--dry-run",
             check=False)
    assert r.returncode != 0 and "--snr-step needs --snr-dist fair" in r.stderr
    r = _run("scripts/run_suite.py", "--scenarios", "sv_upa", "--methods", "ladmm6", "--dry-run")
    assert "--snr-dist" not in r.stdout and "--tag LADMM-T6 " in r.stdout                  # default: as before

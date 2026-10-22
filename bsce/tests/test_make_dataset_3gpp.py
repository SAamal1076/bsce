"""3GPP data sets: scripts/make_dataset.py --dataset 3GPP (raw files of scripts/sionna_make_H.py or matlab/quadriga_make_H.m),
scripts/check_3gpp.py, scripts/compare_3gpp_generators.py, and the two generators when Sionna / Octave + QuaDRiGa are installed.

1. make_dataset: x = U0^H h with ||x||^2 = N per channel (the DeepMIMO lens convention), held-out drops (training =
   the first users, validation and test = the last drops, no drop shared), --authors-bands = the bands of
   sv_*.yaml / deepmimo_*.yaml (10 chunks, validation at 10 dB), the CS matrix of SV; --test random keeps the same
   users; a raw file of another array is refused
2. check_3gpp: LSPs drawn from 38.901 Table 7.5-6 and LOS states drawn from Table 7.4.2-1 come out with offset ~ 0,
   log-sd ~ table, |z| of the LOS fraction < 3
3. quadriga_make_H (Octave + QuaDRiGa, else skipped; QuaDRiGa from $QUADRIGA_SRC or reference/quadriga_src) and
   sionna_make_H.py (Sionna, else skipped):
   a part made alone equals the same part made in sequence; users lie in the 120-degree sector, d2D >= 10 m
4. compare_3gpp_generators.py runs on two data sets
Needs reference/gmlamp/CE_SV/CSmatrix256128.mat for 1 and 4.
"""
import importlib.util
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
import scipy.io as sio

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))
from bsce.config import load_yaml  # noqa: E402

CS = ROOT / "reference/gmlamp/CE_SV/CSmatrix256128.mat"


def fake_parts(folder, geo=(1, 16, 16), drops=6, per_drop=20, per_part=60, seed=0):
    """Files in the format of matlab/quadriga_make_H.m (random channels, LSPs and LOS from the 38.901 tables)."""
    from check_3gpp import UMI, p_los_umi, table
    r = np.random.default_rng(seed)
    K = drops * per_drop
    H = (r.standard_normal((256, K)) + 1j * r.standard_normal((256, K))) * r.uniform(1e-6, 1e-4, K)
    ang = r.uniform(-np.pi / 3, np.pi / 3, K)
    d = np.sqrt(r.uniform(10 ** 2, 133 ** 2, K))
    loc = np.c_[d * np.cos(ang), d * np.sin(ang), np.full(K, 1.5)]
    los = r.random(K) < p_los_umi(d)
    lsp = np.zeros((K, 7))
    for state, m in (("LOS", los), ("NLOS", ~los)):
        for c, name in enumerate(["DS", "ASD", "ASA", "ZSD", "ZSA"]):
            mu, sg = table(28.0, state, name, d[m], 10.0, 1.5)
            lsp[m, c] = 10 ** (mu + sg * r.standard_normal(m.sum()))
        lsp[m, 5] = r.normal(*UMI["LOS"]["K"], m.sum()) if state == "LOS" else -100
    drop = np.repeat(np.arange(1, drops + 1), per_drop)
    params = dict(generator="test", array="UPA" if geo == (1, 16, 16) else "ULA", num_ant_x=geo[0], num_ant_y=geo[1],
                  num_ant_z=geo[2], scenario="3GPP_38.901_UMi", fc=28e9, element="3gpp", h_bs=10.0, h_ut=1.5,
                  isd=200.0, d_min=10.0, drop_users=per_drop, seed=2026)
    files = []
    for p, k0 in enumerate(range(0, K, per_part), 1):
        ii = slice(k0, min(K, k0 + per_part))
        f = folder / f"H_fake_part{p:03d}.mat"
        sio.savemat(f, dict(H=H[:, ii], loc=loc[ii], los=los[ii, None].astype(float), indoor=np.zeros((ii.stop - k0, 1)),
                            drop=drop[ii, None].astype(float), lsp=lsp[ii], lsp_real=np.full((ii.stop - k0, 5), np.nan),
                            params=params))
        files.append(f)
    return H, files


def run(args):
    return subprocess.run([sys.executable, str(ROOT / "scripts/make_dataset.py"), *map(str, args)], capture_output=True,
                          text=True, cwd=ROOT)


@pytest.mark.skipif(not CS.exists(), reason="needs reference/gmlamp (CS matrix)")
def test_make_3gpp_dataset(tmp_path):
    from deepmimo_convert import u0_1d
    H, _ = fake_parts(tmp_path)
    name = "tst_3GPP_UPA"
    yml = ROOT / "configs/scenarios" / f"{name.lower()}.yaml"
    try:
        out = tmp_path / name
        r = run(["--dataset", "3GPP", "--array", "UPA", "--input", str(tmp_path / "H_fake_part*.mat"), "--n-train", 60,
                 "--n-val", 20, "--n-test", 20, "--authors-bands", "--name", name, "--out-dir", out])
        assert r.returncode == 0, r.stdout + r.stderr
        x = {k: sio.loadmat(out / f"3GPP_UPA_{k}_channel.mat")["x"] for k in ("training", "validation", "test")}
        U0 = np.kron(u0_1d(16), u0_1d(16))
        ref = U0.conj().T @ H
        ref *= np.sqrt(256 / np.sum(np.abs(ref) ** 2, 0))
        assert np.allclose(x["training"], ref[:, :60]) and np.allclose(x["validation"], ref[:, 80:100])
        assert np.allclose(x["test"], ref[:, 100:120])                              # the last drop
        assert np.allclose(np.sum(np.abs(x["test"]) ** 2, 0), 256)
        info = json.loads((out / "dataset_info.json").read_text())
        assert set(info["drops_shared_between_sets"].values()) == {0}
        sc = load_yaml(yml)
        assert sc["dataset"] == "3GPP" and sc["n_snr_chunks"] == 10 and sc["cs_matrix"].endswith("CSmatrix256128.mat")
        assert [(b["name"], b["val_snr"]) for b in sc["bands"]] == [("0to10dB", [10]), ("10to20dB", [10])]
        m = sio.loadmat(out / "3GPP_UPA_training_128_measurements_0to10dB.mat")
        assert m["y"].shape == (128, 60) and np.unique(np.round(m["snr_db"], 9)).size == 10
        v = sio.loadmat(out / "3GPP_UPA_validation_128_measurements_10to20dB.mat")
        assert np.allclose(v["snr_db"], 10)
        r = run(["--dataset", "3GPP", "--array", "UPA", "--input", str(tmp_path / "H_fake_part*.mat"), "--n-train", 60,
                 "--n-val", 20, "--n-test", 20, "--test", "random", "--no-measurements", "--name", name, "--out-dir",
                 out, "--force"])
        assert r.returncode == 0, r.stdout + r.stderr
        xr = np.concatenate([sio.loadmat(out / f"3GPP_UPA_{k}_channel.mat")["x"] for k in ("training", "validation", "test")], 1)
        assert xr.shape[1] == 100 and not np.allclose(xr[:, :60], x["training"]) and all(np.isclose(np.abs(ref[0]), v).any() for v in np.abs(xr[0]))
        r = run(["--dataset", "3GPP", "--array", "ULA", "--input", str(tmp_path / "H_fake_part*.mat"), "--n-train", 60,
                 "--n-val", 20, "--n-test", 20, "--name", name, "--out-dir", tmp_path / "other"])
        assert r.returncode != 0 and "(1, 16, 16)" in (r.stdout + r.stderr)
    finally:
        yml.unlink(missing_ok=True)


def test_check_3gpp_against_the_tables(tmp_path):
    from check_3gpp import UMI, p_los_umi, rstd, table
    r = np.random.default_rng(1)
    n = 20000
    d = np.sqrt(r.uniform(10 ** 2, 133 ** 2, n))
    los = r.random(n) < p_los_umi(d)
    exp = p_los_umi(d).mean()
    assert abs(los.mean() - exp) / np.sqrt(exp * (1 - exp) / n) < 3
    assert np.isclose(p_los_umi(18.0), 1) and np.isclose(p_los_umi(36.0), 0.5 + np.exp(-1) * 0.5)
    for state in ("LOS", "NLOS"):
        for name in ("DS", "ASD", "ASA", "ZSD", "ZSA"):
            mu, sg = table(28.0, state, name, d, 10.0, 1.5)
            lg = mu + sg * r.standard_normal(n)
            assert abs(np.median(lg - mu)) < 0.02 and abs(rstd(lg - mu) / np.median(sg) - 1) < 0.05
    mu, _ = table(28.0, "LOS", "DS", d[:1], 10.0, 1.5)
    assert np.isclose(10 ** mu[0] * 1e9, 32.3, atol=0.1)                 # UMi LOS DS median at 28 GHz: 32 ns
    mu, _ = table(28.0, "NLOS", "ASA", d[:1], 10.0, 1.5)
    assert np.isclose(10 ** mu[0], 49.3, atol=0.1)
    mu, _ = table(28.0, "LOS", "ZSD", np.array([0.0, 1000.0]), 10.0, 1.5)
    assert np.allclose(mu, [0.83 + 0.085, -0.21])
    assert UMI["LOS"]["K"] == (9.0, 5.0)
    mu, _ = table(28.0, "LOS", "DS", d[:1], 10.0, 1.5, "v19")
    assert np.isclose(10 ** mu[0] * 1e9, 28.6, atol=0.1)                 # Rel-19 update: 28.6 ns
    _, sg = table(28.0, "LOS", "ASD", d[:1], 10.0, 1.5, "v19")
    assert np.isclose(sg[0], 0.08 * np.log10(29) + 0.29)


def _quadriga():
    for p in (os.environ.get("QUADRIGA_SRC"), ROOT / "reference/quadriga_src"):
        if p and (Path(p) / "@qd_layout").is_dir():
            return Path(p)
    return None


@pytest.mark.skipif(shutil.which("octave") is None or _quadriga() is None, reason="needs Octave and QuaDRiGa")
def test_quadriga_make_H_octave(tmp_path):
    q = _quadriga()
    m = str(ROOT / "matlab").replace("\\", "/")
    seq, alone = (tmp_path / "seq" / "H").as_posix(), (tmp_path / "alone" / "H").as_posix()
    code = (f"addpath('{m}'); quadriga_make_H('{q.as_posix()}','{seq}','ULA_Y',24,'drop_users',6,'part_users',12);"
            f"quadriga_make_H('{q.as_posix()}','{alone}','ULA_Y',24,'drop_users',6,'part_users',12,'parts',2);")
    r = subprocess.run(["octave", "--no-gui", "--quiet", "--eval", code], capture_output=True, text=True, timeout=900)
    assert r.returncode == 0, r.stdout + r.stderr
    a = sio.loadmat(tmp_path / "seq" / "H_part002.mat")
    b = sio.loadmat(tmp_path / "alone" / "H_part002.mat")
    assert not (tmp_path / "alone" / "H_part001.mat").exists()
    assert np.array_equal(a["H"], b["H"]) and np.array_equal(a["los"], b["los"])
    assert a["H"].shape == (256, 12) and np.all(np.sum(np.abs(a["H"]) ** 2, 0) > 0)
    loc = np.r_[sio.loadmat(tmp_path / "seq" / "H_part001.mat")["loc"], a["loc"]]
    d = np.hypot(loc[:, 0], loc[:, 1])
    assert d.min() >= 10 and d.max() <= 2 * 200 / 3 + 1e-9 and np.all(np.abs(np.arctan2(loc[:, 1], loc[:, 0])) <= np.pi / 3 + 1e-9)
    assert np.allclose(loc[:, 2], 1.5) and int(np.squeeze(a["params"]["num_ant_y"][0, 0])) == 256
    assert np.array_equal(np.ravel(a["drop"]), np.repeat([3, 4], 6))


@pytest.mark.skipif(importlib.util.find_spec("sionna") is None, reason="needs Sionna (pip install sionna-no-rt==2.2.0)")
def test_sionna_make_H(tmp_path):
    def gen(out, *extra):
        return subprocess.run([sys.executable, str(ROOT / "scripts/sionna_make_H.py"), "--array", "ULA_Y", "--n", "24",
                               "--drop-users", "6", "--part-users", "12", "--out", str(out), *extra],
                              capture_output=True, text=True, cwd=ROOT, timeout=900)
    r = gen(tmp_path / "seq" / "H")
    assert r.returncode == 0, r.stdout + r.stderr
    r = gen(tmp_path / "alone" / "H", "--parts", "2")
    assert r.returncode == 0, r.stdout + r.stderr
    a, b = (sio.loadmat(tmp_path / s / "H_part002.mat") for s in ("seq", "alone"))
    assert not (tmp_path / "alone" / "H_part001.mat").exists()
    assert np.array_equal(a["H"], b["H"]) and np.array_equal(a["lsp"], b["lsp"])        # a part made alone = in sequence
    assert a["H"].shape == (256, 12) and np.all(np.sum(np.abs(a["H"]) ** 2, 0) > 0)
    assert np.array_equal(np.ravel(a["drop"]), np.repeat([3, 4], 6)) and np.isfinite(a["lsp"][:, :5]).all()
    d = np.hypot(a["loc"][:, 0], a["loc"][:, 1])
    assert d.min() >= 10 and d.max() <= 2 * 200 / 3 + 1e-9
    r = gen(tmp_path / "seq" / "H", "--spec", "19.2")
    assert r.returncode != 0 and "spec" in (r.stdout + r.stderr)                      # other settings: refused


@pytest.mark.skipif(not CS.exists(), reason="needs reference/gmlamp (CS matrix)")
def test_compare_generators(tmp_path):
    (tmp_path / "a").mkdir(); (tmp_path / "b").mkdir()
    fake_parts(tmp_path / "a", drops=4, per_drop=500, per_part=1000, seed=1)
    fake_parts(tmp_path / "b", drops=4, per_drop=500, per_part=1000, seed=2)
    r = subprocess.run([sys.executable, str(ROOT / "scripts/compare_3gpp_generators.py"),
                        "--case", "UPA", "one", str(tmp_path / "a" / "H_fake_part*.mat"),
                        "--case", "UPA", "two", str(tmp_path / "b" / "H_fake_part*.mat"), "--out", str(tmp_path / "o")],
                       capture_output=True, text=True, cwd=ROOT)
    assert r.returncode == 0, r.stdout + r.stderr
    res = json.loads((tmp_path / "o" / "generators.json").read_text())
    assert set(res["UPA"]) == {"one", "two"} and (tmp_path / "o" / "generators.png").exists()
    assert res["UPA"]["one"]["beams95_median"] > 150                     # i.i.d. Gaussian channels: not sparse

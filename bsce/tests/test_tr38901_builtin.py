"""The built-in 3GPP TR 38.901 generator (bsce/data/tr38901.py) and the data sets made with it
(scripts/make_dataset.py --dataset 3GPP without --input, scripts/make_3gpp_data.py, scripts/check_3gpp.py --builtin).

1. generator: a range of drops made alone equals the same drops made in sequence (seed (seed, drop)); the factorised
   array sum equals the direct sum over the element positions for UPA, ULA on y and ULA on z; users lie in the
   120-degree hexagonal sector with d2D >= 10 m; LOS fraction (Table 7.4.2-1) and the LSP medians (Table 7.5-6, v16.1
   and v19.2) agree with the tables; umi_los / umi_nlos force the state; the 38.901 element gives 8 dBi at boresight
2. make_dataset: lens x = U0^H h with ||x||^2 = N; validation and test from drops 1,000,001 and 2,000,001, no drop
   shared, and the same validation / test users for any training size; the meta file; --ula-axis; --sizes presets
3. make_3gpp_data: --dry-run commands of the presets, refusal of incomplete exact sizes, a small real run
4. check_3gpp --builtin on a small sample; run_suite --scenarios all 3gpp; the shipped 3gpp_*.yaml files
Parts 2 to 4 need reference/gmlamp/CE_SV/CSmatrix256128.mat.
"""
import argparse
import json
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
from bsce.data import tr38901  # noqa: E402

CS = ROOT / "reference/gmlamp/CE_SV/CSmatrix256128.mat"
need_cs = pytest.mark.skipif(not CS.exists(), reason="needs reference/gmlamp (CS matrix)")


def py(script, *args, timeout=900):
    return subprocess.run([sys.executable, str(ROOT / script), *map(str, args)], capture_output=True, text=True,
                          cwd=ROOT, timeout=timeout)


# ---------------------------------------------------------------- 1. generator
def test_drops_alone_equal_drops_in_sequence():
    seq = tr38901.generate(400, "ULA_Y", seed=7, drop_users=100)
    alone = tr38901.generate(200, "ULA_Y", seed=7, drop_users=100, first_drop=3)
    assert np.array_equal(seq["H"][:, 200:], alone["H"]) and np.array_equal(seq["lsp"][200:], alone["lsp"])
    assert np.array_equal(seq["drop"], np.repeat([1, 2, 3, 4], 100)) and np.array_equal(alone["drop"][[0, -1]], [3, 4])
    other = tr38901.generate(100, "ULA_Y", seed=8, drop_users=100)
    assert not np.allclose(other["H"], seq["H"][:, :100])
    assert np.all(np.isfinite(seq["H"])) and np.all(np.sum(np.abs(seq["H"]) ** 2, 0) > 0)


@pytest.mark.parametrize("array", ["UPA", "ULA_Y", "ULA"])
def test_array_sum_equals_direct_sum(array):
    r = np.random.default_rng(3)
    R = 41
    t, f = r.uniform(0, np.pi, R), r.uniform(-np.pi, np.pi, R)
    rhat = np.stack([np.sin(t) * np.cos(f), np.sin(t) * np.sin(f), np.cos(t)])
    g = r.standard_normal(R) + 1j * r.standard_normal(R)
    nv, nh = tr38901.GEOMETRY[array]
    direct = np.exp(2j * np.pi * tr38901.array_positions(array) @ rhat) @ g
    assert np.allclose(tr38901._array_sum(rhat, g, nv, nh), direct, atol=1e-10)
    pos = tr38901.array_positions(array)
    assert pos.shape == (256, 3) and np.allclose(pos.mean(0), 0)
    assert np.allclose(np.ptp(pos, 0), [0, 0.5 * (nh - 1), 0.5 * (nv - 1)])


def test_statistics_against_38901_tables():
    from check_3gpp import UMI, rstd, table
    for spec, ver in (("16.1", "v16"), ("19.2", "v19")):
        res = tr38901.generate(4000, "UPA", seed=11, drop_users=250, spec=spec)
        loc, los, lsp = res["loc"], res["los"].astype(bool), res["lsp"]
        d = np.hypot(loc[:, 0], loc[:, 1])
        ang = np.degrees(np.arctan2(loc[:, 1], loc[:, 0]))
        assert d.min() >= 10 and d.max() <= 2 * 200 / 3 + 1e-9 and np.abs(ang).max() <= 60 + 1e-9
        assert np.allclose(loc[:, 2], 1.5)
        p = tr38901.p_los_umi(d)
        assert abs(los.sum() - p.sum()) / np.sqrt(np.sum(p * (1 - p))) < 4             # z of the LOS count
        for state, m in (("LOS", los), ("NLOS", ~los)):
            for c, name in enumerate(["DS", "ASD", "ASA", "ZSD", "ZSA"]):
                mu, sg = table(28.0, state, name, d[m], 10.0, 1.5, ver)
                z = np.log10(lsp[m, c]) - mu
                assert abs(np.median(z)) < 0.06, (spec, state, name, np.median(z))
                assert 0.75 < rstd(z) / np.median(sg) < 1.25, (spec, state, name)
        assert abs(np.median(lsp[los, 5]) - UMI["LOS"]["K"][0]) < 1.0 and np.all(lsp[~los, 5] == -100)
        assert lsp[:, 1].max() <= 104 + 1e-9 and lsp[:, 3].max() <= 52 + 1e-9


def test_forced_state_and_element():
    a = tr38901.generate(60, "UPA", seed=1, drop_users=30, scenario="umi_los")
    b = tr38901.generate(60, "UPA", seed=1, drop_users=30, scenario="umi_nlos", element="omni")
    assert a["los"].all() and not b["los"].any()
    assert np.isclose(tr38901.element_field(np.array([90.0]), np.array([0.0]))[0] ** 2, 10 ** 0.8)
    assert np.isclose(tr38901.element_field(np.array([90.0]), np.array([180.0]))[0] ** 2, 10 ** -2.2)
    assert tr38901.element_field(np.array([30.0]), np.array([90.0]), "omni")[0] == 1
    with pytest.raises(ValueError):
        tr38901.generate(10, "UPA_8x8")
    with pytest.raises(ValueError):
        tr38901.generate(10, "UPA", scenario="uma")
    p = tr38901.params_of("ULA", 10, 2026, 250, spec="19.2")
    assert (p["num_ant_y"], p["num_ant_z"], p["table"]) == (1, 256, "v19")


# ---------------------------------------------------------------- 2. make_dataset
@need_cs
def test_make_dataset_builtin(tmp_path):
    from deepmimo_convert import u0_1d
    from make_dataset import DROP_START, SIZES, sizes
    assert sizes(argparse.Namespace(sizes="dong2019")) == (81000, 9000, 19000) == SIZES["dong2019"]
    assert SIZES["sv"] == (80000, 2000, 2000) and SIZES["deepmimo"] == (50000, 2000, 2000)
    name = "tst_3GPP_ULAZ_builtin"
    yml = ROOT / "configs/scenarios" / f"{name.lower()}.yaml"
    try:
        x = {}
        for n_tr, folder in ((300, "a"), (150, "b")):
            r = py("scripts/make_dataset.py", "--dataset", "3GPP", "--array", "ULA", "--ula-axis", "z", "--n-train", n_tr,
                   "--n-val", 100, "--n-test", 100, "--drop-users", 50, "--seed", 5, "--no-measurements", "--name", name,
                   "--out-dir", tmp_path / folder, "--force")
            assert r.returncode == 0, r.stdout + r.stderr
            x[folder] = {k: sio.loadmat(tmp_path / folder / f"3GPP_ULA_{k}_channel.mat")["x"]
                         for k in ("training", "validation", "test")}
        assert x["a"]["training"].shape == (256, 300) and x["b"]["training"].shape == (256, 150)
        assert np.allclose(x["a"]["training"][:, :150], x["b"]["training"])
        for k in ("validation", "test"):                                   # the same users for any training size
            assert np.array_equal(x["a"][k], x["b"][k])
        ref = tr38901.generate(100, "ULA", seed=5, drop_users=50, first_drop=DROP_START["test"])["H"]
        ref = u0_1d(256).conj().T @ ref
        ref *= np.sqrt(256 / np.sum(np.abs(ref) ** 2, 0))
        assert np.allclose(x["a"]["test"], ref)
        assert np.allclose(np.sum(np.abs(x["a"]["validation"]) ** 2, 0), 256)
        meta = sio.loadmat(tmp_path / "a" / "3GPP_ULA_meta.mat")
        assert np.array_equal(np.unique(meta["validation_drop"]), [1_000_001, 1_000_002])
        assert np.array_equal(np.unique(meta["training_drop"]), np.arange(1, 7))
        assert meta["test_lsp"].shape == (100, 7) and meta["training_loc"].shape == (300, 3)
        info = json.loads((tmp_path / "a" / "dataset_info.json").read_text())
        assert set(info["drops_shared_between_sets"].values()) == {0}
        assert info["generator"]["num_ant_z"] == 256 and info["generator"]["implementation"].startswith("bsce built-in")
        sc = load_yaml(yml)
        assert sc["dataset"] == "3GPP" and sc["array"] == "ULA" and sc["channel_model"]["array"] == "ULA"
        assert sc["cs_matrix"].endswith("CSmatrix256128.mat") and sc["n_train"] == 150
        # the y axis gives other channels
        r = py("scripts/make_dataset.py", "--dataset", "3GPP", "--array", "ULA", "--ula-axis", "y", "--n-train", 50,
               "--n-val", 10, "--n-test", 10, "--drop-users", 50, "--seed", 5, "--no-measurements", "--name", name,
               "--out-dir", tmp_path / "y", "--force")
        assert r.returncode == 0, r.stdout + r.stderr
        xy = sio.loadmat(tmp_path / "y" / "3GPP_ULA_training_channel.mat")["x"]
        top = lambda v: np.mean(np.max(np.abs(v) ** 2, 0) / 256)
        assert top(xy) < top(x["b"]["training"][:, :50])                   # the horizontal ULA is less sparse
    finally:
        yml.unlink(missing_ok=True)


# ---------------------------------------------------------------- 3. make_3gpp_data
@need_cs
def test_make_3gpp_data(tmp_path):
    r = py("scripts/make_3gpp_data.py", "--sizes", "dong2019", "--out-root", tmp_path, "--dry-run")
    assert r.returncode == 0, r.stdout + r.stderr
    cmds = [ln for ln in r.stdout.splitlines() if ln.startswith(">>>")]
    assert len(cmds) == 2 and all("--n-train 81000 --n-val 9000 --n-test 19000" in c for c in cmds)
    assert "--array UPA --ula-axis y" in cmds[0] and "--array ULA --ula-axis y" in cmds[1]
    assert all("--authors-bands" in c for c in cmds)
    r = py("scripts/make_3gpp_data.py", "--only", "3GPP_ULAZ", "--spec", "19.2", "--check", "--out-root", tmp_path,
           "--dry-run")
    assert "--ula-axis z" in r.stdout and "check_3gpp.py --builtin ULA" in r.stdout and "--spec 19.2" in r.stdout
    r = py("scripts/make_3gpp_data.py", "--n-train", 100, "--dry-run")
    assert r.returncode != 0 and "together" in (r.stdout + r.stderr)
    r = py("scripts/make_3gpp_data.py", "--only", "3GPP_XYZ", "--dry-run")
    assert r.returncode != 0
    yml = ROOT / "configs/scenarios/3gpp_ulaz_tsttmp.yaml"
    try:
        r = py("scripts/make_3gpp_data.py", "--only", "3GPP_ULAZ", "--n-train", 200, "--n-val", 50, "--n-test", 50,
               "--drop-users", 50, "--suffix", "_tsttmp", "--out-root", tmp_path)
        assert r.returncode == 0, r.stdout + r.stderr
        d = tmp_path / "3GPP_ULAZ_tsttmp"
        for f in ("3GPP_ULA_training_channel.mat", "3GPP_ULA_test_channel.mat", "3GPP_ULA_meta.mat", "CSmatrix.mat",
                  "3GPP_ULA_training_128_measurements_0to10dB.mat", "3GPP_ULA_validation_128_measurements_10to20dB.mat",
                  "dataset_info.json", "dataset_check.png"):
            assert (d / f).exists(), f
        assert "| 3GPP_ULAZ_tsttmp | 200 | 50 | 50 |" in r.stdout
        sc = load_yaml(yml)
        assert (sc["n_train"], sc["n_val"], sc["n_test"], sc["n_snr_chunks"]) == (200, 50, 50, 10)
        r = py("scripts/make_3gpp_data.py", "--only", "3GPP_ULAZ", "--n-train", 200, "--n-val", 50, "--n-test", 50,
               "--suffix", "_tsttmp", "--out-root", tmp_path)
        assert r.returncode == 0 and "exists, kept" in r.stdout
    finally:
        yml.unlink(missing_ok=True)


# ---------------------------------------------------------------- 4. check, suite, shipped yaml files
@need_cs
def test_check_3gpp_builtin(tmp_path):
    r = py("scripts/check_3gpp.py", "--builtin", "ULA", "--n", 1500, "--n-test", 300, "--n-cov", 1000,
           "--no-transfer", "--results", tmp_path, "--name", "chk")
    assert r.returncode == 0, r.stdout + r.stderr
    res = json.loads((tmp_path / "chk" / "check_3gpp.json").read_text())
    assert "ULA" in json.dumps(res) and (tmp_path / "chk" / "lsp_vs_38901.png").exists()
    assert "built-in" in (tmp_path / "chk" / "check_3gpp.md").read_text()


def test_run_suite_3gpp_scenarios():
    r = py("scripts/run_suite.py", "--scenarios", "all", "3gpp", "--methods", "ladmm4", "--dry-run")
    assert r.returncode == 0, r.stdout + r.stderr
    heads = [ln for ln in r.stdout.splitlines() if ln.startswith("########")]
    assert len(heads) == 6 and "3gpp_upa.yaml" in heads[4] and "3gpp_ula.yaml" in heads[5]


@pytest.mark.parametrize("name,arr,geo", [("3gpp_upa", "UPA", "UPA"), ("3gpp_ula", "ULA", "ULA_Y"), ("3gpp_ulaz", "ULA", "ULA")])
def test_shipped_3gpp_yaml(name, arr, geo):
    sc = load_yaml(ROOT / "configs/scenarios" / f"{name}.yaml")
    assert sc["dataset"] == "3GPP" and sc["array"] == arr and sc["M"] == 128 and sc["n_snr_chunks"] == 10
    assert sc["data_dir"].replace("\\", "/").endswith("data/" + sc["name"]) and sc["cs_matrix"].endswith("CSmatrix256128.mat")
    assert [(b["name"], b["val_snr"]) for b in sc["bands"]] == [("0to10dB", [10]), ("10to20dB", [10])]
    assert (sc["n_train"], sc["n_val"], sc["n_test"]) == (80000, 2000, 2000)
    assert sc["channel_model"]["array"] == geo and sc["channel_model"]["fc"] == 28e9

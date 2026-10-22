"""v30: combiners with any number of RF chains, the combiner check, the suite's protocol for Graph-LADMM, and names.

* set_combiner with another M (Graph-LADMM analytic, LADMM analytic): the network equals one built on that combiner with
  the same weights, and a saved + reloaded network keeps the new shape; networks learned for one combiner refuse;
* evaluate.py --combiner random:<seed>:<M>: the networks that follow run, the others are skipped (not a crash), and
  random:<seed> is the v29 combiner;
* check_combiners.py: fixed test combiners (same on every call), its table, figure and 'cannot run' cells;
* run_suite.py: gladmm / gladmmany keep their own end-to-end protocol (folder _paper) unless --force-protocol;
* compare.py --preset equals the workbench's presets;
* names: method keys, labels, folder names, configs and scripts are unique and exist.
"""
import csv
import json
import re
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
import scipy.io as sio
import torch

from bsce.data import measurements, sv
from bsce.export import load_model, save_model
from bsce.models import build_model
from bsce.rng import NumpyMatlabRNG

ROOT = Path(__file__).resolve().parent.parent
SV_A = ROOT / "data" / "SV_ULA" / "CSmatrix.mat"


def _A(seed=0, M=128, N=256):
    g = np.random.default_rng(seed)
    return ((g.random((M, N)) > 0.5) * 2 - 1) / np.sqrt(M)


FOLLOW = [dict(model="Graph-LADMM", T=3, array="ULA", precision="float64"),
          dict(model="Graph-LADMM", T=2, array="UPA", zstep="gtv_gm", precision="float64"),
          dict(model="LADMM", T=3, operator="analytic", precision="float64"),
          dict(model="LADMM", T=2, operator="analytic1", data_step="whitened", threshold="gm", precision="float64")]


@pytest.mark.parametrize("M", [64, 96, 160])
@pytest.mark.parametrize("cfg", FOLLOW, ids=lambda c: f"{c['model']}-{c.get('operator', c.get('zstep', 'gtv_gate'))}")
def test_set_combiner_any_M(cfg, M, tmp_path, record):
    A0, A1 = _A(0), _A(7, M=M)
    x = sv.beamspace_channels(NumpyMatlabRNG(1), 12, cfg.get("array", "UPA"))
    y = measurements.gen_measurements_matlab(x, A1, 10.0, NumpyMatlabRNG(2))
    a, b = build_model(cfg, A0), build_model(cfg, A1)
    g = torch.Generator().manual_seed(3)
    with torch.no_grad():
        for p in a.parameters():
            p.add_(0.1 * torch.randn(p.shape, generator=g, dtype=p.dtype))
        for k, v in a.named_parameters():
            b.get_parameter(k).copy_(v)
        if hasattr(a, "phase"):
            b.phase.copy_(a.phase)
    a.set_combiner(A1)
    assert a.M == M and a.A.shape == (M, 256) and np.array_equal(a.A_np, A1)
    with torch.no_grad():
        ya, yb = a(torch.tensor(y)), b(torch.tensor(y))
    err = float((ya - yb).abs().max() / yb.abs().max())
    save_model(a, tmp_path / "net")
    c = load_model(next(tmp_path.glob("net*.pt")))
    with torch.no_grad():
        err2 = float((c(torch.tensor(y)) - ya).abs().max() / ya.abs().max())
    record(test="set_combiner_any_M", model=cfg["model"], M=M, rel_err_vs_fresh=err, rel_err_reload=err2)
    assert err < 1e-12 and err2 < 1e-12 and c.M == M


def test_set_combiner_refusals():
    A0 = _A(0)
    for cfg in (dict(model="Graph-LADMM", T=2, array="ULA", xstep="free"), dict(model="LADMM", T=2)):
        with pytest.raises(ValueError):
            build_model(cfg, A0).set_combiner(_A(1, M=64))
    net = build_model(dict(model="Graph-LADMM", T=2, array="ULA"), A0)
    with pytest.raises(ValueError):                       # another N
        net.set_combiner(_A(1, N=128))
    bad = _A(1, M=64)
    bad[1] = bad[0]                                       # dependent rows
    with pytest.raises(ValueError):
        net.set_combiner(bad)
    assert net.M == 128                                   # a refused combiner changes nothing


@pytest.fixture(scope="module")
def runs(tmp_path_factory):
    """A runs folder: an analytic Graph-LADMM (follows), a free-H Graph-LADMM and a LADMM-F (one combiner)."""
    d = tmp_path_factory.mktemp("v30") / "runs"
    A = sio.loadmat(SV_A)["A"].astype(np.float64)
    for tag, cfg in [("G-an", dict(model="Graph-LADMM", T=2, array="ULA")),
                     ("G-free", dict(model="Graph-LADMM", T=2, array="ULA", xstep="free")),
                     ("L-F", dict(model="LADMM", T=2))]:
        save_model(build_model(dict(cfg, precision="float32"), A), d / tag / f"{tag}_0to20dB")
    return d


def _evaluate(runs, out, *x):
    r = subprocess.run([sys.executable, "scripts/evaluate.py", "--scenario", "configs/scenarios/sv_ula.yaml", "--runs",
                        str(runs), "--classical", "", "--n-test", "40", "--device", "cpu", "--out", str(out), *x],
                       cwd=ROOT, capture_output=True, text=True)
    return r


def test_evaluate_other_M(runs, tmp_path):
    r = _evaluate(runs, tmp_path / "m64", "--combiner", "random:11:64")
    assert r.returncode == 0, r.stderr[-2000:]
    log = json.loads((tmp_path / "m64" / "eval_log.json").read_text())
    note = log["network_combiner"]
    assert note["G-an"] == "follows"
    assert note["G-free"].startswith("cannot run") and note["L-F"].startswith("cannot run")
    head = next(csv.reader(open(tmp_path / "m64" / "nmse_vs_snr.csv")))
    assert head == ["SNR_dB", "G-an"]
    # random:<seed> is the combiner of v29 (same draws)
    r = _evaluate(runs, tmp_path / "m128", "--combiner", "random:11", "--tags", "G-an")
    assert r.returncode == 0 and "(128 x 256)" in r.stdout
    for bad in ("random:x", "random:1:0", "random:1:2:3"):
        r = _evaluate(runs, tmp_path / "bad", "--combiner", bad)
        assert r.returncode != 0 and "random:<seed>" in (r.stderr + r.stdout)


def test_check_combiners(runs, tmp_path, record):
    sys.path.insert(0, str(ROOT / "scripts"))
    import check_combiners as cc
    A = sio.loadmat(SV_A)["A"].astype(np.float64)
    C1, C2 = cc.make_combiners(A), cc.make_combiners(A)
    assert list(C1) == ["own", "pm1", "gauss", "sparse", "dct", "corr", "M160", "M96", "M64"]
    assert all(np.array_equal(C1[k], C2[k]) for k in C1)          # the same combiners on every call / PC
    assert np.array_equal(C1["pm1"], _A(11))                     # = evaluate.py --combiner random:11
    assert all(np.linalg.matrix_rank(c) == c.shape[0] for c in C1.values())
    out = tmp_path / "cc"
    r = subprocess.run([sys.executable, "scripts/check_combiners.py", "--scenario", "configs/scenarios/sv_ula.yaml",
                        "--runs", str(runs), "--combiners", "own", "dct", "M--", "--classical", "", "--n-test", "30",
                        "--device", "cpu", "--out", str(out)], cwd=ROOT, capture_output=True, text=True)
    assert r.returncode == 0, r.stderr[-2000:]
    for f in ("combiner_check.md", "combiner_check.csv", "combiner_check.png", "combiners.csv", "own/nmse_vs_snr.csv",
              "M64/eval_log.json", "combiners/dct.mat"):
        assert (out / f).exists(), f
    rows = list(csv.DictReader(open(out / "combiner_check.csv")))
    cell = {(x["network"], x["combiner"]): x for x in rows}
    assert cell[("L-F", "M64")]["note"].startswith("cannot run") and cell[("L-F", "M64")]["mean_nmse_db"] == ""
    assert cell[("G-an", "M64")]["note"] == "follows" and cell[("G-an", "M64")]["mean_nmse_db"] != ""
    assert cell[("L-F", "dct")]["note"].startswith("fixed")
    assert float(cell[("G-an", "own")]["change_vs_own_db"]) == 0.0
    md = (out / "combiner_check.md").read_text(encoding="utf-8")
    assert "cannot run" in md and "M = 64" in md
    props = list(csv.DictReader(open(out / "combiners.csv")))
    assert [p["combiner"] for p in props] == ["own", "dct", "M64"]
    record(test="check_combiners", **{f"{x['network']}@{x['combiner']}": x["mean_nmse_db"] for x in rows})
    r = subprocess.run([sys.executable, "scripts/check_combiners.py", "--scenario", "configs/scenarios/sv_ula.yaml",
                        "--runs", str(runs), "--combiners", "nope", "--out", str(tmp_path / "x")],
                       cwd=ROOT, capture_output=True, text=True)
    assert r.returncode != 0 and "unknown combiners" in (r.stderr + r.stdout)


def _suite(*x):
    r = subprocess.run([sys.executable, "scripts/run_suite.py", "--scenarios", "sv_ula", "--dry-run", *x], cwd=ROOT,
                       capture_output=True, text=True)
    assert r.returncode == 0, r.stderr[-2000:]
    return [ln for ln in r.stdout.splitlines() if ln.startswith(">>> scripts/train.py")]


def test_suite_protocol_of_graph_ladmm():
    cmds = _suite("--methods", "gladmm", "gladmmany", "gladmmf", "ladmm4")
    tag = {re.search(r"--tag (\S+)", c).group(1): c for c in cmds}
    assert set(tag) == {"G-LADMM-T8_paper", "G-LADMM-anyA-T8_paper", "G-LADMM-F-T6", "LADMM-T4"}
    assert "--protocol" not in tag["G-LADMM-T8_paper"] and "--protocol ladmm_best" in tag["G-LADMM-F-T6"]
    (c,) = _suite("--methods", "gladmm", "--force-protocol")
    assert "--tag G-LADMM-T8 " in c + " " and "--protocol ladmm_best" in c
    (c,) = _suite("--methods", "gladmm", "--protocol", "paper")       # the same folder as v29's --protocol paper
    assert "--tag G-LADMM-T8_paper " in c + " "
    (c,) = _suite("--methods", "gladmm", "--set", "train.steps=9000")
    assert "--tag G-LADMM-T8_paper_steps-9000 " in c + " "
    r = subprocess.run([sys.executable, "scripts/run_suite.py", "--scenarios", "sv_ula", "--dry-run", "--methods",
                        "gladmm", "ladmm6", "--ensemble", "gladmm", "ladmm6"], cwd=ROOT, capture_output=True, text=True)
    assert "--tag ENS-G-LADMM-T8_paper_ft_0-10-20dB_20k+LADMM-T6_ft_0-10-20dB_20k" in r.stdout


def test_compare_presets_match_workbench():
    sys.path.insert(0, str(ROOT / "scripts"))
    import compare
    app = ROOT / "app" / "bsce_app.py"
    if not app.exists():                                  # the Spyder edition has no workbench
        pytest.skip("no workbench in this edition")
    txt = app.read_text(encoding="utf-8")
    ns = {}
    exec(re.search(r"^TAG_PRESETS = .*$", txt, re.M).group(0), ns)
    exec(re.search(r"^TAG_EXCLUDE = .*$", txt, re.M).group(0), ns)
    for k, (tags, exc) in compare.PRESETS.items():
        assert ns["TAG_PRESETS"][k] == tags and ns["TAG_EXCLUDE"].get(k, []) == exc
    assert len(set(compare.distinct_colors(40))) == 40


def test_names_unique_and_existing():
    txt = (ROOT / "scripts" / "run_suite.py").read_text(encoding="utf-8")
    body = txt[txt.index("METHODS = {"):txt.index("NOT_IN_ALL")]
    rows = re.findall(r'^\s+"([a-z]+)":\s*\("([^"]+)",\s*"([^"]+)",\s*(\d+)\)', body, re.M)
    keys, labels, cfgs = [r[0] for r in rows], [r[1] for r in rows], [r[2] for r in rows]
    assert len(keys) == len(set(keys)), "duplicate method key"
    assert len(labels) == len(set(labels)), "duplicate folder label"
    assert all((ROOT / c).exists() for c in cfgs), "missing train config"
    sys.path.insert(0, str(ROOT / "scripts"))
    import run_suite
    assert set(run_suite.METHODS) == set(keys) and set(run_suite.OWN_PROTOCOL) <= set(keys)
    assert set(run_suite.NOT_IN_ALL) <= set(keys)
    # two methods never give the same folder, at any depth, protocol or suffix
    names = {}
    for k, (lab, _, _) in run_suite.METHODS.items():
        for T in range(1, 33):
            for proto in run_suite.PROTOCOLS:
                n = run_suite.run_tag(lab, T, proto, [0.0, 20.0], 0, [], [])
                assert names.setdefault(n, k) == k, f"{k} and {names[n]} share the folder {n}"
    # a key is never the start of another key + digits (gladmm8 must not be read as another method)
    for k in keys:
        assert run_suite.method_key(k + "8") == k
    scripts = [p.name.lower() for p in (ROOT / "scripts").glob("*.py")]
    assert len(scripts) == len(set(scripts))
    configs = [p.name.lower() for p in (ROOT / "configs" / "train").glob("*.yaml")]
    assert len(configs) == len(set(configs))

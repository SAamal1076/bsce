"""v27: LVAMP's LMMSE stage for folded noise (option model.noise_model=colored; the default stays the original
white-noise stage); the suite's scenario-setting clash check; the fine-tune of a depth snapshot. Synthetic data only.

1. noise_model 'colored': the LMMSE weights are uniform in the SVD domain (ytilde = V^T (x + n)); 'white': 1/s^2.
   With A A^T = I both are the same network; otherwise they differ, and the colored stage is the exact LMMSE for
   folded noise (checked against a direct computation).
2. A network saved without the key (bsce <= v26) loads as 'white', with the same outputs as before.
3. train.py: the default LVAMP is the original one ('white', folder names as before); --set model.noise_model=colored
   gives the folded-noise stage, in its own folder.
4. run_suite.py: a run folder of another scenario setting (same scenario name, other bands or validation) stops the
   suite instead of being skipped or mixed.
5. finetune_bands.py --from <run>/layers/T01_<run>_<band>.pt: the scenario from <run>/train_log.json, the name
   <run>_T01_ft_..., next to <run>.
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
from bsce.export import load_model, save_model  # noqa: E402
from bsce.models import build_model  # noqa: E402

M, N = 32, 64
A = np.random.default_rng(0).choice([-1.0, 1.0], (M, N)) / np.sqrt(M)


def _x(K, seed):
    r = np.random.default_rng(seed)
    x = np.zeros((N, K), complex)
    for k in range(K):
        idx = r.choice(N, 3, replace=False)
        x[idx, k] = (r.standard_normal(3) + 1j * r.standard_normal(3)) / np.sqrt(2) * np.sqrt(N / 3)
    return x


def _lmmse_first_stage(net, y):
    """The first LMMSE estimate r_1 of the network (the input of the first denoiser)."""
    out = {}
    import bsce.models.lvamp as L
    orig = L.bg_complex

    def spy(r, tau, theta):
        out.setdefault("r", r.detach().clone())
        out.setdefault("tau", tau.detach().clone())
        return orig(r, tau, theta)
    L.bg_complex = spy
    try:
        with torch.no_grad():
            net(torch.as_tensor(y))
    finally:
        L.bg_complex = orig
    return out["r"].numpy(), out["tau"].numpy()


def test_lmmse_weights_follow_the_noise_model():
    x = _x(300, 1)
    s2 = 0.05
    r = np.random.default_rng(2)
    n = (r.standard_normal((N, 300)) + 1j * r.standard_normal((N, 300))) * np.sqrt(s2 / 2)
    y = A @ (x + n)                                                      # folded noise
    nets = {}
    for nm in ("white", "colored"):
        net = build_model(dict(model="LVAMP", T=2, precision="float64", noise_model=nm), A)
        with torch.no_grad():
            net.P("logyvar").fill_(np.log(s2))
        nets[nm] = net
    rw, _ = _lmmse_first_stage(nets["white"], y)
    rc, tau = _lmmse_first_stage(nets["colored"], y)
    assert np.abs(rw - rc).max() > 1e-3                                  # different stages
    # colored = the exact LMMSE of ytilde = V^T (x + n) under the VAMP prior x ~ CN(0, tau_r I): uniform weights
    U_s2, U = np.linalg.eigh(A @ A.T)
    s = np.sqrt(U_s2)
    V = (A.T @ U) / s
    yt = (U / s).T @ y
    taurj = np.sum(np.abs(y) ** 2, 0) / N
    sc = 1 / (1 + s2 / taurj)                                            # per column, the same for every component
    zeta = N / (M * sc)
    ref = V @ (yt * (sc * zeta)[None, :])
    assert np.abs(rc - ref).max() < 1e-10
    # with A A^T = I the two stages are one
    Q, _ = np.linalg.qr(np.random.default_rng(3).standard_normal((N, M)))
    Ao = Q.T
    a = build_model(dict(model="LVAMP", T=2, precision="float64", noise_model="white"), Ao)
    b = build_model(dict(model="LVAMP", T=2, precision="float64", noise_model="colored"), Ao)
    yo = Ao @ (x + n)
    with torch.no_grad():
        assert torch.allclose(a(torch.as_tensor(yo)), b(torch.as_tensor(yo)), atol=1e-10)


def test_old_network_loads_as_trained(tmp_path):
    net = build_model(dict(model="LVAMP", T=3, precision="float64"), A)          # the registry default: white (<= v26)
    assert net.noise_model == "white" and net.cfg["noise_model"] == "white"
    with torch.no_grad():
        for p in net.parameters():
            p.add_(0.1 * torch.randn_like(p))
    pt, _ = save_model(net, tmp_path / "LV" / "LV_0to20dB")
    ck = torch.load(pt, weights_only=False)
    ck["cfg"].pop("noise_model")                                                 # as saved by bsce <= v26
    torch.save(ck, pt)
    old = load_model(pt)
    assert old.noise_model == "white"
    y = A @ _x(50, 4)
    with torch.no_grad():
        assert torch.equal(old(torch.as_tensor(y)), net(torch.as_tensor(y)))
    mat = load_model(tmp_path / "LV" / "LV_0to20dB.mat")                          # the MATLAB export keeps the key
    assert mat.noise_model == "white"


def _scenario(tmp_path):
    d = tmp_path / "data" / "T"
    d.mkdir(parents=True)
    sio.savemat(d / "CSmatrix.mat", {"A": A})
    for k, (part, K) in enumerate((("training", 400), ("validation", 100), ("test", 100))):
        sio.savemat(d / f"SV_ULA_{part}_channel.mat", {"x": _x(K, 20 + k)})
    y = tmp_path / "t.yaml"
    y.write_text(f"name: T\ndataset: SV\narray: ULA\nN1: {N}\nN2: 1\nL: 3\nM: {M}\ndata_dir: {d}\nseed: 3\n"
                 "n_snr_chunks: 10\nbands:\n  - {name: 0to10dB, train: [0, 10], val_snr: [10], use_for: [0, 10]}\n"
                 "  - {name: 10to20dB, train: [10, 20], val_snr: [10], use_for: [10, 20.01]}\n"
                 "test_snr_db: [0, 10, 20]\n", encoding="utf-8")
    y4 = tmp_path / "t4.yaml"                                    # the same name, other bands (as sv_ula_4bins.yaml)
    y4.write_text(y.read_text().split("bands:")[0] + "bands:\n"
                  "  - {name: 0to5dB, train: [0, 5], val_snr: [1], use_for: [0, 5]}\n"
                  "  - {name: 5to20dB, train: [5, 20], val_snr: [6], use_for: [5, 20.01]}\ntest_snr_db: [0, 10, 20]\n",
                  encoding="utf-8")
    return y, y4


def _run(*args, check=True):
    return subprocess.run([sys.executable, *map(str, args)], cwd=ROOT, capture_output=True, text=True, check=check)


def test_train_records_the_noise_model(tmp_path):
    y, _ = _scenario(tmp_path)
    out = tmp_path / "outputs"
    common = ["--scenario", y, "--method", "lvamp2", "--protocol", "e2e", "--snr", "0", "20", "--out", out, "--device",
              "cpu", "--set", "train.steps=20", "train.val_every=10"]
    _run("scripts/train.py", *common)
    _run("scripts/train.py", *common, "model.noise_model=colored")
    got = {}
    for f in sorted(out.rglob("LVAMP-T2*_0to20dB.pt")):
        got[f.parent.name] = torch.load(f, weights_only=False)["cfg"]["noise_model"]
    assert got == {"LVAMP-T2_e2e_steps-20_val-every-10": "white",
                   "LVAMP-T2_e2e_steps-20_val-every-10_noise-model-colored": "colored"}, got
    log = json.loads((out / "T" / "LVAMP-T2_e2e_steps-20_val-every-10" / "train_log.json").read_text())
    assert "noise_model" not in log["config"]["model"]          # the default config (and checkpoint name) as in v26


def test_suite_stops_on_a_scenario_clash(tmp_path):
    y, y4 = _scenario(tmp_path)
    out = tmp_path / "outputs"
    # a run of the 2-band scenario, then the same command with the 4-band scenario of the same name
    common = ["--methods", "lvamp2", "--protocol", "e2e", "--train-snr", "scenario", "--ft-snr", "none", "--out", out,
              "--device", "cpu", "--set", "train.steps=20", "train.val_every=10"]
    _run("scripts/run_suite.py", "--scenarios", y, *common)
    r = _run("scripts/run_suite.py", "--scenarios", y, *common)
    assert "trained, skipped" in r.stdout                                    # the same setting: skipped as before
    r = _run("scripts/run_suite.py", "--scenarios", y4, *common, check=False)
    assert r.returncode != 0 and "holds a run of another scenario setting" in r.stderr and "0to10dB" in r.stderr
    # other validation for the same band names (as deepmimo_upa_2band_valrange.yaml): also a clash
    r = _run("scripts/run_suite.py", "--scenarios", y, *common, "--val-snr", "range", "--dry-run", check=False)
    assert r.returncode == 0                                                 # its own folder (_valrange): no clash
    yv = tmp_path / "tv.yaml"
    yv.write_text(y.read_text().replace("val_snr: [10]", "val_range: [0, 10]", 1), encoding="utf-8")
    r = _run("scripts/run_suite.py", "--scenarios", yv, *common, "--dry-run", check=False)
    assert r.returncode != 0 and "val_snr" in r.stderr


def test_finetune_from_a_depth_snapshot(tmp_path):
    y, _ = _scenario(tmp_path)
    out = tmp_path / "outputs"
    _run("scripts/train.py", "--scenario", y, "--method", "ladmm2", "--snr", "0", "20", "--save-layers", "--out", out,
         "--device", "cpu", "--set", "train.maxit=20", "train.better_wait=10", "model.init_precision=null")
    run = out / "T" / "LADMM-T2_maxit-20_better-wait-10_init-precision-null"
    snap = run / "layers" / f"T01_{run.name}_0to20dB.pt"
    assert snap.exists()
    r = _run("scripts/finetune_bands.py", "--from", snap, "--dry-name")
    assert Path(r.stdout.strip().splitlines()[-1]) == out / "T" / f"{run.name}_T01_ft_0-10-20dB_20k"
    r = _run("scripts/finetune_bands.py", "--from", snap, "--iters", "20", "--batch", "16", "--val-every", "10",
             "--device", "cpu")
    ft = out / "T" / f"{run.name}_T01_ft_0-10-20dB_20it_batch16_valevery10"
    assert (ft / f"{ft.name}_10to20dB.pt").exists(), r.stdout[-1500:]
    assert "scenario taken from the run's train_log.json" in r.stdout

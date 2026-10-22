"""Resume INSIDE a stage (v24): a run interrupted in the middle of a stage continues from its last in-stage
checkpoint and ends with exactly the weights of an uninterrupted run.

1. layer-wise (Borgerding) schedule, LADMM, TF1 Adam (its step counter and beta powers are saved too), warm start off
2. layer-wise with warm start (GM-LAMP's protocol): a stage resumed inside is not warm-started again
3. 'steps' schedule (NF-LADMM, online noise, cosine learning rate)
4. 'end2end' schedule with lr_list and patience (UTRR's training), resumed after a saved epoch
5. the checkpoint is written atomically (no .tmp left) and the in-stage position is gone once the stage is done
6. finetune_bands.py: interrupted inside a band, resumed exactly; a finished band is not trained again
Synthetic data only (no reference/ or data/ needed).
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
from bsce.config import load_yaml, merge  # noqa: E402
from bsce.models import build_model  # noqa: E402
from bsce.train.trainer import Trainer, TrainConfig  # noqa: E402

M, N = 32, 64
A = np.random.default_rng(0).choice([-1.0, 1.0], (M, N)) / np.sqrt(M)


def _data(K, seed, s2=0.05):
    r = np.random.default_rng(seed)
    x = np.zeros((N, K), complex)
    for k in range(K):
        idx = r.choice(N, 3, replace=False)
        x[idx, k] = (r.standard_normal(3) + 1j * r.standard_normal(3)) / np.sqrt(2) * np.sqrt(N / 3)
    n = (r.standard_normal((N, K)) + 1j * r.standard_normal((N, K))) * np.sqrt(s2 / 2)
    return x, A @ (x + n), np.full(K, s2)


class Crash(Exception):
    pass


def _run(mcfg, tcfg, ck, data, crash_at=None, **kw):
    """Train; with crash_at, raise inside the training after that many minibatch steps (an interruption)."""
    (xt, yt, st), (xv, yv, sv) = data
    torch.manual_seed(0)
    net = build_model(mcfg, A)
    tr = Trainer(net, TrainConfig.from_dict(tcfg), xt, yt, xv, yv, checkpoint=str(ck), s2_train=st, s2_val=sv, **kw)
    if crash_at is not None:
        n, step = [0], tr.train_step

        def counting(*a, **k):
            n[0] += 1
            if n[0] == crash_at:
                raise Crash
            return step(*a, **k)
        tr.train_step = counting
        if tcfg.get("schedule") == "end2end":             # end2end does not call train_step: crash in the loss
            lf = tr.loss_fn

            def crash_loss(*a, **k):
                n[0] += 1
                if n[0] == crash_at:
                    raise Crash
                return lf(*a, **k)
            tr.loss_fn = crash_loss
    try:
        state = tr.run()
    except Crash:
        return None, None
    return net, state


def _same(a, b):
    sa, sb = a.state_dict(), b.state_dict()
    return max(float((sa[k] - sb[k]).abs().max()) for k in sa)


def _check(mcfg, tcfg, tmp_path, crash_at, data=None):
    data = data or (_data(300, 1), _data(80, 2))
    a, sa = _run(mcfg, tcfg, tmp_path / "a" / "run", data)
    assert _run(mcfg, tcfg, tmp_path / "b" / "run", data, crash_at=crash_at)[0] is None
    ck = torch.load(tmp_path / "b" / "run.pt", weights_only=False)
    assert ck["train_state"].get("partial") is not None, "no checkpoint inside the stage before the crash"
    b, sb = _run(mcfg, tcfg, tmp_path / "b" / "run", data)
    d = _same(a, b)
    print(f"max |uninterrupted - resumed| = {d}")
    assert d == 0.0
    assert [s["nmse_db"] for s in sa["log"]] == [s["nmse_db"] for s in sb["log"]]
    ck = torch.load(tmp_path / "b" / "run.pt", weights_only=False)
    assert "partial" not in ck["train_state"] and not list((tmp_path / "b").glob("*.tmp"))
    return ck


def test_layerwise_resume_inside_a_stage(tmp_path):
    cfg = merge(load_yaml(ROOT / "configs/train/ladmm_original.yaml"), load_yaml(ROOT / "configs/train/ladmm_best_protocol.yaml"))
    tc = dict(cfg["train"], maxit=60, better_wait=1000, verbose=False, batch_size=32, optimizer="adam_tf1",
              checkpoint_iters=7, checkpoint_minutes=0)
    mc = dict(cfg["model"], T=2, array="ULA", N1=N, N2=1, init_precision=None)
    _check(mc, tc, tmp_path, crash_at=150)                 # stage 3 (61 steps per stage), after its saves at 7, 14, ...


def test_layerwise_warm_start_resume(tmp_path):
    cfg = merge(load_yaml(ROOT / "configs/train/gmlamp_original.yaml"), load_yaml(ROOT / "configs/train/gmlamp_protocol.yaml"))
    tc = dict(cfg["train"], maxit=40, better_wait=1000, verbose=False, batch_size=32, checkpoint_iters=9,
              checkpoint_minutes=0)
    assert tc.get("warm_start")
    mc = dict(cfg["model"], T=2)
    _check(mc, tc, tmp_path, crash_at=4 * 41 + 25)         # inside the first stage of layer 2 (a warm-started stage)


def test_steps_resume(tmp_path):
    tc = dict(schedule="steps", steps=45, val_every=10, lr=1e-3, lr_final_frac=0.05, batch_size=16, noise="online",
              snr_range=[0, 20], loss="log_nse", restore_best=True, optimizer="adam_torch", grad_clip=1.0, verbose=False,
              checkpoint_iters=8, checkpoint_minutes=0)
    mc = dict(model="NF-LADMM", T=2, array="ULA", N1=N, N2=1, denoiser="conv", update="relax", precision="float64")
    _check(mc, tc, tmp_path, crash_at=29)


def test_end2end_lr_list_resume(tmp_path):
    tc = dict(schedule="end2end", lr_list=[2e-3, 1e-3], epochs=4, patience=3, batch_size=50, loss="mse",
              restore_best=True, optimizer="adam_torch", verbose=False, checkpoint_iters=1, checkpoint_minutes=0)
    mc = dict(model="UTRR", T=2, K=4, precision="float64")
    _check(mc, tc, tmp_path, crash_at=6 * 2 + 3)           # epoch 3 (6 minibatches per epoch), first stage


def test_time_based_default():
    c = TrainConfig()
    assert c.checkpoint_minutes == 10.0 and c.checkpoint_iters is None


# ------------------------------------------------------------------------------- finetune_bands.py
def _ft_setup(tmp_path):
    """A tiny scenario + a trained 2-layer LADMM network in <tmp>/outputs/T/LADMM-T2/ (train.py)."""
    d = tmp_path / "data" / "T"
    d.mkdir(parents=True)
    sio.savemat(d / "CSmatrix.mat", {"A": A})
    for k, (part, K) in enumerate((("training", 400), ("validation", 100), ("test", 100))):
        sio.savemat(d / f"SV_ULA_{part}_channel.mat", {"x": _data(K, 10 + k)[0]})
    yml = tmp_path / "t.yaml"
    yml.write_text(f"name: T\ndataset: SV\narray: ULA\nN1: {N}\nN2: 1\nL: 3\nM: {M}\ndata_dir: {d}\nseed: 3\n"
                   "n_snr_chunks: 10\nbands:\n  - {name: 0to20dB, train: [0, 20], val_range: [0, 20], n_snr_chunks: 100, "
                   "use_for: [0, 20.01]}\ntest_snr_db: [0, 10, 20]\n", encoding="utf-8")
    out = tmp_path / "outputs"
    subprocess.run([sys.executable, "scripts/train.py", "--scenario", yml, "--method", "ladmm2", "--out", out,
                    "--device", "cpu", "--set", "train.maxit=20", "train.better_wait=10", "model.init_precision=null"],
                   cwd=ROOT, check=True, capture_output=True, text=True)
    run = out / "T" / "LADMM-T2_maxit-20_better-wait-10_init-precision-null"     # every --set override is in the name
    assert (run / f"{run.name}_0to20dB.pt").exists()
    for f in run.iterdir():                                   # the short name LADMM-T2 for the rest of the test
        if f.name.startswith(run.name + "_"):
            f.rename(run / ("LADMM-T2" + f.name[len(run.name):]))
    run.rename(out / "T" / "LADMM-T2")
    subprocess.run([sys.executable, "-c", "pass"],
                   cwd=ROOT, check=True, capture_output=True, text=True)
    return out / "T" / "LADMM-T2"


def _ft(src, *extra, check=True, crash=None):
    import os
    env = dict(os.environ, BSCE_CRASH_AFTER_STEPS=str(crash or 0))
    return subprocess.run([sys.executable, "scripts/finetune_bands.py", "--from", src, "--iters", "60", "--batch", "16",
                           "--val-every", "10", "--device", "cpu", "--checkpoint-iters", "7", *extra],
                          cwd=ROOT, capture_output=True, text=True, check=check, env=env)


def test_finetune_resume_and_names(tmp_path):
    src = _ft_setup(tmp_path)
    r = _ft(src, "--out", tmp_path / "o1")                  # uninterrupted (into another parent folder)
    assert "band 10to20dB" in r.stdout
    a = tmp_path / "o1" / "T" / "LADMM-T2_ft_0-10-20dB_60it_batch16_valevery10"
    assert (a / "LADMM-T2_ft_0-10-20dB_60it_batch16_valevery10_0to10dB.pt").exists() and (a / "LADMM-T2_ft_0-10-20dB_60it_batch16_valevery10_10to20dB.pt").exists()
    # interrupted inside band 2 (crash after 25 of its 60 steps), then the same command again
    r = _ft(src, check=False, crash=60 + 25)
    assert r.returncode != 0
    b = src.parent / "LADMM-T2_ft_0-10-20dB_60it_batch16_valevery10"             # default: next to the source, automatic name
    assert (b / "LADMM-T2_ft_0-10-20dB_60it_batch16_valevery10_0to10dB.pt").exists() and not (b / "LADMM-T2_ft_0-10-20dB_60it_batch16_valevery10_10to20dB.pt").exists()
    t_band1 = (b / "LADMM-T2_ft_0-10-20dB_60it_batch16_valevery10_0to10dB.pt").stat().st_mtime_ns
    rr = subprocess.run([sys.executable, "scripts/runs.py", "--outputs", src.parent.parent, "--no-write"], cwd=ROOT,
                        capture_output=True, text=True, check=True).stdout
    line = next(ln for ln in rr.splitlines() if ln.startswith("| LADMM-T2_ft_"))
    assert "resumable: band 10to20dB, step" in line and "1/2" in line and "fine-tuned from LADMM-T2 (60 it./band)" in line
    assert "| LADMM-T2 | trained | LADMM, T=2" in rr and "| done |" in rr
    r = _ft(src)
    assert "band 0to10dB: done earlier, kept" in r.stdout and "resumed inside band 10to20dB at step" in r.stdout
    assert (b / "LADMM-T2_ft_0-10-20dB_60it_batch16_valevery10_0to10dB.pt").stat().st_mtime_ns == t_band1     # not trained again
    for band in ("0to10dB", "10to20dB"):
        sa = torch.load(a / f"LADMM-T2_ft_0-10-20dB_60it_batch16_valevery10_{band}.pt", weights_only=False)["state_dict"]
        sb = torch.load(b / f"LADMM-T2_ft_0-10-20dB_60it_batch16_valevery10_{band}.pt", weights_only=False)["state_dict"]
        assert max(float((sa[k] - sb[k]).abs().max()) for k in sa) == 0.0, band
    assert not list((b / "checkpoints").glob("*.pt")) if (b / "checkpoints").exists() else True
    log = json.loads((b / "finetune_log.json").read_text())
    assert set(log["bands"]) == {"0to10dB", "10to20dB"} and log["tag"] == "LADMM-T2_ft_0-10-20dB_60it_batch16_valevery10"
    # the v25 early-stop options at their defaults are not in the setting: a v24 fine-tune folder keeps its bands
    assert "patience" not in log["setting"] and "min_gain" not in log["setting"]
    # the same command once more: nothing to do
    r = _ft(src)
    assert r.stdout.count("done earlier, kept") == 2
    # another budget -> another folder, the first one untouched
    r = subprocess.run([sys.executable, "scripts/finetune_bands.py", "--from", src, "--iters", "20000", "--dry-name"],
                       cwd=ROOT, capture_output=True, text=True, check=True)
    assert r.stdout.strip().endswith("LADMM-T2_ft_0-10-20dB_20k")


def test_run_names_with_a_dot(tmp_path):
    """A run name with a dot (an override such as train.lr=0.0005 -> _lr-0.0005) keeps its whole name in its files and
    checkpoints (v23 cut it at the dot: two bands then shared one checkpoint file)."""
    from bsce.export import save_model
    net = build_model(dict(model="NF-LADMM", T=1, array="ULA", N1=N, N2=1, denoiser="radial"), A)
    pt, mat = save_model(net, tmp_path / "R_lr-0.0005" / "R_lr-0.0005_0to10dB")
    assert pt.name == "R_lr-0.0005_0to10dB.pt" and mat.name == "R_lr-0.0005_0to10dB.mat" and pt.exists()
    (xt, yt, st), (xv, yv, sv) = _data(50, 1), _data(20, 2)
    paths = []
    for band in ("0to10dB", "10to20dB"):
        tr = Trainer(net, TrainConfig.from_dict(dict(schedule="steps", steps=2, val_every=1, noise="online", snr_range=[0, 20],
                                                     verbose=False)), xt, None, xv, yv, s2_val=sv,
                     checkpoint=str(tmp_path / "ck" / f"R_lr-0.0005_{band}_abc"))
        tr.run()
        paths.append(tr._ck_path())
    assert [p.name for p in paths] == ["R_lr-0.0005_0to10dB_abc.pt", "R_lr-0.0005_10to20dB_abc.pt"]
    assert all(p.exists() for p in paths)


# ------------------------------------------------------------------------------- early stop (v25)
def _scripted(tr, values_db):
    """Validation NMSE follows values_db (one value per validation), so the stopping rules can be checked exactly."""
    it = iter(values_db)
    tr.validate = lambda out: 10 ** (next(it) / 10)


def test_steps_early_stop_rule():
    (xt, yt, st), (xv, yv, sv) = _data(60, 1), _data(20, 2)
    mc = dict(model="NF-LADMM", T=1, array="ULA", N1=N, N2=1, denoiser="radial")
    base = dict(schedule="steps", steps=300, val_every=10, noise="online", snr_range=[0, 20], restore_best=True,
                optimizer="adam_torch", verbose=False)
    # -5 dB, then 0.5 dB better every validation up to step 60, then 0.001 dB per validation (tiny: not a gain)
    vals = [-5.0 - 0.5 * min(k, 6) - 0.001 * max(k - 6, 0) for k in range(31)]
    for cfg, stop in ((dict(base, patience_steps=50), 110), (dict(base, patience_steps=50, min_gain_db=0.0005), None),
                      (dict(base, patience_steps=0), None), (dict(base, patience_steps=100), 160)):
        torch.manual_seed(0)
        tr = Trainer(build_model(mc, A), TrainConfig.from_dict(cfg), xt, None, xv, yv, s2_val=sv)
        _scripted(tr, vals)
        s = tr.run()
        rec = s["log"][-1]
        if stop is None:
            assert rec["iters"] == 300 and "early_stop" not in rec
        else:
            assert rec["iters"] == stop and rec["early_stop"]["last_gain_step"] == 60, (cfg, rec.get("early_stop"))
        assert s["layer_nmse_db"][-1] == pytest.approx(min(vals[:rec["iters"] // 10 + 1]))   # the best is kept


def test_layerwise_min_gain():
    cfg = merge(load_yaml(ROOT / "configs/train/ladmm_original.yaml"), load_yaml(ROOT / "configs/train/ladmm_best_protocol.yaml"))
    mc = dict(cfg["model"], T=1, array="ULA", N1=N, N2=1, init_precision=None)
    (xt, yt, st), (xv, yv, sv) = _data(100, 1), _data(30, 2)
    vals = [-5.0 - 0.3 * min(k, 3) - 0.0001 * max(k - 3, 0) for k in range(5000)]    # a gain, then tiny gains only
    out = {}
    for g in (0.0, 0.01):
        tc = dict(cfg["train"], maxit=3000, better_wait=200, ivl=10, check_mult=2, verbose=False, batch_size=16,
                  refinements=[], min_gain_db=g, val_round=None)
        torch.manual_seed(0)
        tr = Trainer(build_model(mc, A), TrainConfig.from_dict(tc), xt, yt, xv, yv)
        _scripted(tr, vals)
        out[g] = [r["iters"] for r in tr.run()["log"]]
    assert out[0.0][0] == 3000                       # the original rule: every tiny new best resets the wait
    assert out[0.01][0] < 300                        # min_gain_db: only real gains count -> stops after better_wait


def test_end2end_min_gain():
    (xt, yt, st), (xv, yv, sv) = _data(100, 1), _data(30, 2)
    vals = [-5.0, -5.5, -5.6] + [-5.6 - 0.001 * k for k in range(1, 40)]
    out = {}
    for g in (0.0, 0.05):
        tc = dict(schedule="end2end", epochs=20, patience=3, batch_size=50, loss="mse", restore_best=True,
                  optimizer="adam_torch", verbose=False, min_gain_db=g)
        torch.manual_seed(0)
        tr = Trainer(build_model(dict(model="UTRR", T=1, K=2), A), TrainConfig.from_dict(tc), xt, yt, xv, yv)
        _scripted(tr, vals[1:])                       # end2end validates after every epoch
        out[g] = len(tr.run()["log"])
    assert out[0.0] == 20 and out[0.05] == 5          # tiny gains: all epochs; min gain 0.05 dB: stop 3 epochs after


def test_steps_early_stop_resume_is_exact(tmp_path):
    tc = dict(schedule="steps", steps=400, val_every=10, lr=1e-7, lr_final_frac=1.0, batch_size=16, noise="online",
              snr_range=[0, 20], loss="nmse", restore_best=True, optimizer="adam_torch", verbose=False,
              patience_steps=40, checkpoint_iters=8, checkpoint_minutes=0)
    mc = dict(model="NF-LADMM", T=1, array="ULA", N1=N, N2=1, denoiser="radial", precision="float64")
    ck = _check(mc, tc, tmp_path, crash_at=27)
    rec = ck["train_state"]["log"][-1]
    assert rec["early_stop"]["step"] < 400 and rec["iters"] == rec["early_stop"]["step"]


def test_finetune_early_stop(tmp_path):
    src = _ft_setup(tmp_path)
    args = ("--iters", "150", "--patience", "20", "--min-gain", "50")   # nothing counts as a gain: each rate ends at 20
    r = _ft(src, *args, "--out", tmp_path / "o1")
    assert r.stdout.count("next learning rate") == 2 and r.stdout.count("band finished early") == 2
    name = "LADMM-T2_ft_0-10-20dB_150it_batch16_valevery10_pat20_mingain50"
    lg = json.loads((tmp_path / "o1" / "T" / name / "finetune_log.json").read_text())
    assert all(b["steps"] == 40 and len(b["early_stops"]) == 2 for b in lg["bands"].values())
    assert lg["setting"]["patience"] == 20 and lg["setting"]["min_gain"] == 50
    r = _ft(src, *args, check=False, crash=40 + 30)          # inside the second rate of band 2
    assert r.returncode != 0
    r = _ft(src, *args)
    assert "resumed inside band 10to20dB" in r.stdout
    for band in ("0to10dB", "10to20dB"):
        sa = torch.load(tmp_path / "o1" / "T" / name / f"{name}_{band}.pt", weights_only=False)["state_dict"]
        sb = torch.load(src.parent / name / f"{name}_{band}.pt", weights_only=False)["state_dict"]
        assert max(float((sa[k] - sb[k]).abs().max()) for k in sa) == 0.0, band

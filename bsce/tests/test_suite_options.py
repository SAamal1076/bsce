"""run_suite.py --schedule / --grad-clip (dry run: the commands only, no data needed).

1. without the new options the commands and folder names are exactly the previous ones
2. --schedule e2e: every method gets the end-to-end schedule, the protocol's other settings are kept, suffix _e2e<k>k
3. --grad-clip X: training (train.grad_clip=X) and fine-tune (--grad-clip X), suffix _clipX; 'none' -> _noclip
4. a merged config with the e2e overrides is the same for every method (fair: one schedule for all)
"""
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from bsce.config import apply_overrides, load_yaml, merge  # noqa: E402


def dry(*opts):
    r = subprocess.run([sys.executable, "scripts/run_suite.py", "--scenarios", "sv_upa", "--dry-run", *opts],
                       cwd=ROOT, capture_output=True, text=True, check=True)
    return [ln[4:] for ln in r.stdout.splitlines() if ln.startswith(">>> ")]


def test_default_commands_unchanged():
    cmds = dry("--methods", "ladmm6", "lamp", "nfladmmsr")
    assert cmds[0].endswith("--tag LADMM-T6 --device auto --out outputs --seed 0 --set model.T=6")
    assert "--tag LAMP-T8 " in cmds[2] and "grad_clip" not in " ".join(cmds) and "schedule" not in " ".join(cmds)
    assert cmds[1].endswith("--tag LADMM-T6_ft_0-10-20dB_20k --iters 20000 --device auto --seed 0")
    assert "--tag NF-LADMM-SR-T4 " in cmds[4]


def test_e2e_and_clip():
    cmds = dry("--methods", "ladmm6", "gmlamp", "--schedule", "e2e", "--e2e-steps", "50000", "--grad-clip", "0.5")
    tr, ft = cmds[0], cmds[1]
    assert "--tag LADMM-T6_e2e50k_clip0p5 " in tr
    for o in ("train.schedule=steps", "train.steps=50000", "train.restore_best=true", "train.grad_clip=0.5"):
        assert o in tr, o
    assert "--protocol ladmm_best" in tr                    # the protocol is kept
    assert ft.endswith("--grad-clip 0.5") and "LADMM-T6_e2e50k_clip0p5_ft_0-10-20dB_20k" in ft
    assert "--tag GM-LAMP-T8_e2e50k_clip0p5 " in cmds[2]


def test_noclip_paper():
    cmds = dry("--methods", "nfladmmsr", "--protocol", "paper", "--grad-clip", "none")
    assert "--tag NF-LADMM-SR-T4_paper_noclip " in cmds[0] and "train.grad_clip=null" in cmds[0]
    assert "--grad-clip" not in cmds[1]                     # fine-tune: no clipping is already the default


@pytest.mark.parametrize("protocol", ["ladmm_best", "gmlamp", "fair"])
def test_e2e_schedule_same_for_every_method(protocol):
    over = ["train.schedule=steps", "train.steps=30000", "train.lr_final_frac=0.05", "train.val_every=500",
            "train.restore_best=true", "train.grad_clip=1"]
    keys = ["schedule", "steps", "lr", "lr_final_frac", "batch_size", "loss", "noise", "grad_clip", "restore_best",
            "val_every", "optimizer"]
    rows = {}
    for m in ["ladmm_original", "lamp_original", "gmlamp_original", "lista", "alista", "lvamp", "admmnet",
              "nfladmm_s", "nfladmm_c", "nfladmm_sr"]:
        c = apply_overrides(merge(load_yaml(ROOT / f"configs/train/{m}.yaml"),
                                  load_yaml(ROOT / f"configs/train/{protocol}_protocol.yaml")), over)
        rows[m] = tuple(c["train"].get(k) for k in keys)
    assert len(set(rows.values())) == 1, rows


def test_data_options():
    """--margin / --ft-margin / --val-snr / --snr-chunks reach train.py and finetune_bands.py, with folder suffixes."""
    cmds = dry("--methods", "ladmm6", "--margin", "2", "--ft-margin", "1", "--val-snr", "10", "--snr-chunks", "10")
    tr, ft = cmds
    assert "--margin 2 --val-snr 10 --snr-chunks 10 --tag LADMM-T6_margin2_val10_c10 " in tr
    assert "--tag LADMM-T6_margin2_val10_c10_ft_0-10-20dB_20k_margin1 " in ft and ft.endswith("--val-snr 10 --margin 1")
    (tr,) = dry("--methods", "gmlamp", "--protocol", "gmlamp", "--train-snr", "scenario", "--ft-snr", "none",
                "--val-snr", "range", "--margin", "1.5")
    assert "--margin 1.5 --val-snr range --tag GM-LAMP-T8_gmlamp_trscenario_margin1p5_valrange " in tr


def _fake_run(folder, bands, log=None):
    folder.mkdir(parents=True)
    for b in bands:
        (folder / f"{folder.name}_{b}.pt").write_bytes(b"")
    if log is not None:
        (folder / "finetune_log.json").write_text(__import__("json").dumps(log))


def test_finetune_names_budgets_and_legacy_folders(tmp_path):
    """v24 names <run>_ft_<edges>dB_<k>; a v23 folder <run>-ft with the asked budget is skipped (not retrained), one with
    another budget is not mixed in (a new folder with the budget in its name is made)."""
    out = tmp_path / "o"
    base = out / "SV_UPA"
    _fake_run(base / "LADMM-T6", ["0to20dB"])
    _fake_run(base / "LADMM-T6-ft", ["0to10dB", "10to20dB"], {"args": {"iters": 20000}})
    _fake_run(base / "LAMP-T8", ["0to20dB"])
    _fake_run(base / "LAMP-T8-ft", ["0to10dB", "10to20dB"], {"args": {"iters": 15000}})
    r = subprocess.run([sys.executable, "scripts/run_suite.py", "--scenarios", "sv_upa", "--methods", "ladmm6", "lamp",
                        "--out", str(out), "--dry-run"], cwd=ROOT, capture_output=True, text=True, check=True)
    assert "--- LADMM-T6-ft: fine-tuned (20000 iterations), skipped" in r.stdout and "LADMM-T6_ft_0-10-20dB_20k" in r.stdout
    cmds = [ln[4:] for ln in r.stdout.splitlines() if ln.startswith(">>> ")]
    assert len(cmds) == 1 and "--tag LAMP-T8_ft_0-10-20dB_20k " in cmds[0]          # 15k is not 20k: a new folder
    assert "SV_UPA                 LADMM-T6-ft" in r.stdout                           # compared under its name
    r = subprocess.run([sys.executable, "scripts/run_suite.py", "--scenarios", "sv_upa", "--methods", "ladmm6",
                        "--ft-iters", "15000", "--ft-snr", "0", "5", "10", "15", "20", "--out", str(out), "--dry-run"],
                       cwd=ROOT, capture_output=True, text=True, check=True)
    assert "--tag LADMM-T6_ft_0-5-10-15-20dB_15k --iters 15000" in r.stdout


def test_rename_runs(tmp_path):
    import json
    out = tmp_path / "outputs"
    _fake_run(out / "SV_UPA" / "LADMM-T6-ft", ["0to10dB", "10to20dB"], {"args": {"iters": 20000, "tag": "LADMM-T6-ft"}})
    (out / "SV_UPA" / "LADMM-T6-ft" / "LADMM-T6-ft_0to10dB.mat").write_bytes(b"")
    (out / "SV_UPA" / "LADMM-T6-ft" / "README.txt").write_text("LADMM-T6-ft: LADMM-T6 fine-tuned\nLADMM-T6-ft_<band>.pt\n")
    _fake_run(out / "SV_UPA" / "LADMM-T6_s1-ft0-5-10-15-20_margin1", ["0to5dB"], {"args": {"iters": 10000}})
    _fake_run(out / "SV_UPA" / "UTRR-E-T30_paper-ft", ["0to10dB"])                      # an ensemble: no log, left
    run = lambda *x: subprocess.run([sys.executable, "scripts/rename_runs.py", "--outputs", str(out), *x], cwd=ROOT,  # noqa: E731
                                    capture_output=True, text=True, check=True).stdout
    r = run()
    assert "LADMM-T6-ft  ->  LADMM-T6_ft_0-10-20dB_20k" in r and "would be renamed" in r
    assert "LADMM-T6_s1-ft0-5-10-15-20_margin1  ->  LADMM-T6_s1_ft_0-5-10-15-20dB_10k_margin1" in r
    assert "UTRR-E-T30_paper-ft: no finetune_log.json" in r and (out / "SV_UPA" / "LADMM-T6-ft").exists()
    run("--apply")
    new = out / "SV_UPA" / "LADMM-T6_ft_0-10-20dB_20k"
    assert sorted(f.name for f in new.iterdir()) == sorted(
        ["LADMM-T6_ft_0-10-20dB_20k_0to10dB.pt", "LADMM-T6_ft_0-10-20dB_20k_10to20dB.pt",
         "LADMM-T6_ft_0-10-20dB_20k_0to10dB.mat", "README.txt", "finetune_log.json"])
    lg = json.loads((new / "finetune_log.json").read_text())
    assert lg["tag"] == lg["args"]["tag"] == new.name and lg["renamed_from"] == "LADMM-T6-ft"
    assert "LADMM-T6_ft_0-10-20dB_20k: " in (new / "README.txt").read_text()
    assert "nothing to rename" in run()                     # the ensemble stays as it is


def test_seed_grouping_old_and_new_names():
    from bsce.naming import drop_seed
    assert {drop_seed(n) for n in ("LADMM-T6_ft_0-10-20dB_20k", "LADMM-T6_s1_ft_0-10-20dB_20k",
                                   "LADMM-T6_s2_ft_0-10-20dB_20k")} == {"LADMM-T6_ft_0-10-20dB_20k"}
    assert {drop_seed(n) for n in ("LADMM-T6-ft", "LADMM-T6_s1-ft", "LADMM-T6_s2-ft")} == {"LADMM-T6-ft"}
    assert {drop_seed(n) for n in ("LADMM-T6", "LADMM-T6_s1")} == {"LADMM-T6"}

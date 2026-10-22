"""v31.3: run_suite.py --warm-start protocol|cold|warm|best and compare.py --best (every method in its best
configuration, the choice made on validation)."""
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))
import run_suite  # noqa: E402


def dry(*opts):
    r = subprocess.run([sys.executable, "scripts/run_suite.py", "--scenarios", "sv_upa", "--dry-run", *opts],
                       cwd=ROOT, capture_output=True, text=True, check=True)
    return [ln[4:] for ln in r.stdout.splitlines() if ln.startswith(">>> ") and "train.py" in ln]


def _tag(cmd):
    return cmd.split("--tag ")[1].split()[0]


def test_default_is_unchanged():
    cmds = dry("--methods", "ladmm", "gmlamp", "--protocol", "fair", "--ft-snr", "none")
    assert [_tag(c) for c in cmds] == ["LADMM-T6_fair", "GM-LAMP-T8_fair"]
    assert all("warm_start" not in c for c in cmds)


def test_cold_and_warm_for_every_method():
    for mode, val in (("cold", "false"), ("warm", "true")):
        cmds = dry("--methods", "ladmm", "gmlamp", "--protocol", "fair", "--ft-snr", "none", "--warm-start", mode)
        assert [_tag(c) for c in cmds] == [f"LADMM-T6_fair_{mode}", f"GM-LAMP-T8_fair_{mode}"]
        assert all(f"train.warm_start={val}" in c for c in cmds)


def test_best_trains_both_variants_but_end_to_end_methods_once():
    cmds = dry("--methods", "ladmmat", "gladmm", "--protocol", "fair", "--ft-snr", "none", "--warm-start", "best")
    tags = [_tag(c) for c in cmds]
    assert tags == ["LADMM-AT-T6_fair_cold", "LADMM-AT-T6_fair_warm", "G-LADMM-T8_paper"]
    assert "train.warm_start=false" in cmds[0] and "train.warm_start=true" in cmds[1]
    assert "warm_start" not in cmds[2]


def _fake_run(base, tag, vals):
    (base / tag).mkdir(parents=True)
    (base / tag / "train_log.json").write_text(json.dumps(
        {"bands": {f"b{i}": {"final_val_nmse_db": v} for i, v in enumerate(vals)}}))


def test_choice_on_validation_and_compare_exclusions(tmp_path):
    _fake_run(tmp_path, "L_cold", [-10.0, -12.0])         # mean -11.0
    _fake_run(tmp_path, "L_warm", [-10.5, -11.0])         # mean -10.75
    _fake_run(tmp_path, "G_cold", [-8.0])
    _fake_run(tmp_path, "G_warm", [-9.0])
    run_suite.choose_best_warm_start(tmp_path, {"ladmm": ["L_cold", "L_warm"], "gmlamp": ["G_cold", "G_warm"]})
    c = json.loads((tmp_path / "best_warm_start.json").read_text())
    assert c["ladmm"]["chosen"] == "L_cold" and c["ladmm"]["exclude"] == ["L_warm", "L_warm_ft_*"]
    assert c["gmlamp"]["chosen"] == "G_warm" and c["gmlamp"]["exclude"] == ["G_cold", "G_cold_ft_*"]
    # a variant without a trained network: no choice written for that method
    _fake_run(tmp_path, "N_cold", [-7.0])
    run_suite.choose_best_warm_start(tmp_path, {"nf": ["N_cold", "N_missing"]})
    assert "nf" not in json.loads((tmp_path / "best_warm_start.json").read_text())

"""The workbench (app/bsce_app.py): the commands it shows and runs, its progress reading and its local-only server.

1. every task builds commands whose options exist in the script's --help
2. the default "Train and fine-tune" command is the documented suite command, and its dry run is identical
3. every suite option of the form reaches run_suite.py (dry run accepted, expected folder names)
4. progress: stages, end-to-end steps and fine-tune steps of real console lines give the expected fractions
5. the server refuses requests without the token, from another host name, and files outside the bsce folder
6. a workbench still running on the port (an older version, another folder) is not shared: the new one takes the
   next port, names the old one on its page, and its "Stop it" button stops it
No data and no GPU needed.
"""
import json
import re
import subprocess
import sys
import threading
import urllib.error
import urllib.request
from http.server import ThreadingHTTPServer
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "app"))
import bsce_app as A  # noqa: E402

FILLED = {   # tasks that need an input before they make a command
    "matlab": {"deepmimo_dir": "D:/DeepMIMO-matlab-1.3"},
    "data_3gpp": {"sizes": "custom", "n_train": "1000", "n_val": "200", "n_test": "300", "only": ["3GPP_ULAZ"],
                  "spec": "19.2", "channel": "umi_los", "element": "omni", "suffix": "_t", "check": False, "seed": "4",
                  "force": True},
    "sionna": {"install": True, "env": "sio", "array": "ULA", "n": "2000", "scenario": "umi_nlos", "spec": "19.2",
               "element": "omni", "indoor": "0.2", "parts": "1-2", "device": "cuda", "threads": "2", "seed": "3"},
    "quadriga": {"quadriga_src": "D:/QuaDRiGa/quadriga_src", "engine": "octave", "array": "ULA_Y", "parts": "1:10",
                 "realized": True, "element": "omni", "indoor": "0.2", "seed": "7"},
    "finetune": {"src": "outputs/SV_UPA/LADMM-T6/LADMM-T6_0to20dB.pt", "tag": "LADMM-T6-ft10k", "iters": "10000",
                 "val": "file", "clip": "1", "bands": "0to10dB", "lr": "1e-4", "batch": "256", "scenario": "sv_upa",
                 "noise_model": "white", "val_snr": "0 10", "margin": "2"},
    "ensemble": {"runs": "outputs/SV_UPA/LADMM-T6_ft_0-10-20dB_20k\noutputs/SV_UPA/LADMM-T8_ft_0-10-20dB_20k",
                 "tag": "LADMM-E-T6-T8_ft", "out": "outputs2/SV_UPA"},
    "utrr_table": {"snr": "10", "K": "0 16 64 256", "bn": True, "ntrain": "20000", "epochs": "100", "patience": "5",
                   "batch": "64", "seed": "1"},
    "seeds": {"csv": "results/final/SV_UPA/nmse_vs_snr.csv"},
    "make_dataset": {"dataset": "DeepMIMO", "input": "data/raw/H_*.mat", "test": "random", "test_meas": True,
                     "name": "DM_UPA_big", "seed": "5", "force": True, "snr": "0 10 20", "n_train": "1000",
                     "n_val": "100", "n_test": "100", "noise_model": "both"},
    "compare": {"preset": "custom", "tags": "LADMM-T6-ft\nNF-LADMM-SR-T4*-ft", "ref": "LADMM-T6-ft", "no_authors": True,
                "classical": ["amp", "omp", "sd"], "smoke": True, "outputs": "outputs2", "results": "results2"},
    "depth_curves": {"band": "0to20dB", "tol": "0.05", "snr": "0 20"},
    "authors_suite": {"val_mode": "points", "val_points": "0 5 10", "margin": "2", "snr_chunks": "100", "precision": "float64"},
    "deep": {"val_mode": "range", "margin": "1.5", "noise_model": "white"},
    "combiner_check": {"runs": "outputs/SV_ULA", "scenario": "sv_ula", "tags": "G-LADMM*\nGM-LAMP-authors",
                       "combiners": ["own", "dct", "M--"], "files": "my_A.mat", "graph_phase": "target", "classical": [],
                       "n_test": "100", "out": "results/cc"},
    "transfer": {"runs": "outputs/SV_ULA", "scenario": "sv_ula", "combiner": "random_m", "seed_m": "4", "rf_chains": "96"},
    "suite": {"schedule": "e2e", "e2e_steps": "50000", "clip_mode": "value", "clip_value": "0.5",
              "train_snr": "0 10 20", "ft_snr": "none", "save_layers": False, "seeds": "0 1", "smoke": True,
              "set": "train.restore_best=true", "protocol": "fair", "out": "outputs2"},
}
_HELP = {}


def help_of(script):
    if script not in _HELP:
        _HELP[script] = subprocess.run([sys.executable, script, "--help"], cwd=ROOT, capture_output=True, text=True).stdout
    return _HELP[script]


@pytest.mark.parametrize("task", sorted(A.TASKS))
def test_every_option_exists(task):
    for values in ({}, FILLED.get(task, {})):
        try:
            cmds = A.build(task, values)
        except ValueError as e:                    # a missing input: a message, never a crash
            assert values == {} and str(e), task
            continue
        for c in cmds:
            assert A.display(c, "spyder") and A.display(c, "terminal")
            if c["kind"] != "script" or c["argv"][1].endswith("run_suite.py") and "--help" in c["argv"]:
                continue
            h = help_of(c["argv"][1])
            bad = [a for a in c["argv"][2:] if a.startswith("--") and not re.search(re.escape(a) + r"\b", h)]
            assert not bad, (task, bad)


def dry(argv):
    r = subprocess.run([sys.executable, *argv, "--dry-run"], cwd=ROOT, capture_output=True, text=True)
    assert r.returncode == 0, r.stderr
    return [ln for ln in r.stdout.splitlines() if ln.startswith(">>> ")]


def test_default_suite_is_the_documented_command():
    (c,) = A.build("suite", {"device": "auto"})
    assert c["argv"][1:] == ["scripts/run_suite.py", "--scenarios", "sv_ula", "sv_upa", "deepmimo_ula", "deepmimo_upa",
                             "--methods", "ladmm6", "ladmm4", "lamp8", "gmlamp8", "nfladmmsr4", "--device", "auto"]
    guide = ["scripts/run_suite.py", "--scenarios", "sv_ula", "sv_upa", "deepmimo_ula", "deepmimo_upa",
             "--methods", "ladmm6", "ladmm4", "lamp8", "gmlamp8", "nfladmmsr4"]
    assert dry(c["argv"][1:]) == dry(guide)


def test_authors_protocol_task_is_the_documented_command():
    (c,) = A.build("authors_suite", {"device": "cuda"})
    assert c["argv"][1:] == ["scripts/run_suite.py", "--scenarios", "sv_ula", "sv_upa", "deepmimo_ula", "deepmimo_upa",
                             "--methods", "ladmm6", "ladmm4", "lamp8", "gmlamp8", "nfladmmsr4", "--protocol", "gmlamp",
                             "--train-snr", "scenario", "--ft-snr", "none", "--device", "cuda"]     # GUIDE 4.8
    lines = dry(c["argv"][1:])
    assert len(lines) == 4 * 5 and not any("finetune_bands.py" in ln for ln in lines)
    assert all("--protocol gmlamp" in ln and "_gmlamp_trscenario " in ln and "--snr" not in ln for ln in lines)
    (c,) = A.build("authors_suite", {"precision": "float64", "scenarios": ["sv_upa"]})
    assert dry(c["argv"][1:])[0].count("model.precision=float64") == 1


def test_suite_options_reach_run_suite():
    cmds = A.build("suite", dict(FILLED["suite"], smoke=False, seeds="0", scenarios=["sv_upa"]))
    (c,) = cmds
    lines = dry(c["argv"][1:])
    assert "--tag LADMM-T6_fair_tr0-10-20_e2e50k_clip0p5_restore-best-true " in lines[0]
    assert "train.steps=50000" in lines[0] and "train.grad_clip=0.5" in lines[0]
    with pytest.raises(ValueError):                    # snapshots need a layer-wise schedule (v25)
        A.build("suite", dict(FILLED["suite"], save_layers=True))
    with pytest.raises(ValueError):
        A.build("deep", {"protocol": "e2e"})
    assert not any("finetune_bands.py" in ln for ln in lines)          # --ft-snr none
    split = A.build("suite", {"split": "method", "seeds": "0 2", "scenarios": ["sv_upa", "sv_ula"]})
    assert len(split) == 2 * 5 and split[-1]["argv"][-2:] == ["--seed", "2"]
    assert A.build("suite", {"clip_mode": "none"})[0]["argv"].count("none") == 1
    with pytest.raises(ValueError):
        A.build("suite", {"train_snr": "20 0"})
    with pytest.raises(ValueError):
        A.build("suite", {"methods": [dict(r, on=False) for r in A.default_methods({})]})


def test_spyder_display_keeps_patterns():
    (c,) = A.build("compare", {"preset": "ft"})
    assert A.display(c, "spyder").startswith("%run -G scripts/compare.py") and '"*-ft"' in A.display(c, "spyder")
    (c,) = A.build("compare", {"preset": "all"})
    assert A.display(c, "spyder").startswith("%run scripts/compare.py")


def test_progress_reading():
    p = A.Progress()
    p.add_cells([dict(label="train", state="todo"), dict(label="fine-tune", state="todo")], 0)
    p.start_next(0)
    p.feed(">>> scripts/train.py --scenario configs/scenarios/sv_upa.yaml --tag LADMM-T2 --set model.T=2")
    for t in (1, 2):
        for k in range(5):
            p.feed(f"LADMM-soft H-update T={t}{'' if k == 0 else f' trainrate={k - 1}'}  [fine tuning H_matrix_0] lr=1e-4")
    assert p.detail == "layer 2 of 2, stage 5 of 5" and 0.85 < p.frac < 1
    p.feed("  i=400     nmse=-6.1140 dB (best=-6.2000)  1.0s")
    assert "-6.1140" in p.metric
    p.start_next(0)
    assert p.cells[0]["state"] == "done" and p.cells[1]["state"] == "run"
    p.feed(">>> scripts/finetune_bands.py --snr 0 10 20 --from x.pt --tag LADMM-T2-ft --iters 1000 --device cpu")
    p.feed("=== LADMM-T2-ft | band 0to10dB | train SNR ~ U(0,10) | validation: x")
    p.feed("  step   1000 lr=5.0e-06  val nmse=-5.9 dB (best -5.95)  3s")
    p.feed("=== LADMM-T2-ft | band 10to20dB | train SNR ~ U(10,20) | validation: x")
    p.feed("  step    500 lr=5.0e-05  val nmse=-7.0 dB (best -7.05)  3s")
    assert abs(p.frac - 0.75) < 1e-9 and "10to20dB" in p.detail
    p.end_cmd(0, True)
    assert [c["state"] for c in p.cells] == ["done", "done"]
    e = A.Progress()
    e.add_cells([dict(label="t", state="todo")], 0)
    e.start_next(0)
    e.feed(">>> scripts/train.py --tag X --set model.T=6 train.schedule=steps train.steps=30000")
    e.feed("  step=7500    val nmse=-9.1 dB (best=-9.2)  lr=1e-4  10.0s")
    assert e.frac == 0.25


def test_suite_plan_marks_finished_steps(tmp_path):
    out = tmp_path / "o" / "SV_UPA" / "LADMM-T6"
    out.mkdir(parents=True)
    (out / "LADMM-T6_0to20dB.pt").write_bytes(b"")
    cells = A.suite_plan([sys.executable, "scripts/run_suite.py", "--scenarios", "sv_upa", "--methods", "ladmm6", "lamp",
                          "--out", str(tmp_path / "o"), "--dry-run"])
    assert [c["state"] for c in cells] == ["skip", "todo", "todo", "todo"]
    assert cells[1]["label"] == "fine-tune LADMM-T6_ft_0-10-20dB_20k on SV_UPA"


@pytest.fixture(scope="module")
def server():
    srv = ThreadingHTTPServer(("127.0.0.1", 0), A.H)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    yield f"http://127.0.0.1:{srv.server_address[1]}"
    srv.shutdown()


def req(url, token=None, body=None, host=None):
    r = urllib.request.Request(url, data=None if body is None else json.dumps(body).encode(), method="POST" if body is not None else "GET")
    if token:
        r.add_header("X-Token", token)
    if host:
        r.add_header("Host", host)
    try:
        with urllib.request.urlopen(r, timeout=30) as f:
            return f.status, f.read()
    except urllib.error.HTTPError as e:
        return e.code, e.read()


def test_server_is_local_only(server):
    assert req(server + "/api/jobs")[0] == 403                                  # no token
    assert req(server + "/api/jobs", token="wrong")[0] == 403
    assert req(server + "/api/jobs", token=A.TOKEN, host="evil.example")[0] == 403  # DNS rebinding
    st, body = req(server + "/api/plan", token=A.TOKEN, body={"task": "suite", "values": {}})
    assert st == 200 and json.loads(body)["commands"][0]["terminal"].startswith("python scripts/run_suite.py")
    assert req(server + "/api/run", body={"task": "suite", "values": {}})[0] == 403   # cannot start a run without the token
    assert req(server + f"/file?p=../../etc/passwd&t={A.TOKEN}")[0] == 404
    assert req(server + f"/file?p=configs/scenarios/sv_upa.yaml&t={A.TOKEN}")[0] == 404   # only result-type files
    st, html = req(server + "/")
    assert st == 200 and A.TOKEN.encode() in html


def test_data_options_from_the_form():
    (c,) = A.build("suite", {"val_mode": "points", "val_points": "10", "margin": "2", "ft_margin": "1", "snr_chunks": "10",
                             "scenarios": ["sv_upa"], "methods": [dict(r, on=r["key"] == "ladmm") for r in A.default_methods({"ladmm": "6"})]})
    tr, ft = dry(c["argv"][1:])
    assert "--margin 2 --val-snr 10 --snr-chunks 10 --tag LADMM-T6_margin2_val10_c10 " in tr
    assert "_ft_0-10-20dB_20k_margin1 " in ft and ft.endswith("--val-snr 10 --margin 1")
    with pytest.raises(ValueError):
        A.build("suite", {"margin": "-1"})


def test_noise_model_options_from_the_form():
    (c,) = A.build("suite", {"noise_model": "white", "scenarios": ["sv_upa"],
                             "methods": [dict(r, on=r["key"] == "nfladmm") for r in A.default_methods({"nfladmm": "8"})]})
    tr, ft = dry(c["argv"][1:])
    assert "--noise-model white --tag NF-LADMM-S-T8 " in tr
    assert "outputs/SV_UPA_white/NF-LADMM-S-T8/NF-LADMM-S-T8_0to20dB.pt" in ft.replace("\\", "/")
    (c,) = A.build("compare", {"noise_model": "both", "scenarios": ["sv_ula"], "preset": "custom", "tags": "NF-LADMM*"})
    assert c["argv"][1].endswith("compare_noise.py") and "--nf-matched" in c["argv"] and "--tags" in c["argv"]
    h = help_of(c["argv"][1])
    assert not [x for x in c["argv"][2:] if x.startswith("--") and x not in h]
    (c,) = A.build("compare", {"noise_model": "white", "scenarios": ["sv_ula"]})
    assert c["argv"][1].endswith("compare.py") and c["argv"][c["argv"].index("--noise-model") + 1] == "white"
    (c,) = A.build("suite", {"scenarios": ["sv_upa"]})
    assert "--noise-model" not in c["argv"]                 # default: as before


def test_finetune_task_needs_no_name():
    (c,) = A.build("finetune", {"src": "outputs/SV_UPA/LADMM-T6"})
    assert "--tag" not in c["argv"] and "--iters" not in c["argv"] and c["argv"][c["argv"].index("--from") + 1] == \
        "outputs/SV_UPA/LADMM-T6"
    (c,) = A.build("finetune", {"src": "outputs/SV_UPA/LADMM-T6", "iters": "10000", "tag": "mine"})
    assert c["argv"][c["argv"].index("--tag") + 1] == "mine" and c["argv"][c["argv"].index("--iters") + 1] == "10000"
    p = A.Progress()
    p.feed(">>> scripts/finetune_bands.py --from outputs/SV_UPA/LADMM-T6 --snr 0 10 20 --device cpu")
    assert p.ctx["iters"] == 20000 and p.ctx["nbands"] == 2              # its default budget, for the progress bar
    assert A.TAG_PRESETS["ft"][0] == "*_ft_*" and "*-ft" in A.TAG_PRESETS["ft"]


# ---------------- one workbench per port (an old workbench still running must not hide the new one) ----------------
def free_port():
    import socket
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def start_app(port, folder=None):
    """A real workbench process (--no-browser); returns (process, url it printed)."""
    import os
    app = (Path(folder) if folder else ROOT) / "app" / "bsce_app.py"
    p = subprocess.Popen([sys.executable, str(app), "--port", str(port), "--no-browser"], stdout=subprocess.PIPE,
                         stderr=subprocess.STDOUT, text=True, start_new_session=(os.name != "nt"))
    lines = []
    for ln in p.stdout:
        lines.append(ln)
        if ln.startswith("bsce workbench") and "http://" in ln:          # its own address (not a NOTE line)
            return p, re.search(r"http://\S+/", ln).group(0), "".join(lines)
    raise AssertionError("".join(lines))


def test_old_workbench_on_the_port_is_found_and_can_be_stopped(tmp_path):
    import socket
    port = free_port()
    assert A.probe_port(port) is None                                    # nothing listens
    plain = socket.socket(); plain.bind(("127.0.0.1", port)); plain.listen()
    try:
        assert A.probe_port(port)["kind"] == "other"                   # some other program
    finally:
        plain.close()
    # an "old" workbench: a copy of this app without /api/whoami (as v16-v20), in another folder
    old = tmp_path / "old_bsce"
    (old / "app").mkdir(parents=True)
    src = (ROOT / "app" / "bsce_app.py").read_text(encoding="utf-8")
    src = src.replace('if u.path == "/api/whoami":', 'if u.path == "/api/whoami-removed":')
    (old / "app" / "bsce_app.py").write_text(src, encoding="utf-8")
    import shutil
    shutil.copytree(ROOT / "app" / "static", old / "app" / "static")
    p_old, url_old, _ = start_app(port, old)
    p_new = None
    try:
        assert url_old.endswith(f":{port}/")
        info = A.probe_port(port)
        assert info["kind"] == "bsce" and info["root"] == ""             # an older version: folder unknown
        p_new, url_new, out = start_app(port)
        assert not url_new.endswith(f":{port}/")                         # the new one never shares the old port
        assert "another bsce workbench is still running" in out
        new_port = int(url_new.rstrip("/").rsplit(":", 1)[1])
        me = A.probe_port(new_port)
        assert me["kind"] == "bsce" and me["version"] == A.VERSION and Path(me["root"]) == ROOT
        html = urllib.request.urlopen(url_new, timeout=30).read().decode()
        tok = re.search(r'name="token" content="([^"]+)"', html).group(1)
        st, body = req(url_new + "api/meta", token=tok)
        m = json.loads(body)
        assert st == 200 and m["version"] == A.VERSION and [o["port"] for o in m["others"]] == [port]
        st, body = req(url_new + "api/stop_other", token=tok, body={"port": port})
        r = json.loads(body)
        assert st == 200 and r["ok"], r
        assert A.probe_port(port) is None and p_old.wait(timeout=30) is not None
        st, body = req(url_new + "api/meta", token=tok)
        assert json.loads(body)["others"] == []
        st, body = req(url_new + "api/stop_other", token=tok, body={"port": port})
        assert not json.loads(body)["ok"]                                # nothing left to stop
    finally:
        for p in (p_old, p_new):
            if p and p.poll() is None:
                p.kill()


def test_fair_snr_from_the_form():
    """v26: 'Training SNRs: fair' (+ grid step) in the suite and fine-tune tasks; the errors; default unchanged."""
    meth = [dict(r, on=r["key"] == "ladmm") for r in A.default_methods({"ladmm": "6"})]
    (c,) = A.build("suite", {"scenarios": ["sv_upa"], "methods": meth, "snr_dist": "fair", "snr_step": "0.2"})
    tr, ft = dry(c["argv"][1:])
    assert "--snr-dist fair --snr-step 0.2 --tag LADMM-T6_fair0p2 " in tr
    assert "--tag LADMM-T6_fair0p2_ft_0-10-20dB_20k_fair0p2" in ft and "--snr-dist fair" in ft
    (c,) = A.build("finetune", {"src": "outputs/SV_UPA/LADMM-T6", "snr_dist": "fair"})
    assert c["argv"][c["argv"].index("--snr-dist") + 1] == "fair" and "--snr-step" not in c["argv"]
    h = help_of(c["argv"][1])
    assert not [x for x in c["argv"][2:] if x.startswith("--") and x not in h]
    for bad in ({"snr_dist": "fair", "snr_chunks": "100"}, {"snr_step": "0.2"}, {"snr_dist": "fair", "snr_step": "-1"}):
        with pytest.raises(ValueError):
            A.build("suite", dict({"scenarios": ["sv_upa"], "methods": meth}, **bad))
    (c,) = A.build("suite", {"scenarios": ["sv_upa"], "methods": meth})
    assert "--snr-dist" not in c["argv"]


def test_v27_forms():
    """Per-method settings (own command per settings group), descriptions, the fine-tune fields, --ft-patience, the
    one-network preset with exclusions, the quick test set."""
    rows = A.default_methods({"ladmm": "6", "nfladmmsr": "4", "lamp": "8"})
    assert all(r["desc"] for r in rows) and "sigma=residual" in next(r for r in rows if r["key"] == "nfladmmsr")["desc"]
    for r in rows:
        if r["key"] == "nfladmmsr":
            r["set"] = "model.sigma=residual; train.loss=log_nse"
    cmds = A.build("suite", {"scenarios": ["sv_upa"], "methods": rows, "set": "train.restore_best=true"})
    assert len(cmds) == 2
    l0, l1 = " ".join(dry(cmds[0]["argv"][1:])), " ".join(dry(cmds[1]["argv"][1:]))
    assert "--tag LADMM-T6_restore-best-true " in l0 and "--tag LAMP-T8_restore-best-true " in l0 and "NF-LADMM" not in l0
    assert "--tag NF-LADMM-SR-T4_restore-best-true_sigma-residual_loss-log-nse " in l1 and "LADMM-T6" not in l1
    assert "model.sigma=residual" not in l0                                  # the row's settings: that method only
    assert "[model.sigma=residual train.loss=log_nse]" in cmds[1]["label"]
    bad = [dict(r, set="model.sigma") if r["key"] == "lamp" else r for r in rows]
    with pytest.raises(ValueError):
        A.build("suite", {"scenarios": ["sv_upa"], "methods": bad})
    (c,) = A.build("suite", {"scenarios": ["sv_upa"], "ft_patience": "0"})
    assert c["argv"][c["argv"].index("--ft-patience") + 1] == "0"
    (c,) = A.build("suite", {"scenarios": ["sv_upa"]})
    assert "--ft-patience" not in c["argv"]
    (c,) = A.build("finetune", {"src": "outputs/SV_ULA/LADMM-T12/layers/T06_LADMM-T12_0to20dB.pt", "scenario": "sv_ula",
                                "val_snr": "10", "margin": "1"})
    a = c["argv"]
    assert a[a.index("--scenario") + 1] == "sv_ula" and a[a.index("--val-snr") + 1] == "10" and "--margin" in a
    (c,) = A.build("compare", {"scenarios": ["sv_upa"], "preset": "onenet"})
    a = c["argv"]
    assert a[a.index("--tags") + 1] == "*-T[0-9]*" and "*_ft_*" in a[a.index("--exclude"):]
    import fnmatch
    sel = lambda n: fnmatch.fnmatchcase(n, "*-T[0-9]*") and not any(fnmatch.fnmatchcase(n, p) for p in A.TAG_EXCLUDE["onenet"])  # noqa: E731
    assert [n for n in ("LADMM-T6", "LADMM-T6_fair", "LAMP-T8_e2e30k", "UTRR-T30_paper", "LADMM-T6_ft_0-10-20dB_20k",
                        "LAMP-T8_paper_trscenario", "LADMM-T6_tr0-10-20", "LADMM-T6_trm10-30", "LADMM-T6-ft",
                        "GM-LAMP-authors") if sel(n)] == ["LADMM-T6", "LADMM-T6_fair", "LAMP-T8_e2e30k", "UTRR-T30_paper"]
    assert all((ROOT / t).exists() for t in A.TEST_SETS["quick"])



def test_v28_forms():
    """Ensemble after training (off by default), polish in Compare (off by default), the CADMM row, LADMM options."""
    (c,) = A.build("suite", {"scenarios": ["sv_ula"]})
    assert "--ensemble" not in c["argv"]
    (c,) = A.build("suite", {"scenarios": ["sv_ula"], "ensemble": "ladmm6 ladmm8"})
    a = c["argv"]
    assert a[a.index("--ensemble") + 1:a.index("--ensemble") + 3] == ["ladmm6", "ladmm8"]
    l = " ".join(dry(a[1:]))
    assert "make_ensemble.py" in l and "ENS-LADMM-T6+LADMM-T8_ft_0-10-20dB_20k" in l
    for bad in ("ladmm6", "ladmm6 foo4"):
        with pytest.raises(ValueError):
            A.build("suite", {"scenarios": ["sv_ula"], "ensemble": bad})
    (c,) = A.build("compare", {"scenarios": ["sv_ula"]})
    assert "--polish" not in c["argv"]
    (c,) = A.build("compare", {"scenarios": ["sv_ula"], "polish": "band"})
    assert c["argv"][c["argv"].index("--polish") + 1] == "band"
    rows = A.default_methods({"ladmm": "6"})
    keys = [r["key"] for r in rows]
    assert "cadmm" in keys and not next(r for r in rows if r["key"] == "cadmm")["on"]
    d = next(r for r in rows if r["key"] == "ladmm")["desc"]
    assert "adaptive" in d and "whitened" in d and "analytic" in d
    import fnmatch
    assert any(fnmatch.fnmatchcase("ENS-LADMM-T6+LADMM-T8", p) for p in A.TAG_EXCLUDE["onenet"])


def test_v29_forms():
    """The Graph-LADMM rows (not ticked) and the transfer task."""
    rows = A.default_methods({"ladmm": "6"})
    for k in ("gladmm", "gladmmf", "gladmmany"):
        r = next(r for r in rows if r["key"] == k)
        assert not r["on"] and r["desc"]
    (c,) = A.build("transfer", {"runs": "outputs/SV_ULA", "scenario": "deepmimo_ula"})
    a = c["argv"]
    assert a[a.index("--scenario") + 1] == "configs/scenarios/deepmimo_ula.yaml" and "--graph-phase" in a
    assert "--combiner" not in a
    (c,) = A.build("transfer", {"runs": "outputs/SV_ULA", "scenario": "sv_ula", "combiner": "random", "seed": "5",
                                "graph_phase": "own"})
    a = c["argv"]
    assert a[a.index("--combiner") + 1] == "random:5" and "--graph-phase" not in a
    with pytest.raises(ValueError):
        A.build("transfer", {"runs": "", "scenario": "sv_ula"})


def test_v30_forms():
    """The suite's Graph-LADMM protocol box, the transfer task with another M, the combiner check, unique names."""
    (c,) = A.build("suite", {"force_protocol": True})
    assert "--force-protocol" in c["argv"]
    (c,) = A.build("suite", {})
    assert "--force-protocol" not in c["argv"]
    (c,) = A.build("transfer", {"runs": "outputs/SV_ULA", "scenario": "sv_ula", "combiner": "random_m", "seed_m": "4",
                                "rf_chains": "96"})
    a = c["argv"]
    assert a[a.index("--combiner") + 1] == "random:4:96" and a[a.index("--out") + 1].endswith("random-M96")
    for bad in ({"rf_chains": "x"}, {"rf_chains": "0"}):
        with pytest.raises(ValueError):
            A.build("transfer", {"runs": "outputs/SV_ULA", "scenario": "sv_ula", "combiner": "random_m", **bad})
    (c,) = A.build("combiner_check", {})
    a = c["argv"]
    assert a[1].endswith("check_combiners.py") and "--combiners" not in a and "--classical" not in a
    (c,) = A.build("combiner_check", FILLED["combiner_check"])
    a = c["argv"]
    assert a[a.index("--combiners") + 1:a.index("--combiners") + 4] == ["own", "dct", "M--"]
    assert a[a.index("--tags") + 1:a.index("--tags") + 3] == ["G-LADMM*", "GM-LAMP-authors"]
    assert a[a.index("--classical") + 1] == "" and "--graph-phase" in a and "--combiner-files" in a
    with pytest.raises(ValueError):
        A.build("combiner_check", {"combiners": [], "files": ""})
    # names: task ids and field keys unique, every method row has a description, every script of a command exists
    ids = [t["id"] for st in A.STEPS for t in st["tasks"]] + [t["id"] for t in A.EXTRA_TASKS]
    assert len(ids) == len(set(ids)), "duplicate task id"
    for st in A.STEPS:
        for t in st["tasks"]:
            keys = [f["key"] for f in t["fields"]]
            assert len(keys) == len(set(keys)), (t["id"], "duplicate field")
    meth = [k for k, _, _ in A.suite_methods()]
    assert len(meth) == len(set(meth)) and set(A.METHOD_INFO) == set(meth)
    src = (ROOT / "app" / "bsce_app.py").read_text(encoding="utf-8")
    for sc in set(re.findall(r'S\("([A-Za-z0-9_]+\.py)"', src)):
        assert (ROOT / "scripts" / sc).exists(), sc


#!/usr/bin/env python3
"""bsce workbench: one window for the whole workflow, from the conda environment to the final figures.

Start it (Windows: Anaconda Prompt, or double-click start_app.bat in the bsce folder):
    python app/bsce_app.py                 opens http://127.0.0.1:8765 in your browser
    python app/bsce_app.py --port 8800 --no-browser

Only the Python standard library is used, so it also runs from the base Anaconda Python before the bsce
environment exists (step 0 creates it). Every step is a form; the commands it will run are shown live, exactly as
they are executed (the server builds them, the page only displays them). Runs go into a queue (one at a time, for
the GPU) or run in parallel; their console output is saved in app/runs/ and their progress is followed on the page.
Closing the browser does not stop a run; closing this program does (start it again and run the same step: finished
work is skipped and an interrupted training resumes from its checkpoint).
"""
from __future__ import annotations

import argparse
import glob
import json
import mimetypes
import os
import re
import secrets
import shlex
import shutil
import signal
import socket
import subprocess
import sys
import threading
import time
import webbrowser
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

APP = Path(__file__).resolve().parent
ROOT = APP.parent                       # the bsce folder
STATIC = APP / "static"
RUNS = APP / "runs"
WIN = os.name == "nt"
TOKEN = secrets.token_urlsafe(16)
try:                                    # the code version, shown in the page header and in /api/whoami
    VERSION = (ROOT / "VERSION.txt").read_text(encoding="utf-8").split()[0]
except (OSError, IndexError):
    VERSION = "?"
OTHERS: list = []                       # other workbenches found on the ports at start (see find_others)

# ----------------------------------------------------------------------------------------------------------------
# settings and detection
# ----------------------------------------------------------------------------------------------------------------
SETTINGS_FILE = APP / "settings.json"
DEFAULT_SETTINGS = {"python": "", "conda": "", "env_name": "bsce", "device": "cuda", "show": "spyder"}


def load_settings():
    s = dict(DEFAULT_SETTINGS)
    try:
        s.update(json.loads(SETTINGS_FILE.read_text(encoding="utf-8")))
    except (OSError, ValueError):
        pass
    return s


SETTINGS = load_settings()


def save_settings():
    try:
        SETTINGS_FILE.write_text(json.dumps(SETTINGS, indent=1), encoding="utf-8")
    except OSError:
        pass


def find_conda():
    c = os.environ.get("CONDA_EXE")
    if c and Path(c).exists():
        return c
    pre = Path(sys.prefix)
    cands = [pre / "Scripts" / "conda.exe", pre.parent.parent / "Scripts" / "conda.exe",
             pre / "bin" / "conda", pre.parent.parent / "bin" / "conda"]
    for base in (Path.home(), Path(os.environ.get("ProgramData", "C:/ProgramData")),
                 Path(os.environ.get("LOCALAPPDATA", str(Path.home())))):
        for n in ("anaconda3", "miniconda3", "Anaconda3", "Miniconda3", "miniforge3", "mambaforge"):
            cands += [base / n / "Scripts" / "conda.exe", base / n / "bin" / "conda"]
    for p in cands:
        if p.exists():
            return str(p)
    w = shutil.which("conda")
    return w or ""


def env_python(name, conda=None):
    """Python executable of the conda environment `name` ('' if not found)."""
    if Path(sys.prefix).name.lower() == name.lower():
        return sys.executable
    conda = conda if conda is not None else conda_exe()
    if not conda:
        return ""
    try:
        out = subprocess.run([conda, "env", "list", "--json"], capture_output=True, text=True, timeout=90).stdout
        envs = json.loads(out).get("envs", [])
    except (OSError, ValueError, subprocess.SubprocessError):
        return ""
    for e in envs:
        if Path(e).name.lower() == name.lower():
            p = Path(e) / ("python.exe" if WIN else "bin/python")
            if p.exists():
                return str(p)
    return ""


def conda_exe():
    return SETTINGS.get("conda") or find_conda()


_PY_CACHE = {}          # env name -> (python path or '', time of the lookup)


def cached_env_python(name):
    p, t = _PY_CACHE.get(name, ("", 0.0))
    if not p and time.time() - t > 30:          # a missing environment is looked up again after 30 s
        p = env_python(name)
        _PY_CACHE[name] = (p, time.time())
    return p


def step_python():
    """The Python that runs the bsce scripts: the setting, else the bsce environment, else this Python."""
    p = SETTINGS.get("python")
    if p and Path(p).exists():
        return p
    return cached_env_python(SETTINGS.get("env_name", "bsce")) or sys.executable


def python_source():
    p = SETTINGS.get("python")
    if p and Path(p).exists():
        return "set by you"
    name = SETTINGS.get("env_name", "bsce")
    if cached_env_python(name):
        return f"conda environment '{name}'"
    return f"this app's Python (environment '{name}' not found yet: step 0 creates it)"


# ----------------------------------------------------------------------------------------------------------------
# what exists in the bsce folder (read from the files, so the app follows the code)
# ----------------------------------------------------------------------------------------------------------------
def suite_methods():
    """[(key, label, default T)] from scripts/run_suite.py METHODS."""
    txt = (ROOT / "scripts" / "run_suite.py").read_text(encoding="utf-8")
    out = re.findall(r'^\s+"([a-z]+)":\s*\("([^"]+)",\s*"[^"]+",\s*(\d+)\)', txt, re.M)
    return [(k, lab, int(t)) for k, lab, t in out]


MAIN_SCEN = ["sv_ula", "sv_upa", "deepmimo_ula", "deepmimo_upa", "3gpp_upa", "3gpp_ula"]


def scenario_files():
    return sorted(p.stem for p in (ROOT / "configs" / "scenarios").glob("*.yaml"))


def scenario_name(stem):
    """'name:' line of a scenario yaml (SV_UPA for sv_upa)."""
    try:
        m = re.search(r"^name:\s*(\S+)", (ROOT / "configs" / "scenarios" / f"{stem}.yaml").read_text(encoding="utf-8"), re.M)
        return m.group(1) if m else stem.upper()
    except OSError:
        return stem.upper()


def output_dirs():
    """outputs and every other folder of trained networks (outputs_*, --out of the suite)."""
    return [p.name for p in sorted(ROOT.iterdir()) if p.is_dir() and p.name.startswith("outputs")]


def output_networks():
    """Every trained network file usable as a fine-tune source: outputs*/<SC>/<run>/<run>_<band>.pt."""
    res = []
    for d in output_dirs():
        res += _networks_in(ROOT / d)
    return res


def _networks_in(base):
    res = []
    for sc in sorted(base.iterdir()):
        if not sc.is_dir():
            continue
        for run in sorted(sc.iterdir()):
            if run.is_dir():
                for p in sorted(run.glob(f"{run.name}_*.pt")):
                    res.append(str(p.relative_to(ROOT)).replace("\\", "/"))
                for p in sorted((run / "layers").glob(f"T[0-9]*_{run.name}_*.pt")):   # depth snapshots (--save-layers)
                    res.append(str(p.relative_to(ROOT)).replace("\\", "/"))
    return res


# ----------------------------------------------------------------------------------------------------------------
# the steps: fields shown on the page + the commands they make
# ----------------------------------------------------------------------------------------------------------------
TORCH = {"cu126": "https://download.pytorch.org/whl/cu126", "cu118": "https://download.pytorch.org/whl/cu118",
         "cu124": "https://download.pytorch.org/whl/cu124", "cu128": "https://download.pytorch.org/whl/cu128",
         "cpu": "https://download.pytorch.org/whl/cpu"}
TORCH_CHECK = ("import torch; ok = torch.cuda.is_available(); "
               "print('torch', torch.__version__, '| CUDA available:', ok, '|', torch.cuda.get_device_name(0) if ok else 'no GPU')")


def F(key, label, kind="text", default="", help="", options=None, show_if=None, placeholder="", wide=False):
    d = dict(key=key, label=label, kind=kind, default=default, help=help, wide=wide)
    if options is not None:
        d["options"] = options
    if show_if:
        d["show_if"] = show_if
    if placeholder:
        d["placeholder"] = placeholder
    return d


def scen_field(default=("sv_ula", "sv_upa", "deepmimo_ula", "deepmimo_upa"), key="scenarios",
               label="Scenarios", help="the four GM-LAMP data sets; other yaml files of configs/scenarios below"):
    return F(key, label, "scenarios", list(default), help)


def device_field():
    return F("device", "Device", "select", "@device", "cuda = the GPU; auto = the GPU if there is one",
             [["cuda", "cuda (GPU)"], ["cpu", "cpu"], ["auto", "auto"]])


def methods_field(default):
    return F("methods", "Methods and number of layers", "methods", default,
             "tick a method; T = layers (several depths: '6 4' trains both)")


def S(script, *args):
    return {"argv": ["{py}", f"scripts/{script}", *[str(a) for a in args if a is not None]], "kind": "script"}


def split_edges(text, allow=()):
    t = str(text).strip()
    if t.lower() in allow:
        return [t.lower()]
    vals = t.replace(",", " ").split()
    try:
        f = [float(v) for v in vals]
    except ValueError:
        raise ValueError(f"'{text}': numbers separated by spaces, e.g. 0 10 20")
    if len(f) < 2 or any(b <= a for a, b in zip(f, f[1:])):
        raise ValueError(f"'{text}': at least two increasing values, e.g. 0 20 or 0 10 20")
    return [f"{x:g}" for x in f]


def lines(text):
    return [ln.strip() for ln in str(text or "").splitlines() if ln.strip() and not ln.strip().startswith("#")]


def scen_values(v, key="scenarios"):
    s = list(v.get(key) or [])
    if not s:
        raise ValueError("choose at least one scenario")
    return s


def method_tokens(rows):
    out = []
    for r in rows or []:
        if not r.get("on"):
            continue
        Ts = str(r.get("T", "")).replace(",", " ").split()
        if not Ts:
            out.append(r["key"])
        for t in Ts:
            if not t.isdigit() or int(t) < 1:
                raise ValueError(f"{r['key']}: T must be whole numbers, e.g. 6 or '6 4'")
            out.append(f"{r['key']}{t}")
    if not out:
        raise ValueError("tick at least one method")
    return out


def set_args(v):
    s = lines(v.get("set", ""))
    for o in s:
        if "=" not in o or o.startswith("-"):
            raise ValueError(f"override '{o}': write key=value, e.g. train.restore_best=true")
    return (["--set", *s] if s else [])


METHOD_INFO = {   # one line per method key: what it is, what to know, its model options (key=value in its settings)
    "ladmm": "proposed; full N×N operator per layer. Options: model.operator=diag|lowrank|kron|psd|kronprior|diagprior|"
             "analytic|analytic1, model.threshold=scalar|group|group_k|group_kc|adaptive|firm|gm, "
             "model.data_step=plain|whitened",
    "lamp": "Borgerding; Wei–Hu–Dai, soft threshold. Option: model.untied=false",
    "gmlamp": "Wei–Hu–Dai, Gaussian-mixture shrinkage. Options: model.K=4, model.gm_init=spread. v31.1: the identical components of the original init are separated before training (train.gm_symmetry); retrain GM-LAMP networks from older versions (check_gm_components.py)",
    "lista": "complex LISTA, one shared B. Option: model.untied=false",
    "alista": "analytic W (computed once per data set)",
    "lvamp": "Borgerding, SVD form, Bernoulli–Gaussian; LMMSE for white noise (model.noise_model=colored: the folded-noise weights)",
    "admmnet": "THADMM-Net loop (Klioui), learned Gram per layer",
    "cadmm": "complex ADMM-Net, untied M1 (N×M), M2 (N×N), over-relaxation (Johnston et al., TSP 2021); "
             "LADMM-F is its tied case",
    "gladmm": "proposed Graph-LADMM, 54 parameters (all trained; v31 no longer stores the 2 that never reached the "
              "output): magnetic graph TV + energy gate on the beam graph, analytic x-step, works with any combiner (any "
              "number of RF chains) and data set. Always its own end-to-end training (protocol 'paper'; layer-wise lost "
              "1.7 dB) unless 'same protocol for Graph-LADMM' is ticked. SV ULA: 9,000 steps reach the plateau, SV UPA "
              "3,000 (settings cell train.steps=9000 for a short run; default 30,000). Options: "
              "model.zstep=l1|gm|gate|gsoft|gtv|gtv_gate|gtv_gm|gm_graph|gtv_gm_graph|rwl1|gtv_rwl1, "
              "model.sigma=residual|blind|known, model.data_step=whitened|plain, model.beam_prior=none|gate",
    "gladmmf": "Graph-LADMM with a free H_t per layer and the graph gate (one combiner, like LADMM-F). Takes the protocol "
               "of the form like every method; its own end-to-end training: protocol 'paper'",
    "gladmmany": "Graph-LADMM trained on a new random combiner every minibatch (any combiner, any data set); always its "
                 "own end-to-end training (protocol 'paper'), like gladmm",
    "gladmmmix": "v31: Graph-LADMM trained on a mixture of combiners (M 64-160; random ±1, Gaussian, partially connected, "
                 "DCT), for other sizes and structured combiners; own end-to-end training. Options: "
                 "train.combiner_M=[...], train.combiner_types=[pm1,gauss,sparse,dct]",
    "gladmmbp": "v31: Graph-LADMM + the beam prior (gate logit + beta_t log beam power; 318 parameters). +0.3/+0.4 dB on "
                "DeepMIMO ULA/UPA, about 0 on SV; still any combiner; own end-to-end training",
    "gladmmrw": "v31: the graph-reweighted l1 (uploaded package) unrolled: reweighted soft threshold from the previous "
                "layer, the package's plain data step, no TV; 44 parameters. SV UPA: 1.33 dB behind gladmm; own training",
    "gladmmrwtv": "v31: graph-reweighted l1 in gladmm's layer (whitened step + magnetic TV); 51 parameters. SV UPA: 0.48 dB "
                  "behind gladmm; own end-to-end training",
    "nfladmm": "proposed, CNN denoiser; gets the true noise variance unless model.sigma=residual",
    "nfladmmc": "proposed, radial denoiser (160 parameters); true noise variance unless model.sigma=residual",
    "nfladmmsr": "proposed, CNN prior + over-relaxed update; true noise variance unless model.sigma=residual",
    "utrr": "Wu–Cheng–Eldar–Cioffi 2025; its own training: protocol 'paper'. Options: model.K=16, model.bn=true",
}


def default_methods(on):
    return [dict(key=k, label=lab, T=on.get(k, str(t)), on=k in on, set="", desc=METHOD_INFO.get(k, ""))
            for k, lab, t in suite_methods()]


def method_setting_groups(v):
    """The ticked methods grouped by their own settings (the 'settings' cell of a row): one form value set per group,
    with the row's settings added to the task's extra settings. Rows without own settings stay together."""
    rows = [r for r in (v.get("methods") or []) if r.get("on")]
    groups = {}
    for r in rows:
        own = lines(str(r.get("set") or "").replace(";", "\n"))
        for o in own:
            if "=" not in o or o.startswith("-"):
                raise ValueError(f"{r.get('key')}: setting '{o}': write key=value, e.g. model.threshold=group")
        groups.setdefault("\n".join(own), []).append(r)
    if not groups:
        return [v]
    out = []
    for own, rs in groups.items():
        out.append(dict(v, methods=rs, set="\n".join(x for x in (str(v.get("set") or "").strip(), own) if x),
                        _own=own))
    return out


# --- step 0: environment ------------------------------------------------------------------------------------------
def b_env_create(v):
    name = (v.get("env_name") or "bsce").strip()
    if not re.fullmatch(r"[A-Za-z0-9_.-]+", name):
        raise ValueError("environment name: letters, digits, _ . - only")
    envpy = f"{{envpy:{name}}}"
    c = []
    if v.get("mode") == "update":
        c.append({"argv": ["{conda}", "env", "update", "-f", "environment.yml", "-n", name, "--prune"], "kind": "conda",
                  "label": f"update the conda environment '{name}' from environment.yml"})
    else:
        c.append({"argv": ["{conda}", "env", "create", "-f", "environment.yml", "-n", name], "kind": "conda",
                  "label": f"create the conda environment '{name}' (Python 3.11 + numpy, scipy, pyyaml, matplotlib)"})
    tb = v.get("torch", "cu126")
    c.append({"argv": [envpy, "-m", "pip", "install", "torch", "--index-url", TORCH[tb]], "kind": "pip",
              "label": f"PyTorch ({tb} build)"})
    if v.get("verify"):
        c.append({"argv": [envpy, "-m", "pip", "install", "-r", "requirements-verify.txt"], "kind": "pip",
                  "label": "packages of the verification tests"})
    if v.get("spyder"):
        c.append({"argv": ["{conda}", "install", "-n", name, "-c", "conda-forge", "-y", "spyder"], "kind": "conda",
                  "label": "Spyder in the environment"})
    if v.get("check", True):
        c.append({"argv": [envpy, "-c", TORCH_CHECK], "kind": "python", "label": "PyTorch sees the GPU?"})
        c.append({"argv": [envpy, "scripts/check_env.py"], "kind": "script", "label": "environment check"})
    return c


def b_env_check(v):
    return [{"argv": ["{py}", "-c", TORCH_CHECK], "kind": "python", "label": "PyTorch sees the GPU?"},
            S("check_env.py") | {"label": "packages, GPU, reference/ and data/"}]


QUICK = ["tests/test_app.py", "tests/test_check_vs_authors.py", "tests/test_band_options.py", "tests/test_review_fixes.py", "tests/test_suite_options.py", "tests/test_save_layers.py",
         "tests/test_restore_best.py", "tests/test_group_threshold.py", "tests/test_proposed_ladmm.py",
         "tests/test_noise_model.py", "tests/test_resume.py", "tests/test_snr_dist.py", "tests/test_utrr.py",
         "tests/test_lvamp_noise.py", "tests/test_v28_options.py",
         "tests/test_v29_graph_combiner.py", "tests/test_v30_combiners_names.py", "tests/test_v31_rwl1.py",
         "tests/test_v31_release.py"]
TEST_SETS = {"quick": QUICK,
             "ref": QUICK + ["tests/test_nfladmm.py", "tests/test_nfladmm_relax.py", "tests/test_make_dataset.py",
                             "tests/test_make_dataset_3gpp.py", "tests/test_tr38901_builtin.py"],
             "all": ["tests"]}
TEST_LABEL = {"quick": "quick tests (no reference/ needed)", "ref": "quick tests + NF-LADMM and data tests (needs reference/)",
              "all": "every test (needs reference/ and the verification packages)"}


def b_tests(v):
    sel = v.get("which", "quick")
    return [{"argv": ["{py}", "-m", "pytest", "-q", *TEST_SETS[sel]], "kind": "python", "label": TEST_LABEL[sel]}]


# --- step 1: data -------------------------------------------------------------------------------------------------
def b_gmlamp_data(v):
    a = []
    if v.get("sv"):
        a.append("--sv")
    if v.get("ula"):
        a += ["--deepmimo-ula", v["ula"].strip()]
    if v.get("upa"):
        a += ["--deepmimo-upa", v["upa"].strip()]
    if not a:
        raise ValueError("tick SV and/or give a raw DeepMIMO file")
    if v.get("only"):
        a += ["--only", ",".join(v["only"])]
    if v.get("out_root", "").strip():
        a += ["--out-root", v["out_root"].strip()]
    return [S("make_gmlamp_data.py", *a) | {"label": "the GM-LAMP data sets (bit-identical to all earlier results)"}]


def _ml(p):
    return str(p).strip().replace("\\", "/").replace("'", "''")


def b_matlab(v):
    d, arr = v.get("deepmimo_dir", "").strip(), v.get("array", "ULA")
    if not d:
        raise ValueError("give the DeepMIMO-matlab-1.3 folder")
    mdir = _ml(ROOT / "matlab")
    if v.get("chunks"):
        out = v.get("out_dir", "").strip() or f"data/raw_{arr.lower()}"
        call = (f"deepmimo_make_H_chunks('{_ml(d)}','{_ml(ROOT / out) if not Path(out).is_absolute() else _ml(out)}',"
                f"'{arr}',{int(v.get('n_users') or 200000)},{int(v.get('chunk_rows') or 300)})")
    else:
        out = v.get("out_file", "").strip() or f"H_{arr.lower()}.mat"
        rows = " ".join(split_edges(v.get("rows") or "1000 1300"))
        call = (f"deepmimo_make_H('{_ml(d)}','{_ml(ROOT / out) if not Path(out).is_absolute() else _ml(out)}',"
                f"'{arr}',[{rows}])")
    return [{"argv": [v.get("matlab") or "matlab", "-batch", f"addpath('{mdir}'); {call}"], "kind": "matlab",
             "label": f"raw DeepMIMO {arr} channels with MATLAB"}]


def b_sionna(v):
    env = (v.get("env") or "sionna").strip()
    if not re.fullmatch(r"[A-Za-z0-9_.-]+", env):
        raise ValueError("environment name: letters, digits, _ . - only")
    envpy = f"{{envpy:{env}}}"
    arr = v.get("array", "UPA")
    out = v.get("out_prefix", "").strip() or f"data/raw3gpp/H_3gpp_{ {'UPA': 'upa', 'ULA_Y': 'ulay', 'ULA': 'ulaz'}[arr] }"
    c = []
    if v.get("install"):
        c.append({"argv": ["{conda}", "create", "-n", env, "python=3.12", "-y"], "kind": "conda",
                  "label": f"create the conda environment '{env}' for Sionna"})
        c.append({"argv": [envpy, "-m", "pip", "install", "sionna-no-rt==2.2.0", "scipy"], "kind": "pip",
                  "label": "Sionna 2.2 without the ray tracer (pulls PyTorch >= 2.9.1)"})
    a = ["--array", arr, "--n", str(int(v.get("n") or 100000)), "--out", out]
    for key, flag, dflt in (("scenario", "--scenario", "umi"), ("spec", "--spec", "16.1"), ("element", "--element", "38.901"),
                            ("device", "--device", "cpu")):
        if v.get(key, dflt) != dflt:
            a += [flag, v[key]]
    for key, flag, dflt in (("indoor", "--indoor", "0"), ("seed", "--seed", "2026"), ("threads", "--threads", "0")):
        val = str(v.get(key, dflt)).strip()
        if val not in ("", dflt):
            a += [flag, val]
    if str(v.get("parts", "")).strip():
        a += ["--parts", str(v["parts"]).strip()]
    c.append({"argv": [envpy, "scripts/sionna_make_H.py", *a], "kind": "script",
              "label": f"raw 3GPP 38.901 {arr} channels with Sionna (environment '{env}')"})
    return c


def b_quadriga(v):
    q, arr = v.get("quadriga_src", "").strip(), v.get("array", "UPA")
    if not q:
        raise ValueError("give the quadriga_src folder of QuaDRiGa v2.8.1")
    K = int(v.get("n") or 100000)
    out = v.get("out_prefix", "").strip() or f"data/raw3gpp/H_3gpp_{arr.lower().replace('_', '')}"
    out = _ml(out if Path(out).is_absolute() else ROOT / out)
    opts = []
    if v.get("scenario", "3GPP_38.901_UMi") != "3GPP_38.901_UMi":
        opts.append(f"'scenario','{v['scenario']}'")
    if v.get("element", "3gpp") != "3gpp":
        opts.append(f"'element','{v['element']}'")
    if str(v.get("indoor", "0")).strip() not in ("", "0"):
        opts.append(f"'indoor',{float(v['indoor']):g}")
    if str(v.get("seed", "2026")).strip() not in ("", "2026"):
        opts.append(f"'seed',{int(v['seed'])}")
    parts = str(v.get("parts", "")).strip()
    if parts:
        if not re.fullmatch(r"[\d:\s,\[\]]+", parts):
            raise ValueError("parts: MATLAB indices such as 1:10 or [1 3 5]")
        opts.append(f"'parts',[{parts.strip('[]')}]")
    if v.get("realized"):
        opts.append("'realized',true")
    call = f"quadriga_make_H('{_ml(q)}','{out}','{arr}',{K}" + "".join("," + o for o in opts) + ")"
    mdir = _ml(ROOT / "matlab")
    eng = v.get("engine", "matlab")
    exe = v.get("exe", "").strip() or eng
    argv = [exe, "-batch", f"addpath('{mdir}'); {call}"] if eng == "matlab" else \
        [exe, "--no-gui", "--quiet", "--eval", f"addpath('{mdir}'); {call}"]
    return [{"argv": argv, "kind": "matlab", "label": f"raw 3GPP 38.901 {arr} channels with {eng.title()} (QuaDRiGa)"}]


SIZE_PRESETS = [["sv", "80,000 / 2,000 / 2,000 (as SV)"], ["deepmimo", "50,000 / 2,000 / 2,000 (as DeepMIMO)"],
                ["dong2019", "81,000 / 9,000 / 19,000 (Dong et al., JSTSP 2019, 3GPP CDL 28 GHz)"],
                ["100k", "96,000 / 2,000 / 2,000"], ["custom", "custom sizes"]]


def b_3gpp_data(v):
    a = []
    pre = v.get("sizes", "sv")
    if pre == "custom":
        n = [str(v.get(k, "")).strip() for k in ("n_train", "n_val", "n_test")]
        if not all(n):
            raise ValueError("custom sizes: give train, val and test")
        a += ["--n-train", n[0], "--n-val", n[1], "--n-test", n[2]]
    elif pre != "sv":
        a += ["--sizes", pre]
    only = v.get("only") or ["3GPP_UPA", "3GPP_ULA"]
    if list(only) != ["3GPP_UPA", "3GPP_ULA"]:
        a += ["--only", ",".join(only)]
    for key, flag, dflt in (("spec", "--spec", "16.1"), ("channel", "--channel", "umi"), ("element", "--element", "38.901")):
        if v.get(key, dflt) != dflt:
            a += [flag, v[key]]
    if str(v.get("suffix", "")).strip():
        suf = v["suffix"].strip()
        if not re.fullmatch(r"[A-Za-z0-9_.-]+", suf):
            raise ValueError("suffix: letters, digits, _ . - only")
        a += ["--suffix", suf]
    if str(v.get("seed", "2026")).strip() not in ("", "2026"):
        a += ["--seed", str(v["seed"]).strip()]
    if v.get("check", True):
        a.append("--check")
    if v.get("force"):
        a.append("--force")
    return [S("make_3gpp_data.py", *a) | {"label": "the 3GPP TR 38.901 data sets (built-in generator, Python only)"}]


def b_make_dataset(v):
    a = ["--dataset", v.get("dataset", "SV"), "--array", v.get("array", "UPA")]
    sizes = [v.get(k, "").strip() for k in ("n_train", "n_val", "n_test")]
    if v.get("sizes", "custom") != "custom":
        a += ["--sizes", v["sizes"]]
    elif any(sizes):
        if not all(sizes):
            raise ValueError("give all three exact sizes, or none")
        a += ["--n-train", sizes[0], "--n-val", sizes[1], "--n-test", sizes[2]]
    else:
        a += ["--n", str(v.get("n") or 200000)]
        sp = v.get("split", "0.90 0.05 0.05").split()
        if sp != ["0.90", "0.05", "0.05"]:
            a += ["--split", *sp]
    if v.get("test", "standard") != "standard":
        a += ["--test", v["test"]]
    if v.get("no_meas"):
        a.append("--no-measurements")
    elif v.get("authors_bands"):
        a.append("--authors-bands")
    else:
        a += ["--snr", *split_edges(v.get("snr") or "0 20")]
    if v.get("test_meas"):
        a.append("--test-measurements")
    if v.get("noise_model", "colored") != "colored" and not v.get("no_meas"):
        a += ["--noise-model", v["noise_model"]]
    if v.get("dataset") in ("DeepMIMO", "3GPP"):
        inp = lines(v.get("input"))
        if not inp and v.get("dataset") == "DeepMIMO":
            raise ValueError("DeepMIMO needs the raw channel files (one per line, * allowed)")
        if inp:
            a += ["--input", *inp]
        elif v.get("dataset") == "3GPP" and v.get("array") == "ULA" and v.get("ula_axis", "y") != "y":
            a += ["--ula-axis", v["ula_axis"]]
    if v.get("name", "").strip():
        a += ["--name", v["name"].strip()]
    if v.get("out_dir", "").strip():
        a += ["--out-dir", v["out_dir"].strip()]
    if str(v.get("seed", "2026")).strip() not in ("", "2026"):
        a += ["--seed", str(v["seed"]).strip()]
    if v.get("force"):
        a.append("--force")
    return [S("make_dataset.py", *a) | {"label": "a data set of any size (+ its scenario yaml)"}]


def b_smoke_data(v):
    return [S("generate_data.py", "--scenario", f"configs/scenarios/{s}.yaml", "--smoke")
            | {"label": f"tiny data for the pipeline check: {s}"} for s in scen_values(v)]


# --- step 2: reproduce the paper ---------------------------------------------------------------------------------
def b_authors(v):
    a = ["--scenarios", *scen_values(v), "--tags", "GM-LAMP-authors", "LAMP-authors", "--name", v.get("name") or "authors",
         "--device", v.get("device", "cuda")]
    cl = v.get("classical") or []
    if cl != ["amp"]:
        a += ["--classical", ",".join(cl)]          # "" = no classical baseline
    return [S("compare.py", *a) | {"label": "the authors' published networks on our test sets"}]


def b_paper(v):
    m = [x for x in v.get("which") or []]
    if not m:
        raise ValueError("choose LAMP and/or GM-LAMP")
    a = ["--scenarios", *scen_values(v), "--methods", *m, "--protocol", "paper", "--train-snr", "scenario",
         "--ft-snr", "none", "--device", v.get("device", "cuda")]
    if v.get("float32"):
        a += ["--set", "model.precision=float32"]
    if v.get("smoke"):
        a.append("--smoke")
    c = [S("run_suite.py", *a) | {"label": "retrain with the authors' own procedure (two networks per scenario)", "suite": True}]
    suffix = "_paper_trscenario" + ("_precision-float32" if v.get("float32") else "")
    if v.get("then_compare"):
        c.append(S("compare.py", "--scenarios", *scen_values(v), "--tags", "*" + suffix, "--name", "paper_check",
                   "--device", v.get("device", "cuda"), *(["--smoke"] if v.get("smoke") else []))
                 | {"label": "compare them with the published networks"})
    if v.get("then_check"):
        lab = {"lamp": "LAMP", "gmlamp": "GM-LAMP"}
        c.append(S("check_vs_authors.py", "--no-data", "--runs", *[f"outputs/*/{lab[x]}-T*{suffix}" for x in m],
                   "--name", "paper_check_stages")
                 | {"label": "stage by stage against the authors' training logs"})
    return c


def b_check_authors(v):
    a = ["--scenarios", *scen_values(v)]
    runs = lines(v.get("runs"))
    if runs:
        a += ["--runs", *runs]
    if (v.get("name") or "authors_check").strip() != "authors_check":
        a += ["--name", v["name"].strip()]
    return [S("check_vs_authors.py", *a) | {"label": "data and retrained networks against the authors' files"}]


def b_authors_suite(v):
    """Every method trained exactly like GM-LAMP (Train_GM-LAMP.py / define_networks.do_training) on the authors' data."""
    return [c for vv in method_setting_groups(v) for c in _b_authors_suite(vv)]


def _b_authors_suite(v):
    a = ["--scenarios", *scen_values(v), "--methods", *method_tokens(v.get("methods")), "--protocol", "gmlamp",
         "--train-snr", "scenario", "--ft-snr", "none", *data_option_args(v, ft=False), "--device", v.get("device", "cuda")]
    if (v.get("out") or "outputs").strip() != "outputs":
        a += ["--out", v["out"].strip()]
    extra = (["model.precision=float64"] if v.get("precision") == "float64" else []) + lines(v.get("set", ""))
    if extra:
        a += set_args({"set": "\n".join(extra)})
    if v.get("smoke"):
        a.append("--smoke")
    return [S("run_suite.py", *a, *(["--seed", sd] if sd != "0" else []))
            | {"label": "every method with the authors' training protocol and data" + (f" (seed {sd})" if sd != "0" else ""),
               "suite": True} for sd in seeds_of(v)]


# --- step 3: depth -----------------------------------------------------------------------------------------------
def suite_args(v, depth=False):
    a = ["--scenarios", *scen_values(v), "--methods", *method_tokens(v.get("methods"))]
    if v.get("protocol", "ladmm_best") != "ladmm_best":
        a += ["--protocol", v["protocol"]]
    if v.get("force_protocol"):
        a.append("--force-protocol")
    if not depth and v.get("schedule", "layerwise") == "e2e":
        a += ["--schedule", "e2e"]
        if str(v.get("e2e_steps", "30000")).strip() not in ("", "30000"):
            a += ["--e2e-steps", str(int(v["e2e_steps"]))]
    gc = v.get("clip_mode", "default")
    if gc == "value":
        x = str(v.get("clip_value", "1.0")).strip()
        try:
            if float(x) <= 0:
                raise ValueError
        except ValueError:
            raise ValueError("clip value: a positive number, e.g. 1.0")
        a += ["--grad-clip", x]
    elif gc == "none":
        a += ["--grad-clip", "none"]
    tr = v.get("train_snr", "0 20").strip() or "0 20"
    tr_e = split_edges(tr, allow=("scenario",))
    if tr_e != ["0", "20"]:
        a += ["--train-snr", *tr_e]
    if depth:
        a += ["--ft-snr", "none", "--save-layers"]
    else:
        ft = (v.get("ft_snr", "0 10 20").strip() or "0 10 20")
        ft_e = split_edges(ft, allow=("none",))
        if ft_e != ["0", "10", "20"]:
            a += ["--ft-snr", *ft_e]
        if ft_e != ["none"] and str(v.get("ft_iters", "20000")).strip() not in ("", "20000"):
            a += ["--ft-iters", str(int(v["ft_iters"]))]
        fp = str(v.get("ft_patience", "4000")).strip()
        if ft_e != ["none"] and fp not in ("", "4000"):
            if not fp.isdigit():
                raise ValueError("fine-tune early stop: a whole number of steps (0 = always the whole budget)")
            a += ["--ft-patience", fp]
        ens = str(v.get("ensemble") or "").replace(",", " ").split()
        if ens:
            keys = {k for k, _, _ in suite_methods()}
            bad = [m for m in ens if re.sub(r"\d+$", "", m.lower()) not in keys]
            if bad or len(ens) < 2:
                raise ValueError("ensemble: at least two method keys with their depth, e.g. ladmm6 ladmm8"
                                 + (f" (unknown: {', '.join(bad)})" if bad else ""))
            a += ["--ensemble", *ens]
        if v.get("save_layers"):
            if v.get("schedule", "layerwise") == "e2e" or v.get("protocol") == "e2e":
                raise ValueError("'save the network after every layer' needs a layer-wise schedule; with end to end every "
                                 "layer is trained together and nothing would be saved per layer")
            a.append("--save-layers")
    a += data_option_args(v, ft=not depth and v.get("ft_snr", "0 10 20").strip().lower() != "none")
    a += ["--device", v.get("device", "cuda")]
    if (v.get("out") or "outputs").strip() != "outputs":
        a += ["--out", v["out"].strip()]
    a += set_args(v)
    if v.get("smoke"):
        a.append("--smoke")
    return a


def data_option_args(v, ft=True):
    """--noise-model / --val-snr / --margin / --ft-margin / --snr-chunks of run_suite.py from the form (nothing when
    left at default)."""
    a = []
    if v.get("noise_model", "colored") == "white":
        a += ["--noise-model", "white"]
    mode = v.get("val_mode", "default")
    if mode == "range":
        a += ["--val-snr", "range"]
    elif mode == "points":
        pts = str(v.get("val_points", "10")).replace(",", " ").split()
        try:
            [float(x) for x in pts]
        except ValueError:
            raise ValueError("validation SNRs: numbers in dB, e.g. 10 or 0 5 10")
        if not pts:
            raise ValueError("give the validation SNRs, e.g. 10")
        a += ["--val-snr", *pts]
    for key, flag in (("margin", "--margin"), ("ft_margin", "--ft-margin")):
        if key == "ft_margin" and not ft:
            continue
        x = str(v.get(key, "0")).strip() or "0"
        try:
            if float(x) < 0:
                raise ValueError
        except ValueError:
            raise ValueError("edge margin: a number of dB, 0 or more")
        if float(x):
            a += [flag, x]
    c = str(v.get("snr_chunks", "")).strip()
    if c:
        if not c.isdigit() or int(c) < 1:
            raise ValueError("SNR chunks: a whole number, e.g. 10 (the authors) or 100")
        a += ["--snr-chunks", c]
    a += snr_dist_args(v)
    if c and "--snr-dist" in a:
        raise ValueError("SNR chunks have no effect with the fair SNR spread (every sample has its own SNR): leave one empty")
    return a


def snr_dist_args(v):
    """--snr-dist fair [--snr-step S] from the form (v26), nothing when left at default."""
    if v.get("snr_dist", "chunks") != "fair":
        if str(v.get("snr_step", "")).strip():
            raise ValueError("the SNR grid step needs the fair SNR spread")
        return []
    st = str(v.get("snr_step", "")).strip()
    if st:
        try:
            if float(st) <= 0:
                raise ValueError
        except ValueError:
            raise ValueError("SNR grid step: a positive number of dB, e.g. 0.2 (empty = continuous)")
        return ["--snr-dist", "fair", "--snr-step", st]
    return ["--snr-dist", "fair"]


SNR_DIST_FIELDS = [
    F("snr_dist", "Training SNRs", "select", "chunks",
      "how the training noise covers each band. Chunks: the measurement file in SNR chunks, one random SNR each (the "
      "authors 10, ranges 100), as before. Fair: every sample its own SNR, spread evenly over the band in random order "
      "(also the online noise and the fine-tune minibatches); validation over a range at equally spaced SNRs. Suffix _fair", [
          ["chunks", "SNR chunks (as before)"], ["fair", "fair: every sample its own SNR, spread evenly"]]),
    F("snr_step", "SNR grid step (dB, empty = continuous)", "text", "",
      "fair only: the SNRs on the grid lo, lo+step, ..., hi, every value equally often, e.g. 0.2 -> 0, 0.2, ..., 20",
      placeholder="0.2", show_if=["snr_dist", "fair"]),
]


DATA_FIELDS = lambda ft=True, authors=False: [  # noqa: E731
    F("noise_model", "Measurement noise", "select", "colored",
      "colored = y = A(x + n), GM-LAMP's model; white = y = Ax + w with the same mean power. White runs go to "
      "outputs/<SCENARIO>_white/ under the same names; compare both in step 5 (noise: both)", [
          ["colored", "colored (noise folding), as GM-LAMP"], ["white", "white (after the combiner)"]]),
    F("val_mode", "Validation noise", "select", "default",
      "the authors validate both networks at 10 dB; ranges (e.g. 0 20) are validated over the whole range", [
          ["default", "as the bands define it (authors' bands: 10 dB; ranges: the whole range)"],
          ["range", "over the whole band range, SNR ~ U(band)"], ["points", "at fixed SNRs"]]),
    F("val_points", "Validation SNRs (dB)", "text", "10", "one or more, e.g. 10 or 0 5 10", show_if=["val_mode", "points"]),
    F("margin", "Edge margin of the training bands (dB)", "number", "0",
      "train each band on [lo − M, hi + M], use it on [lo, hi], so its edges are trained like its middle; 0 = as the authors"),
    *([F("ft_margin", "Edge margin of the fine-tune bands (dB)", "number", "0", "the same for the fine-tune",
         show_if=["ft_snr", "!none"])] if ft else []),
    F("snr_chunks", "SNR chunks per training file", "text", "",
      "empty = as now (the authors' files 10, ranges 100); more chunks cover the band, and its margins, evenly",
      placeholder="10 or 100"),
    *SNR_DIST_FIELDS,
]


def seeds_of(v):
    s = str(v.get("seeds", "0")).replace(",", " ").split() or ["0"]
    if not all(x.isdigit() for x in s):
        raise ValueError("seeds: whole numbers, e.g. 0 1 2")
    return s


def b_deep(v):
    return [c for vv in method_setting_groups(v) for c in _b_deep(vv)]


def _b_deep(v):
    if v.get("protocol", "ladmm_best") == "e2e":
        raise ValueError("deep runs save the network after every layer, which needs a layer-wise protocol: with e2e every "
                         "layer is trained together and nothing would be saved per layer. Choose ladmm_best, gmlamp or "
                         "fair; or train each depth in step 4 (e.g. ladmm 2 4 6 8 10 12).")
    a = suite_args(v, depth=True)
    return [S("run_suite.py", *a, *(["--seed", s] if s != "0" else []))
            | {"label": f"deep runs, network saved after every layer (seed {s})", "suite": True} for s in seeds_of(v)]


def b_depth_curves(v):
    runs = lines(v.get("runs"))
    if not runs:
        raise ValueError("give at least one run folder or pattern")
    a = ["--runs", *runs, "--name", v.get("name") or "depth", "--device", v.get("device", "cuda")]
    if v.get("band", "").strip():
        a += ["--band", v["band"].strip()]
    if str(v.get("tol", "0.1")).strip() not in ("", "0.1"):
        a += ["--tol", str(v["tol"]).strip()]
    sn = split_edges(v.get("snr") or "0 10 20")
    if sn != ["0", "10", "20"]:
        a += ["--snr", *sn]
    return [S("depth_snapshots.py", *a) | {"label": "depth curves on the test set"}]


def b_promote(v):
    runs = lines(v.get("runs"))
    Ts = str(v.get("T", "")).replace(",", " ").split()
    if not runs or not Ts:
        raise ValueError("give the deep run folder(s) and the depth(s)")
    return [S("promote_layer.py", "--run", *runs, "--T", *Ts, *(["--force"] if v.get("force") else []))
            | {"label": "use the chosen depth without retraining"}]


# --- step 4: train ------------------------------------------------------------------------------------------------
def b_suite(v):
    return [c for vv in method_setting_groups(v) for c in _b_suite(vv)]


def _b_suite(v):
    a = suite_args(v)
    seeds = seeds_of(v)
    split = v.get("split", "one")
    groups = [a]
    if split in ("method", "scenario"):
        i_s, i_m = a.index("--scenarios"), a.index("--methods")
        scs = a[i_s + 1:i_m]
        j = next((k for k in range(i_m + 1, len(a)) if a[k].startswith("--")), len(a))
        mts = a[i_m + 1:j]
        rest = a[j:]
        if split == "method":
            groups = [["--scenarios", *scs, "--methods", m, *rest] for m in mts]
        else:
            groups = [["--scenarios", s, "--methods", *mts, *rest] for s in scs]
    out = []
    for s in seeds:
        for g in groups:
            sc = g[g.index("--scenarios") + 1:g.index("--methods")]
            mt_end = next((k for k in range(g.index("--methods") + 1, len(g)) if g[k].startswith("--")), len(g))
            mt = g[g.index("--methods") + 1:mt_end]
            own = (v.get("_own") or "").replace("\n", " ")
            out.append(S("run_suite.py", *g, *(["--seed", s] if s != "0" else []))
                       | {"label": f"train + fine-tune {' '.join(mt)} on {' '.join(sc)}" + (f" (seed {s})" if s != "0" else "")
                          + (f" [{own}]" if own else ""), "suite": True})
    return out


def b_finetune(v):
    src = (v.get("src_manual") or "").strip() or v.get("src") or ""
    if not src:
        raise ValueError("choose the network to fine-tune")
    a = ["--from", src, "--snr", *split_edges(v.get("snr") or "0 10 20")]
    tag = (v.get("tag") or "").strip()
    if tag:                                          # empty: the automatic name <network>_ft_<edges>dB_<iterations>
        a += ["--tag", tag]
    if str(v.get("iters") or "20000").strip() != "20000":
        a += ["--iters", str(int(v["iters"]))]
    for k, flag, dflt in (("batch", "--batch", "400"), ("lr", "--lr", "5e-5"), ("lr2", "--lr2", "5e-6"),
                          ("lr2_frac", "--lr2-frac", "0.3333"), ("val_every", "--val-every", "100"), ("seed", "--seed", "0"),
                          ("patience", "--patience", "4000"), ("min_gain", "--min-gain", "0.01")):
        x = str(v.get(k, dflt)).strip()
        if x and x != dflt:
            a += [flag, x]
    if v.get("val", "range") != "range":
        a += ["--val", v["val"]]
    if str(v.get("clip", "")).strip():
        a += ["--grad-clip", str(v["clip"]).strip()]
    if v.get("bands", "").strip():
        a += ["--bands", v["bands"].strip()]
    if (v.get("scenario") or "").strip():
        a += ["--scenario", v["scenario"].strip()]
    if v.get("noise_model", "") in ("colored", "white"):
        a += ["--noise-model", v["noise_model"]]
    vs = str(v.get("val_snr", "")).replace(",", " ").split()
    if vs:
        try:
            [float(x) for x in vs]
        except ValueError:
            raise ValueError("validation SNRs: numbers in dB, e.g. 10 or 0 5 10")
        a += ["--val-snr", *vs]
    mg = str(v.get("margin", "0")).strip() or "0"
    try:
        if float(mg) < 0:
            raise ValueError
    except ValueError:
        raise ValueError("edge margin: a number of dB, 0 or more")
    if float(mg):
        a += ["--margin", mg]
    a += snr_dist_args(v)
    a += ["--device", v.get("device", "cuda")]
    return [S("finetune_bands.py", *a) | {"label": f"fine-tune {Path(src).name} per band" + (f" -> {tag}" if tag else "")}]


def b_runs(v):
    a = []
    if (v.get("outputs") or "outputs").strip() != "outputs":
        a += ["--outputs", v["outputs"].strip()]
    if (v.get("scenario") or "").strip():
        a += ["--scenario", *v["scenario"].split()]
    return [S("runs.py", *a) | {"label": "every run: done, resumable (where) or incomplete -> outputs/RUNS.md"}]


def b_rename(v):
    return [S("rename_runs.py", *(["--apply"] if v.get("apply") else []))
            | {"label": "fine-tunes named before v24 (-ft) -> <network>_ft_<edges>dB_<iterations>"
                        + ("" if v.get("apply") else " (list only)")}]


# --- step 5: compare ----------------------------------------------------------------------------------------------
TAG_PRESETS = {"ft": ["*_ft_*", "*-ft", "*-ft[0-9]*"], "onenet": ["*-T[0-9]*"], "authors": ["*_gmlamp_trscenario*"], "all": []}
# "one network 0-20 dB": every trained run (any suffix: _fair, _paper, _e2e30k, ...) except fine-tunes and other ranges
TAG_EXCLUDE = {"onenet": ["*_ft_*", "*-ft", "*-ft[0-9]*", "*_trscenario*", "*_tr[0-9m]*", "ENS-*"]}


def b_compare(v):
    if v.get("noise_model", "colored") == "both":
        return b_compare_noise(v)
    a = ["--scenarios", *scen_values(v)]
    preset = v.get("preset", "ft")
    tags = lines(v.get("tags")) if preset == "custom" else TAG_PRESETS[preset]
    if preset == "custom" and not tags:
        raise ValueError("custom selection: give run names or patterns, one per line")
    if preset in TAG_EXCLUDE:
        a += ["--exclude", *TAG_EXCLUDE[preset]]
    if tags:
        a += ["--tags", *tags]
    a += ["--name", (v.get("name") or "comparison").strip()]
    if (v.get("ref") or "GM-LAMP-authors").strip() != "GM-LAMP-authors":
        a += ["--ref", v["ref"].strip()]
    if v.get("no_authors"):
        a.append("--no-authors")
    cl = v.get("classical") or []
    if cl != ["amp"]:
        a += ["--classical", ",".join(cl)]          # "" = no classical baseline
    if (v.get("outputs") or "outputs").strip() != "outputs":
        a += ["--outputs", v["outputs"].strip()]
    if (v.get("results") or "results").strip() != "results":
        a += ["--results", v["results"].strip()]
    if v.get("noise_model", "colored") == "white":
        a += ["--noise-model", "white"]
    if v.get("polish", "none") in ("band", "snr"):
        a += ["--polish", v["polish"]]
    if v.get("no_cost"):
        a.append("--no-cost")
    a += ["--device", v.get("device", "cuda")]
    if v.get("smoke"):
        a.append("--smoke")
    return [S("compare.py", *a) | {"label": "evaluate + figures + paper tables"}]


def b_ensemble(v):
    """make_ensemble.py: the mean of the outputs of several trained networks (UTRR-E style), evaluated like any network."""
    runs = lines(v.get("runs"))
    if len(runs) < 2:
        raise ValueError("give at least two run folders, one per line (e.g. outputs/SV_UPA/LADMM-T6_ft_0-10-20dB_20k)")
    tag = (v.get("tag") or "").strip()
    if not re.fullmatch(r"[A-Za-z0-9_.+-]+", tag):
        raise ValueError("ensemble name: letters, digits and _ . + - only, e.g. LADMM-E-T6-T8_ft_0-10-20dB_20k")
    a = ["--runs", *runs, "--tag", tag]
    if (v.get("out") or "").strip():
        a += ["--out", v["out"].strip()]
    return [S("make_ensemble.py", *a) | {"label": f"ensemble {tag} of {len(runs)} networks"}]


def b_transfer(v):
    """evaluate.py on another combiner and/or another data set (v29): the networks of a runs folder, tested on the
    target scenario's test channels with the chosen combiner."""
    runs = (v.get("runs") or "").strip()
    if not runs:
        raise ValueError("choose the runs folder of the trained networks, e.g. outputs/SV_ULA")
    sc = (v.get("scenario") or "").strip()
    if not sc:
        raise ValueError("choose the target data set")
    a = ["--scenario", sc if sc.endswith(".yaml") else f"configs/scenarios/{sc}.yaml", "--runs", runs]
    comb = v.get("combiner", "scenario")
    if comb in ("random", "random_m"):
        seed = str(v.get("seed_m" if comb == "random_m" else "seed") or "11").strip()
        if not seed.isdigit():
            raise ValueError("combiner seed: a whole number")
        if comb == "random_m":                       # v30: another number of RF chains
            m = str(v.get("rf_chains") or "").strip()
            if not m.isdigit() or int(m) < 1:
                raise ValueError("number of RF chains M: a whole number, e.g. 64")
            a += ["--combiner", f"random:{seed}:{m}"]
            comb = f"random-M{m}"
        else:
            a += ["--combiner", f"random:{seed}"]
    elif comb == "file":
        f = (v.get("file") or "").strip()
        if not f:
            raise ValueError("give the .mat file with the combiner A")
        a += ["--combiner", f]
    if v.get("graph_phase", "target") == "target":
        a += ["--graph-phase", "target"]
    cl = v.get("classical") or []
    a += ["--classical", ",".join(cl)]
    out = (v.get("out") or "").strip() or f"results/transfer_{Path(runs).name}_on_{Path(sc).stem}_{comb}"
    a += ["--out", out, "--device", v.get("device", "cuda")]
    return [S("evaluate.py", *a) | {"label": f"networks of {runs} on {sc} ({comb} combiner)"}]


def b_combiner_check(v):
    """check_combiners.py (v30): every network of a runs folder on the standard test combiners."""
    runs = (v.get("runs") or "").strip()
    if not runs:
        raise ValueError("choose the runs folder of the trained networks, e.g. outputs/SV_ULA")
    sc = (v.get("scenario") or "").strip()
    if not sc:
        raise ValueError("choose the test data set")
    a = ["--scenario", sc if sc.endswith(".yaml") else f"configs/scenarios/{sc}.yaml", "--runs", runs]
    tags = lines(v.get("tags"))
    if tags:
        a += ["--tags", *tags]
    allc = ["own", "pm1", "gauss", "corr", "sparse", "dct", "M+", "M-", "M--"]
    comb = v.get("combiners") or []
    files = lines(v.get("files"))
    if not comb and not files:
        raise ValueError("tick at least one test combiner, or give a combiner file")
    if list(comb) != allc:
        a += ["--combiners", *comb]
    if files:
        a += ["--combiner-files", *files]
    if v.get("graph_phase", "own") == "target":
        a += ["--graph-phase", "target"]
    cl = v.get("classical") or []
    if cl != ["amp"]:
        a += ["--classical", ",".join(cl)]
    nt = str(v.get("n_test") or "").strip()
    if nt:
        if not nt.isdigit() or int(nt) < 1:
            raise ValueError("test channels: a whole number, or empty for all")
        a += ["--n-test", nt]
    if (v.get("out") or "").strip():
        a += ["--out", v["out"].strip()]
    a += ["--device", v.get("device", "cuda")]
    return [S("check_combiners.py", *a) | {"label": f"networks of {runs} on {len(comb) + len(files)} test combiners"}]


def b_tune_classical(v):
    """tune_classical.py (v31): lam (and eps) of the classical l1 ADMM and graph-reweighted l1 per test SNR, on
    validation channels, for the compare task's l1 / grwl1 baselines."""
    out = []
    for sc in scen_values(v):
        a = ["--scenario", sc if sc.endswith(".yaml") else f"configs/scenarios/{sc}.yaml"]
        nv = str(v.get("n_val") or "300").strip()
        if not nv.isdigit() or int(nv) < 5:
            raise ValueError("validation channels: a whole number of at least 5")
        if nv != "300":
            a += ["--n-val", nv]
        if v.get("only_l1"):
            a += ["--methods", "l1"]
        out.append(S("tune_classical.py", *a) | {"label": f"tune the classical l1 baselines on {sc}"})
    return out


def b_utrr_table(v):
    """utrr_reproduce.py: UTRR's Table II in the paper's own setting (noise after the combiner), SV ULA."""
    Ks = str(v.get("K", "16")).replace(",", " ").split()
    if not Ks or not all(k.isdigit() for k in Ks):
        raise ValueError("top-K: whole numbers, e.g. 16 or 0 16 64 256 (four models, then the ensemble UTRR-E)")
    try:
        snr = float(str(v.get("snr", "20")).strip())
    except ValueError:
        raise ValueError("SNR: a number of dB, e.g. 20 or 10")
    a = ["--snr", f"{snr:g}", "--K", *Ks]
    for k, flag, dflt in (("ntrain", "--ntrain", "80000"), ("epochs", "--epochs", "300"), ("patience", "--patience", "10"),
                          ("batch", "--batch", "128"), ("seed", "--seed", "0")):
        x = str(v.get(k, dflt)).strip()
        if x and x != dflt:
            if not x.isdigit():
                raise ValueError(f"{k}: a whole number")
            a += [flag, x]
    if v.get("bn"):
        a += ["--bn", "1"]
    a += ["--device", v.get("device", "cuda")]
    return [S("utrr_reproduce.py", *a) | {"label": f"UTRR Table II at {snr:g} dB, K = {' '.join(Ks)}"}]


def b_compare_noise(v):
    """compare_noise.py: every selected network on colored and on white test noise, trained on either."""
    a = ["--scenarios", *scen_values(v)]
    preset = v.get("preset", "ft")
    tags = lines(v.get("tags")) if preset == "custom" else TAG_PRESETS[preset]
    if preset == "custom" and not tags:
        raise ValueError("custom selection: give run names or patterns, one per line")
    if preset in TAG_EXCLUDE:
        a += ["--exclude", *TAG_EXCLUDE[preset]]
    if tags:
        a += ["--tags", *tags]
    name = (v.get("name") or "comparison").strip()
    a += ["--name", name if name != "final_finetuned" else "noise_comparison"]
    if v.get("nf_matched", True):
        a.append("--nf-matched")
    if v.get("no_authors"):
        a.append("--no-authors")
    cl = v.get("classical") or []
    if cl != ["amp"]:
        a += ["--classical", ",".join(cl)]
    if (v.get("outputs") or "outputs").strip() != "outputs":
        a += ["--outputs", v["outputs"].strip()]
    if (v.get("results") or "results").strip() != "results":
        a += ["--results", v["results"].strip()]
    a += ["--device", v.get("device", "cuda")]
    if v.get("smoke"):
        a.append("--smoke")
    return [S("compare_noise.py", *a) | {"label": "colored against white noise: every network on both, with cross tests"}]


def b_seeds(v):
    c = (v.get("csv_manual") or "").strip() or v.get("csv") or ""
    if not c:
        raise ValueError("choose an nmse_vs_snr.csv (made by Compare)")
    return [S("aggregate_seeds.py", "--csv", c) | {"label": "mean ± std over training seeds"}]


def b_script(v):
    sc = v.get("script") or ""
    if not re.fullmatch(r"[A-Za-z0-9_]+\.py", sc) or not (ROOT / "scripts" / sc).exists():
        raise ValueError("choose a script")
    try:
        args = [x[1:-1] if len(x) > 1 and x[0] == x[-1] and x[0] in "\"'" else x
                for x in shlex.split(v.get("args") or "", posix=False)]
    except ValueError as e:
        raise ValueError(f"arguments: {e}")
    return [S(sc, *args) | {"label": f"run scripts/{sc}"}]


SUITE_FIELDS = lambda deep=False: [  # noqa: E731
    scen_field(),
    methods_field({"ladmm": "6 4", "lamp": "8", "gmlamp": "8", "nfladmmsr": "4"} if not deep else
                  {"ladmm": "12", "lamp": "12", "gmlamp": "12"}),
    F("protocol", "Training protocol (the same for every method)", "select", "ladmm_best",
      "ladmm_best = the recipe of your best LADMM run (untitled2.py), used for all current results. The analytic "
      "Graph-LADMM rows (gladmm, gladmmany) keep their own end-to-end training unless the box below is ticked",
      wide=True, options=[
          ["ladmm_best", "ladmm_best: layer-wise, lr 5e-4, batch 400 (default)"],
          ["gmlamp_best", "gmlamp_best: GM-LAMP's procedure + best validated weights"],
          ["gmlamp", "gmlamp: GM-LAMP's procedure, last weights"],
          ["fair", "fair: layer-wise, lr 1e-3, batch 512, ≤ 50k per stage"],
          ["e2e", "e2e: all layers together, 30k steps"],
          ["paper", "paper: each method's own settings (not the same for all)"]]),
    F("force_protocol", "Same protocol for the analytic Graph-LADMM rows too", "check", False,
      "v30: off = gladmm / gladmmany train end to end with their own settings (folder …_paper), which measured "
      "1.7 dB better than layer-wise; on = they take the protocol above like every method"),
    *([] if deep else [
        F("schedule", "Schedule", "select", "layerwise", "end to end: all layers trained together from the start", [
            ["layerwise", "as the protocol says (layer by layer)"], ["e2e", "end to end, every method"]]),
        F("e2e_steps", "End-to-end steps", "number", "30000", "minibatches", show_if=["schedule", "e2e"])]),
    F("clip_mode", "Gradient clipping", "select", "default",
      "the same for every method, training and fine-tune; the common protocols do not clip", [
        ["default", "as the protocol says"], ["value", "every method, at a value"], ["none", "no method"]]),
    F("clip_value", "Clip at", "number", "1.0", show_if=["clip_mode", "value"]),
    F("train_snr", "Training SNR range (dB)", "text", "0 20",
      "edges: '0 20' = one network; '0 10 20' = two networks; 'scenario' = the authors' bands and files"),
    *([] if deep else [
        F("ft_snr", "Fine-tune bands (dB)", "text", "0 10 20", "edges, or 'none' for no fine-tune"),
        F("ft_iters", "Fine-tune iterations per band", "number", "20000", show_if=["ft_snr", "!none"]),
        F("ft_patience", "Fine-tune: stop a learning rate after … steps without a 0.01 dB gain", "number", "4000",
          "early stop of the fine-tune (then the band); 0 = always the whole budget", show_if=["ft_snr", "!none"]),
        F("save_layers", "Also save the network after every layer", "check", False, "for depth curves"),
        F("ensemble", "Ensemble after training (optional)", "text", "",
          "method keys, e.g. 'ladmm6 ladmm8': their final networks averaged into one (ENS-…); empty = no ensemble",
          placeholder="ladmm6 ladmm8"),
        *DATA_FIELDS(ft=True),
        F("split", "Commands", "select", "one", "split the work to see it step by step, or to run it on two PCs", [
            ["one", "one command for everything"], ["method", "one command per method"],
            ["scenario", "one command per scenario"]])]),
    *(DATA_FIELDS(ft=False) if deep else []),
    F("seeds", "Training seeds", "text", "0", "several: '0 1 2' (one run per seed; then Compare → mean ± std)"),
    device_field(),
    F("out", "Output folder", "text", "outputs"),
    F("set", "Extra settings for every ticked method (one per line)", "lines", "",
      "any config key, e.g. train.restore_best=true, train.loss=log_nse. A setting for one method only (e.g. "
      "model.sigma=residual for NF-LADMM): its 'settings' cell in the methods table", wide=True,
      placeholder="train.restore_best=true"),
    F("smoke", "Smoke test (tiny data and budget, checks the pipeline in minutes)", "check", False),
]

STEPS = [
    dict(id="env", n="0", title="Set up the environment",
         intro="Creates the conda environment, installs PyTorch for your GPU and checks the folder. Run once per PC.",
         tasks=[
             dict(id="env_create", title="Create the conda environment", build=b_env_create, fields=[
                 F("env_name", "Environment name", "text", "bsce"),
                 F("mode", "If it exists", "select", "create",
                   "create stops if the environment exists already; update keeps it and syncs its packages",
                   [["create", "create it (new PC)"],
                                                                    ["update", "update it from environment.yml"]]),
                 F("torch", "PyTorch build", "select", "cu126", "the Quadro P2000 (Pascal) needs cu126 or older", [
                     ["cu126", "CUDA 12.6 (recommended, P2000)"], ["cu124", "CUDA 12.4"], ["cu118", "CUDA 11.8"],
                     ["cu128", "CUDA 12.8 (new GPUs only)"], ["cpu", "CPU only"]]),
                 F("verify", "Also the packages of the verification tests", "check", False),
                 F("spyder", "Also install Spyder in the environment", "check", False),
                 F("check", "Check the result", "check", True)]),
             dict(id="env_check", title="Check this PC", build=b_env_check, fields=[],
                  note="Uses the Python shown at the top. Every line should say OK."),
             dict(id="tests", title="Run the tests", build=b_tests, fields=[
                 F("which", "Tests", "select", "quick", "", [["quick", "quick (a few minutes, no reference/ needed)"],
                                                            ["ref", "quick + NF-LADMM and data (needs reference/)"],
                                                            ["all", "all (needs reference/ and the verification packages; slow)"]])]),
         ]),
    dict(id="data", n="1", title="Make the data",
         intro="The four GM-LAMP data sets for the paper, or any other size. Test noise is never stored: every "
               "evaluation adds the same noise with fixed seeds.",
         tasks=[
             dict(id="gmlamp_data", title="GM-LAMP data sets (for the paper)", build=b_gmlamp_data,
                  note="Needs reference/ (GM-LAMP's matrices and test sets). Writes data/SV_ULA, data/SV_UPA and, with "
                       "the raw MATLAB files, data/DeepMIMO_ULA and data/DeepMIMO_UPA.", fields=[
                 F("sv", "SV ULA and SV UPA (Python, ~1 min)", "check", True),
                 F("ula", "Raw DeepMIMO ULA file", "path", "", "H_ula_1.mat from MATLAB (z axis)", wide=True,
                   placeholder="D:/Academic Profile/LADMM-MIMO/codes/H_ula_1.mat"),
                 F("upa", "Raw DeepMIMO UPA file", "path", "", "H_upa.mat from MATLAB", wide=True,
                   placeholder="D:/Academic Profile/LADMM-MIMO/codes/H_upa.mat"),
                 F("only", "Only these (empty = all given)", "multi", [], "",
                   [["SV_ULA", "SV_ULA"], ["SV_UPA", "SV_UPA"], ["DeepMIMO_ULA", "DeepMIMO_ULA"], ["DeepMIMO_UPA", "DeepMIMO_UPA"]]),
                 F("out_root", "Data folder (empty = data/)", "text", "")]),
             dict(id="matlab", title="Raw DeepMIMO channels (MATLAB)", build=b_matlab, fields=[
                 F("matlab", "MATLAB executable", "text", "matlab", "on PATH, or the full path of matlab.exe"),
                 F("deepmimo_dir", "DeepMIMO-matlab-1.3 folder", "path", "", wide=True,
                   placeholder="D:/DeepMIMO-matlab-1.3"),
                 F("array", "Array", "select", "ULA", "", [["ULA", "ULA (256 on z, GM-LAMP)"], ["UPA", "UPA (16 × 16 on y-z)"]]),
                 F("chunks", "Large data set (several files)", "check", False),
                 F("out_file", "Output file", "text", "", "relative to the bsce folder", show_if=["chunks", False],
                   placeholder="H_ula.mat"),
                 F("rows", "User rows", "text", "1000 1300", show_if=["chunks", False]),
                 F("out_dir", "Output folder", "text", "", show_if=["chunks", True], placeholder="data/raw_ula"),
                 F("n_users", "Users", "number", "200000", show_if=["chunks", True]),
                 F("chunk_rows", "Rows per file", "number", "300", show_if=["chunks", True])]),
             dict(id="data_3gpp", title="3GPP TR 38.901 data sets (built-in, Python only)", build=b_3gpp_data,
                  note="UMi street canyon at 28 GHz with the arrays, lens, normalisation, CS matrix and bands of the "
                       "DeepMIMO data; writes data/3GPP_UPA and data/3GPP_ULA in the format of data/SV_* and "
                       "data/DeepMIMO_* (docs/3GPP_DATASET.md). About 1.5 min per 100,000 channels. Train them as "
                       "scenarios 3gpp_upa and 3gpp_ula.", fields=[
                 F("sizes", "Train / validation / test", "select", "sv", "", SIZE_PRESETS),
                 F("n_train", "train", "number", "", show_if=["sizes", "custom"]),
                 F("n_val", "val", "number", "", show_if=["sizes", "custom"]),
                 F("n_test", "test", "number", "", show_if=["sizes", "custom"]),
                 F("only", "Data sets", "multi", ["3GPP_UPA", "3GPP_ULA"], "", [
                     ["3GPP_UPA", "3GPP_UPA (UPA 16 × 16, y-z)"], ["3GPP_ULA", "3GPP_ULA (ULA 256 on y)"],
                     ["3GPP_ULAZ", "3GPP_ULAZ (ULA 256 on z, as GM-LAMP's DeepMIMO ULA)"]]),
                 F("spec", "38.901 tables", "select", "16.1", "", [["16.1", "v16.1 (as QuaDRiGa; most published work)"],
                                                                  ["19.2", "v19.2 (the Rel-19 update)"]]),
                 F("channel", "Scenario", "select", "umi", "", [["umi", "UMi, LOS/NLOS by the 38.901 LOS probability"],
                                                                ["umi_los", "UMi, LOS only"], ["umi_nlos", "UMi, NLOS only"]]),
                 F("element", "BS element", "select", "38.901", "", [["38.901", "38.901 Table 7.3-1 (65°, 8 dBi)"],
                                                                     ["omni", "isotropic"]]),
                 F("suffix", "Name suffix (a second copy)", "text", "", "e.g. _v19 -> data/3GPP_UPA_v19, 3gpp_upa_v19.yaml"),
                 F("check", "Then check against 38.901, SV and DeepMIMO", "check", True),
                 F("seed", "Seed", "number", "2026"),
                 F("force", "Overwrite existing data sets", "check", False)]),
             dict(id="sionna", title="Raw 3GPP TR 38.901 channels (Sionna, the TR procedure as written)", build=b_sionna,
                  note="A second implementation of the same setting (the built-in generator was checked against it, "
                       "docs/3GPP_DATASET.md). Runs in its own conda environment (Sionna needs PyTorch >= 2.9.1; the "
                       "CPU build is enough). About 10-35 ms per channel. Then 'Data set of any size' with the raw "
                       "files.", fields=[
                 F("install", "First create the environment and install Sionna", "check", False),
                 F("env", "Environment", "text", "sionna"),
                 F("array", "Array", "select", "UPA", "", [["UPA", "UPA 16 × 16 on y-z (as DeepMIMO UPA)"],
                                                           ["ULA_Y", "ULA 256 on y, horizontal (3GPP usual, SV-like)"],
                                                           ["ULA", "ULA 256 on z, vertical (as GM-LAMP's DeepMIMO ULA)"]]),
                 F("n", "Channels", "number", "100000"),
                 F("out_prefix", "Output prefix (empty = data/raw3gpp/H_3gpp_<array>)", "text", ""),
                 F("scenario", "Scenario", "select", "umi", "", [["umi", "UMi, LOS/NLOS by the 38.901 LOS probability"],
                                                                 ["umi_los", "UMi, LOS only"], ["umi_nlos", "UMi, NLOS only"]]),
                 F("spec", "38.901 tables", "select", "16.1", "", [["16.1", "v16.1 (as QuaDRiGa; most published work)"],
                                                                  ["19.2", "v19.2 (the Rel-19 update)"]]),
                 F("element", "BS element", "select", "38.901", "", [["38.901", "38.901 Table 7.3-1 (65°, 8 dBi)"],
                                                                     ["omni", "isotropic"]]),
                 F("indoor", "Indoor (O2I) fraction", "text", "0"),
                 F("parts", "Only these parts (empty = all)", "text", "", "e.g. 1-10"),
                 F("device", "Device", "select", "cpu", "", [["cpu", "CPU"], ["cuda", "GPU (CUDA)"]]),
                 F("threads", "CPU threads (0 = PyTorch default)", "number", "0"),
                 F("seed", "Seed", "number", "2026")]),
             dict(id="quadriga", title="Raw 3GPP TR 38.901 channels (QuaDRiGa; MATLAB or Octave)", build=b_quadriga,
                  note="The MATLAB alternative to Sionna, same setting and file format. With its three 38.901 "
                       "fixes (on by default) the elevation follows the TR; the azimuth spreads stay QuaDRiGa's own "
                       "(wider, docs/3GPP_DATASET.md 4.2). About 70 ms per channel in Octave: 100,000 channels take "
                       "about 2 h. Parts already made are kept, so a stopped run resumes.", fields=[
                 F("engine", "Run with", "select", "matlab", "", [["matlab", "MATLAB"], ["octave", "Octave 8"]]),
                 F("exe", "Executable (empty = matlab / octave on PATH)", "text", ""),
                 F("quadriga_src", "QuaDRiGa quadriga_src folder", "path", "", "github.com/fraunhoferhhi/QuaDRiGa, v2.8.1",
                   wide=True, placeholder="D:/QuaDRiGa/quadriga_src"),
                 F("array", "Array", "select", "UPA", "", [["UPA", "UPA 16 × 16 on y-z (as DeepMIMO UPA)"],
                                                           ["ULA_Y", "ULA 256 on y, horizontal (3GPP usual, SV-like)"],
                                                           ["ULA", "ULA 256 on z, vertical (as GM-LAMP's DeepMIMO ULA)"]]),
                 F("n", "Channels", "number", "100000"),
                 F("out_prefix", "Output prefix (empty = data/raw3gpp/H_3gpp_<array>)", "text", ""),
                 F("scenario", "Scenario", "select", "3GPP_38.901_UMi", "", [
                     ["3GPP_38.901_UMi", "UMi, LOS/NLOS by the 38.901 LOS probability"],
                     ["3GPP_38.901_UMi_LOS", "UMi, LOS only"], ["3GPP_38.901_UMi_NLOS", "UMi, NLOS only"]]),
                 F("element", "BS element", "select", "3gpp", "", [["3gpp", "38.901 Table 7.3-1 (65°, 8 dBi)"],
                                                                   ["omni", "isotropic"]]),
                 F("indoor", "Indoor (O2I) fraction", "text", "0"),
                 F("parts", "Only these parts (empty = all)", "text", "", "e.g. 1:10, to share the work between two PCs"),
                 F("realized", "Also the spreads of the generated paths (slower; for the check)", "check", False),
                 F("seed", "Seed", "number", "2026")]),
             dict(id="make_dataset", title="Data set of any size", build=b_make_dataset, fields=[
                 F("dataset", "Channel model", "select", "SV", "", [["SV", "SV (Saleh-Valenzuela)"], ["DeepMIMO", "DeepMIMO (ray tracing)"],
                                                                    ["3GPP", "3GPP TR 38.901 (QuaDRiGa)"]]),
                 F("array", "Array", "select", "UPA", "", [["ULA", "ULA 256"], ["UPA", "UPA 16 × 16"]]),
                 F("input", "Raw DeepMIMO / 3GPP files (one per line, * allowed)", "lines", "", show_if=["dataset", "!SV"],
                   wide=True, placeholder="data/raw3gpp/H_3gpp_upa_part*.mat"),
                 F("n", "Channels in total", "number", "200000"),
                 F("split", "Train / val / test fractions", "text", "0.90 0.05 0.05"),
                 F("n_train", "or exact sizes: train", "number", ""), F("n_val", "val", "number", ""),
                 F("n_test", "test", "number", ""),
                 F("sizes", "Size preset", "select", "custom", "or the total / fractions / exact sizes below",
                   [["custom", "none (the fields below)"]] + SIZE_PRESETS[:-1]),
                 F("ula_axis", "3GPP ULA axis (built-in generator)", "select", "y", "", [["y", "y (horizontal)"],
                                                                                       ["z", "z (vertical)"]],
                   show_if=["dataset", "3GPP"]),
                 F("test", "Test set", "select", "standard", "standard = the 2000 channels of every paper result "
                   "(3GPP: held-out drops)", [
                     ["standard", "standard (comparable with the paper tables)"], ["random", "random split"]]),
                 F("authors_bands", "Bands of the authors: 0-10 / 10-20 dB, 10 SNR chunks, validation at 10 dB", "check",
                   False, "as sv_*.yaml and deepmimo_*.yaml (GM-LAMP's protocol)"),
                 F("snr", "Measurement files for the SNR bands (dB)", "text", "0 20", show_if=["authors_bands", False]),
                 F("no_meas", "Channels only (no measurement files)", "check", False),
                 F("noise_model", "Noise of the measurement files", "select", "colored",
                   "colored y = A(x + n) as GM-LAMP; white y = Ax + w, same mean power; both = paired sets (the white "
                   "files are the colored ones with the same noise draw whitened)",
                   [["colored", "colored (folded), as GM-LAMP"], ["white", "white"], ["both", "both, paired"]],
                   show_if=["no_meas", False]),
                 F("test_meas", "Also test measurement files (for MATLAB)", "check", False),
                 F("name", "Name", "text", "", "data/<name> and configs/scenarios/<name>.yaml", placeholder="SV_UPA_200k"),
                 F("out_dir", "Folder (empty = data/<name>)", "text", ""),
                 F("seed", "Seed", "number", "2026"),
                 F("force", "Overwrite an existing data set", "check", False)]),
             dict(id="smoke_data", title="Tiny data for a pipeline check", build=b_smoke_data, fields=[
                 scen_field(("sv_upa",), help="writes data/<SCENARIO>_smoke (needs reference/)")]),
         ]),
    dict(id="paper", n="2", title="Reproduce the paper",
         intro="First check that the test pipeline gives the published numbers; optionally retrain LAMP and GM-LAMP "
               "with the authors' procedure.",
         tasks=[
             dict(id="authors", title="The authors' networks on our test sets", build=b_authors, fields=[
                 scen_field(), F("name", "Results name", "text", "authors"),
                 F("classical", "Classical baselines", "multi", ["amp"], "", [["amp", "AMP"], ["omp", "OMP"], ["sd", "SD"]]),
                 device_field()],
                  note="Expected at 0 / 10 / 20 dB (GM-LAMP): SV ULA −9.17 / −16.78 / −22.96, SV UPA −8.45 / −15.11 / −20.32, "
                       "DeepMIMO ULA −13.10 / −21.57 / −29.49, DeepMIMO UPA −13.15 / −21.27 / −28.79."),
             dict(id="paper_retrain", title="Retrain LAMP and GM-LAMP with the paper's procedure", build=b_paper, fields=[
                 scen_field(), F("which", "Methods", "multi", ["lamp", "gmlamp"], "", [["lamp", "LAMP"], ["gmlamp", "GM-LAMP"]]),
                 F("float32", "float32 (faster on the P2000; the paper used float64)", "check", False),
                 F("then_compare", "Then compare with the published networks", "check", True),
                 F("then_check", "Then compare every training stage with the authors' training logs", "check", True),
                 device_field(), F("smoke", "Smoke test", "check", False)],
                  note="About a day on the P2000. Expected: within a few tenths of a dB of the authors' networks."),
             dict(id="utrr_table", title="UTRR: reproduce its Table II (its own setting)", build=b_utrr_table, fields=[
                 F("snr", "SNR (dB)", "number", "20", "the paper's Table II: 10 and 20 dB"),
                 F("K", "Top-K of the last layer (several: one model each, then the ensemble UTRR-E)", "text", "16",
                   placeholder="0 16 64 256"),
                 F("bn", "Batch normalisation (the paper's Fig. 6)", "check", False),
                 F("ntrain", "Training channels", "number", "80000", "fewer for a shorter check, e.g. 20000"),
                 F("epochs", "At most … epochs", "number", "300"), F("patience", "Early stop after … epochs", "number", "10"),
                 F("batch", "Batch (real samples)", "number", "128"), F("seed", "Seed", "number", "0"), device_field()],
                  note="UTRR (Wu, Cheng, Eldar, Cioffi, TCOM 2025) in the paper's own setting: noise after the combiner, "
                       "SV ULA, one model per SNR and top-K, Adam with five learning rates and early stopping. Expected "
                       "(Table II): 20 dB K = 16 −26.93 dB, UTRR-E −30.18 dB; 10 dB −18.92 / −22.06 dB. Resumes when run "
                       "again. Results: results/utrr_reproduce/ (docs/UTRR.md)."),
             dict(id="check_authors", title="Check the data against the authors' files", build=b_check_authors, fields=[
                 scen_field(),
                 F("runs", "Also these retrained LAMP / GM-LAMP runs, stage by stage (one per line, * allowed)", "lines", "",
                   wide=True, placeholder="outputs/*/GM-LAMP-T8_paper_trscenario"),
                 F("name", "Results name", "text", "authors_check")],
                  note="Uses the authors' training states (reference/gmlamp/Train/Trained_Networks/*.npz: the published "
                       "weights and the log of every training stage). Checks the measurement files (y = A(x + n), the "
                       "authors' A, 10 SNR chunks per band, validation at 10 dB) and runs the 16 published networks on our "
                       "validation sets against the validation NMSE the authors logged."),
         ]),
    dict(id="depth", n="3", title="Choose the number of layers",
         intro="One deep run per method saves the network after every layer: depth 1 … T from a single run, "
               "bit-identical to training each depth.",
         tasks=[
             dict(id="deep", title="Deep runs (network saved after every layer)", build=b_deep, fields=SUITE_FIELDS(deep=True)),
             dict(id="depth_curves", title="Depth curves on the test set", build=b_depth_curves, fields=[
                 F("runs", "Run folders (one per line, * allowed)", "lines",
                   "outputs/*/LADMM-T12\noutputs/*/LAMP-T12\noutputs/*/GM-LAMP-T12", wide=True),
                 F("band", "Band (empty = the only one)", "text", ""), F("tol", "Flat point tolerance (dB)", "number", "0.1"),
                 F("snr", "SNRs drawn as curves", "text", "0 10 20"), F("name", "Results name", "text", "depth"),
                 device_field()]),
             dict(id="promote", title="Use the chosen depth", build=b_promote, fields=[
                 F("runs", "Deep run folders (one per line, * allowed)", "lines", "outputs/*/LADMM-T12", wide=True),
                 F("T", "Depth(s)", "text", "6 4", "e.g. '6 4' makes LADMM-T6 and LADMM-T4"),
                 F("force", "Replace existing folders", "check", False)],
                  note="Then step 4 finds LADMM-T6 trained and goes straight to the fine-tune."),
         ]),
    dict(id="train", n="4", title="Train every method",
         intro="One common protocol for every method: one network for 0–20 dB, then fine-tuned per band. Finished "
               "steps are skipped and an interrupted training resumes, so a run can always be started again.",
         tasks=[
             dict(id="suite", title="Train and fine-tune", build=b_suite, fields=SUITE_FIELDS(), dryrun=True),
             dict(id="authors_suite", title="Every method with the authors' protocol", build=b_authors_suite, dryrun=True,
                  fields=[
                      scen_field(),
                      methods_field({"ladmm": "6 4", "lamp": "8", "gmlamp": "8", "nfladmmsr": "4"}),
                      F("precision", "Precision (the same for every method)", "select", "float32",
                        "the authors' TensorFlow code used float64; float64 is about 30 times slower on the P2000", [
                            ["float32", "float32 (the gmlamp protocol)"], ["float64", "float64 (exactly as the authors)"]]),
                      *DATA_FIELDS(ft=False),
                      F("seeds", "Training seeds", "text", "0"), device_field(), F("out", "Output folder", "text", "outputs"),
                      F("set", "Extra settings (one per line)", "lines", "", wide=True, placeholder="model.sigma=residual"),
                      F("smoke", "Smoke test", "check", False)],
                  note="Exactly the training of Train_GM-LAMP.py for every method: two networks per data set (0-10 and "
                       "10-20 dB) from the authors' measurement files (10 SNR chunks), validated at 10 dB; layer by layer, "
                       "Adam 1e-3 then all layers at 1e-3 x {0.5, 0.1, 0.01}, batch 128, new layer initialised from the "
                       "previous one, stop 5000 iterations after the best, last weights. No fine-tune. Folders "
                       "<method>-T<T>_gmlamp_trscenario; compare them with step 5, Networks: trained with the authors' protocol."),
             dict(id="finetune", title="Fine-tune one network (other budget or bands)", build=b_finetune, fields=[
                 F("src", "Network", "network", "", "a trained 0–20 dB network, or a depth snapshot (layers/T06_…): its "
                   "fine-tune is named <run>_T06_ft_…", wide=True),
                 F("src_manual", "or a path", "path", "", wide=True, placeholder="outputs/SV_UPA/LADMM-T6/LADMM-T6_0to20dB.pt"),
                 F("tag", "Name of the fine-tuned folder (empty = automatic)", "text", "",
                   "automatic: <network>_ft_<band edges>dB_<iterations>, next to the network, e.g. LADMM-T6_ft_0-10-20dB_20k; "
                   "the same command again resumes an interrupted fine-tune", placeholder="automatic"),
                 F("snr", "Bands (dB)", "text", "0 10 20"), F("iters", "Iterations per band", "number", "20000"),
                 F("batch", "Batch", "number", "400"), F("lr", "Learning rate", "text", "5e-5"),
                 F("lr2", "Second learning rate", "text", "5e-6"), F("lr2_frac", "Fraction at the second rate", "text", "0.3333"),
                 F("val", "Validation", "select", "range", "", [["range", "fixed-noise channels over the band"],
                                                                ["file", "the band's validation file"]]),
                 F("val_every", "Validate every", "number", "100"), F("clip", "Gradient clip (empty = none)", "text", ""),
                 F("patience", "Stop a learning rate after … steps without gain", "number", "4000",
                   "early stop: no gain of the minimum below during this many steps ends the first learning rate, then the "
                   "band; 0 = always every iteration"),
                 F("min_gain", "Minimum gain (dB)", "text", "0.01"),
                 *SNR_DIST_FIELDS,
                 F("scenario", "Data set (empty = the one the network was trained on)", "select", "",
                   "needed only for a network without its train_log.json, e.g. an untitled2.py checkpoint",
                   [["", "the network's own (from its train_log.json)"]] + [[s, s] for s in scenario_files()]),
                 F("noise_model", "Measurement noise", "select", "", "", [["", "as the network was trained"],
                                                                         ["colored", "colored"], ["white", "white"]]),
                 F("val_snr", "Validation at fixed SNRs (dB; empty = over the band)", "text", "", placeholder="10"),
                 F("margin", "Edge margin (dB)", "number", "0", "train each band on [lo − M, hi + M], use it on [lo, hi]"),
                 F("bands", "Only these bands (empty = all)", "text", "", placeholder="0to10dB"),
                 F("seed", "Seed", "number", "0"), device_field()]),
             dict(id="runs", title="What is done, what resumes", build=b_runs, fields=[
                 F("outputs", "Networks folder", "text", "outputs"),
                 F("scenario", "Only these scenario folders (empty = all)", "text", "", placeholder="SV_UPA DeepMIMO_UPA")],
                  note="Every run folder: trained or fine-tuned (from which network, how many iterations), the band "
                       "networks present, the best validation, and whether it is done, resumable (the stage and "
                       "iteration, or the band and step, where the same command continues) or incomplete. Writes "
                       "outputs/RUNS.md and RUNS.csv."),
             dict(id="rename", title="Old fine-tune names to the new ones", build=b_rename, fields=[
                 F("apply", "Rename (otherwise only list what would change)", "check", False)],
                  note="Fine-tunes made before v24 are named <network>-ft (or -ft0-5-10-15-20). They become "
                       "<network>_ft_<edges>dB_<iterations>, with the iterations from their log; the files inside are "
                       "renamed too. Nothing is retrained."),
         ]),
    dict(id="compare", n="5", title="Compare and plot",
         intro="Every selected network on the same 2000 test channels with the same noise, plus the authors' "
               "GM-LAMP and LAMP. Writes figures and paper tables to results/<name>/.",
         tasks=[
             dict(id="compare", title="Compare methods", build=b_compare, fields=[
                 scen_field(),
                 F("preset", "Networks", "select", "ft", "", [["ft", "fine-tuned (*_ft_*, and the old *-ft)"],
                                                              ["authors", "trained with the authors' protocol (*_gmlamp_trscenario)"],
                                                              ["onenet", "one network 0–20 dB (every trained run, any suffix; no fine-tunes)"],
                                                              ["all", "every folder"], ["custom", "custom list"]]),
                 F("tags", "Names or patterns (one per line)", "lines", "", show_if=["preset", "custom"], wide=True,
                   placeholder="LADMM-T6_ft_*\nNF-LADMM-SR-T4*_ft_*"),
                 F("name", "Results name", "text", "final_finetuned"),
                 F("ref", "Reference for the gains", "text", "GM-LAMP-authors", "a run folder name"),
                 F("no_authors", "Leave out the authors' networks", "check", False),
                 F("noise_model", "Noise", "select", "colored",
                   "colored: outputs/<SCENARIO>/; white: the networks trained with white noise (outputs/<SCENARIO>_white/); "
                   "both: every network on both test noises, trained on either (cross tests, compare_noise.py)", [
                       ["colored", "colored (as GM-LAMP)"], ["white", "white"], ["both", "both, with cross tests"]]),
                 F("nf_matched", "NF-LADMM also with the data step of the test noise (same weights)", "check", True,
                   show_if=["noise_model", "both"]),
                 F("classical", "Classical baselines", "multi", ["amp"], "l1 / graph-reweighted l1: tuned per SNR first (task 'Tune the classical l1 baselines')", [["amp", "AMP"], ["omp", "OMP"], ["sd", "SD"], ["l1", "l1 ADMM, tuned (v31)"], ["grwl1", "graph-reweighted l1, tuned (v31)"]]),
                 F("polish", "Polish (least squares on the strongest beams, no retraining)", "select", "none",
                   "adds a column '<run>+polish' for every network; the number of beams and the regularisation are "
                   "chosen on the validation channels", [
                       ["none", "off"], ["band", "on, chosen per network band"],
                       ["snr", "on, chosen per test SNR (assumes the SNR is known)"]], show_if=["noise_model", "!both"]),
                 F("no_cost", "Leave out the trained-parameter and MACs-per-channel columns (v31)", "check", False,
                   "on: faster; the columns count the parameters that reach the output and the real multiply-"
                   "accumulates of one channel"),
                 F("outputs", "Networks folder", "text", "outputs"), F("results", "Results folder", "text", "results"),
                 device_field(), F("smoke", "Smoke outputs (outputs/<SCENARIO>_smoke)", "check", False)]),
             dict(id="tune_classical", title="Tune the classical l1 baselines (v31)", build=b_tune_classical, fields=[
                 scen_field(),
                 F("n_val", "Validation channels", "text", "300", "the package's setting; about 2-3 min per SNR on one "
                   "CPU core"),
                 F("only_l1", "Only the plain l1 (not the graph-reweighted l1)", "check", False)],
                  note="scripts/tune_classical.py: writes results/classical_params_<SCENARIO>.json, which the compare task "
                       "uses for its 'l1' and 'graph-reweighted l1' baselines (lam and eps chosen per test SNR with the "
                       "true SNR, as the uploaded package does)."),
             dict(id="ensemble", title="Ensemble of trained networks", build=b_ensemble, fields=[
                 F("runs", "Run folders (one per line, at least two)", "lines", "", wide=True,
                   placeholder="outputs/SV_UPA/LADMM-T6_ft_0-10-20dB_20k\noutputs/SV_UPA/LADMM-T8_ft_0-10-20dB_20k"),
                 F("tag", "Name of the ensemble", "text", "", "its folder, next to the first run",
                   placeholder="LADMM-E-T6-T8_ft_0-10-20dB_20k"),
                 F("out", "Parent folder (empty = that of the first run)", "text", "")],
                  note="The mean of the members' outputs (UTRR-E style). It is made for every band all the members have, "
                       "and then compared like any other network (Compare methods: custom list, or every folder). It "
                       "helps when the members are within about 1.5 dB of each other."),
             dict(id="transfer", title="Another combiner or data set", build=b_transfer, fields=[
                 F("runs", "Networks (a runs folder)", "text", "outputs/SV_ULA", "e.g. the networks trained on SV ULA"),
                 F("scenario", "Test data set", "text", "deepmimo_ula", "a scenario name (its test channels and its "
                   "combiner): DeepMIMO's combiner is not SV's"),
                 F("combiner", "Combiner", "select", "scenario", "", [
                     ["scenario", "the test data set's"], ["random", "a new random one"],
                     ["random_m", "a new random one with another number of RF chains (v30)"],
                     ["file", "from a .mat file"]]),
                 F("seed", "Seed of the random combiner", "number", "11", show_if=["combiner", "random"]),
                 F("seed_m", "Seed of the random combiner", "number", "11", show_if=["combiner", "random_m"]),
                 F("rf_chains", "Number of RF chains M", "number", "64", "the scenario's is 128; networks learned for "
                   "one combiner cannot run with another M and are skipped", show_if=["combiner", "random_m"]),
                 F("file", ".mat file with A", "text", "", show_if=["combiner", "file"]),
                 F("graph_phase", "Graph-LADMM edge phases", "select", "target", "the array convention of the test data (analytic Graph-LADMM; free H keeps its own)", [
                     ["target", "of the test data set"], ["own", "as trained"]]),
                 F("classical", "Classical baselines", "multi", ["amp"], "", [["amp", "AMP"], ["omp", "OMP"], ["sd", "SD"]]),
                 F("out", "Results folder (empty = results/transfer_<runs>_on_<data>_<combiner>)", "text", ""),
                 device_field()],
                  note="Networks that can follow a new combiner (Graph-LADMM analytic, LADMM operator=analytic) use it, "
                       "any number of RF chains; the others keep the combiner they were learned for, which is the "
                       "transfer test (skipped when M differs). eval_log.json says which."),
             dict(id="combiner_check", title="Many test combiners at once (v30)", build=b_combiner_check, fields=[
                 F("runs", "Networks (a runs folder)", "text", "outputs/SV_ULA", "e.g. the networks trained on SV ULA"),
                 F("scenario", "Test data set", "text", "sv_ula", "a scenario name: its test channels; the test "
                   "combiners are built from its combiner (its N and M)"),
                 F("tags", "Networks (names or patterns, one per line; empty = every folder)", "lines", "", wide=True,
                   placeholder="G-LADMM*"),
                 F("combiners", "Test combiners", "multi", ["own", "pm1", "gauss", "corr", "sparse", "dct", "M+", "M-", "M--"],
                   "all built the same way on every PC (fixed seeds)", [
                       ["own", "the scenario's (training)"], ["pm1", "random ±1, new draw"], ["gauss", "Gaussian"],
                       ["corr", "correlated rows (badly conditioned)"], ["sparse", "partially connected"],
                       ["dct", "DCT row subset (structured)"], ["M+", "M = 1.25 M0"], ["M-", "M = 0.75 M0"],
                       ["M--", "M = 0.5 M0"]]),
                 F("files", "Your own combiners (.mat with A, one path per line)", "lines", "", wide=True),
                 F("graph_phase", "Graph-LADMM edge phases", "select", "own", "target: when the networks come from "
                   "another data set", [["own", "as trained"], ["target", "of the test data set"]]),
                 F("classical", "Classical baselines", "multi", ["amp"], "", [["amp", "AMP"], ["omp", "OMP"], ["sd", "SD"]]),
                 F("n_test", "Test channels (empty = all)", "text", "", "fewer for a quick look, e.g. 500"),
                 F("out", "Results folder (empty = results/combiner_check_<data>_<runs>)", "text", ""),
                 device_field()],
                  note="scripts/check_combiners.py: one table and one figure, every network × every combiner, mean NMSE "
                       "over the test SNRs and the change against the training combiner; 'cannot run' = learned for "
                       "another M."),
             dict(id="seeds", title="Mean ± std over seeds", build=b_seeds, fields=[
                 F("csv", "Evaluation table", "csv", "", "results/<name>/<SCENARIO>/nmse_vs_snr.csv", wide=True),
                 F("csv_manual", "or a path", "path", "", wide=True)]),
         ]),
]

EXTRA_TASKS = [dict(id="script", title="Any script", build=b_script, fields=[
    F("script", "Script", "script", "run_suite.py"),
    F("args", "Arguments", "text", "--help", "as on the command line", wide=True)])]

TASKS = {t["id"]: t for st in STEPS for t in st["tasks"]} | {t["id"]: t for t in EXTRA_TASKS}


def resolve_defaults(field):
    d = field["default"]
    if d == "@device":
        return SETTINGS.get("device", "cuda")
    if field["kind"] == "methods":
        return default_methods(d)
    return d


def meta():
    def ser(t):
        return dict(id=t["id"], title=t["title"], note=t.get("note", ""), dryrun=t.get("dryrun", False),
                    fields=[dict(f, default=resolve_defaults(f)) for f in t["fields"]])
    return dict(steps=[dict(id=s["id"], n=s["n"], title=s["title"], intro=s["intro"], tasks=[ser(t) for t in s["tasks"]])
                       for s in STEPS],
                extra=[ser(t) for t in EXTRA_TASKS],
                scenarios=scenario_files(), main_scenarios=MAIN_SCEN,
                scripts=sorted(p.name for p in (ROOT / "scripts").glob("*.py")),
                root=str(ROOT), settings=SETTINGS, python=step_python(), python_source=python_source(), conda=conda_exe(),
                platform=("Windows" if WIN else sys.platform), app_python=sys.executable, version=VERSION,
                others=others_alive())


# ----------------------------------------------------------------------------------------------------------------
# commands: build, display, resolve
# ----------------------------------------------------------------------------------------------------------------
def build(task_id, values):
    t = TASKS.get(task_id)
    if not t:
        raise ValueError(f"unknown task {task_id}")
    v = {f["key"]: resolve_defaults(f) for f in t["fields"]}
    v.update(values or {})
    return t["build"](v)


def q(a):
    a = str(a)
    if a == "" or re.search(r'[\s*?&|<>^()"]', a):
        return '"' + a.replace('"', '\\"') + '"'
    return a


def display(cmd, mode):
    argv = cmd["argv"]
    head = argv[0]
    rest = " ".join(q(x) for x in argv[1:])
    if head.startswith("{envpy"):
        return f"python {rest}"      # environment step: the Anaconda Prompt with the new environment activated
    if cmd["kind"] == "script":
        if mode != "spyder":
            return f"python {rest}"
        # %run expands * and ? itself (against the bsce folder); -G passes the patterns to the script unchanged
        glob_arg = any(re.search(r"[*?\[]", str(x)) for x in argv[2:])
        return f"%run -G {rest}" if glob_arg else f"%run {rest}"
    if head == "{conda}":
        return f"conda {rest}"
    if head.startswith("{envpy") or head == "{py}":
        return f"python {rest}"      # not a script: shown for the Anaconda Prompt in both modes
    return f"{q(head)} {rest}"


def where(cmd, mode):
    h = cmd["argv"][0]
    env = SETTINGS.get("env_name") or "bsce"
    if h == "{conda}":
        return "Anaconda Prompt, in the bsce folder"
    if h.startswith("{envpy"):
        return f"Anaconda Prompt, after: conda activate {h[len('{envpy:'):-1]}"
    if cmd["kind"] == "matlab":
        eng = "Octave" if "octave" in Path(str(h)).name.lower() else "MATLAB"
        return f"Anaconda Prompt or cmd ({eng} on PATH), any folder"
    if cmd["kind"] == "script" and mode == "spyder":
        return f"Spyder console of the {env} environment, working folder = the bsce folder"
    return f"Anaconda Prompt, after: conda activate {env} and cd to the bsce folder"


def resolve(cmd):
    argv = list(cmd["argv"])
    h = argv[0]
    if h == "{py}":
        argv[0] = step_python()
    elif h == "{conda}":
        c = conda_exe()
        if not c:
            raise RuntimeError("conda not found: start this app from the Anaconda Prompt, or set the conda path in Settings")
        argv[0] = c
    elif h.startswith("{envpy:"):
        name = h[len("{envpy:"):-1]
        p = env_python(name)
        if not p:
            raise RuntimeError(f"the conda environment '{name}' was not found (create it first)")
        argv[0] = p
        _PY_CACHE[name] = (p, time.time())
    return argv


def plan(task_id, values):
    try:
        cmds = build(task_id, values)
    except ValueError as e:
        return dict(ok=False, error=str(e), commands=[])
    return dict(ok=True, commands=[dict(label=c.get("label", ""), terminal=display(c, "terminal"),
                                        spyder=display(c, "spyder"), where_terminal=where(c, "terminal"),
                                        where_spyder=where(c, "spyder"), suite=c.get("suite", False))
                                   for c in cmds])


# ----------------------------------------------------------------------------------------------------------------
# progress: read the console output of the bsce scripts
# ----------------------------------------------------------------------------------------------------------------
RE_STAGE = re.compile(r"\bT=(\d+)(?: [a-z_]+=\S+)*\s+\[(extending|fine tuning)")
RE_I = re.compile(r"\bi=(\d+)\s+nmse=(-?[\d.]+) dB \(best=(-?[\d.]+)\)")
RE_E2E = re.compile(r"step=(\d+)\s+val nmse=(-?[\d.]+) dB \(best=(-?[\d.]+)\)")
RE_FT_BAND = re.compile(r"^=== (\S+) \| band (\S+) \|")
RE_FT_STEP = re.compile(r"step\s+(\d+) lr=\S+\s+val nmse=(-?[\d.]+) dB \(best (-?[\d.]+)\)")
RE_FT_DONE = re.compile(r"-> best val (-?[\d.]+) dB")
RE_RUN_HDR = re.compile(r"^=== (\S+) \| band (\S+) \| ([\d,]+) parameters")
RE_SNR = re.compile(r"^SNR=\s*(-?[\d.]+)")


def cell_label(line, scen):
    tag = re.search(r"--tag (\S+)", line)
    tag = tag.group(1) if tag else "?"
    kind = "fine-tune" if "finetune_bands.py" in line else ("train" if "train.py" in line else "step")
    return f"{kind} {tag}" + (f" on {scen}" if scen else "")


def suite_plan(argv_dry):
    """Run run_suite.py --dry-run -> [{label, state}] (todo for '>>>' lines, skip for finished steps)."""
    try:
        out = subprocess.run(argv_dry, cwd=ROOT, capture_output=True, text=True, timeout=300,
                             env=child_env(), encoding="utf-8", errors="replace").stdout
    except (OSError, subprocess.SubprocessError):
        return []
    cells, scen = [], ""
    for ln in out.splitlines():
        m = re.match(r"^######## (\S+)", ln)
        if m:
            scen = m.group(1)
        elif ln.startswith(">>> "):
            cells.append(dict(label=cell_label(ln, scen), state="todo"))
        elif re.match(r"^--- .*(skipped|fine-tuned, skipped)", ln):
            t = re.match(r"^--- (\S+?):", ln)
            cells.append(dict(label=f"done earlier: {t.group(1) if t else ln[4:40]} on {scen}", state="skip"))
    return cells


class Progress:
    def __init__(self):
        self.cells = []          # [{label, state: todo|run|done|fail|skip, cmd}]
        self.cur = None
        self.detail = ""
        self.metric = ""
        self.frac = None
        self.cell_t0 = None
        self.ctx = {}

    def add_cells(self, cells, cmd_index):
        for c in cells:
            c["cmd"] = cmd_index
        self.cells += cells

    def start_next(self, cmd_index):
        if self.cur is not None and self.cells[self.cur]["state"] == "run":
            self.cells[self.cur]["state"] = "done"
        nxt = next((i for i, c in enumerate(self.cells) if c["cmd"] == cmd_index and c["state"] == "todo"), None)
        if nxt is None:
            self.cells.append(dict(label="step", state="todo", cmd=cmd_index))
            nxt = len(self.cells) - 1
        self.cur = nxt
        self.cells[nxt]["state"] = "run"
        self.detail, self.metric, self.frac, self.cell_t0 = "starting", "", None, time.time()
        self.ctx = {"T": None, "stages": {}, "per_layer": None}

    def end_cmd(self, cmd_index, ok):
        for c in self.cells:
            if c["cmd"] != cmd_index:
                continue
            if c["state"] == "run":
                c["state"] = "done" if ok else "fail"
            elif c["state"] == "todo" and ok:          # planned, then not needed (finished meanwhile)
                c["state"] = "done"
        self.cur = None
        self.frac = None

    def feed(self, ln):
        ctx = self.ctx
        if ln.startswith(">>> "):
            m = re.search(r"model\.T=(\d+)", ln)
            ctx["T"] = int(m.group(1)) if m else None
            m = re.search(r"train\.steps=(\d+)", ln)
            ctx["steps"] = (20 if "--smoke" in ln else int(m.group(1))) if m else None
            m = re.search(r"--iters (\d+)", ln)
            ctx["iters"] = int(m.group(1)) if m else (20000 if "finetune_bands.py" in ln else None)   # its default
            m = re.search(r"--snr ((?:-?[\d.]+ ?)+)", ln)
            ctx["nbands"] = (len(m.group(1).split()) - 1) if m else 1
            ctx["band_i"] = -1
            return
        m = RE_RUN_HDR.match(ln)
        if m:
            self.detail = f"{m.group(1)}, band {m.group(2)}, {m.group(3)} parameters"
            return
        m = RE_STAGE.search(ln)
        if m and ctx.get("T"):
            t = int(m.group(1))
            st = ctx.setdefault("stages", {})
            st[t] = st.get(t, 0) + 1
            if t >= 2 and ctx.get("per_layer") is None:
                ctx["per_layer"] = st.get(1, 5)
            K = ctx.get("per_layer") or 5
            k = min(st[t], K)
            self.frac = min(0.99, ((t - 1) + (k - 0.5) / K) / ctx["T"])
            self.detail = f"layer {t} of {ctx['T']}, stage {k} of {K}"
            return
        m = RE_I.search(ln)
        if m:
            self.metric = f"validation {m.group(2)} dB (best {m.group(3)}) at i = {int(m.group(1)):,}"
            return
        m = RE_E2E.search(ln)
        if m:
            s = int(m.group(1))
            self.metric = f"validation {m.group(2)} dB (best {m.group(3)})"
            if ctx.get("steps"):
                self.frac = min(1.0, s / ctx["steps"])
                self.detail = f"end-to-end step {s:,} of {ctx['steps']:,}"
            else:
                self.detail = f"end-to-end step {s:,}"
            return
        m = RE_FT_BAND.match(ln)
        if m:
            ctx["band_i"] = ctx.get("band_i", -1) + 1
            ctx["band"] = m.group(2)
            nb = max(ctx.get("nbands") or 1, 1)
            self.frac = ctx["band_i"] / nb
            self.detail = f"fine-tune band {m.group(2)} ({ctx['band_i'] + 1} of {nb})"
            return
        m = RE_FT_STEP.search(ln)
        if m:
            s = int(m.group(1))
            it, nb = ctx.get("iters"), max(ctx.get("nbands") or 1, 1)
            bi = max(ctx.get("band_i", 0), 0)
            if it:
                self.frac = min(0.99, (bi + min(1, (s - bi * it) / it if s > it * bi else s / it)) / nb)
            self.metric = f"validation {m.group(2)} dB (best {m.group(3)}), step {s:,}"
            return
        m = RE_FT_DONE.search(ln)
        if m:
            self.metric = f"band done: best validation {m.group(1)} dB"
            return
        m = RE_SNR.match(ln)
        if m:
            self.detail = f"evaluating, SNR {m.group(1)} dB"
            return
        if ln.startswith("written ") or ln.startswith("saved "):
            self.metric = ln[:160]

    def eta(self):
        if self.frac and self.cell_t0 and self.frac > 0.02:
            el = time.time() - self.cell_t0
            return el * (1 - self.frac) / self.frac
        return None

    def as_dict(self):
        return dict(cells=[dict(label=c["label"], state=c["state"]) for c in self.cells], cur=self.cur,
                    detail=self.detail, metric=self.metric, frac=self.frac, eta=self.eta())


# ----------------------------------------------------------------------------------------------------------------
# jobs
# ----------------------------------------------------------------------------------------------------------------
def child_env():
    e = dict(os.environ)
    e.update(PYTHONUNBUFFERED="1", PYTHONIOENCODING="utf-8", MPLBACKEND="Agg")
    return e


class Job:
    def __init__(self, jid, task_id, title, cmds, parallel):
        self.id, self.task, self.title, self.cmds, self.parallel = jid, task_id, title, cmds, parallel
        self.status = "queued"
        self.created, self.started, self.ended = time.time(), None, None
        self.cmd_i = -1
        self.rc = None
        self.lines = []
        self.n_lines = 0
        self.proc = None
        self.stop_req = False
        self.error = ""
        self.prog = Progress()
        self.log = RUNS / f"{jid}.log"

    def summary(self):
        return dict(id=self.id, task=self.task, title=self.title, status=self.status, created=self.created,
                    started=self.started, ended=self.ended, cmd_i=self.cmd_i, n_cmds=len(self.cmds), rc=self.rc,
                    error=self.error, parallel=self.parallel, progress=self.prog.as_dict(),
                    commands=[display(c, "terminal") for c in self.cmds], labels=[c.get("label", "") for c in self.cmds],
                    n_lines=self.n_lines)


JOBS: dict = {}
ORDER: list = []
LOCK = threading.Lock()
WAKE = threading.Event()


PLOCK = threading.Lock()


def persist():
    with PLOCK:
        _persist()


def _persist():
    try:
        RUNS.mkdir(exist_ok=True)
        data = [JOBS[j].summary() | {"cmds_raw": JOBS[j].cmds} for j in ORDER][-200:]
        (RUNS / "jobs.json").write_text(json.dumps(data, indent=0, default=str), encoding="utf-8")
    except OSError:
        pass


def load_history():
    try:
        data = json.loads((RUNS / "jobs.json").read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return
    for d in data:
        j = Job(d["id"], d["task"], d["title"], d.get("cmds_raw", []), d.get("parallel", False))
        j.status = d["status"] if d["status"] in ("done", "failed", "stopped") else "interrupted"
        j.created, j.started, j.ended, j.rc, j.error = d["created"], d["started"], d["ended"], d.get("rc"), d.get("error", "")
        j.n_lines = d.get("n_lines", 0)
        try:
            j.lines = j.log.read_text(encoding="utf-8", errors="replace").splitlines()[-4000:]
            j.n_lines = len(j.lines)
        except OSError:
            pass
        j.prog.cells = [dict(c, cmd=0) for c in d.get("progress", {}).get("cells", [])]
        for c in j.prog.cells:
            if c["state"] == "run":
                c["state"] = "fail"
        JOBS[j.id] = j
        ORDER.append(j.id)


def add_line(job, ln):
    job.lines.append(ln)
    job.n_lines += 1
    if len(job.lines) > 20000:
        drop = len(job.lines) - 20000
        del job.lines[:drop]
    try:
        with open(job.log, "a", encoding="utf-8") as f:
            f.write(ln + "\n")
    except OSError:
        pass


def kill_tree(p):
    try:
        if WIN:
            subprocess.run(["taskkill", "/PID", str(p.pid), "/T", "/F"], capture_output=True)
        else:
            os.killpg(os.getpgid(p.pid), signal.SIGTERM)
    except (OSError, ProcessLookupError):
        pass


def run_job(job):
    job.status, job.started = "running", time.time()
    RUNS.mkdir(exist_ok=True)
    add_line(job, f"### {job.title}  ({time.strftime('%Y-%m-%d %H:%M:%S')})")
    persist()
    ok = True
    # plan the grid first (dry runs of the suite commands; one cell for every other command)
    for i, c in enumerate(job.cmds):
        cells = []
        if c.get("suite"):
            try:
                cells = suite_plan(resolve(c) + ["--dry-run"])
            except RuntimeError:
                cells = []
        if not cells:
            cells = [dict(label=c.get("label") or display(c, "terminal")[:60], state="todo")]
        job.prog.add_cells(cells, i)
    for i, c in enumerate(job.cmds):
        if job.stop_req:
            break
        job.cmd_i = i
        try:
            argv = resolve(c)
        except RuntimeError as e:
            add_line(job, f"!!! {e}")
            job.error, ok = str(e), False
            job.prog.start_next(i)
            job.prog.end_cmd(i, False)
            break
        add_line(job, "")
        add_line(job, f"$ {display(c, 'terminal')}")
        if not c.get("suite"):
            job.prog.start_next(i)
        kw = dict(cwd=ROOT, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, env=child_env(), bufsize=0)
        if WIN:
            kw["creationflags"] = subprocess.CREATE_NEW_PROCESS_GROUP
        else:
            kw["start_new_session"] = True
        try:
            job.proc = subprocess.Popen(argv, **kw)
        except OSError as e:
            add_line(job, f"!!! cannot start {argv[0]}: {e}")
            job.error, ok = f"cannot start {argv[0]}: {e}", False
            job.prog.end_cmd(i, False)
            break
        buf = b""
        while True:
            chunk = job.proc.stdout.read1(65536) if hasattr(job.proc.stdout, "read1") else job.proc.stdout.read(4096)
            if not chunk:
                break
            buf += chunk
            parts = re.split(rb"\r\n|\n|\r", buf)
            buf = parts.pop()
            for raw in parts:
                ln = raw.decode("utf-8", errors="replace")
                if c.get("suite") and ln.startswith(">>> "):
                    job.prog.start_next(i)
                job.prog.feed(ln)
                add_line(job, ln)
        if buf:
            ln = buf.decode("utf-8", errors="replace")
            job.prog.feed(ln)
            add_line(job, ln)
        rc = job.proc.wait()
        job.rc = rc
        job.prog.end_cmd(i, rc == 0 and not job.stop_req)
        if rc != 0 or job.stop_req:
            ok = False
            if not job.stop_req:
                last = next((x.strip() for x in reversed(job.lines[-60:]) if x.strip() and not x.startswith("$ ")), "")
                job.error = f"command {i + 1} of {len(job.cmds)} stopped with exit code {rc}" + (f". Last output: {last[:300]}" if last else "")
                add_line(job, f"!!! command {i + 1} of {len(job.cmds)} stopped with exit code {rc}")
            break
    job.proc = None
    job.ended = time.time()
    job.status = "stopped" if job.stop_req else ("done" if ok else "failed")
    add_line(job, f"### {job.status} after {fmt_dur(job.ended - job.started)}")
    persist()


def fmt_dur(s):
    s = int(s)
    return f"{s // 3600}h {s % 3600 // 60:02d}m" if s >= 3600 else f"{s // 60}m {s % 60:02d}s"


def worker():
    while True:
        WAKE.wait(1.0)
        WAKE.clear()
        with LOCK:
            busy = any(JOBS[j].status in ("running", "starting") and not JOBS[j].parallel for j in ORDER)
            nxt = None if busy else next((JOBS[j] for j in ORDER if JOBS[j].status == "queued" and not JOBS[j].parallel), None)
            if nxt:
                nxt.status = "starting"
        if nxt:
            threading.Thread(target=lambda j=nxt: (run_job(j), WAKE.set()), daemon=True).start()


def submit(task_id, values, parallel):
    cmds = build(task_id, values)
    t = TASKS[task_id]
    if len(cmds) == 1:
        lab = cmds[0].get("label") or ""
        title = lab[0].upper() + lab[1:] if lab else t["title"]
    else:
        title = f"{t['title']} ({len(cmds)} commands)"
    return submit_cmds(task_id, title, cmds, parallel)


def submit_cmds(task_id, title, cmds, parallel):
    jid = time.strftime("%Y%m%d-%H%M%S-") + secrets.token_hex(2)
    j = Job(jid, task_id, title, cmds, parallel)
    with LOCK:
        JOBS[jid] = j
        ORDER.append(jid)
    persist()
    if parallel:
        j.status = "starting"
        threading.Thread(target=run_job, args=(j,), daemon=True).start()
    else:
        WAKE.set()
    return j


def stop(jid):
    j = JOBS.get(jid)
    if not j:
        return False
    if j.status in ("queued",):
        j.status = "stopped"
        j.ended = time.time()
        persist()
        return True
    j.stop_req = True
    if j.proc and j.proc.poll() is None:
        kill_tree(j.proc)
    return True


# ----------------------------------------------------------------------------------------------------------------
# folders: trained networks and results
# ----------------------------------------------------------------------------------------------------------------
def outputs_status(outputs="outputs"):
    base = ROOT / outputs
    res = []
    if not base.is_dir():
        return res
    for sc in sorted(base.iterdir()):
        if not sc.is_dir():
            continue
        runs = []
        for run in sorted(sc.iterdir()):
            if not run.is_dir():
                continue
            pts = sorted(run.glob(f"{run.name}_*.pt"))
            bands = [p.stem[len(run.name) + 1:] for p in pts]
            ck = list((run / "checkpoints").glob("*.pt")) if (run / "checkpoints").is_dir() else []
            layers = len(list((run / "layers").glob("T*.pt"))) if (run / "layers").is_dir() else 0
            first = ""
            try:
                first = (run / "README.txt").read_text(encoding="utf-8", errors="replace").splitlines()[0][:160]
            except (OSError, IndexError):
                pass
            info = {}
            for lf, kind in (("finetune_log.json", "ft"), ("train_log.json", "train")):
                try:
                    d = json.loads((run / lf).read_text(encoding="utf-8"))
                    if kind == "ft":
                        info["ft_iters"] = d.get("args", {}).get("iters")
                        info["best"] = {k: round(b.get("best_val_db", 0), 3) for k, b in d.get("bands", {}).items()}
                    else:
                        info["best"] = {k: round(b.get("final_val_nmse_db", 0) or 0, 3)
                                        for k, b in d.get("bands", {}).items() if isinstance(b, dict)}
                    break
                except (OSError, ValueError, AttributeError):
                    pass
            mt = max([p.stat().st_mtime for p in pts + ck] or [run.stat().st_mtime])
            newest_pt = max([p.stat().st_mtime for p in pts] or [0])
            resumable = bool(ck) and max(p.stat().st_mtime for p in ck) > newest_pt   # saved after the last network
            state = "in progress (resumable)" if resumable else ("trained" if pts else "empty")
            runs.append(dict(name=run.name, bands=bands, state=state, layers=layers, readme=first, mtime=mt,
                             ft_iters=info.get("ft_iters"), best=info.get("best", {}), checkpoint=resumable))
        res.append(dict(scenario=sc.name, runs=runs))
    return res


def results_list(results="results"):
    base = ROOT / results
    out = []
    if not base.is_dir():
        return out
    for d in sorted(base.iterdir(), key=lambda p: p.stat().st_mtime, reverse=True):
        if not d.is_dir():
            continue
        files = sorted(str(p.relative_to(ROOT)).replace("\\", "/") for p in d.rglob("*") if p.is_file())
        out.append(dict(name=d.name, mtime=d.stat().st_mtime, files=files))
    return out


def csv_files():
    return sorted(str(p.relative_to(ROOT)).replace("\\", "/") for p in (ROOT / "results").glob("*/*/nmse_vs_snr.csv")) \
        if (ROOT / "results").is_dir() else []


_GPU = {"t": 0.0, "v": {}}


def gpu_status():
    """Load, memory and temperature of the GPU(s) from nvidia-smi (cached 3 s; {} without an NVIDIA driver)."""
    if time.time() - _GPU["t"] < 3:
        return _GPU["v"]
    _GPU["t"] = time.time()
    exe = shutil.which("nvidia-smi") or ("C:/Windows/System32/nvidia-smi.exe" if WIN and Path("C:/Windows/System32/nvidia-smi.exe").exists() else "")
    v = {}
    if exe:
        try:
            out = subprocess.run([exe, "--query-gpu=name,utilization.gpu,memory.used,memory.total,temperature.gpu",
                                  "--format=csv,noheader,nounits"], capture_output=True, text=True, timeout=5).stdout
            g = []
            for ln in out.strip().splitlines():
                n, u, mu, mt, t = [x.strip() for x in ln.split(",")]
                g.append(dict(name=n, util=int(float(u)), mem_used=float(mu), mem_total=float(mt), temp=int(float(t))))
            v = {"gpus": g}
        except (OSError, ValueError, subprocess.SubprocessError):
            v = {}
    _GPU["v"] = v
    return v


# ----------------------------------------------------------------------------------------------------------------
# HTTP
# ----------------------------------------------------------------------------------------------------------------
SERVE_EXT = {".png", ".pdf", ".svg", ".md", ".csv", ".txt", ".tex", ".json", ".log", ".html"}


class H(BaseHTTPRequestHandler):
    server_version = "bsce-workbench"

    def log_message(self, *a):
        pass

    def _host_ok(self):
        h = (self.headers.get("Host") or "").split(":")[0]
        return h in ("127.0.0.1", "localhost")

    def _send(self, code, body, ctype="application/json; charset=utf-8", extra=None):
        if isinstance(body, (dict, list)):
            body = json.dumps(body, default=str).encode()
        elif isinstance(body, str):
            body = body.encode()
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        for k, v in (extra or {}).items():
            self.send_header(k, v)
        self.end_headers()
        self.wfile.write(body)

    def _auth(self, qs=None):
        t = self.headers.get("X-Token") or (qs or {}).get("t", [""])[0]
        return self._host_ok() and secrets.compare_digest(t, TOKEN)

    def do_GET(self):
        u = urlparse(self.path)
        qs = parse_qs(u.query)
        if not self._host_ok():
            return self._send(403, {"error": "local use only"})
        if u.path in ("/", "/index.html"):
            html = (STATIC / "index.html").read_text(encoding="utf-8").replace("__TOKEN__", TOKEN)
            return self._send(200, html, "text/html; charset=utf-8")
        if u.path.startswith("/static/"):
            p = (STATIC / u.path[len("/static/"):]).resolve()
            if STATIC in p.parents and p.is_file():
                return self._send(200, p.read_bytes(), mimetypes.guess_type(str(p))[0] or "application/octet-stream")
            return self._send(404, {"error": "not found"})
        if u.path == "/api/whoami":                    # no token: lets a newer workbench tell which one runs here
            return self._send(200, dict(app="bsce-workbench", version=VERSION, root=str(ROOT), pid=os.getpid()))
        if u.path == "/file":
            if not self._auth(qs):
                return self._send(403, {"error": "token"})
            p = (ROOT / qs.get("p", [""])[0]).resolve()
            if (ROOT in p.parents) and p.is_file() and p.suffix.lower() in SERVE_EXT:
                ct = mimetypes.guess_type(str(p))[0] or "text/plain"
                if p.suffix.lower() in (".md", ".csv", ".tex", ".log", ".txt"):
                    ct = "text/plain; charset=utf-8"
                return self._send(200, p.read_bytes(), ct)
            return self._send(404, {"error": "not found"})
        if not u.path.startswith("/api/") or not self._auth(qs):
            return self._send(403, {"error": "token"})
        if u.path == "/api/meta":
            return self._send(200, meta())
        if u.path == "/api/jobs":
            return self._send(200, [JOBS[j].summary() for j in reversed(ORDER)])
        if u.path == "/api/job":
            j = JOBS.get(qs.get("id", [""])[0])
            if not j:
                return self._send(404, {"error": "no such run"})
            frm = int(qs.get("from", ["0"])[0])
            first = j.n_lines - len(j.lines)
            k = max(frm - first, 0)
            return self._send(200, dict(job=j.summary(), lines=j.lines[k:k + 5000], next=first + min(len(j.lines), k + 5000)))
        if u.path == "/api/outputs":
            d = qs.get("dir", ["outputs"])[0]
            if ".." in d or Path(d).is_absolute():
                return self._send(400, {"error": "a folder inside bsce"})
            return self._send(200, outputs_status(d))
        if u.path == "/api/gpu":
            return self._send(200, gpu_status())
        if u.path == "/api/results":
            return self._send(200, results_list())
        if u.path == "/api/choices":
            return self._send(200, dict(networks=output_networks(), csv=csv_files(), scenarios=scenario_files(),
                                        output_dirs=output_dirs()))
        return self._send(404, {"error": "unknown"})

    def do_POST(self):
        u = urlparse(self.path)
        if not self._auth():
            return self._send(403, {"error": "token"})
        n = int(self.headers.get("Content-Length") or 0)
        try:
            body = json.loads(self.rfile.read(n) or b"{}")
        except ValueError:
            return self._send(400, {"error": "bad json"})
        try:
            if u.path == "/api/plan":
                return self._send(200, plan(body.get("task"), body.get("values")))
            if u.path == "/api/run":
                j = submit(body.get("task"), body.get("values"), bool(body.get("parallel")))
                return self._send(200, j.summary())
            if u.path == "/api/dryrun":
                cmds = build(body.get("task"), body.get("values"))
                out = []
                for c in cmds:
                    if not c.get("suite"):
                        continue
                    r = subprocess.run(resolve(c) + ["--dry-run"], cwd=ROOT, capture_output=True, text=True, timeout=300,
                                       env=child_env(), encoding="utf-8", errors="replace")
                    out.append(f"$ {display(c, 'terminal')} --dry-run\n{r.stdout}{r.stderr}")
                return self._send(200, {"text": "\n".join(out) or "(nothing to plan)"})
            if u.path == "/api/stop":
                return self._send(200, {"ok": stop(body.get("id"))})
            if u.path == "/api/remove":
                jid = body.get("id")
                with LOCK:
                    if jid in JOBS and JOBS[jid].status not in ("running", "starting"):
                        ORDER.remove(jid)
                        JOBS.pop(jid)
                persist()
                return self._send(200, {"ok": True})
            if u.path == "/api/settings":
                for k in ("python", "conda", "env_name", "device", "show"):
                    if k in body:
                        SETTINGS[k] = str(body[k]).strip()
                _PY_CACHE.clear()
                save_settings()
                return self._send(200, meta())
            if u.path == "/api/help":
                sc = body.get("script", "")
                if not re.fullmatch(r"[A-Za-z0-9_]+\.py", sc):
                    return self._send(400, {"error": "script"})
                r = subprocess.run([step_python(), f"scripts/{sc}", "--help"], cwd=ROOT, capture_output=True, text=True,
                                   timeout=120, env=child_env(), encoding="utf-8", errors="replace")
                return self._send(200, {"text": r.stdout + r.stderr})
            if u.path == "/api/exists":
                res = []
                for t in body.get("paths", []):
                    p = Path(str(t).strip().strip('"'))
                    p = p if p.is_absolute() else ROOT / p
                    n = len(glob.glob(str(p))) if any(ch in str(p) for ch in "*?[") else int(p.exists())
                    res.append(n)
                return self._send(200, {"found": res})
            if u.path == "/api/stop_other":
                return self._send(200, stop_other(int(body.get("port", 0))))
            if u.path == "/api/rerun":
                old = JOBS.get(body.get("id"))
                if not old or not old.cmds:
                    return self._send(404, {"error": "no such run"})
                return self._send(200, submit_cmds(old.task, old.title, old.cmds, bool(body.get("parallel"))).summary())
        except (ValueError, RuntimeError) as e:
            return self._send(400, {"error": str(e)})
        return self._send(404, {"error": "unknown"})


# ----------------------------------------------------------------------------------------------------------------
# one port per workbench. Python's HTTP server sets SO_REUSEADDR, and on Windows that lets a second program bind a
# port another one is still listening on: the browser then reaches the OLD workbench (older version, other folder,
# its own runs). So: exclusive binding on Windows, and every port is probed before it is used.
# ----------------------------------------------------------------------------------------------------------------
class Server(ThreadingHTTPServer):
    allow_reuse_address = not WIN

    def server_bind(self):
        if WIN and hasattr(socket, "SO_EXCLUSIVEADDRUSE"):
            self.socket.setsockopt(socket.SOL_SOCKET, socket.SO_EXCLUSIVEADDRUSE, 1)
        super().server_bind()


def probe_port(port, timeout=1.0):
    """None when nothing listens on 127.0.0.1:port; else what answers there: a bsce workbench (with its version and
    folder when it is new enough to say) or another program."""
    try:
        with socket.create_connection(("127.0.0.1", port), timeout=timeout):
            pass
    except OSError:
        return None
    import http.client
    info = dict(port=port, url=f"http://127.0.0.1:{port}/", kind="other", version="", root="")
    try:
        c = http.client.HTTPConnection("127.0.0.1", port, timeout=timeout)
        c.request("GET", "/api/whoami", headers={"Host": f"127.0.0.1:{port}"})
        r = c.getresponse()
        body = r.read()
        if (r.getheader("Server") or "").startswith("bsce-workbench"):
            info["kind"] = "bsce"
            if r.status == 200:
                d = json.loads(body)
                info.update(version=str(d.get("version", "")), root=str(d.get("root", "")))
        c.close()
    except (OSError, ValueError, http.client.HTTPException):
        pass
    return info


def others_alive():
    """The other workbenches found at start that still answer."""
    OTHERS[:] = [o for o in OTHERS if probe_port(o["port"], 0.3)]
    return OTHERS


def listening_pids(port):
    """Process ids listening on TCP port `port` (Windows: netstat; Linux: /proc)."""
    pids = set()
    try:
        if WIN:
            out = subprocess.run(["netstat", "-ano", "-p", "TCP"], capture_output=True, text=True, timeout=20).stdout
            for ln in out.splitlines():
                f = ln.split()
                # a listening socket has the foreign address 0.0.0.0:0 (the state word is translated in some Windows)
                if len(f) >= 5 and f[0].upper() == "TCP" and f[1].endswith(f":{port}") and f[2] in ("0.0.0.0:0", "[::]:0") \
                        and f[-1].isdigit():
                    pids.add(int(f[-1]))
        elif Path("/proc/net/tcp").exists():
            inodes = set()
            for t in ("/proc/net/tcp", "/proc/net/tcp6"):
                if Path(t).exists():
                    for ln in Path(t).read_text().splitlines()[1:]:
                        f = ln.split()
                        if int(f[1].split(":")[1], 16) == port and f[3] == "0A":
                            inodes.add(f[9])
            for d in Path("/proc").iterdir():
                if d.name.isdigit():
                    try:
                        if any(os.readlink(x).strip("socket:[]") in inodes for x in (d / "fd").iterdir()):
                            pids.add(int(d.name))
                    except OSError:
                        continue
    except (OSError, ValueError, subprocess.SubprocessError):
        pass
    pids.discard(os.getpid())
    return sorted(pids)


def command_line(pid):
    try:
        if WIN:
            r = subprocess.run(["powershell", "-NoProfile", "-Command",
                                f"(Get-CimInstance Win32_Process -Filter 'ProcessId={pid}').CommandLine"],
                               capture_output=True, text=True, timeout=30)
            return r.stdout.strip()
        return Path(f"/proc/{pid}/cmdline").read_bytes().replace(b"\0", b" ").decode(errors="replace")
    except (OSError, subprocess.SubprocessError):
        return ""


def stop_other(port):
    """Stop the bsce workbench(es) listening on `port` (with the steps they run)."""
    info = probe_port(port)
    if not info or info["kind"] != "bsce":
        return dict(ok=False, error=f"no bsce workbench answers on port {port}")
    def is_workbench(pid):
        cl = command_line(pid).lower()
        if cl:
            return "bsce_app" in cl
        if WIN:                          # no command line (PowerShell missing): a python process is enough here
            r = subprocess.run(["tasklist", "/FI", f"PID eq {pid}", "/NH", "/FO", "CSV"], capture_output=True, text=True)
            return "python" in r.stdout.lower()
        return False
    pids = [p for p in listening_pids(port) if is_workbench(p)]
    if not pids:
        return dict(ok=False, error="its process was not found: close its window (the black console window)")
    for pid in pids:
        try:
            if WIN:
                subprocess.run(["taskkill", "/PID", str(pid), "/T", "/F"], capture_output=True, timeout=60)
            else:
                os.kill(pid, signal.SIGTERM)
        except (OSError, subprocess.SubprocessError):
            pass
    for _ in range(20):
        if not probe_port(port, 0.3):
            break
        time.sleep(0.25)
    left = probe_port(port, 0.3)
    others_alive()
    return dict(ok=not left, pids=pids, error="" if not left else "it is still running: close its window")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--port", type=int, default=8765)
    ap.add_argument("--no-browser", action="store_true")
    a = ap.parse_args()
    RUNS.mkdir(exist_ok=True)
    load_history()
    threading.Thread(target=worker, daemon=True).start()
    srv = None
    for port in range(a.port, a.port + 20):
        other = probe_port(port, 0.5)
        if other:                                   # something already listens here: never share a port
            OTHERS.append(other)
            continue
        try:
            srv = Server(("127.0.0.1", port), H)
            break
        except OSError:
            continue
    if srv is None:
        sys.exit(f"no free port in {a.port}..{a.port + 19}")
    url = f"http://127.0.0.1:{srv.server_address[1]}/"
    for o in OTHERS:
        if o["kind"] == "bsce":
            print(f"NOTE: another bsce workbench is still running at {o['url']}"
                  + (f" (version {o['version']}, folder {o['root']})" if o["root"] else " (an older version)")
                  + ".\n      Its runs and settings are its own. Close its window to stop it (its running steps stop"
                  " too), or use the button on this workbench's page.", flush=True)
        else:
            print(f"NOTE: port {o['port']} is used by another program; this workbench uses another port.", flush=True)
    print(f"bsce workbench {VERSION}: {url}\n  bsce folder: {ROOT}\n  python for the steps: {step_python()}\n"
          f"  keep this window open while runs are going; Ctrl+C stops the app (and its runs).", flush=True)
    if not a.no_browser:
        threading.Timer(0.8, lambda: webbrowser.open(url)).start()
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        print("stopping: running commands are stopped too")
        for j in JOBS.values():
            if j.proc and j.proc.poll() is None:
                kill_tree(j.proc)


if __name__ == "__main__":
    main()

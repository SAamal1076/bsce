"""Run original MATLAB code in GNU Octave for equivalence tests.

* ``run_octave(code, workdir, ...)`` writes ``code`` to a script in ``workdir`` and runs it
  with octave-cli; raises with the Octave output if it fails.
* With ``replay=(u, n)`` the shims in ``octave_shims/`` replace rand / randn / normrnd so
  the *unchanged* original .m files consume exactly the numbers Python replays.
"""
from __future__ import annotations

import os
import shutil
import subprocess
import tempfile
from pathlib import Path

import numpy as np
import scipy.io as sio

HERE = Path(__file__).resolve().parent
SHIMS = HERE / "octave_shims"
REPO = HERE.parent.parent
REFERENCE = REPO / "reference"
# BSCE_OCTAVE = full path to octave-cli(.exe) if it is not on PATH (e.g. on Windows:
# C:\Program Files\GNU Octave\Octave-<version>\mingw64\bin\octave-cli.exe)
OCTAVE = os.environ.get("BSCE_OCTAVE") or shutil.which("octave-cli") or shutil.which("octave")


def octave_available() -> bool:
    return OCTAVE is not None


def tf_available(*extra) -> bool:
    """TensorFlow (+ extra modules) importable? The TF reference runs need requirements-verify.txt."""
    import importlib.util
    return all(importlib.util.find_spec(m) is not None for m in ("tensorflow",) + extra)


def make_streams(seed: int, n_uniform: int, n_normal: int):
    g = np.random.default_rng(seed)
    return g.random(n_uniform), g.standard_normal(n_normal)


def run_octave(code: str, workdir, addpaths=(), replay=None, timeout: int = 3600) -> str:
    workdir = Path(workdir)
    workdir.mkdir(parents=True, exist_ok=True)
    env = dict(os.environ)
    pre = ["warning('off','all');", "more off;", "save_default_options('-v7');  % MATLAB-like default .mat format"]
    if replay is not None:
        u, n = replay
        stream = workdir / "_bsce_stream.mat"
        sio.savemat(stream, {"u": np.asarray(u, float).reshape(-1, 1),
                             "n": np.asarray(n, float).reshape(-1, 1)})
        env["BSCE_STREAM"] = str(stream)
    for p in addpaths:
        pre.append(f"addpath('{Path(p).as_posix()}');")
    if replay is not None:  # shims must come first on the path to shadow rand/randn
        pre.append(f"addpath('{SHIMS.as_posix()}');")
    script = workdir / "_bsce_run.m"
    script.write_text("\n".join(pre) + "\n" + code + "\n", encoding="utf-8")
    proc = subprocess.run([OCTAVE, "--no-gui", "--quiet", "--norc", script.name],
                          cwd=workdir, env=env, capture_output=True, text=True, encoding="utf-8",
                          errors="replace", timeout=timeout)
    if proc.returncode != 0:
        raise RuntimeError(f"Octave failed (code {proc.returncode}):\n{proc.stdout}\n{proc.stderr}")
    return proc.stdout


def tmpdir(prefix="bsce_oct_"):
    return Path(tempfile.mkdtemp(prefix=prefix))


def compare(a, b, name="", tol=1e-12):
    """Return dict(max_abs, rel) and assert rel <= tol (relative Frobenius error)."""
    a = np.asarray(a)
    b = np.asarray(b)
    assert a.shape == b.shape, f"{name}: shape {a.shape} vs {b.shape}"
    diff = np.abs(a - b)
    max_abs = float(diff.max()) if diff.size else 0.0
    denom = float(np.linalg.norm(b.ravel())) or 1.0
    rel = float(np.linalg.norm((a - b).ravel()) / denom)
    assert rel <= tol, f"{name}: relative error {rel:.3e} > {tol:.1e} (max abs {max_abs:.3e})"
    return {"name": name, "max_abs": max_abs, "rel": rel}

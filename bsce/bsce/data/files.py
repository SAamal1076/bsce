"""Measurement files of a scenario band, and SNR bands from the command line.

* ensure_band_files(): makes the missing training / validation measurement files of a band, like
  generate_SV_data.py / generate_measurement_data.py (K columns in n_snr_chunks equal chunks, chunk k at
  SNR_k ~ U(lo, hi)), and stores the SNR of every column as 'snr_db' (1 x K) next to 'y' (models that need
  the noise variance, e.g. NF-LADMM, read it; the original scripts ignore extra variables).
* online_validation(): fixed validation set for online-noise training: one SNR per column ~ U(lo, hi),
  fixed seed, so every method trained with online noise sees the same validation set.
* noise model (scenario key 'noise_model', bsce/data/noise.py): 'colored' (default, y = A (x + n)) or 'white'
  (y = A x + w). White measurement files are named <colored name>_white.mat and are the colored files of the same
  draw with the noise whitened (made from the colored file when it exists, e.g. the authors' files).
* bands_from_edges(): '--snr 0 20' -> one band 0to20dB; '--snr 0 10 20' -> 0to10dB, 10to20dB (100 SNR chunks per
  measurement file, as the user's generate_SV_data.py).
* apply_band_options(): --margin (train each band on a wider range, use it on its own), --val-snr (validation over
  the band range, or at fixed SNRs as the authors), --snr-chunks (chunks of the training file), --snr-dist fair
  (v26: every training column its own SNR, spread evenly over the band, random order; --snr-step S: on the grid
  lo, lo+S, ..., hi with equal counts; validation over a range: equally spaced SNRs instead of random ones). Files
  named <band>_fair[<step>].mat, so no existing file changes.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import List, Optional

import numpy as np
import scipy.io as sio

from . import noise as noise_models
from .measurements import draw_snr_list, evenly_spaced_snr, fair_snr, gen_measurements_percol, gen_measurements_py


def _fmt(v: float) -> str:
    return f"{v:g}".replace("-", "m").replace(".", "p")


def bands_from_edges(edges) -> List[dict]:
    e = [float(v) for v in edges]
    if len(e) < 2 or any(b <= a for a, b in zip(e, e[1:])):
        raise ValueError(f"--snr needs increasing edges, e.g. 0 20 or 0 10 20 (got {edges})")
    out = []
    for i, (lo, hi) in enumerate(zip(e, e[1:])):
        last = i == len(e) - 2
        out.append(dict(name=f"{_fmt(lo)}to{_fmt(hi)}dB", train=[lo, hi], val_range=[lo, hi], n_snr_chunks=100,
                        use_for=[lo, hi + 0.01 if last else hi]))
    return out


def _pts(v) -> str:
    return "-".join(_fmt(float(x)) for x in v)


SNR_DISTS = ("chunks", "fair")


def snr_dist_suffix(snr_dist: Optional[str] = None, snr_step=None) -> str:
    """'' (chunks, as before), 'fair' or 'fair0p2' (with a grid step): the name part of the files and runs."""
    if snr_dist in (None, "chunks"):
        if snr_step:
            raise ValueError("--snr-step needs --snr-dist fair")
        return ""
    if snr_dist != "fair":
        raise ValueError(f"--snr-dist: one of {SNR_DISTS} (got {snr_dist!r})")
    if snr_step is not None and float(snr_step) <= 0:
        raise ValueError(f"--snr-step must be > 0 dB (got {snr_step})")
    return "fair" + (_fmt(float(snr_step)) if snr_step else "")


def apply_band_options(bands, margin: float = 0.0, val=None, chunks: Optional[int] = None,
                       default_chunks: int = 10, snr_dist: Optional[str] = None, snr_step=None) -> List[dict]:
    """The data options of train.py / finetune_bands.py / run_suite.py, applied to a list of bands (new dicts).

    margin M (dB): every band is TRAINED on [lo - M, hi + M] and still USED for [lo, hi] (its name, use_for and
        validation stay those of [lo, hi]). The edges of a band are then trained like its middle. Default 0 = as the
        authors (0-10 and 10-20 dB trained exactly on their range).
    val: None = as the band defines it (the authors' scenario bands: their val_snr, 10 dB for both bands; --snr bands:
        the whole band range); 'range' = validation noise with SNR ~ U(lo, hi) of the band (fixed, n_snr_chunks
        chunks); a list of SNRs in dB = fixed points, equal chunks (the authors: [10]).
    chunks: SNR chunks of the training file (the authors 10; --snr bands 100). default_chunks: the scenario's
    n_snr_chunks (used by bands that do not set their own).
    snr_dist 'fair' (v26): every training column its own SNR, spread evenly over the training range (stratified,
        random order); snr_step S: the grid lo, lo+S, ..., hi, equal counts. A validation over a range then uses
        equally spaced SNRs (the middles of n equal parts) instead of random ones; validation at points is unchanged.
        None / 'chunks': as before.
    Files that differ get their own names (train_name / val_name), so no existing measurement file changes."""
    out = []
    for b0 in bands:
        b = dict(b0)
        lo, hi = map(float, b0["train"])
        c0 = int(b0.get("n_snr_chunks", default_chunks))
        c = int(chunks) if chunks else c0
        if c != c0:
            b["n_snr_chunks"] = c
            b["val_n_snr_chunks"] = int(b0.get("val_n_snr_chunks", c0))    # --snr-chunks changes the training file only
        if margin:
            m = float(margin)
            b["train"] = [lo - m, hi + m]
            b["use_range"] = [lo, hi]
            b["val_seed_range"] = [lo, hi]          # the validation file stays the one of the band itself
            # the same name as a --snr band of that range only if the file is the same (100 chunks)
            b["train_name"] = f"{_fmt(lo - m)}to{_fmt(hi + m)}dB" + ("" if c == 100 else f"_c{c}")
            b["n_snr_chunks"] = c
        elif c != c0:
            b["train_name"] = b0.get("train_name", b0["name"]) + f"_c{c}"
        if val is not None:
            if isinstance(val, str) and val == "range":
                if "val_range" not in b0:            # authors' band (val at points) -> over its range
                    b.pop("val_snr", None)
                    b["val_range"] = [lo, hi]
                    b["val_n_snr_chunks"] = 100
                    b["val_name"] = f"{b0['name']}_valrange"
            else:
                pts = [float(x) for x in val]
                if [float(x) for x in b0.get("val_snr", [])] != pts or "val_range" in b0:
                    b.pop("val_range", None)
                    b["val_snr"] = pts
                    b["val_name"] = f"{b0['name']}_val{_pts(pts)}dB"
                b["val_points"] = True
        sfx = snr_dist_suffix(snr_dist, snr_step)
        if sfx:
            if chunks:
                raise ValueError("--snr-chunks has no effect with --snr-dist fair (every column has its own SNR)")
            b["snr_dist"] = "fair"
            if snr_step:
                b["snr_step"] = float(snr_step)
            b["train_name"] = b.get("train_name", b["name"]) + "_" + sfx
            if "val_range" in b and not b.get("val_points"):
                b["val_name"] = b.get("val_name", b["name"]) + "_fair"
        out.append(b)
    return out


def band_range_from_name(name: str):
    """'0to20dB' -> (0.0, 20.0); 'm5to10dB' -> (-5, 10); None if the name is not of that form."""
    if not name.endswith("dB") or "to" not in name:
        return None
    try:
        lo, hi = name[:-2].split("to")
        f = lambda s: float(s.replace("m", "-").replace("p", "."))
        return f(lo), f(hi)
    except ValueError:
        return None


def band_seed_offset(band: dict) -> int:
    """Noise seed offset of a band, from its SNR edges only: the same band always gets the same measurement files,
    whatever other bands are trained with it and on whichever computer the file is made. 0-20 dB keeps offset 0
    (the files made so far for one 0-20 dB network are unchanged)."""
    lo, hi = (float(v) for v in band["train"])
    if (lo, hi) == (0.0, 20.0):
        return 0
    key = int(round((lo + 100) * 10)) * 10007 + int(round((hi + 100) * 10))
    return 1000 * (1 + key % 99991)


def ensure_band_files(sc: dict, band: dict, band_index: int, data: Path, A: np.ndarray, file_names,
                      seed: int = 0, force: bool = False, only: Optional[str] = None, verbose: bool = True):
    """Create the band's missing measurement files (training: SNR ~ U(band.train); validation: U(band.val_range)
    or the band's val_snr points). Band key snr_dist 'fair' (apply_band_options): training one SNR per column spread
    evenly (fair_snr), validation over a range at equally spaced SNRs. Returns the list of files written."""
    pre = f"{sc['dataset']}_{sc['array']}"
    chunks = int(band.get("n_snr_chunks", sc.get("n_snr_chunks", 10)))
    ftr, fva = file_names
    nm = noise_models.of_scenario(sc)
    if nm == "white":                                   # the colored twin of each file (same draw)
        from ..config import band_file_names
        ctr, cva = band_file_names(dict(sc, noise_model="colored"), band)
        twin = {ftr: ctr, fva: cva}
    jobs = []
    if only in (None, "train"):
        jobs.append(("training", ftr, ("range", band["train"])))
    if only in (None, "val"):
        jobs.append(("validation", fva, ("range", band["val_range"]) if "val_range" in band
                     else ("points", band.get("val_snr", band["train"]))))
    written = []
    for kind, fname, (mode, v) in jobs:
        out = Path(data) / fname
        if out.exists() and not force:
            if verbose:
                print(f"keep   {out}")
            continue
        x = sio.loadmat(Path(data) / f"{pre}_{kind}_channel.mat")["x"]
        if nm == "white" and (Path(data) / twin[fname]).exists():
            d = sio.loadmat(Path(data) / twin[fname])
            if d["y"].shape[1] != x.shape[1]:
                raise ValueError(f"{twin[fname]}: {d['y'].shape[1]} columns for {x.shape[1]} channels")
            out_d = {"y": noise_models.apply(d["y"].astype(np.complex128), x, A, "white"), "noise_model": "white",
                     "colored_twin": twin[fname]}
            if "snr_db" in d:
                out_d["snr_db"] = d["snr_db"]
            sio.savemat(out, out_d)
            js = Path(data) / (twin[fname] + ".snr.json")
            if js.exists():
                Path(str(out) + ".snr.json").write_text(js.read_text(encoding="utf-8"), encoding="utf-8")
            written.append(out)
            if verbose:
                print(f"wrote  {out}  (white: the noise of {twin[fname]} whitened, same draw)")
            continue
        sb = band if kind == "training" or "val_seed_range" not in band else dict(band, train=band["val_seed_range"])
        rng = np.random.RandomState(seed + band_seed_offset(sb) + (0 if kind == "training" else 1))
        fair = band.get("snr_dist") == "fair"
        if mode == "range" and fair and kind == "training":     # v26: one SNR per column, spread evenly
            snr = fair_snr(rng, float(v[0]), float(v[1]), x.shape[1], band.get("snr_step"))
            y = gen_measurements_percol(x, A, snr, rng, noise_model=nm)
            sio.savemat(out, {"y": y, "snr_db": snr[None, :], "noise_model": nm})
            cnt, _ = np.histogram(snr, bins=20, range=(float(v[0]), float(v[1]) + 1e-9))
            Path(str(out) + ".snr.json").write_text(json.dumps(dict(
                snr_dist="fair", snr_step=band.get("snr_step"), range=[float(v[0]), float(v[1])], K=int(x.shape[1]),
                distinct=int(np.unique(snr).size), min=float(snr.min()), max=float(snr.max()), mean=float(snr.mean()),
                counts_20_equal_bins=cnt.tolist(), seed=seed, noise_model=nm)), encoding="utf-8")
            written.append(out)
            if verbose:
                how = (f"on the grid {float(v[0]):g}, {float(v[0]) + band['snr_step']:g}, ... dB, equal counts"
                       if band.get("snr_step") else "spread evenly (stratified)")
                print(f"wrote  {out}  ({x.shape[1]:,} columns, one SNR each in [{snr.min():.2f}, {snr.max():.2f}] dB, "
                      f"{how}, {nm} noise)")
            continue
        if mode == "range":
            n = int(band.get("val_n_snr_chunks", chunks)) if kind == "validation" else chunks
            while x.shape[1] % n:          # the original needs K divisible by the number of chunks
                n -= 1
            snr = evenly_spaced_snr(float(v[0]), float(v[1]), n) if fair else draw_snr_list(rng, float(v[0]), float(v[1]), n)
        else:
            snr = np.asarray(v, float)
            if x.shape[1] % len(snr):
                raise ValueError(f"{kind}: {x.shape[1]} columns not divisible by {len(snr)} SNR points")
        y = gen_measurements_py(x, A, snr, rng, noise_model=nm)
        per_col = np.repeat(snr, x.shape[1] // len(snr))[None, :]
        sio.savemat(out, {"y": y, "snr_db": per_col, "noise_model": nm})
        Path(str(out) + ".snr.json").write_text(json.dumps(dict(snr_db=np.asarray(snr).tolist(), K=int(x.shape[1]),
                                                               chunks=len(snr), seed=seed, noise_model=nm)),
                                                encoding="utf-8")
        written.append(out)
        if verbose:
            print(f"wrote  {out}  ({x.shape[1]:,} columns, {len(snr)} SNR chunks in "
                  f"[{np.min(snr):.2f}, {np.max(snr):.2f}] dB, {nm} noise)")
    return written


def read_snr_db(path, K: int) -> Optional[np.ndarray]:
    """SNR of each of the K columns from 'snr_db' (per column: ensure_band_files; per chunk: generate_data.py,
    genMeasurements chunk order), or None (files of the original scripts, which do not store it)."""
    d = sio.loadmat(path, variable_names=["snr_db"])
    if "snr_db" not in d:
        return None
    s = np.asarray(d["snr_db"], float).ravel()
    if s.size == K:
        return s
    if s.size and K % s.size == 0:
        return np.repeat(s, K // s.size)
    raise ValueError(f"{path}: snr_db has {s.size} values for {K} columns")


def points_validation(x: np.ndarray, A: np.ndarray, snr_db, seed: int, noise_model=None):
    """Fixed validation set at SNR points (equal chunks, as genMeasurements): -> y, s2 per column, snr per column."""
    snr = np.asarray(snr_db, float)
    if x.shape[1] % snr.size:
        raise ValueError(f"{x.shape[1]} validation columns not divisible by {snr.size} SNR points")
    y = gen_measurements_py(x, A, snr, np.random.RandomState(seed), noise_model=noise_model)
    per = np.repeat(snr, x.shape[1] // snr.size)
    return y, 10.0 ** (-per / 10), per


def fair_validation(x: np.ndarray, A: np.ndarray, lo: float, hi: float, seed: int, noise_model=None, n: int = 100):
    """Fixed validation set of --snr-dist fair: n equal chunks at the equally spaced SNRs evenly_spaced_snr(lo, hi, n)
    (n reduced until it divides the number of columns) -> y, s2 per column, snr per column."""
    n = int(n)
    while x.shape[1] % n:
        n -= 1
    return points_validation(x, A, evenly_spaced_snr(lo, hi, n), seed, noise_model=noise_model)


def online_validation(x: np.ndarray, A: np.ndarray, lo: float, hi: float, seed: int, noise_model=None):
    """Fixed validation set for online-noise training: y = A (x + n), SNR of every column ~ U(lo, hi)
    (noise_model 'white': the same draw whitened, y = A x + T A n)."""
    rng = np.random.RandomState(seed)
    N, K = x.shape
    snr = rng.uniform(lo, hi, size=K)
    s2 = 10.0 ** (-snr / 10)
    n = np.sqrt(s2 / 2)[None, :] * (rng.normal(size=(N, K)) + 1j * rng.normal(size=(N, K)))
    return noise_models.apply(A @ (x + n), x, A, noise_model), s2, snr

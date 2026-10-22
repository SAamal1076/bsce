"""Data options of train.py / run_suite.py (bsce.data.files.apply_band_options + ensure_band_files):
--margin (train a band on [lo-M, hi+M], use it on [lo, hi]), --val-snr range|<SNRs> (validation over the band or at
fixed SNRs, as the authors), --snr-chunks. Synthetic data, no reference/ needed.

1. no option -> the bands are unchanged (same names, same files)
2. margin: training SNRs inside [lo-M, hi+M], the band keeps its name and use range, the validation file is
   bit-identical to the one without margin; a 100-chunk margin file equals the --snr file of the wider range
3. --val-snr range on the authors' bands (validation at 10 dB) -> a new file with SNR ~ U(band); --val-snr 10 on a
   range band -> a new file at 10 dB; the existing files are never overwritten
4. --snr-chunks changes the training file only
"""
import sys
from pathlib import Path

import numpy as np
import scipy.io as sio

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from bsce.config import band_file_names  # noqa: E402
from bsce.data.files import apply_band_options, bands_from_edges, ensure_band_files  # noqa: E402

SC = dict(name="T", dataset="SV", array="UPA", M=16, n_snr_chunks=10)
AUTHORS = [dict(name="0to10dB", train=[0, 10], val_snr=[10], use_for=[0, 10]),
           dict(name="10to20dB", train=[10, 20], val_snr=[10], use_for=[10, 20.01])]


def _data(tmp):
    r = np.random.default_rng(0)
    A = r.choice([-1.0, 1.0], (16, 32)) / 4
    for part, K in (("training", 1000), ("validation", 200)):
        x = (r.standard_normal((32, K)) + 1j * r.standard_normal((32, K))) / np.sqrt(2)
        sio.savemat(tmp / f"SV_UPA_{part}_channel.mat", {"x": x})
    return A


def _make(tmp, A, band):
    names = band_file_names(SC, band)
    ensure_band_files(SC, band, 0, tmp, A, names, verbose=False)
    return [sio.loadmat(tmp / n) for n in names], names


def _chunk_snr(f):
    return np.unique(np.round(np.asarray(f["snr_db"], float).ravel(), 9))


def test_no_option_is_identity():
    for bands in (AUTHORS, bands_from_edges([0, 10, 20])):
        assert apply_band_options(bands, default_chunks=10) == [dict(b) for b in bands]


def test_margin(tmp_path):
    A = _data(tmp_path)
    base = bands_from_edges([0, 10, 20])
    (tr0, va0), _ = _make(tmp_path, A, base[0])
    b = apply_band_options(base, margin=2, default_chunks=10)[0]
    assert b["name"] == "0to10dB" and b["use_range"] == [0.0, 10.0] and b["val_range"] == [0.0, 10.0]
    (tr, va), names = _make(tmp_path, A, b)
    s = _chunk_snr(tr)
    assert s.min() >= -2 and s.max() <= 12 and (s.min() < 0 or s.max() > 10)       # the margins are used
    assert names[1] == band_file_names(SC, base[0])[1] and np.array_equal(va["y"], va0["y"])
    other = tmp_path / "other"                     # a separate folder with the same channels
    other.mkdir()
    _data(other)
    (trw, _), _ = _make(other, A, bands_from_edges([-2, 12])[0])
    assert np.array_equal(tr["y"], trw["y"])                                        # = the --snr -2 12 file
    a = apply_band_options(AUTHORS, margin=2, default_chunks=10)[0]
    assert band_file_names(SC, a)[0].endswith("m2to12dB_c10.mat") and a["val_snr"] == [10]


def test_validation_options(tmp_path):
    A = _data(tmp_path)
    r = apply_band_options(AUTHORS, val="range", default_chunks=10)[0]
    (_, va), names = _make(tmp_path, A, r)
    s = _chunk_snr(va)
    assert names[1].endswith("0to10dB_valrange.mat") and s.size == 100 and s.min() >= 0 and s.max() <= 10
    p = apply_band_options(bands_from_edges([0, 10, 20]), val=[10], default_chunks=10)[1]
    (_, va), names = _make(tmp_path, A, p)
    assert names[1].endswith("10to20dB_val10dB.mat") and np.allclose(_chunk_snr(va), [10])
    assert apply_band_options(AUTHORS, val=[10], default_chunks=10)[0]["val_snr"] == [10.0]   # = the authors' file
    assert band_file_names(SC, apply_band_options(AUTHORS, val=[10], default_chunks=10)[0]) == band_file_names(SC, AUTHORS[0])


def test_chunks_change_the_training_file_only(tmp_path):
    A = _data(tmp_path)
    base = bands_from_edges([0, 10, 20])[0]
    (_, va0), n0 = _make(tmp_path, A, base)
    c = apply_band_options([base], chunks=10, default_chunks=10)[0]
    (tr, va), n = _make(tmp_path, A, c)
    assert n[0].endswith("0to10dB_c10.mat") and _chunk_snr(tr).size == 10
    assert n[1] == n0[1] and np.array_equal(va["y"], va0["y"])

"""Measurements y = A (x + n), n ~ CN(0, sigma2 I_N), sigma2 = 1/SNR.

noise_model='white' (bsce/data/noise.py): y = A x + T (y_colored - A x), the same random draw whitened
(white noise of the same mean power; the colored output is unchanged when noise_model is None / 'colored').

Two ports, because the originals use two different noise routines:

* ``gen_measurements_py``  - generate_measurement_data.py (training/validation sets):
  the K columns are split into ``len(snr_db_list)`` equal chunks, chunk k uses SNR
  snr_db_list[k]; noise = sqrt(sigma)*(normal(N,bs) + 1j*normal(N,bs))/sqrt(2).
  Pass ``np.random.RandomState(seed)`` to reproduce the original's np.random.* calls.
* ``gen_measurements_matlab`` - the test scripts (Main.m / Mian_*_full.m):
  noise = sqrt(sigma2)*(randn(N,K)+1i*randn(N,K))/sqrt(2); y = A*(x+noise).
* ``gen_measurements_percol`` (v26, --snr-dist fair): one SNR per column (``fair_snr``: spread evenly over the
  range, in random order) instead of a few chunks.
"""
from __future__ import annotations

import warnings

import numpy as np

from ..rng import MatlabRNG
from . import noise as noise_models


def draw_snr_list(np_rng, low_db: float, high_db: float, n_chunks: int) -> np.ndarray:
    """SNR_dB = np.random.uniform(n1, n2, size=(n_chunks))  (as in the originals)."""
    return np_rng.uniform(low_db, high_db, size=(n_chunks))


def fair_snr(np_rng, low_db: float, high_db: float, K: int, step=None) -> np.ndarray:
    """K SNRs (dB) spread evenly over [low_db, high_db], in random order (--snr-dist fair).

    step None: stratified: the range is cut into K equal parts and every part gets exactly one SNR, uniform inside
        it. The values are continuous and their histogram is flat at every resolution down to (high - low) / K.
    step (dB): the grid low, low + step, ... (up to high), every grid value the same number of times (the counts
        differ by at most one when the grid does not divide K)."""
    lo, hi, K = float(low_db), float(high_db), int(K)
    if not hi > lo:
        raise ValueError(f"fair SNR draw needs low < high (got {lo}, {hi})")
    if step:
        step = float(step)
        n = int(np.floor((hi - lo) / step + 1e-9)) + 1
        grid = lo + step * np.arange(n)
        reps = np.full(n, K // n)
        if K % n:
            reps[np_rng.choice(n, K % n, replace=False)] += 1
        v = np.repeat(grid, reps)
    else:
        v = lo + (hi - lo) * (np.arange(K) + np_rng.uniform(size=K)) / K
    return v[np_rng.permutation(K)]


def evenly_spaced_snr(low_db: float, high_db: float, n: int) -> np.ndarray:
    """n SNRs at the middles of n equal parts of [low, high]: the validation SNRs of --snr-dist fair (equal chunks)."""
    return float(low_db) + (float(high_db) - float(low_db)) * (np.arange(int(n)) + 0.5) / int(n)


def gen_measurements_percol(x: np.ndarray, A: np.ndarray, snr_db, np_rng, noise_model=None,
                            block: int = 4000) -> np.ndarray:
    """y = A (x + n), column k with its own SNR snr_db[k]; the noise of column k is the k-th pair of N normal draws
    (real, imaginary) of np_rng, so the result does not depend on `block` (only the memory used)."""
    N, K = x.shape
    snr = np.asarray(snr_db, float).ravel()
    if snr.size != K:
        raise ValueError(f"{snr.size} SNRs for {K} columns")
    y = np.zeros((A.shape[0], K), dtype=np.complex128)
    T = noise_models.whitening_map(A) if noise_models.canonical(noise_model) == "white" else None
    for s in range(0, K, block):
        e = min(K, s + block)
        sig = np.sqrt(10.0 ** (-snr[s:e] / 10.0) / 2.0)[None, :]
        z = np_rng.normal(size=(e - s, 2, N))           # column by column (real, then imaginary part): the draw does
        noise = sig * (z[:, 0, :] + 1j * z[:, 1, :]).T  # not depend on the block size
        y[:, s:e] = noise_models.apply(A @ (x[:, s:e] + noise), x[:, s:e], A, noise_model, T=T)
    return y


def gen_measurements_py(x: np.ndarray, A: np.ndarray, snr_db_list, np_rng,
                        strict: bool = True, noise_model=None) -> np.ndarray:
    """Port of genMeasurements(x, A, SNR_all) from generate_measurement_data.py.

    Note (kept from the original): if K is not divisible by the number of chunks the
    trailing K % n_chunks columns stay y = 0. With strict=True this raises instead.
    """
    N, K = x.shape
    M = A.shape[0]
    n_chunks = len(snr_db_list)
    bs = int(K / n_chunks)
    if K % n_chunks:
        msg = (f"K={K} not divisible by {n_chunks} SNR chunks: the original leaves the last "
               f"{K % n_chunks} measurement columns equal to zero")
        if strict:
            raise ValueError(msg)
        warnings.warn(msg)
    y = np.zeros((M, K), dtype=np.complex128)
    for k in range(n_chunks):
        xk = x[..., k * bs:(k + 1) * bs]
        snr = np.power(10, snr_db_list[k] / 10.)
        sigma = 1 / snr
        noise = np.sqrt(sigma) * ((np_rng.normal(size=(N, bs)) + 1j * np_rng.normal(size=(N, bs))) / np.sqrt(2))
        y[..., k * bs:(k + 1) * bs] = np.matmul(A, xk + noise)
    if noise_models.canonical(noise_model) == "white":
        n_used = bs * n_chunks                          # trailing columns of a non-divisible K stay 0 (original)
        y[:, :n_used] = noise_models.apply(y[:, :n_used], x[:, :n_used], A, "white")
    return y


def gen_measurements_matlab(x: np.ndarray, A: np.ndarray, snr_db: float, rng: MatlabRNG,
                            return_noise: bool = False, noise_model=None):
    """sigma2=1/SNR; noise=sqrt(sigma2)*(randn(N,K)+1i*randn(N,K))/sqrt(2); y=A*(x+noise).
    noise_model='white': y = A x + T A noise (the same draw whitened; return_noise still returns the N x K draw)."""
    N, K = x.shape
    sigma2 = 1 / (10. ** (snr_db / 10.))
    re = rng.randn(N, K)
    im = rng.randn(N, K)
    noise = np.sqrt(sigma2) * (re + 1j * im) / np.sqrt(2)
    y = A @ (x + noise)
    y = noise_models.apply(y, x, A, noise_model)
    return (y, noise) if return_noise else y


def snr_bin_index(snr_db: float, edges) -> int:
    """Index of the SNR bin used to pick a trained network (edges e.g. [10] -> 2 bins).

    Matches the MATLAB tests: bin i covers [edges[i-1], edges[i]) (left-closed).
    """
    idx = 0
    for e in edges:
        if snr_db >= e:
            idx += 1
    return idx

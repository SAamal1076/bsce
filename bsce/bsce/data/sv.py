"""Saleh-Valenzuela beamspace channels - line-by-line port of the MATLAB generator.

Originals (Wei, Hu & Dai code, also used unchanged in the LADMM project):
  generate_channel.m, generate_channel_data_for_SV_channel_model.m, Main.m (DFT part)

Arithmetic is written in the same left-to-right order as MATLAB so results agree to
machine precision (verified in tests/test_data_vs_octave.py).

Conventions kept from the originals:
  * ULA steering uses exp(+j...), UPA uses exp(-j...)          (as in generate_channel.m)
  * h = sqrt(N/L) * sum_l alpha_l a_l, alpha ~ CN(0,1), angles ~ U(-pi/2, pi/2)
  * beamspace x = U.' * h  (non-conjugate transpose), U = DFT matrix of the lens array
"""
from __future__ import annotations

import numpy as np

from ..rng import MatlabRNG


def _sym_grid(n: int) -> np.ndarray:
    """MATLAB [-(n-1):2:n-1] as float column values."""
    return np.arange(-(n - 1), n, 2, dtype=np.float64)


def dft_matrix_1d(n: int, d: float = 0.5) -> np.ndarray:
    """UN = (1/sqrt(N))*exp(1i*2*pi*[-(N-1):2:N-1]'/2*d*[-(N-1):2:N-1]*(1/N))."""
    g = _sym_grid(n)
    col = (2 * np.pi * g) / 2 * d                      # 2*pi*[..]'/2*d   (left to right)
    phase = np.outer(col, g) * (1.0 / n)               # (...)*[..]*(1/N)
    return (1.0 / np.sqrt(n)) * np.exp(1j * phase)


def dft_matrix(array: str, N1: int = 16, N2: int = 16, d: float = 0.5) -> np.ndarray:
    """Lens-array DFT matrix: ULA with N = N1*N2 elements, or UPA kron(U_N1, U_N2)."""
    array = array.upper()
    if array == "ULA":
        return dft_matrix_1d(N1 * N2, d)
    if array == "UPA":
        return np.kron(dft_matrix_1d(N1, d), dft_matrix_1d(N2, d))
    raise ValueError("array must be 'ULA' or 'UPA'")


def draw_channel_params(rng: MatlabRNG, K: int, L: int, array: str):
    """Draw (alpha, phi1, phi2) for K channels in exactly the MATLAB call order.

    Per channel generate_channel.m calls: normrnd(0,1,L,1), normrnd(0,1,L,1),
    rand(1,L) [ULA: phi]   or   rand(1,L), rand(1,L) [UPA: phi1, phi2].
    Returns arrays of shape (L, K) (phi2 is None for the ULA).
    """
    upa = array.upper() == "UPA"
    alpha = np.empty((L, K), dtype=np.complex128)
    phi1 = np.empty((L, K))
    phi2 = np.empty((L, K)) if upa else None
    for k in range(K):
        re = rng.normrnd(0, 1, L, 1)[:, 0]
        im = rng.normrnd(0, 1, L, 1)[:, 0]
        alpha[:, k] = (re + 1j * im) / np.sqrt(2)
        phi1[:, k] = np.pi * rng.rand(1, L)[0] - np.pi / 2
        if upa:
            phi2[:, k] = np.pi * rng.rand(1, L)[0] - np.pi / 2
    return alpha, phi1, phi2


def spatial_channels(alpha, phi1, phi2, array: str, N1: int = 16, N2: int = 16,
                     d: float = 0.5) -> np.ndarray:
    """Spatial channels h (N x K) from drawn parameters (port of generate_channel.m)."""
    L, K = alpha.shape
    N = N1 * N2
    H = np.zeros((N, K), dtype=np.complex128)
    if array.upper() == "ULA":
        g = _sym_grid(N)[:, None]                                  # (N,1)
        for l in range(L):
            # a = 1/sqrt(N)*exp(1i*2*pi*[-(N-1):2:N-1]'/2*d*sin(phi(l)))
            a = 1 / np.sqrt(N) * np.exp(1j * ((2 * np.pi * g) / 2 * d * np.sin(phi1[l])[None, :]))
            H = H + alpha[l][None, :] * a
    else:
        g1 = _sym_grid(N1)[:, None]
        g2 = _sym_grid(N2)[:, None]
        for l in range(L):
            s1 = np.sin(phi1[l]) * np.sin(phi2[l])                  # sin(phi1(l))*sin(phi2(l))
            a1 = 1 / np.sqrt(N1) * np.exp(-1j * ((2 * np.pi * g1) / 2 * d * s1[None, :]))
            a2 = 1 / np.sqrt(N2) * np.exp(-1j * ((2 * np.pi * g2) / 2 * d * np.cos(phi2[l])[None, :]))
            # kron(a1, a2) for column vectors, done per channel (column)
            a = (a1[:, None, :] * a2[None, :, :]).reshape(N1 * N2, K)
            H = H + alpha[l][None, :] * a
    return np.sqrt(N / L) * H


def beamspace_channels(rng: MatlabRNG, K: int, array: str = "ULA", L: int = 3,
                       N1: int = 16, N2: int = 16, d: float = 0.5,
                       return_spatial: bool = False):
    """x(:,k) = UN.' * generate_channel(N1,N2,L,type), k = 1..K  (N x K complex)."""
    alpha, phi1, phi2 = draw_channel_params(rng, K, L, array)
    H = spatial_channels(alpha, phi1, phi2, array, N1, N2, d)
    UN = dft_matrix(array, N1, N2, d)
    X = UN.T @ H
    return (X, H) if return_spatial else X

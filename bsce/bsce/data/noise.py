"""Noise models of the measurements: 'colored' (noise folding, GM-LAMP's model, bsce's default) and 'white'.

  colored (aliases 'coloured', 'folded'): y = A (x + n),  n ~ CN(0, s2 I_N)   -> noise covariance s2 A A^T
  white                                 : y = A x + w,    w ~ CN(0, s2 c I_M), c = tr(A A^T) / M

s2 = 10^(-SNR/10) in both. At the same SNR the two have the same signal A x and the same mean noise power at the
measurements, E||w||^2 = E||A n||^2 = s2 tr(A A^T); only the colour of the noise differs (for GM-LAMP's
+-1/sqrt(M) combiner c = N/M = 2 and the eigenvalues of A A^T spread over about 0.2 .. 5.6).

bsce makes white noise from the SAME random draw as the colored noise:
    w = T A n,   T = sqrt(c) (A A^T)^(-1/2)   (M x M),   so   y_white = A x + T (y_colored - A x).
T A A^T T^T = c I, so w is exactly white Gaussian with variance s2 c. Consequences:
  * a white data set is the colored one with its noise whitened: every white file, validation set and test set is
    PAIRED with its colored twin (same channels, same random numbers), so a difference between the two is due to the
    noise colour only, not to another noise realisation;
  * every colored file stays bit-identical to the files made before this option existed (the colored path is
    untouched; white is a transform applied afterwards).
The white model here fixes the MEAN SNR (as the colored model); UTRR's paper instead fixes ||A x||^2 / ||w||^2 for every
channel (scripts/utrr_reproduce.py keeps that definition for its Table II reproduction).
"""
from __future__ import annotations

from typing import Optional

import numpy as np

MODELS = ("colored", "white")
_ALIASES = {"colored": "colored", "coloured": "colored", "folded": "colored", "colour": "colored", "color": "colored",
            "white": "white"}


def canonical(name: Optional[str]) -> str:
    """'colored' / 'white' from any accepted spelling (None -> 'colored')."""
    if name is None:
        return "colored"
    k = str(name).strip().lower()
    if k not in _ALIASES:
        raise ValueError(f"unknown noise model '{name}': use colored (= coloured, folded) or white")
    return _ALIASES[k]


def of_scenario(sc: dict) -> str:
    """The noise model of a scenario dict (key 'noise_model', default colored)."""
    return canonical(sc.get("noise_model"))


def suffix(model: Optional[str]) -> str:
    """'' for colored (all names as before), '_white' for white: measurement files and run folders."""
    return "" if canonical(model) == "colored" else "_white"


def white_scale(A) -> float:
    """c = tr(A A^T) / M: the white noise variance is c s2 (same mean noise power as the colored model)."""
    A = np.asarray(A, dtype=np.float64)
    return float(np.sum(A * A) / A.shape[0])


def whitening_map(A) -> np.ndarray:
    """T = sqrt(c) (A A^T)^(-1/2) (M x M, real symmetric): w = T A n is white with variance s2 c."""
    A = np.asarray(A, dtype=np.float64)
    lam, U = np.linalg.eigh(A @ A.T)
    if lam.min() <= 0:
        raise ValueError("A A^T is singular: the white noise model needs a full-row-rank combiner")
    return np.sqrt(white_scale(A)) * (U * (1.0 / np.sqrt(lam))) @ U.T


def apply(y_colored, x, A, model: Optional[str], T=None):
    """Measurements of `model` from the colored measurements y_colored = A (x + n) of the same draw n.
    colored: y_colored itself (unchanged object); white: A x + T (y_colored - A x).
    numpy arrays or torch tensors (columns are samples: y (M, K), x (N, K)); T may be given (whitening_map(A))."""
    if canonical(model) == "colored":
        return y_colored
    try:
        import torch
        is_t = torch.is_tensor(y_colored)
    except ImportError:                                 # pragma: no cover
        is_t = False
    if is_t:
        import torch
        rd = y_colored.real.dtype
        At = torch.as_tensor(np.asarray(A, np.float64) if not torch.is_tensor(A) else A).to(y_colored.device, rd)
        Tt = torch.as_tensor(whitening_map(A.detach().cpu().numpy() if torch.is_tensor(A) else A) if T is None else T)
        Tt = Tt.to(y_colored.device, rd)
        xt = x.to(y_colored.dtype)
        sx = torch.complex(At @ xt.real, At @ xt.imag)
        d = y_colored - sx
        return sx + torch.complex(Tt @ d.real, Tt @ d.imag)
    A = np.asarray(A, dtype=np.float64)
    T = whitening_map(A) if T is None else np.asarray(T)
    sx = A @ x
    return sx + T @ (np.asarray(y_colored) - sx)

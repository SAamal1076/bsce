"""Polish: a least-squares correction on the strongest beams, after any trained network (v28, evaluation only).

    r   = W (y - A z)                  residual, W = (A A^T)^-1/2 for the folded noise y = A(x + n), W = I for white
    S   = the K largest |z_i| of each channel
    d_S = (B_S^T B_S + eps I)^-1 B_S^T r,  B = W A
    z+  = z + d on S; the other beams are unchanged

It removes the shrinkage bias on the few strong beams (soft thresholding shrinks every surviving beam by theta/rho)
without retraining. eps pulls the correction towards 0 (eps -> inf: no change). K and eps are chosen on validation
channels, never on the test set: per band (the network's SNR band; the same knowledge the band networks already
use), or per SNR (the true SNR is known: label it so). K = 0 (no polish) is always a candidate, so on the
validation data the polish never makes a network worse.
Idea from the pilot E13 (UTRR's trimmed ridge, LISTA-CP's support selection); the code here is new.
"""
from __future__ import annotations

from typing import Dict, Iterable, List, Sequence, Tuple

import numpy as np

K_GRID = (0, 4, 8, 12, 16, 24, 32)
EPS_GRID = (1e-3, 1e-2, 3e-2, 1e-1, 3e-1)


def whitener(A: np.ndarray, noise_model: str) -> np.ndarray:
    """W = (A A^T)^-1/2 for the colored (folded) noise, I for white noise."""
    if noise_model == "white":
        return np.eye(A.shape[0])
    ev, U = np.linalg.eigh(A @ A.T)
    return (U / np.sqrt(ev)) @ U.T


class Polisher:
    """Precomputes B = W A for one combiner and noise model; z, y are columns (N, n) and (M, n), complex."""

    def __init__(self, A: np.ndarray, noise_model: str = "colored"):
        self.A = np.asarray(A, np.float64)
        self.noise_model = noise_model
        self.W = whitener(self.A, noise_model)
        self.B = self.W @ self.A                                  # (M, N)

    def _parts(self, z: np.ndarray, y: np.ndarray, K: int):
        r = self.W @ y - self.B @ z                                # (M, n) complex
        idx = np.argsort(-np.abs(z), axis=0)[:K].T                 # (n, K)
        BS = self.B.T[idx]                                         # (n, K, M) real
        G = np.einsum("nkm,nlm->nkl", BS, BS)
        rhs = np.einsum("nkm,mn->nk", BS, r)                       # (n, K) complex
        return idx, G, rhs

    @staticmethod
    def _apply(z, idx, G, rhs, eps):
        K = idx.shape[1]
        d = np.linalg.solve(G + eps * np.eye(K)[None], rhs[..., None])[..., 0]   # real G, complex rhs
        out = z.copy()
        cols = np.arange(z.shape[1])[:, None]
        out[idx, cols] += d
        return out

    def polish(self, z: np.ndarray, y: np.ndarray, K: int, eps: float) -> np.ndarray:
        if K <= 0:
            return z
        return self._apply(z, *self._parts(z, y, int(K)), float(eps))

    def grid(self, z: np.ndarray, y: np.ndarray, x: np.ndarray, Ks: Sequence[int] = K_GRID,
             epss: Sequence[float] = EPS_GRID) -> Dict[Tuple[int, float], float]:
        """NMSE (dB) of every (K, eps) on one set of channels (K = 0: the network itself)."""
        out = {(0, 0.0): nmse_db(z, x)}
        for K in Ks:
            if K <= 0:
                continue
            parts = self._parts(z, y, int(K))
            for e in epss:
                out[(int(K), float(e))] = nmse_db(self._apply(z, *parts, float(e)), x)
        return out


def nmse_db(xh, x):
    return float(10 * np.log10(np.mean(np.sum(np.abs(xh - x) ** 2, 0)) / np.mean(np.sum(np.abs(x) ** 2, 0))))


def choose(grids: Iterable[Dict[Tuple[int, float], float]]) -> Tuple[Tuple[int, float], float, float]:
    """The (K, eps) with the lowest mean NMSE (dB) over the given validation grids (one per SNR).
    Returns ((K, eps), mean NMSE with it, mean NMSE without polish)."""
    grids = list(grids)
    keys = list(grids[0])
    mean = {k: float(np.mean([g[k] for g in grids])) for k in keys}
    best = min(keys, key=lambda k: (mean[k], k[0]))               # ties: the smaller K
    return best, mean[best], mean[(0, 0.0)]


def band_val_snrs(lo: float, hi: float, n: int = 5) -> List[float]:
    """n equally spaced validation SNRs over a band [lo, hi] (the ends included)."""
    return [float(v) for v in np.linspace(lo, hi, n)]

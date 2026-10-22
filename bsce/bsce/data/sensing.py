"""Compressed-sensing (beam-selection) matrix - port of generate_CS_matrix.m.

    A = ((rand(M,N)>0.5)*2-1)/sqrt(M);
"""
from __future__ import annotations

import numpy as np

from ..rng import MatlabRNG


def cs_matrix(rng: MatlabRNG, M: int = 128, N: int = 256) -> np.ndarray:
    return ((rng.rand(M, N) > 0.5) * 2 - 1) / np.sqrt(M)

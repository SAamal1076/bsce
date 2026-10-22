"""Shared pieces for the unrolled networks.

Conventions (same as the original TF / MATLAB code):
  * signals are columns: y is (M, batch), x is (N, batch), complex;
  * the sensing matrix A is real (M, N); learned linear maps are real and are applied to
    the real and imaginary parts separately (``rmm``), exactly like the TF graphs;
  * every model exposes ``stage_plan()`` (the layer-wise training plan of the original
    code) and ``forward_stages(y, upto)`` (the intermediate outputs those stages train).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional

import numpy as np
import torch
import torch.nn as nn

from ..device import dtypes


def rmm(W: torch.Tensor, z: torch.Tensor) -> torch.Tensor:
    """Real matrix times complex matrix, real/imag split (as tf.complex(W@re, W@im))."""
    return torch.complex(W @ z.real, W @ z.imag)


@dataclass
class Stage:
    """One entry of the original `layers` list.

    name     : stage name used by the original code (e.g. 'LAMP-gm linear T=2')
    kind     : 'linear' (flag (0,)) or 'nonlinear' (flag (1,))
    new      : names of the parameters introduced (trained alone in the first sub-stage)
    all      : names of all parameters existing at this point (refinement sub-stages)
    """
    name: str
    kind: str
    new: List[str]
    all: List[str]
    index: int = 0


class UnrolledNet(nn.Module):
    """Base class. Subclasses register parameters with ``self._p(name, value)``."""

    family = "base"
    needs_noise_var = False      # True: forward(y, s2=...) needs the noise variance of every column (NF-LADMM)

    def __init__(self, A: np.ndarray, precision: str = "float64", device=None):
        super().__init__()
        self.rdtype, self.cdtype = dtypes(precision)
        self.precision = precision
        A = np.asarray(A, dtype=np.float64)
        self.M, self.N = A.shape
        self.register_buffer("A", torch.tensor(A, dtype=self.rdtype))
        self.A_np = A
        self._param_order: List[str] = []
        self.cfg: Dict = {}
        if device is not None:
            self.to(device)

    # -- parameter bookkeeping (keeps the original TF variable order) --------------
    def _p(self, name: str, value) -> nn.Parameter:
        t = torch.as_tensor(np.asarray(value), dtype=self.rdtype).clone()
        p = nn.Parameter(t)
        self.register_parameter(name, p)
        self._param_order.append(name)
        return p

    def P(self, name: str) -> nn.Parameter:
        return getattr(self, name)

    def _put_buffer(self, name: str, value) -> None:
        """Set buffer `name` to `value` (same dtype and device). A new shape (another combiner size M, v30) registers
        the buffer again, so the state_dict, a saved checkpoint and a reloaded network all have the new shape."""
        old = getattr(self, name)
        t = torch.as_tensor(np.asarray(value), dtype=old.dtype).to(old.device)
        if tuple(old.shape) == tuple(t.shape):
            old.copy_(t)
        else:                                          # v31: a non-persistent buffer stays non-persistent
            self.register_buffer(name, t, persistent=name not in self._non_persistent_buffers_set)

    def _combiner_array(self, A) -> np.ndarray:
        """v31: a new combiner as float64 (M' x N): any M', the same N, independent rows; else ValueError."""
        A = np.asarray(A.detach().cpu().numpy() if torch.is_tensor(A) else A, np.float64)
        if A.ndim != 2 or A.shape[1] != self.N:
            raise ValueError(f"combiner of shape {A.shape}: it must have N = {self.N} columns (beams)")
        if np.linalg.matrix_rank(A) < A.shape[0]:
            raise ValueError(f"combiner of shape {A.shape} has dependent rows (A A^T is singular)")
        return A

    def param_names(self) -> List[str]:
        return list(self._param_order)

    # -- interface -------------------------------------------------------------
    def stage_plan(self) -> List[Stage]:
        raise NotImplementedError

    def forward_stages(self, y: torch.Tensor, upto: Optional[int] = None) -> List[torch.Tensor]:
        """Outputs of stages 0..upto (inclusive); upto=None -> all stages."""
        raise NotImplementedError

    def forward(self, y: torch.Tensor) -> torch.Tensor:
        return self.forward_stages(y)[-1]

    def as_complex(self, y) -> torch.Tensor:
        if not torch.is_tensor(y):
            y = torch.as_tensor(np.asarray(y))
        return y.to(device=self.A.device, dtype=self.cdtype)

    # -- export ----------------------------------------------------------------
    def mat_dict(self) -> Dict[str, np.ndarray]:
        """Parameters under MATLAB-friendly names (overridden where the originals differ)."""
        return {n: self.P(n).detach().cpu().numpy() for n in self._param_order}

    def load_mat_dict(self, d: Dict[str, np.ndarray]) -> None:
        with torch.no_grad():
            for n in self._param_order:
                v = np.asarray(d[n], dtype=np.float64).reshape(tuple(self.P(n).shape))
                self.P(n).copy_(torch.as_tensor(v, dtype=self.rdtype))

    def n_params(self) -> int:
        """Number of real trainable parameters (a complex entry counts as 2: ADMM-Net's complex W_t)."""
        return int(sum(p.numel() * (2 if p.is_complex() else 1) for p in self.parameters()))

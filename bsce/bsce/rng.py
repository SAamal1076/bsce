"""Random-number sources with MATLAB call semantics.

The MATLAB generator scripts draw ``rand(m,n)`` / ``randn(m,n)`` / ``normrnd(0,1,m,n)``
in a specific order and fill matrices column-major. Every Python routine that ports a
MATLAB script takes a ``MatlabRNG`` so that

* ``NumpyMatlabRNG(seed)`` gives reproducible Python runs, and
* ``ReplayRNG(u_stream, n_stream)`` replays exactly the numbers that an Octave/MATLAB
  shim (``tests/harness/octave_shims``) feeds to the original .m files, which is how the
  ports are verified to machine precision.

Routines that port *Python* originals (e.g. ``genMeasurements``) instead take any object
with numpy's ``normal(size=...)`` / ``uniform(low, high, size)`` methods
(``np.random.RandomState`` reproduces the originals' ``np.random.*`` global calls bit-for-bit).
"""
from __future__ import annotations

import numpy as np


class MatlabRNG:
    """Interface: column-major filled uniform(0,1) and standard-normal draws."""

    def rand(self, m: int, n: int) -> np.ndarray:  # pragma: no cover - interface
        raise NotImplementedError

    def randn(self, m: int, n: int) -> np.ndarray:  # pragma: no cover - interface
        raise NotImplementedError

    # MATLAB normrnd(mu, sigma, m, n) == mu + sigma*randn(m, n) drawn from the normal stream
    def normrnd(self, mu: float, sigma: float, m: int, n: int) -> np.ndarray:
        return mu + sigma * self.randn(m, n)


class NumpyMatlabRNG(MatlabRNG):
    """Reproducible Python source (PCG64). Not bit-identical to MATLAB's own rand."""

    def __init__(self, seed: int | None = 0):
        self.gen = np.random.default_rng(seed)

    def rand(self, m, n):
        return self.gen.random(m * n).reshape((m, n), order="F")

    def randn(self, m, n):
        return self.gen.standard_normal(m * n).reshape((m, n), order="F")


class ReplayRNG(MatlabRNG):
    """Replays pre-generated streams in call order (column-major fill, like MATLAB)."""

    def __init__(self, uniform_stream: np.ndarray, normal_stream: np.ndarray):
        self.u = np.asarray(uniform_stream, dtype=np.float64).ravel()
        self.n = np.asarray(normal_stream, dtype=np.float64).ravel()
        self.iu = 0
        self.inn = 0

    def _take(self, which, count):
        if which == "u":
            out = self.u[self.iu:self.iu + count]
            self.iu += count
        else:
            out = self.n[self.inn:self.inn + count]
            self.inn += count
        if out.size != count:
            raise RuntimeError("ReplayRNG stream exhausted")
        return out

    def rand(self, m, n):
        return self._take("u", m * n).reshape((m, n), order="F")

    def randn(self, m, n):
        return self._take("n", m * n).reshape((m, n), order="F")

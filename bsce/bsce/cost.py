"""Inference cost and trained-parameter count of a network (v31; used by scripts/compare.py).

* macs_per_channel(net): real multiply-accumulates per channel of the ATen ops that the forward really runs, counted by a
  dispatch mode: products (mm, addmm, bmm, mv, convolution; x4 for complex x complex, x2 for real x complex, where a complex
  tensor whose imaginary part is all zero, e.g. a real matrix cast to complex, counts as real) and dense factorisations / solves
  (LU n^3/3 + n^2 per right-hand side, inverse n^3, eigen / SVD / Cholesky 4 n^3; x4 if complex). Counted on the second
  of two calls, so a cache built in the first call (Graph-LADMM's H_t) is not counted, while a factorisation made in
  every call is. Element-wise work is counted apart (sum of output sizes of the other ops) as an indicator.
* trained_parameters(net): the real numbers of the parameters that reach the output, i.e. whose gradient of
  ||x_hat - x||^2 is not exactly zero on two random batches (low and high SNR). Structural zeros only: a parameter that
  never reaches the output (e.g. the last dual step of an unrolled ADMM) has a zero gradient for every input.
"""
from __future__ import annotations

import math

import numpy as np
import torch
from torch.utils._python_dispatch import TorchDispatchMode

_SKIP = ("view", "reshape", "expand", "unsqueeze", "squeeze", "permute", "transpose", "t", "select", "slice", "alias",
         "detach", "_to_copy", "clone", "copy_", "lift_fresh", "empty", "zeros", "ones", "full", "arange", "eye",
         "_unsafe_view", "as_strided", "real", "imag", "view_as_real", "view_as_complex", "conj", "_conj",
         "resolve_conj", "_neg_view", "resolve_neg", "zeros_like", "empty_like", "ones_like", "new_zeros", "new_empty",
         "scalar_tensor", "_local_scalar_dense", "item", "split", "unbind", "chunk", "index_select")


def _complex_factor(*ops) -> float:
    """Real multiplies per complex-arithmetic product of these operands: 4 if both carry an imaginary part, 2 if one."""
    n = sum(1 for t in ops if torch.is_tensor(t) and t.is_complex() and bool((t.imag != 0).any()))
    return (1.0, 2.0, 4.0)[min(n, 2)]


class MacCounter(TorchDispatchMode):
    def __init__(self):
        super().__init__()
        self.macs = 0.0
        self.elem = 0.0

    def __torch_dispatch__(self, func, types, args=(), kwargs=None):
        out = func(*args, **(kwargs or {}))
        name = func.overloadpacket.__name__
        f = 4.0 if any(torch.is_tensor(a) and a.is_complex() for a in args[:3]) else 1.0
        if name in ("mm", "bmm"):
            self.macs += _complex_factor(args[0], args[1]) * args[0].numel() * args[1].shape[-1]
        elif name in ("addmm", "baddbmm"):
            self.macs += _complex_factor(args[1], args[2]) * args[1].numel() * args[2].shape[-1]
        elif name == "mv":
            self.macs += _complex_factor(args[0], args[1]) * args[0].numel()
        elif name in ("convolution", "_convolution", "cudnn_convolution", "mkldnn_convolution"):
            o = out if torch.is_tensor(out) else out[0]
            self.macs += f * o.numel() * (args[1].numel() / args[1].shape[0])
        elif name in ("_linalg_solve_ex", "linalg_solve_ex", "linalg_solve"):
            Am, B = args[0], args[1]
            n = Am.shape[-1]
            nb = Am.numel() / (n * n)
            self.macs += f * nb * (n ** 3 / 3 + n * B.numel() / nb)
        elif name in ("linalg_lu_factor_ex", "linalg_lu_factor", "lu_factor"):
            n = args[0].shape[-1]
            self.macs += f * args[0].numel() / (n * n) * n ** 3 / 3
        elif name in ("linalg_lu_solve", "lu_solve"):
            B = args[2] if name == "linalg_lu_solve" else args[0]
            self.macs += f * B.shape[-2] * B.numel()
        elif name in ("linalg_inv_ex", "linalg_inv", "inverse"):
            n = args[0].shape[-1]
            self.macs += f * args[0].numel() / (n * n) * n ** 3
        elif name in ("linalg_eigh", "_linalg_eigh", "linalg_svd", "_linalg_svd", "cholesky", "linalg_cholesky_ex"):
            n = args[0].shape[-1]
            self.macs += f * args[0].numel() / (n * n) * 4 * n ** 3
        elif name not in _SKIP:
            for o in (out if isinstance(out, (tuple, list)) else (out,)):
                if torch.is_tensor(o) and o.dtype != torch.bool:
                    self.elem += o.numel() * (2 if o.is_complex() else 1)
        return out


def _batch(net, K: int, snr_db: float, seed: int):
    """Random test inputs: x = 8 strong complex Gaussian beams per channel + a weak dense floor (every beam carries some
    energy, so no parameter is left without gradient by chance), N x K; y = A (x + n)."""
    g = np.random.default_rng(seed)
    N = net.N
    x = 0.1 * (g.standard_normal((N, K)) + 1j * g.standard_normal((N, K)))
    for k in range(K):
        idx = g.choice(N, 8, replace=False)
        x[idx, k] += (g.standard_normal(8) + 1j * g.standard_normal(8)) * math.sqrt(N / 16)
    s2 = 10.0 ** (-snr_db / 10)
    n = math.sqrt(s2 / 2) * (g.standard_normal((N, K)) + 1j * g.standard_normal((N, K)))
    y = np.asarray(net.A_np) @ (x + n)
    return x, y, np.full(K, s2)


def _call(net, y, s2):
    yt = torch.as_tensor(y)
    return net(yt, s2=s2) if getattr(net, "needs_noise_var", False) else net(yt)


def macs_per_channel(net, batch: int = 1, seed: int = 0):
    """(MACs per channel, element-wise ops per channel) of one forward call with `batch` channels (second call)."""
    _, y, s2 = _batch(net, batch, 10.0, seed)
    with torch.no_grad():
        _call(net, y, s2)                                  # first call: builds any cache
        c = MacCounter()
        with c:
            _call(net, y, s2)
    return c.macs / batch, c.elem / batch


def trained_parameters(net, K: int = 128, seed: int = 0):
    """Real numbers of the parameters with a non-zero gradient on two random batches (0 and 20 dB); a complex entry
    counts 2, as UnrolledNet.n_params. Returns (trained, stored, names of the untrained ones)."""
    acc = {}
    was = net.training
    net.train()
    try:
        for snr, s in ((0.0, seed + 1), (20.0, seed + 2)):
            net.zero_grad(set_to_none=True)
            x, y, s2 = _batch(net, K, snr, s)
            out = _call(net, y, s2)
            (torch.abs(out - torch.as_tensor(x, dtype=out.dtype, device=out.device)) ** 2).sum().backward()
            for n_, p in net.named_parameters():
                g = (torch.zeros_like(p, dtype=torch.float64) if p.grad is None
                     else p.grad.detach().abs().to(torch.float64))
                acc[n_] = acc.get(n_, torch.zeros_like(g)) + g
    finally:
        net.zero_grad(set_to_none=True)
        net.train(was)
    stored = trained = 0
    dead = []
    for n_, p in net.named_parameters():
        k = 2 if p.is_complex() else 1
        stored += k * p.numel()
        nz = int((acc[n_] != 0).sum())
        trained += k * nz
        if nz < p.numel():
            dead.append(f"{n_} ({p.numel() - nz})")
    return trained, stored, dead

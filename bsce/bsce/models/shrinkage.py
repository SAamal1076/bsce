"""Shrinkage (denoising) functions, mirroring the original TF / MATLAB code op by op.

TensorFlow semantics are reproduced exactly, including where gradients flow:
``tf.maximum(x, c)`` passes the gradient to x where x >= c, and ``tf.maximum(c, x)``
passes it to x where x > c (ties go to the first argument). The helpers ``tfmax_xc``
and ``tfmax_cx`` implement these two cases.
"""
from __future__ import annotations

import math

import torch


def tfmax_xc(x: torch.Tensor, c) -> torch.Tensor:
    """tf.maximum(x, c) with c constant: gradient to x where x >= c."""
    c = torch.as_tensor(c, dtype=x.dtype, device=x.device)
    return torch.where(x >= c, x, c)


def tfmax_cx(c, x: torch.Tensor) -> torch.Tensor:
    """tf.maximum(c, x) with c constant: gradient to x where x > c."""
    c = torch.as_tensor(c, dtype=x.dtype, device=x.device)
    return torch.where(c >= x, c, x)


# ----------------------------------------------------------------------------------
# Wei-Hu-Dai complex LAMP / GM-LAMP (define_shrinkage.py, LAMP.m, eta_gm.m)
# ----------------------------------------------------------------------------------
def soft_wei(r: torch.Tensor, rvar: torch.Tensor, theta: torch.Tensor):
    """Complex soft threshold of LAMP (shrink_soft_threshold in define_shrinkage.py).

    lam = max(|theta[0]*sqrt(rvar)|, 0); xhat = r * (max(|r|-lam,0)/|r|)
    returns xhat, dxdr, dxdr1 (column means of d xhat/d r and d xhat/d r*)
    """
    cdt = r.dtype
    th = theta.reshape(-1)[0]
    lam = th * torch.sqrt(rvar)
    lam = tfmax_xc(torch.abs(lam), 0.0)
    ar = torch.abs(r)
    ampl = tfmax_xc(ar - lam, 0.0)
    xhat = r * (ampl / ar).to(cdt)
    on = ((ar - lam) > 0).to(cdt)
    dxdr = torch.mean(on * (1 - lam / (2 * ar)).to(cdt), 0)
    dxdr1 = torch.mean(on * (0.5 * lam.to(cdt) * torch.sqrt(r) * torch.pow(torch.conj(r), -1.5)), 0)
    return xhat, dxdr, dxdr1


def gm_wei(r: torch.Tensor, rvar: torch.Tensor, theta: torch.Tensor, K: int):
    """Gaussian-mixture MMSE shrinkage of GM-LAMP (shrink_gmest / eta_gm.m).

    theta (3,K): [logits of P ; means ; log variances (floored at -40)]
    """
    cdt = r.dtype
    th = -40.0
    P = (torch.exp(theta[0]) / torch.sum(torch.exp(theta[0]))).to(cdt)
    mu = theta[1].to(cdt)
    var = torch.exp(tfmax_cx(th, theta[2])).to(cdt)
    rv = rvar.to(cdt)
    U = V = dU = dV = dU1 = dV1 = 0
    expos = [tfmax_cx(th, torch.real(-torch.conj(r - mu[k]) * (r - mu[k]) / 2 / (rv + var[k]))) for k in range(K)]
    # complex64 only: divide every component by exp(max_k expo). xhat = U/V and dxdr = (dU V - U dV)/|V|^2 are
    # invariant to a common factor, so the result is the same in exact arithmetic; without it V and |V|^2 underflow
    # in float32 (V ~ e^-40) and the gradient becomes NaN. complex128 keeps the original (verified) arithmetic.
    shift = torch.stack(expos).amax(0).detach() if cdt == torch.complex64 else None
    for k in range(K):
        tmu = (mu[k] * rv + r * var[k]) / (rv + var[k])
        d = r - mu[k]
        e = expos[k] if shift is None else expos[k] - shift
        pdf = (torch.exp(e) / 2 / math.pi / torch.real(var[k] + rv)).to(cdt)
        u_k = tmu * P[k] * pdf
        du_k = var[k] / (rv + var[k]) * P[k] * pdf + u_k * (-torch.conj(d) / 2 / (rv + var[k]))
        du_k1 = u_k * (-d / 2 / (rv + var[k]))
        v_k = P[k] * pdf
        dv_k = v_k * (-torch.conj(d) / 2 / (rv + var[k]))
        dv_k1 = v_k * (-d / 2 / (rv + var[k]))
        U = U + u_k
        dU = dU + du_k
        dU1 = dU1 + du_k1
        V = V + v_k
        dV = dV + dv_k
        dV1 = dV1 + dv_k1
    xhat = U / V
    dxdr = (dU * V - U * dV) / (V * torch.conj(V))
    dxdr1 = (dU1 * V - U * dV1) / (V * torch.conj(V))
    return xhat, torch.mean(dxdr, 0), torch.mean(dxdr1, 0)


# ----------------------------------------------------------------------------------
# LADMM (complex_soft_threshold in the LADMM training scripts / LADMM_test.m)
# ----------------------------------------------------------------------------------
def csoft_ladmm(V: torch.Tensor, threshold: torch.Tensor) -> torch.Tensor:
    """|V| -> max(|V|-thr, 0), phase kept: Z = cast(shrunk)*exp(j*angle(V)); Z(shrunk==0)=0."""
    mag = torch.abs(V)
    phase = torch.angle(V)
    shrunk = tfmax_xc(mag - threshold, 0.0)
    Z = shrunk.to(V.dtype) * torch.exp(torch.complex(torch.zeros_like(phase), phase))
    return torch.where(shrunk > 0, Z, torch.zeros_like(Z))


# ----------------------------------------------------------------------------------
# Real-valued shrinkages of the Borgerding / ALISTA code + complex extensions
# ----------------------------------------------------------------------------------
def soft_real(r: torch.Tensor, lam: torch.Tensor) -> torch.Tensor:
    """simple_soft_threshold / utils.tf.shrink: sign(r)*max(|r|-max(lam,0), 0)."""
    lam = tfmax_xc(lam, 0.0)
    return torch.sign(r) * tfmax_xc(torch.abs(r) - lam, 0.0)


def soft_complex(r: torch.Tensor, lam: torch.Tensor) -> torch.Tensor:
    """Complex extension: r/|r| * max(|r|-max(lam,0), 0) (equals soft_real for real r)."""
    lam = tfmax_xc(lam, 0.0)
    ar = torch.abs(r)
    ampl = tfmax_xc(ar - lam, 0.0)
    scale = torch.where(ar > 0, ampl / torch.where(ar > 0, ar, torch.ones_like(ar)), torch.zeros_like(ar))
    return r * scale.to(r.dtype) if r.is_complex() else r * scale


def percentile_nearest(x: torch.Tensor, q: float, dim: int = 0) -> torch.Tensor:
    """tf.contrib.distributions.percentile(x, q, axis=dim, keep_dims=True), interpolation 'nearest'.

    Same rule as the TF implementation: sort descending, index = round_half_even((n-1)*(1-q/100)).
    """
    n = x.shape[dim]
    frac = 1.0 - q / 100.0
    idx = int(torch.round(torch.tensor((n - 1) * frac, dtype=torch.float64)).item())
    srt = torch.sort(x, dim=dim, descending=True).values
    return srt.narrow(dim, idx, 1)


def shrink_ss_real(inputs: torch.Tensor, theta: torch.Tensor, q: float) -> torch.Tensor:
    """ALISTA support-selection shrinkage (utils/tf.py shrink_ss), real valued."""
    ab = torch.abs(inputs)
    thres = percentile_nearest(ab, 100.0 - q, 0)
    index = torch.logical_and(ab > theta, ab > thres).to(inputs.dtype).detach()
    cindex = 1.0 - index
    shr_in = cindex * inputs
    return index * inputs + torch.sign(shr_in) * tfmax_xc(torch.abs(shr_in) - theta, 0.0)


def shrink_ss_complex(inputs: torch.Tensor, theta: torch.Tensor, q: float) -> torch.Tensor:
    """Complex extension of shrink_ss (magnitude-based selection, phase kept)."""
    ab = torch.abs(inputs)
    thres = percentile_nearest(ab, 100.0 - q, 0)
    index = torch.logical_and(ab > theta, ab > thres).to(ab.dtype).detach()
    cindex = 1.0 - index
    shr_in = cindex.to(inputs.dtype) * inputs
    a = torch.abs(shr_in)
    ampl = tfmax_xc(a - theta, 0.0)
    scale = torch.where(a > 0, ampl / torch.where(a > 0, a, torch.ones_like(a)), torch.zeros_like(a))
    return index.to(inputs.dtype) * inputs + shr_in * scale.to(inputs.dtype)


def bg_real(r: torch.Tensor, rvar: torch.Tensor, theta: torch.Tensor):
    """Bernoulli-Gaussian MMSE estimator (Borgerding shrink_bgest), real valued.

    theta[...,0]: variance of nonzero x (abs); theta[...,1]: log(1/lambda - 1).
    Returns xhat and the column mean of dxhat/dr.
    """
    xvar1 = torch.abs(theta[..., 0])
    loglam = theta[..., 1]
    beta = 1 / (1 + rvar / xvar1)
    r2scale = r * r * beta / rvar
    rho = torch.exp(loglam - .5 * r2scale) * torch.sqrt(1 + xvar1 / rvar)
    rho1 = rho + 1
    xhat = beta * r / rho1
    dxdr = beta * ((1 + rho * (1 + r2scale)) / torch.square(rho1))
    return xhat, torch.mean(dxdr, 0)


def bg_complex(r: torch.Tensor, rvar: torch.Tensor, theta: torch.Tensor):
    """Complex Bernoulli-Gaussian MMSE estimator (circular CN prior), complex extension.

    x ~ (1-lam) delta + lam CN(0, xvar1),  r = x + CN(0, rvar).
    Returns xhat and the column mean of the Wirtinger derivative d xhat / d r
    (the quantity the VAMP Onsager term needs for circular denoisers).
    """
    xvar1 = torch.abs(theta[..., 0])
    loglam = theta[..., 1]
    beta = 1 / (1 + rvar / xvar1)                       # xvar1/(xvar1+rvar)
    a2 = torch.real(r * torch.conj(r))
    r2scale = a2 * beta / rvar
    rho = torch.exp(loglam - r2scale) * (1 + xvar1 / rvar)   # p(off)/p(on) ratio
    rho1 = rho + 1
    xhat = (beta / rho1).to(r.dtype) * r
    # d xhat/dr (Wirtinger) = beta/rho1 * (1 + rho*r2scale/rho1)
    dxdr = beta / rho1 * (1 + rho * r2scale / rho1)
    return xhat, torch.mean(dxdr, 0)


# ----------------------------------------------------------------------------------
# Noise-adaptive shrinkages of LADMM (v28; not in the original code)
# ----------------------------------------------------------------------------------
LN2 = math.log(2.0)


def beam_noise_level(V: torch.Tensor) -> torch.Tensor:
    """Blind noise level of every column: sigma_hat = median_i |V_i| / sqrt(ln 2), shape (1, B).
    For CN(0, s^2) noise |V_i| is Rayleigh with median s sqrt(ln 2); with a sparse channel most entries are noise."""
    return torch.median(torch.abs(V), dim=0, keepdim=True).values / math.sqrt(LN2)


def firm_shrink(V: torch.Tensor, t1: torch.Tensor, t2: torch.Tensor) -> torch.Tensor:
    """Firm (MCP proximal) shrinkage, phase kept: 0 for |V| <= t1, linear ramp to |V| = t2, identity above t2
    (no bias on the strong beams). t1 < t2, broadcastable to V."""
    mag = torch.abs(V)
    g = torch.clamp((mag - t1) / (t2 - t1), 0.0, 1.0) * t2 / torch.clamp(mag, min=1e-12)
    return V * torch.where(mag > t2, torch.ones_like(g), g).to(V.dtype)


def gm_mmse_shrink(V: torch.Tensor, sig2: torch.Tensor, logit_w: torch.Tensor, log_nu: torch.Tensor) -> torch.Tensor:
    """MMSE estimate of x from V = x + CN(0, sig2) under x ~ sum_k w_k CN(0, nu_k) (exact complex density
    exp(-|v|^2/s)/(pi s), zero means): xhat = V sum_k beta_k(V) nu_k/(nu_k + sig2),
    beta_k = posterior weight of component k. w = softmax(logit_w), nu = exp(log_nu). sig2: (1, B)."""
    w = torch.log_softmax(logit_w, dim=0)
    nu = torch.exp(log_nu)
    s = torch.real(V) ** 2 + torch.imag(V) ** 2                      # (N, B)
    v = nu.reshape(-1, 1, 1) + sig2.unsqueeze(0)                     # (K, 1, B)
    logl = w.reshape(-1, 1, 1) - torch.log(v) - s.unsqueeze(0) / v   # (K, N, B)
    beta = torch.softmax(logl, dim=0)
    f = torch.sum(beta * (nu.reshape(-1, 1, 1) / v), dim=0)          # (N, B)
    return V * f.to(V.dtype)


def gm_mmse_shrink_beam(V: torch.Tensor, sig2: torch.Tensor, logits: torch.Tensor, log_nu: torch.Tensor) -> torch.Tensor:
    """gm_mmse_shrink with mixture logits for every entry: logits (K, N, B) (graph-coupled GM, v29)."""
    w = torch.log_softmax(logits, dim=0)
    nu = torch.exp(log_nu)
    s = torch.real(V) ** 2 + torch.imag(V) ** 2
    v = nu.reshape(-1, 1, 1) + sig2.unsqueeze(0)
    logl = w - torch.log(v) - s.unsqueeze(0) / v
    beta = torch.softmax(logl, dim=0)
    return V * torch.sum(beta * (nu.reshape(-1, 1, 1) / v), dim=0).to(V.dtype)

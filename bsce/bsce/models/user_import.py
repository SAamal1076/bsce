"""Read LADMM networks saved by the user's own PyTorch scripts (untitled2.py and its relatives).

untitled2.py saves torch.save({'model_state': state_dict, 'done', 'log', 'layernmse'}) with state_dict keys
A, AT, H_matrices.t, rhos.t, thetas.t, alphas.t (float32). Its layer is exactly bsce's LADMM 'full':
    H = H_t (A^T y + rho_t Z - U);  Z = soft(H + U/rho_t, theta_t/rho_t);  U += alpha_t rho_t (H - Z)
"""
from __future__ import annotations

import numpy as np
import torch

from .ladmm import LADMM


def _safe_load(path):
    try:
        from numpy._core.multiarray import scalar
    except ImportError:                       # numpy < 2
        from numpy.core.multiarray import scalar
    torch.serialization.add_safe_globals([scalar, np.dtype, type(np.dtype("float64")), type(np.dtype("float32"))])
    return torch.load(path, map_location="cpu", weights_only=True)


def is_untitled2_checkpoint(path) -> bool:
    try:
        ck = _safe_load(path)
    except Exception:          # bsce .pt files hold numpy arrays / configs -> not loadable with weights_only
        return False
    return isinstance(ck, dict) and "model_state" in ck and any(k.startswith("H_matrices.") for k in ck["model_state"])


def ladmm_from_untitled2(path, A=None, precision="float32", device=None):
    """-> (bsce LADMM with the checkpoint's weights, checkpoint dict). A defaults to the A stored in the file."""
    ck = _safe_load(path)
    ms = ck["model_state"]
    T = len([k for k in ms if k.startswith("H_matrices.")])
    A_file = ms["A"].double().numpy()
    if A is None:
        A = A_file
    elif np.abs(np.asarray(A, float) - A_file).max() > 1e-6:
        raise ValueError(f"{path}: its A is not the scenario's CS matrix")
    m = LADMM(np.asarray(A, np.float64), T=T, operator="full", precision=precision, init_precision="untitled2")
    W = {}
    for t in range(T):
        W[f"H_matrix_{t}"] = ms[f"H_matrices.{t}"].double().numpy()
        W[f"rho_{t}"] = ms[f"rhos.{t}"].double().numpy()
        W[f"theta_{t}"] = ms[f"thetas.{t}"].double().numpy()
        W[f"alpha_{t}"] = ms[f"alphas.{t}"].double().numpy()
    m.load_mat_dict(W)
    if device is not None:
        m.to(device)
    return m, ck

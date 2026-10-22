"""Save / load trained models in PyTorch (.pt) and MATLAB (.mat) formats.

<prefix>.pt  : {'cfg', 'state_dict', 'A', 'train_state'}  -> bsce.export.load_model()
<prefix>.mat : MATLAB struct-friendly file read by matlab/bsce_load.m + bsce_infer.m:
               model (name), cfg_* fields, A, all learned/derived parameters under the
               names the original MATLAB code uses (B_t/theta{t} for LAMP & GM-LAMP,
               H_matrix_t/rho_t/theta_t/alpha_t for LADMM, ...).
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import scipy.io as sio
import torch

from .models import build_model


def _matsafe(v):
    if isinstance(v, (bool, np.bool_)):
        return np.array(int(v))
    if v is None:
        return np.array([])
    if isinstance(v, (list, tuple)):
        return np.asarray(v)
    return v


def _double(d: dict) -> dict:
    """float32/complex64 arrays -> float64/complex128 (exact). MATLAB computes in single as soon
    as one operand is single, so storing doubles keeps MATLAB/Octave inference in double."""
    out = {}
    for k, v in d.items():
        a = np.asarray(v) if not isinstance(v, str) else v
        if isinstance(a, np.ndarray) and a.dtype == np.float32:
            a = a.astype(np.float64)
        elif isinstance(a, np.ndarray) and a.dtype == np.complex64:
            a = a.astype(np.complex128)
        out[k] = a
    return out


def model_mat_dict(model) -> dict:
    d = dict(model.mat_dict())
    d["model"] = model.cfg["model"]
    d["A"] = model.A_np
    d["T"] = np.array(model.T)
    for k, v in model.cfg.items():
        if k in ("model", "W", "members"):          # an ensemble's members: in its .pt file
            continue
        d[f"cfg_{k}"] = _matsafe(v)
    fam = model.family
    if fam == "lvamp":
        d.update(V=model.V.cpu().numpy(), rS2=model.rS2.cpu().numpy(), UsT=model.UsT.cpu().numpy())
    if fam == "ladmm":
        d["operator"] = model.operator
    return d


def original_format_dict(model):
    """Only the variables the ORIGINAL MATLAB test code reads (drop-in trained-weight file).

    LAMP / GM-LAMP -> B_0..B_{T-1}, theta0..theta{T-1}        (LAMP.m, GM_LAMP.m, *_test.m)
    LADMM (any operator, materialised) -> H_matrix_t, rho_t, theta_t, alpha_t (LADMM_test.m)
    The main .mat also stores A, T, cfg_*; MATLAB's load() inside a function would overwrite
    local variables of the same names, so this file contains nothing else.
    """
    if model.family == "lamp":
        return model.mat_dict()
    if model.family == "ladmm":
        if getattr(model, "data_step", "plain") != "plain" or model.threshold in ("adaptive", "firm", "gm"):
            return None                 # v28 options: LADMM_test.m computes A^T y and the soft threshold only
        d = model.mat_dict()
        keep = ("H_matrix_", "rho_", "theta_", "alpha_")
        return {k: v for k, v in d.items() if k.startswith(keep) and k.split("_")[-1].isdigit()
                and not k.startswith(("rho_raw", "theta_raw", "alpha_raw"))}
    return None


def _ext(prefix: Path, ext: str) -> Path:
    """<prefix><ext>. Not Path.with_suffix: a run name with a dot (an override such as train.lr=0.0005, or
    checkpoint_minutes=0.5) would lose everything after its dot (v23 and older)."""
    return prefix.parent / (prefix.name + ext)


def save_model(model, prefix, train_state=None, data_info=None):
    """Write <prefix>.pt, <prefix>.mat and (LAMP/GM-LAMP/LADMM) <prefix>_orig.mat."""
    prefix = Path(prefix)
    prefix.parent.mkdir(parents=True, exist_ok=True)
    cfg = dict(model.cfg)
    if model.family == "alista":
        cfg["W"] = model.W.cpu().numpy()
    torch.save({"cfg": cfg, "state_dict": model.state_dict(), "A": model.A_np,
                "train_state": train_state, "data_info": data_info}, _ext(prefix, ".pt"))
    md = model_mat_dict(model)
    if train_state:
        md["train_layer_nmse_db"] = np.asarray(train_state.get("layer_nmse_db", []), float)
    sio.savemat(_ext(prefix, ".mat"), _double(md), do_compression=True)
    od = original_format_dict(model)
    if od is not None:
        sio.savemat(prefix.parent / (prefix.name + "_orig.mat"), _double(od), do_compression=True)
    return _ext(prefix, ".pt"), _ext(prefix, ".mat")


def truncate_model(model, T):
    """The first T layers of an unrolled network as a T-layer network (same weights, same output as
    model.forward_stages(y)[2T-1] for LADMM). Used to save the network as it is when layer T is finished."""
    cfg = dict(model.cfg)
    cfg["T"] = int(T)
    if model.family == "alista":
        cfg["W"] = model.W.cpu().numpy()
    m = build_model(cfg, model.A_np).to(model.A.device)
    full = model.state_dict()
    sd = m.state_dict()
    Tf = int(model.cfg.get("T", 0))
    new, missing = {}, []
    for k, v in sd.items():
        f = full.get(k)
        if f is not None and f.shape == v.shape:
            new[k] = f.detach().clone()
        elif (f is not None and f.dim() >= 1 and f.shape[0] == Tf and v.shape[0] == int(T)
              and f.shape[1:] == v.shape[1:]):                  # per-layer vector (e.g. NF-LADMM rho_raw): first T
            new[k] = f[:int(T)].detach().clone()
        else:
            missing.append(k)
    if missing:
        raise ValueError(f"cannot truncate {cfg.get('model')} to T={T}: {missing[:4]}")
    m.load_state_dict(new)
    return m


def load_model(path, device=None, precision=None):
    """Load a .pt (exact) or a .mat (MATLAB export) model file."""
    path = Path(path)
    if path.suffix == ".pt":
        ck = torch.load(path, map_location="cpu", weights_only=False)
        cfg = dict(ck["cfg"])
        if precision:
            cfg["precision"] = precision
        m = build_model(cfg, ck["A"])
        sd = {k: v for k, v in ck["state_dict"].items()}
        own = m.state_dict()
        drop = m._dead_names() if hasattr(m, "_dead_names") else set()     # v31: entries older files still store
        m.load_state_dict({k: v.to(dtype=own[k].dtype) for k, v in sd.items() if k not in drop})
    else:
        d = sio.loadmat(path, squeeze_me=False)
        cfg = {"model": str(np.asarray(d["model"]).ravel()[0])}
        for k, v in d.items():
            if k.startswith("cfg_"):
                v = np.asarray(v).ravel()
                cfg[k[4:]] = (v[0].item() if v.size == 1 and not isinstance(v[0], str) else
                              (str(v[0]) if v.size == 1 else v.tolist()))
        if cfg["model"] == "ALISTA":
            cfg["W"] = d["W"]
        for b in ("untied", "complex_mode", "coord"):
            if b in cfg:
                cfg[b] = bool(cfg[b])
        if precision:
            cfg["precision"] = precision
        for k in ("init_precision",):
            if k in cfg and cfg[k] in ([], ""):
                cfg[k] = None
        m = build_model(cfg, d["A"])
        m.load_mat_dict(d)
    if device is not None:
        m.to(device)
    return m

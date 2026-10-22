"""YAML configs with `_base:` inheritance and dotted command-line overrides."""
from __future__ import annotations

import copy
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parent.parent


def _merge(a: dict, b: dict) -> dict:
    out = copy.deepcopy(a)
    for k, v in b.items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = _merge(out[k], v)
        else:
            out[k] = copy.deepcopy(v)
    return out


def load_yaml(path) -> dict:
    path = Path(path)
    if not path.is_absolute() and not path.exists():
        path = ROOT / path
    d = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    if "_base" in d:
        base = load_yaml(path.parent / d.pop("_base"))
        d = _merge(base, d)
    return d


def apply_overrides(cfg: dict, overrides) -> dict:
    """overrides: ['train.lr=5e-4', 'model.T=10'] (values parsed as YAML)."""
    cfg = copy.deepcopy(cfg)
    for o in overrides or []:
        key, val = o.split("=", 1)
        d = cfg
        parts = key.split(".")
        for p in parts[:-1]:
            d = d.setdefault(p, {})
        v = yaml.safe_load(val)
        if isinstance(v, str):          # YAML 1.1 reads 8e-4 / 1e5 as strings: accept them as numbers
            try:
                v = float(v) if any(ch in v.lower() for ch in ".e") else int(v)
            except ValueError:
                pass
        d[parts[-1]] = v
    return cfg


def resolve(path) -> Path:
    p = Path(path)
    return p if p.is_absolute() else ROOT / p


merge = _merge


# ---------------------------------------------------------------------------------------
# scenario paths (a scenario may keep its own data in a folder without CSmatrix.mat / test set)
# ---------------------------------------------------------------------------------------
def scenario_data_dir(sc: dict, smoke: bool = False) -> Path:
    return resolve(sc["data_dir"] + ("_smoke" if smoke else ""))


def scenario_cs_matrix(sc: dict, smoke: bool = False) -> Path:
    """<data_dir>/CSmatrix.mat if present, else the scenario's `cs_matrix` file."""
    p = scenario_data_dir(sc, smoke) / "CSmatrix.mat"
    if p.exists() or "cs_matrix" not in sc:
        return p
    return resolve(sc["cs_matrix"])


def scenario_test_dir(sc: dict, smoke: bool = False) -> Path:
    """Folder with <D>_<ARR>_test_channel.mat: `test_data_dir` if given, else data_dir."""
    if "test_data_dir" in sc and not smoke:
        return resolve(sc["test_data_dir"])
    return scenario_data_dir(sc, smoke)


def band_file_names(sc: dict, band: dict):
    """(training measurements, validation measurements) file names of one band.
    Band keys `train_name` / `val_name` override the band name in the file names
    (e.g. the user's SV_UPA_validation_128_measurements_1_11dB.mat for band 0to20dB).
    White noise (scenario noise_model: white): the same names + '_white'."""
    from .data.noise import of_scenario, suffix
    pre = f"{sc['dataset']}_{sc['array']}"
    sf = suffix(of_scenario(sc))                       # '' (colored, as before) or '_white'
    tr = f"{pre}_training_{sc['M']}_measurements_{band.get('train_name', band['name'])}{sf}.mat"
    va = f"{pre}_validation_{sc['M']}_measurements_{band.get('val_name', band['name'])}{sf}.mat"
    return tr, va


def scenario_run_name(sc: dict, smoke: bool = False) -> str:
    """Folder of a scenario's runs under outputs/: <name>, <name>_white for white noise, + _smoke."""
    from .data.noise import of_scenario, suffix
    return sc["name"] + suffix(of_scenario(sc)) + ("_smoke" if smoke else "")

#!/usr/bin/env python3
"""Create a scenario variant (data sizes, validation SNRs) without touching the original files.

  python scripts/make_scenario.py --base sv_ula --name SV_ULA_150k --n-train 150000 --n-val 10000 --n-test 10000 --val-points 5
  python scripts/make_scenario.py --base sv_upa --name SV_UPA_valrange --val-points 5
  python scripts/make_scenario.py --base deepmimo_upa --name DeepMIMO_UPA_valrange --val-points 5
  python scripts/make_scenario.py --base sv_ula --name SV_ULA_fast --n-train 15000 --n-val 2000 --n-test 2000 --val-random 10

Writes configs/scenarios/<name lower-case>.yaml (inherits everything from the base file via _base)
with its own data_dir data/<name> and outputs in outputs/<name>, so the reproduction data stay as
they are. --val-points K replaces the single validation SNR of every band by K equally spaced SNRs
over the band's training range (e.g. 0, 2.5, 5, 7.5, 10 dB); --val-random K draws K SNRs uniformly
at random in the band (seeded; validation chunk k gets SNR k, as genMeasurements does for training); training with --protocol gmlamp_best
then validates with the mean over these SNRs (in dB) and keeps the best weights.
SV: the first channels are the same as in the base data (same seeds), the sets are just longer.
DeepMIMO: sizes are limited by the users of H_*.mat (O1_28 rows 1000-1300: 54,481 users, 52,481
without GM-LAMP's test users); for more users generate more rows with matlab/deepmimo_make_H.m and
convert with --test random.
Next: generate_data.py (SV) or deepmimo_convert.py + generate_data.py --measurements-only (DeepMIMO)
with --scenario configs/scenarios/<name>.yaml.
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import yaml

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from bsce.config import load_yaml  # noqa: E402


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--base", required=True, help="sv_ula, sv_upa, deepmimo_ula, deepmimo_upa (or a yaml path)")
    ap.add_argument("--name", required=True)
    ap.add_argument("--n-train", type=int, default=None)
    ap.add_argument("--n-val", type=int, default=None)
    ap.add_argument("--n-test", type=int, default=None)
    ap.add_argument("--val-points", type=int, default=None, help="validation SNRs per band (equally spaced)")
    ap.add_argument("--single-band", action="store_true",
                    help="one network for the whole SNR range (e.g. 0-20 dB) instead of the base bands")
    ap.add_argument("--val-random", type=int, default=None,
                    help="validation SNRs per band drawn uniformly at random in the band (seeded, like the training chunks)")
    a = ap.parse_args()
    bp = Path(a.base) if a.base.endswith(".yaml") else ROOT / "configs/scenarios" / f"{a.base}.yaml"
    base = load_yaml(bp)
    deep = base["dataset"] != "SV"
    d = {"_base": bp.name, "name": a.name, "data_dir": f"data/{a.name}"}
    n_train = a.n_train or base.get("n_train", 50000 if deep else 80000)
    n_val = a.n_val or base.get("n_val", 2000)
    n_test = a.n_test or base.get("n_test", 2000)
    d.update(n_train=n_train, n_val=n_val, n_test=n_test)
    chunks = int(base.get("n_snr_chunks", 10))
    if n_train % chunks:
        sys.exit(f"n_train={n_train} must be divisible by n_snr_chunks={chunks} (SNR chunks of genMeasurements)")
    if a.single_band:
        lo = min(b["train"][0] for b in base["bands"]); hi = max(b["train"][1] for b in base["bands"])
        vs = sorted({v for b in base["bands"] for v in b["val_snr"]})
        base = dict(base)
        base["bands"] = [{"name": f"{lo:g}to{hi:g}dB", "train": [lo, hi], "val_snr": vs, "use_for": [lo, hi + 0.01]}]
        d["bands"] = base["bands"]
    if a.val_points and a.val_random:
        sys.exit("use either --val-points or --val-random")
    k = a.val_points or a.val_random
    if k:
        if n_val % k:
            sys.exit(f"n_val={n_val} must be divisible by the number of validation SNRs ({k})")
        bands = []
        for j, b in enumerate(base["bands"]):
            lo, hi = b["train"]
            nb = dict(b)
            if a.val_points:
                v = np.linspace(lo, hi, k)
            else:   # one random SNR per validation chunk, uniform in the band (as the training chunks)
                v = np.random.RandomState(int(base["seed"]) + 500 + j).uniform(lo, hi, size=k)
            nb["val_snr"] = [float(round(x, 4)) for x in v]
            bands.append(nb)
        d["bands"] = bands
    if deep and n_train + n_val > 52481:
        print(f"NOTE: DeepMIMO O1_28 rows 1000-1300 have 52,481 users outside GM-LAMP's test set; "
              f"{n_train + n_val} requested -> generate more rows (deepmimo_make_H(..., rows)) and convert with --test random.")
    f = ROOT / "configs/scenarios" / f"{a.name.lower()}.yaml"
    head = f"# made by scripts/make_scenario.py from {bp.name}\n"
    f.write_text(head + yaml.safe_dump(d, sort_keys=False), encoding="utf-8")
    print(f"written {f.relative_to(ROOT)}")
    print(yaml.safe_dump(d, sort_keys=False))
    if deep:
        print(f"next: python scripts/deepmimo_convert.py --scenario configs/scenarios/{f.name} --input H_*.mat\n"
              f"      python scripts/generate_data.py --scenario configs/scenarios/{f.name} --measurements-only")
    else:
        print(f"next: python scripts/generate_data.py --scenario configs/scenarios/{f.name}")


if __name__ == "__main__":
    main()

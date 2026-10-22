#!/usr/bin/env python3
"""Put separately trained networks together into one model-averaging ensemble (UTRR-E: Wu, Cheng, Eldar, Cioffi,
IEEE TCOM 2025, Sec. IV-D): xhat = mean of the members' outputs.

  %run scripts/make_ensemble.py --runs outputs/SV_ULA/UTRR-T30_paper_K-0 outputs/SV_ULA/UTRR-T30_paper outputs/SV_ULA/UTRR-T30_paper_K-64 outputs/SV_ULA/UTRR-T30_paper_K-256 --tag UTRR-E-T30_paper

For every band file <run>_<band>.pt that all the runs have (e.g. 0to20dB, or 0to10dB + 10to20dB of fine-tuned runs),
it writes <folder of the first run>/<tag>/<tag>_<band>.pt (+ .mat, + README.txt). evaluate.py and compare.py then use
the ensemble like any other network. The members must use the same CS matrix.
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from bsce.export import load_model, save_model  # noqa: E402
from bsce.models.ensemble import Ensemble  # noqa: E402


def bands_of(run):
    return {f.stem[len(run.name) + 1:]: f for f in sorted(run.glob(f"{run.name}_*.pt"))}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--runs", nargs="+", required=True, help="run folders (at least two)")
    ap.add_argument("--tag", required=True, help="name of the ensemble folder, e.g. UTRR-E-T30_paper")
    ap.add_argument("--out", default=None, help="parent folder of the ensemble (default: that of the first run)")
    a = ap.parse_args()
    runs = [Path(r) if Path(r).is_absolute() else ROOT / r for r in a.runs]
    for r in runs:
        if not r.is_dir():
            sys.exit(f"no such run folder: {r}")
    if len(runs) < 2:
        sys.exit("give at least two runs")
    per = [bands_of(r) for r in runs]
    common = sorted(set.intersection(*[set(b) for b in per]))
    if not common:
        sys.exit("the runs have no band file in common: " + "; ".join(f"{r.name}: {sorted(b)}" for r, b in zip(runs, per)))
    out = (Path(a.out) if a.out else runs[0].parent) / a.tag
    out.mkdir(parents=True, exist_ok=True)
    for band in common:
        members = [load_model(b[band]) for b in per]
        A0 = members[0].A_np
        if any(not np.array_equal(m.A_np, A0) for m in members[1:]):
            sys.exit(f"band {band}: the members use different CS matrices")
        e = Ensemble.from_models(members)
        save_model(e, out / f"{a.tag}_{band}")
        print(f"{a.tag}_{band}.pt: mean of {len(members)} networks, {e.n_params():,} parameters")
    lines = [f"{a.tag}: model-averaging ensemble (xhat = mean of the members' outputs; UTRR-E, Wu et al., TCOM 2025)",
             f"made by: python scripts/make_ensemble.py --runs {' '.join(a.runs)} --tag {a.tag}",
             f"bands: {', '.join(common)}", "members:"]
    lines += [f"  {r}" for r in runs]
    (out / "README.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"written {out}")


if __name__ == "__main__":
    main()

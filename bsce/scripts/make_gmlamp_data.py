#!/usr/bin/env python3
"""Make the four data sets of the GM-LAMP setting (paper reproduction), exactly the data the review used.

  python scripts/make_gmlamp_data.py --sv
  python scripts/make_gmlamp_data.py --deepmimo-ula H_ula_1.mat --deepmimo-upa H_upa.mat
  python scripts/make_gmlamp_data.py --sv --deepmimo-ula H_ula_1.mat --deepmimo-upa H_upa.mat     (all four)
  options: --out-root data (default)   --only SV_ULA,DeepMIMO_UPA

What is made (the GM-LAMP setting, configs/scenarios/sv_ula.yaml, sv_upa.yaml, deepmimo_ula.yaml, deepmimo_upa.yaml):
  SV_ULA, SV_UPA   GM-LAMP's channel generator (generate_channel.m, L = 3 paths, half-wavelength, lens DFT matrix),
                   training 80,000 (as GM-LAMP's generate_channel_data_for_SV_channel_model.m), validation 2,000,
                   test 2,000 (as GM-LAMP's Main.m: K = 2000); fixed seeds -> identical files every time
  DeepMIMO_ULA/UPA your raw DeepMIMO v1.3 channels (O1_28, BS 3, rows 1000-1300, 3 paths, ULA on the z axis /
                   UPA y-z; matlab/deepmimo_make_H.m), x = U0^H h, ||x||^2 = N (GM-LAMP's convention);
                   test = GM-LAMP's own CE_DeepMIMO test file (2,000), its users removed; training 50,000,
                   validation 2,000 drawn from the other 52,481 users
  measurements     as GM-LAMP's generate_measurement_data.py: y = A (x + n), CS matrix of GM-LAMP
                   (CSmatrix256128 for SV, CSmatrix_128256 for DeepMIMO); two bands 0-10 dB and 10-20 dB:
                   training split in 10 chunks with SNR ~ U(band), validation at 10 dB (GM-LAMP's validation SNR)
  test noise       not stored: evaluate.py adds it at 0, 2, ..., 20 dB with fixed seeds (same for every method)
At the end the data fingerprints are compared with the data of the review (status.py --data): 4 x OK expected.
"""
import argparse
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))

SC = {"SV_ULA": "configs/scenarios/sv_ula.yaml", "SV_UPA": "configs/scenarios/sv_upa.yaml",
      "DeepMIMO_ULA": "configs/scenarios/deepmimo_ula.yaml", "DeepMIMO_UPA": "configs/scenarios/deepmimo_upa.yaml"}
NEED = [ROOT / "reference/gmlamp/CE_SV/CSmatrix256128.mat", ROOT / "reference/gmlamp/CE_DeepMIMO/CSmatrix_128256.mat",
        ROOT / "reference/gmlamp/CE_DeepMIMO/ULA_test_channel.mat", ROOT / "reference/gmlamp/CE_DeepMIMO/UPA_test_channel.mat"]


def run(cmd):
    print("\n>>> " + " ".join(str(c) for c in cmd), flush=True)
    r = subprocess.run([str(c) for c in cmd], cwd=ROOT)
    if r.returncode:
        sys.exit(f"failed: {' '.join(str(c) for c in cmd)}")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--sv", action="store_true", help="make SV_ULA and SV_UPA")
    ap.add_argument("--deepmimo-ula", default=None, help="raw DeepMIMO ULA file (deepmimo_make_H(..., 'ULA'))")
    ap.add_argument("--deepmimo-upa", default=None, help="raw DeepMIMO UPA file (deepmimo_make_H(..., 'UPA'))")
    ap.add_argument("--only", default=None, help="comma list of SV_ULA, SV_UPA, DeepMIMO_ULA, DeepMIMO_UPA")
    ap.add_argument("--out-root", default="data")
    a = ap.parse_args()
    need = ([NEED[0]] if a.sv else []) + ([NEED[1], NEED[2]] if a.deepmimo_ula else []) + ([NEED[1], NEED[3]] if a.deepmimo_upa else [])
    miss = [p for p in dict.fromkeys(need) if not p.exists()]
    if miss:
        print("\nThe folder reference/gmlamp is missing in this bsce folder. Copy it from your previous bsce folder "
              "(e.g. bsce_code (10)\\bsce\\reference) into this one, or extract bsce_reference_part1..4.zip here.")
    if miss:
        sys.exit("missing:\n  " + "\n  ".join(map(str, miss)) + "\nExtract bsce_reference_part1.zip, part2.zip and part3.zip "
                 "where you extracted bsce_code.zip (paths start with bsce/).")
    todo = []
    if a.sv:
        todo += ["SV_ULA", "SV_UPA"]
    if a.deepmimo_ula:
        todo.append("DeepMIMO_ULA")
    if a.deepmimo_upa:
        todo.append("DeepMIMO_UPA")
    if a.only:
        todo = [t for t in todo if t in a.only.split(",")]
    if not todo:
        sys.exit("nothing to do: give --sv and/or --deepmimo-ula <file> / --deepmimo-upa <file>")
    py = sys.executable
    root = Path(a.out_root)
    for name in todo:
        dd = (root / name).as_posix()
        if name.startswith("SV"):
            run([py, "scripts/generate_data.py", "--scenario", SC[name], "--set", f"data_dir={dd}"])
        else:
            raw = a.deepmimo_ula if name.endswith("ULA") else a.deepmimo_upa
            if not Path(raw).exists():
                sys.exit(f"raw DeepMIMO file not found: {raw}")
            run([py, "scripts/deepmimo_convert.py", "--scenario", SC[name], "--input", raw, "--set", f"data_dir={dd}"])
            run([py, "scripts/generate_data.py", "--scenario", SC[name], "--measurements-only", "--set", f"data_dir={dd}"])
    # fingerprints of the review data
    import numpy as np
    import scipy.io as sio
    from status import FP
    print("\ndata fingerprints (must match the data of the review):")
    bad = 0
    for name in todo:
        fa, fb, fc = FP[name]
        d = root / name if root.is_absolute() else ROOT / root / name
        x = sio.loadmat(d / f"{name}_test_channel.mat")["x"]
        t = sio.loadmat(d / f"{name}_training_channel.mat")["x"]
        v = sio.loadmat(d / f"{name}_validation_channel.mat")["x"]
        ok = abs(x[0, 0] - fa) < 1e-5 and abs(np.sum(abs(x) ** 2) - fb) < 1e-2 and abs(t[5, 7] - fc) < 1e-5
        bad += not ok
        print(f"  {name:13s} {'OK' if ok else 'DIFFERENT'}   train {t.shape[1]:,} / val {v.shape[1]:,} / test {x.shape[1]:,}"
              f"   files in {d}")
    if bad:
        print("DIFFERENT = not the review's data. SV: check you did not change the scenario files. DeepMIMO: the raw file must "
              "be DeepMIMO v1.3, O1_28, BS 3, rows 1000-1300, 3 paths, ULA on the z axis (deepmimo_make_H.m).")
    else:
        print("all OK: identical to the data used so far.")


if __name__ == "__main__":
    main()

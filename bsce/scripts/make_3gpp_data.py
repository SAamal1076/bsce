#!/usr/bin/env python3
"""Make the 3GPP TR 38.901 data sets of bsce in ONE command (as make_gmlamp_data.py does for SV and DeepMIMO).

  %run scripts/make_3gpp_data.py                          3GPP_UPA and 3GPP_ULA, 80,000 / 2,000 / 2,000 (the sizes of SV)
  %run scripts/make_3gpp_data.py --sizes dong2019         81,000 / 9,000 / 19,000 (Dong et al., IEEE JSTSP 2019)
  %run scripts/make_3gpp_data.py --n-train 96000 --n-val 2000 --n-test 2000
  %run scripts/make_3gpp_data.py --only 3GPP_ULAZ         the vertical ULA (the geometry of GM-LAMP's DeepMIMO ULA)
  %run scripts/make_3gpp_data.py --spec 19.2 --suffix _v19         a second copy with the Rel-19 tables
  options: --check (statistics against 38.901 and against SV / DeepMIMO afterwards)  --force  --dry-run

What is made (configs/scenarios/3gpp_upa.yaml, 3gpp_ula.yaml, 3gpp_ulaz.yaml; data/<NAME>/), Python only:
  channels     3GPP TR 38.901 v16.1 (--spec 19.2: Rel-19 tables), UMi street canyon, LOS/NLOS by Table 7.4.2-1, 28 GHz,
               narrowband, plane waves; BS at 10 m, single-polarised 38.901 elements, lambda/2. 3GPP_UPA: UPA 16 x 16 on
               y-z; 3GPP_ULA: ULA 256 on y (horizontal); 3GPP_ULAZ: ULA 256 on z (vertical). One isotropic UE antenna
               at 1.5 m, users uniform in the 120-degree hexagonal sector (ISD 200 m, d2D >= 10 m), outdoor.
               Built-in generator bsce/data/tr38901.py: the TR procedure as written, checked against Sionna 2.2
               (docs/3GPP_DATASET.md)
  beamspace    x = U0^H h (the DeepMIMO convention), ||x||^2 = N per channel
  split        training / validation / test from separate drops (no drop shared); the validation and test users are
               the same for any size, and the same users for the three arrays (fixed seeds)
  measurements as GM-LAMP's: y = A (x + n), A = CSmatrix256128; bands 0-10 and 10-20 dB, 10 SNR chunks per training
               file, validation at 10 dB (--snr <edges> for other bands)
  files        <D>_<ARR>_{training,validation,test}_channel.mat (x, N x K), the measurement files, CSmatrix.mat,
               <D>_<ARR>_meta.mat (position, LOS state, drop, large-scale parameters of every channel),
               dataset_info.json, dataset_check.png: the format of data/SV_* and data/DeepMIMO_*
Train as the other scenarios: run_suite.py --scenarios 3gpp_upa 3gpp_ula ... (or --scenarios all 3gpp).
"""
import argparse
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))

SETS = {"3GPP_UPA": ("UPA", "y", "UPA"), "3GPP_ULA": ("ULA", "y", "ULA_Y"), "3GPP_ULAZ": ("ULA", "z", "ULA")}
SIZES = {"sv": (80000, 2000, 2000), "deepmimo": (50000, 2000, 2000), "dong2019": (81000, 9000, 19000),
         "100k": (96000, 2000, 2000)}
CS = ROOT / "reference/gmlamp/CE_SV/CSmatrix256128.mat"


def run(cmd, dry):
    print("\n>>> " + " ".join(str(c) for c in cmd), flush=True)
    if dry:
        return
    r = subprocess.run([str(c) for c in cmd], cwd=ROOT)
    if r.returncode:
        sys.exit(f"failed: {' '.join(str(c) for c in cmd)}")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--sizes", choices=list(SIZES), default="sv",
                    help="(train, val, test): " + ", ".join(f"{k} {'/'.join(map(str, v))}" for k, v in SIZES.items()))
    ap.add_argument("--n-train", type=int, default=None, help="exact sizes instead of --sizes (all three)")
    ap.add_argument("--n-val", type=int, default=None)
    ap.add_argument("--n-test", type=int, default=None)
    ap.add_argument("--only", default="3GPP_UPA,3GPP_ULA", help="comma list of " + ", ".join(SETS))
    ap.add_argument("--spec", choices=["16.1", "19.2"], default="16.1")
    ap.add_argument("--channel", choices=["umi", "umi_los", "umi_nlos"], default="umi")
    ap.add_argument("--element", choices=["38.901", "omni"], default="38.901")
    ap.add_argument("--snr", nargs="+", type=float, default=None, help="band edges instead of the authors' bands")
    ap.add_argument("--suffix", default="", help="e.g. _v19: data/3GPP_UPA_v19 and configs/scenarios/3gpp_upa_v19.yaml")
    ap.add_argument("--seed", type=int, default=2026)
    ap.add_argument("--drop-users", type=int, default=250)
    ap.add_argument("--out-root", default="data")
    ap.add_argument("--check", action="store_true", help="afterwards: check_3gpp.py --builtin (10,000 channels)")
    ap.add_argument("--force", action="store_true", help="overwrite existing data sets")
    ap.add_argument("--dry-run", action="store_true", help="print the commands only")
    a = ap.parse_args()
    exact = [a.n_train, a.n_val, a.n_test]
    if any(v is not None for v in exact):
        if not all(v is not None for v in exact):
            sys.exit("give --n-train, --n-val and --n-test together (or one --sizes preset)")
        n = tuple(exact)
        how = "exact sizes"
    else:
        n = SIZES[a.sizes]
        how = f"--sizes {a.sizes}"
    names = [s.strip() for s in a.only.split(",") if s.strip()]
    bad = [s for s in names if s not in SETS]
    if bad:
        sys.exit(f"--only: unknown {bad}; choose from {list(SETS)}")
    if not CS.exists() and not a.dry_run:
        sys.exit(f"missing {CS}\n  Extract bsce_reference_part*.zip into the bsce folder (reference/gmlamp).")
    print(f"3GPP data sets {names}: {n[0]:,} / {n[1]:,} / {n[2]:,} channels (train / val / test, {how}); "
          f"38.901 v{a.spec}, {a.channel}, {a.element} elements, seed {a.seed}")
    py = sys.executable
    made = []
    for s in names:
        arr, axis, _ = SETS[s]
        name = s + a.suffix
        out = Path(a.out_root) / name
        if (ROOT / out / f"3GPP_{arr}_training_channel.mat").exists() and not a.force:
            print(f"\n--- {name}: {out} exists, kept (--force to make it again)")
            made.append(name)
            continue
        cmd = [py, "scripts/make_dataset.py", "--dataset", "3GPP", "--array", arr, "--ula-axis", axis,
               "--n-train", n[0], "--n-val", n[1], "--n-test", n[2], "--spec", a.spec, "--channel", a.channel,
               "--element", a.element, "--seed", a.seed, "--drop-users", a.drop_users, "--name", name,
               "--out-dir", out]
        cmd += ["--snr", *[f"{e:g}" for e in a.snr]] if a.snr else ["--authors-bands"]
        if a.force:
            cmd.append("--force")
        run(cmd, a.dry_run)
        made.append(name)
    if a.check:
        run([py, "scripts/check_3gpp.py", "--builtin", *[SETS[s][2] for s in names], "--n", 10000, "--spec", a.spec,
             "--channel", a.channel, "--element", a.element, "--seed", a.seed, "--name", f"check_3gpp{a.suffix}"],
            a.dry_run)
    if a.dry_run:
        return
    print("\n| data set | training | validation | test | LOS (train) | scenario file |\n|---|---|---|---|---|---|")
    for name in made:
        info = ROOT / a.out_root / name / "dataset_info.json"
        if info.exists():
            d = json.loads(info.read_text(encoding="utf-8"))
            sc = d.get("scenario", {})
            los = d.get("los_fraction", {}).get("training", float("nan"))
            print(f"| {name} | {sc.get('n_train', '?'):,} | {sc.get('n_val', '?'):,} | {sc.get('n_test', '?'):,} | "
                  f"{los:.3f} | configs/scenarios/{name.lower()}.yaml |")
    print("\ntrain: %run scripts/run_suite.py --scenarios " + " ".join(m.lower() for m in made) + " --methods ... --device cuda")


if __name__ == "__main__":
    main()

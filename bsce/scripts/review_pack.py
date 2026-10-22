#!/usr/bin/env python3
"""Collect the small result files of a step into one zip to upload for review.

  python scripts/review_pack.py --step S3_SV_ULA --paths outputs/SV_ULA
  python scripts/review_pack.py --step S2_depth --paths outputs/SV_ULA/LADMM-full_gmlamp outputs/depth_curve

Takes *.json, *.csv, *.png, *.txt, *.md and logs/*.log under the given paths (recursively), never
models (.pt / .mat) or checkpoints, plus env.json (python / torch / GPU / bsce file hashes) and
STATUS.md if present. Writes review/<step>_<date-time>.zip and prints its size (keep < 30 MB).
"""
import argparse
import datetime
import hashlib
import json
import platform
import sys
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
KEEP = {".json", ".csv", ".png", ".txt", ".md", ".log"}


def env_info():
    info = dict(python=sys.version.split()[0], platform=platform.platform())
    try:
        import torch
        info.update(torch=torch.__version__, cuda=torch.cuda.is_available(),
                    gpu=torch.cuda.get_device_name(0) if torch.cuda.is_available() else None)
    except Exception as e:  # noqa: BLE001
        info["torch"] = f"not importable: {e}"
    h = {}
    for f in sorted(list((ROOT / "bsce").rglob("*.py")) + list((ROOT / "scripts").glob("*.py"))):
        h[str(f.relative_to(ROOT)).replace("\\", "/")] = hashlib.sha1(f.read_bytes()).hexdigest()[:10]
    info["code_hashes"] = h
    return info


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--step", required=True)
    ap.add_argument("--paths", nargs="+", required=True)
    a = ap.parse_args()
    dest = ROOT / "review"
    dest.mkdir(exist_ok=True)
    stamp = datetime.datetime.now().strftime("%Y%m%d_%H%M")
    zf = dest / f"{a.step}_{stamp}.zip"
    n = 0
    with zipfile.ZipFile(zf, "w", zipfile.ZIP_DEFLATED) as z:
        for p in a.paths:
            pp = Path(p) if Path(p).is_absolute() else ROOT / p
            if not pp.exists():
                print(f"WARNING: {pp} does not exist")
                continue
            files = [pp] if pp.is_file() else [f for f in pp.rglob("*") if f.is_file()]
            for f in files:
                if f.suffix.lower() in KEEP and "checkpoints" not in f.parts:
                    z.write(f, str(f.relative_to(ROOT)).replace("\\", "/"))
                    n += 1
        z.writestr("env.json", json.dumps(env_info(), indent=1))
        if (ROOT / "STATUS.md").exists():
            z.write(ROOT / "STATUS.md", "STATUS.md")
    mb = zf.stat().st_size / 2 ** 20
    print(f"{zf}  ({n} files, {mb:.1f} MB){'  WARNING: > 30 MB' if mb > 30 else ''}")
    print("upload this zip together with the console output (copy/paste or a .txt file).")


if __name__ == "__main__":
    main()

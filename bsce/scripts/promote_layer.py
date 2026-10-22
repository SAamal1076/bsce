#!/usr/bin/env python3
"""Turn the network saved after layer T of a deep run (--save-layers) into a normal T-layer run folder.

  %run scripts/promote_layer.py --run outputs/SV_UPA/LADMM-T12 --T 6
  %run scripts/promote_layer.py --run "outputs/*/LADMM-T12" --T 6          (every scenario at once)

With layer-wise training, the network saved when layer T is finished IS the network that a run with T layers trains
(same data, same seed, same protocol: the first T layers see exactly the same stages and minibatches; checked bit for
bit on the CPU). So after choosing the depth there is no need to retrain: this script writes

  outputs/<SCENARIO>/<name with -T12 replaced by -T6>/<name>_<band>.pt/.mat/_orig.mat, train_log.json, README.txt

which run_suite.py then finds ("trained, skipped") and continues with the fine-tune. Nothing is overwritten unless
--force is given.
"""
import argparse
import glob
import json
import re
import shutil
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from bsce.config import resolve  # noqa: E402
from bsce.export import load_model, save_model  # noqa: E402
from bsce.runinfo import write_readme  # noqa: E402


def promote(run: Path, T: int, force: bool):
    m = re.search(r"-T(\d+)", run.name)
    if not m:
        sys.exit(f"{run.name}: no -T<number> in the folder name")
    Tfull = int(m.group(1))
    if T > Tfull:
        sys.exit(f"{run.name}: has {Tfull} layers, cannot give T={T}")
    name = run.name[:m.start()] + f"-T{T}" + run.name[m.end():]
    dst = run.parent / name
    snaps = sorted((run / "layers").glob(f"T{T:02d}_{run.name}_*.pt"))
    if not snaps:
        sys.exit(f"{run}/layers has no T{T:02d}_... file (was it trained with --save-layers?)")
    if dst.exists() and any(dst.glob(f"{name}_*.pt")) and not force:
        print(f"[promote] {dst} exists, kept (use --force to replace)")
        return
    dst.mkdir(parents=True, exist_ok=True)
    log = json.loads((run / "train_log.json").read_text(encoding="utf-8"))
    log["config"]["model"]["T"] = T
    log["promoted_from"] = {"run": str(run), "layer": T, "of": Tfull}
    for p in snaps:
        band = p.stem[len(f"T{T:02d}_{run.name}_"):]
        ck = torch.load(p, map_location="cpu", weights_only=False)
        net = load_model(p)
        info = dict(ck.get("data_info") or {})
        info.update(promoted_from=str(p))
        save_model(net, dst / f"{name}_{band}", train_state=ck.get("train_state"), data_info=info)
        b = log["bands"].get(band, {})
        b.update(T=T, final_val_nmse_db=info.get("val_nmse_db"), n_params=net.n_params(),
                 layer_nmse_db=(lambda v: v if T == Tfull else v[: (len(v) // Tfull if Tfull and len(v) >= Tfull else 2) * T])(
                     b.get("layer_nmse_db", [])) if b.get("layer_nmse_db") else [],   # outputs per layer: 2 LADMM/LAMP, 1 others
                 files=[str(dst / f"{name}_{band}.pt")])
        if "train_seconds" in b and info.get("seconds_since_start") is not None:
            b["train_seconds"] = info["seconds_since_start"]
        log["bands"][band] = b
        print(f"[promote] {p.name} -> {name}/{name}_{band}.pt  (val {info.get('val_nmse_db', float('nan')):.4f} dB)")
    log["bands"] = {k: v for k, v in log["bands"].items() if any(s.stem.endswith("_" + k) for s in snaps)}
    (dst / "train_log.json").write_text(json.dumps(log, indent=1, default=str), encoding="utf-8")
    write_readme(dst, f"{name}: the first {T} layers of {run.name} (network saved when layer {T} was finished)", [
        ("origin", [f"{run}/layers/T{T:02d}_*", "identical to a run trained directly with T = "
                    f"{T} (same data, seed and protocol: layer-wise training sees the same stages and minibatches)"]),
        ("bands", [f"{k}: validation {v.get('final_val_nmse_db', float('nan')):.4f} dB" for k, v in log["bands"].items()]),
        ("next", [f"fine-tune: run_suite.py with the method key for T = {T} (e.g. ladmm{T}), or finetune_bands.py "
                  f"--from {dst / (name + '_<band>.pt')}"]),
    ])
    if (run / "README.txt").exists():
        shutil.copy(run / "README.txt", dst / f"README_parent_{run.name}.txt")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run", nargs="+", required=True, help="deep run folder(s); patterns with * allowed")
    ap.add_argument("--T", type=int, nargs="+", required=True, help="layer(s) to promote, e.g. 4 6")
    ap.add_argument("--force", action="store_true")
    a = ap.parse_args()
    runs = []
    for r in a.run:
        runs += [Path(h) for h in (sorted(glob.glob(str(resolve(r)))) if "*" in r else [str(resolve(r))])]
    runs = [r for r in runs if (r / "layers").is_dir()]
    if not runs:
        sys.exit(f"no run folder with layers/ matches {a.run}")
    for r in runs:
        for T in a.T:
            promote(r, T, a.force)


if __name__ == "__main__":
    main()

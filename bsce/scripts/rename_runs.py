#!/usr/bin/env python3
"""Give the fine-tuned folders of bsce v23 and older the v24 names (dry run unless --apply).

  %run scripts/rename_runs.py              lists what would be renamed, changes nothing
  %run scripts/rename_runs.py --apply      renames

  v23 and older:  outputs/SV_UPA/LADMM-T6-ft/LADMM-T6-ft_0to10dB.pt
  v24:            outputs/SV_UPA/LADMM-T6_ft_0-10-20dB_20k/LADMM-T6_ft_0-10-20dB_20k_0to10dB.pt
The bands come from the old name (-ft = 0-10-20 dB, -ft0-5-10-15-20 = those edges), the iterations from the folder's
finetune_log.json; the rest of the name (_margin1, ...) is kept. Every file <old>_<band>.* is renamed, and the name in
finetune_log.json and README.txt is updated. A folder whose new name exists already, or without finetune_log.json (an
ensemble, a folder made by hand), is left as it is.
Comparisons made before keep the old names in their tables; compare.py run again shows the new ones.
--outputs outputs outputs_e4 ...   folders to look in (default: every outputs* folder of bsce)
"""
import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from bsce import naming  # noqa: E402


def plan(outputs):
    for out in outputs:
        for scen in sorted(p for p in out.iterdir() if p.is_dir()):
            for d in sorted(p for p in scen.iterdir() if p.is_dir()):
                info = naming.parse_ft(d.name)
                if not info or not info["legacy"]:
                    continue
                log = d / "finetune_log.json"
                if not log.exists():
                    yield d, None, "no finetune_log.json (not a fine-tune of finetune_bands.py): left"
                    continue
                try:
                    lg = json.loads(log.read_text(encoding="utf-8"))
                    iters = int(lg["args"]["iters"])
                except (ValueError, KeyError, TypeError):
                    yield d, None, "finetune_log.json without the number of iterations: left"
                    continue
                new = naming.ft_tag(info["src"], info["edges"], iters) + info["rest"]
                if (scen / new).exists():
                    yield d, None, f"{new} exists already: left"
                    continue
                yield d, scen / new, ""


def rename(old: Path, new: Path):
    o, n = old.name, new.name
    for f in sorted(old.iterdir()):
        if f.is_file() and f.name.startswith(o + "_"):
            f.rename(old / (n + f.name[len(o):]))
    for txt in ("finetune_log.json", "README.txt"):
        p = old / txt
        if p.exists():
            s = p.read_text(encoding="utf-8")
            if txt.endswith(".json"):
                d = json.loads(s)
                d["tag"] = n
                if isinstance(d.get("args"), dict) and d["args"].get("tag") == o:
                    d["args"]["tag"] = n
                d["renamed_from"] = o
                s = json.dumps(d, indent=1, default=str)
            else:
                s = s.replace(o + "_", n + "_").replace(o + ":", n + ":")
            p.write_text(s, encoding="utf-8")
    old.rename(new)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--outputs", nargs="+", default=None)
    ap.add_argument("--apply", action="store_true")
    a = ap.parse_args()
    outs = [ROOT / o if not Path(o).is_absolute() else Path(o) for o in a.outputs] if a.outputs else \
        sorted(p for p in ROOT.glob("outputs*") if p.is_dir())
    n = 0
    for old, new, why in plan([o for o in outs if o.is_dir()]):
        if new is None:
            print(f"  {old.relative_to(old.parent.parent)}: {why}")
            continue
        n += 1
        print(f"  {old.relative_to(old.parent.parent)}  ->  {new.name}")
        if a.apply:
            rename(old, new)
    if not n:
        print("nothing to rename")
    elif not a.apply:
        print(f"\n{n} folder(s) would be renamed; run again with --apply")
    else:
        print(f"\n{n} folder(s) renamed")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Every trained and fine-tuned network at a glance: what it is, whether it is finished, and where an interrupted one
will resume. Writes outputs/RUNS.md and outputs/RUNS.csv.

  %run scripts/runs.py                       every outputs/<SCENARIO>/<run>/
  %run scripts/runs.py --scenario SV_UPA     one scenario folder (also SV_UPA_white, ...)
  %run scripts/runs.py --outputs outputs outputs_e4 --no-write

Columns: run folder, kind (trained / fine-tuned from <source> / published / other), method and layers, parameters,
bands present (of those expected), state, best validation NMSE per band (dB), last change.
State:
  done                    every expected band network is there
  resumable: <where>      interrupted; the same command continues from the checkpoint (stage and iteration shown)
  incomplete              some band networks are missing and there is no checkpoint (the command starts them again)
"""
import argparse
import csv
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from bsce import naming  # noqa: E402


def _json(p):
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None


def where_train(ck):
    import torch
    try:
        d = torch.load(ck, map_location="cpu", weights_only=False)
    except Exception as e:                              # noqa: BLE001
        return f"checkpoint unreadable ({type(e).__name__})"
    st = d.get("train_state", {})
    part = st.get("partial")
    s = f"{len(st.get('done', []))} stages done"
    return s + (f", inside '{part['stage']}' at {part['where']}" if part else "")


def where_ft(ck):
    import torch
    try:
        d = torch.load(ck, map_location="cpu", weights_only=False)
    except Exception as e:                              # noqa: BLE001
        return f"checkpoint unreadable ({type(e).__name__})"
    return f"band {ck.stem.rsplit('_', 2)[-2]}, step {d.get('step')}"


def describe(run: Path):
    pts = sorted(p for p in run.glob(f"{run.name}_*.pt") if p.is_file())
    have = [p.stem[len(run.name) + 1:] for p in pts]
    cks = sorted((run / "checkpoints").glob("*.pt")) if (run / "checkpoints").is_dir() else []
    newest_pt = max([p.stat().st_mtime for p in pts] or [0])
    live = [c for c in cks if c.stat().st_mtime > newest_pt]
    tl, fl = _json(run / "train_log.json"), _json(run / "finetune_log.json")
    row = dict(run=run.name, kind="other", method="", T="", params="", bands="", state="", best="", source="",
               changed=time.strftime("%Y-%m-%d %H:%M", time.localtime(max([p.stat().st_mtime for p in pts + cks] or
                                                                         [run.stat().st_mtime]))))
    expected = have
    if fl is not None:
        row["kind"] = "fine-tuned"
        row["source"] = Path(fl.get("source", "")).parent.name or Path(fl.get("source", "")).stem
        expected = [b[0] for b in (fl.get("setting") or {}).get("bands", [])] or list(fl.get("bands", {})) or have
        sel = (fl.get("args") or {}).get("bands")
        if sel:
            expected = [b for b in expected if b in sel.split(",")]
        row["best"] = " ".join(f"{k} {v['best_val_db']:.2f}" for k, v in fl.get("bands", {}).items())
        it = (fl.get("args") or {}).get("iters")
        row["source"] += f" ({it} it./band)" if it else ""
    elif tl is not None:
        row["kind"] = "trained"
        m = (tl.get("config") or {}).get("model", {})
        row["method"] = m.get("model", "")
        bands = tl.get("bands", {})
        any_b = next(iter(bands.values()), {}) if bands else {}
        row["T"] = any_b.get("T", m.get("T", ""))
        row["params"] = any_b.get("n_params", "")
        sc_b = [b["name"] for b in (tl.get("scenario") or {}).get("bands", [])]
        expected = sc_b or have
        row["best"] = " ".join(f"{k} {v['final_val_nmse_db']:.2f}" for k, v in bands.items()
                               if isinstance(v, dict) and v.get("final_val_nmse_db") is not None)
    elif run.name.endswith("-authors"):
        row["kind"] = "published"
    info = naming.parse_ft(run.name)
    if info and row["kind"] == "fine-tuned" and info["legacy"]:
        row["kind"] = "fine-tuned (old name: scripts/rename_runs.py)"
    if row["kind"] == "fine-tuned" and not row["method"]:
        sp = run.parent / row["source"].split(" (")[0] / "train_log.json"
        st = _json(sp)
        if st:
            m = (st.get("config") or {}).get("model", {})
            row["method"] = m.get("model", "")
            b = next(iter(st.get("bands", {}).values()), {})
            row["T"], row["params"] = b.get("T", m.get("T", "")), b.get("n_params", "")
    missing = [b for b in expected if b not in have]
    row["bands"] = f"{len(expected) - len(missing)}/{len(expected)}" + (f" ({', '.join(have)})" if have else "")
    if not missing and pts:
        row["state"] = "done"
    elif live:
        row["state"] = "resumable: " + "; ".join(where_ft(c) if fl is not None or "_ft_" in run.name else where_train(c)
                                                for c in live)
    elif missing and (pts or cks or tl or fl):
        row["state"] = "incomplete (missing " + ", ".join(missing) + ")"
    else:
        row["state"] = "empty"
    return row


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--outputs", nargs="+", default=["outputs"])
    ap.add_argument("--scenario", nargs="*", default=None, help="scenario folders (names or prefixes), e.g. SV_UPA")
    ap.add_argument("--no-write", action="store_true")
    a = ap.parse_args()
    for o in a.outputs:
        out = ROOT / o if not Path(o).is_absolute() else Path(o)
        if not out.is_dir():
            print(f"{out}: no such folder")
            continue
        rows = []
        for scen in sorted(p for p in out.iterdir() if p.is_dir()):
            if a.scenario and not any(scen.name == s or scen.name.startswith(s + "_") for s in a.scenario):
                continue
            for run in sorted(p for p in scen.iterdir() if p.is_dir()):
                if any(run.glob(f"{run.name}_*.pt")) or (run / "train_log.json").exists() or (run / "finetune_log.json").exists() \
                        or (run / "checkpoints").is_dir():
                    rows.append(dict(scenario=scen.name, **describe(run)))
        if not rows:
            print(f"{out}: no runs")
            continue
        cols = ["scenario", "run", "kind", "source", "method", "T", "params", "bands", "state", "best", "changed"]
        md = [f"# Runs in {out.name}/ ({time.strftime('%Y-%m-%d %H:%M')}; scripts/runs.py)", ""]
        for sname in dict.fromkeys(r["scenario"] for r in rows):
            md += [f"## {sname}", "", "| run | kind | method, T, parameters | bands | state | best validation (dB) | changed |",
                   "|---|---|---|---|---|---|---|"]
            for r in (r for r in rows if r["scenario"] == sname):
                kind = r["kind"] + (f" from {r['source']}" if r["source"] else "")
                mtp = ", ".join(str(x) for x in (r["method"], f"T={r['T']}" if r["T"] != "" else "",
                                                 f"{r['params']:,}" if isinstance(r["params"], int) else r["params"]) if x)
                md.append(f"| {r['run']} | {kind} | {mtp} | {r['bands']} | {r['state']} | {r['best']} | {r['changed']} |")
            md.append("")
        print("\n".join(md))
        if not a.no_write:
            (out / "RUNS.md").write_text("\n".join(md), encoding="utf-8")
            with open(out / "RUNS.csv", "w", newline="", encoding="utf-8") as f:
                w = csv.DictWriter(f, fieldnames=cols)
                w.writeheader()
                w.writerows({k: r.get(k, "") for k in cols} for r in rows)
            print(f"written {out / 'RUNS.md'} and {out / 'RUNS.csv'}")


if __name__ == "__main__":
    main()

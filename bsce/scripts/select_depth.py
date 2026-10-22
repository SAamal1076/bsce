#!/usr/bin/env python3
"""Choose the number of layers T of one or several methods (e.g. LADMM and NF-LADMM) on the validation set.

  python scripts/select_depth.py --scenario configs/scenarios/sv_upa.yaml --snr 0 20 --T 2 4 6 8 10 12 \
         --train configs/train/ladmm_best.yaml configs/train/nfladmm_s.yaml --device cuda
  options: --tol 0.1          recommended T = smallest T within tol dB of the best T, in every band
           --protocol gmlamp  (passed to train.py)   --noise online|file   --snr <edges> (any SNR bands)
           --set train.steps=10000 ...                (passed to train.py for every config)
           --retrain          train again even if the run exists      --out outputs/depth_select
           --runs-only        do not train; read the existing runs

How T is measured, per training schedule:
  * layer-wise (schedule borgerding: LAMP, GM-LAMP, LADMM): one run with T = max(--T). The validation NMSE of every
    depth t is logged when layer t is finished (with its refinements), so one run gives the whole curve; the prefixes
    of the final network are not used (they are retuned later). Run folder <tag>-T<max>.
  * end to end (schedule steps / end2end: NF-LADMM): one run per T (folders <tag>-T<t>), final validation NMSE.
Each method uses its own validation set (the one its training uses); the chosen T is then trained/evaluated as usual.
Writes <out>/<scenario>/depth_select.png (values printed), depth_select.csv, depth_select.json.
"""
import argparse
import csv
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))
from bsce.config import load_yaml, merge, apply_overrides, resolve  # noqa: E402
from bsce.data.files import bands_from_edges  # noqa: E402
from depth_curve import per_layer  # noqa: E402
from train import model_tag  # noqa: E402


def run_train(a, cfg_path, T, tag, snr):
    cmd = [sys.executable, str(ROOT / "scripts/train.py"), "--scenario", a.scenario, "--train", cfg_path,
           "--protocol", a.protocol, "--device", a.device, "--tag", tag, "--out", a.runs_root]
    if snr:
        cmd += ["--snr"] + [f"{v:g}" for v in snr]
    if a.noise:
        cmd += ["--noise", a.noise]
    if a.smoke:
        cmd += ["--smoke"]
    cmd += ["--set", f"model.T={T}"] + list(a.set)
    print("\n>>> " + " ".join(cmd), flush=True)
    r = subprocess.run(cmd, cwd=ROOT)
    if r.returncode:
        sys.exit(f"training failed: {tag}")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--scenario", required=True)
    ap.add_argument("--train", nargs="+", required=True, help="train configs (one curve each)")
    ap.add_argument("--T", nargs="+", type=int, default=[2, 4, 6, 8, 10, 12])
    ap.add_argument("--snr", nargs="+", type=float, default=None)
    ap.add_argument("--noise", choices=["file", "online"], default=None)
    ap.add_argument("--protocol", default="original")
    ap.add_argument("--device", default="auto")
    ap.add_argument("--set", nargs="*", default=[])
    ap.add_argument("--tol", type=float, default=0.1)
    ap.add_argument("--retrain", action="store_true")
    ap.add_argument("--runs-only", action="store_true")
    ap.add_argument("--smoke", action="store_true", help="tiny CPU check (train.py --smoke)")
    ap.add_argument("--runs-root", default="outputs")
    ap.add_argument("--out", default="outputs/depth_select")
    a = ap.parse_args()

    sc = load_yaml(a.scenario)
    bands = bands_from_edges(a.snr) if a.snr else sc["bands"]
    Ts = sorted(set(a.T))
    root = resolve(a.runs_root) / (sc["name"] + ("_smoke" if a.smoke else ""))
    curves, meta = {}, {}
    for cpath in a.train:
        cfg = load_yaml(cpath)
        if a.protocol in ("fair", "gmlamp", "gmlamp_best", "ladmm_best", "e2e"):
            cfg = merge(cfg, load_yaml(ROOT / f"configs/train/{a.protocol}_protocol.yaml"))
        cfg = apply_overrides(cfg, a.set)
        mcfg = dict(cfg["model"])
        base = (f"{model_tag(mcfg)}_{Path(cpath).stem}" + ("" if a.protocol == "original" else f"_{a.protocol}")
                + "".join(f"_{s.replace('=', '-').replace('.', '-')}" for s in a.set))   # one folder per config + options
        layerwise = cfg["train"].get("schedule", "borgerding") in ("borgerding", "alista")
        runs = [(max(Ts), f"{base}-T{max(Ts)}")] if layerwise else [(T, f"{base}-T{T}") for T in Ts]
        for T, tag in runs:
            logf = root / tag / "train_log.json"
            done = logf.exists() and all(b["name"] in json.loads(logf.read_text(encoding="utf-8")).get("bands", {})
                                         for b in bands)
            if not a.runs_only and (a.retrain or not done):
                run_train(a, cpath, T, tag, a.snr)
            if not logf.exists():
                print(f"[depth] missing {logf}: skipped")
                continue
            log = json.loads(logf.read_text(encoding="utf-8"))
            for b in bands:
                bl = log["bands"].get(b["name"])
                if not bl:
                    continue
                if layerwise:
                    v = per_layer(bl["layer_nmse_db"], T)
                    for t in Ts:
                        if t <= len(v):
                            curves.setdefault(base, {}).setdefault(b["name"], {})[t] = v[t - 1]
                else:
                    curves.setdefault(base, {}).setdefault(b["name"], {})[T] = bl["final_val_nmse_db"]
                meta.setdefault(base, {})[f"params_T{T}"] = bl.get("n_params")
            meta.setdefault(base, {}).update(layerwise=layerwise, config=cpath)
    rec = {}
    for m, bd in curves.items():
        rec[m] = {}
        for bn, d in bd.items():
            best = min(d.values())
            rec[m][bn] = min(t for t, v in d.items() if v - best <= a.tol)
        rec[m]["all_bands"] = max(rec[m].values())
    out = resolve(a.out) / (sc["name"] + ("_smoke" if a.smoke else ""))
    out.mkdir(parents=True, exist_ok=True)
    rows = [dict(method=m, band=bn, T=t, val_nmse_db=round(v, 4), recommended=int(t == rec[m]["all_bands"]))
            for m, bd in curves.items() for bn, d in bd.items() for t, v in sorted(d.items())]
    with open(out / "depth_select.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["method", "band", "T", "val_nmse_db", "recommended"])
        w.writeheader(); w.writerows(rows)
    (out / "depth_select.json").write_text(json.dumps(dict(T_recommended=rec, curves=curves, meta=meta, tol_db=a.tol,
                                                           bands=[b["name"] for b in bands]), indent=1), encoding="utf-8")
    for m in rec:
        print(f"[depth] {m}: recommended T = {rec[m]['all_bands']}  per band {({k: v for k, v in rec[m].items() if k != 'all_bands'})}")
    plot(curves, rec, meta, [b["name"] for b in bands], out / "depth_select.png", sc["name"], a.tol)
    print("written", out / "depth_select.png")


def plot(curves, rec, meta, bands, path, title, tol):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    pal = ["#2a78d6", "#c23b6a", "#1f9e6e", "#eb6834", "#8c5ab4", "#555555"]
    fig, axs = plt.subplots(1, len(bands), figsize=(6.2 * len(bands), 5), squeeze=False)
    for ax, bn in zip(axs[0], bands):
        for k, (m, bd) in enumerate(curves.items()):
            d = bd.get(bn)
            if not d:
                continue
            ts = sorted(d)
            v = [d[t] for t in ts]
            c = pal[k % len(pal)]
            tr = rec[m]["all_bands"]
            ax.plot(ts, v, "-o", color=c, ms=4, lw=1.6,
                    label=f"{m} ({'layer-wise, one run' if meta[m]['layerwise'] else 'one run per T'}): T* = {tr}")
            for t, u in zip(ts, v):
                ax.annotate(f"{u:.2f}", (t, u), fontsize=7, color=c, xytext=(0, 6 if k % 2 == 0 else -11),
                            textcoords="offset points", ha="center")
            if tr in d:
                ax.plot([tr], [d[tr]], "*", color=c, ms=14, mec="k", mew=0.6)
        ax.set_xticks(sorted({t for bd in curves.values() for t in bd.get(bn, {})}))
        ax.set_xlabel("number of layers T"); ax.set_ylabel("validation NMSE (dB)")
        ax.set_title(f"{title} | band {bn}  (T* = smallest T within {tol} dB of the best)", fontsize=9)
        ax.grid(alpha=0.3); ax.legend(frameon=False, fontsize=7.5)
        for s_ in ("top", "right"):
            ax.spines[s_].set_visible(False)
    fig.tight_layout(); fig.savefig(path, dpi=160)


if __name__ == "__main__":
    main()

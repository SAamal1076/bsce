#!/usr/bin/env python3
"""NMSE versus number of layers from layer-wise training logs, and the recommended depth T.

  python scripts/depth_curve.py --runs outputs/SV_ULA/LADMM-full_gmlamp outputs/SV_UPA/LADMM-full_gmlamp
  options: --tol 0.1   recommended T = smallest depth within tol dB of the best depth, in every band
           --out <dir> (default: outputs/depth_curve)

With layer-wise training, the validation NMSE of the t-layer network is logged when layer t is
finished (with its refinements). One run with model.T=12 therefore gives the whole depth curve.
The shallow prefixes of the FINAL model are not usable estimators (later refinements retune the
early layers for the last output), so the curve uses the logged values; the network with the
chosen T is trained afresh in S3. Writes depth_curve.png (values printed),
depth_curve.csv and depth_curve.json (recommended T per run and overall).
"""
import argparse
import csv
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def per_layer(vals, T):
    """layer_nmse_db has one entry per network output; keep the last output of each layer."""
    k = max(1, len(vals) // T)
    out = list(vals[k - 1::k][:T])
    if len(vals) > k * T and out:     # outputs after the last layer (LADMM debias): the full network is the last one
        out[-1] = vals[-1]
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--runs", nargs="+", required=True, help="run folders containing train_log.json")
    ap.add_argument("--tol", type=float, default=0.1)
    ap.add_argument("--out", default="outputs/depth_curve")
    a = ap.parse_args()
    out = Path(a.out) if Path(a.out).is_absolute() else ROOT / a.out
    out.mkdir(parents=True, exist_ok=True)
    curves, rows, rec = {}, [], {}
    for r in a.runs:
        rd = Path(r) if Path(r).is_absolute() else ROOT / r
        f = rd / "train_log.json"
        if not f.exists():
            sys.exit(f"missing {f}")
        log = json.loads(f.read_text(encoding="utf-8"))
        T = int(log["config"]["model"]["T"])
        name = f"{log['scenario']['name']} {rd.name}"
        for band, b in log["bands"].items():
            v = per_layer(b["layer_nmse_db"], T)
            curves[(name, band)] = v
            best = min(v)
            t_rec = next(t + 1 for t, u in enumerate(v) if u - best <= a.tol)
            rec.setdefault(name, {})[band] = dict(T_recommended=t_rec, nmse_at_T_db=v[t_rec - 1], best_db=best,
                                                  T_trained=len(v), train_hours=b.get("train_seconds", 0) / 3600)
            for t, u in enumerate(v):
                gain = None if t == 0 else v[t - 1] - u
                rows.append(dict(run=name, band=band, layers=t + 1, val_nmse_db=round(u, 4),
                                 gain_vs_previous_db=None if gain is None else round(gain, 4)))
    for name, d in rec.items():
        d["T_recommended_all_bands"] = max(x["T_recommended"] for k, x in d.items() if isinstance(x, dict))
    overall = max(d["T_recommended_all_bands"] for d in rec.values())
    with open(out / "depth_curve.csv", "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
    (out / "depth_curve.json").write_text(json.dumps(dict(tol_db=a.tol, per_run=rec, T_recommended_overall=overall),
                                                     indent=1), encoding="utf-8")
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(9, 5.2))
    cols = ["#2a78d6", "#eb6834", "#1f9e6e", "#8c5ab4", "#c23b6a", "#888888", "#b08a1a", "#2bb0c4"]
    for (key, v), c in zip(curves.items(), cols * 4):
        name, band = key
        t_rec = rec[name][band]["T_recommended"]
        ax.plot(range(1, len(v) + 1), v, "-o", color=c, ms=4, lw=1.6,
                label=f"{name} {band}: T={t_rec} -> {v[t_rec - 1]:.2f} dB (best {min(v):.2f})")
        for t, u in enumerate(v):
            ax.annotate(f"{u:.2f}", (t + 1, u), fontsize=6, color=c, xytext=(0, 5), textcoords="offset points",
                        ha="center")
        ax.plot([t_rec], [v[t_rec - 1]], "s", ms=10, mfc="none", mec=c, mew=1.6)
    ax.set_xlabel("number of layers"); ax.set_ylabel("validation NMSE (dB)")
    ax.set_title(f"Depth curve (squares: smallest depth within {a.tol} dB of the best); overall T = {overall}",
                 fontsize=10)
    ax.grid(alpha=.3); ax.legend(frameon=False, fontsize=7)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    fig.tight_layout(); fig.savefig(out / "depth_curve.png", dpi=160)
    print(json.dumps(rec, indent=1))
    print(f"\nrecommended T (all runs, all bands, tol {a.tol} dB): {overall}")
    print(f"written: {out / 'depth_curve.png'}, depth_curve.csv, depth_curve.json")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Summarise tests/results/equivalence.jsonl into a table (markdown + csv) and a figure.

Every row of the jsonl is one equivalence check written by the pytest suite; this script
keeps the latest run of each test id and reports the worst relative error per group.
"""
import csv
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
RES = ROOT / "tests" / "results"

GROUPS = [
    # (test id, label, key for grouping, metrics)
    ("data/beamspace", "SV channels x = U.'h (vs generate_channel.m)", "array", ["x_rel"]),
    ("data/full_script_authors", "Full authors' data script, ULA, S=80000", None, ["rel"]),
    ("data/full_script_ladmm_project", "LADMM-project data script (100k + 10k channels per array)", "array", ["rel"]),
    ("data/cs_matrix", "CS matrix (generate_CS_matrix.m)", None, ["rel"]),
    ("data/measurements_matlab", "Measurements, MATLAB noise (Main.m)", None, ["y_rel"]),
    ("data/measurements_python", "Measurements, genMeasurements (python)", None, ["rel"]),
    ("lamp/vs_matlab", "LAMP / GM-LAMP vs LAMP.m / GM_LAMP.m (trained weights)", "net", ["rel"]),
    ("lamp/vs_tf_forward", "LAMP / GM-LAMP vs TF graph, all outputs (T=8)", "shrink", ["fwd_rel"]),
    ("lamp/vs_tf", "LAMP / GM-LAMP vs TF setup_training (grads, Adam)", "shrink", ["fwd_rel", "grad_rel", "adam_rel"]),
    ("ladmm/vs_matlab", "LADMM vs LADMM_test.m / LADMM_UPA_test.m (trained weights)", "array", ["rel", "layer_nmse_rel"]),
    ("ladmm/equals_admm", "LADMM (analytic init) vs ADMM_vectorized.m", None, ["rel"]),
    ("ladmm/vs_tf", "LADMM scripts (full/diag/lowrank/kron) vs TF graphs", "operator", ["init_rel", "fwd_rel", "grad_rel", "adam_rel"]),
    ("ista/vs_tf", "LISTA / ALISTA vs ALISTA-repo TF", "net", ["fwd_rel", "grad_rel", "adam_rel"]),
    ("lvamp/vs_tf", "LVAMP vs Borgerding TF", "precision", ["fwd_rel", "grad_rel", "adam_rel"]),
    ("alista/W_vs_octave", "ALISTA analytic W vs CalculateW.m", None, ["rel"]),
    ("admmnet/vs_original", "ADMM-Net vs THADMM-Net repo (PyTorch)", None, ["fwd_rel", "grad_rel", "adam_rel"]),
    ("classical/vs_octave", "AMP / OMP / SD / ADMM ports vs .m files", None, ["amp_rel", "omp_rel", "sd_rel", "admm_rel"]),
    ("export/roundtrip", "Export .mat -> MATLAB bsce_infer.m vs PyTorch", "model", ["mat_reload_rel", "matlab_rel"]),
    ("export/original_matlab", "Export _orig.mat -> ORIGINAL LAMP.m/GM_LAMP.m/LADMM_test.m", "model", ["rel"]),
    ("ladmm_proposed/reduces_to_admm", "Proposed LADMM (init) vs classical ADMM", "operator", ["rel"]),
    ("ista/complex_consistency", "Complex LISTA/ALISTA on real data == real version", "net", ["rel"]),
    ("train/do_training", "Original do_training loop, params after every stage (exact part)", "case", ["exact_param_rel", "exact_dnmse_db"]),
    ("train/do_training_f32", "LADMM_train.py loop (float32): final NMSE difference [dB]", "case", ["final_dnmse_db"]),
    ("gpu/vs_cpu", "GPU vs CPU (outputs, grads, Adam, training, export)", "model",
     ["fwd_rel", "grad_rel", "adam_rel", "train_param_rel", "export_pt_rel"]),
]


def main():
    rows = [json.loads(l) for l in (RES / "equivalence.jsonl").read_text(encoding="utf-8").splitlines() if l.strip()]
    latest = {}
    for r in rows:   # keep the latest record of each (test, parameters)
        key = json.dumps({k: v for k, v in r.items()
                          if not (isinstance(v, float) or k.startswith("stage_")) or k in ("snr", "snr_db", "lam", "rho")},
                         sort_keys=True)
        latest[key] = r
    rows = [r for r in latest.values()   # drop records of superseded test versions and GPU dry runs
            if not (r["test"] == "train/do_training" and "n_param_exact_stages" not in r)
            and not (r["test"] == "gpu/vs_cpu" and r.get("device") == "cpu")]
    for r in rows:   # derived metrics of the training-loop tests
        if r["test"] == "train/do_training":
            r["exact_param_rel"] = max(r["stage_param_rel"][:r["n_param_exact_stages"]])
            r["exact_dnmse_db"] = max(r["stage_dnmse_db"][:r["n_exact_stages"]])
        if r["test"] == "train/do_training_f32":
            r["final_dnmse_db"] = abs(r["final_nmse_db_tf"] - r["final_nmse_db_torch"])
    out, md = [], ["| Check | Group | n | " + "worst relative error | " + "|", "|---|---|---|---|"]
    for tid, label, gkey, metrics in GROUPS:
        rr = [r for r in rows if r["test"] == tid]
        if not rr:
            continue
        groups = defaultdict(list)
        for r in rr:
            groups[str(r.get(gkey, "")) + (f" ({r['precision']})" if "precision" in r and gkey != "precision" else "")].append(r)
        for g, items in sorted(groups.items()):
            worst = {m: max(float(it.get(m, 0.0)) for it in items) for m in metrics}
            out.append({"check": label, "group": g, "n": len(items), **worst})
            md.append(f"| {label} | {g} | {len(items)} | " + ", ".join(f"{m}={v:.1e}" for m, v in worst.items()) + " |")
    with open(RES / "summary.csv", "w", newline="", encoding="utf-8") as f:
        keys = sorted({k for o in out for k in o}, key=lambda k: (k not in ("check", "group", "n"), k))
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        w.writerows(out)
    (RES / "summary.md").write_text("\n".join(md) + "\n", encoding="utf-8")
    print("\n".join(md))
    plot(out)
    plot_training_loops(rows)


def plot_training_loops(rows):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    rr = [r for r in rows if r["test"] in ("train/do_training", "train/do_training_f32")]
    if not rr:
        return
    pal = {"GM-LAMP original init": "#e34948", "GM-LAMP spread init": "#2a78d6", "LAMP": "#1baf7a",
           "LADMM-full": "#eda100", "LADMM-kron": "#6250d6", "LADMM-diag": "#e87ba4", "LADMM-full float32": "#52514e"}
    fig, ax = plt.subplots(1, 2, figsize=(13, 5.4))
    for r in rr:
        c = pal.get(r["case"], "#888888")
        v = np.maximum(np.array(r["stage_param_rel"]), 1e-17)
        st = np.arange(1, len(v) + 1)
        ls = "--" if r["test"].endswith("f32") else "-"
        ax[0].semilogy(st, v, ls, color=c, lw=1.8, marker="o", ms=3, label=f"{r['case']} ({r['script']})")
        if v[-1] > 1e-14:
            ax[0].annotate(f"{v[-1]:.0e}", (st[-1], v[-1]), textcoords="offset points", xytext=(4, -3), fontsize=7, color=c)
        d = np.maximum(np.array(r["stage_dnmse_db"]), 1e-9)
        ax[1].semilogy(st, d, ls, color=c, lw=1.8, marker="o", ms=3, label=r["case"])
        if d[-1] > 1e-5:
            ax[1].annotate(f"{d[-1]:.1e} dB", (st[-1], d[-1]), textcoords="offset points",
                           xytext=(4, -11 if r["test"].endswith("f32") else 2), fontsize=7, color=c)
    ax[0].axhline(2.2e-16, color="#52514e", ls=":", lw=0.8)
    exact = [max(r["stage_param_rel"]) for r in rr if r["test"] == "train/do_training" and r["n_param_exact_stages"] == r["n_stages"]]
    if exact:
        ax[0].text(16.5, 2e-13, f"exact cases: worst {max(exact):.0e} over all stages", fontsize=7, color="#52514e")
    ax[0].set_xlabel("training stage (extend / refine, in order)")
    ax[0].set_ylabel("max relative parameter difference vs original")
    ax[0].set_title("Parameters after every stage of the original do_training()", fontsize=9)
    ax[1].axhline(5e-7, color="#52514e", ls=":", lw=0.8)
    ax[1].text(1, 6e-7, "print resolution of the original log (6 decimals)", fontsize=7, color="#52514e")
    ax[1].set_xlabel("training stage"); ax[1].set_ylabel("|validation NMSE difference| (dB)")
    ax[1].set_title("Validation NMSE logged at the end of every stage", fontsize=9)
    for a in ax:
        a.grid(alpha=0.3)
        for sp in ("top", "right"):
            a.spines[sp].set_visible(False)
    h, l = ax[0].get_legend_handles_labels()
    fig.legend(h, l, frameon=False, fontsize=7.5, loc="lower center", ncol=4)
    fig.tight_layout(rect=(0, 0.1, 1, 1))
    fig.savefig(RES / "training_loop_equivalence.png", dpi=180)


def plot(out):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    labels, vals, cols = [], [], []
    for o in out:
        ms = [m for m in o if m not in ("check", "group", "n") and m.endswith("rel")]
        if not ms:      # metrics in dB (float32 training loop) are reported in the table only
            continue
        v = max(o[m] for m in ms)
        labels.append(f"{o['check'][:64]} [{o['group']}]".replace(" []", ""))
        vals.append(v)
    y = np.arange(len(vals))
    fig, ax = plt.subplots(figsize=(12, 0.27 * len(vals) + 1.2))
    c = ["#2a78d6" if v < 1e-9 else "#eda100" for v in vals]
    ax.barh(y, [max(v, 1e-17) for v in vals], color=c, height=0.7)
    for yi, v in zip(y, vals):
        ax.text(max(v, 1e-17) * 1.4, yi, f"{v:.1e}" if v > 0 else "0 (bit-identical)", va="center", fontsize=7)
    ax.set_xscale("log"); ax.set_xlim(1e-17, 1e-1)
    ax.axvline(2.2e-16, color="#52514e", ls=":", lw=0.8)
    ax.text(2.4e-16, -0.9, "float64 eps", fontsize=7, color="#52514e")
    ax.set_yticks(y); ax.set_yticklabels(labels, fontsize=7); ax.set_ylim(len(vals) - 0.4, -0.6)
    ax.set_xlabel("worst relative error, PyTorch/numpy port vs original code (log scale)")
    ax.set_title("Equivalence checks (blue: < 1e-9, float64; amber: float32 references)", fontsize=9)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    fig.tight_layout()
    fig.savefig(RES / "equivalence_summary.png", dpi=180)


if __name__ == "__main__":
    sys.exit(main())

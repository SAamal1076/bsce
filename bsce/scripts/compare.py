#!/usr/bin/env python3
"""Fair comparison of all trained networks: evaluation on the test set + figures + tables, in results/<name>/.

  %run scripts/compare.py --scenarios all
  %run scripts/compare.py --scenarios all 3gpp          (+ 3gpp_upa, 3gpp_ula: gains against a GM-LAMP trained here)
  %run scripts/compare.py --scenarios sv_upa deepmimo_upa --tags "*_ft_*" --name finetuned
  (fine-tuned folders: *_ft_* = the v24 names <run>_ft_<edges>dB_<k>; "*-ft" the v23 names, or scripts/rename_runs.py)
  %run scripts/compare.py --scenarios all --tags "LAMP-T8_paper_trscenario" "GM-LAMP-T8_paper_trscenario" --name paper_check

For every scenario:
  1. the authors' published GM-LAMP and LAMP networks are put in outputs/<SCENARIO>/ (GM-LAMP-authors, LAMP-authors)
     (import_gmlamp_weights.py; skipped with --no-authors or when the folders exist)
  2. every selected network is evaluated on the scenario's test set: the same 2000 test channels and the same noise
     realisation at every SNR for every method (evaluate.py)
Then, for all scenarios together:
  comparison.png / .pdf  one column per scenario: NMSE vs SNR, a table (0 / 10 / 20 dB, mean, gain over the reference,
                         parameters), the gain over the reference at every SNR (values printed)
  summary.csv            scenario, method, NMSE at every SNR, mean, gain over the reference, parameters, inference time,
                         v31: trained parameters (those that reach the output) and real MACs per channel (bsce/cost.py)
  summary.md / .tex      the table at 0 / 10 / 20 dB + mean + gain (paper-ready)
  networks.md            what every compared network is (method, T, protocol, data, bands), from its README / logs
  <SCENARIO>/            evaluate.py output (nmse_vs_snr.csv / .png, nmse_table.png, eval_log.json)

--tags        run folders to compare (names or glob patterns). Default: every folder in outputs/<SCENARIO>/
              (+ the authors' networks). The reference is always added.
--preset      (v30) the usual selections, the same as the workbench's: ft = every per-band fine-tune
              (*_ft_*, and *-ft of v23 and older); onenet = every one-network-0-20 dB run, whatever its suffix
              (G-LADMM-T8_paper, LADMM-T6_fair, ...), without fine-tunes, other training ranges and ensembles.
              --tags and --exclude add to it.
--ref         reference for the gains (default GM-LAMP-authors)
--classical   classical baselines evaluated as well (default amp; also omp, sd; v31: l1, grwl1 tuned per SNR by
              scripts/tune_classical.py, --classical-params)
--noise-model white   the networks trained with white noise (outputs/<SCENARIO>_white/, run_suite --noise-model white),
              tested with white noise. The published GM-LAMP / LAMP were trained with colored noise, so they are not
              imported there: the reference is then a GM-LAMP trained here with white noise (--ref to change it).
              Both noise models side by side, with cross tests: scripts/compare_noise.py.
"""
import argparse
import csv
import fnmatch
import json
import re
import subprocess
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from bsce.config import load_yaml, resolve, scenario_run_name  # noqa: E402
from bsce.classical import METHODS as CLASSICAL  # noqa: E402  (v31: one list)
from bsce.data import noise as noise_models  # noqa: E402

SCEN = {"sv_ula": "configs/scenarios/sv_ula.yaml", "sv_upa": "configs/scenarios/sv_upa.yaml",
        "deepmimo_ula": "configs/scenarios/deepmimo_ula.yaml", "deepmimo_upa": "configs/scenarios/deepmimo_upa.yaml"}
SCEN_3GPP = {"3gpp_upa": "configs/scenarios/3gpp_upa.yaml", "3gpp_ula": "configs/scenarios/3gpp_ula.yaml"}
# reference for the gains where the authors published no network (3GPP data): a GM-LAMP trained here
REF_FALLBACK = ["GM-LAMP-T8_ft_0-10-20dB_*", "GM-LAMP-T8-ft", "GM-LAMP-T8", "GM-LAMP-T8_gmlamp_trscenario",
                "GM-LAMP-T8_paper_trscenario"]          # names or patterns, in this order
# v30: --preset, the same patterns as the workbench (app/bsce_app.py TAG_PRESETS / TAG_EXCLUDE; a test keeps them equal)
PRESETS = {"ft": (["*_ft_*", "*-ft", "*-ft[0-9]*"], []),
           "onenet": (["*-T[0-9]*"], ["*_ft_*", "*-ft", "*-ft[0-9]*", "*_trscenario*", "*_tr[0-9m]*", "ENS-*"])}
PALETTE = ["#7b3fb5", "#1f9e6e", "#e8743b", "#b8860b", "#17becf", "#8c564b", "#e377c2", "#7f7f7f",
           "#bcbd22", "#2ca02c", "#9467bd", "#d62728", "#ff7f0e", "#1f77b4"]


def run(cmd):
    print("\n>>> " + " ".join(str(c) for c in cmd[1:]), flush=True)
    r = subprocess.run([str(c) for c in cmd], cwd=ROOT)
    if r.returncode:
        sys.exit(f"failed: {' '.join(str(c) for c in cmd[1:])}")


def describe(folder):
    """One line about a run folder, from its logs."""
    f = Path(folder)
    try:
        if (f / "finetune_log.json").exists():
            L = json.loads((f / "finetune_log.json").read_text(encoding="utf-8"))
            a = L.get("args", {})
            return (f"fine-tuned from {Path(L['source']).parent.name}, bands {', '.join(L['bands'])}, "
                    f"{a.get('iters')} it./band, lr {a.get('lr')} -> {a.get('lr2')}")
        if (f / "train_log.json").exists():
            L = json.loads((f / "train_log.json").read_text(encoding="utf-8"))
            c, sc = L.get("config", {}), L.get("scenario", {})
            m, t = c.get("model", {}), c.get("train", {})
            opts = [f"{k}={m[k]}" for k in ("operator", "threshold", "data_step", "zstep", "xstep", "sigma") if k in m]
            sched = t.get("schedule")
            if sched == "steps":                          # v30: the training budget and setting of end-to-end runs
                sched = f"end to end, {t.get('steps')} steps"
            extra = [f"{k} {t[k]}" for k in ("loss", "noise") if t.get(k)]
            if t.get("combiner") == "random":
                extra.append("a random combiner every minibatch")
            return (f"{m.get('model')} T={m.get('T')}" + (f" ({', '.join(opts)})" if opts else "")
                    + f", {sched}, lr {t.get('lr')}, batch {t.get('batch_size')}"
                    + (f", {', '.join(extra)}" if extra else "")
                    + f", {m.get('precision')}, data {sc.get('name')} ({sc.get('data_dir')}), bands {', '.join(L['bands'])}")
        if f.name.endswith("-authors"):
            return "published network of Wei, Hu & Dai (two networks, 0-10 and 10-20 dB)"
        if (f / "README.txt").exists() and "ensemble" in (f / "README.txt").read_text(encoding="utf-8")[:200]:
            r = (f / "README.txt").read_text(encoding="utf-8").splitlines()
            mem = [Path(x.strip()).name for x in r[r.index("members:") + 1:] if x.strip()] if "members:" in r else []
            return "ensemble (mean of the outputs) of " + ", ".join(mem)
    except Exception as e:  # noqa: BLE001
        return f"(log unreadable: {e})"
    return "-"


def n_params(folder):
    """-> (parameters of one network, number of networks in the folder), e.g. (196617, 2) for a band pair."""
    p = sorted(Path(folder).glob(f"{Path(folder).name}_*.pt"))
    if not p:
        return None
    from bsce.export import load_model
    return int(load_model(p[0]).n_params()), len(p)


def cost(folder):
    """v31: (trained parameters of one network, real MACs per channel at batch 1) of the first network of a folder
    (bsce/cost.py): the parameters that reach the output, and the cost of one channel as the network runs."""
    p = sorted(Path(folder).glob(f"{Path(folder).name}_*.pt"))
    if not p:
        return None
    from bsce.cost import macs_per_channel, trained_parameters
    from bsce.export import load_model
    net = load_model(p[0])
    tr, _, _ = trained_parameters(net)
    macs, _ = macs_per_channel(net, batch=1)
    return tr, macs


def distinct_colors(n):
    """n colours, all different: the palette, then tab20 / tab20b (v30: more than 14 methods no longer repeat colours)."""
    if n <= len(PALETTE):
        return PALETTE[:n]
    import matplotlib
    extra = [matplotlib.colors.to_hex(c) for name in ("tab20", "tab20b", "tab20c")
             for c in matplotlib.colormaps[name].colors]
    out = list(dict.fromkeys(PALETTE + [c for c in extra if c not in ("#7f7f7f", "#c7c7c7")]))
    while len(out) < n:                                   # more than ~70 methods: evenly spaced hues
        k = len(out)
        out.append(matplotlib.colors.to_hex(matplotlib.colors.hsv_to_rgb(((k * 0.618) % 1, 0.65, 0.8))))
    return out[:n]


def fmt_params(np_, tex=False):
    if np_ is None:
        return ""
    n, k = np_
    if k == 1:
        return str(n) if tex else f"{n:,}"
    return (rf"{k}$\times${n}" if tex else f"{k} × {n:,}")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--scenarios", nargs="+", required=True)
    ap.add_argument("--tags", nargs="*", default=None)
    ap.add_argument("--preset", choices=sorted(PRESETS), default=None,
                    help="ft: every fine-tune; onenet: every one-network-0-20 dB run (any suffix); adds to --tags")
    ap.add_argument("--exclude", nargs="*", default=None,
                    help="leave out the run folders matching these patterns, e.g. '*_ft_*' (passed to evaluate.py)")
    ap.add_argument("--ref", default="GM-LAMP-authors")
    ap.add_argument("--no-authors", action="store_true")
    ap.add_argument("--classical", default="amp")
    ap.add_argument("--classical-params", default="results/classical_params_<SCENARIO>.json",
                    help="v31: for --classical l1 / grwl1, the tune_classical.py file of every scenario (<SCENARIO> is "
                         "replaced by the scenario name)")
    ap.add_argument("--no-cost", action="store_true",
                    help="v31: leave out the trained-parameter and MACs-per-channel columns (bsce/cost.py)")
    ap.add_argument("--name", default="comparison")
    ap.add_argument("--outputs", default="outputs")
    ap.add_argument("--results", default="results")
    ap.add_argument("--device", default="auto")
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--polish", choices=["none", "band", "snr"], default="none",
                    help="v28: also every network + least squares on its strongest beams ('<run>+polish' columns; "
                         "passed to evaluate.py)")
    ap.add_argument("--noise-model", default="colored", choices=["colored", "coloured", "folded", "white"],
                    help="colored (default): outputs/<SCENARIO>/ on colored test noise; white: outputs/<SCENARIO>_white/ "
                         "on white test noise")
    a = ap.parse_args()
    if a.preset:
        tg, ex = PRESETS[a.preset]
        a.tags = list(dict.fromkeys(tg + list(a.tags or [])))
        a.exclude = list(dict.fromkeys(ex + list(a.exclude or []))) or None
    py = sys.executable
    nm = noise_models.canonical(a.noise_model)
    scen = [p for s in a.scenarios for p in (SCEN.values() if s == "all" else SCEN_3GPP.values() if s == "3gpp" else
                                               [SCEN.get(s.lower(), s if s.endswith(".yaml") else f"configs/scenarios/{s.lower()}.yaml")])]
    res = resolve(a.results) / a.name
    res.mkdir(parents=True, exist_ok=True)
    tables = []
    for s in scen:
        sc = load_yaml(s)
        sc["noise_model"] = nm
        runs = resolve(a.outputs) / scenario_run_name(sc, a.smoke)
        if not runs.exists():
            print(f"[compare] {runs} does not exist, {sc['name']} skipped")
            continue
        published = sc["dataset"] in ("SV", "DeepMIMO") and nm == "colored"   # authors' networks: these data, colored
        if not a.no_authors and published and not (runs / "GM-LAMP-authors").exists():
            run([py, "scripts/import_gmlamp_weights.py", "--scenario", s, "--runs", runs])
        if not a.no_authors and not published:
            print(f"[compare] {sc['name']}: no published GM-LAMP networks for {sc['dataset']} data"
                  + (" with white noise (the published ones were trained with colored noise: compare_noise.py tests them "
                     "on white noise)" if nm == "white" else "")
                  + (f"; gains against a GM-LAMP trained here ({', '.join(REF_FALLBACK)}, in this order) or --ref <tag>"
                     if a.ref.endswith("-authors") else ""))
        tags = None
        if a.tags:
            tags = list(dict.fromkeys(list(a.tags) + [a.ref] + ([] if a.no_authors or not published else ["LAMP-authors"])))
        cmd = [py, "scripts/evaluate.py", "--scenario", s, "--runs", runs, "--out", res / scenario_run_name(sc),
               "--classical", a.classical, "--device", a.device, "--noise-model", nm]
        if tags:
            cmd += ["--tags", *tags]
        if a.exclude:
            cmd += ["--exclude", *a.exclude]
        if a.polish != "none":
            cmd += ["--polish", a.polish]
        if {"l1", "grwl1"} & set(a.classical.split(",")):          # v31: the tuned classical baselines
            cmd += ["--classical-params", str(resolve(a.classical_params.replace("<SCENARIO>", sc["name"])))]
        run(cmd)
        with open(res / scenario_run_name(sc) / "nmse_vs_snr.csv", encoding="utf-8") as f:
            rows = list(csv.DictReader(f))
        snr = [float(r["SNR_dB"]) for r in rows]
        meths = [k for k in rows[0] if k != "SNR_dB"]
        val = {m: np.array([float(r[m]) for r in rows]) for m in meths}
        secs = json.loads((res / scenario_run_name(sc) / "eval_log.json").read_text()).get("seconds", {})
        ref_name = a.ref if a.ref in val else next((m for pat in REF_FALLBACK for m in sorted(val)
                                                    if fnmatch.fnmatchcase(m, pat)),
                                                    next((m for m in val if m.startswith("GM-LAMP-T")), None))
        tables.append(dict(name=scenario_run_name(sc), runs=runs, snr=np.array(snr), val=val, secs=secs, ref=ref_name,
                           n_test=int(sc.get("n_test", 2000))))
    if not tables:
        sys.exit("nothing evaluated")

    # ---- summary tables ---------------------------------------------------------------------------------------
    def at(v, snr, q):
        i = int(np.argmin(np.abs(snr - q)))
        return v[i]

    lines_csv, md, tex, nets = [], [], [], []
    for T in tables:
        ref = T["val"].get(T["ref"]) if T["ref"] else None
        order = sorted(T["val"], key=lambda m: np.mean(T["val"][m]))          # best first
        md += [f"\n### {T['name']}\n", f"| method | 0 dB | 10 dB | 20 dB | mean | vs {T['ref']} | parameters (networks × per network) "
               "| trained parameters per network | real MACs per channel |",
               "|---|---|---|---|---|---|---|---|---|"]
        tex += [f"% {T['name']}", r"\begin{tabular}{lrrrrrr}", r"\hline",
                rf"Method & 0 dB & 10 dB & 20 dB & mean & vs ref & params \\ \hline"]
        nets.append(f"\n### {T['name']} ({T['runs']})\n")
        for m in order:
            v = T["val"][m]
            g = None if ref is None else float(np.mean(ref - v))
            base = re.sub(r"\+polish(-snr)?$", "", m)              # v28: '<run>+polish' = that network + polish
            npar = n_params(T["runs"] / base) if (T["runs"] / base).exists() else None
            cst = None if (a.no_cost or not (T["runs"] / base).exists() or base != m) else cost(T["runs"] / base)
            lines_csv.append([T["name"], m] + [f"{x:.4f}" for x in v] + [f"{v.mean():.4f}", "" if g is None else f"{g:.4f}",
                                                                          "" if npar is None else npar[0],
                                                                          "" if npar is None else npar[1],
                                                                          f"{T['secs'].get(m, 0):.3f}",
                                                                          "" if cst is None else cst[0],
                                                                          "" if cst is None else f"{cst[1]:.0f}"])
            gs = "" if g is None else f"{g:+.2f}"
            md.append(f"| {m} | {at(v, T['snr'], 0):.2f} | {at(v, T['snr'], 10):.2f} | {at(v, T['snr'], 20):.2f} | "
                      f"{v.mean():.2f} | {gs} | {fmt_params(npar)} | {'' if cst is None else f'{cst[0]:,}'} | "
                      f"{'' if cst is None else f'{cst[1]:,.0f}'} |")
            tex.append(f"{m} & {at(v, T['snr'], 0):.2f} & {at(v, T['snr'], 10):.2f} & {at(v, T['snr'], 20):.2f} & "
                       f"{v.mean():.2f} & {gs} & {fmt_params(npar, tex=True)} \\\\")
            if base != m and (T["runs"] / base).exists():
                nets.append(f"- **{m}**: {base} + polish (least squares on its K strongest beams; K and eps chosen on "
                            f"validation channels {'per test SNR' if m.endswith('-snr') else 'per band'}: "
                            f"<SCENARIO>/polish_choices.csv)")
            elif (T["runs"] / m).exists():
                nets.append(f"- **{m}**: {describe(T['runs'] / m)}")
            else:
                nets.append(f"- **{m}**: classical baseline (no training)")
        tex += [r"\hline", r"\end{tabular}", ""]
    snr_cols = [f"{x:g}dB" for x in tables[0]["snr"]]
    with open(res / "summary.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["scenario", "method"] + snr_cols + ["mean", "gain_vs_reference", "parameters_per_network", "networks",
                                                         "inference_seconds", "trained_parameters_per_network",
                                                         "macs_per_channel"])
        w.writerows(lines_csv)
    (res / "summary.md").write_text("# NMSE (dB) on the test sets of the scenarios; gain = mean over all test SNRs vs the "
                                    f"reference of the scenario ({a.ref}; where it does not exist, a GM-LAMP trained here)\n"
                                    + "\n".join(md) + "\n", encoding="utf-8")
    (res / "summary.tex").write_text("\n".join(tex), encoding="utf-8")
    (res / "networks.md").write_text("# The compared networks\n" + "\n".join(nets) + "\n", encoding="utf-8")

    # ---- figure -------------------------------------------------------------------------------------------------
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    allm = list(dict.fromkeys(m for T in tables for m in T["val"]))
    refs = {T["ref"] for T in tables if T["ref"]}
    others = [m for m in allm if m not in refs | {"LAMP-authors", *CLASSICAL}]
    color = dict(zip(others, distinct_colors(len(others))))
    color.update({r: "#2a78d6" for r in refs})
    color.update({"LAMP-authors": "#c23b6a", "amp": "#999999", "omp": "#666666", "sd": "#444444"})
    style = lambda m: "--" if (m.endswith("-authors") or m in CLASSICAL) else "-"  # noqa: E731
    n = len(tables)
    nrow_tab = max(len(T["val"]) for T in tables)
    fig, axs = plt.subplots(3, n, figsize=(7.4 * n, 12 + 0.25 * nrow_tab), squeeze=False,
                            gridspec_kw=dict(height_ratios=[1.3, 0.12 + 0.055 * nrow_tab, 1]))
    for j, T in enumerate(tables):
        ax, tx, bx = axs[0, j], axs[1, j], axs[2, j]
        ref = T["val"].get(T["ref"]) if T["ref"] else None
        order = sorted(T["val"], key=lambda m: np.mean(T["val"][m]))          # best (lowest NMSE) first
        for m in order:
            v = T["val"][m]
            ax.plot(T["snr"], v, style(m), marker="o", ms=3, lw=1.5, color=color[m], label=m)
            if ref is not None and m != T["ref"] and m not in CLASSICAL:
                g = ref - v
                bx.plot(T["snr"], g, style(m), marker="o", ms=3, lw=1.4, color=color[m], label=f"{m} ({g.mean():+.2f})")
        cand = [m for m in order if m != T["ref"] and m not in CLASSICAL and not m.endswith("-authors")]
        best = cand[0] if cand else None
        if ref is not None and best is not None:
            g = ref - T["val"][best]
            for xi, gi in zip(T["snr"], g):
                bx.annotate(f"{gi:+.2f}", (xi, gi), xytext=(0, 6), textcoords="offset points", ha="center", fontsize=6.5,
                            color=color[best])
        ax.set_title(f"{T['name']} (test set: {T['n_test']:,} channels)", fontsize=11)
        ax.set_xlabel("SNR (dB)"); ax.set_ylabel("NMSE (dB)"); ax.grid(alpha=.3)
        if len(order) > 8:                                # v30: many methods: one legend, under the gain panel (it would
            ax.text(0.99, 0.99, f"dashed: {T['ref']} (blue), the authors' LAMP, classical (grey)\n"   # cover the curves)
                    "other colours: the legend under the gain panel", transform=ax.transAxes, ha="right", va="top",
                    fontsize=7, color="#555555")
        else:
            ax.legend(fontsize=7, frameon=False, loc="upper right")
        tx.axis("off")
        cells = []
        for m in order:
            v = T["val"][m]
            g = "" if ref is None else f"{np.mean(ref - v):+.2f}"
            cells.append([m, f"{at(v, T['snr'], 0):.2f}", f"{at(v, T['snr'], 10):.2f}", f"{at(v, T['snr'], 20):.2f}",
                          f"{v.mean():.2f}", g])
        tb = tx.table(cellText=cells, colLabels=["method", "0 dB", "10 dB", "20 dB", "mean", f"vs ref"], loc="center",
                      colWidths=[.42, .1, .1, .1, .1, .12], cellLoc="center")
        tb.set_fontsize(7.5); tb.scale(1, 1.15)
        for i, m in enumerate(order):
            tb[i + 1, 0].get_text().set_color(color[m])
        bx.axhline(0, color="#2a78d6", lw=1, ls="--")
        short = best if best is None or len(best) <= 48 else best[:45] + "…"
        bx.set_title(f"gain over {T['ref']} at each SNR\n(values printed: the best method, {short})", fontsize=9.5)
        bx.set_xlabel("SNR (dB)"); bx.set_ylabel("gain (dB, > 0 = better)"); bx.grid(alpha=.3)
        bx.legend(fontsize=6.8, frameon=False, loc="upper center", bbox_to_anchor=(.5, -.13), ncol=2)
        for x_ in (ax, bx):
            for sp in ("top", "right"):
                x_.spines[sp].set_visible(False)
    fig.tight_layout()
    fig.savefig(res / "comparison.png", dpi=140, bbox_inches="tight")      # v30: legends outside the axes are kept
    fig.savefig(res / "comparison.pdf", bbox_inches="tight")
    (res / "README.md").write_text(
        f"# Comparison '{a.name}'\n\nMade by `scripts/compare.py --scenarios {' '.join(a.scenarios)}"
        + (f" --tags {' '.join(a.tags)}" if a.tags else "") + f"`.\n\n"
        "| file | content |\n|---|---|\n"
        "| `comparison.png` / `.pdf` | per scenario: NMSE vs SNR, table, gain over the reference at every SNR |\n"
        "| `summary.csv` | NMSE at every SNR, mean, gain, parameters, inference time, trained parameters, MACs per channel |\n"
        "| `summary.md` / `summary.tex` | 0 / 10 / 20 dB, mean, gain (paper table) |\n"
        "| `networks.md` | what each compared network is (method, depth, protocol, data, bands) |\n"
        "| `<SCENARIO>/` | evaluate.py output per scenario |\n\n"
        "Test: the scenario's 2000 test channels; the noise at every SNR is identical for all methods "
        f"(NumpyMatlabRNG(seed + 1000 + i)); noise model: {nm}.\n", encoding="utf-8")
    print(f"\nwritten {res}: comparison.png/pdf, summary.csv/md/tex, networks.md, README.md")


if __name__ == "__main__":
    main()

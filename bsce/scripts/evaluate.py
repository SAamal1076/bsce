#!/usr/bin/env python3
"""Evaluate trained models (and classical baselines) on the test channels of a scenario.

  python scripts/evaluate.py --scenario configs/scenarios/sv_ula.yaml --runs outputs/SV_ULA \
         --classical amp,omp,sd [--admm-lambdas path.mat] [--device auto]
         --classical amp,l1,grwl1 --classical-params results/classical_params_<SCENARIO>.json   (v31, tune_classical.py)
For every SNR the same noise realisation is used for all methods (MATLAB-style noise).
Noise model (--noise-model): colored y = A (x + n) or white y = A x + w, the same draw whitened (bsce/data/noise.py);
default auto = white for a runs folder <scenario>_white (made by --noise-model white), else the scenario's (colored).
Cross test: e.g. --runs outputs/SV_ULA --noise-model white --out results/SV_ULA_on_white (colored-trained networks on
white test noise). --nf-step matched: NF-LADMM networks use the exact data step of the TEST noise model (same weights)
instead of the one they were trained with.
For each model folder, the band whose `use_for` range contains the SNR is used.
--polish band|snr (v28, default none): also evaluate every network followed by a least-squares correction on its K
strongest beams (bsce/polish.py), as an extra column '<run>+polish' (band) or '<run>+polish-snr'. K and eps are
chosen on the scenario's validation channels (own noise draws), never on the test set: band = one choice per network
band (5 SNRs over the band), snr = one choice per test SNR (the true SNR is known). The choices are in eval_log.json
and polish_choices.csv.
Other combiner or data set (v29): --combiner <file.mat with A> | random:<seed> tests with that combiner instead of the
scenario's. Networks trained on another scenario (e.g. --runs outputs/SV_ULA with --scenario deepmimo_ula, whose
combiner differs) are tested on this scenario's combiner the same way. A network that can follow a new combiner
(analytic LADMM, Graph-LADMM with xstep=analytic) uses it, with any number of RF chains M (v30); the others keep the
combiner they were learned for (that is the transfer test), and are skipped when the test combiner has another M (they
cannot run). eval_log.json says which. --graph-phase target: analytic Graph-LADMM networks take the edge
phases of this data set (estimated from its validation channels: the array's index convention) instead of their own.
Graph-LADMM with xstep=free keeps its phases: its H_t was learned with them.
Writes nmse_vs_snr.csv, nmse_vs_snr.png (values printed on the curves), nmse_table.png
(methods x SNR with every value printed) and eval_log.json.
"""
import argparse
import csv
import json
import sys
import time
from pathlib import Path

import numpy as np
import scipy.io as sio
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from bsce.config import load_yaml, resolve, scenario_cs_matrix, scenario_test_dir  # noqa: E402
from bsce.device import get_device  # noqa: E402
from bsce.export import load_model  # noqa: E402
from bsce.data import measurements  # noqa: E402
from bsce.rng import NumpyMatlabRNG  # noqa: E402
from bsce import classical  # noqa: E402
from bsce.data.files import band_range_from_name  # noqa: E402
from bsce.data import noise as noise_models  # noqa: E402


def nmse_db(xh, x):
    return 10 * np.log10(np.mean(np.sum(np.abs(xh - x) ** 2, 0)) / np.mean(np.sum(np.abs(x) ** 2, 0)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scenario", required=True)
    ap.add_argument("--runs", required=True)
    ap.add_argument("--classical", default="amp")
    ap.add_argument("--admm-lambdas", default=None, help=".mat with 'lambdas' (one per test SNR)")
    ap.add_argument("--classical-params", default=None,
                    help="v31: JSON of scripts/tune_classical.py (lam and eps per test SNR) for --classical l1 / grwl1")
    ap.add_argument("--device", default="auto")
    ap.add_argument("--n-test", type=int, default=None)
    ap.add_argument("--out", default=None)
    ap.add_argument("--tags", nargs="*", default=None,
                    help="only these run folders (names or glob patterns, e.g. '*-ft' 'GM-LAMP-authors')")
    ap.add_argument("--exclude", nargs="*", default=None,
                    help="leave out the run folders matching these names or patterns (after --tags), e.g. '*_ft_*'")
    ap.add_argument("--noise-model", default="auto", choices=["auto", "colored", "coloured", "folded", "white"],
                    help="test noise: auto (default) = white for a runs folder <scenario>_white, else the scenario's "
                         "noise_model (colored); colored = y = A (x + n); white = y = A x + w (same draw, whitened)")
    ap.add_argument("--polish", choices=["none", "band", "snr"], default="none",
                    help="v28: also evaluate every network + least squares on its K strongest beams (columns "
                         "'<run>+polish'); K, eps chosen on validation channels per band (band) or per test SNR (snr)")
    ap.add_argument("--polish-n-val", type=int, default=None, help="validation channels for the polish choice (default all)")
    ap.add_argument("--combiner", default=None,
                    help="v29: test with another combiner: a .mat file with A (M x N), or random:<seed>[:<M>] (+-1/sqrt(M); "
                         "v30: :<M> gives another number of RF chains, e.g. random:11:64)")
    ap.add_argument("--graph-phase", choices=["own", "target"], default="own",
                    help="v29: Graph-LADMM edge phases: own (as trained) or target (estimated from this data set)")
    ap.add_argument("--nf-step", choices=["trained", "matched"], default="trained",
                    help="NF-LADMM data step: trained (default) = the noise model the network was trained with; "
                         "matched = the exact step of the test noise model (same weights, no retraining)")
    a = ap.parse_args()
    sc = load_yaml(a.scenario)
    runs = resolve(a.runs)
    smoke = runs.name.endswith("_smoke")
    stem = runs.name[:-len("_smoke")] if smoke else runs.name
    nm = (noise_models.canonical(a.noise_model) if a.noise_model != "auto" else
          "white" if stem.endswith("_white") else noise_models.of_scenario(sc))
    print(f"[bsce] test noise: {nm}" + (" (y = A x + w, same mean power)" if nm == "white" else " (y = A (x + n))"))
    pre = f"{sc['dataset']}_{sc['array']}"
    A = sio.loadmat(scenario_cs_matrix(sc, smoke))["A"].astype(np.float64)
    if a.combiner:
        if a.combiner.startswith("random:"):             # random:<seed>[:<M>] (v30: another number of RF chains)
            parts = a.combiner.split(":")
            if len(parts) not in (2, 3) or not all(x.isdigit() for x in parts[1:]) or (len(parts) == 3 and int(parts[2]) < 1):
                sys.exit(f"--combiner {a.combiner}: write random:<seed> or random:<seed>:<M>, e.g. random:11 or random:11:64")
            g = np.random.default_rng(int(parts[1]))
            shape = (int(parts[2]) if len(parts) == 3 else A.shape[0], A.shape[1])
            A = ((g.random(shape) > 0.5) * 2 - 1) / np.sqrt(shape[0])
        else:
            A = sio.loadmat(resolve(a.combiner))["A"].astype(np.float64)
        print(f"[bsce] combiner: {a.combiner} ({A.shape[0]} x {A.shape[1]})")
    x = sio.loadmat(scenario_test_dir(sc, smoke) / f"{pre}_test_channel.mat")["x"]
    if a.n_test:
        x = x[:, :a.n_test]
    dev = get_device(a.device)
    models, comb_note = {}, {}
    known = {b["name"]: b["use_for"] for b in sc.get("eval_bands", []) + sc["bands"] if "use_for" in b}
    import fnmatch
    for d in sorted(p for p in runs.iterdir() if p.is_dir()):
        if a.tags and not any(fnmatch.fnmatchcase(d.name, pat) for pat in a.tags):
            continue
        if a.exclude and any(fnmatch.fnmatchcase(d.name, pat) for pat in a.exclude):
            continue
        # every <folder>/<folder>_<band>.pt is used: band ranges from the scenario, else from names like 0to20dB
        found = []
        for f in sorted(d.glob(f"{d.name}_*.pt")):
            bname = f.stem[len(d.name) + 1:]
            rng = known.get(bname)
            if rng is None:
                r = band_range_from_name(bname)
                if r is None:
                    continue
                rng = [r[0], r[1]]
            found.append([bname, list(rng), f])
        if not found:
            continue
        top = max(r[1] for _, r, _ in found)
        for b in found:                                  # the band ending at the top SNR includes it
            if b[1][1] == top and b[0] not in known:
                b[1][1] = top + 0.01
        # narrowest band first: a band pair is preferred to a wide network in the same folder
        found.sort(key=lambda b: b[1][1] - b[1][0])
        models[d.name] = [(rng, load_model(f, device=dev)) for _, rng, f in found]
        for _, net in models[d.name]:                    # v29: another combiner than the network's own
            comb_note[d.name] = adapt_to(net, A, a.graph_phase, sc, smoke, pre)
        m_net = sorted({int(getattr(net, "M", A.shape[0])) for _, net in models[d.name]})
        if m_net != [A.shape[0]]:                        # v30: learned for another number of RF chains: skip, not crash
            comb_note[d.name] = (f"cannot run (learned for M = {', '.join(map(str, m_net))}; the test combiner has "
                                 f"M = {A.shape[0]})")
            print(f"[bsce] {d.name}: {comb_note[d.name]}: skipped")
            del models[d.name]
            continue
        if a.nf_step == "matched":
            for _, net in models[d.name]:
                if getattr(net, "family", "") == "nfladmm":
                    net.noise_model = nm                 # data step of the test noise (weights unchanged)
        print(f"[bsce] {d.name}: " + ", ".join(f"{n} -> [{r[0]:g}, {r[1]:g})" for n, r, _ in found))
    lams = sio.loadmat(a.admm_lambdas)["lambdas"].ravel() if a.admm_lambdas else None
    cpar = json.loads(Path(a.classical_params).read_text(encoding="utf-8")) if a.classical_params else None
    if {"l1", "grwl1"} & set(a.classical.split(",")):
        if cpar is None:
            sys.exit("--classical l1 / grwl1 need --classical-params (scripts/tune_classical.py writes it)")
        miss = [s for s in sc["test_snr_db"] if f"{s:g}" not in cpar]
        if miss:
            sys.exit(f"--classical-params has no values for the test SNRs {miss} (run tune_classical.py for them)")
    methods = [m for m in a.classical.split(",") if m] + list(models)
    pol_choice, pol_rows, pol, psfx = {}, [], None, {"band": "+polish", "snr": "+polish-snr"}.get(a.polish)
    if psfx and models:
        pol, pol_choice, pol_rows = choose_polish(a, sc, smoke, pre, A, nm, models)
        methods += [m + psfx for m in models]
    rows, timing = [], {m: 0.0 for m in methods}
    for i, snr in enumerate(sc["test_snr_db"]):
        y = measurements.gen_measurements_matlab(x, A, snr, NumpyMatlabRNG(int(sc["seed"]) + 1000 + i), noise_model=nm)
        row = {"SNR_dB": snr}
        xhat = {}
        for m in methods:
            t0 = time.time()
            if m == "amp":
                xh = classical.amp(y, A)
            elif m == "omp":
                xh = classical.omp(y, A, 24)
            elif m == "sd":
                xh = classical.sd(y, A, sc["L"], 8)
            elif m == "admm":
                if lams is None:
                    continue
                xh, _ = classical.admm(y, A, float(lams[i]), 1.0, 500, 1e-5)
            elif m == "l1":                                  # v31: tuned per SNR (tune_classical.py), 300 iterations
                xh = classical.l1_admm(y, A, float(cpar[f"{snr:g}"]["lam_l1"]))
            elif m == "grwl1":                               # v31: graph-reweighted l1, tuned per SNR
                p_ = cpar[f"{snr:g}"]
                xh = classical.grwl1(y, A, float(p_["lam_rw"]), float(p_["eps"]), array=sc.get("array", "ULA"),
                                     N1=int(sc.get("N1", 16)), N2=int(sc.get("N2", 16)))
            elif psfx and m.endswith(psfx) and m[:-len(psfx)] in models:
                base = m[:-len(psfx)]
                K, eps = pol_choice[base][snr if a.polish == "snr" else band_index(models[base], snr)]
                xh = pol.polish(xhat[base], y, K, eps)
            else:
                xh = run_net(pick_net(models[m], snr), y, snr)
                xhat[m] = xh
            timing[m] += time.time() - t0
            row[m] = nmse_db(xh, x)
        rows.append(row)
        print("  ".join([f"SNR={snr:5.1f}"] + [f"{k}={v:7.2f}" for k, v in row.items() if k != "SNR_dB"]))
    out = resolve(a.out) if a.out else runs
    out.mkdir(parents=True, exist_ok=True)
    with open(out / "nmse_vs_snr.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["SNR_dB"] + [m for m in methods if m in rows[0]])
        w.writeheader()
        w.writerows(rows)
    log = {"seconds": timing, "n_test": x.shape[1], "noise_model": nm, "nf_step": a.nf_step,
           "combiner": a.combiner or "scenario", "network_combiner": comb_note, "graph_phase": a.graph_phase}
    if psfx:
        log["polish"] = {"mode": a.polish, "choices": pol_rows}
        with open(out / "polish_choices.csv", "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=list(pol_rows[0]) if pol_rows else ["run"])
            w.writeheader()
            w.writerows(pol_rows)
    (out / "eval_log.json").write_text(json.dumps(log, indent=1), encoding="utf-8")
    shown = [m for m in methods if m in rows[0]]
    if not shown:                                        # v31: e.g. no network can run with this combiner's M
        print("[bsce] nothing to evaluate here (no classical method and no network that can run): no figure")
        return
    plot(rows, shown, out / "nmse_vs_snr.png",
         sc["name"] + (f", {nm} noise" if nm == "white" or stem.endswith("_white") else ""))


_VAL_X = {}


def adapt_to(net, A, graph_phase, sc, smoke, pre):
    """The test combiner A for one network: 'own' (the same), 'follows' (set_combiner), or 'fixed' (it keeps the
    combiner it was learned for). Analytic Graph-LADMM with --graph-phase target also takes this data set's edge
    phases; with xstep=free it keeps its own (H_t was learned with them, so new phases would not match it)."""
    note = "own"
    if not np.array_equal(np.asarray(net.A_np), A):
        try:
            net.set_combiner(A)
            note = "follows"
        except (AttributeError, ValueError):
            note = "fixed (learned for its own combiner)"
    if (graph_phase == "target" and getattr(net, "family", "") == "gladmm" and net.edge_phase == "data"
            and net.xstep == "analytic"):
        from bsce.config import scenario_data_dir
        from bsce.models.graph_ladmm import estimate_edge_phases
        key = (sc["name"], smoke)
        if key not in _VAL_X:
            _VAL_X[key] = sio.loadmat(scenario_data_dir(sc, smoke) / f"{pre}_validation_channel.mat")["x"]
        est = estimate_edge_phases(_VAL_X[key], net.array, net.N1, net.N2)
        with torch.no_grad():                           # only the phases; (P + rho (I + D^H D))^-1 follows them
            net.phase.copy_(torch.tensor([est[c][0] for c in net.classes], dtype=net.phase.dtype))
        note += ", edge phases of this data set"
    if note != "own":
        print(f"[bsce]   combiner of the test: {note}")
    return note


def pick_net(bands, snr):
    """The network of the band that contains the SNR, else the nearest band."""
    return bands[band_index(bands, snr)][1]


def band_index(bands, snr):
    for i, (rng, _) in enumerate(bands):
        if rng[0] <= snr < rng[1]:
            return i
    return min(range(len(bands)), key=lambda i: min(abs(snr - bands[i][0][0]), abs(snr - bands[i][0][1])))


def run_net(net, y, snr):
    with torch.no_grad():
        if getattr(net, "needs_noise_var", False):
            return net(torch.as_tensor(y), s2=np.full(y.shape[1], 10.0 ** (-snr / 10))).cpu().numpy()
        return net(torch.as_tensor(y)).cpu().numpy()


def choose_polish(a, sc, smoke, pre, A, nm, models):
    """K, eps of the polish for every network (band mode: per band, snr mode: per test SNR), on validation channels.
    Returns (Polisher, {run: {band index or SNR: (K, eps)}}, rows for polish_choices.csv)."""
    from bsce.config import scenario_data_dir
    from bsce.polish import Polisher, band_val_snrs, choose
    f = scenario_data_dir(sc, smoke) / f"{pre}_validation_channel.mat"
    if not f.exists():
        sys.exit(f"--polish needs the validation channels {f} (made with the data set)")
    xv = sio.loadmat(f)["x"]
    if a.polish_n_val:
        xv = xv[:, :a.polish_n_val]
    pol = Polisher(A, nm)
    tsnr = [float(v) for v in sc["test_snr_db"]]
    yv_cache = {}

    def yv(snr):              # validation noise: own seeds, never the test draws (seed + 1000 + i)
        k = round(float(snr), 6)
        if k not in yv_cache:
            yv_cache[k] = measurements.gen_measurements_matlab(
                xv, A, k, NumpyMatlabRNG(int(sc["seed"]) + 7000 + int(round(100 * k)) % 100000), noise_model=nm)
        return yv_cache[k]

    choice, rows = {}, []
    for m, bands in models.items():
        choice[m] = {}
        if a.polish == "band":
            for i, (rng, net) in enumerate(bands):
                lo, hi = max(rng[0], min(tsnr)), min(rng[1], max(tsnr))
                grids = [pol.grid(run_net(net, yv(s), s), yv(s), xv) for s in band_val_snrs(lo, hi)]
                (K, eps), best, base = choose(grids)
                choice[m][i] = (K, eps)
                rows.append(dict(run=m, band=f"[{rng[0]:g}, {rng[1]:g})", snr="", K=K, eps=eps,
                                 val_nmse_db=round(base, 4), val_polished_db=round(best, 4), val_gain_db=round(base - best, 4)))
        else:
            for s in tsnr:
                net = pick_net(bands, s)
                (K, eps), best, base = choose([pol.grid(run_net(net, yv(s), s), yv(s), xv)])
                choice[m][s] = (K, eps)
                rows.append(dict(run=m, band="", snr=s, K=K, eps=eps, val_nmse_db=round(base, 4),
                                 val_polished_db=round(best, 4), val_gain_db=round(base - best, 4)))
        print(f"[polish] {m}: " + ", ".join(f"{r['band'] or r['snr']}: K={r['K']}, eps={r['eps']:g} "
                                             f"(val {r['val_gain_db']:+.2f} dB)" for r in rows if r["run"] == m))
    return pol, choice, rows


def _style(k, m):
    pal = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948",
           "#52514e", "#00a3c4", "#a05a00", "#b03bb0"]
    classical = m in ("amp", "omp", "sd", "admm")
    marks = "osD^v<>Ph*Xd"
    return dict(color=pal[k % len(pal)], ls="--" if classical else "-", marker=marks[(k // len(pal) + k) % len(marks)])


def _repel(vals, gap):
    """1-D label positions: keep the order of vals, at least `gap` apart."""
    order = np.argsort(vals)
    pos = np.array(vals, float)
    for i in range(1, len(order)):
        a, b = order[i - 1], order[i]
        pos[b] = max(pos[b], pos[a] + gap)
    shift = (np.mean(pos) - np.mean(vals))
    return pos - shift


def plot(rows, methods, path, title):
    """v31: a non-finite NMSE (a diverged network) is not drawn and is labelled 'diverged' (it used to crash the label
    placement and stop check_combiners.py)."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    snr = [r["SNR_dB"] for r in rows]
    fig, ax = plt.subplots(figsize=(10, 5.6))
    allv = [float(r[m]) for r in rows for m in methods]
    fin = [v for v in allv if np.isfinite(v)] or [0.0]
    span = (max(fin) - min(fin)) or 1.0
    finals = [float(rows[-1][m]) if np.isfinite(float(rows[-1][m])) else max(fin) + 0.05 * span for m in methods]
    ypos = _repel(finals, 0.035 * span)
    for k, m in enumerate(methods):
        st = _style(k, m)
        v = [float(r[m]) if np.isfinite(float(r[m])) else np.nan for r in rows]
        ax.plot(snr, v, ls=st["ls"], marker=st["marker"], color=st["color"], lw=1.8, ms=4, label=m)
        last = f"{v[-1]:.2f}" if np.isfinite(v[-1]) else "diverged"
        ax.annotate(f"{m}: {last}", xy=(snr[-1], finals[k]), xytext=(snr[-1] + 0.04 * (snr[-1] - snr[0]), ypos[k]),
                    fontsize=7, color=st["color"], va="center",
                    arrowprops=dict(arrowstyle="-", color=st["color"], lw=0.5))
        j = len(snr) // 2
        if np.isfinite(v[j]):
            ax.annotate(f"{v[j]:.1f}", (snr[j], v[j]), textcoords="offset points", xytext=(0, 5), ha="center",
                        fontsize=6, color=st["color"])
    ax.set_xlim(snr[0] - 0.5, snr[-1] + 0.3 * (snr[-1] - snr[0]))
    ax.set_xlabel("SNR (dB)"); ax.set_ylabel("NMSE (dB)")
    ax.set_title(f"NMSE vs SNR - {title} (value at the last SNR next to each name, middle SNR on the curve)", fontsize=9)
    ax.grid(alpha=0.3)
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    fig.tight_layout(); fig.savefig(path, dpi=180)
    plot_table(rows, methods, Path(path).with_name("nmse_table.png"), title)


def plot_table(rows, methods, path, title):
    """Heat-map table: methods x SNR, NMSE (dB) printed in every cell."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    snr = [r["SNR_dB"] for r in rows]
    M = np.array([[float(r[m]) for r in rows] for m in methods])
    M[~np.isfinite(M)] = np.nan                          # v31: diverged cells are printed as such
    order = np.argsort(np.nan_to_num(M[:, len(snr) // 2], nan=np.inf))
    M, names = M[order], [methods[i] for i in order]
    fig, ax = plt.subplots(figsize=(1.0 + 0.62 * len(snr), 0.9 + 0.32 * len(names)))
    im = ax.imshow(np.ma.masked_invalid(M), cmap="viridis_r", aspect="auto")
    lo, span = (np.nanmin(M), max(np.nanmax(M) - np.nanmin(M), 1e-9)) if np.isfinite(M).any() else (0.0, 1.0)
    for i in range(M.shape[0]):
        for j in range(M.shape[1]):
            if not np.isfinite(M[i, j]):
                ax.text(j, i, "diverged", ha="center", va="center", fontsize=5.5, color="#aa0000")
                continue
            ax.text(j, i, f"{M[i, j]:.2f}", ha="center", va="center", fontsize=6.5,
                    color="white" if (M[i, j] - lo) / span > 0.5 else "black")
    ax.set_xticks(range(len(snr))); ax.set_xticklabels([f"{s:g}" for s in snr], fontsize=7)
    ax.set_yticks(range(len(names))); ax.set_yticklabels(names, fontsize=7)
    ax.set_xlabel("SNR (dB)"); ax.set_title(f"NMSE (dB) - {title}", fontsize=9)
    fig.colorbar(im, ax=ax, shrink=0.8, label="NMSE (dB)")
    fig.tight_layout(); fig.savefig(path, dpi=180)


if __name__ == "__main__":
    main()

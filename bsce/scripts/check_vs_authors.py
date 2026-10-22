#!/usr/bin/env python3
"""Check the GM-LAMP data sets and the paper-protocol training of bsce against the authors' own files.

  %run scripts/check_vs_authors.py
  %run scripts/check_vs_authors.py --runs "outputs/*/GM-LAMP-T8_paper_trscenario" "outputs/*/LAMP-T8_paper_trscenario"

It needs reference/gmlamp (the authors' package, Wei-Hu-Dai 2021):
  CE_SV/CSmatrix256128.mat, CE_DeepMIMO/CSmatrix_128256.mat   the CS matrices of the two data sets
  Train/Trained_Networks/*.npz   the authors' training states: weights (identical to the published
                                 CE_*/Trained_*.mat used by Main.m) and the training log of every stage
and the data sets of scripts/make_gmlamp_data.py (data/SV_ULA, SV_UPA, DeepMIMO_ULA, DeepMIMO_UPA).

1. Measurement files (generate_measurement_data.py): the noise power of every chunk is estimated from y - A x with the
   authors' matrix of the data set. It must equal the stored SNR; training = 10 chunks with SNR in the band,
   validation = 10 dB.
2. Validation sets: the authors' 16 published networks on our validation set at 10 dB, against the validation NMSE the
   authors logged at the end of their training (their own validation set, also 10 dB). The difference is compared
   with the spread of a 2000-channel NMSE (bootstrap): |z| < 2 means "as close as two random validation sets".
3. --runs: a network retrained by bsce with the paper protocol (run_suite.py --protocol paper --train-snr scenario)
   against the authors' log of the same network (method, data set, band): the validation NMSE and the number of
   iterations of each of the 64 training stages.
Writes results/<name>/: authors_check.md, authors_check.json, validation_vs_authors.png, stages_<SCENARIO>_<run>_<band>.png.
The networks are computed with a NumPy port of the authors' TensorFlow graph (define_networks.build_LAMP), not with
bsce's models, so the check does not depend on the code it checks.
"""
import argparse
import glob
import json
import re
import sys
from pathlib import Path

import numpy as np
import scipy.io as sio

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from bsce.config import load_yaml, resolve  # noqa: E402

SCEN = {"SV_ULA": "configs/scenarios/sv_ula.yaml", "SV_UPA": "configs/scenarios/sv_upa.yaml",
        "DeepMIMO_ULA": "configs/scenarios/deepmimo_ula.yaml", "DeepMIMO_UPA": "configs/scenarios/deepmimo_upa.yaml"}
BANDS = {"0to10dB": (0.0, 10.0), "10to20dB": (10.0, 20.0)}


# ------------------------------------------------------------------------------------------------------------------
# the authors' network: NumPy port of define_networks.build_LAMP + define_shrinkage (complex128)
# ------------------------------------------------------------------------------------------------------------------
def _soft(r, rvar, theta):
    lam = np.abs(theta[0] * np.sqrt(rvar))
    ar = np.abs(r)
    on = (ar - lam) > 0
    xhat = r * (np.maximum(ar - lam, 0) / ar)
    return xhat, np.mean(on * (1 - lam / (2 * ar)), 0), np.mean(on * (0.5 * lam * np.sqrt(r) * np.power(np.conj(r), -1.5)), 0)


def _gm(corr, rvar, theta):
    K, th = theta.shape[1], -40.0
    P = np.exp(theta[0]) / np.sum(np.exp(theta[0]))
    mu, var = theta[1], np.exp(np.maximum(th, theta[2]))
    U = V = dU = dV = dU1 = dV1 = 0
    for k in range(K):
        s = rvar + var[k]
        d = corr - mu[k]
        pdf = np.exp(np.maximum(th, np.real(-np.conj(d) * d / 2 / s))) / 2 / np.pi / np.real(var[k] + rvar)
        u = (mu[k] * rvar + corr * var[k]) / s * P[k] * pdf
        v = P[k] * pdf
        U, V = U + u, V + v
        dU = dU + var[k] / s * P[k] * pdf + u * (-np.conj(d) / 2 / s)
        dU1 = dU1 + u * (-d / 2 / s)
        dV, dV1 = dV + v * (-np.conj(d) / 2 / s), dV1 + v * (-d / 2 / s)
    return U / V, np.mean((dU * V - U * dV) / (V * np.conj(V)), 0), np.mean((dU1 * V - U * dV1) / (V * np.conj(V)), 0)


def authors_net(y, A, Bs, thetas, gm):
    M, N = A.shape
    eta = _gm if gm else _soft
    xhat, d, d1 = eta(Bs[0] @ y, np.sum(np.abs(y) ** 2, 0) / M, thetas[0])
    v = y
    for t in range(1, len(Bs)):
        v = y - A @ xhat + N / M * d * v + N / M * d1 * np.conj(v)
        xhat, d, d1 = eta(xhat + Bs[t] @ v, np.sum(np.abs(v) ** 2, 0) / M, thetas[t])
    return xhat


def npz_name(gm, sc, band):
    return f"Trained_{'GM_LAMP' if gm else 'LAMP'}_for_{sc}_256128{band}_10dB{'_K=4' if gm else ''}.npz"


def authors_state(folder, gm, sc, band):
    d = np.load(Path(folder) / npz_name(gm, sc, band), allow_pickle=True)
    log = re.findall(r"\n(.+?) nmse=(-?[\d.]+) dB in (\d+) iterations", str(d["log"]))
    T = len([k for k in d.files if k.startswith("B_")])
    return ([d[f"B_{t}:0"] for t in range(T)], [d[f"theta_{t}:0"] for t in range(T)],
            [dict(stage=s, nmse_db=float(v), iters=int(i)) for s, v, i in log])


def nmse_db(e, en):
    return 10 * np.log10(e.sum() / en.sum())


# ------------------------------------------------------------------------------------------------------------------
def check_data(scen, authors, n_boot, out):
    meas, val = {}, []
    for name in scen:
        sc = load_yaml(ROOT / SCEN[name])
        d = resolve(sc["data_dir"])
        A = sio.loadmat(resolve(sc["cs_matrix"]))["A"].astype(np.float64)
        N = A.shape[1]
        if not (d / f"{name}_validation_channel.mat").exists():
            print(f"[check] {d} has no data: make it first (scripts/make_gmlamp_data.py); {name} skipped")
            continue
        xs = {p: sio.loadmat(d / f"{name}_{p}_channel.mat")["x"] for p in ("training", "validation", "test")}
        meas[name] = dict(sizes={p: int(x.shape[1]) for p, x in xs.items()}, files={})
        for part in ("training", "validation"):
            x = xs[part]
            for band, (lo, hi) in BANDS.items():
                f = sio.loadmat(d / f"{name}_{part}_128_measurements_{band}.mat")
                y, stored = f["y"], np.asarray(f.get("snr_db", np.array([np.nan])), float).ravel()
                nch = int(sc.get("n_snr_chunks", 10)) if part == "training" else 1
                kc = x.shape[1] // nch
                est, sto = [], []
                for k in range(nch):
                    s = slice(k * kc, (k + 1) * kc)
                    r = y[:, s] - A @ x[:, s]
                    est.append(-10 * np.log10(np.sum(np.abs(r) ** 2) / (N * kc)))
                    sto.append(float(stored[k * kc] if stored.size == x.shape[1] else stored[min(k, stored.size - 1)]))
                est, sto = np.array(est), np.array(sto)
                ok = bool(np.max(np.abs(est - sto)) < 0.1 and (np.all((sto >= lo) & (sto <= hi)) if part == "training"
                                                              else np.allclose(sto, 10)))
                meas[name]["files"][f"{part} {band}"] = dict(chunks=nch, snr_db=np.round(sto, 2).tolist(),
                                                             max_error_db=round(float(np.max(np.abs(est - sto))), 3), ok=ok)
                print(f"  {name:13s} {part:10s} {band:9s} {nch:2d} chunks, SNR {np.round(sto, 2)}  "
                      f"estimated from y - A x within {np.max(np.abs(est - sto)):.3f} dB  {'OK' if ok else 'NOT OK'}")
        ch = name.split("_")[0]
        for gm in (False, True):
            for band in BANDS:
                try:
                    Bs, ths, log = authors_state(authors, gm, name, band)
                except FileNotFoundError:
                    print(f"  {npz_name(gm, name, band)} not found in {authors}: skipped")
                    continue
                y = sio.loadmat(d / f"{name}_validation_128_measurements_{band}.mat")["y"]
                xv = xs["validation"]
                xh = np.concatenate([authors_net(y[:, s:s + 1000], A, Bs, ths, gm) for s in range(0, y.shape[1], 1000)], 1)
                e, en = np.sum(np.abs(xh - xv) ** 2, 0), np.sum(np.abs(xv) ** 2, 0)
                ours = nmse_db(e, en)
                rb = np.random.default_rng(7)
                sd = float(np.std([nmse_db(e[i], en[i]) for i in (rb.integers(0, e.size, e.size) for _ in range(n_boot))]))
                logged = log[-1]["nmse_db"]
                z = (ours - logged) / (np.sqrt(2) * sd)
                val.append(dict(network=f"{'GM-LAMP' if gm else 'LAMP'} {name} {band}", data=ch, authors_db=logged,
                                ours_db=round(float(ours), 3), diff_db=round(float(ours - logged), 3), sd_db=round(sd, 3),
                                z=round(float(z), 2)))
                print(f"  {val[-1]['network']:30s} authors' validation {logged:8.3f} dB | ours {ours:8.3f} dB | "
                      f"difference {ours - logged:+.3f} dB (sampling sd {sd:.3f}, z {z:+.2f})")
    return meas, val


def stage_log_of(run, band):
    """Stage log of a bsce run: train_log.json 'stage_log' (bsce >= 17) or the train_state of the .pt."""
    try:
        b = json.loads((run / "train_log.json").read_text(encoding="utf-8"))["bands"][band]
        if b.get("stage_log"):
            return b["stage_log"]
    except (OSError, KeyError, ValueError):
        pass
    import torch
    ck = torch.load(run / f"{run.name}_{band}.pt", map_location="cpu", weights_only=False)
    return [dict(stage=s["stage"], nmse_db=float(s["nmse_db"]), iters=int(s.get("iters") or 0)) for s in ck["train_state"]["log"]]


def check_runs(patterns, authors, out):
    res = []
    runs = [Path(h) for p in patterns for h in (sorted(glob.glob(str(resolve(p)))) if any(c in p for c in "*?[") else [str(resolve(p))])]
    for run in runs:
        try:
            tl = json.loads((run / "train_log.json").read_text(encoding="utf-8"))
        except (OSError, ValueError):
            print(f"[check] {run}: no train_log.json, skipped")
            continue
        model = tl["config"]["model"]["model"]
        if model not in ("LAMP", "GM-LAMP"):
            print(f"[check] {run.name}: {model} has no counterpart in the authors' package, skipped")
            continue
        sc = tl["scenario"]["name"]
        for band in tl["bands"]:
            if band not in BANDS:
                print(f"[check] {run.name} {band}: the authors trained 0to10dB and 10to20dB only, skipped")
                continue
            ours = stage_log_of(run, band)
            try:
                _, _, theirs = authors_state(authors, model == "GM-LAMP", sc, band)
            except FileNotFoundError:
                print(f"[check] {npz_name(model == 'GM-LAMP', sc, band)} not found, skipped")
                continue
            names_ok = [o["stage"] for o in ours] == [t["stage"] for t in theirs[:len(ours)]]
            layer_end = [i for i, t in enumerate(theirs[:len(ours)]) if re.search(r"non-linear T=\d+ trainrate=2$", t["stage"])]
            row = dict(run=str(run.relative_to(ROOT)) if ROOT in run.parents else str(run), scenario=sc, band=band, method=model,
                       stages=len(ours), same_stage_names=names_ok,
                       layer_val_ours=[round(ours[i]["nmse_db"], 3) for i in layer_end],
                       layer_val_authors=[round(theirs[i]["nmse_db"], 3) for i in layer_end],
                       iters_ours=sum(o["iters"] for o in ours), iters_authors=sum(t["iters"] for t in theirs[:len(ours)]),
                       min_iters_ours=min(o["iters"] for o in ours), min_iters_authors=min(t["iters"] for t in theirs))
            res.append(row)
            print(f"  {run.name} {band}: {len(ours)} stages (names {'identical' if names_ok else 'DIFFERENT'}), "
                  f"after each layer ours {row['layer_val_ours']} | authors {row['layer_val_authors']} dB; "
                  f"iterations {row['iters_ours']:,} vs {row['iters_authors']:,}")
            plot_stages(ours, theirs[:max(len(ours), 1)], f"{model} {sc} {band}: {run.name} vs the authors' training",
                        out / f"stages_{sc}_{run.name}_{band}.png")
    return res


def plot_stages(ours, theirs, title, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    n = len(ours)
    fig, ax = plt.subplots(2, 1, figsize=(max(8, 0.22 * n + 3), 7.5), sharex=True, gridspec_kw=dict(height_ratios=[2.2, 1]))
    k = np.arange(1, n + 1)
    a, b = [t["nmse_db"] for t in theirs[:n]], [o["nmse_db"] for o in ours]
    ax[0].plot(k[:len(a)], a, "o-", ms=3, color="#B3412E", label="authors (their log)")
    ax[0].plot(k, b, "s-", ms=3, color="#2F62C8", label="bsce, paper protocol")
    for i, t in enumerate(theirs[:n]):
        end = bool(re.search(r"non-linear T=\d+ trainrate=2$", t["stage"]))
        if end or n <= 16:                         # short runs: the value of every stage; long runs: after each layer
            up = b[i] > a[i]                       # the upper curve gets its label above, the lower one below
            ax[0].annotate(f"{a[i]:.2f}", (k[i], a[i]), textcoords="offset points", xytext=(0, -14 if up else 9), ha="center", fontsize=8, color="#B3412E")
            ax[0].annotate(f"{b[i]:.2f}", (k[i], b[i]), textcoords="offset points", xytext=(0, 9 if up else -14), ha="center", fontsize=8, color="#2F62C8")
            if end:
                ax[0].axvline(k[i] + 0.5, color="0.85", lw=0.8)
    ax[0].margins(y=0.14)
    ax[0].set_ylabel("validation NMSE at 10 dB (dB)")
    ax[0].legend(loc="center right", fontsize=9)
    ax[0].set_title(title, fontsize=10)
    ax[0].grid(alpha=0.3)
    w = 0.4
    ax[1].bar(k[:len(a)] - w / 2, [t["iters"] / 1000 for t in theirs[:n]], w, color="#B3412E", label="authors")
    ax[1].bar(k + w / 2, [o["iters"] / 1000 for o in ours], w, color="#2F62C8", label="bsce")
    ax[1].set_ylabel("iterations (thousands)")
    ax[1].set_xlabel("training stage (per layer: linear, 3 refinements, non-linear, 3 refinements)")
    ax[1].grid(alpha=0.3, axis="y")
    ta, tb = sum(t["iters"] for t in theirs[:n]), sum(o["iters"] for o in ours)
    ax[1].text(0.01, 0.95, f"total: authors {ta:,}, bsce {tb:,}", transform=ax[1].transAxes, va="top", fontsize=9)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def plot_val(val, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    n = len(val)
    fig, ax = plt.subplots(figsize=(10.5, 0.42 * n + 1.6))
    yk = np.arange(n)[::-1]
    for y_, r in zip(yk, val):
        c = "#2F62C8" if abs(r["z"]) < 2 else "#D99A00"
        ax.errorbar(r["diff_db"], y_, xerr=2 * np.sqrt(2) * r["sd_db"], fmt="o", color=c, capsize=3)
        ax.text(0.66, y_, f"ours {r['ours_db']:.2f}, authors {r['authors_db']:.2f}, diff {r['diff_db']:+.2f} (z {r['z']:+.1f})",
                va="center", fontsize=8)
    ax.axvline(0, color="0.4", lw=0.8)
    ax.set_yticks(yk, [r["network"] for r in val], fontsize=8)
    ax.set_xlim(-0.6, 1.75)
    ax.set_xticks(np.arange(-0.5, 0.75, 0.25))
    ax.set_xlabel("our validation NMSE minus the authors' logged validation NMSE (dB); bars: ± 2 sd of two random 2000-channel sets")
    ax.set_title("The authors' published networks on our validation sets (10 dB) vs their own validation sets", fontsize=10)
    ax.grid(alpha=0.3, axis="x")
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--scenarios", nargs="+", default=list(SCEN), help="SV_ULA SV_UPA DeepMIMO_ULA DeepMIMO_UPA")
    ap.add_argument("--runs", nargs="*", default=[], help="bsce LAMP / GM-LAMP run folders trained with the paper protocol")
    ap.add_argument("--authors", default="reference/gmlamp/Train/Trained_Networks", help="folder of the authors' .npz states")
    ap.add_argument("--no-data", action="store_true", help="only the --runs comparison")
    ap.add_argument("--bootstrap", type=int, default=1000)
    ap.add_argument("--name", default="authors_check")
    ap.add_argument("--results", default="results")
    a = ap.parse_args()
    authors = resolve(a.authors)
    if not authors.is_dir():
        sys.exit(f"{authors} not found: it is the 'Train Network/Trained_Networks' folder of the authors' package")
    scen = [s for s in SCEN for t in a.scenarios if t.lower() in (s.lower(), Path(SCEN[s]).stem)]
    out = resolve(a.results) / a.name
    out.mkdir(parents=True, exist_ok=True)
    rep = {}
    lines = ["# bsce against the authors' files (GM-LAMP data and paper protocol)", ""]
    if not a.no_data:
        print("1-2. data sets: measurement files and validation sets")
        meas, val = check_data(scen, authors, a.bootstrap, out)
        rep.update(measurements=meas, validation=val)
        lines += ["## Measurement files (y = A(x + n), the authors' A, SNR of every chunk estimated from y - A x)", "",
                  "| data set | train / val / test | file | chunks | SNR (dB) | max error (dB) | |", "|---|---|---|---|---|---|---|"]
        for nm, m in meas.items():
            for k, f in m["files"].items():
                lines.append(f"| {nm} | {m['sizes']['training']:,} / {m['sizes']['validation']:,} / {m['sizes']['test']:,} | {k} | "
                             f"{f['chunks']} | {', '.join(f'{s:g}' for s in f['snr_db'])} | {f['max_error_db']:.3f} | {'OK' if f['ok'] else 'NOT OK'} |")
        if val:
            plot_val(val, out / "validation_vs_authors.png")
            lines += ["", "## The authors' published networks on our validation sets (10 dB)", "",
                      "Authors = the validation NMSE they logged at the end of training (their validation set).",
                      "sd = sampling spread of a 2000-channel NMSE (bootstrap); |z| < 2: as close as two random validation sets.", "",
                      "| network | authors (dB) | ours (dB) | difference (dB) | sd (dB) | z |", "|---|---|---|---|---|---|"]
            lines += [f"| {r['network']} | {r['authors_db']:.3f} | {r['ours_db']:.3f} | {r['diff_db']:+.3f} | {r['sd_db']:.3f} | {r['z']:+.2f} |"
                      for r in val]
            lines += ["", "![validation](validation_vs_authors.png)"]
    if a.runs:
        print("3. retrained networks against the authors' training logs")
        runs = check_runs(a.runs, authors, out)
        rep["runs"] = runs
        lines += ["", "## Retrained networks (bsce, paper protocol) against the authors' training logs", "",
                  "| run | band | stages | names | validation after each layer, ours (dB) | authors (dB) | iterations ours / authors |",
                  "|---|---|---|---|---|---|---|"]
        lines += [f"| {r['run']} | {r['band']} | {r['stages']} | {'identical' if r['same_stage_names'] else 'different'} | "
                  f"{', '.join(f'{v:.2f}' for v in r['layer_val_ours'])} | {', '.join(f'{v:.2f}' for v in r['layer_val_authors'])} | "
                  f"{r['iters_ours']:,} / {r['iters_authors']:,} |" for r in runs]
    (out / "authors_check.json").write_text(json.dumps(rep, indent=1), encoding="utf-8")
    (out / "authors_check.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"\nwritten {out / 'authors_check.md'}")


if __name__ == "__main__":
    main()

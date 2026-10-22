#!/usr/bin/env python3
"""Train one learned estimator per SNR band of a scenario; save .pt + .mat for each band.

  %run scripts/train.py --scenario sv_upa --method ladmm6             (the short form, v24)
      = one network for 0-20 dB with the common protocol ladmm_best -> outputs/SV_UPA/LADMM-T6/, exactly the network
        and the folder name of run_suite.py --scenarios sv_upa --methods ladmm6 (without its fine-tune)
  %run scripts/train.py --scenario sv_upa --method nfladmmsr4 --protocol paper --snr 0 10 20
      -> outputs/SV_UPA/NF-LADMM-SR-T4_paper_tr0-10-20/
  python scripts/train.py --scenario configs/scenarios/sv_ula.yaml --train configs/train/gmlamp_original.yaml
      (the long form: any train config; default protocol 'original' = the method's own script, folder named by
       the model, e.g. GM-LAMP; as before)
  --scenario   a name (sv_ula, sv_upa, deepmimo_ula, deepmimo_upa, 3gpp_upa, ...) or a yaml path
  --method     <key><layers> as in run_suite.py: ladmm6, lamp8, gmlamp8, lista16, alista16, lvamp6, admmnet15,
               nfladmm8, nfladmmc8, nfladmmsr4, utrr30, cadmm6 (no number = the method's paper depth); with --method the
               defaults are run_suite.py's: --protocol ladmm_best, --snr 0 20, folder <METHOD>-T<T>[_<options>]
  options: --protocol fair        apply configs/train/fair_protocol.yaml on top
           --bands 0to10dB        train only some bands
           --device auto|cpu|cuda (default auto: GPU if available, else CPU)
           --smoke                tiny data + few iterations (CPU check of the pipeline)
           --seed 1               other training seed -> folder <tag>_s1
           --tag NAME             own folder name (e.g. LADMM-full-T12_gmlamp)
           --set train.lr=5e-4 model.T=10   (any override)
           --noise-model white    white measurement noise y = A x + w instead of the colored y = A (x + n) (default:
                                  the scenario's noise_model, else colored); runs go to outputs/<scenario>_white/
           --save-layers          also save the network as it is when each layer is finished (layer-wise schedules):
                                  <tag>/layers/T01_<tag>_<band>.pt/.mat ... T12_... -> scripts/depth_snapshots.py
           --snr-dist fair        (v26) every training sample its own SNR, spread evenly over the band (stratified,
                                  random order) instead of a few SNR chunks; validation over a range at equally spaced
                                  SNRs; online noise: every minibatch spread evenly. Folder suffix _fair
           --snr-step 0.2         with --snr-dist fair: the SNRs on the grid lo, lo+0.2, ..., hi, equal counts (_fair0p2)
Outputs: outputs/<scenario>/<tag>/<tag>_<band>.{pt,mat}, <tag>_<band>_orig.mat, train_log.json
(white noise: outputs/<scenario>_white/<tag>/..., measurement files <colored name>_white.mat).
Resume: an interrupted run continues when the same command is run again. The checkpoint
(outputs/<scenario>/<tag>/checkpoints/) is written after every stage and, inside a stage, every 10 minutes
(train.checkpoint_minutes): weights, optimizer, random streams, best state and position, so the resumed run ends with
exactly the network of an uninterrupted one. Bands already finished are kept. A change of scenario, band or
configuration gives another checkpoint name, so it never resumes a different training.
Early stop (v25): an end-to-end training (schedule 'steps') stops when the best validation has not improved by
train.min_gain_db (0.01 dB) during train.patience_steps (5000) steps; the best state is kept. 0 = every step.
"""
import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

import numpy as np
import scipy.io as sio
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from bsce.config import (load_yaml, apply_overrides, merge, resolve, scenario_cs_matrix,  # noqa: E402
                         band_file_names, scenario_run_name)
from bsce.data import noise as noise_models  # noqa: E402
from bsce.naming import data_extra, run_tag  # noqa: E402
from bsce.device import get_device, describe  # noqa: E402
from bsce.models import build_model, alista_W  # noqa: E402
from bsce.train.trainer import Trainer, TrainConfig  # noqa: E402
from bsce.export import save_model, truncate_model  # noqa: E402
from bsce.runinfo import write_readme, digest  # noqa: E402
from bsce.data.files import (apply_band_options, bands_from_edges, ensure_band_files, fair_validation,  # noqa: E402
                             online_validation, points_validation, read_snr_db, snr_dist_suffix)


def model_tag(mcfg):
    t = mcfg["model"]
    if t == "LADMM":
        t += "-" + mcfg.get("operator", "full")
        if mcfg.get("threshold", "scalar") != "scalar":
            t += "-" + mcfg["threshold"]
        if mcfg.get("noise_level", "median") != "median":
            t += "-" + mcfg["noise_level"]
        if mcfg.get("debias", "none") != "none":
            t += "-debias"
    if t == "NF-LADMM":
        t += "-S" if mcfg.get("denoiser", "conv") == "conv" else "-C"
        if mcfg.get("sigma", "known") != "known":
            t += "-" + mcfg["sigma"]
        if mcfg.get("update", "dual") == "relax":
            t += "-relax"
    if t == "GM-LAMP" and mcfg.get("gm_init", "original") != "original":
        t += "-" + mcfg["gm_init"]
    return t


def _final_nmse_db(state):
    """Validation NMSE of the saved network: the last layer's value (after restore_best), not the last epoch's."""
    done = [v for v in state.get("layer_nmse_db", []) if v is not None]
    if done:
        return done[-1]
    return state["log"][-1]["nmse_db"] if state["log"] else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scenario", required=True, help="a name (sv_upa, deepmimo_ula, 3gpp_upa, ...) or a yaml path")
    ap.add_argument("--method", default=None,
                    help="instead of --train: <key><layers> as in run_suite.py, e.g. ladmm6, lamp8, nfladmmsr4 (defaults "
                         "then as run_suite.py: --protocol ladmm_best, --snr 0 20, folder <METHOD>-T<T>[_options])")
    ap.add_argument("--train", default=None, help="a train config yaml (configs/train/...), instead of --method")
    ap.add_argument("--protocol", choices=["original", "paper", "fair", "gmlamp", "gmlamp_best", "ladmm_best", "e2e"],
                    default=None,
                    help="default: ladmm_best with --method, original with --train. original (= paper): each method's own "
                         "script; fair: configs/train/fair_protocol.yaml; "
                         "gmlamp: GM-LAMP's training procedure for every method (configs/train/gmlamp_protocol.yaml); "
                         "gmlamp_best: the same + best validated weights restored + validation averaged over the "
                         "validation SNRs (configs/train/gmlamp_best_protocol.yaml); ladmm_best: the protocol of the user's "
                         "best SV UPA 0-20 dB LADMM, untitled2.py (configs/train/ladmm_best_protocol.yaml); e2e: all layers "
                         "trained together end to end (configs/train/e2e_protocol.yaml)")
    ap.add_argument("--bands", default=None)
    ap.add_argument("--device", default="auto")
    ap.add_argument("--out", default="outputs")
    ap.add_argument("--tag", default=None)
    ap.add_argument("--save-layers", action="store_true",
                    help="save the network when each layer is finished: <tag>/layers/T<t>_<tag>_<band>.pt/.mat "
                         "(a real t-layer estimator at that moment; test depth curve: scripts/depth_snapshots.py)")
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--seed", type=int, default=None,
                    help="training seed (init + minibatches); seed k > 0 adds _s<k> to the folder name")
    ap.add_argument("--snr", nargs="+", default=None,
                    help="SNR band edges in dB, replacing the scenario's bands: '--snr 0 20' = one network for 0-20 dB "
                         "(the default with --method), '--snr 0 10 20' = two networks (0-10, 10-20), '--snr -5 30' ...; "
                         "'scenario' = the scenario's own bands (the default with --train)")
    ap.add_argument("--margin", type=float, default=0.0,
                    help="edge margin in dB: each band is trained on [lo - M, hi + M] and used for [lo, hi], so its edges are "
                         "trained like its middle (default 0 = as the authors). Validation stays on [lo, hi]")
    ap.add_argument("--val-snr", nargs="+", default=None,
                    help="validation noise: 'range' = SNR ~ U(band) (fixed), or SNRs in dB, e.g. '10' (the authors: 10 dB for "
                         "both bands). Default: as the bands define it (scenario bands: their val_snr; --snr bands: range)")
    ap.add_argument("--snr-chunks", type=int, default=None,
                    help="SNR chunks of the training measurement file (the authors 10; --snr bands 100)")
    ap.add_argument("--snr-dist", choices=["chunks", "fair"], default="chunks",
                    help="how the training SNRs cover a band: chunks (default, as before: the training file in SNR chunks, "
                         "one random SNR each; online noise i.i.d. U(band)); fair: every sample its own SNR, spread evenly "
                         "over the band in random order (files <band>_fair.mat, folder suffix _fair); validation over a "
                         "range at equally spaced SNRs")
    ap.add_argument("--snr-step", type=float, default=None,
                    help="with --snr-dist fair: the SNRs on the grid lo, lo+STEP, ..., hi dB, every value equally often "
                         "(e.g. 0.2 -> 0, 0.2, ..., 20); default: continuous")
    ap.add_argument("--noise", choices=["file", "online"], default=None,
                    help="file: measurement files (made automatically if missing); online: fresh noise every "
                         "minibatch, SNR ~ U(band) per sample, fixed generated validation set (= --set train.noise=...)")
    ap.add_argument("--noise-model", default=None, choices=["colored", "coloured", "folded", "white"],
                    help="measurement noise: colored = y = A (x + n) (GM-LAMP's model, default) or white = y = A x + w "
                         "with the same mean noise power (bsce/data/noise.py). Default: the scenario's noise_model")
    ap.add_argument("--set", nargs="*", default=[])
    a = ap.parse_args()
    if bool(a.method) == bool(a.train):
        sys.exit("give --method (e.g. ladmm6) or --train <config yaml>")
    if not a.scenario.endswith(".yaml") and not Path(a.scenario).exists():
        a.scenario = f"configs/scenarios/{a.scenario.lower()}.yaml"
    label = T_m = None
    if a.method:
        from run_suite import parse_method          # the method keys and paper depths of run_suite.py
        label, a.train, T_m = parse_method(a.method)
        a.set = [f"model.T={T_m}"] + list(a.set)
        if a.protocol is None:
            a.protocol = "ladmm_best"
        if a.snr is None:
            a.snr = ["0", "20"]
    if a.protocol in (None, "paper"):
        a.protocol = "original"
    snr_scenario = a.snr is not None and [v.lower() for v in a.snr] == ["scenario"]
    a.snr = None if (a.snr is None or snr_scenario) else [float(v) for v in a.snr]

    sc = load_yaml(a.scenario)
    nm = noise_models.canonical(a.noise_model or sc.get("noise_model"))
    if nm == "white":
        sc["noise_model"] = "white"
        print("[bsce] noise model: WHITE, y = A x + w (same mean noise power as y = A (x + n); bsce/data/noise.py)")
    else:
        sc.pop("noise_model", None)                     # colored: the scenario dict (and checkpoint names) as before
    cfg = load_yaml(a.train)
    if a.protocol in ("fair", "gmlamp", "gmlamp_best", "ladmm_best", "e2e"):
        cfg = merge(cfg, load_yaml(f"configs/train/{a.protocol}_protocol.yaml"))
    cfg = apply_overrides(cfg, a.set)
    if a.noise:
        cfg["train"]["noise"] = a.noise
    if a.snr:
        sc["bands"] = bands_from_edges(a.snr)
        print(f"[bsce] SNR bands from --snr: {[b['name'] for b in sc['bands']]}")
    try:
        fair_sfx = snr_dist_suffix(a.snr_dist, a.snr_step)
    except ValueError as e:
        sys.exit(str(e))
    if a.margin or a.val_snr or a.snr_chunks or fair_sfx:
        val = None if not a.val_snr else ("range" if a.val_snr == ["range"] else [float(v) for v in a.val_snr])
        try:
            sc["bands"] = apply_band_options(sc["bands"], margin=a.margin, val=val, chunks=a.snr_chunks,
                                             default_chunks=int(sc.get("n_snr_chunks", 10)),
                                             snr_dist=a.snr_dist if fair_sfx else None, snr_step=a.snr_step)
        except ValueError as e:
            sys.exit(str(e))
        for b in sc["bands"]:
            how = (("one SNR per sample, spread evenly" + (f" on a {b['snr_step']:g} dB grid" if b.get("snr_step") else ""))
                   if b.get("snr_dist") == "fair" else f"{b.get('n_snr_chunks', sc.get('n_snr_chunks', 10))} chunks")
            print(f"[bsce] band {b['name']}: training SNR {b['train'][0]:g}..{b['train'][1]:g} dB "
                  f"({how}), used for {b.get('use_range', b['train'])}, "
                  f"validation {'SNR ' + str(b['val_snr']) + ' dB' if 'val_snr' in b else ('equally spaced SNRs in ' if b.get('snr_dist') == 'fair' else 'SNR ~ U') + str(b.get('val_range'))}")
    if a.seed is not None:
        cfg["train"]["seed"] = a.seed
    if a.smoke:
        sc["data_dir"] = sc["data_dir"] + "_smoke"
        cfg["train"].update(maxit=20, better_wait=10, epochs=1, steps=20, val_every=10,
                            batch_size=min(int(cfg["train"].get("batch_size", 128)), 128))
        cfg["model"]["T"] = min(int(cfg["model"].get("T", 4)), 3)
    dev = get_device(a.device)
    print(f"[bsce] device: {describe(dev)}")
    data = resolve(sc["data_dir"])
    pre = f"{sc['dataset']}_{sc['array']}"
    A = sio.loadmat(scenario_cs_matrix(sc))["A"].astype(np.float64)
    mcfg = dict(cfg["model"])
    if mcfg["model"] in ("LADMM", "NF-LADMM", "Graph-LADMM"):
        mcfg.setdefault("array", sc["array"])          # beam grid (group threshold / circular CNN)
        mcfg.setdefault("N1", sc.get("N1", 16)); mcfg.setdefault("N2", sc.get("N2", 16))
    if mcfg["model"] == "NF-LADMM":
        mcfg.setdefault("noise_model", nm)             # the exact data step of the data's noise model
        if noise_models.canonical(mcfg["noise_model"]) != nm:
            print(f"[bsce] NOTE: NF-LADMM data step for {mcfg['noise_model']} noise on {nm} data (mismatch on purpose)")
    if mcfg["model"] in ("LADMM", "Graph-LADMM") and nm == "white":
        step = mcfg.get("data_step", "whitened" if mcfg["model"] == "Graph-LADMM" else "plain")
        if step == "whitened":          # A^+ y, P: exact for y = A (x + n); on white noise it amplifies the noise
            print(f"[bsce] WARNING: {mcfg['model']} data_step 'whitened' is the step of folded noise y = A (x + n), but "
                  f"the data are white (y = A x + w): about 1-2 dB lost (docs/NOISE_MODELS.md). Use "
                  f"--set model.data_step=plain" + (" (Graph-LADMM plain has no TV z-step)" if mcfg["model"] ==
                                                    "Graph-LADMM" else ""))
    if mcfg["model"] == "ALISTA" and mcfg.get("W") is None:
        wf = data / "ALISTA_W.mat"
        if not wf.exists():
            sio.savemat(wf, {"W": alista_W(A)})
        mcfg["W"] = sio.loadmat(wf)["W"]
    if a.tag:
        tag = a.tag
    elif a.method:                                    # the names of run_suite.py (bsce/naming.py)
        extra = data_extra(a.margin, a.val_snr, a.snr_chunks) + ([fair_sfx] if fair_sfx else [])
        tag = run_tag(label, T_m if not a.smoke else min(T_m, 3), "paper" if a.protocol == "original" else a.protocol,
                      a.snr, a.seed or 0, overrides=a.set[1:], extra=extra)
    else:
        tag = model_tag(mcfg) + ("" if a.protocol == "original" else f"_{a.protocol}") + (f"_s{a.seed}" if a.seed else "")
    print(f"[bsce] run folder: {resolve(a.out) / scenario_run_name(sc, a.smoke) / tag}")
    outdir = resolve(a.out) / scenario_run_name(sc, a.smoke) / tag
    outdir.mkdir(parents=True, exist_ok=True)
    for part in ("training", "validation"):
        if not (data / f"{pre}_{part}_channel.mat").exists():
            sys.exit(f"missing {data / f'{pre}_{part}_channel.mat'}\n  this data set is not in this bsce folder. Either copy the "
                     f"folder data/{data.name} from your previous bsce folder, or make it (MANUAL.md section 3):\n"
                     "    GM-LAMP data : python scripts/make_gmlamp_data.py --sv --deepmimo-ula <H_ula.mat> --deepmimo-upa <H_upa.mat>\n"
                     "    3GPP data    : python scripts/make_3gpp_data.py  (data/3GPP_UPA, data/3GPP_ULA; --help for sizes)\n"
                     "    other sizes  : python scripts/make_dataset.py ...")
    xt = sio.loadmat(data / f"{pre}_training_channel.mat")["x"]
    xv = sio.loadmat(data / f"{pre}_validation_channel.mat")["x"]
    bands = [b for b in sc["bands"] if a.bands is None or b["name"] in a.bands.split(",")]
    logf = outdir / "train_log.json"
    log = {"scenario": sc, "config": {k: v for k, v in cfg.items()}, "device": describe(dev), "bands": {}}
    if logf.exists():                                   # keep the bands trained by earlier calls (--bands / --snr)
        try:
            old = json.loads(logf.read_text(encoding="utf-8"))
        except ValueError:
            old = {}
        same = lambda c: json.dumps(c, sort_keys=True, default=str)                          # noqa: E731
        if old.get("config") is not None and same(old["config"]) != same(json.loads(same(log["config"]))):
            sys.exit(f"{outdir} holds a network trained with another configuration (its train_log.json 'config' "
                     f"differs): give this run its own folder with --tag, or move that folder")
        log["bands"] = old.get("bands", {})
    noise = cfg["train"].get("noise", "file")
    for bi, band in enumerate(bands):
        s2t = s2v = None
        if noise == "online":
            lo, hi = map(float, band["train"])
            vlo, vhi = map(float, band.get("val_range", band.get("use_range", band["train"])))
            cfg["train"]["snr_range"] = [lo, hi]
            if fair_sfx:                                # every minibatch spread evenly over the band (v26)
                cfg["train"]["snr_dist"] = "fair"
                if a.snr_step:
                    cfg["train"]["snr_step"] = float(a.snr_step)
            vseed = int(sc.get("seed", 0)) + 7000 + int(round(10 * vlo)) * 1009 + int(round(10 * vhi))
            yt = None
            if band.get("val_points"):                  # --val-snr <SNRs>: fixed noise at those SNRs
                yv, s2v, _ = points_validation(xv, A, band["val_snr"], vseed, noise_model=nm)
                vdesc = f"SNR {band['val_snr']} dB"
            elif fair_sfx:                              # equally spaced validation SNRs, equal chunks (v26)
                yv, s2v, sv_ = fair_validation(xv, A, vlo, vhi, vseed, noise_model=nm,
                                               n=int(band.get("val_n_snr_chunks", band.get("n_snr_chunks", 100))))
                band["val_n_snr_chunks"] = int(np.unique(sv_).size)
                vdesc = f"{band['val_n_snr_chunks']} equally spaced SNRs in [{vlo:g},{vhi:g}] dB"
            else:
                yv, s2v, _ = online_validation(xv, A, vlo, vhi, vseed, noise_model=nm)
                vdesc = f"SNR ~ U({vlo:g},{vhi:g}) dB"
            how = "spread evenly in every minibatch" + (f" on a {a.snr_step:g} dB grid" if a.snr_step else "") \
                if fair_sfx else "per sample"
            print(f"data: {xt.shape[1]:,} training channels, ONLINE {nm} noise SNR ~ U({lo:g},{hi:g}) dB {how}; "
                  f"{xv.shape[1]:,} validation channels, fixed noise {vdesc} (seed {vseed})")
        else:
            ftr, fva = band_file_names(sc, band)
            if not ((data / ftr).exists() and (data / fva).exists()):
                print(f"[bsce] measurement files of band {band['name']} missing -> making them (scripts/make_sv_measurements.py)")
                ensure_band_files(sc, band, sc["bands"].index(band), data, A, (ftr, fva))
            yt = sio.loadmat(data / ftr)["y"]
            yv = sio.loadmat(data / fva)["y"]
            if yt.shape[1] != xt.shape[1] or yv.shape[1] != xv.shape[1]:
                sys.exit(f"channel/measurement size mismatch: x {xt.shape[1]} vs y {yt.shape[1]} (training), "
                         f"x {xv.shape[1]} vs y {yv.shape[1]} (validation)")
            st, sv = read_snr_db(data / ftr, xt.shape[1]), read_snr_db(data / fva, xv.shape[1])
            s2t = None if st is None else 10.0 ** (-st / 10)
            s2v = None if sv is None else 10.0 ** (-sv / 10)
            print(f"data: {xt.shape[1]:,} training ({ftr}), {xv.shape[1]:,} validation ({fva})")
        torch.manual_seed(cfg["train"].get("seed", 0))
        model = build_model(mcfg, A, device=dev)
        if hasattr(model, "init_from_data"):        # v29 Graph-LADMM: edge phases from the clean training channels
            info = model.init_from_data(xt[:, :min(xt.shape[1], 20000)])
            if info:
                print("[bsce] graph edge phases from the training channels (degrees, coherence, samples): " +
                      ", ".join(f"{k} {v[0]:+.2f} ({v[1]:.2f}, {v[2]})" for k, v in info.items()))
        print(f"\n=== {tag} | band {band['name']} | {model.n_params():,} parameters ===")
        tcfg = TrainConfig.from_dict(cfg["train"])
        if tcfg.val_mean_db:
            if "val_snr" in band:
                tcfg.val_groups = len(band["val_snr"])
                print(f"validation: mean over {tcfg.val_groups} SNRs {band['val_snr']} dB (in dB)")
            else:
                # range band (--snr edges): the validation file holds n_snr_chunks equal chunks, one SNR each
                # (ensure_band_files / genMeasurements), so "mean in dB over the validation SNRs" = mean over chunks
                g = int(band.get("val_n_snr_chunks", band.get("n_snr_chunks", sc.get("n_snr_chunks", 10))))
                kv = (xv.shape[1] if yv is None else yv.shape[1])
                if kv % g:
                    g = 1
                    print(f"[bsce] val_mean_db: {kv} validation columns not divisible into SNR chunks -> pooled NMSE")
                tcfg.val_groups = g
                print(f"validation: mean in dB over the {g} SNR chunks of the validation range "
                      f"{band.get('val_range', band['train'])} dB")
        # checkpoints resume an interrupted run only if scenario, band and config are identical
        fp = hashlib.sha1(json.dumps([sc, band, cfg, a.smoke], sort_keys=True, default=str).encode()).hexdigest()[:10]
        ck = outdir / "checkpoints" / f"{tag}_{band['name']}_{fp}"
        layer_cb = None
        if a.save_layers:
            tail = int(getattr(model, "tail_outputs", 0))               # LADMM debias: one output after the last layer
            n_out, T_full = len(model.stage_plan()), max(1, int(getattr(model, "T", 1)))
            per = max(1, (n_out - tail) // T_full)                     # LADMM: 2 outputs per layer
            if tcfg.schedule in ("end2end", "steps"):
                sys.exit(f"--save-layers: the schedule '{tcfg.schedule}' trains all layers together, so there is nothing "
                         "to save per layer (the run would only make the full-depth network). Use a layer-wise protocol "
                         "(ladmm_best, gmlamp, fair) for snapshots, or train every depth itself (--method ladmm4, "
                         "ladmm6, ...), or leave --save-layers out.")

            def layer_cb(out, net, nmse_db, _band=band, _per=per, _t0=time.time()):
                if out >= n_out - tail:                                # the tail: the full network
                    t = T_full
                elif (out + 1) % _per or (tail and (out + 1) // _per == T_full):
                    return                                             # (with a tail, layer T is saved after it)
                else:
                    t = (out + 1) // _per
                sub = truncate_model(net, t)
                p_, _ = save_model(sub, outdir / "layers" / f"T{t:02d}_{tag}_{_band['name']}",
                                   train_state={"layer_nmse_db": [nmse_db]},
                                   data_info={"band": _band, "layer": t, "of_T": int(mcfg.get("T", 0)),
                                              "val_nmse_db": nmse_db, "seconds_since_start": time.time() - _t0,
                                              "n_params": sub.n_params()})
                print(f"[bsce] layer {t} finished: val {nmse_db:.4f} dB -> saved {p_.parent.name}/{p_.name}")
        tr = Trainer(model, tcfg, xt, yt, xv, yv, checkpoint=str(ck), s2_train=s2t, s2_val=s2v, layer_callback=layer_cb,
                     noise_model=nm)
        t0 = time.time()
        state = tr.run()
        files = [data / f"{pre}_training_channel.mat", data / f"{pre}_validation_channel.mat"]
        if noise != "online":
            files += [data / ftr, data / fva]
        else:
            ftr = fva = None
        info = {"band": band, "noise": noise, "noise_model": nm, "T": int(mcfg.get("T", 0)), "train_seconds": time.time() - t0, "n_params": model.n_params(),
                "data_digest": {f.name: digest(f) for f in files}}
        pt, mat = save_model(model, outdir / f"{tag}_{band['name']}", train_state=state, data_info=info)
        log["bands"][band["name"]] = {**info, "final_val_nmse_db": _final_nmse_db(state),
                                      "layer_nmse_db": state["layer_nmse_db"], "files": [str(pt), str(mat)],
                                      # every training stage: name, last validation NMSE, iterations (as the authors' log)
                                      "stage_log": [dict(stage=s["stage"], nmse_db=s["nmse_db"], iters=s.get("iters"))
                                                    for s in state["log"]]}
        print(f"saved {pt.name} and {mat.name}")
        logf.write_text(json.dumps(log, indent=1, default=str), encoding="utf-8")
        tc = cfg["train"]
        write_readme(outdir, f"{tag}: {mcfg['model']} trained on {sc['name']}", [
            ("what", {"method": mcfg["model"], "layers T": mcfg.get("T"), "parameters": model.n_params(),
                      "precision": mcfg.get("precision"), "protocol": a.protocol, "train config": a.train,
                      "overrides": " ".join(a.set) or "-", "seed": tc.get("seed", 0)}),
            ("data", {"scenario": sc["name"], "scenario file": a.scenario, "data folder": str(data),
                      "training channels": xt.shape[1], "validation channels": xv.shape[1],
                      "noise": noise, "noise model": nm + (" (y = A x + w)" if nm == "white" else " (y = A (x + n))"),
                      "training SNRs": ("fair: one SNR per sample, spread evenly over the band" +
                                        (f", grid step {a.snr_step:g} dB" if a.snr_step else "") +
                                        "; validation over a range at equally spaced SNRs") if fair_sfx
                      else "as before (file: SNR chunks; online: i.i.d. U(band))"}),
            ("training", dict({k: tc.get(k) for k in ("schedule", "lr", "refinements", "batch_size", "maxit", "better_wait",
                                                      "steps", "epochs", "warm_start", "restore_best", "optimizer", "loss",
                                                      "noise") if tc.get(k) is not None},
                              grad_clip=tc.get("grad_clip") or "none")),
            ("bands (one network per band)", [f"{b}: train SNR {v['band'].get('train')} dB, validation "
                                              f"{v['band'].get('val_range', v['band'].get('val_snr'))}, "
                                              f"final val {v['final_val_nmse_db']:.3f} dB, "
                                              f"{v['train_seconds'] / 3600:.2f} h" for b, v in log["bands"].items()]),
            ("data files used (SHA-1; equal on two computers = same data)",
             [f"{k}: {v}" for b in log["bands"].values() for k, v in b.get("data_digest", {}).items()]),
            ("files", [f"{tag}_<band>.pt     PyTorch (exact)", f"{tag}_<band>.mat    MATLAB export",
                       f"{tag}_<band>_orig.mat  drop-in file for the original MATLAB test code (LAMP/GM-LAMP/LADMM)",
                       "train_log.json   everything above + validation of every stage",
                       "layers/          the network after every layer (only with --save-layers)",
                       "checkpoints/     resume point of an interrupted run"]),
        ])


if __name__ == "__main__":
    main()

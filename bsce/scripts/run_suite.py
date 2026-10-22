#!/usr/bin/env python3
"""Train (and fine-tune) any set of learned methods on any scenarios with ONE common protocol.

  %run scripts/run_suite.py --scenarios sv_upa --methods ladmm lamp gmlamp --device cuda
  %run scripts/run_suite.py --scenarios all --methods all --device cuda
  %run scripts/run_suite.py ... --dry-run                (print the commands, run nothing)

SCENARIOS  names in configs/scenarios (sv_ula, sv_upa, deepmimo_ula, deepmimo_upa, 3gpp_upa, 3gpp_ula, or any yaml path);
           'all' = the four GM-LAMP data sets (made by scripts/make_gmlamp_data.py); '3gpp' = 3gpp_upa and 3gpp_ula (made by
           scripts/make_3gpp_data.py); 'all 3gpp' = all six
METHODS    <key>[T]; the number is the number of layers, default = the method's paper depth:
             ladmm     LADMM (full 256x256 operator)        T = 6
             lamp      LAMP  (Borgerding; Wei-Hu-Dai)       T = 8
             gmlamp    GM-LAMP (Wei-Hu-Dai, K = 4)          T = 8
             lista     LISTA (complex)                      T = 16
             alista    ALISTA (analytic W)                  T = 16
             lvamp     LVAMP                                T = 6
             admmnet   ADMM-Net (THADMM-Net)                T = 15
             cadmm     complex ADMM-Net, untied M1/M2 (Johnston et al., TSP 2021)  T = 6   (v28; not in 'all': name it)
             gladmm    Graph-LADMM, analytic x-step: magnetic graph TV + energy gate; any combiner          T = 8
             gladmmf   Graph-LADMM with a free H_t per layer + graph gate (one combiner, like LADMM-F)        T = 6
             gladmmany Graph-LADMM trained on random combiners (any combiner / data set)                     T = 8
                       (v29; not in 'all': name them. v30: gladmm and gladmmany always get their own end-to-end
                       training, protocol paper, whatever --protocol is, unless --force-protocol; gladmmf takes
                       --protocol like every method)
             nfladmm   NF-LADMM-S (proposed, CNN denoiser)  T = 8
             nfladmmc  NF-LADMM-C (proposed, radial)        T = 8
             nfladmmsr NF-LADMM-SR (suggested: CNN denoiser + over-relaxed update + slope init 2)   T = 4
             utrr      UTRR (Wu-Cheng-Eldar-Cioffi, TCOM 2025; top-K 16 in the last layer) T = 30
                       (not in 'all': name it; with its own training: --protocol paper)
             ladmmat   LADMM + noise-adaptive soft threshold, blind noise level (median) per channel     T = 6
             ladmmatr  LADMM-AT with the noise level from the previous layer's whitened residual       T = 6
             ladmmatrd LADMM-ATR + a least-squares debias layer on a soft support (3 parameters)       T = 6
                       (v31.3; not in 'all': name them; full operator and training as ladmm)
           LADMM model options (--set, or a method row's own settings in the workbench), v28:
             model.threshold=adaptive|firm|gm   noise-adaptive shrinkage (blind noise level per channel)
             model.data_step=whitened           exact data step of the folded noise y = A (x + n)
             model.operator=analytic|analytic1  (A^T A + rho I)^-1 by SVD, 3 / 2 parameters per layer
             model.noise_level=residual         (v31.3, with adaptive|firm|gm) noise level from the previous residual
             model.debias=ls                    (v31.3) least-squares debias layer after the last layer
           e.g. ladmm4, lamp6, lista12.  'all' = every method at its default depth.
PROTOCOL   --protocol ladmm_best (default, the same for every method) | gmlamp_best | gmlamp | fair | e2e |
           paper (each method's own training settings)          -> configs/train/<name>_protocol.yaml
           (v30: except gladmm / gladmmany, which keep their own, protocol paper; --force-protocol to change)
SNR        --train-snr 0 20       band edges of the networks trained from scratch (default: one network 0-20 dB);
                                  '0 10 20' = two networks; 'scenario' = the scenario's own bands (GM-LAMP: 0-10 and
                                  10-20 dB, 10 SNR chunks, validation at 10 dB)
           --ft-snr 0 10 20       fine-tune the one network per band (default 0-10, 10-20); 'none' = no fine-tune
           --ft-iters 20000       fine-tune iterations per band
           --snr-dist fair        (v26) every training sample its own SNR, spread evenly over the band in random order,
                                  instead of the SNR chunks of the measurement files (the authors 10, ranges 100); the
                                  fine-tune minibatches spread evenly too; validation over a range at equally spaced
                                  SNRs. Suffix _fair (trained and fine-tuned folders); files <band>_fair.mat
           --snr-step 0.2         with --snr-dist fair: the SNRs on the grid lo, lo+0.2, ..., hi, equal counts (_fair0p2)
WARM START --warm-start protocol   (default) as the protocol says (ladmm_best / fair / e2e: cold; gmlamp, gmlamp_best:
                                  warm; paper: each method's own config, LAMP / GM-LAMP warm)
           --warm-start cold|warm  the same for every method (folder suffix _cold / _warm)
           --warm-start best      (v31.3) every layer-wise method trained twice, cold and warm (_cold / _warm), and the
                                  one with the better VALIDATION NMSE kept: outputs/<SCENARIO>/best_warm_start.json;
                                  compare.py --best leaves the other one out. Each method is then compared in its best
                                  configuration, chosen by one rule for all and never on the test set. Methods that train
                                  all layers together (end to end) have no new layer to initialise: trained once.
DEPTH      --save-layers          save the network after every layer (scripts/depth_snapshots.py plots the depth curve);
                                  v31.2: methods that train end to end in a mixed list train at full depth only
SCHEDULE   --schedule layerwise   (default) every method as its protocol says: layer by layer for ladmm_best / gmlamp /
                                  gmlamp_best / fair; the method's own schedule for paper; end to end for e2e
           --schedule e2e         ALL methods end to end: all layers trained together from the start, Adam with cosine
                                  decay lr -> 0.05 lr over --e2e-steps minibatches (default 30000), validation every
                                  500 steps, best validated state kept. lr, batch, loss, noise and precision stay those
                                  of --protocol. Folder suffix _e2e<steps>, e.g. LADMM-T6_e2e30k
CLIPPING   --grad-clip 1.0        clip the global gradient norm of EVERY method at 1.0, in training and in the fine-tune
                                  (folder suffix _clip1). --grad-clip none: no clipping for every method, also where
                                  the method's own config clips (--protocol paper / e2e; suffix _noclip).
                                  Default (option not given): as now. The common protocols clip nobody; e2e clips
                                  everybody at 1.0; paper keeps each method's own setting.
NOISE      --noise-model white    white measurement noise y = A x + w (same mean noise power) instead of the colored
                                  y = A (x + n) of GM-LAMP (default). Runs go to outputs/<SCENARIO>_white/ with the same
                                  folder names, so the two noise models can be compared run by run; the white files are
                                  the colored ones with the same noise draw whitened (bsce/data/noise.py)
ENSEMBLE   --ensemble ladmm6 ladmm8  (v28, off by default) after the training, average the final networks of these
                                  methods (fine-tuned if --ft-snr, else trained; the names this command gives them) into
                                  one network per band: outputs/<SCENARIO>/ENS-LADMM-T6+LADMM-T8[_<shared suffix>]/
                                  (scripts/make_ensemble.py; xhat = mean of the members). The methods need not be in
                                  --methods: the ensemble is made as soon as all members exist (with --split per method,
                                  the last command makes it), and skipped when it exists.
OTHER      --seed 0   --device auto|cuda|cpu   --out outputs   --set train.lr=1e-4 ... (passed to train.py)
           --smoke                tiny data and budget (checks the pipeline on the CPU in minutes)

Folder names (outputs/<SCENARIO>/<name>/), built only from the choices, so every computer makes the same names:
  <METHOD>-T<T>[_<protocol>][_tr<edges>][_<override>][_s<seed>]   trained network(s), e.g. LADMM-T6,
                                                         LAMP-T8_paper_trscenario, LADMM-T6_restore-best-true
  <that>_ft_<edges>dB_<iterations>[_margin<M>]          fine-tuned per band, next to it: LADMM-T6_ft_0-10-20dB_20k,
                                                         LADMM-T6_ft_0-5-10-15-20dB_20k (files <folder>_<band>.pt)
  (_<protocol> only when it is not ladmm_best; _tr only when the training range is not 0-20 dB; every --set override
   is added too, e.g. _restore-best-true. Fine-tunes of v23 and older, <that>-ft[<edges>], are still found and skipped;
   scripts/rename_runs.py gives them the new names.)
Resume: an interrupted training or fine-tune continues from its last checkpoint when the same command is run again
  (inside a stage every 10 minutes; finished bands are kept), and gives exactly the networks of an uninterrupted run.
Every run is independent (own seed, own subprocess, data files made deterministically), so methods can be split
over several commands or computers: e.g. '--methods lamp gmlamp' on one PC and '--methods ladmm ladmm4' on another
give the same networks as one command with all four. A step whose files exist is skipped; the same command
continues an interrupted suite (train.py also resumes inside a run from its checkpoint).
"""
import argparse
import json
import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from bsce.config import load_yaml, resolve, scenario_run_name  # noqa: E402
from bsce.data import noise as noise_models  # noqa: E402
from bsce import naming  # noqa: E402

# key: (folder label, train config, default T)
METHODS = {
    "ladmm": ("LADMM", "configs/train/ladmm_original.yaml", 6),
    "lamp": ("LAMP", "configs/train/lamp_original.yaml", 8),
    "gmlamp": ("GM-LAMP", "configs/train/gmlamp_original.yaml", 8),
    "lista": ("LISTA", "configs/train/lista.yaml", 16),
    "alista": ("ALISTA", "configs/train/alista.yaml", 16),
    "lvamp": ("LVAMP", "configs/train/lvamp.yaml", 6),
    "admmnet": ("ADMM-Net", "configs/train/admmnet.yaml", 15),
    "cadmm": ("CADMM", "configs/train/cadmm.yaml", 6),
    "gladmm": ("G-LADMM", "configs/train/gladmm.yaml", 8),
    "gladmmf": ("G-LADMM-F", "configs/train/gladmm_free.yaml", 6),
    "gladmmany": ("G-LADMM-anyA", "configs/train/gladmm_anyA.yaml", 8),
    "gladmmmix": ("G-LADMM-mixA", "configs/train/gladmm_mixA.yaml", 8),        # v31: a mixture of combiners (M, families)
    "gladmmbp": ("G-LADMM-BP", "configs/train/gladmm_bp.yaml", 8),            # v31: + the beam prior
    "gladmmrw": ("G-LADMM-RW", "configs/train/gladmm_rwl1.yaml", 8),          # v31: graph-reweighted l1, unrolled
    "gladmmrwtv": ("G-LADMM-RW-TV", "configs/train/gladmm_rwl1_tv.yaml", 8),  # v31: the same in gladmm's layer
    "nfladmm": ("NF-LADMM-S", "configs/train/nfladmm_s.yaml", 8),
    "nfladmmc": ("NF-LADMM-C", "configs/train/nfladmm_c.yaml", 8),
    "nfladmmsr": ("NF-LADMM-SR", "configs/train/nfladmm_sr.yaml", 4),
    "utrr": ("UTRR", "configs/train/utrr.yaml", 30),
    "ladmmat": ("LADMM-AT", "configs/train/ladmm_at.yaml", 6),          # v31.3: adaptive threshold (median noise level)
    "ladmmatr": ("LADMM-ATR", "configs/train/ladmm_atr.yaml", 6),       # v31.3: + residual noise level
    "ladmmatrd": ("LADMM-ATRD", "configs/train/ladmm_atrd.yaml", 6),    # v31.3: + debias layer
}
NOT_IN_ALL = {"ladmmat", "ladmmatr", "ladmmatrd", "utrr", "cadmm", "gladmm", "gladmmf", "gladmmany", "gladmmmix", "gladmmbp", "gladmmrw", "gladmmrwtv"}   # 'all' = the methods of the default suite (as before v28); UTRR and CADMM only when named
# v30: the analytic Graph-LADMM networks (56 parameters) train with their own end-to-end settings whatever --protocol is
# (check of 18 Oct, SV ULA: GM-LAMP's layer-wise procedure at the same budget lost 1.7 dB against end to end). Their
# folders are then <label>-T<T>_paper, as with --protocol paper. --force-protocol gives them --protocol as well.
OWN_PROTOCOL = {"gladmm": "paper", "gladmmany": "paper", "gladmmmix": "paper", "gladmmbp": "paper",
                "gladmmrw": "paper", "gladmmrwtv": "paper"}             # v31: the four new analytic variants too
SCEN = {"sv_ula": "configs/scenarios/sv_ula.yaml", "sv_upa": "configs/scenarios/sv_upa.yaml",
        "deepmimo_ula": "configs/scenarios/deepmimo_ula.yaml", "deepmimo_upa": "configs/scenarios/deepmimo_upa.yaml"}
SCEN_3GPP = {"3gpp_upa": "configs/scenarios/3gpp_upa.yaml", "3gpp_ula": "configs/scenarios/3gpp_ula.yaml"}
PROTOCOLS = ["ladmm_best", "gmlamp_best", "gmlamp", "fair", "e2e", "paper"]
DEFAULT_PROTOCOL = "ladmm_best"


def parse_method(m):
    g = re.fullmatch(r"([a-z]+?)(\d*)", m.lower().replace("-", ""))
    if not g or g.group(1) not in METHODS:
        sys.exit(f"unknown method '{m}'. Known: {', '.join(METHODS)} (+ number of layers, e.g. ladmm4)")
    label, cfg, T0 = METHODS[g.group(1)]
    return label, cfg, int(g.group(2)) if g.group(2) else T0


def method_key(m):
    """The METHODS key of a method name with its depth (gladmm8 -> gladmm)."""
    return re.fullmatch(r"([a-z]+?)(\d*)", m.lower().replace("-", "")).group(1)


def protocol_of(a, m):
    """The protocol method m is trained with: --protocol, or its own (OWN_PROTOCOL) unless --force-protocol."""
    k = method_key(m)
    return a.protocol if (getattr(a, "force_protocol", False) or k not in OWN_PROTOCOL) else OWN_PROTOCOL[k]


# the names (bsce/naming.py: one scheme for run_suite.py, train.py and finetune_bands.py)
edges_str, band_names, run_tag = naming.edges_str, naming.band_names, naming.run_tag


def run(cmd, dry):
    print("\n>>> " + " ".join(str(c) for c in cmd[1:]), flush=True)
    if dry:
        return
    r = subprocess.run([str(c) for c in cmd], cwd=ROOT)
    if r.returncode:
        sys.exit(f"step failed (exit {r.returncode}); fix it and run the same command again (finished steps are skipped)")


def folder_clash(run, cur_bands, data_dir):
    """Why an existing run folder belongs to another scenario setting, else ''. Two scenario files with the same name
    (e.g. sv_ula_4bins.yaml and sv_ula.yaml, both SV_ULA; deepmimo_upa_2band_valrange.yaml and deepmimo_upa.yaml) give
    the same folder names, so without this check a run of one would be skipped as, or mixed with, a run of the other.
    cur_bands: the bands this run trains, with the data options applied (as train.py does). A folder without
    train_log.json is not checked."""
    try:
        log = json.loads((run / "train_log.json").read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return ""
    names = [b["name"] for b in cur_bands]
    old = list(log.get("bands", {}))
    if old and not set(old) <= set(names):
        return f"its bands {old} are not those of this run {names}"
    norm = lambda v: [float(x) for x in v] if isinstance(v, (list, tuple)) else v  # noqa: E731
    keys = ("train", "val_snr", "val_range", "train_name", "val_name", "snr_dist", "snr_step")
    stored = {b.get("name"): b for b in (log.get("scenario") or {}).get("bands", [])}
    for b in cur_bands:
        o = stored.get(b["name"])
        if o is None or b["name"] not in old:
            continue
        # a key missing in an older log is not compared (bsce versions differ in the keys they store), except the
        # SNR distribution, whose absence means the chunks of before
        diff = [k for k in keys if (k in o or k in ("snr_dist", "snr_step")) and norm(o.get(k)) != norm(b.get(k))]
        if diff:
            return f"its band {b['name']} was trained with other {', '.join(diff)} ({', '.join(f'{k}={o.get(k)}' for k in diff)})"
    d0 = (log.get("scenario") or {}).get("data_dir")
    if d0 and data_dir and Path(str(d0).replace("_smoke", "")).as_posix() != Path(str(data_dir).replace("_smoke", "")).as_posix():
        return f"its data folder is {d0}, this run's is {data_dir}"
    return ""


def ensemble_tag(tags):
    """ENS-<head1>+<head2>[_<suffix>]: the method-depth heads joined, the suffix once when all members share it."""
    heads = [t.split("_", 1)[0] for t in tags]
    rests = {t.split("_", 1)[1] if "_" in t else "" for t in tags}
    if len(rests) == 1:
        r = rests.pop()
        return "ENS-" + "+".join(heads) + (f"_{r}" if r else "")
    return "ENS-" + "+".join(tags)


def make_suite_ensemble(a, py, base, sc, ens, train_edges, ft_edges, tbands_of, named, extra, ft_extra, plan):
    """--ensemble: the final network of every member as this command names it; made when all exist."""
    tags = []
    for m in ens:
        label, _, T = parse_method(m)
        if a.smoke:
            T = min(T, 3)
        tag = run_tag(label, T, protocol_of(a, m), train_edges, a.seed, a.set, extra)
        if ft_edges and len(tbands_of()) == 1:
            tag = naming.ft_tag(tag, ft_edges, 30 if a.smoke else a.ft_iters, a.ft_margin, ft_extra)
        tags.append(tag)
    etag = ensemble_tag(tags)
    bands = band_names(ft_edges) if (ft_edges and len(tbands_of()) == 1) else tbands_of()
    if all((base / etag / f"{etag}_{b}.pt").exists() for b in bands):
        print(f"--- {etag}: ensemble exists, skipped")
    elif not a.dry_run and not all((base / t / f"{t}_{b}.pt").exists() for t in tags for b in bands):
        missing = [t for t in tags if not all((base / t / f"{t}_{b}.pt").exists() for b in bands)]
        print(f"--- {etag}: ensemble waits for {', '.join(missing)} (run the suite for them; the same command then "
              "makes it)")
        return
    else:
        run([py, "scripts/make_ensemble.py", "--runs", *[base / t for t in tags], "--tag", etag], a.dry_run)
    plan.append((base.name, etag))


def end_to_end_schedule(a, m, overrides):
    """The schedule of method m if it trains all its layers together ('steps', 'end2end'), else None."""
    from bsce.config import apply_overrides, merge
    _, cfgp, _ = parse_method(m)
    c = load_yaml(cfgp)
    if protocol_of(a, m) != "paper":
        c = merge(c, load_yaml(f"configs/train/{protocol_of(a, m)}_protocol.yaml"))
    sch = apply_overrides(c, overrides)["train"].get("schedule", "borgerding")
    return sch if sch in ("steps", "end2end") else None


def mean_val_db(run_dir):
    """Mean over the bands of the validation NMSE (dB) of the network saved in run_dir (train_log.json), or None."""
    try:
        bands = json.loads((run_dir / "train_log.json").read_text(encoding="utf-8"))["bands"]
        v = [b["final_val_nmse_db"] for b in bands.values() if b.get("final_val_nmse_db") is not None]
        return sum(v) / len(v) if v else None
    except (OSError, ValueError, KeyError, TypeError):
        return None


def choose_best_warm_start(base, best_runs, dry_run=False):
    """--warm-start best: per method, keep the variant (cold / warm) with the better validation NMSE; the choice, the
    values and the folders to leave out (the other variant and its fine-tunes) go to <base>/best_warm_start.json."""
    f = base / "best_warm_start.json"
    try:
        out = json.loads(f.read_text(encoding="utf-8")) if f.exists() else {}
    except ValueError:
        out = {}
    print("\n[suite] --warm-start best: the variant with the better validation NMSE (dB, mean over the bands)")
    for m, tags in best_runs.items():
        vals = {t: mean_val_db(base / t) for t in tags}
        if dry_run or any(v is None for v in vals.values()):
            print(f"  {m}: not all variants trained yet ({', '.join(t for t, v in vals.items() if v is None)})")
            continue
        chosen = min(tags, key=lambda t: vals[t])
        others = [t for t in tags if t != chosen]
        out[m] = dict(chosen=chosen, validation_db=vals, exclude=[p for t in others for p in (t, f"{t}_ft_*")])
        print(f"  {m}: " + "  ".join(f"{t} {vals[t]:.3f}" for t in tags) + f"  -> {chosen}")
    if out and not dry_run:
        f.write_text(json.dumps(out, indent=1), encoding="utf-8")
        print(f"  written {f}; compare.py --best leaves the other variants out")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--scenarios", nargs="+", required=True)
    ap.add_argument("--methods", nargs="+", default=["ladmm", "lamp", "gmlamp"])
    ap.add_argument("--protocol", choices=PROTOCOLS, default=DEFAULT_PROTOCOL)
    ap.add_argument("--force-protocol", action="store_true",
                    help="v30: train every method with --protocol, the analytic Graph-LADMM (gladmm, gladmmany) included "
                         "(default: they keep their own end-to-end training, protocol paper)")
    ap.add_argument("--train-snr", nargs="+", default=["0", "20"])
    ap.add_argument("--ft-snr", nargs="+", default=["0", "10", "20"])
    ap.add_argument("--ft-iters", type=int, default=20000)
    ap.add_argument("--ft-patience", type=int, default=None,
                    help="fine-tune early stop: steps without a gain of 0.01 dB before the learning rate (then the band) "
                         "ends; default 4000 (finetune_bands.py); 0 = always the whole --ft-iters (folder suffix _pat0)")
    ap.add_argument("--save-layers", action="store_true")
    ap.add_argument("--warm-start", choices=["protocol", "cold", "warm", "best"], default="protocol",
                    help="v31.3: initialise each new layer from the trained previous one? protocol (default): as the "
                         "protocol says; cold / warm: for every method; best: both, the better one on validation kept")
    ap.add_argument("--schedule", choices=["layerwise", "e2e"], default="layerwise",
                    help="layerwise (default): as the protocol says; e2e: every method end to end")
    ap.add_argument("--e2e-steps", type=int, default=30000, help="minibatches of --schedule e2e (default 30000)")
    ap.add_argument("--grad-clip", default=None,
                    help="gradient-norm clip for every method (training + fine-tune), e.g. 1.0; 'none' = no clipping; "
                         "default: as now"),
    ap.add_argument("--margin", type=float, default=0.0,
                    help="edge margin (dB) of the training bands: each band trained on [lo-M, hi+M], used on [lo, hi]; "
                         "folder suffix _margin<M> (default 0, as the authors)")
    ap.add_argument("--ft-margin", type=float, default=0.0,
                    help="edge margin (dB) of the fine-tune bands; fine-tune folder suffix _margin<M> (default 0)")
    ap.add_argument("--val-snr", nargs="+", default=None,
                    help="validation noise of training and fine-tune: 'range' = SNR ~ U(band), or SNRs, e.g. '10' (the "
                         "authors). Default: as now (authors' bands: 10 dB; --train-snr ranges and fine-tune: the range). "
                         "Folder suffix _valrange / _val10")
    ap.add_argument("--snr-chunks", type=int, default=None,
                    help="SNR chunks of the training measurement files (the authors 10; ranges 100); suffix _c<N>")
    ap.add_argument("--snr-dist", choices=["chunks", "fair"], default="chunks",
                    help="training SNRs: chunks (default, as before) or fair = every sample its own SNR, spread evenly "
                         "over the band (training files, online noise, fine-tune); suffix _fair")
    ap.add_argument("--snr-step", type=float, default=None,
                    help="with --snr-dist fair: SNRs on the grid lo, lo+STEP, ..., hi dB (e.g. 0.2); suffix _fair<step>")
    ap.add_argument("--noise-model", default=None, choices=["colored", "coloured", "folded", "white"],
                    help="measurement noise: colored y = A (x + n) (default: the scenario's noise_model, else colored) or "
                         "white y = A x + w; white runs go to outputs/<SCENARIO>_white/")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default="auto")
    ap.add_argument("--out", default="outputs")
    ap.add_argument("--set", nargs="*", default=[])
    ap.add_argument("--ensemble", nargs="*", default=None,
                    help="v28: after training, average the final networks of these methods (e.g. ladmm6 ladmm8) "
                         "into one ensemble per scenario (off by default)")
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args()
    py = sys.executable

    scen = []
    for s in a.scenarios:
        if s == "all":
            scen += list(SCEN.values())
        elif s == "3gpp":                      # the two 3GPP data sets of scripts/make_3gpp_data.py
            scen += list(SCEN_3GPP.values())
        else:
            scen.append(SCEN.get(s.lower(), s if s.endswith(".yaml") else f"configs/scenarios/{s.lower()}.yaml"))
    meths = [m for x in a.methods for m in ([k for k in METHODS if k not in NOT_IN_ALL] if x == "all" else [x])]
    ens = list(dict.fromkeys(a.ensemble or []))
    for m in ens:
        parse_method(m)                                    # unknown keys stop here
    if a.ensemble is not None and len(ens) < 2:
        sys.exit("--ensemble needs at least two methods, e.g. --ensemble ladmm6 ladmm8")
    train_edges = None if a.train_snr == ["scenario"] else [float(v) for v in a.train_snr]
    ft_edges = None if a.ft_snr == ["none"] else [float(v) for v in a.ft_snr]
    named, extra = [], []                                  # named options -> train.py overrides + folder suffixes
    if a.schedule == "e2e":
        if a.protocol == "e2e":
            print("[suite] --schedule e2e with --protocol e2e: the protocol is already end to end (option ignored)")
        else:
            named += ["train.schedule=steps", f"train.steps={a.e2e_steps}", "train.lr_final_frac=0.05",
                      "train.val_every=500", "train.restore_best=true"]
            extra.append(f"e2e{a.e2e_steps // 1000}k" if a.e2e_steps % 1000 == 0 else f"e2e{a.e2e_steps}")
    if a.warm_start in ("cold", "warm"):
        named.append(f"train.warm_start={'true' if a.warm_start == 'warm' else 'false'}")
        extra.append(a.warm_start)
    clip = None
    if a.grad_clip is not None:
        if a.grad_clip.lower() in ("none", "null", "0", "off"):
            named.append("train.grad_clip=null")
            extra.append("noclip")
        else:
            clip = float(a.grad_clip)
            if clip <= 0:
                sys.exit("--grad-clip needs a positive value or 'none'")
            named.append(f"train.grad_clip={clip:g}")
            extra.append(f"clip{clip:g}".replace(".", "p"))
    fmt = lambda v: f"{v:g}".replace("-", "m").replace(".", "p")
    data_opts, ft_opts = [], []                            # train.py / finetune_bands.py data options
    if a.margin:
        data_opts += ["--margin", f"{a.margin:g}"]
        extra.append(f"margin{fmt(a.margin)}")
    if a.val_snr:
        vs = ["range"] if [v.lower() for v in a.val_snr] == ["range"] else [f"{float(v):g}" for v in a.val_snr]
        data_opts += ["--val-snr", *vs]
        extra.append("valrange" if vs == ["range"] else "val" + "-".join(fmt(float(v)) for v in vs))
        if vs != ["range"]:
            ft_opts += ["--val-snr", *vs]                  # fine-tune: 'range' is already its default
    if a.snr_chunks:
        data_opts += ["--snr-chunks", str(a.snr_chunks)]
        extra.append(f"c{a.snr_chunks}")
    if a.ft_margin:
        ft_opts += ["--margin", f"{a.ft_margin:g}"]
    ft_extra = []
    from bsce.data.files import apply_band_options, bands_from_edges, snr_dist_suffix
    try:
        fair_sfx = snr_dist_suffix(a.snr_dist, a.snr_step)
    except ValueError as e:
        sys.exit(str(e))
    if fair_sfx:                                          # v26: training files, online noise and the fine-tune
        if a.snr_chunks:
            sys.exit("--snr-chunks has no effect with --snr-dist fair (every sample has its own SNR)")
        o = ["--snr-dist", "fair"] + (["--snr-step", f"{a.snr_step:g}"] if a.snr_step else [])
        data_opts += o
        ft_opts += o
        extra.append(fair_sfx)
        ft_extra.append(fair_sfx)
    if a.ft_patience is not None:
        ft_opts += ["--patience", str(a.ft_patience)]
        if a.ft_patience != 4000:
            ft_extra.append(f"pat{a.ft_patience}")
    no_layers = set()                                     # v31.2: methods that train all layers together
    if a.save_layers:                                     # snapshots need a layer-wise schedule: checked before hours of work
        e2e_like = []
        for m in meths:
            sch = end_to_end_schedule(a, m, named + list(a.set))
            if sch:
                no_layers.add(m)
                e2e_like.append(f"{m} ({sch}, protocol {protocol_of(a, m)})")
        if e2e_like and len(no_layers) == len(meths):     # nothing at all would be saved per layer: a mistake
            sys.exit("--save-layers saves the network when each layer is finished, but these methods train all their "
                     "layers together, so nothing would be saved per layer:\n  " + ", ".join(e2e_like) +
                     ("\n  (--schedule e2e)" if a.schedule == "e2e" else "") + "\n"
                     "For a depth curve, either use a layer-wise protocol (the default ladmm_best, or gmlamp / fair), or "
                     "train every depth itself, e.g. --methods ladmm2 ladmm4 ladmm6 ladmm8 ladmm10 ladmm12 (without "
                     "--save-layers). Without --save-layers the same command trains the full-depth networks only.")
        if e2e_like:                                      # v31.2: a mixed list goes on (it used to stop here)
            print("[suite] --save-layers: these methods train all their layers together, so they are trained without "
                  "per-layer snapshots (full depth only):\n  " + ", ".join(e2e_like) + "\n  The other methods save the "
                  "network after every layer. For a depth curve of the ones above, train every depth itself "
                  "(e.g. gladmm4 gladmm6 gladmm8 gladmm10).")
    plan = []
    for s in scen:
        sc = load_yaml(s)
        nm = noise_models.canonical(a.noise_model or sc.get("noise_model"))
        sc["noise_model"] = nm
        base = resolve(a.out) / scenario_run_name(sc, a.smoke)
        nm_opts = ["--noise-model", nm] if a.noise_model else []   # the fine-tune reads it from train_log.json
        tbands = band_names(train_edges) if train_edges else [b["name"] for b in sc["bands"]]
        cur_bands = bands_from_edges(train_edges) if train_edges else [dict(b) for b in sc["bands"]]
        if a.margin or a.val_snr or a.snr_chunks or fair_sfx:       # the bands as train.py makes them (folder_clash)
            vv = None if not a.val_snr else ("range" if [v.lower() for v in a.val_snr] == ["range"] else
                                             [float(v) for v in a.val_snr])
            cur_bands = apply_band_options(cur_bands, margin=a.margin, val=vv, chunks=a.snr_chunks,
                                           default_chunks=int(sc.get("n_snr_chunks", 10)),
                                           snr_dist=a.snr_dist if fair_sfx else None, snr_step=a.snr_step)
        print(f"\n######## {sc['name']} ({s}) -> {base}")
        best_runs = {}                                     # --warm-start best: method -> [cold tag, warm tag]
        for m in meths:
            label, cfg, T = parse_method(m)
            if a.smoke:
                T = min(T, 3)
            proto = protocol_of(a, m)
            if proto != a.protocol:
                print(f"\n[suite] {m}: its own end-to-end training (protocol {proto}), not {a.protocol} "
                      "(v30: measured better; --force-protocol to change)")
            if a.warm_start == "best" and not end_to_end_schedule(a, m, named + list(a.set)):
                variants = [(["cold"], ["train.warm_start=false"]), (["warm"], ["train.warm_start=true"])]
            else:                                 # one run (end-to-end methods: no new layer to initialise)
                variants = [([], [])]
            vtags = []
            for vx, vset in variants:
                tag = run_tag(label, T, proto, train_edges, a.seed, a.set, extra + vx)
                vtags.append(tag)
                clash = folder_clash(base / tag, cur_bands, sc.get("data_dir"))
                if clash:
                    sys.exit(f"{base / tag} holds a run of another scenario setting: {clash}.\n  Its networks would be mixed "
                             "with (or wrongly taken for) this one. Give the scenario file its own 'name:' (its outputs then go "
                             "to their own folder), or use --out <another folder>.")
                done = all((base / tag / f"{tag}_{b}.pt").exists() for b in tbands)
                if done:
                    print(f"\n--- {tag}: trained, skipped ({', '.join(tbands)})")
                    if a.save_layers and m not in no_layers and not (base / tag / "layers").is_dir():
                        print(f"    WARNING: --save-layers was given, but {tag} was trained without it (no layers/ folder).\n"
                              f"    Its per-layer networks cannot be made afterwards; delete {base / tag} to retrain with snapshots.")
                else:
                    cmd = [py, "scripts/train.py", "--scenario", s, "--train", cfg, "--tag", tag, "--device", a.device,
                           "--out", a.out, "--seed", str(a.seed), "--set", f"model.T={T}", *named, *vset, *a.set]
                    cmd[6:6] = data_opts + nm_opts
                    if proto != "paper":
                        cmd[6:6] = ["--protocol", proto]
                    if train_edges:
                        cmd[6:6] = ["--snr", *[f"{e:g}" for e in train_edges]]
                    if a.save_layers and m not in no_layers:
                        cmd.insert(6, "--save-layers")
                    if a.smoke:
                        cmd.insert(6, "--smoke")
                    run(cmd, a.dry_run)
                plan.append((scenario_run_name(sc, a.smoke), tag))
                if not ft_edges:
                    continue
                if len(tbands) != 1:
                    print(f"--- {tag}: {len(tbands)} trained bands -> no fine-tune (fine-tune starts from ONE network; "
                          "use --ft-snr none, or train one range)")
                    continue
                want = 30 if a.smoke else a.ft_iters
                # <tag>_ft_<edges>dB_<iterations>[_margin..]: the budget is in the name, so budgets never mix
                ftag = naming.ft_tag(tag, ft_edges, want, a.ft_margin, ft_extra)
                done = lambda t: all((base / t / f"{t}_{b}.pt").exists() for b in band_names(ft_edges))  # noqa: E731
                legacy = None
                for old in naming.legacy_ft_tags(tag, ft_edges, a.ft_margin):     # v23 and older: <tag>-ft[edges]
                    if not done(ftag) and done(old):
                        try:
                            got = int(json.loads((base / old / "finetune_log.json").read_text(encoding="utf-8"))["args"]["iters"])
                        except (OSError, ValueError, KeyError, TypeError):
                            got = None
                        if got == want:
                            legacy = old
                if legacy:
                    print(f"--- {legacy}: fine-tuned ({want} iterations), skipped. Its name is the old one; new name "
                          f"{ftag} (scripts/rename_runs.py renames it)")
                    plan.append((scenario_run_name(sc, a.smoke), legacy))
                    continue
                if done(ftag):
                    print(f"--- {ftag}: fine-tuned, skipped")
                else:
                    # no --scenario: the fine-tune uses the data of the network's own training (its train_log.json),
                    # and writes next to it (outputs/<scenario>/<ftag>)
                    cmd = [py, "scripts/finetune_bands.py", "--snr", *[f"{e:g}" for e in ft_edges],
                           "--from", base / tag / f"{tag}_{tbands[0]}.pt", "--tag", ftag,
                           "--iters", str(30 if a.smoke else a.ft_iters), "--device", a.device, "--seed", str(a.seed)]
                    if a.smoke:
                        cmd += ["--val-every", "10"]
                    if clip is not None:
                        cmd += ["--grad-clip", f"{clip:g}"]
                    cmd += ft_opts
                    run(cmd, a.dry_run)
                plan.append((scenario_run_name(sc, a.smoke), ftag))
            if len(variants) == 2:
                best_runs[m] = vtags
        if best_runs:
            choose_best_warm_start(base, best_runs, a.dry_run)
        if ens:
            make_suite_ensemble(a, py, base, sc, ens, train_edges, ft_edges, tbands_of=lambda: tbands,
                                named=named, extra=extra, ft_extra=ft_extra, plan=plan)
    print("\nnetworks of this suite:")
    for s, t in plan:
        print(f"  {s:22s} {t}")
    print("\nnext: compare all methods (evaluation + figures + tables):\n"
          f"  %run scripts/compare.py --scenarios {' '.join(a.scenarios)}" + (" --smoke" if a.smoke else "")
          + (" --best" if a.warm_start == "best" else "")
          + (f" --noise-model {noise_models.canonical(a.noise_model)}" if a.noise_model else ""))


if __name__ == "__main__":
    main()

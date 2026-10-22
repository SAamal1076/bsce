#!/usr/bin/env python3
"""Fine-tune a trained network per SNR band (by default 0-10 and 10-20 dB), next to it, with a clear name; resumable.

  %run scripts/finetune_bands.py --from outputs/SV_UPA/LADMM-T6
      -> outputs/SV_UPA/LADMM-T6_ft_0-10-20dB_20k/LADMM-T6_ft_0-10-20dB_20k_0to10dB.pt and ..._10to20dB.pt
  %run scripts/finetune_bands.py --from outputs/SV_UPA/LADMM-T6 --snr 0 5 10 15 20 --iters 10000
      -> outputs/SV_UPA/LADMM-T6_ft_0-5-10-15-20dB_10k/  (four bands)
  %run scripts/finetune_bands.py --from Trained_LADMM_for_SV_UPA_2561280to20dB_1_11dB_better.pt --scenario sv_upa_user_2band

--from     a run folder (its one network file is used) or a .pt file (also an untitled2.py checkpoint, or a depth
           snapshot <run>/layers/T06_<run>_<band>.pt: its scenario comes from <run>/train_log.json and the fine-tune is
           named <run>_T06_ft_..., next to <run>)
--snr      band edges (default 0 10 20; or the scenario's bands with --snr scenario)
--iters    iterations per band (default 20000, as run_suite.py)
Name:      <source folder>_ft_<edges>dB_<iterations>[_<option>...], in the source's folder (outputs/<SCENARIO>/):
           the options that differ from the defaults are added (_margin2, _val10, _clip1, _lr0p0001, _batch256, _seed1 ...: a dot becomes p),
           so another setting never overwrites a fine-tune. --tag NAME to choose the name; --dry-name prints the name.
Early stop: a learning rate ends when the best validation has not improved by --min-gain dB (0.01) during --patience
           steps (4000): the first rate then hands over to the second, and the second ends the band. The best validated
           state is kept as before. --patience 0 = always the whole --iters (as up to v24).
Resume:    the same command again continues an interrupted fine-tune: finished bands are kept, the band that was
           running continues from its last checkpoint (every --checkpoint-minutes, default 10), exactly as if it had
           not stopped. A folder made with other settings or from another version of the source is never mixed in:
           the script stops and says so.
--scenario may be left out: the scenario stored in the source's train_log.json is used (the data the network was
           trained on, also its noise model). Other options: --val range|file, --val-snr 10, --margin 2, --batch 400,
           --lr 5e-5, --lr2 5e-6, --lr2-frac 0.3333, --val-every 100, --seed 0, --grad-clip, --bands 0to10dB,
           --noise-model, --out <parent folder>, --device auto.
--snr-dist fair (v26): the SNRs of every minibatch spread evenly over the band (one in each of `batch` equal parts,
           random order) instead of i.i.d. U(band); --snr-step 0.2: on the grid lo, lo+0.2, ..., hi. The validation
           (--val range) is then at 100 equally spaced SNRs of the band instead of random ones. Suffix _fair / _fair0p2.

Per band (each starts from the same source network):
  * all parameters, torch Adam; lr for the first (1-lr2_frac)*iters steps, then lr2 (restarted from the best state)
    = untitled2.py's last two refinement rates (0.1 and 0.01 x 5e-4)
  * minibatch: `batch` training channels drawn with replacement + FRESH noise each step, one SNR per sample ~ U(band.train)
  * validation every `val-every` steps on the validation channels with fixed noise, SNR ~ U(band) (seed 123)
    (--val file: the band's validation measurement file instead, when it exists);
    pooled linear NMSE; the best validated state is kept
Output: <folder>/<folder>_<band>.pt/.mat (+ _orig.mat), finetune_log.json, README.txt -> evaluate.py uses the set.
On SV UPA the fine-tune of the user's untitled2.py 0-20 dB network gave the best LADMM of 14 Oct (LADMM_why_0to20_SV.md).
"""
import argparse
import hashlib
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import scipy.io as sio
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from bsce.config import (load_yaml, resolve, scenario_cs_matrix, scenario_data_dir, band_file_names,  # noqa: E402
                         scenario_run_name)
from bsce.data import noise as noise_models  # noqa: E402
from bsce.device import get_device, describe  # noqa: E402
from bsce.export import load_model, save_model  # noqa: E402
from bsce.models.user_import import is_untitled2_checkpoint, ladmm_from_untitled2  # noqa: E402
from bsce.data.files import bands_from_edges, read_snr_db, snr_dist_suffix  # noqa: E402
from bsce.data.measurements import evenly_spaced_snr  # noqa: E402
from bsce.runinfo import write_readme, digest  # noqa: E402
from bsce.naming import fmt, ft_tag, iters_str  # noqa: E402
from bsce.train.losses import nmse_loss  # noqa: E402


TWHITE = None          # whitening map T (torch, on the device) when the noise model is white (set in main)


def _white(A_c, x, y):
    """white noise model: the same draw whitened, y = A x + T (y - A x) (bsce/data/noise.py)."""
    if TWHITE is None:
        return y
    sx = A_c @ x
    return sx + TWHITE @ (y - sx)


SNR_DIST, SNR_STEP = "chunks", None    # --snr-dist / --snr-step (set in main)


def batch_snr(lo, hi, K, g):
    """SNRs (dB) of K columns: i.i.d. U(lo, hi) (default, as before), or --snr-dist fair: one SNR in each of K equal
    parts of [lo, hi] (--snr-step: the grid lo, lo+step, ..., hi cycled from a random start), shuffled."""
    if SNR_DIST != "fair":
        return lo + (hi - lo) * torch.rand(K, generator=g)
    perm = torch.randperm(K, generator=g)
    if SNR_STEP:
        n = int(np.floor((hi - lo) / SNR_STEP + 1e-9)) + 1
        j0 = int(torch.randint(0, n, (1,), generator=g))
        return (lo + SNR_STEP * ((torch.arange(K, dtype=torch.float64) + j0) % n)[perm]).to(torch.float32)
    return (lo + (hi - lo) * (perm.to(torch.float64) + torch.rand(K, generator=g, dtype=torch.float64)) / K).to(torch.float32)


def noisy(A_c, x, lo, hi, g, with_s2=False):
    """y = A (x + n), one SNR per column ~ U(lo, hi) dB (noise drawn on the CPU generator, then moved);
    white noise model: y = A x + T A n (the same draw). --snr-dist fair: the column SNRs spread evenly (batch_snr).
    with_s2: also return the noise variance of every column (models that need it, e.g. NF-LADMM)."""
    K = x.shape[1]
    snr = batch_snr(lo, hi, K, g)
    s2 = 10 ** (-snr / 10)
    sig = torch.sqrt(s2 / 2).to(x.device, x.real.dtype)
    n = torch.complex(torch.randn(x.shape, generator=g), torch.randn(x.shape, generator=g)).to(x.device, x.dtype)
    y = _white(A_c, x, A_c @ (x + sig * n))
    return (y, s2.to(x.device, x.real.dtype)) if with_s2 else y


def noisy_points(A_c, x, snr_db, g):
    """y = A (x + n) at fixed SNR points: the columns in len(snr_db) equal chunks, one SNR each (as genMeasurements).
    -> y, noise variance of every column."""
    K, n = x.shape[1], len(snr_db)
    if K % n:
        raise SystemExit(f"{K} validation channels not divisible by {n} SNR points (--val-snr)")
    snr = torch.as_tensor(np.repeat(np.asarray(snr_db, float), K // n))
    s2 = 10 ** (-snr / 10)
    sig = torch.sqrt(s2 / 2).to(x.device, x.real.dtype)
    nz = torch.complex(torch.randn(x.shape, generator=g, dtype=torch.float64),
                       torch.randn(x.shape, generator=g, dtype=torch.float64)).to(x.device, x.dtype)
    return _white(A_c, x, A_c @ (x + sig * nz)), s2.to(x.device, x.real.dtype)


def fwd(m, y, s2=None):
    return m(y, s2=s2) if getattr(m, "needs_noise_var", False) else m(y)


def load_start(path, A, dev):
    if is_untitled2_checkpoint(path):
        m, _ = ladmm_from_untitled2(path, A)
        return m.to(dev), "untitled2.py checkpoint"
    return load_model(path, device=dev), "bsce model"


@torch.no_grad()
def val_nmse(m, y, x, chunk=5000, s2=None):
    num = den = 0.0
    for s in range(0, x.shape[1], chunk):
        xh = fwd(m, y[:, s:s + chunk], None if s2 is None else s2[s:s + chunk])
        num += float(torch.sum(torch.abs(xh - x[:, s:s + chunk]) ** 2))
        den += float(torch.sum(torch.abs(x[:, s:s + chunk]) ** 2))
    return num / den


DEFAULTS = dict(batch=400, lr=5e-5, lr2=5e-6, lr2_frac=1 / 3, val_every=100, seed=0, patience=4000, min_gain=0.01)


def resolve_source(src: Path) -> Path:
    """--from: a .pt file, or a run folder holding exactly one network file <folder>_<band>.pt."""
    if src.is_dir():
        pts = sorted(f for f in src.glob(f"{src.name}_*.pt") if f.is_file())
        if len(pts) == 1:
            return pts[0]
        if not pts:
            sys.exit(f"--from {src}: no network file {src.name}_<band>.pt in this folder")
        sys.exit(f"--from {src}: {len(pts)} network files, give one of them:\n  " + "\n  ".join(str(f) for f in pts))
    if not src.exists():
        sys.exit(f"--from {src}: no such file or folder")
    return src


def auto_tag(a, source_name, edges, band_list):
    """<source>_ft_<edges>dB_<iters>[_<option>...] (bsce/naming.py), with every option that differs from its default."""
    extra = []
    if a.val_snr:
        extra.append("val" + "-".join(fmt(v) for v in a.val_snr))
    elif a.val == "file":
        extra.append("valfile")
    if a.grad_clip:
        extra.append(f"clip{fmt(a.grad_clip)}")
    for k, short in (("lr", "lr"), ("lr2", "lr2"), ("lr2_frac", "lr2frac"), ("batch", "batch"), ("val_every", "valevery"),
                     ("seed", "seed"), ("patience", "pat"), ("min_gain", "mingain")):
        v = getattr(a, k)
        if (abs(v - DEFAULTS[k]) > 1e-12 * max(1.0, abs(DEFAULTS[k]))):
            extra.append(f"{short}{fmt(v)}" if isinstance(v, float) else f"{short}{v}")    # no dot in a name
    if a.noise_model is not None and a.noise_switched:
        extra.append(f"noise-{a.noise_model_canonical}")
    sfx = snr_dist_suffix(a.snr_dist, a.snr_step)
    if sfx:
        extra.append(sfx)
    if edges is None:                                # bands that are not contiguous: their names
        return f"{source_name}_ft_{'_'.join(b['name'] for b in band_list)}_{iters_str(a.iters)}" + \
            (f"_margin{fmt(a.margin)}" if a.margin else "") + "".join("_" + e for e in extra)
    return ft_tag(source_name, edges, a.iters, a.margin, extra)


def band_edges(bands):
    """[0to10, 10to20] -> [0, 10, 20]; None when the bands are not contiguous."""
    e = [float(bands[0]["train"][0])]
    for b in bands:
        lo, hi = map(float, b["train"])
        if abs(lo - e[-1]) > 1e-9:
            return None
        e.append(hi)
    return e


class CkptTimer:
    def __init__(self, minutes, iters):
        self.minutes, self.iters, self.last = minutes, iters, time.time()

    def due(self, step):
        if self.iters and step % self.iters == 0:
            return True
        return bool(self.minutes) and time.time() - self.last >= 60 * self.minutes


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--scenario", default=None,
                    help="scenario name (sv_upa) or yaml; default: the one stored in train_log.json next to --from")
    ap.add_argument("--from", dest="src", required=True, help="a run folder (outputs/SV_UPA/LADMM-T6) or a .pt file")
    ap.add_argument("--tag", default=None, help="folder name; default: <source>_ft_<edges>dB_<iterations>[_<options>]")
    ap.add_argument("--bands", default=None)
    ap.add_argument("--snr", nargs="+", default=None,
                    help="band edges: '0 10 20' (default) -> 0to10dB, 10to20dB; '0 5 10 15 20' -> four bands; "
                         "'scenario' = the scenario's bands")
    ap.add_argument("--val", choices=["range", "file"], default="range",
                    help="range (default): validation = validation channels + fixed noise, SNR ~ U(band) (seed 123); "
                         "file: the band's validation measurement file if it exists (GM-LAMP's files are at 10 dB only)")
    ap.add_argument("--val-snr", nargs="+", type=float, default=None,
                    help="validation at fixed SNRs instead of --val: e.g. '10' (as the authors: 10 dB for both bands) or "
                         "'0 5 10'; equal chunks, fixed noise (seed 123)")
    ap.add_argument("--margin", type=float, default=0.0,
                    help="edge margin in dB: training noise SNR ~ U(lo - M, hi + M) for the band [lo, hi]; validation stays "
                         "on the band (default 0)")
    ap.add_argument("--iters", type=int, default=20000, help="iterations per band (default 20000)")
    ap.add_argument("--batch", type=int, default=DEFAULTS["batch"])
    ap.add_argument("--lr", type=float, default=DEFAULTS["lr"])
    ap.add_argument("--lr2", type=float, default=DEFAULTS["lr2"])
    ap.add_argument("--lr2-frac", type=float, default=DEFAULTS["lr2_frac"])
    ap.add_argument("--val-every", type=int, default=DEFAULTS["val_every"])
    ap.add_argument("--seed", type=int, default=DEFAULTS["seed"])
    ap.add_argument("--grad-clip", type=float, default=None,
                    help="clip the global gradient norm at this value (default: no clipping, as before)")
    ap.add_argument("--device", default="auto")
    ap.add_argument("--noise-model", default=None, choices=["colored", "coloured", "folded", "white"],
                    help="measurement noise of the fine-tune (default: the one the network was trained with)")
    ap.add_argument("--out", default=None,
                    help="parent folder of the result; default: the folder that holds the source network's folder "
                         "(outputs/<scenario>/), so the fine-tuned networks sit next to their source")
    ap.add_argument("--patience", type=int, default=4000,
                    help="early stop: when the best validation has not improved by --min-gain dB during this many steps, "
                         "the first learning rate ends (the second starts), and in the second the band ends "
                         "(default 4000; 0 = always the whole --iters)")
    ap.add_argument("--min-gain", type=float, default=0.01, help="the gain (dB) that counts for --patience (default 0.01)")
    ap.add_argument("--snr-dist", choices=["chunks", "fair"], default="chunks",
                    help="SNRs of the minibatches: chunks (default, as before: i.i.d. U(band) per sample) or fair: spread "
                         "evenly over the band in every minibatch; the validation then at equally spaced SNRs (suffix _fair)")
    ap.add_argument("--combiner", choices=["auto", "fixed", "random"], default="auto",
                    help="v29: random = a new random +-1/sqrt(M) combiner every step (networks that follow the "
                         "combiner: Graph-LADMM analytic, analytic LADMM); auto (default) = random when the source was "
                         "trained with train.combiner=random, else fixed (the scenario's combiner)")
    ap.add_argument("--snr-step", type=float, default=None,
                    help="with --snr-dist fair: SNRs on the grid lo, lo+STEP, ..., hi (suffix _fair<step>)")
    ap.add_argument("--checkpoint-minutes", type=float, default=10.0,
                    help="save the band being trained every this many minutes (resume point; 0 = never)")
    ap.add_argument("--checkpoint-iters", type=int, default=None, help=argparse.SUPPRESS)   # tests
    ap.add_argument("--dry-name", action="store_true", help="print the folder the fine-tune would write, then stop")
    a = ap.parse_args()
    try:
        snr_dist_suffix(a.snr_dist, a.snr_step)
    except ValueError as e:
        sys.exit(str(e))
    global SNR_DIST, SNR_STEP
    SNR_DIST, SNR_STEP = a.snr_dist, a.snr_step

    src = resolve_source(Path(a.src))
    import re as _re
    snap = _re.match(r"T(\d+)_", src.name) if src.parent.name == "layers" else None   # a depth snapshot (--save-layers)
    run_dir = src.parent.parent if snap else src.parent                               # its run folder holds the logs
    sc_run, comb_run = None, "fixed"
    for lf in (run_dir / "train_log.json", run_dir / "finetune_log.json"):   # finetune_log: a fine-tuned source
        if lf.exists():
            try:
                L = json.loads(lf.read_text(encoding="utf-8"))
                sc_run = L.get("scenario")
                comb_run = ((L.get("config") or {}).get("train") or {}).get("combiner") or \
                    (L.get("setting") or {}).get("combiner") or "fixed"
            except ValueError:
                pass
            break
    if a.combiner == "auto":
        a.combiner = comb_run if comb_run in ("fixed", "random") else "fixed"
        if a.combiner == "random":
            print("[bsce] combiner: random (the source was trained on random combiners)")
    if a.scenario:
        sp = a.scenario if a.scenario.endswith(".yaml") else f"configs/scenarios/{a.scenario.lower()}.yaml"
        sc = load_yaml(sp)
        a.scenario = sp
    elif sc_run:
        sc = dict(sc_run)
        a.scenario = f"configs/scenarios/{sc['name'].lower()}.yaml"
        print(f"[bsce] scenario taken from the run's train_log.json: {sc['name']}")
    else:
        sys.exit("give --scenario (no train_log.json next to --from)")
    if sc_run and sc_run.get("name") != sc.get("name"):
        print(f"[bsce] WARNING: {src.name} was trained on scenario {sc_run.get('name')}, fine-tuning on {sc.get('name')}.\n"
              f"        For the same data use --scenario configs/scenarios/{str(sc_run.get('name')).lower()}.yaml "
              "(or leave --scenario out).")
    if a.snr and [v.lower() for v in a.snr] == ["scenario"]:
        if len(sc["bands"]) < 2:
            sys.exit(f"--snr scenario: the scenario {sc['name']} as stored for this network has one band only "
                     f"({sc['bands'][0]['name']}); give the edges, e.g. --snr 0 10 20")
    else:
        edges_in = [float(v) for v in (a.snr or [0, 10, 20])]
        sc["bands"] = bands_from_edges(edges_in)
    nm = noise_models.canonical(a.noise_model or sc.get("noise_model") or (sc_run or {}).get("noise_model"))
    trained_nm = noise_models.canonical((sc_run or {}).get("noise_model"))
    a.noise_switched = sc_run is not None and nm != trained_nm
    a.noise_model_canonical = nm
    if a.noise_switched:
        print(f"[bsce] WARNING: {src.name} was trained with {trained_nm} noise; fine-tuning with {nm} noise")
    if nm == "white":
        sc["noise_model"] = "white"
    else:
        sc.pop("noise_model", None)
    if snap:                                          # layers/T06_LADMM-T12_0to20dB.pt -> LADMM-T12_T06, next to LADMM-T12
        source_name = f"{run_dir.name}_T{int(snap.group(1)):02d}"
    else:
        source_name = src.parent.name if src.parent.name and not is_untitled2_checkpoint(src) else src.stem
    bands = [b for b in sc["bands"] if not a.bands or b["name"] in a.bands.split(",")]
    tag = a.tag or auto_tag(a, source_name, band_edges(sc["bands"]), sc["bands"])
    base = resolve(a.out) / scenario_run_name(sc) if a.out else resolve(run_dir).parent
    outdir = base / tag
    if a.dry_name:
        print(outdir)
        return
    print(f"[bsce] noise model: {nm}" + (" (y = A x + w)" if nm == "white" else " (y = A (x + n))"))
    print(f"[bsce] fine-tune of {src} -> {outdir}")
    dev = get_device(a.device)
    print(f"[bsce] device: {describe(dev)}")
    data = scenario_data_dir(sc)
    pre = f"{sc['dataset']}_{sc['array']}"
    A = sio.loadmat(scenario_cs_matrix(sc))["A"].astype(np.float64)
    # what makes a band's result valid: the source file, the data and every training setting (not the device)
    keys = ("val", "val_snr", "margin", "iters", "batch", "lr", "lr2", "lr2_frac", "val_every", "seed", "grad_clip")
    setting = dict({k: getattr(a, k) for k in keys}, source=str(src.name), source_digest=digest(src), noise_model=nm,
                   scenario=sc["name"], bands=[[b["name"], *map(float, b["train"])] for b in sc["bands"]])
    for k in ("patience", "min_gain"):        # v25 options: in the setting only when not the default, so a v24 fine-tune
        if getattr(a, k) != DEFAULTS[k]:      # (same name) keeps its finished bands and resumes its running band
            setting[k] = getattr(a, k)
    if a.snr_dist == "fair":                  # v26: the same rule
        setting.update(snr_dist="fair", snr_step=a.snr_step)
    if a.combiner == "random":                # v29: the same rule
        setting["combiner"] = "random"
    fp = hashlib.sha1(json.dumps(setting, sort_keys=True, default=str).encode()).hexdigest()[:10]
    logf = outdir / "finetune_log.json"
    log = {"tag": tag, "source": str(src), "args": {k: v for k, v in vars(a).items() if k != "dry_name"},
           "noise_model": nm, "scenario": sc, "setting": setting, "fingerprint": fp, "bands": {}}
    if logf.exists():
        old = json.loads(logf.read_text(encoding="utf-8"))
        if old.get("fingerprint") not in (None, fp):
            diff = [k for k in setting if (old.get("setting") or {}).get(k) != setting[k]]
            sys.exit(f"{outdir} holds a fine-tune made with other settings or from another version of the source "
                     f"({', '.join(diff) or 'unknown'}).\n  Give another --tag, or move/delete that folder to fine-tune again.")
        if old.get("fingerprint") == fp:
            log["bands"] = old.get("bands", {})
    outdir.mkdir(parents=True, exist_ok=True)
    m0, kind = load_start(src, A, dev)
    cd = m0.cdtype
    A_c = torch.as_tensor(A, device=dev).to(cd)
    global TWHITE
    TWHITE = torch.as_tensor(noise_models.whitening_map(A), device=dev).to(cd) if nm == "white" else None
    for part in ("training", "validation"):
        if not (data / f"{pre}_{part}_channel.mat").exists():
            hint = (f"\n  {src.name} was trained on {sc_run['name']}: use --scenario configs/scenarios/"
                    f"{sc_run['name'].lower()}.yaml or leave --scenario out" if sc_run and sc_run.get("name") != sc.get("name") else
                    f"\n  copy data/{data.name} from your previous bsce folder, or make it (MANUAL.md section 3)")
            sys.exit(f"missing {data / f'{pre}_{part}_channel.mat'}{hint}")
    xt = torch.as_tensor(sio.loadmat(data / f"{pre}_training_channel.mat")["x"]).to(dev, cd)
    xv = torch.as_tensor(sio.loadmat(data / f"{pre}_validation_channel.mat")["x"]).to(dev, cd)
    print(f"start: {src} ({kind}), {xt.shape[1]:,} training / {xv.shape[1]:,} validation channels")
    crash = int(os.environ.get("BSCE_CRASH_AFTER_STEPS", "0") or 0)      # tests: an interruption after this many steps
    total = 0
    for band in bands:
        pt_final = outdir / f"{tag}_{band['name']}.pt"
        if band["name"] in log["bands"] and pt_final.exists():
            print(f"\n=== {tag} | band {band['name']}: done earlier, kept (best val "
                  f"{log['bands'][band['name']]['best_val_db']:.4f} dB)")
            continue
        blo, bhi = map(float, band["train"])
        lo, hi = blo - a.margin, bhi + a.margin                  # training range (with the edge margin)
        _, fva = band_file_names(sc, band)
        if a.val_snr:
            yv, s2v = noisy_points(A_c, xv, a.val_snr, torch.Generator().manual_seed(123))
            vsrc = f"validation channels + fixed noise at SNR {a.val_snr} dB, seed 123"
        elif a.val == "file" and (data / fva).exists():
            yv = torch.as_tensor(sio.loadmat(data / fva)["y"]).to(dev, cd)
            sn = read_snr_db(data / fva, xv.shape[1])
            s2v = None if sn is None else torch.as_tensor(10.0 ** (-np.asarray(sn) / 10)).to(dev, xv.real.dtype)
            vsrc = str(data / fva)
        elif a.snr_dist == "fair":                    # v26: equally spaced validation SNRs, equal chunks
            vlo, vhi = map(float, band.get("val_range", [blo, bhi]))
            nv = 100
            while xv.shape[1] % nv:
                nv -= 1
            yv, s2v = noisy_points(A_c, xv, evenly_spaced_snr(vlo, vhi, nv), torch.Generator().manual_seed(123))
            vsrc = f"validation channels + fixed noise at {nv} equally spaced SNRs in [{vlo:g},{vhi:g}] dB, seed 123"
        else:
            vlo, vhi = map(float, band.get("val_range", [blo, bhi]))
            yv, s2v = noisy(A_c, xv, vlo, vhi, torch.Generator().manual_seed(123), with_s2=True)
            vsrc = f"validation channels + fixed noise, SNR ~ U({vlo:g},{vhi:g}) dB, seed 123"
        m, _ = load_start(src, A, dev)
        if hasattr(m, "break_gm_symmetry"):      # v31.1: a GM-LAMP network trained before v31.1 (check_gm_components.py)
            hit = m.break_gm_symmetry(m.param_names())
            if hit:
                print(f"[bsce] warning: {', '.join(hit)} of {src}: the {m.K} mixture components are identical, so this "
                      "GM-LAMP network was trained with a one-Gaussian denoiser (bsce before v31.1). They get the gm_init=spread "
                      "offset, so the fine-tune can separate them; better: retrain the network with v31.1")
        g = torch.Generator().manual_seed(a.seed + 7)
        ckf = outdir / "checkpoints" / f"{tag}_{band['name']}_{fp}.pt"
        n2 = int(round(a.iters * a.lr2_frac))
        phases = [(a.lr, a.iters - n2), (a.lr2, n2)]
        p0 = j0 = step = 0
        ck = None
        print(f"\n=== {tag} | band {band['name']} | train SNR ~ U({lo:g},{hi:g}) | validation: {vsrc}")
        if ckf.exists():
            ck = torch.load(ckf, map_location=dev, weights_only=False)
            m.load_state_dict(ck["model"])
            best, start, hist, step, p0, j0 = ck["best"], ck["start"], ck["hist"], ck["step"], ck["phase"], ck["j"]
            best_state = {k: v.to(dev) for k, v in ck["best_state"].items()}
            g.set_state(ck["gen"])
            torch.set_rng_state(ck["torch_rng"])
            print(f"[bsce] resumed inside band {band['name']} at step {step} of {a.iters} (best val "
                  f"{10 * np.log10(best):.4f} dB)")
        else:
            best = start = val_nmse(m, yv, xv, s2=s2v)
            best_state = {k: v.detach().clone() for k, v in m.state_dict().items()}
            hist = [(0, 10 * np.log10(best))]
        print(f"  start  val nmse = {10 * np.log10(start):.4f} dB")
        timer, t0 = CkptTimer(a.checkpoint_minutes, a.checkpoint_iters), time.time()
        stops = list((ck or {}).get("stops", []))
        for p in range(p0, len(phases)):
            lr, n_it = phases[p]
            if ck is None or p != p0:
                m.load_state_dict(best_state)
                ref_db, ref_step = 10 * np.log10(best), step          # early stop: the last gain >= --min-gain
            else:
                ref_db, ref_step = ck.get("ref_db", 10 * np.log10(best)), ck.get("ref_step", step)
            params = [q for q in m.parameters() if q.requires_grad]
            opt = torch.optim.Adam(params, lr=lr)
            if ck is not None and p == p0:
                opt.load_state_dict(ck["opt"])
            for j in range(j0 if (ck is not None and p == p0) else 0, n_it):
                if step and timer.due(step) and not (ck is not None and p == p0 and j == j0):
                    ckf.parent.mkdir(parents=True, exist_ok=True)
                    if a.combiner == "random":
                        m.set_combiner(A)                  # the checkpoint holds the scenario's combiner
                    torch.save(dict(model=m.state_dict(), opt=opt.state_dict(), best=best, start=start, hist=hist,
                                    step=step, phase=p, j=j, gen=g.get_state(), torch_rng=torch.get_rng_state(),
                                    seconds=time.time() - t0 + (ck or {}).get("seconds", 0),
                                    ref_db=ref_db, ref_step=ref_step, stops=stops,
                                    best_state={k: v.detach().cpu() for k, v in best_state.items()}), str(ckf) + ".tmp")
                    os.replace(str(ckf) + ".tmp", ckf)
                    timer.last = time.time()
                total += 1
                if crash and total >= crash:
                    raise SystemExit(f"[bsce] test interruption after {total} steps")
                idx = torch.randint(0, xt.shape[1], (a.batch,), generator=g).to(dev)
                xb = xt[:, idx]
                if a.combiner == "random":            # v29: a new combiner every step, the network follows it
                    Ab = ((torch.rand(A.shape, generator=g, dtype=torch.float64) > 0.5).double() * 2 - 1) / np.sqrt(A.shape[0])
                    m.set_combiner(Ab.numpy())
                    yb, s2b = noisy(torch.as_tensor(Ab.numpy(), device=dev).to(cd), xb, lo, hi, g, with_s2=True)
                else:
                    yb, s2b = noisy(A_c, xb, lo, hi, g, with_s2=True)
                loss = nmse_loss(fwd(m, yb, s2b), xb)
                opt.zero_grad(set_to_none=True)
                loss.backward()
                if a.grad_clip:
                    torch.nn.utils.clip_grad_norm_(params, a.grad_clip)
                opt.step()
                step += 1
                if step % a.val_every == 0:
                    if a.combiner == "random":
                        m.set_combiner(A)
                    v = val_nmse(m, yv, xv, s2=s2v)
                    hist.append((step, 10 * np.log10(v)))
                    if v < best:
                        best = v
                        best_state = {k: t.detach().clone() for k, t in m.state_dict().items()}
                    if step % (10 * a.val_every) == 0:
                        print(f"  step {step:6d} lr={lr:.1e}  val nmse={10 * np.log10(v):.4f} dB "
                              f"(best {10 * np.log10(best):.4f})  {time.time() - t0:.0f}s")
                    if 10 * np.log10(v) < ref_db - a.min_gain:
                        ref_db, ref_step = 10 * np.log10(v), step
                    if a.patience and step - ref_step >= a.patience and j < n_it - 1:
                        last = p == len(phases) - 1
                        stops.append(dict(phase=p, step=step, last_gain_step=ref_step))
                        print(f"  no gain of {a.min_gain:g} dB or more since step {ref_step} (patience {a.patience}): "
                              + ("band finished early" if last else f"next learning rate ({phases[p + 1][0]:.1e}) now"))
                        break
        m.load_state_dict(best_state)
        info = dict(band=band, start_val_db=10 * np.log10(start), best_val_db=10 * np.log10(best), steps=step,
                    early_stops=stops,
                    seconds=time.time() - t0 + (ck or {}).get("seconds", 0), validation=vsrc, source=str(src))
        pth, _ = save_model(m, outdir / f"{tag}_{band['name']}",
                            train_state={"done": [f"finetune {band['name']}"], "log": hist,
                                         "layer_nmse_db": [10 * np.log10(best)]}, data_info=info)
        log["bands"][band["name"]] = info
        logf.write_text(json.dumps(log, indent=1, default=str), encoding="utf-8")   # after every band (resume)
        if ckf.exists():
            ckf.unlink()
        print(f"  -> best val {10 * np.log10(best):.4f} dB (start {10 * np.log10(start):.4f}); saved {pth}")
    logf.write_text(json.dumps(log, indent=1, default=str), encoding="utf-8")
    write_readme(outdir, f"{tag}: {source_name} fine-tuned per SNR band on {sc['name']}", [
        ("source network", [str(src), f"SHA-1 {setting['source_digest']}"]),
        ("data", {"scenario": sc["name"], "scenario file": a.scenario, "training channels": xt.shape[1],
                  "validation channels": xv.shape[1], "noise model": nm}),
        ("fine-tune (all parameters, Adam)", {"iterations per band": a.iters, "lr": a.lr,
                                              "lr2 (last fraction)": f"{a.lr2:g} for the last {a.lr2_frac:.3g}",
                                              "batch": a.batch, "noise": "fresh every step, SNR ~ U(band) per sample",
                                              "validation": f"every {a.val_every} steps, {a.val}; best state kept",
                                              "margin (dB)": a.margin, "grad clip": a.grad_clip or "none",
                                              "training SNRs": ("fair: spread evenly over the band in every minibatch" +
                                                                (f", grid step {a.snr_step:g} dB" if a.snr_step else "")
                                                                if a.snr_dist == "fair" else "i.i.d. U(band) per sample"),
                                              "early stop": (f"no gain of {a.min_gain:g} dB in {a.patience} steps -> next "
                                                             "learning rate / band finished" if a.patience else "off"),
                                              "seed": a.seed}),
        ("bands", [f"{b}: validation {v['start_val_db']:.3f} -> {v['best_val_db']:.3f} dB, {v['seconds'] / 60:.1f} min"
                   for b, v in log["bands"].items()]),
        ("files", [f"{tag}_<band>.pt / .mat / _orig.mat  one network per band (evaluate.py uses them as a set)",
                   "finetune_log.json   settings + every band (an interrupted fine-tune resumes from it)",
                   "checkpoints/        the band being trained (removed when the band is finished)"]),
    ])
    print(f"\nnext: python scripts/evaluate.py --scenario {a.scenario} --runs {base}")


if __name__ == "__main__":
    main()

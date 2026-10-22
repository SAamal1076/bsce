#!/usr/bin/env python3
"""Which GM-LAMP networks still have identical mixture components (v31.1)?

  %run scripts/check_gm_components.py --runs "outputs/*"
  %run scripts/check_gm_components.py --runs outputs/SV_ULA/GM-LAMP-T8_paper_trscenario_precision-float32

GM-LAMP's original init gives its K mixture components identical values. Before bsce v31.1 they then got bit-identical
gradients in training and never separated. The mixture stayed one Gaussian: a denoiser that scales every entry by the
same factor and cannot use the sparsity. So the network stayed near the level of a linear estimator: about -2.9 dB after
the first layer on SV ULA at 10 dB, against -9.2 dB in the authors' training log.

Every GM-LAMP network trained with bsce before v31.1 from the original init is affected and has to be retrained:
gmlamp_original.yaml, run_suite.py --methods gmlamp, any protocol. These are not affected:
- networks trained with gm_init=spread (gmlamp_spread.yaml);
- the authors' published networks;
- the networks trained with v31.1.

Checked: every .pt file under the given run folders, with the layers whose theta_t has identical components. That
includes the checkpoints/ of an interrupted training: rerun its command with v31.1, which does not resume an affected
checkpoint but starts again. Exit code 1 when an affected network (not a checkpoint) is found.
"""
import argparse
import glob
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from bsce.export import load_model  # noqa: E402


def identical_thetas(sd: dict, K=None):
    """theta_t entries (3 x K, K > 1) of a state dict whose K columns are all equal."""
    out = []
    for n, v in sd.items():
        if n.startswith("theta_") and torch.is_tensor(v) and v.dim() == 2 and v.shape[0] == 3 and v.shape[1] > 1 \
                and (K is None or v.shape[1] == K) and bool(torch.all(v == v[:, :1])):
            out.append(n)
    return sorted(out, key=lambda s: int(s.split("_")[1]) if s.split("_")[1].isdigit() else 0)


def check(f: Path):
    """('gm', identical names) for a GM-LAMP network or checkpoint, (None, None) for anything else."""
    if "checkpoints" in f.parts:
        ck = torch.load(f, map_location="cpu", weights_only=False)
        if not isinstance(ck, dict) or "model_state" not in ck:
            return None, None
        cm = ck.get("cfg_model") or {}
        if cm.get("shrink") != "gm" and cm.get("model") != "GM-LAMP":
            return None, None
        return "gm", identical_thetas(ck["model_state"], cm.get("K"))
    m = load_model(f)
    if getattr(m, "shrink", None) != "gm":
        return None, None
    return "gm", m.identical_gm_components()


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--runs", nargs="+", required=True, help="run folders or patterns (e.g. 'outputs/*')")
    a = ap.parse_args()
    files = sorted({f for pat in a.runs for d in glob.glob(pat) for f in Path(d).rglob("*.pt")
                    if not f.name.endswith(".pre_v31_1.pt")})
    bad = ck = 0
    for f in files:
        try:
            kind, same = check(f)
        except Exception as e:  # noqa: BLE001
            print(f"  ?         {f}: not read ({e})")
            continue
        if kind is None:
            continue
        if same and "checkpoints" in f.parts:
            ck += 1
            print(f"  old ckpt  {f}: a training checkpoint, identical components in {', '.join(same)}. Rerun its "
                  "command with v31.1: it starts again if these layers were trained (else it resumes correctly)")
        elif same:
            bad += 1
            print(f"  AFFECTED  {f}: identical components in {', '.join(same)} (one Gaussian): retrain with v31.1")
        else:
            print(f"  ok        {f}")
    print((f"{bad} affected GM-LAMP network(s)" if bad else "no affected GM-LAMP network")
          + (f"; {ck} training checkpoint(s) with identical components" if ck else ""))
    sys.exit(1 if bad else 0)


if __name__ == "__main__":
    main()

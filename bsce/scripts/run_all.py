#!/usr/bin/env python3
"""Generate data, train the chosen methods and evaluate them, for one scenario (Windows, Linux, macOS).

  python scripts/run_all.py --scenario configs/scenarios/sv_ula.yaml
  python scripts/run_all.py --scenario configs/scenarios/sv_upa.yaml --protocol fair --configs gmlamp_spread ladmm_original
  python scripts/run_all.py --scenario configs/scenarios/sv_ula.yaml --smoke      (tiny CPU check of every config)
options: --device auto|cpu|cuda   --skip-data   --seed 1   --no-eval
         --set train.maxit=50000 model.precision=float32 ...   (applies to every config in --configs)
         --stop-on-error (default: report failed configs at the end and carry on)
Console output of every step is also written to outputs/<scenario>[_smoke]/logs/<step>.log.
"""
import argparse
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from bsce.config import load_yaml  # noqa: E402

DEFAULT = ["lamp_original", "gmlamp_original", "gmlamp_spread", "ladmm_original", "ladmm_diag", "ladmm_lowrank",
           "ladmm_kron", "ladmm_psd", "ladmm_kronprior", "ladmm_diagprior", "lista", "alista", "lvamp", "admmnet"]


def run(cmd, log: Path) -> bool:
    """Run cmd from the repository root; stream its output to the console and to `log`."""
    log.parent.mkdir(parents=True, exist_ok=True)
    print(f"\n>>> {' '.join(str(c) for c in cmd)}", flush=True)
    t0 = time.time()
    with open(log, "w", encoding="utf-8") as f:
        p = subprocess.Popen([str(c) for c in cmd], cwd=ROOT, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                             text=True, encoding="utf-8", errors="replace")
        for line in p.stdout:
            sys.stdout.write(line)
            f.write(line)
        p.wait()
    print(f"<<< exit code {p.returncode} after {time.time() - t0:.0f} s (log: {log})", flush=True)
    return p.returncode == 0


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--scenario", required=True)
    ap.add_argument("--configs", nargs="*", default=DEFAULT, help="names in configs/train (without .yaml)")
    ap.add_argument("--protocol", choices=["original", "fair", "gmlamp", "gmlamp_best", "ladmm_best", "e2e"], default="original")
    ap.add_argument("--snr", nargs="+", default=None, help="SNR band edges for every config (train.py --snr)")
    ap.add_argument("--noise", choices=["file", "online"], default=None, help="train.py --noise for every config")
    ap.add_argument("--device", default="auto")
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--skip-data", action="store_true")
    ap.add_argument("--classical", default=None, help="default: amp,omp,sd (smoke: amp,omp)")
    ap.add_argument("--set", nargs="*", default=[])
    ap.add_argument("--seed", type=int, default=None, help="training seed; k > 0 -> folders <tag>_s<k>")
    ap.add_argument("--no-eval", action="store_true", help="skip the evaluation at the end")
    ap.add_argument("--stop-on-error", action="store_true")
    a = ap.parse_args()

    sc = load_yaml(ROOT / a.scenario)
    name = sc["name"] + ("_smoke" if a.smoke else "")
    logs = ROOT / "outputs" / name / "logs"
    py = sys.executable
    smoke = ["--smoke"] if a.smoke else []
    failed = []
    if not a.skip_data:
        if not run([py, "scripts/generate_data.py", "--scenario", a.scenario] + smoke, logs / "generate_data.log"):
            sys.exit("data generation failed")
    for c in a.configs:
        cmd = [py, "scripts/train.py", "--scenario", a.scenario, "--train", f"configs/train/{c}.yaml",
               "--protocol", a.protocol, "--device", a.device] + smoke + (
                  ["--snr"] + a.snr if a.snr else []) + (["--noise", a.noise] if a.noise else []) + (
                  ["--seed", str(a.seed)] if a.seed is not None else []) + (["--set"] + a.set if a.set else [])
        if not run(cmd, logs / f"train_{c}{f'_s{a.seed}' if a.seed else ''}.log"):
            failed.append(c)
            if a.stop_on_error:
                break
    classical = a.classical or ("amp,omp" if a.smoke else "amp,omp,sd")
    if a.no_eval:
        print("\nfailed configs: " + (", ".join(failed) if failed else "none"))
        sys.exit(1 if failed else 0)
    run([py, "scripts/evaluate.py", "--scenario", a.scenario, "--runs", f"outputs/{name}", "--classical", classical,
         "--device", a.device], logs / "evaluate.log")
    if a.smoke:
        run([py, "scripts/check_exports.py", "--runs", f"outputs/{name}", "--scenario", a.scenario],
            logs / "check_exports.log")
    print("\nfailed configs: " + (", ".join(failed) if failed else "none"))
    print(f"results: outputs/{name}/  (nmse_vs_snr.csv / .png, nmse_table.png, <method>/ models)")
    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()

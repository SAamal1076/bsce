#!/bin/sh
# NF-LADMM smoke test under colored and white measurement noise (bsce v23, CPU).
# Run from the bsce folder:  sh run_nf_noise_smoke.sh [steps] [scenarios]
# Spyder: the same three run_suite.py lines and the compare_noise.py line with %run.
STEPS=${1:-4000}
SCEN=${2:-"sv_ula sv_upa"}
OUT=outputs_noise
for S in $SCEN; do
  # 1. colored data y = A(x + n), colored (exact) data step       -> outputs_noise/<S>/NF-LADMM-SR-T4_paper_steps-<STEPS>
  python scripts/run_suite.py --scenarios $S --methods nfladmmsr4 --protocol paper --ft-snr none --device cpu \
         --out $OUT --set train.steps=$STEPS || exit 1
  # 2. white data y = Ax + w, white (exact) data step             -> outputs_noise/<S>_white/NF-LADMM-SR-T4_paper_steps-<STEPS>
  python scripts/run_suite.py --scenarios $S --methods nfladmmsr4 --protocol paper --ft-snr none --device cpu \
         --out $OUT --noise-model white --set train.steps=$STEPS || exit 1
  # 3. white data, colored data step (the mismatch on purpose)    -> ..._white/NF-LADMM-SR-T4_paper_steps-<STEPS>_noise_model-colored
  python scripts/run_suite.py --scenarios $S --methods nfladmmsr4 --protocol paper --ft-snr none --device cpu \
         --out $OUT --noise-model white --set train.steps=$STEPS model.noise_model=colored || exit 1
done
# 4. every network on both test noises (+ NF with the test's data step, the published GM-LAMP / LAMP and AMP)
python scripts/compare_noise.py --scenarios $SCEN --outputs $OUT --results results_noise --tags "NF-LADMM-SR-T4*" \
       --nf-matched --device cpu --name nf_noise_smoke

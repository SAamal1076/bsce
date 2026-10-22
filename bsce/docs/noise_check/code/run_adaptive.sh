#!/bin/sh
# white data, white step with the SNR-adaptive penalty (model.white_step=adaptive)
for S in sv_ula sv_upa; do
  python scripts/run_suite.py --scenarios $S --methods nfladmmsr4 --protocol paper --ft-snr none --device cpu \
         --out outputs_noise --noise-model white --set train.steps=4000 model.white_step=adaptive || exit 1
done
python scripts/compare_noise.py --scenarios sv_ula sv_upa --outputs outputs_noise --results results_noise --tags "NF-LADMM-SR-T4*" \
       --nf-matched --device cpu --name nf_noise_smoke

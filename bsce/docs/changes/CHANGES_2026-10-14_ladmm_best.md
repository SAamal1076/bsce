# 14 Oct 2026: the best LADMM (your untitled2.py) inside bsce

## New
- `configs/train/ladmm_best.yaml` (+ `ladmm_best_protocol.yaml`, `--protocol ladmm_best`): the training protocol of `untitled2.py`
  - lr 5e-4, refinements (.5, .2, .1, .01), batch 400, maxit 100000, better_wait 5000
  - torch Adam, float32, no warm start, last weights kept, `round(nmse.item(), 5)`
  - init `inv(A32ᵀA32 + I)` in float64
- `configs/scenarios/sv_upa_user.yaml`, `sv_ula_user.yaml`: your data in `data/SV`
  - band 0to20dB; validation file `*_1_11dB.mat`
  - CS matrix from `reference/gmlamp/CE_SV/CSmatrix256128.mat`
  - test on the usual 2000 channels of `data/SV_UPA`, `data/SV_ULA`
- `configs/scenarios/sv_upa_user_2band.yaml`, `sv_ula_user_2band.yaml`: the same channels, bands 0–10 / 10–20 dB.
- `scripts/make_sv_measurements.py`: missing measurement files, made like `generate_SV_data.py` (100 SNR chunks ~ U(band)). Existing files are kept.
- `scripts/import_user_ladmm.py` + `bsce/models/user_import.py`: an `untitled2.py` .pt becomes a bsce model (evaluate, MATLAB export).
- `scripts/finetune_bands.py`: specialises a 0–20 dB LADMM (bsce .pt or `untitled2.py` .pt) to each band.
  - Measured on SV UPA: 2.25 dB mean gain over GM-LAMP, against 1.36 dB for the 0–20 dB network.
- `tests/test_ladmm_best_vs_untitled2.py`: bsce and `reference/user/untitled2.py` run on the same data with the same minibatches.
  - Result: 20 / 20 stages identical, same stopping iterations, final weights identical (difference 0).

## Changed (backward compatible)
- `bsce/config.py`: `scenario_cs_matrix`, `scenario_test_dir`, `band_file_names`.
  - Scenario keys `test_data_dir`, `cs_matrix` (used when `data_dir` has no `CSmatrix.mat`).
  - Band keys `train_name` / `val_name`, `val_range`; scenario key `eval_bands`.
- `scripts/train.py`:
  - uses these helpers
  - gives a clear message for missing or mismatched measurement files
  - new protocol `ladmm_best`
- `scripts/evaluate.py`, `scripts/import_gmlamp_weights.py`: use these helpers.
  - With `eval_bands`, a folder holding a 0–20 dB net and folders holding a band pair are tested in the same table.
- `bsce/models/ladmm.py`: `init_precision: untitled2`.
- `bsce/train/trainer.py`: `val_round_mode: python`.
- `RUNBOOK.md`: Phase D (Spyder commands, checklist D0–D4).

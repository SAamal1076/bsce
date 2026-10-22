# Data guide: the GM-LAMP data sets (paper reproduction), step by step

*14 Oct 2026. Checked end to end in bsce: all four data sets regenerated from scratch are bit-identical to the data used so far (channels and measurement files). Runtime about 1.5 min on 2 CPU cores.*

## 1. What the GM-LAMP setting is

| | SV_ULA / SV_UPA | DeepMIMO_ULA / DeepMIMO_UPA |
|---|---|---|
| Channels | GM-LAMP's generator `generate_channel.m`: L = 3 paths, half-wavelength spacing, lens DFT matrix; ULA 256 or UPA 16 × 16 | DeepMIMO v1.3, O1_28, BS 3, user rows 1000–1300 (54,481 users), 3 paths, one subcarrier; ULA 256 on the z axis, UPA 16 × 16 (y-z); x = U0ᴴh, ‖x‖² = 256 |
| Training | 80,000 (as GM-LAMP's `generate_channel_data_for_SV_channel_model.m`) | 50,000 |
| Validation | 2,000 | 2,000 |
| Test | 2,000 (as GM-LAMP's `Main.m`, K = 2000), fixed seed | **GM-LAMP's own test file** (`CE_DeepMIMO/<ARR>_test_channel.mat`, 2,000). Its users are removed from training/validation |
| CS matrix A (128 × 256) | `CSmatrix256128.mat` (GM-LAMP) | `CSmatrix_128256.mat` (GM-LAMP) |

**Measurements** follow GM-LAMP's `generate_measurement_data.py`: y = A(x + n), σ² = 10^(−SNR/10).
- Two bands: 0–10 dB and 10–20 dB. Each band gets one training file and one validation file.
- Training file: 10 chunks, each at an SNR drawn from U(band).
- Validation file: SNR = 10 dB, the validation SNR of GM-LAMP.

**Test noise is not stored.** `evaluate.py` adds it at 0, 2, …, 20 dB with fixed seeds, so every method sees exactly the same test measurements.

## 2. What you need

| Item | SV | DeepMIMO |
|---|---|---|
| bsce (`bsce_code.zip`) | ✔ | ✔ |
| `bsce/reference/gmlamp/` (`bsce_reference_part1/2/3.zip`, extracted where `bsce_code.zip` is) | ✔ (CS matrix) | ✔ (CS matrix + GM-LAMP test files) |
| Raw DeepMIMO channels | – | `H_ula_1.mat` and `H_upa.mat`. You have them already, so no MATLAB step is needed. |
| MATLAB | – | only to regenerate the raw files (section 5) |

## 3. Make the data (Spyder, bsce folder)
```
%cd "D:\Academic Profile\LADMM-MIMO\codes\bsce_code (6)\bsce"
%run scripts/make_gmlamp_data.py --sv --deepmimo-ula "D:/Academic Profile/LADMM-MIMO/codes/H_ula_1.mat" --deepmimo-upa "D:/Academic Profile/LADMM-MIMO/codes/H_upa.mat"
```
- Only SV: `--sv`.
- Only DeepMIMO: the two `--deepmimo-…` options.
- Expected end of the output:
```
  SV_ULA        OK   train 80,000 / val 2,000 / test 2,000
  SV_UPA        OK   train 80,000 / val 2,000 / test 2,000
  DeepMIMO_ULA  OK   train 50,000 / val 2,000 / test 2,000
  DeepMIMO_UPA  OK   train 50,000 / val 2,000 / test 2,000
all OK: identical to the data used so far.
```
- During DeepMIMO it also prints `1990 / 2000` (ULA) and `1991 / 2000` (UPA) GM-LAMP test channels found. That is expected: the others are nearest users, and they are removed as well.
- `DIFFERENT` means it is not the review's data:
  - SV: check that the scenario files were not changed.
  - DeepMIMO: check the raw file (v1.3, O1_28, BS 3, rows 1000–1300, ULA on z).

## 4. Files written (`data/<name>/`)
| file | content |
|---|---|
| `<name>_training_channel.mat` | `x`, 256 × 80,000 (SV) / 50,000 (DeepMIMO) |
| `<name>_validation_channel.mat` | `x`, 256 × 2,000 |
| `<name>_test_channel.mat` | `x`, 256 × 2,000 |
| `<name>_training_128_measurements_0to10dB.mat`, `…_10to20dB.mat` | `y`, 128 × K training measurements, plus `snr_db` of the 10 chunks |
| `<name>_validation_128_measurements_0to10dB.mat`, `…_10to20dB.mat` | `y`, 128 × 2,000 at 10 dB |
| `CSmatrix.mat` | `A`, GM-LAMP's matrix |
| `dataset_info.json` (+ `deepmimo_stats.json/.png` for DeepMIMO) | what was made, match with GM-LAMP |

The scenario files that use this data already exist: `configs/scenarios/sv_ula.yaml`, `sv_upa.yaml`, `deepmimo_ula.yaml`, `deepmimo_upa.yaml`.
```
%run scripts/train.py --scenario configs/scenarios/deepmimo_upa.yaml --train configs/train/ladmm_original.yaml --protocol gmlamp --device cuda
%run scripts/import_gmlamp_weights.py --scenario configs/scenarios/deepmimo_upa.yaml --runs outputs/DeepMIMO_UPA
%run scripts/evaluate.py --scenario configs/scenarios/deepmimo_upa.yaml --runs outputs/DeepMIMO_UPA --classical amp,omp
```

## 5. Only if the raw DeepMIMO files are lost (MATLAB)
```
addpath('D:\Academic Profile\LADMM-MIMO\codes\bsce_code (6)\bsce\matlab')
dm = 'D:\Academic Profile\LADMM-MIMO\codes\DeepMIMO-matlab-1.3';
deepmimo_make_H(dm, 'D:\Academic Profile\LADMM-MIMO\codes\H_ula_1.mat', 'ULA');
deepmimo_make_H(dm, 'D:\Academic Profile\LADMM-MIMO\codes\H_upa.mat', 'UPA');
```
- Requires `DeepMIMO-matlab-1.3\RayTracing Scenarios\O1_28\O1_28.params.mat` and the other O1_28 files.
- Defaults are GM-LAMP's: rows 1000–1300, BS 3, 3 paths, one subcarrier.

## 6. Beyond the GM-LAMP setting (other sizes, other SNR ranges)
- **Other SNR bands, same data:**
  - `train.py … --snr 0 20` (or any edges): missing measurement files are made automatically.
  - `--noise online`: fresh noise every minibatch.
- **Other sizes or splits:** `scripts/make_dataset.py` (RUNBOOK, section DATA). It keeps the same 2,000-channel test sets by default.
- **More than 54k DeepMIMO users:** `matlab/deepmimo_make_H_chunks.m`. This widens the user area beyond GM-LAMP's, so use it as a separate experiment.

## 7. A third data set: 3GPP TR 38.901 UMi at 28 GHz
`docs/3GPP_DATASET.md` is the full guide. One command makes `data/3GPP_UPA` and `data/3GPP_ULA` in the format of the
data sets above (80,000 / 2,000 / 2,000 by default):
```
%run scripts/make_3gpp_data.py                    (--sizes dong2019 | deepmimo | 100k, or --n-train/--n-val/--n-test)
```
- **The generator:** bsce's own implementation of the TR procedure (`bsce/data/tr38901.py`), checked against Sionna 2.2. It runs in the bsce environment.
- **Matched to the SV and DeepMIMO data:** arrays, λ/2 spacing, 28 GHz, narrowband, lens U0, ‖x‖² = N per channel, CS matrix, SNR bands.
- **Held-out test set:** validation and test users come from drops of their own, and are the same for every training size.
- **Meta file:** `3GPP_<ARR>_meta.mat` holds the position, LOS state, drop and large-scale parameters of every channel, per set.
- **Raw files of Sionna or QuaDRiGa** (`scripts/sionna_make_H.py`, `matlab/quadriga_make_H.m`) become a data set with `make_dataset.py --dataset 3GPP --input "<files>"`.
- **Check:** `scripts/check_3gpp.py --builtin UPA ULA_Y`.

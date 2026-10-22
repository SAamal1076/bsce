# bsce: complete manual, from installation to the paper figures

*Version of 14 Oct 2026. Every command below was run on this version of the package; the numbers quoted are measured
results. Commands are written for Spyder (`%run scripts/<name>.py ...`). In a terminal, use `python scripts/<name>.py ...`
with the same options.*

bsce trains and compares learned beamspace channel estimators for mmWave massive MIMO with a lens array:
- the LADMM family (your method)
- NF-LADMM (the proposal)
- LAMP and GM-LAMP (Wei, Hu & Dai)
- LISTA, ALISTA, LVAMP, ADMM-Net
- the classical AMP, OMP, SD and ADMM

Every port was checked against the original code (see `VERIFICATION_REPORT.md`).

**Contents**
0. The package · 1. Installation · 2. Concepts · 3. Data · 4. Methods · 5. Training schedules · 6. Training protocols ·
7. SNR ranges and one-network vs per-band · 8. Number of layers · 9. Running the training · 10. Evaluation ·
11. Figures and tables · 12. MATLAB · 13. Recipes (complete examples) · 14. Verification tests · 15. Tracking and
review · 16. Troubleshooting · 17. Reference numbers · Appendices: option tables, `docs/CLI_REFERENCE.md`

---

## 0. The package

| zip | content | needed for |
|---|---|---|
| `bsce_code.zip` | the code (`bsce/`, `scripts/`, `configs/`, `matlab/`, `tests/`), the docs (this file, `RUNBOOK.md`, `DATA_GUIDE.md`, `README.md`, `VERIFICATION_REPORT.md`, `docs/CLI_REFERENCE.md`), your best LADMM checkpoint in `models/` | everything |
| `bsce_reference_part1.zip` | `reference/gmlamp/CE_SV`, `reference/gmlamp/Train`: GM-LAMP's SV code, CS matrix and trained weights | data, comparison with GM-LAMP |
| `bsce_reference_part2.zip` | `reference/gmlamp/CE_DeepMIMO` (code, CS matrix, GM-LAMP's DeepMIMO test channels) | DeepMIMO data |
| `bsce_reference_part3.zip` | GM-LAMP's / LAMP's trained DeepMIMO weights | comparison with GM-LAMP on DeepMIMO |
| `bsce_reference_part4.zip` | original code of LADMM (yours), Borgerding, ALISTA, THADMM-Net | only the verification tests |

**Extract all five zips in the same folder.** Every path inside starts with `bsce/`, so you get one folder `bsce/` with everything in it.

Folder layout after extraction (the folders marked * are created by the scripts):
```
bsce/
  bsce/            the Python package (models, training, data, export)
  scripts/         every command of this manual
  configs/scenarios/   data sets (one .yaml per data set)
  configs/train/       methods and training protocols (one .yaml each)
  matlab/          MATLAB/Octave inference + DeepMIMO raw-channel generation
  reference/       original code and weights (from the reference zips)
  models/          your trained LADMM checkpoint(s)
  tests/           verification tests
  data/*           data sets (made in section 3)
  outputs/*        trained networks, logs, evaluation results, figures
  review/*         zips for review (section 15)
```

## 1. Installation and environment check

**With the workbench:** double-click `start_app.bat` → step 0 "Create the conda environment" runs the commands
below (`docs/WORKBENCH.md`). By hand:

**1.1 Conda environment** (Anaconda Prompt, inside the `bsce` folder):
```bat
conda env create -f environment.yml
conda activate bsce
nvidia-smi
pip install torch --index-url https://download.pytorch.org/whl/cu126
python -c "import torch; print(torch.__version__, torch.cuda.is_available(), torch.cuda.get_device_name(0))"
```
- `nvidia-smi` shows the CUDA version your driver supports (top right). Pick the matching PyTorch build on pytorch.org, under "Get started", with package "Pip".
- The last line must print `True` and your GPU name.
- Do not install torch without `--index-url`: on Windows the default wheel runs on the CPU only.
- Without conda, run `pip install -r requirements.txt` after the torch line.

**1.2 Spyder.** Run Spyder from this environment: `conda install spyder`, then `spyder`. Alternatively, select the interpreter of the `bsce` environment in Tools → Preferences → Python interpreter. Then, once per session:
```
%cd "D:\Academic Profile\LADMM-MIMO\codes\bsce_code (6)\bsce"
```
All commands are run from this folder.

**1.3 Check the machine** (nothing is changed):
```
%run scripts/check_env.py
```
- It lists packages, GPU, reference files and data sets as `[OK]` / `[--]`, with what to do for each missing item.
- Expected: `ready.` The data sets appear only after section 3.

**1.4 Quick self-test (2–5 min).**
```
!python -m pytest tests/test_golden.py tests/test_nfladmm.py tests/test_make_dataset.py -q
%run scripts/run_all.py --scenario configs/scenarios/sv_ula.yaml --smoke
```
- The first line checks short trainings against stored numbers, plus NF-LADMM and data making.
- The second line runs data → every method → evaluation on tiny data (the folder `outputs/SV_ULA_smoke/`).
- On the GPU, `!python -m pytest tests/test_gpu_vs_cpu.py -q` also checks the GPU against the CPU.

## 2. Concepts (read once)

| word | meaning | where |
|---|---|---|
| **scenario** | one data set: dataset (SV/DeepMIMO), array (ULA/UPA), CS matrix, sizes, folder, **bands**, test SNRs | `configs/scenarios/<name>.yaml` |
| **band** | an SNR range with its own network, e.g. `0to10dB` (training SNR ~ U(0,10)). A scenario has one or more bands | `bands:` in the scenario, or `--snr` |
| **train config** | a method and its own training settings (model + train) | `configs/train/<name>.yaml` |
| **protocol** | a set of training settings applied **on top of** any train config, so that all methods are trained the same way | `--protocol gmlamp` etc. (`configs/train/<p>_protocol.yaml`) |
| **schedule** | how the layers are trained: layer by layer + fine-tune, or all layers together | `train.schedule` |
| **tag** | the name of one trained method (folder name) | default from the model + protocol, or `--tag` |
| **run folder** | `outputs/<scenario name>/` holds one sub-folder per tag, and the evaluation of all of them | |
| **override** | change any value from the command line: `--set train.lr=5e-4 model.T=8` | every training script |

Output of one training: `outputs/<scenario>/<tag>/`. It contains:
- `<tag>_<band>.pt`: the network in PyTorch.
- `<tag>_<band>.mat`: the network for MATLAB.
- `<tag>_<band>_orig.mat`: for LAMP/GM-LAMP/LADMM, a drop-in file for the original MATLAB code.
- `train_log.json`: configuration, validation NMSE of every stage and depth, time, number of parameters.
- `checkpoints/`: an interrupted run resumes from here when the **same** command is run again. Any change of data, band or settings starts from scratch.

Devices and precision:
- `--device auto` (default) uses the GPU when one is visible, else the CPU. `--device cuda` / `--device cpu` force one.
- `--set model.precision=float32` (or `float64`) sets the arithmetic. float32 is much faster on consumer GPUs; `speed_test.py` measures it on your PC (section 9.6).

## 3. Data

### 3.1 GM-LAMP data sets (paper reproduction): one command
```
%run scripts/make_gmlamp_data.py --sv --deepmimo-ula "D:/Academic Profile/LADMM-MIMO/codes/H_ula_1.mat" --deepmimo-upa "D:/Academic Profile/LADMM-MIMO/codes/H_upa.mat"
```

| | SV_ULA / SV_UPA | DeepMIMO_ULA / DeepMIMO_UPA |
|---|---|---|
| channels | GM-LAMP's generator (`generate_channel.m`, L = 3, lens DFT) | DeepMIMO v1.3 O1_28, BS 3, rows 1000–1300 (54,481 users), 3 paths; x = U0ᴴh, ‖x‖² = 256 |
| train / val / test | 80,000 / 2,000 / 2,000 | 50,000 / 2,000 / 2,000 (test = **GM-LAMP's own test file**; those users removed from train/val) |
| CS matrix | `CSmatrix256128` | `CSmatrix_128256` |
| measurement files | bands 0–10 and 10–20 dB: training in 10 SNR chunks ~ U(band), validation at 10 dB | same |
| test noise | not stored: added by `evaluate.py` at 0, 2, …, 20 dB with fixed seeds | same |

- The script ends with a comparison against the data used so far. Expected: `4 x OK` and `all OK: identical to the data used so far`. Checked: a full regeneration is bit-identical.
- `--sv` alone makes SV only; the two `--deepmimo-…` options alone make DeepMIMO only.
- Without the raw DeepMIMO files, run MATLAB (see `DATA_GUIDE.md` section 5):
  ```
  deepmimo_make_H(dm, 'H_ula_1.mat', 'ULA'); deepmimo_make_H(dm, 'H_upa.mat', 'UPA');
  ```
- The scenario files for these data are `configs/scenarios/sv_ula.yaml`, `sv_upa.yaml`, `deepmimo_ula.yaml`, `deepmimo_upa.yaml`.

### 3.1b 3GPP TR 38.901 data sets: one command
```
%run scripts/make_3gpp_data.py                                  data/3GPP_UPA, data/3GPP_ULA: 80,000 / 2,000 / 2,000
%run scripts/make_3gpp_data.py --sizes dong2019                 81,000 / 9,000 / 19,000 (Dong et al., IEEE JSTSP 2019)
%run scripts/make_3gpp_data.py --n-train 96000 --n-val 2000 --n-test 2000 --only 3GPP_UPA,3GPP_ULA,3GPP_ULAZ
```
- **Channels:** 38.901 UMi street canyon, 28 GHz, LOS/NLOS by the standard, made by bsce's own generator (`bsce/data/tr38901.py`, Python only, about 2 min per set).
- **Format:** the same as SV/DeepMIMO: lens U0, ‖x‖² = 256, CS matrix of SV, the authors' bands. A meta file adds the position, LOS state and large-scale parameters of every channel.
- **Validation and test:** they come from drops of their own and are the same for any training size.
- **Scenario files:** `3gpp_upa.yaml`, `3gpp_ula.yaml`, `3gpp_ulaz.yaml`. In `run_suite.py` and `compare.py`, `--scenarios 3gpp` means the first two.
- **More:** options in `docs/3GPP_DATASET.md` section 3; the checks in section 5.

### 3.2 Any size, any split, any SNR range: `make_dataset.py`
```
%run scripts/make_dataset.py --dataset SV --array UPA --n 200000 --snr 0 20 --name SV_UPA_200k
%run scripts/make_dataset.py --dataset SV --array ULA --n-train 150000 --n-val 10000 --n-test 10000 --test random --snr 0 10 20 --name SV_ULA_170k
%run scripts/make_dataset.py --dataset DeepMIMO --array UPA --input "D:/.../H_upa.mat" --n-train 50481 --n-val 2000 --snr 0 20 --name DeepMIMO_UPA_54k
%run scripts/make_dataset.py --dataset 3GPP --array ULA --ula-axis y --sizes 100k --authors-bands --name 3GPP_ULA_100k
```

| option | meaning |
|---|---|
| `--n 200000 --split 0.90 0.05 0.05` | total and fractions train/val/test (default) |
| `--n-train / --n-val / --n-test` | exact sizes instead |
| `--sizes sv\|deepmimo\|dong2019\|100k` | a (train, val, test) preset: 80000/2000/2000, 50000/2000/2000, 81000/9000/19000, 96000/2000/2000 |
| `--dataset 3GPP` (no `--input`) | the built-in 3GPP generator; `--ula-axis y\|z`, `--spec 16.1\|19.2`, `--channel umi\|umi_los\|umi_nlos`, `--element 38.901\|omni`, `--drop-users`, `--raw-out` |
| `--test standard` (default) | the 2,000 standard test channels: bsce SV test set (bit-identical to `data/SV_*`), or GM-LAMP's DeepMIMO test file. Comparable with every table. |
| `--test random` | a test split of size n_test from the same pool |
| `--snr 0 20` / `0 10 20` / `-5 0 5 10 15 20 25` | band edges: one training + one validation measurement file per band, 100 SNR chunks ~ U(band), SNR of every column stored (`--snr-dist fair` in `train.py` / `make_sv_measurements.py` makes the per-sample files later) |
| `--no-measurements` | channels only (train with `--noise online`, or add bands later) |
| `--test-measurements` | also test measurement files with the evaluation noise (for MATLAB) |
| `--seed`, `--out-dir`, `--force` | seed (default 2026), other folder, overwrite |

**What it writes.**
- In `data/<name>/`: channels, measurement files, CS matrix, `dataset_check.png` and `dataset_info.json`.
- **A ready scenario file** `configs/scenarios/<name lower case>.yaml`.
- What to check in `dataset_check.png`: energy ≈ 256 in all three sets, matching strongest-beam histograms, and SNRs spread over each band.

**DeepMIMO sizes.**
- Rows 1000–1300 hold 54,481 users. After removing GM-LAMP's 2,000 test users, at most 52,481 remain for train + val.
- More users need more rows in MATLAB: `matlab/deepmimo_make_H_chunks.m`, e.g. 200k users = rows 1000–2120. These rows lie outside GM-LAMP's user area, so use them as a separate "more data" experiment. The script tells you which rows are needed if the input is too small.

### 3.3 Noise during training: files or online
- **`--noise file`** (default for every method except NF-LADMM) uses the measurement files of the band.
  - If a file is missing, `train.py` makes it (100 SNR chunks ~ U(band)).
  - To make files ahead of time: `%run scripts/make_sv_measurements.py --scenario <yaml> --snr 0 5 10 15 20`. It works for SV and DeepMIMO.
- **`--noise online`** (default for NF-LADMM) draws fresh noise every minibatch, one SNR per sample ~ U(band).
  - Validation uses a fixed generated set (same seed for every method with the same band).
  - It gives an unlimited number of (x, y) pairs from the same channels, and needs no measurement files.

### 3.3b Noise model: colored (GM-LAMP) or white (`--noise-model`; `docs/NOISE_MODELS.md`)
- **`colored`** (default): y = A(x + n), n ~ CN(0, σ²I_N), noise covariance σ²AAᵀ (noise folding, GM-LAMP's model).
- **`white`**: y = Ax + w, w ~ CN(0, cσ²I_M), c = tr(AAᵀ)/M (= N/M for GM-LAMP's ±1/√M combiner). At the same SNR the
  signal and the mean noise power are the same as for `colored`.
- **Paired draws.** bsce makes the white noise from the colored draw, w = √c (AAᵀ)^(−1/2) A n
  (`bsce/data/noise.py`). Every white measurement file, validation set and test set therefore has the same channels and
  random numbers as its colored twin. The colored files and results are bit-identical to earlier versions.
- **Where it is set:**

  | script | option |
  |---|---|
  | `run_suite.py`, `train.py` | `--noise-model white` (or the scenario key `noise_model`); runs go to `outputs/<SCENARIO>_white/` under the same names |
  | `finetune_bands.py` | takes the noise of its source network (`--noise-model` to override) |
  | `evaluate.py` | `--noise-model auto` (default: white for a `<SCENARIO>_white` folder) / `colored` / `white`; `--nf-step matched` |
  | `compare.py` | `--noise-model white`: the white runs on white test noise |
  | `compare_noise.py` | both noise models side by side, with cross tests |
  | `make_dataset.py` | `--noise-model colored` / `white` / `both` for the measurement files (and `--test-measurements`) |
  | `make_sv_measurements.py` | `--noise-model white` |
  | `depth_snapshots.py` | the noise the run was trained with |

- **White files** are named `<colored name>_white.mat`. They are made from the colored file when it exists (the
  authors' files too), else with the same random numbers. They store `y`, `snr_db` and `noise_model`.
- **NF-LADMM** builds the exact data step of `model.noise_model`, which `train.py` sets to the data's noise model:
  - colored: x = v + β A⁺(y − Av);
  - white: x = v + Aᵀ(AAᵀ + c ρ̃ I)⁻¹(y − Av), with ρ̃ = 1/β − 1.

  The parameters are the same in both cases. `--set model.noise_model=colored` on white data gives the mismatch on purpose.

### 3.4 Scenario variants without new data: `make_scenario.py`
```
%run scripts/make_scenario.py --base sv_ula --name SV_ULA_fast --n-train 15000 --n-val 2000 --n-test 2000 --val-random 10
%run scripts/make_scenario.py --base sv_upa --name SV_UPA_valrange --val-points 5
%run scripts/make_scenario.py --base sv_ula --name SV_ULA_single --single-band
```
- What it does: other sizes, several validation SNRs per band (for `--protocol gmlamp_best`), or one band 0–20 dB.
- Next, SV: `%run scripts/generate_data.py --scenario configs/scenarios/<name>.yaml`.
- Next, DeepMIMO: `deepmimo_convert.py`, then `generate_data.py --measurements-only`.

### 3.5 Your own SV files (`data/SV`, the data of `untitled2.py`)
- Scenarios `sv_upa_user.yaml` / `sv_ula_user.yaml`: one band 0–20 dB, validation file `*_1_11dB.mat`, test on `data/SV_UPA`.
- Scenarios `sv_*_user_2band.yaml`: the same channels with two bands.
- ULA measurements: `%run scripts/make_sv_measurements.py --scenario configs/scenarios/sv_ula_user.yaml`.

### 3.6 File format (all `.mat`, readable by MATLAB and scipy)
| file | variable |
|---|---|
| `<D>_<ARR>_{training,validation,test}_channel.mat` | `x` complex N × K (N = 256) |
| `<D>_<ARR>_{training,validation}_128_measurements_<band>.mat` | `y` complex M × K (M = 128), `snr_db` (per column or per chunk) |
| `CSmatrix.mat` | `A` real M × N |

## 4. Methods

| train config | method | parameters (N = 256) | own schedule |
|---|---|---|---|
| `lamp_original` | LAMP (Wei-Hu-Dai, T = 8) | 262,160 | layer-wise + fine-tune |
| `gmlamp_original` / `gmlamp_spread` | GM-LAMP (K = 4), original / slightly spread init (recommended, reproducible). v31.1: the trainer gives the original init's identical components the spread offset before training them (`train.gm_symmetry`); networks trained with the original init before v31.1 have a one-Gaussian denoiser and must be retrained (`check_gm_components.py`) | 262,240 | layer-wise + fine-tune |
| `ladmm_original` | **LADMM**, full 256×256 operator per layer (your `LADMM_train.py`, T = 6) | 393,234 | layer-wise + fine-tune |
| `ladmm_best` | **LADMM** + the protocol of your best run (`untitled2.py`) | 393,234 | layer-wise + 4 fine-tunes |
| `ladmm_diag` / `ladmm_lowrank` / `ladmm_kron` | LADMM with diagonal / low-rank (r = 16) / Kronecker operator (your A/B/C scripts) | 1,554 / 49,170 / 3,090 | layer-wise |
| `ladmm_psd` / `ladmm_kronprior` / `ladmm_diagprior` | proposed exact-ADMM LADMM: PSD Gram / Kronecker prior / diagonal prior | 393,234 / 3,090 / 1,554 | layer-wise |
| `ladmm_group` / `ladmm_group_k` | LADMM with group threshold (fixed κ = 3 / learned κ) | 393,234 / 393,240 | layer-wise |
| `ladmm_original` + `--set model.threshold=adaptive\|firm\|gm` (v28) | LADMM with a noise-adaptive shrinkage: soft at c_t·σ̂, firm (no bias above t2), or Gaussian-mixture MMSE (4 components, exact complex density); σ̂ = median\|w\|/√ln 2 per channel, no σ² given | 393,234 / +1 / +8 per layer | layer-wise |
| `ladmm_original` + `--set model.data_step=whitened` (v28) | LADMM whose data step is exact for the folded noise y = A(x+n): A⁺y in place of Aᵀy, AᵀA → P = Aᵀ(AAᵀ)⁻¹A in every operator | as the operator | layer-wise |
| `ladmm_original` + `--set model.operator=analytic` / `analytic1` (v28) | LADMM-A: (AᵀA+ρ_tI)⁻¹ by the SVD of A, nothing learned in the operator; works for any combiner. `analytic1` fixes α = 1 (ADMM-CE-Net form) | 3 / 2 per layer (18 / 12 for T = 6) | layer-wise |
| `gladmm` / `gladmmf` / `gladmmany` (v29) | **Graph-LADMM**: unrolled ADMM on the whitened folded model with a magnetic beam-graph prior. A second split z₂ = Dx carries a graph total variation on the edges of the beam ring / torus, whose edge phase is estimated from the training channels. The z-step (`model.zstep`): `gtv_gate` (TV + energy gate, default of `gladmm`), `gate` (default of `gladmmf`), `gtv_gm`, `gtv`, `gm`, `gsoft`, `l1`, `gm_graph`, `gtv_gm_graph`. `gladmm`: analytic x-step (P+ρ(I+DᴴD))⁻¹, nothing learned for one combiner, so it works with any combiner and data set. `gladmmf`: a free H_t per layer (one combiner, like LADMM-F). `gladmmany`: `gladmm` trained on a new random combiner every minibatch. Noise level from the residual (`model.sigma=residual`; `blind`, `known`). | gladmm T = 8: 54 (v31: every one trained; 56 stored before, of which 54 were trained); gladmmf T = 6: 393,250 | `gladmm`, `gladmmany`: always end to end with their own settings in `run_suite.py` (v30; `--force-protocol` for the common one); `gladmmf`: `--protocol` like every method |
| `gladmm_mixA` / `gladmm_bp` / `gladmm_rwl1` / `gladmm_rwl1_tv` (v31; keys `gladmmmix`, `gladmmbp`, `gladmmrw`, `gladmmrwtv`) | Graph-LADMM trained on a mixture of combiners (`train.combiner=mix`: M ∈ {64, 96, 128, 160}, random ±1 / Gaussian / partially connected / DCT rows); + the beam prior (`model.beam_prior=gate`: gate logit + β_t ℓ_i, ℓ = log beam power from the training channels); the graph-reweighted l1 unrolled (`model.zstep=rwl1`, `model.data_step=plain`: the uploaded package's method); the reweighted l1 in gladmm's layer (`gtv_rwl1`) | 54 / 318 / 44 / 51 | end to end, their own settings (as `gladmm`) |
| `cadmm` (v28) | complex ADMM-Net with untied M1 (N×M), M2 (N×N) and over-relaxation (Johnston, Li, Lops, Wang, IEEE TSP 2021); LADMM-F is its tied case with one ρ for all layers | 589,842 (T = 6) | layer-wise (LADMM's) |
| `nfladmm_s` | **NF-LADMM-S**: exact noise-folded x-step + circular-CNN prior, one network for all SNRs | 9,066 (UPA) / 3,434 (ULA) | end-to-end |
| `nfladmm_c` | NF-LADMM-C: radial shrinkage only (certified variant) | 160 | end-to-end |
| `nfladmm_sr` | **NF-LADMM-SR** (suggested, T = 4): the NF-LADMM x-step + radial shrinkage and circular CNN, over-relaxed update (relaxed Douglas–Rachford), slope init ±2 | 8,982 (UPA) / 3,350 (ULA) | end-to-end |
| `lista` / `alista` | LISTA (untied) / ALISTA (analytic W; v31: any combiner, `set_combiner`) | 1,081,360 (1,015,824 trained: W_1 multiplies x₀ = 0) / 32 | ALISTA schedule |
| `lvamp` | LVAMP (Borgerding); its LMMSE stage assumes white noise. `model.noise_model=colored` (v27) uses the weights of folded noise y = A(x+n) (uniform in the SVD domain): matched in theory, but worse after training on SV ULA (test mean −7.83 vs −8.41 dB end to end, −5.66 vs −5.76 dB with its own protocol; `docs/changes/CHANGES_2026-10-17_lvamp_forms.md`) | 13 | layer-wise |
| `admmnet` | ADMM-Net (THADMM-Net loop) | 1,966,110 real numbers = 983,070 complex entries (complex W counted as 2 real; both counts are right in their unit); the last β never reaches the output | end-to-end epochs |
| `utrr` | UTRR (Wu, Cheng, Eldar, Cioffi, IEEE TCOM 2025), T = 30, top-K 16 in the last layer (`docs/UTRR.md`) | 1,966,140 (with `model.bn=true`: 1,997,884) | end-to-end epochs, 5 learning rates with early stopping |
| (`make_ensemble.py`) | model-averaging ensemble of trained networks (UTRR-E: K = 0, 16, 64, 256) | sum of the members | not trained |
| classical (`--classical amp,omp,sd`) | AMP, OMP (24 atoms), SD; `admm` with `--admm-lambdas`; v31: `l1` (300-iteration l1 ADMM) and `grwl1` (graph-reweighted l1, the uploaded package), λ / ε per test SNR from `scripts/tune_classical.py` (`--classical-params`) | – | – |

**Stored and trained parameters (v31).** compare.py's tables give both (bsce/cost.py). A trained parameter is one whose
gradient is not zero on two random probe batches.
- **Never reach the output, so never change from their initial value** in the trained networks:
  - LISTA's W_1 (65,536: it multiplies x₀ = 0);
  - CADMM's first M2 (65,536) and its last η;
  - one of LAMP's two θ entries per layer (so the authors' LAMP trains 262,152 of 262,160);
  - the last dual step of LADMM (α), ADMM-Net (β) and NF-LADMM.
- **Kept as they are.** These methods keep their stored form, for compatibility with the authors' files and older
  networks.
- **Graph-LADMM (v31)** stores only trained parameters, and older Graph-LADMM files load (§4 table).

Model options are changed with `--set model.<option>=<value>` (full list in Appendix A2). The common ones:
- `model.T`: number of layers.
- `model.precision`: `float32` / `float64`.
- `model.operator` (LADMM): `full`, `diag`, `lowrank`, `kron`, `psd`, `kronprior`, `diagprior`.
- `model.threshold` (LADMM): `scalar`, `group`, `group_k`, `group_kc`, and (v28) `adaptive`, `firm`, `gm` (+ `model.gm_K`).
- `model.data_step` (LADMM, v28): `plain` (default, the scripts) or `whitened` (exact for the folded noise).
- `model.operator` (LADMM, v28): also `analytic`, `analytic1`.
- Graph-LADMM (v29): `model.zstep`, `model.xstep` (`analytic`, `free`), `model.sigma` (`residual`, `blind`, `known`), `model.edge_phase` (`data`, `theory`, `blind` or degrees), `model.learn_phase`, `model.ring`, `model.lr_mult`; v31: `model.zstep=rwl1|gtv_rwl1`, `model.data_step=whitened|plain`, `model.beam_prior=none|gate`.
- `train.combiner=random` (v29): a new random combiner every minibatch (online noise; networks that follow the combiner).
- `train.combiner=mix` (v31): every draw takes an M from `train.combiner_M` and a family from `train.combiner_types` (`pm1`, `gauss`, `sparse`, `dct`).
- `model.rank` (LADMM low-rank).
- `model.denoiser` (NF-LADMM): `conv` or `radial`.
- `model.hidden`, `model.kernel` (NF-LADMM CNN size).
- `model.sigma` (NF-LADMM): `known` gives the test noise variance; `residual` estimates it.

## 5. Training schedules (how the layers are trained), `train.schedule`

| schedule | what happens | used by |
|---|---|---|
| `borgerding` (**layer by layer + fine-tune**) | For each layer t: (1) train only the new parameters of layer t at `lr`; (2) **fine-tune** all parameters so far at `lr × refinements` (e.g. 0.5, 0.1, 0.01). Each sub-stage stops after `better_wait` steps without improvement or at `maxit`. Validation every `ivl` steps. `warm_start: true` starts a new layer from the trained previous one (GM-LAMP). LADMM has two stages per layer (H-update, then threshold). | LAMP, GM-LAMP, LADMM (all operators), LVAMP |
| `steps` (**all layers together, end to end**) | All parameters from the start, `steps` minibatches, Adam with cosine decay from `lr` to `lr × lr_final_frac`, optional `grad_clip`, validation every `val_every` steps, best state kept (`restore_best`) | NF-LADMM; any method with `--protocol e2e` |
| `end2end` | All parameters, `epochs` passes over the training set (THADMM-Net loop) | ADMM-Net |
| `alista` | Per-layer training with per-variable learning-rate decay (ALISTA repository) | LISTA, ALISTA |

Variants of the layer-wise schedule:
- **Without fine-tune:** `--set train.refinements=[]`.
- **More fine-tune stages:** `--set "train.refinements=[0.5,0.2,0.1,0.01]"`.
- **Best weights at the end of every stage:** `--set train.restore_best=true`.

## 6. Training protocols (the same training for every method), `--protocol`

A protocol overrides the training part of any train config (the model stays). Available for `train.py`, `run_all.py`, `select_depth.py` and `compare_variants.py`.

| protocol | schedule | lr | fine-tune | batch | max steps / stage | patience | warm start | weights kept | validation | optimizer | use it for |
|---|---|---|---|---|---|---|---|---|---|---|---|
| `paper` (= `original`; the default of `train.py --train <yaml>`) | each method's own | own | own | own | own | own | own | last | own | own | each method as its authors trained it |
| `gmlamp` | layer-wise | 1e-3 | 0.5, 0.1, 0.01 | 128 | 1,000,000 | 5,000 | yes | last | pooled NMSE, round 5 | TF1 Adam | **paper reproduction**, GM-LAMP's procedure for every method |
| `gmlamp_best` | layer-wise | 1e-3 | 0.5, 0.1, 0.01 | 128 | 1,000,000 | 5,000 | yes | **best validated** | **mean in dB over the band's validation SNRs** | TF1 Adam | improved training (scenario with several validation SNRs: `make_scenario.py --val-points 5`) |
| `ladmm_best` | layer-wise | 5e-4 | 0.5, 0.2, 0.1, 0.01 | 400 | 100,000 | 5,000 | no | last | pooled NMSE, round 5 | torch Adam, float32 | **your best LADMM** (`untitled2.py`, reproduced exactly) |
| `fair` | layer-wise | 1e-3 | 0.5, 0.1, 0.01 | 512 | 50,000 | 3,000 | no | last | pooled NMSE | TF1 Adam | equal budget for every method |
| `e2e` | **all layers together** | 1e-3 cosine → 5e-5 | – | 256 | 30,000 steps in total | 5,000 steps without a 0.01 dB gain (v25) | – | best validated | every 500 steps | torch Adam, clip 1 | end-to-end training for any method |

Examples:
```
%run scripts/train.py --scenario configs/scenarios/sv_upa.yaml --train configs/train/ladmm_original.yaml --protocol gmlamp
%run scripts/train.py --scenario configs/scenarios/sv_upa.yaml --train configs/train/ladmm_original.yaml --protocol ladmm_best --snr 0 20
%run scripts/train.py --scenario configs/scenarios/sv_upa.yaml --train configs/train/ladmm_original.yaml --protocol e2e --snr 0 20 --set train.steps=50000
%run scripts/train.py --scenario configs/scenarios/sv_upa_valrange.yaml --train configs/train/gmlamp_spread.yaml --protocol gmlamp_best
```
- Default tag: `<model>_<protocol>`, e.g. `LADMM-full_e2e`.
- Every protocol value can still be changed with `--set`.

## 7. SNR ranges: one network or one per band

- **The scenario's bands** (GM-LAMP: 0–10 and 10–20 dB) are the default.
- **Any bands:** `--snr 0 20` (one network), `--snr 0 10 20` (two), `--snr 0 5 10 15 20` (four), `--snr -5 25` …
  - `evaluate.py` reads each network's band from its file name and uses, at each test SNR, the network whose band contains it.
- **Per-band fine-tune of a 0–20 dB network.** This gave the best LADMM so far on SV UPA (`LADMM_why_0to20_SV.md`):
  ```
  %run scripts/finetune_bands.py --from outputs/SV_UPA/LADMM-T6 --device cuda          (v24: a run folder, automatic name)
  %run scripts/finetune_bands.py --scenario configs/scenarios/sv_upa_user_2band.yaml --snr scenario --from outputs/SV_UPA_user/LADMM-best/LADMM-best_0to20dB.pt --device cuda
  %run scripts/finetune_bands.py --scenario configs/scenarios/sv_upa_user_2band.yaml --from models/Trained_LADMM_for_SV_UPA_2561280to20dB_1_11dB_better.pt --tag LADMM-untitled2-better-ft
  ```
  - Options: `--snr 0 10 20` (any bands, independent of the scenario), `--val range` (default: validation over the whole band) or `--val file`, `--iters 6000 --batch 400 --lr 5e-5 --lr2 5e-6 --val-every 100 --bands 0to10dB --tag NAME`.
  - Example (DeepMIMO UPA, two bands): `%run scripts/finetune_bands.py --scenario configs/scenarios/deepmimo_upa.yaml --snr 0 10 20 --from outputs/DeepMIMO_UPA/LADMM-full/LADMM-full_0to20dB.pt --tag LADMM-full-ft --device cuda`.
  - It works with any 2-band (or n-band) scenario.
  - Measured: 2.25 dB mean gain over GM-LAMP, against 1.36 dB for the 0–20 dB network.
- **NF-LADMM** is one network for the whole band by design: the noise level enters every layer.
- **Data options of a band** (`train.py`, `run_suite.py`; `finetune_bands.py` has `--val-snr` and `--margin`; GUIDE 4.3):
  - `--val-snr range | <SNRs>`: validation over the band, or at fixed SNRs. The authors use 10 dB for both bands.
  - `--margin M`: train on [lo − M, hi + M], use on [lo, hi].
  - `--snr-chunks N`: SNR chunks per training file (the authors 10, ranges 100).
  - `--snr-dist fair [--snr-step S]` (v26; also `finetune_bands.py` and `make_sv_measurements.py`):
    - every training column its own SNR, spread evenly over the band (stratified: one SNR in each of K equal parts, random order), or on the grid lo, lo + S, …, hi with equal counts;
    - online noise: every minibatch spread the same way;
    - validation over a range: equally spaced SNRs (the middles of 100 equal parts) instead of random ones.
    - Files `<band>_fair[<S>].mat`, folders `_fair[<S>]`. The default (chunks) is unchanged.
  - All four are code in `bsce/data/files.py::apply_band_options` (the draw: `bsce/data/measurements.py::fair_snr`); tests in `tests/test_band_options.py` and `tests/test_snr_dist.py`.

## 8. Number of layers T (LADMM and NF-LADMM in one plot)
```
%run scripts/select_depth.py --scenario configs/scenarios/sv_upa.yaml --snr 0 20 --T 2 4 6 8 10 12 --train configs/train/ladmm_best.yaml configs/train/nfladmm_s.yaml --set train.steps=10000 --device cuda
```
- **Layer-wise methods:** one run with T = max T. The validation NMSE logged when each depth is finished gives the whole curve. Do not use the prefixes of the final network: later fine-tuning retunes the early layers.
- **End-to-end methods:** one run per T.
- **Output:** `outputs/depth_select/<scenario>/depth_select.png` (one panel per band, values printed, T* starred), plus `.csv` / `.json`.
  - T* = the smallest T within `--tol 0.1` dB of the best.
  - Then train the final network with `--set model.T=<T*>`.
- Options: `--protocol`, `--noise`, `--retrain`, `--runs-only` (read existing runs), `--smoke`.
- Older single-method tool: `%run scripts/depth_curve.py --runs outputs/SV_ULA/LADMM-full-T12_gmlamp`.

### 8.1 Save the network after every layer and plot the TEST depth curve (`--save-layers`)
Layer-wise training only (`ladmm_best`, `gmlamp`, `fair`, and `paper` for the layer-wise methods). One run with T = 12 saves 12 real estimators. With an end-to-end protocol `train.py` refuses `--save-layers` (v25). `run_suite.py` refuses it only when no method of the list trains layer by layer; in a mixed list the end-to-end methods train at full depth without snapshots (v31.2).
```
%run scripts/train.py --scenario configs/scenarios/deepmimo_upa.yaml --train configs/train/ladmm_best.yaml --snr 0 20 --set model.T=12 --tag LADMM-best-T12 --save-layers --device cuda
%run scripts/train.py --scenario configs/scenarios/deepmimo_ula.yaml --train configs/train/ladmm_best.yaml --snr 0 20 --set model.T=12 --tag LADMM-best-T12 --save-layers --device cuda
%run scripts/depth_snapshots.py --runs outputs/DeepMIMO_UPA/LADMM-best-T12 outputs/DeepMIMO_ULA/LADMM-best-T12
```
- When layer t is finished (with all its refinements), the network **as it is at that moment** is saved:
  `outputs/<scenario>/<tag>/layers/T01_<tag>_<band>.pt/.mat/_orig.mat … T12_…`.
  - This is a real t-layer estimator. On the CPU it is bit-identical to a separate run with `model.T=t`, same data and seed (weight difference 0).
  - So one T = 12 run replaces the T = 1 … 12 runs.
- `depth_snapshots.py` evaluates every snapshot on the scenario's test set, with the noise of `evaluate.py`. It writes to `<first run>/depth_snapshots/`:
  - `depth_curve.png` / `.pdf`, two rows per run. Top: test NMSE vs T at 0 / 10 / 20 dB plus the mean over all test SNRs. Bottom: validation NMSE vs T plus the gain of each added layer. All values are printed.
  - `depth_curve.csv` and `depth_curve.md`.
  - The flat point: the smallest T within `--tol` dB (default 0.1) of the best mean test NMSE.
- The test curve shows what pooled validation hides: high-SNR NMSE keeps improving with depth after low-SNR NMSE has flattened.
- Any snapshot can be fine-tuned per band like a normal network:
  `%run scripts/finetune_bands.py --scenario configs/scenarios/deepmimo_upa.yaml --snr 0 10 20 --from outputs/DeepMIMO_UPA/LADMM-best-T12/layers/T06_LADMM-best-T12_0to20dB.pt --tag LADMM-T06-ft --device cuda`
- Options: `--band`, `--snr 0 10 20` (curves drawn), `--tol`, `--device`, `--out`.
- An interrupted run resumes from its checkpoint when the same command is run again; the snapshots already written are kept.

## 9. Running the training

### 9.1 One method: `train.py`
```
%run scripts/train.py --scenario sv_upa --method ladmm6 [--protocol P] [--snr edges|scenario] [--seed k] [--device cuda] [--set key=value ...]
%run scripts/train.py --scenario <scenario yaml> --train <train yaml> [--protocol P] [--snr edges] [--noise file|online] [--bands 0to10dB] [--tag NAME] [--seed k] [--device cuda] [--set key=value ...]
```
- **The short form (v24):** `--scenario <name> --method <key><layers>`, with the method keys of `run_suite.py`.
  - The defaults are `run_suite.py`'s: `--protocol ladmm_best`, `--snr 0 20`.
  - The folder name is `run_suite.py`'s too (`LADMM-T6`, `LAMP-T8_paper_trscenario`, …), so both commands make and resume the same run.
- **The long form:** any train config.
  - The default protocol is `original` (= `paper`, the method's own script).
  - The folder is named by the model (`LADMM-full`, `GM-LAMP`, …) unless `--tag` is given.
- **Other options:**
  - `--bands` trains only some bands.
  - `--seed k` (k > 0) adds `_s<k>` to the folder.
  - `--smoke` runs a tiny CPU check.
- **Resume:**
  - An interrupted run resumes when the same line is run again.
  - The checkpoint is written after every stage, and inside a stage every `train.checkpoint_minutes` (default 10). It holds the weights, the optimizer, the random streams, the best state and the position.
  - The resumed run gives exactly the network of an uninterrupted one (`tests/test_resume.py`: layer-wise with and without warm start, `steps`, `end2end` with `lr_list`).
  - `--set train.checkpoint_minutes=0` saves only after every stage, as before v24.

### 9.2 Several methods on one scenario: `run_all.py`
```
%run scripts/run_all.py --scenario configs/scenarios/sv_upa.yaml --protocol gmlamp --skip-data --configs lamp_original gmlamp_spread ladmm_original --set model.precision=float32
```
- `--configs` lists names in `configs/train` (default: 14 methods).
- `--skip-data` uses the existing data; without it, the SV data are generated first.
- `--no-eval` skips the evaluation; `--seed`, `--snr`, `--noise` and `--set` apply to every config.
- Logs go to `outputs/<scenario>/logs/`.
- A `--set model.T=…` meant for LADMM changes every method in the call, so run LADMM configs in a separate call from LAMP/GM-LAMP.

### 9.3 Variants of one method on several scenarios: `compare_variants.py`
```
%run scripts/compare_variants.py --scenarios sv_upa deepmimo_upa --variants scalar:model.threshold=scalar group:model.threshold=group --protocol gmlamp_best --seeds 0 1 2 --out outputs_threshold --refs LAMP-authors GM-LAMP-authors
```
It trains every variant, evaluates them on the same test data, and writes `compare.png` / `.csv` / `.json` (difference to the first variant, mean over seeds).

### 9.4 Seeds (mean ± std)
- Run the same command with `--seed 1`, then `--seed 2`.
- Evaluate.
- Then `%run scripts/aggregate_seeds.py --csv outputs/SV_ULA/nmse_vs_snr.csv` (or simply `make_figures.py`, which averages the `_s<k>` columns itself).

### 9.5 Any value from the command line
- `--set train.lr=5e-4 train.batch_size=512 train.maxit=50000 model.T=8 model.precision=float32`. Lists are written `"train.refinements=[0.5,0.1]"`.
- Every option is in Appendix A1 (training) and A2 (models).

### 9.6 Device and precision
- `%run scripts/speed_test.py` trains 2-layer networks on cuda/cpu × float32/float64 and reports the fastest setting per method in `outputs/speed_test.json` / `.png`.

### 9.7 The same training and fine-tune for every method: `run_suite.py` (final workflow)
`run_suite.py`, `depth_snapshots.py`, `promote_layer.py` and `compare.py` form the final workflow. **GUIDE.md** describes it step by step.

In short:
```
%run scripts/run_suite.py --scenarios all --methods all --device cuda      # every method, paper depths, ladmm_best, 0-20 dB + fine-tune
%run scripts/compare.py --scenarios all --preset ft --name final_finetuned
```
**Names:**
- trained: `<METHOD>-T<T>[_<protocol>][_tr<edges>][_<option>][_s<seed>]`;
- fine-tuned: `<that>_ft_<edges>dB_<iterations>[_margin<M>]`, e.g. `LADMM-T6_ft_0-10-20dB_20k`.
- The iterations are in the name, so two budgets never share a folder.
- Fine-tunes named before v24 (`<that>-ft[<edges>]`) are skipped when their budget matches; `rename_runs.py` renames them.

**Resume:** the same command continues an interrupted suite. Finished runs are skipped, and the one that stopped continues from its checkpoint.

**Early stop (v25):**
- **End to end** (`--protocol e2e`, `--schedule e2e`, NF-LADMM's own training): a training stops when the best validation has not improved by 0.01 dB during 5000 steps (`--set train.patience_steps=…`, `train.min_gain_db=…`; `train.patience_steps=0` = every step).
- **Fine-tune:** 4000 steps without a 0.01 dB gain end the first learning rate, then the band (`--ft-patience N`; 0 = the whole `--ft-iters`, folder suffix `_pat0`).
- **Layer-wise:** unchanged (`better_wait`).
- **`--save-layers`** needs a layer-wise protocol. v31.2: in a mixed list, the methods that train end to end (e.g. `gladmm`, whose own training is end to end) are trained at full depth without snapshots, with a note; only when no method of the list trains layer by layer does the suite stop before training.
- GUIDE 4.6 has the table.

`run_suite.py --help` lists every option:
- methods with any depth: `ladmm4`, `lamp8`, `lista12`;
- `--protocol`;
- `--train-snr`, `--ft-snr`, `--ft-iters`, `--ft-patience`, `--snr-dist`, `--snr-step`;
- `--save-layers`, `--seed`, `--set`;
- `--ensemble` (v28);
- `--force-protocol` (v30);
- `--smoke`, `--dry-run`.

**Graph-LADMM (v30):** `gladmm` and `gladmmany` always train end to end with their own settings (`configs/train/gladmm*.yaml`:
online noise, log-NSE loss, 30,000 steps), whatever `--protocol` is, so their folders end in `_paper`
(`G-LADMM-T8_paper`). On SV ULA the layer-wise GM-LAMP procedure at the same budget was 1.7 dB worse. `--force-protocol`
gives them `--protocol` like every method. `gladmmf` takes `--protocol`.

The fine-tune default is 20,000 iterations per band.

**Ensemble (v28, off by default):** `--ensemble ladmm6 ladmm8` averages the final networks of these methods (fine-tuned when there is a fine-tune) into `ENS-LADMM-T6+LADMM-T8_ft_0-10-20dB_20k`, one network per band (`make_ensemble.py`). The members are named as this command names them and need not be in `--methods`. The ensemble is made when they all exist and skipped when it exists. It costs the members' sum in parameters and inference time; the pilot found a gain only when the members are within about 1.5 dB of each other.

**CADMM (v28):** `--methods cadmm` (T = 6) trains the complex ADMM-Net baseline with the same protocol. It is not part of `all`, so `--methods all` trains what it trained before.

## 10. Evaluation
```
%run scripts/import_gmlamp_weights.py --scenario configs/scenarios/sv_upa.yaml --runs outputs/SV_UPA
%run scripts/import_user_ladmm.py --scenario configs/scenarios/sv_upa.yaml --runs outputs/SV_UPA --pt models/Trained_LADMM_for_SV_UPA_2561280to20dB_1_11dB_better.pt --tag LADMM-untitled2-better
%run scripts/evaluate.py --scenario configs/scenarios/sv_upa.yaml --runs outputs/SV_UPA --classical amp,omp,sd
```
- The first line adds the authors' trained LAMP / GM-LAMP (bands 0–10 / 10–20 dB) as `LAMP-authors` / `GM-LAMP-authors`.
- The second adds your untitled2.py checkpoint.
- `evaluate.py` tests **every model folder** of the run folder on the scenario's 2,000 test channels at 0, 2, …, 20 dB.
  - The test noise is the same for every method (MATLAB-style noise, fixed seed).
  - Each network is used in its own band; NF-LADMM gets the noise variance.
  - Options: `--classical amp,omp,sd`, `--admm-lambdas`, `--n-test`, `--device`, `--out`, `--tags`; v31:
    `--classical l1,grwl1 --classical-params results/classical_params_<SCENARIO>.json` (from `tune_classical.py`, GUIDE 4.14).
  - v31: a network whose NMSE is not finite (diverged) is drawn as "diverged" in the figures instead of stopping the run.
  - `--noise-model auto|colored|white` (the test noise, 3.3b) and `--nf-step trained|matched` (NF-LADMM's data step).
  - `--combiner <file.mat>|random:<seed>[:<M>]` (v29; `:<M>` v30): test with another combiner. A runs folder of another
    scenario is tested on this scenario's combiner the same way (DeepMIMO's combiner is not SV's). Networks that can
    follow (analytic LADMM, Graph-LADMM analytic; v31 also NF-LADMM, LVAMP and ALISTA, whose learned parts do not depend
    on A) use the new combiner, with any number of RF chains M (v30); the others keep their own, and are skipped when the
    test combiner has another M (they cannot run). `eval_log.json` lists which.
    `--graph-phase target`: analytic Graph-LADMM takes this data set's edge phases (from its validation channels);
    Graph-LADMM with a free H_t keeps its own, since H_t was learned with them.
  - `--polish band|snr` (v28, default off): every network also followed by a least-squares correction on its K strongest
    beams, as an extra column `<run>+polish` (`+polish-snr`). K ∈ {0, 4, …, 32} and the ridge ε are chosen on the
    validation channels with their own noise draws, never on the test set: per network band (5 SNRs over the band), or
    per test SNR (the true SNR is known; label it so). K = 0 is a candidate, so the polish never loses on validation.
    The choices go to `polish_choices.csv` and `eval_log.json`. `compare.py --polish band` passes it on.
- **Many test combiners at once (v30):** `check_combiners.py` runs `evaluate.py` on nine fixed test combiners and
  tabulates every network × combiner (GUIDE 4.13):
  ```
  %run scripts/check_combiners.py --scenario configs/scenarios/sv_ula.yaml --runs outputs/SV_ULA --tags "G-LADMM*" GM-LAMP-authors
  ```
  - The combiners: `own`, `pm1` (random ±1), `gauss`, `corr` (correlated rows), `sparse` (partially connected), `dct`
    (DCT row subset), and random ±1 with 1.25 / 0.75 / 0.5 × M (`M+`, `M-`, `M--`).
  - Options: `--combiners` (a subset), `--combiner-files`, `--graph-phase`, `--classical`, `--n-test`, `--out`.
  - Output: `results/combiner_check_<SCENARIO>_<runs>/combiner_check.md` / `.csv` / `.png`, `combiners.csv`, and one
    evaluation folder per combiner.
  - v31: a combiner on which `evaluate.py` fails, or where no method can run, is noted and the check goes on.
- Output in the run folder:
  - `nmse_vs_snr.csv`: methods × SNR.
  - `nmse_vs_snr.png`: curves with values.
  - `nmse_table.png`: every value printed.
  - `eval_log.json`: time per method.

## 11. Figures and tables
| what | command | output |
|---|---|---|
| NMSE vs SNR of one scenario | `evaluate.py` | `nmse_vs_snr.png`, `nmse_table.png` |
| **paper figure: one panel per scenario + tables** | `%run scripts/make_figures.py --runs outputs/SV_ULA outputs/SV_UPA outputs/DeepMIMO_ULA outputs/DeepMIMO_UPA --ref GM-LAMP-authors` | `outputs/figures/nmse_panels.png` / `.pdf`, `table.csv` / `.md` / `.tex` (0/10/20 dB, mean over SNRs, gain over the reference, ± std over seeds) |
| choose / rename methods | `--methods "GM-LAMP-authors=GM-LAMP" "LADMM-best=LADMM (ours)" "NF-LADMM-S*"` | |
| mean ± std over seeds | `aggregate_seeds.py --csv <run folder>/nmse_vs_snr.csv` | `nmse_vs_snr_seeds.csv` / `.png` |
| number of layers | `select_depth.py` / `depth_curve.py` / `depth_snapshots.py` (test set, `--save-layers`) | `depth_select.png`, `depth_curve.png`, `depth_snapshots/depth_curve.png` |
| variants | `compare_variants.py` | `compare.png` |
| data checks | `make_dataset.py` | `dataset_check.png` |

Every figure prints its numbers on the curves.

## 12. MATLAB / Octave
```matlab
addpath('<bsce>\matlab')
net  = bsce_load('outputs\SV_UPA\LADMM-full_gmlamp\LADMM-full_gmlamp_0to10dB.mat');
xhat = bsce_infer(net, y);              % y: 128 x K complex; same output as PyTorch (~1e-15)
```
- **Supported:** LAMP, GM-LAMP, LADMM (all operators), LISTA, ALISTA, LVAMP, ADMM-Net.
- **Original MATLAB code:** copy `<tag>_<band>_orig.mat` to the weight file name the original script loads (`LAMP.m`, `GM_LAMP.m`, `LADMM_test.m`).
- **Check the exports:** `%run scripts/check_exports.py --runs outputs/SV_ULA --scenario configs/scenarios/sv_ula.yaml` (needs Octave or `--matlab <path>`).
- **NF-LADMM** runs in Python only. Its `.mat` file holds the parameters, but there is no MATLAB inference for it yet.

## 13. Recipes (complete examples, in order)

**R1. Reproduce the paper setting (GM-LAMP protocol, two bands).**
```
%run scripts/make_gmlamp_data.py --sv --deepmimo-ula <H_ula_1.mat> --deepmimo-upa <H_upa.mat>
%run scripts/run_all.py --scenario configs/scenarios/sv_ula.yaml --protocol gmlamp --skip-data --no-eval --configs lamp_original gmlamp_spread lista alista lvamp admmnet --set model.precision=float32
%run scripts/select_depth.py --scenario configs/scenarios/sv_ula.yaml --T 2 4 6 8 10 12 --train configs/train/ladmm_original.yaml --protocol gmlamp
%run scripts/run_all.py --scenario configs/scenarios/sv_ula.yaml --protocol gmlamp --skip-data --no-eval --configs ladmm_original ladmm_diagprior ladmm_psd --set model.T=<T*> model.precision=float32 model.init_precision=float32
%run scripts/import_gmlamp_weights.py --scenario configs/scenarios/sv_ula.yaml --runs outputs/SV_ULA
%run scripts/evaluate.py --scenario configs/scenarios/sv_ula.yaml --runs outputs/SV_ULA --classical amp,omp,sd
```
Repeat with `sv_upa.yaml`, `deepmimo_ula.yaml` and `deepmimo_upa.yaml`, then build the figure:
```
%run scripts/make_figures.py --runs outputs/SV_ULA outputs/SV_UPA outputs/DeepMIMO_ULA outputs/DeepMIMO_UPA --ref GM-LAMP-authors
```

**R2. Best LADMM: one network 0–20 dB, then per-band fine-tune.**
```
%run scripts/train.py --scenario configs/scenarios/sv_upa.yaml --train configs/train/ladmm_best.yaml --snr 0 20 --tag LADMM-best --device cuda
%run scripts/finetune_bands.py --scenario configs/scenarios/sv_upa.yaml --from outputs/SV_UPA/LADMM-best/LADMM-best_0to20dB.pt --device cuda
%run scripts/evaluate.py --scenario configs/scenarios/sv_upa.yaml --runs outputs/SV_UPA --classical amp
```
`finetune_bands.py` fine-tunes on 0–10 and 10–20 dB (`--snr 0 10 20`, the default; `--snr scenario` = the scenario's bands) and writes `LADMM-best_ft_0-10-20dB_20k` next to the source. The same command resumes an interrupted fine-tune: finished bands are kept and the running band continues from its checkpoint. In the GM-LAMP scenarios the validation files are at 10 dB only. To validate over each whole band, build the data with `make_dataset.py --snr 0 10 20` (its validation files span each band). On your own data, use the `sv_upa_user*.yaml` scenarios (section 3.5).

**R3. NF-LADMM vs LADMM, one network each, with depth selection.**
```
%run scripts/select_depth.py --scenario configs/scenarios/sv_upa.yaml --snr 0 20 --T 2 4 6 8 10 12 --train configs/train/ladmm_best.yaml configs/train/nfladmm_s.yaml --set train.steps=10000 --device cuda
%run scripts/train.py --scenario configs/scenarios/sv_upa.yaml --train configs/train/nfladmm_s.yaml --snr 0 20 --set model.T=<Tn> --tag NF-LADMM-S --device cuda
%run scripts/train.py --scenario configs/scenarios/sv_upa.yaml --train configs/train/nfladmm_c.yaml --snr 0 20 --set model.T=<Tn> --tag NF-LADMM-C --device cuda
%run scripts/evaluate.py --scenario configs/scenarios/sv_upa.yaml --runs outputs/SV_UPA --classical amp
```

**R4. Layer-wise vs end-to-end training of the same LADMM.**
```
%run scripts/train.py --scenario configs/scenarios/sv_upa.yaml --train configs/train/ladmm_original.yaml --protocol ladmm_best --snr 0 20 --tag LADMM-layerwise
%run scripts/train.py --scenario configs/scenarios/sv_upa.yaml --train configs/train/ladmm_original.yaml --protocol e2e --snr 0 20 --set train.steps=100000 --tag LADMM-e2e
%run scripts/evaluate.py --scenario configs/scenarios/sv_upa.yaml --runs outputs/SV_UPA --classical amp
```

**R5. GM-LAMP protocol vs best-validation protocol.**
```
%run scripts/make_scenario.py --base sv_upa --name SV_UPA_valrange --val-points 5
%run scripts/generate_data.py --scenario configs/scenarios/sv_upa_valrange.yaml
%run scripts/run_all.py --scenario configs/scenarios/sv_upa_valrange.yaml --protocol gmlamp --skip-data --configs ladmm_original
%run scripts/run_all.py --scenario configs/scenarios/sv_upa_valrange.yaml --protocol gmlamp_best --skip-data --configs ladmm_original
```

**R6. Group threshold and scalable operators.** Use `compare_variants.py` (section 9.3), and `run_all.py ... --configs ladmm_diag ladmm_lowrank ladmm_kron ladmm_kronprior` on the UPA scenarios.

**R7. Seeds for the main table.** Run the R1 commands again with `--seed 1` and with `--seed 2`, evaluate, then run `make_figures.py`, which shows mean ± std.

**R8. Larger data (e.g. 200k SV, 54k DeepMIMO).** Section 3.2, then the usual commands with `--scenario configs/scenarios/<name>.yaml`.

**R9. Your own trained models.** Use `import_user_ladmm.py` (untitled2.py checkpoints), then `evaluate.py`.

## 14. Verification tests
```
!python -m pytest tests -q
%run scripts/summarize_tests.py
```
- **Time and requirements:** about 1 h on 2 CPU cores. The comparisons with the original code need `requirements-verify.txt` (TensorFlow) and GNU Octave (`BSCE_OCTAVE`); without them those tests are skipped.
- **What is checked:**
  - data vs `generate_channel.m`
  - LAMP/GM-LAMP vs MATLAB and the TF graphs
  - LADMM vs your scripts
  - the training loops replayed with the same minibatches
  - the bsce LADMM protocol vs `untitled2.py` (identical stages and weights)
  - NF-LADMM vs the proposal module
  - GPU vs CPU
  - exports
- **Results:** `tests/results/summary.md` and `VERIFICATION_REPORT.md`.

## 15. Tracking and review
- `%run scripts/status.py` prints the checklist of `RUNBOOK.md` and writes `STATUS.md`; `--data` also checks the data fingerprints.
- `%run scripts/review_pack.py --step S3_SV_UPA --paths outputs/SV_UPA` builds a small zip of results (no models) for review.

## 16. Troubleshooting
| message | fix |
|---|---|
| `missing ... reference/gmlamp/...` | extract `bsce_reference_part1..4.zip` where `bsce_code.zip` was extracted |
| `missing ...measurements_<band>.mat` | nothing to do: `train.py` makes it; or run `make_sv_measurements.py` ahead of time |
| `NF-LADMM ... needs the noise variance` | use `--noise online` (default of `nfladmm_*.yaml`) or files made by bsce (they store `snr_db`) |
| `only N usable users ...` (DeepMIMO) | fewer channels, or more MATLAB rows (`deepmimo_make_H_chunks.m`) |
| `GM-LAMP test channels found: 1991 / 2000` | normal (1990 for ULA): the rest are nearest users |
| `DIFFERENT` in `make_gmlamp_data.py` | not the review's data: check the scenario files / the raw DeepMIMO file |
| `CUDA ... not available` | PyTorch CPU build installed: reinstall with `--index-url` (section 1.1) |
| a run restarts from scratch | some setting, data or band changed; the same command resumes |
| out of GPU memory | `--set train.val_chunk=5000` (validation in chunks) or a smaller `train.batch_size` |

## 17. Reference numbers (measured, SV UPA test set, NMSE dB at 0 / 10 / 20 dB)
| network | 0 dB | 10 dB | 20 dB | note |
|---|---|---|---|---|
| GM-LAMP authors (two nets) | −8.45 | −15.11 | −20.32 | |
| LAMP authors (two nets) | −7.61 | −14.64 | −20.03 | |
| your LADMM, one 0–20 dB net (`untitled2.py`, `ladmm_best`) | −9.52 | −17.59 | −20.17 | 1.72 M iterations |
| the same, fine-tuned per band (`finetune_bands.py`) | −9.80 | −17.40 | −22.61 | 2.25 dB mean gain over GM-LAMP |
| NF-LADMM-S, 2,000 CPU steps only | −6.32 | −12.18 | −15.55 | not converged; use the full 30k GPU steps |

Use these as checks: `import_user_ladmm.py` + `evaluate.py` on SV UPA must reproduce the third row exactly.

---

## Appendix A1: training options (`--set train.<name>=<value>`)
| name | default | meaning |
|---|---|---|
| `schedule` | borgerding | `borgerding` (layer-wise + fine-tune), `steps` (end-to-end), `end2end` (epochs), `alista` |
| `lr` | 1e-3 | learning rate (first sub-stage of a layer; start value for `steps`) |
| `refinements` | [0.5, 0.1, 0.01] | fine-tune learning-rate factors after each layer (`[]` = no fine-tune) |
| `maxit` | 1,000,000 | maximum steps per sub-stage (layer-wise) |
| `better_wait` | 5000 | stop a sub-stage when the best validation is older than this (steps) |
| `ivl` / `check_mult` | 10 / 100 | validation every `ivl` steps; stop check every `ivl × check_mult` |
| `batch_size` | 128 | minibatch (drawn with replacement) |
| `loss` | nmse | `nmse` (pooled), `l2`, `log_nse` (mean per-sample dB: every SNR weighs the same), `mse` (E‖x − x̂‖², UTRR) |
| `val_metric`, `val_round`, `val_round_mode` | nmse, 5, model | validation metric and rounding (GM-LAMP rounds to 5 decimals) |
| `warm_start` | false | start a new layer from the previous one (GM-LAMP) |
| `restore_best` | false | keep the best validated weights at the end of every stage |
| `val_mean_db` / `val_groups` | false / 1 | validation = mean in dB over the band's validation SNRs |
| `optimizer` | adam_tf1 | `adam_tf1` (TF-1 Adam exactly) or `adam_torch` |
| `noise` / `snr_range` | file / – | `online` = fresh noise every minibatch (`snr_range` set from the band) |
| `snr_dist` / `snr_step` | uniform / – | online noise: `uniform` = one SNR per sample i.i.d. U(`snr_range`) (as before); `fair` = every minibatch spread evenly over the range (one SNR in each of B equal parts); with `snr_step` on the grid lo, lo + step, …, hi. `train.py --snr-dist fair` sets them (v26) |
| `steps`, `lr_final_frac`, `grad_clip`, `val_every` | 20000, 0.05, none, 500 | end-to-end schedule `steps` |
| `epochs` | 30 | `end2end` schedule (with `lr_list`: the budget of all its stages together) |
| `lr_list`, `patience` | none, none | `end2end`: one stage per learning rate in `lr_list`; a stage stops after `patience` epochs without improvement (UTRR: 0.005 … 0.0001, 10) |
| `lr_decay`, `decay_rate` | [0.2, 0.02], 0.3 | `alista` schedule |
| `seed` | 0 | initialisation and minibatches (`--seed` sets it) |
| `val_chunk` | none | validation in chunks of this size (memory) |
| `patience_steps` | 5000 | schedule `steps`: stop when the best validation has not improved by `min_gain_db` during this many steps (best state kept); 0 = every step (v24 and older) |
| `min_gain_db` | auto | the gain (dB) that counts as an improvement for the stopping rules: 0.01 for `steps`; 0 (any gain, the original rule) for the layer-wise schedules (`better_wait`) and `end2end` (`patience`) |
| `checkpoint_minutes` | 10 | also checkpoint INSIDE a stage every this many minutes (weights, optimizer, random streams, best state, position), so an interrupted run resumes there exactly; 0 = only after every stage |
| `gm_symmetry` | break | v31.1: GM-LAMP mixture components that are still identical (the original init) get the `gm_init=spread` offset before a stage trains them; in PyTorch they would otherwise get identical gradients, never separate, and the denoiser would stay one Gaussian (no use of the sparsity). `keep` = the original arithmetic (the TF replay tests) |
| `combiner` / `combiner_every` | fixed / 1 | `random` (v29): a new random ±1 combiner every `combiner_every` minibatches; `mix` (v31): size from `combiner_M`, family from `combiner_types` (online noise; networks with `set_combiner`) |
| `combiner_M` / `combiner_types` | none / none | `mix` (v31): the numbers of RF chains (e.g. `[64,96,128,160]`; none = the scenario's M) and the families (`pm1`, `gauss`, `sparse`, `dct`; none = `pm1`) |

## Appendix A2: model options (`--set model.<name>=<value>`)
| model | options |
|---|---|
| all | `T` (layers), `precision` (`float32` / `float64`) |
| LAMP / GM-LAMP | `untied`, `K` (GM components), `gm_init` (`original` / `spread`), `gm_spread` |
| LADMM | `operator` (`full`, `diag`, `lowrank`, `kron`, `psd`, `kronprior`, `diagprior`, v28: `analytic`, `analytic1`), `rank`, `untied`, `init_precision` (`float32`, `float64`, `untitled2`), `prior_init`, `threshold` (`scalar`, `group`, `group_k`, `group_kc`, v28: `adaptive`, `firm`, `gm`), `gm_K` (components of `gm`, default 4), `data_step` (v28: `plain`, `whitened`), `group_kappa`, `group_win`, `N1`, `N2`, `array`. The v28 options save no `_orig.mat` (LADMM_test.m computes Aᵀy and the soft threshold only) |
| CADMM | `rho0` (the ρ of the initial step, default 1) |
| Graph-LADMM | `T`, `zstep` (`l1`, `gm`, `gate`, `gsoft`, `gtv`, `gtv_gate`, `gtv_gm`, `gm_graph`, `gtv_gm_graph`, v31: `rwl1`, `gtv_rwl1`), `data_step` (v31: `whitened`, `plain`), `beam_prior` (v31: `none`, `gate`), `xstep` (`analytic`, `free`), `sigma` (`residual`, `blind`, `known`), `edge_phase` (`data` = from the training channels, `theory` = −π/n, `blind` = 0, or a value in degrees), `learn_phase`, `ring` (wrap edges, default true), `gm_K`, `lr_mult` (the scalars move this many times faster than H_t, default 10), `array`, `N1`, `N2` |
| NF-LADMM | `denoiser` (`conv`, `radial`), `hidden`, `kernel`, `share_cnn`, `thr0`, `sigma` (`known`, `residual`), `array`, `N1`, `N2`, `update` (`dual`, `relax`), `slope_init`, `noise_model` (`colored`, `white`: the exact data step; default = the data's, 3.3b) |
| LISTA / ALISTA | `lam`, `untied`, `complex_mode`; ALISTA `percent`, `max_percent` |
| LVAMP | `complex_mode` |
| ADMM-Net | `beta`, `rho` |
| UTRR | `K` (top-K of the last layer; the paper's ensemble: 0, 16, 64, 256), `rcc` (top-K only in the last layer, default true), `bn` (the batch normalisation of Fig. 6, default false), `rho`, `alpha` (initial values 1, 0.1), `bn_momentum` |

## Appendix A3: scenario file keys
- **`name`, `dataset` (SV/DeepMIMO), `array` (ULA/UPA), `N1`, `N2`, `L`, `M`.**
- **`cs_matrix`:** the reference CS matrix, used when `data_dir` has no `CSmatrix.mat`.
- **`data_dir`:** the data folder.
- **`test_data_dir`** (optional): where the test channels are.
- **`seed`, `n_train`, `n_val`, `n_test`, `n_snr_chunks`.**
- **`bands`:** a list of `{name, train: [lo, hi], val_snr: [...] or val_range: [lo, hi], use_for: [lo, hi], train_name?, val_name?}`.
- **`eval_bands`** (optional).
- **`test_snr_db`.**
- **`noise_model`** (optional): `colored` (default) or `white` (3.3b). Usually given on the command line (`--noise-model white`).

## Appendix A4: every command and its options
`docs/CLI_REFERENCE.md` holds the `--help` of every script. Regenerate it with `%run scripts/make_cli_reference.py`.

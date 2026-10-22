# RUNBOOK: experiments step by step (Spyder)

Every step: run the commands → `%run scripts/status.py` → `%run scripts/review_pack.py ...` →
upload the zip from `review/` together with the console output. A step is ticked when its files
exist (`status.py`); we review it together before the next one.

**Spyder, once per session** (console with the `bsce` interpreter):
```
%cd "D:\Academic Profile\LADMM-MIMO\codes\bsce_code (3)\bsce"
```
All commands below are run from this folder. Long runs: keep Spyder open and the PC awake
(Windows power settings: never sleep). An interrupted training resumes when the same command is
run again (checkpoints). `--set` overrides apply to every config of a `run_all.py` call, so the
LADMM configs (which take `model.T=<T>`) are run in a separate call from the baselines.
`<T>` = LADMM depth chosen in S2, `<Tn>` = NF-LADMM depth chosen in S2; `<P>` = precision chosen in S0b (`float32` or `float64`).

**Any SNR range.** Every training command accepts:
- `--snr <edges>` in dB, which replaces the scenario's bands:
  - `--snr 0 20`: one network for 0–20 dB
  - `--snr 0 10 20`: two networks
  - `--snr -5 0 5 10 15 20 25`: six networks
  - any edges work
- `--noise file|online`:
  - `file` (default for LADMM/LAMP/…): the measurement files. Missing files are made automatically: 100 SNR chunks ~ U(band), and the SNR of every column is stored. With `--snr-dist fair` (v26): one SNR per column, spread evenly over the band.
  - `online` (default for NF-LADMM): fresh noise every minibatch, one SNR per sample ~ U(band), and a fixed generated validation set. It is the same for every method with the same band.
- `evaluate.py` picks up every band file of a run folder by its name (`0to20dB`, `m5to0dB`, …). A one-network run and band-pair runs are compared in the same table.

Examples:
```
%run scripts/train.py --scenario configs/scenarios/sv_upa.yaml --train configs/train/ladmm_best.yaml --snr 0 20 --tag LADMM-best-0to20
%run scripts/train.py --scenario configs/scenarios/sv_upa.yaml --train configs/train/ladmm_best.yaml --snr 0 20 --noise online --tag LADMM-best-0to20-online
%run scripts/train.py --scenario configs/scenarios/sv_upa.yaml --train configs/train/nfladmm_s.yaml --snr -5 25
%run scripts/make_sv_measurements.py --scenario configs/scenarios/sv_ula.yaml --snr 0 5 10 15 20
```

| step | data | purpose | done when |
|---|---|---|---|
| S0a | all 4 | data identical to the reviewed data (`make_gmlamp_data.py`, DATA (A)) | `status.py --data`: 4 × OK |
| S0b | SV_ULA | fastest device / precision on the P2000 | `outputs/speed_test.json` |
| S1 | SV_ULA | retrained LAMP / GM-LAMP ≈ authors' weights (pipeline sanity) | 2 runs |
| S2 | SV_ULA, SV_UPA | depth T of LADMM **and NF-LADMM** chosen on validation data | `outputs/depth_select/<scenario>/depth_select.json` |
| S3 | each of 4 | main comparison, 9 learned + NF-LADMM-S/C + AMP / OMP / SD | 11 runs + `nmse_vs_snr.csv` per scenario |
| S4 | each of 4 | seeds 1, 2 for 5 methods → mean ± std | `nmse_vs_snr_seeds.csv` |
| S5 | SV_UPA, DeepMIMO_UPA | scalable LADMM operators (params vs NMSE) | 4 runs per scenario |
| S6–S10 | — | complexity, robustness (new combiner), cross-dataset, CDF / sum-rate, ablations | scripts provided when we get there |

Recommended order: S0a → S0b → S1 → S2 → S3 (SV_ULA, SV_UPA, DeepMIMO_UPA, DeepMIMO_ULA) → S5 → S4 → S6 …

---

## DATA (A): the GM-LAMP data sets, paper reproduction (one command, see DATA_GUIDE.md)
```
%run scripts/make_gmlamp_data.py --sv --deepmimo-ula "D:/Academic Profile/LADMM-MIMO/codes/H_ula_1.mat" --deepmimo-upa "D:/Academic Profile/LADMM-MIMO/codes/H_upa.mat"
```
- SV: 80,000 / 2,000 / 2,000. DeepMIMO: 50,000 / 2,000 / 2,000, with GM-LAMP's own test file.
- Bands 0–10 / 10–20 dB, validation at 10 dB, GM-LAMP's CS matrices.
- At the end the script compares the data with the data used so far. Expected: `4 x OK`. Checked: a full regeneration is bit-identical.
- Details, the file list and the MATLAB step (only if the raw DeepMIMO files are lost) are in `DATA_GUIDE.md`.

## DATA (B): make a data set of any size, with train/val/test splits and any SNR range (SV and DeepMIMO)
One Python command per data set, `scripts/make_dataset.py`. It writes:
- the channels (training / validation / test)
- one training + validation measurement file per SNR band, with 100 SNR chunks ~ U(band) and the SNR of every column stored
- GM-LAMP's CS matrix
- a check figure
- **a scenario file `configs/scenarios/<name>.yaml` ready for `train.py` / `run_all.py` / `evaluate.py`**

It needs `reference/gmlamp/` (CS matrices, GM-LAMP's DeepMIMO test channels, the authors' weights). That folder comes in `bsce_reference_part1.zip` … `part4.zip`: extract all four where you extract `bsce_code.zip`. Every path starts with `bsce/`, so the files land in `bsce/reference/gmlamp/`.

**Defaults**
- `--n 200000 --split 0.90 0.05 0.05` gives 180,000 / 10,000 training / validation channels.
- `--test standard` (default) uses the standard 2000 test channels: the bsce SV test set, or GM-LAMP's DeepMIMO test file. Results then stay comparable with every earlier table. `--test random` instead gives a 10,000-channel test split from the same pool.
- `--snr` = band edges, as for `train.py`:
  - `0 20`: one band
  - `0 10 20`: two bands
  - `-5 0 5 10 15 20 25`: six bands
- `--no-measurements`: channels only. Train with `--noise online`, or add bands later.
- `--test-measurements`: also writes test measurement files with the evaluation noise (for MATLAB).

### SV (Python only, about 1 min and 1.3 GB for 200k)
```
%run scripts/make_dataset.py --dataset SV --array UPA --n 200000 --snr 0 20 --name SV_UPA_200k
%run scripts/make_dataset.py --dataset SV --array ULA --n 200000 --snr 0 20 --name SV_ULA_200k
```
- Expected console output: `train 180,000 / val 10,000; test = the standard 2000 channels`, then `scenario written: configs/scenarios/sv_upa_200k.yaml`.
- Check `data/SV_UPA_200k/dataset_check.png`:
  - mean energy ≈ 256 in all three sets
  - the three strongest-beam histograms on top of each other
  - SNRs spread over the band

### DeepMIMO, GM-LAMP's area, 54k users (chosen setting; no MATLAB needed)
The raw files you already have, `H_upa.mat` and `H_ula_1.mat`, cover rows 1000–1300 (54,481 users):
- 2,000 of those users form GM-LAMP's test set; they are found and removed.
- 52,481 users remain for training and validation.
```
%run scripts/make_dataset.py --dataset DeepMIMO --array UPA --input "D:/Academic Profile/LADMM-MIMO/codes/H_upa.mat" --n-train 50481 --n-val 2000 --snr 0 20 --name DeepMIMO_UPA_54k
%run scripts/make_dataset.py --dataset DeepMIMO --array ULA --input "D:/Academic Profile/LADMM-MIMO/codes/H_ula_1.mat" --n-train 50481 --n-val 2000 --snr 0 20 --name DeepMIMO_ULA_54k
```
- Result: 50,481 training / 2,000 validation / 2,000 test (GM-LAMP's file).
- Checked: GM-LAMP test users found 1991/2000 (UPA) and 1990/2000 (ULA); train-test strongest-beam TV 0.011 (UPA) and 0.025 (ULA).
- For GM-LAMP's exact sizes, use `--n-train 50000` instead.
- For more training examples, add `--noise online` to `train.py`: fresh noise every minibatch.

### DeepMIMO larger than 54k (optional: MATLAB, then Python)
**Step 1, MATLAB.** The raw channels, in files of 300 rows each (about 54k users per file):
- O1_28, BS 3, 3 paths, one subcarrier, GM-LAMP's arrays.
- 200,000 users need rows 1000–2105 (181 users per row). The function computes this itself.
- Files already present are kept, so a stopped run can simply be restarted.
```
addpath('D:\Academic Profile\LADMM-MIMO\codes\bsce_code (6)\bsce\matlab')
dm = 'D:\Academic Profile\LADMM-MIMO\codes\DeepMIMO-matlab-1.3';
deepmimo_make_H_chunks(dm, 'D:\Academic Profile\LADMM-MIMO\codes\deepmimo_raw', 'UPA', 200000);
deepmimo_make_H_chunks(dm, 'D:\Academic Profile\LADMM-MIMO\codes\deepmimo_raw', 'ULA', 200000);
```
**Step 2, Python (Spyder, bsce folder).**
```
%run scripts/make_dataset.py --dataset DeepMIMO --array UPA --input "D:/Academic Profile/LADMM-MIMO/codes/deepmimo_raw/H_upa_part*.mat" --n 200000 --snr 0 20 --name DeepMIMO_UPA_200k
%run scripts/make_dataset.py --dataset DeepMIMO --array ULA --input "D:/Academic Profile/LADMM-MIMO/codes/deepmimo_raw/H_ula_part*.mat" --n 200000 --snr 0 20 --name DeepMIMO_ULA_200k
```
- Expected: `GM-LAMP test channels found among the users: 1991 / 2000`. The same 1991 was found for the current data set; the other 9 are nearest users. Those users are removed from train/val, and the test set is GM-LAMP's file.
- Every channel has ‖x‖² = 256 (GM-LAMP's convention: x = U0ᴴh).
- **Caveat.** Rows 1301–2105 lie outside GM-LAMP's user area, so 200k unique users means a wider area than GM-LAMP's 54k users.
  - `dataset_check.png` prints the train-test difference (TV of the strongest-beam histograms).
  - To stay inside GM-LAMP's area, use `deepmimo_make_H(dm, file, 'UPA')` (rows 1000–1300) and `--n 52000`. Then get many more training examples with `--noise online` in `train.py`: fresh noise every minibatch.

### Using a data set
```
%run scripts/train.py --scenario configs/scenarios/sv_upa_200k.yaml --train configs/train/ladmm_best.yaml --device cuda
%run scripts/train.py --scenario configs/scenarios/sv_upa_200k.yaml --train configs/train/nfladmm_s.yaml --device cuda
%run scripts/import_gmlamp_weights.py --scenario configs/scenarios/sv_upa_200k.yaml --runs outputs/SV_UPA_200k
%run scripts/evaluate.py --scenario configs/scenarios/sv_upa_200k.yaml --runs outputs/SV_UPA_200k --classical amp
```
- Another SNR range later: `train.py ... --snr 0 10 20`. Missing measurement files are made automatically.
- More bands ahead of time: `make_sv_measurements.py --scenario configs/scenarios/sv_upa_200k.yaml --snr -5 0 5 10 15 20 25`.
- Memory: 180k training channels take about 0.8 GB in RAM while loading and 0.4 GB on the GPU (complex64).

The older per-scenario commands still work: `generate_data.py` for SV, `deepmimo_convert.py` + `generate_data.py --measurements-only` for DeepMIMO. They make the 80k/50k data sets of the paper reproduction (S0a).

## S0a: data check (1 min)
```
%run scripts/status.py --data
```
Expected: `SV_ULA OK`, `SV_UPA OK`, `DeepMIMO_ULA OK`, `DeepMIMO_UPA OK`. Upload: console text only.

## S0b: speed test (10–20 min)
```
%run scripts/speed_test.py
%run scripts/review_pack.py --step S0b --paths outputs/speed_test.json outputs/speed_test.csv outputs/speed_test.png
```
Trains 2-layer LAMP, GM-LAMP, LADMM, LISTA for the same number of iterations on cuda/cpu ×
float64/float32. Output: the fastest setting per method → we fix `<P>` and `--device`.
(The P2000 is slow in float64, so float32 on GPU is expected to win.)

## S1: sanity, LAMP and GM-LAMP retrained (SV_ULA)
```
%run scripts/run_all.py --scenario configs/scenarios/sv_ula.yaml --protocol gmlamp --skip-data --configs lamp_original gmlamp_spread --set model.precision=<P>
%run scripts/review_pack.py --step S1 --paths outputs/SV_ULA
```
Output: `outputs/SV_ULA/LAMP_gmlamp/`, `GM-LAMP-spread_gmlamp/` (models .pt/.mat, train_log.json),
`nmse_vs_snr.csv/.png`, `nmse_table.png`, `logs/`.
Check: NMSE close to the authors' trained weights on the same test set
(GM-LAMP −9.17 / −16.78 / −22.96 dB, LAMP −8.58 / −16.31 / −21.90 dB at 0 / 10 / 20 dB), within a few tenths of a dB.

## S2: depth of LADMM and NF-LADMM (SV_ULA, SV_UPA)
**Recommended: one command per scenario for both methods** (`select_depth.py`).
- **Layer-wise methods (LADMM).** One run with T = max(T). The validation NMSE of every depth is logged when that depth is finished.
- **End-to-end methods (NF-LADMM).** One run per T.
- The output is one plot per band with the curves of both methods (values printed) and T* = the smallest T within 0.1 dB of the best.
- For the selection, NF-LADMM uses 10,000 steps per T. The final run in S3 uses 30,000.
```
%run scripts/select_depth.py --scenario configs/scenarios/sv_ula.yaml --snr 0 20 --T 2 4 6 8 10 12 --train configs/train/ladmm_best.yaml configs/train/nfladmm_s.yaml --set train.steps=10000 model.precision=<P> --device cuda
%run scripts/select_depth.py --scenario configs/scenarios/sv_upa.yaml --snr 0 20 --T 2 4 6 8 10 12 --train configs/train/ladmm_best.yaml configs/train/nfladmm_s.yaml --set train.steps=10000 model.precision=<P> --device cuda
%run scripts/review_pack.py --step S2 --paths outputs/depth_select
```
- `train.steps` only affects NF-LADMM; LADMM keeps its own protocol.
- With the GM-LAMP protocol and bands (as in the paper reproduction), drop `--snr`:
  - LADMM: `--train configs/train/ladmm_original.yaml --protocol gmlamp`
  - NF-LADMM: a separate call without `--protocol`, since the protocol would replace its end-to-end schedule.
- Output: `outputs/depth_select/<scenario>/depth_select.{png,csv,json}` (T* per method and band); runs in `outputs/<scenario>/<tag>-T<t>/`.

Older single-method version for LADMM (still valid):
```
%run scripts/train.py --scenario configs/scenarios/sv_ula.yaml --train configs/train/ladmm_original.yaml --protocol gmlamp --tag LADMM-full-T12_gmlamp --set model.T=12 model.precision=<P> model.init_precision=<P>
%run scripts/train.py --scenario configs/scenarios/sv_upa.yaml --train configs/train/ladmm_original.yaml --protocol gmlamp --tag LADMM-full-T12_gmlamp --set model.T=12 model.precision=<P> model.init_precision=<P>
%run scripts/depth_curve.py --runs outputs/SV_ULA/LADMM-full-T12_gmlamp outputs/SV_UPA/LADMM-full-T12_gmlamp --tol 0.1
%run scripts/review_pack.py --step S2 --paths outputs/depth_curve outputs/SV_ULA/LADMM-full-T12_gmlamp outputs/SV_UPA/LADMM-full-T12_gmlamp
```
A 12-layer layer-wise run logs the validation NMSE of every depth when that depth is finished
(the early layers of the final model are retuned later, so its prefixes are not usable networks);
`depth_curve.py` plots these logged values (printed on the curve) and proposes `<T>` = smallest
depth within 0.1 dB of the best in every band; the T-layer network is then trained in S3. We decide `<T>` together. (A running sweep without `--protocol gmlamp` can be sent as a
preliminary result: `outputs/SV_ULA/LADMM-full/train_log.json`.)

## S3: main comparison (one block per scenario)
SV_ULA (LAMP / GM-LAMP from S1 are reused automatically: same folders):
```
%run scripts/run_all.py --scenario configs/scenarios/sv_ula.yaml --protocol gmlamp --skip-data --no-eval --configs lista alista lvamp admmnet --set model.precision=<P>
%run scripts/run_all.py --scenario configs/scenarios/sv_ula.yaml --protocol gmlamp --skip-data --configs ladmm_original ladmm_diagprior ladmm_psd --set model.T=<T> model.precision=<P> model.init_precision=<P>
%run scripts/train.py --scenario configs/scenarios/sv_ula.yaml --train configs/train/nfladmm_s.yaml --snr 0 20 --set model.T=<Tn> model.precision=<P> --tag NF-LADMM-S
%run scripts/train.py --scenario configs/scenarios/sv_ula.yaml --train configs/train/nfladmm_c.yaml --snr 0 20 --set model.T=<Tn> model.precision=<P> --tag NF-LADMM-C
%run scripts/evaluate.py --scenario configs/scenarios/sv_ula.yaml --runs outputs/SV_ULA --classical amp,omp,sd
%run scripts/review_pack.py --step S3_SV_ULA --paths outputs/SV_ULA
```
About NF-LADMM:
- It is **one network for 0–20 dB** by design: the noise level enters each layer.
- `evaluate.py` gives it the test noise variance. With `--set model.sigma=residual` it estimates the noise level from the residual alone; that ablation is for step S8.
- About 9k parameters (UPA) / 3.4k (ULA) at T = 8.
- For a band-by-band comparison with LADMM, `--snr 0 10 20` also works.

SV_UPA, DeepMIMO_UPA, DeepMIMO_ULA (same three lines with the scenario file changed; here LAMP and
GM-LAMP are part of the first line):
```
%run scripts/run_all.py --scenario configs/scenarios/sv_upa.yaml --protocol gmlamp --skip-data --no-eval --configs lamp_original gmlamp_spread lista alista lvamp admmnet --set model.precision=<P>
%run scripts/run_all.py --scenario configs/scenarios/sv_upa.yaml --protocol gmlamp --skip-data --configs ladmm_original ladmm_diagprior ladmm_psd --set model.T=<T> model.precision=<P> model.init_precision=<P>
%run scripts/train.py --scenario configs/scenarios/sv_upa.yaml --train configs/train/nfladmm_s.yaml --snr 0 20 --set model.T=<Tn> model.precision=<P> --tag NF-LADMM-S
%run scripts/train.py --scenario configs/scenarios/sv_upa.yaml --train configs/train/nfladmm_c.yaml --snr 0 20 --set model.T=<Tn> model.precision=<P> --tag NF-LADMM-C
%run scripts/evaluate.py --scenario configs/scenarios/sv_upa.yaml --runs outputs/SV_UPA --classical amp,omp,sd
%run scripts/review_pack.py --step S3_SV_UPA --paths outputs/SV_UPA
```
(`deepmimo_upa.yaml` → `outputs/DeepMIMO_UPA`, `deepmimo_ula.yaml` → `outputs/DeepMIMO_ULA`.)
Output per scenario: one folder per method, `nmse_vs_snr.csv/.png` (all methods, values printed),
`nmse_table.png`, `eval_log.json`, `logs/train_<config>.log`.
Checks: DeepMIMO LAMP / GM-LAMP close to the authors' weights (ULA 20 dB −28.9 / −29.4, UPA −27.0 / −28.8 dB);
AMP ≈ −15.0 (SV_ULA), −10.2 (SV_UPA), −17.1 (DeepMIMO_ULA), −14.3 (DeepMIMO_UPA) dB at 20 dB.

## S4: seeds (5 methods × 2 extra seeds, per scenario)
```
%run scripts/run_all.py --scenario configs/scenarios/sv_ula.yaml --protocol gmlamp --skip-data --no-eval --seed 1 --configs lamp_original gmlamp_spread lista --set model.precision=<P>
%run scripts/run_all.py --scenario configs/scenarios/sv_ula.yaml --protocol gmlamp --skip-data --no-eval --seed 1 --configs ladmm_original ladmm_diagprior --set model.T=<T> model.precision=<P> model.init_precision=<P>
```
repeat with `--seed 2` (last call without `--no-eval`), then
```
%run scripts/aggregate_seeds.py --csv outputs/SV_ULA/nmse_vs_snr.csv
%run scripts/review_pack.py --step S4_SV_ULA --paths outputs/SV_ULA
```
Output: folders `<tag>_s1`, `<tag>_s2`, `nmse_vs_snr_seeds.csv/.png` (mean ± std per method).

## S5: scalable LADMM operators (SV_UPA, DeepMIMO_UPA)
```
%run scripts/run_all.py --scenario configs/scenarios/sv_upa.yaml --protocol gmlamp --skip-data --configs ladmm_diag ladmm_lowrank ladmm_kron ladmm_kronprior --set model.T=<T> model.precision=<P> model.init_precision=<P>
%run scripts/review_pack.py --step S5_SV_UPA --paths outputs/SV_UPA
```
and the same with `deepmimo_upa.yaml`. Output: 4 more folders + updated `nmse_vs_snr.csv`
(the number of parameters of each run is in its `train_log.json`).

## S6–S10
Complexity (params / FLOPs / latency), robustness to a new combiner and SNR mismatch, cross-dataset
(SV ↔ DeepMIMO), per-channel NMSE CDF and sum-rate, ablations: the scripts are added to bsce when
we reach these steps (they use the trained models of S3).

---

## Phase B (after the reproduction): improved training and larger data sets

Two options beyond the original code (default off, so S0–S5 stay a faithful reproduction):
* `--protocol gmlamp_best` = GM-LAMP's protocol + **best validated weights restored** at the end of
  every stage (original: last weights) + **validation over the whole band**: mean in dB of the
  NMSE at every validation SNR of the band (needs a scenario with several `val_snr`).
* Any data size: `scripts/make_scenario.py` writes a new scenario file with its own `data/<name>`
  and `outputs/<name>` folders (the reproduction data are not touched).

B1, validation over the band, same sizes (SV_ULA as example):
```
%run scripts/make_scenario.py --base sv_ula --name SV_ULA_valrange --val-points 5
%run scripts/generate_data.py --scenario configs/scenarios/sv_ula_valrange.yaml
%run scripts/run_all.py --scenario configs/scenarios/sv_ula_valrange.yaml --protocol gmlamp_best --skip-data --configs lamp_original gmlamp_spread --set model.precision=<P>
%run scripts/run_all.py --scenario configs/scenarios/sv_ula_valrange.yaml --protocol gmlamp_best --skip-data --configs ladmm_original ladmm_diagprior --set model.T=<T> model.precision=<P> model.init_precision=<P>
%run scripts/review_pack.py --step B1_SV_ULA --paths outputs/SV_ULA_valrange
```
Validation SNRs: 0, 2.5, 5, 7.5, 10 dB (0–10 dB band) and 10, 12.5, 15, 17.5, 20 dB (10–20 dB band).
Random validation SNRs instead (uniform in the band, seeded): `--val-random 20` in place of `--val-points 5`.

Quick check (15k training channels, 2000 iterations per stage, 6 layers, minutes on the GPU):
```
%run scripts/make_scenario.py --base sv_ula --name SV_ULA_fast --n-train 15000 --n-val 2000 --n-test 2000 --val-random 20
%run scripts/generate_data.py --scenario configs/scenarios/sv_ula_fast.yaml
%run scripts/train.py --scenario configs/scenarios/sv_ula_fast.yaml --train configs/train/ladmm_original.yaml --protocol gmlamp_best --set model.T=6 train.maxit=2000 model.precision=float32 model.init_precision=float32
%run scripts/evaluate.py --scenario configs/scenarios/sv_ula_fast.yaml --runs outputs/SV_ULA_fast --classical amp
%run scripts/review_pack.py --step quick_SV_ULA --paths outputs/SV_ULA_fast
```


B2, larger sets (150,000 training, 10,000 validation, 10,000 test; any numbers work):
```
%run scripts/make_scenario.py --base sv_ula --name SV_ULA_150k --n-train 150000 --n-val 10000 --n-test 10000 --val-points 5
%run scripts/generate_data.py --scenario configs/scenarios/sv_ula_150k.yaml
```
then the `run_all.py` lines of B1 with `sv_ula_150k.yaml`. Rules: n_train divisible by 10 (SNR
chunks), n_val divisible by `--val-points`. SV: the first channels are identical to the 80k/2k/2k
data (same seeds), the sets are only longer. Disk: about 1.4 GB per 150k scenario.
DeepMIMO: O1_28 rows 1000–1300 give 52,481 users outside GM-LAMP's test set, so at most
~50k/2k with GM-LAMP's test set. More users = more rows in MATLAB
(`deepmimo_make_H(dm, 'H_ula_big.mat', 'ULA', [700 1300])`) and `deepmimo_convert.py ... --test random`
(a different user region than GM-LAMP's: state it in the paper).

---

## Phase C: group threshold (ADMM-Improved idea inside LADMM)
`model.threshold`: `scalar` (default, original), `group` (kappa 3, no extra parameters), `group_k`
(learned kappa per layer), `group_kc` (learned kappa and window). Configs `ladmm_group.yaml`,
`ladmm_group_k.yaml`. Side-by-side comparison of variants on several scenarios:
```
%run scripts/compare_variants.py --scenarios sv_upa deepmimo_upa --variants scalar:model.threshold=scalar group:model.threshold=group --set model.T=<T> model.precision=<P> model.init_precision=<P> --protocol gmlamp_best --seeds 0 1 2 --out outputs_threshold
```
Quick 2-layer / 1000-channel test (13 Oct): group better on SV-ULA (+1.38 dB mean), SV-UPA (+0.28),
DeepMIMO-UPA (+0.13), worse on DeepMIMO-ULA (−0.41); one seed, not conclusive.

## Phase D: best LADMM so far (your untitled2.py protocol, your data in data/SV)
Background: `LADMM_why_0to20_SV.md`. Your `untitled2.py` produced `..._1_11dB_better.pt`.

**What it achieves.**
- One 6-layer network for 0–20 dB: −9.52 / −17.59 / −20.17 dB at 0 / 10 / 20 dB on the SV UPA test set, 1.36 dB better than GM-LAMP on average.
- Fine-tuned per band: −9.80 / −17.40 / −22.61 dB, **2.25 dB** better than GM-LAMP on average.

**bsce reproduces untitled2.py exactly.** `tests/test_ladmm_best_vs_untitled2.py` gives the same stages, the same stopping iterations and the same weights (difference 0).

The pieces:
- `configs/train/ladmm_best.yaml` holds the untitled2.py protocol:
  - lr 5e-4, refinements .5/.2/.1/.01, batch 400
  - maxit 100000, better_wait 5000
  - torch Adam, float32, no warm start, last weights kept
- `configs/scenarios/sv_upa_user.yaml` / `sv_ula_user.yaml`:
  - read `data/SV` (`*_0to20dB.mat` for training, `*_1_11dB.mat` for validation)
  - test on the usual 2000 test channels of `data/SV_UPA` / `data/SV_ULA`
- `*_user_2band.yaml`: the same channels, 0–10 and 10–20 dB bands.

| step | what | done when |
|---|---|---|
| D0 | data: `data/SV` holds your files; ULA measurements made | `SV_ULA_*_0to20dB.mat`, `*_1_11dB.mat` exist |
| D1 | your `better.pt` imported + GM-LAMP/LAMP authors + evaluation (no training, 2 min) | `outputs/SV_UPA_user/nmse_vs_snr.csv`: −9.52 / −17.59 / −20.17 |
| D2 | per-band fine-tune of your `better.pt` (GPU, about 10 min) | `outputs/SV_UPA_user_2band/nmse_vs_snr.csv` ≈ −9.8 / −17.4 / −22.6 |
| D3 | bsce re-training with the protocol, SV UPA 0–20 (GPU, hours: the original took 1.72 M iterations) | `outputs/SV_UPA_user/LADMM-best/` |
| D4 | the same for SV ULA (D3 + D2 with `ula`) | `outputs/SV_ULA_user*/` |

**D0** makes the measurement files that are missing. Your UPA files are kept; only ULA (and the 2-band files) are made:
```
%run scripts/make_sv_measurements.py --scenario configs/scenarios/sv_ula_user.yaml
%run scripts/make_sv_measurements.py --scenario configs/scenarios/sv_upa_user_2band.yaml
%run scripts/make_sv_measurements.py --scenario configs/scenarios/sv_ula_user_2band.yaml
```

**D1** imports your network, imports the GM-LAMP authors' nets and evaluates. Put `better.pt` in `models\`:
```
%run scripts/import_user_ladmm.py --scenario configs/scenarios/sv_upa_user.yaml --runs outputs/SV_UPA_user --pt models/Trained_LADMM_for_SV_UPA_2561280to20dB_1_11dB_better.pt --tag LADMM-untitled2-better
%run scripts/import_gmlamp_weights.py --scenario configs/scenarios/sv_upa_user.yaml --runs outputs/SV_UPA_user
%run scripts/evaluate.py --scenario configs/scenarios/sv_upa_user.yaml --runs outputs/SV_UPA_user --classical amp
```

**D2** specialises the network to each band. `--iters 20000` is worth trying on the GPU:
```
%run scripts/finetune_bands.py --scenario configs/scenarios/sv_upa_user_2band.yaml --from models/Trained_LADMM_for_SV_UPA_2561280to20dB_1_11dB_better.pt --tag LADMM-untitled2-better-ft --device cuda
%run scripts/import_gmlamp_weights.py --scenario configs/scenarios/sv_upa_user_2band.yaml --runs outputs/SV_UPA_user_2band
%run scripts/evaluate.py --scenario configs/scenarios/sv_upa_user_2band.yaml --runs outputs/SV_UPA_user_2band --classical amp
```

**D3b**: NF-LADMM on the same data (`data/SV`, 250k channels, online noise), in the same comparison table:
```
%run scripts/train.py --scenario configs/scenarios/sv_upa_user.yaml --train configs/train/nfladmm_s.yaml --set model.T=<Tn> --tag NF-LADMM-S --device cuda
%run scripts/evaluate.py --scenario configs/scenarios/sv_upa_user.yaml --runs outputs/SV_UPA_user --classical amp
```

**D3** trains one 0–20 dB network with the untitled2.py protocol. It resumes after an interruption: run the same line again. Then fine-tune the result per band:
```
%run scripts/train.py --scenario configs/scenarios/sv_upa_user.yaml --train configs/train/ladmm_best.yaml --tag LADMM-best --device cuda
%run scripts/evaluate.py --scenario configs/scenarios/sv_upa_user.yaml --runs outputs/SV_UPA_user --classical amp
%run scripts/finetune_bands.py --scenario configs/scenarios/sv_upa_user_2band.yaml --from outputs/SV_UPA_user/LADMM-best/LADMM-best_0to20dB.pt --device cuda
%run scripts/evaluate.py --scenario configs/scenarios/sv_upa_user_2band.yaml --runs outputs/SV_UPA_user_2band --classical amp
```

Optional: per-band networks trained from scratch with the same protocol, for comparison with fine-tuning:
```
%run scripts/train.py --scenario configs/scenarios/sv_upa_user_2band.yaml --train configs/train/ladmm_best.yaml --tag LADMM-best-2band --device cuda
```

The protocol can also be applied to any other LADMM config (an operator, group threshold, T):
- `--protocol ladmm_best`
- `--set model.T=8`

To upload for review:
- `outputs/SV_UPA_user/nmse_vs_snr.csv` + `.png`
- `outputs/SV_UPA_user_2band/nmse_vs_snr.csv` + `.png`
- `outputs/SV_UPA_user*/*/train_log.json` or `finetune_log.json`

## Tracking
```
%run scripts/status.py
```
prints `[x]` / `[ ]` per step with the missing files and writes `STATUS.md` (included in every review zip).

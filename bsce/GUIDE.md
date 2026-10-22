# bsce: the final workflow, from data to the paper comparison

*15 Oct 2026, final version. Every command below is run in **Spyder's IPython console** from the `bsce` folder. The same commands work in a terminal, with `python` instead of `%run`.*

This guide is the one path to the paper results:

| step | what | main command | time |
|---|---|---|---|
| 0 | set up the folder and check the environment | `check_env.py` | 5 min |
| 1 | make the data (SV and DeepMIMO, as GM-LAMP) | `make_gmlamp_data.py` | 2 min |
| 2 | check that we reproduce the paper (LAMP, GM-LAMP) | `compare.py`, `run_suite.py --protocol paper` | minutes / hours |
| 3 | choose the number of layers of every method | `run_suite.py --save-layers`, `depth_snapshots.py`, `promote_layer.py` | long (one deep run per method) |
| 4 | train every method with the same protocol (+ fine-tune per band) | `run_suite.py` | long |
| 5 | compare all methods fairly, save figures and tables | `compare.py` | ~10 min per scenario |

**Prefer forms to commands?** Double-click `start_app.bat` (or `python app\bsce_app.py` in the Anaconda Prompt). The
workbench has every step below as a form with all its options. It shows the exact commands while you choose, runs them
in a queue and follows their progress (layer, step, validation NMSE, time left). See `docs/WORKBENCH.md`.

Every step writes into two folders, and nothing else:
- `outputs/<SCENARIO>/<network>/`: the trained networks. Each folder has a `README.txt` that says what the network is.
- `results/<name>/`: evaluations, figures and tables. Each folder has a `README.md`.

---

## 0. Set up

### 0.1 The folder
Extract `bsce_code.zip` into a new folder, for example `D:\Academic Profile\LADMM-MIMO\codes\bsce_final`.

Then put two folders inside `bsce\`:

| folder | where it comes from | needed for |
|---|---|---|
| `bsce\reference\` | copy it from your previous bsce folder, or extract `bsce_reference_part1..4.zip` here | CS matrices, the GM-LAMP test sets and the authors' trained networks |
| `bsce\data\` | step 1 makes it. You can also copy it from your previous bsce folder. | all training and testing |

### 0.2 Spyder
- Open Spyder from the `bsce` environment (Anaconda Navigator → environment `bsce` → Spyder).
- Set the working folder once per session:
```
%cd "D:\Academic Profile\LADMM-MIMO\codes\bsce_final\bsce"
%run scripts/check_env.py
```
- `check_env.py` prints the Python, PyTorch, CUDA and GPU versions, and checks `reference\` and `data\`. Every line should show `[OK]`, and the last line `ready.`.

### 0.3 Rules that hold for every command
- `--device cuda` uses the GPU, `--device cpu` the CPU, and `--device auto` (the default) the GPU if there is one.
- **Interrupted?** Run the same command again. Finished steps are skipped, and a training run continues from its last checkpoint.
- `--dry-run` (in `run_suite.py`) prints the commands it would run, and runs nothing. Use it to check a plan.
- `--smoke` runs the pipeline on tiny data with a tiny budget, in minutes on the CPU. It checks the setup, not the results.

---

## 1. Data (SV and DeepMIMO, exactly as GM-LAMP)

```
%run scripts/make_gmlamp_data.py --sv --deepmimo-ula "D:/Academic Profile/LADMM-MIMO/codes/H_ula_1.mat" --deepmimo-upa "D:/Academic Profile/LADMM-MIMO/codes/H_upa.mat"
```

The command writes four data sets:

| scenario | channels | train / validation / test | CS matrix |
|---|---|---|---|
| `SV_ULA`, `SV_UPA` | GM-LAMP's generator (`generate_channel.m`): L = 3 paths, half-wavelength spacing; ULA 256 or UPA 16×16 | **80,000 / 2,000 / 2,000** (as the authors' `generate_channel_data_for_SV_channel_model.m` and `Main.m`) | `CSmatrix256128` |
| `DeepMIMO_ULA`, `DeepMIMO_UPA` | DeepMIMO v1.3, O1_28, BS 3, rows 1000–1300, 3 paths; x = U0ᴴh with ‖x‖² = 256 | **50,000 / 2,000 / 2,000**. The test set is the authors' own test file; its users are removed from training and validation. | `CSmatrix_128256` |

- **What it writes:** `data/<SCENARIO>/` holds the channels, the authors' measurement files (bands 0–10 and 10–20 dB, validation at 10 dB) and `dataset_info.json`.
- **Expected end of the output:** `SV_ULA OK … DeepMIMO_UPA OK` and `all OK: identical to the data used so far`. The data are then bit-identical to the data of all earlier results.
- **Only some of them:** `--sv` alone makes only SV. The two `--deepmimo-…` options alone make only DeepMIMO.
- **Other SNR ranges need nothing here.** When a training range has no measurement file yet (for example one network for 0–20 dB), `train.py` makes it at the first training.
  - Such a file has 100 SNR chunks drawn from U(range), and the SNR of every column is stored. With `--snr-dist fair` (4.3) every column has its own SNR, spread evenly over the range (file `<range>_fair.mat`).
  - Its noise seed depends only on the range, so every computer makes the same file.
- **Test noise is never stored.** The evaluation adds it at 0, 2, …, 20 dB with fixed seeds, so every method sees exactly the same test measurements.
- **White noise instead of GM-LAMP's colored noise** needs nothing here either: with `--noise-model white` (4.10) the white files are made from these, with the same noise draw whitened.
- **Other sizes or splits** (not for the paper): see `DATA_GUIDE.md` section 6 and `scripts/make_dataset.py`.
- **Check them against the authors' files** (5 min):
  ```
  %run scripts/check_vs_authors.py
  ```
  - **Measurement files:** the SNR of every chunk, re-estimated from y − Ax with the authors' matrix, must match.
  - **Validation sets:** the authors' 16 published networks, run on our validation sets, must give the validation NMSE the authors logged at the end of their training, within the spread of a random 2000-channel set.
  - Results go to `results/authors_check/`. Details and the measured numbers: `docs/AUTHORS_PROTOCOL.md`.

### 1.1 A third data set: 3GPP TR 38.901 (optional)
The standard's stochastic channel (UMi street canyon, 28 GHz) with the arrays, carrier, lens, normalisation, CS matrix
and SNR bands of the DeepMIMO data. It is made by bsce's own Python implementation of the TR procedure
(`bsce/data/tr38901.py`), in the bsce environment: no MATLAB and no other package. Full guide and the measured checks:
`docs/3GPP_DATASET.md`.
```
%run scripts/make_3gpp_data.py
```
- **What it writes:** `data/3GPP_UPA` (UPA 16×16) and `data/3GPP_ULA` (ULA 256, horizontal).
  - Sizes: **80,000 / 2,000 / 2,000**, the sizes of SV.
  - The format of `data/SV_*` and `data/DeepMIMO_*`, plus `3GPP_<ARR>_meta.mat` (position, LOS state, large-scale parameters of every channel).
  - Scenario files: `configs/scenarios/3gpp_upa.yaml` and `3gpp_ula.yaml` (shipped).
  - Time: about 2 min per data set.
- **Other sizes:**
  - `--sizes dong2019` gives 81,000 / 9,000 / 19,000, the split of the 3GPP paper of Dong et al. (IEEE JSTSP 2019).
  - `--sizes deepmimo` gives 50,000 / 2,000 / 2,000, and `--sizes 100k` gives 96,000 / 2,000 / 2,000.
  - `--n-train N --n-val N --n-test N` gives any size.
  - The standard itself defines no size. The validation and test users are the same for every training size.
- **Options:**
  - `--only 3GPP_ULAZ` adds the vertical ULA, the geometry of the DeepMIMO ULA.
  - `--spec 19.2 --suffix _v19` uses the Rel-19 tables.
  - `--check` checks the data against 38.901 and against SV / DeepMIMO (`results/check_3gpp/`).
  - See `--help`.
- **Train and compare like the other scenarios:**
  - The scenario names are `3gpp_upa` and `3gpp_ula`; `--scenarios 3gpp` means both, and `--scenarios all 3gpp` all six data sets.
  - The authors published no networks for these data. `compare.py` therefore takes a GM-LAMP trained here as the reference for the gains (`GM-LAMP-T8_ft_0-10-20dB_*`, else `GM-LAMP-T8-ft`, `GM-LAMP-T8`, …), or the one you give with `--ref <tag>`.
- **Second implementations,** for a robustness check only: `scripts/sionna_make_H.py` (Sionna) and `matlab/quadriga_make_H.m` (QuaDRiGa, MATLAB). See `docs/3GPP_DATASET.md` 3.6.

---

## 2. Check that we reproduce the paper (LAMP, GM-LAMP)

### 2.1 The authors' published networks on our test sets (10 min)
```
%run scripts/compare.py --scenarios all --tags GM-LAMP-authors LAMP-authors --name authors
```
- This copies the authors' trained networks into `outputs/<SCENARIO>/GM-LAMP-authors` and `LAMP-authors`, then evaluates them.
- Results go to `results/authors/`. Expected NMSE in dB at 0 / 10 / 20 dB (measured with this code):

| scenario | GM-LAMP (authors) | LAMP (authors) |
|---|---|---|
| SV ULA | −9.17 / −16.78 / −22.96 | −8.58 / −16.31 / −21.90 |
| SV UPA | −8.45 / −15.11 / −20.32 | −7.61 / −14.64 / −20.03 |
| DeepMIMO ULA | −13.10 / −21.57 / −29.49 | −12.80 / −20.85 / −28.94 |
| DeepMIMO UPA | −13.15 / −21.27 / −28.79 | −11.85 / −19.47 / −27.06 |

These are the published networks on the published test sets, with the authors' noise model and NMSE. The DeepMIMO ULA numbers were checked earlier against the GM-LAMP paper. If you get these numbers, the test pipeline is right.

### 2.2 Retrain LAMP and GM-LAMP with the paper's own procedure (optional; GM-LAMP takes days, see the time below)
```
%run scripts/run_suite.py --scenarios all --methods lamp gmlamp --protocol paper --train-snr scenario --ft-snr none --device cuda
%run scripts/compare.py --scenarios all --tags "*_paper_trscenario" --name paper_check
```
- `--protocol paper` uses each method's own settings. For LAMP and GM-LAMP these are the authors' TF code: lr 1e-3, refinements ×0.5/0.1/0.01, batch 128, warm start, float64.
- `--train-snr scenario` trains the authors' two networks, 0–10 and 10–20 dB, on their files (validation at 10 dB). `--ft-snr none` means no fine-tune.
- **Expected result:** within a few tenths of a dB of the table in 2.1. Then training is reproduced too.
- **Stage by stage:** `%run scripts/check_vs_authors.py --no-data --runs "outputs/*/LAMP-T8_paper_trscenario" "outputs/*/GM-LAMP-T8_paper_trscenario"` compares the validation NMSE and the iterations of all 64 training stages with the authors' own training logs (in `reference/gmlamp/Train/Trained_Networks/*.npz`).
- GM-LAMP in float64 is slow on the P2000. To speed it up, add `--set model.precision=float32`; the folder name then ends in `_precision-float32`.
- **Time of GM-LAMP** (float32, P2000, from the speed of a run on 18 Oct): about 3 days per network.
  - A training iteration costs 32 ms at layer 1, 0.19–0.23 s at layer 4 and 0.25–0.37 s at layer 5.
  - The authors' log has 941,000 iterations per network, most of them in the deep layers.
  - The paper has two networks per scenario. So start with one scenario (`--scenarios sv_ula`), and use the authors'
    networks (2.1) for the others.
- **GM-LAMP before v31.1 trained with a one-Gaussian denoiser; retrain it** (fix of 18 Oct).
  - **The cause.** GM-LAMP's original init gives its 4 mixture components identical values. PyTorch then computes
    bit-identical gradients for them (measured on the CPU in float32 and float64; your GPU log shows the same plateau),
    so they never separated. The denoiser stayed one Gaussian: it scales every entry by the same factor, so it cannot
    use the sparsity, and the network stays near the level of a linear estimator.
  - **What it looked like** on SV ULA, 0–10 dB, validated at 10 dB: −2.93 dB after layer 1, where the authors' log has
    −9.18 dB, and about −4.5 dB at layer 5.
  - **Why the authors' code was fine.** Their TensorFlow code separates the components through round-off.
  - **v31.1** gives identical components the `gm_init=spread` offset before a stage trains them, and prints a note
    (`train.gm_symmetry=keep` turns it off).
  - **Which of your networks are affected:**
    ```
    %run scripts/check_gm_components.py --runs "outputs/*"
    ```
    It lists every GM-LAMP network with identical components. These are all those trained with bsce from the original
    init: `gmlamp_original.yaml`, `run_suite.py --methods gmlamp`, any protocol. Networks trained with `gmlamp_spread.yaml`
    and the authors' published networks are fine.
  - **Retraining.** Move the listed folders away, for example to `outputs_old/`, then run the same command.
    - An interrupted old training is not resumed: its checkpoint is renamed `….pre_v31_1.pt` and the training starts
      again.
    - A finished one would be skipped by `run_suite.py`, because its `.pt` files exist.

---

## 3. Choose the number of layers (all methods, all scenarios)

The idea: train **one deep network per method and scenario**, and save the network after every layer (`--save-layers`).

With layer-wise training, the network saved after layer T is exactly the network a run with T layers produces: same data, same seed, same protocol. This was checked bit for bit for all nine methods. So one deep run gives every depth from 1 to T_max, and you never retrain from scratch.

### 3.1 The deep runs
```
%run scripts/run_suite.py --scenarios all --methods ladmm12 lamp12 gmlamp12 --save-layers --ft-snr none --device cuda
```
- **More methods.** You can use any list, for example `lista24 alista24 lvamp10 admmnet20 nfladmm12 nfladmmc12`. The number is the maximum depth.
- **Splitting the work.** Each method and scenario is its own run, so you can split them over consoles or computers (section 4.3).
- **Output.** Each run writes `outputs/<SCENARIO>/<METHOD>-T12/`. Its `layers/T01_…pt` … `T12_…pt` are the networks after layer 1 … 12.
- **Snapshots need a layer-wise protocol** (`ladmm_best`, the default, or `gmlamp` / `fair`).
  - With `--protocol e2e`, `--schedule e2e`, or a method whose own training (`--protocol paper`) is end to end (NF-LADMM, UTRR), all layers are trained together, so there is no network after layer t to save.
  - Graph-LADMM's `gladmm`, `gladmmany`, `gladmmrw`, `gladmmrwtv`, `gladmmbp` and `gladmmmix` are such methods too: they always train end to end with their own settings.
  - **v31.2.** A list that mixes the two kinds runs: only the layer-wise methods save their layers, and the others train
    at full depth, with a note. A list in which no method trains layer by layer still stops at once, before any training.
    (v25 to v31.1 stopped on any end-to-end method in the list.)
  - **An end-to-end depth curve:** train every depth itself, e.g. `--methods ladmm2 ladmm4 ladmm6 ladmm8 ladmm10 ladmm12 --protocol e2e`, without `--save-layers`.

### 3.2 The depth curves (test set, every SNR)
```
%run scripts/depth_snapshots.py --runs "outputs/*/LADMM-T12" "outputs/*/LAMP-T12" "outputs/*/GM-LAMP-T12" --name depth
```
The results go to `results/depth/`.
- **`depth_curve.png` / `.pdf`, one column per run.**
  - Top: test NMSE against T at 0, 10 and 20 dB, plus the mean over 0–20 dB.
  - Bottom: the validation NMSE logged when each layer finished, plus the gain of each added layer.
  - All values are printed on the curves.
- **`depth_curve.csv` and `depth_curve.md`:** the same numbers, plus the "flat point".
  - The flat point is the smallest T within 0.1 dB of the best **validation** NMSE (v31.3; before, the test mean). `--tol 0.2` changes the threshold.
- **Choose T on validation data, never on the test curves** (that would tune the proposed methods on the test set while the
  baselines keep their papers' depth). The test curves are shown for the record. Pooled validation is dominated by low SNR:
  to weigh high SNR, use `select_depth.py` (validation per band) rather than the 20 dB test curve.

### 3.3 Use the chosen depth without retraining
```
%run scripts/promote_layer.py --run "outputs/*/LADMM-T12" --T 6 4
%run scripts/promote_layer.py --run "outputs/*/LAMP-T12" "outputs/*/GM-LAMP-T12" --T 8
```
- This writes normal run folders, for example `outputs/SV_UPA/LADMM-T6/`.
- `run_suite.py` with `ladmm6` / `ladmm4` / `lamp8` then reports "trained, skipped" and goes straight to the fine-tune.

---

## 4. Train every method with the same protocol (+ fine-tune per band)

### 4.1 The final command (defaults = the paper setting we chose)
```
%run scripts/run_suite.py --scenarios all --methods all --device cuda
```
- **With the 3GPP data sets (1.1):** `--scenarios all 3gpp` instead of `--scenarios all`.
- **`--methods all`** means every learned method at its paper depth:

| key | method | default T | folder |
|---|---|---|---|
| `ladmm` | LADMM, full 256×256 operator per layer | 6 | `LADMM-T6` |
| `lamp` | LAMP (Borgerding; Wei–Hu–Dai) | 8 | `LAMP-T8` |
| `gmlamp` | GM-LAMP (K = 4) | 8 | `GM-LAMP-T8` |
| `lista` | LISTA (complex) | 16 | `LISTA-T16` |
| `alista` | ALISTA (analytic W) | 16 | `ALISTA-T16` |
| `lvamp` | LVAMP | 6 | `LVAMP-T6` |
| `admmnet` | ADMM-Net (THADMM-Net) | 15 | `ADMM-Net-T15` |
| `nfladmm` | **NF-LADMM-S** (proposed, CNN denoiser) | 8 | `NF-LADMM-S-T8` |
| `nfladmmc` | **NF-LADMM-C** (proposed, radial denoiser) | 8 | `NF-LADMM-C-T8` |
| `nfladmmsr` | **NF-LADMM-SR** (proposed: CNN prior + over-relaxed update) | 4 | `NF-LADMM-SR-T4` |
| `utrr` | UTRR (Wu–Cheng–Eldar–Cioffi, TCOM 2025). **Not in `all`**: name it (4.9) | 30 | `UTRR-T30` |

- **Any depth:** add the number to the key, for example `--methods ladmm4 ladmm6 lamp8 lista12`.
- **In the workbench** (step 4, "Train and fine-tune") the methods table starts with LADMM 6 4, LAMP, GM-LAMP and NF-LADMM-SR ticked; tick every row for `--methods all`. A row's "own settings" cell (e.g. `model.sigma=residual` for NF-LADMM) applies to that method only; the "Extra settings" box applies to every ticked method.
- **Two steps per network:**
  1. **Training:** one network for 0–20 dB (`--train-snr 0 20`, the default), with the common protocol below.
  2. **Fine-tune per band:** 0–10 and 10–20 dB (`--ft-snr 0 10 20`, the default).
     - 20,000 iterations per band (`--ft-iters`), all parameters, Adam lr 5e-5 then 5e-6 for the last third, batch 400.
     - Fresh noise every step, with SNR ~ U(band); validation over the whole band; the best validated state is kept.
     - It runs on the data the network was trained on, and writes `<network>_ft_0-10-20dB_20k/` next to it.
     - Early stop (v25): a learning rate ends after 4000 steps without a 0.01 dB gain, and so does the band in the second one (`--ft-patience 0` = always the full budget; 4.6).
- **If anything stops it** (power cut, Ctrl+C, a closed window), run the same command again.
  - Finished networks and fine-tunes are skipped.
  - A training continues from its last checkpoint: after every stage, and inside a stage every 10 minutes.
  - A fine-tune keeps its finished bands and continues the band it was in.
  - The result is exactly the network of a run that never stopped (tested). `scripts/runs.py` shows what is done and where each interrupted run will resume (4.11).

### 4.1b One method, one network, by hand (the short forms)
```
%run scripts/train.py --scenario sv_upa --method ladmm6 --device cuda
%run scripts/finetune_bands.py --from outputs/SV_UPA/LADMM-T6 --device cuda
```
- **`train.py --method ladmm6`** trains exactly the network of `run_suite.py --methods ladmm6`, in the same folder `outputs/SV_UPA/LADMM-T6/`:
  - the common protocol (`ladmm_best`);
  - one network for 0–20 dB.
- **Options:** `--protocol paper`, `--snr 0 10 20`, `--set ...`, `--noise-model white`, `--seed 1`, as in `run_suite.py`.
- **`finetune_bands.py --from <run folder>`** needs no name and no file path. It writes `outputs/SV_UPA/LADMM-T6_ft_0-10-20dB_20k/` with `LADMM-T6_ft_0-10-20dB_20k_0to10dB.pt` and `..._10to20dB.pt`.
  - Other bands or budget: `--snr 0 5 10 15 20 --iters 10000` → `LADMM-T6_ft_0-5-10-15-20dB_10k/`.
  - Every other option that differs from its default is added to the name (`_margin2`, `_val10`, `_clip1`, `_lr0p0001`, …: a dot becomes `p`), so two settings never share a folder.
  - `--dry-name` prints the name without training.
- **The names, the same in every script:**

  | what | name | example |
  |---|---|---|
  | trained network | `<METHOD>-T<T>[_<protocol>][_tr<edges>][_<option>][_s<seed>]` | `LADMM-T6`, `LAMP-T8_paper_trscenario`, `LADMM-T6_restore-best-true_s1` |
  | fine-tuned per band | `<trained name>_ft_<band edges>dB_<iterations per band>[_<option>]` | `LADMM-T6_ft_0-10-20dB_20k`, `LADMM-T6_s1_ft_0-5-10-15-20dB_20k_margin2` |
  | its files | `<folder>_<band>.pt` / `.mat` / `_orig.mat` | `LADMM-T6_ft_0-10-20dB_20k_10to20dB.pt` |

- **Fine-tunes made before v24** are named `<network>-ft` and `<network>-ft0-5-10-15-20`:
  - they are still found, and `run_suite.py` skips them when their budget is the one asked for;
  - `%run scripts/rename_runs.py` lists them with their new names, and `%run scripts/rename_runs.py --apply` renames them.

### 4.2 The protocols (`--protocol`; the same settings for every method)

| protocol | settings | best for |
|---|---|---|
| **`ladmm_best`** (default) | the protocol of your best LADMM (`untitled2.py`): layer-wise, lr 5e-4 then ×0.5/0.2/0.1/0.01, batch 400, stop after 5000 steps without improvement (at most 100k per stage), float32, no warm start, last weights kept, training noise from the file | the paper (all current results) |
| `gmlamp_best` | GM-LAMP's procedure (lr 1e-3, ×0.5/0.1/0.01, batch 128, warm start, float32) + the best validated weights restored after every stage | "best validation" variant |
| `gmlamp` | GM-LAMP's procedure, last weights kept | GM-LAMP's protocol for everyone |
| `fair` | layer-wise, lr 1e-3, batch 512, at most 50k per stage | shorter budget |
| `e2e` | all layers together, 30k steps, cosine lr 1e-3, gradient clipping, best state kept | end-to-end training |
| `paper` | each method's **own** settings (not the same for all) | reproduction (section 2.2) |

- **Variants of any protocol:** add `--set`, for example `--set train.restore_best=true` (`ladmm_best` + best validation).
  - The override becomes part of the folder name (`LADMM-T6_restore-best-true`), so variants never overwrite each other.
- **Training noise:** for every method it comes from the same measurement file, so all methods see the same noisy training data.

### 4.3 Any SNR range
| you want | options | folders |
|---|---|---|
| one network 0–20 dB, then fine-tune 0–10 / 10–20 (default) | – | `LADMM-T6`, `LADMM-T6_ft_0-10-20dB_20k` |
| one network 0–20 dB, no fine-tune | `--ft-snr none` | `LADMM-T6` |
| one network 0–20, fine-tune into four bands | `--ft-snr 0 5 10 15 20` | `LADMM-T6_ft_0-5-10-15-20dB_20k` |
| two networks trained directly (0–10, 10–20) | `--train-snr 0 10 20 --ft-snr none` | `LADMM-T6_tr0-10-20` |
| one network −5 … 30 dB | `--train-snr -5 30 --ft-snr none` | `LADMM-T6_trm5-30` |
| the authors' bands and files (validation at 10 dB) | `--train-snr scenario --ft-snr none` | `LADMM-T6_trscenario` |

The evaluation always picks the narrowest band of a folder that contains each test SNR. When a test SNR is outside every band, it uses the nearest band.

**How the data of a band are made** (validation noise, edges, SNR chunks). Every option combines with every row above and with every protocol. The defaults are as before.

| option | default (as before) | what it does | folder suffix |
|---|---|---|---|
| `--val-snr range` | the authors' bands: validation at 10 dB; `--train-snr` ranges and the fine-tune: over the whole range | validation noise with SNR ~ U(band) for every band, fixed (a new validation file `…_<band>_valrange.mat`) | `_valrange` |
| `--val-snr 10` (or `0 5 10`) | as above | validation at fixed SNRs for every band, also for ranges and the fine-tune (the authors: 10 dB for both networks) | `_val10`, `_val0-5-10` |
| `--margin 2` | 0 (the authors train 0–10 and 10–20 exactly) | **edges:** each band is trained on [lo − 2, hi + 2] dB and still used on [lo, hi], so its edges are trained like its middle. Validation stays on the band. | `_margin2` |
| `--ft-margin 2` | 0 | the same for the fine-tune bands | `_ft_0-10-20dB_20k_margin2` |
| `--snr-chunks 100` | the authors' files 10, ranges 100 | number of SNR chunks of a training file. The authors' 10 chunks cover a band coarsely: our 0–10 dB file has no SNR between 1.5 and 5.1 dB. With a margin, use 100. | `_c100` |
| `--snr-dist fair` (v26) | SNR chunks, as above | **every training sample its own SNR, spread evenly over the band**, in random order (one SNR in each of K equal parts of the range; 80,000 distinct SNRs, no gap). Also every online minibatch (NF-LADMM, the fine-tune) is spread evenly, and a validation over a range uses equally spaced SNRs (100 chunks of 20 channels) instead of random ones. Validation at fixed SNRs (the authors' 10 dB) stays. | `_fair` |
| `--snr-dist fair --snr-step 0.2` | – | the same on the grid lo, lo + 0.2, …, hi (0, 0.2, …, 20 dB: 101 values), every value equally often | `_fair0p2` |

- **Examples:**
  - The default suite, edges kept: `--margin 2 --ft-margin 2`.
  - The authors' protocol with validation over the band: `--protocol gmlamp --train-snr scenario --ft-snr none --val-snr range`.
  - The authors' protocol with edge margins: add `--margin 2 --snr-chunks 100`.
  - Every SNR of a wide range trained equally: `--train-snr -10 30 --ft-snr -10 0 10 20 30 --snr-dist fair`.
- **Why `fair`:** an SNR chunk file has only as many SNRs as chunks, drawn at random. The authors' 0–10 dB file has 10 SNRs and nothing between 1.5 and 5.1 dB; our 0–20 dB file has 100 SNRs and gaps up to 1.06 dB. `fair` has none (figure in `docs/changes/CHANGES_2026-10-16_snr_dist.md`). The fine-tune and NF-LADMM already drew one SNR per sample, i.i.d.; `fair` makes every minibatch cover the band evenly as well.
- **Measured** (LADMM-T2 end to end, SV ULA, CPU; the change note has the table):
  - the mean over the band is 0.05–0.11 dB better;
  - above 3–5 dB fair is better at every SNR (up to +0.30 dB); at the lowest SNRs it is worse (down to −0.33 dB).
  - The old files gave the low end more samples by chance: the authors' 0–10 dB file has 40 % at 2 dB or less, fair has 20 %.
- **`fair` with `--snr-chunks`** is refused, because chunks have no meaning there.
- **No existing file changes.** A different file gets its own name: training `…_m2to12dB.mat`, `…_m2to12dB_c10.mat`; validation `…_0to10dB_valrange.mat`, `…_0to10dB_val10dB.mat`. A band's validation file is the same with and without a margin.
- **In the workbench:** step 4, fields "Validation noise", "Edge margin", "SNR chunks per training file" and "Training SNRs" (chunks / fair, with the grid step), in both training tasks; "Training SNRs" also in "Fine-tune one network".

### 4.4 Several consoles or computers: same networks, nothing to retrain
Every network is trained in its own process, with its own fixed seed. Its data files are made the same way on every computer. So this single command:
```
%run scripts/run_suite.py --scenarios sv_upa --methods ladmm6 ladmm4 lamp gmlamp --device cuda
```
gives the same networks as these two commands, run on two computers:
```
PC 1:  %run scripts/run_suite.py --scenarios sv_upa --methods lamp gmlamp --device cuda
PC 2:  %run scripts/run_suite.py --scenarios sv_upa --methods ladmm6 ladmm4 --device cuda
```
- **Checked here:** the split runs are identical to one command (weight difference 0).
- **Bringing the results together:** copy each `outputs\<SCENARIO>\<network>\` folder into one bsce folder, then run step 5.
- **To be sure the data were identical:** every training `README.txt` lists a SHA-1 of the data files it used. Equal SHA-1 on both computers means the same data. The simplest way is to copy `data\` from one computer to the other.
- **Same GPU model and same software:** bit-identical networks (deterministic mode is on).
- **Different GPU models:** floating-point rounding differs slightly. Results then agree to within a small fraction of a dB, but not bit for bit. For the paper, train all methods of one scenario on the same kind of GPU if you can.

### 4.5 Seeds
- `--seed 1` gives another training seed. The folders get `_s1` (fine-tuned: `_s1_ft_0-10-20dB_20k`).
- `scripts/aggregate_seeds.py --csv results/<name>/<SCENARIO>/nmse_vs_snr.csv` gives mean ± std over seeds, for one-network and fine-tuned runs.

### 4.6 Schedule and gradient clipping (the same for every method)
| option | default (as before) | what it does | folder suffix |
|---|---|---|---|
| `--schedule layerwise` | yes | each method as its protocol says (layer by layer for `ladmm_best`, `gmlamp`, `gmlamp_best`, `fair`) | – |
| `--schedule e2e` | – | **every** method trained end to end (all layers together from the start). Adam, cosine decay lr → 0.05·lr over `--e2e-steps` (default 30000), validation every 500 steps, best state kept. lr, batch, loss, noise and precision stay the protocol's. | `_e2e30k` |
| `--grad-clip 1.0` | – | clips the gradient norm of **every** method at 1.0, in training and in the fine-tune | `_clip1` |
| `--grad-clip none` | – | no clipping for any method, also where a method's own config clips (`--protocol paper`, `e2e`) | `_noclip` |
| (no `--grad-clip`) | yes | as before: the common protocols clip nobody, `e2e` clips everybody at 1.0, `paper` keeps each method's own setting | – |

The options combine, and they combine with every protocol: `--schedule e2e --grad-clip 1.0 --protocol gmlamp`.
Every run's `README.txt` lists `schedule`, `loss`, `noise` and `grad_clip`.

**Early stop (v25): nothing runs on without a gain.**

| training | when it stops | change it |
|---|---|---|
| end to end (`--protocol e2e`, `--schedule e2e`, NF-LADMM's own `paper` training) | when the best validation has not improved by **0.01 dB** during the last **5000 steps**. The best validated state is kept, as before. | `--set train.patience_steps=10000`, `train.min_gain_db=0.02`; `train.patience_steps=0` runs every step, as up to v24 |
| fine-tune per band (`finetune_bands.py`, the suite's fine-tune) | when the best has not improved by 0.01 dB during 4000 steps, the first learning rate ends and the second starts early; in the second, the band ends | `finetune_bands.py --patience 8000 --min-gain 0.02`; `run_suite.py --ft-patience 0` = the whole budget (suffix `_pat0`) |
| layer-wise (`ladmm_best`, `gmlamp`, `fair`) | as before: a stage ends `better_wait` steps (5000) after its last new best | `--set train.min_gain_db=0.01` also ignores gains smaller than 0.01 dB (off by default, so the authors' and `untitled2.py`'s rule stays exact) |
| UTRR (`--protocol paper`) | as before: 10 epochs without improvement (its paper) | `train.min_gain_db` as above |

- **The stop is printed:** `early stop at step 19500: no gain of 0.01 dB or more since step 14500`. It is also recorded in `train_log.json` / `finetune_log.json`.
- **The effect on your own e2e deep runs,** replayed on their logs:
  - LADMM-T12 stops at step 19,500 instead of 30,000, saving 14 min (SV ULA) and 12 min (SV UPA), with the same best network;
  - NF-LADMM-C-T8 saves 7 min and loses 0.004 dB;
  - NF-LADMM-S and -SR still gained 0.01 dB or more per 5000 steps, so they run to the end.
- **Runs trained before v25 had no early stop.** The suite skips them as before; their networks stay valid.

### 4.7 Ablations worth running (each changes ONE thing against the default suite)
| # | question | command (add `--scenarios ...`) | why |
|---|---|---|---|
| A1 | layer-wise or end-to-end training: which is better for each method? | `--methods ladmm6 lamp gmlamp nfladmmsr --schedule e2e` | the CPU pilot favoured end to end for NF-LADMM; layer-wise is the authors' choice for LAMP / GM-LAMP |
| A2 | does gradient clipping help anyone? | `--methods ladmm6 lamp gmlamp nfladmmsr --grad-clip 1.0` | clipping now changes all methods together, so the comparison stays fair |
| A3 | is NF-LADMM's gain from knowing σ²? | `--methods nfladmmsr --set model.sigma=residual` | fairness: the other methods estimate the noise level (see docs/REVIEW_2026-10-15.md, N1) |
| A4 | seed variance (needed to call a gain of ~1 dB significant) | the default suite with `--seed 1`, then `--seed 2`; then `aggregate_seeds.py` | the gains over GM-LAMP are 1–4 dB; their spread over seeds is not measured yet |
| A5 | best validated weights instead of last weights | `--set train.restore_best=true` | the authors keep the last weights; best weights cost nothing and often add a little |
| A6 | per-sample log-NSE loss instead of pooled NMSE | `--set train.loss=log_nse` | pooled NMSE is dominated by low SNR; log-NSE weights every SNR equally and may close the one-network gap at 20 dB |
| A7 | longer end-to-end budget (compute closer to the layer-wise runs) | `--schedule e2e --e2e-steps 100000` | 30k steps are far fewer gradient steps than a layer-wise T = 6 run |
| A8 | wider training range (robust at the edges) | `--train-snr -5 25` | the 0 dB and 20 dB test points sit on the edges of a 0–20 dB network |
| A9 | fine-tune budget | from the trained network, no retraining: `%run scripts/finetune_bands.py --from outputs/SV_UPA/LADMM-T6 --iters 10000` → `LADMM-T6_ft_0-10-20dB_10k` (same for 40000 → `_40k`, and for each method); compare with `--tags "*_ft_0-10-20dB_10k" "*_ft_0-10-20dB_20k" "*_ft_0-10-20dB_40k"` | gives the curve "gain vs fine-tune budget" for every method |
| A10 (v28) | is the SV-ULA gap the soft-threshold bias? (no training) | `%run scripts/compare.py --scenarios sv_ula --tags "*_ft_*" --polish band --name polish_check` | the polish removes the bias on the strongest beams; a gain of 0.5 dB or more at 15–20 dB says the shrinkage is the limit |
| A11 (v28) | **GM shrinkage + the exact data step of the folded noise** (the best variant of the CPU check) | `--scenarios sv_ula --methods ladmm6 --set model.threshold=gm model.data_step=whitened` | CPU check, SV ULA, T = 4, short end-to-end budget: +1.72 dB mean and +2.61 dB at 20 dB over the soft LADMM-F, the same with 2 seeds; GM alone −0.06, whitened alone +0.54 |
| A12 (v28) | each part alone | `--set model.threshold=gm`, `--set model.data_step=whitened` (also `adaptive`, `firm`) | to show the two parts work together (a free H cannot reproduce the whitened step from Aᵀy) |
| A13 (v28) | LADMM-A: nothing learned in the operator | `--methods ladmm6 --set model.operator=analytic` | 18 parameters, any combiner; for the theory and the transfer, not for the best NMSE |
| A14 (v28) | the published untied form (CADMM) | `--methods cadmm` | reviewers' baseline: LADMM-F is its tied case; CPU check +1.17 dB over the soft LADMM-F (its M1 acts on y and can learn the whitening) |
| A15 (v28) | ensemble of two depths | `--methods ladmm6 ladmm8 --ensemble ladmm6 ladmm8` | costs the two networks; helps only when they are within about 1.5 dB |

### 4.8 The authors' protocol for every method (fair comparison on the authors' terms)
The default suite (4.1) trains every method with one common protocol (one 0–20 dB network, then a fine-tune per band).
A reviewer may also ask: **does the method still win when it is trained exactly like GM-LAMP, on GM-LAMP's data?**
This trains every method with the authors' own procedure (`Train_GM-LAMP.py` + `define_networks.do_training`):
```
%run scripts/run_suite.py --scenarios all --methods ladmm6 ladmm4 lamp8 gmlamp8 nfladmmsr4 --protocol gmlamp --train-snr scenario --ft-snr none --device cuda
%run scripts/compare.py --scenarios all --tags "*_gmlamp_trscenario*" --name authors_protocol
```
With the 3GPP data sets: `--scenarios all 3gpp` in both commands; `compare.py` then takes
`GM-LAMP-T8_gmlamp_trscenario` as the reference for the 3GPP tables.
| item | the authors (`Train Network/`) | here |
|---|---|---|
| networks | two per data set: 0–10 dB and 10–20 dB; the test uses the 0–10 dB network below 10 dB | `--train-snr scenario` |
| training data | the measurement files: 10 chunks, one SNR ~ U(band) each, noise fixed in the file | the scenario's files (step 1) |
| validation | 2000 channels at 10 dB, for both networks | the same |
| schedule | per layer: the new linear part at lr 1e-3, then all parameters at 1e-3 × {0.5, 0.1, 0.01}; the same for the non-linear part | `--protocol gmlamp` |
| details | Adam (TF1), batch 128 drawn with replacement, new layer initialised from the previous one, validation every 10 iterations (rounded to 5 decimals), stop when the best is 5000 iterations old (checked every 1000), at most 10⁶ iterations per stage, last weights kept | the same |
| precision | float64 | float32 for every method (fast on the P2000); add `--set model.precision=float64` to be exact |

- Folders: `<method>-T<T>_gmlamp_trscenario` (`…_precision-float64` with float64). No fine-tune: the authors have none.
- **What a difference means:**
  - Between methods trained here, the comparison is exact: every method uses the same files.
  - Against the authors' published networks, a few tenths of a dB is within the recipe's own randomness. The 10 random SNRs of a training file alone move the first layer by ±0.2 dB, and the authors' own two runs differ by up to 0.2 dB (`docs/AUTHORS_PROTOCOL.md` 3.1).
- For LAMP and GM-LAMP this is `--protocol paper` except for the precision (2.2). So `LAMP-T8_gmlamp_trscenario` and `GM-LAMP-T8_gmlamp_trscenario` are the authors' networks retrained with our code. Compare them with the published ones too.
- Workbench: step 4 → "Every method with the authors' protocol"; step 5 → Networks: "trained with the authors' protocol".
- `scripts/check_vs_authors.py` checks the data against the authors' files. With `--runs`, it compares every training stage of a retrained LAMP or GM-LAMP with the authors' own training log (`docs/AUTHORS_PROTOCOL.md`).

### 4.9 UTRR, the accuracy reference on this setup (`docs/UTRR.md`)
UTRR (Wu, Cheng, Eldar, Cioffi, IEEE TCOM 2025) reports the best published NMSE on exactly this setup (SV ULA, N = 256,
M = 128). It has 30 layers and about 2 M parameters, and no code was published: bsce's version is written from the
paper and checked against it (`docs/UTRR.md`).
1. **First reproduce the paper's Table II in its own setting** (noise after the combiner), one model per top-K, then
   the ensemble:
   ```
   %run scripts/utrr_reproduce.py --snr 20 --K 0 16 64 256 --device cuda
   ```
   Paper: −26.97 / −26.93 / −26.73 / −26.94 dB, UTRR-E −30.18 dB (10 dB: `--snr 10`, −18.92 … and −22.06 dB).
2. **Then train it in bsce's setting with the others.** Its own training (`--protocol paper`) is the practical one.
   Under the layer-wise `ladmm_best` protocol, 30 layers × 5 stages would take days on the P2000. That is roughly
   20 times LADMM T = 6: 2.5 times the stages, about 4 times the mean depth per stage, and twice the cost per layer.
   ```
   %run scripts/run_suite.py --scenarios all 3gpp --methods utrr --protocol paper --device cuda
   %run scripts/run_suite.py --scenarios all 3gpp --methods utrr --protocol paper --set model.K=0 --device cuda
   ```
   - Also run `model.K=64` and `model.K=256` for the ensemble.
   - The folders are `UTRR-T30_paper`, `UTRR-T30_paper_K-0`, …, each with its fine-tuned pair `<folder>_ft_0-10-20dB_20k`.
3. **The ensemble UTRR-E**, one call per scenario:
   ```
   %run scripts/make_ensemble.py --runs outputs/SV_ULA/UTRR-T30_paper_K-0_ft_0-10-20dB_20k outputs/SV_ULA/UTRR-T30_paper_ft_0-10-20dB_20k outputs/SV_ULA/UTRR-T30_paper_K-64_ft_0-10-20dB_20k outputs/SV_ULA/UTRR-T30_paper_K-256_ft_0-10-20dB_20k --tag UTRR-E-T30_paper_ft_0-10-20dB_20k
   ```
   `compare.py` then uses it like any other network.

### 4.10 White against colored measurement noise (`--noise-model white`; `docs/NOISE_MODELS.md`)
bsce's data follow GM-LAMP: the noise is added before the combiner, y = A(x + n), so the measurement noise is
**colored** (covariance σ²AAᵀ). UTRR and most other papers add **white** noise after the combiner, y = Ax + w. Every
step can now use either:

| noise model | measurements | noise covariance | where the runs go |
|---|---|---|---|
| `colored` (default, GM-LAMP's) | y = A(x + n), n ~ CN(0, σ²I) | σ² AAᵀ | `outputs/<SCENARIO>/` |
| `white` | y = Ax + w, w ~ CN(0, cσ²I), c = tr(AAᵀ)/M | cσ² I | `outputs/<SCENARIO>_white/` |

- **The same SNR means the same signal and the same mean noise power** (E‖w‖² = E‖An‖²). Only the colour differs.
- **The two data sets are paired.** The white noise is the colored draw whitened, w = √c (AAᵀ)^(−1/2) A n. So every white
  file, validation set and test set uses the same channels and the same random numbers as its colored twin. A
  difference between the two is due to the noise colour, not to another noise draw. The colored files are unchanged.
- **NF-LADMM** uses the exact data step of the noise it is trained on (the white one is
  x = v + Aᵀ(AAᵀ + c ρ̃ I)⁻¹(y − Av), the same parameters). Every other method learns from the data as before.

1. **Train with white noise** (the same command as 4.1 + `--noise-model white`; the folder names stay the same):
   ```
   %run scripts/run_suite.py --scenarios all --methods nfladmmsr4 ladmm6 gmlamp8 --noise-model white --device cuda
   ```
   The white measurement files (`<colored name>_white.mat`) are made automatically, from the colored files of the same
   draw. To make them ahead of time: `make_dataset.py ... --noise-model both` or
   `make_sv_measurements.py --scenario ... --noise-model white`.
2. **Compare the two noise models**, with cross tests (trained on one, tested on the other):
   ```
   %run scripts/compare_noise.py --scenarios all --tags "*_ft_*" --nf-matched --name noise_comparison
   ```
   - It shows each network on colored and on white test noise.
   - Solid lines: trained on the test's noise. Dashed: trained on the other noise. Dotted: NF-LADMM trained on the
     other noise but run with the test's data step (`--nf-matched`).
   - A third panel shows the mismatch loss at every SNR.
   - Writes `noise_comparison.png` / `.pdf` and `noise_summary.csv` / `.md`.
3. **The usual comparison within white noise:**
   ```
   %run scripts/compare.py --scenarios all --tags "*_ft_*" --noise-model white --name final_finetuned_white
   ```
   The published GM-LAMP / LAMP were trained with colored noise, so they are not imported into the white folder. The
   reference there is the GM-LAMP trained here with white noise; `compare_noise.py` shows the published ones on white
   noise.
- **One run by hand:** `train.py --noise-model white`, `evaluate.py --noise-model white` (the test noise; `auto` takes it
  from the folder name), `evaluate.py --nf-step matched`, `finetune_bands.py` (takes the noise of its source network).
- UTRR's own Table II reproduction (`utrr_reproduce.py`, 4.9) keeps the paper's definition: white noise with the SNR
  fixed for every channel, ‖Ax‖²/‖w‖².

### 4.11 What is done, what resumes (`runs.py`)
```
%run scripts/runs.py
```
- **What it lists:** every folder of `outputs/<SCENARIO>/`, with:
  - what it is: trained, or fine-tuned from which network and with how many iterations;
  - method, layers and parameters;
  - the band networks present, out of those expected;
  - the best validation NMSE per band, and the last change.
- **The state of each run:**
  - `done`;
  - `resumable: <stage, iteration or band and step>`: the same command continues there;
  - `incomplete`: a band is missing with no checkpoint, so the same command makes it.
- **Where it writes:** `outputs/RUNS.md` and `outputs/RUNS.csv`. `--scenario SV_UPA` limits it to one scenario (and `SV_UPA_white`).
- **The checkpoint interval:** 10 minutes by default. Change it with `--set train.checkpoint_minutes=5` for training, or `finetune_bands.py --checkpoint-minutes 5`.

---

### 4.12 Graph-LADMM and networks that work with any combiner (v29)
**What it is.** The LADMM layer with a beam graph:
- a second ADMM split z₂ = Dx carries a graph total variation on the magnetic edges of the beam ring (ULA) or torus (UPA);
- the node step is an energy gate on the neighbours (bias-free, `gate`), or GM-MMSE (`gm`, `gm_graph`);
- the edge phase is the phase step of a path's leakage between neighbouring beams, estimated from the training channels
  when the training starts (printed: about ±180° on the SV sets, +1° / +11° on DeepMIMO ULA / UPA).

**Three forms:**

| key | x-step | combiner | parameters |
|---|---|---|---|
| `gladmm` | analytic (P+ρ(I+DᴴD))⁻¹, graph TV + gate | any: it follows the combiner it is given | 54 (T = 8) |
| `gladmmf` | free H_t per layer, graph gate | one, like LADMM-F | 393,250 (T = 6) |
| `gladmmany` | analytic, graph TV + gate, trained on a new random combiner every minibatch | any, by training | 54 (T = 8) |
| `gladmmmix` (v31) | as `gladmm`, trained on a mixture of combiners: M ∈ {64, 96, 128, 160}, random ±1 / Gaussian / partially connected / DCT rows | any, by training | 54 (T = 8) |
| `gladmmbp` (v31) | as `gladmm` + the beam prior (gate logit + β_t ℓ_i, ℓ = the log beam power of the training channels) | any | 318 (T = 8) |
| `gladmmrw` (v31) | graph-reweighted l1, unrolled (the uploaded package's method): reweighted soft threshold, the package's plain data step, no TV | any | 44 (T = 8) |
| `gladmmrwtv` (v31) | graph-reweighted l1 in `gladmm`'s layer: whitened data step + magnetic TV | any | 51 (T = 8) |

**Parameters (v31).** Every stored parameter is trained. Before v31, `gladmm` stored 56 parameters, of which 54 were
trained: k₂ and e of the last layer never reached the output, so they kept their initial values. The last layer's output
is its z₁. The z₂ step and the dual updates computed after it would only feed a layer T + 1, so v31 neither creates
those two parameters nor computes that tail. The same holds for:
- ε, a1 and a2 of layer 1 of the reweighted z-steps, whose weights are 1 there;
- ρ of layer 1 with a free H;
- the pooling weights without graph pooling.

Networks saved by older versions load, and give the same output: those entries are ignored.

Parameter ratios against the authors' two GM-LAMP networks (524,480 parameters):
- one network: 1/9,713;
- the per-band pair (2 × 54): 1/4,856.

**Inference cost (v31).** With the TV x-step, the 8 matrices H_t = (P + ρ_t(I + DᴴD))⁻¹ are now computed once per
combiner (and per set of ρ_t) instead of being factorised in every call. The outputs are the same to the arithmetic
precision, and one channel then costs 2.72 M real multiply-accumulates (MACs), 2.8 × GM-LAMP's 0.98 M; before, it cost
249 M. I + DᴴD is built from the edges. compare.py's tables give the trained parameters and the MACs per channel of
every network (§5).

**Depth (check of 19 Oct; 9,000 steps; mean NMSE over 0–20 dB on the authors' test sets).**

| | T = 8 | T = 10 | T = 12 | GM-LAMP (authors) |
|---|---|---|---|---|
| SV ULA | −16.81 (2 seeds) | **−17.02** (2 seeds) | −17.18 (1 seed) | −16.87 |
| SV UPA | −11.71 | −11.67 | – | −15.18 |

- On SV ULA, 10 layers gain +0.21 dB in both seeds, and 12 layers another +0.12 dB. The gain grows with the SNR.
  `gladmm12` beats GM-LAMP at all 11 SNRs.
- On SV UPA, 10 layers do not help: better at 0–10 dB, worse at 12–20 dB.
- Cost per channel: 2.72 / 3.38 / 4.03 M MACs for T = 8 / 10 / 12.
- For a depth curve, train each depth (`--methods gladmm6 gladmm8 gladmm10 gladmm12`); the end-to-end training has no
  per-layer snapshots.
- Choose the depth per scenario on the validation set.

```
%run scripts/run_suite.py --scenarios all --methods gladmm gladmmany --device cuda
%run scripts/run_suite.py --scenarios all --methods gladmmf --protocol paper --device cuda
```
**Their training (v30).**
- **`gladmm` and `gladmmany` always train with their own settings,** whatever `--protocol` is: end to end, online noise,
  the per-sample log-NSE loss, as NF-LADMM. Their folders end in `_paper` (`G-LADMM-T8_paper`), as with `--protocol paper`.
  - The reason: on SV ULA, GM-LAMP's layer-wise procedure at the same budget was 1.7 dB worse, at every SNR (check of
    18 Oct).
  - `--force-protocol` gives them `--protocol` like every other method.
- **`gladmmf`** takes `--protocol` like every method; its own end-to-end training is `--protocol paper`.
- **The fine-tune per band** keeps the random combiners of `gladmmany`.
- **Budget.** The config trains 30,000 steps. On SV ULA the 54-parameter network reached its plateau at about 4,500
  steps: 9,000 steps gave −16.85 dB on average, level with the authors' two GM-LAMP networks (−16.87 dB). On SV UPA
  every Graph-LADMM variant was flat after 3,000 steps (v31 check, 4,500 steps).
  - A shorter run: `--set train.steps=9000` (folder `G-LADMM-T8_paper_steps-9000`), or the method row's settings cell
    in the workbench.

**Test on another combiner or data set** (no retraining):
```
%run scripts/evaluate.py --scenario configs/scenarios/sv_ula.yaml --runs outputs/SV_ULA --combiner random:11 --out results/SV_ULA_newA
%run scripts/evaluate.py --scenario configs/scenarios/sv_ula.yaml --runs outputs/SV_ULA --combiner random:11:64 --out results/SV_ULA_newA_M64
%run scripts/evaluate.py --scenario configs/scenarios/deepmimo_ula.yaml --runs outputs/SV_ULA --graph-phase target --out results/SV_on_DeepMIMO_ULA
```
- DeepMIMO's combiner is not SV's, so the third line changes the combiner and the channels at once.
- **Networks that can follow use the new combiner.** These are Graph-LADMM analytic and LADMM `operator=analytic`, and
  from v31 also NF-LADMM, LVAMP and ALISTA: nothing they learn depends on A, and `set_combiner` recomputes their A
  buffers.
- **The others keep theirs.** That is the transfer test of LADMM-F, CADMM, LAMP, GM-LAMP and the rest.
- **Another number of RF chains (v30):** `random:<seed>:<M>`, or a `.mat` file with an M × N combiner. The networks that
  follow work with any M. The others cannot run there and are skipped (`eval_log.json` says so).
- `--graph-phase target` gives the analytic Graph-LADMM the edge phases of the target data set, from its validation
  channels. They are the array's index convention, not training. Graph-LADMM with a free H_t keeps its own phases:
  its H_t was learned with them.

### 4.13 Many test combiners at once (v30): `check_combiners.py`
```
%run scripts/check_combiners.py --scenario configs/scenarios/sv_ula.yaml --runs outputs/SV_ULA --tags "G-LADMM*" "LADMM-T*" GM-LAMP-authors
```
**What it does.** Every network of the runs folder on nine test combiners it never saw in training: the scenario's own,
a new random ±1 draw, Gaussian, correlated rows (badly conditioned), partially connected, a DCT row subset, and random
±1 with M = 1.25, 0.75 and 0.5 times the scenario's M.
- The combiners are built with fixed seeds, so every PC tests the same ones.
- `--combiners own pm1 dct M--` takes a subset; `--combiner-files A1.mat A2.mat` adds your own.
- `--graph-phase target` is for networks that come from another data set.

**What it writes** (`results/combiner_check_<SCENARIO>_<runs folder>/`):
- `combiner_check.md` / `.csv` / `.png`: mean NMSE, network × combiner, and the change against the scenario's combiner;
- `combiners.csv`: M, cond(AAᵀ) and the coherence of each combiner;
- one `evaluate.py` folder per combiner, with every SNR.

**Measured on SV ULA** (18 Oct; mean NMSE over 0–20 dB):
- **The 54-parameter network trained 9,000 steps** stays within 0.12 dB on every random, Gaussian or correlated
  combiner. It loses 0.66 dB on the partially connected combiner and 1.21 dB on the DCT one. At M = 160 / 96 / 64 it
  gives −17.51 / −14.47 / −8.37 dB, against −10.98 / −9.07 / −6.72 dB for AMP.
- **With v31's `set_combiner`, NF-LADMM-SR (3,350 parameters) transfers as well** (review of 18 Oct):
  - ±0.13 dB on the random combiners, −17.69 dB at M = 160;
  - but −5.1 dB on the partially connected combiner and −3.6 dB on the DCT one.
  - LVAMP and ALISTA follow too; ALISTA diverges at M = 96 and 64.
- **So the claim to make** is that the analytic-data-step methods transfer, and that Graph-LADMM is the accurate one that
  stays within 1.4 dB of its own-combiner NMSE on every structured and badly conditioned combiner of the same size. Not
  "only Graph-LADMM transfers".
- **The authors' GM-LAMP** is above 0 dB on every new combiner, and cannot run with another M.
- **v31 robustness:** a diverged network (NaN) is printed as "diverged" instead of stopping the check, and a combiner
  on which no method can run is noted and skipped.

### 4.14 Graph-reweighted l1: classical and unrolled (v31)
The uploaded `graph_reweighted_l1` package is iteratively reweighted l1 on the beam graph:
- the weights are w_i = 1/(a_i + ε), normalised to mean 1, where a_i is the pooled amplitude of the beam and its
  neighbours (UPA: 4 on the torus; ULA: ±1, ±2) in the previous estimate, divided by its maximum;
- the solver is 3 rounds of 300 ADMM iterations, with λ and ε tuned per SNR.

bsce has it both ways.

**Classical, as a baseline:**
```
%run scripts/tune_classical.py --scenario configs/scenarios/sv_upa.yaml
%run scripts/compare.py --scenarios sv_upa --preset onenet --classical amp,l1,grwl1 --name upa_with_classical
```
`tune_classical.py` chooses λ (and ε) per test SNR on 300 validation channels, with the true SNR, as the package does.
It writes `results/classical_params_<SCENARIO>.json`, and compare.py / evaluate.py (`--classical-params`) read it.

**Unrolled** (`gladmmrw`: the package's data step, no TV; `gladmmrwtv`: in `gladmm`'s layer):
```
%run scripts/run_suite.py --scenarios sv_upa --methods gladmmrw gladmmrwtv gladmm --set train.steps=4500 --device cuda
```
Each layer is one reweighting round, with the weights from the previous layer's output. ρ, k1, ε, the pooling weights
and the dual step are learned per layer, and the noise level comes from the residual, so no SNR is given.

**Measured on SV UPA** (18 Oct, 4,500 steps, the same test channels and noise for all; mean NMSE 0–20 dB):

| method | trained parameters | mean | vs GM-LAMP |
|---|---|---|---|
| `gladmm` | 54 | −11.70 | −3.48 |
| `gladmmrwtv` | 51 | −11.22 | −3.96 |
| `gladmmrw` (faithful unrolling) | 44 | −10.37 | −4.81 |
| graph-reweighted l1, classical, tuned | – | −9.29 | −5.88 |
| l1 ADMM, classical, tuned | – | −7.90 | −7.27 |

- Unrolling gains 1.08 dB over the classical method.
- The energy gate is 0.5 dB better than the reweighted threshold, with or without TV.
- TV is worth about 1 dB on UPA.

### 4.15 Training over a mixture of combiners (v31)
`train.combiner: mix` draws a new combiner every `combiner_every` minibatches:
- **its size** from `train.combiner_M`;
- **its family** from `train.combiner_types`: `pm1` random ±1, `gauss`, `sparse` (16 beams per RF chain, partially
  connected), `dct` (random rows of the orthonormal DCT-II).

It is for networks with `set_combiner` (any M from v31). `gladmmmix` uses M ∈ {64, 96, 128, 160} and all four
families. Validation and the saved network use the scenario's combiner.

```
%run scripts/run_suite.py --scenarios sv_ula --methods gladmmmix --device cuda
%run scripts/train.py --scenario sv_ula --train configs/train/nfladmm_sr.yaml --protocol paper --snr 0 20 --set train.noise=online train.combiner=mix "train.combiner_M=[64,128,160]" "train.combiner_types=[pm1,sparse,dct]"
```
Check it with `check_combiners.py`. Its structured test combiners are then of families seen in training (other draws);
say so when you report it.

## 5. Compare all methods (fair comparison) and save everything
```
%run scripts/compare.py --scenarios all --preset ft --name final_finetuned
%run scripts/compare.py --scenarios all --preset onenet --name final_onenet
%run scripts/compare.py --scenarios all --name everything
```
- **`--preset`** (v30) chooses the networks as the workbench does:
  - `ft` selects every fine-tuned set (`*_ft_*`, and `*-ft` for the fine-tunes named before v24, or rename them: 4.1b).
  - `onenet` selects every one-network-0–20 dB run, whatever its suffix (`G-LADMM-T8_paper`, `LADMM-T6_fair`), without
    fine-tunes, other training ranges and ensembles. Before v30, `--tags "*-T[0-9]" "*-T[0-9][0-9]"` missed the
    suffixed runs.
- **`--tags`** chooses the networks by name or pattern, and adds to `--preset`. Without either, every folder is compared.
- **Always included:** the authors' GM-LAMP and LAMP (the reference) and AMP (`--classical amp,omp` adds OMP).
  - v31: `--classical amp,l1,grwl1` adds the tuned l1 ADMM and graph-reweighted l1. Run `tune_classical.py` first
    (§4.14).
- **Cost (v31):** every table also gives:
  - the **trained parameters** of one network: the entries whose gradient is not zero on two random probe batches. A
    parameter that never reaches the output is not counted (e.g. LISTA's W_1, which multiplies x₀ = 0; CADMM's first
    M2; the last dual step of an unrolled ADMM), nor an entry stuck at a saturated nonlinearity at the trained point;
  - the **real MACs per channel**: products and factorisations, complex × complex = 4, counted at batch 1 on the
    second call, so a cache built once is not counted.

  `--no-cost` leaves both out.
- **The fairness of the comparison:**
  - every method is tested on the same 2000 test channels, with the same noise at every SNR;
  - every method is trained on the same data with the same protocol and the same fine-tune. The exception is the
    analytic Graph-LADMM rows, which keep their own end-to-end training (4.12) unless `--force-protocol` is given:
    - `networks.md` gives every run's protocol, so say it in the paper's table;
    - give NF-LADMM-SR, LISTA and LAMP the same protocol (`--protocol paper`) when you compare at the same budget;
    - read a short common budget as training efficiency, not capability. For example, LAMP gives −12.94 dB after
      3,000 end-to-end steps, against −15.85 dB with the authors' training;
  - every table lists the number of parameters next to the NMSE.
- **What is written to `results/<name>/`:**

| file | content |
|---|---|
| `comparison.png` / `.pdf` | one column per scenario: NMSE vs SNR; a table (0 / 10 / 20 dB, mean, gain over GM-LAMP); the gain over GM-LAMP at every SNR (values printed for the best method) |
| `summary.csv` | every scenario × method: NMSE at every SNR, mean, gain, parameters, inference time, trained parameters, MACs per channel (v31) |
| `summary.md` / `summary.tex` | the paper table (0 / 10 / 20 dB, mean, gain, parameters) |
| `networks.md` | what each compared network is: method, T, protocol, data, bands; or the fine-tune source and budget |
| `<SCENARIO>/` | per-scenario evaluation: `nmse_vs_snr.csv` / `.png`, `nmse_table.png` (every value printed), `eval_log.json` |
| `README.md` | the command that made the folder and what each file is |

- **A different reference for the gains:** `--ref GM-LAMP-T8_ft_0-10-20dB_20k`, for example, compares against GM-LAMP retrained with the same protocol.
- **Polish (v28, off by default):** `--polish band` adds, for every network, the column `<run>+polish`: the network followed by a least-squares correction on its K strongest beams. K and ε are chosen on the validation channels for each network band, so no retraining and no test data are used. `--polish snr` chooses them per test SNR (the SNR is then assumed known; say so in the paper). The choices are in `<SCENARIO>/polish_choices.csv`.

---

## 6. The folder layout
```
bsce/
  GUIDE.md                      this guide (the workflow)
  MANUAL.md                     every option of every script (reference)
  DATA_GUIDE.md                 the data in detail
  configs/scenarios/*.yaml      data sets (sv_ula, sv_upa, deepmimo_ula, deepmimo_upa, ...)
  configs/train/*.yaml          methods (<method>.yaml) and protocols (<name>_protocol.yaml)
  scripts/                      the commands (main ones below)
  bsce/                         the library (models, training, data, export)
  data/<SCENARIO>/              channels and measurement files
  reference/                    GM-LAMP's matrices, test sets, trained networks; original codes
  outputs/<SCENARIO>/<network>/ trained networks: .pt (PyTorch), .mat (MATLAB), _orig.mat (original MATLAB test code),
                                train_log.json / finetune_log.json, README.txt, layers/ (with --save-layers), checkpoints/
  outputs/<SCENARIO>_white/     the same, trained with white noise (--noise-model white, 4.10)
  results/<name>/               comparisons, depth curves: figures, tables, README.md
  tests/                        verification tests (python -m pytest tests)
  docs/                         CLI reference, change notes, WORKBENCH.md
  app/                          the workbench (bsce_app.py + static/); runs/ keeps its run logs
  start_app.bat                 double-click: opens the workbench
```

**The main scripts, in workflow order:**

| script | purpose |
|---|---|
| `check_env.py` | environment check |
| `make_gmlamp_data.py` | the four GM-LAMP data sets |
| `run_suite.py` | train and fine-tune any methods, scenarios, protocols and ranges |
| `depth_snapshots.py` | depth curves on the test set |
| `promote_layer.py` | use the network after layer T as a T-layer run |
| `check_vs_authors.py` | data and retrained LAMP / GM-LAMP against the authors' files and training logs |
| `compare.py` | evaluation + figures + tables of any set of networks |
| `compare_noise.py` | colored against white noise: every network on both, with cross tests (4.10) |

Lower-level tools that the main scripts call (each with `--help`):
- `train.py`, `finetune_bands.py`, `evaluate.py`, `import_gmlamp_weights.py`, `make_dataset.py`, `make_figures.py`.

**Names inside `outputs/<SCENARIO>/`:**
- `<METHOD>-T<T>`: one network for 0–20 dB, trained with `ladmm_best`.
- `<…>_ft_<edges>dB_<iterations>`: the same network fine-tuned per band, e.g. `LADMM-T6_ft_0-10-20dB_20k` (before v24: `<…>-ft`).
- Suffixes, added only when the setting differs from the default:
  - `_<protocol>`;
  - `_tr<edges>` (training range);
  - `_<override>` (from `--set`);
  - `_s<seed>`;
  - `_ft_<edges>dB_<k>` (fine-tune bands and iterations).
- `GM-LAMP-authors` and `LAMP-authors`: the published networks.

---

## 7. Times on your Quadro P2000 (measured)
| run | time |
|---|---|
| LADMM T = 6, one network 0–20 dB (`ladmm_best`) | 3.9–4.9 h |
| LADMM T = 4 | 2.9–3.1 h |
| fine-tune, 2 bands × 15k iterations (T = 6) | ≈ 35 min |

- **Depth:** a layer-wise run costs about T² / 2 layer-trainings, so T = 12 takes about 3–4 times as long as T = 6.
- **Other methods:** GM-LAMP, ADMM-Net (T = 15) and LISTA/ALISTA (T = 16) have more stages, so they take longer than LADMM. Start them first, or on the second computer.

## 8. Problems
| message | fix |
|---|---|
| the workbench shows old steps, old runs, or an older folder in Settings | an older workbench (another bsce folder) is still running on the same port. Close its black window, or use "Stop it" in the yellow note of the new page (from v21; `docs/WORKBENCH.md`, "An older workbench is still open") |
| `missing … reference\gmlamp\…` | copy `reference\` from the previous bsce folder (section 0.1) |
| `missing …\data\<SCENARIO>\…_training_channel.mat` | step 1, or copy `data\` |
| a run stopped (power, Spyder closed) | run the same command again: it resumes from its last checkpoint (inside a stage at most 10 minutes are lost; a fine-tune keeps its finished bands). `%run scripts/runs.py` shows where (4.11) |
| `… holds a fine-tune made with other settings or from another version of the source` | that folder was fine-tuned with other options, or its source network was retrained since. Give `--tag`, or move the folder away |
| `nmse is NaN` | report it with the console text. GM-LAMP's float32 underflow is fixed. |
| results differ slightly between two GPUs | expected (section 4.4); train one scenario on one kind of GPU |

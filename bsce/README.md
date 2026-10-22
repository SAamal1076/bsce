# bsce — learned beamspace channel estimation in PyTorch, verified against the original code

`bsce` re-implements, in PyTorch, the data generation and every learned estimator used in the
LADMM project — LAMP, GM-LAMP (Wei, Hu & Dai), the LADMM family (your scripts plus the three
proposed exact-ADMM variants), LISTA, ALISTA, LVAMP and ADMM-Net — together with the classical
baselines (AMP, OMP, SD, ADMM). Each port is checked against the original MATLAB / TensorFlow /
PyTorch code: the original files are run unchanged (MATLAB code in GNU Octave, TF-1 graphs
through a `tf.compat.v1` shim) on the same inputs, and the outputs, gradients, optimiser updates
and whole training loops are compared. The numbers are in `VERIFICATION_REPORT.md` and
`tests/results/`.

* One config-driven pipeline: `generate_data.py` → `train.py` → `evaluate.py`.
* `--device auto` (default) uses the GPU when one is visible and falls back to the CPU.
* Every trained model is saved three ways: `.pt` (PyTorch), `.mat` (MATLAB, read by
  `matlab/bsce_infer.m`) and, for LAMP / GM-LAMP / LADMM, `_orig.mat`, a drop-in replacement
  for the original weight files (`LAMP.m`, `GM_LAMP.m`, `LADMM_test.m` load it unchanged).


**Start here: [GUIDE.md](GUIDE.md)**: the final workflow, step by step, with the Spyder commands. It covers:
- data;
- the paper check;
- the number of layers;
- training every method with the same protocol;
- the fair comparison.

Reference documents:
- [CODE_MAP.md](CODE_MAP.md): what each part of the code does and where every option is handled (v31.3).
- [MANUAL.md](MANUAL.md): every script, option, method, schedule and protocol in detail.
- [DATA_GUIDE.md](DATA_GUIDE.md): the GM-LAMP data sets in detail.
- [docs/3GPP_DATASET.md](docs/3GPP_DATASET.md): the 3GPP TR 38.901 data sets (sizes, configuration, checks).
- [docs/CLI_REFERENCE.md](docs/CLI_REFERENCE.md): the `--help` of every command.
- [RUNBOOK.md](RUNBOOK.md): the earlier experiment plan, kept for the record.
- `docs/changes/`: change notes. **v31.1:** GM-LAMP networks trained with bsce before v31.1 from the original init
  have a one-Gaussian denoiser and must be retrained (`CHANGES_2026-10-18_v31_1_gmlamp_fix.md`, `scripts/check_gm_components.py`).
  **v31.2:** `run_suite.py --save-layers` runs a list that mixes layer-wise and end-to-end methods
  (`CHANGES_2026-10-19_v31_2_save_layers_mixed.md`).
  **v31.3:** two code reviews (`docs/REVIEW_2026-10-21.md`), the new methods `ladmmat`, `ladmmatr`, `ladmmatrd`
  (noise-adaptive threshold, residual noise level, debias layer) and their pilot results
  (`CHANGES_2026-10-22_v31_3.md`).

## 1. Install (conda; Windows commands, the same work on Linux/macOS)

The package comes as five zips. Extract all of them into the same folder; every path starts with `bsce/`.
- `bsce_code.zip`: code, configs, documents, your LADMM checkpoint in `models/`.
- `bsce_reference_part1.zip` … `part3.zip`: GM-LAMP's CS matrices, test files and trained weights. The data and comparison scripts need them.
- `bsce_reference_part4.zip`: the original code of LADMM, Borgerding, ALISTA and THADMM-Net. Only the verification tests need it.

In the **Anaconda Prompt** (or Anaconda PowerShell Prompt), from that folder:
```bat
conda env create -f environment.yml
conda activate bsce
:: PyTorch with CUDA: pip only (pytorch.org no longer lists conda). Pick the CUDA version that
:: your driver supports (nvidia-smi, top right) in the selector at https://pytorch.org/get-started/locally/
:: (OS Windows, package Pip); for example CUDA 12.6:
pip install torch --index-url https://download.pytorch.org/whl/cu126
python -c "import torch; print(torch.__version__, torch.cuda.is_available(), torch.cuda.get_device_name(0))"
```
The last line must print `True` and the name of your GPU. Do not `pip install torch` without
`--index-url`: on Windows the default wheel is CPU-only. Without conda:
`pip install -r requirements.txt` after the PyTorch line above.

Only if you also want to re-run the comparisons with the **original** code on that machine:
```bat
pip install -r requirements-verify.txt
```
and install GNU Octave for Windows (https://octave.org/download). Either add its `mingw64\bin`
folder to PATH or tell the tests where `octave-cli.exe` is:
```bat
:: Anaconda Prompt (cmd)
set "BSCE_OCTAVE=C:\Program Files\GNU Octave\Octave-9.2.0\mingw64\bin\octave-cli.exe"
```
```powershell
# Anaconda PowerShell Prompt
$env:BSCE_OCTAVE = "C:\Program Files\GNU Octave\Octave-9.2.0\mingw64\bin\octave-cli.exe"
```
(adapt the version folder). `set` / `$env:` last for the current window only.

## 2. Quick start

**The workbench** (every step as a form, the commands shown live, a run queue with progress): double-click
`start_app.bat`, or run `python app\bsce_app.py` in the Anaconda Prompt. It runs with the base Anaconda Python, so it
can also create the environment of section 1 (step 0). See `docs/WORKBENCH.md`.

All commands are run from the package folder with the `bsce` environment active; they are the
same in cmd, PowerShell and Linux shells.

**1 - check the GPU before training** (PyTorch only, no Octave/TensorFlow needed):
```bat
python -m pytest tests/test_gpu_vs_cpu.py -q
python -m pytest tests/test_golden.py -q
python scripts/summarize_tests.py
```
`test_gpu_vs_cpu.py` runs every model on the CPU and the GPU with the same weights, inputs and
minibatches (outputs, gradients, Adam steps, a short training, export); the CPU path is
verified against the original code by the rest of the suite, so together they bound the
GPU-vs-original difference. `test_golden.py` compares short trainings with the stored numbers
in `tests/golden/golden.json`. Results go to `tests/results/summary.md`. TF32 is disabled on
CUDA (`bsce.device.get_device`), so float32 matmuls keep full single precision.

**2 - pipeline check on tiny data** (a few minutes per method):
```bat
python scripts/run_all.py --scenario configs/scenarios/sv_ula.yaml --smoke
```

**3 - full runs** (GPU chosen automatically; `--device cpu` forces the CPU):
```bat
python scripts/run_all.py --scenario configs/scenarios/sv_ula.yaml
python scripts/run_all.py --scenario configs/scenarios/sv_upa.yaml --protocol fair --configs gmlamp_spread ladmm_original ladmm_kronprior lista
```
`run_all.py` generates the data, trains each config (all 14 by default, or `--configs ...`),
evaluates, and writes every step's console output to `outputs\<scenario>\logs\`. The single
steps can also be run by hand:
```bat
python scripts/generate_data.py --scenario configs/scenarios/sv_ula.yaml
python scripts/train.py --scenario configs/scenarios/sv_ula.yaml --train configs/train/gmlamp_spread.yaml
python scripts/evaluate.py --scenario configs/scenarios/sv_ula.yaml --runs outputs/SV_ULA --classical amp,omp,sd
```
`evaluate.py` writes `nmse_vs_snr.csv`, `nmse_vs_snr.png` (curves, values printed) and
`nmse_table.png` (methods × SNR, every value printed). An interrupted training resumes from
its checkpoint when restarted with the same scenario, band and config; any change of the
config starts from scratch.

## 3. Data (`configs/scenarios/*.yaml`)

| scenario | array | notes |
|---|---|---|
| `sv_ula.yaml` / `sv_upa.yaml` | ULA 256 / UPA 16×16 | GM-LAMP setting: N=256, N_RF=16, Q=8 → M=128, L=3 paths, bands 0–10 and 10–20 dB (validated at 10 dB) |
| `sv_*_4bins.yaml` | ULA / UPA | your LADMM setting: four 5-dB bands (0–5 … 15–20 dB) |
| `sv_*_single.yaml` | ULA / UPA | one network for 0–20 dB |
| `deepmimo_ula.yaml` / `deepmimo_upa.yaml` | ULA / UPA | DeepMIMO O1-28 (GM-LAMP setting). Raw DeepMIMO channels → `scripts/deepmimo_convert.py` (beamspace, ‖x‖² = N per channel as in GM-LAMP's files, 50k/2k/2k split, statistics vs GM-LAMP's data) → `generate_data.py --measurements-only` |
| `3gpp_upa.yaml` / `3gpp_ula.yaml` (`3gpp_ulaz.yaml`) | UPA 16×16 / ULA 256 on y (on z) | 3GPP TR 38.901 UMi street canyon, 28 GHz, configured like the DeepMIMO data (lens U0, ‖x‖² = N, CS matrix of SV, the authors' bands), 80k/2k/2k by default. `%run scripts/make_3gpp_data.py` (Python only, about 2 min per set; `--sizes dong2019` etc.); see [docs/3GPP_DATASET.md](docs/3GPP_DATASET.md) |

* `bsce.data.sv` ports `generate_channel.m` (ULA steering `exp(+j…)`, UPA `exp(−j…)`, the
  draw order of `normrnd`/`rand` per channel is kept), `bsce.data.sensing` ports
  `generate_CS_matrix.m`, `bsce.data.measurements` ports `generate_measurement_data.py`
  (`genMeasurements`, random SNR per chunk) and the MATLAB test-time noise of `Main.m`.
* Random numbers come from `bsce.rng.NumpyMatlabRNG` (a seeded generator with MATLAB's
  column-major fill). Given the same uniform/normal streams, the Python generator reproduces
  the MATLAB scripts exactly (tests replay the streams into Octave).
* Files are named as in the originals: `SV_ULA_training_channel.mat` (`x`, N×K),
  `SV_ULA_training_128_measurements_0to10dB.mat` (`y`, M×K), `CSmatrix.mat` (`A`), …
**DeepMIMO data (O1_28, DeepMIMO v1.3 MATLAB generator) = GM-LAMP's DeepMIMO data.**
1. Unzip `DeepMIMO-matlab-1.3`; put the extracted ray-tracing folder at
   `DeepMIMO-matlab-1.3\RayTracing Scenarios\O1_28\` (folder name with the space, `O1_28.*.mat` directly inside).
2. MATLAB (any folder; the generator files are not modified):
   ```matlab
   addpath('<bsce>\matlab')
   dm = 'D:\...\DeepMIMO-matlab-1.3';
   deepmimo_make_H(dm, '<bsce>\H_ula.mat', 'ULA');   % 256 elements on the z axis (vertical)
   deepmimo_make_H(dm, '<bsce>\H_upa.mat', 'UPA');   % 1 x 16 x 16 on the y, z axes
   ```
   BS 3, rows 1000-1300 (54,481 users), 3 paths, 1 subcarrier, saved with `-v7`.
3. `python scripts/deepmimo_convert.py --scenario configs/scenarios/deepmimo_upa.yaml --input H_upa.mat`
   then `python scripts/generate_data.py --scenario configs/scenarios/deepmimo_upa.yaml --measurements-only`
   (same for ULA). The converter uses GM-LAMP's DeepMIMO beamspace convention
   x = U0^H h, U0(n,k) = exp(j2πn(k−(N−1)/2)/N)/√N (note: not the SV lens matrix), normalises
   ‖x‖² = N, looks up GM-LAMP's 2000 test channels among the users (must report ≥ 99 % found),
   uses GM-LAMP's test file as the test set and draws 50000 / 2000 training / validation channels
   from the other users. Verified for the UPA: 1991 / 2000 test channels found (median rel. error
   1e-5), and GM-LAMP's own trained LAMP / GM-LAMP give the same NMSE (within 0.1 dB, 0–20 dB) on
   our training users as on its test set.

* A new scenario = a new yaml: change `array`, `N1`, `N2`, `M`, `L`, `bands`, sizes, `seed`.

## 4. Models and training (`configs/train/*.yaml`)

| config | model | follows | precision of the original |
|---|---|---|---|
| `lamp_original.yaml` | LAMP (complex, widely-linear Onsager) | `Train_LAMP.py`, `define_networks.py` | float64 |
| `gmlamp_original.yaml` | GM-LAMP (K=4 Gaussian mixture) | `Train_GM-LAMP.py` | float64 |
| `gmlamp_spread.yaml` | GM-LAMP, initial mixture slightly spread (see §7) | — | float64 |
| `ladmm_original.yaml` | LADMM, full H_t | `LADMM_train.py` | float32 |
| `ladmm_diag.yaml`, `ladmm_lowrank.yaml`, `ladmm_kron.yaml` | LADMM operators | `A_*_DiagonalScaling.py`, `B_*_LowRank.py`, `C_*_Kronecker.py` | float64 |
| `ladmm_psd.yaml`, `ladmm_kronprior.yaml`, `ladmm_diagprior.yaml` | proposed exact-ADMM LADMM | new | — |
| `lista.yaml`, `alista.yaml` | LISTA (untied, support selection off), ALISTA | ALISTA repository | float32 |
| `lvamp.yaml` | LVAMP (Bernoulli-Gaussian) | Borgerding `LVAMP.py` | float32 |
| `admmnet.yaml` | ADMM-Net | THADMM-Net repository | float32 |

Training schedules (`bsce/train/trainer.py`):
* `borgerding` — the layer-wise schedule of `setup_training`/`do_training` (GM-LAMP, LADMM,
  LVAMP): each new layer's parameters alone at `lr`, then all parameters at `lr·{0.5,0.1,0.01}`;
  validation every `ivl` steps, early stop when the best validation NMSE is older than
  `better_wait`; `warm_start: true` reproduces GM-LAMP's copy of the previous layer's values;
  `val_round: 5` reproduces `round(nmse, 5)`.
* `alista` — `utils/train.py` of the ALISTA repository (per-variable learning-rate decay).
* `end2end` — the THADMM-Net epoch loop.
* `optimizer: adam_tf1` is TF-1's Adam exactly (update order and float32 hyper-parameters /
  beta-power accumulators); `adam_torch` is `torch.optim.Adam`.

Common options of `scripts/train.py`:
```
--device auto|cpu|cuda      --bands 0to10dB,10to20dB      --protocol fair
--set train.lr=5e-4 train.batch_size=512 model.T=10 model.precision=float32
```
`--protocol gmlamp` trains every method exactly as GM-LAMP/LAMP were trained
(`configs/train/gmlamp_protocol.yaml`: layer-wise with refinements 0.5/0.1/0.01, warm start of each
new layer from the previous one, Adam lr 1e-3, batch 128, validation every 10 steps,
`round(nmse, 5)`, early stop after 5000 steps without improvement); together with
`sv_ula.yaml` / `sv_upa.yaml` (GM-LAMP's `CSmatrix256128.mat`, 80k/2k/2k channels, bands 0–10 and
10–20 dB validated at 10 dB) this is GM-LAMP's complete setting. `--protocol fair` applies `configs/train/fair_protocol.yaml` (same optimiser, batch, iteration
budget and early stopping for every learned method). Training writes checkpoints after every
stage and resumes from them automatically. Outputs:
`outputs/<scenario>/<tag>/<tag>_<band>.{pt,mat}`, `<tag>_<band>_orig.mat`, `train_log.json`.

Speed note: float64 on consumer GPUs is slow; `--set model.precision=float32` gives the same
method in single precision (statistically, not bit-wise, equal to the float64 original).

## 5. Using trained models

Python:
```python
from bsce.export import load_model
net = load_model("outputs/SV_ULA/GM-LAMP/GM-LAMP_0to10dB.pt", device="cuda")   # or the .mat
xhat = net(torch.as_tensor(y))          # y: M x K complex
layers = net.forward_stages(torch.as_tensor(y))   # every intermediate estimate
```
MATLAB / Octave:
```matlab
addpath matlab
net  = bsce_load('outputs/SV_ULA/GM-LAMP/GM-LAMP_0to10dB.mat');
xhat = bsce_infer(net, y);          % same output as PyTorch (tested to ~1e-15)
```
or copy `<tag>_<band>_orig.mat` to the file name your original MATLAB script loads
(e.g. `Trained_GM_LAMP_for_SV_ULA_2561280to10dB.mat`) and run the original code unchanged.

## 6. Verification tests

```bat
python -m pytest tests -q
python -m pytest tests/test_lamp_vs_tf.py -q
python scripts/summarize_tests.py
```
The first line runs everything (about 1 h on 2 CPU cores; needs `requirements-verify.txt` and
Octave, otherwise those files are skipped), the second one family, the third writes
`tests/results/summary.{md,csv}` and `equivalence_summary.png`. Optional environment variables:
`BSCE_GPU_DEVICE` (device of `test_gpu_vs_cpu.py`, default `cuda`; `cpu` = dry run),
`BSCE_DEVICE` (device of `test_golden.py`, default auto), `BSCE_OCTAVE` (path of octave-cli),
`BSCE_REF_CACHE` (folder that caches the original training-loop runs).
| file | compares against |
|---|---|
| `test_data_vs_octave.py` | `generate_channel.m`, the full data scripts (80k / 220k channels), `generate_CS_matrix.m`, `Main.m` noise, `genMeasurements` |
| `test_lamp_vs_matlab.py` | `LAMP.m`, `GM_LAMP.m` with the authors' trained weights (SV & DeepMIMO, ULA & UPA, 8 SNRs) |
| `test_lamp_vs_tf.py` | `define_networks.py`: all layer outputs, stage list, gradients and Adam updates of every stage |
| `test_ladmm_vs_originals.py` | `LADMM_test.m` / `LADMM_UPA_test.m` with your trained weights; `ADMM_vectorized.m`; TF graphs of `LADMM_train.py` (float32) and the full/diag/low-rank/Kronecker scripts (float64) |
| `test_training_loop_vs_tf.py` | the original `do_training` loops end to end (recorded minibatches replayed), parameters after every stage |
| `test_ista_lvamp_vs_tf.py` | ALISTA repository (LISTA, ALISTA), Borgerding `LVAMP.py` (float32 and float64 copies) |
| `test_admmnet_vs_thadmm.py` | THADMM-Net repository (PyTorch) |
| `test_classical_vs_octave.py` | `AMP.m`, `OMP.m`, `SD.m`, `ADMM_vectorized.m` |
| `test_export_matlab.py` | `.pt`/`.mat` round trip and `matlab/bsce_infer.m` in Octave, all models |
| `test_export_to_original_matlab.py` | `_orig.mat` files run by the unchanged `LAMP.m`, `GM_LAMP.m`, `LADMM_test.m` |
| `test_proposed_ladmm.py` | proposed variants reduce to classical ADMM at initialisation; operators are SPD |
| `test_gpu_vs_cpu.py` | every model on GPU vs CPU: stage outputs, gradients, Adam steps, a short training run, export (skips without CUDA) |
| `test_golden.py` | short float64 trainings of 13 model configurations vs `tests/golden/golden.json` (no Octave/TF needed) |
| `diagnostics/gm_symmetry.py` | GM-LAMP round-off sensitivity experiment (report §7) |

Each test appends its numbers to `tests/results/equivalence.jsonl`. Interventions in the
original code are limited to: Octave shims for `rand`/`randn`/`normrnd` that replay a given
stream (`tests/harness/octave_shims`), the `tf1_shim` that maps `tensorflow` to
`tf.compat.v1`, recorded `np.random.choice` indices, and derived copies with Python-3 syntax
or float32→float64 (`tests/harness/make_reference_copies.py`, originals untouched).

## 7. Things the tests revealed (details in VERIFICATION_REPORT.md)

1. GM-LAMP's initial mixture has K identical components; the gradient separating them is zero
   in exact arithmetic, so in the original training the components split through round-off
   amplified by Adam. PyTorch follows the TF trajectory to ~1e-16 until the first warm-started
   GM layer is trained, and only statistically afterwards (the same would happen between TF on
   CPU and on GPU); in PyTorch on the CPU the components can even stay identical.
   `gmlamp_spread.yaml` removes this dependence and is the recommended GM-LAMP config for
   the experiments (outputs tagged `GM-LAMP-spread`).
2. TF-1's Adam keeps its hyper-parameters and beta powers in float32 even for float64 models;
   reproduced by `adam_tf1`.
3. The Kronecker script computes `(H2 ⊗ H1) x` (row-major reshape), not `H1 ⊗ H2` as its
   docstring says; the code is reproduced.
4. In the LADMM scripts `alpha_{T-1}` never receives a gradient (the last dual update does not
   reach the output), so it stays at 1.
5. `LADMM_UPA_test.m` sets `T` before `load(...)`; a weight file containing `T` silently
   overrides it (`_orig.mat` files contain no `T`).
6. In the DeepMIMO MATLAB evaluation `A` is stored in single precision, so MATLAB computes in
   single precision (≈1e-7 differences); the tests pass `A` as double.
7. `genMeasurements` leaves trailing all-zero columns when K is not divisible by the number
   of SNR chunks; `bsce` raises an error instead (use `strict=False` to reproduce).

## 8. Layout
See [CODE_MAP.md](CODE_MAP.md): the folders, every module, every model, the commands by step, and what a run leaves on disk.


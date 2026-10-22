# bsce code map: what is where, and how the pieces fit (v31.3)

The **GUIDE** says what to run, in order. This page says **what each part of the code does**, so you can read it,
change it, or find where an option is handled.

## 1. The pipeline in one picture

```
 data set (channels + measurements)             a trained network per band                 results
 ───────────────────────────────────             ───────────────────────────                 ───────
 make_gmlamp_data.py / make_dataset.py   ──►   train.py  (one method, one scenario)   ──►   evaluate.py  (one scenario)
   bsce/data/sv.py, measurements.py             bsce/models/*  +  bsce/train/trainer.py      bsce/classical.py, bsce/polish.py
   data/<SCENARIO>/*.mat                         outputs/<SCENARIO>/<run>/<run>_<band>.pt     results/<name>/<SCENARIO>/…
                                                         │
                                                         ▼
                                                 finetune_bands.py  (per SNR band, next to the run)
                                                   outputs/<SCENARIO>/<run>_ft_0-10-20dB_20k/

 run_suite.py  = train.py + finetune_bands.py for many methods × scenarios, with ONE protocol (the paper workflow)
 compare.py    = evaluate.py for many scenarios + the paper figures and tables (results/<name>/)
 app/bsce_app.py = the workbench: every step above as a form; it only writes these commands and runs them
```

## 2. Folders

| folder | what is in it |
|---|---|
| `bsce/` | the library (no command-line code): data, models, training loop, export, polish |
| `scripts/` | the commands (each one has `--help`; `docs/CLI_REFERENCE.md` lists them all) |
| `configs/scenarios/` | one yaml per data set: array, sizes, SNR bands, test SNRs, data folder |
| `configs/train/` | one yaml per method (`<method>.yaml`) and per protocol (`<name>_protocol.yaml`) |
| `app/` | the workbench (`bsce_app.py` server + `static/` page); not in the Spyder edition |
| `matlab/` | MATLAB / Octave inference of the exported `.mat` networks, and the DeepMIMO / QuaDRiGa channel makers |
| `tests/` | the checks (`pytest`), including the comparisons with the original code |
| `docs/` | the notes on data, noise models, UTRR, the workbench, and `changes/` (one note per version) |
| `reference/` | the original code and files (separate zips); `reference/user/untitled2.py` is your script (v28) |
| `data/`, `outputs/`, `results/` | made by the commands: data sets, trained networks, comparisons |

## 3. The library `bsce/`

| module | what it does |
|---|---|
| `config.py` | reads the yaml files, applies `--set key=value`, finds the data and output folders of a scenario |
| `naming.py` | the folder names of every run and fine-tune (one scheme for `train.py`, `run_suite.py`, `finetune_bands.py`) |
| `rng.py` | random streams identical to MATLAB's, so the data equal the authors' |
| `data/sv.py`, `sensing.py`, `tr38901.py` | SV channels, the CS (combiner) matrix, the built-in 3GPP generator |
| `data/measurements.py`, `noise.py`, `files.py` | measurements y = A(x+n) or y = Ax + w, the SNR draws (chunks or fair), the band files |
| `models/__init__.py` | the **registry**: model name in the yaml → class and its options |
| `models/common.py` | `UnrolledNet`, the base of every network: parameters in the original order, `stage_plan()` (the layer-wise training plan), `forward_stages()` (every intermediate output) |
| `models/shrinkage.py` | every denoiser: soft thresholds, GM-LAMP's Gaussian mixture, Bernoulli–Gaussian, and (v28) `firm_shrink`, `gm_mmse_shrink`, `beam_noise_level` |
| `train/trainer.py` | the training loop: layer-wise stages (`borgerding`), end to end (`steps`, `end2end`), validation, early stop, checkpoints and resume; (v29) `combiner: random`; (v31) `combiner: mix` (`make_train_combiner`: sizes `combiner_M`, families `combiner_types`) |
| `train/optim.py`, `losses.py` | TF-1 Adam (bit-exact with the original scripts), the losses |
| `export.py` | save / load `.pt` + `.mat` (+ `_orig.mat` for the original MATLAB code), cut a network to its first T layers |
| `classical.py` | AMP, OMP, SD, ADMM-LASSO; (v31) `l1_admm`, `grwl1` (graph-reweighted l1 of the uploaded package), `graph_weights`, and `METHODS` (the names every script treats as classical) |
| `cost.py` | (v31) `trained_parameters` (the parameters with a non-zero gradient on two probe batches) and `macs_per_channel` (an ATen dispatch counter of products and factorisations); compare.py's two cost columns |
| `polish.py` | (v28) least squares on the K strongest beams after a network; K and ε chosen on validation |

### The models (`bsce/models/`)

| file | class | method key | what it is | parameters (N = 256) |
|---|---|---|---|---|
| `lamp.py` | `LAMP` | `lamp`, `gmlamp` | LAMP and GM-LAMP (Wei–Hu–Dai), faithful to the authors' TF code. v31.1 `break_gm_symmetry`: identical mixture components get the `spread` offset before training (the trainer's `gm_symmetry`) | 262,160 / 262,240 (T = 8) |
| `ladmm.py` | `LADMM` | `ladmm` | **the proposed LADMM** and its variants (table below) | 393,234 (T = 6, full) |
| `graph_ladmm.py` | `GraphLADMM` | `gladmm`, `gladmmf`, `gladmmany` | (v29) **proposed Graph-LADMM**: energy gate on the beam graph (+ magnetic graph TV for the analytic form); analytic x-step (any combiner) or free H_t | 56 (T = 8, analytic) / 393,252 (T = 6, free) |
| `cadmm.py` | `CADMM` | `cadmm` | (v28) complex ADMM-Net with untied M1, M2 (Johnston et al., IEEE TSP 2021) | 589,842 (T = 6) |
| `nfladmm.py` | `NFLADMM` | `nfladmm`, `nfladmmc`, `nfladmmsr` | proposed NF-LADMM: exact noise-folded step + CNN / radial prior | 9,066 / 160 / 8,982 (UPA) |
| `lista.py` | `LISTA`, `ALISTA` | `lista`, `alista` | complex LISTA, ALISTA | 1,081,360 / 32 |
| `lvamp.py` | `LVAMP` | `lvamp` | Borgerding's LVAMP | 13 |
| `admmnet.py` | `ADMMNet` | `admmnet` | ADMM-Net (Klioui, THADMM-Net loop) | 1,966,110 |
| `utrr.py` | `UTRR` | `utrr` | UTRR (Wu–Cheng–Eldar–Cioffi, TCOM 2025) | 1,966,140 |
| `ensemble.py` | `Ensemble` | (`make_ensemble.py`, `run_suite --ensemble`) | mean of several trained networks | sum of the members |

### LADMM, layer by layer (`ladmm.py`)

```
x_t = H_t (b + rho_t z_{t-1} - u_{t-1})       b = A^T y (data_step plain) or A^+ y (whitened)      _apply_op()
z_t = D_t (x_t + u_{t-1} / rho_t)             D_t: the threshold option                            _shrink()
u_t = u_{t-1} + alpha_t rho_t (x_t - z_t)                                                          alpha()
output z_T                                    forward_stages() returns x_1, z_1, x_2, z_2, ...
```

Every option is a `--set model.<key>=<value>`, or a method row's own settings in the workbench.
The defaults (in bold) are your original scripts.

| key | values | where in the code |
|---|---|---|
| `operator` (H_t) | **`full`**, `diag`, `lowrank`, `kron` (your A/B/C scripts); `psd`, `kronprior`, `diagprior` (exact ADMM with a learned prior); v28: `analytic` (LADMM-A, 3 parameters per layer), `analytic1` (α = 1) | `__init__` (parameters), `_apply_op`, `H_matrix` |
| `threshold` (D_t) | **`scalar`** (soft, θ_t/ρ_t); `group`, `group_k`, `group_kc`; v28: `adaptive`, `firm`, `gm` (blind noise level σ̂ = median\|w\|/√ln 2) | `_shrink`, `_group_weights`, `shrinkage.py` |
| `data_step` (b) | **`plain`** (Aᵀy, exact for white noise); v28: `whitened` (A⁺y, exact for y = A(x+n); AᵀA → P everywhere) | `__init__` (buffers `Apinv`, `AtA`), `forward_stages` |
| `noise_level` (σ̂ of adaptive / firm / gm) | **`median`** (median\|V\|/√ln2, blind); v31.3: `residual` (‖W(y − A z_{t−1})‖/√M, W = (AAᵀ)^−1/2; layer 1: median) | `_residual_level`, `_shrink`, `forward_stages` |
| `debias` (after z_T) | **`none`**; v31.3: `ls` (least squares on a soft support of z_T, 3 parameters κ, τ, ε ≥ 1e-4; one M×M solve per channel; trained in a last stage; Python only) | `_debias`, `stage_plan`, `tail_outputs` |
| `T`, `untied`, `rank`, `N1`, `N2`, `gm_K`, `precision`, `init_precision` | depth, tying, low-rank rank, UPA grid, GM components, float32/64 | `__init__` |

**The three v31.3 methods** (`run_suite.py`; full operator and the training of `ladmm`; not in `all`, name them):
`ladmmat` = LADMM-AT (`threshold: adaptive`), `ladmmatr` = LADMM-ATR (+ `noise_level: residual`), `ladmmatrd` = LADMM-ATRD
(+ `debias: ls`); configs `configs/train/ladmm_at*.yaml`. With `--save-layers`, the snapshots of layers t < T are z_t without
the debias layer; the T-layer snapshot (the full network) is saved after the debias stage. MATLAB (`bsce_ladmm.m`) refuses
these networks, and every other non-default LADMM option, instead of computing a wrong estimate.

### Graph-LADMM, layer by layer (`graph_ladmm.py`, v29)

```
x_t = H_t (A^+ y + rho_t (z1 - u1) + rho_t D^H (z2 - u2))   H_t = (P + rho_t (I + D^H D))^-1 (analytic) or free   forward_stages
z1  = D_t(x_t + u1)        zstep: l1 | gm | gate | gsoft  (+ TV for gtv, gtv_gate, gtv_gm)                        forward_stages
z2  = soft(D x_t + u2, k2_t s)     graph TV on the edges (D x)_e = x_j - e^{j phi} x_i                             Dop, DHop
u1 += e_t (x_t - z1),  u2 += e_t (D x_t - z2)
```
- The graph: `beam_edges` (ring / torus with wrap edges); the edge phases `estimate_edge_phases` (from the training
  channels, called by `train.py` through `init_from_data`).
- The noise level s: `sigma=residual` (from ‖y_w − A_w z‖²), `blind` (median), `known`.
- Combiner: `set_combiner(A)` (analytic only), with any number of RF chains M (v30: the buffers of another shape are
  registered again by `UnrolledNet._put_buffer`, so a saved and reloaded network has the new shape). The trainer's
  `train.combiner=random|mix`, `evaluate.py --combiner` and `check_combiners.py` use it.
  - The same method exists for LADMM's analytic operators, and (v31) for NF-LADMM, LVAMP and ALISTA
    (`UnrolledNet._combiner_array` checks the new A).
  - `evaluate.py` skips a network that cannot follow when the test combiner has another M.
- v31:
  - **Stored = trained.** `_dead_names()` lists the parameters that cannot reach the output; they are not created,
    and `_load_from_state_dict` drops them from older files. The last layer returns its z₁ before the tail.
  - **Inference cache.** `_cached_H()` keeps H_t per (combiner, edge phases, ρ_t) when no gradient is needed.
  - **Sparse G.** `_G()` is built from the edges.
  - **Options.** `zstep: rwl1 | gtv_rwl1` (`rw_weights`), `data_step: plain` (SVD form of (AᵀA + ρI)⁻¹),
    `beam_prior: gate` (`beam_ell`, `beam_beta_t`; profile in `init_from_data`).
- Protocol (v30): `run_suite.py` gives `gladmm` / `gladmmany` their own end-to-end training (`OWN_PROTOCOL`,
  `protocol_of`) unless `--force-protocol`; v31 adds `gladmmmix`, `gladmmbp`, `gladmmrw`, `gladmmrwtv` to it.

**Training of every layer-wise network** (`stage_plan()`): for layer t, the stage "H-update" trains H_t and ρ_t on x_t,
and the stage "Z-update" trains θ_t (and the threshold's own parameters) on z_t. After each stage come the refinements of
all parameters so far, with smaller learning rates. `trainer.py` runs these stages; `configs/train/*_protocol.yaml`
set the learning rate, batch and stopping.

## 4. The commands, by step of the GUIDE

| step | scripts |
|---|---|
| 0 check the machine | `check_env.py`, `speed_test.py` |
| 1 data | `make_gmlamp_data.py` (the 4 GM-LAMP sets), `make_3gpp_data.py`, `make_dataset.py` (any size), `make_scenario.py`, `generate_data.py`, `make_sv_measurements.py`, `deepmimo_convert.py`, `sionna_make_H.py` |
| 2 paper check | `import_gmlamp_weights.py`, `check_vs_authors.py`, `utrr_reproduce.py`, `check_gm_components.py` (v31.1: which GM-LAMP networks trained with a one-Gaussian denoiser) |
| 3 depth | `run_suite.py --save-layers`, `depth_snapshots.py`, `promote_layer.py`, `depth_curve.py`, `select_depth.py` |
| 4 training | `run_suite.py` (all methods), `train.py` (one), `finetune_bands.py`, `runs.py` (what is done), `rename_runs.py` |
| 5 comparison | `compare.py` (paper tables, `--polish`, `--preset ft` / `onenet`; v31 trained-parameter and MACs columns, `--no-cost`, `--classical-params`), `evaluate.py` (`--combiner`; v31 `--classical l1,grwl1`), `tune_classical.py` (v31), `check_combiners.py` (v30: nine test combiners, any M), `compare_noise.py`, `aggregate_seeds.py`, `make_ensemble.py`, `make_figures.py` |
| other | `import_user_ladmm.py` (your `.pt`), `compare_variants.py`, `run_all.py`, `check_exports.py`, `status.py`, `review_pack.py`, `summarize_tests.py`, `render_report.py`, `make_cli_reference.py` |

## 5. What a run leaves on disk

```
outputs/SV_ULA/LADMM-T6/                    trained: <run>_0to20dB.pt / .mat / _orig.mat, train_log.json, README.txt,
                                            checkpoints/ (resume), layers/ (with --save-layers)
outputs/SV_ULA/LADMM-T6_ft_0-10-20dB_20k/   fine-tuned: <run>_0to10dB.pt, <run>_10to20dB.pt, finetune_log.json
outputs/SV_ULA/ENS-LADMM-T6+LADMM-T8_…/     an ensemble (v28 --ensemble, or make_ensemble.py)
results/<name>/                             compare.py: comparison.png/.pdf, summary.csv/.md/.tex, networks.md,
                                            <SCENARIO>/nmse_vs_snr.csv/.png, nmse_table.png, polish_choices.csv
```

Names: `<METHOD>-T<T>[_<protocol>][_tr<edges>][_<option>][_s<seed>]`. Every `--set` option is added to the name,
e.g. `LADMM-T6_threshold-gm_data-step-whitened`. Fine-tunes add `_ft_<edges>dB_<iterations>`.

## 6. Tests

`python -m pytest tests -q` runs everything. The comparisons with the original code need the reference zips (and Octave /
TensorFlow for some); the rest needs only PyTorch. One file per topic: `test_ladmm_vs_originals.py`,
`test_lamp_vs_tf.py`, `test_ladmm_best_vs_untitled2.py` (bit-exact with your script), `test_resume.py`,
`test_snr_dist.py`, `test_v28_options.py`, `test_v29_graph_combiner.py`, `test_v30_combiners_names.py` (any-M combiner,
`check_combiners.py`, the suite's protocol for Graph-LADMM, unique names everywhere), `test_v31_rwl1.py` (the reweighted
l1 z-step and plain data step), `test_v31_release.py` (stored = trained, older files, cache, beam prior, set_combiner of
NF-LADMM / LVAMP / ALISTA, combiner mixture, classical l1 / grwl1, robust plots, cost, new keys; v31.1: GM-LAMP's identical
components and their separation in training), …

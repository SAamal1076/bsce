# 16 Oct 2026 (v22): UTRR, implemented from the paper and smoke-tested

**Why.** You gave the UTRR paper (Wu, Cheng, Eldar, Cioffi, IEEE TCOM 73(2):1128–1141, 2025) and asked for an implementation, smoke-tested before it is added.
- UTRR reports the best published NMSE on exactly the bsce setup: SV ULA, N = 256, M = 128.
- Its authors published no code.

**What.**
- **UTRR in bsce:**
  - the model, its own training (`--protocol paper`), and the model-averaging ensemble UTRR-E;
  - a script that reproduces the paper's Table II in the paper's own setting, before UTRR is compared in bsce's setting.
- **The checks of this session:** `docs/UTRR.md`, which also lists the points where the paper is ambiguous or wrong and what bsce does there.

No existing default, method, data set or folder name changes. `--methods all` stays the same set of methods: UTRR runs only when it is named (`--methods utrr`).

## Changes
| file | change |
|---|---|
| `bsce/models/utrr.py` (new) | **The UTRR network.** <br>• Layers: T unfolding layers (default 30), each with A_t ∈ ℝ^{M×2N}, ρ_t, α_t; top-K in the last layer only (RCC, default K = 16); β = 1/2. <br>• Inputs: real and imaginary parts as real samples. <br>• Sign: the gradient sign of the paper's eq. 16. <br>• Options: `K`, `rcc`, `bn` (the batch normalisation of Fig. 6, default off), `rho`, `alpha`, `bn_momentum`. <br>• `forward_real` gives the paper's real-valued network directly. <br>• Works with layer-wise training, `--save-layers` / `promote_layer.py`, the fine-tune per band, `.pt` / `.mat` export. |
| `bsce/models/ensemble.py` (new), `scripts/make_ensemble.py` (new) | **Model-averaging ensemble** (UTRR-E, Sec. IV-D): x̂ = mean of the members' outputs. <br>• `make_ensemble.py --runs <run folders> --tag <name>` writes one network folder per common band. <br>• `evaluate.py` and `compare.py` use it like any other network. |
| `configs/train/utrr.yaml` (new) | **The paper's own training:** <br>• end to end, Adam, loss E‖x − x̂‖²; <br>• learning rates 0.005 → 0.001 → 0.0005 → 0.0002 → 0.0001, early stopping after 10 epochs without improvement, at most 300 epochs; <br>• batch 128 real samples (64 complex columns), best weights kept. |
| `bsce/train/trainer.py`, `bsce/train/losses.py` | **New training options:** <br>• `end2end` + `lr_list`: one stage per learning rate, one epoch budget for all of them; <br>• `end2end` + `patience`: early stopping; <br>• loss `mse`. <br>Without these options `end2end` is unchanged (ADMM-Net); it now stops with an error on a NaN validation NMSE, as the other schedules do. |
| `scripts/run_suite.py` | Method key `utrr` (T = 30), not part of `all`. |
| `scripts/utrr_reproduce.py` (new) | **Table II in the paper's own setting** (noise after the combiner, SNR = ‖Φh‖²/‖n‖²), one model per SNR and top-K, then UTRR-E. <br>• A stopped run resumes. <br>• Writes `results/utrr_reproduce/` (json, .pt, table). |
| `tests/test_utrr.py` (new), `tests/proposals/utrr_ref.py` (new) | **9 tests.** <br>• Against the standalone reference, written from the paper alone: with and without BN; batch and running statistics. <br>• At init, the paper's iteration (NumPy). <br>• Parameter count L(2MN + 2). <br>• Truncation and save/load (BN statistics included); the stage plan. <br>• ITRR exact recovery, and divergence with the printed sign. <br>• Trainer `lr_list` / `patience` / `mse`; `end2end` unchanged for ADMM-Net. <br>• `run_suite`; the ensemble and `make_ensemble.py`. |
| `docs/UTRR.md` (new) | **The guide and the smoke tests:** <br>• sparse recovery of ITRR; <br>• network = equations; <br>• short trainings in the paper's setting; <br>• integration. |
| `GUIDE.md` 4.1 and 4.9, `MANUAL.md` 4 and A1–A2, `scripts/README.md`, `docs/CLI_REFERENCE.md`, `VERSION.txt` | Updated. GUIDE 4.1 now also lists `nfladmmsr`, which was missing from its table. |

## Verification
- **UTRR:** see `docs/UTRR.md` section 4.
- **The whole `tests/` folder on a clean copy:** 232 passed, 84 skipped, 0 failed (v21: 223 passed; the 9 new tests). Sionna and Octave + QuaDRiGa were installed, TensorFlow not.

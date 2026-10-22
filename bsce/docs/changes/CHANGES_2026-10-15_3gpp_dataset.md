# 15 Oct 2026: a third data set, 3GPP TR 38.901 (UMi street canyon, 28 GHz), configured like SV and DeepMIMO

**Why.** You asked how to generate 100,000 channels of the 3GPP standard with the same carrier, arrays and other
settings as the DeepMIMO and SV data, so that the comparison is fair.

**What is new.**
- Two generators that write the same raw files:
  - **Sionna**, which follows the TR procedure as written. It is the recommended one.
  - **QuaDRiGa** in MATLAB / Octave, with fixes for three deviations from 38.901 that were found here.
- The data set in bsce's format.
- A check against the standard's tables, against each other, and against SV and DeepMIMO.
- The guide: `docs/3GPP_DATASET.md`.

No existing default, data set or folder name changes.

## Changes
| file | change |
|---|---|
| `scripts/sionna_make_H.py` (new) | Raw 3GPP channels H (N × K) in part files of 5,000, made with Sionna 2.2 (`sionna.phy.channel.tr38901.UMi`, tables 38.901 v16.1 or v19.2). <br>• Model: UMi street canyon, with LOS/NLOS drawn from Table 7.4.2-1, at 28 GHz. <br>• Arrays: UPA 16×16 on y–z, ULA 256 on y or on z, λ/2. <br>• BS element of Table 7.3-1; one isotropic UE antenna. <br>• Users uniform in the 120° hexagonal sector cell (ISD 200 m, d2D ≥ 10 m); BS 10 m, UE 1.5 m; narrowband, plane waves. <br>• Seeded per drop, so any part can be made alone; it resumes after a stop and refuses to mix settings. <br>• Stores LOS state, position and the drawn large-scale parameters per user. <br>• It is stand-alone: it runs in its own `sionna` environment (PyTorch ≥ 2.9.1). |
| `matlab/quadriga_make_H.m` (new) | The same with QuaDRiGa v2.8.1, in MATLAB or Octave, in the same file format. It corrects three deviations of QuaDRiGa from 38.901, each an option that is on by default: <br>• `zsd_fix`: the 0.01·abs(h_UT − h_BS) term of the LOS ZSD mean; <br>• `czsd_fix`: the intra-cluster ZSD (3/8)·10^μlgZSD of eq. 7.5-20 instead of a fixed 4°; <br>• `zod_offset`: the NLOS ZOD offset of Table 7.5-6. <br>Also: `table` `v16` / `v19`. |
| `scripts/make_dataset.py` | `--dataset 3GPP --input <part files>`: <br>• lens x = U0ᴴh (the DeepMIMO convention), ‖x‖² = N per channel; <br>• test set from held-out drops: validation and test are the last drops, and no drop is shared with training; `--test random` is also available; <br>• the CS matrix of SV. <br>New option `--authors-bands` (every data set): bands 0–10 / 10–20 dB with 10 SNR chunks and validation at 10 dB, as `sv_*.yaml` and `deepmimo_*.yaml`. |
| `scripts/check_3gpp.py` (new) | Checks the 3GPP data against: <br>• 38.901 Table 7.4.2-1 (LOS probability per distance) and Table 7.5-6 (DS, ASD, ASA, ZSD, ZSA, K, LOS and NLOS apart, v16 or v19); <br>• SV and DeepMIMO: beamspace sparsity, strongest beam, LMMSE difficulty; <br>• the authors' published GM-LAMP networks on the three test sets. |
| `scripts/compare_3gpp_generators.py` (new) | The same setting from several generators: beamspace statistics, LOS fraction, LMMSE. |
| `scripts/compare.py` | No authors' networks are imported for data sets the authors did not publish networks for (3GPP). Before, the import stopped with "the scenario's CS matrix is not GM-LAMP's". Give `--ref <tag>` for the gains. |
| `app/bsce_app.py` | Step 1: <br>• new task "Raw 3GPP channels (Sionna)", optionally creating the environment first; <br>• new task "Raw 3GPP channels (QuaDRiGa; MATLAB or Octave)"; <br>• "Data set of any size" offers 3GPP and the authors' bands. <br>The reference test set includes the new tests. |
| `tests/test_make_dataset_3gpp.py` (new) | Covers: <br>• make_dataset: lens, normalisation, held-out-drop split, `--authors-bands`, refusal of another array; <br>• the table formulas of check_3gpp; <br>• both generators, when installed: a part made alone equals the same part made in sequence, the users lie in the sector, other settings are refused; <br>• compare_3gpp_generators. |
| `tests/test_app.py` | Values for the two new tasks. |
| `tests/test_admmnet_vs_thadmm.py`, `test_classical_vs_octave.py`, `test_data_vs_octave.py`, `test_export_to_original_matlab.py`, `test_ista_lvamp_vs_tf.py`, `test_ladmm_vs_originals.py` | **Test fix.** These cross-checks against your original MATLAB and THADMM code now skip when that code (`reference/ladmm_user`, `reference/thadmmnet`, `reference/alista`) is missing. <br>Before, they checked only for Octave: with Octave installed (for example for QuaDRiGa) but without those folders, 20 of them failed. The ADMM-Net check had no condition at all. |
| `GUIDE.md` 1.1, `DATA_GUIDE.md` 7, `docs/WORKBENCH.md`, `scripts/README.md`, `docs/CLI_REFERENCE.md` | Updated. |
| `docs/3GPP_DATASET.md` (new) + `docs/3gpp_check/` | The guide and the checks measured here (figures, json, md). |

## Verification (this session)
- Data:
  - Sionna: 10,000 channels per array (UPA, ULA on y, ULA on z), plus 5,000 UPA channels with the v19.2 tables.
  - QuaDRiGa (Octave 8.4): 10,000 channels per array with the fixes, and the same without them.
- Numbers: `docs/3GPP_DATASET.md` section 4.
- End to end on a 3GPP set: `make_dataset.py`, then a short `train.py` run of LADMM and `evaluate.py`.
- Tests:
  - `tests/test_make_dataset_3gpp.py`: 5 passed, with Sionna and Octave + QuaDRiGa installed.
  - The workbench reference set: 102 passed.
  - The whole `tests/` folder (Octave installed, TensorFlow not, `reference/` = the authors' package only): 208 passed, 84 skipped, 0 failed.

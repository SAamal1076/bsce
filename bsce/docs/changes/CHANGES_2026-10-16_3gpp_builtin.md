# 16 Oct 2026: the 3GPP data sets in one command, Python only, in the format of SV and DeepMIMO

**Why.** You asked three things:
- how many examples the 3GPP paper uses, with an option to choose the train / val / test sizes;
- the 3GPP UPA and ULA data sets in the code (or a MATLAB guide, whichever is easier);
- the same format as the DeepMIMO and SV data sets, with options to run them as the others.

**The answer on sizes.**
- 3GPP TR 38.901 defines no data-set size: it is a channel model, and every channel is a new random draw.
- The 3GPP paper closest to this work, Dong et al., IEEE JSTSP 2019 (arXiv 1904.06761; 38.901 CDL at 28 GHz), uses 81,000 / 9,000 / 19,000.
- GM-LAMP's protocol paper, Wei, Hu, Dai, IEEE TCOM (arXiv 1910.12455), uses 80,000 / 2,000 / 2,000 (SV) and 50,000 / 2,000 / 2,000 (DeepMIMO).
- All of these are presets now. The default is the size of SV.

**The choice: Python, not MATLAB.** A new built-in generator, `bsce/data/tr38901.py`, follows the TR procedure as written. It runs in the bsce environment and needs nothing else.
- **Speed:** about 1 s per 1,000 channels, so 1.5 min per 100,000, against about 2 h with QuaDRiGa.
- **Agreement with Sionna 2.2:** beams for 95 % of the energy within 1 beam, LMMSE within 0.04 dB.
- **QuaDRiGa:** it deviates from 38.901 in four places (`docs/3GPP_DATASET.md` 5.2).

No existing default, data set or folder name changes. The SV and DeepMIMO data and all earlier results are untouched.

## Changes
| file | change |
|---|---|
| `bsce/data/tr38901.py` (new) | **The built-in generator**: 3GPP TR 38.901 v16.1 or v19.2 tables, UMi street canyon, 28 GHz, narrowband, plane waves. <br>• LOS state by Table 7.4.2-1. <br>• LSPs: spatially correlated within a drop, cross-correlated by the Cholesky factor of Table 7.5-6, clipped as §7.5. <br>• Clusters: delays, powers, AOD/ZOD with the K-factor terms, the NLOS ZOD offset, the intra-cluster ZSD (3/8)·10^μlgZSD, sub-clusters, ray coupling, the LOS ray. <br>• BS element of Table 7.3-1 (or isotropic); one isotropic UE antenna. <br>• Users uniform in the 120° hexagonal sector (ISD 200 m, d2D ≥ 10 m), BS at 10 m, UE at 1.5 m. <br>• Every drop has its own seed (seed, drop), so any drops can be made alone and every computer makes the same channels. |
| `scripts/make_3gpp_data.py` (new) | **One command for the data sets**, like `make_gmlamp_data.py`. <br>• Sets: `data/3GPP_UPA` (UPA 16×16) and `data/3GPP_ULA` (ULA 256 on y); `--only 3GPP_ULAZ` adds the ULA on z. <br>• Sizes: `--sizes sv\|deepmimo\|dong2019\|100k` or `--n-train/--n-val/--n-test`. <br>• Model options: `--spec`, `--channel umi\|umi_los\|umi_nlos`, `--element`, `--snr`. <br>• Output options: `--suffix`, `--seed`, `--out-root`. <br>• Run options: `--check`, `--force`, `--dry-run`. <br>• It ends with a table of the sizes made and the scenario files. |
| `scripts/make_dataset.py` | **`--dataset 3GPP` without `--input` uses the built-in generator.** <br>• Held-out drops: training from drop 1, validation from 1,000,001, test from 2,000,001, so the validation and test users do not depend on the training size. <br>• Writes `<D>_<ARR>_meta.mat` (position, LOS state, drop, LSPs per set). <br>• New options: `--ula-axis y\|z`, `--spec`, `--channel`, `--element`, `--drop-users`, `--raw-out` (also writes the raw H). <br>• `--sizes` presets work for every data set. <br>• The scenario yaml records the channel model. |
| `configs/scenarios/3gpp_upa.yaml`, `3gpp_ula.yaml`, `3gpp_ulaz.yaml` (new) | **The scenario files.** `data/3GPP_*`, the CS matrix of SV, the authors' bands, 80,000 / 2,000 / 2,000. |
| `scripts/run_suite.py`, `scripts/compare.py` | **The new scenarios.** <br>• Scenario names `3gpp_upa`, `3gpp_ula`; `--scenarios 3gpp` means both, `--scenarios all 3gpp` all six. <br>• `compare.py` takes the reference of each table from the runs it holds: `--ref`, else `GM-LAMP-T8-ft`, `GM-LAMP-T8`, `GM-LAMP-T8_gmlamp_trscenario`, `GM-LAMP-T8_paper_trscenario`. Before, a 3GPP table without the authors' networks had no reference. <br>• `compare.py` titles show the actual test size. |
| `scripts/check_3gpp.py` | **`--builtin UPA ULA_Y ULA --n 10000` checks the built-in generator;** `--raw` is now optional. <br>• Users shared by the arrays (same seeds) are counted once in the standard check. Before, they were counted three times, so the binomial sd of the LOS fraction showed 0.003 instead of 0.005. <br>• The path-spread column is shown only for generators that have it. |
| `app/bsce_app.py`, `app/static/app.js` | **Workbench.** <br>• Step 1 has a new task, "3GPP TR 38.901 data sets (built-in, Python only)". <br>• The scenarios "3GPP UPA" and "3GPP ULA" are in steps 2–4. <br>• "Data set of any size" has size presets and the ULA axis. <br>• The reference test set includes the new tests. |
| `tests/test_tr38901_builtin.py` (new) | **13 tests.** <br>• The generator: drops made alone vs in sequence; the factorised array sum vs the direct sum for the three arrays; the sector geometry; LOS count and LSP medians/spreads against the v16.1 and v19.2 tables; forced LOS/NLOS; the element gain. <br>• make_dataset: lens and normalisation; validation and test identical for two training sizes; the drops and the meta file; ULA axis; size presets. <br>• The scripts: make_3gpp_data dry run, refusals and a small real run; check_3gpp `--builtin`; run_suite `all 3gpp`; the shipped yaml files. |
| `scripts/train.py`, `scripts/check_env.py` | A missing 3GPP data set names `make_3gpp_data.py`; `check_env.py` lists `data/3GPP_UPA` and `data/3GPP_ULA` as optional. |
| `tests/test_app.py` | Values for the new task. |
| `docs/3GPP_DATASET.md` | **Rewritten around the built-in generator.** <br>• Section 1: the sizes question. <br>• Section 3: the commands and options. <br>• Section 5: the checks of three implementations. |
| `docs/3gpp_check/` | **The checks again, for every generator.** <br>• New: `builtin/`, `builtin_v19/`. <br>• Redone with users counted once: `sionna/`, `sionna_v19/`, `quadriga/`. <br>• `generators/` now includes the built-in generator. |
| `GUIDE.md` 1.1, 4.1, 4.8; `DATA_GUIDE.md` 7; `scripts/README.md`; `docs/WORKBENCH.md`; `docs/CLI_REFERENCE.md` | Updated. |

## Verification (this session)
- **Built-in against the tables:**
  - On 100,000 UPA channels: LSP medians within ±0.012 decades, log-sd ratios 0.99–1.01, K-factor 9.06 dB with sd 4.97 dB.
  - LOS fraction 0.3937 against 0.3943 expected.
- **Built-in against Sionna 2.2:** 10,000 channels per array; `docs/3GPP_DATASET.md` 5.2.
- **The data sets:** `make_3gpp_data.py --check` made `data/3GPP_UPA` and `data/3GPP_ULA` (80,000 / 2,000 / 2,000 each) in 4 min 16 s, check included.
  - The folders have the files of `data/SV_UPA`, plus the meta file, `dataset_check.png` and the `.snr.json` files.
- **End to end:** `train.py` (LADMM, T = 1, short run) and `evaluate.py` on `3gpp_upa.yaml` ran with no error.
- **Tests:**
  - `tests/test_tr38901_builtin.py`: 13 passed.
  - The whole `tests/` folder (Sionna and Octave + QuaDRiGa installed, TensorFlow not, `reference/` = the authors' package only): 222 passed, 84 skipped, 0 failed.

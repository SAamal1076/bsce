# 16 Oct 2026 (v23): white or colored measurement noise, for every data set and every method

**Why.** bsce's data follow GM-LAMP: the noise is added before the combiner, y = A(x + n), so the measurement noise is
colored (covariance σ²AAᵀ). UTRR and most other papers add white noise after the combiner, y = Ax + w. The project's
noise analysis measured what that costs: up to 8 dB for the published GM-LAMP when tested on the other noise
(`LADMM_UTRR_noise_analysis`). You asked for data sets in both options, to compare them, and for a smoke test of NF-LADMM.

**What.**
- A noise-model option, `colored` (the default, everything as before) or `white`, in every step that makes or uses
  measurements: data files, training (files or online noise), validation, fine-tune, evaluation, depth curves,
  comparisons.
- White runs go to `outputs/<SCENARIO>_white/` under the same folder names, so each network has a twin trained on the
  other noise.
- A new script, `compare_noise.py`, puts both noise models side by side with cross tests.
- NF-LADMM gets the exact data step of the white likelihood. It has the same parameters, so a network can also be run
  with the other step.
- The guide is `docs/NOISE_MODELS.md`, and GUIDE 4.10 gives the commands.

**The white noise model.** w = √c (AAᵀ)^(−1/2) A n, with c = tr(AAᵀ)/M.
- **Exactly white:** w ~ CN(0, cσ²I).
- **Same power:** the same mean noise power as A n, so the same SNR means the same signal and the same noise power.
- **Paired:** made from the same random draw n as the colored noise. So a white file, validation set or test set has the
  same channels and random numbers as its colored twin, and a difference between the two is due to the noise colour
  alone.
- **The colored path is untouched:** every colored file, validation set, test set and result is bit-identical to v22.
  The checkpoint names of colored runs are unchanged too, so an interrupted v22 run resumes.

## Changes
| file | change |
|---|---|
| `bsce/data/noise.py` (new) | **The two models:** <br>• the noise model of a scenario, aliases (`colored` = `coloured` = `folded`); <br>• `whitening_map(A)` = √c (AAᵀ)^(−1/2); <br>• `apply(y_colored, x, A, model)` = A x + T(y_colored − A x), for numpy or torch; <br>• file and folder suffix `_white`. |
| `bsce/data/measurements.py` | `gen_measurements_py` (training and validation files) and `gen_measurements_matlab` (test noise): `noise_model=`. The random draw is the same; the white output is the colored one whitened. |
| `bsce/data/files.py`, `bsce/config.py` | **White measurement files** `<colored name>_white.mat` (`y`, `snr_db`, `noise_model`): <br>• made from the colored file of the same band when it exists (the authors' files too); <br>• otherwise with the random numbers the colored file would use. <br>Also: `online_validation` / `points_validation` take `noise_model=`; `scenario_run_name()` gives `<name>[_white][_smoke]`. |
| `bsce/train/trainer.py` | `Trainer(..., noise_model=)`: white online minibatches (the same draw, whitened). |
| `bsce/models/nfladmm.py`, `bsce/models/__init__.py` | **NF-LADMM `noise_model` (`colored` default, `white`):** <br>• the white x-step is x = v + Aᵀ U diag(1/(λᵢ + c(1/β_t − 1))) Uᵀ (y − Av), the exact proximal map of the white likelihood; <br>• the colored step is unchanged; <br>• the same parameters, no new saved state, so files of earlier versions load. <br>`data_step()` is the x-step of one layer, used by the tests. |
| `scripts/train.py` | `--noise-model` (default: the scenario's `noise_model`, else colored): <br>• the white folder `outputs/<SCENARIO>_white/`; <br>• NF-LADMM's data step set to the data's noise model (`--set model.noise_model=colored` for the mismatch on purpose); <br>• the noise model logged in `train_log.json` and `README.txt`. |
| `scripts/run_suite.py` | `--noise-model white`: every run of the suite with white noise, same names, in `<SCENARIO>_white`; the fine-tune follows its network. |
| `scripts/finetune_bands.py` | The noise of the source network (from its `train_log.json` / `finetune_log.json`); `--noise-model` to override; logged. |
| `scripts/evaluate.py` | **`--noise-model auto\|colored\|white`:** auto = white for a `<SCENARIO>_white` folder. Cross test example: `--runs outputs/SV_ULA --noise-model white --out ...`. <br>**`--nf-step matched`:** NF-LADMM with the data step of the test noise (same weights). |
| `scripts/compare.py` | `--noise-model white`: the white runs on white test noise. The published networks, trained with colored noise, are not imported there; the reference is then a GM-LAMP trained here with white noise. |
| `scripts/compare_noise.py` (new) | **Both noise models, with cross tests:** <br>• every network on colored and on white test noise, trained on either; <br>• `--nf-matched` adds NF-LADMM with the test's data step; <br>• the published GM-LAMP / LAMP and AMP are added; <br>• the mismatch loss at every SNR; <br>• writes `noise_comparison.png/.pdf`, `noise_summary.csv/.md`. |
| `scripts/make_dataset.py` | `--noise-model colored\|white\|both` for the measurement files and `--test-measurements` (`both` = paired sets). |
| `scripts/make_sv_measurements.py` | `--noise-model white`. |
| `scripts/depth_snapshots.py` | Test noise = the noise the run was trained with. |
| `app/bsce_app.py` | **Workbench:** <br>• step 1, data set of any size: "Noise of the measurement files" (colored / white / both); <br>• steps 3–4: "Measurement noise" (colored / white); <br>• step 5: "Noise" (colored / white / both with cross tests → `compare_noise.py`), and "NF-LADMM also with the data step of the test noise". |
| `RUN_IN_SPYDER.py` (Spyder edition) | `NOISE_METHODS` in Settings; cells 4.10a (the suite with white noise) and 4.10b (`compare_noise.py`, `compare.py --noise-model white`). |
| `tests/test_noise_model.py` (new) | **10 tests:** <br>• the whitening map; names; <br>• paired generators, and the original draw unchanged; <br>• white statistics: variance cσ², uncorrelated, the same mean power, while the colored noise is not white; <br>• the validation sets; the white files from the colored twin; <br>• NF-LADMM's white step = the closed form and optimal; the colored step optimal; no new state, old files load, the step switchable; <br>• trainer online noise; <br>• the whole pipeline: `train.py` both models (files), `evaluate.py` auto, `compare_noise.py --nf-matched`. |
| `tests/test_app.py` | The workbench noise options reach `run_suite.py` / `compare.py` / `compare_noise.py` / `make_dataset.py`. |
| `docs/NOISE_MODELS.md` (new) | The guide, the data check and the NF-LADMM smoke test. |
| `GUIDE.md` 1, 4.10 and 6, `MANUAL.md` 3.3b, 10 and A2–A3, `docs/WORKBENCH.md`, `scripts/README.md`, `docs/CLI_REFERENCE.md`, `VERSION.txt` | Updated. |

Not changed:
- UTRR's own Table II reproduction (`utrr_reproduce.py`) keeps the paper's definition of white noise (SNR fixed per
  channel, ‖Ax‖²/‖w‖²);
- the data checks against the authors (`check_vs_authors.py`, `check_3gpp.py`) stay colored, as the authors' data.

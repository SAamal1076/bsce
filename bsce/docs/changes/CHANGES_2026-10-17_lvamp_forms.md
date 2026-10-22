# 17 Oct 2026 (v27): the audit's points about bsce v26, checked and fixed

**Why.** The consolidated audit of 16 Oct (`LADMM_consolidated_audit_2026-10-16.md`) listed gaps and bugs of the v26 interface and one claim about a baseline.

**Scope.** This version takes only the points that concern bsce v26 itself. Not taken here:
- the graph methods;
- the LADMM+ / analytic LADMM / NOMP ideas;
- the JAX pilot;
- the features of the separate Colab code line.

## 1. LVAMP and the noise model (audit A1): checked; the claim does not hold after training
**The claim.** v26's LVAMP uses the LMMSE weights of white noise, while the data have folded noise y = A(x + n), so LVAMP would be handicapped.

**The theory is right.**
- With A = U S Vᵀ, the LMMSE stage works on ỹ = S⁻¹Uᵀy.
- **White noise** (z = Ax + w): component i has noise variance σ²/sᵢ². This is what LVAMP uses.
- **Folded noise:** ỹ = Vᵀ(x + n), so every component has the same variance σ².
- For the GM-LAMP matrices, sᵢ² spans 0.19 … 5.87.

**v27 adds the matched stage as an option:** `model.noise_model=colored` (uniform weights). A test checks it against a direct LMMSE computation (1e-10), and checks that with AAᵀ = I both stages are the same network.

**Measured (SV ULA, one network 0–20 dB, CPU):**

| training | white (original) | folded (matched) | folded − white |
|---|---|---|---|
| end to end, T = 6, 3000 steps, 2 seeds: test mean 0–20 dB | −8.41 dB | −7.83 dB | **−0.59 dB** (better at 2–6 dB by up to 0.70; worse from 8 dB, by up to 1.40 at 20 dB) |
| validation (both seeds) | −7.878 / −7.877 dB | −7.563 / −7.564 dB | −0.31 dB |
| its own layer-wise protocol, T = 2, at most 5000 iterations per stage: test mean 0–20 dB | −5.76 dB | −5.66 dB | **−0.11 dB** (better at 0–4 dB, by up to 1.58; worse from 6 dB, by up to 0.69) |

**So the original stage stays the default.** In both checks the matched stage is better at low SNR and worse at high SNR, and worse on average.
- **The likely reason:** LVAMP learns one noise variance for the whole band. With the matched weights, that single value fits neither end of a 0–20 dB band. The white weights also damp the poorly conditioned directions, which helps here.
- **LVAMP's real limit on these benchmarks** is that single learned noise level. Both variants saturate near −10 / −9 dB above 10 dB, below AMP. A fair LVAMP would get the per-sample σ², as NF-LADMM does. That is not built here.
- **Old networks:** saved without the key, they load as the white stage, unchanged.
- **Code:** `bsce/models/lvamp.py`, registry `noise_model` (default `white`).
- **Files:** figure `lvamp_check.png`; scripts `run_lvamp_check.sh`, `run_lvamp_check2.sh`, `plot_lvamp.py`.

## 2. Fine-tune (audit §6.2)
- **`finetune_bands.py --from <run>/layers/T06_<run>_<band>.pt`** (a depth snapshot) now works, without `--scenario`:
  - the data set comes from `<run>/train_log.json`;
  - the fine-tune is named `<run>_T06_ft_<edges>dB_<iterations>` and written next to `<run>`.
  - Before, the snapshot found no log, the name became `layers_ft_…`, and the folder landed inside the run.
- **Workbench, "Fine-tune one network":**
  - new fields: "Data set", "Measurement noise", "Validation at fixed SNRs" and "Edge margin";
  - the network list also offers the depth snapshots.
- **Workbench suite:** "Fine-tune: stop a learning rate after … steps" (`--ft-patience`).

## 3. Per-method settings and method descriptions (audit U3, §6.2)
- **Own settings per method:** every row of the methods table has a cell for its own settings (`key=value; key=value`).
  - Methods with their own settings get their own `run_suite.py` command. Their folder names carry their overrides.
  - "Extra settings" still applies to every ticked method.
  - Before, `model.K=64` for UTRR also changed GM-LAMP's K.
- **Descriptions:** every row shows one line about the method, for example:
  - UTRR needs `--protocol paper`;
  - NF-LADMM gets the true σ² unless `model.sigma=residual`.
  - The rows also list the model options of LADMM, GM-LAMP, LISTA and LVAMP.

## 4. Forms for two CLI-only tools (audit §6.2)
- **Step 2, "UTRR: reproduce its Table II":** `utrr_reproduce.py`, with SNR, top-K list, BN, size, epochs, patience, batch, seed and device.
- **Step 5, "Ensemble of trained networks":** `make_ensemble.py`, with the run folders, the name and the parent folder.

## 5. Bugs (audit U1, U2, U5, U6, U7)
| # | was | v27 |
|---|---|---|
| U1 | the Networks view never showed the "in progress (resumable)" style (`=== "in progress"`) | prefix match |
| U2 | two scenario files with the same `name:` share `outputs/<NAME>/` and the folder names. Examples: `sv_ula_4bins.yaml` and `sv_ula.yaml`; `deepmimo_upa_2band_valrange.yaml` and `deepmimo_upa.yaml`. A run of one was skipped as the other, or its networks were mixed into the other's folder. | `run_suite.py` compares the existing folder's `train_log.json` with this run before it skips or trains: band names, ranges, validation, file names, SNR distribution and data folder. On a difference it stops with an explanation. Keys an older log lacks are not compared. |
| U5 | the quick tests of step 0 left out the tests of v22–v26 | + `test_noise_model`, `test_resume`, `test_snr_dist`, `test_utrr`, `test_lvamp_noise` |
| U6 | `/api/help` existed and was never called | "Any script": "Show its options" prints the script's `--help` |
| U7 | the compare preset "one network 0–20 dB" (`*-T6`, …) missed suffixed runs (`_fair`, `_paper`, `_e2e30k`) | every trained run (`*-T[0-9]*`) minus fine-tunes and other training ranges; new `--exclude` in `compare.py`, `compare_noise.py` and `evaluate.py` |

## 6. Documentation (audit U4)
- **WORKBENCH.md:** no longer says that every option is in the forms; it says where the others go.
- **GUIDE 4.1:** the workbench's default selection (LADMM 6 4, LAMP, GM-LAMP, NF-LADMM-SR) against `--methods all`; the "own settings" cell.
- **MANUAL:**
  - the protocol `paper` (= `original`);
  - an NF-LADMM-SR row;
  - the e2e row with its early stop;
  - the depth-snapshot note;
  - the LVAMP option.
- **Folder-name spellings, as the code makes them:** `_restore-best-true` (run_suite help, naming) and `_lr0p0001` (GUIDE 4.1b, finetune help).

## Not changed
- **Every default training and folder name.** LVAMP's default is the v26 network. The checkpoint fingerprints of v26 runs are unchanged, so they still resume.
- **The audit's other items** (A2–A5, the classical baselines, the Colab line) are outside this scope or cosmetic:
  - A4, NF-LADMM's buffers: safe through `load_model`;
  - A5, dead last parameters: they would change the state dicts.

## Tests
- `tests/test_lvamp_noise.py`, 5 new tests:
  - the two LMMSE stages against a direct computation, and their equality for AAᵀ = I;
  - an old checkpoint loads unchanged;
  - `train.py` names and configs;
  - the scenario clash: other bands, other validation, the same setting skipped;
  - the fine-tune of a depth snapshot.
- `tests/test_app.py`:
  - the new forms in `test_every_option_exists` (ensemble, UTRR Table II, fine-tune fields);
  - `test_v27_forms`: per-method settings split into commands with their own names, `--ft-patience`, the one-network preset against 10 example names, and that the quick test files exist.

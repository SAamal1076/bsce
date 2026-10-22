# 15 Oct 2026: data and training checked against the authors' package; the authors' protocol for every method

**Why.** You sent the authors' package (`Deep learning for beamspace channel estimation…zip`) and the raw DeepMIMO
files (`H_ula.mat`, `H_upa.mat`). The request: the data option and the training protocol must be the authors', so the
comparison is fair on their terms.

**Result.** No data or training code had to be corrected: bsce's GM-LAMP data and `--protocol paper` / `--protocol
gmlamp` already follow the authors' code item by item. What is new:
- the evidence;
- one command that trains every method with the authors' protocol;
- a check script.

The full report is `docs/AUTHORS_PROTOCOL.md`.

## Changes

| file | change |
|---|---|
| `scripts/check_vs_authors.py` (new) | Checks the four GM-LAMP data sets against the authors' files: <br>• measurement recipe; <br>• the 16 published networks on our validation sets, against the validation NMSE the authors logged. <br>With `--runs`, it compares every training stage of a retrained LAMP / GM-LAMP with the authors' own training log (`Train Network/Trained_Networks/*.npz`). It uses a NumPy port of the authors' TF graph, not bsce's models. |
| `scripts/train.py` | `train_log.json` now stores `stage_log` per band (stage name, last validation NMSE, iterations), the same fields as the authors' log. Before, only the `.pt` had it. `README.txt` already said "validation of every stage". |
| `app/bsce_app.py` | Step 2: new task "Check the data against the authors' files". <br>Step 2, paper retraining: new option "Then compare every training stage with the authors' training logs" (on by default). <br>Step 4: new task "Every method with the authors' protocol" (`--protocol gmlamp --train-snr scenario --ft-snr none`, float32 or float64 for every method, seeds). <br>Step 5: new preset "trained with the authors' protocol" (`*_gmlamp_trscenario*`). <br>The quick tests include `test_check_vs_authors.py`. |
| `GUIDE.md` | 1: check the data against the authors' files. <br>2.2: stage-by-stage check. <br>4.8 (new): the authors' protocol for every method, with the item-by-item table. |
| `docs/AUTHORS_PROTOCOL.md` (new) | The report. `docs/authors_check/` holds its figures and tables, and `audit/` the stand-alone scripts. |
| `docs/WORKBENCH.md`, `scripts/README.md`, `docs/CLI_REFERENCE.md` | Updated for the new tasks and script. |
| `tests/test_check_vs_authors.py` (new) | The NumPy port equals bsce's LAMP and GM-LAMP to < 1e-12 on a non-trivial output; the stage-log reader. |
| `tests/proposals/nf_ladmm.py`, `nf_ladmm_relax.py` (moved) | **Packaging fix.** `test_nfladmm.py` and `test_nfladmm_relax.py` compare with these two reference modules. They lived in `reference/proposals/`, which is not in the code zip, so on your PC these 12 tests could not find them (since bsce_code_12). They now ship in `tests/proposals/`. |
| `tests/test_app.py` | The new task gives the GUIDE 4.8 command. Its dry run trains 2 networks × 5 methods × 4 data sets, with no fine-tune and no `--snr`, and float64 when chosen. |

**Unchanged.** No default, no existing folder name, no data file. `make_gmlamp_data.py` still gives `4 x OK` (checked
here with your two raw files).

## Verification (this session)
- **Data from your raw files.**
  - `make_gmlamp_data.py --sv --deepmimo-ula H_ula.mat --deepmimo-upa H_upa.mat`: `4 x OK`, identical to the data of all earlier results.
  - The authors' DeepMIMO test channels are found in your files: 1990 / 2000 (ULA), 1991 / 2000 (UPA).
- **Measurement files.**
  - The SNR re-estimated from y − Ax with the authors' matrix is within 0.016 dB of the stored SNR.
  - The noise has the authors' structure: y = A(x + n), covariance σ²AAᵀ.
- **Published networks on our validation sets.** Within −0.24 … +0.23 dB of the authors' logged validation NMSE. 13 of 16 are within the sampling spread.
  - The exception is DeepMIMO ULA. There the authors' own validation set is unusually hard: 98th–99.9th percentile of random 2000-channel sets of our users. Their test set is typical.
- **First layer retrained on our data with the paper protocol** (LAMP, DeepMIMO UPA and ULA, 0–10 dB) against the authors' logs:
  - the same stage sequence and lengths (86k vs 91k, 73k vs 83k iterations);
  - the validation NMSE 0.03–0.30 dB higher. That is within the authors' own run-to-run spread: up to 0.20 dB between their two runs of the same stages. The 10 unseeded chunk SNRs of their recipe alone give ±0.21 dB.
  - Details: `docs/AUTHORS_PROTOCOL.md` 3.1.
- **Smoke run of every method with the authors' protocol** (float64): LADMM, LAMP, GM-LAMP, NF-LADMM-SR, two networks each.
- **Tests:**
  - With the authors' package as `reference/gmlamp`: 130 passed, 146 skipped (Octave / TensorFlow not installed here), 0 failed.
  - The workbench sets: quick (no `reference/`) and quick + reference, 89 passed.

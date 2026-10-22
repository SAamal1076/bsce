# 16 Oct 2026 (v26): training SNRs spread evenly over the band (`--snr-dist fair`)

**Why.** You asked for random training noise with a fair distribution over [min, max] (0, 0.2, …, 20 dB, or any value in between), instead of a few fixed SNRs. Up to v25 a measurement file was cut into SNR chunks, and each chunk had one SNR drawn at random from U(band):

| file | SNRs | largest SNR gap with no sample | empty 0.5 dB bins | KS distance to uniform |
|---|---|---|---|---|
| authors' bands, 0–10 dB (10 chunks) | 10 | 3.64 dB (1.50 → 5.14 dB) | 60 % | 0.250 |
| authors' bands, 10–20 dB (10 chunks) | 10 | 3.38 dB (13.56 → 16.94 dB) | 60 % | 0.344 |
| our 0–20 dB (100 chunks) | 100 | 1.06 dB | 8 % | 0.074 |
| **`--snr-dist fair`, 0–20 dB** | **80,000 (one per sample)** | **0.0005 dB** | **0 %** | **0.00001** |
| `--snr-dist fair --snr-step 0.2` | 101 (0, 0.2, …, 20), 792 or 793 samples each | 0.2 dB (the grid) | 0 % | 0.010 (the grid) |

The online noise (NF-LADMM's own training, the fine-tune) already drew one SNR per sample, i.i.d. U(band). In a minibatch of 400 that gives between 11 and 30 samples per dB; `fair` gives exactly 20.

Script and figure: `fair_check/snr_coverage.py`, `snr_coverage.png`.

## The option
`--snr-dist fair` in `run_suite.py`, `train.py`, `finetune_bands.py` and `make_sv_measurements.py`; `--snr-step S` for the grid.

| where | `chunks` (default, as before) | `fair` |
|---|---|---|
| **training file** (every common protocol: `ladmm_best`, `gmlamp`, `e2e`, …) | n chunks (the authors 10, ranges 100), one random SNR each | every column its own SNR, **stratified**: the band cut into K equal parts, one SNR uniform inside each part, the order random. With `--snr-step S`: the grid lo, lo+S, …, hi, every value equally often (counts differ by at most one). The SNR of every column is stored (`snr_db`), as before. |
| **validation file over a range** | n chunks at random SNRs | n chunks at **equally spaced** SNRs, the middles of n equal parts (0–20 dB, n = 100: 0.1, 0.3, …, 19.9 dB, 20 channels each) |
| **validation at fixed SNRs** (the authors' 10 dB) | – | unchanged |
| **online noise** (`train.noise=online`: NF-LADMM's own training) | i.i.d. U(band) per sample | every minibatch spread evenly: one SNR in each of B equal parts, random order (grid: cycled from a random start). `train.snr_dist=fair`, `train.snr_step` |
| **fine-tune** | i.i.d. U(band) per sample; validation at random SNRs | every minibatch spread evenly; validation at 100 equally spaced SNRs of the band |

**Names.** No existing file or run changes:
- **Files:** `…_measurements_0to20dB_fair.mat` (grid: `_fair0p2.mat`). The validation file over a range is `…_0to20dB_fair.mat`.
- **Trained runs:** `LADMM-T6_fair`, `LADMM-T6_trm10-30_fair0p2`.
- **Fine-tunes:** `<run>_ft_0-10-20dB_20k_fair`. The fine-tune's own setting is in its own suffix, so the suite's fine-tune of `LADMM-T6_fair` is `LADMM-T6_fair_ft_0-10-20dB_20k_fair`.
- **White noise:** `…_fair_white.mat`, the colored fair file with the same draw whitened.

**Refused, with a message:**
- `--snr-step` without `--snr-dist fair`;
- `--snr-chunks` with `--snr-dist fair`, because chunks have no meaning there;
- a step of 0 or less.

**Examples:**
```
%run scripts/run_suite.py --scenarios all --methods all --snr-dist fair --device cuda
%run scripts/run_suite.py --scenarios sv_upa --methods ladmm6 --train-snr -10 30 --ft-snr -10 0 10 20 30 --snr-dist fair --device cuda
%run scripts/train.py --scenario sv_ula --method ladmm6 --snr-dist fair --snr-step 0.2 --device cuda
%run scripts/finetune_bands.py --from outputs/SV_UPA/LADMM-T6_fair --snr-dist fair --device cuda
%run scripts/make_sv_measurements.py --scenario configs/scenarios/sv_upa.yaml --snr 0 20 --snr-dist fair    # files ahead
```
In the workbench: step 4, the field "Training SNRs" (chunks / fair) and "SNR grid step", in the training tasks and in "Fine-tune one network". In Spyder: cell 4.1 has a commented example.

## What changes
| file | change |
|---|---|
| `bsce/data/measurements.py` | `fair_snr()`: stratified or on a grid. <br>`evenly_spaced_snr()`. <br>`gen_measurements_percol()`: one SNR per column. Column k uses the k-th N real and N imaginary normal draws, so the file does not depend on the block size; white noise is the same draw whitened. |
| `bsce/data/files.py` | `apply_band_options(snr_dist=, snr_step=)`: `snr_dist`/`snr_step` in the band and the `_fair` names. <br>`ensure_band_files`: the fair training file, and the equally spaced validation file. Its `.snr.json` holds the counts in 20 equal bins. <br>`fair_validation()` (online). <br>`snr_dist_suffix()`. |
| `bsce/train/trainer.py` | `TrainConfig.snr_dist` (`uniform`), `snr_step`; `_batch_snr()`. The default draws exactly the stream of before. |
| `scripts/train.py`, `run_suite.py`, `finetune_bands.py`, `make_sv_measurements.py` | `--snr-dist`, `--snr-step`; names; README lines ("training SNRs"). In the fine-tune the setting enters the fingerprint only when `fair`, so the v25 fine-tunes are unaffected. |
| `app/bsce_app.py` | The fields "Training SNRs" and "SNR grid step" (suite tasks, authors' protocol task, fine-tune task), with their errors. |
| `tests/test_snr_dist.py` (new, 11 tests), `tests/test_app.py` (1) | • the draw: one SNR per part, a flat histogram down to 0.1 dB on 80,000 samples, grid counts; <br>• the noise power of every column; white = colored whitened; independence of the block size; <br>• the file names and contents (training one SNR per column, validation equally spaced in order); <br>• margin, authors' bands and white; <br>• the online minibatches (one SNR per 0.05 dB for B = 400; the default stream unchanged); <br>• `train.py` (file noise and online noise), `make_sv_measurements.py` (the same files), `finetune_bands.py`, `run_suite.py` dry run and errors, the workbench. |
| `GUIDE.md` 1, 4.3; `MANUAL.md` 3.2, 3.3, 9.7, A1; `RUNBOOK.md`; `scripts/README.md`; `docs/WORKBENCH.md`; `docs/CLI_REFERENCE.md`; `VERSION.txt` | Updated. |

## Training with both, on SV ULA (CPU)
LADMM-T2, end to end, 8000 steps, the same everything except `--snr-dist`. Test set: 2000 channels at 0, 0.5, …, 10 dB and 0, 1, …, 20 dB. Script `fair_check/run_fair_check.sh`, `eval_fair_check.py`; figure `fair_check.png`.

| training | test NMSE, mean over the SNRs: chunks → fair | fair − chunks, per SNR |
|---|---|---|
| band 0–10 dB, the authors' file (10 chunks) vs fair; validation at 10 dB for both; seed 0 | −9.753 → −9.838 dB (**0.085 dB better**) | −0.34 dB at 0 dB … +0.29 dB at 10 dB |
| the same, seed 1 | −9.752 → −9.859 dB (**0.107 dB better**) | −0.32 … +0.30 dB |
| one network 0–20 dB, our file (100 chunks) vs fair; seed 0 | −10.760 → −10.814 dB (**0.054 dB better**) | −0.21 dB at 0 dB … +0.15 dB at 18 dB |

- **Above about 3 dB (0–10) and 5 dB (0–20), fair is better at every test SNR,** by up to 0.30 dB.
- **At the lowest SNRs it is worse,** by up to 0.33 dB.
- **The reason is the old files, not chance in the training:** their random chunks put more samples at low SNR than a uniform draw would.
  - The authors' 0–10 dB file has 40 % of its samples at 2 dB or less; fair has 20 %.
  - Our 0–20 dB file has 28 % at 5 dB or less and 20 % at 15 dB or more; fair has 25 % and 25 %.
- **What fair changes:** every SNR of the band gets the same weight, so the mean over the band improves and the low end loses what the chunk files gave it by chance.
- **The two seeds agree** (0.085 / 0.107 dB), so the effect is not seed noise.
- **Limits of this check:** a small network (T = 2), 8000 steps and the CPU. The size of the gain for the paper's networks has to be measured on them (suite with `--snr-dist fair`).

## Not changed
- **The default (`chunks`):** every file, name and network is as in v25. Checked: the online stream is bit-identical, and the existing tests all pass.
- **Validation at fixed SNRs** (the authors' 10 dB) and the **test set** (fixed seeds per test SNR).

# 15 Oct 2026 (evening): data options — validation noise, edge margins, SNR chunks

**Why.** You asked two things:
- whether the validation noise is over the whole range or at one SNR as the authors do, and whether both options exist;
- whether margins are added before and after each band to keep its edges.

**Before.**
- The validation noise was tied to where the bands came from:
  - the authors' bands (`--train-snr scenario`): one validation file at 10 dB;
  - ranges (`--train-snr 0 20`, `0 10 20`) and the fine-tune: noise over the whole range.
- There were no edge margins. Only a wider range (`--train-snr -5 25`), which then is also used on that wider range.

**Now** (defaults unchanged, every existing folder name and file unchanged):

| option | scripts | effect | folder suffix |
|---|---|---|---|
| `--val-snr range` | `run_suite.py`, `train.py` | validation SNR ~ U(band) for every band, also for the authors' bands (new file `…_<band>_valrange.mat`, 100 chunks) | `_valrange` |
| `--val-snr 10` (any list) | `run_suite.py`, `train.py`, `finetune_bands.py` | validation at fixed SNRs, also for ranges and the fine-tune (new file `…_<band>_val10dB.mat`) | `_val10` |
| `--margin M` | `run_suite.py`, `train.py` | a band [lo, hi] is trained on [lo − M, hi + M] (file `…_m2to12dB[_c10].mat`) and used on [lo, hi]; its validation stays on [lo, hi] (the same file as without margin) | `_margin2` |
| `--ft-margin M` | `run_suite.py` (→ `finetune_bands.py --margin`) | the same for the fine-tune: training noise U(lo − M, hi + M) | `-ft_margin2` |
| `--snr-chunks N` | `run_suite.py`, `train.py` | SNR chunks of a training file (the authors 10, ranges 100); the validation file is unchanged | `_c100` |

- **Code:** `bsce/data/files.py::apply_band_options` turns the options into band definitions. `ensure_band_files` makes the files: the validation seed comes from the band itself, so a margin does not change it, and a separate chunk count is kept for validation. Online noise (`--noise online`) honours the options too.
- **Workbench:** step 4 (both training tasks) and step 3 (deep runs) have the fields "Validation noise", "Edge margin", "Edge margin of the fine-tune bands" and "SNR chunks per training file".
- **Docs:** GUIDE 4.3, MANUAL 7, `docs/AUTHORS_PROTOCOL.md` 4.1.

## Verification
- `tests/test_band_options.py` (new, 4 tests):
  - no option = the same bands;
  - a margin band trains inside [lo − M, hi + M], keeps its name, use range and validation file (bit-identical), and its 100-chunk file equals the `--snr` file of the wider range;
  - `range` / points create new validation files with the right SNRs;
  - chunks change the training file only.
- `tests/test_suite_options.py::test_data_options`, `tests/test_app.py::test_data_options_from_the_form`: flags and folder names.
- **Smoke runs** (SV UPA, CPU):
  - LAMP and NF-LADMM-SR, `--protocol gmlamp --train-snr scenario --margin 2 --val-snr range`;
  - LADMM, default suite with `--margin 2 --ft-margin 1 --val-snr 10`.
  - The files were made with the expected SNRs, for example training 0.34–11.96 dB for the band 0–10 with margin 2 and 10 chunks.
  - `evaluate.py` used every band on its own range: `0to10dB -> [0, 10)`, `10to20dB -> [10, 20.01)`.

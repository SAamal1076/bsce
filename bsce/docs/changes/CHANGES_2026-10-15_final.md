# 15 Oct 2026 (final version): one workflow for every method, scenario, protocol and SNR range

## New
- **`run_suite.py` (rewritten).** It covers every learned method: LADMM, LAMP, GM-LAMP, LISTA, ALISTA, LVAMP, ADMM-Net, NF-LADMM-S and NF-LADMM-C.
  - Methods take any depth (`ladmm4`, `lamp8`, `lista12`; the default is the paper depth).
  - `--protocol`: `ladmm_best` (default), `gmlamp_best`, `gmlamp`, `fair`, `e2e` or `paper`.
  - `--train-snr`: edges, or `scenario`.
  - `--ft-snr`: edges, or `none`; `--ft-iters` defaults to 20,000.
  - Also: `--save-layers`, `--seed`, `--set`, `--smoke`, `--dry-run`.
  - Folder names are made only from the choices, so every computer makes the same names.
- **`compare.py`.** Imports the authors' networks, evaluates the selected networks (`--tags`, glob patterns) on every scenario, and writes `comparison.png/pdf`, `summary.csv/md/tex`, `networks.md` and `README.md` to `results/<name>/`.
- **`promote_layer.py`.** Turns the network saved after layer T into a normal T-layer run folder, so a chosen depth needs no retraining.
- **`depth_snapshots.py`.** Accepts patterns (`"outputs/*/LADMM-T12"`) and writes to `results/<name>/`.
- **`README.txt` in every output folder** (`bsce/runinfo.py`). It records the method, T, protocol, data (with the SHA-1 of the data files), training settings, bands, validation and time.
- **`GUIDE.md`** (the final workflow) and **`scripts/README.md`**.

## Changed
- **Measurement noise seed.** It now depends on the band's SNR edges only (`band_seed_offset`), so the same band gives the same file on every computer, whatever other bands are trained. 0–20 dB keeps its old seed, so existing files are unchanged.
- **Deterministic mode on every device** (`bsce/device.py`): `CUBLAS_WORKSPACE_CONFIG`, cudnn deterministic, `use_deterministic_algorithms(warn_only)`.
- **Common protocols are now fully uniform.**
  - `ladmm_best`, `gmlamp`, `gmlamp_best`, `fair` and `e2e` set `noise: file`.
  - `gmlamp`, `fair` and `e2e` set float32.
  - `fair` sets `warm_start: false` and `restore_best: false`.
- **NF-LADMM stage plan.** Every layer stage trains all its parameters, so layer-wise protocols give it the lr phase as well.
- **`finetune_bands.py`.**
  - Passes the noise variance to models that need it (NF-LADMM).
  - Writes its output next to the source network by default.
  - Uses the network's own scenario when `--scenario` is left out.
- **`truncate_model`.** Slices per-layer parameter vectors (NF-LADMM).
- **`evaluate.py`.** Added `--tags`; `--out` now creates the folder.
- **Documents.** CHANGES files moved to `docs/changes/`.

## Checks
| check | result |
|---|---|
| `tests/test_save_layers.py`: truncated network = output after layer t, all 9 models | 9 passed |
| smoke suite, all methods (train + save-layers + fine-tune), then `compare.py`, `depth_snapshots.py` | runs end to end |
| T = 2 run vs the T02 snapshot of a T = 3 run, all 9 methods (CPU) | weight difference 0 for all |
| `--methods ladmm lista` in one command vs `ladmm` and `lista` in two commands | identical networks (difference 0), trained and fine-tuned |
| `promote_layer.py`, then `run_suite.py` continues with the fine-tune | works |
| paper-protocol path (`--protocol paper --train-snr scenario --ft-snr none`) | trains both bands, makes the missing band file |
| regression tests: `test_save_layers`, `test_nfladmm`, `test_make_dataset`, `test_ladmm_best_vs_untitled2` (bit-exact against `untitled2.py`), `test_restore_best`, `test_training_loop_vs_tf` | 28 passed |
| earlier today: LAMP/GM-LAMP against the original TF/MATLAB (`test_lamp_vs_tf`, `test_golden`, `test_lamp_vs_matlab`) after the GM float32 fix | 59 passed |

# 15 Oct 2026 (final + 2): fixes from the code review (docs/REVIEW_2026-10-15.md)

| fix | files |
|---|---|
| F1: common protocols set `grad_clip: null`, so NF-LADMM is no longer clipped while the other methods are not | `configs/train/{ladmm_best,gmlamp,fair}_protocol.yaml` (`gmlamp_best` inherits) |
| F2: checkpoints save and restore the random state, so a resumed run is bit-identical to an uninterrupted one (old checkpoints resume as before, with a note) | `bsce/train/trainer.py` |
| F3: the suite stops if an existing fine-tune was made with another `--ft-iters`, and warns when `--save-layers` meets a run without `layers/` | `scripts/run_suite.py` |
| F4: `make_dataset.py --force` deletes the old measurement files of the folder | `scripts/make_dataset.py` |
| F5: `--protocol gmlamp_best` works with range bands (mean in dB over the SNR chunks) | `scripts/train.py` |
| F6: parameters column shows `2 × n` for band pairs; csv has `parameters_per_network` and `networks` | `scripts/compare.py` |
| F7: `n_params()` counts complex parameters twice (ADMM-Net) | `bsce/models/common.py`, `MANUAL.md` |
| F8: `promote_layer.py` keeps the right number of per-layer values | `scripts/promote_layer.py` |
| F9: NF-LADMM docstring: initial threshold 1.2·σ | `bsce/models/nfladmm.py` |
| test | `tests/test_review_fixes.py` (7 tests; 5 fail on the code before the fixes, all pass after) |

Checks: 80 tests pass here (NF-LADMM, relax, save-layers, restore-best, group threshold, make_dataset, export, proposed LADMM, review fixes). Smoke suite runs end to end: `ladmm_best`, `paper`, `gmlamp_best`, `--set model.sigma=residual`, `--save-layers` + `promote_layer`, and `compare.py`.

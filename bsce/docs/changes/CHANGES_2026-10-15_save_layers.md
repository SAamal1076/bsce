# 15 Oct 2026: save the network after every layer (`--save-layers`) and the test depth curve

## What changed
- **`train.py --save-layers`.** When layer t is finished, with all its refinements, the network as it is at that moment is saved to `<tag>/layers/T<tt>_<tag>_<band>.pt/.mat/_orig.mat`. This applies to layer-wise schedules only.
- **`bsce/export.py: truncate_model(model, t)`.** Returns the first t layers as a t-layer network.
- **`bsce/train/trainer.py`.** New `layer_callback(out_index, model, val_nmse_db)`, called when every stage of an output is done.
- **New `scripts/depth_snapshots.py`.** Evaluates every snapshot on the test set with the same noise as `evaluate.py`. It writes `depth_curve.png/pdf/csv/md`, with all values printed, the test NMSE vs T at 0 / 10 / 20 dB and the mean, the validation vs T, the gain of each added layer, and the flat point.
- **Docs.** MANUAL section 8.1, `docs/CLI_REFERENCE.md`, and `tests/test_save_layers.py`.

## Checks
| check | result |
|---|---|
| `tests/test_save_layers.py`: truncated net output vs `forward_stages(y)[2t-1]` | identical, for t = 1 … 5 |
| DeepMIMO UPA, T = 3, short budget: snapshot validation recomputed vs value logged by the trainer | difference ≤ 0.0004 dB |
| T03 snapshot vs the final T = 3 network | weight difference 0 |
| Separate T = 2 run vs T02 snapshot of the T = 3 run (CPU) | weight difference 0: one T = 12 run gives the T = 1 … 12 networks |

# 15 Oct 2026 (b): `finetune_bands.py` uses the network's own scenario
- **Problem:** the network in `outputs/DeepMIMO_ULA_52k/LADMM-full` was trained on the scenario `DeepMIMO_ULA_52k` (data from `make_dataset.py`). The fine-tune was started with `deepmimo_ula.yaml`, so it read `data/DeepMIMO_ULA`, which is not in `bsce_code(10)` → FileNotFoundError.
- **Fix:**
  - `--scenario` is now optional. By default the scenario stored in `train_log.json` next to `--from` is used.
  - If a different scenario is given, a warning names the right yaml.
  - If the data are missing, the script stops with a clear message instead of a scipy traceback.
- **Checked:** auto-detection (UPA run, no `--scenario`), warning on a mismatch, and the missing-data message.

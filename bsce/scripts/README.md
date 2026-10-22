# scripts/

The full workflow is described in `../GUIDE.md`.

## Main workflow (in order)
| script | what it does |
|---|---|
| `check_env.py` | checks Python/PyTorch/GPU and that `reference/` and `data/` are complete |
| `make_gmlamp_data.py` | makes the four GM-LAMP data sets (SV ULA/UPA 80k, DeepMIMO ULA/UPA 50k) |
| `check_vs_authors.py` | checks those data sets against the authors' files (measurement recipe; the 16 published networks on our validation sets vs the authors' logged validation NMSE) and, with `--runs`, every training stage of a retrained LAMP / GM-LAMP vs the authors' training log |
| `make_3gpp_data.py` (→ `make_dataset.py --dataset 3GPP`) → `check_3gpp.py --builtin` | optional third data set: 3GPP TR 38.901 UMi at 28 GHz, `data/3GPP_UPA` and `data/3GPP_ULA` (80,000 / 2,000 / 2,000; `--sizes dong2019` etc.) in one command, with bsce's own generator (`bsce/data/tr38901.py`). Second implementations: `sionna_make_H.py` (Sionna) and `matlab/quadriga_make_H.m` (QuaDRiGa), read with `make_dataset.py --input`; `compare_3gpp_generators.py` compares them (`docs/3GPP_DATASET.md`) |
| `run_suite.py` | trains and fine-tunes any methods on any scenarios with one protocol and any SNR ranges; parallel-safe; resumes; `--ensemble` (v28, off by default) averages the final networks of some methods |
| `runs.py` | every run folder: what it is, done / resumable (where) / incomplete, best validation; writes `outputs/RUNS.md` |
| `depth_snapshots.py` | depth curve on the test set from the networks saved after every layer (`--save-layers`) |
| `promote_layer.py` | uses the network saved after layer T as a normal T-layer run (no retraining) |
| `compare.py` | fair comparison: authors' networks + evaluation + figures + tables in `results/<name>/` |
| `compare_noise.py` | colored against white measurement noise: every network on both test noises, trained on either (cross tests); `--nf-matched` |
| `utrr_reproduce.py` → `make_ensemble.py` | UTRR (`docs/UTRR.md`): its Table II in the paper's own setting (noise after the combiner, one model per top-K, and the ensemble); then the model-averaging ensemble UTRR-E of networks trained with `run_suite.py --methods utrr` |

## Building blocks (the main scripts call them; each has `--help`)
| script | what it does |
|---|---|
| `train.py` | one training run (one method, one scenario, one or more SNR bands); short form `--scenario sv_upa --method ladmm6` |
| `finetune_bands.py` | fine-tunes one network per SNR band: `--from <run folder>` → `<run>_ft_<edges>dB_<k>/` next to it; resumes; stops a learning rate after 4000 steps without a 0.01 dB gain (`--patience`) |
| `rename_runs.py` | gives fine-tuned folders of v23 and older (`<run>-ft`) the v24 names (dry run unless `--apply`) |
| `evaluate.py` | NMSE vs SNR on a scenario's test set; `--polish band\|snr` (v28, off by default) adds every network + a least-squares correction on its strongest beams, chosen on validation channels |
| `import_gmlamp_weights.py` | converts the authors' GM-LAMP and LAMP networks |
| `make_dataset.py`, `make_sv_measurements.py`, `make_scenario.py` | other data sizes, splits and SNR bands (`make_sv_measurements.py --snr-dist fair`: one SNR per training sample, spread evenly) |
| `make_figures.py`, `aggregate_seeds.py` | figures across runs, mean ± std over seeds |

## Older or diagnostic tools
- `run_all.py`, `compare_variants.py`, `select_depth.py`, `depth_curve.py`: earlier versions of the workflow.
- `status.py`, `review_pack.py`, `render_report.py`, `summarize_tests.py`: tracking and reports.
- `speed_test.py`, `check_exports.py`, `generate_data.py`, `deepmimo_convert.py`, `import_user_ladmm.py`, `make_cli_reference.py`: utilities.

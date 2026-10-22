# 16 Oct 2026 (v24): easier fine-tune and training, one naming scheme, resume inside a stage

**Why.** You asked for three things:
1. a fine-tune saved in the same path as its network, under a name made of the network's folder name + `_ft` + the range, clear and easy to track;
2. easier training, with easy names;
3. whether a training that is interrupted resumes instead of starting again.

**The answer to 3, before v24:**
- **A training** resumed only from the beginning of the stage it was in, so up to one stage was lost.
  - NF-LADMM's end-to-end training is one stage, so an interruption started it again from zero.
  - UTRR lost the current learning rate.
- **A fine-tune** did not resume at all: every band was trained again.

v24 resumes both, exactly.

## What changes
| file | change |
|---|---|
| `bsce/naming.py` (new) | **One naming scheme** for `run_suite.py`, `train.py` and `finetune_bands.py`: <br>• trained: `<METHOD>-T<T>[_<protocol>][_tr<edges>][_<option>][_s<seed>]` (unchanged); <br>• fine-tuned: `<trained name>_ft_<band edges>dB_<iterations per band>[_<option>]`, e.g. `LADMM-T6_ft_0-10-20dB_20k`, `LADMM-T6_ft_0-5-10-15-20dB_10k_margin2`; files `<folder>_<band>.pt`. <br>Also: `parse_ft`, `drop_seed` (mean ± std over seeds), the v23 names (`legacy_ft_tags`). |
| `bsce/train/trainer.py`, `bsce/train/optim.py` | **Checkpoints inside a stage** every `train.checkpoint_minutes` (default 10): <br>• for the layer-wise, `steps` and `end2end` schedules (end2end after a saved epoch); <br>• they hold the weights, the optimizer (TF-1 Adam: its step counter and beta powers too), the random streams, the best state, the validation history and the position. <br>A resumed run ends with exactly the weights of an uninterrupted one. <br>• A stage resumed inside is not warm-started again. <br>• The file is written atomically (`.tmp`, then renamed). <br>• `checkpoint_minutes=0`: after every stage only, as before. |
| `scripts/finetune_bands.py` | **`--from` takes a run folder** (its one network file is used). <br>**The default name is automatic:** `<source folder>_ft_<edges>dB_<iterations>`, next to the source, with every option that differs from its default added (`_margin2`, `_val10`, `_clip1`, `_lr0.0001`, `_batch256`, …). <br>**Defaults:** `--snr 0 10 20` (was: the scenario's bands; `--snr scenario` keeps that) and `--iters 20000` (was 6000), as `run_suite.py`. <br>**Resume:** <br>• finished bands are kept (`finetune_log.json` is written after every band); <br>• the running band continues from its checkpoint (`--checkpoint-minutes`, default 10), exactly. <br>A folder made with other settings, or from another version of the source (SHA-1 of the source file), stops the script with a message instead of being mixed in. <br>`--dry-name` prints the folder; the README lists the source and its SHA-1. |
| `scripts/train.py` | **The short form** `--scenario sv_upa --method ladmm6`: <br>• the method keys of `run_suite.py`; <br>• `run_suite.py`'s defaults (`--protocol ladmm_best`, `--snr 0 20`) and folder names, so both commands make and resume the same run. <br>• `--scenario` also takes a name; `--protocol paper` = `original`; `--snr scenario`. <br>The long form (`--train <yaml>`) is unchanged. It prints the run folder. |
| `scripts/run_suite.py` | **Fine-tunes get the new names.** The budget is in the name, so a fine-tune with another `--ft-iters` is a new folder instead of an error. <br>A v23 folder `<run>-ft[edges]` with the budget asked for is recognised and skipped, not retrained. <br>The names come from `bsce/naming.py`. |
| `scripts/runs.py` (new) | **Every run at a glance:** <br>• what it is (trained / fine-tuned from which network, iterations), method, layers, parameters; <br>• the band networks present out of those expected; the best validation; <br>• **done / resumable (where: stage and iteration, or band and step) / incomplete**. <br>Writes `outputs/RUNS.md` and `RUNS.csv`. |
| `scripts/rename_runs.py` (new) | **v23 fine-tune folders to the new names:** dry run, `--apply`. The iterations come from their log; the files inside, `finetune_log.json` and `README.txt` are updated. A folder without a log (an ensemble) is left. |
| `scripts/aggregate_seeds.py`, `scripts/compare.py` | Seed grouping and the GM-LAMP reference fallback know both name schemes. |
| `app/bsce_app.py` | **Workbench:** <br>• "Fine-tune one network": the name is optional (automatic), and a run folder is enough; <br>• new tasks "What is done, what resumes" (`runs.py`) and "Old fine-tune names to the new ones" (`rename_runs.py`); <br>• the preset "fine-tuned" selects `*_ft_*` and the old `*-ft`; <br>• the Networks view shows "in progress (resumable)" only when a checkpoint is newer than the networks; <br>• the fine-tune progress bar knows the default budget. |
| `RUN_IN_SPYDER.py` (Spyder edition) | Settings `ONE_SCEN`, `ONE_METHOD`, `ONE_RUN`. **New cells:** <br>• 4.1b: one method, the short form; <br>• 4.1c: fine-tune one network; <br>• 4.1d: `runs.py`; <br>• 4.1e: `rename_runs.py`. <br>The UTRR-E and comparison cells use the new names. |
| `tests/test_resume.py` (new) | **6 tests.** Interrupted inside a stage, then resumed: the weights are identical (difference 0.0) to an uninterrupted run, for: <br>• layer-wise LADMM with TF-1 Adam; <br>• layer-wise GM-LAMP with warm start; <br>• `steps` NF-LADMM-SR; <br>• `end2end` UTRR with `lr_list` and patience. <br>Also: no `.tmp` left; the time-based default; and `finetune_bands.py`, killed inside band 2 and run again: band 1 is kept (not retrained), band 2 is identical to an uninterrupted fine-tune, the names are right, `runs.py` shows "resumable: band 10to20dB, step …", a second run does nothing, and another budget gives another folder. |
| `tests/test_suite_options.py`, `tests/test_app.py` | The new names in the commands. **New tests:** <br>• a v23 `-ft` folder with the asked budget is skipped, one with another budget is not mixed in; <br>• `rename_runs.py` (dry run, apply, ensembles left); <br>• seed grouping for both names; <br>• the fine-tune task needs no name. |
| `GUIDE.md` 4.1, 4.1b, 4.3, 4.5, 4.7, 4.9, 4.10, 4.11, 5, 6, 8; `MANUAL.md` 9.1, 9.7, A1; `scripts/README.md`; `docs/UTRR.md`, `docs/NOISE_MODELS.md`, `docs/3GPP_DATASET.md`, `docs/WORKBENCH.md`; `docs/CLI_REFERENCE.md`; `VERSION.txt` | Updated. |

**Fixed on the way.** A run name with a dot lost everything after its dot in its file names, in v23 and older.
- **Where the dot comes from:** an override with a decimal value, e.g. `--set train.lr=0.0005` gives `_lr-0.0005`.
- **The networks:** `<run>_lr-0.pt` instead of `<run>_lr-0.0005_0to20dB.pt`. So `evaluate.py` and the suite's skip check did not find them.
- **The checkpoints:** the bands of such a run shared one checkpoint file.
- **v24** keeps the whole name (`bsce/export.py`, the trainer). Test `test_run_names_with_a_dot`.
- **An affected v23 run:** its files must be renamed by hand, or it is trained again. Overrides without a dot (`restore_best=true`, `steps=50000`, `K=0`) were never affected.

**Checks outside the tests**, on SV UPA on the CPU:
- **A real `kill -9` in the middle of a stage** (as a power cut), then the same command again. The resumed networks equal the uninterrupted ones exactly (difference 0.0):
  - NF-LADMM-SR T4, `steps`: killed after 45 s, resumed at step 186 of 600;
  - LADMM T2, layer-wise: resumed at iteration 121 of stage 9.
- **v23 and v24, same training without interruption:** identical weights (difference 0.0), LADMM and NF-LADMM-SR.

**Not changed.**
- The names of trained networks: your existing runs are found and skipped as before.
- Every training result: the checkpoints add saves, not computation. Checked: an uninterrupted run gives the same weights as in v23.

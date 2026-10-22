# 16 Oct 2026 (v25): early stop, so no training runs on without a gain

**Why.** Your e2e deep runs (`--protocol e2e`, 30,000 steps) kept training long after the validation had stopped improving:
- **LADMM-T12:** its best was at step 14,500, on SV ULA and on SV UPA. The next 15,500 steps (20 and 18 minutes) only made the validation worse, and `restore_best` threw them away.
- **NF-LADMM-C-T8:** it gained 0.004 dB in its last 9,000 steps.

Before v25, a `steps` training (end to end) always ran every step, and a fine-tune always ran its whole budget. You asked for a stop after a reasonable patience, everywhere this happens.

## The rules
| training | before v25 | v25 |
|---|---|---|
| **end to end, schedule `steps`** (`--protocol e2e`, `--schedule e2e`, NF-LADMM's own training) | every step | **stops when the best validation has not improved by 0.01 dB during the last 5000 steps.** <br>The best validated state is kept, as before. <br>`train.patience_steps` (5000; 0 = every step) and `train.min_gain_db` (0.01). |
| **fine-tune per band** (`finetune_bands.py`, the suite's fine-tune) | the whole budget | **4000 steps without a 0.01 dB gain:** <br>• in the first learning rate (5e-5), that rate ends and the second (5e-6) starts early; <br>• in the second, the band ends. <br>The best validated state is kept, as before. <br>`--patience` (4000; 0 = the whole budget), `--min-gain` (0.01); `run_suite.py --ft-patience`. |
| **layer-wise** (`ladmm_best`, `gmlamp`, `gmlamp_best`, `fair`) | a stage ends `better_wait` steps after its last new best | **unchanged.** <br>New option `train.min_gain_db=0.01`: only gains of 0.01 dB or more reset the wait. <br>Off by default, so the rule of the authors and of `untitled2.py` stays exact. |
| **`end2end` epochs** (UTRR) | `patience` epochs without a new best (its paper: 10) | **unchanged.** Same new option `train.min_gain_db`. |

**The improvement test.** A validation counts as a gain when it is at least `min_gain` dB below the value of the last gain (not just below the best). Many tiny gains therefore do not keep a training alive.

**The stop is printed and recorded:**
- **Training:** the console shows `early stop at step 19500: no gain of 0.01 dB or more since step 14500 (patience 5000 steps; train.patience_steps=0 runs all 30000 steps)`. `train_log.json` records `iters` = the stop step and `early_stop` = {step, last_gain_step, patience_steps, min_gain_db}.
- **Fine-tune:** the console shows `no gain of 0.01 dB or more since step …: next learning rate (5.0e-06) now`, or `… band finished early`. `finetune_log.json` lists `early_stops` for each band, and the README states the rule.

**Resume.** An interrupted training or fine-tune resumes exactly, early stop included: the reference value of the last gain is saved in the checkpoint.

## `--save-layers` with an end-to-end training: now refused
Your deep-run command combined `--save-layers` with `--protocol e2e`:
- **What happened:** an end-to-end training trains all layers together, so there is no network after layer t to save. Before v25, `train.py` printed one line about it (`[bsce] --save-layers: this schedule trains all layers together; nothing to save per layer`, in your log too) and went on, so the runs made the full-depth networks only, with no `layers/` folder.
- **v25:** `run_suite.py`, `train.py` and the workbench stop before any training with an explanation. The check covers every method, with its protocol, its own config and every `--set` override.
- **For an end-to-end depth curve:** train every depth itself, e.g. `--methods ladmm2 ladmm4 ladmm6 ladmm8 ladmm10 ladmm12 --protocol e2e`, without `--save-layers`.

## What changes
| file | change |
|---|---|
| `bsce/train/trainer.py` | `TrainConfig.patience_steps` (5000) and `min_gain_db` (None = 0.01 dB for `steps`, 0 for the others). <br>`last_gain_index()`. <br>The `steps` loop stops on the rule; the layer-wise wait and the `end2end` patience take `min_gain_db` when it is above 0. <br>The `end2end` partial checkpoint holds the reference value. |
| `scripts/finetune_bands.py` | `--patience`, `--min-gain`. <br>Non-default values are added to the folder name (`_pat8000`, `_mingain0p02`) and enter the setting fingerprint. Defaults do not, so a v24 fine-tune folder keeps its finished bands and resumes its running band in v25 (checked). <br>Checkpoint: `ref_db`, `ref_step`, `stops`. |
| `scripts/run_suite.py` | `--ft-patience N` (passes `--patience`; folder suffix `_pat<N>` when not 4000). <br>The `--save-layers` check before planning. |
| `scripts/train.py` | `--save-layers` with `steps` / `end2end`: exits with the explanation (was a print, then the run went on). |
| `app/bsce_app.py` | Fine-tune fields "patience" and "min gain". <br>The deep runs task refuses `--protocol e2e`, and so does every suite task that combines layer snapshots with an end-to-end schedule. |
| `tests/test_resume.py` | 5 new tests: <br>• the `steps` rule with scripted validations: where it stops, the best kept, 0 = off, another patience, another min gain; <br>• layer-wise with `min_gain_db` against the original rule; <br>• `end2end` with `min_gain_db`; <br>• `steps` early stop interrupted and resumed: identical to an uninterrupted run; <br>• the fine-tune early stop: both rates end, the log records it, the name gets `_pat…_mingain…`, and a fine-tune interrupted in the second rate resumes identically. |
| `tests/test_app.py` | The deep-run / e2e refusals; the fine-tune patience fields. |
| `GUIDE.md` 3.1, 4.1, 4.6; `MANUAL.md` 9.7, A1; `docs/WORKBENCH.md`; `docs/CLI_REFERENCE.md`; `VERSION.txt` | Updated. |

## Your e2e deep runs, replayed with the rule (`es_check/simulate_es.py` on the console log)
| scenario | run | stops at | best, full run → with the stop | NMSE lost | time saved |
|---|---|---|---|---|---|
| SV ULA | LADMM-T12_e2e | 19,500 | −12.617 → −12.617 dB | 0.000 dB | 14 min |
| SV ULA | NF-LADMM-S-T8_e2e | no stop (still gaining) | −16.881 dB | 0 | 0 |
| SV ULA | NF-LADMM-C-T8_e2e | 21,000 | −11.563 → −11.559 dB | 0.004 dB | 7 min |
| SV ULA | NF-LADMM-SR-T4_e2e | no stop (still gaining) | −16.573 dB | 0 | 0 |
| SV UPA | LADMM-T12_e2e | 19,500 | −13.532 → −13.532 dB | 0.000 dB | 12 min |
| SV UPA | NF-LADMM-S-T8_e2e | no stop (log ends at 29,000) | −13.660 dB | 0 | 0 |

- **Total:** 33 of the 403 logged minutes saved, for 0.004 dB on one network. LADMM-T12 stops 5000 steps after its best: of the 20 and 18 minutes after its best, the patience costs about 6 each.
- **With `min_gain_db=0.02`:** NF-LADMM-C stops at 19,000 (8 min saved, 0.006 dB lost); the other runs do not change.
- **Why 5000 / 0.01 dB:**
  - **Shorter patience:** the NF-LADMM-S curve still gains about 0.015 dB per 5000 steps near the end, and a shorter patience would cut that.
  - **Larger min gain:** buys little more time for more NMSE lost.

## Not changed
- **Networks trained before v25** had no early stop. The suite skips them as before and they stay valid; nothing has to be retrained.
- **The layer-wise and UTRR rules** are unchanged unless `train.min_gain_db` is set. Every layer-wise network trains bit for bit as in v24.
- **Folder names** are unchanged: the defaults are not in them. A v24 checkpoint (training or fine-tune) resumes in v25. A `steps` training resumed this way stops early if the rule says so.

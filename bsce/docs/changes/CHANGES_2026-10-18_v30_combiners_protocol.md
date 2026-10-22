# v30 (18 Oct 2026): any number of RF chains, a combiner check, Graph-LADMM's own training, and names

Every default is unchanged, except one: in `run_suite.py` (and the workbench), `gladmm` and `gladmmany` now always train
with their own end-to-end settings. Folder names, checkpoint fingerprints and `--methods all` are unchanged, and v29
runs resume and evaluate as before.

## Combiner with another number of RF chains
- **`set_combiner(A)`** of the analytic networks (Graph-LADMM `xstep=analytic`; LADMM `operator=analytic|analytic1`)
  takes any M × N combiner, with the same N.
  - No learned parameter depends on M. The buffers of another shape are registered again (`UnrolledNet._put_buffer`),
    so a network saved after the change reloads with the new shape.
  - It refuses another N, and a combiner with dependent rows.
- **`evaluate.py`:**
  - `--combiner random:<seed>:<M>` gives a random ±1/√M combiner with another M. `random:<seed>` gives the v29 combiner,
    the same draws.
  - A network learned for one combiner cannot run with another M. It is now skipped, with a note in `eval_log.json`
    ("cannot run (learned for M = 128; the test combiner has M = 64)"), instead of stopping the evaluation.

## New script: `scripts/check_combiners.py`
Every network of a runs folder is tested on nine test combiners built with fixed seeds (the same on every PC). The
results go into one table and one figure (GUIDE 4.13, MANUAL §10).

| combiner | what it is |
|---|---|
| `own` | the scenario's combiner |
| `pm1` | a new random ±1 draw (= `--combiner random:11`) |
| `gauss` | Gaussian |
| `corr` | rows correlated 0.9 |
| `sparse` | 16 beams per RF chain |
| `dct` | a DCT-II row subset |
| `M+` / `M-` / `M--` | random ±1 with 1.25 / 0.75 / 0.5 × M |

- Options: `--combiners` (a subset), `--combiner-files`, `--graph-phase`, `--tags`, `--exclude`, `--classical`,
  `--n-test`, `--out`.
- Output: `combiner_check.md` / `.csv` / `.png` and `combiners.csv` (M, cond(AAᵀ), coherence), plus the `evaluate.py`
  folder of every combiner.

## Graph-LADMM's training in the suite
- **`run_suite.py`:** `gladmm` and `gladmmany` are trained with their own settings (`configs/train/gladmm*.yaml`: end to
  end, online noise, log-NSE) whatever `--protocol` is.
  - Their folders end in `_paper`, as with `--protocol paper` in v29.
  - `--force-protocol` gives them `--protocol`. `gladmmf` takes `--protocol` like every method.
- **The reason** (check of 18 Oct, SV ULA, 56 parameters): at the same budget,
  - GM-LAMP's layer-wise procedure gave −14.95 dB on average against −16.66 dB end to end, worse at every SNR;
  - end to end, the network reached its plateau at about 4,500 steps; with 9,000 steps it gave −16.85 dB, against
    −16.87 dB for the authors' two GM-LAMP networks.
- **Workbench:** a box "Same protocol for the analytic Graph-LADMM rows too" (= `--force-protocol`); the method rows say
  this and the `train.steps=9000` short run.

## Comparison
- **`compare.py --preset ft|onenet`:** the workbench's selections.
  - `onenet` takes every one-network-0–20 dB run, whatever its suffix (`G-LADMM-T8_paper`, `LADMM-T6_fair`).
  - The GUIDE's and Spyder's old `--tags "*-T[0-9]" "*-T[0-9][0-9]"` missed those runs. GUIDE §5 and Spyder cell 5 now
    use the preset.
- **Colours:** every method gets its own colour. Before, the palette repeated after 14 methods.
- **`networks.md`:** it gives the model options (z-step, x-step, operator, threshold), the budget of end-to-end runs
  (steps), the loss, the noise and a random combiner.

## Workbench and Spyder
- **Another combiner or data set:** a random combiner with another M.
- **New task "Many test combiners at once"** (`check_combiners.py`).
- **The quick tests** include `test_v30_combiners_names.py`.
- **Spyder:**
  - cell 4.1 notes Graph-LADMM's own training;
  - cell 5c adds an M = 64 test;
  - a new cell, 5d, runs the combiner check.

## Checks
`tests/test_v30_combiners_names.py`:
- **any-M `set_combiner`:**
  - equals a network built on that combiner (≤ 1e-12), for M = 64 / 96 / 160 and four analytic networks;
  - save and reload keep it;
  - the refusals (free H, LADMM full, another N, dependent rows) leave the network unchanged.
- **`evaluate.py` with another M:** the followers run and the others are skipped; `random:<seed>` is unchanged; a bad
  syntax stops with a message.
- **`check_combiners.py`:** the same combiners on every call, `pm1` = `random:11`, the table, CSV, figure and
  "cannot run" cells.
- **The suite:**
  - the protocol and folder names of `gladmm`, `gladmmany`, `gladmmf` and `ladmm4`;
  - `--force-protocol`, `--protocol paper`, the `train.steps=9000` folder, and an ensemble's name.
- **`compare.py`:** its presets equal the workbench's; 40 distinct colours.
- **Names:**
  - method keys and folder labels are unique;
  - no two methods share a folder at any depth and protocol;
  - every train config exists;
  - script and config names are unique.

`tests/test_app.py` gains `test_v30_forms`:
- the new box, the other-M transfer and the combiner-check task;
- task ids and field keys unique;
- every method row described;
- every script of a command exists.

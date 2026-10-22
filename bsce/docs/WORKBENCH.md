# The bsce workbench: the whole workflow in one window

The workbench is a small local web app. It covers every step of `GUIDE.md`, from creating the conda environment
to the final figures. Each step is a form with all the options of its script. While you change an option, the
page shows the exact commands that will run. You start them with one button and follow their progress on the
same page.

It uses only the Python standard library, so the base Anaconda Python can start it before the `bsce`
environment exists (step 0 creates the environment). It listens on `127.0.0.1` only, and every request needs a
key that is created anew each time the app starts.

## Start it

| how | what to do |
|---|---|
| double-click | `start_app.bat` in the `bsce` folder. It finds Anaconda's Python and opens the browser. |
| Anaconda Prompt | `cd "D:\Academic Profile\LADMM-MIMO\codes\bsce_final\bsce"` then `python app\bsce_app.py` |
| other port, no browser | `python app\bsce_app.py --port 8800 --no-browser`, then open the address it prints |

Keep the black window open while runs are going: closing it stops the app and its runs. Closing the browser tab
does not stop anything. Open the address the black window printed again (normally `http://127.0.0.1:8765/`), and
the Runs page shows where things are.

**Which workbench is this?** The top bar shows the version (`v21`) and the last two parts of the bsce folder. The
browser tab says the same, and Settings shows the full folder. Every bsce folder has its own workbench, with its own
runs and settings.

**An older workbench is still open.** Each workbench takes its own port.
- **From v21:** a new workbench never shares a port with one that is still running. If 8765 is taken, it opens on
  8766, and its page shows a yellow note: "Another bsce workbench is still running at http://127.0.0.1:8765/ …".
  - "Open it" shows the old one.
  - "Stop it" stops it, with the steps it is running. Its trainings keep their checkpoints in its own `outputs`
    folder: run the same step again from that folder (or copy its `outputs` here) and they resume.
- **Before v21 (v16–v20), on Windows:** a second workbench could take the same port as one still running. The
  browser then showed the old workbench: its steps, its runs and, in Settings, its folder. Close the old black
  windows first.
  - If you cannot find one: Anaconda Prompt, `netstat -ano | findstr :8765`. The last number of the line is the
    process id. Then `taskkill /PID <id> /T /F` (this also stops the steps it runs).

## The page

```
+---------------------------------------------------------------------------------------------+
| bsce workbench   Python  C:\...\envs\bsce\python.exe (conda environment 'bsce')   1 running  |
+----------------+------------------------------------------------+---------------------------+
| Workflow       | [4] Train every method                         | Commands   [Spyder|Term.] |
|  0 Environment |  Train and fine-tune | Fine-tune one network   |  1 train + fine-tune ...  |
|  1 Data        |  Scenarios   [x] SV ULA [x] SV UPA ...         |  %run scripts/run_suite.. |
|  2 Paper       |  Methods     [x] LADMM  6 4   [x] LAMP 8 ...   |  Run it in: Spyder ...    |
|  3 Depth       |  Protocol    ladmm_best                        |                           |
|  4 Train       |  Schedule / clipping / SNR / seeds / device    |  [Add to queue] [Run now] |
|  5 Compare     |  Extra settings (key=value)                    |  [Preview the plan]       |
| Runs           |                                                |  Last run of this task    |
| Networks       |                                                |  ■■■□□□  layer 3 of 6 ... |
| Results        |                                                |                           |
+----------------+------------------------------------------------+---------------------------+
```

- **Left: the steps of GUIDE.md.** A step's number turns blue once one of its runs has finished. It turns amber
  while one runs, and gets a red outline when its last run failed.
- **Middle: the form.** The options of the step that have a field, with their defaults. Any other `key=value` goes in "Extra settings" (every ticked method) or in a method's own settings cell (that method only); MANUAL A1/A2 list the keys. Only options that apply are shown: for
  example "End-to-end steps" appears once the schedule is end to end. Values are remembered in this browser;
  "Reset to defaults" restores them.
- **Right: the commands.** They are built by the same code that runs them, so what you see is what runs. Only
  options that differ from the script's default appear, so the default form gives exactly the command of
  GUIDE 4.1. "Spyder" shows them as `%run scripts/...` for the Spyder console, "Terminal" as `python scripts/...`.
  Every command has a Copy button, and "Run it in" says where to paste it if you prefer to run it yourself.
- **Add to queue** runs the commands after the runs already waiting, one at a time (one GPU).
- **Run now, in parallel** starts at once, next to the queue (e.g. data or a CPU comparison while the GPU trains).
- **Preview the plan** (train and depth steps) runs `run_suite.py --dry-run`: lines with `>>>` will run, lines
  with `---` are done already and will be skipped.

## The steps and their tasks

| step | tasks | script |
|---|---|---|
| 0 Set up the environment | create or update the conda environment, PyTorch build (cu126 for the P2000), optional Spyder and verification packages, GPU check; check this PC; run the tests | `conda`, `pip`, `check_env.py`, `pytest` |
| 1 Make the data | the GM-LAMP data sets; raw DeepMIMO channels with MATLAB (one file or chunks); the 3GPP TR 38.901 data sets (built-in, Python only: size preset or exact sizes, arrays, tables, LOS/NLOS, check); raw 3GPP channels with Sionna or with QuaDRiGa (MATLAB or Octave), as second implementations; a data set of any size (SV, DeepMIMO or 3GPP; size presets, the authors' bands; colored, white or both noise models for the measurement files); tiny smoke data | `make_gmlamp_data.py`, `make_3gpp_data.py`, `deepmimo_make_H(_chunks).m`, `sionna_make_H.py`, `quadriga_make_H.m`, `make_dataset.py`, `generate_data.py` |
| 2 Reproduce the paper | the authors' networks on our test sets (expected numbers shown); retrain LAMP and GM-LAMP with the paper's procedure, then compare, also stage by stage with the authors' training logs; check the data against the authors' files | `compare.py`, `run_suite.py --protocol paper`, `check_vs_authors.py` |
| 3 Choose the number of layers | deep runs with a snapshot after every layer; depth curves; promote a depth without retraining | `run_suite.py --save-layers`, `depth_snapshots.py`, `promote_layer.py` |
| 4 Train every method | every method with the authors' protocol and data (`--protocol gmlamp --train-snr scenario --ft-snr none`, float32 or float64); the data of a band (measurement noise colored or white, validation noise over the band or at fixed SNRs, edge margins, SNR chunks, training SNRs: chunks or fair, with a grid step); the suite (methods and depths, protocol, schedule, clipping, training SNR, fine-tune bands and budget, seeds, device, output folder, any `key=value`, smoke); fine-tune one network with any budget, bands, learning rates, clipping, training SNRs (chunks / fair) or early stop (patience 4000 steps, min gain 0.01 dB; 0 = the whole budget) (the name is automatic: `<network>_ft_<edges>dB_<iterations>`, next to it); what is done and where an interrupted run resumes; old fine-tune names to the new ones | `run_suite.py`, `finetune_bands.py`, `runs.py`, `rename_runs.py` |
| 5 Compare and plot | compare methods (fine-tuned pairs, runs trained with the authors' protocol, one-network runs, all, or a custom list; reference; classical baselines; folders; noise: colored, white, or both with cross tests); mean ± std over seeds | `compare.py`, `compare_noise.py`, `aggregate_seeds.py` |
| Any script | any script of `scripts/` with your own arguments (`--help` first) | |

"Commands" in step 4 can split the work into one command per method or per scenario. This makes the progress
easier to follow, and it lets you copy half of the commands to a second PC (GUIDE 4.4).

## Following a run

The top bar always shows how many runs are going and waiting. With an NVIDIA GPU it also shows the GPU load,
memory and temperature (from `nvidia-smi`, every 4 s).

The **Runs** page lists every run started from the app, with its console output. The output is also saved in
`app/runs/<id>.log`.

The grid of squares is the plan of the run: one square per step, e.g. "train LADMM-T6 on SV_UPA", then
"fine-tune LADMM-T6_ft_0-10-20dB_20k on SV_UPA". For a suite command the plan comes from its dry run, so steps that are
already finished show as light squares ("done earlier, skipped").

| square | meaning |
|---|---|
| blue | finished |
| light blue, outlined | done earlier, skipped |
| amber | running |
| red | failed |
| empty | to do |

Below the grid, for the running step:

| line | layer-wise training | end-to-end training | fine-tune |
|---|---|---|---|
| progress | layer t of T, stage k of K | step s of N | band i of n, step s of the budget |
| last validation | NMSE and the best so far | NMSE and the best so far | NMSE and the best so far |
| time left | estimate for this step | estimate for this step | estimate for this step |

A failed run shows the exit code and its last line of output.

- **Stop** ends the run and all its child processes. Training checkpoints are kept.
- **Run again** starts the same commands. Finished steps are skipped and an interrupted training resumes from
  its checkpoint, as on the command line.

**Trained networks** lists every folder of `outputs/` (or `outputs_*`): its bands, how many per-layer snapshots
it has, the fine-tune budget and the best validation NMSE of its log. A network that is still training shows as
"in progress". **Results** shows `results/<name>/`: the summary table, the figures, and every file.

## Settings

The Python line at the top says which Python runs the steps. By default it is the Python of the environment
named in Settings (`bsce`). Before step 0 it is the app's own Python. In Settings you can also give the Python
or conda path by hand, the default device, and whether commands are shown for Spyder or the terminal.
Settings are saved in `app/settings.json`.

## Notes

- Spyder: `%run` expands `*` itself, so commands with patterns are shown as `%run -G ...`, which passes the
  pattern to the script unchanged (the scripts expand it themselves).
- The environment step runs `conda` and the new environment's `python -m pip` directly, without activating
  anything. If conda is not found, start the app from the Anaconda Prompt or set the conda path in Settings.
- The app only reads the folders and runs the scripts of this package. Nothing is installed except in step 0.
- Tests: `python -m pytest tests/test_app.py -q`. They check that every option exists in its script, that the
  default form gives the documented commands, the progress reading, and that the server refuses requests without
  its key, from another host name, and for files outside the folder. One test starts an older workbench in another
  folder, then the new one on the same port: the new one takes the next port, names the old one, and stops it.

**Early stop and layer snapshots (v25).** The deep runs task needs a layer-wise protocol. It and every suite task refuse layer snapshots with an end-to-end schedule before any training: such a training has no network after layer t. End-to-end trainings and fine-tunes stop when they no longer gain (GUIDE 4.6).

**v27.**
- **Methods table:** every row has a one-line description (what the method is, what to know: UTRR needs the protocol `paper`, NF-LADMM gets the true noise variance unless `model.sigma=residual`) and its own settings cell (`key=value; key=value`). Methods with their own settings get their own suite command (and folder names with their overrides); the others share one.
- **Fine-tune one network:** a depth snapshot (`layers/T06_…`) can be chosen; new fields "Data set", "Measurement noise", "Validation at fixed SNRs", "Edge margin". The suite has "Fine-tune: stop a learning rate after … steps" (`--ft-patience`).
- **New tasks:** step 2 "UTRR: reproduce its Table II" (`utrr_reproduce.py`), step 5 "Ensemble of trained networks" (`make_ensemble.py`).
- **Any script:** "Show its options" prints the script's `--help`.
- **Fixed:** the Networks view shows the "in progress (resumable)" style; the compare preset "one network 0–20 dB" takes every trained run with any suffix (`_fair`, `_paper`, `_e2e30k`, …) and leaves out fine-tunes and other training ranges; the quick tests of step 0 include the tests of v22–v27.

**v28.**
- **Methods table:** the LADMM row lists the new model options (`model.threshold=adaptive|firm|gm`, `model.data_step=whitened`, `model.operator=analytic|analytic1`); put them in the row's own settings cell. A new row `cadmm` (complex ADMM-Net with untied matrices), not ticked by default.
- **Train and fine-tune (suite):** "Ensemble after training (optional)": method keys such as `ladmm6 ladmm8`; empty (the default) = no ensemble.
- **Compare methods:** "Polish": off (default), per network band, or per test SNR. It adds a column `<run>+polish` for every network.
- **Quick tests** of step 0 include `test_v28_options.py`.

**v29.**
- **Methods table:** three Graph-LADMM rows, not ticked: `gladmm` (any combiner), `gladmmf` (free H_t) and `gladmmany` (trained on random combiners). Train them with the protocol "paper".
- **Compare and plot:** a new task, "Another combiner or data set". It runs `evaluate.py` on a runs folder with another data set (its test channels and its combiner) or a new random combiner, and with the target's edge phases for Graph-LADMM.

**v30.**
- **Train and fine-tune (suite):** the analytic Graph-LADMM rows (`gladmm`, `gladmmany`) always train end to end with their own settings (folders `…_paper`), whatever the protocol of the form: the layer-wise procedure measured 1.7 dB worse for them. The new box "Same protocol for the analytic Graph-LADMM rows too" gives them the form's protocol. For a short run, put `train.steps=9000` in the row's settings cell (SV ULA reached its plateau there).
- **Another combiner or data set:** a new choice, "a new random one with another number of RF chains", with its M. The networks that follow work with any M; the others are skipped there.
- **New task "Many test combiners at once"** (`check_combiners.py`): every network of a runs folder on nine fixed test combiners (random, Gaussian, correlated, partially connected, DCT, and three other M), in one table and one figure.
- **Quick tests** of step 0 include `test_v30_combiners_names.py`.

**v31.**
- **Methods table:** four new Graph-LADMM rows, not ticked. Like `gladmm`, all train end to end with their own settings.
  - `gladmmmix`: trained on a mixture of combiners (M 64–160; random, Gaussian, partially connected, DCT);
  - `gladmmbp`: + the beam prior;
  - `gladmmrw`: the uploaded package's graph-reweighted l1, unrolled;
  - `gladmmrwtv`: the same in gladmm's layer.
- **The `gladmm` row says 54 parameters.** Every one is trained; the two that never reached the output are no longer
  stored.
- **Compare methods:**
  - the classical baselines add "l1 ADMM, tuned" and "graph-reweighted l1, tuned", which need the new task below first;
  - a box "Leave out the trained-parameter and MACs-per-channel columns". Without it, the tables give both.
- **New task "Tune the classical l1 baselines"** (`tune_classical.py`): λ and ε per test SNR on validation channels,
  saved to `results/classical_params_<SCENARIO>.json` for the compare task.
- **Another combiner / many test combiners:** NF-LADMM, LVAMP and ALISTA now follow the new combiner (any M). A
  diverged network is printed as "diverged", and the check goes on past a combiner where nothing can run.
- **Quick tests** of step 0 include `test_v31_rwl1.py` and `test_v31_release.py`.


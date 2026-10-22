# 15 Oct 2026: the bsce workbench (a graphical front end for the whole workflow)

**What it is.** A local web app with every step of `GUIDE.md` as a form, from creating the conda environment to the
final figures. While you choose options it shows the exact commands, runs them in a queue (or in parallel), and
follows their progress.

**How to start it.**
- Double-click `start_app.bat`, or
- in the Anaconda Prompt, run `python app\bsce_app.py`.

The browser opens on `http://127.0.0.1:8765/`. Full user guide: `docs/WORKBENCH.md`. Screenshots:
`docs/workbench/*.png` (taken in a Linux test container, so the Python line shows `/usr/bin/python3`).

**What did not change.**
- No script, model, config or default changed.
- The default form gives exactly the documented commands. For example, step 4's default command is GUIDE 4.1's
  `run_suite.py --scenarios sv_ula sv_upa deepmimo_ula deepmimo_upa --methods ladmm6 ladmm4 lamp8 gmlamp8
  nfladmmsr4`, and its dry run is identical (test).

## Files

| file | what |
|---|---|
| `app/bsce_app.py` | Server (Python standard library only), the steps and their options, the command builders, the run queue, progress reading, folder scanners. |
| `app/static/index.html`, `app.css`, `app.js` | The page (no framework, no build step, no internet needed). |
| `start_app.bat` | Finds Anaconda's Python (user, ProgramData, LocalAppData, miniconda, miniforge) and starts the app. |
| `tests/test_app.py` | 23 tests (below). |
| `docs/WORKBENCH.md`, `docs/workbench/*.png` | User guide and screenshots. |
| `GUIDE.md`, `README.md`, `MANUAL.md` | A short pointer to the workbench, and the `app/` folder in the layout. |

## What it covers

| step | tasks (every option of the script is a field) |
|---|---|
| 0 | Create or update the conda environment, plus: <br>• the PyTorch build: cu126 (P2000), cu124, cu118, cu128 or CPU; <br>• optional verification packages and Spyder; <br>• a GPU check. <br>Also: check this PC; run the tests (quick, quick + reference, all). |
| 1 | The GM-LAMP data sets; raw DeepMIMO channels with MATLAB (one file or chunks); a data set of any size (all `make_dataset.py` options); smoke data. |
| 2 | The authors' networks on our test sets (expected values shown); retraining with the paper procedure (float32 option), then comparison. |
| 3 | Deep runs with per-layer snapshots, depth curves, promoting a depth. |
| 4 | The suite: <br>• methods table with depths; <br>• protocol, schedule (layer-wise or end to end, with its steps) and gradient clipping; <br>• training SNR, fine-tune bands and budget; <br>• per-layer snapshots, seeds, device, output folder, any `key=value`, smoke; <br>• one command, or one per method or per scenario. <br>Also: fine-tune one network with all `finetune_bands.py` options. |
| 5 | Compare: <br>• which runs: the fine-tuned pairs, one-network runs, all, or a custom list; <br>• the reference, the authors' networks, the classical baselines, and the folders. <br>Also: mean ± std over seeds. |
| extra | Any script with free arguments. <br>Runs: queue, live console, progress grid, stop, run again. <br>Trained networks: `outputs*/`, with bands, snapshots, fine-tune budget and best validation. <br>Results: summary table, figures and files of `results/<name>/`. <br>Settings, and a GPU monitor (nvidia-smi). |

## Design decisions

- **The server builds the commands, the page only shows them.** The command you copy is the command that runs.
- **Only options that differ from a script's default are written.** The default form therefore reproduces the
  documented commands, and a changed option is easy to spot.
- **Spyder or terminal view.**
  - Spyder view: `%run scripts/...`. When an argument contains `*`, it becomes `%run -G`, because IPython's `%run`
    would otherwise expand the pattern against the bsce folder.
  - Terminal view: `python scripts/...`.
  - Environment commands are always shown for the Anaconda Prompt, and each command says where to paste it.
- **No dependency.** The base Anaconda Python can run the app before the `bsce` environment exists:
  - it does not import `bsce`;
  - it reads the methods list from `scripts/run_suite.py`.
- **Runs are processes started by the app:**
  - `PYTHONUNBUFFERED=1`, `MPLBACKEND=Agg`;
  - output saved in `app/runs/<id>.log`;
  - Stop kills the whole process tree (`taskkill /T /F` on Windows, process group on Linux);
  - after the app restarts, the list is kept and unfinished runs show as interrupted;
  - "Run again" relies on the scripts' own skip and resume.
- **Progress** is read from the scripts' existing console lines, so no script was changed:
  - the step plan comes from `run_suite.py --dry-run` (`>>>` = to run, `---` = done earlier);
  - layer-wise training: layer t of T and stage k of K (K is counted on layer 1);
  - end to end: step s of `train.steps`;
  - fine-tune: band i of n and step s of `--iters`;
  - the last validation NMSE, and the time left in the current step.
- **Local only:**
  - it listens on 127.0.0.1;
  - every API call needs a random key made at start (CSRF);
  - the Host header must be 127.0.0.1 or localhost (DNS rebinding);
  - it serves only result-type files inside the bsce folder.

## Verification

**`tests/test_app.py`: 23 tests, all pass.**
1. Every task's command, with default and with filled-in values, uses only options listed in the script's
   `--help` (all 17 tasks).
2. The default step-4 command equals GUIDE 4.1, with an identical dry run.
3. Every suite option reaches `run_suite.py`: folder suffix `_fair_tr0-10-20_e2e50k_clip0p5_restore-best-true`,
   `train.steps`, `train.grad_clip`, `--save-layers`, no fine-tune with `none`, per-method split × seeds. Bad
   ranges and an empty method list are refused with a message.
4. The Spyder view adds `-G` only for patterns.
5. Progress reading (layers and stages, end-to-end steps, fine-tune bands and steps), and the dry-run plan marking
   a finished step as skipped.
6. The server refuses no key, a wrong key, a foreign Host header and paths outside the folder, and cannot start a
   run without the key.

**Tests without `reference/`.** The quick set of step 0 (62 tests, including `test_app.py`) passes on a clean copy
of the package without `reference/` and `data/`.

**End to end in a browser** (Chromium driven by Playwright, the real server and scripts, CPU):
- **Train:** step 4, SV UPA, LADMM / LAMP / NF-LADMM-SR at T = 3, smoke, split per method → 6 steps.
  - The grid filled step by step.
  - Layer, stage and validation were read correctly ("train LAMP-T3 on SV_UPA, layer 2 of 3, stage 1 of 10").
  - The comparison queued behind it started when the training finished.
- **Compare:** the results page showed the summary table and the figures.
- **Stop:** stopping a running suite left no `train.py`, `finetune_bands.py` or `run_suite.py` process.
- **Errors:**
  - a bad argument failed with the script's message shown;
  - step 0 without conda failed with "conda not found: start this app from the Anaconda Prompt, or set the conda
    path in Settings".
- **Layout:** 1440, 1100 and 390 px wide, with no horizontal scroll, plus dark mode. No console errors.

**Not testable in this container (Windows-specific; please check on your PC):**
- conda detection and the environment step;
- `taskkill`;
- `start_app.bat`;
- `%run -G` in your Spyder;
- the real `nvidia-smi` (simulated here).

## Limits

- One queue (one GPU). "Run now, in parallel" is the way to run something next to it.
- Runs started outside the app (e.g. in Spyder) do not appear in Runs. Their folders do appear in Trained networks
  ("in progress" while a checkpoint exists).
- Closing the app window stops its runs. Start it again and use Run again: finished steps are skipped and the
  training resumes from its checkpoint.

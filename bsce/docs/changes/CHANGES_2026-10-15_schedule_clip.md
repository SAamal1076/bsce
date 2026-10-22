# 15 Oct 2026 (final + 3): `--schedule` and `--grad-clip` for every method; seed aggregation fix

Defaults are unchanged: without the new options, `run_suite.py` makes exactly the same commands and folder names as before (tested).

## New options of `run_suite.py` (GUIDE.md section 4.6)
| option | effect | folder suffix |
|---|---|---|
| `--schedule layerwise` (default) | as before: the protocol decides (layer by layer for ladmm_best / gmlamp / gmlamp_best / fair) | – |
| `--schedule e2e [--e2e-steps 30000]` | every method end to end: Adam, cosine lr → 0.05·lr, validation every 500 steps, best state kept. lr, batch, loss, noise and precision stay the protocol's. | `_e2e30k` |
| `--grad-clip X` | every method clipped at X, in training and in the fine-tune | `_clip1`, `_clip0p5` |
| `--grad-clip none` | no clipping for any method, including where a method's own config clips (`--protocol paper`, `e2e`) | `_noclip` |

- `finetune_bands.py` has a new `--grad-clip`. Its default is no clipping, as before; the suite passes the option on.
- `README.txt` of every run now lists `loss`, `noise` and `grad_clip` next to the schedule.

## Fix
- `aggregate_seeds.py` grouped seeds only when `_s<k>` was at the end of the name.
- The suite names fine-tuned seed runs `<tag>_s1-ft`, so mean ± std was never formed for the fine-tuned networks.
- It now groups `LADMM-T6-ft`, `LADMM-T6_s1-ft` and `LADMM-T6_s2-ft` (checked: n = 3).

## Suggested ablations
GUIDE.md section 4.7: A1–A9, each changing one thing against the default suite.

## Checks
| check | result |
|---|---|
| `tests/test_suite_options.py` (new, 6 tests) | pass. It covers four things: the default commands are unchanged; the e2e + clip overrides and suffixes; `--grad-clip none`; the merged e2e config is identical for all 10 methods under 3 protocols |
| `tests/test_review_fixes.py`, the NF-LADMM tests and 7 further test files | 86 passed in total |
| smoke suite, `--schedule e2e --grad-clip 1.0`, with LADMM, LAMP, GM-LAMP, ALISTA, ADMM-Net and NF-LADMM-SR | all trained and fine-tuned. The saved configs show `schedule=steps`, `grad_clip=1`, and the protocol's lr, batch and loss; the fine-tune log shows `grad_clip=1.0` |
| `finetune_bands.py --tag ...-ft10k` from an existing network (ablation A9) | runs, and writes next to the source |

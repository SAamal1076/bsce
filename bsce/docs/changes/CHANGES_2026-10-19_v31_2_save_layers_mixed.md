# v31.2 (19 Oct 2026): `--save-layers` with methods that train end to end

**The report.** A workbench deep run stopped after 1 s:

```
run_suite.py --methods ladmm12 lamp12 gmlamp12 gladmm8 gladmmf6 gladmmrw8 nfladmmsr4 --save-layers ...
--save-layers saves the network when each layer is finished, but these methods train all their layers together ...
  gladmm8 (steps), gladmmrw8 (steps)
```

- **The cause.** `gladmm`, `gladmmany`, `gladmmrw`, `gladmmrwtv`, `gladmmbp` and `gladmmmix` always train end to end,
  with their own settings (v30), so they have no network after layer t to save. From v25, `run_suite.py` stopped the
  whole list when any method in it trained end to end.
- **v31.2.** A list that mixes the two kinds now runs:
  - **The layer-wise methods** get `--save-layers` and save their layers.
  - **The end-to-end ones** train at full depth, without `--save-layers`, and the suite prints which they are.
  - **A finished end-to-end run** gets no "trained without --save-layers" warning.
- **Still refused.** A list in which no method trains layer by layer stops before any training, as before, because
  nothing would be saved. `train.py --save-layers` with an end-to-end schedule is refused too.
- **The message** now names each method's own protocol (`gladmm8 (steps, protocol paper)`). Before, it named the
  suite's protocol (`ladmm_best`).

**The user's command**, dry run with v31.2:
- with `--save-layers`: LADMM-T12, LAMP-T12, GM-LAMP-T12, G-LADMM-F-T6, NF-LADMM-SR-T4;
- full depth only: G-LADMM-T8_paper, G-LADMM-RW-T8_paper.

**For a Graph-LADMM depth curve,** train every depth itself, e.g. `--methods gladmm4 gladmm6 gladmm8 gladmm10 gladmm12`.

## Checks
- `tests/test_v31_release.py::test_suite_save_layers_with_end_to_end_methods` (new):
  - a mixed list gives `--save-layers` to exactly the layer-wise methods;
  - an all-end-to-end list is refused.
- The full suite: 428 passed, 85 skipped, 0 failed.

## Docs
- GUIDE 3.1;
- GUIDE 4.12: the depth check of 19 Oct (Graph-LADMM with T = 8 / 10 / 12 on SV ULA and SV UPA);
- MANUAL §8.1 and the early-stop list;
- `run_suite.py --help`;
- the comment in Spyder cell 3.1;
- CLI_REFERENCE.

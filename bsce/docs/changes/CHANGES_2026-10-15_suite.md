# 15 Oct 2026 (c): one protocol for LADMM, LAMP and GM-LAMP (`run_suite.py`); GM-LAMP in float32

## New: `scripts/run_suite.py`
For each scenario × method, the suite runs:
1. `train.py`: one network for 0–20 dB, `ladmm_best` protocol;
2. `finetune_bands.py`: 0–10 / 10–20 dB, 15k iterations.

- Methods: `ladmm6`, `ladmm4`, `lamp` (T = 8), `gmlamp` (T = 8). Any depth is accepted, e.g. `lamp6`.
- Finished steps are skipped. `--dry-run` prints the commands.

## Fixed: GM-LAMP + `ladmm_best` protocol (float32) stopped with `nmse is NaN`
- **Cause:** the GM shrinkage normaliser V = Σ P_k N_k underflows in float32 (each term can be ≈ e^-40), so the gradient of U/V is NaN. It happened at the first step of layer 2.
- **Fix (`bsce/models/shrinkage.py: gm_wei`):** for complex64, every component is divided by exp(max_k exponent).
  - x̂ = U/V and the divergence terms are ratios of the same degree in the components, so they are unchanged in exact arithmetic.
  - complex128 uses the original arithmetic, untouched.

## Checks
| check | result |
|---|---|
| float32 vs float64 GM shrinkage (outputs and both divergence terms), random input | relative difference 1.6e-7, 2.5e-7, 1.1e-4 |
| float32 gradient with inputs 40× larger | finite |
| GM-LAMP T = 2, `ladmm_best` protocol, float32, SV UPA, short budget | trains through layer 2 (was NaN), saved |
| `tests/test_lamp_vs_tf.py`, `test_golden.py`, `test_lamp_vs_matlab.py` | 59 passed (float64 path unchanged) |
| LAMP + `ladmm_best` protocol, then `finetune_bands.py` | runs, pair saved |
| `run_suite.py --dry-run` | 8 commands per scenario, as documented |

# Why QuaDRiGa's azimuth spreads are wider than Sionna's: the cluster-angle mapping

**The test.** QuaDRiGa 2.8.1 draws the cluster angles its own way (`alt_AS_mapping = true` in
`@qd_builder/gen_ssf_parameters.m`): it takes random angles and scales them, so that every user's angle spread equals the drawn
large-scale parameter. Its "strict method as defined in the TR" (eq. 7.5-9 to 7.5-19) is in the same file, but no
setting selects it.

For this test it was switched on in a **local copy** of `quadriga_src`. The copy keeps the folder name, which QuaDRiGa
needs. The patch is one line, `alt_AS_mapping = false;`. The channels were then made with
`quadriga_strict_test.m`: 2,000 per array, with the three elevation fixes of `matlab/quadriga_make_H.m`.
```matlab
quadriga_strict_test('.../strict/quadriga_src', 'ULA_Y', 'strict_ulay.mat', 2000, 1, 1)
```

**Results** (`results.json`; beams holding 95 % of the energy):

| array | QuaDRiGa angles + fixes | QuaDRiGa strict TR angles + fixes | Sionna |
|---|---|---|---|
| ULA_Y (azimuth) | 99 (8–142), NLOS 116 | **63 (6–127), NLOS 93** | 63 (6–118), NLOS 86 |
| UPA | 24 (10–48), NLOS 26 | 29 (10–54), NLOS 34 | 17 (8–38), NLOS 16 |

**ULA_Y.** With the TR's angle formulas, QuaDRiGa's horizontal ULA has Sionna's sparsity. The azimuth difference is
therefore the angle mapping.

**UPA.** The strict branch does not bring the UPA closer. That branch aligns the first cluster with the LOS direction
for every user. In NLOS that first cluster is QuaDRiGa's zero-power LOS slot (eq. 7.5-12 / 7.5-17 do this for LOS only).
This rotates the NLOS cluster set by a random large angle, so the branch is not a faithful TR implementation either.
It was used only for the azimuth test above, and not for any data set.

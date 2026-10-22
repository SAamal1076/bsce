# v28 (17 Oct 2026): the review's recommended additions, all off by default

Source: the review of the audit and the ADMM-gap reports (`LADMM_audit_review_and_plan_2026-10-17.md`). The Colab code line is left out, as asked. Every default is unchanged: training, folder names, checkpoint fingerprints and `--methods all`. Runs of v27 resume and evaluate as before.

## New options

| option | where | what it does | parameters |
|---|---|---|---|
| `model.threshold=adaptive` | LADMM | soft threshold c_t·σ̂. σ̂ = median_i\|w_i\|/√ln 2 is a blind noise level for each channel; no σ² is given to the network. | none extra |
| `model.threshold=firm` | LADMM | firm (MCP) shrinkage: 0 below t1 = c_t·σ̂, a ramp, then identity above t2, so the strong beams are not biased | +1 per layer |
| `model.threshold=gm` | LADMM | MMSE shrinkage under a 4-component complex Gaussian mixture (exact density, zero means), noise variance (c_t·σ̂)² | +8 per layer (`model.gm_K`) |
| `model.data_step=whitened` | LADMM | x_t = H_t(A⁺y + ρz − u), the exact ADMM step for y = A(x+n). Every AᵀA of the operators becomes P = Aᵀ(AAᵀ)⁻¹A. | none extra |
| `model.operator=analytic` / `analytic1` | LADMM | H_t = (G + ρ_t I)⁻¹ through the SVD of A, so nothing is learned in the operator (LADMM-A, any combiner). α_t is in (0, φ); `analytic1` sets α = 1. | 3 / 2 per layer |
| `--methods cadmm` | run_suite, train | complex ADMM-Net with untied M1 (N×M), M2 (N×N) and over-relaxation (Johnston, Li, Lops, Wang, IEEE TSP 69:2818–2832, 2021). It is not in `all`. | 589,842 (T = 6) |
| `--polish band\|snr` | evaluate, compare, workbench Compare | every network also gets a least-squares correction on its K strongest beams, reported as column `<run>+polish`. K and ε are chosen on validation channels. | none (evaluation only) |
| `--ensemble ladmm6 ladmm8` | run_suite, workbench suite field | averages the final networks of these methods into `ENS-…` (`make_ensemble.py`) | the sum of the members |

Also:
- `reference/user/untitled2.py` (your script, from `bsce_code (2)`). The bit-exact test `test_ladmm_best_vs_untitled2.py` now runs when `data/SV_UPA` exists.
- `compare.py` describes the `+polish` columns and the ensembles in `networks.md`.
- The workbench lists the new LADMM options in the method row and has a CADMM row (not ticked).
- The compare preset "one network" leaves out `ENS-*`.

## Notes
- The three new thresholds and the whitened step run in Python only. `LADMM_test.m` computes Aᵀy and the soft threshold, so for these networks no `_orig.mat` is written. The `.pt` and `.mat` files are as usual.
- With `adaptive` / `firm` / `gm`, θ_t is a softplus scale for every operator, so it cannot become negative. The faithful operators keep a raw ρ_t and α_t.
- Polish picks K ∈ {0, 4, 8, 12, 16, 24, 32} and ε ∈ {0.001 … 0.3} on the validation channels with their own noise draws, never on the test set. With `band`, the choice is made per network band over 5 SNRs of the band; this uses no more knowledge than the band networks already need. With `snr`, it is made per test SNR, so it assumes the SNR is known and must be labelled so. K = 0 is always a candidate.

## Checks
See the v28 report: CPU training of every variant on SV ULA, and `tests/test_v28_options.py`.

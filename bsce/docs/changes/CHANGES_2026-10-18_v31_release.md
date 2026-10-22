# v31 (18 Oct 2026): stored = trained, the inference cache, the reweighted l1, combiner mixtures, more transfer

Defaults are unchanged; every new option and method is off by default. Networks saved by v30 and earlier load, and give
the same output; the one difference is Graph-LADMM inference with TV, below.

**Graph-LADMM with TV.** v31 computes each layer's matrix once and reuses it, instead of solving in every call.
- The two results agree to 6 × 10⁻⁶ relative in float32 (an NMSE change below 10⁻⁵ dB).
- The per-call solve itself is bit-identical to v30. It is the one used in training, and with `cache_inference=False`.

## 1. Graph-LADMM: every stored parameter is trained
**What was wrong.** v30's `gladmm` stored 56 parameters but trained only 54. k₂ and e of the last layer had a zero
gradient and kept their initial values (review of 18 Oct). The reason:
- the output is z₁ of the last layer;
- the z₂ step and the dual updates computed after z₁ would only feed a layer T + 1.

**v31:**
- `_dead_names()` lists every parameter that cannot reach the output, and none of them is created:
  - e (and k₂ with TV) of the last layer;
  - ε, a1 and a2 of layer 1 of the reweighted z-steps (z₁ = 0 there, so the weights are 1);
  - ρ of layer 1 with a free H (it multiplies z − u = 0);
  - the pooling weights without graph pooling.
- The last layer returns z₁ and skips the tail.
- Files written before v31 load. `export.load_model` and `_load_from_state_dict` drop those entries; they never changed
  the output.
- An interrupted v30 Graph-LADMM training resumes its weights and position. Its optimizer state does not fit, so it
  starts again, with a note.
- **Counts at T = 8:**

  | network | parameters |
  |---|---|
  | `gladmm`, `gladmmany`, `gladmmmix` | 54 |
  | `gladmmbp` | 318 |
  | `gladmmrw` | 44 |
  | `gladmmrwtv` | 51 |
  | `gladmmf` (T = 6) | 393,250 |

- **The ratios** against the authors' two GM-LAMP networks (524,480 parameters): 1/9,713 for one network, 1/4,856 for
  the per-band pair.

## 2. Graph-LADMM inference
- **The cache.** With the TV x-step and no gradient, H_t = (P + ρ_t(I + DᴴD))⁻¹ is computed in complex128 once per
  (combiner, edge phases, ρ_t) and reused (`_cached_H`, `cache_inference=True`).
- **The cost.** v30 rebuilt I + DᴴD densely and factorised 8 LU systems in every call: 249 M real MACs for one channel
  (SV ULA). v31 needs 2.72 M, 2.8 × GM-LAMP's 0.98 M (`bsce/cost.py`).
- **I + DᴴD** is built from the edges, O(E). It equals the dense product exactly. (The training step time is unchanged: the factorisation dominates it.)

## 3. New Graph-LADMM options and method keys (not in `all`; all train end to end with their own settings)

| key | config | what | parameters (T = 8) |
|---|---|---|---|
| `gladmmrw` | `gladmm_rwl1.yaml` | `zstep: rwl1`, `data_step: plain`: the uploaded `graph_reweighted_l1` package unrolled (soft threshold k1·s·ω, ω from the previous layer's pooled neighbour energy) | 44 |
| `gladmmrwtv` | `gladmm_rwl1_tv.yaml` | `zstep: gtv_rwl1`: the reweighting in gladmm's layer (whitened step + magnetic TV) | 51 |
| `gladmmbp` | `gladmm_bp.yaml` | `beam_prior: gate`: gate logit + β_t ℓ_i, ℓ = the log beam power of the training channels (clipped at log 10⁻³), β_t = 0 at init | 318 |
| `gladmmmix` | `gladmm_mixA.yaml` | `train.combiner: mix` (§5) with M ∈ {64, 96, 128, 160} and all four families | 54 |

**Measured on SV UPA** (4,500 steps; project note `LADMM_graph_reweighted_l1_unrolled_UPA_2026-10-18`):

| network | mean NMSE |
|---|---|
| `gladmm` | −11.70 dB |
| `gladmmrwtv` | −11.22 dB |
| `gladmmrw` | −10.37 dB |
| the classical reweighted l1, tuned | −9.29 dB |
| GM-LAMP (authors) | −15.18 dB |

The beam prior was measured in the four-data-set check of 18 Oct: +0.30 / +0.42 dB on DeepMIMO ULA / UPA, about 0 on SV.

## 4. NF-LADMM, LVAMP and ALISTA follow a new combiner (any M)
- `set_combiner` recomputes their A-dependent buffers:
  - NF-LADMM: A⁺, the eigen-decomposition of AAᵀ, c_white;
  - LVAMP: the SVD buffers;
  - ALISTA: W.
- Nothing they learn depends on A. A moved network equals a fresh one built on the new combiner, and a saved moved
  network reloads.
- `evaluate.py` and `check_combiners.py` use it automatically. v30's statement "only the analytic Graph-LADMM networks
  transfer" no longer holds (review of 18 Oct): NF-LADMM-SR transfers to random combiners and to other M. Graph-LADMM
  is the accurate one that also holds on the structured and badly conditioned combiners.
- `UnrolledNet._combiner_array` checks a new A (same N, independent rows). `_put_buffer` keeps a non-persistent buffer
  non-persistent.
- **ALISTA's `alista_W`** changes in two ways, both exact no-ops on every scenario combiner (unit columns,
  ‖A‖₂² < 6):
  - the default step is min(0.1, 0.6/‖A‖₂²);
  - the projection divides by the column norm when it is not 1.

  Before, W diverged (NaN) on the correlated test combiner.

## 5. Training over a mixture of combiners: `train.combiner: mix`
- Every `combiner_every` minibatches, a new combiner: the size from `combiner_M`, the family from `combiner_types`:
  - `pm1`: random ±1;
  - `gauss`;
  - `sparse`: 16 beams per RF chain;
  - `dct`: random rows of the orthonormal DCT-II.
- They are drawn with the training generator, so a resumed run draws the same ones (`make_train_combiner`).
- It is for networks with `set_combiner`. It needs online noise.
- `combiner_M` / `combiner_types` without `mix`, or an M above N, are refused.

## 6. Classical baselines: l1 ADMM and graph-reweighted l1, tuned per SNR
- **`bsce/classical.py`:**
  - `l1_admm`: 300 ADMM iterations, as the package;
  - `grwl1`: round 0 plus 2 reweighted rounds;
  - `graph_weights`;
  - `METHODS`: the classical names used by every script.
- **`scripts/tune_classical.py`** chooses λ (and ε) per test SNR on validation channels, with the true SNR, and writes
  `results/classical_params_<SCENARIO>.json`.
- **`evaluate.py --classical amp,l1,grwl1 --classical-params <json>`**. compare.py passes the file on: `--classical-params`,
  default `results/classical_params_<SCENARIO>.json`.

## 7. Cost columns in compare.py
`bsce/cost.py` gives two numbers, and every table of compare.py lists both (`summary.csv`: `trained_parameters_per_network`,
`macs_per_channel`; `--no-cost` leaves them out):
- **`trained_parameters`:** the parameters with a non-zero gradient on two random probe batches;
- **`macs_per_channel`:** an ATen dispatch counter of products and factorisations (complex × complex = 4), at batch 1,
  second call.

**What the trained count shows for the other methods.** Their stored form is kept, for compatibility:
- LISTA's W_1 never reaches the output: 1,015,824 of 1,081,360 are trained;
- CADMM's first M2 never reaches it;
- one θ entry per LAMP layer never reaches it: the authors' LAMP trains 262,152 of 262,160;
- nor does the last dual step of LADMM, ADMM-Net and NF-LADMM.

## 8. Robustness of the evaluation
- **`evaluate.py`:** a non-finite NMSE (a diverged network) is drawn and printed as "diverged"; it used to crash the
  label placement. With no method to show, no figure is made.
- **`check_combiners.py`:** a combiner on which `evaluate.py` fails, or where no method can run, is noted, and the check
  goes on.

## 9. Workbench, Spyder, docs
- **Workbench:**
  - four method rows;
  - the `gladmm` text (54 parameters, all options);
  - compare: l1 / grwl1 baselines and a "leave out the cost columns" box;
  - a new task "Tune the classical l1 baselines";
  - the quick tests include the two v31 test files.
- **Spyder:** a new cell 5a (tune, then compare with the classical baselines); comments in cell 4.1; NF-LADMM in cell 5d.
- **Docs:**
  - GUIDE 4.12 (counts, cost), 4.13 (the corrected transfer claim), new 4.14 (the reweighted l1) and 4.15 (combiner
    mixtures), §5 (cost columns, classical);
  - MANUAL §4 (counts, trained parameters of every method, ADMM-Net's two units), §10, A1, A2;
  - CODE_MAP, WORKBENCH.md, CLI_REFERENCE.

## 10. Checks
- **`tests/test_v31_rwl1.py` (12):**
  - the reweighting equals the package's `graph_weights`;
  - layer 1 is plain l1;
  - the plain step equals the explicit solve;
  - no stored parameter without a gradient;
  - save / reload and any M.
- **`tests/test_v31_release.py` (62):**
  - stored = trained for 31 Graph-LADMM configurations (every z-step × array, free H, plain step, no pooling, beam
    prior);
  - pre-v31 files load with the same output;
  - sparse G = dense;
  - cache = solve, following ρ, phases and combiner;
  - the beam prior (β = 0 is the plain network);
  - `set_combiner` of NF-LADMM (radial, CNN, white step), LVAMP and ALISTA at M = 64 / 96 / 160 (moved = fresh,
    save/load, refusals);
  - ALISTA's W solver;
  - the combiner families, and a short `mix` training;
  - classical l1 = bsce's ADMM, grwl1 with a huge ε = l1, the weights = Graph-LADMM's;
  - tune_classical.py and evaluate.py's checks;
  - a plot with NaN, and check_combiners with nothing to run;
  - the cost module (LISTA's W_1, LAMP's and Graph-LADMM's MACs);
  - the new suite keys and their folders.
- **Older tests:** two older tests set k₂ / e of every layer, or counted ρ₀ of a free H. They now skip the parameters
  v31 no longer stores, with the same check of the output.
